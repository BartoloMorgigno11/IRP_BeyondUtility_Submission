from __future__ import annotations

import csv
import socket
import urllib.error
from pathlib import Path
from typing import Any

import pandas as pd

from src.audit import common_manifests
from src.audit.check_plan_c_source_urls import check_source_urls
from src.audit.validate_plan_c_artifacts import REQUIRED_COLUMNS


class FakeResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        url: str = "https://example.test/resource",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self._url = url
        self.headers = headers or {}

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def geturl(self) -> str:
        return self._url

    def read(self, *_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("URL checker must not read response bodies")


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    common_manifests.write_csv_manifest_rows(path, rows, REQUIRED_COLUMNS, fill_missing=True)


def manifest_row(artifact_id: str, source_url: str) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "dataset": "synthetic",
        "task": "ranking",
        "artifact_name": artifact_id,
        "expected_filename": "artifact.txt",
        "expected_relative_path": "artifact.txt",
        "artifact_category": "test",
        "source_url": source_url,
        "source_type": "official_dataset_raw",
        "local_path": "artifact.txt",
        "found_locally": "false",
        "validation_status": "not_validated",
        "reproducibility_status": "requires_manual_download",
        "can_reconstruct_from_raw": "true",
        "last_checked": "2026-06-29",
    }


def read_results(output_path: Path) -> pd.DataFrame:
    return pd.read_csv(output_path)


def test_head_success_is_recorded(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "url_results.csv"
    url = "https://example.test/data"
    write_manifest(manifest, [manifest_row("artifact", url)])

    def opener(request: Any, timeout: float) -> FakeResponse:
        assert request.get_method() == "HEAD"
        return FakeResponse(
            status=200,
            url=url,
            headers={"Content-Type": "text/html", "Content-Length": "123"},
        )

    result = check_source_urls(
        manifest,
        project_root=tmp_path,
        output_path=output,
        opener=opener,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["check_method"] == "HEAD"
    assert row["network_status"] == "checked"
    assert row["availability_status"] == "reachable"
    assert row["status_code"] == 200
    assert row["content_type"] == "text/html"
    assert read_results(output).iloc[0]["availability_status"] == "reachable"


def test_head_failure_falls_back_to_safe_get(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "url_results.csv"
    url = "https://example.test/no-head"
    write_manifest(manifest, [manifest_row("artifact", url)])
    seen_methods: list[str] = []

    def opener(request: Any, timeout: float) -> FakeResponse:
        method = request.get_method()
        seen_methods.append(method)
        if method == "HEAD":
            raise urllib.error.HTTPError(
                url,
                405,
                "Method Not Allowed",
                hdrs={"Content-Type": "text/plain"},
                fp=None,
            )
        return FakeResponse(
            status=200,
            url=url,
            headers={"Content-Type": "application/zip", "Content-Length": "456"},
        )

    result = check_source_urls(
        manifest,
        project_root=tmp_path,
        output_path=output,
        opener=opener,
    )

    row = result.iloc[0]
    assert seen_methods == ["HEAD", "GET"]
    assert row["check_method"] == "GET"
    assert row["availability_status"] == "reachable"
    assert row["content_type"] == "application/zip"
    assert "metadata-only GET" in row["notes"]


def test_network_error_is_network_unverified(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "url_results.csv"
    write_manifest(manifest, [manifest_row("artifact", "https://example.test/offline")])

    def opener(request: Any, timeout: float) -> FakeResponse:
        raise urllib.error.URLError(socket.timeout("timed out"))

    result = check_source_urls(
        manifest,
        project_root=tmp_path,
        output_path=output,
        opener=opener,
    )

    row = result.iloc[0]
    assert row["network_status"] == "network_unverified"
    assert row["availability_status"] == "network_unverified"
    assert "availability is not inferred" in row["notes"]


def test_empty_urls_are_skipped_and_duplicates_grouped(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "url_results.csv"
    url = "https://example.test/shared"
    write_manifest(
        manifest,
        [
            manifest_row("first", url),
            manifest_row("second", url),
            manifest_row("empty", ""),
        ],
    )
    calls = 0

    def opener(request: Any, timeout: float) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse(status=200, url=url)

    result = check_source_urls(
        manifest,
        project_root=tmp_path,
        output_path=output,
        opener=opener,
    )

    assert len(result) == 1
    assert calls == 1
    assert result.iloc[0]["artifact_ids"] == "first;second"


def test_no_remote_content_is_written(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "nested" / "url_results.csv"
    write_manifest(manifest, [manifest_row("artifact", "https://example.test/content")])

    def opener(request: Any, timeout: float) -> FakeResponse:
        return FakeResponse(status=200, url="https://example.test/content")

    check_source_urls(
        manifest,
        project_root=tmp_path,
        output_path=output,
        opener=opener,
    )

    assert output.exists()
    assert sorted(path.name for path in tmp_path.rglob("*") if path.is_file()) == [
        "manifest.csv",
        "url_results.csv",
    ]


def test_status_breakdown_can_be_summarized(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "url_results.csv"
    write_manifest(
        manifest,
        [
            manifest_row("ok", "https://example.test/ok"),
            manifest_row("gone", "https://example.test/gone"),
        ],
    )

    def opener(request: Any, timeout: float) -> FakeResponse:
        url = request.full_url
        if url.endswith("/gone"):
            raise urllib.error.HTTPError(url, 404, "Not Found", hdrs={}, fp=None)
        return FakeResponse(status=200, url=url)

    result = check_source_urls(
        manifest,
        project_root=tmp_path,
        output_path=output,
        opener=opener,
    )

    counts = result["availability_status"].value_counts().to_dict()
    assert counts == {"reachable": 1, "unavailable": 1}
