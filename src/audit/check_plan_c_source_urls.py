"""Safely check Plan C source URLs without downloading artifacts.

Run from the repository root:

    python src/audit/check_plan_c_source_urls.py --manifest data/artifact_manifests/plan_c_artifact_manifest.csv

The checker performs metadata-only requests. It does not save remote content,
download datasets, call LLM APIs, train models, or modify ``original_repo/``.
"""

from __future__ import annotations

import argparse
import csv
import socket
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit import common_manifests


RESULT_COLUMNS = [
    "source_url",
    "artifact_ids",
    "datasets",
    "source_types",
    "check_method",
    "network_status",
    "availability_status",
    "status_code",
    "final_url",
    "content_type",
    "content_length",
    "error_message",
    "checked_at",
    "notes",
]

NETWORK_UNVERIFIED_EXCEPTIONS = (
    TimeoutError,
    socket.timeout,
    urllib.error.URLError,
)

SAFE_GET_FALLBACK_STATUS_CODES = {400, 403, 405, 501}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Plan C source URLs with metadata-only HTTP requests."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to the Plan C artifact manifest CSV.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Network timeout in seconds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tables/plan_c_source_url_check_results.csv"),
        help="Output CSV path for URL check results.",
    )
    return parser.parse_args(argv)


def check_source_urls(
    manifest_path: str | Path,
    *,
    project_root: str | Path | None = None,
    output_path: str | Path = "results/tables/plan_c_source_url_check_results.csv",
    timeout: float = 5.0,
    opener: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Check unique non-empty source URLs from a Plan C artifact manifest."""

    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    manifest = _resolve_path(Path(manifest_path), root)
    output = _resolve_path(Path(output_path), root)
    rows = _read_manifest_rows(manifest)
    grouped = _group_manifest_urls(rows)
    opener = opener or urllib.request.urlopen
    timestamp = datetime.now(timezone.utc).isoformat()

    result_rows = [
        _check_url(url, metadata, timeout=timeout, opener=opener, timestamp=timestamp)
        for url, metadata in grouped.items()
    ]
    result_df = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output, index=False)
    return result_df


def _read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    return common_manifests.read_csv_manifest_rows(manifest_path, missing="raise", strip=True)


def _group_manifest_urls(rows: list[dict[str, str]]) -> dict[str, dict[str, list[str]]]:
    grouped: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        url = row.get("source_url", "").strip()
        if not url:
            continue
        entry = grouped.setdefault(
            url,
            {"artifact_ids": [], "datasets": [], "source_types": []},
        )
        _append_unique(entry["artifact_ids"], row.get("artifact_id", ""))
        _append_unique(entry["datasets"], row.get("dataset", ""))
        _append_unique(entry["source_types"], row.get("source_type", ""))
    return grouped


def _append_unique(values: list[str], value: str) -> None:
    value = value.strip()
    if value and value not in values:
        values.append(value)


def _check_url(
    url: str,
    metadata: dict[str, list[str]],
    *,
    timeout: float,
    opener: Callable[..., Any],
    timestamp: str,
) -> dict[str, Any]:
    head_result = _request_metadata(url, method="HEAD", timeout=timeout, opener=opener)
    if head_result["network_status"] == "checked" and (
        head_result["status_code"] not in SAFE_GET_FALLBACK_STATUS_CODES
    ):
        result = head_result
    elif head_result["network_status"] == "checked" and (
        head_result["status_code"] in SAFE_GET_FALLBACK_STATUS_CODES
    ):
        get_result = _request_metadata(url, method="GET", timeout=timeout, opener=opener)
        result = get_result if get_result["network_status"] == "checked" else head_result
        if result is get_result:
            result["notes"] = "HEAD was blocked or unsupported; used metadata-only GET without reading response body."
    elif head_result["network_status"] == "network_unverified":
        result = head_result
    else:
        result = head_result

    result.update(
        {
            "source_url": url,
            "artifact_ids": ";".join(metadata["artifact_ids"]),
            "datasets": ";".join(metadata["datasets"]),
            "source_types": ";".join(metadata["source_types"]),
            "checked_at": timestamp,
        }
    )
    return {column: result.get(column, "") for column in RESULT_COLUMNS}


def _request_metadata(
    url: str,
    *,
    method: str,
    timeout: float,
    opener: Callable[..., Any],
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        method=method,
        headers={
            "User-Agent": "IRP-BeyondUtility-PlanC-ArtifactChecker/1.0",
            "Accept": "*/*",
        },
    )
    try:
        with opener(request, timeout=timeout) as response:
            return _response_result(method, response)
    except urllib.error.HTTPError as exc:
        return _http_error_result(method, exc)
    except NETWORK_UNVERIFIED_EXCEPTIONS as exc:
        return {
            "check_method": method,
            "network_status": "network_unverified",
            "availability_status": "network_unverified",
            "status_code": "",
            "final_url": "",
            "content_type": "",
            "content_length": "",
            "error_message": _safe_error_message(exc),
            "notes": "Network check could not be completed; availability is not inferred.",
        }
    except Exception as exc:  # pragma: no cover - defensive for unusual URL handlers.
        return {
            "check_method": method,
            "network_status": "checked",
            "availability_status": "unknown",
            "status_code": "",
            "final_url": "",
            "content_type": "",
            "content_length": "",
            "error_message": _safe_error_message(exc),
            "notes": "Unexpected metadata-check error; no remote content was saved.",
        }


def _response_result(method: str, response: Any) -> dict[str, Any]:
    status_code = int(getattr(response, "status", response.getcode()))
    headers = getattr(response, "headers", {})
    content_type = _header_value(headers, "Content-Type")
    content_length = _header_value(headers, "Content-Length")
    return {
        "check_method": method,
        "network_status": "checked",
        "availability_status": _availability_from_status(status_code),
        "status_code": status_code,
        "final_url": response.geturl(),
        "content_type": content_type,
        "content_length": content_length,
        "error_message": "",
        "notes": "Metadata check completed without reading or saving response content.",
    }


def _http_error_result(method: str, exc: urllib.error.HTTPError) -> dict[str, Any]:
    headers = getattr(exc, "headers", {})
    return {
        "check_method": method,
        "network_status": "checked",
        "availability_status": _availability_from_status(int(exc.code)),
        "status_code": int(exc.code),
        "final_url": exc.geturl(),
        "content_type": _header_value(headers, "Content-Type"),
        "content_length": _header_value(headers, "Content-Length"),
        "error_message": _safe_error_message(exc),
        "notes": "HTTP status was returned during metadata check; no remote content was saved.",
    }


def _availability_from_status(status_code: int) -> str:
    if 200 <= status_code < 400:
        return "reachable"
    if status_code in {401, 403}:
        return "access_restricted"
    if status_code == 404 or status_code == 410:
        return "unavailable"
    if 400 <= status_code < 500:
        return "client_error"
    if 500 <= status_code < 600:
        return "server_error"
    return "unknown"


def _header_value(headers: Any, name: str) -> str:
    if hasattr(headers, "get"):
        value = headers.get(name) or headers.get(name.lower()) or ""
    else:
        value = ""
    return str(value)


def _safe_error_message(exc: BaseException) -> str:
    message = str(exc)
    return message.replace("\n", " ").replace("\r", " ")[:500]


def _resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result_df = check_source_urls(
        args.manifest,
        output_path=args.output,
        timeout=args.timeout,
    )
    status_counts = result_df["availability_status"].value_counts().to_dict()
    print(
        "Plan C source URL check complete: "
        f"urls={len(result_df)}, statuses={status_counts}"
    )
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
