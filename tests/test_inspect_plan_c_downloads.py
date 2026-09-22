from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from src.audit.common_io import read_csv_dataframe as _read_csv_dataframe
from src.audit.inspect_plan_c_downloads import inspect_downloads


def read_csv(path: Path) -> pd.DataFrame:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    return _read_csv_dataframe(path, dtype=None, keep_default_na=True)


def test_empty_staging_directory_produces_diagnostic_summary(tmp_path: Path) -> None:
    staging = tmp_path / "data" / "external" / "plan_c"
    staging.mkdir(parents=True)
    output_dir = tmp_path / "results" / "tables"

    frames = inspect_downloads(staging, project_root=tmp_path, output_dir=output_dir)

    summary = frames["summary"].iloc[0]
    assert summary["status"] == "no_staged_files"
    assert summary["staged_files_found"] == 0
    assert summary["possible_artifact_matches"] == 0
    assert read_csv(output_dir / "plan_c_manual_download_summary.csv").iloc[0]["status"] == "no_staged_files"
    assert (output_dir / "plan_c_manual_download_artifact_matches.csv").exists()


def test_non_archive_file_is_inventoried(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    staging.mkdir()
    file_path = staging / "notes.txt"
    file_path.write_text("manual note", encoding="utf-8")

    frames = inspect_downloads(staging, project_root=tmp_path, output_dir=tmp_path / "out")

    inventory = frames["inventory"].iloc[0]
    assert inventory["filename"] == "notes.txt"
    assert bool(inventory["is_archive"]) is False
    assert bool(inventory["archive_opened"]) is False
    assert inventory["artifact_category_guess"] == "unknown"


def test_zip_archive_contents_are_listed_without_extraction(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    staging.mkdir()
    archive_path = staging / "beauty.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("beauty/sequential_data.txt", "1 2 3")
        archive.writestr("beauty/negative_samples.txt", "1 4")
        archive.writestr("beauty/datamaps.json", "{}")
        archive.writestr("beauty/meta.json", "{}")

    frames = inspect_downloads(staging, project_root=tmp_path, output_dir=tmp_path / "out")

    inventory = frames["inventory"].iloc[0]
    contents = frames["contents"]
    matches = frames["matches"]
    assert bool(inventory["is_archive"]) is True
    assert bool(inventory["archive_opened"]) is True
    assert inventory["contained_file_count"] == 4
    assert inventory["top_level_folders"] == "beauty"
    assert bool(inventory["contains_sequential_data"]) is True
    assert bool(inventory["contains_negative_samples"]) is True
    assert bool(inventory["contains_datamaps"]) is True
    assert bool(inventory["contains_meta"]) is True
    assert set(contents["member_filename"]) == {
        "sequential_data.txt",
        "negative_samples.txt",
        "datamaps.json",
        "meta.json",
    }
    assert len(matches) == 4
    assert not (staging / "beauty").exists()


def test_expected_files_inside_zip_create_manual_suggestions(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    staging.mkdir()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "artifact_id,dataset,expected_filename\n"
        "beauty_sequential_data,beauty,sequential_data.txt\n",
        encoding="utf-8",
    )
    archive_path = staging / "beauty.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("beauty/sequential_data.txt", "1 2 3")

    frames = inspect_downloads(
        staging,
        manifest_path=manifest,
        project_root=tmp_path,
        output_dir=tmp_path / "out",
    )

    suggestions = frames["suggestions"]
    assert len(suggestions) == 1
    row = suggestions.iloc[0]
    assert row["artifact_id_or_guess"] == "beauty_sequential_data"
    assert row["dataset_guess"] == "beauty"
    assert row["suggested_artifact_category"] == "author_released_processed"
    assert row["suggested_reproducibility_status"] == "local_artifact_available"
    assert bool(row["requires_manual_confirmation"]) is True
    assert "::beauty/sequential_data.txt" in row["suggested_local_path_for_manifest"]


def test_dataset_guesses_are_deterministic_for_raw_archives(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    staging.mkdir()
    archive_path = staging / "ml-1m.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("ml-1m/ratings.dat", "1::1::5::1")
        archive.writestr("ml-1m/movies.dat", "1::Title::Genre")
        archive.writestr("ml-1m/users.dat", "1::M::25::1::00000")

    first = inspect_downloads(staging, project_root=tmp_path, output_dir=tmp_path / "out1")
    second = inspect_downloads(staging, project_root=tmp_path, output_dir=tmp_path / "out2")

    assert first["inventory"].iloc[0]["dataset_guess"] == "ml1m"
    assert first["inventory"].iloc[0]["artifact_category_guess"] == "raw_dataset_candidate"
    assert second["inventory"].iloc[0]["dataset_guess"] == "ml1m"
    assert list(first["matches"]["matched_name"]) == list(second["matches"]["matched_name"])


def test_no_archive_extraction_occurs_by_default(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    staging.mkdir()
    archive_path = staging / "sports.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("sports/meta.json", "{}")

    inspect_downloads(staging, project_root=tmp_path, output_dir=tmp_path / "out")

    assert not (staging / "sports").exists()
    assert sorted(path.name for path in staging.iterdir()) == ["sports.zip"]


def test_inspector_does_not_modify_original_repo(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    staging.mkdir()
    original_repo = tmp_path / "original_repo"
    original_repo.mkdir()
    marker = original_repo / "marker.txt"
    marker.write_text("unchanged", encoding="utf-8")

    inspect_downloads(staging, project_root=tmp_path, output_dir=tmp_path / "out")

    assert marker.read_text(encoding="utf-8") == "unchanged"
