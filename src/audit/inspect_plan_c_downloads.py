"""Inspect manually staged Plan C downloads without extracting archives.

Run from the repository root:

    python src/audit/inspect_plan_c_downloads.py --staging-dir data/external/plan_c

The inspector scans local files only. It never downloads artifacts, extracts
archives, calls APIs, trains models, computes metrics, or modifies
``original_repo/``.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit import common_manifests


EXPECTED_CORE_FILES = {
    "sequential_data.txt",
    "negative_samples.txt",
    "datamaps.json",
    "meta.json",
}

KNOWN_LLM_OUTPUT_NAMES = {
    "gpt-3.5-turbo-0125.json",
    "gpt-4o-mini-2024-07-18.json",
    "gpt-4o-2024-08-06.json",
    "claude-3-haiku-20240307.json",
    "llama-3-70b-instruct.json",
    "qwen-2-7b-instruct.json",
    "mistral-7b-instruct-v0.2.json",
}

SOTA_NAME_HINTS = (
    "rec-bm25",
    "rec-bprmf",
    "rec-gru4rec",
    "rec-sasrec",
    "rec-lightgcn",
    "mostpop",
    "bprmf",
    "gru4rec",
    "sasrec",
    "lightgcn",
)

ML1M_NAMES = {"ratings.dat", "movies.dat", "users.dat"}
LASTFM_NAMES = {"user_artists.dat", "artists.dat", "user_taggedartists.dat"}
AMAZON_HINTS = ("reviews_", "meta_", "all_beauty", "sports_and_outdoors")

INVENTORY_COLUMNS = [
    "local_path",
    "filename",
    "extension",
    "size_bytes",
    "is_archive",
    "archive_format",
    "archive_opened",
    "contained_file_count",
    "top_level_folders",
    "contains_sequential_data",
    "contains_negative_samples",
    "contains_datamaps",
    "contains_meta",
    "contains_known_llm_output",
    "contains_known_sota_output",
    "dataset_guess",
    "artifact_category_guess",
    "risk_notes",
    "inspected_at",
]

CONTENTS_COLUMNS = [
    "archive_path",
    "member_path",
    "member_filename",
    "member_size_bytes",
    "is_dir",
    "dataset_guess",
    "artifact_match_type",
]

MATCH_COLUMNS = [
    "staged_local_path",
    "archive_member_path",
    "matched_name",
    "match_type",
    "dataset_guess",
    "artifact_category_guess",
    "confidence",
    "reason",
]

SUMMARY_COLUMNS = [
    "staging_dir",
    "manifest_path",
    "staged_files_found",
    "archives_found",
    "archives_opened",
    "archive_members_listed",
    "possible_artifact_matches",
    "suggested_manifest_updates",
    "status",
    "notes",
    "timestamp",
]

SUGGESTION_COLUMNS = [
    "artifact_id_or_guess",
    "dataset_guess",
    "artifact_name_guess",
    "staged_local_path",
    "suggested_local_path_for_manifest",
    "suggested_artifact_category",
    "suggested_reproducibility_status",
    "confidence",
    "reason",
    "requires_manual_confirmation",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect manually staged Plan C downloads without extracting archives."
    )
    parser.add_argument(
        "--staging-dir",
        required=True,
        type=Path,
        help="Directory containing manually downloaded Plan C artifacts.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional Plan C manifest used only for suggestion context.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory for inspection CSV outputs.",
    )
    return parser.parse_args(argv)


def inspect_downloads(
    staging_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    project_root: str | Path | None = None,
    output_dir: str | Path = "results/tables",
) -> dict[str, pd.DataFrame]:
    """Inspect staged files and write Plan C manual-download CSV diagnostics."""

    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    staging = _resolve_path(Path(staging_dir), root)
    output = _resolve_path(Path(output_dir), root)
    manifest = _resolve_path(Path(manifest_path), root) if manifest_path else None
    manifest_rows = _read_manifest(manifest) if manifest else []
    timestamp = datetime.now(timezone.utc).isoformat()

    files = _list_staged_files(staging)
    inventory_rows: list[dict[str, Any]] = []
    content_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []

    for file_path in files:
        inspection = _inspect_file(file_path, root=root, timestamp=timestamp)
        inventory_rows.append(inspection["inventory"])
        content_rows.extend(inspection["contents"])
        match_rows.extend(inspection["matches"])

    suggestion_rows = _build_suggestions(match_rows, manifest_rows)
    summary_row = _build_summary(
        staging=staging,
        root=root,
        manifest=manifest,
        inventory_rows=inventory_rows,
        content_rows=content_rows,
        match_rows=match_rows,
        suggestion_rows=suggestion_rows,
        timestamp=timestamp,
    )

    output.mkdir(parents=True, exist_ok=True)
    frames = {
        "inventory": pd.DataFrame(inventory_rows, columns=INVENTORY_COLUMNS),
        "contents": pd.DataFrame(content_rows, columns=CONTENTS_COLUMNS),
        "matches": pd.DataFrame(match_rows, columns=MATCH_COLUMNS),
        "summary": pd.DataFrame([summary_row], columns=SUMMARY_COLUMNS),
        "suggestions": pd.DataFrame(suggestion_rows, columns=SUGGESTION_COLUMNS),
    }
    frames["inventory"].to_csv(output / "plan_c_manual_download_inventory.csv", index=False)
    frames["contents"].to_csv(
        output / "plan_c_manual_download_archive_contents.csv",
        index=False,
    )
    frames["matches"].to_csv(
        output / "plan_c_manual_download_artifact_matches.csv",
        index=False,
    )
    frames["summary"].to_csv(output / "plan_c_manual_download_summary.csv", index=False)
    frames["suggestions"].to_csv(
        output / "plan_c_suggested_manifest_updates.csv",
        index=False,
    )
    return frames


def _list_staged_files(staging: Path) -> list[Path]:
    if not staging.exists():
        return []
    return sorted(
        path
        for path in staging.rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    )


def _inspect_file(file_path: Path, *, root: Path, timestamp: str) -> dict[str, Any]:
    archive_format = _archive_format(file_path)
    is_archive = archive_format != ""
    members: list[dict[str, Any]] = []
    archive_opened = False
    risk_notes: list[str] = []

    if archive_format == "zip":
        archive_opened, members, note = _read_zip_members(file_path)
        if note:
            risk_notes.append(note)
    elif archive_format == "tar":
        archive_opened, members, note = _read_tar_members(file_path)
        if note:
            risk_notes.append(note)
    elif is_archive:
        risk_notes.append("Archive extension is recognized but unsupported.")

    paths_for_guess = [file_path.name] + [member["member_path"] for member in members]
    dataset_guess = _guess_dataset(paths_for_guess)
    category_guess = _guess_category(file_path.name, members)
    matches = _find_matches(file_path, members, dataset_guess, category_guess, root)
    flags = _match_flags(matches)
    top_levels = _top_level_folders(members)

    if is_archive and not archive_opened:
        risk_notes.append("Archive could not be opened; no contents were trusted.")
    if not is_archive:
        risk_notes.append("Non-archive file was inventoried but not validated.")
    if category_guess == "raw_dataset_candidate":
        risk_notes.append("Raw public data is reconstruction input, not exact paper artifact recovery.")
    if category_guess == "official_original_candidate":
        risk_notes.append("Core filenames are present, but exact reproduction still requires protocol validation.")

    content_rows = [
        {
            "archive_path": _display_path(file_path, root),
            "member_path": member["member_path"],
            "member_filename": Path(member["member_path"]).name,
            "member_size_bytes": member["member_size_bytes"],
            "is_dir": member["is_dir"],
            "dataset_guess": _guess_dataset([member["member_path"], file_path.name]),
            "artifact_match_type": _member_match_type(member["member_path"]),
        }
        for member in members
    ]

    inventory = {
        "local_path": _display_path(file_path, root),
        "filename": file_path.name,
        "extension": _extension(file_path),
        "size_bytes": file_path.stat().st_size,
        "is_archive": is_archive,
        "archive_format": archive_format,
        "archive_opened": archive_opened,
        "contained_file_count": sum(1 for member in members if not member["is_dir"]),
        "top_level_folders": ";".join(top_levels),
        "contains_sequential_data": flags["sequential_data.txt"],
        "contains_negative_samples": flags["negative_samples.txt"],
        "contains_datamaps": flags["datamaps.json"],
        "contains_meta": flags["meta.json"],
        "contains_known_llm_output": any(match["match_type"] == "llm_output" for match in matches),
        "contains_known_sota_output": any(match["match_type"] == "sota_output" for match in matches),
        "dataset_guess": dataset_guess,
        "artifact_category_guess": category_guess,
        "risk_notes": " ".join(risk_notes),
        "inspected_at": timestamp,
    }
    return {"inventory": inventory, "contents": content_rows, "matches": matches}


def _archive_format(path: Path) -> str:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-1:] == [".zip"]:
        return "zip"
    if suffixes[-1:] == [".tar"] or suffixes[-2:] in [[".tar", ".gz"], [".tar", ".bz2"], [".tar", ".xz"]]:
        return "tar"
    if suffixes[-1:] == [".tgz"]:
        return "tar"
    return ""


def _read_zip_members(path: Path) -> tuple[bool, list[dict[str, Any]], str]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = [
                {
                    "member_path": info.filename,
                    "member_size_bytes": info.file_size,
                    "is_dir": info.is_dir(),
                }
                for info in archive.infolist()
            ]
        return True, members, ""
    except (zipfile.BadZipFile, OSError) as exc:
        return False, [], f"ZIP metadata could not be read: {_safe_error(exc)}"


def _read_tar_members(path: Path) -> tuple[bool, list[dict[str, Any]], str]:
    try:
        with tarfile.open(path, "r:*") as archive:
            members = [
                {
                    "member_path": member.name,
                    "member_size_bytes": member.size,
                    "is_dir": member.isdir(),
                }
                for member in archive.getmembers()
            ]
        return True, members, ""
    except (tarfile.TarError, OSError) as exc:
        return False, [], f"TAR metadata could not be read: {_safe_error(exc)}"


def _find_matches(
    file_path: Path,
    members: list[dict[str, Any]],
    dataset_guess: str,
    category_guess: str,
    root: Path,
) -> list[dict[str, Any]]:
    candidate_paths = [file_path.name] if not members else [member["member_path"] for member in members]
    matches: list[dict[str, Any]] = []
    for candidate in candidate_paths:
        name = Path(candidate).name
        match_type = _member_match_type(candidate)
        if match_type == "":
            continue
        confidence = _match_confidence(match_type, dataset_guess)
        reason = _match_reason(match_type, name)
        matches.append(
            {
                "staged_local_path": _display_path(file_path, root),
                "archive_member_path": "" if candidate == file_path.name and not members else candidate,
                "matched_name": name,
                "match_type": match_type,
                "dataset_guess": _guess_dataset([candidate, file_path.name])
                if dataset_guess == "unknown"
                else dataset_guess,
                "artifact_category_guess": category_guess,
                "confidence": confidence,
                "reason": reason,
            }
        )
    return matches


def _member_match_type(path_value: str) -> str:
    name = Path(path_value).name.lower()
    if name in EXPECTED_CORE_FILES:
        return "core_processed_artifact"
    if name in KNOWN_LLM_OUTPUT_NAMES or "llm" in path_value.lower():
        return "llm_output"
    if any(hint in path_value.lower() for hint in SOTA_NAME_HINTS):
        return "sota_output"
    if name in ML1M_NAMES:
        return "ml1m_raw_file"
    if name in LASTFM_NAMES:
        return "lastfm_raw_file"
    if any(hint in path_value.lower() for hint in AMAZON_HINTS):
        return "amazon_raw_file"
    return ""


def _match_flags(matches: list[dict[str, Any]]) -> dict[str, bool]:
    names = {str(match["matched_name"]).lower() for match in matches}
    return {
        "sequential_data.txt": "sequential_data.txt" in names,
        "negative_samples.txt": "negative_samples.txt" in names,
        "datamaps.json": "datamaps.json" in names,
        "meta.json": "meta.json" in names,
    }


def _guess_dataset(path_values: list[str]) -> str:
    text = " ".join(path_values).lower()
    has_beauty = any(token in text for token in ["beauty", "all_beauty"])
    has_sports = any(token in text for token in ["sports", "sports_and_outdoors"])
    has_ml1m = any(token in text for token in ["ml-1m", "ml1m", "ratings.dat", "movies.dat"])
    has_lastfm = any(token in text for token in ["lastfm", "last.fm", "hetrec", "user_artists.dat"])
    if has_beauty and has_sports:
        return "beauty_sports"
    if has_beauty:
        return "beauty"
    if has_sports:
        return "sports"
    if has_ml1m:
        return "ml1m"
    if has_lastfm:
        return "lastfm"
    return "unknown"


def _guess_category(filename: str, members: list[dict[str, Any]]) -> str:
    values = [filename] + [member["member_path"] for member in members]
    match_types = {_member_match_type(value) for value in values}
    if "core_processed_artifact" in match_types:
        return "official_original_candidate"
    if "llm_output" in match_types:
        return "llm_output_candidate"
    if "sota_output" in match_types:
        return "baseline_output_candidate"
    if match_types & {"ml1m_raw_file", "lastfm_raw_file", "amazon_raw_file"}:
        return "raw_dataset_candidate"
    return "unknown"


def _match_confidence(match_type: str, dataset_guess: str) -> str:
    if match_type == "core_processed_artifact" and dataset_guess in {"beauty", "sports", "beauty_sports"}:
        return "medium"
    if match_type in {"ml1m_raw_file", "lastfm_raw_file", "amazon_raw_file"}:
        return "medium"
    if match_type in {"llm_output", "sota_output"}:
        return "low"
    return "low"


def _match_reason(match_type: str, name: str) -> str:
    if match_type == "core_processed_artifact":
        return f"Found expected processed artifact filename {name}."
    if match_type == "ml1m_raw_file":
        return f"Found MovieLens-1M raw filename {name}."
    if match_type == "lastfm_raw_file":
        return f"Found LastFM HetRec raw filename {name}."
    if match_type == "amazon_raw_file":
        return f"Found Amazon raw-data filename pattern {name}."
    if match_type == "llm_output":
        return f"Found possible LLM output filename or path {name}."
    if match_type == "sota_output":
        return f"Found possible baseline/SOTA output filename or path {name}."
    return "Matched known artifact pattern."


def _build_suggestions(
    matches: list[dict[str, Any]],
    manifest_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for match in matches:
        dataset = str(match["dataset_guess"])
        matched_name = str(match["matched_name"])
        key = (str(match["staged_local_path"]), str(match["archive_member_path"]), matched_name)
        if key in seen:
            continue
        seen.add(key)

        artifact_id = _match_manifest_artifact_id(dataset, matched_name, manifest_rows)
        category = _suggested_category(str(match["match_type"]))
        suggestions.append(
            {
                "artifact_id_or_guess": artifact_id,
                "dataset_guess": dataset,
                "artifact_name_guess": matched_name,
                "staged_local_path": match["staged_local_path"],
                "suggested_local_path_for_manifest": _suggested_manifest_path(match),
                "suggested_artifact_category": category,
                "suggested_reproducibility_status": _suggested_repro_status(category),
                "confidence": match["confidence"],
                "reason": match["reason"],
                "requires_manual_confirmation": True,
            }
        )
    return suggestions


def _match_manifest_artifact_id(
    dataset: str,
    matched_name: str,
    manifest_rows: list[dict[str, str]],
) -> str:
    name = matched_name.lower()
    for row in manifest_rows:
        if dataset != "unknown" and row.get("dataset", "").lower() not in {dataset, "beauty_sports"}:
            continue
        if row.get("expected_filename", "").lower() == name:
            return row.get("artifact_id", "")
    if dataset != "unknown":
        return f"{dataset}_{Path(matched_name).stem}".replace("-", "_")
    return Path(matched_name).stem.replace("-", "_")


def _suggested_category(match_type: str) -> str:
    if match_type == "core_processed_artifact":
        return "author_released_processed"
    if match_type in {"ml1m_raw_file", "lastfm_raw_file", "amazon_raw_file"}:
        return "official_dataset_raw"
    if match_type == "llm_output":
        return "author_released_output"
    if match_type == "sota_output":
        return "author_released_output"
    return "unknown"


def _suggested_repro_status(category: str) -> str:
    if category == "author_released_processed":
        return "local_artifact_available"
    if category == "official_dataset_raw":
        return "local_artifact_available"
    if category == "author_released_output":
        return "local_artifact_available"
    return "blocked_unclear_protocol"


def _suggested_manifest_path(match: dict[str, Any]) -> str:
    archive_member = str(match.get("archive_member_path") or "")
    staged_path = str(match["staged_local_path"])
    if archive_member:
        return f"{staged_path}::{archive_member}"
    return staged_path


def _build_summary(
    *,
    staging: Path,
    root: Path,
    manifest: Path | None,
    inventory_rows: list[dict[str, Any]],
    content_rows: list[dict[str, Any]],
    match_rows: list[dict[str, Any]],
    suggestion_rows: list[dict[str, Any]],
    timestamp: str,
) -> dict[str, Any]:
    staged_files = len(inventory_rows)
    archives_found = sum(bool(row["is_archive"]) for row in inventory_rows)
    archives_opened = sum(bool(row["archive_opened"]) for row in inventory_rows)
    status = "no_staged_files" if staged_files == 0 else "inspected"
    notes = (
        "No manually downloaded files were found; no artifact matches were created."
        if staged_files == 0
        else "Staged files were inspected without archive extraction or manifest mutation."
    )
    return {
        "staging_dir": _display_path(staging, root),
        "manifest_path": _display_path(manifest, root) if manifest else "",
        "staged_files_found": staged_files,
        "archives_found": archives_found,
        "archives_opened": archives_opened,
        "archive_members_listed": len(content_rows),
        "possible_artifact_matches": len(match_rows),
        "suggested_manifest_updates": len(suggestion_rows),
        "status": status,
        "notes": notes,
        "timestamp": timestamp,
    }


def _top_level_folders(members: list[dict[str, Any]]) -> list[str]:
    folders: list[str] = []
    for member in members:
        normalized = member["member_path"].replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
        if len(parts) > 1 and parts[0] not in folders:
            folders.append(parts[0])
    return sorted(folders)


def _extension(path: Path) -> str:
    suffixes = path.suffixes
    if len(suffixes) >= 2 and suffixes[-2].lower() == ".tar":
        return "".join(suffixes[-2:]).lower()
    return path.suffix.lower()


def _read_manifest(manifest: Path) -> list[dict[str, str]]:
    return common_manifests.read_csv_manifest_rows(manifest, missing="empty", strip=True)


def _resolve_path(path: Path | None, root: Path) -> Path:
    if path is None:
        return root
    return path if path.is_absolute() else root / path


def _display_path(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _safe_error(exc: BaseException) -> str:
    return str(exc).replace("\n", " ").replace("\r", " ")[:300]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frames = inspect_downloads(
        args.staging_dir,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
    )
    summary = frames["summary"].iloc[0].to_dict()
    print(
        "Plan C manual download inspection complete: "
        f"staged_files={summary['staged_files_found']}, "
        f"archives={summary['archives_found']}, "
        f"matches={summary['possible_artifact_matches']}, "
        f"suggestions={summary['suggested_manifest_updates']}"
    )
    print(f"Summary: {args.output_dir / 'plan_c_manual_download_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
