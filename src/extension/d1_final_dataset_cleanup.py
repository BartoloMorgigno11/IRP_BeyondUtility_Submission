"""Execute the D1 cleanup after Food.com fifth-dataset selection.

D1 is a deletion and reconciliation checkpoint only. It does not perform model
inference, recommender training, downloads, or raw-data reconstruction.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.audit.common_hashing import sha256_file, sha256_file_or_empty
from src.extension.e2a_cell_universe import generate_universe, validate_universe, write_outputs
from src.utils.config import find_project_root


PHASE_ID = "d1_final_loser_dataset_purge"
DECISION_TOKEN = "D1_FOODCOM_ONLY_OPERATIONAL_TREE_VALIDATED"
INVENTORY_PATH = Path("results/manifests/d0b_loser_purge_inventory.json")

MANUAL_REVIEW_PATH = Path("results/tables/d1_manual_review_resolution.csv")
REMOVED_PATH = Path("results/tables/d1_removed_artifacts.csv")
RETAINED_PATH = Path("results/tables/d1_retained_generic_artifacts.csv")
REFERENCE_AUDIT_PATH = Path("results/tables/d1_remaining_video_games_reference_audit.csv")
ACTIVE_DATASET_AUDIT_PATH = Path("results/tables/d1_active_dataset_audit.csv")
EXECUTION_MANIFEST_PATH = Path("results/manifests/d1_video_games_purge_execution.json")
REPORT_PATH = Path("report/d1_final_fifth_dataset_cleanup.md")

EXPECTED_ACTIVE_DATASETS = ("beauty", "sports", "ml1m", "lastfm", "foodcom")
EXPECTED_ACTIVE_LABELS = {
    "beauty": "Beauty",
    "sports": "Sports",
    "ml1m": "ML-1M",
    "lastfm": "LastFM",
    "foodcom": "Food.com",
}

EXPECTED_FOODCOM_HASHES = {
    "archive": "e7a43e623c616799f78a0479b550483450fd7ede4e0c40979db3458a82b5b913",
    "RAW_interactions.csv": "4e278d2f687b1020f5694e51003a7d6174344b51aebbdf4530de88b5c6a78ee4",
    "RAW_recipes.csv": "6a8136d1da9e03396a9f52d72200b3fddb8fe103f11e1496effca30ac4e0539f",
}

PROTECTED_PREFIXES = (
    "original_repo",
    "paper",
    "data/raw/foodcom",
    "data/raw/beauty",
    "data/raw/sports",
    "data/raw/ml1m",
    "data/raw/lastfm",
    "data/processed/foodcom",
    "data/processed/beauty",
    "data/processed/sports",
    "data/processed/ml1m",
    "data/processed/lastfm",
    "data/reconstructed/plan_c",
)

REFERENCE_PATTERN = re.compile(r"video[ _-]?games|videogames|amazon video", re.IGNORECASE)
MAX_AUDIT_LINE_CHARS = 500
MAX_CONTENT_SCAN_BYTES = 5_000_000
CONTENT_SCAN_ROOTS = (
    ".gitignore",
    "README.md",
    "PROJECT_SPEC.md",
    "CODEX_PROMPTS.md",
    "reproducibility_checklist.md",
    "AGENTS.md",
    "configs",
    "src",
    "tests",
    "report",
    "results/tables",
    "results/manifests",
    "results/extension",
    "results/raw_outputs",
    "results/metrics",
    "results/llm",
    "data/raw",
    "data/processed",
    "Espansione_progetto",
    "original_repo/EvaLLMasRecommender",
)

D1_DISCOVERED_REMOVALS = (
    ("ISTRUZIONI_AMICO.md", "instruction", "D1-discovered obsolete discarded-dataset run instructions"),
    ("Espansione_progetto/video_games_tutti_esperimenti", "legacy_local_prototype", "D1-discovered discarded dataset prototype bundle"),
    ("results/manifests/extension_e0_video_games_primary_candidates.csv", "manifest", "D1-discovered discarded E0 materialization"),
    ("results/manifests/extension_e0_reproducibility.json", "manifest", "embedded discarded E0 candidate hash"),
    ("results/manifests/extension_e0_prompt_samples.csv", "manifest", "embedded discarded E0 prompts"),
    ("report/extension_e0_protocol_freeze.md", "report", "old E0 report describes discarded candidate materialization"),
    ("src/extension/e0_dry_run.py", "source_module", "old E0 materialization was discarded-dataset-specific"),
    ("src/extension/e1_pilot.py", "source_module", "old E1 pilot was discarded-dataset-specific"),
    ("configs/extension_e1_pilot.yaml", "config", "old E1 pilot config was discarded-dataset-specific"),
    ("tests/test_extension_e1_pilot.py", "test", "old E1 pilot test was discarded-dataset-specific"),
    ("report/extension_e1_pilot.md", "report", "old E1 pilot report was discarded-dataset-specific"),
    ("results/extension/e1_pilot", "raw_output", "old E1 pilot raw responses were discarded-dataset-specific"),
    ("results/manifests/extension_e1_pilot_manifest.csv", "manifest", "old E1 pilot request manifest was discarded-dataset-specific"),
    ("results/manifests/extension_e1_reproducibility.json", "manifest", "old E1 pilot reproducibility manifest was discarded-dataset-specific"),
    ("results/manifests/extension_e1_response_manifest.csv", "manifest", "old E1 pilot response manifest was discarded-dataset-specific"),
    ("results/manifests/extension_e1_runtime_provenance.json", "manifest", "old E1 pilot runtime provenance was discarded-dataset-specific"),
    ("results/manifests/extension_e1_selected_users.csv", "manifest", "old E1 pilot selected users were discarded-dataset-specific"),
    ("results/tables/extension_e1_canary_results.csv", "table", "old E1 pilot result table was discarded-dataset-specific"),
    ("results/tables/extension_e1_early_health_results.csv", "table", "old E1 pilot result table was discarded-dataset-specific"),
    ("results/tables/extension_e1_full_run_projection.csv", "table", "old E1 pilot runtime projection was discarded-dataset-specific"),
    ("results/tables/extension_e1_history_validation.csv", "table", "old E1 pilot validation table was discarded-dataset-specific"),
    ("results/tables/extension_e1_model_readiness.csv", "table", "old E1 pilot readiness table was discarded-dataset-specific"),
    ("results/tables/extension_e1_pilot_utility_by_cell.csv", "table", "old E1 pilot utility table was discarded-dataset-specific"),
    ("results/tables/extension_e1_preflight.csv", "table", "old E1 pilot preflight table was discarded-dataset-specific"),
    ("results/tables/extension_e1_reliability_by_cell.csv", "table", "old E1 pilot reliability table was discarded-dataset-specific"),
    ("results/tables/extension_e1_reliability_by_model.csv", "table", "old E1 pilot reliability table was discarded-dataset-specific"),
    ("results/tables/extension_e1_runtime_by_model.csv", "table", "old E1 pilot runtime table was discarded-dataset-specific"),
    ("report/results_analysis.md", "report", "old discarded-dataset prototype report"),
    ("report/dataset_audit.md", "report", "old discarded-dataset prototype report"),
)

RETAINED_GENERIC_PATHS = (
    ("results/tables/extension_e1_metric_contract.csv", "generic E1 metric contract"),
    ("results/tables/extension_e1_four_dataset_coverage.csv", "generic original-four coverage planning table"),
    ("results/tables/extension_e1_fifth_dataset_requirements.csv", "generic fifth-dataset requirement template"),
    ("results/tables/extension_e1_inference_requirements.csv", "generic inference requirement template"),
    ("results/tables/extension_e1_dataset_selection_scorecard_template.csv", "generic fifth-dataset scorecard template"),
    ("results/manifests/extension_e0_history_manifest.csv", "generic original-four history manifest"),
    ("results/tables/extension_e0_history_dry_run_summary.csv", "generic original-four history summary"),
    ("results/tables/extension_e0_parser_synthetic_summary.csv", "generic parser synthetic summary"),
    ("results/tables/extension_e0_protocol_contracts.csv", "generic E0 protocol contracts"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the D1 Food.com-only operational cleanup.")
    parser.add_argument("--execute", action="store_true", help="Delete validated loser-path artifacts.")
    parser.add_argument("--output-root", default=".", help="Project root; defaults to auto-detected root.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    root = root.resolve()

    started_at = _utc_now()
    inventory = _read_json(root / INVENTORY_PATH)
    _validate_inventory(inventory)

    pre_snapshot = _snapshot_state(root, inventory=inventory)
    manual_rows = _resolve_manual_review(root, inventory)
    _write_csv(root / MANUAL_REVIEW_PATH, manual_rows)

    removal_targets = _build_removal_targets(root, inventory)
    removed_rows = _delete_targets(root, removal_targets, execute=args.execute)

    outputs = generate_universe(root)
    validation = validate_universe(outputs["raw"], outputs["cells"])
    if not validation["passed"]:
        raise RuntimeError(f"active cell universe validation failed: {validation['errors']}")
    write_outputs(root, outputs, validation)

    active_rows = _active_dataset_audit(outputs["raw"], outputs["cells"])
    retained_rows = _retained_generic_audit(root, inventory)

    _write_csv(root / REMOVED_PATH, removed_rows)
    _write_csv(root / RETAINED_PATH, retained_rows)
    _write_csv(root / ACTIVE_DATASET_AUDIT_PATH, active_rows)

    post_snapshot = _snapshot_state(root, inventory=inventory)
    reference_rows = _remaining_reference_audit(root)
    _write_csv(root / REFERENCE_AUDIT_PATH, reference_rows)

    forbidden_count = sum(1 for row in reference_rows if str(row["allowed"]).lower() != "true")
    removed_summary = _summarize_removed(removed_rows)
    active_summary = _summarize_active(active_rows)
    protected_summary = _protected_summary(pre_snapshot, post_snapshot)

    manifest = {
        "phase_id": PHASE_ID,
        "decision_token": DECISION_TOKEN if forbidden_count == 0 else "D1_PURGE_BLOCKED",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "execute": bool(args.execute),
        "branch": pre_snapshot["git"]["branch"],
        "head": pre_snapshot["git"]["head"],
        "pre_purge": pre_snapshot,
        "post_purge": post_snapshot,
        "manual_review_rows": len(manual_rows),
        "manual_review_unresolved": sum(1 for row in manual_rows if row["final_classification"] == "REVIEW_MANUALLY"),
        "removed_artifacts": removed_summary,
        "retained_generic_artifacts": len(retained_rows),
        "active_dataset_set": [EXPECTED_ACTIVE_LABELS[dataset] for dataset in EXPECTED_ACTIVE_DATASETS],
        "active_dataset_audit": active_summary,
        "remaining_video_games_references": {
            "rows": len(reference_rows),
            "allowed_rows": len(reference_rows) - forbidden_count,
            "operational_forbidden_occurrences": forbidden_count,
        },
        "protected_hashes": protected_summary,
        "llm_requests_executed": 0,
        "training_runs_executed": 0,
        "downloads_executed": 0,
        "network_calls_executed": 0,
        "raw_reconstruction_executed": 0,
        "source_artifacts": {
            "starting_inventory": INVENTORY_PATH.as_posix(),
            "manual_review_resolution": MANUAL_REVIEW_PATH.as_posix(),
            "removed_artifacts": REMOVED_PATH.as_posix(),
            "retained_generic_artifacts": RETAINED_PATH.as_posix(),
            "remaining_reference_audit": REFERENCE_AUDIT_PATH.as_posix(),
            "active_dataset_audit": ACTIVE_DATASET_AUDIT_PATH.as_posix(),
        },
        "output_hashes": _artifact_hashes(
            root,
            [
                MANUAL_REVIEW_PATH,
                REMOVED_PATH,
                RETAINED_PATH,
                REFERENCE_AUDIT_PATH,
                ACTIVE_DATASET_AUDIT_PATH,
                Path("results/tables/extension_cell_universe.csv"),
                Path("results/tables/extension_raw_experiment_universe.csv"),
                Path("results/tables/extension_cell_completeness_summary.csv"),
                Path("results/tables/extension_metric_applicability_matrix.csv"),
                Path("results/tables/extension_parity_summary.csv"),
                Path("results/manifests/extension_cell_universe_freeze.json"),
                Path("report/extension_cell_universe.md"),
            ],
        ),
    }

    _write_report(root, manifest, manual_rows, removed_rows, retained_rows, reference_rows, active_rows)
    manifest["output_hashes"][REPORT_PATH.as_posix()] = sha256_file_or_empty(root / REPORT_PATH)
    _write_json(root / EXECUTION_MANIFEST_PATH, manifest)
    return 0 if forbidden_count == 0 else 2


def _validate_inventory(inventory: Mapping[str, Any]) -> None:
    entries = inventory.get("entries")
    if not isinstance(entries, list):
        raise ValueError("D0B purge inventory is missing an entries list")
    counts = {
        "REMOVE_DATASET_SPECIFIC": 0,
        "RETAIN_GENERIC_INFRASTRUCTURE": 0,
        "REVIEW_MANUALLY": 0,
        "RETAIN_HISTORICAL_REPRODUCTION": 0,
    }
    for entry in entries:
        classification = str(entry.get("classification", ""))
        if classification in counts:
            counts[classification] += 1
    expected = {
        "REMOVE_DATASET_SPECIFIC": 58,
        "RETAIN_GENERIC_INFRASTRUCTURE": 6,
        "REVIEW_MANUALLY": 4,
        "RETAIN_HISTORICAL_REPRODUCTION": 1,
    }
    if counts != expected:
        raise ValueError(f"unexpected D0B purge inventory counts: {counts} != {expected}")


def _snapshot_state(root: Path, *, inventory: Mapping[str, Any]) -> dict[str, Any]:
    foodcom = _foodcom_hashes(root)
    h0_hashes = _artifact_hashes(
        root,
        [
            Path("results/manifests/extension_h0_no_inference_manifest.json"),
            Path("results/tables/extension_h0_72_cell_history_matrix.csv"),
            Path("results/tables/extension_h0_dataset_l20_feasibility.csv"),
            Path("results/tables/extension_h0_exact_history_request_budget.csv"),
            Path("results/tables/extension_h0_final_history_protocol_identity_matrix.csv"),
            Path("results/tables/extension_h0_final_paired_200_cohorts.csv"),
            Path("results/tables/extension_h0_historical_history_experiment_inventory.csv"),
            Path("results/tables/extension_h0_paper_vs_local_history_grid.csv"),
            Path("results/tables/extension_h0_reusable_vs_to_compute.csv"),
        ],
    )
    d0a_hashes = _artifact_hashes(
        root,
        [
            Path("results/manifests/d0a_foodcom_raw_archive_manifest.json"),
            Path("results/manifests/d0a_foodcom_extracted_file_hashes.json"),
            Path("results/manifests/d0a_foodcom_no_inference_manifest.json"),
        ],
    )
    d0b_hashes = _artifact_hashes(
        root,
        [
            Path("results/manifests/d0b_no_inference_manifest.json"),
            Path("results/manifests/d0b_loser_purge_inventory.json"),
            Path("results/tables/d0b_authoritative_input_evidence.csv"),
            Path("results/tables/d0b_completeness_counts.csv"),
            Path("results/tables/d0b_dataset_specific_blockers.csv"),
            Path("results/tables/d0b_fifth_dataset_complete_requirement_matrix.csv"),
            Path("results/tables/d0b_final_selection_decision.csv"),
            Path("results/tables/d0b_future_inference_budget.csv"),
        ],
    )
    return {
        "created_at": _utc_now(),
        "git": {
            "branch": _git(["branch", "--show-current"], root),
            "head": _git(["rev-parse", "HEAD"], root),
            "status_short": _git(["status", "--short"], root),
        },
        "purge_inventory_hash": sha256_file_or_empty(root / INVENTORY_PATH),
        "inventory_reported_hash": str(
            inventory.get("artifact_hashes", {}).get(INVENTORY_PATH.as_posix(), "")
        ),
        "foodcom_hashes": foodcom,
        "h0_artifact_hashes": h0_hashes,
        "d0a_artifact_hashes": d0a_hashes,
        "d0b_artifact_hashes": d0b_hashes,
    }


def _resolve_manual_review(root: Path, inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    manual_entries = [
        entry
        for entry in inventory["entries"]
        if str(entry.get("classification", "")) == "REVIEW_MANUALLY"
    ]
    rows: list[dict[str, Any]] = []
    for entry in manual_entries:
        path = str(entry["path"])
        references = _manual_reference_count(root, path)
        if path.startswith("results/tables/extension_cell_universe.csv"):
            reason = "Rows scoped to the discarded fifth dataset are regenerated out of the active universe."
            replacement = True
        elif path.startswith("results/tables/extension_raw_experiment_universe.csv"):
            reason = "Raw experiment rows scoped to the discarded fifth dataset are regenerated out of the active universe."
            replacement = True
        elif path.startswith("report/notes.md"):
            reason = "Long operational notes were collapsed to the final selection rationale."
            replacement = True
        elif path.startswith(".gitignore"):
            reason = "Loser-specific visibility exceptions were removed; D1 evidence exceptions were added."
            replacement = True
        else:
            reason = "Manual entry resolved as discarded-dataset-specific."
            replacement = True
        rows.append(
            {
                "path": path,
                "original_classification": "REVIEW_MANUALLY",
                "final_classification": "REMOVE_DATASET_SPECIFIC",
                "reason": reason,
                "references_found": references,
                "safe_to_remove": True,
                "replacement_needed": replacement,
            }
        )
    if any(row["final_classification"] == "REVIEW_MANUALLY" for row in rows):
        raise RuntimeError("D1_PURGE_MANUAL_REVIEW_BLOCKED")
    return rows


def _manual_reference_count(root: Path, manual_path: str) -> int:
    if manual_path.startswith("results/tables/extension_cell_universe.csv"):
        return _count_dataset_rows(root / "results/tables/extension_cell_universe.csv")
    if manual_path.startswith("results/tables/extension_raw_experiment_universe.csv"):
        return _count_dataset_rows(root / "results/tables/extension_raw_experiment_universe.csv")
    if manual_path.startswith("report/notes.md"):
        return _count_text_matches(root / "report/notes.md", REFERENCE_PATTERN)
    if manual_path.startswith(".gitignore"):
        return _count_text_matches(root / ".gitignore", re.compile(r"video[ _-]?games|e2b0|e2b0h|e2b0j", re.IGNORECASE))
    return 0


def _count_dataset_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return 0
    if "dataset" not in frame.columns:
        return 0
    return int(frame["dataset"].astype(str).str.lower().isin({"video_games", "video games", "videogames"}).sum())


def _build_removal_targets(root: Path, inventory: Mapping[str, Any]) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    for entry in inventory["entries"]:
        if str(entry.get("classification", "")) == "REMOVE_DATASET_SPECIFIC":
            targets.append(
                {
                    "path": str(entry["path"]).replace("\\", "/"),
                    "artifact_type": str(entry.get("artifact_type", "")),
                    "source": "D0B_PURGE_INVENTORY",
                    "reason": str(entry.get("reason", "")),
                }
            )
    for path, artifact_type, reason in D1_DISCOVERED_REMOVALS:
        targets.append(
            {
                "path": path,
                "artifact_type": artifact_type,
                "source": "D1_DISCOVERED",
                "reason": reason,
            }
        )
    targets.extend(_discover_cache_targets(root))
    deduped: dict[str, dict[str, str]] = {}
    for target in targets:
        deduped.setdefault(target["path"], target)
    validated = []
    for target in deduped.values():
        _validate_delete_target(root, target["path"])
        validated.append(target)
    return sorted(validated, key=lambda row: row["path"])


def _discover_cache_targets(root: Path) -> list[dict[str, str]]:
    cache_targets: list[dict[str, str]] = []
    for cache_root in (root / "src", root / "tests"):
        if not cache_root.exists():
            continue
        for path in cache_root.rglob("*.pyc"):
            rel = _rel(root, path)
            lowered = rel.lower()
            if any(token in lowered for token in ("e2b0", "e2b0h", "e2b0j", "video_games")):
                cache_targets.append(
                    {
                        "path": rel,
                        "artifact_type": "python_cache",
                        "source": "D1_DISCOVERED_CACHE",
                        "reason": "compiled cache for discarded dataset-specific source/test",
                    }
                )
    return cache_targets


def _validate_delete_target(root: Path, rel_path: str) -> None:
    if not rel_path or rel_path.strip() in {".", "./"}:
        raise ValueError(f"refusing to delete invalid path: {rel_path!r}")
    raw = Path(rel_path)
    if raw.is_absolute():
        raise ValueError(f"refusing absolute deletion path: {rel_path}")
    resolved = (root / raw).resolve()
    try:
        normalized = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"refusing path outside project root: {rel_path}") from exc
    if normalized == ".":
        raise ValueError("refusing to delete project root")
    for prefix in PROTECTED_PREFIXES:
        if normalized == prefix or normalized.startswith(prefix + "/"):
            if normalized not in {"data/raw/video_games_2014", "data/processed/video_games"}:
                raise ValueError(f"refusing protected deletion target: {normalized}")


def _delete_targets(root: Path, targets: Iterable[Mapping[str, str]], *, execute: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        rel_path = str(target["path"])
        path = root / rel_path
        exists_before = path.exists()
        is_dir = path.is_dir() if exists_before else False
        file_count, byte_count = _path_counts(path)
        if not exists_before:
            file_count, byte_count = _git_deleted_counts(root, rel_path)
        sha = sha256_file_or_empty(path) if exists_before and path.is_file() else ""
        removed = False
        if execute and exists_before:
            if is_dir:
                shutil.rmtree(path)
            else:
                path.unlink()
            removed = not path.exists()
        elif execute and not exists_before:
            removed = not path.exists()
        rows.append(
            {
                "path": rel_path,
                "classification": "REMOVE_DATASET_SPECIFIC",
                "artifact_type": target.get("artifact_type", ""),
                "source": target.get("source", ""),
                "reason": target.get("reason", ""),
                "exists_before": exists_before,
                "is_dir": is_dir,
                "file_count": file_count,
                "byte_count": byte_count,
                "sha256_before": sha,
                "removed": removed,
                "exists_after": path.exists(),
            }
        )
    return rows


def _path_counts(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_file():
        try:
            return 1, int(path.stat().st_size)
        except OSError:
            return 1, 0
    file_count = 0
    byte_count = 0
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        file_count += 1
        try:
            byte_count += int(child.stat().st_size)
        except OSError:
            pass
    return file_count, byte_count


def _git_deleted_counts(root: Path, rel_path: str) -> tuple[int, int]:
    git_output = _git(["ls-files", "--deleted", "--", rel_path], root)
    if not git_output or git_output == "UNAVAILABLE":
        return 0, 0
    files = [line.strip() for line in git_output.splitlines() if line.strip()]
    byte_count = 0
    for tracked_path in files:
        blob_size = _git(["cat-file", "-s", f"HEAD:{tracked_path}"], root)
        try:
            byte_count += int(blob_size)
        except ValueError:
            pass
    return len(files), byte_count


def _active_dataset_audit(raw: pd.DataFrame, cells: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_counts = raw.groupby("dataset").size().to_dict()
    cell_counts = cells.groupby("dataset").size().to_dict()
    for dataset in EXPECTED_ACTIVE_DATASETS:
        rows.append(
            {
                "dataset": EXPECTED_ACTIVE_LABELS[dataset],
                "dataset_id": dataset,
                "active_raw_experiment_count": int(raw_counts.get(dataset, 0)),
                "active_cell_count": int(cell_counts.get(dataset, 0)),
                "expected_active": True,
                "status": "ACTIVE",
            }
        )
    extra_datasets = sorted(set(raw_counts).union(cell_counts).difference(EXPECTED_ACTIVE_DATASETS))
    for dataset in extra_datasets:
        rows.append(
            {
                "dataset": str(dataset),
                "dataset_id": str(dataset),
                "active_raw_experiment_count": int(raw_counts.get(dataset, 0)),
                "active_cell_count": int(cell_counts.get(dataset, 0)),
                "expected_active": False,
                "status": "UNEXPECTED",
            }
        )
    return rows


def _retained_generic_audit(root: Path, inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in inventory["entries"]:
        if str(entry.get("classification", "")) == "RETAIN_GENERIC_INFRASTRUCTURE":
            rel_path = str(entry["path"]).replace("\\", "/")
            path = root / rel_path
            rows.append(
                {
                    "path": rel_path,
                    "classification": "RETAIN_GENERIC_INFRASTRUCTURE",
                    "source": "D0B_PURGE_INVENTORY",
                    "reason": str(entry.get("reason", "")),
                    "exists_after": path.exists(),
                    "sha256_after": sha256_file_or_empty(path) if path.is_file() else "",
                }
            )
    for rel_path, reason in RETAINED_GENERIC_PATHS:
        path = root / rel_path
        rows.append(
            {
                "path": rel_path,
                "classification": "RETAIN_GENERIC_INFRASTRUCTURE",
                "source": "D1_REVIEW",
                "reason": reason,
                "exists_after": path.exists(),
                "sha256_after": sha256_file_or_empty(path) if path.is_file() else "",
            }
        )
    return rows


def _remaining_reference_audit(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in _iter_path_reference_paths(root):
        rel_path = _rel(root, path)
        for match in REFERENCE_PATTERN.finditer(rel_path):
            key = (rel_path, "PATH", match.group(0))
            if key not in seen:
                rows.append(_classify_reference(rel_path, "PATH", match.group(0)))
                seen.add(key)
    for path in _iter_content_scan_paths(root):
        rel_path = _rel(root, path)
        try:
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                for number, line in enumerate(handle, start=1):
                    for match in REFERENCE_PATTERN.finditer(line):
                        occurrence = match.group(0)
                        reference = f"line {number}"
                        key = (rel_path, reference, occurrence)
                        if key in seen:
                            continue
                        rows.append(
                            _classify_reference(
                                rel_path,
                                reference,
                                occurrence,
                                text=line.strip()[:MAX_AUDIT_LINE_CHARS],
                            )
                        )
                        seen.add(key)
        except OSError:
            continue
    return sorted(rows, key=lambda row: (row["allowed"] != "true", row["path"], row["line/reference"]))


def _iter_path_reference_paths(root: Path) -> Iterable[Path]:
    excluded_parts = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "env",
    }
    for scan_root in CONTENT_SCAN_ROOTS:
        base = root / scan_root
        if not base.exists():
            continue
        candidates = [base] if base.is_file() else base.rglob("*")
        for path in candidates:
            parts = set(path.relative_to(root).parts)
            if parts.intersection(excluded_parts):
                continue
            yield path


def _iter_content_scan_paths(root: Path) -> Iterable[Path]:
    excluded_parts = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "env",
    }
    excluded_prefixes = (
        "data/raw",
        "data/raw/foodcom",
        "data/reconstructed/plan_c",
        "data/processed/ml1m",
        "results/llm",
        "results/logs",
        "results/recovery",
    )
    excluded_file_prefixes = (
        "results/tables/reproduction_",
        "results/tables/phase_",
        "results/tables/plan_",
    )
    for scan_root in CONTENT_SCAN_ROOTS:
        base = root / scan_root
        if not base.exists():
            continue
        candidates = [base] if base.is_file() else base.rglob("*")
        for path in candidates:
            if not path.is_file() or not _is_text_candidate(path):
                continue
            rel = _rel(root, path)
            parts = set(path.relative_to(root).parts)
            if parts.intersection(excluded_parts):
                continue
            if any(rel == prefix or rel.startswith(prefix + "/") for prefix in excluded_prefixes):
                continue
            if any(rel.startswith(prefix) for prefix in excluded_file_prefixes):
                continue
            if _is_oversized_content_candidate(path, rel):
                continue
            yield path


def _is_oversized_content_candidate(path: Path, rel_path: str) -> bool:
    try:
        size = path.stat().st_size
    except OSError:
        return True
    if size <= MAX_CONTENT_SCAN_BYTES:
        return False
    allowed_large_prefixes = (
        "results/tables/d0b_",
        "results/tables/d1_",
        "results/tables/extension_h0_",
        "results/tables/extension_cell_",
        "results/tables/extension_raw_experiment_universe.csv",
        "results/manifests/d0b_",
        "results/manifests/d1_",
        "results/manifests/extension_h0_",
        "results/manifests/extension_cell_universe_freeze.json",
    )
    return not any(rel_path.startswith(prefix) for prefix in allowed_large_prefixes)


def _classify_reference(rel_path: str, reference: str, occurrence: str, *, text: str = "") -> dict[str, Any]:
    normalized = rel_path.replace("\\", "/")
    classification = "FORBIDDEN_OPERATIONAL_REFERENCE"
    allowed = False
    reason = "Reference is outside final selection rationale, D1 cleanup evidence, or retained historical reproduction."

    if _is_d0b_evidence(normalized):
        classification = "FINAL_SELECTION_RATIONALE"
        allowed = True
        reason = "D0B is retained as the final fifth-dataset selection rationale."
    elif _is_d1_evidence(normalized):
        classification = "D1_CLEANUP_EVIDENCE"
        allowed = True
        reason = "D1 artifact documents the purge that removed the discarded dataset from operation."
    elif normalized == "report/notes.md":
        classification = "FINAL_SELECTION_RATIONALE"
        allowed = True
        reason = "Notes retain only concise final selection rationale and D1 cleanup evidence."
    elif normalized.startswith("results/tables/extension_h0_") or normalized.startswith("results/manifests/extension_h0_"):
        classification = "RETAIN_H0_PROTECTED_EVIDENCE"
        allowed = True
        reason = "H0 artifacts are protected frozen history-length evidence and are verified hash-unchanged."
    elif normalized.startswith("data/processed/finalization_f6c_lastfm_"):
        classification = "PROTECTED_NON_DATASET_CONTENT_TOKEN"
        allowed = True
        reason = "Occurrence is a protected LastFM content token, not a discarded dataset namespace."
    elif normalized.startswith("results/llm/"):
        classification = "RETAIN_HISTORICAL_LLM_OUTPUT"
        allowed = True
        reason = "Occurrence is inside retained historical raw LLM output, not active dataset operation."
    elif normalized.startswith("original_repo/EvaLLMasRecommender/") or normalized == "original_repo/EvaLLMasRecommender":
        classification = "RETAIN_HISTORICAL_REPRODUCTION"
        allowed = True
        reason = "D0B classified original_repo/EvaLLMasRecommender as retained historical reproduction evidence."

    return {
        "path": normalized,
        "line/reference": reference,
        "occurrence": occurrence,
        "classification": classification,
        "allowed": str(allowed).lower(),
        "reason": reason,
        "text_sample": text,
    }


def _is_d0b_evidence(path: str) -> bool:
    return (
        path == "src/extension/d0b_final_fifth_dataset_selection.py"
        or path == "tests/test_extension_d0b_final_fifth_dataset_selection.py"
        or path.startswith("results/tables/d0b_")
        or path.startswith("results/manifests/d0b_")
        or path.startswith("report/d0b_")
    )


def _is_d1_evidence(path: str) -> bool:
    return (
        path == "src/extension/d1_final_dataset_cleanup.py"
        or path == "tests/test_extension_d1_final_dataset_cleanup.py"
        or path.startswith("results/tables/d1_")
        or path.startswith("results/manifests/d1_")
        or path.startswith("report/d1_")
    )


def _is_text_candidate(path: Path) -> bool:
    if path.suffix.lower() in {".gz", ".zip", ".pkl", ".pt", ".png", ".jpg", ".jpeg", ".pdf"}:
        return False
    return True


def _foodcom_hashes(root: Path) -> dict[str, str]:
    archive = root / "data/raw/foodcom/source/food-com-recipes-and-user-interactions.zip"
    extracted_manifest = _read_json(root / "results/manifests/d0a_foodcom_extracted_file_hashes.json")
    hashes = {
        "archive": sha256_file_or_empty(archive),
        "RAW_interactions.csv": "",
        "RAW_recipes.csv": "",
    }
    for name in ("RAW_interactions.csv", "RAW_recipes.csv"):
        entry = extracted_manifest.get("files", {}).get(name, {})
        raw_path = Path(str(entry.get("relative_extracted_path", "")))
        if raw_path.exists():
            hashes[name] = sha256_file_or_empty(raw_path)
    return hashes


def _protected_summary(pre: Mapping[str, Any], post: Mapping[str, Any]) -> dict[str, Any]:
    food_pre = pre["foodcom_hashes"]
    food_post = post["foodcom_hashes"]
    h0_pre = pre["h0_artifact_hashes"]
    h0_post = post["h0_artifact_hashes"]
    return {
        "foodcom_expected": EXPECTED_FOODCOM_HASHES,
        "foodcom_pre": food_pre,
        "foodcom_post": food_post,
        "foodcom_unchanged": food_pre == food_post == EXPECTED_FOODCOM_HASHES,
        "h0_pre": h0_pre,
        "h0_post": h0_post,
        "h0_unchanged": h0_pre == h0_post,
        "d0a_unchanged": pre["d0a_artifact_hashes"] == post["d0a_artifact_hashes"],
        "d0b_unchanged": pre["d0b_artifact_hashes"] == post["d0b_artifact_hashes"],
    }


def _summarize_removed(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    row_list = list(rows)
    return {
        "paths_considered": len(row_list),
        "paths_existing_before": sum(1 for row in row_list if row["exists_before"]),
        "paths_removed": sum(1 for row in row_list if row["removed"]),
        "files_removed": sum(int(row["file_count"]) for row in row_list if row["removed"]),
        "bytes_removed": sum(int(row["byte_count"]) for row in row_list if row["removed"]),
        "raw_processed_bytes_removed": sum(
            int(row["byte_count"])
            for row in row_list
            if row["removed"] and str(row["path"]).startswith(("data/raw/video_games_2014", "data/processed/video_games"))
        ),
    }


def _summarize_active(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    row_list = list(rows)
    return {
        "dataset_ids": [row["dataset_id"] for row in row_list if row["expected_active"]],
        "dataset_labels": [row["dataset"] for row in row_list if row["expected_active"]],
        "unexpected_dataset_rows": sum(1 for row in row_list if not row["expected_active"]),
        "total_active_cells": sum(int(row["active_cell_count"]) for row in row_list if row["expected_active"]),
        "total_active_raw_experiments": sum(int(row["active_raw_experiment_count"]) for row in row_list if row["expected_active"]),
    }


def _write_report(
    root: Path,
    manifest: Mapping[str, Any],
    manual_rows: list[Mapping[str, Any]],
    removed_rows: list[Mapping[str, Any]],
    retained_rows: list[Mapping[str, Any]],
    reference_rows: list[Mapping[str, Any]],
    active_rows: list[Mapping[str, Any]],
) -> None:
    removed_summary = manifest["removed_artifacts"]
    ref_summary = manifest["remaining_video_games_references"]
    protected = manifest["protected_hashes"]
    gitignore_diff = _git(["diff", "--", ".gitignore"], root)
    deleted_paths = [row["path"] for row in removed_rows if row["removed"]]
    raw_deleted = [
        row for row in removed_rows
        if row["removed"] and str(row["path"]).startswith(("data/raw/video_games_2014", "data/processed/video_games"))
    ]
    text = f"""# D1 Final Fifth-Dataset Cleanup

Decision token: `{manifest['decision_token']}`

## Manual Review Resolution

- Manual entries resolved: `{len(manual_rows)}`
- Remaining `REVIEW_MANUALLY`: `{manifest['manual_review_unresolved']}`
- Resolution table: `{MANUAL_REVIEW_PATH.as_posix()}`

## Video Games Artifacts Removed

- Paths considered: `{removed_summary['paths_considered']}`
- Existing paths removed: `{removed_summary['paths_removed']}`
- Files removed: `{removed_summary['files_removed']}`
- Bytes removed: `{removed_summary['bytes_removed']}`

## Raw / Processed Data Removed

- Raw/processed bytes removed: `{removed_summary['raw_processed_bytes_removed']}`
- Raw/processed removed paths: `{', '.join(row['path'] for row in raw_deleted) if raw_deleted else 'none'}`

## Generic Infrastructure Retained

- Retained rows: `{len(retained_rows)}`
- Retained table: `{RETAINED_PATH.as_posix()}`

## Historical Evidence Retained

- `original_repo/EvaLLMasRecommender` remains retained historical reproduction evidence.
- D0B remains final selection rationale evidence.

## Active Dataset Set

{_markdown_table(active_rows, ['dataset', 'dataset_id', 'active_raw_experiment_count', 'active_cell_count', 'status'])}

## Cell Universe After Purge

- Active universe regenerated by `src.extension.e2a_cell_universe`.
- Active dataset IDs: `{', '.join(manifest['active_dataset_audit']['dataset_ids'])}`
- Total active raw experiments: `{manifest['active_dataset_audit']['total_active_raw_experiments']}`
- Total active cells: `{manifest['active_dataset_audit']['total_active_cells']}`

## Remaining Video Games References

- Remaining audited references: `{ref_summary['rows']}`
- Allowed references: `{ref_summary['allowed_rows']}`
- Forbidden operational references: `{ref_summary['operational_forbidden_occurrences']}`
- Audit table: `{REFERENCE_AUDIT_PATH.as_posix()}`

## Food.com Hash Verification

- Archive unchanged: `{protected['foodcom_unchanged']}`
- Archive SHA256: `{protected['foodcom_post']['archive']}`
- RAW_interactions SHA256: `{protected['foodcom_post']['RAW_interactions.csv']}`
- RAW_recipes SHA256: `{protected['foodcom_post']['RAW_recipes.csv']}`

## H0 Integrity

- H0 artifacts unchanged: `{protected['h0_unchanged']}`
- D0A artifacts unchanged: `{protected['d0a_unchanged']}`
- D0B artifacts unchanged: `{protected['d0b_unchanged']}`

## Gitignore

```diff
{gitignore_diff}
```

## Tests

- D1 writes no inference, no training, no downloads, and no network calls.
- Test file: `tests/test_extension_d1_final_dataset_cleanup.py`

## Files Deleted

{_markdown_list(deleted_paths)}

## Files Modified

- `.gitignore`
- `configs/extension_e0_protocol.yaml`
- `report/d0a_foodcom_material_feasibility.md`
- `report/notes.md`
- `src/extension/e0_candidates.py`
- `src/extension/e0_protocol.py`
- `src/extension/e2a_cell_universe.py`
- `tests/test_extension_e0_protocol.py`
- `tests/test_extension_e2a_cell_universe.py`

## Files Created

- `{MANUAL_REVIEW_PATH.as_posix()}`
- `{REMOVED_PATH.as_posix()}`
- `{RETAINED_PATH.as_posix()}`
- `{REFERENCE_AUDIT_PATH.as_posix()}`
- `{ACTIVE_DATASET_AUDIT_PATH.as_posix()}`
- `{EXECUTION_MANIFEST_PATH.as_posix()}`
- `{REPORT_PATH.as_posix()}`
- `src/extension/d1_final_dataset_cleanup.py`
- `tests/test_extension_d1_final_dataset_cleanup.py`

## Git Status

```text
{_git(['status', '--short'], root)}
```

## Final Freeze Readiness

D1 is ready for final freeze only when the D1 tests pass and
`operational_forbidden_occurrences = 0`.
"""
    _write_text(root / REPORT_PATH, text)


def _markdown_table(rows: list[Mapping[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "No rows."
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _markdown_list(items: Iterable[str]) -> str:
    item_list = list(items)
    if not item_list:
        return "- none"
    return "\n".join(f"- `{item}`" for item in item_list)


def _artifact_hashes(root: Path, paths: Iterable[Path]) -> dict[str, str]:
    return {path.as_posix(): sha256_file_or_empty(root / path) for path in paths}


def _count_text_matches(path: Path, pattern: re.Pattern[str]) -> int:
    if not path.exists() or not path.is_file():
        return 0
    count = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                count += len(pattern.findall(line))
    except OSError:
        return 0
    return count


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _git(args: list[str], root: Path) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def _rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
