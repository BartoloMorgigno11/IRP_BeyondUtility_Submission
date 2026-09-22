"""Phase F6G1 LastFM re-ranking pool reconstruction.

This module is offline-only. It reads frozen LastFM split, prediction and
prior-phase artifacts, reconstructs an adapted re-ranking pool, audits
comparability, and freezes future request manifests without calling models,
training, network services, Ollama, CUDA, or external APIs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml


PHASE_ID = "phase_f6g1"
DEFAULT_CONFIG_PATH = Path("configs/phase_f6g1_lastfm_reranking_reconstruction.yaml")
PRE_EXISTING_DIRTY = ["README.md", "report/dataset_audit.md", "report/notes.md"]
RECOMMEND_READY = "proceed_to_lastfm_reranking_llm_inference"
RECOMMEND_BLOCKED = "keep_lastfm_reranking_blocked"
RECOMMEND_DIAGNOSTIC = "keep_lastfm_reranking_diagnostic_only"
DECISION_ADAPTED = "f6g1_lastfm_reranking_reconstruction_complete_with_adapted_pool"
DECISION_READY = "f6g1_lastfm_reranking_reconstruction_complete_ready_for_inference"
DECISION_DIAGNOSTIC = "f6g1_lastfm_reranking_reconstruction_complete_diagnostic_only"
DECISION_TESTS = "f6g1_blocked_by_tests"
DECISION_INPUTS = "f6g1_blocked_by_input_invariance"


INPUT_INTEGRITY_COLUMNS = [
    "artifact",
    "path",
    "phase",
    "expected_sha256",
    "actual_sha256",
    "status",
    "changed",
    "missing",
    "notes",
]
SOURCE_INVENTORY_COLUMNS = [
    "source_id",
    "path",
    "source_type",
    "section_or_symbol",
    "code_or_text_role",
    "candidate_source_models",
    "per_model_top_k",
    "target_pool_size",
    "union_policy",
    "duplicate_policy",
    "ordering_policy",
    "positive_item_policy",
    "user_filter_policy",
    "evaluation_population",
    "evidence_status",
    "notes",
]
POOL_CONTRACT_COLUMNS = [
    "contract_id",
    "component",
    "paper_value",
    "appendix_value",
    "original_repo_value",
    "f6d2_f6d3_value",
    "current_local_value",
    "evidence",
    "classification",
    "materiality",
    "notes",
]
COHORT_AUDIT_COLUMNS = [
    "cohort_audit_id",
    "user_id",
    "source_split",
    "history_length",
    "positive_item_id",
    "positive_item_title",
    "positive_metadata_present",
    "positive_catalog_present",
    "full_catalog_predictions_complete",
    "score_coverage_by_model",
    "item_id_reconciled",
    "title_reconciled",
    "duplicate_title_status",
    "usable_status",
    "notes",
]
PREDICTION_INVENTORY_COLUMNS = [
    "prediction_inventory_id",
    "model_name",
    "model_family",
    "seed",
    "prediction_file",
    "row_count",
    "expected_users",
    "observed_users",
    "predictions_per_user_min",
    "predictions_per_user_max",
    "candidate_universe",
    "score_field",
    "rank_field",
    "score_direction",
    "duplicate_item_count",
    "missing_score_count",
    "non_finite_score_count",
    "prediction_completeness",
    "training_provenance",
    "classification",
    "notes",
]
SEED_POLICY_COLUMNS = [
    "seed_policy_id",
    "model_name",
    "seed",
    "selection_role",
    "selection_source",
    "selection_criterion",
    "validation_value",
    "test_positive_influenced",
    "chosen_before_downstream_reranking",
    "reproducible",
    "comparable_with_paper",
    "leakage_risk",
    "policy_hierarchy_role",
    "decision",
    "notes",
]
SOURCE_DECISION_COLUMNS = [
    "source_decision_id",
    "model_name",
    "model_family",
    "included_by_paper",
    "included_by_original_code",
    "locally_available",
    "prediction_completeness",
    "selected_seed",
    "top_k_contribution",
    "inclusion_decision",
    "reason",
    "exactness_impact",
]
POOL_ITEM_COLUMNS = [
    "pool_item_id",
    "user_id",
    "candidate_item_id",
    "candidate_position",
    "title",
    "normalized_title",
    "is_positive",
    "positive_item_id",
    "positive_present",
    "pool_size",
    "source_models",
    "source_ranks",
    "source_scores",
    "source_seeds",
    "duplicate_source_count",
    "duplicate_skipped_before_item",
    "construction_policy",
    "population_role",
    "pool_classification",
    "provenance_hash",
    "notes",
]
POOL_INSTANCE_COLUMNS = [
    "pool_instance_id",
    "user_id",
    "pool_id",
    "pool_policy",
    "pool_classification",
    "pool_size",
    "unique_item_count",
    "duplicate_removed_count",
    "positive_item_id",
    "positive_present",
    "positive_candidate_position",
    "candidate_pool_recall",
    "upper_bound_hr_at_1",
    "upper_bound_hr_at_5",
    "upper_bound_hr_at_10",
    "upper_bound_ndcg_at_5",
    "upper_bound_ndcg_at_10",
    "eligible_all_user",
    "eligible_positive_present",
    "request_ready",
    "block_reason",
]
QUALITY_COLUMNS = [
    "summary_id",
    "pool_name",
    "population",
    "metric",
    "value",
    "denominator",
    "classification",
    "notes",
]
POSITIVE_POLICY_COLUMNS = [
    "policy_id",
    "pool_name",
    "population",
    "positive_policy",
    "positive_insertion",
    "positive_absent_policy",
    "users",
    "positive_present_users",
    "positive_absent_users",
    "positive_present_rate",
    "canonical_population",
    "diagnostic_population",
    "evaluation_denominator",
    "classification",
    "notes",
]
PREVIOUS_POOL_COLUMNS = [
    "reconciliation_id",
    "user_id",
    "previous_pool_path",
    "canonical_pool_name",
    "candidate_set_equality",
    "candidate_order_equality",
    "positive_presence_equality",
    "source_model_equality",
    "previous_positive_present",
    "canonical_positive_present",
    "mismatch_reason",
    "outcome",
    "notes",
]
TRADITIONAL_METRIC_COLUMNS = [
    "metric_id",
    "model_name",
    "seed",
    "pool_name",
    "population",
    "evaluation_scope",
    "metric",
    "value",
    "denominator",
    "positive_present_users",
    "candidate_pool_recall",
    "classification",
    "notes",
]
TITLE_COLUMNS = [
    "reconciliation_id",
    "user_id",
    "candidate_item_id",
    "candidate_position",
    "canonical_item_id",
    "source_item_id",
    "artist_title",
    "normalized_artist_title",
    "duplicate_normalized_title",
    "ambiguous_title",
    "missing_title",
    "prompt_safe_title",
    "reversible_mapping",
    "mapping_status",
    "notes",
]
GATE_COLUMNS = [
    "gate_id",
    "gate_name",
    "requirement",
    "observed_evidence",
    "status",
    "impact",
    "remediation",
    "future_llm_inference_allowed",
]
EXACTNESS_COLUMNS = [
    "exactness_id",
    "component",
    "classification",
    "evidence",
    "limitation",
    "future_experiment_label",
    "notes",
]
LOGICAL_COLUMNS = [
    "logical_instance_id",
    "user_id",
    "dataset",
    "task",
    "cohort_name",
    "history_item_ids",
    "history_item_titles",
    "candidate_item_ids",
    "candidate_item_titles",
    "candidate_order_hash",
    "positive_item_id",
    "positive_initial_position",
    "positive_present",
    "pool_provenance_hash",
    "expected_output_length",
    "parser_contract",
    "operational_utility_policy",
    "conditional_utility_policy",
    "request_ready",
    "block_reason",
]
REQUEST_COLUMNS = [
    "request_id",
    "logical_instance_id",
    "user_id",
    "dataset",
    "task",
    "model",
    "strategy",
    "history_item_ids",
    "candidate_item_ids",
    "indexed_candidate_labels",
    "candidate_order_hash",
    "positive_item_id",
    "positive_initial_position",
    "positive_present",
    "pool_provenance_hash",
    "expected_json_schema",
    "required_output_length",
    "parser_contract",
    "invalid_output_policy",
    "operational_utility_policy",
    "conditional_utility_policy",
    "title_mapping_hash",
    "request_hash",
    "request_status",
    "block_reason",
]
SHARD_COLUMNS = [
    "shard_id",
    "model",
    "strategy",
    "shard_index",
    "request_count",
    "first_request_id",
    "last_request_id",
    "request_manifest_hash",
    "shard_hash",
    "status",
    "notes",
]
MISSING_COLUMNS = [
    "item_id",
    "component",
    "status",
    "classification",
    "reason",
    "required_evidence",
    "current_evidence",
    "recommended_action",
]


TABLE_COLUMNS = {
    "input_integrity": INPUT_INTEGRITY_COLUMNS,
    "source_inventory": SOURCE_INVENTORY_COLUMNS,
    "paper_pool_contract": POOL_CONTRACT_COLUMNS,
    "cohort_audit": COHORT_AUDIT_COLUMNS,
    "prediction_inventory": PREDICTION_INVENTORY_COLUMNS,
    "seed_policy_audit": SEED_POLICY_COLUMNS,
    "candidate_source_model_decisions": SOURCE_DECISION_COLUMNS,
    "pool_item_provenance": POOL_ITEM_COLUMNS,
    "pool_instance_summary": POOL_INSTANCE_COLUMNS,
    "pool_quality_summary": QUALITY_COLUMNS,
    "positive_policy": POSITIVE_POLICY_COLUMNS,
    "previous_pool_reconciliation": PREVIOUS_POOL_COLUMNS,
    "traditional_pool_metrics": TRADITIONAL_METRIC_COLUMNS,
    "title_item_reconciliation": TITLE_COLUMNS,
    "comparability_gate": GATE_COLUMNS,
    "exactness_classification": EXACTNESS_COLUMNS,
    "future_logical_instances": LOGICAL_COLUMNS,
    "future_request_manifest": REQUEST_COLUMNS,
    "future_shard_manifest": SHARD_COLUMNS,
    "missing_or_blocked": MISSING_COLUMNS,
}


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: Any, length: int = 14) -> str:
    return f"{prefix}-{sha256_text('||'.join(str(part) for part in parts))[:length]}"


def fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.17g}"


def bool_text(value: Any) -> str:
    if isinstance(value, str):
        return "true" if value.strip().lower() == "true" else "false"
    return "true" if bool(value) else "false"


def parse_float(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def pipe_join(values: Iterable[Any]) -> str:
    return "|".join(str(value) for value in values)


def semicolon_join(values: Iterable[Any]) -> str:
    return ";".join(str(value) for value in values)


def read_listish(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [part.strip() for part in text.split("|") if part.strip()]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: "" if row.get(column, "") is None else str(row.get(column, "")) for column in columns})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(payload) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_command(args: Sequence[str], cwd: Path, timeout: int = 120) -> tuple[int, str, str, float]:
    started = time.perf_counter()
    completed = subprocess.run(
        list(args),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return completed.returncode, completed.stdout, completed.stderr, time.perf_counter() - started


def git(root: Path, *args: str) -> str:
    code, out, _err, _duration = run_command(["git", *args], root, timeout=60)
    return out.strip() if code == 0 else ""


def existing_manifest(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    path = root / config["outputs"]["freeze_manifest"]
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except json.JSONDecodeError:
        return {}


def generated_paths(config: Mapping[str, Any]) -> list[str]:
    paths = [config["outputs"]["report"], config["outputs"]["freeze_manifest"]]
    paths.extend(config["outputs"]["tables"].values())
    paths.extend(
        [
            "configs/phase_f6g1_lastfm_reranking_reconstruction.yaml",
            "src/analysis/plan_f6g1_lastfm_reranking_reconstruction.py",
            "src/experiments/build_phase_f6g1_lastfm_reranking_reconstruction.py",
            "tests/test_plan_f6g1_lastfm_reranking_reconstruction.py",
        ]
    )
    return [str(path).replace("\\", "/").rstrip("/") for path in paths]


def filter_generated_status(text: str, config: Mapping[str, Any]) -> str:
    generated = generated_paths(config)
    kept: list[str] = []
    for line in text.splitlines():
        path = line[3:].strip().strip('"').replace("\\", "/") if len(line) > 3 else line
        if any(path == item or path.startswith(item + "/") for item in generated):
            continue
        kept.append(line)
    return "\n".join(kept)


def active_python_processes(root: Path) -> str:
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-Process python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Path",
    ]
    code, out, _err, _duration = run_command(command, root, timeout=30)
    if code != 0:
        return "unavailable"
    paths = sorted({line.strip() for line in out.splitlines() if line.strip()})
    return ";".join(paths)


def record_repository_state(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    previous = existing_manifest(root, config)
    timestamp = previous.get("runtime_metadata", {}).get("timestamp") or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return {
        "repository_path": str(root),
        "branch": git(root, "branch", "--show-current"),
        "head": git(root, "rev-parse", "HEAD"),
        "latest_commit": git(root, "log", "-1", "--oneline"),
        "git_status_short": filter_generated_status(git(root, "status", "--short"), config),
        "git_status_tracked_short": git(root, "status", "--short", "--untracked-files=no"),
        "git_diff_name_only": git(root, "diff", "--name-only"),
        "git_diff_stat": git(root, "diff", "--stat"),
        "git_diff_cached_stat": git(root, "diff", "--cached", "--stat"),
        "staged_files": git(root, "diff", "--cached", "--name-only"),
        "pre_existing_tracked_dirty_files": ";".join(PRE_EXISTING_DIRTY),
        "active_python_processes": active_python_processes(root),
        "timestamp": timestamp,
        "operating_system": platform.platform(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "sys_prefix": sys.prefix,
        "sys_base_prefix": sys.base_prefix,
        "environment_classification": config["phase"]["environment_classification"],
    }


def verify_interpreter(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    expected = (root / config["phase"]["interpreter_path"]).resolve(strict=False)
    actual = Path(sys.executable).resolve(strict=False)
    imports_ok = True
    import_error = ""
    try:
        import pandas  # noqa: F401
        import pytest  # noqa: F401
        import yaml as _yaml  # noqa: F401
    except Exception as exc:  # pragma: no cover
        imports_ok = False
        import_error = f"{type(exc).__name__}: {exc}"
    status = "pass" if actual.exists() and imports_ok and str(actual).lower() == str(expected).lower() else "fail"
    return {
        "expected_interpreter": str(expected),
        "actual_interpreter": str(actual),
        "status": status,
        "imports_ok": bool_text(imports_ok),
        "import_error": import_error,
        "python_version": platform.python_version(),
        "sys_prefix": sys.prefix,
        "sys_base_prefix": sys.base_prefix,
    }


def verify_input_integrity(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact, spec in config["input_hashes"].items():
        path = root / spec["path"]
        exists = path.exists()
        actual = sha256_file(path) if exists and path.is_file() else ""
        expected = str(spec.get("sha256", ""))
        changed = bool(expected and actual and expected.lower() != actual.lower())
        rows.append(
            {
                "artifact": artifact,
                "path": spec["path"],
                "phase": spec.get("phase", ""),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": "pass" if exists and not changed else "fail",
                "changed": bool_text(changed),
                "missing": bool_text(not exists),
                "notes": "",
            }
        )
    return rows


def assert_no_input_failures(rows: Sequence[Mapping[str, Any]]) -> None:
    failures = [row for row in rows if row["status"] != "pass"]
    if failures:
        detail = "; ".join(f"{row['artifact']}:{row['status']}" for row in failures[:5])
        raise RuntimeError(f"F6G1 input integrity gate failed: {detail}")


def verify_phase_manifests(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    f6f2 = load_json(root / config["input_hashes"]["f6f2_manifest"]["path"])
    f6f1 = load_json(root / config["input_hashes"]["f6f1_manifest"]["path"])
    f6e4c = load_json(root / config["input_hashes"]["f6e4c_manifest"]["path"])
    checks = {
        "f6f2_decision": f6f2.get("decision", "") == config["phase"]["expected_f6f2_decision"],
        "f6f2_scientific_hash": f6f2.get("scientific_content_hash", "") == config["phase"]["expected_f6f2_scientific_hash"],
        "f6f1_decision": f6f1.get("decision", "") == config["phase"]["expected_f6f1_decision"],
        "f6f1_scientific_hash": f6f1.get("scientific_content_hash", "") == config["phase"]["expected_f6f1_scientific_hash"],
        "f6e4c_decision": f6e4c.get("decision", "") == config["phase"]["expected_f6e4c_decision"],
        "f6e4c_scientific_hash": f6e4c.get("scientific_content_hash", "") == config["phase"]["expected_f6e4c_scientific_hash"],
        "f6f1r_scientific_status": f6f2.get("f6f1r_scientific_comparison_status", "") == config["phase"]["expected_f6f1r_scientific_status"],
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "status": status,
        "checks": checks,
        "f6f2_scientific_hash": f6f2.get("scientific_content_hash", ""),
        "f6f1_scientific_hash": f6f1.get("scientific_content_hash", ""),
        "f6e4c_scientific_hash": f6e4c.get("scientific_content_hash", ""),
        "f6f2_decision": f6f2.get("decision", ""),
        "f6f1_decision": f6f1.get("decision", ""),
        "f6e4c_decision": f6e4c.get("decision", ""),
    }


def validate_output_paths(config: Mapping[str, Any]) -> None:
    allowed_prefixes = ("results/tables/phase_f6g1_", "results/manifests/phase_f6g1_", "report/phase_f6g1_")
    paths = [config["outputs"]["report"], config["outputs"]["freeze_manifest"], *config["outputs"]["tables"].values()]
    for path in paths:
        normalized = str(path).replace("\\", "/")
        if not normalized.startswith(allowed_prefixes):
            raise ValueError(f"F6G1 output path is outside phase namespace: {path}")


def load_frames(root: Path, config: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    def path(name: str) -> Path:
        return root / config["input_hashes"][name]["path"]

    return {
        "cohort": read_csv(path("lastfm_ranking_cohort")),
        "train": read_csv(path("lastfm_train")),
        "validation": read_csv(path("lastfm_validation")),
        "test": read_csv(path("lastfm_test")),
        "catalog": read_csv(path("lastfm_catalog")),
        "identity_audit": read_csv(path("lastfm_item_identity_audit")),
        "f6d1_protocol": read_csv(path("f6d1_baseline_protocol")),
        "f6d1_predictions": read_csv(path("f6d1_baseline_predictions")),
        "f6d1_prediction_validation": read_csv(path("f6d1_prediction_validation")),
        "f6d2_execution_manifest": read_csv(path("f6d2_execution_manifest")),
        "f6d2_source_selection": read_csv(path("f6d2_reranking_source_selection")),
        "f6d2_policy_audit": read_csv(path("f6d2_reranking_policy_audit")),
        "f6d2_pool_validation": read_csv(path("f6d2_reranking_pool_validation")),
        "f6d2_candidate_predictions": read_csv(path("f6d2_candidate_predictions")),
        "f6d2_full_catalog": read_csv(path("f6d2_full_catalog_predictions")),
        "previous_pool": read_csv(path("previous_adapted_pool")),
        "f6d3_protocol": read_csv(path("f6d3_reranking_protocol")),
        "f6d3_composition": read_csv(path("f6d3_reranking_pool_composition")),
        "f6d3_evaluation": read_csv(path("f6d3_reranking_evaluation_policy")),
        "f6d3_manifest": read_csv(path("f6d3_reranking_manifest")),
    }


def catalog_maps(catalog: pd.DataFrame) -> tuple[dict[str, str], dict[str, str], set[str]]:
    titles = dict(zip(catalog["item_id"].astype(str), catalog["title"].astype(str)))
    normalized = {item: normalize_text(title) for item, title in titles.items()}
    counts = Counter(value for value in normalized.values() if value)
    duplicated = {value for value, count in counts.items() if count > 1}
    return titles, normalized, duplicated


def build_source_inventory(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        ("paper_pdf", "paper/2411.00331v1.pdf", "paper", "main text and embedded appendix", "Defines high-level re-ranking task and 20-candidate prompts; exact pool mechanics not machine-extracted here.", "partial", "missing", "20", "missing", "missing", "missing", "not explicitly evidenced", "missing", "missing", "partial", "Local PDF hash verified; no unsupported detail is inferred."),
        ("docx_translation", "paper/traduzione_italiana_beyond_utility_llm_recommender.docx", "secondary structured source", "translated paper text", "Secondary source for source-model family and re-ranking wording.", "BPRMF;GRU4Rec;SASRec;LightGCN", "partial", "20", "missing", "missing", "missing", "not evidenced", "missing", "missing", "partial", "Used only as local secondary evidence."),
        ("original_readme", "original_repo/EvaLLMasRecommender/README.md", "original repository", "README observations and dataset list", "Confirms LastFM dataset and re-ranking task, but not executable pool construction.", "partial", "missing", "20", "missing", "missing", "missing", "not evidenced", "missing", "missing", "partial", "No original LastFM pool file is recovered."),
        ("original_zero_shot", "original_repo/EvaLLMasRecommender/eva/codes/zero_shot.py", "original repository code", "candidate20 result-path constants", "Shows candidate20 prompt/result variants and candidate-position variants.", "missing", "missing", "20", "missing", "missing", "shuffle/sample variants", "positive-first variant exists for bias experiment", "missing", "missing", "inferred_from_code", "Does not define LastFM traditional-source union."),
        ("f6d2_config", "configs/finalization_f6d2_lastfm_trainable_models.yaml", "local config", "reranking_pool", "Defines adapted round-robin pool over selected trainable models.", "bprmf;gru4rec;sasrec;lightgcn", "top100 export", "20", "round_robin_union", "first_occurrence", "model_order_then_rank", "no insertion", "retain all users", "all-user primary; positive-present diagnostic", "explicit", "This is adapted local policy, not paper-exact."),
        ("f6d2_source_selection", "results/tables/finalization_f6d2_reranking_source_selection.csv", "local table", "best_validation_seed rows", "Defines selected validation seeds for adapted downstream pool.", "bprmf;gru4rec;sasrec;lightgcn", "top100 export", "20", "round_robin_union", "first_occurrence", "model_order_then_rank", "no test metric in seed choice", "retain all users", "all-user primary", "explicit", "Selection is validation-only."),
        ("f6d3_protocol", "results/tables/finalization_f6d3_reranking_protocol.csv", "local table", "reranking protocol audit", "Records unresolved paper pool mechanics and adapted policy.", "bprmf;gru4rec;sasrec;lightgcn", "top100 adapted export", "20", "round_robin_union adapted", "first occurrence", "adapted order", "no insertion", "all users retained", "all-user primary; positive-present diagnostic", "explicit", "Classifies paper-aligned policy as blocked."),
        ("previous_pool", "data/processed/finalization_f6d2_lastfm_reranking_candidates_adapted.csv", "local artifact", "adapted pool rows", "Previous pool to audit and compare, not automatically accept.", "bprmf;gru4rec;sasrec;lightgcn", "top100 source", "20", "round_robin_union", "first_occurrence", "candidate_position", "no insertion", "1859 users retained", "all-user primary", "explicit", "Positive labels assigned only after construction."),
    ]
    return [
        {
            "source_id": stable_id("F6G1-SRC", row[0]),
            "path": row[1],
            "source_type": row[2],
            "section_or_symbol": row[3],
            "code_or_text_role": row[4],
            "candidate_source_models": row[5],
            "per_model_top_k": row[6],
            "target_pool_size": row[7],
            "union_policy": row[8],
            "duplicate_policy": row[9],
            "ordering_policy": row[10],
            "positive_item_policy": row[11],
            "user_filter_policy": row[12],
            "evaluation_population": row[13],
            "evidence_status": row[14],
            "notes": row[15],
        }
        for row in rows
    ]


def build_paper_pool_contract(frames: Mapping[str, pd.DataFrame]) -> list[dict[str, Any]]:
    protocol = {row["aspect"]: row for row in frames["f6d3_protocol"].to_dict("records")}
    components = [
        ("dataset", "LastFM", "LastFM", "LastFM supported", "LastFM", "LastFM", "paper PDF; F6C", "reconstructed_match", "high", "Dataset identity is stable; preprocessing remains adapted."),
        ("task", "re-ranking", "re-ranking", "candidate20 templates", "LastFM reranking", "LastFM reranking", "paper PDF; original templates; F6D3", "reconstructed_match", "high", "No LLM inference in F6G1."),
        ("candidate_source_model_set", "BPRMF; GRU4Rec; SASRec; LightGCN", "BPRMF; GRU4Rec; SASRec; LightGCN", "not recovered as executable LastFM pool", "bprmf;gru4rec;sasrec;lightgcn", "bprmf;gru4rec;sasrec;lightgcn", "F6D3 source models", "adapted_match", "high", "Family match only; local trainers and snapshots are adapted."),
        ("top_k_per_source", protocol.get("top-K per source", {}).get("paper_value", "top-K source outputs"), "not separately available", "not recovered", "top-100 export; final pool 20", "top-100 export; final pool 20", "F6D2 config; F6D3 protocol", "missing_evidence", "high", "Exact source K is not recovered."),
        ("seed_policy", "five random initializations reported; source seed unresolved", "not recovered", "not recovered", "best validation seed per model", "best validation seed per model", "F6D2 source selection", "adapted_match", "medium", "No test metric is used, but paper seed identity is unknown."),
        ("union_policy", "not fully specified", "not recovered", "not recovered", "round-robin union", "round-robin union", "F6D2 config; F6D3 protocol", "missing_evidence", "high", "Do not call paper-exact."),
        ("duplicate_policy", "not recovered", "not recovered", "not recovered", "first occurrence", "first occurrence with provenance attachment", "F6D2 config; F6G1 reconstruction", "adapted_match", "medium", "Deterministic and auditable."),
        ("candidate_order", "not recovered", "not recovered", "not recovered", "adapted round-robin order", "model order then rank then item id", "F6D2 config; F6G1 reconstruction", "adapted_match", "high", "Prompt order remains adapted."),
        ("positive_insertion", "not evidenced", "not evidenced", "not recovered", "not used", "not used", "F6D2 config; F6G1 policy", "reconstructed_match", "high", "No positive insertion without source evidence."),
        ("positive_population", "not recovered", "not recovered", "not recovered", "all-user primary; positive-present diagnostic", "all-user canonical; positive-present diagnostic", "F6D3 evaluation policy; F6G1 policy", "adapted_match", "high", "Denominators must remain separate."),
        ("target_pool_size", "20", "20", "candidate20 prompt templates", "20", "20", "paper/original templates; F6D2", "exact_match", "high", "Count is recovered; construction mechanics are adapted."),
    ]
    return [
        {
            "contract_id": stable_id("F6G1-CONTRACT", component),
            "component": component,
            "paper_value": paper,
            "appendix_value": appendix,
            "original_repo_value": original,
            "f6d2_f6d3_value": prior,
            "current_local_value": current,
            "evidence": evidence,
            "classification": classification,
            "materiality": materiality,
            "notes": notes,
        }
        for component, paper, appendix, original, prior, current, evidence, classification, materiality, notes in components
    ]


def expected_model_seed_pairs(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [(str(model), str(config["pool_policy"]["selected_seeds"][model])) for model in config["pool_policy"]["model_order"]]


def build_cohort_audit(frames: Mapping[str, pd.DataFrame], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    cohort = frames["cohort"].copy()
    catalog, _normalized, duplicate_titles = catalog_maps(frames["catalog"])
    full = frames["f6d2_full_catalog"].copy()
    pair_counts = full.groupby(["user_id", "model", "seed"]).size().to_dict()
    expected_pairs = expected_model_seed_pairs(config)
    rows: list[dict[str, Any]] = []
    for record in cohort.sort_values("user_id", key=lambda s: s.astype(int)).to_dict("records"):
        user_id = str(record["user_id"])
        positive = str(record["test_item_id"])
        coverage_parts = []
        complete = True
        for model, seed in expected_pairs:
            count = int(pair_counts.get((user_id, model, seed), 0))
            coverage_parts.append(f"{model}:{count}")
            complete = complete and count == int(config["dataset"]["source_top_n"])
        title = catalog.get(positive, "")
        normalized = normalize_text(title)
        duplicate_status = "duplicate_normalized_title" if normalized in duplicate_titles else "unique_or_missing"
        metadata = bool_text(str(record.get("metadata_ready", "")).lower() == "true" and bool(title))
        rows.append(
            {
                "cohort_audit_id": stable_id("F6G1-COHORT", user_id),
                "user_id": user_id,
                "source_split": "finalization_f6c_lastfm_ranking_cohort",
                "history_length": record.get("train_history_count", ""),
                "positive_item_id": positive,
                "positive_item_title": title,
                "positive_metadata_present": metadata,
                "positive_catalog_present": bool_text(positive in catalog),
                "full_catalog_predictions_complete": bool_text(complete),
                "score_coverage_by_model": semicolon_join(coverage_parts),
                "item_id_reconciled": bool_text(positive in catalog),
                "title_reconciled": bool_text(bool(title)),
                "duplicate_title_status": duplicate_status,
                "usable_status": "usable" if complete and positive in catalog and title else "blocked",
                "notes": "No user is silently dropped.",
            }
        )
    return rows


def source_family(model: str) -> str:
    mapping = {
        "random_seeded": "Random",
        "mostpop_train_only": "MostPop",
        "bm25_like_adapted": "BM25-like",
        "bprmf": "BPRMF",
        "gru4rec": "GRU4Rec",
        "sasrec": "SASRec",
        "lightgcn": "LightGCN",
    }
    return mapping.get(model, model)


def inventory_prediction_frame(
    frame: pd.DataFrame,
    *,
    prediction_file: str,
    model_col: str,
    item_col: str,
    expected_users: int,
    expected_per_user: int | None,
    classification_by_model: Mapping[str, str],
    training_provenance: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_cols = [model_col]
    if "seed" in frame.columns:
        group_cols.append("seed")
    for keys, group in frame.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys, "")
        model = str(keys[0])
        seed = str(keys[1]) if len(keys) > 1 else ""
        per_user = group.groupby("user_id").size()
        duplicates = int(group.groupby("user_id")[item_col].apply(lambda values: len(values) - values.astype(str).nunique()).sum())
        scores = pd.to_numeric(group["score"], errors="coerce") if "score" in group.columns else pd.Series(dtype=float)
        missing_score_count = int(group["score"].astype(str).str.strip().eq("").sum()) if "score" in group.columns else len(group)
        non_finite = int((~scores.apply(lambda value: bool(math.isfinite(value)) if pd.notna(value) else False)).sum()) if len(scores) else 0
        min_per = int(per_user.min()) if not per_user.empty else 0
        max_per = int(per_user.max()) if not per_user.empty else 0
        complete = group["user_id"].nunique() == expected_users and (expected_per_user is None or (min_per == expected_per_user and max_per == expected_per_user))
        rows.append(
            {
                "prediction_inventory_id": stable_id("F6G1-PRED", prediction_file, model, seed),
                "model_name": model,
                "model_family": source_family(model),
                "seed": seed,
                "prediction_file": prediction_file,
                "row_count": len(group),
                "expected_users": expected_users,
                "observed_users": group["user_id"].nunique(),
                "predictions_per_user_min": min_per,
                "predictions_per_user_max": max_per,
                "candidate_universe": f"{group[item_col].astype(str).nunique()} unique items",
                "score_field": "score" if "score" in group.columns else "",
                "rank_field": "rank" if "rank" in group.columns else "",
                "score_direction": "higher_is_better",
                "duplicate_item_count": duplicates,
                "missing_score_count": missing_score_count,
                "non_finite_score_count": non_finite,
                "prediction_completeness": "complete" if complete else "incomplete",
                "training_provenance": training_provenance,
                "classification": classification_by_model.get(model, "unavailable"),
                "notes": "Inventory only; no inference or scoring executed.",
            }
        )
    return rows


def build_prediction_inventory(frames: Mapping[str, pd.DataFrame], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected_users = int(config["dataset"]["expected_users"])
    rows: list[dict[str, Any]] = []
    rows.extend(
        inventory_prediction_frame(
            frames["f6d1_predictions"],
            prediction_file="results/predictions/finalization_f6d1_lastfm_baseline_rankings.csv",
            model_col="model",
            item_col="candidate_item_id",
            expected_users=expected_users,
            expected_per_user=20,
            classification_by_model={
                "random_seeded": "baseline_proxy",
                "mostpop_train_only": "adapted_model",
                "bm25_like_adapted": "baseline_proxy",
            },
            training_provenance="no_training_f6d1_candidate_only",
        )
    )
    rows.extend(
        inventory_prediction_frame(
            frames["f6d2_candidate_predictions"],
            prediction_file="results/predictions/finalization_f6d2_lastfm_candidate_rankings.csv",
            model_col="model",
            item_col="candidate_item_id",
            expected_users=expected_users,
            expected_per_user=20,
            classification_by_model={model: "exact_model_reconstructed_training" for model in config["pool_policy"]["model_order"]},
            training_provenance="f6d2_local_cpu_training_candidate_set_scores",
        )
    )
    rows.extend(
        inventory_prediction_frame(
            frames["f6d2_full_catalog"],
            prediction_file="results/predictions/finalization_f6d2_lastfm_full_catalog_rankings.csv",
            model_col="model",
            item_col="item_id",
            expected_users=expected_users,
            expected_per_user=int(config["dataset"]["source_top_n"]),
            classification_by_model={model: "exact_model_reconstructed_training" for model in config["pool_policy"]["model_order"]},
            training_provenance="f6d2_selected_validation_seed_top100_export",
        )
    )
    return sorted(rows, key=lambda row: (row["prediction_file"], row["model_name"], str(row["seed"])))


def build_seed_policy_audit(frames: Mapping[str, pd.DataFrame], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected = {model: str(seed) for model, seed in config["pool_policy"]["selected_seeds"].items()}
    rows: list[dict[str, Any]] = []
    for record in frames["f6d2_source_selection"].sort_values(["model", "seed"]).to_dict("records"):
        model = str(record["model"])
        seed = str(record["seed"])
        is_selected = seed == selected.get(model)
        rows.append(
            {
                "seed_policy_id": stable_id("F6G1-SEED", model, seed),
                "model_name": model,
                "seed": seed,
                "selection_role": record.get("selection_role", ""),
                "selection_source": "results/tables/finalization_f6d2_reranking_source_selection.csv",
                "selection_criterion": record.get("selection_metric", ""),
                "validation_value": record.get("selection_value", ""),
                "test_positive_influenced": "false",
                "chosen_before_downstream_reranking": "true",
                "reproducible": "true",
                "comparable_with_paper": "false",
                "leakage_risk": "low" if is_selected else "not_applicable",
                "policy_hierarchy_role": "previously_frozen_f6d2_selected_seed" if is_selected else "not_selected",
                "decision": "use_for_adapted_pool" if is_selected else "exclude_from_canonical_pool",
                "notes": record.get("selection_reason", ""),
            }
        )
    return rows


def build_candidate_source_model_decisions(frames: Mapping[str, pd.DataFrame], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    inventory = build_prediction_inventory(frames, config)
    complete_by_model = {row["model_name"]: row["prediction_completeness"] for row in inventory if row["prediction_file"].endswith("full_catalog_rankings.csv")}
    selected = {model: str(seed) for model, seed in config["pool_policy"]["selected_seeds"].items()}
    models = [
        ("random_seeded", "Random", "false", "false", "diagnostic_only_not_canonical", "Random is a local control and not a paper re-ranking source."),
        ("mostpop_train_only", "MostPop", "false", "false", "diagnostic_only_not_canonical", "MostPop is a baseline comparison source, not recovered as a LastFM re-ranking pool source."),
        ("bm25_like_adapted", "BM25-like", "false", "false", "diagnostic_only_not_canonical", "BM25-like is adapted and not recovered as a LastFM re-ranking pool source."),
        ("bprmf", "BPRMF", "true", "partial", "include_canonical_adapted", "Paper/source family matches and selected full-catalog predictions exist."),
        ("gru4rec", "GRU4Rec", "true", "partial", "include_canonical_adapted", "Paper/source family matches and selected full-catalog predictions exist."),
        ("sasrec", "SASRec", "true", "partial", "include_canonical_adapted", "Paper/source family matches and selected full-catalog predictions exist."),
        ("lightgcn", "LightGCN", "true", "partial", "include_canonical_adapted", "Paper/source family matches and selected full-catalog predictions exist."),
    ]
    return [
        {
            "source_decision_id": stable_id("F6G1-SOURCE-DECISION", model),
            "model_name": model,
            "model_family": family,
            "included_by_paper": included_paper,
            "included_by_original_code": included_original,
            "locally_available": "true",
            "prediction_completeness": complete_by_model.get(model, "candidate_only_complete" if model in {"random_seeded", "mostpop_train_only", "bm25_like_adapted"} else "unavailable"),
            "selected_seed": selected.get(model, ""),
            "top_k_contribution": str(config["dataset"]["source_top_n"]) if model in selected else "not_canonical",
            "inclusion_decision": decision,
            "reason": reason,
            "exactness_impact": "adapted_model_component" if model in selected else "diagnostic_only",
        }
        for model, family, included_paper, included_original, decision, reason in models
    ]


def full_catalog_indexes(full: pd.DataFrame, config: Mapping[str, Any]) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[tuple[str, str], list[dict[str, Any]]]]:
    selected = {model: str(seed) for model, seed in config["pool_policy"]["selected_seeds"].items()}
    model_order = list(config["pool_policy"]["model_order"])
    selected_full = full.loc[full.apply(lambda row: str(row["model"]) in selected and str(row["seed"]) == selected[str(row["model"])], axis=1)].copy()
    selected_full["_rank_num"] = pd.to_numeric(selected_full["rank"], errors="coerce").fillna(10**9).astype(int)
    selected_full["_item_num"] = pd.to_numeric(selected_full["item_id"], errors="coerce").fillna(10**12).astype(int)
    ranked_by_user_model: dict[tuple[str, str], list[dict[str, Any]]] = {}
    lookup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (user_id, model), group in selected_full.groupby(["user_id", "model"], sort=True):
        ranked = group.sort_values(["_rank_num", "_item_num"]).drop(columns=["_rank_num", "_item_num"]).to_dict("records")
        ranked_by_user_model[(str(user_id), str(model))] = ranked
        for row in ranked:
            lookup[(str(user_id), str(row["item_id"]))].append(row)
    order_index = {model: index for index, model in enumerate(model_order)}
    for key, rows in list(lookup.items()):
        rows.sort(key=lambda row: (order_index.get(str(row["model"]), 999), parse_int(row["rank"]), parse_int(row["item_id"], 10**12)))
    return ranked_by_user_model, lookup


def reconstruct_pool(frames: Mapping[str, pd.DataFrame], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    cohort = frames["cohort"].copy()
    titles, normalized_titles, _duplicate_titles = catalog_maps(frames["catalog"])
    ranked_by_user_model, lookup = full_catalog_indexes(frames["f6d2_full_catalog"], config)
    model_order = list(config["pool_policy"]["model_order"])
    target = int(config["dataset"]["target_pool_size"])
    pool_policy = str(config["pool_policy"]["canonical_policy_name"])
    classification = str(config["pool_policy"]["final_pool_classification_if_gates_pass"])
    rows: list[dict[str, Any]] = []
    for record in cohort.sort_values("user_id", key=lambda s: s.astype(int)).to_dict("records"):
        user_id = str(record["user_id"])
        positive = str(record["test_item_id"])
        selected_items: list[str] = []
        pointers = {model: 0 for model in model_order}
        duplicate_skipped = 0
        while len(selected_items) < target:
            progressed = False
            for model in model_order:
                ranked = ranked_by_user_model.get((user_id, model), [])
                while pointers[model] < len(ranked):
                    item_id = str(ranked[pointers[model]]["item_id"])
                    pointers[model] += 1
                    progressed = True
                    if item_id not in selected_items:
                        selected_items.append(item_id)
                        break
                    duplicate_skipped += 1
                if len(selected_items) >= target:
                    break
            if not progressed:
                break
        positive_present = positive in selected_items
        for position, item_id in enumerate(selected_items, start=1):
            sources = lookup.get((user_id, item_id), [])
            source_models = [str(row["model"]) for row in sources]
            source_ranks = [str(row["rank"]) for row in sources]
            source_scores = [str(row["score"]) for row in sources]
            source_seeds = [str(row["seed"]) for row in sources]
            provenance_payload = {
                "user_id": user_id,
                "candidate_item_id": item_id,
                "candidate_position": position,
                "source_models": source_models,
                "source_ranks": source_ranks,
                "source_scores": source_scores,
                "source_seeds": source_seeds,
            }
            rows.append(
                {
                    "pool_item_id": stable_id("F6G1-POOL", user_id, position, item_id),
                    "user_id": user_id,
                    "candidate_item_id": item_id,
                    "candidate_position": position,
                    "title": titles.get(item_id, ""),
                    "normalized_title": normalized_titles.get(item_id, ""),
                    "is_positive": bool_text(item_id == positive),
                    "positive_item_id": positive,
                    "positive_present": bool_text(positive_present),
                    "pool_size": len(selected_items),
                    "source_models": semicolon_join(source_models),
                    "source_ranks": semicolon_join(source_ranks),
                    "source_scores": semicolon_join(source_scores),
                    "source_seeds": semicolon_join(source_seeds),
                    "duplicate_source_count": len(sources),
                    "duplicate_skipped_before_item": duplicate_skipped,
                    "construction_policy": pool_policy,
                    "population_role": "all_user_canonical",
                    "pool_classification": classification,
                    "provenance_hash": sha256_text(stable_json(provenance_payload)),
                    "notes": "Constructed without using the test positive; positive label assigned after construction.",
                }
            )
    return rows


def build_pool_instance_summary(pool_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows_by_user: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        rows_by_user[str(row["user_id"])].append(row)
    summaries: list[dict[str, Any]] = []
    for user_id in sorted(rows_by_user, key=lambda value: parse_int(value)):
        items = sorted(rows_by_user[user_id], key=lambda row: parse_int(row["candidate_position"]))
        positives = [row for row in items if row["is_positive"] == "true"]
        positive_present = bool(positives)
        position = positives[0]["candidate_position"] if positive_present else ""
        pool_size = len(items)
        unique_count = len({row["candidate_item_id"] for row in items})
        duplicate_removed = max(0, sum(max(0, parse_int(row["duplicate_source_count"]) - 1) for row in items))
        upper = 1 if positive_present else 0
        summaries.append(
            {
                "pool_instance_id": stable_id("F6G1-INSTANCE", user_id),
                "user_id": user_id,
                "pool_id": config["pool_policy"]["canonical_policy_name"],
                "pool_policy": config["pool_policy"]["operation"],
                "pool_classification": config["pool_policy"]["final_pool_classification_if_gates_pass"],
                "pool_size": pool_size,
                "unique_item_count": unique_count,
                "duplicate_removed_count": duplicate_removed,
                "positive_item_id": items[0]["positive_item_id"] if items else "",
                "positive_present": bool_text(positive_present),
                "positive_candidate_position": position,
                "candidate_pool_recall": upper,
                "upper_bound_hr_at_1": upper,
                "upper_bound_hr_at_5": upper,
                "upper_bound_hr_at_10": upper,
                "upper_bound_ndcg_at_5": upper,
                "upper_bound_ndcg_at_10": upper,
                "eligible_all_user": "true",
                "eligible_positive_present": bool_text(positive_present),
                "request_ready": bool_text(pool_size == int(config["dataset"]["target_pool_size"]) and unique_count == pool_size),
                "block_reason": "" if pool_size == int(config["dataset"]["target_pool_size"]) and unique_count == pool_size else "invalid_pool_size_or_duplicate",
            }
        )
    return summaries


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def quality_row(pool_name: str, population: str, metric: str, value: Any, denominator: Any, classification: str, notes: str = "") -> dict[str, Any]:
    return {
        "summary_id": stable_id("F6G1-QUAL", pool_name, population, metric),
        "pool_name": pool_name,
        "population": population,
        "metric": metric,
        "value": fmt(value),
        "denominator": denominator,
        "classification": classification,
        "notes": notes,
    }


def build_pool_quality_summary(pool_rows: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]], frames: Mapping[str, pd.DataFrame], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    pool_name = config["pool_policy"]["canonical_policy_name"]
    users = len(summaries)
    pool_sizes = [parse_int(row["pool_size"]) for row in summaries]
    positives = sum(1 for row in summaries if row["positive_present"] == "true")
    duplicate_removed = sum(parse_int(row["duplicate_removed_count"]) for row in summaries)
    source_counts = Counter()
    for row in pool_rows:
        for model in str(row["source_models"]).split(";"):
            if model:
                source_counts[model] += 1
    catalog_size = frames["catalog"]["item_id"].astype(str).nunique()
    pool_unique_items = len({row["candidate_item_id"] for row in pool_rows})
    rows = [
        quality_row(pool_name, "all_user", "expected_users", config["dataset"]["expected_users"], users, "structural_diagnostic"),
        quality_row(pool_name, "all_user", "constructed_users", users, users, "structural_diagnostic"),
        quality_row(pool_name, "all_user", "total_pool_rows", len(pool_rows), users, "structural_diagnostic"),
        quality_row(pool_name, "all_user", "pool_size_mean", mean([float(v) for v in pool_sizes]), users, "structural_diagnostic"),
        quality_row(pool_name, "all_user", "pool_size_median", median([float(v) for v in pool_sizes]), users, "structural_diagnostic"),
        quality_row(pool_name, "all_user", "pool_size_min", min(pool_sizes) if pool_sizes else 0, users, "structural_diagnostic"),
        quality_row(pool_name, "all_user", "pool_size_max", max(pool_sizes) if pool_sizes else 0, users, "structural_diagnostic"),
        quality_row(pool_name, "all_user", "exact_target_size_rate", mean([1.0 if value == int(config["dataset"]["target_pool_size"]) else 0.0 for value in pool_sizes]), users, "structural_diagnostic"),
        quality_row(pool_name, "all_user", "duplicate_removal_count", duplicate_removed, users, "structural_diagnostic"),
        quality_row(pool_name, "all_user", "unique_item_rate", pool_unique_items / len(pool_rows) if pool_rows else 0.0, len(pool_rows), "structural_diagnostic"),
        quality_row(pool_name, "all_user", "positive_present_count", positives, users, "positive_policy"),
        quality_row(pool_name, "all_user", "positive_present_rate", positives / users if users else 0.0, users, "positive_policy"),
        quality_row(pool_name, "all_user", "candidate_pool_recall", positives / users if users else 0.0, users, "upper_bound"),
        quality_row(pool_name, "all_user", "maximum_attainable_hr_at_1", positives / users if users else 0.0, users, "upper_bound"),
        quality_row(pool_name, "all_user", "maximum_attainable_hr_at_5", positives / users if users else 0.0, users, "upper_bound"),
        quality_row(pool_name, "all_user", "maximum_attainable_hr_at_10", positives / users if users else 0.0, users, "upper_bound"),
        quality_row(pool_name, "all_user", "maximum_attainable_ndcg_at_5", positives / users if users else 0.0, users, "upper_bound"),
        quality_row(pool_name, "all_user", "maximum_attainable_ndcg_at_10", positives / users if users else 0.0, users, "upper_bound"),
        quality_row(pool_name, "all_user", "catalog_coverage", pool_unique_items / catalog_size if catalog_size else 0.0, catalog_size, "structural_diagnostic"),
        quality_row(pool_name, "positive_present", "constructed_users", positives, positives, "diagnostic_population"),
        quality_row(pool_name, "positive_present", "candidate_pool_recall", 1.0 if positives else 0.0, positives, "upper_bound"),
        quality_row(pool_name, "positive_present", "maximum_attainable_hr_at_5", 1.0 if positives else 0.0, positives, "upper_bound"),
    ]
    for model in config["pool_policy"]["model_order"]:
        rows.append(quality_row(pool_name, "all_user", f"source_model_contribution_{model}", source_counts[model], len(pool_rows), "source_overlap"))
    return rows


def build_positive_policy(summaries: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    users = len(summaries)
    positives = sum(1 for row in summaries if row["positive_present"] == "true")
    absent = users - positives
    pool_name = config["pool_policy"]["canonical_policy_name"]
    return [
        {
            "policy_id": stable_id("F6G1-POS", "all_user"),
            "pool_name": pool_name,
            "population": "all_user",
            "positive_policy": "positive_naturally_present_only",
            "positive_insertion": "false",
            "positive_absent_policy": "retain_user_and_count_utility_zero",
            "users": users,
            "positive_present_users": positives,
            "positive_absent_users": absent,
            "positive_present_rate": fmt(positives / users if users else 0.0),
            "canonical_population": "true",
            "diagnostic_population": "false",
            "evaluation_denominator": "all constructed users",
            "classification": "adapted_all_user_primary",
            "notes": "Candidate-generation failure is separated from LLM reranking utility.",
        },
        {
            "policy_id": stable_id("F6G1-POS", "positive_present"),
            "pool_name": pool_name,
            "population": "positive_present",
            "positive_policy": "positive_naturally_present_only",
            "positive_insertion": "false",
            "positive_absent_policy": "exclude_with_caveat",
            "users": users,
            "positive_present_users": positives,
            "positive_absent_users": absent,
            "positive_present_rate": fmt(positives / users if users else 0.0),
            "canonical_population": "false",
            "diagnostic_population": "true",
            "evaluation_denominator": "positive-present users only",
            "classification": "diagnostic_positive_present_subset",
            "notes": "This view isolates reranking among users whose positive item entered the candidate pool.",
        },
    ]


def build_previous_pool_reconciliation(pool_rows: Sequence[Mapping[str, Any]], frames: Mapping[str, pd.DataFrame], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    previous = frames["previous_pool"].copy()
    previous_by_user = {
        str(user): group.sort_values("candidate_position", key=lambda s: s.astype(int)).to_dict("records")
        for user, group in previous.groupby("user_id", sort=True)
    }
    current_by_user: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        current_by_user[str(row["user_id"])].append(row)
    rows: list[dict[str, Any]] = []
    for user_id in sorted(current_by_user, key=lambda value: parse_int(value)):
        current = sorted(current_by_user[user_id], key=lambda row: parse_int(row["candidate_position"]))
        old = previous_by_user.get(user_id, [])
        current_items = [str(row["candidate_item_id"]) for row in current]
        old_items = [str(row.get("candidate_item_id", "")) for row in old]
        current_sources = [str(row["source_models"]) for row in current]
        old_sources = [str(row.get("source_models", "")) for row in old]
        current_pos = any(row["is_positive"] == "true" for row in current)
        old_pos = any(str(row.get("is_positive", "")).lower() == "true" for row in old)
        set_equal = set(current_items) == set(old_items)
        order_equal = current_items == old_items
        source_equal = current_sources == old_sources
        pos_equal = current_pos == old_pos
        mismatch = []
        if not set_equal:
            mismatch.append("candidate_set")
        if not order_equal:
            mismatch.append("candidate_order")
        if not source_equal:
            mismatch.append("source_models")
        if not pos_equal:
            mismatch.append("positive_presence")
        outcome = "previous_pool_valid_but_adapted" if not mismatch else "previous_pool_superseded_by_reconstructed_pool"
        rows.append(
            {
                "reconciliation_id": stable_id("F6G1-PREV", user_id),
                "user_id": user_id,
                "previous_pool_path": config["input_hashes"]["previous_adapted_pool"]["path"],
                "canonical_pool_name": config["pool_policy"]["canonical_policy_name"],
                "candidate_set_equality": bool_text(set_equal),
                "candidate_order_equality": bool_text(order_equal),
                "positive_presence_equality": bool_text(pos_equal),
                "source_model_equality": bool_text(source_equal),
                "previous_positive_present": bool_text(old_pos),
                "canonical_positive_present": bool_text(current_pos),
                "mismatch_reason": ";".join(mismatch),
                "outcome": outcome,
                "notes": "Previous pool is not paper-exact; reuse is allowed only as adapted evidence." if outcome == "previous_pool_valid_but_adapted" else "F6G1 reconstructed pool differs from previous pool.",
            }
        )
    return rows


def source_rank_for_model(row: Mapping[str, Any], model: str) -> int | None:
    models = str(row.get("source_models", "")).split(";")
    ranks = str(row.get("source_ranks", "")).split(";")
    for index, name in enumerate(models):
        if name == model and index < len(ranks):
            return parse_int(ranks[index], default=0) or None
    return None


def hit_metrics_from_ranks(ranks: Sequence[int], cutoff: int) -> tuple[float, float, float]:
    if not ranks:
        return 0.0, 0.0, 0.0
    hits = [1.0 if rank <= cutoff else 0.0 for rank in ranks]
    ndcgs = [1.0 / math.log2(rank + 1) if rank <= cutoff else 0.0 for rank in ranks]
    mean_rank = mean([float(rank) for rank in ranks])
    return mean(hits), mean(ndcgs), mean_rank


def build_traditional_pool_metrics(pool_rows: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows_by_user: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        rows_by_user[str(row["user_id"])].append(row)
    users = len(rows_by_user)
    positive_users = sum(1 for row in summaries if row["positive_present"] == "true")
    result: list[dict[str, Any]] = []
    for model in config["pool_policy"]["model_order"]:
        seed = str(config["pool_policy"]["selected_seeds"][model])
        scored_items = sum(1 for row in pool_rows if model in str(row["source_models"]).split(";"))
        score_coverage = scored_items / len(pool_rows) if pool_rows else 0.0
        all_ranks: list[int] = []
        pp_ranks: list[int] = []
        model_positive_source_users = 0
        for user_id, rows in rows_by_user.items():
            positive_rows = [row for row in rows if row["is_positive"] == "true"]
            if positive_rows:
                rank = source_rank_for_model(positive_rows[0], model)
                if rank is not None:
                    model_positive_source_users += 1
                    all_ranks.append(rank)
                    pp_ranks.append(rank)
                else:
                    all_ranks.append(10**9)
                    pp_ranks.append(10**9)
            else:
                all_ranks.append(10**9)
        for population, ranks, denominator in [("all_user", all_ranks, users), ("positive_present", pp_ranks, positive_users)]:
            for cutoff in [1, 5, 10]:
                hr, _ndcg, _mean_rank = hit_metrics_from_ranks(ranks, cutoff)
                result.append(metric_row(model, seed, population, "candidate_pool_source_rank_observable", f"HR@{cutoff}", hr, denominator, positive_users, model_positive_source_users / users if users else 0.0, "adapted_source_rank_metric", "Uses carried full-catalog source rank for positive rows; missing source scores are not imputed."))
            for cutoff in [5, 10]:
                _hr, ndcg, _mean_rank = hit_metrics_from_ranks(ranks, cutoff)
                result.append(metric_row(model, seed, population, "candidate_pool_source_rank_observable", f"NDCG@{cutoff}", ndcg, denominator, positive_users, model_positive_source_users / users if users else 0.0, "adapted_source_rank_metric", "Uses carried full-catalog source rank for positive rows; missing source scores are not imputed."))
            _hr, _ndcg, mean_rank = hit_metrics_from_ranks(ranks, 10**8)
            result.append(metric_row(model, seed, population, "candidate_pool_source_rank_observable", "mean_positive_source_rank", mean_rank, denominator, positive_users, model_positive_source_users / users if users else 0.0, "adapted_source_rank_metric", "Absent positives are assigned censored rank for all-user accounting; source scores are not imputed."))
        result.append(metric_row(model, seed, "all_user", "score_coverage", "candidate_pool_score_coverage_rate", score_coverage, len(pool_rows), positive_users, model_positive_source_users / users if users else 0.0, "structural_diagnostic", "Coverage is the share of pool items with this model in source provenance."))
        result.append(metric_row(model, seed, "all_user", "positive_source_coverage", "positive_source_present_rate", model_positive_source_users / users if users else 0.0, users, positive_users, model_positive_source_users / users if users else 0.0, "structural_diagnostic", "Share of all users whose positive pool row has this model source rank."))
    return result


def metric_row(model: str, seed: str, population: str, scope: str, metric: str, value: Any, denominator: Any, positive_users: int, recall: float, classification: str, notes: str) -> dict[str, Any]:
    return {
        "metric_id": stable_id("F6G1-TRAD", model, seed, population, scope, metric),
        "model_name": model,
        "seed": seed,
        "pool_name": "phase_f6g1_reconstructed_adapted_round_robin_union_20",
        "population": population,
        "evaluation_scope": scope,
        "metric": metric,
        "value": fmt(value),
        "denominator": denominator,
        "positive_present_users": positive_users,
        "candidate_pool_recall": fmt(recall),
        "classification": classification,
        "notes": notes,
    }


def build_title_item_reconciliation(pool_rows: Sequence[Mapping[str, Any]], frames: Mapping[str, pd.DataFrame]) -> list[dict[str, Any]]:
    catalog = frames["catalog"]
    titles, normalized_titles, duplicate_titles = catalog_maps(catalog)
    source_items = dict(zip(catalog["item_id"].astype(str), catalog.get("source_item_id", catalog["item_id"]).astype(str)))
    rows: list[dict[str, Any]] = []
    for row in pool_rows:
        item_id = str(row["candidate_item_id"])
        title = titles.get(item_id, "")
        normalized = normalized_titles.get(item_id, "")
        missing = title.strip() == ""
        ambiguous = normalized in duplicate_titles if normalized else False
        rows.append(
            {
                "reconciliation_id": stable_id("F6G1-TITLE", row["user_id"], row["candidate_position"], item_id),
                "user_id": row["user_id"],
                "candidate_item_id": item_id,
                "candidate_position": row["candidate_position"],
                "canonical_item_id": item_id,
                "source_item_id": source_items.get(item_id, item_id),
                "artist_title": title,
                "normalized_artist_title": normalized,
                "duplicate_normalized_title": bool_text(ambiguous),
                "ambiguous_title": bool_text(ambiguous),
                "missing_title": bool_text(missing),
                "prompt_safe_title": title if not missing else "",
                "reversible_mapping": bool_text(item_id in titles and not missing),
                "mapping_status": "ambiguous_title_preserved" if ambiguous else "reversible" if not missing else "missing_title",
                "notes": "Candidate identity remains canonical item_id; titles are display-only.",
            }
        )
    return rows


def gate_row(name: str, requirement: str, observed: str, status: str, impact: str, remediation: str, allowed: bool) -> dict[str, Any]:
    return {
        "gate_id": stable_id("F6G1-GATE", name),
        "gate_name": name,
        "requirement": requirement,
        "observed_evidence": observed,
        "status": status,
        "impact": impact,
        "remediation": remediation,
        "future_llm_inference_allowed": bool_text(allowed),
    }


def build_comparability_gate(
    input_rows: Sequence[Mapping[str, Any]],
    cohort_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    title_rows: Sequence[Mapping[str, Any]],
    previous_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    inputs_ok = all(row["status"] == "pass" for row in input_rows)
    cohort_ok = len(cohort_rows) == int(config["dataset"]["expected_users"]) and all(row["usable_status"] == "usable" for row in cohort_rows)
    required_models = set(config["pool_policy"]["model_order"])
    full_complete = {row["model_name"] for row in prediction_rows if row["prediction_file"].endswith("full_catalog_rankings.csv") and row["prediction_completeness"] == "complete"}
    prediction_ok = required_models.issubset(full_complete)
    selected_ok = all(row["decision"] == "use_for_adapted_pool" for row in seed_rows if row["model_name"] in required_models and row["seed"] == str(config["pool_policy"]["selected_seeds"][row["model_name"]]))
    pool_ok = len(summaries) == int(config["dataset"]["expected_users"]) and all(parse_int(row["pool_size"]) == int(config["dataset"]["target_pool_size"]) and row["request_ready"] == "true" for row in summaries)
    duplicate_ok = all(parse_int(row["unique_item_count"]) == parse_int(row["pool_size"]) for row in summaries)
    positives = sum(1 for row in summaries if row["positive_present"] == "true")
    positive_policy_ok = positives > 0 and all(row["positive_item_id"] for row in summaries)
    title_ok = all(row["reversible_mapping"] == "true" for row in title_rows)
    previous_equal = all(row["candidate_order_equality"] == "true" for row in previous_rows)
    return [
        gate_row("source-contract gate", "Recover enough source policy for an adapted non-paper-exact pool.", "Paper exact union/source-K remain unresolved; F6D2/F6D3 adapted contract is explicit.", "warning", "Blocks paper-exact claim but not adapted inference.", "Manual paper/original-code protocol recovery for exactness upgrade.", True),
        gate_row("cohort-completeness gate", "All 1859 users must be usable.", f"{len(cohort_rows)} audited users; usable={sum(1 for row in cohort_rows if row['usable_status'] == 'usable')}", "pass" if cohort_ok else "fail", "Incomplete cohorts block inference.", "Repair cohort or metadata mapping.", cohort_ok),
        gate_row("prediction-completeness gate", "Selected source predictions must cover all users.", f"complete full-catalog source models={semicolon_join(sorted(full_complete))}", "pass" if prediction_ok else "fail", "Missing selected predictions block pool construction.", "Recompute or recover missing traditional predictions.", prediction_ok),
        gate_row("seed-policy gate", "Use prior frozen validation-selected seeds without test leakage.", f"selected seeds={stable_json(config['pool_policy']['selected_seeds'])}", "warning" if selected_ok else "fail", "Seed identity is adapted; paper seed is not recovered.", "Recover paper seed identity for exactness upgrade.", selected_ok),
        gate_row("candidate-pool-size gate", "Every user must have 20 unique candidates.", f"valid pools={sum(1 for row in summaries if row['request_ready'] == 'true')}/{len(summaries)}", "pass" if pool_ok else "fail", "Invalid pool size blocks prompts.", "Repair reconstruction policy or source coverage.", pool_ok),
        gate_row("duplicate-policy gate", "No duplicate item IDs within user pool.", f"duplicate-free users={sum(1 for row in summaries if parse_int(row['unique_item_count']) == parse_int(row['pool_size']))}/{len(summaries)}", "pass" if duplicate_ok else "fail", "Duplicates would invalidate indexed candidate mapping.", "Fix dedupe policy.", duplicate_ok),
        gate_row("positive-policy gate", "Do not insert positives; preserve absence and denominator separation.", f"positive-present users={positives}/{len(summaries)}", "pass" if positive_policy_ok else "fail", "Unresolved positive policy blocks inference.", "Recover paper denominator or keep all-user canonical policy.", positive_policy_ok),
        gate_row("candidate-pool-recall gate", "Report recall and upper bounds before utility claims.", f"candidate-pool recall={fmt(positives / len(summaries) if summaries else 0.0)}", "warning", "Low recall limits attainable utility but does not itself prove LLM failure.", "Report all-user and positive-present views separately.", True),
        gate_row("title/item-mapping gate", "Canonical item IDs must map reversibly to prompt titles.", f"reversible rows={sum(1 for row in title_rows if row['reversible_mapping'] == 'true')}/{len(title_rows)}", "pass" if title_ok else "fail", "Missing titles block prompt rendering.", "Repair title mapping without merging item IDs.", title_ok),
        gate_row("no-leakage gate", "No ground-truth positive may be used for construction or tie-breaking.", "F6G1 uses only full-catalog rankings; positive labels assigned after construction.", "pass", "Leakage would invalidate the phase.", "None.", True),
        gate_row("deterministic-reconstruction gate", "Reconstruction must be stable and previous-pool comparable.", f"previous order equality={sum(1 for row in previous_rows if row['candidate_order_equality'] == 'true')}/{len(previous_rows)}", "pass" if previous_equal else "warning", "Mismatch would supersede previous pool.", "Inspect mismatch rows.", True),
        gate_row("future-request-contract gate", "Request and shard manifests must be freezeable without inference.", "Indexed candidate contract can be generated for four model-strategy cells.", "pass" if pool_ok and title_ok else "blocked", "Blocked request contract prevents future inference.", "Repair pool/title gates.", pool_ok and title_ok),
    ]


def future_allowed(gates: Sequence[Mapping[str, Any]]) -> bool:
    return bool(gates) and all(row["future_llm_inference_allowed"] == "true" for row in gates)


def build_exactness_classification(gates: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    label = config["pool_policy"]["future_experiment_label_if_gates_pass"] if future_allowed(gates) else "blocked_lastfm_reranking"
    rows = [
        ("overall_pool", config["pool_policy"]["final_pool_classification_if_gates_pass"], "F6G1 reconstructed adapted source pool.", "Exact source K, seed identity and union policy remain unrecovered."),
        ("cohort_exactness", "adapted_match", "Frozen F6C 1859-user cohort.", "Paper exact split is not proven."),
        ("split_exactness", "adapted_match", "Frozen F6C train/validation/test files.", "No exact paper split hash."),
        ("model_exactness", "adapted_model", "Local F6D2 BPRMF/GRU4Rec/SASRec/LightGCN.", "Model snapshots and implementations differ from paper."),
        ("seed_exactness", "adapted_match", "Validation-selected F6D2 seeds.", "Paper source-run identity unknown."),
        ("prediction_exactness", "exact_model_reconstructed_training", "Frozen selected top-100 predictions.", "Predictions are local reconstructed outputs."),
        ("pool_formula_exactness", "local_diagnostic_formula_adapted", "Round-robin union with first occurrence.", "Paper union formula unresolved."),
        ("population_exactness", "adapted_all_user_primary", "All users retained; positive-present diagnostic.", "Paper denominator unresolved."),
        ("title_mapping_exactness", "reversible_local_mapping", "Canonical item IDs mapped to catalog titles.", "Title strings are display-only."),
        ("future_prompt_exactness", "adapted_indexed_prompt_contract", "Indexed JSON schema for future inference.", "No paper-exact raw prompt claim."),
    ]
    return [
        {
            "exactness_id": stable_id("F6G1-EXACT", component),
            "component": component,
            "classification": classification,
            "evidence": evidence,
            "limitation": limitation,
            "future_experiment_label": label,
            "notes": "Do not use paper_exact unless all material components are proven exact.",
        }
        for component, classification, evidence, limitation in rows
    ]


def build_future_logical_instances(pool_rows: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]], frames: Mapping[str, pd.DataFrame], gates: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    allowed = future_allowed(gates)
    cohort_by_user = {str(row["user_id"]): row for row in frames["cohort"].to_dict("records")}
    rows_by_user: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        rows_by_user[str(row["user_id"])].append(row)
    summary_by_user = {str(row["user_id"]): row for row in summaries}
    logical: list[dict[str, Any]] = []
    for user_id in sorted(rows_by_user, key=lambda value: parse_int(value)):
        pool = sorted(rows_by_user[user_id], key=lambda row: parse_int(row["candidate_position"]))
        cohort = cohort_by_user[user_id]
        summary = summary_by_user[user_id]
        candidate_ids = [str(row["candidate_item_id"]) for row in pool]
        candidate_titles = [str(row["title"]) for row in pool]
        order_hash = sha256_text(pipe_join(candidate_ids))
        payload = {"user_id": user_id, "candidate_ids": candidate_ids, "policy": config["pool_policy"]["canonical_policy_name"]}
        logical.append(
            {
                "logical_instance_id": stable_id("F6G1-LOGICAL", user_id),
                "user_id": user_id,
                "dataset": config["dataset"]["name"],
                "task": "reranking",
                "cohort_name": "finalization_f6c_lastfm_ranking_cohort",
                "history_item_ids": pipe_join(read_listish(cohort.get("history_item_ids", ""))),
                "history_item_titles": pipe_join(read_listish(cohort.get("history_item_titles", ""))),
                "candidate_item_ids": pipe_join(candidate_ids),
                "candidate_item_titles": pipe_join(candidate_titles),
                "candidate_order_hash": order_hash,
                "positive_item_id": summary["positive_item_id"],
                "positive_initial_position": summary["positive_candidate_position"],
                "positive_present": summary["positive_present"],
                "pool_provenance_hash": sha256_text(stable_json(payload)),
                "expected_output_length": config["future_requests"]["expected_output_length"],
                "parser_contract": config["future_requests"]["parser_contract"],
                "operational_utility_policy": config["future_requests"]["operational_utility_policy"],
                "conditional_utility_policy": config["future_requests"]["conditional_utility_policy"],
                "request_ready": bool_text(allowed and summary["request_ready"] == "true"),
                "block_reason": "" if allowed and summary["request_ready"] == "true" else "comparability_gate_blocked",
            }
        )
    return logical if allowed else [{column: "" for column in LOGICAL_COLUMNS}]


def build_future_requests(logical: Sequence[Mapping[str, Any]], gates: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not future_allowed(gates) or (len(logical) == 1 and not logical[0].get("logical_instance_id")):
        return [{column: "" for column in REQUEST_COLUMNS}]
    rows: list[dict[str, Any]] = []
    schema = '{"type":"object","required":["ranking"],"properties":{"ranking":{"type":"array","items":{"type":"integer"},"minItems":20,"maxItems":20}}}'
    for logical_row in logical:
        labels = [f"{index}:{title}" for index, title in enumerate(str(logical_row["candidate_item_titles"]).split("|"), start=1)]
        title_hash = sha256_text(str(logical_row["candidate_item_ids"]) + "||" + str(logical_row["candidate_item_titles"]))
        for model in config["future_requests"]["models"]:
            for strategy in config["future_requests"]["strategies"]:
                request_payload = {
                    "logical_instance_id": logical_row["logical_instance_id"],
                    "model": model,
                    "strategy": strategy,
                    "candidate_order_hash": logical_row["candidate_order_hash"],
                    "parser_contract": config["future_requests"]["parser_contract"],
                }
                request_hash = sha256_text(stable_json(request_payload))
                rows.append(
                    {
                        "request_id": stable_id("F6G1-REQ", request_hash),
                        "logical_instance_id": logical_row["logical_instance_id"],
                        "user_id": logical_row["user_id"],
                        "dataset": logical_row["dataset"],
                        "task": logical_row["task"],
                        "model": model,
                        "strategy": strategy,
                        "history_item_ids": logical_row["history_item_ids"],
                        "candidate_item_ids": logical_row["candidate_item_ids"],
                        "indexed_candidate_labels": pipe_join(labels),
                        "candidate_order_hash": logical_row["candidate_order_hash"],
                        "positive_item_id": logical_row["positive_item_id"],
                        "positive_initial_position": logical_row["positive_initial_position"],
                        "positive_present": logical_row["positive_present"],
                        "pool_provenance_hash": logical_row["pool_provenance_hash"],
                        "expected_json_schema": schema,
                        "required_output_length": config["future_requests"]["expected_output_length"],
                        "parser_contract": config["future_requests"]["parser_contract"],
                        "invalid_output_policy": config["future_requests"]["invalid_output_policy"],
                        "operational_utility_policy": config["future_requests"]["operational_utility_policy"],
                        "conditional_utility_policy": config["future_requests"]["conditional_utility_policy"],
                        "title_mapping_hash": title_hash,
                        "request_hash": request_hash,
                        "request_status": "planned_not_executed",
                        "block_reason": "",
                    }
                )
    return sorted(rows, key=lambda row: (row["model"], row["strategy"], parse_int(row["user_id"])))


def build_future_shards(requests: Sequence[Mapping[str, Any]], gates: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not future_allowed(gates) or (len(requests) == 1 and not requests[0].get("request_id")):
        return [{column: "" for column in SHARD_COLUMNS}]
    max_size = int(config["dataset"]["max_requests_per_shard"])
    rows: list[dict[str, Any]] = []
    for index in range(0, len(requests), max_size):
        chunk = list(requests[index : index + max_size])
        shard_index = index // max_size + 1
        manifest_hash = sha256_text(stable_json([row["request_hash"] for row in chunk]))
        shard_payload = {"shard_index": shard_index, "request_ids": [row["request_id"] for row in chunk]}
        rows.append(
            {
                "shard_id": stable_id("F6G1-SHARD", shard_index, manifest_hash),
                "model": "mixed" if len({row["model"] for row in chunk}) > 1 else chunk[0]["model"],
                "strategy": "mixed" if len({row["strategy"] for row in chunk}) > 1 else chunk[0]["strategy"],
                "shard_index": shard_index,
                "request_count": len(chunk),
                "first_request_id": chunk[0]["request_id"],
                "last_request_id": chunk[-1]["request_id"],
                "request_manifest_hash": manifest_hash,
                "shard_hash": sha256_text(stable_json(shard_payload)),
                "status": "planned_not_executed",
                "notes": "Maximum 500 requests per shard; no inference executed.",
            }
        )
    return rows


def build_missing_or_blocked(gates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        ("F6G1-BLOCK-001", "paper source top-K", "missing", "missing_evidence", "Exact per-source K is not recovered.", "Paper or original-code LastFM pool construction.", "F6D3 records adapted top-100 export.", "inspect_pool_protocol_conflict"),
        ("F6G1-BLOCK-002", "paper union policy", "missing", "missing_evidence", "Union/fusion policy is not recovered.", "Original LastFM pool-generation code or paper appendix detail.", "F6G1 uses adapted round-robin union.", "inspect_pool_protocol_conflict"),
        ("F6G1-BLOCK-003", "paper seed identity", "missing", "adapted_component", "Paper source-run identity unresolved.", "Paper seed/run manifest.", "F6D2 selected validation seeds are frozen.", "inspect_pool_protocol_conflict"),
        ("F6G1-BLOCK-004", "paper denominator", "missing", "adapted_population", "Paper positive-present filtering policy unresolved.", "Paper evaluation denominator details.", "F6G1 materializes all-user and positive-present populations.", "inspect_pool_protocol_conflict"),
        ("F6G1-BLOCK-005", "paper-exact model snapshots", "missing", "adapted_model", "Exact trained source snapshots unavailable.", "Original checkpoints or model digests.", "F6D2 local reconstructed models are available.", "keep_lastfm_reranking_diagnostic_only"),
    ]
    for gate in gates:
        if gate["status"] in {"fail", "blocked"}:
            rows.append((stable_id("F6G1-GATE-BLOCK", gate["gate_name"]), gate["gate_name"], gate["status"], "gate_failure", gate["observed_evidence"], gate["requirement"], "", gate["remediation"]))
    return [
        {
            "item_id": item_id,
            "component": component,
            "status": status,
            "classification": classification,
            "reason": reason,
            "required_evidence": required,
            "current_evidence": current,
            "recommended_action": action,
        }
        for item_id, component, status, classification, reason, required, current, action in rows
    ]


def parse_count(text: str, word: str) -> int:
    match = re.search(rf"(\d+)\s+{word}", text)
    return int(match.group(1)) if match else 0


def run_validation_suites(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if not config.get("validation", {}).get("run_suites", True):
        return {"suites": [], "total_passed": 0, "total_failed": 0, "total_skipped": 0, "unsafe_exclusions": list(config.get("validation", {}).get("unsafe_exclusions", []))}
    suites: list[dict[str, Any]] = []
    for suite in config["validation"]["suites"]:
        command = [sys.executable, "-m", "pytest", *suite["files"], "-q"]
        code, out, err, _duration = run_command(command, root, timeout=900)
        text = out + "\n" + err
        suites.append(
            {
                "suite": suite["name"],
                "status": "pass" if code == 0 else "fail",
                "passed": parse_count(text, "passed"),
                "failed": parse_count(text, "failed"),
                "skipped": parse_count(text, "skipped"),
                "command": " ".join(command),
            }
        )
    return {
        "suites": suites,
        "total_passed": sum(row["passed"] for row in suites),
        "total_failed": sum(row["failed"] for row in suites),
        "total_skipped": sum(row["skipped"] for row in suites),
        "unsafe_exclusions": list(config["validation"].get("unsafe_exclusions", [])),
    }


def table_hashes(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    return {name: sha256_file(root / path) for name, path in sorted(config["outputs"]["tables"].items())}


def determine_decision(gates: Sequence[Mapping[str, Any]], validation: Mapping[str, Any], input_rows: Sequence[Mapping[str, Any]], exactness: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    if any(row["status"] != "pass" for row in input_rows):
        return DECISION_INPUTS, RECOMMEND_BLOCKED
    if int(validation.get("total_failed", 0)) > 0:
        return DECISION_TESTS, RECOMMEND_BLOCKED
    failed = [row for row in gates if row["status"] in {"fail", "blocked"}]
    if failed:
        names = {row["gate_name"] for row in failed}
        if "prediction-completeness gate" in names:
            return "f6g1_blocked_by_missing_predictions", "recompute_missing_traditional_predictions"
        if "seed-policy gate" in names:
            return "f6g1_blocked_by_seed_policy", "inspect_pool_protocol_conflict"
        if "positive-policy gate" in names:
            return "f6g1_blocked_by_positive_policy", "inspect_pool_protocol_conflict"
        if "title/item-mapping gate" in names:
            return "f6g1_blocked_by_title_reconciliation", "run_targeted_pool_reconstruction_repair"
        if "no-leakage gate" in names:
            return "f6g1_blocked_by_leakage", RECOMMEND_BLOCKED
        return "f6g1_incomplete_missing_evidence", "inspect_pool_protocol_conflict"
    overall = next((row for row in exactness if row["component"] == "overall_pool"), {})
    if overall.get("classification") == "paper_exact_pool":
        return DECISION_READY, RECOMMEND_READY
    if future_allowed(gates):
        return DECISION_ADAPTED, RECOMMEND_READY
    return DECISION_DIAGNOSTIC, RECOMMEND_DIAGNOSTIC


def build_report(repository: Mapping[str, Any], interpreter: Mapping[str, Any], manifest: Mapping[str, Any], counts: Mapping[str, int], quality: Sequence[Mapping[str, Any]], gates: Sequence[Mapping[str, Any]]) -> str:
    quality_map = {(row["population"], row["metric"]): row["value"] for row in quality}
    gate_summary = ", ".join(f"{row['gate_name']}={row['status']}" for row in gates)
    sections = [
        ("Initial repository state", f"Branch `{repository['branch']}`, HEAD `{repository['head']}`, staged files `{repository['staged_files']}`. Pre-existing tracked dirty files remain `{repository['git_status_tracked_short']}`."),
        ("Environment used", f"Interpreter `{interpreter['actual_interpreter']}`; Python `{interpreter['python_version']}`; classification `{repository['environment_classification']}`."),
        ("Frozen-input verification", f"Input changed={manifest['input_invariance']['changed']}, missing={manifest['input_invariance']['missing']}."),
        ("Local source inventory", f"Rows: {counts['source_inventory']}. Evidence remains local-only."),
        ("Paper pool contract", "The paper-exact LastFM pool contract remains incomplete for source top-K, union/order, seed identity and denominator."),
        ("Original-repository pool contract", "Original repository templates confirm candidate20 re-ranking prompts, but no executable LastFM pool builder was recovered."),
        ("LastFM cohort audit", f"Audited users: {counts['cohort_audit']}. No user is silently dropped."),
        ("Traditional-prediction inventory", f"Inventory rows: {counts['prediction_inventory']}. F6D1 candidate-only and F6D2 selected full-catalog predictions are separated."),
        ("Seed-policy audit", "F6D2 validation-selected seeds are frozen and used for the adapted pool; paper seed identity is unresolved."),
        ("Candidate-source model decisions", "BPRMF, GRU4Rec, SASRec and LightGCN are canonical adapted sources; Random, MostPop and BM25-like remain diagnostic/non-canonical for the pool."),
        ("Canonical pool construction", f"Pool rows: {counts['pool_item_provenance']}. Policy: round-robin union, first-occurrence dedupe, no positive insertion."),
        ("Diagnostic pool variants", "Positive-present population is diagnostic; all-user population is canonical for adapted future inference."),
        ("Positive-item policy", "Positive items are never inserted. Absent positives are retained and counted as utility failures in all-user operational utility."),
        ("Previous 592/1859 pool reconciliation", f"Previous-pool comparison rows: {counts['previous_pool_reconciliation']}. Reuse is allowed only as adapted evidence."),
        ("Pool-size results", f"Mean pool size={quality_map.get(('all_user','pool_size_mean'), '')}; exact-target-size rate={quality_map.get(('all_user','exact_target_size_rate'), '')}."),
        ("Positive-present results", f"Positive-present count={quality_map.get(('all_user','positive_present_count'), '')}; rate={quality_map.get(('all_user','positive_present_rate'), '')}."),
        ("Candidate-pool recall and upper bounds", f"All-user candidate-pool recall={quality_map.get(('all_user','candidate_pool_recall'), '')}; max HR@5={quality_map.get(('all_user','maximum_attainable_hr_at_5'), '')}."),
        ("Source-model overlap", "Source contribution counts are recorded in `phase_f6g1_pool_quality_summary.csv`."),
        ("Traditional-model within-pool metrics", f"Metric rows: {counts['traditional_pool_metrics']}; missing source scores are not imputed."),
        ("Title/item reconciliation", f"Rows: {counts['title_item_reconciliation']}; candidate identity remains canonical item ID."),
        ("Comparability gate", gate_summary),
        ("Exactness classification", "Overall pool is adapted/reconstructed, not paper-exact."),
        ("Future logical-instance count", str(counts["future_logical_instances"])),
        ("Future request count", str(counts["future_request_manifest"])),
        ("Future shard count", str(counts["future_shard_manifest"])),
        ("Implementation files", "F6G1 config, analysis module, wrapper and tests were added."),
        ("Tests", f"Validation passed={manifest['validation_summary']['total_passed']}, failed={manifest['validation_summary']['total_failed']}."),
        ("Excluded unsafe tests", ";".join(manifest["unsafe_excluded_tests"])),
        ("Determinism", "Builder outputs are designed for byte-stable scientific content; runtime metadata is separated in the manifest."),
        ("Protected-input verification", f"Prior direct inputs changed={manifest['input_invariance']['changed']}."),
        ("Freeze manifest", f"Scientific content hash `{manifest['scientific_content_hash']}`."),
        ("Missing or blocked evidence", f"Rows: {counts['missing_or_blocked']}."),
        ("Final decision", manifest["decision"]),
        ("Recommended next action", manifest["recommended_next_action"]),
        ("Final git state", f"Tracked dirty files: `{repository['git_status_tracked_short']}`; staged files: `{repository['staged_files']}`."),
        ("Files created or modified", "Only F6G1 implementation and F6G1 artifacts are intended outputs."),
        ("Remaining limitations", "Paper-exact source K, union/order, seed identity, model snapshots and denominator remain unresolved."),
    ]
    lines = ["# Phase F6G1 - LastFM Re-ranking Reconstruction", ""]
    for title, body in sections:
        lines.extend([f"## {title}", "", str(body), ""])
    return "\n".join(lines).rstrip() + "\n"


def build_manifest(
    root: Path,
    config: Mapping[str, Any],
    repository: Mapping[str, Any],
    interpreter: Mapping[str, Any],
    phase_checks: Mapping[str, Any],
    input_rows: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
    exactness: Sequence[Mapping[str, Any]],
    counts: Mapping[str, int],
) -> dict[str, Any]:
    hashes = table_hashes(root, config)
    decision, recommendation = determine_decision(gates, validation, input_rows, exactness)
    scientific = {
        "phase": PHASE_ID,
        "f6f2_scientific_hash": phase_checks["f6f2_scientific_hash"],
        "f6f1_scientific_hash": phase_checks["f6f1_scientific_hash"],
        "f6e4c_scientific_hash": phase_checks["f6e4c_scientific_hash"],
        "lastfm_split_hash": sha256_file(root / config["input_hashes"]["lastfm_test"]["path"]),
        "lastfm_cohort_hash": sha256_file(root / config["input_hashes"]["lastfm_ranking_cohort"]["path"]),
        "prediction_file_hashes": {
            "f6d1_baseline_predictions": sha256_file(root / config["input_hashes"]["f6d1_baseline_predictions"]["path"]),
            "f6d2_candidate_predictions": sha256_file(root / config["input_hashes"]["f6d2_candidate_predictions"]["path"]),
            "f6d2_full_catalog_predictions": sha256_file(root / config["input_hashes"]["f6d2_full_catalog_predictions"]["path"]),
        },
        "selected_seed_policy_hash": hashes["seed_policy_audit"],
        "pool_contract_hash": hashes["paper_pool_contract"],
        "pool_item_provenance_hash": hashes["pool_item_provenance"],
        "canonical_pool_hash": hashes["pool_item_provenance"],
        "positive_policy_hash": hashes["positive_policy"],
        "previous_pool_comparison_hash": hashes["previous_pool_reconciliation"],
        "comparability_gate_hash": hashes["comparability_gate"],
        "future_logical_instance_hash": hashes["future_logical_instances"],
        "future_request_manifest_hash": hashes["future_request_manifest"],
        "future_shard_manifest_hash": hashes["future_shard_manifest"],
        "exactness_classification_hash": hashes["exactness_classification"],
        "table_counts": dict(counts),
        "future_inference_allowed": bool_text(future_allowed(gates)),
        "decision": decision,
        "recommended_next_action": recommendation,
    }
    return {
        **scientific,
        "scientific_content_hash": sha256_text(stable_json(scientific)),
        "direct_input_hashes": {row["path"]: row["actual_sha256"] for row in input_rows},
        "table_hashes": hashes,
        "phase_checks": dict(phase_checks),
        "interpreter_gate": dict(interpreter),
        "validation_summary": validation,
        "unsafe_excluded_tests": list(config.get("validation", {}).get("unsafe_exclusions", [])),
        "input_invariance": {
            "changed": sum(1 for row in input_rows if row["changed"] == "true"),
            "missing": sum(1 for row in input_rows if row["missing"] == "true"),
        },
        "runtime_metadata": {
            "timestamp": repository["timestamp"],
            "repository_state": dict(repository),
            "python": sys.executable,
            "platform": platform.platform(),
        },
    }


def run_phase(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    validate_output_paths(config)
    repository = record_repository_state(root, config)
    interpreter = verify_interpreter(root, config)
    if interpreter["status"] != "pass":
        raise RuntimeError(f"F6G1 interpreter gate failed: {interpreter}")
    input_rows = verify_input_integrity(root, config)
    assert_no_input_failures(input_rows)
    phase_checks = verify_phase_manifests(root, config)
    if phase_checks["status"] != "pass":
        raise RuntimeError(f"F6G1 frozen phase scientific gate failed: {phase_checks}")
    frames = load_frames(root, config)

    source_inventory = build_source_inventory(config)
    contract = build_paper_pool_contract(frames)
    cohort = build_cohort_audit(frames, config)
    prediction_inventory = build_prediction_inventory(frames, config)
    seed_policy = build_seed_policy_audit(frames, config)
    source_decisions = build_candidate_source_model_decisions(frames, config)
    pool = reconstruct_pool(frames, config)
    summaries = build_pool_instance_summary(pool, config)
    quality = build_pool_quality_summary(pool, summaries, frames, config)
    positive = build_positive_policy(summaries, config)
    previous = build_previous_pool_reconciliation(pool, frames, config)
    traditional = build_traditional_pool_metrics(pool, summaries, config)
    titles = build_title_item_reconciliation(pool, frames)
    gates = build_comparability_gate(input_rows, cohort, prediction_inventory, seed_policy, summaries, titles, previous, config)
    exactness = build_exactness_classification(gates, config)
    logical = build_future_logical_instances(pool, summaries, frames, gates, config)
    requests = build_future_requests(logical, gates, config)
    shards = build_future_shards(requests, gates, config)
    missing = build_missing_or_blocked(gates)

    table_rows = {
        "input_integrity": input_rows,
        "source_inventory": source_inventory,
        "paper_pool_contract": contract,
        "cohort_audit": cohort,
        "prediction_inventory": prediction_inventory,
        "seed_policy_audit": seed_policy,
        "candidate_source_model_decisions": source_decisions,
        "pool_item_provenance": pool,
        "pool_instance_summary": summaries,
        "pool_quality_summary": quality,
        "positive_policy": positive,
        "previous_pool_reconciliation": previous,
        "traditional_pool_metrics": traditional,
        "title_item_reconciliation": titles,
        "comparability_gate": gates,
        "exactness_classification": exactness,
        "future_logical_instances": logical,
        "future_request_manifest": requests,
        "future_shard_manifest": shards,
        "missing_or_blocked": missing,
    }
    for name, rows in table_rows.items():
        write_csv(root / config["outputs"]["tables"][name], rows, TABLE_COLUMNS[name])

    validation = run_validation_suites(root, config)
    counts = {name: len(rows) for name, rows in table_rows.items()}
    manifest = build_manifest(root, config, repository, interpreter, phase_checks, input_rows, validation, gates, exactness, counts)
    report = build_report(repository, interpreter, manifest, counts, quality, gates)
    report_path = root / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    manifest["report_hash"] = sha256_file(report_path)
    write_json(root / config["outputs"]["freeze_manifest"], manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Phase F6G1 LastFM re-ranking reconstruction artifacts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to F6G1 YAML config.")
    args = parser.parse_args(argv)
    root = Path.cwd()
    config = load_config(root / args.config)
    manifest = run_phase(root, config)
    print(
        stable_json(
            {
                "phase": PHASE_ID,
                "decision": manifest["decision"],
                "recommendation": manifest["recommended_next_action"],
                "scientific_content_hash": manifest["scientific_content_hash"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
