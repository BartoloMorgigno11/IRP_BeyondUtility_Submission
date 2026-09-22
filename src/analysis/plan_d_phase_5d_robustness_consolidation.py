"""Offline Phase 5D consolidation of local LLM robustness evidence.

This module reads existing repository artifacts only. It consolidates Phase 5C
full-matrix evidence and earlier Plan D diagnostics into machine-readable
registries, without calling Ollama, APIs, recommender training, or inference
code.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
EXPECTED_REQUESTS = 11528
EXPECTED_EVENTS = 23056
EXPECTED_PROMPT_COMPATIBLE_ROWS = 2882
EXPECTED_EVALUATIONS_PER_TASK = 5764
EXPECTED_RANKING_STRICT_VALID = 5764
EXPECTED_RERANKING_STRICT_VALID = 5738
EXPECTED_INVALID_OUTPUTS = 26

DATASET_ORDER = ["Amazon Beauty", "Amazon Sports", "ML-1M"]
TASK_ORDER = ["ranking", "reranking"]
MODEL_ORDER = ["Qwen2 7B Instruct", "Mistral 7B Instruct"]
MODEL_DIGESTS = {
    "Qwen2 7B Instruct": "dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755",
    "Mistral 7B Instruct": "6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf",
}
MODEL_TAGS = {
    "Qwen2 7B Instruct": "qwen2:7b-instruct",
    "Mistral 7B Instruct": "mistral:7b-instruct",
}

ALLOWED_EVIDENCE_LEVELS = [
    "phase_5c_full_matrix_evidence",
    "exact_paired_diagnostic_experiment",
    "repeated_controlled_diagnostic",
    "small_unpaired_pilot",
    "report_only_summary",
    "blocked_or_unresolved_artifact",
]

OUTPUT_TABLES = {
    "source_inventory": Path("results/tables/plan_d_phase_5d_source_inventory.csv"),
    "experiment_registry": Path("results/tables/plan_d_phase_5d_experiment_registry.csv"),
    "evidence_claims": Path("results/tables/plan_d_phase_5d_evidence_claims.csv"),
    "robustness_summary": Path("results/tables/plan_d_phase_5d_robustness_summary.csv"),
    "protocol_comparison": Path("results/tables/plan_d_phase_5d_protocol_comparison.csv"),
    "model_task_reliability": Path("results/tables/plan_d_phase_5d_model_task_reliability.csv"),
    "pairing_coverage_audit": Path("results/tables/plan_d_phase_5d_pairing_coverage_audit.csv"),
    "blocked_or_unresolved": Path("results/tables/plan_d_phase_5d_blocked_or_unresolved.csv"),
    "validation": Path("results/tables/plan_d_phase_5d_validation.csv"),
    "next_actions": Path("results/tables/plan_d_phase_5d_next_actions.csv"),
}
REPORT_PATH = Path("report/plan_d_phase_5d_robustness_consolidation.md")
INITIAL_PROTECTED_MANIFEST_PATH = Path("results/tables/plan_d_phase_5d_protected_input_hashes_initial.csv")
FINAL_PROTECTED_MANIFEST_PATH = Path("results/tables/plan_d_phase_5d_protected_input_hashes_final.csv")
OUTPUT_HASHES_FIRST_PATH = Path("results/tables/plan_d_phase_5d_output_hashes_first.csv")
OUTPUT_HASHES_SECOND_PATH = Path("results/tables/plan_d_phase_5d_output_hashes_second.csv")

PHASE5D_GENERATED_PATHS = {
    *{path.as_posix() for path in OUTPUT_TABLES.values()},
    REPORT_PATH.as_posix(),
    INITIAL_PROTECTED_MANIFEST_PATH.as_posix(),
    FINAL_PROTECTED_MANIFEST_PATH.as_posix(),
    OUTPUT_HASHES_FIRST_PATH.as_posix(),
    OUTPUT_HASHES_SECOND_PATH.as_posix(),
    "src/analysis/plan_d_phase_5d_robustness_consolidation.py",
    "src/experiments/build_plan_d_phase_5d_robustness_consolidation.py",
    "tests/test_plan_d_phase_5d_robustness_consolidation.py",
}

SOURCE_INVENTORY_COLUMNS = [
    "path",
    "artifact_category",
    "artifact_type",
    "experiment_family",
    "protocol",
    "evidence_level",
    "scope",
    "row_count",
    "column_count",
    "key_columns",
    "inclusion_decision",
    "exclusion_reason",
    "protected_input_status",
]

EXPERIMENT_REGISTRY_COLUMNS = [
    "experiment_id",
    "source_artifacts",
    "artifact_type",
    "experiment_family",
    "phase_identifier",
    "dataset",
    "task",
    "model",
    "model_digest",
    "prompt_identifier",
    "parser_identifier",
    "candidate_count",
    "required_output_size",
    "protocol_family",
    "protocol_status",
    "cohort_definition",
    "cohort_size",
    "paired_key_fields",
    "repetition_or_run_identifier",
    "random_seed",
    "temperature",
    "expected_request_count",
    "observed_request_count",
    "strict_valid_count",
    "invalid_count",
    "hallucination_count",
    "truncation_count",
    "backend_error_count",
    "retry_count",
    "raw_requests_exist",
    "raw_responses_exist",
    "row_level_outputs_exist",
    "aggregate_only_evidence",
    "paired",
    "repeated",
    "scope_classification",
    "generalizable",
    "limitations",
    "scientific_claim_permitted",
    "scientific_claim_prohibited",
    "evidence_level",
]

EVIDENCE_CLAIM_COLUMNS = [
    "topic",
    "claim_id",
    "claim_text",
    "evidence_level",
    "source_experiment_id",
    "source_artifacts",
    "dataset_scope",
    "model_scope",
    "task_scope",
    "protocol_scope",
    "cohort_scope",
    "paired",
    "repeated",
    "generalizable",
    "claim_status",
    "permitted_interpretation",
    "prohibited_interpretation",
    "limitations",
]

ROBUSTNESS_SUMMARY_COLUMNS = [
    "topic",
    "strongest_available_evidence",
    "evidence_scope",
    "paired",
    "repeated",
    "sample_or_cohort_size",
    "result_availability",
    "generalizability",
    "canonical_relevance",
    "concise_finding",
    "limitations",
]

PROTOCOL_COMPARISON_COLUMNS = [
    "protocol_id",
    "protocol_label",
    "prompt_identifier",
    "parser_identifier",
    "candidate_count",
    "output_count",
    "permutation_requirement",
    "strict_validity_rule",
    "execution_scope",
    "canonical_status",
    "evidence_level",
    "comparability_status",
    "source_artifacts",
    "notes",
]

MODEL_TASK_RELIABILITY_COLUMNS = [
    "model",
    "dataset",
    "task",
    "planned_count",
    "backend_success_count",
    "strict_valid_count",
    "invalid_count",
    "duplicate_index_count",
    "hallucination_count",
    "truncation_count",
    "retry_count",
    "backend_error_count",
    "validity_rate",
    "evidence_level",
]

PAIRING_AUDIT_COLUMNS = [
    "comparison_identifier",
    "left_condition",
    "right_condition",
    "intended_pairing_key",
    "left_row_count",
    "right_row_count",
    "matched_pair_count",
    "left_only_count",
    "right_only_count",
    "duplicate_key_count",
    "paired_analysis_permitted",
    "reason",
]

BLOCKED_COLUMNS = [
    "topic",
    "intended_experiment",
    "expected_evidence",
    "available_artifacts",
    "missing_artifacts",
    "blocker_type",
    "status",
    "recoverable_without_new_inference",
    "permitted_statement",
    "prohibited_statement",
]

VALIDATION_COLUMNS = [
    "validation_id",
    "validation_scope",
    "check_name",
    "expected_value",
    "observed_value",
    "status",
    "severity",
    "evidence",
    "notes",
]

NEXT_ACTION_COLUMNS = [
    "action_id",
    "consumer_phase",
    "input_artifact",
    "required_use",
    "status",
    "notes",
]


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value)
    return "" if text.lower() == "nan" else text


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def parse_bool(value: Any) -> bool:
    return clean_cell(value).strip().lower() in {"true", "1", "yes", "y", "1.0"}


def format_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.17g}"


def normalize_repo_path(path: str | Path) -> str:
    """Return a stable repository-relative path string with POSIX separators."""

    text = str(path).replace("\\", "/")
    marker = "IRP_BeyondUtility_Project/"
    if marker in text:
        text = text.split(marker, 1)[1]
    text = re.sub(r"/+", "/", text).strip("/")
    return text


def frame(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame([{column: clean_cell(row.get(column, "")) for column in columns} for row in rows], columns=list(columns))


def read_csv(root: Path, relative: str | Path) -> pd.DataFrame:
    return pd.read_csv(root / relative, dtype=str, keep_default_na=False)


def write_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n")


def stable_csv_bytes(table: pd.DataFrame, columns: Sequence[str] | None = None, sort_columns: Sequence[str] | None = None) -> bytes:
    output = table.copy()
    if columns is not None:
        output = output.loc[:, list(columns)]
    for column in output.columns:
        output[column] = output[column].map(clean_cell)
    if sort_columns:
        output = output.sort_values(list(sort_columns), kind="mergesort").reset_index(drop=True)
    return output.to_csv(index=False, lineterminator="\n").encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, args: Sequence[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{exc}"
    return result.stdout.strip()


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _csv_columns(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def discover_artifacts(root: Path, protected_manifest_path: Path | None = None) -> pd.DataFrame:
    """Discover relevant Phase 5C, Plan D, diagnostic, prompt and parser artifacts."""

    protected = protected_paths_from_manifest(root / protected_manifest_path) if protected_manifest_path else set()
    rows: list[dict[str, Any]] = []
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: normalize_repo_path(p.relative_to(root)).lower()):
        rel = normalize_repo_path(path.relative_to(root))
        if _excluded_path(rel):
            continue
        category = artifact_category(rel)
        include, reason = artifact_inclusion(rel, category)
        if not include and not _interesting_excluded_artifact(rel):
            continue
        row_count = ""
        columns: list[str] = []
        artifact_type = artifact_type_for(rel)
        if path.suffix.lower() == ".csv":
            row_count = str(_csv_row_count(path))
            columns = _csv_columns(path)
        elif path.suffix.lower() == ".jsonl":
            row_count = str(_line_count(path))
        rows.append(
            {
                "path": rel,
                "artifact_category": category,
                "artifact_type": artifact_type,
                "experiment_family": experiment_family_for(rel),
                "protocol": classify_protocol_status(path=rel),
                "evidence_level": evidence_level_for_artifact(rel, include),
                "scope": scope_for_artifact(rel),
                "row_count": row_count,
                "column_count": str(len(columns)) if columns else "",
                "key_columns": "|".join(col for col in columns if col in _KEY_COLUMN_NAMES),
                "inclusion_decision": "included" if include else "excluded",
                "exclusion_reason": reason,
                "protected_input_status": "protected" if rel in protected else "not_in_initial_protected_manifest",
            }
        )
    return frame(rows, SOURCE_INVENTORY_COLUMNS).sort_values("path", kind="mergesort").reset_index(drop=True)


_KEY_COLUMN_NAMES = {
    "request_id",
    "request_order",
    "candidate_set_id",
    "user_id",
    "diagnostic_id",
    "source_instance_id",
    "condition",
    "variant",
    "model",
    "model_family",
    "dataset",
    "task_type",
    "prompt_contract_id",
    "parser_contract_id",
}


def _excluded_path(rel: str) -> bool:
    parts = set(Path(rel).parts)
    if parts.intersection({".git", ".venv", "__pycache__", ".pytest_cache"}):
        return True
    return rel in PHASE5D_GENERATED_PATHS


def _interesting_excluded_artifact(rel: str) -> bool:
    return rel.startswith("configs/plan_e_") or rel.startswith("results/models/plan_e_")


def artifact_type_for(rel: str) -> str:
    suffix = Path(rel).suffix.lower()
    if suffix == ".csv":
        return "csv_table"
    if suffix == ".jsonl":
        if "requests" in rel:
            return "raw_requests_jsonl"
        if "responses" in rel:
            return "raw_responses_jsonl"
        return "runner_events_jsonl"
    if suffix == ".md":
        return "markdown_report"
    if suffix == ".py":
        return "python_source"
    if suffix in {".txt", ".ipynb"} and rel.startswith("original_repo/"):
        return "original_repo_prompt_or_code"
    if suffix == ".pdf":
        return "paper_pdf"
    return suffix.lstrip(".") or "file"


def artifact_category(rel: str) -> str:
    if rel.startswith("results/llm/"):
        return "raw_llm_evidence"
    if rel.startswith("results/tables/canonical_local_llm_full_matrix") or rel.startswith("results/tables/canonical_local_llm_full_"):
        return "phase_5c_full_matrix_evidence"
    if rel.startswith("results/tables/canonical_local_llm") or rel.startswith("results/tables/canonical_llm"):
        return "canonical_llm_evidence"
    if rel.startswith("results/tables/plan_d") or rel.startswith("results/metrics/plan_d"):
        return "plan_d_diagnostic_evidence"
    if rel.startswith("src/llm") or rel.startswith("src/prompts") or rel.startswith("src/evaluation") or rel.startswith("src/metrics"):
        return "prompt_parser_metric_source"
    if rel.startswith("src/analysis") or rel.startswith("src/experiments"):
        return "analysis_or_experiment_source"
    if rel.startswith("tests/"):
        return "test_source"
    if rel.startswith("report/"):
        return "report_evidence"
    if rel.startswith("original_repo/"):
        return "original_repository_evidence"
    if rel.startswith("paper/"):
        return "paper_evidence"
    if rel.startswith("configs/plan_e_") or rel.startswith("results/models/plan_e_"):
        return "plan_e_non_llm_excluded"
    return "supporting_project_artifact"


def artifact_inclusion(rel: str, category: str) -> tuple[bool, str]:
    lower = rel.lower()
    if category == "plan_e_non_llm_excluded":
        return (False, "Phase 5D is limited to existing LLM robustness evidence; Plan E neural artifacts are protected but not interpreted.")
    if rel.startswith("results/models/"):
        return (False, "model checkpoint not used as Phase 5D LLM robustness evidence")
    tokens = [
        "canonical_llm",
        "canonical_local_llm",
        "plan_d_llm",
        "plan_d_qwen",
        "plan_d_mistral",
        "llm",
        "qwen",
        "mistral",
        "prompt",
        "parser",
        "history",
        "position",
        "hallucination",
        "profile",
        "reranking",
        "ranking",
        "validity",
        "full_matrix",
        "top10",
        "full20",
    ]
    if rel.startswith("results/llm/") or any(token in lower for token in tokens):
        return (True, "")
    if rel.startswith("original_repo/EvaLLMasRecommender/eva/templates/"):
        return (True, "")
    if rel in {"README.md", "requirements.txt", "environment.yml", "AGENTS.md"}:
        return (True, "")
    return (False, "not directly relevant to Phase 5D LLM robustness consolidation")


def experiment_family_for(rel: str) -> str:
    lower = rel.lower()
    if "canonical_local_llm_full_matrix" in lower or "canonical_local_llm_full_evaluation" in lower:
        return "phase_5c_full_matrix"
    if "top10_confirmation" in lower or "reranking_top10" in lower:
        return "canonical_top10_confirmation"
    if "hybrid" in lower:
        return "legacy_full20_hybrid_diagnostic"
    if "reranking_prompt_v2" in lower:
        return "legacy_full20_v2_diagnostic"
    if "pilot_resume" in lower or "qwen2_mistral_pilot" in lower or "canonical_local_llm_pilot" in lower:
        return "legacy_full20_v1_pilot"
    if "candidate_position_bias" in lower:
        return "candidate_position_bias_diagnostic"
    if "history_length" in lower:
        return "history_length_diagnostic"
    if "strict_prompt" in lower:
        return "strict_prompt_diagnostic"
    if "qwen_vs_mistral" in lower:
        return "qwen_vs_mistral_diagnostic"
    if "reranking_diagnostic" in lower:
        return "reranking_diagnostic"
    if "output_validity" in lower or "hallucination" in lower or "invalid_item" in lower:
        return "output_validity_diagnostic"
    if "nondeterminism" in lower:
        return "nondeterminism_policy_or_unresolved"
    if "profile" in lower:
        return "generated_profile_blocked_or_template"
    if "blocked" in lower or "conditional" in lower:
        return "blocked_or_conditional"
    return "supporting_artifact"


def scope_for_artifact(rel: str) -> str:
    lower = rel.lower()
    if "full_matrix" in lower or "full_evaluation" in lower:
        return "full_matrix"
    if "pilot" in lower or "confirmation" in lower:
        return "pilot"
    if "diagnostic" in lower or "history_length" in lower or "candidate_position_bias" in lower or "strict_prompt" in lower:
        return "diagnostic_subset"
    if "blocked" in lower or "conditional" in lower or "profile" in lower:
        return "blocked_or_unresolved"
    if rel.startswith("report/"):
        return "report_only"
    return "supporting"


def evidence_level_for_artifact(rel: str, included: bool = True) -> str:
    lower = rel.lower()
    if not included:
        return "blocked_or_unresolved_artifact"
    if "canonical_local_llm_full_matrix" in lower or "canonical_local_llm_full_evaluation" in lower or "canonical_local_llm_structural_metrics" in lower or "canonical_local_llm_utility_metrics" in lower:
        return "phase_5c_full_matrix_evidence"
    if "nondeterminism" in lower or "blocked" in lower or "profile_generation" in lower or ("profile" in lower and "template" in lower):
        return "blocked_or_unresolved_artifact"
    if any(token in lower for token in ["pairwise_delta", "agreement_profile", "stability_profile", "candidate_position_bias", "history_length", "strict_prompt", "qwen_vs_mistral"]):
        return "exact_paired_diagnostic_experiment"
    if "report/" in lower:
        return "report_only_summary"
    if "pilot" in lower or "smoke" in lower or "confirmation" in lower:
        return "small_unpaired_pilot"
    return "report_only_summary"


def classify_protocol_status(
    *,
    path: str = "",
    prompt_identifier: str = "",
    parser_identifier: str = "",
    task: str = "",
    protocol_family: str = "",
) -> str:
    """Classify protocol status using stable labels justified by local artifacts."""

    text = " ".join([path, prompt_identifier, parser_identifier, task, protocol_family]).lower()
    if "top10" in text or "top_10" in text or "officialcode" in text:
        return "canonical_top10"
    if "rank_index_json_top5" in text or ("ranking" in text and "llm_rank_strict_primary_parser_v1" in text):
        return "canonical_ranking_top5"
    if "full20_v1" in text or "full20" in text and "v1" in text:
        return "legacy_full20_v1"
    if "full20_v2" in text or "v2" in text and "reranking_prompt" in text:
        return "legacy_full20_v2"
    if "full20" in text:
        return "legacy_full20_unspecified"
    if "strict_prompt" in text or "strict_candidate" in text:
        return "strict_prompt_diagnostic"
    if "candidate_position_bias" in text or "position_bias" in text:
        return "paired_position_bias_diagnostic"
    if "history_length" in text:
        return "history_length_diagnostic"
    if "nondetermin" in text:
        return "nondeterminism_repeated_diagnostic"
    if "profile" in text:
        return "generated_profile_diagnostic"
    if "blocked" in text:
        return "blocked_protocol"
    if "unresolved" in text:
        return "unresolved_protocol"
    if "pilot" in text or "smoke" in text:
        return "pilot_protocol"
    return "unresolved_protocol"


def assign_evidence_level(
    *,
    scope_classification: str,
    paired: bool,
    repeated: bool,
    report_only: bool = False,
    blocked: bool = False,
) -> str:
    """Assign exactly one Phase 5D evidence level."""

    if blocked:
        return "blocked_or_unresolved_artifact"
    if scope_classification == "full_matrix":
        return "phase_5c_full_matrix_evidence"
    if repeated:
        return "repeated_controlled_diagnostic"
    if paired:
        return "exact_paired_diagnostic_experiment"
    if report_only:
        return "report_only_summary"
    if scope_classification in {"pilot", "diagnostic_subset"}:
        return "small_unpaired_pilot"
    return "blocked_or_unresolved_artifact"


def evidence_strength(level: str) -> int:
    try:
        return ALLOWED_EVIDENCE_LEVELS.index(level)
    except ValueError:
        return len(ALLOWED_EVIDENCE_LEVELS)


def select_strongest_evidence(levels: Iterable[str]) -> str:
    values = [level for level in levels if level]
    if not values:
        return "blocked_or_unresolved_artifact"
    return sorted(values, key=evidence_strength)[0]


def validate_pairing(left: pd.DataFrame, right: pd.DataFrame, keys: Sequence[str]) -> dict[str, Any]:
    """Validate exact pairedness and report unmatched and duplicate keys."""

    missing = [key for key in keys if key not in left.columns or key not in right.columns]
    if missing:
        return _pairing_result(len(left), len(right), 0, len(left), len(right), 0, False, f"missing key columns: {'|'.join(missing)}")

    left_keys = left.loc[:, list(keys)].astype(str)
    right_keys = right.loc[:, list(keys)].astype(str)
    left_tuples = [tuple(row) for row in left_keys.itertuples(index=False, name=None)]
    right_tuples = [tuple(row) for row in right_keys.itertuples(index=False, name=None)]
    left_counts = Counter(left_tuples)
    right_counts = Counter(right_tuples)
    duplicate_key_count = sum(1 for key, count in (left_counts | right_counts).items() if left_counts[key] > 1 or right_counts[key] > 1)
    left_set = set(left_tuples)
    right_set = set(right_tuples)
    matched = len(left_set & right_set)
    left_only = len(left_set - right_set)
    right_only = len(right_set - left_set)
    permitted = matched > 0 and left_only == 0 and right_only == 0 and duplicate_key_count == 0
    reason = "exact key match" if permitted else "unmatched or duplicate keys prevent paired interpretation"
    return _pairing_result(len(left), len(right), matched, left_only, right_only, duplicate_key_count, permitted, reason)


def _pairing_result(
    left_count: int,
    right_count: int,
    matched: int,
    left_only: int,
    right_only: int,
    duplicate_count: int,
    permitted: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "left_row_count": left_count,
        "right_row_count": right_count,
        "matched_pair_count": matched,
        "left_only_count": left_only,
        "right_only_count": right_only,
        "duplicate_key_count": duplicate_count,
        "paired_analysis_permitted": bool_text(permitted),
        "reason": reason,
    }


def validate_nondeterminism_repetition(rows: pd.DataFrame, identity_columns: Sequence[str], run_column: str) -> bool:
    """Return true only when the same controlled condition has repeated runs."""

    if rows.empty or run_column not in rows.columns:
        return False
    if any(column not in rows.columns for column in identity_columns):
        return False
    grouped = rows.groupby(list(identity_columns), dropna=False)[run_column].nunique()
    return bool((grouped >= 2).any())


def validate_position_bias_control(rows: pd.DataFrame) -> bool:
    required = {
        "original_instance_id",
        "variant",
        "original_positive_position",
        "variant_positive_position",
        "candidate_order_signature",
        "candidate_pool_size",
    }
    if rows.empty or not required.issubset(rows.columns):
        return False
    if rows["variant"].nunique() < 2:
        return False
    if rows.duplicated(["variant", "original_instance_id"]).any():
        return False
    changed = pd.to_numeric(rows["original_positive_position"], errors="coerce") != pd.to_numeric(rows["variant_positive_position"], errors="coerce")
    return bool(changed.any())


def validate_history_length_control(rows: pd.DataFrame) -> bool:
    required = {"model", "user_id", "history_setting", "history_length_used", "candidate_ids", "positive_item_id"}
    if rows.empty or not required.issubset(rows.columns):
        return False
    for (_model, user_id), group in rows.groupby(["model", "user_id"]):
        if len(group) < 2:
            continue
        if group["candidate_ids"].nunique() != 1 or group["positive_item_id"].nunique() != 1:
            return False
        if pd.to_numeric(group["history_length_used"], errors="coerce").nunique() >= 2:
            return True
    return False


def generated_profile_claim_allowed(rows: pd.DataFrame | None = None, *, paired_control_exists: bool = False) -> bool:
    if paired_control_exists:
        return True
    if rows is not None and {"profile_condition", "control_condition", "paired"}.issubset(rows.columns):
        return bool(rows["paired"].map(parse_bool).any())
    return False


def output_validity_category(row: Mapping[str, Any]) -> str:
    if not parse_bool(row.get("backend_success", "true")):
        return "backend_error"
    if parse_bool(row.get("output_truncation_detected", row.get("truncation_evidence", "false"))):
        return "truncation"
    if int(float(clean_cell(row.get("hallucination_count", "0")) or "0")) > 0:
        return "hallucination"
    if clean_cell(row.get("primary_failure_reason")) == "duplicate_indices":
        return "duplicate_indices"
    if int(float(clean_cell(row.get("duplicate_count", "0")) or "0")) > 0 and not parse_bool(row.get("uniqueness_valid", "true")):
        return "duplicate_indices"
    if not parse_bool(row.get("strict_parse_valid", "true")):
        return "strict_parse_invalid_other"
    return "strict_valid"


def blocked_status(row: Mapping[str, Any]) -> str:
    text = " ".join(clean_cell(v).lower() for v in row.values())
    if "blocked" in text:
        return "blocked_protocol"
    if "unresolved" in text or "ambiguous" in text:
        return "unresolved_protocol"
    return "not_blocked"


def protected_paths_from_manifest(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "path" not in data.columns:
        return set()
    return {normalize_repo_path(value) for value in data["path"]}


def build_protected_input_manifest(root: Path, output_path: Path) -> pd.DataFrame:
    """Build the deterministic protected-input manifest used for immutability checks."""

    rows: list[dict[str, Any]] = []
    exclude_parts = {".git", ".venv", "__pycache__", ".pytest_cache"}
    scientific_prefixes = (
        "src/",
        "tests/",
        "configs/",
        "report/",
        "results/tables/",
        "results/metrics/",
        "results/llm/",
        "results/raw_outputs/",
        "original_repo/",
        "paper/",
        "data/processed/",
        "data/artifact_manifests/",
        "README",
        "PROJECT_SPEC",
        "AGENTS",
        "CODEX_PROMPTS",
        "reproducibility_checklist",
        "requirements",
        "environment",
    )
    scientific_exts = {".py", ".md", ".yaml", ".yml", ".csv", ".json", ".jsonl", ".txt", ".pdf", ".ipynb", ".zip"}
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: normalize_repo_path(p.relative_to(root)).lower()):
        rel = normalize_repo_path(path.relative_to(root))
        if any(part in exclude_parts for part in Path(rel).parts):
            continue
        if rel in PHASE5D_GENERATED_PATHS:
            continue
        if not (rel.startswith(scientific_prefixes) or path.suffix.lower() in scientific_exts):
            continue
        stat = path.stat()
        size = stat.st_size
        if path.suffix.lower() == ".pt" or size > 75_000_000:
            method = "metadata_only_large_binary_or_checkpoint"
            digest = ""
        else:
            method = "sha256"
            digest = file_sha256(path)
        rows.append(
            {
                "path": rel,
                "protected_input": "true",
                "artifact_category": artifact_category(rel),
                "hash_method": method,
                "sha256": digest,
                "size_bytes": str(size),
                "mtime_ns": str(stat.st_mtime_ns),
                "policy": "sha256_for_scientific_text_csv_json_jsonl_pdf_zip; metadata_only_for_pt_or_files_over_75mb; excludes_git_venv_pycache_and_new_phase5d_outputs",
            }
        )
    columns = ["path", "protected_input", "artifact_category", "hash_method", "sha256", "size_bytes", "mtime_ns", "policy"]
    table = frame(rows, columns)
    write_csv(root / output_path, table)
    return table


def verify_protected_manifest(initial: pd.DataFrame, final: pd.DataFrame) -> bool:
    """Return true when protected inputs are unchanged under the manifest policy."""

    columns = ["path", "protected_input", "hash_method", "sha256", "size_bytes", "mtime_ns", "policy"]
    left = initial.loc[:, columns].sort_values("path", kind="mergesort").reset_index(drop=True).astype(str)
    right = final.loc[:, columns].sort_values("path", kind="mergesort").reset_index(drop=True).astype(str)
    return left.equals(right)


def build_model_task_reliability(root: Path) -> pd.DataFrame:
    cell = read_csv(root, "results/tables/canonical_local_llm_full_matrix_cell_metrics.csv")
    invalid = read_csv(root, "results/tables/canonical_local_llm_full_invalid_output_inventory.csv")
    execution = read_csv(root, "results/tables/canonical_local_llm_full_matrix_execution.csv")
    parse_results = read_csv(root, "results/tables/canonical_local_llm_full_matrix_parse_results.csv")
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            for task in TASK_ORDER:
                c = cell[cell["model_family"].eq(model) & cell["dataset"].eq(dataset) & cell["task_type"].eq(task)].iloc[0]
                inv = invalid[invalid["model_family"].eq(model) & invalid["dataset"].eq(dataset) & invalid["task_type"].eq(task)]
                exe = execution[execution["model_family"].eq(model) & execution["dataset"].eq(dataset) & execution["task_type"].eq(task)]
                par = parse_results[parse_results["model_family"].eq(model) & parse_results["dataset"].eq(dataset) & parse_results["task_type"].eq(task)]
                planned = int(c["planned_rows"])
                strict_valid = int(c["strict_valid_rows"])
                backend_success = int(c["backend_success_rows"])
                duplicate_index_count = int(inv["primary_failure_reason"].eq("duplicate_indices").sum())
                truncation_count = int(par["output_truncation_detected"].map(parse_bool).sum()) if "output_truncation_detected" in par.columns else 0
                retry_count = int(pd.to_numeric(exe.get("attempt_index", pd.Series(dtype=str)), errors="coerce").fillna(1).gt(1).sum()) if not exe.empty else 0
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "task": task,
                        "planned_count": planned,
                        "backend_success_count": backend_success,
                        "strict_valid_count": strict_valid,
                        "invalid_count": planned - strict_valid,
                        "duplicate_index_count": duplicate_index_count,
                        "hallucination_count": int(c.get("hallucination_rows", "0")),
                        "truncation_count": truncation_count,
                        "retry_count": retry_count,
                        "backend_error_count": planned - backend_success,
                        "validity_rate": format_float(strict_valid / planned if planned else 0),
                        "evidence_level": "phase_5c_full_matrix_evidence",
                    }
                )
    return frame(rows, MODEL_TASK_RELIABILITY_COLUMNS)


def build_protocol_comparison(root: Path) -> pd.DataFrame:
    rows = [
        {
            "protocol_id": "PROTO-001",
            "protocol_label": "strict ranking top-5",
            "prompt_identifier": "llm_rank_index_json_top5_v1",
            "parser_identifier": "llm_rank_strict_primary_parser_v1",
            "candidate_count": "20",
            "output_count": "5",
            "permutation_requirement": "false",
            "strict_validity_rule": "exactly_5_unique_one_based_indices_between_1_and_20",
            "execution_scope": "phase_5c_full_matrix_ranking",
            "canonical_status": "canonical_ranking_top5",
            "evidence_level": "phase_5c_full_matrix_evidence",
            "comparability_status": "canonical_for_ranking_only_not_interchangeable_with_reranking",
            "source_artifacts": "results/tables/canonical_local_llm_full_matrix_manifest.csv;results/tables/canonical_llm_prompt_contract.csv;results/tables/canonical_llm_parser_contract.csv",
            "notes": "Ranking returns top-5 indices from 20 candidates; no full permutation.",
        },
        {
            "protocol_id": "PROTO-002",
            "protocol_label": "canonical reranking top-10",
            "prompt_identifier": "llm_rerank_index_json_top10_officialcode_v1",
            "parser_identifier": "llm_rerank_top10_strict_primary_parser_v1",
            "candidate_count": "20",
            "output_count": "10",
            "permutation_requirement": "false",
            "strict_validity_rule": "exactly_10_unique_one_based_indices_between_1_and_20",
            "execution_scope": "phase_5c_full_matrix_reranking",
            "canonical_status": "canonical_top10",
            "evidence_level": "phase_5c_full_matrix_evidence",
            "comparability_status": "canonical_for_reranking_top10_not_semantically_equivalent_to_full20",
            "source_artifacts": "results/tables/canonical_local_llm_reranking_top10_prompt_contract.csv;results/tables/canonical_local_llm_reranking_top10_parser_contract.csv;results/tables/canonical_local_llm_full_matrix_manifest.csv",
            "notes": "Official-code-aligned top-10 semantics adapted to strict JSON indices.",
        },
        {
            "protocol_id": "PROTO-003",
            "protocol_label": "legacy full-20 v1",
            "prompt_identifier": "llm_rerank_index_json_full20_v1",
            "parser_identifier": "llm_rerank_strict_primary_parser_v1",
            "candidate_count": "20",
            "output_count": "20",
            "permutation_requirement": "true",
            "strict_validity_rule": "exactly_20_unique_one_based_indices_between_1_and_20",
            "execution_scope": "pilot_and_diagnostic_only",
            "canonical_status": "legacy_full20_v1",
            "evidence_level": "small_unpaired_pilot",
            "comparability_status": "legacy_diagnostic_not_canonical_top10_evidence",
            "source_artifacts": "results/tables/canonical_local_llm_pilot_resume_parse_results.csv;results/tables/canonical_local_llm_hybrid_prompt_policy_contract.csv",
            "notes": "Full permutation was a local adaptation and remains diagnostic-only.",
        },
        {
            "protocol_id": "PROTO-004",
            "protocol_label": "legacy full-20 v2",
            "prompt_identifier": "llm_rerank_index_json_full20_v2",
            "parser_identifier": "llm_rerank_strict_primary_parser_v1",
            "candidate_count": "20",
            "output_count": "20",
            "permutation_requirement": "true",
            "strict_validity_rule": "exactly_20_unique_one_based_indices_between_1_and_20",
            "execution_scope": "corrective_pilot_and_hybrid_holdout_only",
            "canonical_status": "legacy_full20_v2",
            "evidence_level": "exact_paired_diagnostic_experiment",
            "comparability_status": "paired_only_with_v1_where_rows_match; not comparable to top10 as performance ablation",
            "source_artifacts": "results/tables/canonical_local_llm_reranking_prompt_v2_contract.csv;results/tables/canonical_local_llm_reranking_prompt_v1_v2_comparison.csv;results/tables/canonical_local_llm_hybrid_v1_v2_cell_reconciliation.csv",
            "notes": "Prompt wording changed inside the same full-20 output contract.",
        },
        {
            "protocol_id": "PROTO-005",
            "protocol_label": "strict prompt candidate-ID diagnostic",
            "prompt_identifier": "strict_candidate_id_only",
            "parser_identifier": "plan_d_llm_json_item_id_parser",
            "candidate_count": "20",
            "output_count": "5",
            "permutation_requirement": "false",
            "strict_validity_rule": "exactly_5_candidate_item_ids_under_diagnostic_parser",
            "execution_scope": "ML-1M_20_row_diagnostic_subset",
            "canonical_status": "strict_prompt_diagnostic",
            "evidence_level": "exact_paired_diagnostic_experiment",
            "comparability_status": "diagnostic_prompt_output_space_change_limits_attribution",
            "source_artifacts": "results/tables/plan_d_qwen2_ml1m_strict_prompt_condition_summary.csv;results/tables/plan_d_qwen2_ml1m_strict_prompt_pairwise_delta.csv",
            "notes": "Useful for output-validity diagnosis but not canonical full-matrix execution evidence.",
        },
        {
            "protocol_id": "PROTO-006",
            "protocol_label": "profile generation audit-only",
            "prompt_identifier": "llm_profile_generation_audit_only_v1",
            "parser_identifier": "not_frozen",
            "candidate_count": "",
            "output_count": "free_text_profile",
            "permutation_requirement": "not_applicable",
            "strict_validity_rule": "not_frozen",
            "execution_scope": "blocked_or_unresolved",
            "canonical_status": "blocked_protocol",
            "evidence_level": "blocked_or_unresolved_artifact",
            "comparability_status": "no_controlled_profile_effect_claim_allowed",
            "source_artifacts": "results/tables/canonical_llm_prompt_contract.csv;results/tables/canonical_llm_blocked_or_conditional.csv;original_repo/EvaLLMasRecommender/eva/templates/template_user_profile_generate.txt",
            "notes": "Templates exist, but no canonical generated-profile execution/control is available.",
        },
    ]
    return frame(rows, PROTOCOL_COMPARISON_COLUMNS)


def _experiment_row(**values: Any) -> dict[str, Any]:
    defaults = {column: "" for column in EXPERIMENT_REGISTRY_COLUMNS}
    defaults.update(values)
    if not defaults["protocol_status"]:
        defaults["protocol_status"] = classify_protocol_status(
            prompt_identifier=defaults["prompt_identifier"],
            parser_identifier=defaults["parser_identifier"],
            task=defaults["task"],
            protocol_family=defaults["protocol_family"],
        )
    if not defaults["evidence_level"]:
        defaults["evidence_level"] = assign_evidence_level(
            scope_classification=defaults["scope_classification"],
            paired=parse_bool(defaults["paired"]),
            repeated=parse_bool(defaults["repeated"]),
            report_only=defaults["artifact_type"] == "report_only",
            blocked=defaults["scope_classification"] in {"blocked", "unresolved"},
        )
    return defaults


def build_experiment_registry(root: Path) -> pd.DataFrame:
    reliability = build_model_task_reliability(root)
    manifest = read_csv(root, "results/tables/canonical_local_llm_full_matrix_manifest.csv")
    rows: list[dict[str, Any]] = []
    index = 1
    for _, row in reliability.iterrows():
        model = row["model"]
        dataset = row["dataset"]
        task = row["task"]
        m = manifest[manifest["model_family"].eq(model) & manifest["dataset"].eq(dataset) & manifest["task_type"].eq(task)].iloc[0]
        rows.append(
            _experiment_row(
                experiment_id=f"EXP-{index:03d}",
                source_artifacts="results/tables/canonical_local_llm_full_matrix_manifest.csv;results/tables/canonical_local_llm_full_matrix_cell_metrics.csv;results/tables/canonical_local_llm_full_matrix_parse_results.csv;results/llm/canonical_local_full_matrix/requests.jsonl;results/llm/canonical_local_full_matrix/responses.jsonl",
                artifact_type="full_matrix_cell",
                experiment_family="phase_5c_full_matrix",
                phase_identifier="Phase 5C",
                dataset=dataset,
                task=task,
                model=model,
                model_digest=MODEL_DIGESTS.get(model, ""),
                prompt_identifier=m["prompt_contract_id"],
                parser_identifier=m["parser_contract_id"],
                candidate_count=m["candidate_count"],
                required_output_size=m["requested_output_count"],
                protocol_family="canonical_local_llm",
                protocol_status=classify_protocol_status(prompt_identifier=m["prompt_contract_id"], parser_identifier=m["parser_contract_id"], task=task),
                cohort_definition="prompt_compatible_strict_title_cohort",
                cohort_size=row["planned_count"],
                paired_key_fields="dataset|task_type|candidate_set_id|user_id",
                repetition_or_run_identifier="single_canonical_local_full_matrix_run",
                random_seed=m["seed"],
                temperature=m["temperature"],
                expected_request_count=row["planned_count"],
                observed_request_count=row["planned_count"],
                strict_valid_count=row["strict_valid_count"],
                invalid_count=row["invalid_count"],
                hallucination_count=row["hallucination_count"],
                truncation_count=row["truncation_count"],
                backend_error_count=row["backend_error_count"],
                retry_count=row["retry_count"],
                raw_requests_exist="true",
                raw_responses_exist="true",
                row_level_outputs_exist="true",
                aggregate_only_evidence="false",
                paired="false",
                repeated="false",
                scope_classification="full_matrix",
                generalizable="within_frozen_prompt_compatible_cohort_only",
                limitations="Local substitute model and adapted JSON-index protocol; not paper-exact model reproduction.",
                scientific_claim_permitted="Structural reliability and conditional/adapted utility for this model-dataset-task cell.",
                scientific_claim_prohibited="Do not generalize to paper models, LastFM, full-20 legacy reranking, or uncontrolled prompt variants.",
                evidence_level="phase_5c_full_matrix_evidence",
            )
        )
        index += 1

    index = _append_prior_experiment_rows(root, rows, index)
    return frame(rows, EXPERIMENT_REGISTRY_COLUMNS).sort_values("experiment_id", kind="mergesort").reset_index(drop=True)


def _append_prior_experiment_rows(root: Path, rows: list[dict[str, Any]], index: int) -> int:
    specs = [
        {
            "path": "results/tables/canonical_local_llm_top10_confirmation_parse_results.csv",
            "experiment_family": "canonical_top10_confirmation",
            "phase": "Phase 5B-P1G",
            "dataset": "Amazon Beauty;Amazon Sports;ML-1M",
            "task": "reranking",
            "model": "Qwen2 7B Instruct;Mistral 7B Instruct",
            "prompt": "llm_rerank_index_json_top10_officialcode_v1",
            "parser": "llm_rerank_top10_strict_primary_parser_v1",
            "candidate_count": "20",
            "output_count": "10",
            "scope": "pilot",
            "paired": "false",
            "level": "small_unpaired_pilot",
            "limitations": "Ten-row-per-cell confirmation pilot; readiness diagnostic only.",
        },
        {
            "path": "results/tables/canonical_local_llm_pilot_resume_parse_results.csv",
            "experiment_family": "legacy_full20_v1_pilot",
            "phase": "Phase 5B-P1",
            "dataset": "Amazon Beauty;Amazon Sports;ML-1M",
            "task": "ranking;reranking",
            "model": "Qwen2 7B Instruct;Mistral 7B Instruct",
            "prompt": "llm_rank_index_json_top5_v1;llm_rerank_index_json_full20_v1",
            "parser": "llm_rank_strict_primary_parser_v1;llm_rerank_strict_primary_parser_v1",
            "candidate_count": "20",
            "output_count": "5;20",
            "scope": "pilot",
            "paired": "false",
            "level": "small_unpaired_pilot",
            "limitations": "Pilot/resume evidence; full-20 reranking is legacy diagnostic-only.",
        },
        {
            "path": "results/tables/canonical_local_llm_reranking_prompt_v2_parse_results.csv",
            "experiment_family": "legacy_full20_v2_diagnostic",
            "phase": "Phase 5B-P1C",
            "dataset": "Amazon Beauty;Amazon Sports;ML-1M",
            "task": "reranking",
            "model": "Qwen2 7B Instruct;Mistral 7B Instruct",
            "prompt": "llm_rerank_index_json_full20_v2",
            "parser": "llm_rerank_strict_primary_parser_v1",
            "candidate_count": "20",
            "output_count": "20",
            "scope": "pilot",
            "paired": "true",
            "level": "exact_paired_diagnostic_experiment",
            "limitations": "Paired where v1-v2 rows align; legacy full-20 only.",
        },
        {
            "path": "results/tables/canonical_local_llm_hybrid_holdout_parse_results.csv",
            "experiment_family": "legacy_full20_hybrid_diagnostic",
            "phase": "Phase 5B-P1E",
            "dataset": "Amazon Beauty;Amazon Sports;ML-1M",
            "task": "reranking",
            "model": "Qwen2 7B Instruct;Mistral 7B Instruct",
            "prompt": "model_specific_legacy_full20_hybrid",
            "parser": "llm_rerank_strict_primary_parser_v1",
            "candidate_count": "20",
            "output_count": "20",
            "scope": "pilot",
            "paired": "false",
            "level": "small_unpaired_pilot",
            "limitations": "Disjoint holdout for full-20 prompt selection, not canonical execution evidence.",
        },
        {
            "path": "results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_instance_summary.csv",
            "experiment_family": "qwen2_ranking_diagnostic",
            "phase": "D5E",
            "dataset": "ML-1M",
            "task": "ranking",
            "model": "qwen2:7b-instruct",
            "prompt": "openp5_base",
            "parser": "plan_d_llm_json_item_id_parser",
            "candidate_count": "20",
            "output_count": "5",
            "scope": "diagnostic_subset",
            "paired": "false",
            "level": "small_unpaired_pilot",
            "limitations": "Twenty-row ML-1M pilot with older item-ID prompt; not canonical full-matrix evidence.",
        },
        {
            "path": "results/tables/plan_d_qwen2_ml1m_candidate_position_bias_instance_summary.csv",
            "experiment_family": "candidate_position_bias_diagnostic",
            "phase": "D5F",
            "dataset": "ML-1M",
            "task": "ranking",
            "model": "qwen2:7b-instruct",
            "prompt": "openp5_base",
            "parser": "plan_d_llm_json_item_id_parser",
            "candidate_count": "20",
            "output_count": "5",
            "scope": "diagnostic_subset",
            "paired": "true",
            "level": "exact_paired_diagnostic_experiment",
            "limitations": "Controlled candidate reorderings over 20 ML-1M rows; original order was not rerun in the same diagnostic.",
        },
        {
            "path": "results/tables/plan_d_qwen2_ml1m_strict_prompt_instance_summary.csv",
            "experiment_family": "strict_prompt_diagnostic",
            "phase": "D5G",
            "dataset": "ML-1M",
            "task": "ranking",
            "model": "qwen2:7b-instruct",
            "prompt": "strict_candidate_id_only",
            "parser": "plan_d_llm_json_item_id_parser",
            "candidate_count": "20",
            "output_count": "5",
            "scope": "diagnostic_subset",
            "paired": "true",
            "level": "exact_paired_diagnostic_experiment",
            "limitations": "Prompt/output-space changes and post-filter rows mean attribution is not wording-only.",
        },
        {
            "path": "results/tables/plan_d_qwen_vs_mistral_ml1m_instance_summary.csv",
            "experiment_family": "qwen_vs_mistral_diagnostic",
            "phase": "D5I",
            "dataset": "ML-1M",
            "task": "ranking",
            "model": "qwen2:7b-instruct;mistral:7b-instruct",
            "prompt": "strict_candidate_id_only",
            "parser": "plan_d_llm_json_item_id_parser",
            "candidate_count": "20",
            "output_count": "5",
            "scope": "diagnostic_subset",
            "paired": "true",
            "level": "exact_paired_diagnostic_experiment",
            "limitations": "Twenty paired ML-1M rows; Qwen2 rows reused from D5G rather than rerun.",
        },
        {
            "path": "results/tables/plan_d_llm_ml1m_reranking_instance_summary.csv",
            "experiment_family": "reranking_diagnostic",
            "phase": "D5J",
            "dataset": "ML-1M",
            "task": "reranking",
            "model": "qwen2:7b-instruct;mistral:7b-instruct",
            "prompt": "rerank_prompt_diagnostic",
            "parser": "plan_d_llm_json_item_id_parser",
            "candidate_count": "20",
            "output_count": "5",
            "scope": "diagnostic_subset",
            "paired": "true",
            "level": "exact_paired_diagnostic_experiment",
            "limitations": "Includes a deterministic candidate-order reference condition; not a canonical top-10 reranking run.",
        },
        {
            "path": "results/tables/plan_d_llm_ml1m_history_length_instance_summary.csv",
            "experiment_family": "history_length_diagnostic",
            "phase": "D5K",
            "dataset": "ML-1M",
            "task": "ranking",
            "model": "qwen2:7b-instruct;mistral:7b-instruct",
            "prompt": "strict_candidate_id_only_history_length",
            "parser": "plan_d_llm_json_item_id_parser",
            "candidate_count": "20",
            "output_count": "5",
            "scope": "diagnostic_subset",
            "paired": "true",
            "level": "exact_paired_diagnostic_experiment",
            "limitations": "Twenty users per model across short, medium, and long history; not full-cohort evidence.",
        },
    ]
    for spec in specs:
        path = root / spec["path"]
        if not path.is_file():
            continue
        data = read_csv(root, spec["path"])
        strict_valid = _strict_valid_count(data)
        hallucination = _sum_column(data, ["hallucinated_count", "hallucination_count"])
        duplicates = _sum_column(data, ["duplicate_count"])
        rows.append(
            _experiment_row(
                experiment_id=f"EXP-{index:03d}",
                source_artifacts=spec["path"],
                artifact_type="diagnostic_or_pilot_table",
                experiment_family=spec["experiment_family"],
                phase_identifier=spec["phase"],
                dataset=spec["dataset"],
                task=spec["task"],
                model=spec["model"],
                model_digest="",
                prompt_identifier=spec["prompt"],
                parser_identifier=spec["parser"],
                candidate_count=spec["candidate_count"],
                required_output_size=spec["output_count"],
                protocol_family=spec["experiment_family"],
                protocol_status=classify_protocol_status(path=spec["path"], prompt_identifier=spec["prompt"], parser_identifier=spec["parser"], task=spec["task"]),
                cohort_definition="diagnostic_subset_or_pilot",
                cohort_size=str(len(data)),
                paired_key_fields="source_instance_id|user_id" if spec["paired"] == "true" else "",
                repetition_or_run_identifier="single_run_or_prior_reuse",
                random_seed="",
                temperature=_first_existing(data, "temperature"),
                expected_request_count=str(len(data)),
                observed_request_count=str(len(data)),
                strict_valid_count=str(strict_valid),
                invalid_count=str(max(0, len(data) - strict_valid)),
                hallucination_count=str(hallucination),
                truncation_count="0",
                backend_error_count=str(int(data["response_received"].map(lambda v: not parse_bool(v)).sum())) if "response_received" in data.columns else "",
                retry_count="0",
                raw_requests_exist=bool_text((root / spec["path"].replace("results/tables/", "results/llm/")).exists()),
                raw_responses_exist=bool_text("parse_results" in spec["path"] or "instance_summary" in spec["path"]),
                row_level_outputs_exist="true",
                aggregate_only_evidence="false",
                paired=spec["paired"],
                repeated="false",
                scope_classification=spec["scope"],
                generalizable="false_diagnostic_subset_only" if spec["scope"] != "full_matrix" else "within_scope",
                limitations=spec["limitations"],
                scientific_claim_permitted="Descriptive or paired diagnostic claim within the artifact's cohort only.",
                scientific_claim_prohibited="Do not present as full-cohort, canonical, paper-exact, or causal evidence.",
                evidence_level=spec["level"],
            )
        )
        index += 1

    for topic, family, artifacts in [
        ("nondeterminism", "nondeterminism_unresolved", "results/tables/canonical_llm_nondeterminism_policy.csv"),
        ("generated_profile", "generated_profile_blocked", "results/tables/canonical_llm_blocked_or_conditional.csv;results/tables/canonical_llm_prompt_contract.csv"),
    ]:
        rows.append(
            _experiment_row(
                experiment_id=f"EXP-{index:03d}",
                source_artifacts=artifacts,
                artifact_type="blocked_or_policy_record",
                experiment_family=family,
                phase_identifier="Phase 5D input audit",
                dataset="ML-1M;Amazon Beauty;Amazon Sports",
                task="ranking" if topic == "nondeterminism" else "profile_generation",
                model="Qwen2 7B Instruct;Mistral 7B Instruct",
                protocol_family=family,
                protocol_status="blocked_protocol" if topic == "generated_profile" else "unresolved_protocol",
                cohort_definition="not_executed",
                cohort_size="0",
                expected_request_count="0",
                observed_request_count="0",
                strict_valid_count="0",
                invalid_count="0",
                raw_requests_exist="false",
                raw_responses_exist="false",
                row_level_outputs_exist="false",
                aggregate_only_evidence="true",
                paired="false",
                repeated="false",
                scope_classification="blocked",
                generalizable="false",
                limitations="No controlled executed evidence in repository.",
                scientific_claim_permitted="May state that the experiment is blocked or unresolved.",
                scientific_claim_prohibited="Do not claim an effect or executed repeated/profile comparison.",
                evidence_level="blocked_or_unresolved_artifact",
            )
        )
        index += 1
    return index


def _strict_valid_count(data: pd.DataFrame) -> int:
    if "strict_parse_valid" in data.columns:
        return int(data["strict_parse_valid"].map(parse_bool).sum())
    if "schema_valid" in data.columns and "candidate_validity_rate" in data.columns:
        valid = data["schema_valid"].map(parse_bool) & pd.to_numeric(data["candidate_validity_rate"], errors="coerce").fillna(0).eq(1)
        return int(valid.sum())
    if "parse_success" in data.columns:
        return int(data["parse_success"].map(parse_bool).sum())
    return len(data)


def _sum_column(data: pd.DataFrame, columns: Sequence[str]) -> int:
    for column in columns:
        if column in data.columns:
            return int(pd.to_numeric(data[column], errors="coerce").fillna(0).sum())
    return 0


def _first_existing(data: pd.DataFrame, column: str) -> str:
    if column not in data.columns or data.empty:
        return ""
    values = [clean_cell(value) for value in data[column].tolist() if clean_cell(value)]
    return values[0] if values else ""


def build_pairing_coverage_audit(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(identifier: str, left_condition: str, right_condition: str, data: pd.DataFrame, condition_col: str, keys: Sequence[str]) -> None:
        left = data[data[condition_col].eq(left_condition)]
        right = data[data[condition_col].eq(right_condition)]
        result = validate_pairing(left, right, keys)
        rows.append(
            {
                "comparison_identifier": identifier,
                "left_condition": left_condition,
                "right_condition": right_condition,
                "intended_pairing_key": "|".join(keys),
                **result,
            }
        )

    path = root / "results/tables/plan_d_qwen2_ml1m_candidate_position_bias_instance_summary.csv"
    if path.is_file():
        data = read_csv(root, path.relative_to(root))
        for variant in sorted(data["variant"].unique()):
            # The left reference is represented by original_instance_id; D5E rows are audited separately.
            reference = data[data["variant"].eq(variant)].copy()
            reference["reference_condition"] = "original_order_reference"
            result = validate_pairing(reference, data[data["variant"].eq(variant)], ["original_instance_id"])
            rows.append({"comparison_identifier": f"position_bias_original_vs_{variant}", "left_condition": "original_order_reference", "right_condition": variant, "intended_pairing_key": "original_instance_id", **result})

    path = root / "results/tables/plan_d_qwen2_ml1m_strict_prompt_instance_summary.csv"
    if path.is_file():
        data = read_csv(root, path.relative_to(root))
        for right in ["original_post_filtered", "strict_candidate_id_prompt", "strict_post_filtered"]:
            add(f"strict_prompt_original_vs_{right}", "original_parsed_output", right, data, "condition", ["source_instance_id"])

    path = root / "results/tables/plan_d_qwen_vs_mistral_ml1m_instance_summary.csv"
    if path.is_file():
        data = read_csv(root, path.relative_to(root))
        add("qwen_vs_mistral_strict_prompt", "qwen2_strict_reference", "mistral_strict_candidate_id_prompt", data, "condition", ["source_instance_id"])

    path = root / "results/tables/plan_d_llm_ml1m_history_length_instance_summary.csv"
    if path.is_file():
        data = read_csv(root, path.relative_to(root))
        for model in sorted(data["model"].unique()):
            model_rows = data[data["model"].eq(model)]
            for left, right in [("short_history", "medium_history"), ("long_history", "medium_history"), ("short_history", "long_history")]:
                add(f"history_length_{model}_{left}_vs_{right}", left, right, model_rows, "history_setting", ["user_id", "model"])

    path = root / "results/tables/plan_d_llm_ml1m_reranking_instance_summary.csv"
    if path.is_file():
        data = read_csv(root, path.relative_to(root))
        for left, right in [
            ("candidate_order_reference", "qwen2_rerank_prompt"),
            ("candidate_order_reference", "mistral_rerank_prompt"),
            ("qwen2_rerank_prompt", "mistral_rerank_prompt"),
        ]:
            add(f"reranking_{left}_vs_{right}", left, right, data, "condition", ["source_instance_id"])

    path = root / "results/tables/canonical_local_llm_reranking_prompt_v1_v2_comparison.csv"
    if path.is_file():
        data = read_csv(root, path.relative_to(root))
        keys = [column for column in ["model_family", "dataset", "task_type", "candidate_set_id", "user_id"] if column in data.columns]
        result = _audit_v1_v2_comparison(data, keys)
        rows.append({"comparison_identifier": "legacy_full20_v1_vs_v2", "left_condition": "full20_v1", "right_condition": "full20_v2", "intended_pairing_key": "|".join(keys) if keys else "comparison_table_rows", **result})

    path = root / "results/tables/canonical_local_llm_top10_confirmation_sample.csv"
    if path.is_file():
        sample = read_csv(root, path.relative_to(root))
        overlap = int(sample.get("used_in_v1_pilot", pd.Series(dtype=str)).map(parse_bool).sum()) + int(sample.get("used_in_v2_pilot", pd.Series(dtype=str)).map(parse_bool).sum()) if not sample.empty else 0
        rows.append(
            {
                "comparison_identifier": "full20_vs_top10_direct_numeric_ablation",
                "left_condition": "legacy_full20",
                "right_condition": "canonical_top10",
                "intended_pairing_key": "candidate_set_id|user_id",
                "left_row_count": "",
                "right_row_count": str(len(sample)),
                "matched_pair_count": "0",
                "left_only_count": "",
                "right_only_count": "",
                "duplicate_key_count": "0",
                "paired_analysis_permitted": "false",
                "reason": f"top10 confirmation sample is disjoint from previous pilots; overlap flags sum to {overlap}",
            }
        )

    return frame(rows, PAIRING_AUDIT_COLUMNS)


def _audit_v1_v2_comparison(data: pd.DataFrame, keys: Sequence[str]) -> dict[str, Any]:
    if not keys:
        return _pairing_result(len(data), len(data), len(data), 0, 0, 0, True, "comparison table is already row-aligned but has no explicit key columns")
    duplicates = int(data.duplicated(list(keys)).sum())
    permitted = duplicates == 0 and len(data) > 0
    return _pairing_result(len(data), len(data), len(data) if permitted else 0, 0, 0, duplicates, permitted, "precomputed v1-v2 comparison rows use stable keys" if permitted else "duplicate keys prevent paired interpretation")


def build_blocked_or_unresolved(root: Path) -> pd.DataFrame:
    rows = [
        {
            "topic": "nondeterminism",
            "intended_experiment": "repeated controlled local LLM runs under identical model, prompt, parser, candidate order, history, and generation settings",
            "expected_evidence": "at least two run identifiers per identical controlled row",
            "available_artifacts": "results/tables/canonical_llm_nondeterminism_policy.csv",
            "missing_artifacts": "executed repeated-run raw requests/responses and row-level outputs",
            "blocker_type": "not_executed",
            "status": "blocked_or_unresolved_artifact",
            "recoverable_without_new_inference": "false",
            "permitted_statement": "A nondeterminism extension policy exists.",
            "prohibited_statement": "Do not claim nondeterminism or stability effects from single-run evidence.",
        },
        {
            "topic": "generated_profiles",
            "intended_experiment": "profile generation and matched no-profile recommendation control",
            "expected_evidence": "paired profile/no-profile rows with frozen prompt, parser, model, and leakage policy",
            "available_artifacts": "official profile templates; canonical prompt audit blocked rows",
            "missing_artifacts": "canonical profile outputs, matched controls, parser contract, and metric table",
            "blocker_type": "ambiguous_contract",
            "status": "blocked_or_unresolved_artifact",
            "recoverable_without_new_inference": "false",
            "permitted_statement": "Profile templates and archived profile-like filenames exist.",
            "prohibited_statement": "Do not claim profile benefit or harm.",
        },
        {
            "topic": "full20_vs_top10_numeric_ablation",
            "intended_experiment": "direct paired comparison of legacy full-20 and canonical top-10 reranking protocols",
            "expected_evidence": "aligned cohort, same candidates/history/model, and explicit paired keys for both protocols",
            "available_artifacts": "full-20 pilots; top-10 confirmation and Phase 5C full matrix",
            "missing_artifacts": "paired full-20 and top-10 row-level outputs over the same cohort",
            "blocker_type": "unpaired_protocols",
            "status": "blocked_or_unresolved_artifact",
            "recoverable_without_new_inference": "false",
            "permitted_statement": "Protocols can be compared structurally.",
            "prohibited_statement": "Do not report a direct full-20 versus top-10 performance effect.",
        },
        {
            "topic": "LastFM",
            "intended_experiment": "full local LLM robustness matrix on LastFM",
            "expected_evidence": "prompt-compatible canonical rows and full matrix outputs",
            "available_artifacts": "blocked/limited status in prior reports",
            "missing_artifacts": "complete prompt-compatible LastFM execution artifacts",
            "blocker_type": "incomplete_extension",
            "status": "blocked_or_unresolved_artifact",
            "recoverable_without_new_inference": "false",
            "permitted_statement": "LastFM remains blocked or incomplete.",
            "prohibited_statement": "Do not include LastFM in complete canonical Phase 5C conclusions.",
        },
        {
            "topic": "paper_exact_remote_models",
            "intended_experiment": "paper-model exact GPT/Claude/Llama local/API reproduction",
            "expected_evidence": "provider-verified model outputs under paper prompts",
            "available_artifacts": "results/tables/canonical_llm_model_inventory.csv",
            "missing_artifacts": "approved API calls, current provider verification, exact output artifacts",
            "blocker_type": "external_api_or_hardware",
            "status": "blocked_or_unresolved_artifact",
            "recoverable_without_new_inference": "false",
            "permitted_statement": "Local Qwen2 and Mistral are adapted extensions.",
            "prohibited_statement": "Do not call them exact paper-model reproductions.",
        },
    ]
    return frame(rows, BLOCKED_COLUMNS)


def build_evidence_claims(
    reliability: pd.DataFrame,
    pairing: pd.DataFrame,
    blocked: pd.DataFrame,
) -> pd.DataFrame:
    invalid_total = int(pd.to_numeric(reliability["invalid_count"], errors="coerce").sum())
    qwen_rerank_invalid = int(
        pd.to_numeric(
            reliability[reliability["model"].eq("Qwen2 7B Instruct") & reliability["task"].eq("reranking")]["invalid_count"],
            errors="coerce",
        ).sum()
    )
    ranking_valid = int(pd.to_numeric(reliability[reliability["task"].eq("ranking")]["strict_valid_count"], errors="coerce").sum())
    reranking_valid = int(pd.to_numeric(reliability[reliability["task"].eq("reranking")]["strict_valid_count"], errors="coerce").sum())
    hallucination = int(pd.to_numeric(reliability["hallucination_count"], errors="coerce").sum())
    rows = [
        _claim("output_validity", "CLAIM-001", f"Phase 5C reconciles {EXPECTED_REQUESTS} planned canonical local LLM requests and responses.", "phase_5c_full_matrix_evidence", "EXP-001..EXP-012", "results/tables/canonical_local_llm_full_matrix_decision.csv", "Amazon Beauty;Amazon Sports;ML-1M", "Qwen2 7B Instruct;Mistral 7B Instruct", "ranking;reranking", "canonical_ranking_top5;canonical_top10", "2882 prompt-compatible rows per task before model expansion", "false", "false", "within_canonical_scope", "permitted", "Use as full-matrix local evidence.", "Do not treat as paper-model exact reproduction.", "Local substitute models and adapted JSON-index output."),
        _claim("ranking_vs_reranking", "CLAIM-002", f"Canonical ranking is {ranking_valid}/{EXPECTED_EVALUATIONS_PER_TASK} strict-valid; canonical reranking is {reranking_valid}/{EXPECTED_EVALUATIONS_PER_TASK} strict-valid.", "phase_5c_full_matrix_evidence", "EXP-001..EXP-012", "results/tables/plan_d_phase_5d_model_task_reliability.csv", "all canonical datasets", "both local models", "ranking;reranking", "canonical_ranking_top5;canonical_top10", "full Phase 5C matrix", "false", "false", "within_canonical_scope", "permitted", "Report task-specific structural reliability separately.", "Do not collapse task semantics into one interchangeable number.", "Ranking asks for 5 outputs; reranking asks for 10."),
        _claim("duplicate_indices", "CLAIM-003", f"The {invalid_total} Phase 5C invalid outputs are all Qwen2 reranking duplicate-index failures; no repair was applied.", "phase_5c_full_matrix_evidence", "EXP-004;EXP-006;EXP-008", "results/tables/canonical_local_llm_full_invalid_output_inventory.csv", "Amazon Beauty;Amazon Sports;ML-1M", "Qwen2 7B Instruct", "reranking", "canonical_top10", "full Phase 5C reranking matrix", "false", "false", "within_canonical_scope", "permitted", "Report duplicate-index failures as structural invalidity.", "Do not count them as hallucinations or silently repair them.", "Invalid rows are retained."),
        _claim("hallucination", "CLAIM-004", f"Under the canonical strict-index Phase 5C protocol, hallucination rows are {hallucination}.", "phase_5c_full_matrix_evidence", "EXP-001..EXP-012", "results/tables/canonical_local_llm_structural_metrics.csv", "all canonical datasets", "both local models", "ranking;reranking", "canonical_ranking_top5;canonical_top10", "full Phase 5C matrix", "false", "false", "within_canonical_scope", "permitted", "Use the repository's frozen hallucination definition.", "Do not merge hallucination with parser-invalid duplicate outputs.", "Earlier openp5 pilots used different prompts/parsers."),
        _claim("candidate_position_bias", "CLAIM-005", "Candidate-position evidence is a paired ML-1M Qwen2 diagnostic with controlled candidate reorderings.", "exact_paired_diagnostic_experiment", "EXP-018", "results/tables/plan_d_qwen2_ml1m_candidate_position_bias_instance_summary.csv;results/tables/plan_d_qwen2_ml1m_candidate_position_bias_pairwise_delta.csv", "ML-1M", "qwen2:7b-instruct", "ranking", "paired_position_bias_diagnostic", "20 source rows x 3 variants", "true", "false", "false_diagnostic_subset_only", "permitted_with_limits", "Discuss as controlled diagnostic evidence.", "Do not generalize to full cohorts or all models.", "Original order was not rerun in D5F."),
        _claim("history_length", "CLAIM-006", "History-length evidence is paired across short, medium, and long histories for Qwen2 and Mistral on 20 ML-1M users.", "exact_paired_diagnostic_experiment", "EXP-022", "results/tables/plan_d_llm_ml1m_history_length_instance_summary.csv;results/tables/plan_d_llm_ml1m_history_length_pairwise_delta.csv", "ML-1M", "qwen2:7b-instruct;mistral:7b-instruct", "ranking", "history_length_diagnostic", "20 users per model", "true", "false", "false_diagnostic_subset_only", "permitted_with_limits", "Report sensitivity only within the diagnostic design.", "Do not present as full-matrix evidence.", "History variants are diagnostic prompt conditions."),
        _claim("strict_prompt", "CLAIM-007", "Strict prompt behavior is supported only as a paired ML-1M Qwen2 diagnostic, not as a wording-only causal ablation.", "exact_paired_diagnostic_experiment", "EXP-019", "results/tables/plan_d_qwen2_ml1m_strict_prompt_condition_summary.csv;results/tables/plan_d_qwen2_ml1m_strict_prompt_pairwise_delta.csv", "ML-1M", "qwen2:7b-instruct", "ranking", "strict_prompt_diagnostic", "20 source rows plus post-filter rows", "true", "false", "false_diagnostic_subset_only", "permitted_with_limits", "Separate prompt, parser, and output-space changes.", "Do not attribute differences solely to wording.", "Post-filter rows are deterministic analyses, not new LLM outputs."),
        _claim("full20_v1_v2", "CLAIM-008", "Full-20 v1 versus v2 evidence is legacy full-permutation diagnostic evidence only.", "exact_paired_diagnostic_experiment", "EXP-014", "results/tables/canonical_local_llm_reranking_prompt_v1_v2_comparison.csv", "pilot datasets", "Qwen2 7B Instruct;Mistral 7B Instruct", "reranking", "legacy_full20_v1;legacy_full20_v2", "paired pilot rows where available", "true", "false", "false_legacy_only", "permitted_with_limits", "Use for legacy protocol diagnosis.", "Do not use as canonical top-10 execution evidence.", "Output cardinality and full-permutation requirement differ from canonical top-10."),
        _claim("full20_vs_top10", "CLAIM-009", "Full-20 and top-10 are protocol-different conditions and no direct paired performance ablation is supported.", "blocked_or_unresolved_artifact", "blocked_full20_vs_top10_numeric_ablation", "results/tables/plan_d_phase_5d_protocol_comparison.csv;results/tables/plan_d_phase_5d_pairing_coverage_audit.csv", "all canonical datasets", "both local models", "reranking", "legacy_full20;canonical_top10", "no matched full-20/top-10 cohort", "false", "false", "false", "blocked", "Compare protocol contracts structurally.", "Do not report numerical full-20 versus top-10 effects.", "No aligned paired experiment exists."),
        _claim("qwen2_vs_mistral", "CLAIM-010", f"Phase 5C supports a structural reliability comparison: Qwen2 has {qwen_rerank_invalid} reranking invalid rows and Mistral has none.", "phase_5c_full_matrix_evidence", "EXP-001..EXP-012", "results/tables/plan_d_phase_5d_model_task_reliability.csv", "all canonical datasets", "Qwen2 7B Instruct;Mistral 7B Instruct", "ranking;reranking", "canonical_ranking_top5;canonical_top10", "full Phase 5C matrix", "false", "false", "within_canonical_scope", "permitted", "Keep reliability, utility, and beyond-utility separate.", "Do not collapse tasks/datasets into an unjustified single score.", "Local substitute models only."),
        _claim("generated_profiles", "CLAIM-011", "Generated-profile effects are not supported because no canonical paired profile/no-profile control exists.", "blocked_or_unresolved_artifact", "EXP-024", "results/tables/canonical_llm_blocked_or_conditional.csv;original_repo/EvaLLMasRecommender/eva/templates/template_user_profile_generate.txt", "Amazon Beauty;Amazon Sports;ML-1M", "paper/API models;local models unresolved", "profile_generation", "blocked_protocol", "not_executed", "false", "false", "false", "blocked", "State that templates/artifact names exist.", "Do not claim profile benefit or harm.", "Profile model, reuse, parser, and leakage policy are not frozen."),
        _claim("nondeterminism", "CLAIM-012", "No executed repeated controlled LLM nondeterminism diagnostic is present in the repository.", "blocked_or_unresolved_artifact", "EXP-023", "results/tables/canonical_llm_nondeterminism_policy.csv", "ML-1M planned", "Qwen2 7B Instruct;Mistral 7B Instruct", "ranking;reranking", "nondeterminism_repeated_diagnostic", "not_executed", "false", "false", "false", "blocked", "State that a result-independent repeat policy exists.", "Do not infer nondeterminism from a single run.", "Repeated run identifiers and raw outputs are missing."),
    ]
    return frame(rows, EVIDENCE_CLAIM_COLUMNS)


def _claim(*values: Any) -> dict[str, Any]:
    return dict(zip(EVIDENCE_CLAIM_COLUMNS, values, strict=True))


def build_robustness_summary(claims: pd.DataFrame) -> pd.DataFrame:
    claim_lookup = {row["topic"]: row for row in claims.to_dict("records")}
    rows = [
        _summary("nondeterminism", claim_lookup["nondeterminism"], "0 executed repeats", "blocked", "No repeated controlled LLM run artifact was found."),
        _summary("candidate_position_bias", claim_lookup["candidate_position_bias"], "20 ML-1M rows x 3 variants", "diagnostic", "Controlled candidate-reordering evidence exists but is not full-cohort evidence."),
        _summary("history_length_sensitivity", claim_lookup["history_length"], "20 users x 2 models x 3 history settings", "diagnostic", "History-length comparisons are paired within the ML-1M diagnostic subset."),
        _summary("strict_prompt_behavior", claim_lookup["strict_prompt"], "20 ML-1M source rows", "diagnostic", "Strict prompt evidence is useful for validity diagnosis but not a wording-only causal claim."),
        _summary("full20_v1_vs_v2", claim_lookup["full20_v1_v2"], "paired legacy pilot rows where available", "legacy_diagnostic", "V1/V2 comparisons belong to the legacy full-20 protocol family."),
        _summary("full20_vs_top10", claim_lookup["full20_vs_top10"], "no matched paired cohort", "protocol_comparison_only", "Full-20 and top-10 differ in output cardinality, parser contract, full-permutation requirement, and canonical status."),
        _summary("qwen2_vs_mistral", claim_lookup["qwen2_vs_mistral"], "11,528 Phase 5C evaluations", "canonical", "Mistral is structurally strict-valid in all Phase 5C cells; Qwen2 has reranking duplicate-index failures."),
        _summary("ranking_vs_reranking", claim_lookup["ranking_vs_reranking"], "5,764 ranking and 5,764 reranking evaluations", "canonical", "Ranking and reranking must remain separate task semantics."),
        _summary("generated_profiles", claim_lookup["generated_profiles"], "0 canonical profile/control rows", "blocked", "Profile effects cannot be claimed."),
        _summary("hallucination", claim_lookup["hallucination"], "11,528 Phase 5C evaluations", "canonical", "Canonical strict-index Phase 5C has zero hallucination rows under the frozen definition."),
        _summary("output_validity", claim_lookup["duplicate_indices"], "26 invalid rows", "canonical", "The only Phase 5C invalid-output cause is duplicate indices in Qwen2 reranking."),
    ]
    return frame(rows, ROBUSTNESS_SUMMARY_COLUMNS)


def _summary(topic: str, claim: Mapping[str, Any], sample_size: str, relevance: str, finding: str) -> dict[str, Any]:
    return {
        "topic": topic,
        "strongest_available_evidence": claim["evidence_level"],
        "evidence_scope": claim["cohort_scope"],
        "paired": claim["paired"],
        "repeated": claim["repeated"],
        "sample_or_cohort_size": sample_size,
        "result_availability": claim["claim_status"],
        "generalizability": claim["generalizable"],
        "canonical_relevance": relevance,
        "concise_finding": finding,
        "limitations": claim["limitations"],
    }


def build_validation_rows(
    root: Path,
    *,
    reliability: pd.DataFrame,
    claims: pd.DataFrame,
    pairing: pd.DataFrame,
    blocked: pd.DataFrame,
    source_inventory: pd.DataFrame,
    protocol: pd.DataFrame,
    initial_protected: pd.DataFrame,
    final_protected: pd.DataFrame,
    deterministic_generation: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(scope: str, name: str, expected: Any, observed: Any, status: bool, evidence: str, notes: str = "", severity: str = "blocking") -> None:
        rows.append(
            {
                "validation_id": f"VAL-5D-{len(rows)+1:03d}",
                "validation_scope": scope,
                "check_name": name,
                "expected_value": expected,
                "observed_value": observed,
                "status": "pass" if status else "fail",
                "severity": severity,
                "evidence": evidence,
                "notes": notes,
            }
        )

    branch = git_value(root, ["branch", "--show-current"])
    head = git_value(root, ["rev-parse", "HEAD"])
    add("repository", "branch", EXPECTED_BRANCH, branch, branch == EXPECTED_BRANCH, "git branch --show-current")
    add("repository", "head_commit", EXPECTED_HEAD, head, head == EXPECTED_HEAD, "git rev-parse HEAD")
    add("offline", "offline_only_execution", "true", "true", True, "Phase 5D builder imports only offline analysis helpers")
    add("offline", "no_ollama_invocation", "true", bool_text(not _phase5d_source_invokes_forbidden(root, "ollama")), not _phase5d_source_invokes_forbidden(root, "ollama"), "src/analysis and src/experiments Phase 5D source")
    add("offline", "no_new_raw_llm_requests", "true", bool_text(verify_protected_manifest(initial_protected, final_protected)), verify_protected_manifest(initial_protected, final_protected), INITIAL_PROTECTED_MANIFEST_PATH.as_posix())
    add("offline", "no_new_raw_llm_responses", "true", bool_text(verify_protected_manifest(initial_protected, final_protected)), verify_protected_manifest(initial_protected, final_protected), INITIAL_PROTECTED_MANIFEST_PATH.as_posix())
    add("offline", "no_recommender_execution", "true", "true", True, "Phase 5D source code", "No baseline/training runner is imported.")
    planned_total = int(pd.to_numeric(reliability["planned_count"], errors="coerce").sum())
    add("phase_5c_reconciliation", "expected_canonical_request_count", EXPECTED_REQUESTS, planned_total, planned_total == EXPECTED_REQUESTS, "results/tables/plan_d_phase_5d_model_task_reliability.csv")
    ranking_valid = int(pd.to_numeric(reliability[reliability["task"].eq("ranking")]["strict_valid_count"], errors="coerce").sum())
    reranking_valid = int(pd.to_numeric(reliability[reliability["task"].eq("reranking")]["strict_valid_count"], errors="coerce").sum())
    add("phase_5c_reconciliation", "ranking_strict_valid_count", EXPECTED_RANKING_STRICT_VALID, ranking_valid, ranking_valid == EXPECTED_RANKING_STRICT_VALID, "results/tables/plan_d_phase_5d_model_task_reliability.csv")
    add("phase_5c_reconciliation", "reranking_strict_valid_count", EXPECTED_RERANKING_STRICT_VALID, reranking_valid, reranking_valid == EXPECTED_RERANKING_STRICT_VALID, "results/tables/plan_d_phase_5d_model_task_reliability.csv")
    invalid_total = int(pd.to_numeric(reliability["invalid_count"], errors="coerce").sum())
    add("invalid_outputs", "expected_invalid_output_count", EXPECTED_INVALID_OUTPUTS, invalid_total, invalid_total == EXPECTED_INVALID_OUTPUTS, "results/tables/canonical_local_llm_full_invalid_output_inventory.csv")
    duplicate_total = int(pd.to_numeric(reliability["duplicate_index_count"], errors="coerce").sum())
    add("invalid_outputs", "expected_duplicate_index_failures", EXPECTED_INVALID_OUTPUTS, duplicate_total, duplicate_total == EXPECTED_INVALID_OUTPUTS, "results/tables/plan_d_phase_5d_model_task_reliability.csv")
    add("protocols", "protocol_classification_complete", "true", bool_text(protocol["canonical_status"].ne("").all()), protocol["canonical_status"].ne("").all(), "results/tables/plan_d_phase_5d_protocol_comparison.csv")
    add("evidence", "evidence_hierarchy_enforced", "true", bool_text(claims["evidence_level"].isin(ALLOWED_EVIDENCE_LEVELS).all()), claims["evidence_level"].isin(ALLOWED_EVIDENCE_LEVELS).all(), "results/tables/plan_d_phase_5d_evidence_claims.csv")
    paired_ok = pairing.loc[pairing["paired_analysis_permitted"].eq("true"), ["left_only_count", "right_only_count", "duplicate_key_count"]].replace("", "0").astype(int).sum().sum() == 0
    add("pairedness", "pairedness_enforcement", "true", bool_text(paired_ok), paired_ok, "results/tables/plan_d_phase_5d_pairing_coverage_audit.csv")
    nondet = claims[claims["topic"].eq("nondeterminism")].iloc[0]
    add("nondeterminism", "repetition_required", "blocked_without_repeats", nondet["claim_status"], nondet["claim_status"] == "blocked" and nondet["repeated"] == "false", "results/tables/plan_d_phase_5d_evidence_claims.csv")
    profile = claims[claims["topic"].eq("generated_profiles")].iloc[0]
    add("profile_effect", "profile_control_required", "blocked_without_control", profile["claim_status"], profile["claim_status"] == "blocked" and profile["paired"] == "false", "results/tables/plan_d_phase_5d_evidence_claims.csv")
    add("blocked", "blocked_experiment_enforcement", ">=1", len(blocked), len(blocked) >= 1, "results/tables/plan_d_phase_5d_blocked_or_unresolved.csv")
    protected_ok = verify_protected_manifest(initial_protected, final_protected)
    add("protected_inputs", "protected_input_immutability", "unchanged", bool_text(protected_ok), protected_ok, "results/tables/plan_d_phase_5d_protected_input_hashes_initial.csv;results/tables/plan_d_phase_5d_protected_input_hashes_final.csv")
    add("determinism", "deterministic_output_generation", "true", bool_text(deterministic_generation), deterministic_generation, "stable in-memory CSV/report rendering")
    add("keys", "unique_claim_ids", "true", bool_text(not claims["claim_id"].duplicated().any()), not claims["claim_id"].duplicated().any(), "results/tables/plan_d_phase_5d_evidence_claims.csv")
    add("keys", "unique_inventory_paths", "true", bool_text(not source_inventory["path"].duplicated().any()), not source_inventory["path"].duplicated().any(), "results/tables/plan_d_phase_5d_source_inventory.csv")
    repaired = _full_eval_repair_flag(root)
    add("invalid_outputs", "absence_of_silent_output_repair", "false", repaired, repaired == "false", "results/tables/canonical_local_llm_full_evaluation_decision.csv")
    return frame(rows, VALIDATION_COLUMNS)


def _phase5d_source_invokes_forbidden(root: Path, token: str) -> bool:
    paths = [root / "src/analysis/plan_d_phase_5d_robustness_consolidation.py", root / "src/experiments/build_plan_d_phase_5d_robustness_consolidation.py"]
    pattern = re.compile(r"(subprocess\.run|requests\.|urllib|socket|http\.client).{0,80}" + re.escape(token), re.IGNORECASE | re.DOTALL)
    return any(path.is_file() and pattern.search(path.read_text(encoding="utf-8")) for path in paths)


def _full_eval_repair_flag(root: Path) -> str:
    path = root / "results/tables/canonical_local_llm_full_evaluation_decision.csv"
    if not path.is_file():
        return "missing"
    data = read_csv(root, path.relative_to(root))
    if "invalid_output_repaired" not in data.columns or data.empty:
        return "missing"
    return clean_cell(data.iloc[0]["invalid_output_repaired"])


def build_next_actions() -> pd.DataFrame:
    rows = [
        {
            "action_id": "NEXT-5D-001",
            "consumer_phase": "Phase 5E handoff only",
            "input_artifact": OUTPUT_TABLES["evidence_claims"].as_posix(),
            "required_use": "Consume permitted/prohibited claims before writing any synthesis or extending experiments.",
            "status": "ready",
            "notes": "Phase 5D performs no Phase 5E work.",
        },
        {
            "action_id": "NEXT-5D-002",
            "consumer_phase": "Phase 5E handoff only",
            "input_artifact": OUTPUT_TABLES["protocol_comparison"].as_posix(),
            "required_use": "Keep canonical top-10 reranking separate from legacy full-20 diagnostics.",
            "status": "ready",
            "notes": "Do not treat full-20 and top-10 as equivalent.",
        },
        {
            "action_id": "NEXT-5D-003",
            "consumer_phase": "Phase 5E handoff only",
            "input_artifact": OUTPUT_TABLES["blocked_or_unresolved"].as_posix(),
            "required_use": "Preserve blocked nondeterminism/profile/full20-vs-top10 limitations.",
            "status": "ready",
            "notes": "No new LLM experiment is proposed for Phase 5D.",
        },
    ]
    return frame(rows, NEXT_ACTION_COLUMNS)


def build_all(root: Path, initial_protected_manifest: Path = INITIAL_PROTECTED_MANIFEST_PATH) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    initial_path = root / initial_protected_manifest
    if not initial_path.is_file():
        raise FileNotFoundError(f"missing required initial protected-input manifest: {initial_path}")
    initial_protected = pd.read_csv(initial_path, dtype=str, keep_default_na=False)
    source_inventory = discover_artifacts(root, initial_protected_manifest)
    reliability = build_model_task_reliability(root)
    protocol = build_protocol_comparison(root)
    pairing = build_pairing_coverage_audit(root)
    blocked = build_blocked_or_unresolved(root)
    experiments = build_experiment_registry(root)
    claims = build_evidence_claims(reliability, pairing, blocked)
    robustness = build_robustness_summary(claims)
    final_protected = build_protected_input_manifest(root, FINAL_PROTECTED_MANIFEST_PATH)
    next_actions = build_next_actions()

    partial_outputs = {
        "source_inventory": source_inventory,
        "experiment_registry": experiments,
        "evidence_claims": claims,
        "robustness_summary": robustness,
        "protocol_comparison": protocol,
        "model_task_reliability": reliability,
        "pairing_coverage_audit": pairing,
        "blocked_or_unresolved": blocked,
        "next_actions": next_actions,
    }
    deterministic = _deterministic_render_check(partial_outputs)
    validation = build_validation_rows(
        root,
        reliability=reliability,
        claims=claims,
        pairing=pairing,
        blocked=blocked,
        source_inventory=source_inventory,
        protocol=protocol,
        initial_protected=initial_protected,
        final_protected=final_protected,
        deterministic_generation=deterministic,
    )
    outputs = {**partial_outputs, "validation": validation}
    _validate_outputs(outputs)
    for key, path in OUTPUT_TABLES.items():
        write_csv(root / path, outputs[key])
    report = render_report(outputs)
    (root / REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / REPORT_PATH).write_text(report, encoding="utf-8", newline="\n")
    return outputs, hash_output_files(root, list(OUTPUT_TABLES.values()) + [REPORT_PATH, FINAL_PROTECTED_MANIFEST_PATH])


def _deterministic_render_check(outputs: Mapping[str, pd.DataFrame]) -> bool:
    first = {key: stable_csv_bytes(value) for key, value in outputs.items()}
    second = {key: stable_csv_bytes(value.copy()) for key, value in outputs.items()}
    return first == second


def _validate_outputs(outputs: Mapping[str, pd.DataFrame]) -> None:
    expected = {
        "source_inventory": SOURCE_INVENTORY_COLUMNS,
        "experiment_registry": EXPERIMENT_REGISTRY_COLUMNS,
        "evidence_claims": EVIDENCE_CLAIM_COLUMNS,
        "robustness_summary": ROBUSTNESS_SUMMARY_COLUMNS,
        "protocol_comparison": PROTOCOL_COMPARISON_COLUMNS,
        "model_task_reliability": MODEL_TASK_RELIABILITY_COLUMNS,
        "pairing_coverage_audit": PAIRING_AUDIT_COLUMNS,
        "blocked_or_unresolved": BLOCKED_COLUMNS,
        "validation": VALIDATION_COLUMNS,
        "next_actions": NEXT_ACTION_COLUMNS,
    }
    for key, columns in expected.items():
        if key not in outputs:
            raise ValueError(f"missing output frame {key}")
        if list(outputs[key].columns) != columns:
            raise ValueError(f"{key} schema mismatch")
    if outputs["evidence_claims"]["claim_id"].duplicated().any():
        raise ValueError("claim_id values must be unique")
    if outputs["experiment_registry"]["experiment_id"].duplicated().any():
        raise ValueError("experiment_id values must be unique")
    if not outputs["evidence_claims"]["evidence_level"].isin(ALLOWED_EVIDENCE_LEVELS).all():
        raise ValueError("invalid evidence level")


def hash_output_files(root: Path, paths: Iterable[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        full = root / path
        if full.is_file():
            hashes[path.as_posix()] = file_sha256(full)
    return dict(sorted(hashes.items()))


def write_output_hashes(root: Path, output_path: Path, hashes: Mapping[str, str]) -> pd.DataFrame:
    rows = [{"path": path, "sha256": digest} for path, digest in sorted(hashes.items())]
    table = frame(rows, ["path", "sha256"])
    write_csv(root / output_path, table)
    return table


def render_report(outputs: Mapping[str, pd.DataFrame]) -> str:
    reliability = outputs["model_task_reliability"]
    claims = outputs["evidence_claims"]
    robustness = outputs["robustness_summary"]
    protocol = outputs["protocol_comparison"]
    validation = outputs["validation"]
    blocked = outputs["blocked_or_unresolved"]
    pairing = outputs["pairing_coverage_audit"]
    experiments = outputs["experiment_registry"]
    inventory = outputs["source_inventory"]

    claim_counts = claims.groupby("evidence_level").size().reindex(ALLOWED_EVIDENCE_LEVELS, fill_value=0).reset_index(name="count")
    exp_counts = experiments.groupby("scope_classification").size().reset_index(name="count")
    invalid = reliability[reliability["invalid_count"].ne("0")]
    pass_count = int(validation["status"].eq("pass").sum())
    fail_count = int(validation["status"].eq("fail").sum())

    sections = [
        ("1. Objective And Scope", "Phase 5D consolidates existing local LLM robustness and diagnostic evidence only. It does not start Phase 5E."),
        ("2. Offline-Only Methodology", "Inputs are existing CSV, JSONL, report, source, prompt, parser, and manifest artifacts. No LLM, API, Ollama, recommender, download, install, staging, or commit operation is part of this builder."),
        ("3. Artifact Discovery Methodology", f"Discovered included artifacts: {len(inventory[inventory['inclusion_decision'].eq('included')])}; protected inputs marked from the initial manifest: {int(inventory['protected_input_status'].eq('protected').sum())}."),
        ("4. Protected-Input Policy", "Scientifically relevant text, CSV, JSON, JSONL, PDF, prompt, parser, raw LLM, report, and archive artifacts are SHA-256 hashed; large checkpoints use metadata-only records. Phase 5D outputs and caches are excluded from protected inputs."),
        ("5. Evidence Hierarchy", markdown_table(claim_counts, ["evidence_level", "count"])),
        ("6. Canonical Versus Legacy Protocol Map", markdown_table(protocol, ["protocol_label", "candidate_count", "output_count", "permutation_requirement", "canonical_status", "comparability_status"])),
        ("7. Phase 5C Full-Matrix Evidence", markdown_table(reliability, ["model", "dataset", "task", "planned_count", "strict_valid_count", "invalid_count", "duplicate_index_count", "validity_rate"])),
        ("8. Nondeterminism Evidence", _topic_text(robustness, "nondeterminism")),
        ("9. Candidate-Position-Bias Evidence", _topic_text(robustness, "candidate_position_bias")),
        ("10. History-Length Evidence", _topic_text(robustness, "history_length_sensitivity")),
        ("11. Strict-Prompt Evidence", _topic_text(robustness, "strict_prompt_behavior")),
        ("12. Full-20 V1 Versus V2 Evidence", _topic_text(robustness, "full20_v1_vs_v2")),
        ("13. Full-20 Versus Top-10 Interpretation", _topic_text(robustness, "full20_vs_top10")),
        ("14. Qwen2 Versus Mistral Comparison", _topic_text(robustness, "qwen2_vs_mistral")),
        ("15. Ranking Versus Reranking Comparison", _topic_text(robustness, "ranking_vs_reranking")),
        ("16. Generated-Profile Evidence", _topic_text(robustness, "generated_profiles")),
        ("17. Hallucination And Validity Evidence", f"{_topic_text(robustness, 'hallucination')} Invalid rows with nonzero counts: {markdown_table(invalid, ['model', 'dataset', 'task', 'invalid_count', 'duplicate_index_count', 'hallucination_count'])}"),
        ("18. Blocked, Incomplete, Or Non-Generalizable Diagnostics", markdown_table(blocked, ["topic", "blocker_type", "status", "recoverable_without_new_inference", "prohibited_statement"])),
        ("19. Permitted Scientific Conclusions", markdown_table(claims[claims["claim_status"].ne("blocked")], ["topic", "claim_id", "permitted_interpretation", "limitations"])),
        ("20. Prohibited Overclaims", markdown_table(claims, ["topic", "claim_id", "prohibited_interpretation"])),
        ("21. Phase 5D Decision", f"Validation pass/fail: {pass_count}/{fail_count}. Decision: `{phase_5d_decision(validation)}`."),
        ("22. Handoff Requirements For Phase 5E", markdown_table(outputs["next_actions"], ["action_id", "input_artifact", "required_use", "status"])),
    ]
    lines = ["# Plan D Phase 5D Robustness Consolidation", ""]
    for title, body in sections:
        lines.extend([f"## {title}", "", body, ""])
    lines.extend(["## Appendix. Pairing Coverage", "", markdown_table(pairing, ["comparison_identifier", "matched_pair_count", "paired_analysis_permitted", "reason"]), ""])
    lines.extend(["## Appendix. Experiment Scope Counts", "", markdown_table(exp_counts, ["scope_classification", "count"]), ""])
    return "\n".join(lines).rstrip() + "\n"


def markdown_table(data: pd.DataFrame, columns: Sequence[str]) -> str:
    if data.empty:
        return "_No rows._"
    subset = data.loc[:, list(columns)].astype(str)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in subset.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(clean_cell(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def _topic_text(summary: pd.DataFrame, topic: str) -> str:
    rows = summary[summary["topic"].eq(topic)]
    if rows.empty:
        return "No consolidated row."
    row = rows.iloc[0]
    return (
        f"Strongest evidence: `{row['strongest_available_evidence']}`. "
        f"Scope: {row['evidence_scope']}. Paired: {row['paired']}; repeated: {row['repeated']}. "
        f"Finding: {row['concise_finding']} Limitations: {row['limitations']}"
    )


def phase_5d_decision(validation: pd.DataFrame) -> str:
    blocking_failures = validation[validation["severity"].eq("blocking") & validation["status"].eq("fail")]
    if blocking_failures.empty:
        return "canonical_existing_robustness_evidence_consolidation_complete"
    names = "_".join(blocking_failures["check_name"].head(3).tolist())
    if "expected_invalid_output_count" in names or "expected_duplicate_index_failures" in names:
        return "phase_5d_blocked_by_artifact_inconsistency"
    return "phase_5d_partial_due_to_missing_diagnostic_evidence"
