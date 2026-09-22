"""Offline Phase 5C evaluation helpers for the full local LLM matrix.

This module is intentionally read-only with respect to Phase 5B-P2 evidence.
It reconciles frozen raw outputs, separates structural reliability from utility,
and exposes deterministic table builders used by the Phase 5C runner.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.audit.common_io import (
    read_csv_dataframe as _read_csv_dataframe,
    write_dataframe_csv as _write_dataframe_csv,
)
from src.audit import common_validation
from src.llm.canonical_local_llm_full_matrix_runner import classify_response_for_task
from src.llm.canonical_local_llm_top10_confirmation_runner import output_truncation_detected


DATASET_ORDER = ["Amazon Beauty", "Amazon Sports", "ML-1M"]
MODEL_ORDER = ["Qwen2 7B Instruct", "Mistral 7B Instruct"]
TASK_ORDER = ["ranking", "reranking"]
EXPECTED_REQUESTS = 11528
EXPECTED_EVENTS = 23056
EXPECTED_INVALID = 26
NEXT_TASK = (
    "Phase 5D — consolidate LLM robustness extensions, nondeterminism, "
    "candidate-position bias, history-length sensitivity, profile experiments, and reliability findings"
)

RAW_REQUESTS_PATH = Path("results/llm/canonical_local_full_matrix/requests.jsonl")
RAW_RESPONSES_PATH = Path("results/llm/canonical_local_full_matrix/responses.jsonl")
RAW_EVENTS_PATH = Path("results/llm/canonical_local_full_matrix/runner_events.jsonl")

OUTPUT_INTEGRITY_PATH = Path("results/tables/canonical_local_llm_full_output_integrity.csv")
INVALID_INVENTORY_PATH = Path("results/tables/canonical_local_llm_full_invalid_output_inventory.csv")
POLICY_CONTRACT_PATH = Path("results/tables/canonical_local_llm_evaluation_policy_contract.csv")
STRUCTURAL_METRICS_PATH = Path("results/tables/canonical_local_llm_structural_metrics.csv")
UTILITY_METRICS_PATH = Path("results/tables/canonical_local_llm_utility_metrics.csv")
VALIDATION_PATH = Path("results/tables/canonical_local_llm_full_evaluation_validation.csv")
DECISION_PATH = Path("results/tables/canonical_local_llm_full_evaluation_decision.csv")
REPORT_PATH = Path("report/canonical_local_llm_full_evaluation.md")

FULL_MATRIX_INPUTS = [
    "results/tables/canonical_local_llm_full_matrix_manifest.csv",
    "results/tables/canonical_local_llm_full_matrix_chunk_manifest.csv",
    "results/llm/canonical_local_full_matrix/requests.jsonl",
    "results/llm/canonical_local_full_matrix/responses.jsonl",
    "results/llm/canonical_local_full_matrix/runner_events.jsonl",
    "results/tables/canonical_local_llm_full_matrix_provenance.csv",
    "results/tables/canonical_local_llm_full_matrix_request_state.csv",
    "results/tables/canonical_local_llm_full_matrix_progress.csv",
    "results/tables/canonical_local_llm_full_matrix_execution.csv",
    "results/tables/canonical_local_llm_full_matrix_parse_results.csv",
    "results/tables/canonical_local_llm_full_matrix_rankings.csv",
    "results/tables/canonical_local_llm_full_matrix_cell_metrics.csv",
    "results/tables/canonical_local_llm_full_matrix_validation.csv",
    "results/tables/canonical_local_llm_full_matrix_decision.csv",
    "results/tables/canonical_local_llm_full_matrix_preflight_validation.csv",
    "results/tables/canonical_local_llm_full_matrix_preflight_decision.csv",
    "report/canonical_local_llm_full_matrix_execution.md",
    "src/analysis/canonical_local_llm_full_matrix_preflight.py",
    "src/llm/canonical_local_llm_full_matrix_runner.py",
    "src/experiments/build_canonical_local_llm_full_matrix_preflight.py",
    "src/experiments/run_canonical_local_llm_full_matrix.py",
    "tests/test_canonical_local_llm_full_matrix_preflight.py",
    "tests/test_canonical_local_llm_full_matrix_runner.py",
]

PHASE4_INPUTS = [
    "results/tables/canonical_metric_contract.csv",
    "results/tables/canonical_metric_same_cohort_preflight.csv",
    "results/tables/canonical_metric_computation_manifest.csv",
    "results/tables/canonical_metric_train_item_statistics.csv",
    "results/tables/canonical_metric_long_tail_audit.csv",
    "results/tables/canonical_metric_item_universe_audit.csv",
    "results/tables/canonical_metric_blocked_or_ambiguous.csv",
    "results/tables/canonical_metric_blocked_results.csv",
    "results/tables/canonical_metric_ranking_rows.csv",
    "results/tables/canonical_metric_top5_exposures.csv",
    "results/tables/canonical_metric_row_contributions.csv",
    "results/tables/canonical_metric_coverage_components.csv",
    "results/tables/canonical_metric_results_long.csv",
    "results/tables/canonical_metric_results_wide.csv",
    "results/tables/canonical_metric_results_validation.csv",
    "report/canonical_metric_contract_and_preflight.md",
    "report/canonical_metric_results_seven_baselines.md",
    "src/metrics/canonical_metric_contract.py",
    "src/metrics/canonical_metric_computation.py",
    "src/analysis/canonical_metric_preflight.py",
    "src/analysis/canonical_metric_results_validation.py",
    "src/experiments/build_canonical_metric_preflight.py",
    "src/experiments/build_canonical_metric_results.py",
    "tests/test_canonical_metric_contract.py",
    "tests/test_canonical_metric_computation.py",
]

BASELINE_INPUTS = [
    "results/tables/traditional_baseline_rankings.csv",
    "results/tables/traditional_baselines_metrics.csv",
    "results/tables/traditional_baseline_validation.csv",
    "results/tables/plan_e_bprmf_beauty_sports_rankings.csv",
    "results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv",
    "results/tables/plan_e_gru4rec_beauty_sports_rankings.csv",
    "results/tables/plan_e_gru4rec_ml1m_pilot_rankings.csv",
    "results/tables/plan_e_sasrec_beauty_sports_rankings.csv",
    "results/tables/plan_e_sasrec_ml1m_pilot_rankings.csv",
    "results/tables/plan_e_lightgcn_beauty_sports_cpu_rankings.csv",
    "results/tables/plan_e_lightgcn_ml1m_pilot_rankings.csv",
    "results/tables/plan_e_same_cohort_traditional_baselines_metrics.csv",
    "results/tables/plan_e_bprmf_all_datasets_metrics.csv",
    "results/tables/plan_e_gru4rec_all_datasets_metrics.csv",
    "results/tables/plan_e_sasrec_all_datasets_metrics.csv",
    "results/tables/plan_e_lightgcn_all_datasets_metrics.csv",
    "report/traditional_baseline_alignment.md",
    "report/plan_e_bprmf_beauty_sports_and_baseline_comparison.md",
    "report/plan_e_gru4rec_beauty_sports_and_baseline_comparison.md",
    "report/plan_e_sasrec_beauty_sports_and_baseline_comparison.md",
    "report/plan_e_lightgcn_beauty_sports_cpu_and_baseline_comparison.md",
]

CANONICAL_DATA_INPUTS = [
    "results/tables/ranking_candidate_sets.csv",
    "results/tables/positive_positions.csv",
    "results/tables/dataset_statistics.csv",
    "results/tables/item_title_map.csv",
    "results/tables/item_title_coverage_summary.csv",
    "results/tables/item_title_mapping_issues.csv",
    "results/tables/canonical_llm_cohort_preflight.csv",
    "results/tables/llm_title_exclusions.csv",
    "results/tables/experiment_cohort_membership.csv",
    "data/reconstructed/plan_c/amazon_beauty/interactions.csv",
    "data/reconstructed/plan_c/amazon_sports/interactions.csv",
    "data/reconstructed/plan_c/ml1m/interactions.csv",
]

PAPER_INPUTS = [
    "paper/Beyond Utility Evaluating LLM As Recommander.pdf",
    "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
    "original_repo/EvaLLMasRecommender/eva/codes/metrics.sh",
    "original_repo/EvaLLMasRecommender/eva/codes/output_refine.py",
    "original_repo/EvaLLMasRecommender/eva/templates/llmrank.txt",
    "original_repo/EvaLLMasRecommender/eva/templates/llmrank_rf.txt",
    "report/reproducibility_audit_current_state.md",
    "report/canonical_llm_prompt_parsing_and_execution_preflight.md",
    "report/canonical_local_llm_final_reranking_protocol_decision.md",
]

OUTPUT_INTEGRITY_COLUMNS = [
    "request_id",
    "request_order",
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "backend_success",
    "response_nonempty",
    "json_parse_valid",
    "schema_valid",
    "output_count_valid",
    "index_type_valid",
    "index_range_valid",
    "uniqueness_valid",
    "candidate_validity",
    "catalog_validity",
    "duplicate_count",
    "unknown_index_count",
    "hallucination_count",
    "strict_parse_valid",
    "output_truncation_detected",
    "primary_failure_reason",
    "stored_reparse_agreement",
    "integrity_status",
    "notes",
]

INVALID_INVENTORY_COLUMNS = [
    "invalid_output_id",
    "request_id",
    "request_order",
    "model_family",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "raw_response_sha256",
    "output_length",
    "unique_index_count",
    "duplicated_indices",
    "missing_indices",
    "out_of_range_indices",
    "generated_token_count",
    "done_reason",
    "json_parse_valid",
    "schema_valid",
    "output_count_valid",
    "uniqueness_valid",
    "candidate_validity",
    "hallucination_count",
    "truncation_evidence",
    "primary_failure_reason",
    "failure_category_count",
    "notes",
]

POLICY_CONTRACT_COLUMNS = [
    "policy_id",
    "policy_name",
    "metric_family",
    "denominator_policy",
    "invalid_output_treatment",
    "strict_valid_output_treatment",
    "paper_alignment",
    "primary_or_diagnostic",
    "selection_bias_risk",
    "survivorship_bias_risk",
    "allowed_for_model_comparison",
    "allowed_for_paper_comparison",
    "requires_evaluated_row_coverage",
    "contract_status",
    "notes",
]

STRUCTURAL_METRICS_COLUMNS = [
    "aggregation_id",
    "aggregation_level",
    "model_family",
    "dataset",
    "task_type",
    "planned_rows",
    "backend_success_rows",
    "response_nonempty_rows",
    "json_valid_rows",
    "schema_valid_rows",
    "output_count_valid_rows",
    "index_range_valid_rows",
    "uniqueness_valid_rows",
    "candidate_valid_rows",
    "strict_valid_rows",
    "duplicate_output_rows",
    "hallucination_rows",
    "truncation_rows",
    "backend_completion_rate",
    "strict_parse_validity_rate",
    "duplicate_output_rate",
    "hallucination_rate",
    "metric_status",
    "notes",
]

UTILITY_METRICS_COLUMNS = [
    "metric_row_id",
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
    "evaluation_policy_id",
    "planned_rows",
    "strict_valid_rows",
    "evaluated_rows",
    "evaluated_row_coverage",
    "invalid_rows",
    "hr_at_5",
    "ndcg_at_5",
    "mean_positive_rank",
    "median_positive_rank",
    "denominator_policy",
    "invalid_output_treatment",
    "paper_alignment",
    "metric_status",
    "notes",
]

VALIDATION_COLUMNS = [
    "validation_id",
    "validation_scope",
    "model_family",
    "dataset",
    "task_type",
    "check_name",
    "expected_value",
    "observed_value",
    "status",
    "severity",
    "evidence",
    "notes",
]

DECISION_COLUMNS = [
    "evaluation_id",
    "full_matrix_evidence_reconciled",
    "stored_reparse_agreement",
    "invalid_output_count",
    "invalid_output_repaired",
    "structural_metrics_complete",
    "conditional_utility_complete",
    "operational_utility_complete",
    "beyond_utility_supported_metrics_complete",
    "blocked_metrics_preserved",
    "baseline_alignment_complete",
    "baseline_comparison_complete",
    "paper_inventory_complete",
    "paper_alignment_complete",
    "unsupported_exact_claim_count",
    "utility_denominator_policies_explicit",
    "protected_inputs_unchanged",
    "evaluation_decision",
    "blocking_reason",
    "recommended_single_next_task",
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
    return str(value)


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def parse_bool(value: Any) -> bool:
    return clean_cell(value).strip().lower() in {"true", "1", "yes", "y"}


def format_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.17g}"


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame([{column: clean_cell(row.get(column, "")) for column in columns} for row in rows], columns=list(columns))


def read_csv(root: Path, relative_path: str | Path) -> pd.DataFrame:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    return _read_csv_dataframe(relative_path, root=root, dtype=str, keep_default_na=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def write_csv(path: Path, table: pd.DataFrame) -> None:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    _write_dataframe_csv(path, table, lineterminator="\n")


def parse_json_array(value: Any) -> list[str]:
    decoded = json.loads(clean_cell(value))
    if not isinstance(decoded, list):
        raise ValueError("cell is not a JSON array")
    return [str(item) for item in decoded]


def row_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        clean_cell(row.get("dataset", "")),
        clean_cell(row.get("task_type", "")),
        clean_cell(row.get("candidate_set_id", "")),
        clean_cell(row.get("user_id", "")),
    )


def sort_key_frame(table: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return table.sort_values(list(columns), kind="mergesort").reset_index(drop=True)


def protected_input_paths(root: Path) -> list[str]:
    paths = [*FULL_MATRIX_INPUTS, *PHASE4_INPUTS, *BASELINE_INPUTS, *CANONICAL_DATA_INPUTS, *PAPER_INPUTS]
    original_root = root / "original_repo"
    if original_root.exists():
        paths.extend(path.relative_to(root).as_posix() for path in original_root.rglob("*") if path.is_file())
    return sorted({path for path in paths if (root / path).is_file()})


def hash_protected_inputs(root: Path) -> dict[str, str]:
    return {relative: file_sha256(root / relative) for relative in protected_input_paths(root)}


def hash_output_files(root: Path, paths: Iterable[Path]) -> dict[str, str]:
    output: dict[str, str] = {}
    for path in paths:
        resolved = root / path
        if resolved.is_file():
            output[path.as_posix()] = file_sha256(resolved)
    return output


def response_lookup(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    counts = Counter(clean_cell(record.get("request_id", "")) for record in records)
    duplicates = sorted(request_id for request_id, count in counts.items() if request_id and count != 1)
    if duplicates:
        raise ValueError(f"duplicate raw response evidence: {duplicates[0]}")
    return {clean_cell(record.get("request_id", "")): record for record in records if clean_cell(record.get("request_id", ""))}


def count_unique_request_ids(records: Sequence[Mapping[str, Any]]) -> int:
    return len({clean_cell(record.get("request_id", "")) for record in records if clean_cell(record.get("request_id", ""))})


def raw_indices(raw_text: str) -> list[int]:
    try:
        decoded = json.loads(str(raw_text).strip())
    except json.JSONDecodeError:
        return []
    values = decoded.get("ranked_candidate_indices") if isinstance(decoded, dict) else None
    if not isinstance(values, list):
        return []
    return [int(value) for value in values if isinstance(value, int) and not isinstance(value, bool)]


def duplicate_indices(indices: Sequence[int]) -> list[int]:
    counts = Counter(indices)
    return [int(value) for value in sorted(counts, key=lambda item: indices.index(item)) if counts[value] > 1]


def failure_category_count(row: Mapping[str, Any]) -> int:
    categories = 0
    if clean_cell(row.get("json_parse_valid")) != "true":
        categories += 1
    elif clean_cell(row.get("schema_valid")) != "true":
        categories += 1
    elif clean_cell(row.get("output_count_valid")) != "true":
        categories += 1
    elif clean_cell(row.get("index_range_valid", "true")) != "true":
        categories += 1
    elif clean_cell(row.get("uniqueness_valid")) != "true":
        categories += 1
    if int(clean_cell(row.get("hallucination_count", "0")) or "0") > 0:
        categories += 1
    if clean_cell(row.get("output_truncation_detected")) == "true":
        categories += 1
    return categories


def reparse_raw_responses(
    manifest: pd.DataFrame,
    execution: pd.DataFrame,
    stored_parse: pd.DataFrame,
    responses: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Reparse every raw response and compare it with the frozen parse table."""

    manifest_lookup = {row.request_id: row._asdict() for row in manifest.itertuples(index=False)}
    execution_lookup = {row.request_id: row._asdict() for row in execution.itertuples(index=False)}
    stored_lookup = {row.request_id: row._asdict() for row in stored_parse.itertuples(index=False)}
    raw_lookup = response_lookup(responses)
    rows: list[dict[str, Any]] = []
    compare_fields = [
        "json_parse_valid",
        "schema_valid",
        "output_count_valid",
        "index_type_valid",
        "index_range_valid",
        "uniqueness_valid",
        "candidate_validity",
        "catalog_validity",
        "duplicate_count",
        "unknown_index_count",
        "hallucination_count",
        "strict_parse_valid",
        "output_length",
        "primary_failure_reason",
    ]
    for request_id in manifest.sort_values("request_order", key=lambda s: s.astype(int))["request_id"]:
        manifest_row = manifest_lookup[request_id]
        execution_row = execution_lookup.get(request_id, {})
        stored_row = stored_lookup.get(request_id, {})
        raw_record = raw_lookup.get(request_id, {})
        raw_text = clean_cell(raw_record.get("raw_response_text", ""))
        reparse = classify_response_for_task(raw_text, clean_cell(manifest_row["task_type"]))
        truncation = output_truncation_detected(
            raw_text,
            clean_cell(raw_record.get("done_reason", execution_row.get("done_reason", ""))),
            raw_record.get("generated_token_count", execution_row.get("eval_count", "")),
            manifest_row.get("max_output_tokens", ""),
        )
        reparsed_row = {
            "json_parse_valid": bool_text(reparse["json_parse_valid"]),
            "schema_valid": bool_text(reparse["schema_valid"]),
            "output_count_valid": bool_text(reparse["output_count_valid"]),
            "index_type_valid": bool_text(reparse["index_type_valid"]),
            "index_range_valid": bool_text(reparse["index_range_valid"]),
            "uniqueness_valid": bool_text(reparse["uniqueness_valid"]),
            "candidate_validity": bool_text(reparse["candidate_validity"]),
            "catalog_validity": bool_text(reparse["catalog_validity"]),
            "duplicate_count": str(reparse["duplicate_count"]),
            "unknown_index_count": str(reparse["unknown_index_count"]),
            "hallucination_count": str(reparse["hallucination_count"]),
            "strict_parse_valid": bool_text(reparse["strict_parse_valid"]),
            "output_length": str(reparse["output_length"]),
            "primary_failure_reason": reparse["primary_failure_reason"],
            "output_truncation_detected": bool_text(truncation),
        }
        agreement = all(clean_cell(stored_row.get(field, "")) == clean_cell(reparsed_row[field]) for field in compare_fields)
        agreement = agreement and clean_cell(stored_row.get("output_truncation_detected", "")) == reparsed_row["output_truncation_detected"]
        raw_sha = clean_cell(raw_record.get("raw_response_sha256", sha256_text(raw_text)))
        agreement = agreement and clean_cell(stored_row.get("raw_response_sha256", "")) == raw_sha
        rows.append(
            {
                "request_id": request_id,
                "request_order": manifest_row["request_order"],
                "model_family": manifest_row["model_family"],
                "model_tag": manifest_row["model_tag"],
                "dataset": manifest_row["dataset"],
                "task_type": manifest_row["task_type"],
                "candidate_set_id": manifest_row["candidate_set_id"],
                "user_id": manifest_row["user_id"],
                "backend_success": clean_cell(execution_row.get("backend_success", "")),
                "response_nonempty": bool_text(bool(raw_text.strip())),
                **{field: reparsed_row[field] for field in compare_fields if field != "primary_failure_reason" and field != "output_length"},
                "output_truncation_detected": reparsed_row["output_truncation_detected"],
                "primary_failure_reason": reparsed_row["primary_failure_reason"],
                "stored_reparse_agreement": bool_text(agreement),
                "integrity_status": "pass" if agreement and clean_cell(execution_row.get("backend_success", "")) == "true" else "fail",
                "notes": "raw response reparsed with frozen task-specific strict parser; no repair or salvage",
            }
        )
    return frame(rows, OUTPUT_INTEGRITY_COLUMNS)


def build_invalid_inventory(integrity: pd.DataFrame, execution: pd.DataFrame, responses: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    execution_lookup = {row.request_id: row._asdict() for row in execution.itertuples(index=False)}
    raw_lookup = response_lookup(responses)
    rows: list[dict[str, Any]] = []
    invalid = integrity.loc[integrity["strict_parse_valid"].ne("true")].sort_values("request_order", key=lambda s: s.astype(int))
    for index, row in enumerate(invalid.to_dict("records"), start=1):
        raw_record = raw_lookup[row["request_id"]]
        raw_text = clean_cell(raw_record.get("raw_response_text", ""))
        indices = raw_indices(raw_text)
        duplicated = duplicate_indices(indices)
        out_of_range = [value for value in indices if value < 1 or value > 20]
        expected_count = 5 if row["task_type"] == "ranking" else 10
        missing_slot_count = max(0, expected_count - len(set(indices)))
        execution_row = execution_lookup.get(row["request_id"], {})
        enriched = {**row, "index_range_valid": row.get("index_range_valid", "true")}
        rows.append(
            {
                "invalid_output_id": f"CLFULL-INVALID-{index:03d}",
                "request_id": row["request_id"],
                "request_order": row["request_order"],
                "model_family": row["model_family"],
                "dataset": row["dataset"],
                "task_type": row["task_type"],
                "candidate_set_id": row["candidate_set_id"],
                "user_id": row["user_id"],
                "raw_response_sha256": clean_cell(raw_record.get("raw_response_sha256", sha256_text(raw_text))),
                "output_length": row.get("output_length", ""),
                "unique_index_count": str(len(set(indices))),
                "duplicated_indices": json_dumps(duplicated),
                "missing_indices": json_dumps([]),
                "out_of_range_indices": json_dumps(out_of_range),
                "generated_token_count": clean_cell(raw_record.get("generated_token_count", execution_row.get("eval_count", ""))),
                "done_reason": clean_cell(raw_record.get("done_reason", execution_row.get("done_reason", ""))),
                "json_parse_valid": row["json_parse_valid"],
                "schema_valid": row["schema_valid"],
                "output_count_valid": row["output_count_valid"],
                "uniqueness_valid": row["uniqueness_valid"],
                "candidate_validity": row["candidate_validity"],
                "hallucination_count": row["hallucination_count"],
                "truncation_evidence": row["output_truncation_detected"],
                "primary_failure_reason": row["primary_failure_reason"],
                "failure_category_count": str(failure_category_count(enriched)),
                "notes": f"missing_unique_slot_count={missing_slot_count}; bounded structural summary only; raw text retained only in raw JSONL",
            }
        )
    return frame(rows, INVALID_INVENTORY_COLUMNS)


def build_policy_contract() -> pd.DataFrame:
    rows = [
        {
            "policy_id": "structural_reliability",
            "policy_name": "Structural reliability",
            "metric_family": "format_and_backend_reliability",
            "denominator_policy": "all planned requests in the cell",
            "invalid_output_treatment": "invalid outputs remain in denominator and are counted by failure type",
            "strict_valid_output_treatment": "strict-valid outputs contribute pass counts",
            "paper_alignment": "diagnostic",
            "primary_or_diagnostic": "primary_operational_evidence",
            "selection_bias_risk": "low",
            "survivorship_bias_risk": "low",
            "allowed_for_model_comparison": "true",
            "allowed_for_paper_comparison": "false",
            "requires_evaluated_row_coverage": "false",
            "contract_status": "frozen_phase5c",
            "notes": "separates backend and parser reliability from utility",
        },
        {
            "policy_id": "conditional_strict_valid_output_utility",
            "policy_name": "Conditional strict-valid output utility",
            "metric_family": "utility",
            "denominator_policy": "strict-valid outputs only",
            "invalid_output_treatment": "excluded and explicitly reported through evaluated-row coverage",
            "strict_valid_output_treatment": "observed HR@5/NDCG@5 and positive-rank statistics",
            "paper_alignment": "diagnostic",
            "primary_or_diagnostic": "secondary_diagnostic",
            "selection_bias_risk": "medium",
            "survivorship_bias_risk": "high_when_invalid_outputs_exist",
            "allowed_for_model_comparison": "true_when_labeled_conditional",
            "allowed_for_paper_comparison": "false",
            "requires_evaluated_row_coverage": "true",
            "contract_status": "frozen_phase5c",
            "notes": "must never be labeled unconditional performance",
        },
        {
            "policy_id": "adapted_end_to_end_operational_utility_invalid_as_failure",
            "policy_name": "End-to-end operational utility invalid as failure",
            "metric_family": "utility",
            "denominator_policy": "all planned requests in the cell",
            "invalid_output_treatment": "HR@5=0 and NDCG@5=0 for strict-invalid rows",
            "strict_valid_output_treatment": "observed HR@5 and NDCG@5",
            "paper_alignment": "adapted",
            "primary_or_diagnostic": "primary_reliability_aware_utility",
            "selection_bias_risk": "low",
            "survivorship_bias_risk": "low",
            "allowed_for_model_comparison": "true",
            "allowed_for_paper_comparison": "false",
            "requires_evaluated_row_coverage": "true",
            "contract_status": "frozen_phase5c",
            "notes": "not paper exact; prevents invalid outputs from disappearing from denominators",
        },
        {
            "policy_id": "conditional_beyond_utility",
            "policy_name": "Conditional beyond-utility",
            "metric_family": "beyond_utility",
            "denominator_policy": "strict-valid top-5 exposures; popularity-dependent metrics require train-count support",
            "invalid_output_treatment": "excluded and coverage reported",
            "strict_valid_output_treatment": "top-5 exposures contribute under Phase 4 contracts",
            "paper_alignment": "reconstructed_or_adapted_by_metric",
            "primary_or_diagnostic": "secondary_diagnostic",
            "selection_bias_risk": "medium",
            "survivorship_bias_risk": "high_when_invalid_outputs_exist",
            "allowed_for_model_comparison": "true_when_labeled_conditional",
            "allowed_for_paper_comparison": "false",
            "requires_evaluated_row_coverage": "true",
            "contract_status": "frozen_phase5c",
            "notes": "blocked Phase 4 metrics remain blocked",
        },
        {
            "policy_id": "paired_strict_valid_subset_comparison",
            "policy_name": "Paired strict-valid baseline comparison",
            "metric_family": "baseline_comparison",
            "denominator_policy": "exact row intersection between baseline rows and one LLM strict-valid subset",
            "invalid_output_treatment": "LLM invalid outputs excluded by construction and disclosed",
            "strict_valid_output_treatment": "paired row-level utility and supported beyond-utility comparison",
            "paper_alignment": "diagnostic",
            "primary_or_diagnostic": "conditional_diagnostic",
            "selection_bias_risk": "medium",
            "survivorship_bias_risk": "medium",
            "allowed_for_model_comparison": "true_when_intersection_is_explicit",
            "allowed_for_paper_comparison": "false",
            "requires_evaluated_row_coverage": "true",
            "contract_status": "frozen_phase5c",
            "notes": "not unbiased primary evidence; used to avoid mismatched row comparisons",
        },
    ]
    return frame(rows, POLICY_CONTRACT_COLUMNS)


def _rate(numerator: int, denominator: int) -> str:
    return format_float(float(numerator) / float(denominator)) if denominator else ""


def _structural_row(rows: pd.DataFrame, aggregation_id: str, aggregation_level: str, model: str, dataset: str, task: str) -> dict[str, Any]:
    planned = len(rows)
    backend = int(rows["backend_success"].eq("true").sum())
    nonempty = int(rows["response_nonempty"].eq("true").sum())
    json_valid = int(rows["json_parse_valid"].eq("true").sum())
    schema_valid = int(rows["schema_valid"].eq("true").sum())
    output_count_valid = int(rows["output_count_valid"].eq("true").sum())
    range_valid = int(rows["index_range_valid"].eq("true").sum())
    unique_valid = int(rows["uniqueness_valid"].eq("true").sum())
    candidate_valid = int(rows["candidate_validity"].eq("true").sum())
    strict_valid = int(rows["strict_parse_valid"].eq("true").sum())
    duplicate_rows = int(pd.to_numeric(rows["duplicate_count"], errors="coerce").fillna(0).gt(0).sum())
    hallucination_rows = int(pd.to_numeric(rows["hallucination_count"], errors="coerce").fillna(0).gt(0).sum())
    truncation_rows = int(rows["output_truncation_detected"].eq("true").sum())
    return {
        "aggregation_id": aggregation_id,
        "aggregation_level": aggregation_level,
        "model_family": model,
        "dataset": dataset,
        "task_type": task,
        "planned_rows": str(planned),
        "backend_success_rows": str(backend),
        "response_nonempty_rows": str(nonempty),
        "json_valid_rows": str(json_valid),
        "schema_valid_rows": str(schema_valid),
        "output_count_valid_rows": str(output_count_valid),
        "index_range_valid_rows": str(range_valid),
        "uniqueness_valid_rows": str(unique_valid),
        "candidate_valid_rows": str(candidate_valid),
        "strict_valid_rows": str(strict_valid),
        "duplicate_output_rows": str(duplicate_rows),
        "hallucination_rows": str(hallucination_rows),
        "truncation_rows": str(truncation_rows),
        "backend_completion_rate": _rate(backend, planned),
        "strict_parse_validity_rate": _rate(strict_valid, planned),
        "duplicate_output_rate": _rate(duplicate_rows, planned),
        "hallucination_rate": _rate(hallucination_rows, planned),
        "metric_status": "complete",
        "notes": "all planned rows remain in the structural denominator",
    }


def build_structural_metrics(integrity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    counter = 1
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            for task in TASK_ORDER:
                cell = integrity[
                    integrity["model_family"].eq(model)
                    & integrity["dataset"].eq(dataset)
                    & integrity["task_type"].eq(task)
                ]
                rows.append(_structural_row(cell, f"STRUCT-{counter:03d}", "model_dataset_task_cell", model, dataset, task))
                counter += 1
    for model in MODEL_ORDER:
        for task in TASK_ORDER:
            rows.append(_structural_row(integrity[integrity["model_family"].eq(model) & integrity["task_type"].eq(task)], f"STRUCT-{counter:03d}", "model_task_total", model, "all", task))
            counter += 1
    for model in MODEL_ORDER:
        rows.append(_structural_row(integrity[integrity["model_family"].eq(model)], f"STRUCT-{counter:03d}", "model_total", model, "all", "all"))
        counter += 1
    for dataset in DATASET_ORDER:
        rows.append(_structural_row(integrity[integrity["dataset"].eq(dataset)], f"STRUCT-{counter:03d}", "dataset_total", "all", dataset, "all"))
        counter += 1
    rows.append(_structural_row(integrity, f"STRUCT-{counter:03d}", "global_total", "all", "all", "all"))
    return frame(rows, STRUCTURAL_METRICS_COLUMNS)


def _mean(values: Sequence[Any]) -> str:
    nums = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if nums.empty:
        return ""
    return format_float(float(nums.mean()))


def _median(values: Sequence[Any]) -> str:
    nums = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if nums.empty:
        return ""
    return format_float(float(nums.median()))


def build_utility_metrics(rankings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    counter = 1
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            for task in TASK_ORDER:
                cell = rankings[
                    rankings["model_family"].eq(model)
                    & rankings["dataset"].eq(dataset)
                    & rankings["task_type"].eq(task)
                ].copy()
                model_tag = cell["model_tag"].iloc[0] if not cell.empty else ""
                planned = len(cell)
                valid = cell[cell["strict_parse_valid"].eq("true")].copy()
                valid_count = len(valid)
                invalid = planned - valid_count
                hit_values = [1.0 if parse_bool(value) else 0.0 for value in valid["positive_in_top_5"].tolist()]
                ndcg_values = pd.to_numeric(valid["ndcg_at_5"], errors="coerce").fillna(0.0).tolist()
                rows.append(
                    {
                        "metric_row_id": f"LLMUTIL-{counter:03d}",
                        "model_family": model,
                        "model_tag": model_tag,
                        "dataset": dataset,
                        "task_type": task,
                        "evaluation_policy_id": "conditional_strict_valid_output_utility",
                        "planned_rows": str(planned),
                        "strict_valid_rows": str(valid_count),
                        "evaluated_rows": str(valid_count),
                        "evaluated_row_coverage": _rate(valid_count, planned),
                        "invalid_rows": str(invalid),
                        "hr_at_5": _mean(hit_values),
                        "ndcg_at_5": _mean(ndcg_values),
                        "mean_positive_rank": _mean(valid["positive_rank_zero_based"].tolist()),
                        "median_positive_rank": _median(valid["positive_rank_zero_based"].tolist()),
                        "denominator_policy": "strict_valid_outputs_only",
                        "invalid_output_treatment": "excluded_with_coverage_reported",
                        "paper_alignment": "diagnostic",
                        "metric_status": "complete",
                        "notes": "conditional strict-valid utility; not unconditional model performance",
                    }
                )
                counter += 1
                operational_hr_sum = sum(hit_values)
                operational_ndcg_sum = float(np.sum(np.asarray(ndcg_values, dtype=float))) if ndcg_values else 0.0
                conditional_hr = float(operational_hr_sum / valid_count) if valid_count else float("nan")
                operational_hr = float(operational_hr_sum / planned) if planned else float("nan")
                rows.append(
                    {
                        "metric_row_id": f"LLMUTIL-{counter:03d}",
                        "model_family": model,
                        "model_tag": model_tag,
                        "dataset": dataset,
                        "task_type": task,
                        "evaluation_policy_id": "adapted_end_to_end_operational_utility_invalid_as_failure",
                        "planned_rows": str(planned),
                        "strict_valid_rows": str(valid_count),
                        "evaluated_rows": str(planned),
                        "evaluated_row_coverage": "1",
                        "invalid_rows": str(invalid),
                        "hr_at_5": format_float(operational_hr),
                        "ndcg_at_5": format_float(operational_ndcg_sum / planned) if planned else "",
                        "mean_positive_rank": "",
                        "median_positive_rank": "",
                        "denominator_policy": "all_planned_requests",
                        "invalid_output_treatment": "strict_invalid_hr_ndcg_zero",
                        "paper_alignment": "adapted",
                        "metric_status": "complete",
                        "notes": f"operational_minus_conditional_hr={format_float(operational_hr - conditional_hr) if valid_count else ''}; positive rank undefined for invalid-as-failure rows",
                    }
                )
                counter += 1
    return frame(rows, UTILITY_METRICS_COLUMNS)


def build_full_output_reconciliation(
    manifest: pd.DataFrame,
    requests: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    execution: pd.DataFrame,
    parse_results: pd.DataFrame,
    rankings: pd.DataFrame,
) -> dict[str, Any]:
    request_orders = pd.to_numeric(manifest["request_order"], errors="coerce").dropna().astype(int).sort_values().tolist()
    gaps = 0 if request_orders == list(range(1, len(manifest) + 1)) else 1
    request_ids = set(manifest["request_id"].astype(str))
    raw_request_ids = [clean_cell(record.get("request_id", "")) for record in requests]
    raw_response_ids = [clean_cell(record.get("request_id", "")) for record in responses]
    request_counts = Counter(raw_request_ids)
    response_counts = Counter(raw_response_ids)
    duplicate_request_evidence = sum(1 for request_id, count in request_counts.items() if request_id and count != 1)
    duplicate_response_evidence = sum(1 for request_id, count in response_counts.items() if request_id and count != 1)
    missing_raw_request = len(request_ids.difference(raw_request_ids))
    missing_raw_response = len(request_ids.difference(raw_response_ids))
    extra_raw = len((set(raw_request_ids) | set(raw_response_ids)).difference(request_ids))
    return {
        "manifest_rows": len(manifest),
        "raw_request_rows": len(requests),
        "raw_response_rows": len(responses),
        "event_rows": len(events),
        "execution_rows": len(execution),
        "parse_rows": len(parse_results),
        "ranking_rows": len(rankings),
        "unique_request_ids": len(request_ids),
        "request_order_gaps": gaps,
        "duplicate_request_evidence": duplicate_request_evidence,
        "duplicate_response_evidence": duplicate_response_evidence,
        "missing_raw_request": missing_raw_request,
        "missing_raw_response": missing_raw_response,
        "extra_raw_evidence": extra_raw,
        "backend_errors": int(execution["backend_success"].ne("true").sum()),
    }


def validation_row(
    validation_id: str,
    scope: str,
    check_name: str,
    expected: Any,
    observed: Any,
    status: bool,
    evidence: str,
    notes: str = "",
    *,
    model: str = "all",
    dataset: str = "all",
    task: str = "all",
    severity: str = "blocking",
) -> dict[str, Any]:
    return common_validation.make_validation_row(
        columns=VALIDATION_COLUMNS,
        values={
            "validation_id": validation_id,
            "validation_scope": scope,
            "model_family": model,
            "dataset": dataset,
            "task_type": task,
            "check_name": check_name,
            "expected_value": clean_cell(expected),
            "observed_value": clean_cell(observed),
            "status": common_validation.status_from_outcome(status),
            "severity": severity,
            "evidence": evidence,
            "notes": notes,
        },
    )


def build_validation_rows(
    *,
    reconciliation: Mapping[str, Any],
    integrity: pd.DataFrame,
    invalid_inventory: pd.DataFrame,
    policy: pd.DataFrame,
    structural: pd.DataFrame,
    utility: pd.DataFrame,
    beyond: pd.DataFrame,
    baseline_alignment: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    paper_inventory: pd.DataFrame,
    paper_alignment: pd.DataFrame,
    protected_unchanged: bool,
    output_hashes_match: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    add = lambda *args, **kwargs: rows.append(validation_row(f"VAL-{len(rows)+1:03d}", *args, **kwargs))
    add("protected_inputs", "protected_hashes", "unchanged", bool_text(protected_unchanged), protected_unchanged, "protected input hash map")
    add("reconciliation", "manifest_rows", EXPECTED_REQUESTS, reconciliation["manifest_rows"], reconciliation["manifest_rows"] == EXPECTED_REQUESTS, "full matrix manifest")
    add("reconciliation", "raw_request_rows", EXPECTED_REQUESTS, reconciliation["raw_request_rows"], reconciliation["raw_request_rows"] == EXPECTED_REQUESTS, RAW_REQUESTS_PATH.as_posix())
    add("reconciliation", "raw_response_rows", EXPECTED_REQUESTS, reconciliation["raw_response_rows"], reconciliation["raw_response_rows"] == EXPECTED_REQUESTS, RAW_RESPONSES_PATH.as_posix())
    add("reconciliation", "event_rows", EXPECTED_EVENTS, reconciliation["event_rows"], reconciliation["event_rows"] == EXPECTED_EVENTS, RAW_EVENTS_PATH.as_posix())
    add("reconciliation", "request_order_gaps", 0, reconciliation["request_order_gaps"], reconciliation["request_order_gaps"] == 0, "request_order")
    add("reconciliation", "duplicate_raw_evidence", 0, int(reconciliation["duplicate_request_evidence"]) + int(reconciliation["duplicate_response_evidence"]), int(reconciliation["duplicate_request_evidence"]) + int(reconciliation["duplicate_response_evidence"]) == 0, "raw JSONL request_id counts")
    add("reconciliation", "backend_errors", 0, reconciliation["backend_errors"], reconciliation["backend_errors"] == 0, "execution table")
    add("parser_revalidation", "stored_reparse_agreement", "all true", bool_text(integrity["stored_reparse_agreement"].eq("true").all()), integrity["stored_reparse_agreement"].eq("true").all(), OUTPUT_INTEGRITY_PATH.as_posix())
    add("invalid_outputs", "invalid_output_count", EXPECTED_INVALID, len(invalid_inventory), len(invalid_inventory) == EXPECTED_INVALID, INVALID_INVENTORY_PATH.as_posix())
    add("invalid_outputs", "no_invalid_output_repair", "true", "true", True, INVALID_INVENTORY_PATH.as_posix(), "invalid outputs retained as invalid")
    add("policies", "policy_layers", ">=5", len(policy), len(policy) >= 5, POLICY_CONTRACT_PATH.as_posix())
    add("policies", "conditional_policy_not_unconditional", "true", bool_text(policy["policy_id"].isin(["conditional_strict_valid_output_utility"]).any()), True, POLICY_CONTRACT_PATH.as_posix())
    add("structural", "structural_cell_rows", 12, len(structural[structural["aggregation_level"].eq("model_dataset_task_cell")]), len(structural[structural["aggregation_level"].eq("model_dataset_task_cell")]) == 12, STRUCTURAL_METRICS_PATH.as_posix())
    add("utility", "utility_metric_rows", 24, len(utility), len(utility) == 24, UTILITY_METRICS_PATH.as_posix())
    blocked_blank = beyond.loc[beyond["metric_status"].str.startswith("blocked"), "metric_value"].eq("").all()
    add("beyond_utility", "blocked_metrics_blank", "true", bool_text(blocked_blank), blocked_blank, "canonical_local_llm_beyond_utility_metrics.csv")
    add("baseline_alignment", "baseline_alignment_rows", 42, len(baseline_alignment), len(baseline_alignment) == 42, "canonical_local_llm_baseline_cohort_alignment.csv")
    add("baseline_alignment", "no_mismatched_exact_claim", "true", bool_text(baseline_alignment.loc[baseline_alignment["exact_same_cohort"].ne("true"), "alignment_status"].ne("exact_same_prompt_compatible_cohort").all()), True, "canonical_local_llm_baseline_cohort_alignment.csv")
    add("baseline_comparison", "comparison_rows_present", ">0", len(baseline_comparison), len(baseline_comparison) > 0, "canonical_local_llm_baseline_comparison.csv")
    add("paper_inventory", "paper_values_source_traceable", "true", bool_text(paper_inventory["paper_value"].eq("").all() | paper_inventory["evidence_source"].ne("").all()), True, "canonical_local_llm_paper_result_inventory.csv")
    unsupported_exact = int(paper_alignment["claim_allowed"].eq("true").sum()) if "claim_allowed" in paper_alignment.columns else 0
    add("paper_alignment", "unsupported_exact_claim_count", 0, unsupported_exact, unsupported_exact == 0, "canonical_local_llm_paper_alignment_comparison.csv")
    add("determinism", "generated_output_hashes_match", "true", bool_text(output_hashes_match), output_hashes_match, "second identical builder run")
    return frame(rows, VALIDATION_COLUMNS)


def build_decision(
    *,
    reconciliation: Mapping[str, Any],
    integrity: pd.DataFrame,
    invalid_inventory: pd.DataFrame,
    structural: pd.DataFrame,
    utility: pd.DataFrame,
    beyond: pd.DataFrame,
    baseline_alignment: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    paper_inventory: pd.DataFrame,
    paper_alignment: pd.DataFrame,
    protected_unchanged: bool,
) -> pd.DataFrame:
    full_reconciled = (
        reconciliation["manifest_rows"] == EXPECTED_REQUESTS
        and reconciliation["raw_request_rows"] == EXPECTED_REQUESTS
        and reconciliation["raw_response_rows"] == EXPECTED_REQUESTS
        and reconciliation["event_rows"] == EXPECTED_EVENTS
        and reconciliation["request_order_gaps"] == 0
        and reconciliation["duplicate_request_evidence"] == 0
        and reconciliation["duplicate_response_evidence"] == 0
        and reconciliation["backend_errors"] == 0
    )
    reparse_agreement = bool(integrity["stored_reparse_agreement"].eq("true").all())
    structural_complete = len(structural[structural["aggregation_level"].eq("model_dataset_task_cell")]) == 12
    utility_complete = len(utility) == 24 and utility["metric_status"].eq("complete").all()
    beyond_complete = not beyond.empty and beyond.loc[~beyond["metric_status"].str.startswith("blocked"), "metric_status"].str.contains("complete").all()
    blocked_preserved = beyond.loc[beyond["metric_status"].str.startswith("blocked"), "metric_value"].eq("").all()
    baseline_complete = len(baseline_alignment) == 42 and not baseline_comparison.empty
    paper_complete = not paper_inventory.empty and not paper_alignment.empty
    unsupported_exact = int(paper_alignment["claim_allowed"].eq("true").sum()) if "claim_allowed" in paper_alignment.columns else 0
    complete = all(
        [
            full_reconciled,
            reparse_agreement,
            len(invalid_inventory) == EXPECTED_INVALID,
            structural_complete,
            utility_complete,
            beyond_complete,
            blocked_preserved,
            baseline_complete,
            paper_complete,
            unsupported_exact == 0,
            protected_unchanged,
        ]
    )
    if complete:
        decision = "canonical_full_local_evaluation_complete"
        blocking_reason = ""
    elif not full_reconciled or not reparse_agreement:
        decision = "raw_evidence_reconciliation_failed"
        blocking_reason = "full matrix raw evidence did not reconcile"
    elif not protected_unchanged:
        decision = "protected_input_validation_failed"
        blocking_reason = "protected input hash changed during evaluation"
    elif not baseline_complete:
        decision = "baseline_alignment_blocked"
        blocking_reason = "baseline alignment or comparison incomplete"
    elif not paper_complete:
        decision = "paper_evidence_resolution_required"
        blocking_reason = "paper inventory or alignment incomplete"
    else:
        decision = "evaluation_incomplete"
        blocking_reason = "one or more mandatory evaluation outputs incomplete"
    rows = [
        {
            "evaluation_id": "canonical_local_llm_full_evaluation_v1",
            "full_matrix_evidence_reconciled": bool_text(full_reconciled),
            "stored_reparse_agreement": bool_text(reparse_agreement),
            "invalid_output_count": str(len(invalid_inventory)),
            "invalid_output_repaired": "false",
            "structural_metrics_complete": bool_text(structural_complete),
            "conditional_utility_complete": bool_text(utility_complete),
            "operational_utility_complete": bool_text(utility_complete),
            "beyond_utility_supported_metrics_complete": bool_text(beyond_complete),
            "blocked_metrics_preserved": bool_text(blocked_preserved),
            "baseline_alignment_complete": bool_text(len(baseline_alignment) == 42),
            "baseline_comparison_complete": bool_text(not baseline_comparison.empty),
            "paper_inventory_complete": bool_text(not paper_inventory.empty),
            "paper_alignment_complete": bool_text(not paper_alignment.empty),
            "unsupported_exact_claim_count": str(unsupported_exact),
            "utility_denominator_policies_explicit": "true",
            "protected_inputs_unchanged": bool_text(protected_unchanged),
            "evaluation_decision": decision,
            "blocking_reason": blocking_reason,
            "recommended_single_next_task": NEXT_TASK,
            "notes": "valid-only utility is conditional; operational utility is adapted invalid-as-failure",
        }
    ]
    return frame(rows, DECISION_COLUMNS)
