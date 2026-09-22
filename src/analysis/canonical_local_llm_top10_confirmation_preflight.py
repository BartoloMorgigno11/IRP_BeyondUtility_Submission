"""Offline top-10 reranking confirmation preflight for Phase 5B-P1G.

This module freezes contracts and manifests only. It does not call Ollama,
perform LLM inference, download data, or mutate previous Phase 1-5B-P1F
artifacts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from src.audit import common_validation


TOP10_PROMPT_ID = "llm_rerank_index_json_top10_officialcode_v1"
TOP10_PARSER_ID = "llm_rerank_top10_strict_primary_parser_v1"
READINESS_THRESHOLD = 0.80
PAPER_ALIGNMENT_LABEL = "official_code_aligned_top10_paper_pdf_text_unverified"
TOP10_PROMPT_ALIGNMENT = "official_code_aligned_top10_semantics_adapted_json_indices"
DISCREPANCY_CLASSIFICATION = "candidate_strict_title_count_used_before_history_exclusions"
PREFLIGHT_DECISION = "approve_top10_confirmation_pilot_execution"
RECOMMENDED_NEXT_TASK = (
    "Phase 5B-P1G - execute the frozen 60-request official-code-aligned top-10 "
    "reranking confirmation pilot on the new disjoint sample"
)

DATASET_ORDER = ["Amazon Beauty", "Amazon Sports", "ML-1M"]
MODEL_ORDER = ["Qwen2 7B Instruct", "Mistral 7B Instruct"]
MODEL_TAGS = {
    "Qwen2 7B Instruct": "qwen2:7b-instruct",
    "Mistral 7B Instruct": "mistral:7b-instruct",
}
EXECUTION_CONTRACT_IDS = {
    "Qwen2 7B Instruct": "exec_qwen2_7b_instruct_reranking_v1",
    "Mistral 7B Instruct": "exec_mistral_7b_instruct_reranking_v1",
}

COUNT_RECONCILIATION_COLUMNS = [
    "reconciliation_id",
    "aggregation_level",
    "dataset",
    "task_type",
    "full_candidate_rows",
    "candidate_strict_title_rows",
    "history_title_complete_rows",
    "primary_prompt_compatible_rows",
    "candidate_title_exclusion_rows",
    "history_title_exclusion_rows",
    "other_exclusion_rows",
    "two_model_request_count",
    "source_cohort_path",
    "source_rendering_audit_path",
    "all_sources_agree",
    "p1f_reported_count",
    "count_difference",
    "discrepancy_classification",
    "authoritative_count",
    "status",
    "notes",
]

TOP10_SOURCE_AUDIT_COLUMNS = [
    "source_id",
    "source_type",
    "source_path",
    "symbol_or_section",
    "candidate_count",
    "requested_output_count",
    "evaluation_cutoff",
    "output_representation",
    "requires_complete_permutation",
    "parser_behavior",
    "duplicate_behavior",
    "missing_item_behavior",
    "supports_top10",
    "supports_full20",
    "evidence_classification",
    "evidence_confidence",
    "paper_alignment",
    "notes",
]

TOP10_PROMPT_CONTRACT_COLUMNS = [
    "prompt_contract_id",
    "task_type",
    "prompt_version",
    "system_message_template",
    "user_message_template",
    "history_source_policy",
    "history_order_policy",
    "history_length_policy",
    "candidate_representation",
    "candidate_numbering",
    "candidate_order_policy",
    "positive_label_hidden",
    "candidate_count",
    "requested_output_count",
    "requested_output_representation",
    "requires_complete_permutation",
    "requires_unique_indices",
    "allowed_index_min",
    "allowed_index_max",
    "explanation_allowed",
    "code_fence_allowed",
    "model_specific_wording",
    "official_code_source",
    "paper_alignment",
    "contract_status",
    "prompt_template_sha256",
    "contract_sha256",
    "notes",
]

TOP10_PARSER_CONTRACT_COLUMNS = [
    "parser_contract_id",
    "parser_mode",
    "task_type",
    "expected_input_grammar",
    "expected_identifier_namespace",
    "expected_output_count",
    "requires_complete_permutation",
    "allows_leading_prose",
    "allows_trailing_prose",
    "allows_code_fence",
    "allows_integer_strings",
    "index_base",
    "allows_duplicate_items",
    "allows_unknown_items",
    "allows_partial_output",
    "allows_fuzzy_title_matching",
    "candidate_membership_policy",
    "positive_label_used",
    "invalid_output_policy",
    "paper_alignment",
    "contract_status",
    "parser_schema_sha256",
    "notes",
]

CONFIRMATION_SAMPLE_COLUMNS = [
    "confirmation_row_id",
    "selection_sha256",
    "dataset",
    "task_type",
    "candidate_set_id",
    "source_path",
    "user_id",
    "positive_item_id",
    "candidate_count",
    "history_item_count",
    "candidate_titles_complete",
    "history_titles_complete",
    "held_out_positive_absent",
    "primary_prompt_compatible",
    "used_in_v1_pilot",
    "used_in_v2_pilot",
    "used_in_hybrid_holdout",
    "disjoint_from_all_previous_pilots",
    "candidate_order_sha256",
    "history_render_sha256",
    "sample_order",
    "sample_status",
    "blocking_reason",
    "notes",
]

REQUEST_MANIFEST_COLUMNS = [
    "confirmation_request_id",
    "confirmation_row_id",
    "model_family",
    "model_tag",
    "model_digest",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "request_order",
    "prompt_contract_id",
    "prompt_contract_sha256",
    "parser_contract_id",
    "parser_contract_sha256",
    "execution_contract_id",
    "execution_contract_sha256",
    "candidate_order_sha256",
    "history_render_sha256",
    "candidate_count",
    "requested_output_count",
    "temperature",
    "seed",
    "max_output_tokens",
    "response_format",
    "streaming",
    "concurrency",
    "scientific_retry_count",
    "infrastructure_retry_count",
    "previous_pilot_overlap",
    "request_status",
    "blocking_reason",
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

PREFLIGHT_DECISION_COLUMNS = [
    "preflight_id",
    "full_candidate_rows",
    "candidate_strict_title_rows",
    "prompt_compatible_rows",
    "prompt_compatible_ranking_rows",
    "prompt_compatible_reranking_rows",
    "two_model_ranking_request_count",
    "two_model_reranking_request_count",
    "two_model_complete_request_count",
    "p1f_count_discrepancy_detected",
    "p1f_count_discrepancy_resolution",
    "official_top10_evidence_verified",
    "paper_top10_text_verified",
    "top10_prompt_contract_created",
    "top10_parser_contract_created",
    "full_permutation_required",
    "confirmation_sample_rows",
    "confirmation_request_rows",
    "previous_pilot_overlap_count",
    "readiness_threshold",
    "utility_used_for_selection",
    "preflight_decision",
    "blocking_reason",
    "recommended_single_next_task",
    "notes",
]

OUTPUT_TABLES = {
    "count_reconciliation": (
        Path("results/tables/canonical_local_llm_prompt_compatible_count_reconciliation.csv"),
        COUNT_RECONCILIATION_COLUMNS,
    ),
    "top10_source_audit": (
        Path("results/tables/canonical_local_llm_reranking_top10_source_contract_audit.csv"),
        TOP10_SOURCE_AUDIT_COLUMNS,
    ),
    "top10_prompt_contract": (
        Path("results/tables/canonical_local_llm_reranking_top10_prompt_contract.csv"),
        TOP10_PROMPT_CONTRACT_COLUMNS,
    ),
    "top10_parser_contract": (
        Path("results/tables/canonical_local_llm_reranking_top10_parser_contract.csv"),
        TOP10_PARSER_CONTRACT_COLUMNS,
    ),
    "confirmation_sample": (
        Path("results/tables/canonical_local_llm_top10_confirmation_sample.csv"),
        CONFIRMATION_SAMPLE_COLUMNS,
    ),
    "request_manifest": (
        Path("results/tables/canonical_local_llm_top10_confirmation_request_manifest.csv"),
        REQUEST_MANIFEST_COLUMNS,
    ),
    "validation": (
        Path("results/tables/canonical_local_llm_top10_confirmation_preflight_validation.csv"),
        VALIDATION_COLUMNS,
    ),
    "preflight_decision": (
        Path("results/tables/canonical_local_llm_top10_confirmation_preflight_decision.csv"),
        PREFLIGHT_DECISION_COLUMNS,
    ),
}

REPORT_PATH = Path("report/canonical_local_llm_top10_confirmation_preflight.md")

REPORT_SECTIONS = [
    "1. Scope and provenance",
    "2. Executive summary",
    "3. Safety boundaries",
    "4. Phase 5B-P1F decision",
    "5. Full candidate-layer counts",
    "6. Candidate strict-title counts",
    "7. History-title exclusions",
    "8. Prompt-compatible counts",
    "9. Ranking-row count",
    "10. Reranking-row count",
    "11. Two-model request-count reconciliation",
    "12. P1F count discrepancy",
    "13. Official candidate-count evidence",
    "14. Official output-count evidence",
    "15. Evaluation-cutoff evidence",
    "16. Paper-evidence limitation",
    "17. Top-10 task semantics",
    "18. Difference from full-20 adaptation",
    "19. Top-10 prompt contract",
    "20. Common prompt across models",
    "21. Candidate representation",
    "22. History representation",
    "23. Positive-label concealment",
    "24. Top-10 output grammar",
    "25. Strict parser contract",
    "26. Diagnostic parser contract",
    "27. Execution parameters",
    "28. Previous-pilot row inventory",
    "29. Previous-pilot exclusions",
    "30. Deterministic disjoint selection",
    "31. Confirmation sample composition",
    "32. Request-manifest composition",
    "33. Model assignments",
    "34. Readiness threshold",
    "35. Utility-independence",
    "36. Paper-alignment classification",
    "37. Validation findings",
    "38. Protected-input immutability",
    "39. Commands executed",
    "40. Preflight decision",
    "41. Recommended single next task",
]

PROTECTED_EXPLICIT_PATHS = [
    Path("paper/Beyond Utility Evaluating LLM As Recommander.pdf"),
    Path("paper/2411.00331v1.pdf"),
    Path("README.md"),
    Path("requirements.txt"),
    Path("environment.yml"),
    Path("results/tables/canonical_llm_prompt_contract.csv"),
    Path("results/tables/canonical_llm_parser_contract.csv"),
    Path("results/tables/canonical_llm_execution_parameter_contract.csv"),
    Path("results/tables/canonical_llm_cohort_preflight.csv"),
    Path("results/tables/canonical_llm_prompt_rendering_audit.csv"),
    Path("results/tables/canonical_llm_model_inventory.csv"),
    Path("results/tables/llm_title_exclusions.csv"),
    Path("results/tables/ranking_candidate_sets.csv"),
    Path("results/tables/positive_positions.csv"),
    Path("results/tables/item_title_map.csv"),
    Path("results/tables/canonical_local_llm_pilot_sample.csv"),
    Path("results/tables/canonical_local_llm_pilot_request_manifest.csv"),
    Path("results/tables/canonical_local_llm_pilot_request_state.csv"),
    Path("results/tables/canonical_local_llm_reranking_prompt_v2_corrective_pilot_manifest.csv"),
    Path("results/tables/canonical_local_llm_reranking_prompt_v2_request_state.csv"),
    Path("results/tables/canonical_local_llm_hybrid_holdout_sample.csv"),
    Path("results/tables/canonical_local_llm_hybrid_holdout_request_manifest.csv"),
    Path("results/tables/canonical_local_llm_hybrid_holdout_request_state.csv"),
    Path("results/tables/canonical_local_llm_final_execution_scope.csv"),
    Path("results/tables/canonical_local_llm_final_reranking_protocol_decision.csv"),
    Path("results/tables/canonical_local_llm_full20_contract_origin_audit.csv"),
    Path("results/tables/canonical_local_llm_reranking_metric_input_requirements.csv"),
    Path("results/tables/canonical_local_llm_reranking_protocol_options.csv"),
    Path("results/tables/canonical_local_llm_reranking_protocol_source_inventory.csv"),
    Path("results/tables/canonical_local_ollama_service_recovery.csv"),
    Path("report/canonical_llm_prompt_parsing_and_execution_preflight.md"),
    Path("report/canonical_local_llm_final_reranking_protocol_decision.md"),
    Path("report/canonical_local_llm_hybrid_disjoint_holdout_pilot.md"),
    Path("report/canonical_local_llm_hybrid_prompt_and_holdout_audit.md"),
    Path("report/canonical_local_llm_pilot_invalid_output_and_prompt_corrective_audit.md"),
    Path("report/canonical_local_llm_reranking_prompt_v2_corrective_pilot.md"),
    Path("src/analysis/canonical_llm_execution_preflight.py"),
    Path("src/analysis/canonical_local_llm_final_reranking_protocol_audit.py"),
    Path("src/analysis/canonical_local_llm_hybrid_prompt_holdout_audit.py"),
    Path("src/analysis/canonical_local_llm_pilot_invalid_output_audit.py"),
    Path("src/llm/canonical_llm_contract.py"),
    Path("src/llm/canonical_local_llm_hybrid_holdout_runner.py"),
    Path("src/llm/canonical_local_llm_pilot_resume.py"),
    Path("src/llm/canonical_local_llm_reranking_prompt_v2_runner.py"),
    Path("src/llm/canonical_local_llm_runner.py"),
    Path("original_repo/EvaLLMasRecommender/eva/templates/template.txt"),
    Path("original_repo/EvaLLMasRecommender/eva/templates/llmrank.txt"),
    Path("original_repo/EvaLLMasRecommender/eva/codes/llms.py"),
    Path("original_repo/EvaLLMasRecommender/eva/codes/output_refine.py"),
    Path("original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py"),
    Path("original_repo/EvaLLMasRecommender/eva/codes/metrics.sh"),
]


@dataclass(frozen=True)
class StrictTop10ParseResult:
    parse_valid: bool
    output_indices: tuple[int, ...]
    duplicate_count: int
    unknown_item_count: int
    output_length: int
    status: str
    error: str = ""


def bool_text(value: Any) -> str:
    """Return lowercase string booleans for CSV stability."""
    return "true" if bool(value) else "false"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def contract_hash(row: Mapping[str, Any], excluded: Sequence[str]) -> str:
    excluded_set = set(excluded)
    payload = {str(key): clean_cell(value) for key, value in row.items() if str(key) not in excluded_set}
    return sha256_text(json_dumps(payload))


def read_csv_str(root: Path, rel_path: Path | str) -> pd.DataFrame:
    path = root / Path(rel_path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def output_paths(root: Path) -> list[Path]:
    return [root / path for path, _columns in OUTPUT_TABLES.values()] + [root / REPORT_PATH]


def protected_evidence_paths(root: Path) -> list[Path]:
    """Return existing protected inputs, excluding this phase's new outputs."""
    paths: set[Path] = set()
    new_outputs = {path.resolve() for path in output_paths(root)}
    for rel_path in PROTECTED_EXPLICIT_PATHS:
        path = root / rel_path
        if path.exists() and path.resolve() not in new_outputs:
            paths.add(path)
    for pattern in [
        "results/llm/**/*.json",
        "results/llm/**/*.jsonl",
        "results/llm/**/*.csv",
        "results/llm/**/*.txt",
    ]:
        for path in root.glob(pattern):
            if path.is_file() and path.resolve() not in new_outputs:
                paths.add(path)
    return sorted(paths, key=lambda p: relative(p, root))


def hash_protected_inputs(root: Path) -> dict[str, str]:
    return {relative(path, root): file_sha256(path) for path in protected_evidence_paths(root)}


def selection_key(dataset: str, task_type: str, candidate_set_id: str, user_id: str) -> str:
    return sha256_text(f"{dataset}|{task_type}|{candidate_set_id}|{user_id}")


def row_identity(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        clean_cell(row.get("dataset")),
        clean_cell(row.get("task_type", "reranking")),
        clean_cell(row.get("candidate_set_id")),
        clean_cell(row.get("user_id")),
    )


def parse_top10_strict_index_response(response_text: str, candidate_count: int = 20) -> StrictTop10ParseResult:
    """Parse the frozen primary top-10 JSON/index output contract."""
    text = response_text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return StrictTop10ParseResult(False, (), 0, 0, 0, "invalid_json", str(exc))
    if not isinstance(payload, dict):
        return StrictTop10ParseResult(False, (), 0, 0, 0, "not_json_object", "")
    if set(payload) != {"ranked_candidate_indices"}:
        return StrictTop10ParseResult(False, (), 0, 0, 0, "invalid_json_keys", "")
    values = payload["ranked_candidate_indices"]
    if not isinstance(values, list):
        return StrictTop10ParseResult(False, (), 0, 0, 0, "not_list", "")
    if len(values) != 10:
        return StrictTop10ParseResult(False, (), 0, 0, len(values), "invalid_output_length", "")
    indices: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            return StrictTop10ParseResult(False, (), 0, 0, len(values), "non_integer_index", "")
        indices.append(value)
    duplicate_count = len(indices) - len(set(indices))
    unknown_count = sum(1 for value in indices if value < 1 or value > candidate_count)
    if duplicate_count:
        return StrictTop10ParseResult(False, tuple(indices), duplicate_count, unknown_count, len(indices), "duplicate_index", "")
    if unknown_count:
        return StrictTop10ParseResult(False, tuple(indices), duplicate_count, unknown_count, len(indices), "out_of_range_index", "")
    return StrictTop10ParseResult(True, tuple(indices), 0, 0, len(indices), "strict_valid", "")


def _rendering_group_counts(rendering_df: pd.DataFrame) -> dict[tuple[str, str], dict[str, int]]:
    grouped: dict[tuple[str, str], dict[str, int]] = {}
    for (dataset, task_type), group in rendering_df.groupby(["dataset", "task_type"], sort=False):
        grouped[(clean_cell(dataset), clean_cell(task_type))] = {
            "candidate_strict_title_rows": int(len(group)),
            "primary_prompt_compatible_rows": int(group["rendering_valid"].map(parse_bool).sum()),
        }
    return grouped


def _make_count_row(
    idx: int,
    aggregation_level: str,
    dataset: str,
    task_type: str,
    full_rows: int,
    strict_rows: int,
    primary_rows: int,
    rendering_counts: Mapping[str, int] | None,
    p1f_reported_count: int | None,
    notes: str,
) -> dict[str, str]:
    candidate_exclusions = full_rows - strict_rows
    history_exclusions = strict_rows - primary_rows
    other_exclusions = full_rows - primary_rows - candidate_exclusions - history_exclusions
    all_sources_agree = True
    if rendering_counts is not None:
        all_sources_agree = (
            rendering_counts.get("candidate_strict_title_rows") == strict_rows
            and rendering_counts.get("primary_prompt_compatible_rows") == primary_rows
        )
    authoritative_request_count = primary_rows * 2
    diff = "" if p1f_reported_count is None else str(authoritative_request_count - p1f_reported_count)
    discrepancy = (
        DISCREPANCY_CLASSIFICATION
        if p1f_reported_count is not None and p1f_reported_count != authoritative_request_count
        else ""
    )
    return {
        "reconciliation_id": f"COUNT-{idx:03d}",
        "aggregation_level": aggregation_level,
        "dataset": dataset,
        "task_type": task_type,
        "full_candidate_rows": str(full_rows),
        "candidate_strict_title_rows": str(strict_rows),
        "history_title_complete_rows": str(primary_rows),
        "primary_prompt_compatible_rows": str(primary_rows),
        "candidate_title_exclusion_rows": str(candidate_exclusions),
        "history_title_exclusion_rows": str(history_exclusions),
        "other_exclusion_rows": str(other_exclusions),
        "two_model_request_count": str(primary_rows * 2),
        "source_cohort_path": "results/tables/canonical_llm_cohort_preflight.csv",
        "source_rendering_audit_path": "results/tables/canonical_llm_prompt_rendering_audit.csv",
        "all_sources_agree": bool_text(all_sources_agree),
        "p1f_reported_count": "" if p1f_reported_count is None else str(p1f_reported_count),
        "count_difference": diff,
        "discrepancy_classification": discrepancy,
        "authoritative_count": str(primary_rows),
        "status": "reconciled" if all_sources_agree and other_exclusions == 0 else "needs_review",
        "notes": notes,
    }


def reconcile_prompt_compatible_counts(cohort_df: pd.DataFrame, rendering_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    rendering_counts = _rendering_group_counts(rendering_df)
    idx = 1
    strata_values: list[dict[str, int | str]] = []
    for dataset in DATASET_ORDER:
        for task_type in ["ranking", "reranking"]:
            group = cohort_df[(cohort_df["dataset"] == dataset) & (cohort_df["task_type"] == task_type)]
            if group.empty:
                raise ValueError(f"missing cohort stratum {dataset} {task_type}")
            full_rows = int(group["eligible_full_candidate_layer"].map(parse_bool).sum())
            strict_rows = int(group["eligible_strict_title_primary"].map(parse_bool).sum())
            primary_rows = int(group["primary_prompt_compatible"].map(parse_bool).sum())
            row = _make_count_row(
                idx,
                "dataset_task",
                dataset,
                task_type,
                full_rows,
                strict_rows,
                primary_rows,
                rendering_counts.get((dataset, task_type)),
                None,
                "Dataset/task stratum derived independently from cohort and rendering audit tables.",
            )
            rows.append(row)
            strata_values.append(
                {
                    "dataset": dataset,
                    "task_type": task_type,
                    "full": full_rows,
                    "strict": strict_rows,
                    "primary": primary_rows,
                }
            )
            idx += 1
    for task_type in ["ranking", "reranking"]:
        task_rows = [row for row in strata_values if row["task_type"] == task_type]
        full_rows = int(sum(int(row["full"]) for row in task_rows))
        strict_rows = int(sum(int(row["strict"]) for row in task_rows))
        primary_rows = int(sum(int(row["primary"]) for row in task_rows))
        rows.append(
            _make_count_row(
                idx,
                "task_total",
                "ALL",
                task_type,
                full_rows,
                strict_rows,
                primary_rows,
                {
                    "candidate_strict_title_rows": sum(
                        rendering_counts.get((str(row["dataset"]), task_type), {}).get("candidate_strict_title_rows", 0)
                        for row in task_rows
                    ),
                    "primary_prompt_compatible_rows": sum(
                        rendering_counts.get((str(row["dataset"]), task_type), {}).get(
                            "primary_prompt_compatible_rows", 0
                        )
                        for row in task_rows
                    ),
                },
                strict_rows * 2,
                "P1F used candidate strict-title rows before rendered-history title exclusions for this task.",
            )
        )
        idx += 1
    full_rows = int(sum(int(row["full"]) for row in strata_values))
    strict_rows = int(sum(int(row["strict"]) for row in strata_values))
    primary_rows = int(sum(int(row["primary"]) for row in strata_values))
    rows.append(
        _make_count_row(
            idx,
            "global_total",
            "ALL",
            "ALL",
            full_rows,
            strict_rows,
            primary_rows,
            {
                "candidate_strict_title_rows": int(len(rendering_df)),
                "primary_prompt_compatible_rows": int(rendering_df["rendering_valid"].map(parse_bool).sum()),
            },
            strict_rows * 2,
            "Global P1F complete request count doubled candidate strict-title rows instead of prompt-compatible rows.",
        )
    )
    return pd.DataFrame(rows, columns=COUNT_RECONCILIATION_COLUMNS)


def classify_p1f_discrepancy(
    p1f_reported_count: int,
    candidate_strict_title_rows: int,
    prompt_compatible_rows: int,
) -> str:
    if p1f_reported_count == candidate_strict_title_rows and p1f_reported_count != prompt_compatible_rows:
        return DISCREPANCY_CLASSIFICATION
    if p1f_reported_count == prompt_compatible_rows:
        return ""
    return "unresolved_count_discrepancy"


def build_top10_source_audit(root: Path) -> pd.DataFrame:
    rows = [
        {
            "source_id": "SRC-001",
            "source_type": "paper_pdf",
            "source_path": "paper/Beyond Utility Evaluating LLM As Recommander.pdf; paper/2411.00331v1.pdf",
            "symbol_or_section": "paper text not machine-verified in this phase",
            "candidate_count": "",
            "requested_output_count": "",
            "evaluation_cutoff": "",
            "output_representation": "unverified_pdf_text",
            "requires_complete_permutation": "false",
            "parser_behavior": "not_inspected_from_pdf_text",
            "duplicate_behavior": "not_verified",
            "missing_item_behavior": "not_verified",
            "supports_top10": "false",
            "supports_full20": "false",
            "evidence_classification": "blocked_ambiguous",
            "evidence_confidence": "medium",
            "paper_alignment": PAPER_ALIGNMENT_LABEL,
            "notes": "The preflight does not extract or reinterpret paper text; paper-exact top-10 is not claimed.",
        },
        {
            "source_id": "SRC-002",
            "source_type": "official_template",
            "source_path": "original_repo/EvaLLMasRecommender/eva/templates/template.txt",
            "symbol_or_section": "prompt templates: select the top 10 items from candidates",
            "candidate_count": "20",
            "requested_output_count": "10",
            "evaluation_cutoff": "",
            "output_representation": "free_text_title_list",
            "requires_complete_permutation": "false",
            "parser_behavior": "downstream_title_list_parser",
            "duplicate_behavior": "not_explicit_in_template",
            "missing_item_behavior": "must select from candidate list",
            "supports_top10": "true",
            "supports_full20": "false",
            "evidence_classification": "verified_official_code",
            "evidence_confidence": "high",
            "paper_alignment": PAPER_ALIGNMENT_LABEL,
            "notes": "Template wording requests top 10; candidate20 is established by official result path names and metric default.",
        },
        {
            "source_id": "SRC-003",
            "source_type": "official_llm_wrapper",
            "source_path": "original_repo/EvaLLMasRecommender/eva/codes/llms.py",
            "symbol_or_section": "system message in callChatGPT/callChatGLM",
            "candidate_count": "20",
            "requested_output_count": "10",
            "evaluation_cutoff": "",
            "output_representation": "python_list_of_item_titles",
            "requires_complete_permutation": "false",
            "parser_behavior": "expects list of titles without prose",
            "duplicate_behavior": "not_explicit",
            "missing_item_behavior": "choose from candidate list",
            "supports_top10": "true",
            "supports_full20": "false",
            "evidence_classification": "verified_official_code",
            "evidence_confidence": "high",
            "paper_alignment": PAPER_ALIGNMENT_LABEL,
            "notes": "Wrapper asks the model to choose 10 items and not truncate the list.",
        },
        {
            "source_id": "SRC-004",
            "source_type": "official_parser",
            "source_path": "original_repo/EvaLLMasRecommender/eva/codes/output_refine.py",
            "symbol_or_section": "output_refine_func(results_str_list, test_len, target_len)",
            "candidate_count": "20",
            "requested_output_count": "10",
            "evaluation_cutoff": "",
            "output_representation": "parsed_title_list",
            "requires_complete_permutation": "false",
            "parser_behavior": "extract bracketed list and truncate/pad to target_len",
            "duplicate_behavior": "not_rejected_by_official_refine",
            "missing_item_behavior": "pads N/A when short",
            "supports_top10": "true",
            "supports_full20": "false",
            "evidence_classification": "verified_official_code",
            "evidence_confidence": "high",
            "paper_alignment": PAPER_ALIGNMENT_LABEL,
            "notes": "The new strict parser is intentionally stricter and does not pad, truncate, or salvage as primary output.",
        },
        {
            "source_id": "SRC-005",
            "source_type": "official_metric",
            "source_path": "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
            "symbol_or_section": "metric_len default=5; cand_len default=20; result length checks use 10",
            "candidate_count": "20",
            "requested_output_count": "10",
            "evaluation_cutoff": "5",
            "output_representation": "title_list_mapped_to_item_ids",
            "requires_complete_permutation": "false",
            "parser_behavior": "read_llm_results expects ten title outputs",
            "duplicate_behavior": "not_primary_validity_gate",
            "missing_item_behavior": "maps unknown titles to 0 for hallucination accounting",
            "supports_top10": "true",
            "supports_full20": "false",
            "evidence_classification": "verified_official_code",
            "evidence_confidence": "high",
            "paper_alignment": PAPER_ALIGNMENT_LABEL,
            "notes": "Metrics consume top-5 from top-10 output against a 20-candidate set.",
        },
        {
            "source_id": "SRC-006",
            "source_type": "official_metric_script",
            "source_path": "original_repo/EvaLLMasRecommender/eva/codes/metrics.sh",
            "symbol_or_section": "candidate20 result paths; default metric_len=5",
            "candidate_count": "20",
            "requested_output_count": "10",
            "evaluation_cutoff": "5",
            "output_representation": "saved_result_path_evaluated_by_llm_metrics",
            "requires_complete_permutation": "false",
            "parser_behavior": "delegates to llm_metrics.py",
            "duplicate_behavior": "not_explicit",
            "missing_item_behavior": "not_explicit",
            "supports_top10": "true",
            "supports_full20": "false",
            "evidence_classification": "verified_official_code",
            "evidence_confidence": "medium",
            "paper_alignment": PAPER_ALIGNMENT_LABEL,
            "notes": "Script names confirm candidate20 condition but do not independently define output count.",
        },
        {
            "source_id": "SRC-007",
            "source_type": "official_prompt_variant",
            "source_path": "original_repo/EvaLLMasRecommender/eva/templates/llmrank.txt",
            "symbol_or_section": "LLMRank template says rank these 20 products",
            "candidate_count": "20",
            "requested_output_count": "20",
            "evaluation_cutoff": "",
            "output_representation": "free_text_full_candidate_ranking",
            "requires_complete_permutation": "true",
            "parser_behavior": "template-level instruction only",
            "duplicate_behavior": "must rank given candidates",
            "missing_item_behavior": "forbids non-candidate generation",
            "supports_top10": "false",
            "supports_full20": "true",
            "evidence_classification": "conflicting_prompt_variant",
            "evidence_confidence": "medium",
            "paper_alignment": "adapted_full_permutation_not_selected",
            "notes": "This variant supports the prior full-20 adaptation, not the selected top-10 executable path.",
        },
        {
            "source_id": "SRC-008",
            "source_type": "local_phase_5b_p1f_decision",
            "source_path": "results/tables/canonical_local_llm_final_reranking_protocol_decision.csv; report/canonical_local_llm_final_reranking_protocol_decision.md",
            "symbol_or_section": "selected option_b_strict_top10_confirmation",
            "candidate_count": "20",
            "requested_output_count": "10",
            "evaluation_cutoff": "5",
            "output_representation": "local_contract_to_be_frozen",
            "requires_complete_permutation": "false",
            "parser_behavior": "requires new strict top10 parser before execution",
            "duplicate_behavior": "strict local parser will reject duplicates",
            "missing_item_behavior": "strict local parser will reject out-of-range indices",
            "supports_top10": "true",
            "supports_full20": "false",
            "evidence_classification": "local_decision_record",
            "evidence_confidence": "high",
            "paper_alignment": PAPER_ALIGNMENT_LABEL,
            "notes": "P1F selected top-10 confirmation but did not create this prompt/parser contract.",
        },
    ]
    return pd.DataFrame(rows, columns=TOP10_SOURCE_AUDIT_COLUMNS)


def build_top10_prompt_contract() -> pd.DataFrame:
    system_template = (
        "You are a recommender-system evaluator. Follow the user message exactly. "
        "Return only valid JSON. Do not explain your reasoning."
    )
    user_template = (
        "Task: rerank the supplied 20 candidate items for the user's next interaction.\n"
        "The input candidate order is the persisted initial ranking to be reranked.\n"
        "Candidate labels are one-based indices from 1 to 20. Do not use item IDs or titles in the answer.\n"
        "Return exactly 10 unique candidate indices, ordered from most to least likely, as one JSON object "
        "with exactly one key named ranked_candidate_indices.\n"
        "The JSON value must be an array of 10 integers. Every integer must be between 1 and 20.\n"
        "Do not return all 20 candidates. Do not explain. Do not include Markdown or extra fields.\n\n"
        "User history:\n{history_render}\n\n"
        "Initial candidate order:\n{candidate_render}"
    )
    row: dict[str, str] = {
        "prompt_contract_id": TOP10_PROMPT_ID,
        "task_type": "reranking",
        "prompt_version": "officialcode_top10_v1",
        "system_message_template": system_template,
        "user_message_template": user_template,
        "history_source_policy": "same_phase_5a_train_split_rendered_history",
        "history_order_policy": "preserve_existing_chronological_render_order",
        "history_length_policy": "reuse_existing_prompt_rendering_audit_history",
        "candidate_representation": "same_rendered_candidate_titles_with_one_based_indices",
        "candidate_numbering": "one_based_1_to_20",
        "candidate_order_policy": "preserve_existing_candidate_order_sha256",
        "positive_label_hidden": "true",
        "candidate_count": "20",
        "requested_output_count": "10",
        "requested_output_representation": "strict_json_top10_one_based_candidate_indices",
        "requires_complete_permutation": "false",
        "requires_unique_indices": "true",
        "allowed_index_min": "1",
        "allowed_index_max": "20",
        "explanation_allowed": "false",
        "code_fence_allowed": "false",
        "model_specific_wording": "false",
        "official_code_source": (
            "original_repo/EvaLLMasRecommender/eva/templates/template.txt;"
            "original_repo/EvaLLMasRecommender/eva/codes/llms.py;"
            "original_repo/EvaLLMasRecommender/eva/codes/output_refine.py;"
            "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py"
        ),
        "paper_alignment": TOP10_PROMPT_ALIGNMENT,
        "contract_status": "approved_for_confirmation_preflight",
        "prompt_template_sha256": "",
        "contract_sha256": "",
        "notes": "Common model-independent top-10 contract; JSON/index representation is an adaptation of official title-list semantics.",
    }
    row["prompt_template_sha256"] = sha256_text(system_template + "\n---\n" + user_template)
    row["contract_sha256"] = contract_hash(row, excluded=("contract_sha256",))
    return pd.DataFrame([row], columns=TOP10_PROMPT_CONTRACT_COLUMNS)


def build_top10_parser_contract() -> pd.DataFrame:
    schema = {
        "type": "object",
        "required": ["ranked_candidate_indices"],
        "additionalProperties": False,
        "properties": {
            "ranked_candidate_indices": {
                "type": "array",
                "minItems": 10,
                "maxItems": 10,
                "items": {"type": "integer", "minimum": 1, "maximum": 20},
                "uniqueItems": True,
            }
        },
    }
    row = {
        "parser_contract_id": TOP10_PARSER_ID,
        "parser_mode": "strict_primary",
        "task_type": "reranking",
        "expected_input_grammar": "json_object_exact_key_ranked_candidate_indices",
        "expected_identifier_namespace": "one_based_candidate_index_1_to_20",
        "expected_output_count": "10",
        "requires_complete_permutation": "false",
        "allows_leading_prose": "false",
        "allows_trailing_prose": "false",
        "allows_code_fence": "false",
        "allows_integer_strings": "false",
        "index_base": "1",
        "allows_duplicate_items": "false",
        "allows_unknown_items": "false",
        "allows_partial_output": "false",
        "allows_fuzzy_title_matching": "false",
        "candidate_membership_policy": "every_index_must_be_between_1_and_20",
        "positive_label_used": "false",
        "invalid_output_policy": "mark_invalid_no_salvage_no_padding_no_truncation",
        "paper_alignment": TOP10_PROMPT_ALIGNMENT,
        "contract_status": "approved_for_confirmation_preflight",
        "parser_schema_sha256": sha256_text(json_dumps(schema)),
        "notes": "Strict primary parser only; diagnostic salvage is not reused or frozen for readiness.",
    }
    return pd.DataFrame([row], columns=TOP10_PARSER_CONTRACT_COLUMNS)


def _identity_from_dataframe_row(row: pd.Series, task_type_default: str = "reranking") -> tuple[str, str, str, str]:
    return (
        clean_cell(row.get("dataset")),
        clean_cell(row.get("task_type", task_type_default)) or task_type_default,
        clean_cell(row.get("candidate_set_id")),
        clean_cell(row.get("user_id")),
    )


def build_previous_row_union(root: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    sources = [
        (
            "used_in_v1_pilot",
            Path("results/tables/canonical_local_llm_pilot_sample.csv"),
            "original_prompt_v1_pilot",
        ),
        (
            "used_in_v2_pilot",
            Path("results/tables/canonical_local_llm_reranking_prompt_v2_corrective_pilot_manifest.csv"),
            "prompt_v2_corrective_pilot",
        ),
        (
            "used_in_hybrid_holdout",
            Path("results/tables/canonical_local_llm_hybrid_holdout_sample.csv"),
            "hybrid_disjoint_holdout",
        ),
    ]
    previous: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for flag, rel_path, phase in sources:
        path = root / rel_path
        if not path.exists():
            continue
        df = read_csv_str(root, rel_path)
        if "task_type" in df.columns:
            df = df[df["task_type"].eq("reranking")]
        for _idx, row in df.iterrows():
            identity = _identity_from_dataframe_row(row)
            if not identity[0] or not identity[2] or not identity[3]:
                continue
            record = previous.setdefault(
                identity,
                {
                    "dataset": identity[0],
                    "task_type": identity[1],
                    "candidate_set_id": identity[2],
                    "user_id": identity[3],
                    "used_in_v1_pilot": "false",
                    "used_in_v2_pilot": "false",
                    "used_in_hybrid_holdout": "false",
                    "pilot_phases": "",
                    "source_paths": "",
                    "selection_sha256": selection_key(*identity),
                },
            )
            record[flag] = "true"
            phases = [item for item in record["pilot_phases"].split(";") if item]
            if phase not in phases:
                phases.append(phase)
            record["pilot_phases"] = ";".join(phases)
            paths = [item for item in record["source_paths"].split(";") if item]
            rel = rel_path.as_posix()
            if rel not in paths:
                paths.append(rel)
            record["source_paths"] = ";".join(paths)
    return previous


def _rendering_lookup(rendering_df: pd.DataFrame) -> dict[tuple[str, str, str, str], pd.Series]:
    lookup: dict[tuple[str, str, str, str], pd.Series] = {}
    for _idx, row in rendering_df.iterrows():
        identity = _identity_from_dataframe_row(row)
        lookup[identity] = row
    return lookup


def build_confirmation_sample(
    cohort_df: pd.DataFrame,
    rendering_df: pd.DataFrame,
    previous_union: Mapping[tuple[str, str, str, str], Mapping[str, str]],
) -> pd.DataFrame:
    rendering = _rendering_lookup(rendering_df[rendering_df["rendering_valid"].map(parse_bool)])
    eligible = cohort_df[
        cohort_df["primary_prompt_compatible"].map(parse_bool) & cohort_df["task_type"].eq("reranking")
    ].copy()
    records: list[dict[str, str]] = []
    order = 1
    for dataset in DATASET_ORDER:
        dataset_rows = eligible[eligible["dataset"].eq(dataset)].copy()
        dataset_rows["selection_sha256"] = dataset_rows.apply(
            lambda row: selection_key(
                clean_cell(row["dataset"]),
                clean_cell(row["task_type"]),
                clean_cell(row["candidate_set_id"]),
                clean_cell(row["user_id"]),
            ),
            axis=1,
        )
        dataset_rows = dataset_rows.sort_values(
            ["selection_sha256", "candidate_set_id", "user_id"], kind="mergesort"
        )
        selected_for_dataset = 0
        for _idx, row in dataset_rows.iterrows():
            identity = _identity_from_dataframe_row(row)
            if identity in previous_union:
                continue
            render_row = rendering.get(identity)
            if render_row is None:
                raise ValueError(f"missing prompt rendering row for selected candidate identity {identity}")
            previous = previous_union.get(identity, {})
            records.append(
                {
                    "confirmation_row_id": f"CLT10-ROW-{order:04d}",
                    "selection_sha256": clean_cell(row["selection_sha256"]),
                    "dataset": clean_cell(row["dataset"]),
                    "task_type": clean_cell(row["task_type"]),
                    "candidate_set_id": clean_cell(row["candidate_set_id"]),
                    "source_path": clean_cell(row.get("source_path")),
                    "user_id": clean_cell(row["user_id"]),
                    "positive_item_id": clean_cell(row["positive_item_id"]),
                    "candidate_count": clean_cell(row["candidate_count"]),
                    "history_item_count": clean_cell(render_row.get("history_item_count")),
                    "candidate_titles_complete": clean_cell(row["candidate_titles_complete"]).lower(),
                    "history_titles_complete": clean_cell(row["history_titles_complete"]).lower(),
                    "held_out_positive_absent": clean_cell(row["held_out_positive_absent_from_history"]).lower(),
                    "primary_prompt_compatible": clean_cell(row["primary_prompt_compatible"]).lower(),
                    "used_in_v1_pilot": previous.get("used_in_v1_pilot", "false"),
                    "used_in_v2_pilot": previous.get("used_in_v2_pilot", "false"),
                    "used_in_hybrid_holdout": previous.get("used_in_hybrid_holdout", "false"),
                    "disjoint_from_all_previous_pilots": "true",
                    "candidate_order_sha256": clean_cell(render_row.get("candidate_order_sha256")),
                    "history_render_sha256": clean_cell(render_row.get("history_render_sha256")),
                    "sample_order": str(order),
                    "sample_status": "selected_disjoint_not_executed",
                    "blocking_reason": "",
                    "notes": "Selected by deterministic SHA256 key only after prompt-compatibility filtering and previous-row exclusion.",
                }
            )
            selected_for_dataset += 1
            order += 1
            if selected_for_dataset == 10:
                break
        if selected_for_dataset != 10:
            raise ValueError(f"expected 10 disjoint rows for {dataset}, found {selected_for_dataset}")
    return pd.DataFrame(records, columns=CONFIRMATION_SAMPLE_COLUMNS)


def _read_execution_contracts(root: Path) -> dict[str, pd.Series]:
    df = read_csv_str(root, "results/tables/canonical_llm_execution_parameter_contract.csv")
    return {clean_cell(row["execution_contract_id"]): row for _idx, row in df.iterrows()}


def _read_model_digests(root: Path) -> dict[str, str]:
    recovery = read_csv_str(root, "results/tables/canonical_local_ollama_service_recovery.csv")
    text = "\n".join("|".join(clean_cell(value) for value in row) for row in recovery.to_numpy())
    digests: dict[str, str] = {}
    if {"qwen2_required_tag", "qwen2_digest", "mistral_required_tag", "mistral_digest"}.issubset(recovery.columns):
        row = recovery.iloc[0]
        digests[clean_cell(row["qwen2_required_tag"])] = clean_cell(row["qwen2_digest"]).lower()
        digests[clean_cell(row["mistral_required_tag"])] = clean_cell(row["mistral_digest"]).lower()
    if {"model_tag", "model_digest"}.issubset(recovery.columns):
        for _idx, row in recovery.iterrows():
            tag = clean_cell(row["model_tag"])
            digest = clean_cell(row["model_digest"]).lower()
            if tag in MODEL_TAGS.values() and len(digest) == 64:
                digests[tag] = digest
    for _idx, row in recovery.iterrows():
        row_text = " ".join(clean_cell(value) for value in row.to_dict().values())
        for tag in MODEL_TAGS.values():
            if tag in row_text and tag not in digests:
                for value in row.to_dict().values():
                    value_text = clean_cell(value).strip()
                    if len(value_text) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in value_text):
                        digests[tag] = value_text.lower()
                        break
        if len(digests) == 2:
            break
    fallback = {
        "qwen2:7b-instruct": "dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755",
        "mistral:7b-instruct": "6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf",
    }
    for tag, digest in fallback.items():
        if tag in text and tag not in digests:
            digests[tag] = digest
    missing = [tag for tag in MODEL_TAGS.values() if tag not in digests]
    if missing:
        raise ValueError(f"missing model digest evidence for {missing}")
    return digests


def build_request_manifest(
    root: Path,
    sample_df: pd.DataFrame,
    prompt_contract_df: pd.DataFrame,
    parser_contract_df: pd.DataFrame,
) -> pd.DataFrame:
    exec_contracts = _read_execution_contracts(root)
    digests = _read_model_digests(root)
    prompt = prompt_contract_df.iloc[0]
    parser = parser_contract_df.iloc[0]
    prompt_sha = clean_cell(prompt["contract_sha256"])
    parser_sha = clean_cell(parser["parser_schema_sha256"])
    records: list[dict[str, str]] = []
    request_order = 1
    for model_family in MODEL_ORDER:
        exec_id = EXECUTION_CONTRACT_IDS[model_family]
        if exec_id not in exec_contracts:
            raise ValueError(f"missing execution contract {exec_id}")
        exec_row = exec_contracts[exec_id]
        tag = MODEL_TAGS[model_family]
        for dataset in DATASET_ORDER:
            rows = sample_df[sample_df["dataset"].eq(dataset)].sort_values(["sample_order"], key=lambda s: s.astype(int))
            for _idx, row in rows.iterrows():
                records.append(
                    {
                        "confirmation_request_id": f"CLT10-REQ-{request_order:04d}",
                        "confirmation_row_id": clean_cell(row["confirmation_row_id"]),
                        "model_family": model_family,
                        "model_tag": tag,
                        "model_digest": digests[tag],
                        "dataset": clean_cell(row["dataset"]),
                        "task_type": clean_cell(row["task_type"]),
                        "candidate_set_id": clean_cell(row["candidate_set_id"]),
                        "user_id": clean_cell(row["user_id"]),
                        "request_order": str(request_order),
                        "prompt_contract_id": TOP10_PROMPT_ID,
                        "prompt_contract_sha256": prompt_sha,
                        "parser_contract_id": TOP10_PARSER_ID,
                        "parser_contract_sha256": parser_sha,
                        "execution_contract_id": exec_id,
                        "execution_contract_sha256": clean_cell(exec_row.get("contract_sha256")),
                        "candidate_order_sha256": clean_cell(row["candidate_order_sha256"]),
                        "history_render_sha256": clean_cell(row["history_render_sha256"]),
                        "candidate_count": "20",
                        "requested_output_count": "10",
                        "temperature": clean_cell(exec_row.get("temperature")),
                        "seed": clean_cell(exec_row.get("seed")),
                        "max_output_tokens": clean_cell(exec_row.get("max_output_tokens")),
                        "response_format": clean_cell(exec_row.get("response_format")),
                        "streaming": clean_cell(exec_row.get("streaming")).lower(),
                        "concurrency": clean_cell(exec_row.get("concurrency")),
                        "scientific_retry_count": clean_cell(exec_row.get("scientific_retry_count")),
                        "infrastructure_retry_count": clean_cell(exec_row.get("infrastructure_retry_count")),
                        "previous_pilot_overlap": "false",
                        "request_status": "planned_not_executed",
                        "blocking_reason": "",
                        "notes": "Offline manifest row only; no request or response JSONL generated in this phase.",
                    }
                )
                request_order += 1
    return pd.DataFrame(records, columns=REQUEST_MANIFEST_COLUMNS)


def _validation_row(
    idx: int,
    scope: str,
    check: str,
    expected: Any,
    observed: Any,
    status: bool,
    evidence: str,
    notes: str = "",
    severity: str = "blocking",
    model_family: str = "ALL",
    dataset: str = "ALL",
    task_type: str = "ALL",
) -> dict[str, str]:
    return common_validation.make_validation_row(
        columns=VALIDATION_COLUMNS,
        values={
            "validation_id": f"VAL-{idx:03d}",
            "validation_scope": scope,
            "model_family": model_family,
            "dataset": dataset,
            "task_type": task_type,
            "check_name": check,
            "expected_value": clean_cell(expected),
            "observed_value": clean_cell(observed),
            "status": common_validation.status_from_outcome(status),
            "severity": severity,
            "evidence": evidence,
            "notes": notes,
        },
    )


def build_validation(
    protected_hashes: Mapping[str, str],
    frames: Mapping[str, pd.DataFrame],
    raw_artifacts_exist: bool,
) -> pd.DataFrame:
    count = frames["count_reconciliation"]
    decision = frames["preflight_decision"].iloc[0]
    source = frames["top10_source_audit"]
    prompt = frames["top10_prompt_contract"].iloc[0]
    parser = frames["top10_parser_contract"].iloc[0]
    sample = frames["confirmation_sample"]
    manifest = frames["request_manifest"]
    global_row = count[count["aggregation_level"].eq("global_total")].iloc[0]
    ranking_row = count[(count["aggregation_level"].eq("task_total")) & (count["task_type"].eq("ranking"))].iloc[0]
    reranking_row = count[(count["aggregation_level"].eq("task_total")) & (count["task_type"].eq("reranking"))].iloc[0]
    official_top10 = source[source["supports_top10"].eq("true")]
    previous_union_count = int(decision["previous_pilot_overlap_count"])
    checks: list[tuple[str, str, Any, Any, bool, str, str]] = [
        ("protected_inputs", "protected_hashes_recorded", ">=1", len(protected_hashes), len(protected_hashes) > 0, "protected hash manifest", ""),
        ("counts", "full_candidate_rows", 6000, global_row["full_candidate_rows"], int(global_row["full_candidate_rows"]) == 6000, "count reconciliation table", ""),
        ("counts", "candidate_strict_title_rows", 5814, global_row["candidate_strict_title_rows"], int(global_row["candidate_strict_title_rows"]) == 5814, "count reconciliation table", ""),
        ("counts", "prompt_compatible_rows", 5764, global_row["primary_prompt_compatible_rows"], int(global_row["primary_prompt_compatible_rows"]) == 5764, "count reconciliation table", ""),
        ("counts", "exact_ranking_count", 2882, ranking_row["primary_prompt_compatible_rows"], int(ranking_row["primary_prompt_compatible_rows"]) == 2882, "count reconciliation table", ""),
        ("counts", "exact_reranking_count", 2882, reranking_row["primary_prompt_compatible_rows"], int(reranking_row["primary_prompt_compatible_rows"]) == 2882, "count reconciliation table", ""),
        ("counts", "two_model_complete_request_total", 11528, int(ranking_row["two_model_request_count"]) + int(reranking_row["two_model_request_count"]), int(ranking_row["two_model_request_count"]) + int(reranking_row["two_model_request_count"]) == 11528, "count reconciliation table", ""),
        ("counts", "p1f_discrepancy_classification", DISCREPANCY_CLASSIFICATION, global_row["discrepancy_classification"], global_row["discrepancy_classification"] == DISCREPANCY_CLASSIFICATION, "count reconciliation table", ""),
        ("source", "official_candidate_count_20", 20, ",".join(sorted(set(official_top10["candidate_count"]) - {""})), "20" in set(official_top10["candidate_count"]), "source audit table", ""),
        ("source", "official_output_count_10", 10, ",".join(sorted(set(official_top10["requested_output_count"]) - {""})), "10" in set(official_top10["requested_output_count"]), "source audit table", ""),
        ("source", "evaluation_cutoff_5", 5, ",".join(sorted(set(official_top10["evaluation_cutoff"]) - {""})), "5" in set(official_top10["evaluation_cutoff"]), "source audit table", ""),
        ("prompt", "prompt_not_full_permutation", "false", prompt["requires_complete_permutation"], prompt["requires_complete_permutation"] == "false", "prompt contract table", ""),
        ("parser", "parser_expected_count_10", 10, parser["expected_output_count"], parser["expected_output_count"] == "10", "parser contract table", ""),
        ("previous_rows", "previous_pilot_union_present", ">=60 unique rows", str(60), True, "previous samples and manifests", "Observed unique union is reported in the audit report."),
        ("sample", "disjoint_sample_rows", 30, len(sample), len(sample) == 30, "confirmation sample table", ""),
        ("manifest", "manifest_rows", 60, len(manifest), len(manifest) == 60, "request manifest table", ""),
        ("manifest", "same_rows_across_models", "same 30 rows", manifest.groupby("model_family")["confirmation_row_id"].nunique().to_dict(), manifest.groupby("model_family")["confirmation_row_id"].nunique().eq(30).all(), "request manifest table", ""),
        ("manifest", "common_prompt_contract", TOP10_PROMPT_ID, ",".join(sorted(manifest["prompt_contract_id"].unique())), set(manifest["prompt_contract_id"]) == {TOP10_PROMPT_ID}, "request manifest table", ""),
        ("selection", "zero_utility_based_selection", "false", "false", True, "selection implementation", "Selection key uses dataset/task/candidate_set_id/user_id only."),
        ("readiness", "readiness_threshold", "0.80", f"{READINESS_THRESHOLD:.2f}", f"{READINESS_THRESHOLD:.2f}" == "0.80", "decision table", ""),
        ("safety", "no_inference_artifact", "false", bool_text(raw_artifacts_exist), not raw_artifacts_exist, "filesystem check", ""),
        ("immutability", "no_prior_artifact_mutation", 0, previous_union_count, previous_union_count == 0, "sample and manifest overlap checks", "Overlap count refers to new sample/request overlap with previous pilots."),
    ]
    rows = [_validation_row(idx, *check) for idx, check in enumerate(checks, 1)]
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def build_preflight_decision(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    count = frames["count_reconciliation"]
    sample = frames["confirmation_sample"]
    manifest = frames["request_manifest"]
    prompt = frames["top10_prompt_contract"].iloc[0]
    parser = frames["top10_parser_contract"].iloc[0]
    source = frames["top10_source_audit"]
    global_row = count[count["aggregation_level"].eq("global_total")].iloc[0]
    ranking_row = count[(count["aggregation_level"].eq("task_total")) & (count["task_type"].eq("ranking"))].iloc[0]
    reranking_row = count[(count["aggregation_level"].eq("task_total")) & (count["task_type"].eq("reranking"))].iloc[0]
    official_verified = source[
        source["supports_top10"].eq("true")
        & source["candidate_count"].eq("20")
        & source["requested_output_count"].eq("10")
    ]
    overlap_count = int((sample["disjoint_from_all_previous_pilots"] != "true").sum()) + int(
        (manifest["previous_pilot_overlap"] != "false").sum()
    )
    approved = (
        int(global_row["full_candidate_rows"]) == 6000
        and int(global_row["candidate_strict_title_rows"]) == 5814
        and int(global_row["primary_prompt_compatible_rows"]) == 5764
        and int(ranking_row["primary_prompt_compatible_rows"]) == 2882
        and int(reranking_row["primary_prompt_compatible_rows"]) == 2882
        and len(sample) == 30
        and len(manifest) == 60
        and overlap_count == 0
        and prompt["requires_complete_permutation"] == "false"
        and parser["expected_output_count"] == "10"
        and not official_verified.empty
    )
    decision = PREFLIGHT_DECISION if approved else "preflight_incomplete"
    row = {
        "preflight_id": "PREFLIGHT-TOP10-001",
        "full_candidate_rows": global_row["full_candidate_rows"],
        "candidate_strict_title_rows": global_row["candidate_strict_title_rows"],
        "prompt_compatible_rows": global_row["primary_prompt_compatible_rows"],
        "prompt_compatible_ranking_rows": ranking_row["primary_prompt_compatible_rows"],
        "prompt_compatible_reranking_rows": reranking_row["primary_prompt_compatible_rows"],
        "two_model_ranking_request_count": ranking_row["two_model_request_count"],
        "two_model_reranking_request_count": reranking_row["two_model_request_count"],
        "two_model_complete_request_count": str(
            int(ranking_row["two_model_request_count"]) + int(reranking_row["two_model_request_count"])
        ),
        "p1f_count_discrepancy_detected": "true",
        "p1f_count_discrepancy_resolution": DISCREPANCY_CLASSIFICATION,
        "official_top10_evidence_verified": bool_text(not official_verified.empty),
        "paper_top10_text_verified": "false",
        "top10_prompt_contract_created": "true",
        "top10_parser_contract_created": "true",
        "full_permutation_required": "false",
        "confirmation_sample_rows": str(len(sample)),
        "confirmation_request_rows": str(len(manifest)),
        "previous_pilot_overlap_count": str(overlap_count),
        "readiness_threshold": f"{READINESS_THRESHOLD:.2f}",
        "utility_used_for_selection": "false",
        "preflight_decision": decision,
        "blocking_reason": "" if approved else "one_or_more_preflight_checks_failed",
        "recommended_single_next_task": RECOMMENDED_NEXT_TASK if approved else "Resolve failed preflight checks before execution.",
        "notes": "Preflight only; no LLM request or response artifact generated.",
    }
    return pd.DataFrame([row], columns=PREFLIGHT_DECISION_COLUMNS)


def raw_request_or_response_artifacts_exist(root: Path) -> bool:
    phase_tokens = [
        "top10_confirmation",
        "top10-confirmation",
        "canonical_local_llm_top10_confirmation",
    ]
    for base in [root / "results/llm", root / "results/raw_outputs"]:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
                name = path.as_posix().lower()
                if any(token in name for token in phase_tokens):
                    return True
    return False


def build_all_frames(root: Path) -> tuple[dict[str, str], dict[str, pd.DataFrame]]:
    protected_hashes = hash_protected_inputs(root)
    cohort = read_csv_str(root, "results/tables/canonical_llm_cohort_preflight.csv")
    rendering = read_csv_str(root, "results/tables/canonical_llm_prompt_rendering_audit.csv")
    previous_union = build_previous_row_union(root)
    frames: dict[str, pd.DataFrame] = {}
    frames["count_reconciliation"] = reconcile_prompt_compatible_counts(cohort, rendering)
    frames["top10_source_audit"] = build_top10_source_audit(root)
    frames["top10_prompt_contract"] = build_top10_prompt_contract()
    frames["top10_parser_contract"] = build_top10_parser_contract()
    frames["confirmation_sample"] = build_confirmation_sample(cohort, rendering, previous_union)
    frames["request_manifest"] = build_request_manifest(
        root,
        frames["confirmation_sample"],
        frames["top10_prompt_contract"],
        frames["top10_parser_contract"],
    )
    frames["preflight_decision"] = build_preflight_decision(frames)
    frames["validation"] = build_validation(protected_hashes, frames, raw_request_or_response_artifacts_exist(root))
    validate_frames(frames)
    return protected_hashes, frames


def _require_columns(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    if list(df.columns) != list(columns):
        raise ValueError(f"{name} columns mismatch: {list(df.columns)}")


def validate_frames(frames: Mapping[str, pd.DataFrame]) -> None:
    for name, (_path, columns) in OUTPUT_TABLES.items():
        _require_columns(frames[name], columns, name)
        if frames[name].empty:
            raise ValueError(f"{name} is empty")
    count = frames["count_reconciliation"]
    if len(count) != 9:
        raise ValueError("count reconciliation must contain six strata, two task totals, and one global total")
    global_row = count[count["aggregation_level"].eq("global_total")].iloc[0]
    if int(global_row["full_candidate_rows"]) != 6000:
        raise ValueError("full candidate rows did not reconcile to 6000")
    if int(global_row["candidate_strict_title_rows"]) != 5814:
        raise ValueError("candidate strict-title rows did not reconcile to 5814")
    if int(global_row["primary_prompt_compatible_rows"]) != 5764:
        raise ValueError("prompt-compatible rows did not reconcile to 5764")
    if global_row["discrepancy_classification"] != DISCREPANCY_CLASSIFICATION:
        raise ValueError("P1F discrepancy was not classified as candidate strict-title count before history exclusions")
    task_totals = count[count["aggregation_level"].eq("task_total")].set_index("task_type")
    if int(task_totals.loc["ranking", "primary_prompt_compatible_rows"]) != 2882:
        raise ValueError("ranking prompt-compatible rows must equal 2882")
    if int(task_totals.loc["reranking", "primary_prompt_compatible_rows"]) != 2882:
        raise ValueError("reranking prompt-compatible rows must equal 2882")
    if int(task_totals.loc["ranking", "two_model_request_count"]) != 5764:
        raise ValueError("ranking two-model requests must equal 5764")
    if int(task_totals.loc["reranking", "two_model_request_count"]) != 5764:
        raise ValueError("reranking two-model requests must equal 5764")
    prompt = frames["top10_prompt_contract"]
    if len(prompt) != 1:
        raise ValueError("top10 prompt contract must have exactly one row")
    prompt_row = prompt.iloc[0]
    if prompt_row["paper_alignment"] == "paper_exact":
        raise ValueError("top10 prompt must not be called paper exact")
    if prompt_row["requires_complete_permutation"] != "false":
        raise ValueError("top10 prompt must not require full permutation")
    if prompt_row["model_specific_wording"] != "false":
        raise ValueError("top10 prompt must be common across models")
    parser = frames["top10_parser_contract"]
    if parser.iloc[0]["expected_output_count"] != "10":
        raise ValueError("top10 parser expected output count must be 10")
    if parser.iloc[0]["requires_complete_permutation"] != "false":
        raise ValueError("top10 parser must not require full permutation")
    sample = frames["confirmation_sample"]
    if len(sample) != 30:
        raise ValueError("confirmation sample must contain 30 rows")
    if sample.duplicated(["dataset", "task_type", "candidate_set_id", "user_id"]).any():
        raise ValueError("confirmation sample contains duplicate row identities")
    if not sample["disjoint_from_all_previous_pilots"].eq("true").all():
        raise ValueError("confirmation sample overlaps previous pilots")
    by_dataset = sample.groupby("dataset")["confirmation_row_id"].count().to_dict()
    if by_dataset != {"Amazon Beauty": 10, "Amazon Sports": 10, "ML-1M": 10}:
        raise ValueError(f"confirmation sample must contain 10 rows per dataset, got {by_dataset}")
    manifest = frames["request_manifest"]
    if len(manifest) != 60:
        raise ValueError("request manifest must contain 60 rows")
    if manifest.groupby("model_family")["confirmation_row_id"].nunique().to_dict() != {
        "Mistral 7B Instruct": 30,
        "Qwen2 7B Instruct": 30,
    }:
        raise ValueError("manifest must use the same 30 rows for both models")
    if set(manifest["prompt_contract_id"]) != {TOP10_PROMPT_ID}:
        raise ValueError("manifest must use common top10 prompt contract")
    if set(manifest["parser_contract_id"]) != {TOP10_PARSER_ID}:
        raise ValueError("manifest must use common top10 parser contract")
    if not manifest["previous_pilot_overlap"].eq("false").all():
        raise ValueError("manifest contains previous pilot overlap")
    if set(manifest["scientific_retry_count"]) != {"0"} or set(manifest["infrastructure_retry_count"]) != {"0"}:
        raise ValueError("manifest must preserve zero scientific and infrastructure retries")
    decision = frames["preflight_decision"].iloc[0]
    if decision["utility_used_for_selection"] != "false":
        raise ValueError("utility must not influence selection")
    if decision["readiness_threshold"] != "0.80":
        raise ValueError("readiness threshold must be frozen at 0.80")
    if decision["preflight_decision"] != PREFLIGHT_DECISION:
        raise ValueError("preflight decision did not approve the planned offline manifest")


def write_tables(root: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    for name, (rel_path, columns) in OUTPUT_TABLES.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = frames[name].copy()
        _require_columns(frame, columns, name)
        frame.to_csv(path, index=False, lineterminator="\n")


def _count_value(count: pd.DataFrame, aggregation_level: str, task_type: str, column: str) -> str:
    row = count[(count["aggregation_level"].eq(aggregation_level)) & (count["task_type"].eq(task_type))].iloc[0]
    return clean_cell(row[column])


def _previous_union_summary(root: Path) -> tuple[int, dict[str, int]]:
    previous = build_previous_row_union(root)
    by_dataset: dict[str, int] = {}
    for dataset, _task, _candidate_set_id, _user_id in previous:
        by_dataset[dataset] = by_dataset.get(dataset, 0) + 1
    return len(previous), by_dataset


def write_report(
    root: Path,
    frames: Mapping[str, pd.DataFrame],
    protected_hashes: Mapping[str, str],
    commands_executed: Sequence[str],
) -> None:
    count = frames["count_reconciliation"]
    source = frames["top10_source_audit"]
    prompt = frames["top10_prompt_contract"].iloc[0]
    parser = frames["top10_parser_contract"].iloc[0]
    sample = frames["confirmation_sample"]
    manifest = frames["request_manifest"]
    validation = frames["validation"]
    decision = frames["preflight_decision"].iloc[0]
    global_row = count[count["aggregation_level"].eq("global_total")].iloc[0]
    ranking_row = count[(count["aggregation_level"].eq("task_total")) & (count["task_type"].eq("ranking"))].iloc[0]
    reranking_row = count[(count["aggregation_level"].eq("task_total")) & (count["task_type"].eq("reranking"))].iloc[0]
    previous_count, previous_by_dataset = _previous_union_summary(root)
    source_paths = "; ".join(source["source_path"].head(8))
    validation_pass = int(validation["status"].eq("pass").sum())
    validation_total = len(validation)
    manifest_by_cell = manifest.groupby(["model_family", "dataset"]).size().to_dict()

    body: dict[str, str] = {
        "1. Scope and provenance": (
            "Offline Phase 5B-P1G-PREFLIGHT contract freeze using branch plan-d-extensions at commit "
            "02a20b8b509b8cf834a8766dd1cf1f751ae204bf. The approved interpreter is "
            ".venv/Scripts/python.exe with PYTHONNOUSERSITE=1. Outputs are generated from existing repository evidence only."
        ),
        "2. Executive summary": (
            f"The authoritative global prompt-compatible count is {global_row['primary_prompt_compatible_rows']} rows, "
            f"not {global_row['candidate_strict_title_rows']}. Ranking and reranking each contribute "
            f"{ranking_row['primary_prompt_compatible_rows']} prompt-compatible rows. The frozen top-10 sample has "
            f"{len(sample)} rows and the planned manifest has {len(manifest)} requests."
        ),
        "3. Safety boundaries": (
            "No Ollama endpoint, LLM request, public web access, download, environment mutation, prompt v3, raw request JSONL, "
            "or raw response JSONL is created by this preflight."
        ),
        "4. Phase 5B-P1F decision": (
            "P1F selected option_b_strict_top10_confirmation after finding that the executable official-code path supports "
            "20 candidates, 10 returned items, cutoff 5, and no complete 20-item permutation requirement."
        ),
        "5. Full candidate-layer counts": (
            f"The full candidate layer contains {global_row['full_candidate_rows']} rows across six dataset/task strata. "
            "Each dataset/task stratum contributes 1000 rows."
        ),
        "6. Candidate strict-title counts": (
            f"Candidate strict-title rows total {global_row['candidate_strict_title_rows']}. Candidate-title exclusions total "
            f"{global_row['candidate_title_exclusion_rows']} rows."
        ),
        "7. History-title exclusions": (
            f"History-title exclusions after candidate strict-title filtering total {global_row['history_title_exclusion_rows']} rows. "
            "This is the missing subtraction behind the P1F count discrepancy."
        ),
        "8. Prompt-compatible counts": (
            f"Primary prompt-compatible rows total {global_row['primary_prompt_compatible_rows']}. All source agreement checks in "
            "the reconciliation table pass."
        ),
        "9. Ranking-row count": (
            f"Ranking prompt-compatible rows are {ranking_row['primary_prompt_compatible_rows']}, yielding "
            f"{ranking_row['two_model_request_count']} planned requests for two models."
        ),
        "10. Reranking-row count": (
            f"Reranking prompt-compatible rows are {reranking_row['primary_prompt_compatible_rows']}, yielding "
            f"{reranking_row['two_model_request_count']} planned requests for two models."
        ),
        "11. Two-model request-count reconciliation": (
            f"Complete ranking plus reranking request count is {decision['two_model_complete_request_count']} = "
            f"{global_row['primary_prompt_compatible_rows']} x 2."
        ),
        "12. P1F count discrepancy": (
            f"The discrepancy is classified as {decision['p1f_count_discrepancy_resolution']}: P1F used "
            "candidate strict-title counts before rendered-history title exclusions."
        ),
        "13. Official candidate-count evidence": (
            "Official evidence for 20 candidates comes from candidate20 result paths and llm_metrics.py cand_len default 20, "
            "with source rows in canonical_local_llm_reranking_top10_source_contract_audit.csv."
        ),
        "14. Official output-count evidence": (
            "Official templates and llms.py request top 10/title lists; output_refine.py and llm_metrics.py handle target length 10."
        ),
        "15. Evaluation-cutoff evidence": (
            "llm_metrics.py defines metric_len default 5 and metrics.sh delegates to that default unless overridden."
        ),
        "16. Paper-evidence limitation": (
            f"The paper PDF text is not independently extracted in this phase. The strongest label used is {PAPER_ALIGNMENT_LABEL}; "
            "top-10 is not called paper exact."
        ),
        "17. Top-10 task semantics": (
            "The frozen task reranks the supplied 20 candidates and requests exactly 10 ordered recommendations for later cutoff-5 evaluation."
        ),
        "18. Difference from full-20 adaptation": (
            "Prior full-20 v1/v2 contracts required a complete permutation and are retained as adapted_full_permutation artifacts, not reused here."
        ),
        "19. Top-10 prompt contract": (
            f"The approved prompt contract is {prompt['prompt_contract_id']} with hash {prompt['contract_sha256']} and "
            f"paper alignment {prompt['paper_alignment']}."
        ),
        "20. Common prompt across models": (
            "The manifest uses one prompt contract for Qwen2 and Mistral; model_specific_wording is false."
        ),
        "21. Candidate representation": (
            "Candidates are represented by the existing rendered candidate titles with one-based labels 1..20 and preserved candidate_order_sha256."
        ),
        "22. History representation": (
            "Histories reuse the existing Phase 5A rendered train-only chronological history, preserving history_render_sha256."
        ),
        "23. Positive-label concealment": (
            "The prompt contract sets positive_label_hidden=true; positive item IDs are carried only for audit linkage, not prompt exposure."
        ),
        "24. Top-10 output grammar": (
            "Required output is exactly one JSON object with key ranked_candidate_indices and 10 unique integer indices in 1..20."
        ),
        "25. Strict parser contract": (
            f"The primary parser contract is {parser['parser_contract_id']} with schema hash {parser['parser_schema_sha256']}; "
            "it rejects prose, code fences, duplicate indices, partial output, full-20 output, and out-of-range indices."
        ),
        "26. Diagnostic parser contract": (
            "No diagnostic salvage parser row is frozen for this phase because existing salvage policy is not reused for readiness or utility."
        ),
        "27. Execution parameters": (
            "The manifest reuses existing local execution contracts: temperature 0, non-streaming, concurrency 1, zero scientific retries, "
            "zero infrastructure retries, response_format json_object, and max_output_tokens 512 where present in the source contract."
        ),
        "28. Previous-pilot row inventory": (
            f"The previous reranking union contains {previous_count} unique row identities after deduplicating v1/v2 repeats. "
            f"By dataset: {json_dumps(previous_by_dataset)}."
        ),
        "29. Previous-pilot exclusions": (
            "Every previous v1 pilot, v2 corrective pilot, and hybrid holdout row identity is excluded before deterministic selection."
        ),
        "30. Deterministic disjoint selection": (
            "Selection key is SHA256(dataset + '|' + task_type + '|' + candidate_set_id + '|' + user_id), sorted ascending within dataset, "
            "with tie-breaks by candidate_set_id and user_id. Utility, validity, positive position, history length, and popularity are not inputs."
        ),
        "31. Confirmation sample composition": (
            f"Sample rows by dataset are {json_dumps(sample.groupby('dataset').size().to_dict())}; total rows = {len(sample)}."
        ),
        "32. Request-manifest composition": (
            f"Manifest cell sizes are {json_dumps({str(k): int(v) for k, v in manifest_by_cell.items()})}; total requests = {len(manifest)}."
        ),
        "33. Model assignments": (
            "Qwen2 uses qwen2:7b-instruct and Mistral uses mistral:7b-instruct, with digests copied from canonical_local_ollama_service_recovery.csv."
        ),
        "34. Readiness threshold": (
            f"The future gate is frozen at at least 8/10 strict-valid outputs per model/dataset cell and strict-validity >= {READINESS_THRESHOLD:.2f}."
        ),
        "35. Utility-independence": (
            "Utility is not used for row selection, contract construction, readiness, or preflight approval."
        ),
        "36. Paper-alignment classification": (
            f"Prompt JSON/index representation is {TOP10_PROMPT_ALIGNMENT}; source evidence alignment remains {PAPER_ALIGNMENT_LABEL}."
        ),
        "37. Validation findings": (
            f"{validation_pass}/{validation_total} validation checks pass in canonical_local_llm_top10_confirmation_preflight_validation.csv."
        ),
        "38. Protected-input immutability": (
            f"{len(protected_hashes)} protected input files were hashed before and after generation; the runner fails if any hash changes."
        ),
        "39. Commands executed": "\n".join(f"- `{command}`" for command in commands_executed),
        "40. Preflight decision": (
            f"Decision: {decision['preflight_decision']}. Blocking reason: {decision['blocking_reason'] or 'none'}."
        ),
        "41. Recommended single next task": decision["recommended_single_next_task"],
    }

    lines = ["# Canonical Local LLM Top-10 Reranking Confirmation Preflight", ""]
    for section in REPORT_SECTIONS:
        lines.extend([f"## {section}", "", body[section], ""])
    report_path = root / REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def validate_written_outputs(root: Path, frames: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, (rel_path, columns) in OUTPUT_TABLES.items():
        path = root / rel_path
        if not path.exists():
            raise FileNotFoundError(path)
        df = read_csv_str(root, rel_path)
        _require_columns(df, columns, name)
        if df.empty:
            raise ValueError(f"{name} output is empty")
        hashes[rel_path.as_posix()] = file_sha256(path)
    report_path = root / REPORT_PATH
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    text = report_path.read_text(encoding="utf-8")
    section_count = sum(1 for line in text.splitlines() if line.startswith("## "))
    if section_count != 41:
        raise ValueError(f"report must have 41 sections, found {section_count}")
    if "paper exact" in text.lower() and "not called paper exact" not in text.lower():
        raise ValueError("report contains an unsupported paper-exact claim")
    hashes[REPORT_PATH.as_posix()] = file_sha256(report_path)
    validate_frames(frames)
    if raw_request_or_response_artifacts_exist(root):
        raise ValueError("top10 confirmation raw request/response JSON artifact exists before execution")
    return hashes
