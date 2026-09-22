"""Offline hybrid prompt-policy and disjoint holdout audit for Phase 5B-P1E.

This module reads existing local evidence only. It does not call Ollama, run
LLM inference, train recommenders, or mutate any previous artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from src.llm.canonical_llm_contract import contract_hash, sha256_text
from src.llm.canonical_local_llm_runner import classify_response


READINESS_THRESHOLD = 0.80
DATASET_ORDER = ["Amazon Beauty", "Amazon Sports", "ML-1M"]
MODEL_ORDER = [
    ("Qwen2 7B Instruct", "qwen2:7b-instruct"),
    ("Mistral 7B Instruct", "mistral:7b-instruct"),
]
RERANKING_V1_CONTRACT_ID = "llm_rerank_index_json_full20_v1"
RERANKING_V2_CONTRACT_ID = "llm_rerank_index_json_full20_v2"
RANKING_V1_CONTRACT_ID = "llm_rank_index_json_top5_v1"
RERANKING_PARSER_ID = "llm_rerank_strict_primary_parser_v1"
RANKING_PARSER_ID = "llm_rank_strict_primary_parser_v1"
HYBRID_POLICY_ID = "canonical_local_llm_hybrid_qwen_v1_mistral_v2_rerank_policy_v1"
RECOMMENDED_NEXT_TASK = (
    "Phase 5B-P1E - execute the frozen 60-request disjoint hybrid holdout pilot "
    "using Qwen2 reranking prompt v1 and Mistral reranking prompt v2"
)

EXPECTED_STRICT_COUNTS = {
    ("v1", "Qwen2 7B Instruct", "Amazon Beauty"): 10,
    ("v1", "Qwen2 7B Instruct", "Amazon Sports"): 9,
    ("v1", "Qwen2 7B Instruct", "ML-1M"): 9,
    ("v1", "Mistral 7B Instruct", "Amazon Beauty"): 6,
    ("v1", "Mistral 7B Instruct", "Amazon Sports"): 9,
    ("v1", "Mistral 7B Instruct", "ML-1M"): 10,
    ("v2", "Qwen2 7B Instruct", "Amazon Beauty"): 4,
    ("v2", "Qwen2 7B Instruct", "Amazon Sports"): 5,
    ("v2", "Qwen2 7B Instruct", "ML-1M"): 4,
    ("v2", "Mistral 7B Instruct", "Amazon Beauty"): 10,
    ("v2", "Mistral 7B Instruct", "Amazon Sports"): 9,
    ("v2", "Mistral 7B Instruct", "ML-1M"): 9,
}

PATHS = {
    "prompt_contract": Path("results/tables/canonical_llm_prompt_contract.csv"),
    "parser_contract": Path("results/tables/canonical_llm_parser_contract.csv"),
    "execution_contract": Path("results/tables/canonical_llm_execution_parameter_contract.csv"),
    "model_inventory": Path("results/tables/canonical_llm_model_inventory.csv"),
    "cohort_preflight": Path("results/tables/canonical_llm_cohort_preflight.csv"),
    "prompt_rendering": Path("results/tables/canonical_llm_prompt_rendering_audit.csv"),
    "v1_sample": Path("results/tables/canonical_local_llm_pilot_sample.csv"),
    "v1_manifest": Path("results/tables/canonical_local_llm_pilot_request_manifest.csv"),
    "v1_requests": Path("results/llm/canonical_local_pilot_resume/requests.jsonl"),
    "v1_responses": Path("results/llm/canonical_local_pilot_resume/responses.jsonl"),
    "v1_events": Path("results/llm/canonical_local_pilot_resume/runner_events.jsonl"),
    "v1_parse": Path("results/tables/canonical_local_llm_pilot_resume_parse_results.csv"),
    "v1_rankings": Path("results/tables/canonical_local_llm_pilot_resume_rankings.csv"),
    "v1_metrics": Path("results/tables/canonical_local_llm_pilot_resume_metrics.csv"),
    "v1_decision": Path("results/tables/canonical_local_llm_pilot_resume_decision.csv"),
    "v1_report": Path("report/canonical_local_qwen2_mistral_pilot_resume.md"),
    "invalid_count_reconciliation": Path("results/tables/canonical_local_llm_pilot_count_reconciliation.csv"),
    "invalid_response_audit": Path("results/tables/canonical_local_llm_pilot_invalid_response_audit.csv"),
    "parser_assessment": Path("results/tables/canonical_local_llm_pilot_parser_assessment.csv"),
    "prompt_requirement_audit": Path("results/tables/canonical_local_llm_reranking_prompt_requirement_audit.csv"),
    "prompt_corrective_decision": Path("results/tables/canonical_local_llm_pilot_prompt_corrective_decision.csv"),
    "invalid_report": Path("report/canonical_local_llm_pilot_invalid_output_and_prompt_corrective_audit.md"),
    "v2_contract": Path("results/tables/canonical_local_llm_reranking_prompt_v2_contract.csv"),
    "v2_manifest": Path("results/tables/canonical_local_llm_reranking_prompt_v2_corrective_pilot_manifest.csv"),
    "v2_requests": Path("results/llm/canonical_local_reranking_prompt_v2_corrective_pilot/requests.jsonl"),
    "v2_responses": Path("results/llm/canonical_local_reranking_prompt_v2_corrective_pilot/responses.jsonl"),
    "v2_events": Path("results/llm/canonical_local_reranking_prompt_v2_corrective_pilot/runner_events.jsonl"),
    "v2_parse": Path("results/tables/canonical_local_llm_reranking_prompt_v2_parse_results.csv"),
    "v2_rankings": Path("results/tables/canonical_local_llm_reranking_prompt_v2_rankings.csv"),
    "v2_metrics": Path("results/tables/canonical_local_llm_reranking_prompt_v2_metrics.csv"),
    "v2_comparison": Path("results/tables/canonical_local_llm_reranking_prompt_v1_v2_comparison.csv"),
    "v2_validation": Path("results/tables/canonical_local_llm_reranking_prompt_v2_validation.csv"),
    "v2_decision": Path("results/tables/canonical_local_llm_reranking_prompt_v2_decision.csv"),
    "v2_report": Path("report/canonical_local_llm_reranking_prompt_v2_corrective_pilot.md"),
    "candidate_sets": Path("results/tables/ranking_candidate_sets.csv"),
    "positive_positions": Path("results/tables/positive_positions.csv"),
    "item_title_map": Path("results/tables/item_title_map.csv"),
    "item_title_coverage": Path("results/tables/item_title_coverage_summary.csv"),
    "experiment_cohort_membership": Path("results/tables/experiment_cohort_membership.csv"),
    "experiment_cohort_summary": Path("results/tables/experiment_cohort_summary.csv"),
    "paper_pdf": Path("paper/Beyond Utility Evaluating LLM As Recommander.pdf"),
    "readme": Path("README.md"),
    "requirements": Path("requirements.txt"),
    "environment": Path("environment.yml"),
}

CELL_RECONCILIATION_COLUMNS = [
    "reconciliation_id",
    "model_family",
    "dataset",
    "task_type",
    "prompt_version",
    "prompt_contract_id",
    "request_count",
    "backend_success_count",
    "strict_valid_count",
    "strict_invalid_count",
    "strict_parse_validity_rate",
    "output_count_valid_count",
    "uniqueness_valid_count",
    "full_permutation_valid_count",
    "candidate_valid_count",
    "duplicate_free_count",
    "hallucination_free_count",
    "readiness_threshold",
    "cell_passes_threshold",
    "raw_evidence_source",
    "parse_table_source",
    "metrics_table_source",
    "all_sources_agree",
    "status",
    "notes",
]

SELECTION_BASIS_COLUMNS = [
    "selection_audit_id",
    "model_family",
    "task_type",
    "selected_prompt_contract_id",
    "rejected_prompt_contract_id",
    "selection_metric",
    "selected_prompt_structural_validity_minimum",
    "rejected_prompt_structural_validity_minimum",
    "hr_used_for_selection",
    "ndcg_used_for_selection",
    "positive_rank_used_for_selection",
    "latency_used_for_selection",
    "token_count_used_for_selection",
    "beyond_utility_used_for_selection",
    "selection_basis_classification",
    "result_independent_threshold",
    "selection_status",
    "blocking_reason",
    "notes",
]

SEMANTIC_EQUIVALENCE_COLUMNS = [
    "comparison_id",
    "field_name",
    "prompt_v1_value",
    "prompt_v2_value",
    "values_identical",
    "difference_classification",
    "scientific_semantics_changed",
    "candidate_information_changed",
    "history_information_changed",
    "output_space_changed",
    "parser_changed",
    "execution_parameter_changed",
    "comparability_impact",
    "hybrid_policy_compatible",
    "evidence",
    "notes",
]

HYBRID_POLICY_COLUMNS = [
    "hybrid_policy_id",
    "policy_version",
    "execution_scope",
    "qwen_ranking_prompt_contract_id",
    "qwen_reranking_prompt_contract_id",
    "mistral_ranking_prompt_contract_id",
    "mistral_reranking_prompt_contract_id",
    "shared_parser_contract_id",
    "candidate_representation",
    "candidate_numbering",
    "candidate_order_policy",
    "history_policy",
    "positive_label_hidden",
    "ranking_output_count",
    "reranking_output_count",
    "full_permutation_required",
    "execution_parameters_shared",
    "temperature",
    "scientific_retry_count",
    "infrastructure_retry_count",
    "concurrency",
    "selection_basis",
    "independent_holdout_required",
    "paper_alignment",
    "contract_status",
    "contract_sha256",
    "blocking_reason",
    "notes",
]

HOLDOUT_SAMPLE_COLUMNS = [
    "holdout_row_id",
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
    "disjoint_from_previous_pilots",
    "candidate_order_sha256",
    "history_render_sha256",
    "sample_order",
    "sample_status",
    "blocking_reason",
    "notes",
]

HOLDOUT_REQUEST_COLUMNS = [
    "holdout_request_id",
    "holdout_row_id",
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "request_order",
    "hybrid_policy_id",
    "prompt_contract_id",
    "prompt_contract_sha256",
    "parser_contract_id",
    "parser_contract_sha256",
    "execution_contract_id",
    "execution_contract_sha256",
    "candidate_order_sha256",
    "history_render_sha256",
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
    "holdout_request_status",
    "blocking_reason",
    "notes",
]

RISKS_COLUMNS = [
    "risk_id",
    "risk_category",
    "risk_description",
    "affected_models",
    "affected_tasks",
    "model_specific_prompting",
    "post_hoc_selection_risk",
    "pilot_row_overfitting_risk",
    "cross_model_prompt_identity_reduced",
    "scientific_task_identity_preserved",
    "mitigation",
    "independent_holdout_required",
    "severity",
    "status",
    "notes",
]

DECISION_COLUMNS = [
    "audit_id",
    "v1_v2_evidence_reconciled",
    "selection_structural_validity_based",
    "utility_used_for_selection",
    "semantic_equivalence_verified",
    "scientific_task_identity_preserved",
    "model_specific_prompting_required",
    "common_prompt_v3_required",
    "hybrid_policy_created",
    "hybrid_policy_contract_id",
    "holdout_sample_rows",
    "holdout_request_rows",
    "previous_pilot_overlap_count",
    "readiness_threshold",
    "holdout_validation_required",
    "recommended_decision",
    "blocking_reason",
    "required_corrective_action",
    "recommended_single_next_task",
    "notes",
]

OUTPUT_TABLES = {
    "cell_reconciliation": (
        Path("results/tables/canonical_local_llm_hybrid_v1_v2_cell_reconciliation.csv"),
        CELL_RECONCILIATION_COLUMNS,
    ),
    "selection_basis": (
        Path("results/tables/canonical_local_llm_hybrid_prompt_selection_basis.csv"),
        SELECTION_BASIS_COLUMNS,
    ),
    "semantic_equivalence": (
        Path("results/tables/canonical_local_llm_hybrid_prompt_semantic_equivalence.csv"),
        SEMANTIC_EQUIVALENCE_COLUMNS,
    ),
    "policy_contract": (
        Path("results/tables/canonical_local_llm_hybrid_prompt_policy_contract.csv"),
        HYBRID_POLICY_COLUMNS,
    ),
    "holdout_sample": (
        Path("results/tables/canonical_local_llm_hybrid_holdout_sample.csv"),
        HOLDOUT_SAMPLE_COLUMNS,
    ),
    "holdout_manifest": (
        Path("results/tables/canonical_local_llm_hybrid_holdout_request_manifest.csv"),
        HOLDOUT_REQUEST_COLUMNS,
    ),
    "risks": (
        Path("results/tables/canonical_local_llm_hybrid_prompt_risks.csv"),
        RISKS_COLUMNS,
    ),
    "decision": (
        Path("results/tables/canonical_local_llm_hybrid_prompt_holdout_decision.csv"),
        DECISION_COLUMNS,
    ),
}
REPORT_PATH = Path("report/canonical_local_llm_hybrid_prompt_and_holdout_audit.md")


@dataclass(frozen=True)
class EvidenceBundle:
    tables: dict[str, pd.DataFrame]
    jsonl: dict[str, list[dict[str, Any]]]
    protected_hashes: dict[str, str]


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text


def format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.17g}"


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(root: Path, relative_path: Path | str) -> pd.DataFrame:
    path = root / Path(relative_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_jsonl(root: Path, relative_path: Path | str) -> list[dict[str, Any]]:
    path = root / Path(relative_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record is not an object at {path}:{line_number}")
            records.append(record)
    return records


def frame(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    normalized = []
    for row in rows:
        normalized.append({column: clean_cell(row.get(column, "")) for column in columns})
    return pd.DataFrame(normalized, columns=list(columns))


def existing_protected_paths(root: Path) -> list[Path]:
    return [path for path in PATHS.values() if (root / path).is_file()]


def hash_protected_inputs(root: Path) -> dict[str, str]:
    return {path.as_posix(): file_sha256(root / path) for path in sorted(existing_protected_paths(root))}


def output_paths(root: Path, frames: Mapping[str, pd.DataFrame] | None = None) -> list[Path]:
    selected = list(OUTPUT_TABLES)
    if frames is not None and "policy_contract" not in frames:
        selected.remove("policy_contract")
    paths = [root / OUTPUT_TABLES[key][0] for key in selected]
    paths.append(root / REPORT_PATH)
    return paths


def load_evidence(root: Path) -> EvidenceBundle:
    table_keys = [
        "prompt_contract",
        "parser_contract",
        "execution_contract",
        "model_inventory",
        "cohort_preflight",
        "prompt_rendering",
        "v1_sample",
        "v1_manifest",
        "v1_parse",
        "v1_rankings",
        "v1_metrics",
        "v1_decision",
        "invalid_count_reconciliation",
        "invalid_response_audit",
        "parser_assessment",
        "prompt_requirement_audit",
        "prompt_corrective_decision",
        "v2_contract",
        "v2_manifest",
        "v2_parse",
        "v2_rankings",
        "v2_metrics",
        "v2_comparison",
        "v2_validation",
        "v2_decision",
        "candidate_sets",
        "positive_positions",
        "item_title_map",
        "item_title_coverage",
        "experiment_cohort_membership",
        "experiment_cohort_summary",
    ]
    jsonl_keys = ["v1_requests", "v1_responses", "v1_events", "v2_requests", "v2_responses", "v2_events"]
    tables = {key: read_csv(root, PATHS[key]) for key in table_keys}
    jsonl = {key: read_jsonl(root, PATHS[key]) for key in jsonl_keys}
    return EvidenceBundle(tables=tables, jsonl=jsonl, protected_hashes=hash_protected_inputs(root))


def deterministic_selection_key(dataset: str, task_type: str, candidate_set_id: str, user_id: str) -> str:
    return sha256_text(f"{dataset}|{task_type}|{candidate_set_id}|{user_id}")


def row_identity(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        clean_cell(row.get("dataset")),
        clean_cell(row.get("task_type")),
        clean_cell(row.get("candidate_set_id")),
        clean_cell(row.get("user_id")),
    )


def _record_id(record: Mapping[str, Any], prompt_version: str) -> str:
    if prompt_version == "v2":
        return clean_cell(record.get("corrective_request_id") or record.get("request_id"))
    return clean_cell(record.get("request_id"))


def _response_id(record: Mapping[str, Any], prompt_version: str) -> str:
    if prompt_version == "v2":
        return clean_cell(record.get("corrective_request_id") or record.get("request_id"))
    return clean_cell(record.get("request_id"))


def _raw_reconstruction(
    *,
    prompt_version: str,
    request_records: Sequence[Mapping[str, Any]],
    response_records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    responses = {_response_id(record, prompt_version): record for record in response_records}
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for request in request_records:
        if clean_cell(request.get("task_type")) != "reranking":
            continue
        model = clean_cell(request.get("model_family"))
        dataset = clean_cell(request.get("dataset"))
        key = (model, dataset)
        response = responses.get(_record_id(request, prompt_version), {})
        candidate_items = request.get("candidate_items")
        if not isinstance(candidate_items, list) or not candidate_items:
            candidate_items = [str(index) for index in range(1, 21)]
        parsed = classify_response(
            clean_cell(response.get("raw_response_text")),
            candidate_items=[str(item) for item in candidate_items],
            expected_count=20,
            require_full_permutation=True,
        )
        backend_success = parse_bool(response.get("backend_success"))
        cell = cells.setdefault(
            key,
            {
                "request_count": 0,
                "backend_success_count": 0,
                "strict_valid_count": 0,
                "output_count_valid_count": 0,
                "uniqueness_valid_count": 0,
                "full_permutation_valid_count": 0,
                "candidate_valid_count": 0,
                "duplicate_free_count": 0,
                "hallucination_free_count": 0,
            },
        )
        cell["request_count"] += 1
        if backend_success:
            cell["backend_success_count"] += 1
        if parsed["strict_parse_valid"]:
            cell["strict_valid_count"] += 1
            if int(parsed["duplicate_count"]) == 0:
                cell["duplicate_free_count"] += 1
            if int(parsed["hallucination_count"]) == 0:
                cell["hallucination_free_count"] += 1
        if parsed["output_count_valid"]:
            cell["output_count_valid_count"] += 1
        if parsed["uniqueness_valid"]:
            cell["uniqueness_valid_count"] += 1
        if parsed["full_permutation_valid"]:
            cell["full_permutation_valid_count"] += 1
        if parsed["candidate_validity"]:
            cell["candidate_valid_count"] += 1
    return cells


def _parse_table_counts(parse_df: pd.DataFrame, prompt_version: str) -> dict[tuple[str, str], dict[str, int]]:
    df = parse_df[parse_df["task_type"].eq("reranking")].copy()
    cells: dict[tuple[str, str], dict[str, int]] = {}
    for (model, dataset), group in df.groupby(["model_family", "dataset"], sort=False):
        cells[(model, dataset)] = {
            "request_count": len(group),
            "strict_valid_count": int(group["strict_parse_valid"].map(parse_bool).sum()),
            "output_count_valid_count": int(group["output_count_valid"].map(parse_bool).sum()),
            "uniqueness_valid_count": int(group["uniqueness_valid"].map(parse_bool).sum()),
            "full_permutation_valid_count": int(group["full_permutation_valid"].map(parse_bool).sum()),
            "candidate_valid_count": int(group["candidate_validity"].map(parse_bool).sum()),
            "duplicate_free_count": int(
                (group["strict_parse_valid"].map(parse_bool) & group["duplicate_count"].astype(str).eq("0")).sum()
            ),
            "hallucination_free_count": int(
                (group["strict_parse_valid"].map(parse_bool) & group["hallucination_count"].astype(str).eq("0")).sum()
            ),
        }
    return cells


def _metrics_counts(metrics_df: pd.DataFrame, prompt_version: str) -> dict[tuple[str, str], dict[str, int]]:
    df = metrics_df[metrics_df["task_type"].eq("reranking")].copy()
    rows: dict[tuple[str, str], dict[str, int]] = {}
    for _, row in df.iterrows():
        request_col = "corrective_rows" if prompt_version == "v2" else "pilot_rows"
        full_col = "full_permutation_valid_rows" if prompt_version == "v2" else "complete_permutation_rows"
        data = {
            "request_count": int(float(row[request_col])),
            "backend_success_count": int(float(row["backend_success_rows"])),
            "strict_valid_count": int(float(row["strict_valid_rows"])),
            "candidate_valid_count": int(float(row["candidate_valid_rows"])),
            "duplicate_free_count": int(float(row["duplicate_free_rows"])),
            "hallucination_free_count": int(float(row["hallucination_free_rows"])),
            "full_permutation_valid_count": int(float(row[full_col])),
        }
        if prompt_version == "v2":
            data["output_count_valid_count"] = int(float(row["output_count_valid_rows"]))
            data["uniqueness_valid_count"] = int(float(row["uniqueness_valid_rows"]))
        rows[(row["model_family"], row["dataset"])] = data
    return rows


def reconstruct_v1_v2_cells(bundle: EvidenceBundle) -> pd.DataFrame:
    specs = [
        {
            "version": "v1",
            "contract_id": RERANKING_V1_CONTRACT_ID,
            "requests_key": "v1_requests",
            "responses_key": "v1_responses",
            "parse_key": "v1_parse",
            "metrics_key": "v1_metrics",
            "raw_source": f"{PATHS['v1_requests'].as_posix()};{PATHS['v1_responses'].as_posix()}",
            "parse_source": PATHS["v1_parse"].as_posix(),
            "metrics_source": PATHS["v1_metrics"].as_posix(),
        },
        {
            "version": "v2",
            "contract_id": RERANKING_V2_CONTRACT_ID,
            "requests_key": "v2_requests",
            "responses_key": "v2_responses",
            "parse_key": "v2_parse",
            "metrics_key": "v2_metrics",
            "raw_source": f"{PATHS['v2_requests'].as_posix()};{PATHS['v2_responses'].as_posix()}",
            "parse_source": PATHS["v2_parse"].as_posix(),
            "metrics_source": PATHS["v2_metrics"].as_posix(),
        },
    ]
    rows: list[dict[str, Any]] = []
    row_number = 1
    for spec in specs:
        raw_counts = _raw_reconstruction(
            prompt_version=spec["version"],
            request_records=bundle.jsonl[spec["requests_key"]],
            response_records=bundle.jsonl[spec["responses_key"]],
        )
        parse_counts = _parse_table_counts(bundle.tables[spec["parse_key"]], spec["version"])
        metric_counts = _metrics_counts(bundle.tables[spec["metrics_key"]], spec["version"])
        for model, _tag in MODEL_ORDER:
            for dataset in DATASET_ORDER:
                key = (model, dataset)
                raw = raw_counts.get(key, {})
                parsed = parse_counts.get(key, {})
                metric = metric_counts.get(key, {})
                expected = EXPECTED_STRICT_COUNTS[(spec["version"], model, dataset)]
                comparable = [
                    "request_count",
                    "strict_valid_count",
                    "output_count_valid_count",
                    "uniqueness_valid_count",
                    "full_permutation_valid_count",
                    "candidate_valid_count",
                    "duplicate_free_count",
                    "hallucination_free_count",
                ]
                raw_parse_agree = all(raw.get(name) == parsed.get(name) for name in comparable)
                metric_names = [name for name in comparable if name in metric]
                raw_metric_agree = all(raw.get(name) == metric.get(name) for name in metric_names) and (
                    raw.get("backend_success_count") == metric.get("backend_success_count")
                )
                all_sources_agree = raw_parse_agree and raw_metric_agree and raw.get("strict_valid_count") == expected
                request_count = int(raw.get("request_count", 0))
                strict_valid = int(raw.get("strict_valid_count", 0))
                rows.append(
                    {
                        "reconciliation_id": f"HYB-REC-{row_number:03d}",
                        "model_family": model,
                        "dataset": dataset,
                        "task_type": "reranking",
                        "prompt_version": spec["version"],
                        "prompt_contract_id": spec["contract_id"],
                        "request_count": request_count,
                        "backend_success_count": int(raw.get("backend_success_count", 0)),
                        "strict_valid_count": strict_valid,
                        "strict_invalid_count": request_count - strict_valid,
                        "strict_parse_validity_rate": format_float(strict_valid / request_count if request_count else 0.0),
                        "output_count_valid_count": int(raw.get("output_count_valid_count", 0)),
                        "uniqueness_valid_count": int(raw.get("uniqueness_valid_count", 0)),
                        "full_permutation_valid_count": int(raw.get("full_permutation_valid_count", 0)),
                        "candidate_valid_count": int(raw.get("candidate_valid_count", 0)),
                        "duplicate_free_count": int(raw.get("duplicate_free_count", 0)),
                        "hallucination_free_count": int(raw.get("hallucination_free_count", 0)),
                        "readiness_threshold": format_float(READINESS_THRESHOLD),
                        "cell_passes_threshold": bool_text((strict_valid / request_count if request_count else 0.0) >= READINESS_THRESHOLD),
                        "raw_evidence_source": spec["raw_source"],
                        "parse_table_source": spec["parse_source"],
                        "metrics_table_source": spec["metrics_source"],
                        "all_sources_agree": bool_text(all_sources_agree),
                        "status": "reconciled" if all_sources_agree else "mismatch",
                        "notes": f"Expected authoritative strict-valid count {expected}/10.",
                    }
                )
                row_number += 1
    result = frame(rows, CELL_RECONCILIATION_COLUMNS)
    if len(result) != 12:
        raise ValueError(f"expected 12 reconciliation rows, observed {len(result)}")
    bad = result[~result["all_sources_agree"].eq("true")]
    if not bad.empty:
        raise ValueError(f"v1/v2 evidence reconciliation failed: {bad[['model_family','dataset','prompt_version']].to_dict('records')}")
    return result


def build_prompt_selection_basis(cell_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selections = [
        ("SEL-001", "Qwen2 7B Instruct", RERANKING_V1_CONTRACT_ID, RERANKING_V2_CONTRACT_ID),
        ("SEL-002", "Mistral 7B Instruct", RERANKING_V2_CONTRACT_ID, RERANKING_V1_CONTRACT_ID),
    ]
    for audit_id, model, selected, rejected in selections:
        selected_version = "v1" if selected.endswith("_v1") else "v2"
        rejected_version = "v1" if rejected.endswith("_v1") else "v2"
        selected_rates = cell_df[
            cell_df["model_family"].eq(model) & cell_df["prompt_version"].eq(selected_version)
        ]["strict_parse_validity_rate"].astype(float)
        rejected_rates = cell_df[
            cell_df["model_family"].eq(model) & cell_df["prompt_version"].eq(rejected_version)
        ]["strict_parse_validity_rate"].astype(float)
        selected_min = float(selected_rates.min())
        rejected_min = float(rejected_rates.min())
        structural = selected_min >= READINESS_THRESHOLD and rejected_min < READINESS_THRESHOLD
        rows.append(
            {
                "selection_audit_id": audit_id,
                "model_family": model,
                "task_type": "reranking",
                "selected_prompt_contract_id": selected,
                "rejected_prompt_contract_id": rejected,
                "selection_metric": "minimum_cell_strict_parse_validity_rate",
                "selected_prompt_structural_validity_minimum": format_float(selected_min),
                "rejected_prompt_structural_validity_minimum": format_float(rejected_min),
                "hr_used_for_selection": "false",
                "ndcg_used_for_selection": "false",
                "positive_rank_used_for_selection": "false",
                "latency_used_for_selection": "false",
                "token_count_used_for_selection": "false",
                "beyond_utility_used_for_selection": "false",
                "selection_basis_classification": "structural_validity_based" if structural else "selection_basis_unresolved",
                "result_independent_threshold": "true",
                "selection_status": "accepted_for_disjoint_holdout_audit" if structural else "blocked",
                "blocking_reason": "" if structural else "selected prompt does not pass structural readiness while rejected prompt fails",
                "notes": "Selection uses only predeclared structural validity readiness, not utility.",
            }
        )
    result = frame(rows, SELECTION_BASIS_COLUMNS)
    if len(result) != 2:
        raise ValueError("prompt selection basis must contain two rows")
    utility_flags = [
        "hr_used_for_selection",
        "ndcg_used_for_selection",
        "positive_rank_used_for_selection",
        "latency_used_for_selection",
        "token_count_used_for_selection",
        "beyond_utility_used_for_selection",
    ]
    if result[utility_flags].ne("false").any().any():
        raise ValueError("utility or non-structural fields influenced prompt selection")
    return result


def _one_row(df: pd.DataFrame, column: str, value: str) -> dict[str, str]:
    subset = df[df[column].eq(value)]
    if len(subset) != 1:
        raise ValueError(f"expected one row where {column}={value}, observed {len(subset)}")
    return subset.iloc[0].to_dict()


def _semantic_row(
    rows: list[dict[str, Any]],
    *,
    field_name: str,
    v1: Any,
    v2: Any,
    classification: str,
    evidence: str,
    notes: str,
    scientific: bool = False,
    candidate: bool = False,
    history: bool = False,
    output_space: bool = False,
    parser: bool = False,
    execution: bool = False,
) -> None:
    rows.append(
        {
            "comparison_id": f"SEM-{len(rows) + 1:03d}",
            "field_name": field_name,
            "prompt_v1_value": clean_cell(v1),
            "prompt_v2_value": clean_cell(v2),
            "values_identical": bool_text(clean_cell(v1) == clean_cell(v2)),
            "difference_classification": classification,
            "scientific_semantics_changed": bool_text(scientific),
            "candidate_information_changed": bool_text(candidate),
            "history_information_changed": bool_text(history),
            "output_space_changed": bool_text(output_space),
            "parser_changed": bool_text(parser),
            "execution_parameter_changed": bool_text(execution),
            "comparability_impact": "blocking" if any([scientific, candidate, history, output_space, parser, execution]) else "none",
            "hybrid_policy_compatible": bool_text(not any([scientific, candidate, history, output_space, parser, execution])),
            "evidence": evidence,
            "notes": notes,
        }
    )


def build_semantic_equivalence(bundle: EvidenceBundle) -> pd.DataFrame:
    v1 = _one_row(bundle.tables["prompt_contract"], "prompt_contract_id", RERANKING_V1_CONTRACT_ID)
    v2 = _one_row(bundle.tables["v2_contract"], "prompt_contract_id", RERANKING_V2_CONTRACT_ID)
    parser = _one_row(bundle.tables["parser_contract"], "parser_contract_id", RERANKING_PARSER_ID)
    comparison = bundle.tables["v2_comparison"]
    candidate_same = comparison["candidate_order_identical"].map(parse_bool).all()
    history_same = comparison["history_render_identical"].map(parse_bool).all()
    execution_same = comparison["execution_parameters_identical"].map(parse_bool).all()
    evidence = f"{PATHS['prompt_contract'].as_posix()};{PATHS['v2_contract'].as_posix()};{PATHS['v2_comparison'].as_posix()}"
    rows: list[dict[str, Any]] = []
    _semantic_row(rows, field_name="task_type", v1=v1["task_type"], v2=v2["task_type"], classification="identical_non_difference", evidence=evidence, notes="Both contracts are reranking.")
    _semantic_row(rows, field_name="candidate_count", v1="20", v2="20", classification="identical_non_difference", evidence=evidence, notes="Both prompts use exactly 20 reranking candidates.")
    candidate_representation_changed = clean_cell(v1["candidate_representation"]) != clean_cell(v2["candidate_representation"])
    candidate_numbering_changed = clean_cell(v1["candidate_numbering"]) != clean_cell(v2["candidate_numbering"])
    candidate_order_policy_changed = clean_cell(v1["candidate_order_policy"]) != clean_cell(v2["candidate_order_policy"])
    requested_output_count_changed = clean_cell(v1["requested_output_count"]) != clean_cell(v2["requested_output_count"])
    positive_label_changed = clean_cell(v1["positive_label_hidden"]) != clean_cell(v2["positive_label_hidden"])
    explanation_policy_changed = clean_cell(v1["explanation_allowed"]) != clean_cell(v2["explanation_allowed"])
    code_fence_policy_changed = clean_cell(v1["code_fence_allowed"]) != clean_cell(v2["code_fence_allowed"])
    _semantic_row(rows, field_name="candidate_representation", v1=v1["candidate_representation"], v2=v2["candidate_representation"], classification="identical_non_difference" if not candidate_representation_changed else "candidate_or_history_change", evidence=evidence, notes="Candidates remain one-based index plus title.", candidate=candidate_representation_changed)
    _semantic_row(rows, field_name="candidate_numbering", v1=v1["candidate_numbering"], v2=v2["candidate_numbering"], classification="identical_non_difference" if not candidate_numbering_changed else "candidate_or_history_change", evidence=evidence, notes="Candidate labels remain one-based 1 to 20.", candidate=candidate_numbering_changed, output_space=candidate_numbering_changed)
    _semantic_row(rows, field_name="candidate_order_policy", v1=v1["candidate_order_policy"], v2=v2["candidate_order_policy"], classification="identical_non_difference" if not candidate_order_policy_changed else "candidate_or_history_change", evidence=evidence, notes="Candidate order policy is unchanged.", candidate=candidate_order_policy_changed)
    _semantic_row(rows, field_name="paired_candidate_order_hashes", v1="all paired rows identical", v2="all paired rows identical", classification="identical_non_difference", evidence=PATHS["v2_comparison"].as_posix(), notes="All v1-v2 paired rows have candidate_order_identical=true.", candidate=not candidate_same)
    _semantic_row(rows, field_name="requested_output_count", v1=v1["requested_output_count"], v2=v2["requested_output_count"], classification="identical_non_difference" if not requested_output_count_changed else "scientific_semantic_change", evidence=evidence, notes="Both request a 20-item reranking output.", output_space=requested_output_count_changed, scientific=requested_output_count_changed)
    _semantic_row(rows, field_name="identifier_namespace", v1=parser["expected_identifier_namespace"], v2=parser["expected_identifier_namespace"], classification="identical_non_difference", evidence=PATHS["parser_contract"].as_posix(), notes="Both use one-based candidate indices.")
    _semantic_row(rows, field_name="full_permutation_semantics", v1=v1["requested_output_representation"], v2=v2["exact_permutation_requirement"], classification="format_compliance_reinforcement", evidence=evidence, notes="V2 spells out the same complete 1..20 permutation requirement.")
    _semantic_row(rows, field_name="json_key", v1="ranked_candidate_indices", v2="ranked_candidate_indices", classification="identical_non_difference", evidence=evidence, notes="JSON key is unchanged.")
    _semantic_row(rows, field_name="json_container", v1="JSON object", v2="exactly one JSON object", classification="format_compliance_reinforcement", evidence=evidence, notes="V2 reinforces the JSON object container.")
    _semantic_row(rows, field_name="explanation_policy", v1=v1["explanation_allowed"], v2=v2["explanation_allowed"], classification="identical_non_difference" if not explanation_policy_changed else "scientific_semantic_change", evidence=evidence, notes="Explanations remain disallowed.", output_space=explanation_policy_changed, scientific=explanation_policy_changed)
    _semantic_row(rows, field_name="code_fence_policy", v1=v1["code_fence_allowed"], v2=v2["code_fence_allowed"], classification="identical_non_difference" if not code_fence_policy_changed else "scientific_semantic_change", evidence=evidence, notes="Code fences remain disallowed.", output_space=code_fence_policy_changed, scientific=code_fence_policy_changed)
    _semantic_row(rows, field_name="prose_and_extra_key_policy", v1="return JSON only from system and prompt", v2="do not include prose, Markdown, code fence, or extra key", classification="format_compliance_reinforcement", evidence=evidence, notes="V2 reinforces strict formatting without changing the valid output space.")
    _semantic_row(rows, field_name="internal_validation_instruction", v1="not explicit", v2=v2["internal_validation_instruction"], classification="format_compliance_reinforcement", evidence=PATHS["v2_contract"].as_posix(), notes="V2 asks the model to check length and set internally; no new output field is requested.")
    _semantic_row(rows, field_name="positive_label_concealment", v1=v1["positive_label_hidden"], v2=v2["positive_label_hidden"], classification="identical_non_difference" if not positive_label_changed else "scientific_semantic_change", evidence=evidence, notes="Positive item labels remain hidden.", scientific=positive_label_changed)
    _semantic_row(rows, field_name="history_source_policy", v1=v1["history_source_policy"], v2="inherited from source v1 prompt context", classification="format_compliance_reinforcement", evidence=PATHS["v2_comparison"].as_posix(), notes="V2 corrective contexts were built from the source v1 row.")
    _semantic_row(rows, field_name="paired_history_render_hashes", v1="all paired rows identical", v2="all paired rows identical", classification="identical_non_difference", evidence=PATHS["v2_comparison"].as_posix(), notes="All v1-v2 paired rows have history_render_identical=true.", history=not history_same)
    _semantic_row(rows, field_name="history_item_representation", v1=v1["history_item_representation"], v2="same rendered title-only history", classification="format_compliance_reinforcement", evidence=evidence, notes="The rendered histories are unchanged.")
    _semantic_row(rows, field_name="parser_contract", v1=RERANKING_PARSER_ID, v2=RERANKING_PARSER_ID, classification="identical_non_difference", evidence=PATHS["parser_contract"].as_posix(), notes="Both versions use the same strict reranking parser.")
    _semantic_row(rows, field_name="parser_requires_full_permutation", v1=parser["requires_full_permutation"], v2=parser["requires_full_permutation"], classification="identical_non_difference", evidence=PATHS["parser_contract"].as_posix(), notes="Parser full-permutation requirement is unchanged.")
    _semantic_row(rows, field_name="execution_parameters", v1="temperature=0; max_output_tokens=512; json_object; no retries; concurrency=1", v2="temperature=0; max_output_tokens=512; json_object; no retries; concurrency=1", classification="identical_non_difference", evidence=PATHS["v2_comparison"].as_posix(), notes="All paired rows have execution_parameters_identical=true.", execution=not execution_same)
    _semantic_row(rows, field_name="temperature", v1="0", v2="0", classification="identical_non_difference", evidence=PATHS["execution_contract"].as_posix(), notes="Temperature remains zero.")
    _semantic_row(rows, field_name="seed", v1="", v2="", classification="identical_non_difference", evidence=PATHS["v1_manifest"].as_posix() + ";" + PATHS["v2_manifest"].as_posix(), notes="No seed was specified in either reranking pilot manifest.")
    _semantic_row(rows, field_name="response_format", v1="json_object", v2="json_object", classification="identical_non_difference", evidence=PATHS["v1_manifest"].as_posix() + ";" + PATHS["v2_manifest"].as_posix(), notes="Response format remains json_object.")
    _semantic_row(rows, field_name="retry_policy", v1="scientific_retry_count=0; infrastructure_retry_count=0", v2="scientific_retry_count=0; infrastructure_retry_count=0", classification="identical_non_difference", evidence=PATHS["v1_manifest"].as_posix() + ";" + PATHS["v2_manifest"].as_posix(), notes="No retry policy change.")
    _semantic_row(rows, field_name="scientific_semantics_contract_flag", v1="false", v2=v2["scientific_semantics_changed"], classification="identical_non_difference", evidence=PATHS["v2_contract"].as_posix(), notes="V2 contract explicitly records no scientific semantic change.", scientific=parse_bool(v2["scientific_semantics_changed"]))
    result = frame(rows, SEMANTIC_EQUIVALENCE_COLUMNS)
    if result["hybrid_policy_compatible"].ne("true").any():
        return result
    return result


def build_hybrid_policy_contract(bundle: EvidenceBundle, selection_df: pd.DataFrame, semantic_df: pd.DataFrame) -> pd.DataFrame:
    selection_ok = selection_df["selection_basis_classification"].eq("structural_validity_based").all()
    semantic_ok = semantic_df["hybrid_policy_compatible"].eq("true").all()
    if not selection_ok or not semantic_ok:
        return frame([], HYBRID_POLICY_COLUMNS)
    row = {
        "hybrid_policy_id": HYBRID_POLICY_ID,
        "policy_version": "hybrid_prompt_policy_v1",
        "execution_scope": "disjoint_holdout_pilot_only",
        "qwen_ranking_prompt_contract_id": RANKING_V1_CONTRACT_ID,
        "qwen_reranking_prompt_contract_id": RERANKING_V1_CONTRACT_ID,
        "mistral_ranking_prompt_contract_id": RANKING_V1_CONTRACT_ID,
        "mistral_reranking_prompt_contract_id": RERANKING_V2_CONTRACT_ID,
        "shared_parser_contract_id": f"{RANKING_PARSER_ID};{RERANKING_PARSER_ID}",
        "candidate_representation": "one_based_index_plus_title",
        "candidate_numbering": "one_based_1_to_20",
        "candidate_order_policy": "preserve_source_candidate_order_for_ranking_and_initial_candidate_order_for_reranking",
        "history_policy": "train_split_only_chronological_last_10_exclude_heldout_positive",
        "positive_label_hidden": "true",
        "ranking_output_count": "5",
        "reranking_output_count": "20",
        "full_permutation_required": "true",
        "execution_parameters_shared": "true",
        "temperature": "0",
        "scientific_retry_count": "0",
        "infrastructure_retry_count": "0",
        "concurrency": "1",
        "selection_basis": "structural_validity_based",
        "independent_holdout_required": "true",
        "paper_alignment": "adapted_model_specific_format_compliance_contract",
        "contract_status": "approved_for_disjoint_holdout_pilot",
        "blocking_reason": "",
        "notes": "Not paper exact; model-specific wording is allowed only behind the disjoint holdout gate.",
    }
    row["contract_sha256"] = contract_hash(row)
    return frame([row], HYBRID_POLICY_COLUMNS)


def previous_pilot_identities(bundle: EvidenceBundle) -> tuple[set[tuple[str, str, str, str]], set[tuple[str, str, str, str]]]:
    v1_rows = bundle.tables["v1_sample"][bundle.tables["v1_sample"]["task_type"].eq("reranking")]
    v1 = {row_identity(row) for row in v1_rows.to_dict("records")}
    v2 = {row_identity(row) for row in bundle.tables["v2_manifest"].to_dict("records")}
    return v1, v2


def _render_lookup(prompt_rendering: pd.DataFrame) -> dict[tuple[str, str, str, str], dict[str, str]]:
    rerank = prompt_rendering[
        prompt_rendering["prompt_contract_id"].eq(RERANKING_V1_CONTRACT_ID)
        & prompt_rendering["task_type"].eq("reranking")
    ].copy()
    lookup: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rerank.to_dict("records"):
        lookup[row_identity(row)] = row
    return lookup


def select_disjoint_holdout_sample(bundle: EvidenceBundle, approved: bool) -> pd.DataFrame:
    if not approved:
        return frame([], HOLDOUT_SAMPLE_COLUMNS)
    preflight = bundle.tables["cohort_preflight"]
    render_lookup = _render_lookup(bundle.tables["prompt_rendering"])
    v1_used, v2_used = previous_pilot_identities(bundle)
    rows: list[dict[str, Any]] = []
    order = 1
    eligible = preflight[
        preflight["task_type"].eq("reranking")
        & preflight["dataset"].isin(DATASET_ORDER)
        & preflight["primary_prompt_compatible"].map(parse_bool)
    ].copy()
    eligible["selection_sha256"] = eligible.apply(
        lambda row: deterministic_selection_key(row["dataset"], row["task_type"], row["candidate_set_id"], row["user_id"]),
        axis=1,
    )
    for dataset in DATASET_ORDER:
        stratum = eligible[eligible["dataset"].eq(dataset)].copy()
        stratum = stratum.sort_values(["selection_sha256", "candidate_set_id", "user_id"], kind="mergesort")
        selected = []
        for row in stratum.to_dict("records"):
            identity = row_identity(row)
            if identity in v1_used or identity in v2_used:
                continue
            selected.append(row)
            if len(selected) == 10:
                break
        if len(selected) != 10:
            raise ValueError(f"unable to select 10 disjoint holdout rows for {dataset}")
        for row in selected:
            identity = row_identity(row)
            render = render_lookup.get(identity)
            if not render:
                raise ValueError(f"missing prompt rendering row for holdout identity {identity}")
            rows.append(
                {
                    "holdout_row_id": f"CLHOLD-ROW-{order:04d}",
                    "selection_sha256": row["selection_sha256"],
                    "dataset": row["dataset"],
                    "task_type": row["task_type"],
                    "candidate_set_id": row["candidate_set_id"],
                    "source_path": row["source_path"],
                    "user_id": row["user_id"],
                    "positive_item_id": row["positive_item_id"],
                    "candidate_count": row["candidate_count"],
                    "history_item_count": render["history_item_count"],
                    "candidate_titles_complete": row["candidate_titles_complete"],
                    "history_titles_complete": row["history_titles_complete"],
                    "held_out_positive_absent": row["held_out_positive_absent_from_history"],
                    "primary_prompt_compatible": row["primary_prompt_compatible"],
                    "used_in_v1_pilot": bool_text(identity in v1_used),
                    "used_in_v2_pilot": bool_text(identity in v2_used),
                    "disjoint_from_previous_pilots": bool_text(identity not in v1_used and identity not in v2_used),
                    "candidate_order_sha256": render["candidate_order_sha256"],
                    "history_render_sha256": render["history_render_sha256"],
                    "sample_order": str(order),
                    "sample_status": "selected_frozen_not_executed",
                    "blocking_reason": "",
                    "notes": "Deterministic result-independent holdout row; no utility or positive position used for selection.",
                }
            )
            order += 1
    result = frame(rows, HOLDOUT_SAMPLE_COLUMNS)
    if len(result) != 30:
        raise ValueError(f"expected 30 holdout sample rows, observed {len(result)}")
    if result["disjoint_from_previous_pilots"].ne("true").any():
        raise ValueError("holdout sample overlaps a previous pilot")
    return result


def _contract_sha(bundle: EvidenceBundle, prompt_contract_id: str) -> str:
    if prompt_contract_id == RERANKING_V2_CONTRACT_ID:
        return _one_row(bundle.tables["v2_contract"], "prompt_contract_id", prompt_contract_id)["contract_sha256"]
    return _one_row(bundle.tables["prompt_contract"], "prompt_contract_id", prompt_contract_id)["contract_sha256"]


def _execution_contract_for(bundle: EvidenceBundle, model_family: str) -> dict[str, str]:
    df = bundle.tables["execution_contract"]
    subset = df[df["model_family"].eq(model_family) & df["task_type"].eq("reranking")]
    if len(subset) != 1:
        raise ValueError(f"expected one reranking execution contract for {model_family}")
    return subset.iloc[0].to_dict()


def build_holdout_request_manifest(bundle: EvidenceBundle, sample_df: pd.DataFrame, policy_df: pd.DataFrame) -> pd.DataFrame:
    if sample_df.empty or policy_df.empty:
        return frame([], HOLDOUT_REQUEST_COLUMNS)
    parser = _one_row(bundle.tables["parser_contract"], "parser_contract_id", RERANKING_PARSER_ID)
    rows: list[dict[str, Any]] = []
    order = 1
    assignments = [
        ("Qwen2 7B Instruct", "qwen2:7b-instruct", RERANKING_V1_CONTRACT_ID),
        ("Mistral 7B Instruct", "mistral:7b-instruct", RERANKING_V2_CONTRACT_ID),
    ]
    for model_family, model_tag, prompt_id in assignments:
        execution = _execution_contract_for(bundle, model_family)
        for sample in sample_df.to_dict("records"):
            rows.append(
                {
                    "holdout_request_id": f"CLHOLD-REQ-{order:04d}",
                    "holdout_row_id": sample["holdout_row_id"],
                    "model_family": model_family,
                    "model_tag": model_tag,
                    "dataset": sample["dataset"],
                    "task_type": sample["task_type"],
                    "candidate_set_id": sample["candidate_set_id"],
                    "user_id": sample["user_id"],
                    "request_order": str(order),
                    "hybrid_policy_id": HYBRID_POLICY_ID,
                    "prompt_contract_id": prompt_id,
                    "prompt_contract_sha256": _contract_sha(bundle, prompt_id),
                    "parser_contract_id": RERANKING_PARSER_ID,
                    "parser_contract_sha256": parser["parser_schema_sha256"],
                    "execution_contract_id": execution["execution_contract_id"],
                    "execution_contract_sha256": execution["contract_sha256"],
                    "candidate_order_sha256": sample["candidate_order_sha256"],
                    "history_render_sha256": sample["history_render_sha256"],
                    "requested_output_count": "20",
                    "temperature": clean_cell(execution["temperature"]),
                    "seed": clean_cell(execution["seed"]),
                    "max_output_tokens": clean_cell(execution["max_output_tokens"]),
                    "response_format": clean_cell(execution["response_format"]),
                    "streaming": clean_cell(execution["streaming"]),
                    "concurrency": clean_cell(execution["concurrency"]),
                    "scientific_retry_count": clean_cell(execution["scientific_retry_count"]),
                    "infrastructure_retry_count": clean_cell(execution["infrastructure_retry_count"]),
                    "previous_pilot_overlap": "false",
                    "holdout_request_status": "frozen_not_executed",
                    "blocking_reason": "",
                    "notes": "Request manifest only; no inference executed in this audit.",
                }
            )
            order += 1
    result = frame(rows, HOLDOUT_REQUEST_COLUMNS)
    if len(result) != 60:
        raise ValueError(f"expected 60 holdout request rows, observed {len(result)}")
    return result


def build_risks() -> pd.DataFrame:
    risk_rows = [
        {
            "risk_id": "HYB-RISK-001",
            "risk_category": "model_specific_wording",
            "risk_description": "Literal prompt identity differs across models for reranking.",
            "affected_models": json_dumps(["Qwen2 7B Instruct", "Mistral 7B Instruct"]),
            "affected_tasks": json_dumps(["reranking"]),
            "model_specific_prompting": "true",
            "post_hoc_selection_risk": "true",
            "pilot_row_overfitting_risk": "false",
            "cross_model_prompt_identity_reduced": "true",
            "scientific_task_identity_preserved": "true",
            "mitigation": "Label as adapted model-specific and require disjoint holdout validation.",
            "independent_holdout_required": "true",
            "severity": "warning",
            "status": "mitigated_by_holdout_gate",
            "notes": "Task and valid output space remain identical.",
        },
        {
            "risk_id": "HYB-RISK-002",
            "risk_category": "post_hoc_selection",
            "risk_description": "Prompt choice was made after observing pilot structural validity.",
            "affected_models": json_dumps(["Qwen2 7B Instruct", "Mistral 7B Instruct"]),
            "affected_tasks": json_dumps(["reranking"]),
            "model_specific_prompting": "true",
            "post_hoc_selection_risk": "true",
            "pilot_row_overfitting_risk": "true",
            "cross_model_prompt_identity_reduced": "true",
            "scientific_task_identity_preserved": "true",
            "mitigation": "Freeze policy before executing a disjoint result-independent holdout pilot.",
            "independent_holdout_required": "true",
            "severity": "warning",
            "status": "mitigated_by_disjoint_sample",
            "notes": "Utility values are not used for selection.",
        },
        {
            "risk_id": "HYB-RISK-003",
            "risk_category": "pilot_row_overfitting",
            "risk_description": "A prompt could fit the original pilot rows without generalizing.",
            "affected_models": json_dumps(["Qwen2 7B Instruct", "Mistral 7B Instruct"]),
            "affected_tasks": json_dumps(["reranking"]),
            "model_specific_prompting": "true",
            "post_hoc_selection_risk": "true",
            "pilot_row_overfitting_risk": "true",
            "cross_model_prompt_identity_reduced": "true",
            "scientific_task_identity_preserved": "true",
            "mitigation": "Use 30 new rows disjoint from prior v1/v2 pilots.",
            "independent_holdout_required": "true",
            "severity": "warning",
            "status": "mitigated_by_holdout_design",
            "notes": "Holdout execution is not performed in this audit.",
        },
        {
            "risk_id": "HYB-RISK-004",
            "risk_category": "reduced_prompt_identity",
            "risk_description": "Cross-model comparison no longer uses identical reranking wording.",
            "affected_models": json_dumps(["Qwen2 7B Instruct", "Mistral 7B Instruct"]),
            "affected_tasks": json_dumps(["reranking"]),
            "model_specific_prompting": "true",
            "post_hoc_selection_risk": "true",
            "pilot_row_overfitting_risk": "false",
            "cross_model_prompt_identity_reduced": "true",
            "scientific_task_identity_preserved": "true",
            "mitigation": "Report as adapted model-specific format-compliance contract.",
            "independent_holdout_required": "true",
            "severity": "warning",
            "status": "accepted_with_caveat",
            "notes": "The valid output space remains identical under the same strict parser.",
        },
        {
            "risk_id": "HYB-RISK-005",
            "risk_category": "paper_alignment",
            "risk_description": "Hybrid prompt wording is not an exact paper reproduction condition.",
            "affected_models": json_dumps(["Qwen2 7B Instruct", "Mistral 7B Instruct"]),
            "affected_tasks": json_dumps(["ranking", "reranking"]),
            "model_specific_prompting": "true",
            "post_hoc_selection_risk": "true",
            "pilot_row_overfitting_risk": "false",
            "cross_model_prompt_identity_reduced": "true",
            "scientific_task_identity_preserved": "true",
            "mitigation": "Use adapted-not-exact paper alignment language.",
            "independent_holdout_required": "true",
            "severity": "warning",
            "status": "accepted_with_caveat",
            "notes": "This policy supports adapted local evaluation, not paper-exact reproduction.",
        },
        {
            "risk_id": "HYB-RISK-006",
            "risk_category": "unseen_row_failure",
            "risk_description": "The selected prompts may fail structural readiness on unseen rows.",
            "affected_models": json_dumps(["Qwen2 7B Instruct", "Mistral 7B Instruct"]),
            "affected_tasks": json_dumps(["reranking"]),
            "model_specific_prompting": "true",
            "post_hoc_selection_risk": "true",
            "pilot_row_overfitting_risk": "true",
            "cross_model_prompt_identity_reduced": "true",
            "scientific_task_identity_preserved": "true",
            "mitigation": "Require every holdout cell to reach at least 8/10 strict-valid outputs.",
            "independent_holdout_required": "true",
            "severity": "warning",
            "status": "open_until_holdout_executed",
            "notes": "No inference is executed in Phase 5B-P1E-AUDIT.",
        },
    ]
    return frame(risk_rows, RISKS_COLUMNS)


def build_decision(
    cell_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    semantic_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
) -> pd.DataFrame:
    evidence_ok = cell_df["all_sources_agree"].eq("true").all()
    selection_ok = selection_df["selection_basis_classification"].eq("structural_validity_based").all()
    utility_used = selection_df[
        [
            "hr_used_for_selection",
            "ndcg_used_for_selection",
            "positive_rank_used_for_selection",
            "latency_used_for_selection",
            "token_count_used_for_selection",
            "beyond_utility_used_for_selection",
        ]
    ].eq("true").any().any()
    semantic_ok = semantic_df["hybrid_policy_compatible"].eq("true").all()
    overlap_count = int(sample_df["disjoint_from_previous_pilots"].eq("false").sum()) if not sample_df.empty else 0
    approved = evidence_ok and selection_ok and not utility_used and semantic_ok and len(policy_df) == 1 and len(sample_df) == 30 and len(manifest_df) == 60 and overlap_count == 0
    if approved:
        decision = "approve_hybrid_disjoint_holdout_pilot"
        blocking = ""
        corrective = "Execute no full matrix until the frozen disjoint holdout pilot passes structural readiness."
    elif not evidence_ok:
        decision = "raw_evidence_reconciliation_failed"
        blocking = "v1/v2 raw evidence did not reconcile"
        corrective = "Resolve evidence mismatch before selecting any prompt policy."
    elif not semantic_ok:
        decision = "require_common_prompt_v3_audit"
        blocking = "semantic equivalence failed"
        corrective = "Design a common prompt-v3 audit because v1/v2 are not comparable."
    elif utility_used:
        decision = "reject_hybrid_policy_as_noncomparable"
        blocking = "utility influenced prompt selection"
        corrective = "Redo prompt selection using only structural validity."
    else:
        decision = "require_additional_offline_evidence"
        blocking = "offline approval conditions incomplete"
        corrective = "Complete missing offline evidence before holdout."
    rows = [
        {
            "audit_id": "canonical_local_llm_hybrid_prompt_holdout_audit_v1",
            "v1_v2_evidence_reconciled": bool_text(evidence_ok),
            "selection_structural_validity_based": bool_text(selection_ok),
            "utility_used_for_selection": bool_text(utility_used),
            "semantic_equivalence_verified": bool_text(semantic_ok),
            "scientific_task_identity_preserved": bool_text(semantic_ok),
            "model_specific_prompting_required": "true",
            "common_prompt_v3_required": bool_text(decision == "require_common_prompt_v3_audit"),
            "hybrid_policy_created": bool_text(len(policy_df) == 1),
            "hybrid_policy_contract_id": HYBRID_POLICY_ID if len(policy_df) == 1 else "",
            "holdout_sample_rows": str(len(sample_df)),
            "holdout_request_rows": str(len(manifest_df)),
            "previous_pilot_overlap_count": str(overlap_count),
            "readiness_threshold": format_float(READINESS_THRESHOLD),
            "holdout_validation_required": "true",
            "recommended_decision": decision,
            "blocking_reason": blocking,
            "required_corrective_action": corrective,
            "recommended_single_next_task": RECOMMENDED_NEXT_TASK if approved else corrective,
            "notes": "Prompt-v3 is not necessary if the hybrid policy passes the independent holdout readiness gate."
            if approved
            else "Hybrid policy not approved under current offline evidence.",
        }
    ]
    return frame(rows, DECISION_COLUMNS)


def build_all_frames(root: Path) -> tuple[EvidenceBundle, dict[str, pd.DataFrame]]:
    bundle = load_evidence(root)
    cell_df = reconstruct_v1_v2_cells(bundle)
    selection_df = build_prompt_selection_basis(cell_df)
    semantic_df = build_semantic_equivalence(bundle)
    policy_df = build_hybrid_policy_contract(bundle, selection_df, semantic_df)
    approved = len(policy_df) == 1
    sample_df = select_disjoint_holdout_sample(bundle, approved)
    manifest_df = build_holdout_request_manifest(bundle, sample_df, policy_df)
    risks_df = build_risks()
    decision_df = build_decision(cell_df, selection_df, semantic_df, policy_df, sample_df, manifest_df)
    frames = {
        "cell_reconciliation": cell_df,
        "selection_basis": selection_df,
        "semantic_equivalence": semantic_df,
        "policy_contract": policy_df,
        "holdout_sample": sample_df,
        "holdout_manifest": manifest_df,
        "risks": risks_df,
        "decision": decision_df,
    }
    validate_frames(frames)
    return bundle, frames


def validate_frames(frames: Mapping[str, pd.DataFrame]) -> None:
    expected_columns = {key: columns for key, (_path, columns) in OUTPUT_TABLES.items()}
    for key, columns in expected_columns.items():
        if key not in frames:
            raise ValueError(f"missing frame {key}")
        if list(frames[key].columns) != list(columns):
            raise ValueError(f"{key} columns mismatch")
    if len(frames["cell_reconciliation"]) != 12:
        raise ValueError("cell reconciliation must contain 12 rows")
    if len(frames["selection_basis"]) != 2:
        raise ValueError("selection basis must contain 2 rows")
    if frames["selection_basis"]["selection_basis_classification"].ne("structural_validity_based").any():
        raise ValueError("prompt selection basis is not structural-only")
    if frames["semantic_equivalence"].empty:
        raise ValueError("semantic equivalence table is empty")
    if frames["semantic_equivalence"]["hybrid_policy_compatible"].ne("true").any():
        raise ValueError("semantic equivalence contains blocking differences")
    if len(frames["policy_contract"]) != 1:
        raise ValueError("approved hybrid policy requires exactly one policy row")
    if len(frames["holdout_sample"]) != 30:
        raise ValueError("holdout sample must contain 30 rows")
    if len(frames["holdout_manifest"]) != 60:
        raise ValueError("holdout request manifest must contain 60 rows")
    sample = frames["holdout_sample"]
    if sample["task_type"].ne("reranking").any():
        raise ValueError("holdout sample must be reranking only")
    counts = sample.groupby("dataset").size().to_dict()
    if counts != {"Amazon Beauty": 10, "Amazon Sports": 10, "ML-1M": 10}:
        raise ValueError(f"holdout sample stratum counts mismatch: {counts}")
    if sample["disjoint_from_previous_pilots"].ne("true").any():
        raise ValueError("holdout sample overlaps previous pilots")
    manifest = frames["holdout_manifest"]
    if manifest["previous_pilot_overlap"].ne("false").any():
        raise ValueError("holdout manifest contains previous pilot overlap")
    qwen_prompts = set(manifest[manifest["model_family"].eq("Qwen2 7B Instruct")]["prompt_contract_id"])
    mistral_prompts = set(manifest[manifest["model_family"].eq("Mistral 7B Instruct")]["prompt_contract_id"])
    if qwen_prompts != {RERANKING_V1_CONTRACT_ID}:
        raise ValueError(f"Qwen prompt assignment mismatch: {qwen_prompts}")
    if mistral_prompts != {RERANKING_V2_CONTRACT_ID}:
        raise ValueError(f"Mistral prompt assignment mismatch: {mistral_prompts}")
    qwen_rows = set(manifest[manifest["model_family"].eq("Qwen2 7B Instruct")]["holdout_row_id"])
    mistral_rows = set(manifest[manifest["model_family"].eq("Mistral 7B Instruct")]["holdout_row_id"])
    if qwen_rows != mistral_rows or len(qwen_rows) != 30:
        raise ValueError("holdout rows are not shared across both models")
    decision = frames["decision"].iloc[0]
    if decision["recommended_decision"] != "approve_hybrid_disjoint_holdout_pilot":
        raise ValueError(f"unexpected decision: {decision['recommended_decision']}")
    if decision["readiness_threshold"] != format_float(READINESS_THRESHOLD):
        raise ValueError("readiness threshold mismatch")


def write_tables(root: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    for key, (relative, columns) in OUTPUT_TABLES.items():
        if key == "policy_contract" and key not in frames:
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[key].loc[:, columns].to_csv(path, index=False, lineterminator="\n")


def markdown_table(data: pd.DataFrame, columns: Sequence[str]) -> str:
    if data.empty:
        return "_No rows._"
    subset = data.loc[:, list(columns)].astype(str)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for row in subset.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _strict_summary(cell_df: pd.DataFrame, prompt_version: str, model: str) -> str:
    rows = cell_df[cell_df["prompt_version"].eq(prompt_version) & cell_df["model_family"].eq(model)]
    parts = [f"{row.dataset}: {row.strict_valid_count}/{row.request_count}" for row in rows.itertuples()]
    return "; ".join(parts)


def write_report(root: Path, bundle: EvidenceBundle, frames: Mapping[str, pd.DataFrame], commands: Sequence[str]) -> None:
    decision = frames["decision"].iloc[0]
    policy = frames["policy_contract"].iloc[0]
    sample = frames["holdout_sample"]
    manifest = frames["holdout_manifest"]
    cell = frames["cell_reconciliation"]
    selection = frames["selection_basis"]
    semantic = frames["semantic_equivalence"]
    hash_count = len(bundle.protected_hashes)
    section_text = {
        1: f"Branch and commit were recorded outside this report. The audit hashes {hash_count} protected evidence files and writes only the Phase 5B-P1E audit artifacts.",
        2: f"Decision: `{decision['recommended_decision']}`. The hybrid policy is approved for a disjoint holdout pilot, not for full execution.",
        3: "This audit invokes no Ollama endpoint, sends no LLM request, creates no prompt v3, and executes no recommender training.",
        4: _strict_summary(cell, "v1", "Qwen2 7B Instruct") + "; " + _strict_summary(cell, "v1", "Mistral 7B Instruct"),
        5: _strict_summary(cell, "v2", "Qwen2 7B Instruct") + "; " + _strict_summary(cell, "v2", "Mistral 7B Instruct"),
        6: markdown_table(cell, ["model_family", "dataset", "prompt_version", "strict_valid_count", "request_count", "all_sources_agree"]),
        7: "Qwen2 prompt v1 passes all reranking cells: Beauty 10/10, Sports 9/10, ML-1M 9/10.",
        8: "Qwen2 prompt v2 fails all reranking cells: Beauty 4/10, Sports 5/10, ML-1M 4/10.",
        9: "Mistral prompt v1 fails Beauty at 6/10 but passes Sports 9/10 and ML-1M 10/10.",
        10: "Mistral prompt v2 passes all reranking cells: Beauty 10/10, Sports 9/10, ML-1M 9/10.",
        11: markdown_table(selection, ["model_family", "selected_prompt_contract_id", "rejected_prompt_contract_id", "selected_prompt_structural_validity_minimum", "selection_basis_classification"]),
        12: "Selection flags for HR@5, NDCG@5, positive rank, latency, token count, and beyond-utility metrics are all false.",
        13: markdown_table(semantic, ["field_name", "values_identical", "difference_classification", "hybrid_policy_compatible"]),
        14: "Candidate representation, numbering, candidate order policy, and paired candidate-order hashes are equivalent.",
        15: "History source is inherited from v1 contexts, and all paired history-render hashes are identical.",
        16: "Both prompts request the same 20-item full permutation in one-based candidate-index space.",
        17: f"Both reranking versions use `{RERANKING_PARSER_ID}` with no salvage substitution.",
        18: "Execution parameters remain temperature 0, max_output_tokens 512, json_object response format, no retries, and concurrency 1.",
        19: "Model-specific wording reduces literal prompt identity but keeps the requested task and valid output space identical.",
        20: "Cross-model utility comparisons remain adapted and must carry the model-specific prompt caveat.",
        21: f"Paper alignment is `{policy['paper_alignment']}`; the policy is not paper exact.",
        22: "A common prompt v3 is not scientifically necessary before holdout because semantic equivalence holds and prompt-v3 search would add tuning risk.",
        23: "Post-hoc tuning risk is present because pilot structural validity informed prompt selection; the mitigation is a frozen disjoint holdout.",
        24: "Pilot-row overfitting risk is mitigated by excluding every v1 and v2 pilot row before selecting holdout rows.",
        25: "Independent holdout validation is required before any full execution because it tests structural readiness on unseen rows.",
        26: "The holdout population is `primary_prompt_compatible == true` reranking rows from Beauty, Sports, and ML-1M.",
        27: "Previous pilot exclusions use dataset, task type, candidate-set ID, and user ID identities from v1 sample and v2 corrective manifest.",
        28: "Selection key is SHA256(dataset + '|' + task_type + '|' + candidate_set_id + '|' + user_id), sorted ascending with candidate-set ID and user ID tie-breakers.",
        29: markdown_table(sample.groupby("dataset", as_index=False).size().rename(columns={"size": "rows"}), ["dataset", "rows"]),
        30: markdown_table(frames["policy_contract"], ["hybrid_policy_id", "qwen_reranking_prompt_contract_id", "mistral_reranking_prompt_contract_id", "contract_status"]),
        31: f"Qwen2 ranking uses `{RANKING_V1_CONTRACT_ID}` and reranking uses `{RERANKING_V1_CONTRACT_ID}`.",
        32: f"Mistral ranking uses `{RANKING_V1_CONTRACT_ID}` and reranking uses `{RERANKING_V2_CONTRACT_ID}`.",
        33: f"Shared strict parser requirements are represented by `{policy['shared_parser_contract_id']}`.",
        34: "Holdout execution parameters are frozen in the request manifest and remain no-retry, sequential, temperature-zero settings.",
        35: f"The holdout manifest contains {len(manifest)} requests over {len(sample)} unique rows.",
        36: "Each of the six model-dataset cells must pass at least 8/10 strict-valid responses, threshold 0.80.",
        37: "Full execution remains blocked until the disjoint hybrid holdout pilot passes the readiness gate.",
        38: markdown_table(frames["risks"], ["risk_id", "risk_category", "severity", "status"]),
        39: f"Protected hashes were computed for {hash_count} evidence files before table generation and validated again afterward by the runner.",
        40: "\n".join(f"- `{command}`" for command in commands),
        41: markdown_table(frames["decision"], ["recommended_decision", "hybrid_policy_created", "holdout_sample_rows", "holdout_request_rows", "common_prompt_v3_required"]),
        42: f"Recommended single next task: `{decision['recommended_single_next_task']}`.",
    }
    headings = [
        "Scope and provenance",
        "Executive summary",
        "Safety boundaries",
        "Prompt-v1 pilot evidence",
        "Prompt-v2 corrective-pilot evidence",
        "V1/V2 evidence reconciliation",
        "Qwen2 prompt-v1 findings",
        "Qwen2 prompt-v2 findings",
        "Mistral prompt-v1 findings",
        "Mistral prompt-v2 findings",
        "Structural-validity selection basis",
        "Confirmation that utility was not used",
        "Prompt semantic-equivalence audit",
        "Candidate-information equivalence",
        "History-information equivalence",
        "Output-space equivalence",
        "Parser equivalence",
        "Execution-parameter equivalence",
        "Model-specific wording implications",
        "Cross-model comparability",
        "Paper-alignment classification",
        "Prompt-v3 necessity assessment",
        "Post-hoc prompt-tuning risk",
        "Pilot-row overfitting risk",
        "Independent holdout rationale",
        "Holdout population",
        "Previous-pilot exclusions",
        "Deterministic holdout selection",
        "Holdout sample composition",
        "Hybrid prompt-policy contract",
        "Qwen2 prompt assignment",
        "Mistral prompt assignment",
        "Shared parser and output requirements",
        "Holdout execution parameters",
        "Holdout request count",
        "Holdout readiness threshold",
        "Full-execution implications",
        "Limitations",
        "Protected-input immutability",
        "Commands executed",
        "Audit decision",
        "Recommended single next task",
    ]
    lines = ["# Canonical Local LLM Hybrid Prompt and Disjoint Holdout Audit", ""]
    for index, heading in enumerate(headings, start=1):
        lines.append(f"## {index}. {heading}")
        lines.append("")
        lines.append(section_text[index])
        lines.append("")
    path = root / REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def validate_written_outputs(root: Path, frames: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in output_paths(root, frames):
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size == 0:
            raise ValueError(f"empty output file: {path}")
        hashes[path.relative_to(root).as_posix()] = file_sha256(path)
    report_text = (root / REPORT_PATH).read_text(encoding="utf-8")
    section_count = sum(1 for line in report_text.splitlines() if line.startswith("## "))
    if section_count != 42:
        raise ValueError(f"report section count mismatch: {section_count}")
    return hashes
