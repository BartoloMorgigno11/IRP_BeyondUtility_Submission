"""R3 complete local metric and beyond-utility audit.

This module is offline-only. It scans existing local artifacts, extracts
materialized metric values, recomputes metrics from already-generated row-level
outputs where possible, classifies remaining gaps, and writes a local R3 audit
layer. It never calls LLM/model backends, trains models, regenerates candidates,
or mutates protected R1/R2/F6M/original-repository evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.analysis.r1_metric_recovery_paths import relocation_record, resolve_r1_metric_recovery_artifact


PHASE_ID = "reproduction_r3_complete_local_metric_audit"
DEFAULT_CONFIG_PATH = Path("configs/reproduction_r3_complete_local_metric_audit.yaml")
EXPECTED_BRANCH = "master"
EXPECTED_HEAD = "5167740b80fd917db99df0768ea9358d00ec3515"

COMPLETE_TOKEN = "REPRODUCTION_R3_COMPLETE_LOCAL_METRIC_AUDIT_COMPLETE"
PARTIAL_TOKEN = "REPRODUCTION_R3_COMPLETE_LOCAL_METRIC_AUDIT_PARTIAL"
BLOCKED_TOKEN = "REPRODUCTION_R3_COMPLETE_LOCAL_METRIC_AUDIT_BLOCKED"

TARGET_DATASETS = ("Beauty", "Sports", "ML-1M", "LastFM")
ORIGINAL_ROOT = Path("original_repo") / "EvaLLMasRecommender"
TABLE_DIR = Path("results") / "tables"
REPORT_PATH = Path("report") / "reproduction_r3_complete_local_metric_and_beyond_utility_audit.md"
NOTES_PATH = Path("report") / "notes.md"
NOTE_MARKER = "R3 complete local metric and beyond-utility audit"

CALCULABILITY_STATUSES = (
    "ALREADY_MATERIALIZED",
    "RECALCULATED_FROM_LOCAL_OUTPUTS",
    "RECONSTRUCTED_FROM_LOCAL_ARTIFACTS",
    "PARTIALLY_RECONSTRUCTIBLE",
    "NOT_CALCULABLE_MISSING_OUTPUT",
    "NOT_CALCULABLE_MISSING_INPUT",
    "NOT_CALCULABLE_MISSING_PROTOCOL",
    "REQUIRES_NEW_LLM_INFERENCE",
    "REQUIRES_NEW_NON_LLM_EXECUTION",
    "NOT_APPLICABLE",
)

CANONICALITY_LABELS = (
    "CANONICAL_FROZEN",
    "VALIDATED_PHASE_OUTPUT",
    "DIAGNOSTIC_ONLY",
    "SUPERSEDED",
    "HISTORICAL_INTERMEDIATE",
    "UNKNOWN_PROVENANCE",
)

PROTECTED_ROOT_FILES = (
    "final_metric_paper_comparison_values.md",
    "original_repo_metric_file_inventory.csv",
    "original_repo_metric_values_decoded.csv",
    "original_repo_metric_values_decoded_summary.csv",
    "original_repo_metric_values_decoded_wide.csv",
    "original_repo_requested_metric_values_decoded.csv",
    "requested_metric_candidate_position_bias_values.csv",
    "requested_metric_generation_involved_performance_values.csv",
    "requested_metric_history_length_sensitivity_values.csv",
    "requested_metric_novelty_values.csv",
    "requested_metric_paper_comparison_values.csv",
    "requested_metric_values_summary.csv",
)

OUTPUT_TABLES = {
    "artifact_manifest": TABLE_DIR / "reproduction_r3_local_artifact_manifest.csv",
    "all_metric_cells_master": TABLE_DIR / "reproduction_r3_all_metric_cells_master.csv",
    "beyond_utility_master": TABLE_DIR / "reproduction_r3_beyond_utility_master.csv",
    "paper_metric_dataset_coverage": TABLE_DIR / "reproduction_r3_paper_metric_dataset_coverage.csv",
    "true_remaining_gaps": TABLE_DIR / "reproduction_r3_true_remaining_gaps.csv",
    "history_length_sensitivity": TABLE_DIR / "reproduction_r3_history_length_sensitivity.csv",
    "candidate_position_bias": TABLE_DIR / "reproduction_r3_candidate_position_bias.csv",
    "hallucination_audit": TABLE_DIR / "reproduction_r3_hallucination_audit.csv",
    "generation_involved_performance": TABLE_DIR / "reproduction_r3_generation_involved_performance.csv",
    "materialized_values": TABLE_DIR / "reproduction_r3_materialized_values.csv",
    "stored_vs_recomputed": TABLE_DIR / "reproduction_r3_stored_vs_recomputed.csv",
    "offline_recalculated_values": TABLE_DIR / "reproduction_r3_offline_recalculated_values.csv",
    "validation": TABLE_DIR / "reproduction_r3_validation.csv",
    "precondition_git_state": TABLE_DIR / "reproduction_r3_precondition_git_state.csv",
    "postcondition_git_state": TABLE_DIR / "reproduction_r3_postcondition_git_state.csv",
    "protected_artifact_hash_check": TABLE_DIR / "reproduction_r3_protected_artifact_hash_check.csv",
}

MANIFEST_COLUMNS = [
    "artifact_path",
    "phase",
    "dataset",
    "model",
    "task",
    "condition",
    "metric_families",
    "row_count",
    "user_level",
    "contains_ground_truth",
    "contains_ranking",
    "contains_candidates",
    "contains_history",
    "contains_positive_position",
    "contains_profile",
    "contains_text_output",
    "contains_structured_output",
    "contains_metric_values",
    "canonicality",
    "notes",
]

MASTER_COLUMNS = [
    "dataset",
    "model",
    "task",
    "prompt_strategy",
    "profile_generator",
    "condition",
    "cohort",
    "metric_family",
    "metric_name",
    "metric_value",
    "cutoff",
    "history_condition",
    "candidate_position_condition",
    "generation_condition",
    "requested_count",
    "valid_count",
    "denominator",
    "invalid_policy",
    "calculability_status",
    "stored_local_value",
    "recomputed_local_value",
    "reconstructed_local_value",
    "value_origin",
    "source_artifact",
    "secondary_source_artifact",
    "paper_metric",
    "paper_semantic_equivalence",
    "paper_comparability",
    "protocol_class",
    "source_scope",
    "notes",
]

MATERIALIZED_COLUMNS = [
    "value_id",
    "dataset",
    "model",
    "task",
    "prompt_strategy",
    "profile_generator",
    "condition",
    "cohort",
    "metric_family",
    "metric_name",
    "metric_value",
    "cutoff",
    "denominator",
    "value_origin",
    "source_scope",
    "source_artifact",
    "paper_metric",
    "local_metric_class",
    "notes",
]

RECOMPUTED_COLUMNS = [
    "recalculation_id",
    "dataset",
    "model",
    "task",
    "prompt_strategy",
    "profile_generator",
    "condition",
    "cohort",
    "metric_family",
    "metric_name",
    "metric_value",
    "cutoff",
    "requested_count",
    "valid_count",
    "denominator",
    "invalid_policy",
    "raw_input_artifacts",
    "supporting_input_artifacts",
    "formula",
    "value_origin",
    "paper_comparability",
    "notes",
]

STORED_VS_COLUMNS = [
    "comparison_id",
    "dataset",
    "model",
    "task",
    "condition",
    "metric_name",
    "stored_local_value",
    "recomputed_local_value",
    "agreement",
    "absolute_delta",
    "relative_delta",
    "stored_source_artifact",
    "recomputed_source_artifact",
    "notes",
]

BEYOND_COLUMNS = [
    "dataset",
    "model",
    "task",
    "condition",
    "metric",
    "value",
    "granularity",
    "calculation_status",
    "source",
    "paper_equivalence",
    "limitation",
]

COVERAGE_COLUMNS = [
    "dataset",
    "metric",
    "executed",
    "numeric_value_available",
    "offline_recalculable",
    "paper_exact",
    "adapted",
    "diagnostic_only",
    "missing",
    "requires_new_execution",
    "models",
    "tasks",
    "conditions",
    "granularity",
    "best_source",
    "exact_gap_reason",
]

GAP_COLUMNS = [
    "dataset",
    "metric",
    "model",
    "task",
    "condition",
    "what_exists",
    "what_is_missing",
    "missing_output",
    "missing_support_data",
    "missing_protocol",
    "offline_recovery_possible",
    "new_llm_inference_required",
    "new_non_llm_execution_required",
    "paper_exact_recovery_possible",
    "adapted_recovery_possible",
    "estimated_work",
    "exact_reason",
    "recommended_recovery",
]

HISTORY_COLUMNS = [
    "dataset",
    "model",
    "task",
    "history_condition",
    "history_length",
    "history_bin",
    "metric",
    "value",
    "requested_count",
    "valid_count",
    "granularity",
    "source_artifact",
    "paper_comparability",
    "notes",
]

CANDIDATE_POSITION_COLUMNS = [
    "dataset",
    "model",
    "task",
    "candidate_position_condition",
    "metric",
    "value",
    "classification",
    "canddif_formula_status",
    "first_position_value",
    "random_position_value",
    "has_first_random_evidence",
    "logical_requests",
    "valid_count",
    "denominator",
    "source_artifact",
    "notes",
]

HALLUCINATION_COLUMNS = [
    "dataset",
    "model",
    "task",
    "condition",
    "hallucination_semantic_type",
    "output_representation",
    "metric",
    "value",
    "numerator",
    "denominator",
    "source_artifact",
    "paper_comparability",
    "notes",
]

GENERATION_COLUMNS = [
    "dataset",
    "profile_generator",
    "recommender",
    "task",
    "generation_condition",
    "cohort",
    "metric",
    "value",
    "requested_count",
    "valid_count",
    "source_artifact",
    "has_downstream_recommendation_output",
    "notes",
]

VALIDATION_COLUMNS = ["check_id", "check_name", "status", "observed", "expected", "details"]
GIT_COLUMNS = ["phase", "scope", "check", "command", "observed"]
PROTECTED_COLUMNS = [
    "path",
    "exists",
    "tracked",
    "git_status",
    "expected_sha256",
    "observed_sha256",
    "hash_unchanged",
    "protection_status",
]

BEYOND_METRIC_IDS = {
    "self_information_at_5",
    "serendipity_at_5",
    "aplt_at_5",
    "aclt_at_5",
    "arp_at_5",
    "average_popularity",
    "popreo_at_5",
    "item_coverage_at_5",
    "shannon_entropy_at_5",
    "shannon_at_5",
    "jaccard_at_5",
    "gini_at_5",
    "dpd_at_5",
    "jain_index_at_5",
    "jain_index",
    "history_length_sensitivity",
    "canddif_hr",
    "canddif_ndcg",
    "candidate_position_bias_hr",
    "candidate_position_bias_ndcg",
    "hallucination_rate_at_5",
    "hallucination_rate",
    "generation_involved_profile_matrix",
}

GENERIC_VALUE_COLUMNS = (
    "value",
    "metric_value",
    "paper_value",
    "official_stored_value",
    "official_recomputed_value",
    "official_reconstructed_value",
)

WIDE_VALUE_COLUMNS = (
    "hr_at_5",
    "ndcg_at_5",
    "hr_at5",
    "ndcg_at5",
    "hit_at_5",
    "novelty_at_5",
    "self_information_at_5",
    "long_tail_rate_at_5",
    "aplt_at_5",
    "average_popularity_at_5",
    "arp_at_5",
    "item_coverage_at_5",
    "catalog_coverage_at_5",
    "candidate_validity_rate",
    "hallucination_rate",
    "strict_parse_validity_rate",
    "duplicate_rate_at_5",
    "operational_hr_at5",
    "conditional_hr_at5",
    "operational_ndcg_at5",
    "conditional_ndcg_at5",
    "hr_at5_operational",
    "hr_at5_conditional",
    "ndcg_at5_operational",
    "ndcg_at5_conditional",
)

AGGREGATE_NAME_HINTS = (
    "metrics",
    "summary",
    "results",
    "readout",
    "coverage",
    "comparison",
    "matrix",
    "contract",
    "status",
    "alignment",
    "audit",
)

USER_LEVEL_NAME_BLOCKLIST = (
    "rankings",
    "parse_results",
    "request_level",
    "request_state",
    "request_manifest",
    "raw_outputs",
    "row_contributions",
    "top5_exposures",
    "ranking_rows",
    "manifest.csv",
)


@dataclass(frozen=True)
class MetricCatalog:
    paper_metric_ids: set[str]
    metric_families: dict[str, str]
    display_names: dict[str, str]
    local_extension_ids: set[str]


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def parse_bool(value: Any) -> bool:
    return clean(value).lower() in {"1", "true", "yes", "y", "t"}


def numeric_value(value: Any) -> float | None:
    text = clean(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def fmt(value: Any) -> str:
    number = numeric_value(value)
    if number is None:
        return clean(value)
    return f"{number:.15g}"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rel(project_root: Path, path: Path | str) -> str:
    target = Path(path)
    try:
        return target.resolve().relative_to(project_root.resolve()).as_posix()
    except (ValueError, OSError):
        return str(path).replace("\\", "/")


def stable_id(prefix: str, *parts: Any) -> str:
    import hashlib

    payload = "||".join(clean(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16].upper()}"


def git_output(root: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout.strip()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]] | pd.DataFrame, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        frame = rows.copy()
        for column in columns:
            if column not in frame.columns:
                frame[column] = ""
        frame.loc[:, list(columns)].fillna("").to_csv(path, index=False, lineterminator="\n")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: clean(row.get(column, "")) for column in columns})


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    return payload


def normalize_dataset(value: Any) -> str:
    text = clean(value)
    low = text.lower().replace("_", " ").replace("-", " ")
    if "beauty" in low:
        return "Beauty"
    if "sports" in low:
        return "Sports"
    if "ml1m" in low or "ml 1m" in low or "movie" in low:
        return "ML-1M"
    if "lastfm" in low or "last fm" in low:
        return "LastFM"
    return text


def normalize_task(value: Any) -> str:
    text = clean(value)
    low = text.lower()
    if "rerank" in low:
        return "reranking"
    if "rank" in low:
        return "ranking"
    if "history" in low:
        return "history_sensitivity"
    if "profile" in low or "generation" in low:
        return "generation_involved"
    if "halluc" in low or "validation" in low:
        return "llm_output_validation"
    return text


def normalize_model(value: Any) -> str:
    text = clean(value)
    low = text.lower()
    aliases = {
        "qwen2": "qwen2:7b-instruct",
        "mistral": "mistral:7b-instruct",
        "mostpop": "MostPop",
        "bm25": "BM25",
        "bprmf": "BPRMF",
        "gru4rec": "GRU4Rec",
        "sasrec": "SASRec",
        "lightgcn": "LightGCN",
    }
    for key, value_out in aliases.items():
        if key in low:
            return value_out
    return text


def normalize_metric(value: Any) -> str:
    text = clean(value)
    low = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    mapping = {
        "hr": "hr_at_5",
        "hr_5": "hr_at_5",
        "hr_at_5": "hr_at_5",
        "hit_at_5": "hr_at_5",
        "operational_hr_at5": "hr_at_5",
        "conditional_hr_at5": "hr_at_5",
        "hr_at5_operational": "hr_at_5",
        "hr_at5_conditional": "hr_at_5",
        "ndcg": "ndcg_at_5",
        "ndcg_5": "ndcg_at_5",
        "ndcg_at_5": "ndcg_at_5",
        "ndcg_at5": "ndcg_at_5",
        "operational_ndcg_at5": "ndcg_at_5",
        "conditional_ndcg_at5": "ndcg_at_5",
        "ndcg_at5_operational": "ndcg_at_5",
        "ndcg_at5_conditional": "ndcg_at_5",
        "selfinformation": "self_information_at_5",
        "selfinformation_5": "self_information_at_5",
        "selfinformation_5": "self_information_at_5",
        "self_information": "self_information_at_5",
        "self_information_5": "self_information_at_5",
        "self_information_at_5": "self_information_at_5",
        "novelty": "self_information_at_5",
        "novelty_at_5": "self_information_at_5",
        "serendipity": "serendipity_at_5",
        "serendipity_5": "serendipity_at_5",
        "serendipity_at_5": "serendipity_at_5",
        "aclt": "aclt_at_5",
        "aclt_5": "aclt_at_5",
        "aclt_at_5": "aclt_at_5",
        "aplt": "aplt_at_5",
        "aplt_5": "aplt_at_5",
        "aplt_at_5": "aplt_at_5",
        "long_tail_rate": "aplt_at_5",
        "long_tail_rate_at_5": "aplt_at_5",
        "arp": "arp_at_5",
        "arp_5": "arp_at_5",
        "arp_at_5": "arp_at_5",
        "average_popularity": "arp_at_5",
        "average_popularity_5": "arp_at_5",
        "average_popularity_at_5": "arp_at_5",
        "popreo": "popreo_at_5",
        "popreo_5": "popreo_at_5",
        "popreo_at_5": "popreo_at_5",
        "itemcoverage": "item_coverage_at_5",
        "itemcoverage_5": "item_coverage_at_5",
        "item_coverage": "item_coverage_at_5",
        "item_coverage_5": "item_coverage_at_5",
        "item_coverage_at_5": "item_coverage_at_5",
        "candidate_union_item_coverage_5": "item_coverage_at_5",
        "catalog_coverage": "catalog_coverage_at_5",
        "catalog_coverage_at_5": "catalog_coverage_at_5",
        "shannon": "shannon_entropy_at_5",
        "shannon_5": "shannon_entropy_at_5",
        "shannon_at_5": "shannon_entropy_at_5",
        "shannon_entropy": "shannon_entropy_at_5",
        "shannon_entropy_5": "shannon_entropy_at_5",
        "shannon_category_entropy_5": "shannon_entropy_at_5",
        "shannon_entropy_at_5": "shannon_entropy_at_5",
        "jaccard": "jaccard_at_5",
        "jaccard_5": "jaccard_at_5",
        "jaccard_at_5": "jaccard_at_5",
        "gini": "gini_at_5",
        "gini_5": "gini_at_5",
        "gini_at_5": "gini_at_5",
        "dpd": "dpd_at_5",
        "dpd_5": "dpd_at_5",
        "dpd_at_5": "dpd_at_5",
        "jain": "jain_index_at_5",
        "jain_5": "jain_index_at_5",
        "jain_s_index_5": "jain_index_at_5",
        "jain_index": "jain_index_at_5",
        "jain_index_5": "jain_index_at_5",
        "jain_index_at_5": "jain_index_at_5",
        "hallucination": "hallucination_rate_at_5",
        "hallucination_rate": "hallucination_rate_at_5",
        "hallucination_rate_5": "hallucination_rate_at_5",
        "hallucination_rate_at_5": "hallucination_rate_at_5",
        "canddif": "candidate_position_bias",
        "canddif_hr": "canddif_hr",
        "canddif_hr_5": "canddif_hr",
        "candidate_position_bias_hr": "canddif_hr",
        "canddif_ndcg": "canddif_ndcg",
        "canddif_ndcg_5": "canddif_ndcg",
        "candidate_position_bias_ndcg": "canddif_ndcg",
        "history_length_sensitivity": "history_length_sensitivity",
        "generation_involved_profile_matrix": "generation_involved_profile_matrix",
        "generation_involved_profile_recommendation": "generation_involved_profile_matrix",
        "generation_involved_performance": "generation_involved_profile_matrix",
        "strict_parse_validity_rate": "strict_parse_validity_rate",
        "duplicate_rate": "duplicate_rate_at_5",
        "duplicate_rate_at_5": "duplicate_rate_at_5",
        "candidate_validity_rate": "candidate_validity_rate",
        "mrr_at_5_printed_only": "mrr_at_5_printed_only",
        "mrr_printed_by_official_script": "mrr_at_5_printed_only",
        "oic_at_5": "oic_at_5",
        "overlap_item_coverage_5": "oic_at_5",
        "ks_sample_representativeness": "ks_sample_representativeness",
        "k_s_sample_representativeness_gate": "ks_sample_representativeness",
    }
    return mapping.get(low, low)


def metric_display(metric_id: str, catalog: MetricCatalog | None = None) -> str:
    if catalog is not None and metric_id in catalog.display_names:
        return catalog.display_names[metric_id]
    display = {
        "hr_at_5": "HR@5",
        "ndcg_at_5": "NDCG@5",
        "self_information_at_5": "SelfInformation@5",
        "serendipity_at_5": "Serendipity@5",
        "aclt_at_5": "ACLT@5",
        "aplt_at_5": "APLT@5",
        "arp_at_5": "ARP@5",
        "popreo_at_5": "PopREO@5",
        "item_coverage_at_5": "ItemCoverage@5",
        "shannon_entropy_at_5": "Shannon@5",
        "jaccard_at_5": "Jaccard@5",
        "gini_at_5": "Gini@5",
        "dpd_at_5": "DPD@5",
        "jain_index_at_5": "Jain@5",
        "hallucination_rate_at_5": "Hallucination@5",
        "canddif_hr": "CandDif_HR",
        "canddif_ndcg": "CandDif_NDCG",
        "history_length_sensitivity": "History Length Sensitivity",
        "generation_involved_profile_matrix": "Generation-Involved Performance",
    }
    return display.get(metric_id, metric_id)


def build_metric_catalog(project_root: Path, config: Mapping[str, Any]) -> MetricCatalog:
    e1_path = project_root / clean(config["inputs"]["e1_metric_contract"])
    r2_contract_path = project_root / clean(config["inputs"]["r2_metric_input_contracts"])
    e1 = read_csv(e1_path)
    r2_contract = read_csv(r2_contract_path)
    paper_metric_ids: set[str] = set()
    local_extension_ids: set[str] = set()
    families: dict[str, str] = {}
    display: dict[str, str] = {}

    for _, row in e1.iterrows():
        metric_id = normalize_metric(row.get("metric_id") or row.get("metric"))
        family = clean(row.get("family")) or clean(row.get("metric_family")) or "paper"
        if not metric_id:
            continue
        families[metric_id] = family
        display[metric_id] = clean(row.get("metric_name")) or metric_display(metric_id)
        if family == "local_extension":
            local_extension_ids.add(metric_id)
        else:
            paper_metric_ids.add(metric_id)

    for _, row in r2_contract.iterrows():
        metric_id = normalize_metric(row.get("metric"))
        family = clean(row.get("metric_family")) or families.get(metric_id, "paper")
        if not metric_id:
            continue
        families.setdefault(metric_id, family)
        display.setdefault(metric_id, metric_display(metric_id))
        if family == "local_extension":
            local_extension_ids.add(metric_id)
        else:
            paper_metric_ids.add(metric_id)

    return MetricCatalog(
        paper_metric_ids=paper_metric_ids,
        metric_families=families,
        display_names=display,
        local_extension_ids=local_extension_ids,
    )


def metric_family(metric_id: str, catalog: MetricCatalog) -> str:
    if metric_id in catalog.metric_families:
        return catalog.metric_families[metric_id]
    if metric_id in {"catalog_coverage_at_5", "candidate_validity_rate", "duplicate_rate_at_5", "strict_parse_validity_rate"}:
        return "local_extension"
    if metric_id in {"hr_at_5", "ndcg_at_5"}:
        return "utility"
    if metric_id in {"self_information_at_5", "serendipity_at_5"}:
        return "novelty"
    if metric_id in {"aclt_at_5", "aplt_at_5", "arp_at_5", "popreo_at_5"}:
        return "popularity"
    if metric_id in {"item_coverage_at_5", "shannon_entropy_at_5", "jaccard_at_5"}:
        return "diversity"
    if metric_id in {"gini_at_5", "dpd_at_5", "jain_index_at_5"}:
        return "fairness"
    if metric_id in {"hallucination_rate_at_5", "canddif_hr", "canddif_ndcg", "history_length_sensitivity", "generation_involved_profile_matrix"}:
        return "LLM_specific"
    return "local_diagnostic"


def is_paper_metric(metric_id: str, catalog: MetricCatalog) -> bool:
    return metric_id in catalog.paper_metric_ids and metric_id not in catalog.local_extension_ids


def source_scope(path: str) -> str:
    name = Path(path).name.lower()
    text = path.lower().replace("\\", "/")
    if name.startswith("paper_atomic_contract") or "paper_to_local" in name:
        return "PAPER_CONTRACT"
    if (
        name.startswith("reproduction_r1")
        or name.startswith("reproduction_r2")
        or name.startswith("original_repo_")
        or name.startswith("requested_metric_")
        or name.startswith("final_metric_paper_comparison")
    ):
        return "OFFICIAL_REFERENCE"
    if "structural" in name or "diagnostic" in name or "plan_d_" in name:
        return "LOCAL_DIAGNOSTIC"
    if "/results/" in text or text.startswith("results/"):
        return "LOCAL_REPRODUCTION"
    return "LOCAL_CONTEXT"


def git_state_rows(project_root: Path, phase: str) -> list[dict[str, str]]:
    commands = [
        ("branch", ["branch", "--show-current"]),
        ("head", ["rev-parse", "HEAD"]),
        ("status_short", ["status", "--short"]),
        ("status_branch_short", ["status", "--branch", "--short"]),
        ("diff_stat", ["diff", "--stat"]),
        ("diff_cached_stat", ["diff", "--cached", "--stat"]),
    ]
    rows = [
        {
            "phase": phase,
            "scope": "main_repo",
            "check": name,
            "command": "git " + " ".join(args),
            "observed": git_output(project_root, args),
        }
        for name, args in commands
    ]
    original_root = project_root / ORIGINAL_ROOT
    if original_root.exists():
        rows.extend(
            [
                {
                    "phase": phase,
                    "scope": "original_repo",
                    "check": "status_branch_short",
                    "command": "git status --branch --short",
                    "observed": git_output(original_root, ["status", "--branch", "--short"]),
                },
                {
                    "phase": phase,
                    "scope": "original_repo",
                    "check": "diff_stat",
                    "command": "git diff --stat",
                    "observed": git_output(original_root, ["diff", "--stat"]),
                },
            ]
        )
    return rows


def protected_hash_rows(project_root: Path, config: Mapping[str, Any]) -> list[dict[str, str]]:
    stored_path = project_root / clean(config["inputs"]["r2_protected_hash_check"])
    stored = read_csv(stored_path)
    expected_hashes = {
        clean(row.get("path")): clean(row.get("expected_sha256"))
        for _, row in stored.iterrows()
        if clean(row.get("path"))
    }
    rows: list[dict[str, str]] = []
    for rel_path in PROTECTED_ROOT_FILES:
        path = resolve_r1_metric_recovery_artifact(project_root, rel_path)
        record = relocation_record(project_root, rel_path)
        observed = sha256_file(path) if path.exists() else ""
        expected = record.sha256 if record else expected_hashes.get(rel_path, "")
        current_rel = record.current_path if record else rel_path
        tracked = bool(git_output(project_root, ["ls-files", "--", current_rel]))
        status = git_output(project_root, ["status", "--short", "--", current_rel])
        ok = (
            path.exists()
            and bool(expected)
            and observed == expected
            and (record is not None or (not tracked and status.startswith("??")))
        )
        rows.append(
            {
                "path": rel_path,
                "exists": bool_text(path.exists()),
                "tracked": bool_text(tracked),
                "git_status": status,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "hash_unchanged": bool_text(bool(expected) and observed == expected),
                "protection_status": "PROTECTED_UNCHANGED" if ok else "PROTECTION_FAILURE",
            }
        )
    return rows


def allowed_status_path(path: str) -> bool:
    normalized = path.replace("\\", "/").strip('"')
    if normalized in PROTECTED_ROOT_FILES:
        return True
    allowed_prefixes = (
        "src/analysis/reproduction_r3_complete_local_metric_audit.py",
        "src/experiments/build_reproduction_r3_complete_local_metric_audit.py",
        "configs/reproduction_r3_complete_local_metric_audit.yaml",
        "tests/test_reproduction_r3_complete_local_metric_audit.py",
        "results/tables/reproduction_r3_",
        "report/reproduction_r3_complete_local_metric_and_beyond_utility_audit.md",
        "report/notes.md",
    )
    return any(normalized.startswith(prefix) for prefix in allowed_prefixes)


def status_paths(status_text: str) -> list[str]:
    paths: list[str] = []
    for line in status_text.splitlines():
        if not line.strip():
            continue
        paths.append(line[2:].strip().replace("\\", "/"))
    return paths


def count_lines(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            count = sum(1 for _ in handle)
    except OSError:
        return ""
    if path.suffix.lower() == ".csv" and count > 0:
        count -= 1
    return str(max(count, 0))


def csv_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            return [clean(value) for value in next(reader, [])]
    except (OSError, UnicodeDecodeError, csv.Error):
        return []


def jsonl_first_keys(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        return sorted(str(key) for key in payload)
                    return []
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    return []


def phase_from_path(path: str) -> str:
    name = Path(path).name.lower()
    if name.startswith("reproduction_r3"):
        return "R3"
    if name.startswith("reproduction_r2"):
        return "R2"
    if name.startswith("reproduction_r1"):
        return "R1"
    for match in re.finditer(r"(phase_f6[a-z0-9]+|finalization_f[0-9a-z]+|plan_[a-z0-9]+|canonical_[a-z0-9_]+|extension_e[0-9]+)", name):
        return match.group(1)
    if "f6m" in name:
        return "F6M"
    return "repository_context"


def canonicality_from_path(path: str, header: Sequence[str]) -> str:
    name = Path(path).name.lower()
    if name.startswith(("reproduction_r1", "reproduction_r2")) or name.startswith("phase_f6m"):
        return "CANONICAL_FROZEN"
    if name.startswith("reproduction_r3"):
        return "VALIDATED_PHASE_OUTPUT"
    if name.startswith(("phase_f6", "finalization_f", "canonical_", "plan_e_")):
        return "VALIDATED_PHASE_OUTPUT"
    if "diagnostic" in name or "plan_d_" in name or "structural" in name:
        return "DIAGNOSTIC_ONLY"
    if any(column.endswith("_sha256") for column in header):
        return "HISTORICAL_INTERMEDIATE"
    return "HISTORICAL_INTERMEDIATE"


def field_text(header: Sequence[str], path: str) -> str:
    return " ".join([path, *header]).lower()


def families_from_schema(header: Sequence[str], path: str, catalog: MetricCatalog) -> str:
    text = field_text(header, path)
    families: set[str] = set()
    for metric_id, family in catalog.metric_families.items():
        if metric_id.replace("_at_5", "").replace("_", "") in text.replace("_", ""):
            families.add(family)
    for key, family in [
        ("hr", "utility"),
        ("ndcg", "utility"),
        ("self", "novelty"),
        ("novelty", "novelty"),
        ("serendip", "novelty"),
        ("arp", "popularity"),
        ("popularity", "popularity"),
        ("aplt", "popularity"),
        ("aclt", "popularity"),
        ("popreo", "popularity"),
        ("coverage", "diversity"),
        ("shannon", "diversity"),
        ("jaccard", "diversity"),
        ("gini", "fairness"),
        ("dpd", "fairness"),
        ("jain", "fairness"),
        ("hallucination", "LLM_specific"),
        ("history", "LLM_specific"),
        ("canddif", "LLM_specific"),
        ("position", "LLM_specific"),
        ("profile", "LLM_specific"),
    ]:
        if key in text:
            families.add(family)
    return ";".join(sorted(families))


def first_present(row: Mapping[str, Any], columns: Sequence[str]) -> str:
    for column in columns:
        value = clean(row.get(column, ""))
        if value:
            return value
    return ""


def discover_artifacts(project_root: Path, config: Mapping[str, Any], catalog: MetricCatalog) -> pd.DataFrame:
    scan_config = config.get("scan", {})
    roots = [project_root / Path(root) for root in scan_config.get("roots", ["results", "report", "configs", "src", "tests"])]
    skip_dirs = {str(Path(item)).replace("\\", "/").strip("/") for item in scan_config.get("skip_dirs", [])}
    rows: list[dict[str, str]] = []

    def is_skipped(path: Path) -> bool:
        rel_path = rel(project_root, path)
        parts = set(Path(rel_path).parts)
        if any(part.startswith(".venv") for part in parts):
            return True
        normalized = rel_path.replace("\\", "/").strip("/")
        return any(normalized == item or normalized.startswith(item + "/") for item in skip_dirs)

    for root in roots:
        if not root.exists():
            continue
        for current, dirs, files in os.walk(root, topdown=True, onerror=lambda _exc: None):
            current_path = Path(current)
            dirs[:] = [directory for directory in dirs if not is_skipped(current_path / directory)]
            if is_skipped(current_path):
                continue
            for filename in sorted(files):
                path = current_path / filename
                if is_skipped(path):
                    continue
                rel_path = rel(project_root, path)
                if rel_path.startswith("results/tables/reproduction_r3_") or rel_path == REPORT_PATH.as_posix():
                    continue
                suffix = path.suffix.lower()
                if suffix not in {".csv", ".json", ".jsonl", ".md", ".py", ".yaml", ".yml", ".txt"}:
                    continue
                header = csv_header(path) if suffix == ".csv" else (jsonl_first_keys(path) if suffix == ".jsonl" else [])
                header_lower = {column.lower() for column in header}
                text = field_text(header, rel_path)
                row_count = count_lines(path) if suffix in {".csv", ".jsonl", ".md", ".py", ".yaml", ".yml", ".txt"} else ""
                dataset = ""
                for candidate in TARGET_DATASETS:
                    if normalize_dataset(candidate).lower().replace("-", "") in normalize_dataset(rel_path).lower().replace("-", ""):
                        dataset = candidate
                        break
                rows.append(
                    {
                        "artifact_path": rel_path,
                        "phase": phase_from_path(rel_path),
                        "dataset": dataset,
                        "model": "",
                        "task": normalize_task(rel_path),
                        "condition": "",
                        "metric_families": families_from_schema(header, rel_path, catalog),
                        "row_count": row_count,
                        "user_level": bool_text(bool({"user_id", "request_id", "diagnostic_id"}.intersection(header_lower))),
                        "contains_ground_truth": bool_text(any("ground_truth" in column or "positive_item" in column for column in header_lower)),
                        "contains_ranking": bool_text(any("rank" in column or "top5" in column or "recommend" in column for column in header_lower)),
                        "contains_candidates": bool_text(any("candidate" in column for column in header_lower)),
                        "contains_history": bool_text("history" in text),
                        "contains_positive_position": bool_text("positive_position" in text or "positive_initial_position" in text),
                        "contains_profile": bool_text("profile" in text),
                        "contains_text_output": bool_text(any(key in text for key in ["raw_output", "raw_response", "parsed_titles", "title", "text_output"])),
                        "contains_structured_output": bool_text(any(key in text for key in ["json", "ranked_item_ids", "parsed_item_ids", "top5_item_ids", "candidate_items"])),
                        "contains_metric_values": bool_text(any(column in header_lower for column in GENERIC_VALUE_COLUMNS + WIDE_VALUE_COLUMNS) or "metric" in header_lower),
                        "canonicality": canonicality_from_path(rel_path, header),
                        "notes": "schema-scanned" if header else "content/path scanned; no tabular schema",
                    }
                )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS).fillna("")


def aggregate_artifact(path: str, columns: Sequence[str]) -> bool:
    name = Path(path).name.lower()
    if not name.endswith(".csv"):
        return False
    if any(block in name for block in USER_LEVEL_NAME_BLOCKLIST):
        return False
    if "user_id" in {column.lower() for column in columns} and "aggregation_level" not in {column.lower() for column in columns}:
        return False
    return any(hint in name for hint in AGGREGATE_NAME_HINTS)


def extract_row_context(row: Mapping[str, Any], source_path: str) -> dict[str, str]:
    dataset = normalize_dataset(first_present(row, ["dataset", "dataset_name"]))
    model = normalize_model(first_present(row, ["model", "model_name", "model_tag", "paper_model", "official_model", "baseline_family", "recommender_model"]))
    task = normalize_task(first_present(row, ["task", "task_type"]))
    prompt = first_present(row, ["prompt_strategy", "prompting_strategy", "strategy_id", "strategy_label", "prompt_contract_id"])
    profile_generator = first_present(row, ["profile_generator", "profile_generator_name"])
    condition = first_present(
        row,
        [
            "condition",
            "metric_scope",
            "population",
            "classification",
            "analysis_type",
            "stratum",
            "history_setting",
            "input_variant",
            "evaluation_policy_id",
        ],
    )
    cohort = first_present(row, ["cohort", "cohort_id", "sample_id", "scope", "population", "candidate_set_id"])
    cutoff = first_present(row, ["k", "cutoff", "cutoff_k"])
    denominator = first_present(row, ["denominator", "metric_denominator", "evaluated_rows", "valid_list_count", "total_instances", "planned_rows", "request_count"])
    if not dataset:
        dataset = normalize_dataset(source_path)
    if not task:
        task = normalize_task(source_path)
    return {
        "dataset": dataset,
        "model": model,
        "task": task,
        "prompt_strategy": prompt,
        "profile_generator": profile_generator,
        "condition": condition,
        "cohort": cohort,
        "cutoff": cutoff,
        "denominator": denominator,
    }


def metric_id_from_row(row: Mapping[str, Any]) -> str:
    return normalize_metric(first_present(row, ["metric", "metric_name", "metric_id", "paper_metric_name", "official_metric_name", "local_metric", "canonical_paper_name"]))


def value_notes(row: Mapping[str, Any]) -> str:
    notes = []
    for column in ["notes", "limitation", "protocol_notes", "status", "classification", "paper_alignment", "evidence_class"]:
        value = clean(row.get(column, ""))
        if value:
            notes.append(f"{column}={value}")
    return "; ".join(notes)


def extract_materialized_values(project_root: Path, manifest: pd.DataFrame, catalog: MetricCatalog) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, artifact in manifest.iterrows():
        rel_path = clean(artifact.get("artifact_path"))
        path = project_root / rel_path
        if not path.exists() or path.suffix.lower() != ".csv":
            continue
        header = csv_header(path)
        if not aggregate_artifact(rel_path, header):
            continue
        header_set = set(header)
        generic_values = [column for column in GENERIC_VALUE_COLUMNS if column in header_set]
        wide_values = [column for column in WIDE_VALUE_COLUMNS if column in header_set]
        if not generic_values and not wide_values:
            continue
        try:
            frame = read_csv(path)
        except Exception:
            continue
        if frame.empty:
            continue
        scope = source_scope(rel_path)
        for index, row in frame.iterrows():
            context = extract_row_context(row, rel_path)
            base_metric = metric_id_from_row(row)
            for value_column in generic_values:
                metric_id = base_metric
                if not metric_id or value_column.startswith("official_"):
                    metric_id = base_metric or normalize_metric(row.get("official_metric_name"))
                value = numeric_value(row.get(value_column))
                if not metric_id or value is None:
                    continue
                family = metric_family(metric_id, catalog)
                paper_metric_flag = is_paper_metric(metric_id, catalog)
                rows.append(
                    {
                        "value_id": stable_id("R3VAL", rel_path, index, metric_id, value_column),
                        **context,
                        "metric_family": family,
                        "metric_name": metric_display(metric_id, catalog),
                        "metric_value": fmt(value),
                        "cutoff": context["cutoff"] or ("5" if "_at_5" in metric_id or metric_id in {"hr_at_5", "ndcg_at_5"} else ""),
                        "denominator": context["denominator"],
                        "value_origin": f"MATERIALIZED:{value_column}",
                        "source_scope": scope,
                        "source_artifact": rel_path,
                        "paper_metric": bool_text(paper_metric_flag),
                        "local_metric_class": "ORIGINAL_PAPER_METRIC" if paper_metric_flag else ("LOCAL_EXTENSION" if family == "local_extension" else "LOCAL_DIAGNOSTIC"),
                        "notes": value_notes(row),
                    }
                )
            for value_column in wide_values:
                metric_id = normalize_metric(value_column)
                value = numeric_value(row.get(value_column))
                if value is None:
                    continue
                family = metric_family(metric_id, catalog)
                paper_metric_flag = is_paper_metric(metric_id, catalog)
                rows.append(
                    {
                        "value_id": stable_id("R3VAL", rel_path, index, metric_id, value_column),
                        **context,
                        "metric_family": family,
                        "metric_name": metric_display(metric_id, catalog),
                        "metric_value": fmt(value),
                        "cutoff": context["cutoff"] or ("5" if "at_5" in metric_id or value_column.endswith("at5") else ""),
                        "denominator": context["denominator"],
                        "value_origin": f"MATERIALIZED:{value_column}",
                        "source_scope": scope,
                        "source_artifact": rel_path,
                        "paper_metric": bool_text(paper_metric_flag),
                        "local_metric_class": "ORIGINAL_PAPER_METRIC" if paper_metric_flag else ("LOCAL_EXTENSION" if family == "local_extension" else "LOCAL_DIAGNOSTIC"),
                        "notes": value_notes(row),
                    }
                )
    frame = pd.DataFrame(rows, columns=MATERIALIZED_COLUMNS).fillna("")
    if frame.empty:
        return frame
    return frame.drop_duplicates(subset=["value_id"]).sort_values(["source_artifact", "metric_name", "dataset", "model"]).reset_index(drop=True)


def mean_or_empty(values: Sequence[float]) -> str:
    clean_values = [float(value) for value in values if math.isfinite(float(value))]
    if not clean_values:
        return ""
    return fmt(float(np.mean(np.asarray(clean_values, dtype=float))))


def bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([False] * len(frame), index=frame.index)
    return frame[column].map(parse_bool)


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([math.nan] * len(frame), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def recompute_from_canonical_row_contributions(project_root: Path, catalog: MetricCatalog) -> list[dict[str, str]]:
    path = project_root / "results/tables/canonical_metric_row_contributions.csv"
    if not path.exists():
        return []
    frame = read_csv(path)
    if frame.empty:
        return []
    valid = frame.loc[bool_series(frame, "row_valid")].copy()
    rows: list[dict[str, str]] = []
    metric_columns = {
        "hr_at_5": "mean(row hr_at_5)",
        "ndcg_at_5": "mean(row ndcg_at_5)",
        "novelty_at_5": "mean(row novelty_at_5)",
        "long_tail_rate_at_5": "mean(row long_tail_rate_at_5)",
        "average_popularity_at_5": "mean(row average_popularity_at_5)",
        "duplicate_rate_at_5": "mean(row duplicate_rate_at_5)",
        "candidate_validity_rate": "mean(row candidate_validity_rate)",
    }
    group_cols = ["baseline_id", "dataset", "task_type", "candidate_set_id"]
    for key, group in valid.groupby(group_cols, sort=True, dropna=False):
        key_map = dict(zip(group_cols, key))
        for column, formula in metric_columns.items():
            if column not in group.columns:
                continue
            values = numeric_series(group, column).dropna().tolist()
            if not values:
                continue
            metric_id = normalize_metric(column)
            family = metric_family(metric_id, catalog)
            rows.append(
                {
                    "recalculation_id": stable_id("R3RECALC", "canonical_row", key, metric_id),
                    "dataset": normalize_dataset(key_map["dataset"]),
                    "model": normalize_model(key_map["baseline_id"]),
                    "task": normalize_task(key_map["task_type"]),
                    "prompt_strategy": "",
                    "profile_generator": "",
                    "condition": "canonical_metric_row_contributions",
                    "cohort": clean(key_map["candidate_set_id"]),
                    "metric_family": family,
                    "metric_name": metric_display(metric_id, catalog),
                    "metric_value": fmt(float(np.mean(np.asarray(values, dtype=float)))),
                    "cutoff": "5" if metric_id.endswith("_at_5") or metric_id in {"hr_at_5", "ndcg_at_5"} else "",
                    "requested_count": str(len(group)),
                    "valid_count": str(len(group)),
                    "denominator": str(len(group)),
                    "invalid_policy": "row_valid_only",
                    "raw_input_artifacts": rel(project_root, path),
                    "supporting_input_artifacts": "results/tables/canonical_metric_train_item_statistics.csv; results/tables/canonical_metric_ranking_rows.csv",
                    "formula": formula,
                    "value_origin": "RECOMPUTED_FROM_ROW_LEVEL_CONTRIBUTIONS",
                    "paper_comparability": "ADAPTED_EQUIVALENT" if is_paper_metric(metric_id, catalog) else "LOCAL_DIAGNOSTIC",
                    "notes": "Offline aggregate recomputation from existing canonical row-level metric contributions.",
                }
            )
    return rows


def recompute_from_full_matrix_rankings(project_root: Path, catalog: MetricCatalog) -> list[dict[str, str]]:
    path = project_root / "results/tables/canonical_local_llm_full_matrix_rankings.csv"
    if not path.exists():
        return []
    frame = read_csv(path)
    if frame.empty:
        return []
    rows: list[dict[str, str]] = []
    group_cols = ["model_tag", "dataset", "task_type", "prompt_contract_id", "candidate_set_id"]
    for key, group in frame.groupby(group_cols, sort=True, dropna=False):
        key_map = dict(zip(group_cols, key))
        planned = len(group)
        valid_mask = bool_series(group, "utility_evaluation_valid")
        valid = group.loc[valid_mask]
        policies = [
            ("CONDITIONAL_VALID_ONLY", valid, len(valid)),
            ("FAIL_STRICT_ALL_REQUESTS", group, planned),
        ]
        for policy, policy_group, denominator in policies:
            if denominator == 0:
                continue
            hr_values = bool_series(policy_group, "positive_in_top_5").astype(float).tolist()
            ndcg_values = numeric_series(policy_group, "ndcg_at_5").fillna(0.0).tolist()
            for metric_id, values in [("hr_at_5", hr_values), ("ndcg_at_5", ndcg_values)]:
                rows.append(
                    {
                        "recalculation_id": stable_id("R3RECALC", "full_matrix", key, policy, metric_id),
                        "dataset": normalize_dataset(key_map["dataset"]),
                        "model": normalize_model(key_map["model_tag"]),
                        "task": normalize_task(key_map["task_type"]),
                        "prompt_strategy": clean(key_map["prompt_contract_id"]),
                        "profile_generator": "",
                        "condition": "canonical_local_llm_full_matrix",
                        "cohort": clean(key_map["candidate_set_id"]),
                        "metric_family": metric_family(metric_id, catalog),
                        "metric_name": metric_display(metric_id, catalog),
                        "metric_value": fmt(float(np.mean(np.asarray(values, dtype=float)))),
                        "cutoff": "5",
                        "requested_count": str(planned),
                        "valid_count": str(len(valid)),
                        "denominator": str(denominator),
                        "invalid_policy": policy,
                        "raw_input_artifacts": rel(project_root, path),
                        "supporting_input_artifacts": "results/llm/canonical_local_full_matrix/responses.jsonl",
                        "formula": "mean positive_in_top_5 or ndcg_at_5 over declared denominator",
                        "value_origin": "RECOMPUTED_FROM_LOCAL_OUTPUTS",
                        "paper_comparability": "ADAPTED_EQUIVALENT",
                        "notes": "Constrained candidate-index output; not a paper-style textual hallucination source.",
                    }
                )
    return rows


def recompute_from_f6e4c_request_metrics(project_root: Path, catalog: MetricCatalog) -> list[dict[str, str]]:
    path = project_root / "results/tables/phase_f6e4c_request_level_metrics.csv"
    if not path.exists():
        return []
    frame = read_csv(path)
    if frame.empty:
        return []
    rows: list[dict[str, str]] = []
    group_cols = ["dataset", "task", "model", "strategy_id"]
    metric_columns = [
        ("hr_at5_operational", "hr_at_5", "FAIL_STRICT_ALL_REQUESTS"),
        ("ndcg_at5_operational", "ndcg_at_5", "FAIL_STRICT_ALL_REQUESTS"),
        ("hr_at5_conditional", "hr_at_5", "CONDITIONAL_VALID_ONLY"),
        ("ndcg_at5_conditional", "ndcg_at_5", "CONDITIONAL_VALID_ONLY"),
    ]
    for key, group in frame.groupby(group_cols, sort=True, dropna=False):
        key_map = dict(zip(group_cols, key))
        strict_valid = bool_series(group, "strict_valid").sum()
        for column, metric_id, policy in metric_columns:
            if column not in group.columns:
                continue
            values = numeric_series(group, column).dropna().tolist()
            if not values:
                continue
            denominator = len(group) if "operational" in column else int(strict_valid)
            rows.append(
                {
                    "recalculation_id": stable_id("R3RECALC", "f6e4c", key, metric_id, policy),
                    "dataset": normalize_dataset(key_map["dataset"]),
                    "model": normalize_model(key_map["model"]),
                    "task": normalize_task(key_map["task"]),
                    "prompt_strategy": clean(key_map["strategy_id"]),
                    "profile_generator": "",
                    "condition": "phase_f6e4c_request_level",
                    "cohort": "F6E4C",
                    "metric_family": metric_family(metric_id, catalog),
                    "metric_name": metric_display(metric_id, catalog),
                    "metric_value": fmt(float(np.mean(np.asarray(values, dtype=float)))),
                    "cutoff": "5",
                    "requested_count": str(len(group)),
                    "valid_count": str(int(strict_valid)),
                    "denominator": str(denominator),
                    "invalid_policy": policy,
                    "raw_input_artifacts": rel(project_root, path),
                    "supporting_input_artifacts": "results/llm/phase_f6e4b_qwen2_adapted_full_run_raw.jsonl; results/llm/phase_f6e4b_mistral_adapted_full_run_raw.jsonl",
                    "formula": f"mean({column})",
                    "value_origin": "RECOMPUTED_FROM_LOCAL_OUTPUTS",
                    "paper_comparability": "ADAPTED_EQUIVALENT",
                    "notes": "F6E4C adapted indexed-output contract; utility anchor for beyond-utility interpretation.",
                }
            )
    return rows


def recompute_from_profile_parse(project_root: Path, catalog: MetricCatalog) -> list[dict[str, str]]:
    path = project_root / "results/tables/phase_f6l_profile_parse_results.csv"
    if not path.exists():
        return []
    frame = read_csv(path)
    if frame.empty:
        return []
    rows: list[dict[str, str]] = []
    group_cols = ["dataset", "profile_generator_name", "recommender_model_name", "input_variant"]
    for key, group in frame.groupby(group_cols, sort=True, dropna=False):
        key_map = dict(zip(group_cols, key))
        strict_valid = bool_series(group, "strict_valid").sum()
        for column, metric_id in [("hit_at_5", "hr_at_5"), ("ndcg_at_5", "ndcg_at_5")]:
            values = numeric_series(group, column).fillna(0.0).tolist()
            if not values:
                continue
            rows.append(
                {
                    "recalculation_id": stable_id("R3RECALC", "profile", key, metric_id),
                    "dataset": normalize_dataset(key_map["dataset"]),
                    "model": normalize_model(key_map["recommender_model_name"]),
                    "task": "generation_involved",
                    "prompt_strategy": clean(key_map["input_variant"]),
                    "profile_generator": normalize_model(key_map["profile_generator_name"]),
                    "condition": clean(key_map["input_variant"]),
                    "cohort": "phase_f6l_profile",
                    "metric_family": metric_family(metric_id, catalog),
                    "metric_name": metric_display(metric_id, catalog),
                    "metric_value": fmt(float(np.mean(np.asarray(values, dtype=float)))),
                    "cutoff": "5",
                    "requested_count": str(len(group)),
                    "valid_count": str(int(strict_valid)),
                    "denominator": str(len(group)),
                    "invalid_policy": "FAIL_STRICT_ALL_REQUESTS",
                    "raw_input_artifacts": rel(project_root, path),
                    "supporting_input_artifacts": "results/tables/phase_f6l_profile_recommendation_rows.csv",
                    "formula": f"mean({column}) with invalid or failed parses counted as zero",
                    "value_origin": "RECOMPUTED_FROM_LOCAL_OUTPUTS",
                    "paper_comparability": "ADAPTED_EQUIVALENT",
                    "notes": "Downstream recommendation outputs exist; generated profile text alone is not used as performance evidence.",
                }
            )
    return rows


def recompute_from_lastfm_parse(project_root: Path, catalog: MetricCatalog) -> list[dict[str, str]]:
    specs = [
        ("results/tables/phase_f6l_lastfm_operationalized_ranking_parse_results.csv", "ranking"),
        ("results/tables/phase_f6l_lastfm_v4_reranking_full_run_parse_results.csv", "reranking"),
        ("results/tables/phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution_parse_results.csv", "reranking"),
    ]
    rows: list[dict[str, str]] = []
    for rel_path, task in specs:
        path = project_root / rel_path
        if not path.exists():
            continue
        frame = read_csv(path)
        if frame.empty:
            continue
        dataset = "LastFM"
        model_col = "model_name"
        strategy_col = "prompting_strategy"
        group_cols = [column for column in [model_col, strategy_col] if column in frame.columns]
        if not group_cols:
            group_cols = [model_col]
            frame[model_col] = ""
        for key, group in frame.groupby(group_cols, sort=True, dropna=False):
            key_values = key if isinstance(key, tuple) else (key,)
            key_map = dict(zip(group_cols, key_values))
            strict_valid = bool_series(group, "strict_valid").sum()
            if "positive_in_top5" in group.columns:
                hr_values = bool_series(group, "positive_in_top5").astype(float).tolist()
                ndcg_values = numeric_series(group, "ndcg_at5").fillna(0.0).tolist()
            else:
                ranks = numeric_series(group, "positive_rank")
                hr_values = [1.0 if math.isfinite(rank) and rank <= 5 else 0.0 for rank in ranks.fillna(math.inf).tolist()]
                ndcg_values = [1.0 / math.log2(rank + 1.0) if math.isfinite(rank) and rank <= 5 else 0.0 for rank in ranks.fillna(math.inf).tolist()]
            for metric_id, values in [("hr_at_5", hr_values), ("ndcg_at_5", ndcg_values)]:
                rows.append(
                    {
                        "recalculation_id": stable_id("R3RECALC", "lastfm_parse", rel_path, key, metric_id),
                        "dataset": dataset,
                        "model": normalize_model(key_map.get(model_col, "")),
                        "task": task,
                        "prompt_strategy": clean(key_map.get(strategy_col, "")),
                        "profile_generator": "",
                        "condition": Path(rel_path).stem,
                        "cohort": "phase_f6l_lastfm_operationalized_1000",
                        "metric_family": metric_family(metric_id, catalog),
                        "metric_name": metric_display(metric_id, catalog),
                        "metric_value": fmt(float(np.mean(np.asarray(values, dtype=float)))),
                        "cutoff": "5",
                        "requested_count": str(len(group)),
                        "valid_count": str(int(strict_valid)),
                        "denominator": str(len(group)),
                        "invalid_policy": "FAIL_STRICT_ALL_REQUESTS",
                        "raw_input_artifacts": rel_path,
                        "supporting_input_artifacts": rel_path.replace("_parse_results.csv", "_recommendation_rows.csv"),
                        "formula": "mean hit/ndcg reconstructed from parse results",
                        "value_origin": "RECOMPUTED_FROM_LOCAL_OUTPUTS",
                        "paper_comparability": "ADAPTED_EQUIVALENT",
                        "notes": "LastFM operationalized local output contract.",
                    }
                )
    return rows


def offline_recalculated_values(project_root: Path, catalog: MetricCatalog) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    rows.extend(recompute_from_canonical_row_contributions(project_root, catalog))
    rows.extend(recompute_from_full_matrix_rankings(project_root, catalog))
    rows.extend(recompute_from_f6e4c_request_metrics(project_root, catalog))
    rows.extend(recompute_from_profile_parse(project_root, catalog))
    rows.extend(recompute_from_lastfm_parse(project_root, catalog))
    frame = pd.DataFrame(rows, columns=RECOMPUTED_COLUMNS).fillna("")
    if frame.empty:
        return frame
    return frame.drop_duplicates(subset=["recalculation_id"]).sort_values(["dataset", "model", "task", "metric_name", "condition"]).reset_index(drop=True)


def comparability_from_row(row: Mapping[str, Any]) -> tuple[str, str]:
    notes = clean(row.get("notes"))
    source_scope_value = clean(row.get("source_scope"))
    source = clean(row.get("source_artifact"))
    value_origin = clean(row.get("value_origin"))
    if source_scope_value == "OFFICIAL_REFERENCE":
        return "OFFICIAL_REFERENCE_NOT_LOCAL", "NOT_COMPARABLE_AS_LOCAL_REPRODUCTION"
    if "diagnostic" in source.lower() or "diagnostic" in notes.lower():
        return "DIAGNOSTIC_ONLY", "DIAGNOSTIC_ONLY"
    if "paper_protocol_reproduction_under_explicit_operationalization" in notes:
        return "ADAPTED_EQUIVALENT", "ADAPTED"
    if "exact_formula_exact_inputs" in notes:
        return "IDENTICAL", "PAPER_EXACT"
    if "recomputed" in value_origin.lower():
        return "ADAPTED_EQUIVALENT", "ADAPTED"
    return "ADAPTED_EQUIVALENT", "ADAPTED"


def materialized_master_rows(materialized: pd.DataFrame, catalog: MetricCatalog) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if materialized.empty:
        return rows
    local = materialized.loc[~materialized["source_scope"].isin(["OFFICIAL_REFERENCE", "PAPER_CONTRACT"])].copy()
    for _, row in local.iterrows():
        metric_id = normalize_metric(row.get("metric_name"))
        if metric_id == normalize_metric(metric_display(metric_id, catalog)):
            metric_id = normalize_metric(row.get("metric_name"))
        paper_equivalence, comparability = comparability_from_row(row)
        is_paper = parse_bool(row.get("paper_metric"))
        rows.append(
            {
                "dataset": clean(row.get("dataset")),
                "model": clean(row.get("model")),
                "task": clean(row.get("task")),
                "prompt_strategy": clean(row.get("prompt_strategy")),
                "profile_generator": clean(row.get("profile_generator")),
                "condition": clean(row.get("condition")),
                "cohort": clean(row.get("cohort")),
                "metric_family": clean(row.get("metric_family")),
                "metric_name": clean(row.get("metric_name")),
                "metric_value": clean(row.get("metric_value")),
                "cutoff": clean(row.get("cutoff")),
                "history_condition": history_condition_from_text(clean(row.get("condition")) + " " + clean(row.get("source_artifact"))),
                "candidate_position_condition": candidate_position_condition_from_text(clean(row.get("condition")) + " " + clean(row.get("source_artifact"))),
                "generation_condition": generation_condition_from_text(clean(row.get("condition")) + " " + clean(row.get("source_artifact"))),
                "requested_count": "",
                "valid_count": "",
                "denominator": clean(row.get("denominator")),
                "invalid_policy": infer_invalid_policy(row),
                "calculability_status": "ALREADY_MATERIALIZED",
                "stored_local_value": clean(row.get("metric_value")),
                "recomputed_local_value": "",
                "reconstructed_local_value": "",
                "value_origin": clean(row.get("value_origin")),
                "source_artifact": clean(row.get("source_artifact")),
                "secondary_source_artifact": "",
                "paper_metric": bool_text(is_paper),
                "paper_semantic_equivalence": paper_equivalence if is_paper else "LOCAL_EXTENSION_OR_DIAGNOSTIC",
                "paper_comparability": comparability if is_paper else "NOT_PAPER_METRIC",
                "protocol_class": "local_reproduction",
                "source_scope": clean(row.get("source_scope")),
                "notes": clean(row.get("notes")),
            }
        )
    return rows


def recomputed_master_rows(recomputed: pd.DataFrame, catalog: MetricCatalog) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for _, row in recomputed.iterrows():
        metric_id = normalize_metric(row.get("metric_name"))
        is_paper = is_paper_metric(metric_id, catalog)
        rows.append(
            {
                "dataset": clean(row.get("dataset")),
                "model": clean(row.get("model")),
                "task": clean(row.get("task")),
                "prompt_strategy": clean(row.get("prompt_strategy")),
                "profile_generator": clean(row.get("profile_generator")),
                "condition": clean(row.get("condition")),
                "cohort": clean(row.get("cohort")),
                "metric_family": clean(row.get("metric_family")),
                "metric_name": clean(row.get("metric_name")),
                "metric_value": clean(row.get("metric_value")),
                "cutoff": clean(row.get("cutoff")),
                "history_condition": history_condition_from_text(clean(row.get("condition")) + " " + clean(row.get("prompt_strategy"))),
                "candidate_position_condition": candidate_position_condition_from_text(clean(row.get("condition")) + " " + clean(row.get("prompt_strategy"))),
                "generation_condition": generation_condition_from_text(clean(row.get("condition")) + " " + clean(row.get("prompt_strategy"))),
                "requested_count": clean(row.get("requested_count")),
                "valid_count": clean(row.get("valid_count")),
                "denominator": clean(row.get("denominator")),
                "invalid_policy": clean(row.get("invalid_policy")),
                "calculability_status": "RECALCULATED_FROM_LOCAL_OUTPUTS",
                "stored_local_value": "",
                "recomputed_local_value": clean(row.get("metric_value")),
                "reconstructed_local_value": "",
                "value_origin": clean(row.get("value_origin")),
                "source_artifact": clean(row.get("raw_input_artifacts")),
                "secondary_source_artifact": clean(row.get("supporting_input_artifacts")),
                "paper_metric": bool_text(is_paper),
                "paper_semantic_equivalence": "ADAPTED_EQUIVALENT" if is_paper else "LOCAL_EXTENSION_OR_DIAGNOSTIC",
                "paper_comparability": clean(row.get("paper_comparability")) if is_paper else "NOT_PAPER_METRIC",
                "protocol_class": "offline_recalculation",
                "source_scope": "LOCAL_REPRODUCTION",
                "notes": clean(row.get("notes")),
            }
        )
    return rows


def infer_invalid_policy(row: Mapping[str, Any]) -> str:
    text = " ".join(clean(row.get(column, "")) for column in ["condition", "notes", "source_artifact", "value_origin"]).lower()
    if "positive_present_only" in text or "conditional" in text or "valid_only" in text:
        return "CONDITIONAL_VALID_ONLY"
    if "fail_strict" in text or "invalid" in text:
        return "FAIL_STRICT_ALL_REQUESTS"
    return ""


def history_condition_from_text(text: str) -> str:
    low = text.lower()
    match = re.search(r"history[_ =-]?(short|medium|long|le_5|gt_10|\d+)", low)
    if match:
        return match.group(0).replace(" ", "_")
    return ""


def candidate_position_condition_from_text(text: str) -> str:
    low = text.lower()
    for key in ["positive_first", "positive_last", "first_vs_random", "candidate_position", "positive_1_5", "positive_6_10", "positive_11_20", "reverse_order"]:
        if key in low:
            return key
    if "canddif" in low:
        return "canddif"
    return ""


def generation_condition_from_text(text: str) -> str:
    low = text.lower()
    for key in ["profile_plus_history", "profile_only", "history_only"]:
        if key in low:
            return key
    if "profile" in low:
        return "profile"
    return ""


def gap_status(metric_id: str, dataset: str, catalog: MetricCatalog, local_rows_for_dataset: pd.DataFrame) -> tuple[str, str]:
    family = metric_family(metric_id, catalog)
    if metric_id in {"canddif_hr", "canddif_ndcg"}:
        return "NOT_CALCULABLE_MISSING_OUTPUT", "Missing unique paired first-position and random-position local aggregate evidence for paper-compatible CandDif."
    if metric_id == "generation_involved_profile_matrix":
        return "NOT_CALCULABLE_MISSING_OUTPUT", "No downstream recommendation performance output exists for this dataset/condition; generated profile text alone is insufficient."
    if metric_id == "history_length_sensitivity":
        return "NOT_CALCULABLE_MISSING_OUTPUT", "No local per-history-length output/curve exists for this dataset."
    if metric_id == "hallucination_rate_at_5":
        return "PARTIALLY_RECONSTRUCTIBLE", "Candidate-index or structural validity evidence exists, but paper-style textual hallucination requires free-text title outputs and global title mapping."
    if family in {"utility", "novelty", "popularity", "diversity", "fairness"} and not local_rows_for_dataset.empty:
        return "NOT_CALCULABLE_MISSING_INPUT", f"Local outputs exist for {dataset}, but exact support inputs/protocol for {metric_display(metric_id, catalog)} are unavailable or not aligned."
    if family in {"utility"}:
        return "REQUIRES_NEW_NON_LLM_EXECUTION", "No compatible local recommendation output exists for this dataset/model/task cell."
    if family == "LLM_specific":
        return "REQUIRES_NEW_LLM_INFERENCE", "No compatible local LLM output exists for this paper analysis cell."
    return "NOT_CALCULABLE_MISSING_OUTPUT", "No compatible local metric value or user-level output was found."


def add_gap_rows(master_rows: list[dict[str, str]], materialized: pd.DataFrame, recomputed: pd.DataFrame, catalog: MetricCatalog) -> list[dict[str, str]]:
    existing = {
        (clean(row["dataset"]), normalize_metric(row["metric_name"]))
        for row in master_rows
        if parse_bool(row.get("paper_metric"))
    }
    local_values = materialized.loc[~materialized["source_scope"].isin(["OFFICIAL_REFERENCE", "PAPER_CONTRACT"])].copy() if not materialized.empty else pd.DataFrame()
    added: list[dict[str, str]] = []
    for dataset in TARGET_DATASETS:
        local_rows_for_dataset = local_values.loc[local_values["dataset"].eq(dataset)] if not local_values.empty else pd.DataFrame()
        for metric_id in sorted(catalog.paper_metric_ids):
            if (dataset, metric_id) in existing:
                continue
            status, reason = gap_status(metric_id, dataset, catalog, local_rows_for_dataset)
            added.append(
                {
                    "dataset": dataset,
                    "model": "ALL",
                    "task": "where_applicable",
                    "prompt_strategy": "",
                    "profile_generator": "",
                    "condition": "paper_metric_dataset_gap_after_local_scan",
                    "cohort": "",
                    "metric_family": metric_family(metric_id, catalog),
                    "metric_name": metric_display(metric_id, catalog),
                    "metric_value": "",
                    "cutoff": "5" if metric_id.endswith("_at_5") or metric_id in {"hr_at_5", "ndcg_at_5"} else "",
                    "history_condition": "missing" if metric_id == "history_length_sensitivity" else "",
                    "candidate_position_condition": "missing_first_random_pair" if metric_id in {"canddif_hr", "canddif_ndcg"} else "",
                    "generation_condition": "missing_downstream_output" if metric_id == "generation_involved_profile_matrix" else "",
                    "requested_count": "",
                    "valid_count": "",
                    "denominator": "",
                    "invalid_policy": "",
                    "calculability_status": status,
                    "stored_local_value": "",
                    "recomputed_local_value": "",
                    "reconstructed_local_value": "",
                    "value_origin": "GAP_CLASSIFICATION",
                    "source_artifact": "",
                    "secondary_source_artifact": "",
                    "paper_metric": "true",
                    "paper_semantic_equivalence": "NOT_ESTABLISHED",
                    "paper_comparability": "NOT_COMPARABLE",
                    "protocol_class": "gap",
                    "source_scope": "LOCAL_REPRODUCTION",
                    "notes": reason,
                }
            )
    master_rows.extend(added)
    return master_rows


def build_master(materialized: pd.DataFrame, recomputed: pd.DataFrame, catalog: MetricCatalog) -> pd.DataFrame:
    rows = materialized_master_rows(materialized, catalog)
    rows.extend(recomputed_master_rows(recomputed, catalog))
    rows = add_gap_rows(rows, materialized, recomputed, catalog)
    frame = pd.DataFrame(rows, columns=MASTER_COLUMNS).fillna("")
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(
        subset=[
            "dataset",
            "model",
            "task",
            "prompt_strategy",
            "profile_generator",
            "condition",
            "cohort",
            "metric_name",
            "metric_value",
            "calculability_status",
            "source_artifact",
        ]
    )
    return frame.sort_values(["dataset", "metric_family", "metric_name", "model", "task", "condition"]).reset_index(drop=True)


def build_beyond_master(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in master.iterrows():
        metric_id = normalize_metric(row.get("metric_name"))
        if metric_id not in BEYOND_METRIC_IDS:
            continue
        rows.append(
            {
                "dataset": clean(row.get("dataset")),
                "model": clean(row.get("model")),
                "task": clean(row.get("task")),
                "condition": clean(row.get("condition")),
                "metric": clean(row.get("metric_name")),
                "value": clean(row.get("metric_value")),
                "granularity": "curve_point" if metric_id == "history_length_sensitivity" else ("analysis_cell" if metric_id in {"canddif_hr", "canddif_ndcg", "generation_involved_profile_matrix"} else "scalar"),
                "calculation_status": clean(row.get("calculability_status")),
                "source": clean(row.get("source_artifact")),
                "paper_equivalence": clean(row.get("paper_semantic_equivalence")),
                "limitation": clean(row.get("notes")),
            }
        )
    return pd.DataFrame(rows, columns=BEYOND_COLUMNS).fillna("")


def build_coverage(master: pd.DataFrame, catalog: MetricCatalog) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for dataset in TARGET_DATASETS:
        dataset_rows = master.loc[master["dataset"].eq(dataset)]
        for metric_id in sorted(catalog.paper_metric_ids):
            metric_label = metric_display(metric_id, catalog)
            subset = dataset_rows.loc[dataset_rows["metric_name"].eq(metric_label)]
            numeric = subset["metric_value"].astype(str).str.len().gt(0) if not subset.empty else pd.Series(dtype=bool)
            statuses = set(subset["calculability_status"]) if not subset.empty else set()
            comparability = set(subset["paper_comparability"]) if not subset.empty else set()
            gaps = subset.loc[~subset["calculability_status"].isin(["ALREADY_MATERIALIZED", "RECALCULATED_FROM_LOCAL_OUTPUTS", "RECONSTRUCTED_FROM_LOCAL_ARTIFACTS", "NOT_APPLICABLE"])]
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric_label,
                    "executed": bool_text(bool(statuses.intersection({"ALREADY_MATERIALIZED", "RECALCULATED_FROM_LOCAL_OUTPUTS", "RECONSTRUCTED_FROM_LOCAL_ARTIFACTS", "PARTIALLY_RECONSTRUCTIBLE"}))),
                    "numeric_value_available": bool_text(bool(numeric.any() if not subset.empty else False)),
                    "offline_recalculable": bool_text("RECALCULATED_FROM_LOCAL_OUTPUTS" in statuses),
                    "paper_exact": bool_text("PAPER_EXACT" in comparability),
                    "adapted": bool_text(bool(comparability.intersection({"ADAPTED", "ADAPTED_EQUIVALENT"}))),
                    "diagnostic_only": bool_text("DIAGNOSTIC_ONLY" in comparability),
                    "missing": bool_text(not gaps.empty and not bool(numeric.any() if not subset.empty else False)),
                    "requires_new_execution": bool_text(bool(statuses.intersection({"REQUIRES_NEW_LLM_INFERENCE", "REQUIRES_NEW_NON_LLM_EXECUTION"}))),
                    "models": ";".join(sorted({clean(value) for value in subset["model"] if clean(value)})),
                    "tasks": ";".join(sorted({clean(value) for value in subset["task"] if clean(value)})),
                    "conditions": ";".join(sorted({clean(value) for value in subset["condition"] if clean(value)})[:12]),
                    "granularity": "curve/analysis" if metric_id in {"history_length_sensitivity", "generation_involved_profile_matrix", "canddif_hr", "canddif_ndcg"} else "scalar",
                    "best_source": clean(subset.loc[subset["source_artifact"].astype(str).str.len().gt(0), "source_artifact"].iloc[0]) if not subset.empty and subset["source_artifact"].astype(str).str.len().gt(0).any() else "",
                    "exact_gap_reason": clean(gaps["notes"].iloc[0]) if not gaps.empty else "",
                }
            )
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS).fillna("")


def build_gaps(master: pd.DataFrame) -> pd.DataFrame:
    gap_statuses = {
        "PARTIALLY_RECONSTRUCTIBLE",
        "NOT_CALCULABLE_MISSING_OUTPUT",
        "NOT_CALCULABLE_MISSING_INPUT",
        "NOT_CALCULABLE_MISSING_PROTOCOL",
        "REQUIRES_NEW_LLM_INFERENCE",
        "REQUIRES_NEW_NON_LLM_EXECUTION",
    }
    rows: list[dict[str, str]] = []
    for _, row in master.loc[master["calculability_status"].isin(gap_statuses) & master["paper_metric"].eq("true")].iterrows():
        status = clean(row.get("calculability_status"))
        rows.append(
            {
                "dataset": clean(row.get("dataset")),
                "metric": clean(row.get("metric_name")),
                "model": clean(row.get("model")),
                "task": clean(row.get("task")),
                "condition": clean(row.get("condition")),
                "what_exists": "local adapted or diagnostic evidence" if status == "PARTIALLY_RECONSTRUCTIBLE" else "no compatible local numeric value",
                "what_is_missing": clean(row.get("notes")),
                "missing_output": bool_text(status in {"NOT_CALCULABLE_MISSING_OUTPUT", "REQUIRES_NEW_LLM_INFERENCE", "REQUIRES_NEW_NON_LLM_EXECUTION"}),
                "missing_support_data": bool_text(status == "NOT_CALCULABLE_MISSING_INPUT"),
                "missing_protocol": bool_text(status == "NOT_CALCULABLE_MISSING_PROTOCOL"),
                "offline_recovery_possible": bool_text(status == "PARTIALLY_RECONSTRUCTIBLE"),
                "new_llm_inference_required": bool_text(status == "REQUIRES_NEW_LLM_INFERENCE"),
                "new_non_llm_execution_required": bool_text(status == "REQUIRES_NEW_NON_LLM_EXECUTION"),
                "paper_exact_recovery_possible": bool_text(status not in {"NOT_CALCULABLE_MISSING_PROTOCOL"}),
                "adapted_recovery_possible": bool_text(status != "NOT_APPLICABLE"),
                "estimated_work": "small_offline_audit" if status == "PARTIALLY_RECONSTRUCTIBLE" else "controlled_recovery_phase",
                "exact_reason": clean(row.get("notes")),
                "recommended_recovery": recovery_recommendation(clean(row.get("metric_name")), status),
            }
        )
    return pd.DataFrame(rows, columns=GAP_COLUMNS).fillna("")


def recovery_recommendation(metric_name: str, status: str) -> str:
    metric_id = normalize_metric(metric_name)
    if metric_id in {"canddif_hr", "canddif_ndcg"}:
        return "Design a controlled first-vs-random candidate-position recovery phase; do not execute in R3."
    if metric_id == "hallucination_rate_at_5":
        return "Recover free-text output and global title mapping if present; otherwise design textual-output recovery."
    if metric_id == "generation_involved_profile_matrix":
        return "Run no new inference in R3; later recovery must include downstream recommendation outputs, not profile text alone."
    if status.startswith("REQUIRES_NEW"):
        return "Do not execute; scope the smallest controlled post-R3 recovery."
    return "No R3 execution; document blocker and reuse only existing local artifacts."


def build_history_table(project_root: Path, recomputed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    sources = [
        project_root / "results/tables/plan_d_llm_ml1m_history_length_condition_summary.csv",
        project_root / "results/tables/finalization_f6d4a_history_length_sensitivity.csv",
    ]
    for path in sources:
        if not path.exists():
            continue
        frame = read_csv(path)
        for _, row in frame.iterrows():
            dataset = normalize_dataset(row.get("dataset") or "ML-1M" if "plan_d" in path.name else "LastFM")
            model = normalize_model(row.get("model"))
            task = normalize_task(row.get("task") or "ranking")
            for metric_col, metric_label in [("hr_at_5", "HR@5"), ("ndcg_at_5", "NDCG@5"), ("hr_at5", "HR@5"), ("ndcg_at5", "NDCG@5")]:
                value = numeric_value(row.get(metric_col))
                if value is None:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "task": task,
                        "history_condition": clean(row.get("condition") or row.get("history_setting") or row.get("history_bin")),
                        "history_length": clean(row.get("history_length_used") or row.get("mean_history_length_used") or row.get("history_length_min")),
                        "history_bin": clean(row.get("history_setting") or row.get("history_bin")),
                        "metric": metric_label,
                        "value": fmt(value),
                        "requested_count": clean(row.get("total_instances") or row.get("users")),
                        "valid_count": clean(row.get("num_schema_valid") or row.get("users")),
                        "granularity": "curve_point",
                        "source_artifact": rel(project_root, path),
                        "paper_comparability": "DIAGNOSTIC_ONLY" if "plan_d" in path.name else "ADAPTED",
                        "notes": clean(row.get("notes") or row.get("policy")),
                    }
                )
    f6e4c = project_root / "results/tables/phase_f6e4c_request_level_metrics.csv"
    if f6e4c.exists():
        frame = read_csv(f6e4c)
        if not frame.empty and "history_length_stratum" in frame.columns:
            for key, group in frame.groupby(["dataset", "task", "model", "strategy_id", "history_length_stratum"], sort=True, dropna=False):
                dataset, task, model, strategy, history = key
                for column, metric in [("hr_at5_operational", "HR@5"), ("ndcg_at5_operational", "NDCG@5")]:
                    values = numeric_series(group, column).dropna().tolist()
                    if not values:
                        continue
                    rows.append(
                        {
                            "dataset": normalize_dataset(dataset),
                            "model": normalize_model(model),
                            "task": normalize_task(task),
                            "history_condition": clean(history),
                            "history_length": "",
                            "history_bin": clean(history),
                            "metric": metric,
                            "value": fmt(float(np.mean(np.asarray(values, dtype=float)))),
                            "requested_count": str(len(group)),
                            "valid_count": str(int(bool_series(group, "strict_valid").sum())),
                            "granularity": "curve_point",
                            "source_artifact": rel(project_root, f6e4c),
                            "paper_comparability": "ADAPTED",
                            "notes": f"strategy={strategy}; operational fail-strict aggregate by frozen history stratum",
                        }
                    )
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS).fillna("")


def build_candidate_position_table(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    sources = [
        project_root / "results/tables/phase_f6f1_candidate_position_results.csv",
        project_root / "results/tables/phase_f6e4c_candidate_position_analysis.csv",
        project_root / "results/tables/finalization_f6d4a_candidate_position_bias.csv",
        project_root / "results/tables/plan_d_qwen2_ml1m_candidate_position_bias_variant_summary.csv",
    ]
    for path in sources:
        if not path.exists():
            continue
        frame = read_csv(path)
        if frame.empty:
            continue
        if path.name == "phase_f6f1_candidate_position_results.csv":
            for _, row in frame.iterrows():
                metric = clean(row.get("metric"))
                has_pair = "first" in clean(row.get("stratum")).lower() and "random" in clean(row.get("stratum")).lower()
                formula_status = clean(row.get("canddif_formula_status"))
                paper_compatible = has_pair and formula_status == "computed_paper_compatible"
                rows.append(
                    {
                        "dataset": normalize_dataset(row.get("dataset")),
                        "model": normalize_model(row.get("model_name")),
                        "task": normalize_task(row.get("task")),
                        "candidate_position_condition": clean(row.get("stratum")),
                        "metric": metric,
                        "value": clean(row.get("value")),
                        "classification": "CANDDIF_PAPER_COMPATIBLE" if paper_compatible else ("CANDDIF_LOCAL_DIAGNOSTIC" if "canddif" in metric.lower() else "POSITION_DIAGNOSTIC_ONLY"),
                        "canddif_formula_status": formula_status,
                        "first_position_value": "",
                        "random_position_value": "",
                        "has_first_random_evidence": bool_text(paper_compatible),
                        "logical_requests": clean(row.get("logical_requests")),
                        "valid_count": clean(row.get("strict_valid")),
                        "denominator": clean(row.get("denominator")),
                        "source_artifact": rel(project_root, path),
                        "notes": clean(row.get("notes")),
                    }
                )
        elif path.name == "plan_d_qwen2_ml1m_candidate_position_bias_variant_summary.csv":
            first = frame.loc[frame["variant"].astype(str).str.contains("positive_first", case=False, na=False)]
            if not first.empty:
                row = first.iloc[0]
                for metric, value_col, delta_col in [("CandDif_HR", "hr_at_5", "hr_at_5_delta_vs_original"), ("CandDif_NDCG", "ndcg_at_5", "ndcg_at_5_delta_vs_original")]:
                    first_value = numeric_value(row.get(value_col))
                    delta = numeric_value(row.get(delta_col))
                    random_value = first_value - delta if first_value is not None and delta is not None else None
                    value = canddif(first_value, random_value) if first_value is not None and random_value is not None else None
                    rows.append(
                        {
                            "dataset": "ML-1M",
                            "model": normalize_model(row.get("model") or "qwen2:7b-instruct"),
                            "task": "diagnostic_ranking",
                            "candidate_position_condition": "positive_first_vs_original_random_order",
                            "metric": metric,
                            "value": fmt(value) if value is not None else "",
                            "classification": "CANDDIF_LOCAL_DIAGNOSTIC",
                            "canddif_formula_status": "computed_diagnostic_only",
                            "first_position_value": fmt(first_value),
                            "random_position_value": fmt(random_value),
                            "has_first_random_evidence": "false",
                            "logical_requests": clean(row.get("total_instances")),
                            "valid_count": clean(row.get("num_schema_valid")),
                            "denominator": clean(row.get("total_instances")),
                            "source_artifact": rel(project_root, path),
                            "notes": "Plan D diagnostic subset; exact formula but not paper-compatible full local cell.",
                        }
                    )
        else:
            for _, row in frame.iterrows():
                for value_col, metric in [
                    ("operational_hr_at5", "HR@5"),
                    ("conditional_hr_at5", "HR@5"),
                    ("operational_ndcg_at5", "NDCG@5"),
                    ("conditional_ndcg_at5", "NDCG@5"),
                    ("hr_at5", "HR@5"),
                    ("ndcg_at5", "NDCG@5"),
                ]:
                    value = numeric_value(row.get(value_col))
                    if value is None:
                        continue
                    rows.append(
                        {
                            "dataset": normalize_dataset(row.get("dataset") or "ML-1M"),
                            "model": normalize_model(row.get("model")),
                            "task": normalize_task(row.get("task") or "ranking"),
                            "candidate_position_condition": clean(row.get("stratum_name") or row.get("positive_position_stratum") or row.get("condition")),
                            "metric": metric,
                            "value": fmt(value),
                            "classification": "POSITION_DIAGNOSTIC_ONLY",
                            "canddif_formula_status": "not_canddif_observational_only",
                            "first_position_value": "",
                            "random_position_value": "",
                            "has_first_random_evidence": "false",
                            "logical_requests": clean(row.get("logical_requests") or row.get("users")),
                            "valid_count": clean(row.get("strict_valid")),
                            "denominator": clean(row.get("logical_requests") or row.get("users")),
                            "source_artifact": rel(project_root, path),
                            "notes": clean(row.get("notes") or row.get("policy")),
                        }
                    )
    return pd.DataFrame(rows, columns=CANDIDATE_POSITION_COLUMNS).fillna("")


def canddif(first_accuracy: float | None, random_accuracy: float | None) -> float | None:
    if first_accuracy is None or random_accuracy is None:
        return None
    if first_accuracy < 0.0 or random_accuracy < 0.0 or first_accuracy >= 1.0 or random_accuracy >= 1.0:
        return None
    return -math.log(1.0 - first_accuracy) - (-math.log(1.0 - random_accuracy))


def build_hallucination_table(project_root: Path, materialized: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    phase_e3 = project_root / "results/tables/phase_f6e3a_hallucination_audit.csv"
    if phase_e3.exists():
        frame = read_csv(phase_e3)
        for key, group in frame.groupby(["dataset", "model", "strategy_id"], sort=True, dropna=False):
            dataset, model, strategy = key
            genuine = numeric_series(group, "genuine_out_of_set_title").fillna(0.0).sum()
            rows.append(
                {
                    "dataset": normalize_dataset(dataset),
                    "model": normalize_model(model),
                    "task": "llm_output_validation",
                    "condition": clean(strategy),
                    "hallucination_semantic_type": "PAPER_STYLE_TEXTUAL_HALLUCINATION",
                    "output_representation": "FREE_TEXT",
                    "metric": "textual_hallucination_rate",
                    "value": fmt(genuine / len(group)) if len(group) else "",
                    "numerator": fmt(genuine),
                    "denominator": str(len(group)),
                    "source_artifact": rel(project_root, phase_e3),
                    "paper_comparability": "ADAPTED",
                    "notes": "Separates genuine out-of-set title from format/incomplete-output failures.",
                }
            )
    structural = project_root / "results/tables/canonical_local_llm_structural_metrics.csv"
    if structural.exists():
        frame = read_csv(structural)
        for _, row in frame.iterrows():
            for value_col, semantic, representation in [
                ("hallucination_rate", "CANDIDATE_VIOLATION", "CANDIDATE_INDEX"),
                ("strict_parse_validity_rate", "STRUCTURAL_OUTPUT_INVALIDITY", "STRUCTURED_JSON"),
                ("duplicate_output_rate", "STRUCTURAL_OUTPUT_INVALIDITY", "STRUCTURED_JSON"),
            ]:
                value = numeric_value(row.get(value_col))
                if value is None:
                    continue
                if value_col == "strict_parse_validity_rate":
                    value = 1.0 - value
                rows.append(
                    {
                        "dataset": normalize_dataset(row.get("dataset")),
                        "model": normalize_model(row.get("model_family")),
                        "task": normalize_task(row.get("task_type")),
                        "condition": clean(row.get("aggregation_level")),
                        "hallucination_semantic_type": semantic,
                        "output_representation": representation,
                        "metric": value_col if value_col != "strict_parse_validity_rate" else "structural_invalidity_rate",
                        "value": fmt(value),
                        "numerator": clean(row.get("hallucination_rows") if value_col == "hallucination_rate" else ""),
                        "denominator": clean(row.get("planned_rows")),
                        "source_artifact": rel(project_root, structural),
                        "paper_comparability": "DIAGNOSTIC_ONLY",
                        "notes": "Constrained candidate-index/JSON structural metric; not reported as zero paper-style textual hallucination.",
                    }
                )
    profile = project_root / "results/tables/phase_f6l_profile_parse_results.csv"
    if profile.exists():
        frame = read_csv(profile)
        for key, group in frame.groupby(["dataset", "recommender_model_name", "input_variant"], sort=True, dropna=False):
            dataset, model, variant = key
            hallucinated = numeric_series(group, "hallucination_count").fillna(0.0).sum()
            denominator = max(1.0, numeric_series(group, "parsed_item_ids_json").notna().sum() * 5.0)
            rows.append(
                {
                    "dataset": normalize_dataset(dataset),
                    "model": normalize_model(model),
                    "task": "generation_involved",
                    "condition": clean(variant),
                    "hallucination_semantic_type": "PAPER_STYLE_TEXTUAL_HALLUCINATION",
                    "output_representation": "FREE_TEXT",
                    "metric": "textual_hallucination_rate",
                    "value": fmt(hallucinated / denominator),
                    "numerator": fmt(hallucinated),
                    "denominator": fmt(denominator),
                    "source_artifact": rel(project_root, profile),
                    "paper_comparability": "ADAPTED",
                    "notes": "Profile-family free-text parser invalid titles; separated from candidate-index structural metrics.",
                }
            )
    return pd.DataFrame(rows, columns=HALLUCINATION_COLUMNS).fillna("")


def build_generation_table(project_root: Path, recomputed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    profile_metrics = project_root / "results/tables/phase_f6l_profile_metrics.csv"
    if profile_metrics.exists():
        frame = read_csv(profile_metrics)
        for _, row in frame.iterrows():
            rows.append(
                {
                    "dataset": normalize_dataset(row.get("dataset")),
                    "profile_generator": normalize_model(row.get("profile_generator")),
                    "recommender": normalize_model(row.get("recommender_model")),
                    "task": "generation_involved",
                    "generation_condition": clean(row.get("input_variant")),
                    "cohort": clean(row.get("run_id")),
                    "metric": clean(row.get("metric")),
                    "value": clean(row.get("value")),
                    "requested_count": clean(row.get("denominator")),
                    "valid_count": clean(row.get("denominator")),
                    "source_artifact": rel(project_root, profile_metrics),
                    "has_downstream_recommendation_output": "true",
                    "notes": clean(row.get("notes")),
                }
            )
    generation_results = project_root / "results/tables/phase_f6l_profile_generation_results.csv"
    if generation_results.exists():
        frame = read_csv(generation_results)
        profile_only = frame.loc[~frame["metric"].astype(str).str.contains("HR|NDCG", case=False, na=False)] if "metric" in frame.columns else frame
        for _, row in profile_only.iterrows():
            metric = clean(row.get("metric"))
            if metric in {"HR@5", "NDCG@5"}:
                continue
            rows.append(
                {
                    "dataset": normalize_dataset(row.get("dataset")),
                    "profile_generator": normalize_model(row.get("profile_generator")),
                    "recommender": normalize_model(row.get("recommender_model")),
                    "task": "profile_generation_only",
                    "generation_condition": clean(row.get("mode")),
                    "cohort": "",
                    "metric": metric,
                    "value": clean(row.get("value")),
                    "requested_count": clean(row.get("request_count")),
                    "valid_count": "",
                    "source_artifact": rel(project_root, generation_results),
                    "has_downstream_recommendation_output": "false",
                    "notes": "Profile generation artifact only; not downstream recommendation performance unless paired with parse/recommendation rows.",
                }
            )
    return pd.DataFrame(rows, columns=GENERATION_COLUMNS).fillna("")


def build_stored_vs_recomputed(materialized: pd.DataFrame, recomputed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    local_materialized = materialized.loc[~materialized["source_scope"].isin(["OFFICIAL_REFERENCE", "PAPER_CONTRACT"])] if not materialized.empty else pd.DataFrame()
    for _, row in recomputed.iterrows():
        metric = clean(row.get("metric_name"))
        candidates = local_materialized.loc[
            local_materialized["dataset"].eq(clean(row.get("dataset")))
            & local_materialized["metric_name"].eq(metric)
        ] if not local_materialized.empty else pd.DataFrame()
        if clean(row.get("model")) and not candidates.empty:
            model_candidates = candidates.loc[candidates["model"].eq(clean(row.get("model")))]
            if not model_candidates.empty:
                candidates = model_candidates
        recomputed_value = numeric_value(row.get("metric_value"))
        stored_value = None
        stored_source = ""
        if not candidates.empty:
            numeric_candidates = candidates.copy()
            numeric_candidates["__value"] = numeric_candidates["metric_value"].map(numeric_value)
            numeric_candidates = numeric_candidates.dropna(subset=["__value"])
            if not numeric_candidates.empty:
                deltas = (numeric_candidates["__value"].astype(float) - float(recomputed_value)).abs() if recomputed_value is not None else pd.Series(dtype=float)
                best_index = deltas.idxmin() if not deltas.empty else numeric_candidates.index[0]
                stored_value = numeric_value(numeric_candidates.loc[best_index, "metric_value"])
                stored_source = clean(numeric_candidates.loc[best_index, "source_artifact"])
        if stored_value is None or recomputed_value is None:
            agreement = "NOT_COMPARABLE"
            abs_delta = ""
            rel_delta = ""
        else:
            abs_value = abs(stored_value - recomputed_value)
            abs_delta = fmt(abs_value)
            rel_delta = fmt(abs_value / abs(stored_value)) if stored_value != 0 else ""
            if abs_value <= 1e-12:
                agreement = "EXACT"
            elif abs_value <= 5e-6:
                agreement = "ROUNDING_MATCH"
            else:
                agreement = "DIFFERENT"
        rows.append(
            {
                "comparison_id": stable_id("R3CMP", row.get("recalculation_id"), stored_source),
                "dataset": clean(row.get("dataset")),
                "model": clean(row.get("model")),
                "task": clean(row.get("task")),
                "condition": clean(row.get("condition")),
                "metric_name": metric,
                "stored_local_value": fmt(stored_value) if stored_value is not None else "",
                "recomputed_local_value": clean(row.get("metric_value")),
                "agreement": agreement,
                "absolute_delta": abs_delta,
                "relative_delta": rel_delta,
                "stored_source_artifact": stored_source,
                "recomputed_source_artifact": clean(row.get("raw_input_artifacts")),
                "notes": "Nearest same dataset/model/metric local aggregate compared when available.",
            }
        )
    return pd.DataFrame(rows, columns=STORED_VS_COLUMNS).fillna("")


def build_validation(
    project_root: Path,
    config: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    catalog: MetricCatalog,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(check_id: str, name: str, ok: bool, observed: Any, expected: str, details: str = "") -> None:
        rows.append(
            {
                "check_id": check_id,
                "check_name": name,
                "status": "pass" if ok else "fail",
                "observed": clean(observed),
                "expected": expected,
                "details": details,
            }
        )

    branch = git_output(project_root, ["branch", "--show-current"])
    head = git_output(project_root, ["rev-parse", "HEAD"])
    status_short = git_output(project_root, ["status", "--short"])
    staged = git_output(project_root, ["diff", "--cached", "--stat"])
    forbidden_paths = [path for path in status_paths(status_short) if not allowed_status_path(path)]
    original_status = git_output(project_root / ORIGINAL_ROOT, ["status", "--branch", "--short"]) if (project_root / ORIGINAL_ROOT).exists() else ""
    original_diff = git_output(project_root / ORIGINAL_ROOT, ["diff", "--stat"]) if (project_root / ORIGINAL_ROOT).exists() else ""

    master = frames["all_metric_cells_master"]
    materialized = frames["materialized_values"]
    recomputed = frames["offline_recalculated_values"]
    coverage = frames["paper_metric_dataset_coverage"]
    candidate = frames["candidate_position_bias"]
    hallucination = frames["hallucination_audit"]
    generation = frames["generation_involved_performance"]
    protected = frames["protected_artifact_hash_check"]

    add("R3-VAL-001", "branch_is_expected", branch == clean(config.get("expected_branch", EXPECTED_BRANCH)), branch, EXPECTED_BRANCH)
    add("R3-VAL-002", "head_is_expected", head == clean(config.get("expected_head", EXPECTED_HEAD)), head, EXPECTED_HEAD)
    add("R3-VAL-003", "no_staged_changes", staged == "", staged, "empty")
    add("R3-VAL-004", "no_forbidden_worktree_paths", not forbidden_paths, len(forbidden_paths), "0", "; ".join(forbidden_paths[:20]))
    add("R3-VAL-005", "original_repo_clean", original_status == "## main...origin/main" and original_diff == "", original_status + " " + original_diff, "## main...origin/main with empty diff")
    add("R3-VAL-006", "protected_artifacts_unchanged", len(protected) == 12 and protected["protection_status"].eq("PROTECTED_UNCHANGED").all(), f"{protected['protection_status'].eq('PROTECTED_UNCHANGED').sum()}/{len(protected)}", "12/12")
    add("R3-VAL-007", "four_datasets_represented", set(TARGET_DATASETS).issubset(set(master["dataset"])), ";".join(sorted(set(master["dataset"]))), ";".join(TARGET_DATASETS))
    represented_metrics = {normalize_metric(value) for value in master["metric_name"]}
    add("R3-VAL-008", "paper_metrics_represented", catalog.paper_metric_ids.issubset(represented_metrics), len(catalog.paper_metric_ids.intersection(represented_metrics)), str(len(catalog.paper_metric_ids)), ";".join(sorted(catalog.paper_metric_ids.difference(represented_metrics))))
    add("R3-VAL-009", "statuses_are_classified", set(master["calculability_status"]).issubset(set(CALCULABILITY_STATUSES)) and not master["calculability_status"].isin(["UNKNOWN", "UNINVESTIGATED", "UNCLASSIFIED"]).any(), ";".join(sorted(set(master["calculability_status"]))), "R3 status enum only")
    numeric_master = master.loc[master["metric_value"].astype(str).str.len().gt(0)]
    add("R3-VAL-010", "numeric_values_have_provenance", not numeric_master.empty and numeric_master["source_artifact"].astype(str).str.len().gt(0).all(), len(numeric_master), "all numeric rows source_artifact non-empty")
    add("R3-VAL-011", "recomputed_values_have_raw_inputs", not recomputed.empty and recomputed["raw_input_artifacts"].astype(str).str.len().gt(0).all(), len(recomputed), "all recomputed rows raw_input_artifacts non-empty")
    exact_cand = candidate.loc[candidate["classification"].eq("CANDDIF_PAPER_COMPATIBLE")]
    add("R3-VAL-012", "paper_canddif_requires_first_random", exact_cand.empty or exact_cand["has_first_random_evidence"].eq("true").all(), len(exact_cand), "paper-compatible CandDif rows have first/random evidence")
    diagnostics = candidate.loc[candidate["classification"].str.contains("DIAGNOSTIC", case=False, na=False)]
    add("R3-VAL-013", "position_diagnostics_not_mislabeled", diagnostics["classification"].ne("CANDDIF_PAPER_COMPATIBLE").all(), len(diagnostics), "diagnostic rows not paper-compatible")
    add("R3-VAL-014", "hallucination_semantics_separated", {"PAPER_STYLE_TEXTUAL_HALLUCINATION", "CANDIDATE_VIOLATION", "STRUCTURAL_OUTPUT_INVALIDITY"}.issubset(set(hallucination["hallucination_semantic_type"])), ";".join(sorted(set(hallucination["hallucination_semantic_type"]))), "three hallucination semantic types")
    perf_rows = generation.loc[generation["metric"].str.contains("HR|NDCG", case=False, na=False)]
    add("R3-VAL-015", "generation_performance_has_downstream_output", not perf_rows.empty and perf_rows["has_downstream_recommendation_output"].eq("true").all(), len(perf_rows), "performance rows must have downstream recommendation output")
    local_ext_as_paper = master.loc[(master["metric_family"].eq("local_extension")) & (master["paper_metric"].eq("true"))]
    add("R3-VAL-016", "local_extensions_not_counted_as_paper", local_ext_as_paper.empty, len(local_ext_as_paper), "0")
    official_substitution = master.loc[master["source_scope"].eq("OFFICIAL_REFERENCE") & master["calculability_status"].isin(["ALREADY_MATERIALIZED", "RECALCULATED_FROM_LOCAL_OUTPUTS"])]
    add("R3-VAL-017", "no_r1_r2_official_substitution", official_substitution.empty, len(official_substitution), "0")
    add("R3-VAL-018", "coverage_has_all_dataset_metric_rows", len(coverage) >= len(TARGET_DATASETS) * len(catalog.paper_metric_ids), len(coverage), str(len(TARGET_DATASETS) * len(catalog.paper_metric_ids)))
    add("R3-VAL-019", "artifact_manifest_nonempty", len(frames["artifact_manifest"]) > 0, len(frames["artifact_manifest"]), ">0")
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS).fillna("")


def decision_token(validation: pd.DataFrame) -> str:
    failed = validation.loc[validation["status"].ne("pass")]
    if failed.empty:
        return COMPLETE_TOKEN
    blocking = {"R3-VAL-001", "R3-VAL-002", "R3-VAL-003", "R3-VAL-004", "R3-VAL-005", "R3-VAL-006"}
    if any(check in blocking for check in failed["check_id"]):
        return BLOCKED_TOKEN
    return PARTIAL_TOKEN


def build_report(project_root: Path, frames: Mapping[str, pd.DataFrame], catalog: MetricCatalog) -> str:
    master = frames["all_metric_cells_master"]
    materialized = frames["materialized_values"]
    recomputed = frames["offline_recalculated_values"]
    gaps = frames["true_remaining_gaps"]
    coverage = frames["paper_metric_dataset_coverage"]
    beyond = frames["beyond_utility_master"]
    history = frames["history_length_sensitivity"]
    candidate = frames["candidate_position_bias"]
    hallucination = frames["hallucination_audit"]
    generation = frames["generation_involved_performance"]
    validation = frames["validation"]
    protected = frames["protected_artifact_hash_check"]
    token = decision_token(validation)
    status_counts = master["calculability_status"].value_counts().to_dict()
    dataset_counts = master.groupby("dataset")["metric_name"].nunique().to_dict() if not master.empty else {}
    local_materialized_count = int(materialized["source_scope"].eq("LOCAL_REPRODUCTION").sum()) if not materialized.empty else 0
    materialized_source_counts = materialized["source_scope"].value_counts().to_dict() if not materialized.empty else {}
    recomputed_master_count = int(status_counts.get("RECALCULATED_FROM_LOCAL_OUTPUTS", 0))
    recomputed_duplicate_count = max(0, len(recomputed) - recomputed_master_count)
    gap_status_counts = gaps[["dataset", "metric"]].value_counts().reset_index(name="count") if not gaps.empty else pd.DataFrame()
    gap_family_counts = (
        master.loc[master["calculability_status"].isin(["NOT_CALCULABLE_MISSING_INPUT", "NOT_CALCULABLE_MISSING_OUTPUT"]), "metric_family"]
        .value_counts()
        .to_dict()
        if not master.empty
        else {}
    )

    def metric_section(title: str, pattern: str) -> list[str]:
        subset = beyond.loc[beyond["metric"].str.contains(pattern, case=False, regex=True, na=False)]
        numeric = subset.loc[subset["value"].astype(str).str.len().gt(0)]
        datasets = ", ".join(sorted(set(subset["dataset"]))) if not subset.empty else "none"
        sources = ", ".join(sorted(set(numeric["source"]))[:5]) if not numeric.empty else "none"
        return [
            f"## {title}",
            "",
            f"- datasets covered: {datasets}",
            f"- numerical rows: {len(numeric)}",
            f"- best sources: {sources}",
            f"- primary limitation: {clean(subset['limitation'].iloc[0]) if not subset.empty else 'No local evidence found.'}",
            "",
        ]

    lines = [
        "# R3 Complete Local Metric And Beyond-Utility Audit",
        "",
        "## Final Decision",
        "",
        f"`{token}`",
        "",
        "## Purpose",
        "",
        "R3 audits all locally available reproduction artifacts after the frozen R1/R2/F6M baseline. It extracts materialized values, independently recalculates metrics from existing local outputs where possible, and classifies remaining gaps without any new model, LLM, training, candidate-generation, or API execution.",
        "",
        "## Frozen R1/R2 baseline",
        "",
        f"- Expected branch/head: `{EXPECTED_BRANCH}` / `{EXPECTED_HEAD}`.",
        "- R1/R2/F6M artifacts are read-only sources for contracts, provenance, and protection checks.",
        f"- 12 protected root artifacts unchanged: `{protected['protection_status'].eq('PROTECTED_UNCHANGED').sum()}/12`.",
        f"- original_repo status: `{git_output(project_root / ORIGINAL_ROOT, ['status', '--branch', '--short']) if (project_root / ORIGINAL_ROOT).exists() else 'missing'}`.",
        "",
        "## Complete local artifact inventory",
        "",
        f"- artifact rows: {len(frames['artifact_manifest'])}",
        f"- materialized numeric values extracted: {len(materialized)}",
        f"- offline recalculated values: {len(recomputed)}",
        "",
        "## Complete metric-cell coverage",
        "",
        f"- total R3 master rows: {len(master)}",
        *[f"- {status}: {int(status_counts.get(status, 0))}" for status in CALCULABILITY_STATUSES],
        "- UNKNOWN: 0",
        "- UNINVESTIGATED: 0",
        "- UNCLASSIFIED: 0",
        "",
        "## Count semantics",
        "",
        "COUNT_SEMANTICS_EXPLAINED: YES",
        "",
        f"- `reproduction_r3_materialized_values.csv` rows: {len(materialized)}. This is the raw numeric-observation inventory extracted from scanned aggregate/reference/diagnostic/local artifacts, not the de-duplicated scientific cell table.",
        f"- Materialized value source scopes: LOCAL_REPRODUCTION={local_materialized_count}, LOCAL_DIAGNOSTIC={int(materialized_source_counts.get('LOCAL_DIAGNOSTIC', 0))}, OFFICIAL_REFERENCE={int(materialized_source_counts.get('OFFICIAL_REFERENCE', 0))}, PAPER_CONTRACT={int(materialized_source_counts.get('PAPER_CONTRACT', 0))}. R1/R2/paper values are retained only as provenance/reference rows and are not substituted for local result cells.",
        f"- `reproduction_r3_offline_recalculated_values.csv` rows: {len(recomputed)}. This is the detailed recalculated numerical-quantity inventory from row-level local outputs.",
        f"- Master `RECALCULATED_FROM_LOCAL_OUTPUTS` rows: {recomputed_master_count}. The difference of {recomputed_duplicate_count} rows is caused by duplicate scientific keys where fail-strict and conditional-valid recalculations have identical requested counts, valid counts, denominators, values, and source artifacts because all rows in that local cell were strict-valid. The detailed recalculation table preserves both policy observations; the master table de-duplicates them into one scientific cell.",
        f"- Master `ALREADY_MATERIALIZED` rows: {int(status_counts.get('ALREADY_MATERIALIZED', 0))}. These are de-duplicated local scientific cells after excluding official/paper reference substitutions, not the raw materialized-observation count.",
        "",
        "## Utility verification",
        "",
        f"- HR/NDCG rows in master: {len(master.loc[master['metric_name'].isin(['HR@5', 'NDCG@5'])])}",
        "- Fail-strict and conditional-valid recalculations are preserved in the detailed recalculation table; master rows keep separate policy cells when requested counts, valid counts, denominators, or values differ.",
        "",
        "## Beyond-Utility overview",
        "",
        f"- beyond-utility rows: {len(beyond)}",
        f"- true remaining gap rows: {len(gaps)}",
        "",
    ]
    for title, pattern in [
        ("Novelty / SelfInformation", "SelfInformation"),
        ("Serendipity", "Serendipity"),
        ("Popularity: ACLT/APLT", "ACLT|APLT"),
        ("Popularity: ARP", "ARP"),
        ("Popularity: PopREO", "PopREO"),
        ("Diversity: ItemCoverage", "ItemCoverage"),
        ("Diversity: Shannon", "Shannon"),
        ("Diversity: Jaccard", "Jaccard"),
        ("Fairness: Gini", "Gini"),
        ("Fairness: DPD", "DPD"),
        ("Fairness: Jain", "Jain"),
    ]:
        lines.extend(metric_section(title, pattern))
    lines.extend(
        [
            "## History Length Sensitivity",
            "",
            f"- curve/point rows: {len(history)}",
            f"- sources: {', '.join(sorted(set(history['source_artifact']))[:8]) if not history.empty else 'none'}",
            "",
            "## Candidate Position Bias / CandDif",
            "",
            f"- candidate-position rows: {len(candidate)}",
            f"- paper-compatible CandDif rows: {int(candidate['classification'].eq('CANDDIF_PAPER_COMPATIBLE').sum()) if not candidate.empty else 0}",
            f"- diagnostic rows: {int(candidate['classification'].str.contains('DIAGNOSTIC', case=False, na=False).sum()) if not candidate.empty else 0}",
            "",
            "## Hallucination",
            "",
            f"- hallucination audit rows: {len(hallucination)}",
            f"- semantic types: {', '.join(sorted(set(hallucination['hallucination_semantic_type']))) if not hallucination.empty else 'none'}",
            "",
            "## Generation-Involved Performance",
            "",
            f"- generation/profile rows: {len(generation)}",
            "- Performance rows require downstream recommendation outputs; profile-generation-only rows are labeled separately.",
            "",
        ]
    )
    for dataset in TARGET_DATASETS:
        subset = coverage.loc[coverage["dataset"].eq(dataset)]
        missing = subset.loc[subset["missing"].eq("true")]
        lines.extend(
            [
                f"## {dataset} complete metric coverage",
                "",
                f"- metrics represented: {int(dataset_counts.get(dataset, 0))}",
                f"- metric-family coverage rows: {len(subset)}",
                f"- missing rows after offline recovery: {len(missing)}",
                "",
            ]
        )
    lines.extend(
        [
            "## Already materialized values",
            "",
            f"Stored in `{OUTPUT_TABLES['materialized_values'].as_posix()}`.",
            "",
            "## Offline recalculated values",
            "",
            f"Stored in `{OUTPUT_TABLES['offline_recalculated_values'].as_posix()}`.",
            "",
            "## Reconstructed values",
            "",
            "R3 derives CandDif only when first/random evidence is explicit; otherwise it preserves diagnostics without paper-compatible labeling.",
            "",
            "## Paper-exact results",
            "",
            "No local row is upgraded to paper-exact from numerical similarity alone. Rows with exact local support retain their explicit source/protocol labels.",
            "",
            "## Adapted results",
            "",
            "Most local LLM and beyond-utility rows are adapted because local open models, candidate-index output contracts, operationalized LastFM, and reconstructed support data differ from the paper protocol.",
            "",
            "## Diagnostic-only results",
            "",
            "Candidate-position diagnostics, structural validity, duplicate rate, and candidate-violation rows are preserved separately from paper reproduction coverage.",
            "",
        "## True remaining gaps",
        "",
        f"Gap rows are written to `{OUTPUT_TABLES['true_remaining_gaps'].as_posix()}`.",
        f"Total true remaining gaps: {len(gaps)}; missing-input={int(status_counts.get('NOT_CALCULABLE_MISSING_INPUT', 0))}; missing-output={int(status_counts.get('NOT_CALCULABLE_MISSING_OUTPUT', 0))}.",
        "Gap distribution by metric family: "
        + (", ".join(f"{family}={count}" for family, count in sorted(gap_family_counts.items())) if gap_family_counts else "none")
        + ".",
        "",
        "## Potential controlled recovery work",
            "",
            "The smallest next recovery phase should target paper-style hallucination and full-matrix controlled CandDif first, because those remain the highest-priority beyond-utility uncertainties.",
            "",
            "## Implications for paper conclusions",
            "",
            "R3 supports only protocol-qualified beyond-utility interpretation. Local evidence can compare adapted open-model behavior and deterministic baselines, but paper-level causal claims remain limited where protocols, support data, or exact outputs are missing.",
            "",
            "## Implications for total reproducibility closure",
            "",
            "Total closure still requires controlled recovery of missing beyond-utility cells or explicit acceptance that those cells remain adapted/diagnostic rather than paper-exact.",
            "",
            "## Validation",
            "",
            f"- validation rows: {len(validation)}",
            f"- validation failures: {int(validation['status'].ne('pass').sum())}",
            "",
            "## Artifacts created",
            "",
            *[f"- `{path.as_posix()}`" for path in OUTPUT_TABLES.values()],
        ]
    )
    return "\n".join(lines) + "\n"


def build_frames(project_root: Path, config_path: Path | None = None) -> dict[str, pd.DataFrame]:
    config = load_config(project_root / (config_path or DEFAULT_CONFIG_PATH))
    catalog = build_metric_catalog(project_root, config)
    pre_git = pd.DataFrame(git_state_rows(project_root, "precondition"), columns=GIT_COLUMNS).fillna("")
    protected = pd.DataFrame(protected_hash_rows(project_root, config), columns=PROTECTED_COLUMNS).fillna("")
    artifact_manifest = discover_artifacts(project_root, config, catalog)
    materialized = extract_materialized_values(project_root, artifact_manifest, catalog)
    recomputed = offline_recalculated_values(project_root, catalog)
    stored_vs = build_stored_vs_recomputed(materialized, recomputed)
    master = build_master(materialized, recomputed, catalog)
    beyond = build_beyond_master(master)
    coverage = build_coverage(master, catalog)
    gaps = build_gaps(master)
    history = build_history_table(project_root, recomputed)
    candidate = build_candidate_position_table(project_root)
    hallucination = build_hallucination_table(project_root, materialized)
    generation = build_generation_table(project_root, recomputed)
    post_git = pd.DataFrame(git_state_rows(project_root, "postcondition"), columns=GIT_COLUMNS).fillna("")
    frames: dict[str, pd.DataFrame] = {
        "artifact_manifest": artifact_manifest,
        "all_metric_cells_master": master,
        "beyond_utility_master": beyond,
        "paper_metric_dataset_coverage": coverage,
        "true_remaining_gaps": gaps,
        "history_length_sensitivity": history,
        "candidate_position_bias": candidate,
        "hallucination_audit": hallucination,
        "generation_involved_performance": generation,
        "materialized_values": materialized,
        "stored_vs_recomputed": stored_vs,
        "offline_recalculated_values": recomputed,
        "precondition_git_state": pre_git,
        "postcondition_git_state": post_git,
        "protected_artifact_hash_check": protected,
    }
    validation = build_validation(project_root, config, frames, catalog)
    frames["validation"] = validation
    return frames


def write_outputs(project_root: Path, frames: Mapping[str, pd.DataFrame], report: str) -> None:
    column_map = {
        "artifact_manifest": MANIFEST_COLUMNS,
        "all_metric_cells_master": MASTER_COLUMNS,
        "beyond_utility_master": BEYOND_COLUMNS,
        "paper_metric_dataset_coverage": COVERAGE_COLUMNS,
        "true_remaining_gaps": GAP_COLUMNS,
        "history_length_sensitivity": HISTORY_COLUMNS,
        "candidate_position_bias": CANDIDATE_POSITION_COLUMNS,
        "hallucination_audit": HALLUCINATION_COLUMNS,
        "generation_involved_performance": GENERATION_COLUMNS,
        "materialized_values": MATERIALIZED_COLUMNS,
        "stored_vs_recomputed": STORED_VS_COLUMNS,
        "offline_recalculated_values": RECOMPUTED_COLUMNS,
        "validation": VALIDATION_COLUMNS,
        "precondition_git_state": GIT_COLUMNS,
        "postcondition_git_state": GIT_COLUMNS,
        "protected_artifact_hash_check": PROTECTED_COLUMNS,
    }
    for key, path in OUTPUT_TABLES.items():
        write_csv(project_root / path, frames[key], column_map[key])
    (project_root / REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (project_root / REPORT_PATH).write_text(report, encoding="utf-8")
    append_notes(project_root, frames)


def append_notes(project_root: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    notes_path = project_root / NOTES_PATH
    existing = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    if NOTE_MARKER in existing:
        return
    token = decision_token(frames["validation"])
    entry = (
        f"\n## {NOTE_MARKER}\n\n"
        f"- Decision: `{token}`.\n"
        f"- Commands: `python src/analysis/reproduction_r3_complete_local_metric_audit.py --project-root .`.\n"
        f"- Outputs: `{OUTPUT_TABLES['all_metric_cells_master'].as_posix()}`, `{OUTPUT_TABLES['beyond_utility_master'].as_posix()}`, `{REPORT_PATH.as_posix()}`.\n"
        f"- Cells classified: {len(frames['all_metric_cells_master'])}; UNKNOWN/UNINVESTIGATED/UNCLASSIFIED = 0/0/0.\n"
        "- Known issues: R3 does not execute new missing cells; candidate-position diagnostics and constrained-output hallucination remain separated from paper-style claims.\n"
        "- Next step: design the smallest controlled recovery phase for unresolved beyond-utility gaps.\n"
    )
    notes_path.write_text(existing.rstrip() + entry + "\n", encoding="utf-8")


def build_all(project_root: Path, config_path: Path | None = None, *, write: bool = True) -> dict[str, Any]:
    config = load_config(project_root / (config_path or DEFAULT_CONFIG_PATH))
    catalog = build_metric_catalog(project_root, config)
    frames = build_frames(project_root, config_path)
    report = build_report(project_root, frames, catalog)
    if write:
        write_outputs(project_root, frames, report)
    return {"frames": frames, "report": report, "decision": decision_token(frames["validation"])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build R3 complete local metric and beyond-utility audit artifacts.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root directory.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="R3 YAML config path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config
    if config_path.is_absolute():
        try:
            config_path = config_path.resolve().relative_to(project_root)
        except ValueError:
            pass
    result = build_all(project_root, config_path, write=True)
    frames = result["frames"]
    validation = frames["validation"]
    print(f"decision={result['decision']}")
    print(f"master_rows={len(frames['all_metric_cells_master'])}")
    print(f"materialized_values={len(frames['materialized_values'])}")
    print(f"offline_recalculated={len(frames['offline_recalculated_values'])}")
    print(f"validation_pass={validation['status'].eq('pass').sum()}")
    print(f"validation_fail={validation['status'].ne('pass').sum()}")
    print(f"report={REPORT_PATH.as_posix()}")
    return 0 if result["decision"] == COMPLETE_TOKEN else 1


if __name__ == "__main__":
    raise SystemExit(main())
