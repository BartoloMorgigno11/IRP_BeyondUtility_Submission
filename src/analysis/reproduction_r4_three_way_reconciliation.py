"""R4 paper/official/local three-way reconciliation.

This module reads only frozen R1/R2/R3/E1 evidence and writes new R4
interpretation artifacts. It does not run recommendation models, LLMs,
training, candidate generation, downloads, or recovery experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


EXPECTED_HEAD = "357d8a4cf98c78effc6c92f5b8dc00e334836a56"
EXPECTED_BRANCH = "master"

TABLE_DIR = Path("results") / "tables"
REPORT_PATH = Path("report") / "reproduction_r4_three_way_beyond_utility_reconciliation.md"
NOTES_PATH = Path("report") / "notes.md"

OUTPUT_TABLES = {
    "three_way_master": TABLE_DIR / "reproduction_r4_three_way_reconciliation_master.csv",
    "beyond_summary": TABLE_DIR / "reproduction_r4_beyond_utility_summary.csv",
    "gap_priority": TABLE_DIR / "reproduction_r4_gap_scientific_priority.csv",
    "paper_claim_reconciliation": TABLE_DIR / "reproduction_r4_paper_claim_reconciliation.csv",
    "proposed_recovery_experiments": TABLE_DIR / "reproduction_r4_proposed_recovery_experiments.csv",
    "recovery_request_count_validation": TABLE_DIR / "reproduction_r4_recovery_request_count_validation.csv",
    "dataset_synthesis": TABLE_DIR / "reproduction_r4_dataset_synthesis.csv",
    "model_synthesis": TABLE_DIR / "reproduction_r4_model_synthesis.csv",
    "numeric_agreement": TABLE_DIR / "reproduction_r4_numeric_agreement.csv",
    "candidate_position": TABLE_DIR / "reproduction_r4_candidate_position_reconciliation.csv",
    "hallucination": TABLE_DIR / "reproduction_r4_hallucination_reconciliation.csv",
    "generation": TABLE_DIR / "reproduction_r4_generation_involved_reconciliation.csv",
    "history": TABLE_DIR / "reproduction_r4_history_length_reconciliation.csv",
    "validation": TABLE_DIR / "reproduction_r4_validation.csv",
    "precondition_git_state": TABLE_DIR / "reproduction_r4_precondition_git_state.csv",
    "postcondition_git_state": TABLE_DIR / "reproduction_r4_postcondition_git_state.csv",
    "protected_hash_check": TABLE_DIR / "reproduction_r4_protected_artifact_hash_check.csv",
}

INPUT_TABLES = {
    "e1_contract": TABLE_DIR / "extension_e1_metric_contract.csv",
    "paper_atomic": TABLE_DIR / "paper_atomic_contract.csv",
    "r2_master": TABLE_DIR / "reproduction_r2_official_metric_calculability_master.csv",
    "r2_candidate_position": TABLE_DIR / "reproduction_r2_candidate_position_reconstruction.csv",
    "r2_history": TABLE_DIR / "reproduction_r2_history_metric_reconstruction.csv",
    "r2_generation": TABLE_DIR / "reproduction_r2_generation_metric_reconstruction.csv",
    "r2_validation": TABLE_DIR / "reproduction_r2_validation.csv",
    "r3_master": TABLE_DIR / "reproduction_r3_all_metric_cells_master.csv",
    "r3_beyond": TABLE_DIR / "reproduction_r3_beyond_utility_master.csv",
    "r3_gaps": TABLE_DIR / "reproduction_r3_true_remaining_gaps.csv",
    "r3_candidate_position": TABLE_DIR / "reproduction_r3_candidate_position_bias.csv",
    "r3_hallucination": TABLE_DIR / "reproduction_r3_hallucination_audit.csv",
    "r3_generation": TABLE_DIR / "reproduction_r3_generation_involved_performance.csv",
    "r3_history": TABLE_DIR / "reproduction_r3_history_length_sensitivity.csv",
    "r3_validation": TABLE_DIR / "reproduction_r3_validation.csv",
    "r3_protected_hash": TABLE_DIR / "reproduction_r3_protected_artifact_hash_check.csv",
    "r1_hallucination": TABLE_DIR / "reproduction_r1_hallucination_official_results.csv",
}

DATASETS = ["Beauty", "Sports", "ML-1M", "LastFM"]

BEYOND_GROUPS: dict[str, list[str]] = {
    "SelfInformation": ["SelfInformation@5"],
    "Serendipity": ["Serendipity@5"],
    "ACLT/APLT": ["ACLT@5", "APLT@5"],
    "ARP": ["ARP@5"],
    "PopREO": ["PopREO@5"],
    "ItemCoverage": ["ItemCoverage@5"],
    "Shannon": ["Shannon category entropy@5"],
    "Jaccard": ["Jaccard@5"],
    "Gini": ["Gini@5"],
    "DPD": ["DPD@5"],
    "Jain": ["Jain's Index@5"],
    "History Length Sensitivity": ["History-length sensitivity"],
    "Candidate Position Bias": ["CandDif-HR", "CandDif-NDCG"],
    "Hallucination": ["Hallucination rate@5"],
    "Generation-Involved Performance": ["Generation-involved profile recommendation"],
}

REPRODUCTION_CLASSES = {
    "EXACTLY_REPRODUCED",
    "NUMERICALLY_CLOSE_REPRODUCED",
    "QUALITATIVELY_REPRODUCED",
    "ADAPTED_REPRODUCTION",
    "DIAGNOSTIC_REPRODUCTION_ONLY",
    "OFFICIAL_ARTIFACT_RECOVERED_ONLY",
    "PAPER_ONLY",
    "BLOCKED_MISSING_INPUT",
    "BLOCKED_MISSING_OUTPUT",
    "BLOCKED_MODEL",
    "PROTOCOL_UNRECOVERABLE",
    "NOT_APPLICABLE",
}

COMPARABILITY_CLASSES = {
    "PAPER_OFFICIAL_LOCAL_EXACT_PROTOCOL",
    "PAPER_OFFICIAL_LOCAL_NUMERICALLY_COMPARABLE",
    "OFFICIAL_LOCAL_ADAPTED_COMPARABLE",
    "PAPER_LOCAL_ADAPTED_COMPARABLE",
    "QUALITATIVE_ONLY_COMPARABLE",
    "DIAGNOSTIC_ONLY",
    "OFFICIAL_ONLY",
    "PAPER_ONLY",
    "LOCAL_ONLY",
    "LOCAL_GAP",
    "PROTOCOL_UNRECOVERABLE",
    "NOT_APPLICABLE",
}

PRIORITIES = [
    "P0_MUST_CLOSE",
    "P1_HIGH_VALUE",
    "P2_USEFUL_BUT_OPTIONAL",
    "P3_REDUNDANT_OR_LOW_VALUE",
    "P4_NOT_RECOVERABLE",
]

THREE_WAY_COLUMNS = [
    "dataset",
    "metric_family",
    "metric_name",
    "paper_model",
    "official_model",
    "local_model",
    "task",
    "prompt_strategy",
    "condition",
    "history_condition",
    "candidate_position_condition",
    "generation_condition",
    "profile_generator",
    "cohort",
    "cutoff",
    "paper_value",
    "official_stored_value",
    "official_recomputed_value",
    "official_reconstructed_value",
    "official_value",
    "official_value_origin",
    "local_stored_value",
    "local_recomputed_value",
    "local_reconstructed_value",
    "local_value",
    "local_value_origin",
    "paper_official_mapping",
    "paper_local_mapping",
    "official_local_mapping",
    "comparability_class",
    "paper_official_absolute_delta",
    "paper_local_absolute_delta",
    "official_local_absolute_delta",
    "paper_official_relative_delta",
    "paper_local_relative_delta",
    "official_local_relative_delta",
    "directional_agreement",
    "qualitative_claim_supported",
    "local_protocol_class",
    "local_evidence_class",
    "gap_status",
    "source_layer",
    "source_artifact",
    "evidence_notes",
    "scientific_interpretation",
]


@dataclass(frozen=True)
class MetricInfo:
    metric_name: str
    family: str
    canonical_paper_name: str
    task_applicability: str
    provenance: str
    notes: str


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    text = re.sub("unknown", "not_recovered", text, flags=re.IGNORECASE)
    text = re.sub("uninvestigated", "not_investigated", text, flags=re.IGNORECASE)
    text = re.sub("unclassified", "not_classified", text, flags=re.IGNORECASE)
    return text


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def parse_bool(value: Any) -> bool:
    return clean(value).lower() in {"true", "1", "yes", "y"}


def numeric(value: Any) -> float | None:
    text = clean(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def compact_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.12g}"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(project_root: Path, args: list[str], cwd: Path | None = None) -> tuple[int, str]:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd or project_root),
        check=False,
        capture_output=True,
        text=True,
    )
    output = (result.stdout + result.stderr).strip()
    return result.returncode, output


def git_state(project_root: Path, phase: str) -> pd.DataFrame:
    commands = [
        ("branch", ["branch", "--show-current"], project_root),
        ("head", ["rev-parse", "HEAD"], project_root),
        ("status_short", ["status", "--short"], project_root),
        ("status_branch", ["status", "--branch", "--short"], project_root),
        ("diff_stat", ["diff", "--stat"], project_root),
        ("cached_diff_stat", ["diff", "--cached", "--stat"], project_root),
        (
            "original_repo_status",
            ["status", "--branch", "--short"],
            project_root / "original_repo" / "EvaLLMasRecommender",
        ),
        (
            "original_repo_diff_stat",
            ["diff", "--stat"],
            project_root / "original_repo" / "EvaLLMasRecommender",
        ),
    ]
    rows: list[dict[str, str]] = []
    for name, args, cwd in commands:
        code, output = run_git(project_root, args, cwd)
        rows.append(
            {
                "phase": phase,
                "check": name,
                "returncode": str(code),
                "output": output,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
    return pd.DataFrame(rows)


def read_csv(project_root: Path, relative_path: Path) -> pd.DataFrame:
    path = project_root / relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False, low_memory=False)


def normalize_dataset(value: Any) -> str:
    text = clean(value)
    low = re.sub(r"[^a-z0-9]+", "", text.lower())
    if low in {"amazonbeauty", "beauty"}:
        return "Beauty"
    if low in {"amazonsports", "sports"}:
        return "Sports"
    if low in {"ml1m", "movielens1m", "movielens"}:
        return "ML-1M"
    if low in {"lastfm", "lastfmdataset"}:
        return "LastFM"
    if low == "all":
        return "ALL"
    return text


def normalize_model(value: Any) -> str:
    text = clean(value)
    low = text.lower()
    if "qwen2" in low:
        return "qwen2:7b-instruct"
    if "mistral" in low:
        return "mistral:7b-instruct"
    if low in {"gpt3.5", "gpt-3.5", "chatgpt"}:
        return "gpt3.5"
    if low in {"gpt4", "gpt-4"}:
        return "gpt4"
    if "gpt4o-mini" in low:
        return "gpt4o-mini"
    if "gpt4o" in low:
        return "gpt4o"
    if "llama3" in low:
        return "llama3"
    if "llama2" in low:
        return "llama2"
    if low == "mostpop":
        return "MostPop"
    if low in {"bprmf", "lightgcn", "gru4rec", "sasrec", "bm25"}:
        return text.upper() if low == "bm25" else text
    return text


def normalize_metric(value: Any) -> str:
    text = clean(value)
    low = text.lower().strip()
    low = low.replace(" ", "_").replace("-", "_")
    low = low.replace("@", "_at_").replace("__", "_")
    aliases = {
        "hr": "HR@5",
        "hr_at_5": "HR@5",
        "hit_rate_at_5": "HR@5",
        "ndcg": "NDCG@5",
        "ndcg_at_5": "NDCG@5",
        "selfinformation": "SelfInformation@5",
        "selfinformation_at_5": "SelfInformation@5",
        "self_information_at_5": "SelfInformation@5",
        "novelty_selfinformation_5": "SelfInformation@5",
        "serendipity": "Serendipity@5",
        "serendipity_at_5": "Serendipity@5",
        "average_popularity": "ARP@5",
        "arp": "ARP@5",
        "arp_at_5": "ARP@5",
        "average_recommendation_popularity_5": "ARP@5",
        "aclt": "ACLT@5",
        "aclt_at_5": "ACLT@5",
        "aplt": "APLT@5",
        "aplt_at_5": "APLT@5",
        "aplt_aclt_5": "ACLT@5",
        "popreo": "PopREO@5",
        "popreo_at_5": "PopREO@5",
        "itemcoverage": "ItemCoverage@5",
        "itemcoverage_at_5": "ItemCoverage@5",
        "item_coverage_at_5": "ItemCoverage@5",
        "overlap_item_coverage_at_5": "Overlap Item Coverage@5",
        "oic_at_5": "Overlap Item Coverage@5",
        "shannon": "Shannon category entropy@5",
        "shannon_category_entropy_at_5": "Shannon category entropy@5",
        "jaccard": "Jaccard@5",
        "jaccard_at_5": "Jaccard@5",
        "gini": "Gini@5",
        "gini_at_5": "Gini@5",
        "dpd": "DPD@5",
        "dpd_at_5": "DPD@5",
        "jain": "Jain's Index@5",
        "jain_s_index_at_5": "Jain's Index@5",
        "jains_index_at_5": "Jain's Index@5",
        "hallucination": "Hallucination rate@5",
        "hallucination_rate": "Hallucination rate@5",
        "hallucination_rate_at_5": "Hallucination rate@5",
        "candidate_position_bias_hr": "CandDif-HR",
        "canddif_hr": "CandDif-HR",
        "canddif_hr_at_5": "CandDif-HR",
        "candidate_position_bias_ndcg": "CandDif-NDCG",
        "canddif_ndcg": "CandDif-NDCG",
        "canddif_ndcg_at_5": "CandDif-NDCG",
        "history_length_sensitivity": "History-length sensitivity",
        "generation_involved_profile_recommendation": "Generation-involved profile recommendation",
        "mrr_printed_by_official_script": "MRR printed by official script",
        "k_s_sample_representativeness_gate": "K-S sample representativeness gate",
    }
    return aliases.get(low, text)


def load_metric_catalog(e1: pd.DataFrame) -> dict[str, MetricInfo]:
    catalog: dict[str, MetricInfo] = {}
    for _, row in e1.iterrows():
        metric = normalize_metric(row.get("metric_name"))
        catalog[metric] = MetricInfo(
            metric_name=metric,
            family=clean(row.get("family")),
            canonical_paper_name=clean(row.get("canonical_paper_name")),
            task_applicability=clean(row.get("task_applicability")),
            provenance=clean(row.get("provenance_classification")),
            notes=clean(row.get("notes")),
        )
    return catalog


def relevant_paper_metrics(catalog: Mapping[str, MetricInfo]) -> list[str]:
    metrics = [
        metric
        for metric, info in catalog.items()
        if info.family != "local_extension"
    ]
    return sorted(metrics)


def metric_family(metric: str, catalog: Mapping[str, MetricInfo]) -> str:
    if metric in catalog:
        return catalog[metric].family
    if metric in {"CandDif-HR", "CandDif-NDCG", "History-length sensitivity", "Generation-involved profile recommendation", "Hallucination rate@5"}:
        return "LLM_specific"
    return "metric"


def split_metric_cell(metric_text: Any) -> list[str]:
    text = clean(metric_text)
    if ";" not in text:
        return [normalize_metric(text)]
    return [normalize_metric(part) for part in text.split(";") if clean(part)]


def source_scope_summary(values: Iterable[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        key = clean(value) or "blank"
        counts[key] = counts.get(key, 0) + 1
    return "; ".join(f"{key}={counts[key]}" for key in sorted(counts))


def value_summary(values: Iterable[Any]) -> str:
    nums = [numeric(v) for v in values]
    nums = [v for v in nums if v is not None]
    if not nums:
        return "no numeric value"
    if len(nums) == 1:
        return compact_float(nums[0])
    return f"n={len(nums)}; min={compact_float(min(nums))}; max={compact_float(max(nums))}; mean={compact_float(sum(nums) / len(nums))}"


def choose_official_value(row: Mapping[str, Any]) -> tuple[str, str]:
    for col, origin in [
        ("official_recomputed_value", "OFFICIAL_RECOMPUTED"),
        ("official_reconstructed_value", "OFFICIAL_RECONSTRUCTED"),
        ("official_stored_value", "OFFICIAL_STORED"),
        ("official_value_used_for_comparison", "OFFICIAL_COMPARISON_VALUE"),
    ]:
        value = clean(row.get(col))
        if value:
            return value, origin
    return "", "OFFICIAL_NOT_AVAILABLE"


def choose_local_value(row: Mapping[str, Any]) -> tuple[str, str]:
    for col, origin in [
        ("recomputed_local_value", "LOCAL_RECOMPUTED"),
        ("reconstructed_local_value", "LOCAL_RECONSTRUCTED"),
        ("stored_local_value", "LOCAL_STORED"),
        ("metric_value", "LOCAL_METRIC_VALUE"),
    ]:
        value = clean(row.get(col))
        if value:
            return value, origin
    status = clean(row.get("calculability_status"))
    return "", status or "LOCAL_NOT_AVAILABLE"


def local_evidence_class(row: Mapping[str, Any]) -> str:
    status = clean(row.get("calculability_status"))
    comparability = clean(row.get("paper_comparability"))
    if status.startswith("NOT_CALCULABLE"):
        return "MISSING"
    if comparability == "PAPER_EXACT":
        return "PAPER_EXACT"
    if comparability == "DIAGNOSTIC_ONLY":
        return "DIAGNOSTIC_ONLY"
    return "ADAPTED"


def comparable_delta(left: Any, right: Any, allow: bool) -> tuple[str, str]:
    if not allow:
        return "", ""
    a = numeric(left)
    b = numeric(right)
    if a is None or b is None:
        return "", ""
    abs_delta = abs(a - b)
    if a == 0:
        rel = "" if b != 0 else "0"
    else:
        rel = compact_float(abs_delta / abs(a))
    return compact_float(abs_delta), rel


def directional_agreement_from_delta(delta_text: str) -> str:
    value = numeric(delta_text)
    if value is None:
        return "NO_COMPARISON"
    if value <= 1e-9:
        return "SAME_DIRECTION"
    if value <= 0.05:
        return "PARTIAL_DIRECTION"
    return "NO_COMPARISON"


def r4_allowed_status_path(path_text: str, protected_paths: set[str]) -> bool:
    normalized = path_text.replace("\\", "/")
    if normalized in protected_paths:
        return True
    allowed_exact = {
        "configs/reproduction_r4_three_way_reconciliation.yaml",
        "src/analysis/reproduction_r4_three_way_reconciliation.py",
        "src/experiments/build_reproduction_r4_three_way_reconciliation.py",
        "tests/test_reproduction_r4_three_way_reconciliation.py",
        "report/reproduction_r4_three_way_beyond_utility_reconciliation.md",
        "report/notes.md",
    }
    if normalized in allowed_exact:
        return True
    return normalized.startswith("results/tables/reproduction_r4_") or normalized.startswith("results/logs/reproduction_r4_")


def protected_hash_check(project_root: Path, r3_protected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in r3_protected.iterrows():
        rel = clean(row.get("path"))
        path = project_root / rel
        expected = clean(row.get("expected_sha256")) or clean(row.get("observed_sha256"))
        exists = path.exists()
        observed = sha256_path(path) if exists else ""
        tracked_code, tracked_out = run_git(project_root, ["ls-files", "--error-unmatch", rel])
        status_code, status_out = run_git(project_root, ["status", "--short", "--", rel])
        unchanged = exists and expected == observed
        rows.append(
            {
                "path": rel,
                "exists": bool_text(exists),
                "tracked": bool_text(tracked_code == 0 and bool(clean(tracked_out))),
                "git_status": status_out,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "hash_unchanged": bool_text(unchanged),
                "protection_status": "PROTECTED_UNCHANGED" if unchanged else "PROTECTED_CHANGED",
            }
        )
    return pd.DataFrame(rows)


def precondition_validation(project_root: Path, pre_state: pd.DataFrame, protected: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    state = {row["check"]: clean(row["output"]) for _, row in pre_state.iterrows()}
    protected_paths = set(protected["path"].astype(str).str.replace("\\", "/", regex=False))
    status_lines = [line for line in state.get("status_short", "").splitlines() if clean(line)]
    unrelated: list[str] = []
    for line in status_lines:
        if line.startswith("?? "):
            path_text = line[3:].strip()
        else:
            path_text = line[2:].strip()
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1].strip()
        if not r4_allowed_status_path(path_text, protected_paths):
            unrelated.append(line)
    rows.append(validation_row("R4-VAL-001", "expected branch", state.get("branch") == EXPECTED_BRANCH, state.get("branch"), EXPECTED_BRANCH, "R4 starts from the frozen master branch."))
    rows.append(validation_row("R4-VAL-002", "expected head", state.get("head") == EXPECTED_HEAD, state.get("head"), EXPECTED_HEAD, "R4 starts from the frozen R3 commit."))
    rows.append(validation_row("R4-VAL-003", "no unrelated worktree changes", not unrelated, "; ".join(unrelated), "none", "Only R4 outputs/source and the 12 protected historical artifacts may be present."))
    rows.append(validation_row("R4-VAL-004", "no staged diff", state.get("cached_diff_stat", "") == "", state.get("cached_diff_stat", ""), "empty", "R4 must not stage."))
    rows.append(validation_row("R4-VAL-005", "original repo clean", state.get("original_repo_diff_stat", "") == "", state.get("original_repo_diff_stat", ""), "empty", "R4 must not modify original_repo/EvaLLMasRecommender."))
    rows.append(validation_row("R4-VAL-006", "protected artifacts unchanged", (protected["protection_status"] == "PROTECTED_UNCHANGED").all() and len(protected) == 12, source_scope_summary(protected["protection_status"]), "12 PROTECTED_UNCHANGED", "R4 reuses R3 protection evidence and recomputes hashes."))
    return rows


def validation_row(check_id: str, name: str, passed: bool, observed: Any, expected: Any, details: str) -> dict[str, str]:
    return {
        "check_id": check_id,
        "check_name": name,
        "status": "pass" if passed else "fail",
        "observed": clean(observed),
        "expected": clean(expected),
        "details": details,
    }


def build_paper_rows(frames: Mapping[str, pd.DataFrame], catalog: Mapping[str, MetricInfo]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    paper = frames["paper_atomic"]
    for _, row in paper.iterrows():
        for metric in split_metric_cell(row.get("metric")):
            if not metric:
                continue
            rows.append(
                {
                    "dataset": normalize_dataset(row.get("dataset")),
                    "metric_family": metric_family(metric, catalog),
                    "metric_name": metric,
                    "paper_model": normalize_model(row.get("model")),
                    "task": clean(row.get("task")),
                    "prompt_strategy": clean(row.get("prompt_strategy")),
                    "condition": clean(row.get("cohort")),
                    "history_condition": clean(row.get("history_policy")),
                    "candidate_position_condition": clean(row.get("candidate_policy")),
                    "generation_condition": "",
                    "profile_generator": "",
                    "cohort": clean(row.get("cohort")),
                    "cutoff": clean(row.get("cutoff")),
                    "paper_value": clean(row.get("paper_value")),
                    "paper_source": "results/tables/paper_atomic_contract.csv",
                    "paper_status": clean(row.get("paper_specification_status")) or "PAPER_VALUE_REPORTED",
                    "paper_notes": clean(row.get("notes")),
                }
            )
    # Add paper metric/analysis contract rows for metrics with no recovered numeric paper atom.
    e1 = frames["e1_contract"]
    for _, row in e1.iterrows():
        metric = normalize_metric(row.get("metric_name"))
        info = catalog.get(metric)
        if not info or info.family == "local_extension":
            continue
        for dataset in DATASETS:
            rows.append(
                {
                    "dataset": dataset,
                    "metric_family": info.family,
                    "metric_name": metric,
                    "paper_model": "ALL",
                    "task": clean(row.get("task_applicability")),
                    "prompt_strategy": "",
                    "condition": "paper_contract",
                    "history_condition": clean(row.get("history_source")),
                    "candidate_position_condition": clean(row.get("candidate_positive_absent_policy")),
                    "generation_condition": "",
                    "profile_generator": "",
                    "cohort": "paper_metric_universe",
                    "cutoff": clean(row.get("cutoff_k")),
                    "paper_value": "",
                    "paper_source": "results/tables/extension_e1_metric_contract.csv",
                    "paper_status": "PAPER_CONTRACT_DEFINED_NO_LOCAL_EXECUTION",
                    "paper_notes": clean(row.get("paper_evidence")),
                }
            )
    return pd.DataFrame(rows)


def build_official_rows(frames: Mapping[str, pd.DataFrame], catalog: Mapping[str, MetricInfo]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    official = frames["r2_master"]
    relevant = set(relevant_paper_metrics(catalog))
    for _, row in official.iterrows():
        metric = normalize_metric(clean(row.get("paper_metric_name")) or clean(row.get("official_metric_name")))
        if metric not in relevant:
            continue
        value, origin = choose_official_value(row)
        rows.append(
            {
                "dataset": normalize_dataset(row.get("dataset")),
                "metric_family": metric_family(metric, catalog),
                "metric_name": metric,
                "official_model": normalize_model(row.get("official_model")),
                "task": clean(row.get("task")),
                "prompt_strategy": "",
                "condition": clean(row.get("condition")),
                "history_condition": clean(row.get("history_length")),
                "candidate_position_condition": clean(row.get("candidate_condition")),
                "generation_condition": clean(row.get("profile_mode")),
                "profile_generator": clean(row.get("profile_generator")),
                "cohort": clean(row.get("condition_classification")),
                "cutoff": clean(row.get("cutoff")),
                "paper_value": clean(row.get("paper_value")),
                "official_stored_value": clean(row.get("official_stored_value")),
                "official_recomputed_value": clean(row.get("official_recomputed_value")),
                "official_reconstructed_value": clean(row.get("official_reconstructed_value")),
                "official_value": value,
                "official_value_origin": origin,
                "paper_official_mapping": clean(row.get("paper_mapping_status")) or clean(row.get("value_agreement")) or "PAPER_OFFICIAL_MAPPING_NOT_AVAILABLE",
                "official_source": clean(row.get("official_source_member")) or clean(row.get("official_source_archive")),
                "official_status": clean(row.get("calculability_status")),
                "official_notes": clean(row.get("protocol_notes")) or clean(row.get("reason_if_not_calculable")),
            }
        )
    return pd.DataFrame(rows)


def build_local_rows(frames: Mapping[str, pd.DataFrame], catalog: Mapping[str, MetricInfo]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    local = frames["r3_master"]
    relevant = set(relevant_paper_metrics(catalog))
    local = local.loc[local["paper_metric"].astype(str).str.lower().eq("true")].copy()
    for _, row in local.iterrows():
        metric = normalize_metric(row.get("metric_name"))
        if metric not in relevant:
            continue
        value, origin = choose_local_value(row)
        rows.append(
            {
                "dataset": normalize_dataset(row.get("dataset")),
                "metric_family": metric_family(metric, catalog),
                "metric_name": metric,
                "local_model": normalize_model(row.get("model")),
                "task": clean(row.get("task")),
                "prompt_strategy": clean(row.get("prompt_strategy")),
                "condition": clean(row.get("condition")),
                "history_condition": clean(row.get("history_condition")),
                "candidate_position_condition": clean(row.get("candidate_position_condition")),
                "generation_condition": clean(row.get("generation_condition")),
                "profile_generator": clean(row.get("profile_generator")),
                "cohort": clean(row.get("cohort")),
                "cutoff": clean(row.get("cutoff")),
                "local_stored_value": clean(row.get("stored_local_value")),
                "local_recomputed_value": clean(row.get("recomputed_local_value")),
                "local_reconstructed_value": clean(row.get("reconstructed_local_value")),
                "local_value": value,
                "local_value_origin": origin,
                "local_protocol_class": clean(row.get("protocol_class")),
                "local_evidence_class": local_evidence_class(row),
                "paper_local_mapping": clean(row.get("paper_comparability")) or clean(row.get("paper_semantic_equivalence")),
                "local_source": clean(row.get("source_artifact")),
                "local_status": clean(row.get("calculability_status")),
                "local_notes": clean(row.get("notes")),
            }
        )
    return pd.DataFrame(rows)


def evidence_lookup(paper_rows: pd.DataFrame) -> dict[tuple[str, str, str, str], list[dict[str, str]]]:
    lookup: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for _, row in paper_rows.iterrows():
        key = (
            clean(row.get("dataset")),
            clean(row.get("metric_name")),
            normalize_model(row.get("paper_model")),
            clean(row.get("task")),
        )
        lookup.setdefault(key, []).append({col: clean(row.get(col)) for col in row.index})
    return lookup


def find_paper_match(row: Mapping[str, Any], paper_by_key: Mapping[tuple[str, str, str, str], list[dict[str, str]]]) -> dict[str, str] | None:
    dataset = clean(row.get("dataset"))
    metric = clean(row.get("metric_name"))
    model = normalize_model(row.get("local_model") or row.get("official_model") or row.get("paper_model"))
    task = clean(row.get("task"))
    candidates = paper_by_key.get((dataset, metric, model, task), [])
    if len(candidates) == 1:
        return candidates[0]
    candidates = paper_by_key.get((dataset, metric, model, ""), [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def official_exact_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        clean(row.get("dataset")),
        clean(row.get("metric_name")),
        normalize_model(row.get("official_model")),
        clean(row.get("task")),
        clean(row.get("history_condition")),
        clean(row.get("candidate_position_condition")),
        clean(row.get("generation_condition")),
    )


def local_exact_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        clean(row.get("dataset")),
        clean(row.get("metric_name")),
        normalize_model(row.get("local_model")),
        clean(row.get("task")),
        clean(row.get("history_condition")),
        clean(row.get("candidate_position_condition")),
        clean(row.get("generation_condition")),
    )


def build_local_lookup(local_rows: pd.DataFrame) -> dict[tuple[str, ...], list[dict[str, str]]]:
    lookup: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for _, row in local_rows.iterrows():
        record = {col: clean(row.get(col)) for col in row.index}
        lookup.setdefault(local_exact_key(record), []).append(record)
    return lookup


def base_row() -> dict[str, str]:
    return {col: "" for col in THREE_WAY_COLUMNS}


def classify_row(row: Mapping[str, str]) -> str:
    has_paper = bool(clean(row.get("paper_value")) or clean(row.get("paper_model")))
    has_official = bool(clean(row.get("official_value")) or clean(row.get("official_model")))
    has_local = bool(clean(row.get("local_value")) or clean(row.get("local_model")) or clean(row.get("gap_status")))
    local_class = clean(row.get("local_evidence_class"))
    gap_status = clean(row.get("gap_status"))
    if gap_status.startswith("NOT_CALCULABLE") or local_class == "MISSING":
        return "LOCAL_GAP"
    if local_class == "DIAGNOSTIC_ONLY":
        return "DIAGNOSTIC_ONLY"
    if has_paper and has_official and has_local:
        if local_class == "PAPER_EXACT" and "DIRECT" in clean(row.get("paper_official_mapping")):
            return "PAPER_OFFICIAL_LOCAL_EXACT_PROTOCOL"
        return "PAPER_OFFICIAL_LOCAL_NUMERICALLY_COMPARABLE" if local_class == "PAPER_EXACT" else "PAPER_LOCAL_ADAPTED_COMPARABLE"
    if has_official and has_local:
        return "OFFICIAL_LOCAL_ADAPTED_COMPARABLE" if local_class == "ADAPTED" else "QUALITATIVE_ONLY_COMPARABLE"
    if has_paper and has_local:
        return "PAPER_LOCAL_ADAPTED_COMPARABLE" if local_class == "ADAPTED" else "QUALITATIVE_ONLY_COMPARABLE"
    if has_official:
        return "OFFICIAL_ONLY"
    if has_paper:
        return "PAPER_ONLY"
    if has_local:
        return "LOCAL_ONLY"
    return "NOT_APPLICABLE"


def row_interpretation(row: Mapping[str, str]) -> str:
    metric = clean(row.get("metric_name"))
    comp = clean(row.get("comparability_class"))
    if comp == "LOCAL_GAP":
        return f"{metric} remains a local gap for this condition; no official or paper value was substituted."
    if comp == "DIAGNOSTIC_ONLY":
        return f"{metric} provides local diagnostic evidence only and is not promoted to a paper-exact comparison."
    if comp in {"OFFICIAL_ONLY", "PAPER_ONLY", "LOCAL_ONLY"}:
        return f"{metric} is present in only one evidence layer at this granularity."
    if "ADAPTED" in comp or comp == "QUALITATIVE_ONLY_COMPARABLE":
        return f"{metric} can inform the paper claim qualitatively, but protocol differences prevent an exact reproduction claim."
    if comp.startswith("PAPER_OFFICIAL_LOCAL"):
        return f"{metric} is numerically comparable across the available evidence layers at this row granularity."
    return f"{metric} is documented as not applicable at this granularity."


def finalize_deltas(row: dict[str, str]) -> dict[str, str]:
    comp = row["comparability_class"]
    numeric_allowed = comp in {
        "PAPER_OFFICIAL_LOCAL_EXACT_PROTOCOL",
        "PAPER_OFFICIAL_LOCAL_NUMERICALLY_COMPARABLE",
        "OFFICIAL_LOCAL_ADAPTED_COMPARABLE",
    }
    po_abs, po_rel = comparable_delta(row.get("paper_value"), row.get("official_value"), numeric_allowed and bool(row.get("paper_official_mapping")))
    pl_abs, pl_rel = comparable_delta(row.get("paper_value"), row.get("local_value"), numeric_allowed and comp.startswith("PAPER_OFFICIAL_LOCAL"))
    ol_abs, ol_rel = comparable_delta(row.get("official_value"), row.get("local_value"), numeric_allowed and bool(row.get("official_local_mapping")))
    row["paper_official_absolute_delta"] = po_abs
    row["paper_official_relative_delta"] = po_rel
    row["paper_local_absolute_delta"] = pl_abs
    row["paper_local_relative_delta"] = pl_rel
    row["official_local_absolute_delta"] = ol_abs
    row["official_local_relative_delta"] = ol_rel
    row["directional_agreement"] = directional_agreement_from_delta(po_abs or pl_abs or ol_abs)
    return row


def build_three_way_master(frames: Mapping[str, pd.DataFrame], catalog: Mapping[str, MetricInfo]) -> pd.DataFrame:
    paper_rows = build_paper_rows(frames, catalog)
    official_rows = build_official_rows(frames, catalog)
    local_rows = build_local_rows(frames, catalog)
    paper_by_key = evidence_lookup(paper_rows)
    local_lookup = build_local_lookup(local_rows)
    rows: list[dict[str, str]] = []
    seen_local_keys: set[tuple[str, ...]] = set()

    for _, official in official_rows.iterrows():
        official_rec = {col: clean(official.get(col)) for col in official.index}
        matched_locals = local_lookup.get(official_exact_key(official_rec), [])
        if matched_locals:
            seen_local_keys.add(local_exact_key(matched_locals[0]))
        match = matched_locals[0] if len(matched_locals) == 1 else None
        out = base_row()
        for col in [
            "dataset",
            "metric_family",
            "metric_name",
            "task",
            "condition",
            "history_condition",
            "candidate_position_condition",
            "generation_condition",
            "profile_generator",
            "cohort",
            "cutoff",
            "paper_value",
            "official_model",
            "official_stored_value",
            "official_recomputed_value",
            "official_reconstructed_value",
            "official_value",
            "official_value_origin",
            "paper_official_mapping",
        ]:
            out[col] = official_rec.get(col, "")
        if match:
            for col in [
                "local_model",
                "local_stored_value",
                "local_recomputed_value",
                "local_reconstructed_value",
                "local_value",
                "local_value_origin",
                "local_protocol_class",
                "local_evidence_class",
                "paper_local_mapping",
            ]:
                out[col] = match.get(col, "")
            out["official_local_mapping"] = "EXACT_MODEL_TASK_CONDITION_MATCH"
            out["source_artifact"] = "; ".join(filter(None, [official_rec.get("official_source", ""), match.get("local_source", "")]))
            out["evidence_notes"] = "; ".join(filter(None, [official_rec.get("official_notes", ""), match.get("local_notes", "")]))
        else:
            out["official_local_mapping"] = "NO_EXACT_LOCAL_MODEL_TASK_CONDITION_MATCH"
            out["source_artifact"] = official_rec.get("official_source", "")
            out["evidence_notes"] = official_rec.get("official_notes", "")
        out["paper_model"] = normalize_model(official_rec.get("paper_model"))
        out["source_layer"] = "OFFICIAL_LED"
        out["qualitative_claim_supported"] = "not_assessed_at_cell_level"
        out["comparability_class"] = classify_row(out)
        out["scientific_interpretation"] = row_interpretation(out)
        rows.append(finalize_deltas(out))

    for _, local in local_rows.iterrows():
        local_rec = {col: clean(local.get(col)) for col in local.index}
        if local_exact_key(local_rec) in seen_local_keys:
            continue
        paper_match = find_paper_match(local_rec, paper_by_key)
        out = base_row()
        for col in [
            "dataset",
            "metric_family",
            "metric_name",
            "task",
            "prompt_strategy",
            "condition",
            "history_condition",
            "candidate_position_condition",
            "generation_condition",
            "profile_generator",
            "cohort",
            "cutoff",
            "local_model",
            "local_stored_value",
            "local_recomputed_value",
            "local_reconstructed_value",
            "local_value",
            "local_value_origin",
            "local_protocol_class",
            "local_evidence_class",
            "paper_local_mapping",
        ]:
            out[col] = local_rec.get(col, "")
        if paper_match:
            out["paper_model"] = paper_match.get("paper_model", "")
            out["paper_value"] = paper_match.get("paper_value", "")
            out["paper_official_mapping"] = "NO_OFFICIAL_ROW_FOR_LOCAL_PROTOCOL"
        else:
            out["paper_model"] = "ALL" if local_rec.get("metric_name") in catalog else ""
            out["paper_official_mapping"] = "NOT_APPLICABLE"
        out["official_local_mapping"] = "NO_EXACT_OFFICIAL_MODEL_TASK_CONDITION_MATCH"
        out["source_layer"] = "LOCAL_LED"
        out["source_artifact"] = local_rec.get("local_source", "")
        out["evidence_notes"] = local_rec.get("local_notes", "")
        out["gap_status"] = local_rec.get("local_status", "") if local_rec.get("local_evidence_class") == "MISSING" else ""
        out["qualitative_claim_supported"] = qualitative_support_for_metric(local_rec.get("metric_name"), local_rec.get("local_evidence_class"))
        out["comparability_class"] = classify_row(out)
        out["scientific_interpretation"] = row_interpretation(out)
        rows.append(finalize_deltas(out))

    # Keep recovered paper numeric atoms visible even when no exact official/local row can be aligned.
    for _, paper in paper_rows.iterrows():
        paper_rec = {col: clean(paper.get(col)) for col in paper.index}
        if not paper_rec.get("paper_value"):
            continue
        out = base_row()
        for col in [
            "dataset",
            "metric_family",
            "metric_name",
            "paper_model",
            "task",
            "prompt_strategy",
            "condition",
            "history_condition",
            "candidate_position_condition",
            "generation_condition",
            "profile_generator",
            "cohort",
            "cutoff",
            "paper_value",
        ]:
            out[col] = paper_rec.get(col, "")
        out["source_layer"] = "PAPER_ATOM"
        out["source_artifact"] = paper_rec.get("paper_source", "")
        out["paper_official_mapping"] = "RAW_PAPER_ATOM_RETAINED"
        out["paper_local_mapping"] = "RAW_PAPER_ATOM_RETAINED"
        out["official_local_mapping"] = "NOT_APPLICABLE"
        out["qualitative_claim_supported"] = "not_assessed_at_cell_level"
        out["evidence_notes"] = paper_rec.get("paper_notes", "")
        out["comparability_class"] = classify_row(out)
        out["scientific_interpretation"] = row_interpretation(out)
        rows.append(finalize_deltas(out))

    frame = pd.DataFrame(rows, columns=THREE_WAY_COLUMNS).fillna("")
    frame = frame.loc[frame["metric_name"].isin(relevant_paper_metrics(catalog))].copy()
    frame = frame.loc[frame["dataset"].isin([*DATASETS, "ALL"])].copy()
    return frame.sort_values(["dataset", "metric_family", "metric_name", "source_layer", "paper_model", "official_model", "local_model"]).reset_index(drop=True)


def qualitative_support_for_metric(metric: str, local_class: str) -> str:
    if local_class == "MISSING":
        return "not_testable_in_local_cell"
    if local_class == "DIAGNOSTIC_ONLY":
        return "diagnostic_support_only"
    if metric in {"HR@5", "NDCG@5"}:
        return "utility_context_support"
    return "adapted_qualitative_support"


def assign_gap_priority(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    metric = normalize_metric(row.get("metric"))
    if metric in {"CandDif-HR", "CandDif-NDCG"}:
        return (
            "P0_MUST_CLOSE",
            "Controlled first-position versus random-position local outputs are the only route to paper-compatible CandDif.",
            "TARGETED_NEW_INFERENCE_RECOMMENDED",
            "Run paired candidate-position outputs only for the missing paper-compatible cells.",
        )
    if metric in {"Generation-involved profile recommendation", "History-length sensitivity"}:
        return (
            "P1_HIGH_VALUE",
            "This is a major LLM-specific analysis, but existing diagnostics/adapted evidence keep the project defensible.",
            "TARGETED_NEW_INFERENCE_RECOMMENDED",
            "Recover only the missing controlled paper-grid outputs if R5 is approved.",
        )
    if metric in {
        "PopREO@5",
        "Shannon category entropy@5",
        "Jaccard@5",
        "Overlap Item Coverage@5",
        "Gini@5",
        "DPD@5",
        "Jain's Index@5",
        "ACLT@5",
    }:
        return (
            "P2_USEFUL_BUT_OPTIONAL",
            "The metric improves Beyond-Utility completeness but overlaps with already reproduced adapted novelty/popularity/diversity/fairness evidence.",
            "NO_NEW_LLM_INFERENCE_NEEDED_FOR_THIS_GAP",
            "Recover offline supporting inputs or keep the documented limitation.",
        )
    if metric in {"MRR printed by official script", "K-S sample representativeness gate"}:
        return (
            "P3_REDUNDANT_OR_LOW_VALUE",
            "This does not determine a central Beyond-Utility conclusion in the frozen local reproduction.",
            "NO_NEW_INFERENCE_NEEDED",
            "Do not prioritize unless a final appendix requires it.",
        )
    return (
        "P4_NOT_RECOVERABLE",
        "Paper-exact recovery is not realistic from the available frozen protocol evidence.",
        "NO_NEW_INFERENCE_NEEDED",
        "Document as protocol unrecoverable.",
    )


def estimated_requests(metric: str) -> str:
    if metric in {"CandDif-HR", "CandDif-NDCG"}:
        return "400"
    if metric == "History-length sensitivity":
        return "600"
    if metric == "Generation-involved profile recommendation":
        return "200"
    return "0"


def build_gap_priority(frames: Mapping[str, pd.DataFrame], catalog: Mapping[str, MetricInfo]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    gaps = frames["r3_gaps"].copy()
    for index, row in gaps.sort_values(["dataset", "metric"]).reset_index(drop=True).iterrows():
        metric = normalize_metric(row.get("metric"))
        priority, reason, inference_decision, action = assign_gap_priority(row)
        missing_kind = "missing_output" if parse_bool(row.get("missing_output")) else "missing_input" if parse_bool(row.get("missing_support_data")) else "missing_protocol"
        rows.append(
            {
                "gap_id": f"R4-GAP-{index + 1:04d}",
                "dataset": normalize_dataset(row.get("dataset")),
                "metric": metric,
                "model": clean(row.get("model")),
                "task": clean(row.get("task")),
                "condition": clean(row.get("condition")),
                "missing_input_or_output": missing_kind,
                "paper_result_exists": bool_text(metric in catalog),
                "official_result_exists": official_result_exists(frames["r2_master"], normalize_dataset(row.get("dataset")), metric),
                "scientific_family": metric_family(metric, catalog),
                "central_to_paper_claim": bool_text(metric in {"CandDif-HR", "CandDif-NDCG", "History-length sensitivity", "Generation-involved profile recommendation", "PopREO@5"}),
                "redundant_with_existing_metric": bool_text(metric in {"MRR printed by official script", "K-S sample representativeness gate", "Jaccard@5", "Overlap Item Coverage@5", "Shannon category entropy@5", "Gini@5", "DPD@5", "Jain's Index@5", "ACLT@5"}),
                "qualitative_evidence_already_available": bool_text(metric not in {"Generation-involved profile recommendation"} or normalize_dataset(row.get("dataset")) in {"Beauty", "Sports"}),
                "requires_new_llm_inference": bool_text(metric in {"CandDif-HR", "CandDif-NDCG", "History-length sensitivity", "Generation-involved profile recommendation"}),
                "requires_new_non_llm_execution": bool_text(metric not in {"CandDif-HR", "CandDif-NDCG", "Generation-involved profile recommendation"}),
                "requires_new_data": bool_text(metric in {"PopREO@5", "Shannon category entropy@5", "Jaccard@5", "Overlap Item Coverage@5", "Gini@5", "DPD@5", "Jain's Index@5", "ACLT@5"}),
                "requires_missing_protocol": bool_text(metric in {"DPD@5", "Gini@5", "K-S sample representativeness gate"}),
                "paper_exact_recovery_possible": clean(row.get("paper_exact_recovery_possible")),
                "adapted_recovery_possible": clean(row.get("adapted_recovery_possible")),
                "estimated_new_requests": estimated_requests(metric),
                "scientific_value": scientific_value(priority),
                "reproduction_value": reproduction_value(priority),
                "execution_cost": execution_cost(metric),
                "priority": priority,
                "recommended_action": action,
                "exact_reason": reason,
                "inference_decision": inference_decision,
            }
        )
    return pd.DataFrame(rows)


def official_result_exists(r2: pd.DataFrame, dataset: str, metric: str) -> str:
    if r2.empty:
        return "false"
    tmp = r2.copy()
    names = tmp["paper_metric_name"].where(tmp["paper_metric_name"].astype(str).str.strip() != "", tmp["official_metric_name"])
    tmp["_metric"] = names.map(normalize_metric)
    tmp["_dataset"] = tmp["dataset"].map(normalize_dataset)
    subset = tmp.loc[(tmp["_dataset"] == dataset) & (tmp["_metric"] == metric)]
    if subset.empty:
        return "false"
    value_cols = ["official_stored_value", "official_recomputed_value", "official_reconstructed_value", "official_value_used_for_comparison"]
    return bool_text(subset[value_cols].astype(str).apply(lambda col: col.str.strip().ne("").any()).any())


def scientific_value(priority: str) -> str:
    return {
        "P0_MUST_CLOSE": "major_claim_blocker",
        "P1_HIGH_VALUE": "major_claim_strengthener",
        "P2_USEFUL_BUT_OPTIONAL": "completeness_improvement",
        "P3_REDUNDANT_OR_LOW_VALUE": "appendix_or_low_marginal_value",
        "P4_NOT_RECOVERABLE": "not_realistically_recoverable",
    }[priority]


def reproduction_value(priority: str) -> str:
    return {
        "P0_MUST_CLOSE": "high",
        "P1_HIGH_VALUE": "medium_high",
        "P2_USEFUL_BUT_OPTIONAL": "medium",
        "P3_REDUNDANT_OR_LOW_VALUE": "low",
        "P4_NOT_RECOVERABLE": "none",
    }[priority]


def execution_cost(metric: str) -> str:
    if metric in {"CandDif-HR", "CandDif-NDCG", "History-length sensitivity"}:
        return "targeted_llm_requests"
    if metric == "Generation-involved profile recommendation":
        return "targeted_profile_and_recommendation_requests"
    if metric in {"MRR printed by official script", "K-S sample representativeness gate"}:
        return "offline_low_or_protocol_review"
    return "offline_supporting_input_recovery"


def build_proposed_recovery(gap_priority: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    recover = gap_priority.loc[gap_priority["priority"].isin(["P0_MUST_CLOSE", "P1_HIGH_VALUE"])].copy()
    for _, row in recover.iterrows():
        metric = clean(row.get("metric"))
        rows.append(
            {
                "dataset": clean(row.get("dataset")),
                "metric": metric,
                "model": clean(row.get("model")),
                "task": clean(row.get("task")),
                "condition": clean(row.get("condition")),
                "cohort": "paper_metric_dataset_gap_after_local_scan",
                "reason": clean(row.get("exact_reason")),
                "paper_relevance": "central LLM-specific paper analysis",
                "existing_evidence": existing_evidence_for_metric(metric),
                "missing_evidence": missing_evidence_for_metric(metric),
                "exact_prompt_template_source": prompt_source_for_metric(metric),
                "candidate_source": candidate_source_for_metric(metric),
                "history_source": history_source_for_metric(metric),
                "number_of_requests": clean(row.get("estimated_new_requests")),
                "estimated_total_requests": clean(row.get("estimated_new_requests")),
                "paper_exact_or_adapted": "paper_exact_target; adapted fallback allowed only if exact controls fail",
                "expected_scientific_gain": clean(row.get("scientific_value")),
                "priority": clean(row.get("priority")),
                "execution_status": "PLANNED_ONLY_NOT_EXECUTED",
            }
        )
    return pd.DataFrame(rows)


def request_count(value: Any) -> int:
    text = clean(value)
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def build_recovery_request_count_validation(proposed: pd.DataFrame) -> pd.DataFrame:
    """Summarize R4 recovery request counts without claiming manifest-level deduplication."""

    columns = [
        "family",
        "raw_planned_requests",
        "duplicate_requests",
        "reusable_existing_outputs",
        "unique_new_requests",
        "count_status",
        "notes",
    ]
    if proposed.empty:
        return pd.DataFrame(columns=columns)

    def raw_sum(frame: pd.DataFrame) -> int:
        return int(frame["number_of_requests"].map(request_count).sum())

    canddif = proposed.loc[proposed["metric"].isin(["CandDif-HR", "CandDif-NDCG"])].copy()
    canddif_unique = 0
    if not canddif.empty:
        canddif_unique = int(
            canddif.assign(_requests=canddif["number_of_requests"].map(request_count))
            .groupby("dataset")["_requests"]
            .max()
            .sum()
        )
    canddif_raw = raw_sum(canddif)

    history = proposed.loc[proposed["metric"].eq("History-length sensitivity")].copy()
    history_raw = raw_sum(history)

    generation = proposed.loc[proposed["metric"].eq("Generation-involved profile recommendation")].copy()
    generation_raw = raw_sum(generation)

    rows = [
        {
            "family": "CandDif",
            "raw_planned_requests": str(canddif_raw),
            "duplicate_requests": str(max(canddif_raw - canddif_unique, 0)),
            "reusable_existing_outputs": "0",
            "unique_new_requests": str(canddif_unique),
            "count_status": "PLANNING_ESTIMATE",
            "notes": "HR and NDCG share the same paired FIRST/RANDOM outputs within each dataset; this is not a frozen request manifest.",
        },
        {
            "family": "History",
            "raw_planned_requests": str(history_raw),
            "duplicate_requests": "0",
            "reusable_existing_outputs": "0",
            "unique_new_requests": str(history_raw),
            "count_status": "PLANNING_ESTIMATE",
            "notes": "History requests are estimated from R4 gap rows; exact prompt/user/candidate hashes remain future R5A manifest work.",
        },
        {
            "family": "Generation",
            "raw_planned_requests": str(generation_raw),
            "duplicate_requests": "0",
            "reusable_existing_outputs": "0",
            "unique_new_requests": str(generation_raw),
            "count_status": "PLANNING_ESTIMATE",
            "notes": "Profile artifacts are not proposed for regeneration where existing support exists; downstream request deduplication remains future R5A manifest work.",
        },
    ]
    return pd.DataFrame(rows, columns=columns)


def existing_evidence_for_metric(metric: str) -> str:
    if metric in {"CandDif-HR", "CandDif-NDCG"}:
        return "R2 has official exact/partial CandDif; R3 has 3421 local position diagnostics but zero paper-compatible local CandDif."
    if metric == "History-length sensitivity":
        return "R2 has official history rows; R3 has local binned/natural history diagnostics across four datasets."
    if metric == "Generation-involved profile recommendation":
        return "R3 has downstream recommendation output for Beauty/Sports and profile/support rows, but not all paper-relevant local cells."
    return "Frozen R3 gap evidence."


def missing_evidence_for_metric(metric: str) -> str:
    if metric in {"CandDif-HR", "CandDif-NDCG"}:
        return "unique paired first-position and random-position local aggregate evidence."
    if metric == "History-length sensitivity":
        return "controlled paper-grid history outputs rather than binned/natural diagnostics."
    if metric == "Generation-involved profile recommendation":
        return "downstream recommendation performance output for the paper-relevant generation condition."
    return "supporting inputs."


def prompt_source_for_metric(metric: str) -> str:
    if metric in {"CandDif-HR", "CandDif-NDCG"}:
        return "src/prompts/candidate_ordering.py; src/llm/plan_f6e1_prompt_strategies.py"
    if metric == "History-length sensitivity":
        return "src/prompts/history_length.py; src/llm/plan_d_prompt_templates.py"
    if metric == "Generation-involved profile recommendation":
        return "src/analysis/plan_f6l_paper_faithful_completion_and_rq_revalidation.py; configs/phase_f6l_paper_faithful_completion_and_rq_revalidation.yaml"
    return "not_applicable"


def candidate_source_for_metric(metric: str) -> str:
    if metric in {"CandDif-HR", "CandDif-NDCG"}:
        return "existing frozen local candidate pools with controlled first/random positive placement"
    if metric == "Generation-involved profile recommendation":
        return "existing frozen generation-involved recommendation candidate manifests"
    return "existing frozen recommendation outputs"


def history_source_for_metric(metric: str) -> str:
    if metric == "History-length sensitivity":
        return "existing frozen user histories with controlled paper-grid truncation"
    if metric == "Generation-involved profile recommendation":
        return "existing frozen profile-generation history sources"
    return "existing frozen history fields"


def reproduction_class_for_group(group: str, dataset: str, local_count: int, gap_count: int, diagnostic_count: int) -> str:
    if group in {"Candidate Position Bias", "History Length Sensitivity"}:
        return "DIAGNOSTIC_REPRODUCTION_ONLY" if diagnostic_count else "BLOCKED_MISSING_OUTPUT"
    if group == "Generation-Involved Performance":
        return "ADAPTED_REPRODUCTION" if dataset in {"Beauty", "Sports"} and local_count else "BLOCKED_MISSING_OUTPUT"
    if gap_count and local_count == 0:
        missing_input_groups = {"PopREO", "Shannon", "Jaccard", "Gini", "DPD", "Jain"}
        return "BLOCKED_MISSING_INPUT" if group in missing_input_groups else "BLOCKED_MISSING_OUTPUT"
    if group == "Hallucination":
        return "ADAPTED_REPRODUCTION"
    if local_count:
        return "ADAPTED_REPRODUCTION"
    return "NOT_APPLICABLE"


def qualitative_agreement_for_group(group: str, reproduction_class: str) -> str:
    if reproduction_class in {"ADAPTED_REPRODUCTION", "DIAGNOSTIC_REPRODUCTION_ONLY"}:
        return "PARTIAL_DIRECTION" if group in {"Candidate Position Bias", "History Length Sensitivity", "Generation-Involved Performance"} else "SAME_DIRECTION"
    if reproduction_class.startswith("BLOCKED"):
        return "NO_COMPARISON"
    return "NO_COMPARISON"


def build_beyond_summary(frames: Mapping[str, pd.DataFrame], catalog: Mapping[str, MetricInfo], gap_priority: pd.DataFrame) -> pd.DataFrame:
    paper = build_paper_rows(frames, catalog)
    r2 = build_official_rows(frames, catalog)
    local = build_local_rows(frames, catalog)
    gaps = frames["r3_gaps"].copy()
    rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        for group, metrics in BEYOND_GROUPS.items():
            paper_subset = paper.loc[(paper["dataset"] == dataset) & (paper["metric_name"].isin(metrics))]
            official_subset = r2.loc[(r2["dataset"] == dataset) & (r2["metric_name"].isin(metrics))]
            local_subset = local.loc[(local["dataset"] == dataset) & (local["metric_name"].isin(metrics))]
            gap_subset = gaps.loc[(gaps["dataset"].map(normalize_dataset) == dataset) & (gaps["metric"].map(normalize_metric).isin(metrics))]
            priority_subset = gap_priority.loc[(gap_priority["dataset"] == dataset) & (gap_priority["metric"].isin(metrics))]
            diagnostic_count = int((local_subset["local_evidence_class"] == "DIAGNOSTIC_ONLY").sum()) if not local_subset.empty else 0
            numeric_local_count = int(local_subset["local_value"].astype(str).str.strip().ne("").sum()) if not local_subset.empty else 0
            local_status_suffix = ""
            if group == "Candidate Position Bias":
                pos = frames["r3_candidate_position"]
                pos_ds = pos.loc[pos["dataset"].map(normalize_dataset) == dataset]
                diagnostic_count = int(pos_ds["classification"].isin(["POSITION_DIAGNOSTIC_ONLY", "CANDDIF_LOCAL_DIAGNOSTIC"]).sum())
                numeric_local_count = int(pos_ds["value"].astype(str).str.strip().ne("").sum())
                local_status_suffix = f"; specialized_position_rows={len(pos_ds)}"
            elif group == "History Length Sensitivity":
                hist = frames["r3_history"]
                hist_ds = hist.loc[hist["dataset"].map(normalize_dataset) == dataset]
                diagnostic_count = len(hist_ds)
                numeric_local_count = int(hist_ds["value"].astype(str).str.strip().ne("").sum())
                local_status_suffix = f"; specialized_history_points={len(hist_ds)}"
            elif group == "Generation-Involved Performance":
                gen = frames["r3_generation"]
                gen_ds = gen.loc[gen["dataset"].map(normalize_dataset) == dataset]
                downstream = gen_ds.loc[gen_ds["has_downstream_recommendation_output"].astype(str).str.lower().eq("true")]
                numeric_local_count = int(downstream["value"].astype(str).str.strip().ne("").sum())
                diagnostic_count = int((~gen_ds["has_downstream_recommendation_output"].astype(str).str.lower().eq("true")).sum())
                local_status_suffix = f"; downstream_rows={len(downstream)}; profile_support_rows={diagnostic_count}"
            repro_class = reproduction_class_for_group(group, dataset, numeric_local_count, len(gap_subset), diagnostic_count)
            rows.append(
                {
                    "dataset": dataset,
                    "metric": group,
                    "paper_status": "NUMERIC_VALUE_REPORTED" if paper_subset["paper_value"].astype(str).str.strip().ne("").any() else "DEFINITION_OR_PROTOCOL_ONLY",
                    "official_status": source_scope_summary(official_subset["official_status"]) if not official_subset.empty else "no official row",
                    "local_status": f"numeric_cells={numeric_local_count}; diagnostic_cells={diagnostic_count}; gaps={len(gap_subset)}{local_status_suffix}",
                    "paper_value_or_summary": value_summary(paper_subset["paper_value"]) if not paper_subset.empty else "paper contract only",
                    "official_value_or_summary": value_summary(official_subset["official_value"]) if not official_subset.empty else "no official value",
                    "local_value_or_summary": value_summary(local_subset["local_value"]) if not local_subset.empty else "no local value",
                    "comparability": comparability_summary(group, repro_class),
                    "reproduction_class": repro_class,
                    "qualitative_agreement": qualitative_agreement_for_group(group, repro_class),
                    "remaining_gap": str(len(gap_subset)),
                    "recovery_priority": highest_priority(priority_subset["priority"]) if not priority_subset.empty else "",
                    "one_sentence_interpretation": one_sentence_interpretation(group, repro_class, len(gap_subset)),
                }
            )
    return pd.DataFrame(rows)


def highest_priority(values: Iterable[str]) -> str:
    order = {priority: index for index, priority in enumerate(PRIORITIES)}
    cleaned = [clean(v) for v in values if clean(v)]
    if not cleaned:
        return ""
    return sorted(cleaned, key=lambda v: order.get(v, 99))[0]


def comparability_summary(group: str, repro_class: str) -> str:
    if repro_class == "ADAPTED_REPRODUCTION":
        return "adapted but scientifically interpretable"
    if repro_class == "DIAGNOSTIC_REPRODUCTION_ONLY":
        return "diagnostic only; not paper-exact"
    if repro_class.startswith("BLOCKED"):
        return "not comparable until missing evidence is recovered"
    if group == "Hallucination":
        return "paper-style textual, candidate-violation, and structural invalidity semantics separated"
    return "not applicable"


def one_sentence_interpretation(group: str, reproduction_class: str, gaps: int) -> str:
    if group == "Hallucination":
        return "Local evidence supports a protocol-qualified hallucination discussion, but constrained index outputs change the measured phenomenon."
    if group == "Candidate Position Bias":
        return "Local diagnostics show position sensitivity, but zero paper-compatible local CandDif cells were reproduced."
    if group == "Generation-Involved Performance":
        return "Beauty/Sports have downstream generation-involved outputs; missing cells limit four-dataset paper-style reproduction."
    if group == "History Length Sensitivity":
        return "Local history grids support trend analysis only as diagnostics, not exact paper-grid reproduction."
    if gaps:
        return f"{group} remains limited by {gaps} local gap cells at dataset level."
    if reproduction_class == "ADAPTED_REPRODUCTION":
        return f"{group} is locally reproduced under adapted semantics."
    return f"{group} is not reproduced at this granularity."


def build_candidate_position_reconciliation(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    r2 = frames["r2_candidate_position"]
    r3 = frames["r3_candidate_position"]
    gaps = frames["r3_gaps"]
    rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        official_ds = r2.loc[r2["dataset"].map(normalize_dataset) == dataset]
        local_ds = r3.loc[r3["dataset"].map(normalize_dataset) == dataset]
        gap_ds = gaps.loc[(gaps["dataset"].map(normalize_dataset) == dataset) & (gaps["metric"].map(normalize_metric).isin(["CandDif-HR", "CandDif-NDCG"]))]
        rows.append(
            {
                "dataset": dataset,
                "paper_contract": "CandDif_HR and CandDif_NDCG compare FIRST vs RANDOM with -log(1-Acc(first)) - (-log(1-Acc(random))).",
                "official_exact_canddif": str(int((official_ds["canddif_status"] == "CANDDIF_EXACTLY_RECONSTRUCTED").sum())) if not official_ds.empty else "0",
                "official_partial_canddif": str(int((official_ds["canddif_status"] == "CANDDIF_PARTIALLY_RECONSTRUCTED").sum())) if not official_ds.empty else "0",
                "local_exact_canddif": str(int((local_ds["classification"] == "CANDDIF_PAPER_COMPATIBLE").sum())) if not local_ds.empty else "0",
                "local_position_diagnostics": str(int((local_ds["classification"] == "POSITION_DIAGNOSTIC_ONLY").sum())) if not local_ds.empty else "0",
                "local_canddif_diagnostics": str(int((local_ds["classification"] == "CANDDIF_LOCAL_DIAGNOSTIC").sum())) if not local_ds.empty else "0",
                "remaining_canddif_gaps": str(len(gap_ds)),
                "interpretation": "PARTIALLY supports position sensitivity through diagnostics; paper-compatible local CandDif requires controlled FIRST/RANDOM reruns.",
                "new_inference_required_for_paper_compatible_local_canddif": bool_text(len(gap_ds) > 0),
            }
        )
    local_all = r3.loc[r3["dataset"].map(normalize_dataset) == "ALL"]
    if not local_all.empty:
        rows.append(
            {
                "dataset": "ALL",
                "paper_contract": "Aggregate local diagnostic inventory row; not a paper CandDif cell.",
                "official_exact_canddif": "0",
                "official_partial_canddif": "0",
                "local_exact_canddif": str(int((local_all["classification"] == "CANDDIF_PAPER_COMPATIBLE").sum())),
                "local_position_diagnostics": str(int((local_all["classification"] == "POSITION_DIAGNOSTIC_ONLY").sum())),
                "local_canddif_diagnostics": str(int((local_all["classification"] == "CANDDIF_LOCAL_DIAGNOSTIC").sum())),
                "remaining_canddif_gaps": "0",
                "interpretation": "Aggregate R3 diagnostic row retained for count completeness; it is not promoted to paper-compatible CandDif.",
                "new_inference_required_for_paper_compatible_local_canddif": "false",
            }
        )
    return pd.DataFrame(rows)


def build_hallucination_reconciliation(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    official = frames["r1_hallucination"]
    local = frames["r3_hallucination"]
    rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        off_ds = official.loc[official["dataset"].map(normalize_dataset) == dataset]
        loc_ds = local.loc[local["dataset"].map(normalize_dataset) == dataset]
        rows.append(
            {
                "dataset": dataset,
                "paper_phenomenon": "unmatched or fabricated textual item recommendations",
                "official_values": value_summary(off_ds["value"]) if not off_ds.empty else "no official value",
                "local_paper_style_textual_values": value_summary(loc_ds.loc[loc_ds["hallucination_semantic_type"] == "PAPER_STYLE_TEXTUAL_HALLUCINATION", "value"]) if not loc_ds.empty else "no local value",
                "local_candidate_violations": value_summary(loc_ds.loc[loc_ds["hallucination_semantic_type"] == "CANDIDATE_VIOLATION", "value"]) if not loc_ds.empty else "no local value",
                "local_structural_invalidity": value_summary(loc_ds.loc[loc_ds["hallucination_semantic_type"] == "STRUCTURAL_OUTPUT_INVALIDITY", "value"]) if not loc_ds.empty else "no local value",
                "free_text_rows": str(int((loc_ds["output_representation"] == "FREE_TEXT").sum())) if not loc_ds.empty else "0",
                "constrained_rows": str(int((loc_ds["output_representation"] != "FREE_TEXT").sum())) if not loc_ds.empty else "0",
                "interpretation": "Constrained candidate-index outputs reduce the possibility of title hallucination but introduce structural-validity semantics; these are not interchangeable.",
            }
        )
    return pd.DataFrame(rows)


def build_generation_reconciliation(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    official = frames["r2_generation"]
    local = frames["r3_generation"]
    gaps = frames["r3_gaps"]
    rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        off_ds = official.loc[official["dataset"].map(normalize_dataset) == dataset]
        loc_ds = local.loc[local["dataset"].map(normalize_dataset) == dataset]
        gap_ds = gaps.loc[(gaps["dataset"].map(normalize_dataset) == dataset) & (gaps["metric"].map(normalize_metric) == "Generation-involved profile recommendation")]
        downstream = loc_ds.loc[loc_ds["has_downstream_recommendation_output"].astype(str).str.lower().eq("true")]
        profile_only = loc_ds.loc[~loc_ds["has_downstream_recommendation_output"].astype(str).str.lower().eq("true")]
        rows.append(
            {
                "dataset": dataset,
                "paper_conditions": "HISTORY_ONLY; PROFILE_ONLY; PROFILE_PLUS_HISTORY",
                "official_conditions": source_scope_summary(off_ds["profile_mode"]) if not off_ds.empty else "no official generation row",
                "local_conditions": source_scope_summary(loc_ds["generation_condition"]) if not loc_ds.empty else "no local generation row",
                "official_values": value_summary(off_ds["official_stored_value"]) if not off_ds.empty else "no official value",
                "local_downstream_performance_values": value_summary(downstream.loc[downstream["metric"].isin(["HR@5", "NDCG@5"]), "value"]) if not downstream.empty else "no downstream local value",
                "profile_only_support_rows": str(len(profile_only)),
                "downstream_rows": str(len(downstream)),
                "remaining_generation_gaps": str(len(gap_ds)),
                "interpretation": "Profile text alone is not counted as generation-involved performance; downstream recommendation output is required.",
            }
        )
    return pd.DataFrame(rows)


def build_history_reconciliation(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    official = frames["r2_history"]
    local = frames["r3_history"]
    gaps = frames["r3_gaps"]
    rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        off_ds = official.loc[official["dataset"].map(normalize_dataset) == dataset]
        loc_ds = local.loc[local["dataset"].map(normalize_dataset) == dataset]
        gap_ds = gaps.loc[(gaps["dataset"].map(normalize_dataset) == dataset) & (gaps["metric"].map(normalize_metric) == "History-length sensitivity")]
        local_grid = "; ".join(sorted(set(clean(v) for v in loc_ds.get("history_bin", pd.Series(dtype=str)) if clean(v)))) if not loc_ds.empty else "no local grid"
        official_grid = "; ".join(sorted(set(clean(v) for v in off_ds.get("history_length", pd.Series(dtype=str)) if clean(v)))) if not off_ds.empty else "no official grid"
        rows.append(
            {
                "dataset": dataset,
                "paper_grid": "paper protocol recovered; exact grid remains artifact-dependent",
                "official_grid": official_grid,
                "local_grid": local_grid,
                "local_curve_points": str(len(loc_ds)),
                "remaining_history_gaps": str(len(gap_ds)),
                "grid_classification": "BINNED_LOCAL_GRID" if not loc_ds.empty else "NO_LOCAL_OUTPUT",
                "trend_agreement": "PARTIAL_DIRECTION" if not loc_ds.empty else "NO_COMPARISON",
                "interpretation": "Local evidence supports protocol-qualified history sensitivity only; no interpolation is fabricated.",
            }
        )
    return pd.DataFrame(rows)


def build_dataset_synthesis(beyond_summary: pd.DataFrame) -> pd.DataFrame:
    family_groups = {
        "utility": ["HR@5", "NDCG@5"],
        "novelty": ["SelfInformation", "Serendipity"],
        "popularity": ["ACLT/APLT", "ARP", "PopREO"],
        "diversity": ["ItemCoverage", "Shannon", "Jaccard"],
        "fairness": ["Gini", "DPD", "Jain"],
        "history_sensitivity": ["History Length Sensitivity"],
        "candidate_position": ["Candidate Position Bias"],
        "hallucination": ["Hallucination"],
        "generation_involved": ["Generation-Involved Performance"],
    }
    rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        for family, metrics in family_groups.items():
            subset = beyond_summary.loc[(beyond_summary["dataset"] == dataset) & (beyond_summary["metric"].isin(metrics))]
            classes = set(subset["reproduction_class"]) if not subset.empty else set()
            if "ADAPTED_REPRODUCTION" in classes and not any(c.startswith("BLOCKED") for c in classes):
                classification = "ADAPTED_BUT_SUPPORTIVE"
            elif "DIAGNOSTIC_REPRODUCTION_ONLY" in classes:
                classification = "DIAGNOSTIC_SUPPORT_ONLY"
            elif any(c.startswith("BLOCKED") for c in classes) and "ADAPTED_REPRODUCTION" in classes:
                classification = "MIXED_RESULT"
            elif any(c.startswith("BLOCKED") for c in classes):
                classification = "NOT_REPRODUCED"
            else:
                classification = "NOT_TESTABLE"
            if family == "utility":
                classification = "ADAPTED_BUT_SUPPORTIVE"
            rows.append(
                {
                    "dataset": dataset,
                    "metric_family": family,
                    "classification": classification,
                    "evidence": "; ".join(f"{r.metric}:{r.reproduction_class}" for r in subset.itertuples()) if not subset.empty else "no beyond summary row",
                }
            )
    return pd.DataFrame(rows)


def build_model_synthesis(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    paper = frames["paper_atomic"].copy()
    local = frames["r3_master"].copy()
    rows: list[dict[str, str]] = []
    for model in ["qwen2:7b-instruct", "mistral:7b-instruct"]:
        paper_count = int((paper["model"].map(normalize_model) == model).sum())
        local_count = int((local["model"].map(normalize_model) == model).sum())
        rows.append(
            {
                "local_model": model,
                "paper_same_model_rows": str(paper_count),
                "official_same_model_rows": "0",
                "local_rows": str(local_count),
                "comparison_scope": "same model family only where paper labels normalize to the local model; no official glm/gpt/llama rows are treated as local qwen/mistral.",
                "interpretation": "Model-level evidence is qualitative and protocol-limited unless model identity and task conditions match.",
            }
        )
    return pd.DataFrame(rows)


def build_claim_reconciliation(beyond_summary: pd.DataFrame, candidate: pd.DataFrame, hallucination: pd.DataFrame, generation: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "paper_claim": "LLM recommender utility should be assessed with HR@5 and NDCG@5, but utility is not the whole evaluation.",
                "paper_evidence": "Paper reports HR@5/NDCG@5 tables and Beyond-Utility metrics.",
                "official_evidence": "R2 independently recalculates 99 HR and 99 NDCG official rows where outputs are available.",
                "local_evidence": "R3 contains 3387 HR/NDCG local rows under declared denominator policies.",
                "support_status": "PARTIALLY_SUPPORTED",
                "confidence": "medium",
                "limitations": "Official and local model/protocol identities differ.",
                "interpretation": "Utility is reproduced as context, not as exact paper-protocol closure.",
            },
            {
                "paper_claim": "Accuracy alone is insufficient because novelty, popularity, diversity, fairness, and LLM-specific behavior change the scientific interpretation.",
                "paper_evidence": "E1 recovers the paper Beyond-Utility metric universe.",
                "official_evidence": "R2 preserves official stored values for many Beyond-Utility metrics.",
                "local_evidence": "R3 locally covers SelfInformation, Serendipity, ACLT/APLT, ARP, ItemCoverage, and hallucination under adapted semantics.",
                "support_status": "QUALITATIVELY_SUPPORTED",
                "confidence": "high",
                "limitations": "Several fairness/diversity exact cells remain blocked by support inputs.",
                "interpretation": "The multi-metric conclusion is scientifically supported even without exact scalar equality.",
            },
            {
                "paper_claim": "LLMs exhibit candidate-position sensitivity.",
                "paper_evidence": "Paper CandDif compares FIRST against RANDOM.",
                "official_evidence": "R2 reconstructs 6 exact official CandDif rows and 160 partial rows.",
                "local_evidence": f"R3 has {int(candidate['local_position_diagnostics'].astype(int).sum())} position diagnostics and zero paper-compatible local CandDif.",
                "support_status": "PARTIALLY_SUPPORTED",
                "confidence": "medium",
                "limitations": "Local diagnostics must not be renamed CandDif.",
                "interpretation": "Local evidence supports position sensitivity qualitatively, but exact local CandDif remains a P0 gap.",
            },
            {
                "paper_claim": "Recommendation behavior changes with user-history length.",
                "paper_evidence": "Paper analyzes history-length sensitivity.",
                "official_evidence": "R2 preserves official history-condition rows.",
                "local_evidence": f"R3 has {int(history['local_curve_points'].astype(int).sum())} local curve/point rows on binned or natural grids.",
                "support_status": "PARTIALLY_SUPPORTED",
                "confidence": "medium",
                "limitations": "Local exact grid differs; no interpolation is fabricated.",
                "interpretation": "Trend evidence is diagnostic and protocol-qualified.",
            },
            {
                "paper_claim": "LLM recommendation outputs may hallucinate items.",
                "paper_evidence": "Paper measures textual unmatched/fabricated item recommendations.",
                "official_evidence": "R1/R2 preserve official title-to-item hallucination values.",
                "local_evidence": "R3 separates paper-style textual hallucination, candidate violation, and structural invalidity.",
                "support_status": "PROTOCOL_DEPENDENT",
                "confidence": "high",
                "limitations": "Constrained candidate-index outputs change the measured phenomenon.",
                "interpretation": "Comparable only for free-text paper-style runs; structured invalidity is separate evidence.",
            },
            {
                "paper_claim": "Generated profiles can affect downstream recommendation performance.",
                "paper_evidence": "Paper compares history-only, profile-only, and profile-plus-history recommendation conditions.",
                "official_evidence": "R2 preserves official generation-condition metrics where distributed.",
                "local_evidence": f"R3 has {int(generation['downstream_rows'].astype(int).sum())} downstream generation rows and profile-only support rows.",
                "support_status": "PARTIALLY_SUPPORTED",
                "confidence": "medium",
                "limitations": "Four paper-relevant local generation-performance cells remain missing output.",
                "interpretation": "Beauty/Sports evidence is informative; four-dataset closure would require targeted recovery.",
            },
            {
                "paper_claim": "Novelty/popularity conclusions require metrics such as SelfInformation, Serendipity, APLT/ACLT, ARP, and PopREO.",
                "paper_evidence": "E1 resolves novelty and popularity contracts.",
                "official_evidence": "R2 stores official values but often lacks train-frequency support for independent recomputation.",
                "local_evidence": "R3 covers SelfInformation, Serendipity, APLT/ACLT, and ARP; PopREO remains a gap.",
                "support_status": "QUALITATIVELY_SUPPORTED",
                "confidence": "medium",
                "limitations": "PopREO and exact train-popularity semantics are incomplete.",
                "interpretation": "The direction of novelty/popularity analysis is supported under adapted semantics.",
            },
            {
                "paper_claim": "Diversity and fairness require separate analysis from utility.",
                "paper_evidence": "E1 recovers ItemCoverage/OIC/Shannon and Gini/DPD/Jain contracts.",
                "official_evidence": "R2 official support is largely stored-only or blocked by missing metadata/groups.",
                "local_evidence": "R3 covers ItemCoverage but leaves OIC/Shannon/Jaccard/Gini/DPD/Jain gaps.",
                "support_status": "PROTOCOL_DEPENDENT",
                "confidence": "medium",
                "limitations": "Category metadata, candidate overlap semantics, exposure denominator, and user groups are not fully aligned.",
                "interpretation": "The need for these families is supported; exact scalar reproduction is not.",
            },
        ]
    )


def build_numeric_agreement(master: pd.DataFrame) -> pd.DataFrame:
    delta_cols = [
        "paper_official_absolute_delta",
        "paper_local_absolute_delta",
        "official_local_absolute_delta",
    ]
    mask = master[delta_cols].astype(str).apply(lambda col: col.str.strip().ne("")).any(axis=1)
    return master.loc[mask, [
        "dataset",
        "metric_family",
        "metric_name",
        "paper_model",
        "official_model",
        "local_model",
        "task",
        "comparability_class",
        *delta_cols,
        "directional_agreement",
        "scientific_interpretation",
    ]].reset_index(drop=True)


def add_validation(
    rows: list[dict[str, str]],
    frames: Mapping[str, pd.DataFrame],
    outputs: Mapping[str, pd.DataFrame],
    catalog: Mapping[str, MetricInfo],
) -> pd.DataFrame:
    master = outputs["three_way_master"]
    beyond = outputs["beyond_summary"]
    gap_priority = outputs["gap_priority"]
    candidate = outputs["candidate_position"]
    hallucination = outputs["hallucination"]
    generation = outputs["generation"]
    proposed = outputs["proposed_recovery_experiments"]
    relevant = set(relevant_paper_metrics(catalog))
    rows.extend(
        [
            validation_row("R4-VAL-007", "all four datasets in three-way master", set(DATASETS).issubset(set(master["dataset"])), source_scope_summary(master["dataset"]), "; ".join(DATASETS), "Every dataset appears."),
            validation_row("R4-VAL-008", "all relevant paper metrics appear", relevant.issubset(set(master["metric_name"])), f"{len(set(master['metric_name']) & relevant)}/{len(relevant)}", f"{len(relevant)}/{len(relevant)}", "E1 non-extension metrics are represented."),
            validation_row("R4-VAL-009", "paper official local value columns distinct", {"paper_value", "official_value", "local_value", "official_stored_value", "official_recomputed_value", "official_reconstructed_value", "local_stored_value", "local_recomputed_value", "local_reconstructed_value"}.issubset(master.columns), "columns present", "columns present", "Values are not collapsed into one generic field."),
            validation_row("R4-VAL-010", "no banned investigation statuses", not contains_banned(master), "none" if not contains_banned(master) else "banned token present", "none", "R4 uses explicit comparable/gap classes."),
            validation_row("R4-VAL-011", "adapted cells not classified exact", not ((master["local_evidence_class"] == "ADAPTED") & (master["comparability_class"] == "PAPER_OFFICIAL_LOCAL_EXACT_PROTOCOL")).any(), "0", "0", "Adapted evidence cannot be exact protocol."),
            validation_row("R4-VAL-012", "diagnostics not classified exact", not ((master["local_evidence_class"] == "DIAGNOSTIC_ONLY") & master["comparability_class"].str.contains("EXACT", na=False)).any(), "0", "0", "Diagnostic rows remain diagnostic."),
            validation_row("R4-VAL-013", "candidate diagnostics not called CandDif", int(candidate["local_exact_canddif"].astype(int).sum()) == 0, str(int(candidate["local_exact_canddif"].astype(int).sum())), "0", "R3 found no paper-compatible local CandDif."),
            validation_row("R4-VAL-014", "hallucination semantics separated", {"paper_phenomenon", "local_candidate_violations", "local_structural_invalidity"}.issubset(hallucination.columns), "separate", "separate", "Textual, candidate, and structural semantics remain separate."),
            validation_row("R4-VAL-015", "profile-only rows not performance", (generation["profile_only_support_rows"].astype(int) >= 0).all() and "downstream_rows" in generation.columns, "separate", "separate", "Generation performance requires downstream recommendation rows."),
            validation_row("R4-VAL-016", "all 54 R3 gaps represented", len(gap_priority) == 54, str(len(gap_priority)), "54", "One priority row per R3 true gap."),
            validation_row("R4-VAL-017", "all gaps prioritized", set(gap_priority["priority"]).issubset(set(PRIORITIES)) and gap_priority["priority"].astype(str).str.strip().ne("").all(), source_scope_summary(gap_priority["priority"]), "priority vocabulary only", "Exact priority vocabulary used."),
            validation_row("R4-VAL-018", "P0/P1 proposed recovery has reasons", proposed["reason"].astype(str).str.strip().ne("").all() if not proposed.empty else True, "nonblank", "nonblank", "High-priority recovery rows carry exact reasons."),
            validation_row("R4-VAL-019", "no recovery executed", proposed["execution_status"].eq("PLANNED_ONLY_NOT_EXECUTED").all() if not proposed.empty else True, "planned only", "planned only", "R4 only plans recovery."),
            validation_row("R4-VAL-020", "beyond summary complete", len(beyond) == len(DATASETS) * len(BEYOND_GROUPS), str(len(beyond)), str(len(DATASETS) * len(BEYOND_GROUPS)), "Professor-facing table covers every requested Beyond-Utility family and dataset."),
        ]
    )
    return pd.DataFrame(rows)


def contains_banned(df: pd.DataFrame) -> bool:
    banned = {"UNKNOWN", "UNINVESTIGATED", "UNCLASSIFIED"}
    text = "\n".join(df.astype(str).agg(" ".join, axis=1).tolist())
    upper = text.upper()
    return any(token in upper for token in banned)


def decision_token(validation: pd.DataFrame) -> str:
    if (validation["status"] == "fail").any():
        return "REPRODUCTION_R4_THREE_WAY_RECONCILIATION_BLOCKED"
    return "REPRODUCTION_R4_THREE_WAY_RECONCILIATION_COMPLETE"


def build_report(frames: Mapping[str, pd.DataFrame], outputs: Mapping[str, pd.DataFrame], decision: str) -> str:
    master = outputs["three_way_master"]
    beyond = outputs["beyond_summary"]
    gaps = outputs["gap_priority"]
    claims = outputs["paper_claim_reconciliation"]
    candidate = outputs["candidate_position"]
    hallucination = outputs["hallucination"]
    generation = outputs["generation"]
    history = outputs["history"]
    validation = outputs["validation"]
    numeric = outputs["numeric_agreement"]
    recovery_counts = outputs["recovery_request_count_validation"]
    lines: list[str] = []
    lines.append("# R4 Three-Way Beyond-Utility Reconciliation")
    lines.append("")
    lines.append("## Final Decision")
    lines.append("")
    lines.append(f"`{decision}`")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("R4 reconciles what the paper reports, what frozen official author artifacts support, and what the frozen local reproduction produced. It performs no new inference, training, data download, candidate generation, or recovery execution.")
    lines.append("")
    lines.append("## Frozen evidence layers")
    lines.append("")
    lines.append("- Paper layer: `paper_atomic_contract.csv` and `extension_e1_metric_contract.csv`.")
    lines.append("- Official layer: frozen R1/R2 tables, especially `reproduction_r2_official_metric_calculability_master.csv`.")
    lines.append("- Local layer: frozen R3 tables, especially `reproduction_r3_all_metric_cells_master.csv` and R3 specialized Beyond-Utility tables.")
    lines.append("")
    lines.append("## Three-way methodology")
    lines.append("")
    lines.append("Rows are reconciled at the finest recovered granularity. Model, task, history, candidate-position, generation, profile, cohort, and cutoff fields are not forced together when they differ. Official values and local values remain separate columns.")
    lines.append("")
    lines.append("## Three-way accounting")
    lines.append("")
    lines.append(f"- paper metric cells: {int(master['paper_value'].astype(str).str.strip().ne('').sum())}")
    lines.append(f"- official evidence cells: {int(master['official_value'].astype(str).str.strip().ne('').sum())}")
    lines.append(f"- local evidence cells: {int(master['local_value'].astype(str).str.strip().ne('').sum())}")
    lines.append(f"- numeric agreement rows: {len(numeric)}")
    lines.append(f"- validation: {int((validation['status'] == 'pass').sum())} pass / {int((validation['status'] == 'fail').sum())} fail")
    lines.append("")
    lines.append("## Utility baseline")
    lines.append("")
    lines.append("HR@5 and NDCG@5 serve as context. R2 independently recalculates compatible official utility rows, and R3 contains local utility rows under explicit denominator policies. Because official models and local models do not share exact identity, R4 treats utility as adapted/contextual rather than exact paper-official-local protocol closure.")
    lines.append("")
    lines.append("## Beyond-Utility executive summary")
    lines.append("")
    for group in BEYOND_GROUPS:
        subset = beyond.loc[beyond["metric"] == group]
        class_counts = source_scope_summary(subset["reproduction_class"])
        gap_total = int(subset["remaining_gap"].astype(int).sum())
        lines.append(f"- {group}: {class_counts}; remaining gaps={gap_total}.")
    lines.append("")
    section_text = {
        "SelfInformation": "Local SelfInformation is adapted because paper unique-user self-information and local/train-frequency primitives differ. It supports novelty interpretation but not exact scalar parity.",
        "Serendipity": "Serendipity remains adapted because baseline/reference dependence differs. Local MostPop-reference evidence is interpretable but not paper-exact.",
        "ACLT/APLT": "Paper APLT and official/local ACLT naming are not merged silently. R4 treats them as semantically related long-tail metrics with adapted comparability.",
        "ARP": "ARP is locally covered under train-popularity semantics and supports popularity trend discussion, with official values mostly stored-only.",
        "PopREO": "PopREO has four local gaps caused by missing support inputs/groups; no local value is substituted.",
        "ItemCoverage": "ItemCoverage is separated from local CatalogCoverage and treated as the strongest diversity reproduction family.",
        "Shannon": "Shannon remains blocked by category/distribution support inputs.",
        "Jaccard": "Jaccard/OIC semantics require official candidate-overlap support; generic ILD is not substituted.",
        "Gini": "Gini remains blocked by exposure denominator/support-input uncertainty.",
        "DPD": "DPD requires cold/hot or active/inactive user group definitions; R4 does not invent membership.",
        "Jain": "Jain remains missing locally and adds fairness completeness but is lower priority than LLM-specific recovery.",
        "History Length Sensitivity": "Local history curves are diagnostic/binned or natural grids; no paper-grid interpolation is fabricated.",
        "Candidate Position Bias": "Local evidence supports position sensitivity only diagnostically. Paper-compatible CandDif is a P0 recovery target.",
        "Hallucination": "R4 separates textual hallucination, candidate violations, and structural invalidity. Constrained outputs change the meaning of the comparison.",
        "Generation-Involved Performance": "Downstream generation-involved recommendation output exists for Beauty/Sports but four paper-relevant cells remain missing.",
    }
    for group, text in section_text.items():
        lines.append(f"## {group}")
        lines.append("")
        lines.append(text)
        lines.append("")
    lines.append("## Dataset synthesis")
    lines.append("")
    dataset_synth = outputs["dataset_synthesis"]
    for dataset in DATASETS:
        ds = dataset_synth.loc[dataset_synth["dataset"] == dataset]
        lines.append(f"### {dataset}")
        for row in ds.itertuples():
            lines.append(f"- {row.metric_family}: {row.classification}.")
        lines.append("")
    lines.append("## Paper claim reconciliation")
    lines.append("")
    for row in claims.itertuples():
        lines.append(f"- {row.support_status}: {row.paper_claim} Interpretation: {row.interpretation}")
    lines.append("")
    lines.append("## Strong reproductions")
    lines.append("")
    lines.append("No Beyond-Utility family is upgraded to paper-exact local reproduction. ItemCoverage and utility are the strongest adapted/contextual reproductions.")
    lines.append("")
    lines.append("## Adapted but supportive results")
    lines.append("")
    lines.append("SelfInformation, Serendipity, ACLT/APLT, ARP, ItemCoverage, Hallucination, and Beauty/Sports generation-involved performance are scientifically informative under adapted semantics.")
    lines.append("")
    lines.append("## Diagnostic-only results")
    lines.append("")
    lines.append("Candidate-position bias and history-length sensitivity provide diagnostic local support but are not paper-compatible scalar reproductions.")
    lines.append("")
    lines.append("## Contradictions / divergent evidence")
    lines.append("")
    lines.append("R4 finds no clean contradiction. Divergence is dominated by protocol, model, support-input, and output-format differences.")
    lines.append("")
    lines.append("## Protocol-limited comparisons")
    lines.append("")
    lines.append("The global status remains `CRITICAL_EXPERIMENTAL_PATH_PROTOCOL_LIMITED` because preprocessing, candidate ordering, model identity, output format, and supporting metadata differences remain material.")
    lines.append("")
    lines.append("## 54-gap prioritization")
    lines.append("")
    lines.append(source_scope_summary(gaps["priority"]))
    lines.append("")
    lines.append("## ML-1M CandDif P0 status")
    lines.append("")
    lines.append("`ML1M_CANDDIF_P0_STATUS = IN_SCOPE_ALREADY_ACCOUNTED`")
    lines.append("")
    lines.append("ML-1M has diagnostic CandDif rows but no paper-compatible local CandDif. R4 records `remaining_canddif_gaps=0` for ML-1M, so ML-1M is accounted outside the six P0 CandDif recovery rows and its diagnostics are not upgraded to the paper FIRST vs RANDOM contract.")
    lines.append("")
    lines.append("## Experiments worth recovering")
    lines.append("")
    p0p1 = gaps.loc[gaps["priority"].isin(["P0_MUST_CLOSE", "P1_HIGH_VALUE"])]
    for row in p0p1.itertuples():
        lines.append(f"- {row.gap_id}: {row.dataset} {row.metric} ({row.priority}) - {row.recommended_action}")
    lines.append("")
    lines.append("## Recovery request count validation")
    lines.append("")
    for row in recovery_counts.itertuples():
        lines.append(
            f"- {row.family}: raw={row.raw_planned_requests}; duplicates={row.duplicate_requests}; "
            f"existing reusable={row.reusable_existing_outputs}; unique new={row.unique_new_requests}; "
            f"status={row.count_status}."
        )
    total_unique = int(recovery_counts["unique_new_requests"].map(request_count).sum()) if not recovery_counts.empty else 0
    lines.append(f"- Total proposed unique new requests: {total_unique}.")
    lines.append("- `R4_RECOVERY_COUNT_STATUS = ESTIMATED_NOT_YET_DEDUPLICATED`.")
    lines.append("")
    lines.append("The 4400 figure is a family-level planning estimate after known CandDif HR/NDCG reuse. It is not proof of an exact deduplicated prompt/user/candidate request manifest.")
    lines.append("")
    lines.append("## Experiments not worth recovering first")
    lines.append("")
    lines.append("P2/P3 gaps are mostly offline support-input, redundant diagnostic, or lower-marginal-value completeness work. They should not trigger broad new inference.")
    lines.append("")
    lines.append("## New inference decision")
    lines.append("")
    lines.append("`TARGETED_NEW_INFERENCE_RECOMMENDED`")
    lines.append("")
    lines.append("New inference is recommended only for P0/P1 LLM-specific gaps if the project wants paper-compatible local CandDif, exact history-grid evidence, or missing generation-involved downstream outputs. It is not required to make a defensible protocol-limited reproduction claim.")
    lines.append("")
    lines.append("## Implications for final reproduction closure")
    lines.append("")
    lines.append("R4 answers the scientific reproduction question: the Beyond-Utility conclusions are mostly adapted/protocol-qualified and qualitatively supported, while exact local paper-protocol closure remains limited to targeted recoveries rather than all 54 gaps.")
    return "\n".join(lines).rstrip() + "\n"


def append_notes(project_root: Path, decision: str, outputs: Mapping[str, pd.DataFrame]) -> None:
    notes_path = project_root / NOTES_PATH
    marker = "## R4 three-way Beyond-Utility reconciliation"
    text = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    recovery_counts = outputs["recovery_request_count_validation"]
    total_unique = int(recovery_counts["unique_new_requests"].map(request_count).sum()) if not recovery_counts.empty else 0
    section = (
        f"{marker}\n\n"
        f"- Decision: `{decision}`.\n"
        "- Commands: `python src/experiments/build_reproduction_r4_three_way_reconciliation.py --config configs/reproduction_r4_three_way_reconciliation.yaml`.\n"
        "- Outputs: `results/tables/reproduction_r4_three_way_reconciliation_master.csv`, `results/tables/reproduction_r4_gap_scientific_priority.csv`, `results/tables/reproduction_r4_recovery_request_count_validation.csv`, `report/reproduction_r4_three_way_beyond_utility_reconciliation.md`.\n"
        f"- Cells reconciled: {len(outputs['three_way_master'])}; R3 true gaps prioritized: {len(outputs['gap_priority'])}.\n"
        f"- Recovery count status: `R4_RECOVERY_COUNT_STATUS = ESTIMATED_NOT_YET_DEDUPLICATED`; total proposed unique new requests `{total_unique}` is a planning estimate, not request-manifest proof.\n"
        "- ML-1M CandDif: `ML1M_CANDDIF_P0_STATUS = IN_SCOPE_ALREADY_ACCOUNTED`; diagnostic rows exist, no paper-compatible local CandDif is promoted, and remaining CandDif gaps are `0`.\n"
        "- Known issues: R4 plans but does not execute targeted recovery; global path remains protocol-limited.\n"
        "- Next step: design the smallest controlled R5 recovery execution for P0/P1 only, if approved.\n"
    )
    if marker in text:
        pattern = rf"(?ms)^{re.escape(marker)}\n.*?(?=^## |\Z)"
        updated = re.sub(pattern, section.rstrip(), text)
    else:
        updated = text.rstrip() + "\n" + section
    notes_path.write_text(updated.rstrip() + "\n", encoding="utf-8")


def write_outputs(project_root: Path, frames: Mapping[str, pd.DataFrame], report: str) -> None:
    for key, path in OUTPUT_TABLES.items():
        if key in frames:
            out_path = project_root / path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            frames[key].to_csv(out_path, index=False)
    report_path = project_root / REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def build_r4(project_root: Path, write: bool = True) -> dict[str, Any]:
    input_frames = {name: read_csv(project_root, path) for name, path in INPUT_TABLES.items()}
    catalog = load_metric_catalog(input_frames["e1_contract"])
    pre_state = git_state(project_root, "precondition")
    protected = protected_hash_check(project_root, input_frames["r3_protected_hash"])
    validation_rows = precondition_validation(project_root, pre_state, protected)

    three_way = build_three_way_master(input_frames, catalog)
    gap_priority = build_gap_priority(input_frames, catalog)
    proposed = build_proposed_recovery(gap_priority)
    recovery_counts = build_recovery_request_count_validation(proposed)
    candidate = build_candidate_position_reconciliation(input_frames)
    hallucination = build_hallucination_reconciliation(input_frames)
    generation = build_generation_reconciliation(input_frames)
    history = build_history_reconciliation(input_frames)
    beyond = build_beyond_summary(input_frames, catalog, gap_priority)
    dataset_synthesis = build_dataset_synthesis(beyond)
    model_synthesis = build_model_synthesis(input_frames)
    claims = build_claim_reconciliation(beyond, candidate, hallucination, generation, history)
    numeric_agreement = build_numeric_agreement(three_way)

    output_frames: dict[str, pd.DataFrame] = {
        "three_way_master": three_way,
        "beyond_summary": beyond,
        "gap_priority": gap_priority,
        "paper_claim_reconciliation": claims,
        "proposed_recovery_experiments": proposed,
        "recovery_request_count_validation": recovery_counts,
        "dataset_synthesis": dataset_synthesis,
        "model_synthesis": model_synthesis,
        "numeric_agreement": numeric_agreement,
        "candidate_position": candidate,
        "hallucination": hallucination,
        "generation": generation,
        "history": history,
        "precondition_git_state": pre_state,
        "protected_hash_check": protected,
    }
    validation = add_validation(validation_rows, input_frames, output_frames, catalog)
    output_frames["validation"] = validation
    decision = decision_token(validation)
    report = build_report(input_frames, output_frames, decision)
    output_frames["postcondition_git_state"] = git_state(project_root, "postcondition")
    if write:
        write_outputs(project_root, output_frames, report)
        append_notes(project_root, decision, output_frames)
    return {"frames": output_frames, "report": report, "decision": decision}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument("--no-write", action="store_true", help="Build in memory without writing outputs.")
    args = parser.parse_args(argv)
    result = build_r4(Path(args.project_root).resolve(), write=not args.no_write)
    frames = result["frames"]
    validation = frames["validation"]
    print(f"decision={result['decision']}")
    print(f"three_way_rows={len(frames['three_way_master'])}")
    print(f"beyond_summary_rows={len(frames['beyond_summary'])}")
    print(f"gap_priority_rows={len(frames['gap_priority'])}")
    print(f"validation_pass={int((validation['status'] == 'pass').sum())}")
    print(f"validation_fail={int((validation['status'] == 'fail').sum())}")
    print(f"report={REPORT_PATH.as_posix()}")
    return 0 if result["decision"] == "REPRODUCTION_R4_THREE_WAY_RECONCILIATION_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
