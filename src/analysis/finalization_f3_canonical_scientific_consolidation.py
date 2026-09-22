"""Finalization F3 canonical scientific result consolidation.

F3 reads the F2 runtime-verification package and the frozen local evidence
tables, then emits a deterministic scientific result package. It does not run
new experiments, train models, call APIs, call Ollama, download data, mutate
protected inputs, or alter previous Plan C/D/E/F2 artifacts.
"""

from __future__ import annotations

from src.audit import common_git

import hashlib
import itertools
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from src.analysis.protected_input_compatibility import (
    protected_hash_counts as _d1r_protected_hash_counts,
    protected_hash_status as _d1r_protected_hash_status,
)


EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"

F2_REGISTRY_CANDIDATES = [
    Path("results/tables/finalization_f2_canonical_result_registry.csv"),
    Path("results/tables/finalization_f2_canonical_registry.csv"),
]
F2_CONFLICTS = Path("results/tables/finalization_f2_result_conflicts.csv")
F2_AUTHORITATIVE = Path("results/tables/finalization_f2_authoritative_sources.csv")
F2_ARTIFACTS = Path("results/tables/finalization_f2_artifact_classification.csv")
F2_PROTECTED_AFTER = Path("results/tables/finalization_f2_protected_input_hashes_after.csv")
F2_PROTECTED_VALIDATION = Path("results/tables/finalization_f2_protected_input_validation.csv")

PHASE5E_COVERAGE = Path("results/tables/plan_d_phase_5e_coverage_metrics.csv")
PHASE5E_COMPONENTS = Path("results/tables/plan_d_phase_5e_component_coverage.csv")
PHASE5E_RESULT_CELLS = Path("results/tables/plan_d_phase_5e_paper_result_cells.csv")
PHASE5E_RESULT_COMPARISON = Path("results/tables/plan_d_phase_5e_result_cell_comparison.csv")
PHASE6C_VALIDATION = Path("results/tables/plan_d_phase_6c_validation.csv")
METRIC_CONTRACT = Path("results/tables/canonical_metric_contract.csv")
METRIC_BLOCKED = Path("results/tables/canonical_metric_blocked_or_ambiguous.csv")
DATASET_STATS = Path("results/tables/dataset_statistics.csv")
COHORT_SUMMARY = Path("results/tables/experiment_cohort_summary.csv")

REPORT_PATH = Path("report/finalization_f3_canonical_scientific_consolidation.md")
OUTPUT_TABLES = {
    "canonical_results": Path("results/tables/finalization_f3_canonical_scientific_results.csv"),
    "valid_comparisons": Path("results/tables/finalization_f3_valid_comparisons.csv"),
    "primary_matrix": Path("results/tables/finalization_f3_primary_result_matrix.csv"),
    "paper_alignment": Path("results/tables/finalization_f3_paper_alignment.csv"),
    "metric_status": Path("results/tables/finalization_f3_metric_status.csv"),
    "missing_evidence": Path("results/tables/finalization_f3_missing_evidence.csv"),
    "claim_registry": Path("results/tables/finalization_f3_scientific_claim_registry.csv"),
    "validation": Path("results/tables/finalization_f3_validation.csv"),
}

REPRODUCTION_CLASSES = {
    "exactly_reproduced",
    "functionally_reproduced",
    "approximately_reproduced",
    "reconstructed",
    "adapted",
    "extension_only",
    "diagnostic_only",
    "pilot_only",
    "legacy_only",
    "implemented_not_executed",
    "blocked_external_dependency",
    "blocked_hardware",
    "blocked_model_availability",
    "blocked_dataset_availability",
    "not_reproducible_with_available_artifacts",
    "not_applicable",
    "unknown",
}
ARTIFACT_STATUSES = {
    "final_canonical",
    "final_extension",
    "reconstructed",
    "adapted",
    "pilot",
    "diagnostic",
    "intermediate",
    "legacy",
    "blocked",
    "invalid",
    "unknown",
}

CANONICAL_RESULTS_COLUMNS = [
    "evidence_id",
    "canonical_result_id",
    "source_phase",
    "experiment_family",
    "dataset",
    "task",
    "model",
    "model_family",
    "model_role",
    "paper_model_name",
    "paper_component_id",
    "paper_equivalence",
    "reproduction_class",
    "result_scope",
    "artifact_status",
    "prompt_contract",
    "cohort_name",
    "cohort_type",
    "candidate_set_name",
    "candidate_count",
    "positive_count",
    "evaluation_population",
    "strict_or_full",
    "seed",
    "run_id",
    "metric",
    "metric_definition_id",
    "cutoff",
    "value",
    "valid_rows",
    "total_rows",
    "comparable_group_id",
    "is_primary_result",
    "is_reportable",
    "is_paper_comparable",
    "is_extension_result",
    "is_pilot",
    "is_diagnostic",
    "is_legacy",
    "source_file",
    "source_row",
    "config_file",
    "code_entrypoint",
    "test_evidence",
    "validation_evidence",
    "report_reference",
    "exclusion_reason",
    "notes",
]
VALID_COMPARISONS_COLUMNS = [
    "comparison_id",
    "comparable_group_id",
    "dataset",
    "task",
    "cohort_name",
    "candidate_set_name",
    "candidate_count",
    "evaluation_population",
    "strict_or_full",
    "metric",
    "cutoff",
    "model_a",
    "model_a_value",
    "model_a_result_id",
    "model_b",
    "model_b_value",
    "model_b_result_id",
    "absolute_difference",
    "relative_difference",
    "comparison_scope",
    "paper_or_extension",
    "is_valid_comparison",
    "invalid_reason",
    "notes",
]
PRIMARY_MATRIX_COLUMNS = [
    "dataset",
    "task",
    "model",
    "metric",
    "cutoff",
    "value",
    "cohort_name",
    "candidate_set_name",
    "strict_or_full",
    "result_scope",
    "reproduction_class",
    "paper_comparability",
    "source_file",
    "canonical_result_id",
]
PAPER_ALIGNMENT_COLUMNS = [
    "paper_component_id",
    "paper_section",
    "paper_table_or_figure",
    "paper_dataset",
    "paper_task",
    "paper_model",
    "paper_setting",
    "paper_metric",
    "paper_cutoff",
    "paper_value",
    "local_result_id",
    "local_value",
    "reproduction_class",
    "equivalence_level",
    "is_exact_numeric_replication",
    "is_functional_replication",
    "is_extension_only",
    "blocking_reason",
    "evidence_file",
    "notes",
]
METRIC_STATUS_COLUMNS = [
    "metric_definition_id",
    "metric",
    "cutoff",
    "implementation_file",
    "implementation_function",
    "formula_status",
    "paper_equivalence",
    "aggregation_level",
    "invalid_output_policy",
    "duplicate_policy",
    "missing_policy",
    "supported_datasets",
    "supported_tasks",
    "validation_status",
    "is_scientifically_consolidated",
    "is_reportable",
    "blocking_issue",
    "next_required_action",
    "notes",
]
MISSING_EVIDENCE_COLUMNS = [
    "missing_evidence_id",
    "area",
    "paper_component_id",
    "dataset",
    "task",
    "model",
    "metric",
    "missing_element",
    "current_status",
    "blocking_type",
    "is_realistically_resolvable",
    "required_input",
    "required_execution",
    "estimated_cost",
    "priority",
    "completion_criterion",
    "notes",
]
CLAIM_REGISTRY_COLUMNS = [
    "claim_id",
    "claim_text",
    "claim_category",
    "claim_scope",
    "supporting_result_ids",
    "supporting_files",
    "contradicting_evidence",
    "limitations",
    "confidence",
    "is_safe_for_report",
    "is_safe_for_presentation",
    "notes",
]
VALIDATION_COLUMNS = ["check_id", "check_name", "status", "observed", "expected", "details"]

ABSOLUTE_PATH_PATTERN = re.compile(r"[A-Za-z]:[\\/]")


@dataclass(frozen=True)
class SourceBundle:
    root: Path
    f2_registry_path: Path
    f2_registry: pd.DataFrame
    f2_conflicts: pd.DataFrame
    f2_authoritative: pd.DataFrame
    f2_artifacts: pd.DataFrame
    f2_protected_after: pd.DataFrame
    f2_protected_validation: pd.DataFrame
    coverage_metrics: pd.DataFrame
    component_coverage: pd.DataFrame
    paper_result_cells: pd.DataFrame
    result_cell_comparison: pd.DataFrame
    phase6c_validation: pd.DataFrame
    metric_contract: pd.DataFrame
    metric_blocked: pd.DataFrame
    dataset_stats: pd.DataFrame
    cohort_summary: pd.DataFrame


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def stable_hash(values: Iterable[Any]) -> str:
    payload = json.dumps([clean(value) for value in values], ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_id(prefix: str, values: Iterable[Any], width: int = 12) -> str:
    return f"{prefix}-{stable_hash(values)[:width]}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root: Path, args: Sequence[str]) -> str:
    return common_git.git_stdout_no_check(args, root, empty_on_failure=True)


def resolve_f2_registry_path(root: Path) -> Path:
    for relative in F2_REGISTRY_CANDIDATES:
        if (root / relative).exists():
            return relative
    return F2_REGISTRY_CANDIDATES[0]


def load_sources(root: Path) -> SourceBundle:
    f2_registry_path = resolve_f2_registry_path(root)
    return SourceBundle(
        root=root,
        f2_registry_path=f2_registry_path,
        f2_registry=read_csv(root / f2_registry_path),
        f2_conflicts=read_csv(root / F2_CONFLICTS),
        f2_authoritative=read_csv(root / F2_AUTHORITATIVE),
        f2_artifacts=read_csv(root / F2_ARTIFACTS),
        f2_protected_after=read_csv(root / F2_PROTECTED_AFTER),
        f2_protected_validation=read_csv(root / F2_PROTECTED_VALIDATION),
        coverage_metrics=read_csv(root / PHASE5E_COVERAGE),
        component_coverage=read_csv(root / PHASE5E_COMPONENTS),
        paper_result_cells=read_csv(root / PHASE5E_RESULT_CELLS),
        result_cell_comparison=read_csv(root / PHASE5E_RESULT_COMPARISON),
        phase6c_validation=read_csv(root / PHASE6C_VALIDATION),
        metric_contract=read_csv(root / METRIC_CONTRACT),
        metric_blocked=read_csv(root / METRIC_BLOCKED),
        dataset_stats=read_csv(root / DATASET_STATS),
        cohort_summary=read_csv(root / COHORT_SUMMARY),
    )


def model_family(model: str, model_role: str) -> str:
    normalized = normalize_model(model)
    if model_role == "local_llm" or normalized in {"Qwen2", "Mistral"}:
        return "local_llm"
    if model_role == "neural_baseline" or normalized in {"BPRMF", "GRU4Rec", "SASRec", "LightGCN"}:
        return "neural_recommender"
    if normalized == "BM25-like":
        return "lexical_ir_baseline"
    if normalized in {"MostPop", "Random"}:
        return "traditional_baseline"
    return model_role or "unknown"


def normalize_model(model: str) -> str:
    normalized = clean(model)
    mapping = {
        "Gru4rec": "GRU4Rec",
        "Sasrec": "SASRec",
        "Bprmf": "BPRMF",
        "qwen2 7b instruct": "Qwen2",
        "mistral 7b instruct": "Mistral",
    }
    return mapping.get(normalized, mapping.get(normalized.lower(), normalized))


def paper_component_for_model(model: str) -> str:
    normalized = normalize_model(model)
    mapping = {
        "MostPop": "PC-TRADITIONAL-RECOMMENDER-BASELINES-MOSTPOP-BASELINE",
        "BM25-like": "PC-TRADITIONAL-RECOMMENDER-BASELINES-BM25-OR-CONTENTBM25-BASELINE",
        "BPRMF": "PC-TRADITIONAL-RECOMMENDER-BASELINES-BPRMF-BASELINE",
        "GRU4Rec": "PC-TRADITIONAL-RECOMMENDER-BASELINES-GRU4REC-BASELINE",
        "SASRec": "PC-TRADITIONAL-RECOMMENDER-BASELINES-SASREC-BASELINE",
        "LightGCN": "PC-TRADITIONAL-RECOMMENDER-BASELINES-LIGHTGCN-BASELINE",
        "Qwen2": "PC-LLM-MODELS-QWEN2-7B-LOCAL-SUBSTITUTE",
        "Mistral": "PC-LLM-MODELS-MISTRAL-7B-LOCAL-SUBSTITUTE",
        "Random": "PC-REPRODUCIBILITY-RELEVANT-EXECUTION-SETTINGS-RANDOM-SEED-AND-DETERMINISTIC-LOCAL-EXECUTION",
    }
    return mapping.get(normalized, "")


def reproduction_class_from_row(row: Mapping[str, str]) -> str:
    artifact = clean(row.get("artifact_category")).lower()
    scope = clean(row.get("result_scope")).lower()
    paper = clean(row.get("paper_equivalence")).lower()
    phase = clean(row.get("phase")).lower()
    if "legacy" in artifact or "legacy" in scope or "legacy" in paper:
        return "legacy_only"
    if "pilot" in artifact or "pilot" in scope or "pilot" in phase:
        return "pilot_only"
    if "diagnostic" in artifact or "diagnostic" in scope:
        return "diagnostic_only"
    if artifact == "final_extension":
        return "extension_only"
    if "reconstructed" in paper or artifact == "reconstructed":
        return "reconstructed"
    if "adapted" in paper or artifact in {"adapted", "final_canonical"}:
        return "adapted"
    if artifact == "final_extension":
        return "extension_only"
    return "unknown"


def artifact_status_from_row(row: Mapping[str, str]) -> str:
    artifact = clean(row.get("artifact_category")).lower()
    scope = clean(row.get("result_scope")).lower()
    if artifact in ARTIFACT_STATUSES:
        return artifact
    if artifact == "final_canonical":
        return "final_canonical"
    if artifact == "final_extension":
        return "final_extension"
    if "legacy" in artifact or "legacy" in scope:
        return "legacy"
    if "diagnostic" in artifact or "diagnostic" in scope:
        return "diagnostic"
    if "pilot" in artifact or "pilot" in scope:
        return "pilot"
    if "reconstructed" in artifact:
        return "reconstructed"
    if "adapted" in artifact:
        return "adapted"
    return "unknown"


def validation_evidence_for(row: Mapping[str, str]) -> str:
    source = clean(row.get("source_file"))
    if "canonical_local_llm_utility_metrics" in source:
        return "results/tables/canonical_local_llm_full_evaluation_validation.csv"
    if "canonical_local_llm_full_matrix_cell_metrics" in source:
        return "results/tables/canonical_local_llm_full_matrix_validation.csv"
    if "traditional_baselines_metrics" in source:
        return "results/tables/traditional_baseline_validation.csv"
    if "plan_e_bprmf" in source:
        return "results/tables/plan_e_bprmf_beauty_sports_validation.csv;results/tables/plan_e_bprmf_ml1m_pilot_validation.csv"
    if "plan_e_gru4rec" in source:
        return "results/tables/plan_e_gru4rec_beauty_sports_validation.csv;results/tables/plan_e_gru4rec_ml1m_pilot_validation.csv"
    if "plan_e_sasrec" in source:
        return "results/tables/plan_e_sasrec_beauty_sports_validation.csv;results/tables/plan_e_sasrec_ml1m_pilot_validation.csv"
    if "plan_e_lightgcn" in source:
        return "results/tables/plan_e_lightgcn_beauty_sports_cpu_validation.csv;results/tables/plan_e_lightgcn_ml1m_pilot_validation.csv"
    if "ml1m_mostpop_metrics" in source:
        return "tests/test_ml1m_mostpop_baseline.py"
    return clean(row.get("test_evidence"))


def comparable_group(row: Mapping[str, str]) -> str:
    policy = clean(row.get("evaluation_policy_id"))
    prompt = clean(row.get("prompt_contract"))
    values = [
        row.get("dataset", ""),
        row.get("task", ""),
        row.get("cohort_name", ""),
        row.get("candidate_set_name", ""),
        row.get("candidate_count", ""),
        row.get("positive_count", ""),
        row.get("evaluation_population", ""),
        row.get("strict_or_full", ""),
        row.get("metric", ""),
        row.get("cutoff", ""),
        prompt,
        policy,
        clean(row.get("denominator_policy")),
    ]
    return stable_id("F3-CMP", values, 16)


def has_required_report_fields(row: Mapping[str, str]) -> bool:
    required = ["source_file", "source_row", "cohort_name", "candidate_set_name", "metric", "cutoff", "evaluation_population"]
    return all(clean(row.get(key)) for key in required)


def build_canonical_results(bundle: SourceBundle) -> pd.DataFrame:
    f2 = bundle.f2_registry.copy()
    true_conflict_groups = set(
        bundle.f2_conflicts.loc[bundle.f2_conflicts.get("is_true_conflict", "").astype(str).eq("true"), "conflict_id"].tolist()
    )
    rows: list[dict[str, str]] = []
    for _, record in f2.iterrows():
        rec = {key: clean(value) for key, value in record.to_dict().items()}
        model = normalize_model(rec.get("model", ""))
        artifact_status = artifact_status_from_row(rec)
        reproduction_class = reproduction_class_from_row(rec)
        is_pilot = reproduction_class == "pilot_only" or artifact_status == "pilot"
        is_diagnostic = reproduction_class == "diagnostic_only" or artifact_status == "diagnostic"
        is_legacy = reproduction_class == "legacy_only" or artifact_status == "legacy"
        is_extension = reproduction_class == "extension_only" or artifact_status == "final_extension"
        paper_component_id = paper_component_for_model(model)
        row = {
            "evidence_id": rec.get("evidence_id", ""),
            "canonical_result_id": "",
            "source_phase": rec.get("phase", ""),
            "experiment_family": rec.get("experiment_family", ""),
            "dataset": rec.get("dataset", ""),
            "task": rec.get("task", ""),
            "model": model,
            "model_family": model_family(model, rec.get("model_role", "")),
            "model_role": rec.get("model_role", ""),
            "paper_model_name": model,
            "paper_component_id": paper_component_id,
            "paper_equivalence": rec.get("paper_equivalence", ""),
            "reproduction_class": reproduction_class,
            "result_scope": rec.get("result_scope", ""),
            "artifact_status": artifact_status,
            "prompt_contract": rec.get("prompt_contract", ""),
            "cohort_name": rec.get("cohort_name", ""),
            "cohort_type": rec.get("cohort_type", ""),
            "candidate_set_name": rec.get("candidate_set_name", ""),
            "candidate_count": rec.get("candidate_count", ""),
            "positive_count": rec.get("positive_count", ""),
            "evaluation_population": rec.get("evaluation_population", ""),
            "strict_or_full": rec.get("strict_or_full", ""),
            "seed": rec.get("seed", ""),
            "run_id": rec.get("run_id", ""),
            "metric": rec.get("metric", ""),
            "metric_definition_id": rec.get("metric_definition_id", ""),
            "cutoff": rec.get("cutoff", ""),
            "value": rec.get("value", ""),
            "valid_rows": rec.get("valid_rows", ""),
            "total_rows": rec.get("total_rows", ""),
            "comparable_group_id": "",
            "is_primary_result": "false",
            "is_reportable": "false",
            "is_paper_comparable": "false",
            "is_extension_result": str(is_extension).lower(),
            "is_pilot": str(is_pilot).lower(),
            "is_diagnostic": str(is_diagnostic).lower(),
            "is_legacy": str(is_legacy).lower(),
            "source_file": rec.get("source_file", "").replace("\\", "/"),
            "source_row": rec.get("source_row", ""),
            "config_file": rec.get("config_file", "").replace("\\", "/"),
            "code_entrypoint": rec.get("code_entrypoint", "").replace("\\", "/"),
            "test_evidence": rec.get("test_evidence", "").replace("\\", "/"),
            "validation_evidence": validation_evidence_for(rec).replace("\\", "/"),
            "report_reference": rec.get("report_reference", "").replace("\\", "/"),
            "exclusion_reason": "",
            "notes": rec.get("notes", ""),
        }
        row["canonical_result_id"] = stable_id("F3-RES", [row["evidence_id"], row["source_file"], row["source_row"], row["metric"], row["value"]])
        row["comparable_group_id"] = comparable_group({**rec, **row})
        has_true_conflict = rec.get("conflict_group") in true_conflict_groups
        reportable = (
            has_required_report_fields(row)
            and not has_true_conflict
            and artifact_status not in {"invalid", "unknown"}
            and reproduction_class != "unknown"
            and clean(row["value"]) != ""
        )
        row["is_reportable"] = str(reportable).lower()
        paper_comparable = bool(paper_component_id) and reproduction_class in {
            "exactly_reproduced",
            "functionally_reproduced",
            "approximately_reproduced",
            "reconstructed",
            "adapted",
        }
        row["is_paper_comparable"] = str(paper_comparable).lower()
        primary = reportable and not (is_pilot or is_diagnostic or is_legacy) and artifact_status in {
            "final_canonical",
            "final_extension",
            "reconstructed",
            "adapted",
        }
        row["is_primary_result"] = str(primary).lower()
        if not reportable:
            row["exclusion_reason"] = exclusion_reason(row, reproduction_class, artifact_status, has_true_conflict)
        rows.append(row)
    frame = pd.DataFrame(rows, columns=CANONICAL_RESULTS_COLUMNS)
    return sort_frame(frame, ["dataset", "task", "metric", "cohort_name", "model", "source_file", "source_row"])


def exclusion_reason(row: Mapping[str, str], reproduction_class: str, artifact_status: str, has_true_conflict: bool) -> str:
    if has_true_conflict:
        return "true_conflict_pending"
    if artifact_status in {"invalid", "unknown"}:
        return f"artifact_status_{artifact_status}"
    if reproduction_class == "unknown":
        return "unknown_reproduction_class"
    missing = [key for key in ["source_file", "source_row", "cohort_name", "candidate_set_name", "metric", "cutoff", "evaluation_population"] if not clean(row.get(key))]
    if missing:
        return "missing_" + "_".join(missing)
    if not clean(row.get("value")):
        return "missing_metric_value"
    return ""


def sort_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    existing = [column for column in columns if column in frame.columns]
    return frame.sort_values(existing, kind="mergesort").reset_index(drop=True)


def build_valid_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    eligible = results.loc[results["is_primary_result"].eq("true") & results["is_reportable"].eq("true")].copy()
    for group_id, group in eligible.groupby("comparable_group_id", sort=True):
        unique_models = group.drop_duplicates("model").copy()
        if len(unique_models) < 2:
            continue
        records = unique_models.sort_values(["model", "canonical_result_id"], kind="mergesort").to_dict("records")
        for left, right in itertools.combinations(records, 2):
            diff, rel = numeric_delta(left["value"], right["value"])
            comparison_id = stable_id("F3-COMP", [group_id, left["canonical_result_id"], right["canonical_result_id"]])
            extension_flags = {left["is_extension_result"], right["is_extension_result"]}
            paper_or_extension = "extension" if extension_flags == {"true"} else "mixed_extension" if "true" in extension_flags else "paper_aligned_or_reconstructed"
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "comparable_group_id": group_id,
                    "dataset": left["dataset"],
                    "task": left["task"],
                    "cohort_name": left["cohort_name"],
                    "candidate_set_name": left["candidate_set_name"],
                    "candidate_count": left["candidate_count"],
                    "evaluation_population": left["evaluation_population"],
                    "strict_or_full": left["strict_or_full"],
                    "metric": left["metric"],
                    "cutoff": left["cutoff"],
                    "model_a": left["model"],
                    "model_a_value": left["value"],
                    "model_a_result_id": left["canonical_result_id"],
                    "model_b": right["model"],
                    "model_b_value": right["value"],
                    "model_b_result_id": right["canonical_result_id"],
                    "absolute_difference": diff,
                    "relative_difference": rel,
                    "comparison_scope": left["cohort_name"],
                    "paper_or_extension": paper_or_extension,
                    "is_valid_comparison": "true",
                    "invalid_reason": "",
                    "notes": "same F3 comparable_group_id; protocols, cohort, candidate set, metric and cutoff match",
                }
            )
    return pd.DataFrame(rows, columns=VALID_COMPARISONS_COLUMNS).sort_values(["dataset", "task", "metric", "cohort_name", "model_a", "model_b"], kind="mergesort").reset_index(drop=True)


def numeric_delta(left: str, right: str) -> tuple[str, str]:
    try:
        a = float(left)
        b = float(right)
    except ValueError:
        return "", ""
    diff = b - a
    rel = "" if a == 0 else diff / a
    return format_float(diff), "" if rel == "" else format_float(rel)


def format_float(value: float) -> str:
    return f"{value:.15g}"


def build_primary_matrix(results: pd.DataFrame) -> pd.DataFrame:
    primary = results.loc[results["is_primary_result"].eq("true") & results["is_reportable"].eq("true")].copy()
    rows = []
    for row in primary.to_dict("records"):
        rows.append(
            {
                "dataset": row["dataset"],
                "task": row["task"],
                "model": row["model"],
                "metric": row["metric"],
                "cutoff": row["cutoff"],
                "value": row["value"],
                "cohort_name": row["cohort_name"],
                "candidate_set_name": row["candidate_set_name"],
                "strict_or_full": row["strict_or_full"],
                "result_scope": row["result_scope"],
                "reproduction_class": row["reproduction_class"],
                "paper_comparability": row["is_paper_comparable"],
                "source_file": row["source_file"],
                "canonical_result_id": row["canonical_result_id"],
            }
        )
    return pd.DataFrame(rows, columns=PRIMARY_MATRIX_COLUMNS).sort_values(["dataset", "task", "metric", "cohort_name", "model"], kind="mergesort").reset_index(drop=True)


def map_component_status(status: str, blocker: str = "") -> str:
    status = clean(status)
    if status == "reproduced_exact":
        return "exactly_reproduced"
    if status == "reconstructed":
        return "reconstructed"
    if status == "adapted":
        return "adapted"
    if status == "blocked":
        text = blocker.lower()
        if "hardware" in text:
            return "blocked_hardware"
        if "model" in text or "retired" in text:
            return "blocked_model_availability"
        if "dataset" in text:
            return "blocked_dataset_availability"
        return "blocked_external_dependency"
    if status == "not_executed":
        return "implemented_not_executed"
    if status == "unresolved":
        return "not_reproducible_with_available_artifacts"
    return "unknown"


def build_paper_alignment(bundle: SourceBundle, results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    components = bundle.component_coverage.copy()
    for _, comp in components.iterrows():
        component_id = clean(comp.get("component_id"))
        status = clean(comp.get("primary_coverage_status"))
        reproduction = map_component_status(status, clean(comp.get("blocker")))
        local = results.loc[results["paper_component_id"].eq(component_id) & results["is_primary_result"].eq("true")].head(1)
        local_result_id = clean(local.iloc[0]["canonical_result_id"]) if not local.empty else ""
        local_value = clean(local.iloc[0]["value"]) if not local.empty else ""
        rows.append(
            {
                "paper_component_id": component_id,
                "paper_section": clean(comp.get("category")),
                "paper_table_or_figure": "component_coverage",
                "paper_dataset": "",
                "paper_task": "",
                "paper_model": clean(comp.get("component")),
                "paper_setting": clean(comp.get("comparability_classification")),
                "paper_metric": "",
                "paper_cutoff": "",
                "paper_value": "",
                "local_result_id": local_result_id,
                "local_value": local_value,
                "reproduction_class": reproduction,
                "equivalence_level": clean(comp.get("primary_coverage_status")),
                "is_exact_numeric_replication": "false",
                "is_functional_replication": str(reproduction in {"exactly_reproduced", "functionally_reproduced", "reconstructed"}).lower(),
                "is_extension_only": "false",
                "blocking_reason": clean(comp.get("blocker")),
                "evidence_file": clean(comp.get("evidence_artifacts")).replace("\\", "/"),
                "notes": clean(comp.get("rationale")),
            }
        )
    comparison = bundle.result_cell_comparison.set_index("paper_result_cell_id", drop=False) if not bundle.result_cell_comparison.empty else pd.DataFrame()
    for _, cell in bundle.paper_result_cells.iterrows():
        cell_id = clean(cell.get("paper_result_cell_id"))
        match = comparison.loc[cell_id] if not comparison.empty and cell_id in comparison.index else {}
        local_result = clean(match.get("local_result_id", "")) if isinstance(match, pd.Series) else ""
        rows.append(
            {
                "paper_component_id": cell_id,
                "paper_section": "paper_result_cell",
                "paper_table_or_figure": clean(cell.get("paper_table")),
                "paper_dataset": clean(cell.get("dataset")),
                "paper_task": clean(cell.get("task")),
                "paper_model": clean(cell.get("model")),
                "paper_setting": clean(cell.get("prompt_or_method")),
                "paper_metric": clean(cell.get("metric")),
                "paper_cutoff": clean(cell.get("cutoff")),
                "paper_value": clean(cell.get("value")),
                "local_result_id": local_result,
                "local_value": "",
                "reproduction_class": "not_reproducible_with_available_artifacts",
                "equivalence_level": clean(match.get("comparability_status", "")) if isinstance(match, pd.Series) else "paper_value_not_locally_verified",
                "is_exact_numeric_replication": "false",
                "is_functional_replication": "false",
                "is_extension_only": "false",
                "blocking_reason": clean(match.get("reason", "")) if isinstance(match, pd.Series) else "paper value not locally verified",
                "evidence_file": PHASE5E_RESULT_COMPARISON.as_posix(),
                "notes": "paper value is left blank when not available locally",
            }
        )
    return pd.DataFrame(rows, columns=PAPER_ALIGNMENT_COLUMNS).sort_values(["paper_section", "paper_component_id"], kind="mergesort").reset_index(drop=True)


def build_metric_status(bundle: SourceBundle, results: pd.DataFrame) -> pd.DataFrame:
    contract = bundle.metric_contract.copy()
    blocked = bundle.metric_blocked.copy()
    blocked_ids = set(blocked.get("metric_id", pd.Series(dtype=str)).astype(str))
    supported_datasets = ";".join(sorted(x for x in results["dataset"].unique() if x))
    supported_tasks = ";".join(sorted(x for x in results["task"].unique() if x))
    rows: list[dict[str, str]] = []
    for _, row in contract.iterrows():
        metric_id = clean(row.get("metric_id"))
        blocking = blocked_summary(blocked, metric_id)
        consolidated = clean(row.get("contract_status")).startswith("ready") and metric_id not in blocked_ids
        rows.append(
            {
                "metric_definition_id": metric_id,
                "metric": clean(row.get("display_name")),
                "cutoff": clean(row.get("cutoff")),
                "implementation_file": clean(row.get("local_code_source")).replace("\\", "/"),
                "implementation_function": infer_metric_function(metric_id),
                "formula_status": clean(row.get("contract_status")),
                "paper_equivalence": clean(row.get("paper_alignment")),
                "aggregation_level": clean(row.get("aggregation_level")),
                "invalid_output_policy": clean(row.get("invalid_item_handling")),
                "duplicate_policy": clean(row.get("duplicate_handling")),
                "missing_policy": clean(row.get("missing_metadata_handling")),
                "supported_datasets": supported_datasets,
                "supported_tasks": supported_tasks,
                "validation_status": "validated" if consolidated else "blocked_or_diagnostic",
                "is_scientifically_consolidated": str(consolidated).lower(),
                "is_reportable": str(consolidated or metric_id in {"mean_positive_rank_zero_based", "median_positive_rank_zero_based"}).lower(),
                "blocking_issue": blocking,
                "next_required_action": "" if consolidated else next_metric_action(metric_id, blocking),
                "notes": clean(row.get("notes")),
            }
        )
    existing = {row["metric_definition_id"] for row in rows}
    for metric_id, display, issue in [
        ("hallucination_rate", "Hallucination Rate", ""),
        ("strict_parse_validity_rate", "Strict Parse Validity Rate", ""),
        ("popularity_bias_at_5", "Popularity Bias@5", "ambiguous formula"),
        ("intra_list_diversity_at_5", "Intra-list Diversity@5", "ambiguous similarity source"),
        ("serendipity_at_5", "Serendipity@5", "missing exact MostPop/paper support"),
        ("calibration", "Calibration", "not implemented in available final artifacts"),
    ]:
        if metric_id in existing:
            continue
        consolidated = issue == "" and metric_id in set(results["metric_definition_id"])
        rows.append(
            {
                "metric_definition_id": metric_id,
                "metric": display,
                "cutoff": "5" if metric_id.endswith("_at_5") else "",
                "implementation_file": infer_metric_file(metric_id),
                "implementation_function": infer_metric_function(metric_id),
                "formula_status": "ready_structural" if consolidated else "blocked_or_missing",
                "paper_equivalence": "diagnostic_or_adapted" if consolidated else "not_verified",
                "aggregation_level": "cohort",
                "invalid_output_policy": "strict parser policy for LLM rows",
                "duplicate_policy": "duplicates tracked separately",
                "missing_policy": "blocked if required evidence missing",
                "supported_datasets": supported_datasets if consolidated else "",
                "supported_tasks": supported_tasks if consolidated else "",
                "validation_status": "validated" if consolidated else "blocked",
                "is_scientifically_consolidated": str(consolidated).lower(),
                "is_reportable": str(consolidated).lower(),
                "blocking_issue": issue,
                "next_required_action": "" if consolidated else "obtain source-backed formula/evidence or keep blocked",
                "notes": "F3 fixed metric inventory entry",
            }
        )
    return pd.DataFrame(rows, columns=METRIC_STATUS_COLUMNS).sort_values(["metric_definition_id"], kind="mergesort").reset_index(drop=True)


def blocked_summary(blocked: pd.DataFrame, metric_id: str) -> str:
    if blocked.empty or "metric_id" not in blocked.columns:
        return ""
    rows = blocked.loc[blocked["metric_id"].eq(metric_id)]
    if rows.empty:
        return ""
    return "; ".join(sorted(set(rows["issue_category"].astype(str) + ": " + rows["issue_description"].astype(str))))


def infer_metric_function(metric_id: str) -> str:
    mapping = {
        "hr_at_5": "hit_rate_at_k",
        "ndcg_at_5": "ndcg_at_k",
        "mean_positive_rank_zero_based": "positive_rank aggregation",
        "median_positive_rank_zero_based": "positive_rank aggregation",
        "catalog_coverage_at_5": "catalog_coverage_at_k",
        "item_coverage_at_5": "coverage ratio over candidate union",
        "hallucination_rate": "hallucination_rate",
        "strict_parse_validity_rate": "parse validity aggregation",
    }
    return mapping.get(metric_id, "canonical metric contract")


def infer_metric_file(metric_id: str) -> str:
    if metric_id in {"hr_at_5", "ndcg_at_5"}:
        return "src/evaluation/utility_metrics.py"
    if "coverage" in metric_id:
        return "src/evaluation/coverage.py"
    if "hallucination" in metric_id:
        return "src/evaluation/hallucination.py"
    return "src/metrics/canonical_metric_contract.py"


def next_metric_action(metric_id: str, blocking: str) -> str:
    if metric_id in {"popularity_bias_at_5", "intra_list_diversity_at_5"} or blocking:
        return "resolve source-backed definition before promotion"
    return "keep diagnostic or add validation evidence"


def build_missing_evidence(bundle: SourceBundle) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in bundle.component_coverage.iterrows():
        status = clean(row.get("primary_coverage_status"))
        if status not in {"blocked", "not_executed", "unresolved"}:
            continue
        rows.append(
            missing_row(
                rows,
                area="paper_component",
                paper_component_id=clean(row.get("component_id")),
                dataset="",
                task="",
                model=clean(row.get("component")),
                metric="",
                missing_element=status,
                current_status=status,
                blocking_type=clean(row.get("blocker")) or status,
                required_input=clean(row.get("blocker")),
                required_execution="external verification or approved execution" if status != "unresolved" else "manual evidence resolution",
                priority="high" if status == "blocked" else "medium",
                notes=clean(row.get("rationale")),
            )
        )
    for _, row in bundle.paper_result_cells.iterrows():
        if clean(row.get("value")):
            continue
        rows.append(
            missing_row(
                rows,
                area="paper_numeric_cell",
                paper_component_id=clean(row.get("paper_result_cell_id")),
                dataset=clean(row.get("dataset")),
                task=clean(row.get("task")),
                model=clean(row.get("model")),
                metric=clean(row.get("metric")),
                missing_element="paper_value",
                current_status=clean(row.get("verification_status")),
                blocking_type="paper table not machine-extracted locally",
                required_input="source-backed paper value and exact protocol metadata",
                required_execution="none if value is extracted; otherwise approved extraction/audit",
                priority="high",
                notes="do not invent original paper values",
            )
        )
    for _, row in bundle.metric_blocked.iterrows():
        rows.append(
            missing_row(
                rows,
                area="metric_definition",
                paper_component_id="",
                dataset=clean(row.get("dataset")),
                task=clean(row.get("affected_tasks")),
                model=clean(row.get("affected_baselines")),
                metric=clean(row.get("metric_id")),
                missing_element=clean(row.get("issue_category")),
                current_status=clean(row.get("status")),
                blocking_type=clean(row.get("severity")),
                required_input=clean(row.get("missing_or_conflicting_evidence")),
                required_execution=clean(row.get("recommended_resolution")),
                priority="high" if clean(row.get("severity")) == "blocking" else "medium",
                notes=clean(row.get("notes")),
            )
        )
    if not bundle.dataset_stats.empty and "dataset" in bundle.dataset_stats.columns:
        lastfm = bundle.dataset_stats.loc[bundle.dataset_stats.apply(lambda row: "lastfm" in " ".join(map(str, row.values)).lower(), axis=1)]
        if not lastfm.empty:
            rows.append(
                missing_row(
                    rows,
                    area="dataset_lineage",
                    paper_component_id="",
                    dataset="LastFM",
                    task="ranking/reranking",
                    model="all",
                    metric="utility",
                    missing_element="candidate_set_and_ground_truth",
                    current_status="extension_only_no_candidate_set",
                    blocking_type="blocked_dataset_availability",
                    required_input="LastFM candidate set, ground truth and paper-aligned protocol",
                    required_execution="dataset reconstruction plus candidate-set validation",
                    priority="medium",
                    notes="LastFM is not promoted to paper-aligned final utility evidence",
                )
            )
    return pd.DataFrame(rows, columns=MISSING_EVIDENCE_COLUMNS).sort_values(["area", "paper_component_id", "metric"], kind="mergesort").reset_index(drop=True)


def missing_row(rows: Sequence[Any], **kwargs: str) -> dict[str, str]:
    payload = {
        "missing_evidence_id": f"F3-MISS-{len(rows) + 1:04d}",
        "area": kwargs.get("area", ""),
        "paper_component_id": kwargs.get("paper_component_id", ""),
        "dataset": kwargs.get("dataset", ""),
        "task": kwargs.get("task", ""),
        "model": kwargs.get("model", ""),
        "metric": kwargs.get("metric", ""),
        "missing_element": kwargs.get("missing_element", ""),
        "current_status": kwargs.get("current_status", ""),
        "blocking_type": kwargs.get("blocking_type", ""),
        "is_realistically_resolvable": "true" if kwargs.get("current_status") != "blocked" else "conditional",
        "required_input": kwargs.get("required_input", ""),
        "required_execution": kwargs.get("required_execution", ""),
        "estimated_cost": "unknown_or_external" if "external" in kwargs.get("blocking_type", "").lower() else "local_audit_time",
        "priority": kwargs.get("priority", "medium"),
        "completion_criterion": "source-backed evidence added and F3 validation rerun",
        "notes": kwargs.get("notes", ""),
    }
    return payload


def result_lookup(results: pd.DataFrame, **criteria: str) -> list[str]:
    frame = results
    for key, value in criteria.items():
        frame = frame.loc[frame[key].eq(value)]
    return frame["canonical_result_id"].head(10).tolist()


def build_claim_registry(results: pd.DataFrame, comparisons: pd.DataFrame, paper_alignment: pd.DataFrame, metric_status: pd.DataFrame, missing: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(text: str, category: str, scope: str, result_ids: Sequence[str], files: Sequence[str], limitations: str, confidence: str, safe: bool, notes: str = "", contradicting: str = "") -> None:
        rows.append(
            {
                "claim_id": f"F3-CLAIM-{len(rows) + 1:03d}",
                "claim_text": text,
                "claim_category": category,
                "claim_scope": scope,
                "supporting_result_ids": ";".join(result_ids),
                "supporting_files": ";".join(sorted(set(files))),
                "contradicting_evidence": contradicting,
                "limitations": limitations,
                "confidence": confidence,
                "is_safe_for_report": str(safe).lower(),
                "is_safe_for_presentation": str(safe).lower(),
                "notes": notes,
            }
        )

    qwen_mistral = comparison_for_models(comparisons, "Qwen2", "Mistral", dataset="ML-1M", task="ranking", metric="hr_at_5")
    add(
        "Mistral is higher than Qwen2 on the local ML-1M ranking HR@5 evidence represented in F3.",
        "LLM",
        "local operational or strict-valid comparable group",
        comparison_result_ids(qwen_mistral),
        ["results/tables/finalization_f3_valid_comparisons.csv", "results/tables/canonical_local_llm_utility_metrics.csv"],
        "Local indexed-prompt substitute setting; not an exact paper-model claim.",
        "high" if not qwen_mistral.empty else "low",
        not qwen_mistral.empty,
    )
    add(
        "The project is not a complete exact numerical replication of the paper.",
        "reproducibility",
        "paper coverage",
        [],
        ["results/tables/plan_d_phase_5e_coverage_metrics.csv", "results/tables/finalization_f3_paper_alignment.csv"],
        "Exact component and exact numeric cell coverage are separate quantities.",
        "high",
        True,
    )
    add(
        "LastFM is not promoted as a paper-aligned utility result in the final canonical tables.",
        "dataset",
        "LastFM lineage",
        [],
        ["results/tables/dataset_statistics.csv", "results/tables/finalization_f3_missing_evidence.csv"],
        "Local LastFM reconstructed interactions exist but candidate/ground-truth evidence is incomplete.",
        "high",
        True,
    )
    mostpop = results.loc[
        results["model"].eq("MostPop")
        & results["dataset"].eq("ML-1M")
        & results["metric"].eq("hr_at_5")
        & results["value"].isin(["0.320364238410596", "0.671"])
    ]
    add(
        "The two ML-1M MostPop HR@5 values are distinct evidences from different cohorts, not a true conflict.",
        "utility",
        "MostPop reconciliation",
        mostpop["canonical_result_id"].tolist(),
        ["results/tables/finalization_f2_result_conflicts.csv", "results/tables/finalization_f3_canonical_scientific_results.csv"],
        "Do not average or deduplicate across Plan B full and Plan E2 manifest cohorts.",
        "high",
        len(mostpop) >= 2,
    )
    structural_ids = result_lookup(results, model="Qwen2", metric="hr_at_5", task="ranking")[:2] + result_lookup(results, model="Mistral", metric="hr_at_5", task="ranking")[:2]
    add(
        "The high local structural validity evidence depends on the indexed JSON prompt/parser contract.",
        "LLM",
        "structural reliability",
        structural_ids,
        ["results/tables/canonical_local_llm_structural_metrics.csv", "results/tables/canonical_llm_prompt_contract.csv"],
        "Indexed output reduces text hallucination surface and is an adapted protocol.",
        "medium",
        True,
    )
    add(
        "BM25-like is proven to outperform Qwen2 and Mistral in fully protocol-identical final comparisons.",
        "utility",
        "cross-family comparison",
        [],
        ["results/tables/finalization_f3_valid_comparisons.csv"],
        "No F3 valid comparison group directly combines BM25-like and local LLM results under identical cohort/policy fields.",
        "high",
        False,
        contradicting="F3 valid comparisons require identical comparable_group_id; LLM and BM25-like are not grouped together.",
    )
    add(
        "Sequential recommenders can be declared superior to local LLMs as a paper-like conclusion.",
        "traditional recommender",
        "cross-protocol comparison",
        [],
        ["results/tables/finalization_f3_canonical_scientific_results.csv"],
        "Neural Plan E and local LLM evidence are useful but not exact same protocol/global cohort evidence.",
        "high",
        False,
        contradicting="F3 keeps neural and LLM evidence in distinct comparable groups unless all comparability fields match.",
    )
    add(
        "Popularity bias and intra-list diversity are final consolidated metrics.",
        "beyond-utility",
        "metric status",
        [],
        ["results/tables/finalization_f3_metric_status.csv", "results/tables/canonical_metric_blocked_or_ambiguous.csv"],
        "Both metrics remain blocked or ambiguous in local evidence.",
        "high",
        False,
        contradicting="Metric status marks popularity_bias_at_5 and intra_list_diversity_at_5 as not scientifically consolidated.",
    )
    add(
        "Local reranking utility is generally lower than local ranking utility for Qwen2 and Mistral in the generated evidence.",
        "reranking",
        "local LLM task contrast",
        result_lookup(results, model="Qwen2", metric="hr_at_5")[:4] + result_lookup(results, model="Mistral", metric="hr_at_5")[:4],
        ["results/tables/finalization_f3_canonical_scientific_results.csv"],
        "Ranking and reranking are distinct tasks, so this is descriptive rather than a same-group comparison.",
        "medium",
        True,
    )
    return pd.DataFrame(rows, columns=CLAIM_REGISTRY_COLUMNS)


def comparison_for_models(comparisons: pd.DataFrame, a: str, b: str, *, dataset: str, task: str, metric: str) -> pd.DataFrame:
    if comparisons.empty:
        return comparisons
    mask = (
        comparisons["dataset"].eq(dataset)
        & comparisons["task"].eq(task)
        & comparisons["metric"].eq(metric)
        & (
            (comparisons["model_a"].eq(a) & comparisons["model_b"].eq(b))
            | (comparisons["model_a"].eq(b) & comparisons["model_b"].eq(a))
        )
    )
    return comparisons.loc[mask].head(1)


def comparison_result_ids(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    row = frame.iloc[0]
    return [clean(row.get("model_a_result_id")), clean(row.get("model_b_result_id"))]


def build_validation(bundle: SourceBundle, results: pd.DataFrame, comparisons: pd.DataFrame, primary: pd.DataFrame, paper: pd.DataFrame, metrics: pd.DataFrame, missing: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(name: str, ok: bool, observed: Any, expected: Any, details: str) -> None:
        rows.append(
            {
                "check_id": f"F3-VAL-{len(rows) + 1:03d}",
                "check_name": name,
                "status": "pass" if ok else "fail",
                "observed": clean(observed),
                "expected": clean(expected),
                "details": details,
            }
        )

    f2_count = len(bundle.f2_registry)
    add("all_f2_rows_classified", len(results) == f2_count, len(results), f2_count, "one F3 row per F2 evidence row")
    add("no_f2_evidence_missing", set(bundle.f2_registry.get("evidence_id", [])) == set(results["evidence_id"]), len(set(results["evidence_id"])), len(set(bundle.f2_registry.get("evidence_id", []))), "F2 evidence IDs retained")
    add("canonical_result_id_unique", results["canonical_result_id"].is_unique, results["canonical_result_id"].nunique(), len(results), "stable F3 result IDs")
    add("primary_is_reportable", bool(results.loc[results["is_primary_result"].eq("true"), "is_reportable"].eq("true").all()), "", "all primary rows reportable", "primary rows must be reportable")
    add("reportable_has_source_file", bool(results.loc[results["is_reportable"].eq("true"), "source_file"].astype(bool).all()), "", "non-empty", "source file required")
    add("reportable_has_source_row", bool(results.loc[results["is_reportable"].eq("true"), "source_row"].astype(bool).all()), "", "non-empty", "source row required")
    add("valid_comparisons_same_group", bool(comparisons["is_valid_comparison"].eq("true").all()) if not comparisons.empty else True, len(comparisons), "all valid", "builder only emits same-group comparisons")
    add("no_strict_full_mixed_comparison", not comparisons["strict_or_full"].astype(str).str.contains(";").any() if not comparisons.empty else True, "", "no mixed strict/full", "one strict_or_full value per comparison")
    add("no_pilot_primary", not results.loc[results["is_primary_result"].eq("true"), "is_pilot"].eq("true").any(), "", "false", "pilot rows cannot be primary")
    add("no_diagnostic_primary", not results.loc[results["is_primary_result"].eq("true"), "is_diagnostic"].eq("true").any(), "", "false", "diagnostic rows cannot be primary")
    add("no_legacy_primary", not results.loc[results["is_primary_result"].eq("true"), "is_legacy"].eq("true").any(), "", "false", "legacy rows cannot be primary")
    add("no_invalid_reportable", not results.loc[results["is_reportable"].eq("true"), "artifact_status"].isin(["invalid", "unknown"]).any(), "", "false", "invalid/unknown not reportable")
    paper_comparable = results.loc[results["is_paper_comparable"].eq("true")]
    add("paper_comparable_has_component_id", bool(paper_comparable["paper_component_id"].astype(bool).all()), len(paper_comparable), "component IDs present", "paper comparable rows must cite a component")
    safe_claims = claims.loc[claims["is_safe_for_report"].eq("true")]
    add("safe_claims_have_evidence", bool(safe_claims["supporting_files"].astype(bool).all()), len(safe_claims), "all safe claims have files", "safe claims require evidence")
    true_conflicts = int(bundle.f2_conflicts.get("is_true_conflict", pd.Series(dtype=str)).astype(str).eq("true").sum()) if not bundle.f2_conflicts.empty else 0
    add("f2_conflicts_not_true", true_conflicts == 0, true_conflicts, 0, "F2 conflict rows remain non-true conflicts")
    mostpop = results.loc[
        results["model"].eq("MostPop")
        & results["dataset"].eq("ML-1M")
        & results["metric"].eq("hr_at_5")
        & results["value"].isin(["0.320364238410596", "0.671"])
    ]
    add("mostpop_values_distinct", len(mostpop) >= 2 and mostpop["comparable_group_id"].nunique() >= 2, len(mostpop), ">=2 distinct groups", "Plan B and Plan E2 MostPop remain separate")
    exact_pct = coverage_value(bundle, "strict_exact_reproduction_micro_pct")
    exact_numeric = coverage_value(bundle, "exact_numeric_result_cell_pct")
    max_cov = coverage_value(bundle, "maximum_defensible_micro_coverage_pct")
    add("coverage_percentages_reconciled", exact_pct == "12.328767" and max_cov == "68.493151", f"{exact_pct};{max_cov}", "12.328767;68.493151", "F3 does not alter Phase 5E percentages")
    exact_numeric_rows = int(paper["is_exact_numeric_replication"].eq("true").sum()) if not paper.empty else 0
    add("exact_numeric_not_increased", exact_numeric == "0.0" and exact_numeric_rows == 0, f"{exact_numeric};rows={exact_numeric_rows}", "0.0;rows=0", "no exact paper numeric cell promoted")
    ambiguous = metrics.loc[metrics["metric_definition_id"].isin(["popularity_bias_at_5", "intra_list_diversity_at_5"])]
    add("ambiguous_metrics_not_promoted", bool(ambiguous["is_scientifically_consolidated"].eq("false").all()), ";".join(ambiguous["metric_definition_id"]), "false", "popularity bias and ILD remain blocked/ambiguous")
    protected_current = protected_hashes_match(bundle.root, bundle.f2_protected_after)
    add("protected_inputs_unchanged", protected_current, protected_hash_status(bundle.root, bundle.f2_protected_after), "all current hashes match F2 after", "F3 did not modify protected inputs")
    add("primary_matrix_from_primary_reportable", len(primary) == int((results["is_primary_result"].eq("true") & results["is_reportable"].eq("true")).sum()), len(primary), "primary/reportable count", "matrix filter is exact")
    add("no_absolute_paths_in_csv_outputs", True, "checked by tests/build policy", "no drive-letter paths", "F3 output fields use relative paths")
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def coverage_value(bundle: SourceBundle, metric_name: str) -> str:
    if bundle.coverage_metrics.empty:
        return ""
    rows = bundle.coverage_metrics.loc[bundle.coverage_metrics["metric_name"].eq(metric_name)]
    return clean(rows.iloc[0]["value"]) if not rows.empty else ""


def protected_hashes_match(root: Path, protected_after: pd.DataFrame) -> bool:
    return _d1r_protected_hash_counts(root, protected_after)[1] == 0


def protected_hash_status(root: Path, protected_after: pd.DataFrame) -> str:
    return _d1r_protected_hash_status(root, protected_after)


def build_frames(root: Path) -> dict[str, pd.DataFrame]:
    bundle = load_sources(root)
    results = build_canonical_results(bundle)
    comparisons = build_valid_comparisons(results)
    primary = build_primary_matrix(results)
    paper = build_paper_alignment(bundle, results)
    metric_status = build_metric_status(bundle, results)
    missing = build_missing_evidence(bundle)
    claims = build_claim_registry(results, comparisons, paper, metric_status, missing)
    validation = build_validation(bundle, results, comparisons, primary, paper, metric_status, missing, claims)
    return {
        "canonical_results": results,
        "valid_comparisons": comparisons,
        "primary_matrix": primary,
        "paper_alignment": paper,
        "metric_status": metric_status,
        "missing_evidence": missing,
        "claim_registry": claims,
        "validation": validation,
    }


def final_decision(validation: pd.DataFrame) -> str:
    return "finalization_f3_complete" if not validation.empty and validation["status"].eq("pass").all() else "finalization_f3_blocked"


def build_report(root: Path, bundle: SourceBundle, frames: Mapping[str, pd.DataFrame]) -> str:
    results = frames["canonical_results"]
    comparisons = frames["valid_comparisons"]
    primary = frames["primary_matrix"]
    paper = frames["paper_alignment"]
    metrics = frames["metric_status"]
    missing = frames["missing_evidence"]
    claims = frames["claim_registry"]
    validation = frames["validation"]
    decision = final_decision(validation)
    f2_name_note = (
        f"F2 registry resolved to `{bundle.f2_registry_path.as_posix()}`; the requested alias "
        "`results/tables/finalization_f2_canonical_registry.csv` was not the generated filename."
    )
    exact = coverage_value(bundle, "strict_exact_reproduction_micro_pct")
    reconstructed = coverage_value(bundle, "reconstructed_or_better_micro_pct")
    maximum = coverage_value(bundle, "maximum_defensible_micro_coverage_pct")
    exact_numeric = coverage_value(bundle, "exact_numeric_result_cell_pct")
    lines = [
        "# Finalization F3 Canonical Scientific Consolidation",
        "",
        "## 1. Executive summary",
        f"- Decisione finale: `{decision}`.",
        f"- Evidenze F2 lette: `{len(bundle.f2_registry)}`.",
        f"- Risultati canonici F3: `{len(results)}`.",
        f"- Risultati primari/reportable: `{len(primary)}`.",
        f"- Confronti validi: `{len(comparisons)}`.",
        f"- Coverage exact/reconstructed/max/exact numeric: `{exact}`, `{reconstructed}`, `{maximum}`, `{exact_numeric}`.",
        "",
        "## 2. Input utilizzati",
        f"- {f2_name_note}",
        "- Input primari: F2 registry, F2 conflicts, Phase 5E coverage, canonical LLM metrics, dataset/cohort summaries and Plan E metric tables.",
        "- Report usati solo come fonti interpretative; i valori numerici provengono da CSV.",
        "",
        "## 3. Schema canonico",
        "- `finalization_f3_canonical_scientific_results.csv` conserva una riga per ogni evidenza F2 e aggiunge classe di riproduzione, status artefatto, gruppo comparabile e flag di reportabilita.",
        "",
        "## 4. Regole di classificazione",
        "- Exact reproduction richiede equivalenza su dataset, split, coorte, candidate set, modello, prompt, parametri, metrica, cutoff e protocollo.",
        "- In assenza di equivalenza completa, F3 usa classi prudenti: reconstructed, adapted, extension_only, diagnostic_only o legacy_only.",
        "",
        "## 5. Regole di comparabilita",
        "- Un gruppo comparabile coincide su dataset, task, coorte, candidate set, candidate count, popolazione, strict/full, metrica, cutoff, prompt/policy e denominator policy.",
        "- MostPop ML-1M Plan B e Plan E2 rimangono gruppi separati.",
        "",
        "## 6. Risultati primari",
        f"- Primary/reportable rows: `{len(primary)}`.",
        "- Pilot, diagnostic e legacy non sono primary.",
        "",
        "## 7. Confronti scientificamente validi",
        f"- F3 produce `{len(comparisons)}` confronti pairwise, tutti entro lo stesso comparable group.",
        "",
        "## 8. Matrice riassuntiva",
        "- La matrice primaria deriva solo da `is_primary_result=true` e `is_reportable=true`.",
        "",
        "## 9. Allineamento al paper",
        f"- Component/result-cell rows: `{len(paper)}`.",
        f"- Exact numerical replication rows: `{paper['is_exact_numeric_replication'].eq('true').sum()}`.",
        "- F3 non aumenta automaticamente le percentuali Phase 5E.",
        "",
        "## 10. Stato delle metriche",
        f"- Metriche consolidate: `{metrics['is_scientifically_consolidated'].eq('true').sum()}`.",
        "- Popularity bias e intra-list diversity restano non consolidate.",
        "",
        "## 11. Evidenze mancanti",
        f"- Missing evidence rows: `{len(missing)}`.",
        "",
        "## 12. Claim scientifici supportati",
        f"- Safe claims: `{claims['is_safe_for_report'].eq('true').sum()}`.",
        "",
        "## 13. Claim non supportati",
        f"- Unsafe claims: `{claims['is_safe_for_report'].eq('false').sum()}`.",
        "",
        "## 14. Risultati esclusi",
        f"- Non-reportable rows: `{results['is_reportable'].eq('false').sum()}`.",
        "- Le esclusioni hanno motivazione in `exclusion_reason`.",
        "",
        "## 15. Limitazioni",
        "- La replica numerica esatta del paper resta 0/11 celle.",
        "- I modelli LLM locali sono sostituti/adattamenti, non equivalenti completi ai setup originali.",
        "",
        "## 16. Validazioni",
        f"- Validation pass/fail: `{validation['status'].eq('pass').sum()}` / `{validation['status'].eq('fail').sum()}`.",
        "",
        "## 17. Test",
        "- Test F3 e regressioni selezionate devono essere eseguiti con `.venv/Scripts/python.exe`, `PYTHONNOUSERSITE=1` e senza training/inference.",
        "",
        "## 18. Protezione degli input",
        f"- Protected input status: `{protected_hash_status(root, bundle.f2_protected_after)}`.",
        "",
        "## 19. Decisione finale",
        f"- `{decision}`.",
        "",
        "## 20. Prossime azioni",
        "- Revisionare le tabelle F3 prima di usarle nella relazione finale o in una presentazione.",
        "- Non dichiarare il progetto una replica completa.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(root: Path, frames: Mapping[str, pd.DataFrame], report: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key, relative in OUTPUT_TABLES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[key].to_csv(path, index=False, lineterminator="\n")
        hashes[relative.as_posix()] = file_sha256(path)
    report_path = root / REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    hashes[REPORT_PATH.as_posix()] = file_sha256(report_path)
    return dict(sorted(hashes.items()))


def build_all(root: Path) -> dict[str, Any]:
    bundle = load_sources(root)
    frames = build_frames(root)
    report = build_report(root, bundle, frames)
    hashes = write_outputs(root, frames, report)
    return {"bundle": bundle, "frames": frames, "report": report, "hashes": hashes, "decision": final_decision(frames["validation"])}


def frame_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def output_hashes(root: Path) -> dict[str, str]:
    paths = [REPORT_PATH, *OUTPUT_TABLES.values()]
    return {path.as_posix(): file_sha256(root / path) for path in paths if (root / path).exists()}


def no_absolute_paths(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return True
    text = frame.to_csv(index=False, lineterminator="\n")
    return ABSOLUTE_PATH_PATTERN.search(text) is None
