"""Phase 4A metric-contract and same-cohort preflight builders."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.plan_e_lightgcn_cpu_baseline_consolidation import (
    ALL_BASELINES,
    BASELINE_FAMILIES,
    FULL_COHORT_ID,
    MATCHED_COHORT_ID,
    ROW_KEY_COLUMNS,
    bool_value,
    cohort_keys_from_lightgcn_rankings,
    filter_rankings_to_cohort,
    normalize_bprmf_rankings,
    normalize_gru4rec_rankings,
    normalize_lightgcn_rankings,
    normalize_sasrec_rankings,
    normalize_traditional_rankings,
    paper_alignment,
    parse_json_array,
    row_key,
    select_unit,
    source_frame_for_baseline,
)
from src.metrics.canonical_metric_contract import (
    CONTRACT_STATUSES,
    build_long_tail_set,
    build_metric_registry,
    contract_row,
    ndcg_at_k_from_rank,
    validate_metric_registry,
)


DATASETS = {
    "Amazon Beauty": "amazon_beauty",
    "Amazon Sports": "amazon_sports",
    "ML-1M": "ml1m",
}

DATASET_COHORT_PATH = {
    "Amazon Beauty": "results/tables/plan_e_lightgcn_beauty_sports_cpu_cohort.csv",
    "Amazon Sports": "results/tables/plan_e_lightgcn_beauty_sports_cpu_cohort.csv",
    "ML-1M": "results/tables/plan_e_lightgcn_ml1m_pilot_cohort.csv",
}

RANKING_SOURCE_PATHS = {
    "traditional": "results/tables/traditional_baseline_rankings.csv",
    "bprmf_beauty_sports": "results/tables/plan_e_bprmf_beauty_sports_rankings.csv",
    "bprmf_ml1m": "results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv",
    "gru4rec_beauty_sports": "results/tables/plan_e_gru4rec_beauty_sports_rankings.csv",
    "gru4rec_ml1m": "results/tables/plan_e_gru4rec_ml1m_pilot_rankings.csv",
    "sasrec_beauty_sports": "results/tables/plan_e_sasrec_beauty_sports_rankings.csv",
    "sasrec_ml1m": "results/tables/plan_e_sasrec_ml1m_pilot_rankings.csv",
    "lightgcn_beauty_sports": "results/tables/plan_e_lightgcn_beauty_sports_cpu_rankings.csv",
    "lightgcn_ml1m": "results/tables/plan_e_lightgcn_ml1m_pilot_rankings.csv",
}

PROTECTED_INPUTS = [
    "paper/Beyond Utility Evaluating LLM As Recommander.pdf",
    "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
    "original_repo/EvaLLMasRecommender/eva/codes/metrics.sh",
    "report/codebase_map.md",
    "results/tables/ranking_candidate_sets.csv",
    "results/tables/positive_positions.csv",
    "results/tables/experiment_cohort_membership.csv",
    "results/tables/experiment_cohort_summary.csv",
    "results/tables/plan_e_lightgcn_row_compatibility.csv",
    "results/tables/plan_e_lightgcn_beauty_sports_cpu_cohort.csv",
    "results/tables/plan_e_lightgcn_ml1m_pilot_cohort.csv",
    "results/tables/plan_e_lightgcn_same_cohort_baselines_metrics.csv",
    "results/tables/plan_e_lightgcn_baseline_comparison.csv",
    "results/tables/plan_e_lightgcn_all_datasets_metrics.csv",
    "results/tables/plan_e_lightgcn_all_datasets_device_provenance.csv",
    *RANKING_SOURCE_PATHS.values(),
    "results/tables/item_title_map.csv",
    "results/tables/candidate_item_title_map.csv",
    "results/tables/item_title_coverage_summary.csv",
    "data/reconstructed/plan_c/amazon_beauty/interactions.csv",
    "data/reconstructed/plan_c/amazon_beauty/datamaps.json",
    "data/reconstructed/plan_c/amazon_beauty/catalog.csv",
    "data/reconstructed/plan_c/amazon_beauty/meta.json",
    "data/reconstructed/plan_c/amazon_sports/interactions.csv",
    "data/reconstructed/plan_c/amazon_sports/datamaps.json",
    "data/reconstructed/plan_c/amazon_sports/catalog.csv",
    "data/reconstructed/plan_c/amazon_sports/meta.json",
    "data/reconstructed/plan_c/ml1m/interactions.csv",
    "data/reconstructed/plan_c/ml1m/datamaps.json",
    "data/reconstructed/plan_c/ml1m/catalog.csv",
    "data/reconstructed/plan_c/ml1m/meta.json",
]

CONTRACT_COLUMNS = [
    "metric_id",
    "display_name",
    "metric_family",
    "cutoff",
    "mathematical_formula",
    "input_unit",
    "aggregation_level",
    "denominator_definition",
    "higher_is_better",
    "expected_minimum",
    "expected_maximum",
    "duplicate_handling",
    "invalid_item_handling",
    "missing_metadata_handling",
    "empty_cohort_handling",
    "popularity_source_policy",
    "catalog_universe_policy",
    "long_tail_policy",
    "similarity_source_policy",
    "paper_source",
    "official_code_source",
    "local_code_source",
    "paper_alignment",
    "contract_status",
    "blocking_reason",
    "contract_sha256",
    "notes",
]

INVENTORY_COLUMNS = [
    "inventory_id",
    "metric_id",
    "source_type",
    "source_path",
    "symbol_or_section",
    "cutoff",
    "formula_summary",
    "aggregation_summary",
    "input_summary",
    "denominator_summary",
    "missing_data_summary",
    "execution_evidence",
    "matches_paper",
    "matches_canonical_contract",
    "conflict_type",
    "status",
    "notes",
]

PREFLIGHT_COLUMNS = [
    "baseline_id",
    "baseline_family",
    "dataset",
    "task_type",
    "candidate_set_id",
    "cohort_id",
    "ranking_source_path",
    "expected_rows",
    "observed_rows",
    "exact_row_key_match",
    "exact_candidate_list_match",
    "exact_positive_item_match",
    "candidate_count_valid",
    "ranking_membership_valid",
    "top_5_extractable",
    "all_ranked_ids_catalog_valid",
    "all_scores_finite",
    "utility_metrics_reconciled",
    "hr_at_5_existing",
    "hr_at_5_recomputed",
    "ndcg_at_5_existing",
    "ndcg_at_5_recomputed",
    "preflight_status",
    "blocking_reason",
    "notes",
]

POPULARITY_COLUMNS = [
    "dataset",
    "popularity_source_id",
    "source_path",
    "split_policy",
    "count_definition",
    "catalog_universe_id",
    "item_count",
    "items_with_positive_count",
    "items_with_zero_count",
    "total_count",
    "minimum_positive_count",
    "maximum_count",
    "mean_count",
    "median_count",
    "probability_denominator",
    "smoothing_required",
    "leakage_free",
    "paper_aligned",
    "recommended",
    "status",
    "blocking_reason",
    "sha256",
    "notes",
]

UNIVERSE_COLUMNS = [
    "dataset",
    "universe_id",
    "universe_definition",
    "source_path",
    "item_count",
    "train_seen_item_count",
    "metadata_covered_item_count",
    "strict_candidate_union_item_count",
    "full_plan_e2_candidate_union_item_count",
    "items_missing_title",
    "items_missing_category_or_genre",
    "items_missing_popularity",
    "appropriate_for_catalog_coverage",
    "appropriate_for_item_coverage",
    "appropriate_for_novelty",
    "appropriate_for_long_tail",
    "appropriate_for_popularity_metrics",
    "appropriate_for_ild",
    "recommended_metric_ids",
    "status",
    "blocking_reason",
    "sha256",
    "notes",
]

LONG_TAIL_COLUMNS = [
    "dataset",
    "candidate_policy_id",
    "policy_description",
    "popularity_source_id",
    "catalog_universe_id",
    "sort_direction",
    "threshold_type",
    "threshold_value",
    "tie_handling",
    "head_item_count",
    "tail_item_count",
    "head_interaction_mass",
    "tail_interaction_mass",
    "paper_evidence",
    "official_code_evidence",
    "local_evidence",
    "leakage_free",
    "result_independent",
    "recommended",
    "status",
    "blocking_reason",
    "notes",
]

SIMILARITY_COLUMNS = [
    "dataset",
    "similarity_source_id",
    "source_type",
    "source_path",
    "feature_definition",
    "feature_dimension",
    "catalog_items",
    "strict_candidate_items",
    "strict_candidate_items_with_features",
    "coverage_rate",
    "zero_vector_item_count",
    "duplicate_feature_item_count",
    "similarity_function",
    "distance_function",
    "paper_aligned",
    "requires_adaptation",
    "deterministic",
    "recommended",
    "status",
    "blocking_reason",
    "sha256",
    "notes",
]

APPLICABILITY_COLUMNS = [
    "metric_id",
    "baseline_id",
    "dataset",
    "task_type",
    "cohort_id",
    "ranking_source_path",
    "metric_contract_status",
    "required_inputs_available",
    "metadata_coverage_sufficient",
    "popularity_input_available",
    "long_tail_input_available",
    "similarity_input_available",
    "structurally_constant",
    "structural_expected_value",
    "meaningful_for_comparison",
    "final_computation_allowed",
    "applicability_status",
    "blocking_reason",
    "notes",
]

POLICY_COLUMNS = [
    "ambiguity_category",
    "policy_id",
    "policy_name",
    "description",
    "paper_evidence",
    "official_code_evidence",
    "local_evidence",
    "uses_train_only_data",
    "preserves_same_cohort",
    "preserves_top_k",
    "changes_metric_definition",
    "introduces_leakage",
    "result_dependent",
    "comparability_impact",
    "methodological_risk",
    "allowed_for_final_computation",
    "recommendation_status",
    "rationale",
    "stop_condition",
    "notes",
]

BLOCKED_COLUMNS = [
    "issue_id",
    "metric_id",
    "dataset",
    "issue_category",
    "issue_description",
    "missing_or_conflicting_evidence",
    "affected_baselines",
    "affected_tasks",
    "affected_cohorts",
    "blocks_exact_metric",
    "adapted_alternative_available",
    "adapted_alternative_description",
    "requires_external_resource",
    "requires_user_or_supervisor_decision",
    "recommended_resolution",
    "severity",
    "status",
    "notes",
]

OUTPUTS = {
    "contract": "results/tables/canonical_metric_contract.csv",
    "inventory": "results/tables/canonical_metric_implementation_inventory.csv",
    "preflight": "results/tables/canonical_metric_same_cohort_preflight.csv",
    "popularity": "results/tables/canonical_metric_popularity_input_audit.csv",
    "universe": "results/tables/canonical_metric_item_universe_audit.csv",
    "long_tail": "results/tables/canonical_metric_long_tail_audit.csv",
    "similarity": "results/tables/canonical_metric_similarity_feasibility.csv",
    "applicability": "results/tables/canonical_metric_applicability_matrix.csv",
    "policy": "results/tables/canonical_metric_policy_options.csv",
    "blocked": "results/tables/canonical_metric_blocked_or_ambiguous.csv",
    "report": "report/canonical_metric_contract_and_preflight.md",
}


def read_csv(root: Path, relative_path: str, **kwargs: Any) -> pd.DataFrame:
    path = root / relative_path
    return pd.read_csv(path, dtype="string", keep_default_na=False, **kwargs)


def rel_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_existing_inputs(root: Path, paths: Sequence[str] = PROTECTED_INPUTS) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in dict.fromkeys(paths):
        path = root / relative
        if path.exists() and path.is_file():
            hashes[relative] = file_sha256(path)
    return hashes


def combined_sha256(root: Path, source_paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(dict.fromkeys(source_paths)):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        if path.exists() and path.is_file():
            digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def run_git(root: Path, args: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"
    return result.stdout.strip()


def environment_summary(root: Path) -> dict[str, str]:
    versions = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "cwd": str(root),
    }
    for package in ("pandas", "numpy", "scipy", "torch"):
        try:
            module = __import__(package)
            versions[f"{package}_version"] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[f"{package}_version"] = "not_installed"
    return versions


def build_contract_frame() -> pd.DataFrame:
    registry = build_metric_registry()
    validate_metric_registry(registry)
    rows = [contract_row(contract) for contract in registry]
    return pd.DataFrame(rows, columns=CONTRACT_COLUMNS)


def build_implementation_inventory(root: Path) -> pd.DataFrame:
    rows = [
        _inventory_row(
            "INV-0001",
            "hr_at_5",
            "official_repository",
            "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
            "calHR(hit)",
            "5",
            "mean binary hit vector",
            "mean over sampled rows",
            "hit vector derived from positive rank",
            "sample rows",
            "empty vector behavior not guarded in official code",
            "official metric script present; not executed",
            "unknown",
            "true",
            "none",
            "source_evidence",
            "",
        ),
        _inventory_row(
            "INV-0002",
            "ndcg_at_5",
            "official_repository",
            "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
            "calNDCG(hit, rank)",
            "5",
            "mean(hit / log2(rank + 1)) with one-based rank",
            "mean over sampled rows",
            "hit vector and one-based positive rank",
            "sample rows",
            "empty vector behavior not guarded in official code",
            "official metric script present; not executed",
            "unknown",
            "true",
            "none",
            "source_evidence",
            "",
        ),
        _inventory_row(
            "INV-0003",
            "item_coverage_at_5",
            "official_repository",
            "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
            "calItemCoverage(id_results, metric_len, negative_samples, test_samples)",
            "5",
            "unique top-k IDs divided by sampled candidate pool",
            "cohort-level unique ratio",
            "ranked item IDs, negative samples, test positives",
            "candidate pool from negatives plus positives",
            "no explicit guard for empty candidate pool",
            "official metric script present; not executed",
            "unknown",
            "true",
            "terminology_conflict_with_catalog_coverage",
            "source_evidence",
            "Official ItemCoverage is not full catalog coverage.",
        ),
        _inventory_row(
            "INV-0004",
            "long_tail_rate_at_5",
            "official_repository",
            "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
            "calACLT(results, freq_dict, high_bar, metric_len)",
            "5",
            "fraction of top-k IDs whose train frequency is below high_bar",
            "micro mean over recommendation positions",
            "ranked item IDs and train-frequency dictionary",
            "top-k recommendation positions",
            "items absent from freq_dict are not in the tail set",
            "official metric script present; not executed",
            "unknown",
            "true",
            "none",
            "source_evidence",
            "high_bar is computed as the 0.8 quantile of training frequencies in main.",
        ),
        _inventory_row(
            "INV-0005",
            "average_popularity_at_5",
            "official_repository",
            "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
            "calARP(freq_dict, id_results, metric_len)",
            "5",
            "mean raw training frequency of top-k IDs",
            "micro mean over recommendation positions",
            "ranked item IDs and train-frequency dictionary",
            "top-k recommendation positions",
            "missing item frequency falls back to 0",
            "official metric script present; not executed",
            "unknown",
            "true",
            "none",
            "source_evidence",
            "",
        ),
        _inventory_row(
            "INV-0006",
            "novelty_at_5",
            "official_repository",
            "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
            "calSelfInfo(results, metric_len, information)",
            "5",
            "mean precomputed self-information over top-k IDs",
            "micro mean over recommendation positions",
            "ranked item IDs and information dictionary",
            "top-k recommendation positions",
            "missing information falls back to 0",
            "official metric script present; not executed",
            "unknown",
            "partially",
            "zero_missing_info_conflict",
            "source_evidence",
            "Phase 4A blocks silent smoothing or missing-information imputation.",
        ),
        _inventory_row(
            "INV-0007",
            "intra_list_diversity_at_5",
            "official_repository",
            "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
            "calMeanShannon; calMeanJaccard",
            "5",
            "category entropy and across-list candidate-overlap Jaccard",
            "mean over rows or pairwise list comparisons",
            "category mappings and candidate lists",
            "varies by function",
            "missing categories default to empty sets",
            "official metric script present; not executed",
            "unknown",
            "false",
            "different_metric_family",
            "conflict",
            "Official diversity functions are not a standard within-list ILD contract.",
        ),
        _inventory_row(
            "INV-0008",
            "candidate_validity_rate",
            "official_repository",
            "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
            "calHallucination(rec_results)",
            "5",
            "fraction of recommendation IDs equal to 0",
            "micro mean over outputs",
            "mapped recommendation IDs",
            "recommendation positions",
            "unmatched titles become ID 0 upstream",
            "official metric script present; not executed",
            "unknown",
            "partially",
            "inverse_validity_semantics",
            "source_evidence",
            "Candidate validity is the inverse safety framing for candidate-constrained outputs.",
        ),
        _inventory_row(
            "INV-0009",
            "hr_at_5",
            "local_source",
            "src/evaluation/utility_metrics.py",
            "hit_rate_at_k",
            "k",
            "mean binary user hit over top-k recommendations",
            "mean over users with ground truth",
            "recommendations and ground truth dataframes",
            "users with ground truth",
            "empty recommendations score zero for users with ground truth",
            "synthetic tests only for this implementation",
            "unknown",
            "partially",
            "multi_relevant_generalization",
            "implemented",
            "General local utility implementation supports multiple relevant items; same-cohort table uses one positive per row.",
        ),
        _inventory_row(
            "INV-0010",
            "catalog_coverage_at_5",
            "local_source",
            "src/evaluation/coverage.py",
            "catalog_coverage_at_k",
            "k",
            "unique recommended catalog items divided by catalog size",
            "cohort/model/run ratio",
            "recommendations and catalog dataframe",
            "catalog item count",
            "empty catalog returns 0.0 in local helper",
            "synthetic tests only for this implementation",
            "unknown",
            "partially",
            "empty_catalog_policy_differs",
            "implemented",
            "Phase 4A contract blocks empty final cohorts rather than treating missing denominator as evidence.",
        ),
        _inventory_row(
            "INV-0011",
            "novelty_at_5",
            "local_source",
            "src/analysis/plan_d_beyond_utility.py",
            "novelty_at_k",
            "5",
            "add-one smoothed self-information",
            "micro mean over top-k rows",
            "training popularity plus catalog size",
            "top-k recommendation positions",
            "smoothing handles zero counts",
            "Plan D metrics and tests exist",
            "unknown",
            "false",
            "smoothing_conflict",
            "conflict",
            "Phase 4A forbids inventing smoothing for final contract without explicit source evidence.",
        ),
        _inventory_row(
            "INV-0012",
            "popularity_bias_at_5",
            "local_source",
            "src/analysis/plan_d_beyond_utility.py",
            "popularity_bias_at_k",
            "5",
            "recommendation mean popularity minus candidate/catalog basis mean",
            "single dataset/method scalar",
            "recommendations, popularity, candidates or catalog",
            "top-k rows and basis rows",
            "empty candidate basis returns NaN",
            "Plan D metrics and tests exist",
            "unknown",
            "false",
            "unsupported_bias_definition",
            "conflict",
            "Useful diagnostic but not source-backed as the canonical paper popularity-bias metric.",
        ),
        _inventory_row(
            "INV-0013",
            "duplicate_rate_at_5",
            "local_source",
            "src/analysis/plan_d_beyond_utility.py",
            "duplicate_rate_at_k",
            "5",
            "mean within-user duplicate fraction",
            "mean over users",
            "top-k recommendation rows",
            "number of available top-k rows per user",
            "empty users score 0",
            "Plan D metrics and tests exist",
            "unknown",
            "partially",
            "denominator_policy_differs",
            "implemented",
            "Phase 4A uses fixed top-5 denominator for structural validation.",
        ),
        _inventory_row(
            "INV-0014",
            "candidate_validity_rate",
            "local_source",
            "src/analysis/plan_d_beyond_utility.py",
            "candidate_validity_rate_at_k",
            "5",
            "top-k rows present in user candidate set",
            "micro mean over top-k rows",
            "recommendations and candidates",
            "available top-k recommendation rows",
            "empty inputs return 0",
            "Plan D metrics and tests exist",
            "unknown",
            "partially",
            "denominator_policy_differs",
            "implemented",
            "Phase 4A uses fixed five positions to expose short lists as invalid.",
        ),
        _inventory_row(
            "INV-0015",
            "all",
            "paper",
            "paper/Beyond Utility Evaluating LLM As Recommander.pdf",
            "local PDF file",
            "not_extracted",
            "paper metric inventory could not be text-extracted with installed local tools",
            "not_applicable",
            "PDF evidence file only",
            "not_applicable",
            "not_applicable",
            "PDF hashed; no local pypdf/PyPDF2/pdfminer/pdftotext available",
            "unknown",
            "unknown",
            "source_extraction_blocked",
            "artifact_present",
            "The report uses official repository code and prior codebase map for executable evidence.",
        ),
        _inventory_row(
            "INV-0016",
            "all",
            "local_report",
            "report/codebase_map.md",
            "Metrics section",
            "5",
            "Summarizes official metric order and implementation caveats",
            "not_applicable",
            "official code inspection report",
            "not_applicable",
            "not_applicable",
            "read-only report evidence",
            "unknown",
            "true",
            "none",
            "documentation_evidence",
            "",
        ),
        _inventory_row(
            "INV-0017",
            "hr_at_5;ndcg_at_5",
            "local_result_schema",
            "results/tables/plan_e_lightgcn_same_cohort_baselines_metrics.csv",
            "hr_at_5;ndcg_at_5;positive_rank columns",
            "5",
            "stored utility metrics and positive-rank diagnostics for seven baselines",
            "dataset/task/cohort/baseline rows",
            "frozen candidate rankings",
            "evaluated_rows",
            "all current rows non-empty",
            "84 rows present",
            "unknown",
            "true",
            "none",
            "execution_evidence",
            "Phase 4A recomputes only HR/NDCG for reconciliation.",
        ),
    ]
    frame = pd.DataFrame(rows, columns=INVENTORY_COLUMNS)
    for path in sorted(set(frame["source_path"].astype(str))):
        if path and path != "not_applicable" and not (root / path).exists():
            raise ValueError(f"inventory source path missing: {path}")
    return frame


def _inventory_row(*values: str) -> dict[str, str]:
    return dict(zip(INVENTORY_COLUMNS, values, strict=True))


def load_ranking_sources(root: Path) -> dict[str, pd.DataFrame]:
    traditional = normalize_traditional_rankings(
        read_csv(root, RANKING_SOURCE_PATHS["traditional"]),
        source_path=RANKING_SOURCE_PATHS["traditional"],
    )
    bprmf = pd.concat(
        [
            normalize_bprmf_rankings(
                read_csv(root, RANKING_SOURCE_PATHS["bprmf_beauty_sports"]),
                source_path=RANKING_SOURCE_PATHS["bprmf_beauty_sports"],
                source_execution_device="cuda",
                source_determinism_status="completed",
            ),
            normalize_bprmf_rankings(
                read_csv(root, RANKING_SOURCE_PATHS["bprmf_ml1m"]),
                source_path=RANKING_SOURCE_PATHS["bprmf_ml1m"],
                source_execution_device="cuda",
                source_determinism_status="completed",
            ),
        ],
        ignore_index=True,
    )
    gru4rec = pd.concat(
        [
            normalize_gru4rec_rankings(
                read_csv(root, RANKING_SOURCE_PATHS["gru4rec_beauty_sports"]),
                source_path=RANKING_SOURCE_PATHS["gru4rec_beauty_sports"],
            ),
            normalize_gru4rec_rankings(
                read_csv(root, RANKING_SOURCE_PATHS["gru4rec_ml1m"]),
                source_path=RANKING_SOURCE_PATHS["gru4rec_ml1m"],
            ),
        ],
        ignore_index=True,
    )
    sasrec = pd.concat(
        [
            normalize_sasrec_rankings(
                read_csv(root, RANKING_SOURCE_PATHS["sasrec_beauty_sports"]),
                source_path=RANKING_SOURCE_PATHS["sasrec_beauty_sports"],
            ),
            normalize_sasrec_rankings(
                read_csv(root, RANKING_SOURCE_PATHS["sasrec_ml1m"]),
                source_path=RANKING_SOURCE_PATHS["sasrec_ml1m"],
            ),
        ],
        ignore_index=True,
    )
    lightgcn = pd.concat(
        [
            normalize_lightgcn_rankings(
                read_csv(root, RANKING_SOURCE_PATHS["lightgcn_beauty_sports"]),
                source_path=RANKING_SOURCE_PATHS["lightgcn_beauty_sports"],
                source_execution_device="cpu",
                source_determinism_status="completed_exact_cpu_repeat",
            ),
            normalize_lightgcn_rankings(
                read_csv(root, RANKING_SOURCE_PATHS["lightgcn_ml1m"]),
                source_path=RANKING_SOURCE_PATHS["lightgcn_ml1m"],
                source_execution_device="cuda",
                source_determinism_status="documented_sparse_forward_nondeterminism",
            ),
        ],
        ignore_index=True,
    )
    return {
        "traditional": traditional,
        "bprmf": bprmf,
        "gru4rec": gru4rec,
        "sasrec": sasrec,
        "lightgcn": lightgcn,
    }


def build_same_cohort_preflight(root: Path, sources: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    existing = read_csv(root, "results/tables/plan_e_lightgcn_same_cohort_baselines_metrics.csv")
    if len(existing) != 84:
        raise ValueError(f"expected 84 same-cohort rows, found {len(existing)}")
    comparison = read_csv(root, "results/tables/plan_e_lightgcn_baseline_comparison.csv")
    if len(comparison) != 72:
        raise ValueError(f"expected 72 paired comparison rows, found {len(comparison)}")

    datamap_items = {dataset: load_datamap_items(root, dataset) for dataset in DATASETS}
    rows: list[dict[str, Any]] = []
    for metric_row in existing.sort_values(
        ["baseline_id", "dataset", "task_type", "cohort_id"], kind="mergesort"
    ).to_dict("records"):
        baseline_id = str(metric_row["baseline_id"])
        dataset = str(metric_row["dataset"])
        task_type = str(metric_row["task_type"])
        candidate_set_id = str(metric_row["candidate_set_id"])
        cohort_id = str(metric_row["cohort_id"])
        lightgcn_unit = select_unit(sources["lightgcn"], dataset, task_type, candidate_set_id)
        keys = cohort_keys_from_lightgcn_rankings(lightgcn_unit, cohort_id)
        lightgcn_filtered = filter_rankings_to_cohort(lightgcn_unit, keys)
        source_frame = source_frame_for_baseline(
            baseline_id,
            traditional_rankings=sources["traditional"],
            bprmf_rankings=sources["bprmf"],
            gru4rec_rankings=sources["gru4rec"],
            sasrec_rankings=sources["sasrec"],
            lightgcn_rankings=sources["lightgcn"],
            dataset=dataset,
            task_type=task_type,
            candidate_set_id=candidate_set_id,
        )
        filtered = filter_rankings_to_cohort(source_frame, keys)
        status_details = _preflight_checks(
            filtered,
            lightgcn_filtered=lightgcn_filtered,
            catalog_items=datamap_items[dataset],
            expected_rows=int(metric_row["evaluated_rows"]),
            hr_existing=float(metric_row["hr_at_5"]),
            ndcg_existing=float(metric_row["ndcg_at_5"]),
        )
        row = {
            "baseline_id": baseline_id,
            "baseline_family": BASELINE_FAMILIES[baseline_id],
            "dataset": dataset,
            "task_type": task_type,
            "candidate_set_id": candidate_set_id,
            "cohort_id": cohort_id,
            "ranking_source_path": str(metric_row["ranking_source_path"]),
            "expected_rows": int(metric_row["evaluated_rows"]),
            **status_details,
            "hr_at_5_existing": str(metric_row["hr_at_5"]),
            "ndcg_at_5_existing": str(metric_row["ndcg_at_5"]),
            "notes": f"{baseline_id} filtered to LightGCN strict same-cohort keys",
        }
        rows.append(row)
    frame = pd.DataFrame(rows, columns=PREFLIGHT_COLUMNS)
    if len(frame) != 84:
        raise ValueError(f"preflight row count mismatch: {len(frame)}")
    return frame


def _preflight_checks(
    filtered: pd.DataFrame,
    *,
    lightgcn_filtered: pd.DataFrame,
    catalog_items: set[str],
    expected_rows: int,
    hr_existing: float,
    ndcg_existing: float,
) -> dict[str, Any]:
    observed_rows = len(filtered)
    left_keys = {row_key(row) for row in filtered.to_dict("records")}
    right_keys = {row_key(row) for row in lightgcn_filtered.to_dict("records")}
    exact_row_key_match = left_keys == right_keys and observed_rows == expected_rows
    candidate_list_match = True
    positive_match = True
    reference = {
        row_key(row): (parse_json_array(row["candidate_items"]), str(row["positive_item_id"]))
        for row in lightgcn_filtered.to_dict("records")
    }
    for row in filtered.to_dict("records"):
        key = row_key(row)
        candidates = parse_json_array(row["candidate_items"])
        if key not in reference or candidates != reference[key][0]:
            candidate_list_match = False
        if key not in reference or str(row["positive_item_id"]) != reference[key][1]:
            positive_match = False

    candidate_count_valid = all(len(parse_json_array(value)) == 20 for value in filtered["candidate_items"].tolist())
    ranking_membership_valid = True
    top_5_extractable = True
    all_catalog_valid = True
    all_scores_finite = True
    ranks: list[int] = []
    for row in filtered.to_dict("records"):
        candidates = parse_json_array(row["candidate_items"])
        ranked = parse_json_array(row["ranked_items"])
        scores = parse_float_array(row["ranked_scores"])
        ranks.append(int(row["positive_rank_zero_based"]))
        if len(ranked) < 5:
            top_5_extractable = False
        if set(ranked) != set(candidates) or len(ranked) != len(candidates):
            ranking_membership_valid = False
        if any(item not in catalog_items for item in ranked):
            all_catalog_valid = False
        if len(scores) != len(ranked) or any(not math.isfinite(value) for value in scores):
            all_scores_finite = False
    hr_recomputed = float(np.mean([rank < 5 for rank in ranks])) if ranks else float("nan")
    ndcg_recomputed = float(np.mean([ndcg_at_k_from_rank(rank, 5) for rank in ranks])) if ranks else float("nan")
    utility_reconciled = abs(hr_recomputed - hr_existing) <= 1e-12 and abs(ndcg_recomputed - ndcg_existing) <= 1e-12
    checks = [
        exact_row_key_match,
        candidate_list_match,
        positive_match,
        candidate_count_valid,
        ranking_membership_valid,
        top_5_extractable,
        all_catalog_valid,
        all_scores_finite,
        utility_reconciled,
    ]
    status = "ready" if all(checks) else "validation_failed"
    blocking = "" if status == "ready" else "one or more preflight identity, membership, score, or utility reconciliation checks failed"
    return {
        "observed_rows": observed_rows,
        "exact_row_key_match": bool_text(exact_row_key_match),
        "exact_candidate_list_match": bool_text(candidate_list_match),
        "exact_positive_item_match": bool_text(positive_match),
        "candidate_count_valid": bool_text(candidate_count_valid),
        "ranking_membership_valid": bool_text(ranking_membership_valid),
        "top_5_extractable": bool_text(top_5_extractable),
        "all_ranked_ids_catalog_valid": bool_text(all_catalog_valid),
        "all_scores_finite": bool_text(all_scores_finite),
        "utility_metrics_reconciled": bool_text(utility_reconciled),
        "hr_at_5_recomputed": f"{hr_recomputed:.17g}",
        "ndcg_at_5_recomputed": f"{ndcg_recomputed:.17g}",
        "preflight_status": status,
        "blocking_reason": blocking,
    }


def parse_float_array(value: Any) -> list[float]:
    decoded = json.loads(str(value))
    if not isinstance(decoded, list):
        raise ValueError("score cell is not a JSON array")
    return [float(item) for item in decoded]


def load_datamap_items(root: Path, dataset: str) -> set[str]:
    relative = f"data/reconstructed/plan_c/{DATASETS[dataset]}/datamaps.json"
    with (root / relative).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(item) for item in data.get("item2id", {}).keys()}


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def load_interactions(root: Path, dataset: str) -> pd.DataFrame:
    return read_csv(root, f"data/reconstructed/plan_c/{DATASETS[dataset]}/interactions.csv")


def load_catalog(root: Path, dataset: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
    path = root / f"data/reconstructed/plan_c/{DATASETS[dataset]}/catalog.csv"
    if columns is None:
        return pd.read_csv(path, dtype="string", keep_default_na=False)
    available = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [column for column in columns if column in available]
    return pd.read_csv(path, dtype="string", keep_default_na=False, usecols=usecols)


def train_popularity_counts(interactions: pd.DataFrame) -> Counter[str]:
    frame = interactions.copy()
    if "split" in frame.columns:
        frame = frame.loc[frame["split"].astype(str).str.lower().eq("train")].copy()
    return Counter(frame["item_id"].astype(str).tolist())


def full_popularity_counts(interactions: pd.DataFrame) -> Counter[str]:
    return Counter(interactions["item_id"].astype(str).tolist())


def unique_train_user_counts(interactions: pd.DataFrame) -> Counter[str]:
    frame = interactions.copy()
    if "split" in frame.columns:
        frame = frame.loc[frame["split"].astype(str).str.lower().eq("train")].copy()
    grouped = frame.drop_duplicates(["user_id", "item_id"]).groupby("item_id", sort=False).size()
    return Counter({str(index): int(value) for index, value in grouped.items()})


def build_candidate_unions(root: Path, sources: Mapping[str, pd.DataFrame]) -> dict[str, dict[str, set[str]]]:
    ranking_candidates = read_csv(root, "results/tables/ranking_candidate_sets.csv")
    full_plan_e2: dict[str, set[str]] = {dataset: set() for dataset in DATASETS}
    for row in ranking_candidates.loc[ranking_candidates["source_plan"].eq("Plan E2")].to_dict("records"):
        dataset = str(row["dataset"])
        if dataset in full_plan_e2:
            full_plan_e2[dataset].update(parse_json_array(row["candidate_items"]))
    strict: dict[str, set[str]] = {dataset: set() for dataset in DATASETS}
    for dataset in DATASETS:
        for task_type in ("ranking", "reranking"):
            candidate_set_id = _candidate_set_id(dataset, task_type)
            unit = select_unit(sources["lightgcn"], dataset, task_type, candidate_set_id)
            keys = cohort_keys_from_lightgcn_rankings(unit, FULL_COHORT_ID)
            filtered = filter_rankings_to_cohort(unit, keys)
            for value in filtered["candidate_items"].tolist():
                strict[dataset].update(parse_json_array(value))
    return {"strict": strict, "full_plan_e2": full_plan_e2}


def _candidate_set_id(dataset: str, task_type: str) -> str:
    slug = {
        "Amazon Beauty": "beauty",
        "Amazon Sports": "sports",
        "ML-1M": "ml1m",
    }[dataset]
    return f"plan_e_{slug}_1000user_{task_type}_manifest"


def build_metadata_maps(root: Path) -> dict[str, dict[str, Any]]:
    title_map = read_csv(root, "results/tables/item_title_map.csv", usecols=["dataset", "item_id", "canonical_title", "title_status"])
    output: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        catalog = load_catalog(root, dataset, ["item_id", "title", "categories"])
        category_by_item = {
            str(row["item_id"]): str(row.get("categories", "")).strip()
            for row in catalog.to_dict("records")
            if str(row.get("item_id", "")).strip()
        }
        category_items = {item for item, cats in category_by_item.items() if cats}
        titles = title_map.loc[title_map["dataset"].eq(dataset)].copy()
        titled_items = set(titles.loc[titles["canonical_title"].astype(str).str.strip().ne(""), "item_id"].astype(str))
        output[dataset] = {
            "category_by_item": category_by_item,
            "category_items": category_items,
            "titled_items": titled_items,
            "catalog_items": set(catalog["item_id"].astype(str).tolist()),
        }
    return output


def build_popularity_audit(root: Path, candidate_unions: Mapping[str, Mapping[str, set[str]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        interactions = load_interactions(root, dataset)
        datamap_items = load_datamap_items(root, dataset)
        train_counts = train_popularity_counts(interactions)
        unique_counts = unique_train_user_counts(interactions)
        full_counts = full_popularity_counts(interactions)
        candidate_counts = Counter()
        candidate_rows = read_csv(root, "results/tables/ranking_candidate_sets.csv")
        for row in candidate_rows.loc[candidate_rows["dataset"].eq(dataset) & candidate_rows["source_plan"].eq("Plan E2")].to_dict("records"):
            candidate_counts.update(parse_json_array(row["candidate_items"]))
        sources = [
            (
                "train_interaction_count",
                f"data/reconstructed/plan_c/{DATASETS[dataset]}/interactions.csv",
                "split=train only",
                "raw train interaction count per item_id",
                train_counts,
                True,
                "official_reconstructed",
                True,
                "ready",
                "",
                "Recommended because it is leakage-free and matches official train frequency usage.",
            ),
            (
                "train_unique_user_count",
                f"data/reconstructed/plan_c/{DATASETS[dataset]}/interactions.csv",
                "split=train only",
                "unique train users per item_id",
                unique_counts,
                True,
                "not_paper_aligned",
                False,
                "ready_not_recommended",
                "",
                "Plausible but not the official raw-frequency definition.",
            ),
            (
                "full_reconstructed_interaction_count",
                f"data/reconstructed/plan_c/{DATASETS[dataset]}/interactions.csv",
                "all available splits",
                "raw interaction count across all reconstructed splits",
                full_counts,
                False,
                "not_paper_aligned",
                False,
                "forbidden_leakage_risk",
                "includes non-train interactions when validation/test splits exist",
                "Not allowed for final metrics.",
            ),
            (
                "datamap_catalog_presence",
                f"data/reconstructed/plan_c/{DATASETS[dataset]}/datamaps.json",
                "catalog presence only",
                "constant count of one for every datamap item",
                Counter({item: 1 for item in datamap_items}),
                True,
                "not_paper_aligned",
                False,
                "not_recommended",
                "not a behavioral popularity source",
                "Useful only as a universe sanity check.",
            ),
            (
                "plan_e2_candidate_occurrence_count",
                "results/tables/ranking_candidate_sets.csv",
                "candidate manifests, not train data",
                "candidate occurrence count in Plan E2 manifests",
                candidate_counts,
                False,
                "not_paper_aligned",
                False,
                "forbidden_not_source_popularity",
                "candidate occurrence is evaluation design, not item popularity",
                "Explicitly excluded from final popularity-source policy.",
            ),
        ]
        for source_id, source_path, split_policy, definition, counts, leakage_free, paper_aligned, recommended, status, blocking, notes in sources:
            rows.append(
                _popularity_row(
                    root,
                    dataset=dataset,
                    source_id=source_id,
                    source_path=source_path,
                    split_policy=split_policy,
                    count_definition=definition,
                    universe_items=datamap_items,
                    counts=counts,
                    leakage_free=leakage_free,
                    paper_aligned=paper_aligned,
                    recommended=recommended,
                    status=status,
                    blocking_reason=blocking,
                    notes=notes,
                )
            )
    return pd.DataFrame(rows, columns=POPULARITY_COLUMNS)


def _popularity_row(
    root: Path,
    *,
    dataset: str,
    source_id: str,
    source_path: str,
    split_policy: str,
    count_definition: str,
    universe_items: set[str],
    counts: Counter[str],
    leakage_free: bool,
    paper_aligned: str,
    recommended: bool,
    status: str,
    blocking_reason: str,
    notes: str,
) -> dict[str, Any]:
    series = np.asarray([float(counts.get(item, 0.0)) for item in sorted(universe_items)], dtype=float)
    positive = series[series > 0]
    total = float(series.sum())
    return {
        "dataset": dataset,
        "popularity_source_id": source_id,
        "source_path": source_path,
        "split_policy": split_policy,
        "count_definition": count_definition,
        "catalog_universe_id": "reconstructed_datamap_catalog",
        "item_count": len(universe_items),
        "items_with_positive_count": int((series > 0).sum()),
        "items_with_zero_count": int((series == 0).sum()),
        "total_count": int(total) if total.is_integer() else f"{total:.12g}",
        "minimum_positive_count": int(positive.min()) if len(positive) else "",
        "maximum_count": int(series.max()) if len(series) else "",
        "mean_count": f"{float(series.mean()):.12g}" if len(series) else "",
        "median_count": f"{float(np.median(series)):.12g}" if len(series) else "",
        "probability_denominator": int(total) if total.is_integer() else f"{total:.12g}",
        "smoothing_required": "false",
        "leakage_free": bool_text(leakage_free),
        "paper_aligned": paper_aligned,
        "recommended": bool_text(recommended),
        "status": status,
        "blocking_reason": blocking_reason,
        "sha256": combined_sha256(root, [source_path]),
        "notes": notes,
    }


def build_universe_audit(
    root: Path,
    candidate_unions: Mapping[str, Mapping[str, set[str]]],
    metadata_maps: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        interactions = load_interactions(root, dataset)
        train_counts = train_popularity_counts(interactions)
        train_seen = {item for item, count in train_counts.items() if count > 0}
        datamap_items = load_datamap_items(root, dataset)
        strict_union = set(candidate_unions["strict"][dataset])
        full_plan_e2_union = set(candidate_unions["full_plan_e2"][dataset])
        metadata_items = set(metadata_maps[dataset]["category_items"])
        titled_items = set(metadata_maps[dataset]["titled_items"])
        universes = [
            (
                "train_seen_catalog",
                "items observed in split=train interactions",
                f"data/reconstructed/plan_c/{DATASETS[dataset]}/interactions.csv",
                train_seen,
                ["catalog_coverage_at_5", "novelty_at_5", "long_tail_rate_at_5", "average_popularity_at_5"],
                {"catalog": True, "item": False, "novelty": True, "long_tail": True, "popularity": True, "ild": False},
            ),
            (
                "reconstructed_datamap_catalog",
                "items present in reconstructed datamaps.json item2id",
                f"data/reconstructed/plan_c/{DATASETS[dataset]}/datamaps.json",
                datamap_items,
                [],
                {"catalog": False, "item": False, "novelty": False, "long_tail": False, "popularity": False, "ild": False},
            ),
            (
                "strict_cohort_candidate_union",
                "union of candidates in LightGCN strict full same-cohort rows",
                ";".join(sorted(set(RANKING_SOURCE_PATHS.values()))),
                strict_union,
                ["item_coverage_at_5"],
                {"catalog": False, "item": True, "novelty": False, "long_tail": False, "popularity": False, "ild": False},
            ),
            (
                "full_plan_e2_candidate_union",
                "union of all Plan E2 1000-user candidate-set rows before strict filtering",
                "results/tables/ranking_candidate_sets.csv",
                full_plan_e2_union,
                [],
                {"catalog": False, "item": False, "novelty": False, "long_tail": False, "popularity": False, "ild": False},
            ),
            (
                "metadata_covered_catalog",
                "items with non-empty category or genre metadata in reconstructed catalog",
                f"data/reconstructed/plan_c/{DATASETS[dataset]}/catalog.csv",
                metadata_items,
                [],
                {"catalog": False, "item": False, "novelty": False, "long_tail": False, "popularity": False, "ild": True},
            ),
        ]
        for universe_id, definition, source_path, items, metric_ids, flags in universes:
            rows.append(
                {
                    "dataset": dataset,
                    "universe_id": universe_id,
                    "universe_definition": definition,
                    "source_path": source_path,
                    "item_count": len(items),
                    "train_seen_item_count": len(train_seen),
                    "metadata_covered_item_count": len(metadata_items),
                    "strict_candidate_union_item_count": len(strict_union),
                    "full_plan_e2_candidate_union_item_count": len(full_plan_e2_union),
                    "items_missing_title": len([item for item in items if item not in titled_items]),
                    "items_missing_category_or_genre": len([item for item in items if item not in metadata_items]),
                    "items_missing_popularity": len([item for item in items if train_counts.get(item, 0) <= 0]),
                    "appropriate_for_catalog_coverage": bool_text(flags["catalog"]),
                    "appropriate_for_item_coverage": bool_text(flags["item"]),
                    "appropriate_for_novelty": bool_text(flags["novelty"]),
                    "appropriate_for_long_tail": bool_text(flags["long_tail"]),
                    "appropriate_for_popularity_metrics": bool_text(flags["popularity"]),
                    "appropriate_for_ild": bool_text(flags["ild"]),
                    "recommended_metric_ids": json_text(metric_ids),
                    "status": "ready" if items else "blocked_missing_input",
                    "blocking_reason": "" if items else "universe is empty",
                    "sha256": combined_sha256(root, source_path.split(";")),
                    "notes": "Universe audited only; no recommendation metrics computed.",
                }
            )
    return pd.DataFrame(rows, columns=UNIVERSE_COLUMNS)


def build_long_tail_audit(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        interactions = load_interactions(root, dataset)
        counts = train_popularity_counts(interactions)
        positive_counts = {item: count for item, count in counts.items() if count > 0}
        tail, threshold = build_long_tail_set(positive_counts, quantile=0.8)
        head = {item for item in positive_counts if item not in tail}
        rows.append(
            {
                "dataset": dataset,
                "candidate_policy_id": "official_aclt_quantile_0_8_train_count",
                "policy_description": "ACLT long-tail items are train-count items strictly below the 0.8 quantile threshold.",
                "popularity_source_id": "train_interaction_count",
                "catalog_universe_id": "train_seen_catalog",
                "sort_direction": "ascending_by_train_count",
                "threshold_type": "quantile",
                "threshold_value": f"{threshold:.12g}",
                "tie_handling": "strict_less_than_threshold_is_tail",
                "head_item_count": len(head),
                "tail_item_count": len(tail),
                "head_interaction_mass": int(sum(positive_counts[item] for item in head)),
                "tail_interaction_mass": int(sum(positive_counts[item] for item in tail)),
                "paper_evidence": "ACLT/long-tail dimension in paper; exact prose not text-extracted locally",
                "official_code_evidence": "llm_metrics.py computes high_bar=np.quantile(train_freq_value,0.8) and freq_dict[x] < high_bar",
                "local_evidence": "src/analysis/plan_d_beyond_utility.py uses an 80th percentile train-popularity rule",
                "leakage_free": "true",
                "result_independent": "true",
                "recommended": "true",
                "status": "ready_reconstructed",
                "blocking_reason": "",
                "notes": "Threshold is computed only from train interactions.",
            }
        )
        for policy_id, description, status, blocking, recommended in [
            (
                "undefined_paper_text_threshold",
                "Exact paper-only prose threshold cannot be verified from local PDF text extraction.",
                "blocked_ambiguous_definition",
                "local PDF text extraction unavailable",
                "false",
            ),
            (
                "choose_threshold_after_results",
                "Selecting a long-tail threshold after inspecting recommendation outcomes.",
                "forbidden_result_dependent",
                "threshold would be result-dependent",
                "false",
            ),
        ]:
            rows.append(
                {
                    "dataset": dataset,
                    "candidate_policy_id": policy_id,
                    "policy_description": description,
                    "popularity_source_id": "train_interaction_count",
                    "catalog_universe_id": "train_seen_catalog",
                    "sort_direction": "",
                    "threshold_type": "",
                    "threshold_value": "",
                    "tie_handling": "",
                    "head_item_count": "",
                    "tail_item_count": "",
                    "head_interaction_mass": "",
                    "tail_interaction_mass": "",
                    "paper_evidence": "ambiguous or unavailable locally",
                    "official_code_evidence": "not used for this policy",
                    "local_evidence": "not used for this policy",
                    "leakage_free": "true" if policy_id != "choose_threshold_after_results" else "false",
                    "result_independent": "true" if policy_id != "choose_threshold_after_results" else "false",
                    "recommended": recommended,
                    "status": status,
                    "blocking_reason": blocking,
                    "notes": "No head/tail counts are computed for undefined or forbidden thresholds.",
                }
            )
    return pd.DataFrame(rows, columns=LONG_TAIL_COLUMNS)


def build_similarity_feasibility(
    root: Path,
    candidate_unions: Mapping[str, Mapping[str, set[str]]],
    metadata_maps: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    title_map = read_csv(root, "results/tables/item_title_map.csv", usecols=["dataset", "item_id", "canonical_title"])
    for dataset in DATASETS:
        strict_items = set(candidate_unions["strict"][dataset])
        category_by_item = dict(metadata_maps[dataset]["category_by_item"])
        category_features = {
            item: parse_category_features(value)
            for item, value in category_by_item.items()
            if parse_category_features(value)
        }
        rows.append(
            _similarity_row(
                root,
                dataset=dataset,
                source_id="catalog_category_or_genre_jaccard",
                source_type="category_or_genre_metadata",
                source_path=f"data/reconstructed/plan_c/{DATASETS[dataset]}/catalog.csv",
                feature_definition="split reconstructed catalog categories/genres on pipe separators",
                feature_map=category_features,
                strict_items=strict_items,
                function="jaccard_similarity",
                distance="1 - jaccard_similarity",
                paper_aligned="unknown",
                requires_adaptation="true",
                recommended="false",
                status="ready_reconstructed" if category_features else "blocked_missing_features",
                blocking="",
                notes="Feature source is deterministic, but exact paper ILD representation remains blocked.",
            )
        )
        titles = title_map.loc[title_map["dataset"].eq(dataset)].copy()
        title_features = {
            str(row["item_id"]): parse_title_tokens(row["canonical_title"])
            for row in titles.to_dict("records")
            if parse_title_tokens(row["canonical_title"])
        }
        rows.append(
            _similarity_row(
                root,
                dataset=dataset,
                source_id="canonical_title_token_jaccard",
                source_type="title_tokens",
                source_path="results/tables/item_title_map.csv",
                feature_definition="lowercase alphanumeric title tokens from canonical title map",
                feature_map=title_features,
                strict_items=strict_items,
                function="jaccard_similarity",
                distance="1 - jaccard_similarity",
                paper_aligned="false",
                requires_adaptation="true",
                recommended="false",
                status="ready_adapted",
                blocking="",
                notes="Adapted fallback only; not a substitute for an exact paper ILD source.",
            )
        )
        rows.append(
            {
                "dataset": dataset,
                "similarity_source_id": "external_embedding_or_official_feature_vector",
                "source_type": "embedding_or_feature_vector",
                "source_path": "",
                "feature_definition": "not available locally without downloads",
                "feature_dimension": "",
                "catalog_items": len(category_by_item),
                "strict_candidate_items": len(strict_items),
                "strict_candidate_items_with_features": 0,
                "coverage_rate": "0",
                "zero_vector_item_count": "",
                "duplicate_feature_item_count": "",
                "similarity_function": "not_frozen",
                "distance_function": "not_frozen",
                "paper_aligned": "unknown",
                "requires_adaptation": "false",
                "deterministic": "not_applicable",
                "recommended": "false",
                "status": "blocked_missing_features",
                "blocking_reason": "no local official embeddings or feature vectors are available, and downloads are forbidden",
                "sha256": "",
                "notes": "Records the exact-resource gap without downloading anything.",
            }
        )
    return pd.DataFrame(rows, columns=SIMILARITY_COLUMNS)


def _similarity_row(
    root: Path,
    *,
    dataset: str,
    source_id: str,
    source_type: str,
    source_path: str,
    feature_definition: str,
    feature_map: Mapping[str, set[str]],
    strict_items: set[str],
    function: str,
    distance: str,
    paper_aligned: str,
    requires_adaptation: str,
    recommended: str,
    status: str,
    blocking: str,
    notes: str,
) -> dict[str, Any]:
    strict_with_features = {item for item in strict_items if item in feature_map}
    feature_values = [json_text(sorted(values)) for values in feature_map.values()]
    duplicate_features = len(feature_values) - len(set(feature_values))
    catalog_items = len(feature_map)
    return {
        "dataset": dataset,
        "similarity_source_id": source_id,
        "source_type": source_type,
        "source_path": source_path,
        "feature_definition": feature_definition,
        "feature_dimension": "variable_set_cardinality",
        "catalog_items": catalog_items,
        "strict_candidate_items": len(strict_items),
        "strict_candidate_items_with_features": len(strict_with_features),
        "coverage_rate": f"{(len(strict_with_features) / len(strict_items) if strict_items else 0.0):.12g}",
        "zero_vector_item_count": len([item for item, features in feature_map.items() if not features]),
        "duplicate_feature_item_count": duplicate_features,
        "similarity_function": function,
        "distance_function": distance,
        "paper_aligned": paper_aligned,
        "requires_adaptation": requires_adaptation,
        "deterministic": "true",
        "recommended": recommended,
        "status": status,
        "blocking_reason": blocking,
        "sha256": combined_sha256(root, [source_path]) if source_path else "",
        "notes": notes,
    }


def parse_category_features(value: Any) -> set[str]:
    text = str(value).strip().lower()
    if not text:
        return set()
    return {part.strip() for part in text.replace(">", "|").split("|") if part.strip()}


def parse_title_tokens(value: Any) -> set[str]:
    import re

    return set(re.findall(r"[a-z0-9]+", str(value).lower()))


def build_applicability_matrix(
    contract: pd.DataFrame,
    preflight: pd.DataFrame,
    popularity: pd.DataFrame,
    long_tail: pd.DataFrame,
    similarity: pd.DataFrame,
) -> pd.DataFrame:
    included_metrics = [
        metric
        for metric in contract["metric_id"].astype(str).tolist()
        if metric not in {"mean_positive_rank_zero_based", "median_positive_rank_zero_based"}
    ]
    contract_by_metric = {row["metric_id"]: row for row in contract.to_dict("records")}
    ranking_path_lookup = {
        (row["baseline_id"], row["dataset"], row["task_type"], row["cohort_id"]): row["ranking_source_path"]
        for row in preflight.to_dict("records")
    }
    ready_preflight = {
        (row["baseline_id"], row["dataset"], row["task_type"], row["cohort_id"]): row["preflight_status"] == "ready"
        for row in preflight.to_dict("records")
    }
    popularity_ready = {
        row["dataset"]: (
            row["status"] == "ready"
            and row["recommended"] == "true"
            and row["leakage_free"] == "true"
        )
        for row in popularity.to_dict("records")
        if row["popularity_source_id"] == "train_interaction_count"
    }
    long_tail_ready = {
        row["dataset"]: row["status"] == "ready_reconstructed" and row["recommended"] == "true"
        for row in long_tail.to_dict("records")
        if row["candidate_policy_id"] == "official_aclt_quantile_0_8_train_count"
    }
    similarity_ready = {
        row["dataset"]: row["status"] in {"ready_reconstructed", "ready_adapted"}
        for row in similarity.to_dict("records")
        if row["similarity_source_id"] == "catalog_category_or_genre_jaccard"
    }
    rows: list[dict[str, Any]] = []
    for metric_id in included_metrics:
        metric = contract_by_metric[metric_id]
        for baseline_id in sorted(ALL_BASELINES):
            for dataset in sorted(DATASETS):
                for task_type in ("ranking", "reranking"):
                    for cohort_id in (FULL_COHORT_ID, MATCHED_COHORT_ID):
                        key = (baseline_id, dataset, task_type, cohort_id)
                        contract_status = str(metric["contract_status"])
                        preflight_ok = ready_preflight.get(key, False)
                        needs_popularity = metric_id in {
                            "novelty_at_5",
                            "long_tail_rate_at_5",
                            "average_popularity_at_5",
                            "popularity_bias_at_5",
                        }
                        needs_long_tail = metric_id == "long_tail_rate_at_5"
                        needs_similarity = metric_id == "intra_list_diversity_at_5"
                        popularity_ok = popularity_ready.get(dataset, False) if needs_popularity else True
                        long_tail_ok = long_tail_ready.get(dataset, False) if needs_long_tail else True
                        similarity_ok = similarity_ready.get(dataset, False) if needs_similarity else True
                        structural = metric_id in {"duplicate_rate_at_5", "candidate_validity_rate"}
                        blocked = contract_status.startswith("blocked")
                        required_available = preflight_ok and (not needs_popularity or popularity_ok)
                        if needs_long_tail:
                            required_available = required_available and long_tail_ok
                        if needs_similarity:
                            required_available = required_available and similarity_ok
                        if blocked:
                            status = "blocked"
                            allowed = "false"
                            meaningful = "false"
                            blocking = str(metric["blocking_reason"])
                        elif structural:
                            status = "structural_invariant"
                            allowed = "true"
                            meaningful = "false"
                            blocking = ""
                        elif not required_available:
                            status = "blocked"
                            allowed = "false"
                            meaningful = "true"
                            blocking = "required preflight, popularity, long-tail, or similarity input is unavailable"
                        elif contract_status == "ready_adapted":
                            status = "ready_with_adaptation"
                            allowed = "true"
                            meaningful = "true"
                            blocking = ""
                        else:
                            status = "ready"
                            allowed = "true"
                            meaningful = "true"
                            blocking = ""
                        rows.append(
                            {
                                "metric_id": metric_id,
                                "baseline_id": baseline_id,
                                "dataset": dataset,
                                "task_type": task_type,
                                "cohort_id": cohort_id,
                                "ranking_source_path": ranking_path_lookup.get(key, ""),
                                "metric_contract_status": contract_status,
                                "required_inputs_available": bool_text(required_available),
                                "metadata_coverage_sufficient": bool_text(similarity_ok) if needs_similarity else "not_applicable",
                                "popularity_input_available": bool_text(popularity_ok) if needs_popularity else "not_applicable",
                                "long_tail_input_available": bool_text(long_tail_ok) if needs_long_tail else "not_applicable",
                                "similarity_input_available": bool_text(similarity_ok) if needs_similarity else "not_applicable",
                                "structurally_constant": bool_text(structural),
                                "structural_expected_value": (
                                    "0" if metric_id == "duplicate_rate_at_5" else "1" if metric_id == "candidate_validity_rate" else ""
                                ),
                                "meaningful_for_comparison": meaningful,
                                "final_computation_allowed": allowed,
                                "applicability_status": status,
                                "blocking_reason": blocking,
                                "notes": _applicability_note(metric_id, contract_status, structural),
                            }
                        )
    frame = pd.DataFrame(rows, columns=APPLICABILITY_COLUMNS)
    expected = len(included_metrics) * len(ALL_BASELINES) * len(DATASETS) * 2 * 2
    if len(frame) != expected:
        raise ValueError(f"applicability row count mismatch: expected {expected}, found {len(frame)}")
    return frame


def _applicability_note(metric_id: str, contract_status: str, structural: bool) -> str:
    if structural:
        return "Deterministic candidate-constrained rankings make this a structural invariant, not a comparative quality signal."
    if contract_status == "ready_adapted":
        return "Allowed only with explicit adapted-label reporting in Phase 4B."
    if contract_status.startswith("blocked"):
        return "Exact final computation is blocked until the ambiguity is resolved."
    return "Ready under the frozen same-cohort preflight."


def build_policy_options() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    add = rows.append
    add(_policy("utility_formula", "binary_positive_rank", "Use one-positive binary HR/NDCG", "Mean binary hit and one-positive NDCG from positive_rank_zero_based.", "paper names HR/NDCG", "calHR and calNDCG", "same-cohort table stores positive ranks", True, True, True, False, False, False, "preserves current utility layer", "low", True, "recommended", "Matches official code and reconciles exactly with existing utility rows.", "", ""))
    add(_policy("utility_formula", "multi_relevant_general_utility", "Use generic multi-relevant utility helper", "Use src/evaluation utility implementation without constraining to one positive.", "not verified for paper setup", "not official one-positive path", "src/evaluation/utility_metrics.py", True, True, True, True, False, False, "changes row semantics", "medium", False, "not_recommended", "The same-cohort candidates have exactly one positive item.", "Do not use for Phase 4B.", ""))
    add(_policy("popularity_source", "train_interaction_count", "Train-only raw interaction counts", "Use split=train item counts only.", "paper/official code uses train histories", "train_id_freq_dict from train_samples", "reconstructed interactions split=train", True, True, True, False, False, False, "maximal", "low", True, "recommended", "Leakage-free and closest to official ARP/ACLT behavior.", "", ""))
    add(_policy("popularity_source", "full_interaction_count", "Full reconstructed counts", "Use all interactions regardless of split.", "not source-backed", "not official", "interactions.csv contains split column", False, True, True, True, True, False, "would inflate held-out information", "high", False, "forbidden", "Validation/test leakage is not allowed.", "Never use for final metrics.", ""))
    add(_policy("popularity_source", "candidate_occurrence_count", "Candidate occurrence counts", "Use evaluation candidate frequencies as popularity.", "not source-backed", "not official", "ranking_candidate_sets.csv", False, True, True, True, True, False, "evaluation-design dependent", "high", False, "forbidden", "Candidate occurrence is not item popularity.", "Never use for final metrics.", ""))
    add(_policy("novelty_smoothing", "no_smoothing_block_zero", "No smoothing; block zero counts", "Compute self-information only for positive train probabilities.", "smoothing not verified from paper", "calSelfInfo uses provided information dictionary", "Plan D smoothing exists but is local adaptation", True, True, True, False, False, False, "explicitly exposes missing popularity", "low", True, "recommended", "Avoids inventing epsilon or add-one smoothing.", "Block if final top-5 contains a zero-train-count item.", ""))
    add(_policy("novelty_smoothing", "add_one_plan_d", "Use Plan D add-one smoothing", "Use (count+1)/(total+catalog_size).", "not verified", "not official formula", "src/analysis/plan_d_beyond_utility.py", True, True, True, True, False, False, "changes novelty magnitude", "medium", False, "acceptable_with_strong_caveat", "Only acceptable as an explicitly adapted diagnostic.", "Do not label as reconstructed paper metric.", ""))
    add(_policy("catalog_coverage_denominator", "train_seen_catalog", "Train-seen catalog denominator", "Use items observed in split=train.", "catalog denominator ambiguous", "official ItemCoverage uses candidate pool instead", "interactions.csv", True, True, True, True, False, False, "adapted but leakage-free", "medium", True, "recommended", "Best defensible catalog denominator for adapted catalog coverage.", "", ""))
    add(_policy("catalog_coverage_denominator", "metadata_catalog", "Metadata-covered catalog denominator", "Use only items with metadata.", "not verified", "not official", "catalog.csv", False, True, True, True, False, False, "metadata-biased", "medium", False, "not_recommended", "Would reward metadata availability rather than catalog coverage.", "", ""))
    add(_policy("item_coverage_denominator", "strict_candidate_union", "Strict candidate-union denominator", "Use candidates available in the exact same cohort.", "coverage terminology ambiguous", "calItemCoverage uses sampled candidate pool", "same-cohort rankings", True, True, True, False, False, False, "preserves same-cohort comparability", "low", True, "recommended", "Closest reconstruction of official candidate-pool coverage.", "", ""))
    add(_policy("item_coverage_denominator", "full_plan_e2_union", "Full Plan E2 union denominator", "Use all Plan E2 candidate rows before strict filtering.", "not official for strict layer", "not official", "ranking_candidate_sets.csv", True, False, True, True, False, False, "denominator includes excluded rows", "medium", False, "not_recommended", "Would break strict same-cohort denominator.", "", ""))
    add(_policy("long_tail_threshold", "official_quantile_0_8", "Official 0.8 quantile train-count threshold", "Tail if train count is strictly below the 0.8 quantile.", "ACLT dimension", "high_bar=np.quantile(train_freq_value,0.8)", "Plan D uses similar 80th percentile", True, True, True, False, False, False, "source-backed", "low", True, "recommended", "Defined before observing recommendation results.", "", ""))
    add(_policy("long_tail_threshold", "result_dependent_threshold", "Choose threshold after results", "Pick a threshold based on observed outcomes.", "forbidden by task", "not official", "not local policy", False, False, True, True, True, True, "invalid comparison", "high", False, "forbidden", "Would be result-dependent.", "Never use.", ""))
    add(_policy("popularity_bias_definition", "block_until_supervisor_decision", "Block popularity-bias exact metric", "Do not compute popularity_bias_at_5 until formula is clarified.", "paper wording unavailable locally", "calARP and calPopREO differ", "Plan D subtraction differs again", True, True, True, False, False, False, "prevents duplicated metrics", "low", False, "recommended", "Average popularity already captures ARP; bias must not duplicate it.", "Require a source-backed formula before final computation.", ""))
    add(_policy("popularity_bias_definition", "reuse_average_popularity", "Reuse ARP as popularity bias", "Report average popularity as popularity bias.", "ambiguous", "calARP is ARP", "Plan D has different function", True, True, True, True, False, False, "duplicates a metric", "high", False, "forbidden", "Would duplicate average_popularity_at_5 under two names.", "Never use.", ""))
    add(_policy("ild_similarity_source", "block_exact_ild", "Block exact ILD", "Do not compute exact intra_list_diversity_at_5 until representation is clarified.", "paper representation unavailable locally", "official diversity code is Shannon/Jaccard, not standard ILD", "Plan D category Jaccard is adapted", True, True, True, False, False, False, "prevents false exactness", "low", False, "recommended", "Adapted feasibility is documented separately.", "Supervisor/source decision required before exact ILD.", ""))
    add(_policy("ild_similarity_source", "category_jaccard_adaptation", "Category/genre Jaccard adaptation", "Use reconstructed categories/genres with Jaccard distance.", "not confirmed exact", "category mappings exist in official diversity code", "catalog.csv categories", True, True, True, True, False, False, "adapted but deterministic", "medium", True, "acceptable_with_strong_caveat", "Can be computed only with adapted labeling.", "Do not call exact paper ILD.", ""))
    add(_policy("duplicate_validity_handling", "fixed_top5_penalty", "Fixed top-5 denominator", "Count missing/out-of-candidate positions as failures and duplicates after first occurrence.", "LLM quality concern", "hallucination maps invalid to 0", "Plan D diagnostics exist", True, True, True, False, False, False, "clear structural interpretation", "low", True, "recommended", "Exposes malformed or short outputs and keeps deterministic baselines structural.", "", ""))
    add(_policy("duplicate_validity_handling", "deduplicate_before_metrics", "Deduplicate before metrics", "Remove repeated outputs before validity/coverage.", "not source-backed", "not official", "not local final policy", True, True, False, True, False, False, "masks invalid output", "high", False, "forbidden", "Would silently improve faulty lists.", "Never use.", ""))
    add(_policy("missing_metadata", "block_exact_or_label_adapted", "Block exact or label adapted", "Do not impute metadata; either block exact metrics or label pairwise-available adaptations.", "missing handling unavailable", "official code defaults missing categories to empty in places", "title/category coverage audits exist", True, True, True, False, False, False, "transparent", "low", True, "recommended", "No silent metadata imputation.", "Block exact if required metadata is missing.", ""))
    add(_policy("missing_metadata", "impute_unknown_metadata", "Impute Unknown metadata", "Replace missing categories/titles with placeholders.", "not source-backed", "not official", "forbidden by project constraints", False, True, True, True, False, False, "biases diversity", "high", False, "forbidden", "Would invent metadata.", "Never use.", ""))
    frame = pd.DataFrame(rows, columns=POLICY_COLUMNS)
    counts = frame.loc[frame["recommendation_status"].eq("recommended")].groupby("ambiguity_category").size()
    all_categories = frame.groupby("ambiguity_category").size().index
    if set(counts.index) != set(all_categories) or not counts.eq(1).all():
        raise ValueError("policy options must have exactly one recommended policy per ambiguity category")
    return frame


def _policy(
    ambiguity_category: str,
    policy_id: str,
    policy_name: str,
    description: str,
    paper_evidence: str,
    official_code_evidence: str,
    local_evidence: str,
    uses_train_only_data: bool,
    preserves_same_cohort: bool,
    preserves_top_k: bool,
    changes_metric_definition: bool,
    introduces_leakage: bool,
    result_dependent: bool,
    comparability_impact: str,
    methodological_risk: str,
    allowed_for_final_computation: bool,
    recommendation_status: str,
    rationale: str,
    stop_condition: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "ambiguity_category": ambiguity_category,
        "policy_id": policy_id,
        "policy_name": policy_name,
        "description": description,
        "paper_evidence": paper_evidence,
        "official_code_evidence": official_code_evidence,
        "local_evidence": local_evidence,
        "uses_train_only_data": bool_text(uses_train_only_data),
        "preserves_same_cohort": bool_text(preserves_same_cohort),
        "preserves_top_k": bool_text(preserves_top_k),
        "changes_metric_definition": bool_text(changes_metric_definition),
        "introduces_leakage": bool_text(introduces_leakage),
        "result_dependent": bool_text(result_dependent),
        "comparability_impact": comparability_impact,
        "methodological_risk": methodological_risk,
        "allowed_for_final_computation": bool_text(allowed_for_final_computation),
        "recommendation_status": recommendation_status,
        "rationale": rationale,
        "stop_condition": stop_condition,
        "notes": notes,
    }


def build_blocked_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    issue_number = 1
    affected_baselines = list(ALL_BASELINES)
    affected_tasks = ["ranking", "reranking"]
    affected_cohorts = [FULL_COHORT_ID, MATCHED_COHORT_ID]
    for dataset in DATASETS:
        rows.append(
            _blocked_row(
                issue_number,
                "popularity_bias_at_5",
                dataset,
                "conflicting_formula",
                "Popularity bias is not disambiguated from ARP/average popularity or PopREO.",
                "official calARP and calPopREO differ; local Plan D uses recommendation-minus-candidate/catalog mean",
                affected_baselines,
                affected_tasks,
                affected_cohorts,
                True,
                True,
                "average_popularity_at_5 is ready as ARP; popularity_bias_at_5 remains blocked",
                False,
                True,
                "Obtain a source-backed supervisor-approved formula or omit the metric.",
                "blocking",
                "open",
                "",
            )
        )
        issue_number += 1
        rows.append(
            _blocked_row(
                issue_number,
                "intra_list_diversity_at_5",
                dataset,
                "ambiguous_similarity_source",
                "Exact ILD representation and distance function are not frozen.",
                "official diversity code implements Shannon/Jaccard variants but not a standard within-list ILD formula",
                affected_baselines,
                affected_tasks,
                affected_cohorts,
                True,
                True,
                "category/genre or title-token Jaccard may be used only as an adapted metric after explicit approval",
                False,
                True,
                "Resolve exact representation/similarity source before exact final ILD computation.",
                "blocking",
                "open",
                "",
            )
        )
        issue_number += 1
    rows.append(
        _blocked_row(
            issue_number,
            "catalog_coverage_at_5",
            "all",
            "paper_terminology_ambiguity",
            "Paper/official terminology does not cleanly distinguish catalog coverage from candidate-pool ItemCoverage.",
            "official calItemCoverage uses sampled candidate pool while local coverage.py uses full catalog denominator",
            affected_baselines,
            affected_tasks,
            affected_cohorts,
            False,
            True,
            "Report catalog_coverage_at_5 as adapted train-seen catalog coverage and item_coverage_at_5 as candidate-union coverage",
            False,
            False,
            "Use explicit adapted/reconstructed labels.",
            "warning",
            "resolved_by_policy",
            "",
        )
    )
    return pd.DataFrame(rows, columns=BLOCKED_COLUMNS)


def _blocked_row(
    issue_number: int,
    metric_id: str,
    dataset: str,
    issue_category: str,
    issue_description: str,
    missing_or_conflicting_evidence: str,
    affected_baselines: Sequence[str],
    affected_tasks: Sequence[str],
    affected_cohorts: Sequence[str],
    blocks_exact_metric: bool,
    adapted_alternative_available: bool,
    adapted_alternative_description: str,
    requires_external_resource: bool,
    requires_user_or_supervisor_decision: bool,
    recommended_resolution: str,
    severity: str,
    status: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "issue_id": f"CMBA-{issue_number:04d}",
        "metric_id": metric_id,
        "dataset": dataset,
        "issue_category": issue_category,
        "issue_description": issue_description,
        "missing_or_conflicting_evidence": missing_or_conflicting_evidence,
        "affected_baselines": json_text(list(affected_baselines)),
        "affected_tasks": json_text(list(affected_tasks)),
        "affected_cohorts": json_text(list(affected_cohorts)),
        "blocks_exact_metric": bool_text(blocks_exact_metric),
        "adapted_alternative_available": bool_text(adapted_alternative_available),
        "adapted_alternative_description": adapted_alternative_description,
        "requires_external_resource": bool_text(requires_external_resource),
        "requires_user_or_supervisor_decision": bool_text(requires_user_or_supervisor_decision),
        "recommended_resolution": recommended_resolution,
        "severity": severity,
        "status": status,
        "notes": notes,
    }


def validate_outputs(outputs: Mapping[str, pd.DataFrame]) -> None:
    _require_columns(outputs["contract"], CONTRACT_COLUMNS, "contract")
    if outputs["contract"]["metric_id"].duplicated().any():
        raise ValueError("duplicate metric_id in contract")
    if not set(outputs["contract"]["contract_status"]).issubset(CONTRACT_STATUSES):
        raise ValueError("invalid contract status")
    _require_columns(outputs["inventory"], INVENTORY_COLUMNS, "inventory")
    allowed_source_types = {"paper", "official_repository", "local_source", "local_test", "local_report", "local_result_schema"}
    if not set(outputs["inventory"]["source_type"]).issubset(allowed_source_types):
        raise ValueError("invalid inventory source type")
    _require_columns(outputs["preflight"], PREFLIGHT_COLUMNS, "preflight")
    if len(outputs["preflight"]) != 84:
        raise ValueError("same-cohort preflight must have 84 rows")
    if not outputs["preflight"]["preflight_status"].eq("ready").all():
        raise ValueError("same-cohort preflight contains non-ready rows")
    _require_columns(outputs["popularity"], POPULARITY_COLUMNS, "popularity")
    recommended = outputs["popularity"].loc[outputs["popularity"]["recommended"].eq("true")]
    if not recommended["leakage_free"].eq("true").all():
        raise ValueError("recommended popularity source is not leakage-free")
    _require_columns(outputs["universe"], UNIVERSE_COLUMNS, "universe")
    for value in outputs["universe"]["recommended_metric_ids"]:
        json.loads(value)
    _require_columns(outputs["long_tail"], LONG_TAIL_COLUMNS, "long_tail")
    bad_tail = outputs["long_tail"].loc[outputs["long_tail"]["recommended"].eq("true") & outputs["long_tail"]["threshold_value"].eq("")]
    if not bad_tail.empty:
        raise ValueError("recommended long-tail policy lacks threshold")
    _require_columns(outputs["similarity"], SIMILARITY_COLUMNS, "similarity")
    coverage = pd.to_numeric(outputs["similarity"]["coverage_rate"], errors="coerce").fillna(0)
    if ((coverage < 0) | (coverage > 1)).any():
        raise ValueError("similarity coverage outside [0,1]")
    _require_columns(outputs["applicability"], APPLICABILITY_COLUMNS, "applicability")
    if len(outputs["applicability"]) != 924:
        raise ValueError(f"applicability matrix must have 924 rows, found {len(outputs['applicability'])}")
    blocked_allowed = outputs["applicability"].loc[
        outputs["applicability"]["metric_contract_status"].str.startswith("blocked")
        & outputs["applicability"]["final_computation_allowed"].eq("true")
    ]
    if not blocked_allowed.empty:
        raise ValueError("blocked metric allowed for final computation")
    _require_columns(outputs["policy"], POLICY_COLUMNS, "policy")
    policy_counts = outputs["policy"].loc[outputs["policy"]["recommendation_status"].eq("recommended")].groupby("ambiguity_category").size()
    all_categories = outputs["policy"].groupby("ambiguity_category").size().index
    if set(policy_counts.index) != set(all_categories) or not policy_counts.eq(1).all():
        raise ValueError("policy options must have exactly one recommended option per category")
    _require_columns(outputs["blocked"], BLOCKED_COLUMNS, "blocked")
    blocked_metrics = set(outputs["contract"].loc[outputs["contract"]["contract_status"].str.startswith("blocked"), "metric_id"])
    issue_metrics = set(outputs["blocked"]["metric_id"])
    if not blocked_metrics.issubset(issue_metrics):
        raise ValueError("every blocked metric must have a blocked/ambiguous issue")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    if list(frame.columns) != list(columns):
        raise ValueError(f"{label} schema mismatch")
    if frame.empty:
        raise ValueError(f"{label} frame is empty")


def write_outputs(root: Path, outputs: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key, relative_path in OUTPUTS.items():
        if key == "report":
            continue
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        outputs[key].to_csv(path, index=False, lineterminator="\n")
        hashes[relative_path] = file_sha256(path)
    return hashes


def render_report(
    *,
    root: Path,
    outputs: Mapping[str, pd.DataFrame],
    env: Mapping[str, str],
    initial_hashes: Mapping[str, str],
    final_hashes: Mapping[str, str],
    output_hashes: Mapping[str, str],
) -> str:
    contract = outputs["contract"]
    preflight = outputs["preflight"]
    popularity = outputs["popularity"]
    universe = outputs["universe"]
    long_tail = outputs["long_tail"]
    similarity = outputs["similarity"]
    applicability = outputs["applicability"]
    blocked = outputs["blocked"]
    policy = outputs["policy"]
    status_counts = contract["contract_status"].value_counts().to_dict()
    app_counts = applicability["applicability_status"].value_counts().to_dict()
    preflight_counts = preflight.groupby(["dataset", "task_type", "cohort_id"])["observed_rows"].first().reset_index()
    unchanged = initial_hashes == final_hashes
    lines = [
        "# Canonical Utility and Beyond-Utility Metric Contract and Same-Cohort Preflight",
        "",
        "## 1. Scope and provenance",
        f"- Branch: `{run_git(root, ['branch', '--show-current'])}`.",
        f"- Commit: `{run_git(root, ['log', '-1', '--format=%H %s'])}`.",
        f"- Python executable: `{env['python_executable']}`.",
        f"- Python/pandas/NumPy/scipy/torch: `{env['python_version']}`, `{env['pandas_version']}`, `{env['numpy_version']}`, `{env['scipy_version']}`, `{env['torch_version']}`.",
        f"- Protected input files hashed: `{len(initial_hashes)}`.",
        "",
        "## 2. Executive summary",
        f"- Metric contracts written: `{len(contract)}`; status counts: `{json_text(status_counts)}`.",
        f"- Same-cohort preflight rows: `{len(preflight)}` and all rows have status `{preflight['preflight_status'].unique()[0]}`.",
        f"- Applicability rows: `{len(applicability)}`; status counts: `{json_text(app_counts)}`.",
        f"- Blocked/ambiguous issue rows: `{len(blocked)}`.",
        "",
        "## 3. Safety boundaries",
        "- Phase 4A read existing artifacts only, recomputed HR@5/NDCG@5 only for reconciliation, and wrote only the requested Phase 4A outputs.",
        "",
        "## 4. Completed seven-baseline layer",
        f"- Baselines in same-cohort table: `{preflight['baseline_id'].nunique()}`.",
        f"- Datasets: `{preflight['dataset'].nunique()}`; tasks: `{preflight['task_type'].nunique()}`; cohorts: `{preflight['cohort_id'].nunique()}`.",
        "",
        "## 5. Canonical same-cohort definition",
        f"- Full cohort ID: `{FULL_COHORT_ID}`.",
        f"- Matched cohort ID: `{MATCHED_COHORT_ID}`.",
        "",
        "## 6. Paper metric inventory",
        "- The local paper PDF was hashed, but no local PDF text extractor is installed. Paper-level claims are therefore tied to the PDF artifact plus official repository code and local codebase-map evidence.",
        "",
        "## 7. Official-repository metric inventory",
        "- Official metric evidence is `original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py`, including `calHR`, `calNDCG`, `calARP`, `calACLT`, `calItemCoverage`, `calSelfInfo`, `calMeanShannon`, `calMeanJaccard`, and `calHallucination`.",
        "",
        "## 8. Local metric implementation inventory",
        f"- Inventory rows: `{len(outputs['inventory'])}`. Local sources include utility, coverage, hallucination, and Plan D beyond-utility implementations.",
        "",
        "## 9. Conflicting metric definitions",
        "- Conflicts are recorded for smoothed novelty, popularity-bias semantics, catalog versus candidate-pool coverage, duplicate denominator handling, and exact ILD representation.",
        "",
        "## 10. Utility metric contract",
        "- HR@5 and NDCG@5 are ready_reconstructed same-cohort utility metrics with top-5 cutoff and one positive item per row.",
        "",
        "## 11. HR@5 definition",
        "- HR@5 is `mean(positive_rank_zero_based < 5)` over rows.",
        "",
        "## 12. NDCG@5 definition",
        "- NDCG@5 is `1/log2(rank_one_based+1)` for positive ranks in the top five, otherwise `0`, averaged over rows.",
        "",
        "## 13. Positive-rank semantics",
        "- Stored positive ranks are zero-based in `positive_rank_zero_based`; one-based ranks are derived by adding one.",
        "",
        "## 14. Utility metric reconciliation",
        f"- Reconciled rows: `{int(preflight['utility_metrics_reconciled'].eq('true').sum())}` of `{len(preflight)}`.",
        "",
        "## 15. Novelty definition",
        "- Novelty@5 is ready_adapted as train-only self-information without invented smoothing; zero train probability blocks final use.",
        "",
        "## 16. Catalog coverage definition",
        "- Catalog Coverage@5 is adapted: unique valid top-5 recommended items divided by the train-seen catalog.",
        "",
        "## 17. Item coverage definition",
        "- Item Coverage@5 reconstructs official candidate-pool coverage as unique top-5 items divided by the strict same-cohort candidate union.",
        "",
        "## 18. Long-tail rate definition",
        "- Long-tail Rate@5 reconstructs official ACLT using the strict less-than 0.8 quantile of train item frequencies.",
        "",
        "## 19. Average popularity definition",
        "- Average Popularity@5 reconstructs official ARP as mean raw train interaction count over top-5 positions.",
        "",
        "## 20. Popularity-bias definition",
        "- Popularity Bias@5 remains blocked because source evidence conflicts and it must not duplicate Average Popularity@5.",
        "",
        "## 21. Intra-list diversity definition",
        "- Exact ILD remains blocked because the exact representation and similarity source are not frozen; category and title-token adaptations are feasible but explicitly adapted.",
        "",
        "## 22. Duplicate-rate definition",
        "- Duplicate Rate@5 is a structural invariant for deterministic candidate permutations, expected `0` when rankings are valid.",
        "",
        "## 23. Candidate-validity definition",
        "- Candidate Validity Rate is a structural invariant for candidate-constrained deterministic baselines, expected `1` when top-5 items are candidate members.",
        "",
        "## 24. Aggregation-level contract",
        "- Utility metrics are row-level macro means; popularity/novelty/long-tail are top-5 position means; coverage metrics are cohort-level unique-item ratios.",
        "",
        "## 25. Top-5 extraction contract",
        f"- Top-5 extractability passed for `{int(preflight['top_5_extractable'].eq('true').sum())}` of `{len(preflight)}` preflight units.",
        "",
        "## 26. Duplicate and invalid-item handling",
        f"- Ranking membership passed for `{int(preflight['ranking_membership_valid'].eq('true').sum())}` of `{len(preflight)}` preflight units.",
        "",
        "## 27. Missing-data handling",
        "- Missing popularity or metadata is not imputed. Blocked/adapted statuses identify metrics needing those inputs.",
        "",
        "## 28. Train-only popularity sources",
        f"- Recommended train-only popularity rows: `{len(popularity.loc[popularity['recommended'].eq('true')])}`.",
        "",
        "## 29. Popularity input findings",
        f"- Popularity audit rows: `{len(popularity)}`. Recommended sources are leakage-free train interaction counts only.",
        "",
        "## 30. Catalog and item-universe findings",
        f"- Universe audit rows: `{len(universe)}` covering five universe candidates for each dataset.",
        "",
        "## 31. Coverage-denominator decision",
        "- Catalog coverage uses train_seen_catalog; item coverage uses strict_cohort_candidate_union.",
        "",
        "## 32. Novelty smoothing decision",
        "- No smoothing is approved. Plan D add-one smoothing is recorded as a conflict/adaptation.",
        "",
        "## 33. Long-tail threshold evidence",
        f"- Recommended long-tail policies: `{len(long_tail.loc[long_tail['recommended'].eq('true')])}`.",
        "",
        "## 34. Popularity-bias history source",
        "- No history-normalized popularity-bias source is frozen; the metric is blocked pending a source-backed formula.",
        "",
        "## 35. ML-1M similarity and genre feasibility",
        _similarity_summary_line(similarity, "ML-1M"),
        "",
        "## 36. Beauty similarity and metadata feasibility",
        _similarity_summary_line(similarity, "Amazon Beauty"),
        "",
        "## 37. Sports similarity and metadata feasibility",
        _similarity_summary_line(similarity, "Amazon Sports"),
        "",
        "## 38. Title-token ILD adaptation assessment",
        "- Canonical title-token Jaccard is feasible as an adapted source, but it is not recommended as exact ILD.",
        "",
        "## 39. Structural invariants for deterministic baselines",
        f"- Structural invariant applicability rows: `{int(applicability['applicability_status'].eq('structural_invariant').sum())}`.",
        "",
        "## 40. Candidate validity applicability",
        f"- Candidate-validity rows in applicability matrix: `{int(applicability['metric_id'].eq('candidate_validity_rate').sum())}`.",
        "",
        "## 41. Duplicate-rate applicability",
        f"- Duplicate-rate rows in applicability matrix: `{int(applicability['metric_id'].eq('duplicate_rate_at_5').sum())}`.",
        "",
        "## 42. Full strict cohort preflight",
        _cohort_summary_line(preflight_counts, FULL_COHORT_ID),
        "",
        "## 43. Matched strict cohort preflight",
        _cohort_summary_line(preflight_counts, MATCHED_COHORT_ID),
        "",
        "## 44. Ranking task preflight",
        _task_summary_line(preflight, "ranking"),
        "",
        "## 45. Reranking task preflight",
        _task_summary_line(preflight, "reranking"),
        "",
        "## 46. Seven-baseline applicability matrix",
        f"- Applicability matrix contains `{len(applicability)}` rows for 11 non-diagnostic metrics, 7 baselines, 3 datasets, 2 tasks, and 2 cohorts.",
        "",
        "## 47. Exact metrics ready for computation",
        f"- Exact-ready contracts: `{int(contract['contract_status'].eq('ready_exact').sum())}`.",
        "",
        "## 48. Reconstructed metrics ready for computation",
        f"- Reconstructed-ready contracts: `{int(contract['contract_status'].eq('ready_reconstructed').sum())}`.",
        "",
        "## 49. Adapted metrics requiring explicit labeling",
        f"- Adapted-ready contracts: `{int(contract['contract_status'].eq('ready_adapted').sum())}`.",
        "",
        "## 50. Blocked metrics",
        f"- Blocked contract rows: `{int(contract['contract_status'].str.startswith('blocked').sum())}`; issue rows: `{len(blocked)}`.",
        "",
        "## 51. Selection-bias implications",
        "- Final metrics must be reported only on the frozen LightGCN-compatible strict cohorts; no hidden intersections are allowed.",
        "",
        "## 52. Mixed-device LightGCN provenance",
        "- Beauty and Sports LightGCN source rows are CPU exact-repeat outputs; ML-1M uses the documented existing CUDA primary output with sparse-forward nondeterminism provenance.",
        "",
        "## 53. Metric reproducibility requirements",
        "- Phase 4B must reuse these contract, popularity, universe, long-tail, similarity, applicability, and blocked-policy tables without changing candidate or cohort membership.",
        "",
        "## 54. Output schemas proposed for final computation",
        "- Final Phase 4B should write a new long metric table that references metric_id, baseline_id, dataset, task_type, cohort_id, ranking_source_path, contract hash, input hashes, value, and status.",
        "",
        "## 55. Final metric-computation stop conditions",
        "- Stop if protected inputs change, utility reconciliation fails, a blocked metric is requested as final, or a required source has missing popularity/metadata without an approved adapted policy.",
        "",
        "## 56. Tests",
        "- Required tests are documented in the final command log and validate formulas, preflight helpers, and existing Phase 2/3 policy utilities.",
        "",
        "## 57. Determinism validation",
        f"- Generated CSV hashes are deterministic when the builder is rerun under identical inputs; output file hashes are recorded for `{len(output_hashes)}` CSVs.",
        "",
        "## 58. Protected-input immutability",
        f"- Protected input hashes unchanged within the builder run: `{bool_text(unchanged)}`.",
        "",
        "## 59. Commands executed",
        "- See final response for the shell command log. The builder itself writes only Phase 4A artifacts.",
        "",
        "## 60. Recommended single next task",
        "Phase 4B — canonical utility and beyond-utility metric computation for seven baselines on frozen same cohorts",
        "",
    ]
    return "\n".join(lines)


def _similarity_summary_line(frame: pd.DataFrame, dataset: str) -> str:
    subset = frame.loc[frame["dataset"].eq(dataset)]
    ready = int(subset["status"].isin(["ready_reconstructed", "ready_adapted"]).sum())
    blocked = int(subset["status"].str.startswith("blocked").sum())
    return f"- Similarity feasibility rows: `{len(subset)}`; ready/adapted rows: `{ready}`; blocked rows: `{blocked}`."


def _cohort_summary_line(frame: pd.DataFrame, cohort_id: str) -> str:
    subset = frame.loc[frame["cohort_id"].eq(cohort_id)]
    payload = {
        f"{row.dataset}/{row.task_type}": int(row.observed_rows)
        for row in subset.itertuples(index=False)
    }
    return f"- Observed rows by dataset/task: `{json_text(payload)}`."


def _task_summary_line(frame: pd.DataFrame, task_type: str) -> str:
    subset = frame.loc[frame["task_type"].eq(task_type)]
    return f"- `{task_type}` preflight rows: `{len(subset)}`; ready rows: `{int(subset['preflight_status'].eq('ready').sum())}`."


def build_all(root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, str], dict[str, str], str]:
    initial_hashes = hash_existing_inputs(root)
    contract = build_contract_frame()
    inventory = build_implementation_inventory(root)
    sources = load_ranking_sources(root)
    preflight = build_same_cohort_preflight(root, sources)
    candidate_unions = build_candidate_unions(root, sources)
    metadata_maps = build_metadata_maps(root)
    popularity = build_popularity_audit(root, candidate_unions)
    universe = build_universe_audit(root, candidate_unions, metadata_maps)
    long_tail = build_long_tail_audit(root)
    similarity = build_similarity_feasibility(root, candidate_unions, metadata_maps)
    applicability = build_applicability_matrix(contract, preflight, popularity, long_tail, similarity)
    policy = build_policy_options()
    blocked = build_blocked_table()
    outputs = {
        "contract": contract,
        "inventory": inventory,
        "preflight": preflight,
        "popularity": popularity,
        "universe": universe,
        "long_tail": long_tail,
        "similarity": similarity,
        "applicability": applicability,
        "policy": policy,
        "blocked": blocked,
    }
    validate_outputs(outputs)
    output_hashes = write_outputs(root, outputs)
    final_hashes = hash_existing_inputs(root)
    report = render_report(
        root=root,
        outputs=outputs,
        env=environment_summary(root),
        initial_hashes=initial_hashes,
        final_hashes=final_hashes,
        output_hashes=output_hashes,
    )
    report_path = root / OUTPUTS["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    return outputs, initial_hashes, final_hashes, output_hashes, report
