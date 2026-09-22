"""Canonical metric contract helpers for Phase 4A.

The functions in this module are intentionally small and side-effect free.
They define formulas and validation behavior used by the Phase 4A preflight,
but they do not read project artifacts or compute final project beyond-utility
results.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CUTOFF = 5

CONTRACT_STATUSES = {
    "ready_exact",
    "ready_reconstructed",
    "ready_adapted",
    "structural_invariant",
    "diagnostic_only",
    "blocked_ambiguous_definition",
    "blocked_missing_input",
    "not_applicable",
    "validation_failed",
}

COMPUTATION_READY_STATUSES = {
    "ready_exact",
    "ready_reconstructed",
    "ready_adapted",
    "structural_invariant",
    "diagnostic_only",
}


@dataclass(frozen=True)
class MetricContract:
    metric_id: str
    display_name: str
    metric_family: str
    cutoff: str
    mathematical_formula: str
    input_unit: str
    aggregation_level: str
    denominator_definition: str
    higher_is_better: str
    expected_minimum: str
    expected_maximum: str
    duplicate_handling: str
    invalid_item_handling: str
    missing_metadata_handling: str
    empty_cohort_handling: str
    popularity_source_policy: str
    catalog_universe_policy: str
    long_tail_policy: str
    similarity_source_policy: str
    paper_source: str
    official_code_source: str
    local_code_source: str
    paper_alignment: str
    contract_status: str
    blocking_reason: str
    notes: str


def build_metric_registry() -> list[MetricContract]:
    """Return the Phase 4A canonical metric registry in deterministic order."""

    official = "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py"
    same_cohort = "results/tables/plan_e_lightgcn_same_cohort_baselines_metrics.csv"
    local_plan_d = "src/analysis/plan_d_beyond_utility.py"
    return [
        MetricContract(
            metric_id="hr_at_5",
            display_name="HR@5",
            metric_family="utility",
            cutoff="5",
            mathematical_formula="mean(1[positive_rank_zero_based < 5]) over rows",
            input_unit="one candidate-row ranking with one positive item",
            aggregation_level="cohort macro mean over rows",
            denominator_definition="number of rows in dataset/task/cohort/baseline unit",
            higher_is_better="true",
            expected_minimum="0",
            expected_maximum="1",
            duplicate_handling="ranking duplicates retain positions; hit is binary for the single positive",
            invalid_item_handling="invalid or missing positive rank fails validation",
            missing_metadata_handling="metadata not required",
            empty_cohort_handling="blocked; empty cohort cannot be evaluated",
            popularity_source_policy="not_applicable",
            catalog_universe_policy="candidate row contains the positive and candidates",
            long_tail_policy="not_applicable",
            similarity_source_policy="not_applicable",
            paper_source="paper utility metric HR@5; local PDF present but not text-extracted",
            official_code_source=f"{official}: calHR(hit)",
            local_code_source=f"{same_cohort}; src/evaluation/utility_metrics.py",
            paper_alignment="reconstructed_same_cohort_top5_binary_hit",
            contract_status="ready_reconstructed",
            blocking_reason="",
            notes="Used only for reconciliation in Phase 4A; existing table is not rewritten.",
        ),
        MetricContract(
            metric_id="ndcg_at_5",
            display_name="NDCG@5",
            metric_family="utility",
            cutoff="5",
            mathematical_formula="mean(1/log2(positive_rank_one_based+1) if positive_rank_zero_based < 5 else 0)",
            input_unit="one candidate-row ranking with one positive item",
            aggregation_level="cohort macro mean over rows",
            denominator_definition="number of rows in dataset/task/cohort/baseline unit",
            higher_is_better="true",
            expected_minimum="0",
            expected_maximum="1",
            duplicate_handling="ranking duplicates retain positions; only the positive position contributes",
            invalid_item_handling="invalid or missing positive rank fails validation",
            missing_metadata_handling="metadata not required",
            empty_cohort_handling="blocked; empty cohort cannot be evaluated",
            popularity_source_policy="not_applicable",
            catalog_universe_policy="candidate row contains the positive and candidates",
            long_tail_policy="not_applicable",
            similarity_source_policy="not_applicable",
            paper_source="paper utility metric NDCG@5; local PDF present but not text-extracted",
            official_code_source=f"{official}: calNDCG(hit, rank)",
            local_code_source=f"{same_cohort}; src/evaluation/utility_metrics.py",
            paper_alignment="reconstructed_same_cohort_top5_binary_ndcg",
            contract_status="ready_reconstructed",
            blocking_reason="",
            notes="One relevant item makes IDCG equal to 1.",
        ),
        MetricContract(
            metric_id="mean_positive_rank_zero_based",
            display_name="Mean Positive Rank",
            metric_family="ranking_diagnostic",
            cutoff="all",
            mathematical_formula="mean(positive_rank_zero_based) over rows",
            input_unit="one candidate-row ranking with one positive item",
            aggregation_level="cohort macro mean over rows",
            denominator_definition="number of rows in dataset/task/cohort/baseline unit",
            higher_is_better="false",
            expected_minimum="0",
            expected_maximum="candidate_count - 1",
            duplicate_handling="positive rank is the first occurrence of the positive item in ranked_items",
            invalid_item_handling="missing positive rank fails validation",
            missing_metadata_handling="metadata not required",
            empty_cohort_handling="blocked; empty cohort cannot be evaluated",
            popularity_source_policy="not_applicable",
            catalog_universe_policy="candidate row candidates",
            long_tail_policy="not_applicable",
            similarity_source_policy="not_applicable",
            paper_source="not a paper metric in the audited official metric vector",
            official_code_source=f"{official}: rank helper used for calNDCG/calMRR",
            local_code_source=same_cohort,
            paper_alignment="diagnostic_only",
            contract_status="diagnostic_only",
            blocking_reason="",
            notes="Kept as a ranking diagnostic and comparison sanity check.",
        ),
        MetricContract(
            metric_id="median_positive_rank_zero_based",
            display_name="Median Positive Rank",
            metric_family="ranking_diagnostic",
            cutoff="all",
            mathematical_formula="median(positive_rank_zero_based) over rows",
            input_unit="one candidate-row ranking with one positive item",
            aggregation_level="cohort median over rows",
            denominator_definition="number of rows in dataset/task/cohort/baseline unit",
            higher_is_better="false",
            expected_minimum="0",
            expected_maximum="candidate_count - 1",
            duplicate_handling="positive rank is the first occurrence of the positive item in ranked_items",
            invalid_item_handling="missing positive rank fails validation",
            missing_metadata_handling="metadata not required",
            empty_cohort_handling="blocked; empty cohort cannot be evaluated",
            popularity_source_policy="not_applicable",
            catalog_universe_policy="candidate row candidates",
            long_tail_policy="not_applicable",
            similarity_source_policy="not_applicable",
            paper_source="not a paper metric in the audited official metric vector",
            official_code_source=f"{official}: rank helper used for calNDCG/calMRR",
            local_code_source=same_cohort,
            paper_alignment="diagnostic_only",
            contract_status="diagnostic_only",
            blocking_reason="",
            notes="Kept as a ranking diagnostic and comparison sanity check.",
        ),
        MetricContract(
            metric_id="novelty_at_5",
            display_name="Novelty@5",
            metric_family="beyond_utility",
            cutoff="5",
            mathematical_formula="mean(-log2(train_count(item)/total_train_interactions)) over top-5 items",
            input_unit="top-5 item ids plus train-only item popularity counts",
            aggregation_level="micro mean over top-5 recommendation positions within the cohort",
            denominator_definition="5 * number of cohort rows when all rows have five recommendations",
            higher_is_better="true",
            expected_minimum="0",
            expected_maximum="unbounded above for very rare positive-probability train items",
            duplicate_handling="duplicates retain their positions and contribute repeatedly",
            invalid_item_handling="invalid or zero-train-count items block the metric for final computation; no smoothing is invented",
            missing_metadata_handling="metadata not required",
            empty_cohort_handling="blocked; empty cohort cannot be evaluated",
            popularity_source_policy="train-only raw interaction counts normalized by total train interactions",
            catalog_universe_policy="train_seen_catalog for positive probability support",
            long_tail_policy="not_applicable",
            similarity_source_policy="not_applicable",
            paper_source="paper novelty/self-information dimension; exact equation not text-extracted locally",
            official_code_source=f"{official}: calSelfInfo(results, metric_len, information)",
            local_code_source=local_plan_d,
            paper_alignment="adapted_from_official_self_information_without_plan_d_smoothing",
            contract_status="ready_adapted",
            blocking_reason="",
            notes="Official code uses an information dictionary; Plan D used add-one smoothing. Phase 4A selects no smoothing and blocks zero popularity.",
        ),
        MetricContract(
            metric_id="catalog_coverage_at_5",
            display_name="Catalog Coverage@5",
            metric_family="beyond_utility",
            cutoff="5",
            mathematical_formula="|unique valid top-5 recommended items| / |train_seen_catalog|",
            input_unit="cohort top-5 recommendations and train-seen catalog universe",
            aggregation_level="cohort-level unique-item ratio",
            denominator_definition="unique train-seen items for the dataset",
            higher_is_better="true",
            expected_minimum="0",
            expected_maximum="1",
            duplicate_handling="duplicates counted once in numerator",
            invalid_item_handling="invalid items excluded from numerator and surfaced by candidate_validity_rate",
            missing_metadata_handling="metadata not required",
            empty_cohort_handling="0 numerator but empty cohort blocks final computation",
            popularity_source_policy="not_applicable",
            catalog_universe_policy="train_seen_catalog",
            long_tail_policy="not_applicable",
            similarity_source_policy="not_applicable",
            paper_source="paper/report coverage terminology ambiguous between catalog and item coverage",
            official_code_source=f"{official}: calItemCoverage uses sampled candidate pool denominator, not train catalog",
            local_code_source="src/evaluation/coverage.py; src/analysis/plan_d_beyond_utility.py",
            paper_alignment="adapted_extension_distinct_from_official_item_coverage",
            contract_status="ready_adapted",
            blocking_reason="",
            notes="Explicitly labeled as adapted catalog coverage, not the official ItemCoverage formula.",
        ),
        MetricContract(
            metric_id="item_coverage_at_5",
            display_name="Candidate-Union Item Coverage@5",
            metric_family="beyond_utility",
            cutoff="5",
            mathematical_formula="|unique top-5 recommended items| / |strict_cohort_candidate_union|",
            input_unit="cohort top-5 recommendations and strict candidate union",
            aggregation_level="cohort-level unique-item ratio",
            denominator_definition="unique candidates available in the same dataset/task/cohort unit",
            higher_is_better="true",
            expected_minimum="0",
            expected_maximum="1",
            duplicate_handling="duplicates counted once in numerator",
            invalid_item_handling="invalid items excluded from numerator and surfaced by candidate_validity_rate",
            missing_metadata_handling="metadata not required",
            empty_cohort_handling="blocked; empty cohort cannot be evaluated",
            popularity_source_policy="not_applicable",
            catalog_universe_policy="strict_cohort_candidate_union",
            long_tail_policy="not_applicable",
            similarity_source_policy="not_applicable",
            paper_source="paper/report item coverage terminology maps most closely to official candidate-pool coverage",
            official_code_source=f"{official}: calItemCoverage(id_results, metric_len, negative_samples, test_samples)",
            local_code_source=local_plan_d,
            paper_alignment="reconstructed_candidate_pool_coverage_on_strict_same_cohort",
            contract_status="ready_reconstructed",
            blocking_reason="",
            notes="Named candidate-union item coverage in prose to avoid conflating it with full catalog coverage.",
        ),
        MetricContract(
            metric_id="long_tail_rate_at_5",
            display_name="Long-Tail Rate@5",
            metric_family="beyond_utility",
            cutoff="5",
            mathematical_formula="mean(1[train_count(item) < quantile_0.8(positive_train_counts)]) over top-5 items",
            input_unit="top-5 item ids and train-only item popularity counts",
            aggregation_level="micro mean over top-5 recommendation positions within the cohort",
            denominator_definition="5 * number of cohort rows when all rows have five recommendations",
            higher_is_better="true",
            expected_minimum="0",
            expected_maximum="1",
            duplicate_handling="duplicates retain their positions and contribute repeatedly",
            invalid_item_handling="invalid or non-train-seen items are not long-tail under the reconstructed official rule",
            missing_metadata_handling="metadata not required",
            empty_cohort_handling="blocked; empty cohort cannot be evaluated",
            popularity_source_policy="train-only raw interaction counts",
            catalog_universe_policy="train_seen_catalog with positive train count",
            long_tail_policy="official ACLT: strict less-than 80th percentile of train frequencies",
            similarity_source_policy="not_applicable",
            paper_source="paper ACLT/long-tail metric; exact prose not text-extracted locally",
            official_code_source=f"{official}: calACLT(results, freq_dict, high_bar, metric_len)",
            local_code_source=local_plan_d,
            paper_alignment="reconstructed_from_official_code",
            contract_status="ready_reconstructed",
            blocking_reason="",
            notes="Threshold is frozen from training counts before recommendation results.",
        ),
        MetricContract(
            metric_id="average_popularity_at_5",
            display_name="Average Popularity@5",
            metric_family="beyond_utility",
            cutoff="5",
            mathematical_formula="mean(train_count(item)) over top-5 recommendation positions",
            input_unit="top-5 item ids and train-only item popularity counts",
            aggregation_level="micro mean over top-5 recommendation positions within the cohort",
            denominator_definition="5 * number of cohort rows when all rows have five recommendations",
            higher_is_better="false",
            expected_minimum="0",
            expected_maximum="maximum train item count",
            duplicate_handling="duplicates retain positions and contribute repeatedly",
            invalid_item_handling="invalid items receive no silent imputation; candidate-constrained rows should validate first",
            missing_metadata_handling="metadata not required",
            empty_cohort_handling="blocked; empty cohort cannot be evaluated",
            popularity_source_policy="train-only raw interaction counts",
            catalog_universe_policy="train_seen_catalog",
            long_tail_policy="not_applicable",
            similarity_source_policy="not_applicable",
            paper_source="paper popularity/ARP metric; exact prose not text-extracted locally",
            official_code_source=f"{official}: calARP(freq_dict, id_results, metric_len)",
            local_code_source=local_plan_d,
            paper_alignment="reconstructed_from_official_code",
            contract_status="ready_reconstructed",
            blocking_reason="",
            notes="Kept distinct from popularity bias; it is not a profile-normalized difference.",
        ),
        MetricContract(
            metric_id="popularity_bias_at_5",
            display_name="Popularity Bias@5",
            metric_family="beyond_utility",
            cutoff="5",
            mathematical_formula="blocked: source evidence conflicts between average popularity, PopREO, and local candidate/catalog subtraction",
            input_unit="not frozen",
            aggregation_level="not frozen",
            denominator_definition="not frozen",
            higher_is_better="false",
            expected_minimum="not_frozen",
            expected_maximum="not_frozen",
            duplicate_handling="not frozen",
            invalid_item_handling="not frozen",
            missing_metadata_handling="metadata not required",
            empty_cohort_handling="blocked",
            popularity_source_policy="train-only counts required if a future formula is approved",
            catalog_universe_policy="not frozen",
            long_tail_policy="not_applicable",
            similarity_source_policy="not_applicable",
            paper_source="paper popularity-bias terminology not locally disambiguated",
            official_code_source=f"{official}: calARP and calPopREO are distinct popularity/fairness calculations",
            local_code_source=local_plan_d,
            paper_alignment="blocked_conflicting_definitions",
            contract_status="blocked_ambiguous_definition",
            blocking_reason="Average popularity and popularity bias cannot be assigned the same formula; official and local evidence support different semantics.",
            notes="Final computation must not duplicate average_popularity_at_5 under this metric ID.",
        ),
        MetricContract(
            metric_id="intra_list_diversity_at_5",
            display_name="Intra-List Diversity@5",
            metric_family="beyond_utility",
            cutoff="5",
            mathematical_formula="blocked exact metric: item representation and similarity source are not uniquely identified by local paper/official evidence",
            input_unit="top-5 item ids plus item feature vectors or sets",
            aggregation_level="mean over row-level pairwise distances, then cohort mean",
            denominator_definition="10 unordered item pairs for rows with five feature-covered items",
            higher_is_better="true",
            expected_minimum="0",
            expected_maximum="1 for Jaccard distance; feature-dependent otherwise",
            duplicate_handling="duplicate item pair distance is 0 if the same item is compared",
            invalid_item_handling="invalid or missing-feature items block exact ILD for that row unless an adapted policy is approved",
            missing_metadata_handling="do not silently impute; report feature coverage and adaptation",
            empty_cohort_handling="blocked; empty cohort cannot be evaluated",
            popularity_source_policy="not_applicable",
            catalog_universe_policy="metadata_covered_catalog or strict_candidate_union with explicit missing-feature policy",
            long_tail_policy="not_applicable",
            similarity_source_policy="blocked exact; ML-1M genres and Amazon categories/title tokens are feasible adaptations",
            paper_source="paper diversity dimension; exact representation not text-extracted locally",
            official_code_source=f"{official}: calMeanShannon and calMeanJaccard, not standard within-list ILD",
            local_code_source=local_plan_d,
            paper_alignment="blocked_exact_adapted_sources_available",
            contract_status="blocked_ambiguous_definition",
            blocking_reason="Exact item representation and similarity function are ambiguous; title-token or category Jaccard would be adapted.",
            notes="Similarity feasibility table records dataset-specific adapted options without computing ILD values.",
        ),
        MetricContract(
            metric_id="duplicate_rate_at_5",
            display_name="Duplicate Rate@5",
            metric_family="structural_quality",
            cutoff="5",
            mathematical_formula="duplicate top-5 positions after first occurrence / 5",
            input_unit="one row top-5 ranked item ids",
            aggregation_level="cohort macro mean over rows",
            denominator_definition="5 positions per row",
            higher_is_better="false",
            expected_minimum="0",
            expected_maximum="1",
            duplicate_handling="metric counts repeated items after the first occurrence",
            invalid_item_handling="malformed missing positions count as invalid elsewhere, not duplicate matches",
            missing_metadata_handling="metadata not required",
            empty_cohort_handling="blocked; empty cohort cannot be evaluated",
            popularity_source_policy="not_applicable",
            catalog_universe_policy="candidate row candidates",
            long_tail_policy="not_applicable",
            similarity_source_policy="not_applicable",
            paper_source="not an audited paper utility metric; useful LLM output-quality diagnostic",
            official_code_source="not present as a named official metric",
            local_code_source=local_plan_d,
            paper_alignment="structural_invariant_for_deterministic_candidate_permutations",
            contract_status="structural_invariant",
            blocking_reason="",
            notes="Expected value is 0 for seven deterministic baselines if ranking_valid holds.",
        ),
        MetricContract(
            metric_id="candidate_validity_rate",
            display_name="Candidate Validity Rate",
            metric_family="structural_quality",
            cutoff="5",
            mathematical_formula="valid top-5 positions whose item id is in the row candidate set / 5",
            input_unit="one row top-5 ranked item ids and row candidate list",
            aggregation_level="cohort macro mean over rows",
            denominator_definition="5 positions per row",
            higher_is_better="true",
            expected_minimum="0",
            expected_maximum="1",
            duplicate_handling="duplicates can still be valid candidates; duplicate metric reports repetition separately",
            invalid_item_handling="out-of-candidate, malformed, or missing positions count as invalid",
            missing_metadata_handling="metadata not required",
            empty_cohort_handling="blocked; empty cohort cannot be evaluated",
            popularity_source_policy="not_applicable",
            catalog_universe_policy="row candidate set",
            long_tail_policy="not_applicable",
            similarity_source_policy="not_applicable",
            paper_source="hallucination/validity concern from LLM evaluation, not a traditional baseline differentiator",
            official_code_source=f"{official}: calHallucination maps unmatched outputs to item id 0",
            local_code_source="src/evaluation/hallucination.py; src/analysis/plan_d_beyond_utility.py",
            paper_alignment="structural_invariant_for_candidate_constrained_deterministic_rankings",
            contract_status="structural_invariant",
            blocking_reason="",
            notes="Expected value is 1 for seven deterministic candidate-constrained baselines if top-5 extraction is valid.",
        ),
    ]


def contract_sha256(contract: MetricContract) -> str:
    """Return a stable SHA256 over the contract payload excluding no fields."""

    payload = json.dumps(asdict(contract), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def contract_row(contract: MetricContract) -> dict[str, str]:
    row = {key: str(value) for key, value in asdict(contract).items()}
    row["contract_sha256"] = contract_sha256(contract)
    return row


def validate_metric_registry(registry: Sequence[MetricContract]) -> None:
    """Validate uniqueness and internal consistency of metric contracts."""

    seen: set[str] = set()
    for contract in registry:
        if contract.metric_id in seen:
            raise ValueError(f"duplicate metric_id: {contract.metric_id}")
        seen.add(contract.metric_id)
        if contract.contract_status not in CONTRACT_STATUSES:
            raise ValueError(f"invalid contract_status for {contract.metric_id}: {contract.contract_status}")
        if contract.cutoff not in {"5", "all"}:
            raise ValueError(f"invalid cutoff for {contract.metric_id}: {contract.cutoff}")
        if contract.higher_is_better not in {"true", "false", "not_applicable"}:
            raise ValueError(f"invalid direction for {contract.metric_id}: {contract.higher_is_better}")
        if contract.contract_status.startswith("blocked") and not contract.blocking_reason:
            raise ValueError(f"blocked metric lacks blocking_reason: {contract.metric_id}")
        if contract.contract_status in COMPUTATION_READY_STATUSES and contract.contract_status.startswith("blocked"):
            raise ValueError(f"blocked metric marked computation ready: {contract.metric_id}")
        if contract.contract_status in {"ready_exact", "ready_reconstructed", "ready_adapted"}:
            if not contract.mathematical_formula or contract.mathematical_formula.startswith("blocked"):
                raise ValueError(f"ready metric lacks a formula: {contract.metric_id}")


def validate_cutoff(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("cutoff must be a positive integer")


def first_positive_rank_zero_based(ranked_items: Sequence[Any], positive_item: Any) -> int | None:
    """Return the first zero-based positive rank, or ``None`` when absent."""

    target = str(positive_item)
    for index, item in enumerate(ranked_items):
        if str(item) == target:
            return index
    return None


def positive_occurrence_count(ranked_items: Sequence[Any], positive_item: Any) -> int:
    target = str(positive_item)
    return sum(1 for item in ranked_items if str(item) == target)


def validate_one_positive(candidates: Sequence[Any], positive_item: Any) -> None:
    count = positive_occurrence_count(candidates, positive_item)
    if count != 1:
        raise ValueError(f"expected exactly one positive occurrence, found {count}")


def rank_zero_to_one_based(rank_zero_based: int) -> int:
    if rank_zero_based < 0:
        raise ValueError("rank must be non-negative")
    return rank_zero_based + 1


def hr_at_k_from_rank(rank_zero_based: int | None, k: int = CUTOFF) -> float:
    validate_cutoff(k)
    if rank_zero_based is None:
        return 0.0
    if rank_zero_based < 0:
        raise ValueError("rank must be non-negative")
    return 1.0 if rank_zero_based < k else 0.0


def ndcg_at_k_from_rank(rank_zero_based: int | None, k: int = CUTOFF) -> float:
    validate_cutoff(k)
    if rank_zero_based is None:
        return 0.0
    if rank_zero_based < 0:
        raise ValueError("rank must be non-negative")
    if rank_zero_based >= k:
        return 0.0
    rank_one_based = rank_zero_to_one_based(rank_zero_based)
    return 1.0 / math.log2(rank_one_based + 1.0)


def macro_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("empty cohort cannot be evaluated")
    return float(np.mean(np.asarray(values, dtype=float)))


def utility_from_ranks(ranks_zero_based: Sequence[int | None], k: int = CUTOFF) -> dict[str, float]:
    if not ranks_zero_based:
        raise ValueError("empty cohort cannot be evaluated")
    hr_values = [hr_at_k_from_rank(rank, k) for rank in ranks_zero_based]
    ndcg_values = [ndcg_at_k_from_rank(rank, k) for rank in ranks_zero_based]
    return {"hr_at_5": macro_mean(hr_values), "ndcg_at_5": macro_mean(ndcg_values)}


def self_information_from_probability(probability: float) -> float:
    if probability <= 0.0 or probability > 1.0 or math.isnan(probability):
        raise ValueError("self-information probability must be in (0, 1]")
    return -math.log2(probability)


def novelty_self_information(count: int | float, total_train_interactions: int | float) -> float:
    if count <= 0:
        raise ValueError("zero-popularity item cannot be smoothed silently")
    if total_train_interactions <= 0:
        raise ValueError("total_train_interactions must be positive")
    return self_information_from_probability(float(count) / float(total_train_interactions))


def catalog_coverage_ratio(recommended_items: Iterable[Any], universe_items: Iterable[Any]) -> float:
    universe = {str(item) for item in universe_items if str(item) != ""}
    if not universe:
        raise ValueError("coverage universe is empty")
    recommended = {str(item) for item in recommended_items if str(item) != ""}
    return len(recommended.intersection(universe)) / len(universe)


def item_coverage_count(recommended_items: Iterable[Any]) -> int:
    return len({str(item) for item in recommended_items if str(item) != ""})


def build_long_tail_set(
    popularity_counts: Mapping[Any, int | float],
    *,
    quantile: float = 0.8,
    threshold: float | None = None,
) -> tuple[set[str], float]:
    """Return long-tail item IDs using strict less-than threshold semantics."""

    if threshold is None:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be in [0, 1]")
        positive_counts = [float(value) for value in popularity_counts.values() if float(value) > 0.0]
        if not positive_counts:
            raise ValueError("cannot derive long-tail threshold without positive counts")
        threshold = float(np.quantile(np.asarray(positive_counts, dtype=float), quantile))
    tail = {str(item) for item, count in popularity_counts.items() if float(count) > 0.0 and float(count) < threshold}
    return tail, float(threshold)


def long_tail_rate(items: Sequence[Any], long_tail_items: Iterable[Any]) -> float:
    if not items:
        raise ValueError("empty recommendation list cannot be evaluated")
    tail = {str(item) for item in long_tail_items}
    return sum(1 for item in items if str(item) in tail) / len(items)


def average_popularity(items: Sequence[Any], popularity_counts: Mapping[Any, int | float]) -> float:
    if not items:
        raise ValueError("empty recommendation list cannot be evaluated")
    counts = [float(popularity_counts.get(str(item), 0.0)) for item in items]
    return float(np.mean(counts))


def popularity_bias_difference(
    recommendation_items: Sequence[Any],
    profile_items: Sequence[Any],
    popularity_counts: Mapping[Any, int | float],
) -> float:
    """Return recommendation mean popularity minus profile-history mean popularity."""

    if not profile_items:
        raise ValueError("profile history is empty")
    return average_popularity(recommendation_items, popularity_counts) - average_popularity(
        profile_items, popularity_counts
    )


def jaccard_distance(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        raise ValueError("cannot compare two empty feature sets")
    union = {str(item) for item in left}.union(str(item) for item in right)
    if not union:
        raise ValueError("cannot compare two empty feature sets")
    intersection = {str(item) for item in left}.intersection(str(item) for item in right)
    return 1.0 - (len(intersection) / len(union))


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.shape != right_arr.shape:
        raise ValueError("cosine vectors must have the same shape")
    left_norm = float(np.linalg.norm(left_arr))
    right_norm = float(np.linalg.norm(right_arr))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("zero-vector item cannot be compared")
    similarity = float(np.dot(left_arr, right_arr) / (left_norm * right_norm))
    return 1.0 - similarity


def intra_list_diversity_jaccard(items: Sequence[Any], feature_sets: Mapping[Any, set[Any]]) -> float:
    if len(items) < 2:
        raise ValueError("at least two items are required for ILD")
    distances: list[float] = []
    for left, right in combinations([str(item) for item in items], 2):
        if left == right:
            distances.append(0.0)
            continue
        left_features = feature_sets.get(left)
        right_features = feature_sets.get(right)
        if not left_features or not right_features:
            raise ValueError(f"missing feature set for pair {left}, {right}")
        distances.append(jaccard_distance(left_features, right_features))
    return macro_mean(distances)


def duplicate_rate_at_k(items: Sequence[Any], k: int = CUTOFF) -> float:
    validate_cutoff(k)
    top = [str(item) for item in items[:k]]
    seen: set[str] = set()
    duplicate_positions = 0
    for item in top:
        if item in seen:
            duplicate_positions += 1
        seen.add(item)
    missing_positions = max(0, k - len(top))
    return (duplicate_positions + missing_positions) / k


def candidate_validity_rate_at_k(
    recommendations: Sequence[Any],
    candidate_items: Iterable[Any],
    *,
    k: int = CUTOFF,
) -> float:
    validate_cutoff(k)
    candidates = {str(item) for item in candidate_items if str(item) != ""}
    valid = 0
    for item in recommendations[:k]:
        text = str(item)
        if text and text in candidates:
            valid += 1
    return valid / k


def exact_row_keys(rows: Iterable[Mapping[str, Any]], key_columns: Sequence[str]) -> set[tuple[str, ...]]:
    return {tuple(str(row[column]) for column in key_columns) for row in rows}


def same_row_keys(
    left_rows: Iterable[Mapping[str, Any]],
    right_rows: Iterable[Mapping[str, Any]],
    key_columns: Sequence[str],
) -> bool:
    return exact_row_keys(left_rows, key_columns) == exact_row_keys(right_rows, key_columns)


def same_candidate_lists(
    left: Mapping[tuple[str, ...], Sequence[Any]],
    right: Mapping[tuple[str, ...], Sequence[Any]],
) -> bool:
    if set(left) != set(right):
        return False
    return all([str(item) for item in left[key]] == [str(item) for item in right[key]] for key in left)
