from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.extension import f1arr_popreo_dpd_recovery as f1arr


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def f1arr_outputs() -> dict[str, object]:
    return f1arr.generate_f1arr(ROOT)


def reference_dpd(ranks: list[int], lengths: list[int], median: float, k: int = 5) -> float:
    ndcg = np.array([int(0 < rank <= k) / np.log2(rank + 1) for rank in ranks], dtype=float)
    cold = [index for index, length in enumerate(lengths) if length <= median]
    hot = [index for index, length in enumerate(lengths) if length > median]
    return float(abs(ndcg[hot].mean() - ndcg[cold].mean()))


def reference_popreo(hits: list[int], targets: list[str], freq: dict[str, int]) -> float:
    quantiles = [float(np.quantile(list(freq.values()), q)) for q in [0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    quantiles[0] -= 1.0
    groups = []
    for group in range(5):
        groups.append(
            [
                index
                for index, target in enumerate(targets)
                if quantiles[group] < freq.get(str(target), 0) <= quantiles[group + 1]
            ]
        )
    group_hits = [np.array(hits)[indices].mean() for indices in groups]
    return float(np.std(group_hits) / np.mean(group_hits))


def test_dpd_matches_repository_transliteration_and_absolute_difference() -> None:
    ranks = [1, 7, 4]
    lengths = [3, 3, 5]
    result = f1arr.dpd_at_k_from_ranks(ranks, lengths, median_history_length=3, k=5)

    assert result["computation_status"] == "COMPUTED"
    assert result["inactive_user_N"] == 2
    assert result["active_user_N"] == 1
    assert result["metric_value"] == pytest.approx(reference_dpd(ranks, lengths, median=3))


def test_dpd_median_ties_are_inactive_and_strictly_greater_is_active() -> None:
    result = f1arr.dpd_at_k_from_ranks([1, 2, 3, 4], [2, 4, 4, 8], median_history_length=4, k=5)

    assert result["inactive_user_N"] == 3
    assert result["active_user_N"] == 1
    assert result["median_history_length"] == 4


def test_dpd_empty_group_is_degenerate_without_smoothing() -> None:
    result = f1arr.dpd_at_k_from_ranks([1, 2], [20, 21], median_history_length=10, k=5)

    assert result["computation_status"] == "DEGENERATE_PROTOCOL_INPUT"
    assert math.isnan(result["metric_value"])
    assert result["inactive_user_N"] == 0
    assert result["active_user_N"] == 2


def test_popreo_quantiles_and_boundary_inclusivity_follow_repository() -> None:
    freq = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    quantiles = f1arr.popreo_quantile_boundaries(freq)

    assert quantiles == pytest.approx([0.0, 1.8, 2.6, 3.4, 4.2, 5.0])
    assert f1arr.popreo_group_for_target("b", freq, [0, 2, 4, 6, 8, 10]) == 1
    assert f1arr.popreo_group_for_target("c", freq, [0, 2, 4, 6, 8, 10]) == 2


def test_popreo_coefficient_of_variation_uses_numpy_population_std() -> None:
    freq = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    targets = ["a", "b", "c", "d", "e"]
    hits = [1, 0, 1, 0, 1]

    result = f1arr.popreo_at_k_from_hits(hits, targets, freq)

    assert result["computation_status"] == "COMPUTED"
    assert result["metric_value"] == pytest.approx(reference_popreo(hits, targets, freq))


def test_popreo_empty_group_and_zero_mean_are_degenerate_protocol_inputs() -> None:
    freq = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    empty = f1arr.popreo_at_k_from_hits([1, 0], ["a", "e"], freq)
    zero_mean = f1arr.popreo_at_k_from_hits([0, 0, 0, 0, 0], ["a", "b", "c", "d", "e"], freq)

    assert empty["computation_status"] == "DEGENERATE_PROTOCOL_INPUT"
    assert math.isnan(empty["metric_value"])
    assert zero_mean["computation_status"] == "DEGENERATE_PROTOCOL_INPUT"
    assert math.isnan(zero_mean["metric_value"])
    assert "zero denominator" in zero_mean["reason"]


def test_f1arr_recovers_protocols_and_removes_global_blockers(f1arr_outputs: dict[str, object]) -> None:
    validation = f1arr_outputs["validation"]
    updated = f1arr_outputs["updated_f1a"]
    readiness = updated["result_cell_readiness"]
    blockers = f1arr_outputs["final_metric_blockers"]
    manifest = f1arr_outputs["manifest"]

    assert validation["status"].eq("PASS").all()
    assert blockers.empty
    assert not readiness.loc[readiness["metric"].isin(["popreo_at_5", "dpd_at_5"]), "f1a_status"].eq("BLOCKED_GLOBAL_PROTOCOL").any()
    assert set(readiness["dataset"]) == {"beauty", "sports", "ml1m", "lastfm", "foodcom", "GLOBAL"}
    assert not readiness.astype(str).apply(lambda column: column.str.contains("video_games|video games|videogames", case=False, regex=True)).any().any()
    assert manifest["llm_inference_calls_executed"] == 0
    assert manifest["training_runs_executed"] == 0
    assert manifest["downloads_executed"] == 0
    assert manifest["network_calls_executed"] == 0
    assert manifest["f0_raw_request_budget_changed"] == 0
