from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

import pytest

from src.extension import f1a_offline_metric_closure as f1a


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _outputs() -> dict:
    return f1a.generate_f1a(ROOT)


def test_rank_utility_and_novelty_formulas_have_known_values() -> None:
    assert f1a.hit_rate_at_k_from_rank(0) == 1.0
    assert f1a.hit_rate_at_k_from_rank(5) == 0.0
    assert f1a.ndcg_at_k_from_rank(2) == pytest.approx(0.5)
    assert f1a.reciprocal_rank(3) == pytest.approx(0.25)

    top = ["a", "b", "a", "c", "b"]
    counts = {"a": 4, "b": 2, "c": 1}
    assert f1a.self_information_at_k(top, counts, total_train=16) == pytest.approx(2.8)
    assert f1a.average_recommendation_popularity(top, counts) == pytest.approx(2.6)
    assert f1a.long_tail_rate_at_k(top, counts, threshold=3) == pytest.approx(0.6)
    assert f1a.self_information_at_k(["missing"] * 5, counts, total_train=16) is None


def test_oic_shannon_ild_and_calibration_edge_cases_are_deterministic() -> None:
    recommendations = [
        ["a", "b", "c", "d", "e"],
        ["a", "b", "x", "y", "z"],
        ["q", "r", "s", "t", "u"],
    ]
    assert f1a.oic_at_k(recommendations) == pytest.approx((2 / 5) / 3)
    assert f1a.oic_at_k([recommendations[0]]) is None

    features = {
        "i1": ["A", "B"],
        "i2": ["B"],
        "i3": ["A"],
        "i4": ["B"],
        "i5": ["B"],
        "j1": ["C"],
    }
    dist = f1a.weighted_feature_distribution(["i1", "i2"], features)
    assert dist == pytest.approx({"a": 0.25, "b": 0.75})
    assert f1a.shannon_entropy_from_distribution({"only": 1.0}, normalized=True) == 0.0
    assert f1a.shannon_category_entropy_at_k(["i2", "i4", "i5", "i2", "i4"], features) == 0.0

    assert f1a.jaccard_distance(["A", "B"], ["B", "C"]) == pytest.approx(2 / 3)
    assert f1a.ild_jaccard_at_k(["i3", "i3", "i4", "i4", "missing"], features) == pytest.approx(2 / 3)
    assert f1a.ild_jaccard_at_k(["missing"] * 5, features) is None

    assert f1a.calibration_at_k(["i3"], ["i3", "i3", "i3", "i3", "i3"], features) == pytest.approx(1.0)
    assert f1a.calibration_at_k(["i3"], ["j1", "j1", "j1", "j1", "j1"], features) == pytest.approx(0.0)
    assert f1a.calibration_at_k([], ["j1"] * 5, features) is None


def test_exposure_fairness_duplicate_and_ks_edge_cases_are_deterministic() -> None:
    assert f1a.gini_index([]) is None
    assert f1a.gini_index([0, 0, 0]) == 0.0
    assert f1a.gini_index([1, 1, 1]) == pytest.approx(0.0)
    assert f1a.gini_index([0, 0, 10]) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        f1a.gini_index([1, -1])

    assert f1a.jain_index([]) is None
    assert f1a.jain_index([0, 0, 0]) == 0.0
    assert f1a.jain_index([1, 1, 1]) == pytest.approx(1.0)
    assert f1a.jain_index([0, 0, 10]) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        f1a.jain_index([1, -1])

    assert f1a.duplicate_rate_at_k(["a", "a", "b", "b", "c"]) == pytest.approx(0.4)
    assert f1a.duplicate_rate_at_k(["a", "b"]) is None

    identical = f1a.ks_2samp([1, 2, 3], [1, 2, 3])
    assert identical is not None
    assert identical["statistic"] == 0.0
    assert identical["p_value"] == 1.0
    separated = f1a.ks_2samp([1, 1], [9, 9])
    assert separated is not None
    assert separated["statistic"] == 1.0
    assert 0.0 <= separated["p_value"] <= 1.0


def test_f1a_artifact_generation_validates_complete_metric_and_cell_accounting() -> None:
    outputs = _outputs()
    registry = outputs["metric_formula_registry"]
    resolution = outputs["metric_resolution_audit"]
    readiness = outputs["result_cell_readiness"]
    raw_mapping = outputs["raw_source_mapping"]
    validation = outputs["validation"]

    assert outputs["validation_passed"], validation.loc[validation["status"].ne("PASS")].to_dict("records")
    assert len(registry) == 39
    assert set(registry["metric"]) == set(resolution["metric"])
    assert len(readiness) == 8_938
    assert len(raw_mapping) == 8_938
    assert readiness["f1a_status"].isin(f1a.FINAL_READINESS_STATUSES).all()
    assert not readiness.astype(str).apply(lambda column: column.str.contains("UNKNOWN|video_games|video games|videogames", case=False, regex=True)).any().any()
    assert not raw_mapping["excluded_sources"].str.contains("phase_snapshot", case=False, regex=False).any()


def test_f1a_closes_local_implementations_and_retains_only_defensible_blockers() -> None:
    outputs = _outputs()
    registry = outputs["metric_formula_registry"]
    resolution = outputs["metric_resolution_audit"]
    blockers = outputs["remaining_blockers"]

    statuses = dict(zip(registry["metric"], registry["f1a_implementation_status"]))
    for metric in ["oic_at_5", "shannon_category_entropy_at_5", "gini_at_5", "jain_index_at_5", "ks_sample_representativeness"]:
        assert statuses[metric] == "IMPLEMENTED_F1A"

    resolved = dict(zip(resolution["metric"], resolution["f1a_status"]))
    assert resolved["popreo_at_5"] == "BLOCKED_GLOBAL_PROTOCOL"
    assert resolved["dpd_at_5"] == "BLOCKED_GLOBAL_PROTOCOL"
    assert resolved["ks_sample_representativeness"] == "COMPUTED_F1A"
    assert {"popreo_at_5", "dpd_at_5", "ks_sample_representativeness"}.issubset(set(blockers["metric"]))
    assert blockers["reason"].str.len().gt(0).all()


def test_five_dataset_applicability_and_foodcom_tag_policy_are_explicit() -> None:
    applicability = _outputs()["metric_dataset_applicability"]

    assert len(applicability) == 39 * 5
    assert set(applicability["dataset"]) == {"beauty", "sports", "ml1m", "lastfm", "foodcom"}
    foodcom_features = applicability.loc[
        applicability["dataset"].eq("foodcom")
        & applicability["metric"].isin(["shannon_category_entropy_at_5", "ild_at_5", "calibration_at_5"])
    ]
    assert not foodcom_features.empty
    assert foodcom_features["reason"].str.contains("RAW_recipes.tags").all()
    ks_original_four = applicability.loc[
        applicability["metric"].eq("ks_sample_representativeness")
        & applicability["dataset"].isin(["beauty", "sports", "ml1m", "lastfm"])
    ]
    assert ks_original_four["applicability_status"].eq("IMPLEMENTATION_READY_BLOCKED_DATA_REQUIREMENT").all()


def test_offline_results_completeness_budget_and_zero_execution_guards() -> None:
    outputs = _outputs()
    offline = outputs["offline_metric_results"]
    manifest = outputs["manifest"]
    artifacts = f1a.load_f0_artifacts(ROOT)
    budget = {row["budget_item"]: int(row["request_count"]) for row in artifacts["budget"].to_dict("records")}

    hardware_cells = outputs["result_cell_readiness"].loc[
        outputs["result_cell_readiness"]["metric"].eq("hardware_runtime_provenance")
    ]
    assert len(offline) == len(hardware_cells) + 2
    assert offline["adds_llm_requests"].astype(int).sum() == 0
    assert offline.loc[offline["metric"].eq("ks_sample_representativeness"), "dataset"].eq("GLOBAL").all()
    assert offline.loc[offline["metric"].eq("ks_sample_representativeness"), "value_json"].str.contains("foodcom_primary1000|foodcom_history200", regex=True).all()

    assert manifest["llm_inference_calls_executed"] == 0
    assert manifest["training_runs_executed"] == 0
    assert manifest["downloads_executed"] == 0
    assert manifest["network_calls_executed"] == 0
    assert budget["MANDATORY_RAW_REQUESTS_TOTAL"] == 50_292
    assert budget["CONCEPTUAL_ALL_APPLICABLE_REQUESTS_TOTAL"] == 100_584


def test_original_four_and_foodcom_are_separately_accounted() -> None:
    outputs = _outputs()
    original_four = outputs["original_four_completeness"]
    foodcom = outputs["foodcom_completeness"]

    assert original_four["dataset"].tolist() == ["beauty", "sports", "ml1m", "lastfm"]
    assert original_four["status"].eq("ACCOUNTED").all()
    assert foodcom["dataset"].tolist() == ["foodcom"]
    assert foodcom["status"].eq("ACCOUNTED").all()
    assert int(original_four["applicable_result_cells"].sum()) + int(foodcom["applicable_result_cells"].sum()) < 8_938


def test_f1a_source_has_no_model_network_or_training_imports() -> None:
    tree = ast.parse((ROOT / "src/extension/f1a_offline_metric_closure.py").read_text(encoding="utf-8"))
    banned_roots = {"requests", "urllib", "socket", "httpx", "openai", "anthropic", "ollama", "torch", "tensorflow"}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(banned_roots)
    assert "phase_snapshot" not in (ROOT / "src/extension/f1a_offline_metric_closure.py").read_text(encoding="utf-8").lower()
