from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.finalization_f6d1_lastfm_baselines import (
    BASELINE_ORDER,
    BASELINE_PROTOCOL_COLUMNS,
    EXECUTION_MANIFEST_COLUMNS,
    INSTANCE_SUMMARY_COLUMNS,
    MISSING_UPDATE_COLUMNS,
    OUTPUT_TABLES,
    PAPER_ALIGNMENT_COLUMNS,
    PREDICTION_COLUMNS,
    PREDICTION_VALIDATION_COLUMNS,
    READINESS_UPDATE_COLUMNS,
    RERANKING_SOURCE_COLUMNS,
    UTILITY_METRIC_COLUMNS,
    build_baseline_protocol,
    build_frames,
    build_instance_summary,
    build_missing_evidence_update,
    build_predictions,
    build_report,
    build_utility_metrics,
    file_sha256,
    final_decision,
    frame_sha256,
    no_absolute_paths,
    normalize_candidates,
    protected_hash_status,
    score_candidate_frame,
    stable_random_score,
    write_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def frames() -> dict[str, pd.DataFrame]:
    return build_frames(PROJECT_ROOT)


def synthetic_inputs() -> dict[str, pd.DataFrame]:
    train = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "1", "split": "train", "rating": "1"},
            {"user_id": "u1", "item_id": "2", "split": "train", "rating": "1"},
            {"user_id": "u2", "item_id": "2", "split": "train", "rating": "1"},
            {"user_id": "u2", "item_id": "3", "split": "train", "rating": "1"},
            {"user_id": "u2", "item_id": "3", "split": "test", "rating": "1"},
        ]
    )
    catalog = pd.DataFrame(
        {
            "item_id": ["1", "2", "3", "10"],
            "title": ["Rock Alpha", "Rock Beta", "Jazz Gamma", "Rock Delta"],
            "categories": ["artist", "artist", "artist", "artist"],
            "tag_values": ["rock|live", "rock", "jazz", "rock|delta"],
            "normalized_text": ["rock alpha artist", "rock beta artist", "jazz gamma artist", "rock delta artist"],
        }
    )
    candidates = pd.DataFrame(
        [
            {"candidate_set_id": "set-u1", "task": "ranking", "user_id": "u1", "ground_truth_item_id": "10", "candidate_item_id": item, "candidate_position": pos, "is_ground_truth": str(item == "10").lower(), "candidate_count": "4", "positive_count": "1"}
            for pos, item in enumerate(["3", "10", "1", "2"], start=1)
        ]
        + [
            {"candidate_set_id": "set-u2", "task": "ranking", "user_id": "u2", "ground_truth_item_id": "1", "candidate_item_id": item, "candidate_position": pos, "is_ground_truth": str(item == "1").lower(), "candidate_count": "4", "positive_count": "1"}
            for pos, item in enumerate(["1", "2", "3", "10"], start=1)
        ]
    )
    empty = pd.DataFrame()
    return {
        "train": train,
        "catalog": catalog,
        "ranking_candidates": candidates,
        "metadata_coverage": pd.DataFrame([{"dataset_stage": "final_selected_dataset", "tag_coverage": "75.000000"}]),
        "validation": empty,
        "test": empty,
        "validation_split": empty,
        "ranking_cohort": empty,
        "preprocessing_policy": empty,
        "split_policy": empty,
        "cohort_policy": empty,
        "candidate_policy": empty,
        "readiness_matrix": empty,
        "execution_plan": empty,
        "anomalies": empty,
    }


def test_f6c_candidate_parsing_and_random_stability() -> None:
    candidates = normalize_candidates(synthetic_inputs()["ranking_candidates"])
    assert candidates.groupby("candidate_set_id").size().eq(4).all()
    assert stable_random_score(42, "LastFM", "ranking", "s", "u", "i") == stable_random_score(42, "LastFM", "ranking", "s", "u", "i")
    assert stable_random_score(42, "LastFM", "ranking", "s", "u1", "i") != stable_random_score(42, "LastFM", "ranking", "s", "u2", "i")


def test_mostpop_train_only_counts_and_tie_breaking() -> None:
    inputs = synthetic_inputs()
    scored = score_candidate_frame(inputs["ranking_candidates"], inputs["train"], inputs["catalog"], "mostpop_train_only")
    u1 = scored.loc[scored["user_id"].eq("u1")]
    assert dict(zip(u1["candidate_item_id"], u1["score"]))["2"] == 2.0
    assert dict(zip(u1["candidate_item_id"], u1["score"]))["3"] == 1.0
    predictions = build_predictions(inputs)
    mostpop_u1 = predictions.loc[(predictions["model"].eq("mostpop_train_only")) & (predictions["user_id"].eq("u1"))]
    assert mostpop_u1.sort_values("rank")["candidate_item_id"].tolist()[0] == "2"


def test_bm25_scoring_fixture_and_tie_breaking() -> None:
    inputs = synthetic_inputs()
    scored = score_candidate_frame(inputs["ranking_candidates"], inputs["train"], inputs["catalog"], "bm25_like_adapted")
    u1_scores = dict(zip(scored.loc[scored["user_id"].eq("u1"), "candidate_item_id"], scored.loc[scored["user_id"].eq("u1"), "score"]))
    assert u1_scores["10"] > u1_scores["3"]
    predictions = build_predictions(inputs)
    assert set(predictions["model"]) == set(BASELINE_ORDER)
    assert predictions.groupby(["model", "user_id"]).size().eq(4).all()


def test_complete_ordering_one_positive_and_metrics_on_synthetic() -> None:
    inputs = synthetic_inputs()
    predictions = build_predictions(inputs)
    summary = build_instance_summary(predictions, inputs["catalog"])
    metrics = build_utility_metrics(predictions, summary)
    assert list(predictions.columns) == PREDICTION_COLUMNS
    assert summary["positive_count"].astype(int).eq(1).all()
    assert summary["duplicate_count"].astype(int).eq(0).all()
    assert {"HR@5", "NDCG@5", "mean_positive_rank", "median_positive_rank"}.issubset(set(metrics["metric"]))
    assert pd.to_numeric(metrics.loc[metrics["metric"].eq("HR@5"), "value"]).between(0, 1).all()


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_real_output_schemas_and_validation() -> None:
    built = frames()
    assert list(built["baseline_protocol"].columns) == BASELINE_PROTOCOL_COLUMNS
    assert list(built["execution_manifest"].columns) == EXECUTION_MANIFEST_COLUMNS
    assert list(built["predictions"].columns) == PREDICTION_COLUMNS
    assert list(built["instance_summary"].columns) == INSTANCE_SUMMARY_COLUMNS
    assert list(built["utility_metrics"].columns) == UTILITY_METRIC_COLUMNS
    assert list(built["prediction_validation"].columns) == PREDICTION_VALIDATION_COLUMNS
    assert list(built["reranking_source_audit"].columns) == RERANKING_SOURCE_COLUMNS
    assert list(built["paper_alignment"].columns) == PAPER_ALIGNMENT_COLUMNS
    assert list(built["readiness_update"].columns) == READINESS_UPDATE_COLUMNS
    assert list(built["missing_evidence_update"].columns) == MISSING_UPDATE_COLUMNS
    assert built["validation"]["status"].eq("pass").all()
    assert final_decision(built["validation"]) == "finalization_f6d1_complete"


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_real_predictions_are_complete_candidate_only_and_valid() -> None:
    built = frames()
    predictions = built["predictions"]
    assert len(predictions) == 3 * 1859 * 20
    assert predictions.groupby("model")["user_id"].nunique().eq(1859).all()
    assert predictions.groupby(["model", "user_id"]).size().eq(20).all()
    assert predictions.groupby(["model", "user_id"])["candidate_item_id"].nunique().eq(20).all()
    assert built["instance_summary"]["strict_valid"].eq("true").all()
    assert built["prediction_validation"]["status"].eq("pass").all()


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_reranking_source_policy_missing_evidence_and_coverage() -> None:
    built = frames()
    assert built["reranking_source_audit"]["can_generate_reranking_candidates"].eq("false").all()
    missing = built["missing_evidence_update"]
    assert "was_active_gap" in missing.columns
    assert missing["partially_resolved_by_f6d1"].eq("true").sum() >= 2
    coverage = built["coverage_impact"].set_index("coverage_family")
    assert coverage.loc["protocol-exact numeric replication", "f6d1_value"] == "0/187"
    assert "48/187" in coverage.loc["local numeric mapping coverage", "f6d1_value"]


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_protocol_bm25_not_exact_and_absence_of_absolute_paths() -> None:
    built = frames()
    protocol = build_baseline_protocol().set_index("model")
    assert protocol.loc["bm25_like_adapted", "reproduction_class"] == "bm25_like_adapted"
    assert protocol.loc["bm25_like_adapted", "blocking_reason"]
    assert built["paper_alignment"]["is_exact_numeric_replication"].eq("false").all()
    assert all(no_absolute_paths(frame) for frame in built.values())


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_idempotent_tmp_write_protected_inputs_and_report(tmp_path: Path) -> None:
    built = frames()
    report = build_report(built)
    first_hash = {key: frame_sha256(value) for key, value in built.items()}
    second_hash = {key: frame_sha256(value.copy()) for key, value in built.items()}
    assert first_hash == second_hash
    write_outputs(tmp_path, built, report)
    first_files = {path: file_sha256(tmp_path / path) for path in [OUTPUT_TABLES["validation"], OUTPUT_TABLES["utility_metrics"]]}
    write_outputs(tmp_path, built, report)
    second_files = {path: file_sha256(tmp_path / path) for path in [OUTPUT_TABLES["validation"], OUTPUT_TABLES["utility_metrics"]]}
    assert first_files == second_files
    assert protected_hash_status(PROJECT_ROOT) == "checked=1144;changed=0"


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_build_all_materializes_only_f6d1_paths_in_tmp(tmp_path: Path) -> None:
    built = frames()
    report = build_report(built)
    hashes = write_outputs(tmp_path, built, report)
    for path in hashes:
        assert "finalization_f6d1_" in path
    assert "finalization_f6d1_lastfm_reranking_candidates.csv" not in hashes
