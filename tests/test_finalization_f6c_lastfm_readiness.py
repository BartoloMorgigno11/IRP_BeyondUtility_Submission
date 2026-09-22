from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.finalization_f6c_lastfm_readiness import (
    ANOMALY_COLUMNS,
    CANDIDATE_OUTPUT_COLUMNS,
    CANDIDATE_POLICY_COLUMNS,
    CATALOG_OUTPUT_COLUMNS,
    COHORT_COLUMNS,
    COHORT_OUTPUT_COLUMNS,
    COVERAGE_IMPACT_COLUMNS,
    DERIVED_PATHS,
    EXECUTION_PLAN_COLUMNS,
    EXISTING_RESULT_COLUMNS,
    INTERACTION_AUDIT_COLUMNS,
    INTERACTION_OUTPUT_COLUMNS,
    ITEM_IDENTITY_COLUMNS,
    METADATA_COLUMNS,
    MISSING_UPDATE_COLUMNS,
    OUTPUT_TABLES,
    PREPROCESSING_COLUMNS,
    PROTOCOL_COLUMNS,
    READINESS_COLUMNS,
    REPORT_PATH,
    RERANKING_COLUMNS,
    SOURCE_COLUMNS,
    SPLIT_POLICY_COLUMNS,
    SPLIT_VALIDATION_COLUMNS,
    VALIDATION_COLUMNS,
    build_adapted_split,
    build_catalog,
    build_frames,
    build_report,
    deduplicate_interactions,
    final_decision,
    frame_sha256,
    iterative_k_core,
    no_absolute_paths,
    normalize_name,
    output_hashes,
    protected_hash_status,
    readiness_statuses,
    sample_ranking_candidates,
    write_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def frames() -> dict[str, pd.DataFrame]:
    return build_frames(PROJECT_ROOT)


def synthetic_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interactions = pd.DataFrame(
        [
            {"user_id": user, "item_id": item, "rating": rating, "timestamp": ""}
            for user, items in {
                "u1": [("i1", 1), ("i2", 2), ("i3", 3), ("i4", 4), ("i5", 5)],
                "u2": [("i1", 1), ("i2", 2), ("i3", 3), ("i4", 4), ("i5", 5)],
                "u3": [("i1", 1), ("i2", 2), ("i3", 3), ("i4", 4), ("i5", 5)],
                "u4": [("i1", 1), ("i2", 2), ("i3", 3), ("i4", 4), ("i5", 5)],
                "u5": [("i1", 1), ("i2", 2), ("i3", 3), ("i4", 4), ("i5", 5)],
            }.items()
            for item, rating in items
        ]
    )
    artists = pd.DataFrame({"item_id": [f"i{i}" for i in range(1, 8)], "title": [f"Artist {i}" for i in range(1, 8)]})
    raw_artists = artists.rename(columns={"item_id": "id", "title": "name"})
    tags = pd.DataFrame({"tagID": ["t1"], "tagValue": ["rock"]})
    tagged = pd.DataFrame({"userID": ["u1", "u2"], "artistID": ["i1", "i2"], "tagID": ["t1", "t1"]})
    return interactions, raw_artists, tags, tagged


def test_synthetic_dedup_kcore_and_name_normalization() -> None:
    interactions, _, _, _ = synthetic_raw()
    duplicated = pd.concat([interactions, interactions.iloc[[0]]], ignore_index=True)
    assert len(deduplicate_interactions(duplicated)) == len(interactions)
    core, history = iterative_k_core(interactions, 5, 5)
    assert len(core) == 25
    assert history[-1]["after_users"] == "5"
    assert normalize_name("  The   Cure  ") == "the cure"


def test_synthetic_split_candidates_and_metadata() -> None:
    interactions, raw_artists, tags, tagged = synthetic_raw()
    catalog = build_catalog(raw_artists, tagged, tags, set(interactions["item_id"]))
    split = build_adapted_split(interactions, catalog)
    assert split["train"].groupby("user_id").size().eq(3).all()
    assert split["validation"].groupby("user_id").size().eq(1).all()
    assert split["test"].groupby("user_id").size().eq(1).all()
    candidates = sample_ranking_candidates(split, catalog, seed=7, candidate_count=3)
    assert candidates.groupby("candidate_set_id").size().eq(3).all()
    assert candidates.groupby("candidate_set_id")["is_ground_truth"].apply(lambda s: s.eq("true").sum()).eq(1).all()
    assert set(candidates.columns) == set(CANDIDATE_OUTPUT_COLUMNS)


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_f6c_output_schemas() -> None:
    built = frames()
    assert list(built["source_inventory"].columns) == SOURCE_COLUMNS
    assert list(built["paper_protocol"].columns) == PROTOCOL_COLUMNS
    assert list(built["interaction_audit"].columns) == INTERACTION_AUDIT_COLUMNS
    assert list(built["item_identity_audit"].columns) == ITEM_IDENTITY_COLUMNS
    assert list(built["metadata_coverage"].columns) == METADATA_COLUMNS
    assert list(built["preprocessing_policy"].columns) == PREPROCESSING_COLUMNS
    assert list(built["split_policy"].columns) == SPLIT_POLICY_COLUMNS
    assert list(built["split_validation"].columns) == SPLIT_VALIDATION_COLUMNS
    assert list(built["cohort_policy"].columns) == COHORT_COLUMNS
    assert list(built["candidate_policy"].columns) == CANDIDATE_POLICY_COLUMNS
    assert list(built["reranking_readiness"].columns) == RERANKING_COLUMNS
    assert list(built["existing_result_inventory"].columns) == EXISTING_RESULT_COLUMNS
    assert list(built["readiness_matrix"].columns) == READINESS_COLUMNS
    assert list(built["execution_plan"].columns) == EXECUTION_PLAN_COLUMNS
    assert list(built["missing_evidence_update"].columns) == MISSING_UPDATE_COLUMNS
    assert list(built["coverage_impact"].columns) == COVERAGE_IMPACT_COLUMNS
    assert list(built["anomalies"].columns) == ANOMALY_COLUMNS
    assert list(built["validation"].columns) == VALIDATION_COLUMNS
    assert list(built["_derived_interactions"].columns) == INTERACTION_OUTPUT_COLUMNS
    assert list(built["_derived_catalog"].columns) == CATALOG_OUTPUT_COLUMNS
    assert list(built["_derived_ranking_cohort"].columns) == COHORT_OUTPUT_COLUMNS


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_lastfm_source_inventory_and_protocol() -> None:
    built = frames()
    source = built["source_inventory"]
    assert {"F6C-SRC-USER-ARTISTS", "F6C-SRC-ARTISTS", "F6C-SRC-USER-TAGGEDARTISTS-TIMESTAMPS"}.issubset(set(source["source_id"]))
    assert source.loc[source["source_id"].eq("F6C-SRC-USER-ARTISTS"), "interaction_count"].iloc[0] == "92834"
    protocol = built["paper_protocol"].set_index("aspect")
    assert protocol.loc["dataset_origin", "paper_value"].startswith("LastFM")
    assert protocol.loc["timestamp_policy", "equivalence"] == "blocked_exact"
    assert protocol.loc["split_strategy", "reproduction_class" if "reproduction_class" in protocol.columns else "equivalence"] != ""


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_real_interaction_audit_core_and_metadata() -> None:
    built = frames()
    audit = built["interaction_audit"].set_index("audit_stage")
    assert audit.loc["raw", "interaction_count"] == "92834"
    assert audit.loc["final_selected_dataset", "user_count"] == "1859"
    assert audit.loc["final_selected_dataset", "item_count"] == "2823"
    assert audit.loc["final_selected_dataset", "minimum_user_degree"] == "5"
    assert audit.loc["final_selected_dataset", "minimum_item_degree"] == "5"
    metadata = built["metadata_coverage"].set_index("dataset_stage")
    assert metadata.loc["final_selected_dataset", "name_coverage"] == "100.000000"
    assert float(metadata.loc["final_selected_dataset", "tag_coverage"]) > 0.0


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_split_candidate_and_cohort_readiness() -> None:
    built = frames()
    assert built["split_validation"]["status"].eq("pass").all()
    candidates = built["_derived_ranking_candidates"]
    assert candidates["candidate_set_id"].nunique() == 1859
    assert len(candidates) == 1859 * 20
    assert candidates.groupby("candidate_set_id")["candidate_item_id"].nunique().eq(20).all()
    statuses = readiness_statuses(built)
    assert statuses == {"overall": "partially_ready", "ranking": "ready_adapted", "reranking": "blocked"}
    reranking = built["reranking_readiness"]
    assert reranking["can_build_reranking_manifest"].eq("false").all()


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_existing_results_missing_update_and_coverage_impact() -> None:
    built = frames()
    existing = built["existing_result_inventory"]
    assert len(existing) > 0
    assert existing["has_prediction_lists"].eq("false").all()
    missing = built["missing_evidence_update"]
    assert len(missing) == 44
    assert missing["resolved_by_f6c"].eq("true").sum() == 2
    coverage = built["coverage_impact"].set_index("coverage_family")
    assert coverage.loc["dataset evaluation-ready coverage", "f6c_value"].startswith("4/4")
    assert coverage.loc["protocol-exact numeric replication", "f6c_value"] == "0/187"


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_validation_decision_protected_inputs_and_paths() -> None:
    built = frames()
    validation = built["validation"]
    assert validation["status"].eq("pass").all()
    assert final_decision(validation) == "finalization_f6c_complete"
    assert protected_hash_status(PROJECT_ROOT) == "checked=1144;changed=0"
    assert all(no_absolute_paths(frame) for frame in built.values())
    assert all(not frame.isna().any().any() for frame in built.values())


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_determinism_tmp_output_idempotence_and_report(tmp_path: Path) -> None:
    built = frames()
    first = {key: frame_sha256(value) for key, value in built.items()}
    second = {key: frame_sha256(value) for key, value in build_frames(PROJECT_ROOT).items()}
    assert first == second
    report = build_report(PROJECT_ROOT, built)
    write_outputs(tmp_path, built, report)
    first_hashes = output_hashes(tmp_path)
    write_outputs(tmp_path, built, report)
    second_hashes = output_hashes(tmp_path)
    assert first_hashes == second_hashes
    assert set(first_hashes) == {REPORT_PATH.as_posix(), *(path.as_posix() for path in OUTPUT_TABLES.values()), *(path.as_posix() for path in DERIVED_PATHS.values())}
    for index in range(1, 26):
        assert f"# {index}. " in report

