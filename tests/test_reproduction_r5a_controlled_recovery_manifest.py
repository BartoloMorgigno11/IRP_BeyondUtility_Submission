from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.reproduction_r5a_controlled_recovery_manifest import (
    DECISION_COMPLETE,
    P0_EXPECTED,
    P1_EXPECTED,
    build_r5a,
)
from src.experiments.run_reproduction_r5_recovery import dry_run_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "reproduction_r5a_controlled_recovery_manifest.yaml"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"


@pytest.fixture(scope="session", autouse=True)
def r5a_outputs() -> dict[str, pd.DataFrame]:
    result = build_r5a(PROJECT_ROOT, config_path=CONFIG_PATH, write=True)
    assert result["decision"] == DECISION_COMPLETE
    return result["frames"]


def test_validation_passes(r5a_outputs: dict[str, pd.DataFrame]) -> None:
    validation = r5a_outputs["validation"]
    assert validation["status"].eq("pass").all()
    assert len(validation) == 23


def test_exact_counts_supersede_r4_estimate(r5a_outputs: dict[str, pd.DataFrame]) -> None:
    counts = r5a_outputs["exact_request_counts"].set_index("family")
    assert int(counts.loc["CandDif", "raw_planned_requests"]) == 2400
    assert int(counts.loc["CandDif", "duplicate_requests"]) == 1200
    assert int(counts.loc["CandDif", "unique_new_requests"]) == 1200
    assert int(counts.loc["History", "raw_planned_requests"]) == 2400
    assert int(counts.loc["History", "unique_new_requests"]) == 2400
    assert int(counts.loc["Generation", "new_profile_generation_requests"]) == 400
    assert int(counts.loc["Generation", "downstream_recommendation_requests"]) == 800
    assert int(counts.loc["Generation", "unique_new_requests"]) == 1200
    assert int(counts.loc["TOTAL", "unique_new_requests"]) == 4800
    assert int(counts.loc["TOTAL", "unique_new_requests"]) != 4400


def test_gap_scope_is_p0_p1_only(r5a_outputs: dict[str, pd.DataFrame]) -> None:
    scope = r5a_outputs["gap_scope"]
    assert len(scope) == 54
    p0 = set(map(tuple, scope.loc[scope["priority"].eq("P0_MUST_CLOSE"), ["dataset", "metric"]].to_numpy()))
    p1 = set(map(tuple, scope.loc[scope["priority"].eq("P1_HIGH_VALUE"), ["dataset", "metric"]].to_numpy()))
    assert p0 == P0_EXPECTED
    assert p1 == P1_EXPECTED
    lower = scope.loc[~scope["priority"].isin(["P0_MUST_CLOSE", "P1_HIGH_VALUE"])]
    assert lower["unique_request_count"].astype(int).sum() == 0
    assert not scope.loc[
        scope["dataset"].eq("ML-1M") & scope["metric"].str.startswith("CandDif")
    ]["scheduled_in_manifest"].eq("true").any()


def test_canddif_metric_duplicates_are_deduplicated(r5a_outputs: dict[str, pd.DataFrame]) -> None:
    raw = r5a_outputs["raw_planned_requests"]
    canddif = raw.loc[raw["recovery_family"].eq("CANDDIF_RECOVERY")]
    assert len(canddif) == 2400
    assert canddif["canonical_request_id"].nunique() == 1200
    by_request = canddif.groupby("canonical_request_id")["metric"].nunique()
    assert by_request.eq(2).all()
    assert set(canddif["duplicate_type"]) == {"NONE", "METRIC_ONLY_DUPLICATE"}


def test_generation_profile_dependencies_are_explicit(r5a_outputs: dict[str, pd.DataFrame]) -> None:
    manifest = r5a_outputs["recovery_request_manifest"]
    generation = manifest.loc[manifest["recovery_family"].eq("GENERATION_INVOLVED_RECOVERY")]
    profile_generation = generation.loc[generation["task"].eq("profile_generation")]
    downstream = generation.loc[generation["task"].eq("ranking")]
    assert len(profile_generation) == 400
    assert len(downstream) == 800
    assert set(profile_generation["dataset"]) == {"ML-1M", "LastFM"}
    assert downstream.loc[downstream["dataset"].isin(["Beauty", "Sports"])]["profile_reuse_status"].eq(
        "EXISTING_PROFILE_DEPENDENCY_RECORDED"
    ).all()
    assert downstream.loc[downstream["dataset"].isin(["ML-1M", "LastFM"])]["profile_reuse_status"].eq(
        "PENDING_PROFILE_GENERATION_REQUEST_RECORDED"
    ).all()
    assert downstream["profile_dependency_request_id"].str.len().gt(0).all()


def test_manifest_freezes_material_hashes_and_never_executes(r5a_outputs: dict[str, pd.DataFrame]) -> None:
    manifest = r5a_outputs["recovery_request_manifest"]
    assert len(manifest) == 4800
    assert manifest["request_id"].is_unique
    assert manifest["request_key_sha256"].is_unique
    assert manifest["execution_allowed_in_r5a"].eq("false").all()
    assert manifest["inference_decision"].eq("FUTURE_APPROVAL_REQUIRED_ZERO_R5A_INFERENCE").all()
    for column in [
        "candidate_list_hash",
        "unordered_candidate_item_set_hash",
        "history_hash",
        "prompt_hash",
        "material_context_hash",
        "output_destination",
    ]:
        assert manifest[column].astype(str).str.len().gt(0).all()
    assert manifest["requires_new_llm_inference"].eq("true").all()


def test_canddif_first_random_pairs_share_only_unordered_candidate_set(
    r5a_outputs: dict[str, pd.DataFrame],
) -> None:
    canddif = r5a_outputs["recovery_request_manifest"].loc[
        r5a_outputs["recovery_request_manifest"]["recovery_family"].eq("CANDDIF_RECOVERY")
    ]
    assert len(canddif) == 1200
    grouped = canddif.groupby(["dataset", "model_name", "user_id"], sort=False)
    assert len(grouped) == 600
    for _, group in grouped:
        assert set(group["experimental_condition"]) == {"CANDDIF_FIRST", "CANDDIF_RANDOM"}
        first = group.loc[group["experimental_condition"].eq("CANDDIF_FIRST")].iloc[0]
        random = group.loc[group["experimental_condition"].eq("CANDDIF_RANDOM")].iloc[0]
        assert first["history_hash"] == random["history_hash"]
        assert first["unordered_candidate_item_set_hash"] == random["unordered_candidate_item_set_hash"]
        assert first["candidate_list_hash"] != random["candidate_list_hash"]
        assert first["prompt_template_id"] == random["prompt_template_id"]
        assert first["parser_contract"] == random["parser_contract"]
        assert random["candidate_order_seed"] == "R5A_CANDDIF_RANDOM_POSITION_V1"
        assert random["candidate_order_algorithm"] == "sha256_seeded_sort_rotate_if_positive_first"


def test_profile_hashes_are_not_fabricated_for_future_profile_outputs(
    r5a_outputs: dict[str, pd.DataFrame],
) -> None:
    manifest = r5a_outputs["recovery_request_manifest"]
    profile_generation = manifest.loc[manifest["task"].eq("profile_generation")]
    assert profile_generation["profile_hash"].eq("").all()
    assert profile_generation["profile_hash_status"].eq("PROFILE_OUTPUT_NOT_YET_GENERATED").all()

    downstream = manifest.loc[
        manifest["recovery_family"].eq("GENERATION_INVOLVED_RECOVERY") & manifest["task"].eq("ranking")
    ]
    pending = downstream.loc[downstream["dataset"].isin(["ML-1M", "LastFM"])]
    existing = downstream.loc[downstream["dataset"].isin(["Beauty", "Sports"])]
    assert pending["profile_hash"].eq("").all()
    assert pending["profile_hash_status"].eq("PROFILE_DEPENDENCY_REQUEST_ID_PENDING_OUTPUT").all()
    assert pending["profile_dependency_request_id"].str.startswith("R5A-REQ-").all()
    assert existing["profile_hash"].str.len().eq(64).all()
    assert existing["profile_hash_status"].eq("INPUT_PROFILE_HASH_AVAILABLE").all()

    generated_ids = set(profile_generation["request_id"])
    assert set(pending["profile_dependency_request_id"]).issubset(generated_ids)


def test_existing_output_reuse_is_checked_without_exact_reuse(r5a_outputs: dict[str, pd.DataFrame]) -> None:
    reuse = r5a_outputs["existing_output_reuse"]
    manifest = r5a_outputs["recovery_request_manifest"]
    assert len(reuse) == len(manifest)
    assert not reuse["reuse_status"].eq("EXACT_REUSE").any()
    assert reuse["reusable_existing_output"].eq("false").all()
    assert reuse["checked_sources"].str.contains("phase_f6l_profile_request_manifest").all()


def test_protected_artifacts_remain_unchanged(r5a_outputs: dict[str, pd.DataFrame]) -> None:
    protected = r5a_outputs["protected_hash_check"]
    assert len(protected) == 12
    assert protected["protection_status"].eq("PROTECTED_UNCHANGED").all()
    assert protected["exists"].eq("true").all()


def test_dry_run_runner_validates_manifest(r5a_outputs: dict[str, pd.DataFrame]) -> None:
    summary = dry_run_manifest(TABLE_DIR / "reproduction_r5a_recovery_request_manifest.csv")
    assert summary == {
        "manifest_rows": 4800,
        "profile_generation_rows": 400,
        "ranking_rows": 4400,
    }
