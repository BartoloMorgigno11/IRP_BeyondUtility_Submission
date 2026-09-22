from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.extension import d0b_final_fifth_dataset_selection as d0b


TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
SCRIPT_PATH = Path("src/extension/d0b_final_fifth_dataset_selection.py")

FOOD_ARCHIVE_SHA = "e7a43e623c616799f78a0479b550483450fd7ede4e0c40979db3458a82b5b913"
FOOD_INTERACTIONS_SHA = "4e278d2f687b1020f5694e51003a7d6174344b51aebbdf4530de88b5c6a78ee4"
FOOD_RECIPES_SHA = "6a8136d1da9e03396a9f52d72200b3fddb8fe103f11e1496effca30ac4e0539f"
VG_USER_HASH = "90f4f4694d97d50e72ddd6475b85b911d29cae24a556a0799a2a751f78e3a66b"
VG_CANDIDATE_HASH = "c48f48a0353e73c44589965cb48463df4bdd0397355b358d46917d10b812fbad"

REQUIRED_COLUMNS = {
    "requirement_id",
    "evidence_family",
    "experiment_family",
    "metric_or_analysis",
    "mandatory_status",
    "foodcom_status",
    "foodcom_reason",
    "videogames_status",
    "videogames_reason",
    "blocker_scope",
    "preferred_dataset_for_requirement",
    "additional_llm_requests",
    "additional_training",
    "adaptation_class",
    "evidence_source",
}
STATUS_VALUES = {
    "READY",
    "READY_WITH_ADAPTATION",
    "READY_AFTER_TRAINING",
    "BLOCKED_DATASET",
    "BLOCKED_GLOBAL_PROTOCOL",
    "NOT_APPLICABLE",
}


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / name, keep_default_na=False)


def _manifest(name: str) -> dict[str, object]:
    return json.loads((MANIFEST_DIR / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_every_requirement_receives_valid_statuses() -> None:
    matrix = _read("d0b_fifth_dataset_complete_requirement_matrix.csv")

    assert len(matrix) == 72
    assert REQUIRED_COLUMNS.issubset(matrix.columns)
    assert not matrix[list(REQUIRED_COLUMNS)].astype(str).eq("").any().any()
    assert not matrix["requirement_id"].duplicated().any()
    assert set(matrix["foodcom_status"]).issubset(STATUS_VALUES)
    assert set(matrix["videogames_status"]).issubset(STATUS_VALUES)


def test_global_and_dataset_blockers_are_separated() -> None:
    matrix = _read("d0b_fifth_dataset_complete_requirement_matrix.csv")
    blockers = _read("d0b_dataset_specific_blockers.csv")
    mandatory = matrix.loc[~matrix["mandatory_status"].eq("OPTIONAL")]

    global_rows = mandatory.loc[mandatory["blocker_scope"].eq("GLOBAL_PROTOCOL")]
    assert set(global_rows["foodcom_status"]) == {"BLOCKED_GLOBAL_PROTOCOL"}
    assert set(global_rows["videogames_status"]) == {"BLOCKED_GLOBAL_PROTOCOL"}

    assert int(mandatory["foodcom_status"].eq("BLOCKED_DATASET").sum()) == 0
    assert int(mandatory["videogames_status"].eq("BLOCKED_DATASET").sum()) == 25
    assert set(blockers.loc[blockers["dataset_candidate"].eq("Food.com"), "requirement_id"]) == {"NONE"}
    vg_blockers = blockers.loc[blockers["dataset_candidate"].eq("Video Games")]
    assert len(vg_blockers) == 25
    assert vg_blockers["blocker_scope"].eq("DATASET").all()
    assert vg_blockers["experiment_family"].eq("history").sum() == 18


def test_foodcom_d0a_hashes_are_unchanged() -> None:
    manifest = _manifest("d0b_no_inference_manifest.json")
    evidence = _read("d0b_authoritative_input_evidence.csv").set_index(["dataset", "metric"])["value"]

    assert manifest["input_hashes"]["foodcom_archive_sha256"] == FOOD_ARCHIVE_SHA
    assert manifest["input_hashes"]["foodcom_raw_interactions_sha256"] == FOOD_INTERACTIONS_SHA
    assert manifest["input_hashes"]["foodcom_raw_recipes_sha256"] == FOOD_RECIPES_SHA
    assert evidence.loc[("Food.com", "archive_sha256")] == FOOD_ARCHIVE_SHA
    assert evidence.loc[("Food.com", "raw_interactions_sha256")] == FOOD_INTERACTIONS_SHA
    assert evidence.loc[("Food.com", "raw_recipes_sha256")] == FOOD_RECIPES_SHA


def test_video_games_canonical_hashes_are_unchanged() -> None:
    decision = _read("d0b_final_selection_decision.csv").iloc[0]
    evidence = _read("d0b_authoritative_input_evidence.csv").set_index(["dataset", "metric"])["value"]

    assert decision["winner_token"] == d0b.FOODCOM_WINNER
    assert evidence.loc[("Video Games", "primary_user_hash")] == VG_USER_HASH
    assert evidence.loc[("Video Games", "primary_candidate_hash")] == VG_CANDIDATE_HASH


def test_h0_artifacts_are_unchanged_and_referenced_by_hash() -> None:
    h0_manifest = _manifest("extension_h0_no_inference_manifest.json")
    d0b_manifest = _manifest("d0b_no_inference_manifest.json")

    assert int(h0_manifest["final_cells"]) == 72
    assert int(h0_manifest["blocked_cells"]) == 0
    assert int(h0_manifest["to_compute_requests"]) == 14400
    assert d0b_manifest["input_hashes"]["h0_manifest_hash"] == _sha256(MANIFEST_DIR / "extension_h0_no_inference_manifest.json")
    assert d0b_manifest["input_hashes"]["h0_matrix_hash"] == _sha256(TABLE_DIR / "extension_h0_72_cell_history_matrix.csv")


def test_no_inference_training_download_or_purge_actions() -> None:
    manifest = _manifest("d0b_no_inference_manifest.json")
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned_roots = {"anthropic", "ollama", "openai", "requests", "socket", "torch", "tensorflow", "urllib"}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert manifest["llm_requests_executed"] == 0
    assert manifest["training_runs_executed"] == 0
    assert manifest["remote_api_calls_executed"] == 0
    assert manifest["downloads_executed"] == 0
    assert manifest["purge_actions_executed"] == 0
    assert not imported.intersection(banned_roots)


def test_deterministic_winner_logic_selects_foodcom() -> None:
    matrix = _read("d0b_fifth_dataset_complete_requirement_matrix.csv")

    assert d0b.select_winner(matrix) == d0b.FOODCOM_WINNER


def test_purge_inventory_contains_only_classified_paths_and_does_not_purge() -> None:
    inventory = _manifest("d0b_loser_purge_inventory.json")
    allowed = {
        "REMOVE_DATASET_SPECIFIC",
        "RETAIN_GENERIC_INFRASTRUCTURE",
        "RETAIN_HISTORICAL_REPRODUCTION",
        "REVIEW_MANUALLY",
    }
    entries = inventory["entries"]

    assert inventory["winner"] == d0b.FOODCOM_WINNER
    assert inventory["loser"] == "Video Games"
    assert inventory["no_purge_executed"] is True
    assert entries
    assert {entry["classification"] for entry in entries}.issubset(allowed)
    assert any("e2b0" in entry["path"].lower() and entry["classification"] == "REMOVE_DATASET_SPECIFIC" for entry in entries)
    assert not any("foodcom" in entry["path"].lower() and entry["classification"] == "REMOVE_DATASET_SPECIFIC" for entry in entries)


def test_no_silent_fifth_dataset_requirement_omission() -> None:
    matrix = _read("d0b_fifth_dataset_complete_requirement_matrix.csv")
    family_counts = matrix["experiment_family"].value_counts().to_dict()

    assert family_counts["primary"] == 6
    assert family_counts["history"] == 18
    assert family_counts["traditional recommenders"] == 7
    assert family_counts["reranking"] == 4
    assert family_counts["profiles"] == 4
    assert family_counts["CandDif"] == 3
    for metric in ["HR@5", "NDCG@5", "PopREO@5", "DPD@5", "K-S representativeness if globally recoverable"]:
        assert metric in set(matrix["metric_or_analysis"])
    assert matrix["foodcom_status"].astype(str).str.len().gt(0).all()
    assert matrix["videogames_status"].astype(str).str.len().gt(0).all()


def test_future_inference_budget_is_explicit_and_no_training_is_run() -> None:
    budget = _read("d0b_future_inference_budget.csv").set_index("component")

    assert int(budget.loc["original_four_history", "raw_llm_requests"]) == 14400
    assert int(budget.loc["fifth_dataset_primary", "raw_llm_requests"]) == 6000
    assert int(budget.loc["fifth_dataset_history", "raw_llm_requests"]) == 3600
    assert int(budget.loc["gemma_original_four_primary_parity", "raw_llm_requests"]) == 15528
    assert int(budget.loc["required_original_four_gemma_canddif", "raw_llm_requests"]) == 600
    assert int(budget.loc["fifth_dataset_canddif_primary", "raw_llm_requests"]) == 3000
    assert int(budget.loc["profile_generation", "raw_llm_requests"]) == 600
    assert int(budget.loc["profile_only_recommendation", "raw_llm_requests"]) == 600
    assert int(budget.loc["profile_plus_history_recommendation", "raw_llm_requests"]) == 600
    assert int(budget.loc["derived_metrics", "raw_llm_requests"]) == 0
    assert int(budget.loc["grand_total_mandatory_and_conditional", "raw_llm_requests"]) == 44928
    assert budget["training_required"].astype(str).str.lower().eq("false").all()
