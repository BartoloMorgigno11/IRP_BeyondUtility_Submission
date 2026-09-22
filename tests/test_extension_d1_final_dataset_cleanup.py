from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.extension import e0_candidates, e0_protocol


ROOT = Path(__file__).resolve().parents[1]
D1_MANIFEST = ROOT / "results/manifests/d1_video_games_purge_execution.json"
D1_REFERENCE_AUDIT = ROOT / "results/tables/d1_remaining_video_games_reference_audit.csv"
D1_REMOVED_ARTIFACTS = ROOT / "results/tables/d1_removed_artifacts.csv"
D1_MANUAL_REVIEW = ROOT / "results/tables/d1_manual_review_resolution.csv"
D1_ACTIVE_DATASETS = ROOT / "results/tables/d1_active_dataset_audit.csv"

EXPECTED_ACTIVE_DATASETS = {"beauty", "sports", "ml1m", "lastfm", "foodcom"}
EXPECTED_ACTIVE_LABELS = {"Beauty", "Sports", "ML-1M", "LastFM", "Food.com"}
EXPECTED_FOODCOM_HASHES = {
    "archive": "e7a43e623c616799f78a0479b550483450fd7ede4e0c40979db3458a82b5b913",
    "RAW_interactions.csv": "4e278d2f687b1020f5694e51003a7d6174344b51aebbdf4530de88b5c6a78ee4",
    "RAW_recipes.csv": "6a8136d1da9e03396a9f52d72200b3fddb8fe103f11e1496effca30ac4e0539f",
}


def _manifest() -> dict:
    with D1_MANIFEST.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_active_dataset_set_is_foodcom_only_as_fifth_dataset() -> None:
    manifest = _manifest()
    audit = pd.read_csv(D1_ACTIVE_DATASETS)
    expected_rows = audit.loc[audit["expected_active"].astype(str).str.lower().eq("true")]

    assert set(expected_rows["dataset_id"]) == EXPECTED_ACTIVE_DATASETS
    assert set(expected_rows["dataset"]) == EXPECTED_ACTIVE_LABELS
    assert not audit.loc[audit["status"].eq("UNEXPECTED")].any(axis=None)
    assert manifest["active_dataset_audit"]["dataset_ids"] == ["beauty", "sports", "ml1m", "lastfm", "foodcom"]


def test_active_universe_has_no_discarded_dataset_rows() -> None:
    raw = pd.read_csv(ROOT / "results/tables/extension_raw_experiment_universe.csv")
    cells = pd.read_csv(ROOT / "results/tables/extension_cell_universe.csv")

    for frame in (raw, cells):
        lowered = frame.astype(str).apply(lambda column: column.str.lower())
        assert not lowered.apply(lambda column: column.str.contains("video_games|video games|videogames", regex=True)).any().any()
        assert set(frame["dataset"]) == EXPECTED_ACTIVE_DATASETS


def test_no_active_discarded_dataset_configs_runners_or_namespaces() -> None:
    absent_paths = [
        "configs/video_games.yaml",
        "configs/extension_e1_pilot.yaml",
        "src/data/prepare_video_games.py",
        "src/experiments/run_video_games_gemma2.py",
        "src/extension/e0_dry_run.py",
        "src/extension/e1_pilot.py",
        "src/extension/e2b0_video_games_2014_feasibility.py",
        "src/extension/e2b0h_video_games_history_title_blockers.py",
        "src/extension/e2b0j_joint_title_recovery_frontier.py",
        "tests/test_extension_e1_pilot.py",
        "tests/test_extension_e2b0_video_games_2014_feasibility.py",
        "tests/test_extension_e2b0h_video_games_history_title_blockers.py",
        "tests/test_extension_e2b0j_joint_title_recovery_frontier.py",
        "data/raw/video_games_2014",
        "data/processed/video_games",
        "Espansione_progetto/video_games_tutti_esperimenti",
    ]
    for rel_path in absent_paths:
        assert not (ROOT / rel_path).exists(), rel_path


def test_removed_artifacts_table_paths_are_absent() -> None:
    removed = pd.read_csv(D1_REMOVED_ARTIFACTS)
    assert not removed.empty
    assert removed["removed"].astype(str).str.lower().eq("true").all()
    assert removed["exists_after"].astype(str).str.lower().eq("false").all()
    for rel_path in removed["path"]:
        assert not (ROOT / str(rel_path)).exists(), rel_path


def test_manual_review_has_no_unresolved_entries() -> None:
    manual = pd.read_csv(D1_MANUAL_REVIEW)
    assert len(manual) == 4
    assert set(manual["original_classification"]) == {"REVIEW_MANUALLY"}
    assert set(manual["final_classification"]) == {"REMOVE_DATASET_SPECIFIC"}
    assert manual["safe_to_remove"].astype(str).str.lower().eq("true").all()


def test_remaining_reference_audit_has_no_forbidden_operational_occurrences() -> None:
    manifest = _manifest()
    refs = pd.read_csv(D1_REFERENCE_AUDIT)

    assert manifest["remaining_video_games_references"]["operational_forbidden_occurrences"] == 0
    assert refs["allowed"].astype(str).str.lower().eq("true").all()
    assert set(refs["classification"]).issubset(
        {
            "D1_CLEANUP_EVIDENCE",
            "FINAL_SELECTION_RATIONALE",
            "PROTECTED_NON_DATASET_CONTENT_TOKEN",
            "RETAIN_H0_PROTECTED_EVIDENCE",
            "RETAIN_HISTORICAL_LLM_OUTPUT",
            "RETAIN_HISTORICAL_REPRODUCTION",
        }
    )


def test_foodcom_hashes_are_unchanged_from_authoritative_values() -> None:
    manifest = _manifest()
    assert manifest["protected_hashes"]["foodcom_unchanged"] is True

    archive = ROOT / "data/raw/foodcom/source/food-com-recipes-and-user-interactions.zip"
    extracted = ROOT / "results/manifests/d0a_foodcom_extracted_file_hashes.json"
    with extracted.open("r", encoding="utf-8") as handle:
        extracted_manifest = json.load(handle)

    assert _sha256(archive) == EXPECTED_FOODCOM_HASHES["archive"]
    for filename in ("RAW_interactions.csv", "RAW_recipes.csv"):
        raw_path = Path(extracted_manifest["files"][filename]["relative_extracted_path"])
        assert _sha256(raw_path) == EXPECTED_FOODCOM_HASHES[filename]


def test_h0_and_selection_inputs_are_unchanged() -> None:
    manifest = _manifest()
    protected = manifest["protected_hashes"]

    assert protected["h0_unchanged"] is True
    assert protected["d0a_unchanged"] is True
    assert protected["d0b_unchanged"] is True
    assert protected["h0_post"]["results/manifests/extension_h0_no_inference_manifest.json"] == (
        manifest["pre_purge"]["h0_artifact_hashes"]["results/manifests/extension_h0_no_inference_manifest.json"]
    )
    assert protected["h0_post"]["results/tables/extension_h0_72_cell_history_matrix.csv"] == (
        manifest["pre_purge"]["h0_artifact_hashes"]["results/tables/extension_h0_72_cell_history_matrix.csv"]
    )


def test_generic_e0_infrastructure_and_historical_reproduction_are_retained() -> None:
    retained_paths = [
        "src/extension/e0_protocol.py",
        "src/extension/e0_candidates.py",
        "src/extension/e0_history.py",
        "src/extension/e0_prompts.py",
        "src/extension/e0_parser.py",
        "tests/test_extension_h0_history_final_parity_audit.py",
        "original_repo/EvaLLMasRecommender",
    ]
    for rel_path in retained_paths:
        assert (ROOT / rel_path).exists(), rel_path

    assert e0_protocol.PRIMARY_DATASETS == ("beauty", "sports", "foodcom")
    assert e0_protocol.ACTIVE_EXTENSION_DATASETS == ("beauty", "sports", "ml1m", "lastfm", "foodcom")
    assert hasattr(e0_candidates, "build_primary_candidates")
    assert not hasattr(e0_candidates, "build_video_games_primary_candidates")


def test_no_inference_training_download_or_network_was_executed() -> None:
    manifest = _manifest()
    assert manifest["llm_requests_executed"] == 0
    assert manifest["training_runs_executed"] == 0
    assert manifest["downloads_executed"] == 0
    assert manifest["network_calls_executed"] == 0
    assert manifest["raw_reconstruction_executed"] == 0
