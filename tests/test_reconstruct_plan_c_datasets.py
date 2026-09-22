from __future__ import annotations

import gzip
import inspect
import json
import zipfile
from pathlib import Path

import pandas as pd
import yaml

from src.data.reconstruct_plan_c_datasets import (
    _build_sequential_lines,
    _prepare_candidate_state,
    reconstruct_plan_c_datasets,
)


def write_config(
    path: Path,
    *,
    staging_dir: str = "stage",
    output_dir: str = "reconstructed",
    tables_dir: str = "tables",
    negatives: int = 2,
    item_universe: str = "interaction_items",
    datasets: list[str] | None = None,
) -> None:
    datasets = datasets or ["amazon_beauty", "amazon_sports", "ml1m", "lastfm"]
    config = {
        "dataset": {
            "name": "plan_c_reconstruction",
            "raw_dir": staging_dir,
            "processed_dir": output_dir,
            "sample_dir": "samples",
            "interactions_path": None,
            "catalog_path": None,
            "candidates_path": None,
            "ground_truth_path": None,
            "llm_outputs_path": None,
            "baseline_outputs_path": None,
        },
        "paths": {
            "raw_dir": staging_dir,
            "processed_dir": output_dir,
            "sample_dir": "samples",
            "interactions_path": None,
            "catalog_path": None,
            "candidates_path": None,
            "ground_truth_path": None,
            "llm_outputs_path": None,
            "baseline_outputs_path": None,
        },
        "original_repo": {"root": "original_repo/EvaLLMasRecommender"},
        "experiment": {
            "task": "ranking",
            "k_values": [5],
            "seed": 7,
            "sample_size": None,
            "allow_api_calls": False,
        },
        "negative_sampling": {"item_universe": item_universe},
        "plan_c_reconstruction": {
            "input_staging_dir": staging_dir,
            "output_base_dir": output_dir,
            "seed": 7,
            "negatives_per_user": negatives,
            "min_interactions_per_user": 2,
            "datasets_enabled": datasets,
            "datasets": {
                "amazon_beauty": {
                    "reviews_file": "reviews_Beauty_5.json.gz",
                    "metadata_file": "meta_Beauty.json.gz",
                    "split_type": "chronological_leave_one_out",
                },
                "amazon_sports": {
                    "reviews_file": "reviews_Sports_and_Outdoors_5.json.gz",
                    "metadata_file": "meta_Sports_and_Outdoors.json.gz",
                    "split_type": "chronological_leave_one_out",
                },
                "ml1m": {
                    "archive_file": "ml-1m.zip",
                    "ratings_member": "ml-1m/ratings.dat",
                    "movies_member": "ml-1m/movies.dat",
                    "users_member": "ml-1m/users.dat",
                    "split_type": "chronological_leave_one_out",
                },
                "lastfm": {
                    "archive_file": "hetrec2011-lastfm-2k.zip",
                    "user_artists_member": "user_artists.dat",
                    "artists_member": "artists.dat",
                    "split_type": "interactions_catalog_only_no_chronological_timestamp",
                },
            },
        },
        "api": {"allow_api_calls": False, "provider": None, "model": None, "api_key_env_var": None},
        "models": {"llm_outputs": [], "baselines": [], "baseline_outputs": []},
        "metrics": {"utility": [], "beyond_utility": [], "matching_protocols": []},
        "output": {
            "metrics_dir": "results/metrics",
            "tables_dir": tables_dir,
            "plots_dir": "results/plots",
            "raw_outputs_dir": "results/raw_outputs",
            "dataset_audit_path": "tables/audit.csv",
            "metrics_path": "results/metrics/not_written.csv",
            "summary_path": "tables/summary.csv",
            "analysis_path": "report/results_analysis.md",
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def write_json_gz(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_amazon_fixture(staging: Path, *, prefix: str = "Beauty") -> None:
    review_name = "reviews_Beauty_5.json.gz" if prefix == "Beauty" else "reviews_Sports_and_Outdoors_5.json.gz"
    meta_name = "meta_Beauty.json.gz" if prefix == "Beauty" else "meta_Sports_and_Outdoors.json.gz"
    reviews = [
        {"reviewerID": "u1", "asin": "i1", "overall": 5.0, "unixReviewTime": 100},
        {"reviewerID": "u1", "asin": "i2", "overall": 4.0, "unixReviewTime": 200},
        {"reviewerID": "u1", "asin": "i3", "overall": 3.0, "unixReviewTime": 300},
        {"reviewerID": "u2", "asin": "i4", "overall": 5.0, "unixReviewTime": 400},
        {"reviewerID": "u3", "asin": "i5", "overall": 4.0, "unixReviewTime": 100},
        {"reviewerID": "u3", "asin": "i6", "overall": 5.0, "unixReviewTime": 200},
    ]
    metadata = [
        {"asin": f"i{i}", "title": f"Item {i}", "categories": [[prefix]], "description": f"Desc {i}"}
        for i in range(1, 10)
    ]
    write_json_gz(staging / review_name, reviews)
    write_json_gz(staging / meta_name, metadata)


def test_amazon_reconstruction_chronological_split_and_candidates(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    write_amazon_fixture(staging)
    config = tmp_path / "config.yaml"
    write_config(config, datasets=["amazon_beauty"])

    frames = reconstruct_plan_c_datasets(config, project_root=tmp_path)

    summary = frames["summary"].set_index("dataset").loc["amazon_beauty"]
    assert summary["status"] == "completed"
    assert bool(summary["diagnostic_subset"]) is False
    assert summary["train_interactions"] == 3
    assert summary["ground_truth_rows"] == 2
    assert summary["candidate_rows"] == 6
    assert summary["reconstruction_status"] == "reconstructed_from_public_raw"

    output = tmp_path / "reconstructed" / "amazon_beauty"
    interactions = pd.read_csv(output / "interactions.csv", dtype=str)
    ground_truth = pd.read_csv(output / "ground_truth.csv", dtype=str)
    candidates = pd.read_csv(output / "candidates.csv", dtype=str)

    assert set(interactions["user_id"]) == {"u1", "u3"}
    assert "u2" not in set(interactions["user_id"])
    assert ground_truth.set_index("user_id").loc["u1", "ground_truth_item_id"] == "i3"
    assert ground_truth.set_index("user_id").loc["u3", "ground_truth_item_id"] == "i6"
    for user_id, group in candidates.groupby("user_id"):
        assert len(group) == 3
        assert group["is_ground_truth"].astype(str).str.lower().eq("true").sum() == 1
        ranks = sorted(group["candidate_position"].astype(int).tolist())
        assert ranks == [1, 2, 3]
    full_history = {"u1": {"i1", "i2", "i3"}, "u3": {"i5", "i6"}}
    for _, row in candidates.iterrows():
        if str(row["is_ground_truth"]).lower() == "false":
            assert row["candidate_item_id"] not in full_history[row["user_id"]]
            assert row["candidate_item_id"] not in {"i7", "i8", "i9"}
    assert summary["candidate_item_universe"] == "interaction_items"
    assert summary["candidate_universe_size"] == 6
    assert summary["catalog_size"] == 9
    assert summary["interaction_item_count"] == 6
    assert summary["users_with_insufficient_negatives"] == 0


def test_catalog_only_items_require_explicit_catalog_item_universe(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    reviews = [
        {"reviewerID": "u1", "asin": "i1", "overall": 5.0, "unixReviewTime": 100},
        {"reviewerID": "u1", "asin": "i2", "overall": 4.0, "unixReviewTime": 200},
    ]
    metadata = [
        {"asin": f"i{i}", "title": f"Item {i}", "categories": [["Beauty"]], "description": f"Desc {i}"}
        for i in range(1, 6)
    ]
    write_json_gz(staging / "reviews_Beauty_5.json.gz", reviews)
    write_json_gz(staging / "meta_Beauty.json.gz", metadata)

    blocked_config = tmp_path / "blocked.yaml"
    write_config(blocked_config, datasets=["amazon_beauty"], negatives=2)
    blocked = reconstruct_plan_c_datasets(blocked_config, project_root=tmp_path)
    blocked_summary = blocked["summary"].set_index("dataset").loc["amazon_beauty"]
    assert blocked_summary["status"] == "blocked_insufficient_negative_pool"
    assert blocked_summary["candidate_item_universe"] == ""
    assert not (tmp_path / "reconstructed" / "amazon_beauty" / "candidates.csv").exists()

    catalog_config = tmp_path / "catalog.yaml"
    write_config(
        catalog_config,
        output_dir="reconstructed_catalog",
        tables_dir="tables_catalog",
        datasets=["amazon_beauty"],
        negatives=2,
        item_universe="catalog_items",
    )
    frames = reconstruct_plan_c_datasets(catalog_config, project_root=tmp_path)
    summary = frames["summary"].set_index("dataset").loc["amazon_beauty"]
    candidates = pd.read_csv(
        tmp_path / "reconstructed_catalog" / "amazon_beauty" / "candidates.csv",
        dtype=str,
    )
    negatives = candidates.loc[
        candidates["is_ground_truth"].astype(str).str.lower() == "false",
        "candidate_item_id",
    ].tolist()
    assert summary["status"] == "completed"
    assert summary["candidate_item_universe"] == "catalog_items"
    assert summary["candidate_universe_size"] == 5
    assert summary["interaction_item_count"] == 2
    assert set(negatives).issubset({"i3", "i4", "i5"})


def test_negative_sampling_is_deterministic(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    write_amazon_fixture(staging)
    config = tmp_path / "config.yaml"
    write_config(config, output_dir="reconstructed_a", tables_dir="tables_a", datasets=["amazon_beauty"])

    reconstruct_plan_c_datasets(config, project_root=tmp_path)
    first = (tmp_path / "reconstructed_a" / "amazon_beauty" / "candidates.csv").read_text(encoding="utf-8")
    reconstruct_plan_c_datasets(config, project_root=tmp_path)
    second = (tmp_path / "reconstructed_a" / "amazon_beauty" / "candidates.csv").read_text(encoding="utf-8")

    assert first == second


def test_build_sequential_lines_uses_grouped_logic() -> None:
    eligible = pd.DataFrame(
        [
            {"user_id": "u2", "item_id": "i3", "timestamp": 30, "__input_order": 0},
            {"user_id": "u1", "item_id": "i2", "timestamp": 20, "__input_order": 1},
            {"user_id": "u1", "item_id": "i1", "timestamp": 10, "__input_order": 2},
            {"user_id": "u2", "item_id": "i4", "timestamp": 30, "__input_order": 3},
        ]
    )
    user2internal = {"u1": "1", "u2": "2"}
    item2internal = {"i1": "1", "i2": "2", "i3": "3", "i4": "4"}

    lines = _build_sequential_lines(eligible, user2internal, item2internal)

    assert lines == ["1 1 2", "2 3 4"]
    source = inspect.getsource(_build_sequential_lines)
    assert 'eligible.loc[eligible["user_id"]' not in source
    assert ".astype(str) == str(user_id)" not in source


def test_candidate_generation_avoids_full_negative_pool_scan() -> None:
    source = inspect.getsource(_prepare_candidate_state)
    assert "negative_pool = [item for item in all_items" not in source


def test_movielens_reconstruction_writes_plan_c_outputs(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    staging.mkdir()
    with zipfile.ZipFile(staging / "ml-1m.zip", "w") as archive:
        archive.writestr("ml-1m/ratings.dat", "1::10::5::100\n1::11::4::200\n2::12::3::100\n2::13::5::300\n")
        archive.writestr(
            "ml-1m/movies.dat",
            "\n".join(f"{item}::Movie {item}::Drama" for item in range(10, 18)) + "\n",
        )
        archive.writestr("ml-1m/users.dat", "1::M::25::1::00000\n")
    config = tmp_path / "config.yaml"
    write_config(config, datasets=["ml1m"])

    frames = reconstruct_plan_c_datasets(config, project_root=tmp_path)

    summary = frames["summary"].set_index("dataset").loc["ml1m"]
    assert list(frames["summary"]["dataset"]) == ["ml1m"]
    assert summary["status"] == "completed"
    assert summary["output_dir"] == "reconstructed/ml1m"
    output = tmp_path / "reconstructed" / "ml1m"
    assert (output / "interactions.csv").exists()
    assert (output / "ground_truth.csv").exists()
    assert (output / "negative_samples.txt").exists()


def test_lastfm_is_extension_only_without_chronological_split(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    staging.mkdir()
    with zipfile.ZipFile(staging / "hetrec2011-lastfm-2k.zip", "w") as archive:
        archive.writestr("user_artists.dat", "userID\tartistID\tweight\n1\t10\t5\n1\t11\t2\n")
        archive.writestr("artists.dat", "id\tname\turl\tpictureURL\n10\tArtist A\turl\tpic\n11\tArtist B\turl\tpic\n")
    config = tmp_path / "config.yaml"
    write_config(config, datasets=["lastfm"])

    frames = reconstruct_plan_c_datasets(config, project_root=tmp_path)

    summary = frames["summary"].set_index("dataset").loc["lastfm"]
    assert summary["status"] == "partial_extension_only"
    assert summary["reconstruction_status"] == "extension_only"
    output = tmp_path / "reconstructed" / "lastfm"
    assert (output / "interactions.csv").exists()
    assert (output / "catalog.csv").exists()
    assert not (output / "ground_truth.csv").exists()
    blockers = frames["blockers"].set_index("dataset").loc["lastfm"]
    assert blockers["blocker_type"] == "blocked_chronological_split"


def test_missing_raw_files_write_diagnostics_and_no_metrics(tmp_path: Path) -> None:
    (tmp_path / "stage").mkdir()
    config = tmp_path / "config.yaml"
    write_config(config, datasets=["amazon_beauty"])

    frames = reconstruct_plan_c_datasets(config, project_root=tmp_path)

    summary = frames["summary"].set_index("dataset").loc["amazon_beauty"]
    assert summary["status"] == "missing_raw"
    assert summary["artifact_status"] == "not_written"
    assert not (tmp_path / "results" / "metrics").exists()
    assert (tmp_path / "tables" / "plan_c_reconstructed_dataset_summary.csv").exists()


def test_max_users_creates_deterministic_diagnostic_subset(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    write_amazon_fixture(staging)
    config = tmp_path / "config.yaml"
    write_config(config, datasets=["amazon_beauty"])

    frames = reconstruct_plan_c_datasets(
        config,
        project_root=tmp_path,
        max_users=1,
    )

    summary = frames["summary"].set_index("dataset").loc["amazon_beauty"]
    assert summary["status"] == "diagnostic_subset_completed"
    assert bool(summary["diagnostic_subset"]) is True
    assert summary["max_users"] == 1
    assert summary["eligible_users"] == 1
    assert summary["ground_truth_rows"] == 1
    sequential = (tmp_path / "reconstructed" / "amazon_beauty" / "sequential_data.txt").read_text(
        encoding="utf-8"
    )
    assert sequential.splitlines()[0].startswith("1 ")


def test_skip_paper_like_artifacts_keeps_canonical_csvs(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    write_amazon_fixture(staging)
    config = tmp_path / "config.yaml"
    write_config(config, datasets=["amazon_beauty"])

    reconstruct_plan_c_datasets(
        config,
        project_root=tmp_path,
        skip_paper_like_artifacts=True,
    )

    output = tmp_path / "reconstructed" / "amazon_beauty"
    assert (output / "interactions.csv").exists()
    assert (output / "ground_truth.csv").exists()
    assert (output / "candidates.csv").exists()
    assert (output / "catalog.csv").exists()
    assert not (output / "datamaps.json").exists()
    assert not (output / "meta.json").exists()
    assert not (output / "sequential_data.txt").exists()
    assert not (output / "negative_samples.txt").exists()


def test_paper_like_artifacts_are_marked_reconstructed(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    write_amazon_fixture(staging)
    config = tmp_path / "config.yaml"
    write_config(config, datasets=["amazon_beauty"])

    reconstruct_plan_c_datasets(config, project_root=tmp_path)

    output = tmp_path / "reconstructed" / "amazon_beauty"
    datamaps = json.loads((output / "datamaps.json").read_text(encoding="utf-8"))
    assert datamaps["reconstruction_metadata"]["format_status"] == "reconstructed_format"
    assert datamaps["reconstruction_metadata"]["exact_original_artifact_recovered"] == "no"
    assert (output / "sequential_data.txt").read_text(encoding="utf-8").strip()
    assert (output / "negative_samples.txt").read_text(encoding="utf-8").strip()
    meta_payload = json.loads((output / "meta.json").read_text(encoding="utf-8"))
    assert "reconstructed_format" in meta_payload
