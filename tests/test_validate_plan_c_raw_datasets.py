from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path

import pandas as pd

from src.audit.validate_plan_c_raw_datasets import validate_raw_datasets


def write_json_gz(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def make_amazon_reviews() -> list[dict[str, object]]:
    return [
        {"reviewerID": "u1", "asin": "i1", "overall": 5.0, "unixReviewTime": 100},
        {"reviewerID": "u1", "asin": "i2", "overall": 4.0, "unixReviewTime": 200},
        {"reviewerID": "u2", "asin": "i1", "overall": 3.0, "unixReviewTime": 300},
    ]


def make_amazon_meta() -> list[dict[str, object]]:
    return [
        {
            "asin": "i1",
            "title": "Item One",
            "categories": [["Beauty"]],
            "description": "desc",
            "brand": "Brand",
        },
        {"asin": "i2", "title": "Item Two", "categories": [["Beauty"]]},
    ]


def test_amazon_review_schema_detection_and_sampling(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    write_json_gz(staging / "reviews_Beauty_5.json.gz", make_amazon_reviews())

    frames = validate_raw_datasets(
        staging,
        sample_size=2,
        project_root=tmp_path,
        output_dir=tmp_path / "out",
    )

    summary = frames["summary"].set_index(["dataset", "component"])
    row = summary.loc[("amazon_beauty", "amazon_reviews")]
    assert row["status"] == "valid"
    assert row["sampled_rows"] == 2
    assert row["parsed_rows"] == 2
    assert bool(row["required_fields_present"]) is True
    assert bool(row["supports_reconstruction_role"]) is True

    stats = frames["stats"].set_index(["dataset", "component"])
    review_stats = stats.loc[("amazon_beauty", "amazon_reviews")]
    assert review_stats["unique_users_sample"] == 1
    assert review_stats["unique_items_sample"] == 2
    assert review_stats["min_timestamp_sample"] == 100
    assert review_stats["max_timestamp_sample"] == 200


def test_amazon_metadata_schema_detection(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    write_json_gz(staging / "meta_Beauty.json.gz", make_amazon_meta())

    frames = validate_raw_datasets(
        staging,
        sample_size=10,
        project_root=tmp_path,
        output_dir=tmp_path / "out",
    )

    summary = frames["summary"].set_index(["dataset", "component"])
    row = summary.loc[("amazon_beauty", "amazon_metadata")]
    assert row["status"] == "valid"
    assert bool(row["supports_reconstruction_role"]) is True

    schema = frames["schema"]
    title = schema[
        (schema["dataset"] == "amazon_beauty")
        & (schema["component"] == "amazon_metadata")
        & (schema["field"] == "title")
    ].iloc[0]
    assert title["present_in_sample_rows"] == 2
    assert title["status"] == "present_all"


def test_movielens_zip_member_parsing_and_readiness(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    staging.mkdir()
    with zipfile.ZipFile(staging / "ml-1m.zip", "w") as archive:
        archive.writestr("ml-1m/ratings.dat", "1::10::5::100\n2::11::4::200\n")
        archive.writestr("ml-1m/movies.dat", "10::Movie A::Drama\n11::Movie B::Comedy\n")
        archive.writestr("ml-1m/users.dat", "1::M::25::1::00000\n")

    frames = validate_raw_datasets(
        staging,
        sample_size=5,
        project_root=tmp_path,
        output_dir=tmp_path / "out",
    )

    summary = frames["summary"].set_index(["dataset", "component"])
    assert summary.loc[("ml1m", "ml1m_ratings"), "status"] == "valid"
    assert summary.loc[("ml1m", "ml1m_movies"), "status"] == "valid"
    readiness = frames["readiness"].set_index("dataset").loc["ml1m"]
    assert readiness["raw_reconstruction_input_available"] == "yes"
    assert readiness["chronological_split_supported"] == "yes"
    assert readiness["expected_reconstruction_status"] == "reconstructible_from_public_raw"


def test_lastfm_zip_member_parsing_and_partial_chronology(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    staging.mkdir()
    with zipfile.ZipFile(staging / "hetrec2011-lastfm-2k.zip", "w") as archive:
        archive.writestr("user_artists.dat", "userID\tartistID\tweight\n1\t10\t5\n")
        archive.writestr("artists.dat", "id\tname\turl\tpictureURL\n10\tArtist\turl\tpic\n")
        archive.writestr(
            "user_taggedartists.dat",
            "userID\tartistID\ttagID\tday\tmonth\tyear\n1\t10\t99\t1\t2\t2011\n",
        )

    frames = validate_raw_datasets(
        staging,
        sample_size=5,
        project_root=tmp_path,
        output_dir=tmp_path / "out",
    )

    summary = frames["summary"].set_index(["dataset", "component"])
    assert summary.loc[("lastfm", "lastfm_user_artists"), "status"] == "valid"
    assert summary.loc[("lastfm", "lastfm_artists"), "status"] == "valid"
    readiness = frames["readiness"].set_index("dataset").loc["lastfm"]
    assert readiness["chronological_split_supported"] == "partial"
    assert readiness["candidate_generation_possible"] == "yes"
    assert readiness["expected_reconstruction_status"] == "extension_only"


def test_missing_files_produce_diagnostics_not_crashes(tmp_path: Path) -> None:
    staging = tmp_path / "empty"
    staging.mkdir()

    frames = validate_raw_datasets(
        staging,
        sample_size=5,
        project_root=tmp_path,
        output_dir=tmp_path / "out",
    )

    summary = frames["summary"]
    assert "missing" in set(summary["status"])
    readiness = frames["readiness"].set_index("dataset")
    assert readiness.loc["amazon_beauty", "expected_reconstruction_status"] == "blocked_missing_raw"
    assert readiness.loc["ml1m", "raw_reconstruction_input_available"] == "no"


def test_no_permanent_extraction_and_no_metric_outputs(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    staging.mkdir()
    with zipfile.ZipFile(staging / "ml-1m.zip", "w") as archive:
        archive.writestr("ml-1m/ratings.dat", "1::10::5::100\n")
        archive.writestr("ml-1m/movies.dat", "10::Movie A::Drama\n")

    validate_raw_datasets(
        staging,
        sample_size=1,
        project_root=tmp_path,
        output_dir=tmp_path / "results" / "tables",
    )

    assert sorted(path.name for path in staging.iterdir()) == ["ml-1m.zip"]
    assert not (tmp_path / "results" / "metrics").exists()
    assert (tmp_path / "results" / "tables" / "plan_c_raw_dataset_validation_summary.csv").exists()
