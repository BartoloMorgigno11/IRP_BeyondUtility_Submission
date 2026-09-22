from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import yaml

from src.data.audit_datasets import audit_dataset
from src.data.load_data import list_zip_members, read_csv_from_zip, read_json_from_zip


def test_zip_helpers_read_without_extracting(tmp_path: Path) -> None:
    zip_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("records.json", json.dumps([{"user": 1, "item": "A"}]))
        archive.writestr("table.tsv", "user_id\trec_items\n1\t[1, 2]\n")

    members = list_zip_members(zip_path)
    records = read_json_from_zip(zip_path, "records.json")
    table = read_csv_from_zip(zip_path, "table.tsv", sep="\t")

    assert {member["name"] for member in members} == {"records.json", "table.tsv"}
    assert records == [{"user": 1, "item": "A"}]
    assert table.to_dict("records") == [{"user_id": 1, "rec_items": "[1, 2]"}]


def test_audit_dataset_reports_unknown_core_stats_without_canonical_data() -> None:
    audit = audit_dataset(Path("configs/sports.yaml"))

    core = audit[audit["category"] == "core_dataset_stat"].set_index("statistic")
    llm_stats = audit[audit["statistic"] == "llm_record_count"]

    assert core.loc["user_count", "value"] == "unknown"
    assert core.loc["interaction_count", "value"] == "unknown"
    assert core.loc["candidate_list_availability", "value"] == "available_in_configured_llm_outputs"
    assert not llm_stats.empty
    assert int(llm_stats.iloc[0]["value"]) == 200


def test_audit_dataset_has_expected_columns() -> None:
    audit = audit_dataset(Path("configs/beauty.yaml"))

    expected_columns = {
        "dataset",
        "config_path",
        "category",
        "item",
        "config_key",
        "configured_value",
        "resolved_path",
        "exists",
        "status",
        "statistic",
        "value",
        "note",
    }

    assert set(audit.columns) == expected_columns
    assert isinstance(audit, pd.DataFrame)


def test_dataset_audit_outputs_are_configured_under_results_tables() -> None:
    for config_path in (Path("configs/beauty.yaml"), Path("configs/sports.yaml")):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        assert config["output"]["dataset_audit_path"].startswith("results/tables/")
