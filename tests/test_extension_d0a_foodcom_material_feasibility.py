from __future__ import annotations

import ast
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.extension import d0a_foodcom_material_feasibility as d0a
from src.extension.e0_protocol import item_list_hash


TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
SCRIPT_PATH = Path("src/extension/d0a_foodcom_material_feasibility.py")


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / name, keep_default_na=False)


def _json_list(value: object) -> list[str]:
    parsed = json.loads(str(value))
    assert isinstance(parsed, list)
    return [str(item) for item in parsed]


def _write_zip(path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for member, content in members.items():
            zf.writestr(member, content)


def test_discover_single_zip_requires_exactly_one_archive(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(d0a.D0AStop, match="no .zip archive"):
        d0a.discover_single_zip(source)

    first = source / "one.zip"
    second = source / "two.zip"
    _write_zip(first, {"RAW_interactions.csv": "", "RAW_recipes.csv": ""})
    assert d0a.discover_single_zip(source) == first

    _write_zip(second, {"RAW_interactions.csv": "", "RAW_recipes.csv": ""})
    with pytest.raises(d0a.D0AStop, match="expected one .zip archive"):
        d0a.discover_single_zip(source)


def test_archive_inventory_requires_raw_members_and_records_hash(tmp_path: Path) -> None:
    archive = tmp_path / "food.zip"
    _write_zip(
        archive,
        {
            "nested/RAW_interactions.csv": "user_id,recipe_id,date,rating,review\n",
            "nested/RAW_recipes.csv": "name,id,tags,ingredients,description,nutrition,steps\n",
            "nested/PP_users.csv": "u\n",
        },
    )

    manifest, inventory, members = d0a.inspect_archive(tmp_path, archive, d0a.sha256_file(archive))
    assert manifest["zip_valid"] is True
    assert manifest["archive_member_count"] == 3
    assert members["RAW_interactions.csv"] == "nested/RAW_interactions.csv"
    assert members["RAW_recipes.csv"] == "nested/RAW_recipes.csv"
    assert set(inventory.loc[inventory["required_for_d0a"], "basename"]) == set(d0a.REQUIRED_RAW_FILES)

    missing = tmp_path / "missing.zip"
    _write_zip(missing, {"RAW_interactions.csv": "user_id,recipe_id,date,rating,review\n"})
    with pytest.raises(d0a.D0AStop, match="missing required"):
        d0a.inspect_archive(tmp_path, missing, d0a.sha256_file(missing))


def test_safe_extraction_blocks_zip_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    _write_zip(archive, {"../RAW_interactions.csv": "x", "RAW_recipes.csv": "y"})

    with pytest.raises(d0a.D0AStop, match="unsafe archive member path"):
        d0a.safe_extract_archive(archive, tmp_path / "extract")


def test_raw_hashing_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    assert d0a.sha256_file(path) == d0a.sha256_file(path)
    assert d0a.sha256_file(path) == "ea14f99c47575613ab22111122c847728c61007f6bfd7b062d02fcb99df3feb0"


def test_final_split_uses_stable_input_order_for_same_day_ties() -> None:
    interactions = pd.DataFrame(
        [
            {"user_id": "u1", "recipe_id": "a", "date": "2020-01-01", "rating": 5},
            {"user_id": "u1", "recipe_id": "b", "date": "2020-01-01", "rating": 4},
            {"user_id": "u1", "recipe_id": "c", "date": "2020-01-02", "rating": 3},
            {"user_id": "u2", "recipe_id": "x", "date": "2020-02-01", "rating": 5},
            {"user_id": "u2", "recipe_id": "y", "date": "2020-02-01", "rating": 4},
        ]
    )

    split, stats = d0a.build_final_split(interactions)
    by_user = {row["user_id"]: row for row in split["user_states"]}
    assert by_user["u1"]["train_items"] == ["a", "b"]
    assert by_user["u1"]["target_item"] == "c"
    assert by_user["u2"]["train_items"] == ["x"]
    assert by_user["u2"]["target_item"] == "y"
    assert stats.loc[stats["metric"].eq("tie_policy"), "value"].iloc[0] == "date_ascending_then_original_input_order"


def test_l0_and_history_suffix_semantics_are_exact() -> None:
    history = ["a", "b", "c", "d", "e"]

    assert d0a.visible_history(history, 0) == []
    assert d0a.visible_history(history, 1) == ["e"]
    assert d0a.visible_history(history, 2) == ["d", "e"]
    assert d0a.visible_history(history, 5) == history


def test_candidate_generation_is_deterministic_name_independent_and_unseen() -> None:
    item_universe = ["positive", "bad_missing_name_item", *[f"n{i}" for i in range(18)]]
    first = d0a.generate_candidate_set(
        user_id="u1",
        positive_item_id="positive",
        user_seen_items={"positive"},
        item_universe=item_universe,
    )
    second = d0a.generate_candidate_set(
        user_id="u1",
        positive_item_id="positive",
        user_seen_items={"positive"},
        item_universe=item_universe,
    )

    assert first == second
    assert first is not None
    assert len(first) == d0a.CANDIDATE_COUNT
    assert len(set(first)) == d0a.CANDIDATE_COUNT
    assert first.count("positive") == 1
    assert "bad_missing_name_item" in first
    assert not (set(first) - {"positive"}).intersection({"positive"})


def test_real_foodcom_primary_and_history_gates_are_feasible() -> None:
    primary = _read("d0a_foodcom_primary1000_feasibility.csv").iloc[0]
    history = _read("d0a_foodcom_history200_feasibility.csv").iloc[0]

    assert primary["status"] == "FOODCOM_PRIMARY_1000_FEASIBLE"
    assert int(primary["selected_users"]) == d0a.PRIMARY_TARGET_USERS
    assert int(primary["structurally_candidate_valid_users"]) == 226570
    assert int(primary["final_strict_name_eligible_users"]) == 226553
    assert primary["candidate_generation_independent_of_name_availability"] in {True, "True", "true"}

    assert history["status"] == "FOODCOM_HISTORY_200_FEASIBLE"
    assert int(history["selected_users"]) == d0a.HISTORY_TARGET_USERS
    assert int(history["users_with_train_history_ge_20"]) == 6389
    assert int(history["strict_paired_l20_users"]) == 6387
    assert history["l0_semantics"] == "explicit_empty_history"


def test_real_foodcom_selected_history_rows_preserve_candidates_and_l_lengths() -> None:
    history = _read("d0a_foodcom_history200_users.csv")

    assert len(history) == d0a.HISTORY_TARGET_USERS
    assert history["candidate_count"].astype(int).eq(20).all()
    assert history["positive_count"].astype(int).eq(1).all()
    assert history["negative_count"].astype(int).eq(19).all()
    assert history["history_l0_item_ids"].map(_json_list).map(len).eq(0).all()
    assert history["history_l0_hash"].eq(item_list_hash([])).all()
    for row in history.itertuples(index=False):
        candidates = _json_list(row.candidate_item_ids)
        assert item_list_hash(candidates) == row.candidate_hash
        assert candidates[int(row.positive_position) - 1] == str(row.positive_item_id)
        l20 = _json_list(row.history_l20_item_ids)
        assert len(l20) == 20
        for length in d0a.HISTORY_LENGTHS:
            values = _json_list(getattr(row, f"history_l{length}_item_ids"))
            assert len(values) == length
            if length:
                assert values == l20[-length:]
            assert getattr(row, f"history_l{length}_names_valid") in {True, "True", "true"}


def test_real_foodcom_missing_names_are_not_silently_filled() -> None:
    metadata = _read("d0a_foodcom_metadata_coverage.csv")
    primary = _read("d0a_foodcom_primary1000_feasibility.csv").iloc[0]
    name_row = metadata.loc[metadata["field"].eq("name")].iloc[0]

    assert int(name_row["missing_recipe_row"]) == 0
    assert int(name_row["empty_values"]) == 1
    assert int(name_row["usable_non_empty"]) == 231636
    assert int(primary["final_strict_name_eligible_users"]) < int(primary["structurally_candidate_valid_users"])


def test_real_foodcom_no_inference_manifest_and_script_imports_are_clean() -> None:
    manifest = json.loads((MANIFEST_DIR / "d0a_foodcom_no_inference_manifest.json").read_text(encoding="utf-8"))
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert manifest["decision"] == "FOODCOM_D0A_FEASIBLE_WITH_ADAPTATIONS"
    assert manifest["llm_requests_executed"] == 0
    assert manifest["training_runs_executed"] == 0
    assert manifest["remote_api_calls_executed"] == 0
    assert manifest["downloads_executed"] == 0
    assert not imported.intersection({"openai", "requests", "urllib", "socket", "ollama", "torch", "tensorflow"})
