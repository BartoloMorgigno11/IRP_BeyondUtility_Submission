from pathlib import Path

from src.analysis import reproduction_r5d_generation_involved_recovery as r5d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "reproduction_r5d_generation_involved_recovery.yaml"


def r5d_frames():
    config = r5d.load_config(CONFIG_PATH)
    _, frame, manifest_sha = r5d.load_r5d_manifest(PROJECT_ROOT, config)
    return config, frame, manifest_sha


def test_r5d_scope_is_generation_family_only() -> None:
    config, frame, _ = r5d_frames()

    assert len(frame) == 1200
    assert set(frame["recovery_family"]) == {"GENERATION_INVOLVED_RECOVERY"}
    assert set(frame["task"]) == {"profile_generation", "ranking"}
    assert int(frame["task"].eq("profile_generation").sum()) == 400
    assert int(frame["task"].eq("ranking").sum()) == 800
    assert set(frame["dataset"]) == set(config["scope"]["datasets"])
    assert set(frame["model_name"]) == set(config["scope"]["models"])
    assert frame["output_destination"].astype(str).str.startswith(
        config["outputs"]["result_namespace"].rstrip("/") + "/"
    ).all()


def test_r5d_dependency_graph_is_closed() -> None:
    _, frame, _ = r5d_frames()
    dependency = r5d.build_dependency_validation(frame)

    assert len(dependency) == 800
    assert dependency["dependency_valid"].eq("true").all()
    generated = dependency.loc[dependency["dataset"].isin(["ML-1M", "LastFM"])]
    reused = dependency.loc[dependency["dataset"].isin(["Beauty", "Sports"])]
    assert len(generated) == 400
    assert generated["r5d_profile_dependency_request_id"].astype(str).str.startswith("R5D-PROF-").all()
    assert len(reused) == 400
    assert reused["r5d_profile_dependency_request_id"].eq("").all()


def test_r5d_prompts_make_contract_unambiguous() -> None:
    _, frame, _ = r5d_frames()
    ranking = frame.loc[frame["task"].eq("ranking"), "prompt_render"].astype(str)
    profile = frame.loc[frame["task"].eq("profile_generation"), "prompt_render"].astype(str)

    assert ranking.str.contains("EXACTLY 10", regex=False).all()
    assert ranking.str.contains("candidate POSITIONS 1..20", regex=False).all()
    assert ranking.str.contains("not dataset item IDs", regex=False).all()
    assert ranking.str.contains("Do not return item IDs", regex=False).all()
    assert profile.str.contains("exactly one key user_profile", regex=False).all()
    assert profile.str.contains("at most 80 words", regex=False).all()
    assert profile.str.contains("must not contain JSON", regex=False).all()


def test_r5d_synthetic_rejects_invalid_formats() -> None:
    _, frame, _ = r5d_frames()
    synthetic = r5d.build_synthetic_tests(frame)

    assert synthetic["pass"].eq("true").all()
    item_id = synthetic.loc[synthetic["case_id"].eq("ranking_item_ids_in_indices_field")].iloc[0]
    assert item_id["observed_status"] == "unknown_indices"
    assert item_id["expected_valid"] == "false"
    profile_extra = synthetic.loc[synthetic["case_id"].eq("profile_extra_key")].iloc[0]
    assert profile_extra["observed_status"] == "invalid_schema"


def test_r5d_canary_selection_covers_all_distinct_cells_and_dependencies() -> None:
    _, frame, _ = r5d_frames()
    canary = r5d.select_canary_rows(frame)
    cells = {
        (row["task"], row["model_name"], row["dataset"], row["experimental_condition"])
        for row in canary
    }

    assert len(canary) == 12
    assert len(cells) == 12
    profile_ids = {row["request_id"] for row in canary if row["task"] == "profile_generation"}
    dependent = [
        row for row in canary
        if row["task"] == "ranking" and row["dataset"] in {"ML-1M", "LastFM"}
    ]
    assert dependent
    assert all(row["r5d_profile_dependency_request_id"] in profile_ids for row in dependent)


def test_r5d_protocol_keeps_zero_retries() -> None:
    config = r5d.load_config(CONFIG_PATH)

    assert int(config["runtime"]["max_retries"]) == 0
