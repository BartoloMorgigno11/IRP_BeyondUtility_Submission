from pathlib import Path

from src.analysis import reproduction_r5c_history_length_corrected as corrected


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "reproduction_r5c_history_length_corrected.yaml"


def corrected_frames():
    config = corrected.load_config(CONFIG_PATH)
    _, history, manifest_sha = corrected.load_r5c_manifest(PROJECT_ROOT, config)
    return config, history, manifest_sha


def test_corrected_scope_preserves_2400_rows_and_new_namespace() -> None:
    config, history, _ = corrected_frames()

    assert len(history) == 2400
    assert set(history["recovery_family"]) == {"HISTORY_LENGTH_RECOVERY"}
    assert set(history["task"]) == {"ranking"}
    assert set(history["dataset"]) == set(config["scope"]["datasets"])
    assert set(history["model_name"]) == set(config["scope"]["models"])
    assert history["output_destination"].astype(str).str.startswith(
        "results/recovery/r5c_history_length_corrected/"
    ).all()
    assert not history["output_destination"].astype(str).str.startswith(
        "results/recovery/r5c_history_length/"
    ).any()


def test_corrected_prompt_makes_candidate_positions_unambiguous() -> None:
    _, history, _ = corrected_frames()
    prompts = history["prompt_render"].astype(str)

    assert prompts.str.contains("Return exactly 5 1-based candidate indices", regex=False).all()
    assert prompts.str.contains("candidate POSITIONS 1..20", regex=False).all()
    assert prompts.str.contains("not dataset item IDs", regex=False).all()
    assert prompts.str.contains("Do not return item IDs", regex=False).all()
    assert prompts.str.contains("EXACTLY 5", regex=False).all()


def test_corrected_protocol_hash_differs_from_original_r5c() -> None:
    config, history, manifest_sha = corrected_frames()
    groups = corrected.base.build_group_validation(history, config)
    context = corrected.base.build_context_preflight(history, config, {})
    protocol_hash, _ = corrected.protocol_metadata_corrected(
        PROJECT_ROOT,
        CONFIG_PATH,
        config,
        history,
        manifest_sha,
        groups,
        {},
    )
    static = corrected.build_static_preflight_corrected(
        PROJECT_ROOT,
        config,
        corrected.read_csv(PROJECT_ROOT / config["frozen_r5a"]["manifest"]),
        history,
        manifest_sha,
        groups,
        context,
        protocol_hash,
    )

    assert protocol_hash != corrected.R5C_ORIGINAL_PROTOCOL_HASH
    assert len(protocol_hash) == 64
    head_guard = static.loc[static["check_id"].eq("STATIC-002")].iloc[0]
    remaining_checks = static.loc[~static["check_id"].eq("STATIC-002")]

    # STATIC-002 was the historical R5C freeze precondition. Later freeze
    # commits legitimately move repository HEAD while preserving the R5B
    # baseline SHA in config provenance.
    assert remaining_checks["status"].eq("pass").all()
    assert head_guard["check_name"] == "repository HEAD is R5B corrected freeze"
    assert head_guard["expected"] == config["frozen_r5b_corrected"]["freeze_commit"]
    if head_guard["status"] == "fail":
        assert head_guard["observed"] != head_guard["expected"]
    else:
        assert head_guard["status"] == "pass"


def test_synthetic_rejects_actual_ml1m_item_ids_as_out_of_range() -> None:
    _, history, _ = corrected_frames()
    synthetic = corrected.build_synthetic_tests_corrected(history)
    row = synthetic.loc[synthetic["case_id"].eq("ml1m_item_ids_in_indices_field")].iloc[0]

    assert row["expected_status"] == "unknown_indices"
    assert row["observed_status"] == "unknown_indices"
    assert row["pass"] == "true"


def test_corrected_canary_contains_required_qwen2_ml1m_history_lengths() -> None:
    _, history, _ = corrected_frames()
    canary = corrected.base.select_first_n_per_cell(history, 1)
    cells = {(row["model_name"], row["dataset"], row["experimental_condition"]) for row in canary}

    assert ("qwen2:7b-instruct", "ML-1M", "HISTORY_LENGTH_5") in cells
    assert ("qwen2:7b-instruct", "ML-1M", "HISTORY_LENGTH_10") in cells
    assert ("qwen2:7b-instruct", "ML-1M", "HISTORY_LENGTH_20") in cells
    assert ("qwen2:7b-instruct", "Beauty", "HISTORY_LENGTH_0") in cells
    assert len(cells) == 24


def test_corrected_runtime_keeps_zero_retries() -> None:
    config = corrected.load_config(CONFIG_PATH)

    assert int(config["runtime"]["max_retries"]) == 0
