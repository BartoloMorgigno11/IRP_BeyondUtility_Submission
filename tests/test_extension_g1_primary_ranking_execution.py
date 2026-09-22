from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import pandas as pd
import pytest

from src.extension import g1_primary_ranking_execution as g1


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g1_primary_ranking_execution.yaml"


@lru_cache(maxsize=1)
def _offline_plan() -> dict:
    return g1.build_offline_g1_plan(ROOT, CONFIG)


def _config_copy() -> dict:
    config = g1.load_config(ROOT, CONFIG)
    config = dict(config)
    config["execution"] = dict(config["execution"])
    return config


def _g0_inputs() -> SimpleNamespace:
    return SimpleNamespace(g0_execution_manifest=g1.read_json(ROOT / "results" / "manifests" / "g0_canary_execution_manifest.json"))


def _tags_payload_from_g0(*, drop: set[str] | None = None, override_digest: dict[str, str] | None = None) -> dict:
    drop = drop or set()
    override_digest = override_digest or {}
    models = []
    for row in _g0_inputs().g0_execution_manifest["model_preflight"]:
        model = str(row["model"])
        if model in drop:
            continue
        models.append(
            {
                "name": model,
                "model": model,
                "modified_at": row["modified_at"],
                "size": int(row["size_bytes"]),
                "digest": override_digest.get(model, row["digest"]),
                "details": {
                    "parameter_size": {"qwen2:7b-instruct": "7.6B", "mistral:7b-instruct": "7.2B", "gemma2:9b": "9.2B"}[model],
                    "quantization_level": {"qwen2:7b-instruct": "Q4_0", "mistral:7b-instruct": "Q4_K_M", "gemma2:9b": "Q4_0"}[model],
                    "context_length": {"qwen2:7b-instruct": 32768, "mistral:7b-instruct": 32768, "gemma2:9b": 8192}[model],
                },
                "capabilities": ["completion"],
            }
        )
    return {"models": models}


def _fake_http(tags_payload: dict) -> tuple:
    calls: list[str] = []

    def fake(method: str, url: str, payload: dict | None, *, timeout: int) -> dict:
        calls.append(url)
        if url.endswith("/api/version"):
            return {"version": "0.33.2"}
        if url.endswith("/api/tags"):
            return tags_payload
        if url.endswith("/api/ps"):
            return {"models": [{"name": "gemma2:9b"}]}
        raise AssertionError(url)

    return fake, calls


def test_g1_config_is_primary_local_only_and_batchable() -> None:
    config = g1.load_config(ROOT, CONFIG)

    assert "expected_head" not in config["phase"]
    assert config["phase"]["expected_base_checkpoint"] == g1.SCIENTIFIC_BASE_CHECKPOINT
    assert config["phase"]["base_checkpoint_required_as_ancestor"] is True
    assert config["tasks"] == ["ranking"]
    assert config["models"] == list(g1.MODEL_ORDER)
    assert config["prompt_strategies"] == list(g1.PROMPT_ORDER)
    assert config["execution"]["ollama_base_url"] == "http://127.0.0.1:11434"
    assert config["execution"]["backend_retry_count"] == 2
    assert config["execution"]["backend_retry_backoff_seconds"] == [5, 15]
    assert config["execution"]["scientific_retry_count"] == 0
    assert config["execution"]["allow_local_ollama_metadata"] is True
    assert config["execution"]["canonical_request_order"] == "dataset_user_prompt_model"
    assert config["execution"]["physical_execution_order"] == "model_dataset_prompt_user"

    forbidden = [
        "allow_external_api_calls",
        "allow_downloads",
        "allow_network_downloads",
        "allow_training",
        "allow_tuning",
        "allow_reranking_execution",
        "allow_history_grid_execution",
        "allow_canddif_execution",
        "allow_profile_generation",
        "allow_pool_size_sensitivity",
        "allow_commit",
    ]
    assert all(config["execution"][key] is False for key in forbidden)
    g1.validate_local_endpoint(config)

    nonlocal_config = dict(config)
    nonlocal_config["execution"] = dict(config["execution"], ollama_base_url="https://api.example.test")
    with pytest.raises(ValueError, match="loopback"):
        g1.validate_local_endpoint(nonlocal_config)


def test_g1p_local_metadata_preflight_reports_unreachable_without_generic_missing() -> None:
    config = _config_copy()

    def unreachable(method: str, url: str, payload: dict | None, *, timeout: int) -> dict:
        return {"_error": "URLError: connection refused", "models": []} if url.endswith("/api/tags") else {}

    frame = g1.ollama_model_preflight(config, _g0_inputs(), http_get=unreachable)

    assert set(frame["current_model_identity_status"]) == {g1.OLLAMA_UNREACHABLE}
    assert set(frame["status"]) == {g1.OLLAMA_UNREACHABLE}
    assert frame["available"].eq("false").all()
    assert not frame["status"].isin({"AVAILABLE_LOCAL", "BLOCKED_MISSING_LOCAL_MODEL"}).any()


def test_g1p_local_metadata_preflight_reports_permission_block_without_http_call() -> None:
    config = _config_copy()
    config["execution"]["allow_local_ollama_metadata"] = False

    def forbidden_call(method: str, url: str, payload: dict | None, *, timeout: int) -> dict:
        raise AssertionError("metadata HTTP should not be called when permission is disabled")

    frame = g1.ollama_model_preflight(config, _g0_inputs(), http_get=forbidden_call)

    assert set(frame["current_model_identity_status"]) == {g1.LOCAL_METADATA_CHECK_NOT_PERMITTED}
    assert frame["local_metadata_request_count"].astype(int).eq(0).all()
    assert frame["llm_generation_call"].eq("false").all()


def test_g1p_preflight_reports_missing_model_and_digest_mismatch_separately() -> None:
    config = _config_copy()
    fake, _calls = _fake_http(_tags_payload_from_g0(drop={"gemma2:9b"}))
    missing = g1.ollama_model_preflight(config, _g0_inputs(), http_get=fake).set_index("model")

    assert missing.loc["gemma2:9b", "current_model_identity_status"] == g1.MODEL_NOT_INSTALLED
    assert missing.loc["qwen2:7b-instruct", "current_model_identity_status"] == g1.MODEL_AVAILABLE_MATCHED

    fake, _calls = _fake_http(_tags_payload_from_g0(override_digest={"qwen2:7b-instruct": "0" * 64}))
    mismatch = g1.ollama_model_preflight(config, _g0_inputs(), http_get=fake).set_index("model")

    assert mismatch.loc["qwen2:7b-instruct", "current_model_identity_status"] == g1.MODEL_DIGEST_MISMATCH
    assert mismatch.loc["qwen2:7b-instruct", "g0_expected_digest"].startswith("dd314f039b9d")
    assert mismatch.loc["qwen2:7b-instruct", "current_local_digest"] == "0" * 64
    assert mismatch.loc["qwen2:7b-instruct", "digest_match"] == "false"


def test_g1p_preflight_parses_current_ollama_tags_schema_and_matches_g0_digests() -> None:
    config = _config_copy()
    fake, calls = _fake_http(_tags_payload_from_g0())
    frame = g1.ollama_model_preflight(config, _g0_inputs(), http_get=fake)

    assert frame["current_model_identity_status"].tolist() == [g1.MODEL_AVAILABLE_MATCHED] * 3
    assert frame["g0_model_identity_status"].eq(g1.G0_PROVENANCE_AVAILABLE).all()
    assert frame["digest_match"].eq("true").all()
    assert frame["llm_generation_call"].eq("false").all()
    assert frame["local_metadata_request_count"].astype(int).eq(3).all()
    assert {url.rsplit("/", 1)[-1] for url in calls} == {"version", "tags", "ps"}

    qwen = frame.set_index("model").loc["qwen2:7b-instruct"]
    assert qwen["parameter_size"] == "7.6B"
    assert qwen["quantization_level"] == "Q4_0"
    assert qwen["context_length"] == "32768"


def test_g1p_rejects_external_endpoint_and_non_loopback_redirect() -> None:
    config = _config_copy()
    config["execution"]["ollama_base_url"] = "http://192.168.1.10:11434"
    with pytest.raises(ValueError, match="loopback"):
        g1.validate_local_endpoint(config)

    request = urllib.request.Request("http://127.0.0.1:11434/api/tags")
    handler = g1.LoopbackOnlyRedirectHandler()
    with pytest.raises(ValueError, match="loopback"):
        handler.redirect_request(request, None, 302, "Found", {}, "http://192.168.1.10:11434/api/tags")


def test_g1p_validate_only_cli_path_does_not_require_generation_permission(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, bool] = {}

    def fake_run(root: Path, config_path: Path, *, resume: bool, validate_only: bool, max_new_calls: int | None) -> dict:
        assert validate_only is True
        assert max_new_calls is None
        return {
            "manifests": {
                "execution_manifest": {
                    "decision_token": g1.DECISION_BLOCKED,
                    "planned_requests": 29_292,
                    "g0_reused_ranking_requests": 36,
                    "new_g1_terminal_requests": 0,
                    "new_g1_ollama_calls_expected": 29_256,
                    "llm_inference_calls_executed_in_this_invocation": 0,
                    "terminal_requests": 36,
                    "strict_valid_requests": 36,
                    "validation_passed": False,
                }
            }
        }

    monkeypatch.setattr(g1, "run_g1", fake_run)
    monkeypatch.setattr(g1, "write_outputs", lambda root, outputs: calls.setdefault("write_outputs", True))
    monkeypatch.setattr(g1, "append_phase_notes", lambda root, outputs, validate_only: calls.setdefault("append_notes", validate_only))

    exit_code = g1.main(["--config", str(CONFIG), "--validate-only"])

    assert exit_code == 1
    assert calls == {"write_outputs": True, "append_notes": True}


def test_g1f_validate_only_no_write_keeps_post_commit_gate_side_effect_free(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, bool] = {}

    def fake_run(root: Path, config_path: Path, *, resume: bool, validate_only: bool, max_new_calls: int | None) -> dict:
        assert validate_only is True
        return {
            "manifests": {
                "execution_manifest": {
                    "decision_token": g1.DECISION_BLOCKED,
                    "planned_requests": 29_292,
                    "g0_reused_ranking_requests": 36,
                    "new_g1_terminal_requests": 0,
                    "new_g1_ollama_calls_expected": 29_256,
                    "llm_inference_calls_executed_in_this_invocation": 0,
                    "terminal_requests": 36,
                    "strict_valid_requests": 36,
                    "validation_passed": False,
                }
            }
        }

    monkeypatch.setattr(g1, "run_g1", fake_run)
    monkeypatch.setattr(g1, "write_outputs", lambda root, outputs: calls.setdefault("write_outputs", True))
    monkeypatch.setattr(g1, "append_phase_notes", lambda root, outputs, validate_only: calls.setdefault("append_notes", validate_only))

    exit_code = g1.main(["--config", str(CONFIG), "--validate-only", "--no-write"])

    assert exit_code == 1
    assert calls == {}


def test_g1_cli_requires_explicit_allow_api_calls_for_inference(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = g1.main(["--config", str(CONFIG), "--max-new-calls", "0"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--allow-api-calls" in captured.out


def test_g1_offline_universe_size_cells_order_and_contract() -> None:
    request_universe = _offline_plan()["request_universe"]

    assert len(request_universe) == 29_292
    assert request_universe["canonical_request_id"].is_unique
    assert request_universe.groupby(["dataset", "model", "prompt_strategy"]).ngroups == 30
    assert request_universe["task"].eq("ranking").all()
    assert request_universe["candidate_count"].astype(int).eq(20).all()
    assert request_universe["requested_output_length"].astype(int).eq(10).all()
    assert request_universe["parser_expected_length"].astype(int).eq(10).all()
    assert request_universe["metric_k"].astype(int).eq(5).all()
    assert request_universe["candidate_hash"].str.len().eq(64).all()
    assert request_universe["prompt_hash"].str.len().eq(64).all()
    assert request_universe["model_digest"].str.len().eq(64).all()
    assert not request_universe.astype(str).apply(
        lambda column: column.str.contains("video_games|video games|videogames", case=False, regex=True)
    ).any().any()

    dataset_users = request_universe.drop_duplicates(["dataset", "user_id"]).groupby("dataset").size().to_dict()
    assert dataset_users == {"beauty": 990, "sports": 892, "ml1m": 1000, "lastfm": 1000, "foodcom": 1000}

    first_beauty_user = request_universe.loc[request_universe["dataset"].eq("beauty"), "user_id"].iloc[0]
    first_user = request_universe[request_universe["dataset"].eq("beauty") & request_universe["user_id"].eq(first_beauty_user)]
    assert first_user[["prompt_strategy", "model"]].to_records(index=False).tolist() == [
        ("BASE", "qwen2:7b-instruct"),
        ("BASE", "mistral:7b-instruct"),
        ("BASE", "gemma2:9b"),
        ("LLMRANK", "qwen2:7b-instruct"),
        ("LLMRANK", "mistral:7b-instruct"),
        ("LLMRANK", "gemma2:9b"),
    ]


def test_g1f_head_semantics_use_base_ancestor_not_current_head_equality(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config_copy()
    inputs = _offline_plan()["inputs"]
    request_universe = pd.DataFrame(
        [
                {
                    "canonical_request_id": "REQ-1",
                    "request_order": 1,
                    "dataset": "beauty",
                "user_id": "1",
                "model": "qwen2:7b-instruct",
                "prompt_strategy": "BASE",
                "task": "ranking",
            }
        ]
    )
    physical = g1.build_physical_execution_order_table(request_universe, pd.DataFrame())
    model_preflight = pd.DataFrame(
        [
            {"model": model, "available": "true", "digest": row["digest"], "current_model_identity_status": g1.MODEL_AVAILABLE_MATCHED}
            for model, row in pd.DataFrame(inputs.g0_execution_manifest["model_preflight"]).set_index("model").iterrows()
        ]
    )

    monkeypatch.setattr(g1, "git_output", lambda root, args: "new-committed-head" if args == ["rev-parse", "HEAD"] else "")
    monkeypatch.setattr(g1, "git_is_ancestor", lambda root, ancestor, descendant: ancestor == g1.SCIENTIFIC_BASE_CHECKPOINT and descendant == "new-committed-head")

    defects = g1.pre_execution_protocol_defects(ROOT, config, inputs, request_universe, physical, pd.DataFrame(), model_preflight, "")

    assert "precondition_head_mismatch" not in defects
    assert "scientific_base_not_ancestor_of_execution_head" not in defects


def test_g1f_generation_and_retry_contracts_remain_f0_frozen() -> None:
    config = _config_copy()
    inputs = _offline_plan()["inputs"]
    contract = inputs.f0_inference_contract
    generation = contract["generation_parameters"]

    assert generation["temperature"] == 0.0
    assert generation["num_predict"] == 256
    assert generation["stream"] is False
    assert generation["timeout_seconds"] == 150
    assert generation["top_p"] == ""
    assert generation["top_k"] == ""
    assert generation["seed"] == ""
    assert generation["context_window"] == ""
    assert generation["stop_sequences"] == ""
    assert generation["keep_alive"] == ""
    assert config["execution"]["response_format"] == "json"
    assert g1.backend_retry_count(config, inputs) == 2
    assert g1.backend_retry_backoffs(config, inputs) == [5, 15]
    assert config["execution"]["scientific_retry_count"] == 0


def test_g1f_physical_execution_order_is_model_grouped_without_changing_logical_order() -> None:
    plan = _offline_plan()
    request_universe = plan["request_universe"]
    physical = plan["physical_execution_order"]

    assert g1.logical_order_is_dataset_user_prompt_model(request_universe)
    assert g1.physical_order_is_model_grouped(physical)
    assert g1.physical_model_switch_count(physical) == 2
    assert physical["canonical_request_id"].nunique() == len(request_universe)
    assert physical.sort_values("physical_execution_order")["model"].drop_duplicates().tolist() == list(g1.MODEL_ORDER)
    assert physical.loc[physical["execution_action"].eq("REUSE_G0_NO_CALL"), "canonical_request_id"].nunique() == 36
    assert request_universe.sort_values("request_order")["canonical_request_id"].tolist() != physical.sort_values("physical_execution_order")["canonical_request_id"].tolist()


def test_g1_g0_ranking_reuse_exact_and_new_call_accounting() -> None:
    plan = _offline_plan()
    request_universe = plan["request_universe"]
    reuse = plan["g0_reuse_inventory"]

    assert len(reuse) == 36
    assert reuse["reused_in_g1"].eq("true").all()
    assert reuse["all_identity_fields_match"].eq("true").all()
    assert set(reuse["task"]) == {"ranking"}
    assert len(request_universe) - int(reuse["reused_in_g1"].eq("true").sum()) == 29_256
    assert reuse["g0_raw_response_path"].str.startswith("results/llm/extension_canonical/g0_canary/responses/").all()
    assert not reuse["g1_raw_response_path"].str.startswith("results/llm/extension_canonical/g0_canary/").any()


def test_g1_request_identity_matches_g0_ranking_evidence() -> None:
    plan = _offline_plan()
    request_universe = plan["request_universe"].set_index("canonical_request_id")
    reuse = plan["g0_reuse_inventory"]

    sample = reuse.iloc[0]
    request = request_universe.loc[sample["canonical_request_id"]]
    assert request["dataset"] == sample["dataset"]
    assert request["user_id"] == sample["user_id"]
    assert request["model"] == sample["model"]
    assert request["prompt_strategy"] == sample["prompt_strategy"]
    assert request["candidate_hash"] == sample["g0_candidate_hash"] == sample["g1_candidate_hash"]
    assert request["prompt_hash"] == sample["g0_prompt_hash"] == sample["g1_prompt_hash"]
    assert request["candidate_items"] == sample["g0_candidate_items"] == sample["g1_candidate_items"]
    assert request["raw_response_path"].startswith("results/llm/extension_canonical/g1_primary_ranking/responses/")


def test_g1_strict_parser_invalid_cases_do_not_trigger_scientific_retry() -> None:
    config = g1.load_config(ROOT, CONFIG)
    candidates = [f"item-{index}" for index in range(1, 21)]

    assert config["execution"]["scientific_retry_count"] == 0
    assert g1.normalized_structural_status("parsed_valid", True) == "STRICT_VALID"
    assert g1.normalized_structural_status("invalid_length", True) == "INVALID_LENGTH"
    assert g1.normalized_structural_status("duplicate_indices", True) == "INVALID_DUPLICATE"
    assert g1.normalized_structural_status("unknown_indices", True) == "INVALID_CANDIDATE"
    assert g1.normalized_structural_status("parse_failed", True) == "INVALID_FORMAT"
    assert g1.normalized_structural_status("backend_failure_not_parsed", False) == "BACKEND_FAILURE"
    assert g1.normalized_structural_status("not_executed", False) == "NOT_TERMINAL"

    assert g1.parse_extension_top10_response(
        '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10]}',
        candidate_items=candidates,
    ).parse_valid is True
    assert g1.parse_extension_top10_response(
        '{"ranked_candidate_indices":[1,1,2,3,4,5,6,7,8,9]}',
        candidate_items=candidates,
    ).parse_valid is False
    assert g1.parse_extension_top10_response(
        '{"ranked_candidate_indices":[1,2,3,4,5]}',
        candidate_items=candidates,
    ).parse_valid is False
    assert g1.parse_extension_top10_response(
        '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,21,10]}',
        candidate_items=candidates,
    ).parse_valid is False


def test_g1_fail_strict_utility_keeps_invalid_rows_in_denominator() -> None:
    per_request_stats = pd.DataFrame(
        [
            {
                "canonical_request_id": "REQ-1",
                "dataset": "beauty",
                "model": "qwen2:7b-instruct",
                "prompt_strategy": "BASE",
                "hit_at_5": 1,
                "ndcg_at_5": 1.0,
                "mrr": 1.0,
            },
            {
                "canonical_request_id": "REQ-2",
                "dataset": "beauty",
                "model": "qwen2:7b-instruct",
                "prompt_strategy": "BASE",
                "hit_at_5": 0,
                "ndcg_at_5": 0.0,
                "mrr": 0.0,
            },
        ]
    )
    strict_parsing = pd.DataFrame(
        [
            {
                "canonical_request_id": "REQ-1",
                "strict_parse_valid": "true",
                "terminal_structural_status": "STRICT_VALID",
                "status": "COMPUTED",
            },
            {
                "canonical_request_id": "REQ-2",
                "strict_parse_valid": "false",
                "terminal_structural_status": "INVALID_LENGTH",
                "status": "COMPUTED",
            },
        ]
    )

    utility = g1.build_utility_results(per_request_stats, strict_parsing).set_index("metric")

    assert set(utility.index) == {"hr_at_5", "ndcg_at_5", "mrr"}
    assert utility.loc["hr_at_5", "value"] == "0.5"
    assert utility.loc["ndcg_at_5", "value"] == "0.5"
    assert utility.loc["mrr", "value"] == "0.5"
    assert utility["planned_users"].astype(int).eq(2).all()
    assert utility["strict_valid_users"].astype(int).eq(1).all()
    assert utility["backend_success_users"].astype(int).eq(2).all()
    assert utility["status"].eq("COMPUTED").all()


def test_g1_incomplete_execution_defers_beyond_utility_diagnostics() -> None:
    per_request_stats = pd.DataFrame(
        [
            {
                "canonical_request_id": "REQ-1",
                "dataset": "beauty",
                "model": "qwen2:7b-instruct",
                "prompt_strategy": "BASE",
                "strict_parse_valid": "false",
            }
        ]
    )

    beyond = g1.build_pending_beyond_utility_results(per_request_stats)

    assert len(beyond) == len(g1.BEYOND_METRICS)
    assert beyond["status"].eq("PENDING_TERMINAL_EVIDENCE").all()
    assert beyond["planned_users"].astype(int).eq(1).all()
    assert beyond["undefined_user_count"].astype(int).eq(1).all()


def test_g1_raw_state_rejects_duplicate_interrupted_conflicting_and_g0_rewrites() -> None:
    request_universe = pd.DataFrame(
        [
            {
                "canonical_request_id": "REQ-1",
                "dataset": "beauty",
                "model": "qwen2:7b-instruct",
                "prompt_strategy": "BASE",
                "user_id": "u1",
                "candidate_hash": "c" * 64,
                "prompt_hash": "p" * 64,
            },
            {
                "canonical_request_id": "REQ-G0",
                "dataset": "beauty",
                "model": "qwen2:7b-instruct",
                "prompt_strategy": "BASE",
                "user_id": "u2",
                "candidate_hash": "d" * 64,
                "prompt_hash": "q" * 64,
            },
        ]
    )
    valid_response = {
        "canonical_request_id": "REQ-1",
        "dataset": "beauty",
        "model": "qwen2:7b-instruct",
        "prompt_strategy": "BASE",
        "user_id": "u1",
        "candidate_hash": "c" * 64,
        "prompt_hash": "p" * 64,
    }

    g1.verify_existing_g1_raw_state(request_universe, {"REQ-G0"}, [valid_response], [valid_response])
    with pytest.raises(RuntimeError, match="outside G1 new-call set"):
        g1.verify_existing_g1_raw_state(request_universe, {"REQ-G0"}, [{"canonical_request_id": "REQ-G0"}], [])
    with pytest.raises(RuntimeError, match="invalid_duplicate_evidence"):
        g1.verify_existing_g1_raw_state(request_universe, {"REQ-G0"}, [valid_response], [valid_response, valid_response])
    with pytest.raises(RuntimeError, match="ambiguous_interrupted"):
        g1.verify_existing_g1_raw_state(request_universe, {"REQ-G0"}, [valid_response], [])

    conflict = dict(valid_response, prompt_hash="x" * 64)
    with pytest.raises(RuntimeError, match="conflicting terminal evidence"):
        g1.verify_existing_g1_raw_state(request_universe, {"REQ-G0"}, [valid_response], [conflict])


def test_g1_foodcom_1000_users_retained_with_warm_cold_diagnostics() -> None:
    request_universe = _offline_plan()["request_universe"]
    food = request_universe[request_universe["dataset"].eq("foodcom")].drop_duplicates("user_id")

    assert len(food) == 1000
    assert food["foodcom_history_class"].value_counts().to_dict() == {
        "TRUE_ZERO_HISTORY_USERS": 727,
        "WARM_USERS": 273,
    }
    assert food.loc[food["foodcom_history_class"].eq("TRUE_ZERO_HISTORY_USERS"), "history_length"].astype(int).eq(0).all()
    assert food.loc[food["foodcom_history_class"].eq("WARM_USERS"), "history_length"].astype(int).gt(0).all()
