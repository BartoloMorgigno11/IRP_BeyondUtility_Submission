import json
from pathlib import Path

import pandas as pd
import pytest

from src.llm import canonical_local_llm_runner as runner
from src.llm.canonical_llm_contract import (
    EXECUTION_PARAMETER_COLUMNS,
    contract_hash,
    json_dumps,
    parser_contract_rows,
    prompt_contract_rows,
    sha256_items,
    sha256_text,
)


def _execution_rows() -> list[dict[str, str]]:
    rows = []
    for family, _tag in runner.MODEL_ORDER:
        for task_type, max_tokens in [("ranking", "256"), ("reranking", "512")]:
            row = {
                "execution_contract_id": f"exec_{family.lower().replace(' ', '_')}_{task_type}_v1",
                "model_family": family,
                "task_type": task_type,
                "temperature": "0",
                "top_p": "",
                "top_k": "",
                "seed": "",
                "max_output_tokens": max_tokens,
                "context_limit": "",
                "stop_sequences": "[]",
                "response_format": "json_object",
                "streaming": "false",
                "concurrency": "1",
                "timeout_seconds": "120",
                "infrastructure_retry_count": "0",
                "scientific_retry_count": "0",
                "request_order_policy": "deterministic_dataset_task_user_order",
                "raw_response_retention": "required",
                "parameter_source": "synthetic test",
                "paper_alignment": "adapted_project_policy",
                "contract_status": "ready_adapted",
                "blocking_reason": "",
                "notes": "test",
            }
            row["contract_sha256"] = contract_hash(row)
            rows.append({column: row.get(column, "") for column in EXECUTION_PARAMETER_COLUMNS})
    return rows


def _validation_bundle(cohort: pd.DataFrame | None = None, execution: pd.DataFrame | None = None) -> runner.ContractBundle:
    if cohort is None:
        rows = []
        for index in range(6000):
            rows.append(
                {
                    "eligible_strict_title_primary": "true" if index < 5814 else "false",
                    "primary_prompt_compatible": "true" if index < 5764 else "false",
                }
            )
        cohort = pd.DataFrame(rows)
    if execution is None:
        execution = pd.DataFrame(_execution_rows())
    return runner.ContractBundle(
        prompt_contract=pd.DataFrame(prompt_contract_rows()),
        parser_contract=pd.DataFrame(parser_contract_rows()),
        model_inventory=pd.DataFrame(
            [
                {"model_family": "Qwen2 7B Instruct", "availability_status": "local_model_available_and_tested"},
                {"model_family": "Mistral 7B Instruct", "availability_status": "local_model_available_and_tested"},
            ]
        ),
        execution_contract=execution,
        cohort_preflight=cohort,
        prompt_rendering=pd.DataFrame(),
        policy_options=pd.DataFrame(),
        blocked_or_conditional=pd.DataFrame(),
    )


def test_contract_loading_accepts_valid_style_bundle_and_rejects_hash_or_missing_execution():
    bundle = _validation_bundle()
    runner.validate_contract_bundle(bundle)

    corrupted = _validation_bundle()
    corrupted.prompt_contract.loc[0, "contract_sha256"] = "bad"
    with pytest.raises(ValueError, match="prompt contract hash"):
        runner.validate_contract_bundle(corrupted)

    missing = _validation_bundle(execution=pd.DataFrame(_execution_rows()[:-1]))
    with pytest.raises(ValueError, match="missing ready execution"):
        runner.validate_contract_bundle(missing)

    no_timeout = _validation_bundle()
    no_timeout.execution_contract.loc[0, "timeout_seconds"] = ""
    no_timeout.execution_contract.loc[0, "contract_sha256"] = contract_hash(no_timeout.execution_contract.iloc[0].to_dict())
    with pytest.raises(ValueError, match="missing execution parameter"):
        runner.validate_contract_bundle(no_timeout)


def _synthetic_sample_bundle(monkeypatch: pytest.MonkeyPatch) -> runner.ContractBundle:
    prompt_rows = prompt_contract_rows()
    parser_rows = parser_contract_rows()
    prompt_by_task = {row["task_type"]: row for row in prompt_rows if row["contract_status"] == "ready_adapted"}
    title_map: dict[tuple[str, str], str] = {}
    histories: dict[str, dict[str, list[dict[str, str]]]] = {}
    candidate_lookup: dict[tuple[str, str], list[str]] = {}
    cohort_rows = []
    render_rows = []
    for dataset, task_type in runner.STRATUM_ORDER:
        histories.setdefault(dataset, {})
        prefix = dataset.lower().replace(" ", "_").replace("-", "_")
        for index in range(10):
            user_id = f"{prefix}_{task_type}_u{index:02d}"
            candidate_set_id = f"{prefix}_{task_type}_cs{index:02d}"
            positive_item_id = f"{prefix}_{task_type}_c{index:02d}_00"
            candidate_ids = [f"{prefix}_{task_type}_c{index:02d}_{candidate_index:02d}" for candidate_index in range(20)]
            candidate_lookup[(candidate_set_id, user_id)] = candidate_ids
            for item_id in candidate_ids:
                title_map[(dataset, item_id)] = f"{dataset} {task_type} title {item_id}"
            history_items = []
            for hist_index in range(10):
                item_id = f"{prefix}_{task_type}_h{index:02d}_{hist_index:02d}"
                title_map[(dataset, item_id)] = f"{dataset} history {hist_index}"
                history_items.append({"item_id": item_id, "title": title_map[(dataset, item_id)]})
            histories[dataset][user_id] = history_items
            candidate_items = [{"item_id": item_id, "title": title_map[(dataset, item_id)]} for item_id in candidate_ids]
            system_message, user_message, history_render, candidate_render = runner.render_from_contract(
                prompt_by_task[task_type],
                history_items=history_items,
                candidate_items=candidate_items,
            )
            render_rows.append(
                {
                    "candidate_set_id": candidate_set_id,
                    "user_id": user_id,
                    "system_message_sha256": sha256_text(system_message),
                    "user_message_sha256": sha256_text(user_message),
                    "complete_prompt_sha256": sha256_text(system_message + "\n\n" + user_message),
                    "history_render_sha256": sha256_text(history_render),
                    "candidate_render_sha256": sha256_text(candidate_render),
                    "candidate_order_sha256": sha256_items(candidate_ids),
                }
            )
            cohort_rows.append(
                {
                    "dataset": dataset,
                    "task_type": task_type,
                    "candidate_set_id": candidate_set_id,
                    "source_path": "synthetic.csv",
                    "user_id": user_id,
                    "positive_item_id": positive_item_id,
                    "candidate_count": "20",
                    "candidate_titles_complete": "true",
                    "history_titles_complete": "true",
                    "held_out_positive_absent_from_history": "true",
                    "primary_prompt_compatible": "true",
                    "eligible_strict_title_primary": "true",
                }
            )
    monkeypatch.setattr(runner, "load_prompt_context", lambda _root: (title_map, histories, candidate_lookup))
    return runner.ContractBundle(
        prompt_contract=pd.DataFrame(prompt_rows),
        parser_contract=pd.DataFrame(parser_rows),
        model_inventory=pd.DataFrame(),
        execution_contract=pd.DataFrame(_execution_rows()),
        cohort_preflight=pd.DataFrame(cohort_rows),
        prompt_rendering=pd.DataFrame(render_rows),
        policy_options=pd.DataFrame(),
        blocked_or_conditional=pd.DataFrame(),
    )


def test_sampling_is_deterministic_ten_per_stratum_and_shared_across_models(tmp_path, monkeypatch):
    bundle = _synthetic_sample_bundle(monkeypatch)

    sample, contexts = runner.build_pilot_sample(tmp_path, bundle)
    second, _ = runner.build_pilot_sample(tmp_path, bundle)

    assert sample["pilot_row_id"].tolist() == second["pilot_row_id"].tolist()
    assert sample.groupby(["dataset", "task_type"]).size().eq(10).all()
    assert sample["candidate_count"].eq("20").all()
    assert all("positive_item_id" not in contexts[row]["user_message"].lower() for row in contexts)
    manifest = runner.build_request_manifest(bundle, sample, contexts, {"Qwen2 7B Instruct": "qwenhash", "Mistral 7B Instruct": "mistralhash"})
    assert len(manifest) == 120
    assert manifest.groupby("pilot_row_id")["model_family"].nunique().eq(2).all()
    assert manifest["request_id"].tolist()[0] == "CLLP-REQ-0001"
    assert manifest["request_id"].tolist()[-1] == "CLLP-REQ-0120"


def test_selection_key_and_request_id_are_stable():
    assert runner.selection_key("A", "ranking", "c", "u") == runner.selection_key("A", "ranking", "c", "u")
    assert runner.selection_key("A", "ranking", "c", "u") != runner.selection_key("A", "reranking", "c", "u")
    assert runner.request_id_for_order(7) == "CLLP-REQ-0007"


def test_payload_uses_contract_parameters_non_streaming_and_no_positive_label():
    manifest_row = {
        "model_tag": "qwen2:7b-instruct",
        "temperature": "0",
        "max_output_tokens": "256",
        "seed": "42",
    }
    context = {"system_message": "System", "user_message": "Rank these candidates only."}

    payload = runner.build_payload(manifest_row, context)

    assert payload["stream"] is False
    assert payload["format"] == "json"
    assert payload["options"]["temperature"] == 0.0
    assert payload["options"]["num_predict"] == 256
    assert payload["options"]["seed"] == 42
    assert "positive" not in json_dumps(payload).lower()


def test_strict_parser_accepts_valid_ranking_and_reranking_outputs():
    candidates = [f"item{i}" for i in range(1, 21)]

    ranking = runner.classify_response(
        '{"ranked_candidate_indices":[1,2,3,4,5]}',
        candidate_items=candidates,
        expected_count=5,
        require_full_permutation=False,
    )
    reranking = runner.classify_response(
        '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}',
        candidate_items=candidates,
        expected_count=20,
        require_full_permutation=True,
    )

    assert ranking["strict_parse_valid"] is True
    assert ranking["strict_indices"] == [1, 2, 3, 4, 5]
    assert reranking["strict_parse_valid"] is True
    assert reranking["full_permutation_valid"] is True


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        ("not json", "json_parse_failed"),
        ('Answer: {"ranked_candidate_indices":[1,2,3,4,5]}', "json_parse_failed"),
        ('```json\n{"ranked_candidate_indices":[1,2,3,4,5]}\n```', "json_parse_failed"),
        ('{"ranked_candidate_indices":[1,1,2,3,4]}', "duplicate_indices"),
        ('{"ranked_candidate_indices":[1,2,3,4,21]}', "index_range_invalid"),
        ('{"ranked_candidate_indices":[0,1,2,3,4]}', "index_range_invalid"),
        ('{"items":[1,2,3,4,5]}', "schema_invalid"),
        ('{"ranked_candidate_indices":[1,2,3,4]}', "output_count_invalid"),
        ('{"ranked_candidate_indices":[1,2,3,4,5,6]}', "output_count_invalid"),
    ],
)
def test_strict_parser_rejects_malformed_and_invalid_ranking_outputs(response, reason):
    result = runner.classify_response(
        response,
        candidate_items=[f"item{i}" for i in range(1, 21)],
        expected_count=5,
        require_full_permutation=False,
    )

    assert result["strict_parse_valid"] is False
    assert result["primary_failure_reason"] == reason


def test_strict_parser_rejects_partial_and_repeated_reranking_outputs_without_positive_repair():
    candidates = [f"item{i}" for i in range(1, 21)]
    partial = runner.classify_response(
        '{"ranked_candidate_indices":[1,2,3,4,5]}',
        candidate_items=candidates,
        expected_count=20,
        require_full_permutation=True,
    )
    repeated = runner.classify_response(
        '{"positive_item_id":"item1","ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,19]}',
        candidate_items=candidates,
        expected_count=20,
        require_full_permutation=True,
    )

    assert partial["strict_parse_valid"] is False
    assert partial["primary_failure_reason"] == "output_count_invalid"
    assert repeated["strict_parse_valid"] is False
    assert repeated["primary_failure_reason"] == "schema_invalid"
    assert repeated["strict_indices"] == []


def test_salvage_records_transformations_but_does_not_accept_fuzzy_titles():
    candidates = [f"item{i}" for i in range(1, 21)]
    salvaged = runner.classify_response(
        'Answer:\n```json\n{"ranked_candidate_indices":["1","2","3","4","5"]}\n```',
        candidate_items=candidates,
        expected_count=5,
        require_full_permutation=False,
    )
    fuzzy = runner.classify_response(
        '{"ranked_candidate_indices":["Some title","Other title","Third","Fourth","Fifth"]}',
        candidate_items=candidates,
        expected_count=5,
        require_full_permutation=False,
    )

    assert salvaged["strict_parse_valid"] is False
    assert salvaged["diagnostic_salvage_valid"] is True
    assert set(salvaged["diagnostic_transformations"]) == {"extracted_json_object", "converted_integer_strings"}
    assert fuzzy["diagnostic_salvage_valid"] is False


def test_append_jsonl_is_utf8_append_only_and_rejects_duplicate_request_ids(tmp_path):
    path = tmp_path / "records.jsonl"
    seen: set[str] = set()

    runner.append_jsonl(path, {"request_id": "r1", "text": "café"}, seen)
    runner.append_jsonl(path, {"request_id": "r2", "text": "ok"}, seen)

    assert path.read_text(encoding="utf-8").count("\n") == 2
    with pytest.raises(ValueError, match="duplicate request_id"):
        runner.append_jsonl(path, {"request_id": "r1"}, seen)


def _manifest_execution_parse_ranking_frames(strict_rate: float = 1.0, candidate_valid: bool = True, rerank_perm: bool = True):
    manifest_rows = []
    execution_rows = []
    parse_rows = []
    ranking_rows = []
    valid_rows = int(120 * strict_rate)
    for index in range(120):
        request_id = runner.request_id_for_order(index + 1)
        task_type = "reranking" if index % 2 else "ranking"
        family = "Qwen2 7B Instruct" if index < 60 else "Mistral 7B Instruct"
        manifest_rows.append(
            {
                "request_id": request_id,
                "complete_prompt_sha256": "a" * 64,
                "candidate_order_sha256": "b" * 64,
            }
        )
        execution_rows.append({"request_id": request_id, "backend_success": "true", "done_reason": ""})
        strict_valid = index < valid_rows
        parse_rows.append(
            {
                "request_id": request_id,
                "task_type": task_type,
                "strict_parse_valid": "true" if strict_valid else "false",
                "candidate_validity": "true" if candidate_valid else "false",
                "duplicate_count": "0",
                "full_permutation_valid": "true" if (task_type == "ranking" or rerank_perm) else "false",
            }
        )
        ranking_rows.append({"request_id": request_id})
    metric_rows = []
    for family in ["Qwen2 7B Instruct", "Mistral 7B Instruct"]:
        for dataset, task in runner.STRATUM_ORDER:
            metric_rows.append(
                {
                    "model_family": family,
                    "dataset": dataset,
                    "task_type": task,
                    "strict_parse_validity_rate": str(strict_rate),
                }
            )
    return pd.DataFrame(manifest_rows), pd.DataFrame(execution_rows), pd.DataFrame(parse_rows), pd.DataFrame(ranking_rows), pd.DataFrame(metric_rows)


def _write_jsonl_records(tmp_path: Path, request_count: int = 120, response_count: int = 120) -> None:
    path = tmp_path / runner.PILOT_DIR
    path.mkdir(parents=True, exist_ok=True)
    (tmp_path / runner.REQUEST_JSONL).write_text("".join(json.dumps({"request_id": f"r{i}"}) + "\n" for i in range(request_count)), encoding="utf-8")
    (tmp_path / runner.RESPONSE_JSONL).write_text("".join(json.dumps({"request_id": f"r{i}"}) + "\n" for i in range(response_count)), encoding="utf-8")


def test_decision_approval_and_rejection_gates(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_jsonl_records(tmp_path)
    manifest, execution, parse_results, rankings, metrics = _manifest_execution_parse_ranking_frames()

    approved = runner.build_decision(manifest, execution, parse_results, rankings, metrics, True)

    assert approved.iloc[0]["pilot_decision"] == "approve_full_local_execution"
    assert approved.iloc[0]["readiness_threshold"] == "0.80000000000000004"
    assert approved.iloc[0]["recommended_single_next_task"].startswith("Phase 5B-P2")

    below_threshold = runner.build_decision(*_manifest_execution_parse_ranking_frames(strict_rate=0.79), protected_unchanged=True)
    assert below_threshold.iloc[0]["pilot_decision"] == "require_prompt_contract_revision"

    candidate_mutation = runner.build_decision(*_manifest_execution_parse_ranking_frames(candidate_valid=False), protected_unchanged=True)
    assert candidate_mutation.iloc[0]["pilot_decision"] == "require_parser_contract_revision"

    incomplete_reranking = runner.build_decision(*_manifest_execution_parse_ranking_frames(rerank_perm=False), protected_unchanged=True)
    assert incomplete_reranking.iloc[0]["pilot_decision"] == "require_parser_contract_revision"

    _write_jsonl_records(tmp_path, request_count=120, response_count=119)
    missing_response = runner.build_decision(manifest, execution, parse_results, rankings, metrics, True)
    assert missing_response.iloc[0]["pilot_decision"] == "pilot_incomplete"


def test_metrics_are_valid_only_and_do_not_fabricate_invalid_as_zero():
    manifest = pd.DataFrame(
        [
            {"request_id": "r1", "model_family": "Qwen2 7B Instruct", "model_tag": "qwen2:7b-instruct", "dataset": "ML-1M", "task_type": "ranking"},
            {"request_id": "r2", "model_family": "Qwen2 7B Instruct", "model_tag": "qwen2:7b-instruct", "dataset": "ML-1M", "task_type": "ranking"},
        ]
    )
    execution = pd.DataFrame(
        [
            {"request_id": "r1", "model_family": "Qwen2 7B Instruct", "model_tag": "qwen2:7b-instruct", "dataset": "ML-1M", "task_type": "ranking", "backend_success": "true", "duration_seconds": "1", "eval_count": "5"},
            {"request_id": "r2", "model_family": "Qwen2 7B Instruct", "model_tag": "qwen2:7b-instruct", "dataset": "ML-1M", "task_type": "ranking", "backend_success": "true", "duration_seconds": "2", "eval_count": "9"},
        ]
    )
    parse_results = pd.DataFrame(
        [
            {"request_id": "r1", "model_family": "Qwen2 7B Instruct", "dataset": "ML-1M", "task_type": "ranking", "strict_parse_valid": "true", "diagnostic_salvage_valid": "true", "candidate_validity": "true", "duplicate_count": "0", "hallucination_count": "0", "full_permutation_valid": "true"},
            {"request_id": "r2", "model_family": "Qwen2 7B Instruct", "dataset": "ML-1M", "task_type": "ranking", "strict_parse_valid": "false", "diagnostic_salvage_valid": "false", "candidate_validity": "false", "duplicate_count": "0", "hallucination_count": "0", "full_permutation_valid": "true"},
        ]
    )
    rankings = pd.DataFrame(
        [
            {"request_id": "r1", "positive_in_top_5": "true", "ndcg_at_5": "1", "positive_rank_zero_based": "0", "utility_evaluation_valid": "true"},
            {"request_id": "r2", "positive_in_top_5": "", "ndcg_at_5": "", "positive_rank_zero_based": "", "utility_evaluation_valid": "false"},
        ]
    )

    metrics = runner.build_metrics(manifest, execution, parse_results, rankings)

    assert metrics.iloc[0]["utility_evaluated_rows"] == "1"
    assert metrics.iloc[0]["hr_at_5_valid_only"] == "1"
    assert metrics.iloc[0]["ndcg_at_5_valid_only"] == "1"
    assert metrics.iloc[0]["strict_parse_validity_rate"] == "0.5"
