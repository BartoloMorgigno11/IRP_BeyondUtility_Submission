from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis import finalization_f6d4a_lastfm_llm_ranking as f6d4a


def _manifest_row(model: str, user_id: str, request_id: str, positive_position: int = 3, history_count: int = 10) -> dict[str, str]:
    history_ids = [f"h{idx}" for idx in range(1, history_count + 1)]
    history_names = [f"History {idx}" for idx in range(1, history_count + 1)]
    candidate_ids = [f"i{idx}" for idx in range(1, 21)]
    candidate_names = [f"Item {idx}" for idx in range(1, 21)]
    positive = candidate_ids[positive_position - 1]
    contract_id = (
        "F6D3-QWEN2-RANKING-llm_rank_index_json_top5_v1"
        if model.startswith("qwen2")
        else "F6D3-MISTRAL-RANKING-llm_rank_index_json_top5_v1"
    )
    history_text = "|".join(history_ids[-10:])
    row = {
        "request_id": request_id,
        "task": "ranking",
        "model": model,
        "user_id": user_id,
        "cohort_name": "finalization_f6c_lastfm_ranking_cohort_full_1859",
        "history_item_ids": history_text,
        "history_item_names": "|".join(history_names[-10:]),
        "candidate_item_ids": "|".join(candidate_ids),
        "candidate_item_names": "|".join(candidate_names),
        "positive_item_id": positive,
        "positive_candidate_position": str(positive_position),
        "positive_present": "true",
        "prompt_contract_id": contract_id,
        "prompt_sha256": "",
        "expected_output_count": "5",
        "request_status": "planned_not_executed",
        "blocking_reason": "",
        "notes": "synthetic",
    }
    row["prompt_sha256"] = f6d4a.prompt_digest(
        "ranking",
        model,
        user_id,
        row["history_item_ids"],
        row["candidate_item_ids"],
        contract_id,
    )
    return row


def _phase_inputs(manifest: pd.DataFrame) -> f6d4a.PhaseInputs:
    prompt = pd.DataFrame(
        [
            {"model": "qwen2:7b-instruct", "task": "ranking", "prompt_contract_id": "F6D3-QWEN2-RANKING-llm_rank_index_json_top5_v1"},
            {"model": "mistral:7b-instruct", "task": "ranking", "prompt_contract_id": "F6D3-MISTRAL-RANKING-llm_rank_index_json_top5_v1"},
        ]
    )
    go = pd.DataFrame(
        [
            {"model": "qwen2:7b-instruct", "task": "ranking", "decision": "go_adapted"},
            {"model": "mistral:7b-instruct", "task": "ranking", "decision": "go_adapted"},
        ]
    )
    f6d1 = pd.DataFrame(
        [
            {"model": "random_seeded", "metric": "HR@5", "value": "0.25", "paper_equivalence": "not_paper", "reproduction_class": "local_adapted"},
            {"model": "random_seeded", "metric": "NDCG@5", "value": "0.15", "paper_equivalence": "not_paper", "reproduction_class": "local_adapted"},
            {"model": "mostpop_train_only", "metric": "HR@5", "value": "0.75", "paper_equivalence": "not_exact", "reproduction_class": "local_adapted"},
            {"model": "mostpop_train_only", "metric": "NDCG@5", "value": "0.65", "paper_equivalence": "not_exact", "reproduction_class": "local_adapted"},
        ]
    )
    f6d2 = pd.DataFrame(
        [
            {"model": model, "evaluation_scope": "candidate_set_20", "metric": "HR@5", "mean": "0.6"}
            for model in ["bprmf", "gru4rec", "sasrec", "lightgcn"]
        ]
        + [
            {"model": model, "evaluation_scope": "candidate_set_20", "metric": "NDCG@5", "mean": "0.5"}
            for model in ["bprmf", "gru4rec", "sasrec", "lightgcn"]
        ]
    )
    missing = pd.DataFrame(
        [
            {"resolution_category": "active_gap", "current_status": "still_missing"},
            {"resolution_category": "non_active", "current_status": "resolved"},
        ]
    )
    coverage = pd.DataFrame([{"coverage_family": "exact numeric replication", "numerator": "0", "denominator": "187"}])
    atomic = pd.DataFrame(
        [
            {
                "atomic_cell_id": "ATOM-Q-HR",
                "dataset": "LastFM",
                "task": "ranking",
                "model": "Qwen2",
                "metric": "hr_at_5",
                "paper_value_numeric": "0.4",
            },
            {
                "atomic_cell_id": "ATOM-Q-NDCG",
                "dataset": "LastFM",
                "task": "ranking",
                "model": "Qwen2",
                "metric": "ndcg_at_5",
                "paper_value_numeric": "0.3",
            },
        ]
    )
    cohort = pd.DataFrame(
        [{"user_id": row.user_id, "train_history_count": str(5 + idx * 20)} for idx, row in enumerate(manifest.itertuples(index=False))]
    )
    return f6d4a.PhaseInputs(
        manifest=manifest,
        prompt_contract=prompt,
        go_no_go=go,
        execution_plan=pd.DataFrame(),
        cohort=cohort,
        f6d1_metrics=f6d1,
        f6d2_aggregate=f6d2,
        f6b_missing=missing,
        f6b_coverage=coverage,
        atomic_cells=atomic,
        manifest_sha256="manifest-sha",
        input_hashes={},
    )


def _raw(row: dict[str, str], response: str, success: bool = True) -> dict[str, str]:
    return {
        "request_id": row["request_id"],
        "model": row["model"],
        "task": "ranking",
        "user_id": row["user_id"],
        "prompt_sha256": row["prompt_sha256"],
        "manifest_sha256": "manifest-sha",
        "attempt": 1,
        "backend": "ollama_local_loopback",
        "local_model_identifier": row["model"],
        "raw_response": response,
        "response_sha256": f6d4a.sha256_text(response),
        "transport_status": "success" if success else "failed",
        "completion_status": "completed" if success else "transport_failed",
        "error_type": "" if success else "URLError",
        "error_message": "" if success else "connection refused",
        "is_final_attempt": "true",
        "notes": "synthetic",
    }


def _manifest() -> pd.DataFrame:
    rows = [
        _manifest_row("qwen2:7b-instruct", "u1", "REQ-Q-1", positive_position=3),
        _manifest_row("qwen2:7b-instruct", "u2", "REQ-Q-2", positive_position=10),
        _manifest_row("mistral:7b-instruct", "u1", "REQ-M-1", positive_position=1),
        _manifest_row("mistral:7b-instruct", "u2", "REQ-M-2", positive_position=8),
    ]
    return pd.DataFrame(rows)


def test_manifest_prompt_sha_and_context_rendering() -> None:
    row = _manifest_row("qwen2:7b-instruct", "u1", "REQ-Q-1")
    context = f6d4a.context_from_manifest_row(row)
    assert context["prompt_sha256"] == row["prompt_sha256"]
    assert "Return exactly 5 unique candidate indices" in context["user_message"]
    assert len(context["candidate_items"]) == 20


def test_strict_parser_accepts_json_and_maps_items() -> None:
    row = _manifest_row("qwen2:7b-instruct", "u1", "REQ-Q-1", positive_position=3)
    parsed = f6d4a.classify_response(row, _raw(row, '{"ranked_candidate_indices":[3,1,2,4,5]}'))
    assert parsed["strict_valid"] == "true"
    assert parsed["ranked_item_ids"].split("|")[0] == "i3"
    assert parsed["positive_rank"] == "1"
    assert parsed["positive_in_top5"] == "true"


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ('{"ranked_candidate_indices":[1,2,3,4,4]}', "duplicate_indices"),
        ('{"ranked_candidate_indices":[1,2,3,4,21]}', "unknown_indices"),
        ('{"ranked_candidate_indices":[1,2,3,4]}', "invalid_length"),
        ("not-json", "parse_failed"),
    ],
)
def test_invalid_outputs_are_not_salvaged(payload: str, reason: str) -> None:
    row = _manifest_row("qwen2:7b-instruct", "u1", "REQ-Q-1")
    parsed = f6d4a.classify_response(row, _raw(row, payload))
    assert parsed["strict_valid"] == "false"
    assert parsed["failure_reason"] == reason
    assert parsed["ranked_item_ids"] == ""


def test_all_request_and_strict_valid_metrics_are_separate() -> None:
    manifest = _manifest()
    inputs = _phase_inputs(manifest)
    rows = [
        f6d4a.classify_response(manifest.iloc[0].to_dict(), _raw(manifest.iloc[0].to_dict(), '{"ranked_candidate_indices":[3,1,2,4,5]}')),
        f6d4a.classify_response(manifest.iloc[1].to_dict(), _raw(manifest.iloc[1].to_dict(), '{"ranked_candidate_indices":[1,2,3,4,5]}')),
    ]
    validation = f6d4a.frame(rows, f6d4a.RESPONSE_VALIDATION_COLUMNS)
    metrics = f6d4a.build_utility_metrics(inputs, validation)
    assert set(metrics["evaluation_population"]) == {"all_request", "strict_valid"}
    assert metrics.loc[(metrics["metric"].eq("HR@5")) & (metrics["evaluation_population"].eq("all_request")), "value"].iloc[0] == "0.5"


def test_position_bias_and_history_grouping() -> None:
    manifest = _manifest()
    inputs = _phase_inputs(manifest)
    validation = f6d4a.frame(
        [
            f6d4a.classify_response(row, _raw(row, '{"ranked_candidate_indices":[1,2,3,4,5]}'))
            for row in manifest.to_dict("records")
        ],
        f6d4a.RESPONSE_VALIDATION_COLUMNS,
    )
    bias = f6d4a.build_position_bias(inputs, validation)
    history = f6d4a.build_history_sensitivity(inputs, validation)
    assert len(bias) == 40
    assert set(history["policy"]) == {"adapted_train_history_bins"}
    assert history["users"].astype(int).sum() == 8


def test_baseline_comparison_marks_trainable_f6d2_as_configuration_limited() -> None:
    manifest = _manifest()
    inputs = _phase_inputs(manifest)
    validation = f6d4a.frame(
        [f6d4a.classify_response(row, _raw(row, '{"ranked_candidate_indices":[1,2,3,4,5]}')) for row in manifest.to_dict("records")],
        f6d4a.RESPONSE_VALIDATION_COLUMNS,
    )
    utility = f6d4a.build_utility_metrics(inputs, validation)
    comparison = f6d4a.build_baseline_comparison(inputs, utility)
    limited = comparison.loc[comparison["model"].isin(["bprmf", "gru4rec", "sasrec", "lightgcn"])]
    assert set(limited["model_group"]) == {"configuration_limited_diagnostic"}


def test_paper_alignment_keeps_exact_replication_false() -> None:
    manifest = _manifest()
    inputs = _phase_inputs(manifest)
    validation = f6d4a.frame(
        [f6d4a.classify_response(row, _raw(row, '{"ranked_candidate_indices":[1,2,3,4,5]}')) for row in manifest.to_dict("records")],
        f6d4a.RESPONSE_VALIDATION_COLUMNS,
    )
    utility = f6d4a.build_utility_metrics(inputs, validation)
    alignment = f6d4a.build_paper_alignment(inputs, utility)
    assert alignment["is_exact_numeric_replication"].eq("false").all()
    assert alignment.loc[alignment["local_model_identifier"].eq("qwen2:7b-instruct"), "source_atomic_cell"].ne("").any()


def test_missing_and_coverage_do_not_increase_exact_coverage() -> None:
    manifest = _manifest()
    inputs = _phase_inputs(manifest)
    validation = f6d4a.frame(
        [f6d4a.classify_response(row, _raw(row, '{"ranked_candidate_indices":[1,2,3,4,5]}')) for row in manifest.to_dict("records")],
        f6d4a.RESPONSE_VALIDATION_COLUMNS,
    )
    utility = f6d4a.build_utility_metrics(inputs, validation)
    _readiness, missing, coverage = f6d4a.build_readiness_missing_coverage(inputs, validation, utility)
    assert missing["active_before"].iloc[0] == "1"
    exact = coverage.loc[coverage["coverage_family"].eq("protocol-exact numeric replication")].iloc[0]
    assert exact.after_f6d4a == "0/187"


def test_resume_skips_completed_requests_and_uses_mock_executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f6d4a, "EXPECTED_REQUESTS_PER_MODEL", 2)
    manifest = _manifest()
    inputs = _phase_inputs(manifest)
    raw_path = tmp_path / f6d4a.raw_path_for_model("qwen2:7b-instruct")
    raw_path.parent.mkdir(parents=True)
    f6d4a.append_jsonl_fsync(raw_path, _raw(manifest.iloc[0].to_dict(), '{"ranked_candidate_indices":[3,1,2,4,5]}'))
    state = f6d4a.OllamaState(True, "ollama", "ollama version test", True, "test", {"qwen2:7b-instruct": "digest", "mistral:7b-instruct": "digest"}, "", "gpu-test")
    calls: list[str] = []

    def fake_executor(payload, timeout):
        calls.append(payload["model"])
        return f6d4a.RequestResult(True, '{"ranked_candidate_indices":[1,2,3,4,5]}', {"done": True}, "", "", "t0", "t1", 1.0)

    sent = f6d4a.execute_missing_requests(tmp_path, inputs, state, max_new_requests=1, request_executor=fake_executor)
    assert sent == 1
    assert calls == ["qwen2:7b-instruct"]
    assert len(f6d4a.read_jsonl(raw_path)) == 2


def test_duplicate_and_conflicting_raw_are_flagged(tmp_path: Path) -> None:
    manifest = _manifest()
    inputs = _phase_inputs(manifest)
    raw_path = tmp_path / f6d4a.raw_path_for_model("qwen2:7b-instruct")
    raw_path.parent.mkdir(parents=True)
    first = _raw(manifest.iloc[0].to_dict(), '{"ranked_candidate_indices":[1,2,3,4,5]}')
    second = _raw(manifest.iloc[0].to_dict(), '{"ranked_candidate_indices":[2,3,4,5,6]}')
    f6d4a.append_jsonl_fsync(raw_path, first)
    f6d4a.append_jsonl_fsync(raw_path, second)
    validation = f6d4a.build_response_validation(tmp_path, inputs)
    audit = f6d4a.build_request_audit(tmp_path, inputs, validation)
    row = audit.loc[audit["request_id"].eq("REQ-Q-1")].iloc[0]
    assert row.duplicate_request == "true"
    assert row.conflicting_response == "true"


def test_deterministic_analysis_rebuild_and_relative_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f6d4a, "EXPECTED_REQUESTS_PER_MODEL", 2)
    monkeypatch.setattr(f6d4a, "EXPECTED_TOTAL_REQUESTS", 4)
    monkeypatch.setattr(
        f6d4a,
        "inspect_ollama",
        lambda: f6d4a.OllamaState(True, "ollama", "ollama version test", True, "test", {"qwen2:7b-instruct": "digest", "mistral:7b-instruct": "digest"}, "", "gpu-test"),
    )
    monkeypatch.setattr(
        f6d4a,
        "git_output",
        lambda _root, args: f6d4a.EXPECTED_BRANCH if "branch" in args else f6d4a.EXPECTED_HEAD,
    )
    manifest = _manifest()
    _write_fixture_inputs(tmp_path, manifest)
    for row in manifest.to_dict("records"):
        f6d4a.append_jsonl_fsync(
            tmp_path / f6d4a.raw_path_for_model(row["model"]),
            _raw(row, '{"ranked_candidate_indices":[1,2,3,4,5]}'),
        )
    first = f6d4a.build_all_analysis_frames(tmp_path)
    second = f6d4a.build_all_analysis_frames(tmp_path)
    first_hashes = {key: f6d4a.frame_sha256(value) for key, value in first.items()}
    second_hashes = {key: f6d4a.frame_sha256(value) for key, value in second.items()}
    assert first_hashes == second_hashes
    assert all(f6d4a.no_absolute_paths(value) for value in first.values())
    assert first["validation"]["status"].eq("pass").all()


def test_analysis_only_uses_offline_state_and_preserves_raw_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f6d4a, "EXPECTED_REQUESTS_PER_MODEL", 2)
    monkeypatch.setattr(f6d4a, "EXPECTED_TOTAL_REQUESTS", 4)
    monkeypatch.setattr(
        f6d4a,
        "inspect_ollama",
        lambda: pytest.fail("analysis-only must not inspect Ollama"),
    )
    monkeypatch.setattr(
        f6d4a.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("analysis-only must not open HTTP connections"),
    )
    monkeypatch.setattr(
        f6d4a,
        "git_output",
        lambda _root, args: f6d4a.EXPECTED_BRANCH if "branch" in args else f6d4a.EXPECTED_HEAD,
    )
    manifest = _manifest()
    _write_fixture_inputs(tmp_path, manifest)
    responses = {
        "REQ-Q-1": '{"ranked_candidate_indices":[1,2,3,4,5]}',
        "REQ-Q-2": '{"ranked_candidate_indices":[1,2,3,4,5]}',
        "REQ-M-1": '{"ranked_candidate_indices":[1,2,3,4,5]}',
        "REQ-M-2": '{"ranked_candidate_indices":[1,2,3,4,21]}',
    }
    raw_paths = set()
    for row in manifest.to_dict("records"):
        path = tmp_path / f6d4a.raw_path_for_model(row["model"])
        raw_paths.add(path)
        f6d4a.append_jsonl_fsync(path, _raw(row, responses[row["request_id"]]))
    before = {path: (f6d4a.file_sha256(path), path.stat().st_mtime_ns) for path in raw_paths}

    result = f6d4a.run_phase(tmp_path, analysis_only=True)

    after = {path: (f6d4a.file_sha256(path), path.stat().st_mtime_ns) for path in raw_paths}
    assert result["new_requests_sent"] == 0
    assert result["ollama_state"] == f6d4a.offline_ollama_state()
    assert before == after
    response = result["frames"]["response_validation"]
    counts = response.groupby("model")["strict_valid"].apply(lambda values: int(values.eq("true").sum())).to_dict()
    assert counts == {"mistral:7b-instruct": 1, "qwen2:7b-instruct": 2}
    totals = response.groupby("model").size().to_dict()
    assert totals == {"mistral:7b-instruct": 2, "qwen2:7b-instruct": 2}
    preflight = result["frames"]["model_preflight"]
    assert set(preflight["backend_reachable"]) == {"false"}
    assert set(preflight["backend_version"]) == {"not_queried_analysis_only"}


def test_analysis_only_missing_raw_fails_without_creating_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f6d4a, "EXPECTED_REQUESTS_PER_MODEL", 2)
    _write_fixture_inputs(tmp_path, _manifest())

    with pytest.raises(FileNotFoundError, match="analysis_only_missing_raw_response"):
        f6d4a.run_phase(tmp_path, analysis_only=True)

    for relative in f6d4a.RAW_PATHS.values():
        assert not (tmp_path / relative).exists()


def _write_fixture_inputs(root: Path, manifest: pd.DataFrame) -> None:
    inputs = _phase_inputs(manifest)
    paths = {
        f6d4a.F6D3_INPUTS["ranking_manifest"]: manifest,
        f6d4a.F6D3_INPUTS["prompt_contract"]: inputs.prompt_contract,
        f6d4a.F6D3_INPUTS["go_no_go"]: inputs.go_no_go,
        f6d4a.F6D3_INPUTS["execution_plan"]: pd.DataFrame([{"task": "ranking"}]),
        f6d4a.F6C_INPUTS["ranking_cohort"]: inputs.cohort,
        f6d4a.F6D1_INPUTS["utility_metrics"]: inputs.f6d1_metrics,
        f6d4a.F6D2_INPUTS["aggregate_metrics"]: inputs.f6d2_aggregate,
        f6d4a.F6B_INPUTS["missing_evidence"]: inputs.f6b_missing,
        f6d4a.F6B_INPUTS["coverage_dashboard"]: inputs.f6b_coverage,
        f6d4a.F5AF_INPUTS["atomic_cells"]: inputs.atomic_cells,
    }
    for relative, frame in paths.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, lineterminator="\n")
    for relative in [
        f6d4a.F6D3_INPUTS["local_llm_inventory"],
        f6d4a.F6D3_INPUTS["cohort_policy"],
        f6d4a.F6D3_INPUTS["validation"],
        f6d4a.F6C_INPUTS["catalog"],
        f6d4a.F6C_INPUTS["train"],
        f6d4a.F6C_INPUTS["ranking_candidates"],
        f6d4a.F6D1_INPUTS["baseline_rankings"],
        f6d4a.F6D2_INPUTS["execution_manifest"],
        f6d4a.F6D2_INPUTS["candidate_rankings"],
        f6d4a.F6B_INPUTS["model_prompt"],
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"placeholder": "ok"}]).to_csv(path, index=False, lineterminator="\n")


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_real_outputs_schema_when_materialized() -> None:
    root = Path(__file__).resolve().parents[1]
    validation_path = root / f6d4a.OUTPUT_TABLES["validation"]
    if not validation_path.exists():
        return
    validation = pd.read_csv(validation_path, dtype=str, keep_default_na=False)
    assert set(validation["status"]).issubset({"pass", "fail"})
    assert (root / f6d4a.PREDICTIONS_PATH).exists()


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_real_frozen_raw_counts_and_strict_invalid_records_when_materialized() -> None:
    root = Path(__file__).resolve().parents[1]
    qwen_raw = root / f6d4a.raw_path_for_model("qwen2:7b-instruct")
    mistral_raw = root / f6d4a.raw_path_for_model("mistral:7b-instruct")
    validation_path = root / f6d4a.OUTPUT_TABLES["response_validation"]
    if not qwen_raw.exists() or not mistral_raw.exists() or not validation_path.exists():
        return
    assert f6d4a.file_sha256(qwen_raw) == "1384a47058ed8dd8c06fc52d6bc425020b1746b60ad11e4bde6153f948a493b4"
    assert f6d4a.file_sha256(mistral_raw) == "7031f37d40092b4f318b5e03273772531ffda37cef0c38bfed32edfef935736a"
    assert len(f6d4a.read_jsonl(qwen_raw)) == 1859
    assert len(f6d4a.read_jsonl(mistral_raw)) == 1859
    validation = pd.read_csv(validation_path, dtype=str, keep_default_na=False)
    strict = validation.groupby("model")["strict_valid"].apply(lambda values: int(values.eq("true").sum())).to_dict()
    assert strict["qwen2:7b-instruct"] == 1859
    assert strict["mistral:7b-instruct"] == 1857
    invalid = validation.loc[validation["model"].eq("mistral:7b-instruct") & validation["strict_valid"].eq("false")]
    assert set(invalid["request_id"]) == {"F6D3-REQ-RANK-50a9669640c3", "F6D3-REQ-RANK-bad258ab8b57"}
