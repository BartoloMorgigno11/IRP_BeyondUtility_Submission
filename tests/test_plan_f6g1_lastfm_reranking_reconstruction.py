from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.analysis import plan_f6g1_lastfm_reranking_reconstruction as f6g1


MODELS = ["bprmf", "gru4rec", "sasrec", "lightgcn"]
SEEDS = {"bprmf": 45, "gru4rec": 46, "sasrec": 44, "lightgcn": 46}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(root: Path, rel: str, rows: list[dict[str, object]]) -> dict[str, str]:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")
    return {"path": rel, "sha256": _sha(path), "phase": "test"}


def _write_json(root: Path, rel: str, payload: dict[str, object]) -> dict[str, str]:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f6g1.stable_json(payload) + "\n", encoding="utf-8")
    return {"path": rel, "sha256": _sha(path), "phase": "test"}


def _full_catalog_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_user_model = {
        ("1", "bprmf"): ["10", "20", "30", "40", "50", "60"],
        ("1", "gru4rec"): ["10", "20", "70", "40", "50", "60"],
        ("1", "sasrec"): ["30", "20", "40", "70", "50", "60"],
        ("1", "lightgcn"): ["40", "20", "30", "70", "50", "60"],
        ("2", "bprmf"): ["11", "21", "31", "41", "51", "61"],
        ("2", "gru4rec"): ["21", "11", "31", "41", "51", "61"],
        ("2", "sasrec"): ["31", "21", "11", "41", "51", "61"],
        ("2", "lightgcn"): ["41", "21", "31", "11", "51", "61"],
    }
    for (user, model), items in by_user_model.items():
        for rank, item in enumerate(items, start=1):
            rows.append(
                {
                    "execution_id": f"run-{model}",
                    "model": model,
                    "seed": SEEDS[model],
                    "user_id": user,
                    "item_id": item,
                    "rank": rank,
                    "score": f"{1.0 / rank:.6f}",
                    "is_positive": str(user == "1" and item == "40").lower(),
                    "was_seen_in_train": "false",
                    "was_seen_in_validation": "false",
                    "eligible_for_recommendation": "true",
                    "catalog_name": f"artist {item}",
                    "notes": "fixture",
                }
            )
    return rows


def _candidate_prediction_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for seed in [42, SEEDS[model]]:
            for user in ["1", "2"]:
                for rank, item in enumerate(["10", "20", "30", "40"], start=1):
                    rows.append(
                        {
                            "execution_id": f"run-{model}-{seed}",
                            "model": model,
                            "seed": seed,
                            "user_id": user,
                            "candidate_item_id": item,
                            "rank": rank,
                            "score": "0.1",
                            "is_positive": str(user == "1" and item == "40").lower(),
                            "candidate_original_position": rank,
                            "cohort_name": "cohort",
                            "candidate_set_name": "candidates",
                            "source_candidate_file": "source.csv",
                            "notes": "fixture",
                        }
                    )
    return rows


def _previous_pool_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current = {
        "1": [("10", "bprmf;gru4rec", "1;1"), ("20", "bprmf;gru4rec;sasrec;lightgcn", "2;2;2;2"), ("30", "bprmf;sasrec;lightgcn", "3;1;3"), ("40", "bprmf;gru4rec;sasrec;lightgcn", "4;4;3;1")],
        "2": [("11", "bprmf;gru4rec;sasrec;lightgcn", "1;2;3;4"), ("21", "bprmf;gru4rec;sasrec;lightgcn", "2;1;2;2"), ("31", "bprmf;gru4rec;sasrec;lightgcn", "3;3;1;3"), ("41", "bprmf;gru4rec;sasrec;lightgcn", "4;4;4;1")],
    }
    for user, items in current.items():
        positive = "40" if user == "1" else "99"
        for position, (item, models, ranks) in enumerate(items, start=1):
            model_list = models.split(";")
            rows.append(
                {
                    "reranking_pool_id": f"prev-{user}-{position}",
                    "user_id": user,
                    "candidate_item_id": item,
                    "candidate_position": position,
                    "source_models": models,
                    "source_ranks": ranks,
                    "source_scores": ";".join(["0.1"] * len(model_list)),
                    "source_seed": ";".join(str(SEEDS[m]) for m in model_list),
                    "is_positive": str(item == positive).lower(),
                    "positive_item_id": positive,
                    "cohort_name": "cohort",
                    "candidate_set_name": "previous",
                    "pool_policy": "adapted",
                    "paper_equivalence": "not_exact",
                    "notes": "fixture",
                }
            )
    return rows


def _fixture_config(root: Path) -> dict[str, object]:
    f6f2_hash = "f6f2hash"
    f6f1_hash = "f6f1hash"
    f6e4c_hash = "f6e4chash"
    input_hashes: dict[str, dict[str, str]] = {}
    input_hashes["f6f2_manifest"] = _write_json(root, "inputs/f6f2_manifest.json", {"decision": "f6f2_ok", "scientific_content_hash": f6f2_hash, "f6f1r_scientific_comparison_status": "identical"})
    input_hashes["f6f1_manifest"] = _write_json(root, "inputs/f6f1_manifest.json", {"decision": "f6f1_ok", "scientific_content_hash": f6f1_hash})
    input_hashes["f6e4c_manifest"] = _write_json(root, "inputs/f6e4c_manifest.json", {"decision": "f6e4c_ok", "scientific_content_hash": f6e4c_hash})
    input_hashes["f6f2_remaining_gaps"] = _write_csv(root, "inputs/f6f2_gaps.csv", [{"gap": "LastFM"}])
    input_hashes["f6f2_next_phase_requirements"] = _write_csv(root, "inputs/f6f2_req.csv", [{"requirement": "pool"}])
    input_hashes["f6d1_baseline_protocol"] = _write_csv(root, "inputs/f6d1_protocol.csv", [{"model": "random_seeded", "model_family": "Random"}])
    input_hashes["f6d1_prediction_validation"] = _write_csv(root, "inputs/f6d1_validation.csv", [{"status": "pass"}])
    input_hashes["f6d1_utility_metrics"] = _write_csv(root, "inputs/f6d1_utility.csv", [{"model": "random_seeded", "metric": "HR@5", "value": "1"}])
    f6d1_rows = []
    for model in ["random_seeded", "mostpop_train_only", "bm25_like_adapted"]:
        for user in ["1", "2"]:
            for rank, item in enumerate(["10", "20", "30", "40"], start=1):
                f6d1_rows.append({"model": model, "seed": 42, "user_id": user, "candidate_item_id": item, "rank": rank, "score": "0.1"})
    input_hashes["f6d1_baseline_predictions"] = _write_csv(root, "inputs/f6d1_predictions.csv", f6d1_rows)
    input_hashes["f6d2_execution_manifest"] = _write_csv(root, "inputs/f6d2_exec.csv", [{"model": model, "seed": seed} for model, seed in SEEDS.items()])
    input_hashes["f6d2_reranking_policy_audit"] = _write_csv(root, "inputs/f6d2_policy.csv", [{"aspect": "union", "local_recovered_value": "round-robin"}])
    source_selection = []
    for model in MODELS:
        for seed in [42, SEEDS[model]]:
            selected = seed == SEEDS[model]
            source_selection.append({"model": model, "seed": seed, "selection_role": "best_validation_seed" if selected else "aggregate_not_directly_rankable", "selection_metric": "validation_hr_at_5_full_catalog", "selection_value": "0.5" if selected else "0.1", "selection_reason": "Highest validation HR@5" if selected else "Not selected"})
    input_hashes["f6d2_reranking_source_selection"] = _write_csv(root, "inputs/f6d2_source_selection.csv", source_selection)
    input_hashes["f6d2_reranking_pool_validation"] = _write_csv(root, "inputs/f6d2_pool_validation.csv", [{"status": "pass"}])
    input_hashes["f6d2_candidate_predictions"] = _write_csv(root, "inputs/f6d2_candidate.csv", _candidate_prediction_rows())
    input_hashes["f6d2_full_catalog_predictions"] = _write_csv(root, "inputs/f6d2_full.csv", _full_catalog_rows())
    input_hashes["previous_adapted_pool"] = _write_csv(root, "inputs/previous_pool.csv", _previous_pool_rows())
    cohort = [
        {"user_id": "1", "train_history_count": 3, "test_item_id": "40", "test_item_title": "artist 40", "history_item_ids": "['1','2','3']", "history_item_titles": "['a','b','c']", "metadata_ready": "true"},
        {"user_id": "2", "train_history_count": 3, "test_item_id": "99", "test_item_title": "artist 99", "history_item_ids": "['4','5','6']", "history_item_titles": "['d','e','f']", "metadata_ready": "true"},
    ]
    input_hashes["lastfm_ranking_cohort"] = _write_csv(root, "inputs/cohort.csv", cohort)
    input_hashes["lastfm_train"] = _write_csv(root, "inputs/train.csv", [{"user_id": "1", "item_id": "1"}])
    input_hashes["lastfm_validation"] = _write_csv(root, "inputs/validation.csv", [{"user_id": "1", "item_id": "2"}])
    input_hashes["lastfm_test"] = _write_csv(root, "inputs/test.csv", [{"user_id": "1", "item_id": "40"}])
    catalog_items = sorted({"10", "20", "30", "40", "50", "60", "70", "80", "90", "99", "100", "11", "21", "31", "41", "51", "61"})
    catalog = [{"item_id": item, "source_item_id": item, "title": "same title" if item in {"20", "21"} else f"artist {item}"} for item in catalog_items]
    input_hashes["lastfm_catalog"] = _write_csv(root, "inputs/catalog.csv", catalog)
    input_hashes["lastfm_interactions"] = _write_csv(root, "inputs/interactions.csv", [{"user_id": "1", "item_id": "1"}])
    input_hashes["lastfm_item_identity_audit"] = _write_csv(root, "inputs/identity.csv", [{"item_id": "10", "identity_status": "unique"}])
    input_hashes["f6d3_reranking_protocol"] = _write_csv(root, "inputs/f6d3_protocol.csv", [{"aspect": "top-K per source", "paper_value": "top-K source outputs"}])
    input_hashes["f6d3_reranking_pool_composition"] = _write_csv(root, "inputs/f6d3_comp.csv", [{"summary_key": "users", "value": "2"}])
    input_hashes["f6d3_reranking_evaluation_policy"] = _write_csv(root, "inputs/f6d3_eval.csv", [{"policy_role": "adapted_primary"}])
    input_hashes["f6d3_reranking_manifest"] = _write_csv(root, "inputs/f6d3_manifest.csv", [{"request_id": "r"}])
    input_hashes["f6d3_ranking_manifest"] = _write_csv(root, "inputs/f6d3_ranking_manifest.csv", [{"request_id": "r"}])
    input_hashes["f6d4a_request_audit"] = _write_csv(root, "inputs/f6d4a_audit.csv", [{"request_id": "r"}])
    input_hashes["f6d4a_qwen2_raw"] = _write_csv(root, "inputs/qwen.jsonl", [{"raw": "x"}])
    input_hashes["f6d4a_mistral_raw"] = _write_csv(root, "inputs/mistral.jsonl", [{"raw": "x"}])
    input_hashes["f2_protected_hashes"] = _write_csv(root, "inputs/protected.csv", [{"path": "x", "sha256": "y"}])
    input_hashes["paper_pdf"] = _write_csv(root, "inputs/paper.pdf", [{"bytes": "x"}])
    input_hashes["paper_translation_docx"] = _write_csv(root, "inputs/paper.docx", [{"bytes": "x"}])
    input_hashes["original_repo_readme"] = _write_csv(root, "inputs/readme.md", [{"text": "x"}])
    input_hashes["original_repo_llm_metrics"] = _write_csv(root, "inputs/llm_metrics.py", [{"text": "x"}])
    input_hashes["original_repo_zero_shot"] = _write_csv(root, "inputs/zero_shot.py", [{"text": "x"}])

    tables = {name: f"results/tables/phase_f6g1_{name}.csv" for name in f6g1.TABLE_COLUMNS}
    return {
        "phase": {
            "id": "phase_f6g1",
            "expected_branch": "",
            "expected_head": "",
            "interpreter_path": str(Path(sys.executable)),
            "environment_classification": "test",
            "expected_f6f2_decision": "f6f2_ok",
            "expected_f6f2_scientific_hash": f6f2_hash,
            "expected_f6f1_decision": "f6f1_ok",
            "expected_f6f1_scientific_hash": f6f1_hash,
            "expected_f6e4c_decision": "f6e4c_ok",
            "expected_f6e4c_scientific_hash": f6e4c_hash,
            "expected_f6f1r_scientific_status": "identical",
        },
        "dataset": {"name": "LastFM", "expected_users": 2, "target_pool_size": 4, "source_top_n": 6, "max_requests_per_shard": 3},
        "pool_policy": {
            "canonical_policy_name": "fixture_pool",
            "previous_pool_name": "previous",
            "operation": "round_robin_union",
            "model_order": MODELS,
            "selected_seeds": SEEDS,
            "final_pool_classification_if_gates_pass": "reconstructed_pool_with_adapted_components",
            "future_experiment_label_if_gates_pass": "adapted_lastfm_reranking",
        },
        "future_requests": {
            "models": ["qwen2", "mistral"],
            "strategies": ["base", "llmrank"],
            "expected_output_length": 4,
            "parser_contract": "indexed_json",
            "invalid_output_policy": "invalid_zero",
            "operational_utility_policy": "all_user",
            "conditional_utility_policy": "positive_present",
        },
        "input_hashes": input_hashes,
        "outputs": {"report": "report/phase_f6g1_report.md", "freeze_manifest": "results/manifests/phase_f6g1_manifest.json", "tables": tables},
        "validation": {"run_suites": False, "suites": [], "unsafe_exclusions": ["training", "ollama"]},
    }


@pytest.fixture()
def fixture_state(tmp_path: Path):
    config = _fixture_config(tmp_path)
    frames = f6g1.load_frames(tmp_path, config)
    return tmp_path, config, frames


def test_import_without_side_effects() -> None:
    assert f6g1.PHASE_ID == "phase_f6g1"


def test_no_network_imports() -> None:
    imports = _imports()
    assert not {"requests", "httpx", "socket", "urllib"}.intersection(imports)


def test_no_ollama_import() -> None:
    assert "ollama" not in _imports()


def test_no_api_or_llm_inference_imports() -> None:
    assert not {"openai", "anthropic", "transformers", "llama_cpp"}.intersection(_imports())


def test_no_training_or_cuda_imports() -> None:
    assert not {"torch", "tensorflow", "keras"}.intersection(_imports())


def _imports() -> set[str]:
    tree = ast.parse(Path(f6g1.__file__).read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    return imports


def test_explicit_interpreter_check_passes(fixture_state) -> None:
    root, config, _frames = fixture_state
    assert f6g1.verify_interpreter(root, config)["status"] == "pass"


def test_output_path_guard_rejects_prior_phase_path(fixture_state) -> None:
    _root, config, _frames = fixture_state
    bad = dict(config)
    bad["outputs"] = {**config["outputs"], "report": "report/phase_f6f2_bad.md"}
    with pytest.raises(ValueError):
        f6g1.validate_output_paths(bad)


def test_input_integrity_pass_and_mismatch_detection(fixture_state, tmp_path: Path) -> None:
    root, config, _frames = fixture_state
    rows = f6g1.verify_input_integrity(root, config)
    assert all(row["status"] == "pass" for row in rows)
    changed = _fixture_config(tmp_path / "changed")
    changed["input_hashes"]["lastfm_test"]["sha256"] = "0" * 64
    rows = f6g1.verify_input_integrity(tmp_path / "changed", changed)
    assert any(row["artifact"] == "lastfm_test" and row["changed"] == "true" for row in rows)


def test_phase_manifest_hash_gates(fixture_state) -> None:
    root, config, _frames = fixture_state
    assert f6g1.verify_phase_manifests(root, config)["status"] == "pass"


def test_source_inventory_local_only(fixture_state) -> None:
    _root, config, _frames = fixture_state
    rows = f6g1.build_source_inventory(config)
    assert rows
    assert all(not row["path"].startswith("http") for row in rows)


def test_paper_contract_keeps_missing_union_explicit(fixture_state) -> None:
    _root, _config, frames = fixture_state
    rows = f6g1.build_paper_pool_contract(frames)
    union = next(row for row in rows if row["component"] == "union_policy")
    assert union["classification"] == "missing_evidence"


def test_complete_cohort_audit(fixture_state) -> None:
    _root, config, frames = fixture_state
    rows = f6g1.build_cohort_audit(frames, config)
    assert len(rows) == 2
    assert all(row["usable_status"] == "usable" for row in rows)


def test_cohort_audit_blocks_missing_metadata(fixture_state) -> None:
    _root, config, frames = fixture_state
    altered = dict(frames)
    altered["catalog"] = frames["catalog"].loc[frames["catalog"]["item_id"].ne("99")].copy()
    rows = f6g1.build_cohort_audit(altered, config)
    assert any(row["user_id"] == "2" and row["usable_status"] == "blocked" for row in rows)


def test_prediction_inventory_covers_baseline_and_full_catalog(fixture_state) -> None:
    _root, config, frames = fixture_state
    rows = f6g1.build_prediction_inventory(frames, config)
    assert {"random_seeded", "bprmf", "lightgcn"}.issubset({row["model_name"] for row in rows})
    assert any(row["prediction_file"].endswith("full_catalog_rankings.csv") and row["prediction_completeness"] == "complete" for row in rows)


def test_seed_policy_selected_seeds_are_frozen(fixture_state) -> None:
    _root, config, frames = fixture_state
    rows = f6g1.build_seed_policy_audit(frames, config)
    selected = {(row["model_name"], row["seed"]) for row in rows if row["decision"] == "use_for_adapted_pool"}
    assert selected == {(model, str(seed)) for model, seed in SEEDS.items()}


def test_candidate_source_model_decisions_separate_diagnostics(fixture_state) -> None:
    _root, config, frames = fixture_state
    rows = f6g1.build_candidate_source_model_decisions(frames, config)
    decisions = {row["model_name"]: row["inclusion_decision"] for row in rows}
    assert decisions["bprmf"] == "include_canonical_adapted"
    assert decisions["random_seeded"] == "diagnostic_only_not_canonical"


def test_deterministic_round_robin_pool_construction(fixture_state) -> None:
    _root, config, frames = fixture_state
    first = f6g1.reconstruct_pool(frames, config)
    second = f6g1.reconstruct_pool(frames, config)
    assert first == second
    assert [row["candidate_item_id"] for row in first if row["user_id"] == "1"] == ["10", "20", "30", "40"]


def test_duplicate_provenance_attached_without_duplicate_rows(fixture_state) -> None:
    _root, config, frames = fixture_state
    rows = f6g1.reconstruct_pool(frames, config)
    item10 = next(row for row in rows if row["user_id"] == "1" and row["candidate_item_id"] == "10")
    assert item10["source_models"] == "bprmf;gru4rec"
    assert len({row["candidate_item_id"] for row in rows if row["user_id"] == "1"}) == 4


def test_no_positive_insertion_without_contract(fixture_state) -> None:
    _root, config, frames = fixture_state
    rows = f6g1.reconstruct_pool(frames, config)
    assert "99" not in [row["candidate_item_id"] for row in rows if row["user_id"] == "2"]
    assert all(row["positive_present"] == "false" for row in rows if row["user_id"] == "2")


def test_pool_instance_upper_bounds_and_positive_present_separation(fixture_state) -> None:
    _root, config, frames = fixture_state
    rows = f6g1.reconstruct_pool(frames, config)
    summaries = f6g1.build_pool_instance_summary(rows, config)
    by_user = {row["user_id"]: row for row in summaries}
    assert by_user["1"]["upper_bound_hr_at_5"] == 1
    assert by_user["2"]["upper_bound_hr_at_5"] == 0


def test_positive_policy_materializes_two_populations(fixture_state) -> None:
    _root, config, frames = fixture_state
    pool = f6g1.reconstruct_pool(frames, config)
    summaries = f6g1.build_pool_instance_summary(pool, config)
    rows = f6g1.build_positive_policy(summaries, config)
    assert {row["population"] for row in rows} == {"all_user", "positive_present"}
    assert next(row for row in rows if row["population"] == "all_user")["canonical_population"] == "true"


def test_previous_pool_reconciliation_valid_but_adapted(fixture_state) -> None:
    _root, config, frames = fixture_state
    pool = f6g1.reconstruct_pool(frames, config)
    rows = f6g1.build_previous_pool_reconciliation(pool, frames, config)
    assert all(row["outcome"] == "previous_pool_valid_but_adapted" for row in rows)


def test_previous_pool_mismatch_is_detected(fixture_state) -> None:
    _root, config, frames = fixture_state
    altered = dict(frames)
    altered["previous_pool"] = frames["previous_pool"].copy()
    altered["previous_pool"].loc[0, "candidate_item_id"] = "999"
    pool = f6g1.reconstruct_pool(frames, config)
    rows = f6g1.build_previous_pool_reconciliation(pool, altered, config)
    assert any(row["outcome"] == "previous_pool_superseded_by_reconstructed_pool" for row in rows)


def test_title_reconciliation_reversible_and_ambiguous_preserved(fixture_state) -> None:
    _root, config, frames = fixture_state
    pool = f6g1.reconstruct_pool(frames, config)
    rows = f6g1.build_title_item_reconciliation(pool, frames)
    assert all(row["reversible_mapping"] == "true" for row in rows)
    assert any(row["ambiguous_title"] == "true" for row in rows)


def test_traditional_pool_metrics_do_not_impute_missing_scores(fixture_state) -> None:
    _root, config, frames = fixture_state
    pool = f6g1.reconstruct_pool(frames, config)
    summaries = f6g1.build_pool_instance_summary(pool, config)
    rows = f6g1.build_traditional_pool_metrics(pool, summaries, config)
    assert any(row["metric"] == "candidate_pool_score_coverage_rate" for row in rows)
    assert all("not imputed" in row["notes"] or row["classification"] == "structural_diagnostic" for row in rows)


def _gates(fixture_state):
    _root, config, frames = fixture_state
    input_rows = f6g1.verify_input_integrity(_root, config)
    cohort = f6g1.build_cohort_audit(frames, config)
    pred = f6g1.build_prediction_inventory(frames, config)
    seed = f6g1.build_seed_policy_audit(frames, config)
    pool = f6g1.reconstruct_pool(frames, config)
    summaries = f6g1.build_pool_instance_summary(pool, config)
    titles = f6g1.build_title_item_reconciliation(pool, frames)
    previous = f6g1.build_previous_pool_reconciliation(pool, frames, config)
    return f6g1.build_comparability_gate(input_rows, cohort, pred, seed, summaries, titles, previous, config), pool, summaries


def test_comparability_gate_allows_adapted_inference_with_warnings(fixture_state) -> None:
    gates, _pool, _summaries = _gates(fixture_state)
    assert f6g1.future_allowed(gates)
    assert any(row["status"] == "warning" for row in gates)


def test_blocked_gate_blocks_future_inference() -> None:
    gates = [f6g1.gate_row("x", "r", "o", "blocked", "i", "fix", False)]
    assert not f6g1.future_allowed(gates)


def test_exactness_guard_does_not_promote_to_paper_exact(fixture_state) -> None:
    _root, config, _frames = fixture_state
    gates, _pool, _summaries = _gates(fixture_state)
    rows = f6g1.build_exactness_classification(gates, config)
    assert "paper_exact_pool" not in {row["classification"] for row in rows}


def test_future_logical_instances_are_all_user(fixture_state) -> None:
    _root, config, frames = fixture_state
    gates, pool, summaries = _gates(fixture_state)
    rows = f6g1.build_future_logical_instances(pool, summaries, frames, gates, config)
    assert len(rows) == 2
    assert {row["positive_present"] for row in rows} == {"true", "false"}


def test_future_request_ids_are_unique(fixture_state) -> None:
    _root, config, frames = fixture_state
    gates, pool, summaries = _gates(fixture_state)
    logical = f6g1.build_future_logical_instances(pool, summaries, frames, gates, config)
    requests = f6g1.build_future_requests(logical, gates, config)
    assert len(requests) == 8
    assert len({row["request_id"] for row in requests}) == len(requests)


def test_future_shards_respect_max_size(fixture_state) -> None:
    _root, config, frames = fixture_state
    gates, pool, summaries = _gates(fixture_state)
    logical = f6g1.build_future_logical_instances(pool, summaries, frames, gates, config)
    requests = f6g1.build_future_requests(logical, gates, config)
    shards = f6g1.build_future_shards(requests, gates, config)
    assert max(int(row["request_count"]) for row in shards) <= 3


def test_empty_future_manifests_when_blocked(fixture_state) -> None:
    _root, config, frames = fixture_state
    gates, pool, summaries = _gates(fixture_state)
    gates[0]["future_llm_inference_allowed"] = "false"
    logical = f6g1.build_future_logical_instances(pool, summaries, frames, gates, config)
    requests = f6g1.build_future_requests(logical, gates, config)
    shards = f6g1.build_future_shards(requests, gates, config)
    assert logical == [{column: "" for column in f6g1.LOGICAL_COLUMNS}]
    assert requests == [{column: "" for column in f6g1.REQUEST_COLUMNS}]
    assert shards == [{column: "" for column in f6g1.SHARD_COLUMNS}]


def test_missing_or_blocked_records_protocol_gaps(fixture_state) -> None:
    gates, _pool, _summaries = _gates(fixture_state)
    rows = f6g1.build_missing_or_blocked(gates)
    assert {"paper source top-K", "paper union policy"}.issubset({row["component"] for row in rows})


def test_decision_complete_with_adapted_pool(fixture_state) -> None:
    _root, config, _frames = fixture_state
    gates, _pool, _summaries = _gates(fixture_state)
    exactness = f6g1.build_exactness_classification(gates, config)
    decision, recommendation = f6g1.determine_decision(gates, {"total_failed": 0}, f6g1.verify_input_integrity(_root, config), exactness)
    assert decision == f6g1.DECISION_ADAPTED
    assert recommendation == f6g1.RECOMMEND_READY


def test_decision_blocked_by_tests(fixture_state) -> None:
    _root, config, _frames = fixture_state
    gates, _pool, _summaries = _gates(fixture_state)
    exactness = f6g1.build_exactness_classification(gates, config)
    decision, _recommendation = f6g1.determine_decision(gates, {"total_failed": 1}, f6g1.verify_input_integrity(_root, config), exactness)
    assert decision == f6g1.DECISION_TESTS


def test_run_phase_writes_only_configured_outputs_and_is_deterministic(fixture_state) -> None:
    root, config, _frames = fixture_state
    first = f6g1.run_phase(root, config)
    first_hashes = {name: _sha(root / path) for name, path in config["outputs"]["tables"].items()}
    first_report = _sha(root / config["outputs"]["report"])
    second = f6g1.run_phase(root, config)
    second_hashes = {name: _sha(root / path) for name, path in config["outputs"]["tables"].items()}
    assert first_hashes == second_hashes
    assert first_report == _sha(root / config["outputs"]["report"])
    assert first["scientific_content_hash"] == second["scientific_content_hash"]


def test_manifest_scientific_section_is_deterministic(fixture_state) -> None:
    root, config, _frames = fixture_state
    first = f6g1.run_phase(root, config)
    second = f6g1.run_phase(root, config)
    keys = ["canonical_pool_hash", "future_request_manifest_hash", "future_shard_manifest_hash", "decision"]
    assert {key: first[key] for key in keys} == {key: second[key] for key in keys}
