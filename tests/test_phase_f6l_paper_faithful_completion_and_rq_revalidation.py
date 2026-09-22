"""Focused checks for F6L paper-faithful checkpoint artifacts."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.analysis import plan_f6l_paper_faithful_completion_and_rq_revalidation as f6l
from src.analysis import phase_f6l_profile_execution as profile_exec


pytestmark = [
    pytest.mark.cleanup_historical,
    pytest.mark.phase_snapshot,
    pytest.mark.refactor_regression,
]

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase_f6l_paper_faithful_completion_and_rq_revalidation.yaml"
REPORT = ROOT / "report" / "phase_f6l_paper_faithful_completion_and_rq_revalidation.md"
MANIFEST = ROOT / "results" / "manifests" / "phase_f6l_execution_checkpoint_manifest.json"
PROTECTED = ROOT / "results" / "manifests" / "phase_f6l_protected_input_hashes.json"
TABLES = ROOT / "results" / "tables"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_runtime_and_checkpoint_token() -> None:
    assert Path(sys.executable).resolve() == f6l.RECOVERED_INTERPRETER.resolve()
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["decision_token"] == f6l.DECISION_CHECKPOINT
    assert payload["policy"]["inference_performed"] is False
    assert payload["policy"]["ollama_started"] is False
    assert payload["policy"]["api_calls_performed"] is False
    assert payload["policy"]["downloads_performed"] is False
    assert payload["request_count"] == 8800


def test_required_tables_exist_with_expected_schema() -> None:
    for name, path in f6l.OUTPUT_TABLES.items():
        assert path.exists(), name
        table_rows = rows(path)
        assert table_rows, name
        assert list(table_rows[0]) == f6l.TABLE_FIELDS[name], name

    assert len(rows(TABLES / "phase_f6l_paper_repository_exactness_matrix.csv")) >= 14
    assert len(rows(TABLES / "phase_f6l_rq_revalidation.csv")) == 14
    assert len(rows(TABLES / "phase_f6l_execution_checkpoint_summary.csv")) == 22


def test_classification_and_rq_status_enums_are_strict() -> None:
    for name, path in f6l.OUTPUT_TABLES.items():
        for row in rows(path):
            value = row.get("evidence_classification")
            if value:
                assert value in f6l.EVIDENCE_CLASSES, (name, value)

    for row in rows(TABLES / "phase_f6l_rq_revalidation.csv"):
        assert row["new_status"] in f6l.RQ_STATUSES


def test_existing_lastfm_evidence_is_reclassified_not_promoted() -> None:
    ranking = rows(TABLES / "phase_f6l_lastfm_ranking_exactness.csv")
    existing = next(row for row in ranking if row["cell_id"] == "F6L-LASTFM-RANKING-EXISTING-F6D4A")
    assert existing["evidence_classification"] == "adapted_protocol_reproduction"
    assert existing["users"] == "1859"
    assert existing["output_length"] == "5"
    assert "not_paper_1000" in existing["sample_exactness"]

    reranking = rows(TABLES / "phase_f6l_lastfm_reranking_results.csv")
    assert any(row["status"] == "existing_adapted_blocked_no_raw_responses" for row in reranking)
    paper_blocked = [row for row in reranking if row["status"] == "blocked_before_checkpoint_execution"]
    assert paper_blocked
    assert {row["evidence_classification"] for row in paper_blocked} == {"blocked_missing_original_specification"}
    assert {row["request_count"] for row in paper_blocked} == {"0"}


def test_request_manifest_count_excludes_stale_7436_reranking_plan() -> None:
    request_rows = rows(TABLES / "phase_f6l_execution_request_manifest.csv")
    total = sum(int(row["total_requests"]) for row in request_rows if row["evidence_classification"] != "blocked_missing_original_specification")
    assert total == 8800

    rerank = next(row for row in request_rows if row["request_group_id"] == "F6L-REQ-LASTFM-RERANKING-PAPER")
    assert rerank["total_requests"] == "0"
    assert rerank["evidence_classification"] == "blocked_missing_original_specification"
    assert "7436" in rerank["notes"]
    assert "not carried into F6L paper-faithful count" in rerank["notes"]


def test_atomic_request_breakdown_and_authorization_matrix_recalculate_8800() -> None:
    breakdown = rows(TABLES / "phase_f6l_execution_request_breakdown.csv")
    assert sum(int(row["expected_requests"]) for row in breakdown) == 8800
    assert sum(int(row["expected_requests"]) for row in breakdown if row["experiment_family"] == "profile_generation") == 800
    assert sum(int(row["expected_requests"]) for row in breakdown if row["input_variant"] == "history_only") == 800
    assert sum(int(row["expected_requests"]) for row in breakdown if row["input_variant"] == "profile_only") == 1600
    assert sum(int(row["expected_requests"]) for row in breakdown if row["input_variant"] == "profile_plus_history") == 1600
    assert sum(int(row["expected_requests"]) for row in breakdown if row["experiment_family"] == "lastfm_ranking") == 4000
    assert sum(int(row["expected_requests"]) for row in breakdown if row["experiment_family"] == "lastfm_reranking") == 0
    assert any(row["input_variant"] == "LLMSRec" and row["authorization_status"] == "blocked" for row in breakdown)

    matrix = rows(TABLES / "phase_f6l_final_execution_authorization_matrix.csv")
    assert sum(int(row["expected_requests"]) for row in matrix if row["included_in_final_request_count"] == "true") == 8800
    assert any(row["evidence_classification"] == "adapted_protocol_reproduction" for row in matrix)
    assert any(row["evidence_classification"] == "blocked_paid_or_unavailable_model" for row in matrix)


def test_lastfm_reranking_prerequisite_audit_identifies_non_generic_hard_blockers() -> None:
    audit = rows(TABLES / "phase_f6l_lastfm_reranking_prerequisite_audit.csv")
    by_name = {row["requirement"]: row for row in audit}
    for requirement in [
        "BPRMF recommendations",
        "GRU4Rec recommendations",
        "SASRec recommendations",
        "LightGCN recommendations",
        "pool finale da 20 candidati",
        "candidate ordering",
    ]:
        assert by_name[requirement]["causes_zero_reranking_request_count"] == "true"
        assert by_name[requirement]["missing_not_recoverable"] == "true"
    assert by_name["regola ufficiale di aggregazione"]["reconstructed_from_official_code"] == "true"


def test_official_artifact_recovery_records_hashes_without_downloads() -> None:
    recovery = rows(TABLES / "phase_f6l_official_artifact_recovery.csv")
    assert recovery
    assert {row["download_performed"] for row in recovery} == {"false"}
    present = [row for row in recovery if row["recovery_status"].startswith("local_official_artifact")]
    assert present
    assert all(row["sha256"] for row in present)
    missing = [row for row in recovery if row["recovery_status"] == "not_found_in_official_sources"]
    assert {row["prerequisite_after_recovery"] for row in missing} == {"missing_not_recoverable"}


def test_profile_contract_uses_official_artifacts_and_blocks_missing_cells() -> None:
    profile_rows = rows(TABLES / "phase_f6l_profile_generation_contract.csv")
    official = [row for row in profile_rows if row["local_execution_status"] == "read_only_no_inference"]
    assert len(official) == 6
    assert {row["evidence_classification"] for row in official} == {"exact_original_artifact_reproduction"}
    assert all(row["request_count"] == "0" for row in official)

    blocked = [row for row in profile_rows if row["evidence_classification"] == "blocked_missing_original_specification"]
    assert {row["dataset"] for row in blocked} == {"ml1m", "lastfm"}


def test_protected_inputs_include_23_f6i2p_yaml_and_no_missing() -> None:
    payload = json.loads(PROTECTED.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["missing"] == []
    assert payload["f6i2p_yaml_count"] == 23
    protected_paths = {entry["path"] for entry in payload["protected_inputs"]}
    assert "original_repo/EvaLLMasRecommender/eva/templates/template.txt" in protected_paths
    assert "results/llm/finalization_f6d4a_lastfm_ranking_qwen2_raw.jsonl" in protected_paths


def test_report_and_handoff_contain_checkpoint_stop_and_no_inference_policy() -> None:
    report = REPORT.read_text(encoding="utf-8")
    handoff = (ROOT / "report" / "future_operations_context_handoff.md").read_text(encoding="utf-8")
    notes = (ROOT / "report" / "notes.md").read_text(encoding="utf-8")

    assert f6l.DECISION_CHECKPOINT in report
    assert "No Ollama start" in report
    assert "Total authorized future open-model requests after explicit user approval: `8800`" in report
    assert "LastFM reranking: `0`" in report
    assert f6l.DECISION_CHECKPOINT in handoff
    assert "Recalculated future local open-model request count after explicit approval: `8800`" in handoff
    assert f6l.DECISION_CHECKPOINT in notes


def test_no_unapproved_phase_f6l_raw_outputs_created_and_downloads_not_performed() -> None:
    raw_root = ROOT / "results" / "llm"
    offenders = list(raw_root.glob("*phase_f6l*")) if raw_root.exists() else []
    allowed_prefixes = (
        "phase_f6l_profile_execution_",
        "phase_f6l_lastfm_operationalized_ranking_",
    )
    unapproved = [
        path
        for path in offenders
        if not path.name.startswith(allowed_prefixes)
    ]
    assert unapproved == []

    downloads = rows(TABLES / "phase_f6l_external_download_manifest.csv")
    assert downloads
    assert {row["download_performed"] for row in downloads} == {"false"}


def test_builder_is_deterministic_in_memory() -> None:
    first = f6l.build(CONFIG, write_outputs_enabled=False)
    second = f6l.build(CONFIG, write_outputs_enabled=False)
    assert f6l.semantic_digest(first) == f6l.semantic_digest(second)
    assert first.decision_token == f6l.DECISION_CHECKPOINT


def test_execute_mode_is_fail_closed_without_required_flags() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "src" / "experiments" / "build_phase_f6l_paper_faithful_completion_and_rq_revalidation.py"),
            "--config",
            str(CONFIG),
            "--mode",
            "execute",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["decision"] == "blocked_missing_execution_flags"
    assert "--checkpoint-id" in payload["missing"]


def test_authorized_execute_preflight_blocks_missing_frozen_request_inputs() -> None:
    args = SimpleNamespace(
        config=CONFIG,
        mode="execute",
        allow_api_calls=True,
        local_only=True,
        checkpoint_id=f6l.DECISION_CHECKPOINT,
        raw_output_dir=Path("results/llm/phase_f6l_unit_preflight_raw"),
    )
    fake_ollama = {
        "models": [
            {
                "name": "qwen2:7b-instruct",
                "digest": "qwen2-test-digest",
                "modified_at": "2026-08-01T00:00:00+02:00",
                "size": 1,
                "details": {"family": "qwen2", "parameter_size": "7.6B", "quantization_level": "Q4_0"},
            },
            {
                "name": "mistral:7b-instruct",
                "digest": "mistral-test-digest",
                "modified_at": "2026-08-01T00:00:00+02:00",
                "size": 1,
                "details": {"family": "llama", "parameter_size": "7.2B", "quantization_level": "Q4_K_M"},
            },
        ]
    }

    result = f6l.build_execute_preflight(args, write_outputs_enabled=False, ollama_payload=fake_ollama)
    preflight = {row["check_name"]: row for row in result.tables["execute_preflight"]}
    blocker = result.tables["execution_blocker"][0]

    assert result.decision_token == f6l.DECISION_EXECUTION_BLOCKED
    assert preflight["authorization_count_8800"]["status"] == "pass"
    assert preflight["allow_api_calls_local_only_guard"]["status"] == "pass"
    assert preflight["ollama_local_model_identities_available"]["status"] == "pass"
    assert preflight["per_request_manifest_exists_and_unique"]["status"] == "fail"
    assert preflight["lastfm_frozen_sample_artifact"]["status"] == "fail"
    assert blocker["planned_requests"] == "8800"
    assert blocker["completed_requests"] == "0"
    assert blocker["strict_valid_requests"] == "0"
    assert blocker["failed_requests"] == "0"
    assert blocker["residual_requests"] == "8800"
    assert not (ROOT / "results" / "llm" / "phase_f6l_unit_preflight_raw").exists()


def test_contract_materialization_blocks_on_missing_operational_ks_specification() -> None:
    result = f6l.build_contract_materialization(CONFIG, write_outputs_enabled=False)
    audit = {row["contract_area"]: row for row in result.tables["lastfm_contract_reconstruction_audit"]}
    blockers = {row["contract_area"]: row for row in result.tables["execution_contract_materialization_blockers"]}
    summary = {row["item"]: row for row in result.tables["execution_contract_materialization_summary"]}

    assert result.decision_token == f6l.DECISION_CONTRACT_MATERIALIZATION_BLOCKED
    assert audit["ks_gate_hypothesis"]["status"] == "partially_recovered_not_operational"
    assert audit["ks_feature"]["status"] == "missing"
    assert audit["ks_threshold"]["status"] == "missing"
    assert audit["ks_feature"]["blocks_materialization"] == "true"
    assert "ks_gate_operational_definition" in blockers
    assert summary["authorization_matrix_requests"]["value"] == "8800"
    assert summary["sample_materialized"]["value"] == "false"
    assert summary["candidate_sets_materialized"]["value"] == "false"
    assert summary["request_manifest_materialized"]["value"] == "false"
    assert result.manifest["deterministic_in_memory_digest_match"] is True
    assert not f6l.LASTFM_FROZEN_SAMPLE_COHORT_PATH.exists()
    assert not f6l.LASTFM_FROZEN_SAMPLE_CANDIDATES_PATH.exists()
    assert not f6l.PER_REQUEST_MANIFEST_PATH.exists()


def test_separated_profile_contract_materializes_4800_independent_requests() -> None:
    result = f6l.build_separated_contracts(CONFIG, write_outputs_enabled=False)
    manifest = result.tables["profile_request_manifest"]
    by_id = {row["request_id"]: row for row in manifest}

    assert result.decision_token == f6l.DECISION_SEPARATED_PROFILE_READY
    assert len(manifest) == 4800
    assert len(by_id) == 4800
    assert Counter(row["input_variant"] for row in manifest) == {
        "profile_generation": 800,
        "history_only": 800,
        "profile_only": 1600,
        "profile_plus_history": 1600,
    }
    assert Counter(row["dataset"] for row in manifest) == {"beauty": 2400, "sports": 2400}
    assert set(row["model_name"] for row in manifest) == set(f6l.MODEL_DIGESTS)
    assert {row["model_digest"] for row in manifest} == set(f6l.MODEL_DIGESTS.values())
    assert all(row["dataset"].lower() != "lastfm" for row in manifest)
    assert all(row["evidence_class"] == "paper_protocol_reproduction_different_open_model" for row in manifest)
    assert all(row["candidate_pool_size"] == "20" for row in manifest if row["input_variant"] != "profile_generation")
    assert all(row["positive_item_id"] for row in manifest if row["input_variant"] != "profile_generation")
    assert all(row["positive_candidate_position"] for row in manifest if row["input_variant"] != "profile_generation")
    assert all(row["prompt_template_hash"] for row in manifest)
    assert all(row["rendered_prompt_hash"] for row in manifest)
    assert len({row["request_contract_hash"] for row in manifest}) == 4800

    for row in manifest:
        dependency = row["dependency_request_id"]
        if row["input_variant"] in {"history_only", "profile_generation"}:
            assert dependency == ""
        else:
            parent = by_id[dependency]
            assert parent["input_variant"] == "profile_generation"
            assert parent["dataset"] == row["dataset"]
            assert parent["user_id"] == row["user_id"]
            assert parent["model_name"] == row["profile_generator_name"]
            assert parent["dependency_request_id"] == ""


def test_profile_execute_preflight_accepts_exact_4800_manifest_without_inference() -> None:
    args = SimpleNamespace(
        config=CONFIG,
        mode="execute-profile",
        allow_api_calls=True,
        local_only=True,
        checkpoint_id=f6l.DECISION_SEPARATED_PROFILE_READY,
        raw_output_dir=Path("results/llm/phase_f6l_profile_execution_unit_preflight"),
    )
    fake_ollama = {
        "models": [
            {
                "name": "qwen2:7b-instruct",
                "digest": f6l.MODEL_DIGESTS["qwen2:7b-instruct"],
                "modified_at": "2026-08-01T00:00:00+02:00",
                "size": 1,
                "details": {"family": "qwen2", "parameter_size": "7.6B", "quantization_level": "Q4_0"},
            },
            {
                "name": "mistral:7b-instruct",
                "digest": f6l.MODEL_DIGESTS["mistral:7b-instruct"],
                "modified_at": "2026-08-01T00:00:00+02:00",
                "size": 1,
                "details": {"family": "llama", "parameter_size": "7.2B", "quantization_level": "Q4_K_M"},
            },
        ]
    }

    preflight, model_rows, manifest = profile_exec.build_profile_preflight(args, ollama_payload=fake_ollama)

    assert all(row["status"] == "pass" for row in preflight)
    assert all(row["status"] == "pass" for row in model_rows)
    assert manifest["planned_requests"] == 4800
    assert manifest["counts_by_input_variant"] == {
        "profile_generation": 800,
        "history_only": 800,
        "profile_only": 1600,
        "profile_plus_history": 1600,
    }
    assert manifest["inference_performed"] is False
    assert not (ROOT / "results" / "llm" / "phase_f6l_profile_execution_unit_preflight").exists()


def test_profile_title_list_parser_requires_exact_candidate_titles() -> None:
    candidate_ids = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    candidate_titles = [f"Item {index}" for index in range(1, 11)]
    response = repr(candidate_titles)

    parsed = profile_exec.parse_recommendation_response(response, candidate_ids, candidate_titles, "3")

    assert parsed["strict_valid"] is True
    assert parsed["hit_at_5"] == 1
    assert parsed["ndcg_at_5"] == pytest.approx(1 / 2)
    assert parsed["hallucination_count"] == 0

    invalid = profile_exec.parse_recommendation_response("['Not In Pool']", candidate_ids, candidate_titles, "3")
    assert invalid["strict_valid"] is False
    assert invalid["hallucination_count"] == 1


def test_profile_dependent_prompt_renderer_replaces_author_profile_only() -> None:
    prompt = "The Sports user A has been profiled as: 'old profile'. Please select the top 10 items from the list of candidates below. The candidates are: ['Item 1']."
    rendered = profile_exec.replace_profile_text(prompt, "new local profile")

    assert "new local profile" in rendered
    assert "old profile" not in rendered
    assert "The candidates are: ['Item 1']" in rendered


def test_profile_metric_differences_keep_profile_generators_disaggregated() -> None:
    rows = []
    for user_id in ["u1", "u2"]:
        base_row = {
            "dataset": "beauty",
            "user_id": user_id,
            "model_name": "qwen2:7b-instruct",
            "recommender_model_name": "qwen2:7b-instruct",
            "candidate_list_hash": "same_candidates",
            "ndcg_at_5": "0.000000",
        }
        rows.append({**base_row, "input_variant": "history_only", "profile_generator_name": "", "hit_at_5": "0"})
        for generator, hit in [("qwen2:7b-instruct", "1"), ("mistral:7b-instruct", "0")]:
            rows.append({**base_row, "input_variant": "profile_only", "profile_generator_name": generator, "hit_at_5": hit})
            rows.append({**base_row, "input_variant": "profile_plus_history", "profile_generator_name": generator, "hit_at_5": hit})

    diffs = profile_exec.aggregate_metric_differences("unit_run", rows)

    assert len(diffs) == 12
    assert {row["profile_generator"] for row in diffs} == {"qwen2:7b-instruct", "mistral:7b-instruct"}
    assert all(row["paired_users"] == "2" for row in diffs)
    assert sum(row["metric"] == "HR@5" for row in diffs) == 6


def test_separated_lastfm_operationalization_and_rq_hierarchy_are_non_exact() -> None:
    result = f6l.build_separated_contracts(CONFIG, write_outputs_enabled=False)
    counts = {row["source"]: row for row in result.tables["lastfm_preprocessing_count_comparison"]}
    op = result.tables["lastfm_operationalization_contract"]
    full = result.tables["lastfm_full_cohort_extension_plan"][0]
    hierarchy = {row["classification"]: row for row in result.tables["rq_evidence_hierarchy_contract"]}
    author_report = result.reports["author_clarification_request"]

    assert counts["paper"]["users"] == "1543"
    assert counts["paper"]["items"] == "7286"
    assert counts["paper"]["interactions"] == "173778"
    assert counts["coorte locale F6C da 1859 utenti"]["matches_paper_counts"] == "false"
    assert all(row["classification"] == "paper_protocol_reproduction_under_explicit_operationalization" for row in op)
    assert all(row["author_attribution"] == "local_operationalization_not_author_claim" for row in op)
    assert full["classification"] == "original_project_extension_full_cohort"
    assert full["planned_requests"] == "7436"
    assert hierarchy["paper_protocol_reproduction_under_explicit_operationalization"]["can_support_or_contradict_paper_rq_alone"] == "false"
    assert hierarchy["original_project_extension_full_cohort"]["can_support_or_contradict_paper_rq_alone"] == "false"
    for phrase in ["dataset LastFM preprocessato", "alpha", "sample dei 1000 utenti", "raw output LastFM"]:
        assert phrase in author_report


def test_no_paths_are_staged_by_f6l() -> None:
    completed = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == ""
