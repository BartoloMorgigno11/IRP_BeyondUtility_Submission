from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from src.analysis import plan_f6e4c_scientific_consolidation as f6e4c


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tables() -> dict[str, list[dict[str, str]]]:
    return f6e4c.build_all_tables(ROOT)


def test_import_is_offline_and_has_no_ollama_http_calls() -> None:
    source = (ROOT / "src/analysis/plan_f6e4c_scientific_consolidation.py").read_text(encoding="utf-8")
    forbidden = ["execute_ollama", "inspect_ollama", "urlopen", "/api/version", "/api/tags", "/api/generate"]
    assert all(token not in source for token in forbidden)


def test_reconstructs_full_request_level_matrix(tables: dict[str, list[dict[str, str]]]) -> None:
    rows = tables["request_level_metrics"]
    assert len(rows) == f6e4c.EXPECTED_REQUESTS
    assert len({row["request_id"] for row in rows}) == f6e4c.EXPECTED_REQUESTS
    assert sum(row["strict_valid"] == "true" for row in rows) == f6e4c.EXPECTED_STRICT_VALID
    assert sum(row["strict_valid"] != "true" for row in rows) == f6e4c.EXPECTED_INVALID
    assert all(row["transport_status"] == "success" for row in rows)


def test_invalid_reason_counts_are_exact(tables: dict[str, list[dict[str, str]]]) -> None:
    counts = Counter(row["invalid_reason"] for row in tables["invalid_output_audit"])
    assert counts == {
        "invalid_length": 1271,
        "extra_text_or_code_fence": 111,
        "duplicate_indices": 50,
        "json_parse_failed": 5,
    }


def test_non_strict_contract_sets_operational_zero_and_conditional_missing(tables: dict[str, list[dict[str, str]]]) -> None:
    invalid_rows = [row for row in tables["request_level_metrics"] if row["strict_valid"] != "true"]
    assert invalid_rows
    for row in invalid_rows:
        assert row["hr_at1_operational"] == "0"
        assert row["hr_at5_operational"] == "0"
        assert row["hr_at10_operational"] == "0"
        assert row["ndcg_at5_operational"] == "0"
        assert row["ndcg_at10_operational"] == "0"
        assert row["hr_at5_conditional"] == ""
        assert row["ndcg_at5_conditional"] == ""


def test_metric_contract_invariants_pass(tables: dict[str, list[dict[str, str]]]) -> None:
    assert tables["metric_contract"]
    assert all(row["status"] == "pass" for row in tables["metric_contract"])
    micro_rows = [row for row in tables["cell_summary"] if row["aggregation"] == "micro"]
    assert len(micro_rows) == 4
    for row in micro_rows:
        rate = float(row["strict_valid_rate"])
        for metric in ["hr_at1", "hr_at5", "hr_at10", "ndcg_at5", "ndcg_at10"]:
            operational = float(row[f"operational_{metric}"])
            conditional = float(row[f"conditional_{metric}"])
            assert abs(operational - conditional * rate) <= f6e4c.TOLERANCE
            assert operational <= conditional + f6e4c.TOLERANCE


def test_summaries_and_reconciliation_are_scientifically_separated(tables: dict[str, list[dict[str, str]]]) -> None:
    cell_rows = tables["cell_summary"]
    assert sum(row["aggregation"] == "micro" for row in cell_rows) == 4
    assert sum(row["aggregation"] == "macro_dataset_task" for row in cell_rows) == 4
    assert tables["dataset_task_summary"]
    assert any(row["status"] == "superseded" for row in tables["metric_reconciliation"])


def test_paired_comparisons_and_bootstrap_are_deterministic(tables: dict[str, list[dict[str, str]]]) -> None:
    strategy_rows = tables["strategy_paired_comparison"]
    model_rows = tables["model_paired_comparison"]
    assert strategy_rows
    assert model_rows
    assert all(row["bootstrap_resamples"] == str(f6e4c.BOOTSTRAP_RESAMPLES) for row in strategy_rows + model_rows)
    again = f6e4c.build_strategy_paired_comparison(tables["request_level_metrics"])
    assert f6e4c.table_hash(strategy_rows, f6e4c.table_columns("strategy_paired_comparison")) == f6e4c.table_hash(
        again, f6e4c.table_columns("strategy_paired_comparison")
    )


def test_strata_and_mistral_lastfm_audit_are_present(tables: dict[str, list[dict[str, str]]]) -> None:
    assert tables["history_length_analysis"]
    assert tables["candidate_position_analysis"]
    audit = tables["mistral_llmrank_lastfm_audit"][0]
    assert audit["logical_requests"] == "1859"
    assert int(audit["invalid"]) > 0
    assert audit["classified_cause"] in {"core_adapter_instruction_conflict", "mixed_causes"}


def test_exact_vs_adapted_and_claim_guards(tables: dict[str, list[dict[str, str]]]) -> None:
    by_phase = {row["phase"]: row for row in tables["exact_vs_adapted_evidence"]}
    assert by_phase["F6E3"]["sample_size"] == "32"
    assert by_phase["F6E3"]["strict_valid"] == "0"
    assert by_phase["F6E4B"]["sample_size"] == str(f6e4c.EXPECTED_REQUESTS)
    assert by_phase["F6E4B"]["strict_valid"] == str(f6e4c.EXPECTED_STRICT_VALID)
    prohibited = " ".join(row["prohibited_wording"] for row in tables["claim_registry"]).lower()
    assert "paper-exact" in prohibited
    assert "exactly reproduced" in prohibited


def test_input_integrity_and_manifest_contract(tables: dict[str, list[dict[str, str]]]) -> None:
    assert all(row["status"] == "pass" for row in tables["input_integrity"])
    decision = f6e4c.final_decision(tables["metric_reconciliation"])
    manifest = f6e4c.build_manifest(ROOT, tables, decision)
    assert manifest["request_level_rows"] == f6e4c.EXPECTED_REQUESTS
    assert manifest["strict_valid"] == f6e4c.EXPECTED_STRICT_VALID
    assert manifest["invalid_outputs"] == f6e4c.EXPECTED_INVALID
    assert manifest["protected_summary"]["changed"] == 0
    assert manifest["protected_summary"]["missing"] == 0
    assert manifest["recommended_next_action"] == "keep_current_matrix_frozen"
    assert json.loads(json.dumps(manifest, sort_keys=True))
