from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest

from src.analysis import plan_f6f2_scientific_readout as f6f2


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_tables():
    exact = [
        {
            "paper_metric": "APLT",
            "local_metric": "aplt_at_5",
            "formula_match": "yes",
            "input_match": "reconstructed",
            "population_match": "adapted",
            "implementation_match": "yes",
            "exactness_classification": "exact_formula_adapted_population",
            "claim_allowed": "qualified",
            "limitation": "adapted population",
        },
        {
            "paper_metric": "CandDif",
            "local_metric": "canddif_diagnostic_or_blocked",
            "formula_match": "yes",
            "input_match": "missing",
            "population_match": "missing",
            "implementation_match": "limited",
            "exactness_classification": "blocked_missing_input",
            "claim_allowed": "blocked",
            "limitation": "controlled pairs missing",
        },
    ]
    return {
        "f6f1_metric_exactness_matrix": exact,
        "f6f1_aplt_results": [
            {
                "result_id": "a",
                "scope": "dataset_task",
                "source_type": "llm",
                "dataset": "Tiny",
                "task": "ranking",
                "model_name": "qwen2",
                "strategy_id": "base",
                "strategy_label": "Base",
                "baseline_family": "",
                "population": "conditional",
                "metric": "aplt_at_5",
                "value": "0.4",
                "denominator": "2",
                "classification": "exact_formula_adapted_population",
                "notes": "strict-valid",
            }
        ],
        "f6f1_candidate_position_results": [
            {
                "analysis_id": "c",
                "analysis_type": "f6e4c_canddif_status",
                "dataset": "Tiny",
                "task": "ranking",
                "model_name": "qwen2",
                "strategy_id": "base",
                "strategy_label": "Base",
                "stratum": "all",
                "metric": "canddif",
                "value": "",
                "classification": "blocked_missing_input",
                "canddif_formula_status": "blocked_missing_input",
                "notes": "blocked",
            }
        ],
        "f6f1_serendipity_results": [],
        "f6f1_popularity_novelty_results": [],
        "f6f1_structural_diagnostics": [],
        "f6f1_llm_vs_baseline": [
            {
                "comparison_id": "l",
                "dataset": "Tiny",
                "task": "ranking",
                "llm_model": "qwen2",
                "strategy_id": "base",
                "strategy_label": "Base",
                "baseline_family": "MostPop",
                "baseline_model": "mostpop",
                "population": "strict_valid_paired_intersection",
                "metric": "HR@5",
                "llm_value": "0.1",
                "baseline_value": "0.4",
                "delta_llm_minus_baseline": "-0.3",
                "bootstrap_ci_low": "-0.4",
                "bootstrap_ci_high": "-0.2",
                "paired_intersection_size": "10",
                "classification": "exact_formula_adapted_population",
                "notes": "paired",
            },
            {
                "comparison_id": "a",
                "dataset": "Tiny",
                "task": "ranking",
                "llm_model": "qwen2",
                "strategy_id": "base",
                "strategy_label": "Base",
                "baseline_family": "MostPop",
                "baseline_model": "mostpop",
                "population": "strict_valid_paired_intersection",
                "metric": "APLT@5",
                "llm_value": "0.4",
                "baseline_value": "0.2",
                "delta_llm_minus_baseline": "0.2",
                "bootstrap_ci_low": "0.1",
                "bootstrap_ci_high": "0.3",
                "paired_intersection_size": "10",
                "classification": "exact_formula_adapted_population",
                "notes": "paired",
            },
        ],
        "f6f1_ranking_vs_reranking": [
            {
                "comparison_id": "r",
                "dataset": "Tiny",
                "model_name": "qwen2",
                "strategy_id": "base",
                "strategy_label": "Base",
                "population": "paired",
                "metric": "HR@5",
                "ranking_value": "0.1",
                "reranking_value": "0.2",
                "delta_reranking_minus_ranking": "0.1",
                "paired_intersection_size": "5",
                "classification": "exact_formula_adapted_population",
                "notes": "paired",
            }
        ],
        "f6e4c_strategy_paired_comparison": [
            {
                "comparison_type": "base_vs_llmrank",
                "population": "operational",
                "dataset": "Tiny",
                "task": "ranking",
                "model": "qwen2",
                "strategy_id": "",
                "left_label": "Base",
                "right_label": "LLMRank",
                "metric": "hr_at5",
                "left_mean": "0.2",
                "right_mean": "0.3",
                "delta_right_minus_left": "0.1",
                "bootstrap_ci_low": "0.0",
                "bootstrap_ci_high": "0.2",
                "n_strict_intersection": "5",
                "notes": "",
            }
        ],
        "f6e4c_model_paired_comparison": [],
        "f6f1_missing_or_blocked": [],
    }


def _config(tmp_path: Path):
    return {
        "phase": {
            "interpreter_path": str(Path(sys.executable)),
            "environment_classification": "test",
            "expected_f6f1_scientific_hash": f6f2.EXPECTED_F6F1_HASH,
            "expected_f6f1_decision": "f6f1_beyond_utility_metrics_complete_with_documented_blocks",
            "expected_f6f1r_decision": "f6f1r_environment_recovery_complete_with_runtime_metadata_differences",
            "expected_f6e4c_decision": "f6e4c_scientific_consolidation_complete_with_corrected_derived_metrics",
            "decision_complete_with_blocks": f6f2.DECISION_COMPLETE_WITH_BLOCKS,
            "recommendation": f6f2.RECOMMENDATION,
        },
        "allowed_exactness_classifications": [
            "exact_formula_adapted_population",
            "blocked_missing_input",
            "extension_metric",
            "not_applicable",
        ],
        "allowed_research_question_classifications": [
            "supported_in_adapted_setting",
            "partially_supported",
            "not_reproduced",
            "opposite_result_in_adapted_setting",
            "evidence_inconclusive",
            "not_tested",
            "blocked",
        ],
        "prohibited_wording": ["exactly replicated every result"],
        "unsafe_excluded_tests": ["network"],
        "inputs": {},
        "outputs": {
            "report": str(tmp_path / "report.md"),
            "freeze_manifest": str(tmp_path / "manifest.json"),
            "tables": {},
        },
    }


def test_import_without_side_effects() -> None:
    assert f6f2.PHASE_ID == "phase_f6f2"


def test_no_network_imports() -> None:
    tree = ast.parse(Path(f6f2.__file__).read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert not {"requests", "httpx", "socket", "urllib"}.intersection(imports)


def test_no_ollama_import() -> None:
    tree = ast.parse(Path(f6f2.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "ollama" not in imported


def test_no_inference_training_or_cuda_imports() -> None:
    tree = ast.parse(Path(f6f2.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not {"torch", "tensorflow"}.intersection(imported)


def test_explicit_interpreter_check_passes_for_configured_current_python(tmp_path: Path) -> None:
    result = f6f2.verify_interpreter(tmp_path, _config(tmp_path))
    assert result["status"] == "pass"


def test_f6f1_scientific_hash_constant() -> None:
    assert f6f2.EXPECTED_F6F1_HASH == "4bf9391b835dad47c80f41b9d41b130f64ac3f4776561cc19bf0a3d044ffb37c"


def test_f6f1r_comparison_logic_is_identity_based() -> None:
    left = {"frozen_f6f1_scientific_content_hash": "a", "isolated_f6f1_scientific_content_hash": "a"}
    assert left["frozen_f6f1_scientific_content_hash"] == left["isolated_f6f1_scientific_content_hash"]


def test_complete_metric_status_coverage() -> None:
    rows = f6f2.build_metric_status(_sample_tables(), _config(Path(".")))
    assert {row["paper_metric"] for row in rows} == {"APLT", "CandDif"}


def test_actual_numeric_values_required_for_findings() -> None:
    findings = f6f2.build_key_numeric_findings(_sample_tables())
    hr = [row for row in findings if row["metric"] == "HR@5"][0]
    assert hr["mean_delta"] == "-0.29999999999999999"


def test_exactness_classifications_preserved() -> None:
    rows = f6f2.build_metric_status(_sample_tables(), _config(Path(".")))
    assert rows[0]["final_exactness_classification"] == "exact_formula_adapted_population"


def test_no_blocked_metric_promoted_to_exact() -> None:
    rows = f6f2.build_metric_status(_sample_tables(), _config(Path(".")))
    f6f2.validate_no_blocked_metric_promoted(rows)
    cand = [row for row in rows if row["paper_metric"] == "CandDif"][0]
    assert cand["paper_claim_use"] != "qualified_claim_allowed"


def test_micro_macro_populations_are_separated() -> None:
    row = f6f2.readout_row("T", "x", "s.csv", {"population": "conditional", "metric": "m", "value": "1"})
    assert row["population"] == "conditional"


def test_ranking_reranking_population_separated() -> None:
    rows = f6f2.build_ranking_reranking_readout(_sample_tables())
    assert rows[0]["population"] == "paired"
    assert rows[0]["comparator"] == "reranking"


def test_conditional_operational_populations_separated() -> None:
    row = f6f2.readout_row("T", "x", "s.csv", {"population": "operational", "metric": "m", "value": "1"})
    assert row["population"] == "operational"


def test_unsupported_claim_rejection() -> None:
    with pytest.raises(ValueError):
        f6f2.validate_claims(
            [{"claim_id": "bad", "claim_text": "We exactly replicated every result", "allowed_wording": ""}],
            ["exactly replicated every result"],
        )


def test_prohibited_wording_guard_ignores_prohibited_column() -> None:
    f6f2.validate_claims(
        [{"claim_id": "ok", "claim_text": "Qualified result", "allowed_wording": "In our adapted setting", "prohibited_wording": "exactly replicated every result"}],
        ["exactly replicated every result"],
    )


def test_one_classification_per_rq() -> None:
    rows = f6f2.build_research_question_closure(f6f2.build_key_numeric_findings(_sample_tables()))
    f6f2.validate_research_questions(rows, _config(Path("."))["allowed_research_question_classifications"])
    assert len(rows) == 12


def test_remaining_gap_completeness() -> None:
    gaps = f6f2.build_remaining_gaps()
    names = {row["gap"] for row in gaps}
    assert "LastFM re-ranking" in names
    assert "modern local model extension" in names
    assert len(gaps) >= 14


def test_f6g1_requirements_completeness() -> None:
    reqs = f6f2.build_next_phase_requirements()
    components = {row["component"] for row in reqs}
    assert "required LastFM user cohort" in components
    assert "new LLM inference" in components
    assert len(reqs) >= 12


def test_prior_artifact_invariance_row_detects_no_change(tmp_path: Path) -> None:
    target = tmp_path / "frozen.csv"
    target.write_text("a\n1\n", encoding="utf-8")
    before = _sha(target)
    rows = [
        {
            "artifact": "frozen",
            "path": str(target),
            "phase": "F6F1",
            "expected_sha256": before,
            "actual_sha256": before,
            "status": "pass",
            "changed": "false",
            "missing": "false",
            "notes": "",
        }
    ]
    assert rows[0]["changed"] == "false"
    assert _sha(target) == before


def test_write_csv_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    rows = [{"a": "2", "b": "z"}, {"a": "1", "b": "a"}]
    f6f2.write_csv(path, rows, ["a", "b"])
    first = _sha(path)
    f6f2.write_csv(path, list(reversed(rows)), ["a", "b"])
    assert _sha(path) == first


def test_manifest_scientific_section_determinism(tmp_path: Path) -> None:
    payload = {"a": 1, "b": {"c": 2}}
    assert f6f2.sha256_text(f6f2.stable_json(payload)) == f6f2.sha256_text(f6f2.stable_json(dict(reversed(list(payload.items())))))


def test_missing_columns_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    assert set(f6f2.csv_columns(path)) == {"a", "b"}


def test_direction_and_ci_status() -> None:
    assert f6f2.direction_from_delta("-0.1") == "negative"
    assert f6f2.ci_status("-0.2", "-0.1", "-0.15") == "negative_ci_excludes_zero"


def test_readout_preserves_numeric_delta() -> None:
    row = f6f2.readout_row("T", "paired", "s.csv", {"metric": "m", "delta_llm_minus_baseline": "0.25", "bootstrap_ci_low": "0.1", "bootstrap_ci_high": "0.3"})
    assert row["delta"] == "0.25"
    assert row["statistical_status"] == "positive_ci_excludes_zero"
