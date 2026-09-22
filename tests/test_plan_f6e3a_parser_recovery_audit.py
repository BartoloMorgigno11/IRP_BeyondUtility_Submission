from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from src.analysis import plan_f6e3a_parser_recovery_audit as f6e3a


def test_import_has_no_side_effects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    before = sorted(tmp_path.rglob("*"))
    importlib.reload(f6e3a)
    after = sorted(tmp_path.rglob("*"))
    assert before == after


def test_source_contains_no_network_or_generation_calls() -> None:
    source = Path(f6e3a.__file__).read_text(encoding="utf-8").lower()
    forbidden = ["urlopen", "import requests", "from requests", "api/generate", "api/version", "api/tags", "ollama pull"]
    assert not any(token in source for token in forbidden)


def test_frozen_raw_hashes_are_verified() -> None:
    hashes = f6e3a.verify_frozen_raw_hashes(Path.cwd())
    assert hashes[f6e3a.RAW_PATHS[f6e3a.QWEN2_MODEL].as_posix()] == f6e3a.EXPECTED_RAW_HASHES[f6e3a.RAW_PATHS[f6e3a.QWEN2_MODEL].as_posix()]
    assert hashes[f6e3a.RAW_PATHS[f6e3a.MISTRAL_MODEL].as_posix()] == f6e3a.EXPECTED_RAW_HASHES[f6e3a.RAW_PATHS[f6e3a.MISTRAL_MODEL].as_posix()]


def test_build_all_has_expected_schemas_and_counts() -> None:
    outputs = f6e3a.build_all(Path.cwd())
    assert outputs.manifest["phase"] == f6e3a.PHASE_ID
    assert outputs.manifest["no_new_requests_sent"] is True
    assert outputs.manifest["network_calls_made"] == 0
    assert len(outputs.tables["output_format_inventory"]) == 32
    assert len(outputs.tables["parser_ladder_results"]) == 32 * 4
    assert len(outputs.tables["cell_summary"]) == 4
    assert {row["parser_level"] for row in outputs.tables["parser_ladder_results"]} == {
        "current_local_strict",
        "upstream_exact_or_functional",
        "deterministic_normalized_title_match",
        "diagnostic_permissive_recovery",
    }


def test_candidate_title_pipe_serialization_is_detected_but_prompt_titles_recover() -> None:
    row = {
        "candidate_item_ids": "|".join(str(index) for index in range(1, 21)),
        "candidate_item_titles": "A|B|C with | pipe|" + "|".join(f"T{index}" for index in range(4, 21)),
        "candidate_count": "20",
        "prompt": "History ['x']. The candidates are: " + repr(["A", "B", "C with | pipe"] + [f"T{index}" for index in range(4, 21)]) + ".",
    }
    context = f6e3a.candidate_context(row)
    assert context.serialization_delimiter_collision is True
    assert len(context.prompt_titles) == 20
    assert context.title_source == "prompt_literal"


def test_upstream_adapter_accepts_python_title_list() -> None:
    titles = tuple(f"Candidate {index}" for index in range(1, 21))
    ids = tuple(f"I{index}" for index in range(1, 21))
    response = repr(list(titles[:10]))
    outcome = f6e3a.parse_with_level("upstream_exact_or_functional", response, ids, titles)
    assert outcome.parse_valid is True
    assert outcome.strict_valid is True
    assert outcome.matched_ids == ids[:10]


def test_numbered_reasoning_is_not_strict_but_permissive_recoverable() -> None:
    titles = tuple(f"Candidate {index}" for index in range(1, 21))
    ids = tuple(f"I{index}" for index in range(1, 21))
    response = "\n".join(f"{index}. **{title}** - explanation" for index, title in enumerate(titles[:10], start=1))
    current = f6e3a.parse_with_level("current_local_strict", response, ids, titles)
    normalized = f6e3a.parse_with_level("deterministic_normalized_title_match", response, ids, titles)
    permissive = f6e3a.parse_with_level("diagnostic_permissive_recovery", response, ids, titles)
    assert current.strict_valid is False
    assert normalized.strict_valid is False
    assert permissive.strict_valid is False
    assert permissive.recovered_valid is True


def test_conservative_normalization_handles_quotes_case_and_spaces() -> None:
    titles = tuple(f"Candidate {index}" for index in range(1, 21))
    ids = tuple(f"I{index}" for index in range(1, 21))
    response = "\n".join(f'{index}.   "{title.upper()}  "' for index, title in enumerate(titles[:10], start=1))
    outcome = f6e3a.parse_with_level("deterministic_normalized_title_match", response, ids, titles)
    assert outcome.strict_valid is True
    assert outcome.matched_ids == ids[:10]


def test_ambiguous_titles_are_not_promoted_to_strict() -> None:
    titles = ("Alpha!", "Alpha?") + tuple(f"Candidate {index}" for index in range(3, 21))
    ids = tuple(f"I{index}" for index in range(1, 21))
    response = "\n".join([f"{index}. Alpha" for index in range(1, 11)])
    outcome = f6e3a.parse_with_level("deterministic_normalized_title_match", response, ids, titles)
    assert outcome.strict_valid is False
    assert outcome.ambiguous_items


def test_genuine_out_of_candidate_item_remains_unmatched() -> None:
    titles = tuple(f"Candidate {index}" for index in range(1, 21))
    ids = tuple(f"I{index}" for index in range(1, 21))
    response = "\n".join([f"{index}. Candidate {index}" for index in range(1, 10)] + ["10. Outside Item"])
    outcome = f6e3a.parse_with_level("deterministic_normalized_title_match", response, ids, titles)
    assert outcome.strict_valid is False
    assert outcome.hallucination_count == 1
    assert "Outside Item" in outcome.unmatched_items


def test_incomplete_output_is_detected() -> None:
    titles = tuple(f"Candidate {index}" for index in range(1, 21))
    ids = tuple(f"I{index}" for index in range(1, 21))
    response = "\n".join(f"{index}. Candidate {index}" for index in range(1, 6))
    outcome = f6e3a.parse_with_level("current_local_strict", response, ids, titles)
    assert outcome.parse_valid is True
    assert outcome.strict_valid is False
    assert "incomplete_output_length" in outcome.failure_reason


def test_one_based_index_and_item_id_are_permissive_only() -> None:
    titles = tuple(f"Candidate {index}" for index in range(1, 21))
    ids = tuple(f"I{index}" for index in range(1, 21))
    response = "\n".join(["1. 1", "2. I2"] + [f"{index}. Candidate {index}" for index in range(3, 11)])
    normalized = f6e3a.parse_with_level("deterministic_normalized_title_match", response, ids, titles)
    permissive = f6e3a.parse_with_level("diagnostic_permissive_recovery", response, ids, titles)
    assert normalized.strict_valid is False
    assert permissive.recovered_valid is True


def test_no_fuzzy_promotion_to_strict() -> None:
    titles = ("Alpha Beta Gamma",) + tuple(f"Candidate {index}" for index in range(2, 21))
    ids = tuple(f"I{index}" for index in range(1, 21))
    response = "\n".join(["1. Alpha Beta"] + [f"{index}. Candidate {index}" for index in range(2, 11)])
    normalized = f6e3a.parse_with_level("deterministic_normalized_title_match", response, ids, titles)
    assert normalized.strict_valid is False
    assert normalized.unmatched_items[0] == "Alpha Beta"


def test_write_outputs_are_deterministic_and_scoped(tmp_path: Path) -> None:
    first = f6e3a.build_all(Path.cwd())
    second = f6e3a.build_all(Path.cwd())
    assert first.manifest["scientific_content_hash"] == second.manifest["scientific_content_hash"]
    h1 = f6e3a.write_outputs(first, tmp_path / "a")
    h2 = f6e3a.write_outputs(second, tmp_path / "b")
    assert h1.keys() == h2.keys()
    for path, digest in h1.items():
        assert digest == h2[path]
        assert path.startswith(("results/tables/phase_f6e3a_", "results/manifests/phase_f6e3a_", "report/phase_f6e3a_"))


def test_raw_invariance_after_tmp_write(tmp_path: Path) -> None:
    before = f6e3a.verify_frozen_raw_hashes(Path.cwd())
    f6e3a.write_outputs(f6e3a.build_all(Path.cwd()), tmp_path)
    after = f6e3a.verify_frozen_raw_hashes(Path.cwd())
    assert before == after


def test_final_decision_and_recommendation_are_allowed() -> None:
    outputs = f6e3a.build_all(Path.cwd())
    allowed_decisions = {
        "f6e3a_parser_recovery_complete_gate_passed",
        "f6e3a_parser_recovery_complete_gate_failed_model_outputs",
        "f6e3a_parser_recovery_complete_with_ambiguous_mapping_blocks",
        "f6e3a_blocked_candidate_data_inconsistency",
    }
    allowed_recommendations = {
        "proceed_to_full_run",
        "repeat_analysis_only_with_corrected_parser",
        "repeat_exact_prompt_pilot_required",
        "run_adapted_output_contract_pilot",
        "keep_exact_prompt_cells_blocked",
        "repair_candidate_metadata_first",
    }
    assert outputs.manifest["decision"] in allowed_decisions
    assert outputs.manifest["recommendation"] in allowed_recommendations
