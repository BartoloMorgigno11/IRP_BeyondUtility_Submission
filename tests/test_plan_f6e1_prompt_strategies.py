from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.llm import plan_f6e1_prompt_strategies as f6e1


ROOT = Path.cwd()
EXPECTED_RAW_HASHES = {
    Path("results/llm/finalization_f6d4a_lastfm_ranking_qwen2_raw.jsonl"): "1384a47058ed8dd8c06fc52d6bc425020b1746b60ad11e4bde6153f948a493b4",
    Path("results/llm/finalization_f6d4a_lastfm_ranking_mistral_raw.jsonl"): "7031f37d40092b4f318b5e03273772531ffda37cef0c38bfed32edfef935736a",
}


def _context() -> f6e1.PromptRenderContext:
    return f6e1.PromptRenderContext(
        dataset="Amazon Beauty",
        user_name="user-001",
        user_history_titles=("History 01", "History 02", "History 03"),
        candidate_titles=tuple(f"Candidate {index:02d}" for index in range(1, 21)),
    )


def test_module_import_has_no_side_effects_and_no_llm_or_network_imports():
    before = f6e1.output_hashes(ROOT)
    source = Path("src/llm/plan_f6e1_prompt_strategies.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert {"urllib", "requests", "httpx", "openai", "anthropic", "ollama", "socket"}.isdisjoint(imports)
    assert before == f6e1.output_hashes(ROOT)


def test_source_inventory_contains_required_prompt_and_parser_sources():
    rows = f6e1.build_prompt_source_inventory(ROOT)
    paths = {row["relative_path"] for row in rows}
    assert "original_repo/EvaLLMasRecommender/eva/templates/template.txt" in paths
    assert "original_repo/EvaLLMasRecommender/eva/templates/llmrank_rf.txt" in paths
    assert "original_repo/EvaLLMasRecommender/eva/templates/template_in_context.txt" in paths
    assert "src/llm/canonical_llm_contract.py" in paths
    assert "paper/traduzione_italiana_beyond_utility_llm_recommender.docx" in paths
    assert all(row["sha256"] != "missing" for row in rows if row["relative_path"] != "paper/2411.00331v1.pdf")


def test_strategy_classifications_are_single_label_and_scientifically_separated():
    rows = {row["strategy_id"]: row for row in f6e1.build_prompt_strategy_spec(ROOT)}
    assert rows[f6e1.BASE_STRATEGY_ID]["classification"] == "exact_parametric"
    assert rows[f6e1.LLMRANK_STRATEGY_ID]["classification"] == "exact_parametric"
    assert rows[f6e1.LLMSREC_STRATEGY_ID]["classification"] == "reconstructed"
    assert rows[f6e1.JSON_INDEX_STRATEGY_ID]["classification"] == "local_ablation"
    assert "not_exact_paper_prompt" in rows[f6e1.JSON_INDEX_STRATEGY_ID]["paper_alignment"]


def test_official_prompt_rendering_is_deterministic_and_preserves_history_and_candidate_order():
    context = _context()
    first = f6e1.render_strategy_prompt(f6e1.BASE_STRATEGY_ID, context, root=ROOT)
    second = f6e1.render_strategy_prompt(f6e1.BASE_STRATEGY_ID, context, root=ROOT)
    assert first == second
    assert f6e1.sha256_text(first) == f6e1.sha256_text(second)
    assert first.index("History 01") < first.index("History 02") < first.index("History 03")
    assert first.index("Candidate 01") < first.index("Candidate 20")


def test_llmrank_recency_rendering_keeps_last_item_and_literal_newline_policy():
    prompt = f6e1.render_strategy_prompt(f6e1.LLMRANK_STRATEGY_ID, _context(), root=ROOT)
    assert "most recently purchased product" in prompt
    assert "History 03" in prompt
    assert "\\n" in prompt
    assert prompt.index("Candidate 01") < prompt.index("Candidate 20")


def test_llmsrec_reconstructed_prompt_has_in_context_history_components_and_blocked_full_pipeline():
    prompt = f6e1.render_strategy_prompt(f6e1.LLMSREC_STRATEGY_ID, _context(), root=ROOT)
    assert "History 01" in prompt
    assert "History 03" in prompt
    assert "Candidate 01" in prompt
    with pytest.raises(f6e1.BlockedPromptStrategyError):
        f6e1.render_strategy_prompt(f6e1.BLOCKED_FULL_LLMSREC_ID, _context(), root=ROOT)


def test_json_index_local_ablation_prompt_and_parser_contracts():
    rendered = f6e1.render_json_index_prompt(context=_context(), task_type="ranking", top_k=5)
    assert rendered["system_message"].startswith("You are a recommender-system evaluator")
    assert "Return exactly 5 unique candidate indices" in rendered["user_message"]
    assert "1-based" in rendered["user_message"]
    assert rendered["user_message"].index("1. Candidate 01") < rendered["user_message"].index("20. Candidate 20")

    valid = f6e1.validate_json_index_output(
        '{"ranked_candidate_indices":[1,2,3,4,5]}',
        candidate_count=20,
        expected_count=5,
    )
    duplicate = f6e1.validate_json_index_output(
        '{"ranked_candidate_indices":[1,1,3,4,5]}',
        candidate_count=20,
        expected_count=5,
    )
    unknown = f6e1.validate_json_index_output(
        '{"ranked_candidate_indices":[1,2,3,4,21]}',
        candidate_count=20,
        expected_count=5,
    )
    short = f6e1.validate_json_index_output(
        '{"ranked_candidate_indices":[1,2,3,4]}',
        candidate_count=20,
        expected_count=5,
    )
    not_full = f6e1.validate_json_index_output(
        '{"ranked_candidate_indices":[1,2,3,4,5]}',
        candidate_count=20,
        expected_count=5,
        require_full_permutation=True,
    )
    assert valid["strict_parse_valid"] is True
    assert duplicate["status"] == "duplicate_indices"
    assert unknown["status"] == "unknown_indices"
    assert short["status"] == "invalid_length"
    assert not_full["status"] == "missing_permutation_indices"
    assert valid["index_base"] == 1


def test_placeholder_contract_covers_all_canonical_placeholders_per_strategy():
    rows = f6e1.build_placeholder_contract(ROOT)
    strategy_ids = {row["strategy_id"] for row in f6e1.build_prompt_strategy_spec(ROOT)}
    assert len(rows) == len(strategy_ids) * len(f6e1.CANONICAL_PLACEHOLDERS)
    for strategy_id in strategy_ids:
        names = {row["placeholder_name"] for row in rows if row["strategy_id"] == strategy_id}
        assert names == set(f6e1.CANONICAL_PLACEHOLDERS)
    json_rows = [row for row in rows if row["strategy_id"] == f6e1.JSON_INDEX_STRATEGY_ID]
    candidate_indices = [row for row in json_rows if row["placeholder_name"] == "candidate_indices"][0]
    assert candidate_indices["serialization"] == "JSON array one-based integers"


def test_parser_contract_distinguishes_strict_validity_from_hallucination():
    rows = f6e1.build_parser_contract(ROOT)
    local_rows = [row for row in rows if row["strategy_id"] == f6e1.JSON_INDEX_STRATEGY_ID]
    official_rows = [row for row in rows if row["strategy_id"] == f6e1.BASE_STRATEGY_ID]
    assert official_rows[0]["json_required"] == "false"
    assert any(row["index_base"] == "1" and row["allows_duplicates"] == "false" for row in local_rows)
    for row in rows:
        assert row["strict_validity_definition"] != row["candidate_hallucination_definition"]


def test_model_strategy_matrix_has_required_core_extension_and_ablation_cells():
    rows = f6e1.build_model_strategy_matrix(ROOT)
    assert len(rows) == 11
    keys = {(row["model_family"], row["strategy_id"], row["scope"]) for row in rows}
    assert ("Qwen2 7B Instruct", f6e1.BASE_STRATEGY_ID, "paper_aligned_core") in keys
    assert ("Mistral 7B Instruct", f6e1.LLMRANK_STRATEGY_ID, "cross_model_extension") in keys
    assert ("Qwen2 7B Instruct", f6e1.JSON_INDEX_STRATEGY_ID, "local_ablation") in keys
    llama_llmsrec = [
        row
        for row in rows
        if row["model_family"] == "Llama3-70B-Instruct" and row["strategy_id"] == f6e1.LLMSREC_STRATEGY_ID
    ][0]
    assert "blocked_hardware" in llama_llmsrec["local_feasibility"]


def test_builder_outputs_are_deterministic_in_tempdir(tmp_path):
    first = f6e1.build_all(ROOT, output_root=tmp_path)
    first_hashes = f6e1.output_hashes(tmp_path)
    second = f6e1.build_all(ROOT, output_root=tmp_path)
    second_hashes = f6e1.output_hashes(tmp_path)
    assert first["decision"] == "f6e1_prompt_strategy_freeze_complete_with_documented_blocks"
    assert first["manifest"]["scientific_content_hash"] == second["manifest"]["scientific_content_hash"]
    assert first_hashes == second_hashes
    assert set(first_hashes) == {
        "report/phase_f6e1_prompt_strategy_recovery.md",
        "results/manifests/phase_f6e1_prompt_freeze.json",
        "results/tables/phase_f6e1_prompt_source_inventory.csv",
        "results/tables/phase_f6e1_prompt_strategy_spec.csv",
        "results/tables/phase_f6e1_placeholder_contract.csv",
        "results/tables/phase_f6e1_parser_contract.csv",
        "results/tables/phase_f6e1_model_strategy_matrix.csv",
        "results/tables/phase_f6e1_missing_or_blocked.csv",
    }


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_f6d4a_raw_hashes_and_f2_protected_inputs_are_unchanged():
    for relative_path, expected_hash in EXPECTED_RAW_HASHES.items():
        path = ROOT / relative_path
        assert path.exists()
        assert f6e1.sha256_file(path) == expected_hash
    summary = f6e1.protected_input_summary(ROOT)
    assert summary["checked"] >= 1000
    assert summary["missing"] == 0
    assert summary["changed"] == 0
