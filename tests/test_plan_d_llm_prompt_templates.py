from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from src.llm.plan_d_llm_ranking_parser import parse_llm_ranking_response
from src.llm.plan_d_prompt_templates import build_ranking_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "results/tables/plan_d_llm_prompt_template_registry.csv"


def test_prompt_template_registry_contains_required_rows() -> None:
    registry = pd.read_csv(REGISTRY_PATH, dtype=str).fillna("")
    strategies = set(registry["prompt_strategy"])

    assert {"openp5_base", "llmrank_recency", "llmsrec_icl_stub"}.issubset(strategies)
    openp5 = registry.loc[registry["prompt_strategy"] == "openp5_base"].iloc[0]
    assert openp5["status"] == "implemented"
    assert openp5["uses_user_history"] == "true"
    assert openp5["uses_candidate_list"] == "true"
    assert openp5["output_format"] == "strict_json"


def test_openp5_prompt_contains_candidates_and_strict_json_rules() -> None:
    prompt = build_ranking_prompt(
        user_id="u1",
        history_items=[
            {"item_id": "h1", "title": "Toy Story", "metadata": "Animation"},
            {"item_id": "h2", "title": "Star Wars", "metadata": "Sci-Fi"},
        ],
        candidate_items=[
            {"item_id": "item_A", "title": "Space Adventure", "metadata": "Sci-Fi"},
            {"item_id": "item_B", "title": "Romantic Drama", "metadata": "Drama"},
            {"item_id": "item_C", "title": "Family Adventure", "metadata": "Animation"},
            {"item_id": "item_D", "title": "Documentary", "metadata": "Documentary"},
            {"item_id": "item_E", "title": "Cyberpunk Thriller", "metadata": "Thriller"},
        ],
        top_k=5,
        prompt_strategy="openp5_base",
    )

    for candidate_id in ("item_A", "item_B", "item_C", "item_D", "item_E"):
        assert candidate_id in prompt
    assert "Return only valid JSON." in prompt
    assert "Rank exactly 5 item IDs." in prompt
    assert "Use only candidate IDs" in prompt
    assert "Do not include explanations." in prompt
    assert '"ranked_item_ids"' in prompt
    assert '"status": "ok"' in prompt


def test_ranking_parser_handles_valid_json() -> None:
    result = parse_llm_ranking_response(
        '{"ranked_item_ids": ["A", "B", "C"], "status": "ok"}',
        candidate_ids={"A", "B", "C", "D"},
        expected_top_k=3,
    )

    assert result.status == "parsed_valid"
    assert result.parse_success is True
    assert result.schema_valid is True
    assert result.candidate_validity_rate == 1.0


def test_ranking_parser_handles_malformed_json_and_empty_response() -> None:
    malformed = parse_llm_ranking_response("{not json", candidate_ids={"A"}, expected_top_k=1)
    empty = parse_llm_ranking_response("", candidate_ids={"A"}, expected_top_k=1)

    assert malformed.status == "parse_failed"
    assert malformed.parse_success is False
    assert empty.status == "empty_response"
    assert empty.selected_count == 0


def test_ranking_parser_detects_schema_errors() -> None:
    missing = parse_llm_ranking_response('{"status": "ok"}', candidate_ids={"A"}, expected_top_k=1)
    non_list = parse_llm_ranking_response(
        '{"ranked_item_ids": "A", "status": "ok"}',
        candidate_ids={"A"},
        expected_top_k=1,
    )
    fewer = parse_llm_ranking_response(
        '{"ranked_item_ids": ["A"], "status": "ok"}',
        candidate_ids={"A", "B"},
        expected_top_k=2,
    )
    more = parse_llm_ranking_response(
        '{"ranked_item_ids": ["A", "B", "C"], "status": "ok"}',
        candidate_ids={"A", "B", "C"},
        expected_top_k=2,
    )

    assert missing.status == "parsed_invalid_schema"
    assert non_list.status == "parsed_invalid_schema"
    assert fewer.status == "parsed_invalid_schema"
    assert more.status == "parsed_invalid_schema"


def test_ranking_parser_detects_hallucinated_ids_and_duplicates() -> None:
    hallucinated = parse_llm_ranking_response(
        '{"ranked_item_ids": ["A", "Z"], "status": "ok"}',
        candidate_ids={"A", "B"},
        expected_top_k=2,
    )
    duplicated = parse_llm_ranking_response(
        '{"ranked_item_ids": ["A", "A"], "status": "ok"}',
        candidate_ids={"A", "B"},
        expected_top_k=2,
    )

    assert hallucinated.status == "parsed_with_hallucinations"
    assert hallucinated.invalid_item_ids == ["Z"]
    assert hallucinated.hallucinated_count == 1
    assert hallucinated.candidate_validity_rate == 0.5
    assert duplicated.status == "parsed_with_duplicates"
    assert duplicated.duplicate_item_ids == ["A"]
    assert duplicated.duplicate_count == 1


def test_prompt_template_tests_do_not_call_ollama_or_external_apis() -> None:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported_modules = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    string_constants = [
        node.value.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]

    assert not {"requests", "urllib", "subprocess", "socket", "openai", "anthropic"} & imported_modules
    assert not any(("ollama " + "run") in value for value in string_constants)
    assert not any(("localhost:" + "11434") in value for value in string_constants)
