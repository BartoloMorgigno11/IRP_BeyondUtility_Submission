"""Focused tests for F6L LastFM parser forensics."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from src.analysis import phase_f6l_lastfm_parser_forensics as forensic
from src.analysis import plan_f6l_paper_faithful_completion_and_rq_revalidation as f6l


def candidate_rows() -> list[dict[str, str]]:
    titles = [
        "Massive Attack",
        "Lamb",
        "65daysofstatic",
        "Emilie Autumn",
        "The Crystal Method",
        "Stephen Jerzak",
        "Adema",
        "808 State",
        "Jimi Hendrix",
        "Duran Duran",
        "John Zorn",
    ]
    return [
        {"candidate_item_id": str(index), "candidate_item_title": title}
        for index, title in enumerate(titles, start=1)
    ]


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_runtime_and_no_api_flags_required() -> None:
    assert Path(sys.executable).resolve() == f6l.RECOVERED_INTERPRETER.resolve()
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    assert forensic.PARSER_V2_VERSION.endswith("_v2")


def test_v1_rejects_numbered_text_while_v2_recovers() -> None:
    text = "\n".join(
        [
            "Here are my picks:",
            "1. **Massive Attack** - similar style.",
            "2. Lamb - electronic/folk.",
            "3. 65daysofstatic",
            "4. Emilie Autumn",
            "5. The Crystal Method",
            "6. Stephen Jerzak",
            "7. Adema",
            "8. 808 State",
            "9. Jimi Hendrix",
            "10. Duran Duran",
        ]
    )
    v1 = f6l.parse_paper_title_list_response(text, candidate_rows(), expected_count=10)
    v2 = forensic.parse_v2_response(text, candidate_rows(), expected_count=10)
    assert not v1["strict_valid"]
    assert v1["failure_reason"] == "missing_or_multiple_bracketed_lists"
    assert v2.conservative_valid
    assert v2.extraction_rule == "numbered_or_bullet_candidate_rank_top10"
    assert v2.ranked_item_ids[:2] == ["1", "2"]


def test_v2_rejects_hallucinated_top10_item() -> None:
    text = "\n".join(
        [
            "1. Massive Attack",
            "2. Lamb",
            "3. Not A Candidate",
            "4. Emilie Autumn",
            "5. The Crystal Method",
            "6. Stephen Jerzak",
            "7. Adema",
            "8. 808 State",
            "9. Jimi Hendrix",
            "10. Duran Duran",
        ]
    )
    result = forensic.parse_v2_response(text, candidate_rows(), expected_count=10)
    assert not result.conservative_valid
    assert result.failure_reason == "hallucinated_or_unmapped_top10_item"


def test_v2_rejects_duplicate_top10_item() -> None:
    text = "\n".join(
        [
            "1. Massive Attack",
            "2. Massive Attack",
            "3. Lamb",
            "4. Emilie Autumn",
            "5. The Crystal Method",
            "6. Stephen Jerzak",
            "7. Adema",
            "8. 808 State",
            "9. Jimi Hendrix",
            "10. Duran Duran",
        ]
    )
    result = forensic.parse_v2_response(text, candidate_rows(), expected_count=10)
    assert not result.conservative_valid
    assert result.failure_reason == "duplicate_candidate_in_top10"


def test_audit_sample_selection_is_deterministic_and_blind_to_relevance() -> None:
    rows = [
        {"request_id": f"r{index}", "model_name": "m", "prompting_strategy": "s", "v1_failure_category": "c"}
        for index in range(20)
    ]
    first = forensic.select_audit_sample(rows, max_per_cell_category=4, max_total=10)
    second = forensic.select_audit_sample(rows, max_per_cell_category=4, max_total=10)
    assert first == second
    assert len(first) == 4
    assert all("positive_item" not in row and "ndcg" not in row for row in rows)
