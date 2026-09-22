from pathlib import Path

from src.analysis import reproduction_r5c_corrected_postrun_validation as postrun


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_postrun_comparison_counts_original_and_corrected_validity() -> None:
    original = postrun.read_csv(PROJECT_ROOT / postrun.ORIGINAL_STATUS)
    corrected = postrun.read_csv(PROJECT_ROOT / postrun.CORRECTED_STATUS)
    comparison = postrun.build_comparison(original, corrected)
    overall = comparison.set_index(["protocol", "scope", "key"])

    assert int(overall.loc[("R5C_ORIGINAL", "overall", "all"), "valid"]) == 2385
    assert int(overall.loc[("R5C_ORIGINAL", "overall", "all"), "out_of_range"]) == 15
    assert int(overall.loc[("R5C_CORRECTED", "overall", "all"), "valid"]) == 2400
    assert int(overall.loc[("R5C_CORRECTED", "overall", "all"), "out_of_range"]) == 0


def test_postrun_summary_marks_item_id_substitution_eliminated() -> None:
    summary = postrun.validate(PROJECT_ROOT, write=False)

    assert summary["llm_calls_in_postrun_validation"] == 0
    assert summary["valid"] == 2400
    assert summary["invalid"] == 0
    assert summary["qwen2_ml1m_valid"] == 300
    assert summary["qwen2_ml1m_accounted"] == 300
    assert summary["target_gaps_all_valid"] == 4
    assert summary["item_id_substitution_eliminated"] is True
