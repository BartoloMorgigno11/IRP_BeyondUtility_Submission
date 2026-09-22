from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.finalization_f3_canonical_scientific_consolidation import (
    CANONICAL_RESULTS_COLUMNS,
    CLAIM_REGISTRY_COLUMNS,
    MISSING_EVIDENCE_COLUMNS,
    METRIC_STATUS_COLUMNS,
    PAPER_ALIGNMENT_COLUMNS,
    PRIMARY_MATRIX_COLUMNS,
    VALID_COMPARISONS_COLUMNS,
    VALIDATION_COLUMNS,
    build_frames,
    final_decision,
    frame_sha256,
    load_sources,
    no_absolute_paths,
    output_hashes,
    write_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

pytestmark = [pytest.mark.cleanup_historical, pytest.mark.phase_snapshot]


@lru_cache(maxsize=1)
def frames() -> dict[str, pd.DataFrame]:
    return build_frames(PROJECT_ROOT)


def test_f3_output_schemas() -> None:
    built = frames()
    assert list(built["canonical_results"].columns) == CANONICAL_RESULTS_COLUMNS
    assert list(built["valid_comparisons"].columns) == VALID_COMPARISONS_COLUMNS
    assert list(built["primary_matrix"].columns) == PRIMARY_MATRIX_COLUMNS
    assert list(built["paper_alignment"].columns) == PAPER_ALIGNMENT_COLUMNS
    assert list(built["metric_status"].columns) == METRIC_STATUS_COLUMNS
    assert list(built["missing_evidence"].columns) == MISSING_EVIDENCE_COLUMNS
    assert list(built["claim_registry"].columns) == CLAIM_REGISTRY_COLUMNS
    assert list(built["validation"].columns) == VALIDATION_COLUMNS


def test_f3_is_deterministic_in_memory() -> None:
    first = {key: frame_sha256(value) for key, value in frames().items()}
    second = {key: frame_sha256(value) for key, value in frames().items()}
    assert first == second


def test_f3_output_hashes_are_stable_in_tmpdir(tmp_path: Path) -> None:
    built = frames()
    write_outputs(tmp_path, built, "temporary report\n")
    first = output_hashes(tmp_path)
    write_outputs(tmp_path, built, "temporary report\n")
    second = output_hashes(tmp_path)
    assert first == second


def test_f3_outputs_have_stable_ordering() -> None:
    built = frames()
    rebuilt = build_frames(PROJECT_ROOT)
    for key, frame in built.items():
        assert frame.to_dict("records") == rebuilt[key].to_dict("records")
    assert list(built["validation"]["check_id"]) == sorted(built["validation"]["check_id"])


def test_all_f2_rows_are_classified_and_ids_are_unique() -> None:
    bundle = load_sources(PROJECT_ROOT)
    result = frames()["canonical_results"]
    assert len(result) == 314
    assert len(result) == len(bundle.f2_registry)
    assert result["evidence_id"].is_unique
    assert result["canonical_result_id"].is_unique
    assert set(bundle.f2_registry["evidence_id"]) == set(result["evidence_id"])


def test_primary_selection_excludes_pilot_diagnostic_and_legacy() -> None:
    result = frames()["canonical_results"]
    primary = result[result["is_primary_result"].eq("true")]
    assert not primary["is_pilot"].eq("true").any()
    assert not primary["is_diagnostic"].eq("true").any()
    assert not primary["is_legacy"].eq("true").any()
    assert primary["is_reportable"].eq("true").all()


def test_mostpop_plan_b_and_plan_e_are_distinct_not_deduped() -> None:
    result = frames()["canonical_results"]
    rows = result[
        result["model"].eq("MostPop")
        & result["dataset"].eq("ML-1M")
        & result["metric"].eq("hr_at_5")
        & result["value"].isin(["0.320364238410596", "0.671"])
    ]
    assert len(rows) >= 2
    assert rows["comparable_group_id"].nunique() >= 2


def test_valid_comparisons_share_group_and_do_not_mix_strict_full() -> None:
    comparisons = frames()["valid_comparisons"]
    assert not comparisons.empty
    assert comparisons["is_valid_comparison"].eq("true").all()
    assert not comparisons["strict_or_full"].str.contains(";").any()
    assert comparisons["comparable_group_id"].str.startswith("F3-CMP-").all()


def test_ambiguous_metrics_are_not_consolidated() -> None:
    metric_status = frames()["metric_status"]
    ambiguous = metric_status[
        metric_status["metric_definition_id"].isin(["popularity_bias_at_5", "intra_list_diversity_at_5"])
    ]
    assert len(ambiguous) == 2
    assert ambiguous["is_scientifically_consolidated"].eq("false").all()
    assert ambiguous["blocking_issue"].str.len().gt(0).all()


def test_paper_alignment_preserves_exact_numeric_zero() -> None:
    paper = frames()["paper_alignment"]
    assert len(paper) >= 73
    assert paper["is_exact_numeric_replication"].eq("true").sum() == 0
    assert "exactly_reproduced" in set(paper["reproduction_class"])


def test_claim_registry_has_safe_and_unsafe_claims_with_evidence_policy() -> None:
    claims = frames()["claim_registry"]
    assert claims["is_safe_for_report"].eq("true").any()
    assert claims["is_safe_for_report"].eq("false").any()
    safe = claims[claims["is_safe_for_report"].eq("true")]
    assert safe["supporting_files"].str.len().gt(0).all()


def test_missing_evidence_covers_paper_values_and_metrics() -> None:
    missing = frames()["missing_evidence"]
    assert (missing["area"] == "paper_numeric_cell").any()
    assert (missing["area"] == "metric_definition").any()
    assert missing["missing_evidence_id"].is_unique


def test_validation_passes_and_decision_complete() -> None:
    validation = frames()["validation"]
    assert validation["status"].eq("pass").all()
    assert final_decision(validation) == "finalization_f3_complete"


def test_protected_input_invariance_is_validated() -> None:
    validation = frames()["validation"]
    protected = validation[validation["check_name"].eq("protected_inputs_unchanged")]
    assert len(protected) == 1
    row = protected.iloc[0]
    assert row["status"] == "pass"
    assert "checked=1144" in row["observed"]
    assert "changed=0" in row["observed"]


def test_no_absolute_paths_or_nan_in_final_csv_frames() -> None:
    for frame in frames().values():
        assert no_absolute_paths(frame)
        assert not frame.isna().any().any()
