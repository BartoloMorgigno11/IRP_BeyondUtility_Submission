from __future__ import annotations

import csv
from pathlib import Path

from src.analysis import extension_e1_metric_contract as e1


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _tokens(value: str) -> set[str]:
    return {token for token in value.split(";") if token}


def test_extension_e1_internal_registry_validates() -> None:
    e1.validate_rows()


def test_extension_e1_required_artifacts_exist_with_expected_columns() -> None:
    expected = {
        RESULTS_DIR / "extension_e1_metric_contract.csv": e1.METRIC_COLUMNS,
        RESULTS_DIR / "extension_e1_four_dataset_coverage.csv": e1.COVERAGE_COLUMNS,
        RESULTS_DIR / "extension_e1_fifth_dataset_requirements.csv": e1.REQUIREMENT_COLUMNS,
        RESULTS_DIR / "extension_e1_inference_requirements.csv": e1.INFERENCE_COLUMNS,
        RESULTS_DIR / "extension_e1_dataset_selection_scorecard_template.csv": e1.SCORECARD_COLUMNS,
    }

    for path, columns in expected.items():
        assert path.exists(), path
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == columns
            assert list(reader)

    assert (PROJECT_ROOT / "report" / "extension_e1_complete_metric_contract.md").exists()


def test_extension_e1_metric_contract_has_explicit_provenance_and_applicability() -> None:
    rows = _read_csv(RESULTS_DIR / "extension_e1_metric_contract.csv")

    required_metric_ids = {
        "hr_at_5",
        "ndcg_at_5",
        "self_information_at_5",
        "serendipity_at_5",
        "arp_at_5",
        "aplt_at_5",
        "popreo_at_5",
        "item_coverage_at_5",
        "oic_at_5",
        "shannon_entropy_at_5",
        "gini_at_5",
        "dpd_at_5",
        "jain_index_at_5",
        "hallucination_rate_at_5",
        "canddif_hr",
        "canddif_ndcg",
        "history_length_sensitivity",
        "generation_involved_profile_matrix",
    }
    assert required_metric_ids.issubset({row["metric_id"] for row in rows})

    for row in rows:
        provenance = _tokens(row["provenance_classification"])
        inference_flags = _tokens(row["new_inference_requirement"])

        assert provenance
        assert provenance.issubset(e1.PROVENANCE_CLASSES)
        assert row["task_applicability"] in e1.TASK_VALUES
        assert row["recommender_applicability"] in e1.RECOMMENDER_VALUES
        assert row["fifth_dataset_requirement_classification"] in e1.REQUIREMENT_CLASSES
        assert inference_flags
        assert inference_flags.issubset(e1.INFERENCE_FLAGS)


def test_extension_e1_paper_metrics_are_distinguished_from_local_extensions() -> None:
    rows = _read_csv(RESULTS_DIR / "extension_e1_metric_contract.csv")
    by_id = {row["metric_id"]: row for row in rows}

    for row in rows:
        provenance = _tokens(row["provenance_classification"])
        if row["family"] == "local_extension":
            assert "PAPER_SPECIFIED" not in provenance

    assert by_id["intra_list_diversity_at_5"]["family"] == "local_extension"
    assert "LOCAL_ADAPTED" in _tokens(by_id["intra_list_diversity_at_5"]["provenance_classification"])
    assert "PAPER_SPECIFIED" not in _tokens(by_id["intra_list_diversity_at_5"]["provenance_classification"])

    shannon = by_id["shannon_entropy_at_5"]
    assert "PAPER_UNDERSPECIFIED" in _tokens(shannon["provenance_classification"])
    assert "OFFICIAL_CODE_RECOVERED" in _tokens(shannon["provenance_classification"])
    assert shannon["implementation_status"] == "MISSING"


def test_extension_e1_coverage_requirements_and_inference_rows_are_complete() -> None:
    coverage = _read_csv(RESULTS_DIR / "extension_e1_four_dataset_coverage.csv")
    requirements = _read_csv(RESULTS_DIR / "extension_e1_fifth_dataset_requirements.csv")
    inference = _read_csv(RESULTS_DIR / "extension_e1_inference_requirements.csv")

    covered_ids = {row["metric_or_analysis_id"] for row in coverage}
    assert {
        "hr_at_5",
        "ndcg_at_5",
        "popreo_at_5",
        "oic_at_5",
        "shannon_entropy_at_5",
        "gini_at_5",
        "dpd_at_5",
        "jain_index_at_5",
    }.issubset(covered_ids)

    for row in coverage:
        for dataset in ["beauty", "sports", "ml_1m", "lastfm"]:
            assert row[dataset] in e1.COVERAGE_VALUES

    for row in requirements:
        assert row["needed_by_metric_analysis"]
        assert row["requirement_class"] in e1.REQUIREMENT_CLASSES

    for row in inference:
        assert _tokens(row["inference_requirement"]).issubset(e1.INFERENCE_FLAGS)
        assert row["extra_llm_inference"]
