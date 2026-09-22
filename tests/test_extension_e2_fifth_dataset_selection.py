from __future__ import annotations

import csv
from pathlib import Path

from src.analysis import extension_e2_fifth_dataset_selection as e2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_extension_e2_artifact_builder_validates() -> None:
    e2.validate_artifacts()


def test_extension_e2_required_artifacts_exist() -> None:
    expected = [
        PROJECT_ROOT / "report" / "extension_e2_fifth_dataset_selection.md",
        RESULTS_DIR / "extension_e2_dataset_candidates.csv",
        RESULTS_DIR / "extension_e2_metric_compatibility.csv",
        RESULTS_DIR / "extension_e2_model_compatibility.csv",
        RESULTS_DIR / "extension_e2_dataset_scorecard.csv",
        RESULTS_DIR / "extension_e2_dataset_hard_gates.csv",
        RESULTS_DIR / "extension_e2_dataset_pareto.csv",
        RESULTS_DIR / "extension_e2_dataset_compute_estimates.csv",
        RESULTS_DIR / "extension_e2_dataset_selection_decision.csv",
    ]
    for path in expected:
        assert path.exists(), path


def test_extension_e2_matrices_cover_frozen_e1_contract() -> None:
    metrics = _read_csv(RESULTS_DIR / "extension_e1_metric_contract.csv")
    requirements = _read_csv(RESULTS_DIR / "extension_e1_fifth_dataset_requirements.csv")
    scorecard = _read_csv(RESULTS_DIR / "extension_e1_dataset_selection_scorecard_template.csv")
    candidates = _read_csv(RESULTS_DIR / "extension_e2_dataset_candidates.csv")
    metric_rows = _read_csv(RESULTS_DIR / "extension_e2_metric_compatibility.csv")
    gate_rows = _read_csv(RESULTS_DIR / "extension_e2_dataset_hard_gates.csv")
    scorecard_rows = _read_csv(RESULTS_DIR / "extension_e2_dataset_scorecard.csv")
    model_rows = _read_csv(RESULTS_DIR / "extension_e2_model_compatibility.csv")

    assert len(metrics) == 29
    assert len(requirements) == 23
    assert len(scorecard) == 25
    assert len(candidates) == 6
    assert len(metric_rows) == 29 * 6
    assert len(gate_rows) == 23 * 6
    assert len(scorecard_rows) == 25 * 6
    assert len(model_rows) == len(e2.MODEL_NAMES) * 6

    assert {row["candidate_id"] for row in candidates} == set(e2.CANDIDATE_IDS)
    assert {row["metric_id"] for row in metric_rows} == {row["metric_id"] for row in metrics}
    assert {row["requirement"] for row in gate_rows} == {row["requirement"] for row in requirements}
    assert {row["criterion"] for row in scorecard_rows} == {row["criterion"] for row in scorecard}


def test_extension_e2_decision_is_single_primary_and_backup() -> None:
    decision = _read_csv(RESULTS_DIR / "extension_e2_dataset_selection_decision.csv")
    assert len(decision) == 1
    row = decision[0]

    assert row["decision_token"] == e2.DECISION_TOKEN
    assert row["primary_dataset_id"] == "foodcom"
    assert row["backup_dataset_id"] == "steam"
    assert row["primary_dataset_id"] != row["backup_dataset_id"]
    assert row["can_receive_complete_applicable_paper_local_reproduction_before_extensions"] == "YES"


def test_extension_e2_primary_has_no_fail_or_unknown_hard_gate() -> None:
    gate_rows = _read_csv(RESULTS_DIR / "extension_e2_dataset_hard_gates.csv")
    primary_rows = [row for row in gate_rows if row["candidate_id"] == e2.PRIMARY_DATASET]

    assert primary_rows
    assert not [row for row in primary_rows if row["gate_status"] in {"FAIL", "UNKNOWN"}]
    assert {row["gate_status"] for row in primary_rows}.issubset(e2.GATE_VALUES)


def test_extension_e2_no_missing_status_or_provenance_cells() -> None:
    checked_files = {
        "extension_e2_metric_compatibility.csv": ["compatibility_status", "rationale", "source_ids"],
        "extension_e2_model_compatibility.csv": ["compatibility_status", "rationale", "source_ids"],
        "extension_e2_dataset_scorecard.csv": ["rating", "evidence", "source_ids"],
        "extension_e2_dataset_hard_gates.csv": ["gate_status", "evidence", "source_ids"],
        "extension_e2_dataset_pareto.csv": ["pareto_class", "rationale", "source_ids"],
        "extension_e2_dataset_compute_estimates.csv": ["scale_classification", "compute_rationale", "source_ids"],
    }

    for filename, columns in checked_files.items():
        rows = _read_csv(RESULTS_DIR / filename)
        assert rows
        for row in rows:
            for column in columns:
                assert row[column], (filename, column, row)
