from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = PROJECT_ROOT / "results/tables/plan_d_paper_component_matrix.csv"
LEDGER_PATH = PROJECT_ROOT / "results/tables/plan_d_feasibility_ledger.csv"
NEXT_ACTIONS_PATH = PROJECT_ROOT / "results/tables/plan_d_next_actions_priority.csv"
REPORT_PATH = PROJECT_ROOT / "report/plan_d_reproducibility_matrix.md"
NOTES_PATH = PROJECT_ROOT / "report/notes.md"

MATRIX_COLUMNS = {
    "component_id",
    "category",
    "paper_component",
    "paper_intent",
    "project_status",
    "implementation_level",
    "project_artifacts",
    "evidence_files",
    "blocker_or_gap",
    "next_action",
    "priority",
    "notes",
}
LEDGER_COLUMNS = {
    "item_id",
    "area",
    "target",
    "feasibility_status",
    "required_resources",
    "estimated_effort",
    "risk_level",
    "external_dependency",
    "recommended_decision",
    "notes",
}
NEXT_ACTION_COLUMNS = {
    "priority_rank",
    "action_id",
    "action_title",
    "related_component",
    "why_it_matters",
    "expected_output",
    "prerequisites",
    "risk",
    "recommended_order",
    "notes",
}

ALLOWED_PROJECT_STATUSES = {
    "DONE",
    "PARTIAL",
    "DIAGNOSTIC_ONLY",
    "FEASIBLE",
    "CONDITIONAL",
    "BLOCKED",
    "EXTENSION",
    "NOT_ATTEMPTED",
}
ALLOWED_IMPLEMENTATION_LEVELS = {
    "exact",
    "reconstructed",
    "diagnostic",
    "planned",
    "blocked",
    "extension",
    "not_attempted",
}
ALLOWED_PRIORITIES = {"critical", "high", "medium", "low"}
ALLOWED_FEASIBILITY_STATUSES = {
    "feasible_now",
    "feasible_with_compute",
    "feasible_with_backend",
    "feasible_with_api_budget",
    "blocked_missing_artifacts",
    "blocked_missing_backend",
    "not_recommended",
    "already_done",
    "extended_available",
    "extension_only_no_valid_utility_evaluation",
}
ALLOWED_RISK_LEVELS = {"low", "medium", "high"}

REQUIRED_COMPONENTS = {
    "amazon_beauty",
    "amazon_sports",
    "ml1m",
    "lastfm",
    "mostpop",
    "bm25",
    "bprmf",
    "lightgcn",
    "sasrec",
    "gru4rec",
    "llm_ranking",
    "llm_reranking",
    "position_bias",
    "hallucination",
    "api_availability",
    "local_llm_backend",
    "final_report",
}


def test_plan_d_matrix_csvs_exist_with_required_columns_and_rows() -> None:
    assert REPORT_PATH.exists(), f"missing {REPORT_PATH}"
    assert NOTES_PATH.exists(), f"missing {NOTES_PATH}"
    for path, required_columns in (
        (MATRIX_PATH, MATRIX_COLUMNS),
        (LEDGER_PATH, LEDGER_COLUMNS),
        (NEXT_ACTIONS_PATH, NEXT_ACTION_COLUMNS),
    ):
        assert path.exists(), f"missing {path}"
        frame = pd.read_csv(path, dtype=str)
        assert not frame.empty, f"{path} must not be empty"
        assert required_columns.issubset(frame.columns)


def test_plan_d_matrix_uses_only_allowed_status_values() -> None:
    matrix = pd.read_csv(MATRIX_PATH, dtype=str).fillna("")
    ledger = pd.read_csv(LEDGER_PATH, dtype=str).fillna("")

    assert set(matrix["project_status"]).issubset(ALLOWED_PROJECT_STATUSES)
    assert set(matrix["implementation_level"]).issubset(ALLOWED_IMPLEMENTATION_LEVELS)
    assert set(matrix["priority"]).issubset(ALLOWED_PRIORITIES)
    assert set(ledger["feasibility_status"]).issubset(ALLOWED_FEASIBILITY_STATUSES)
    assert set(ledger["risk_level"]).issubset(ALLOWED_RISK_LEVELS)


def test_required_plan_d_reproduction_components_are_represented() -> None:
    matrix = pd.read_csv(MATRIX_PATH, dtype=str)
    component_ids = set(matrix["component_id"])

    assert REQUIRED_COMPONENTS.issubset(component_ids)


def test_missing_official_artifacts_are_not_labeled_done_or_exact() -> None:
    matrix = pd.read_csv(MATRIX_PATH, dtype=str)
    row = _component_row(matrix, "official_preprocessed_dataset_artifacts")

    assert row["project_status"] != "DONE"
    assert row["implementation_level"] != "exact"
    assert row["project_status"] == "BLOCKED"


def test_neural_baselines_record_extended_subset_coverage() -> None:
    matrix = pd.read_csv(MATRIX_PATH, dtype=str)
    ledger = pd.read_csv(LEDGER_PATH, dtype=str).fillna("")

    for model in ("bprmf", "lightgcn", "sasrec", "gru4rec"):
        row = _component_row(matrix, model)
        assert row["project_status"] == "DONE"
        assert row["implementation_level"] == "reconstructed"
        assert "diagnostic_subset" in row["notes"]
        assert "extended_subset" in row["notes"]

        ledger_rows = ledger.loc[ledger["item_id"] == f"{model}_extended_baseline"]
        assert len(ledger_rows) == 1
        ledger_row = ledger_rows.iloc[0]
        assert ledger_row["feasibility_status"] == "extended_available"
        assert "ml1m" in ledger_row["notes"]
        assert "amazon_beauty" in ledger_row["notes"]
        assert "amazon_sports" in ledger_row["notes"]


def test_lastfm_status_and_local_llm_backend_are_not_done() -> None:
    matrix = pd.read_csv(MATRIX_PATH, dtype=str)
    ledger = pd.read_csv(LEDGER_PATH, dtype=str).fillna("")

    lastfm = _component_row(matrix, "lastfm")
    assert lastfm["project_status"] != "DONE"
    assert "extension_only_no_valid_utility_evaluation" in lastfm["notes"]
    assert "no fake utility metrics" in lastfm["notes"]

    lastfm_ledger = ledger.loc[ledger["item_id"] == "lastfm_full_eval"].iloc[0]
    assert lastfm_ledger["feasibility_status"] == "extension_only_no_valid_utility_evaluation"
    assert _component_row(matrix, "local_llm_backend")["project_status"] != "DONE"


def test_next_actions_prioritize_llm_backend_path_and_final_report_not_extended_runs() -> None:
    actions = pd.read_csv(NEXT_ACTIONS_PATH, dtype=str).fillna("")
    ledger = pd.read_csv(LEDGER_PATH, dtype=str).fillna("")
    action_ids = set(actions["action_id"])
    action_text = " ".join(actions["action_id"].tolist() + actions["action_title"].tolist()).lower()

    assert "review_plan_d_final_reproducibility_package" in action_ids
    assert "prepare_exam_presentation_tables" in action_ids
    assert "optional_openai_api_diagnostic_if_budget_approved" in action_ids
    assert "final_report" in action_ids
    assert ledger.loc[ledger["item_id"] == "history_length_sensitivity"].iloc[0]["feasibility_status"] == "already_done"
    assert ledger.loc[ledger["item_id"] == "plan_d_final_reproducibility_package"].iloc[0]["feasibility_status"] == "already_done"
    assert "update_reproducibility_matrix_after_extended_baselines" not in action_ids
    assert "run_extended_neural_baselines" not in action_ids
    assert "extended neural baselines as pending" not in action_text


def test_report_and_notes_document_global_summary_and_lastfm_exclusion() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")
    notes = NOTES_PATH.read_text(encoding="utf-8")

    for text in (report, notes):
        assert "plan_d_global_extended_neural_baselines_summary.csv" in text
        assert "LastFM" in text
        assert "No fake LastFM utility metrics" in text
        assert "SASRec > GRU4Rec > LightGCN > BPRMF" in text
        assert "GRU4Rec > LightGCN > BPRMF > SASRec" in text


def _component_row(matrix: pd.DataFrame, component_id: str) -> pd.Series:
    rows = matrix.loc[matrix["component_id"] == component_id]
    assert len(rows) == 1, f"expected one matrix row for {component_id}"
    return rows.iloc[0]
