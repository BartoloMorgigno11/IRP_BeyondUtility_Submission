from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_d_final_aggregation import (
    BLOCKED_COLUMNS,
    EXPERIMENT_INDEX_COLUMNS,
    HYGIENE_COLUMNS,
    LLM_DIAGNOSTIC_COLUMNS,
    METRIC_SUMMARY_COLUMNS,
    PHASE,
    REPRO_STATUS_COLUMNS,
    FinalPackagePaths,
    build_blocked_items,
    build_experiment_index,
    build_hygiene_audit,
    build_llm_diagnostic_summary,
    build_metric_summary,
    build_report,
    build_reproducibility_status,
)


LOGGER = logging.getLogger(__name__)

RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORT_DIR = PROJECT_ROOT / "report"
NOTES_PATH = REPORT_DIR / "notes.md"
REPRO_MATRIX_PATH = REPORT_DIR / "plan_d_reproducibility_matrix.md"
TASK_ROADMAP_PATH = RESULTS_TABLES_DIR / "plan_d_llm_task_roadmap.csv"
NEXT_ACTIONS_PATH = RESULTS_TABLES_DIR / "plan_d_next_actions_priority.csv"
FEASIBILITY_LEDGER_PATH = RESULTS_TABLES_DIR / "plan_d_feasibility_ledger.csv"

TASK_ROADMAP_COLUMNS = [
    "phase",
    "task",
    "dataset",
    "model",
    "backend",
    "users",
    "candidate_pool_size",
    "prompt_strategy",
    "output_format",
    "metrics",
    "prerequisites",
    "status",
    "notes",
]

NEXT_ACTION_COLUMNS = [
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
]

LEDGER_COLUMNS = [
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
]


def build_package() -> dict[str, pd.DataFrame]:
    experiment_index = build_experiment_index()
    metric_summary = build_metric_summary()
    llm_summary = build_llm_diagnostic_summary()
    repro_status = build_reproducibility_status()
    blocked_items = build_blocked_items()
    hygiene = build_hygiene_audit()
    return {
        "experiment_index": experiment_index,
        "metric_summary": metric_summary,
        "llm_summary": llm_summary,
        "repro_status": repro_status,
        "blocked_items": blocked_items,
        "hygiene": hygiene,
    }


def write_package(package: dict[str, pd.DataFrame], paths: FinalPackagePaths | None = None) -> None:
    paths = paths or FinalPackagePaths()
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    package["experiment_index"].to_csv(paths.experiment_index, index=False, columns=EXPERIMENT_INDEX_COLUMNS)
    package["metric_summary"].to_csv(paths.metric_summary, index=False, columns=METRIC_SUMMARY_COLUMNS)
    package["llm_summary"].to_csv(paths.llm_diagnostic_summary, index=False, columns=LLM_DIAGNOSTIC_COLUMNS)
    package["repro_status"].to_csv(paths.reproducibility_status, index=False, columns=REPRO_STATUS_COLUMNS)
    package["blocked_items"].to_csv(paths.blocked_items, index=False, columns=BLOCKED_COLUMNS)
    package["hygiene"].to_csv(paths.hygiene_audit, index=False, columns=HYGIENE_COLUMNS)

    report = build_report(
        package["experiment_index"],
        package["metric_summary"],
        package["llm_summary"],
        package["repro_status"],
        package["blocked_items"],
        package["hygiene"],
    )
    paths.report.write_text(report, encoding="utf-8")
    _update_notes(package)
    _update_reproducibility_matrix()
    _update_task_roadmap()
    _update_next_actions()
    _update_feasibility_ledger()


def _update_notes(package: dict[str, pd.DataFrame]) -> None:
    llm_summary = package["llm_summary"]
    d5e = _summary_row(llm_summary, "D5E", "openp5_base")
    d5g = _summary_row(llm_summary, "D5G", "strict_candidate_id_prompt")
    d5k = _summary_row(llm_summary, "D5K", "qwen2_medium_history")
    section = f"""
## 2026-07-01 - Plan D D5L: Final Aggregation and Reproducibility Package

### Changes

- Built final Plan D aggregate tables, LLM diagnostic summary, blocked-items table, reproducibility status table, and repository hygiene audit.
- Created `report/plan_d_final_reproducibility_package.md`.
- Performed only non-destructive audit checks; no cleanup, deletion, inference, API call, download, model pull, training, or LastFM run occurred.

### Outputs Generated

- `results/tables/plan_d_final_experiment_index.csv`
- `results/tables/plan_d_final_metric_summary.csv`
- `results/tables/plan_d_final_llm_diagnostic_summary.csv`
- `results/tables/plan_d_final_reproducibility_status.csv`
- `results/tables/plan_d_final_blocked_items.csv`
- `results/tables/plan_d_repository_hygiene_audit.csv`
- `report/plan_d_final_reproducibility_package.md`

### Result

- D5E base prompt HR@5/NDCG@5: `{d5e['HR@5']}` / `{d5e['NDCG@5']}` with hallucination `{d5e['hallucination_rate']}`.
- D5G strict prompt HR@5/NDCG@5: `{d5g['HR@5']}` / `{d5g['NDCG@5']}` with hallucination `{d5g['hallucination_rate']}`.
- D5K Qwen2 medium-history HR@5/NDCG@5: `{d5k['HR@5']}` / `{d5k['NDCG@5']}`.

### Notes And Limitations

- D5L is aggregation/reporting and hygiene audit only.
- The project remains reconstructed reproduction plus diagnostic/extended local work, not exact reproduction and not full LLM reproduction.
"""
    _replace_or_append_section(NOTES_PATH, "## 2026-07-01 - Plan D D5L: Final Aggregation and Reproducibility Package", section)


def _update_reproducibility_matrix() -> None:
    if not REPRO_MATRIX_PATH.exists():
        return
    text = REPRO_MATRIX_PATH.read_text(encoding="utf-8")
    if "D5L built the final Plan D reproducibility package" not in text:
        text = text.replace(
            "D5K completed local LLM history-length sensitivity as diagnostic_subset work.",
            "D5K completed local LLM history-length sensitivity as diagnostic_subset work. D5L built the final Plan D reproducibility package and hygiene audit without new inference.",
        )
    text = _replace_table_row(
        text,
        "Final report",
        "| Final report | Synthesize claims and limitations | `FEASIBLE` | `planned` | D5L final package is generated; final narrative can now be assembled from CSV evidence. |",
    )
    text = _replace_table_row(
        text,
        "Full LLM reproduction",
        "| Full LLM reproduction | `not_completed` | D5E-D5K are diagnostic_subset local LLM diagnostics; closed-source/full matrix reproduction remains conditional. |",
        occurrence=1,
    )
    text = _replace_section_text(
        text,
        "## Recommended next steps",
        """## Recommended next steps

1. Review the D5L final reproducibility package.
2. Prepare exam presentation tables from generated CSV files.
3. Optionally run an OpenAI API diagnostic only if budget is explicitly approved.
4. Finalize the report.
""",
    )
    REPRO_MATRIX_PATH.write_text(text, encoding="utf-8")


def _update_task_roadmap() -> None:
    frame = _read_or_empty(TASK_ROADMAP_PATH, TASK_ROADMAP_COLUMNS)
    row = {
        "phase": "D5L",
        "task": "final_aggregation_reproducibility_package",
        "dataset": "all",
        "model": "all_completed_plan_d_components",
        "backend": "none",
        "users": "n/a",
        "candidate_pool_size": "n/a",
        "prompt_strategy": "n/a",
        "output_format": "csv_markdown",
        "metrics": "final_index,metric_summary,llm_summary,reproducibility_status,hygiene_audit",
        "prerequisites": "D5E-D5K completed",
        "status": "completed_no_inference",
        "notes": "D5L aggregated existing Plan D artifacts and performed non-destructive hygiene audit.",
    }
    mask = frame["phase"] == "D5L"
    if mask.any():
        for column, value in row.items():
            frame.loc[mask, column] = value
    else:
        frame.loc[len(frame)] = row
    frame.to_csv(TASK_ROADMAP_PATH, index=False)


def _update_next_actions() -> None:
    rows = [
        _next_action("1", "review_plan_d_final_reproducibility_package", "Review Plan D final reproducibility package", "final_report", "D5L generated the final aggregate evidence tables.", "Reviewed final package and claim boundaries.", "D5L completed.", "low", "next", "Check raw outputs remain ignored before committing."),
        _next_action("2", "prepare_exam_presentation_tables", "Prepare exam presentation tables", "final_report", "The oral exam needs compact generated tables.", "Presentation-ready generated tables.", "D5L package available.", "low", "after package review", "Use generated CSV values."),
        _next_action("3", "final_report", "Finalize report", "final_report", "The project needs final claims grounded in generated artifacts.", "Final report with limitations and generated metrics.", "D5L package reviewed.", "low", "final", "Do not invent metric values."),
        _next_action("4", "optional_openai_api_diagnostic_if_budget_approved", "Optional OpenAI API diagnostic if budget approved", "api_availability", "Closed-source comparison remains conditional.", "Small approved API diagnostic or documented no-API decision.", "Explicit user approval, API key, and budget.", "high", "optional", "No API calls by default."),
    ]
    pd.DataFrame(rows, columns=NEXT_ACTION_COLUMNS).to_csv(NEXT_ACTIONS_PATH, index=False)


def _update_feasibility_ledger() -> None:
    frame = _read_or_empty(FEASIBILITY_LEDGER_PATH, LEDGER_COLUMNS)
    _upsert_ledger(
        frame,
        {
            "item_id": "plan_d_final_reproducibility_package",
            "area": "final_reporting",
            "target": "Plan D final aggregation, hygiene audit, and reproducibility package",
            "feasibility_status": "already_done",
            "required_resources": "Existing Plan D CSVs, reports, and Git-visible file state",
            "estimated_effort": "done",
            "risk_level": "low",
            "external_dependency": "no",
            "recommended_decision": "Use D5L outputs for final report and exam presentation.",
            "notes": "D5L completed without inference, API calls, downloads, model pulls, training, or destructive cleanup.",
        },
    )
    frame.to_csv(FEASIBILITY_LEDGER_PATH, index=False)


def _summary_row(frame: pd.DataFrame, phase: str, condition: str) -> pd.Series:
    rows = frame.loc[(frame["diagnostic_phase"] == phase) & (frame["condition"] == condition)]
    if rows.empty:
        return pd.Series({"HR@5": "missing", "NDCG@5": "missing", "hallucination_rate": "missing"})
    return rows.iloc[0]


def _read_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype=str).fillna("")
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame[columns]


def _upsert_ledger(frame: pd.DataFrame, row: dict[str, str]) -> None:
    mask = frame["item_id"] == row["item_id"]
    if mask.any():
        for column, value in row.items():
            frame.loc[mask, column] = value
    else:
        frame.loc[len(frame)] = row


def _next_action(
    priority_rank: str,
    action_id: str,
    action_title: str,
    related_component: str,
    why_it_matters: str,
    expected_output: str,
    prerequisites: str,
    risk: str,
    recommended_order: str,
    notes: str,
) -> dict[str, str]:
    return {
        "priority_rank": priority_rank,
        "action_id": action_id,
        "action_title": action_title,
        "related_component": related_component,
        "why_it_matters": why_it_matters,
        "expected_output": expected_output,
        "prerequisites": prerequisites,
        "risk": risk,
        "recommended_order": recommended_order,
        "notes": notes,
    }


def _replace_table_row(text: str, first_cell: str, replacement: str, *, occurrence: int = 1) -> str:
    prefix = f"| {first_cell} |"
    seen = 0
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith(prefix):
            seen += 1
            lines.append(replacement if seen == occurrence else line)
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def _replace_or_append_section(path: Path, header: str, section: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if header in text:
        text = _replace_section_text(text, header, section.strip() + "\n")
    else:
        text = text.rstrip() + "\n\n" + section.strip() + "\n"
    path.write_text(text, encoding="utf-8")


def _replace_section_text(text: str, header: str, replacement: str) -> str:
    start = text.find(header)
    if start == -1:
        return text.rstrip() + "\n\n" + replacement.strip() + "\n"
    next_start = text.find("\n## ", start + 1)
    if next_start == -1:
        return text[:start] + replacement.strip() + "\n"
    return text[:start] + replacement.strip() + "\n" + text[next_start:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Plan D final reproducibility package.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    package = build_package()
    write_package(package)
    LOGGER.info("Wrote final Plan D reproducibility package outputs for %s", PHASE)


if __name__ == "__main__":
    main()
