from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.plan_d_mistral_smoke_test import (
    BACKEND,
    DEFAULT_MODEL,
    METRIC_COLUMNS,
)

from src.llm.plan_d_mistral_smoke_test import (
    METHOD,
    PHASE,
    PLAN,
    PROMPT_STRATEGY,
    SUMMARY_COLUMNS,
    TEMPERATURE,
    TOP_K,
    SmokeEvaluation,
    blocked_evaluation,
    build_mistral_smoke_prompt,
    check_local_ollama_models,
    detect_mistral_candidates,
    evaluate_smoke_response,
    evaluation_summary_row,
    metric_rows,
    raw_smoke_trace_row,
    select_local_mistral_model,
)
from src.llm.plan_d_ollama_client import OllamaClient


LOGGER = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_METRICS_DIR = RESULTS_DIR / "metrics"
RAW_OUTPUTS_DIR = RESULTS_DIR / "raw_outputs"
REPORT_DIR = PROJECT_ROOT / "report"

D5H_SUMMARY_PATH = RESULTS_TABLES_DIR / "plan_d_mistral_smoke_test_summary.csv"
D5H_METRICS_PATH = RESULTS_METRICS_DIR / "plan_d_mistral_smoke_test_metrics.csv"
D5H_REPORT_PATH = REPORT_DIR / "plan_d_mistral_smoke_test.md"
D5H_RAW_PATH = RAW_OUTPUTS_DIR / "plan_d_mistral_smoke_test_raw.jsonl"

TASK_ROADMAP_PATH = RESULTS_TABLES_DIR / "plan_d_llm_task_roadmap.csv"
NEXT_ACTIONS_PATH = RESULTS_TABLES_DIR / "plan_d_next_actions_priority.csv"
FEASIBILITY_LEDGER_PATH = RESULTS_TABLES_DIR / "plan_d_feasibility_ledger.csv"
NOTES_PATH = REPORT_DIR / "notes.md"
REPRO_MATRIX_PATH = REPORT_DIR / "plan_d_reproducibility_matrix.md"

TIMEOUT_SECONDS = 90.0

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


def run_smoke_test(requested_model: str) -> dict[str, object]:
    inventory = check_local_ollama_models()
    selected_model = select_local_mistral_model(inventory.model_names, requested_model)
    if selected_model is None:
        evaluation = blocked_evaluation(requested_model=requested_model, inventory=inventory)
        return {"evaluation": evaluation, "raw_row": None}

    prompt = build_mistral_smoke_prompt()
    client = OllamaClient(timeout_seconds=TIMEOUT_SECONDS)
    LOGGER.info("Running one D5H local Mistral smoke request with %s", selected_model)
    result = client.generate_json(model=selected_model, prompt=prompt, temperature=TEMPERATURE)
    parsed, metrics = evaluate_smoke_response(result.response_text, response_received=result.response_received)
    status = _status_from_smoke_metrics(metrics)
    evaluation = SmokeEvaluation(
        requested_model=requested_model,
        selected_model=selected_model,
        model_available=True,
        mistral_candidates=detect_mistral_candidates(inventory.model_names),
        ollama_list_checked=True,
        local_ollama_used=True,
        local_request_count=1,
        response_received=bool(metrics["response_received"]),
        parse_success=bool(metrics["parse_success"]),
        schema_valid=bool(metrics["schema_valid"]),
        exact_top_k=bool(metrics["exact_top_k"]),
        candidate_validity_rate=float(metrics["candidate_validity_rate"]),
        hallucination_rate=float(metrics["hallucination_rate"]),
        duplicate_rate=float(metrics["duplicate_rate"]),
        latency_seconds=result.latency_seconds,
        status=status,
        raw_output_path=_relative(D5H_RAW_PATH),
        notes=result.error or parsed.error_message or _notes_for_status(status),
    )
    raw_row = raw_smoke_trace_row(
        evaluation=evaluation,
        raw_response_text=result.response_text,
        parsed_result=asdict(parsed),
    )
    return {"evaluation": evaluation, "raw_row": raw_row}


def write_outputs(result: dict[str, object]) -> None:
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    evaluation = result["evaluation"]
    raw_row = result["raw_row"]
    assert isinstance(evaluation, SmokeEvaluation)

    pd.DataFrame([evaluation_summary_row(evaluation)], columns=SUMMARY_COLUMNS).to_csv(D5H_SUMMARY_PATH, index=False)
    pd.DataFrame(metric_rows(evaluation), columns=METRIC_COLUMNS).to_csv(D5H_METRICS_PATH, index=False)
    if raw_row is not None:
        RAW_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        with D5H_RAW_PATH.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(raw_row, ensure_ascii=False, sort_keys=True) + "\n")

    D5H_REPORT_PATH.write_text(_report(evaluation), encoding="utf-8")
    _update_notes(evaluation)
    _update_task_roadmap(evaluation)
    _update_next_actions(evaluation)
    _update_feasibility_ledger(evaluation)
    _update_reproducibility_matrix(evaluation)


def _report(evaluation: SmokeEvaluation) -> str:
    if evaluation.status == "blocked_model_not_available":
        result_section = f"""## Result

- Status: `blocked_model_not_available`.
- Requested model: `{evaluation.requested_model}`.
- Model available locally: `false`.
- Local Ollama used for inference: `false`.
- Local request count: `0`.
- No Mistral inference was run.
- No download was performed.
- No model pull was performed.
- Manual command the user may run later: `ollama pull mistral:7b-instruct`.
- If the exact tag differs locally, run `ollama list` and choose the installed Mistral-compatible model name.
- Detected local Mistral candidates: `{', '.join(evaluation.mistral_candidates) if evaluation.mistral_candidates else 'none'}`.
"""
        metrics_section = "No smoke-test inference metrics are claimed because the requested model was unavailable."
    else:
        result_section = f"""## Result

- Status: `{evaluation.status}`.
- Requested model: `{evaluation.requested_model}`.
- Exact model name used: `{evaluation.selected_model}`.
- Model available locally: `true`.
- Local Ollama was used.
- Exactly 1 smoke-test request was run.
- Raw JSONL output: `{evaluation.raw_output_path}`.
- The raw JSONL output remains Git-ignored.
"""
        metrics_section = f"""| Metric | Value |
|---|---:|
| response_received | `{_bool_text(evaluation.response_received)}` |
| parse_success | `{_bool_text(evaluation.parse_success)}` |
| schema_valid | `{_bool_text(evaluation.schema_valid)}` |
| exact_top_k | `{_bool_text(evaluation.exact_top_k)}` |
| candidate_validity_rate | `{evaluation.candidate_validity_rate:.6f}` |
| hallucination_rate | `{evaluation.hallucination_rate:.6f}` |
| duplicate_rate | `{evaluation.duplicate_rate:.6f}` |
| latency_seconds | `{evaluation.latency_seconds:.6f}` |
"""
    return f"""# Plan D Mistral Smoke Test

## Purpose

D5H checks whether a local Mistral 7B Instruct-compatible Ollama model is available and, only if available, runs a single tiny candidate-ID-only JSON smoke test. It does not run ML-1M ranking diagnostics.

## Scope And Classification

- D5H is a local model availability/smoke-test phase.
- This is not exact reproduction.
- This is not full LLM reproduction.
- This is a local open model smoke test only when the requested Mistral model is already available.
- Target default model: `{DEFAULT_MODEL}`.
- Configured requested model: `{evaluation.requested_model}`.
- Temperature: `0`.
- Top-k: `{TOP_K}`.

## Execution Boundaries

- No OpenAI API call was made.
- No Anthropic API call was made.
- No external API call was made.
- No recommender training was performed.
- Plan C was not modified.
- D5H does not provide ML-1M ranking metrics.
- D5H only prepares a future Qwen-vs-Mistral comparison.

{result_section}

## Smoke-Test Metrics

{metrics_section}

## Notes

{evaluation.notes}
"""


def _update_notes(evaluation: SmokeEvaluation) -> None:
    raw_line = (
        f"- `results/raw_outputs/plan_d_mistral_smoke_test_raw.jsonl` (Git-ignored)"
        if evaluation.local_request_count == 1
        else "- No raw Mistral smoke-test JSONL was created because no inference was run."
    )
    section = f"""
## 2026-07-01 - Plan D D5H: Mistral Local Availability And Smoke Test

### Changes

- Added D5H local Mistral availability detection and conservative smoke-test artifacts.
- Checked local Ollama model inventory through `ollama list`.
- Ran a smoke test only if the configured Mistral model was already locally available.

### Outputs Generated

- `results/metrics/plan_d_mistral_smoke_test_metrics.csv`
- `results/tables/plan_d_mistral_smoke_test_summary.csv`
- `report/plan_d_mistral_smoke_test.md`
{raw_line}

### Result

- Requested model: `{evaluation.requested_model}`
- Selected model: `{evaluation.selected_model or 'none'}`
- Status: `{evaluation.status}`
- Local Ollama inference requests: `{evaluation.local_request_count}`
- Model pull/download performed: `false`

### Notes And Limitations

- D5H is not exact reproduction and not full LLM reproduction.
- No OpenAI API call, Anthropic API call, external API call, download, model pull, recommender training, LastFM experiment, or Plan C modification occurred.
"""
    _replace_or_append_section(NOTES_PATH, "## 2026-07-01 - Plan D D5H: Mistral Local Availability And Smoke Test", section)


def _update_task_roadmap(evaluation: SmokeEvaluation) -> None:
    frame = _read_or_empty(TASK_ROADMAP_PATH, TASK_ROADMAP_COLUMNS)
    for index, row in frame.iterrows():
        if row["phase"] == "D5H":
            frame.at[index, "task"] = "mistral_local_availability_smoke_test"
            frame.at[index, "dataset"] = "synthetic_toy"
            frame.at[index, "model"] = evaluation.requested_model
            frame.at[index, "backend"] = BACKEND
            frame.at[index, "users"] = "0"
            frame.at[index, "candidate_pool_size"] = "5"
            frame.at[index, "prompt_strategy"] = PROMPT_STRATEGY
            frame.at[index, "output_format"] = "strict_json"
            frame.at[index, "metrics"] = "response_received,parse_success,schema_valid,candidate_validity_rate,hallucination_rate"
            frame.at[index, "prerequisites"] = "D5G completed; local Mistral model manually available"
            frame.at[index, "status"] = evaluation.status
            frame.at[index, "notes"] = (
                "D5H blocked because the requested Mistral model is not local; no inference was run."
                if evaluation.status == "blocked_model_not_available"
                else "D5H completed exactly one local Mistral smoke-test request; no ML-1M ranking was run."
            )
    frame.to_csv(TASK_ROADMAP_PATH, index=False)


def _update_next_actions(evaluation: SmokeEvaluation) -> None:
    if evaluation.status == "blocked_model_not_available":
        rows = [
            _next_action("1", "pull_mistral_7b_instruct_v02_quantized", "Manually pull Mistral 7B instruct", "paper_llm_mistral_7b_instruct_v0_2", "D5H found the requested Mistral model unavailable locally.", "A Mistral-compatible model appears in ollama list.", "User manual action only.", "medium", "next", "Suggested manual command: ollama pull mistral:7b-instruct. Codex did not pull it."),
            _next_action("2", "rerun_d5h_mistral_smoke_test", "Rerun D5H Mistral smoke test", "mistral_smoke_test", "The Mistral smoke test should only run after a local model is present.", "One local Mistral smoke-test row.", "Mistral model available locally.", "low", "after manual pull", "No ranking diagnostic yet."),
            _next_action("3", "run_mistral_ml1m_ranking_diagnostic_20_users", "Run Mistral ML-1M ranking diagnostic", "llm_ranking; paper_llm_mistral_7b_instruct_v0_2", "A second local model helps compare validity and hallucination against Qwen.", "20-user Mistral ML-1M diagnostic metrics.", "D5H smoke test passed.", "high", "after D5H pass", "Keep as diagnostic_subset."),
            _next_action("4", "run_llm_reranking_diagnostic", "Run LLM reranking diagnostic", "llm_reranking", "Reranking remains a paper LLM task not yet evaluated.", "Reranking diagnostics with validity and utility metrics.", "Stable local model policy.", "high", "after Mistral decision", "Do not claim exact reproduction."),
            _next_action("5", "run_history_length_sensitivity_diagnostic", "Run history length sensitivity diagnostic", "history_length_sensitivity", "History length sensitivity remains pending.", "Bounded local LLM sensitivity diagnostic.", "Stable prompt/parser path.", "medium", "after reranking planning", "Use local models only unless scope changes."),
            _next_action("6", "test_stricter_candidate_id_only_prompt", "Keep strict prompt policy for future diagnostics", "prompting; output_validity", "D5G showed strict candidate-ID-only prompting removed Qwen hallucinations in the diagnostic subset.", "Documented prompt policy for later phases.", "D5G completed.", "low", "ongoing", "No extra inference implied."),
            _next_action("7", "integrate_llm_results_or_blocker_into_final_report", "Integrate LLM results or blocker into final report", "final_report; local_llm_backend", "The final report should include Qwen diagnostics and the Mistral availability blocker.", "Final LLM section grounded in generated CSVs.", "D5H documented.", "low", "after follow-up diagnostics", "Do not claim exact paper reproduction."),
            _next_action("8", "optional_openai_api_diagnostic_if_budget_approved", "Optional OpenAI API diagnostic if budget approved", "api_availability", "Closed-source comparison remains conditional.", "Small approved API diagnostic or documented no-API decision.", "Explicit user approval, API key, and budget.", "high", "optional", "No API calls by default."),
            _next_action("9", "prepare_exam_presentation_tables", "Prepare exam presentation tables", "final_report", "The oral exam needs compact generated tables.", "Presentation-ready generated tables.", "Stable final report data tables.", "low", "after final report draft", "Use generated CSV values."),
            _next_action("10", "final_report", "Finalize report", "final_report", "The project needs final claims grounded in generated artifacts.", "Final report with limitations and generated metrics.", "LLM diagnostics integrated.", "low", "final", "Do not invent metric values."),
        ]
    else:
        rows = [
            _next_action("1", "run_mistral_ml1m_ranking_diagnostic_20_users", "Run Mistral ML-1M ranking diagnostic", "llm_ranking; paper_llm_mistral_7b_instruct_v0_2", "D5H confirmed local Mistral smoke-test viability.", "20-user Mistral ML-1M diagnostic metrics.", "D5H smoke test passed.", "high", "next", "Keep as diagnostic_subset."),
            _next_action("2", "run_llm_reranking_diagnostic", "Run LLM reranking diagnostic", "llm_reranking", "Reranking remains a paper LLM task not yet evaluated.", "Reranking diagnostics with validity and utility metrics.", "Stable local model policy.", "high", "after Mistral ranking", "Do not claim exact reproduction."),
            _next_action("3", "run_history_length_sensitivity_diagnostic", "Run history length sensitivity diagnostic", "history_length_sensitivity", "History length sensitivity remains pending.", "Bounded local LLM sensitivity diagnostic.", "Stable prompt/parser path.", "medium", "after reranking planning", "Use local models only unless scope changes."),
            _next_action("4", "test_stricter_candidate_id_only_prompt", "Keep strict prompt policy for future diagnostics", "prompting; output_validity", "D5G showed strict candidate-ID-only prompting removed Qwen hallucinations in the diagnostic subset.", "Documented prompt policy for later phases.", "D5G completed.", "low", "ongoing", "No extra inference implied."),
            _next_action("5", "pull_mistral_7b_instruct_v02_quantized", "Confirm Mistral local model remains available", "paper_llm_mistral_7b_instruct_v0_2", "The installed local Mistral model should remain documented.", "Confirmed local model in ollama list.", "None if already installed.", "low", "as needed", "Do not pull automatically."),
            _next_action("6", "integrate_llm_results_or_blocker_into_final_report", "Integrate LLM results into final report", "final_report; local_llm_backend", "The final report should include Qwen diagnostics and Mistral smoke status.", "Final LLM section grounded in generated CSVs.", "D5H documented.", "low", "after follow-up diagnostics", "Do not claim exact paper reproduction."),
            _next_action("7", "optional_openai_api_diagnostic_if_budget_approved", "Optional OpenAI API diagnostic if budget approved", "api_availability", "Closed-source comparison remains conditional.", "Small approved API diagnostic or documented no-API decision.", "Explicit user approval, API key, and budget.", "high", "optional", "No API calls by default."),
            _next_action("8", "prepare_exam_presentation_tables", "Prepare exam presentation tables", "final_report", "The oral exam needs compact generated tables.", "Presentation-ready generated tables.", "Stable final report data tables.", "low", "after final report draft", "Use generated CSV values."),
            _next_action("9", "final_report", "Finalize report", "final_report", "The project needs final claims grounded in generated artifacts.", "Final report with limitations and generated metrics.", "LLM diagnostics integrated.", "low", "final", "Do not invent metric values."),
        ]
    pd.DataFrame(rows, columns=NEXT_ACTION_COLUMNS).to_csv(NEXT_ACTIONS_PATH, index=False)


def _update_feasibility_ledger(evaluation: SmokeEvaluation) -> None:
    frame = _read_or_empty(FEASIBILITY_LEDGER_PATH, LEDGER_COLUMNS)
    feasibility_status = "blocked_missing_backend" if evaluation.status == "blocked_model_not_available" else "already_done"
    _upsert_ledger(
        frame,
        {
            "item_id": "mistral_local_smoke_test",
            "area": "llm_model",
            "target": "Mistral local availability and smoke test",
            "feasibility_status": feasibility_status,
            "required_resources": "Ollama and a local Mistral 7B Instruct-compatible model",
            "estimated_effort": "manual pull required" if evaluation.status == "blocked_model_not_available" else "done",
            "risk_level": "medium",
            "external_dependency": "yes",
            "recommended_decision": (
                "Ask the user to manually pull a Mistral model before rerunning D5H."
                if evaluation.status == "blocked_model_not_available"
                else "Use D5H smoke status before any Mistral ranking diagnostic."
            ),
            "notes": f"D5H status={evaluation.status}; requested_model={evaluation.requested_model}; selected_model={evaluation.selected_model or 'none'}.",
        },
    )
    mask = frame["item_id"] == "paper_llm_mistral_7b_instruct_v0_2"
    if mask.any():
        frame.loc[mask, "feasibility_status"] = feasibility_status
        frame.loc[mask, "recommended_decision"] = (
            "Manual local model pull is required before a Mistral diagnostic."
            if evaluation.status == "blocked_model_not_available"
            else "Mistral smoke test completed; a bounded diagnostic can be planned."
        )
        frame.loc[mask, "notes"] = f"D5H status={evaluation.status}; no automatic pull/download was performed."
    frame.to_csv(FEASIBILITY_LEDGER_PATH, index=False)


def _update_reproducibility_matrix(evaluation: SmokeEvaluation) -> None:
    if not REPRO_MATRIX_PATH.exists():
        return
    text = REPRO_MATRIX_PATH.read_text(encoding="utf-8")
    status_label = (
        "blocked_model_not_available / conditional_manual_pull"
        if evaluation.status == "blocked_model_not_available"
        else "smoke_test_completed"
    )
    notes = (
        "D5H found the requested local Mistral model unavailable; no inference, pull, or download was run."
        if evaluation.status == "blocked_model_not_available"
        else f"D5H ran exactly one local Ollama smoke-test request with `{evaluation.selected_model}`; no ML-1M ranking metrics are claimed."
    )
    text = _replace_table_row(
        text,
        "mistral-7b-instruct-v0.2",
        f"| mistral-7b-instruct-v0.2 | `{status_label}` | `local_open_model_near_exact` | {notes} |",
    )
    text = _replace_table_row(
        text,
        "Mistral local comparison diagnostic",
        f"| Mistral local comparison diagnostic | `{evaluation.status}` | {notes} |",
    )
    insertion = (
        f" D5H checked local Mistral availability with no automatic pull/download; status `{evaluation.status}`."
    )
    if "D5H checked local Mistral availability" not in text:
        text = text.replace(
            "D5G completed a strict candidate-ID-only prompt and deterministic post-filter diagnostic as diagnostic_subset work; it is not exact reproduction.",
            "D5G completed a strict candidate-ID-only prompt and deterministic post-filter diagnostic as diagnostic_subset work; it is not exact reproduction." + insertion,
        )
    recommended = _recommended_section(evaluation)
    text = _replace_section_text(text, "## Recommended next steps", recommended)
    REPRO_MATRIX_PATH.write_text(text, encoding="utf-8")


def _status_from_smoke_metrics(metrics: dict[str, float | bool]) -> str:
    if not bool(metrics["response_received"]):
        return "smoke_test_failed_backend"
    if not bool(metrics["parse_success"]):
        return "smoke_test_failed_parse"
    if not bool(metrics["schema_valid"]) or not bool(metrics["exact_top_k"]):
        return "smoke_test_failed_schema"
    if float(metrics["candidate_validity_rate"]) < 1.0 or float(metrics["hallucination_rate"]) > 0.0:
        return "smoke_test_failed_candidate_validity"
    return "smoke_test_passed"


def _notes_for_status(status: str) -> str:
    if status == "smoke_test_passed":
        return "Mistral local smoke test produced strict parseable candidate-valid JSON."
    if status == "smoke_test_failed_candidate_validity":
        return "Mistral smoke output parsed but contained invalid candidate IDs."
    if status == "smoke_test_failed_schema":
        return "Mistral smoke output parsed but did not meet the strict schema."
    if status == "smoke_test_failed_parse":
        return "Mistral smoke output was not parseable JSON."
    return "Local Ollama request failed; no output was fabricated."


def _recommended_section(evaluation: SmokeEvaluation) -> str:
    if evaluation.status == "blocked_model_not_available":
        return """## Recommended next steps

1. Manually run `ollama pull mistral:7b-instruct` if a Mistral comparison is desired.
2. Rerun D5H Mistral smoke test after the model appears in `ollama list`.
3. Run the Mistral ML-1M ranking diagnostic only after D5H passes.
4. Run LLM reranking diagnostics.
5. Run history length sensitivity diagnostics.
6. Integrate LLM results or the backend blocker into the final report.
7. Optionally run an OpenAI API diagnostic if budget is explicitly approved.
8. Prepare exam presentation tables.
9. Finalize the report.
"""
    return """## Recommended next steps

1. Run the Mistral ML-1M ranking diagnostic for 20 users.
2. Run LLM reranking diagnostics.
3. Run history length sensitivity diagnostics.
4. Integrate LLM results into the final report.
5. Optionally run an OpenAI API diagnostic if budget is explicitly approved.
6. Prepare exam presentation tables.
7. Finalize the report.
"""


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


def _replace_table_row(text: str, first_cell: str, replacement: str) -> str:
    prefix = f"| {first_cell} |"
    return "\n".join(replacement if line.startswith(prefix) else line for line in text.splitlines()) + "\n"


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


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D5H Mistral local availability and smoke test.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local Ollama model name to check and smoke test.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    result = run_smoke_test(args.model)
    write_outputs(result)
    LOGGER.info("Wrote %s", D5H_SUMMARY_PATH)
    LOGGER.info("Wrote %s", D5H_REPORT_PATH)


if __name__ == "__main__":
    main()
