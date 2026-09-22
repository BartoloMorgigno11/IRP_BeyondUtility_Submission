from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.plan_d_llm_json_parser import SmokeJsonValidation, parse_and_validate_smoke_json
from src.llm.plan_d_ollama_client import OllamaClient, OllamaGenerateResult


LOGGER = logging.getLogger(__name__)

PLAN = "Plan D"
PHASE = "D5C1"
BACKEND = "ollama"
MODEL = "qwen2:7b-instruct"
PROMPT_STRATEGY = "toy_strict_json"
TEMPERATURE = 0.0
TIMEOUT_SECONDS = 45.0

RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORT_DIR = PROJECT_ROOT / "report"
SMOKE_CSV_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ollama_smoke_test.csv"
SMOKE_REPORT_PATH = REPORT_DIR / "plan_d_qwen2_ollama_smoke_test.md"
TASK_ROADMAP_PATH = RESULTS_TABLES_DIR / "plan_d_llm_task_roadmap.csv"
NEXT_ACTIONS_PATH = RESULTS_TABLES_DIR / "plan_d_next_actions_priority.csv"
FEASIBILITY_LEDGER_PATH = RESULTS_TABLES_DIR / "plan_d_feasibility_ledger.csv"
NOTES_PATH = REPORT_DIR / "notes.md"
REPRO_MATRIX_PATH = REPORT_DIR / "plan_d_reproducibility_matrix.md"

SMOKE_COLUMNS = [
    "plan",
    "phase",
    "backend",
    "model",
    "prompt_strategy",
    "temperature",
    "response_received",
    "parse_success",
    "schema_valid",
    "selected_count",
    "candidate_validity_rate",
    "duplicate_count",
    "hallucinated_count",
    "latency_seconds",
    "raw_output_path",
    "status",
    "notes",
]

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

CANDIDATES = {
    "item_A": "space opera adventure",
    "item_B": "romantic drama",
    "item_C": "animated family adventure",
    "item_D": "historical documentary",
    "item_E": "cyberpunk science fiction thriller",
}
ALLOWED_STATUSES = {
    "smoke_test_passed",
    "smoke_test_failed_parse",
    "smoke_test_failed_schema",
    "smoke_test_failed_backend",
    "smoke_test_failed_candidate_validity",
}


@dataclass(frozen=True)
class SmokeSummary:
    response_received: bool
    parse_success: bool
    schema_valid: bool
    selected_count: int
    candidate_validity_rate: float
    duplicate_count: int
    hallucinated_count: int
    latency_seconds: float
    status: str
    notes: str

    def row(self) -> dict[str, str]:
        return {
            "plan": PLAN,
            "phase": PHASE,
            "backend": BACKEND,
            "model": MODEL,
            "prompt_strategy": PROMPT_STRATEGY,
            "temperature": "0",
            "response_received": _bool_text(self.response_received),
            "parse_success": _bool_text(self.parse_success),
            "schema_valid": _bool_text(self.schema_valid),
            "selected_count": str(self.selected_count),
            "candidate_validity_rate": f"{self.candidate_validity_rate:.6f}",
            "duplicate_count": str(self.duplicate_count),
            "hallucinated_count": str(self.hallucinated_count),
            "latency_seconds": f"{self.latency_seconds:.6f}",
            "raw_output_path": "not_saved",
            "status": self.status,
            "notes": self.notes,
        }


def build_toy_prompt() -> str:
    candidate_lines = "\n".join(f"- {item_id}: {description}" for item_id, description in CANDIDATES.items())
    return f"""You are a recommendation ranking assistant.

The user liked science fiction, adventure, and animated family movies.

Candidate items:
{candidate_lines}

Return only valid JSON with exactly this schema:
{{
  "ranked_item_ids": ["item_id_1", "item_id_2", "item_id_3"],
  "status": "ok"
}}

Rules:
- Return only valid JSON.
- Choose exactly 3 item IDs.
- Use only candidate IDs from the provided candidate list.
- Do not include explanations.
- Do not invent item IDs.
"""


def run_smoke_test() -> SmokeSummary:
    client = OllamaClient(timeout_seconds=TIMEOUT_SECONDS)
    result = client.generate_json(model=MODEL, prompt=build_toy_prompt(), temperature=TEMPERATURE)
    if not result.response_received:
        note = f"Backend failed or returned no response: {result.error or 'unknown error'}"
        return SmokeSummary(
            response_received=False,
            parse_success=False,
            schema_valid=False,
            selected_count=0,
            candidate_validity_rate=0.0,
            duplicate_count=0,
            hallucinated_count=0,
            latency_seconds=result.latency_seconds,
            status="smoke_test_failed_backend",
            notes=note,
        )

    validation = parse_and_validate_smoke_json(
        result.response_text,
        candidate_ids=set(CANDIDATES),
        expected_count=3,
    )
    status = _status_from_validation(validation)
    notes = _notes_from_validation(validation)
    return SmokeSummary(
        response_received=True,
        parse_success=validation.parse_success,
        schema_valid=validation.schema_valid,
        selected_count=validation.selected_count,
        candidate_validity_rate=validation.candidate_validity_rate,
        duplicate_count=validation.duplicate_count,
        hallucinated_count=validation.hallucinated_count,
        latency_seconds=result.latency_seconds,
        status=status,
        notes=notes,
    )


def write_outputs(summary: SmokeSummary) -> None:
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary.row()], columns=SMOKE_COLUMNS).to_csv(SMOKE_CSV_PATH, index=False)
    SMOKE_REPORT_PATH.write_text(_smoke_report(summary), encoding="utf-8")
    _update_task_roadmap(summary)
    _update_next_actions(summary)
    _update_feasibility_ledger(summary)
    _update_notes(summary)
    _update_reproducibility_matrix(summary)


def _status_from_validation(validation: SmokeJsonValidation) -> str:
    if not validation.parse_success:
        return "smoke_test_failed_parse"
    if not validation.schema_valid:
        return "smoke_test_failed_schema"
    if validation.hallucinated_count > 0 or validation.candidate_validity_rate < 1.0:
        return "smoke_test_failed_candidate_validity"
    return "smoke_test_passed"


def _notes_from_validation(validation: SmokeJsonValidation) -> str:
    if not validation.parse_success:
        return f"Model returned malformed JSON: {validation.parse_error}"
    if not validation.schema_valid:
        return f"Model JSON failed schema validation: {validation.schema_error}"
    if validation.hallucinated_count > 0:
        return "Model returned one or more item IDs outside the toy candidate set."
    if validation.duplicate_count > 0:
        return "Model returned parseable valid candidate IDs with duplicate selections recorded."
    return "Model returned parseable JSON with valid toy candidate IDs."


def _smoke_report(summary: SmokeSummary) -> str:
    pass_text = "passed" if summary.status == "smoke_test_passed" else "did not pass"
    hallucination_text = "yes" if summary.hallucinated_count > 0 else "no"
    duplicate_text = "yes" if summary.duplicate_count > 0 else "no"
    next_step = (
        "Implement reusable LLM prompt templates for the real diagnostic path, then run the bounded Qwen ML-1M diagnostic only after that parser/template layer is ready."
        if summary.status == "smoke_test_passed"
        else "Fix the local Ollama/Qwen smoke test failure before any ranking diagnostic."
    )
    return f"""# Plan D Qwen2 Ollama Smoke Test

## Purpose

D5C1 verifies that the local Ollama backend and `qwen2:7b-instruct` can produce a strictly parseable JSON response for a toy recommendation-style prompt. This is a local smoke test only.

## Backend And Model

- Backend: `ollama`
- Model: `qwen2:7b-instruct`
- Temperature: `0`
- Endpoint: local Ollama generate API on `localhost`

## Prompt Schema

The toy prompt used no real user data. It asked for exactly:

```json
{{
  "ranked_item_ids": ["item_id_1", "item_id_2", "item_id_3"],
  "status": "ok"
}}
```

The candidate IDs were `item_A`, `item_B`, `item_C`, `item_D`, and `item_E`.

## Validation Criteria

- A response must be received from local Ollama.
- The response text must parse as JSON.
- The JSON must contain `ranked_item_ids` as exactly three strings and `status` equal to `ok`.
- Returned IDs must come from the toy candidate set.
- Duplicate IDs and hallucinated IDs are counted without manual correction.

## Result Summary

The smoke test {pass_text}: `{summary.status}`.

| Field | Value |
|---|---|
| response_received | `{_bool_text(summary.response_received)}` |
| parse_success | `{_bool_text(summary.parse_success)}` |
| schema_valid | `{_bool_text(summary.schema_valid)}` |
| selected_count | `{summary.selected_count}` |
| candidate_validity_rate | `{summary.candidate_validity_rate:.6f}` |
| duplicate_count | `{summary.duplicate_count}` |
| hallucinated_count | `{summary.hallucinated_count}` |
| hallucinations_occurred | `{hallucination_text}` |
| duplicate_ids_occurred | `{duplicate_text}` |
| latency_seconds | `{summary.latency_seconds:.6f}` |
| status | `{summary.status}` |

Notes: {summary.notes}

## Scope Limits

- This is not a dataset ranking experiment.
- This is not an exact paper LLM reproduction.
- No external API was called.
- No OpenAI API or Anthropic API was called.
- No model was pulled or downloaded in this phase.
- No recommender model was trained.
- No Plan C output was modified.
- No recommender raw output was modified.
- No fake LLM output was created.

Raw LLM response text was not saved; the committed CSV/report contain only summary fields.

## Recommended Next Step

{next_step}
"""


def _update_task_roadmap(summary: SmokeSummary) -> None:
    frame = _read_or_empty(TASK_ROADMAP_PATH, TASK_ROADMAP_COLUMNS)
    passed = summary.status == "smoke_test_passed"
    for index, row in frame.iterrows():
        phase = row["phase"]
        if phase == "D5C":
            frame.at[index, "status"] = "completed_smoke_test" if passed else "failed_smoke_test"
            frame.at[index, "notes"] = (
                "D5C1 passed with parseable JSON and valid toy candidate IDs; no dataset ranking was run."
                if passed
                else f"D5C1 failed with status={summary.status}; no dataset ranking was run."
            )
        elif phase == "D5D":
            frame.at[index, "status"] = "pending_prompt_templates" if passed else "blocked_smoke_test"
            frame.at[index, "notes"] = (
                "Smoke test passed; implement real prompt templates/parser integration before ranking diagnostics."
                if passed
                else "Fix the smoke test failure before any ranking, reranking, or position-bias diagnostic."
            )
        elif phase in {"D5F", "D5G"}:
            frame.at[index, "status"] = "pending_ranking_diagnostic" if passed else "blocked_smoke_test"
            frame.at[index, "notes"] = (
                "Requires successful Qwen ranking diagnostic over real outputs before this phase."
                if passed
                else "Fix the smoke test failure before any ranking, reranking, or position-bias diagnostic."
            )
    frame.to_csv(TASK_ROADMAP_PATH, index=False)


def _update_next_actions(summary: SmokeSummary) -> None:
    if summary.status == "smoke_test_passed":
        rows = [
            _next_action("1", "implement_llm_prompt_templates_and_parser", "Implement LLM prompt templates and parser", "llm_ranking; hallucination", "Real diagnostics need reusable prompt construction and strict output validation.", "Prompt templates, parser tests, and validation rules.", "D5C smoke test passed.", "medium", "next", "Parser exists for the smoke test; real diagnostic templates still need implementation."),
            _next_action("2", "run_qwen2_ml1m_ranking_diagnostic_20_users", "Run Qwen ML-1M ranking diagnostic", "llm_ranking", "This is the smallest real local LLM ranking experiment after parser/template readiness.", "20-user ML-1M ranking diagnostics with strict validity and hallucination metrics.", "Prompt templates ready and D5C passed.", "high", "after parser/templates", "Use real outputs only; do not fabricate raw outputs."),
            _next_action("3", "pull_mistral_7b_instruct_v02_quantized", "Pull Mistral 7B instruct v0.2 quantized", "paper_llm_mistral_7b_instruct_v0_2", "Mistral is the second local model for output-validity comparison.", "A recommended Mistral v0.2 quantized model appears in ollama list.", "Manual user pull only.", "medium", "after Qwen path works", "Mistral remains unavailable locally."),
            _next_action("4", "run_mistral_ml1m_ranking_diagnostic_20_users", "Run Mistral ML-1M ranking diagnostic", "llm_ranking; paper_llm_mistral_7b_instruct_v0_2", "A second local model helps compare validity and hallucination behavior.", "20-user Mistral ML-1M ranking diagnostic metrics.", "D5C passed, parser ready, Mistral pulled.", "high", "after Qwen D5D", "Keep as diagnostic; no exact paper reproduction claim."),
            _next_action("5", "run_candidate_position_bias_diagnostic", "Run candidate position-bias diagnostic", "position_bias", "Candidate position bias is meaningful only with real LLM selections.", "Position-bias metrics from real parsed LLM outputs.", "D5D passed.", "medium", "after Qwen ranking diagnostic", "Do not compute from fake or empty outputs."),
            _next_action("6", "run_llm_reranking_diagnostic", "Run LLM reranking diagnostic", "llm_reranking", "Reranking is closer to the paper task but needs a stable candidate-pool policy.", "Reranking diagnostics with HR@5, NDCG@5, validity, and hallucination metrics.", "D5D passed and baseline top20 candidate source fixed.", "high", "after position-bias diagnostic", "Keep as diagnostic_subset unless official candidate pools are recovered."),
            _next_action("7", "integrate_llm_results_or_blocker_into_final_report", "Integrate LLM results or blocker into final report", "final_report; local_llm_backend; api_availability", "The final report must close the LLM loop with real results or a documented blocker.", "Final LLM section grounded in generated CSVs and audit artifacts.", "D5C-D5H outputs or explicit backend blocker.", "low", "after attempted LLM path or blocker decision", "Do not manually invent metric values."),
            _next_action("8", "run_d5c_qwen2_smoke_test", "Run D5C Qwen smoke test", "ollama_smoke_test", "The smoke test gates all local Qwen diagnostics.", "D5C smoke-test summary CSV and report.", "Ollama installed and qwen2 available.", "low", "satisfied_by_d5c1", "Current status: smoke_test_passed."),
            _next_action("9", "install_or_confirm_ollama_backend", "Install or confirm Ollama backend", "local_ollama_backend", "D5C-D5H local LLM diagnostics require a real local backend.", "Ollama available on PATH and documented in the backend audit.", "Manual user installation if missing.", "low", "satisfied_by_d5c0", "Ollama is available."),
            _next_action("10", "pull_qwen2_7b_instruct", "Pull qwen2 7B instruct", "paper_llm_qwen_2_7b_instruct", "Qwen is the first local model target for D5C smoke testing.", "qwen2:7b-instruct appears in ollama list.", "Ollama installed.", "low", "satisfied_by_d5c0", "qwen2:7b-instruct is available locally."),
        ]
    else:
        rows = [
            _next_action("1", "fix_local_ollama_qwen_smoke_test", "Fix local Ollama Qwen smoke test", "ollama_smoke_test", "No real LLM ranking diagnostic should run until the smoke test produces valid output.", "A rerun plan for D5C after the local failure is understood.", "Review D5C1 failure notes.", "high", "next", f"Current status: {summary.status}."),
            _next_action("2", "run_d5c_qwen2_smoke_test", "Run D5C Qwen smoke test", "ollama_smoke_test", "The smoke test must pass before ranking diagnostics.", "D5C smoke-test summary CSV and report.", "Local failure resolved.", "medium", "after fix", "Run exactly one local smoke inference per approved retry."),
            _next_action("3", "install_or_confirm_ollama_backend", "Install or confirm Ollama backend", "local_ollama_backend", "D5C-D5H local LLM diagnostics require a real local backend.", "Ollama available on PATH and documented in the backend audit.", "Manual user installation if missing.", "low", "satisfied_by_d5c0", "Ollama is available."),
            _next_action("4", "pull_qwen2_7b_instruct", "Pull qwen2 7B instruct", "paper_llm_qwen_2_7b_instruct", "Qwen is the first local model target for D5C smoke testing.", "qwen2:7b-instruct appears in ollama list.", "Ollama installed.", "low", "satisfied_by_d5c0", "qwen2:7b-instruct is available locally."),
        ]
    pd.DataFrame(rows, columns=NEXT_ACTION_COLUMNS).to_csv(NEXT_ACTIONS_PATH, index=False)


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


def _update_feasibility_ledger(summary: SmokeSummary) -> None:
    frame = _read_or_empty(FEASIBILITY_LEDGER_PATH, LEDGER_COLUMNS)
    decision = (
        "Proceed to prompt/template implementation before ranking diagnostics."
        if summary.status == "smoke_test_passed"
        else "Fix the local smoke test before ranking diagnostics."
    )
    note = (
        "D5C1 smoke test passed; local toy JSON output was parseable and candidate-valid. No dataset ranking was run."
        if summary.status == "smoke_test_passed"
        else f"D5C1 smoke test failed with status={summary.status}; no dataset ranking was run."
    )
    for item_id in (
        "local_llm_backend",
        "ollama_backend",
        "local_ollama_backend",
        "paper_llm_qwen_2_7b_instruct",
    ):
        mask = frame["item_id"] == item_id
        if mask.any():
            frame.loc[mask, "feasibility_status"] = "feasible_with_backend"
            frame.loc[mask, "recommended_decision"] = decision
            if item_id == "paper_llm_qwen_2_7b_instruct":
                frame.loc[mask, "notes"] = (
                    "registry_status=recommended_first_local_model; availability=available_locally; "
                    f"smoke_status={summary.status}."
                )
            else:
                frame.loc[mask, "notes"] = note
    frame.to_csv(FEASIBILITY_LEDGER_PATH, index=False)


def _update_notes(summary: SmokeSummary) -> None:
    section = f"""
## 2026-07-01 - Plan D D5C1: Qwen2 Ollama Smoke Test

### Changes

- Added a local Ollama client, strict JSON parser, and smoke-test runner for
  `qwen2:7b-instruct`.
- Ran exactly one local Qwen2/Ollama inference against a toy
  recommendation-style prompt with no real user or dataset data.
- Wrote a one-row smoke summary CSV and a smoke-test report.

### Commands Run

```powershell
python src/experiments/run_plan_d_qwen_smoke_test.py
```

### Outputs Generated

- `results/tables/plan_d_qwen2_ollama_smoke_test.csv`
- `report/plan_d_qwen2_ollama_smoke_test.md`

### Result

- response_received: `{_bool_text(summary.response_received)}`
- parse_success: `{_bool_text(summary.parse_success)}`
- schema_valid: `{_bool_text(summary.schema_valid)}`
- selected_count: `{summary.selected_count}`
- candidate_validity_rate: `{summary.candidate_validity_rate:.6f}`
- duplicate_count: `{summary.duplicate_count}`
- hallucinated_count: `{summary.hallucinated_count}`
- latency_seconds: `{summary.latency_seconds:.6f}`
- status: `{summary.status}`

### Notes And Limitations

- This was one toy local smoke inference only.
- This is not a dataset ranking experiment.
- This is not an exact paper LLM reproduction.
- No external API, OpenAI API, or Anthropic API was called.
- No model was pulled or downloaded in this phase.
- No recommender model was trained.
- No Plan C output, source metric CSV, recommender raw output, or LLM
  recommendation output was modified or created.
"""
    _replace_or_append_section(NOTES_PATH, "## 2026-07-01 - Plan D D5C1: Qwen2 Ollama Smoke Test", section)


def _update_reproducibility_matrix(summary: SmokeSummary) -> None:
    if not REPRO_MATRIX_PATH.exists():
        return
    text = REPRO_MATRIX_PATH.read_text(encoding="utf-8")
    passed = summary.status == "smoke_test_passed"
    intro = (
        "D5B added an LLM model registry and backend feasibility audit before any LLM experiment. "
        "D5C0 reran the audit after manual Ollama installation and manual `qwen2:7b-instruct` pull. "
        f"D5C1 ran exactly one toy local Qwen2/Ollama smoke inference with status `{summary.status}`. "
        "No dataset ranking was run, no LLM recommendation metrics are claimed, and closed-source API reproduction remains conditional on explicit approval, keys, availability, and budget."
    )
    text = _replace_paragraph_starting(text, "D5B added an LLM model registry", intro)
    qwen_status = "smoke_test_passed" if passed else summary.status
    text = _replace_table_row(
        text,
        "qwen-2-7b-instruct",
        f"| qwen-2-7b-instruct | `{qwen_status}` | `local_open_model_near_exact` | D5C1 ran exactly one toy local Qwen2/Ollama smoke inference; no dataset ranking result is claimed. |",
    )
    text = _replace_table_row(
        text,
        "Ollama",
        f"| Ollama | `{qwen_status}` | `planned` | D5C1 used Ollama `0.31.1` for exactly one local toy smoke inference; no ranking diagnostic has run. |",
    )
    text = _replace_table_row(
        text,
        "D5C Qwen smoke test",
        f"| D5C Qwen smoke test | `{qwen_status if passed else summary.status}` | Exactly one toy local smoke inference was run; no dataset ranking experiment was run. |",
    )
    text = _replace_table_row(
        text,
        "Qwen ML-1M ranking diagnostic",
        "| Qwen ML-1M ranking diagnostic | `pending_prompt_templates` | Run only after prompt/template integration and parser validation. |",
    )
    text = _replace_table_row(
        text,
        "Local LLM diagnostic ranking",
        "| Local LLM diagnostic ranking | `pending_prompt_templates` | D5C1 smoke test is complete, but no dataset ranking inference has run yet. |"
        if passed
        else "| Local LLM diagnostic ranking | `blocked_smoke_test` | D5C1 smoke test failed; no dataset ranking inference should run until fixed. |",
    )
    recommended = (
        "## Recommended next steps\n\n"
        "1. Implement LLM prompt templates and parser.\n"
        "2. Run the Qwen ML-1M ranking diagnostic for 20 users.\n"
        "3. Pull a quantized Mistral 7B Instruct v0.2 model manually.\n"
        "4. Run the Mistral ML-1M ranking diagnostic for 20 users.\n"
        "5. Run candidate position-bias diagnostics over real LLM outputs.\n"
        "6. Run LLM reranking diagnostics.\n"
        "7. Integrate LLM results or the backend blocker into the final report.\n"
        if passed
        else "## Recommended next steps\n\n"
        "1. Fix the local Ollama/Qwen smoke test failure.\n"
        "2. Rerun the D5C Qwen smoke test only after the failure is understood.\n"
        "3. Do not run Qwen ranking, reranking, or position-bias diagnostics until D5C passes.\n"
        "4. Integrate the smoke-test blocker into the final report if it remains unresolved.\n"
    )
    text = _replace_section_text(text, "## Recommended next steps", recommended)
    REPRO_MATRIX_PATH.write_text(text, encoding="utf-8")


def _replace_table_row(text: str, first_cell: str, replacement: str) -> str:
    prefix = f"| {first_cell} |"
    lines = [replacement if line.startswith(prefix) else line for line in text.splitlines()]
    return "\n".join(lines) + "\n"


def _replace_paragraph_starting(text: str, startswith: str, replacement: str) -> str:
    paragraphs = text.split("\n\n")
    for index, paragraph in enumerate(paragraphs):
        if paragraph.startswith(startswith):
            paragraphs[index] = replacement
            break
    return "\n\n".join(paragraphs)


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


def _read_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype=str).fillna("")
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame[columns]


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one local Qwen2/Ollama Plan D smoke test. No dataset ranking is performed."
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    summary = run_smoke_test()
    if summary.status not in ALLOWED_STATUSES:
        raise RuntimeError(f"Unexpected smoke status: {summary.status}")
    write_outputs(summary)
    LOGGER.info("Wrote %s", SMOKE_CSV_PATH)
    LOGGER.info("Wrote %s", SMOKE_REPORT_PATH)
    LOGGER.info("Smoke status: %s", summary.status)


if __name__ == "__main__":
    main()
