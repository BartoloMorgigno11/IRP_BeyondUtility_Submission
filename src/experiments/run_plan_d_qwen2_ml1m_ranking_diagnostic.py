from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.plan_d_llm_ranking_evaluator import (
    AGGREGATE_SUMMARY_COLUMNS,
    BACKEND,
    CANDIDATE_POOL_SIZE,
    DATASET,
    INSTANCE_SUMMARY_COLUMNS,
    METRIC_COLUMNS,
    MODEL,
    PHASE,
    PLAN,
    PROMPT_STRATEGY,
    SCOPE,
    TEMPERATURE,
    TOP_K,
    ManifestPromptBuilder,
    aggregate_evaluations,
    evaluate_response,
    load_and_validate_manifest,
    raw_trace_row,
)
from src.llm.plan_d_ollama_client import OllamaClient


LOGGER = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_METRICS_DIR = RESULTS_DIR / "metrics"
RAW_OUTPUTS_DIR = RESULTS_DIR / "raw_outputs"
REPORT_DIR = PROJECT_ROOT / "report"

MANIFEST_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ml1m_ranking_diagnostic_manifest.csv"
INSTANCE_SUMMARY_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ml1m_ranking_diagnostic_instance_summary.csv"
AGGREGATE_SUMMARY_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ml1m_ranking_diagnostic_summary.csv"
METRICS_PATH = RESULTS_METRICS_DIR / "plan_d_qwen2_ml1m_ranking_diagnostic_metrics.csv"
RAW_OUTPUT_PATH = RAW_OUTPUTS_DIR / "plan_d_qwen2_ml1m_ranking_diagnostic_raw.jsonl"
REPORT_PATH = REPORT_DIR / "plan_d_qwen2_ml1m_ranking_diagnostic.md"
TASK_ROADMAP_PATH = RESULTS_TABLES_DIR / "plan_d_llm_task_roadmap.csv"
NEXT_ACTIONS_PATH = RESULTS_TABLES_DIR / "plan_d_next_actions_priority.csv"
FEASIBILITY_LEDGER_PATH = RESULTS_TABLES_DIR / "plan_d_feasibility_ledger.csv"
NOTES_PATH = REPORT_DIR / "notes.md"
REPRO_MATRIX_PATH = REPORT_DIR / "plan_d_reproducibility_matrix.md"

TIMEOUT_SECONDS = 180.0

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


def run_diagnostic() -> dict[str, object]:
    """Run the bounded D5E local Qwen2/Ollama diagnostic over the 20-row manifest."""

    manifest = load_and_validate_manifest(MANIFEST_PATH)
    prompt_builder = ManifestPromptBuilder(PROJECT_ROOT / "data" / "reconstructed" / "plan_c")
    client = OllamaClient(timeout_seconds=TIMEOUT_SECONDS)

    evaluations = []
    raw_rows = []
    started = time.perf_counter()
    for index, (_, manifest_row) in enumerate(manifest.iterrows(), start=1):
        payload = prompt_builder.build_from_manifest_row(manifest_row)
        LOGGER.info("Running local Qwen2 request %s/%s for %s", index, len(manifest), payload.instance_id)
        result = client.generate_json(
            model=MODEL,
            prompt=payload.prompt,
            temperature=TEMPERATURE,
        )
        evaluation, parsed = evaluate_response(
            payload=payload,
            response_received=result.response_received,
            raw_response_text=result.response_text,
            latency_seconds=result.latency_seconds,
            error_message=result.error,
        )
        evaluations.append(evaluation)
        raw_rows.append(
            raw_trace_row(
                payload=payload,
                raw_response_text=result.response_text,
                parsed_result=parsed,
                latency_seconds=result.latency_seconds,
                error_message=result.error,
            )
        )
    total_runtime_seconds = time.perf_counter() - started
    aggregate = aggregate_evaluations(
        evaluations,
        total_runtime_seconds=total_runtime_seconds,
        raw_output_path=_relative_path(RAW_OUTPUT_PATH),
    )
    return {
        "evaluations": evaluations,
        "raw_rows": raw_rows,
        "aggregate": aggregate,
    }


def write_outputs(result: dict[str, object]) -> None:
    evaluations = result["evaluations"]
    raw_rows = result["raw_rows"]
    aggregate = result["aggregate"]
    assert isinstance(evaluations, list)
    assert isinstance(raw_rows, list)

    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with RAW_OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    pd.DataFrame([item.row() for item in evaluations], columns=INSTANCE_SUMMARY_COLUMNS).to_csv(
        INSTANCE_SUMMARY_PATH,
        index=False,
    )
    pd.DataFrame([aggregate.summary_row()], columns=AGGREGATE_SUMMARY_COLUMNS).to_csv(
        AGGREGATE_SUMMARY_PATH,
        index=False,
    )
    pd.DataFrame(aggregate.metric_rows(), columns=METRIC_COLUMNS).to_csv(METRICS_PATH, index=False)
    REPORT_PATH.write_text(_diagnostic_report(aggregate), encoding="utf-8")
    _update_task_roadmap(aggregate)
    _update_next_actions(aggregate)
    _update_feasibility_ledger(aggregate)
    _update_notes(aggregate)
    _update_reproducibility_matrix(aggregate)


def _diagnostic_report(aggregate) -> str:
    next_step = (
        "Inspect output validity and hallucination details, then run the candidate position-bias diagnostic."
        if aggregate.status == "completed_diagnostic_subset"
        else "Fix the local Qwen2 ML-1M ranking diagnostic failure before running position-bias, Mistral, or reranking diagnostics."
    )
    return f"""# Plan D Qwen2 ML-1M Ranking Diagnostic

## Purpose

D5E runs the first real local LLM-as-recommender diagnostic experiment in Plan D. It evaluates `qwen2:7b-instruct` through local Ollama on the deterministic 20-user ML-1M manifest prepared in D5D.

This is not exact paper reproduction. It is a bounded reconstructed diagnostic subset using local open-weight inference, reconstructed ML-1M candidate pools, and one prompt strategy.

## Exact Configuration

- Plan: `{PLAN}`
- Phase: `{PHASE}`
- Dataset: `{DATASET}`
- Scope: `{SCOPE}`
- Backend: `{BACKEND}`
- Model: `{MODEL}`
- Prompt strategy: `{PROMPT_STRATEGY}`
- Temperature: `{TEMPERATURE:.0f}`
- Users: `{aggregate.num_instances}`
- Top-k: `{TOP_K}`
- Candidate pool size: `{CANDIDATE_POOL_SIZE}`
- Output format: `strict_json`
- Local only: `true`

## Execution Boundaries

- This was local-only Ollama inference.
- No OpenAI, Anthropic, or remote API was called.
- No model was pulled or downloaded in this phase.
- No recommender model was trained.
- No BPRMF, LightGCN, SASRec, GRU4Rec, or LastFM experiment was run.
- No Plan C output, dataset reconstruction file, recommender raw output, or previous source metric CSV was modified.

## Manifest Summary

- Manifest path: `results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_manifest.csv`
- Manifest rows processed: `{aggregate.num_instances}`
- Required manifest status: `ready_for_llm_inference`
- Required positives in candidates: `true`
- Candidate count per instance: `{CANDIDATE_POOL_SIZE}`

## Prompt Strategy

The run used `openp5_base`, which includes chronological user history, the fixed 20-item candidate list, item titles/metadata where available, and a strict JSON schema requesting exactly five candidate IDs. Full prompts are not stored in committed tables.

## Parser And Output-Validity Checks

Responses were parsed with the strict D5D ranking parser. The evaluator did not repair malformed JSON, did not manually correct invalid IDs, and did not silently drop hallucinated IDs. Utility metrics use the returned parsed ranking order without manual correction; parse failures receive HR@5 = `0` and NDCG@5 = `0`.

## Utility Results

- HR@5: `{aggregate.hit_rate_at_5:.6f}`
- NDCG@5: `{aggregate.ndcg_at_5:.6f}`

## Output Validity Results

- response_received_rate: `{aggregate.response_received_rate:.6f}`
- parse_success_rate: `{aggregate.parse_success_rate:.6f}`
- schema_valid_rate: `{aggregate.schema_valid_rate:.6f}`
- mean_candidate_validity_rate: `{aggregate.mean_candidate_validity_rate:.6f}`
- hallucination_rate: `{aggregate.hallucination_rate:.6f}`
- duplicate_rate: `{aggregate.duplicate_rate:.6f}`
- exact_top_k_rate: `{aggregate.exact_top_k_rate:.6f}`
- empty_output_rate: `{aggregate.empty_output_rate:.6f}`

## Runtime And Latency

- mean_latency_seconds: `{aggregate.mean_latency_seconds:.6f}`
- total_runtime_seconds: `{aggregate.total_runtime_seconds:.6f}`
- raw_output_path: `{aggregate.raw_output_path}`

## Limitations

- Only 20 users.
- Only ML-1M.
- Only `qwen2:7b-instruct`.
- Only `openp5_base`.
- Not 1000 users.
- Not 5 repeated experiments.
- Not all 7 paper LLMs.
- Re-ranking, position bias, and history sensitivity are still pending.
- Candidate pools are reconstructed and diagnostic, not exact paper pools.

## Result Status

- status: `{aggregate.status}`
- notes: {aggregate.notes}

## Recommended Next Steps

{next_step} Optionally pull and run Mistral later, then run the reranking diagnostic after the ranking path is stable.
"""


def _update_task_roadmap(aggregate) -> None:
    frame = _read_or_empty(TASK_ROADMAP_PATH, TASK_ROADMAP_COLUMNS)
    completed = aggregate.status == "completed_diagnostic_subset"
    for index, row in frame.iterrows():
        phase = row["phase"]
        if phase == "D5E":
            frame.at[index, "task"] = "llm_ranking_diagnostic"
            frame.at[index, "dataset"] = DATASET
            frame.at[index, "model"] = MODEL
            frame.at[index, "backend"] = BACKEND
            frame.at[index, "users"] = str(aggregate.num_instances)
            frame.at[index, "candidate_pool_size"] = str(CANDIDATE_POOL_SIZE)
            frame.at[index, "prompt_strategy"] = PROMPT_STRATEGY
            frame.at[index, "output_format"] = "strict_json"
            frame.at[index, "metrics"] = "HR@5,NDCG@5,candidate_validity_rate,duplicate_rate,hallucination_rate,parse_failure_rate,runtime"
            frame.at[index, "prerequisites"] = "D5D manifest ready"
            frame.at[index, "status"] = "completed_diagnostic_subset" if completed else "failed_diagnostic"
            frame.at[index, "notes"] = (
                f"D5E completed local Qwen2/Ollama ranking over {aggregate.num_instances} ML-1M manifest rows; no exact paper reproduction claim."
                if completed
                else f"D5E failed with status={aggregate.status}; inspect D5E report before further LLM diagnostics."
            )
        elif phase in {"D5F", "D5G", "D5H"}:
            frame.at[index, "status"] = "pending_after_d5e" if completed else "blocked_d5e_failure"
            frame.at[index, "prerequisites"] = "D5E completed" if completed else "Fix D5E diagnostic"
            frame.at[index, "notes"] = (
                "Use real D5E parsed outputs; do not fabricate or repair LLM outputs."
                if completed
                else "Blocked until the Qwen2 ML-1M ranking diagnostic is fixed."
            )
    frame.to_csv(TASK_ROADMAP_PATH, index=False)


def _update_next_actions(aggregate) -> None:
    if aggregate.status == "completed_diagnostic_subset":
        rows = [
            _next_action("1", "inspect_qwen2_output_validity_and_hallucination", "Inspect Qwen output validity and hallucination", "hallucination; output_validity", "D5E produced real local Qwen outputs that need validity and hallucination review before broader diagnostics.", "Output-validity inspection grounded in D5E instance summary and raw ignored traces.", "D5E completed.", "medium", "next", "Do not manually correct malformed or hallucinated outputs."),
            _next_action("2", "run_candidate_position_bias_diagnostic", "Run candidate position-bias diagnostic", "position_bias", "Position bias is now meaningful because D5E created real LLM selections.", "Position-bias metrics over D5E parsed outputs.", "D5E outputs.", "medium", "after validity inspection", "Use candidate positions from the manifest/candidate artifacts."),
            _next_action("3", "pull_mistral_7b_instruct_v02_quantized", "Pull Mistral 7B instruct v0.2 quantized", "paper_llm_mistral_7b_instruct_v0_2", "Mistral is the second local paper-family model for output-validity comparison.", "A recommended Mistral v0.2 quantized model appears in ollama list.", "Manual user pull only.", "medium", "after Qwen review", "Do not pull automatically."),
            _next_action("4", "run_mistral_ml1m_ranking_diagnostic_20_users", "Run Mistral ML-1M ranking diagnostic", "llm_ranking; paper_llm_mistral_7b_instruct_v0_2", "A second local model helps compare validity and hallucination behavior.", "20-user Mistral ML-1M ranking diagnostic metrics.", "Mistral pulled and Qwen path stable.", "high", "after Mistral pull", "Keep as diagnostic; no exact paper reproduction claim."),
            _next_action("5", "run_llm_reranking_diagnostic", "Run LLM reranking diagnostic", "llm_reranking", "Reranking is a paper task still pending after the first local ranking diagnostic.", "Reranking diagnostics with HR@5, NDCG@5, validity, and hallucination metrics.", "Stable D5E output-validity review.", "high", "after ranking diagnostics", "Keep as diagnostic_subset unless official pools are recovered."),
            _next_action("6", "run_history_length_sensitivity_diagnostic", "Run history length sensitivity diagnostic", "history_length_sensitivity", "History length sensitivity is one of the paper LLM dimensions not yet evaluated.", "History-length diagnostic over a bounded local LLM subset.", "Stable prompt/parser path.", "medium", "after ranking diagnostics", "Use local Qwen unless scope changes."),
            _next_action("7", "optional_openai_api_diagnostic_if_budget_approved", "Optional OpenAI API diagnostic if budget approved", "api_availability", "Closed-source comparison remains conditional and optional.", "Small approved API diagnostic or documented no-API decision.", "Explicit user approval, API key, and budget.", "high", "optional", "No API calls by default."),
            _next_action("8", "integrate_llm_results_or_blocker_into_final_report", "Integrate LLM results or blocker into final report", "final_report; local_llm_backend", "The final report should now include the real D5E diagnostic results and limitations.", "Final LLM section grounded in generated D5E CSVs.", "D5E completed and reviewed.", "low", "after follow-up diagnostics or blocker decision", "Do not claim exact paper reproduction."),
            _next_action("9", "prepare_exam_presentation_tables", "Prepare exam presentation tables", "final_report", "The oral exam needs compact generated tables for claims and limitations.", "Presentation-ready generated tables.", "Stable final report data tables.", "low", "after final report draft", "Use generated CSV values."),
            _next_action("10", "final_report", "Finalize report", "final_report", "The project needs final claims grounded in generated artifacts.", "Final report with limitations and generated metrics.", "LLM diagnostics integrated.", "low", "final", "Do not invent metric values."),
        ]
    else:
        rows = [
            _next_action("1", "fix_qwen2_ml1m_ranking_diagnostic", "Fix Qwen2 ML-1M ranking diagnostic", "llm_ranking", "D5E did not complete successfully; further LLM diagnostics should wait.", "A corrected local diagnostic path or documented blocker.", "Inspect D5E report and local Ollama state.", "high", "next", f"Current status: {aggregate.status}."),
        ]
    pd.DataFrame(rows, columns=NEXT_ACTION_COLUMNS).to_csv(NEXT_ACTIONS_PATH, index=False)


def _update_feasibility_ledger(aggregate) -> None:
    frame = _read_or_empty(FEASIBILITY_LEDGER_PATH, LEDGER_COLUMNS)
    completed = aggregate.status == "completed_diagnostic_subset"
    _upsert_ledger(
        frame,
        {
            "item_id": "qwen2_ml1m_ranking_diagnostic_20_users",
            "area": "llm_experiment",
            "target": "Qwen2 local Ollama ML-1M ranking diagnostic over 20 users",
            "feasibility_status": "already_done" if completed else "feasible_with_backend",
            "required_resources": "Ollama, qwen2:7b-instruct, D5D manifest, strict parser",
            "estimated_effort": "done" if completed else "requires fix",
            "risk_level": "medium",
            "external_dependency": "yes",
            "recommended_decision": (
                "Inspect output validity and hallucination, then proceed to position-bias diagnostics."
                if completed
                else "Fix D5E before running further LLM diagnostics."
            ),
            "notes": (
                f"D5E status={aggregate.status}; HR@5={aggregate.hit_rate_at_5:.6f}; NDCG@5={aggregate.ndcg_at_5:.6f}; "
                "diagnostic subset only, not exact paper reproduction."
            ),
        },
    )
    for item_id in (
        "llm_diagnostic_ranking",
        "paper_llm_qwen_2_7b_instruct",
        "local_llm_backend",
        "ollama_backend",
        "local_ollama_backend",
    ):
        mask = frame["item_id"] == item_id
        if mask.any():
            frame.loc[mask, "recommended_decision"] = (
                "D5E diagnostic_completed; inspect output validity and position bias next."
                if completed
                else "D5E failed; fix local Qwen diagnostic before broader LLM work."
            )
            frame.loc[mask, "notes"] = (
                f"qwen2 local LLM ranking diagnostic status={aggregate.status}; "
                f"response_received_rate={aggregate.response_received_rate:.6f}; "
                f"parse_success_rate={aggregate.parse_success_rate:.6f}; "
                "no exact paper reproduction claim."
            )
    frame.to_csv(FEASIBILITY_LEDGER_PATH, index=False)


def _update_notes(aggregate) -> None:
    section = f"""
## 2026-07-01 - Plan D D5E: Qwen2 ML-1M Ranking Diagnostic

### Changes

- Ran the bounded local Qwen2/Ollama ML-1M ranking diagnostic over the 20 D5D manifest rows.
- Parsed every response with the strict D5D ranking parser.
- Wrote per-instance, aggregate, metric-style, raw trace, and report artifacts.
- Updated the Plan D LLM roadmap, next actions, feasibility ledger, and reproducibility matrix.

### Commands Run

```powershell
python src/experiments/run_plan_d_qwen2_ml1m_ranking_diagnostic.py
```

### Outputs Generated

- `results/metrics/plan_d_qwen2_ml1m_ranking_diagnostic_metrics.csv`
- `results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_instance_summary.csv`
- `results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_summary.csv`
- `results/raw_outputs/plan_d_qwen2_ml1m_ranking_diagnostic_raw.jsonl` (Git-ignored raw local trace)
- `report/plan_d_qwen2_ml1m_ranking_diagnostic.md`

### Result

- status: `{aggregate.status}`
- num_instances: `{aggregate.num_instances}`
- HR@5: `{aggregate.hit_rate_at_5:.6f}`
- NDCG@5: `{aggregate.ndcg_at_5:.6f}`
- response_received_rate: `{aggregate.response_received_rate:.6f}`
- parse_success_rate: `{aggregate.parse_success_rate:.6f}`
- schema_valid_rate: `{aggregate.schema_valid_rate:.6f}`
- mean_candidate_validity_rate: `{aggregate.mean_candidate_validity_rate:.6f}`
- hallucination_rate: `{aggregate.hallucination_rate:.6f}`
- duplicate_rate: `{aggregate.duplicate_rate:.6f}`
- exact_top_k_rate: `{aggregate.exact_top_k_rate:.6f}`
- empty_output_rate: `{aggregate.empty_output_rate:.6f}`
- mean_latency_seconds: `{aggregate.mean_latency_seconds:.6f}`
- total_runtime_seconds: `{aggregate.total_runtime_seconds:.6f}`

### Notes And Limitations

- This was local-only Ollama inference with `qwen2:7b-instruct`.
- This is not exact paper reproduction and not a full-scope LLM run.
- No OpenAI API, Anthropic API, or remote API was called.
- No model was pulled or downloaded in this phase.
- No recommender model was trained.
- No BPRMF, LightGCN, SASRec, GRU4Rec, or LastFM experiment was run.
- No Plan C output, source metric CSV, recommender raw output, or dataset reconstruction file was modified.
"""
    _replace_or_append_section(NOTES_PATH, "## 2026-07-01 - Plan D D5E: Qwen2 ML-1M Ranking Diagnostic", section)


def _update_reproducibility_matrix(aggregate) -> None:
    if not REPRO_MATRIX_PATH.exists():
        return
    text = REPRO_MATRIX_PATH.read_text(encoding="utf-8")
    intro = (
        "D5B added an LLM model registry and backend feasibility audit before any LLM experiment. "
        "D5C0 verified Ollama and `qwen2:7b-instruct`; D5C1 ran exactly one successful toy local smoke inference. "
        "D5D completed reusable prompt templates, strict parser utilities, and a no-inference ML-1M manifest with 20 ready rows. "
        f"D5E ran the bounded 20-user local Qwen2/Ollama ML-1M ranking diagnostic with status `{aggregate.status}`. "
        "This remains diagnostic_subset work, not exact paper reproduction, and closed-source API reproduction remains conditional on explicit approval, keys, availability, and budget."
    )
    text = _replace_paragraph_starting(text, "D5B added an LLM model registry", intro)
    text = _replace_table_row(
        text,
        "qwen-2-7b-instruct",
        f"| qwen-2-7b-instruct | `diagnostic_completed` | `local_open_model_near_exact` | D5E ran 20-user local Qwen2/Ollama ML-1M ranking diagnostic; HR@5 `{aggregate.hit_rate_at_5:.6f}`, NDCG@5 `{aggregate.ndcg_at_5:.6f}`; not exact paper reproduction. |",
    )
    text = _replace_table_row(
        text,
        "Ollama",
        f"| Ollama | `diagnostic_completed` | `planned` | D5E used local Ollama for 20 Qwen2 ML-1M ranking requests; no remote API was called. |",
    )
    text = _replace_table_row(
        text,
        "Candidate position bias",
        "| Candidate position bias | `CONDITIONAL` | `planned` | Real D5E LLM outputs now exist; position-bias diagnostic is the next analysis step. |",
    )
    text = _replace_table_row(
        text,
        "Hallucination",
        f"| Hallucination | `PARTIAL` | `extension` | D5E measured hallucination_rate `{aggregate.hallucination_rate:.6f}` over the Qwen diagnostic subset; broader reports pending. |",
    )
    text = _replace_table_row(
        text,
        "Output validity and parsing robustness",
        f"| Output validity and parsing robustness | `PARTIAL` | `extension` | D5E measured parse_success_rate `{aggregate.parse_success_rate:.6f}`, schema_valid_rate `{aggregate.schema_valid_rate:.6f}`, and mean_candidate_validity_rate `{aggregate.mean_candidate_validity_rate:.6f}`. |",
    )
    text = _replace_table_row(
        text,
        "Qwen ML-1M ranking diagnostic",
        f"| Qwen ML-1M ranking diagnostic | `completed_diagnostic_subset` | D5E ran local Qwen2/Ollama for 20 ML-1M users with HR@5 `{aggregate.hit_rate_at_5:.6f}` and NDCG@5 `{aggregate.ndcg_at_5:.6f}`. |",
    )
    text = _replace_table_row(
        text,
        "Candidate position bias over real LLM outputs",
        "| Candidate position bias over real LLM outputs | `pending_after_d5e` | Real D5E outputs now exist; run the position-bias diagnostic next. |",
    )
    text = _replace_table_row(
        text,
        "LLM reranking diagnostic",
        "| LLM reranking diagnostic | `pending_after_d5e` | Requires D5E output-validity review plus candidate-pool policy. |",
    )
    text = _replace_table_row(
        text,
        "History length sensitivity",
        "| History length sensitivity | `pending_after_d5e` | Prefer a bounded local Qwen diagnostic after output-validity review. |",
    )
    text = _replace_table_row(
        text,
        "Local LLM diagnostic ranking",
        "| Local LLM diagnostic ranking | `completed_diagnostic_subset` | D5E produced real local Qwen2 diagnostic outputs for 20 ML-1M users; no exact or full-scope LLM reproduction is claimed. |",
    )
    text = _replace_table_row(
        text,
        "Local LLM diagnostic reranking",
        "| Local LLM diagnostic reranking | `pending_after_d5e` | Needs D5E output-validity review plus candidate-pool policy and real LLM outputs. |",
    )
    text = text.replace(
        "LLM ranking, reranking, candidate position bias, and history-length sensitivity are still pending prompt/template integration and real LLM diagnostic outputs. They must be run only over real LLM outputs, never fabricated outputs.",
        "LLM ranking now has a bounded D5E local Qwen2 diagnostic result. Reranking, candidate position bias, and history-length sensitivity remain pending and must use real D5E or future LLM outputs, never fabricated outputs.",
    )
    recommended = """## Recommended next steps

1. Inspect Qwen output validity and hallucination.
2. Run candidate position-bias diagnostics.
3. Pull a quantized Mistral 7B Instruct v0.2 model manually.
4. Run the Mistral ML-1M ranking diagnostic for 20 users.
5. Run LLM reranking diagnostics.
6. Run history length sensitivity diagnostics.
7. Optionally run an OpenAI API diagnostic if budget is explicitly approved.
8. Integrate LLM results or the backend blocker into the final report.
9. Prepare exam presentation tables.
10. Finalize the report.
"""
    text = _replace_section_text(text, "## Recommended next steps", recommended)
    REPRO_MATRIX_PATH.write_text(text, encoding="utf-8")


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


def _relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bounded D5E local Qwen2/Ollama ML-1M ranking diagnostic."
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    result = run_diagnostic()
    write_outputs(result)
    aggregate = result["aggregate"]
    LOGGER.info("Wrote %s", INSTANCE_SUMMARY_PATH)
    LOGGER.info("Wrote %s", AGGREGATE_SUMMARY_PATH)
    LOGGER.info("Wrote %s", METRICS_PATH)
    LOGGER.info("Wrote %s", REPORT_PATH)
    LOGGER.info("D5E status: %s", aggregate.status)


if __name__ == "__main__":
    main()
