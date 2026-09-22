from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_d_llm_reranking_analysis import (
    AGREEMENT_COLUMNS,
    BACKEND,
    CANDIDATE_POOL_SIZE,
    CONDITION_SUMMARY_COLUMNS,
    CONDITIONS,
    DATASET,
    INSTANCE_COLUMNS,
    MANIFEST_COLUMNS,
    METRIC_COLUMNS,
    MISTRAL_CONDITION,
    MISTRAL_MODEL,
    PAIRWISE_COLUMNS,
    PHASE,
    PLAN,
    POSITION_SHIFT_COLUMNS,
    PROMPT_STRATEGY,
    QWEN_CONDITION,
    QWEN_MODEL,
    REFERENCE_CONDITION,
    SCOPE,
    TEMPERATURE,
    TOP_K,
    build_agreement_profile,
    build_condition_summary,
    build_metric_rows,
    build_pairwise_delta,
    build_position_shift_profile,
    build_reranking_manifest,
    build_reranking_prompt,
    evaluate_reranked_ids,
    instance_row,
    parse_candidate_ids,
    raw_trace_row,
    reference_top_k,
    summary_row,
)
from src.llm.plan_d_llm_ranking_parser import parse_llm_ranking_response
from src.llm.plan_d_mistral_smoke_test import check_local_ollama_models, select_local_mistral_model
from src.llm.plan_d_ollama_client import OllamaClient


LOGGER = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_METRICS_DIR = RESULTS_DIR / "metrics"
RAW_OUTPUTS_DIR = RESULTS_DIR / "raw_outputs"
REPORT_DIR = PROJECT_ROOT / "report"

D5E_MANIFEST_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ml1m_ranking_diagnostic_manifest.csv"

D5J_MANIFEST_PATH = RESULTS_TABLES_DIR / "plan_d_llm_ml1m_reranking_manifest.csv"
D5J_INSTANCE_SUMMARY_PATH = RESULTS_TABLES_DIR / "plan_d_llm_ml1m_reranking_instance_summary.csv"
D5J_CONDITION_SUMMARY_PATH = RESULTS_TABLES_DIR / "plan_d_llm_ml1m_reranking_condition_summary.csv"
D5J_PAIRWISE_DELTA_PATH = RESULTS_TABLES_DIR / "plan_d_llm_ml1m_reranking_pairwise_delta.csv"
D5J_AGREEMENT_PROFILE_PATH = RESULTS_TABLES_DIR / "plan_d_llm_ml1m_reranking_agreement_profile.csv"
D5J_POSITION_SHIFT_PROFILE_PATH = RESULTS_TABLES_DIR / "plan_d_llm_ml1m_reranking_position_shift_profile.csv"
D5J_METRICS_PATH = RESULTS_METRICS_DIR / "plan_d_llm_ml1m_reranking_diagnostic_metrics.csv"
D5J_REPORT_PATH = REPORT_DIR / "plan_d_llm_ml1m_reranking_diagnostic.md"
D5J_RAW_PATH = RAW_OUTPUTS_DIR / "plan_d_llm_ml1m_reranking_diagnostic_raw.jsonl"

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
    manifest = _load_and_validate_manifest(D5E_MANIFEST_PATH)
    inventory = check_local_ollama_models()
    if QWEN_MODEL not in set(inventory.model_names):
        raise RuntimeError(f"Required local Qwen model is unavailable: {QWEN_MODEL}")
    if select_local_mistral_model(inventory.model_names, MISTRAL_MODEL) != MISTRAL_MODEL:
        raise RuntimeError(f"Required local Mistral model is unavailable: {MISTRAL_MODEL}")

    d5j_manifest = build_reranking_manifest(manifest)
    client = OllamaClient(timeout_seconds=TIMEOUT_SECONDS)
    instances = []
    raw_rows: list[dict[str, object]] = []

    for _, row in manifest.iterrows():
        candidate_ids = parse_candidate_ids(str(row["candidate_ids"]))
        instances.append(
            evaluate_reranked_ids(
                reference_top_k(candidate_ids, TOP_K),
                condition=REFERENCE_CONDITION,
                user_id=str(row["user_id"]),
                source_instance_id=str(row["instance_id"]),
                diagnostic_id=f"{PHASE}_{REFERENCE_CONDITION}_{str(row['user_id'])}",
                candidate_ids=candidate_ids,
                positive_item_id=str(row["positive_item_id"]),
                response_received=True,
                parse_success=True,
                schema_valid=True,
                latency_seconds=0.0,
                parse_status="reference_from_candidate_order",
                notes="Reference top-5 from original candidate order; no LLM inference.",
            )
        )

    runtimes: dict[str, float] = {}
    request_count = 0
    for condition, model in ((QWEN_CONDITION, QWEN_MODEL), (MISTRAL_CONDITION, MISTRAL_MODEL)):
        started = time.perf_counter()
        for index, (_, row) in enumerate(manifest.iterrows(), start=1):
            candidate_ids = parse_candidate_ids(str(row["candidate_ids"]))
            prompt = build_reranking_prompt(
                user_id=str(row["user_id"]),
                history_preview=str(row.get("history_preview", "")),
                candidate_ids=candidate_ids,
                top_k=TOP_K,
            )
            LOGGER.info("Running D5J %s local request %s/20 for %s", model, index, row["instance_id"])
            request_count += 1
            result = client.generate_json(model=model, prompt=prompt, temperature=TEMPERATURE)
            parsed = parse_llm_ranking_response(
                result.response_text if result.response_received else "",
                candidate_ids=set(candidate_ids),
                expected_top_k=TOP_K,
            )
            instance = evaluate_reranked_ids(
                parsed.ranked_item_ids,
                condition=condition,
                user_id=str(row["user_id"]),
                source_instance_id=str(row["instance_id"]),
                diagnostic_id=f"{PHASE}_{condition}_{str(row['user_id'])}",
                candidate_ids=candidate_ids,
                positive_item_id=str(row["positive_item_id"]),
                response_received=result.response_received,
                parse_success=parsed.parse_success,
                schema_valid=parsed.schema_valid,
                latency_seconds=result.latency_seconds,
                parse_status=parsed.status,
                notes=result.error or parsed.error_message or _notes_for_llm(condition, parsed.hallucinated_count),
            )
            instances.append(instance)
            raw_rows.append(raw_trace_row(instance=instance, raw_response_text=result.response_text, parsed_result=asdict(parsed)))
        runtimes[condition] = time.perf_counter() - started

    instance_summary = pd.DataFrame([instance_row(instance) for instance in instances], columns=INSTANCE_COLUMNS)
    condition_summary = build_condition_summary(instance_summary, runtimes=runtimes)
    pairwise_delta = build_pairwise_delta(condition_summary, build_agreement_profile(instance_summary))
    agreement_profile = build_agreement_profile(instance_summary)
    position_shift_profile = build_position_shift_profile(instance_summary)
    metrics = build_metric_rows(condition_summary)
    return {
        "manifest": d5j_manifest,
        "instance_summary": instance_summary,
        "condition_summary": condition_summary,
        "pairwise_delta": pairwise_delta,
        "agreement_profile": agreement_profile,
        "position_shift_profile": position_shift_profile,
        "metrics": metrics,
        "raw_rows": raw_rows,
        "runtimes": runtimes,
        "request_count": request_count,
    }


def write_outputs(result: dict[str, object]) -> None:
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = result["manifest"]
    instance_summary = result["instance_summary"]
    condition_summary = result["condition_summary"]
    pairwise_delta = result["pairwise_delta"]
    agreement_profile = result["agreement_profile"]
    position_shift_profile = result["position_shift_profile"]
    metrics = result["metrics"]
    raw_rows = result["raw_rows"]
    runtimes = result["runtimes"]
    request_count = int(result["request_count"])

    assert isinstance(manifest, pd.DataFrame)
    assert isinstance(instance_summary, pd.DataFrame)
    assert isinstance(condition_summary, pd.DataFrame)
    assert isinstance(pairwise_delta, pd.DataFrame)
    assert isinstance(agreement_profile, pd.DataFrame)
    assert isinstance(position_shift_profile, pd.DataFrame)
    assert isinstance(metrics, pd.DataFrame)
    assert isinstance(raw_rows, list)
    assert isinstance(runtimes, dict)

    manifest.to_csv(D5J_MANIFEST_PATH, index=False, columns=MANIFEST_COLUMNS)
    instance_summary.to_csv(D5J_INSTANCE_SUMMARY_PATH, index=False, columns=INSTANCE_COLUMNS)
    condition_summary.to_csv(D5J_CONDITION_SUMMARY_PATH, index=False, columns=CONDITION_SUMMARY_COLUMNS)
    pairwise_delta.to_csv(D5J_PAIRWISE_DELTA_PATH, index=False, columns=PAIRWISE_COLUMNS)
    agreement_profile.to_csv(D5J_AGREEMENT_PROFILE_PATH, index=False, columns=AGREEMENT_COLUMNS)
    position_shift_profile.to_csv(D5J_POSITION_SHIFT_PROFILE_PATH, index=False, columns=POSITION_SHIFT_COLUMNS)
    metrics.to_csv(D5J_METRICS_PATH, index=False, columns=METRIC_COLUMNS)
    with D5J_RAW_PATH.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    D5J_REPORT_PATH.write_text(_report(condition_summary, pairwise_delta, agreement_profile, position_shift_profile, request_count), encoding="utf-8")
    _update_notes(condition_summary, request_count, runtimes)
    _update_task_roadmap()
    _update_next_actions()
    _update_feasibility_ledger(condition_summary)
    _update_reproducibility_matrix(condition_summary)


def _report(
    condition_summary: pd.DataFrame,
    pairwise_delta: pd.DataFrame,
    agreement_profile: pd.DataFrame,
    position_shift_profile: pd.DataFrame,
    request_count: int,
) -> str:
    rows = "\n".join(
        f"| `{row['condition']}` | `{row['model']}` | `{row['hr_at_5']}` | `{row['ndcg_at_5']}` | `{row['mean_candidate_validity_rate']}` | `{row['hallucination_rate']}` | `{row['mean_selected_input_position']}` | `{row['mean_absolute_position_shift_vs_reference']}` |"
        for _, row in condition_summary.iterrows()
    )
    delta_rows = "\n".join(
        f"| `{row['comparison']}` | `{row['metric_name']}` | `{row['reference_value']}` | `{row['qwen2_value']}` | `{row['mistral_value']}` | `{row['condition_value']}` | `{row['delta']}` | `{row['pairwise_value']}` |"
        for _, row in pairwise_delta.iterrows()
    )
    agreement_counts = agreement_profile["agreement_category"].value_counts().to_dict()
    shift_summary = position_shift_profile.groupby("model")["position_shift"].apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().abs().mean()).to_dict()
    return f"""# Plan D Local LLM ML-1M Re-Ranking Diagnostic

## Purpose

D5J evaluates whether local LLMs can re-rank an already ordered candidate list on the same ML-1M 20-user diagnostic subset. The original candidate order is treated as a baseline/retriever reference.

## Scope And Classification

- D5J is a diagnostic_subset local LLM re-ranking diagnostic.
- D5J is a robustness/reliability diagnostic.
- D5J is not exact reproduction.
- D5J is not full LLM reproduction.
- D5J uses local Ollama only.
- Qwen2 and Mistral are local open models.
- Exactly {request_count} new local LLM requests were run.
- D5J does not reproduce the paper's full LLM matrix or closed-source setup.

## Execution Boundaries

- No OpenAI API call was made.
- No Anthropic API call was made.
- No external API call was made.
- No download was performed during D5J.
- No model pull was performed during D5J.
- No recommender training was performed.
- Plan C was not modified.
- Raw JSONL outputs are stored under `results/raw_outputs/` and remain Git-ignored.

## Condition Summary

| Condition | Model | HR@5 | NDCG@5 | Candidate Validity | Hallucination Rate | Mean Input Position | Mean Abs Shift |
|---|---|---:|---:|---:|---:|---:|---:|
{rows}

## Pairwise Deltas

| Comparison | Metric | Reference | Qwen2 | Mistral | Condition | Delta | Pairwise |
|---|---|---:|---:|---:|---:|---:|---:|
{delta_rows}

Agreement category counts: `{json.dumps(agreement_counts, sort_keys=True)}`.
Mean absolute position shift by model from the position-shift profile: `{json.dumps({k: round(float(v), 6) for k, v in shift_summary.items()}, sort_keys=True)}`.

## Interpretation

D5J evaluates utility separately from reliability. HR@5 and NDCG@5 show whether re-ranking improves or worsens utility compared with `candidate_order_reference`; candidate validity, hallucination, duplicate rate, parse/schema rates, and exact-top-k rates describe output reliability. Position-shift metrics describe how far each local LLM moves selected items away from the original input order.

D5J supports the broader Beyond Utility argument because a local LLM can produce parse-valid and candidate-valid output while still changing utility and position behavior relative to the original candidate-order reference. These results remain diagnostic_subset results, not exact reproduction, because official paper reranking pools, full user counts, repeated runs, and closed-source models are not reproduced.

## Recommended Next Steps

1. Run history length sensitivity diagnostics.
2. Integrate D5E-D5J results into the final report.
3. Prepare exam presentation tables.
"""


def _update_notes(condition_summary: pd.DataFrame, request_count: int, runtimes: dict[str, float]) -> None:
    reference = summary_row(condition_summary, REFERENCE_CONDITION)
    qwen = summary_row(condition_summary, QWEN_CONDITION)
    mistral = summary_row(condition_summary, MISTRAL_CONDITION)
    section = f"""
## 2026-07-01 - Plan D D5J: Local LLM ML-1M Re-Ranking Diagnostic

### Changes

- Ran D5J local LLM re-ranking diagnostics for Qwen2 and Mistral over the same 20 ML-1M diagnostic instances.
- Added a no-inference `candidate_order_reference` condition using the first 5 original candidate IDs as the baseline top-5.
- Wrote manifest, instance summary, condition summary, pairwise deltas, agreement profile, position-shift profile, metric-style CSV, raw ignored JSONL, and report artifacts.

### Outputs Generated

- `results/metrics/plan_d_llm_ml1m_reranking_diagnostic_metrics.csv`
- `results/tables/plan_d_llm_ml1m_reranking_instance_summary.csv`
- `results/tables/plan_d_llm_ml1m_reranking_condition_summary.csv`
- `results/tables/plan_d_llm_ml1m_reranking_pairwise_delta.csv`
- `results/tables/plan_d_llm_ml1m_reranking_agreement_profile.csv`
- `results/tables/plan_d_llm_ml1m_reranking_position_shift_profile.csv`
- `results/tables/plan_d_llm_ml1m_reranking_manifest.csv`
- `results/raw_outputs/plan_d_llm_ml1m_reranking_diagnostic_raw.jsonl` (Git-ignored)
- `report/plan_d_llm_ml1m_reranking_diagnostic.md`

### Result

- New local LLM requests: `{request_count}`
- Qwen2 runtime seconds: `{runtimes.get(QWEN_CONDITION, 0.0):.6f}`
- Mistral runtime seconds: `{runtimes.get(MISTRAL_CONDITION, 0.0):.6f}`
- Reference HR@5/NDCG@5: `{reference['hr_at_5']}` / `{reference['ndcg_at_5']}`
- Qwen2 HR@5/NDCG@5: `{qwen['hr_at_5']}` / `{qwen['ndcg_at_5']}`
- Mistral HR@5/NDCG@5: `{mistral['hr_at_5']}` / `{mistral['ndcg_at_5']}`

### Notes And Limitations

- D5J is a diagnostic_subset local LLM re-ranking diagnostic, not exact reproduction and not full LLM reproduction.
- No OpenAI API call, Anthropic API call, external API call, download, model pull, recommender training, LastFM experiment, or Plan C modification occurred.
"""
    _replace_or_append_section(NOTES_PATH, "## 2026-07-01 - Plan D D5J: Local LLM ML-1M Re-Ranking Diagnostic", section)


def _update_task_roadmap() -> None:
    frame = _read_or_empty(TASK_ROADMAP_PATH, TASK_ROADMAP_COLUMNS)
    for index, row in frame.iterrows():
        if row["phase"] == "D5J":
            frame.at[index, "task"] = "llm_reranking_diagnostic"
            frame.at[index, "dataset"] = DATASET
            frame.at[index, "model"] = "qwen2:7b-instruct+mistral:7b-instruct"
            frame.at[index, "backend"] = BACKEND
            frame.at[index, "users"] = "20"
            frame.at[index, "candidate_pool_size"] = "20"
            frame.at[index, "prompt_strategy"] = PROMPT_STRATEGY
            frame.at[index, "output_format"] = "strict_json"
            frame.at[index, "metrics"] = "HR@5,NDCG@5,rerank_delta,validity,hallucination,position_shift"
            frame.at[index, "prerequisites"] = "D5I completed and local models available"
            frame.at[index, "status"] = "completed_diagnostic_subset"
            frame.at[index, "notes"] = "D5J completed 40 local LLM reranking requests plus no-inference candidate-order reference."
    frame.to_csv(TASK_ROADMAP_PATH, index=False)


def _update_next_actions() -> None:
    rows = [
        _next_action("1", "run_history_length_sensitivity_diagnostic", "Run history length sensitivity diagnostic", "history_length_sensitivity", "History length sensitivity remains one of the paper LLM dimensions not yet evaluated.", "Bounded local LLM sensitivity diagnostic.", "D5J completed.", "medium", "next", "Use local models only unless scope changes."),
        _next_action("2", "integrate_llm_results_or_blocker_into_final_report", "Integrate LLM results into final report", "final_report; local_llm_backend", "The final report should include D5E-D5J diagnostic limitations.", "Final LLM section grounded in generated CSVs.", "D5J documented.", "low", "after sensitivity decision", "Do not claim exact paper reproduction."),
        _next_action("3", "test_stricter_candidate_id_only_prompt", "Keep strict prompt policy for future diagnostics", "prompting; output_validity", "D5G-D5J use strict candidate-ID-only prompting for local reliability diagnostics.", "Documented prompt policy for later phases.", "D5J completed.", "low", "ongoing", "No extra inference implied."),
        _next_action("4", "run_mistral_ml1m_ranking_diagnostic_20_users", "Use D5I/D5J as Mistral diagnostic evidence", "llm_ranking; paper_llm_mistral_7b_instruct_v0_2", "Mistral has local smoke, ranking-comparison, and reranking diagnostics.", "Mistral diagnostic metrics integrated through D5I/D5J.", "D5I and D5J completed.", "low", "done", "Action retained for continuity with earlier roadmap."),
        _next_action("5", "pull_mistral_7b_instruct_v02_quantized", "Confirm Mistral local model remains available", "paper_llm_mistral_7b_instruct_v0_2", "D5H-D5J used a manually installed local Mistral model.", "Confirmed local model in ollama list.", "None if already installed.", "low", "as needed", "Do not pull automatically."),
        _next_action("6", "optional_openai_api_diagnostic_if_budget_approved", "Optional OpenAI API diagnostic if budget approved", "api_availability", "Closed-source comparison remains conditional.", "Small approved API diagnostic or documented no-API decision.", "Explicit user approval, API key, and budget.", "high", "optional", "No API calls by default."),
        _next_action("7", "prepare_exam_presentation_tables", "Prepare exam presentation tables", "final_report", "The oral exam needs compact generated tables.", "Presentation-ready generated tables.", "Stable final report data tables.", "low", "after final report draft", "Use generated CSV values."),
        _next_action("8", "final_report", "Finalize report", "final_report", "The project needs final claims grounded in generated artifacts.", "Final report with limitations and generated metrics.", "LLM diagnostics integrated.", "low", "final", "Do not invent metric values."),
    ]
    pd.DataFrame(rows, columns=NEXT_ACTION_COLUMNS).to_csv(NEXT_ACTIONS_PATH, index=False)


def _update_feasibility_ledger(condition_summary: pd.DataFrame) -> None:
    frame = _read_or_empty(FEASIBILITY_LEDGER_PATH, LEDGER_COLUMNS)
    qwen = summary_row(condition_summary, QWEN_CONDITION)
    mistral = summary_row(condition_summary, MISTRAL_CONDITION)
    _upsert_ledger(
        frame,
        {
            "item_id": "local_llm_reranking_diagnostic",
            "area": "llm_analysis",
            "target": "Qwen2 and Mistral ML-1M re-ranking diagnostic",
            "feasibility_status": "already_done",
            "required_resources": "D5E manifest, local Ollama, qwen2:7b-instruct, mistral:7b-instruct",
            "estimated_effort": "done",
            "risk_level": "medium",
            "external_dependency": "yes",
            "recommended_decision": "Use D5J findings before final LLM report integration.",
            "notes": f"D5J completed; Qwen HR@5={qwen['hr_at_5']}, Mistral HR@5={mistral['hr_at_5']}.",
        },
    )
    mask = frame["item_id"] == "llm_diagnostic_reranking"
    if mask.any():
        frame.loc[mask, "feasibility_status"] = "already_done"
        frame.loc[mask, "recommended_decision"] = "Use D5J re-ranking results in the final diagnostic LLM section."
        frame.loc[mask, "notes"] = "D5J completed as diagnostic_subset; not exact LLM reproduction."
    frame.to_csv(FEASIBILITY_LEDGER_PATH, index=False)


def _update_reproducibility_matrix(condition_summary: pd.DataFrame) -> None:
    if not REPRO_MATRIX_PATH.exists():
        return
    qwen = summary_row(condition_summary, QWEN_CONDITION)
    mistral = summary_row(condition_summary, MISTRAL_CONDITION)
    text = REPRO_MATRIX_PATH.read_text(encoding="utf-8")
    if "D5J completed local LLM reranking" not in text:
        text = text.replace(
            "D5I compared Qwen2 and Mistral on the same strict-prompt ML-1M diagnostic subset; Qwen2 was not rerun.",
            "D5I compared Qwen2 and Mistral on the same strict-prompt ML-1M diagnostic subset; Qwen2 was not rerun. D5J completed local LLM reranking as diagnostic_subset work.",
        )
    text = _replace_table_row(
        text,
        "Re-ranking task",
        "| Re-ranking task | LLM reranks recommender-generated pools | `DIAGNOSTIC_ONLY` | `diagnostic` | D5J completed local open-model reranking over reconstructed 20-user ML-1M candidate-order reference; not exact paper pools. |",
    )
    text = _replace_table_row(
        text,
        "Local LLM diagnostic reranking",
        "| Local LLM diagnostic reranking | `completed_diagnostic_subset` | D5J completed Qwen2 and Mistral local reranking diagnostics against candidate_order_reference. |",
    )
    text = _insert_after_line(
        text,
        "| Qwen2 vs Mistral local strict-prompt comparison | `DIAGNOSTIC_ONLY` | `diagnostic` | D5I compared local open models on 20 ML-1M strict-prompt instances; not exact reproduction. |",
        f"| Local LLM re-ranking diagnostic | `DIAGNOSTIC_ONLY` | `diagnostic` | D5J completed Qwen2 and Mistral reranking diagnostics; Qwen HR@5 `{qwen['hr_at_5']}`, Mistral HR@5 `{mistral['hr_at_5']}`. |",
    )
    text = _replace_section_text(
        text,
        "## Recommended next steps",
        """## Recommended next steps

1. Run history length sensitivity diagnostics.
2. Integrate D5E-D5J LLM diagnostic results into the final report.
3. Optionally run an OpenAI API diagnostic if budget is explicitly approved.
4. Prepare exam presentation tables.
5. Finalize the report.
""",
    )
    REPRO_MATRIX_PATH.write_text(text, encoding="utf-8")


def _load_and_validate_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if len(frame) != 20:
        raise ValueError(f"D5J requires exactly 20 manifest rows, found {len(frame)}")
    for _, row in frame.iterrows():
        candidate_ids = parse_candidate_ids(str(row["candidate_ids"]))
        if len(candidate_ids) != CANDIDATE_POOL_SIZE:
            raise ValueError(f"{row['instance_id']} has {len(candidate_ids)} candidates")
        if str(row["positive_item_id"]) not in set(candidate_ids):
            raise ValueError(f"{row['instance_id']} positive item missing from candidates")
    return frame


def _notes_for_llm(condition: str, hallucinated_count: int) -> str:
    if hallucinated_count > 0:
        return f"{condition} output contains IDs outside the candidate set."
    return f"{condition} output evaluated without manual correction."


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


def _insert_after_line(text: str, anchor: str, line_to_insert: str) -> str:
    if line_to_insert in text:
        return text
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line == anchor:
            lines.insert(index + 1, line_to_insert)
            return "\n".join(lines) + "\n"
    return text.rstrip() + "\n" + line_to_insert + "\n"


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
    parser = argparse.ArgumentParser(description="Run D5J local LLM ML-1M reranking diagnostic.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    result = run_diagnostic()
    write_outputs(result)
    LOGGER.info("Wrote %s", D5J_CONDITION_SUMMARY_PATH)
    LOGGER.info("Wrote %s", D5J_REPORT_PATH)


if __name__ == "__main__":
    main()
