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

from src.analysis.plan_d_qwen_vs_mistral_comparison_analysis import (
    AGREEMENT_COLUMNS,
    BACKEND,
    CANDIDATE_POOL_SIZE,
    DATASET,
    INSTANCE_COLUMNS,
    MANIFEST_COLUMNS,
    METRIC_COLUMNS,
    MISTRAL_CONDITION,
    MISTRAL_MODEL,
    MODEL_SUMMARY_COLUMNS,
    PAIRWISE_COLUMNS,
    PHASE,
    PLAN,
    PROMPT_STRATEGY,
    QWEN_CONDITION,
    QWEN_MODEL,
    SCOPE,
    TEMPERATURE,
    TOP_K,
    add_selected_id_columns,
    build_agreement_profile,
    build_metric_rows,
    build_mistral_manifest,
    build_model_summary,
    build_pairwise_delta,
    candidate_order_signature,
    evaluate_ranked_ids,
    instance_row,
    parse_candidate_ids,
    raw_mistral_trace_row,
    summary_row,
)
from src.analysis.plan_d_qwen2_strict_prompt_analysis import build_strict_candidate_id_prompt
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
D5G_RAW_PATH = RAW_OUTPUTS_DIR / "plan_d_qwen2_ml1m_strict_prompt_diagnostic_raw.jsonl"

D5I_MANIFEST_PATH = RESULTS_TABLES_DIR / "plan_d_qwen_vs_mistral_ml1m_manifest.csv"
D5I_INSTANCE_SUMMARY_PATH = RESULTS_TABLES_DIR / "plan_d_qwen_vs_mistral_ml1m_instance_summary.csv"
D5I_MODEL_SUMMARY_PATH = RESULTS_TABLES_DIR / "plan_d_qwen_vs_mistral_ml1m_model_summary.csv"
D5I_PAIRWISE_DELTA_PATH = RESULTS_TABLES_DIR / "plan_d_qwen_vs_mistral_ml1m_pairwise_delta.csv"
D5I_AGREEMENT_PROFILE_PATH = RESULTS_TABLES_DIR / "plan_d_qwen_vs_mistral_ml1m_agreement_profile.csv"
D5I_METRICS_PATH = RESULTS_METRICS_DIR / "plan_d_qwen_vs_mistral_ml1m_diagnostic_metrics.csv"
D5I_REPORT_PATH = REPORT_DIR / "plan_d_qwen_vs_mistral_ml1m_diagnostic.md"
D5I_RAW_PATH = RAW_OUTPUTS_DIR / "plan_d_qwen_vs_mistral_ml1m_diagnostic_raw.jsonl"

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
    qwen_raw = _load_qwen_reference_rows(D5G_RAW_PATH)
    inventory = check_local_ollama_models()
    selected_model = select_local_mistral_model(inventory.model_names, MISTRAL_MODEL)
    if selected_model != MISTRAL_MODEL:
        raise RuntimeError(f"Required local Mistral model is unavailable: {MISTRAL_MODEL}")

    mistral_manifest = build_mistral_manifest(manifest)
    client = OllamaClient(timeout_seconds=TIMEOUT_SECONDS)
    qwen_raw_by_source = {str(row["source_instance_id"]): row for row in qwen_raw}

    instances = []
    raw_rows: list[dict[str, object]] = []
    local_mistral_requests = 0
    started = time.perf_counter()
    for index, (_, manifest_row) in enumerate(manifest.iterrows(), start=1):
        source_instance_id = str(manifest_row["instance_id"])
        candidate_ids = parse_candidate_ids(str(manifest_row["candidate_ids"]))
        positive_item_id = str(manifest_row["positive_item_id"])
        qwen_instance = _qwen_instance_from_raw(
            raw_row=qwen_raw_by_source[source_instance_id],
            manifest_row=manifest_row,
            candidate_ids=candidate_ids,
            positive_item_id=positive_item_id,
        )
        instances.append(qwen_instance)

        prompt = build_strict_candidate_id_prompt(
            user_id=str(manifest_row["user_id"]),
            history_preview=str(manifest_row.get("history_preview", "")),
            candidate_ids=candidate_ids,
            top_k=TOP_K,
        )
        diagnostic_id = f"{PHASE}_{MISTRAL_CONDITION}_{str(manifest_row['user_id'])}"
        LOGGER.info("Running D5I local Mistral request %s/20 for %s", index, source_instance_id)
        local_mistral_requests += 1
        result = client.generate_json(model=MISTRAL_MODEL, prompt=prompt, temperature=TEMPERATURE)
        parsed = parse_llm_ranking_response(
            result.response_text if result.response_received else "",
            candidate_ids=set(candidate_ids),
            expected_top_k=TOP_K,
        )
        mistral_instance = evaluate_ranked_ids(
            parsed.ranked_item_ids,
            condition=MISTRAL_CONDITION,
            user_id=str(manifest_row["user_id"]),
            source_instance_id=source_instance_id,
            diagnostic_id=diagnostic_id,
            candidate_ids=candidate_ids,
            positive_item_id=positive_item_id,
            response_received=result.response_received,
            parse_success=parsed.parse_success,
            schema_valid=parsed.schema_valid,
            latency_seconds=result.latency_seconds,
            parse_status=parsed.status,
            notes=result.error or parsed.error_message or _notes_for_mistral(parsed.hallucinated_count),
        )
        instances.append(mistral_instance)
        raw_rows.append(
            raw_mistral_trace_row(
                instance=mistral_instance,
                raw_response_text=result.response_text,
                parsed_result=asdict(parsed),
            )
        )
    mistral_runtime_seconds = time.perf_counter() - started
    instance_summary_base = pd.DataFrame([instance_row(instance) for instance in instances], columns=INSTANCE_COLUMNS)
    instance_summary_augmented = add_selected_id_columns(instance_summary_base, instances)
    model_summary = build_model_summary(instance_summary_augmented, mistral_runtime_seconds=mistral_runtime_seconds)
    agreement_profile = build_agreement_profile(instance_summary_augmented)
    pairwise_delta = build_pairwise_delta(model_summary, agreement_profile)
    metrics = build_metric_rows(model_summary)
    return {
        "manifest": mistral_manifest,
        "instance_summary": instance_summary_base,
        "model_summary": model_summary,
        "agreement_profile": agreement_profile,
        "pairwise_delta": pairwise_delta,
        "metrics": metrics,
        "raw_rows": raw_rows,
        "mistral_runtime_seconds": mistral_runtime_seconds,
        "local_mistral_requests": local_mistral_requests,
    }


def write_outputs(result: dict[str, object]) -> None:
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = result["manifest"]
    instance_summary = result["instance_summary"]
    model_summary = result["model_summary"]
    agreement_profile = result["agreement_profile"]
    pairwise_delta = result["pairwise_delta"]
    metrics = result["metrics"]
    raw_rows = result["raw_rows"]
    mistral_runtime_seconds = float(result["mistral_runtime_seconds"])
    local_mistral_requests = int(result["local_mistral_requests"])

    assert isinstance(manifest, pd.DataFrame)
    assert isinstance(instance_summary, pd.DataFrame)
    assert isinstance(model_summary, pd.DataFrame)
    assert isinstance(agreement_profile, pd.DataFrame)
    assert isinstance(pairwise_delta, pd.DataFrame)
    assert isinstance(metrics, pd.DataFrame)
    assert isinstance(raw_rows, list)

    manifest.to_csv(D5I_MANIFEST_PATH, index=False, columns=MANIFEST_COLUMNS)
    instance_summary.to_csv(D5I_INSTANCE_SUMMARY_PATH, index=False, columns=INSTANCE_COLUMNS)
    model_summary.to_csv(D5I_MODEL_SUMMARY_PATH, index=False, columns=MODEL_SUMMARY_COLUMNS)
    pairwise_delta.to_csv(D5I_PAIRWISE_DELTA_PATH, index=False, columns=PAIRWISE_COLUMNS)
    agreement_profile.to_csv(D5I_AGREEMENT_PROFILE_PATH, index=False, columns=AGREEMENT_COLUMNS)
    metrics.to_csv(D5I_METRICS_PATH, index=False, columns=METRIC_COLUMNS)
    with D5I_RAW_PATH.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    D5I_REPORT_PATH.write_text(
        _report(model_summary, pairwise_delta, agreement_profile, mistral_runtime_seconds, local_mistral_requests),
        encoding="utf-8",
    )
    _update_notes(model_summary, pairwise_delta, mistral_runtime_seconds, local_mistral_requests)
    _update_task_roadmap()
    _update_next_actions()
    _update_feasibility_ledger(model_summary)
    _update_reproducibility_matrix(model_summary)


def _qwen_instance_from_raw(
    *,
    raw_row: dict[str, object],
    manifest_row: pd.Series,
    candidate_ids: list[str],
    positive_item_id: str,
):
    parsed = raw_row.get("parsed_result", {})
    if not isinstance(parsed, dict):
        parsed = {}
    return evaluate_ranked_ids(
        [str(item_id) for item_id in parsed.get("ranked_item_ids", [])],
        condition=QWEN_CONDITION,
        user_id=str(manifest_row["user_id"]),
        source_instance_id=str(manifest_row["instance_id"]),
        diagnostic_id=f"{PHASE}_{QWEN_CONDITION}_{str(manifest_row['user_id'])}",
        candidate_ids=candidate_ids,
        positive_item_id=positive_item_id,
        response_received=bool(raw_row.get("raw_response_text", "")),
        parse_success=bool(parsed.get("parse_success", False)),
        schema_valid=bool(parsed.get("schema_valid", False)),
        latency_seconds=float(raw_row.get("latency_seconds", 0.0) or 0.0),
        parse_status=str(parsed.get("status", "")),
        notes="Reused D5G Qwen2 strict-prompt output; Qwen2 was not rerun.",
    )


def _report(
    model_summary: pd.DataFrame,
    pairwise_delta: pd.DataFrame,
    agreement_profile: pd.DataFrame,
    mistral_runtime_seconds: float,
    local_mistral_requests: int,
) -> str:
    summary_rows = "\n".join(
        f"| `{row['model']}` | `{row['condition']}` | `{row['hr_at_5']}` | `{row['ndcg_at_5']}` | `{row['mean_candidate_validity_rate']}` | `{row['hallucination_rate']}` | `{row['parse_success_rate']}` | `{row['schema_valid_rate']}` | `{row['exact_top_k_rate']}` | `{row['mean_latency_seconds']}` |"
        for _, row in model_summary.iterrows()
    )
    delta_rows = "\n".join(
        f"| `{row['metric_name']}` | `{row['qwen2_value']}` | `{row['mistral_value']}` | `{row['delta']}` | `{row['pairwise_value']}` |"
        for _, row in pairwise_delta.iterrows()
    )
    agreement_counts = agreement_profile["agreement_category"].value_counts().to_dict()
    return f"""# Plan D Qwen2 vs Mistral ML-1M Strict-Prompt Diagnostic

## Purpose

D5I compares local Qwen2 and local Mistral on the same ML-1M 20-user diagnostic subset using the strict candidate-ID-only prompt setup introduced in D5G.

## Scope And Classification

- D5I is a diagnostic_subset local open-model comparison.
- D5I is a robustness/reliability diagnostic.
- D5I is not exact reproduction.
- D5I is not full LLM reproduction.
- D5I compares local open models only and does not reproduce the paper's closed-source LLM matrix.
- Qwen2 outputs are reused from D5G and are not rerun.
- Mistral uses local Ollama only.
- Exactly {local_mistral_requests} new Mistral requests were run.
- Mistral was installed manually before D5I/D5H completion.
- Temperature: `0`.
- Top-k: `5`.
- Candidate pool size: `20`.
- Mistral runtime seconds: `{mistral_runtime_seconds:.6f}`.

## Execution Boundaries

- No OpenAI API call was made.
- No Anthropic API call was made.
- No external API call was made.
- No download was performed during D5I.
- No model pull was performed during D5I.
- No recommender training was performed.
- Plan C was not modified.
- Raw JSONL outputs are stored under `results/raw_outputs/` and remain Git-ignored.

## Model Summary

| Model | Condition | HR@5 | NDCG@5 | Candidate Validity | Hallucination Rate | Parse Success | Schema Valid | Exact Top-k | Mean Latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{summary_rows}

## Pairwise Delta And Agreement

| Metric | Qwen2 | Mistral | Delta | Pairwise Value |
|---|---:|---:|---:|---:|
{delta_rows}

Agreement category counts: `{json.dumps(agreement_counts, sort_keys=True)}`.

## Interpretation

Utility differences are captured by HR@5 and NDCG@5 deltas. Reliability differences are captured by parse/schema rates, candidate validity, hallucination rate, duplicate rate, and exact top-k rate. Agreement metrics show whether the two local open models hit the same positive instances or diverge even when both follow the strict candidate-ID output format.

D5I supports the broader Beyond Utility argument because two parseable local LLM recommenders can differ in candidate adherence, agreement, latency, and hit behavior even under the same candidate-ID-only constraints. These local diagnostic results are useful for method tracing, but they are diagnostic_subset results rather than exact paper reproduction because the official paper candidate pools, full 1000-user runs, repeated trials, and full LLM matrix are not reproduced here.

## Recommended Next Steps

1. Inspect Mistral validity and agreement cases before scaling.
2. Run LLM reranking diagnostics.
3. Run history length sensitivity diagnostics.
4. Integrate D5E-D5I findings into the final report.
"""


def _update_notes(
    model_summary: pd.DataFrame,
    pairwise_delta: pd.DataFrame,
    mistral_runtime_seconds: float,
    local_mistral_requests: int,
) -> None:
    qwen = summary_row(model_summary, QWEN_CONDITION)
    mistral = summary_row(model_summary, MISTRAL_CONDITION)
    section = f"""
## 2026-07-01 - Plan D D5I: Qwen2 vs Mistral Strict-Prompt Diagnostic

### Changes

- Ran the D5I local Mistral/Ollama strict candidate-ID-only diagnostic over the same 20 ML-1M D5G instances.
- Reused D5G Qwen2 strict-prompt outputs as the Qwen2 reference; Qwen2 was not rerun.
- Wrote manifest, instance summary, model summary, pairwise deltas, agreement profile, metric-style CSV, raw ignored JSONL, and report artifacts.

### Outputs Generated

- `results/metrics/plan_d_qwen_vs_mistral_ml1m_diagnostic_metrics.csv`
- `results/tables/plan_d_qwen_vs_mistral_ml1m_instance_summary.csv`
- `results/tables/plan_d_qwen_vs_mistral_ml1m_model_summary.csv`
- `results/tables/plan_d_qwen_vs_mistral_ml1m_pairwise_delta.csv`
- `results/tables/plan_d_qwen_vs_mistral_ml1m_agreement_profile.csv`
- `results/tables/plan_d_qwen_vs_mistral_ml1m_manifest.csv`
- `results/raw_outputs/plan_d_qwen_vs_mistral_ml1m_diagnostic_raw.jsonl` (Git-ignored)
- `report/plan_d_qwen_vs_mistral_ml1m_diagnostic.md`

### Result

- New local Mistral requests: `{local_mistral_requests}`
- Mistral runtime seconds: `{mistral_runtime_seconds:.6f}`
- Qwen HR@5/NDCG@5: `{qwen['hr_at_5']}` / `{qwen['ndcg_at_5']}`
- Mistral HR@5/NDCG@5: `{mistral['hr_at_5']}` / `{mistral['ndcg_at_5']}`
- Qwen hallucination rate: `{qwen['hallucination_rate']}`
- Mistral hallucination rate: `{mistral['hallucination_rate']}`

### Notes And Limitations

- D5I is a diagnostic_subset local open-model comparison, not exact reproduction and not full LLM reproduction.
- No OpenAI API call, Anthropic API call, external API call, download, model pull, recommender training, LastFM experiment, or Plan C modification occurred.
"""
    _replace_or_append_section(NOTES_PATH, "## 2026-07-01 - Plan D D5I: Qwen2 vs Mistral Strict-Prompt Diagnostic", section)


def _update_task_roadmap() -> None:
    frame = _read_or_empty(TASK_ROADMAP_PATH, TASK_ROADMAP_COLUMNS)
    for index, row in frame.iterrows():
        if row["phase"] == "D5I":
            frame.at[index, "task"] = "qwen_vs_mistral_ml1m_strict_prompt_diagnostic"
            frame.at[index, "dataset"] = DATASET
            frame.at[index, "model"] = "qwen2:7b-instruct+mistral:7b-instruct"
            frame.at[index, "backend"] = BACKEND
            frame.at[index, "users"] = "20"
            frame.at[index, "candidate_pool_size"] = "20"
            frame.at[index, "prompt_strategy"] = PROMPT_STRATEGY
            frame.at[index, "output_format"] = "strict_json"
            frame.at[index, "metrics"] = "HR@5,NDCG@5,candidate_validity_rate,hallucination_rate,agreement,jaccard"
            frame.at[index, "prerequisites"] = "D5G Qwen strict outputs and D5H Mistral smoke test passed"
            frame.at[index, "status"] = "completed_diagnostic_subset"
            frame.at[index, "notes"] = "D5I reused Qwen2 D5G outputs and ran 20 local Mistral requests; not exact reproduction."
    frame.to_csv(TASK_ROADMAP_PATH, index=False)


def _update_next_actions() -> None:
    rows = [
        _next_action("1", "run_llm_reranking_diagnostic", "Run LLM reranking diagnostic", "llm_reranking", "Ranking, validity, position-bias, strict-prompt, and Qwen-vs-Mistral diagnostics are now available.", "Reranking diagnostics with validity and utility metrics.", "D5I completed.", "high", "next", "Keep as diagnostic_subset unless official pools are recovered."),
        _next_action("2", "run_history_length_sensitivity_diagnostic", "Run history length sensitivity diagnostic", "history_length_sensitivity", "History length sensitivity remains one of the paper LLM dimensions not yet evaluated.", "Bounded local LLM sensitivity diagnostic.", "Stable prompt/parser path.", "medium", "after reranking planning", "Use local models only unless scope changes."),
        _next_action("3", "integrate_llm_results_or_blocker_into_final_report", "Integrate LLM results into final report", "final_report; local_llm_backend", "The final report should include D5E-D5I diagnostic limitations.", "Final LLM section grounded in generated CSVs.", "D5I documented.", "low", "after follow-up diagnostics", "Do not claim exact paper reproduction."),
        _next_action("4", "test_stricter_candidate_id_only_prompt", "Keep strict prompt policy for future diagnostics", "prompting; output_validity", "D5G/D5I use strict candidate-ID-only prompting for local model reliability comparison.", "Documented prompt policy for later phases.", "D5G and D5I completed.", "low", "ongoing", "No extra inference implied."),
        _next_action("5", "run_mistral_ml1m_ranking_diagnostic_20_users", "Use D5I as the Mistral ranking diagnostic reference", "llm_ranking; paper_llm_mistral_7b_instruct_v0_2", "D5I already ran the 20-user local Mistral strict-prompt condition.", "Mistral diagnostic metrics integrated through D5I.", "D5I completed.", "low", "done", "Action retained for continuity with earlier roadmap."),
        _next_action("6", "pull_mistral_7b_instruct_v02_quantized", "Confirm Mistral local model remains available", "paper_llm_mistral_7b_instruct_v0_2", "D5H/D5I used a manually installed local Mistral model.", "Confirmed local model in ollama list.", "None if already installed.", "low", "as needed", "Do not pull automatically."),
        _next_action("7", "optional_openai_api_diagnostic_if_budget_approved", "Optional OpenAI API diagnostic if budget approved", "api_availability", "Closed-source comparison remains conditional.", "Small approved API diagnostic or documented no-API decision.", "Explicit user approval, API key, and budget.", "high", "optional", "No API calls by default."),
        _next_action("8", "prepare_exam_presentation_tables", "Prepare exam presentation tables", "final_report", "The oral exam needs compact generated tables.", "Presentation-ready generated tables.", "Stable final report data tables.", "low", "after final report draft", "Use generated CSV values."),
        _next_action("9", "final_report", "Finalize report", "final_report", "The project needs final claims grounded in generated artifacts.", "Final report with limitations and generated metrics.", "LLM diagnostics integrated.", "low", "final", "Do not invent metric values."),
    ]
    pd.DataFrame(rows, columns=NEXT_ACTION_COLUMNS).to_csv(NEXT_ACTIONS_PATH, index=False)


def _update_feasibility_ledger(model_summary: pd.DataFrame) -> None:
    frame = _read_or_empty(FEASIBILITY_LEDGER_PATH, LEDGER_COLUMNS)
    mistral = summary_row(model_summary, MISTRAL_CONDITION)
    _upsert_ledger(
        frame,
        {
            "item_id": "qwen_vs_mistral_strict_prompt_diagnostic",
            "area": "llm_analysis",
            "target": "Qwen2 vs Mistral ML-1M strict-prompt diagnostic comparison",
            "feasibility_status": "already_done",
            "required_resources": "D5G Qwen strict outputs, local Ollama, local mistral:7b-instruct",
            "estimated_effort": "done",
            "risk_level": "medium",
            "external_dependency": "yes",
            "recommended_decision": "Use D5I findings before reranking or final-report integration.",
            "notes": f"D5I completed 20 Mistral local requests; Mistral HR@5={mistral['hr_at_5']}, hallucination_rate={mistral['hallucination_rate']}.",
        },
    )
    for item_id in ("paper_llm_mistral_7b_instruct_v0_2", "mistral_local_smoke_test", "llm_diagnostic_ranking"):
        mask = frame["item_id"] == item_id
        if mask.any():
            frame.loc[mask, "feasibility_status"] = "already_done"
            frame.loc[mask, "recommended_decision"] = "Use D5I Qwen-vs-Mistral comparison results before broader diagnostics."
            frame.loc[mask, "notes"] = "D5I completed local open-model comparison as diagnostic_subset; not exact LLM reproduction."
    frame.to_csv(FEASIBILITY_LEDGER_PATH, index=False)


def _update_reproducibility_matrix(model_summary: pd.DataFrame) -> None:
    if not REPRO_MATRIX_PATH.exists():
        return
    qwen = summary_row(model_summary, QWEN_CONDITION)
    mistral = summary_row(model_summary, MISTRAL_CONDITION)
    text = REPRO_MATRIX_PATH.read_text(encoding="utf-8")
    insertion = " D5I compared Qwen2 and Mistral on the same strict-prompt ML-1M diagnostic subset; Qwen2 was not rerun."
    if "D5I compared Qwen2 and Mistral" not in text:
        text = text.replace(
            "D5H checked local Mistral availability",
            insertion.strip() + " D5H checked local Mistral availability",
        )
    text = _replace_table_row(
        text,
        "qwen-2-7b-instruct",
        f"| qwen-2-7b-instruct | `diagnostic_completed_mitigation_analyzed` | `local_open_model_near_exact` | D5E-D5I include Qwen2 ranking, validity, position-bias, strict-prompt, and Qwen-vs-Mistral comparison diagnostics; D5I reused Qwen HR@5 `{qwen['hr_at_5']}` without rerunning Qwen2. |",
    )
    text = _replace_table_row(
        text,
        "mistral-7b-instruct-v0.2",
        f"| mistral-7b-instruct-v0.2 | `diagnostic_completed_local_comparison` | `local_open_model_near_exact` | D5H smoke test passed after manual install; D5I ran local Mistral strict-prompt diagnostic with HR@5 `{mistral['hr_at_5']}` and hallucination_rate `{mistral['hallucination_rate']}`. |",
    )
    text = _replace_table_row(
        text,
        "Mistral local comparison diagnostic",
        f"| Mistral local comparison diagnostic | `completed_diagnostic_subset` | D5I compared local Qwen2 and Mistral on the same 20 ML-1M strict-prompt instances. |",
    )
    text = _insert_after_line(
        text,
        "| Strict candidate-ID prompt and post-filter mitigation | `DIAGNOSTIC_ONLY` | `diagnostic` | D5G completed a diagnostic_subset prompt mitigation analysis; not exact reproduction and not full LLM reproduction. |",
        "| Qwen2 vs Mistral local strict-prompt comparison | `DIAGNOSTIC_ONLY` | `diagnostic` | D5I compared local open models on 20 ML-1M strict-prompt instances; not exact reproduction. |",
    )
    text = _replace_section_text(
        text,
        "## Recommended next steps",
        """## Recommended next steps

1. Run LLM reranking diagnostics.
2. Run history length sensitivity diagnostics.
3. Integrate D5E-D5I LLM diagnostic results into the final report.
4. Optionally run an OpenAI API diagnostic if budget is explicitly approved.
5. Prepare exam presentation tables.
6. Finalize the report.
""",
    )
    REPRO_MATRIX_PATH.write_text(text, encoding="utf-8")


def _load_and_validate_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if len(frame) != 20:
        raise ValueError(f"D5I requires exactly 20 D5E manifest rows, found {len(frame)}")
    for _, row in frame.iterrows():
        candidate_ids = parse_candidate_ids(str(row["candidate_ids"]))
        if len(candidate_ids) != CANDIDATE_POOL_SIZE:
            raise ValueError(f"{row['instance_id']} has {len(candidate_ids)} candidates")
        if str(row["positive_item_id"]) not in set(candidate_ids):
            raise ValueError(f"{row['instance_id']} positive item missing from candidates")
    return frame


def _load_qwen_reference_rows(path: Path) -> list[dict[str, object]]:
    rows = _read_jsonl(path)
    if len(rows) != 20:
        raise ValueError(f"D5I requires 20 D5G Qwen raw rows, found {len(rows)}")
    return rows


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _notes_for_mistral(hallucinated_count: int) -> str:
    if hallucinated_count > 0:
        return "Mistral strict-prompt output contains IDs outside the candidate set."
    return "Mistral strict-prompt output evaluated without manual correction."


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
    parser = argparse.ArgumentParser(description="Run D5I Qwen2 vs Mistral strict-prompt ML-1M diagnostic.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    result = run_diagnostic()
    write_outputs(result)
    LOGGER.info("Wrote %s", D5I_MODEL_SUMMARY_PATH)
    LOGGER.info("Wrote %s", D5I_REPORT_PATH)


if __name__ == "__main__":
    main()
