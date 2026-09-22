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

from src.analysis.plan_d_history_length_sensitivity_analysis import (
    BACKEND,
    CANDIDATE_POOL_SIZE,
    CONDITION_SUMMARY_COLUMNS,
    DATASET,
    HISTORY_LIMITS,
    HISTORY_SETTINGS,
    INSTANCE_COLUMNS,
    LONG_HISTORY,
    MANIFEST_COLUMNS,
    MEDIUM_HISTORY,
    METRIC_COLUMNS,
    MISTRAL_MODEL,
    PAIRWISE_COLUMNS,
    PHASE,
    PLAN,
    PROMPT_STRATEGY,
    QWEN_MODEL,
    SCOPE,
    SHORT_HISTORY,
    STABILITY_COLUMNS,
    TEMPERATURE,
    TOP_K,
    HistoryItem,
    build_condition_summary,
    build_history_length_manifest,
    build_history_length_prompt,
    build_metric_rows,
    build_pairwise_delta,
    build_stability_profile,
    condition_for,
    evaluate_history_output,
    instance_row,
    parse_candidate_ids,
    raw_trace_row,
    truncate_recent_history,
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
DATA_DIR = PROJECT_ROOT / "data" / "processed" / DATASET

D5E_MANIFEST_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ml1m_ranking_diagnostic_manifest.csv"

D5K_MANIFEST_PATH = RESULTS_TABLES_DIR / "plan_d_llm_ml1m_history_length_manifest.csv"
D5K_INSTANCE_SUMMARY_PATH = RESULTS_TABLES_DIR / "plan_d_llm_ml1m_history_length_instance_summary.csv"
D5K_CONDITION_SUMMARY_PATH = RESULTS_TABLES_DIR / "plan_d_llm_ml1m_history_length_condition_summary.csv"
D5K_PAIRWISE_DELTA_PATH = RESULTS_TABLES_DIR / "plan_d_llm_ml1m_history_length_pairwise_delta.csv"
D5K_STABILITY_PROFILE_PATH = RESULTS_TABLES_DIR / "plan_d_llm_ml1m_history_length_stability_profile.csv"
D5K_METRICS_PATH = RESULTS_METRICS_DIR / "plan_d_llm_ml1m_history_length_sensitivity_metrics.csv"
D5K_REPORT_PATH = REPORT_DIR / "plan_d_llm_ml1m_history_length_sensitivity.md"
D5K_RAW_PATH = RAW_OUTPUTS_DIR / "plan_d_llm_ml1m_history_length_sensitivity_raw.jsonl"

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


def run_diagnostic(models: tuple[str, ...], history_settings: tuple[str, ...]) -> dict[str, object]:
    manifest = _load_and_validate_manifest(D5E_MANIFEST_PATH)
    histories_by_user = _load_histories(DATA_DIR)
    inventory = check_local_ollama_models()
    if QWEN_MODEL in models and QWEN_MODEL not in set(inventory.model_names):
        raise RuntimeError(f"Required local Qwen model is unavailable: {QWEN_MODEL}")
    if MISTRAL_MODEL in models and select_local_mistral_model(inventory.model_names, MISTRAL_MODEL) != MISTRAL_MODEL:
        raise RuntimeError(f"Required local Mistral model is unavailable: {MISTRAL_MODEL}")

    d5k_manifest = build_history_length_manifest(manifest, histories_by_user)
    d5k_manifest = d5k_manifest.loc[
        d5k_manifest["model"].isin(models) & d5k_manifest["history_setting"].isin(history_settings)
    ].reset_index(drop=True)

    client = OllamaClient(timeout_seconds=TIMEOUT_SECONDS)
    instances = []
    raw_rows: list[dict[str, object]] = []
    runtimes: dict[str, float] = {}
    request_count = 0

    for model in models:
        for setting in history_settings:
            condition = condition_for(model, setting)
            started = time.perf_counter()
            for index, (_, row) in enumerate(manifest.iterrows(), start=1):
                user_id = str(row["user_id"])
                history = histories_by_user.get(user_id, [])
                if not history:
                    raise RuntimeError(f"D5K requires non-empty history for user {user_id}")
                history_window = truncate_recent_history(history, HISTORY_LIMITS[setting])
                candidate_ids = parse_candidate_ids(str(row["candidate_ids"]))
                prompt = build_history_length_prompt(
                    user_id=user_id,
                    history_items=history_window,
                    full_history_length=len(history),
                    history_setting=setting,
                    candidate_ids=candidate_ids,
                    top_k=TOP_K,
                )
                LOGGER.info(
                    "Running D5K %s %s local request %s/20 for %s",
                    model,
                    setting,
                    index,
                    row["instance_id"],
                )
                request_count += 1
                result = client.generate_json(model=model, prompt=prompt, temperature=TEMPERATURE)
                parsed = parse_llm_ranking_response(
                    result.response_text if result.response_received else "",
                    candidate_ids=set(candidate_ids),
                    expected_top_k=TOP_K,
                )
                instance = evaluate_history_output(
                    parsed.ranked_item_ids,
                    condition=condition,
                    history_setting=setting,
                    user_id=user_id,
                    source_instance_id=str(row["instance_id"]),
                    diagnostic_id=f"{PHASE}_{condition}_{user_id}",
                    model=model,
                    candidate_ids=candidate_ids,
                    positive_item_id=str(row["positive_item_id"]),
                    history_length_used=len(history_window),
                    full_history_length=len(history),
                    response_received=result.response_received,
                    parse_success=parsed.parse_success,
                    schema_valid=parsed.schema_valid,
                    latency_seconds=result.latency_seconds,
                    parse_status=parsed.status,
                    notes=result.error or parsed.error_message or _notes_for_output(condition, parsed.hallucinated_count),
                )
                instances.append(instance)
                raw_rows.append(
                    raw_trace_row(
                        instance=instance,
                        raw_response_text=result.response_text,
                        parsed_result=asdict(parsed),
                    )
                )
            runtimes[condition] = time.perf_counter() - started

    instance_summary = pd.DataFrame([instance_row(instance) for instance in instances], columns=INSTANCE_COLUMNS)
    condition_summary = build_condition_summary(instance_summary, runtimes=runtimes)
    pairwise_delta = build_pairwise_delta(condition_summary, instance_summary)
    stability_profile = build_stability_profile(instance_summary)
    metrics = build_metric_rows(condition_summary)
    return {
        "manifest": d5k_manifest,
        "instance_summary": instance_summary,
        "condition_summary": condition_summary,
        "pairwise_delta": pairwise_delta,
        "stability_profile": stability_profile,
        "metrics": metrics,
        "raw_rows": raw_rows,
        "request_count": request_count,
        "runtimes": runtimes,
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
    stability_profile = result["stability_profile"]
    metrics = result["metrics"]
    raw_rows = result["raw_rows"]
    request_count = int(result["request_count"])
    runtimes = result["runtimes"]

    assert isinstance(manifest, pd.DataFrame)
    assert isinstance(instance_summary, pd.DataFrame)
    assert isinstance(condition_summary, pd.DataFrame)
    assert isinstance(pairwise_delta, pd.DataFrame)
    assert isinstance(stability_profile, pd.DataFrame)
    assert isinstance(metrics, pd.DataFrame)
    assert isinstance(raw_rows, list)
    assert isinstance(runtimes, dict)

    manifest.to_csv(D5K_MANIFEST_PATH, index=False, columns=MANIFEST_COLUMNS)
    instance_summary.to_csv(D5K_INSTANCE_SUMMARY_PATH, index=False, columns=INSTANCE_COLUMNS)
    condition_summary.to_csv(D5K_CONDITION_SUMMARY_PATH, index=False, columns=CONDITION_SUMMARY_COLUMNS)
    pairwise_delta.to_csv(D5K_PAIRWISE_DELTA_PATH, index=False, columns=PAIRWISE_COLUMNS)
    stability_profile.to_csv(D5K_STABILITY_PROFILE_PATH, index=False, columns=STABILITY_COLUMNS)
    metrics.to_csv(D5K_METRICS_PATH, index=False, columns=METRIC_COLUMNS)
    with D5K_RAW_PATH.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    D5K_REPORT_PATH.write_text(_report(condition_summary, pairwise_delta, stability_profile, request_count), encoding="utf-8")
    _update_notes(condition_summary, request_count, runtimes)
    _update_task_roadmap()
    _update_next_actions()
    _update_feasibility_ledger(condition_summary)
    _update_reproducibility_matrix(condition_summary)


def _report(
    condition_summary: pd.DataFrame,
    pairwise_delta: pd.DataFrame,
    stability_profile: pd.DataFrame,
    request_count: int,
) -> str:
    rows = "\n".join(
        f"| `{row['condition']}` | `{row['model']}` | `{row['history_setting']}` | `{row['hr_at_5']}` | `{row['ndcg_at_5']}` | `{row['mean_candidate_validity_rate']}` | `{row['hallucination_rate']}` | `{row['selected_item_jaccard_vs_medium_history']}` | `{row['output_stability_rate_across_history_lengths']}` |"
        for _, row in condition_summary.iterrows()
    )
    delta_rows = "\n".join(
        f"| `{row['model']}` | `{row['condition_a']}` | `{row['condition_b']}` | `{row['HR@5_delta']}` | `{row['NDCG@5_delta']}` | `{row['candidate_validity_delta']}` | `{row['hallucination_rate_delta']}` | `{row['selected_item_jaccard']}` | `{row['interpretation']}` |"
        for _, row in pairwise_delta.iterrows()
    )
    stability_counts = stability_profile["hit_pattern"].value_counts().to_dict()
    return f"""# Plan D Local LLM ML-1M History Length Sensitivity

## Purpose

D5K evaluates whether local LLM recommendation outputs are sensitive to the amount of user history shown in the prompt. It uses the same ML-1M 20-user diagnostic subset as D5E/D5G/D5I/D5J and varies only the user-history length.

## Scope And Classification

- D5K is a diagnostic_subset local LLM history length sensitivity diagnostic.
- D5K is not exact reproduction.
- D5K is not full LLM reproduction.
- D5K uses local Ollama only.
- Qwen2 and Mistral are local open models.
- Maximum 120 new local LLM requests were allowed.
- Exactly {request_count} new local LLM requests were run.
- D5K does not reproduce the paper's full LLM matrix or closed-source setup.

## Execution Boundaries

- No OpenAI API call was made.
- No Anthropic API call was made.
- No external API call was made.
- No download was performed during D5K.
- No model pull was performed during D5K.
- No recommender training was performed.
- Plan C was not modified.
- Raw JSONL outputs are stored under `results/raw_outputs/` and remain Git-ignored.

## Controlled Design

D5K varies only user-history length while keeping candidate sets, positive items, model, top_k, temperature, backend, and strict candidate-ID-only JSON constraints fixed. The three history settings are `short_history` with the most recent 5 items, `medium_history` with the most recent 10 items, and `long_history` with the most recent 20 items or fewer if unavailable.

## Condition Summary

| Condition | Model | History | HR@5 | NDCG@5 | Candidate Validity | Hallucination | Jaccard vs Medium | Output Stability |
|---|---|---|---:|---:|---:|---:|---:|---:|
{rows}

## Pairwise History Deltas

| Model | Condition A | Condition B | HR@5 Delta | NDCG@5 Delta | Validity Delta | Hallucination Delta | Selected Jaccard | Interpretation |
|---|---|---|---:|---:|---:|---:|---:|---|
{delta_rows}

Stability hit-pattern counts: `{json.dumps(stability_counts, sort_keys=True)}`.

## Interpretation

D5K evaluates utility and reliability separately. HR@5 and NDCG@5 describe utility. Candidate validity, hallucination, duplicate rate, response/parse/schema rates, and exact-top-k rates describe output reliability. Jaccard and hit-pattern metrics describe output stability as history length changes.

The diagnostic distinguishes candidate validity, hallucination, utility, and output stability because a model can preserve syntactic JSON compliance while changing recommendation utility or drifting toward invalid candidate IDs. This supports the broader Beyond Utility argument: utility metrics alone are insufficient to characterize local LLM recommender behavior.

## Limitations

- Only 20 ML-1M diagnostic users are evaluated.
- Only Qwen2 and Mistral local open models are evaluated.
- Only strict candidate-ID-only prompts are used.
- No closed-source API model is evaluated.
- This is diagnostic_subset evidence, not exact paper reproduction.

## Recommended Next Steps

1. Integrate D5E-D5K local LLM diagnostics into the final report.
2. Prepare exam presentation tables from generated CSV files.
3. Optionally run an OpenAI API diagnostic only if budget and explicit approval are provided.
"""


def _update_notes(condition_summary: pd.DataFrame, request_count: int, runtimes: dict[str, float]) -> None:
    qwen_medium = condition_summary.loc[condition_summary["condition"] == "qwen2_medium_history"].iloc[0]
    mistral_medium = condition_summary.loc[condition_summary["condition"] == "mistral_medium_history"].iloc[0]
    section = f"""
## 2026-07-01 - Plan D D5K: Local LLM ML-1M History Length Sensitivity

### Changes

- Ran D5K history-length diagnostics for Qwen2 and Mistral over the same 20 ML-1M diagnostic instances.
- Varied only the user-history window: most recent 5, 10, and 20 interactions.
- Kept candidate sets, positive items, top_k, temperature, backend, and strict candidate-ID-only JSON constraints fixed.
- Wrote D5K manifest, instance summary, condition summary, pairwise deltas, stability profile, metric-style CSV, ignored raw JSONL, and report.

### Outputs Generated

- `results/metrics/plan_d_llm_ml1m_history_length_sensitivity_metrics.csv`
- `results/tables/plan_d_llm_ml1m_history_length_instance_summary.csv`
- `results/tables/plan_d_llm_ml1m_history_length_condition_summary.csv`
- `results/tables/plan_d_llm_ml1m_history_length_pairwise_delta.csv`
- `results/tables/plan_d_llm_ml1m_history_length_stability_profile.csv`
- `results/tables/plan_d_llm_ml1m_history_length_manifest.csv`
- `results/raw_outputs/plan_d_llm_ml1m_history_length_sensitivity_raw.jsonl` (Git-ignored)
- `report/plan_d_llm_ml1m_history_length_sensitivity.md`

### Result

- New local LLM requests: `{request_count}`
- Qwen2 medium-history HR@5/NDCG@5: `{qwen_medium['hr_at_5']}` / `{qwen_medium['ndcg_at_5']}`
- Mistral medium-history HR@5/NDCG@5: `{mistral_medium['hr_at_5']}` / `{mistral_medium['ndcg_at_5']}`
- Qwen2 total runtime seconds: `{sum(float(value) for key, value in runtimes.items() if key.startswith('qwen2_')):.6f}`
- Mistral total runtime seconds: `{sum(float(value) for key, value in runtimes.items() if key.startswith('mistral_')):.6f}`

### Notes And Limitations

- D5K is a diagnostic_subset local LLM history-length sensitivity diagnostic, not exact reproduction and not full LLM reproduction.
- No OpenAI API call, Anthropic API call, external API call, download, model pull, recommender training, LastFM experiment, or Plan C modification occurred.
"""
    _replace_or_append_section(NOTES_PATH, "## 2026-07-01 - Plan D D5K: Local LLM ML-1M History Length Sensitivity", section)


def _update_task_roadmap() -> None:
    frame = _read_or_empty(TASK_ROADMAP_PATH, TASK_ROADMAP_COLUMNS)
    mask = frame["phase"] == "D5K"
    row = {
        "phase": "D5K",
        "task": "history_length_sensitivity_diagnostic",
        "dataset": DATASET,
        "model": "qwen2:7b-instruct+mistral:7b-instruct",
        "backend": BACKEND,
        "users": "20",
        "candidate_pool_size": "20",
        "prompt_strategy": PROMPT_STRATEGY,
        "output_format": "strict_json",
        "metrics": "HR@5,NDCG@5,validity,hallucination,history_stability",
        "prerequisites": "D5J completed and local models available",
        "status": "completed_diagnostic_subset",
        "notes": "D5K completed 120 local LLM requests varying only short/medium/long history windows.",
    }
    if mask.any():
        for column, value in row.items():
            frame.loc[mask, column] = value
    else:
        frame.loc[len(frame)] = row
    frame.to_csv(TASK_ROADMAP_PATH, index=False)


def _update_next_actions() -> None:
    rows = [
        _next_action("1", "integrate_llm_results_or_blocker_into_final_report", "Integrate LLM diagnostics into final report", "final_report; local_llm_backend", "D5E-D5K now provide bounded local LLM diagnostic evidence.", "Final LLM section grounded in generated CSVs.", "D5K completed.", "low", "next", "Do not claim exact paper reproduction."),
        _next_action("2", "prepare_exam_presentation_tables", "Prepare exam presentation tables", "final_report", "The oral exam needs compact generated tables.", "Presentation-ready generated tables.", "D5E-D5K tables available.", "low", "after integration draft", "Use generated CSV values."),
        _next_action("3", "optional_openai_api_diagnostic_if_budget_approved", "Optional OpenAI API diagnostic if budget approved", "api_availability", "Closed-source comparison remains conditional.", "Small approved API diagnostic or documented no-API decision.", "Explicit user approval, API key, and budget.", "high", "optional", "No API calls by default."),
        _next_action("4", "final_report", "Finalize report", "final_report", "The project needs final claims grounded in generated artifacts.", "Final report with limitations and generated metrics.", "LLM diagnostics integrated.", "low", "final", "Do not invent metric values."),
    ]
    pd.DataFrame(rows, columns=NEXT_ACTION_COLUMNS).to_csv(NEXT_ACTIONS_PATH, index=False)


def _update_feasibility_ledger(condition_summary: pd.DataFrame) -> None:
    frame = _read_or_empty(FEASIBILITY_LEDGER_PATH, LEDGER_COLUMNS)
    qwen_medium = condition_summary.loc[condition_summary["condition"] == "qwen2_medium_history"].iloc[0]
    mistral_medium = condition_summary.loc[condition_summary["condition"] == "mistral_medium_history"].iloc[0]
    row = {
        "item_id": "history_length_sensitivity",
        "area": "llm_analysis",
        "target": "Qwen2 and Mistral ML-1M history length sensitivity diagnostic",
        "feasibility_status": "already_done",
        "required_resources": "D5E manifest, local Ollama, qwen2:7b-instruct, mistral:7b-instruct",
        "estimated_effort": "done",
        "risk_level": "medium",
        "external_dependency": "yes",
        "recommended_decision": "Use D5K findings in final LLM diagnostic integration.",
        "notes": f"D5K completed; Qwen2 medium HR@5={qwen_medium['hr_at_5']}, Mistral medium HR@5={mistral_medium['hr_at_5']}.",
    }
    _upsert_ledger(frame, row)
    _upsert_ledger(
        frame,
        {
            "item_id": "local_llm_history_length_sensitivity_diagnostic",
            "area": "llm_analysis",
            "target": "D5K local open-model history-length robustness diagnostic",
            "feasibility_status": "already_done",
            "required_resources": "Local Ollama and installed Qwen2/Mistral models",
            "estimated_effort": "done",
            "risk_level": "medium",
            "external_dependency": "yes",
            "recommended_decision": "Integrate with D5E-D5J in final report.",
            "notes": "Completed as diagnostic_subset; not exact LLM reproduction.",
        },
    )
    frame.to_csv(FEASIBILITY_LEDGER_PATH, index=False)


def _update_reproducibility_matrix(condition_summary: pd.DataFrame) -> None:
    if not REPRO_MATRIX_PATH.exists():
        return
    qwen_medium = condition_summary.loc[condition_summary["condition"] == "qwen2_medium_history"].iloc[0]
    mistral_medium = condition_summary.loc[condition_summary["condition"] == "mistral_medium_history"].iloc[0]
    text = REPRO_MATRIX_PATH.read_text(encoding="utf-8")
    if "D5K completed local LLM history-length sensitivity" not in text:
        text = text.replace(
            "D5J completed local LLM reranking as diagnostic_subset work.",
            "D5J completed local LLM reranking as diagnostic_subset work. D5K completed local LLM history-length sensitivity as diagnostic_subset work.",
        )
    text = text.replace("D5C-D5J roadmap", "D5C-D5K roadmap")
    text = _replace_table_row(
        text,
        "History length sensitivity",
        f"| History length sensitivity | `DIAGNOSTIC_ONLY` | `diagnostic` | D5K completed Qwen2 and Mistral local history-length sensitivity diagnostics; Qwen2 medium HR@5 `{qwen_medium['hr_at_5']}`, Mistral medium HR@5 `{mistral_medium['hr_at_5']}`. |",
    )
    text = _replace_table_row(
        text,
        "History length sensitivity",
        "| History length sensitivity | `completed_diagnostic_subset` | D5K varied short/medium/long history windows while holding candidate sets fixed. |",
        occurrence=2,
    )
    text = _replace_table_row(
        text,
        "LLM reranking diagnostic",
        "| LLM reranking diagnostic | `completed_diagnostic_subset` | D5J completed Qwen2 and Mistral local reranking diagnostics against candidate_order_reference. |",
    )
    text = _replace_table_row(
        text,
        "Local LLM diagnostic reranking",
        "| Local LLM diagnostic reranking | `completed_diagnostic_subset` | D5J completed Qwen2 and Mistral local reranking diagnostics against candidate_order_reference. |",
    )
    text = _replace_table_row(
        text,
        "Local LLM diagnostic ranking",
        "| Local LLM diagnostic ranking | `completed_diagnostic_subset` | D5E-D5K produced bounded local LLM diagnostic outputs; no exact or full-scope LLM reproduction is claimed. |",
    )
    text = _replace_section_text(
        text,
        "## Recommended next steps",
        """## Recommended next steps

1. Integrate D5E-D5K LLM diagnostic results into the final report.
2. Prepare exam presentation tables.
3. Optionally run an OpenAI API diagnostic if budget is explicitly approved.
4. Finalize the report.
""",
    )
    REPRO_MATRIX_PATH.write_text(text, encoding="utf-8")


def _load_and_validate_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if len(frame) != 20:
        raise ValueError(f"D5K requires exactly 20 D5E manifest rows, found {len(frame)}")
    for _, row in frame.iterrows():
        candidate_ids = parse_candidate_ids(str(row["candidate_ids"]))
        if len(candidate_ids) != CANDIDATE_POOL_SIZE:
            raise ValueError(f"{row['instance_id']} has {len(candidate_ids)} candidates")
        if str(row["positive_item_id"]) not in set(candidate_ids):
            raise ValueError(f"{row['instance_id']} positive item missing from candidates")
        if str(row["positive_in_candidates"]).lower() != "true":
            raise ValueError(f"{row['instance_id']} positive_in_candidates must be true")
    return frame


def _load_histories(data_dir: Path) -> dict[str, list[HistoryItem]]:
    interactions = pd.read_csv(data_dir / "interactions.csv", dtype=str).fillna("")
    catalog = pd.read_csv(data_dir / "catalog.csv", dtype=str).fillna("")
    catalog_by_item = {
        str(row["item_id"]): str(row.get("title", "")).strip() or str(row["item_id"])
        for _, row in catalog.iterrows()
    }
    train = interactions.loc[interactions["split"].astype(str).str.lower() == "train"].copy()
    train["__timestamp_sort"] = pd.to_numeric(train.get("timestamp", -1), errors="coerce").fillna(-1)
    train["__item_sort"] = train["item_id"].map(_id_sort_key)
    train = train.sort_values(["user_id", "__timestamp_sort", "__item_sort"], kind="mergesort")

    histories: dict[str, list[HistoryItem]] = {}
    for user_id, group in train.groupby(train["user_id"].astype(str), sort=False):
        histories[str(user_id)] = [
            HistoryItem(item_id=str(row["item_id"]), title=catalog_by_item.get(str(row["item_id"]), str(row["item_id"])))
            for _, row in group.iterrows()
        ]
    return histories


def _notes_for_output(condition: str, hallucinated_count: int) -> str:
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


def _id_sort_key(value: object) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def _parse_models(value: str) -> tuple[str, ...]:
    mapping = {"qwen2": QWEN_MODEL, QWEN_MODEL: QWEN_MODEL, "mistral": MISTRAL_MODEL, MISTRAL_MODEL: MISTRAL_MODEL}
    models = []
    for token in value.split(","):
        normalized = token.strip()
        if not normalized:
            continue
        if normalized not in mapping:
            raise ValueError(f"unsupported model selector: {normalized}")
        models.append(mapping[normalized])
    return tuple(dict.fromkeys(models))


def _parse_history_settings(value: str) -> tuple[str, ...]:
    settings = []
    for token in value.split(","):
        normalized = token.strip()
        if not normalized:
            continue
        if normalized not in set(HISTORY_SETTINGS):
            raise ValueError(f"unsupported history setting: {normalized}")
        settings.append(normalized)
    return tuple(dict.fromkeys(settings))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D5K local LLM ML-1M history length sensitivity diagnostic.")
    parser.add_argument("--models", default="qwen2,mistral", help="Comma-separated model selectors: qwen2,mistral")
    parser.add_argument(
        "--history-settings",
        default="short_history,medium_history,long_history",
        help="Comma-separated history settings.",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    result = run_diagnostic(models=_parse_models(args.models), history_settings=_parse_history_settings(args.history_settings))
    write_outputs(result)
    LOGGER.info("Wrote %s", D5K_CONDITION_SUMMARY_PATH)
    LOGGER.info("Wrote %s", D5K_REPORT_PATH)


if __name__ == "__main__":
    main()
