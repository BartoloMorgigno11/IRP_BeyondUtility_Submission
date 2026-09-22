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

from src.analysis.plan_d_qwen2_strict_prompt_analysis import (
    BACKEND,
    CANDIDATE_POOL_SIZE,
    CONDITION_SUMMARY_COLUMNS,
    DATASET,
    INSTANCE_COLUMNS,
    MANIFEST_COLUMNS,
    METRIC_COLUMNS,
    MODEL,
    ORIGINAL_CONDITION,
    ORIGINAL_POST_FILTERED_CONDITION,
    PAIRWISE_COLUMNS,
    PHASE,
    PLAN,
    POST_FILTER_PROFILE_COLUMNS,
    PROMPT_STRATEGY,
    SCOPE,
    STRICT_CONDITION,
    STRICT_POST_FILTERED_CONDITION,
    TEMPERATURE,
    TOP_K,
    build_condition_summary,
    build_metric_rows,
    build_pairwise_delta,
    build_post_filter_profile,
    build_strict_candidate_id_prompt,
    build_strict_prompt_manifest,
    condition_instance_from_selection,
    instance_row,
    post_filtered_instance,
    raw_strict_trace_row,
)
from src.llm.plan_d_llm_ranking_parser import parse_llm_ranking_response
from src.llm.plan_d_ollama_client import OllamaClient


LOGGER = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_METRICS_DIR = RESULTS_DIR / "metrics"
RAW_OUTPUTS_DIR = RESULTS_DIR / "raw_outputs"
REPORT_DIR = PROJECT_ROOT / "report"

D5E_MANIFEST_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ml1m_ranking_diagnostic_manifest.csv"
D5E_RAW_PATH = RAW_OUTPUTS_DIR / "plan_d_qwen2_ml1m_ranking_diagnostic_raw.jsonl"

D5G_MANIFEST_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ml1m_strict_prompt_manifest.csv"
D5G_INSTANCE_SUMMARY_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ml1m_strict_prompt_instance_summary.csv"
D5G_CONDITION_SUMMARY_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ml1m_strict_prompt_condition_summary.csv"
D5G_PAIRWISE_DELTA_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ml1m_strict_prompt_pairwise_delta.csv"
D5G_POST_FILTER_PROFILE_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ml1m_strict_prompt_post_filter_profile.csv"
D5G_METRICS_PATH = RESULTS_METRICS_DIR / "plan_d_qwen2_ml1m_strict_prompt_metrics.csv"
D5G_RAW_PATH = RAW_OUTPUTS_DIR / "plan_d_qwen2_ml1m_strict_prompt_diagnostic_raw.jsonl"
D5G_REPORT_PATH = REPORT_DIR / "plan_d_qwen2_ml1m_strict_prompt_diagnostic.md"

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
    d5e_manifest = _load_and_validate_d5e_manifest(D5E_MANIFEST_PATH)
    d5e_raw_rows = _load_d5e_raw_rows(D5E_RAW_PATH)
    raw_by_instance = {str(row["instance_id"]): row for row in d5e_raw_rows}
    d5g_manifest = build_strict_prompt_manifest(d5e_manifest)

    client = OllamaClient(timeout_seconds=TIMEOUT_SECONDS)
    instance_rows: list[dict[str, str]] = []
    raw_rows: list[dict[str, object]] = []
    local_qwen_requests = 0
    started = time.perf_counter()

    for index, (_, manifest_row) in enumerate(d5e_manifest.iterrows(), start=1):
        source_instance_id = str(manifest_row["instance_id"])
        raw_row = raw_by_instance[source_instance_id]
        candidate_ids = json.loads(str(manifest_row["candidate_ids"]))
        positive_item_id = str(manifest_row["positive_item_id"])
        original = _original_instance_from_d5e_raw(
            raw_row=raw_row,
            manifest_row=manifest_row,
            candidate_ids=candidate_ids,
            positive_item_id=positive_item_id,
        )
        original_post_filtered = post_filtered_instance(original, condition=ORIGINAL_POST_FILTERED_CONDITION)
        instance_rows.append(instance_row(original))
        instance_rows.append(instance_row(original_post_filtered))

        prompt = build_strict_candidate_id_prompt(
            user_id=str(manifest_row["user_id"]),
            history_preview=str(manifest_row.get("history_preview", "")),
            candidate_ids=candidate_ids,
            top_k=TOP_K,
        )
        diagnostic_id = f"{PHASE}_{STRICT_CONDITION}_{str(manifest_row['user_id'])}"
        LOGGER.info("Running D5G local Qwen2 strict prompt request %s/20 for %s", index, source_instance_id)
        local_qwen_requests += 1
        result = client.generate_json(model=MODEL, prompt=prompt, temperature=TEMPERATURE)
        parsed = parse_llm_ranking_response(
            result.response_text if result.response_received else "",
            candidate_ids=set(candidate_ids),
            expected_top_k=TOP_K,
        )
        strict = condition_instance_from_selection(
            condition=STRICT_CONDITION,
            user_id=str(manifest_row["user_id"]),
            source_instance_id=source_instance_id,
            diagnostic_id=diagnostic_id,
            candidate_ids=candidate_ids,
            positive_item_id=positive_item_id,
            response_received=result.response_received,
            parse_success=parsed.parse_success,
            schema_valid=parsed.schema_valid,
            ranked_item_ids=parsed.ranked_item_ids,
            latency_seconds=result.latency_seconds,
            parse_status=parsed.status,
            items_removed_by_filter=0,
            filtered_from_condition="",
            notes=result.error or parsed.error_message or _strict_notes(parsed.hallucinated_count),
        )
        strict_post_filtered = post_filtered_instance(strict, condition=STRICT_POST_FILTERED_CONDITION)
        instance_rows.append(instance_row(strict))
        instance_rows.append(instance_row(strict_post_filtered))
        raw_rows.append(
            raw_strict_trace_row(
                instance=strict,
                raw_response_text=result.response_text,
                parsed_result=asdict(parsed),
            )
        )

    total_runtime_seconds = time.perf_counter() - started
    instance_summary = pd.DataFrame(instance_rows, columns=INSTANCE_COLUMNS)
    condition_summary = build_condition_summary(instance_summary)
    pairwise_delta = build_pairwise_delta(condition_summary)
    post_filter_profile = build_post_filter_profile(instance_summary, condition_summary)
    metrics = build_metric_rows(condition_summary)

    return {
        "manifest": d5g_manifest,
        "instance_summary": instance_summary,
        "condition_summary": condition_summary,
        "pairwise_delta": pairwise_delta,
        "post_filter_profile": post_filter_profile,
        "metrics": metrics,
        "raw_rows": raw_rows,
        "total_runtime_seconds": total_runtime_seconds,
        "local_qwen_requests": local_qwen_requests,
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
    post_filter_profile = result["post_filter_profile"]
    metrics = result["metrics"]
    raw_rows = result["raw_rows"]
    total_runtime_seconds = float(result["total_runtime_seconds"])
    local_qwen_requests = int(result["local_qwen_requests"])

    assert isinstance(manifest, pd.DataFrame)
    assert isinstance(instance_summary, pd.DataFrame)
    assert isinstance(condition_summary, pd.DataFrame)
    assert isinstance(pairwise_delta, pd.DataFrame)
    assert isinstance(post_filter_profile, pd.DataFrame)
    assert isinstance(metrics, pd.DataFrame)
    assert isinstance(raw_rows, list)

    manifest.to_csv(D5G_MANIFEST_PATH, index=False, columns=MANIFEST_COLUMNS)
    instance_summary.to_csv(D5G_INSTANCE_SUMMARY_PATH, index=False, columns=INSTANCE_COLUMNS)
    condition_summary.to_csv(D5G_CONDITION_SUMMARY_PATH, index=False, columns=CONDITION_SUMMARY_COLUMNS)
    pairwise_delta.to_csv(D5G_PAIRWISE_DELTA_PATH, index=False, columns=PAIRWISE_COLUMNS)
    post_filter_profile.to_csv(D5G_POST_FILTER_PROFILE_PATH, index=False, columns=POST_FILTER_PROFILE_COLUMNS)
    metrics.to_csv(D5G_METRICS_PATH, index=False, columns=METRIC_COLUMNS)
    with D5G_RAW_PATH.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    D5G_REPORT_PATH.write_text(
        _report(condition_summary, post_filter_profile, pairwise_delta, total_runtime_seconds, local_qwen_requests),
        encoding="utf-8",
    )
    _update_notes(condition_summary, post_filter_profile, total_runtime_seconds, local_qwen_requests)
    _update_task_roadmap()
    _update_next_actions()
    _update_feasibility_ledger(condition_summary)
    _update_reproducibility_matrix(condition_summary)


def _original_instance_from_d5e_raw(
    *,
    raw_row: dict[str, object],
    manifest_row: pd.Series,
    candidate_ids: list[str],
    positive_item_id: str,
):
    parsed = raw_row.get("parsed_result", {})
    if not isinstance(parsed, dict):
        parsed = {}
    ranked_item_ids = [str(item_id) for item_id in parsed.get("ranked_item_ids", [])]
    return condition_instance_from_selection(
        condition=ORIGINAL_CONDITION,
        user_id=str(manifest_row["user_id"]),
        source_instance_id=str(manifest_row["instance_id"]),
        diagnostic_id=f"{PHASE}_{ORIGINAL_CONDITION}_{str(manifest_row['user_id'])}",
        candidate_ids=candidate_ids,
        positive_item_id=positive_item_id,
        response_received=bool(raw_row.get("raw_response_text", "")),
        parse_success=bool(parsed.get("parse_success", False)),
        schema_valid=bool(parsed.get("schema_valid", False)),
        ranked_item_ids=ranked_item_ids,
        latency_seconds=float(raw_row.get("latency_seconds", 0.0) or 0.0),
        parse_status=str(parsed.get("status", "")),
        items_removed_by_filter=0,
        filtered_from_condition="",
        notes="Original D5E parsed output reference; no new inference.",
    )


def _report(
    condition_summary: pd.DataFrame,
    post_filter_profile: pd.DataFrame,
    pairwise_delta: pd.DataFrame,
    total_runtime_seconds: float,
    local_qwen_requests: int,
) -> str:
    condition_rows = "\n".join(
        f"| `{row['condition']}` | `{row['hr_at_5']}` | `{row['ndcg_at_5']}` | `{row['mean_candidate_validity_rate']}` | `{row['micro_candidate_validity_rate']}` | `{row['hallucination_rate']}` | `{row['exact_top_k_rate']}` | `{row['exact_5_valid_ids_rate']}` |"
        for _, row in condition_summary.iterrows()
    )
    post_rows = "\n".join(
        f"| `{row['condition']}` | `{row['mean_items_removed_by_filter']}` | `{row['total_items_removed_by_filter']}` | `{row['filtered_list_mean_length']}` | `{row['filtered_HR@5']}` | `{row['filtered_NDCG@5']}` | `{row['filtered_hallucination_rate']}` |"
        for _, row in post_filter_profile.iterrows()
    )
    return f"""# Plan D Qwen2 ML-1M Strict Prompt Diagnostic

## Purpose

D5G tests whether the hallucination and candidate-invalidity observed in D5E can be reduced by a stricter candidate-ID-only prompt and by deterministic post-filtering.

## Scope And Classification

- D5G is a diagnostic_subset prompt mitigation diagnostic.
- Classification: `diagnostic_subset` robustness/reliability diagnostic.
- This is not exact reproduction.
- This is not full LLM reproduction.
- This is a local open model diagnostic only.
- Backend/model: local Ollama with `qwen2:7b-instruct`.
- Strict prompt local Qwen2 requests: `{local_qwen_requests}`.
- The strict prompt condition uses exactly the same 20 ML-1M diagnostic instances and same candidate sets as D5E.
- Temperature: `0`.
- Top-k: `5`.
- Candidate pool size: `20`.
- Total runtime seconds: `{total_runtime_seconds:.6f}`.

## Execution Boundaries

- D5G uses local Qwen2 through Ollama only.
- No OpenAI API call was made.
- No Anthropic API call was made.
- No external API call was made.
- No download was performed.
- No model pull was performed.
- No recommender training was performed.
- Plan C was not modified.
- Raw JSONL outputs are stored under `results/raw_outputs/` and remain Git-ignored.

## Conditions

- `original_parsed_output`: existing D5E parsed output reference; no new inference.
- `original_post_filtered`: deterministic post-filtering of D5E outputs; no new inference.
- `strict_candidate_id_prompt`: 20 new local Qwen2/Ollama requests using a candidate-ID-only prompt.
- `strict_post_filtered`: deterministic post-filtering of strict prompt outputs; no extra inference.

The strict prompt tries to mitigate hallucination by making candidate IDs the only allowed output domain. Post-filtering removes IDs outside the candidate set, preserves valid order, and does not backfill.

## Metric Summary

| Condition | HR@5 | NDCG@5 | Mean Candidate Validity | Micro Candidate Validity | Hallucination Rate | Exact Top-k Rate | Exact 5 Valid IDs Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
{condition_rows}

## Post-Filter Summary

| Condition | Mean Removed | Total Removed | Filtered Mean Length | Filtered HR@5 | Filtered NDCG@5 | Filtered Hallucination Rate |
|---|---:|---:|---:|---:|---:|---:|
{post_rows}

## Interpretation

{_prompt_mitigation_interpretation(condition_summary)}

{_post_filter_interpretation(post_filter_profile)}

Syntactic validity means the model returned parseable JSON with the expected fields. Candidate validity means selected IDs belong to the provided candidate set. Hallucination means selected IDs fall outside that candidate set. Utility is measured separately with HR@5 and NDCG@5 against the known positive item.

## Beyond Utility Interpretation

D5G supports the broader Beyond Utility argument because a model can be parse-valid while still failing candidate adherence. Prompt-level mitigation and deterministic post-filtering affect reliability and utility differently, so HR/NDCG alone are insufficient to describe LLM recommender behavior.

## Limitations

- Only 20 users.
- Only ML-1M.
- Only Qwen2.
- Only the original D5E candidate order.
- Only one strict candidate-ID prompt variant.
- Not exact paper reproduction.
- Not full LLM reproduction.

## Recommended Next Steps

1. Compare with Mistral after the user manually pulls a quantized Mistral v0.2 model.
2. Run LLM reranking diagnostics.
3. Run history length sensitivity diagnostics.
4. Integrate D5E-D5G reliability findings into the final report.
"""


def _update_notes(
    condition_summary: pd.DataFrame,
    post_filter_profile: pd.DataFrame,
    total_runtime_seconds: float,
    local_qwen_requests: int,
) -> None:
    strict = _summary_row(condition_summary, STRICT_CONDITION)
    strict_post = _summary_row(condition_summary, STRICT_POST_FILTERED_CONDITION)
    section = f"""
## 2026-07-01 - Plan D D5G: Qwen2 Strict Prompt And Post-Filter Diagnostic

### Changes

- Ran the D5G local Qwen2/Ollama strict candidate-ID-only prompt diagnostic over the same 20 ML-1M D5E instances.
- Added deterministic post-filter analyses for the original D5E outputs and the D5G strict-prompt outputs.
- Wrote manifest, instance summary, condition summary, pairwise deltas, post-filter profile, metric-style CSV, raw ignored JSONL, and report artifacts.

### Commands Run

```powershell
python src/experiments/run_plan_d_qwen2_ml1m_strict_prompt_diagnostic.py
```

### Outputs Generated

- `results/metrics/plan_d_qwen2_ml1m_strict_prompt_metrics.csv`
- `results/tables/plan_d_qwen2_ml1m_strict_prompt_instance_summary.csv`
- `results/tables/plan_d_qwen2_ml1m_strict_prompt_condition_summary.csv`
- `results/tables/plan_d_qwen2_ml1m_strict_prompt_pairwise_delta.csv`
- `results/tables/plan_d_qwen2_ml1m_strict_prompt_post_filter_profile.csv`
- `results/tables/plan_d_qwen2_ml1m_strict_prompt_manifest.csv`
- `results/raw_outputs/plan_d_qwen2_ml1m_strict_prompt_diagnostic_raw.jsonl` (Git-ignored)
- `report/plan_d_qwen2_ml1m_strict_prompt_diagnostic.md`

### Result

- Local Qwen2 requests: `{local_qwen_requests}`
- Runtime seconds: `{total_runtime_seconds:.6f}`
- Strict prompt HR@5: `{strict['hr_at_5']}`
- Strict prompt NDCG@5: `{strict['ndcg_at_5']}`
- Strict prompt hallucination rate: `{strict['hallucination_rate']}`
- Strict post-filter hallucination rate: `{strict_post['hallucination_rate']}`

### Notes And Limitations

- D5G is a diagnostic_subset prompt mitigation diagnostic, not exact reproduction and not full LLM reproduction.
- No OpenAI API call, Anthropic API call, external API call, download, model pull, recommender training, LastFM experiment, or Plan C modification occurred.
"""
    _replace_or_append_section(NOTES_PATH, "## 2026-07-01 - Plan D D5G: Qwen2 Strict Prompt And Post-Filter Diagnostic", section)


def _update_task_roadmap() -> None:
    frame = _read_or_empty(TASK_ROADMAP_PATH, TASK_ROADMAP_COLUMNS)
    for index, row in frame.iterrows():
        if row["phase"] == "D5G":
            frame.at[index, "task"] = "strict_candidate_id_prompt_and_post_filter_diagnostic"
            frame.at[index, "dataset"] = DATASET
            frame.at[index, "model"] = MODEL
            frame.at[index, "backend"] = BACKEND
            frame.at[index, "users"] = "20"
            frame.at[index, "candidate_pool_size"] = "20"
            frame.at[index, "prompt_strategy"] = PROMPT_STRATEGY
            frame.at[index, "output_format"] = "strict_json"
            frame.at[index, "metrics"] = "HR@5,NDCG@5,candidate_validity_rate,hallucination_rate,post_filter_delta"
            frame.at[index, "prerequisites"] = "D5E and D5F completed"
            frame.at[index, "status"] = "completed_diagnostic_subset"
            frame.at[index, "notes"] = "D5G completed 20 local Qwen2 strict-prompt requests plus deterministic post-filter analysis."
        elif row["phase"] == "D5H":
            frame.at[index, "task"] = "llm_reranking_diagnostic"
            frame.at[index, "status"] = "pending_after_d5g"
            frame.at[index, "prerequisites"] = "D5G completed"
            frame.at[index, "notes"] = "Use D5E-D5G reliability findings before reranking diagnostics."
    frame.to_csv(TASK_ROADMAP_PATH, index=False)


def _update_next_actions() -> None:
    rows = [
        _next_action("1", "test_stricter_candidate_id_only_prompt", "Review stricter prompt findings", "prompting; output_validity", "D5G tested the stricter prompt; the next step is to decide whether the mitigation is enough for later phases.", "Decision on strict prompt and post-filter policy.", "D5G completed.", "medium", "next", "Keep this action ID for continuity with the D5F roadmap; no extra inference is implied."),
        _next_action("2", "pull_mistral_7b_instruct_v02_quantized", "Pull Mistral 7B instruct v0.2 quantized", "paper_llm_mistral_7b_instruct_v0_2", "Mistral is the second local paper-family model for output-validity comparison.", "A recommended Mistral v0.2 quantized model appears in ollama list.", "Manual user pull only.", "medium", "after D5G review", "Do not pull automatically."),
        _next_action("3", "run_mistral_ml1m_ranking_diagnostic_20_users", "Run Mistral ML-1M ranking diagnostic", "llm_ranking; paper_llm_mistral_7b_instruct_v0_2", "A second local model helps compare validity, hallucination, and prompt-mitigation behavior.", "20-user Mistral ML-1M ranking diagnostic metrics.", "Mistral pulled and Qwen path stable.", "high", "after Mistral pull", "Keep as diagnostic; no exact paper reproduction claim."),
        _next_action("4", "run_llm_reranking_diagnostic", "Run LLM reranking diagnostic", "llm_reranking", "Reranking is a paper task still pending after local ranking, position-bias, and prompt-mitigation diagnostics.", "Reranking diagnostics with HR@5, NDCG@5, validity, and hallucination metrics.", "Stable D5G policy.", "high", "after ranking diagnostics", "Keep as diagnostic_subset unless official pools are recovered."),
        _next_action("5", "run_history_length_sensitivity_diagnostic", "Run history length sensitivity diagnostic", "history_length_sensitivity", "History length sensitivity is one of the paper LLM dimensions not yet evaluated.", "History-length diagnostic over a bounded local LLM subset.", "Stable prompt/parser path.", "medium", "after reranking planning", "Use local Qwen unless scope changes."),
        _next_action("6", "integrate_llm_results_or_blocker_into_final_report", "Integrate LLM results into final report", "final_report; local_llm_backend", "The final report should include D5E-D5G diagnostic limitations.", "Final LLM section grounded in generated CSVs.", "Follow-up diagnostics or blocker decision.", "low", "after follow-up diagnostics", "Do not claim exact paper reproduction."),
        _next_action("7", "optional_openai_api_diagnostic_if_budget_approved", "Optional OpenAI API diagnostic if budget approved", "api_availability", "Closed-source comparison remains conditional and optional.", "Small approved API diagnostic or documented no-API decision.", "Explicit user approval, API key, and budget.", "high", "optional", "No API calls by default."),
        _next_action("8", "prepare_exam_presentation_tables", "Prepare exam presentation tables", "final_report", "The oral exam needs compact generated tables for claims and limitations.", "Presentation-ready generated tables.", "Stable final report data tables.", "low", "after final report draft", "Use generated CSV values."),
        _next_action("9", "final_report", "Finalize report", "final_report", "The project needs final claims grounded in generated artifacts.", "Final report with limitations and generated metrics.", "LLM diagnostics integrated.", "low", "final", "Do not invent metric values."),
    ]
    pd.DataFrame(rows, columns=NEXT_ACTION_COLUMNS).to_csv(NEXT_ACTIONS_PATH, index=False)


def _update_feasibility_ledger(condition_summary: pd.DataFrame) -> None:
    frame = _read_or_empty(FEASIBILITY_LEDGER_PATH, LEDGER_COLUMNS)
    strict = _summary_row(condition_summary, STRICT_CONDITION)
    _upsert_ledger(
        frame,
        {
            "item_id": "qwen2_strict_prompt_post_filter_diagnostic",
            "area": "llm_analysis",
            "target": "Qwen2 strict candidate-ID prompt and post-filter diagnostic",
            "feasibility_status": "already_done",
            "required_resources": "D5E manifest, D5E raw JSONL, local Ollama qwen2:7b-instruct, strict parser",
            "estimated_effort": "done",
            "risk_level": "medium",
            "external_dependency": "yes",
            "recommended_decision": "Use D5G findings to decide whether strict prompts and post-filtering are required in later LLM diagnostics.",
            "notes": f"D5G completed as diagnostic_subset; strict prompt hallucination_rate={strict['hallucination_rate']}, mean_candidate_validity_rate={strict['mean_candidate_validity_rate']}.",
        },
    )
    for item_id in (
        "local_llm_backend",
        "ollama_backend",
        "llm_diagnostic_ranking",
        "paper_llm_qwen_2_7b_instruct",
        "qwen2_ml1m_ranking_diagnostic_20_users",
        "qwen2_candidate_position_bias_diagnostic",
    ):
        mask = frame["item_id"] == item_id
        if mask.any():
            frame.loc[mask, "recommended_decision"] = "Use D5G prompt-mitigation findings before broader LLM diagnostics."
            frame.loc[mask, "notes"] = "D5G completed strict candidate-ID prompt and deterministic post-filter analysis as diagnostic_subset; not exact LLM reproduction."
    frame.to_csv(FEASIBILITY_LEDGER_PATH, index=False)


def _update_reproducibility_matrix(condition_summary: pd.DataFrame) -> None:
    if not REPRO_MATRIX_PATH.exists():
        return
    strict = _summary_row(condition_summary, STRICT_CONDITION)
    strict_post = _summary_row(condition_summary, STRICT_POST_FILTERED_CONDITION)
    text = REPRO_MATRIX_PATH.read_text(encoding="utf-8")
    addition = (
        " D5F completed candidate position-bias diagnostics. D5G completed a strict candidate-ID-only prompt "
        "and deterministic post-filter diagnostic as diagnostic_subset work; it is not exact reproduction."
    )
    if "D5G completed a strict candidate-ID-only prompt" not in text:
        text = text.replace(
            "closed-source API reproduction remains conditional on explicit approval, keys, availability, and budget.",
            "closed-source API reproduction remains conditional on explicit approval, keys, availability, and budget." + addition,
        )
    text = _replace_table_row(
        text,
        "qwen-2-7b-instruct",
        f"| qwen-2-7b-instruct | `diagnostic_completed_mitigation_analyzed` | `local_open_model_near_exact` | D5E-D5G ran local Qwen2 ranking, validity, position-bias, strict-prompt, and post-filter diagnostics; D5G strict hallucination_rate `{strict['hallucination_rate']}` and strict post-filter hallucination_rate `{strict_post['hallucination_rate']}`. |",
    )
    text = _replace_table_row(
        text,
        "Hallucination",
        f"| Hallucination | `PARTIAL` | `extension` | D5E.1 confirmed hallucination_rate `0.370000`; D5G tested strict prompting and post-filtering, with strict hallucination_rate `{strict['hallucination_rate']}`. |",
    )
    text = _insert_after_line(
        text,
        "| Candidate position bias | `DIAGNOSTIC_ONLY` | `diagnostic` | D5F completed Qwen2 candidate-order robustness diagnostics over 20 ML-1M users and 3 variants. |",
        f"| Strict candidate-ID prompt and post-filter mitigation | `DIAGNOSTIC_ONLY` | `diagnostic` | D5G completed a diagnostic_subset prompt mitigation analysis; not exact reproduction and not full LLM reproduction. |",
    )
    text = _insert_after_line(
        text,
        "| Candidate position bias over real LLM outputs | `completed_diagnostic_subset` | D5F ran reverse_order, positive_first, and positive_last candidate-order diagnostics with local Qwen2. |",
        f"| Strict candidate-ID prompt and post-filter mitigation | `completed_diagnostic_subset` | D5G ran 20 local Qwen2 strict-prompt requests and deterministic original/strict post-filter analyses. |",
    )
    recommended = """## Recommended next steps

1. Review D5G strict prompt and post-filter findings.
2. Pull a quantized Mistral 7B Instruct v0.2 model manually.
3. Run the Mistral ML-1M ranking diagnostic for 20 users.
4. Run LLM reranking diagnostics.
5. Run history length sensitivity diagnostics.
6. Optionally run an OpenAI API diagnostic if budget is explicitly approved.
7. Integrate LLM results or the backend blocker into the final report.
8. Prepare exam presentation tables.
9. Finalize the report.
"""
    text = _replace_section_text(text, "## Recommended next steps", recommended)
    REPRO_MATRIX_PATH.write_text(text, encoding="utf-8")


def _load_and_validate_d5e_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing D5E manifest: {path}")
    frame = pd.read_csv(path, dtype=str).fillna("")
    errors: list[str] = []
    if len(frame) != 20:
        errors.append(f"expected exactly 20 D5E manifest rows, found {len(frame)}")
    for column, expected in (
        ("dataset", DATASET),
        ("model", MODEL),
        ("backend", BACKEND),
        ("candidate_pool_size", str(CANDIDATE_POOL_SIZE)),
        ("candidate_count", str(CANDIDATE_POOL_SIZE)),
        ("top_k", str(TOP_K)),
        ("positive_in_candidates", "true"),
        ("manifest_status", "ready_for_llm_inference"),
    ):
        values = {str(value).strip().lower() for value in frame[column].tolist()}
        if values != {expected.lower()}:
            errors.append(f"{column} expected {expected}, observed {sorted(values)}")
    for _, row in frame.iterrows():
        candidate_ids = json.loads(str(row["candidate_ids"]))
        if len(candidate_ids) != CANDIDATE_POOL_SIZE:
            errors.append(f"{row['instance_id']} has {len(candidate_ids)} candidates")
        if str(row["positive_item_id"]) not in set(candidate_ids):
            errors.append(f"{row['instance_id']} positive item missing from candidates")
    if errors:
        raise ValueError("Invalid D5E manifest for D5G: " + "; ".join(errors))
    return frame


def _load_d5e_raw_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"D5G requires existing D5E raw JSONL for post-filter analysis: {path}")
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    if len(rows) != 20:
        raise ValueError(f"Expected 20 D5E raw rows, found {len(rows)}")
    return rows


def _strict_notes(hallucinated_count: int) -> str:
    if hallucinated_count > 0:
        return "Strict prompt output still contains IDs outside the candidate set."
    return "Strict prompt output evaluated without manual correction."


def _prompt_mitigation_interpretation(condition_summary: pd.DataFrame) -> str:
    original = _summary_row(condition_summary, ORIGINAL_CONDITION)
    strict = _summary_row(condition_summary, STRICT_CONDITION)
    hallucination_delta = float(strict["hallucination_rate"]) - float(original["hallucination_rate"])
    validity_delta = float(strict["mean_candidate_validity_rate"]) - float(original["mean_candidate_validity_rate"])
    utility_delta = float(strict["hr_at_5"]) - float(original["hr_at_5"])
    if hallucination_delta < 0:
        hallucination_text = "The strict prompt reduced hallucination relative to D5E."
    elif hallucination_delta > 0:
        hallucination_text = "The strict prompt increased hallucination relative to D5E."
    else:
        hallucination_text = "The strict prompt did not change hallucination relative to D5E."
    return (
        f"{hallucination_text} Hallucination delta is `{hallucination_delta:.6f}`, "
        f"candidate-validity delta is `{validity_delta:.6f}`, and HR@5 delta is `{utility_delta:.6f}`."
    )


def _post_filter_interpretation(post_filter_profile: pd.DataFrame) -> str:
    parts = []
    for _, row in post_filter_profile.iterrows():
        parts.append(
            f"`{row['condition']}` removed `{row['total_items_removed_by_filter']}` invalid IDs, "
            f"leaving hallucination_rate `{row['filtered_hallucination_rate']}` and HR@5 `{row['filtered_HR@5']}`."
        )
    return " ".join(parts)


def _summary_row(condition_summary: pd.DataFrame, condition: str) -> pd.Series:
    rows = condition_summary.loc[condition_summary["condition"] == condition]
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row for {condition}, found {len(rows)}")
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
    parser = argparse.ArgumentParser(description="Run D5G Qwen2 strict prompt and post-filter diagnostic.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    result = run_diagnostic()
    write_outputs(result)
    LOGGER.info("Wrote %s", D5G_INSTANCE_SUMMARY_PATH)
    LOGGER.info("Wrote %s", D5G_CONDITION_SUMMARY_PATH)
    LOGGER.info("Wrote %s", D5G_REPORT_PATH)


if __name__ == "__main__":
    main()
