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

from src.analysis.plan_d_candidate_position_bias_analysis import (
    BACKEND,
    CANDIDATE_POOL_SIZE,
    DATASET,
    DISTRIBUTION_COLUMNS,
    INSTANCE_COLUMNS,
    MANIFEST_COLUMNS,
    METRIC_COLUMNS,
    METHOD,
    MODEL,
    PAIRWISE_COLUMNS,
    PHASE,
    PLAN,
    PROMPT_STRATEGY,
    SCOPE,
    TEMPERATURE,
    TOP_K,
    VARIANT_SUMMARY_COLUMNS,
    VARIANTS,
    ParsedInstance,
    build_metric_rows,
    build_pairwise_delta,
    build_position_bias_manifest,
    build_selected_position_distribution,
    build_variant_summary,
    evaluate_parsed_instance,
    original_reference_from_d5e_summary,
    original_valid_items_from_raw,
    raw_position_bias_trace_row,
)
from src.llm.plan_d_llm_ranking_evaluator import ManifestPromptBuilder
from src.llm.plan_d_llm_ranking_parser import parse_llm_ranking_response
from src.llm.plan_d_ollama_client import OllamaClient


LOGGER = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_METRICS_DIR = RESULTS_DIR / "metrics"
RAW_OUTPUTS_DIR = RESULTS_DIR / "raw_outputs"
REPORT_DIR = PROJECT_ROOT / "report"

D5E_MANIFEST_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ml1m_ranking_diagnostic_manifest.csv"
D5E_SUMMARY_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ml1m_ranking_diagnostic_summary.csv"
D5E_RAW_PATH = RAW_OUTPUTS_DIR / "plan_d_qwen2_ml1m_ranking_diagnostic_raw.jsonl"

D5F_MANIFEST_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ml1m_candidate_position_bias_manifest.csv"
D5F_INSTANCE_SUMMARY_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ml1m_candidate_position_bias_instance_summary.csv"
D5F_VARIANT_SUMMARY_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ml1m_candidate_position_bias_variant_summary.csv"
D5F_PAIRWISE_DELTA_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ml1m_candidate_position_bias_pairwise_delta.csv"
D5F_POSITION_DISTRIBUTION_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ml1m_candidate_position_bias_selected_position_distribution.csv"
D5F_METRICS_PATH = RESULTS_METRICS_DIR / "plan_d_qwen2_ml1m_candidate_position_bias_metrics.csv"
D5F_RAW_PATH = RAW_OUTPUTS_DIR / "plan_d_qwen2_ml1m_candidate_position_bias_diagnostic_raw.jsonl"
D5F_REPORT_PATH = REPORT_DIR / "plan_d_qwen2_ml1m_candidate_position_bias_diagnostic.md"

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
    d5e_manifest = pd.read_csv(D5E_MANIFEST_PATH, dtype=str).fillna("")
    d5e_summary = pd.read_csv(D5E_SUMMARY_PATH, dtype=str).fillna("")
    d5f_manifest = build_position_bias_manifest(d5e_manifest)
    _validate_manifest(d5f_manifest)
    d5f_manifest.to_csv(D5F_MANIFEST_PATH, index=False, columns=MANIFEST_COLUMNS)

    prompt_builder = ManifestPromptBuilder(PROJECT_ROOT / "data" / "reconstructed" / "plan_c")
    client = OllamaClient(timeout_seconds=TIMEOUT_SECONDS)
    d5e_rows_by_instance = {str(row["instance_id"]): row for _, row in d5e_manifest.iterrows()}

    instance_rows: list[dict[str, str]] = []
    raw_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, (_, manifest_row) in enumerate(d5f_manifest.iterrows(), start=1):
        original_row = d5e_rows_by_instance[str(manifest_row["original_instance_id"])].copy()
        original_row["instance_id"] = manifest_row["diagnostic_id"]
        original_row["candidate_ids"] = manifest_row["candidate_ids"]
        original_row["positive_item_id"] = manifest_row["positive_item_id"]
        payload = prompt_builder.build_from_manifest_row(pd.Series(original_row))
        candidate_ids = json.loads(str(manifest_row["candidate_ids"]))

        LOGGER.info("Running D5F local Qwen2 request %s/%s for %s", index, len(d5f_manifest), payload.instance_id)
        result = client.generate_json(model=MODEL, prompt=payload.prompt, temperature=TEMPERATURE)
        parsed = parse_llm_ranking_response(
            result.response_text if result.response_received else "",
            candidate_ids=set(candidate_ids),
            expected_top_k=TOP_K,
        )
        parsed_instance = ParsedInstance(
            diagnostic_id=str(manifest_row["diagnostic_id"]),
            original_instance_id=str(manifest_row["original_instance_id"]),
            user_id=str(manifest_row["user_id"]),
            variant=str(manifest_row["variant"]),
            candidate_ids=candidate_ids,
            positive_item_id=str(manifest_row["positive_item_id"]),
            original_positive_position=int(str(manifest_row["original_positive_position"])),
            variant_positive_position=int(str(manifest_row["variant_positive_position"])),
            response_received=result.response_received,
            parse_success=parsed.parse_success,
            schema_valid=parsed.schema_valid,
            ranked_item_ids=parsed.ranked_item_ids,
            valid_item_ids=parsed.valid_item_ids,
            invalid_item_ids=parsed.invalid_item_ids,
            selected_count=parsed.selected_count,
            candidate_validity_rate=parsed.candidate_validity_rate,
            duplicate_count=parsed.duplicate_count,
            hallucinated_count=parsed.hallucinated_count,
            latency_seconds=result.latency_seconds,
            parse_status=parsed.status,
            error_message=result.error or parsed.error_message,
        )
        instance_rows.append(evaluate_parsed_instance(parsed_instance))
        raw_rows.append(raw_position_bias_trace_row(parsed_instance=parsed_instance, raw_response_text=result.response_text))

    total_runtime_seconds = time.perf_counter() - started
    instance_summary = pd.DataFrame(instance_rows, columns=INSTANCE_COLUMNS)
    d5e_raw_rows = _read_jsonl(D5E_RAW_PATH)
    original_reference = original_reference_from_d5e_summary(d5e_summary)
    original_valid_items = original_valid_items_from_raw(d5e_raw_rows)
    variant_summary = build_variant_summary(
        instance_summary,
        original_reference=original_reference,
        original_valid_items_by_instance=original_valid_items,
    )
    pairwise_delta = build_pairwise_delta(variant_summary, original_reference)
    distribution = build_selected_position_distribution(instance_summary)
    metrics = build_metric_rows(variant_summary)
    return {
        "manifest": d5f_manifest,
        "instance_summary": instance_summary,
        "variant_summary": variant_summary,
        "pairwise_delta": pairwise_delta,
        "distribution": distribution,
        "metrics": metrics,
        "raw_rows": raw_rows,
        "total_runtime_seconds": total_runtime_seconds,
    }


def write_outputs(result: dict[str, object]) -> None:
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = result["manifest"]
    instance_summary = result["instance_summary"]
    variant_summary = result["variant_summary"]
    pairwise_delta = result["pairwise_delta"]
    distribution = result["distribution"]
    metrics = result["metrics"]
    raw_rows = result["raw_rows"]
    total_runtime_seconds = float(result["total_runtime_seconds"])

    assert isinstance(manifest, pd.DataFrame)
    assert isinstance(instance_summary, pd.DataFrame)
    assert isinstance(variant_summary, pd.DataFrame)
    assert isinstance(pairwise_delta, pd.DataFrame)
    assert isinstance(distribution, pd.DataFrame)
    assert isinstance(metrics, pd.DataFrame)
    assert isinstance(raw_rows, list)

    manifest.to_csv(D5F_MANIFEST_PATH, index=False, columns=MANIFEST_COLUMNS)
    instance_summary.to_csv(D5F_INSTANCE_SUMMARY_PATH, index=False, columns=INSTANCE_COLUMNS)
    variant_summary.to_csv(D5F_VARIANT_SUMMARY_PATH, index=False, columns=VARIANT_SUMMARY_COLUMNS)
    pairwise_delta.to_csv(D5F_PAIRWISE_DELTA_PATH, index=False, columns=PAIRWISE_COLUMNS)
    distribution.to_csv(D5F_POSITION_DISTRIBUTION_PATH, index=False, columns=DISTRIBUTION_COLUMNS)
    metrics.to_csv(D5F_METRICS_PATH, index=False, columns=METRIC_COLUMNS)
    with D5F_RAW_PATH.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    D5F_REPORT_PATH.write_text(_report(variant_summary, pairwise_delta, total_runtime_seconds), encoding="utf-8")
    _update_notes(variant_summary, total_runtime_seconds)
    _update_task_roadmap()
    _update_next_actions()
    _update_feasibility_ledger(variant_summary)
    _update_reproducibility_matrix(variant_summary)


def _report(variant_summary: pd.DataFrame, pairwise_delta: pd.DataFrame, total_runtime_seconds: float) -> str:
    front_interpretation = _position_interpretation(variant_summary)
    utility_interpretation = _utility_interpretation(pairwise_delta)
    reliability_interpretation = _reliability_interpretation(pairwise_delta)
    rows = "\n".join(
        f"| `{row['variant']}` | `{row['hr_at_5']}` | `{row['ndcg_at_5']}` | `{row['mean_candidate_validity_rate']}` | `{row['hallucination_rate']}` | `{row['mean_selected_input_position']}` | `{row['front_half_selection_rate']}` | `{row['position_1_selection_rate']}` | `{row['position_20_selection_rate']}` |"
        for _, row in variant_summary.iterrows()
    )
    return f"""# Plan D Qwen2 ML-1M Candidate Position Bias Diagnostic

## Purpose

D5F tests whether the local Qwen2 ML-1M ranking diagnostic is sensitive to candidate input order. It reuses the same 20 D5E diagnostic instances and reorders the same candidate sets into `reverse_order`, `positive_first`, and `positive_last`.

## Scope And Classification

- D5F is a diagnostic_subset robustness diagnostic.
- Classification: `diagnostic_subset` robustness diagnostic.
- This is not exact reproduction.
- This is not full LLM reproduction.
- This is a local open model diagnostic only.
- Backend/model: local Ollama with `qwen2:7b-instruct`.
- New local Qwen2 requests: `60` (`20` users times `3` variants).
- Temperature: `0`.
- Top-k: `5`.
- Candidate pool size: `20`.
- Total runtime seconds: `{total_runtime_seconds:.6f}`.

## Execution Boundaries

- No OpenAI API call was made.
- No Anthropic API call was made.
- No external API call was made.
- No download was performed.
- No model pull was performed.
- No recommender training was performed.
- Plan C was not modified.
- Raw JSONL outputs are stored under `results/raw_outputs/` and remain Git-ignored.
- The original D5E order was used as a reference and was not rerun.

## Variant Summary

| Variant | HR@5 | NDCG@5 | Candidate Validity | Hallucination Rate | Mean Selected Position | Front-Half Rate | Position-1 Rate | Position-20 Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

## Position Bias Interpretation

{front_interpretation}

The selected-position metrics distinguish candidate validity, hallucination, utility, and position bias. Hallucinated IDs have no candidate input position, so position-bias rates are computed over valid selected candidate IDs.

## Utility Interpretation

{utility_interpretation}

## Reliability Interpretation

{reliability_interpretation}

## Beyond Utility Interpretation

D5F supports the broader Beyond Utility argument because candidate reordering can affect reliability and candidate-position behavior independently of HR/NDCG. A parse-valid LLM output can still be invalid, hallucinated, or position-sensitive.

## Limitations

- Only 20 users.
- Only ML-1M.
- Only Qwen2.
- Only `openp5_base`.
- Only three deterministic reorderings.
- Not exact paper reproduction.
- Not full LLM reproduction.

## Recommended Next Steps

1. Test a stricter candidate-ID-only prompt or post-filtered evaluation.
2. Pull and test Mistral.
3. Run LLM reranking diagnostic.
4. Run history length sensitivity diagnostic.
"""


def _update_notes(variant_summary: pd.DataFrame, total_runtime_seconds: float) -> None:
    section = f"""
## 2026-07-01 - Plan D D5F: Qwen2 Candidate Position Bias Diagnostic

### Changes

- Ran the D5F local Qwen2/Ollama candidate position-bias diagnostic over 20 ML-1M users and 3 deterministic candidate-order variants.
- Wrote manifest, instance summary, variant summary, pairwise deltas, selected-position distribution, metric-style CSV, raw ignored JSONL, and report artifacts.
- Used existing D5E original-order outputs as the reference; original order was not rerun.

### Commands Run

```powershell
python src/experiments/run_plan_d_qwen2_ml1m_candidate_position_bias_diagnostic.py
```

### Outputs Generated

- `results/metrics/plan_d_qwen2_ml1m_candidate_position_bias_metrics.csv`
- `results/tables/plan_d_qwen2_ml1m_candidate_position_bias_instance_summary.csv`
- `results/tables/plan_d_qwen2_ml1m_candidate_position_bias_variant_summary.csv`
- `results/tables/plan_d_qwen2_ml1m_candidate_position_bias_pairwise_delta.csv`
- `results/tables/plan_d_qwen2_ml1m_candidate_position_bias_selected_position_distribution.csv`
- `results/tables/plan_d_qwen2_ml1m_candidate_position_bias_manifest.csv`
- `results/raw_outputs/plan_d_qwen2_ml1m_candidate_position_bias_diagnostic_raw.jsonl` (Git-ignored)
- `report/plan_d_qwen2_ml1m_candidate_position_bias_diagnostic.md`

### Result

- Local Qwen2 requests: `60`
- Runtime seconds: `{total_runtime_seconds:.6f}`
- Variants completed: `{', '.join(variant_summary['variant'].tolist())}`

### Notes And Limitations

- D5F is a diagnostic_subset robustness diagnostic, not exact reproduction and not full LLM reproduction.
- No OpenAI API call, Anthropic API call, external API call, download, model pull, recommender training, LastFM experiment, or Plan C modification occurred.
"""
    _replace_or_append_section(NOTES_PATH, "## 2026-07-01 - Plan D D5F: Qwen2 Candidate Position Bias Diagnostic", section)


def _update_task_roadmap() -> None:
    frame = _read_or_empty(TASK_ROADMAP_PATH, TASK_ROADMAP_COLUMNS)
    for index, row in frame.iterrows():
        if row["phase"] == "D5F":
            frame.at[index, "status"] = "completed_diagnostic_subset"
            frame.at[index, "prerequisites"] = "D5E.1 completed"
            frame.at[index, "notes"] = "D5F completed 60 local Qwen2 candidate-order requests; not exact reproduction."
        elif row["phase"] in {"D5G", "D5H"}:
            frame.at[index, "status"] = "pending_after_d5f"
            frame.at[index, "prerequisites"] = "D5F completed"
            frame.at[index, "notes"] = "Use D5F position-bias findings before broader LLM diagnostics."
    frame.to_csv(TASK_ROADMAP_PATH, index=False)


def _update_next_actions() -> None:
    rows = [
        _next_action("1", "test_stricter_candidate_id_only_prompt", "Test stricter candidate-ID-only prompt", "prompting; output_validity", "D5E/D5F show candidate adherence and position behavior need stronger constraints.", "A bounded prompt-constraint diagnostic or documented design.", "D5F completed.", "medium", "next", "Run only if a later phase explicitly allows local inference."),
        _next_action("2", "pull_mistral_7b_instruct_v02_quantized", "Pull Mistral 7B instruct v0.2 quantized", "paper_llm_mistral_7b_instruct_v0_2", "Mistral is the second local paper-family model for output-validity comparison.", "A recommended Mistral v0.2 quantized model appears in ollama list.", "Manual user pull only.", "medium", "after Qwen review", "Do not pull automatically."),
        _next_action("3", "run_mistral_ml1m_ranking_diagnostic_20_users", "Run Mistral ML-1M ranking diagnostic", "llm_ranking; paper_llm_mistral_7b_instruct_v0_2", "A second local model helps compare validity, hallucination, and position behavior.", "20-user Mistral ML-1M ranking diagnostic metrics.", "Mistral pulled and Qwen path stable.", "high", "after Mistral pull", "Keep as diagnostic; no exact paper reproduction claim."),
        _next_action("4", "run_llm_reranking_diagnostic", "Run LLM reranking diagnostic", "llm_reranking", "Reranking is a paper task still pending after local ranking and position-bias diagnostics.", "Reranking diagnostics with HR@5, NDCG@5, validity, and hallucination metrics.", "Stable D5F findings.", "high", "after ranking diagnostics", "Keep as diagnostic_subset unless official pools are recovered."),
        _next_action("5", "run_history_length_sensitivity_diagnostic", "Run history length sensitivity diagnostic", "history_length_sensitivity", "History length sensitivity is one of the paper LLM dimensions not yet evaluated.", "History-length diagnostic over a bounded local LLM subset.", "Stable prompt/parser path.", "medium", "after ranking diagnostics", "Use local Qwen unless scope changes."),
        _next_action("6", "optional_openai_api_diagnostic_if_budget_approved", "Optional OpenAI API diagnostic if budget approved", "api_availability", "Closed-source comparison remains conditional and optional.", "Small approved API diagnostic or documented no-API decision.", "Explicit user approval, API key, and budget.", "high", "optional", "No API calls by default."),
        _next_action("7", "integrate_llm_results_or_blocker_into_final_report", "Integrate LLM results or blocker into final report", "final_report; local_llm_backend", "The final report should include D5E/D5E.1/D5F diagnostic limitations.", "Final LLM section grounded in generated CSVs.", "Follow-up diagnostics or blocker decision.", "low", "after follow-up diagnostics", "Do not claim exact paper reproduction."),
        _next_action("8", "prepare_exam_presentation_tables", "Prepare exam presentation tables", "final_report", "The oral exam needs compact generated tables for claims and limitations.", "Presentation-ready generated tables.", "Stable final report data tables.", "low", "after final report draft", "Use generated CSV values."),
        _next_action("9", "final_report", "Finalize report", "final_report", "The project needs final claims grounded in generated artifacts.", "Final report with limitations and generated metrics.", "LLM diagnostics integrated.", "low", "final", "Do not invent metric values."),
    ]
    pd.DataFrame(rows, columns=NEXT_ACTION_COLUMNS).to_csv(NEXT_ACTIONS_PATH, index=False)


def _update_feasibility_ledger(variant_summary: pd.DataFrame) -> None:
    frame = _read_or_empty(FEASIBILITY_LEDGER_PATH, LEDGER_COLUMNS)
    _upsert_ledger(
        frame,
        {
            "item_id": "qwen2_candidate_position_bias_diagnostic",
            "area": "llm_analysis",
            "target": "Qwen2 ML-1M candidate position-bias diagnostic",
            "feasibility_status": "already_done",
            "required_resources": "D5E manifest, local Ollama qwen2:7b-instruct, strict parser",
            "estimated_effort": "done",
            "risk_level": "medium",
            "external_dependency": "yes",
            "recommended_decision": "Use D5F findings to design stricter candidate-ID-only prompts and later reranking diagnostics.",
            "notes": "D5F completed 60 local Qwen2 requests across reverse_order, positive_first, and positive_last.",
        },
    )
    for item_id in ("position_bias_after_llm", "llm_diagnostic_ranking", "paper_llm_qwen_2_7b_instruct"):
        mask = frame["item_id"] == item_id
        if mask.any():
            frame.loc[mask, "recommended_decision"] = "Use D5F candidate position-bias findings before broader LLM diagnostics."
            frame.loc[mask, "notes"] = "D5F completed as diagnostic_subset robustness analysis; not exact LLM reproduction."
    frame.to_csv(FEASIBILITY_LEDGER_PATH, index=False)


def _update_reproducibility_matrix(variant_summary: pd.DataFrame) -> None:
    if not REPRO_MATRIX_PATH.exists():
        return
    text = REPRO_MATRIX_PATH.read_text(encoding="utf-8")
    text = _replace_table_row(
        text,
        "Candidate position bias",
        "| Candidate position bias | `DIAGNOSTIC_ONLY` | `diagnostic` | D5F completed Qwen2 candidate-order robustness diagnostics over 20 ML-1M users and 3 variants. |",
    )
    text = _replace_table_row(
        text,
        "Candidate position bias over real LLM outputs",
        "| Candidate position bias over real LLM outputs | `completed_diagnostic_subset` | D5F ran reverse_order, positive_first, and positive_last candidate-order diagnostics with local Qwen2. |",
    )
    text = text.replace(
        "Reranking, candidate position bias, and history-length sensitivity remain pending",
        "Reranking and history-length sensitivity remain pending",
    )
    recommended = """## Recommended next steps

1. Test a stricter candidate-ID-only prompt.
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


def _validate_manifest(frame: pd.DataFrame) -> None:
    if len(frame) != 60:
        raise ValueError(f"D5F manifest must contain 60 rows, found {len(frame)}")
    if set(frame["variant"]) != set(VARIANTS):
        raise ValueError("D5F manifest missing one or more variants")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _position_interpretation(variant_summary: pd.DataFrame) -> str:
    parts = []
    for _, row in variant_summary.iterrows():
        front = float(row["front_half_selection_rate"])
        back = float(row["back_half_selection_rate"])
        if front > back:
            preference = "front-position preference"
        elif back > front:
            preference = "back-position preference"
        else:
            preference = "balanced front/back selection"
        parts.append(f"`{row['variant']}` shows {preference} ({front:.6f} front, {back:.6f} back).")
    return " ".join(parts)


def _utility_interpretation(pairwise_delta: pd.DataFrame) -> str:
    rows = pairwise_delta.loc[pairwise_delta["metric_name"].isin(["HR@5", "NDCG@5"])]
    return " ".join(
        f"`{row['variant']}` {row['metric_name']} delta vs original is `{row['delta']}` ({row['interpretation']})."
        for _, row in rows.iterrows()
    )


def _reliability_interpretation(pairwise_delta: pd.DataFrame) -> str:
    rows = pairwise_delta.loc[pairwise_delta["metric_name"].isin(["hallucination_rate", "mean_candidate_validity_rate"])]
    return " ".join(
        f"`{row['variant']}` {row['metric_name']} delta vs original is `{row['delta']}` ({row['interpretation']})."
        for _, row in rows.iterrows()
    )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D5F Qwen2 candidate position-bias diagnostic.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    result = run_diagnostic()
    write_outputs(result)
    LOGGER.info("Wrote %s", D5F_INSTANCE_SUMMARY_PATH)
    LOGGER.info("Wrote %s", D5F_VARIANT_SUMMARY_PATH)
    LOGGER.info("Wrote %s", D5F_REPORT_PATH)


if __name__ == "__main__":
    main()
