from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_d_qwen2_output_validity_analysis import (
    ANALYSIS_PHASE,
    BACKEND,
    BREAKDOWN_COLUMNS,
    DATASET,
    INVALID_PATTERN_COLUMNS,
    MODEL,
    PHASE,
    PLAN,
    PROFILE_COLUMNS,
    PROMPT_STRATEGY,
    SCOPE,
    VALID_ONLY_COLUMNS,
    build_output_validity_analysis,
)


LOGGER = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
REPORT_DIR = PROJECT_ROOT / "report"

D5E_INSTANCE_SUMMARY_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ml1m_ranking_diagnostic_instance_summary.csv"
D5E_AGGREGATE_SUMMARY_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ml1m_ranking_diagnostic_summary.csv"
D5E_RAW_OUTPUT_PATH = RESULTS_DIR / "raw_outputs" / "plan_d_qwen2_ml1m_ranking_diagnostic_raw.jsonl"

BREAKDOWN_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_output_validity_breakdown.csv"
VALID_ONLY_UTILITY_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_valid_only_utility_summary.csv"
HALLUCINATION_PROFILE_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_hallucination_instance_profile.csv"
INVALID_ITEM_PATTERNS_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_invalid_item_patterns.csv"
REPORT_PATH = REPORT_DIR / "plan_d_qwen2_output_validity_and_hallucination_analysis.md"

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


def run_analysis() -> None:
    analysis = build_output_validity_analysis(
        instance_summary_path=D5E_INSTANCE_SUMMARY_PATH,
        aggregate_summary_path=D5E_AGGREGATE_SUMMARY_PATH,
        raw_output_path=D5E_RAW_OUTPUT_PATH,
    )
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    analysis.breakdown.to_csv(BREAKDOWN_PATH, index=False, columns=BREAKDOWN_COLUMNS)
    analysis.valid_only_utility.to_csv(VALID_ONLY_UTILITY_PATH, index=False, columns=VALID_ONLY_COLUMNS)
    analysis.hallucination_profile.to_csv(HALLUCINATION_PROFILE_PATH, index=False, columns=PROFILE_COLUMNS)
    analysis.invalid_item_patterns.to_csv(INVALID_ITEM_PATTERNS_PATH, index=False, columns=INVALID_PATTERN_COLUMNS)
    REPORT_PATH.write_text(_analysis_report(analysis.raw_available), encoding="utf-8")
    _update_task_roadmap()
    _update_next_actions()
    _update_feasibility_ledger()
    _update_notes(analysis.raw_available)
    _update_reproducibility_matrix()


def _analysis_report(raw_available: bool) -> str:
    breakdown = _breakdown_row()
    utility = _utility_frame()
    profile = _profile_frame()
    patterns = _patterns_frame()
    original = _utility_row(utility, "original_parsed_output")
    valid_only = _utility_row(utility, "valid_only_output")
    strict = _utility_row(utility, "invalid_as_failure")
    hallucinated_instances = int(profile["has_hallucination"].astype(str).str.lower().eq("true").sum())
    high_severity = int(profile["hallucination_severity"].astype(str).eq("high").sum())
    raw_text = (
        "The ignored local D5E raw JSONL was available, so compact invalid-ID pattern counts and valid-only utility were computed from parsed fields."
        if raw_available
        else "The ignored local D5E raw JSONL was unavailable, so raw-level invalid-ID pattern analysis and valid-only utility are marked unavailable."
    )
    top_patterns = _top_patterns_text(patterns)
    return f"""# Plan D Qwen2 Output Validity and Hallucination Analysis

## Purpose

D5E.1 analyzes the D5E Qwen2 ML-1M ranking diagnostic outputs without running new LLM inference. The goal is to explain the gap between perfect JSON/schema compliance and weaker candidate adherence.

## Input Artifacts

- `results/metrics/plan_d_qwen2_ml1m_ranking_diagnostic_metrics.csv`
- `results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_instance_summary.csv`
- `results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_summary.csv`
- `report/plan_d_qwen2_ml1m_ranking_diagnostic.md`
- Optional ignored raw trace: `results/raw_outputs/plan_d_qwen2_ml1m_ranking_diagnostic_raw.jsonl`

{raw_text}

No new LLM inference was run. No Ollama command, Ollama HTTP API call, OpenAI API call, Anthropic API call, remote API call, download, model pull, recommender training, BPRMF/LightGCN/SASRec/GRU4Rec run, or LastFM experiment occurred.

## D5E Result Recap

- HR@5: `{original['hit_rate_at_5']}`
- NDCG@5: `{original['ndcg_at_5']}`
- response_received_rate: `{breakdown['response_received_rate']}`
- parse_success_rate: `{breakdown['parse_success_rate']}`
- schema_valid_rate: `{breakdown['schema_valid_rate']}`
- exact_top_k_rate: `{breakdown['exact_top_k_rate']}`
- mean_candidate_validity_rate: `{breakdown['mean_candidate_validity_rate']}`
- hallucination_rate: `{breakdown['hallucination_rate']}`
- duplicate_rate: `{breakdown['duplicate_rate']}`

## Output Validity Analysis

Qwen2 was reliable at the transport, JSON, and schema levels in this diagnostic subset: responses were received for all `{breakdown['num_instances']}` instances, every response parsed as JSON, every response passed schema validation, and every response returned exactly five selected IDs.

Candidate adherence was the main failure mode. The run selected `{breakdown['total_selected_items']}` item IDs, of which `{breakdown['total_valid_selected_items']}` were valid candidates and `{breakdown['total_hallucinated_items']}` were outside the candidate sets. The micro candidate-validity rate was `{breakdown['micro_candidate_validity_rate']}`.

## Hallucination Analysis

- Instances with hallucinations: `{hallucinated_instances}` of `{breakdown['num_instances']}`
- High-severity hallucination instances: `{high_severity}`
- Total hallucinated items: `{breakdown['total_hallucinated_items']}`
- Invalid item pattern summary: {top_patterns}

The invalid IDs mostly preserve the MovieLens numeric ID format when raw parsed fields are available. This suggests the model often selected plausible movie IDs rather than malformed strings, but those IDs were not members of the provided candidate set.

## Valid-Only Utility Analysis

| Variant | HR@5 | NDCG@5 | Mean Positive Rank | Hits | Notes |
|---|---:|---:|---:|---:|---|
| original_parsed_output | `{original['hit_rate_at_5']}` | `{original['ndcg_at_5']}` | `{original['mean_positive_rank']}` | `{original['num_hits']}` | Original D5E utility. |
| valid_only_output | `{valid_only['hit_rate_at_5']}` | `{valid_only['ndcg_at_5']}` | `{valid_only['mean_positive_rank']}` | `{valid_only['num_hits']}` | Removes hallucinated IDs while preserving valid returned order when raw parsed fields are available. |
| invalid_as_failure | `{strict['hit_rate_at_5']}` | `{strict['ndcg_at_5']}` | `{strict['mean_positive_rank']}` | `{strict['num_hits']}` | Counts any hallucinating instance as a utility failure. |

## Operational Interpretation

Qwen2 is reliable at JSON/schema level in this run, but it is not fully reliable at candidate-adherence level. A production recommender would need constrained decoding, post-filtering, stronger prompt constraints, or a candidate-ID-only representation before using these outputs directly.

## Methodological Importance

This supports the Beyond Utility motivation. HR/NDCG alone are insufficient because an LLM can return parseable outputs and still recommend invalid items outside the candidate set. Candidate validity, hallucination rate, duplicate rate, and parse/schema rates must be reported alongside utility.

## Limitations

- Only 20 users.
- Only ML-1M.
- Only Qwen2.
- Only `openp5_base`.
- Raw-level pattern analysis depends on local ignored raw JSONL availability.
- No new inference in this phase.

## Recommended Next Steps

1. Run candidate position bias diagnostic.
2. Test a stricter candidate-ID-only prompt or post-filtered evaluation.
3. Pull and test Mistral.
4. Run LLM reranking diagnostic.
"""


def _update_task_roadmap() -> None:
    frame = _read_or_empty(TASK_ROADMAP_PATH, TASK_ROADMAP_COLUMNS)
    rows = []
    has_d5e1 = bool((frame["phase"] == "D5E.1").any()) if not frame.empty else False
    inserted = False
    for _, current in frame.iterrows():
        row = current.to_dict()
        if row["phase"] == "D5E.1":
            row.update(_d5e1_roadmap_row())
        if row["phase"] == "D5F":
            row["status"] = "pending_after_validity_analysis"
            row["prerequisites"] = "D5E.1 completed"
            row["notes"] = "Run over real D5E parsed outputs; candidate adherence is the main Qwen2 failure mode."
        if row["phase"] == "D5G":
            row["status"] = "pending_after_validity_analysis"
            row["prerequisites"] = "D5E.1 completed and candidate-pool policy fixed"
            row["notes"] = "Use D5E.1 candidate-adherence findings before reranking diagnostics."
        if row["phase"] == "D5H":
            row["status"] = "partial_qwen_completed"
            row["prerequisites"] = "D5E.1 completed for Qwen; Mistral still pending"
            row["notes"] = "D5E.1 completed Qwen output-validity/hallucination analysis without inference."
        rows.append(row)
        if row["phase"] == "D5E" and not has_d5e1 and not inserted:
            rows.append(_d5e1_roadmap_row())
            inserted = True
    if not has_d5e1 and not inserted:
        rows.append(_d5e1_roadmap_row())
    pd.DataFrame(rows, columns=TASK_ROADMAP_COLUMNS).to_csv(TASK_ROADMAP_PATH, index=False)


def _d5e1_roadmap_row() -> dict[str, str]:
    return {
        "phase": "D5E.1",
        "task": "qwen2_output_validity_hallucination_analysis",
        "dataset": DATASET,
        "model": MODEL,
        "backend": "none",
        "users": "20",
        "candidate_pool_size": "20",
        "prompt_strategy": PROMPT_STRATEGY,
        "output_format": "analysis_tables",
        "metrics": "candidate_validity_rate,hallucination_rate,valid_only_utility,invalid_item_patterns",
        "prerequisites": "D5E completed",
        "status": "completed_no_inference",
        "notes": "Analyzed D5E Qwen2 output validity and hallucination behavior; no new LLM inference was run.",
    }


def _update_next_actions() -> None:
    rows = [
        _next_action("1", "run_candidate_position_bias_diagnostic", "Run candidate position-bias diagnostic", "position_bias", "D5E.1 showed candidate adherence is the main Qwen2 failure mode; position bias should be checked on real parsed outputs.", "Position-bias metrics over D5E parsed outputs.", "D5E.1 completed.", "medium", "next", "Use generated D5E/D5E.1 CSVs; no fabricated outputs."),
        _next_action("2", "test_stricter_candidate_id_only_prompt", "Test stricter candidate-ID-only prompt", "prompting; output_validity", "Qwen2 followed JSON schema but hallucinated candidate IDs, so prompt constraints should target candidate adherence.", "A bounded prompt-constraint diagnostic or documented design.", "D5E.1 completed.", "medium", "after position-bias plan", "Run only if a later phase explicitly allows local inference."),
        _next_action("3", "pull_mistral_7b_instruct_v02_quantized", "Pull Mistral 7B instruct v0.2 quantized", "paper_llm_mistral_7b_instruct_v0_2", "Mistral is the second local paper-family model for output-validity comparison.", "A recommended Mistral v0.2 quantized model appears in ollama list.", "Manual user pull only.", "medium", "after Qwen review", "Do not pull automatically."),
        _next_action("4", "run_mistral_ml1m_ranking_diagnostic_20_users", "Run Mistral ML-1M ranking diagnostic", "llm_ranking; paper_llm_mistral_7b_instruct_v0_2", "A second local model helps compare validity and hallucination behavior.", "20-user Mistral ML-1M ranking diagnostic metrics.", "Mistral pulled and Qwen path stable.", "high", "after Mistral pull", "Keep as diagnostic; no exact paper reproduction claim."),
        _next_action("5", "run_llm_reranking_diagnostic", "Run LLM reranking diagnostic", "llm_reranking", "Reranking is a paper task still pending after the first local ranking diagnostic.", "Reranking diagnostics with HR@5, NDCG@5, validity, and hallucination metrics.", "Stable D5E.1 output-validity analysis.", "high", "after ranking diagnostics", "Keep as diagnostic_subset unless official pools are recovered."),
        _next_action("6", "run_history_length_sensitivity_diagnostic", "Run history length sensitivity diagnostic", "history_length_sensitivity", "History length sensitivity is one of the paper LLM dimensions not yet evaluated.", "History-length diagnostic over a bounded local LLM subset.", "Stable prompt/parser path.", "medium", "after ranking diagnostics", "Use local Qwen unless scope changes."),
        _next_action("7", "optional_openai_api_diagnostic_if_budget_approved", "Optional OpenAI API diagnostic if budget approved", "api_availability", "Closed-source comparison remains conditional and optional.", "Small approved API diagnostic or documented no-API decision.", "Explicit user approval, API key, and budget.", "high", "optional", "No API calls by default."),
        _next_action("8", "integrate_llm_results_or_blocker_into_final_report", "Integrate LLM results or blocker into final report", "final_report; local_llm_backend", "The final report should include D5E utility and D5E.1 validity limitations.", "Final LLM section grounded in generated D5E/D5E.1 CSVs.", "Follow-up diagnostics or blocker decision.", "low", "after follow-up diagnostics", "Do not claim exact paper reproduction."),
        _next_action("9", "prepare_exam_presentation_tables", "Prepare exam presentation tables", "final_report", "The oral exam needs compact generated tables for claims and limitations.", "Presentation-ready generated tables.", "Stable final report data tables.", "low", "after final report draft", "Use generated CSV values."),
        _next_action("10", "final_report", "Finalize report", "final_report", "The project needs final claims grounded in generated artifacts.", "Final report with limitations and generated metrics.", "LLM diagnostics integrated.", "low", "final", "Do not invent metric values."),
    ]
    pd.DataFrame(rows, columns=NEXT_ACTION_COLUMNS).to_csv(NEXT_ACTIONS_PATH, index=False)


def _update_feasibility_ledger() -> None:
    breakdown = _breakdown_row()
    frame = _read_or_empty(FEASIBILITY_LEDGER_PATH, LEDGER_COLUMNS)
    _upsert_ledger(
        frame,
        {
            "item_id": "qwen2_output_validity_hallucination_analysis",
            "area": "llm_analysis",
            "target": "D5E Qwen2 output validity, hallucination, and valid-only utility analysis",
            "feasibility_status": "already_done",
            "required_resources": "D5E committed summary tables; optional ignored raw JSONL for invalid-ID patterns",
            "estimated_effort": "done",
            "risk_level": "low",
            "external_dependency": "no",
            "recommended_decision": "Use D5E.1 analysis to prioritize position bias and stricter candidate-ID constraints.",
            "notes": (
                f"completed_no_inference; response_received_rate={breakdown['response_received_rate']}; "
                f"parse_success_rate={breakdown['parse_success_rate']}; schema_valid_rate={breakdown['schema_valid_rate']}; "
                f"hallucination_rate={breakdown['hallucination_rate']}."
            ),
        },
    )
    for item_id in (
        "local_llm_backend",
        "ollama_backend",
        "local_ollama_backend",
        "llm_diagnostic_ranking",
        "paper_llm_qwen_2_7b_instruct",
        "qwen2_ml1m_ranking_diagnostic_20_users",
        "position_bias_after_llm",
    ):
        mask = frame["item_id"] == item_id
        if mask.any():
            frame.loc[mask, "recommended_decision"] = "Use D5E.1 findings before broader LLM diagnostics."
            frame.loc[mask, "notes"] = (
                "D5E.1 found perfect response/parse/schema rates but high hallucination; "
                "candidate adherence is the main Qwen2 failure mode."
            )
    frame.to_csv(FEASIBILITY_LEDGER_PATH, index=False)


def _update_notes(raw_available: bool) -> None:
    breakdown = _breakdown_row()
    utility = _utility_frame()
    profile = _profile_frame()
    hallucinated_instances = int(profile["has_hallucination"].astype(str).str.lower().eq("true").sum())
    valid_only = _utility_row(utility, "valid_only_output")
    raw_note = "available for compact parsed invalid-ID analysis" if raw_available else "unavailable; raw-level analysis marked unavailable"
    section = f"""
## 2026-07-01 - Plan D D5E.1: Qwen2 Output Validity and Hallucination Analysis

### Changes

- Analyzed D5E Qwen2 ML-1M ranking outputs without running new LLM inference.
- Wrote aggregate output-validity, valid-only utility, hallucination instance profile, and invalid item pattern tables.
- Updated Plan D LLM roadmap, next actions, feasibility ledger, and reproducibility matrix.

### Commands Run

```powershell
python src/experiments/analyze_plan_d_qwen2_output_validity.py
```

### Outputs Generated

- `results/tables/plan_d_qwen2_output_validity_breakdown.csv`
- `results/tables/plan_d_qwen2_valid_only_utility_summary.csv`
- `results/tables/plan_d_qwen2_hallucination_instance_profile.csv`
- `results/tables/plan_d_qwen2_invalid_item_patterns.csv`
- `report/plan_d_qwen2_output_validity_and_hallucination_analysis.md`

### Result

- response_received_rate: `{breakdown['response_received_rate']}`
- parse_success_rate: `{breakdown['parse_success_rate']}`
- schema_valid_rate: `{breakdown['schema_valid_rate']}`
- exact_top_k_rate: `{breakdown['exact_top_k_rate']}`
- mean_candidate_validity_rate: `{breakdown['mean_candidate_validity_rate']}`
- micro_candidate_validity_rate: `{breakdown['micro_candidate_validity_rate']}`
- hallucination_rate: `{breakdown['hallucination_rate']}`
- duplicate_rate: `{breakdown['duplicate_rate']}`
- instances_with_hallucinations: `{hallucinated_instances}`
- valid_only_HR@5: `{valid_only['hit_rate_at_5']}`
- valid_only_NDCG@5: `{valid_only['ndcg_at_5']}`

### Notes And Limitations

- No new LLM inference was run.
- No Ollama command, Ollama HTTP API, OpenAI API, Anthropic API, remote API, download, model pull, recommender training, or LastFM experiment occurred.
- No Plan C output, recommender metric CSV, D5E source metric CSV, recommender raw output, or raw LLM output was modified.
- Ignored raw JSONL status: {raw_note}.
- Qwen2 is reliable at response/parse/schema level in D5E but not fully reliable at candidate-adherence level.
"""
    _replace_or_append_section(NOTES_PATH, "## 2026-07-01 - Plan D D5E.1: Qwen2 Output Validity and Hallucination Analysis", section)


def _update_reproducibility_matrix() -> None:
    if not REPRO_MATRIX_PATH.exists():
        return
    breakdown = _breakdown_row()
    utility = _utility_frame()
    valid_only = _utility_row(utility, "valid_only_output")
    text = REPRO_MATRIX_PATH.read_text(encoding="utf-8")
    intro = (
        "D5B added an LLM model registry and backend feasibility audit before any LLM experiment. "
        "D5C0 verified Ollama and `qwen2:7b-instruct`; D5C1 ran exactly one successful toy local smoke inference. "
        "D5D completed reusable prompt templates, strict parser utilities, and a no-inference ML-1M manifest with 20 ready rows. "
        "D5E ran the bounded 20-user local Qwen2/Ollama ML-1M ranking diagnostic. "
        f"D5E.1 analyzed the outputs without new inference and found perfect response/parse/schema rates but hallucination_rate `{breakdown['hallucination_rate']}`; candidate adherence is the main Qwen2 failure mode. "
        "This remains diagnostic_subset work, not exact paper reproduction, and closed-source API reproduction remains conditional on explicit approval, keys, availability, and budget."
    )
    text = _replace_paragraph_starting(text, "D5B added an LLM model registry", intro)
    text = _replace_table_row(
        text,
        "qwen-2-7b-instruct",
        f"| qwen-2-7b-instruct | `diagnostic_completed_validity_analyzed` | `local_open_model_near_exact` | D5E ran the local Qwen2 ranking diagnostic; D5E.1 found parse/schema rates `{breakdown['parse_success_rate']}`/`{breakdown['schema_valid_rate']}` and hallucination_rate `{breakdown['hallucination_rate']}`. |",
    )
    text = _replace_table_row(
        text,
        "Candidate validity",
        f"| Candidate validity | `EXTENSION` | `extension` | D5E.1 measured mean candidate validity `{breakdown['mean_candidate_validity_rate']}` and micro candidate validity `{breakdown['micro_candidate_validity_rate']}` for Qwen2. |",
    )
    text = _replace_table_row(
        text,
        "Hallucination",
        f"| Hallucination | `PARTIAL` | `extension` | D5E.1 confirmed hallucination_rate `{breakdown['hallucination_rate']}`; Qwen2 candidate adherence is the main failure mode. |",
    )
    text = _replace_table_row(
        text,
        "Output validity and parsing robustness",
        f"| Output validity and parsing robustness | `PARTIAL` | `extension` | D5E.1 found response/parse/schema/exact-top-k rates all `{breakdown['parse_success_rate']}` but mean candidate validity `{breakdown['mean_candidate_validity_rate']}`. |",
    )
    text = _replace_table_row(
        text,
        "Qwen ML-1M ranking diagnostic",
        f"| Qwen ML-1M ranking diagnostic | `completed_diagnostic_subset_validity_analyzed` | D5E HR@5 `0.150000`, NDCG@5 `0.100889`; D5E.1 valid-only HR@5 `{valid_only['hit_rate_at_5']}`, NDCG@5 `{valid_only['ndcg_at_5']}`. |",
    )
    recommended = """## Recommended next steps

1. Run candidate position-bias diagnostics.
2. Test a stricter candidate-ID-only prompt.
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


def _breakdown_row() -> pd.Series:
    return pd.read_csv(BREAKDOWN_PATH, dtype=str).fillna("").iloc[0]


def _utility_frame() -> pd.DataFrame:
    return pd.read_csv(VALID_ONLY_UTILITY_PATH, dtype=str).fillna("")


def _profile_frame() -> pd.DataFrame:
    return pd.read_csv(HALLUCINATION_PROFILE_PATH, dtype=str).fillna("")


def _patterns_frame() -> pd.DataFrame:
    return pd.read_csv(INVALID_ITEM_PATTERNS_PATH, dtype=str).fillna("")


def _utility_row(frame: pd.DataFrame, utility_variant: str) -> pd.Series:
    rows = frame.loc[frame["utility_variant"] == utility_variant]
    if rows.empty:
        raise ValueError(f"Missing utility variant: {utility_variant}")
    return rows.iloc[0]


def _top_patterns_text(patterns: pd.DataFrame) -> str:
    if patterns.empty:
        return "none"
    top = patterns.head(5)
    return "; ".join(
        f"{row['invalid_item_id']} ({row['frequency']}, {row['pattern_type']})"
        for _, row in top.iterrows()
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze D5E Qwen2 output validity and hallucination behavior without new inference."
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    run_analysis()
    LOGGER.info("Wrote %s", BREAKDOWN_PATH)
    LOGGER.info("Wrote %s", VALID_ONLY_UTILITY_PATH)
    LOGGER.info("Wrote %s", HALLUCINATION_PROFILE_PATH)
    LOGGER.info("Wrote %s", INVALID_ITEM_PATTERNS_PATH)
    LOGGER.info("Wrote %s", REPORT_PATH)


if __name__ == "__main__":
    main()
