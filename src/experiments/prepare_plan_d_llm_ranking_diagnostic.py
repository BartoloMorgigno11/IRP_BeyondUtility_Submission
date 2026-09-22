from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.plan_d_llm_ranking_manifest import (
    MANIFEST_COLUMNS,
    ManifestSummary,
    build_ml1m_qwen_manifest,
)
from src.llm.plan_d_prompt_templates import prompt_template_registry_rows


LOGGER = logging.getLogger(__name__)

RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORT_DIR = PROJECT_ROOT / "report"
MANIFEST_PATH = RESULTS_TABLES_DIR / "plan_d_qwen2_ml1m_ranking_diagnostic_manifest.csv"
PROMPT_REGISTRY_PATH = RESULTS_TABLES_DIR / "plan_d_llm_prompt_template_registry.csv"
REPORT_PATH = REPORT_DIR / "plan_d_llm_prompt_templates_and_ranking_manifest.md"
TASK_ROADMAP_PATH = RESULTS_TABLES_DIR / "plan_d_llm_task_roadmap.csv"
NEXT_ACTIONS_PATH = RESULTS_TABLES_DIR / "plan_d_next_actions_priority.csv"
FEASIBILITY_LEDGER_PATH = RESULTS_TABLES_DIR / "plan_d_feasibility_ledger.csv"
NOTES_PATH = REPORT_DIR / "notes.md"
REPRO_MATRIX_PATH = REPORT_DIR / "plan_d_reproducibility_matrix.md"


def prepare_outputs() -> ManifestSummary:
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prompt_template_registry_rows()).to_csv(PROMPT_REGISTRY_PATH, index=False)
    summary = build_ml1m_qwen_manifest(
        data_root=PROJECT_ROOT / "data" / "reconstructed" / "plan_c",
        output_path=MANIFEST_PATH,
    )
    _write_report(summary)
    _update_task_roadmap(summary)
    _update_next_actions()
    _update_feasibility_ledger(summary)
    _update_notes(summary)
    _update_reproducibility_matrix(summary)
    return summary


def _write_report(summary: ManifestSummary) -> None:
    manifest = _manifest_frame()
    positive_rows = int(manifest["positive_in_candidates"].astype(str).str.lower().eq("true").sum()) if not manifest.empty else 0
    candidate_counts = sorted(manifest["candidate_count"].astype(int).unique().tolist()) if not manifest.empty else []
    text = f"""# Plan D LLM Prompt Templates and ML-1M Ranking Manifest

## Purpose

D5D prepares reusable LLM prompt templates, strict parser utilities, and a deterministic ML-1M 20-user diagnostic manifest for the future Qwen2 local ranking run.

No LLM inference was run in this phase. No external API was called. No Ollama HTTP request, OpenAI API call, Anthropic API call, download, model pull, recommender training, LastFM experiment, Plan C modification, metric CSV update, or raw-output generation occurred.

## Prompt Strategies Implemented Or Documented

- `openp5_base`: implemented first diagnostic template. It uses user history, the candidate list, and item metadata, and asks the model to rank only candidate IDs.
- `llmrank_recency`: implemented/documented recency-focused variant inspired by LLMRank, using recent history first.
- `llmsrec_icl_stub`: documented ICL placeholder for future work; not used in D5E unless examples are implemented.

The registry is stored in `results/tables/plan_d_llm_prompt_template_registry.csv`.

## Strict JSON Output Schema

```json
{{
  "ranked_item_ids": ["item_id_1", "item_id_2", "item_id_3", "item_id_4", "item_id_5"],
  "status": "ok"
}}
```

Prompts instruct the model to return only valid JSON, rank exactly `top_k` item IDs, use only candidate IDs, avoid invented item IDs, avoid explanations, and choose from candidates even when uncertain.

## Parser Validation Rules

The parser accepts JSON objects only. It requires `ranked_item_ids` to be a list, records selected IDs exactly as returned, detects invalid/hallucinated IDs, detects duplicates, records length mismatches, and computes `candidate_validity_rate = valid returned IDs / selected_count` with `0.0` for empty selections.

Parser status values are `parsed_valid`, `parsed_invalid_schema`, `parsed_with_hallucinations`, `parsed_with_duplicates`, `parse_failed`, and `empty_response`.

## ML-1M 20-User Diagnostic Manifest

- Manifest path: `results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_manifest.csv`
- Dataset: `ml1m`
- Phase: `D5E_qwen2_ml1m_ranking_diagnostic`
- Model: `qwen2:7b-instruct`
- Backend: `ollama`
- Prompt strategy: `openp5_base`
- Top-k: `5`
- Candidate pool size: `20`
- Manifest rows: `{summary.row_count}`
- Ready rows: `{summary.ready_rows}`
- Rows with positives in candidates: `{positive_rows}`
- Candidate-count values present: `{candidate_counts}`
- Manifest summary: `{summary.notes}`

## Candidate Pool Setup

The manifest reads existing reconstructed Plan C ML-1M artifacts from `data/reconstructed/plan_c/ml1m`: `interactions.csv`, `candidates.csv`, `ground_truth.csv`, and `catalog.csv`. Candidate sets are not invented. The reconstructed ML-1M candidate artifact contains 100 candidates per user; D5D deterministically prepares a 20-candidate diagnostic pool by retaining the positive target and selecting the earliest reconstructed negative candidates by candidate position. Ready rows require non-empty training history, exactly 20 selected candidates, exactly one positive candidate according to the reconstructed candidate setup, and a positive target present in the selected candidate list.

## Limitations

- This is a no-inference preparation manifest, not an LLM output file.
- The manifest uses reconstructed Plan C candidate pools, not official paper candidate pools.
- Full prompts are not saved in the manifest; only prompt character lengths and short previews are stored.
- No LLM ranking reproduction is claimed yet.
- LastFM is not part of this phase.

## Recommended Next Step

Run D5E: Qwen2 ML-1M ranking diagnostic, 20 users, local Ollama backend, `temperature=0`, `prompt_strategy=openp5_base`, and strict JSON parsing.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def _update_task_roadmap(summary: ManifestSummary) -> None:
    columns = [
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
    frame = _read_or_empty(TASK_ROADMAP_PATH, columns)
    rows = []
    d5d_present = False
    for _, row in frame.iterrows():
        row = row.to_dict()
        if row["phase"] == "D5D":
            row["task"] = "llm_prompt_templates_parser_manifest"
            row["dataset"] = "ml1m"
            row["model"] = "qwen2:7b-instruct"
            row["backend"] = "none"
            row["users"] = "20"
            row["candidate_pool_size"] = "20"
            row["prompt_strategy"] = "openp5_base"
            row["output_format"] = "strict_json"
            row["metrics"] = "manifest_ready_rows,parser_test_coverage"
            row["prerequisites"] = "D5C passed"
            row["status"] = "completed_no_inference"
            row["notes"] = f"Prepared prompt templates, strict parser, and {summary.ready_rows} ready ML-1M manifest rows; no LLM inference was run."
            d5d_present = True
        elif row["phase"] == "D5E":
            row["task"] = "llm_ranking_diagnostic"
            row["dataset"] = "ml1m"
            row["model"] = "qwen2:7b-instruct"
            row["backend"] = "ollama"
            row["users"] = "20"
            row["candidate_pool_size"] = "20"
            row["prompt_strategy"] = "openp5_base"
            row["output_format"] = "strict_json"
            row["metrics"] = "HR@5,NDCG@5,candidate_validity_rate,duplicate_rate,hallucination_rate,parse_failure_rate,runtime"
            row["prerequisites"] = "D5D manifest ready"
            row["status"] = "pending_ready" if summary.ready_rows else "blocked_manifest_empty"
            row["notes"] = "Ready for future local Qwen inference; no LLM ranking has been run yet."
        rows.append(row)
    if not d5d_present:
        rows.append(
            {
                "phase": "D5D",
                "task": "llm_prompt_templates_parser_manifest",
                "dataset": "ml1m",
                "model": "qwen2:7b-instruct",
                "backend": "none",
                "users": "20",
                "candidate_pool_size": "20",
                "prompt_strategy": "openp5_base",
                "output_format": "strict_json",
                "metrics": "manifest_ready_rows,parser_test_coverage",
                "prerequisites": "D5C passed",
                "status": "completed_no_inference",
                "notes": f"Prepared prompt templates, strict parser, and {summary.ready_rows} ready ML-1M manifest rows; no LLM inference was run.",
            }
        )
    pd.DataFrame(rows, columns=columns).to_csv(TASK_ROADMAP_PATH, index=False)


def _update_next_actions() -> None:
    columns = [
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
    rows = [
        _next_action("1", "run_qwen2_ml1m_ranking_diagnostic_20_users", "Run Qwen ML-1M ranking diagnostic", "llm_ranking", "This is the first real local LLM-as-recommender diagnostic after D5D preparation.", "20-user ML-1M Qwen ranking outputs and metrics.", "D5D manifest ready; explicit approval to run local inference.", "high", "next", "Use local Ollama, qwen2:7b-instruct, temperature 0, and strict JSON parsing."),
        _next_action("2", "inspect_qwen2_output_validity_and_hallucination", "Inspect Qwen output validity and hallucination", "hallucination; output_validity", "The project must validate parser success, candidate validity, duplicates, and hallucinations before broader analysis.", "Output-validity and hallucination summary from real Qwen outputs.", "D5E outputs.", "medium", "after D5E", "No manual correction of LLM outputs."),
        _next_action("3", "run_candidate_position_bias_diagnostic", "Run candidate position-bias diagnostic", "position_bias", "Position bias is meaningful only with real LLM selections.", "Position-bias metrics from real parsed LLM outputs.", "D5E outputs.", "medium", "after output-validity inspection", "Use generated CSV outputs only."),
        _next_action("4", "pull_mistral_7b_instruct_v02_quantized", "Pull Mistral 7B instruct v0.2 quantized", "paper_llm_mistral_7b_instruct_v0_2", "Mistral is the second local model for output-validity comparison.", "A recommended Mistral v0.2 quantized model appears in ollama list.", "Manual user pull only.", "medium", "after Qwen diagnostic", "Do not pull automatically."),
        _next_action("5", "run_mistral_ml1m_ranking_diagnostic_20_users", "Run Mistral ML-1M ranking diagnostic", "llm_ranking; paper_llm_mistral_7b_instruct_v0_2", "A second local model helps compare validity and hallucination behavior.", "20-user Mistral ML-1M ranking diagnostic metrics.", "Mistral pulled and Qwen path stable.", "high", "after Mistral pull", "Keep as diagnostic; no exact paper reproduction claim."),
        _next_action("6", "run_llm_reranking_diagnostic", "Run LLM reranking diagnostic", "llm_reranking", "Reranking is closer to the paper task but needs stable candidate-pool policy.", "Reranking diagnostics with HR@5, NDCG@5, validity, and hallucination metrics.", "D5E outputs and candidate-pool policy.", "high", "after ranking diagnostics", "Keep as diagnostic_subset unless official pools are recovered."),
        _next_action("7", "integrate_llm_results_or_blocker_into_final_report", "Integrate LLM results or blocker into final report", "final_report; local_llm_backend", "The final report must close the LLM loop with real results or documented blockers.", "Final LLM section grounded in generated CSVs and audit artifacts.", "D5E-D5H outputs or blocker.", "low", "after attempted LLM path", "Do not manually invent metric values."),
        _next_action("8", "prepare_exam_presentation_tables", "Prepare exam presentation tables", "final_report", "The oral exam needs compact tables for method coverage, rankings, and limitations.", "Presentation-ready generated tables.", "Stable final report data tables.", "low", "after final report draft", "Use generated CSV values."),
        _next_action("9", "optional_openai_api_diagnostic_if_budget_approved", "Optional OpenAI API diagnostic if budget approved", "api_availability", "Closed-source comparison is conditional and should remain optional.", "Small approved API diagnostic or documented no-API decision.", "Explicit user approval, key, and budget.", "high", "optional", "No API calls by default."),
        _next_action("10", "final_report", "Finalize report", "final_report", "The project needs final claims grounded in generated artifacts.", "Final report with limitations and generated metrics.", "LLM results or blocker integrated.", "low", "final", "Do not claim exact LLM reproduction unless supported."),
    ]
    pd.DataFrame(rows, columns=columns).to_csv(NEXT_ACTIONS_PATH, index=False)


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


def _update_feasibility_ledger(summary: ManifestSummary) -> None:
    frame = _read_or_empty(
        FEASIBILITY_LEDGER_PATH,
        [
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
        ],
    )
    _upsert_ledger(
        frame,
        {
            "item_id": "llm_prompt_templates_parser_manifest",
            "area": "llm_preparation",
            "target": "Prompt templates, strict parser, and ML-1M Qwen manifest",
            "feasibility_status": "already_done",
            "required_resources": "Existing reconstructed Plan C ML-1M interactions, candidates, ground truth, and catalog",
            "estimated_effort": "done",
            "risk_level": "low",
            "external_dependency": "no",
            "recommended_decision": "Use D5D artifacts to run D5E only with explicit local inference approval.",
            "notes": f"D5D completed_no_inference with {summary.ready_rows} ready manifest rows.",
        },
    )
    for item_id in ("local_llm_backend", "ollama_backend", "local_ollama_backend", "paper_llm_qwen_2_7b_instruct"):
        mask = frame["item_id"] == item_id
        if mask.any():
            frame.loc[mask, "recommended_decision"] = "Ready for D5E ranking diagnostic after D5D no-inference preparation."
            frame.loc[mask, "notes"] = (
                "D5D prompt templates/parser/manifest completed_no_inference; "
                "qwen2 local backend remains ready_for_ranking_diagnostic."
            )
    frame.to_csv(FEASIBILITY_LEDGER_PATH, index=False)


def _upsert_ledger(frame: pd.DataFrame, row: dict[str, str]) -> None:
    mask = frame["item_id"] == row["item_id"]
    if mask.any():
        for column, value in row.items():
            frame.loc[mask, column] = value
    else:
        frame.loc[len(frame)] = row


def _update_notes(summary: ManifestSummary) -> None:
    section = f"""
## 2026-07-01 - Plan D D5D: Prompt Templates, Parser, and ML-1M Manifest

### Changes

- Added reusable LLM ranking prompt templates for `openp5_base`,
  `llmrank_recency`, and `llmsrec_icl_stub`.
- Added strict JSON parser utilities for future LLM ranking outputs.
- Prepared a deterministic no-inference ML-1M Qwen2 ranking manifest.
- Wrote the prompt-template registry and D5D report.

### Commands Run

```powershell
python src/experiments/prepare_plan_d_llm_ranking_diagnostic.py
```

### Outputs Generated

- `results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_manifest.csv`
- `results/tables/plan_d_llm_prompt_template_registry.csv`
- `report/plan_d_llm_prompt_templates_and_ranking_manifest.md`

### Result

- Manifest rows: `{summary.row_count}`
- Ready rows: `{summary.ready_rows}`
- Rows with positives in candidates: `{summary.positives_present_rows}`
- Status: `completed_no_inference`

### Notes And Limitations

- No LLM inference was run.
- No Ollama HTTP API, OpenAI API, Anthropic API, or external API call was made.
- No model was pulled or downloaded.
- No recommender training, dataset ranking, or LastFM experiment was run.
- No Plan C output, source metric CSV, recommender raw output, or LLM output was modified or created.
- No LLM ranking reproduction is claimed yet.
"""
    _replace_or_append_section(NOTES_PATH, "## 2026-07-01 - Plan D D5D: Prompt Templates, Parser, and ML-1M Manifest", section)


def _update_reproducibility_matrix(summary: ManifestSummary) -> None:
    if not REPRO_MATRIX_PATH.exists():
        return
    text = REPRO_MATRIX_PATH.read_text(encoding="utf-8")
    intro = (
        "D5B added an LLM model registry and backend feasibility audit before any LLM experiment. "
        "D5C0 verified Ollama and `qwen2:7b-instruct`; D5C1 ran exactly one successful toy local smoke inference. "
        f"D5D completed reusable prompt templates, strict parser utilities, and a no-inference ML-1M manifest with {summary.ready_rows} ready rows. "
        "No LLM ranking inference was run, no LLM recommendation metrics are claimed, and closed-source API reproduction remains conditional on explicit approval, keys, availability, and budget."
    )
    text = _replace_paragraph_starting(text, "D5B added an LLM model registry", intro)
    text = _replace_table_row(
        text,
        "qwen-2-7b-instruct",
        "| qwen-2-7b-instruct | `ready_for_ranking_diagnostic` | `local_open_model_near_exact` | D5D prepared prompt templates, parser utilities, and a no-inference ML-1M manifest; no ranking result is claimed. |",
    )
    text = _replace_table_row(
        text,
        "D5C Qwen smoke test",
        "| D5C Qwen smoke test | `smoke_test_passed` | Exactly one toy local smoke inference was run; no dataset ranking experiment was run. |",
    )
    text = _replace_table_row(
        text,
        "LLM prompt templates and parser",
        f"| LLM prompt templates and parser | `completed_no_inference` | D5D prepared templates, parser utilities, and {summary.ready_rows} ML-1M manifest rows without LLM inference. |",
    )
    text = _replace_table_row(
        text,
        "Qwen ML-1M ranking diagnostic",
        "| Qwen ML-1M ranking diagnostic | `pending_ready` | D5E is ready to run with local Ollama/Qwen, 20 ML-1M users, temperature 0, and strict JSON parsing. |",
    )
    text = _replace_table_row(
        text,
        "Local LLM diagnostic ranking",
        "| Local LLM diagnostic ranking | `pending_ready` | D5D manifest is ready, but no dataset ranking inference has run yet. |",
    )
    recommended = """## Recommended next steps

1. Run the Qwen ML-1M ranking diagnostic for 20 users.
2. Inspect Qwen output validity and hallucination.
3. Run candidate position-bias diagnostics.
4. Pull a quantized Mistral 7B Instruct v0.2 model manually.
5. Run the Mistral ML-1M ranking diagnostic for 20 users.
6. Run LLM reranking diagnostics.
7. Integrate LLM results or the backend blocker into the final report.
8. Prepare exam presentation tables.
9. Optionally run an OpenAI API diagnostic if budget is explicitly approved.
10. Finalize the report.
"""
    text = _replace_section_text(text, "## Recommended next steps", recommended)
    REPRO_MATRIX_PATH.write_text(text, encoding="utf-8")


def _manifest_frame() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    return pd.read_csv(MANIFEST_PATH, dtype=str).fillna("")


def _read_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype=str).fillna("")
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame[columns]


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
        description="Prepare D5D LLM prompt templates, parser registry, and ML-1M manifest without inference."
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    summary = prepare_outputs()
    LOGGER.info("Wrote %s", PROMPT_REGISTRY_PATH)
    LOGGER.info("Wrote %s", MANIFEST_PATH)
    LOGGER.info("Wrote %s", REPORT_PATH)
    LOGGER.info("Manifest rows=%s ready_rows=%s", summary.row_count, summary.ready_rows)


if __name__ == "__main__":
    main()
