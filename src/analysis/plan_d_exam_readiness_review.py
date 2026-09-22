from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "report"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

FINAL_REPORT_PATH = REPORT_DIR / "plan_d_final_reproducibility_package.md"
MATRIX_PATH = REPORT_DIR / "plan_d_reproducibility_matrix.md"
NOTES_PATH = REPORT_DIR / "notes.md"
FINAL_EXPERIMENT_INDEX_PATH = TABLES_DIR / "plan_d_final_experiment_index.csv"
FINAL_METRIC_SUMMARY_PATH = TABLES_DIR / "plan_d_final_metric_summary.csv"
FINAL_LLM_SUMMARY_PATH = TABLES_DIR / "plan_d_final_llm_diagnostic_summary.csv"
FINAL_REPRO_STATUS_PATH = TABLES_DIR / "plan_d_final_reproducibility_status.csv"
FINAL_BLOCKED_ITEMS_PATH = TABLES_DIR / "plan_d_final_blocked_items.csv"
HYGIENE_AUDIT_PATH = TABLES_DIR / "plan_d_repository_hygiene_audit.csv"

CHECKLIST_COLUMNS = ["check_id", "area", "status", "finding", "recommendation"]
CLAIM_AUDIT_COLUMNS = ["claim_area", "risky_claim", "safe_rewrite", "status"]
ORAL_POINTS_COLUMNS = ["topic", "concise_answer", "evidence_file", "classification"]


def build_exam_readiness_checklist() -> pd.DataFrame:
    final_report = _read_text(FINAL_REPORT_PATH).lower()
    matrix = _read_text(MATRIX_PATH).lower()
    llm_summary = _read_csv(FINAL_LLM_SUMMARY_PATH)
    blocked = _read_csv(FINAL_BLOCKED_ITEMS_PATH)
    hygiene = _read_csv(HYGIENE_AUDIT_PATH)

    rows = [
        _check("C01", "final report states no exact LLM reproduction", _contains(final_report, "not full llm reproduction") and _contains(final_report, "not exact reproduction"), "The final package explicitly rejects exact/full LLM reproduction.", "Keep this wording in oral defense."),
        _check("C02", "final report states no full LLM reproduction", _contains(final_report, "this is not full llm reproduction"), "The full LLM reproduction boundary is explicit.", "Do not describe D5E-D5K as the paper's full LLM matrix."),
        _check("C03", "Plan C classified as reconstructed reproduction", _contains(final_report, "plan c is reconstructed reproduction") and _contains(matrix, "reconstructed reproduction"), "Plan C is consistently framed as reconstructed reproduction.", "Use the tag as stability evidence, not exact-paper evidence."),
        _check("C04", "D5E-D5K classified as diagnostic_subset", _contains(final_report, "diagnostic") and _phase_set_present(llm_summary, {"D5E", "D5G", "D5I", "D5J", "D5K"}), "The LLM diagnostics are represented as diagnostic subset work.", "Mention 20 ML-1M users when discussing LLM findings."),
        _check("C05", "blocked/conditional items are explicit", _blocked_items_present(blocked), "Blocked and conditional items are documented.", "Lead with blockers before discussing optional future API work."),
        _check("C06", "strict prompt mitigation result is clearly explained", _contains(final_report, "strict candidate-id-only prompting improved grounding"), "Strict prompt mitigation is stated in the final package.", "Say it improved this diagnostic, not that it solves hallucination generally."),
        _check("C07", "hallucination/candidate validity distinction is clear", _contains(final_report, "candidate validity") and _contains(final_report, "hallucination"), "The final package separates validity from utility.", "Explain that valid JSON can still contain invalid candidate IDs."),
        _check("C08", "position bias and history-length sensitivity are discussed", _contains(final_report, "candidate order") and _contains(final_report, "history length"), "Order and history-length robustness are included in the story.", "Frame these as robustness diagnostics, not full causal claims."),
        _check("C09", "Qwen vs Mistral comparison is interpreted correctly", _contains(final_report, "qwen2 and mistral behaved differently"), "The model comparison is local and bounded.", "Avoid generalizing to the full benchmark or all Mistral/Qwen variants."),
        _check("C10", "re-ranking diagnostic is interpreted without overclaiming", _contains(final_report, "re-ranking was not automatically beneficial"), "The report avoids claiming LLM reranking is generally worse or better.", "Say it did not help this candidate-order reference in this subset."),
        _check("C11", "raw outputs are said to be Git-ignored", _contains(final_report, "git-ignored") and _hygiene_passed(hygiene, "raw_outputs_gitignored"), "Raw JSONL handling is clearly documented.", "Before any commit, verify raw JSONL remains ignored."),
        _check("C12", "no API/no-download/no-training boundaries are stated", _boundary_statements_present(final_report), "No-inference/no-API/no-download/no-training boundaries are explicit.", "Keep the same boundary paragraph in final slides."),
        _check("C13", "D5H Mistral status wording consistency", "blocked_model_not_available" not in matrix, "The matrix still contains an old blocked_model_not_available phrase for D5H." if "blocked_model_not_available" in matrix else "D5H wording is consistent: initially conditional, final local Mistral smoke test passed after manual user installation.", "Keep D5H framed as diagnostic_subset local model availability/smoke test, not exact reproduction."),
    ]
    return pd.DataFrame(rows, columns=CHECKLIST_COLUMNS)


def build_claim_safety_audit() -> pd.DataFrame:
    rows = [
        _claim("LLM reproduction scope", "we reproduced the full LLM part", "We ran bounded local diagnostic_subset LLM studies on ML-1M; full LLM reproduction is not completed.", "unsafe_rewrite_required"),
        _claim("Overall reproduction scope", "exact reproduction of Beyond Utility", "This is reconstructed reproduction plus diagnostic/extended local analysis; exact paper reproduction is blocked by missing artifacts and conditional LLMs.", "unsafe_rewrite_required"),
        _claim("Generalization", "Mistral/Qwen results generalize to the full benchmark", "Qwen2 and Mistral results apply only to the local ML-1M 20-user diagnostic subset and prompt settings tested.", "unsafe_rewrite_required"),
        _claim("Model comparison", "LLMs outperform neural recommenders", "The project compares local LLM diagnostics and reconstructed neural baselines in separate scopes; it does not establish broad LLM superiority.", "unsafe_rewrite_required"),
        _claim("Hallucination", "hallucination is solved", "Strict candidate-ID-only prompts eliminated hallucination in the tested strict-prompt diagnostics, but broader hallucination remains a risk.", "unsafe_rewrite_required"),
        _claim("LastFM", "LastFM was reproduced", "LastFM remains extension_only_no_valid_utility_evaluation because valid comparable utility evaluation is unavailable.", "unsafe_rewrite_required"),
        _claim("Closed-source models", "closed-source LLMs were reproduced", "Closed-source LLM reproduction remains conditional/API-dependent and was not run.", "unsafe_rewrite_required"),
        _claim("Plan D", "Plan D is exact reproduction", "Plan D contains reconstructed baselines, extended_subset neural results, and diagnostic_subset local LLM diagnostics; it is not exact reproduction.", "unsafe_rewrite_required"),
        _claim("D5H status", "Mistral stayed blocked", "Mistral was initially conditional, then smoke-test completed after the user manually installed the local model.", "resolved_safe_rewrite"),
    ]
    return pd.DataFrame(rows, columns=CLAIM_AUDIT_COLUMNS)


def build_oral_defense_key_points() -> pd.DataFrame:
    rows = [
        _point("What exactly was reproduced?", "The original code was preserved, and Plan C/Plan D reconstruct the evaluation around available artifacts; no exact paper-number reproduction is claimed.", "report/plan_d_reproducibility_matrix.md", "reconstructed reproduction"),
        _point("What was reconstructed?", "Datasets, candidate evaluation, utility metrics, and neural baselines were reconstructed or extended using local processed artifacts.", "results/tables/plan_d_final_experiment_index.csv", "reconstructed reproduction"),
        _point("What is only diagnostic_subset?", "D5E-D5K local LLM work uses a 20-user ML-1M diagnostic subset and should not be described as full paper LLM reproduction.", "results/tables/plan_d_final_llm_diagnostic_summary.csv", "diagnostic_subset"),
        _point("Why is the LLM part not exact reproduction?", "It does not run all seven paper LLMs, closed-source APIs, full user counts, or repeated paper experiments; some exact models are retired or hardware/API blocked.", "results/tables/plan_d_final_blocked_items.csv", "blocked/conditional"),
        _point("Why are hallucination and candidate validity important?", "D5E showed valid JSON can still contain non-candidate IDs, so HR/NDCG alone miss reliability failures.", "results/tables/plan_d_final_llm_diagnostic_summary.csv", "beyond-utility reliability"),
        _point("What did strict prompting show?", "D5G strict candidate-ID-only prompting improved Qwen2 utility and candidate validity in the diagnostic subset.", "results/tables/plan_d_final_llm_diagnostic_summary.csv", "diagnostic_subset"),
        _point("What did candidate position bias show?", "D5F showed that reordering the same candidates changed utility and reliability, so candidate order is a robustness factor.", "report/plan_d_qwen2_ml1m_candidate_position_bias_diagnostic.md", "diagnostic_subset"),
        _point("What did Qwen vs Mistral show?", "D5I showed Qwen2 strict had higher utility than local Mistral strict in this subset while both achieved perfect candidate validity.", "results/tables/plan_d_final_llm_diagnostic_summary.csv", "diagnostic_subset"),
        _point("What did re-ranking show?", "D5J showed local LLM re-ranking did not improve the candidate-order reference and reliability differed by model.", "report/plan_d_llm_ml1m_reranking_diagnostic.md", "diagnostic_subset"),
        _point("What did history length sensitivity show?", "D5K showed history window length changed utility and output stability even under strict prompts with zero hallucination.", "report/plan_d_llm_ml1m_history_length_sensitivity.md", "diagnostic_subset"),
        _point("What is blocked and why?", "Claude 3 Haiku exact is retired, Llama3 70B is blocked locally by hardware, closed APIs need approval/budget, and LastFM lacks valid utility setup.", "results/tables/plan_d_final_blocked_items.csv", "blocked/conditional"),
        _point("What is the main contribution of the project?", "It turns an incomplete exact reproduction into a transparent reconstructed reproduction plus reliability-focused local LLM diagnostics.", "report/plan_d_final_reproducibility_package.md", "aggregation/reporting"),
        _point("What would be Plan E?", "A separate approved phase could recover official artifacts, broaden user counts, add constrained decoding, or run small closed-source API diagnostics with explicit budget.", "results/tables/plan_d_final_reproducibility_status.csv", "future work"),
    ]
    return pd.DataFrame(rows, columns=ORAL_POINTS_COLUMNS)


def build_exam_readiness_report(
    checklist: pd.DataFrame,
    claim_audit: pd.DataFrame,
    oral_points: pd.DataFrame,
) -> str:
    failed_or_review = checklist.loc[~checklist["status"].isin(["pass", "ready"])]
    review_note = (
        "One wording issue should be reviewed before final submission: stale D5H blocked wording appears in the matrix."
        if not failed_or_review.empty
        else "No blocking exam-readiness issue was found in the D5M checklist."
    )
    return f"""# Plan D Exam Readiness Review

## Executive Summary

D5M is a final reporting audit and exam-readiness review. It did not run any new experiment. The Plan D package is exam-ready in its core scientific positioning: it clearly supports a reconstructed reproduction plus local diagnostic/extended subset analysis, while avoiding an exact or full LLM reproduction claim. {review_note}

## Safe Final Positioning

This is not exact reproduction of the full paper. It is reconstructed reproduction plus local diagnostic/extended subset analysis. Plan C remains reconstructed reproduction. Plan D adds extended neural baselines and diagnostic_subset local LLM studies that test output validity, hallucination, prompt mitigation, candidate order sensitivity, model differences, re-ranking, and history length sensitivity. Closed-source LLM replication is conditional/API-dependent, Claude 3 Haiku exact is blocked/retired, Llama3 70B local is blocked by hardware, and LastFM is extension_only_no_valid_utility_evaluation.

## What Not To Claim

- Do not claim full LLM reproduction.
- Do not claim exact reproduction of Beyond Utility.
- Do not claim Mistral/Qwen results generalize to the full benchmark.
- Do not claim LLMs outperform neural recommenders globally.
- Do not claim hallucination is solved.
- Do not claim LastFM was reproduced.
- Do not claim closed-source LLMs were reproduced.
- Do not claim Plan D is exact reproduction.

## What To Emphasize

- The project is transparent about missing official artifacts and exact-model blockers.
- The final package separates exact, reconstructed, diagnostic_subset, extended_subset, extension, and blocked/conditional categories.
- D5E showed syntactic validity is not enough because candidate hallucination remained high.
- D5G showed strict candidate-ID-only prompting can improve grounding in the tested local diagnostic setting.
- D5F, D5J, and D5K show that order, re-ranking framing, and history window length affect local LLM behavior.
- D5I shows Qwen2 and Mistral behave differently under the same local strict-prompt setup.
- These findings support the Beyond Utility argument: reliability metrics matter alongside HR/NDCG.

## Remaining Limitations

- No full paper LLM matrix was reproduced.
- No OpenAI or Anthropic API diagnostics were run.
- Claude 3 Haiku exact reproduction is blocked/retired.
- Llama3 70B local execution is blocked by hardware.
- LastFM remains extension_only_no_valid_utility_evaluation.
- D5E-D5K are ML-1M diagnostic_subset studies, not full benchmark results.
- D5H should be described as initially conditional, then completed as a local Mistral smoke test after manual user installation.

## Execution Boundaries

- No new inference/API/download/training was run in D5M.
- No Ollama call was made.
- No Qwen2 call was made.
- No Mistral call was made.
- No OpenAI API call was made.
- No Anthropic API call was made.
- No external API call was made.
- No download was performed.
- No model pull was performed.
- No recommender training was performed.
- No LastFM run was performed.
- No destructive cleanup was performed.

## Recommended Next Step

Use `results/tables/plan_d_oral_defense_key_points.csv` as the exam preparation sheet, and keep `results/tables/plan_d_claim_safety_audit.csv` open while drafting slides or the final oral script.

## Generated Review Artifacts

- Checklist rows: `{len(checklist)}`
- Claim safety rows: `{len(claim_audit)}`
- Oral defense key-point rows: `{len(oral_points)}`
"""


def _check(check_id: str, area: str, passed: bool, finding: str, recommendation: str) -> dict[str, str]:
    return {
        "check_id": check_id,
        "area": area,
        "status": "pass" if passed else "needs_review",
        "finding": finding if passed else finding,
        "recommendation": recommendation,
    }


def _claim(claim_area: str, risky_claim: str, safe_rewrite: str, status: str) -> dict[str, str]:
    return {
        "claim_area": claim_area,
        "risky_claim": risky_claim,
        "safe_rewrite": safe_rewrite,
        "status": status,
    }


def _point(topic: str, concise_answer: str, evidence_file: str, classification: str) -> dict[str, str]:
    return {
        "topic": topic,
        "concise_answer": concise_answer,
        "evidence_file": evidence_file,
        "classification": classification,
    }


def _contains(text: str, phrase: str) -> bool:
    return phrase.lower() in text


def _boundary_statements_present(text: str) -> bool:
    required = [
        "no openai api call was made",
        "no anthropic api call was made",
        "no external api call was made",
        "no download was performed",
        "no recommender training was performed",
    ]
    return all(phrase in text for phrase in required)


def _phase_set_present(frame: pd.DataFrame, phases: set[str]) -> bool:
    return not frame.empty and phases.issubset(set(frame["diagnostic_phase"]))


def _blocked_items_present(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    text = " ".join(frame["item"].tolist() + frame["current_status"].tolist()).lower()
    return all(token in text for token in ("claude", "llama3", "closed-source", "lastfm"))


def _hygiene_passed(frame: pd.DataFrame, check_name: str) -> bool:
    if frame.empty:
        return False
    rows = frame.loc[frame["check_name"] == check_name]
    return len(rows) == 1 and rows.iloc[0]["status"] == "passed"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")
