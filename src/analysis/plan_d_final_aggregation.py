from __future__ import annotations

from src.audit import common_git

import fnmatch
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PLAN = "Plan D"
PHASE = "D5L_final_aggregation_reproducibility_package"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
METRICS_DIR = RESULTS_DIR / "metrics"
RAW_OUTPUTS_DIR = RESULTS_DIR / "raw_outputs"
REPORT_DIR = PROJECT_ROOT / "report"

EXPERIMENT_INDEX_COLUMNS = [
    "phase_id",
    "title",
    "classification",
    "dataset",
    "model_or_method",
    "backend",
    "num_instances_or_scope",
    "new_inference_run",
    "new_training_run",
    "raw_output_path_if_any",
    "committed_artifact_type",
    "status",
    "report_path",
]

METRIC_SUMMARY_COLUMNS = [
    "source_phase",
    "category",
    "dataset",
    "model_or_method",
    "condition",
    "metric",
    "value",
    "source_path",
    "status",
    "notes",
]

LLM_DIAGNOSTIC_COLUMNS = [
    "diagnostic_phase",
    "model",
    "condition",
    "HR@5",
    "NDCG@5",
    "candidate_validity_rate",
    "hallucination_rate",
    "exact_top_k_rate",
    "duplicate_rate",
    "main_finding",
    "interpretation",
]

REPRO_STATUS_COLUMNS = [
    "area",
    "classification",
    "status",
    "evidence",
    "limitation",
    "allowed_claim",
]

BLOCKED_COLUMNS = [
    "item",
    "classification",
    "reason",
    "current_status",
    "possible_future_action",
    "whether_required_for_exam_project",
]

HYGIENE_COLUMNS = [
    "check_name",
    "status",
    "finding",
    "recommendation",
]


@dataclass(frozen=True)
class FinalPackagePaths:
    experiment_index: Path = TABLES_DIR / "plan_d_final_experiment_index.csv"
    metric_summary: Path = TABLES_DIR / "plan_d_final_metric_summary.csv"
    llm_diagnostic_summary: Path = TABLES_DIR / "plan_d_final_llm_diagnostic_summary.csv"
    reproducibility_status: Path = TABLES_DIR / "plan_d_final_reproducibility_status.csv"
    blocked_items: Path = TABLES_DIR / "plan_d_final_blocked_items.csv"
    hygiene_audit: Path = TABLES_DIR / "plan_d_repository_hygiene_audit.csv"
    report: Path = REPORT_DIR / "plan_d_final_reproducibility_package.md"


def build_experiment_index() -> pd.DataFrame:
    rows = [
        _experiment("D1", "Beyond-utility metrics", "extension", "ml1m; amazon_beauty; amazon_sports", "beyond_utility_metrics", "none", "generated metric tables", False, False, "", "metrics_csv", "completed", "report/notes.md"),
        _experiment("D2", "BPRMF diagnostic and extended baselines", "reconstructed; extended_subset", "ml1m; amazon_beauty; amazon_sports", "bprmf", "local_python", "diagnostic_subset and extended_subset", False, True, "results/raw_outputs/plan_d_*_bprmf_*_recommendations.csv", "metrics_csv; summary_csv", "completed", "report/plan_d_global_extended_neural_baselines.md"),
        _experiment("D3", "LightGCN diagnostic and extended baselines", "reconstructed; extended_subset", "ml1m; amazon_beauty; amazon_sports", "lightgcn", "local_python", "diagnostic_subset and extended_subset", False, True, "results/raw_outputs/plan_d_*_lightgcn_*_recommendations.csv", "metrics_csv; summary_csv", "completed", "report/plan_d_global_extended_neural_baselines.md"),
        _experiment("D4", "SASRec diagnostic and extended baselines", "reconstructed; extended_subset", "ml1m; amazon_beauty; amazon_sports", "sasrec", "local_python", "diagnostic_subset and extended_subset", False, True, "results/raw_outputs/plan_d_*_sasrec_*_recommendations.csv", "metrics_csv; summary_csv", "completed", "report/plan_d_global_extended_neural_baselines.md"),
        _experiment("D4.6", "GRU4Rec diagnostic and extended baselines", "reconstructed; extended_subset", "ml1m; amazon_beauty; amazon_sports", "gru4rec", "local_python", "diagnostic_subset and extended_subset", False, True, "results/raw_outputs/plan_d_*_gru4rec_*_recommendations.csv", "metrics_csv; summary_csv", "completed", "report/plan_d_global_extended_neural_baselines.md"),
        _experiment("D4.7", "Global extended neural baseline summary", "aggregation/reporting", "ml1m; amazon_beauty; amazon_sports; lastfm coverage note", "bprmf; lightgcn; sasrec; gru4rec", "none", "12 utility rows plus LastFM coverage note", False, False, "", "summary_csv; report", "completed", "report/plan_d_global_extended_neural_baselines.md"),
        _experiment("D5B", "LLM model registry and backend audit", "audit/documentation", "all", "paper LLM registry", "local_env_audit", "7 paper LLMs and backend checks", False, False, "", "registry_csv; audit_csv; report", "completed", "report/plan_d_llm_model_registry_and_backend_audit.md"),
        _experiment("D5C0", "Ollama and Qwen2 backend availability update", "audit/documentation", "none", "qwen2:7b-instruct", "ollama", "local backend status", False, False, "", "audit_csv; notes", "completed", "report/notes.md"),
        _experiment("D5C1", "Qwen2 local smoke test", "smoke_test", "synthetic_toy", "qwen2:7b-instruct", "ollama", "1 local toy request", True, False, "results/raw_outputs/plan_d_qwen2_ollama_smoke_test_raw.jsonl", "summary_csv; report", "completed", "report/plan_d_qwen2_ollama_smoke_test.md"),
        _experiment("D5D", "Prompt templates and ML-1M manifest", "no-inference preparation", "ml1m", "qwen2:7b-instruct", "none", "20 manifest rows", False, False, "", "manifest_csv; registry_csv; report", "completed", "report/plan_d_llm_prompt_templates_and_ranking_manifest.md"),
        _experiment("D5E", "Qwen2 ML-1M ranking diagnostic", "diagnostic_subset", "ml1m", "qwen2:7b-instruct", "ollama", "20 local LLM requests", True, False, "results/raw_outputs/plan_d_qwen2_ml1m_ranking_diagnostic_raw.jsonl", "metrics_csv; summary_csv; report", "completed", "report/plan_d_qwen2_ml1m_ranking_diagnostic.md"),
        _experiment("D5E.1", "Qwen2 validity and hallucination analysis", "no-inference analysis", "ml1m", "qwen2:7b-instruct", "none", "20 existing D5E rows", False, False, "", "analysis_csv; report", "completed", "report/plan_d_qwen2_output_validity_and_hallucination_analysis.md"),
        _experiment("D5F", "Qwen2 candidate position bias diagnostic", "diagnostic_subset", "ml1m", "qwen2:7b-instruct", "ollama", "60 local LLM requests", True, False, "results/raw_outputs/plan_d_qwen2_ml1m_candidate_position_bias_diagnostic_raw.jsonl", "metrics_csv; summary_csv; report", "completed", "report/plan_d_qwen2_ml1m_candidate_position_bias_diagnostic.md"),
        _experiment("D5G", "Qwen2 strict prompt and post-filter diagnostic", "diagnostic_subset", "ml1m", "qwen2:7b-instruct", "ollama", "20 local LLM requests plus post-filter analysis", True, False, "results/raw_outputs/plan_d_qwen2_ml1m_strict_prompt_diagnostic_raw.jsonl", "metrics_csv; summary_csv; report", "completed", "report/plan_d_qwen2_ml1m_strict_prompt_diagnostic.md"),
        _experiment("D5H", "Mistral availability and smoke test", "conditional smoke_test", "synthetic_toy", "mistral:7b-instruct", "ollama", "1 local toy request after manual pull", True, False, "results/raw_outputs/plan_d_mistral_smoke_test_raw.jsonl", "metrics_csv; summary_csv; report", "completed", "report/plan_d_mistral_smoke_test.md"),
        _experiment("D5I", "Qwen2 vs Mistral strict-prompt comparison", "diagnostic_subset", "ml1m", "qwen2:7b-instruct; mistral:7b-instruct", "ollama", "20 new Mistral requests; Qwen2 reused", True, False, "results/raw_outputs/plan_d_qwen_vs_mistral_ml1m_diagnostic_raw.jsonl", "metrics_csv; summary_csv; report", "completed", "report/plan_d_qwen_vs_mistral_ml1m_diagnostic.md"),
        _experiment("D5J", "Local LLM re-ranking diagnostic", "diagnostic_subset", "ml1m", "qwen2:7b-instruct; mistral:7b-instruct", "ollama", "40 local LLM requests", True, False, "results/raw_outputs/plan_d_llm_ml1m_reranking_diagnostic_raw.jsonl", "metrics_csv; summary_csv; report", "completed", "report/plan_d_llm_ml1m_reranking_diagnostic.md"),
        _experiment("D5K", "Local LLM history length sensitivity diagnostic", "diagnostic_subset", "ml1m", "qwen2:7b-instruct; mistral:7b-instruct", "ollama", "120 local LLM requests", True, False, "results/raw_outputs/plan_d_llm_ml1m_history_length_sensitivity_raw.jsonl", "metrics_csv; summary_csv; report", "completed", "report/plan_d_llm_ml1m_history_length_sensitivity.md"),
        _experiment("D5L", "Final aggregation and reproducibility package", "aggregation/reporting hygiene-audit", "all available Plan D artifacts", "all completed Plan D components", "none", "no new experiment", False, False, "", "final_csv; final_report", "completed", "report/plan_d_final_reproducibility_package.md"),
    ]
    return pd.DataFrame(rows, columns=EXPERIMENT_INDEX_COLUMNS)


def build_metric_summary() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    _append_extended_baseline_metrics(rows)
    _append_d5e_metrics(rows)
    _append_condition_metrics(rows, "D5G", TABLES_DIR / "plan_d_qwen2_ml1m_strict_prompt_condition_summary.csv")
    _append_condition_metrics(rows, "D5I", TABLES_DIR / "plan_d_qwen_vs_mistral_ml1m_model_summary.csv")
    _append_condition_metrics(rows, "D5J", TABLES_DIR / "plan_d_llm_ml1m_reranking_condition_summary.csv")
    _append_condition_metrics(rows, "D5K", TABLES_DIR / "plan_d_llm_ml1m_history_length_condition_summary.csv")
    return pd.DataFrame(rows, columns=METRIC_SUMMARY_COLUMNS)


def build_llm_diagnostic_summary() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    d5e = _read_csv(TABLES_DIR / "plan_d_qwen2_ml1m_ranking_diagnostic_summary.csv")
    if not d5e.empty:
        row = d5e.iloc[0]
        rows.append(_llm_row("D5E", row["model"], "openp5_base", row["hit_rate_at_5"], row["ndcg_at_5"], row["mean_candidate_validity_rate"], row["hallucination_rate"], row["exact_top_k_rate"], row["duplicate_rate"], "Base prompt produced parse/schema-valid JSON but high hallucination.", "Candidate adherence failed despite syntactic validity."))

    d5g = _read_csv(TABLES_DIR / "plan_d_qwen2_ml1m_strict_prompt_condition_summary.csv")
    _append_llm_condition_rows(
        rows,
        d5g.loc[d5g["condition"].isin(["strict_candidate_id_prompt", "strict_post_filtered"])] if not d5g.empty else d5g,
        phase="D5G",
        finding="Strict candidate-ID-only prompting eliminated hallucination and improved Qwen2 utility.",
        interpretation="Prompt-level grounding improved reliability and HR/NDCG on the 20-user diagnostic subset.",
    )

    d5i = _read_csv(TABLES_DIR / "plan_d_qwen_vs_mistral_ml1m_model_summary.csv")
    _append_llm_condition_rows(
        rows,
        d5i,
        phase="D5I",
        finding="Qwen2 strict outperformed Mistral strict in utility while both had perfect candidate validity.",
        interpretation="Local open models behaved differently under the same strict prompt.",
    )

    d5j = _read_csv(TABLES_DIR / "plan_d_llm_ml1m_reranking_condition_summary.csv")
    _append_llm_condition_rows(
        rows,
        d5j,
        phase="D5J",
        finding="LLM re-ranking did not improve the candidate-order reference; reliability differed sharply between Qwen2 and Mistral.",
        interpretation="Re-ranking is not automatically beneficial and still needs reliability checks.",
    )

    d5k = _read_csv(TABLES_DIR / "plan_d_llm_ml1m_history_length_condition_summary.csv")
    _append_llm_condition_rows(
        rows,
        d5k,
        phase="D5K",
        finding="History length changed utility and output stability even with zero hallucination under strict prompts.",
        interpretation="History-window choice is a robustness variable, not a harmless formatting detail.",
    )
    return pd.DataFrame(rows, columns=LLM_DIAGNOSTIC_COLUMNS)


def build_reproducibility_status() -> pd.DataFrame:
    rows = [
        _status("Exact reproduction", "exact reproduction", "not_completed", "Original code exists, but official preprocessed artifacts and full paper LLM matrix are unavailable.", "Exact paper inputs and closed-source model availability are missing.", "Do not claim exact reproduction."),
        _status("Plan C", "reconstructed reproduction", "completed", "Plan C is stable and tagged as plan-c-reconstruction-v1.", "Uses reconstructed artifacts rather than official paper-ready inputs.", "Plan C is reconstructed reproduction."),
        _status("Plan D neural baselines", "extended_subset", "completed", "BPRMF, LightGCN, SASRec, and GRU4Rec have diagnostic and extended reconstructed outputs.", "Not full official paper reproduction.", "Plan D extended neural baselines are reconstructed extensions."),
        _status("Plan D beyond-utility metrics", "extension", "completed", "D1 generated beyond-utility metric tables.", "Extension over the reconstructed setup.", "Beyond-utility metrics are project extensions."),
        _status("D5E-D5K local LLM diagnostics", "diagnostic_subset", "completed", "Qwen2 and Mistral local diagnostics exist for ranking, mitigation, comparison, reranking, and history sensitivity.", "Only ML-1M 20-user diagnostic subset; not all paper LLMs.", "D5E-D5K are diagnostic_subset local LLM diagnostics."),
        _status("Full LLM reproduction", "blocked/conditional", "not_completed", "Closed-source models and full paper matrix were not run.", "Requires APIs, budget, model availability, and full experimental scope.", "LLM full reproduction is not completed."),
        _status("Claude 3 Haiku", "blocked/conditional", "blocked_retired", "D5B registry records exact Claude 3 Haiku model as retired.", "Replacement would be non-exact.", "Claude 3 Haiku exact reproduction is blocked/retired."),
        _status("Llama3 70B", "blocked/conditional", "blocked_local_hardware", "D5B registry records 70B local run as unrealistic on RTX 4060 8GB.", "Cloud/API would be needed.", "Llama3 70B local execution is blocked by hardware."),
        _status("Closed-source LLM original matrix", "blocked/conditional", "conditional_api_dependent", "OpenAI/Anthropic paths were documented but not run.", "Requires explicit approval, keys, budget, and availability checks.", "Closed-source LLM reproduction is conditional/API-dependent."),
        _status("Mistral local path", "diagnostic_subset", "smoke_and_diagnostics_completed", "D5H was initially conditional until manual user pull; D5H smoke and later Mistral diagnostics completed.", "Local model is not exact paper Mistral tag unless documented as compatible.", "D5H became local smoke-test completed after manual user pull."),
        _status("LastFM", "extension", "extension_only_no_valid_utility_evaluation", "LastFM coverage note is documented.", "No valid comparable chronological split/candidates/ground truth for HR@5/NDCG@5.", "LastFM remains extension_only_no_valid_utility_evaluation."),
    ]
    return pd.DataFrame(rows, columns=REPRO_STATUS_COLUMNS)


def build_blocked_items() -> pd.DataFrame:
    rows = [
        _blocked("Claude 3 Haiku exact model", "blocked/conditional", "Exact claude-3-haiku-20240307 is retired.", "blocked_exact_model_retired", "Use a current Claude Haiku replacement only as non-exact and with explicit approval.", "no"),
        _blocked("Llama3 70B local execution", "blocked/conditional", "70B model is not practical on local RTX 4060 8GB.", "blocked_local_hardware", "Use cloud/API only if explicitly approved and clearly labeled non-local.", "no"),
        _blocked("Closed-source LLM original matrix", "blocked/conditional", "OpenAI/Anthropic API calls require keys, budget, and explicit approval.", "conditional_api_dependent", "Run only a bounded optional API diagnostic if approved.", "no"),
        _blocked("LastFM utility evaluation", "extension_only_no_valid_utility_evaluation", "No valid comparable split/candidate/ground-truth setup exists locally.", "blocked_for_valid_utility", "Keep coverage note; do not fabricate HR/NDCG.", "no"),
        _blocked("Official preprocessed paper artifacts", "blocked/conditional", "Exact paper-ready artifacts are unavailable locally.", "blocked_missing_artifacts", "Recover official artifacts if they become available.", "no"),
        _blocked("Full LLM reproduction", "blocked/conditional", "All seven paper LLMs and full repeated experiments were not run.", "not_completed", "Requires separate approved scope, budget, and backend plan.", "no"),
    ]
    return pd.DataFrame(rows, columns=BLOCKED_COLUMNS)


def build_hygiene_audit() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    raw_jsonl = sorted(RAW_OUTPUTS_DIR.glob("*.jsonl")) if RAW_OUTPUTS_DIR.exists() else []
    rows.append(_hygiene("raw_outputs_present", "audit_only", f"{len(raw_jsonl)} raw JSONL files found under results/raw_outputs/.", "Keep raw local traces Git-ignored; do not commit .jsonl responses."))

    gitignore_text = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8") if (PROJECT_ROOT / ".gitignore").exists() else ""
    raw_ignored = "results/raw_outputs/plan_d_*_raw.jsonl" in gitignore_text or "results/raw_outputs/" in gitignore_text
    rows.append(_hygiene("raw_outputs_gitignored", "passed" if raw_ignored else "review", "Raw-output ignore pattern is present." if raw_ignored else "No obvious raw-output ignore pattern found.", "Maintain a Git ignore rule for results/raw_outputs/*.jsonl."))

    tracked_raw = _git_lines(["git", "ls-files", "results/raw_outputs"])
    tracked_jsonl = [path for path in tracked_raw if path.lower().endswith(".jsonl")]
    rows.append(_hygiene("no_raw_jsonl_intended_for_commit", "passed" if not tracked_jsonl else "review", "No tracked raw JSONL files under results/raw_outputs/." if not tracked_jsonl else f"Tracked raw JSONL paths detected: {'; '.join(tracked_jsonl[:10])}", "Do not stage or commit raw JSONL responses."))

    original_status = _git_lines(["git", "status", "--short", "original_repo"])
    rows.append(_hygiene("original_repo_not_modified", "passed" if not original_status else "review", "No Git-visible original_repo/ modifications detected." if not original_status else "; ".join(original_status), "Do not modify original_repo/ unless explicitly instructed."))

    tracked_files = _git_lines(["git", "ls-files"])
    heavy_patterns = ("*.pt", "*.pth", "*.ckpt", "*.bin", "*.safetensors", "*.onnx")
    heavy_tracked = [path for path in tracked_files if any(fnmatch.fnmatch(path.lower(), pattern) for pattern in heavy_patterns)]
    rows.append(_hygiene("no_checkpoint_model_cache_files_tracked", "passed" if not heavy_tracked else "review", "No tracked checkpoint/model/cache files detected by suffix scan." if not heavy_tracked else "; ".join(heavy_tracked[:10]), "Keep model weights, checkpoints, and caches out of Git."))

    large_files = _large_file_summary(PROJECT_ROOT, threshold_bytes=50 * 1024 * 1024)
    rows.append(_hygiene("large_files_summary", "audit_only", large_files or "No files over 50MB detected outside .git during lightweight scan.", "Review large files manually before committing if any are listed."))

    rows.append(_hygiene("test_suite_command", "audit_only", "Recommended command: python -m pytest tests/ -q", "Run before final handoff."))
    rows.append(_hygiene("final_git_status_command", "audit_only", "Recommended command: git status --short", "Review staged/untracked files manually before committing."))
    rows.append(_hygiene("destructive_cleanup", "passed", "No destructive cleanup was performed by D5L.", "Do not run git clean -fd or delete raw outputs automatically."))
    return pd.DataFrame(rows, columns=HYGIENE_COLUMNS)


def build_report(
    experiment_index: pd.DataFrame,
    metric_summary: pd.DataFrame,
    llm_summary: pd.DataFrame,
    repro_status: pd.DataFrame,
    blocked_items: pd.DataFrame,
    hygiene: pd.DataFrame,
) -> str:
    d5e = _llm_summary_row(llm_summary, "D5E", "openp5_base")
    d5g = _llm_summary_row(llm_summary, "D5G", "strict_candidate_id_prompt")
    d5i_qwen = _llm_summary_row(llm_summary, "D5I", "qwen2_strict_reference")
    d5i_mistral = _llm_summary_row(llm_summary, "D5I", "mistral_strict_candidate_id_prompt")
    d5j_ref = _llm_summary_row(llm_summary, "D5J", "candidate_order_reference")
    d5k_qwen_medium = _llm_summary_row(llm_summary, "D5K", "qwen2_medium_history")

    return f"""# Plan D Final Reproducibility Package

## Purpose

D5L is an aggregation/reporting and hygiene-audit phase. It reads existing Plan D outputs and creates compact final package tables without running a new experiment.

## Execution Boundaries

- No new inference was run.
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
- Plan C was not modified.
- No destructive cleanup was performed.
- Raw JSONL outputs remain under `results/raw_outputs/` and Git-ignored.

## Reproduction Claim

This is not exact reproduction. This is not full LLM reproduction. The project should be described as reconstructed reproduction plus diagnostic/extended local reproduction work. Plan C is reconstructed reproduction. Plan D extends that work with extended neural baselines and diagnostic local LLM studies.

## Generated Final Tables

- `results/tables/plan_d_final_experiment_index.csv` with `{len(experiment_index)}` rows.
- `results/tables/plan_d_final_metric_summary.csv` with `{len(metric_summary)}` rows.
- `results/tables/plan_d_final_llm_diagnostic_summary.csv` with `{len(llm_summary)}` rows.
- `results/tables/plan_d_final_reproducibility_status.csv` with `{len(repro_status)}` rows.
- `results/tables/plan_d_final_blocked_items.csv` with `{len(blocked_items)}` rows.
- `results/tables/plan_d_repository_hygiene_audit.csv` with `{len(hygiene)}` rows.

## Main Scientific Story

1. Classical and neural baselines were reconstructed or extended over available reconstructed datasets.
2. Local LLM diagnostics were feasible on the ML-1M 20-user diagnostic subset.
3. D5E base prompting produced valid JSON but high hallucination: HR@5 `{d5e['HR@5']}`, NDCG@5 `{d5e['NDCG@5']}`, candidate validity `{d5e['candidate_validity_rate']}`, hallucination `{d5e['hallucination_rate']}`.
4. D5G strict candidate-ID-only prompting improved grounding: HR@5 `{d5g['HR@5']}`, NDCG@5 `{d5g['NDCG@5']}`, candidate validity `{d5g['candidate_validity_rate']}`, hallucination `{d5g['hallucination_rate']}`.
5. Candidate order and history length affected results, so prompt context is a robustness variable.
6. Qwen2 and Mistral behaved differently: D5I Qwen2 strict HR@5 `{d5i_qwen['HR@5']}` versus Mistral strict HR@5 `{d5i_mistral['HR@5']}`.
7. Re-ranking was not automatically beneficial: D5J candidate-order reference HR@5 `{d5j_ref['HR@5']}` while local LLM reranking conditions did not improve that reference.
8. Beyond-utility reliability metrics are necessary because HR/NDCG alone hide parse validity, candidate validity, hallucination, duplicate behavior, and stability.

## D5K History-Length Note

D5K showed that strict prompts can keep hallucination at `{d5k_qwen_medium['hallucination_rate']}` for Qwen2 medium-history while utility and stability still vary with the history window. Qwen2 medium-history HR@5 was `{d5k_qwen_medium['HR@5']}`.

## Blocked And Conditional Items

The final blocked-items table documents Claude 3 Haiku exact reproduction as retired, Llama3 70B as blocked by local hardware, closed-source LLMs as API/budget conditional, LastFM as `extension_only_no_valid_utility_evaluation`, official preprocessed artifacts as missing, and full LLM reproduction as not completed.

## Hygiene Audit

D5L performed a non-destructive repository hygiene audit. It checked raw-output location, ignore patterns, tracked raw-output paths, `original_repo/` Git status, checkpoint/model/cache suffixes in tracked paths, large files, and recommended final verification commands. No files were deleted.

## Recommended Final Steps

1. Review `git status --short`.
2. Run `python -m pytest tests/ -q`.
3. Use the final CSV tables when drafting final report and presentation claims.
4. Commit only after manually confirming raw JSONL files remain ignored.
"""


def _append_extended_baseline_metrics(rows: list[dict[str, str]]) -> None:
    path = TABLES_DIR / "plan_d_global_extended_neural_baselines_summary.csv"
    frame = _read_csv(path)
    if frame.empty:
        rows.append(_metric("D4.7", "extended_baseline", "", "", "", "input_status", "missing_input", str(path), "missing_input", "Extended baseline summary not found."))
        return
    for _, row in frame.sort_values(["dataset", "method"], kind="mergesort").iterrows():
        for metric in ("hr_at_5", "ndcg_at_5"):
            rows.append(_metric("D4.7", "extended_baseline", row["dataset"], row["method"], row.get("scope", "extended_subset"), metric, row.get(metric, ""), str(path), "available", "Read from global extended neural baseline summary."))


def _append_d5e_metrics(rows: list[dict[str, str]]) -> None:
    path = TABLES_DIR / "plan_d_qwen2_ml1m_ranking_diagnostic_summary.csv"
    frame = _read_csv(path)
    if frame.empty:
        rows.append(_metric("D5E", "llm_diagnostic", "ml1m", "qwen2:7b-instruct", "openp5_base", "input_status", "missing_input", str(path), "missing_input", "D5E summary not found."))
        return
    row = frame.iloc[0]
    for metric, column in {
        "HR@5": "hit_rate_at_5",
        "NDCG@5": "ndcg_at_5",
        "candidate_validity_rate": "mean_candidate_validity_rate",
        "hallucination_rate": "hallucination_rate",
    }.items():
        rows.append(_metric("D5E", "llm_diagnostic", row["dataset"], row["model"], row["prompt_strategy"], metric, row[column], str(path), "available", "Read from D5E Qwen2 ranking diagnostic summary."))


def _append_condition_metrics(rows: list[dict[str, str]], phase: str, path: Path) -> None:
    frame = _read_csv(path)
    if frame.empty:
        rows.append(_metric(phase, "llm_diagnostic", "", "", "", "input_status", "missing_input", str(path), "missing_input", f"{phase} condition summary not found."))
        return
    for _, row in frame.iterrows():
        dataset = row.get("dataset", "ml1m")
        model = row.get("model", "")
        condition = row.get("condition", "")
        for metric, column in {
            "HR@5": "hr_at_5",
            "NDCG@5": "ndcg_at_5",
            "candidate_validity_rate": "mean_candidate_validity_rate",
            "hallucination_rate": "hallucination_rate",
        }.items():
            if column in row:
                rows.append(_metric(phase, "llm_diagnostic", dataset, model, condition, metric, row[column], str(path), "available", f"Read from {phase} condition/model summary."))


def _append_llm_condition_rows(
    rows: list[dict[str, str]],
    frame: pd.DataFrame,
    *,
    phase: str,
    finding: str,
    interpretation: str,
) -> None:
    if frame.empty:
        rows.append(_llm_row(phase, "missing_input", "missing_input", "", "", "", "", "", "", f"{phase} input summary missing.", "No values fabricated."))
        return
    for _, row in frame.iterrows():
        rows.append(
            _llm_row(
                phase,
                row.get("model", ""),
                row.get("condition", ""),
                row.get("hr_at_5", row.get("hit_rate_at_5", "")),
                row.get("ndcg_at_5", ""),
                row.get("mean_candidate_validity_rate", ""),
                row.get("hallucination_rate", ""),
                row.get("exact_top_k_rate", ""),
                row.get("duplicate_rate", ""),
                finding,
                interpretation,
            )
        )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def _experiment(*values: object) -> dict[str, str]:
    return {column: _text(value) for column, value in zip(EXPERIMENT_INDEX_COLUMNS, values)}


def _metric(*values: object) -> dict[str, str]:
    return {column: _text(value) for column, value in zip(METRIC_SUMMARY_COLUMNS, values)}


def _llm_row(*values: object) -> dict[str, str]:
    return {column: _text(value) for column, value in zip(LLM_DIAGNOSTIC_COLUMNS, values)}


def _status(*values: object) -> dict[str, str]:
    return {column: _text(value) for column, value in zip(REPRO_STATUS_COLUMNS, values)}


def _blocked(*values: object) -> dict[str, str]:
    return {column: _text(value) for column, value in zip(BLOCKED_COLUMNS, values)}


def _hygiene(*values: object) -> dict[str, str]:
    return {column: _text(value) for column, value in zip(HYGIENE_COLUMNS, values)}


def _text(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _git_lines(command: list[str]) -> list[str]:
    return common_git.git_lines_unavailable_on_error(command, PROJECT_ROOT)


def _large_file_summary(root: Path, *, threshold_bytes: int) -> str:
    findings: list[str] = []
    skip_parts = {".git", "__pycache__", ".pytest_cache"}
    for path in sorted(root.rglob("*")):
        if any(part in skip_parts for part in path.parts):
            continue
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size >= threshold_bytes:
            findings.append(f"{path.relative_to(root).as_posix()} ({size} bytes)")
        if len(findings) >= 10:
            break
    return "; ".join(findings)


def _llm_summary_row(frame: pd.DataFrame, phase: str, condition: str) -> pd.Series:
    rows = frame.loc[(frame["diagnostic_phase"] == phase) & (frame["condition"] == condition)]
    if rows.empty:
        return pd.Series({column: "missing" for column in LLM_DIAGNOSTIC_COLUMNS})
    return rows.iloc[0]
