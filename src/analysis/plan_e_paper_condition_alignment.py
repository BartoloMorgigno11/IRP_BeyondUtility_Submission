from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORT_DIR = PROJECT_ROOT / "report"

PHASE_ID = "E0"
PHASE_TITLE = "Paper-condition alignment and scale-up feasibility audit"

ALIGNMENT_COLUMNS = [
    "condition_id",
    "condition_group",
    "paper_condition",
    "paper_target",
    "current_project_status",
    "current_artifacts_or_evidence",
    "gap",
    "required_action",
    "feasibility_class",
    "exactness_class",
    "risk_level",
    "plan_e_phase",
    "notes",
]

MISSING_COLUMNS = [
    "missing_condition_id",
    "condition_group",
    "missing_condition",
    "current_project_status",
    "why_missing_or_not_aligned",
    "required_action",
    "feasibility_class",
    "exactness_class",
    "risk_level",
    "plan_e_phase",
    "notes",
]

EXACTNESS_COLUMNS = [
    "component_id",
    "component",
    "current_project_phase",
    "current_project_status",
    "exactness_class",
    "feasibility_class",
    "evidence",
    "required_for_exact_reproduction",
    "gap",
    "next_action",
    "notes",
]

SCALEUP_COLUMNS = [
    "scenario_id",
    "users_per_dataset",
    "datasets",
    "dataset_count",
    "models",
    "model_count",
    "prompt_strategies",
    "prompt_strategy_count",
    "tasks",
    "task_count",
    "calls_per_user_task",
    "estimated_total_calls",
    "classification",
    "notes",
]

LOCAL_RUNTIME_COLUMNS = [
    "scenario_id",
    "model_scope",
    "estimated_total_calls",
    "assumed_seconds_per_call",
    "estimated_seconds",
    "estimated_minutes",
    "estimated_hours",
    "estimated_days",
    "feasibility_note",
]

API_COST_COLUMNS = [
    "scenario_id",
    "model_family",
    "pricing_status",
    "assumed_cost_per_1000_calls_usd",
    "estimated_total_calls",
    "estimated_cost_usd",
    "requires_current_price_check",
    "source_status",
    "notes",
]

ROADMAP_COLUMNS = [
    "phase_id",
    "title",
    "objective",
    "requires_inference",
    "requires_api",
    "requires_download",
    "requires_training",
    "expected_outputs",
    "risk_level",
    "recommended_order",
    "stop_condition",
    "notes",
]


@dataclass(frozen=True)
class PlanEAuditPaths:
    alignment_matrix: Path = TABLES_DIR / "plan_e_paper_condition_alignment_matrix.csv"
    missing_conditions: Path = TABLES_DIR / "plan_e_missing_conditions.csv"
    exactness_feasibility: Path = TABLES_DIR / "plan_e_exactness_feasibility.csv"
    scaleup_call_estimates: Path = TABLES_DIR / "plan_e_scaleup_call_estimates.csv"
    local_runtime_estimates: Path = TABLES_DIR / "plan_e_local_runtime_estimates.csv"
    api_cost_scenarios: Path = TABLES_DIR / "plan_e_api_cost_scenarios.csv"
    recommended_roadmap: Path = TABLES_DIR / "plan_e_recommended_roadmap.csv"
    report: Path = REPORT_DIR / "plan_e_paper_condition_alignment_audit.md"

    def required_output_paths(self) -> list[Path]:
        return [
            self.alignment_matrix,
            self.missing_conditions,
            self.exactness_feasibility,
            self.scaleup_call_estimates,
            self.local_runtime_estimates,
            self.api_cost_scenarios,
            self.recommended_roadmap,
            self.report,
        ]


def build_paper_condition_alignment_matrix(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Build an explicit paper-condition alignment matrix for Plan E0."""

    evidence_plan_c = _evidence(
        project_root,
        "results/tables/plan_c_final_results_overview.csv",
        "report/plan_c_final_report.md",
    )
    evidence_plan_d_neural = _evidence(
        project_root,
        "results/tables/plan_d_global_extended_neural_baselines_summary.csv",
        "report/plan_d_global_extended_neural_baselines.md",
    )
    evidence_d5 = _evidence(
        project_root,
        "results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_summary.csv",
        "results/tables/plan_d_qwen2_ml1m_strict_prompt_condition_summary.csv",
        "results/tables/plan_d_qwen_vs_mistral_ml1m_model_summary.csv",
        "report/plan_d_final_reproducibility_package.md",
    )
    evidence_model_registry = _evidence(
        project_root,
        "results/tables/plan_d_llm_model_registry.csv",
        "report/plan_d_llm_model_registry_and_backend_audit.md",
    )
    evidence_beyond = _evidence(
        project_root,
        "results/metrics/plan_d_beyond_utility_metrics.csv",
        "results/tables/plan_d_beyond_utility_comparison.csv",
    )

    rows = [
        _alignment("DATASET_AMAZON_BEAUTY", "dataset", "Amazon Beauty", "Included in paper datasets.", "Available through reconstructed Plan C and Plan D extended baselines.", evidence_plan_c + "; " + evidence_plan_d_neural, "Official paper-preprocessed artifacts are not proven identical.", "Keep reconstructed status; E2 can build paper-like 1000-user candidate manifests.", "near reproduction", "reconstructed reproduction", "medium", "E2", "Accessible dataset condition is covered, but not exact-paper input state."),
        _alignment("DATASET_AMAZON_SPORTS", "dataset", "Amazon Sports", "Included in paper datasets.", "Available through reconstructed Plan C and Plan D extended baselines.", evidence_plan_c + "; " + evidence_plan_d_neural, "Official paper-preprocessed artifacts are not proven identical.", "Keep reconstructed status; E2 can build paper-like 1000-user candidate manifests.", "near reproduction", "reconstructed reproduction", "medium", "E2", "Accessible dataset condition is covered, but not exact-paper input state."),
        _alignment("DATASET_ML1M", "dataset", "ML-1M", "Included in paper datasets.", "Available through reconstructed Plan C, processed ML-1M artifacts, Plan D neural baselines, and D5 local LLM diagnostics.", _evidence(project_root, "data/processed/ml1m/interactions.csv", "results/tables/plan_d_ml1m_extended_neural_baselines_summary.csv") + "; " + evidence_d5, "Official paper-preprocessed artifacts are not proven identical.", "Keep reconstructed status; E2 can scale candidate manifests from current ML-1M pipeline.", "near reproduction", "reconstructed reproduction", "medium", "E2", "ML-1M is the strongest local LLM diagnostic base so far."),
        _alignment("DATASET_LASTFM", "dataset", "LastFM", "Included in paper datasets.", "Documented as problematic and not valid for comparable utility evaluation.", evidence_plan_d_neural, "No valid local chronological split/candidate/ground-truth setup for comparable HR@5/NDCG@5.", "Keep blocked or extension-only unless valid artifacts are found.", "blocked/conditional", "extension_only_no_valid_utility_evaluation", "high", "E1", "Do not run LastFM or fabricate utility metrics in E0."),
        _alignment("TASK_RANKING", "task", "Ranking", "Paper evaluates ranking.", "Plan C and D5 include ranking-style evaluation; D5 is 20-user diagnostic_subset.", evidence_plan_c + "; " + evidence_d5, "Full paper LLM ranking matrix at 1000 users is not completed.", "E2 and E5 can scale valid local ranking manifests if approved.", "conditional/API-dependent", "diagnostic_subset", "medium", "E5", "Ranking logic exists, but full paper scale is missing."),
        _alignment("TASK_RERANKING", "task", "Re-ranking", "Paper evaluates re-ranking.", "D5J local LLM reranking diagnostic exists for ML-1M 20 users.", _evidence(project_root, "report/plan_d_llm_ml1m_reranking_diagnostic.md", "results/metrics/plan_d_llm_ml1m_reranking_diagnostic_metrics.csv"), "Full paper reranking matrix across datasets/models/prompts is not completed.", "E4 should align prompts; E5 can scale local reranking if feasible.", "conditional/API-dependent", "diagnostic_subset", "medium", "E4", "D5J is diagnostic_subset, not paper-exact reranking reproduction."),
        _alignment("METRIC_HR_AT_5", "utility_metric", "HR@5", "Paper utility metric.", "Implemented and used in Plan C/D/D5.", _evidence(project_root, "src/evaluation/utility_metrics.py", "tests/test_utility_metrics.py"), "Metric implementation exists, but exact input conditions are not exact-paper.", "Keep using generated CSV metrics only.", "near reproduction", "reconstructed reproduction", "low", "E3", "Metric condition is locally covered."),
        _alignment("METRIC_NDCG_AT_5", "utility_metric", "NDCG@5", "Paper utility metric.", "Implemented and used in Plan C/D/D5.", _evidence(project_root, "src/evaluation/utility_metrics.py", "tests/test_utility_metrics.py"), "Metric implementation exists, but exact input conditions are not exact-paper.", "Keep using generated CSV metrics only.", "near reproduction", "reconstructed reproduction", "low", "E3", "Metric condition is locally covered."),
        _alignment("CANDIDATE_POOL_SIZE_20", "candidate_pool", "20 candidate items per user", "20 items per test user.", "Matched in D5 diagnostics.", evidence_d5, "Matched only for diagnostic_subset, not all datasets at 1000 users.", "E2 should generate 1000-user candidate manifests for each valid dataset.", "near reproduction", "diagnostic_subset", "medium", "E2", "Candidate pool size 20 is already represented in D5."),
        _alignment("CANDIDATE_POS_NEG_1_19", "candidate_pool", "1 positive item plus 19 negative items", "One positive target and 19 negatives per user.", "Matched in D5 diagnostics.", evidence_d5, "Matched only for diagnostic_subset, not all datasets at 1000 users.", "E2 should validate one-positive/19-negative manifests at scale.", "near reproduction", "diagnostic_subset", "medium", "E2", "The 1 positive + 19 negatives condition is already represented in D5."),
        _alignment("USER_SCALE_1000", "scale", "1000 test users", "Main LLM experiments use 1000 test users.", "Not completed for local LLMs; D5 uses 20-user diagnostic_subset.", evidence_d5, "Full 1000-user local LLM scale-up has not been run.", "E1 estimate cost/runtime; E5 run local scale-up only if approved.", "conditional/API-dependent", "diagnostic_subset", "high", "E5", "E0 only estimates feasibility and makes no inference calls."),
        _alignment("BASELINE_BM25", "baseline", "BM25", "Paper baseline/recommender matrix includes BM25.", "Plan C includes ContentBM25/BM25-like reconstructed baseline.", _evidence(project_root, "src/baselines/content_bm25_baseline.py", "results/tables/plan_c_baseline_metrics_summary.csv"), "BM25-like is not guaranteed exact to paper implementation.", "E3 can document delta to paper BM25 if original settings are recoverable.", "near reproduction", "reconstructed reproduction", "medium", "E3", "Use BM25-like wording unless exact implementation is proven."),
        _alignment("BASELINE_MOSTPOP", "baseline", "MostPop", "Paper baseline/recommender matrix includes MostPop.", "Plan C includes MostPop on reconstructed datasets.", _evidence(project_root, "src/baselines/mostpop.py", "results/tables/plan_c_baseline_metrics_summary.csv"), "Official paper inputs are not proven identical.", "Keep reconstructed reproduction classification.", "near reproduction", "reconstructed reproduction", "low", "E3", "MostPop is locally covered."),
        _alignment("BASELINE_BPRMF", "baseline", "BPRMF", "Paper baseline/recommender matrix includes BPRMF.", "Plan D extended neural baseline completed on ML-1M, Beauty, and Sports.", evidence_plan_d_neural, "Not exact paper training setup or official data state.", "E3 can compare settings against paper where documented.", "near reproduction", "extended_subset", "medium", "E3", "Extended neural baseline condition is accessible but not exact."),
        _alignment("BASELINE_GRU4REC", "baseline", "GRU4Rec", "Paper baseline/recommender matrix includes GRU4Rec.", "Plan D extended neural baseline completed on ML-1M, Beauty, and Sports.", evidence_plan_d_neural, "Not exact paper training setup or official data state.", "E3 can compare settings against paper where documented.", "near reproduction", "extended_subset", "medium", "E3", "Extended neural baseline condition is accessible but not exact."),
        _alignment("BASELINE_SASREC", "baseline", "SASRec", "Paper baseline/recommender matrix includes SASRec.", "Plan D extended neural baseline completed on ML-1M, Beauty, and Sports.", evidence_plan_d_neural, "Not exact paper training setup or official data state.", "E3 can compare settings against paper where documented.", "near reproduction", "extended_subset", "medium", "E3", "Extended neural baseline condition is accessible but not exact."),
        _alignment("BASELINE_LIGHTGCN", "baseline", "LightGCN", "Paper baseline/recommender matrix includes LightGCN.", "Plan D extended neural baseline completed on ML-1M, Beauty, and Sports.", evidence_plan_d_neural, "Not exact paper training setup or official data state.", "E3 can compare settings against paper where documented.", "near reproduction", "extended_subset", "medium", "E3", "Extended neural baseline condition is accessible but not exact."),
        _alignment("LLM_GPT35", "llm", "GPT-3.5", "Paper LLM matrix includes GPT-3.5.", "Not run; closed-source model remains API-dependent.", evidence_model_registry, "Requires current API availability, key, budget, and explicit approval.", "E6 audit availability/pricing; E7 only mini API reproduction if approved.", "conditional/API-dependent", "conditional/API-dependent", "high", "E6", "No OpenAI API calls were used in Plan D or E0."),
        _alignment("LLM_GPT4O_MINI", "llm", "GPT-4o-mini", "Paper LLM matrix includes GPT-4o-mini.", "Not run; closed-source model remains API-dependent.", evidence_model_registry, "Requires current API availability, key, budget, and explicit approval.", "E6 audit availability/pricing; E7 only mini API reproduction if approved.", "conditional/API-dependent", "conditional/API-dependent", "high", "E6", "No OpenAI API calls were used in Plan D or E0."),
        _alignment("LLM_GPT4O", "llm", "GPT-4o", "Paper LLM matrix includes GPT-4o.", "Not run; closed-source model remains API-dependent.", evidence_model_registry, "Requires current API availability, key, budget, and explicit approval.", "E6 audit availability/pricing; E7 only mini API reproduction if approved.", "conditional/API-dependent", "conditional/API-dependent", "high", "E6", "No OpenAI API calls were used in Plan D or E0."),
        _alignment("LLM_CLAUDE3_HAIKU", "llm", "Claude 3 Haiku", "Paper LLM matrix includes Claude 3 Haiku.", "Not run; exact model availability is blocked/conditional if retired or unavailable.", evidence_model_registry, "Requires current Anthropic availability verification; replacement would not be exact.", "E6 availability audit only; any API run requires explicit approval.", "blocked/conditional", "blocked/conditional", "high", "E6", "Treat as blocked/conditional unless official exact model access is confirmed."),
        _alignment("LLM_LLAMA3_70B", "llm", "Llama3-70B", "Paper LLM matrix includes Llama3-70B.", "Not run locally; current laptop GPU is insufficient for local 70B execution.", evidence_model_registry, "Requires cloud GPU, API, or other external hardware path.", "E8 assess cloud/API feasibility only if approved.", "hardware/cloud/API-dependent", "hardware/cloud/API-dependent", "high", "E8", "Do not attempt local heavy run in E0."),
        _alignment("LLM_QWEN2_7B", "llm", "Qwen2-7B", "Paper LLM matrix includes Qwen2-7B.", "Local qwen2:7b-instruct diagnostics completed through D5.", evidence_d5, "Only 20-user ML-1M diagnostic_subset; strict prompt is not paper-exact.", "E5 can scale local Qwen2 only after E2/E4 and approval.", "near reproduction", "diagnostic_subset", "medium", "E5", "Local Qwen2 status is represented but not full paper scale."),
        _alignment("LLM_MISTRAL_7B", "llm", "Mistral-7B", "Paper LLM matrix includes Mistral-7B.", "Local mistral:7b-instruct smoke and diagnostics completed after manual availability.", evidence_d5, "Only 20-user ML-1M diagnostic_subset; exact paper tag compatibility is not proven.", "E5 can scale local Mistral only after E2/E4 and approval.", "near reproduction", "diagnostic_subset", "medium", "E5", "Local Mistral status is represented but not full paper scale."),
        _alignment("PROMPT_OPENP5_LIKE", "prompt_strategy", "OpenP5-like", "Paper prompting strategy.", "D5 base prompt is described as OpenP5-like diagnostic, not exact paper prompt reproduction.", evidence_d5, "Exact prompt templates and generation pipeline are not fully aligned.", "E4 should map original templates and project prompts side by side.", "near reproduction", "diagnostic_subset", "medium", "E4", "Prompt strategy names must stay qualified as -like when not exact."),
        _alignment("PROMPT_LLMRANK_LIKE", "prompt_strategy", "LLMRank-like", "Paper prompting strategy.", "Some original templates exist; project reranking diagnostic is not full LLMRank reproduction.", _evidence(project_root, "original_repo/EvaLLMasRecommender/eva/templates/llmrank.txt", "report/plan_d_llm_ml1m_reranking_diagnostic.md"), "Full exact LLMRank prompt matrix was not executed.", "E4 should align prompt text and output schema before scale-up.", "conditional/API-dependent", "diagnostic_subset", "medium", "E4", "Reading original_repo templates is allowed; modifying original_repo is not."),
        _alignment("PROMPT_LLMSREC_ICL", "prompt_strategy", "LLMSRec/ICL", "Paper prompting strategy.", "Not fully reconstructed or executed.", evidence_model_registry, "Exact ICL examples, demonstrations, and full matrix are missing.", "E4 should reconstruct prompt registry without running inference.", "conditional/API-dependent", "blocked/conditional", "medium", "E4", "Prompt alignment is a prerequisite for any future LLM scale-up."),
        _alignment("BU_HALLUCINATION", "beyond_utility", "Hallucination", "Paper beyond-utility dimension.", "Implemented in project and used in D5 diagnostics.", _evidence(project_root, "src/evaluation/hallucination.py", "results/tables/plan_d_qwen2_output_validity_breakdown.csv"), "Full paper matrix hallucination analysis is not complete.", "E10 can aggregate across scaled local/API outputs only if generated.", "near reproduction", "diagnostic_subset", "medium", "E10", "D5E.1 showed grounding failure rather than parsing failure."),
        _alignment("BU_CANDIDATE_POSITION_BIAS", "beyond_utility", "Candidate position bias", "Paper beyond-utility dimension.", "D5F candidate position bias diagnostic completed.", _evidence(project_root, "report/plan_d_qwen2_ml1m_candidate_position_bias_diagnostic.md", "results/metrics/plan_d_qwen2_ml1m_candidate_position_bias_metrics.csv"), "Only ML-1M 20-user local Qwen2 diagnostic.", "E10 can scale after E2/E4/E5 if approved.", "near reproduction", "diagnostic_subset", "medium", "E10", "Position effects are represented but not full paper scale."),
        _alignment("BU_HISTORY_LENGTH_SENSITIVITY", "beyond_utility", "History length sensitivity", "Paper beyond-utility dimension.", "D5K local history length sensitivity diagnostic completed.", _evidence(project_root, "report/plan_d_llm_ml1m_history_length_sensitivity.md", "results/metrics/plan_d_llm_ml1m_history_length_sensitivity_metrics.csv"), "Only local diagnostic_subset.", "E10 can scale after E5 if useful.", "near reproduction", "diagnostic_subset", "medium", "E10", "History length is documented as a robustness variable."),
        _alignment("BU_GENERATION_INVOLVED_PROFILES", "beyond_utility", "Generation-involved performance/user profiles", "Paper beyond-utility/profile generation analysis.", "Not completed in current project.", evidence_model_registry, "Requires profile-generation prompts and additional inference outputs.", "E9 can implement only after prompt alignment and approval.", "conditional/API-dependent", "blocked/conditional", "high", "E9", "No generation-involved profile experiment is run in E0."),
        _alignment("BU_NOVELTY", "beyond_utility", "Novelty", "Paper beyond-utility dimension.", "Partially reconstructed as project beyond-utility metric extension.", evidence_beyond, "Full paper-equivalent novelty matrix is not complete.", "E10 can complete if required inputs exist.", "near reproduction", "extension", "medium", "E10", "Keep as project extension unless paper-exact definition is proven."),
        _alignment("BU_POPULARITY", "beyond_utility", "Popularity", "Paper beyond-utility dimension.", "Partially reconstructed as project beyond-utility metric extension.", evidence_beyond, "Full paper-equivalent popularity matrix is not complete.", "E10 can complete if required inputs exist.", "near reproduction", "extension", "medium", "E10", "Keep as project extension unless paper-exact definition is proven."),
        _alignment("BU_FAIRNESS", "beyond_utility", "Fairness", "Paper beyond-utility dimension.", "Only partially reconstructed or not yet completed depending on available attributes.", evidence_beyond, "Fairness requires validated group definitions and paper-equivalent setup.", "E10 should audit whether valid fairness attributes exist.", "blocked/conditional", "extension", "high", "E10", "Do not invent group labels or fairness values."),
        _alignment("BU_CANDIDATE_VALIDITY_EXTENSION", "beyond_utility", "Candidate validity", "Not a core paper dimension; project reliability extension.", "Implemented and central to D5 diagnostics.", evidence_d5, "This is a useful extension, not a paper-exact condition.", "Keep separate from paper beyond-utility claims.", "near reproduction", "extension", "low", "E10", "Candidate validity should remain labelled as a project extension."),
        _alignment("PROJECT_PHASE_PLAN_C", "project_phase", "Plan C", "Accessible reconstructed reproduction foundation.", "Completed, stable, tagged plan-c-reconstruction-v1.", evidence_plan_c, "Not exact reproduction because official artifacts are missing.", "Do not modify Plan C; use it as fixed evidence.", "near reproduction", "reconstructed reproduction", "low", "E0", "Plan C is represented and preserved."),
        _alignment("PROJECT_PHASE_PLAN_D_NEURAL", "project_phase", "Plan D extended neural baselines", "Extended accessible neural baseline coverage.", "Completed through D5M on plan-d-extensions.", evidence_plan_d_neural, "Not exact paper training/input conditions.", "Keep as extended_subset or near reproduction depending on condition.", "near reproduction", "extended_subset", "low", "E0", "Plan D neural outputs remain unchanged in E0."),
        _alignment("PROJECT_PHASE_D5_LOCAL_LLM", "project_phase", "D5 local LLM diagnostics", "Accessible local LLM diagnostics.", "Completed D5E-D5M with Qwen2 and Mistral local diagnostics.", evidence_d5, "Only diagnostic_subset and local extension, not full paper LLM matrix.", "Use as feasibility evidence for E5 planning.", "near reproduction", "diagnostic_subset", "medium", "E0", "D5 is represented as diagnostic_subset/local extension."),
        _alignment("PROJECT_PROMPT_STRICT_ID", "project_extension", "Strict candidate-ID-only prompt", "Not paper-exact.", "Completed as D5G mitigation extension.", evidence_d5, "Improves grounding but changes prompt condition.", "Label as extension, not paper-exact prompt strategy.", "near reproduction", "extension", "low", "E4", "This prompt must not be overclaimed as OpenP5/LLMRank/LLMSRec exact."),
    ]
    return pd.DataFrame(rows, columns=ALIGNMENT_COLUMNS)


def build_missing_conditions_table(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Build a table of paper conditions still missing or only conditionally feasible."""

    rows = [
        _missing("MISS_FULL_1000_LOCAL_LLM", "scale", "Full 1000-user local LLM scale-up", "Not completed; D5 local LLM results are 20-user diagnostic_subset.", "E0 is audit-only and D5 deliberately stayed small.", "E5 can run only after candidate manifests, prompt alignment, and explicit approval.", "conditional/API-dependent", "diagnostic_subset", "high", "E5", "No inference is run in E0."),
        _missing("MISS_1000_CANDIDATE_MANIFESTS", "candidate_pool", "1000-user candidate manifests for all valid datasets", "D5 has 20-user ML-1M manifests; full valid dataset manifests are missing.", "Need deterministic 1-positive/19-negative construction at scale.", "E2 should build and validate manifests for ML-1M, Beauty, and Sports; LastFM stays conditional.", "near reproduction", "blocked/conditional", "medium", "E2", "Candidate manifests must not write raw LLM outputs."),
        _missing("MISS_EXACT_PROMPTS", "prompt_strategy", "Exact paper prompt strategies", "OpenP5-like, LLMRank-like, and LLMSRec/ICL are not fully aligned or executed.", "Prompt text, demonstrations, schema, and ordering need side-by-side mapping.", "E4 should build prompt registry and alignment notes without inference first.", "conditional/API-dependent", "blocked/conditional", "medium", "E4", "Strict candidate-ID-only prompting is an extension, not paper-exact."),
        _missing("MISS_CLOSED_SOURCE_LLMS", "llm", "Closed-source LLMs", "GPT-3.5, GPT-4o-mini, GPT-4o, and Claude 3 Haiku were not run.", "Remote API calls are disallowed without explicit approval and cost checks.", "E6 should audit current availability/prices; E7 can run a mini API reproduction only if approved.", "conditional/API-dependent", "conditional/API-dependent", "high", "E6", "No OpenAI, Anthropic, Google/Gemini, Mistral API, or other remote API call in E0."),
        _missing("MISS_CLAUDE3_HAIKU", "llm", "Claude 3 Haiku availability issue", "Exact Claude 3 Haiku access is blocked/conditional if retired or unavailable.", "Current official availability must be verified later; replacements would be non-exact.", "E6 should verify official availability before any claim or run.", "blocked/conditional", "blocked/conditional", "high", "E6", "No current price or availability lookup is performed in E0."),
        _missing("MISS_LLAMA3_70B", "llm", "Llama3-70B execution path", "Not locally feasible on RTX 4060 Laptop GPU with 8 GB VRAM.", "70B local execution requires hardware beyond the current machine.", "E8 should assess cloud/API feasibility only if approved.", "hardware/cloud/API-dependent", "hardware/cloud/API-dependent", "high", "E8", "No model pull, download, or cloud GPU use in E0."),
        _missing("MISS_LASTFM_EXACT_EVAL", "dataset", "LastFM exact evaluation issue", "LastFM remains extension_only_no_valid_utility_evaluation or blocked unless valid artifacts are found.", "Valid comparable split/candidate/ground-truth artifacts are unavailable locally.", "E1 can inventory artifacts; do not run LastFM until valid setup exists.", "blocked/conditional", "extension_only_no_valid_utility_evaluation", "high", "E1", "No LastFM run in E0."),
        _missing("MISS_API_DEPENDENT_MODELS", "llm", "API-dependent models", "Closed-source/API models are not executed.", "Need keys, budget, approval, current availability, and official pricing.", "E6 availability/cost audit; E7 optional mini run only if approved.", "conditional/API-dependent", "conditional/API-dependent", "high", "E6", "E0 uses placeholder cost assumptions only."),
        _missing("MISS_FULL_PAPER_LLM_MATRIX", "llm", "Full paper 7-LLM matrix", "Only local Qwen2/Mistral diagnostic_subset work exists.", "Five paper LLM conditions are missing or blocked; scale is incomplete.", "E5/E6/E7/E8 split local, API, and 70B feasibility paths.", "conditional/API-dependent", "blocked/conditional", "high", "E5", "No full paper LLM matrix has been reproduced."),
        _missing("MISS_FULL_RANKING_RERANKING_MATRIX", "task", "Full ranking and reranking matrix", "Ranking and reranking diagnostics exist only in bounded scopes.", "All datasets, models, prompts, and 1000-user scale are not complete.", "E4/E5 can define and optionally run a scaled matrix.", "conditional/API-dependent", "diagnostic_subset", "high", "E5", "D5J reranking is diagnostic_subset only."),
        _missing("MISS_GENERATION_INVOLVED_PROFILES", "beyond_utility", "Generation-involved profile experiments", "Not completed.", "Requires profile-generation prompt pipeline and inference outputs.", "E9 should implement only after E4 and with explicit inference approval.", "conditional/API-dependent", "blocked/conditional", "high", "E9", "No profile-generation experiment is run in E0."),
        _missing("MISS_NOVELTY_POPULARITY_FAIRNESS", "beyond_utility", "Novelty, popularity, and fairness paper-equivalent metrics", "Novelty/popularity are partially reconstructed as extensions; fairness remains conditional on valid attributes.", "Paper-equivalent definitions and required attributes are not fully aligned.", "E10 should complete only where definitions and inputs are valid.", "blocked/conditional", "extension", "medium", "E10", "Do not invent fairness attributes or values."),
        _missing("MISS_OFFICIAL_PREPROCESSED_ARTIFACTS", "artifact", "Official paper-preprocessed artifacts", "Original repo exists, but exact paper-ready data artifacts are not fully available.", "Exact reproduction depends on official inputs and outputs.", "E1 should inventory all available and missing artifacts.", "blocked/conditional", "blocked/conditional", "high", "E1", "This is why Plan C remains reconstructed reproduction."),
    ]
    return pd.DataFrame(rows, columns=MISSING_COLUMNS)


def build_exactness_feasibility_table(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Build the Plan E0 exactness and feasibility classification table."""

    rows = [
        _exact("PLAN_C", "Plan C reconstruction", "Plan C", "Completed and stable; tag plan-c-reconstruction-v1.", "reconstructed reproduction", "near reproduction", _evidence(project_root, "report/plan_c_final_report.md", "results/tables/plan_c_final_results_overview.csv"), "No", "Official paper artifacts are missing, so exact reproduction is not supported.", "Do not modify; cite as reconstructed reproduction.", "Plan C is fixed evidence for Plan E."),
        _exact("PLAN_D_NEURAL_EXTENDED", "Plan D extended neural baselines", "Plan D", "BPRMF, LightGCN, SASRec, and GRU4Rec completed on ML-1M, Beauty, and Sports.", "extended_subset", "near reproduction", _evidence(project_root, "results/tables/plan_d_global_extended_neural_baselines_summary.csv"), "Partly", "Training details and exact data state are not paper-exact.", "E3 can audit setting deltas without rerunning heavy training.", "Depending on condition, this is extended_subset and near reproduction, not exact reproduction."),
        _exact("D5_QWEN_MISTRAL_LOCAL", "D5 local Qwen2/Mistral diagnostics", "D5E-D5K", "Local Qwen2 and Mistral diagnostics completed on ML-1M 20 users.", "diagnostic_subset", "near reproduction", _evidence(project_root, "results/tables/plan_d_qwen_vs_mistral_ml1m_model_summary.csv", "report/plan_d_final_reproducibility_package.md"), "No", "Only local 20-user diagnostic_subset work.", "Use as scale-up feasibility evidence for E5.", "This is a local extension and diagnostic_subset, not full paper LLM reproduction."),
        _exact("STRICT_PROMPT", "Strict candidate-ID-only prompt", "D5G", "Completed and effective for grounding in diagnostic_subset.", "extension", "near reproduction", _evidence(project_root, "results/tables/plan_d_qwen2_ml1m_strict_prompt_condition_summary.csv"), "No", "Prompt condition is intentionally stricter than paper prompt strategies.", "Label as extension; do not call paper-exact.", "Strict prompt is not an exact OpenP5-like, LLMRank-like, or LLMSRec/ICL condition."),
        _exact("CLOSED_SOURCE_GPT_CLAUDE", "Closed-source GPT/Claude models", "Plan D/E0", "Not run.", "conditional/API-dependent", "conditional/API-dependent", _evidence(project_root, "results/tables/plan_d_llm_model_registry.csv"), "Yes for full paper LLM matrix", "Requires APIs, budget, current availability, keys, and explicit approval.", "E6 availability/cost audit, then E7 mini API reproduction only if approved.", "No remote API calls have been made."),
        _exact("CLAUDE3_HAIKU", "Claude 3 Haiku exact model", "Plan D/E0", "Blocked/conditional if retired or unavailable.", "blocked/conditional", "blocked/conditional", _evidence(project_root, "report/plan_d_llm_model_registry_and_backend_audit.md"), "Yes for exact paper LLM matrix", "Exact model availability may no longer exist.", "E6 must verify official current availability.", "Replacement models must be labelled non-exact."),
        _exact("LLAMA3_70B", "Llama3-70B", "Plan D/E0", "Not locally feasible on RTX 4060 Laptop GPU, 8 GB VRAM.", "hardware/cloud/API-dependent", "hardware/cloud/API-dependent", _evidence(project_root, "report/plan_d_llm_model_registry_and_backend_audit.md"), "Yes for full paper matrix", "Requires cloud GPU or API path.", "E8 cloud/API feasibility audit only if approved.", "No model pull or cloud run in E0."),
        _exact("LASTFM", "LastFM", "Plan C/D/E0", "extension_only_no_valid_utility_evaluation or blocked unless valid artifacts are found.", "extension_only_no_valid_utility_evaluation", "blocked/conditional", _evidence(project_root, "report/plan_d_global_extended_neural_baselines.md", "results/tables/plan_d_global_extended_dataset_model_coverage.csv"), "Yes for exact four-dataset paper condition", "No valid comparable local utility setup.", "E1 inventory only; no LastFM run until valid artifacts exist.", "Do not claim LastFM reproduction."),
        _exact("TARGET_1000_USERS", "1000-user LLM target", "D5/E0", "Not yet completed; D5 local LLM target was 20 users.", "diagnostic_subset", "conditional/API-dependent", _evidence(project_root, "results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_manifest.csv"), "Yes for main paper LLM scale", "Scale-up cost/runtime and reliability are not yet validated.", "E1/E2/E5 should estimate, prepare, then optionally run.", "1000-user target remains not completed."),
        _exact("CANDIDATE_POOL_20_1_19", "20 candidates with 1 positive plus 19 negatives", "D5", "Matched in D5 diagnostics.", "diagnostic_subset", "near reproduction", _evidence(project_root, "results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_manifest.csv"), "Yes", "Matched only on diagnostic_subset, not all datasets at scale.", "E2 should validate at 1000-user scale for valid datasets.", "This paper condition is already represented locally in D5 diagnostics."),
    ]
    return pd.DataFrame(rows, columns=EXACTNESS_COLUMNS)


def build_scaleup_call_estimates() -> pd.DataFrame:
    """Build deterministic feasibility-level call estimates."""

    rows = [
        _scale("CALLS_20_USERS", 20, "ML-1M", 1, "Qwen2-7B", 1, "OpenP5-like", 1, "ranking", 1, 1, "diagnostic_subset", "Comparable to the small D5 diagnostic scale; no inference run in E0."),
        _scale("CALLS_100_USERS", 100, "ML-1M", 1, "Qwen2-7B", 1, "OpenP5-like", 1, "ranking", 1, 1, "diagnostic_subset", "Intermediate local planning scenario."),
        _scale("CALLS_1000_USERS", 1000, "ML-1M", 1, "Qwen2-7B", 1, "OpenP5-like", 1, "ranking", 1, 1, "near reproduction", "Single dataset, single local model, ranking-only scale target."),
        _scale("CALLS_1000_USERS_3_DATASETS", 1000, "ML-1M; Amazon Beauty; Amazon Sports", 3, "Qwen2-7B", 1, "OpenP5-like", 1, "ranking", 1, 1, "near reproduction", "Three valid reconstructed datasets, ranking-only, one model."),
        _scale("CALLS_1000_USERS_4_DATASETS_LASTFM_CONDITIONAL", 1000, "ML-1M; Amazon Beauty; Amazon Sports; LastFM", 4, "Qwen2-7B", 1, "OpenP5-like", 1, "ranking", 1, 1, "blocked/conditional", "Includes LastFM, which remains conditional/blocked unless valid artifacts are found."),
        _scale("CALLS_RANKING_ONLY", 1000, "ML-1M; Amazon Beauty; Amazon Sports", 3, "Qwen2-7B", 1, "OpenP5-like", 1, "ranking", 1, 1, "near reproduction", "Ranking-only paper-like local scale for valid datasets."),
        _scale("CALLS_RANKING_PLUS_RERANKING", 1000, "ML-1M; Amazon Beauty; Amazon Sports", 3, "Qwen2-7B", 1, "OpenP5-like", 1, "ranking; re-ranking", 2, 1, "near reproduction", "Ranking plus re-ranking doubles calls relative to ranking-only."),
        _scale("CALLS_PROMPT_STRATEGY_MATRIX", 1000, "ML-1M; Amazon Beauty; Amazon Sports", 3, "Qwen2-7B", 1, "OpenP5-like; LLMRank-like; LLMSRec/ICL", 3, "ranking; re-ranking", 2, 1, "conditional/API-dependent", "Prompt matrix after E4 alignment."),
        _scale("CALLS_LOCAL_2_MODEL_MATRIX", 1000, "ML-1M; Amazon Beauty; Amazon Sports", 3, "Qwen2-7B; Mistral-7B", 2, "OpenP5-like", 1, "ranking; re-ranking", 2, 1, "near reproduction", "Local two-model matrix using already available Ollama model names, if approved later."),
        _scale("CALLS_PAPER_7_LLM_MATRIX", 1000, "ML-1M; Amazon Beauty; Amazon Sports", 3, "GPT-3.5; GPT-4o-mini; GPT-4o; Claude 3 Haiku; Llama3-70B; Qwen2-7B; Mistral-7B", 7, "OpenP5-like", 1, "ranking; re-ranking", 2, 1, "conditional/API-dependent", "Seven-model paper matrix without prompt expansion; API and hardware blockers remain."),
        _scale("CALLS_7_LLM_3_PROMPT_3_DATASET_1000_USERS", 1000, "ML-1M; Amazon Beauty; Amazon Sports", 3, "GPT-3.5; GPT-4o-mini; GPT-4o; Claude 3 Haiku; Llama3-70B; Qwen2-7B; Mistral-7B", 7, "OpenP5-like; LLMRank-like; LLMSRec/ICL", 3, "ranking; re-ranking", 2, 1, "conditional/API-dependent", "7 LLM x 3 prompt strategies x 3 valid datasets x 1000 users; full feasibility scenario, not an executed experiment."),
    ]
    return pd.DataFrame(rows, columns=SCALEUP_COLUMNS)


def build_local_runtime_estimates(scaleup: pd.DataFrame | None = None) -> pd.DataFrame:
    """Estimate local runtime from assumed seconds per call; not benchmark results."""

    scaleup = scaleup if scaleup is not None else build_scaleup_call_estimates()
    scopes = [
        ("LOCAL_RUNTIME_20_USER_DIAGNOSTIC", "20-user single-model diagnostic", "CALLS_20_USERS"),
        ("LOCAL_RUNTIME_1000_USERS_3_DATASETS", "1000 users x 3 valid datasets x one local model x ranking", "CALLS_1000_USERS_3_DATASETS"),
        ("LOCAL_RUNTIME_LOCAL_2_MODEL_MATRIX", "local Qwen2/Mistral x 3 valid datasets x ranking + re-ranking", "CALLS_LOCAL_2_MODEL_MATRIX"),
        ("LOCAL_RUNTIME_FULL_PAPER_FEASIBILITY", "paper 7-LLM x 3 prompt x 3 valid dataset feasibility scope", "CALLS_7_LLM_3_PROMPT_3_DATASET_1000_USERS"),
    ]
    seconds_per_call = [10, 20, 45, 90]
    rows: list[dict[str, object]] = []
    for scope_id, model_scope, scale_id in scopes:
        calls = _calls_for(scaleup, scale_id)
        for assumed_seconds in seconds_per_call:
            estimated_seconds = calls * assumed_seconds
            rows.append(
                {
                    "scenario_id": f"{scope_id}_{assumed_seconds}S_PER_CALL",
                    "model_scope": model_scope,
                    "estimated_total_calls": calls,
                    "assumed_seconds_per_call": assumed_seconds,
                    "estimated_seconds": estimated_seconds,
                    "estimated_minutes": round(estimated_seconds / 60, 6),
                    "estimated_hours": round(estimated_seconds / 3600, 6),
                    "estimated_days": round(estimated_seconds / 86400, 6),
                    "feasibility_note": "Planning estimate only; E0 did not run inference or benchmark local models.",
                }
            )
    return pd.DataFrame(rows, columns=LOCAL_RUNTIME_COLUMNS)


def build_api_cost_scenarios(scaleup: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build placeholder API cost scenarios requiring later official price verification."""

    scaleup = scaleup if scaleup is not None else build_scaleup_call_estimates()
    pricing_cases = [
        ("very_low_cost", 1.0),
        ("low_cost", 5.0),
        ("medium_cost", 25.0),
        ("high_cost", 100.0),
    ]
    call_scopes = [
        ("mini_api_100_user_3_dataset_ranking_reranking", 100 * 3 * 1 * 1 * 2),
        ("full_paper_7_llm_3_prompt_3_dataset_1000_user", _calls_for(scaleup, "CALLS_7_LLM_3_PROMPT_3_DATASET_1000_USERS")),
    ]
    rows: list[dict[str, object]] = []
    for call_scope, calls in call_scopes:
        for pricing_id, cost_per_1000 in pricing_cases:
            rows.append(
                {
                    "scenario_id": f"{call_scope}_{pricing_id}",
                    "model_family": "closed_source_api_or_hosted_llm_placeholder",
                    "pricing_status": "hypothetical_manual_input",
                    "assumed_cost_per_1000_calls_usd": cost_per_1000,
                    "estimated_total_calls": calls,
                    "estimated_cost_usd": round((calls / 1000) * cost_per_1000, 6),
                    "requires_current_price_check": "true",
                    "source_status": "manual_placeholder_not_official",
                    "notes": "E0 did not fetch prices or call APIs; Plan E6 must verify current official pricing before any budget claim.",
                }
            )
    return pd.DataFrame(rows, columns=API_COST_COLUMNS)


def build_recommended_roadmap() -> pd.DataFrame:
    """Build the recommended Plan E roadmap."""

    rows = [
        _roadmap("E1", "Full-data inventory and cost/runtime estimator", "Inventory available artifacts, validate which datasets can support paper-like evaluation, and refine estimator inputs.", False, False, False, False, "artifact inventory; refined feasibility ledger; LastFM decision record", "medium", 1, "Stop if valid artifacts for a dataset are missing.", "Audit-only phase; no inference required."),
        _roadmap("E2", "1000-user paper-like candidate pool construction", "Build deterministic 1000-user candidate manifests with 20 candidates, 1 positive, and 19 negatives for valid datasets.", False, False, False, False, "candidate manifests; validation tables; blocker table", "medium", 2, "Stop if candidate construction cannot match paper-like assumptions.", "LastFM remains conditional unless valid artifacts exist."),
        _roadmap("E3", "Paper-like baseline completion", "Align BM25, MostPop, BPRMF, GRU4Rec, SASRec, and LightGCN settings as closely as feasible.", False, False, False, True, "baseline setting comparison; optional metric updates; training summaries", "medium", 3, "Stop before heavy training unless explicitly approved.", "May require training; keep Plan C fixed."),
        _roadmap("E4", "Paper prompt strategy alignment", "Map OpenP5-like, LLMRank-like, and LLMSRec/ICL prompts to project templates and identify deltas.", False, False, False, False, "prompt registry; alignment matrix; exactness notes", "medium", 4, "Stop if prompt text or examples cannot be reconstructed safely.", "Strict candidate-ID-only prompt remains extension."),
        _roadmap("E5", "Full local LLM scale-up", "Run local Qwen2/Mistral scale-up only after E2/E4 and explicit approval.", True, False, False, False, "local LLM metrics; raw outputs under ignored results/raw_outputs; reliability tables", "high", 5, "Stop if runtime, validity, or hardware limits are unacceptable.", "Not part of E0; no Ollama calls without approval."),
        _roadmap("E6", "Closed-source/API model availability and cost audit", "Verify current official availability and pricing for API-dependent models.", False, False, False, False, "official price/availability audit; budget scenarios; approval checklist", "high", 6, "Stop if availability, policy, or budget blocks exact/near reproduction.", "E0 placeholder costs are not official price claims."),
        _roadmap("E7", "Mini API reproduction, only if approved", "Run a bounded API diagnostic only with explicit approval, keys, budget, and price verification.", True, True, False, False, "mini API outputs; cost log; metrics; limitation notes", "high", 7, "Stop immediately if approval or budget is absent.", "No default API calls."),
        _roadmap("E8", "Llama3 70B cloud/API feasibility", "Assess whether Llama3-70B can be evaluated through cloud GPU or hosted API.", False, True, True, False, "hardware/API feasibility report; cost/runtime estimate", "high", 8, "Stop if cloud/API path is not approved.", "hardware/cloud/API-dependent."),
        _roadmap("E9", "Generation-involved performance", "Implement profile-generation experiments only after prompt alignment and inference approval.", True, False, False, False, "profile-generation manifests; metrics; failure analysis", "high", 9, "Stop if profile prompts or outputs cannot be validated.", "No generation-involved experiment is run in E0."),
        _roadmap("E10", "Beyond-utility completion", "Complete hallucination, position bias, history length, novelty, popularity, fairness, and candidate validity reporting where valid.", False, False, False, False, "beyond-utility summary tables; plots from CSV; claim-safety notes", "medium", 10, "Stop if definitions or required attributes are invalid.", "Candidate validity remains a project extension."),
    ]
    return pd.DataFrame(rows, columns=ROADMAP_COLUMNS)


def build_all_tables(project_root: Path = PROJECT_ROOT) -> dict[str, pd.DataFrame]:
    """Build every Plan E0 table without side effects."""

    alignment = build_paper_condition_alignment_matrix(project_root)
    missing = build_missing_conditions_table(project_root)
    exactness = build_exactness_feasibility_table(project_root)
    scaleup = build_scaleup_call_estimates()
    runtime = build_local_runtime_estimates(scaleup)
    api_cost = build_api_cost_scenarios(scaleup)
    roadmap = build_recommended_roadmap()
    return {
        "alignment_matrix": alignment,
        "missing_conditions": missing,
        "exactness_feasibility": exactness,
        "scaleup_call_estimates": scaleup,
        "local_runtime_estimates": runtime,
        "api_cost_scenarios": api_cost,
        "recommended_roadmap": roadmap,
    }


def build_markdown_report(tables: dict[str, pd.DataFrame]) -> str:
    """Build the Plan E0 Markdown report from generated table data."""

    alignment = tables["alignment_matrix"]
    missing = tables["missing_conditions"]
    exactness = tables["exactness_feasibility"]
    scaleup = tables["scaleup_call_estimates"]
    runtime = tables["local_runtime_estimates"]
    api_cost = tables["api_cost_scenarios"]
    roadmap = tables["recommended_roadmap"]

    full_matrix_calls = _calls_for(scaleup, "CALLS_7_LLM_3_PROMPT_3_DATASET_1000_USERS")
    local_matrix_calls = _calls_for(scaleup, "CALLS_LOCAL_2_MODEL_MATRIX")
    max_runtime = runtime.loc[runtime["scenario_id"] == "LOCAL_RUNTIME_FULL_PAPER_FEASIBILITY_90S_PER_CALL"].iloc[0]
    max_api = api_cost.loc[api_cost["scenario_id"] == "full_paper_7_llm_3_prompt_3_dataset_1000_user_high_cost"].iloc[0]

    return f"""# Plan E0 Paper-Condition Alignment and Scale-Up Feasibility Audit

## Executive Summary

Plan E0 is an audit-only phase for maximum feasible alignment with the paper conditions of Beyond Utility: Evaluating LLM as Recommender. It does not claim exact reproduction. It identifies which paper conditions are already reconstructed, which are diagnostic_subset or extended_subset work, and which remain conditional, blocked, hardware/cloud/API-dependent, or extension-only.

E0 performed no inference. E0 made no Ollama calls. E0 made no Qwen2/Mistral calls. E0 made no API calls. E0 performed no downloads. E0 performed no model pulls. E0 performed no training. E0 performed no LastFM run. E0 performed no Plan C modification. E0 performed no destructive cleanup.

Plan E is about maximum feasible alignment, not overclaiming exact reproduction. No full exact reproduction has been achieved yet. The project has reproduced or reconstructed only the accessible parts and audited the inaccessible parts.

## Paper Condition Overview

The E0 matrix covers four datasets: Amazon Beauty, Amazon Sports, ML-1M, and LastFM. It covers ranking and re-ranking tasks, HR@5 and NDCG@5, the 20-item candidate pool with 1 positive and 19 negatives, the 1000-test-user paper scale, six baseline/recommender families, seven LLMs, three prompt strategy families, and beyond-utility dimensions.

Core LLM paper matrix:

- GPT-3.5
- GPT-4o-mini
- GPT-4o
- Claude 3 Haiku
- Llama3-70B
- Qwen2-7B
- Mistral-7B

Prompt strategies:

- OpenP5-like
- LLMRank-like
- LLMSRec/ICL

Beyond-utility dimensions:

- hallucination
- candidate position bias
- history length sensitivity
- generation-involved performance/user profiles
- novelty
- popularity
- fairness
- candidate validity as a project extension

## Current Project Status Overview

Plan C is classified as reconstructed reproduction and must remain unchanged. Plan D extended neural baselines are classified as extended_subset, with near reproduction only where the method family is accessible but the exact paper input and training conditions are not guaranteed. D5 local Qwen2/Mistral diagnostics are diagnostic_subset/local extension work.

Strict candidate-ID-only prompting is a project extension, not a paper-exact prompt strategy. Local Qwen2/Mistral 20-user diagnostics are diagnostic_subset results. Closed-source models remain conditional/API-dependent. Llama3-70B is hardware/cloud/API-dependent. LastFM remains extension_only_no_valid_utility_evaluation or blocked unless valid artifacts are found.

## Exactness and Feasibility Classification

{_markdown_table(exactness, ["component_id", "exactness_class", "feasibility_class", "current_project_status"], max_rows=20)}

## Missing Conditions and Required Actions

The missing-condition table contains {len(missing)} rows. The highest-risk gaps are full 1000-user local LLM scale-up, exact prompt strategy alignment, closed-source/API-dependent models, Claude 3 Haiku availability, Llama3-70B hardware/cloud/API dependency, LastFM exact evaluation, and generation-involved profile experiments.

{_markdown_table(missing, ["missing_condition_id", "feasibility_class", "exactness_class", "plan_e_phase"], max_rows=20)}

## Scale-Up Estimates

E0 estimates call counts at feasibility level only. It does not estimate exact token counts and does not call any model or external service. The largest paper-like scenario in this audit is {full_matrix_calls} calls for 7 LLM x 3 prompt strategies x 3 valid datasets x 1000 users x ranking/re-ranking. The local Qwen2/Mistral two-model matrix is estimated at {local_matrix_calls} calls.

{_markdown_table(scaleup, ["scenario_id", "users_per_dataset", "dataset_count", "model_count", "prompt_strategy_count", "task_count", "estimated_total_calls", "classification"], max_rows=20)}

## Local Runtime Estimate Interpretation

Local runtime estimates use assumed seconds-per-call values of 10, 20, 45, and 90 seconds. These are planning assumptions, not measured benchmark results. Under the largest audited paper-like scope and 90 seconds per call, the estimate is {max_runtime["estimated_hours"]} hours, or {max_runtime["estimated_days"]} days.

{_markdown_table(runtime, ["scenario_id", "estimated_total_calls", "assumed_seconds_per_call", "estimated_hours", "estimated_days"], max_rows=16)}

## API Cost Estimate Interpretation

API cost rows use placeholder manual-input assumptions only. E0 did not fetch current prices from the web and did not call APIs. Closed-source/API costs require current official price verification in Plan E6 before any budget claim. Under the largest audited paper-like scope and the high placeholder scenario, the placeholder estimate is USD {max_api["estimated_cost_usd"]}; this is not an official price claim.

{_markdown_table(api_cost, ["scenario_id", "assumed_cost_per_1000_calls_usd", "estimated_total_calls", "estimated_cost_usd", "requires_current_price_check"], max_rows=12)}

## Recommended Plan E Roadmap

{_markdown_table(roadmap, ["phase_id", "title", "requires_inference", "requires_api", "requires_download", "requires_training", "risk_level"], max_rows=20)}

## Claim-Safety Notes for Oral Defense

- Do not claim exact reproduction.
- Do not claim full LLM reproduction.
- Do not claim closed-source models were reproduced.
- Do not claim LastFM was reproduced.
- Do not claim strict candidate-ID-only prompting is paper-exact.
- Say Plan C is reconstructed reproduction.
- Say Plan D neural baselines are extended_subset and near reproduction only for accessible method families.
- Say D5 Qwen2/Mistral results are local diagnostic_subset results on a 20-user ML-1M diagnostic setting.
- Say candidate validity is a project extension that helps explain reliability beyond HR@5/NDCG@5.
- Say API, cloud, and 70B results are conditional until explicitly approved and executed.

## Next Recommended Phase

The next recommended phase is E1 - Full-data inventory and cost/runtime estimator. E1 should inventory all available artifacts, validate which datasets can support paper-like 1000-user evaluation, keep LastFM blocked unless valid artifacts are found, and refine call/runtime/cost estimates without running inference.

## Generated Outputs

- `results/tables/plan_e_paper_condition_alignment_matrix.csv`
- `results/tables/plan_e_missing_conditions.csv`
- `results/tables/plan_e_exactness_feasibility.csv`
- `results/tables/plan_e_scaleup_call_estimates.csv`
- `results/tables/plan_e_local_runtime_estimates.csv`
- `results/tables/plan_e_api_cost_scenarios.csv`
- `results/tables/plan_e_recommended_roadmap.csv`
- `report/plan_e_paper_condition_alignment_audit.md`

## Audit Boundary

All E0 outputs are deterministic planning artifacts. They are based on repository evidence and explicit planning assumptions, not new inference, not API calls, not downloads, not model pulls, not training, and not LastFM execution. No Plan C files and no original_repo files are modified by the E0 build.
"""


def write_audit_outputs(
    tables: dict[str, pd.DataFrame] | None = None,
    paths: PlanEAuditPaths | None = None,
) -> dict[str, pd.DataFrame]:
    """Write all Plan E0 outputs and return the generated table data."""

    tables = tables if tables is not None else build_all_tables()
    paths = paths or PlanEAuditPaths()
    paths.alignment_matrix.parent.mkdir(parents=True, exist_ok=True)
    paths.report.parent.mkdir(parents=True, exist_ok=True)

    _write_csv(tables["alignment_matrix"], paths.alignment_matrix, ALIGNMENT_COLUMNS)
    _write_csv(tables["missing_conditions"], paths.missing_conditions, MISSING_COLUMNS)
    _write_csv(tables["exactness_feasibility"], paths.exactness_feasibility, EXACTNESS_COLUMNS)
    _write_csv(tables["scaleup_call_estimates"], paths.scaleup_call_estimates, SCALEUP_COLUMNS)
    _write_csv(tables["local_runtime_estimates"], paths.local_runtime_estimates, LOCAL_RUNTIME_COLUMNS)
    _write_csv(tables["api_cost_scenarios"], paths.api_cost_scenarios, API_COST_COLUMNS)
    _write_csv(tables["recommended_roadmap"], paths.recommended_roadmap, ROADMAP_COLUMNS)
    paths.report.write_text(build_markdown_report(tables), encoding="utf-8", newline="\n")
    return tables


def _write_csv(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, columns=columns, lineterminator="\n")


def _alignment(*values: object) -> dict[str, str]:
    return {column: _text(value) for column, value in zip(ALIGNMENT_COLUMNS, values)}


def _missing(*values: object) -> dict[str, str]:
    return {column: _text(value) for column, value in zip(MISSING_COLUMNS, values)}


def _exact(*values: object) -> dict[str, str]:
    return {column: _text(value) for column, value in zip(EXACTNESS_COLUMNS, values)}


def _scale(
    scenario_id: str,
    users_per_dataset: int,
    datasets: str,
    dataset_count: int,
    models: str,
    model_count: int,
    prompt_strategies: str,
    prompt_strategy_count: int,
    tasks: str,
    task_count: int,
    calls_per_user_task: int,
    classification: str,
    notes: str,
) -> dict[str, object]:
    return {
        "scenario_id": scenario_id,
        "users_per_dataset": users_per_dataset,
        "datasets": datasets,
        "dataset_count": dataset_count,
        "models": models,
        "model_count": model_count,
        "prompt_strategies": prompt_strategies,
        "prompt_strategy_count": prompt_strategy_count,
        "tasks": tasks,
        "task_count": task_count,
        "calls_per_user_task": calls_per_user_task,
        "estimated_total_calls": users_per_dataset * dataset_count * model_count * prompt_strategy_count * task_count * calls_per_user_task,
        "classification": classification,
        "notes": notes,
    }


def _roadmap(
    phase_id: str,
    title: str,
    objective: str,
    requires_inference: bool,
    requires_api: bool,
    requires_download: bool,
    requires_training: bool,
    expected_outputs: str,
    risk_level: str,
    recommended_order: int,
    stop_condition: str,
    notes: str,
) -> dict[str, object]:
    return {
        "phase_id": phase_id,
        "title": title,
        "objective": objective,
        "requires_inference": _bool_text(requires_inference),
        "requires_api": _bool_text(requires_api),
        "requires_download": _bool_text(requires_download),
        "requires_training": _bool_text(requires_training),
        "expected_outputs": expected_outputs,
        "risk_level": risk_level,
        "recommended_order": recommended_order,
        "stop_condition": stop_condition,
        "notes": notes,
    }


def _calls_for(scaleup: pd.DataFrame, scenario_id: str) -> int:
    rows = scaleup.loc[scaleup["scenario_id"] == scenario_id]
    if rows.empty:
        raise ValueError(f"Missing scale-up scenario: {scenario_id}")
    return int(rows.iloc[0]["estimated_total_calls"])


def _evidence(project_root: Path, *relative_paths: str) -> str:
    statuses: list[str] = []
    for relative_path in relative_paths:
        path = project_root / relative_path
        status = "available" if path.exists() else "missing"
        statuses.append(f"{status}: {relative_path}")
    return "; ".join(statuses)


def _markdown_table(frame: pd.DataFrame, columns: Iterable[str], *, max_rows: int) -> str:
    selected_columns = list(columns)
    shown = frame[selected_columns].head(max_rows).copy()
    lines = [
        "| " + " | ".join(selected_columns) + " |",
        "| " + " | ".join("---" for _ in selected_columns) + " |",
    ]
    for _, row in shown.iterrows():
        values = [_escape_markdown(_text(row[column])) for column in selected_columns]
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        lines.append("| " + " | ".join("..." for _ in selected_columns) + " |")
    return "\n".join(lines)


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _text(value: object) -> str:
    if isinstance(value, bool):
        return _bool_text(value)
    return str(value)
