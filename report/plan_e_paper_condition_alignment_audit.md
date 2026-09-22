# Plan E0 Paper-Condition Alignment and Scale-Up Feasibility Audit

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

| component_id | exactness_class | feasibility_class | current_project_status |
| --- | --- | --- | --- |
| PLAN_C | reconstructed reproduction | near reproduction | Completed and stable; tag plan-c-reconstruction-v1. |
| PLAN_D_NEURAL_EXTENDED | extended_subset | near reproduction | BPRMF, LightGCN, SASRec, and GRU4Rec completed on ML-1M, Beauty, and Sports. |
| D5_QWEN_MISTRAL_LOCAL | diagnostic_subset | near reproduction | Local Qwen2 and Mistral diagnostics completed on ML-1M 20 users. |
| STRICT_PROMPT | extension | near reproduction | Completed and effective for grounding in diagnostic_subset. |
| CLOSED_SOURCE_GPT_CLAUDE | conditional/API-dependent | conditional/API-dependent | Not run. |
| CLAUDE3_HAIKU | blocked/conditional | blocked/conditional | Blocked/conditional if retired or unavailable. |
| LLAMA3_70B | hardware/cloud/API-dependent | hardware/cloud/API-dependent | Not locally feasible on RTX 4060 Laptop GPU, 8 GB VRAM. |
| LASTFM | extension_only_no_valid_utility_evaluation | blocked/conditional | extension_only_no_valid_utility_evaluation or blocked unless valid artifacts are found. |
| TARGET_1000_USERS | diagnostic_subset | conditional/API-dependent | Not yet completed; D5 local LLM target was 20 users. |
| CANDIDATE_POOL_20_1_19 | diagnostic_subset | near reproduction | Matched in D5 diagnostics. |

## Missing Conditions and Required Actions

The missing-condition table contains 13 rows. The highest-risk gaps are full 1000-user local LLM scale-up, exact prompt strategy alignment, closed-source/API-dependent models, Claude 3 Haiku availability, Llama3-70B hardware/cloud/API dependency, LastFM exact evaluation, and generation-involved profile experiments.

| missing_condition_id | feasibility_class | exactness_class | plan_e_phase |
| --- | --- | --- | --- |
| MISS_FULL_1000_LOCAL_LLM | conditional/API-dependent | diagnostic_subset | E5 |
| MISS_1000_CANDIDATE_MANIFESTS | near reproduction | blocked/conditional | E2 |
| MISS_EXACT_PROMPTS | conditional/API-dependent | blocked/conditional | E4 |
| MISS_CLOSED_SOURCE_LLMS | conditional/API-dependent | conditional/API-dependent | E6 |
| MISS_CLAUDE3_HAIKU | blocked/conditional | blocked/conditional | E6 |
| MISS_LLAMA3_70B | hardware/cloud/API-dependent | hardware/cloud/API-dependent | E8 |
| MISS_LASTFM_EXACT_EVAL | blocked/conditional | extension_only_no_valid_utility_evaluation | E1 |
| MISS_API_DEPENDENT_MODELS | conditional/API-dependent | conditional/API-dependent | E6 |
| MISS_FULL_PAPER_LLM_MATRIX | conditional/API-dependent | blocked/conditional | E5 |
| MISS_FULL_RANKING_RERANKING_MATRIX | conditional/API-dependent | diagnostic_subset | E5 |
| MISS_GENERATION_INVOLVED_PROFILES | conditional/API-dependent | blocked/conditional | E9 |
| MISS_NOVELTY_POPULARITY_FAIRNESS | blocked/conditional | extension | E10 |
| MISS_OFFICIAL_PREPROCESSED_ARTIFACTS | blocked/conditional | blocked/conditional | E1 |

## Scale-Up Estimates

E0 estimates call counts at feasibility level only. It does not estimate exact token counts and does not call any model or external service. The largest paper-like scenario in this audit is 126000 calls for 7 LLM x 3 prompt strategies x 3 valid datasets x 1000 users x ranking/re-ranking. The local Qwen2/Mistral two-model matrix is estimated at 12000 calls.

| scenario_id | users_per_dataset | dataset_count | model_count | prompt_strategy_count | task_count | estimated_total_calls | classification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CALLS_20_USERS | 20 | 1 | 1 | 1 | 1 | 20 | diagnostic_subset |
| CALLS_100_USERS | 100 | 1 | 1 | 1 | 1 | 100 | diagnostic_subset |
| CALLS_1000_USERS | 1000 | 1 | 1 | 1 | 1 | 1000 | near reproduction |
| CALLS_1000_USERS_3_DATASETS | 1000 | 3 | 1 | 1 | 1 | 3000 | near reproduction |
| CALLS_1000_USERS_4_DATASETS_LASTFM_CONDITIONAL | 1000 | 4 | 1 | 1 | 1 | 4000 | blocked/conditional |
| CALLS_RANKING_ONLY | 1000 | 3 | 1 | 1 | 1 | 3000 | near reproduction |
| CALLS_RANKING_PLUS_RERANKING | 1000 | 3 | 1 | 1 | 2 | 6000 | near reproduction |
| CALLS_PROMPT_STRATEGY_MATRIX | 1000 | 3 | 1 | 3 | 2 | 18000 | conditional/API-dependent |
| CALLS_LOCAL_2_MODEL_MATRIX | 1000 | 3 | 2 | 1 | 2 | 12000 | near reproduction |
| CALLS_PAPER_7_LLM_MATRIX | 1000 | 3 | 7 | 1 | 2 | 42000 | conditional/API-dependent |
| CALLS_7_LLM_3_PROMPT_3_DATASET_1000_USERS | 1000 | 3 | 7 | 3 | 2 | 126000 | conditional/API-dependent |

## Local Runtime Estimate Interpretation

Local runtime estimates use assumed seconds-per-call values of 10, 20, 45, and 90 seconds. These are planning assumptions, not measured benchmark results. Under the largest audited paper-like scope and 90 seconds per call, the estimate is 3150.0 hours, or 131.25 days.

| scenario_id | estimated_total_calls | assumed_seconds_per_call | estimated_hours | estimated_days |
| --- | --- | --- | --- | --- |
| LOCAL_RUNTIME_20_USER_DIAGNOSTIC_10S_PER_CALL | 20 | 10 | 0.055556 | 0.002315 |
| LOCAL_RUNTIME_20_USER_DIAGNOSTIC_20S_PER_CALL | 20 | 20 | 0.111111 | 0.00463 |
| LOCAL_RUNTIME_20_USER_DIAGNOSTIC_45S_PER_CALL | 20 | 45 | 0.25 | 0.010417 |
| LOCAL_RUNTIME_20_USER_DIAGNOSTIC_90S_PER_CALL | 20 | 90 | 0.5 | 0.020833 |
| LOCAL_RUNTIME_1000_USERS_3_DATASETS_10S_PER_CALL | 3000 | 10 | 8.333333 | 0.347222 |
| LOCAL_RUNTIME_1000_USERS_3_DATASETS_20S_PER_CALL | 3000 | 20 | 16.666667 | 0.694444 |
| LOCAL_RUNTIME_1000_USERS_3_DATASETS_45S_PER_CALL | 3000 | 45 | 37.5 | 1.5625 |
| LOCAL_RUNTIME_1000_USERS_3_DATASETS_90S_PER_CALL | 3000 | 90 | 75.0 | 3.125 |
| LOCAL_RUNTIME_LOCAL_2_MODEL_MATRIX_10S_PER_CALL | 12000 | 10 | 33.333333 | 1.388889 |
| LOCAL_RUNTIME_LOCAL_2_MODEL_MATRIX_20S_PER_CALL | 12000 | 20 | 66.666667 | 2.777778 |
| LOCAL_RUNTIME_LOCAL_2_MODEL_MATRIX_45S_PER_CALL | 12000 | 45 | 150.0 | 6.25 |
| LOCAL_RUNTIME_LOCAL_2_MODEL_MATRIX_90S_PER_CALL | 12000 | 90 | 300.0 | 12.5 |
| LOCAL_RUNTIME_FULL_PAPER_FEASIBILITY_10S_PER_CALL | 126000 | 10 | 350.0 | 14.583333 |
| LOCAL_RUNTIME_FULL_PAPER_FEASIBILITY_20S_PER_CALL | 126000 | 20 | 700.0 | 29.166667 |
| LOCAL_RUNTIME_FULL_PAPER_FEASIBILITY_45S_PER_CALL | 126000 | 45 | 1575.0 | 65.625 |
| LOCAL_RUNTIME_FULL_PAPER_FEASIBILITY_90S_PER_CALL | 126000 | 90 | 3150.0 | 131.25 |

## API Cost Estimate Interpretation

API cost rows use placeholder manual-input assumptions only. E0 did not fetch current prices from the web and did not call APIs. Closed-source/API costs require current official price verification in Plan E6 before any budget claim. Under the largest audited paper-like scope and the high placeholder scenario, the placeholder estimate is USD 12600.0; this is not an official price claim.

| scenario_id | assumed_cost_per_1000_calls_usd | estimated_total_calls | estimated_cost_usd | requires_current_price_check |
| --- | --- | --- | --- | --- |
| mini_api_100_user_3_dataset_ranking_reranking_very_low_cost | 1.0 | 600 | 0.6 | true |
| mini_api_100_user_3_dataset_ranking_reranking_low_cost | 5.0 | 600 | 3.0 | true |
| mini_api_100_user_3_dataset_ranking_reranking_medium_cost | 25.0 | 600 | 15.0 | true |
| mini_api_100_user_3_dataset_ranking_reranking_high_cost | 100.0 | 600 | 60.0 | true |
| full_paper_7_llm_3_prompt_3_dataset_1000_user_very_low_cost | 1.0 | 126000 | 126.0 | true |
| full_paper_7_llm_3_prompt_3_dataset_1000_user_low_cost | 5.0 | 126000 | 630.0 | true |
| full_paper_7_llm_3_prompt_3_dataset_1000_user_medium_cost | 25.0 | 126000 | 3150.0 | true |
| full_paper_7_llm_3_prompt_3_dataset_1000_user_high_cost | 100.0 | 126000 | 12600.0 | true |

## Recommended Plan E Roadmap

| phase_id | title | requires_inference | requires_api | requires_download | requires_training | risk_level |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | Full-data inventory and cost/runtime estimator | false | false | false | false | medium |
| E2 | 1000-user paper-like candidate pool construction | false | false | false | false | medium |
| E3 | Paper-like baseline completion | false | false | false | true | medium |
| E4 | Paper prompt strategy alignment | false | false | false | false | medium |
| E5 | Full local LLM scale-up | true | false | false | false | high |
| E6 | Closed-source/API model availability and cost audit | false | false | false | false | high |
| E7 | Mini API reproduction, only if approved | true | true | false | false | high |
| E8 | Llama3 70B cloud/API feasibility | false | true | true | false | high |
| E9 | Generation-involved performance | true | false | false | false | high |
| E10 | Beyond-utility completion | false | false | false | false | medium |

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
