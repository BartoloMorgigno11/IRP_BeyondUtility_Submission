# Phase F6M Reproduction Closure

Historical status note: this report freezes the F6M reproduction-only closure. Statements that the future extension had not started are F6M-era status statements and are superseded at the repository level by the later frozen G1-G4 extension artifacts.

Decision token: `f6m_reproduction_closure_complete_with_documented_limitations`.

## Closure Status

REPRODUCIBILITY CORE - CLOSED.

FUTURE EXTENSION - NOT STARTED at F6M closure; superseded for repository-wide status by later G1-G4 extension freezes.

F6M is an offline consolidation phase. It reads existing generated CSV/JSON evidence and does not run LLM inference, APIs, downloads, recommender training, full LastFM extension, raw-output repair, or deduplication.

## LastFM Final Status

| component | task | evidence_status | run_id | completed_final_responses | positive_present_rows | primary_metric_policy |
| --- | --- | --- | --- | --- | --- | --- |
| local_llm_ranking_v4_freeze | ranking | completed_and_frozen | 20260804_131959 | 4000 | not_applicable_for_ranking | fail_strict_all_1000_users_per_cell |
| local_llm_reranking_v4_1_execution | reranking | completed | 20260807_111348 | 4000 | 314 | fail_strict_all_1000_users_per_cell; positive_present_only metrics diagnostic |
| full_lastfm_extension | future extension | not_started |  | 0 |  | outside_f6m_closure |

## LastFM Primary Utility

Ranking and reranking are separate protocols. The reranking positive-present condition is a reranking-pool diagnostic only and is not applied to ranking.

| model_name | prompting_strategy | hr_at_5 | ndcg_at_5 | denominator |
| --- | --- | --- | --- | --- |
| mistral:7b-instruct | LLMRank recency | 0.319 | 0.23287621762 | 1000 |
| mistral:7b-instruct | base/OpenP5 title-list | 0.307 | 0.223073398036 | 1000 |
| qwen2:7b-instruct | LLMRank recency | 0.419 | 0.308222730898 | 1000 |
| qwen2:7b-instruct | base/OpenP5 title-list | 0.458 | 0.323647507835 | 1000 |

| model_name | prompting_strategy | hr_at_5 | ndcg_at_5 | denominator | positive_present_rows |
| --- | --- | --- | --- | --- | --- |
| mistral:7b-instruct | LLMRank recency-focused reranking | 0.151 | 0.101497110768 | 1000 | 314 |
| mistral:7b-instruct | base/OpenP5-derived reranking | 0.154 | 0.116216279854 | 1000 | 314 |
| qwen2:7b-instruct | LLMRank recency-focused reranking | 0.143 | 0.102618646645 | 1000 | 314 |
| qwen2:7b-instruct | base/OpenP5-derived reranking | 0.143 | 0.101374569024 | 1000 | 314 |

## Diagnostic Positive-Present Reranking

| model_name | prompting_strategy | hr_at_5 | ndcg_at_5 | positive_present_rows |
| --- | --- | --- | --- | --- |
| mistral:7b-instruct | LLMRank recency-focused reranking | 0.480891719745 | 0.323239206267 | 314 |
| mistral:7b-instruct | base/OpenP5-derived reranking | 0.490445859873 | 0.370115540935 | 314 |
| qwen2:7b-instruct | LLMRank recency-focused reranking | 0.455414012739 | 0.326810976577 | 314 |
| qwen2:7b-instruct | base/OpenP5-derived reranking | 0.455414012739 | 0.322848945936 | 314 |

## Claims

| claim_id | claim_status | allowed_wording | prohibited_wording |
| --- | --- | --- | --- |
| F6M-CLAIM-001 | supported | REPRODUCIBILITY CORE - CLOSED. | All paper results were exactly replicated. |
| F6M-CLAIM-002 | supported | LastFM ranking is complete under the frozen F6L operationalization. | LastFM ranking exactly reproduces the paper's full protocol. |
| F6M-CLAIM-003 | supported | LastFM reranking V4.1 completed 4000/4000 aggregate final responses. | The positive-present diagnostic subset is the primary reranking denominator. |
| F6M-CLAIM-004 | supported | FUTURE EXTENSION - NOT STARTED at F6M closure; superseded for repository-wide status by later G1-G4 extension freezes. | F6M extension results exist. |

## Limitations

| area | status | mitigation_or_boundary |
| --- | --- | --- |
| paper artifact availability | open | Use exact/reconstructed/adapted labels and prohibit exact numeric claims. |
| model identity | bounded | Label all local LLM rows as substitute-model evidence. |
| cohort | bounded | State the 1000-user operationalized scope in every LastFM claim. |
| candidate pool | bounded | Keep all-1000 fail-strict metrics primary; mark positive-present metrics diagnostic. |
| statistics | bounded | Do not report p-values where paired per-user contrasts are not extracted. |
| scope control | closed_for_f6m | Stop after the reproduction closure commit. |

## Validation

| gate | status | result |
| --- | --- | --- |
| precondition_collect | pass | 1949 tests collected before F6M files were added |
| precondition_f6l_focused | pass | 13 passed |
| f6m_focused | pass | 5 passed |
| f6l_affected | pass | 17 passed |
| documentation_and_evidence | pass | 107 passed |
| postchange_collect | pass | 1954 tests collected |
| cleanup_current | residual_historical_suite_failure | timed out after 1200 seconds with historical protected-input guard failures; first isolated failure: tests/test_phase_f6i2g_first_python_consolidation_execution.py reports f6i2g_failed_protected_input_change after intentional F6M edits |
| broad_noncleanup | residual_historical_suite_failure | timed out after 1200 seconds; first isolated failure under -x is tests/test_exactness_frontier_and_evidence_closure.py::test_required_schemas_and_exact_row_counts due older direct script path ModuleNotFoundError: No module named src |

## Generated Artifacts

- `results/tables/phase_f6m_artifact_index.csv`
- `results/tables/phase_f6m_input_validation.csv`
- `results/tables/phase_f6m_master_reproducibility_matrix.csv`
- `results/tables/phase_f6m_master_results_long.csv`
- `results/tables/phase_f6m_master_results_wide.csv`
- `results/tables/phase_f6m_research_question_matrix.csv`
- `results/tables/phase_f6m_paper_claim_evidence_matrix.csv`
- `results/tables/phase_f6m_limitations_register.csv`
- `results/tables/phase_f6m_lastfm_final_status.csv`
- `results/tables/phase_f6m_statistical_assessment.csv`
- `results/tables/phase_f6m_reproduction_extension_boundary.csv`
- `report/phase_f6m_reproduction_closure.md`
- `results/manifests/phase_f6m_reproduction_closure_manifest.json`

## Next Step

Design the original extension starting from the frozen reproduction commit.
