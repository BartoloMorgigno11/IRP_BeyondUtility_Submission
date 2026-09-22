# Plan E1 Full-Data Inventory and Cost/Runtime Estimator

## Executive Summary

E1 is an inventory and estimation phase only. It reads repository artifacts, summarizes available datasets, manifests, metrics, raw-output hygiene, and scale-up readiness, and estimates call/runtime planning scenarios for future Plan E phases.

E1 performed no inference. E1 made no Ollama calls. E1 made no Qwen2/Mistral calls. E1 made no API calls. E1 performed no downloads. E1 performed no model pulls. E1 performed no training. E1 performed no LastFM run. E1 performed no Plan C modification. E1 performed no original_repo modification. E1 performed no destructive cleanup.

E1 does not prove that 1000-user paper-like evaluation is complete. E1 prepares E2 by identifying what manifests need to be built. No full exact reproduction is claimed.

## Dataset Inventory Summary

| dataset | exists | file_count | estimated_users_if_available | estimated_items_if_available | estimated_interactions_if_available | has_candidate_manifest | has_20_user_diagnostic_manifest | has_1000_user_manifest | lastfm_valid_for_utility_evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ML-1M | true | 106 | 6040 | 3883 | 1000209 | true | true | false | true |
| Amazon Beauty | true | 48 | 22363 | 259204 | 198502 | true | false | false | true |
| Amazon Sports | true | 46 | 35598 | 532197 | 296337 | true | false | false | true |
| LastFM | true | 6 | 1892 | 17632 | 92834 | false | false | false | false |

LastFM status remains false for valid utility evaluation. LastFM remains conditional/blocked unless valid candidate, ground-truth, and split artifacts are found.

## Manifest Inventory Summary

| manifest_id | dataset | task | rows | user_count_if_available | candidate_pool_size_if_available | matches_paper_candidate_pool_20 | matches_1_positive_19_negative | scale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data__processed__ml1m__candidates_csv | ML-1M | ranking | 604000 | 6040 | 100 | false | false | 1000_user_or_larger_nonpaper_pool |
| data__reconstructed__plan_c__amazon_beauty__candidates_csv | Amazon Beauty | ranking | 2236300 | 22363 | 100 | false | false | 1000_user_or_larger_nonpaper_pool |
| data__reconstructed__plan_c__amazon_sports__candidates_csv | Amazon Sports | ranking | 3559800 | 35598 | 100 | false | false | 1000_user_or_larger_nonpaper_pool |
| data__reconstructed__plan_c__ml1m__candidates_csv | ML-1M | ranking | 604000 | 6040 | 100 | false | false | 1000_user_or_larger_nonpaper_pool |
| results__tables__plan_d_llm_ml1m_history_length_manifest_csv | ML-1M | ranking | 120 | 20 | 20 | true | true | 20_user_diagnostic |
| results__tables__plan_d_llm_ml1m_reranking_manifest_csv | ML-1M | reranking | 60 | 20 | 20 | true | true | 20_user_diagnostic |
| results__tables__plan_d_qwen2_ml1m_candidate_position_bias_manifest_csv | ML-1M | ranking | 60 | 20 | 20 | true | true | 20_user_diagnostic |
| results__tables__plan_d_qwen2_ml1m_ranking_diagnostic_manifest_csv | ML-1M | ranking | 20 | 20 | 20 | true | true | 20_user_diagnostic |
| results__tables__plan_d_qwen2_ml1m_strict_prompt_manifest_csv | ML-1M | ranking | 60 | 20 | 20 | true | true | 20_user_diagnostic |
| results__tables__plan_d_qwen_vs_mistral_ml1m_manifest_csv | ML-1M | ranking | 20 | 20 | 20 | true | true | 20_user_diagnostic |

The current repository contains 20-user D5 diagnostic manifests for ML-1M. Existing reconstructed Plan C candidate files are useful inputs, but their candidate pools are not the paper-like 20-item LLM manifests required for E2.

## Metric Inventory Summary

| artifact_id | phase | dataset | contains_hr_at_5 | contains_ndcg_at_5 | contains_validity | contains_hallucination | classification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| report__oral_exam_plan_c_guide_md | Plan C | all | true | true | false | false | reconstructed reproduction |
| report__plan_c_artifact_recovery_protocol_md | Plan C | all | false | false | false | false | reconstructed reproduction |
| report__plan_c_author_contact_request_md | Plan C | all | false | false | false | false | reconstructed reproduction |
| report__plan_c_external_source_recovery_report_md | Plan C | all | true | true | false | false | reconstructed reproduction |
| report__plan_c_final_report_md | Plan C | all | true | true | true | false | reconstructed reproduction |
| report__plan_c_full_replication_matrix_md | Plan C | all | true | true | false | true | reconstructed reproduction |
| report__plan_c_manual_artifact_intake_guide_md | Plan C | all | false | false | false | false | reconstructed reproduction |
| report__plan_c_manual_download_checklist_md | Plan C | all | false | false | false | false | reconstructed reproduction |
| report__plan_d_amazon_bprmf_lightgcn_extended_summary_md | Plan D/D5 | all | true | true | false | false | extended_subset |
| report__plan_d_exam_readiness_review_md | Plan D/D5 | all | false | false | true | true | extended_subset |
| report__plan_d_final_reproducibility_package_md | Plan D/D5 | all | true | true | true | true | extended_subset |
| report__plan_d_global_extended_neural_baselines_md | Plan D/D5 | all | true | true | false | false | extended_subset |
| report__plan_d_llm_ml1m_history_length_sensitivity_md | Plan D/D5 | ML-1M | true | true | true | true | diagnostic_subset |
| report__plan_d_llm_ml1m_reranking_diagnostic_md | Plan D/D5 | ML-1M | true | true | true | true | diagnostic_subset |
| report__plan_d_llm_model_registry_and_backend_audit_md | Plan D/D5 | all | false | false | true | true | diagnostic_subset |
| report__plan_d_llm_prompt_templates_and_ranking_manifest_md | Plan D/D5 | all | false | false | true | true | diagnostic_subset |
| report__plan_d_mistral_smoke_test_md | Plan D/D5 | all | false | false | true | true | diagnostic_subset |
| report__plan_d_ml1m_extended_neural_baselines_md | Plan D/D5 | ML-1M | true | true | false | false | extended_subset |
| ... | ... | ... | ... | ... | ... | ... | ... |

The metric inventory represents Plan C reconstructed metrics, Plan D extended neural baselines, D5 local LLM diagnostics, and E0 audit outputs. Local Qwen2/Mistral status remains local diagnostic, not paper-exact.

## Raw-Output Hygiene Summary

| check_id | check_name | status | count |
| --- | --- | --- | --- |
| H01 | results_raw_outputs_exists | passed | 1 |
| H02 | jsonl_files_under_results_raw_outputs | review | 7 |
| H03 | tracked_jsonl_files_under_results_raw_outputs | passed | 0 |
| H04 | tracked_checkpoint_cache_model_like_files | review | 4 |
| H05 | untracked_forbidden_file_patterns | passed | 0 |
| H06 | original_repo_git_status | passed | 0 |
| H07 | plan_c_git_status | passed | 0 |
| H08 | e1_raw_output_creation | passed | 0 |

Raw LLM outputs must remain under `results/raw_outputs/` and Git-ignored. E1 created no raw output and no JSONL file.

## Scale-Up Readiness Summary

| dataset | task | target_users | candidate_pool_size | current_manifest_status | current_data_status | readiness_class | blocking_issue |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ML-1M | ranking | 20 | 20 | 20_user_diagnostic_manifest_available | reconstructed_data_available | ready |  |
| ML-1M | ranking | 100 | 20 | nonpaper_candidate_manifest_available | reconstructed_data_available | needs_manifest_construction | Existing candidates do not match the paper 20-item, 1-positive/19-negative LLM manifest. |
| ML-1M | ranking | 1000 | 20 | nonpaper_candidate_manifest_available | reconstructed_data_available | needs_manifest_construction | Existing candidates do not match the paper 20-item, 1-positive/19-negative LLM manifest. |
| ML-1M | reranking | 20 | 20 | 20_user_diagnostic_manifest_available | reconstructed_data_available | ready |  |
| ML-1M | reranking | 100 | 20 | needs_manifest_construction | reconstructed_data_available | needs_manifest_construction | Paper-like manifest is not yet available. |
| ML-1M | reranking | 1000 | 20 | needs_manifest_construction | reconstructed_data_available | needs_manifest_construction | Paper-like manifest is not yet available. |
| Amazon Beauty | ranking | 20 | 20 | nonpaper_candidate_manifest_available | reconstructed_data_available | needs_manifest_construction | Existing candidates do not match the paper 20-item, 1-positive/19-negative LLM manifest. |
| Amazon Beauty | ranking | 100 | 20 | nonpaper_candidate_manifest_available | reconstructed_data_available | needs_manifest_construction | Existing candidates do not match the paper 20-item, 1-positive/19-negative LLM manifest. |
| Amazon Beauty | ranking | 1000 | 20 | nonpaper_candidate_manifest_available | reconstructed_data_available | needs_manifest_construction | Existing candidates do not match the paper 20-item, 1-positive/19-negative LLM manifest. |
| Amazon Beauty | reranking | 20 | 20 | missing | reconstructed_data_available | partially_ready | Paper-like diagnostic manifest needs construction for this dataset/task. |
| Amazon Beauty | reranking | 100 | 20 | missing | reconstructed_data_available | needs_manifest_construction | Paper-like manifest is not yet available. |
| Amazon Beauty | reranking | 1000 | 20 | missing | reconstructed_data_available | needs_manifest_construction | Paper-like manifest is not yet available. |
| Amazon Sports | ranking | 20 | 20 | nonpaper_candidate_manifest_available | reconstructed_data_available | needs_manifest_construction | Existing candidates do not match the paper 20-item, 1-positive/19-negative LLM manifest. |
| Amazon Sports | ranking | 100 | 20 | nonpaper_candidate_manifest_available | reconstructed_data_available | needs_manifest_construction | Existing candidates do not match the paper 20-item, 1-positive/19-negative LLM manifest. |
| Amazon Sports | ranking | 1000 | 20 | nonpaper_candidate_manifest_available | reconstructed_data_available | needs_manifest_construction | Existing candidates do not match the paper 20-item, 1-positive/19-negative LLM manifest. |
| Amazon Sports | reranking | 20 | 20 | missing | reconstructed_data_available | partially_ready | Paper-like diagnostic manifest needs construction for this dataset/task. |
| Amazon Sports | reranking | 100 | 20 | missing | reconstructed_data_available | needs_manifest_construction | Paper-like manifest is not yet available. |
| Amazon Sports | reranking | 1000 | 20 | missing | reconstructed_data_available | needs_manifest_construction | Paper-like manifest is not yet available. |
| LastFM | ranking | 20 | 20 | missing | conditional_or_blocked_for_valid_utility | blocked | LastFM lacks valid comparable utility-evaluation candidate/ground-truth artifacts. |
| LastFM | ranking | 100 | 20 | missing | conditional_or_blocked_for_valid_utility | blocked | LastFM lacks valid comparable utility-evaluation candidate/ground-truth artifacts. |
| LastFM | ranking | 1000 | 20 | missing | conditional_or_blocked_for_valid_utility | blocked | LastFM lacks valid comparable utility-evaluation candidate/ground-truth artifacts. |
| LastFM | reranking | 20 | 20 | missing | conditional_or_blocked_for_valid_utility | blocked | LastFM lacks valid comparable utility-evaluation candidate/ground-truth artifacts. |
| LastFM | reranking | 100 | 20 | missing | conditional_or_blocked_for_valid_utility | blocked | LastFM lacks valid comparable utility-evaluation candidate/ground-truth artifacts. |
| LastFM | reranking | 1000 | 20 | missing | conditional_or_blocked_for_valid_utility | blocked | LastFM lacks valid comparable utility-evaluation candidate/ground-truth artifacts. |

Readiness is strongest for existing ML-1M 20-user diagnostics. ML-1M, Amazon Beauty, and Amazon Sports need E2 paper-like manifest construction for 100-user pilots and 1000-user scale. LastFM remains conditional/blocked unless valid utility-evaluation artifacts are found.

## Local Call Estimate Interpretation

| scenario_id | dataset_count | users_per_dataset | model_count | prompt_strategy_count | task_count | estimated_total_calls | classification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| local_Qwen2_Mistral_20_users_ranking | 1 | 20 | 2 | 1 | 1 | 40 | diagnostic_subset |
| local_Qwen2_Mistral_100_users_ranking | 1 | 100 | 2 | 1 | 1 | 200 | needs_manifest_construction |
| local_Qwen2_Mistral_1000_users_ranking_ML1M | 1 | 1000 | 2 | 1 | 1 | 2000 | needs_manifest_construction |
| local_Qwen2_Mistral_1000_users_ranking_3_valid_datasets | 3 | 1000 | 2 | 1 | 1 | 6000 | needs_manifest_construction |
| local_Qwen2_Mistral_1000_users_ranking_plus_reranking_3_valid_datasets | 3 | 1000 | 2 | 1 | 2 | 12000 | needs_manifest_construction |
| paper_7LLM_1000_users_3_valid_datasets_ranking | 3 | 1000 | 7 | 1 | 1 | 21000 | conditional/API-dependent |
| paper_7LLM_1000_users_3_valid_datasets_ranking_plus_reranking | 3 | 1000 | 7 | 1 | 2 | 42000 | conditional/API-dependent |
| paper_7LLM_3_prompt_strategies_1000_users_3_valid_datasets_ranking | 3 | 1000 | 7 | 3 | 1 | 63000 | conditional/API-dependent |
| paper_7LLM_3_prompt_strategies_1000_users_3_valid_datasets_ranking_plus_reranking | 3 | 1000 | 7 | 3 | 2 | 126000 | conditional/API-dependent |
| local_Qwen2_Mistral_1000_users_ranking_4_datasets_LastFM_conditional | 4 | 1000 | 2 | 1 | 1 | 8000 | conditional |
| paper_7LLM_1000_users_4_datasets_ranking_plus_reranking_LastFM_conditional | 4 | 1000 | 7 | 1 | 2 | 56000 | conditional |
| paper_7LLM_3_prompt_strategies_1000_users_4_datasets_ranking_plus_reranking_LastFM_conditional | 4 | 1000 | 7 | 3 | 2 | 168000 | conditional |

The largest three-valid-dataset paper-style scenario in E1 is `paper_7LLM_3_prompt_strategies_1000_users_3_valid_datasets_ranking_plus_reranking` with `126000` estimated calls. This is a planning estimate only.

## Runtime Estimate Interpretation

Runtime estimates use assumed seconds-per-call values of 10, 20, 45, and 90. All rows are marked `assumed_planning_estimate`, not measured inference. At 90 seconds per call, `paper_7LLM_3_prompt_strategies_1000_users_3_valid_datasets_ranking_plus_reranking` is estimated at `3150.0` hours or `131.25` days.

Closed-source model costs are not verified in E1. API pricing verification belongs to E6, not E1.

## Risks and Blockers

- No 1000-user paper-like 20-candidate manifests are complete yet.
- LastFM remains conditional/blocked unless valid utility artifacts are found.
- Closed-source/API-dependent models require E6 availability and pricing checks.
- Llama3-70B remains hardware/cloud/API-dependent.
- Local Qwen2/Mistral diagnostics are useful feasibility evidence but are not paper-exact.
- E1 does not run any experiment and does not validate utility at paper scale.

## Recommended E2 Scope

E2 should validate eligible users for ML-1M, Amazon Beauty, and Amazon Sports; keep LastFM conditional; build deterministic ranking and reranking manifests; preserve the 20-candidate pool; preserve 1 positive plus 19 negatives; and record deterministic seed metadata. E2 should still avoid LLM inference unless the project explicitly changes phase and approval scope.

## Claim-Safety Notes for Oral Defense

- Say E1 is inventory and planning, not reproduction.
- Say E1 does not complete 1000-user paper-like evaluation.
- Say Plan C remains reconstructed reproduction.
- Say Plan D local LLM diagnostics remain diagnostic_subset.
- Say strict candidate-ID-only prompting remains a project extension.
- Say no full exact reproduction is claimed.
- Say closed-source/API model costs are not verified in E1.
- Say API pricing verification belongs to E6.

## Generated Outputs

- `results/tables/plan_e_full_data_inventory_dataset_summary.csv`
- `results/tables/plan_e_full_data_inventory_artifact_summary.csv`
- `results/tables/plan_e_full_data_inventory_manifest_summary.csv`
- `results/tables/plan_e_full_data_inventory_metric_summary.csv`
- `results/tables/plan_e_full_data_inventory_raw_output_hygiene.csv`
- `results/tables/plan_e_full_data_inventory_scaleup_readiness.csv`
- `results/tables/plan_e_full_data_inventory_call_estimates.csv`
- `results/tables/plan_e_full_data_inventory_runtime_estimates.csv`
- `results/tables/plan_e_full_data_inventory_next_actions.csv`
- `report/plan_e_full_data_inventory.md`
