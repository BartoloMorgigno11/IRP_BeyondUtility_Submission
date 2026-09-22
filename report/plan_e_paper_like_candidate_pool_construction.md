# Plan E2 Paper-Like Candidate Pool Construction

## Executive Summary

E2 constructs paper-like candidate manifests, not LLM results. It creates deterministic 1000-user ranking and reranking manifests for ML-1M, Amazon Beauty, and Amazon Sports where eligible users are available. Each generated user pool has 20 candidates, exactly 1 positive item, and exactly 19 negative items.

E2 performed no inference. E2 made no Ollama calls. E2 made no Qwen2/Mistral calls. E2 made no API calls. E2 performed no downloads. E2 performed no model pulls. E2 performed no training. E2 performed no LastFM run. E2 performed no Plan C modification. E2 performed no original_repo modification. E2 performed no destructive cleanup.

E2 does not run recommendation training. E2 does not run LLM inference. E2 does not complete full paper reproduction. Generated manifests are reconstructed paper-like artifacts unless exact original paper candidate files are verified. No full exact reproduction is claimed.

## Source Artifact Audit

| dataset | candidate_source_path | source_class | source_rows | source_user_count | source_candidate_pool_size_mode | source_is_paper_exact | source_is_reconstructed | source_can_support_e2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ML-1M | data/processed/ml1m/candidates.csv | processed_project_candidate_pool | 604000 | 6040 | 100 | false | false | true |
| Amazon Beauty | data/reconstructed/plan_c/amazon_beauty/candidates.csv | reconstructed_plan_c_candidate_pool | 2236300 | 22363 | 100 | false | true | true |
| Amazon Sports | data/reconstructed/plan_c/amazon_sports/candidates.csv | reconstructed_plan_c_candidate_pool | 3559800 | 35598 | 100 | false | true | true |
| ML-1M | data/reconstructed/plan_c/ml1m/candidates.csv | reconstructed_plan_c_candidate_pool | 604000 | 6040 | 100 | false | true | true |
| LastFM |  | missing_candidate_source | 0 | 0 |  | false | false | false |

Reconstructed Plan C candidate pools are useful inputs but not exact original paper candidate files.

## Dataset Eligibility Summary

| dataset | source_path | source_rows | source_user_count | eligible_user_count | selected_user_count | can_build_1000_user_manifest | eligibility_class |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ML-1M | data/reconstructed/plan_c/ml1m/candidates.csv | 604000 | 6040 | 6040 | 1000 | true | complete_1000_user_manifest_ready |
| Amazon Beauty | data/reconstructed/plan_c/amazon_beauty/candidates.csv | 2236300 | 22363 | 22363 | 1000 | true | complete_1000_user_manifest_ready |
| Amazon Sports | data/reconstructed/plan_c/amazon_sports/candidates.csv | 3559800 | 35598 | 35598 | 1000 | true | complete_1000_user_manifest_ready |
| LastFM |  | 0 | 0 | 0 | 0 | false | blocked/conditional |

ML-1M, Amazon Beauty, and Amazon Sports are valid available datasets for reconstructed paper-like E2 candidate manifests. LastFM remains conditional/blocked unless valid utility artifacts are found.

## Manifest Construction Summary

| manifest_id | dataset | task | rows | user_count | candidate_pool_size | positive_items_per_user | negative_items_per_user | exact_1000_user_target_met | classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plan_e_ml1m_1000user_ranking_manifest | ML-1M | ranking | 20000 | 1000 | 20 | 1 | 19 | true | reconstructed paper-like candidate manifest |
| plan_e_ml1m_1000user_reranking_manifest | ML-1M | reranking | 20000 | 1000 | 20 | 1 | 19 | true | reconstructed paper-like candidate manifest |
| plan_e_beauty_1000user_ranking_manifest | Amazon Beauty | ranking | 20000 | 1000 | 20 | 1 | 19 | true | reconstructed paper-like candidate manifest |
| plan_e_beauty_1000user_reranking_manifest | Amazon Beauty | reranking | 20000 | 1000 | 20 | 1 | 19 | true | reconstructed paper-like candidate manifest |
| plan_e_sports_1000user_ranking_manifest | Amazon Sports | ranking | 20000 | 1000 | 20 | 1 | 19 | true | reconstructed paper-like candidate manifest |
| plan_e_sports_1000user_reranking_manifest | Amazon Sports | reranking | 20000 | 1000 | 20 | 1 | 19 | true | reconstructed paper-like candidate manifest |

Complete generated manifests: `6` of `6`.

## Validation Summary

| manifest_id | dataset | task | check_name | passed | observed_value | expected_value | severity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| plan_e_beauty_1000user_ranking_manifest | Amazon Beauty | ranking | file_exists | true | will_write_non_empty_manifest | non_empty_manifest | critical |
| plan_e_beauty_1000user_ranking_manifest | Amazon Beauty | ranking | row_count_is_20000_when_complete | true | 20000 | 20000 | critical |
| plan_e_beauty_1000user_ranking_manifest | Amazon Beauty | ranking | user_count_is_1000_when_complete | true | 1000 | 1000 | critical |
| plan_e_beauty_1000user_ranking_manifest | Amazon Beauty | ranking | candidate_pool_size_is_20 | true | 20 | 20 | critical |
| plan_e_beauty_1000user_ranking_manifest | Amazon Beauty | ranking | one_positive_per_user | true | 1 | 1 | critical |
| plan_e_beauty_1000user_ranking_manifest | Amazon Beauty | ranking | nineteen_negatives_per_user | true | 19 | 19 | critical |
| plan_e_beauty_1000user_ranking_manifest | Amazon Beauty | ranking | no_duplicate_items_per_user | true | 20 | 20 | critical |
| plan_e_beauty_1000user_ranking_manifest | Amazon Beauty | ranking | candidate_positions_are_1_to_20 | true | 1..20_per_user | 1..20_per_user | critical |
| plan_e_beauty_1000user_ranking_manifest | Amazon Beauty | ranking | no_missing_user_ids | true | 0 | 0 | critical |
| plan_e_beauty_1000user_ranking_manifest | Amazon Beauty | ranking | no_missing_item_ids | true | 0 | 0 | critical |
| plan_e_beauty_1000user_ranking_manifest | Amazon Beauty | ranking | deterministic_seed_recorded | true | 20260702 | 20260702 | major |
| plan_e_beauty_1000user_ranking_manifest | Amazon Beauty | ranking | expected_row_count_for_selected_users | true | 20000 | 20000 | major |
| plan_e_beauty_1000user_ranking_manifest | Amazon Beauty | ranking | no_original_repo_output | true | results/tables/plan_e_beauty_1000user_ranking_manifest.csv | not_under_original_repo | critical |
| plan_e_beauty_1000user_reranking_manifest | Amazon Beauty | reranking | file_exists | true | will_write_non_empty_manifest | non_empty_manifest | critical |
| plan_e_beauty_1000user_reranking_manifest | Amazon Beauty | reranking | row_count_is_20000_when_complete | true | 20000 | 20000 | critical |
| plan_e_beauty_1000user_reranking_manifest | Amazon Beauty | reranking | user_count_is_1000_when_complete | true | 1000 | 1000 | critical |
| plan_e_beauty_1000user_reranking_manifest | Amazon Beauty | reranking | candidate_pool_size_is_20 | true | 20 | 20 | critical |
| plan_e_beauty_1000user_reranking_manifest | Amazon Beauty | reranking | one_positive_per_user | true | 1 | 1 | critical |
| plan_e_beauty_1000user_reranking_manifest | Amazon Beauty | reranking | nineteen_negatives_per_user | true | 19 | 19 | critical |
| plan_e_beauty_1000user_reranking_manifest | Amazon Beauty | reranking | no_duplicate_items_per_user | true | 20 | 20 | critical |
| plan_e_beauty_1000user_reranking_manifest | Amazon Beauty | reranking | candidate_positions_are_1_to_20 | true | 1..20_per_user | 1..20_per_user | critical |
| plan_e_beauty_1000user_reranking_manifest | Amazon Beauty | reranking | no_missing_user_ids | true | 0 | 0 | critical |
| plan_e_beauty_1000user_reranking_manifest | Amazon Beauty | reranking | no_missing_item_ids | true | 0 | 0 | critical |
| plan_e_beauty_1000user_reranking_manifest | Amazon Beauty | reranking | deterministic_seed_recorded | true | 20260702 | 20260702 | major |
| ... | ... | ... | ... | ... | ... | ... | ... |

Validation failure rows: `0`. Any non-passing row should be reviewed before E3/E4/E5 use.

## LastFM Blocked/Conditional Status

| dataset | task | blocked_or_conditional | classification | reason |
| --- | --- | --- | --- | --- |
| LastFM | ranking | blocked/conditional | extension_only_no_valid_utility_evaluation | LastFM lacks valid comparable candidate, split, and ground-truth artifacts for paper-like utility evaluation. |
| LastFM | reranking | blocked/conditional | extension_only_no_valid_utility_evaluation | LastFM lacks valid comparable candidate, split, and ground-truth artifacts for paper-like utility evaluation. |

No LastFM paper-like manifest is generated by E2.

## Exactness and Claim-Safety Notes

- E2 constructs reconstructed paper-like candidate manifests, not exact reproduction.
- Exact original paper candidate sampling files have not been verified.
- Strict prompt remains a Plan D/E extension, not a paper-exact prompt strategy.
- Local Qwen2/Mistral diagnostics remain diagnostic_subset results until a later approved E5 run.
- E2 outputs are near/reconstructed reproduction support artifacts.
- No full exact reproduction is claimed.

## Recommended Next Phase

The recommended next phase is E3 baseline alignment using the generated candidate manifests, followed by E4 prompt strategy alignment. E5 local LLM scale-up should only occur after explicit approval and must use raw-output hygiene controls.
