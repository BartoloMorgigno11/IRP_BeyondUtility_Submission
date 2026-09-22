# Phase F6L LastFM V4 Ranking Metrics Freeze

- Run ID: `20260804_131959`
- Freeze state: `ranking_metrics_frozen_before_reranking`
- Decision token: `f6l_lastfm_v4_ranking_metrics_consolidated_and_frozen_for_reranking`
- Freeze SHA-256: `D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC`
- Scope: ranking only; no reranking, no new inference, no API call, no raw repair, no deduplication.

## Audit

| check_name | status | observed | expected | evidence |
| --- | --- | --- | --- | --- |
| completed_4000 | pass | 4000 | 4000 | results/tables/phase_f6l_lastfm_corrective_output_contract_v4_full_run_final_validation.csv |
| four_cells_1000_requested | pass | cells=4;planned=1000;1000;1000;1000 | 4 cells, each planned=1000 | results/tables/phase_f6l_lastfm_corrective_output_contract_v4_full_run_validity_by_cell.csv |
| strict_valid_plus_structural_invalid_equals_completed_per_cell | pass | qwen2:7b-instruct/base/OpenP5 title-list=1000/1000;qwen2:7b-instruct/LLMRank recency=1000/1000;mistral:7b-instruct/base/OpenP5 title-list=1000/1000;mistral:7b-instruct/LLMRank recency=1000/1000 | sum equals completed | results/tables/phase_f6l_lastfm_corrective_output_contract_v4_full_run_validity_by_cell.csv |
| fail_strict_denominator_1000_per_cell | pass | pass | 1000 | results/tables/phase_f6l_lastfm_corrective_output_contract_v4_full_run_fail_strict_utility.csv |
| complete_case_denominator_equals_strict_valid_outputs | pass | pass | valid_rows=strict_valid_rows | results/tables/phase_f6l_lastfm_corrective_output_contract_v4_full_run_complete_case_utility.csv |
| all_50_invalid_duplicate_token | pass | {"duplicate_token": "50", "strict_valid": "3950"} | duplicate_token=50 | results/tables/phase_f6l_lastfm_corrective_output_contract_v4_full_run_failure_taxonomy.csv |
| no_malformed_json | pass | 0 | 0 | results/tables/phase_f6l_lastfm_corrective_output_contract_v4_full_run_validity_by_cell.csv |
| no_unknown_token | pass | 0 | 0 | results/tables/phase_f6l_lastfm_corrective_output_contract_v4_full_run_validity_by_cell.csv |
| no_request_outside_manifest | pass | pass | pass | results/tables/phase_f6l_lastfm_corrective_output_contract_v4_full_run_final_validation.csv |
| no_completed_repeated | pass | pass | pass | results/tables/phase_f6l_lastfm_corrective_output_contract_v4_full_run_final_validation.csv |
| retry_exclusively_infrastructural | pass | retry=13;scientific_retry=0;structural_invalid_retried=false | transport retry only | results/tables/phase_f6l_lastfm_corrective_output_contract_v4_full_run_final_validation.csv |
| raw_jsonl_coherence_pass | pass | pass | pass | results/tables/phase_f6l_lastfm_corrective_output_contract_v4_full_run_final_validation.csv |
| manifest_parser_schema_prompt_hashes_unchanged | pass | pass | pass | results/tables/phase_f6l_lastfm_corrective_output_contract_v4_full_run_final_validation.csv |
| metrics_not_contaminated_by_prior_runs | pass | pass | pass | results/tables/phase_f6l_lastfm_corrective_output_contract_v4_full_run_final_validation.csv |
| no_metric_repair_or_deduplication | pass | pass | pass | results/tables/phase_f6l_lastfm_corrective_output_contract_v4_full_run_beyond_utility_metrics.csv |
| statistical_delta_recomputed_from_parse_matches_fail_strict_artifacts | pass | pass | pass | results/tables/phase_f6l_lastfm_corrective_output_contract_v4_full_run_parse_results.csv |

## Table 1 - Primary Fail-Strict Utility

| model | strategy | total_users | strict_valid | structural_invalid | hit_count_at_5 | HR@5 | NDCG@5 | MRR | mean_positive_rank | median_positive_rank | validity_rate | duplicate_token_rate | retry_count | retry_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2:7b-instruct | base/OpenP5 title-list | 1000 | 1000 | 0 | 458 | 0.458 | 0.323647507835 | not_calculated | 9.712 | not_calculated | 1 | 0 | 4 | 0.004 |
| qwen2:7b-instruct | LLMRank recency | 1000 | 1000 | 0 | 419 | 0.419 | 0.308222730898 | not_calculated | 10.107 | not_calculated | 1 | 0 | 0 | 0 |
| mistral:7b-instruct | base/OpenP5 title-list | 1000 | 983 | 17 | 307 | 0.307 | 0.223073398036 | not_calculated | 12.202 | not_calculated | 0.983 | 0.017 | 9 | 0.009 |
| mistral:7b-instruct | LLMRank recency | 1000 | 967 | 33 | 319 | 0.319 | 0.23287621762 | not_calculated | 12.056 | not_calculated | 0.967 | 0.033 | 0 | 0 |

## Table 2 - Secondary Complete Case

| model | strategy | valid_users | excluded_structural_invalid | HR@5 | NDCG@5 | MRR | mean_positive_rank | median_positive_rank | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2:7b-instruct | base/OpenP5 title-list | 1000 | 0 | 0.458 | 0.323647507835 | not_calculated | 9.712 | not_calculated | secondary_complete_case_not_primary |
| qwen2:7b-instruct | LLMRank recency | 1000 | 0 | 0.419 | 0.308222730898 | not_calculated | 10.107 | not_calculated | secondary_complete_case_not_primary |
| mistral:7b-instruct | base/OpenP5 title-list | 983 | 17 | 0.312309257375 | 0.226931228928 | not_calculated | 12.0671414039 | not_calculated | secondary_complete_case_not_primary |
| mistral:7b-instruct | LLMRank recency | 967 | 33 | 0.329886246122 | 0.240823389472 | not_calculated | 11.784901758 | not_calculated | secondary_complete_case_not_primary |

## Table 3 - Reliability

| model | strategy | strict_valid_count | strict_valid_rate | structural_invalid_count | structural_invalid_rate | duplicate_token_count | duplicate_token_rate | malformed_count | unknown_token_count | retry_count | retry_rate | timeout_count | elevated_duplicate_windows | max_consecutive_duplicate_token_invalid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2:7b-instruct | base/OpenP5 title-list | 1000 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0.004 | not_available_as_canonical_count | none | 0 |
| qwen2:7b-instruct | LLMRank recency | 1000 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | not_available_as_canonical_count | none | 0 |
| mistral:7b-instruct | base/OpenP5 title-list | 983 | 0.983 | 17 | 0.017 | 17 | 0.017 | 0 | 0 | 9 | 0.009 | not_available_as_canonical_count | none | 2 |
| mistral:7b-instruct | LLMRank recency | 967 | 0.967 | 33 | 0.033 | 33 | 0.033 | 0 | 0 | 0 | 0 | not_available_as_canonical_count | none | 2 |

## Table 4 - Beyond Utility

| model | strategy | metric | k | value | denominator | valid_output_coverage | alignment | definition_implemented |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2:7b-instruct | base/OpenP5 title-list | novelty | 5 | 11.9031441553 | 5000 | 1000/1000 | paper_aligned_reconstructed_inputs | Mean train-popularity self-information over strict-valid top-5 recommendation rows. |
| qwen2:7b-instruct | base/OpenP5 title-list | APLT_long_tail | 5 | 0.716 | 5000 | 1000/1000 | paper_aligned_local_operationalization | Top-5 share whose train popularity is in the local long-tail set; cutoff recorded in source notes. |
| qwen2:7b-instruct | base/OpenP5 title-list | average_popularity | 5 | 39.6762 | 5000 | 1000/1000 | paper_aligned_local_operationalization | Mean train interaction count of strict-valid top-5 recommended items. |
| qwen2:7b-instruct | base/OpenP5 title-list | popularity_bias | 5 | 39.6762 | 5000 | 1000/1000 | local_proxy | Same average-popularity primitive retained under a separate audit label; not PopREO. |
| qwen2:7b-instruct | base/OpenP5 title-list | catalog_coverage | 5 | 0.807160581354 | 2821 | 1000/1000 | extension | Unique strict-valid top-5 recommended items divided by the operationalized candidate universe. |
| qwen2:7b-instruct | base/OpenP5 title-list | ILD_diversity | 5 | 0.963655639101 | 1000 | 1000/1000 | local_proxy | Mean within-list Jaccard distance over LastFM tag sets for strict-valid top-5 lists. |
| qwen2:7b-instruct | base/OpenP5 title-list | calibration | 5 | 0.339339014495 | 1000 | 1000/1000 | extension | One minus Jensen-Shannon divergence between user history tags and top-5 recommendation tags. |
| qwen2:7b-instruct | LLMRank recency | novelty | 5 | 11.9530517957 | 5000 | 1000/1000 | paper_aligned_reconstructed_inputs | Mean train-popularity self-information over strict-valid top-5 recommendation rows. |
| qwen2:7b-instruct | LLMRank recency | APLT_long_tail | 5 | 0.731 | 5000 | 1000/1000 | paper_aligned_local_operationalization | Top-5 share whose train popularity is in the local long-tail set; cutoff recorded in source notes. |
| qwen2:7b-instruct | LLMRank recency | average_popularity | 5 | 38.1846 | 5000 | 1000/1000 | paper_aligned_local_operationalization | Mean train interaction count of strict-valid top-5 recommended items. |
| qwen2:7b-instruct | LLMRank recency | popularity_bias | 5 | 38.1846 | 5000 | 1000/1000 | local_proxy | Same average-popularity primitive retained under a separate audit label; not PopREO. |
| qwen2:7b-instruct | LLMRank recency | catalog_coverage | 5 | 0.812832328961 | 2821 | 1000/1000 | extension | Unique strict-valid top-5 recommended items divided by the operationalized candidate universe. |
| qwen2:7b-instruct | LLMRank recency | ILD_diversity | 5 | 0.964457015589 | 1000 | 1000/1000 | local_proxy | Mean within-list Jaccard distance over LastFM tag sets for strict-valid top-5 lists. |
| qwen2:7b-instruct | LLMRank recency | calibration | 5 | 0.331375348668 | 1000 | 1000/1000 | extension | One minus Jensen-Shannon divergence between user history tags and top-5 recommendation tags. |
| mistral:7b-instruct | base/OpenP5 title-list | novelty | 5 | 12.1049753646 | 4915 | 983/1000 | paper_aligned_reconstructed_inputs | Mean train-popularity self-information over strict-valid top-5 recommendation rows. |
| mistral:7b-instruct | base/OpenP5 title-list | APLT_long_tail | 5 | 0.766022380468 | 4915 | 983/1000 | paper_aligned_local_operationalization | Top-5 share whose train popularity is in the local long-tail set; cutoff recorded in source notes. |
| mistral:7b-instruct | base/OpenP5 title-list | average_popularity | 5 | 31.8158697864 | 4915 | 983/1000 | paper_aligned_local_operationalization | Mean train interaction count of strict-valid top-5 recommended items. |
| mistral:7b-instruct | base/OpenP5 title-list | popularity_bias | 5 | 31.8158697864 | 4915 | 983/1000 | local_proxy | Same average-popularity primitive retained under a separate audit label; not PopREO. |
| mistral:7b-instruct | base/OpenP5 title-list | catalog_coverage | 5 | 0.825948245303 | 2821 | 983/1000 | extension | Unique strict-valid top-5 recommended items divided by the operationalized candidate universe. |
| mistral:7b-instruct | base/OpenP5 title-list | ILD_diversity | 5 | 0.967508844 | 983 | 983/1000 | local_proxy | Mean within-list Jaccard distance over LastFM tag sets for strict-valid top-5 lists. |
| mistral:7b-instruct | base/OpenP5 title-list | calibration | 5 | 0.304882122427 | 983 | 983/1000 | extension | One minus Jensen-Shannon divergence between user history tags and top-5 recommendation tags. |
| mistral:7b-instruct | LLMRank recency | novelty | 5 | 12.082198655 | 4835 | 967/1000 | paper_aligned_reconstructed_inputs | Mean train-popularity self-information over strict-valid top-5 recommendation rows. |
| mistral:7b-instruct | LLMRank recency | APLT_long_tail | 5 | 0.759669079628 | 4835 | 967/1000 | paper_aligned_local_operationalization | Top-5 share whose train popularity is in the local long-tail set; cutoff recorded in source notes. |
| mistral:7b-instruct | LLMRank recency | average_popularity | 5 | 32.5453981386 | 4835 | 967/1000 | paper_aligned_local_operationalization | Mean train interaction count of strict-valid top-5 recommended items. |
| mistral:7b-instruct | LLMRank recency | popularity_bias | 5 | 32.5453981386 | 4835 | 967/1000 | local_proxy | Same average-popularity primitive retained under a separate audit label; not PopREO. |
| mistral:7b-instruct | LLMRank recency | catalog_coverage | 5 | 0.817440623892 | 2821 | 967/1000 | extension | Unique strict-valid top-5 recommended items divided by the operationalized candidate universe. |
| mistral:7b-instruct | LLMRank recency | ILD_diversity | 5 | 0.967408854005 | 967 | 967/1000 | local_proxy | Mean within-list Jaccard distance over LastFM tag sets for strict-valid top-5 lists. |
| mistral:7b-instruct | LLMRank recency | calibration | 5 | 0.308307719271 | 967 | 967/1000 | extension | One minus Jensen-Shannon divergence between user history tags and top-5 recommendation tags. |

## Table 5 - Cell Comparisons

| comparison | metric | left_value | right_value | delta | notes |
| --- | --- | --- | --- | --- | --- |
| qwen2_llmrank_minus_qwen2_base | HR@5_fail_strict | 0.458 | 0.419 | -0.039 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| qwen2_llmrank_minus_qwen2_base | NDCG@5_fail_strict | 0.323647507835 | 0.308222730898 | -0.015424776937 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| qwen2_llmrank_minus_qwen2_base | HR@5_complete_case | 0.458 | 0.419 | -0.039 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| qwen2_llmrank_minus_qwen2_base | NDCG@5_complete_case | 0.323647507835 | 0.308222730898 | -0.015424776937 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| qwen2_llmrank_minus_qwen2_base | validity_rate | 1 | 1 | 0 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| qwen2_llmrank_minus_qwen2_base | duplicate_token_rate | 0 | 0 | 0 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| qwen2_llmrank_minus_qwen2_base | novelty | 11.9031441553 | 11.9530517957 | 0.0499076404 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| qwen2_llmrank_minus_qwen2_base | APLT_long_tail | 0.716 | 0.731 | 0.015 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| qwen2_llmrank_minus_qwen2_base | average_popularity | 39.6762 | 38.1846 | -1.4916 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| qwen2_llmrank_minus_qwen2_base | catalog_coverage | 0.807160581354 | 0.812832328961 | 0.005671747607 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| qwen2_llmrank_minus_qwen2_base | ILD_diversity | 0.963655639101 | 0.964457015589 | 0.000801376488 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_mistral_base | HR@5_fail_strict | 0.307 | 0.319 | 0.012 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_mistral_base | NDCG@5_fail_strict | 0.223073398036 | 0.23287621762 | 0.009802819584 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_mistral_base | HR@5_complete_case | 0.312309257375 | 0.329886246122 | 0.017576988747 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_mistral_base | NDCG@5_complete_case | 0.226931228928 | 0.240823389472 | 0.013892160544 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_mistral_base | validity_rate | 0.983 | 0.967 | -0.016 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_mistral_base | duplicate_token_rate | 0.017 | 0.033 | 0.016 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_mistral_base | novelty | 12.1049753646 | 12.082198655 | -0.0227767096 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_mistral_base | APLT_long_tail | 0.766022380468 | 0.759669079628 | -0.00635330084 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_mistral_base | average_popularity | 31.8158697864 | 32.5453981386 | 0.7295283522 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_mistral_base | catalog_coverage | 0.825948245303 | 0.817440623892 | -0.008507621411 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_mistral_base | ILD_diversity | 0.967508844 | 0.967408854005 | -0.000099989995 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_base_minus_qwen2_base | HR@5_fail_strict | 0.458 | 0.307 | -0.151 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_base_minus_qwen2_base | NDCG@5_fail_strict | 0.323647507835 | 0.223073398036 | -0.100574109799 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_base_minus_qwen2_base | HR@5_complete_case | 0.458 | 0.312309257375 | -0.145690742625 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_base_minus_qwen2_base | NDCG@5_complete_case | 0.323647507835 | 0.226931228928 | -0.096716278907 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_base_minus_qwen2_base | validity_rate | 1 | 0.983 | -0.017 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_base_minus_qwen2_base | duplicate_token_rate | 0 | 0.017 | 0.017 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_base_minus_qwen2_base | novelty | 11.9031441553 | 12.1049753646 | 0.2018312093 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_base_minus_qwen2_base | APLT_long_tail | 0.716 | 0.766022380468 | 0.050022380468 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_base_minus_qwen2_base | average_popularity | 39.6762 | 31.8158697864 | -7.8603302136 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_base_minus_qwen2_base | catalog_coverage | 0.807160581354 | 0.825948245303 | 0.018787663949 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_base_minus_qwen2_base | ILD_diversity | 0.963655639101 | 0.967508844 | 0.003853204899 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_qwen2_llmrank | HR@5_fail_strict | 0.419 | 0.319 | -0.1 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_qwen2_llmrank | NDCG@5_fail_strict | 0.308222730898 | 0.23287621762 | -0.075346513278 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_qwen2_llmrank | HR@5_complete_case | 0.419 | 0.329886246122 | -0.089113753878 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_qwen2_llmrank | NDCG@5_complete_case | 0.308222730898 | 0.240823389472 | -0.067399341426 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_qwen2_llmrank | validity_rate | 1 | 0.967 | -0.033 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_qwen2_llmrank | duplicate_token_rate | 0 | 0.033 | 0.033 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_qwen2_llmrank | novelty | 11.9530517957 | 12.082198655 | 0.1291468593 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_qwen2_llmrank | APLT_long_tail | 0.731 | 0.759669079628 | 0.028669079628 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_qwen2_llmrank | average_popularity | 38.1846 | 32.5453981386 | -5.6392018614 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_qwen2_llmrank | catalog_coverage | 0.812832328961 | 0.817440623892 | 0.004608294931 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |
| mistral_llmrank_minus_qwen2_llmrank | ILD_diversity | 0.964457015589 | 0.967408854005 | 0.002951838416 | Interpret with validity and denominator context; beyond-utility uses strict-valid coverage. |

## Statistical Tests

| comparison | scope | metric | pairs_valid | delta_mean_right_minus_left | paired_bootstrap_ci_low | paired_bootstrap_ci_high | test | p_value | p_value_bonferroni | effect_size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2_llmrank_minus_qwen2_base | fail_strict_all_1000 | hit_at_5 | 1000 | -0.039 | -0.063 | -0.015 | mcnemar_exact_binomial | 0.0021681266074654398 | 0.0346900257194470368 | -0.039 |
| qwen2_llmrank_minus_qwen2_base | fail_strict_all_1000 | ndcg_at_5 | 1000 | -0.015424776936502585 | -0.0323360120473819 | 0.002326800544559455 | paired_sign_flip_randomization | 0.08418316336732654 | 1 | -0.055831412377004334 |
| qwen2_llmrank_minus_qwen2_base | complete_case_intersection | hit_at_5 | 1000 | -0.039 | -0.064 | -0.015 | mcnemar_exact_binomial | 0.0021681266074654398 | 0.0346900257194470368 | -0.039 |
| qwen2_llmrank_minus_qwen2_base | complete_case_intersection | ndcg_at_5 | 1000 | -0.015424776936502585 | -0.03261643965466208 | 0.0016959858177826507 | paired_sign_flip_randomization | 0.07498500299940011 | 1 | -0.055831412377004334 |
| mistral_llmrank_minus_mistral_base | fail_strict_all_1000 | hit_at_5 | 1000 | 0.012 | -0.005 | 0.029 | mcnemar_exact_binomial | 0.21851803947870374 | 1 | 0.012 |
| mistral_llmrank_minus_mistral_base | fail_strict_all_1000 | ndcg_at_5 | 1000 | 0.009802819583289647 | -0.0050351352197075825 | 0.025124994739789016 | paired_sign_flip_randomization | 0.20955808838232354 | 1 | 0.040308532952368054 |
| mistral_llmrank_minus_mistral_base | complete_case_intersection | hit_at_5 | 959 | 0.020855057351407715 | 0.005213764337851929 | 0.03753910323253389 | mcnemar_exact_binomial | 0.01685819004428113 | 0.26973104070849808 | 0.02085505735140771637122002086 |
| mistral_llmrank_minus_mistral_base | complete_case_intersection | ndcg_at_5 | 959 | 0.016600725498539644 | 0.00239336495370032 | 0.03100562880113863 | paired_sign_flip_randomization | 0.024795040991801638 | 0.396720655868826208 | 0.07419382367484327 |
| mistral_base_minus_qwen2_base | fail_strict_all_1000 | hit_at_5 | 1000 | -0.151 | -0.182 | -0.12 | mcnemar_exact_binomial | 0.00000000000000000011257018911866366 | 0.00000000000000000180112302589861856 | -0.151 |
| mistral_base_minus_qwen2_base | fail_strict_all_1000 | ndcg_at_5 | 1000 | -0.10057410979844955 | -0.12485912142749064 | -0.07650980285482961 | paired_sign_flip_randomization | 0.0001999600079984003 | 0.0031993601279744048 | -0.2602403474937764 |
| mistral_base_minus_qwen2_base | complete_case_intersection | hit_at_5 | 983 | -0.14445574771108852 | -0.1770091556459817 | -0.11393692777212615 | mcnemar_exact_binomial | 0.000000000000000004949790935765681 | 0.000000000000000079196654972250896 | -0.1444557477110885045778229908 |
| mistral_base_minus_qwen2_base | complete_case_intersection | ndcg_at_5 | 983 | -0.09625606737140421 | -0.1194988944391542 | -0.07223269172386808 | paired_sign_flip_randomization | 0.0001999600079984003 | 0.0031993601279744048 | -0.24968982616664478 |
| mistral_llmrank_minus_qwen2_llmrank | fail_strict_all_1000 | hit_at_5 | 1000 | -0.1 | -0.131 | -0.07 | mcnemar_exact_binomial | 0.00000000026276492897173625 | 0.00000000420423886354778 | -0.1 |
| mistral_llmrank_minus_qwen2_llmrank | fail_strict_all_1000 | ndcg_at_5 | 1000 | -0.07534651327865732 | -0.09964947414172771 | -0.05226091791280771 | paired_sign_flip_randomization | 0.0001999600079984003 | 0.0031993601279744048 | -0.19785695585717758 |
| mistral_llmrank_minus_qwen2_llmrank | complete_case_intersection | hit_at_5 | 967 | -0.08686659772492245 | -0.11789038262668046 | -0.055842812823164424 | mcnemar_exact_binomial | 0.00000004783169486571541 | 0.00000076530711785144656 | -0.08686659772492244053774560496 |
| mistral_llmrank_minus_qwen2_llmrank | complete_case_intersection | ndcg_at_5 | 967 | -0.06763781932787973 | -0.09178013517753367 | -0.04405006029087788 | paired_sign_flip_randomization | 0.0001999600079984003 | 0.0031993601279744048 | -0.17822752302106545 |

## Runtime

| scope | model_name | prompting_strategy | wall_clock_seconds | duration_mean_seconds | duration_median_seconds | duration_p90_seconds | duration_p95_seconds | duration_p99_seconds | throughput_requests_per_hour | prompt_eval_count_total | eval_count_total | raw_output_bytes | parsing_time_seconds | consolidation_time_seconds | marginal_api_cost | hardware_cost | energy_cost | retry | failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall |  |  | 82637.0416262 | 20.6592604066 | 30.2815626001 | 38.86247251 | 63.83238558 | 90.363817293 | 174.25599606 | 2817012 | 182036 | 139995619 | 7.15033079998 | 15.775442 | 0 | not_monetized | unavailable | 13 | 50 |
| cell | qwen2:7b-instruct | base/OpenP5 title-list | 35400.2218138 | 35.4002218138 | 30.9871049 | 53.85497218 | 69.333567675 | 92.098178976 | 101.694278045 | 655113 | 44000 |  | 0 | 0 | 0 | not_monetized | unavailable | 0 | 0 |
| cell | qwen2:7b-instruct | LLMRank recency | 1514.4539107 | 1.5144539107 | 1.37278119999 | 1.66384232998 | 1.83705178 | 1.98591552302 | 2377.09445931 | 686113 | 44000 |  | 0 | 0 | 0 | not_monetized | unavailable | 0 | 0 |
| cell | mistral:7b-instruct | base/OpenP5 title-list | 43930.3258035 | 43.9303258035 | 38.565772 | 64.95619923 | 86.811350475 | 99.176360074 | 81.9479467578 | 720893 | 47000 |  | 0 | 0 | 0 | not_monetized | unavailable | 0 | 17 |
| cell | mistral:7b-instruct | LLMRank recency | 1792.0400982 | 1.7920400982 | 1.42221059999 | 1.61755778998 | 1.73303925501 | 1.81635978098 | 2008.88362019 | 754893 | 47036 |  | 0 | 0 | 0 | not_monetized | unavailable | 0 | 33 |

## Research Questions

| rq_id | previous_status | new_status | qwen2_base | qwen2_llmrank | mistral_base | mistral_llmrank | limits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F6L-LASTFM-V4-AMENDED-RQ-UTILITY | unresolved_for_original_paper_sample | supported_under_local_operationalization_with_observed_runtime_validity | HR@5_fail_strict=0.458;NDCG@5_fail_strict=0.323647507835;validity=1 | HR@5_fail_strict=0.419;NDCG@5_fail_strict=0.308222730898;validity=1 | HR@5_fail_strict=0.307;NDCG@5_fail_strict=0.223073398036;validity=0.983 | HR@5_fail_strict=0.319;NDCG@5_fail_strict=0.23287621762;validity=0.967 | No exact paper reproduction, original paper sample, original output contract, closed models, LLMSRec, reranking, or full cohort claim. |
| F6L-LASTFM-V4-AMENDED-RQ-PROMPT | unresolved_for_original_paper_sample | mixed_prompt_dependent | HR@5_fail_strict=0.458;NDCG@5_fail_strict=0.323647507835;validity=1 | HR@5_fail_strict=0.419;NDCG@5_fail_strict=0.308222730898;validity=1 | HR@5_fail_strict=0.307;NDCG@5_fail_strict=0.223073398036;validity=0.983 | HR@5_fail_strict=0.319;NDCG@5_fail_strict=0.23287621762;validity=0.967 | No universal paper-level prompt claim from local open models alone. |
| F6L-LASTFM-V4-AMENDED-RQ-STRUCTURED-VALIDITY | not_established_by_pilot_alone | duplicate_token_rate_is_an_observed_model_runtime_property | HR@5_fail_strict=0.458;NDCG@5_fail_strict=0.323647507835;validity=1 | HR@5_fail_strict=0.419;NDCG@5_fail_strict=0.308222730898;validity=1 | HR@5_fail_strict=0.307;NDCG@5_fail_strict=0.223073398036;validity=0.983 | HR@5_fail_strict=0.319;NDCG@5_fail_strict=0.23287621762;validity=0.967 | This is not corruption, transport failure, malformed JSON, or proof of full JSON Schema enforcement. |

## Paper Comparison

| model | strategy | metric | paper_lastfm_value | local_fail_strict_value | absolute_difference | relative_difference | comparability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2:7b-instruct | base/OpenP5 title-list | HR@5 | not_machine_extracted_locally | 0.458 | not_available | not_available | not_exact_numerical_replica |
| qwen2:7b-instruct | base/OpenP5 title-list | NDCG@5 | not_machine_extracted_locally | 0.323647507835 | not_available | not_available | not_exact_numerical_replica |
| qwen2:7b-instruct | LLMRank recency | HR@5 | not_machine_extracted_locally | 0.419 | not_available | not_available | not_exact_numerical_replica |
| qwen2:7b-instruct | LLMRank recency | NDCG@5 | not_machine_extracted_locally | 0.308222730898 | not_available | not_available | not_exact_numerical_replica |
| mistral:7b-instruct | base/OpenP5 title-list | HR@5 | not_machine_extracted_locally | 0.307 | not_available | not_available | not_exact_numerical_replica |
| mistral:7b-instruct | base/OpenP5 title-list | NDCG@5 | not_machine_extracted_locally | 0.223073398036 | not_available | not_available | not_exact_numerical_replica |
| mistral:7b-instruct | LLMRank recency | HR@5 | not_machine_extracted_locally | 0.319 | not_available | not_available | not_exact_numerical_replica |
| mistral:7b-instruct | LLMRank recency | NDCG@5 | not_machine_extracted_locally | 0.23287621762 | not_available | not_available | not_exact_numerical_replica |

## Limits

- No exact paper numerical replication is claimed.
- Local LastFM sample, seed, output contract, and Qwen2/Mistral model cells differ from the paper's unrecovered LastFM artifacts and closed/provider models.
- Duplicate-token invalidity is structural output invalidity, not hallucination.
- Complete-case results are secondary and use strict-valid output denominators.
- Calibration is a local extension, not a central original-paper metric.

## Artifacts

| artifact |
| --- |
| results/tables/phase_f6l_lastfm_v4_ranking_freeze_audit.csv |
| results/tables/phase_f6l_lastfm_v4_ranking_freeze_metrics.csv |
| results/tables/phase_f6l_lastfm_v4_ranking_freeze_comparisons.csv |
| results/tables/phase_f6l_lastfm_v4_ranking_freeze_statistical_tests.csv |
| results/tables/phase_f6l_lastfm_v4_ranking_freeze_paper_comparison.csv |
| report/phase_f6l_lastfm_corrective_output_contract_v4_full_ranking.md |
| report/future_operations_context_handoff.md |
| report/notes.md |
