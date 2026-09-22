# Experiment Cohort and Title-Exclusion Policy

## 1. Scope and provenance

- Audit timestamp: 2026-07-10T21:14:34+02:00
- Phase: 2C experiment cohort definition and title-gap exclusion policy
- Python executable: `C:\ProgramData\anaconda3\python.exe` with `PYTHONNOUSERSITE=1`
- No recommender, baseline, LLM, training, scoring, or metric experiment was run.

## 2. Executive summary

- Candidate-only title map rows: 34341.
- Cohort membership rows: 76121.
- LLM title exclusions: 186.
- Plan E2 full baseline eligible rows: 6000.
- Plan E2 strict-title LLM rows: 5814.
- Plan E2 matched comparison rows: 5814.
- Legacy one-hundred-candidate rows: 70041.
- Plan D diagnostic rows: 80.

## 3. Why unresolved titles are not repaired

- Fabricating titles, substituting ASINs, using descriptions/categories, deleting candidates, replacing candidates, or changing order would alter the experimental condition.
- Traditional baselines do not require text titles, so structurally valid full-cohort rows can remain available for baseline-only reporting.
- LLM prompts require resolved human-readable titles; unresolved title rows are therefore excluded from strict-title and matched cohorts rather than repaired.

## 4. Input integrity verification

| field | value |
| --- | --- |
| candidate_rows | 76121 |
| candidate_set_ids | 14 |
| candidate_count_distribution | {"20": 6080, "100": 70041} |
| row_status_distribution | {"valid": 60551, "valid_with_warnings": 15570} |
| unique_candidate_items | 34341 |
| positive_position_rows | 76121 |
| coverage_rows | 14 |

Input hashes:
- `results/tables/ranking_candidate_sets.csv`: `640AD9C00B7D7C825AD10746A0E720C161D1BA9D75CC791A019DF7FE14C7A955`
- `results/tables/positive_positions.csv`: `0C93DD55D5C96ED99FB7F1AFC8F5319E90F3092ECB6A48F2E9D46854B38CD6D2`
- `results/tables/dataset_statistics.csv`: `3B60324836D67EBFBC827B8962AC0EAA68F3A433259D3987FCDC2909643CD560`
- `results/tables/item_title_map.csv`: `E1E3A80D2AE3A825BE7B36AA9EA1EF418B1C68B0EF4734E08C42C61F3FCA68F5`
- `results/tables/item_title_mapping_issues.csv`: `01DCA683A478B951E2B68F33F64FBCC211B276F356CB283C182D6DB3A6C2381E`
- `results/tables/item_title_coverage_summary.csv`: `C93D0A8DE8EA6BF16065F8B5B9AB72B7FD68FC62F959B9D0F34BEB8311350BF5`

## 5. Candidate-set families

| source_plan | candidate_sets | rows |
| --- | --- | --- |
| Plan B | 1 | 6040 |
| Plan C | 3 | 64001 |
| Plan D | 4 | 80 |
| Plan E2 | 6 | 6000 |

## 6. Task-type classification

| candidate_set_id | dataset | source_plan | task_type | rows |
| --- | --- | --- | --- | --- |
| plan_b_ml1m_processed_100_candidate_pool | ML-1M | Plan B | legacy_ranking | 6040 |
| plan_c_amazon_beauty_reconstructed_100_candidate_pool | Amazon Beauty | Plan C | legacy_ranking | 22363 |
| plan_c_amazon_sports_reconstructed_100_candidate_pool | Amazon Sports | Plan C | legacy_ranking | 35598 |
| plan_c_ml1m_reconstructed_100_candidate_pool | ML-1M | Plan C | legacy_ranking | 6040 |
| plan_d_qwen2_ml1m_candidate_position_bias_positive_first | ML-1M | Plan D | diagnostic_ranking | 20 |
| plan_d_qwen2_ml1m_candidate_position_bias_positive_last | ML-1M | Plan D | diagnostic_ranking | 20 |
| plan_d_qwen2_ml1m_candidate_position_bias_reverse_order | ML-1M | Plan D | diagnostic_ranking | 20 |
| plan_d_qwen2_ml1m_ranking_diagnostic_openp5_base | ML-1M | Plan D | diagnostic_ranking | 20 |
| plan_e_beauty_1000user_ranking_manifest | Amazon Beauty | Plan E2 | ranking | 1000 |
| plan_e_beauty_1000user_reranking_manifest | Amazon Beauty | Plan E2 | reranking | 1000 |
| plan_e_ml1m_1000user_ranking_manifest | ML-1M | Plan E2 | ranking | 1000 |
| plan_e_ml1m_1000user_reranking_manifest | ML-1M | Plan E2 | reranking | 1000 |
| plan_e_sports_1000user_ranking_manifest | Amazon Sports | Plan E2 | ranking | 1000 |
| plan_e_sports_1000user_reranking_manifest | Amazon Sports | Plan E2 | reranking | 1000 |

## 7. Authoritative cohort definitions

- `plan_e2_paper_like_full`: Plan E2, 20 candidates, structurally valid, positive present exactly once, ground truth not invalid; titles not required.
- `plan_e2_llm_strict_title`: full Plan E2 eligibility plus resolved positive and candidate titles.
- `plan_e2_matched_comparison`: exactly identical membership to strict-title LLM cohort.
- `legacy_reconstructed_100_candidate`: Plan B or Plan C 100-candidate structurally valid rows.
- `plan_d_diagnostic_20_candidate`: Plan D 20-candidate structurally valid rows, diagnostic only.

## 8. Plan E2 full baseline cohort

| dataset | task_type | candidate_set_id | total | eligible | excluded | rate |
| --- | --- | --- | --- | --- | --- | --- |
| Amazon Beauty | ranking | plan_e_beauty_1000user_ranking_manifest | 1000 | 1000 | 0 | 1.000000 |
| Amazon Beauty | reranking | plan_e_beauty_1000user_reranking_manifest | 1000 | 1000 | 0 | 1.000000 |
| Amazon Sports | ranking | plan_e_sports_1000user_ranking_manifest | 1000 | 1000 | 0 | 1.000000 |
| Amazon Sports | reranking | plan_e_sports_1000user_reranking_manifest | 1000 | 1000 | 0 | 1.000000 |
| ML-1M | ranking | plan_e_ml1m_1000user_ranking_manifest | 1000 | 1000 | 0 | 1.000000 |
| ML-1M | reranking | plan_e_ml1m_1000user_reranking_manifest | 1000 | 1000 | 0 | 1.000000 |

## 9. Plan E2 strict-title LLM cohort

| dataset | task_type | candidate_set_id | total | eligible | excluded | rate |
| --- | --- | --- | --- | --- | --- | --- |
| Amazon Beauty | ranking | plan_e_beauty_1000user_ranking_manifest | 1000 | 994 | 6 | 0.994000 |
| Amazon Beauty | reranking | plan_e_beauty_1000user_reranking_manifest | 1000 | 994 | 6 | 0.994000 |
| Amazon Sports | ranking | plan_e_sports_1000user_ranking_manifest | 1000 | 913 | 87 | 0.913000 |
| Amazon Sports | reranking | plan_e_sports_1000user_reranking_manifest | 1000 | 913 | 87 | 0.913000 |
| ML-1M | ranking | plan_e_ml1m_1000user_ranking_manifest | 1000 | 1000 | 0 | 1.000000 |
| ML-1M | reranking | plan_e_ml1m_1000user_reranking_manifest | 1000 | 1000 | 0 | 1.000000 |

## 10. Plan E2 matched comparison cohort

| dataset | task_type | candidate_set_id | total | eligible | excluded | rate |
| --- | --- | --- | --- | --- | --- | --- |
| Amazon Beauty | ranking | plan_e_beauty_1000user_ranking_manifest | 1000 | 994 | 6 | 0.994000 |
| Amazon Beauty | reranking | plan_e_beauty_1000user_reranking_manifest | 1000 | 994 | 6 | 0.994000 |
| Amazon Sports | ranking | plan_e_sports_1000user_ranking_manifest | 1000 | 913 | 87 | 0.913000 |
| Amazon Sports | reranking | plan_e_sports_1000user_reranking_manifest | 1000 | 913 | 87 | 0.913000 |
| ML-1M | ranking | plan_e_ml1m_1000user_ranking_manifest | 1000 | 1000 | 0 | 1.000000 |
| ML-1M | reranking | plan_e_ml1m_1000user_reranking_manifest | 1000 | 1000 | 0 | 1.000000 |

## 11. Legacy one-hundred-candidate cohort

| dataset | task_type | candidate_set_id | total | eligible | excluded | rate |
| --- | --- | --- | --- | --- | --- | --- |
| Amazon Beauty | legacy_ranking | plan_c_amazon_beauty_reconstructed_100_candidate_pool | 22363 | 22363 | 0 | 1.000000 |
| Amazon Sports | legacy_ranking | plan_c_amazon_sports_reconstructed_100_candidate_pool | 35598 | 35598 | 0 | 1.000000 |
| ML-1M | legacy_ranking | plan_b_ml1m_processed_100_candidate_pool | 6040 | 6040 | 0 | 1.000000 |
| ML-1M | legacy_ranking | plan_c_ml1m_reconstructed_100_candidate_pool | 6040 | 6040 | 0 | 1.000000 |

## 12. Plan D diagnostic cohort

| dataset | task_type | candidate_set_id | total | eligible | excluded | rate |
| --- | --- | --- | --- | --- | --- | --- |
| ML-1M | diagnostic_ranking | plan_d_qwen2_ml1m_candidate_position_bias_positive_first | 20 | 20 | 0 | 1.000000 |
| ML-1M | diagnostic_ranking | plan_d_qwen2_ml1m_candidate_position_bias_positive_last | 20 | 20 | 0 | 1.000000 |
| ML-1M | diagnostic_ranking | plan_d_qwen2_ml1m_candidate_position_bias_reverse_order | 20 | 20 | 0 | 1.000000 |
| ML-1M | diagnostic_ranking | plan_d_qwen2_ml1m_ranking_diagnostic_openp5_base | 20 | 20 | 0 | 1.000000 |

## 13. Beauty exclusions

- Beauty title exclusions: 12 rows.
- Unique missing Beauty title items among exclusions: 4.

## 14. Sports exclusions

- Sports title exclusions: 174 rows.
- Unique missing Sports title items among exclusions: 53.

## 15. ML-1M exclusions

- ML-1M title exclusions: 0 rows.
- ML-1M Plan E2 rows have complete candidate and positive title coverage.

## 16. Positive-title exclusions

- Amazon Beauty: 0 rows have unresolved positive titles.
- Amazon Sports: 4 rows have unresolved positive titles.
- ML-1M: 0 rows have unresolved positive titles.

## 17. Negative-candidate-title exclusions

- Amazon Beauty: 12 title-excluded rows have only non-positive candidate title gaps.
- Amazon Sports: 170 title-excluded rows have only non-positive candidate title gaps.
- ML-1M: 0 title-excluded rows have only non-positive candidate title gaps.

## 18. Cohort-size and retention analysis

| cohort_id | eligible_rows |
| --- | --- |
| plan_e2_paper_like_full | 6000 |
| plan_e2_llm_strict_title | 5814 |
| plan_e2_matched_comparison | 5814 |
| legacy_reconstructed_100_candidate | 70041 |
| plan_d_diagnostic_20_candidate | 80 |

## 19. Positive-position distribution comparison

| dataset | task_type | candidate_set_id | full_rows | strict_rows | full_distribution | strict_distribution |
| --- | --- | --- | --- | --- | --- | --- |
| Amazon Beauty | ranking | plan_e_beauty_1000user_ranking_manifest | 1000 | 994 | {"0": 46, "1": 61, "10": 60, "11": 54, "12": 49, "13": 56, "14": 44, "15": 53, "16": 45, "17": 55, "18": 44, "19": 51, "2": 48, "3": 57, "4": 37, "5": 56, "6": 46, "7": 41, "8": 52, "9": 45} | {"0": 46, "1": 61, "10": 60, "11": 54, "12": 49, "13": 56, "14": 44, "15": 52, "16": 45, "17": 55, "18": 44, "19": 51, "2": 48, "3": 57, "4": 37, "5": 55, "6": 45, "7": 39, "8": 51, "9": 45} |
| Amazon Beauty | reranking | plan_e_beauty_1000user_reranking_manifest | 1000 | 994 | {"0": 17, "1": 11, "10": 14, "11": 6, "12": 14, "13": 11, "14": 10, "15": 9, "16": 7, "17": 10, "18": 9, "19": 807, "2": 12, "3": 8, "4": 12, "5": 9, "6": 10, "7": 8, "8": 10, "9": 6} | {"0": 17, "1": 11, "10": 14, "11": 6, "12": 14, "13": 11, "14": 10, "15": 9, "16": 7, "17": 10, "18": 9, "19": 803, "2": 12, "3": 8, "4": 11, "5": 9, "6": 10, "7": 8, "8": 10, "9": 5} |
| ML-1M | ranking | plan_e_ml1m_1000user_ranking_manifest | 1000 | 1000 | {"0": 59, "1": 50, "10": 46, "11": 51, "12": 48, "13": 44, "14": 52, "15": 48, "16": 54, "17": 59, "18": 52, "19": 58, "2": 52, "3": 53, "4": 49, "5": 46, "6": 35, "7": 42, "8": 56, "9": 46} | {"0": 59, "1": 50, "10": 46, "11": 51, "12": 48, "13": 44, "14": 52, "15": 48, "16": 54, "17": 59, "18": 52, "19": 58, "2": 52, "3": 53, "4": 49, "5": 46, "6": 35, "7": 42, "8": 56, "9": 46} |
| ML-1M | reranking | plan_e_ml1m_1000user_reranking_manifest | 1000 | 1000 | {"0": 9, "1": 6, "10": 10, "11": 8, "12": 9, "13": 13, "14": 13, "15": 8, "16": 17, "17": 11, "18": 7, "19": 795, "2": 12, "3": 8, "4": 10, "5": 17, "6": 10, "7": 15, "8": 10, "9": 12} | {"0": 9, "1": 6, "10": 10, "11": 8, "12": 9, "13": 13, "14": 13, "15": 8, "16": 17, "17": 11, "18": 7, "19": 795, "2": 12, "3": 8, "4": 10, "5": 17, "6": 10, "7": 15, "8": 10, "9": 12} |
| Amazon Sports | ranking | plan_e_sports_1000user_ranking_manifest | 1000 | 913 | {"0": 48, "1": 58, "10": 55, "11": 49, "12": 48, "13": 48, "14": 35, "15": 53, "16": 42, "17": 54, "18": 54, "19": 54, "2": 41, "3": 41, "4": 49, "5": 52, "6": 50, "7": 51, "8": 59, "9": 59} | {"0": 43, "1": 53, "10": 51, "11": 44, "12": 44, "13": 44, "14": 35, "15": 48, "16": 42, "17": 47, "18": 50, "19": 50, "2": 39, "3": 31, "4": 44, "5": 48, "6": 44, "7": 44, "8": 56, "9": 56} |
| Amazon Sports | reranking | plan_e_sports_1000user_reranking_manifest | 1000 | 913 | {"0": 9, "1": 15, "10": 10, "11": 6, "12": 10, "13": 13, "14": 9, "15": 10, "16": 9, "17": 4, "18": 7, "19": 831, "2": 9, "3": 10, "4": 9, "5": 10, "6": 2, "7": 8, "8": 9, "9": 10} | {"0": 8, "1": 14, "10": 9, "11": 6, "12": 8, "13": 11, "14": 7, "15": 8, "16": 9, "17": 4, "18": 7, "19": 764, "2": 8, "3": 9, "4": 7, "5": 10, "6": 2, "7": 7, "8": 6, "9": 9} |

## 20. Potential selection bias introduced by title exclusions

- Beauty and Sports strict-title cohorts exclude rows with unresolved local metadata; this can change user/item composition and positive-position distributions.
- ML-1M has no Plan E2 title exclusions, so no title-gap selection filter is introduced for ML-1M in this phase.
- High retention does not prove absence of bias; future reports must label full-baseline and strict-title/matched cohorts separately.

## 21. Rules for future traditional-baseline evaluation

- Baselines may be run on `plan_e2_paper_like_full` for full-cohort baseline-only reporting.
- Baselines must also be run on `plan_e2_matched_comparison` before direct comparison with LLM results.

## 22. Rules for future LLM evaluation

- LLM prompt construction may use only `plan_e2_llm_strict_title` rows.
- Rows listed in `llm_title_exclusions.csv` must not be repaired or silently included.

## 23. Rules for direct baseline-versus-LLM comparisons

- Direct comparisons must use `plan_e2_matched_comparison`, whose membership is identical to the strict-title LLM cohort.
- Full-cohort baseline numbers may be reported only as a separately labeled baseline-only condition.

## 24. Canonical files to use in later phases

- `results/tables/experiment_cohort_membership.csv`
- `results/tables/experiment_cohort_summary.csv`
- `results/tables/llm_title_exclusions.csv`
- `results/tables/candidate_item_title_map.csv`

## 25. Files that must not be mixed

- Do not mix Plan B/C 100-candidate rows with Plan E2 20-candidate paper-like rows.
- Do not mix Plan D diagnostic rows with main paper-like results.
- Do not compare Plan E2 full baseline rows directly against strict-title LLM rows without matched-cohort recomputation.

## 26. File-size and repository-management note

- Full item-title map size: 164876923 bytes.
- Candidate-only title map size: 6632503 bytes.
- No file was deleted, compressed, or replaced in this task.

## 27. Tests and validation commands

- `$env:PYTHONNOUSERSITE='1'; & 'C:\ProgramData\anaconda3\python.exe' src/experiments/build_experiment_cohorts.py`
- `$env:PYTHONNOUSERSITE='1'; & 'C:\ProgramData\anaconda3\python.exe' -m pytest tests/test_experiment_cohort_policy.py tests/test_item_title_map_reconciliation.py tests/test_dataset_candidate_set_validation.py`
- validated 34341 candidate_item_title_map rows
- validated 76121 experiment_cohort_membership rows
- validated 186 llm_title_exclusions rows
- validated 26 experiment_cohort_summary rows

## 28. Recommended single next task

Phase 3 traditional-baseline consolidation on both the Plan E2 full cohort and the Plan E2 matched comparison cohort.
