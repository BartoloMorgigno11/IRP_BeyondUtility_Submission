# Canonical Local LLM Reranking Prompt-V2 Corrective Pilot

## 1. Scope and provenance

Phase 5B-P1D executed the frozen 60-request reranking prompt-v2 corrective pilot. Environment: `{'executable': 'C:\\Users\\Bartolo\\Desktop\\PROGETTI\\IRP_BeyondUtility_Project\\.venv\\Scripts\\python.exe', 'python_version': '3.12.10', 'pandas_version': '3.0.3', 'numpy_version': '2.5.0', 'os': 'Windows-11-10.0.26200-SP0', 'cpu': 'Intel64 Family 6 Model 183 Stepping 1, GenuineIntel', 'ram_bytes': '34053414912', 'gpu': 'Intel(R) UHD Graphics;NVIDIA GeForce RTX 4060 Laptop GPU', 'cwd': 'C:\\Users\\Bartolo\\Desktop\\PROGETTI\\IRP_BeyondUtility_Project'}`.

## 2. Executive summary

Completed `60/60` requests. Decision: `require_reranking_prompt_v3_audit`.

## 3. Safety boundaries

Local loopback Ollama only; no web, external API, model pull, ranking prompts, full matrix, recommender, or beyond-utility execution.

## 4. Prompt-v1 pilot findings

Prompt v1 reranking had 53/60 strict-valid responses and seven complete-JSON contract violations.

## 5. Prompt-corrective audit findings

The prompt-corrective audit found no parser defect or truncation root cause and approved a common reranking prompt v2.

## 6. Reranking prompt-v2 contract

`llm_rerank_index_json_full20_v2` explicitly requires one full permutation of integers 1 through 20.

## 7. Scientific-semantic invariance

Only reranking prompt wording changed; candidates, histories, model tags, parser, and execution parameters were preserved.

## 8. Frozen corrective manifest

The manifest was validated as 60 reranking rows: 30 Qwen2, 30 Mistral, 10 per model/dataset.

## 9. Sample and row identity

Every corrective row maps to exactly one original v1 reranking source request.

## 10. Candidate-order identity

All paired rows preserve source candidate-order hashes.

## 11. History identity

All paired rows preserve source history-render hashes.

## 12. Local Ollama verification

Verified `/api/version`, `/api/tags`, and `/api/ps` before execution.

## 13. Qwen2 digest verification

dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755

## 14. Mistral digest verification

6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf

## 15. Resume-safe execution

Completed requests are skipped on rerun; ambiguous or duplicate raw evidence blocks resend.

## 16. Raw-request retention

Request records retained: `60`.

## 17. Raw-response retention

Response/error records retained: `60`.

## 18. Backend completion

Backend completion rate: `1`.

## 19. Latency and token counts

| model_family | dataset | median_duration_seconds | mean_duration_seconds | median_generated_token_count |
| --- | --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | 4.71687390003 | 4.89880967997 | 77 |
| Qwen2 7B Instruct | Amazon Sports | 4.60894885008 | 4.65703872 | 77 |
| Qwen2 7B Instruct | ML-1M | 4.66184684995 | 4.69276400998 | 78 |
| Mistral 7B Instruct | Amazon Beauty | 4.55066165002 | 4.84191829998 | 81 |
| Mistral 7B Instruct | Amazon Sports | 4.44898754987 | 4.45158338994 | 81 |
| Mistral 7B Instruct | ML-1M | 4.35452455002 | 4.35366396 | 81 |

## 20. Strict-parser findings

model_family         dataset        strict_parse_valid
Mistral 7B Instruct  Amazon Beauty  true                  10
                     Amazon Sports  false                  1
                                    true                   9
                     ML-1M          false                  1
                                    true                   9
Qwen2 7B Instruct    Amazon Beauty  false                  6
                                    true                   4
                     Amazon Sports  false                  5
                                    true                   5
                     ML-1M          false                  6
                                    true                   4

## 21. Diagnostic salvage findings

model_family         dataset        diagnostic_salvage_valid
Mistral 7B Instruct  Amazon Beauty  true                        10
                     Amazon Sports  false                        1
                                    true                         9
                     ML-1M          false                        1
                                    true                         9
Qwen2 7B Instruct    Amazon Beauty  false                        6
                                    true                         4
                     Amazon Sports  false                        5
                                    true                         5
                     ML-1M          false                        6
                                    true                         4

## 22. Output-count findings

model_family         dataset        output_count_valid
Mistral 7B Instruct  Amazon Beauty  true                  10
                     Amazon Sports  false                  1
                                    true                   9
                     ML-1M          false                  1
                                    true                   9
Qwen2 7B Instruct    Amazon Beauty  false                  5
                                    true                   5
                     Amazon Sports  false                  5
                                    true                   5
                     ML-1M          false                  4
                                    true                   6

## 23. Uniqueness findings

model_family         dataset        uniqueness_valid
Mistral 7B Instruct  Amazon Beauty  true                10
                     Amazon Sports  true                10
                     ML-1M          true                10
Qwen2 7B Instruct    Amazon Beauty  false                1
                                    true                 9
                     Amazon Sports  true                10
                     ML-1M          false                2
                                    true                 8

## 24. Full-permutation findings

model_family         dataset        full_permutation_valid
Mistral 7B Instruct  Amazon Beauty  true                      10
                     Amazon Sports  false                      1
                                    true                       9
                     ML-1M          false                      1
                                    true                       9
Qwen2 7B Instruct    Amazon Beauty  false                      6
                                    true                       4
                     Amazon Sports  false                      5
                                    true                       5
                     ML-1M          false                      6
                                    true                       4

## 25. Candidate-validity findings

model_family         dataset        candidate_validity
Mistral 7B Instruct  Amazon Beauty  true                  10
                     Amazon Sports  false                  1
                                    true                   9
                     ML-1M          false                  1
                                    true                   9
Qwen2 7B Instruct    Amazon Beauty  false                  6
                                    true                   4
                     Amazon Sports  false                  5
                                    true                   5
                     ML-1M          false                  6
                                    true                   4

## 26. Duplicate and hallucination findings

model_family         duplicate_count  hallucination_count
Mistral 7B Instruct  0                0                      30
Qwen2 7B Instruct    0                0                      27
                     1                0                       2
                     2                0                       1

## 27. Truncation assessment

output_truncation_detected
false    60

## 28. Qwen2 Beauty findings

| dataset | task_type | corrective_rows | backend_success_rows | strict_valid_rows | strict_parse_validity_rate | diagnostic_salvage_valid_rows | diagnostic_salvage_validity_rate | output_count_valid_rows | uniqueness_valid_rows | full_permutation_valid_rows | candidate_valid_rows | duplicate_free_rows | hallucination_free_rows | utility_evaluated_rows | hr_at_5_valid_only | ndcg_at_5_valid_only | mean_positive_rank_valid_only | median_duration_seconds | mean_duration_seconds | median_generated_token_count | readiness_threshold | cell_readiness_pass | metric_status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Amazon Beauty | reranking | 10 | 10 | 4 | 0.4 | 4 | 0.4 | 5 | 9 | 4 | 4 | 4 | 4 | 4 | 0 | 0 | 18.25 | 4.71687390003 | 4.89880967997 | 77 | 0.8 | false | valid_only_utility_diagnostic | Corrective-pilot utility is diagnostic only and not final model performance. |

## 29. Qwen2 Sports findings

| dataset | task_type | corrective_rows | backend_success_rows | strict_valid_rows | strict_parse_validity_rate | diagnostic_salvage_valid_rows | diagnostic_salvage_validity_rate | output_count_valid_rows | uniqueness_valid_rows | full_permutation_valid_rows | candidate_valid_rows | duplicate_free_rows | hallucination_free_rows | utility_evaluated_rows | hr_at_5_valid_only | ndcg_at_5_valid_only | mean_positive_rank_valid_only | median_duration_seconds | mean_duration_seconds | median_generated_token_count | readiness_threshold | cell_readiness_pass | metric_status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Amazon Sports | reranking | 10 | 10 | 5 | 0.5 | 5 | 0.5 | 5 | 10 | 5 | 5 | 5 | 5 | 5 | 0 | 0 | 14.4 | 4.60894885008 | 4.65703872 | 77 | 0.8 | false | valid_only_utility_diagnostic | Corrective-pilot utility is diagnostic only and not final model performance. |

## 30. Qwen2 ML-1M findings

| dataset | task_type | corrective_rows | backend_success_rows | strict_valid_rows | strict_parse_validity_rate | diagnostic_salvage_valid_rows | diagnostic_salvage_validity_rate | output_count_valid_rows | uniqueness_valid_rows | full_permutation_valid_rows | candidate_valid_rows | duplicate_free_rows | hallucination_free_rows | utility_evaluated_rows | hr_at_5_valid_only | ndcg_at_5_valid_only | mean_positive_rank_valid_only | median_duration_seconds | mean_duration_seconds | median_generated_token_count | readiness_threshold | cell_readiness_pass | metric_status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ML-1M | reranking | 10 | 10 | 4 | 0.4 | 4 | 0.4 | 6 | 8 | 4 | 4 | 4 | 4 | 4 | 0.25 | 0.25 | 9.75 | 4.66184684995 | 4.69276400998 | 78 | 0.8 | false | valid_only_utility_diagnostic | Corrective-pilot utility is diagnostic only and not final model performance. |

## 31. Mistral Beauty findings

| dataset | task_type | corrective_rows | backend_success_rows | strict_valid_rows | strict_parse_validity_rate | diagnostic_salvage_valid_rows | diagnostic_salvage_validity_rate | output_count_valid_rows | uniqueness_valid_rows | full_permutation_valid_rows | candidate_valid_rows | duplicate_free_rows | hallucination_free_rows | utility_evaluated_rows | hr_at_5_valid_only | ndcg_at_5_valid_only | mean_positive_rank_valid_only | median_duration_seconds | mean_duration_seconds | median_generated_token_count | readiness_threshold | cell_readiness_pass | metric_status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Amazon Beauty | reranking | 10 | 10 | 10 | 1 | 10 | 1 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 0.2 | 0.2 | 12.9 | 4.55066165002 | 4.84191829998 | 81 | 0.8 | true | valid_only_utility_diagnostic | Corrective-pilot utility is diagnostic only and not final model performance. |

## 32. Mistral Sports findings

| dataset | task_type | corrective_rows | backend_success_rows | strict_valid_rows | strict_parse_validity_rate | diagnostic_salvage_valid_rows | diagnostic_salvage_validity_rate | output_count_valid_rows | uniqueness_valid_rows | full_permutation_valid_rows | candidate_valid_rows | duplicate_free_rows | hallucination_free_rows | utility_evaluated_rows | hr_at_5_valid_only | ndcg_at_5_valid_only | mean_positive_rank_valid_only | median_duration_seconds | mean_duration_seconds | median_generated_token_count | readiness_threshold | cell_readiness_pass | metric_status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Amazon Sports | reranking | 10 | 10 | 9 | 0.9 | 9 | 0.9 | 9 | 10 | 9 | 9 | 9 | 9 | 9 | 0.111111111111 | 0.0429836452483 | 17 | 4.44898754987 | 4.45158338994 | 81 | 0.8 | true | valid_only_utility_diagnostic | Corrective-pilot utility is diagnostic only and not final model performance. |

## 33. Mistral ML-1M findings

| dataset | task_type | corrective_rows | backend_success_rows | strict_valid_rows | strict_parse_validity_rate | diagnostic_salvage_valid_rows | diagnostic_salvage_validity_rate | output_count_valid_rows | uniqueness_valid_rows | full_permutation_valid_rows | candidate_valid_rows | duplicate_free_rows | hallucination_free_rows | utility_evaluated_rows | hr_at_5_valid_only | ndcg_at_5_valid_only | mean_positive_rank_valid_only | median_duration_seconds | mean_duration_seconds | median_generated_token_count | readiness_threshold | cell_readiness_pass | metric_status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ML-1M | reranking | 10 | 10 | 9 | 0.9 | 9 | 0.9 | 9 | 10 | 9 | 9 | 9 | 9 | 9 | 0.333333333333 | 0.251317723016 | 9.88888888889 | 4.35452455002 | 4.35366396 | 81 | 0.8 | true | valid_only_utility_diagnostic | Corrective-pilot utility is diagnostic only and not final model performance. |

## 34. Prompt-v1 versus prompt-v2 validity

model_family         dataset        strict_validity_change
Mistral 7B Instruct  Amazon Beauty  improved                  4
                                    same_valid                6
                     Amazon Sports  improved                  1
                                    regressed                 1
                                    same_valid                8
                     ML-1M          regressed                 1
                                    same_valid                9
Qwen2 7B Instruct    Amazon Beauty  regressed                 6
                                    same_valid                4
                     Amazon Sports  improved                  1
                                    regressed                 5
                                    same_valid                4
                     ML-1M          regressed                 5
                                    same_invalid              1
                                    same_valid                4

## 35. Utility diagnostics

| model_family | dataset | utility_evaluated_rows | hr_at_5_valid_only | ndcg_at_5_valid_only | mean_positive_rank_valid_only |
| --- | --- | --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | 4 | 0 | 0 | 18.25 |
| Qwen2 7B Instruct | Amazon Sports | 5 | 0 | 0 | 14.4 |
| Qwen2 7B Instruct | ML-1M | 4 | 0.25 | 0.25 | 9.75 |
| Mistral 7B Instruct | Amazon Beauty | 10 | 0.2 | 0.2 | 12.9 |
| Mistral 7B Instruct | Amazon Sports | 9 | 0.111111111111 | 0.0429836452483 | 17 |
| Mistral 7B Instruct | ML-1M | 9 | 0.333333333333 | 0.251317723016 | 9.88888888889 |

## 36. Why utility does not determine readiness

Readiness is based on strict parse validity, candidate validity, complete permutations, backend completion, hashes, and retention; utility is descriptive only.

## 37. Readiness threshold

Threshold remains `0.8` for each of six cells.

## 38. Full-execution decision

`require_reranking_prompt_v3_audit`.

## 39. Approved full prompt combination

Ranking: `llm_rank_index_json_top5_v1`; reranking: `llm_rerank_index_json_full20_v2`.

## 40. Required corrective action

Run a focused prompt-v3 audit for readiness cells below 0.80.

## 41. Limitations

This is a 60-request corrective pilot; it is not final model performance and does not run Phase 5B-P2.

## 42. Protected-input immutability

Protected-input validation: `true` prompt hashes, `true` candidate hashes, `true` history hashes.

## 43. Commands executed

- `git status --short --branch`
- `git branch --show-current`
- `git log -1 --format="%H %s"`
- `$env:PYTHONNOUSERSITE='1'; & '.venv\Scripts\python.exe' -m pytest tests\test_canonical_local_llm_reranking_prompt_v2_runner.py tests\test_canonical_local_llm_pilot_invalid_output_audit.py tests\test_canonical_local_llm_pilot_resume.py tests\test_canonical_local_llm_runner.py tests\test_canonical_llm_contract.py tests\test_experiment_cohort_policy.py tests\test_dataset_candidate_set_validation.py -q`
- `$env:PYTHONNOUSERSITE='1'; & '.venv\Scripts\python.exe' src\experiments\run_canonical_local_llm_reranking_prompt_v2_corrective_pilot.py`

## 44. Recommended single next task

Run a focused prompt-v3 audit for readiness cells below 0.80.
