# Canonical Local LLM Hybrid Disjoint Holdout Pilot

## 1. Scope and provenance

Executed Phase 5B-P1E on branch `plan-d-extensions` with Python `3.12.10` from `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe`.

## 2. Executive summary

Decision: `hybrid_policy_failed_disjoint_holdout`. Completed 60/60 frozen holdout requests.

## 3. Safety boundaries

Only local loopback Ollama was used. No ranking request, full matrix execution, model pull, download, prompt-v3 creation, or recommender training was performed.

## 4. Hybrid-policy audit background

Hybrid policy `canonical_local_llm_hybrid_qwen_v1_mistral_v2_rerank_policy_v1` was approved in the offline audit as an adapted model-specific format-compliance contract.

## 5. Why a disjoint holdout was required

The disjoint holdout was required to mitigate post-hoc prompt-selection risk after v1/v2 structural-validity pilots.

## 6. Frozen hybrid policy

| hybrid_policy_path | holdout_sample_path | holdout_manifest_path | previous_pilot_overlap_count |
| --- | --- | --- | --- |
| results/tables/canonical_local_llm_hybrid_prompt_policy_contract.csv | results/tables/canonical_local_llm_hybrid_holdout_sample.csv | results/tables/canonical_local_llm_hybrid_holdout_request_manifest.csv | 0 |

## 7. Qwen2 prompt assignment

Qwen2 used `llm_rerank_index_json_full20_v1` for every reranking holdout request.

## 8. Mistral prompt assignment

Mistral used `llm_rerank_index_json_full20_v2` for every reranking holdout request.

## 9. Shared parser and output space

All requests used `llm_rerank_strict_primary_parser_v1` and required a 20-item one-based full permutation.

## 10. Holdout population

The population was frozen to prompt-compatible reranking rows from Beauty, Sports, and ML-1M.

## 11. Previous-pilot exclusion

Independent overlap verification found `0` previous-pilot overlaps.

## 12. Deterministic holdout sample

The 30-row holdout sample was not rebuilt or resampled during execution.

## 13. Holdout sample composition

| dataset | holdout_rows |
| --- | --- |
| Amazon Beauty | 1010 |
| Amazon Sports | 1010 |
| ML-1M | 1010 |

## 14. Frozen request manifest

The manifest contains 60 stored reranking requests, 30 per model, with contiguous request order.

## 15. Local Ollama verification

Ollama server version `0.31.1` was verified through local admin endpoints.

## 16. Qwen2 digest verification

Qwen2 digest verified as `dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755`.

## 17. Mistral digest verification

Mistral digest verified as `6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf`.

## 18. Resume-safe execution

Execution was append-only and resume-safe: completed requests are skipped, ambiguous requests block rerun.

## 19. Raw-request retention

Raw request records retained: `60`.

## 20. Raw-response retention

Raw response/error records retained: `60`.

## 21. Backend completion

Backend completion rate: `1`.

## 22. Latency and token findings

| model_family | dataset | median_duration_seconds | median_generated_token_count |
| --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | 4.31399239995 | 58 |
| Qwen2 7B Instruct | Amazon Sports | 4.25297070004 | 58 |
| Qwen2 7B Instruct | ML-1M | 4.22406729997 | 58 |
| Mistral 7B Instruct | Amazon Beauty | 4.50569224998 | 81 |
| Mistral 7B Instruct | Amazon Sports | 4.44568055007 | 81 |
| Mistral 7B Instruct | ML-1M | 4.33630634984 | 81 |

## 23. Strict-parser findings

| model_family | dataset | strict_valid_rows | strict_parse_validity_rate | cell_readiness_pass |
| --- | --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | 8 | 0.8 | true |
| Qwen2 7B Instruct | Amazon Sports | 10 | 1 | true |
| Qwen2 7B Instruct | ML-1M | 10 | 1 | true |
| Mistral 7B Instruct | Amazon Beauty | 9 | 0.9 | true |
| Mistral 7B Instruct | Amazon Sports | 7 | 0.7 | false |
| Mistral 7B Instruct | ML-1M | 9 | 0.9 | true |

## 24. Diagnostic salvage findings

| model_family | dataset | diagnostic_salvage_valid_rows | diagnostic_salvage_validity_rate |
| --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | 8 | 0.8 |
| Qwen2 7B Instruct | Amazon Sports | 10 | 1 |
| Qwen2 7B Instruct | ML-1M | 10 | 1 |
| Mistral 7B Instruct | Amazon Beauty | 9 | 0.9 |
| Mistral 7B Instruct | Amazon Sports | 7 | 0.7 |
| Mistral 7B Instruct | ML-1M | 9 | 0.9 |

## 25. Output-count findings

| model_family | dataset | output_count_valid_rows | holdout_rows |
| --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | 9 | 10 |
| Qwen2 7B Instruct | Amazon Sports | 10 | 10 |
| Qwen2 7B Instruct | ML-1M | 10 | 10 |
| Mistral 7B Instruct | Amazon Beauty | 10 | 10 |
| Mistral 7B Instruct | Amazon Sports | 7 | 10 |
| Mistral 7B Instruct | ML-1M | 9 | 10 |

## 26. Uniqueness findings

| model_family | dataset | uniqueness_valid_rows | holdout_rows |
| --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | 9 | 10 |
| Qwen2 7B Instruct | Amazon Sports | 10 | 10 |
| Qwen2 7B Instruct | ML-1M | 10 | 10 |
| Mistral 7B Instruct | Amazon Beauty | 10 | 10 |
| Mistral 7B Instruct | Amazon Sports | 8 | 10 |
| Mistral 7B Instruct | ML-1M | 10 | 10 |

## 27. Full-permutation findings

| model_family | dataset | full_permutation_valid_rows | holdout_rows |
| --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | 8 | 10 |
| Qwen2 7B Instruct | Amazon Sports | 10 | 10 |
| Qwen2 7B Instruct | ML-1M | 10 | 10 |
| Mistral 7B Instruct | Amazon Beauty | 9 | 10 |
| Mistral 7B Instruct | Amazon Sports | 7 | 10 |
| Mistral 7B Instruct | ML-1M | 9 | 10 |

## 28. Candidate-validity findings

| model_family | dataset | candidate_valid_rows | strict_valid_rows |
| --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | 8 | 8 |
| Qwen2 7B Instruct | Amazon Sports | 10 | 10 |
| Qwen2 7B Instruct | ML-1M | 10 | 10 |
| Mistral 7B Instruct | Amazon Beauty | 9 | 9 |
| Mistral 7B Instruct | Amazon Sports | 7 | 7 |
| Mistral 7B Instruct | ML-1M | 9 | 9 |

## 29. Duplicate and hallucination findings

| model_family | dataset | duplicate_free_rows | hallucination_free_rows | strict_valid_rows |
| --- | --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | 8 | 8 | 8 |
| Qwen2 7B Instruct | Amazon Sports | 10 | 10 | 10 |
| Qwen2 7B Instruct | ML-1M | 10 | 10 | 10 |
| Mistral 7B Instruct | Amazon Beauty | 9 | 9 | 9 |
| Mistral 7B Instruct | Amazon Sports | 7 | 7 | 7 |
| Mistral 7B Instruct | ML-1M | 9 | 9 | 9 |

## 30. Truncation assessment

Output truncation detected: `false`.

## 31. Qwen2 Beauty readiness

Qwen2 7B Instruct Amazon Beauty: `8/10` strict-valid, rate `0.8`, readiness pass `true`.

## 32. Qwen2 Sports readiness

Qwen2 7B Instruct Amazon Sports: `10/10` strict-valid, rate `1`, readiness pass `true`.

## 33. Qwen2 ML-1M readiness

Qwen2 7B Instruct ML-1M: `10/10` strict-valid, rate `1`, readiness pass `true`.

## 34. Mistral Beauty readiness

Mistral 7B Instruct Amazon Beauty: `9/10` strict-valid, rate `0.9`, readiness pass `true`.

## 35. Mistral Sports readiness

Mistral 7B Instruct Amazon Sports: `7/10` strict-valid, rate `0.7`, readiness pass `false`.

## 36. Mistral ML-1M readiness

Mistral 7B Instruct ML-1M: `9/10` strict-valid, rate `0.9`, readiness pass `true`.

## 37. Prior-pilot versus holdout validity

| model_family | dataset | prior_strict_parse_validity_rate | holdout_strict_parse_validity_rate | validity_rate_difference |
| --- | --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | 1 | 0.8 | -0.2 |
| Qwen2 7B Instruct | Amazon Sports | 0.9 | 1 | 0.1 |
| Qwen2 7B Instruct | ML-1M | 0.9 | 1 | 0.1 |
| Mistral 7B Instruct | Amazon Beauty | 1 | 0.9 | -0.1 |
| Mistral 7B Instruct | Amazon Sports | 0.9 | 0.7 | -0.2 |
| Mistral 7B Instruct | ML-1M | 0.9 | 0.9 | 0 |

## 38. Utility diagnostics

| model_family | dataset | utility_evaluated_rows | hr_at_5_valid_only | ndcg_at_5_valid_only |
| --- | --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | 8 | 0.125 | 0.125 |
| Qwen2 7B Instruct | Amazon Sports | 10 | 0.2 | 0.138685280724 |
| Qwen2 7B Instruct | ML-1M | 10 | 0.1 | 0.1 |
| Mistral 7B Instruct | Amazon Beauty | 9 | 0.222222222222 | 0.140206611905 |
| Mistral 7B Instruct | Amazon Sports | 7 | 0 | 0 |
| Mistral 7B Instruct | ML-1M | 9 | 0.333333333333 | 0.183190257153 |

## 39. Why utility does not determine readiness

Utility diagnostics were computed only for strict-valid rows and are not used for readiness or prompt selection.

## 40. Holdout readiness threshold

Frozen readiness threshold: `0.8` with at least 8/10 strict-valid rows per cell.

## 41. Hybrid-policy generalization assessment

Hybrid-policy generalization assessment: `hybrid_policy_failed_disjoint_holdout`.

## 42. Full-execution decision

Full-execution decision: `hybrid_policy_failed_disjoint_holdout`.

## 43. Approved full prompt combination

Approved full prompt combination records ranking `llm_rank_index_json_top5_v1`, Qwen2 reranking `llm_rerank_index_json_full20_v1`, Mistral reranking `llm_rerank_index_json_full20_v2`.

## 44. Required corrective action

Phase 5B-P1F - final reranking protocol decision audit after hybrid holdout failure

## 45. Limitations

This is a 60-request structural holdout pilot, not a final full-cohort performance estimate.

## 46. Protected-input immutability

| check_name | status | observed_value |
| --- | --- | --- |
| protected_hashes_unchanged | pass | true |
| no_prior_artifact_mutation | pass | true |

## 47. Commands executed

- `git status --short --branch`
- `git branch --show-current`
- `git log -1 --format="%H %s"`
- `$env:PYTHONNOUSERSITE='1'; & '.venv\Scripts\python.exe' -m pytest tests\test_canonical_local_llm_hybrid_holdout_runner.py tests\test_canonical_local_llm_hybrid_prompt_holdout_audit.py tests\test_canonical_local_llm_reranking_prompt_v2_runner.py tests\test_canonical_local_llm_pilot_invalid_output_audit.py tests\test_canonical_local_llm_pilot_resume.py tests\test_canonical_local_llm_runner.py tests\test_canonical_llm_contract.py tests\test_experiment_cohort_policy.py tests\test_dataset_candidate_set_validation.py -q`
- `$env:PYTHONNOUSERSITE='1'; & '.venv\Scripts\python.exe' src\experiments\run_canonical_local_llm_hybrid_holdout_pilot.py`

## 48. Recommended single next task

Recommended single next task: `Phase 5B-P1F - final reranking protocol decision audit after hybrid holdout failure`.
