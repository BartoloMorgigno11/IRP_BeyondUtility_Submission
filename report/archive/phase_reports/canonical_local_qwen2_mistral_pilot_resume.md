# Resumed Canonical Local Qwen2 and Mistral Pilot

## 1. Scope and provenance

Phase 5B-P1C resumed the frozen 120-request local pilot on branch `plan-d-extensions`. Environment: `{'executable': 'C:\\Users\\Bartolo\\Desktop\\PROGETTI\\IRP_BeyondUtility_Project\\.venv\\Scripts\\python.exe', 'python_version': '3.12.10', 'pandas_version': '3.0.3', 'numpy_version': '2.5.0', 'os': 'Windows-11-10.0.26200-SP0', 'cpu': 'Intel64 Family 6 Model 183 Stepping 1, GenuineIntel', 'ram_bytes': '34053414912', 'gpu': 'Intel(R) UHD Graphics;NVIDIA GeForce RTX 4060 Laptop GPU', 'cwd': 'C:\\Users\\Bartolo\\Desktop\\PROGETTI\\IRP_BeyondUtility_Project'}`.

## 2. Executive summary

Completed requests: `120`. Decision: `require_prompt_contract_revision`. This is a pilot readiness result, not final full-cohort performance.

## 3. Safety boundaries

No external API, public web, model pull, package install, full matrix execution, nondeterminism, position, history-length, profile, or recommender execution was performed.

## 4. Frozen Phase 5B-P1 sample

`results/tables/canonical_local_llm_pilot_sample.csv` was validated and not modified.

## 5. Frozen Phase 5B-P1 manifest

`results/tables/canonical_local_llm_pilot_request_manifest.csv` was validated and not modified.

## 6. Phase 5B-P1 blocked evidence

The original blocked decision remains `backend_blocked` and was not altered.

## 7. Phase 5B-P1B recovery evidence

The recovery CSV verified Ollama `0.31.1` and both exact local digests.

## 8. Ollama service verification

Only `/api/version`, `/api/tags`, `/api/ps`, and `/api/generate` for the frozen requests were used; no remote endpoint was contacted.

## 9. Qwen2 tag and digest verification

`qwen2:7b-instruct` digest `dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755`.

## 10. Mistral tag and digest verification

`mistral:7b-instruct` digest `6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf`.

## 11. Resume-safe execution design

Raw requests, responses, and runner events were written append-only with fsync. Completed requests are skipped on rerun; ambiguous request state blocks resend.

## 12. Initial resume state

Initial pending/completed/ambiguous counts are in `results/tables/canonical_local_llm_pilot_resume_provenance.csv`.

## 13. Ambiguous-request prevention

Ambiguous request count: `0`.

## 14. Request ordering

Requests followed the exact stored `request_order` in the frozen manifest.

## 15. Prompt and candidate integrity

Stored prompt hashes and candidate-order hashes were reconciled before each send.

## 16. Positive-label concealment

The model-visible payload omitted positive item IDs, positions, relevance labels, and expected responses.

## 17. Raw-request retention

Request records retained: `120`.

## 18. Raw-response retention

Response/error records retained: `120`.

## 19. Backend execution findings

Backend completion rate: `1`; backend-error records: `0`.

## 20. Qwen2 latency and token findings

| dataset | task_type | median_duration_seconds | median_generated_token_count |
| --- | --- | --- | --- |
| Amazon Beauty | ranking | 4.065374750061892 | 18 |
| Amazon Beauty | reranking | 4.4398190499050543 | 58 |
| Amazon Sports | ranking | 3.4425909500569105 | 18.5 |
| Amazon Sports | reranking | 4.8832687999820337 | 58 |
| ML-1M | ranking | 3.7796093000797555 | 18.5 |
| ML-1M | reranking | 4.9626983000198379 | 58 |

## 21. Mistral latency and token findings

| dataset | task_type | median_duration_seconds | median_generated_token_count |
| --- | --- | --- | --- |
| Amazon Beauty | ranking | 4.1596961501054466 | 27 |
| Amazon Beauty | reranking | 5.6844665999524295 | 83 |
| Amazon Sports | ranking | 3.8822132499190047 | 28 |
| Amazon Sports | reranking | 5.0840945499949157 | 82 |
| ML-1M | ranking | 3.6146396500989795 | 28 |
| ML-1M | reranking | 4.9930805000476539 | 81 |

## 22. Strict parser findings

model_family         dataset        task_type  strict_parse_valid
Mistral 7B Instruct  Amazon Beauty  ranking    true                  10
                                    reranking  false                  4
                                               true                   6
                     Amazon Sports  ranking    true                  10
                                    reranking  false                  1
                                               true                   9
                     ML-1M          ranking    true                  10
                                    reranking  true                  10
Qwen2 7B Instruct    Amazon Beauty  ranking    true                  10
                                    reranking  true                  10
                     Amazon Sports  ranking    true                  10
                                    reranking  false                  1
                                               true                   9
                     ML-1M          ranking    true                  10
                                    reranking  false                  1
                                               true                   9

## 23. Diagnostic salvage findings

model_family         dataset        task_type  diagnostic_salvage_valid
Mistral 7B Instruct  Amazon Beauty  ranking    true                        10
                                    reranking  false                        4
                                               true                         6
                     Amazon Sports  ranking    true                        10
                                    reranking  false                        1
                                               true                         9
                     ML-1M          ranking    true                        10
                                    reranking  true                        10
Qwen2 7B Instruct    Amazon Beauty  ranking    true                        10
                                    reranking  true                        10
                     Amazon Sports  ranking    true                        10
                                    reranking  false                        1
                                               true                         9
                     ML-1M          ranking    true                        10
                                    reranking  false                        1
                                               true                         9

## 24. Output-length findings

task_type  output_count_valid
ranking    true                  60
reranking  false                  6
           true                  54

## 25. Candidate-validity findings

model_family         candidate_validity
Mistral 7B Instruct  false                  5
                     true                  55
Qwen2 7B Instruct    false                  2
                     true                  58

## 26. Duplicate findings

model_family         duplicate_count
Mistral 7B Instruct  0                  56
                     1                   1
                     2                   2
                     9                   1
Qwen2 7B Instruct    0                  60

## 27. Hallucination findings

model_family         hallucination_count
Mistral 7B Instruct  0                      59
                     1                       1
Qwen2 7B Instruct    0                      60

## 28. Reranking-permutation findings

model_family         full_permutation_valid
Mistral 7B Instruct  false                      5
                     true                      25
Qwen2 7B Instruct    false                      2
                     true                      28

## 29. Qwen2 utility findings

| dataset | task_type | utility_evaluated_rows | hr_at_5_valid_only | ndcg_at_5_valid_only |
| --- | --- | --- | --- | --- |
| Amazon Beauty | ranking | 10 | 0.10000000000000001 | 0.063092975357145753 |
| Amazon Beauty | reranking | 10 | 0.10000000000000001 | 0.050000000000000003 |
| Amazon Sports | ranking | 10 | 0.40000000000000002 | 0.24922828697182436 |
| Amazon Sports | reranking | 9 | 0.22222222222222221 | 0.15409475635939351 |
| ML-1M | ranking | 10 | 0.10000000000000001 | 0.063092975357145753 |
| ML-1M | reranking | 9 | 0.1111111111111111 | 0.042983645248282396 |

## 30. Mistral utility findings

| dataset | task_type | utility_evaluated_rows | hr_at_5_valid_only | ndcg_at_5_valid_only |
| --- | --- | --- | --- | --- |
| Amazon Beauty | ranking | 10 | 0.20000000000000001 | 0.088685280723454163 |
| Amazon Beauty | reranking | 6 | 1 | 0.43917645510264486 |
| Amazon Sports | ranking | 10 | 0.20000000000000001 | 0.16309297535714576 |
| Amazon Sports | reranking | 9 | 0.1111111111111111 | 0.042983645248282396 |
| ML-1M | ranking | 10 | 0.59999999999999998 | 0.35538891813630941 |
| ML-1M | reranking | 10 | 0.5 | 0.33175293653079346 |

## 31. Dataset-specific findings

dataset        model_family       
Amazon Beauty  Mistral 7B Instruct    2
               Qwen2 7B Instruct      2
Amazon Sports  Mistral 7B Instruct    2
               Qwen2 7B Instruct      2
ML-1M          Mistral 7B Instruct    2
               Qwen2 7B Instruct      2

## 32. Ranking versus reranking findings

task_type  model_family       
ranking    Mistral 7B Instruct    10,10,10
           Qwen2 7B Instruct      10,10,10
reranking  Mistral 7B Instruct      6,9,10
           Qwen2 7B Instruct        10,9,9

## 33. Pilot utility interpretation

Utility is computed on strict-valid pilot outputs only. Invalid outputs are retained and excluded rather than counted as zero. These values are not final full-cohort performance.

## 34. Readiness threshold

Threshold remains `0.80000000000000004` and is result-independent.

## 35. Full-execution decision

`require_prompt_contract_revision`.

## 36. Required corrective action

Phase 5B-P1C-PROMPT - focused prompt/parser corrective audit for cells below threshold

## 37. Limitations

This is a 120-request pilot. It does not execute the 11,528-request full matrix or beyond-utility metrics.

## 38. Protected-input immutability

Protected hashes were recomputed after execution and validated unchanged.

## 39. Commands executed

- `git status --short --branch`
- `git branch --show-current`
- `git log -1 --format="%H %s"`
- `PYTHONNOUSERSITE=1 C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe -m pytest tests\test_canonical_local_llm_pilot_resume.py tests\test_canonical_local_llm_runner.py tests\test_canonical_llm_contract.py tests\test_experiment_cohort_policy.py tests\test_dataset_candidate_set_validation.py`
- `PYTHONNOUSERSITE=1 C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe src\experiments\run_canonical_local_llm_pilot_resume.py`

## 40. Recommended single next task

Phase 5B-P1C-PROMPT - focused prompt/parser corrective audit for cells below threshold
