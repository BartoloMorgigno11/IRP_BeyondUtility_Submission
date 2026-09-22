# Canonical Local LLM Full Matrix Execution

## 1. Scope and provenance

Phase: canonical_local_llm_full_matrix_v1. This report is generated from full-matrix raw JSONL and derived CSV files.

## 2. Environment

- executable: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe`
- python: `3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]`
- pandas: `3.0.3`
- numpy: `2.5.0`
- platform: `Windows-11-10.0.26200-SP0`
- cpu: `Intel64 Family 6 Model 183 Stepping 1, GenuineIntel`
- ram_bytes: `34053414912`
- gpu: `Intel(R) UHD Graphics; NVIDIA GeForce RTX 4060 Laptop GPU`
- cwd: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project`

## 3. Protected-input policy

Protected inputs recorded: true.

## 4. Full cohort

Prompt-compatible rows: 5764; ranking rows: 2882; reranking rows: 2882.

## 5. Manifest

Manifest rows: 11528; chunks: 46; expected requests: 11528.

## 6. Execution progress

| progress_scope | chunk_id | planned_request_count | completed_requests | pending_requests | progress_status | next_pending_request_order |
| --- | --- | --- | --- | --- | --- | --- |
| chunk | CLFULL-CHUNK-001 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-002 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-003 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-004 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-005 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-006 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-007 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-008 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-009 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-010 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-011 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-012 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-013 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-014 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-015 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-016 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-017 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-018 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-019 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-020 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-021 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-022 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-023 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-024 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-025 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-026 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-027 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-028 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-029 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-030 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-031 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-032 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-033 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-034 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-035 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-036 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-037 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-038 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-039 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-040 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-041 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-042 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-043 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-044 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-045 | 256 | 256 | 0 | complete |  |
| chunk | CLFULL-CHUNK-046 | 8 | 8 | 0 | complete |  |
| global | GLOBAL | 11528 | 11528 | 0 | complete |  |

## 7. Raw retention

Requests: 11528; responses/errors: 11528; events: 23056.

## 8. Backend reliability

Backend error records: 0.

## 9. Strict format validity

| model_family | dataset | task_type | planned_rows | completed_rows | strict_valid_rows | strict_validity_rate |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | ranking | 990 | 990 | 990 | 1 |
| Qwen2 7B Instruct | Amazon Sports | ranking | 892 | 892 | 892 | 1 |
| Qwen2 7B Instruct | ML-1M | ranking | 1000 | 1000 | 1000 | 1 |
| Qwen2 7B Instruct | Amazon Beauty | reranking | 990 | 990 | 989 | 0.99898989898989898 |
| Qwen2 7B Instruct | Amazon Sports | reranking | 892 | 892 | 889 | 0.99663677130044848 |
| Qwen2 7B Instruct | ML-1M | reranking | 1000 | 1000 | 978 | 0.97799999999999998 |
| Mistral 7B Instruct | Amazon Beauty | ranking | 990 | 990 | 990 | 1 |
| Mistral 7B Instruct | Amazon Sports | ranking | 892 | 892 | 892 | 1 |
| Mistral 7B Instruct | ML-1M | ranking | 1000 | 1000 | 1000 | 1 |
| Mistral 7B Instruct | Amazon Beauty | reranking | 990 | 990 | 990 | 1 |
| Mistral 7B Instruct | Amazon Sports | reranking | 892 | 892 | 892 | 1 |
| Mistral 7B Instruct | ML-1M | reranking | 1000 | 1000 | 1000 | 1 |

## 10. Ranking diagnostics

| model_family | dataset | strict_valid_rows | valid_only_hr_at_5 | valid_only_ndcg_at_5 |
| --- | --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | 990 | 0.29292929292929293 | 0.18224538229211928 |
| Qwen2 7B Instruct | Amazon Sports | 892 | 0.29372197309417042 | 0.1774608432394085 |
| Qwen2 7B Instruct | ML-1M | 1000 | 0.29599999999999999 | 0.18031628535335303 |
| Mistral 7B Instruct | Amazon Beauty | 990 | 0.30909090909090908 | 0.19325117285693641 |
| Mistral 7B Instruct | Amazon Sports | 892 | 0.2858744394618834 | 0.16952494573506738 |
| Mistral 7B Instruct | ML-1M | 1000 | 0.42099999999999999 | 0.25123707935237044 |

## 11. Reranking diagnostics

| model_family | dataset | strict_valid_rows | valid_only_hr_at_5 | valid_only_ndcg_at_5 |
| --- | --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | 989 | 0.070778564206268962 | 0.043425466343970737 |
| Qwen2 7B Instruct | Amazon Sports | 889 | 0.047244094488188976 | 0.033250067784469393 |
| Qwen2 7B Instruct | ML-1M | 978 | 0.067484662576687116 | 0.039490242045712866 |
| Mistral 7B Instruct | Amazon Beauty | 990 | 0.062626262626262627 | 0.03839165848939511 |
| Mistral 7B Instruct | Amazon Sports | 892 | 0.053811659192825115 | 0.030860844664448068 |
| Mistral 7B Instruct | ML-1M | 1000 | 0.104 | 0.072402906265091507 |

## 12. Utility scope

HR@5 and NDCG@5 are valid-only execution diagnostics in this phase, not final scientific performance.

## 13. Validation

| validation_scope | check_name | expected_value | observed_value | status |
| --- | --- | --- | --- | --- |
| manifest | manifest_rows | 11528 | 11528 | pass |
| chunks | chunk_rows | 46 | 46 | pass |
| raw | raw_request_records | 11528 | 11528 | pass |
| raw | raw_response_records | 11528 | 11528 | pass |
| raw | runner_events | 23056 | 23056 | pass |
| raw | duplicate_request_ids | 0 | 0 | pass |
| raw | duplicate_response_ids | 0 | 0 | pass |
| state | request_state_rows | 11528 | 11528 | pass |
| state | completed_requests | 11528 | 11528 | pass |
| state | ambiguous_requests | 0 | 0 | pass |
| state | duplicate_evidence_requests | 0 | 0 | pass |
| execution | execution_rows | 11528 | 11528 | pass |
| parse | parse_rows | 11528 | 11528 | pass |
| rankings | ranking_rows | 11528 | 11528 | pass |
| metrics | cell_metric_rows | 12 | 12 | pass |
| execution | retry_count | 0 | 0 | pass |
| execution | request_order_gaps | 0 | 0 | pass |
| protected | protected_inputs_unchanged | true | true | pass |
| utility | valid_only_not_final | false | false | pass |

## 14. Decision

Decision: `full_matrix_execution_complete_ready_for_evaluation`. Blocking reason: ``.

## 15. Commands

- `$env:PYTHONNOUSERSITE='1'; & 'C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe' src/experiments/run_canonical_local_llm_full_matrix.py --max-new-requests 256`
- `$env:PYTHONNOUSERSITE='1'; & 'C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe' src/experiments/run_canonical_local_llm_full_matrix.py --report-only`

## 16. Next task

Phase 5C - canonical evaluation of full local LLM outputs, beyond-utility computation, baseline comparison, and paper-result alignment analysis
