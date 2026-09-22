# Canonical Local LLM Pilot Invalid-Output and Prompt-Corrective Audit

## 1. Scope and provenance

Offline Phase 5B-P1C-PROMPT-AUDIT generated deterministically from immutable Phase 5B-P1C evidence. Environment: `executable=C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe;python=3.12.10;pandas=3.0.3;cwd=C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project`.

## 2. Executive summary

Authoritative strict-valid count is `113/120`; strict-invalid count is `7`. Decision: `approve_reranking_prompt_v2_corrective_pilot`.

## 3. Safety boundaries

No Ollama endpoint, LLM inference, package install, download, recommender execution, raw-evidence modification, or previous-artifact overwrite was performed.

## 4. Immutable Phase 5B-P1C evidence

Read-only raw evidence counts: requests `120`, responses `120`, events `240`.

## 5. Raw request and response reconciliation

Raw request IDs and response IDs are unique, complete, and non-ambiguous.

## 6. Global count reconciliation

| strict_valid_count_raw_reparse | strict_invalid_count_raw_reparse | all_sources_agree | resolution |
| --- | --- | --- | --- |
| 113 | 7 | true | all_generated_sources_agree |

## 7. Dataset count reconciliation

| dataset | strict_valid_count_raw_reparse | strict_invalid_count_raw_reparse | notes |
| --- | --- | --- | --- |
| Amazon Beauty | 36 | 4 |  |
| Amazon Sports | 38 | 2 | pasted_summary_transcription_error: pasted dataset count 39/40, raw evidence 38/40 |
| ML-1M | 39 | 1 |  |

## 8. Model count reconciliation

| model_family | strict_valid_count_raw_reparse | strict_invalid_count_raw_reparse | all_sources_agree |
| --- | --- | --- | --- |
| Qwen2 7B Instruct | 58 | 2 | true |
| Mistral 7B Instruct | 55 | 5 | true |

## 9. Model-dataset-task count reconciliation

| model_family | dataset | task_type | strict_valid_count_raw_reparse | strict_invalid_count_raw_reparse |
| --- | --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | ranking | 10 | 0 |
| Qwen2 7B Instruct | Amazon Beauty | reranking | 10 | 0 |
| Qwen2 7B Instruct | Amazon Sports | ranking | 10 | 0 |
| Qwen2 7B Instruct | Amazon Sports | reranking | 9 | 1 |
| Qwen2 7B Instruct | ML-1M | ranking | 10 | 0 |
| Qwen2 7B Instruct | ML-1M | reranking | 9 | 1 |
| Mistral 7B Instruct | Amazon Beauty | ranking | 10 | 0 |
| Mistral 7B Instruct | Amazon Beauty | reranking | 6 | 4 |
| Mistral 7B Instruct | Amazon Sports | ranking | 10 | 0 |
| Mistral 7B Instruct | Amazon Sports | reranking | 9 | 1 |
| Mistral 7B Instruct | ML-1M | ranking | 10 | 0 |
| Mistral 7B Instruct | ML-1M | reranking | 10 | 0 |

## 10. Reported 113-versus-114 inconsistency

pasted_summary_transcription_error: raw, parse, ranking, metrics and generated report agree on 113/120; Sports is 38/40, not pasted 39/40

## 11. Authoritative strict-valid count

`113` strict-valid and `7` strict-invalid, from raw responses reparsed with the frozen strict parser.

## 12. Strict-invalid response inventory

| request_id | model_family | dataset | task_type | strict_failure_reason | output_length | missing_indices | duplicated_indices |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLLP-REQ-0035 | Qwen2 7B Instruct | Amazon Sports | reranking | output_count_invalid | 19 | [8] | {} |
| CLLP-REQ-0056 | Qwen2 7B Instruct | ML-1M | reranking | output_count_invalid | 19 | [17] | {} |
| CLLP-REQ-0071 | Mistral 7B Instruct | Amazon Beauty | reranking | duplicate_indices | 20 | [8] | {"19":2} |
| CLLP-REQ-0073 | Mistral 7B Instruct | Amazon Beauty | reranking | output_count_invalid | 21 | [] | {} |
| CLLP-REQ-0074 | Mistral 7B Instruct | Amazon Beauty | reranking | output_count_invalid | 29 | [] | {"13":2,"14":2,"15":2,"16":2,"17":2,"18":2,"19":2,"2":2,"20":2} |
| CLLP-REQ-0077 | Mistral 7B Instruct | Amazon Beauty | reranking | output_count_invalid | 22 | [] | {"1":2,"20":2} |
| CLLP-REQ-0100 | Mistral 7B Instruct | Amazon Sports | reranking | output_count_invalid | 19 | [13,14,15] | {"3":2,"4":2} |

## 13. Output-count failure analysis

| request_id | output_length | missing_indices | duplicated_indices | out_of_range_indices | forensic_classification |
| --- | --- | --- | --- | --- | --- |
| CLLP-REQ-0035 | 19 | [8] | {} | [] | complete_json_partial_permutation |
| CLLP-REQ-0056 | 19 | [17] | {} | [] | complete_json_partial_permutation |
| CLLP-REQ-0073 | 21 | [] | {} | [50] | complete_json_extra_indices |
| CLLP-REQ-0074 | 29 | [] | {"13":2,"14":2,"15":2,"16":2,"17":2,"18":2,"19":2,"2":2,"20":2} | [] | complete_json_extra_indices |
| CLLP-REQ-0077 | 22 | [] | {"1":2,"20":2} | [] | complete_json_extra_indices |
| CLLP-REQ-0100 | 19 | [13,14,15] | {"3":2,"4":2} | [] | complete_json_partial_permutation |

## 14. Duplicate-index failure analysis

| request_id | output_length | missing_indices | duplicated_indices | forensic_classification |
| --- | --- | --- | --- | --- |
| CLLP-REQ-0071 | 20 | [8] | {"19":2} | complete_json_duplicate_index |

## 15. JSON and schema findings

All seven invalid responses were complete JSON objects with the required key; failures were count, duplicate, or range/permutation noncompliance.

## 16. Truncation assessment

| request_id | done_reason | generated_token_count | configured_max_output_tokens | raw_response_complete_json | truncation_evidence |
| --- | --- | --- | --- | --- | --- |
| CLLP-REQ-0035 | stop | 56 | 512 | true | none |
| CLLP-REQ-0056 | stop | 55 | 512 | true | none |
| CLLP-REQ-0071 | stop | 83 | 512 | true | none |
| CLLP-REQ-0073 | stop | 87 | 512 | true | none |
| CLLP-REQ-0074 | stop | 118 | 512 | true | none |
| CLLP-REQ-0077 | stop | 89 | 512 | true | none |
| CLLP-REQ-0100 | stop | 77 | 512 | true | none |

## 17. Backend and token-limit assessment

All invalid responses had backend success and `done_reason=stop`; generated-token counts were far below the configured 512-token reranking limit.

## 18. Existing reranking prompt requirements

| requirement_name | requirement_status | exact_prompt_evidence |
| --- | --- | --- |
| exactly_20_indices | explicit | Return all 20 unique candidate indices |
| integers_1_to_20 | implicit | Candidate labels are 1-based indices |
| every_integer_exactly_once | implicit | all 20 unique candidate indices |
| no_omitted_integer | absent |  |
| no_repeated_integer | implicit | unique candidate indices |
| complete_permutation | absent |  |
| only_required_json_object | implicit | Return all 20 unique candidate indices as JSON |
| no_prose | implicit | Return only valid JSON. Do not explain your reasoning. |
| no_code_fence | implicit | Return only valid JSON |
| exact_json_key | explicit | ranked_candidate_indices |

## 19. Missing or ambiguous prompt requirements

| requirement_name | requirement_status | recommended_wording |
| --- | --- | --- |
| integers_1_to_20 | implicit | Explicitly require a 20-item permutation whose set is exactly integers 1 through 20, with no omissions or repeats. |
| every_integer_exactly_once | implicit | Explicitly require a 20-item permutation whose set is exactly integers 1 through 20, with no omissions or repeats. |
| no_omitted_integer | absent | Explicitly require a 20-item permutation whose set is exactly integers 1 through 20, with no omissions or repeats. |
| complete_permutation | absent | Explicitly require a 20-item permutation whose set is exactly integers 1 through 20, with no omissions or repeats. |

## 20. Strict-parser reassessment

| parser_contract_id | responses_reparsed | stored_and_reparsed_agreement_count | stored_and_reparsed_disagreement_count | index_base_correct | expected_output_count_correct | json_key_correct | whitespace_behavior_contract_aligned | integer_type_behavior_contract_aligned | duplicate_behavior_contract_aligned | partial_output_behavior_contract_aligned | positive_label_used | parser_conclusion | parser_revision_recommended | blocking_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llm_rerank_strict_primary_parser_v1 | 120 | 120 | 0 | true | true | true | true | true | true | true | false | strict_parser_correct_no_revision | false |  |

## 21. Diagnostic salvage reassessment

| parser_contract_id | responses_reparsed | stored_and_reparsed_agreement_count | stored_and_reparsed_disagreement_count | index_base_correct | expected_output_count_correct | json_key_correct | whitespace_behavior_contract_aligned | integer_type_behavior_contract_aligned | duplicate_behavior_contract_aligned | partial_output_behavior_contract_aligned | positive_label_used | parser_conclusion | parser_revision_recommended | blocking_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llm_rerank_diagnostic_salvage_parser_v1 | 120 | 120 | 0 | true | true | true | true | true | true | true | false | strict_parser_correct_no_revision | false |  |

## 22. Whether parser relaxation is justified

Parser relaxation is not justified: stored and independently reparsed results agree, and invalid outputs violate the frozen full-permutation contract.

## 23. Prompt-noncompliance root cause

The demonstrated root cause is prompt noncompliance: complete JSON outputs omitted, repeated, or added indices while backend and parser checks behaved as specified.

## 24. Minimal reranking-prompt revision

A common v2 reranking prompt is justified to make the existing full-permutation semantics explicit without changing candidates, order, output namespace, model parameters, or parser.

## 25. Why ranking prompt remains unchanged

Ranking was `60/60` strict-valid; no ranking prompt revision is required.

## 26. Candidate-order and semantic invariance

The corrective manifest preserves source candidate order hashes, histories, models, parser contracts, execution parameters, and zero-retry policy.

## 27. Positive-label concealment

Prompt v2 and the corrective manifest do not add positive IDs, positions, labels, utility values, or expected responses.

## 28. Reranking prompt-v2 contract

| prompt_contract_id | supersedes_prompt_contract_id | prompt_version | scientific_semantics_changed | contract_status |
| --- | --- | --- | --- | --- |
| llm_rerank_index_json_full20_v2 | llm_rerank_index_json_full20_v1 | v2 | false | ready_for_corrective_pilot |

## 29. Corrective-pilot design

Design only: run the 60 frozen reranking request rows with the common v2 prompt, no ranking rows, no retries, concurrency one, separate output root.

## 30. Corrective-pilot request count

Corrective manifest rows: `60`.

## 31. Readiness threshold

The result-independent readiness threshold remains `0.8` for every model x dataset x reranking cell.

## 32. Original-v1 evidence preservation

The original v1 raw requests, responses, tables, and report remain the primary evidence for v1 and were not changed.

## 33. Limitations

This audit does not execute prompt v2, does not estimate utility changes, and does not prove the corrective prompt will pass.

## 34. Protected-input immutability

Protected input hashes were computed before and after writing outputs and matched.

## 35. Commands executed

- `git status --short --branch`
- `git branch --show-current`
- `git log -1 --format="%H %s"`
- `$env:PYTHONNOUSERSITE='1'; & '.venv\Scripts\python.exe' src\experiments\build_canonical_local_llm_pilot_invalid_output_audit.py`

## 36. Audit decision

`approve_reranking_prompt_v2_corrective_pilot`: Execute the separate reranking prompt-v2 corrective pilot; do not overwrite v1 evidence.

## 37. Recommended single next task

Phase 5B-P1D — execute the frozen 60-request reranking prompt-v2 corrective pilot for Qwen2 and Mistral
