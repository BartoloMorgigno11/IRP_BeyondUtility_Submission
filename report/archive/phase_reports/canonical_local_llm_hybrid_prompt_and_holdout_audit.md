# Canonical Local LLM Hybrid Prompt and Disjoint Holdout Audit

## 1. Scope and provenance

Branch and commit were recorded outside this report. The audit hashes 44 protected evidence files and writes only the Phase 5B-P1E audit artifacts.

## 2. Executive summary

Decision: `approve_hybrid_disjoint_holdout_pilot`. The hybrid policy is approved for a disjoint holdout pilot, not for full execution.

## 3. Safety boundaries

This audit invokes no Ollama endpoint, sends no LLM request, creates no prompt v3, and executes no recommender training.

## 4. Prompt-v1 pilot evidence

Amazon Beauty: 10/10; Amazon Sports: 9/10; ML-1M: 9/10; Amazon Beauty: 6/10; Amazon Sports: 9/10; ML-1M: 10/10

## 5. Prompt-v2 corrective-pilot evidence

Amazon Beauty: 4/10; Amazon Sports: 5/10; ML-1M: 4/10; Amazon Beauty: 10/10; Amazon Sports: 9/10; ML-1M: 9/10

## 6. V1/V2 evidence reconciliation

| model_family | dataset | prompt_version | strict_valid_count | request_count | all_sources_agree |
| --- | --- | --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | v1 | 10 | 10 | true |
| Qwen2 7B Instruct | Amazon Sports | v1 | 9 | 10 | true |
| Qwen2 7B Instruct | ML-1M | v1 | 9 | 10 | true |
| Mistral 7B Instruct | Amazon Beauty | v1 | 6 | 10 | true |
| Mistral 7B Instruct | Amazon Sports | v1 | 9 | 10 | true |
| Mistral 7B Instruct | ML-1M | v1 | 10 | 10 | true |
| Qwen2 7B Instruct | Amazon Beauty | v2 | 4 | 10 | true |
| Qwen2 7B Instruct | Amazon Sports | v2 | 5 | 10 | true |
| Qwen2 7B Instruct | ML-1M | v2 | 4 | 10 | true |
| Mistral 7B Instruct | Amazon Beauty | v2 | 10 | 10 | true |
| Mistral 7B Instruct | Amazon Sports | v2 | 9 | 10 | true |
| Mistral 7B Instruct | ML-1M | v2 | 9 | 10 | true |

## 7. Qwen2 prompt-v1 findings

Qwen2 prompt v1 passes all reranking cells: Beauty 10/10, Sports 9/10, ML-1M 9/10.

## 8. Qwen2 prompt-v2 findings

Qwen2 prompt v2 fails all reranking cells: Beauty 4/10, Sports 5/10, ML-1M 4/10.

## 9. Mistral prompt-v1 findings

Mistral prompt v1 fails Beauty at 6/10 but passes Sports 9/10 and ML-1M 10/10.

## 10. Mistral prompt-v2 findings

Mistral prompt v2 passes all reranking cells: Beauty 10/10, Sports 9/10, ML-1M 9/10.

## 11. Structural-validity selection basis

| model_family | selected_prompt_contract_id | rejected_prompt_contract_id | selected_prompt_structural_validity_minimum | selection_basis_classification |
| --- | --- | --- | --- | --- |
| Qwen2 7B Instruct | llm_rerank_index_json_full20_v1 | llm_rerank_index_json_full20_v2 | 0.90000000000000002 | structural_validity_based |
| Mistral 7B Instruct | llm_rerank_index_json_full20_v2 | llm_rerank_index_json_full20_v1 | 0.90000000000000002 | structural_validity_based |

## 12. Confirmation that utility was not used

Selection flags for HR@5, NDCG@5, positive rank, latency, token count, and beyond-utility metrics are all false.

## 13. Prompt semantic-equivalence audit

| field_name | values_identical | difference_classification | hybrid_policy_compatible |
| --- | --- | --- | --- |
| task_type | true | identical_non_difference | true |
| candidate_count | true | identical_non_difference | true |
| candidate_representation | true | identical_non_difference | true |
| candidate_numbering | true | identical_non_difference | true |
| candidate_order_policy | true | identical_non_difference | true |
| paired_candidate_order_hashes | true | identical_non_difference | true |
| requested_output_count | true | identical_non_difference | true |
| identifier_namespace | true | identical_non_difference | true |
| full_permutation_semantics | false | format_compliance_reinforcement | true |
| json_key | true | identical_non_difference | true |
| json_container | false | format_compliance_reinforcement | true |
| explanation_policy | true | identical_non_difference | true |
| code_fence_policy | true | identical_non_difference | true |
| prose_and_extra_key_policy | false | format_compliance_reinforcement | true |
| internal_validation_instruction | false | format_compliance_reinforcement | true |
| positive_label_concealment | true | identical_non_difference | true |
| history_source_policy | false | format_compliance_reinforcement | true |
| paired_history_render_hashes | true | identical_non_difference | true |
| history_item_representation | false | format_compliance_reinforcement | true |
| parser_contract | true | identical_non_difference | true |
| parser_requires_full_permutation | true | identical_non_difference | true |
| execution_parameters | true | identical_non_difference | true |
| temperature | true | identical_non_difference | true |
| seed | true | identical_non_difference | true |
| response_format | true | identical_non_difference | true |
| retry_policy | true | identical_non_difference | true |
| scientific_semantics_contract_flag | true | identical_non_difference | true |

## 14. Candidate-information equivalence

Candidate representation, numbering, candidate order policy, and paired candidate-order hashes are equivalent.

## 15. History-information equivalence

History source is inherited from v1 contexts, and all paired history-render hashes are identical.

## 16. Output-space equivalence

Both prompts request the same 20-item full permutation in one-based candidate-index space.

## 17. Parser equivalence

Both reranking versions use `llm_rerank_strict_primary_parser_v1` with no salvage substitution.

## 18. Execution-parameter equivalence

Execution parameters remain temperature 0, max_output_tokens 512, json_object response format, no retries, and concurrency 1.

## 19. Model-specific wording implications

Model-specific wording reduces literal prompt identity but keeps the requested task and valid output space identical.

## 20. Cross-model comparability

Cross-model utility comparisons remain adapted and must carry the model-specific prompt caveat.

## 21. Paper-alignment classification

Paper alignment is `adapted_model_specific_format_compliance_contract`; the policy is not paper exact.

## 22. Prompt-v3 necessity assessment

A common prompt v3 is not scientifically necessary before holdout because semantic equivalence holds and prompt-v3 search would add tuning risk.

## 23. Post-hoc prompt-tuning risk

Post-hoc tuning risk is present because pilot structural validity informed prompt selection; the mitigation is a frozen disjoint holdout.

## 24. Pilot-row overfitting risk

Pilot-row overfitting risk is mitigated by excluding every v1 and v2 pilot row before selecting holdout rows.

## 25. Independent holdout rationale

Independent holdout validation is required before any full execution because it tests structural readiness on unseen rows.

## 26. Holdout population

The holdout population is `primary_prompt_compatible == true` reranking rows from Beauty, Sports, and ML-1M.

## 27. Previous-pilot exclusions

Previous pilot exclusions use dataset, task type, candidate-set ID, and user ID identities from v1 sample and v2 corrective manifest.

## 28. Deterministic holdout selection

Selection key is SHA256(dataset + '|' + task_type + '|' + candidate_set_id + '|' + user_id), sorted ascending with candidate-set ID and user ID tie-breakers.

## 29. Holdout sample composition

| dataset | rows |
| --- | --- |
| Amazon Beauty | 10 |
| Amazon Sports | 10 |
| ML-1M | 10 |

## 30. Hybrid prompt-policy contract

| hybrid_policy_id | qwen_reranking_prompt_contract_id | mistral_reranking_prompt_contract_id | contract_status |
| --- | --- | --- | --- |
| canonical_local_llm_hybrid_qwen_v1_mistral_v2_rerank_policy_v1 | llm_rerank_index_json_full20_v1 | llm_rerank_index_json_full20_v2 | approved_for_disjoint_holdout_pilot |

## 31. Qwen2 prompt assignment

Qwen2 ranking uses `llm_rank_index_json_top5_v1` and reranking uses `llm_rerank_index_json_full20_v1`.

## 32. Mistral prompt assignment

Mistral ranking uses `llm_rank_index_json_top5_v1` and reranking uses `llm_rerank_index_json_full20_v2`.

## 33. Shared parser and output requirements

Shared strict parser requirements are represented by `llm_rank_strict_primary_parser_v1;llm_rerank_strict_primary_parser_v1`.

## 34. Holdout execution parameters

Holdout execution parameters are frozen in the request manifest and remain no-retry, sequential, temperature-zero settings.

## 35. Holdout request count

The holdout manifest contains 60 requests over 30 unique rows.

## 36. Holdout readiness threshold

Each of the six model-dataset cells must pass at least 8/10 strict-valid responses, threshold 0.80.

## 37. Full-execution implications

Full execution remains blocked until the disjoint hybrid holdout pilot passes the readiness gate.

## 38. Limitations

| risk_id | risk_category | severity | status |
| --- | --- | --- | --- |
| HYB-RISK-001 | model_specific_wording | warning | mitigated_by_holdout_gate |
| HYB-RISK-002 | post_hoc_selection | warning | mitigated_by_disjoint_sample |
| HYB-RISK-003 | pilot_row_overfitting | warning | mitigated_by_holdout_design |
| HYB-RISK-004 | reduced_prompt_identity | warning | accepted_with_caveat |
| HYB-RISK-005 | paper_alignment | warning | accepted_with_caveat |
| HYB-RISK-006 | unseen_row_failure | warning | open_until_holdout_executed |

## 39. Protected-input immutability

Protected hashes were computed for 44 evidence files before table generation and validated again afterward by the runner.

## 40. Commands executed

- `git status --short --branch`
- `git branch --show-current`
- `git log -1 --format="%H %s"`
- `$env:PYTHONNOUSERSITE='1'; & '.venv\Scripts\python.exe' src\experiments\build_canonical_local_llm_hybrid_prompt_holdout_audit.py`

## 41. Audit decision

| recommended_decision | hybrid_policy_created | holdout_sample_rows | holdout_request_rows | common_prompt_v3_required |
| --- | --- | --- | --- | --- |
| approve_hybrid_disjoint_holdout_pilot | true | 30 | 60 | false |

## 42. Recommended single next task

Recommended single next task: `Phase 5B-P1E - execute the frozen 60-request disjoint hybrid holdout pilot using Qwen2 reranking prompt v1 and Mistral reranking prompt v2`.
