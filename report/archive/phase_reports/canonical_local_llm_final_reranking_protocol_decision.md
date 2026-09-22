# Final Canonical Local LLM Reranking Protocol Decision

## 1. Scope and provenance

Offline Phase 5B-P1F audit on branch `plan-d-extensions`, commit `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`. Protected evidence files hashed: `62`.

## 2. Executive summary

Decision: `approve_paper_aligned_topk_reranking_confirmation_pilot`. The failed full20 holdout is preserved, and no full execution is approved.

## 3. Safety boundaries

This audit invokes no Ollama endpoint, creates no request/response evidence, and does not create prompt v3.

## 4. Phase 5A reranking contract

| prompt_contract_id | origin_classification | scientific_necessity | status |
| --- | --- | --- | --- |
| llm_rerank_index_json_full20_v1 | adapted_full_permutation | not_required_for_hr_ndcg_or_top5_beyond_utility | diagnostic_only_after_failed_holdout |
| llm_rerank_index_json_full20_v2 | adapted_full_permutation | not_required_for_hr_ndcg_or_top5_beyond_utility | diagnostic_only_after_failed_holdout |

## 5. Prompt-v1 pilot evidence

Prompt v1 selected Qwen2 full20 reranking only as an adapted local format contract.

## 6. Prompt-v2 corrective-pilot evidence

Prompt v2 selected Mistral full20 reranking only as an adapted local format contract.

## 7. Hybrid holdout evidence

| model_family | dataset | holdout_strict_valid_rows | holdout_rows | holdout_gate_passed |
| --- | --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | 8 | 10 | true |
| Qwen2 7B Instruct | Amazon Sports | 10 | 10 | true |
| Qwen2 7B Instruct | ML-1M | 10 | 10 | true |
| Mistral 7B Instruct | Amazon Beauty | 9 | 10 | true |
| Mistral 7B Instruct | Amazon Sports | 7 | 10 | false |
| Mistral 7B Instruct | ML-1M | 9 | 10 | true |

## 8. Preservation of the failed holdout gate

Mistral Sports remains failed at 7/10. Combined pilot+holdout evidence is descriptive and never overrides the gate.

## 9. Paper reranking task evidence

The local paper PDF is hashed but not locally text-extracted, so no paper-exact output-count claim is made.

## 10. Official repository reranking evidence

| source_path | requested_output_count | requires_complete_permutation | supports_top10 |
| --- | --- | --- | --- |
| original_repo/EvaLLMasRecommender/eva/templates/template.txt | 10 | false | true |
| original_repo/EvaLLMasRecommender/eva/templates/llmrank.txt | ambiguous_rank_20 | ambiguous | false |
| original_repo/EvaLLMasRecommender/eva/codes/llms.py | 10 | false | true |
| original_repo/EvaLLMasRecommender/eva/codes/output_refine.py | 10 | false | true |
| original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py | 10 | false | true |

## 11. Candidate-set size

The frozen canonical candidate sets use 20 candidates per row for Plan E2 LLM-compatible ranking and reranking.

## 12. Requested output count

Official executable code supports 10 returned recommendations; local full20 contracts require 20 but are adapted.

## 13. Top-K versus full-permutation semantics

Top-K semantics preserve the recommendation task because HR@5/NDCG@5 and beyond-utility@5 consume the first five positions.

## 14. Parser assumptions

Official parsing accepts title lists and checks length 10; local Phase 5A parsing accepts strict JSON one-based indices.

## 15. Evaluation cutoff

The official and canonical metric cutoff is 5 for HR/NDCG and top-5 beyond-utility calculations.

## 16. Metric input requirements

| metric_id | evaluation_cutoff | requires_top5 | requires_top10 | requires_full20 |
| --- | --- | --- | --- | --- |
| hr_at_5 | 5 | true | false | false |
| ndcg_at_5 | 5 | true | false | false |
| novelty_at_5 | 5 | true | false | false |
| catalog_coverage_at_5 | 5 | true | false | false |
| item_coverage_at_5 | 5 | true | false | false |
| long_tail_rate_at_5 | 5 | true | false | false |
| average_popularity_at_5 | 5 | true | false | false |
| popularity_bias_at_5 | 5 | true | false | false |
| intra_list_diversity_at_5 | 5 | true | false | false |
| duplicate_rate_at_5 | 5 | true | false | false |
| candidate_validity_rate | 5 | true | false | false |
| hallucination_rate_at_5 | 5 | true | false | false |
| strict_parse_validity_rate | contract | false | true | false |
| operational_success_rate | contract | false | true | false |
| position_bias_top5_diagnostic | 5 | true | false | false |
| position_bias_full20_diagnostic | 20 | false | false | true |

## 17. Whether HR@5 requires full permutation

HR@5 requires only the positive item's presence in the first five returned recommendations.

## 18. Whether NDCG@5 requires full permutation

NDCG@5 requires only the positive item's one-based rank when that rank is at most 5.

## 19. Whether beyond-utility metrics require full permutation

Beyond-utility metrics frozen at @5 use top-5 exposures; only full-rank position diagnostics require full20.

## 20. Origin of the local full-20 contract

The full20 requirement originated in local reconstruction, parser convenience, and diagnostic engineering.

## 21. Full-20 paper-alignment classification

Full20 classification: `adapted_full_permutation`; paper alignment: `adapted_not_paper_exact`.

## 22. Full-permutation reliability findings

Full20 reliability is insufficient for primary execution because one disjoint holdout cell failed the 0.80 threshold.

## 23. Prompt-tuning history

Prompt tuning history includes v1, v2, hybrid selection, and disjoint holdout; more wording tuning would be post-hoc.

## 24. No-more-wording-tuning policy

No further wording-only prompt revision may be approved using existing pilot or holdout outcomes.

## 25. Prompt-v3 rejection or necessity

Prompt v3 is rejected in this phase; a new contract must be justified by paper/official code, not pass rates.

## 26. Holdout statistical interpretation

| model_family | dataset | combined_validity_rate_descriptive | wilson_interval_low | wilson_interval_high |
| --- | --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | 0.9 | 0.490162471537 | 0.943317848546 |
| Qwen2 7B Instruct | Amazon Sports | 0.95 | 0.722467200137 | 1 |
| Qwen2 7B Instruct | ML-1M | 0.95 | 0.722467200137 | 1 |
| Mistral 7B Instruct | Amazon Beauty | 0.95 | 0.595849973205 | 0.982123786905 |
| Mistral 7B Instruct | Amazon Sports | 0.8 | 0.396778147461 | 0.892208732594 |
| Mistral 7B Instruct | ML-1M | 0.9 | 0.595849973205 | 0.982123786905 |

## 27. Combined evidence limitations

Combined validity counts estimate format reliability only and are not a readiness decision.

## 28. Invalid-output selection bias

| reporting_policy | selection_bias_risk | allowed_for_primary_results | allowed_for_diagnostic_results |
| --- | --- | --- | --- |
| strict-valid-only utility | high | false | true |
| invalid-as-zero utility | medium | false | false |
| separate validity plus valid-only utility | low | true | true |
| ranking-only primary results | low | true | true |
| full-permutation diagnostic results | low | true | true |

## 29. Valid-only utility limitations

Valid-only utility has model- and dataset-dependent denominators, so it is diagnostic only.

## 30. Ranking-only protocol option

Ranking-only remains a fallback but is not the selected path because official-code top10 evidence supports a reranking confirmation pilot.

## 31. Top-5 reranking option

Strict top-5 is metric-sufficient but weaker than top10 against official-code evidence.

## 32. Top-10 reranking option

Strict top-10 is the selected top-K reranking contract to validate next.

## 33. Full-20 diagnostic option

Full20 can only be diagnostic after the failed holdout and is not selected for full execution.

## 34. Structured-decoding option

Structured decoding is not selected because local support is not documented in the audited evidence.

## 35. Protocol comparison

| protocol_option_id | requested_output_count | execution_classification | recommendation_status |
| --- | --- | --- | --- |
| option_a_full20_strict_permutation | 20 | blocked_for_primary | not_recommended |
| option_b_strict_top10_confirmation | 10 | confirmation_pilot_required | recommended |
| option_c_strict_top5_output | 5 | not_selected | not_recommended |
| option_d_ranking_only_reranking_blocked | 5 | available_fallback_not_selected | not_recommended |
| option_e_full20_diagnostic_only | 20 | diagnostic_only_not_selected | not_recommended |
| option_f_structured_decoding_adaptation | contract_dependent | offline_audit_required_if_pursued | not_recommended |

## 36. Selected final protocol

Selected protocol: `option_b_strict_top10_confirmation`.

## 37. Paper-alignment label

Paper-alignment label: `official_code_aligned_top10_paper_pdf_text_unverified`.

## 38. Primary versus diagnostic execution

Primary execution is limited to a future top10 reranking confirmation pilot; no full matrix is approved.

## 39. Full execution scope

| model_family | dataset | task_type | request_count | allowed_to_execute |
| --- | --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | ranking | 0 | false |
| Qwen2 7B Instruct | Amazon Beauty | reranking | 10 | true |
| Qwen2 7B Instruct | Amazon Sports | ranking | 0 | false |
| Qwen2 7B Instruct | Amazon Sports | reranking | 10 | true |
| Qwen2 7B Instruct | ML-1M | ranking | 0 | false |
| Qwen2 7B Instruct | ML-1M | reranking | 10 | true |
| Mistral 7B Instruct | Amazon Beauty | ranking | 0 | false |
| Mistral 7B Instruct | Amazon Beauty | reranking | 10 | true |
| Mistral 7B Instruct | Amazon Sports | ranking | 0 | false |
| Mistral 7B Instruct | Amazon Sports | reranking | 10 | true |
| Mistral 7B Instruct | ML-1M | ranking | 0 | false |
| Mistral 7B Instruct | ML-1M | reranking | 10 | true |

## 40. Request-count implications

Request-count implications: ranking-only full `5814`, ranking+reranking full `11628`, top-K confirmation `60`, full20 diagnostic reranking `5814`.

## 41. Blocked components

Blocked components: full20 primary reranking and full canonical matrix execution.

## 42. Required confirmation phase

Required confirmation phase is a new disjoint top10 reranking confirmation pilot.

## 43. Reporting requirements

Future reports must separate parse validity, operational success, and conditional utility.

## 44. Limitations

Limitations: paper PDF text remains unverified locally, and the top10 strict JSON contract has not yet been piloted.

## 45. Protected-input immutability

Protected hashes were computed before and after generation; the runner fails if evidence hashes change.

## 46. Commands executed

- `git status --short --branch`
- `git branch --show-current`
- `git log -1 --format="%H %s"`
- `$env:PYTHONNOUSERSITE='1'; & '.venv\Scripts\python.exe' src\experiments\build_canonical_local_llm_final_reranking_protocol_audit.py`

## 47. Final audit decision

| recommended_decision | selected_protocol_option_id | confirmation_pilot_required | utility_used_for_decision |
| --- | --- | --- | --- |
| approve_paper_aligned_topk_reranking_confirmation_pilot | option_b_strict_top10_confirmation | true | false |

## 48. Recommended single next task

Recommended single next task: `Phase 5B-P1G - execute a frozen paper-aligned top-K reranking confirmation pilot on a new disjoint sample`.
