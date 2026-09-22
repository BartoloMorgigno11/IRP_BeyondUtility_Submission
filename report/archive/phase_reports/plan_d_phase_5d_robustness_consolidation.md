# Plan D Phase 5D Robustness Consolidation

## 1. Objective And Scope

Phase 5D consolidates existing local LLM robustness and diagnostic evidence only. It does not start Phase 5E.

## 2. Offline-Only Methodology

Inputs are existing CSV, JSONL, report, source, prompt, parser, and manifest artifacts. No LLM, API, Ollama, recommender, download, install, staging, or commit operation is part of this builder.

## 3. Artifact Discovery Methodology

Discovered included artifacts: 372; protected inputs marked from the initial manifest: 395.

## 4. Protected-Input Policy

Scientifically relevant text, CSV, JSON, JSONL, PDF, prompt, parser, raw LLM, report, and archive artifacts are SHA-256 hashed; large checkpoints use metadata-only records. Phase 5D outputs and caches are excluded from protected inputs.

## 5. Evidence Hierarchy

| evidence_level | count |
| --- | --- |
| phase_5c_full_matrix_evidence | 5 |
| exact_paired_diagnostic_experiment | 4 |
| repeated_controlled_diagnostic | 0 |
| small_unpaired_pilot | 0 |
| report_only_summary | 0 |
| blocked_or_unresolved_artifact | 3 |

## 6. Canonical Versus Legacy Protocol Map

| protocol_label | candidate_count | output_count | permutation_requirement | canonical_status | comparability_status |
| --- | --- | --- | --- | --- | --- |
| strict ranking top-5 | 20 | 5 | false | canonical_ranking_top5 | canonical_for_ranking_only_not_interchangeable_with_reranking |
| canonical reranking top-10 | 20 | 10 | false | canonical_top10 | canonical_for_reranking_top10_not_semantically_equivalent_to_full20 |
| legacy full-20 v1 | 20 | 20 | true | legacy_full20_v1 | legacy_diagnostic_not_canonical_top10_evidence |
| legacy full-20 v2 | 20 | 20 | true | legacy_full20_v2 | paired_only_with_v1_where_rows_match; not comparable to top10 as performance ablation |
| strict prompt candidate-ID diagnostic | 20 | 5 | false | strict_prompt_diagnostic | diagnostic_prompt_output_space_change_limits_attribution |
| profile generation audit-only |  | free_text_profile | not_applicable | blocked_protocol | no_controlled_profile_effect_claim_allowed |

## 7. Phase 5C Full-Matrix Evidence

| model | dataset | task | planned_count | strict_valid_count | invalid_count | duplicate_index_count | validity_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | ranking | 990 | 990 | 0 | 0 | 1 |
| Qwen2 7B Instruct | Amazon Beauty | reranking | 990 | 989 | 1 | 1 | 0.99898989898989898 |
| Qwen2 7B Instruct | Amazon Sports | ranking | 892 | 892 | 0 | 0 | 1 |
| Qwen2 7B Instruct | Amazon Sports | reranking | 892 | 889 | 3 | 3 | 0.99663677130044848 |
| Qwen2 7B Instruct | ML-1M | ranking | 1000 | 1000 | 0 | 0 | 1 |
| Qwen2 7B Instruct | ML-1M | reranking | 1000 | 978 | 22 | 22 | 0.97799999999999998 |
| Mistral 7B Instruct | Amazon Beauty | ranking | 990 | 990 | 0 | 0 | 1 |
| Mistral 7B Instruct | Amazon Beauty | reranking | 990 | 990 | 0 | 0 | 1 |
| Mistral 7B Instruct | Amazon Sports | ranking | 892 | 892 | 0 | 0 | 1 |
| Mistral 7B Instruct | Amazon Sports | reranking | 892 | 892 | 0 | 0 | 1 |
| Mistral 7B Instruct | ML-1M | ranking | 1000 | 1000 | 0 | 0 | 1 |
| Mistral 7B Instruct | ML-1M | reranking | 1000 | 1000 | 0 | 0 | 1 |

## 8. Nondeterminism Evidence

Strongest evidence: `blocked_or_unresolved_artifact`. Scope: not_executed. Paired: false; repeated: false. Finding: No repeated controlled LLM run artifact was found. Limitations: Repeated run identifiers and raw outputs are missing.

## 9. Candidate-Position-Bias Evidence

Strongest evidence: `exact_paired_diagnostic_experiment`. Scope: 20 source rows x 3 variants. Paired: true; repeated: false. Finding: Controlled candidate-reordering evidence exists but is not full-cohort evidence. Limitations: Original order was not rerun in D5F.

## 10. History-Length Evidence

Strongest evidence: `exact_paired_diagnostic_experiment`. Scope: 20 users per model. Paired: true; repeated: false. Finding: History-length comparisons are paired within the ML-1M diagnostic subset. Limitations: History variants are diagnostic prompt conditions.

## 11. Strict-Prompt Evidence

Strongest evidence: `exact_paired_diagnostic_experiment`. Scope: 20 source rows plus post-filter rows. Paired: true; repeated: false. Finding: Strict prompt evidence is useful for validity diagnosis but not a wording-only causal claim. Limitations: Post-filter rows are deterministic analyses, not new LLM outputs.

## 12. Full-20 V1 Versus V2 Evidence

Strongest evidence: `exact_paired_diagnostic_experiment`. Scope: paired pilot rows where available. Paired: true; repeated: false. Finding: V1/V2 comparisons belong to the legacy full-20 protocol family. Limitations: Output cardinality and full-permutation requirement differ from canonical top-10.

## 13. Full-20 Versus Top-10 Interpretation

Strongest evidence: `blocked_or_unresolved_artifact`. Scope: no matched full-20/top-10 cohort. Paired: false; repeated: false. Finding: Full-20 and top-10 differ in output cardinality, parser contract, full-permutation requirement, and canonical status. Limitations: No aligned paired experiment exists.

## 14. Qwen2 Versus Mistral Comparison

Strongest evidence: `phase_5c_full_matrix_evidence`. Scope: full Phase 5C matrix. Paired: false; repeated: false. Finding: Mistral is structurally strict-valid in all Phase 5C cells; Qwen2 has reranking duplicate-index failures. Limitations: Local substitute models only.

## 15. Ranking Versus Reranking Comparison

Strongest evidence: `phase_5c_full_matrix_evidence`. Scope: full Phase 5C matrix. Paired: false; repeated: false. Finding: Ranking and reranking must remain separate task semantics. Limitations: Ranking asks for 5 outputs; reranking asks for 10.

## 16. Generated-Profile Evidence

Strongest evidence: `blocked_or_unresolved_artifact`. Scope: not_executed. Paired: false; repeated: false. Finding: Profile effects cannot be claimed. Limitations: Profile model, reuse, parser, and leakage policy are not frozen.

## 17. Hallucination And Validity Evidence

Strongest evidence: `phase_5c_full_matrix_evidence`. Scope: full Phase 5C matrix. Paired: false; repeated: false. Finding: Canonical strict-index Phase 5C has zero hallucination rows under the frozen definition. Limitations: Earlier openp5 pilots used different prompts/parsers. Invalid rows with nonzero counts: | model | dataset | task | invalid_count | duplicate_index_count | hallucination_count |
| --- | --- | --- | --- | --- | --- |
| Qwen2 7B Instruct | Amazon Beauty | reranking | 1 | 1 | 0 |
| Qwen2 7B Instruct | Amazon Sports | reranking | 3 | 3 | 0 |
| Qwen2 7B Instruct | ML-1M | reranking | 22 | 22 | 0 |

## 18. Blocked, Incomplete, Or Non-Generalizable Diagnostics

| topic | blocker_type | status | recoverable_without_new_inference | prohibited_statement |
| --- | --- | --- | --- | --- |
| nondeterminism | not_executed | blocked_or_unresolved_artifact | false | Do not claim nondeterminism or stability effects from single-run evidence. |
| generated_profiles | ambiguous_contract | blocked_or_unresolved_artifact | false | Do not claim profile benefit or harm. |
| full20_vs_top10_numeric_ablation | unpaired_protocols | blocked_or_unresolved_artifact | false | Do not report a direct full-20 versus top-10 performance effect. |
| LastFM | incomplete_extension | blocked_or_unresolved_artifact | false | Do not include LastFM in complete canonical Phase 5C conclusions. |
| paper_exact_remote_models | external_api_or_hardware | blocked_or_unresolved_artifact | false | Do not call them exact paper-model reproductions. |

## 19. Permitted Scientific Conclusions

| topic | claim_id | permitted_interpretation | limitations |
| --- | --- | --- | --- |
| output_validity | CLAIM-001 | Use as full-matrix local evidence. | Local substitute models and adapted JSON-index output. |
| ranking_vs_reranking | CLAIM-002 | Report task-specific structural reliability separately. | Ranking asks for 5 outputs; reranking asks for 10. |
| duplicate_indices | CLAIM-003 | Report duplicate-index failures as structural invalidity. | Invalid rows are retained. |
| hallucination | CLAIM-004 | Use the repository's frozen hallucination definition. | Earlier openp5 pilots used different prompts/parsers. |
| candidate_position_bias | CLAIM-005 | Discuss as controlled diagnostic evidence. | Original order was not rerun in D5F. |
| history_length | CLAIM-006 | Report sensitivity only within the diagnostic design. | History variants are diagnostic prompt conditions. |
| strict_prompt | CLAIM-007 | Separate prompt, parser, and output-space changes. | Post-filter rows are deterministic analyses, not new LLM outputs. |
| full20_v1_v2 | CLAIM-008 | Use for legacy protocol diagnosis. | Output cardinality and full-permutation requirement differ from canonical top-10. |
| qwen2_vs_mistral | CLAIM-010 | Keep reliability, utility, and beyond-utility separate. | Local substitute models only. |

## 20. Prohibited Overclaims

| topic | claim_id | prohibited_interpretation |
| --- | --- | --- |
| output_validity | CLAIM-001 | Do not treat as paper-model exact reproduction. |
| ranking_vs_reranking | CLAIM-002 | Do not collapse task semantics into one interchangeable number. |
| duplicate_indices | CLAIM-003 | Do not count them as hallucinations or silently repair them. |
| hallucination | CLAIM-004 | Do not merge hallucination with parser-invalid duplicate outputs. |
| candidate_position_bias | CLAIM-005 | Do not generalize to full cohorts or all models. |
| history_length | CLAIM-006 | Do not present as full-matrix evidence. |
| strict_prompt | CLAIM-007 | Do not attribute differences solely to wording. |
| full20_v1_v2 | CLAIM-008 | Do not use as canonical top-10 execution evidence. |
| full20_vs_top10 | CLAIM-009 | Do not report numerical full-20 versus top-10 effects. |
| qwen2_vs_mistral | CLAIM-010 | Do not collapse tasks/datasets into an unjustified single score. |
| generated_profiles | CLAIM-011 | Do not claim profile benefit or harm. |
| nondeterminism | CLAIM-012 | Do not infer nondeterminism from a single run. |

## 21. Phase 5D Decision

Validation pass/fail: 23/0. Decision: `canonical_existing_robustness_evidence_consolidation_complete`.

## 22. Handoff Requirements For Phase 5E

| action_id | input_artifact | required_use | status |
| --- | --- | --- | --- |
| NEXT-5D-001 | results/tables/plan_d_phase_5d_evidence_claims.csv | Consume permitted/prohibited claims before writing any synthesis or extending experiments. | ready |
| NEXT-5D-002 | results/tables/plan_d_phase_5d_protocol_comparison.csv | Keep canonical top-10 reranking separate from legacy full-20 diagnostics. | ready |
| NEXT-5D-003 | results/tables/plan_d_phase_5d_blocked_or_unresolved.csv | Preserve blocked nondeterminism/profile/full20-vs-top10 limitations. | ready |

## Appendix. Pairing Coverage

| comparison_identifier | matched_pair_count | paired_analysis_permitted | reason |
| --- | --- | --- | --- |
| position_bias_original_vs_positive_first | 20 | true | exact key match |
| position_bias_original_vs_positive_last | 20 | true | exact key match |
| position_bias_original_vs_reverse_order | 20 | true | exact key match |
| strict_prompt_original_vs_original_post_filtered | 20 | true | exact key match |
| strict_prompt_original_vs_strict_candidate_id_prompt | 20 | true | exact key match |
| strict_prompt_original_vs_strict_post_filtered | 20 | true | exact key match |
| qwen_vs_mistral_strict_prompt | 20 | true | exact key match |
| history_length_mistral:7b-instruct_short_history_vs_medium_history | 20 | true | exact key match |
| history_length_mistral:7b-instruct_long_history_vs_medium_history | 20 | true | exact key match |
| history_length_mistral:7b-instruct_short_history_vs_long_history | 20 | true | exact key match |
| history_length_qwen2:7b-instruct_short_history_vs_medium_history | 20 | true | exact key match |
| history_length_qwen2:7b-instruct_long_history_vs_medium_history | 20 | true | exact key match |
| history_length_qwen2:7b-instruct_short_history_vs_long_history | 20 | true | exact key match |
| reranking_candidate_order_reference_vs_qwen2_rerank_prompt | 20 | true | exact key match |
| reranking_candidate_order_reference_vs_mistral_rerank_prompt | 20 | true | exact key match |
| reranking_qwen2_rerank_prompt_vs_mistral_rerank_prompt | 20 | true | exact key match |
| legacy_full20_v1_vs_v2 | 60 | true | precomputed v1-v2 comparison rows use stable keys |
| full20_vs_top10_direct_numeric_ablation | 0 | false | top10 confirmation sample is disjoint from previous pilots; overlap flags sum to 0 |

## Appendix. Experiment Scope Counts

| scope_classification | count |
| --- | --- |
| blocked | 2 |
| diagnostic_subset | 6 |
| full_matrix | 12 |
| pilot | 4 |
