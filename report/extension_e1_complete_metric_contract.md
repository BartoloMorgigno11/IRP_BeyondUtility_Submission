# E1 Complete Paper Metric Contract

POST-F6M EXTENSION DESIGN EVIDENCE

## Final Decision

EXTENSION_E1_COMPLETE_METRIC_CONTRACT_FROZEN

## Repository Precondition

- branch: master
- HEAD: 4ca3c9842939d545b68dd988ec80dc090462323d
- runtime: .\.venv_consolidation_candidate\Scripts\python.exe, Python 3.12.10
- Git: lightweight precondition observed clean before E1 edits, master...origin/master [ahead 4]
- original_repo: clean; original_repo/EvaLLMasRecommender reports main...origin/main
- infrastructure validation: not reopened by E1

## Complete Paper Metric Inventory

| family | metric_name | canonical_paper_name | formula_status | task_applicability | cutoff_k | denominator | provenance_classification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| utility | HR@5 | HR@K | paper_prose_specified_and_official_code_recovered | BOTH | 5 | sample user count with ground-truth target | PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_PAPER_EQUIVALENT |
| utility | NDCG@5 | NDCG@K | paper_formula_recovered_and_official_code_recovered | BOTH | 5 | sample user count with ground-truth target | PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_PAPER_EQUIVALENT |
| utility_diagnostic | MRR printed by official script | not a paper result metric in recovered paper tables | official_code_recovered_not_saved_in_final_result_vector | BOTH | 5 | sample user count | OFFICIAL_CODE_RECOVERED;PAPER_UNDERSPECIFIED;LOCAL_ADAPTED |
| novelty | SelfInformation@5 | SelfInformation@K | paper_formula_recovered; official_code_uses_information_dictionary | BOTH | 5 | sample users and K positions | PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_ADAPTED |
| novelty | Serendipity@5 | Serendipity@K | paper_formula_and_prose_differ; official_code_recovered; local_adapted | BOTH | 5 | sample users | PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_ADAPTED |
| popularity | ARP@5 | Average Recommendation Popularity@K | paper_formula_recovered_and_official_code_recovered | BOTH | 5 | sample users times K | PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_ADAPTED |
| popularity | APLT@5 | Average Percentage of Long Tail Items@K | paper_formula_recovered_and_official_code_recovered | BOTH | 5 | sample users and K positions | PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_ADAPTED |
| popularity | PopREO@5 | Popularity-based Ranking-based Equal Opportunity@K | paper_formula_recovered_and_official_code_recovered_but_policy_incomplete | BOTH | 5 | per-group relevant positives; zero-group policy not recovered | PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;NOT_IMPLEMENTED |
| diversity | ItemCoverage@5 | ItemCoverage@K | paper_formula_recovered_and_official_code_recovered | BOTH | 5 | unique items in sampled candidate pools | PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_PAPER_EQUIVALENT |
| diversity | Overlap Item Coverage@5 | OIC@K | paper_formula_recovered_and_official_code_recovered | BOTH | 5 | number of qualifying user pairs and their candidate intersections | PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;NOT_IMPLEMENTED |
| diversity | Shannon category entropy@5 | third diversity metric not named in extracted paper text | paper_underspecified_third_diversity_metric; official_code_recovered | BOTH | 5 | sample users | PAPER_UNDERSPECIFIED;OFFICIAL_CODE_RECOVERED;NOT_IMPLEMENTED |
| fairness | Gini@5 | Gini@K | paper_formula_recovered_and_official_code_recovered_with_denominator_difference | BOTH | 5 | paper: all items I; official code: only items appearing in top-K recommendations | PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;NOT_IMPLEMENTED |
| fairness | DPD@5 | Demographic Parity Difference@K | paper_formula_recovered_and_official_code_recovered | BOTH | 5 | active and inactive user groups | PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;NOT_IMPLEMENTED |
| fairness | Jain's Index@5 | Jain's Index | paper_formula_recovered_and_official_code_recovered | BOTH | 5 | sample users and squared NDCG sum | PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;NOT_IMPLEMENTED |
| LLM_specific | Hallucination rate@5 | Hallucination | paper_prose_recovered_with_formula_typo; official_code_recovered; local_adapted | BOTH | 5 | LLM generated recommendation positions | PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_ADAPTED |
| LLM_specific | CandDif-HR | CandDif_Acc with Acc=HR | paper_formula_recovered; official_batch_commands_partial | RANKING | 5 | same sampled users in both conditions | PAPER_SPECIFIED;OFFICIAL_CODE_UNRECOVERABLE;LOCAL_ADAPTED |
| LLM_specific | CandDif-NDCG | CandDif_Acc with Acc=NDCG | paper_formula_recovered; official_batch_commands_partial | RANKING | 5 | same sampled users in both conditions | PAPER_SPECIFIED;OFFICIAL_CODE_UNRECOVERABLE;LOCAL_ADAPTED |
| LLM_specific | History-length sensitivity | History Length Sensitivity | paper_protocol_recovered; exact bins are script/artifact dependent | BOTH | 5 | same sampled users per history-length condition | PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_ADAPTED |
| LLM_specific | Generation-involved profile recommendation | Generation-Involved Performance | paper_protocol_recovered; profile quality metric not separately specified | RANKING | 5 | sample users with generated profiles and recommendation outputs | PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_ADAPTED |
| sampling_protocol | K-S sample representativeness gate | Kolmogorov-Smirnov sample test | paper_protocol_recovered_but_operational_details_underspecified | OTHER | not_applicable | full test users and sampled users | PAPER_SPECIFIED;OFFICIAL_CODE_UNRECOVERABLE;NOT_IMPLEMENTED |

## Paper Analysis Dimensions

- History-length sensitivity: compare HR@5/NDCG@5 across controlled history lengths; requires repeated history-variant outputs for Dataset 5.
- Candidate-position bias: CandDif-HR and CandDif-NDCG compare first-position versus random-position positive placement using the inverse-log transform.
- Generation-involved performance: generate user profiles from histories, then compare history-only, profile-only, and profile-plus-history recommendation conditions.
- Hallucination: paper prose defines unmatched/fabricated title recommendations using string matching; local JSON/index structural validity is separate.
- Sampling protocol: paper uses a K-S representativeness gate, but exact local alpha/metric-vector implementation is unrecovered.

## Paper vs Official Code

- SelfInformation: paper formula uses unique-user probability |U|/|U_i|; official/local artifacts use frequency dictionaries and local train-only popularity adaptations.
- Serendipity: paper prose says correct recommendations not made by MostPop, while the displayed formula omits the correctness indicator; official code uses hit_results and MostPop hit_results.
- APLT: paper names bottom 80 percent by popularity; official code computes frequency < 0.8 quantile high_bar under the name calACLT.
- Gini: paper denominator is all items I; official code explicitly considers only items appearing in top-K recommendation lists.
- Diversity: paper extracted text says three diversity metrics but only names/formula-recovers ItemCoverage and OIC; official code recovers Shannon/category entropy as the third implementation.
- Hallucination: paper prose and optimal-value sentence indicate unmatched fabricated items; extracted formula indicator appears inverted, while official code counts mapped item id 0.

## Local Implementation Map

| metric_name | local_implementation_file | local_function_or_class | tests | implementation_status |
| --- | --- | --- | --- | --- |
| HR@5 | src/evaluation/utility_metrics.py | hit_rate_at_k; evaluate_utility_at_k; hr_at_k_from_rank | tests/test_utility_metrics.py; tests/test_canonical_metric_contract.py; tests/test_canonical_metric_computation.py | IMPLEMENTED_AND_VALIDATED |
| NDCG@5 | src/evaluation/utility_metrics.py | ndcg_at_k; evaluate_utility_at_k; ndcg_at_k_from_rank | tests/test_utility_metrics.py; tests/test_canonical_metric_contract.py; tests/test_canonical_metric_computation.py | IMPLEMENTED_AND_VALIDATED |
| MRR printed by official script | src/metrics/canonical_metric_contract.py | mean_positive_rank_zero_based; median_positive_rank_zero_based | tests/test_canonical_metric_contract.py | IMPLEMENTED_ADAPTED |
| SelfInformation@5 | src/metrics/canonical_metric_contract.py; src/metrics/canonical_metric_computation.py | novelty_self_information; self_information_bits | tests/test_canonical_metric_contract.py; tests/test_canonical_metric_computation.py | IMPLEMENTED_ADAPTED |
| Serendipity@5 | src/analysis/finalization_f5d_serendipity.py | F5D adapted offline computation | tests/test_finalization_f5d_serendipity.py | IMPLEMENTED_ADAPTED |
| ARP@5 | src/metrics/canonical_metric_computation.py; src/analysis/finalization_f5c_popularity_bias.py | average_popularity; F5C ARP computation | tests/test_canonical_metric_computation.py; tests/test_finalization_f5c_popularity_bias.py | IMPLEMENTED_ADAPTED |
| APLT@5 | src/metrics/canonical_metric_contract.py; src/metrics/canonical_metric_computation.py | build_long_tail_set; long_tail_rate | tests/test_canonical_metric_contract.py; tests/test_canonical_metric_computation.py | IMPLEMENTED_ADAPTED |
| PopREO@5 | not implemented as final local metric | not_applicable | tests/test_finalization_f5c_popularity_bias.py validates blocked status | MISSING |
| ItemCoverage@5 | src/metrics/canonical_metric_contract.py; src/metrics/canonical_metric_computation.py | coverage_ratio with strict candidate union | tests/test_canonical_metric_computation.py | IMPLEMENTED_AND_VALIDATED |
| Overlap Item Coverage@5 | not implemented as final local metric | not_applicable | not_applicable | MISSING |
| Shannon category entropy@5 | not implemented as final local metric | not_applicable | not_applicable | MISSING |
| Gini@5 |  |  |  | MISSING |
| DPD@5 |  |  |  | MISSING |
| Jain's Index@5 |  |  |  | MISSING |
| Hallucination rate@5 | src/evaluation/hallucination.py | hallucination_rate; catalog_hallucination_rate; candidate_hallucination_rate | tests/test_hallucination.py; tests/test_output_parser.py | IMPLEMENTED_ADAPTED |
| CandDif-HR | src/evaluation/position_bias.py | selected_candidate_position_distribution; position_bias_summary | tests/test_position_bias.py | IMPLEMENTED_ADAPTED |
| CandDif-NDCG | src/evaluation/position_bias.py | position-bias summary utilities plus offline metric calculation | tests/test_position_bias.py | IMPLEMENTED_ADAPTED |
| History-length sensitivity | src/prompts/history_length.py | history prompt/control helpers | tests/test_history_length.py; tests/test_plan_d_history_length_sensitivity.py | IMPLEMENTED_ADAPTED |
| Generation-involved profile recommendation | src/analysis/plan_f6l_paper_faithful_completion_and_rq_revalidation.py | profile request/metric artifact builders | tests/test_phase_f6l_paper_faithful_completion_and_rq_revalidation.py | IMPLEMENTED_ADAPTED |
| K-S sample representativeness gate |  |  |  | MISSING |
| Catalog Coverage@5 | src/evaluation/coverage.py | catalog_coverage_at_k; coverage_by_model | tests/test_coverage.py | IMPLEMENTED_AND_VALIDATED |
| Intra-List Diversity@5 | src/metrics/canonical_metric_contract.py; src/analysis/finalization_f5b_ild_policy.py | intra_list_diversity_jaccard; F5B adapted policy | tests/test_finalization_f5b_ild_policy.py; tests/test_canonical_metric_contract.py | IMPLEMENTED_ADAPTED |
| Calibration@5 | src/analysis/finalization_f5e_calibration.py | F5E calibration builders; JS/Hellinger helpers in local analysis | tests/test_finalization_f5e_calibration.py | IMPLEMENTED_ADAPTED |
| Strict parse validity rate | src/llm/canonical_local_llm_runner.py; src/llm/plan_d_llm_json_parser.py | strict output parser and validation summaries | tests/test_canonical_local_llm_runner.py; tests/test_output_parser.py | IMPLEMENTED_AND_VALIDATED |
| Duplicate rate@5 | src/metrics/canonical_metric_contract.py; src/metrics/canonical_metric_computation.py | duplicate_rate_at_k; duplicate_rate | tests/test_canonical_metric_contract.py; tests/test_canonical_metric_computation.py | IMPLEMENTED_AND_VALIDATED |
| Candidate Recall@N | future extension, not final metric module | not_applicable | not_applicable | IMPLEMENTED_ADAPTED |
| Conditional HR/NDCG@5 | src/analysis/phase_f6l_lastfm_v4_reranking_continuation_v4_1.py | positive-present diagnostic aggregation | tests/test_phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution.py | IMPLEMENTED_ADAPTED |
| Normalized popularity, popularity shift, popularity lift | src/analysis/finalization_f5c_popularity_bias.py | F5C extension computations | tests/test_finalization_f5c_popularity_bias.py | IMPLEMENTED_ADAPTED |
| Cost/token/latency summary | src/evaluation/cost_metrics.py | summarize_tokens_by_model; summarize_latency_by_model; aggregate_cost_metrics_by_model | tests/test_cost_metrics.py | IMPLEMENTED_AND_VALIDATED |

## Four-Dataset Coverage

| metric_or_analysis | beauty | sports | ml_1m | lastfm |
| --- | --- | --- | --- | --- |
| HR@5 | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| NDCG@5 | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| SelfInformation@5 | ADAPTED | ADAPTED | ADAPTED | PARTIAL |
| Serendipity@5 | ADAPTED | ADAPTED | ADAPTED | NOT_RUN |
| ARP@5 | ADAPTED | ADAPTED | ADAPTED | PARTIAL |
| APLT@5 | ADAPTED | ADAPTED | ADAPTED | PARTIAL |
| PopREO@5 | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| ItemCoverage@5 | COMPLETE | COMPLETE | COMPLETE | NOT_RUN |
| OIC@5 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Shannon category entropy@5 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Gini@5 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| DPD@5 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Jain's Index@5 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Hallucination rate@5 | ADAPTED | ADAPTED | ADAPTED | ADAPTED |
| CandDif-HR | PARTIAL | PARTIAL | PARTIAL | NOT_RUN |
| CandDif-NDCG | PARTIAL | PARTIAL | PARTIAL | NOT_RUN |
| History-length sensitivity | PARTIAL | PARTIAL | PARTIAL | NOT_RUN |
| Generation-involved profile matrix | PARTIAL | PARTIAL | NOT_RUN | NOT_RUN |
| K-S sample representativeness | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| Catalog Coverage@5 | COMPLETE | COMPLETE | COMPLETE | PARTIAL |
| ILD@5 | ADAPTED | ADAPTED | ADAPTED | PARTIAL |
| Calibration@5 | ADAPTED | ADAPTED | ADAPTED | PARTIAL |
| Strict parse validity rate | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| Duplicate rate@5 | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| Candidate Recall@N | NOT_RUN | NOT_RUN | NOT_RUN | PARTIAL |
| Conditional HR/NDCG@5 | NOT_RUN | NOT_RUN | NOT_RUN | PARTIAL |

## Fifth-Dataset Mandatory Contract

| requirement | needed_by_metric_analysis | mandatory | requirement_class | why | evidence_or_policy |
| --- | --- | --- | --- | --- | --- |
| Public, reproducible dataset access and license | all Dataset 5 work | yes | MANDATORY_FOR_PAPER_FRAMEWORK | The fifth dataset must be auditable before preprocessing or scoring. | E2 scorecard gate |
| User-item interaction rows with stable user_id and item_id | HR/NDCG, all beyond-utility metrics, baselines | yes | MANDATORY_FOR_PAPER_FRAMEWORK | Every paper metric depends on user/item identity and held-out targets. | paper protocol; local schemas |
| Chronological order or timestamps | leave-one-out split, history sensitivity, sequential models, profile generation | yes | MANDATORY_FOR_PAPER_FRAMEWORK | Histories must be truncated and targets held out consistently. | paper 5-core/leave-one-out protocol |
| Item titles or names | LLM prompts, hallucination, profile generation, BM25 | yes | MANDATORY_FOR_PAPER_FRAMEWORK | The paper forms histories and candidates from item titles. | paper lines 383-386; official templates |
| Item metadata categories/tags/genres | Shannon, adapted ILD, calibration | yes_if_claimed | MANDATORY_FOR_LOCAL_REPRODUCTION_PARITY | Paper/code diversity and local extensions need feature coverage. | official calMeanShannon; F5B/F5E |
| Train-only item popularity counts | ARP, APLT, PopREO, SelfInformation local adaptation, MostPop | yes | MANDATORY_FOR_PAPER_FRAMEWORK | Popularity metrics and MostPop primitive require frozen popularity source. | official llm_metrics.py; F5C |
| Unique-user item popularity counts | paper SelfInformation | yes | MANDATORY_FOR_PAPER_FRAMEWORK | Paper formula uses \|U\| and \|U_i\|. | arXiv PDF appendix lines 959-975 |
| Ground-truth positive target per sampled user | HR/NDCG, Serendipity, PopREO, reranking diagnostics | yes | MANDATORY_FOR_PAPER_FRAMEWORK | Utility and hit-conditioned metrics require the held-out item. | paper ranking setup |
| Ranking candidate pools with one positive and 19 negatives | ranking utility, hallucination/candidate validity, ItemCoverage, OIC, CandDif | yes | MANDATORY_FOR_PAPER_FRAMEWORK | The paper ranking setting uses 20 candidates. | paper lines 441-446 |
| Reranking candidate pools from BPRMF, GRU4Rec, SASRec, and LightGCN top-K outputs | reranking utility and beyond-utility | yes_for_reranking | MANDATORY_FOR_PAPER_FRAMEWORK | The paper reranking setting forms 20-item pools from four traditional models. | paper lines 447-450 |
| Candidate order and positive position metadata | CandDif-HR, CandDif-NDCG, position-bias diagnostics | yes | MANDATORY_FOR_PAPER_FRAMEWORK | Position perturbation cannot be evaluated without candidate positions. | paper Eq. 8 |
| History length bins with sufficient depth | history-length sensitivity, profile generation | yes | MANDATORY_FOR_PAPER_FRAMEWORK | Users must have histories deep enough for the selected bins. | metrics.sh history commands |
| Aligned MostPop recommendation/hit vector | Serendipity, MostPop baseline | yes | MANDATORY_FOR_PAPER_FRAMEWORK | Serendipity uses a MostPop primitive and must share cohort/candidates. | official calSerendipity; F5D |
| Popularity quintile groups with zero-denominator policy | PopREO | yes | MANDATORY_FOR_PAPER_FRAMEWORK | PopREO divides items into five popularity groups and can fail with empty groups. | paper Eq. 16-17; F5C blocker |
| User activity groups based on median history length | DPD | yes | MANDATORY_FOR_PAPER_FRAMEWORK | DPD compares active and inactive users. | paper Eq. 21 |
| Catalog item universe with exposure denominator policy | Gini, catalog coverage extension, hallucination | yes | MANDATORY_FOR_PAPER_FRAMEWORK | Paper Gini and hallucination require item universe definitions. | paper Eq. 20 and hallucination prose |
| Free-text/title-list LLM outputs or paper-equivalent parser outputs | Hallucination | yes_for_paper_hallucination | MANDATORY_FOR_PAPER_FRAMEWORK | Paper hallucination is based on string matching, not only JSON/index validity. | paper hallucination section |
| Profile-generation outputs and dependency manifest | generation-involved performance | yes | MANDATORY_FOR_PAPER_FRAMEWORK | Profile-only and profile-plus-history conditions depend on generated profiles. | paper Eq. 9; official templates |
| Traditional model training-ready split and configs | BM25, BPRMF, GRU4Rec, SASRec, LightGCN | yes_if_reproducing_traditional_matrix | MANDATORY_FOR_PAPER_FRAMEWORK | Paper compares against six trained/traditional systems. | paper model section; appendix A.2 |
| Raw output manifests, parser provenance, and invalid-output policy | LLM utility, hallucination, strict validity, duplicate rate | yes | MANDATORY_FOR_LOCAL_REPRODUCTION_PARITY | Local reproduction parity depends on explicit denominator and parser policies. | canonical local LLM artifacts |
| Candidate retrieval and aggregation intermediate artifacts | Candidate Recall, oracle ceiling, pipeline decomposition | yes_later | MANDATORY_FOR_NEW_EXTENSION | Pipeline-aware extensions require upstream candidate-stage evidence. | extension order freeze |
| Positive-injection intervention records | positive injection, conditional vs end-to-end utility | yes_later | MANDATORY_FOR_NEW_EXTENSION | Future interventions must be auditable and separated from paper framework. | extension order freeze |
| Token, latency, and cost logs | cost/token/latency summary | optional | OPTIONAL | Useful for budget planning but not a paper metric. | src/evaluation/cost_metrics.py |

## Additional Inference Requirements

| analysis_id | metric_or_analysis | task | recommender_scope | inference_requirement | extra_llm_inference | approximate_multiplicative_factor | why | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| utility | HR@5 and NDCG@5 | BOTH | both | REQUIRES_NEW_RANKING_INFERENCE;REQUIRES_NEW_RERANKING_INFERENCE;NO_LLM_REQUIRED | yes_for_LLM_rows | 1x per model/prompt/task | Metrics are offline once outputs exist; Dataset 5 has no outputs yet. |  |
| novelty_popularity_diversity_fairness | SelfInformation, Serendipity, ARP, APLT, PopREO, ItemCoverage, OIC, Shannon, Gini, DPD, Jain | BOTH | both | OFFLINE_FROM_EXISTING_RANKING;NO_LLM_REQUIRED | no_beyond_base_outputs | 0x beyond the base output matrix | These metrics consume existing ranked outputs and dataset metadata. |  |
| hallucination | Paper hallucination | BOTH | LLM | REQUIRES_FREE_TEXT_OUTPUT;REQUIRES_NEW_RANKING_INFERENCE;REQUIRES_NEW_RERANKING_INFERENCE | yes | 1x per LLM model/prompt/task if title-list outputs are not already generated | Paper hallucination needs title/string matching against item universe. | Local JSON/index validity alone is not enough. |
| history_length_sensitivity | History-length sensitivity | BOTH | LLM and traditional comparison | REQUIRES_HISTORY_VARIATION | yes_for_LLM | number_of_history_lengths x model/prompt/task | Each controlled history length needs its own LLM outputs unless cached. |  |
| candidate_position_bias | CandDif-HR and CandDif-NDCG | RANKING | LLM | REQUIRES_POSITION_PERTURBATION | yes | at least 2x ranking per model/prompt: random position plus first position | CandDif compares accuracy under two candidate-position conditions. |  |
| profile_generation | Generation-involved profile matrix | RANKING | LLM | REQUIRES_PROFILE_GENERATION;REQUIRES_NEW_RANKING_INFERENCE | yes | profile generation plus profile-only/profile-plus-history recommendation cells | Profiles must be generated before dependent recommendation prompts. |  |
| ks_sample | K-S representativeness gate | OTHER | traditional | NO_LLM_REQUIRED | no | 0x LLM; requires traditional full/sample output distributions | Sampling gate should be offline from traditional model metrics. |  |
| local_extensions | Calibration, ILD, candidate recall, strict validity, duplicate rate | BOTH | both/LLM by metric | OFFLINE_FROM_EXISTING_RANKING;NO_LLM_REQUIRED | no_after_outputs | 0x beyond output matrix, except strict validity requires raw LLM outputs | These must run only after paper/local parity is complete. |  |

## Traditional Model Contract

| Model | Paper role | Ranking/reranking role | Local implementation | Dataset requirements |
| --- | --- | --- | --- | --- |
| Random | local baseline; not in paper model list recovered from Section 4.1.2 | ranking diagnostic | implemented locally | candidate pools and seed |
| MostPop | paper traditional ID-based model | ranking and candidate primitive | official code and local implementation | train popularity counts |
| BM25 | paper traditional content-based model | ranking and reranking pool source | local content BM25 exists for ML-1M/Plan C scope | item titles/text and histories |
| BPRMF | paper traditional ID-based model | ranking and reranking pool source | local Plan E/F6D adapted trainers | implicit feedback train split |
| GRU4Rec | paper traditional ID-based sequential model | ranking and reranking pool source | local Plan E/F6D adapted trainers | chronological sequences |
| SASRec | paper traditional ID-based sequential model | ranking and reranking pool source | local Plan E/F6D adapted trainers | chronological sequences |
| LightGCN | paper traditional ID-based graph model | ranking and reranking pool source | local Plan E/F6D adapted trainers | user-item graph |

## LLM Contract

| Paper model | Exact identifier | Open/closed | Local feasibility |
| --- | --- | --- | --- |
| GPT-3.5 | gpt-3.5-turbo-0125 | closed | blocked: paid API/no local exact run |
| GPT-4o-mini | gpt-4o-mini-2024-07-18 | closed | blocked: paid API/no local exact run |
| GPT-4o | gpt-4o-2024-08-06 | closed | blocked: paid API/no local exact run |
| Claude 3 Haiku | claude-3-haiku-20240307 | closed | blocked: paid API/no local exact run |
| Llama3 70B Instruct | llama-3-70b-instruct | open/open-weight | blocked locally by hardware/exact execution |
| Qwen2 7B Instruct | qwen-2-7b-instruct | open/open-weight | locally adapted via qwen2:7b-instruct, not exact |
| Mistral 7B Instruct | mistral-7b-instruct-v0.2 | open/open-weight | locally adapted via mistral:7b-instruct, not exact |

## Local Extensions

| metric_name | formula_status | provenance_classification | fifth_dataset_requirement_classification | notes |
| --- | --- | --- | --- | --- |
| Catalog Coverage@5 | local_extension_formula | LOCAL_EXTENSION | MANDATORY_FOR_LOCAL_REPRODUCTION_PARITY |  |
| Intra-List Diversity@5 | local_adapted_formula_not_paper_native | LOCAL_ADAPTED;LOCAL_EXTENSION | MANDATORY_FOR_LOCAL_REPRODUCTION_PARITY |  |
| Calibration@5 | local_extension_formula | LOCAL_EXTENSION | MANDATORY_FOR_NEW_EXTENSION |  |
| Strict parse validity rate | local_extension_formula | LOCAL_EXTENSION | MANDATORY_FOR_LOCAL_REPRODUCTION_PARITY |  |
| Duplicate rate@5 | local_extension_formula | LOCAL_EXTENSION | MANDATORY_FOR_LOCAL_REPRODUCTION_PARITY |  |
| Candidate Recall@N | future_extension_formula_from_local_planning | LOCAL_EXTENSION | MANDATORY_FOR_NEW_EXTENSION | Must remain after paper/local-reproduction parity, not before it. |
| Conditional HR/NDCG@5 | local_diagnostic_formula | LOCAL_EXTENSION | MANDATORY_FOR_NEW_EXTENSION |  |
| Normalized popularity, popularity shift, popularity lift | local_extension_formula | LOCAL_EXTENSION | MANDATORY_FOR_NEW_EXTENSION |  |
| Cost/token/latency summary | local_extension_supporting_analysis | LOCAL_EXTENSION | OPTIONAL |  |

## Unresolved Metric Details

- PopREO: formula recovered, but empty-group and zero-denominator policy remains unfrozen locally.
- OIC and Shannon: recovered from paper/code, but not implemented as final local metrics.
- Gini: paper/code denominator conflict remains explicit.
- K-S sample gate: exact alpha and metric-vector policy unrecovered.
- Hallucination: paper formula indicator appears inconsistent with prose/code; use paper prose plus official unmatched-id semantics, and record the discrepancy.

## Fifth-Dataset Scorecard

| criterion | category | weight | mandatory_gate | evidence_needed | scoring_guidance |
| --- | --- | --- | --- | --- | --- |
| Public availability | access | high | yes | source URL, access instructions | Reject if not reproducibly accessible. |
| License/access terms | access | high | yes | license text or terms summary | Reject if redistribution/use is unclear for academic reproducibility. |
| Reproducible download | access | high | yes | download command or manual intake manifest | Prefer stable public archive with checksum. |
| User-item interactions | schema | high | yes | raw interaction columns | Must map to user_id and item_id. |
| Timestamps/order | schema | high | yes | timestamp/order field | Must support leave-one-out and history truncation. |
| Item titles | LLM compatibility | high | yes | title/name field coverage | Must support prompts and hallucination matching. |
| Item metadata/categories | beyond-utility | medium | yes_if_claiming_diversity_calibration | category/tag/genre fields and coverage | Needed for Shannon, adapted ILD, and calibration. |
| History depth | protocol | high | yes | post-filter distribution | Must support history bins and profile generation. |
| Post-filter user count | scale | medium | yes | 5-core statistics | Prefer enough users for 1000-user sample or explicit smaller policy. |
| Post-filter item count | scale | medium | yes | 5-core statistics | Must be large enough for candidate pools and long-tail groups. |
| Interaction density | scale | medium | no | density statistics | Assess traditional model feasibility. |
| Long-tail properties | beyond-utility | high | yes | popularity distribution | Must make APLT/ARP/PopREO meaningful. |
| Leave-one-out compatibility | protocol | high | yes | split construction memo | Reject if no chronological target can be held out. |
| Paper metric compatibility | protocol | high | yes | requirements coverage checklist | Must satisfy mandatory paper metrics before extensions. |
| Profile generation compatibility | LLM compatibility | medium | yes | history/title suitability | Need histories readable as prompt text. |
| Hallucination evaluation compatibility | LLM compatibility | high | yes | title uniqueness and parser policy | Need matchable item universe. |
| Candidate-position experiment compatibility | protocol | high | yes | candidate order manipulation plan | Need position metadata and deterministic permutations. |
| Calibration compatibility | extension | medium | yes_later | item feature metadata | Extension only after paper parity. |
| BPRMF compatibility | traditional models | medium | yes_if_reproducing_traditional_matrix | implicit feedback matrix | Assess training feasibility. |
| GRU4Rec compatibility | traditional models | medium | yes_if_reproducing_traditional_matrix | chronological sessions/sequences | Assess sequence preparation. |
| SASRec compatibility | traditional models | medium | yes_if_reproducing_traditional_matrix | chronological sequences | Assess training feasibility. |
| LightGCN compatibility | traditional models | medium | yes_if_reproducing_traditional_matrix | user-item graph | Assess graph size and memory. |
| Compute cost | runtime | medium | yes | row counts, model/training estimates | Reject if impossible under local no-paid-API policy. |
| LLM prompt suitability | LLM compatibility | medium | yes | sample titles/histories | Avoid domains where titles are cryptic or non-recommendable. |
| Domain novelty relative to Beauty/Sports/Movies/Music | scientific value | medium | no | domain description | Prefer a genuinely new domain. |

## Extension Ordering

paper framework on Dataset 5 -> parity validation -> new metric -> pipeline-aware extension -> secondary ablations

## Repository Postcondition

- Git: E1 creates new design-evidence files; inspect `git status --short --branch` after generation.
- original_repo: not modified by this script.
- tracked changes: generated E1 report/tables plus optional validation test and notes when applied.
- staged changes: none created by this script.

## Commit

No commit is created by the generation script. If the user explicitly requests it after validation, use message `extension: freeze complete paper metric contract`.

## Interpretation

YES. We now have a provenance-aware contract for reproducing on Dataset 5 the paper/local-reproduction framework before applying original extensions, with conflicts and missing local implementations explicitly marked rather than silently reconciled.

## Next Recommended Action

perform E2 fifth-dataset audit and selection against the frozen metric contract

## Source Notes

- Local PDF: `paper/2411.00331v1.pdf sha256=A143265AE42F027C079766B1C006C438D698897B732634D78A010EB0A62F0F23`.
- arXiv PDF text checked for formulas: https://arxiv.org/pdf/2411.00331.
- Official repository metric source: `original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py`.
- Official repository execution batch source: `original_repo/EvaLLMasRecommender/eva/codes/metrics.sh`.
- Official README: `original_repo/EvaLLMasRecommender/README.md`.
