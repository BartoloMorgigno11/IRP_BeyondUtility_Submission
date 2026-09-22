# Plan D Phase 5E Final Paper Coverage Audit

Prominent constraints: the maximum defensible coverage percentage is not an exact paper-replication percentage; Qwen2 and Mistral are local substitutes/extensions; numerical similarity is not proof of replication; extensions do not increase paper-reproduction coverage; blocked components remain in the applicable denominator; same-metric/different-cohort results are non-exact comparisons.

## 1. Objective

Phase 5E audits final paper coverage from existing evidence only.

## 2. Scope

This is not repository consolidation, README finalization, plotting, presentation work, or Phase 6.

## 3. Offline Methodology

The builder reads existing paper, official repo, Phase 5C, Phase 5D, metric, baseline, and report artifacts. It performs no inference, training, downloads, API calls, prompt tuning, parser edits, staging, or commits.

## 4. Source Hierarchy

1. immutable_raw_local_experiment_artifact
2. phase_5c_full_matrix_evidence
3. phase_5d_robustness_consolidation
4. canonical_manifest_or_metric_contract
5. exact_paired_diagnostic_experiment
6. source_or_generated_result_table
7. earlier_phase_report
8. paper_pdf
9. original_repository_code_or_documentation
10. report_only_summary
11. none

## 5. Protected-Input Policy

Phase 5E excludes only Phase 5E outputs from protected manifests. Paper, original repo, raw LLM artifacts, manifests, prompts, parsers, metric contracts, baseline outputs, Phase 5C, Phase 5D, and previous reports are protected.

## 6. Paper Component Taxonomy

| category | total_components | applicable_components | maximum_defensible_pct | blocked_pct |
| --- | --- | --- | --- | --- |
| LLM models | 7 | 7 | 28.571429 | 71.428571 |
| beyond-utility metrics | 6 | 6 | 66.666667 | 33.333333 |
| candidate-pool construction | 5 | 5 | 100.0 | 0.0 |
| candidate-position analyses | 2 | 2 | 50.0 | 0.0 |
| dataset preprocessing and filtering | 4 | 4 | 75.0 | 25.0 |
| datasets | 4 | 4 | 75.0 | 25.0 |
| generated-profile analyses | 2 | 2 | 0.0 | 100.0 |
| hallucination and output-reliability analyses | 2 | 2 | 100.0 | 0.0 |
| history-length analyses | 2 | 2 | 100.0 | 0.0 |
| paper figures and qualitative analysis groups | 2 | 2 | 50.0 | 0.0 |
| paper tables and quantitative result groups | 3 | 3 | 0.0 | 0.0 |
| prompt protocols | 5 | 5 | 40.0 | 20.0 |
| ranking task | 2 | 2 | 100.0 | 0.0 |
| released code and data artifacts | 6 | 6 | 100.0 | 0.0 |
| reproducibility-relevant execution settings | 4 | 4 | 50.0 | 50.0 |
| reranking task | 2 | 2 | 50.0 | 0.0 |
| response-format and parser contracts | 3 | 3 | 100.0 | 0.0 |
| traditional recommender baselines | 6 | 6 | 100.0 | 0.0 |
| user cohorts and sampling | 4 | 4 | 75.0 | 0.0 |
| utility metrics | 2 | 2 | 100.0 | 0.0 |

## 7. Denominator Construction

Applicable component denominator: 73. Blocked, not-executed, and unresolved rows remain in the denominator. Not-applicable rows alone are excluded.

## 8. Coverage-Status Definitions

Statuses are count-based: reproduced_exact, reconstructed, adapted, extension_only, blocked, not_executed, unresolved, and not_applicable. No partial scores or subjective weights are used.

## 9. Component-by-Component Findings

| component_id | category | component | primary_coverage_status | comparability_classification |
| --- | --- | --- | --- | --- |
| PC-LLM-MODELS-CLAUDE-3-HAIKU | LLM models | Claude 3 Haiku | blocked | blocked_not_comparable |
| PC-LLM-MODELS-GPT-3-5 | LLM models | GPT-3.5 | blocked | blocked_not_comparable |
| PC-LLM-MODELS-GPT-4O | LLM models | GPT-4o | blocked | blocked_not_comparable |
| PC-LLM-MODELS-GPT-4O-MINI | LLM models | GPT-4o-mini | blocked | blocked_not_comparable |
| PC-LLM-MODELS-LLAMA3-70B-INSTRUCT | LLM models | Llama3-70B-Instruct | blocked | blocked_not_comparable |
| PC-LLM-MODELS-MISTRAL-7B-LOCAL-SUBSTITUTE | LLM models | Mistral 7B local substitute | adapted | adapted_local_model_extension |
| PC-LLM-MODELS-QWEN2-7B-LOCAL-SUBSTITUTE | LLM models | Qwen2 7B local substitute | adapted | adapted_local_model_extension |
| PC-BEYOND-UTILITY-METRICS-AVERAGE-POPULARITY-METRIC | beyond-utility metrics | Average popularity metric | reconstructed | reconstructed_paper_like |
| PC-BEYOND-UTILITY-METRICS-DIVERSITY-METRIC | beyond-utility metrics | Diversity metric | blocked | blocked_not_comparable |
| PC-BEYOND-UTILITY-METRICS-ITEM-COVERAGE-METRIC | beyond-utility metrics | Item coverage metric | reconstructed | reconstructed_paper_like |
| PC-BEYOND-UTILITY-METRICS-LONG-TAIL-RATE-METRIC | beyond-utility metrics | Long-tail rate metric | reconstructed | reconstructed_paper_like |
| PC-BEYOND-UTILITY-METRICS-NOVELTY-OR-SELF-INFORMATION-METRIC | beyond-utility metrics | Novelty or self-information metric | reconstructed | reconstructed_paper_like |
| PC-BEYOND-UTILITY-METRICS-POPULARITY-BIAS-FAIRNESS-METRICS | beyond-utility metrics | Popularity-bias/fairness metrics | blocked | blocked_not_comparable |
| PC-CANDIDATE-POOL-CONSTRUCTION-1-POSITIVE-PLUS-19-NEGATIVES | candidate-pool construction | 1 positive plus 19 negatives | reconstructed | reconstructed_paper_like |
| PC-CANDIDATE-POOL-CONSTRUCTION-20-ITEM-CANDIDATE-POOL | candidate-pool construction | 20-item candidate pool | reconstructed | reconstructed_paper_like |
| PC-CANDIDATE-POOL-CONSTRUCTION-CANDIDATE-ORDER-VARIANTS | candidate-pool construction | Candidate order variants | adapted | diagnostic_only |
| PC-CANDIDATE-POOL-CONSTRUCTION-RANKING-NEGATIVE-CANDIDATE-CONSTRUCTION | candidate-pool construction | Ranking negative-candidate construction | reconstructed | reconstructed_paper_like |
| PC-CANDIDATE-POOL-CONSTRUCTION-RERANKING-CANDIDATE-CONSTRUCTION | candidate-pool construction | Reranking candidate construction | adapted | same_dataset_different_protocol |
| PC-CANDIDATE-POSITION-ANALYSES-CANDIDATE-FIRST-PROMPT-CONDITION | candidate-position analyses | Candidate-first prompt condition | not_executed | blocked_not_comparable |
| PC-CANDIDATE-POSITION-ANALYSES-CANDIDATE-POSITION-BIAS-ANALYSIS | candidate-position analyses | Candidate position bias analysis | adapted | diagnostic_only |
| PC-DATASET-PREPROCESSING-AND-FILTERING-AMAZON-TITLE-AND-ITEM-MAPPING | dataset preprocessing and filtering | Amazon title and item mapping | reconstructed | reconstructed_paper_like |
| PC-DATASET-PREPROCESSING-AND-FILTERING-MOVIELENS-SEQUENTIAL-SPLIT | dataset preprocessing and filtering | MovieLens sequential split | reconstructed | reconstructed_paper_like |
| PC-DATASET-PREPROCESSING-AND-FILTERING-OFFICIAL-PROCESSED-DATA-ARTIFACTS | dataset preprocessing and filtering | Official processed data artifacts | blocked | blocked_not_comparable |
| PC-DATASET-PREPROCESSING-AND-FILTERING-STRICT-TITLE-PROMPT-COMPATIBLE-FILTERING | dataset preprocessing and filtering | Strict-title prompt-compatible filtering | adapted | same_dataset_different_protocol |
| PC-DATASETS-AMAZON-BEAUTY-DATASET | datasets | Amazon Beauty dataset | reconstructed | reconstructed_paper_like |
| PC-DATASETS-AMAZON-SPORTS-DATASET | datasets | Amazon Sports dataset | reconstructed | reconstructed_paper_like |
| PC-DATASETS-LASTFM-DATASET | datasets | LastFM dataset | blocked | blocked_not_comparable |
| PC-DATASETS-ML-1M-DATASET | datasets | ML-1M dataset | reconstructed | reconstructed_paper_like |
| PC-GENERATED-PROFILE-ANALYSES-PROFILE-BASED-RECOMMENDATION-EFFECT | generated-profile analyses | Profile-based recommendation effect | blocked | blocked_not_comparable |
| PC-GENERATED-PROFILE-ANALYSES-USER-PROFILE-GENERATION | generated-profile analyses | User profile generation | blocked | blocked_not_comparable |
| PC-HALLUCINATION-AND-OUTPUT-RELIABILITY-ANALYSES-HALLUCINATION-RATE | hallucination and output-reliability analyses | Hallucination rate | adapted | adapted_local_model_extension |
| PC-HALLUCINATION-AND-OUTPUT-RELIABILITY-ANALYSES-OUTPUT-COMPLIANCE-AND-DUPLICATE-FAILURES | hallucination and output-reliability analyses | Output compliance and duplicate failures | adapted | adapted_local_model_extension |
| PC-HISTORY-LENGTH-ANALYSES-COLD-START-SHORT-HISTORY-OBSERVATION | history-length analyses | Cold-start short-history observation | adapted | diagnostic_only |
| PC-HISTORY-LENGTH-ANALYSES-HISTORY-LENGTH-SENSITIVITY-ANALYSIS | history-length analyses | History length sensitivity analysis | adapted | diagnostic_only |
| PC-PAPER-FIGURES-AND-QUALITATIVE-ANALYSIS-GROUPS-FRAMEWORK-ARCHITECTURE-FIGURE | paper figures and qualitative analysis groups | Framework architecture figure | reproduced_exact | paper_exact |
| PC-PAPER-FIGURES-AND-QUALITATIVE-ANALYSIS-GROUPS-SEVEN-QUALITATIVE-OBSERVATIONS | paper figures and qualitative analysis groups | Seven qualitative observations | unresolved | unresolved_comparability |
| PC-PAPER-TABLES-AND-QUANTITATIVE-RESULT-GROUPS-BEYOND-UTILITY-RESULT-TABLES | paper tables and quantitative result groups | Beyond-utility result tables | unresolved | unresolved_comparability |
| PC-PAPER-TABLES-AND-QUANTITATIVE-RESULT-GROUPS-MODEL-COMPARISON-RESULT-GROUPS | paper tables and quantitative result groups | Model comparison result groups | unresolved | unresolved_comparability |
| PC-PAPER-TABLES-AND-QUANTITATIVE-RESULT-GROUPS-UTILITY-RESULT-TABLES | paper tables and quantitative result groups | Utility result tables | unresolved | unresolved_comparability |
| PC-PROMPT-PROTOCOLS-GENERATED-PROFILE-PROMPT-FAMILY | prompt protocols | Generated-profile prompt family | blocked | blocked_not_comparable |
| PC-PROMPT-PROTOCOLS-HISTORY-NO-HISTORY-PROMPT-VARIANTS | prompt protocols | History/no-history prompt variants | adapted | diagnostic_only |
| PC-PROMPT-PROTOCOLS-LLMRANK-LIKE-PROMPT-FAMILY | prompt protocols | LLMRank-like prompt family | not_executed | blocked_not_comparable |
| PC-PROMPT-PROTOCOLS-LLMSREC-OR-IN-CONTEXT-PROMPT-FAMILY | prompt protocols | LLMSRec or in-context prompt family | not_executed | blocked_not_comparable |
| PC-PROMPT-PROTOCOLS-OPENP5-LIKE-PROMPT-FAMILY | prompt protocols | OpenP5-like prompt family | adapted | same_dataset_different_protocol |
| PC-RANKING-TASK-LLM-RANKING-TASK | ranking task | LLM ranking task | adapted | adapted_local_model_extension |
| PC-RANKING-TASK-RANKING-TOP-5-UTILITY-CUTOFF | ranking task | Ranking top-5 utility cutoff | adapted | official_code_aligned_local_model_substitute |
| PC-RELEASED-CODE-AND-DATA-ARTIFACTS-OFFICIAL-METRIC-SOURCE-FILE | released code and data artifacts | Official metric source file | reproduced_exact | paper_exact |
| PC-RELEASED-CODE-AND-DATA-ARTIFACTS-OFFICIAL-OUTPUT-REFINEMENT-SOURCE-FILE | released code and data artifacts | Official output-refinement source file | reproduced_exact | paper_exact |
| PC-RELEASED-CODE-AND-DATA-ARTIFACTS-OFFICIAL-PROMPT-TEMPLATES | released code and data artifacts | Official prompt templates | reproduced_exact | paper_exact |
| PC-RELEASED-CODE-AND-DATA-ARTIFACTS-OFFICIAL-REPOSITORY-README | released code and data artifacts | Official repository README | reproduced_exact | paper_exact |
| PC-RELEASED-CODE-AND-DATA-ARTIFACTS-RELEASED-BEAUTY-AND-SPORTS-LLM-RESULT-ARCHIVES | released code and data artifacts | Released Beauty and Sports LLM result archives | reproduced_exact | paper_exact |
| PC-RELEASED-CODE-AND-DATA-ARTIFACTS-RELEASED-SOTA-RESULT-ARCHIVE-SAMPLE | released code and data artifacts | Released SOTA result archive sample | reproduced_exact | paper_exact |
| PC-REPRODUCIBILITY-RELEVANT-EXECUTION-SETTINGS-HARDWARE-AND-API-EXECUTION-ENVIRONMENT | reproducibility-relevant execution settings | Hardware and API execution environment | blocked | blocked_not_comparable |
| PC-REPRODUCIBILITY-RELEVANT-EXECUTION-SETTINGS-PAPER-MODEL-VERSIONS-AND-PROVIDER-SETTINGS | reproducibility-relevant execution settings | Paper model versions and provider settings | blocked | blocked_not_comparable |
| PC-REPRODUCIBILITY-RELEVANT-EXECUTION-SETTINGS-RANDOM-SEED-AND-DETERMINISTIC-LOCAL-EXECUTION | reproducibility-relevant execution settings | Random seed and deterministic local execution | reconstructed | reconstructed_paper_like |
| PC-REPRODUCIBILITY-RELEVANT-EXECUTION-SETTINGS-TEMPERATURE-ZERO-LOCAL-LLM-SETTING | reproducibility-relevant execution settings | Temperature zero local LLM setting | adapted | same_protocol_different_model |
| PC-RERANKING-TASK-LEGACY-FULL-20-RERANKING-DIAGNOSTIC | reranking task | Legacy full-20 reranking diagnostic | not_executed | same_dataset_different_protocol |
| PC-RERANKING-TASK-LLM-RERANKING-TASK | reranking task | LLM reranking task | adapted | adapted_local_model_extension |
| PC-RESPONSE-FORMAT-AND-PARSER-CONTRACTS-INVALID-OUTPUT-HANDLING-POLICY | response-format and parser contracts | Invalid-output handling policy | adapted | official_code_aligned_local_model_substitute |
| PC-RESPONSE-FORMAT-AND-PARSER-CONTRACTS-ORIGINAL-TITLE-LIST-PARSER-AND-REFINEMENT | response-format and parser contracts | Original title-list parser and refinement | reconstructed | reconstructed_paper_like |
| PC-RESPONSE-FORMAT-AND-PARSER-CONTRACTS-STRICT-JSON-INDEX-PARSER | response-format and parser contracts | Strict JSON index parser | adapted | same_dataset_different_protocol |
| PC-TRADITIONAL-RECOMMENDER-BASELINES-BM25-OR-CONTENTBM25-BASELINE | traditional recommender baselines | BM25 or ContentBM25 baseline | reconstructed | reconstructed_paper_like |
| PC-TRADITIONAL-RECOMMENDER-BASELINES-BPRMF-BASELINE | traditional recommender baselines | BPRMF baseline | adapted | reconstructed_paper_like |
| PC-TRADITIONAL-RECOMMENDER-BASELINES-GRU4REC-BASELINE | traditional recommender baselines | GRU4Rec baseline | adapted | reconstructed_paper_like |
| PC-TRADITIONAL-RECOMMENDER-BASELINES-LIGHTGCN-BASELINE | traditional recommender baselines | LightGCN baseline | adapted | reconstructed_paper_like |
| PC-TRADITIONAL-RECOMMENDER-BASELINES-MOSTPOP-BASELINE | traditional recommender baselines | MostPop baseline | reconstructed | reconstructed_paper_like |
| PC-TRADITIONAL-RECOMMENDER-BASELINES-SASREC-BASELINE | traditional recommender baselines | SASRec baseline | adapted | reconstructed_paper_like |
| PC-USER-COHORTS-AND-SAMPLING-COLD-START-OR-SHORT-HISTORY-GROUPING | user cohorts and sampling | Cold-start or short-history grouping | adapted | diagnostic_only |
| PC-USER-COHORTS-AND-SAMPLING-K-S-SAMPLING-VALIDITY-CHECK | user cohorts and sampling | K-S sampling validity check | not_executed | blocked_not_comparable |
| PC-USER-COHORTS-AND-SAMPLING-PAPER-1000-USER-SAMPLE-SCALE | user cohorts and sampling | Paper 1000-user sample scale | adapted | same_metric_different_cohort |
| PC-USER-COHORTS-AND-SAMPLING-PROMPT-COMPATIBLE-LOCAL-LLM-COHORTS | user cohorts and sampling | Prompt-compatible local LLM cohorts | adapted | same_metric_different_cohort |
| PC-UTILITY-METRICS-HR-5-METRIC-DEFINITION | utility metrics | HR@5 metric definition | reproduced_exact | paper_exact |
| PC-UTILITY-METRICS-NDCG-5-METRIC-DEFINITION | utility metrics | NDCG@5 metric definition | reproduced_exact | paper_exact |

## 10. Datasets

| component | primary_coverage_status | comparability_classification | rationale |
| --- | --- | --- | --- |
| Amazon Beauty dataset | reconstructed | reconstructed_paper_like | A paper-like local counterpart exists but official exact artifacts are missing or unproven. |
| Amazon Sports dataset | reconstructed | reconstructed_paper_like | A paper-like local counterpart exists but official exact artifacts are missing or unproven. |
| LastFM dataset | blocked | blocked_not_comparable | Required artifacts, models, APIs, or exact conditions are unavailable offline. |
| ML-1M dataset | reconstructed | reconstructed_paper_like | A paper-like local counterpart exists but official exact artifacts are missing or unproven. |

## 11. Tasks

| component | primary_coverage_status | comparability_classification | rationale |
| --- | --- | --- | --- |
| LLM ranking task | adapted | adapted_local_model_extension | A meaningful local counterpart exists with material changes. |
| Ranking top-5 utility cutoff | adapted | official_code_aligned_local_model_substitute | A meaningful local counterpart exists with material changes. |

| component | primary_coverage_status | comparability_classification | rationale |
| --- | --- | --- | --- |
| Legacy full-20 reranking diagnostic | not_executed | same_dataset_different_protocol | No valid local execution artifact exists for this component. |
| LLM reranking task | adapted | adapted_local_model_extension | A meaningful local counterpart exists with material changes. |

## 12. LLM Models

| component | primary_coverage_status | comparability_classification | rationale |
| --- | --- | --- | --- |
| Claude 3 Haiku | blocked | blocked_not_comparable | Required artifacts, models, APIs, or exact conditions are unavailable offline. |
| GPT-3.5 | blocked | blocked_not_comparable | Required artifacts, models, APIs, or exact conditions are unavailable offline. |
| GPT-4o | blocked | blocked_not_comparable | Required artifacts, models, APIs, or exact conditions are unavailable offline. |
| GPT-4o-mini | blocked | blocked_not_comparable | Required artifacts, models, APIs, or exact conditions are unavailable offline. |
| Llama3-70B-Instruct | blocked | blocked_not_comparable | Required artifacts, models, APIs, or exact conditions are unavailable offline. |
| Mistral 7B local substitute | adapted | adapted_local_model_extension | A meaningful local counterpart exists with material changes. |
| Qwen2 7B local substitute | adapted | adapted_local_model_extension | A meaningful local counterpart exists with material changes. |

## 13. Traditional Baselines

| component | primary_coverage_status | comparability_classification | rationale |
| --- | --- | --- | --- |
| BM25 or ContentBM25 baseline | reconstructed | reconstructed_paper_like | A paper-like local counterpart exists but official exact artifacts are missing or unproven. |
| BPRMF baseline | adapted | reconstructed_paper_like | A meaningful local counterpart exists with material changes. |
| GRU4Rec baseline | adapted | reconstructed_paper_like | A meaningful local counterpart exists with material changes. |
| LightGCN baseline | adapted | reconstructed_paper_like | A meaningful local counterpart exists with material changes. |
| MostPop baseline | reconstructed | reconstructed_paper_like | A paper-like local counterpart exists but official exact artifacts are missing or unproven. |
| SASRec baseline | adapted | reconstructed_paper_like | A meaningful local counterpart exists with material changes. |

## 14. Prompts And Parsers

| component | primary_coverage_status | comparability_classification | rationale |
| --- | --- | --- | --- |
| Generated-profile prompt family | blocked | blocked_not_comparable | Required artifacts, models, APIs, or exact conditions are unavailable offline. |
| History/no-history prompt variants | adapted | diagnostic_only | A meaningful local counterpart exists with material changes. |
| LLMRank-like prompt family | not_executed | blocked_not_comparable | No valid local execution artifact exists for this component. |
| LLMSRec or in-context prompt family | not_executed | blocked_not_comparable | No valid local execution artifact exists for this component. |
| OpenP5-like prompt family | adapted | same_dataset_different_protocol | A meaningful local counterpart exists with material changes. |

| component | primary_coverage_status | comparability_classification | rationale |
| --- | --- | --- | --- |
| Invalid-output handling policy | adapted | official_code_aligned_local_model_substitute | A meaningful local counterpart exists with material changes. |
| Original title-list parser and refinement | reconstructed | reconstructed_paper_like | A paper-like local counterpart exists but official exact artifacts are missing or unproven. |
| Strict JSON index parser | adapted | same_dataset_different_protocol | A meaningful local counterpart exists with material changes. |

## 15. Candidate Construction And Cohorts

| component | primary_coverage_status | comparability_classification | rationale |
| --- | --- | --- | --- |
| 1 positive plus 19 negatives | reconstructed | reconstructed_paper_like | A paper-like local counterpart exists but official exact artifacts are missing or unproven. |
| 20-item candidate pool | reconstructed | reconstructed_paper_like | A paper-like local counterpart exists but official exact artifacts are missing or unproven. |
| Candidate order variants | adapted | diagnostic_only | A meaningful local counterpart exists with material changes. |
| Ranking negative-candidate construction | reconstructed | reconstructed_paper_like | A paper-like local counterpart exists but official exact artifacts are missing or unproven. |
| Reranking candidate construction | adapted | same_dataset_different_protocol | A meaningful local counterpart exists with material changes. |

| component | primary_coverage_status | comparability_classification | rationale |
| --- | --- | --- | --- |
| Cold-start or short-history grouping | adapted | diagnostic_only | A meaningful local counterpart exists with material changes. |
| K-S sampling validity check | not_executed | blocked_not_comparable | No valid local execution artifact exists for this component. |
| Paper 1000-user sample scale | adapted | same_metric_different_cohort | A meaningful local counterpart exists with material changes. |
| Prompt-compatible local LLM cohorts | adapted | same_metric_different_cohort | A meaningful local counterpart exists with material changes. |

## 16. Utility Metrics

| component | primary_coverage_status | comparability_classification | rationale |
| --- | --- | --- | --- |
| HR@5 metric definition | reproduced_exact | paper_exact | The locally available artifact matches the component-level paper resource. |
| NDCG@5 metric definition | reproduced_exact | paper_exact | The locally available artifact matches the component-level paper resource. |

## 17. Beyond-Utility Metrics

| component | primary_coverage_status | comparability_classification | rationale |
| --- | --- | --- | --- |
| Average popularity metric | reconstructed | reconstructed_paper_like | A paper-like local counterpart exists but official exact artifacts are missing or unproven. |
| Diversity metric | blocked | blocked_not_comparable | Required artifacts, models, APIs, or exact conditions are unavailable offline. |
| Item coverage metric | reconstructed | reconstructed_paper_like | A paper-like local counterpart exists but official exact artifacts are missing or unproven. |
| Long-tail rate metric | reconstructed | reconstructed_paper_like | A paper-like local counterpart exists but official exact artifacts are missing or unproven. |
| Novelty or self-information metric | reconstructed | reconstructed_paper_like | A paper-like local counterpart exists but official exact artifacts are missing or unproven. |
| Popularity-bias/fairness metrics | blocked | blocked_not_comparable | Required artifacts, models, APIs, or exact conditions are unavailable offline. |

## 18. Robustness Analyses

| component | primary_coverage_status | comparability_classification | rationale |
| --- | --- | --- | --- |
| Cold-start short-history observation | adapted | diagnostic_only | A meaningful local counterpart exists with material changes. |
| History length sensitivity analysis | adapted | diagnostic_only | A meaningful local counterpart exists with material changes. |

| component | primary_coverage_status | comparability_classification | rationale |
| --- | --- | --- | --- |
| Candidate-first prompt condition | not_executed | blocked_not_comparable | No valid local execution artifact exists for this component. |
| Candidate position bias analysis | adapted | diagnostic_only | A meaningful local counterpart exists with material changes. |

| component | primary_coverage_status | comparability_classification | rationale |
| --- | --- | --- | --- |
| Profile-based recommendation effect | blocked | blocked_not_comparable | Required artifacts, models, APIs, or exact conditions are unavailable offline. |
| User profile generation | blocked | blocked_not_comparable | Required artifacts, models, APIs, or exact conditions are unavailable offline. |

## 19. Reliability Analyses

| component | primary_coverage_status | comparability_classification | rationale |
| --- | --- | --- | --- |
| Hallucination rate | adapted | adapted_local_model_extension | A meaningful local counterpart exists with material changes. |
| Output compliance and duplicate failures | adapted | adapted_local_model_extension | A meaningful local counterpart exists with material changes. |

## 20. Paper Tables And Result-Cell Coverage

Result cells inventoried: 11. Exact numeric comparisons supported: 0.

| paper_result_cell_id | local_result_id | comparability_status | numerical_comparison_permitted |
| --- | --- | --- | --- |
| PRC-0073DE5853 |  | same_task_different_model | false |
| PRC-195BD94B8B | local_counterpart_ndcg-at-5 | paper_value_not_locally_verified | false |
| PRC-3B1816C463 | local_counterpart_average-popularity-at-5 | paper_value_not_locally_verified | false |
| PRC-3C2F4B3551 |  | same_task_different_model | false |
| PRC-88936A0830 |  | same_task_different_model | false |
| PRC-941ACB79D8 | local_counterpart_hr-at-5 | paper_value_not_locally_verified | false |
| PRC-97892A414B | local_counterpart_long-tail-rate-at-5 | paper_value_not_locally_verified | false |
| PRC-B3B613E7D6 | local_counterpart_item-coverage-at-5 | paper_value_not_locally_verified | false |
| PRC-C9EE65DFB5 |  | same_task_different_model | false |
| PRC-D388B98E99 | local_counterpart_novelty-at-5 | paper_value_not_locally_verified | false |
| PRC-E92F01D27A |  | same_task_different_model | false |

## 21. Exact Reproduction Percentage

Strict exact reproduction micro percentage: 12.328767%.

## 22. Reconstructed-Or-Better Percentage

Reconstructed-or-better micro percentage: 34.246575%.

## 23. Maximum Defensible Paper-Component Coverage Percentage

Maximum defensible micro coverage: 68.493151%. This is not an exact paper-replication percentage.

## 24. Micro Versus Macro Interpretation

Macro exact: 12.5%. Macro reconstructed-or-better: 29.666667%. Macro maximum defensible: 68.011905%.

## 25. Extension Inventory

| extension_id | description | why_it_does_not_increase_reproduction_percentage |
| --- | --- | --- |
| EXT-001 | Local Qwen2 versus Mistral comparison | Qwen2 and Mistral are not exact paper LLM identities. |
| EXT-002 | Strict JSON index prompt and parser | The paper used title/free-text-style outputs and refinement code. |
| EXT-003 | Operational invalid-as-failure utility | It changes invalid-output treatment relative to unresolved paper policy. |
| EXT-004 | Candidate validity and duplicate-index structural metrics | It is a local diagnostic not a paper result cell. |
| EXT-005 | Phase 5D robustness consolidation | It consolidates evidence rather than executing paper experiments. |
| EXT-006 | Reconstructed neural baseline local runs | Training/data/cohort conditions are not proven paper-exact. |

## 26. Blocked And Unresolved Components

| blocker_id | component | blocker_type | recoverable_without_new_inference |
| --- | --- | --- | --- |
| BLK-5E-001 | Claude 3 Haiku | missing_artifact_or_external_dependency | false |
| BLK-5E-002 | GPT-3.5 | missing_artifact_or_external_dependency | false |
| BLK-5E-003 | GPT-4o | missing_artifact_or_external_dependency | false |
| BLK-5E-004 | GPT-4o-mini | missing_artifact_or_external_dependency | false |
| BLK-5E-005 | Llama3-70B-Instruct | missing_artifact_or_external_dependency | false |
| BLK-5E-006 | Diversity metric | missing_artifact_or_external_dependency | false |
| BLK-5E-007 | Popularity-bias/fairness metrics | missing_artifact_or_external_dependency | false |
| BLK-5E-008 | Candidate-first prompt condition | not_executed_locally | unknown |
| BLK-5E-009 | Official processed data artifacts | missing_artifact_or_external_dependency | false |
| BLK-5E-010 | LastFM dataset | missing_artifact_or_external_dependency | false |
| BLK-5E-011 | Profile-based recommendation effect | missing_artifact_or_external_dependency | false |
| BLK-5E-012 | User profile generation | missing_artifact_or_external_dependency | false |
| BLK-5E-013 | Seven qualitative observations | insufficient_or_conflicting_evidence | unknown |
| BLK-5E-014 | Beyond-utility result tables | insufficient_or_conflicting_evidence | unknown |
| BLK-5E-015 | Model comparison result groups | insufficient_or_conflicting_evidence | unknown |
| BLK-5E-016 | Utility result tables | insufficient_or_conflicting_evidence | unknown |
| BLK-5E-017 | Generated-profile prompt family | missing_artifact_or_external_dependency | false |
| BLK-5E-018 | LLMRank-like prompt family | not_executed_locally | unknown |
| BLK-5E-019 | LLMSRec or in-context prompt family | not_executed_locally | unknown |
| BLK-5E-020 | Hardware and API execution environment | missing_artifact_or_external_dependency | false |
| BLK-5E-021 | Paper model versions and provider settings | missing_artifact_or_external_dependency | false |
| BLK-5E-022 | Legacy full-20 reranking diagnostic | not_executed_locally | unknown |
| BLK-5E-023 | K-S sampling validity check | not_executed_locally | unknown |

## 27. Exact Numerical Comparison Limitations

Numerical similarity is not proof of replication. Same-metric/different-cohort, same-task/different-model, and same-dataset/different-protocol rows are non-exact comparisons.

## 28. Permitted Final Claims

| claim_id | claim | permitted_statement |
| --- | --- | --- |
| CLAIM-5E-001 | Strict exact component reproduction is 12.328767 percent. | Report as component-level exact coverage. |
| CLAIM-5E-002 | Maximum defensible component coverage is 68.493151 percent. | Report as exact+reconstructed+adapted local counterparts. |
| CLAIM-5E-003 | Qwen2 and Mistral are local substitutes/extensions. | Use them for local evidence only. |
| CLAIM-5E-004 | Blocked components remain in the applicable denominator. | State blocked rate transparently. |
| CLAIM-5E-005 | Exact numeric paper result-cell comparisons supported: 0. | Do not calculate replication deltas. |

## 29. Prohibited Final Claims

| claim_id | claim | prohibited_statement |
| --- | --- | --- |
| CLAIM-5E-006 | The project exactly reproduced the paper's LLM result tables. | Do not report exact replication deltas. |
| CLAIM-5E-007 | Legacy full-20 experiments are equivalent to canonical top-10 reranking. | Do not merge full-20 and top-10 evidence. |
| CLAIM-5E-008 | Generated profiles improved or harmed recommendations. | Do not claim profile benefit or harm. |

## 30. Threats To Validity

PDF numeric extraction is unavailable in the local environment; exact processed paper data are not identity-proven; closed/API models are not run; local Qwen2/Mistral are substitutes; Phase 5D diagnostics are not full-cohort evidence.

## 31. Phase 5E Decision

Validation pass/fail: 31/0. Decision: `maximum_defensible_paper_coverage_audit_complete`.

## 32. Phase 6 Handoff

| action_id | input_artifact | required_use | status |
| --- | --- | --- | --- |
| NEXT-5E-001 | results/tables/plan_d_phase_5e_component_coverage.csv | Use as the frozen component denominator and status registry before any Phase 6 synthesis. | ready |
| NEXT-5E-002 | results/tables/plan_d_phase_5e_result_cell_comparison.csv | Preserve exact numeric comparison blocks and avoid replication deltas. | ready |
| NEXT-5E-003 | results/tables/plan_d_phase_5e_blockers.csv | Carry residual blockers into Phase 6 planning without running experiments here. | ready |

## Appendix. Status Counts

| primary_coverage_status | count |
| --- | --- |
| reproduced_exact | 9 |
| reconstructed | 16 |
| adapted | 25 |
| extension_only | 0 |
| blocked | 14 |
| not_executed | 5 |
| unresolved | 4 |
| not_applicable | 0 |

## Appendix. Comparability Counts

| comparability_classification | count |
| --- | --- |
| paper_exact | 9 |
| official_code_aligned_local_model_substitute | 2 |
| adapted_local_model_extension | 6 |
| reconstructed_paper_like | 20 |
| same_metric_different_cohort | 2 |
| same_task_different_model | 0 |
| same_dataset_different_protocol | 5 |
| same_protocol_different_model | 1 |
| diagnostic_only | 6 |
| extension_not_in_paper | 0 |
| blocked_not_comparable | 18 |
| unresolved_comparability | 4 |
| not_applicable | 0 |
