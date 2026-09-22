# Plan E3A Baseline Alignment and No-Training Evaluation Audit

## Executive Summary

E3A is not full E3 baseline completion. E3A is a no-training alignment/evaluation audit over the E2 reconstructed paper-like manifests. It computes only metrics that are safely computable without recommender training, neural baseline training, LLM inference, Ollama calls, APIs, downloads, model pulls, or LastFM runs.

E3A performed no inference. E3A made no Ollama calls. E3A made no Qwen2/Mistral calls. E3A made no API calls. E3A performed no downloads. E3A performed no model pulls. E3A performed no training. E3A performed no neural-baseline training. E3A performed no LastFM run. E3A performed no Plan C modification. E3A performed no original_repo modification. E3A performed no destructive cleanup.

E3A does not complete full paper baseline reproduction. Candidate_order_reference and random_deterministic_reference are sanity checks, not paper baselines. MostPop is a paper baseline here because it is computed from valid available train interactions without model training. Neural baselines require aligned scores or future approved scoring/training. LastFM remains blocked/conditional. No full exact reproduction is claimed.

## E2 Manifest Overview

| manifest_id | dataset | task | rows | user_count | candidate_pool_size | positive_items_per_user | negative_items_per_user | classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plan_e_ml1m_1000user_ranking_manifest | ML-1M | ranking | 20000 | 1000 | 20 | 1 | 19 | reconstructed_paper_like |
| plan_e_ml1m_1000user_reranking_manifest | ML-1M | reranking | 20000 | 1000 | 20 | 1 | 19 | reconstructed_paper_like |
| plan_e_beauty_1000user_ranking_manifest | Amazon Beauty | ranking | 20000 | 1000 | 20 | 1 | 19 | reconstructed_paper_like |
| plan_e_beauty_1000user_reranking_manifest | Amazon Beauty | reranking | 20000 | 1000 | 20 | 1 | 19 | reconstructed_paper_like |
| plan_e_sports_1000user_ranking_manifest | Amazon Sports | ranking | 20000 | 1000 | 20 | 1 | 19 | reconstructed_paper_like |
| plan_e_sports_1000user_reranking_manifest | Amazon Sports | reranking | 20000 | 1000 | 20 | 1 | 19 | reconstructed_paper_like |

## Baseline Availability Overview

| dataset | task | model_or_method | can_compute_hr_ndcg_on_e2 | classification | exactness_class |
| --- | --- | --- | --- | --- | --- |
| ML-1M | ranking | BM25 | false | needs_scoring | needs_scoring |
| ML-1M | ranking | MostPop | true | no_training_evaluation | reconstructed_paper_like |
| ML-1M | ranking | BPRMF | false | needs_training_or_existing_scores | needs_scoring |
| ML-1M | ranking | GRU4Rec | false | needs_training_or_existing_scores | needs_scoring |
| ML-1M | ranking | SASRec | false | needs_training_or_existing_scores | needs_scoring |
| ML-1M | ranking | LightGCN | false | needs_training_or_existing_scores | needs_scoring |
| ML-1M | ranking | candidate_order_reference | true | no_training_evaluation | reconstructed_paper_like |
| ML-1M | ranking | random_deterministic_reference | true | no_training_evaluation | reconstructed_paper_like |
| ML-1M | reranking | BM25 | false | needs_scoring | needs_scoring |
| ML-1M | reranking | MostPop | true | no_training_evaluation | reconstructed_paper_like |
| ML-1M | reranking | BPRMF | false | needs_training_or_existing_scores | needs_scoring |
| ML-1M | reranking | GRU4Rec | false | needs_training_or_existing_scores | needs_scoring |
| ML-1M | reranking | SASRec | false | needs_training_or_existing_scores | needs_scoring |
| ML-1M | reranking | LightGCN | false | needs_training_or_existing_scores | needs_scoring |
| ML-1M | reranking | candidate_order_reference | true | no_training_evaluation | reconstructed_paper_like |
| ML-1M | reranking | random_deterministic_reference | true | no_training_evaluation | reconstructed_paper_like |
| Amazon Beauty | ranking | BM25 | false | needs_scoring | needs_scoring |
| Amazon Beauty | ranking | MostPop | true | no_training_evaluation | reconstructed_paper_like |
| Amazon Beauty | ranking | BPRMF | false | needs_training_or_existing_scores | needs_scoring |
| Amazon Beauty | ranking | GRU4Rec | false | needs_training_or_existing_scores | needs_scoring |
| Amazon Beauty | ranking | SASRec | false | needs_training_or_existing_scores | needs_scoring |
| Amazon Beauty | ranking | LightGCN | false | needs_training_or_existing_scores | needs_scoring |
| Amazon Beauty | ranking | candidate_order_reference | true | no_training_evaluation | reconstructed_paper_like |
| Amazon Beauty | ranking | random_deterministic_reference | true | no_training_evaluation | reconstructed_paper_like |
| Amazon Beauty | reranking | BM25 | false | needs_scoring | needs_scoring |
| Amazon Beauty | reranking | MostPop | true | no_training_evaluation | reconstructed_paper_like |
| Amazon Beauty | reranking | BPRMF | false | needs_training_or_existing_scores | needs_scoring |
| Amazon Beauty | reranking | GRU4Rec | false | needs_training_or_existing_scores | needs_scoring |
| Amazon Beauty | reranking | SASRec | false | needs_training_or_existing_scores | needs_scoring |
| Amazon Beauty | reranking | LightGCN | false | needs_training_or_existing_scores | needs_scoring |
| ... | ... | ... | ... | ... | ... |

## No-Training Metric Results

Metrics created in E3A: `MostPop; candidate_order_reference; random_deterministic_reference`.

| dataset | task | model_or_method | hr_at_5 | ndcg_at_5 | evaluated_users | classification |
| --- | --- | --- | --- | --- | --- | --- |
| ML-1M | ranking | candidate_order_reference | 0.263 | 0.158328 | 1000 | no_training_evaluation |
| ML-1M | ranking | random_deterministic_reference | 0.252 | 0.145389 | 1000 | no_training_evaluation |
| ML-1M | ranking | MostPop | 0.671 | 0.475902 | 1000 | no_training_evaluation |
| ML-1M | reranking | candidate_order_reference | 0.045 | 0.0261 | 1000 | no_training_evaluation |
| ML-1M | reranking | random_deterministic_reference | 0.255 | 0.153186 | 1000 | no_training_evaluation |
| ML-1M | reranking | MostPop | 0.671 | 0.475902 | 1000 | no_training_evaluation |
| Amazon Beauty | ranking | candidate_order_reference | 0.249 | 0.147349 | 1000 | no_training_evaluation |
| Amazon Beauty | ranking | random_deterministic_reference | 0.245 | 0.146252 | 1000 | no_training_evaluation |
| Amazon Beauty | ranking | MostPop | 0.48 | 0.325884 | 1000 | no_training_evaluation |
| Amazon Beauty | reranking | candidate_order_reference | 0.06 | 0.038028 | 1000 | no_training_evaluation |
| Amazon Beauty | reranking | random_deterministic_reference | 0.246 | 0.144499 | 1000 | no_training_evaluation |
| Amazon Beauty | reranking | MostPop | 0.48 | 0.325884 | 1000 | no_training_evaluation |
| Amazon Sports | ranking | candidate_order_reference | 0.237 | 0.141707 | 1000 | no_training_evaluation |
| Amazon Sports | ranking | random_deterministic_reference | 0.252 | 0.152059 | 1000 | no_training_evaluation |
| Amazon Sports | ranking | MostPop | 0.479 | 0.346301 | 1000 | no_training_evaluation |
| Amazon Sports | reranking | candidate_order_reference | 0.052 | 0.030752 | 1000 | no_training_evaluation |
| Amazon Sports | reranking | random_deterministic_reference | 0.263 | 0.155466 | 1000 | no_training_evaluation |
| Amazon Sports | reranking | MostPop | 0.479 | 0.346301 | 1000 | no_training_evaluation |

## Missing Baseline Analysis

| dataset | task | model_or_method | missing_reason | required_future_phase | classification |
| --- | --- | --- | --- | --- | --- |
| ML-1M | ranking | BM25 | No verified BM25 user-item score artifact aligned to E2 manifests. | E3B_no_training_lightweight_BM25_or_MostPop_fix | needs_scoring |
| ML-1M | ranking | BPRMF | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| ML-1M | ranking | GRU4Rec | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| ML-1M | ranking | SASRec | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| ML-1M | ranking | LightGCN | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| ML-1M | reranking | BM25 | No verified BM25 user-item score artifact aligned to E2 manifests. | E3B_no_training_lightweight_BM25_or_MostPop_fix | needs_scoring |
| ML-1M | reranking | BPRMF | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| ML-1M | reranking | GRU4Rec | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| ML-1M | reranking | SASRec | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| ML-1M | reranking | LightGCN | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| Amazon Beauty | ranking | BM25 | No verified BM25 user-item score artifact aligned to E2 manifests. | E3B_no_training_lightweight_BM25_or_MostPop_fix | needs_scoring |
| Amazon Beauty | ranking | BPRMF | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| Amazon Beauty | ranking | GRU4Rec | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| Amazon Beauty | ranking | SASRec | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| Amazon Beauty | ranking | LightGCN | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| Amazon Beauty | reranking | BM25 | No verified BM25 user-item score artifact aligned to E2 manifests. | E3B_no_training_lightweight_BM25_or_MostPop_fix | needs_scoring |
| Amazon Beauty | reranking | BPRMF | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| Amazon Beauty | reranking | GRU4Rec | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| Amazon Beauty | reranking | SASRec | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| Amazon Beauty | reranking | LightGCN | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| Amazon Sports | ranking | BM25 | No verified BM25 user-item score artifact aligned to E2 manifests. | E3B_no_training_lightweight_BM25_or_MostPop_fix | needs_scoring |
| Amazon Sports | ranking | BPRMF | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| Amazon Sports | ranking | GRU4Rec | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| Amazon Sports | ranking | SASRec | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| Amazon Sports | ranking | LightGCN | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| Amazon Sports | reranking | BM25 | No verified BM25 user-item score artifact aligned to E2 manifests. | E3B_no_training_lightweight_BM25_or_MostPop_fix | needs_scoring |
| Amazon Sports | reranking | BPRMF | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| Amazon Sports | reranking | GRU4Rec | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| Amazon Sports | reranking | SASRec | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |
| Amazon Sports | reranking | LightGCN | No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training. | E3C_neural_baseline_scoring_or_E3D_training_if_approved | needs_training_or_existing_scores |

## Validation Summary

Validation failures: `0`.

| check_name | passed | observed_value | expected_value | severity |
| --- | --- | --- | --- | --- |
| all_E2_manifests_present | true | 6 | 6 | critical |
| manifest_schema_valid | true | valid | valid | critical |
| candidate_pool_size_20 | true | 6 | 6 | critical |
| one_positive_per_user | true | 1 | 1 | critical |
| nineteen_negatives_per_user | true | 19 | 19 | critical |
| no_duplicate_items_per_user | true | no_duplicates | no_duplicates | critical |
| metric_formula_hr_at_5_available | true | implemented | implemented | critical |
| metric_formula_ndcg_at_5_available | true | implemented | implemented | critical |
| candidate_order_reference_metrics_created | true | 6 | 6 | critical |
| random_deterministic_reference_metrics_created | true | 6 | 6 | critical |
| all_paper_baselines_represented | true | BM25;MostPop;BPRMF;GRU4Rec;SASRec;LightGCN | BM25;MostPop;BPRMF;GRU4Rec;SASRec;LightGCN | critical |
| no_neural_training_performed | true | no_training | no_training | critical |
| no_llm_inference_performed | true | no_llm_inference | no_llm_inference | critical |
| no_ollama_call_performed | true | no_ollama_call | no_ollama_call | critical |
| no_api_call_performed | true | no_api_call | no_api_call | critical |
| no_download_performed | true | no_download | no_download | critical |
| no_lastfm_run_performed | true | no_lastfm_run | no_lastfm_run | critical |
| no_raw_outputs_created | true | 7 | unchanged_existing_only | critical |
| no_original_repo_output | true | clean | clean | critical |
| no_plan_c_modification | true | clean | clean | critical |
| no_exact_reproduction_overclaim | true | no_exact_claim | no_exact_claim | critical |

## Exactness and Claim-Safety Notes

- E3A is a no-training audit, not exact reproduction.
- E3A computes only metrics safely computable without training or new inference.
- E3A does not complete full paper baseline reproduction.
- Candidate-order and deterministic-random references are sanity checks, not paper baselines.
- MostPop uses reconstructed train interactions and E2 manifests, so it is reconstructed_paper_like/no_training_evaluation.
- BM25, BPRMF, GRU4Rec, SASRec, and LightGCN need aligned E2 scores or future approved scoring/training before E2 HR@5/NDCG@5 can be reported.
- D5 local LLM diagnostics are not paper baseline scores and are not mixed into E3A baseline metrics.
- No full exact reproduction is claimed.

## Recommended Next Phase

| action_id | phase | priority | action | risk_level |
| --- | --- | --- | --- | --- |
| E3A_A01 | E3B | 1 | decide whether to implement lightweight BM25 scoring on E2 manifests | medium |
| E3A_A02 | E3C | 2 | decide whether to run neural baseline scoring on E2 manifests | high |
| E3A_A03 | E3B | 3 | prepare E3B for paper-like baseline scoring | medium |
| E3A_A04 | E5 | 4 | keep LLM inference for E5 | high |
| E3A_A05 | E6 | 5 | keep API audit for E6 | high |
| E3A_A06 | blocked | 6 | keep LastFM conditional | high |
| E3A_A07 | reporting | 7 | do not mix D5 LLM diagnostic metrics with baseline metrics | medium |
| E3A_A08 | E3B/E4/E5 | 8 | preserve E2 manifests as the common evaluation substrate | low |
