# Plan E3B BM25/BM25-like Scoring

## Executive Summary

E3B is not full E3 baseline completion. E3B is no-training local text scoring over the E2 reconstructed paper-like manifests. E3B computes BM25-like HR@5/NDCG@5 only where local item text and train/history interactions are available.

E3B performed no inference. E3B made no Ollama calls. E3B made no Qwen2/Mistral calls. E3B made no API calls. E3B performed no downloads. E3B performed no model pulls. E3B performed no training. E3B performed no neural-baseline training. E3B performed no LastFM run. E3B performed no Plan C modification. E3B performed no original_repo modification. E3B performed no destructive cleanup.

E3B does not run LLM inference. E3B does not run neural baseline training. E3B does not complete full paper baseline reproduction. BM25 is exact only if exact paper BM25 conditions are verified. Because the exact paper BM25 configuration, corpus, preprocessing, query construction, and candidate scoring files are not verified here, computed rows are classified as BM25_like_reconstructed / reconstructed_BM25_like and no_training_text_scoring. Candidate_order_reference, random_deterministic_reference, and MostPop remain from E3A. Neural baselines remain for E3C/E3D. LastFM remains blocked/conditional. No full exact reproduction is claimed.

## E2 Manifest Overview

E3B read the six E2 manifests for ML-1M, Amazon Beauty, and Amazon Sports. It did not regenerate or modify them.

## BM25 Source Discovery Summary

| path | dataset | artifact_type | detected_item_column | detected_text_columns | detected_user_column | can_support_item_documents | can_support_user_queries |
| --- | --- | --- | --- | --- | --- | --- | --- |
| data/artifact_manifests/plan_c_artifact_manifest.csv | all | e2_or_plan_manifest |  |  |  | false | false |
| data/artifact_manifests/plan_c_artifact_manifest_template.csv | all | e2_or_plan_manifest |  |  |  | false | false |
| data/external/plan_c/.gitkeep | all | artifact |  |  |  | false | false |
| data/external/plan_c/hetrec2011-lastfm-2k.zip | LastFM | artifact |  |  |  | false | false |
| data/external/plan_c/meta_Beauty.json.gz | Amazon Beauty | artifact |  |  |  | false | false |
| data/external/plan_c/meta_Sports_and_Outdoors.json.gz | Amazon Sports | artifact |  |  |  | false | false |
| data/external/plan_c/ml-1m.zip | ML-1M | artifact |  |  |  | false | false |
| data/external/plan_c/reviews_Beauty_5.json.gz | Amazon Beauty | artifact |  |  |  | false | false |
| data/external/plan_c/reviews_Sports_and_Outdoors_5.json.gz | Amazon Sports | artifact |  |  |  | false | false |
| data/processed/ml1m/candidates.csv | ML-1M | item_metadata | candidate_item_id | candidate_item_title | user_id | true | false |
| data/processed/ml1m/catalog.csv | ML-1M | item_metadata | item_id | normalized_text;title;genres |  | true | false |
| data/processed/ml1m/interactions.csv | ML-1M | user_history_interactions | item_id |  | user_id | false | true |
| data/reconstructed/plan_c/.gitkeep | all | artifact |  |  |  | false | false |
| data/reconstructed/plan_c/amazon_beauty/candidates.csv | Amazon Beauty | item_metadata | candidate_item_id | candidate_item_title | user_id | true | false |
| data/reconstructed/plan_c/amazon_beauty/catalog.csv | Amazon Beauty | item_metadata | item_id | normalized_text;title;description;categories;brand |  | true | false |
| data/reconstructed/plan_c/amazon_beauty/datamaps.json | Amazon Beauty | artifact |  |  |  | false | false |
| data/reconstructed/plan_c/amazon_beauty/ground_truth.csv | Amazon Beauty | artifact | ground_truth_item_id |  | user_id | false | false |
| data/reconstructed/plan_c/amazon_beauty/interactions.csv | Amazon Beauty | user_history_interactions | item_id |  | user_id | false | true |
| data/reconstructed/plan_c/amazon_beauty/meta.json | Amazon Beauty | artifact |  |  |  | false | false |
| data/reconstructed/plan_c/amazon_beauty/negative_samples.txt | Amazon Beauty | artifact |  |  |  | false | false |
| ... | ... | ... | ... | ... | ... | ... | ... |

## Query/Corpus Construction Summary

Queries are built from local train/history interaction item text, with all E2 candidate items for the user excluded from the query when constructing the history text. Item documents use local reconstructed catalog text, preferring normalized_text when available. This is a reconstructed BM25-like setup, not verified exact paper BM25.

| dataset | task | users_with_queries | users_missing_queries | candidates_with_documents | candidates_missing_documents | can_score | classification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ML-1M | ranking | 1000 | 0 | 20000 | 0 | true | no_training_text_scoring |
| ML-1M | reranking | 1000 | 0 | 20000 | 0 | true | no_training_text_scoring |
| Amazon Beauty | ranking | 1000 | 0 | 20000 | 0 | true | no_training_text_scoring |
| Amazon Beauty | reranking | 1000 | 0 | 20000 | 0 | true | no_training_text_scoring |
| Amazon Sports | ranking | 1000 | 0 | 20000 | 0 | true | no_training_text_scoring |
| Amazon Sports | reranking | 1000 | 0 | 20000 | 0 | true | no_training_text_scoring |

## BM25 Scoring Summary

Score rows created: `120000`.

| dataset | task | user_id | item_id | bm25_score | bm25_rank | classification |
| --- | --- | --- | --- | --- | --- | --- |
| ML-1M | ranking | 1 | 1912 | 27.396759 | 3 | no_training_text_scoring |
| ML-1M | ranking | 1 | 3389 | 23.06573815 | 5 | no_training_text_scoring |
| ML-1M | ranking | 1 | 3266 | 20.54906568 | 6 | no_training_text_scoring |
| ML-1M | ranking | 1 | 1194 | 4.51521186 | 20 | no_training_text_scoring |
| ML-1M | ranking | 1 | 147 | 14.16360172 | 12 | no_training_text_scoring |
| ML-1M | ranking | 1 | 478 | 4.87413641 | 19 | no_training_text_scoring |
| ML-1M | ranking | 1 | 3208 | 14.74042963 | 11 | no_training_text_scoring |
| ML-1M | ranking | 1 | 1650 | 9.12693995 | 16 | no_training_text_scoring |
| ML-1M | ranking | 1 | 318 | 8.60717019 | 17 | no_training_text_scoring |
| ML-1M | ranking | 1 | 48 | 52.72022749 | 1 | no_training_text_scoring |
| ML-1M | ranking | 1 | 3538 | 11.18520608 | 15 | no_training_text_scoring |
| ML-1M | ranking | 1 | 3078 | 11.44333665 | 14 | no_training_text_scoring |
| ... | ... | ... | ... | ... | ... | ... |

## BM25 HR@5/NDCG@5 Results

| dataset | task | model_or_method | hr_at_5 | ndcg_at_5 | evaluated_users | classification |
| --- | --- | --- | --- | --- | --- | --- |
| ML-1M | ranking | BM25_like_reconstructed | 0.522 | 0.364931 | 1000 | no_training_text_scoring |
| ML-1M | reranking | BM25_like_reconstructed | 0.522 | 0.364931 | 1000 | no_training_text_scoring |
| Amazon Beauty | ranking | BM25_like_reconstructed | 0.557 | 0.417612 | 1000 | no_training_text_scoring |
| Amazon Beauty | reranking | BM25_like_reconstructed | 0.557 | 0.417612 | 1000 | no_training_text_scoring |
| Amazon Sports | ranking | BM25_like_reconstructed | 0.515 | 0.373698 | 1000 | no_training_text_scoring |
| Amazon Sports | reranking | BM25_like_reconstructed | 0.515 | 0.373698 | 1000 | no_training_text_scoring |

## Blocked/Missing Dataset Analysis

| dataset | task | blocked_or_missing | reason | classification |
| --- | --- | --- | --- | --- |
| LastFM | ranking | blocked/conditional | LastFM has no valid E2 paper-like candidate manifest and remains unsuitable for comparable utility evaluation. | blocked/conditional |
| LastFM | reranking | blocked/conditional | LastFM has no valid E2 paper-like candidate manifest and remains unsuitable for comparable utility evaluation. | blocked/conditional |

## Validation Summary

Validation failures: `0`.

| check_name | passed | observed_value | expected_value | severity |
| --- | --- | --- | --- | --- |
| all_E2_manifests_present | true | 6 | 6 | critical |
| item_text_sources_discovered_or_blocked | true | present_or_blocked | present_or_blocked | critical |
| user_query_sources_discovered_or_blocked | true | present_or_blocked | present_or_blocked | critical |
| no_training_performed | true | no_training | no_training | critical |
| no_neural_training_performed | true | no_neural_training | no_neural_training | critical |
| no_llm_inference_performed | true | no_llm_inference | no_llm_inference | critical |
| no_ollama_call_performed | true | no_ollama_call | no_ollama_call | critical |
| no_api_call_performed | true | no_api_call | no_api_call | critical |
| no_download_performed | true | no_download | no_download | critical |
| no_lastfm_run_performed | true | no_lastfm_run | no_lastfm_run | critical |
| no_raw_outputs_created | true | 7 | unchanged_existing_only | critical |
| no_original_repo_output | true | clean | clean | critical |
| no_plan_c_modification | true | clean | clean | critical |
| bm25_scores_schema_valid_if_created | true | valid | valid | critical |
| bm25_metrics_schema_valid | true | valid | valid | critical |
| hr_at_5_between_0_and_1 | true | 0<=hr<=1 | 0<=hr<=1 | critical |
| ndcg_at_5_between_0_and_1 | true | 0<=ndcg<=1 | 0<=ndcg<=1 | critical |
| no_exact_reproduction_overclaim | true | no_exact_claim | no_exact_claim | critical |

## Exactness and Claim-Safety Notes

- E3B is no-training local text scoring, not exact reproduction.
- E3B does not run LLM inference and does not run neural baseline training.
- E3B does not complete full paper baseline reproduction.
- BM25 is exact only if exact paper BM25 conditions are verified.
- E3B computed rows are BM25_like_reconstructed / reconstructed_BM25_like unless exact original paper BM25 artifacts are later verified.
- Candidate_order_reference, random_deterministic_reference, and MostPop remain from E3A.
- Neural baselines remain for E3C/E3D.
- LastFM remains blocked/conditional.
- No full exact reproduction is claimed.

## Recommended Next Phase

| action_id | phase | priority | action | risk_level |
| --- | --- | --- | --- | --- |
| E3B_A01 | E3_summary | 1 | integrate BM25-like metrics into the later E3 summary | medium |
| E3B_A02 | E3B_source_fix | 2 | document exact missing metadata/history source for any blocked dataset | medium |
| E3B_A03 | E3C | 3 | decide whether to proceed to neural baseline scoring | high |
| E3B_A04 | E3C/E4/E5 | 4 | keep E2 manifests as the common evaluation substrate | low |
| E3B_A05 | E5 | 5 | keep LLM inference for E5 | high |
| E3B_A06 | E6 | 6 | keep API audit for E6 | high |
| E3B_A07 | blocked | 7 | keep LastFM conditional | high |
