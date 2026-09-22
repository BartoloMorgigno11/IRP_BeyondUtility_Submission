# Plan D Global Extended Neural Baseline Summary

## Purpose

This report aggregates reconstructed Plan D extended_subset results across the utility-evaluable datasets:
ML-1M, Amazon Beauty, and Amazon Sports.

## Paper Dataset Coverage

The original paper includes four datasets:

- Amazon Beauty
- Amazon Sports and Outdoors
- MovieLens-1M
- LastFM

## LastFM Status and Exclusion Rationale

LastFM is present in the broader project as extension_only/problematic, but it is excluded from HR@5/NDCG@5
ranking because the reconstructed project does not have a valid comparable chronological split/candidate/
ground-truth setup. No fake LastFM utility metrics were generated.

## Scope

This is not an exact paper reproduction because the official preprocessed artifacts and exact paper candidate
pools are unavailable. The results are controlled Plan D extended_subset evaluations on reconstructed inputs.

## Common Setup

- Scope: extended_subset
- k: 5
- max_users: 1000
- max_train_interactions: 100000 cap
- epochs: 5
- Evaluation: candidate-set ranking
- Device: CUDA where available
- Hyperparameters: controlled safe settings, not exhaustive tuning

## Global Summary

| dataset | method | hr_at_5 | ndcg_at_5 | final_train_loss | best_train_loss | num_users_evaluated | num_train_interactions | runtime_seconds | device | dataset_method_hr_rank | dataset_method_ndcg_rank | global_hr_rank | global_ndcg_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml1m | sasrec | 0.3333 | 0.2317 | 0.1756 | 0.1756 | 672 | 100000 | 46.1953 | cuda | 1 | 1 | 1 | 1 |
| ml1m | gru4rec | 0.2932 | 0.1843 | 0.3075 | 0.3075 | 672 | 100000 | 15.2164 | cuda | 2 | 2 | 2 | 2 |
| ml1m | lightgcn | 0.2590 | 0.1706 | 0.4225 | 0.4225 | 1000 | 100000 | 9.5907 | cuda | 3 | 3 | 3 | 3 |
| ml1m | bprmf | 0.2430 | 0.1594 | 0.4722 | 0.4722 | 1000 | 100000 | 8.9641 | cuda | 4 | 4 | 4 | 4 |
| amazon_beauty | gru4rec | 0.1590 | 0.1054 | 0.3646 | 0.3646 | 1000 | 7495 | 13.2107 | cuda | 1 | 1 | 5 | 5 |
| amazon_beauty | lightgcn | 0.1230 | 0.0876 | 0.6908 | 0.6908 | 1000 | 7495 | 8.9191 | cuda | 2 | 2 | 7 | 7 |
| amazon_beauty | bprmf | 0.0600 | 0.0365 | 0.6916 | 0.6916 | 1000 | 7495 | 8.7254 | cuda | 3 | 3 | 9 | 9 |
| amazon_beauty | sasrec | 0.0320 | 0.0195 | 0.4107 | 0.4107 | 1000 | 7495 | 18.5524 | cuda | 4 | 4 | 11 | 11 |
| amazon_sports | gru4rec | 0.1470 | 0.1006 | 0.3365 | 0.3365 | 1000 | 7292 | 16.0358 | cuda | 1 | 1 | 6 | 6 |
| amazon_sports | lightgcn | 0.0820 | 0.0504 | 0.6906 | 0.6906 | 1000 | 7292 | 24.5627 | cuda | 2 | 2 | 8 | 8 |
| amazon_sports | bprmf | 0.0550 | 0.0311 | 0.6915 | 0.6915 | 1000 | 7292 | 11.7197 | cuda | 3 | 3 | 10 | 10 |
| amazon_sports | sasrec | 0.0300 | 0.0169 | 0.4079 | 0.4079 | 1000 | 7292 | 23.0519 | cuda | 4 | 4 | 12 | 12 |

## Dataset-Level Findings

- ML-1M HR@5 ranking: SASRec > GRU4Rec > LightGCN > BPRMF.
- ML-1M NDCG@5 ranking: SASRec > GRU4Rec > LightGCN > BPRMF.
- Amazon Beauty HR@5 ranking: GRU4Rec > LightGCN > BPRMF > SASRec.
- Amazon Beauty NDCG@5 ranking: GRU4Rec > LightGCN > BPRMF > SASRec.
- Amazon Sports HR@5 ranking: GRU4Rec > LightGCN > BPRMF > SASRec.
- Amazon Sports NDCG@5 ranking: GRU4Rec > LightGCN > BPRMF > SASRec.

## Interpretation

- Global HR@5 top 5: ML-1M SASRec > ML-1M GRU4Rec > ML-1M LightGCN > ML-1M BPRMF > Amazon Beauty GRU4Rec.
- SASRec is strongest on ML-1M extended_subset by HR@5 and NDCG@5.
- GRU4Rec is strongest on Amazon Beauty extended_subset by HR@5 and NDCG@5.
- GRU4Rec is strongest on Amazon Sports extended_subset by HR@5 and NDCG@5.
- LightGCN consistently beats BPRMF on both Amazon datasets by HR@5 and NDCG@5.
- SASRec performs poorly on the Amazon extended subsets despite strong ML-1M performance.
- The reconstructed Amazon extended subsets are sparse, with 7292, 7495 train interactions for 1000 users, which likely affects sequential transformer performance.
- ML-1M SASRec evaluated 672 users rather than 1000 because of valid sequence/history constraints.
- ML-1M GRU4Rec evaluated 672 users rather than 1000 because of valid sequence/history constraints.

## Diagnostic vs Extended

The diagnostic-vs-extended comparison table contains 48 rows.
All utility-evaluable dataset/method pairs with extended metrics also have diagnostic comparisons.

## Dataset/Model Coverage

The coverage table contains 16 rows across four datasets and four methods.
LastFM rows are included with extension_only_no_valid_utility_evaluation status unless real comparable
metric files exist.

## Limitations

- The datasets are reconstructed, not official preprocessed paper artifacts.
- These are extended_subset results, not full reproduction results.
- LastFM is excluded from utility ranking for methodological reasons.
- Candidate-set evaluation depends on reconstructed candidates.
- Hyperparameters are controlled but not exhaustively tuned.
- LLM ranking/reranking remains blocked until a backend/API is approved.

## Recommended Next Steps

1. Update the reproducibility matrix with completed extended baselines and the LastFM coverage note.
2. Create the final Plan D baseline report section.
3. Decide the LLM backend: Ollama, local transformers, or API.
4. If an LLM backend is available, run LLM ranking diagnostic.
5. Otherwise document the LLM blocker and proceed to the final report.
