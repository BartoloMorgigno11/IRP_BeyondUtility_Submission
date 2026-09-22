# Plan D Amazon Extended BPRMF/LightGCN Summary

## Purpose

This report aggregates existing Plan D Amazon Beauty and Amazon Sports extended_subset metric CSVs for BPRMF
and LightGCN. It is a reconstructed Plan D extended_subset evaluation, not an exact paper reproduction, because
the official preprocessed Amazon artifacts were unavailable. The CSV summary preserves exact values read from
the metric files; the markdown tables format values to 4 decimals for readability.

## Common Setup

- Datasets: amazon_beauty and amazon_sports
- Scope: extended_subset
- max_users: 1000
- max_train_interactions: 100000 cap, with actual interactions lower because these Amazon subsets are sparse
- epochs: 5
- Evaluation: candidate-set ranking at k=5
- Device: CUDA where available

## Amazon Beauty Results

| method | hr_at_5 | ndcg_at_5 | final_train_loss | best_train_loss | num_users_evaluated | num_train_interactions | runtime_seconds | device | epochs_completed | dataset_method_hr_rank | dataset_method_ndcg_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgcn | 0.1230 | 0.0876 | 0.6908 | 0.6908 | 1000 | 7495 | 8.9191 | cuda | 5 | 1 | 1 |
| bprmf | 0.0600 | 0.0365 | 0.6916 | 0.6916 | 1000 | 7495 | 8.7254 | cuda | 5 | 2 | 2 |

## Amazon Sports Results

| method | hr_at_5 | ndcg_at_5 | final_train_loss | best_train_loss | num_users_evaluated | num_train_interactions | runtime_seconds | device | epochs_completed | dataset_method_hr_rank | dataset_method_ndcg_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgcn | 0.0820 | 0.0504 | 0.6906 | 0.6906 | 1000 | 7292 | 24.5627 | cuda | 5 | 1 | 1 |
| bprmf | 0.0550 | 0.0311 | 0.6915 | 0.6915 | 1000 | 7292 | 11.7197 | cuda | 5 | 2 | 2 |

## Diagnostic vs Extended

| dataset | method | metric | diagnostic_value | extended_value | absolute_delta | relative_delta | diagnostic_num_users_evaluated | extended_num_users_evaluated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| amazon_beauty | bprmf | hit_rate | 0.0600 | 0.0600 | 0.0000 | 0.0000 | 100 | 1000 |
| amazon_beauty | bprmf | ndcg | 0.0318 | 0.0365 | 0.0047 | 0.1473 | 100 | 1000 |
| amazon_beauty | bprmf | final_train_loss | 0.6931 | 0.6916 | -0.0015 | -0.0021 | 100 | 1000 |
| amazon_beauty | bprmf | best_train_loss | 0.6931 | 0.6916 | -0.0015 | -0.0021 | 100 | 1000 |
| amazon_beauty | lightgcn | hit_rate | 0.0600 | 0.1230 | 0.0630 | 1.0500 | 100 | 1000 |
| amazon_beauty | lightgcn | ndcg | 0.0427 | 0.0876 | 0.0449 | 1.0499 | 100 | 1000 |
| amazon_beauty | lightgcn | final_train_loss | 0.6927 | 0.6908 | -0.0019 | -0.0027 | 100 | 1000 |
| amazon_beauty | lightgcn | best_train_loss | 0.6927 | 0.6908 | -0.0019 | -0.0027 | 100 | 1000 |
| amazon_sports | bprmf | hit_rate | 0.0700 | 0.0550 | -0.0150 | -0.2143 | 100 | 1000 |
| amazon_sports | bprmf | ndcg | 0.0394 | 0.0311 | -0.0083 | -0.2106 | 100 | 1000 |
| amazon_sports | bprmf | final_train_loss | 0.6931 | 0.6915 | -0.0016 | -0.0024 | 100 | 1000 |
| amazon_sports | bprmf | best_train_loss | 0.6931 | 0.6915 | -0.0016 | -0.0024 | 100 | 1000 |
| amazon_sports | lightgcn | hit_rate | 0.0800 | 0.0820 | 0.0020 | 0.0250 | 100 | 1000 |
| amazon_sports | lightgcn | ndcg | 0.0559 | 0.0504 | -0.0055 | -0.0990 | 100 | 1000 |
| amazon_sports | lightgcn | final_train_loss | 0.6926 | 0.6906 | -0.0021 | -0.0030 | 100 | 1000 |
| amazon_sports | lightgcn | best_train_loss | 0.6926 | 0.6906 | -0.0021 | -0.0030 | 100 | 1000 |

## Interpretation

- Amazon Beauty best HR@5: LightGCN (0.1230).
- Amazon Beauty best NDCG@5: LightGCN (0.0876).
- Amazon Sports best HR@5: LightGCN (0.0820).
- Amazon Sports best NDCG@5: LightGCN (0.0504).
- LightGCN outperforms BPRMF on both Amazon datasets by HR@5 and NDCG@5 in these source metrics.
- The Amazon extended subsets use 7292, 7495 train interactions, well below the 100000 cap.
- Runtime ranges from 8.7254 seconds (Amazon Beauty BPRMF) to 24.5627 seconds (Amazon Sports LightGCN).
- Runtime comparisons are descriptive only because the models use different graph or embedding workloads.

## Limitations

- The Amazon datasets are reconstructed, not official preprocessed paper artifacts.
- This is an extended_subset evaluation, not a full reproduction.
- Only BPRMF and LightGCN are included here; SASRec and GRU4Rec Amazon extended runs remain future work.
- Hyperparameters are controlled and safe rather than exhaustively tuned.

## Recommended Next Steps

1. Run SASRec Amazon Beauty extended_subset.
2. Run SASRec Amazon Sports extended_subset.
3. Run GRU4Rec Amazon Beauty and Amazon Sports extended_subset if time permits.
4. Create a global extended summary across all datasets and methods.
