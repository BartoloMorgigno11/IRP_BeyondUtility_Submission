# Plan D ML-1M Extended Neural Baselines Summary

## Purpose

This report aggregates existing Plan D ML-1M extended_subset metric CSVs for BPRMF, LightGCN, SASRec,
and GRU4Rec.
It is not an exact reproduction of the paper because it uses reconstructed Plan C data rather than official
preprocessed artifacts. The CSV summary preserves exact values read from the metric files; the markdown tables
format values to 4 decimals for readability.

## Common Setup

- Dataset: ml1m
- Scope: extended_subset
- max_users: 1000
- max_train_interactions: 100000
- epochs: 5
- Evaluation: candidate-set ranking at k=5
- Device: CUDA where available

## Extended Subset Results

| method | hr_at_5 | ndcg_at_5 | final_train_loss | best_train_loss | num_users_evaluated | num_train_interactions | runtime_seconds | device | epochs_completed | hr_rank | ndcg_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sasrec | 0.3333 | 0.2317 | 0.1756 | 0.1756 | 672 | 100000 | 46.1953 | cuda | 5 | 1 | 1 |
| gru4rec | 0.2932 | 0.1843 | 0.3075 | 0.3075 | 672 | 100000 | 15.2164 | cuda | 5 | 2 | 2 |
| lightgcn | 0.2590 | 0.1706 | 0.4225 | 0.4225 | 1000 | 100000 | 9.5907 | cuda | 5 | 3 | 3 |
| bprmf | 0.2430 | 0.1594 | 0.4722 | 0.4722 | 1000 | 100000 | 8.9641 | cuda | 5 | 4 | 4 |

## Diagnostic vs Extended

| method | metric | diagnostic_value | extended_value | absolute_delta | relative_delta | diagnostic_num_users_evaluated | extended_num_users_evaluated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bprmf | hit_rate | 0.0300 | 0.2430 | 0.2130 | 7.1000 | 100 | 1000 |
| bprmf | ndcg | 0.0140 | 0.1594 | 0.1453 | 10.3460 | 100 | 1000 |
| bprmf | final_train_loss | 0.6931 | 0.4722 | -0.2209 | -0.3188 | 100 | 1000 |
| bprmf | best_train_loss | 0.6931 | 0.4722 | -0.2209 | -0.3188 | 100 | 1000 |
| lightgcn | hit_rate | 0.0400 | 0.2590 | 0.2190 | 5.4750 | 100 | 1000 |
| lightgcn | ndcg | 0.0215 | 0.1706 | 0.1491 | 6.9382 | 100 | 1000 |
| lightgcn | final_train_loss | 0.6931 | 0.4225 | -0.2706 | -0.3904 | 100 | 1000 |
| lightgcn | best_train_loss | 0.6931 | 0.4225 | -0.2706 | -0.3904 | 100 | 1000 |
| sasrec | hit_rate | 0.0833 | 0.3333 | 0.2500 | 3.0000 | 36 | 672 |
| sasrec | ndcg | 0.0422 | 0.2317 | 0.1895 | 4.4953 | 36 | 672 |
| sasrec | final_train_loss | 0.6722 | 0.1756 | -0.4965 | -0.7387 | 36 | 672 |
| sasrec | best_train_loss | 0.6722 | 0.1756 | -0.4965 | -0.7387 | 36 | 672 |
| gru4rec | hit_rate | 0.1389 | 0.2932 | 0.1543 | 1.1107 | 36 | 672 |
| gru4rec | ndcg | 0.0887 | 0.1843 | 0.0956 | 1.0778 | 36 | 672 |
| gru4rec | final_train_loss | 0.6883 | 0.3075 | -0.3808 | -0.5532 | 36 | 672 |
| gru4rec | best_train_loss | 0.6883 | 0.3075 | -0.3808 | -0.5532 | 36 | 672 |

## Interpretation

- Best HR@5: SASRec (0.3333).
- Best NDCG@5: SASRec (0.2317).
- HR@5 ranking: SASRec > GRU4Rec > LightGCN > BPRMF.
- NDCG@5 ranking: SASRec > GRU4Rec > LightGCN > BPRMF.
- Fastest runtime: bprmf (8.9641 seconds).
- SASRec evaluated 672 users rather than 1000, because the sequential evaluation requires usable non-empty training histories and candidate rows.
- GRU4Rec evaluated 672 users rather than 1000, because the sequential evaluation requires usable non-empty training histories and candidate rows.
- Runtime comparisons are descriptive only because the architectures perform different amounts of work per epoch.

## Limitations

- The dataset is reconstructed, not the official preprocessed paper artifact.
- This is an extended_subset run, not a full paper reproduction.
- Hyperparameters are controlled and safe rather than exhaustively tuned.
- SASRec required deterministic CUDA setup through CUBLAS workspace and attention-kernel configuration.

## Recommended Next Steps

1. Run Amazon Beauty and Amazon Sports extended_subset experiments for BPRMF, LightGCN, SASRec, and GRU4Rec.
2. Attempt ML-1M full runs one model at a time if extended results remain stable.
3. Decide the LLM backend before starting LLM ranking or reranking work.
