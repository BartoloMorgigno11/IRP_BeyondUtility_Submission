# F1BZ Final Traditional Execution

Decision token: `F1BZ_FINAL_TRADITIONAL_EXECUTION_VALIDATED`

## Precondition
- HEAD: `5f29945d1cfe5e6cce5b6cc0522db6f0efe8a976`; expected F1BY checkpoint: `5f29945d1cfe5e6cce5b6cc0522db6f0efe8a976`.
- Working tree was clean before F1BZ edits; F1BZ does not commit.

## Resource Preflight
- Python `3.12.10`; Torch `2.5.1+cu121`; CUDA `true`; GPU `NVIDIA GeForce RTX 4060 Laptop GPU`.

## Final Run Accounting
- Final runs: `100`; completed `100`; failed `0`; retries `0`.
- Final-test metrics computed after winner freeze; final-test targets used in tuning: `false`.

## Five-Seed Aggregates
- Utility aggregate rows: `60`; beyond-utility aggregate rows: `280`.

## Traditional 35-Row Completeness
- Rows: `35`; completed `35`.

## First-Stage Rankings
- Ranking rows: `97640`; source policy `F1BZ_FIVE_SEED_MEAN_SCORE_THEN_FROZEN_MODEL_ORDER`.

## Candidate Recall
- Dataset recall rows: `5`.

## Reranking Readiness
- READY_TO_INFER rows: `30`; future LLM requests unlocked `29292`; executed `0`.

## Failure / Retry Audit
- Review rows: `0`.

## LLM Inference Counter
- LLM inference requests `0`; Ollama recommendation calls `0`; external API calls `0`; network/download calls `0`.

## Validation
- Focused validation passed: `true`.

## beauty
- Runs: `20`; completed `20`.

## sports
- Runs: `20`; completed `20`.

## ml1m
- Runs: `20`; completed `20`.

## lastfm
- Runs: `20`; completed `20`.

## foodcom
- Runs: `20`; completed `20`.

## BPRMF
- Runs: `25`; completed `25`.

## GRU4Rec
- Runs: `25`; completed `25`.

## SASRec
- Runs: `25`; completed `25`.

## LightGCN
- Runs: `25`; completed `25`.

## Files
- `results/tables/f1bz_final_run_manifest.csv`
- `results/tables/f1bz_training_curves.csv`
- `results/tables/f1bz_per_seed_utility_results.csv`
- `results/tables/f1bz_aggregate_utility_results.csv`
- `results/tables/f1bz_per_seed_beyond_utility_results.csv`
- `results/tables/f1bz_aggregate_beyond_utility_results.csv`
- `results/tables/f1bz_traditional_35row_completeness.csv`
- `results/tables/f1bz_first_stage_rankings.csv`
- `results/tables/f1bz_reranking_pools.csv`
- `results/tables/f1bz_candidate_recall.csv`
- `results/tables/f1bz_reranking_readiness.csv`
- `results/tables/f1bz_runtime_provenance.csv`
- `results/tables/f1bz_failure_retry_audit.csv`
- `results/tables/f1bz_validation.csv`
