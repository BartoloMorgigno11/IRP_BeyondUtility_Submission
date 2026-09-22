# F1BR Traditional Protocol Recovery

Decision token: `F1BR_TRADITIONAL_PROTOCOL_RESOLVED`

## ReChorus Lineage
- Gitlink commit: `fb646d4eed13a8d81ae308965b08c816f40d1a57`.
- Local ReChorus source/defaults are not available; defaults are not fabricated.
- Project adapter zero-training smoke: `PASS`.

## Paper Hyperparameters
- Frozen rows: `20`.
- Appendix LightGCN repeated-label typo documented as non-operational because the final two configurations are identical.

## Seed Policy
- Final seeds: `[42,43,44,45,46]`.
- Classification: `LOCAL_REPRODUCIBLE_FIVE_SEED_POLICY`; no paper seed identity claim.

## Execution Readiness
- Deterministic baseline rows ready: `15`.
- Trainable dataset/model rows ready: `20`.
- Original-four paper-hyperparameter aligned local implementation rows: `16`.

## Food.com
- Validation search configs: `40`.
- Tuning contracts: `4`.
- Selection metric: `validation_ndcg_at_5`; final test metrics cannot tune hyperparameters.

## Training Budget
- `ORIGINAL_FOUR_FINAL_RUNS` = `80`.
- `FOODCOM_FINAL_RUNS` = `20`.
- `FOODCOM_TUNING_RUNS` = `40`.
- `FINAL_EVIDENCE_RUNS` = `100`.
- `TOTAL_TRAINING_RUNS` = `140`.

## Reranking Source
- Rows: `5`.
- Paper ranker set recovered; union ordering frozen as adapted ordered deduplicated top-20 policy.

## Integrity
- Active datasets: `beauty,sports,ml1m,lastfm,foodcom`.
- Video Games rows: `0`.
- LLM inference/training/download/network counts are all zero.

## Validation
- Passed: `true`.
