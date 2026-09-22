# F1BX Baselines and Food.com Validation Tuning

Decision token: `F1BX_TUNING_AND_BASELINES_BLOCKED`

## Deterministic Baselines
- Baseline rows computed: `15/15`.
- `BM25-like/content` computed rows: `5`.
- `MostPop` computed rows: `5`.
- `Random` computed rows: `5`.
- Food.com `BM25-like/content` text source: `RAW_recipes.tags` only.
- Utility metric rows: `45`.
- Beyond-utility metric rows: `210`.
- Degenerate PopREO/DPD result rows after observed recommendations: `3`.

## Food.com Tuning
- Planned tuning configs: `40`.
- Training runs executed: `0`.
- Winners selected: `0`.
- Blocker: `BLOCKED_NOT_EXECUTED_MISSING_VALIDATION_CANDIDATE_CONTRACT`.
- Reason: F1BR freezes Food.com search space, tuning seed, validation metric, and no-test-tuning policy, but no separate frozen Food.com validation candidate/cohort contract is materialized; using the F0 primary final-test candidate set for tuning would leak final-test evidence.

## Resource Preflight
- Python: `3.12.10`.
- Torch: `2.5.1+cu121`.
- CUDA: `true`.
- Free disk bytes: `166096908288`.
- RAM bytes: `34053414912`.

## Integrity
- Active datasets: `beauty,sports,ml1m,lastfm,foodcom`.
- Video Games rows: `0`.
- F0 budget unchanged: `true`.
- Food.com hashes unchanged: `true`.
- H0 hashes unchanged: `true`.
- LLM/API/download/network counters: `0`.

## Validation
- Focused validation checks passed: `true`.
