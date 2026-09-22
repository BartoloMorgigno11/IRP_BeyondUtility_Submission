# F0 Final Extension Freeze

Decision token: `FINAL_EXTENSION_FREEZE_VALIDATED`

## Preconditions
- HEAD: `30d6c3e5fcb0e97451a4420829b88de10bb57213`
- Expected PRE-F0 checkpoint: `30d6c3e5fcb0e97451a4420829b88de10bb57213`
- F0 executed offline: inference `0`, training `0`, downloads `0`, network calls `0`.

## Final datasets
- `beauty`, `sports`, `ml1m`, `lastfm`, `foodcom`.
- Video Games is absent from active raw and result rows.

## Final models
- `qwen2:7b-instruct` manifest `dd314f039b9d` (PAPER_PROTOCOL_DIFFERENT_OPEN_MODEL)
- `mistral:7b-instruct` manifest `6577803aa9a0` (PAPER_PROTOCOL_DIFFERENT_OPEN_MODEL)
- `gemma2:9b` manifest `ff02c3702f32` (ORIGINAL_EXTENSION)

## Protocol
- Candidate pool `20` = 1 held-out positive + 19 unseen negatives.
- Output length `10`, parser `llm_rank_strict_index_top10_v1`, K `5`.
- Structural invalidity is fail-strict, preserved raw, no semantic repair, no retry solely for model invalidity.

## Universe counts
- Raw experiment rows: `323`
- Result cell rows: `8938`
- Primary ranking rows: `30`
- History rows: `90`

## Request budget
- `EXISTING_REUSABLE_REQUESTS` = `0`
- `CORE_MANDATORY_NOW_REQUESTS` = `29292`
- `REQUIRED_SENSITIVITY_REQUESTS` = `21000`
- `TRAINING_DEPENDENT_REQUESTS` = `29292`
- `CONDITIONAL_REQUESTS` = `9000`
- `OPTIONAL_REQUESTS` = `12000`
- `BLOCKED_REQUESTS` = `0`
- `MANDATORY_RAW_REQUESTS_TOTAL` = `50292`
- `FULL_PLANNED_IF_CONDITIONAL_REQUESTS_TOTAL` = `88584`
- `CONCEPTUAL_ALL_APPLICABLE_REQUESTS_TOTAL` = `100584`
- `PREVIOUS_D0B_ESTIMATE` = `44928`
- `MANDATORY_MINUS_PREVIOUS_D0B_ESTIMATE` = `5364`

## Global blockers
- `PopREO@5`: `BLOCKED_GLOBAL_PROTOCOL` - paper/original repository/frozen audits do not recover the grouping policy sufficiently for defensible execution
- `DPD@5`: `BLOCKED_GLOBAL_PROTOCOL` - dataset-spanning protected/group attribute definition is not specified
- `K-S sample representativeness`: `IMPLEMENTATION_TO_COMPLETE` - scientific diagnostic is recoverable as a predeclared cohort-vs-population distribution comparison; implementation is incomplete

## Food.com feature policy
- Primary semantic representation: `RAW_recipes.tags`; ingredients are secondary sensitivity only.

## Execution DAG
- DAG is acyclic and materialized in `results/tables/f0_final_execution_dag.csv`.

## Validation
- Validation passed: `true`
