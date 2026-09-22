# F1BXR Food.com Validation Contract

Decision token: `F1BXR_FOODCOM_VALIDATION_CONTRACT_RESOLVED`

## Eligibility
- Eligible validation users supporting candidate construction: `37392`.
- Required validation users: `1000`.

## Nested Split
- Tuning train: `i1..i[T-2]`; validation target: `i[T-1]`; final-test target: `i[T]`.
- Later final training policy: retrain using `i1..i[T-1]` after winner freeze; not executed now.

## Validation Contract
- Validation users selected: `1000`.
- Candidate rows: `20000`.
- Candidate sets with audit failures: `0`.

## Leakage
- Leakage audit failures: `0`.
- Final/test validation cohort overlap: `0`.

## Tuning Readiness
- READY_TO_TUNE rows: `40/40`.
- Selection metric: `validation_ndcg_at_5`.

## Preservation
- F1BX baseline rows: `15`.
- F1BX recommendation rows: `292920`.
- Baseline hashes unchanged: `true`.
- F0 mandatory budget unchanged: `50292`.
- F0 conceptual all-applicable budget unchanged: `100584`.
- Training/LLM/network/download counters: `0`.

## Validation
- Focused validation passed: `true`.
