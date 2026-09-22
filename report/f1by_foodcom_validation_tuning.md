# F1BY Food.com Validation Tuning

Decision token: `F1BY_FOODCOM_WINNERS_SELECTED`

## Execution
- Planned tuning configs: `40`.
- Completed configs: `40`.
- Failed configs: `0`.
- Winners selected: `4`.
- Classification: `ORIGINAL_EXTENSION_VALIDATION_PROTOCOL`.

## Winners
- `BPRMF`: `F1BR-FOODCOM-SEARCH-CF0C268538BB66EF` NDCG@5 `0.25997168809994115` HR@5 `0.34699999999999998`.
- `GRU4Rec`: `F1BR-FOODCOM-SEARCH-83800622AC730568` NDCG@5 `0.34343431550189368` HR@5 `0.42699999999999999`.
- `SASRec`: `F1BR-FOODCOM-SEARCH-5F3EA9FE6EEF42A3` NDCG@5 `0.30964535860646131` HR@5 `0.39500000000000002`.
- `LightGCN`: `F1BR-FOODCOM-SEARCH-CDA9D3AE06F9E4D0` NDCG@5 `0.36593488245019623` HR@5 `0.46999999999999997`.

## Runtime
- Python: `3.12.10`.
- Torch: `2.5.1+cu121`.
- CUDA: `true`.
- Device rows: `cuda`.

## Leakage
- Leakage audit failures: `0`.
- Final-test metrics available at selection: `false`.

## Preservation
- Baseline rows: `15`.
- Baseline recommendation rows: `292920`.
- Baseline hashes unchanged: `true`.
- Original-four final runs: `0`; Food.com final five-seed runs: `0`; LLM inference: `0`.

## Future Final Contract
- Food.com final runs will use each selected config, train on `i1..i[T-1]`, use seeds `[42,43,44,45,46]`, evaluate once on frozen final target `i[T]` with the pre-existing F0 final-test candidate contract, and never retune on final-test performance.

## Validation
- Focused validation passed: `true`.
