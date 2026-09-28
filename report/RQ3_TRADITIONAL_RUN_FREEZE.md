# RQ3 Traditional Run Freeze

Status: `LOCAL_OPERATIONALIZATION_NOT_PAPER_EXACT`.

Under the frozen local D'_H exposure contract, a one-interaction training history cannot generate next-item sequence examples for GRU4Rec or SASRec. Their H=1 cells are therefore structurally not applicable rather than failed or zero-performing.

## Accounting
- Applicable trainable jobs completed: `600` / `600`.
- Structural NA trainable jobs: `40` / `40`.
- Deterministic cells completed: `64` / `64`.
- BM25 cells completed: `32` / `32`.
- MostPop cells completed: `32` / `32`.

## Validation
- Validation passed: `true`.
- Validation failures: `0`.
- FoodCom run: `false`.
- Paper exact: `false`.
- Ollama used: `false`.

## Outputs
- `results/tables/rq3_traditional_metrics_per_seed.csv`
- `results/tables/rq3_traditional_metrics_aggregated.csv`
- `results/tables/rq3_traditional_job_accounting.csv`
- `results/tables/rq3_traditional_validation.csv`
- `artifacts/rq3_traditional_run/run_manifest.json`
- `results/plots/rq3_traditional_diagnostic/`
