# F0R Final Freeze Sanity

Decision token: `F0_FINAL_SANITY_VALIDATED`

## Primary breakdown
- Dataset N sum: `4882`.
- Primary requests: `29292`.

## Reranking breakdown
- Reranking requests: `29292`; training-dependent only.

## Sensitivity breakdown
- History requests: `18000`.
- CandDif/order rows: `15`; requests `3000`.

## Profile and optional
- Profile rows: `60`; conditional requests `9000`.
- Optional pool rows: `60`; requests `12000`.

## Runtime
- Python: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_audit_recovery\Scripts\python.exe` / `3.12.10`.
- Ollama: `0.32.15`.
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`.
- RAM total GB: `31.715`.

## Request totals
- `CORE_MANDATORY` = `29292`
- `REQUIRED_SENSITIVITY` = `21000`
- `MANDATORY_TOTAL` = `50292`
- `TRAINING_DEPENDENT` = `29292`
- `MANDATORY_PLUS_TRAINING_DEPENDENT` = `79584`
- `CONDITIONAL` = `9000`
- `FULL_PLANNED_IF_CONDITIONAL` = `88584`
- `OPTIONAL` = `12000`
- `CONCEPTUAL_ALL_APPLICABLE` = `100584`
- `EXISTING_REUSABLE` = `0`
- `CANARY_INCREMENTAL` = `0`

## Validation
- Passed: `true`.
