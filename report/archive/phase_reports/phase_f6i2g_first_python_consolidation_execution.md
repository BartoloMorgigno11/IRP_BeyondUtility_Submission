# F6I2G First Python Consolidation Execution

## Runtime

- Approved/current audit runtime: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_audit_recovery\Scripts\python.exe`
- Runtime classification: `defensible_repaired_clone_exact_package_versions`
- Caveat: repaired clone exact package tree only; not exact original executable environment.
- Required environment: `PYTHONNOUSERSITE=1`

## Consolidation

- Selected group: `deterministic_csv_io`
- Canonical shared module: `src/audit/common_io.py`
- Centralized duplicated helper names: `read_csv`, `read_csv_rows`, `write_csv`
- Member files: `22`
- Wrapper functions retained: `33`
- File reduction claimed: `0`

## Equivalence

- Function-equivalence mismatches: `0`
- Output hash mismatches: `0`
- Dependency graph before: `421` nodes / `7218` edges
- Dependency graph after: `425` nodes / `3433` edges
- Circular imports: `none`

## Validation

- Decision token: `f6i2g_deterministic_csv_io_consolidation_complete`
- Protected inputs: `pass` with changed `0` and missing `0`
- No unrelated Python group, scientific logic, dataset, prompt, parser, metric, model, history, candidate, or result was modified.
- No training, inference, Ollama/API call, download, package change, staging, or commit occurred.
