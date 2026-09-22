# F6I2I Path Normalization Consolidation Execution

## Runtime

- Approved/current audit runtime: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_audit_recovery\Scripts\python.exe`
- Runtime classification: `defensible_repaired_clone_exact_package_versions`
- Caveat: repaired clone exact package tree only; not exact original executable environment.
- Required environment: `PYTHONNOUSERSITE=1`

## Consolidation

- Selected group: `path_normalization`
- Canonical shared module: `src/audit/common_paths.py`
- Centralized helper entry point: `rel`
- Member files: `7`
- Wrappers retained: `7`
- File reduction claimed: `0`

## Equivalence

- Function-equivalence rows/failures: `266` / `0`
- Output hash mismatches: `0`
- Dependency graph before: `422` nodes / `3402` edges
- Dependency graph after: `422` nodes / `3409` edges
- Circular imports: `none`

## Validation

- Decision token: `f6i2i_path_normalization_consolidation_complete`
- Protected inputs: `pass` with changed `0` and missing `0`
- No unrelated helper group, scientific logic, dataset, prompt, parser, metric, model, history, candidate, or result was modified.
- No training, inference, Ollama/API call, download, package change, staging, or commit occurred.
