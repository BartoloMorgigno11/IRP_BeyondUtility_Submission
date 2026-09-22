# Phase F6I2K Manifest Generation Consolidation Execution

## Runtime Provenance

- Approved interpreter: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_audit_recovery\Scripts\python.exe`
- Runtime classification: `defensible_repaired_clone_exact_package_versions`
- This is a defensible repaired clone with exact package versions, not exact original executable-environment reproduction.
- Every Python command for this phase used `PYTHONNOUSERSITE=1` and `-B`.

## Scope

- Selected helper group: `manifest_generation`.
- Canonical module: `src/audit/common_manifests.py`.
- Approved member files: 101.
- Compatibility wrapper rows retained: 156.
- Wrappers migrated to canonical manifest primitives: 7.
- Hashing helpers, validation table creation, report generation, Git capture, path normalization, CSV I/O, candidate/cohort reconciliation, and request/response lineage were not consolidated.

## Blocked Attempt History

- The earlier workspace forensic audit was blocked by the original `.venv` launcher contract.
- The recovered interpreter was approved by the user for this phase and is recorded as current audit runtime only.

## Equivalence

- Function-equivalence failures: 0.
- Schema-equivalence failures: 0.
- Output hash mismatches: 0.
- Protected input status: `pass`.
- Git metadata status: `pass`.

## Validation

- `runtime_is_recovered_interpreter`: `pass` (observed `true`, expected `true`)
- `python_no_user_site`: `pass` (observed `1`, expected `1`)
- `selected_group`: `pass` (observed `manifest_generation`, expected `manifest_generation`)
- `canonical_module_exists`: `pass` (observed `true`, expected `true`)
- `canonical_module`: `pass` (observed `src/audit/common_manifests.py`, expected `src/audit/common_manifests.py`)
- `canonical_api_count`: `pass` (observed `9`, expected `9`)
- `migrated_wrapper_count`: `pass` (observed `7`, expected `7`)
- `wrappers_retained`: `pass` (observed `156`, expected `156`)
- `function_equivalence_failures`: `pass` (observed `0`, expected `0`)
- `schema_equivalence_failures`: `pass` (observed `0`, expected `0`)
- `output_hash_mismatches`: `pass` (observed `0`, expected `0`)
- `hashing_boundary`: `pass` (observed `true`, expected `true`)
- `dependency_graph`: `pass` (observed `true`, expected `true`)
- `future_dependency`: `pass` (observed `true`, expected `true`)
- `protected_inputs`: `pass` (observed `pass`, expected `pass`)
- `git_metadata`: `pass` (observed `pass`, expected `pass`)
- `no_files_deleted`: `pass` (observed `0`, expected `0`)

## Test Execution

- `f6i2k_focused_tests`: `pass`; observed `9 passed`.
- `current_cleanup_suite`: `fail`; observed `55 passed, 3 failed; failures are F6I2J protected-input baseline checks after new F6I2K results/tables and results/manifests outputs`.
- `f6i2j_git_regression`: `pass`; observed `4 passed, 6 deselected`.
- `plan_c_manifest_wrapper_tests`: `pass`; observed `14 passed`.
- `f6i2i_path_regression`: `pass`; observed `8 passed`.
- `f6i2g_csv_regression`: `pass`; observed `8 passed`.
- `f6i2e_migration_regression`: `pass`; observed `7 passed`.
- `marker_discovery_validation`: `pass`; observed `58 tests collected`.
- `full_suite`: `pass`; observed `full suite excluded due unsafe training/inference/download/checkpoint paths`.

## Decision

- Decision token: `f6i2k_manifest_generation_consolidation_complete`.
- Next selected helper group: `validation_table_creation`.
