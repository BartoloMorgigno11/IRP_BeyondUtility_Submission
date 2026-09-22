# Phase F6I2L Validation Table Creation Consolidation Execution

## Runtime

- Approved interpreter: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_audit_recovery\Scripts\python.exe`.
- Runtime classification: `defensible_repaired_clone_exact_package_versions`.
- This is not exact original executable-environment reproduction.

## Scope

- Selected helper group: `validation_table_creation`.
- Canonical module: `src/audit/common_validation.py`.
- Approved member files: 116.
- Shared validation-family helpers inventoried: 122.
- Wrappers migrated: 5.
- Hashing helpers, manifest generation, Git capture, path normalization, CSV byte semantics, scientific metrics, prompts, parsers, datasets, candidates, histories, and model outputs were not consolidated.

## Equivalence

- Function equivalence failures: 0.
- Schema equivalence failures: 0.
- Output hash mismatches: 0.
- Protected input status: `pass`.
- Git metadata status: `pass`.

## Validation

- `runtime_is_recovered_interpreter`: `pass`.
- `python_no_user_site`: `pass`.
- `canonical_module_exists`: `pass`.
- `canonical_module_has_no_hashing`: `pass`.
- `function_equivalence`: `pass`.
- `schema_equivalence`: `pass`.
- `output_hash_equivalence`: `pass`.
- `failure_policy_equivalence`: `pass`.
- `dependency_graph`: `pass`.
- `future_dependency`: `pass`.
- `protected_inputs`: `pass`.
- `git_metadata`: `pass`.
- `current_suite`: `pass`.

## Executed Validation Commands

- `f6i2l_focused_tests`: `pass`; observed `10 passed`; command `$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\.venv_audit_recovery\Scripts\python.exe -B -m pytest -p no:cacheprovider --basetemp results\logs\phase_f6i2l_pytest tests\test_phase_f6i2l_validation_table_creation_consolidation_execution.py`.
- `current_cleanup_suite`: `pass`; observed `77 passed`; command `$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\.venv_audit_recovery\Scripts\python.exe -B -m pytest -p no:cacheprovider --basetemp results\logs\phase_f6i2l_pytest -m "cleanup_current and not unsafe_external" <F6I2E..F6I2L current cleanup tests>`.
- `f6i2k_manifest_regression`: `pass`; observed `9 passed`; command `$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\.venv_audit_recovery\Scripts\python.exe -B -m pytest -p no:cacheprovider --basetemp results\logs\phase_f6i2l_pytest tests\test_phase_f6i2k_manifest_generation_consolidation_execution.py`.
- `f6i2j_git_regression`: `pass`; observed `4 passed, 6 deselected`; command `$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\.venv_audit_recovery\Scripts\python.exe -B -m pytest -p no:cacheprovider --basetemp results\logs\phase_f6i2l_pytest tests\test_phase_f6i2j_git_state_capture_consolidation_execution.py -k "common_git or fixture_matrix or current_project_read_only"`.
- `f6i2i_path_regression`: `pass`; observed `8 passed`; command `$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\.venv_audit_recovery\Scripts\python.exe -B -m pytest -p no:cacheprovider --basetemp results\logs\phase_f6i2l_pytest tests\test_phase_f6i2i_path_normalization_consolidation_execution.py`.
- `f6i2g_csv_regression`: `pass`; observed `8 passed`; command `$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\.venv_audit_recovery\Scripts\python.exe -B -m pytest -p no:cacheprovider --basetemp results\logs\phase_f6i2l_pytest tests\test_phase_f6i2g_first_python_consolidation_execution.py`.
- `f6i2e_migration_regression`: `pass`; observed `7 passed`; command `$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\.venv_audit_recovery\Scripts\python.exe -B -m pytest -p no:cacheprovider --basetemp results\logs\phase_f6i2l_pytest tests\test_phase_f6i2e_substantive_archive_migration_execution.py`.
- `marker_discovery_validation`: `pass`; observed `77 tests collected`; command `$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\.venv_audit_recovery\Scripts\python.exe -B -m pytest -p no:cacheprovider --basetemp results\logs\phase_f6i2l_pytest --collect-only -q -m "cleanup_current or cleanup_historical" <F6I2E..F6I2L cleanup tests>`.
- `full_suite`: `pass`; observed `full suite excluded due unsafe training/inference/download/checkpoint paths`; command `not_run`.

## Decision

- Decision token: `f6i2l_validation_table_consolidation_complete`.
- Next helper group: `hashing_helpers`.
