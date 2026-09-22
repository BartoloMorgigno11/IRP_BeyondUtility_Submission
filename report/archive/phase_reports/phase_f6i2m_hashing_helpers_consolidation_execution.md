# Phase F6I2M Hashing Helpers Consolidation Execution

## Runtime

- Approved interpreter: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_audit_recovery\Scripts\python.exe`.
- Runtime classification: `defensible_repaired_clone_exact_package_versions`.
- This is not exact original executable-environment reproduction.

## Scope

- Selected helper group: `hashing_helpers`.
- Canonical module: `src/audit/common_hashing.py`.
- Approved member files inventoried: 117.
- Pre-refactor helper records: 285.
- Wrappers migrated: 4.
- Scientific files, datasets, prompts, parsers, metrics, models, histories, candidates, and experiment results were not modified.

## Equivalence

- Function equivalence failures: 0.
- Real-hash mismatches: 0.
- Protected input status: `pass`.
- Git metadata status: `pass`.

## Dependency Graph

- F6I2F before nodes: 421; before edges: 7218.
- New common_hashing importers: 3.
- Circular imports: none.

## Validation Commands

- `f6i2m_focused_tests`: `pass`; observed `19 passed`; command `$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\.venv_audit_recovery\Scripts\python.exe -B -m pytest -p no:cacheprovider --basetemp results\logs\phase_f6i2m_pytest tests\test_phase_f6i2m_hashing_helpers_consolidation_execution.py`.
- `current_cleanup_suite`: `pass`; observed `96 passed`; command `$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\.venv_audit_recovery\Scripts\python.exe -B -m pytest -p no:cacheprovider --basetemp results\logs\phase_f6i2m_pytest -m "cleanup_current and not unsafe_external" <F6I2E..F6I2M current cleanup tests>`.
- `f6i2l_validation_regression`: `pass`; observed `10 passed`; command `$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\.venv_audit_recovery\Scripts\python.exe -B -m pytest -p no:cacheprovider --basetemp results\logs\phase_f6i2m_pytest tests\test_phase_f6i2l_validation_table_creation_consolidation_execution.py`.
- `f6i2k_manifest_regression`: `pass`; observed `9 passed`; command `$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\.venv_audit_recovery\Scripts\python.exe -B -m pytest -p no:cacheprovider --basetemp results\logs\phase_f6i2m_pytest tests\test_phase_f6i2k_manifest_generation_consolidation_execution.py`.
- `f6i2j_git_regression`: `pass`; observed `4 passed, 6 deselected`; command `$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\.venv_audit_recovery\Scripts\python.exe -B -m pytest -p no:cacheprovider --basetemp results\logs\phase_f6i2m_pytest tests\test_phase_f6i2j_git_state_capture_consolidation_execution.py -k "common_git or fixture_matrix or current_project_read_only"`.
- `f6i2i_path_regression`: `pass`; observed `8 passed`; command `$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\.venv_audit_recovery\Scripts\python.exe -B -m pytest -p no:cacheprovider --basetemp results\logs\phase_f6i2m_pytest tests\test_phase_f6i2i_path_normalization_consolidation_execution.py`.
- `f6i2g_csv_regression`: `pass`; observed `8 passed`; command `$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\.venv_audit_recovery\Scripts\python.exe -B -m pytest -p no:cacheprovider --basetemp results\logs\phase_f6i2m_pytest tests\test_phase_f6i2g_first_python_consolidation_execution.py`.
- `f6i2e_migration_regression`: `pass`; observed `7 passed`; command `$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\.venv_audit_recovery\Scripts\python.exe -B -m pytest -p no:cacheprovider --basetemp results\logs\phase_f6i2m_pytest tests\test_phase_f6i2e_substantive_archive_migration_execution.py`.
- `marker_discovery_validation`: `pass`; observed `96 tests collected`; command `$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\.venv_audit_recovery\Scripts\python.exe -B -m pytest -p no:cacheprovider --basetemp results\logs\phase_f6i2m_pytest --collect-only -q -m "cleanup_current or cleanup_historical" <F6I2E..F6I2M cleanup tests>`.
- `full_suite`: `pass`; observed `full suite excluded due unsafe training/inference/download/checkpoint paths`; command `not_run`.

## Validation

- `runtime_is_recovered_interpreter`: `pass`.
- `python_no_user_site`: `pass`.
- `phase_id`: `pass`.
- `f6i2l_authoritative`: `pass`.
- `canonical_module_exists`: `pass`.
- `canonical_module_has_no_phase_or_scientific_terms`: `pass`.
- `function_equivalence`: `pass`.
- `real_hash_equivalence`: `pass`.
- `protected_guard_equivalence`: `pass`.
- `manifest_boundary`: `pass`.
- `dependency_graph`: `pass`.
- `protected_inputs`: `pass`.
- `git_metadata`: `pass`.
- `current_suite`: `pass`.

## Decision

- Decision token: `f6i2m_hashing_helpers_consolidation_complete`.
- Next repository-cleanup phase: `phase_f6i2n_repository_cleanup_after_explicit_approval`.
