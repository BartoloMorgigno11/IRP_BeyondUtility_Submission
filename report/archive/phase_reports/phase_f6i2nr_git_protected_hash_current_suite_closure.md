# Phase F6I2NR Git Protected Hash Current-Suite Closure

## Runtime

- Approved interpreter: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_audit_recovery\Scripts\python.exe`.
- Runtime classification: `defensible_repaired_clone_exact_package_versions`.
- This is not exact original executable-environment reproduction.

## Original Failures

- `test_runtime_phase_and_decision`: `f6i2m_failed_protected_input_change`.
- `test_real_hash_and_protected_guard_equivalence`: `{fail, pass}`.
- `test_validation_and_git_metadata_pass`: `protected_inputs/real_hash/protected_guard validation fails during live F6I2M rerun`.

## Git Difference

- Historical F6I2M recursive `.git` digest: `0FB9F154BE3A92B2AD454FA50E4ED48512B852AA4094C138937BB2E8C43B84EC`.
- Current recursive `.git` digest under the same algorithm: `A6CFA44CE8A45459769B742B44F3E5EED2AC94FBD94E6839ABC46A45A552C7FE`.
- Historical/current file counts: `23857` -> `23971`.
- Exact historical file-level paths are not recoverable because F6I2M did not persist a `.git` file inventory.
- Current semantic guard protects branch, HEAD, index, staged set/content, refs, packed refs, config, and lock files.

## Semantic Git Guard

- Branch: `plan-d-extensions`.
- HEAD: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`.
- Staged paths: ``.
- Lock files remaining: `0`.
- Guard status: `pass`.

## F6I2N Readiness

- `wrappers_inventoried`: `pass` (`551`).
- `unresolved_dynamic_references`: `pass` (`0`).
- `function_removal_candidates`: `pass` (`0`).
- `file_removal_candidates`: `pass` (`0`).
- `archive_candidates`: `pass` (`0`).
- `safe_removal_bundles`: `pass` (`0`).
- `selected_config_group`: `pass` (`model_bprmf_overrides`).
- `determinism_mismatch_count`: `pass` (`0`).
- `protected_inputs`: `pass` (`pass`).
- `f6i2n_only_failure_is_current_suite`: `pass` (`current_suite`).

## Current Suite

- `f6i2nr_focused_tests`: `pass`; observed `11 passed`.
- `original_f6i2m_three_assertions`: `pass`; observed `3 passed, 16 deselected`.
- `f6i2n_focused_tests`: `pass`; observed `13 passed`.
- `f6i2m_focused_regression`: `pass`; observed `19 passed`.
- `canonical_current_cleanup_suite`: `pass`; observed `120 passed`.
- `f6i2l_validation_regression`: `pass`; observed `10 passed`.
- `f6i2k_manifest_regression`: `pass`; observed `9 passed`.
- `f6i2j_git_regression`: `pass`; observed `4 passed, 6 deselected`.
- `f6i2i_path_regression`: `pass`; observed `8 passed`.
- `f6i2g_csv_regression`: `pass`; observed `8 passed`.
- `f6i2e_migration_regression`: `pass`; observed `7 passed`.
- `marker_discovery_validation`: `pass`; observed `120 tests collected`.
- `full_suite`: `pass`; observed `full suite excluded due unsafe training/inference/download/checkpoint paths`.

## Decision

- Decision token: `f6i2nr_git_guard_current_suite_closure_complete`.
- Next phase: `phase_f6i2o_yaml_config_consolidation_after_explicit_approval`.
