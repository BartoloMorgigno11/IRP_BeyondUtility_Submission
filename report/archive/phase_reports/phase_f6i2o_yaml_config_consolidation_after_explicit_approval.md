# F6I2O YAML Config Consolidation After Explicit Approval

## Decision

Decision token: `f6i2o_model_bprmf_overrides_no_change_retained_distinct`.

F6I2O retains the three active BPRMF YAML configs as distinct full configs. This is a deliberate no-change consolidation closure, not a physical YAML merge. The active BPRMF loaders use direct `yaml.safe_load` on hardcoded config paths and do not provide an include, extends, or override contract.

## Runtime

Approved interpreter: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_audit_recovery\Scripts\python.exe`.

Observed interpreter: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_audit_recovery\Scripts\python.exe`.

Runtime classification: `defensible_repaired_clone_exact_package_versions`. This is not claimed as exact original executable-runtime reproduction; the original Microsoft Store base interpreter was not recovered and complete original wheel provenance is unavailable.

## Scope

Selected group: `model_bprmf_overrides`.

Member configs:

- `configs/plan_e_bprmf_beauty.yaml`
- `configs/plan_e_bprmf_ml1m_pilot.yaml`
- `configs/plan_e_bprmf_sports.yaml`

No training, LLM inference, Ollama, API call, download, package mutation, staging, or commit was performed.

## Loader Contract Finding

The safe strategy is `retain_distinct_full_configs_no_change`. Physical factoring was rejected because active consumers expect complete top-level YAML mappings and have no active merge/delegation loader. Existing references are retained and documented in `results/tables/phase_f6i2o_reference_migration.csv`.

## Validation

Validation status: `pass`.

- Member config byte hashes are unchanged.
- Pre/post resolved config digests are equal.
- Existing BPRMF outputs were hashed only as frozen evidence; they were not regenerated.
- The complete repository suite was excluded by policy.

Current validation commands:

- `complete_repository_suite`: `pass` observed `excluded_by_phase_policy`.
- `f6i2o_builder`: `pass` observed `completed`.
- `f6i2o_focused_tests`: `pass` observed `11 passed`.
- `f6i2o_guard_regression`: `pass` observed `40 passed`.

The first guard-regression attempt without `--basetemp` reached `37 passed` and then hit three pytest setup errors from `PermissionError` on `C:\Users\Bartolo\AppData\Local\Temp\pytest-of-Bartolo`. The same guard regression was rerun with workspace basetemp and passed.

## Git State

Branch: `plan-d-extensions`.

HEAD: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`.

Git status counts at final builder runtime: `{"added":0,"deleted":11,"modified":16,"other":1,"untracked":1609}`. Full `git status --short --branch` text is recorded in `results/manifests/phase_f6i2o_git_metadata_guard.json`.

Staged paths recorded by the final builder: `none`.

## Files

Created or regenerated F6I2O artifacts:

- `configs/phase_f6i2o_yaml_config_consolidation_after_explicit_approval.yaml`
- `report/phase_f6i2o_yaml_config_consolidation_after_explicit_approval.md`
- `results/logs/phase_f6i2o_builder_final.log`
- `results/logs/phase_f6i2o_focused_tests_final.log`
- `results/logs/phase_f6i2o_guard_regression_final.log`
- `results/logs/phase_f6i2o_regression_pytest/`
- `results/manifests/phase_f6i2o_consolidation_manifest.json`
- `results/manifests/phase_f6i2o_git_metadata_guard.json`
- `results/manifests/phase_f6i2o_pre_consolidation_config_manifest.json`
- `results/manifests/phase_f6i2o_protected_input_hashes.json`
- `results/manifests/phase_f6i2o_rollback_manifest.json`
- `results/tables/phase_f6i2o_config_actions.csv`
- `results/tables/phase_f6i2o_config_dependency_graph.csv`
- `results/tables/phase_f6i2o_config_group_members.csv`
- `results/tables/phase_f6i2o_config_lifecycle_classification.csv`
- `results/tables/phase_f6i2o_consolidation_strategy.csv`
- `results/tables/phase_f6i2o_current_suite_validation.csv`
- `results/tables/phase_f6i2o_determinism.csv`
- `results/tables/phase_f6i2o_duplicate_key_analysis.csv`
- `results/tables/phase_f6i2o_historical_config_validation.csv`
- `results/tables/phase_f6i2o_key_value_inventory.csv`
- `results/tables/phase_f6i2o_loader_contract_inventory.csv`
- `results/tables/phase_f6i2o_next_actions.csv`
- `results/tables/phase_f6i2o_output_equivalence.csv`
- `results/tables/phase_f6i2o_override_precedence.csv`
- `results/tables/phase_f6i2o_post_consolidation_resolved_configs.csv`
- `results/tables/phase_f6i2o_pre_consolidation_resolved_configs.csv`
- `results/tables/phase_f6i2o_reference_migration.csv`
- `results/tables/phase_f6i2o_remaining_yaml_group_readiness.csv`
- `results/tables/phase_f6i2o_resolved_config_equivalence.csv`
- `results/tables/phase_f6i2o_rollback_plan.csv`
- `results/tables/phase_f6i2o_validation.csv`
- `src/analysis/plan_f6i2o_yaml_config_consolidation_after_explicit_approval.py`
- `src/experiments/build_phase_f6i2o_yaml_config_consolidation_after_explicit_approval.py`
- `tests/test_phase_f6i2o_yaml_config_consolidation_after_explicit_approval.py`

Modified current guard registration files:

- `src/analysis/plan_f6i2j_git_state_capture_consolidation_execution.py`
- `src/analysis/plan_f6i2k_manifest_generation_consolidation_execution.py`
- `src/analysis/plan_f6i2l_validation_table_creation_consolidation_execution.py`

## Outputs

Tables: `21` under `results/tables/phase_f6i2o_*.csv`.

Manifests: `5` under `results/manifests/phase_f6i2o_*.json`.

## Next YAML Group

Selected next YAML group: `model_gru4rec_overrides`. It must be handled in a later explicitly approved phase and cannot inherit F6I2O's no-change result without its own loader-contract inspection.
