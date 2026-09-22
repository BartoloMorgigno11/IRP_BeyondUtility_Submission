# Phase F6I1 Repository Retention Audit And Safe Cleanup Batch 1

Decision: `f6i1_repository_retention_audit_completed_with_safe_batch1_cleanup`.

Runtime classification: `defensible_repaired_clone_exact_package_versions`. This is not exact original runtime reproduction.

## Scope

- Phase namespace selected: `phase_f6i1`.
- Phase registry inspected: `results/tables/workspace_phase_and_supersession_registry.csv`.
- Namespace occupied before generation: `False`.
- Branch: `plan-d-extensions`.
- HEAD: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`.
- Cleanup policy: delete only Python bytecode cache files outside protected paths when the retained source file exists.

## Initial Capture

- Git status rows: `1225`.
- Tracked files: `456`.
- Modified tracked files: `3`.
- Untracked files: `1222`.
- Ignored files: `1703`.
- Relevant inventory rows: `1948`.
- Dependency edges: `45826`.
- Duplicate group member rows: `38`.
- Repository relevant size bytes, excluding expanded environment package trees: `6714964996`.

## Classification Summary

- `keep_canonical_builder`: `222`
- `keep_canonical_runtime`: `7`
- `keep_canonical_test`: `143`
- `keep_current_documentation`: `43`
- `keep_future_dependency`: `1061`
- `keep_protected`: `183`
- `keep_provenance_only`: `133`
- `manual_review`: `156`

## Category Summary

- `config`: `27`
- `directory_unit`: `10`
- `docs`: `3`
- `local_data`: `57`
- `manifest`: `24`
- `model_artifact`: `24`
- `model_checkpoint`: `40`
- `official_repository`: `27`
- `paper`: `3`
- `prediction`: `4`
- `raw_data`: `5`
- `raw_llm_output`: `23`
- `report`: `121`
- `result_other`: `153`
- `result_table`: `1015`
- `root_or_other`: `9`
- `source_code`: `260`
- `test`: `143`

## Pre-Deletion Manifests

- `results/tables/phase_f6i1_relevant_file_inventory.csv`
- `results/tables/phase_f6i1_dependency_edges.csv`
- `results/tables/phase_f6i1_retention_classification.csv`
- `results/tables/phase_f6i1_delete_now_candidates.csv`
- `results/tables/phase_f6i1_manual_review.csv`
- `results/tables/phase_f6i1_future_dependency_map.csv`
- `results/tables/phase_f6i1_duplicate_groups.csv`

The phase wrote these manifests before each cleanup pass. The final inventory reflects the retained post-cleanup file set; the delete-now candidate and cleanup execution tables preserve cumulative deletion evidence. The delete manifest fails closed when a row lacks a retained source/regeneration path.

## Cleanup Result

- Delete-now candidates: `152`.
- Files deleted: `152`.
- Cleanup status counts: `{'pass': 152}`.
- Non-cache exact duplicate scientific artifacts deleted: `0`.
- Original repository, paper files, raw data, raw LLM outputs, predictions, checkpoints, candidates, histories, prompt templates, parsers, metrics, active configs, reports, and runtime units were retained.

## Validation

- Blocking validation failures: `0`.
- Validation table: `results/tables/phase_f6i1_validation.csv`.
- Cleanup execution table: `results/tables/phase_f6i1_cleanup_execution.csv`.

## Limitations

- `.venv`, `.venv_audit_recovery`, sibling recovery environments, packaged CPython runtime trees, and inaccessible pytest/log directories are represented as directory units, not package-file inventories.
- Manual-review rows are intentionally retained. No import/reference absence was used as standalone deletion evidence.
- Markdown consolidation, Python module merging, YAML architecture rewriting, and non-regenerable script/config/test deletion were not performed in batch 1.
