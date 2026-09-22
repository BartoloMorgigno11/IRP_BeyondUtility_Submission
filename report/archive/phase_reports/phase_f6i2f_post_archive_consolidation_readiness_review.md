# F6I2F Post-Archive Consolidation Readiness Review

## Runtime

- Approved/current audit runtime: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_audit_recovery\Scripts\python.exe`
- Runtime classification: `defensible_repaired_clone_exact_package_versions`
- Caveat: repaired clone exact package tree only; not exact original executable environment.
- Required environment: `PYTHONNOUSERSITE=1`
- Original project package-tree evidence remains distinct from the recovered executable environment and this current audit runtime.

## Archive Reconciliation

- Moves reconciled: `18`
- Tracked-source moves: `11`
- Untracked-source moves: `7`
- Hash mismatches: `0`
- Git reports 11 deleted tracked paths because only tracked originals can appear as deletions; the other 7 moved originals were untracked, and the archive destinations are untracked because no staging was performed.

## Source Scope

- F6I2A Python audit count: `408` structured rows.
- F6I2E AST parse count: `1429` reported broad retained non-archive Python files.
- Authoritative active-local Python count for future consolidation: `272`.
- The broad F6I2E count includes non-active sources such as extracted environment/runtime Python under `results/environment`, tests, and upstream code; it is not the consolidation scope.

## Test Lifecycle

- Cleanup test rows classified: `8`.
- F6I2B retained-path assertions are classified as `migration_superseded_invariant`, not current-state invariants.
- F6I2E and F6I2F are the current cleanup validation anchors.

## Dependency Graph

- Current graph nodes: `421`.
- Current graph edges: `7218`.
- Historical textual mentions are retained as historical/reference edges and are not counted as active dependencies.

## Consolidation Readiness

- Python merge groups revalidated: `9`.
- Ready groups: `deterministic_csv_io`.
- Deferred/kept-separate groups: `candidate_and_cohort_reconciliation;phase_report_generation;request_response_lineage`.
- First selected Python bundle: `deterministic_csv_io`.
- The selected first bundle must retain compatibility wrappers at first and compare deterministic output hashes before/after implementation.
- YAML config groups revalidated: `6`.

## Checkpoint Readiness

- Decision: `defer_commit_until_first_python_consolidation_contract_reviewed`.
- Future staging must be manifest-driven and exact-path only. `git add .`, `git add -A`, and broad wildcard staging are forbidden.

## Validation

- Decision token: `f6i2f_post_archive_readiness_complete_first_python_bundle_selected`.
- Blocking validation failures: `0`.
- Full suite status: excluded; focused offline validation only.
- Forbidden actions did not occur: no training, inference, Ollama/API call, download, package change, staging, or commit.
