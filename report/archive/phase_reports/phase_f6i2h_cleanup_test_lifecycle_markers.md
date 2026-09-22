# F6I2H Cleanup Test Lifecycle Markers

## Runtime

- Approved/current audit runtime: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_audit_recovery\Scripts\python.exe`
- Runtime classification: `defensible_repaired_clone_exact_package_versions`
- Caveat: repaired clone exact package tree only; not exact original executable environment.
- Required environment: `PYTHONNOUSERSITE=1`

## Lifecycle Policy

- A current-state invariant validates the repository state that is required now.
- A phase-snapshot test preserves assertions for the filesystem and artifact state of a specific historical phase.
- Superseded invariants are retained and marked, but are replaced as current blockers by explicit migration/refactor regression tests.
- Historical tests remain runnable intentionally through lifecycle markers, preferably with manifest-backed or archived-copy fixtures.
- Current suites are selected with registered markers instead of hidden permanent addopts exclusions.
- Future cleanup/refactor phases must update lifecycle metadata when they authorize migration or refactor supersession.
- Historical assertions must not be silently edited into current tests because that would erase provenance.

## Results

- Decision token: `f6i2h_test_lifecycle_complete_next_helper_selected`
- Current cleanup tests: `31`
- Historical cleanup tests: `35`
- Superseded F6I2B invariant rows: `2`
- Migration supersession rows/failures: `18` / `0`
- F6I2G refactor regression rows/failures: `10` / `0`
- Ready helper groups: `path_normalization;git_state_capture;manifest_generation;validation_table_creation;hashing_helpers`
- Next selected helper bundle: `path_normalization`
- Protected inputs: `pass` with changed `0` and missing `0`

## Boundaries

- No helper consolidation, wrapper removal, test deletion/move/archive/rename, scientific modification, training, inference, Ollama/API call, download, package change, staging, or commit occurred.
