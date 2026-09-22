# F6I2E Substantive Archive Migration Execution

## Runtime

- Approved/current audit runtime: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_audit_recovery\Scripts\python.exe`
- Runtime classification: `defensible_repaired_clone_exact_package_versions`
- Caveat: defensible repaired clone exact package versions only; not exact original executable-environment reproduction.
- Required environment: `PYTHONNOUSERSITE=1`

## Results

- Decision token: `f6i2e_substantive_archive_migration_complete`
- Approved candidates expected/passing/rejected: `18` / `18` / `0`
- Files moved or restart-validated: `18`
- Bytes migrated: `303523`
- Active reference rows closed: `15` of `15`
- Physical textual replacements applied: `11`
- Unresolved active references: `0`
- Historical reference rows retained or classified non-text/binary: `695`
- Compatibility stubs/import aliases created: `0`
- Target collisions: `0`
- Empty directories removed: `0`
- Archive index entries: `18`
- Rollback entries: `18`
- Next phase: `phase_f6i2f_post_archive_consolidation_readiness_review`

## Protected Inputs

- Result: `pass`
- Changed protected entries: `0`
- Missing protected entries: `0`

## Validation

- Blocking validation failures: `0`
- Full suite status: excluded; focused offline validation only.
- Focused command log: `dry F6I2E tests 7 passed; dry builder no migration; execution builder moved 18 files and closed 15 active reference rows; post-migration F6I2E tests 7 passed twice; F6I2B prior focused test no longer applicable after approved archive path changes: 5 passed, 1 protected-input historical-path failure; AST parse active retained Python 1429 parsed; F6I2E collect-only 7 tests; archive/rollback/current-context schema check passed; active-reference validation 15 pass; deterministic two-run hash comparison passed`

## Boundaries

F6I2E moved only the approved 18 files, rewrote only approved active references, created no compatibility stubs, and performed no training, inference, Ollama/API call, download, package change, staging, or commit.
