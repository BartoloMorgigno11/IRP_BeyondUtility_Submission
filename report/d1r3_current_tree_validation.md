# D1R3 Current Tree Validation

## TAXONOMY CORRECTION
- `tests/test_exactness_frontier_and_evidence_closure.py` and `tests/test_workspace_forensic_state_audit.py` are classified as `HISTORICAL_FREEZE_VALIDATION`.
- Both files carry explicit `cleanup_historical` and `phase_snapshot` markers.
- Historical assertions such as row_count `136` were not modified.

## HISTORICAL SNAPSHOTS RESTORED
- D1R3 did not run historical builders on current master.
- The archived workspace forensic report remains the available frozen authority and records protected/manifested rows compared = `136`.
- The full historical `workspace_relevant_file_inventory.csv` content is not tracked in commit `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`; no replacement CSV was fabricated.

## DIRECT SCRIPT FIXES RETAINED
- Import/bootstrap fixes for exactness and workspace forensic builders were retained.
- Non-mutating `--help` smoke checks for both builders passed.

## CURRENT SUITE CONTENT
- Current pytest node IDs selected: 306.
- Direct smoke checks selected: 2.
- Historical snapshot tests and old HEAD/branch worktree tests are excluded and documented.

## CURRENT SUITE RESULT
- Pytest: 304 passed, 2 skipped, 0 failed, 0 errors.
- Duration: 42.053 seconds.
- Exit code: 0.

## SIDE EFFECT AUDIT
- Current suite used pytest cache disabled and no historical builders.
- Historical snapshot artifact rewrites observed in D1R3: 0.

## FOOD.COM INTEGRITY
- Food.com hashes unchanged: `true`.

## H0 INTEGRITY
- H0 hashes unchanged: `true`.

## ACTIVE DATASETS
- Active datasets: `beauty;sports;ml1m;lastfm;foodcom`.

## VIDEO GAMES ABSENCE
- Forbidden operational Video Games references: 0.

## HISTORICAL FREEZE MAPPING
- D1R2 six detached-worktree historical groups are preserved.
- Historical phase-snapshot tests are testable in frozen commit contexts and are not required to pass on current master.

## GIT STATUS
- Branch: `master`.
- HEAD: `adbf060f93872f6a75a32d38bbfe34ce30ed059e`.
- Porcelain rows after D1R3 materialization: 286.

## F0 READINESS
- CURRENT_TREE_VALID: `true`.
- HISTORICAL_FREEZE_MAPPING_COMPLETE: `true`.
- D1_INTEGRITY_VALID: `true`.

D1R3_CURRENT_TREE_VALIDATED
