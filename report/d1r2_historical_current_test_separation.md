# D1R2 Historical Freeze vs Current Extension Test Separation

## FAILURE CLASSIFICATION
- D1R lastfailed nodes classified: 145 (102 failed, 43 errors).
- Classification counts: {'CURRENT_REGRESSION': 4, 'DESTRUCTIVE_OR_PERSISTENT_SIDE_EFFECT': 31, 'HISTORICAL_FREEZE_VALIDATION': 87, 'HISTORICAL_REPRODUCTION_LOGIC': 23}.
- UNKNOWN classifications: 0.

## HISTORICAL FREEZE FAILURES
- Historical freeze validation nodes: 87.
- These remain excluded from the current suite and must be validated in detached worktrees at their expected commits.

## REAL CURRENT REGRESSIONS
- Current regression nodes: 4.
- The direct-script import regression in `build_workspace_forensic_state_audit.py` was fixed.
- A new current blocker remains: exactness fingerprint reads a mutable forensic inventory regenerated during D1R2.

## DIRECT SCRIPT IMPORT FAILURES
- `src/analysis/build_workspace_forensic_state_audit.py --help` exits 0 after the bootstrap fix.
- Full forensic pytest tests were not accepted into the light current suite because they materialize shared audit artifacts repeatedly.

## PERSISTENT SIDE EFFECTS
- Persistent side-effect rows recorded: 23.
- D1R R4/R5 builders rewrote persistent reports/tables, including `report/notes.md`.
- D1R2 forensic pytest attempt regenerated `workspace_relevant_file_inventory.csv`, causing exactness fingerprint row_count 17579 vs expected 136.

## FIXES APPLIED
- Added repository-root bootstrap before `src.audit` imports in `src/analysis/build_workspace_forensic_state_audit.py`.
- No historical expected HEAD/branch/hash/token assertions were weakened.

## CURRENT REGRESSION SUITE
- Included D0A, D0B, D1, H0, E0, E2A, exactness frontier, and a direct forensic `--help` import probe.
- Excluded historical freeze/destructive R/F6 nodes from the current suite.

## CURRENT SUITE RESULT
- Direct forensic import probe: passed.
- Light current pytest suite: 58 passed, 1 failed, 0 errors.
- Blocking issue: tests/test_exactness_frontier_and_evidence_closure.py::test_determinism_and_protected_input_fingerprint observed row_count=17579, expected 136.

## HISTORICAL FREEZE GROUPS
- Historical worktree groups designed: 6.
- Strategy: create one temporary detached worktree per expected commit, run only that group's historical tests, then remove the worktree.
- No historical worktrees were executed in D1R2.

## FOOD.COM INTEGRITY
- Food.com gates in the light current suite passed before the exactness failure.
- D1 token: `D1_FOODCOM_ONLY_OPERATIONAL_TREE_VALIDATED`.

## H0 INTEGRITY
- H0 gates in the light current suite passed before the exactness failure.

## VIDEO GAMES ABSENCE
- D1 active dataset artifact remains available; D1 manifest hash: `f33db07f293ef8321483ba66397bd7ee5135e8053e5af0774303cb844ba36e6e`.
- No Video Games re-materialization was performed.

## GIT STATUS
- Branch: `master`.
- HEAD: `adbf060f93872f6a75a32d38bbfe34ce30ed059e`.
- Porcelain status rows: 284.

## F0 PRECONDITION
- F0 is blocked. Current suite is not green because exactness fingerprint remains contaminated by a mutable forensic inventory.

## NEXT STEP
- Restore or isolate the forensic inventory side effect, then rerun the D1R2 current suite.

D1R2_CURRENT_TREE_BLOCKED
