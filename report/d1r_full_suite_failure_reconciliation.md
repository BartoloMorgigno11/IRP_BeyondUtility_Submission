# D1R Post-Purge Full-Suite Failure Reconciliation

Decision token: `D1R_FULL_SUITE_BLOCKED`

## Baseline

- Branch: `master`
- HEAD: `adbf060f93872f6a75a32d38bbfe34ce30ed059e`
- D1 manifest SHA256: `f33db07f293ef8321483ba66397bd7ee5135e8053e5af0774303cb844ba36e6e`
- D1 removed ledger SHA256: `9d5574e9091484de282c723c2c6721c3b4d8101ace6d24726204dadaf8fdb7a6`
- Raw F2 protected status before fixes: `checked=1144;changed=275;missing=52;hash_mismatch=223;existence_mismatch=52`

## Root Causes Reconciled

- Exactness-frontier direct script failure: `PREEXISTING_ENTRYPOINT_BUG`; fixed with repository-root bootstrap.
- Observed F2 protected-input failures: historical F2 freeze was being used as a permanent current-tree invariant. D1R now validates frozen historical status separately from current explained deltas without changing `results/tables/finalization_f2_protected_input_hashes_after.csv`.
- All 275 original F2 deltas are represented in `results/tables/d1r_protected_input_changes.csv` with no `UNKNOWN` rows.

## Targeted Gates

- Exactness-frontier tests: `5 passed`.
- Observed finalization protected-input nodes: `13 passed in 980.67s`.
- D1/D0B/D0A/H0/E0/E2A gate: `54 passed in 6.01s`.

## Full Suite

- Command completed: `true`.
- Exit code: `1`.
- Result: `1975 passed`, `102 failed`, `43 errors`, `2 skipped`, `6 warnings`.
- Elapsed: `7507.72s` (`2:05:07`).

## Post-Suite Integrity

- Food.com authoritative hashes unchanged: `true`.
- H0 hashes unchanged: `true`.
- Active datasets: `beauty`, `sports`, `ml1m`, `lastfm`, `foodcom`.
- Forbidden operational Video Games references: `0`.
- Raw F2 protected status: `checked=1144;changed=275`.
- Compatible protected status after full suite: `checked=1144;changed=1`.
- Unresolved protected side effect: `report/notes.md` was rewritten again by the full-suite run.

## Blockers

- Full suite is not green.
- Historical F6I2/R4/R5 and later phase tests still assert freeze-specific HEAD/worktree/report state against the current D1 operational tree.
- Several historical builders write blocked-state reports/tables into the current tree during tests instead of isolated temp output.
- `src/analysis/build_workspace_forensic_state_audit.py` has the same direct-script `src` import bootstrap defect as the exactness builder.

No F0 freeze was attempted. No commit was created.
