# Phase 6CLEAN Repository Pruning Report

## Decision
Decision: `balanced_reversible_repository_pruning_complete`.

## Initial State
Branch: `plan-d-extensions`.
HEAD: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`.
Initial project files: `1245`.
Initial project directories: `78`.
Initial project size bytes: `5938027963`.

## Dry-Run Classification
Keep files: `1142`.
Archive files: `0`.
Technical delete files: `103`.
Archive bytes: `0`.
Technical delete bytes: `3670341`.

## Archive
Archive root: `../IRP_BeyondUtility_Project_ARCHIVE_PRE_PRUNING/`.
Archive files preserve original relative paths under `files/`.

## Final State
Final project files: `1144`.
Final project directories: `66`.
Final project size bytes: `5934841585`.

## Validation
Validation pass/fail: `24/0`.

## Tests
- `pytest targeted Phase 5D/5E/6A/6B/6CLEAN/contracts/parser/prompts/candidate integrity after final Phase 6CLEAN updates: 278 passed, 1 skipped`

## Scientific Claims
No scientific result, prompt, parser, Phase 5E accounting value, or frozen conclusion was changed.

## Phase 6C Handoff
Future final scientific report generation should consume resident canonical artifacts first and consult the external archive only for deep provenance reconstruction.
