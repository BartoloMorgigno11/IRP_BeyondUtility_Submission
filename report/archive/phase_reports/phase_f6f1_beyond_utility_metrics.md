# Phase F6F1 Beyond-Utility Metrics Recovery

## 1. Initial repository state
Branch `plan-d-extensions`, HEAD `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`, staged files `none`. Tracked dirty state remains `M README.md
 M report/dataset_audit.md
 M report/notes.md`.

## 2. Frozen-input verification
Checked `41` configured inputs and `1144` protected rows; changed `0`, missing `0`.

## 3. Paper metric source inventory
Rows written: `8`. PDF is authoritative; DOCX is secondary structured evidence.

## 4. Metric contracts
Rows written: `7` with exact/adapted/reconstructed/blocked classifications.

## 5. Exactness classifications
Rows written: `14`. No proxy is promoted to paper-exact.

## 6. APLT results
Rows written: `126`. Invalid LLM lists are excluded for conditional and operational is not defined.

## 7. Serendipity results
Rows written: `110` using frozen MostPop references where paired.

## 8. Popularity and novelty results
Rows written: `884`. ARP, novelty, coverage, and reliability remain separate metrics.

## 9. Candidate-position results
Rows written: `3083`. F6E4C position analysis is observational; CandDif is blocked for full matrix.

## 10. CandDif status
Plan D diagnostic CandDif is reported as diagnostic only; F6E4C full-matrix CandDif is `blocked_missing_input`.

## 11. Ranking-vs-reranking results
Rows written: `97`. LastFM re-ranking is excluded.

## 12. LLM-vs-baseline results
Rows written: `1008` using paired intersections and bootstrap seed 42.

## 13. Structural diagnostics
Rows written: `752`. These rows keep reliability, positive-presence, duplicate-rate, and operational utility diagnostics separate from paper metric claims.

## 14. Invalid-output policies
Utility operational invalid equals zero. List beyond-utility metrics use strict-valid conditional rows; operational list values are `not_defined` for invalid LLM outputs.

## 15. Updated research-question evidence
Rows written: `8`.

## 16. Implementation files
`src/analysis/plan_f6f1_beyond_utility_metrics.py`, `src/experiments/build_phase_f6f1_beyond_utility_metrics.py`, and `configs/phase_f6f1_beyond_utility_metrics.yaml`.

## 17. Tests added
`tests/test_plan_f6f1_beyond_utility_metrics.py`.

## 18. Regression results
Recorded by operator command log; unsafe suites are excluded by config.

## 19. Excluded unsafe tests
training, cuda_required, downloads, url_access, ollama, remote_api, fresh_inference

## 20. Determinism
Builder preserves first-run timestamp in the freeze manifest and writes sorted CSV/JSON outputs.

## 21. Protected-input verification
Protected summary: `{'checked': 1144, 'changed': 0, 'missing': 0}`.

## 22. Freeze manifest
`results/manifests/phase_f6f1_beyond_utility_freeze.json` with scientific content hash `4bf9391b835dad47c80f41b9d41b130f64ac3f4776561cc19bf0a3d044ffb37c`.

## 23. Missing or blocked evidence
Rows written: `5`.

## 24. Final decision
`f6f1_beyond_utility_metrics_complete_with_documented_blocks`.

## 25. Recommended next action
review phase_f6f1_missing_or_blocked.csv before any PDF/manual exactness upgrade

## 26. Final git state
No git add, commit, reset, clean, checkout, or protected input modification is performed by the builder.

## 27. Files created or modified
Only new `phase_f6f1_*` files plus the F6F1 config/code/test files.

## 28. Remaining limitations
Paper-exact claims remain limited by PDF extraction, adapted local candidate populations, and missing controlled full-matrix CandDif permutations.
