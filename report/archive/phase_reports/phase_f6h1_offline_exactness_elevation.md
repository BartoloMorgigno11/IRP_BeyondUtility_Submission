# F6H1 Offline Exactness Elevation

## 1. Objective

Convert scientifically justified exactness-frontier rows into demonstrated evidence using only existing offline evidence.

Decision token: `phase_f6h1_offline_exactness_elevation_complete_no_elevations`.

## 2. Runtime Provenance

- Runtime: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_audit_recovery\Scripts\python.exe`
- Runtime classification: `defensible_repaired_clone_exact_package_versions`
- Caveat: repaired package-tree clone only, not exact original executable environment.
- Required Python environment: `PYTHONNOUSERSITE=1`
- No training, LLM inference, Ollama, API calls, downloads, package changes, prompt changes, parser changes, candidate/cohort changes, metric-contract changes, staging, or commits occurred.

## 3. Starting Frontier

- Component candidates: `54`
- Documentary candidates: `5`
- Offline recomputation candidates: `38`
- Existing raw-output candidates: `11`
- Numeric offline/raw candidates: `49`
- LastFM reranking paper rows in target inventory: `0`

## 4. F6G3 Negative Evidence Incorporated

F6G3 remains authoritative negative evidence for LastFM reranking: `7436` expected corrective requests, `1859` logical instances, `4` cells, `0` reconciled raw responses, `0` admitted top-10 rows, and `7436` blocked rows. LastFM reranking is excluded from paper-elevation work because the exactness frontier contains no `LastFM` x `reranking` paper-contract rows.

## 5. Target Inventory

The target inventory is in `results/tables/phase_f6h1_target_candidate_inventory.csv`. Every one of the `54` frontier candidates is represented exactly once.

## 6. Documentary Candidates

The five documentary rows were checked against local paper-file hashes and approved-runtime PDF tooling. The two local PDFs are byte-identical, but no Python PDF module or `pdftotext` executable is available in the approved runtime, so direct page/statement verification remains blocked. Repository evidence was inspected separately and is not used to silently fill paper material details.

## 7. Offline Recomposition Candidates

The 38 offline recomputation candidates point to official Beauty SOTA recommendation CSVs in `original_repo/EvaLLMasRecommender/sota_results/beauty_his10.zip`. These files exist, but exact official test/sample/cohort data, seed/repetition policy, and direct PDF source verification are not fully linked to the target cells. No recomputation was performed.

## 8. Existing Raw-Output Candidates

The 11 existing raw-output candidates point to official Beauty/Sports LLM ZIP files. Candidate JSON/CSV entries exist, but they do not provide complete request ID, response ID, runner-event ID, exact endpoint/version, prompt, cohort, candidate-order, parser, and denominator lineage for exact target-cell admission.

## 9. Raw Evidence Reconciliation

Raw evidence was inventoried read-only. Admitted raw evidence count: `0`.

## 10. Rejected Candidate Evidence

Rejected/unresolved reason counts:

- `direct_paper_verification_unavailable`: `5`
- `incomplete_official_population_lineage`: `38`
- `incomplete_raw_request_response_lineage`: `11`

## 11. Recomposition Methods

Formulas and denominators are recorded in `results/tables/phase_f6h1_offline_recomputation.csv`, but every recomputation row is `blocked_before_recomputation`. No output repair, missing-row substitution, intersection-only full-cohort claim, or zero-fill was used.

## 12. Component Elevation Decisions

Component-elevated rows: `0`. Every target has exactly one row in `results/tables/phase_f6h1_component_elevation_decisions.csv`.

## 13. Numeric-Cell Elevation Decisions

Numeric-cell-elevated rows: `0`. Numerical similarity alone is explicitly disallowed and no numeric cell is exact without exact protocol, population, formula, and rounding evidence.

## 14. Coverage Before And After

- `demonstrated_exact_component_coverage`: `0/192` -> `0/192`; delta `0`; No target passed all exactness gates; no component changed.
- `adapted_reconstructed_coverage`: `192/192` -> `192/192`; delta `0`; Historical adapted/reconstructed coverage preserved; not exact coverage.
- `operational_executable_coverage`: `176/192` -> `176/192`; delta `0`; No execution was run; operational coverage unchanged.
- `documentary_frontier_remaining`: `5/192` -> `5/192`; delta `0`; Documentary candidates remain potential-only because direct PDF verification is unavailable.
- `offline_raw_frontier_remaining`: `49/192` -> `49/192`; delta `0`; Offline/raw candidates remain potential-only; exact lineage and paper verification are incomplete.
- `new_execution_frontier`: `14/192` -> `14/192`; delta `0`; New-execution frontier preserved; F6H1 performed no execution.
- `unavailable_external_resource_blockers`: `100/192` -> `100/192`; delta `0`; No external raw/model/checkpoint evidence was added.
- `paper_unspecified_blockers`: `16/192` -> `16/192`; delta `0`; Paper-unspecified material-detail blockers unchanged.
- `genuine_divergence_rows`: `8/192` -> `8/192`; delta `0`; Adapted-only rows unchanged.
- `exact_numeric_cell_coverage`: `0/187` -> `0/187`; delta `0`; No numeric cell was recomputed under exact comparable protocol.
- `remaining_numeric_offline_raw_frontier`: `49/187` -> `49/187`; delta `0`; Numeric offline/raw candidates remain potential-only.

## 15. Components Elevated

No component was elevated to `exact_paper_replication`.

## 16. Remaining Recoverable Frontier

The prior theoretical frontier remains potential-only: 5 documentary candidates and 49 offline/raw numeric candidates. F6H1 found zero candidates with evidence usable for exact elevation under the strict gate.

## 17. Irreducible Blockers

The dominant blockers are direct paper verification unavailable in the approved runtime, incomplete official raw request/response lineage, incomplete official sample/cohort lineage, exact model/version ambiguity, prompt/parser comparability uncertainty, and metric denominator ambiguity.

## 18. Threats To Validity

This phase is limited by local PDF extraction availability and by official ZIP artifacts that expose outputs but not complete exact experimental lineage. Repository behavior is useful evidence but does not prove paper-explicit material details.

## 19. Files Created Or Modified

Created F6H1 config, builder, wrapper, tests, report, tables, and manifests under the selected phase namespace. Modified current-state artifacts only: `report/future_operations_context_handoff.md` and `results/manifests/current_project_context_snapshot.json`.

## 20. Commands Executed

- Read attachment request with `Get-Content`.
- Verified branch and HEAD with `git rev-parse`.
- Captured Git status with `git status --porcelain=v1`.
- Checked `phase_f6h1` namespace using `rg --files` with cache exclusions.
- Inspected frontier CSVs with the approved recovered Python runtime.
- Inspected PDF tool availability and paper hashes read-only.
- Inspected official ZIP inventories and representative entries read-only.
- Ran the F6H1 builder with `PYTHONNOUSERSITE=1 .venv_audit_recovery/Scripts/python.exe`.
- Ran focused tests with `PYTHONNOUSERSITE=1 .venv_audit_recovery/Scripts/python.exe -m pytest tests/test_phase_f6h1_offline_exactness_elevation.py -q -p no:cacheprovider`.

## 21. Test Results And Determinism

Focused test result recorded for this work package: `10 passed`. The focused tests run the builder twice and compare all required deterministic output hashes. Full repository tests are excluded.

The deterministic output-hash registry is stored in `results/manifests/phase_f6h1_offline_exactness_elevation_manifest.json`; the manifest's own self hash is recorded as `self_reference_excluded_for_determinism`.

## 22. Protected Input Verification

Protected input hashes were captured before and after the builder. Changed: `0`; missing: `0`.

## 23. Initial And Final Git State

- Branch: `plan-d-extensions`
- HEAD: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`
- Phase baseline rows: `1176` reconstructed from final status minus `16` F6H1 paths.
- Baseline counts: modified `3`, untracked `1173`.
- Final rows: `1192`
- Final counts: modified `3`, untracked `1189`.

## 24. Recommended Next Work Package

- `F6H1-NEXT-001`: Perform manual visual PDF/appendix verification or add an approved offline PDF extraction tool, then rerun F6H1.
- `F6H1-NEXT-002`: Locate exact request/response/runner-event lineage for released LLM raw outputs; do not use ZIP existence alone.
- `F6H1-NEXT-003`: Locate exact official test/sample/cohort files and repetition policy for Beauty SOTA CSVs before read-only metric recomputation.
