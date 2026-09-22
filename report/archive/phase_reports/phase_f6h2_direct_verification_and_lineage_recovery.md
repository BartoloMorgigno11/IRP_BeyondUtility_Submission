# F6H2 Direct Verification And Lineage Recovery

## 1. Objective

Recover or conclusively classify missing evidence prerequisites from F6H1 before any exactness elevation or metric recomputation.

Decision token: `phase_f6h2_evidence_recovery_complete_no_strict_readiness`.

## 2. Runtime Provenance

- Runtime: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_audit_recovery\Scripts\python.exe`
- Runtime classification: `defensible_repaired_clone_exact_package_versions`
- Caveat: repaired package-tree clone only, not exact original executable environment.
- Required Python environment: `PYTHONNOUSERSITE=1`
- No training, LLM inference, Ollama, API calls, downloads, package changes, prompt changes, parser changes, candidate/cohort changes, metric recomputation, staging, or commits occurred.

## 3. Starting Evidence Failures

- Direct paper verification unavailable: `5`
- Incomplete official population lineage: `38`
- Incomplete raw request/response lineage: `11`
- F6H1 component elevations: `0`
- F6H1 numeric-cell elevations: `0`

## 4. Paper Files Inspected

- `paper/2411.00331v1.pdf`: sha256 `A143265AE42F027C079766B1C006C438D698897B732634D78A010EB0A62F0F23`, bytes `4383085`
- `paper/Beyond Utility Evaluating LLM As Recommander.pdf`: sha256 `A143265AE42F027C079766B1C006C438D698897B732634D78A010EB0A62F0F23`, bytes `4383085`
- `paper/traduzione_italiana_beyond_utility_llm_recommender.docx`: sha256 `30E1251171239F96D044A5DF123DC5D5CDE5886C2634A7EFB1A37F98CEBA5F44`, bytes `53299`

The two PDF files are byte-identical. The DOCX translation is local supporting evidence only and is not treated as authoritative paper-page verification.

## 5. PDF Extraction And Verification Methods

- Python module `PyPDF2`: `false`
- Python module `fitz`: `false`
- Python module `pdfminer`: `false`
- Python module `pdfplumber`: `false`
- Python module `pymupdf`: `false`
- Python module `pypdf`: `false`
- Command `gs`: `false`
- Command `magick`: `false`
- Command `mutool`: `false`
- Command `pdftoppm`: `false`
- Command `pdftotext`: `false`

No installed local PDF text extraction or page rendering method is available under the approved runtime, so direct paper-page verification remains unrecovered.

## 6. Documentary Verification Results

- Documentary candidates inspected: `5`
- Documentary candidates directly verified: `0`
- Documentary candidates confirmed paper-unspecified: `0`

## 7. Paper-Unspecified Details

No material detail was confirmed paper-unspecified in F6H2 because direct paper-page inspection could not be performed. All five documentary candidates remain `verification_unavailable`.

## 8. Official Population Lineage Methodology

F6H2 inventoried official SOTA ZIP recommendation outputs, local reconstructed dataset directories, F6H1 target rows, and upstream metric code. It did not assume identity from matching labels or counts.

## 9. Recovered Population Chains

Complete official population lineages recovered: `0`. Repository-only complete population lineages recovered: `0`.

## 10. Population Fingerprint Comparisons

Recommendation-output fingerprints and local reconstructed-directory fingerprints were generated, but all comparisons remain `unverifiable` because no bijective identifier mapping or exact official source-population chain was recovered.

## 11. Incomplete Population Chains

All `38` population-lineage candidates remain partial: source data, filtering, split, eligible users, sample/cohort, target/negative construction, candidate set, candidate order, and metric population are not fully recoverable from the workspace.

## 12. Request/Response Lineage Methodology

F6H2 searched official LLM ZIP entries and local request/response/runner-event surfaces. Official GPT3.5 output entries exist, but no stable request ID, response ID, runner event ID, prompt hash, parser-result ID, or metric-population graph was recovered.

## 13. Reconciled Lineages

Complete request/response lineages recovered: `0`.

## 14. Orphan And Duplicate Findings

- `responses_without_requests`: `226`; admitted `false`
- `parsed_or_metric_rows_without_request_graph`: `105`; admitted `false`
- `recommendation_outputs_without_source_population_chain`: `26`; admitted `false`
- `empty_artifacts`: `0`; admitted `false`
- `duplicate_response_or_output_hashes`: `2`; admitted `false`

No orphan, duplicate, empty, transport-only, or superseded artifact was admitted as exact evidence.

## 15. Source-Code Lineage

Source-code lineage is recorded in `results/tables/phase_f6h2_source_code_lineage.csv`. Upstream metric and generation code exists, but historical producer-code provenance for the official ZIP outputs remains unresolved.

## 16. Evidence Recovery Decisions

Each of the `54` F6H1 candidates has exactly one recovery decision. Decision classes: `{'lineage_partial_only': 49, 'unresolved': 5}`.

## 17. Recomposition Readiness

Readiness classes: `{'not_ready_missing_paper_specification': 5, 'not_ready_missing_population_lineage': 38, 'not_ready_missing_request_response_lineage': 11}`.

## 18. Candidates Ready For Strict Recomposition

Candidates ready for strict offline recomputation: `0`.

## 19. Candidates Ready Only For Repository-Derived Recomposition

Candidates ready only for repository-derived recomputation: `0`.

## 20. Candidates Still Blocked

Candidates still blocked: `54`.

## 21. Permanently Unrecoverable Evidence

Candidates classified permanently unrecoverable from workspace: `0`. F6H2 stops short of permanent classification because external/manual provenance might still be supplied later.

## 22. Threats To Validity

Direct PDF verification is unavailable with installed tooling. Official ZIPs expose outputs but not complete execution lineage. Local reconstructed data is useful evidence but cannot prove paper-exact official population identity without bijective mapping and chain provenance.

## 23. Files Created

F6H2 config, builder, wrapper, tests, report, tables, and manifests were created under the `phase_f6h2` namespace.

## 24. Files Modified

Only current-state artifacts were updated: `report/future_operations_context_handoff.md` and `results/manifests/current_project_context_snapshot.json`.

## 25. Commands Executed

- Read attachment request with `Get-Content`.
- Verified branch and HEAD with `git rev-parse`.
- Captured Git status with `git status --porcelain=v1`.
- Checked `phase_f6h2` namespace using `rg --files` with cache exclusions.
- Inspected F6H1 candidate, documentary, raw, component, and numeric tables with the approved runtime.
- Inspected paper hashes and local PDF extraction/rendering tool availability.
- Inspected official ZIP inventories, entry counts, response-like files, and producer code paths.
- Ran the F6H2 builder with `PYTHONNOUSERSITE=1 .venv_audit_recovery/Scripts/python.exe`.
- Ran focused tests with `PYTHONNOUSERSITE=1 .venv_audit_recovery/Scripts/python.exe -m pytest tests/test_phase_f6h2_direct_verification_and_lineage_recovery.py -q -p no:cacheprovider`.

## 26. Focused Tests

Focused test result recorded for this work package: `10 passed`. The full repository suite is excluded.

## 27. Determinism

Determinism result: `passed_by_focused_test_suite_two_run_hash_comparison`. The focused tests run the builder twice and compare all required deterministic output hashes. The manifest self hash is recorded as `self_reference_excluded_for_determinism`.

## 28. Protected-Input Invariance

Protected input hashes changed: `0`; missing: `0`.

## 29. Initial And Final Git State

- Branch: `plan-d-extensions`
- HEAD: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`
- Phase baseline rows: `1192` reconstructed from final status minus `20` F6H2 paths.
- Baseline counts: modified `3`, untracked `1189`.
- Final rows: `1212`
- Final counts: modified `3`, untracked `1209`.

## 30. Decision Token

`phase_f6h2_evidence_recovery_complete_no_strict_readiness`

## 31. Recommended Next Work Package

- `F6H2-NEXT-001`: Use an approved local PDF extraction/rendering method or manual visual page review to verify the five documentary facts.
- `F6H2-NEXT-002`: Recover official processed Beauty data/sample/cohort/candidate lineage or mark SOTA cells repository-output-only forever.
- `F6H2-NEXT-003`: Recover official request/response/runner-event manifests for GPT3.5 ZIP outputs, including prompt hashes and parser linkage.
