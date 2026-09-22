# Exactness Frontier And Evidence Closure

## 1. Decision

Decision: `exactness_frontier_closed_conservative_no_elevation`.

No component is elevated to demonstrated exact paper reproduction. The completed forensic audit remains correct at the strict frontier: exact paper coverage is `0/192`, while many components have adapted or potentially recoverable evidence that must not be combined with exact coverage.

## 2. Runtime And Scope

- Runtime: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_audit_recovery\Scripts\python.exe`
- Runtime classification: `defensible_repaired_clone_exact_package_versions`
- Runtime caveat: repaired package-tree clone only, not exact original executable environment.
- No training, LLM inference, Ollama, API calls, downloads, package changes, staging, or commits were performed.
- Python PDF modules available: `{"PyPDF2": false, "fitz": false, "pdfplumber": false, "pymupdf": false, "pypdf": false}`
- Direct PDF text extraction available in approved runtime: `false`
- Byte-identical local paper PDF groups: `1`

## 3. Inputs Reconciled

Starting point artifacts were the completed forensic audit report, handoff, snapshot, paper contract, traceability, status matrix, claims, numeric reconciliation, lineage, and blocker tables. Direct source inspection covered local paper files, upstream prompt templates, upstream LLM wrappers, upstream parser/metric code, and released upstream ZIP inventories.

## 4. Git State

- Branch: `plan-d-extensions`
- HEAD: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`
- Closure-resume baseline rows: `1147` reconstructed from final status minus `8` new exactness-frontier paths.
- Closure-resume baseline counts: modified `3`, added `0`, deleted `0`, untracked `1144`, other `0`.
- Initial rows before this builder run: `1155`
- Final rows after this builder run: `1155`
- Final counts: modified `3`, added `0`, deleted `0`, untracked `1152`, other `0`.

## 5. Frontier Percentages

- `adapted_reconstructed_coverage`: `192/192` = `100.000000%`; rule: forensic status rows classified adapted or local extension; limitation: This is inherited from the completed forensic audit and is not exact coverage.
- `blocked_by_paper_unspecified_material_detail`: `16/192` = `8.333333%`; rule: components where the paper leaves a material detail unresolved and repository-derived behavior cannot prove paper exactness; limitation: Repository evidence supports executable behavior but not paper-explicit exactness.
- `blocked_by_unavailable_external_resources`: `100/192` = `52.083333%`; rule: components blocked by exact model/service/raw-output/checkpoint absence; limitation: May change only if exact external evidence is supplied.
- `currently_demonstrated_exact_coverage`: `0/192` = `0.000000%`; rule: frontier class exact_already_supported_but_previously_underclassified; limitation: No components were elevated because all material exactness conditions remain unclosed.
- `documentary_recoverable_exact_coverage`: `5/192` = `2.604167%`; rule: components requiring only direct paper/appendix/source verification and no computation; limitation: Potential only; direct PDF parser is unavailable in the approved runtime.
- `genuine_divergence_or_adapted_only`: `8/192` = `4.166667%`; rule: components whose current local evidence is materially adapted or divergent; limitation: These are useful local evidence, not exact paper evidence.
- `numeric_currently_demonstrated_exact`: `0/187` = `0.000000%`; rule: numeric cells with demonstrated exact class; limitation: No numeric cell is exact under the strict frontier.
- `numeric_offline_or_raw_recoverable`: `49/187` = `26.203209%`; rule: numeric cells with official released outputs or recommendation CSVs available for safe recomputation; limitation: Potential only; no numeric values were recomputed here.
- `offline_recomputation_recoverable_exact_coverage`: `49/192` = `25.520833%`; rule: components with existing official raw outputs or recommendation CSVs and no training/inference requirement; limitation: Potential only; values were not recomputed in this package.
- `operational_coverage`: `176/192` = `91.666667%`; rule: forensic status rows with execution_status=completed; limitation: Operational does not imply exact paper comparability.
- `requires_new_local_execution_exact_coverage`: `14/192` = `7.291667%`; rule: components needing a future local run but no LLM/API/download in this package; limitation: Not currently recoverable within the allowed closure work.

## 6. Frontier Class Counts

- `adapted_replication_only`: `8`
- `exact_blocked_by_missing_original_data_or_checkpoint`: `24`
- `exact_blocked_by_paid_or_retired_service`: `48`
- `exact_blocked_by_paper_unspecified_material_detail`: `16`
- `exact_blocked_by_unavailable_model_or_version`: `28`
- `exact_possible_but_requires_new_local_execution`: `14`
- `exact_possible_by_documentary_verification_only`: `5`
- `exact_possible_by_offline_recomputation`: `38`
- `exact_possible_with_existing_raw_outputs`: `11`

## 7. Mismatch Categories

- `adapted_local_implementation`: `8`
- `cohort_candidate_prompt_parser_metric_or_denominator_mismatch`: `14`
- `incomplete_direct_paper_verification`: `5`
- `paper_unspecified_repository_resolved_detail`: `16`
- `unavailable_exact_model_or_model_version`: `76`
- `unavailable_exact_raw_output_or_checkpoint`: `73`

## 8. Numeric-Cell Frontier

Numeric cells represented exactly once: `187`.

- `adapted_replication_only`: `8`
- `exact_blocked_by_missing_original_data_or_checkpoint`: `24`
- `exact_blocked_by_paid_or_retired_service`: `48`
- `exact_blocked_by_paper_unspecified_material_detail`: `16`
- `exact_blocked_by_unavailable_model_or_version`: `28`
- `exact_possible_but_requires_new_local_execution`: `14`
- `exact_possible_by_offline_recomputation`: `38`
- `exact_possible_with_existing_raw_outputs`: `11`

No numeric cell is currently exact. Numeric cells with potential safe recovery require direct paper verification plus read-only recomputation from existing official raw outputs or recommendation CSVs; this package did not compute new numeric outputs.

## 9. Evidence Elevation Candidates

Elevation candidates without training, LLM inference, API calls, downloads, or protocol changes: `54` in `results/tables/exactness_elevation_candidates.csv`.

These are potential-only. They remain non-exact until all material exactness requirements are demonstrated.

## 10. Irreducible Or External Blockers

Irreducible/external blocker rows: `124` in `results/tables/irreducible_exactness_blockers.csv`.

The dominant blockers are exact service/model version absence, missing original raw outputs or checkpoints, and paper-unspecified output/prompt details that the repository resolves only as executable behavior, not paper-explicit exactness.

## 11. Upstream Repository Findings

- Nested repo HEAD ref: `ref: refs/heads/main`
- Nested repo HEAD commit from direct ref read: `1e12982ca5c32d271cec6d1cb723e071f8c6587c`
- Beauty LLM ZIP entries: `224`; model tokens: `glm4, gpt3.5, gpt4, llama3`
- Sports LLM ZIP entries: `113`; model tokens: `glm4, gpt3.5, gpt4, llama2, llama3`
- Beauty SOTA ZIP entries: `27`; model tokens: `BM25, BPRMF, GRU4Rec, LightGCN, SASRec`
- Official executable contract evidence remains top-10 recommendation output with top-5 metrics, while some template wording remains top-20 or paper-unspecified.

## 12. Fifth-Dataset Requirement

LastFM is a paper dataset and cannot satisfy the university requirement for an additional dataset. The future fifth-dataset phase must use a dataset outside Amazon Beauty, Amazon Sports, ML-1M, and LastFM, and must report its extension percentages separately from exact paper replication.

Requirements are in `results/tables/fifth_dataset_extension_requirements.csv` with `7` rows.

## 13. Recommended Next Work Package

Single next work package: corrective offline LastFM reranking top-10 freeze, preserving F6G1 evidence and regenerating only request/shard manifests under the repository-derived top-10 executable contract. Do not run inference.

## 14. Protected-Input Verification

Protected inputs are represented by the completed forensic inventory fingerprint. The closure builder records the protected aggregate in `results/manifests/exactness_frontier_snapshot.json`; focused tests compare it across builder runs.

## 15. Files Created Or Updated

- `report/exactness_frontier_and_evidence_closure.md`
- `results/tables/exactness_frontier_by_component.csv`
- `results/tables/numeric_cell_exactness_frontier.csv`
- `results/tables/exactness_elevation_candidates.csv`
- `results/tables/irreducible_exactness_blockers.csv`
- `results/tables/fifth_dataset_extension_requirements.csv`
- `results/manifests/exactness_frontier_snapshot.json`
- `report/future_operations_context_handoff.md`
- `src/analysis/build_exactness_frontier_and_evidence_closure.py`
- `tests/test_exactness_frontier_and_evidence_closure.py`
