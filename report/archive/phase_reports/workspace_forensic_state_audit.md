# Workspace Forensic State Audit

## 1. Executive summary

Decision: `forensic_audit_completed_with_conservative_classification`.

This audit restarted from the beginning after explicit approval to use the recovered interpreter. It recomputed inventories, phase/supersession records, paper-to-local traceability, claim checks, status matrices, lineage, numeric reconciliation placeholders, blockers, and a context snapshot from current workspace evidence.

The maximum defensible statement is conservative: the project contains substantial adapted/reconstructed local evidence, but this audit found `0` exact paper-replication status rows under the strict gate because exact model versions, exact prompts, exact runtime, PDF verification, and several protocol factors remain unresolved or adapted.

## 2. Audit scope and limitations

- No training, LLM inference, Ollama invocation, API calls, package installs, downloads, staging, or commits were performed.
- The original `.venv` was not modified.
- Current audit runtime: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_audit_recovery\Scripts\python.exe`.
- Runtime classification: `defensible_repaired_clone_exact_package_versions`.
- Runtime caveat: this is not exact original executable-environment reproduction.
- Paper PDF extraction remains partial; many paper rows derive from existing structured artifacts and are marked accordingly.

## 3. Git and environment state

- Branch: `plan-d-extensions`
- HEAD: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`
- Audit-resume baseline status rows: `1134` reconstructed from final status minus `13` new complete-audit paths.
- Audit-resume baseline counts: modified `3`, added `0`, deleted `0`, untracked `1131`, other `0`.
- Initial status rows captured before this builder execution: `1147`.
- Initial-before-builder counts: modified `3`, added `0`, deleted `0`, untracked `1144`, other `0`.
- Final status rows captured by builder: `1147`.
- Final counts: modified `3`, added `0`, deleted `0`, untracked `1144`, other `0`.
- Python executable: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_audit_recovery\Scripts\python.exe`
- Python version: `3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]`
- Package-tree recovery evidence: `results/manifests/python_environment_recovery_snapshot.json`

## 4. Repository architecture

Relevant file inventory rows: `1852`.

Category counts:

- `config`: `24`
- `docs`: `3`
- `local_data`: `57`
- `manifest`: `16`
- `model_checkpoint`: `40`
- `official_repository`: `27`
- `paper`: `3`
- `prediction`: `4`
- `raw_data`: `5`
- `raw_llm_output`: `23`
- `report`: `116`
- `result_table`: `959`
- `root_or_other`: `186`
- `source_code`: `251`
- `test`: `138`

## 5. Source-authority hierarchy

The audit used this hierarchy: raw inputs and raw outputs; executable source/configs; frozen manifests and hashes; validation outputs; derived CSV/JSON; reports; README and notes. Narrative claims were not trusted without cross-reference.

## 6. Relevant file inventory

Full inventory: `results/tables/workspace_relevant_file_inventory.csv`.

The inventory excludes `.git`, virtualenv package trees, Python runtime package trees, cache directories, and pytest temporary logs, while retaining scientific data, reports, manifests, tables, source, tests, raw LLM outputs, and checkpoints.

## 7. Phase and supersession reconstruction

Phase registry rows: `26` in `results/tables/workspace_phase_and_supersession_registry.csv`.

Key supersession: F6G1/F6G2/F6G2R top-20 LastFM reranking execution artifacts are not current executable contracts after `phase_f6g2_output_length_contract_resolution`; the current executable rule is repository-derived top-10 output over 20 candidates with top-5 metrics.

## 8. Paper contract

Paper atomic contract rows: `192`. Numeric paper cells represented: `187`.

Rows sourced from F5AF remain marked `secondary_structured_source_pdf_unverified` where the prior extraction did not directly verify PDF pages. The paper contract is separated from official repository semantics.

## 9. Official repository contract

The official repository snapshot under `original_repo/EvaLLMasRecommender` remains protected. Normal nested Git commands are blocked by dubious ownership unless Git config is changed, so this audit uses source-tree evidence and direct `.git` metadata only where available.

Recovered official executable semantics from prior and current evidence include candidate pool size 20, parser/evaluator top-10 behavior for executable LLM outputs, and top-5 metrics. LLMRank template wording remains a repository-internal conflict because template text can ask for all 20 while parser/evaluator behavior consumes top 10.

## 10. Paper-versus-repository conflicts

Conflict rows: `8` in `results/tables/paper_repository_contract_conflicts.csv`.

Material conflicts include paper-unspecified output length versus repository-derived top-10 executable behavior, runtime provenance differences, and incomplete direct PDF verification.

## 11. Actual local pipeline

Pipeline evidence is reconstructed through data files, configs, source, manifests, raw outputs, parser/result tables, and reports. The local pipeline is adapted/reconstructed, not exact paper execution. LastFM is a paper dataset locally reconstructed/adapted, not a fifth-dataset extension.

## 12. Dataset and cohort audit

Dataset/task/model status is in `results/tables/current_reproduction_status_matrix.csv`. LastFM ranking has adapted local evidence; LastFM reranking LLM execution remains blocked until a corrective top-10 offline freeze exists.

## 13. Candidate-set audit

Candidate evidence is represented in inventory and lineage tables. The current candidate/output contract is: 20-item candidate pools where available; repository-derived 10-item LLM returned list; metric cutoff `K=5`. Positive inclusion and candidate-order invariants are not recomputed by costly experiment reruns in this audit.

## 14. Model audit

The status matrix separates open-weight local substitutes (`Qwen2`, `Mistral`) from exact paper LLM endpoints. Exact paper LLM-version coverage is not claimed. Traditional baselines are local or repository-derived reimplementations unless exact official outputs are traceably matched.

## 15. Prompt and parser audit

Prompt/parser contracts are traced through canonical LLM tables, F6E prompt recovery artifacts, F6E3B/F6E4 adapted indexed prompts, and F6G2 output-length resolution. The local JSON/index prompt is classified as adapted and must not be called exact paper prompting.

## 16. Output-length contract audit

Current resolution: `D_repository_derived_top10_paper_unspecified`. F6G1/F6G2 full-20 request manifests are preserved but superseded for execution.

## 17. Metric audit

Metric evidence is traced through `canonical_metric_*`, F5B-F5E, F6F1, and local utility tables. Unsupported or unresolved paper-exact metrics remain blocked or adapted rather than zero-filled.

## 18. Experimental evidence audit

Evidence exists for adapted local LLM and baseline runs, especially canonical local LLM, F6D2 LastFM trainable models, and F6D4A LastFM LLM ranking. No new experimental evidence was generated by this audit.

## 19. Numerical reconciliation

Numeric reconciliation rows: `187` in `results/tables/numeric_result_reconciliation.csv`.

Exact paper numeric-cell coverage is `0/187 = 0.000000%` under the current strict gate. Existing local mappings are preserved as evidence but not upgraded to exact paper reproduction.

## 20. Claim-verification findings

Claim verification rows: `250` in `results/tables/project_claim_verification.csv`.

High-risk claims include any wording that implies full reproduction, exact numerical reproduction, exact prompt reproduction, or 100% paper coverage. Stale historical blocker records are preserved but superseded where later evidence resolves them.

## 21. Exact reproduction status

Exact paper-component coverage: `0/192 = 0.000000%`.

This is intentionally strict. Completed adapted work does not count as exact paper replication.

## 22. Reconstructed/adapted status

Adapted/reconstructed component coverage: `192/192 = 100.000000%`.

## 23. Local extension status

The university-extension contract is partially supported by added metrics and local adapted analyses. A true fifth dataset beyond the paper is not verified because LastFM is a paper dataset.

## 24. University-requirement status

The project has documented code, saved logs/results, paper-vs-local comparison artifacts, limitations, and report-ready evidence. Extension claims must distinguish added metrics/local analyses from exact paper reproduction.

## 25. Supported scientific claims

- The project is not a complete exact numerical replication of the paper.
- The current executable LLM output-length contract is repository-derived top-10, not paper-explicit.
- Local Qwen2/Mistral evidence is adapted open-weight/local evidence, not exact paper model execution.
- The recovered runtime is scientifically defensible for audit execution but not exact original runtime reproduction.

## 26. Unsupported or overstated claims

- 100% paper reproduction.
- Exact prompt reproduction for local JSON/index prompts.
- Exact model reproduction for local substitutes.
- Full LastFM reranking LLM execution after F6G2R without a corrective top-10 freeze.
- Paper-exact fairness/diversity/serendipity where formula inputs and denominators remain unresolved.

## 27. Stale and superseded reports

Reports with `final`, `canonical`, or earlier blocker labels are not automatically current. The lineage table records superseded artifacts, especially Plan D/F6B coverage taxonomy and F6G1/F6G2R top-20 reranking artifacts.

## 28. Genuine blockers

Blockers are listed in `results/tables/blockers_gaps_and_next_actions.csv`. The top blocker is exact paper-runtime provenance; the top executable next work package is a corrective LastFM reranking top-10 freeze, not inference.

## 29. Unresolved ambiguities

- Direct PDF verification of all paper cells.
- Exact paper LLM versions and endpoint availability.
- Full official repository provenance for nested repo cleanliness without changing Git safe-directory config.
- Some denominator/formula details for appendix metrics.

## 30. Multiple progress percentages

- `adapted_or_reconstructed_component_coverage`: `192/192` = `100.000000%`; rule: status rows classified adapted or local extension
- `exact_paper_component_coverage`: `0/192` = `0.000000%`; rule: status rows classified exact_paper_replication
- `operational_executable_coverage`: `176/192` = `91.666667%`; rule: status rows completed
- `paper_contract_rows_recovered`: `192/192` = `100.000000%`; rule: rows represented in paper_atomic_contract.csv; not all PDF-verified
- `paper_numeric_cell_exact_coverage`: `0/187` = `0.000000%`; rule: numeric cells protocol-exact verified in current evidence

## 31. Maximum defensible completion statement

The project has a substantial adapted/reconstructed reproducibility package, but exact paper replication remains `0%` under the strict atomic matrix used here. Operational/adapted evidence is meaningfully broader than exact evidence and must be reported separately.

## 32. Threats to validity

This audit uses available local artifacts and does not rerun costly experiments. Paper PDF extraction is incomplete. Git state is dirty and many artifacts are untracked. Runtime is a repaired package-tree clone rather than the original Store base.

## 33. Recommended single next work package

Create a corrective offline LastFM reranking top-10 freeze that preserves F6G1 evidence but regenerates request/shard manifests with `candidate_pool_size=20`, `requested_output_length=10`, `parser_expected_length=10`, and `metric_cutoff_K=5`. Do not run inference in that work package.

## 34. Files created

Current audit outputs are the report, handoff, 10 CSV tables, one JSON snapshot, builder, and focused tests.

## 35. Commands executed

Commands are summarized in the final response and in the JSON snapshot. Python commands used only `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_audit_recovery\Scripts\python.exe` with `PYTHONNOUSERSITE=1`.

## 36. Test results

Focused audit tests: `passed`.

- Command: `PYTHONNOUSERSITE=1 .\.venv_audit_recovery\Scripts\python.exe -m pytest tests\test_workspace_forensic_state_audit.py -q`
- Passed: `4`
- Failed: `0`
- Warnings: `1`
- Duration seconds: `81.53`
- Excluded tests: `full_project_suite`
- Exclusion reason: `Excluded because the repository contains training/checkpoint/writer paths and the audit scope forbids full-suite execution.`

## 37. Determinism results

Determinism status: `passed`.

- Command: `Run build_workspace_forensic_state_audit.py twice with PYTHONNOUSERSITE=1 and compare SHA-256 hashes of the report, handoff, required CSV tables, and context snapshot.`
- Checked artifacts: `13`
- Mismatches: `0`
- Notes: `Current audit outputs use fixed self-reference markers for current audit artifacts to avoid self-referential inventory nondeterminism.`

## 38. Protected-input verification

Protected inputs were not intentionally modified. Current audit artifacts are isolated. Existing recovery artifacts are runtime provenance, not scientific experimental outputs.

- Protected hash-invariance status: `passed`
- Protected/manifested rows compared: `136`
- Protected hash check command: `Compare protected_input and protected_or_manifested inventory SHA-256 maps before and after a read-only builder run with PYTHONNOUSERSITE=1 .\.venv_audit_recovery\Scripts\python.exe`

## 39. Initial and final Git status

The audit-resume baseline was rechecked before implementation for branch and HEAD. The exact live raw status listing was not frozen before artifact generation; this report therefore records a deterministic reconstructed baseline from final status by removing only the new complete-audit paths. Initial and final status are dirty. No staging or commit was performed.

Audit-created status entries:

- `results/tables/artifact_lineage_and_supersession.csv`
- `results/tables/current_reproduction_status_matrix.csv`
- `results/tables/numeric_result_reconciliation.csv`
- `results/tables/paper_atomic_contract.csv`
- `results/tables/paper_repository_contract_conflicts.csv`
- `results/tables/paper_to_local_traceability.csv`
- `results/tables/project_claim_verification.csv`
- `results/tables/workspace_forensic_audit_determinism.csv`
- `results/tables/workspace_forensic_audit_test_results.csv`
- `results/tables/workspace_phase_and_supersession_registry.csv`
- `results/tables/workspace_relevant_file_inventory.csv`
- `src/analysis/build_workspace_forensic_state_audit.py`
- `tests/test_workspace_forensic_state_audit.py`
