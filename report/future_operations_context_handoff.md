# Future Operations Context Handoff

## Current Full-Research State - 2026-09-06

- Reproduction status: `LOCAL_ADAPTED_REPRODUCTION_COMPLETE_NOT_PAPER_EXACT`.
- F6M reproduction core: complete and frozen with documented limitations.
- Extension status: G1 primary ranking, G2 history-length sensitivity, G3 candidate-position sensitivity, and G4 F1BZQ reranking are complete and frozen as adapted local extension evidence.
- Delivery branch: `delivery/full-research`.
- Pre-GitHub scientific checkpoint: `cea96181cb78b249f5aadc490d6715d89e86ebde`.
- GitHub package manifest: `results/manifests/github_packaged_artifacts.json`.
- Commit provenance map: `report/GITHUB_PROVENANCE_MAP.md`.
- Raw G1-G4 LLM response evidence and `archive_external/` ZIP bundles remain outside normal Git.
- No professor-facing curated submission branch has been created.

Historical phase handoffs below are retained for lineage and may describe states that were later superseded.

- Latest scientific phase: `phase_f6j_scientific_evidence_synthesis`.
- Decision token: `f6j_scientific_evidence_synthesis_complete`.
- Cleanup preflight before synthesis: exact F6J `cleanup_current and not unsafe_external` gate passed with `140 passed, 1698 deselected`.
- Current cleanup gate after F6J artifacts and guard registration: `151 passed, 1698 deselected`.
- Cleanup status: terminally closed from F6I2P; remaining YAML cleanup is closed without physical YAML changes and repository cleanup is closed as `terminal_retain_current_compatibility_architecture`.
- Current guard note: F6J tables are registered as terminal later scientific artifacts; the F6G3 runtime manifest rewrite is accepted only when its blocked/no-inference/no-API semantics remain intact.
- F6J evidence policy: use existing validated CSVs and manifests only; no training, inference, API calls, downloads, package mutation, raw-output repair, Git staging, or commit were performed.
- F6J artifacts:
  - `results/tables/phase_f6j_research_question_evidence_matrix.csv`
  - `results/tables/phase_f6j_scientific_claim_registry.csv`
  - `results/tables/phase_f6j_paper_reproduction_extension_matrix.csv`
  - `results/tables/phase_f6j_limitations_and_threats.csv`
  - `report/phase_f6j_scientific_evidence_synthesis.md`
  - `tests/test_phase_f6j_scientific_evidence_synthesis.py`
- Research-question status summary: supported `4`; contradicted `3`; mixed/context-dependent `4`; unresolved `3`.
- Next phase: scientific execution planning can proceed from the F6J registry and matrix; do not reopen cleanup unless a future current gate actually fails.

## F6K Scientific Gap Readiness

- Latest scientific gate after F6J: `phase_f6k_scientific_gap_readiness`.
- F6K decision token: `f6k_scientific_gap_readiness_complete`.
- F6K cleanup_current preflight: `151 passed, 1698 deselected in 404.58s` with `.venv_audit_recovery`, `PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, and marker expression `cleanup_current and not unsafe_external`.
- F6K decisions: `profile_generation_effectiveness` -> `blocked_missing_frozen_inputs`; `runtime_cost_characterization` -> `execute_now_high_value`; `lastfm_llm_reranking` -> `execute_now_limited_value`.
- F6K execution policy: no training, no LLM inference, no Ollama start, no API calls, no downloads, no package changes, no raw-output regeneration, no YAML/archive/hash cleanup, and no `original_repo/` edits.
- Recommended order: runtime/cost characterization first; LastFM reranking only as separate explicitly authorized adapted local inference; profile generation remains blocked.
- Finalization can proceed if optional LastFM reranking is not required before the report.

<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation:start -->
## F6L Paper-Faithful Completion Authorization Checkpoint

- Latest scientific checkpoint: `phase_f6l_paper_faithful_completion_and_rq_revalidation`.
- Decision token: `f6l_execution_matrix_partially_blocked_by_missing_official_artifacts`.
- Previous checkpoint token: `f6l_paper_faithful_execution_checkpoint_ready`.
- No inference, Ollama start, paid API call, download, training, cleanup, staging, commit, raw-output overwrite, or `original_repo/` mutation was performed.
- Existing LastFM ranking is retained as adapted evidence; existing LastFM reranking remains adapted/no-raw and is not paper-faithful.
- Recalculated future local open-model request count after explicit approval: `8800`.
- Request composition: profile generation `800`, history-only `800`, profile-only `1600`, profile-plus-history `1600`, LastFM ranking `4000`, LastFM reranking `0`.
- Paper-faithful LastFM reranking request count remains `0` because official LastFM BPRMF/GRU4Rec/SASRec/LightGCN source top-K snapshots and the resulting final 20-item pool identity were not recovered.
- Official recovery audit: Beauty/Sports and Beauty SOTA ZIPs are already present locally with hashes; no new downloads were performed; no official LastFM sample/seed/top-K/pool artifact was found.
- Current guard note: F6L tables/manifests remain isolated terminal checkpoint artifacts.
<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation:end -->

<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_execute_preflight:start -->
## F6L Authorized Execute Preflight

- Decision token: `f6l_execution_blocked_by_contract_runtime_or_protected_input_regression`.
- Planned/completed/strict-valid/failed/residual requests: `8800`/`0`/`0`/`0`/`8800`.
- Local-only Ollama model identity check was performed against `http://127.0.0.1:11434/api/tags`.
- No inference, raw-output directory creation, paid/remote API call, staging, commit, cleanup, download, training, or `original_repo/` mutation was performed.
- Execution blocker checks: per_request_manifest_exists_and_unique, lastfm_frozen_sample_artifact.
- Next action stays inside F6L: produce and freeze the per-request open-model manifest and F6L LastFM 1000-user K-S sample, then rerun execute preflight before any model call.
<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_execute_preflight:end -->

<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_contract_materialization:start -->
## F6L Execution Contract Materialization

- Decision token: `f6l_execution_contract_materialization_blocked_by_missing_paper_specification`.
- Materialization stopped before creating the LastFM 1000-user sample, candidate sets, or 8800-row request manifest.
- Reason: recovered sources do not define an operational K-S gate with exact feature/metric vector, threshold/alpha, and decision rule; LastFM temporal leave-one-out ordering is also not certifiable from verified source artifacts.
- No inference, Ollama generation, remote API call, raw-output directory, download, training, staging, commit, cleanup, or `original_repo/` mutation was performed.
- Blocking contract areas: ks_gate_operational_definition; lastfm_temporal_leave_one_out_order.
<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_contract_materialization:end -->

<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_separated_checkpoints:start -->
## F6L Separated Profile And LastFM Checkpoints

- Decision token: `f6l_profile_execution_ready_lastfm_operationalization_checkpoint_ready`.
- Profile manifest: `4800` rows, `4800` unique request IDs, no LastFM rows.
- Profile execution is separately ready for explicit local authorization; no inference was run.
- LastFM exact reproduction remains blocked by missing author/paper specification.
- LastFM local operationalization and full-cohort extension are preregistered plans only and cannot support/contradict paper RQs alone.
- LastFM status: exact_lastfm_blocked_operationalization_and_full_cohort_planned_not_executed.
<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_separated_checkpoints:end -->

<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_profile_open_model_execution:start -->
## F6L Profile Open-Model Execution

- Decision token: `f6l_profile_open_model_execution_and_rq_revalidation_complete`.
- Run ID: `20260801_010000`.
- Raw output directory: `results/llm/phase_f6l_profile_execution_20260801_010000`.
- Planned/completed/strict-valid/failed/residual: `4800`/`4800`/`807`/`0`/`0`.
- LastFM, remote API, paid models, downloads, training, staging, and original_repo mutation remain out of scope.
<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_profile_open_model_execution:end -->

<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_lastfm_execution_contracts:start -->
## F6L LastFM Execution Contracts

- Decision token: `f6l_lastfm_execution_contracts_ready_for_inference_authorization`.
- Profile generation remains terminally closed and was not reopened.
- Exact paper LastFM remains `blocked_missing_original_specification`.
- Local 1000-user operationalized LastFM sample is separate from paper-exact evidence.
- Full 1859-user cohort is separate extension evidence.
- Operationalized ranking manifests: `4000` rows.
- Full-cohort ranking manifests: `7436` rows.
- Reranking readiness keeps paper-exact blocked and local pools separate: operationalized `4000` future requests, full cohort `7436` future requests.
- No inference or raw output directory was created.
<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_lastfm_execution_contracts:end -->

<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_lastfm_operationalized_ranking_execution:start -->
## F6L LastFM Operationalized Ranking Execution

- Decision token: `f6l_lastfm_operationalized_ranking_execution_and_rq_revalidation_complete`.
- Run ID: `20260802_184502`.
- Raw output directory: `results/llm/phase_f6l_lastfm_operationalized_ranking_20260802_184502`.
- Planned/completed/strict-valid/failed/residual: `4000`/`4000`/`0`/`4000`/`0`.
- Scope executed: operationalized 1000-user LastFM ranking only.
- Full cohort, reranking, LLMSRec, profile prompts, remote APIs, downloads, and training were not executed.
<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_lastfm_operationalized_ranking_execution:end -->

<!-- phase_f6l_lastfm_parser_forensics:start -->
## F6L LastFM Parser Forensics

- Decision token: `f6l_lastfm_ranking_reparse_complete_with_conservative_parser_recovery`.
- Run audited: `20260802_184502`.
- Parser V2 hash: `AFCF8936A3C8A6133A2EA76C18E2BDA36C2FA1668D66EC60E262A78EEB1F906F`.
- Raw protection status: `pass`.
- No rerun is authorized or executed by this phase.
<!-- phase_f6l_lastfm_parser_forensics:end -->

## phase_f6l_lastfm_corrective_output_contract_pilot (2026-08-04T10:19:32Z)

- Decision token: `f6l_lastfm_corrective_output_contract_pilot_failed_redesign_required`
- Run id: `20260804_115330`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_pilot_20260804_115330`
- Scope: corrective V3 JSON/index pilot, 40 local Ollama requests maximum.
- Acceptance: 11/14 checks passed; strict-valid 33/40.
- Original raw unchanged: `AF1ECB11AABFE8BE793888FC244C67DB349C42CE18F8AF7B7060D47167C6E45C` -> `AF1ECB11AABFE8BE793888FC244C67DB349C42CE18F8AF7B7060D47167C6E45C`.
- Scientific boundary: validity diagnostics only; smoke HR/NDCG is not RQ evidence.

## phase_f6l_lastfm_corrective_output_contract_v4_pilot (2026-08-04T10:52:51Z)

- Decision token: `f6l_lastfm_corrective_output_contract_v4_pilot_passed_full_rerun_ready`
- Run id: `20260804_123438`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_pilot_20260804_123438`
- Completed/strict-valid: 40/40 of 40.
- Regression users: `640, 709, 1333, 1749`; holdout users: `223, 387, 2090, 208, 895, 609`.
- Scientific boundary: V4 validity only; smoke HR/NDCG is not RQ evidence.

## phase_f6l_lastfm_corrective_output_contract_v4_full_ranking (2026-08-04T13:02:32Z)

- Decision token: `f6l_lastfm_v4_constrained_full_ranking_stopped_by_contract_circuit_breaker`
- Run ID: `20260804_131959`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_full_ranking_20260804_131959`
- Planned/completed/strict-valid/failed/residual: `4000`/`291`/`290`/`1`/`3709`.
- Metrics rows: `0`; beyond-utility rows: `0`; RQ rows: `0`; acceptance pass: `8/13`.
- Scope boundary: no reranking, no full cohort, no LLMSRec, no remote APIs, no training, no download.

## phase_f6l_lastfm_corrective_output_contract_v4_full_ranking (2026-08-04T17:34:52Z)

- Decision token: `f6l_lastfm_v4_full_ranking_resume_stopped_by_systemic_contract_regression`
- Run ID: `20260804_131959`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_full_ranking_20260804_131959`
- Planned/completed/strict-valid/failed/residual: `4000`/`491`/`487`/`4`/`3509`.
- Metrics rows: `0`; beyond-utility rows: `0`; RQ rows: `0`; acceptance pass: `9/13`.
- Scope boundary: no reranking, no full cohort, no LLMSRec, no remote APIs, no training, no download.

## phase_f6l_lastfm_corrective_output_contract_v4_full_ranking (2026-08-05T07:09:10Z)

- Decision token: `f6l_lastfm_v4_full_ranking_partial_resumable_infrastructure_interruption`
- Run ID: `20260804_131959`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_full_ranking_20260804_131959`
- Planned/completed/strict-valid/failed/residual: `4000`/`3054`/`3020`/`34`/`946`.
- Metrics rows: `0`; beyond-utility rows: `0`; RQ rows: `0`; acceptance pass: `9/13`.
- Scope boundary: no reranking, no full cohort, no LLMSRec, no remote APIs, no training, no download.

## phase_f6l_lastfm_corrective_output_contract_v4_full_ranking (2026-08-05T07:54:31Z)

- Decision token: `f6l_lastfm_v4_full_ranking_partial_resumable_infrastructure_interruption`
- Run ID: `20260804_131959`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_full_ranking_20260804_131959`
- Planned/completed/strict-valid/failed/residual: `4000`/`3054`/`3020`/`34`/`946`.
- Metrics rows: `0`; beyond-utility rows: `0`; RQ rows: `0`; acceptance pass: `8/13`.
- Scope boundary: no reranking, no full cohort, no LLMSRec, no remote APIs, no training, no download.

## phase_f6l_lastfm_corrective_output_contract_v4_full_ranking (2026-08-05T08:08:27Z)

- Decision token: `f6l_lastfm_v4_full_ranking_partial_resumable_infrastructure_interruption`
- Run ID: `20260804_131959`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_full_ranking_20260804_131959`
- Planned/completed/strict-valid/failed/residual: `4000`/`3054`/`3020`/`34`/`946`.
- Metrics rows: `0`; beyond-utility rows: `0`; RQ rows: `0`; acceptance pass: `8/13`.
- Scope boundary: no reranking, no full cohort, no LLMSRec, no remote APIs, no training, no download.

## phase_f6l_lastfm_corrective_output_contract_v4_full_ranking (2026-08-05T08:22:43Z)

- Decision token: `f6l_lastfm_v4_full_ranking_partial_resumable_infrastructure_interruption`
- Run ID: `20260804_131959`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_full_ranking_20260804_131959`
- Planned/completed/strict-valid/failed/residual: `4000`/`3054`/`3020`/`34`/`946`.
- Metrics rows: `0`; beyond-utility rows: `0`; RQ rows: `0`; acceptance pass: `8/13`.
- Scope boundary: no reranking, no full cohort, no LLMSRec, no remote APIs, no training, no download.

## phase_f6l_lastfm_corrective_output_contract_v4_full_ranking (2026-08-05T16:11:15Z)

- Decision token: `f6l_lastfm_v4_full_ranking_partial_resumable_infrastructure_interruption`
- Run ID: `20260804_131959`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_full_ranking_20260804_131959`
- Planned/completed/strict-valid/failed/residual: `4000`/`3991`/`3942`/`49`/`9`.
- Metrics rows: `0`; beyond-utility rows: `0`; RQ rows: `0`; acceptance pass: `8/13`.
- Scope boundary: no reranking, no full cohort, no LLMSRec, no remote APIs, no training, no download.

## phase_f6l_lastfm_corrective_output_contract_v4_full_ranking (2026-08-05T18:52:45Z)

- Decision token: `f6l_lastfm_v4_full_ranking_complete_with_observed_duplicate_invalidity`
- Run ID: `20260804_131959`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_full_ranking_20260804_131959`
- Planned/completed/strict-valid/failed/residual: `4000`/`4000`/`3950`/`50`/`0`.
- Metrics rows: `28`; beyond-utility rows: `28`; RQ rows: `3`; acceptance pass: `12/13`.
- Scope boundary: no reranking, no full cohort, no LLMSRec, no remote APIs, no training, no download.

## phase_f6l_lastfm_corrective_output_contract_v4_full_ranking (2026-08-05T19:11:05Z)

- Decision token: `f6l_lastfm_v4_full_ranking_complete_with_observed_duplicate_invalidity`
- Run ID: `20260804_131959`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_full_ranking_20260804_131959`
- Planned/completed/strict-valid/failed/residual: `4000`/`4000`/`3950`/`50`/`0`.
- Metrics rows: `28`; beyond-utility rows: `28`; RQ rows: `3`; acceptance pass: `13/13`.
- Scope boundary: no reranking, no full cohort, no LLMSRec, no remote APIs, no training, no download.

<!-- phase_f6l_lastfm_v4_ranking_metrics_freeze:start -->
## Ranking Metrics Freeze Before Reranking

- Freeze state: `ranking_metrics_frozen_before_reranking`.
- Run ID: `20260804_131959`.
- Decision token: `f6l_lastfm_v4_ranking_metrics_consolidated_and_frozen_for_reranking`.
- Freeze SHA-256: `D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC`.
- Canonical metrics: `results/tables/phase_f6l_lastfm_v4_ranking_freeze_metrics.csv`.
- Canonical comparisons: `results/tables/phase_f6l_lastfm_v4_ranking_freeze_comparisons.csv`.
- Summary JSON: `results/manifests/phase_f6l_lastfm_v4_ranking_metrics_freeze_summary.json`.
- Commands run: `.venv_audit_recovery\Scripts\python.exe -B src\analysis\phase_f6l_lastfm_v4_ranking_metrics_freeze.py --config configs\phase_f6l_paper_faithful_completion_and_rq_revalidation.yaml`.
- Tests passed: `24 passed` for `tests/test_phase_f6l_lastfm_v4_ranking_metrics_freeze.py`, `tests/test_phase_f6l_lastfm_corrective_output_contract_v4_pilot.py`, and `tests/test_phase_f6l_lastfm_parser_forensics.py` with `PYTHONNOUSERSITE=1`.
- Known unrelated test gap: `tests/test_phase_f6l_lastfm_execution_contracts.py` still reports `f6l_lastfm_execution_contracts_blocked_by_missing_local_evidence`.
- Scope boundary: no reranking, no new inference, no raw repair, no deduplication, no API call.
<!-- phase_f6l_lastfm_v4_ranking_metrics_freeze:end -->

## phase_f6l_lastfm_v4_reranking_post_breaker_handoff

- Current reranking post-breaker decision token: `f6l_lastfm_v4_reranking_post_breaker_blocked_pending_evidence`.
- Audited run ID: `20260806_034027`.
- Do not launch continuation inference from this state without a new explicit human authorization.
- The operational breaker stop remains valid: window 12, final response ordinals `2201-2400`, duplicate-token invalids `11/200 = 0.055`, threshold `>0.05`.
- The scientific interpretation is narrower than the operational token name: evidence supports local/cell-specific Mistral duplicate-token spike, not global systemic regression.
- Baseline ranking freeze remains authoritative and immutable: decision token `f6l_lastfm_v4_ranking_metrics_consolidated_and_frozen_for_reranking`, freeze hash `D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC`.
- Raw outputs under `results/llm/phase_f6l_lastfm_v4_reranking_full_run_20260806_034027` must remain byte-preserved.
- Exact final-response residual count is `1600`, but the residual manifest ordinal set is non-contiguous: `48` non-final timeout attempts occurred before ordinal `2449`, and never-attempted ordinals are `2449-4000`.
- Required blocker for any V4.1 continuation: human policy must decide whether the 48 timeout attempts are included in the continuation manifest, excluded as transport evidence, or handled under a separate recovery contract.
- Post-breaker evidence paths:
  - `report/phase_f6l_lastfm_v4_reranking_post_breaker_forensic_consolidation.md`
  - `results/manifests/phase_f6l_lastfm_v4_reranking_post_breaker_summary.json`
  - `results/manifests/phase_f6l_lastfm_v4_reranking_post_breaker_hashes.json`
  - `results/tables/phase_f6l_lastfm_v4_reranking_post_breaker_decision_matrix.csv`
  - `results/tables/phase_f6l_lastfm_v4_reranking_post_breaker_window_analysis.csv`
  - `results/tables/phase_f6l_lastfm_v4_reranking_post_breaker_cell_coverage.csv`
- Focused tests: `tests/test_phase_f6l_lastfm_v4_reranking_post_breaker.py` returned `8 passed`.

## phase_f6l_lastfm_v4_reranking_continuation_v4_1_handoff

- Current continuation freeze decision token: `f6l_lastfm_v4_reranking_post_breaker_versioned_continuation_ready`.
- Continuation ID: `phase_f6l_lastfm_v4_reranking_continuation_v4_1_20260807`.
- This is a freeze/proposal only: no continuation inference has been launched.
- Policy resolution supersedes the prior blocked condition `pending_manifest_interval == exactly 2401-4000`.
- New target rule: all and only canonical requests without a final response.
- Continuation manifest size: `1600` rows.
- Continuation classes:
  - `retry_after_non_final_transport_timeout`: `48`
  - `first_attempt_never_attempted`: `1552`
- Every continuation row has `prior_final_response_exists=false`.
- Historical timeout raw evidence is preserved by path and SHA-256; raw artifact manifest hash remains `AB6449BD7BE8B8288A6FB0DB5FB3F3439F616662AE93FF572065120D2B6BE030`.
- V4.1 duplicate-token breaker: duplicate_token remains structural-invalid fail-strict; stop only if two consecutive closed 200-final-response windows in the same model/strategy cell exceed `0.05`, or if the same-cell cumulative duplicate rate exceeds `0.05` after at least `400` final responses.
- Immediate hard stops remain malformed JSON, unknown/out-of-domain token, wrong length, mapping failure, parser drift, schema drift, and manifest/hash drift.
- Do not execute V4.1 without a new explicit human authorization.
- Freeze artifacts:
  - `configs/phase_f6l_lastfm_v4_reranking_continuation_v4_1.yaml`
  - `results/tables/phase_f6l_lastfm_v4_reranking_continuation_v4_1_manifest.csv`
  - `results/tables/phase_f6l_lastfm_v4_reranking_continuation_v4_1_policy_audit.csv`
  - `results/manifests/phase_f6l_lastfm_v4_reranking_continuation_v4_1_summary.json`
  - `results/manifests/phase_f6l_lastfm_v4_reranking_continuation_v4_1_hashes.json`
  - `report/phase_f6l_lastfm_v4_reranking_continuation_v4_1_freeze.md`
- Focused tests: `tests/test_phase_f6l_lastfm_v4_reranking_continuation_v4_1.py` returned `8 passed`.

## phase_f6m_reproduction_closure_handoff

- Decision token: `f6m_reproduction_closure_complete_with_documented_limitations`.
- Status: `REPRODUCIBILITY CORE - CLOSED`; `FUTURE EXTENSION - NOT STARTED` as a historical F6M-only statement, superseded for repository-wide status by later G1-G4 extension freezes.
- Phase F6M is offline-only: no LLM/API calls, no recommender training, no downloads, no raw-output repair, no deduplication, and no full LastFM extension.
- F6M builder command: `.venv_audit_recovery\Scripts\python.exe -B src\experiments\build_phase_f6m_reproduction_closure.py --config configs\phase_f6m_reproduction_closure.yaml`.
- Manifest: `results/manifests/phase_f6m_reproduction_closure_manifest.json`.
- Closure report: `report/phase_f6m_reproduction_closure.md`.
- LastFM ranking remains frozen with hash `D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC`.
- LastFM V4.1 reranking is complete with `4000/4000` aggregate final responses and decision token `f6l_lastfm_v4_reranking_continuation_v4_1_complete`.
- Reranking positive-present coverage is `314/1000 = 0.314` and is diagnostic only; it must not be applied to ranking.
- Primary F6M tables:
  - `results/tables/phase_f6m_master_results_long.csv`
  - `results/tables/phase_f6m_master_results_wide.csv`
  - `results/tables/phase_f6m_master_reproducibility_matrix.csv`
  - `results/tables/phase_f6m_research_question_matrix.csv`
  - `results/tables/phase_f6m_paper_claim_evidence_matrix.csv`
  - `results/tables/phase_f6m_limitations_register.csv`
  - `results/tables/phase_f6m_lastfm_final_status.csv`
  - `results/tables/phase_f6m_reproduction_extension_boundary.csv`
- Historical future operation at F6M time: design the original extension starting from the frozen reproduction commit. This was superseded by the later G1-G4 extension freezes; do not reinterpret F6M as containing extension results.
