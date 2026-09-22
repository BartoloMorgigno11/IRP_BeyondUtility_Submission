# Phase F6I2A Substantive Repository Consolidation Design

Decision: `f6i2a_consolidation_design_ready_for_approval_review`.

Runtime classification: `defensible_repaired_clone_exact_package_versions`. This is not exact original runtime reproduction.

## Scope

- Selected phase identifier: `phase_f6i2a`.
- F6I1 authoritative inventory rows: `1948`.
- F6I1 dependency edges: `45826`.
- Cleanup executed in F6I2A: `0`.
- Deletions, moves, renames, archives, merges, training, inference, API calls, downloads, package changes, staging, commits: `0`.

## Counts

- `relevant_files`: `1948`
- `python_files_audited`: `408`
- `yaml_files_audited`: `27`
- `tests_audited`: `143`
- `markdown_files_audited`: `134`
- `agents_codex_files_audited`: `70`
- `canonical_runtime_files`: `16`
- `canonical_builders`: `30`
- `future_dependency_files`: `1017`
- `frozen_evidence_files`: `109`
- `provenance_only_files`: `0`
- `low_risk_deletion_candidates`: `8`
- `archive_candidates`: `179`
- `python_merge_groups`: `9`
- `config_merge_groups`: `6`
- `test_merge_groups`: `0`
- `documentation_merge_groups`: `5`
- `manual_review_count`: `309`

## Target Architecture

source code->src/audit/; source code->src/data/; source code->src/baselines/; source code->src/llm/; source code->src/evaluation/;src/metrics/; source code->src/experiments/; configs->configs/base/;configs/datasets/;configs/models/;configs/prompts/;configs/experiments/;configs/archive/phases/; tests->tests/unit/;tests/integration/;tests/regression/;tests/offline_audit/;tests/manual_unsafe/; documentation->docs/current_status.md;docs/architecture.md;docs/reproduction_protocol.md;docs/roadmap.md;docs/archive/; results->results/raw/;results/frozen/;results/current/;results/derived/;results/archive/

## Approval Bundles

- Bundle A low-risk deletion: `8` paths.
- Bundle B archive-only: `179` paths.
- Bundle C code consolidation: `9` groups.
- Bundle D config consolidation: `6` groups.
- Bundle E test consolidation: `0` groups.
- Bundle F documentation consolidation: `5` groups.

Expected file-count reduction after approved F6I2B low-risk deletion bundle: `8` files. Archive and merge bundles require separate approval and compatibility wrappers before any count reduction is defensible.

## Validation

- Blocking validation failures: `0`.
- Protected input hashes unchanged: `True`.
- Determinism semantic two-pass result: `pass`.

## Git State

Initial git status:

```text
## plan-d-extensions
 M README.md
 M report/dataset_audit.md
 M report/notes.md
?? .venv_audit_recovery/
?? .venv_f6g2_py312/
?? .venv_repro/
?? .venv_repro_system_site/
?? configs/finalization_f6d2_lastfm_trainable_models.yaml
?? configs/phase_f6f1_beyond_utility_metrics.yaml
?? configs/phase_f6f2_scientific_readout.yaml
?? configs/phase_f6g1_lastfm_reranking_reconstruction.yaml
?? configs/phase_f6g2_lastfm_reranking_execution.yaml
?? configs/phase_f6g2r_runtime_recovery.yaml
?? configs/phase_f6g3_lastfm_reranking_top10_freeze.yaml
?? configs/phase_f6h1_offline_exactness_elevation.yaml
?? configs/phase_f6h2_direct_verification_and_lineage_recovery.yaml
?? configs/phase_f6i2a_substantive_repository_consolidation_design.yaml
?? configs/plan_e_bprmf_beauty.yaml
?? configs/plan_e_bprmf_ml1m_pilot.yaml
?? configs/plan_e_bprmf_sports.yaml
?? configs/plan_e_gru4rec_beauty.yaml
?? configs/plan_e_gru4rec_ml1m_pilot.yaml
?? configs/plan_e_gru4rec_sports.yaml
?? configs/plan_e_lightgcn_beauty_cpu.yaml
?? configs/plan_e_lightgcn_ml1m_pilot.yaml
?? configs/plan_e_lightgcn_sports_cpu.yaml
?? configs/plan_e_sasrec_beauty.yaml
?? configs/plan_e_sasrec_ml1m_pilot.yaml
?? configs/plan_e_sasrec_sports.yaml
?? data/processed/finalization_f6c_lastfm_catalog.csv
?? data/processed/finalization_f6c_lastfm_interactions.csv
?? data/processed/finalization_f6c_lastfm_ranking_candidates.csv
?? data/processed/finalization_f6c_lastfm_ranking_cohort.csv
?? data/processed/finalization_f6c_lastfm_reranking_cohort.csv
?? data/processed/finalization_f6c_lastfm_test.csv
?? data/processed/finalization_f6c_lastfm_train.csv
?? data/processed/finalization_f6c_lastfm_validation.csv
?? data/processed/finalization_f6d2_lastfm_reranking_candidates_adapted.csv
?? docs/
?? "paper/Beyond Utility Evaluating LLM As Recommander.pdf"
?? paper/traduzione_italiana_beyond_utility_llm_recommender.docx
?? pytest.ini
?? report/canonical_llm_prompt_parsing_and_execution_preflight.md
?? report/canonical_local_llm_final_reranking_protocol_decision.md
?? report/canonical_local_llm_full_evaluation.md
?? report/canonical_local_llm_full_matrix_execution.md
?? report/canonical_local_llm_hybrid_disjoint_holdout_pilot.md
?? report/canonical_local_llm_hybrid_prompt_and_holdout_audit.md
?? report/canonical_local_llm_pilot_invalid_output_and_prompt_corrective_audit.md
?? report/canonical_local_llm_reranking_prompt_v2_corrective_pilot.md
?? report/canonical_local_llm_top10_confirmation_preflight.md
?? report/canonical_local_llm_top10_reranking_confirmation_pilot.md
?? report/canonical_local_ollama_service_recovery.md
?? report/canonical_local_qwen2_mistral_pilot.md
?? report/canonical_local_qwen2_mistral_pilot_resume.md
?? report/canonical_metric_contract_and_preflight.md
?? report/canonical_metric_results_seven_baselines.md
?? report/exactness_frontier_and_evidence_closure.md
?? report/experiment_cohort_policy.md
?? report/final_scientific_report.md
?? report/finalization_f2_runtime_verification.md
?? report/finalization_f3_canonical_scientific_consolidation.md
?? report/finalization_f4_metric_resolution.md
?? report/finalization_f5a_paper_numeric_evidence.md
?? report/finalization_f5af_atomic_paper_cells.md
?? report/finalization_f5ar_structured_numeric_recovery.md
?? report/finalization_f5b_ild_policy.md
?? report/finalization_f5c_popularity_bias.md
?? report/finalization_f5d_serendipity.md
?? report/finalization_f5e_calibration.md
?? report/finalization_f6a_paper_scope_rebaseline.md
?? report/finalization_f6b_atomic_scope_registry.md
?? report/finalization_f6c_lastfm_readiness.md
?? report/finalization_f6d1_lastfm_baselines.md
?? report/finalization_f6d2_lastfm_trainable_models.md
?? report/finalization_f6d3_lastfm_llm_preflight.md
?? report/finalization_f6d4a_closure.md
?? report/finalization_f6d4a_lastfm_llm_ranking.md
?? report/future_operations_context_handoff.md
?? report/item_title_map_audit.md
?? report/neural_baseline_feasibility.md
?? report/phase_f6e1_prompt_strategy_recovery.md
?? 
```

Final git status:

```text
## plan-d-extensions
 M README.md
 M report/dataset_audit.md
 M report/notes.md
?? .venv_audit_recovery/
?? .venv_f6g2_py312/
?? .venv_repro/
?? .venv_repro_system_site/
?? configs/finalization_f6d2_lastfm_trainable_models.yaml
?? configs/phase_f6f1_beyond_utility_metrics.yaml
?? configs/phase_f6f2_scientific_readout.yaml
?? configs/phase_f6g1_lastfm_reranking_reconstruction.yaml
?? configs/phase_f6g2_lastfm_reranking_execution.yaml
?? configs/phase_f6g2r_runtime_recovery.yaml
?? configs/phase_f6g3_lastfm_reranking_top10_freeze.yaml
?? configs/phase_f6h1_offline_exactness_elevation.yaml
?? configs/phase_f6h2_direct_verification_and_lineage_recovery.yaml
?? configs/phase_f6i2a_substantive_repository_consolidation_design.yaml
?? configs/plan_e_bprmf_beauty.yaml
?? configs/plan_e_bprmf_ml1m_pilot.yaml
?? configs/plan_e_bprmf_sports.yaml
?? configs/plan_e_gru4rec_beauty.yaml
?? configs/plan_e_gru4rec_ml1m_pilot.yaml
?? configs/plan_e_gru4rec_sports.yaml
?? configs/plan_e_lightgcn_beauty_cpu.yaml
?? configs/plan_e_lightgcn_ml1m_pilot.yaml
?? configs/plan_e_lightgcn_sports_cpu.yaml
?? configs/plan_e_sasrec_beauty.yaml
?? configs/plan_e_sasrec_ml1m_pilot.yaml
?? configs/plan_e_sasrec_sports.yaml
?? data/processed/finalization_f6c_lastfm_catalog.csv
?? data/processed/finalization_f6c_lastfm_interactions.csv
?? data/processed/finalization_f6c_lastfm_ranking_candidates.csv
?? data/processed/finalization_f6c_lastfm_ranking_cohort.csv
?? data/processed/finalization_f6c_lastfm_reranking_cohort.csv
?? data/processed/finalization_f6c_lastfm_test.csv
?? data/processed/finalization_f6c_lastfm_train.csv
?? data/processed/finalization_f6c_lastfm_validation.csv
?? data/processed/finalization_f6d2_lastfm_reranking_candidates_adapted.csv
?? docs/
?? "paper/Beyond Utility Evaluating LLM As Recommander.pdf"
?? paper/traduzione_italiana_beyond_utility_llm_recommender.docx
?? pytest.ini
?? report/canonical_llm_prompt_parsing_and_execution_preflight.md
?? report/canonical_local_llm_final_reranking_protocol_decision.md
?? report/canonical_local_llm_full_evaluation.md
?? report/canonical_local_llm_full_matrix_execution.md
?? report/canonical_local_llm_hybrid_disjoint_holdout_pilot.md
?? report/canonical_local_llm_hybrid_prompt_and_holdout_audit.md
?? report/canonical_local_llm_pilot_invalid_output_and_prompt_corrective_audit.md
?? report/canonical_local_llm_reranking_prompt_v2_corrective_pilot.md
?? report/canonical_local_llm_top10_confirmation_preflight.md
?? report/canonical_local_llm_top10_reranking_confirmation_pilot.md
?? report/canonical_local_ollama_service_recovery.md
?? report/canonical_local_qwen2_mistral_pilot.md
?? report/canonical_local_qwen2_mistral_pilot_resume.md
?? report/canonical_metric_contract_and_preflight.md
?? report/canonical_metric_results_seven_baselines.md
?? report/exactness_frontier_and_evidence_closure.md
?? report/experiment_cohort_policy.md
?? report/final_scientific_report.md
?? report/finalization_f2_runtime_verification.md
?? report/finalization_f3_canonical_scientific_consolidation.md
?? report/finalization_f4_metric_resolution.md
?? report/finalization_f5a_paper_numeric_evidence.md
?? report/finalization_f5af_atomic_paper_cells.md
?? report/finalization_f5ar_structured_numeric_recovery.md
?? report/finalization_f5b_ild_policy.md
?? report/finalization_f5c_popularity_bias.md
?? report/finalization_f5d_serendipity.md
?? report/finalization_f5e_calibration.md
?? report/finalization_f6a_paper_scope_rebaseline.md
?? report/finalization_f6b_atomic_scope_registry.md
?? report/finalization_f6c_lastfm_readiness.md
?? report/finalization_f6d1_lastfm_baselines.md
?? report/finalization_f6d2_lastfm_trainable_models.md
?? report/finalization_f6d3_lastfm_llm_preflight.md
?? report/finalization_f6d4a_closure.md
?? report/finalization_f6d4a_lastfm_llm_ranking.md
?? report/future_operations_context_handoff.md
?? report/item_title_map_audit.md
?? report/neural_baseline_feasibility.md
?? report/phase_f6e1_prompt_strategy_recovery.md
?? 
```
