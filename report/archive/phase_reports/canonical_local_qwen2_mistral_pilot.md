# Canonical Local Qwen2 and Mistral Pilot

## 1. Scope and provenance

Branch `plan-d-extensions` at `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests`. Initial status:

```text
## plan-d-extensions
 M report/dataset_audit.md
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
?? "paper/Beyond Utility Evaluating LLM As Recommander.pdf"
?? report/canonical_llm_prompt_parsing_and_execution_preflight.md
?? report/canonical_metric_contract_and_preflight.md
?? report/canonical_metric_results_seven_baselines.md
?? report/experiment_cohort_policy.md
?? report/item_title_map_audit.md
?? report/neural_baseline_feasibility.md
?? report/plan_e_bprmf_adapter_preflight.md
?? report/plan_e_bprmf_beauty_sports_and_baseline_comparison.md
?? report/plan_e_bprmf_ml1m_pilot.md
?? report/plan_e_gru4rec_adapter_preflight.md
?? report/plan_e_gru4rec_beauty_sports_and_baseline_comparison.md
?? report/plan_e_gru4rec_ml1m_pilot.md
?? report/plan_e_lightgcn_adapter_preflight.md
?? report/plan_e_lightgcn_beauty_sports_cpu_and_baseline_comparison.md
?? report/plan_e_lightgcn_ml1m_cuda_first_step_diagnostic.md
?? report/plan_e_lightgcn_ml1m_determinism_diagnostic.md
?? report/plan_e_lightgcn_ml1m_pilot.md
?? report/plan_e_sasrec_adapter_preflight.md
?? report/plan_e_sasrec_beauty_sports_and_baseline_comparison.md
?? report/plan_e_sasrec_ml1m_pilot.md
?? report/reproducibility_audit_current_state.md
?? report/traditional_baseline_alignment.md
?? results/models/
?? results/tables/candidate_item_title_map.csv
?? results/tables/canonical_llm_blocked_or_conditional.csv
?? results/tables/canonical_llm_cohort_preflight.csv
?? results/tables/canonical_llm_execution_matrix.csv
?? results/tables/canonical_llm_execution_parameter_contract.csv
?? results/tables/canonical_llm_existing_parser_audit.csv
?? results/tables/canonical_llm_model_inventory.csv
?? results/tables/canonical_llm_nondeterminism_policy.csv
?? results/tables/canonical_llm_parser_contract.csv
?? results/tables/canonical_llm_policy_options.csv
?? results/tables/canonical_llm_prompt_contract.csv
?? results/tables/canonical_llm_prompt_rendering_audit.csv
?? results/tables/canonical_metric_applicability_matrix.csv
?? results/tables/canonical_metric_blocked_or_ambiguous.csv
?? results/tables/canonical_metric_blocked_results.csv
?? results/tables/canonical_metric_computation_manifest.csv
?? results/tables/canonical_metric_contract.csv
?? results/tables/canonical_metric_coverage_components.csv
?? results/tables/canonical_metric_implementation_inventory.csv
?? results/tables/canonical_metric_item_universe_audit.csv
?? results/tables/canonical_metric_long_tail_audit.csv
?? results/tables/canonical_metric_policy_options.csv
?? results/tables/canonical_metric_popularity_input_audit.csv
?? results/tables/canonical_metric_ranking_rows.csv
?? results/tables/canonical_metric_results_long.csv
?? results/tables/canonical_metric_results_validation.csv
?? results/tables/canonical_metric_results_wide.csv
?? results/tables/canonical_metric_row_contributions.csv
?? results/tables/canonical_metric_same_cohort_preflight.csv
?? results/tables/canonical_metric_similarity_feasibility.csv
?? results/tables/canonical_metric_top5_exposures.csv
?? results/tables/canonical_metric_train_item_statistics.csv
?? results/tables/dataset_statistics.csv
?? results/tables/experiment_cohort_membership.csv
?? results/tables/experiment_cohort_summary.csv
?? results/tables/item_title_coverage_summary.csv
?? results/tables/item_title_map.csv
?? results/tables/item_title_mapping_issues.csv
?? results/tables/llm_title_exclusions.csv
?? results/tables/neural_baseline_execution_roadmap.csv
?? results/tables/neural_baseline_feasibility_matrix.csv
?? results/tables/neural_baseline_gaps.csv
?? results/tables/neural_environment_inventory.csv
?? results/tables/neural_hardware_inventory.csv
?? results/tables/plan_e_bprmf_all_datasets_metrics.csv
?? results/tables/plan_e_bprmf_beauty_sports_cohort.csv
?? results/tables/plan_e_bprmf_beauty_sports_determinism.csv
?? results/tables/plan_e_bprmf_beauty_sports_metrics.csv
?? results/tables/plan_e_bprmf_beauty_sports_model_id_map.csv
?? results/tables/plan_e_bprmf_beauty_sports_rankings.csv
?? results/tables/plan_e_bprmf_beauty_sports_run_manifest.csv
?? results/tables/plan_e_bprmf_beauty_sports_training_history.csv
?? results/tables/plan_e_bprmf_beauty_sports_validation.csv
?? results/tables/plan_e_bprmf_cold_entity_impact.csv
?? results/tables/plan_e_bprmf_mapping_audit.csv
?? results/tables/plan_e_bprmf_ml1m_model_id_map.csv
?? results/tables/plan_e_bprmf_ml1m_pilot_cohort.csv
?? results/tables/plan_e_bprmf_ml1m_pilot_determinism.csv
?? results/tables/plan_e_bprmf_ml1m_pilot_metrics.csv
?? results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv
?? results/tables/plan_e_bprmf_ml1m_pilot_run_manifest.csv
?? results/tables/plan_e_bprmf_ml1m_pilot_training_history.csv
?? results/tables/plan_e_bprmf_ml1m_pilot_validation.csv
?? results/tables/plan_e_bprmf_non_neural_comparison.csv
?? results/tables/plan_e_bprmf_policy_options.csv
?? results/tables/plan_e_bprmf_preflight_manifest.csv
?? results/tables/plan_e_bprmf_row_compatibility.csv
?? results/tables/plan_e_gru4rec_all_datasets_metrics.csv
?? results/tables/plan_e_gru4rec_baseline_comparison.csv
?? results/tables/plan_e_gru4rec_beauty_sports_cohort.csv
?? results/tables/plan_e_gru4rec_beauty_sports_determinism.csv
?? results/tables/plan_e_gru4rec_beauty_sports_item_token_map.csv
?? results/tables/plan_e_gru4rec_beauty_sports_metrics.csv
?? results/tables/plan_e_gru4rec_beauty_sports_rankings.csv
?? results/tables/plan_e_gru4rec_beauty_sports_run_manifest.csv
?? results/tables/plan_e_gru4rec_beauty_sports_sequence_summary.csv
?? results/tables/plan_e_gru4rec_beauty_sports_training_history.csv
?? results/tables/plan_e_gru4rec_beauty_sports_validation.csv
?? results/tables/plan_e_gru4rec_history_impact.csv
?? results/tables/plan_e_gru4rec_index_audit.csv
?? results/tables/plan_e_gru4rec_ml1m_item_token_map.csv
?? results/tables/plan_e_gru4rec_ml1m_pilot_cohort.csv
?? results/tables/plan_e_gru4rec_ml1m_pilot_determinism.csv
?? results/tables/plan_e_gru4rec_ml1m_pilot_metrics.csv
?? results/tables/plan_e_gru4rec_ml1m_pilot_rankings.csv
?? results/tables/plan_e_gru4rec_ml1m_pilot_run_manifest.csv
?? results/tables/plan_e_gru4rec_ml1m_pilot_training_history.csv
?? results/tables/plan_e_gru4rec_ml1m_pilot_validation.csv
?? results/tables/plan_e_gru4rec_ml1m_sequence_summary.csv
?? results/tables/plan_e_gru4rec_policy_options.csv
?? results/tables/plan_e_gru4rec_preflight_manifest.csv
?? results/tables/plan_e_gru4rec_row_compatibility.csv
?? results/tables/plan_e_gru4rec_same_cohort_baselines_metrics.csv
?? results/tables/plan_e_gru4rec_sequence_audit.csv
?? results/tables/plan_e_lightgcn_all_datasets_device_provenance.csv
?? results/tables/plan_e_lightgcn_all_datasets_metrics.csv
?? results/tables/plan_e_lightgcn_baseline_comparison.csv
?? results/tables/plan_e_lightgcn_beauty_sports_cpu_cohort.csv
?? results/tables/plan_e_lightgcn_beauty_sports_cpu_determinism.csv
?? results/tables/plan_e_lightgcn_beauty_sports_cpu_graph_summary.csv
?? results/tables/plan_e_lightgcn_beauty_sports_cpu_metrics.csv
?? results/tables/plan_e_lightgcn_beauty_sports_cpu_model_node_map.csv
?? results/tables/plan_e_lightgcn_beauty_sports_cpu_rankings.csv
?? results/tables/plan_e_lightgcn_beauty_sports_cpu_run_manifest.csv
?? results/tables/plan_e_lightgcn_beauty_sports_cpu_training_history.csv
?? results/tables/plan_e_lightgcn_beauty_sports_cpu_validation.csv
?? results/tables/plan_e_lightgcn_connectivity_impact.csv
?? results/tables/plan_e_lightgcn_graph_audit.csv
?? results/tables/plan_e_lightgcn_index_audit.csv
?? results/tables/plan_e_lightgcn_ml1m_cpu_cuda_comparison.csv
?? results/tables/plan_e_lightgcn_ml1m_cpu_diagnostic_run_manifest.csv
?? results/tables/plan_e_lightgcn_ml1m_cpu_diagnostic_training_history.csv
?? results/tables/plan_e_lightgcn_ml1m_cuda_first_step_decision.csv
?? results/tables/plan_e_lightgcn_ml1m_cuda_first_step_manifest.csv
?? results/tables/plan_e_lightgcn_ml1m_cuda_first_step_stage_comparison.csv
?? results/tables/plan_e_lightgcn_ml1m_cuda_first_step_variant_summary.csv
?? results/tables/plan_e_lightgcn_ml1m_cuda_optimizer_state_comparison.csv
?? results/tables/plan_e_lightgcn_ml1m_cuda_parameter_gradient_comparison.csv
?? results/tables/plan_e_lightgcn_ml1m_cuda_step_diagnostic.csv
?? results/tables/plan_e_lightgcn_ml1m_determinism_diagnostic.csv
?? results/tables/plan_e_lightgcn_ml1m_diagnostic_decision.csv
?? results/tables/plan_e_lightgcn_ml1m_graph_summary.csv
?? results/tables/plan_e_lightgcn_ml1m_loss_reconciliation.csv
?? results/tables/plan_e_lightgcn_ml1m_model_node_map.csv
?? results/tables/plan_e_lightgcn_ml1m_pilot_cohort.csv
?? results/tables/plan_e_lightgcn_ml1m_pilot_determinism.csv
?? results/tables/plan_e_lightgcn_ml1m_pilot_metrics.csv
?? results/tables/plan_e_lightgcn_ml1m_pilot_rankings.csv
?? results/tables/plan_e_lightgcn_ml1m_pilot_run_manifest.csv
?? results/tables/plan_e_lightgcn_ml1m_pilot_training_history.csv
?? results/tables/plan_e_lightgcn_ml1m_pilot_validation.csv
?? results/tables/plan_e_lightgcn_ml1m_state_hash_audit.csv
?? results/tables/plan_e_lightgcn_normalization_audit.csv
?? results/tables/plan_e_lightgcn_policy_options.csv
?? results/tables/plan_e_lightgcn_preflight_manifest.csv
?? results/tables/plan_e_lightgcn_row_compatibility.csv
?? results/tables/plan_e_lightgcn_same_cohort_baselines_metrics.csv
?? results/tables/plan_e_lightgcn_sparse_execution_audit.csv
?? results/tables/plan_e_same_cohort_traditional_baselines_metrics.csv
?? results/tables/plan_e_sasrec_all_datasets_metrics.csv
?? results/tables/plan_e_sasrec_baseline_comparison.csv
?? results/tables/plan_e_sasrec_beauty_sports_cohort.csv
?? results/tables/plan_e_sasrec_beauty_sports_determinism.csv
?? results/tables/plan_e_sasrec_beauty_sports_item_token_map.csv
?? results/tables/plan_e_sasrec_beauty_sports_metrics.csv
?? results/tables/plan_e_sasrec_beauty_sports_rankings.csv
?? results/tables/plan_e_sasrec_beauty_sports_run_manifest.csv
?? results/tables/plan_e_sasrec_beauty_sports_sequence_objective_summary.csv
?? results/tables/plan_e_sasrec_beauty_sports_training_history.csv
?? results/tables/plan_e_sasrec_beauty_sports_validation.csv
?? results/tables/plan_e_sasrec_history_attention_impact.csv
?? results/tables/plan_e_sasrec_index_audit.csv
?? results/tables/plan_e_sasrec_mask_position_audit.csv
?? results/tables/plan_e_sasrec_ml1m_item_token_map.csv
?? results/tables/plan_e_sasrec_ml1m_pilot_cohort.csv
?? results/tables/plan_e_sasrec_ml1m_pilot_determinism.csv
?? results/tables/plan_e_sasrec_ml1m_pilot_metrics.csv
?? results/tables/plan_e_sasrec_ml1m_pilot_rankings.csv
?? results/tables/plan_e_sasrec_ml1m_pilot_run_manifest.csv
?? results/tables/plan_e_sasrec_ml1m_pilot_training_history.csv
?? results/tables/plan_e_sasrec_ml1m_pilot_validation.csv
?? results/tables/plan_e_sasrec_ml1m_sequence_objective_summary.csv
?? results/tables/plan_e_sasrec_policy_options.csv
?? results/tables/plan_e_sasrec_preflight_manifest.csv
?? results/tables/plan_e_sasrec_row_compatibility.csv
?? results/tables/plan_e_sasrec_same_cohort_baselines_metrics.csv
?? results/tables/plan_e_sasrec_sequence_audit.csv
?? results/tables/plan_e_sasrec_warning_inventory.csv
?? results/tables/positive_positions.csv
?? results/tables/ranking_candidate_sets.csv
?? results/tables/reproducibility_component_matrix.csv
?? results/tables/traditional_baseline_rankings.csv
?? results/tables/traditional_baseline_run_manifest.csv
?? results/tables/traditional_baseline_validation.csv
?? results/tables/traditional_baselines_blocked.csv
?? results/tables/traditional_baselines_metrics.csv
?? src/analysis/canonical_llm_execution_preflight.py
?? src/analysis/canonical_metric_preflight.py
?? src/analysis/canonical_metric_results_validation.py
?? src/analysis/dataset_candidate_set_validation.py
?? src/analysis/experiment_cohort_policy.py
?? src/analysis/item_title_map_reconciliation.py
?? src/analysis/plan_e_bprmf_baseline_consolidation.py
?? src/analysis/plan_e_gru4rec_baseline_consolidation.py
?? src/analysis/plan_e_lightgcn_cpu_baseline_consolidation.py
?? src/analysis/plan_e_lightgcn_cuda_first_step_diagnostic.py
?? src/analysis/plan_e_lightgcn_determinism_diagnostic.py
?? src/analysis/plan_e_sasrec_baseline_consolidation.py
?? src/baselines/plan_e_bprmf_adapter.py
?? src/baselines/plan_e_bprmf_training.py
?? src/baselines/plan_e_gru4rec_adapter.py
?? src/baselines/plan_e_gru4rec_training.py
?? src/baselines/plan_e_lightgcn_adapter.py
?? src/baselines/plan_e_lightgcn_training.py
?? src/baselines/plan_e_non_neural_baselines.py
?? src/baselines/plan_e_sasrec_adapter.py
?? src/baselines/plan_e_sasrec_training.py
?? src/experiments/build_canonical_llm_execution_preflight.py
?? src/experiments/build_canonical_metric_preflight.py
?? src/experiments/build_canonical_metric_results.py
?? src/experiments/build_dataset_candidate_set_audit.py
?? src/experiments/build_experiment_cohorts.py
?? src/experiments/build_item_title_map_audit.py
?? src/experiments/build_plan_e_bprmf_baseline_consolidation.py
?? src/experiments/build_plan_e_bprmf_preflight.py
?? src/experiments/build_plan_e_gru4rec_baseline_consolidation.py
?? src/experiments/build_plan_e_gru4rec_preflight.py
?? src/experiments/build_plan_e_lightgcn_cpu_baseline_consolidation.py
?? src/experiments/build_plan_e_lightgcn_preflight.py
?? src/experiments/build_plan_e_non_neural_baselines.py
?? src/experiments/build_plan_e_sasrec_baseline_consolidation.py
?? src/experiments/build_plan_e_sasrec_preflight.py
?? src/experiments/run_canonical_local_llm_pilot.py
?? src/experiments/run_plan_e_bprmf_beauty_sports.py
?? src/experiments/run_plan_e_bprmf_ml1m_pilot.py
?? src/experiments/run_plan_e_gru4rec_beauty_sports.py
?? src/experiments/run_plan_e_gru4rec_ml1m_pilot.py
?? src/experiments/run_plan_e_lightgcn_beauty_sports_cpu.py
?? src/experiments/run_plan_e_lightgcn_ml1m_cuda_first_step_diagnostic.py
?? src/experiments/run_plan_e_lightgcn_ml1m_determinism_diagnostic.py
?? src/experiments/run_plan_e_lightgcn_ml1m_pilot.py
?? src/experiments/run_plan_e_sasrec_beauty_sports.py
?? src/experiments/run_plan_e_sasrec_ml1m_pilot.py
?? src/llm/canonical_llm_contract.py
?? src/llm/canonical_local_llm_runner.py
?? src/metrics/
?? tests/test_canonical_llm_contract.py
?? tests/test_canonical_local_llm_runner.py
?? tests/test_canonical_metric_computation.py
?? tests/test_canonical_metric_contract.py
?? tests/test_dataset_candidate_set_validation.py
?? tests/test_experiment_cohort_policy.py
?? tests/test_item_title_map_reconciliation.py
?? tests/test_plan_e_bprmf_adapter.py
?? tests/test_plan_e_bprmf_baseline_consolidation.py
?? tests/test_plan_e_bprmf_training.py
?? tests/test_plan_e_gru4rec_adapter.py
?? tests/test_plan_e_gru4rec_baseline_consolidation.py
?? tests/test_plan_e_gru4rec_training.py
?? tests/test_plan_e_lightgcn_adapter.py
?? tests/test_plan_e_lightgcn_cpu_baseline_consolidation.py
?? tests/test_plan_e_lightgcn_cuda_first_step_diagnostic.py
?? tests/test_plan_e_lightgcn_determinism_diagnostic.py
?? tests/test_plan_e_lightgcn_training.py
?? tests/test_plan_e_non_neural_baselines.py
?? tests/test_plan_e_sasrec_adapter.py
?? tests/test_plan_e_sasrec_baseline_consolidation.py
?? tests/test_plan_e_sasrec_training.py
```

Environment: `{'executable': 'C:\\Users\\Bartolo\\Desktop\\PROGETTI\\IRP_BeyondUtility_Project\\.venv\\Scripts\\python.exe', 'python_version': '3.12.10', 'pandas_version': '3.0.3', 'numpy_version': '2.5.0', 'os': 'Windows-11-10.0.26200-SP0', 'cpu': 'Intel64 Family 6 Model 183 Stepping 1, GenuineIntel', 'ram_bytes': '34053414912', 'gpu': 'Intel(R) UHD Graphics;NVIDIA GeForce RTX 4060 Laptop GPU', 'cwd': 'C:\\Users\\Bartolo\\Desktop\\PROGETTI\\IRP_BeyondUtility_Project'}`.

## 2. Executive summary

Phase 5B-P1 built the deterministic sample and 120-row planned manifest. Pilot execution status: `backend_blocked`. Blocker: backend_blocked: Ollama loopback service unavailable: <urlopen error [WinError 10061] Impossibile stabilire la connessione. Rifiuto persistente del computer di destinazione>

## 3. Safety boundaries

No external API, public web, model pull, package install, recommender training, full matrix execution, or prior artifact modification is performed by this runner.

## 4. Phase 5A contract

Contracts are loaded from `results/tables/canonical_llm_*` and hash-validated by `src/llm/canonical_local_llm_runner.py`.

## 5. Prompt-compatible primary cohort

The Phase 5A prompt-compatible primary count is verified as 5,764 before any request path.

## 6. Candidate-title exclusions

Phase 5A candidate-title exclusions remain upstream evidence; this task does not alter title maps or exclusions.

## 7. History-title exclusions

Phase 5A rendered-history title exclusions remain unchanged; this task does not repair or invent titles.

## 8. Deterministic pilot sampling

Sample uses result-independent SHA256 ordering and selects 10 rows per dataset/task. Observed composition: `{('Amazon Beauty', 'ranking'): 10, ('Amazon Beauty', 'reranking'): 10, ('Amazon Sports', 'ranking'): 10, ('Amazon Sports', 'reranking'): 10, ('ML-1M', 'ranking'): 10, ('ML-1M', 'reranking'): 10}`.

## 9. Pilot sample composition

`results/tables/canonical_local_llm_pilot_sample.csv` contains 60 rows.

## 10. Local Ollama environment

executable=C:\Users\Bartolo\AppData\Local\Programs\Ollama\ollama.EXE; version=Warning: could not connect to a running Ollama instance Warning: client version is 0.31.1; service_available=False; service_version=<urlopen error [WinError 10061] Impossibile stabilire la connessione. Rifiuto persistente del computer di destinazione>

## 11. Qwen2 local model evidence

- Qwen2 7B Instruct: required tag `qwen2:7b-instruct`; current installed-model evidence from service: ``.

## 12. Mistral local model evidence

- Mistral 7B Instruct: required tag `mistral:7b-instruct`; current installed-model evidence from service: ``.

## 13. Model-tag and manifest verification

`results/tables/canonical_local_llm_pilot_request_manifest.csv` contains 120 planned requests. Required tags are loaded from Phase 5A inventory.

## 14. Ranking prompt execution

No ranking prompt was sent.

## 15. Reranking prompt execution

No reranking prompt was sent.

## 16. Candidate-order preservation

Candidate-order hashes are present in the sample and request manifest; no candidate order is changed.

## 17. Positive-label concealment

Positive item IDs are retained only in audit tables for later evaluation and are not included in Ollama payload construction.

## 18. Raw-request retention

Raw request JSONL records: 0.

## 19. Raw-response retention

Raw response/error JSONL records: 0.

## 20. Execution parameters

Temperature, output limits, JSON format, streaming=false, concurrency=1, and retry counts are loaded from Phase 5A execution contracts.

## 21. Request ordering

Request IDs are deterministic from `CLLP-REQ-0001` through `CLLP-REQ-0120` in the planned manifest.

## 22. Backend completion findings

Backend counts: `{}`. Decision backend completion rate: `0`.

## 23. Latency findings

No latency values exist because no model requests completed.

## 24. Token-count findings

No generated-token counts exist because no model requests completed.

## 25. Strict parser findings

Strict parse counts: `{}`.

## 26. Diagnostic salvage findings

Diagnostic salvage was not applied to model responses because no responses exist.

## 27. Output-length findings

No output-length findings exist because no responses exist.

## 28. Candidate-validity findings

No candidate-validity findings exist because no strict-valid responses exist.

## 29. Duplicate findings

No duplicate-output findings exist because no responses exist.

## 30. Hallucination findings

No hallucination findings exist because no responses exist.

## 31. Reranking-permutation findings

No reranking permutation could be evaluated because no reranking response exists.

## 32. Qwen2 pilot metrics

- No Qwen2 pilot metrics exist because execution was blocked.

## 33. Mistral pilot metrics

- No Mistral pilot metrics exist because execution was blocked.

## 34. Dataset-specific findings

Sample dataset/task composition: `{('Amazon Beauty', 'ranking'): 10, ('Amazon Beauty', 'reranking'): 10, ('Amazon Sports', 'ranking'): 10, ('Amazon Sports', 'reranking'): 10, ('ML-1M', 'ranking'): 10, ('ML-1M', 'reranking'): 10}`.

## 35. Ranking versus reranking findings

Both tasks are represented in the planned sample and manifest; neither was executed under the current backend blocker.

## 36. Invalid-output taxonomy

Invalid-output taxonomy tables are empty because no model output exists; the blocking issue is infrastructure, not a scientific response.

## 37. Pilot utility findings

Pilot utility is not computed without strict-valid model outputs.

## 38. Why pilot utility is not a final result

The pilot is a readiness gate on 60 prompt rows and is never a full-cohort LLM result.

## 39. Full-execution request estimate

The frozen Phase 5A primary matrix remains 5,764 prompt-compatible rows times 2 local models = 11,528 requests.

## 40. Full-execution readiness decision

`backend_blocked` with threshold `0.80000000000000004`.

## 41. Corrective action, if required

backend_blocked: Ollama loopback service unavailable: <urlopen error [WinError 10061] Impossibile stabilire la connessione. Rifiuto persistente del computer di destinazione>

## 42. Limitations

Current execution evidence is blocked before first request; no local model quality, parser-validity, latency, token, HR@5, or NDCG@5 conclusion can be drawn.

## 43. Protected-input immutability

Protected inputs unchanged after runner work: `True`.

## 44. Commands executed

- `git status --short --branch`
- `git branch --show-current`
- `git log -1 --format="%H %s"`
- `PYTHONNOUSERSITE=1 C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe src/experiments/run_canonical_local_llm_pilot.py`

## 45. Recommended single next task

Phase 5B-P1B — restore the existing local Ollama loopback service without changing model tags or scientific contracts.
