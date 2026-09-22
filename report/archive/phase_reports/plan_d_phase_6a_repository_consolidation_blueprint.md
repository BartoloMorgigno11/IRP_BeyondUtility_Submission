# Phase 6A Repository Consolidation Blueprint

## 1. Objective
Freeze current repository evidence, classify every relevant artifact, reconcile Plan E, and define the Phase 6B documentation input contract without changing scientific results.

## 2. Repository State
Branch expectation: `plan-d-extensions`. HEAD expectation: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`. The worktree intentionally remains dirty with prior artifacts.

## 3. Protected-Input Policy
All pre-existing files discovered outside Phase 6A outputs are protected inputs. Phase 6A writes only `plan_d_phase_6a_*` tables, this blueprint, and Phase 6A source/test files.

## 4. Artifact-Discovery Methodology
Discovery uses deterministic recursive traversal, excludes tool caches and Phase 6A self-outputs, normalizes Windows paths, assigns SHA-256 hashes, and applies filename/content heuristics for producer and consumer mapping.

## 5. Artifact Counts By Phase
| producing_phase | count |
| --- | ---: |
| baseline_phase | 48 |
| canonical_pre_5e | 195 |
| dataset_preparation | 53 |
| generated_results | 223 |
| implementation | 83 |
| original_repository | 27 |
| paper | 2 |
| phase_5c | 20 |
| phase_5d | 19 |
| phase_5e | 22 |
| plan_e | 285 |
| reporting | 35 |
| repository_scaffold | 14 |
| test_suite | 56 |

## 6. Artifact Counts By Lifecycle Class
| lifecycle_class | count |
| --- | ---: |
| adapted_component_artifact | 85 |
| blocked_or_incomplete_artifact | 10 |
| canonical_final_evidence | 40 |
| canonical_supporting_artifact | 193 |
| exact_paired_diagnostic | 67 |
| extension_artifact | 108 |
| generated_manifest | 87 |
| implementation_support | 105 |
| legacy_protocol_artifact | 1 |
| raw_immutable_input | 44 |
| reconstructed_component_artifact | 75 |
| repeated_controlled_diagnostic | 9 |
| small_pilot | 71 |
| test_artifact | 101 |
| unrelated_to_final_delivery | 86 |

## 7. Canonical Evidence Map
| topic | primary artifacts | caveat |
| --- | --- | --- |
| Phase 5C canonical local LLM matrix | results/tables/canonical_local_llm_full_matrix_cell_metrics.csv;results/tables/canonical_local_llm_full_matrix_rankings.csv;results/tables/canonical_local_llm_full_matrix_validation.csv | Local Qwen2/Mistral evidence is adapted local-model evidence, not exact paper LLM replication. |
| Phase 5D robustness evidence | results/tables/plan_d_phase_5d_robustness_summary.csv;results/tables/plan_d_phase_5d_model_task_reliability.csv | Robustness diagnostics do not upgrade pilot evidence to full-matrix replication. |
| Phase 5E paper coverage audit | results/tables/plan_d_phase_5e_component_coverage.csv;results/tables/plan_d_phase_5e_coverage_metrics.csv;report/plan_d_phase_5e_final_paper_coverage_audit.md | Maximum defensible coverage is not exact replication. |
| Traditional and metric support | results/tables/traditional_baselines_metrics.csv;results/tables/canonical_metric_results_long.csv;results/tables/canonical_metric_contract.csv | Baseline comparisons require explicit cohort/protocol alignment. |
| Plan E extension evidence | results/tables/plan_d_phase_6a_plan_e_reconciliation.csv | Plan E does not retroactively change Phase 5E coverage conclusions. |

## 8. Phase 5C Canonical Evidence
Phase 5C full-matrix artifacts remain the canonical local LLM matrix evidence. Pilots and legacy full-20 artifacts are preserved but are not promoted.

## 9. Phase 5D Robustness Evidence
Phase 5D robustness consolidation remains the canonical robustness-evidence scope.

## 10. Phase 5E Coverage Evidence
Phase 5E coverage tables remain frozen and authoritative for paper-component coverage.

## 11. Complete Phase 5E Status Accounting
| status_or_metric | count/value | verification |
| --- | ---: | --- |
| reproduced_exact | 9 | pass |
| reconstructed | 16 | pass |
| adapted | 25 | pass |
| blocked | 14 | pass |
| not_executed | 5 | pass |
| unresolved | 4 | pass |
| extension_only | 0 | pass |
| not_applicable | 0 | pass |
| applicable_component_count | 73 | pass |
| strict_exact_reproduction_micro_pct | 12.328767 | pass |
| maximum_defensible_micro_coverage_pct | 68.493151 | pass |
| exact_numeric_result_cell_pct | 0.0 | pass |

## 12. Plan E Artifact Reconciliation
Plan E artifact rows: 285. Plan E is preserved as extension/support evidence and does not change frozen Phase 5E conclusions.

## 13. Legacy And Superseded Artifacts
Legacy, pilot, and diagnostic rows indexed: 148. They remain available for traceability and are not foregrounded as final evidence.

## 14. Extension Artifacts
Extension-class artifacts: 108. These can support extension sections only.

## 15. Blocked And Incomplete Artifacts
Blocked/incomplete artifacts: 10. They must be carried as limitations.

## 16. Orphaned And Unresolved Artifacts
Orphaned/unresolved/path-risk rows: 372. These require review before Phase 6B foregrounds affected artifacts.

## 17. Reproducibility Entry Points
Entry points classified: 96. Safe README candidates are those marked `safe_for_final_documentation=true`.

## 18. Path Portability Risks
Absolute path risks are inventoried only; Phase 6A does not rewrite prior reports or source files.

## 19. Final-Delivery Inclusion Policy
| family | disposition | report role |
| --- | --- | --- |
| Phase 5C full matrix | include_primary | primary result source |
| Phase 5D robustness | include_primary | robustness limitations |
| Phase 5E coverage | include_primary | reproduction assessment |
| Plan E artifacts | preserve_not_foreground | extensions/future-work caveat |
| Pilots and legacy protocols | preserve_legacy | methods history only |
| Blocked and unresolved artifacts | preserve_blocked | blockers/threats |
| Implementation support and tests | exclude_from_final_narrative | not primary evidence |

## 20. Final README Input Map
Phase 6B should use `plan_d_phase_6a_final_documentation_input_map.csv` and `plan_d_phase_6a_reproducibility_entrypoints.csv`; unsafe runners must not be default commands.

## 21. Final Scientific Report Input Map
The final scientific report must read numerical claims from canonical CSVs, especially Phase 5C, Phase 5D, Phase 5E, and metric-contract tables.

## 22. Required Future Figures And Derived Tables
Later phases may derive figures from canonical CSVs. No final figures are created in Phase 6A.

## 23. Prohibited Final-Document Sources
Do not use pilot results as full-matrix evidence, Plan E as a silent Phase 5E update, report-only numeric claims without CSV support, or unsafe runner commands as default reproduction commands.

## 24. Residual Blockers
Exact paper processed artifacts, exact paper LLMs, and exact numeric result-cell replication remain blocked or unsupported per Phase 5E.

## 25. Phase 6B Input Contract
Phase 6B must consume the Phase 6A registry, input map, entrypoint registry, final-delivery policy, and Phase 5E accounting. It must not recompute scientific results implicitly.

## 26. Phase 6A Decision
Validation pass/fail: 32/0. Decision: `final_repository_evidence_freeze_complete`.
