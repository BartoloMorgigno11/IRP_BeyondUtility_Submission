# R3 Complete Local Metric And Beyond-Utility Audit

## Final Decision

`REPRODUCTION_R3_COMPLETE_LOCAL_METRIC_AUDIT_COMPLETE`

## Purpose

R3 audits all locally available reproduction artifacts after the frozen R1/R2/F6M baseline. It extracts materialized values, independently recalculates metrics from existing local outputs where possible, and classifies remaining gaps without any new model, LLM, training, candidate-generation, or API execution.

## Frozen R1/R2 baseline

- Expected branch/head: `master` / `5167740b80fd917db99df0768ea9358d00ec3515`.
- R1/R2/F6M artifacts are read-only sources for contracts, provenance, and protection checks.
- 12 protected root artifacts unchanged: `12/12`.
- original_repo status: `## main...origin/main`.

## Complete local artifact inventory

- artifact rows: 19647
- materialized numeric values extracted: 139779
- offline recalculated values: 518

## Complete metric-cell coverage

- total R3 master rows: 17042
- ALREADY_MATERIALIZED: 16500
- RECALCULATED_FROM_LOCAL_OUTPUTS: 488
- RECONSTRUCTED_FROM_LOCAL_ARTIFACTS: 0
- PARTIALLY_RECONSTRUCTIBLE: 0
- NOT_CALCULABLE_MISSING_OUTPUT: 22
- NOT_CALCULABLE_MISSING_INPUT: 32
- NOT_CALCULABLE_MISSING_PROTOCOL: 0
- REQUIRES_NEW_LLM_INFERENCE: 0
- REQUIRES_NEW_NON_LLM_EXECUTION: 0
- NOT_APPLICABLE: 0
- UNKNOWN: 0
- UNINVESTIGATED: 0
- UNCLASSIFIED: 0

## Count semantics

COUNT_SEMANTICS_EXPLAINED: YES

- `reproduction_r3_materialized_values.csv` rows: 139779. This is the raw numeric-observation inventory extracted from scanned aggregate/reference/diagnostic/local artifacts, not the de-duplicated scientific cell table.
- Materialized value source scopes: LOCAL_REPRODUCTION=137186, LOCAL_DIAGNOSTIC=1000, OFFICIAL_REFERENCE=1410, PAPER_CONTRACT=183. R1/R2/paper values are retained only as provenance/reference rows and are not substituted for local result cells.
- `reproduction_r3_offline_recalculated_values.csv` rows: 518. This is the detailed recalculated numerical-quantity inventory from row-level local outputs.
- Master `RECALCULATED_FROM_LOCAL_OUTPUTS` rows: 488. The difference of 30 rows is caused by duplicate scientific keys where fail-strict and conditional-valid recalculations have identical requested counts, valid counts, denominators, values, and source artifacts because all rows in that local cell were strict-valid. The detailed recalculation table preserves both policy observations; the master table de-duplicates them into one scientific cell.
- Master `ALREADY_MATERIALIZED` rows: 16500. These are de-duplicated local scientific cells after excluding official/paper reference substitutions, not the raw materialized-observation count.

## Utility verification

- HR/NDCG rows in master: 3387
- Fail-strict and conditional-valid recalculations are preserved in the detailed recalculation table; master rows keep separate policy cells when requested counts, valid counts, denominators, or values differ.

## Beyond-Utility overview

- beyond-utility rows: 2732
- true remaining gap rows: 54

## Novelty / SelfInformation

- datasets covered: Beauty, LastFM, ML-1M, Sports
- numerical rows: 331
- best sources: results/metrics/plan_d_beyond_utility_metrics.csv, results/tables/canonical_local_llm_baseline_recomputed_metrics.csv, results/tables/canonical_local_llm_beyond_utility_metrics.csv, results/tables/canonical_metric_results_long.csv, results/tables/canonical_metric_results_wide.csv
- primary limitation: notes=baseline recomputed on exact row intersection with this LLM strict-valid subset

## Serendipity

- datasets covered: ALL, Beauty, LastFM, ML-1M, Sports
- numerical rows: 418
- best sources: results/tables/finalization_f5d_serendipity_results.csv, results/tables/phase_f6f1_serendipity_results.csv, results/tables/phase_f6f2_llm_baseline_readout.csv, results/tables/phase_f6f2_serendipity_readout.csv
- primary limitation: notes=MostPop reference is frozen train-only baseline; no paper-exact claim.; status=computed; classification=reconstructed_formula

## Popularity: ACLT/APLT

- datasets covered: ALL, Beauty, LastFM, ML-1M, Sports
- numerical rows: 587
- best sources: results/metrics/plan_d_beyond_utility_metrics.csv, results/tables/canonical_local_llm_baseline_recomputed_metrics.csv, results/tables/canonical_local_llm_beyond_utility_metrics.csv, results/tables/canonical_metric_results_long.csv, results/tables/canonical_metric_results_wide.csv
- primary limitation: notes=per-user top-5 value then macro mean; coverage metrics are separately cohort-level when listed; status=computed; classification=exact_formula_adapted_population

## Popularity: ARP

- datasets covered: ALL, Beauty, LastFM, ML-1M, Sports
- numerical rows: 719
- best sources: results/metrics/plan_d_beyond_utility_metrics.csv, results/tables/canonical_local_llm_baseline_recomputed_metrics.csv, results/tables/canonical_local_llm_beyond_utility_metrics.csv, results/tables/canonical_metric_results_long.csv, results/tables/canonical_metric_results_wide.csv
- primary limitation: notes=per-user top-5 value then macro mean; coverage metrics are separately cohort-level when listed; status=computed; classification=exact_formula_adapted_population

## Popularity: PopREO

- datasets covered: Beauty, LastFM, ML-1M, Sports
- numerical rows: 0
- best sources: none
- primary limitation: Local outputs exist for Beauty, but exact support inputs/protocol for PopREO@5 are unavailable or not aligned.

## Diversity: ItemCoverage

- datasets covered: Beauty, LastFM, ML-1M, Sports
- numerical rows: 485
- best sources: results/metrics/plan_d_beyond_utility_metrics.csv, results/tables/canonical_local_llm_baseline_recomputed_metrics.csv, results/tables/canonical_local_llm_beyond_utility_metrics.csv, results/tables/canonical_metric_coverage_components.csv, results/tables/canonical_metric_results_long.csv
- primary limitation: notes=baseline recomputed on exact row intersection with this LLM strict-valid subset

## Diversity: Shannon

- datasets covered: Beauty, LastFM, ML-1M, Sports
- numerical rows: 0
- best sources: none
- primary limitation: Local outputs exist for Beauty, but exact support inputs/protocol for Shannon category entropy@5 are unavailable or not aligned.

## Diversity: Jaccard

- datasets covered: Beauty, LastFM, ML-1M, Sports
- numerical rows: 0
- best sources: none
- primary limitation: Local outputs exist for Beauty, but exact support inputs/protocol for Jaccard@5 are unavailable or not aligned.

## Fairness: Gini

- datasets covered: Beauty, LastFM, ML-1M, Sports
- numerical rows: 0
- best sources: none
- primary limitation: Local outputs exist for Beauty, but exact support inputs/protocol for Gini@5 are unavailable or not aligned.

## Fairness: DPD

- datasets covered: Beauty, LastFM, ML-1M, Sports
- numerical rows: 0
- best sources: none
- primary limitation: Local outputs exist for Beauty, but exact support inputs/protocol for DPD@5 are unavailable or not aligned.

## Fairness: Jain

- datasets covered: Beauty, LastFM, ML-1M, Sports
- numerical rows: 0
- best sources: none
- primary limitation: Local outputs exist for Beauty, but exact support inputs/protocol for Jain's Index@5 are unavailable or not aligned.

## History Length Sensitivity

- curve/point rows: 120
- sources: results/tables/finalization_f6d4a_history_length_sensitivity.csv, results/tables/phase_f6e4c_request_level_metrics.csv, results/tables/plan_d_llm_ml1m_history_length_condition_summary.csv

## Candidate Position Bias / CandDif

- candidate-position rows: 3421
- paper-compatible CandDif rows: 0
- diagnostic rows: 3421

## Hallucination

- hallucination audit rows: 94
- semantic types: CANDIDATE_VIOLATION, PAPER_STYLE_TEXTUAL_HALLUCINATION, STRUCTURAL_OUTPUT_INVALIDITY

## Generation-Involved Performance

- generation/profile rows: 360
- Performance rows require downstream recommendation outputs; profile-generation-only rows are labeled separately.

## Beauty complete metric coverage

- metrics represented: 59
- metric-family coverage rows: 22
- missing rows after offline recovery: 14

## Sports complete metric coverage

- metrics represented: 59
- metric-family coverage rows: 22
- missing rows after offline recovery: 14

## ML-1M complete metric coverage

- metrics represented: 91
- metric-family coverage rows: 22
- missing rows after offline recovery: 12

## LastFM complete metric coverage

- metrics represented: 99
- metric-family coverage rows: 22
- missing rows after offline recovery: 14

## Already materialized values

Stored in `results/tables/reproduction_r3_materialized_values.csv`.

## Offline recalculated values

Stored in `results/tables/reproduction_r3_offline_recalculated_values.csv`.

## Reconstructed values

R3 derives CandDif only when first/random evidence is explicit; otherwise it preserves diagnostics without paper-compatible labeling.

## Paper-exact results

No local row is upgraded to paper-exact from numerical similarity alone. Rows with exact local support retain their explicit source/protocol labels.

## Adapted results

Most local LLM and beyond-utility rows are adapted because local open models, candidate-index output contracts, operationalized LastFM, and reconstructed support data differ from the paper protocol.

## Diagnostic-only results

Candidate-position diagnostics, structural validity, duplicate rate, and candidate-violation rows are preserved separately from paper reproduction coverage.

## True remaining gaps

Gap rows are written to `results/tables/reproduction_r3_true_remaining_gaps.csv`.
Total true remaining gaps: 54; missing-input=32; missing-output=22.
Gap distribution by metric family: LLM_specific=14, diversity=12, fairness=12, popularity=8, sampling_protocol=4, utility_diagnostic=4.

## Potential controlled recovery work

The smallest next recovery phase should target paper-style hallucination and full-matrix controlled CandDif first, because those remain the highest-priority beyond-utility uncertainties.

## Implications for paper conclusions

R3 supports only protocol-qualified beyond-utility interpretation. Local evidence can compare adapted open-model behavior and deterministic baselines, but paper-level causal claims remain limited where protocols, support data, or exact outputs are missing.

## Implications for total reproducibility closure

Total closure still requires controlled recovery of missing beyond-utility cells or explicit acceptance that those cells remain adapted/diagnostic rather than paper-exact.

## Validation

- validation rows: 19
- validation failures: 0

## Artifacts created

- `results/tables/reproduction_r3_local_artifact_manifest.csv`
- `results/tables/reproduction_r3_all_metric_cells_master.csv`
- `results/tables/reproduction_r3_beyond_utility_master.csv`
- `results/tables/reproduction_r3_paper_metric_dataset_coverage.csv`
- `results/tables/reproduction_r3_true_remaining_gaps.csv`
- `results/tables/reproduction_r3_history_length_sensitivity.csv`
- `results/tables/reproduction_r3_candidate_position_bias.csv`
- `results/tables/reproduction_r3_hallucination_audit.csv`
- `results/tables/reproduction_r3_generation_involved_performance.csv`
- `results/tables/reproduction_r3_materialized_values.csv`
- `results/tables/reproduction_r3_stored_vs_recomputed.csv`
- `results/tables/reproduction_r3_offline_recalculated_values.csv`
- `results/tables/reproduction_r3_validation.csv`
- `results/tables/reproduction_r3_precondition_git_state.csv`
- `results/tables/reproduction_r3_postcondition_git_state.csv`
- `results/tables/reproduction_r3_protected_artifact_hash_check.csv`
