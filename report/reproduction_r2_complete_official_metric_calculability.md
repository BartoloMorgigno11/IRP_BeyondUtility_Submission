# R2 Complete Official Metric Calculability

Label: POST-F6M TOTAL REPRODUCIBILITY CLOSURE EVIDENCE

## Final Decision

`REPRODUCTION_R2_COMPLETE_METRIC_CALCULABILITY_AUDIT_COMPLETE`

## Purpose

R2 pauses extension work and audits whether every paper metric or analysis element can be calculated, reconstructed, preserved only as an official stored value, or blocked with an exact scientific reason from currently available official artifacts.

## R1 frozen baseline

- R1 decision: `REPRODUCTION_R1_OFFICIAL_ARTIFACT_RECOVERY_COMPLETE`.
- R1 decoded official metric values: 1470.
- R1 JSON artifacts: 226.
- R1 history rows: 462.
- R1 generation/profile rows: 546.
- R1 candidate-position evidence rows: 403.

## Repository Precondition

- branch: `master`
- HEAD: `10529058c3579c42686b76d5d3ac8b96ba7d6337`
- Initial manual pre-execution state: no staged/tracked diffs; exactly the 12 protected root artifacts were untracked.
- Final generated git-state tables reflect the post-implementation R2 working tree and are written to `results/tables/reproduction_r2_precondition_git_state.csv` and `results/tables/reproduction_r2_postcondition_git_state.csv`.
- original_repo: `## main...origin/main`
- 12 protected artifacts unchanged: `12/12`

## Complete paper metric universe

- total cells investigated: 1891
- Beauty: 1195
- Sports: 577
- ML-1M: 57
- LastFM: 57

By metric family:
- LLM_specific: 188
- diversity: 323
- fairness: 319
- hallucination: 133
- local_extension: 36
- method: 1
- metric: 1
- novelty: 214
- popularity: 327
- sampling_protocol: 4
- scope: 3
- utility: 338
- utility_diagnostic: 4

## Calculability methodology

R2 first builds metric/input contracts from E1 and official `llm_metrics.py`, then determines input availability, then independently recomputes or reconstructs where the required official inputs are present. Paper comparison is performed only after this status assignment.

## Metric input contracts

Input contracts written to `results/tables/reproduction_r2_metric_input_contracts.csv`. The official data tree under `original_repo/EvaLLMasRecommender/data` is absent, so train-popularity, category metadata, global title mapping, and full-history group metrics are not independently recalculable unless equivalent evidence exists inside official ZIP members.

## Calculability Totals

- DIRECTLY_RECALCULABLE: 297
- RECONSTRUCTIBLE_FROM_OFFICIAL_ARTIFACTS: 6
- PARTIALLY_RECONSTRUCTIBLE: 160
- OFFICIAL_VALUE_AVAILABLE_ONLY: 1173
- NOT_CALCULABLE_MISSING_OUTPUT: 219
- NOT_CALCULABLE_MISSING_DATA: 0
- NOT_CALCULABLE_MISSING_PROTOCOL: 0
- NOT_CALCULABLE_MODEL_UNAVAILABLE: 0
- AMBIGUOUS_RECONSTRUCTION: 0
- NOT_APPLICABLE: 36
- UNKNOWN: 0
- UNINVESTIGATED: 0
- UNCLASSIFIED: 0

## Beauty

Rows: 1195. DIRECTLY_RECALCULABLE=204; RECONSTRUCTIBLE_FROM_OFFICIAL_ARTIFACTS=6; PARTIALLY_RECONSTRUCTIBLE=112; OFFICIAL_VALUE_AVAILABLE_ONLY=790; NOT_CALCULABLE_MISSING_OUTPUT=74; NOT_APPLICABLE=9.

## Sports

Rows: 577. DIRECTLY_RECALCULABLE=93; PARTIALLY_RECONSTRUCTIBLE=48; OFFICIAL_VALUE_AVAILABLE_ONLY=383; NOT_CALCULABLE_MISSING_OUTPUT=44; NOT_APPLICABLE=9.

## ML-1M

Rows: 57. NOT_CALCULABLE_MISSING_OUTPUT=48; NOT_APPLICABLE=9.
R1 classified ML-1M as REFERENCE_ONLY, so official recalculation is blocked by missing official outputs. Local ML-1M artifacts are intentionally not used in R2.

## LastFM

Rows: 57. NOT_CALCULABLE_MISSING_OUTPUT=48; NOT_APPLICABLE=9.
R1 classified LastFM as REFERENCE_ONLY, so official recalculation is blocked by missing official outputs. Local LastFM artifacts are intentionally not used in R2.

## Utility

HR/NDCG are independently recomputed from compatible official JSON title outputs where every record can be parsed. HR recomputed rows: 99. NDCG recomputed rows: 99.

## Novelty

SelfInformation and Serendipity are official stored-only where present because their official definitions require training-frequency information and aligned MostPop hits that are not available in the official data tree.

## Popularity

ACLT/APLT, ARP, and PopREO are official stored-only where present because official train item frequencies, quantiles, and popularity groups are missing from the available official checkout.

## Diversity

ItemCoverage, Gini, and Jaccard are independently recomputed only where official JSON top-K titles map unambiguously to candidate IDs. Recomputed rows: ItemCoverage=0, Gini=0, Jaccard=0. Shannon remains stored-only because category metadata is missing.

## Fairness

Jain is independently recomputed from the reconstructed per-user NDCG vector where possible: 99 rows. DPD is stored-only because cold/hot user groups require full official history lengths.

## Hallucination

Official hallucination values are preserved as stored-only because official title-to-item matching requires `datamaps.json` and `meta.json`; candidate-level validity is not substituted for official global hallucination.

## History sensitivity

Official history lengths found: 0, 1, 2, 3, 4, 5, 10, 15, 20. History reconstruction rows are written to `results/tables/reproduction_r2_history_metric_reconstruction.csv`.

## Generation/profile

Generation/profile reconstruction rows: 1636. Profile modes found: profile_only, profile_plus_history. Written to `results/tables/reproduction_r2_generation_metric_reconstruction.csv`.

## Candidate position/CandDif

exact rows: 6
partial rows: 160
derived rows: 166
output: `results/tables/reproduction_r2_candidate_position_reconstruction.csv`

## Independent recomputation results

- HR: 99
- Jain: 99
- NDCG: 99

## Stored-vs-recomputed agreement

- total compared: 297
- exact: 115
- rounding-equivalent: 1
- different: 181
- not comparable: 0
- discrepancy table: `results/tables/reproduction_r2_official_stored_vs_recomputed.csv`

## Paper comparison

- secure/direct mappings: 0
- likely mappings: 125
- partial mappings: 2
- ambiguous mappings: 0
- unavailable mappings: 170
- exact/rounding/different/not comparable counts: {'not_available': 172, 'different': 121, 'exact': 3, 'rounding_match': 1}

## Official extra conditions

Count: 1317. These official conditions are retained as provenance even without secure paper-cell mapping.

## Not-calculable metrics and exact reasons

Reason rows: 379. Counts by status: {'NOT_CALCULABLE_MISSING_OUTPUT': 219, 'PARTIALLY_RECONSTRUCTIBLE': 160}. Full table: `results/tables/reproduction_r2_not_calculable_reasons.csv`.

## Reconstruction opportunities

Written to `results/tables/reproduction_r2_reconstruction_opportunities.csv`. CandDif is included and labeled as derived from official artifacts when exact paired aggregate values are available.

## What can be completed without new inference

Offline recomputation can continue for any newly recovered official output/data artifacts. No model execution is needed for metrics once outputs, candidate pools, and supporting data are available.

## What cannot be completed and why

Metrics requiring absent official data files, absent ML-1M/LastFM official outputs, unrecovered K-S protocol details, or unavailable closed-model outputs cannot be made paper-exact from current official artifacts.

## What R2 changes about our reproduction understanding

R2 distinguishes independent recomputation from official stored values and preserves exact missing-input reasons for every uncalculated cell.

## What R2 does NOT establish

R2 does not make the reproduction fully paper-exact, does not infer mappings from numerical similarity, and does not use local ML-1M/LastFM/F6 artifacts as official evidence.

## Artifacts Created

- `results/tables/reproduction_r2_official_metric_calculability_master.csv`
- `results/tables/reproduction_r2_official_metric_calculability_summary.csv`
- `results/tables/reproduction_r2_not_calculable_reasons.csv`
- `results/tables/reproduction_r2_reconstruction_opportunities.csv`
- `results/tables/reproduction_r2_official_paper_value_comparison.csv`
- `results/tables/reproduction_r2_history_metric_reconstruction.csv`
- `results/tables/reproduction_r2_generation_metric_reconstruction.csv`
- `results/tables/reproduction_r2_candidate_position_reconstruction.csv`
- `results/tables/reproduction_r2_metric_input_contracts.csv`
- `results/tables/reproduction_r2_official_stored_vs_recomputed.csv`
- `results/tables/reproduction_r2_protected_artifact_hash_check.csv`
- `results/tables/reproduction_r2_precondition_git_state.csv`
- `results/tables/reproduction_r2_postcondition_git_state.csv`
- `results/tables/reproduction_r2_validation.csv`
- `report/reproduction_r2_complete_official_metric_calculability.md`

## Tests

- `.venv_consolidation_candidate\Scripts\python.exe -B -m pytest tests\test_reproduction_r2_official_metric_calculability.py -q -p no:cacheprovider --basetemp results\logs\reproduction_r2_freeze_tmp`: `12 passed in 18.89s`.
- `.venv_consolidation_candidate\Scripts\python.exe -B -m pytest tests\test_reproduction_r1_original_repo_recovery.py tests\test_extension_e1_metric_contract.py -q -p no:cacheprovider --basetemp results\logs\reproduction_r2_freeze_r1_e1_tmp`: `9 passed in 0.64s`.
- Full `pytest tests/` was not run because the requested focused R2 and R1/E1 regression scopes completed successfully and the broader historical suite is large.

## Protection Status

- F6M: not modified by R2 script.
- E1: source contract read only; not modified.
- E2: not modified.
- E3: not resumed or modified.
- R1: source artifacts read only; not modified.
- original_repo: `## main...origin/main`
- 12 protected root artifacts: `12/12 unchanged`

## Repository Postcondition

The R2 builder itself does not stage or commit files. The later R2 freeze commit is handled as a separate repository operation.

## Commit

NONE

## Interpretation

1. Have all metrics/analyses that are currently calculable from official repository evidence now actually been calculated or reconstructed? YES
2. For every metric/analysis that cannot be calculated, do we now know the exact scientific reason? YES
3. Are the calculable/reconstructed official results generally coherent with the published paper where semantic mapping is secure? PARTIALLY
4. Does R2 make the reproduction fully paper-exact? NO
5. What remains before total reproduction closure: perform local-artifact metric reconstruction in R3 using validated local outputs without new model or LLM inference.

## Next Recommended Action

perform R3 complete local-artifact metric calculability and reconstruction audit, calculating every still-calculable paper metric from the validated local reproduction outputs without new model or LLM inference
