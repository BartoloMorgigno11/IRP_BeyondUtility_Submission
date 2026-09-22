# G3 Candidate-Position Metrics Freeze

- Phase: `g3_candidate_position_metrics_freeze`
- HEAD before generation: `b21c0cff1f3cdb474ef2944fcc1aa5e6fae5325e`
- Inputs: frozen G3 analytic universe and pair table only.
- Inference/API calls: `0`.
- Bootstrap/statistical tests: `0`.

## Metric Completeness

- Utility metric rows: `180` (`HR@5=60`, `NDCG@5=60`, `MRR=60`).
- CandDif rows: `60` (`CandDif_HR@5=30`, `CandDif_NDCG@5=30`).
- Reliability rows: `25`.
- Denominator rows: `116`.

## Denominator Semantics

- Utility: planned/applicable users; invalid terminal responses contribute zero.
- Reliability: planned requests.
- CandDif: paired FIRST/RANDOM cell accuracies over the frozen 6000-pair universe.
- Beyond-utility metrics: not computed in this phase unless separately frozen/applicable to G3.

## Reliability Summary

- Overall requests: `12000`.
- Strict valid: `11834`.
- Invalid: `166`.
- Backend errors: `0`.

| scope_value | planned_requests | strict_valid | invalid | backend_errors |
| ----------- | ---------------- | ------------ | ------- | -------------- |
| FIRST       | 6000             | 5904         | 96      | 0              |
| RANDOM      | 6000             | 5930         | 70      | 0              |

## Validation

- PASS: `26/26`.
- Invalid nonzero utility contribution rows: `0`.
- Strict-valid positive absent from top-10 rows: `3530`.

| check_id        | description                                                                         | observed                                                                                                                    | expected                                                                                                        | status |
| --------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------ |
| STATIC-001      | current HEAD matches G3 analytic-universe freeze precondition                       | b21c0cff1f3cdb474ef2944fcc1aa5e6fae5325e                                                                                    | b21c0cff1f3cdb474ef2944fcc1aa5e6fae5325e                                                                        | PASS   |
| STATIC-002      | git status contains only expected generator/output paths                            | ['report/g3_candidate_position_metrics_validation.md', 'report/notes.md', 'src/extension/g3_candidate_position_metrics.py'] | clean_or_expected_dirty                                                                                         | PASS   |
| STATIC-003      | git diff --check passes before metric freeze                                        | 0                                                                                                                           | 0                                                                                                               | PASS   |
| STATIC-004      | analysis manifest decision token is frozen                                          | G3_ANALYTIC_UNIVERSE_FROZEN                                                                                                 | G3_ANALYTIC_UNIVERSE_FROZEN                                                                                     | PASS   |
| STATIC-005      | analysis manifest validation passed                                                 | True                                                                                                                        | True                                                                                                            | PASS   |
| STATIC-006      | frozen analysis inputs and outputs still match manifest hashes                      | 0                                                                                                                           | 0                                                                                                               | PASS   |
| UNIVERSE-001    | analysis universe row count                                                         | 12000                                                                                                                       | 12000                                                                                                           | PASS   |
| UNIVERSE-002    | analysis universe canonical IDs are unique                                          | 12000                                                                                                                       | 12000                                                                                                           | PASS   |
| UNIVERSE-003    | FIRST/RANDOM request counts                                                         | {'FIRST': 6000, 'RANDOM': 6000}                                                                                             | {'FIRST': 6000, 'RANDOM': 6000}                                                                                 | PASS   |
| UNIVERSE-004    | terminal-state counts match frozen execution accounting                             | {'HALLUCINATION_OR_OUT_OF_CANDIDATE': 1, 'INVALID_DUPLICATE': 51, 'INVALID_LENGTH': 114, 'STRICT_VALID': 11834}             | {'STRICT_VALID': 11834, 'INVALID_LENGTH': 114, 'INVALID_DUPLICATE': 51, 'HALLUCINATION_OR_OUT_OF_CANDIDATE': 1} | PASS   |
| UNIVERSE-005    | backend errors remain zero                                                          | 0                                                                                                                           | 0                                                                                                               | PASS   |
| UNIVERSE-006    | no missing duplicate ambiguous evidence in analysis manifest                        | []                                                                                                                          | []                                                                                                              | PASS   |
| PAIR-001        | frozen pair row count                                                               | 6000                                                                                                                        | 6000                                                                                                            | PASS   |
| PAIR-002        | frozen FIRST/RANDOM pair membership unchanged                                       | 6000                                                                                                                        | 6000                                                                                                            | PASS   |
| PAIR-003        | FIRST positive position is 1 and RANDOM positive position is 2..20                  | 6000                                                                                                                        | 6000                                                                                                            | PASS   |
| UTILITY-001     | request-level utility contributions cover all logical rows                          | 12000                                                                                                                       | 12000                                                                                                           | PASS   |
| UTILITY-002     | invalid terminal utility contributions are zero                                     | 0                                                                                                                           | 0                                                                                                               | PASS   |
| UTILITY-003     | utility result row count is complete                                                | 180                                                                                                                         | 180                                                                                                             | PASS   |
| UTILITY-004     | HR/NDCG/MRR each have all expected cells                                            | {'HR@5': 60, 'MRR': 60, 'NDCG@5': 60}                                                                                       | {'HR@5': 60, 'NDCG@5': 60, 'MRR': 60}                                                                           | PASS   |
| UTILITY-005     | utility denominators are 200 planned users per order cell                           | ['200']                                                                                                                     | ['200']                                                                                                         | PASS   |
| CANDDIF-001     | CandDif result rows cover HR@5 and NDCG@5 cells                                     | 60                                                                                                                          | 60                                                                                                              | PASS   |
| CANDDIF-002     | CandDif metric-specific cell counts are complete                                    | {'CandDif_HR@5': 30, 'CandDif_NDCG@5': 30}                                                                                  | {'CandDif_HR@5': 30, 'CandDif_NDCG@5': 30}                                                                      | PASS   |
| CANDDIF-003     | CandDif paired-user denominator is 200 for every dataset/model/prompt               | ['200']                                                                                                                     | ['200']                                                                                                         | PASS   |
| RELIABILITY-001 | reliability summary rows cover required scopes                                      | 25                                                                                                                          | 25                                                                                                              | PASS   |
| DENOM-001       | denominator table covers utility CandDif reliability and beyond-utility policy rows | 116                                                                                                                         | 116                                                                                                             | PASS   |
| PHASE-001       | zero inference and no bootstrap/statistical tests declared                          | zero_inference=true;statistical_analysis_executed=false                                                                     | zero_inference=true;statistical_analysis_executed=false                                                         | PASS   |

## Outputs

- `results/tables/g3_candidate_position_metric_results.csv`
- `results/tables/g3_candidate_position_canddif.csv`
- `results/tables/g3_candidate_position_reliability.csv`
- `results/tables/g3_candidate_position_metric_denominators.csv`
- `results/tables/g3_candidate_position_metric_validation.csv`
- `results/manifests/g3_candidate_position_metrics_freeze.json`
