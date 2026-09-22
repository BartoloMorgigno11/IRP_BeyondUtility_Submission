# G2 Frozen Statistical Analysis

- Phase: `g2_history_length_statistical_analysis`
- Decision: `G2_STATISTICAL_ANALYSIS_FROZEN`
- HEAD before analysis: `c120b788d2023e29027eccc6dbf9e38f5676c537`
- Scope: frozen retained G2/G1 evidence, universe, metric contracts, readout contract, and statistical plan only.
- Zero Ollama calls; no G3; no protocol, runner, evidence, universe, metric, or readout changes.

## Plan Coverage

- Frozen comparison templates executed: `27` / `27`.
- Dataset x model x prompt statistical result rows: `810`.
- Paired-user contribution rows: `122382`.
- Utility metrics: `["hr_at_5","ndcg_at_5","mrr"]`.

## Bootstrap Contract Used

- Primary test: `paired_user_bootstrap_mean_delta`.
- Resamples: `10000`.
- Seed: `20260901`.
- Confidence interval: `two_sided_95_percent_percentile_ci`.
- Percentile implementation detail: `numpy_quantile_linear`.
- Bootstrap engine: `empirical_multinomial_equivalent_to_paired_user_resampling_for_mean_delta`.

## Paired User Counts

| view | dataset | paired users |
|---|---|---:|
| LONG_HISTORY_B2_GE20 | beauty | 38 |
| LONG_HISTORY_B2_GE20 | foodcom | 35 |
| LONG_HISTORY_B2_GE20 | lastfm | 932 |
| LONG_HISTORY_B2_GE20 | ml1m | 993 |
| LONG_HISTORY_B2_GE20 | sports | 35 |
| PRIMARY_B1_GE10 | beauty | 181 |
| PRIMARY_B1_GE10 | foodcom | 61 |
| PRIMARY_B1_GE10 | lastfm | 981 |
| PRIMARY_B1_GE10 | ml1m | 1000 |
| PRIMARY_B1_GE10 | sports | 160 |
| SECONDARY_FULL_CURVE_GE20 | beauty | 38 |
| SECONDARY_FULL_CURVE_GE20 | foodcom | 35 |
| SECONDARY_FULL_CURVE_GE20 | lastfm | 932 |
| SECONDARY_FULL_CURVE_GE20 | ml1m | 993 |
| SECONDARY_FULL_CURVE_GE20 | sports | 35 |

## CI Decision Counts

- Overall primary CI decisions: `{"NOT_SIGNIFICANT_CI_INCLUDES_ZERO":499,"SIGNIFICANT_NEGATIVE":67,"SIGNIFICANT_POSITIVE":244}`.
- Corrected p-value decisions: `{"NOT_DEFINED_NO_FROZEN_P_VALUE_METHOD":810}`.
- Corrected p-values are not computed because no frozen primary p-value method exists in the G2 plan.

| view | metric | CI decision | result rows |
|---|---|---|---:|
| LONG_HISTORY_B2_GE20 | hr_at_5 | NOT_SIGNIFICANT_CI_INCLUDES_ZERO | 25 |
| LONG_HISTORY_B2_GE20 | hr_at_5 | SIGNIFICANT_NEGATIVE | 5 |
| LONG_HISTORY_B2_GE20 | mrr | NOT_SIGNIFICANT_CI_INCLUDES_ZERO | 21 |
| LONG_HISTORY_B2_GE20 | mrr | SIGNIFICANT_NEGATIVE | 7 |
| LONG_HISTORY_B2_GE20 | mrr | SIGNIFICANT_POSITIVE | 2 |
| LONG_HISTORY_B2_GE20 | ndcg_at_5 | NOT_SIGNIFICANT_CI_INCLUDES_ZERO | 23 |
| LONG_HISTORY_B2_GE20 | ndcg_at_5 | SIGNIFICANT_NEGATIVE | 6 |
| LONG_HISTORY_B2_GE20 | ndcg_at_5 | SIGNIFICANT_POSITIVE | 1 |
| PRIMARY_B1_GE10 | hr_at_5 | NOT_SIGNIFICANT_CI_INCLUDES_ZERO | 115 |
| PRIMARY_B1_GE10 | hr_at_5 | SIGNIFICANT_NEGATIVE | 6 |
| PRIMARY_B1_GE10 | hr_at_5 | SIGNIFICANT_POSITIVE | 59 |
| PRIMARY_B1_GE10 | mrr | NOT_SIGNIFICANT_CI_INCLUDES_ZERO | 102 |
| PRIMARY_B1_GE10 | mrr | SIGNIFICANT_NEGATIVE | 11 |
| PRIMARY_B1_GE10 | mrr | SIGNIFICANT_POSITIVE | 67 |
| PRIMARY_B1_GE10 | ndcg_at_5 | NOT_SIGNIFICANT_CI_INCLUDES_ZERO | 105 |
| PRIMARY_B1_GE10 | ndcg_at_5 | SIGNIFICANT_NEGATIVE | 9 |
| PRIMARY_B1_GE10 | ndcg_at_5 | SIGNIFICANT_POSITIVE | 66 |
| SECONDARY_FULL_CURVE_GE20 | hr_at_5 | NOT_SIGNIFICANT_CI_INCLUDES_ZERO | 38 |
| SECONDARY_FULL_CURVE_GE20 | hr_at_5 | SIGNIFICANT_NEGATIVE | 6 |
| SECONDARY_FULL_CURVE_GE20 | hr_at_5 | SIGNIFICANT_POSITIVE | 16 |
| SECONDARY_FULL_CURVE_GE20 | mrr | NOT_SIGNIFICANT_CI_INCLUDES_ZERO | 33 |
| SECONDARY_FULL_CURVE_GE20 | mrr | SIGNIFICANT_NEGATIVE | 10 |
| SECONDARY_FULL_CURVE_GE20 | mrr | SIGNIFICANT_POSITIVE | 17 |
| SECONDARY_FULL_CURVE_GE20 | ndcg_at_5 | NOT_SIGNIFICANT_CI_INCLUDES_ZERO | 37 |
| SECONDARY_FULL_CURVE_GE20 | ndcg_at_5 | SIGNIFICANT_NEGATIVE | 7 |
| SECONDARY_FULL_CURVE_GE20 | ndcg_at_5 | SIGNIFICANT_POSITIVE | 16 |

## Validation

- Validation checks: `34` / `34` PASS.
- Frozen input hash checks: `56`.
- Frozen metric mean checks: `1620`.

## Output Artifacts

- `report/g2_history_length_statistical_analysis.md` SHA256 `5959e632822764327c23851be04ae7c335d96d9bd5b39d827ea54d1ad4e20de9`
- `results/tables/g2_history_length_statistical_paired_contributions.csv` SHA256 `ad089c3353a6ea761ef245e787333f4d7c3cfcc593ed9a7567ef2853c8281f7e`
- `results/tables/g2_history_length_statistical_results.csv` SHA256 `55fc45b824489b43da69373c2c56a821e4be874f24f146ef29bf081f7736fe5e`
- `results/tables/g2_history_length_statistical_validation.csv` SHA256 `909ad199dad91928775b93701f76af6f553fd7190227718e8881979c17fecdc4`

## Next Safe Step

Integrate the frozen descriptive and statistical G2 readouts into the final report. Do not start G3 unless separately authorized.
