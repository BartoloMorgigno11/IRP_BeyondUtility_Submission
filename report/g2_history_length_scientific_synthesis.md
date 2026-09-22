# G2 Final Scientific Synthesis

- Phase: `g2_history_length_scientific_synthesis`
- Decision: `G2_SCIENTIFIC_SYNTHESIS_FROZEN`
- HEAD before synthesis: `3a083be53b5333a83deff619ffe150610ed12951`
- Inputs: frozen G2 descriptive readout and frozen G2 statistical analysis artifacts only.
- No Ollama calls, no G3, no new tests, no new hypotheses, and no p-values invented.

## Statistical Basis

- Frozen primary decision basis: paired-user bootstrap mean-delta 95 percent percentile CI.
- Corrected p-value decisions remain `NOT_DEFINED_NO_FROZEN_P_VALUE_METHOD` because the frozen plan did not define a primary p-value method.
- Frozen statistical result rows: `810` across `27` comparison templates.

## Every Frozen Comparison Template

| analysis_membership | contrast | metric | significant_positive | significant_negative | non_significant | mean_delta_across_cells | median_delta_across_cells |
|---|---|---|---|---|---|---|---|
| LONG_HISTORY_B2_GE20 | H20_minus_H10 | hr_at_5 | 0 | 5 | 25 | -0.015204938067716386 | -0.0118025751072961 |
| LONG_HISTORY_B2_GE20 | H20_minus_H10 | mrr | 2 | 7 | 21 | -0.016919007347379968 | -0.019860066833751001 |
| LONG_HISTORY_B2_GE20 | H20_minus_H10 | ndcg_at_5 | 1 | 6 | 23 | -0.011487461119742538 | -0.01396829605528175 |
| PRIMARY_B1_GE10 | H1_minus_H0 | hr_at_5 | 12 | 0 | 18 | 0.073179984369567672 | 0.049590163934426203 |
| PRIMARY_B1_GE10 | H1_minus_H0 | mrr | 16 | 1 | 13 | 0.050970453041434369 | 0.043304021530070555 |
| PRIMARY_B1_GE10 | H1_minus_H0 | ndcg_at_5 | 15 | 1 | 14 | 0.055996210680488039 | 0.038115795764893552 |
| PRIMARY_B1_GE10 | H10_minus_H0 | hr_at_5 | 17 | 0 | 13 | 0.085724995459114564 | 0.079657639706548239 |
| PRIMARY_B1_GE10 | H10_minus_H0 | mrr | 20 | 1 | 9 | 0.06208180596925035 | 0.048089754889064251 |
| PRIMARY_B1_GE10 | H10_minus_H0 | ndcg_at_5 | 20 | 0 | 10 | 0.068350487733131376 | 0.054924925843686953 |
| PRIMARY_B1_GE10 | H5_minus_H0 | hr_at_5 | 13 | 0 | 17 | 0.081655767683840336 | 0.06275 |
| PRIMARY_B1_GE10 | H5_minus_H0 | mrr | 16 | 0 | 14 | 0.061707308615137374 | 0.047650395199063202 |
| PRIMARY_B1_GE10 | H5_minus_H0 | ndcg_at_5 | 15 | 0 | 15 | 0.066079602647112767 | 0.045974627407690351 |
| PRIMARY_B1_GE10 | H2_minus_H1 | hr_at_5 | 10 | 0 | 20 | 0.019911447402836669 | 0.025213047910295598 |
| PRIMARY_B1_GE10 | H2_minus_H1 | mrr | 8 | 2 | 20 | 0.0073422470797195301 | 0.0037037567957865498 |
| PRIMARY_B1_GE10 | H2_minus_H1 | ndcg_at_5 | 10 | 2 | 18 | 0.011910624794851223 | 0.0081133829343416498 |
| PRIMARY_B1_GE10 | H5_minus_H2 | hr_at_5 | 4 | 4 | 22 | -0.011435664088563993 | 0.0015 |
| PRIMARY_B1_GE10 | H5_minus_H2 | mrr | 4 | 1 | 25 | 0.0033946084939834777 | 0.0032309523809523001 |
| PRIMARY_B1_GE10 | H5_minus_H2 | ndcg_at_5 | 3 | 2 | 25 | -0.0018272328282265002 | -1.6294151238999978e-06 |
| PRIMARY_B1_GE10 | H10_minus_H5 | hr_at_5 | 3 | 2 | 25 | 0.0040692277752742093 | 0.0025484199796125999 |
| PRIMARY_B1_GE10 | H10_minus_H5 | mrr | 3 | 6 | 21 | 0.00037449735411297831 | -0.00095624403346110008 |
| PRIMARY_B1_GE10 | H10_minus_H5 | ndcg_at_5 | 3 | 4 | 23 | 0.0022708850860186035 | 0.0042026201046743497 |
| SECONDARY_FULL_CURVE_GE20 | H20_minus_H0 | hr_at_5 | 13 | 1 | 16 | 0.057336022632899712 | 0.04330312185297075 |
| SECONDARY_FULL_CURVE_GE20 | H20_minus_H0 | mrr | 14 | 1 | 15 | 0.030932123259181554 | 0.015385487528344599 |
| SECONDARY_FULL_CURVE_GE20 | H20_minus_H0 | ndcg_at_5 | 13 | 1 | 16 | 0.040263676935146188 | 0.018398581388225199 |
| SECONDARY_FULL_CURVE_GE20 | H20_minus_H5 | hr_at_5 | 3 | 5 | 22 | -0.015716781429259166 | -0.02120510503727125 |
| SECONDARY_FULL_CURVE_GE20 | H20_minus_H5 | mrr | 3 | 9 | 18 | -0.01842344074580303 | -0.014954911135316801 |
| SECONDARY_FULL_CURVE_GE20 | H20_minus_H5 | ndcg_at_5 | 3 | 6 | 21 | -0.012907157472585905 | -0.0090402051308636003 |

## Primary And Long-History Reconciliation

| analysis_membership | contrast | descriptive_improved | descriptive_no_change | descriptive_worsened | significant_positive | significant_negative | non_significant | median_delta_across_cells |
|---|---|---|---|---|---|---|---|---|
| PRIMARY_B1_GE10 | H10_minus_H0 | 80 | 1 | 9 | 57 | 1 | 32 | 0.055918898809523804 |
| LONG_HISTORY_B2_GE20 | H20_minus_H10 | 27 | 4 | 59 | 3 | 18 | 69 | -0.01513145221423025 |

Descriptive sign counts describe the direction of cell-level deltas. CI-supported counts require the frozen paired-user bootstrap CI to exclude zero.

## Key Adjacent Comparisons

| analysis_membership | contrast | metric | significant_positive | significant_negative | non_significant |
|---|---|---|---|---|---|
| LONG_HISTORY_B2_GE20 | H20_minus_H10 | hr_at_5 | 0 | 5 | 25 |
| LONG_HISTORY_B2_GE20 | H20_minus_H10 | mrr | 2 | 7 | 21 |
| LONG_HISTORY_B2_GE20 | H20_minus_H10 | ndcg_at_5 | 1 | 6 | 23 |
| PRIMARY_B1_GE10 | H1_minus_H0 | hr_at_5 | 12 | 0 | 18 |
| PRIMARY_B1_GE10 | H1_minus_H0 | mrr | 16 | 1 | 13 |
| PRIMARY_B1_GE10 | H1_minus_H0 | ndcg_at_5 | 15 | 1 | 14 |
| PRIMARY_B1_GE10 | H10_minus_H0 | hr_at_5 | 17 | 0 | 13 |
| PRIMARY_B1_GE10 | H10_minus_H0 | mrr | 20 | 1 | 9 |
| PRIMARY_B1_GE10 | H10_minus_H0 | ndcg_at_5 | 20 | 0 | 10 |
| PRIMARY_B1_GE10 | H2_minus_H1 | hr_at_5 | 10 | 0 | 20 |
| PRIMARY_B1_GE10 | H2_minus_H1 | mrr | 8 | 2 | 20 |
| PRIMARY_B1_GE10 | H2_minus_H1 | ndcg_at_5 | 10 | 2 | 18 |
| PRIMARY_B1_GE10 | H5_minus_H2 | hr_at_5 | 4 | 4 | 22 |
| PRIMARY_B1_GE10 | H5_minus_H2 | mrr | 4 | 1 | 25 |
| PRIMARY_B1_GE10 | H5_minus_H2 | ndcg_at_5 | 3 | 2 | 25 |
| PRIMARY_B1_GE10 | H10_minus_H5 | hr_at_5 | 3 | 2 | 25 |
| PRIMARY_B1_GE10 | H10_minus_H5 | mrr | 3 | 6 | 21 |
| PRIMARY_B1_GE10 | H10_minus_H5 | ndcg_at_5 | 3 | 4 | 23 |

## Dataset / Model / Prompt Breakdowns

### dataset

| analysis_membership | contrast | breakdown_value | significant_positive | significant_negative | non_significant | descriptive_improved | descriptive_worsened |
|---|---|---|---|---|---|---|---|
| LONG_HISTORY_B2_GE20 | H20_minus_H10 | beauty | 0 | 1 | 17 | 3 | 15 |
| LONG_HISTORY_B2_GE20 | H20_minus_H10 | foodcom | 0 | 0 | 18 | 6 | 11 |
| LONG_HISTORY_B2_GE20 | H20_minus_H10 | lastfm | 0 | 11 | 7 | 2 | 16 |
| LONG_HISTORY_B2_GE20 | H20_minus_H10 | ml1m | 2 | 6 | 10 | 7 | 11 |
| LONG_HISTORY_B2_GE20 | H20_minus_H10 | sports | 1 | 0 | 17 | 9 | 6 |
| PRIMARY_B1_GE10 | H10_minus_H0 | beauty | 8 | 0 | 10 | 16 | 2 |
| PRIMARY_B1_GE10 | H10_minus_H0 | foodcom | 7 | 1 | 10 | 15 | 3 |
| PRIMARY_B1_GE10 | H10_minus_H0 | lastfm | 15 | 0 | 3 | 16 | 2 |
| PRIMARY_B1_GE10 | H10_minus_H0 | ml1m | 18 | 0 | 0 | 18 | 0 |
| PRIMARY_B1_GE10 | H10_minus_H0 | sports | 9 | 0 | 9 | 15 | 2 |

### model

| analysis_membership | contrast | breakdown_value | significant_positive | significant_negative | non_significant | descriptive_improved | descriptive_worsened |
|---|---|---|---|---|---|---|---|
| LONG_HISTORY_B2_GE20 | H20_minus_H10 | gemma2:9b | 1 | 12 | 17 | 9 | 20 |
| LONG_HISTORY_B2_GE20 | H20_minus_H10 | mistral:7b-instruct | 2 | 3 | 25 | 10 | 19 |
| LONG_HISTORY_B2_GE20 | H20_minus_H10 | qwen2:7b-instruct | 0 | 3 | 27 | 8 | 20 |
| PRIMARY_B1_GE10 | H10_minus_H0 | gemma2:9b | 25 | 0 | 5 | 30 | 0 |
| PRIMARY_B1_GE10 | H10_minus_H0 | mistral:7b-instruct | 11 | 1 | 18 | 25 | 5 |
| PRIMARY_B1_GE10 | H10_minus_H0 | qwen2:7b-instruct | 21 | 0 | 9 | 25 | 4 |

### prompt_strategy

| analysis_membership | contrast | breakdown_value | significant_positive | significant_negative | non_significant | descriptive_improved | descriptive_worsened |
|---|---|---|---|---|---|---|---|
| LONG_HISTORY_B2_GE20 | H20_minus_H10 | BASE | 0 | 7 | 38 | 10 | 33 |
| LONG_HISTORY_B2_GE20 | H20_minus_H10 | LLMRANK | 3 | 11 | 31 | 17 | 26 |
| PRIMARY_B1_GE10 | H10_minus_H0 | BASE | 30 | 0 | 15 | 40 | 4 |
| PRIMARY_B1_GE10 | H10_minus_H0 | LLMRANK | 27 | 1 | 17 | 40 | 5 |

## CI-Supported Scientific Statements

- `history_helps`: In the primary GE10 analysis, H10 versus H0 shows many more CI-supported positive than negative utility effects across the frozen HR@5/NDCG@5/MRR cells.
- `long_history_degradation`: In the long-history GE20 extension, H20 versus H10 is more often CI-supported negative than positive for utility.
- `saturation`: The frozen adjacent-H comparisons support early gains from adding short history, while later primary transitions have mixed or mostly non-significant CI support.
- `prompt_dependence`: BASE and LLMRANK both show primary H10-versus-H0 positive CI-supported effects, but the long H20-versus-H10 degradation is stronger for LLMRANK in the frozen CI counts.
- `dataset_dependence`: Primary H10-versus-H0 support is dataset-dependent: all five datasets have positive CI-supported cells, but the counts are uneven across datasets.
- `model_dependence`: All three models have primary H10-versus-H0 positive CI-supported cells, with the strongest count concentration in Gemma2.

## Validation

- Validation checks: `22` / `22` PASS.

## Output Artifacts

- `report/g2_history_length_scientific_synthesis.md` SHA256 `d24e57a012aa53e30aa38dcc20b97717088509f02e6fa7c0dd20dc27518b305d`
- `results/tables/g2_history_length_scientific_synthesis_breakdowns.csv` SHA256 `063593de8e56b2a09335b2e79a985fc48b2e68cbc319a7e1255fbb3bf5c5dbc5`
- `results/tables/g2_history_length_scientific_synthesis_comparison_summary.csv` SHA256 `ba4ab12c43906f7059037fbec2cf345bd7b89e2c61d2ab7238b1b49ca3ef6ebb`
- `results/tables/g2_history_length_scientific_synthesis_focus_reconciliation.csv` SHA256 `af8f2db79930105400979f8444f2adefb7a11a7ff6c030a2ad6db661d24c9ad8`
- `results/tables/g2_history_length_scientific_synthesis_supported_statements.csv` SHA256 `76c2c78b8a9ebbd6b2229844b16494f2a2736b5e83b407b73e6707ae9bb36b2b`
- `results/tables/g2_history_length_scientific_synthesis_validation.csv` SHA256 `f6aff8764e28a52a72cfcedf8da245f77c7201f251748b7ea0494da48f2af798`

## Next Safe Step

Use the frozen G2 synthesis for report integration. No relation rewrite or G3 has been started.
