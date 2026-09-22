# G3 Frozen Statistical Analysis

- Phase: `g3_candidate_position_statistical_analysis`
- Decision: `G3_STATISTICAL_ANALYSIS_FROZEN`
- HEAD before analysis: `e0c9b2afbbdfca2b9824e2001fd3f0703d440726`
- Scope: frozen G3 paired universe, frozen metric artifacts, and frozen statistical plan only.
- Zero Ollama/API calls; no protocol, evidence, universe, or metric changes.

## Plan Coverage

- Frozen FIRST-vs-RANDOM comparisons executed: `90` / `90`.
- Paired-user contribution rows: `18000`.
- Preserved paired users: `6000` / `6000`.
- Contrast: `FIRST_minus_RANDOM`.

| metric    | comparisons |
| --------- | ----------- |
| hr_at_5   | 30          |
| mrr       | 30          |
| ndcg_at_5 | 30          |

## Bootstrap Contract Used

- Primary test: `paired_user_bootstrap_mean_delta`.
- Resamples: `10000`.
- Seed: `20260901`.
- Confidence interval: `two_sided_95_percent_percentile_ci`.
- Percentile implementation detail: `numpy_quantile_linear`.
- Bootstrap engine: `empirical_multinomial_equivalent_to_paired_user_resampling_for_mean_delta`.

## Decisions

- Primary CI decisions: `{"CI_SUPPORTED_NEGATIVE":1,"CI_SUPPORTED_POSITIVE":85,"NON_SIGNIFICANT_CI_INCLUDES_ZERO":4}`.
- Corrected decisions: `{"NOT_DEFINED_NO_FROZEN_P_VALUE_METHOD_FOR_G3":90}`.
- Multiplicity status: `NOT_APPLIED_NO_FROZEN_P_VALUE_METHOD_FOR_G3`.
- CandDif HR/NDCG retained as descriptive cell-level measures only; no CandDif significance test computed.

### By Prompt

| prompt_strategy | CI_SUPPORTED_POSITIVE | CI_SUPPORTED_NEGATIVE | NON_SIGNIFICANT_CI_INCLUDES_ZERO |
| --------------- | --------------------- | --------------------- | -------------------------------- |
| BASE            | 45                    | 0                     | 0                                |
| LLMRANK         | 40                    | 1                     | 4                                |

### By Model

| model               | CI_SUPPORTED_POSITIVE | CI_SUPPORTED_NEGATIVE | NON_SIGNIFICANT_CI_INCLUDES_ZERO |
| ------------------- | --------------------- | --------------------- | -------------------------------- |
| gemma2:9b           | 30                    | 0                     | 0                                |
| mistral:7b-instruct | 25                    | 1                     | 4                                |
| qwen2:7b-instruct   | 30                    | 0                     | 0                                |

### By Dataset

| dataset | CI_SUPPORTED_POSITIVE | CI_SUPPORTED_NEGATIVE | NON_SIGNIFICANT_CI_INCLUDES_ZERO |
| ------- | --------------------- | --------------------- | -------------------------------- |
| beauty  | 18                    | 0                     | 0                                |
| foodcom | 18                    | 0                     | 0                                |
| lastfm  | 16                    | 0                     | 2                                |
| ml1m    | 15                    | 1                     | 2                                |
| sports  | 18                    | 0                     | 0                                |

## Validation

- PASS: `34` / `34`.

| check_id                                    | check_name                                                  | status | expected                                         | actual                                                                                                   | notes                                                                                                           |
| ------------------------------------------- | ----------------------------------------------------------- | ------ | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| G3STAT-CHK-9836E5A6EE0AFDE899AF6E52F3BDA61B | head_matches_precondition                                   | PASS   | e0c9b2afbbdfca2b9824e2001fd3f0703d440726         | e0c9b2afbbdfca2b9824e2001fd3f0703d440726                                                                 |                                                                                                                 |
| G3STAT-CHK-F85E5C5CFE2E72175AACE4FB0E4C82BE | git_clean_or_expected_statistical_outputs_before_generation | PASS   | clean_or_expected_dirty                          | src/extension/g3_candidate_position_statistical_analysis.py                                              |                                                                                                                 |
| G3STAT-CHK-30558A2F7F38C018AC75A2306F2F8D72 | git_diff_check_clean_before_generation                      | PASS   | 0                                                | 0                                                                                                        |                                                                                                                 |
| G3STAT-CHK-C58AAF08F7AD9090D877DB9772A2AFD0 | protocol_manifest_frozen                                    | PASS   | G3_BASE_LLMRANK_FIRST_RANDOM_PROTOCOL_FROZEN     | G3_BASE_LLMRANK_FIRST_RANDOM_PROTOCOL_FROZEN                                                             |                                                                                                                 |
| G3STAT-CHK-D19A2CCE6700FE722EDAEF822D92EC43 | execution_manifest_frozen                                   | PASS   | G3_EXECUTION_RUNNER_FROZEN                       | G3_EXECUTION_RUNNER_FROZEN                                                                               |                                                                                                                 |
| G3STAT-CHK-21D0560F19A44CB9CEC24F72C06FD02A | retention_manifest_frozen                                   | PASS   | G3_EXECUTION_RETENTION_FROZEN                    | G3_EXECUTION_RETENTION_FROZEN                                                                            |                                                                                                                 |
| G3STAT-CHK-EAF191C3B8FCC6419F5A07686FF1522F | analysis_manifest_frozen                                    | PASS   | G3_ANALYTIC_UNIVERSE_FROZEN                      | G3_ANALYTIC_UNIVERSE_FROZEN                                                                              |                                                                                                                 |
| G3STAT-CHK-02C20E70911A7783943FC27AF3810DDE | metrics_manifest_frozen                                     | PASS   | G3_METRICS_FROZEN                                | G3_METRICS_FROZEN                                                                                        |                                                                                                                 |
| G3STAT-CHK-BD027C284F85FB67D47F453A7FF3CEFF | frozen_artifact_hashes_match                                | PASS   | >0 hash checks and no mismatch failures          | 54                                                                                                       |                                                                                                                 |
| G3STAT-CHK-D030CF9395DAA9C7B4C37101609E1717 | statistical_plan_rows_90                                    | PASS   | 90                                               | 90                                                                                                       |                                                                                                                 |
| G3STAT-CHK-F9CBC0A948B9DB94FCA3C471CFDBA674 | statistical_plan_metric_counts                              | PASS   | {"hr_at_5":30,"mrr":30,"ndcg_at_5":30}           | {"hr_at_5":30,"mrr":30,"ndcg_at_5":30}                                                                   |                                                                                                                 |
| G3STAT-CHK-DDC6F1BACA3C9992AF620DB75DC880D5 | statistical_plan_parameters_frozen                          | PASS   | frozen G3 bootstrap/contrast parameters          | {"contrast":["FIRST_minus_RANDOM"],"resamples":["10000"],"seed":["20260901"]}                            |                                                                                                                 |
| G3STAT-CHK-4EEC129406FDC8E2F9E11F15F858EB62 | g3_secondary_p_values_not_frozen                            | PASS   | not_frozen_for_g3_no_secondary_p_values          | {"not_frozen_for_g3_no_secondary_p_values":90}                                                           | Holm correction is retained as planned metadata only; corrected decisions require p-values that are not frozen. |
| G3STAT-CHK-FBE5848DEB6FD58E9EA8CDE9AC45C015 | analysis_universe_rows_12000                                | PASS   | 12000                                            | 12000                                                                                                    |                                                                                                                 |
| G3STAT-CHK-F78256E8EC3E0D921416AF7C2239ABDD | analysis_pair_rows_6000                                     | PASS   | 6000                                             | 6000                                                                                                     |                                                                                                                 |
| G3STAT-CHK-47F97D0DDA351DEA6FD4FB80B6EB6C7F | analysis_pair_membership_unchanged                          | PASS   | 6000                                             | 6000                                                                                                     |                                                                                                                 |
| G3STAT-CHK-A9F1BED6B959ABBAD3E3AD59156F5764 | first_random_order_counts                                   | PASS   | {"FIRST":6000,"RANDOM":6000}                     | {"FIRST":6000,"RANDOM":6000}                                                                             |                                                                                                                 |
| G3STAT-CHK-B461F56F764FAD887F44D868F6AEACAB | terminal_state_counts_match_g3_execution                    | PASS   | frozen G3 execution accounting                   | {"HALLUCINATION_OR_OUT_OF_CANDIDATE":1,"INVALID_DUPLICATE":51,"INVALID_LENGTH":114,"STRICT_VALID":11834} |                                                                                                                 |
| G3STAT-CHK-B6664A42AE7EB5D48F322A69B8616BFD | metric_results_rows_180                                     | PASS   | 180                                              | 180                                                                                                      |                                                                                                                 |
| G3STAT-CHK-5608B4929E8AC4378973B308BA113923 | canddif_rows_60_retained_descriptive                        | PASS   | 60                                               | 60                                                                                                       |                                                                                                                 |
| G3STAT-CHK-6EFA7B2FEBA931AF418C6D154907466B | paired_contribution_rows_18000                              | PASS   | 18000                                            | 18000                                                                                                    |                                                                                                                 |
| G3STAT-CHK-2409E12418035C4E192967EAF120C3AA | paired_contribution_ids_unique                              | PASS   | 0 duplicates                                     | 0                                                                                                        |                                                                                                                 |
| G3STAT-CHK-E9E641E9C93984C873222682B3695CF6 | all_6000_pairs_preserved_with_three_metrics                 | PASS   | {"3":6000}                                       | {"3":6000}                                                                                               |                                                                                                                 |
| G3STAT-CHK-E82658A5C4C5016578FAAE1E8423081C | each_comparison_has_200_paired_users                        | PASS   | 90 comparisons each with 200                     | {"comparisons":90,"paired_n_values":[200]}                                                               |                                                                                                                 |
| G3STAT-CHK-2842529DDC5892780A9E395E1706B8A4 | statistical_results_rows_90                                 | PASS   | 90                                               | 90                                                                                                       |                                                                                                                 |
| G3STAT-CHK-69AA2FF0613AAB73CB23C13944231B24 | comparison_coverage_90_of_90                                | PASS   | 90                                               | 90                                                                                                       |                                                                                                                 |
| G3STAT-CHK-6963B82871E01EF433787E69B39E45E2 | paired_n_200_all_results                                    | PASS   | ["200"]                                          | ["200"]                                                                                                  |                                                                                                                 |
| G3STAT-CHK-0F62660D543FE4F03E5172AB6A09236F | bootstrap_settings_used                                     | PASS   | 10000 resamples seed 20260901                    | {"resamples":["10000"],"seed":["20260901"]}                                                              |                                                                                                                 |
| G3STAT-CHK-4E66E51B5B7123CB6D5ACB31953E8864 | bootstrap_reproducibility_verified                          | PASS   | no bootstrap reproducibility failures            | []                                                                                                       |                                                                                                                 |
| G3STAT-CHK-E0CC2E042BC48338481F9CE5B0CE9F18 | metric_means_match_frozen_metric_table                      | PASS   | 180                                              | 180                                                                                                      |                                                                                                                 |
| G3STAT-CHK-9C1E90271F20C9E462742F463021E0D0 | no_p_values_or_holm_decisions_invented                      | PASS   | blank p-values and undefined corrected decisions | {"NOT_DEFINED_NO_FROZEN_P_VALUE_METHOD_FOR_G3":90}                                                       |                                                                                                                 |
| G3STAT-CHK-AC1F38047402A22ADC25969059BBF20C | primary_ci_decisions_defined                                | PASS   | all non-empty                                    | {"CI_SUPPORTED_NEGATIVE":1,"CI_SUPPORTED_POSITIVE":85,"NON_SIGNIFICANT_CI_INCLUDES_ZERO":4}              |                                                                                                                 |
| G3STAT-CHK-358F3CF42689131A08353B2ACBEB506B | no_ollama_or_api_calls_executed                             | PASS   | 0                                                | 0                                                                                                        |                                                                                                                 |
| G3STAT-CHK-CBD557C8B849DD59EECF2943E0D650D5 | no_downstream_g3_or_g4_started                              | PASS   | false                                            | false                                                                                                    |                                                                                                                 |

## Output Artifacts

- `results/tables/g3_candidate_position_statistical_paired_contributions.csv` SHA256 `5b5080c76cd90caca1db4fde7b574e65e677702fbd6ed7bc07ef391ec42a8a8a`
- `results/tables/g3_candidate_position_statistical_results.csv` SHA256 `2b2e1fa3d661fa1c4c2e81d56350548eeb33d1fb9594ef5b924b548af7a555a3`
- `results/tables/g3_candidate_position_statistical_validation.csv` SHA256 `9108202377fd9b2f40c26b7279f40606e7982614b502e8ed2bc6cedb941cccd8`

## Next Safe Step

Generate the frozen G3 scientific readout/synthesis from the frozen metric and statistical artifacts.
