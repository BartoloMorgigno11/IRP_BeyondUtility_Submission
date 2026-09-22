# H0 Final History-Length Parity Audit

Phase token: `HISTORY_FINAL_PARITY_READY`

No LLM inference, recommender training, downloads, network calls, fifth-dataset
selection, or commits were performed.

## Final Contract

Datasets: Beauty, Sports, ML-1M, LastFM.
Models: Qwen2 7B, Mistral 7B, Gemma2 9B.
Lengths: 0, 1, 2, 5, 10, 20.
Prompt: BASE only. Candidate pool: 20. Requested output: 10 unique positions.
Parser: `llm_rank_strict_index_top10_v1`. Metric K: 5.

## Dataset Feasibility

| dataset   |   eligible_users_ge20 |   structurally_valid_long_history_users |   selected_users |   e0_manifest_rows |   e0_manifest_selected_users |   models | history_lengths   |   candidate_pool_20_users |   candidate_titles_valid_users |   strict_l1_users |   strict_l2_users |   strict_l5_users |   strict_l10_users |   strict_l20_users |   strict_paired_l20_users |   l0_empty_history_users |   same_candidates_all_l_users |   same_positive_all_l_users |   selection_seed | selection_rule                                                                              | status                    | blocker   | cohort_user_hash                                                 | cohort_candidate_hash                                            |
|:----------|----------------------:|----------------------------------------:|-----------------:|-------------------:|-----------------------------:|---------:|:------------------|--------------------------:|-------------------------------:|------------------:|------------------:|------------------:|-------------------:|-------------------:|--------------------------:|-------------------------:|------------------------------:|----------------------------:|-----------------:|:--------------------------------------------------------------------------------------------|:--------------------------|:----------|:-----------------------------------------------------------------|:-----------------------------------------------------------------|
| beauty    |                  1136 |                                    1136 |              200 |               3600 |                          200 |        3 | 0;1;2;5;10;20     |                      1136 |                           1121 |              1120 |              1120 |              1117 |               1114 |               1105 |                      1105 |                     1136 |                          1136 |                        1136 |         20260824 | sort_by_stable_int_seed(seed,'history_cohort',dataset,user_id),user_id over strict L20 pool | FEASIBLE_FINAL_PAIRED_200 |           | 11f92e68740ed57ad481eee8952ec5f8c8f29b2a01d36cf064f4fb61d3c4083e | 15859027aaab2949a32b1ebb5434bc3203fbbb45bdd0872421b409456b24e569 |
| sports    |                  1277 |                                    1277 |              200 |               3600 |                          200 |        3 | 0;1;2;5;10;20     |                      1277 |                           1146 |              1141 |              1135 |              1127 |               1108 |               1071 |                      1071 |                     1277 |                          1277 |                        1277 |         20260824 | sort_by_stable_int_seed(seed,'history_cohort',dataset,user_id),user_id over strict L20 pool | FEASIBLE_FINAL_PAIRED_200 |           | eae9e3cd3f4b93d4652f58f4c0b25b64bf8cfc033e5a88be79ec99effae4a2e4 | cd99501fee88b52f8548dfd1fa72f7595999282d5f63eb9b939b405403fa5d66 |
| ml1m      |                  5954 |                                    5954 |              200 |               3600 |                          200 |        3 | 0;1;2;5;10;20     |                      5954 |                           5954 |              5954 |              5954 |              5954 |               5954 |               5954 |                      5954 |                     5954 |                          5954 |                        5954 |         20260824 | sort_by_stable_int_seed(seed,'history_cohort',dataset,user_id),user_id over strict L20 pool | FEASIBLE_FINAL_PAIRED_200 |           | c0e76f67f11b85dcb76cf641c420ced590acfc6685c1b8beb94d70ac568319c0 | 3c97ed50379a1d538ef8e91d962367105f6c9001ff3c1ef139f301f4deba94c4 |
| lastfm    |                  1720 |                                    1720 |              200 |               3600 |                          200 |        3 | 0;1;2;5;10;20     |                      1720 |                           1720 |              1720 |              1720 |              1720 |               1720 |               1720 |                      1720 |                     1720 |                          1720 |                        1720 |         20260824 | sort_by_stable_int_seed(seed,'history_cohort',dataset,user_id),user_id over strict L20 pool | FEASIBLE_FINAL_PAIRED_200 |           | 09470c0b12b638de05916518c137434d626d5c64b54ff3c4afa7b61ae372e166 | 2c16ae0a88655d00c4e0d27df0db0e9a70e75c7e41a399723b7e0f11ea6f561c |

## Cell Matrix

| status     |   cells |
|:-----------|--------:|
| TO_COMPUTE |      72 |

## Request Budget

| scope       | dataset      | model_name   | users_per_dataset_model   | history_conditions   |   required_raw_requests |   reusable_identical_raw_requests |   to_compute_raw_requests |   blocked_raw_requests |   derived_metric_extra_requests |
|:------------|:-------------|:-------------|:--------------------------|:---------------------|------------------------:|----------------------------------:|--------------------------:|-----------------------:|--------------------------------:|
| grand_total | ALL_DATASETS | ALL_MODELS   |                           |                      |                   14400 |                                 0 |                     14400 |                      0 |                               0 |

## Paper vs Local Grid

| dataset   |   history_length | classification          | paper_alignment_note                                                                          | official_code_supported   | local_final_grid   | l0_semantics           |
|:----------|-----------------:|:------------------------|:----------------------------------------------------------------------------------------------|:--------------------------|:-------------------|:-----------------------|
| beauty    |                0 | OFFICIAL_CODE_SUPPORTED | paper history protocol exists, but exact paper length grid is not proven from local artifacts | True                      | True               | explicit_empty_history |
| beauty    |                1 | OFFICIAL_CODE_SUPPORTED | paper history protocol exists, but exact paper length grid is not proven from local artifacts | True                      | True               |                        |
| beauty    |                2 | OFFICIAL_CODE_SUPPORTED | paper history protocol exists, but exact paper length grid is not proven from local artifacts | True                      | True               |                        |
| beauty    |                5 | OFFICIAL_CODE_SUPPORTED | paper history protocol exists, but exact paper length grid is not proven from local artifacts | True                      | True               |                        |
| beauty    |               10 | OFFICIAL_CODE_SUPPORTED | paper history protocol exists, but exact paper length grid is not proven from local artifacts | True                      | True               |                        |
| beauty    |               20 | OFFICIAL_CODE_SUPPORTED | paper history protocol exists, but exact paper length grid is not proven from local artifacts | True                      | True               |                        |
| sports    |                0 | LOCAL_EXTENSION_GRID    | paper history protocol exists, but exact paper length grid is not proven from local artifacts | False                     | True               | explicit_empty_history |
| sports    |                1 | LOCAL_EXTENSION_GRID    | paper history protocol exists, but exact paper length grid is not proven from local artifacts | False                     | True               |                        |
| sports    |                2 | LOCAL_EXTENSION_GRID    | paper history protocol exists, but exact paper length grid is not proven from local artifacts | False                     | True               |                        |
| sports    |                5 | OFFICIAL_CODE_SUPPORTED | paper history protocol exists, but exact paper length grid is not proven from local artifacts | True                      | True               |                        |
| sports    |               10 | OFFICIAL_CODE_SUPPORTED | paper history protocol exists, but exact paper length grid is not proven from local artifacts | True                      | True               |                        |
| sports    |               20 | OFFICIAL_CODE_SUPPORTED | paper history protocol exists, but exact paper length grid is not proven from local artifacts | True                      | True               |                        |
| ml1m      |                0 | LOCAL_EXTENSION_GRID    | paper history protocol exists, but exact paper length grid is not proven from local artifacts | False                     | True               | explicit_empty_history |
| ml1m      |                1 | LOCAL_EXTENSION_GRID    | paper history protocol exists, but exact paper length grid is not proven from local artifacts | False                     | True               |                        |
| ml1m      |                2 | LOCAL_EXTENSION_GRID    | paper history protocol exists, but exact paper length grid is not proven from local artifacts | False                     | True               |                        |
| ml1m      |                5 | LOCAL_EXTENSION_GRID    | paper history protocol exists, but exact paper length grid is not proven from local artifacts | False                     | True               |                        |
| ml1m      |               10 | LOCAL_EXTENSION_GRID    | paper history protocol exists, but exact paper length grid is not proven from local artifacts | False                     | True               |                        |
| ml1m      |               20 | LOCAL_EXTENSION_GRID    | paper history protocol exists, but exact paper length grid is not proven from local artifacts | False                     | True               |                        |
| lastfm    |                0 | LOCAL_EXTENSION_GRID    | paper history protocol exists, but exact paper length grid is not proven from local artifacts | False                     | True               | explicit_empty_history |
| lastfm    |                1 | LOCAL_EXTENSION_GRID    | paper history protocol exists, but exact paper length grid is not proven from local artifacts | False                     | True               |                        |
| lastfm    |                2 | LOCAL_EXTENSION_GRID    | paper history protocol exists, but exact paper length grid is not proven from local artifacts | False                     | True               |                        |
| lastfm    |                5 | LOCAL_EXTENSION_GRID    | paper history protocol exists, but exact paper length grid is not proven from local artifacts | False                     | True               |                        |
| lastfm    |               10 | LOCAL_EXTENSION_GRID    | paper history protocol exists, but exact paper length grid is not proven from local artifacts | False                     | True               |                        |
| lastfm    |               20 | LOCAL_EXTENSION_GRID    | paper history protocol exists, but exact paper length grid is not proven from local artifacts | False                     | True               |                        |
