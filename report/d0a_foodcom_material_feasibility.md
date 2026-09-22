# D0A Food.com Material Raw Intake and Final-Protocol Feasibility

Decision: `FOODCOM_D0A_FEASIBLE_WITH_ADAPTATIONS`

No LLM inference, recommender training, remote API calls, network downloads,
purges, commits, competing-dataset modifications, or `original_repo/`
modifications were performed.

## Raw Archive

- File size: `280350373` bytes
- SHA256: `e7a43e623c616799f78a0479b550483450fd7ede4e0c40979db3458a82b5b913`

## Archive Contents

| member_path                 | basename                    |      size |   compressed_size | required_for_d0a   | recognized_foodcom_file   | notes   |
|:----------------------------|:----------------------------|----------:|------------------:|:-------------------|:--------------------------|:--------|
| PP_recipes.csv              | PP_recipes.csv              | 204912912 |          48484170 | False              | True                      |         |
| PP_users.csv                | PP_users.csv                |  13555204 |           3434261 | False              | True                      |         |
| RAW_interactions.csv        | RAW_interactions.csv        | 349436524 |         131288410 | True               | True                      |         |
| RAW_recipes.csv             | RAW_recipes.csv             | 294520189 |          87292002 | True               | True                      |         |
| ingr_map.pkl                | ingr_map.pkl                |    918790 |            226160 | False              | False                     |         |
| interactions_test.csv       | interactions_test.csv       |    522731 |            212819 | False              | True                      |         |
| interactions_train.csv      | interactions_train.csv      |  28248536 |           9291026 | False              | True                      |         |
| interactions_validation.csv | interactions_validation.csv |    292316 |            120289 | False              | True                      |         |

## Schema

| source_file          |   column_position | column         | dtype   |   null_count |   empty_string_count | required_by_d0a   | schema_missing_required_columns   | schema_extra_columns   | status   |
|:---------------------|------------------:|:---------------|:--------|-------------:|---------------------:|:------------------|:----------------------------------|:-----------------------|:---------|
| RAW_interactions.csv |                 1 | user_id        | string  |            0 |                    0 | True              |                                   |                        | PRESENT  |
| RAW_interactions.csv |                 2 | recipe_id      | string  |            0 |                    0 | True              |                                   |                        | PRESENT  |
| RAW_interactions.csv |                 3 | date           | string  |            0 |                    0 | True              |                                   |                        | PRESENT  |
| RAW_interactions.csv |                 4 | rating         | string  |            0 |                    0 | True              |                                   |                        | PRESENT  |
| RAW_interactions.csv |                 5 | review         | string  |            0 |                  165 | True              |                                   |                        | PRESENT  |
| RAW_recipes.csv      |                 1 | name           | object  |            0 |                    1 | True              |                                   |                        | PRESENT  |
| RAW_recipes.csv      |                 2 | id             | string  |            0 |                    0 | True              |                                   |                        | PRESENT  |
| RAW_recipes.csv      |                 3 | minutes        | int64   |            0 |                    0 | True              |                                   |                        | PRESENT  |
| RAW_recipes.csv      |                 4 | contributor_id | int64   |            0 |                    0 | True              |                                   |                        | PRESENT  |
| RAW_recipes.csv      |                 5 | submitted      | object  |            0 |                    0 | True              |                                   |                        | PRESENT  |
| RAW_recipes.csv      |                 6 | tags           | object  |            0 |                    0 | True              |                                   |                        | PRESENT  |
| RAW_recipes.csv      |                 7 | nutrition      | object  |            0 |                    0 | True              |                                   |                        | PRESENT  |
| RAW_recipes.csv      |                 8 | n_steps        | int64   |            0 |                    0 | True              |                                   |                        | PRESENT  |
| RAW_recipes.csv      |                 9 | steps          | object  |            0 |                    0 | True              |                                   |                        | PRESENT  |
| RAW_recipes.csv      |                10 | description    | object  |            0 |                 4979 | True              |                                   |                        | PRESENT  |
| RAW_recipes.csv      |                11 | ingredients    | object  |            0 |                    0 | True              |                                   |                        | PRESENT  |
| RAW_recipes.csv      |                12 | n_ingredients  | int64   |            0 |                    0 | True              |                                   |                        | PRESENT  |

## Raw Statistics

| metric                                    | value      |
|:------------------------------------------|:-----------|
| interaction_rows                          | 1132367    |
| unique_users                              | 226570     |
| unique_recipes_in_raw_recipes             | 231637     |
| unique_recipes_in_interactions            | 231637     |
| date_min                                  | 2000-01-25 |
| date_max                                  | 2018-12-20 |
| invalid_dates                             | 0          |
| exact_duplicate_rows                      | 0          |
| repeated_user_recipe_pairs                | 0          |
| repeated_user_recipe_rows                 | 0          |
| recipe_ids_referenced_absent_from_recipes | 0          |
| recipes_never_referenced_by_interactions  | 0          |
| users_with_interactions_ge_1              | 226570     |
| users_with_interactions_ge_2              | 60314      |
| users_with_interactions_ge_5              | 23086      |
| users_with_interactions_ge_10             | 12486      |
| users_with_interactions_ge_20             | 6687       |
| users_with_interactions_ge_21             | 6389       |
| users_with_interactions_ge_50             | 2755       |
| users_with_interactions_ge_100            | 1382       |
| interactions_per_user_q0                  | 1.0        |
| interactions_per_user_q0.25               | 1.0        |
| interactions_per_user_q0.5                | 1.0        |
| interactions_per_user_q0.75               | 2.0        |
| interactions_per_user_q0.9                | 5.0        |
| interactions_per_user_q0.95               | 11.0       |
| interactions_per_user_q0.99               | 61.0       |
| interactions_per_user_q1                  | 7671.0     |

## Final Split

| metric                          | value                                    |
|:--------------------------------|:-----------------------------------------|
| split_policy                    | chronological_leave_one_target_per_user  |
| tie_policy                      | date_ascending_then_original_input_order |
| duplicate_user_recipe_policy    | retain_repeated_event_rows               |
| users                           | 226570                                   |
| items                           | 231637                                   |
| interactions                    | 1132367                                  |
| train_rows                      | 905797                                   |
| test_rows                       | 226570                                   |
| users_with_train_history_ge_20  | 6389                                     |
| users_with_empty_train_history  | 166256                                   |
| max_train_history_length        | 7670                                     |
| median_train_history_length     | 0.0                                      |
| max_total_interactions_per_user | 7671                                     |

## Timestamp and Duplicate Audit

| metric                               |   value | policy                       | notes                                               |
|:-------------------------------------|--------:|:-----------------------------|:----------------------------------------------------|
| invalid_date_rows                    |       0 | BLOCK_IF_NONZERO             | final chronological split needs parseable dates     |
| exact_duplicate_rows                 |       0 | RETAIN_AND_REPORT            | D0A does not collapse raw event rows                |
| repeated_user_recipe_pairs           |       0 | RETAIN_EVENT_SEMANTICS       | same user can interact with a recipe more than once |
| repeated_user_recipe_rows            |       0 | RETAIN_EVENT_SEMANTICS       | affected rows in repeated user-recipe pairs         |
| same_day_user_tie_groups             |  161062 | STABLE_INPUT_ORDER_TIE_BREAK | date granularity is one day                         |
| same_day_user_tie_rows               |  451467 | STABLE_INPUT_ORDER_TIE_BREAK | rows in user-date tie groups                        |
| same_day_same_user_recipe_tie_groups |       0 | STABLE_INPUT_ORDER_TIE_BREAK | same recipe repeated on same date                   |
| same_day_same_user_recipe_tie_rows   |       0 | STABLE_INPUT_ORDER_TIE_BREAK | rows in same user-recipe-date tie groups            |

## Name and Metadata Coverage

| field       |   interaction_item_universe |   missing_recipe_row | null_values   |   empty_values | whitespace_only_values   |   usable_non_empty |   coverage | notes                                                                    |
|:------------|----------------------------:|---------------------:|:--------------|---------------:|:-------------------------|-------------------:|-----------:|:-------------------------------------------------------------------------|
| name        |                      231637 |                    0 | 0             |              1 | 0                        |             231636 |   0.999996 | recipe names are not filled from descriptions, tags, ingredients, or IDs |
| tags        |                      231637 |                    0 |               |              0 |                          |             231637 |   1        | coverage over recipes referenced by RAW_interactions                     |
| ingredients |                      231637 |                    0 |               |              0 |                          |             231637 |   1        | coverage over recipes referenced by RAW_interactions                     |
| description |                      231637 |                    0 |               |           4983 |                          |             226654 |   0.978488 | coverage over recipes referenced by RAW_interactions                     |
| nutrition   |                      231637 |                    0 |               |              0 |                          |             231637 |   1        | coverage over recipes referenced by RAW_interactions                     |
| steps       |                      231637 |                    0 |               |              0 |                          |             231637 |   1        | coverage over recipes referenced by RAW_interactions                     |

## Feature Coverage

| feature_source   |   interaction_catalog_items |   parseable_items |   nonempty_items |   nonempty_coverage |   unique_tokens |   mean_tokens_per_item |   median_tokens_per_item | top_tokens                                                                                                                                                                                                                                                                                      | canonical_feature_space_status   |
|:-----------------|----------------------------:|------------------:|-----------------:|--------------------:|----------------:|-----------------------:|-------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------|
| tags             |                      231637 |            231637 |           231528 |            0.999529 |             551 |               17.8796  |                       17 | ["preparation","time-to-make","course","main-ingredient","dietary","easy","occasion","cuisine","low-in-something","main-dish","equipment","60-minutes-or-less","number-of-servings","meat","30-minutes-or-less","vegetables","taste-mood","4-hours-or-less","north-american","3-steps-or-less"] | READY_WITH_ADAPTATION            |
| ingredients      |                      231637 |            231637 |           231637 |            1        |           14942 |                9.05115 |                        9 | ["salt","butter","sugar","onion","water","eggs","olive oil","flour","milk","garlic cloves","pepper","brown sugar","garlic","all-purpose flour","baking powder","egg","salt and pepper","parmesan cheese","lemon juice","baking soda"]                                                           | READY_WITH_ADAPTATION            |

## Primary 1000 Feasibility

| dataset   |   structurally_candidate_valid_users |   positive_name_valid_users |   candidate_all_names_valid_users |   final_strict_name_eligible_users |   selected_users |   candidate_generation_seed | candidate_generation_independent_of_name_availability   | status                        | blocker   | selected_user_hash                                               | selected_candidate_hash                                          |
|:----------|-------------------------------------:|----------------------------:|----------------------------------:|-----------------------------------:|-----------------:|----------------------------:|:--------------------------------------------------------|:------------------------------|:----------|:-----------------------------------------------------------------|:-----------------------------------------------------------------|
| foodcom   |                               226570 |                      226569 |                            226553 |                             226553 |             1000 |                    20260824 | True                                                    | FOODCOM_PRIMARY_1000_FEASIBLE |           | 53c6bd4969102154776f6a67489fc402fa2af9608f77b0ea5512fcca0bce9717 | f946195b02da768691c417ea31bf42893b06c31fc0e1ea5488247263c523347b |

## History 200 Feasibility

| dataset   |   users_with_train_history_ge_20 |   structurally_candidate_valid_long_history_users |   strict_l1_users |   strict_l2_users |   strict_l5_users |   strict_l10_users |   strict_l20_users |   strict_paired_l20_users |   selected_users | history_lengths   | l0_semantics           | same_candidates_all_l   | same_positive_all_l   | status                       | blocker   | selected_user_hash                                               | selected_candidate_hash                                          |
|:----------|---------------------------------:|--------------------------------------------------:|------------------:|------------------:|------------------:|-------------------:|-------------------:|--------------------------:|-----------------:|:------------------|:-----------------------|:------------------------|:----------------------|:-----------------------------|:----------|:-----------------------------------------------------------------|:-----------------------------------------------------------------|
| foodcom   |                             6389 |                                              6389 |              6387 |              6387 |              6387 |               6387 |               6387 |                      6387 |              200 | 0;1;2;5;10;20     | explicit_empty_history | True                    | True                  | FOODCOM_HISTORY_200_FEASIBLE |           | 7fc1107317dd0c9d60c22f641092127d185203227e479d088f1b6b0bdb5af5a0 | 9097e40ef4ea2fad8d4f22635ddf78f35b4343a28d2c49fafb3a90cfe8eeda1d |

## Metric Capability

| metric                      | family                 | status                  | evidence                                                                   |
|:----------------------------|:-----------------------|:------------------------|:---------------------------------------------------------------------------|
| HR@5                        | utility                | READY_DATA              | held-out target and 20-candidate ranking contract are materialized         |
| NDCG@5                      | utility                | READY_DATA              | held-out target and candidate order are materialized                       |
| MRR diagnostic              | utility                | READY_DATA              | ranked candidate outputs will be sufficient                                |
| SelfInformation@5           | novelty / popularity   | READY_DATA              | train item popularity is available after chronological split               |
| Serendipity@5               | novelty / popularity   | READY_WITH_ADAPTATION   | requires predeclared recipe-feature or popularity baseline definition      |
| ARP@5                       | novelty / popularity   | READY_DATA              | train item popularity is available                                         |
| APLT@5                      | novelty / popularity   | READY_DATA              | long-tail threshold can be computed from train popularity                  |
| PopREO@5                    | novelty / popularity   | BLOCKED_GLOBAL_PROTOCOL | grouping policy is not recovered globally; not a Food.com metadata blocker |
| ItemCoverage@5              | diversity / coverage   | READY_DATA              | candidate and catalog item IDs are materialized                            |
| Catalog Coverage@5          | diversity / coverage   | READY_DATA              | recipe catalog is available                                                |
| OIC@5                       | diversity / coverage   | READY_DATA              | item coverage over observed recommendation outputs can be computed later   |
| Shannon category entropy@5  | diversity / coverage   | READY_WITH_ADAPTATION   | tags provide the canonical category feature space                          |
| ILD@5                       | diversity / coverage   | READY_WITH_ADAPTATION   | requires a declared Jaccard/cosine distance over tags or ingredients       |
| Calibration@5               | calibration / fairness | READY_WITH_ADAPTATION   | requires user-profile feature distribution over tags or ingredients        |
| Gini@5                      | calibration / fairness | READY_DATA              | exposure distribution over recommended item IDs is sufficient              |
| DPD@5                       | calibration / fairness | BLOCKED_GLOBAL_PROTOCOL | requires a global group definition before dataset-specific execution       |
| Jain's Index@5              | calibration / fairness | READY_DATA              | exposure distribution over recommended item IDs is sufficient              |
| hallucination               | LLM behaviour          | READY_DATA              | strict recipe-name candidate catalog is available                          |
| strict parse validity       | LLM behaviour          | READY_DATA              | parser contract llm_rank_strict_index_top10_v1 is global                   |
| duplicate rate              | LLM behaviour          | READY_DATA              | ranked candidate outputs can be checked for duplicates                     |
| invalid reason distribution | LLM behaviour          | READY_DATA              | strict parser emits machine-readable invalid reasons                       |
| latency                     | efficiency             | NOT_APPLICABLE          | execution telemetry, not raw dataset material capability                   |
| throughput                  | efficiency             | NOT_APPLICABLE          | execution telemetry, not raw dataset material capability                   |
| runtime distributions       | efficiency             | NOT_APPLICABLE          | execution telemetry, not raw dataset material capability                   |
| tokens/cost where available | efficiency             | NOT_APPLICABLE          | provider telemetry, not raw dataset material capability                    |

## Paper-Dimension Capability

| dimension                                      | status                  | evidence                                                        |
|:-----------------------------------------------|:------------------------|:----------------------------------------------------------------|
| BASE                                           | READY                   | primary 1000 strict-name candidate gate                         |
| LLMRank                                        | READY                   | same candidate contract with LLMRank prompt family              |
| History L0/1/2/5/10/20                         | READY                   | paired 200 strict L20 gate                                      |
| CandDif / candidate-position bias              | READY_WITH_ADAPTATION   | requires future order perturbation cells over frozen candidates |
| hallucination                                  | READY                   | candidate recipe names support strict candidate-set matching    |
| profile generation                             | READY_WITH_ADAPTATION   | recipe history can be rendered from names/tags/ingredients      |
| profile-only recommendation                    | READY_WITH_ADAPTATION   | profile prompt adaptation required                              |
| profile + history recommendation               | READY_WITH_ADAPTATION   | profile prompt plus visible history adaptation required         |
| ranking                                        | READY                   | one held-out target and 20-candidate sets                       |
| reranking                                      | READY                   | candidate order is frozen                                       |
| candidate-order sensitivity                    | READY_WITH_ADAPTATION   | requires future deterministic order variants                    |
| candidate-pool sensitivity                     | READY_WITH_ADAPTATION   | requires future deterministic pool-size variants                |
| K-S representativeness if globally recoverable | BLOCKED_GLOBAL_PROTOCOL | global K-S representativeness protocol is not recovered here    |

## Traditional Model Capability

| model     | required_inputs              | available_inputs                              | adaptation_required                                         | status                | blocker   |
|:----------|:-----------------------------|:----------------------------------------------|:------------------------------------------------------------|:----------------------|:----------|
| Random    | candidate sets               | 20-candidate sets                             | none                                                        | READY_DATA            |           |
| MostPop   | train interactions           | chronological train rows                      | none                                                        | READY_DATA            |           |
| BM25-like | item text metadata           | name, description, tags, ingredients          | recipe text field concatenation                             | READY_WITH_ADAPTATION |           |
| BPRMF     | user-item train interactions | chronological train rows with IDs             | map raw IDs to contiguous internals in future preprocessing | READY_DATA            |           |
| GRU4Rec   | ordered user sequences       | date plus input-order chronological sequences | day-level tie policy documented                             | READY_WITH_ADAPTATION |           |
| SASRec    | ordered user sequences       | date plus input-order chronological sequences | day-level tie policy documented                             | READY_WITH_ADAPTATION |           |
| LightGCN  | user-item train graph        | chronological train rows with IDs             | map raw IDs to contiguous internals in future preprocessing | READY_DATA            |           |
