# D0B Final Fifth-Dataset Selection

Decision: `FOODCOM_SELECTED_FINAL_FIFTH_DATASET`

No LLM inference, recommender training, downloads, purge, commit, or Git
history rewrite was performed.

## Authoritative Input Evidence

| dataset     | metric                           | value                                                            | evidence_source           |
|:------------|:---------------------------------|:-----------------------------------------------------------------|:--------------------------|
| Food.com    | archive_sha256                   | e7a43e623c616799f78a0479b550483450fd7ede4e0c40979db3458a82b5b913 | D0A archive manifest      |
| Food.com    | raw_interactions_sha256          | 4e278d2f687b1020f5694e51003a7d6174344b51aebbdf4530de88b5c6a78ee4 | D0A extracted hashes      |
| Food.com    | raw_recipes_sha256               | 6a8136d1da9e03396a9f52d72200b3fddb8fe103f11e1496effca30ac4e0539f | D0A extracted hashes      |
| Food.com    | primary_strict_users             | 226553                                                           | D0A primary feasibility   |
| Food.com    | history_strict_l20_users         | 6387                                                             | D0A history feasibility   |
| Food.com    | usable_name_coverage             | 0.9999956829004002                                               | D0A metadata coverage     |
| Video Games | primary_strict_users             | 2785                                                             | E2B-0 primary feasibility |
| Video Games | primary_user_hash                | 90f4f4694d97d50e72ddd6475b85b911d29cae24a556a0799a2a751f78e3a66b | E2B-0 primary feasibility |
| Video Games | primary_candidate_hash           | c48f48a0353e73c44589965cb48463df4bdd0397355b358d46917d10b812fbad | E2B-0 primary feasibility |
| Video Games | history_strict_l20_users         | 35                                                               | E2B-0 history feasibility |
| Video Games | title_coverage                   | 0.9008620689655172                                               | E2B-0 title provenance    |
| H0          | original_four_history_to_compute | 14400                                                            | H0 no-inference manifest  |

## Completeness Counts

| dataset_candidate   | scope           | family                   |   total |   READY |   READY_WITH_ADAPTATION |   READY_AFTER_TRAINING |   BLOCKED_DATASET |   BLOCKED_GLOBAL_PROTOCOL |   NOT_APPLICABLE |
|:--------------------|:----------------|:-------------------------|--------:|--------:|------------------------:|-----------------------:|------------------:|--------------------------:|-----------------:|
| Food.com            | mandatory_total | ALL                      |      72 |      45 |                      11 |                      8 |                 0 |                         3 |                5 |
| Food.com            | family          | CandDif                  |       3 |       0 |                       3 |                      0 |                 0 |                         0 |                0 |
| Food.com            | family          | beyond utility           |       6 |       3 |                       1 |                      0 |                 0 |                         2 |                0 |
| Food.com            | family          | diversity                |       5 |       3 |                       2 |                      0 |                 0 |                         0 |                0 |
| Food.com            | family          | efficiency               |       5 |       0 |                       0 |                      0 |                 0 |                         0 |                5 |
| Food.com            | family          | fairness                 |       4 |       2 |                       1 |                      0 |                 0 |                         1 |                0 |
| Food.com            | family          | history                  |      18 |      18 |                       0 |                      0 |                 0 |                         0 |                0 |
| Food.com            | family          | primary                  |       6 |       6 |                       0 |                      0 |                 0 |                         0 |                0 |
| Food.com            | family          | profiles                 |       4 |       1 |                       3 |                      0 |                 0 |                         0 |                0 |
| Food.com            | family          | reliability              |       7 |       7 |                       0 |                      0 |                 0 |                         0 |                0 |
| Food.com            | family          | reranking                |       4 |       0 |                       0 |                      4 |                 0 |                         0 |                0 |
| Food.com            | family          | traditional recommenders |       7 |       2 |                       1 |                      4 |                 0 |                         0 |                0 |
| Food.com            | family          | utility                  |       3 |       3 |                       0 |                      0 |                 0 |                         0 |                0 |
| Video Games         | mandatory_total | ALL                      |      72 |      26 |                       5 |                      8 |                25 |                         3 |                5 |
| Video Games         | family          | CandDif                  |       3 |       0 |                       3 |                      0 |                 0 |                         0 |                0 |
| Video Games         | family          | beyond utility           |       6 |       3 |                       1 |                      0 |                 0 |                         2 |                0 |
| Video Games         | family          | diversity                |       5 |       3 |                       0 |                      0 |                 2 |                         0 |                0 |
| Video Games         | family          | efficiency               |       5 |       0 |                       0 |                      0 |                 0 |                         0 |                5 |
| Video Games         | family          | fairness                 |       4 |       2 |                       0 |                      0 |                 1 |                         1 |                0 |
| Video Games         | family          | history                  |      18 |       0 |                       0 |                      0 |                18 |                         0 |                0 |
| Video Games         | family          | primary                  |       6 |       6 |                       0 |                      0 |                 0 |                         0 |                0 |
| Video Games         | family          | profiles                 |       4 |       0 |                       0 |                      0 |                 4 |                         0 |                0 |
| Video Games         | family          | reliability              |       7 |       7 |                       0 |                      0 |                 0 |                         0 |                0 |
| Video Games         | family          | reranking                |       4 |       0 |                       0 |                      4 |                 0 |                         0 |                0 |
| Video Games         | family          | traditional recommenders |       7 |       2 |                       1 |                      4 |                 0 |                         0 |                0 |
| Video Games         | family          | utility                  |       3 |       3 |                       0 |                      0 |                 0 |                         0 |                0 |

## Dataset-Specific Blockers

| dataset_candidate   | requirement_id                           | experiment_family   | metric_or_analysis                         | blocker_scope   | reason                                                                                         |
|:--------------------|:-----------------------------------------|:--------------------|:-------------------------------------------|:----------------|:-----------------------------------------------------------------------------------------------|
| Food.com            | NONE                                     | ALL                 | none                                       | NONE            | no mandatory dataset-specific blockers                                                         |
| Video Games         | HISTORY_qwen2:7b-instruct_L0             | history             | history N=200 BASE qwen2:7b-instruct L0    | DATASET         | Video Games strict L20 users 35, below required 200                                            |
| Video Games         | HISTORY_qwen2:7b-instruct_L1             | history             | history N=200 BASE qwen2:7b-instruct L1    | DATASET         | Video Games strict L20 users 35, below required 200                                            |
| Video Games         | HISTORY_qwen2:7b-instruct_L2             | history             | history N=200 BASE qwen2:7b-instruct L2    | DATASET         | Video Games strict L20 users 35, below required 200                                            |
| Video Games         | HISTORY_qwen2:7b-instruct_L5             | history             | history N=200 BASE qwen2:7b-instruct L5    | DATASET         | Video Games strict L20 users 35, below required 200                                            |
| Video Games         | HISTORY_qwen2:7b-instruct_L10            | history             | history N=200 BASE qwen2:7b-instruct L10   | DATASET         | Video Games strict L20 users 35, below required 200                                            |
| Video Games         | HISTORY_qwen2:7b-instruct_L20            | history             | history N=200 BASE qwen2:7b-instruct L20   | DATASET         | Video Games strict L20 users 35, below required 200                                            |
| Video Games         | HISTORY_mistral:7b-instruct_L0           | history             | history N=200 BASE mistral:7b-instruct L0  | DATASET         | Video Games strict L20 users 35, below required 200                                            |
| Video Games         | HISTORY_mistral:7b-instruct_L1           | history             | history N=200 BASE mistral:7b-instruct L1  | DATASET         | Video Games strict L20 users 35, below required 200                                            |
| Video Games         | HISTORY_mistral:7b-instruct_L2           | history             | history N=200 BASE mistral:7b-instruct L2  | DATASET         | Video Games strict L20 users 35, below required 200                                            |
| Video Games         | HISTORY_mistral:7b-instruct_L5           | history             | history N=200 BASE mistral:7b-instruct L5  | DATASET         | Video Games strict L20 users 35, below required 200                                            |
| Video Games         | HISTORY_mistral:7b-instruct_L10          | history             | history N=200 BASE mistral:7b-instruct L10 | DATASET         | Video Games strict L20 users 35, below required 200                                            |
| Video Games         | HISTORY_mistral:7b-instruct_L20          | history             | history N=200 BASE mistral:7b-instruct L20 | DATASET         | Video Games strict L20 users 35, below required 200                                            |
| Video Games         | HISTORY_gemma2:9b_L0                     | history             | history N=200 BASE gemma2:9b L0            | DATASET         | Video Games strict L20 users 35, below required 200                                            |
| Video Games         | HISTORY_gemma2:9b_L1                     | history             | history N=200 BASE gemma2:9b L1            | DATASET         | Video Games strict L20 users 35, below required 200                                            |
| Video Games         | HISTORY_gemma2:9b_L2                     | history             | history N=200 BASE gemma2:9b L2            | DATASET         | Video Games strict L20 users 35, below required 200                                            |
| Video Games         | HISTORY_gemma2:9b_L5                     | history             | history N=200 BASE gemma2:9b L5            | DATASET         | Video Games strict L20 users 35, below required 200                                            |
| Video Games         | HISTORY_gemma2:9b_L10                    | history             | history N=200 BASE gemma2:9b L10           | DATASET         | Video Games strict L20 users 35, below required 200                                            |
| Video Games         | HISTORY_gemma2:9b_L20                    | history             | history N=200 BASE gemma2:9b L20           | DATASET         | Video Games strict L20 users 35, below required 200                                            |
| Video Games         | SHANNON_CATEGORY_ENTROPY_5               | diversity           | Shannon category entropy@5                 | DATASET         | Video Games lacks frozen dataset metadata/feature evidence for Shannon category entropy@5      |
| Video Games         | ILD_5                                    | diversity           | ILD@5                                      | DATASET         | Video Games lacks frozen dataset metadata/feature evidence for ILD@5                           |
| Video Games         | CALIBRATION_5                            | fairness            | Calibration@5                              | DATASET         | Video Games lacks frozen dataset metadata/feature evidence for Calibration@5                   |
| Video Games         | PROFILE_HISTORY_ONLY                     | profiles            | history-only                               | DATASET         | Video Games paired L20 history blocked and canonical feature/profile representation not frozen |
| Video Games         | PROFILE_PROFILE_GENERATION               | profiles            | profile generation                         | DATASET         | Video Games paired L20 history blocked and canonical feature/profile representation not frozen |
| Video Games         | PROFILE_PROFILE_ONLY_RECOMMENDATION      | profiles            | profile-only recommendation                | DATASET         | Video Games paired L20 history blocked and canonical feature/profile representation not frozen |
| Video Games         | PROFILE_PROFILE___HISTORY_RECOMMENDATION | profiles            | profile + history recommendation           | DATASET         | Video Games paired L20 history blocked and canonical feature/profile representation not frozen |

## Requirement Matrix Summary

| experiment_family        | foodcom_status          | videogames_status       |   requirements |
|:-------------------------|:------------------------|:------------------------|---------------:|
| CandDif                  | READY_WITH_ADAPTATION   | READY_WITH_ADAPTATION   |              3 |
| beyond utility           | BLOCKED_GLOBAL_PROTOCOL | BLOCKED_GLOBAL_PROTOCOL |              2 |
| beyond utility           | READY                   | READY                   |              3 |
| beyond utility           | READY_WITH_ADAPTATION   | READY_WITH_ADAPTATION   |              1 |
| diversity                | READY                   | READY                   |              3 |
| diversity                | READY_WITH_ADAPTATION   | BLOCKED_DATASET         |              2 |
| efficiency               | NOT_APPLICABLE          | NOT_APPLICABLE          |              5 |
| fairness                 | BLOCKED_GLOBAL_PROTOCOL | BLOCKED_GLOBAL_PROTOCOL |              1 |
| fairness                 | READY                   | READY                   |              2 |
| fairness                 | READY_WITH_ADAPTATION   | BLOCKED_DATASET         |              1 |
| history                  | READY                   | BLOCKED_DATASET         |             18 |
| primary                  | READY                   | READY                   |              6 |
| profiles                 | READY                   | BLOCKED_DATASET         |              1 |
| profiles                 | READY_WITH_ADAPTATION   | BLOCKED_DATASET         |              3 |
| reliability              | READY                   | READY                   |              7 |
| reranking                | READY_AFTER_TRAINING    | READY_AFTER_TRAINING    |              4 |
| traditional recommenders | READY                   | READY                   |              2 |
| traditional recommenders | READY_AFTER_TRAINING    | READY_AFTER_TRAINING    |              4 |
| traditional recommenders | READY_WITH_ADAPTATION   | READY_WITH_ADAPTATION   |              1 |
| utility                  | READY                   | READY                   |              3 |

## Future Inference Budget

| component                             |   raw_llm_requests | budget_class         | training_required   |   derived_metric_extra_requests | evidence                                                     |
|:--------------------------------------|-------------------:|:---------------------|:--------------------|--------------------------------:|:-------------------------------------------------------------|
| original_four_history                 |              14400 | MANDATORY            | False               |                               0 | H0 final history matrix; 72 cells                            |
| fifth_dataset_primary                 |               6000 | MANDATORY            | False               |                               0 | 3 models x 2 prompts x 1000 users                            |
| fifth_dataset_history                 |               3600 | MANDATORY            | False               |                               0 | 3 models x 6 lengths x 200 users; feasible only for Food.com |
| gemma_original_four_primary_parity    |              15528 | MANDATORY            | False               |                               0 | derived from E2A gemma_original_four_parity raw rows         |
| required_original_four_gemma_canddif  |                600 | MANDATORY            | False               |                               0 | derived from E2A gemma_canddif_parity raw rows               |
| fifth_dataset_canddif_primary         |               3000 | MANDATORY            | False               |                               0 | 3 models x one order-variant x 1000 users                    |
| other_sensitivities_predeclared       |                  0 | NOT_PREDECLARED      | False               |                               0 | no additional mandatory sensitivity budget is frozen in D0B  |
| profile_generation                    |                600 | REQUIRED_CONDITIONAL | False               |                               0 | 3 models x 200 paired users                                  |
| profile_only_recommendation           |                600 | REQUIRED_CONDITIONAL | False               |                               0 | 3 models x 200 paired users                                  |
| profile_plus_history_recommendation   |                600 | REQUIRED_CONDITIONAL | False               |                               0 | 3 models x 200 paired users                                  |
| derived_metrics                       |                  0 | DERIVED              | False               |                               0 | derived metrics add zero raw LLM requests                    |
| grand_total_mandatory_and_conditional |              44928 | TOTAL                | False               |                               0 | excludes non-predeclared optional sensitivities              |

## Loser Purge Inventory

Loser: `Video Games`. Entries: `69`. No purge executed.
