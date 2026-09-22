# G3 Candidate-Position FIRST/RANDOM Protocol Freeze

Decision token: `G3_BASE_LLMRANK_FIRST_RANDOM_PROTOCOL_FROZEN`.

## Scope

G3 freezes BASE and LLMRANK candidate-position/order sensitivity over FIRST versus RANDOM. LLMSRec is excluded because repository evidence classifies it as reconstructed and not a faithful frozen executable strategy.

## Cohort

- Rule: `first_200_users_by_sort_key_from_frozen_G1_primary_candidate_users_after_candidate_contract_validation`.
- Users: `200` per dataset, shared across all models, prompts, and orders.
- Outcome-dependent filtering: `false`.

## Candidate Orders

- FIRST: `positive_at_position_1_then_remaining_negatives_in_frozen_primary_candidate_order`.
- RANDOM: `sha256_seeded_sort_rotate_if_positive_first`.
- RANDOM seed: `R5A_CANDDIF_RANDOM_POSITION_V1`.
- RANDOM key: `seed|dataset|user_id|candidate_item_id|candidate_title`.

## Prompt And Parser

- Prompts: `BASE;LLMRANK`.
- Parser: `llm_rank_strict_index_top10_v1`.
- Output: strict JSON top-10 one-based candidate indices.
- History: `visible_last_10_or_available`.

## Accounting

| accounting_scope | scope_value | logical_requests | new_ollama_calls_required | exact_g1_g2_reuse |
| --- | --- | --- | --- | --- |
| GRAND_TOTAL | ALL | 12000 | 12000 | 0 |

## By Dataset

| scope_value | logical_requests | new_ollama_calls_required | users |
| --- | --- | --- | --- |
| beauty | 2400 | 2400 | 200 |
| sports | 2400 | 2400 | 200 |
| ml1m | 2400 | 2400 | 200 |
| lastfm | 2400 | 2400 | 200 |
| foodcom | 2400 | 2400 | 200 |

## By Model

| scope_value | logical_requests | new_ollama_calls_required |
| --- | --- | --- |
| qwen2:7b-instruct | 4000 | 4000 |
| mistral:7b-instruct | 4000 | 4000 |
| gemma2:9b | 4000 | 4000 |

## By Prompt

| scope_value | logical_requests | new_ollama_calls_required |
| --- | --- | --- |
| BASE | 6000 | 6000 |
| LLMRANK | 6000 | 6000 |

## By Order

| scope_value | logical_requests | new_ollama_calls_required |
| --- | --- | --- |
| FIRST | 6000 | 6000 |
| RANDOM | 6000 | 6000 |

## Statistics

- Primary test: paired-user bootstrap mean delta, FIRST minus RANDOM.
- Bootstrap: 10000 resamples, seed 20260901, two-sided 95 percent percentile CI.
- Multiplicity: Holm-Bonferroni within metric/dataset/model/prompt family, reused from the frozen G2 convention.
- CandDif HR/NDCG: log((1-Acc_random)/(1-Acc_first)) with the R5B epsilon safety policy.

## Validation

- PASS: `25`.
- FAIL: `0`.
