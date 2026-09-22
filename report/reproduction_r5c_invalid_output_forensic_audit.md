# R5C Invalid Output Forensic Audit

## Scope

Offline-only audit of the 15 R5C invalid outputs with `qwen2:7b-instruct`, `ML-1M`, and `OUT_OF_RANGE / unknown_indices`.

No LLM/Ollama calls, raw-output edits, R5C rerun, R5D execution, commit, or push were performed.

## Decision

`R5C_SELECTIVE_RERUN_REQUIRED`

All 15 outputs are technically mappable as candidate item IDs, but post-hoc item-ID to candidate-index normalization is not allowed by the frozen R5C protocol.

## Case Counts

- Cases audited: `15/15`
- Exact offline mappable: `15`
- Partial or ambiguous: `0`
- Genuine invalid: `0`
- Item-ID to index mapping: `DETERMINISTIC`
- Post-hoc normalization scientifically allowed: `NO`
- Selective rerun required: `YES`

## Fifteen Request Audit

| request_id | r5a_request_id | history_length | experimental_condition | validation_failure | parsed_values | mapped_candidate_indices | candidate_item_id_values_count | ambiguous_values_count | request_classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R5C-HIST-0511C5418AE2F5F3 | R5A-REQ-2CE10B903A230C2E | 10 | HISTORY_LENGTH_10 | OUT_OF_RANGE / unknown_indices | [3795,1038,3835,2309,1312] | [3,4,5,6,7] | 5 | 0 | A_EXACT_OFFLINE_MAPPING |
| R5C-HIST-3682BCB95E686347 | R5A-REQ-37B9B3CB405ACCF2 | 10 | HISTORY_LENGTH_10 | OUT_OF_RANGE / unknown_indices | [578,3681,1114,3753,2756] | [1,2,3,4,5] | 5 | 0 | A_EXACT_OFFLINE_MAPPING |
| R5C-HIST-3B3BCB495C88086C | R5A-REQ-764C41D0ED415DFA | 10 | HISTORY_LENGTH_10 | OUT_OF_RANGE / unknown_indices | [369,3902,1657,987,3800] | [1,2,3,4,5] | 5 | 0 | A_EXACT_OFFLINE_MAPPING |
| R5C-HIST-69A6B40E04C7DE5A | R5A-REQ-271A1B2F127AC7AF | 20 | HISTORY_LENGTH_20 | OUT_OF_RANGE / unknown_indices | [3952,1430,1686,2279,3439] | [3,5,6,15,17] | 5 | 0 | A_EXACT_OFFLINE_MAPPING |
| R5C-HIST-D5880B18CED18E3C | R5A-REQ-2CF150F1FB0BB783 | 20 | HISTORY_LENGTH_20 | OUT_OF_RANGE / unknown_indices | [369,1657,987,3800,3457] | [1,3,4,5,15] | 5 | 0 | A_EXACT_OFFLINE_MAPPING |
| R5C-HIST-D5C6D0F95274B3EC | R5A-REQ-EFF606C628FE7B4C | 20 | HISTORY_LENGTH_20 | OUT_OF_RANGE / unknown_indices | [3031,2573,2487,2969,1480] | [1,2,3,4,5] | 5 | 0 | A_EXACT_OFFLINE_MAPPING |
| R5C-HIST-1488249A15B05E91 | R5A-REQ-A8DADC66A9FA87EB | 5 | HISTORY_LENGTH_5 | OUT_OF_RANGE / unknown_indices | [3534,3564,2390,1603,3508] | [3,4,5,7,11] | 5 | 0 | A_EXACT_OFFLINE_MAPPING |
| R5C-HIST-33FE537B137AAA80 | R5A-REQ-1189454A53B65516 | 5 | HISTORY_LENGTH_5 | OUT_OF_RANGE / unknown_indices | [3152,186,502,2002,1784] | [3,2,5,8,20] | 5 | 0 | A_EXACT_OFFLINE_MAPPING |
| R5C-HIST-397F15AD6B87D885 | R5A-REQ-8BC122C7BC25729B | 5 | HISTORY_LENGTH_5 | OUT_OF_RANGE / unknown_indices | [369,1657,987,3800,3457] | [1,3,4,5,15] | 5 | 0 | A_EXACT_OFFLINE_MAPPING |
| R5C-HIST-3A9CDDF1B9807079 | R5A-REQ-F1EE1E2D24C33D6E | 5 | HISTORY_LENGTH_5 | OUT_OF_RANGE / unknown_indices | [1307,1266,2045,377,1809] | [1,5,6,7,15] | 5 | 0 | A_EXACT_OFFLINE_MAPPING |
| R5C-HIST-403121CD5854DF8C | R5A-REQ-4C66C6CC90924DF5 | 5 | HISTORY_LENGTH_5 | OUT_OF_RANGE / unknown_indices | [3795,1135,1038,3835,1659] | [3,2,4,5,11] | 5 | 0 | A_EXACT_OFFLINE_MAPPING |
| R5C-HIST-618C41D1A22C8557 | R5A-REQ-6886136EA7F1A7B8 | 5 | HISTORY_LENGTH_5 | OUT_OF_RANGE / unknown_indices | [3785,3181,3302,649,2481] | [2,3,4,5,6] | 5 | 0 | A_EXACT_OFFLINE_MAPPING |
| R5C-HIST-8AD3F8034BBE875E | R5A-REQ-F3E41E784A5BA1D8 | 5 | HISTORY_LENGTH_5 | OUT_OF_RANGE / unknown_indices | [3152,586,179,2915,1538] | [1,2,3,4,5] | 5 | 0 | A_EXACT_OFFLINE_MAPPING |
| R5C-HIST-8B9BDAD2E93014C4 | R5A-REQ-17DC30027D8CA66F | 5 | HISTORY_LENGTH_5 | OUT_OF_RANGE / unknown_indices | [304,499,2669,2398,778] | [3,1,4,5,7] | 5 | 0 | A_EXACT_OFFLINE_MAPPING |
| R5C-HIST-98DA53A00FCCC5D5 | R5A-REQ-1279A42C159103D7 | 5 | HISTORY_LENGTH_5 | OUT_OF_RANGE / unknown_indices | [578,3681,1114,3753,2756] | [1,2,3,4,5] | 5 | 0 | A_EXACT_OFFLINE_MAPPING |

## Protocol Assessment

| component | source_path | source_exists | allows_item_id_to_index_normalization | finding |
| --- | --- | --- | --- | --- |
| R5C frozen prompt | src\analysis\reproduction_r5c_history_length_recovery.py | true | false | R5C corrected instruction requires exactly five unique one-based candidate indices and explicitly says not to use item IDs. |
| R5C parser/validator | src\llm\canonical_llm_contract.py | true | false | Strict parser interprets ranked_candidate_indices as one-based candidate positions; values outside 1..N are unknown_indices. |
| R5C synthetic tests | report\reproduction_r5c_history_length_recovery.md | true | false | The frozen run validates candidate-set violations as unknown_indices and does not include item-ID rescue in the scientific parser. |
| Original implementation | original_repo\EvaLLMasRecommender\eva\codes\llms.py | true | false | Original LLM calls request a Python list of item titles, not local R5C one-based candidate indices or numeric item IDs. |
| Original output refinement | original_repo\EvaLLMasRecommender\eva\codes\output_refine.py | true | false | Original refinement extracts list outputs and length-adjusts title lists; it is not an R5C index parser and does not authorize numeric item-ID rescue. |
| Original metric mapping | original_repo\EvaLLMasRecommender\eva\codes\llm_metrics.py | true | false | Original metrics map returned titles to item IDs through title matching; this differs from changing R5C's declared index namespace after inference. |
| Prior local ML-1M item-ID contract | report\plan_d_llm_prompt_templates_and_ranking_manifest.md | true | false | Plan D documented a separate item-ID JSON contract for a prior ML-1M diagnostic, but R5C did not use that contract. |
| Protocol decision | configs\reproduction_r5c_history_length_recovery.yaml | true | false | Changing invalid R5C outputs to valid through post-hoc item-ID normalization would modify the frozen output contract and parser semantics. |

## Selective Rerun Rows

| request_id | r5a_request_id | history_length | experimental_condition | user_id | failure_mode | future_rerun_required_check |
| --- | --- | --- | --- | --- | --- | --- |
| R5C-HIST-0511C5418AE2F5F3 | R5A-REQ-2CE10B903A230C2E | 10 | HISTORY_LENGTH_10 | 1086 | qwen_returned_candidate_item_ids_in_ranked_candidate_indices_field | Qwen must return exactly five values in the 1..20 candidate-index namespace. |
| R5C-HIST-3682BCB95E686347 | R5A-REQ-37B9B3CB405ACCF2 | 10 | HISTORY_LENGTH_10 | 1000 | qwen_returned_candidate_item_ids_in_ranked_candidate_indices_field | Qwen must return exactly five values in the 1..20 candidate-index namespace. |
| R5C-HIST-3B3BCB495C88086C | R5A-REQ-764C41D0ED415DFA | 10 | HISTORY_LENGTH_10 | 1070 | qwen_returned_candidate_item_ids_in_ranked_candidate_indices_field | Qwen must return exactly five values in the 1..20 candidate-index namespace. |
| R5C-HIST-69A6B40E04C7DE5A | R5A-REQ-271A1B2F127AC7AF | 20 | HISTORY_LENGTH_20 | 1042 | qwen_returned_candidate_item_ids_in_ranked_candidate_indices_field | Qwen must return exactly five values in the 1..20 candidate-index namespace. |
| R5C-HIST-D5880B18CED18E3C | R5A-REQ-2CF150F1FB0BB783 | 20 | HISTORY_LENGTH_20 | 1070 | qwen_returned_candidate_item_ids_in_ranked_candidate_indices_field | Qwen must return exactly five values in the 1..20 candidate-index namespace. |
| R5C-HIST-D5C6D0F95274B3EC | R5A-REQ-EFF606C628FE7B4C | 20 | HISTORY_LENGTH_20 | 1079 | qwen_returned_candidate_item_ids_in_ranked_candidate_indices_field | Qwen must return exactly five values in the 1..20 candidate-index namespace. |
| R5C-HIST-1488249A15B05E91 | R5A-REQ-A8DADC66A9FA87EB | 5 | HISTORY_LENGTH_5 | 1037 | qwen_returned_candidate_item_ids_in_ranked_candidate_indices_field | Qwen must return exactly five values in the 1..20 candidate-index namespace. |
| R5C-HIST-33FE537B137AAA80 | R5A-REQ-1189454A53B65516 | 5 | HISTORY_LENGTH_5 | 1021 | qwen_returned_candidate_item_ids_in_ranked_candidate_indices_field | Qwen must return exactly five values in the 1..20 candidate-index namespace. |
| R5C-HIST-397F15AD6B87D885 | R5A-REQ-8BC122C7BC25729B | 5 | HISTORY_LENGTH_5 | 1070 | qwen_returned_candidate_item_ids_in_ranked_candidate_indices_field | Qwen must return exactly five values in the 1..20 candidate-index namespace. |
| R5C-HIST-3A9CDDF1B9807079 | R5A-REQ-F1EE1E2D24C33D6E | 5 | HISTORY_LENGTH_5 | 1034 | qwen_returned_candidate_item_ids_in_ranked_candidate_indices_field | Qwen must return exactly five values in the 1..20 candidate-index namespace. |
| R5C-HIST-403121CD5854DF8C | R5A-REQ-4C66C6CC90924DF5 | 5 | HISTORY_LENGTH_5 | 1086 | qwen_returned_candidate_item_ids_in_ranked_candidate_indices_field | Qwen must return exactly five values in the 1..20 candidate-index namespace. |
| R5C-HIST-618C41D1A22C8557 | R5A-REQ-6886136EA7F1A7B8 | 5 | HISTORY_LENGTH_5 | 1057 | qwen_returned_candidate_item_ids_in_ranked_candidate_indices_field | Qwen must return exactly five values in the 1..20 candidate-index namespace. |
| R5C-HIST-8AD3F8034BBE875E | R5A-REQ-F3E41E784A5BA1D8 | 5 | HISTORY_LENGTH_5 | 1013 | qwen_returned_candidate_item_ids_in_ranked_candidate_indices_field | Qwen must return exactly five values in the 1..20 candidate-index namespace. |
| R5C-HIST-8B9BDAD2E93014C4 | R5A-REQ-17DC30027D8CA66F | 5 | HISTORY_LENGTH_5 | 1049 | qwen_returned_candidate_item_ids_in_ranked_candidate_indices_field | Qwen must return exactly five values in the 1..20 candidate-index namespace. |
| R5C-HIST-98DA53A00FCCC5D5 | R5A-REQ-1279A42C159103D7 | 5 | HISTORY_LENGTH_5 | 1000 | qwen_returned_candidate_item_ids_in_ranked_candidate_indices_field | Qwen must return exactly five values in the 1..20 candidate-index namespace. |

## Future Minimal Rerun Protocol

1. Static preflight over only the 15 frozen manifest rows.
2. Synthetic test that rejects item IDs inside `ranked_candidate_indices` and accepts only `1..20` candidate indices.
3. Qwen x ML-1M canary with at least one affected history length cell.
4. Immediate validation that raw outputs contain exactly five 1-based candidate indices, not item IDs.
5. Only then rerun the 15 affected rows with the unchanged frozen R5C protocol identity or a separately named corrected selective-rerun identity.

## Hypothetical Offline Mapping

The existing raw outputs are semantically recoverable in a technical sense. If the mapping were allowed, the 15 rows would map deterministically and R5C would be `2400/2400` valid with `4/4` target gaps all-valid. This is not applied because the protocol disallows the normalization.
