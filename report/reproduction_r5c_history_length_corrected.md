# R5C_CORRECTED History-Length Recovery

## Scope

R5C_CORRECTED executes only frozen R5A `HISTORY_LENGTH_RECOVERY` ranking rows. CandDif, Generation-involved, profile-generation, downloads, training, remote APIs, R5D, and `original_repo/` edits are out of scope.

## R5B corrected freeze dependency

- R5B corrected freeze commit: `83b46454866ea61f3e05eec81d555be53e4c41e8`
- R5B corrected protocol hash: `9596F36E1B1DBC15B68679DDF32114E6AE09914B418FB31DC112072384B708D5`
- R5B original namespace preserved: `results/recovery/r5/`
- R5B corrected namespace preserved: `results/recovery/r5b_corrected/`

## Frozen manifest verification

Manifest SHA256: `290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD`.

Selected R5C rows: `2400`.

## Static preflight

| check_id | check_name | status | observed | expected |
| --- | --- | --- | --- | --- |
| STATIC-001 | manifest SHA | pass | 290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD | 290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD |
| STATIC-002 | repository HEAD is R5B corrected freeze | pass | 83b46454866ea61f3e05eec81d555be53e4c41e8 | 83b46454866ea61f3e05eec81d555be53e4c41e8 |
| STATIC-003 | R5B corrected freeze manifest exists | pass | C:/Users/Bartolo/Desktop/PROGETTI/IRP_BeyondUtility_Project/results/manifests/reproduction_r5b_corrected_freeze.json | exists |
| STATIC-004 | R5B corrected protocol hash | pass | 9596F36E1B1DBC15B68679DDF32114E6AE09914B418FB31DC112072384B708D5 | 9596F36E1B1DBC15B68679DDF32114E6AE09914B418FB31DC112072384B708D5 |
| STATIC-005 | manifest rows | pass | 4800 | 4800 |
| STATIC-006 | selected history rows | pass | 2400 | 2400 |
| STATIC-007 | out-of-scope rows remain unselected | pass | 2400 | 2400 |
| STATIC-008 | no CandDif rows selected | pass | 0 | 0 |
| STATIC-009 | no Generation rows selected | pass | 0 | 0 |
| STATIC-010 | ranking task only | pass | ['ranking'] | ranking |
| STATIC-011 | datasets | pass | ['Beauty', 'LastFM', 'ML-1M', 'Sports'] | ['Beauty', 'LastFM', 'ML-1M', 'Sports'] |
| STATIC-012 | models | pass | ['mistral:7b-instruct', 'qwen2:7b-instruct'] | ['mistral:7b-instruct', 'qwen2:7b-instruct'] |
| STATIC-013 | expected cell count | pass | 24 cells; min=100; max=100 | 24 cells of 100 |
| STATIC-014 | expected group count | pass | 800 groups; min=3; max=3 | 800 groups of 3 |
| STATIC-015 | conditions by dataset | pass | {'Beauty': ['HISTORY_LENGTH_0', 'HISTORY_LENGTH_10', 'HISTORY_LENGTH_20'], 'Sports': ['HISTORY_LENGTH_10', 'HISTORY_LENGTH_20', 'HISTORY_LENGTH_5'], 'ML-1M': ['HISTORY_LENGTH_10', 'HISTORY_LENGTH_20', 'HISTORY_LENGTH_5'], 'LastFM': ['HISTORY_LENGTH_10', 'HISTORY_LENGTH_20', 'HISTORY_LENGTH_5']} | {'Beauty': ['HISTORY_LENGTH_0', 'HISTORY_LENGTH_10', 'HISTORY_LENGTH_20'], 'Sports': ['HISTORY_LENGTH_10', 'HISTORY_LENGTH_20', 'HISTORY_LENGTH_5'], 'ML-1M': ['HISTORY_LENGTH_10', 'HISTORY_LENGTH_20', 'HISTORY_LENGTH_5'], 'LastFM': ['HISTORY_LENGTH_10', 'HISTORY_LENGTH_20', 'HISTORY_LENGTH_5']} |
| STATIC-016 | expected K is 5 | pass | [5] | [5] |
| STATIC-017 | candidate count 20 | pass | ['20'] | 20 |
| STATIC-018 | group validation | pass | 800/800 | 800/800 |
| STATIC-019 | corrected prompt says EXACTLY 5 | pass | 2400 | 2400 |
| STATIC-020 | corrected prompt states positions 1..N | pass | 2400 | 2400 |
| STATIC-021 | corrected prompt rejects item IDs | pass | checked | all prompts |
| STATIC-022 | parser contract unchanged strict index parser | pass | ['r5c_ranked_index_json_top5_v1'] | r5c_ranked_index_json_top5_v1 |
| STATIC-023 | local endpoint only | pass | http://127.0.0.1:11434/api/generate | localhost Ollama |
| STATIC-024 | format json stream false | pass | format=json;stream=False | format=json;stream=false |
| STATIC-025 | max_retries remains zero | pass | 0 | 0 |
| STATIC-026 | output namespace R5C corrected | pass | results/recovery/r5c_history_length_corrected/ | results/recovery/r5c_history_length_corrected/ |
| STATIC-027 | no output path overlap with original R5A destinations | pass | 0 | 0 |
| STATIC-028 | no output path overlap with original R5C namespace | pass | checked | none |
| STATIC-029 | original R5C raw preserved | pass | 2400 | 2400 |
| STATIC-030 | original repo clean | pass |  | empty |
| STATIC-031 | protected artifacts unchanged | pass | 12/12 | 12/12 |
| STATIC-032 | disk free guard | pass | 159162011648 | >=1073741824 |
| STATIC-033 | context envelope | pass | 32/32 | all pass |
| STATIC-034 | protocol hash present and new | pass | 3AAD4945AF443A7F50846BFD044C54FEFD5B77EDBD3725B1D1F1C62D829ECDEE | 64-char SHA256 different from original R5C |

## Synthetic tests

| case_id | expected_valid | observed_valid | expected_status | observed_status | pass |
| --- | --- | --- | --- | --- | --- |
| valid_json_object_indices | true | true | parsed_valid | parsed_valid | true |
| valid_json_multiline | true | true | parsed_valid | parsed_valid | true |
| k_minus_1 | false | false | invalid_length | invalid_length | true |
| k_plus_1 | false | false | invalid_length | invalid_length | true |
| numbered_ranking | false | false | parse_failed | parse_failed | true |
| plain_json_array_candidate_ids | false | false | invalid_schema | invalid_schema | true |
| leading_prose | false | false | parse_failed | parse_failed | true |
| trailing_prose | false | false | parse_failed | parse_failed | true |
| duplicate_index | false | false | duplicate_indices | duplicate_indices | true |
| candidate_set_violation | false | false | unknown_indices | unknown_indices | true |
| truncated_output | false | false | parse_failed | parse_failed | true |
| empty_history_row_valid | true | true | parsed_valid | parsed_valid | true |
| backend_incomplete_response | false | false | backend_incomplete | backend_incomplete | true |
| ml1m_item_ids_in_indices_field | false | false | unknown_indices | unknown_indices | true |

## Context envelope

| scope | model_name | dataset | experimental_condition | history_length | approx_input_tokens | output_budget_tokens | context_length | margin_tokens | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_dataset_condition | mistral:7b-instruct | Beauty | HISTORY_LENGTH_0 | 0 | 818 | 512 | 32768 | 31438 | pass |
| model_dataset_condition | mistral:7b-instruct | Beauty | HISTORY_LENGTH_10 | 10 | 1091 | 512 | 32768 | 31165 | pass |
| model_dataset_condition | mistral:7b-instruct | Beauty | HISTORY_LENGTH_20 | 20 | 1091 | 512 | 32768 | 31165 | pass |
| model_dataset_condition | mistral:7b-instruct | LastFM | HISTORY_LENGTH_10 | 10 | 370 | 512 | 32768 | 31886 | pass |
| model_dataset_condition | mistral:7b-instruct | LastFM | HISTORY_LENGTH_20 | 20 | 430 | 512 | 32768 | 31826 | pass |
| model_dataset_condition | mistral:7b-instruct | LastFM | HISTORY_LENGTH_5 | 5 | 337 | 512 | 32768 | 31919 | pass |
| model_dataset_condition | mistral:7b-instruct | ML-1M | HISTORY_LENGTH_10 | 10 | 464 | 512 | 32768 | 31792 | pass |
| model_dataset_condition | mistral:7b-instruct | ML-1M | HISTORY_LENGTH_20 | 20 | 560 | 512 | 32768 | 31696 | pass |
| model_dataset_condition | mistral:7b-instruct | ML-1M | HISTORY_LENGTH_5 | 5 | 422 | 512 | 32768 | 31834 | pass |
| model_dataset_condition | mistral:7b-instruct | Sports | HISTORY_LENGTH_10 | 10 | 866 | 512 | 32768 | 31390 | pass |
| model_dataset_condition | mistral:7b-instruct | Sports | HISTORY_LENGTH_20 | 20 | 938 | 512 | 32768 | 31318 | pass |
| model_dataset_condition | mistral:7b-instruct | Sports | HISTORY_LENGTH_5 | 5 | 750 | 512 | 32768 | 31506 | pass |
| model_dataset_condition | qwen2:7b-instruct | Beauty | HISTORY_LENGTH_0 | 0 | 818 | 512 | 32768 | 31438 | pass |
| model_dataset_condition | qwen2:7b-instruct | Beauty | HISTORY_LENGTH_10 | 10 | 1091 | 512 | 32768 | 31165 | pass |
| model_dataset_condition | qwen2:7b-instruct | Beauty | HISTORY_LENGTH_20 | 20 | 1091 | 512 | 32768 | 31165 | pass |
| model_dataset_condition | qwen2:7b-instruct | LastFM | HISTORY_LENGTH_10 | 10 | 370 | 512 | 32768 | 31886 | pass |
| model_dataset_condition | qwen2:7b-instruct | LastFM | HISTORY_LENGTH_20 | 20 | 430 | 512 | 32768 | 31826 | pass |
| model_dataset_condition | qwen2:7b-instruct | LastFM | HISTORY_LENGTH_5 | 5 | 337 | 512 | 32768 | 31919 | pass |
| model_dataset_condition | qwen2:7b-instruct | ML-1M | HISTORY_LENGTH_10 | 10 | 464 | 512 | 32768 | 31792 | pass |
| model_dataset_condition | qwen2:7b-instruct | ML-1M | HISTORY_LENGTH_20 | 20 | 560 | 512 | 32768 | 31696 | pass |
| model_dataset_condition | qwen2:7b-instruct | ML-1M | HISTORY_LENGTH_5 | 5 | 422 | 512 | 32768 | 31834 | pass |
| model_dataset_condition | qwen2:7b-instruct | Sports | HISTORY_LENGTH_10 | 10 | 866 | 512 | 32768 | 31390 | pass |
| model_dataset_condition | qwen2:7b-instruct | Sports | HISTORY_LENGTH_20 | 20 | 938 | 512 | 32768 | 31318 | pass |
| model_dataset_condition | qwen2:7b-instruct | Sports | HISTORY_LENGTH_5 | 5 | 750 | 512 | 32768 | 31506 | pass |
| long_vs_short | mistral:7b-instruct | Beauty | HISTORY_LENGTH_0->HISTORY_LENGTH_20 | 0->20 | 1091 | 512 | 32768 | 31165 | pass |
| long_vs_short | mistral:7b-instruct | LastFM | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 5->20 | 430 | 512 | 32768 | 31826 | pass |
| long_vs_short | mistral:7b-instruct | ML-1M | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 5->20 | 560 | 512 | 32768 | 31696 | pass |
| long_vs_short | mistral:7b-instruct | Sports | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 5->20 | 938 | 512 | 32768 | 31318 | pass |
| long_vs_short | qwen2:7b-instruct | Beauty | HISTORY_LENGTH_0->HISTORY_LENGTH_20 | 0->20 | 1091 | 512 | 32768 | 31165 | pass |
| long_vs_short | qwen2:7b-instruct | LastFM | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 5->20 | 430 | 512 | 32768 | 31826 | pass |
| long_vs_short | qwen2:7b-instruct | ML-1M | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 5->20 | 560 | 512 | 32768 | 31696 | pass |
| long_vs_short | qwen2:7b-instruct | Sports | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 5->20 | 938 | 512 | 32768 | 31318 | pass |

## Protocol freeze

```text
R5C_PROTOCOL_PREFLIGHT_READY
PROTOCOL_HASH=3AAD4945AF443A7F50846BFD044C54FEFD5B77EDBD3725B1D1F1C62D829ECDEE
```

| component | value | sha256 |
| --- | --- | --- |
| manifest | 290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD | 290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD |
| prompt_family | r5c_corrected_history_length_json_index_exact5_v2 | 94FD1210B496EBEAA8CC06705D8F9E6E1B88A8F42A00962FB35BFD6F8247B6D3 |
| parser | r5c_ranked_index_json_top5_v1 | CB5DA25F85CC9B90449FC11EA436D26379C225F2F2194CDB3CF9B2306A92A364 |
| runner | C:/Users/Bartolo/Desktop/PROGETTI/IRP_BeyondUtility_Project/src/analysis/reproduction_r5c_history_length_corrected.py | E5A2D904ADA07C94E8C56698DAED8985FAF65195B497A0A2F9FF6E708FDA098B |
| config | configs/reproduction_r5c_history_length_corrected.yaml | 271BDC3F6E5CE9C1794C0F58F7BC133F3507FD52443A4B28EAE057F5B86FE59D |
| history_candidate_construction | R5A frozen history-length rows | 38E48348F29AE2E429EC30D883C9264F2D4EA72DABA65C9B55F755AEA7C32701 |
| group_validation | same candidates and positive item across history lengths | E6CE60617C88112E8BAA30811B2F5A9C70AED2270217CA7D0AA6D7E1DF3FD083 |
| metrics | reuse R5C HR@5/NDCG@5 implementation | 77E4871D862D2997242A61E84EC8D0C2CC311468906CBE30323D40A160134FFC |
| generation_config | {"endpoint":"http://127.0.0.1:11434/api/generate","format":"json","max_retries":"0","num_predict_values":["512"],"seed_values":["0"],"stream":"false","temperature":"0","timeout_seconds":"120"} | E62BD8A4FC52A56CBB9CCBC5AE8FA24861835197175629B091FFE188E5964801 |
| protocol | R5C_CORRECTED_PROTOCOL_PREFLIGHT_READY | 3AAD4945AF443A7F50846BFD044C54FEFD5B77EDBD3725B1D1F1C62D829ECDEE |
| model_identifier | mistral:7b-instruct | 6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf |
| model_identifier | qwen2:7b-instruct | dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755 |

## Canary results

| Model | Dataset | Condition | HistoryLength | Complete | Parsed K | Valid | Failure | request_id | r5a_request_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mistral:7b-instruct | Beauty | HISTORY_LENGTH_0 | 0 | true | 5 | true | VALID | R5C-CORR-16D676B05ABC07CE | R5A-REQ-A047C418739887F3 |
| mistral:7b-instruct | Beauty | HISTORY_LENGTH_10 | 10 | true | 5 | true | VALID | R5C-CORR-B2094B16F72783D2 | R5A-REQ-4B59FA646FA5E44B |
| mistral:7b-instruct | Beauty | HISTORY_LENGTH_20 | 20 | true | 5 | true | VALID | R5C-CORR-748206131D610439 | R5A-REQ-91B45BCD83A063BE |
| qwen2:7b-instruct | Beauty | HISTORY_LENGTH_0 | 0 | true | 5 | true | VALID | R5C-CORR-48031A3E5EE3FC3C | R5A-REQ-964FFF65E0CDD226 |
| qwen2:7b-instruct | Beauty | HISTORY_LENGTH_10 | 10 | true | 5 | true | VALID | R5C-CORR-4E24C197EB1842D2 | R5A-REQ-F9B0A51E75F1A054 |
| qwen2:7b-instruct | Beauty | HISTORY_LENGTH_20 | 20 | true | 5 | true | VALID | R5C-CORR-A9DC7B173C2E6C0E | R5A-REQ-D7205BB7F519FFAE |
| mistral:7b-instruct | LastFM | HISTORY_LENGTH_10 | 10 | true | 5 | true | VALID | R5C-CORR-204F5AEC37231B1A | R5A-REQ-E62101FEF9C5EF0A |
| mistral:7b-instruct | LastFM | HISTORY_LENGTH_20 | 20 | true | 5 | true | VALID | R5C-CORR-10E70304F6CCD6CE | R5A-REQ-01B6B77B1F7A4A9A |
| mistral:7b-instruct | LastFM | HISTORY_LENGTH_5 | 5 | true | 5 | true | VALID | R5C-CORR-498376B9A5FA0590 | R5A-REQ-4F710474A2B2C090 |
| qwen2:7b-instruct | LastFM | HISTORY_LENGTH_10 | 10 | true | 5 | true | VALID | R5C-CORR-7CD703BA8ECCD09C | R5A-REQ-99DEDCDD00697232 |
| qwen2:7b-instruct | LastFM | HISTORY_LENGTH_20 | 20 | true | 5 | true | VALID | R5C-CORR-22B4399348A7C911 | R5A-REQ-FAC19D3433A064A0 |
| qwen2:7b-instruct | LastFM | HISTORY_LENGTH_5 | 5 | true | 5 | true | VALID | R5C-CORR-84702A67101F9846 | R5A-REQ-CE3564ADF2D7D19A |
| mistral:7b-instruct | ML-1M | HISTORY_LENGTH_10 | 10 | true | 5 | true | VALID | R5C-CORR-732E39DD37DC576F | R5A-REQ-E48FB83AD8659AFC |
| mistral:7b-instruct | ML-1M | HISTORY_LENGTH_20 | 20 | true | 5 | true | VALID | R5C-CORR-8E11C9844C553376 | R5A-REQ-5413BC1549E14434 |
| mistral:7b-instruct | ML-1M | HISTORY_LENGTH_5 | 5 | true | 5 | true | VALID | R5C-CORR-1C9D66FD561834CA | R5A-REQ-16F19EBA4BFEF741 |
| qwen2:7b-instruct | ML-1M | HISTORY_LENGTH_10 | 10 | true | 5 | true | VALID | R5C-CORR-91DA3B1D416F1414 | R5A-REQ-55F08DAB82172AD9 |
| qwen2:7b-instruct | ML-1M | HISTORY_LENGTH_20 | 20 | true | 5 | true | VALID | R5C-CORR-B5B839CD4F6A931D | R5A-REQ-4FB3FB62FCF4F158 |
| qwen2:7b-instruct | ML-1M | HISTORY_LENGTH_5 | 5 | true | 5 | true | VALID | R5C-CORR-03450F51E2DA0FF3 | R5A-REQ-C151AC4AEE2FDC6A |
| mistral:7b-instruct | Sports | HISTORY_LENGTH_10 | 10 | true | 5 | true | VALID | R5C-CORR-5B8A5A680208B5CC | R5A-REQ-17C941BD12EC9064 |
| mistral:7b-instruct | Sports | HISTORY_LENGTH_20 | 20 | true | 5 | true | VALID | R5C-CORR-C786B1523CEB8332 | R5A-REQ-571ABBF91E9C0C00 |
| mistral:7b-instruct | Sports | HISTORY_LENGTH_5 | 5 | true | 5 | true | VALID | R5C-CORR-B85A68B6E194F4E5 | R5A-REQ-C5E1D53258F097D2 |
| qwen2:7b-instruct | Sports | HISTORY_LENGTH_10 | 10 | true | 5 | true | VALID | R5C-CORR-4001077CD594CF80 | R5A-REQ-2DFFFDE404E9690A |
| qwen2:7b-instruct | Sports | HISTORY_LENGTH_20 | 20 | true | 5 | true | VALID | R5C-CORR-7EABF9CC5C8D34F8 | R5A-REQ-40DDB170C9EFD001 |
| qwen2:7b-instruct | Sports | HISTORY_LENGTH_5 | 5 | true | 5 | true | VALID | R5C-CORR-5061E183F2269C08 | R5A-REQ-793E529CE1F90888 |

## Early health check

| scope | key | accounted | valid | invalid | validity_rate | wrong_length | parse_failure | backend_incomplete |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | all | 72 | 72 | 0 | 1.0 | 0 | 0 | 0 |
| model | mistral:7b-instruct | 36 | 36 | 0 | 1.0 | 0 | 0 | 0 |
| model | qwen2:7b-instruct | 36 | 36 | 0 | 1.0 | 0 | 0 | 0 |
| dataset | Beauty | 18 | 18 | 0 | 1.0 | 0 | 0 | 0 |
| dataset | LastFM | 18 | 18 | 0 | 1.0 | 0 | 0 | 0 |
| dataset | ML-1M | 18 | 18 | 0 | 1.0 | 0 | 0 | 0 |
| dataset | Sports | 18 | 18 | 0 | 1.0 | 0 | 0 | 0 |
| condition | HISTORY_LENGTH_0 | 6 | 6 | 0 | 1.0 | 0 | 0 | 0 |
| condition | HISTORY_LENGTH_10 | 24 | 24 | 0 | 1.0 | 0 | 0 | 0 |
| condition | HISTORY_LENGTH_20 | 24 | 24 | 0 | 1.0 | 0 | 0 | 0 |
| condition | HISTORY_LENGTH_5 | 18 | 18 | 0 | 1.0 | 0 | 0 | 0 |
| history_length | 0 | 6 | 6 | 0 | 1.0 | 0 | 0 | 0 |
| history_length | 10 | 24 | 24 | 0 | 1.0 | 0 | 0 | 0 |
| history_length | 20 | 24 | 24 | 0 | 1.0 | 0 | 0 | 0 |
| history_length | 5 | 18 | 18 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | mistral:7b-instruct|Beauty|HISTORY_LENGTH_0 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | mistral:7b-instruct|Beauty|HISTORY_LENGTH_10 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | mistral:7b-instruct|Beauty|HISTORY_LENGTH_20 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | mistral:7b-instruct|LastFM|HISTORY_LENGTH_10 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | mistral:7b-instruct|LastFM|HISTORY_LENGTH_20 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | mistral:7b-instruct|LastFM|HISTORY_LENGTH_5 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | mistral:7b-instruct|ML-1M|HISTORY_LENGTH_10 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | mistral:7b-instruct|ML-1M|HISTORY_LENGTH_20 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | mistral:7b-instruct|ML-1M|HISTORY_LENGTH_5 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | mistral:7b-instruct|Sports|HISTORY_LENGTH_10 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | mistral:7b-instruct|Sports|HISTORY_LENGTH_20 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | mistral:7b-instruct|Sports|HISTORY_LENGTH_5 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | qwen2:7b-instruct|Beauty|HISTORY_LENGTH_0 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | qwen2:7b-instruct|Beauty|HISTORY_LENGTH_10 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | qwen2:7b-instruct|Beauty|HISTORY_LENGTH_20 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | qwen2:7b-instruct|LastFM|HISTORY_LENGTH_10 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | qwen2:7b-instruct|LastFM|HISTORY_LENGTH_20 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | qwen2:7b-instruct|LastFM|HISTORY_LENGTH_5 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | qwen2:7b-instruct|ML-1M|HISTORY_LENGTH_10 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | qwen2:7b-instruct|ML-1M|HISTORY_LENGTH_20 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | qwen2:7b-instruct|ML-1M|HISTORY_LENGTH_5 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | qwen2:7b-instruct|Sports|HISTORY_LENGTH_10 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | qwen2:7b-instruct|Sports|HISTORY_LENGTH_20 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | qwen2:7b-instruct|Sports|HISTORY_LENGTH_5 | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |

## Full-run authorization

FULL RUN AUTHORIZED: `YES`.

Blocked reason: ``.

## Execution accounting

- Requests expected: `2400`
- Requests accounted: `2400`
- Executed this invocation: `2328`
- Valid: `2400`
- Invalid: `0`
- Wrong length: `0`
- Parse failure: `0`
- Backend incomplete: `0`

## Output validity

| dataset | model_name | n_manifest_requests | n_terminal | n_valid_outputs | n_invalid_outputs | n_wrong_length | n_parse_failure | n_backend_incomplete |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Beauty | mistral:7b-instruct | 300 | 300 | 300 | 0 | 0 | 0 | 0 |
| Beauty | qwen2:7b-instruct | 300 | 300 | 300 | 0 | 0 | 0 | 0 |
| LastFM | mistral:7b-instruct | 300 | 300 | 300 | 0 | 0 | 0 | 0 |
| LastFM | qwen2:7b-instruct | 300 | 300 | 300 | 0 | 0 | 0 | 0 |
| ML-1M | mistral:7b-instruct | 300 | 300 | 300 | 0 | 0 | 0 | 0 |
| ML-1M | qwen2:7b-instruct | 300 | 300 | 300 | 0 | 0 | 0 | 0 |
| Sports | mistral:7b-instruct | 300 | 300 | 300 | 0 | 0 | 0 | 0 |
| Sports | qwen2:7b-instruct | 300 | 300 | 300 | 0 | 0 | 0 | 0 |

## Metrics by history length

| dataset | model_name | experimental_condition | history_length | n_manifest | n_valid | n_invalid | HR@5 | NDCG@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Beauty | mistral:7b-instruct | HISTORY_LENGTH_0 | 0 | 100 | 100 | 0 | 0.17 | 0.10126800101389016 |
| Beauty | mistral:7b-instruct | HISTORY_LENGTH_10 | 10 | 100 | 100 | 0 | 0.25 | 0.15398725039056177 |
| Beauty | mistral:7b-instruct | HISTORY_LENGTH_20 | 20 | 100 | 100 | 0 | 0.28 | 0.17672972152470354 |
| Beauty | qwen2:7b-instruct | HISTORY_LENGTH_0 | 0 | 100 | 100 | 0 | 0.25 | 0.13508825577529385 |
| Beauty | qwen2:7b-instruct | HISTORY_LENGTH_10 | 10 | 100 | 100 | 0 | 0.3 | 0.1786550821768666 |
| Beauty | qwen2:7b-instruct | HISTORY_LENGTH_20 | 20 | 100 | 100 | 0 | 0.32 | 0.18052107829423142 |
| LastFM | mistral:7b-instruct | HISTORY_LENGTH_10 | 10 | 100 | 100 | 0 | 0.67 | 0.4401011767951517 |
| LastFM | mistral:7b-instruct | HISTORY_LENGTH_20 | 20 | 100 | 100 | 0 | 0.6 | 0.3836680031967245 |
| LastFM | mistral:7b-instruct | HISTORY_LENGTH_5 | 5 | 100 | 100 | 0 | 0.58 | 0.37959118297339656 |
| LastFM | qwen2:7b-instruct | HISTORY_LENGTH_10 | 10 | 100 | 100 | 0 | 0.33 | 0.19982784387496536 |
| LastFM | qwen2:7b-instruct | HISTORY_LENGTH_20 | 20 | 100 | 100 | 0 | 0.32 | 0.20665255022188597 |
| LastFM | qwen2:7b-instruct | HISTORY_LENGTH_5 | 5 | 100 | 100 | 0 | 0.38 | 0.24109043665078483 |
| ML-1M | mistral:7b-instruct | HISTORY_LENGTH_10 | 10 | 100 | 100 | 0 | 0.44 | 0.26162391347496483 |
| ML-1M | mistral:7b-instruct | HISTORY_LENGTH_20 | 20 | 100 | 100 | 0 | 0.45 | 0.2856521991423963 |
| ML-1M | mistral:7b-instruct | HISTORY_LENGTH_5 | 5 | 100 | 100 | 0 | 0.35 | 0.2061317264212289 |
| ML-1M | qwen2:7b-instruct | HISTORY_LENGTH_10 | 10 | 100 | 100 | 0 | 0.34 | 0.22856708764652744 |
| ML-1M | qwen2:7b-instruct | HISTORY_LENGTH_20 | 20 | 100 | 100 | 0 | 0.3 | 0.20755949178158553 |
| ML-1M | qwen2:7b-instruct | HISTORY_LENGTH_5 | 5 | 100 | 100 | 0 | 0.37 | 0.24141021308591162 |
| Sports | mistral:7b-instruct | HISTORY_LENGTH_10 | 10 | 100 | 100 | 0 | 0.39 | 0.2287735432501283 |
| Sports | mistral:7b-instruct | HISTORY_LENGTH_20 | 20 | 100 | 100 | 0 | 0.42 | 0.24054456129665625 |
| Sports | mistral:7b-instruct | HISTORY_LENGTH_5 | 5 | 100 | 100 | 0 | 0.4 | 0.23274814061716712 |
| Sports | qwen2:7b-instruct | HISTORY_LENGTH_10 | 10 | 100 | 100 | 0 | 0.26 | 0.1741292012176286 |
| Sports | qwen2:7b-instruct | HISTORY_LENGTH_20 | 20 | 100 | 100 | 0 | 0.32 | 0.20073478543466483 |
| Sports | qwen2:7b-instruct | HISTORY_LENGTH_5 | 5 | 100 | 100 | 0 | 0.24 | 0.16043849875334318 |

## History-length results

| dataset | model_name | metric | history_length | experimental_condition | value | result_type |
| --- | --- | --- | --- | --- | --- | --- |
| Beauty | mistral:7b-instruct | HR@5 | 0 | HISTORY_LENGTH_0 | 0.17 | level |
| Beauty | mistral:7b-instruct | NDCG@5 | 0 | HISTORY_LENGTH_0 | 0.10126800101389016 | level |
| Beauty | mistral:7b-instruct | HR@5 | 10 | HISTORY_LENGTH_10 | 0.25 | level |
| Beauty | mistral:7b-instruct | NDCG@5 | 10 | HISTORY_LENGTH_10 | 0.15398725039056177 | level |
| Beauty | mistral:7b-instruct | HR@5 | 20 | HISTORY_LENGTH_20 | 0.28 | level |
| Beauty | mistral:7b-instruct | NDCG@5 | 20 | HISTORY_LENGTH_20 | 0.17672972152470354 | level |
| Beauty | qwen2:7b-instruct | HR@5 | 0 | HISTORY_LENGTH_0 | 0.25 | level |
| Beauty | qwen2:7b-instruct | NDCG@5 | 0 | HISTORY_LENGTH_0 | 0.13508825577529385 | level |
| Beauty | qwen2:7b-instruct | HR@5 | 10 | HISTORY_LENGTH_10 | 0.3 | level |
| Beauty | qwen2:7b-instruct | NDCG@5 | 10 | HISTORY_LENGTH_10 | 0.1786550821768666 | level |
| Beauty | qwen2:7b-instruct | HR@5 | 20 | HISTORY_LENGTH_20 | 0.32 | level |
| Beauty | qwen2:7b-instruct | NDCG@5 | 20 | HISTORY_LENGTH_20 | 0.18052107829423142 | level |
| LastFM | mistral:7b-instruct | HR@5 | 10 | HISTORY_LENGTH_10 | 0.67 | level |
| LastFM | mistral:7b-instruct | NDCG@5 | 10 | HISTORY_LENGTH_10 | 0.4401011767951517 | level |
| LastFM | mistral:7b-instruct | HR@5 | 20 | HISTORY_LENGTH_20 | 0.6 | level |
| LastFM | mistral:7b-instruct | NDCG@5 | 20 | HISTORY_LENGTH_20 | 0.3836680031967245 | level |
| LastFM | mistral:7b-instruct | HR@5 | 5 | HISTORY_LENGTH_5 | 0.58 | level |
| LastFM | mistral:7b-instruct | NDCG@5 | 5 | HISTORY_LENGTH_5 | 0.37959118297339656 | level |
| LastFM | qwen2:7b-instruct | HR@5 | 10 | HISTORY_LENGTH_10 | 0.33 | level |
| LastFM | qwen2:7b-instruct | NDCG@5 | 10 | HISTORY_LENGTH_10 | 0.19982784387496536 | level |
| LastFM | qwen2:7b-instruct | HR@5 | 20 | HISTORY_LENGTH_20 | 0.32 | level |
| LastFM | qwen2:7b-instruct | NDCG@5 | 20 | HISTORY_LENGTH_20 | 0.20665255022188597 | level |
| LastFM | qwen2:7b-instruct | HR@5 | 5 | HISTORY_LENGTH_5 | 0.38 | level |
| LastFM | qwen2:7b-instruct | NDCG@5 | 5 | HISTORY_LENGTH_5 | 0.24109043665078483 | level |
| ML-1M | mistral:7b-instruct | HR@5 | 10 | HISTORY_LENGTH_10 | 0.44 | level |
| ML-1M | mistral:7b-instruct | NDCG@5 | 10 | HISTORY_LENGTH_10 | 0.26162391347496483 | level |
| ML-1M | mistral:7b-instruct | HR@5 | 20 | HISTORY_LENGTH_20 | 0.45 | level |
| ML-1M | mistral:7b-instruct | NDCG@5 | 20 | HISTORY_LENGTH_20 | 0.2856521991423963 | level |
| ML-1M | mistral:7b-instruct | HR@5 | 5 | HISTORY_LENGTH_5 | 0.35 | level |
| ML-1M | mistral:7b-instruct | NDCG@5 | 5 | HISTORY_LENGTH_5 | 0.2061317264212289 | level |
| ML-1M | qwen2:7b-instruct | HR@5 | 10 | HISTORY_LENGTH_10 | 0.34 | level |
| ML-1M | qwen2:7b-instruct | NDCG@5 | 10 | HISTORY_LENGTH_10 | 0.22856708764652744 | level |
| ML-1M | qwen2:7b-instruct | HR@5 | 20 | HISTORY_LENGTH_20 | 0.3 | level |
| ML-1M | qwen2:7b-instruct | NDCG@5 | 20 | HISTORY_LENGTH_20 | 0.20755949178158553 | level |
| ML-1M | qwen2:7b-instruct | HR@5 | 5 | HISTORY_LENGTH_5 | 0.37 | level |
| ML-1M | qwen2:7b-instruct | NDCG@5 | 5 | HISTORY_LENGTH_5 | 0.24141021308591162 | level |
| Sports | mistral:7b-instruct | HR@5 | 10 | HISTORY_LENGTH_10 | 0.39 | level |
| Sports | mistral:7b-instruct | NDCG@5 | 10 | HISTORY_LENGTH_10 | 0.2287735432501283 | level |
| Sports | mistral:7b-instruct | HR@5 | 20 | HISTORY_LENGTH_20 | 0.42 | level |
| Sports | mistral:7b-instruct | NDCG@5 | 20 | HISTORY_LENGTH_20 | 0.24054456129665625 | level |
| Sports | mistral:7b-instruct | HR@5 | 5 | HISTORY_LENGTH_5 | 0.4 | level |
| Sports | mistral:7b-instruct | NDCG@5 | 5 | HISTORY_LENGTH_5 | 0.23274814061716712 | level |
| Sports | qwen2:7b-instruct | HR@5 | 10 | HISTORY_LENGTH_10 | 0.26 | level |
| Sports | qwen2:7b-instruct | NDCG@5 | 10 | HISTORY_LENGTH_10 | 0.1741292012176286 | level |
| Sports | qwen2:7b-instruct | HR@5 | 20 | HISTORY_LENGTH_20 | 0.32 | level |
| Sports | qwen2:7b-instruct | NDCG@5 | 20 | HISTORY_LENGTH_20 | 0.20073478543466483 | level |
| Sports | qwen2:7b-instruct | HR@5 | 5 | HISTORY_LENGTH_5 | 0.24 | level |
| Sports | qwen2:7b-instruct | NDCG@5 | 5 | HISTORY_LENGTH_5 | 0.16043849875334318 | level |
| Beauty | mistral:7b-instruct | HR@5_delta_long_minus_short | 0->20 | HISTORY_LENGTH_0->HISTORY_LENGTH_20 | 0.11000000000000001 | delta |
| Beauty | mistral:7b-instruct | NDCG@5_delta_long_minus_short | 0->20 | HISTORY_LENGTH_0->HISTORY_LENGTH_20 | 0.07546172051081337 | delta |
| Beauty | qwen2:7b-instruct | HR@5_delta_long_minus_short | 0->20 | HISTORY_LENGTH_0->HISTORY_LENGTH_20 | 0.07 | delta |
| Beauty | qwen2:7b-instruct | NDCG@5_delta_long_minus_short | 0->20 | HISTORY_LENGTH_0->HISTORY_LENGTH_20 | 0.04543282251893757 | delta |
| LastFM | mistral:7b-instruct | HR@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 0.020000000000000018 | delta |
| LastFM | mistral:7b-instruct | NDCG@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 0.004076820223327948 | delta |
| LastFM | qwen2:7b-instruct | HR@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | -0.06 | delta |
| LastFM | qwen2:7b-instruct | NDCG@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | -0.03443788642889886 | delta |
| ML-1M | mistral:7b-instruct | HR@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 0.10000000000000003 | delta |
| ML-1M | mistral:7b-instruct | NDCG@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 0.07952047272116741 | delta |
| ML-1M | qwen2:7b-instruct | HR@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | -0.07 | delta |
| ML-1M | qwen2:7b-instruct | NDCG@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | -0.03385072130432609 | delta |
| Sports | mistral:7b-instruct | HR@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 0.019999999999999962 | delta |
| Sports | mistral:7b-instruct | NDCG@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 0.00779642067948913 | delta |
| Sports | qwen2:7b-instruct | HR@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 0.08000000000000002 | delta |
| Sports | qwen2:7b-instruct | NDCG@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 0.04029628668132165 | delta |

## Official comparison

| dataset | model_name | metric | history_length | local_adapted_value | paper_exact | comparison_class |
| --- | --- | --- | --- | --- | --- | --- |
| Beauty | mistral:7b-instruct | HR@5 | 0 | 0.17 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | mistral:7b-instruct | NDCG@5 | 0 | 0.10126800101389016 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | mistral:7b-instruct | HR@5 | 10 | 0.25 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | mistral:7b-instruct | NDCG@5 | 10 | 0.15398725039056177 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | mistral:7b-instruct | HR@5 | 20 | 0.28 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | mistral:7b-instruct | NDCG@5 | 20 | 0.17672972152470354 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | qwen2:7b-instruct | HR@5 | 0 | 0.25 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | qwen2:7b-instruct | NDCG@5 | 0 | 0.13508825577529385 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | qwen2:7b-instruct | HR@5 | 10 | 0.3 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | qwen2:7b-instruct | NDCG@5 | 10 | 0.1786550821768666 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | qwen2:7b-instruct | HR@5 | 20 | 0.32 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | qwen2:7b-instruct | NDCG@5 | 20 | 0.18052107829423142 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | mistral:7b-instruct | HR@5 | 10 | 0.67 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | mistral:7b-instruct | NDCG@5 | 10 | 0.4401011767951517 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | mistral:7b-instruct | HR@5 | 20 | 0.6 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | mistral:7b-instruct | NDCG@5 | 20 | 0.3836680031967245 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | mistral:7b-instruct | HR@5 | 5 | 0.58 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | mistral:7b-instruct | NDCG@5 | 5 | 0.37959118297339656 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | qwen2:7b-instruct | HR@5 | 10 | 0.33 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | qwen2:7b-instruct | NDCG@5 | 10 | 0.19982784387496536 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | qwen2:7b-instruct | HR@5 | 20 | 0.32 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | qwen2:7b-instruct | NDCG@5 | 20 | 0.20665255022188597 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | qwen2:7b-instruct | HR@5 | 5 | 0.38 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | qwen2:7b-instruct | NDCG@5 | 5 | 0.24109043665078483 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | mistral:7b-instruct | HR@5 | 10 | 0.44 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | mistral:7b-instruct | NDCG@5 | 10 | 0.26162391347496483 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | mistral:7b-instruct | HR@5 | 20 | 0.45 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | mistral:7b-instruct | NDCG@5 | 20 | 0.2856521991423963 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | mistral:7b-instruct | HR@5 | 5 | 0.35 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | mistral:7b-instruct | NDCG@5 | 5 | 0.2061317264212289 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | qwen2:7b-instruct | HR@5 | 10 | 0.34 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | qwen2:7b-instruct | NDCG@5 | 10 | 0.22856708764652744 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | qwen2:7b-instruct | HR@5 | 20 | 0.3 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | qwen2:7b-instruct | NDCG@5 | 20 | 0.20755949178158553 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | qwen2:7b-instruct | HR@5 | 5 | 0.37 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | qwen2:7b-instruct | NDCG@5 | 5 | 0.24141021308591162 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | mistral:7b-instruct | HR@5 | 10 | 0.39 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | mistral:7b-instruct | NDCG@5 | 10 | 0.2287735432501283 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | mistral:7b-instruct | HR@5 | 20 | 0.42 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | mistral:7b-instruct | NDCG@5 | 20 | 0.24054456129665625 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | mistral:7b-instruct | HR@5 | 5 | 0.4 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | mistral:7b-instruct | NDCG@5 | 5 | 0.23274814061716712 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | qwen2:7b-instruct | HR@5 | 10 | 0.26 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | qwen2:7b-instruct | NDCG@5 | 10 | 0.1741292012176286 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | qwen2:7b-instruct | HR@5 | 20 | 0.32 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | qwen2:7b-instruct | NDCG@5 | 20 | 0.20073478543466483 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | qwen2:7b-instruct | HR@5 | 5 | 0.24 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | qwen2:7b-instruct | NDCG@5 | 5 | 0.16043849875334318 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |

## Gap resolution

| gap_id | dataset | metric | requests_expected | requests_accounted | valid | invalid | gap_recovered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R4-GAP-0007 | Beauty | History-length sensitivity | 600 | 600 | 600 | 0 | true |
| R4-GAP-0021 | LastFM | History-length sensitivity | 600 | 600 | 600 | 0 | true |
| R4-GAP-0033 | ML-1M | History-length sensitivity | 600 | 600 | 600 | 0 | true |
| R4-GAP-0047 | Sports | History-length sensitivity | 600 | 600 | 600 | 0 | true |

## Scientific interpretation

R5C_CORRECTED is a local adapted history-length recovery over frozen R5A candidate/history contexts. It is not paper-exact because the local model identifiers and corrected strict JSON index protocol differ from the paper execution environment.

## Validation

| check_id | check_name | status | observed | expected |
| --- | --- | --- | --- | --- |
| VAL-001 | static preflight | pass | 34/34 | all pass |
| VAL-002 | synthetic tests | pass | 14/14 | all pass |
| VAL-003 | context preflight | pass | 32/32 | all pass |
| VAL-004 | canary complete when attempted | pass | 24/24 | all complete |
| VAL-005 | canary parsed K=5 when attempted | pass | 24/24 | all 5 |
| VAL-006 | early health no backend incomplete | pass | checked | 0 |
| VAL-007 | group validation | pass | 800/800 | 800/800 |
| VAL-008 | incomplete not succeeded | pass | checked | none |
| VAL-009 | full run authorization consistent | pass | authorized=True;decision=R5C_CORRECTED_HISTORY_LENGTH_COMPLETE;terminal=2400 | consistent |
| VAL-CORR-010 | canary values are candidate positions | pass | checked | all true |
| VAL-CORR-011 | canary has no item-ID substitution | pass | checked | 0 |
| VAL-CORR-012 | max retries zero in terminal outputs | pass | checked | 0 |
| VAL-CORR-013 | no out_of_range when complete | pass | 0 | 0 |

## Repository state

This runner does not commit or push.

## Final decision

`R5C_CORRECTED_HISTORY_LENGTH_COMPLETE`
