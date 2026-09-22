# R5C History-Length Recovery

## Scope

R5C executes only frozen R5A `HISTORY_LENGTH_RECOVERY` ranking rows. CandDif, Generation-involved, profile-generation, downloads, training, remote APIs, R5D, and `original_repo/` edits are out of scope.

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
| STATIC-020 | corrected prompt forbids more/fewer | pass | checked | all prompts |
| STATIC-021 | corrected prompt forbids prose | pass | checked | all prompts |
| STATIC-022 | parser contract corrected | pass | ['r5c_ranked_index_json_top5_v1'] | r5c_ranked_index_json_top5_v1 |
| STATIC-023 | local endpoint only | pass | http://127.0.0.1:11434/api/generate | localhost Ollama |
| STATIC-024 | format json stream false | pass | format=json;stream=False | format=json;stream=false |
| STATIC-025 | no retry policy introduced | pass | 0 | 0 |
| STATIC-026 | output namespace R5C | pass | results/recovery/r5c_history_length/ | results/recovery/r5c_history_length/ |
| STATIC-027 | no output path overlap with original R5 | pass | 0 | 0 |
| STATIC-028 | no R5C output under R5/R5B namespaces | pass | checked | none |
| STATIC-029 | original repo clean | pass |  | empty |
| STATIC-030 | protected artifacts unchanged | pass | 12/12 | 12/12 |
| STATIC-031 | disk free guard | pass | 161153851392 | >=1073741824 |
| STATIC-032 | context envelope | pass | 32/32 | all pass |
| STATIC-033 | protocol hash present | pass | 30E4BCA9E1456EE9A7DE74A09FE03BF1A26EFCF758915052FD88F39AF9A836F2 | 64-char SHA256 |

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

## Context envelope

| scope | model_name | dataset | experimental_condition | history_length | approx_input_tokens | output_budget_tokens | context_length | margin_tokens | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_dataset_condition | mistral:7b-instruct | Beauty | HISTORY_LENGTH_0 | 0 | 788 | 512 | 32768 | 31468 | pass |
| model_dataset_condition | mistral:7b-instruct | Beauty | HISTORY_LENGTH_10 | 10 | 1061 | 512 | 32768 | 31195 | pass |
| model_dataset_condition | mistral:7b-instruct | Beauty | HISTORY_LENGTH_20 | 20 | 1061 | 512 | 32768 | 31195 | pass |
| model_dataset_condition | mistral:7b-instruct | LastFM | HISTORY_LENGTH_10 | 10 | 339 | 512 | 32768 | 31917 | pass |
| model_dataset_condition | mistral:7b-instruct | LastFM | HISTORY_LENGTH_20 | 20 | 400 | 512 | 32768 | 31856 | pass |
| model_dataset_condition | mistral:7b-instruct | LastFM | HISTORY_LENGTH_5 | 5 | 307 | 512 | 32768 | 31949 | pass |
| model_dataset_condition | mistral:7b-instruct | ML-1M | HISTORY_LENGTH_10 | 10 | 434 | 512 | 32768 | 31822 | pass |
| model_dataset_condition | mistral:7b-instruct | ML-1M | HISTORY_LENGTH_20 | 20 | 530 | 512 | 32768 | 31726 | pass |
| model_dataset_condition | mistral:7b-instruct | ML-1M | HISTORY_LENGTH_5 | 5 | 392 | 512 | 32768 | 31864 | pass |
| model_dataset_condition | mistral:7b-instruct | Sports | HISTORY_LENGTH_10 | 10 | 836 | 512 | 32768 | 31420 | pass |
| model_dataset_condition | mistral:7b-instruct | Sports | HISTORY_LENGTH_20 | 20 | 908 | 512 | 32768 | 31348 | pass |
| model_dataset_condition | mistral:7b-instruct | Sports | HISTORY_LENGTH_5 | 5 | 720 | 512 | 32768 | 31536 | pass |
| model_dataset_condition | qwen2:7b-instruct | Beauty | HISTORY_LENGTH_0 | 0 | 788 | 512 | 32768 | 31468 | pass |
| model_dataset_condition | qwen2:7b-instruct | Beauty | HISTORY_LENGTH_10 | 10 | 1061 | 512 | 32768 | 31195 | pass |
| model_dataset_condition | qwen2:7b-instruct | Beauty | HISTORY_LENGTH_20 | 20 | 1061 | 512 | 32768 | 31195 | pass |
| model_dataset_condition | qwen2:7b-instruct | LastFM | HISTORY_LENGTH_10 | 10 | 339 | 512 | 32768 | 31917 | pass |
| model_dataset_condition | qwen2:7b-instruct | LastFM | HISTORY_LENGTH_20 | 20 | 400 | 512 | 32768 | 31856 | pass |
| model_dataset_condition | qwen2:7b-instruct | LastFM | HISTORY_LENGTH_5 | 5 | 307 | 512 | 32768 | 31949 | pass |
| model_dataset_condition | qwen2:7b-instruct | ML-1M | HISTORY_LENGTH_10 | 10 | 434 | 512 | 32768 | 31822 | pass |
| model_dataset_condition | qwen2:7b-instruct | ML-1M | HISTORY_LENGTH_20 | 20 | 530 | 512 | 32768 | 31726 | pass |
| model_dataset_condition | qwen2:7b-instruct | ML-1M | HISTORY_LENGTH_5 | 5 | 392 | 512 | 32768 | 31864 | pass |
| model_dataset_condition | qwen2:7b-instruct | Sports | HISTORY_LENGTH_10 | 10 | 836 | 512 | 32768 | 31420 | pass |
| model_dataset_condition | qwen2:7b-instruct | Sports | HISTORY_LENGTH_20 | 20 | 908 | 512 | 32768 | 31348 | pass |
| model_dataset_condition | qwen2:7b-instruct | Sports | HISTORY_LENGTH_5 | 5 | 720 | 512 | 32768 | 31536 | pass |
| long_vs_short | mistral:7b-instruct | Beauty | HISTORY_LENGTH_0->HISTORY_LENGTH_20 | 0->20 | 1061 | 512 | 32768 | 31195 | pass |
| long_vs_short | mistral:7b-instruct | LastFM | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 5->20 | 400 | 512 | 32768 | 31856 | pass |
| long_vs_short | mistral:7b-instruct | ML-1M | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 5->20 | 530 | 512 | 32768 | 31726 | pass |
| long_vs_short | mistral:7b-instruct | Sports | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 5->20 | 908 | 512 | 32768 | 31348 | pass |
| long_vs_short | qwen2:7b-instruct | Beauty | HISTORY_LENGTH_0->HISTORY_LENGTH_20 | 0->20 | 1061 | 512 | 32768 | 31195 | pass |
| long_vs_short | qwen2:7b-instruct | LastFM | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 5->20 | 400 | 512 | 32768 | 31856 | pass |
| long_vs_short | qwen2:7b-instruct | ML-1M | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 5->20 | 530 | 512 | 32768 | 31726 | pass |
| long_vs_short | qwen2:7b-instruct | Sports | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 5->20 | 908 | 512 | 32768 | 31348 | pass |

## Protocol freeze

```text
R5C_PROTOCOL_PREFLIGHT_READY
PROTOCOL_HASH=30E4BCA9E1456EE9A7DE74A09FE03BF1A26EFCF758915052FD88F39AF9A836F2
```

| component | value | sha256 |
| --- | --- | --- |
| manifest | 290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD | 290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD |
| prompt_family | r5c_history_length_json_index_exact5_v1 | AA752903D223CD27337301078BE26432B382B76D9E2164A1D842E43CDE900324 |
| parser | r5c_ranked_index_json_top5_v1 | CB5DA25F85CC9B90449FC11EA436D26379C225F2F2194CDB3CF9B2306A92A364 |
| runner | C:/Users/Bartolo/Desktop/PROGETTI/IRP_BeyondUtility_Project/src/analysis/reproduction_r5c_history_length_recovery.py | 77E4871D862D2997242A61E84EC8D0C2CC311468906CBE30323D40A160134FFC |
| config | configs/reproduction_r5c_history_length_recovery.yaml | 0A1F8AAF26CBCD5FF10D1AB5694B3E4F2658498F34BA805CA711E5F4AFDDCFE8 |
| history_candidate_construction | R5A frozen history-length rows | 38E48348F29AE2E429EC30D883C9264F2D4EA72DABA65C9B55F755AEA7C32701 |
| group_validation | same candidates and positive item across history lengths | E6CE60617C88112E8BAA30811B2F5A9C70AED2270217CA7D0AA6D7E1DF3FD083 |
| metrics | HR@5/NDCG@5 by history length | 77E4871D862D2997242A61E84EC8D0C2CC311468906CBE30323D40A160134FFC |
| generation_config | {"endpoint":"http://127.0.0.1:11434/api/generate","format":"json","max_retries":"0","num_predict_values":["512"],"seed_values":["0"],"stream":"false","temperature":"0","timeout_seconds":"120"} | E62BD8A4FC52A56CBB9CCBC5AE8FA24861835197175629B091FFE188E5964801 |
| protocol | R5C_PROTOCOL_PREFLIGHT_READY | 30E4BCA9E1456EE9A7DE74A09FE03BF1A26EFCF758915052FD88F39AF9A836F2 |
| model_identifier | mistral:7b-instruct | 6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf |
| model_identifier | qwen2:7b-instruct | dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755 |

## Canary results

| Model | Dataset | Condition | HistoryLength | Complete | Parsed K | Valid | Failure | request_id | r5a_request_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mistral:7b-instruct | Beauty | HISTORY_LENGTH_0 | 0 | true | 5 | true | VALID | R5C-HIST-A9BFF35307CEC628 | R5A-REQ-A047C418739887F3 |
| mistral:7b-instruct | Beauty | HISTORY_LENGTH_10 | 10 | true | 5 | true | VALID | R5C-HIST-1CC1F0DCE614DEBD | R5A-REQ-4B59FA646FA5E44B |
| mistral:7b-instruct | Beauty | HISTORY_LENGTH_20 | 20 | true | 5 | true | VALID | R5C-HIST-8E4199E9C623C6FB | R5A-REQ-91B45BCD83A063BE |
| qwen2:7b-instruct | Beauty | HISTORY_LENGTH_0 | 0 | true | 5 | true | VALID | R5C-HIST-6E82E1DFF7C37E23 | R5A-REQ-964FFF65E0CDD226 |
| qwen2:7b-instruct | Beauty | HISTORY_LENGTH_10 | 10 | true | 5 | true | VALID | R5C-HIST-2DCF937780D75727 | R5A-REQ-F9B0A51E75F1A054 |
| qwen2:7b-instruct | Beauty | HISTORY_LENGTH_20 | 20 | true | 5 | true | VALID | R5C-HIST-06E9657BD6863953 | R5A-REQ-D7205BB7F519FFAE |
| mistral:7b-instruct | LastFM | HISTORY_LENGTH_10 | 10 | true | 5 | true | VALID | R5C-HIST-B6BA5106F7B59F2E | R5A-REQ-E62101FEF9C5EF0A |
| mistral:7b-instruct | LastFM | HISTORY_LENGTH_20 | 20 | true | 5 | true | VALID | R5C-HIST-B4B6C84C11CC1354 | R5A-REQ-01B6B77B1F7A4A9A |
| mistral:7b-instruct | LastFM | HISTORY_LENGTH_5 | 5 | true | 5 | true | VALID | R5C-HIST-9F40D290A240268A | R5A-REQ-4F710474A2B2C090 |
| qwen2:7b-instruct | LastFM | HISTORY_LENGTH_10 | 10 | true | 5 | true | VALID | R5C-HIST-76C26AA291F844B4 | R5A-REQ-99DEDCDD00697232 |
| qwen2:7b-instruct | LastFM | HISTORY_LENGTH_20 | 20 | true | 5 | true | VALID | R5C-HIST-8B4F2C2E37E8EA66 | R5A-REQ-FAC19D3433A064A0 |
| qwen2:7b-instruct | LastFM | HISTORY_LENGTH_5 | 5 | true | 5 | true | VALID | R5C-HIST-D07B6DDA21A6141D | R5A-REQ-CE3564ADF2D7D19A |
| mistral:7b-instruct | ML-1M | HISTORY_LENGTH_10 | 10 | true | 5 | true | VALID | R5C-HIST-DEBA613A9B7C4C81 | R5A-REQ-E48FB83AD8659AFC |
| mistral:7b-instruct | ML-1M | HISTORY_LENGTH_20 | 20 | true | 5 | true | VALID | R5C-HIST-A5F7821981D75794 | R5A-REQ-5413BC1549E14434 |
| mistral:7b-instruct | ML-1M | HISTORY_LENGTH_5 | 5 | true | 5 | true | VALID | R5C-HIST-C9ADCCD809197175 | R5A-REQ-16F19EBA4BFEF741 |
| qwen2:7b-instruct | ML-1M | HISTORY_LENGTH_10 | 10 | true | 5 | true | VALID | R5C-HIST-64372A9492BEF629 | R5A-REQ-55F08DAB82172AD9 |
| qwen2:7b-instruct | ML-1M | HISTORY_LENGTH_20 | 20 | true | 5 | true | VALID | R5C-HIST-B27CE16FDDE90519 | R5A-REQ-4FB3FB62FCF4F158 |
| qwen2:7b-instruct | ML-1M | HISTORY_LENGTH_5 | 5 | true | 5 | true | VALID | R5C-HIST-F249130D47FD23CA | R5A-REQ-C151AC4AEE2FDC6A |
| mistral:7b-instruct | Sports | HISTORY_LENGTH_10 | 10 | true | 5 | true | VALID | R5C-HIST-5949FD94C4CEB9C5 | R5A-REQ-17C941BD12EC9064 |
| mistral:7b-instruct | Sports | HISTORY_LENGTH_20 | 20 | true | 5 | true | VALID | R5C-HIST-F002878350209675 | R5A-REQ-571ABBF91E9C0C00 |
| mistral:7b-instruct | Sports | HISTORY_LENGTH_5 | 5 | true | 5 | true | VALID | R5C-HIST-BC27542CC7B8C45C | R5A-REQ-C5E1D53258F097D2 |
| qwen2:7b-instruct | Sports | HISTORY_LENGTH_10 | 10 | true | 5 | true | VALID | R5C-HIST-24E77DBB9CFD67B0 | R5A-REQ-2DFFFDE404E9690A |
| qwen2:7b-instruct | Sports | HISTORY_LENGTH_20 | 20 | true | 5 | true | VALID | R5C-HIST-F29B3CD6B5DEFE32 | R5A-REQ-40DDB170C9EFD001 |
| qwen2:7b-instruct | Sports | HISTORY_LENGTH_5 | 5 | true | 5 | true | VALID | R5C-HIST-EDBEB40F378D6DD7 | R5A-REQ-793E529CE1F90888 |

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
- Valid: `2385`
- Invalid: `15`
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
| ML-1M | qwen2:7b-instruct | 300 | 300 | 285 | 15 | 0 | 0 | 0 |
| Sports | mistral:7b-instruct | 300 | 300 | 300 | 0 | 0 | 0 | 0 |
| Sports | qwen2:7b-instruct | 300 | 300 | 300 | 0 | 0 | 0 | 0 |

## Metrics by history length

| dataset | model_name | experimental_condition | history_length | n_manifest | n_valid | n_invalid | HR@5 | NDCG@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Beauty | mistral:7b-instruct | HISTORY_LENGTH_0 | 0 | 100 | 100 | 0 | 0.2 | 0.10866747395543491 |
| Beauty | mistral:7b-instruct | HISTORY_LENGTH_10 | 10 | 100 | 100 | 0 | 0.35 | 0.2011371414106799 |
| Beauty | mistral:7b-instruct | HISTORY_LENGTH_20 | 20 | 100 | 100 | 0 | 0.34 | 0.2011676079536024 |
| Beauty | qwen2:7b-instruct | HISTORY_LENGTH_0 | 0 | 100 | 100 | 0 | 0.27 | 0.12908763669946924 |
| Beauty | qwen2:7b-instruct | HISTORY_LENGTH_10 | 10 | 100 | 100 | 0 | 0.31 | 0.16002347625104454 |
| Beauty | qwen2:7b-instruct | HISTORY_LENGTH_20 | 20 | 100 | 100 | 0 | 0.33 | 0.17407524492090098 |
| LastFM | mistral:7b-instruct | HISTORY_LENGTH_10 | 10 | 100 | 100 | 0 | 0.6 | 0.3647744235709076 |
| LastFM | mistral:7b-instruct | HISTORY_LENGTH_20 | 20 | 100 | 100 | 0 | 0.51 | 0.3311913589523622 |
| LastFM | mistral:7b-instruct | HISTORY_LENGTH_5 | 5 | 100 | 100 | 0 | 0.52 | 0.307008201438006 |
| LastFM | qwen2:7b-instruct | HISTORY_LENGTH_10 | 10 | 100 | 100 | 0 | 0.48 | 0.2956702671553702 |
| LastFM | qwen2:7b-instruct | HISTORY_LENGTH_20 | 20 | 100 | 100 | 0 | 0.44 | 0.27248258172372253 |
| LastFM | qwen2:7b-instruct | HISTORY_LENGTH_5 | 5 | 100 | 100 | 0 | 0.4 | 0.25726861333833445 |
| ML-1M | mistral:7b-instruct | HISTORY_LENGTH_10 | 10 | 100 | 100 | 0 | 0.48 | 0.2798639866522934 |
| ML-1M | mistral:7b-instruct | HISTORY_LENGTH_20 | 20 | 100 | 100 | 0 | 0.4 | 0.2556521991423963 |
| ML-1M | mistral:7b-instruct | HISTORY_LENGTH_5 | 5 | 100 | 100 | 0 | 0.44 | 0.28214473727562184 |
| ML-1M | qwen2:7b-instruct | HISTORY_LENGTH_10 | 10 | 100 | 97 | 3 | 0.39 | 0.23319362291100787 |
| ML-1M | qwen2:7b-instruct | HISTORY_LENGTH_20 | 20 | 100 | 97 | 3 | 0.38 | 0.2356989039022914 |
| ML-1M | qwen2:7b-instruct | HISTORY_LENGTH_5 | 5 | 100 | 91 | 9 | 0.34 | 0.210770660215658 |
| Sports | mistral:7b-instruct | HISTORY_LENGTH_10 | 10 | 100 | 100 | 0 | 0.35 | 0.20045629876998214 |
| Sports | mistral:7b-instruct | HISTORY_LENGTH_20 | 20 | 100 | 100 | 0 | 0.28 | 0.1602838881513731 |
| Sports | mistral:7b-instruct | HISTORY_LENGTH_5 | 5 | 100 | 100 | 0 | 0.26 | 0.16224513033590962 |
| Sports | qwen2:7b-instruct | HISTORY_LENGTH_10 | 10 | 100 | 100 | 0 | 0.42 | 0.22946135592856315 |
| Sports | qwen2:7b-instruct | HISTORY_LENGTH_20 | 20 | 100 | 100 | 0 | 0.38 | 0.22255407004075423 |
| Sports | qwen2:7b-instruct | HISTORY_LENGTH_5 | 5 | 100 | 100 | 0 | 0.4 | 0.23135084179973306 |

## History-length results

| dataset | model_name | metric | history_length | experimental_condition | value | result_type |
| --- | --- | --- | --- | --- | --- | --- |
| Beauty | mistral:7b-instruct | HR@5 | 0 | HISTORY_LENGTH_0 | 0.2 | level |
| Beauty | mistral:7b-instruct | NDCG@5 | 0 | HISTORY_LENGTH_0 | 0.10866747395543491 | level |
| Beauty | mistral:7b-instruct | HR@5 | 10 | HISTORY_LENGTH_10 | 0.35 | level |
| Beauty | mistral:7b-instruct | NDCG@5 | 10 | HISTORY_LENGTH_10 | 0.2011371414106799 | level |
| Beauty | mistral:7b-instruct | HR@5 | 20 | HISTORY_LENGTH_20 | 0.34 | level |
| Beauty | mistral:7b-instruct | NDCG@5 | 20 | HISTORY_LENGTH_20 | 0.2011676079536024 | level |
| Beauty | qwen2:7b-instruct | HR@5 | 0 | HISTORY_LENGTH_0 | 0.27 | level |
| Beauty | qwen2:7b-instruct | NDCG@5 | 0 | HISTORY_LENGTH_0 | 0.12908763669946924 | level |
| Beauty | qwen2:7b-instruct | HR@5 | 10 | HISTORY_LENGTH_10 | 0.31 | level |
| Beauty | qwen2:7b-instruct | NDCG@5 | 10 | HISTORY_LENGTH_10 | 0.16002347625104454 | level |
| Beauty | qwen2:7b-instruct | HR@5 | 20 | HISTORY_LENGTH_20 | 0.33 | level |
| Beauty | qwen2:7b-instruct | NDCG@5 | 20 | HISTORY_LENGTH_20 | 0.17407524492090098 | level |
| LastFM | mistral:7b-instruct | HR@5 | 10 | HISTORY_LENGTH_10 | 0.6 | level |
| LastFM | mistral:7b-instruct | NDCG@5 | 10 | HISTORY_LENGTH_10 | 0.3647744235709076 | level |
| LastFM | mistral:7b-instruct | HR@5 | 20 | HISTORY_LENGTH_20 | 0.51 | level |
| LastFM | mistral:7b-instruct | NDCG@5 | 20 | HISTORY_LENGTH_20 | 0.3311913589523622 | level |
| LastFM | mistral:7b-instruct | HR@5 | 5 | HISTORY_LENGTH_5 | 0.52 | level |
| LastFM | mistral:7b-instruct | NDCG@5 | 5 | HISTORY_LENGTH_5 | 0.307008201438006 | level |
| LastFM | qwen2:7b-instruct | HR@5 | 10 | HISTORY_LENGTH_10 | 0.48 | level |
| LastFM | qwen2:7b-instruct | NDCG@5 | 10 | HISTORY_LENGTH_10 | 0.2956702671553702 | level |
| LastFM | qwen2:7b-instruct | HR@5 | 20 | HISTORY_LENGTH_20 | 0.44 | level |
| LastFM | qwen2:7b-instruct | NDCG@5 | 20 | HISTORY_LENGTH_20 | 0.27248258172372253 | level |
| LastFM | qwen2:7b-instruct | HR@5 | 5 | HISTORY_LENGTH_5 | 0.4 | level |
| LastFM | qwen2:7b-instruct | NDCG@5 | 5 | HISTORY_LENGTH_5 | 0.25726861333833445 | level |
| ML-1M | mistral:7b-instruct | HR@5 | 10 | HISTORY_LENGTH_10 | 0.48 | level |
| ML-1M | mistral:7b-instruct | NDCG@5 | 10 | HISTORY_LENGTH_10 | 0.2798639866522934 | level |
| ML-1M | mistral:7b-instruct | HR@5 | 20 | HISTORY_LENGTH_20 | 0.4 | level |
| ML-1M | mistral:7b-instruct | NDCG@5 | 20 | HISTORY_LENGTH_20 | 0.2556521991423963 | level |
| ML-1M | mistral:7b-instruct | HR@5 | 5 | HISTORY_LENGTH_5 | 0.44 | level |
| ML-1M | mistral:7b-instruct | NDCG@5 | 5 | HISTORY_LENGTH_5 | 0.28214473727562184 | level |
| ML-1M | qwen2:7b-instruct | HR@5 | 10 | HISTORY_LENGTH_10 | 0.39 | level |
| ML-1M | qwen2:7b-instruct | NDCG@5 | 10 | HISTORY_LENGTH_10 | 0.23319362291100787 | level |
| ML-1M | qwen2:7b-instruct | HR@5 | 20 | HISTORY_LENGTH_20 | 0.38 | level |
| ML-1M | qwen2:7b-instruct | NDCG@5 | 20 | HISTORY_LENGTH_20 | 0.2356989039022914 | level |
| ML-1M | qwen2:7b-instruct | HR@5 | 5 | HISTORY_LENGTH_5 | 0.34 | level |
| ML-1M | qwen2:7b-instruct | NDCG@5 | 5 | HISTORY_LENGTH_5 | 0.210770660215658 | level |
| Sports | mistral:7b-instruct | HR@5 | 10 | HISTORY_LENGTH_10 | 0.35 | level |
| Sports | mistral:7b-instruct | NDCG@5 | 10 | HISTORY_LENGTH_10 | 0.20045629876998214 | level |
| Sports | mistral:7b-instruct | HR@5 | 20 | HISTORY_LENGTH_20 | 0.28 | level |
| Sports | mistral:7b-instruct | NDCG@5 | 20 | HISTORY_LENGTH_20 | 0.1602838881513731 | level |
| Sports | mistral:7b-instruct | HR@5 | 5 | HISTORY_LENGTH_5 | 0.26 | level |
| Sports | mistral:7b-instruct | NDCG@5 | 5 | HISTORY_LENGTH_5 | 0.16224513033590962 | level |
| Sports | qwen2:7b-instruct | HR@5 | 10 | HISTORY_LENGTH_10 | 0.42 | level |
| Sports | qwen2:7b-instruct | NDCG@5 | 10 | HISTORY_LENGTH_10 | 0.22946135592856315 | level |
| Sports | qwen2:7b-instruct | HR@5 | 20 | HISTORY_LENGTH_20 | 0.38 | level |
| Sports | qwen2:7b-instruct | NDCG@5 | 20 | HISTORY_LENGTH_20 | 0.22255407004075423 | level |
| Sports | qwen2:7b-instruct | HR@5 | 5 | HISTORY_LENGTH_5 | 0.4 | level |
| Sports | qwen2:7b-instruct | NDCG@5 | 5 | HISTORY_LENGTH_5 | 0.23135084179973306 | level |
| Beauty | mistral:7b-instruct | HR@5_delta_long_minus_short | 0->20 | HISTORY_LENGTH_0->HISTORY_LENGTH_20 | 0.14 | delta |
| Beauty | mistral:7b-instruct | NDCG@5_delta_long_minus_short | 0->20 | HISTORY_LENGTH_0->HISTORY_LENGTH_20 | 0.09250013399816748 | delta |
| Beauty | qwen2:7b-instruct | HR@5_delta_long_minus_short | 0->20 | HISTORY_LENGTH_0->HISTORY_LENGTH_20 | 0.06 | delta |
| Beauty | qwen2:7b-instruct | NDCG@5_delta_long_minus_short | 0->20 | HISTORY_LENGTH_0->HISTORY_LENGTH_20 | 0.044987608221431746 | delta |
| LastFM | mistral:7b-instruct | HR@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | -0.010000000000000009 | delta |
| LastFM | mistral:7b-instruct | NDCG@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 0.02418315751435618 | delta |
| LastFM | qwen2:7b-instruct | HR@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 0.03999999999999998 | delta |
| LastFM | qwen2:7b-instruct | NDCG@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 0.015213968385388077 | delta |
| ML-1M | mistral:7b-instruct | HR@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | -0.03999999999999998 | delta |
| ML-1M | mistral:7b-instruct | NDCG@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | -0.02649253813322555 | delta |
| ML-1M | qwen2:7b-instruct | HR@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 0.03999999999999998 | delta |
| ML-1M | qwen2:7b-instruct | NDCG@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 0.024928243686633406 | delta |
| Sports | mistral:7b-instruct | HR@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | 0.020000000000000018 | delta |
| Sports | mistral:7b-instruct | NDCG@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | -0.001961242184536527 | delta |
| Sports | qwen2:7b-instruct | HR@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | -0.020000000000000018 | delta |
| Sports | qwen2:7b-instruct | NDCG@5_delta_long_minus_short | 5->20 | HISTORY_LENGTH_5->HISTORY_LENGTH_20 | -0.008796771758978833 | delta |

## Official comparison

| dataset | model_name | metric | history_length | local_adapted_value | paper_exact | comparison_class |
| --- | --- | --- | --- | --- | --- | --- |
| Beauty | mistral:7b-instruct | HR@5 | 0 | 0.2 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | mistral:7b-instruct | NDCG@5 | 0 | 0.10866747395543491 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | mistral:7b-instruct | HR@5 | 10 | 0.35 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | mistral:7b-instruct | NDCG@5 | 10 | 0.2011371414106799 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | mistral:7b-instruct | HR@5 | 20 | 0.34 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | mistral:7b-instruct | NDCG@5 | 20 | 0.2011676079536024 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | qwen2:7b-instruct | HR@5 | 0 | 0.27 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | qwen2:7b-instruct | NDCG@5 | 0 | 0.12908763669946924 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | qwen2:7b-instruct | HR@5 | 10 | 0.31 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | qwen2:7b-instruct | NDCG@5 | 10 | 0.16002347625104454 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | qwen2:7b-instruct | HR@5 | 20 | 0.33 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | qwen2:7b-instruct | NDCG@5 | 20 | 0.17407524492090098 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | mistral:7b-instruct | HR@5 | 10 | 0.6 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | mistral:7b-instruct | NDCG@5 | 10 | 0.3647744235709076 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | mistral:7b-instruct | HR@5 | 20 | 0.51 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | mistral:7b-instruct | NDCG@5 | 20 | 0.3311913589523622 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | mistral:7b-instruct | HR@5 | 5 | 0.52 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | mistral:7b-instruct | NDCG@5 | 5 | 0.307008201438006 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | qwen2:7b-instruct | HR@5 | 10 | 0.48 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | qwen2:7b-instruct | NDCG@5 | 10 | 0.2956702671553702 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | qwen2:7b-instruct | HR@5 | 20 | 0.44 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | qwen2:7b-instruct | NDCG@5 | 20 | 0.27248258172372253 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | qwen2:7b-instruct | HR@5 | 5 | 0.4 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | qwen2:7b-instruct | NDCG@5 | 5 | 0.25726861333833445 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | mistral:7b-instruct | HR@5 | 10 | 0.48 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | mistral:7b-instruct | NDCG@5 | 10 | 0.2798639866522934 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | mistral:7b-instruct | HR@5 | 20 | 0.4 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | mistral:7b-instruct | NDCG@5 | 20 | 0.2556521991423963 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | mistral:7b-instruct | HR@5 | 5 | 0.44 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | mistral:7b-instruct | NDCG@5 | 5 | 0.28214473727562184 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | qwen2:7b-instruct | HR@5 | 10 | 0.39 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | qwen2:7b-instruct | NDCG@5 | 10 | 0.23319362291100787 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | qwen2:7b-instruct | HR@5 | 20 | 0.38 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | qwen2:7b-instruct | NDCG@5 | 20 | 0.2356989039022914 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | qwen2:7b-instruct | HR@5 | 5 | 0.34 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| ML-1M | qwen2:7b-instruct | NDCG@5 | 5 | 0.210770660215658 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | mistral:7b-instruct | HR@5 | 10 | 0.35 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | mistral:7b-instruct | NDCG@5 | 10 | 0.20045629876998214 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | mistral:7b-instruct | HR@5 | 20 | 0.28 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | mistral:7b-instruct | NDCG@5 | 20 | 0.1602838881513731 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | mistral:7b-instruct | HR@5 | 5 | 0.26 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | mistral:7b-instruct | NDCG@5 | 5 | 0.16224513033590962 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | qwen2:7b-instruct | HR@5 | 10 | 0.42 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | qwen2:7b-instruct | NDCG@5 | 10 | 0.22946135592856315 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | qwen2:7b-instruct | HR@5 | 20 | 0.38 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | qwen2:7b-instruct | NDCG@5 | 20 | 0.22255407004075423 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | qwen2:7b-instruct | HR@5 | 5 | 0.4 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | qwen2:7b-instruct | NDCG@5 | 5 | 0.23135084179973306 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |

## Gap resolution

| gap_id | dataset | metric | requests_expected | requests_accounted | valid | invalid | gap_recovered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R4-GAP-0007 | Beauty | History-length sensitivity | 600 | 600 | 600 | 0 | true |
| R4-GAP-0021 | LastFM | History-length sensitivity | 600 | 600 | 600 | 0 | true |
| R4-GAP-0033 | ML-1M | History-length sensitivity | 600 | 600 | 585 | 15 | false |
| R4-GAP-0047 | Sports | History-length sensitivity | 600 | 600 | 600 | 0 | true |

## Scientific interpretation

R5C is a local adapted history-length recovery over frozen R5A candidate/history contexts. It is not paper-exact because the local model identifiers and corrected strict JSON index protocol differ from the paper execution environment.

## Validation

| check_id | check_name | status | observed | expected |
| --- | --- | --- | --- | --- |
| VAL-001 | static preflight | pass | 33/33 | all pass |
| VAL-002 | synthetic tests | pass | 13/13 | all pass |
| VAL-003 | context preflight | pass | 32/32 | all pass |
| VAL-004 | canary complete when attempted | pass | 24/24 | all complete |
| VAL-005 | canary parsed K=5 when attempted | pass | 24/24 | all 5 |
| VAL-006 | early health no backend incomplete | pass | checked | 0 |
| VAL-007 | group validation | pass | 800/800 | 800/800 |
| VAL-008 | incomplete not succeeded | pass | checked | none |
| VAL-009 | full run authorization consistent | pass | authorized=True;decision=R5C_HISTORY_LENGTH_RECOVERY_COMPLETE;terminal=2400 | consistent |

## Repository state

This runner does not commit or push.

## Final decision

`R5C_HISTORY_LENGTH_RECOVERY_COMPLETE`
