# R5B Corrected CandDif Recovery

## Baseline

Previous forensic decision: `R5B_MULTIPLE_PROTOCOL_FAILURES`. Original R5B artifacts are preserved as forensic evidence and this corrected run uses separate paths.

## Forensic cause being corrected

The failed R5B prompt asked for JSON indices but did not ask for `EXACTLY 5`. The corrected protocol also prevents Ollama payloads with `done != true` from becoming `SUCCEEDED` metric-ready records.

## Minimal protocol changes

OLD PROMPT instruction:

```text
Return JSON with key ranked_candidate_indices containing unique 1-based candidate indices.
```

CORRECTED PROMPT instruction:

```text
Use only the train-history and candidate context below. Return only valid JSON with key ranked_candidate_indices containing EXACTLY 5 unique 1-based candidate indices, ordered from rank 1 to rank 5: {"ranked_candidate_indices":[1,2,3,4,5]}. Do not return more than 5 indices. Do not return fewer than 5 indices. Use only candidate indices from the provided candidate set. Do not use item IDs in the answer. Do not include rationale, explanation, prose, Markdown, or extra keys.
```

The output namespace remains one-based candidate indices because that is the frozen parser/schema contract. Candidate sets, target items, users, FIRST/RANDOM relation, models, temperature, seed, and max token settings are unchanged.

## Protected artifact verification

Original R5B raw output path is not used. Corrected raw outputs are under `results/recovery/r5b_corrected/`.

## Frozen manifest verification

Manifest SHA256: `290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD`.

Selected corrected rows: `1200`.

## Static preflight

| check_id | check_name | status | observed | expected |
| --- | --- | --- | --- | --- |
| STATIC-001 | manifest SHA | pass | 290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD | 290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD |
| STATIC-002 | repository HEAD | pass | 98c28c5364c40e571765a3b50d4dab8affe80743 | 98c28c5364c40e571765a3b50d4dab8affe80743 |
| STATIC-003 | manifest rows | pass | 4800 | 4800 |
| STATIC-004 | selected CandDif rows | pass | 1200 | 1200 |
| STATIC-005 | FIRST/RANDOM rows | pass | {'CANDDIF_FIRST': np.int64(600), 'CANDDIF_RANDOM': np.int64(600)} | {CANDDIF_FIRST:600,CANDDIF_RANDOM:600} |
| STATIC-006 | datasets | pass | ['Beauty', 'LastFM', 'Sports'] | ['Beauty', 'LastFM', 'Sports'] |
| STATIC-007 | models | pass | ['mistral:7b-instruct', 'qwen2:7b-instruct'] | ['mistral:7b-instruct', 'qwen2:7b-instruct'] |
| STATIC-008 | no ML-1M CandDif | pass | 0 | 0 |
| STATIC-009 | ranking task only | pass | ['ranking'] | ranking |
| STATIC-010 | expected K is 5 | pass | [5] | [5] |
| STATIC-011 | candidate count 20 | pass | ['20'] | 20 |
| STATIC-012 | pair count and validity | pass | 600/600 | 600/600 |
| STATIC-013 | corrected prompt says EXACTLY 5 | pass | 1200 | 1200 |
| STATIC-014 | corrected prompt forbids more/fewer | pass | checked | all prompts |
| STATIC-015 | corrected prompt forbids prose | pass | checked | all prompts |
| STATIC-016 | parser contract corrected | pass | ['r5b_corrected_ranked_index_json_top5_v1'] | r5b_corrected_ranked_index_json_top5_v1 |
| STATIC-017 | local endpoint only | pass | http://127.0.0.1:11434/api/generate | localhost Ollama |
| STATIC-018 | format json stream false | pass | format=json;stream=False | format=json;stream=false |
| STATIC-019 | no retry policy introduced | pass | 0 | 0 |
| STATIC-020 | output namespace corrected | pass | results/recovery/r5b_corrected/ | results/recovery/r5b_corrected/ |
| STATIC-021 | no output path overlap with old R5B | pass | 0 | 0 |
| STATIC-022 | no corrected output under original namespace | pass | checked | none |
| STATIC-023 | original repo clean | pass |  | empty |
| STATIC-024 | protected artifacts unchanged | pass | 12/12 | 12/12 |
| STATIC-025 | disk free guard | pass | 162166550528 | >=1073741824 |
| STATIC-026 | protocol hash present | pass | 9596F36E1B1DBC15B68679DDF32114E6AE09914B418FB31DC112072384B708D5 | 64-char SHA256 |

## Synthetic tests

| case_id | expected_valid | observed_valid | expected_status | observed_status | pass |
| --- | --- | --- | --- | --- | --- |
| valid_json_object_indices | true | true | parsed_valid | parsed_valid | true |
| plain_json_array_candidate_ids | false | false | invalid_schema | invalid_schema | true |
| numbered_lines | false | false | parse_failed | parse_failed | true |
| wrong_key_json | false | false | invalid_schema | invalid_schema | true |
| line_separated_ids | false | false | parse_failed | parse_failed | true |
| four_ids | false | false | invalid_length | invalid_length | true |
| six_ids | false | false | invalid_length | invalid_length | true |
| full_candidate_set | false | false | invalid_length | invalid_length | true |
| duplicate | false | false | duplicate_indices | duplicate_indices | true |
| out_of_candidate_index | false | false | unknown_indices | unknown_indices | true |
| empty_output | false | false | empty_output | empty_output | true |
| truncated_output | false | false | parse_failed | parse_failed | true |

## Protocol freeze

```text
CORRECTED_PROTOCOL_PREFLIGHT_READY
PROTOCOL_HASH=9596F36E1B1DBC15B68679DDF32114E6AE09914B418FB31DC112072384B708D5
```

| component | value | sha256 |
| --- | --- | --- |
| manifest | 290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD | 290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD |
| prompt_family | r5b_corrected_canddif_json_index_exact5_v1 | CBC367079200778AF6175AEF39A65F03C6BE77C950B0D86B0C25CA203C8EBF29 |
| parser | r5b_corrected_ranked_index_json_top5_v1 | CB5DA25F85CC9B90449FC11EA436D26379C225F2F2194CDB3CF9B2306A92A364 |
| runner | C:/Users/Bartolo/Desktop/PROGETTI/IRP_BeyondUtility_Project/src/analysis/reproduction_r5b_corrected_canddif_recovery.py | A152523E8C471CFA4488E274A58B4924048486C597199520AD3135B0D15C292D |
| config | configs/reproduction_r5b_corrected_canddif_recovery.yaml | 0EF1EF47A8C8FD5B29446599A9C0C81F94D2A0D7127A333CA5D5FCA53D9B57FE |
| candidate_construction | R5A frozen candidate rows | 38E48348F29AE2E429EC30D883C9264F2D4EA72DABA65C9B55F755AEA7C32701 |
| metrics | HR@5/NDCG@5/CandDif in corrected runner | A152523E8C471CFA4488E274A58B4924048486C597199520AD3135B0D15C292D |
| generation_config | {"endpoint":"http://127.0.0.1:11434/api/generate","format":"json","max_retries":"0","num_predict_values":["512"],"seed_values":["0"],"stream":"false","temperature":"0","timeout_seconds":"120"} | E62BD8A4FC52A56CBB9CCBC5AE8FA24861835197175629B091FFE188E5964801 |
| protocol | CORRECTED_PROTOCOL_PREFLIGHT_READY | 9596F36E1B1DBC15B68679DDF32114E6AE09914B418FB31DC112072384B708D5 |
| model_identifier | mistral:7b-instruct | 6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf |
| model_identifier | qwen2:7b-instruct | dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755 |

## Canary design

The canary selects one deterministic first eligible row per `model x dataset x condition` cell, then allows at most two additional deterministic rows in failed model-compliance cells.

## Canary results

| Model | Dataset | Condition | Complete | Parsed K | Valid | Failure | request_id | r5a_request_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mistral:7b-instruct | Beauty | CANDDIF_FIRST | true | 5 | true | VALID | R5B-CORR-8EBF063B2B6DC8A6 | R5A-REQ-4EA016A34526A13C |
| mistral:7b-instruct | Beauty | CANDDIF_RANDOM | true | 5 | true | VALID | R5B-CORR-B8019E6094210922 | R5A-REQ-6D0E5FD8D3303B11 |
| qwen2:7b-instruct | Beauty | CANDDIF_FIRST | true | 5 | true | VALID | R5B-CORR-40A9AAC1B3B5BD2B | R5A-REQ-D821EF2CB333B9AF |
| qwen2:7b-instruct | Beauty | CANDDIF_RANDOM | true | 5 | true | VALID | R5B-CORR-B2413D01B74EE8EF | R5A-REQ-2C2F4A86B733A59E |
| mistral:7b-instruct | LastFM | CANDDIF_FIRST | true | 5 | true | VALID | R5B-CORR-6AAB517A001CF4E6 | R5A-REQ-A4B6923D7FC3EE3E |
| mistral:7b-instruct | LastFM | CANDDIF_RANDOM | true | 5 | true | VALID | R5B-CORR-BD2A0C729C6361BA | R5A-REQ-0AA88B47C4B5D295 |
| qwen2:7b-instruct | LastFM | CANDDIF_FIRST | true | 5 | true | VALID | R5B-CORR-916F720065748AEA | R5A-REQ-5B87F75D8D5F7D40 |
| qwen2:7b-instruct | LastFM | CANDDIF_RANDOM | true | 5 | true | VALID | R5B-CORR-849D06AC018DF537 | R5A-REQ-CC4B95959A9AA11A |
| mistral:7b-instruct | Sports | CANDDIF_FIRST | true | 5 | true | VALID | R5B-CORR-F0556B3940FAE434 | R5A-REQ-416FD71C587644DD |
| mistral:7b-instruct | Sports | CANDDIF_RANDOM | true | 5 | true | VALID | R5B-CORR-B4D610F634496AB6 | R5A-REQ-00A6D1E3E1E426C8 |
| qwen2:7b-instruct | Sports | CANDDIF_FIRST | true | 5 | true | VALID | R5B-CORR-77F18B2299FC50E9 | R5A-REQ-3B7261A69136CBBC |
| qwen2:7b-instruct | Sports | CANDDIF_RANDOM | true | 5 | true | VALID | R5B-CORR-2E33E73162B1940C | R5A-REQ-E0F1F449BD5AFCCE |

## Early health check

| scope | key | accounted | valid | invalid | validity_rate | wrong_length | parse_failure | backend_incomplete |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | all | 36 | 36 | 0 | 1.0 | 0 | 0 | 0 |
| model | mistral:7b-instruct | 18 | 18 | 0 | 1.0 | 0 | 0 | 0 |
| model | qwen2:7b-instruct | 18 | 18 | 0 | 1.0 | 0 | 0 | 0 |
| dataset | Beauty | 12 | 12 | 0 | 1.0 | 0 | 0 | 0 |
| dataset | LastFM | 12 | 12 | 0 | 1.0 | 0 | 0 | 0 |
| dataset | Sports | 12 | 12 | 0 | 1.0 | 0 | 0 | 0 |
| condition | CANDDIF_FIRST | 18 | 18 | 0 | 1.0 | 0 | 0 | 0 |
| condition | CANDDIF_RANDOM | 18 | 18 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | mistral:7b-instruct|Beauty|CANDDIF_FIRST | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | mistral:7b-instruct|Beauty|CANDDIF_RANDOM | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | mistral:7b-instruct|LastFM|CANDDIF_FIRST | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | mistral:7b-instruct|LastFM|CANDDIF_RANDOM | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | mistral:7b-instruct|Sports|CANDDIF_FIRST | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | mistral:7b-instruct|Sports|CANDDIF_RANDOM | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | qwen2:7b-instruct|Beauty|CANDDIF_FIRST | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | qwen2:7b-instruct|Beauty|CANDDIF_RANDOM | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | qwen2:7b-instruct|LastFM|CANDDIF_FIRST | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | qwen2:7b-instruct|LastFM|CANDDIF_RANDOM | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | qwen2:7b-instruct|Sports|CANDDIF_FIRST | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |
| model_dataset_condition | qwen2:7b-instruct|Sports|CANDDIF_RANDOM | 3 | 3 | 0 | 1.0 | 0 | 0 | 0 |

## Full-run authorization decision

FULL RUN AUTHORIZED: `YES`.

Blocked reason: ``.

## Full execution

- Requests expected: `1200`
- Requests accounted: `1200`
- Executed this invocation: `1164`
- Valid: `1200`
- Invalid: `0`
- Wrong length: `0`
- Parse failure: `0`
- Backend incomplete: `0`

## Output validity

| dataset | model_name | n_manifest_requests | n_terminal | n_valid_outputs | n_invalid_outputs | n_wrong_length | n_parse_failure | n_backend_incomplete |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Beauty | mistral:7b-instruct | 200 | 200 | 200 | 0 | 0 | 0 | 0 |
| Beauty | qwen2:7b-instruct | 200 | 200 | 200 | 0 | 0 | 0 | 0 |
| LastFM | mistral:7b-instruct | 200 | 200 | 200 | 0 | 0 | 0 | 0 |
| LastFM | qwen2:7b-instruct | 200 | 200 | 200 | 0 | 0 | 0 | 0 |
| Sports | mistral:7b-instruct | 200 | 200 | 200 | 0 | 0 | 0 | 0 |
| Sports | qwen2:7b-instruct | 200 | 200 | 200 | 0 | 0 | 0 | 0 |

## FIRST vs RANDOM metrics

| dataset | model_name | experimental_condition | n_manifest | n_valid | n_invalid | HR@5 | NDCG@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Beauty | mistral:7b-instruct | CANDDIF_FIRST | 100 | 100 | 0 | 0.26 | 0.22422417928846541 |
| Beauty | mistral:7b-instruct | CANDDIF_RANDOM | 100 | 100 | 0 | 0.34 | 0.1854617137594331 |
| Beauty | qwen2:7b-instruct | CANDDIF_FIRST | 100 | 100 | 0 | 0.49 | 0.41543281576755725 |
| Beauty | qwen2:7b-instruct | CANDDIF_RANDOM | 100 | 100 | 0 | 0.26 | 0.14462653526448047 |
| LastFM | mistral:7b-instruct | CANDDIF_FIRST | 100 | 100 | 0 | 0.59 | 0.5277551241576647 |
| LastFM | mistral:7b-instruct | CANDDIF_RANDOM | 100 | 100 | 0 | 0.5 | 0.32688459337162823 |
| LastFM | qwen2:7b-instruct | CANDDIF_FIRST | 100 | 100 | 0 | 0.81 | 0.6490598039415529 |
| LastFM | qwen2:7b-instruct | CANDDIF_RANDOM | 100 | 100 | 0 | 0.43 | 0.297008201438006 |
| Sports | mistral:7b-instruct | CANDDIF_FIRST | 100 | 100 | 0 | 0.42 | 0.38239947294154475 |
| Sports | mistral:7b-instruct | CANDDIF_RANDOM | 100 | 100 | 0 | 0.35 | 0.2127786071600896 |
| Sports | qwen2:7b-instruct | CANDDIF_FIRST | 100 | 100 | 0 | 0.49 | 0.44495870347817557 |
| Sports | qwen2:7b-instruct | CANDDIF_RANDOM | 100 | 100 | 0 | 0.33 | 0.2130209442960639 |

## CandDif results

| dataset | model_name | n_users | HR_first | HR_random | CandDif_HR | NDCG_first | NDCG_random | CandDif_NDCG |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Beauty | mistral:7b-instruct | 100 | 0.26 | 0.34 | -0.11441035117774434 | 0.22422417928846541 | 0.1854617137594331 | 0.04875784514700657 |
| Beauty | qwen2:7b-instruct | 100 | 0.49 | 0.26 | 0.37223946047984396 | 0.41543281576755725 | 0.14462653526448047 | 0.3806664570368846 |
| LastFM | mistral:7b-instruct | 100 | 0.59 | 0.5 | 0.19845093872383823 | 0.5277551241576647 | 0.32688459337162823 | 0.3544191400142057 |
| LastFM | qwen2:7b-instruct | 100 | 0.81 | 0.43 | 1.0986122886681102 | 0.6490598039415529 | 0.297008201438006 | 0.694729398051804 |
| Sports | mistral:7b-instruct | 100 | 0.42 | 0.35 | 0.11394425934921769 | 0.38239947294154475 | 0.2127786071600896 | 0.24266766912453377 |
| Sports | qwen2:7b-instruct | 100 | 0.49 | 0.33 | 0.2728669866666401 | 0.44495870347817557 | 0.2130209442960639 | 0.3491591161090149 |

## Original R5B vs Corrected R5B

| quantity | original_r5b | corrected_r5b |
| --- | --- | --- |
| total | 1200 | 1200 |
| accounted | 1200 | 1200 |
| valid | 52 | 1200 |
| invalid | 1148 | 0 |
| wrong_length | 1132 | 0 |
| parse_failure | 16 | 0 |
| backend_incomplete | 16 | 0 |
| mistral_valid | 50 | 600 |
| qwen_valid | 2 | 600 |

## Official comparison

| dataset | model_name | metric | corrected_local_adapted_canddif | paper_exact | comparison_class |
| --- | --- | --- | --- | --- | --- |
| Beauty | mistral:7b-instruct | CandDif-HR | -0.11441035117774434 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | mistral:7b-instruct | CandDif-NDCG | 0.04875784514700657 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | qwen2:7b-instruct | CandDif-HR | 0.37223946047984396 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Beauty | qwen2:7b-instruct | CandDif-NDCG | 0.3806664570368846 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | mistral:7b-instruct | CandDif-HR | 0.19845093872383823 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | mistral:7b-instruct | CandDif-NDCG | 0.3544191400142057 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | qwen2:7b-instruct | CandDif-HR | 1.0986122886681102 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| LastFM | qwen2:7b-instruct | CandDif-NDCG | 0.694729398051804 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | mistral:7b-instruct | CandDif-HR | 0.11394425934921769 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | mistral:7b-instruct | CandDif-NDCG | 0.24266766912453377 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | qwen2:7b-instruct | CandDif-HR | 0.2728669866666401 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |
| Sports | qwen2:7b-instruct | CandDif-NDCG | 0.3491591161090149 | false | LOCAL_ADAPTED_NOT_PAPER_EXACT |

## Scientific interpretation

The official end-to-end metrics retain the fail-strict denominator: every frozen request remains in the denominator and invalid outputs score zero. Conditional-valid metrics are diagnostic only.

## Limitations

This is still a local adapted recovery, not a paper-exact rerun. Model identities, output namespace, and local Ollama execution differ from the original closed/open model setup in the paper repository.

## Validation

| check_id | check_name | status | observed | expected |
| --- | --- | --- | --- | --- |
| VAL-001 | static preflight | pass | 26/26 | all pass |
| VAL-002 | synthetic tests | pass | 12/12 | all pass |
| VAL-003 | canary complete when attempted | pass | 12/12 | all complete |
| VAL-004 | canary parsed K=5 when attempted | pass | 12/12 | all 5 |
| VAL-005 | early health no backend incomplete | pass | checked | 0 |
| VAL-006 | pair validation | pass | 600/600 | 600/600 |
| VAL-007 | incomplete not succeeded | pass | checked | none |
| VAL-008 | full run authorization consistent | pass | authorized=True;decision=R5B_CORRECTED_RECOVERY_COMPLETE;terminal=1200 | consistent |

## Repository state

No commit and no push were performed by this runner.

## Final decision

`R5B_CORRECTED_RECOVERY_COMPLETE`
