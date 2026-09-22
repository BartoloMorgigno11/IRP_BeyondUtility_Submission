# R5D Generation-Involved Recovery

Decision: `R5D_CANARY_PASS`

R5D executes only frozen R5A `GENERATION_INVOLVED_RECOVERY` rows. It preserves R5A/R5B/R5C raw outputs and writes a separate R5D namespace.

## Scope

| scope | key | count | notes |
| --- | --- | --- | --- |
| purpose | R5D | 1200 | Generation-involved profile recommendation recovery from frozen R5A manifest. |
| dataset | Beauty | 200 |  |
| dataset | LastFM | 400 |  |
| dataset | ML-1M | 400 |  |
| dataset | Sports | 200 |  |
| model | mistral:7b-instruct | 600 |  |
| model | qwen2:7b-instruct | 600 |  |
| condition | PROFILE_GENERATION | 400 |  |
| condition | PROFILE_PLUS_HISTORY | 800 |  |
| distinct_cell | profile_generation|LastFM|mistral:7b-instruct|PROFILE_GENERATION | 100 |  |
| distinct_cell | profile_generation|LastFM|qwen2:7b-instruct|PROFILE_GENERATION | 100 |  |
| distinct_cell | profile_generation|ML-1M|mistral:7b-instruct|PROFILE_GENERATION | 100 |  |
| distinct_cell | profile_generation|ML-1M|qwen2:7b-instruct|PROFILE_GENERATION | 100 |  |
| distinct_cell | ranking|Beauty|mistral:7b-instruct|PROFILE_PLUS_HISTORY | 100 |  |
| distinct_cell | ranking|Beauty|qwen2:7b-instruct|PROFILE_PLUS_HISTORY | 100 |  |
| distinct_cell | ranking|LastFM|mistral:7b-instruct|PROFILE_PLUS_HISTORY | 100 |  |
| distinct_cell | ranking|LastFM|qwen2:7b-instruct|PROFILE_PLUS_HISTORY | 100 |  |
| distinct_cell | ranking|ML-1M|mistral:7b-instruct|PROFILE_PLUS_HISTORY | 100 |  |
| distinct_cell | ranking|ML-1M|qwen2:7b-instruct|PROFILE_PLUS_HISTORY | 100 |  |
| distinct_cell | ranking|Sports|mistral:7b-instruct|PROFILE_PLUS_HISTORY | 100 |  |
| distinct_cell | ranking|Sports|qwen2:7b-instruct|PROFILE_PLUS_HISTORY | 100 |  |

## Protocol

- Manifest SHA256: `290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD`
- Protocol hash: `D55997795A10C0FFE83A9539D20A28CC21A8475A031262C893B0E7B43DA42C72`
- Profile parser: `r5d_profile_json_profile_parser_v1`
- Ranking parser: `r5d_ranked_index_json_top10_v1`
- Max retries: `0`
- Paper exact: `NO`

| component | value | sha256 |
| --- | --- | --- |
| manifest | 290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD | 290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD |
| prompt_templates | r5d_profile_generation_json_profile_v1;r5d_profile_plus_history_json_index_exact10_v1 | 815A182931722F63BF43901755F5DF36E19750FA3DDF6D625177C11CA03F7383 |
| parser | r5d_profile_json_profile_parser_v1;r5d_ranked_index_json_top10_v1 | CB5DA25F85CC9B90449FC11EA436D26379C225F2F2194CDB3CF9B2306A92A364 |
| runner | C:/Users/Bartolo/Desktop/PROGETTI/IRP_BeyondUtility_Project/src/analysis/reproduction_r5d_generation_involved_recovery.py | 9CCD4DB02546EB8D6086AC0B63256E532E5304AFA37760E436E5C31415A5997E |
| config | configs/reproduction_r5d_generation_involved_recovery.yaml | 7E6B52F884AFF06976B731543EB0194BFF363FE933C0FA1AEF2A4632F9A75482 |
| manifest_builder | C:/Users/Bartolo/Desktop/PROGETTI/IRP_BeyondUtility_Project/src/analysis/reproduction_r5a_controlled_recovery_manifest.py | 38E48348F29AE2E429EC30D883C9264F2D4EA72DABA65C9B55F755AEA7C32701 |
| dependency_validation | profile-generation dependency graph | 8DBB215B30865AD045317AF7728045BE1CDF15334B7226E09F39EED2F1E210BC |
| metrics | HR@5/NDCG@5 over profile-plus-history ranking rows | 9CCD4DB02546EB8D6086AC0B63256E532E5304AFA37760E436E5C31415A5997E |
| generation_config | {"endpoint":"http://127.0.0.1:11434/api/generate","format":"json","max_retries":"0","num_predict_values":["512"],"seed_values":["0"],"stream":"false","temperature":"0","timeout_seconds":"120"} | E62BD8A4FC52A56CBB9CCBC5AE8FA24861835197175629B091FFE188E5964801 |
| protocol | R5D_PROTOCOL_PREFLIGHT_READY | D55997795A10C0FFE83A9539D20A28CC21A8475A031262C893B0E7B43DA42C72 |
| model_identifier | mistral:7b-instruct | 6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf |
| model_identifier | qwen2:7b-instruct | dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755 |

## Static Preflight

| check_id | check_name | status | observed | expected |
| --- | --- | --- | --- | --- |
| STATIC-001 | manifest SHA | pass | 290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD | 290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD |
| STATIC-002 | repository HEAD is R5C corrected freeze | pass | 98a93d9892d8e9b21c9cdfa981e0ebea0a987500 | 98a93d9892d8e9b21c9cdfa981e0ebea0a987500 |
| STATIC-003 | R5C corrected freeze manifest exists | pass | C:/Users/Bartolo/Desktop/PROGETTI/IRP_BeyondUtility_Project/results/manifests/reproduction_r5c_history_length_corrected_freeze_candidate.json | exists |
| STATIC-004 | R5C corrected protocol hash | pass | 3AAD4945AF443A7F50846BFD044C54FEFD5B77EDBD3725B1D1F1C62D829ECDEE | 3AAD4945AF443A7F50846BFD044C54FEFD5B77EDBD3725B1D1F1C62D829ECDEE |
| STATIC-005 | R5C corrected freeze ready | pass | true | true |
| STATIC-006 | manifest rows | pass | 4800 | 4800 |
| STATIC-007 | selected generation rows | pass | 1200 | 1200 |
| STATIC-008 | out-of-scope rows remain unselected | pass | 3600 | 3600 |
| STATIC-009 | profile-generation rows | pass | 400 | 400 |
| STATIC-010 | downstream ranking rows | pass | 800 | 800 |
| STATIC-011 | datasets | pass | ['Beauty', 'LastFM', 'ML-1M', 'Sports'] | ['Beauty', 'LastFM', 'ML-1M', 'Sports'] |
| STATIC-012 | models | pass | ['mistral:7b-instruct', 'qwen2:7b-instruct'] | ['mistral:7b-instruct', 'qwen2:7b-instruct'] |
| STATIC-013 | distinct cells | pass | 12 cells; min=100; max=100 | 12 cells of 100 |
| STATIC-014 | profile datasets | pass | ['LastFM', 'ML-1M'] | ['LastFM', 'ML-1M'] |
| STATIC-015 | ranking datasets | pass | ['Beauty', 'LastFM', 'ML-1M', 'Sports'] | ['Beauty', 'LastFM', 'ML-1M', 'Sports'] |
| STATIC-016 | ranking expected K is 10 | pass | [10] | [10] |
| STATIC-017 | ranking candidate count 20 | pass | ['20'] | 20 |
| STATIC-018 | profile candidate count 0 | pass | ['0'] | 0 |
| STATIC-019 | profile prompt strict JSON | pass | 400 | 400 |
| STATIC-020 | ranking prompt says EXACTLY 10 | pass | 800 | 800 |
| STATIC-021 | ranking prompt states positions 1..20 | pass | 800 | 800 |
| STATIC-022 | ranking prompt rejects item IDs | pass | checked | all ranking prompts |
| STATIC-023 | parser contracts | pass | ['r5d_profile_json_profile_parser_v1', 'r5d_ranked_index_json_top10_v1'] | r5d_profile_json_profile_parser_v1;r5d_ranked_index_json_top10_v1 |
| STATIC-024 | dependency graph valid | pass | 800/800 | 800/800 |
| STATIC-025 | output namespace R5D | pass | results/recovery/r5d_generation_involved/ | results/recovery/r5d_generation_involved/ |
| STATIC-026 | no output path overlap with R5A destinations | pass | 0 | 0 |
| STATIC-027 | no protected namespace overlap | pass | checked | none |
| STATIC-028 | local endpoint only | pass | http://127.0.0.1:11434/api/generate | localhost Ollama |
| STATIC-029 | format json stream false | pass | format=json;stream=False | format=json;stream=false |
| STATIC-030 | max_retries remains zero | pass | 0 | 0 |
| STATIC-031 | context envelope | pass | 12/12 | all pass |
| STATIC-032 | disk free guard | pass | 157828542464 | >=1073741824 |
| STATIC-033 | original repo clean | pass |  | empty |
| STATIC-034 | protected artifacts unchanged | pass | 12/12 | 12/12 |
| STATIC-035 | protocol hash present | pass | D55997795A10C0FFE83A9539D20A28CC21A8475A031262C893B0E7B43DA42C72 | 64-char SHA256 |

## Synthetic Tests

| case_id | task | expected_status | observed_status | pass | notes |
| --- | --- | --- | --- | --- | --- |
| ranking_valid_exact10 | ranking | parsed_valid | parsed_valid | true |  |
| ranking_k_minus_1 | ranking | invalid_length | invalid_length | true |  |
| ranking_k_plus_1 | ranking | invalid_length | invalid_length | true |  |
| ranking_numbered_output | ranking | parse_failed | parse_failed | true |  |
| ranking_duplicate | ranking | duplicate_indices | duplicate_indices | true |  |
| ranking_out_of_range | ranking | unknown_indices | unknown_indices | true |  |
| ranking_item_ids_in_indices_field | ranking | unknown_indices | unknown_indices | true | actual dataset item IDs must not be accepted as candidate positions |
| ranking_truncated_output | ranking | parse_failed | parse_failed | true |  |
| ranking_backend_incomplete | ranking | backend_incomplete | backend_incomplete | true |  |
| profile_valid | profile_generation | parsed_valid | parsed_valid | true |  |
| profile_empty | profile_generation | empty_profile | empty_profile | true |  |
| profile_extra_key | profile_generation | invalid_schema | invalid_schema | true |  |
| profile_plain_text | profile_generation | parse_failed | parse_failed | true |  |
| profile_backend_incomplete | profile_generation | backend_incomplete | backend_incomplete | true |  |

## Canary

| Task | Model | Dataset | Condition | Complete | ParsedLength | ExpectedLength | Valid | Failure | Within1ToN | ItemIdSubstitutionCount |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ranking | mistral:7b-instruct | Beauty | PROFILE_PLUS_HISTORY | true | 10 | 10 | true | VALID | true | 0 |
| ranking | qwen2:7b-instruct | Beauty | PROFILE_PLUS_HISTORY | true | 10 | 10 | true | VALID | true | 0 |
| profile_generation | mistral:7b-instruct | LastFM | PROFILE_GENERATION | true | 1 | 1 | true | VALID | not_applicable | 0 |
| ranking | mistral:7b-instruct | LastFM | PROFILE_PLUS_HISTORY | true | 10 | 10 | true | VALID | true | 0 |
| profile_generation | qwen2:7b-instruct | LastFM | PROFILE_GENERATION | true | 1 | 1 | true | VALID | not_applicable | 0 |
| ranking | qwen2:7b-instruct | LastFM | PROFILE_PLUS_HISTORY | true | 10 | 10 | true | VALID | true | 0 |
| profile_generation | mistral:7b-instruct | ML-1M | PROFILE_GENERATION | true | 1 | 1 | true | VALID | not_applicable | 0 |
| ranking | mistral:7b-instruct | ML-1M | PROFILE_PLUS_HISTORY | true | 10 | 10 | true | VALID | true | 0 |
| profile_generation | qwen2:7b-instruct | ML-1M | PROFILE_GENERATION | true | 1 | 1 | true | VALID | not_applicable | 0 |
| ranking | qwen2:7b-instruct | ML-1M | PROFILE_PLUS_HISTORY | true | 10 | 10 | true | VALID | true | 0 |
| ranking | mistral:7b-instruct | Sports | PROFILE_PLUS_HISTORY | true | 10 | 10 | true | VALID | true | 0 |
| ranking | qwen2:7b-instruct | Sports | PROFILE_PLUS_HISTORY | true | 10 | 10 | true | VALID | true | 0 |

## Early Health

_No rows._

## Request Accounting

- Requests expected: `1200`
- Requests accounted: `12`
- Valid: `12`
- Invalid: `0`
- Executed this invocation: `12`
- Full run authorized: `NO`
- Blocked reason: `canary pass; early/full run not requested in this invocation`

## Metrics

| dataset | model_name | experimental_condition | n_manifest | n_valid | n_invalid | HR@5 | NDCG@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Beauty | mistral:7b-instruct | PROFILE_PLUS_HISTORY | 100 | 1 | 0 | 0.0 | 0.0 |
| Beauty | qwen2:7b-instruct | PROFILE_PLUS_HISTORY | 100 | 1 | 0 | 0.0 | 0.0 |
| LastFM | mistral:7b-instruct | PROFILE_PLUS_HISTORY | 100 | 1 | 0 | 0.01 | 0.006309297535714575 |
| LastFM | qwen2:7b-instruct | PROFILE_PLUS_HISTORY | 100 | 1 | 0 | 0.0 | 0.0 |
| ML-1M | mistral:7b-instruct | PROFILE_PLUS_HISTORY | 100 | 1 | 0 | 0.01 | 0.00430676558073393 |
| ML-1M | qwen2:7b-instruct | PROFILE_PLUS_HISTORY | 100 | 1 | 0 | 0.01 | 0.005 |
| Sports | mistral:7b-instruct | PROFILE_PLUS_HISTORY | 100 | 1 | 0 | 0.0 | 0.0 |
| Sports | qwen2:7b-instruct | PROFILE_PLUS_HISTORY | 100 | 1 | 0 | 0.0 | 0.0 |

## Gap Resolution

Target gaps recovered: `0/4`

| gap_id | dataset | requests_expected | requests_accounted | valid | invalid | gap_recovered |
| --- | --- | --- | --- | --- | --- | --- |
| R4-GAP-0005 | Beauty | 200 | 2 | 2 | 0 | false |
| R4-GAP-0019 | LastFM | 400 | 4 | 4 | 0 | false |
| R4-GAP-0031 | ML-1M | 400 | 4 | 4 | 0 | false |
| R4-GAP-0045 | Sports | 200 | 2 | 2 | 0 | false |

## Validation

| check_id | check_name | status | observed | expected |
| --- | --- | --- | --- | --- |
| VAL-001 | static preflight pass | pass | 35/35 | 35/35 |
| VAL-002 | synthetic pass | pass | 14/14 | 14/14 |
| VAL-003 | context preflight pass | pass | 12/12 | 12/12 |
| VAL-004 | dependency graph pass | pass | 800/800 | 800/800 |
| VAL-005 | canary table valid when present | pass | checked | all true |
| VAL-006 | early health pass when present | pass | checked | pass |
| VAL-007 | max retries zero | pass | checked | 0 |
| VAL-008 | incomplete is not succeeded | pass | checked | no incomplete succeeded |
| VAL-009 | ranking exactly K when valid | pass | checked | 10 |
| VAL-010 | profile JSON valid when valid | pass | checked | 1 |
| VAL-011 | full authorization consistent | pass | authorized=False;decision=R5D_CANARY_PASS | consistent |
| VAL-012 | complete means all valid | pass | terminal=12;valid=12 | 1200/1200 |

## Scientific Interpretation

R5D is local adapted generation-involved recovery. It is not paper-exact because local open models, strict JSON contracts, and explicit same-run profile materialization differ from paper execution conditions.
