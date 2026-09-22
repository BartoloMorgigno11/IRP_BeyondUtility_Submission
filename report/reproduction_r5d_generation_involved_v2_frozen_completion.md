# R5D Frozen-Protocol Completion

Decision: `R5D_FROZEN_PROTOCOL_COMPLETION_VALID`

Protocol hash: `AA76CD3E936A969A57CDA2E9E54E22B73D6C1D3DB65C0795F86E181CEDFE3B1A`

## Accounting

- Requests expected: `1200`
- Requests accounted: `1200`
- Valid: `1195`
- Invalid: `5`
- Invalid policy: `FAIL_STRICT_ALL_MANIFEST_REQUESTS_INVALID_ZERO`
- Executed in frozen resume: `600`
- Retried rows: `0`

## Invalid Outputs

```json
[
  {
    "model_name": "mistral:7b-instruct",
    "dataset": "Beauty",
    "experimental_condition": "PROFILE_PLUS_HISTORY",
    "validity_class": "DUPLICATE",
    "n": 1,
    "request_ids": [
      "R5D-RANK-59C06DDCFD78BFF4"
    ]
  },
  {
    "model_name": "mistral:7b-instruct",
    "dataset": "Beauty",
    "experimental_condition": "PROFILE_PLUS_HISTORY",
    "validity_class": "OUT_OF_RANGE",
    "n": 2,
    "request_ids": [
      "R5D-RANK-167AC5F3AB797DBA",
      "R5D-RANK-E491A3315449174D"
    ]
  },
  {
    "model_name": "mistral:7b-instruct",
    "dataset": "Sports",
    "experimental_condition": "PROFILE_PLUS_HISTORY",
    "validity_class": "WRONG_LENGTH",
    "n": 1,
    "request_ids": [
      "R5D-RANK-FA218507694B581F"
    ]
  },
  {
    "model_name": "qwen2:7b-instruct",
    "dataset": "Sports",
    "experimental_condition": "PROFILE_PLUS_HISTORY",
    "validity_class": "DUPLICATE",
    "n": 1,
    "request_ids": [
      "R5D-RANK-94B4099CCA078776"
    ]
  }
]
```

## Target Gaps

- Accounted under invalid-output policy: `4/4`
- All-valid diagnostic: `2/4`

| gap_id | dataset | requests_expected | requests_accounted | valid | invalid | gap_recovered |
| --- | --- | --- | --- | --- | --- | --- |
| R4-GAP-0005 | Beauty | 200 | 200 | 197 | 3 | false |
| R4-GAP-0019 | LastFM | 400 | 400 | 400 | 0 | true |
| R4-GAP-0031 | ML-1M | 400 | 400 | 400 | 0 | true |
| R4-GAP-0045 | Sports | 200 | 200 | 198 | 2 | false |

## Metrics

| dataset | model_name | experimental_condition | n_manifest | n_valid | n_invalid | HR@5 | NDCG@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Beauty | mistral:7b-instruct | PROFILE_PLUS_HISTORY | 100 | 97 | 3 | 0.33 | 0.2130390190604181 |
| Beauty | qwen2:7b-instruct | PROFILE_PLUS_HISTORY | 100 | 100 | 0 | 0.29 | 0.21055695982660486 |
| LastFM | mistral:7b-instruct | PROFILE_PLUS_HISTORY | 100 | 100 | 0 | 0.57 | 0.3580338653159218 |
| LastFM | qwen2:7b-instruct | PROFILE_PLUS_HISTORY | 100 | 100 | 0 | 0.25 | 0.1588148397719527 |
| ML-1M | mistral:7b-instruct | PROFILE_PLUS_HISTORY | 100 | 100 | 0 | 0.43 | 0.2729313812146781 |
| ML-1M | qwen2:7b-instruct | PROFILE_PLUS_HISTORY | 100 | 100 | 0 | 0.4 | 0.2836137924064225 |
| Sports | mistral:7b-instruct | PROFILE_PLUS_HISTORY | 100 | 99 | 1 | 0.24 | 0.1514927993781802 |
| Sports | qwen2:7b-instruct | PROFILE_PLUS_HISTORY | 100 | 99 | 1 | 0.25 | 0.14516542707811142 |

## Resume Preflight

| check_id | check_name | status | observed | expected |
| --- | --- | --- | --- | --- |
| RESUME-001 | manifest row count unchanged | pass | 1200 | 1200 |
| RESUME-002 | r5a manifest sha unchanged | pass | 290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD | 290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD |
| RESUME-003 | stored frozen protocol hash unchanged | pass | AA76CD3E936A969A57CDA2E9E54E22B73D6C1D3DB65C0795F86E181CEDFE3B1A | AA76CD3E936A969A57CDA2E9E54E22B73D6C1D3DB65C0795F86E181CEDFE3B1A |
| RESUME-004 | computed runtime protocol hash unchanged | pass | AA76CD3E936A969A57CDA2E9E54E22B73D6C1D3DB65C0795F86E181CEDFE3B1A | AA76CD3E936A969A57CDA2E9E54E22B73D6C1D3DB65C0795F86E181CEDFE3B1A |
| RESUME-005 | runtime models available with frozen identifiers | pass | available | available |
| RESUME-006 | completed terminal rows before resume | pass | 600 | 600 |
| RESUME-007 | remaining rows before resume | pass | 600 | 600 |
| RESUME-008 | terminal retry count zero | pass | 0 | 0 |
| RESUME-009 | original invalids terminal and unchanged | pass | R5D-RANK-167AC5F3AB797DBA;R5D-RANK-59C06DDCFD78BFF4;R5D-RANK-E491A3315449174D | R5D-RANK-167AC5F3AB797DBA;R5D-RANK-59C06DDCFD78BFF4;R5D-RANK-E491A3315449174D |
| RESUME-010 | original invalids excluded from missing set | pass | 0 | 0 |
| RESUME-011 | missing output files absent | pass | 0 | 0 |
| RESUME-012 | terminal output files present | pass | 600 | 600 |
| RESUME-013 | request ids and output destinations unique | pass | requests_unique=True;outputs_unique=True | true;true |
| RESUME-014 | missing rows are downstream ranking rows | pass | ranking | ranking |
| RESUME-015 | profile-generation dependencies complete | pass | profiles_valid=400/400;missing_dependencies_ok=True | 400/400;true |
| RESUME-016 | invalid-output policy is manifest-denominator zero credit | pass | FAIL_STRICT_ALL_MANIFEST_REQUESTS_INVALID_ZERO | FAIL_STRICT_ALL_MANIFEST_REQUESTS_INVALID_ZERO |
| RESUME-017 | max retries frozen to zero | pass | 0 | 0 |

## Checkpoints

| checkpoint | requests_accounted | valid | invalid | missing | new_rows_terminal | new_rows_valid | new_rows_invalid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| before_resume | 600 | 597 | 3 | 600 | 0 | 0 | 0 |
| after_batch_001 | 650 | 647 | 3 | 550 | 50 | 50 | 0 |
| after_batch_002 | 700 | 697 | 3 | 500 | 100 | 100 | 0 |
| after_batch_003 | 750 | 747 | 3 | 450 | 150 | 150 | 0 |
| after_batch_004 | 800 | 797 | 3 | 400 | 200 | 200 | 0 |
| after_batch_005 | 850 | 847 | 3 | 350 | 250 | 250 | 0 |
| after_batch_006 | 900 | 897 | 3 | 300 | 300 | 300 | 0 |
| after_batch_007 | 950 | 947 | 3 | 250 | 350 | 350 | 0 |
| after_batch_008 | 1000 | 997 | 3 | 200 | 400 | 400 | 0 |
| after_batch_009 | 1050 | 1046 | 4 | 150 | 450 | 449 | 1 |
| after_batch_010 | 1100 | 1096 | 4 | 100 | 500 | 499 | 1 |
| after_batch_011 | 1150 | 1145 | 5 | 50 | 550 | 548 | 2 |
| after_batch_012 | 1200 | 1195 | 5 | 0 | 600 | 598 | 2 |

## Validation

| check_id | check_name | status | observed | expected |
| --- | --- | --- | --- | --- |
| RESVAL-001 | offline resume preflight pass | pass | 17/17 | 17/17 |
| RESVAL-002 | all manifest rows accounted | pass | 1200/1200 | 1200/1200 |
| RESVAL-003 | one raw per manifest row | pass | 1200 | 1200 |
| RESVAL-004 | no retries | pass | 0 | 0 |
| RESVAL-005 | original invalids retained | pass | checked | retained |
| RESVAL-006 | invalids allowed by denominator policy | pass | FAIL_STRICT_ALL_MANIFEST_REQUESTS_INVALID_ZERO | FAIL_STRICT_ALL_MANIFEST_REQUESTS_INVALID_ZERO |
| RESVAL-007 | target gaps accounted | pass | 4/4 | 4/4 |
| RESVAL-008 | decision matches frozen completion | pass | R5D_FROZEN_PROTOCOL_COMPLETION_VALID | R5D_FROZEN_PROTOCOL_COMPLETION_VALID |
| RESVAL-009 | validity is not forced to all-valid | pass | 1195/1200 | invalids terminal |

## Interpretation

The three known invalid R5D outputs were not retried or repaired. They remain
terminal manifest rows and are counted in the denominator with zero HR/NDCG
credit under `FAIL_STRICT_ALL_MANIFEST_REQUESTS_INVALID_ZERO`. This completion therefore freezes a
one-attempt-per-row R5D run, not an all-valid run.
