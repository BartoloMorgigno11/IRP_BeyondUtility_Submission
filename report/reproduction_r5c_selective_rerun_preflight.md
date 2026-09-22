# R5C Selective Rerun Hard Preflight

## Decision

`R5C_SELECTIVE_RERUN_NOT_ALLOWED_UNDER_FROZEN_PROTOCOL`

Selective rerun under the same frozen R5C protocol hash is not scientifically allowed. The R5C protocol hash includes `max_retries=0`, the frozen report records `STATIC-025 no retry policy introduced`, and the runner preserves existing terminal outputs rather than replacing invalid terminal attempts.

## Stop Condition

- Selective rerun scientifically allowed: `NO`
- LLM calls: `0`
- Canary executed: `NO`
- Selective rows executed: `0/15`

## Key Reason

The 15 rows were selected after observing invalid outputs. Replacing only those rows would create an invalid-only retry policy that was not predeclared in R5C and is not represented by protocol hash `30E4BCA9E1456EE9A7DE74A09FE03BF1A26EFCF758915052FD88F39AF9A836F2`. The original repository contains a selective retry loop for non-parseable title-list format failures, but R5C deliberately froze a different strict JSON candidate-index contract with zero retries.

## Checks

| check_id | evidence_area | status | blocks_selective_rerun | observed | expected | finding |
| --- | --- | --- | --- | --- | --- | --- |
| PREFLIGHT-001 | scope | pass | false | 15 rows | 15 rows | The affected manifest rows are precisely listed for offline review. |
| PREFLIGHT-002 | protocol_hash | pass | false | 30E4BCA9E1456EE9A7DE74A09FE03BF1A26EFCF758915052FD88F39AF9A836F2 | 30E4BCA9E1456EE9A7DE74A09FE03BF1A26EFCF758915052FD88F39AF9A836F2 | The frozen completed R5C attempt set is bound to one protocol hash. |
| PREFLIGHT-003 | generation_config | pass | true | 0 | 0 | R5C generation config explicitly freezes no retry policy. |
| PREFLIGHT-004 | frozen_report | pass | true | found | STATIC-025 no retry policy introduced | The R5C static gate treated absence of retry policy as part of the protocol. |
| PREFLIGHT-005 | runner | pass | true | validate_existing_result_on_existing_destination | terminal outputs are validated, not replaced | The R5C runner preserves terminal outputs and does not define invalid-only replacement semantics. |
| PREFLIGHT-006 | retry_accounting | pass | true | 0 | 0 | All terminal R5C outputs are first attempts under the frozen attempt accounting. |
| PREFLIGHT-007 | invalids | pass | false | 15 invalid; datasets=['ML-1M']; parse_status=['unknown_indices'] | 15 Qwen2 ML-1M unknown_indices rows | The proposed rows match the observed R5C structural-invalid failure set. |
| PREFLIGHT-008 | forensic_audit | pass | false | False | False | The prior audit disallowed repairing existing raw outputs by item-ID normalization. |
| PREFLIGHT-009 | forensic_audit | pass | false | found | normalization disallowed | The previous future-rerun recommendation did not itself authorize replacement under the same hash. |
| PREFLIGHT-010 | parser_protocol | pass | false | false | false | The parser and prompt evidence keep item IDs outside the R5C valid namespace. |
| PREFLIGHT-011 | original_repository | pass | false | selective invalid_format loop present | repository evidence recorded | The original code has a selective retry loop for format failures in its title-list contract. |
| PREFLIGHT-012 | original_repository | pass | true | title-list generation and title-to-item metric mapping | not R5C index namespace | The original selective retry is not an authorization for R5C item-ID-in-index-field failures. |
| PREFLIGHT-013 | paper_protocol | pass | true | paper_pdf_exists=True; populated_repetition_policy=False | no explicit paper retry policy in local atomic contract | Local paper-derived protocol evidence does not predeclare an invalid-only rerun policy for this adapted R5C. |
| PREFLIGHT-014 | bias_risk | fail | true | rerun set selected after observing invalid outputs | predeclared attempt policy before inference | Replacing only observed invalid outputs would introduce selective-retry bias under the frozen R5C attempt set. |
| PREFLIGHT-015 | protocol_mixing | fail | true | new recovery namespace would be combined with original first-attempt namespace | single predeclared protocol identity or separate new protocol | Using a separate namespace as a replacement while retaining the same protocol hash would mix attempt policies. |
| PREFLIGHT-016 | raw_preservation | pass | false | 15/15 original raw files present | 15/15 | Original invalid raw outputs remain present and were not overwritten by this preflight. |

## Remediation

`FULL R5C RERUN REQUIRED` if the target is an all-valid R5C freeze under one unbiased attempt policy. A future remediation should use a new protocol identity with predeclared retry/replacement rules before inference. The existing 15-row list remains diagnostic evidence and cannot repair the frozen R5C freeze by itself.
