# G3 Candidate-Position Execution Runner Freeze

Decision token: `G3_EXECUTION_RUNNER_FROZEN`.

## Accounting

| accounting_scope | logical_requests | new_ollama_calls_required | existing_g3_terminal_requests | pending_new_requests | g1_g2_reuse_requests |
| --- | --- | --- | --- | --- | --- |
| GRAND_TOTAL | 12000 | 12000 | 0 | 12000 | 0 |

## Runner Contract

- Live execution requires `--allow-api-calls`.
- Dry-run performs zero Ollama calls.
- Raw response file is written before terminal parsing ledger append.
- Structural invalid responses are terminal scientific evidence and are not retried.
- Resume skips only matching terminal G3 evidence by canonical request ID.
