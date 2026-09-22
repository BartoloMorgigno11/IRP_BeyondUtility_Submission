# G3 Candidate-Position Execution Retention Freeze

## Scope
- Decision source: `G3_FULL_EXECUTION_COMPLETED_AND_RECONCILED`.
- Freeze token: `G3_EXECUTION_RETENTION_FROZEN`.
- No new inference, no Ollama/API calls, no metric computation, no statistical analysis, no protocol/config/runner changes, and no evidence rewrites were performed during this retention freeze.

## Accounting
- Logical/backend G3 requests: `12000`.
- FIRST/RANDOM: `FIRST=6000`, `RANDOM=6000`.
- Strict valid terminal responses: `11834`.
- Terminal invalid scientific evidence: `166` (`INVALID_LENGTH=114`, `INVALID_DUPLICATE=51`, `HALLUCINATION_OR_OUT_OF_CANDIDATE=1`).
- Backend errors: `0`.
- Retries: `0`.
- Remaining calls: `0`.

## Integrity
- Raw response files: `12000`; unique IDs: `12000`.
- Terminal response rows: `12000`; unique IDs: `12000`.
- Request ledger rows: `12000`; unique IDs: `12000`.
- Event ledger rows: `36000`.
- Orphans: `0`; missing evidence: `0`; raw duplicates: `0`; terminal duplicates: `0`; ambiguous evidence: `0`.
- Raw file name, zero-length, tiny-file, JSON parse, ID mismatch, raw-record hash mismatch, terminal-record hash mismatch, and canonical identity mismatch checks: `0` failures.

## Model Evidence
- `qwen2:7b-instruct`: `4000` raw responses, digest `dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755`.
- `mistral:7b-instruct`: `4000` raw responses, digest `6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf`.
- `gemma2:9b`: `4000` raw responses, digest `ff02c3702f322b9e075e9568332d96c0a7028002f1a5a056e0a6784320a4db0b`.

## Retention
- Tracked hash inventory: `results/tables/g3_candidate_position_execution_evidence_inventory.csv`.
- Retention manifest: `results/manifests/g3_candidate_position_execution_retention_manifest.json`.
- External archive: `archive_external/g3_candidate_position_execution/G3_candidate_position_06088e9_external_evidence.zip`.
- External archive SHA256: `605c075af4a602d914d527d2ae555367abab76dcae95ede4bdff59a3b033696e`.
- Archive entries: `12003`; uncompressed bytes: `237006057`.
- The raw responses and JSONL ledgers are ignored by normal Git and retained through the external archive plus tracked per-file SHA256 inventory.

## Runtime Provenance
- First request start UTC: `2026-09-04T07:06:53.827000Z`.
- Last request completion UTC: `2026-09-04T13:25:53.429000Z`.
- Backend runtime sum seconds: `22361.579943798948`.
- Backend throughput calls/hour: `1931.884961106235`.
- Ledger wall throughput calls/hour: `1899.7693978988727`.

G3_EXECUTION_RETENTION_FROZEN
