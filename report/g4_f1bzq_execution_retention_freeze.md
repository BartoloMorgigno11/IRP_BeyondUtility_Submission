# G4 F1BZQ Reranking Execution Retention Freeze

## Scope
- Decision source: `G4_FULL_EXECUTION_COMPLETED_AND_RECONCILED`.
- Freeze token: `G4_EXECUTION_RETENTION_FROZEN`.
- No new inference, no Ollama/API calls, no metric computation, no statistical analysis, no protocol/config/runner changes, and no runtime ledger cleanup were performed during this retention freeze.

## Accounting
- Logical/backend G4 requests: `29292`.
- Response reuse: `0`.
- Strict valid terminal responses: `28986`.
- Terminal invalid scientific evidence: `306` (`invalid_length=255`, `duplicate_indices=51`).
- Backend errors: `0`.
- Remaining calls: `0`.

## Integrity
- Raw response files: `29292`; unique IDs: `29292`.
- Terminal response rows: `29292`; unique IDs: `29292`.
- Request ledger rows: `29292`; unique IDs: `29292`.
- Event ledger rows: `87878`.
- Cells observed: `30`.
- Orphans: `0`; missing evidence: `0`; raw duplicates: `0`; terminal duplicates: `0`; ambiguous evidence: `0`.
- Raw file name, zero-length, tiny-file, JSON parse, ID mismatch, raw-terminal record hash mismatch, canonical identity mismatch, candidate hash mismatch, and model digest mismatch checks: `0` failures.

## Recovery Provenance
- Request-only pre-response orphan incidents recovered: `2`.
- Each recovered request ID resolves to exactly one final raw response file and one terminal response row.
- Recovery manifests are retained in the external archive and summarized in the retention manifest.

## Model Evidence
- `qwen2:7b-instruct`: `9764` raw responses, digest `dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755`.
- `mistral:7b-instruct`: `9764` raw responses, digest `6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf`.
- `gemma2:9b`: `9764` raw responses, digest `ff02c3702f322b9e075e9568332d96c0a7028002f1a5a056e0a6784320a4db0b`.

## Retention
- Tracked hash inventory: `results/tables/g4_f1bzq_execution_evidence_inventory.csv`.
- Retention manifest: `results/manifests/g4_f1bzq_execution_retention_manifest.json`.
- External archive: `archive_external/g4_f1bzq_reranking_execution/G4_f1bzq_reranking_df8700c_external_evidence.zip`.
- External archive SHA256: `dc32bb824475be37f13616d9d7f5080b5f7da8de5ccba11ce78cfccd39089cd8`.
- Archive entries: `29299`; uncompressed bytes: `692338223`.
- The raw responses, JSONL ledgers, and recovery provenance are ignored by normal Git and retained through the external archive plus tracked per-file SHA256 inventory.

## Runtime Provenance
- First request start UTC: `2026-09-05T08:39:14.505000Z`.
- Last request completion UTC: `2026-09-06T03:05:59.675000Z`.
- Backend runtime sum seconds: `54581.288043096836`.
- Backend throughput calls/hour: `1932.0027756900276`.
- Ledger wall throughput calls/hour: `1587.9968381979897`.

G4_EXECUTION_RETENTION_FROZEN
