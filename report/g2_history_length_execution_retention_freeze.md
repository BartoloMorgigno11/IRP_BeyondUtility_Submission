# G2 History-Length Execution Retention Freeze

## Scope
- Decision source: `G2_FULL_EXECUTION_COMPLETED_AND_RECONCILED`.
- Freeze token: `G2_EXECUTION_RETENTION_FROZEN`.
- No new inference, no Ollama calls, no G3, no protocol change, and no metric interpretation were performed during this retention freeze.

## Accounting
- Logical G2 requests: `88050`.
- Exact G1 zero-call reuse: `18660`.
- G2 backend calls: `69390`.
- Strict valid terminal responses: `68274`.
- Terminal invalid scientific evidence: `1116` (`INVALID_LENGTH=940`, `INVALID_DUPLICATE=166`, `HALLUCINATION_OR_OUT_OF_CANDIDATE=10`).
- Backend errors: `0`.

## Integrity
- Raw response files: `69390`; unique IDs: `69390`.
- Terminal response rows: `69390`; unique IDs: `69390`.
- Request ledger unique IDs: `69390`; rows: `69392`.
- Preserved duplicate pre-send resume records: `2`, both same-payload duplicates with exactly one raw file and one terminal response each.
- Orphans: `0`; raw duplicates: `0`; terminal duplicates: `0`; ambiguous evidence: `0`.
- Raw file name, zero-length, tiny-file, JSON parse, ID mismatch, and model digest mismatch checks: `0` failures.

## Model Evidence
- `qwen2:7b-instruct`: `23130` raw responses, digest `dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755`.
- `mistral:7b-instruct`: `23130` raw responses, digest `6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf`.
- `gemma2:9b`: `23130` raw responses, digest `ff02c3702f322b9e075e9568332d96c0a7028002f1a5a056e0a6784320a4db0b`.

## Retention
- Tracked hash inventory: `results/tables/g2_history_length_execution_evidence_inventory.csv`.
- Retention manifest: `results/manifests/g2_history_length_execution_retention_manifest.json`.
- External archive: `archive_external/g2_history_length_execution/G2_history_length_8fb6dc9_external_evidence.zip`.
- External archive SHA256: `8d8f6192435b3c996306f1d2a8305ef70e3cdf263a1e23b8d6b466dfb1272a46`.
- Archive entries: `69393`; uncompressed bytes: `1169038183`.
- The raw responses and JSONL ledgers are ignored by normal Git and retained through the external archive plus tracked per-file SHA256 inventory.

## Runtime Provenance
- First request start UTC: `2026-09-01T16:50:02.675000Z`.
- Last request completion UTC: `2026-09-03T13:06:05.508000Z`.
- Backend runtime sum seconds: `114627.56629200754`.
- Backend throughput calls/hour: `2179.266367425422`.

G2_EXECUTION_RETENTION_FROZEN
