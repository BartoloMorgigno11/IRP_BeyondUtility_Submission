# Phase F6G2 - Contract-Gated LastFM Re-ranking Execution Freeze

## Initial Repository State

Branch `plan-d-extensions`, HEAD `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`, staged files ``. Protected dirty tracked files remain `M README.md
 M report/dataset_audit.md
 M report/notes.md`.

## Frozen Inputs

Changed inputs: `0`; missing inputs: `0`.

## Contract Reconciliation

F6G1 future manifests explicitly require candidate pool size 20, output length 20 with parser indexed_json_top20_unique_candidates, while approved reranking adapter evidence requires top-10.

## Candidate Pool Size

F6G1 candidate pool size evidence: `20`.

## Requested Ranking Length

F6G1 requested length: `20`; approved required length: `10`.

## Parser Expected Length

F6G1 parser schema length: `20`; approved parser length: `10`.

## Execution Status

No Ollama generation, prompt rendering, retries or raw JSONL appends were attempted.

## Shard Status

Shard rows written: `15`; all shards are blocked before inference.

## Request Status

Request rows written: `7436`; each request is marked `blocked_pre_inference_contract_conflict`.

## Raw Outputs

F6G2 raw files are configured but are not created by this blocked freeze.

## Population Summaries

All-user, strict-valid, positive-present operational and positive-present strict-valid populations are materialized separately with blocked status.

## Tests

Validation passed `0`, failed `0`.

## Final Decision

f6g2_blocked_by_runtime

## Recommendation

repair_runtime_before_resume

## Scientific Content Hash

172e9d28acbccac78187b0638028a7d379dd96e67961b8e1b4db2058c61893b0
