# R5A Controlled Recovery Manifest

Decision: `REPRODUCTION_R5A_RECOVERY_MANIFEST_COMPLETE`

R5A built a deterministic request manifest for the approved R4 P0/P1 recovery gaps only. It executed no LLM/Ollama/API calls, no training, no downloads, and no `original_repo/` edits.

## Exact Count

- Raw metric-level planned requests: `6000`
- Metric-only duplicate requests: `1200`
- Reusable existing outputs: `0`
- Unique new future requests: `4800`
- Manifest SHA256: `290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD`

The R4 value `4400` remains a planning estimate. R5A supersedes it with the exact manifest-derived count of `4800`. The delta is `+400` requests, a `9.09%` increase over the R4 estimate, because ML-1M and LastFM need `400` explicit profile-generation prerequisite requests before their downstream profile-plus-history recommendation requests. This is not scope expansion; it resolves a prerequisite that the R4 family-level estimate had not materialized.

## Generation Split

- New profile-generation requests: `400`
- Downstream recommendation requests: `800`
- Total generation-family requests: `1200`

Beauty and Sports record existing F6L profile artifacts as dependencies. ML-1M and LastFM record future profile-generation dependencies explicitly; downstream rows reference those dependency request IDs and cannot be executed before the profile outputs are materialized.

R5A does not fabricate future profile output hashes. Reused Beauty/Sports profiles use `profile_hash_status=INPUT_PROFILE_HASH_AVAILABLE`; ML-1M/LastFM downstream rows use `profile_hash_status=PROFILE_DEPENDENCY_REQUEST_ID_PENDING_OUTPUT` and a deterministic `profile_dependency_request_id`.

## P0 Scope

- Beauty CandDif-HR
- Beauty CandDif-NDCG
- LastFM CandDif-HR
- LastFM CandDif-NDCG
- Sports CandDif-HR
- Sports CandDif-NDCG

## P1 Scope

- Beauty Generation-involved profile recommendation
- Beauty History-length sensitivity
- LastFM Generation-involved profile recommendation
- LastFM History-length sensitivity
- ML-1M Generation-involved profile recommendation
- ML-1M History-length sensitivity
- Sports Generation-involved profile recommendation
- Sports History-length sensitivity

## Validation

Validation result: `23 PASS / 0 FAIL`.

R5A excludes all P2/P3/P4 gaps, keeps ML-1M CandDif out of P0, and records every material prompt, candidate, history, profile dependency, parser contract, expected output schema, and output destination in `results/tables/reproduction_r5a_recovery_request_manifest.csv`.
