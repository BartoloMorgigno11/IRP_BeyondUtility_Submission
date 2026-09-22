# F6L LastFM V4.1 Reranking Continuation Freeze

Decision token: `f6l_lastfm_v4_reranking_post_breaker_versioned_continuation_ready`.

## Scope

- Freeze only: no inference, Ollama call, API call, detached process, commit, raw overwrite, repair, or deduplication was performed.
- Base run: `20260806_034027`.
- Continuation id: `phase_f6l_lastfm_v4_reranking_continuation_v4_1_20260807`.

## Policy Resolution

- A canonical request is completed only when it has a terminal final response and parser classification.
- Non-final transport timeout evidence does not make a request completed.
- V4.1 targets all and only canonical requests without a final response, not a contiguous manifest ordinal interval.
- The 48 non-final timeout requests are included as `retry_after_non_final_transport_timeout`.
- The 1552 never-attempted requests are included as `first_attempt_never_attempted`.

## Counts

- Continuation rows: `1600`.
- Timeout retry rows: `48`.
- Never-attempted rows: `1552`.
- Prior final response exists: `0`.
- Raw hash before/after freeze: `AB6449BD7BE8B8288A6FB0DB5FB3F3439F616662AE93FF572065120D2B6BE030` / `AB6449BD7BE8B8288A6FB0DB5FB3F3439F616662AE93FF572065120D2B6BE030`.

## Breaker V4.1

- Duplicate-token remains structural-invalid and fail-strict.
- Stop if two consecutive closed 200-final-response windows in the same model/strategy cell both exceed 5%.
- Or stop if, after at least 400 final responses in the same cell, cumulative duplicate-token rate exceeds 5%.
- Hard stop remains immediate for malformed JSON, unknown/out-of-domain token, wrong length, mapping failure, parser drift, schema drift, or manifest/hash drift.

## Artifacts

- Config: `configs/phase_f6l_lastfm_v4_reranking_continuation_v4_1.yaml`.
- Manifest: `results/tables/phase_f6l_lastfm_v4_reranking_continuation_v4_1_manifest.csv`.
- Policy audit: `results/tables/phase_f6l_lastfm_v4_reranking_continuation_v4_1_policy_audit.csv`.
- Summary: `results/manifests/phase_f6l_lastfm_v4_reranking_continuation_v4_1_summary.json`.
- Hashes: `results/manifests/phase_f6l_lastfm_v4_reranking_continuation_v4_1_hashes.json`.
