# F6L LastFM V4 Full Ranking Protocol Amendment

- Amendment ID: `phase_f6l_lastfm_v4_full_ranking_final_protocol_amendment_20260804`
- Run ID: `20260804_131959`
- Created at: `2026-08-05T07:57:12Z`
- Amendment hash: `7FD1BDE5B50B308FE171A634D97FA61D5441EDAD2C911059259201E4C33D9308`
- Interpretation: `runtime_structured_output_does_not_hard_enforce_array_uniqueness`.
- Scientific classification: `paper_protocol_reproduction_under_explicit_operationalization`.
- Additional classification: `duplicate_token_rate_is_an_observed_model_runtime_property`.
- The prior stop used a conservative 100-response 99% validity window; that rule is retired for this final resume.
- Duplicate-token systemic stop now requires either `5` consecutive duplicate-token structural-invalid outputs or a duplicate-token response rate greater than `0.05` in a closed non-overlapping `200`-new-response window.
- Contract unchanged: parser, schema, prompts, model digests, generation parameters, candidate mapping, request IDs, run ID.
- Structural-invalid outputs are retained as observed behavior and evaluated fail-strict; they are not retried or repaired.
- Completion no longer requires `strict-valid = 4000`; it requires `strict-valid + structural-invalid = 4000` with raw coherence and unchanged contracts.
