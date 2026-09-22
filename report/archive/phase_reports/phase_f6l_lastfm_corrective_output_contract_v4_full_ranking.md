# Phase F6L LastFM V4 Constrained Full Ranking

- Decision token: `f6l_lastfm_v4_full_ranking_complete_with_observed_duplicate_invalidity`
- Run ID: `20260804_131959`
- Raw output directory: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_full_ranking_20260804_131959`
- Classification: `paper_protocol_reproduction_under_explicit_operationalization`
- Planned/completed/strict-valid/failed/residual: `4000`/`4000`/`3950`/`50`/`0`.
- Canary status: `pass`.
- Circuit breaker: `clear`.
- Parser hash: `5527024992F87984078F46F15491F1E8CB6EB7DF6659559416EBD67F8A098854`.
- Manifest hash: `0189529D45ECBBB35A4E46E035928C6361EC5670113982B04C84DED141983650`.

## Metrics

- `qwen2:7b-instruct` `base/OpenP5 title-list` `HR@5_fail_strict` = `0.458`.
- `qwen2:7b-instruct` `base/OpenP5 title-list` `NDCG@5_fail_strict` = `0.323647507835`.
- `qwen2:7b-instruct` `base/OpenP5 title-list` `strict_valid_rate` = `1`.
- `qwen2:7b-instruct` `base/OpenP5 title-list` `structural_invalid_rate` = `0`.
- `qwen2:7b-instruct` `base/OpenP5 title-list` `duplicate_token_rate` = `0`.
- `qwen2:7b-instruct` `base/OpenP5 title-list` `retry_rate` = `0.004`.
- `qwen2:7b-instruct` `LLMRank recency` `HR@5_fail_strict` = `0.419`.
- `qwen2:7b-instruct` `LLMRank recency` `NDCG@5_fail_strict` = `0.308222730898`.
- `qwen2:7b-instruct` `LLMRank recency` `strict_valid_rate` = `1`.
- `qwen2:7b-instruct` `LLMRank recency` `structural_invalid_rate` = `0`.
- `qwen2:7b-instruct` `LLMRank recency` `duplicate_token_rate` = `0`.
- `qwen2:7b-instruct` `LLMRank recency` `retry_rate` = `0`.
- `mistral:7b-instruct` `base/OpenP5 title-list` `HR@5_fail_strict` = `0.307`.
- `mistral:7b-instruct` `base/OpenP5 title-list` `NDCG@5_fail_strict` = `0.223073398036`.
- `mistral:7b-instruct` `base/OpenP5 title-list` `strict_valid_rate` = `0.983`.
- `mistral:7b-instruct` `base/OpenP5 title-list` `structural_invalid_rate` = `0.017`.
- `mistral:7b-instruct` `base/OpenP5 title-list` `duplicate_token_rate` = `0.017`.
- `mistral:7b-instruct` `base/OpenP5 title-list` `retry_rate` = `0.009`.
- `mistral:7b-instruct` `LLMRank recency` `HR@5_fail_strict` = `0.319`.
- `mistral:7b-instruct` `LLMRank recency` `NDCG@5_fail_strict` = `0.23287621762`.
- `mistral:7b-instruct` `LLMRank recency` `strict_valid_rate` = `0.967`.
- `mistral:7b-instruct` `LLMRank recency` `structural_invalid_rate` = `0.033`.
- `mistral:7b-instruct` `LLMRank recency` `duplicate_token_rate` = `0.033`.
- `mistral:7b-instruct` `LLMRank recency` `retry_rate` = `0`.

## Boundaries

- This is `paper_protocol_reproduction_under_explicit_operationalization`.
- Additional classification: `structured_output_validity_is_an_observed_model_runtime_property`.
- Additional classification: `duplicate_token_rate_is_an_observed_model_runtime_property`.
- It is not exact paper reproduction, not the original paper sample, and not the original paper output contract.
- V1/V2/V3 and pilot V4 outputs remain separate fragility evidence.

<!-- phase_f6l_lastfm_v4_ranking_metrics_freeze:start -->
## Ranking Metrics Freeze Before Reranking

- Freeze state: `ranking_metrics_frozen_before_reranking`.
- Run ID: `20260804_131959`.
- Decision token: `f6l_lastfm_v4_ranking_metrics_consolidated_and_frozen_for_reranking`.
- Freeze SHA-256: `D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC`.
- Canonical metrics: `results/tables/phase_f6l_lastfm_v4_ranking_freeze_metrics.csv`.
- Canonical comparisons: `results/tables/phase_f6l_lastfm_v4_ranking_freeze_comparisons.csv`.
- Summary JSON: `results/manifests/phase_f6l_lastfm_v4_ranking_metrics_freeze_summary.json`.
- Scope boundary: no reranking, no new inference, no raw repair, no deduplication, no API call.
<!-- phase_f6l_lastfm_v4_ranking_metrics_freeze:end -->
