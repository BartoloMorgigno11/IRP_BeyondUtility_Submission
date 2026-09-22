# Phase F6L LastFM Operationalized Ranking Execution

- Decision token: `f6l_lastfm_operationalized_ranking_execution_and_rq_revalidation_complete`.
- Run ID: `20260802_184502`.
- Raw output directory: `results/llm/phase_f6l_lastfm_operationalized_ranking_20260802_184502`.
- Evidence class: `paper_protocol_reproduction_under_explicit_operationalization`.
- Exact paper LastFM remains `blocked_missing_original_specification`.
- Full cohort and reranking were not executed.

## Metrics

- `qwen2:7b-instruct` `base/OpenP5 title-list` `HR@5` = `0`.
- `qwen2:7b-instruct` `base/OpenP5 title-list` `NDCG@5` = `0`.
- `qwen2:7b-instruct` `base/OpenP5 title-list` `validity_rate` = `0`.
- `qwen2:7b-instruct` `base/OpenP5 title-list` `hallucination_rate` = `1`.
- `qwen2:7b-instruct` `base/OpenP5 title-list` `duplicate_rate` = `0`.
- `qwen2:7b-instruct` `LLMRank recency` `HR@5` = `0`.
- `qwen2:7b-instruct` `LLMRank recency` `NDCG@5` = `0`.
- `qwen2:7b-instruct` `LLMRank recency` `validity_rate` = `0`.
- `qwen2:7b-instruct` `LLMRank recency` `hallucination_rate` = `0`.
- `qwen2:7b-instruct` `LLMRank recency` `duplicate_rate` = `0`.
- `mistral:7b-instruct` `base/OpenP5 title-list` `HR@5` = `0`.
- `mistral:7b-instruct` `base/OpenP5 title-list` `NDCG@5` = `0`.
- `mistral:7b-instruct` `base/OpenP5 title-list` `validity_rate` = `0`.
- `mistral:7b-instruct` `base/OpenP5 title-list` `hallucination_rate` = `0.05`.
- `mistral:7b-instruct` `base/OpenP5 title-list` `duplicate_rate` = `0`.
- `mistral:7b-instruct` `LLMRank recency` `HR@5` = `0`.
- `mistral:7b-instruct` `LLMRank recency` `NDCG@5` = `0`.
- `mistral:7b-instruct` `LLMRank recency` `validity_rate` = `0`.
- `mistral:7b-instruct` `LLMRank recency` `hallucination_rate` = `0`.
- `mistral:7b-instruct` `LLMRank recency` `duplicate_rate` = `0`.

## Limits

- This is not the original paper sample and not an exact paper numerical replica.
- Closed/remote models, LLMSRec, full cohort, and reranking remain outside this run.