# Finalization F6D4A - LastFM local LLM ranking

Decisione: `finalization_f6d4a_complete`.

F6D4A esegue solo ranking LastFM su manifest F6D3 congelato. Non esegue reranking, training, pull, download o API remote.

## Execution

- `qwen2:7b-instruct`: completed `1859/1859`, transport success `1859`, strict-valid `1859`, status `complete`.
- `mistral:7b-instruct`: completed `1859/1859`, transport success `1859`, strict-valid `1857`, status `complete`.

## Utility

- `mistral:7b-instruct` `HR@5` all-request: `0.528778913394`.
- `mistral:7b-instruct` `NDCG@5` all-request: `0.344688254765`.
- `mistral:7b-instruct` `mean_positive_rank` all-request: `2.55747711089`.
- `mistral:7b-instruct` `median_positive_rank` all-request: `2`.
- `qwen2:7b-instruct` `HR@5` all-request: `0.485745024207`.
- `qwen2:7b-instruct` `NDCG@5` all-request: `0.293906202156`.
- `qwen2:7b-instruct` `mean_positive_rank` all-request: `3.01439645626`.
- `qwen2:7b-instruct` `median_positive_rank` all-request: `3`.

## Structural Validity

- `mistral:7b-instruct` `strict_validity_rate`: `0.99892415277` (`1857/1859`).
- `mistral:7b-instruct` `duplicate_index_rate`: `0` (`0/1859`).
- `mistral:7b-instruct` `out_of_range_index_rate`: `0.000537923614847` (`1/1859`).
- `mistral:7b-instruct` `malformed_output_rate`: `0` (`0/1859`).
- `qwen2:7b-instruct` `strict_validity_rate`: `1` (`1859/1859`).
- `qwen2:7b-instruct` `duplicate_index_rate`: `0` (`0/1859`).
- `qwen2:7b-instruct` `out_of_range_index_rate`: `0` (`0/1859`).
- `qwen2:7b-instruct` `malformed_output_rate`: `0` (`0/1859`).

## Scope Caveats

- Local Qwen2 and Mistral are original paper model families but local adapted Ollama executions, not exact paper versions.
- All-request utility is primary. Strict-valid utility is secondary and does not replace invalid-as-failure aggregation.
- Structural validity is not a paper-exact textual hallucination metric.
- Protocol-exact numeric replication remains `0/187`.

## Validation

- Pass: `31`; fail: `0`.
