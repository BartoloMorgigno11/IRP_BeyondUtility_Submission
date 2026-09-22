# Plan D Qwen2 Ollama Smoke Test

## Purpose

D5C1 verifies that the local Ollama backend and `qwen2:7b-instruct` can produce a strictly parseable JSON response for a toy recommendation-style prompt. This is a local smoke test only.

## Backend And Model

- Backend: `ollama`
- Model: `qwen2:7b-instruct`
- Temperature: `0`
- Endpoint: local Ollama generate API on `localhost`

## Prompt Schema

The toy prompt used no real user data. It asked for exactly:

```json
{
  "ranked_item_ids": ["item_id_1", "item_id_2", "item_id_3"],
  "status": "ok"
}
```

The candidate IDs were `item_A`, `item_B`, `item_C`, `item_D`, and `item_E`.

## Validation Criteria

- A response must be received from local Ollama.
- The response text must parse as JSON.
- The JSON must contain `ranked_item_ids` as exactly three strings and `status` equal to `ok`.
- Returned IDs must come from the toy candidate set.
- Duplicate IDs and hallucinated IDs are counted without manual correction.

## Result Summary

The smoke test passed: `smoke_test_passed`.

| Field | Value |
|---|---|
| response_received | `true` |
| parse_success | `true` |
| schema_valid | `true` |
| selected_count | `3` |
| candidate_validity_rate | `1.000000` |
| duplicate_count | `0` |
| hallucinated_count | `0` |
| hallucinations_occurred | `no` |
| duplicate_ids_occurred | `no` |
| latency_seconds | `9.196662` |
| status | `smoke_test_passed` |

Notes: Model returned parseable JSON with valid toy candidate IDs.

## Scope Limits

- This is not a dataset ranking experiment.
- This is not an exact paper LLM reproduction.
- No external API was called.
- No OpenAI API or Anthropic API was called.
- No model was pulled or downloaded in this phase.
- No recommender model was trained.
- No Plan C output was modified.
- No recommender raw output was modified.
- No fake LLM output was created.

Raw LLM response text was not saved; the committed CSV/report contain only summary fields.

## Recommended Next Step

Implement reusable LLM prompt templates for the real diagnostic path, then run the bounded Qwen ML-1M diagnostic only after that parser/template layer is ready.
