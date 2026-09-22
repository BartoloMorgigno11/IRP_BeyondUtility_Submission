# Plan D Mistral Smoke Test

## Purpose

D5H checks whether a local Mistral 7B Instruct-compatible Ollama model is available and, only if available, runs a single tiny candidate-ID-only JSON smoke test. It does not run ML-1M ranking diagnostics.

## Scope And Classification

- D5H is a local model availability/smoke-test phase.
- This is not exact reproduction.
- This is not full LLM reproduction.
- This is a local open model smoke test only when the requested Mistral model is already available.
- Target default model: `mistral:7b-instruct`.
- Configured requested model: `mistral:7b-instruct`.
- Temperature: `0`.
- Top-k: `3`.

## Execution Boundaries

- No OpenAI API call was made.
- No Anthropic API call was made.
- No external API call was made.
- No recommender training was performed.
- Plan C was not modified.
- D5H does not provide ML-1M ranking metrics.
- D5H only prepares a future Qwen-vs-Mistral comparison.

## Result

- Status: `smoke_test_passed`.
- Requested model: `mistral:7b-instruct`.
- Exact model name used: `mistral:7b-instruct`.
- Model available locally: `true`.
- Local Ollama was used.
- Exactly 1 smoke-test request was run.
- Raw JSONL output: `results/raw_outputs/plan_d_mistral_smoke_test_raw.jsonl`.
- The raw JSONL output remains Git-ignored.


## Smoke-Test Metrics

| Metric | Value |
|---|---:|
| response_received | `true` |
| parse_success | `true` |
| schema_valid | `true` |
| exact_top_k | `true` |
| candidate_validity_rate | `1.000000` |
| hallucination_rate | `0.000000` |
| duplicate_rate | `0.000000` |
| latency_seconds | `56.699009` |


## Notes

Mistral local smoke test produced strict parseable candidate-valid JSON.
