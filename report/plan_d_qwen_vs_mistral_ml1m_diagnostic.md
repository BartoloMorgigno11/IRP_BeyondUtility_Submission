# Plan D Qwen2 vs Mistral ML-1M Strict-Prompt Diagnostic

## Purpose

D5I compares local Qwen2 and local Mistral on the same ML-1M 20-user diagnostic subset using the strict candidate-ID-only prompt setup introduced in D5G.

## Scope And Classification

- D5I is a diagnostic_subset local open-model comparison.
- D5I is a robustness/reliability diagnostic.
- D5I is not exact reproduction.
- D5I is not full LLM reproduction.
- D5I compares local open models only and does not reproduce the paper's closed-source LLM matrix.
- Qwen2 outputs are reused from D5G and are not rerun.
- Mistral uses local Ollama only.
- Exactly 20 new Mistral requests were run.
- Mistral was installed manually before D5I/D5H completion.
- Temperature: `0`.
- Top-k: `5`.
- Candidate pool size: `20`.
- Mistral runtime seconds: `89.110282`.

## Execution Boundaries

- No OpenAI API call was made.
- No Anthropic API call was made.
- No external API call was made.
- No download was performed during D5I.
- No model pull was performed during D5I.
- No recommender training was performed.
- Plan C was not modified.
- Raw JSONL outputs are stored under `results/raw_outputs/` and remain Git-ignored.

## Model Summary

| Model | Condition | HR@5 | NDCG@5 | Candidate Validity | Hallucination Rate | Parse Success | Schema Valid | Exact Top-k | Mean Latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen2:7b-instruct` | `qwen2_strict_reference` | `0.550000` | `0.288479` | `1.000000` | `0.000000` | `1.000000` | `1.000000` | `1.000000` | `4.776439` |
| `mistral:7b-instruct` | `mistral_strict_candidate_id_prompt` | `0.100000` | `0.044343` | `1.000000` | `0.000000` | `1.000000` | `1.000000` | `1.000000` | `4.454589` |

## Pairwise Delta And Agreement

| Metric | Qwen2 | Mistral | Delta | Pairwise Value |
|---|---:|---:|---:|---:|
| `HR@5_delta` | `0.550000` | `0.100000` | `-0.450000` | `` |
| `NDCG@5_delta` | `0.288479` | `0.044343` | `-0.244136` | `` |
| `candidate_validity_delta` | `1.000000` | `1.000000` | `0.000000` | `` |
| `hallucination_rate_delta` | `0.000000` | `0.000000` | `0.000000` | `` |
| `parse_success_delta` | `1.000000` | `1.000000` | `0.000000` | `` |
| `schema_valid_delta` | `1.000000` | `1.000000` | `0.000000` | `` |
| `exact_top_k_delta` | `1.000000` | `1.000000` | `0.000000` | `` |
| `duplicate_rate_delta` | `0.000000` | `0.000000` | `0.000000` | `` |
| `mean_latency_delta` | `4.776439` | `4.454589` | `-0.321850` | `` |
| `valid_recommendation_jaccard` | `` | `` | `` | `0.260119` |
| `selected_item_jaccard` | `` | `` | `` | `0.260119` |
| `positive_selection_agreement_rate` | `` | `` | `` | `0.550000` |
| `both_hit_rate` | `` | `` | `` | `0.100000` |
| `qwen_only_hit_rate` | `` | `` | `` | `0.450000` |
| `mistral_only_hit_rate` | `` | `` | `` | `0.000000` |
| `neither_hit_rate` | `` | `` | `` | `0.450000` |

Agreement category counts: `{"both_hit": 2, "neither_hit": 9, "qwen_only_hit": 9}`.

## Interpretation

Utility differences are captured by HR@5 and NDCG@5 deltas. Reliability differences are captured by parse/schema rates, candidate validity, hallucination rate, duplicate rate, and exact top-k rate. Agreement metrics show whether the two local open models hit the same positive instances or diverge even when both follow the strict candidate-ID output format.

D5I supports the broader Beyond Utility argument because two parseable local LLM recommenders can differ in candidate adherence, agreement, latency, and hit behavior even under the same candidate-ID-only constraints. These local diagnostic results are useful for method tracing, but they are diagnostic_subset results rather than exact paper reproduction because the official paper candidate pools, full 1000-user runs, repeated trials, and full LLM matrix are not reproduced here.

## Recommended Next Steps

1. Inspect Mistral validity and agreement cases before scaling.
2. Run LLM reranking diagnostics.
3. Run history length sensitivity diagnostics.
4. Integrate D5E-D5I findings into the final report.
