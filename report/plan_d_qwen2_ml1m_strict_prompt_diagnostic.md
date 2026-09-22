# Plan D Qwen2 ML-1M Strict Prompt Diagnostic

## Purpose

D5G tests whether the hallucination and candidate-invalidity observed in D5E can be reduced by a stricter candidate-ID-only prompt and by deterministic post-filtering.

## Scope And Classification

- D5G is a diagnostic_subset prompt mitigation diagnostic.
- Classification: `diagnostic_subset` robustness/reliability diagnostic.
- This is not exact reproduction.
- This is not full LLM reproduction.
- This is a local open model diagnostic only.
- Backend/model: local Ollama with `qwen2:7b-instruct`.
- Strict prompt local Qwen2 requests: `20`.
- The strict prompt condition uses exactly the same 20 ML-1M diagnostic instances and same candidate sets as D5E.
- Temperature: `0`.
- Top-k: `5`.
- Candidate pool size: `20`.
- Total runtime seconds: `95.546373`.

## Execution Boundaries

- D5G uses local Qwen2 through Ollama only.
- No OpenAI API call was made.
- No Anthropic API call was made.
- No external API call was made.
- No download was performed.
- No model pull was performed.
- No recommender training was performed.
- Plan C was not modified.
- Raw JSONL outputs are stored under `results/raw_outputs/` and remain Git-ignored.

## Conditions

- `original_parsed_output`: existing D5E parsed output reference; no new inference.
- `original_post_filtered`: deterministic post-filtering of D5E outputs; no new inference.
- `strict_candidate_id_prompt`: 20 new local Qwen2/Ollama requests using a candidate-ID-only prompt.
- `strict_post_filtered`: deterministic post-filtering of strict prompt outputs; no extra inference.

The strict prompt tries to mitigate hallucination by making candidate IDs the only allowed output domain. Post-filtering removes IDs outside the candidate set, preserves valid order, and does not backfill.

## Metric Summary

| Condition | HR@5 | NDCG@5 | Mean Candidate Validity | Micro Candidate Validity | Hallucination Rate | Exact Top-k Rate | Exact 5 Valid IDs Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `original_parsed_output` | `0.150000` | `0.100889` | `0.630000` | `0.630000` | `0.370000` | `1.000000` | `0.600000` |
| `original_post_filtered` | `0.150000` | `0.100889` | `0.750000` | `1.000000` | `0.000000` | `0.600000` | `0.600000` |
| `strict_candidate_id_prompt` | `0.550000` | `0.288479` | `1.000000` | `1.000000` | `0.000000` | `1.000000` | `1.000000` |
| `strict_post_filtered` | `0.550000` | `0.288479` | `1.000000` | `1.000000` | `0.000000` | `1.000000` | `1.000000` |

## Post-Filter Summary

| Condition | Mean Removed | Total Removed | Filtered Mean Length | Filtered HR@5 | Filtered NDCG@5 | Filtered Hallucination Rate |
|---|---:|---:|---:|---:|---:|---:|
| `original_post_filtered` | `1.850000` | `37` | `3.150000` | `0.150000` | `0.100889` | `0.000000` |
| `strict_post_filtered` | `0.000000` | `0` | `5.000000` | `0.550000` | `0.288479` | `0.000000` |

## Interpretation

The strict prompt reduced hallucination relative to D5E. Hallucination delta is `-0.370000`, candidate-validity delta is `0.370000`, and HR@5 delta is `0.400000`.

`original_post_filtered` removed `37` invalid IDs, leaving hallucination_rate `0.000000` and HR@5 `0.150000`. `strict_post_filtered` removed `0` invalid IDs, leaving hallucination_rate `0.000000` and HR@5 `0.550000`.

Syntactic validity means the model returned parseable JSON with the expected fields. Candidate validity means selected IDs belong to the provided candidate set. Hallucination means selected IDs fall outside that candidate set. Utility is measured separately with HR@5 and NDCG@5 against the known positive item.

## Beyond Utility Interpretation

D5G supports the broader Beyond Utility argument because a model can be parse-valid while still failing candidate adherence. Prompt-level mitigation and deterministic post-filtering affect reliability and utility differently, so HR/NDCG alone are insufficient to describe LLM recommender behavior.

## Limitations

- Only 20 users.
- Only ML-1M.
- Only Qwen2.
- Only the original D5E candidate order.
- Only one strict candidate-ID prompt variant.
- Not exact paper reproduction.
- Not full LLM reproduction.

## Recommended Next Steps

1. Compare with Mistral after the user manually pulls a quantized Mistral v0.2 model.
2. Run LLM reranking diagnostics.
3. Run history length sensitivity diagnostics.
4. Integrate D5E-D5G reliability findings into the final report.
