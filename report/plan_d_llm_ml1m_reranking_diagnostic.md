# Plan D Local LLM ML-1M Re-Ranking Diagnostic

## Purpose

D5J evaluates whether local LLMs can re-rank an already ordered candidate list on the same ML-1M 20-user diagnostic subset. The original candidate order is treated as a baseline/retriever reference.

## Scope And Classification

- D5J is a diagnostic_subset local LLM re-ranking diagnostic.
- D5J is a robustness/reliability diagnostic.
- D5J is not exact reproduction.
- D5J is not full LLM reproduction.
- D5J uses local Ollama only.
- Qwen2 and Mistral are local open models.
- Exactly 40 new local LLM requests were run.
- D5J does not reproduce the paper's full LLM matrix or closed-source setup.

## Execution Boundaries

- No OpenAI API call was made.
- No Anthropic API call was made.
- No external API call was made.
- No download was performed during D5J.
- No model pull was performed during D5J.
- No recommender training was performed.
- Plan C was not modified.
- Raw JSONL outputs are stored under `results/raw_outputs/` and remain Git-ignored.

## Condition Summary

| Condition | Model | HR@5 | NDCG@5 | Candidate Validity | Hallucination Rate | Mean Input Position | Mean Abs Shift |
|---|---|---:|---:|---:|---:|---:|---:|
| `candidate_order_reference` | `candidate_order_reference` | `0.100000` | `0.056546` | `1.000000` | `0.000000` | `3.000000` | `0.000000` |
| `qwen2_rerank_prompt` | `qwen2:7b-instruct` | `0.050000` | `0.031546` | `0.400000` | `0.600000` | `3.550000` | `0.600000` |
| `mistral_rerank_prompt` | `mistral:7b-instruct` | `0.050000` | `0.021534` | `0.930000` | `0.070000` | `4.268817` | `1.666667` |

## Pairwise Deltas

| Comparison | Metric | Reference | Qwen2 | Mistral | Condition | Delta | Pairwise |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen2_rerank_prompt_vs_reference` | `HR@5_delta` | `0.100000` | `` | `` | `0.050000` | `-0.050000` | `` |
| `qwen2_rerank_prompt_vs_reference` | `NDCG@5_delta` | `0.056546` | `` | `` | `0.031546` | `-0.025000` | `` |
| `qwen2_rerank_prompt_vs_reference` | `candidate_validity_delta` | `1.000000` | `` | `` | `0.400000` | `-0.600000` | `` |
| `qwen2_rerank_prompt_vs_reference` | `hallucination_rate_delta` | `0.000000` | `` | `` | `0.600000` | `0.600000` | `` |
| `qwen2_rerank_prompt_vs_reference` | `duplicate_rate_delta` | `0.000000` | `` | `` | `0.000000` | `0.000000` | `` |
| `qwen2_rerank_prompt_vs_reference` | `mean_selected_input_position_delta` | `3.000000` | `` | `` | `3.550000` | `0.550000` | `` |
| `qwen2_rerank_prompt_vs_reference` | `mean_absolute_position_shift` | `0.000000` | `` | `` | `0.600000` | `0.600000` | `` |
| `qwen2_rerank_prompt_vs_reference` | `selected_item_jaccard` | `1.000000` | `` | `` | `0.338095` | `-0.661905` | `` |
| `qwen2_rerank_prompt_vs_reference` | `valid_item_jaccard` | `1.000000` | `` | `` | `0.338095` | `-0.661905` | `` |
| `mistral_rerank_prompt_vs_reference` | `HR@5_delta` | `0.100000` | `` | `` | `0.050000` | `-0.050000` | `` |
| `mistral_rerank_prompt_vs_reference` | `NDCG@5_delta` | `0.056546` | `` | `` | `0.021534` | `-0.035012` | `` |
| `mistral_rerank_prompt_vs_reference` | `candidate_validity_delta` | `1.000000` | `` | `` | `0.930000` | `-0.070000` | `` |
| `mistral_rerank_prompt_vs_reference` | `hallucination_rate_delta` | `0.000000` | `` | `` | `0.070000` | `0.070000` | `` |
| `mistral_rerank_prompt_vs_reference` | `duplicate_rate_delta` | `0.000000` | `` | `` | `0.000000` | `0.000000` | `` |
| `mistral_rerank_prompt_vs_reference` | `mean_selected_input_position_delta` | `3.000000` | `` | `` | `4.268817` | `1.268817` | `` |
| `mistral_rerank_prompt_vs_reference` | `mean_absolute_position_shift` | `0.000000` | `` | `` | `1.666667` | `1.666667` | `` |
| `mistral_rerank_prompt_vs_reference` | `selected_item_jaccard` | `1.000000` | `` | `` | `0.683135` | `-0.316865` | `` |
| `mistral_rerank_prompt_vs_reference` | `valid_item_jaccard` | `1.000000` | `` | `` | `0.696726` | `-0.303274` | `` |
| `qwen2_rerank_vs_mistral_rerank` | `HR@5_delta` | `` | `0.050000` | `0.050000` | `` | `0.000000` | `` |
| `qwen2_rerank_vs_mistral_rerank` | `NDCG@5_delta` | `` | `0.031546` | `0.021534` | `` | `-0.010012` | `` |
| `qwen2_rerank_vs_mistral_rerank` | `candidate_validity_delta` | `` | `0.400000` | `0.930000` | `` | `0.530000` | `` |
| `qwen2_rerank_vs_mistral_rerank` | `hallucination_rate_delta` | `` | `0.600000` | `0.070000` | `` | `-0.530000` | `` |
| `qwen2_rerank_vs_mistral_rerank` | `selected_item_jaccard` | `` | `` | `` | `` | `` | `0.276984` |
| `qwen2_rerank_vs_mistral_rerank` | `valid_item_jaccard` | `` | `` | `` | `` | `` | `0.277679` |
| `qwen2_rerank_vs_mistral_rerank` | `positive_selection_agreement_rate` | `` | `` | `` | `` | `` | `0.900000` |
| `qwen2_rerank_vs_mistral_rerank` | `both_hit_rate` | `` | `` | `` | `` | `` | `0.000000` |
| `qwen2_rerank_vs_mistral_rerank` | `qwen_only_hit_rate` | `` | `` | `` | `` | `` | `0.000000` |
| `qwen2_rerank_vs_mistral_rerank` | `mistral_only_hit_rate` | `` | `` | `` | `` | `` | `0.050000` |
| `qwen2_rerank_vs_mistral_rerank` | `neither_hit_rate` | `` | `` | `` | `` | `` | `0.900000` |

Agreement category counts: `{"mistral_only_hit": 1, "none_hit": 17, "reference_only_hit": 1, "reference_qwen_hit_only": 1}`.
Mean absolute position shift by model from the position-shift profile: `{"mistral:7b-instruct": 1.666667, "qwen2:7b-instruct": 0.6}`.

## Interpretation

D5J evaluates utility separately from reliability. HR@5 and NDCG@5 show whether re-ranking improves or worsens utility compared with `candidate_order_reference`; candidate validity, hallucination, duplicate rate, parse/schema rates, and exact-top-k rates describe output reliability. Position-shift metrics describe how far each local LLM moves selected items away from the original input order.

D5J supports the broader Beyond Utility argument because a local LLM can produce parse-valid and candidate-valid output while still changing utility and position behavior relative to the original candidate-order reference. These results remain diagnostic_subset results, not exact reproduction, because official paper reranking pools, full user counts, repeated runs, and closed-source models are not reproduced.

## Recommended Next Steps

1. Run history length sensitivity diagnostics.
2. Integrate D5E-D5J results into the final report.
3. Prepare exam presentation tables.
