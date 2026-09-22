# Plan D Qwen2 Output Validity and Hallucination Analysis

## Purpose

D5E.1 analyzes the D5E Qwen2 ML-1M ranking diagnostic outputs without running new LLM inference. The goal is to explain the gap between perfect JSON/schema compliance and weaker candidate adherence.

## Input Artifacts

- `results/metrics/plan_d_qwen2_ml1m_ranking_diagnostic_metrics.csv`
- `results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_instance_summary.csv`
- `results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_summary.csv`
- `report/plan_d_qwen2_ml1m_ranking_diagnostic.md`
- Optional ignored raw trace: `results/raw_outputs/plan_d_qwen2_ml1m_ranking_diagnostic_raw.jsonl`

The ignored local D5E raw JSONL was available, so compact invalid-ID pattern counts and valid-only utility were computed from parsed fields.

No new LLM inference was run. No OpenAI API call was made. No Anthropic API call was made. No Ollama command, Ollama HTTP API call, remote API call, download, model pull, recommender training, BPRMF/LightGCN/SASRec/GRU4Rec run, or LastFM experiment occurred.

## D5E Result Recap

- HR@5: `0.150000`
- NDCG@5: `0.100889`
- response_received_rate: `1.000000`
- parse_success_rate: `1.000000`
- schema_valid_rate: `1.000000`
- exact_top_k_rate: `1.000000`
- mean_candidate_validity_rate: `0.630000`
- hallucination_rate: `0.370000`
- duplicate_rate: `0.000000`

## Output Validity Analysis

Qwen2 was reliable at the transport, JSON, and schema levels in this diagnostic subset: responses were received for all `20` instances, every response parsed as JSON, every response passed schema validation, and every response returned exactly five selected IDs.

Candidate adherence was the main failure mode. The run selected `100` item IDs, of which `63` were valid candidates and `37` were outside the candidate sets. The micro candidate-validity rate was `0.630000`.

## Hallucination Analysis

- Instances with hallucinations: `8` of `20`
- High-severity hallucination instances: `8`
- Total hallucinated items: `37`
- Invalid item pattern summary: 1088 (1, non_candidate_existing_format); 1197 (1, non_candidate_existing_format); 1266 (1, non_candidate_existing_format); 1269 (1, non_candidate_existing_format); 1374 (1, non_candidate_existing_format)

The invalid IDs mostly preserve the MovieLens numeric ID format when raw parsed fields are available. This suggests the model often selected plausible movie IDs rather than malformed strings, but those IDs were not members of the provided candidate set.

## Valid-Only Utility Analysis

| Variant | HR@5 | NDCG@5 | Mean Positive Rank | Hits | Notes |
|---|---:|---:|---:|---:|---|
| original_parsed_output | `0.150000` | `0.100889` | `2.666667` | `3` | Original D5E utility. |
| valid_only_output | `0.150000` | `0.100889` | `2.666667` | `3` | Removes hallucinated IDs while preserving valid returned order when raw parsed fields are available. |
| invalid_as_failure | `0.150000` | `0.100889` | `2.666667` | `3` | Counts any hallucinating instance as a utility failure. |

## Operational Interpretation

Qwen2 is reliable at JSON/schema level in this run, but it is not fully reliable at candidate-adherence level. A production recommender would need constrained decoding, post-filtering, stronger prompt constraints, or a candidate-ID-only representation before using these outputs directly.

## Methodological Importance

This supports the Beyond Utility motivation. HR/NDCG alone are insufficient because an LLM can return parseable outputs and still recommend invalid items outside the candidate set. Candidate validity, hallucination rate, duplicate rate, and parse/schema rates must be reported alongside utility.

## Limitations

- Only 20 users.
- Only ML-1M.
- Only Qwen2.
- Only `openp5_base`.
- Raw-level pattern analysis depends on local ignored raw JSONL availability.
- No new inference in this phase.

## Recommended Next Steps

1. Run candidate position bias diagnostic.
2. Test a stricter candidate-ID-only prompt or post-filtered evaluation.
3. Pull and test Mistral.
4. Run LLM reranking diagnostic.
