# Plan D Qwen2 ML-1M Candidate Position Bias Diagnostic

## Purpose

D5F tests whether the local Qwen2 ML-1M ranking diagnostic is sensitive to candidate input order. It reuses the same 20 D5E diagnostic instances and reorders the same candidate sets into `reverse_order`, `positive_first`, and `positive_last`.

## Scope And Classification

- D5F is a diagnostic_subset robustness diagnostic.
- Classification: `diagnostic_subset` robustness diagnostic.
- This is not exact reproduction.
- This is not full LLM reproduction.
- This is a local open model diagnostic only.
- Backend/model: local Ollama with `qwen2:7b-instruct`.
- New local Qwen2 requests: `60` (`20` users times `3` variants).
- Temperature: `0`.
- Top-k: `5`.
- Candidate pool size: `20`.
- Total runtime seconds: `330.391387`.

## Execution Boundaries

- No OpenAI API call was made.
- No Anthropic API call was made.
- No external API call was made.
- No download was performed.
- No model pull was performed.
- No recommender training was performed.
- Plan C was not modified.
- Raw JSONL outputs are stored under `results/raw_outputs/` and remain Git-ignored.
- The original D5E order was used as a reference and was not rerun.

## Variant Summary

| Variant | HR@5 | NDCG@5 | Candidate Validity | Hallucination Rate | Mean Selected Position | Front-Half Rate | Position-1 Rate | Position-20 Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `reverse_order` | `0.250000` | `0.250000` | `0.660000` | `0.340000` | `6.757576` | `0.772727` | `0.106061` | `0.015152` |
| `positive_first` | `0.400000` | `0.344343` | `0.690000` | `0.310000` | `8.347826` | `0.666667` | `0.115942` | `0.043478` |
| `positive_last` | `0.050000` | `0.019343` | `0.570000` | `0.430000` | `8.456140` | `0.684211` | `0.052632` | `0.017544` |

## Position Bias Interpretation

`reverse_order` shows front-position preference (0.772727 front, 0.227273 back). `positive_first` shows front-position preference (0.666667 front, 0.333333 back). `positive_last` shows front-position preference (0.684211 front, 0.315789 back).

The selected-position metrics distinguish candidate validity, hallucination, utility, and position bias. Hallucinated IDs have no candidate input position, so position-bias rates are computed over valid selected candidate IDs.

## Utility Interpretation

`reverse_order` HR@5 delta vs original is `0.100000` (HR@5_increased_vs_original). `reverse_order` NDCG@5 delta vs original is `0.149111` (NDCG@5_increased_vs_original). `positive_first` HR@5 delta vs original is `0.250000` (HR@5_increased_vs_original). `positive_first` NDCG@5 delta vs original is `0.243454` (NDCG@5_increased_vs_original). `positive_last` HR@5 delta vs original is `-0.100000` (HR@5_decreased_vs_original). `positive_last` NDCG@5 delta vs original is `-0.081546` (NDCG@5_decreased_vs_original).

## Reliability Interpretation

`reverse_order` hallucination_rate delta vs original is `-0.030000` (hallucination_rate_decreased_vs_original). `reverse_order` mean_candidate_validity_rate delta vs original is `0.030000` (mean_candidate_validity_rate_increased_vs_original). `positive_first` hallucination_rate delta vs original is `-0.060000` (hallucination_rate_decreased_vs_original). `positive_first` mean_candidate_validity_rate delta vs original is `0.060000` (mean_candidate_validity_rate_increased_vs_original). `positive_last` hallucination_rate delta vs original is `0.060000` (hallucination_rate_increased_vs_original). `positive_last` mean_candidate_validity_rate delta vs original is `-0.060000` (mean_candidate_validity_rate_decreased_vs_original).

## Beyond Utility Interpretation

D5F supports the broader Beyond Utility argument because candidate reordering can affect reliability and candidate-position behavior independently of HR/NDCG. A parse-valid LLM output can still be invalid, hallucinated, or position-sensitive.

## Limitations

- Only 20 users.
- Only ML-1M.
- Only Qwen2.
- Only `openp5_base`.
- Only three deterministic reorderings.
- Not exact paper reproduction.
- Not full LLM reproduction.

## Recommended Next Steps

1. Test a stricter candidate-ID-only prompt or post-filtered evaluation.
2. Pull and test Mistral.
3. Run LLM reranking diagnostic.
4. Run history length sensitivity diagnostic.
