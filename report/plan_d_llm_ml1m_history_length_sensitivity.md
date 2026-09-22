# Plan D Local LLM ML-1M History Length Sensitivity

## Purpose

D5K evaluates whether local LLM recommendation outputs are sensitive to the amount of user history shown in the prompt. It uses the same ML-1M 20-user diagnostic subset as D5E/D5G/D5I/D5J and varies only the user-history length.

## Scope And Classification

- D5K is a diagnostic_subset local LLM history length sensitivity diagnostic.
- D5K is not exact reproduction.
- D5K is not full LLM reproduction.
- D5K uses local Ollama only.
- Qwen2 and Mistral are local open models.
- Maximum 120 new local LLM requests were allowed.
- Exactly 120 new local LLM requests were run.
- D5K does not reproduce the paper's full LLM matrix or closed-source setup.

## Execution Boundaries

- No OpenAI API call was made.
- No Anthropic API call was made.
- No external API call was made.
- No download was performed during D5K.
- No model pull was performed during D5K.
- No recommender training was performed.
- Plan C was not modified.
- Raw JSONL outputs are stored under `results/raw_outputs/` and remain Git-ignored.

## Controlled Design

D5K varies only user-history length while keeping candidate sets, positive items, model, top_k, temperature, backend, and strict candidate-ID-only JSON constraints fixed. The three history settings are `short_history` with the most recent 5 items, `medium_history` with the most recent 10 items, and `long_history` with the most recent 20 items or fewer if unavailable.

## Condition Summary

| Condition | Model | History | HR@5 | NDCG@5 | Candidate Validity | Hallucination | Jaccard vs Medium | Output Stability |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `qwen2_short_history` | `qwen2:7b-instruct` | `short_history` | `0.450000` | `0.237590` | `1.000000` | `0.000000` | `0.562698` | `0.150000` |
| `qwen2_medium_history` | `qwen2:7b-instruct` | `medium_history` | `0.450000` | `0.247602` | `1.000000` | `0.000000` | `1.000000` | `0.150000` |
| `qwen2_long_history` | `qwen2:7b-instruct` | `long_history` | `0.350000` | `0.176096` | `1.000000` | `0.000000` | `0.660119` | `0.150000` |
| `mistral_short_history` | `mistral:7b-instruct` | `short_history` | `0.150000` | `0.090876` | `1.000000` | `0.000000` | `0.608532` | `0.150000` |
| `mistral_medium_history` | `mistral:7b-instruct` | `medium_history` | `0.100000` | `0.040876` | `1.000000` | `0.000000` | `1.000000` | `0.150000` |
| `mistral_long_history` | `mistral:7b-instruct` | `long_history` | `0.150000` | `0.090876` | `1.000000` | `0.000000` | `0.704365` | `0.150000` |

## Pairwise History Deltas

| Model | Condition A | Condition B | HR@5 Delta | NDCG@5 Delta | Validity Delta | Hallucination Delta | Selected Jaccard | Interpretation |
|---|---|---|---:|---:|---:|---:|---:|---|
| `qwen2:7b-instruct` | `short_history` | `medium_history` | `0.000000` | `0.010012` | `0.000000` | `0.000000` | `0.562698` | `qwen2:7b-instruct_medium_history_minus_short_history_utility_unchanged_validity_unchanged` |
| `qwen2:7b-instruct` | `long_history` | `medium_history` | `0.100000` | `0.071506` | `0.000000` | `0.000000` | `0.660119` | `qwen2:7b-instruct_medium_history_minus_long_history_utility_improved_validity_unchanged` |
| `qwen2:7b-instruct` | `short_history` | `long_history` | `-0.100000` | `-0.061494` | `0.000000` | `0.000000` | `0.576389` | `qwen2:7b-instruct_long_history_minus_short_history_utility_decreased_validity_unchanged` |
| `mistral:7b-instruct` | `short_history` | `medium_history` | `-0.050000` | `-0.050000` | `0.000000` | `0.000000` | `0.608532` | `mistral:7b-instruct_medium_history_minus_short_history_utility_decreased_validity_unchanged` |
| `mistral:7b-instruct` | `long_history` | `medium_history` | `-0.050000` | `-0.050000` | `0.000000` | `0.000000` | `0.704365` | `mistral:7b-instruct_medium_history_minus_long_history_utility_decreased_validity_unchanged` |
| `mistral:7b-instruct` | `short_history` | `long_history` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.576984` | `mistral:7b-instruct_long_history_minus_short_history_utility_unchanged_validity_unchanged` |

Stability hit-pattern counts: `{"all_hit": 8, "medium_long_only": 1, "medium_only": 1, "none_hit": 26, "short_long_only": 1, "short_medium_only": 1, "short_only": 2}`.

## Interpretation

D5K evaluates utility and reliability separately. HR@5 and NDCG@5 describe utility. Candidate validity, hallucination, duplicate rate, response/parse/schema rates, and exact-top-k rates describe output reliability. Jaccard and hit-pattern metrics describe output stability as history length changes.

The diagnostic distinguishes candidate validity, hallucination, utility, and output stability because a model can preserve syntactic JSON compliance while changing recommendation utility or drifting toward invalid candidate IDs. This supports the broader Beyond Utility argument: utility metrics alone are insufficient to characterize local LLM recommender behavior.

## Limitations

- Only 20 ML-1M diagnostic users are evaluated.
- Only Qwen2 and Mistral local open models are evaluated.
- Only strict candidate-ID-only prompts are used.
- No closed-source API model is evaluated.
- This is diagnostic_subset evidence, not exact paper reproduction.

## Recommended Next Steps

1. Integrate D5E-D5K local LLM diagnostics into the final report.
2. Prepare exam presentation tables from generated CSV files.
3. Optionally run an OpenAI API diagnostic only if budget and explicit approval are provided.
