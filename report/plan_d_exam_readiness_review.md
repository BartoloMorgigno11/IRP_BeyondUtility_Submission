# Plan D Exam Readiness Review

## Executive Summary

D5M is a final reporting audit and exam-readiness review. It did not run any new experiment. The Plan D package is exam-ready in its core scientific positioning: it clearly supports a reconstructed reproduction plus local diagnostic/extended subset analysis, while avoiding an exact or full LLM reproduction claim. No blocking exam-readiness issue was found in the D5M checklist.

## Safe Final Positioning

This is not exact reproduction of the full paper. It is reconstructed reproduction plus local diagnostic/extended subset analysis. Plan C remains reconstructed reproduction. Plan D adds extended neural baselines and diagnostic_subset local LLM studies that test output validity, hallucination, prompt mitigation, candidate order sensitivity, model differences, re-ranking, and history length sensitivity. Closed-source LLM replication is conditional/API-dependent, Claude 3 Haiku exact is blocked/retired, Llama3 70B local is blocked by hardware, and LastFM is extension_only_no_valid_utility_evaluation.

## What Not To Claim

- Do not claim full LLM reproduction.
- Do not claim exact reproduction of Beyond Utility.
- Do not claim Mistral/Qwen results generalize to the full benchmark.
- Do not claim LLMs outperform neural recommenders globally.
- Do not claim hallucination is solved.
- Do not claim LastFM was reproduced.
- Do not claim closed-source LLMs were reproduced.
- Do not claim Plan D is exact reproduction.

## What To Emphasize

- The project is transparent about missing official artifacts and exact-model blockers.
- The final package separates exact, reconstructed, diagnostic_subset, extended_subset, extension, and blocked/conditional categories.
- D5E showed syntactic validity is not enough because candidate hallucination remained high.
- D5G showed strict candidate-ID-only prompting can improve grounding in the tested local diagnostic setting.
- D5F, D5J, and D5K show that order, re-ranking framing, and history window length affect local LLM behavior.
- D5I shows Qwen2 and Mistral behave differently under the same local strict-prompt setup.
- These findings support the Beyond Utility argument: reliability metrics matter alongside HR/NDCG.

## Remaining Limitations

- No full paper LLM matrix was reproduced.
- No OpenAI or Anthropic API diagnostics were run.
- Claude 3 Haiku exact reproduction is blocked/retired.
- Llama3 70B local execution is blocked by hardware.
- LastFM remains extension_only_no_valid_utility_evaluation.
- D5E-D5K are ML-1M diagnostic_subset studies, not full benchmark results.
- D5H should be described as initially conditional, then completed as a local Mistral smoke test after manual user installation.

## Execution Boundaries

- No new inference/API/download/training was run in D5M.
- No Ollama call was made.
- No Qwen2 call was made.
- No Mistral call was made.
- No OpenAI API call was made.
- No Anthropic API call was made.
- No external API call was made.
- No download was performed.
- No model pull was performed.
- No recommender training was performed.
- No LastFM run was performed.
- No destructive cleanup was performed.

## Recommended Next Step

Use `results/tables/plan_d_oral_defense_key_points.csv` as the exam preparation sheet, and keep `results/tables/plan_d_claim_safety_audit.csv` open while drafting slides or the final oral script.

## Generated Review Artifacts

- Checklist rows: `13`
- Claim safety rows: `9`
- Oral defense key-point rows: `13`
