# Plan D Final Reproducibility Package

## Purpose

D5L is an aggregation/reporting and hygiene-audit phase. It reads existing Plan D outputs and creates compact final package tables without running a new experiment.

## Execution Boundaries

- No new inference was run.
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
- Plan C was not modified.
- No destructive cleanup was performed.
- Raw JSONL outputs remain under `results/raw_outputs/` and Git-ignored.

## Reproduction Claim

This is not exact reproduction. This is not full LLM reproduction. The project should be described as reconstructed reproduction plus diagnostic/extended local reproduction work. Plan C is reconstructed reproduction. Plan D extends that work with extended neural baselines and diagnostic local LLM studies.

## Generated Final Tables

- `results/tables/plan_d_final_experiment_index.csv` with `19` rows.
- `results/tables/plan_d_final_metric_summary.csv` with `88` rows.
- `results/tables/plan_d_final_llm_diagnostic_summary.csv` with `14` rows.
- `results/tables/plan_d_final_reproducibility_status.csv` with `11` rows.
- `results/tables/plan_d_final_blocked_items.csv` with `6` rows.
- `results/tables/plan_d_repository_hygiene_audit.csv` with `9` rows.

## Main Scientific Story

1. Classical and neural baselines were reconstructed or extended over available reconstructed datasets.
2. Local LLM diagnostics were feasible on the ML-1M 20-user diagnostic subset.
3. D5E base prompting produced valid JSON but high hallucination: HR@5 `0.150000`, NDCG@5 `0.100889`, candidate validity `0.630000`, hallucination `0.370000`.
4. D5G strict candidate-ID-only prompting improved grounding: HR@5 `0.550000`, NDCG@5 `0.288479`, candidate validity `1.000000`, hallucination `0.000000`.
5. Candidate order and history length affected results, so prompt context is a robustness variable.
6. Qwen2 and Mistral behaved differently: D5I Qwen2 strict HR@5 `0.550000` versus Mistral strict HR@5 `0.100000`.
7. Re-ranking was not automatically beneficial: D5J candidate-order reference HR@5 `0.100000` while local LLM reranking conditions did not improve that reference.
8. Beyond-utility reliability metrics are necessary because HR/NDCG alone hide parse validity, candidate validity, hallucination, duplicate behavior, and stability.

## D5K History-Length Note

D5K showed that strict prompts can keep hallucination at `0.000000` for Qwen2 medium-history while utility and stability still vary with the history window. Qwen2 medium-history HR@5 was `0.450000`.

## Blocked And Conditional Items

The final blocked-items table documents Claude 3 Haiku exact reproduction as retired, Llama3 70B as blocked by local hardware, closed-source LLMs as API/budget conditional, LastFM as `extension_only_no_valid_utility_evaluation`, official preprocessed artifacts as missing, and full LLM reproduction as not completed.

## Hygiene Audit

D5L performed a non-destructive repository hygiene audit. It checked raw-output location, ignore patterns, tracked raw-output paths, `original_repo/` Git status, checkpoint/model/cache suffixes in tracked paths, large files, and recommended final verification commands. No files were deleted.

## Recommended Final Steps

1. Review `git status --short`.
2. Run `python -m pytest tests/ -q`.
3. Use the final CSV tables when drafting final report and presentation claims.
4. Commit only after manually confirming raw JSONL files remain ignored.
