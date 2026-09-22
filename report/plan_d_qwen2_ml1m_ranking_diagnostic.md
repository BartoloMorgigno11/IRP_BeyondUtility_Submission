# Plan D Qwen2 ML-1M Ranking Diagnostic

## Purpose

D5E runs the first real local LLM-as-recommender diagnostic experiment in Plan D. It evaluates `qwen2:7b-instruct` through local Ollama on the deterministic 20-user ML-1M manifest prepared in D5D.

This is not exact paper reproduction. It is a bounded reconstructed diagnostic subset using local open-weight inference, reconstructed ML-1M candidate pools, and one prompt strategy.

## Exact Configuration

- Plan: `Plan D`
- Phase: `D5E_qwen2_ml1m_ranking_diagnostic`
- Dataset: `ml1m`
- Scope: `diagnostic_subset`
- Backend: `ollama`
- Model: `qwen2:7b-instruct`
- Prompt strategy: `openp5_base`
- Temperature: `0`
- Users: `20`
- Top-k: `5`
- Candidate pool size: `20`
- Output format: `strict_json`
- Local only: `true`

## Execution Boundaries

- This was local-only Ollama inference.
- No OpenAI, Anthropic, or remote API was called.
- No model was pulled or downloaded in this phase.
- No recommender model was trained.
- No BPRMF, LightGCN, SASRec, GRU4Rec, or LastFM experiment was run.
- No Plan C output, dataset reconstruction file, recommender raw output, or previous source metric CSV was modified.

## Manifest Summary

- Manifest path: `results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_manifest.csv`
- Manifest rows processed: `20`
- Required manifest status: `ready_for_llm_inference`
- Required positives in candidates: `true`
- Candidate count per instance: `20`

## Prompt Strategy

The run used `openp5_base`, which includes chronological user history, the fixed 20-item candidate list, item titles/metadata where available, and a strict JSON schema requesting exactly five candidate IDs. Full prompts are not stored in committed tables.

## Parser And Output-Validity Checks

Responses were parsed with the strict D5D ranking parser. The evaluator did not repair malformed JSON, did not manually correct invalid IDs, and did not silently drop hallucinated IDs. Utility metrics use the returned parsed ranking order without manual correction; parse failures receive HR@5 = `0` and NDCG@5 = `0`.

## Utility Results

- HR@5: `0.150000`
- NDCG@5: `0.100889`

## Output Validity Results

- response_received_rate: `1.000000`
- parse_success_rate: `1.000000`
- schema_valid_rate: `1.000000`
- mean_candidate_validity_rate: `0.630000`
- hallucination_rate: `0.370000`
- duplicate_rate: `0.000000`
- exact_top_k_rate: `1.000000`
- empty_output_rate: `0.000000`

## Runtime And Latency

- mean_latency_seconds: `5.652650`
- total_runtime_seconds: `113.389987`
- raw_output_path: `results/raw_outputs/plan_d_qwen2_ml1m_ranking_diagnostic_raw.jsonl`

## Limitations

- Only 20 users.
- Only ML-1M.
- Only `qwen2:7b-instruct`.
- Only `openp5_base`.
- Not 1000 users.
- Not 5 repeated experiments.
- Not all 7 paper LLMs.
- Re-ranking, position bias, and history sensitivity are still pending.
- Candidate pools are reconstructed and diagnostic, not exact paper pools.

## Result Status

- status: `completed_diagnostic_subset`
- notes: Diagnostic processed all 20 D5D manifest rows with local Qwen2/Ollama inference; not exact paper reproduction.

## Recommended Next Steps

Inspect output validity and hallucination details, then run the candidate position-bias diagnostic. Optionally pull and run Mistral later, then run the reranking diagnostic after the ranking path is stable.
