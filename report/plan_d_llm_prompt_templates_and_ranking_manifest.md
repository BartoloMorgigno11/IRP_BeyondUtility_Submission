# Plan D LLM Prompt Templates and ML-1M Ranking Manifest

## Purpose

D5D prepares reusable LLM prompt templates, strict parser utilities, and a deterministic ML-1M 20-user diagnostic manifest for the future Qwen2 local ranking run.

No LLM inference was run in this phase. No external API was called. No Ollama HTTP request, OpenAI API call, Anthropic API call, download, model pull, recommender training, LastFM experiment, Plan C modification, metric CSV update, or raw-output generation occurred.

## Prompt Strategies Implemented Or Documented

- `openp5_base`: implemented first diagnostic template. It uses user history, the candidate list, and item metadata, and asks the model to rank only candidate IDs.
- `llmrank_recency`: implemented/documented recency-focused variant inspired by LLMRank, using recent history first.
- `llmsrec_icl_stub`: documented ICL placeholder for future work; not used in D5E unless examples are implemented.

The registry is stored in `results/tables/plan_d_llm_prompt_template_registry.csv`.

## Strict JSON Output Schema

```json
{
  "ranked_item_ids": ["item_id_1", "item_id_2", "item_id_3", "item_id_4", "item_id_5"],
  "status": "ok"
}
```

Prompts instruct the model to return only valid JSON, rank exactly `top_k` item IDs, use only candidate IDs, avoid invented item IDs, avoid explanations, and choose from candidates even when uncertain.

## Parser Validation Rules

The parser accepts JSON objects only. It requires `ranked_item_ids` to be a list, records selected IDs exactly as returned, detects invalid/hallucinated IDs, detects duplicates, records length mismatches, and computes `candidate_validity_rate = valid returned IDs / selected_count` with `0.0` for empty selections.

Parser status values are `parsed_valid`, `parsed_invalid_schema`, `parsed_with_hallucinations`, `parsed_with_duplicates`, `parse_failed`, and `empty_response`.

## ML-1M 20-User Diagnostic Manifest

- Manifest path: `results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_manifest.csv`
- Dataset: `ml1m`
- Phase: `D5E_qwen2_ml1m_ranking_diagnostic`
- Model: `qwen2:7b-instruct`
- Backend: `ollama`
- Prompt strategy: `openp5_base`
- Top-k: `5`
- Candidate pool size: `20`
- Manifest rows: `20`
- Ready rows: `20`
- Rows with positives in candidates: `20`
- Candidate-count values present: `[20]`
- Manifest summary: `prepared_full_20_user_manifest`

## Candidate Pool Setup

The manifest reads existing reconstructed Plan C ML-1M artifacts from `data/reconstructed/plan_c/ml1m`: `interactions.csv`, `candidates.csv`, `ground_truth.csv`, and `catalog.csv`. Candidate sets are not invented. The reconstructed ML-1M candidate artifact contains 100 candidates per user; D5D deterministically prepares a 20-candidate diagnostic pool by retaining the positive target and selecting the earliest reconstructed negative candidates by candidate position. Ready rows require non-empty training history, exactly 20 selected candidates, exactly one positive candidate according to the reconstructed candidate setup, and a positive target present in the selected candidate list.

## Limitations

- This is a no-inference preparation manifest, not an LLM output file.
- The manifest uses reconstructed Plan C candidate pools, not official paper candidate pools.
- Full prompts are not saved in the manifest; only prompt character lengths and short previews are stored.
- No LLM ranking reproduction is claimed yet.
- LastFM is not part of this phase.

## Recommended Next Step

Run D5E: Qwen2 ML-1M ranking diagnostic, 20 users, local Ollama backend, `temperature=0`, `prompt_strategy=openp5_base`, and strict JSON parsing.
