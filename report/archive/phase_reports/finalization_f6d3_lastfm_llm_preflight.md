# Finalization F6D3 - LastFM LLM preflight

Decisione: `finalization_f6d3_complete`.

F6D3 non esegue training, Ollama o inference. Ricarica i checkpoint F6D2 in sola lettura, audita la sufficienza del training e congela manifest/policy per una possibile F6D4.

## Training adequacy

- bprmf: `configuration_limited`, seeds `5`, epoch range `1-1`.
- gru4rec: `configuration_limited`, seeds `5`, epoch range `1-1`.
- lightgcn: `configuration_limited`, seeds `5`, epoch range `1-1`.
- sasrec: `configuration_limited`, seeds `5`, epoch range `1-1`.

## Reranking

- Paper-aligned pool: non creato; union/source seed/top-K non recuperati.
- Adapted policy primaria: all-user denominator, positives absent count as utility failures.
- Adapted policy diagnostica: positive-present subset, not comparable to all-user.
- Positive inclusion: `592/1859` = `0.318450779989`.

## LLM preflight

- mistral:7b-instruct ranking: `go_adapted`.
- mistral:7b-instruct reranking: `go_diagnostic_only`.
- qwen2:7b-instruct ranking: `go_adapted`.
- qwen2:7b-instruct reranking: `go_diagnostic_only`.

## Validation

- Pass: `32`; fail: `0`.
- Ranking requests: `3718`.
- Reranking requests: `3718`.

## Output

- `training_adequacy`: `results/tables/finalization_f6d3_training_adequacy.csv`
- `model_adequacy_summary`: `results/tables/finalization_f6d3_model_adequacy_summary.csv`
- `checkpoint_reload_audit`: `results/tables/finalization_f6d3_checkpoint_reload_audit.csv`
- `full_catalog_audit`: `results/tables/finalization_f6d3_full_catalog_audit.csv`
- `reranking_protocol`: `results/tables/finalization_f6d3_reranking_protocol.csv`
- `reranking_pool_composition`: `results/tables/finalization_f6d3_reranking_pool_composition.csv`
- `positive_inclusion_summary`: `results/tables/finalization_f6d3_positive_inclusion_summary.csv`
- `reranking_upper_bounds`: `results/tables/finalization_f6d3_reranking_upper_bounds.csv`
- `reranking_evaluation_policy`: `results/tables/finalization_f6d3_reranking_evaluation_policy.csv`
- `llm_cohort_policy`: `results/tables/finalization_f6d3_llm_cohort_policy.csv`
- `local_llm_inventory`: `results/tables/finalization_f6d3_local_llm_inventory.csv`
- `prompt_contract`: `results/tables/finalization_f6d3_prompt_contract.csv`
- `ranking_manifest`: `results/tables/finalization_f6d3_lastfm_ranking_llm_manifest.csv`
- `reranking_manifest`: `results/tables/finalization_f6d3_lastfm_reranking_llm_manifest.csv`
- `llm_execution_plan`: `results/tables/finalization_f6d3_llm_execution_plan.csv`
- `llm_go_no_go`: `results/tables/finalization_f6d3_llm_go_no_go.csv`
- `readiness_update`: `results/tables/finalization_f6d3_readiness_update.csv`
- `missing_evidence_update`: `results/tables/finalization_f6d3_missing_evidence_update.csv`
- `coverage_impact`: `results/tables/finalization_f6d3_coverage_impact.csv`
- `anomalies`: `results/tables/finalization_f6d3_anomalies.csv`
- `validation`: `results/tables/finalization_f6d3_validation.csv`
