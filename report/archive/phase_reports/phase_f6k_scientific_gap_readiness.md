# F6K Scientific Gap Readiness Gate

Decision token: `f6k_scientific_gap_readiness_complete`

## Preflight

Canonical preflight recorded for this gate:

```text
$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONPATH='.'; .\.venv_audit_recovery\Scripts\python.exe -B -m pytest -q --tb=short -ra -p no:cacheprovider --basetemp results\logs\phase_f6k_cleanup_preflight_tmp -m "cleanup_current and not unsafe_external" tests
151 passed, 1698 deselected in 404.58s
```

No training, LLM inference, API calls, downloads, package changes, raw-output generation, YAML edits, archive edits, or `original_repo/` edits were performed for F6K.

## Readiness Decisions

| Gap | Decision | Minimal order | New requests now |
| --- | --- | --- | --- |
| `profile_generation_effectiveness` | `blocked_missing_frozen_inputs` | `never/currently blocked` | `0` |
| `runtime_cost_characterization` | `execute_now_high_value` | `1` | `0` |
| `lastfm_llm_reranking` | `execute_now_limited_value` | `2 if explicitly authorized as adapted local inference` | `0 in F6K; 7436 for a future run` |

## Direct Answers

Execute now only the observational runtime/cost characterization. It uses existing `phase_f6e4b_runtime_summary.csv`, F6E2 runtime/environment tables, and Plan E cost placeholders, so it requires `0` new requests and no model runtime.

LastFM reranking is useful only if the user authorizes a separate new Ollama inference run. The frozen adapted plan has 1859 instances, 4 cells, 7436 requests, pool size 20, top-10 prompt and parser contracts, verified local Qwen2/Mistral model tags, and protected input hashes. It is not paper-exact because historical raw responses are missing, the paper output-length evidence is unresolved, and only 592/1859 instances have the positive item present.

Profile-generation effectiveness remains blocked. RQ8, F6E1, and the canonical execution matrix show that profile templates exist, but there are no frozen generated profiles, no profile request/output manifest, no frozen profile parser or quality contract, and no homogeneous profile vs no-profile comparison. New executable requests now: `0`.

The definitive current limits to close are: profile-generation effectiveness without a frozen profile workflow; and official/current API cost claims without a network price check. Local monetary cost can be reported as `0` only for API/cash spend when no API is used, not as hardware, electricity, depreciation, or opportunity cost.

The best value/cost activity is runtime/cost characterization because it needs no new inference and improves RQ14 wording immediately.

Minimal execution order: `1` runtime/cost characterization; `2` LastFM reranking adapted only if explicitly authorized; `never/currently blocked` profile generation.

Estimated new executions for F6K: `0`. Estimated future LastFM reranking requests: `7436`. Estimated profile-generation requests: `0` until the workflow is unblocked and frozen.

After F6K, finalization can proceed if the user accepts not executing LastFM reranking before the report. If LastFM evidence is required first, it should be opened as a separate adapted local-inference phase.

## Decision Details

### `profile_generation_effectiveness`

Decision: `blocked_missing_frozen_inputs`.

Evidence basis: `results/tables/phase_f6j_research_question_evidence_matrix.csv#RQ8`, `results/tables/phase_f6e1_prompt_source_inventory.csv`, `results/tables/phase_f6e1_parser_contract.csv`, `results/tables/phase_f6e1_model_strategy_matrix.csv`, `results/tables/canonical_llm_execution_matrix.csv`, and F6H2 lineage limitations.

The project has official profile-related templates, but templates do not create a frozen executable profile workflow. Current artifacts do not provide generated profile outputs, request/response lineage, a parser or profile-quality contract, or a matched profile/no-profile recommendation matrix. Any run now would create new adapted evidence, not a controlled readout of an already frozen profile mechanism.

### `runtime_cost_characterization`

Decision: `execute_now_high_value`.

Evidence basis: `results/tables/phase_f6e4b_runtime_summary.csv`, `results/tables/phase_f6e2_runtime_estimates.csv`, `results/tables/phase_f6e2_runtime_inventory.csv`, `results/tables/plan_e_api_cost_scenarios.csv`, `results/tables/plan_e_local_runtime_estimates.csv`, and F6J RQ14/claim registry entries.

This is ready only as observational characterization. It can summarize observed local latency, throughput, and token-count telemetry, then state that Plan E API cost rows remain placeholders requiring current price verification. It must not claim official current pricing without a price check.

### `lastfm_llm_reranking`

Decision: `execute_now_limited_value`.

Evidence basis: F6G3 frozen request manifest, candidate validation, request/response reconciliation, cell summary, top-10 prompt/parser contracts, F6G2 output-length resolution, F6G2R Ollama preflight, local execution parameter contract, nondeterminism policy, and protected input hashes.

The adapted execution is technically ready if authorized: 1859 instances across 4 cells produce 7436 requests, with retained raw responses required. Its value is limited because existing raw responses are absent, paper-exact comparability is not available, and positive-item presence is incomplete at 592/1859.

## F6K Stop Rules

F6K itself stops at readiness artifacts. It must not start Ollama, generate raw outputs, run LLM requests, train models, download models or data, modify YAML contracts, alter protected hashes, stage paths, or edit `original_repo/`.

