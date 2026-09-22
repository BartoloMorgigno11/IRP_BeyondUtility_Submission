# Plan D LLM Model Registry and Backend Feasibility Audit

D5B was a documentation and local audit phase only. D5C0 reran the backend audit after manual Ollama installation and manual `qwen2:7b-instruct` pull. No LLM inference was run, no LLM recommendations were generated, and no API calls/downloads were performed.

The goal is methodological traceability before any LLM experiment. The registry records which paper LLMs can be reproduced exactly, reconstructed with a local open model, approximated with a proxy, or only documented as blocked/conditional.

## Paper LLMs Covered

- `gpt-3.5-turbo-0125`
- `gpt-4o-mini-2024-07-18`
- `gpt-4o-2024-08-06`
- `claude-3-haiku-20240307`
- `llama-3-70b-instruct`
- `qwen-2-7b-instruct`
- `mistral-7b-instruct-v0.2`

## Reproduction Levels

Exact reproduction means using the same model identifier or a verifiably still-available API snapshot. Reconstructed local reproduction means using an open-weight family/model path that is close enough for a clearly labeled diagnostic, without claiming exact numerical reproduction. Proxy reproduction means a replacement model is used because the exact model is unavailable; this must be labeled non-exact.

No exact LLM reproduction is claimed in D5B.

## Recommended Local Path

1. Use Ollama as the local backend.
2. Start with `qwen2:7b-instruct`.
3. Add `mistral:7b-instruct-v0.2-q4_0` or `mistral:7b-instruct-v0.2-q4_K_M` as the second local model.

Qwen is the first local target because it is a 7B open local model, feasible in quantized form on the RTX 4060 8GB planning target, and gives the project the shortest path to D5C smoke testing and D5D ranking diagnostics. Mistral is the second target because it is another feasible 7B instruct model and is useful for comparing hallucination and output-validity behavior against Qwen.

The following commands are recommendations only and were not executed:

```powershell
ollama pull qwen2:7b-instruct
ollama pull mistral:7b-instruct-v0.2-q4_0
ollama pull mistral:7b-instruct-v0.2-q4_K_M
```

`llama-3-70b-instruct` is not feasible locally on an RTX 4060 8GB target. Even though it is open weight, practical evaluation requires cloud hardware or a provider API, so it is documented as `blocked_local_hardware` / cloud-or-API required.

## Closed-Source API Path

- `gpt-4o-mini-2024-07-18` is the best first optional OpenAI API model if the user explicitly approves API access and budget.
- `gpt-4o-2024-08-06` is stronger but more expensive, so it should be limited to a small diagnostic subset if approved.
- `gpt-3.5-turbo-0125` is a legacy model whose availability must be verified with an API model-list check before any exact claim.
- `claude-3-haiku-20240307` is retired for exact reproduction; a current Claude Haiku replacement would be a non-exact proxy and would require explicit approval.

No OpenAI API calls and no Anthropic/Claude API calls were made in D5B.

## Local Backend Audit Result

Ollama command is available and `qwen2:7b-instruct` is present in `ollama list`. The local Ollama backend is available for D5C, and the Qwen smoke test is now `pending_smoke_test`.

Additional local audit facts:

- Ollama version: `0.31.1`.
- Qwen local model status: `qwen2:7b-instruct` is present.
- Mistral local model status: not found in `ollama list`.
- Torch importable: `true`.
- CUDA status from torch: available (`NVIDIA GeForce RTX 4060 Laptop GPU`).
- LastFM is not required here; D5B only audits LLM model/backend feasibility and does not run LastFM experiments.

D5C can now start as a smoke test only. No ranking, reranking, position-bias, history-sensitivity, or output-validity LLM experiment has been run yet.

## Recommended Next Step

Proceed to the D5C Qwen smoke test, still without claiming any LLM recommendation results. The smoke test should run before any 20-user ranking diagnostic.

If Ollama is missing, install Ollama manually, then rerun D5B backend audit or proceed to D5C smoke test after pulling Qwen. If Ollama exists but Qwen is missing, the user may manually run `ollama pull qwen2:7b-instruct`. If Qwen exists, proceed to the D5C smoke test.

## Documentation Claims

- No exact LLM reproduction is claimed yet.
- Local open-model reproduction is planned.
- Closed API reproduction is conditional.
- Retired or unavailable models are documented honestly.
- The previous local backend blocker is resolved for Qwen smoke testing only.
- No LLM inference, API call, download, model pull, recommender training, or raw LLM output generation occurred in D5C0.

## Generated Artifacts

- `results/tables/plan_d_llm_model_registry.csv`
- `results/tables/plan_d_llm_backend_audit.csv`
- `results/tables/plan_d_llm_task_roadmap.csv`
