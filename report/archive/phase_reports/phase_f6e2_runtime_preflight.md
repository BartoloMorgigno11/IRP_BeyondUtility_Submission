# Phase F6E2 - Runtime, Hardware and Execution Preflight

Decision: `f6e2_runtime_preflight_complete_with_documented_blocks`.

F6E2 is an offline preflight. It inventories local hardware, runtime, model identity, workloads, prompt sizes, execution risk, and pilot design. No inference, model acquisition, download, API call, or raw-response mutation is performed.

## Initial Repository State

- Expected branch: `plan-d-extensions`.
- Expected HEAD: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`.
- Dirty tracked files are pre-existing and intentionally untouched: `README.md`, `report/dataset_audit.md`, `report/notes.md`.
- F6E2 created only new `phase_f6e2_*` artifacts plus its source and tests.

## Hardware Inventory

- CPU: Intel(R) Core(TM) i9-14900HX (24 physical cores; 32 logical processors; max clock 2200 MHz; x64).
- RAM: 33255288 KiB total; 19571940 KiB free; 35352440 KiB virtual total.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB total; 7866 MiB free; 91 MiB used; driver 595.79; compute capability 8.9; CUDA 13.2 exposed).
- Storage: 1023010664448 total; 185582936064 free; NTFS.

## Runtime Inventory

- Ollama client is installed but server was not reachable in this preflight; a read-only list attempt hit a local log-permission failure and was not repeated.
- PyTorch 2.6.0+cu124 is installed and CUDA-capable in the project venv.
- Transformers, accelerate, bitsandbytes, vLLM, llama.cpp, GPTQ/AWQ runtimes are not installed in the audited environment.

## Local Model Inventory

- `qwen2:7b-instruct`: available_manifest_and_blob; blob bytes `4431388192`; digest `43f7a214e5329f672bb05404cfba1913cbb70fdaa1a17497224e1925046b0ed5`.
- `mistral:7b-instruct`: available_manifest_and_blob; blob bytes `4372811712`; digest `f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f`.
- `llama3:70b-instruct`: missing; blob bytes `0`; digest `missing`.

## Model Identity Alignment

- Qwen2 and Mistral are paper model families with local Ollama adapted executions; exact paper model-version equivalence is not proven.
- Llama 3 70B is unavailable locally and blocked by model, runtime, RAM, and VRAM evidence.
- Closed-source GPT/Claude models remain blocked under the no-API policy.

## F6E1 Matrix Loaded

- F6E1 matrix rows evaluated: `11`.
- Ready/preparable local cells: `8`.
- Blocked cells: `3`.
- LLMSRec remains `reconstructed` and is never classified as exact.

## Request Workload Inventory

- Ready logical instances across ready manifests: `7859`.
- Beauty, Sports, ML-1M ranking/reranking use Plan E 1000-user manifests.
- LastFM ranking uses the F6D3/F6D4A 1859-user manifest. LastFM reranking is inventoried as partial because positive inclusion is not comparable.

## Prompt Token Estimates

- Token counts use deterministic char/4 estimates because exact local tokenizers are not available and no tokenizer download is allowed.
- The estimates distinguish prompt strategy, dataset, task, output budget, and context overflow thresholds.

## Call Multiplicity

- Base, LLMRank, and JSON/index are one-call strategies.
- LLMSRec is represented as a conservative interval of one to two calls per logical request because profile orchestration remains reconstructed.

## Runtime And Feasibility

- Runtime estimates use prior observed Qwen2/Mistral telemetry from frozen local runs when available.
- Llama 3 70B has no local observed timing and is not estimable as an executable local full-matrix cell.
- Existing 7B Ollama models are locally feasible once the Ollama operational gate is passed.

## Llama 3 70B

- FP16, INT8, 6-bit, and 5-bit are blocked by VRAM and RAM margins.
- 4-bit is blocked by RAM/runtime/model availability for stable local full-matrix use.
- 3-bit is classified as technically possible but impractical and still blocked by missing model/runtime.

## Execution Order

1. Analysis-only gate and hash freeze.
2. Ollama operational gate in user session.
3. Tiny JSON/index pilot on installed 7B models to verify resume/parser plumbing.
4. Base prompt pilot on Qwen2/Mistral.
5. LLMRank paired pilot.
6. LLMSRec reconstructed pilot only after scope is explicitly accepted.
7. Full run only after pilot, protected-input checks, and append-only raw policy pass.

## Pilot Design

- Pilot definitions are written to `results/tables/phase_f6e2_pilot_design.csv` only; no raw responses are materialized.
- Pilot rows specify deterministic selection, parser checks, timeout, resume, append-only raw policy, and stop conditions.

## Protected Inputs

- F2 protected inputs checked: `1144`; changed: `0`.
- F6D4A protected/input checks carried forward: `21`; changed: `0`.
- Qwen2 raw hash: `1384a47058ed8dd8c06fc52d6bc425020b1746b60ad11e4bde6153f948a493b4`.
- Mistral raw hash: `7031f37d40092b4f318b5e03273772531ffda37cef0c38bfed32edfef935736a`.

## Manifest

- Scientific content hash: `d17f308f10ba623d4f238e8ff9a5def7efbe71e2bedc20dc53a3c78fdf966174`.
- Hardware fingerprint: `23cf8447c0154204c93521b470a4fb1ffb01ab3366ad10ec0f6a8a6ba520b74b`.
- Runtime metadata that can vary between sessions is separated from scientific table hashes.

## Missing Or Blocked Evidence

- `F6E2-BLOCK-OLLAMA-SERVER`: conditional_runtime_risk for all future local Ollama inference cells.
- `F6E2-BLOCK-LLAMA70B`: blocked_hardware for Llama 3 70B base, LLMRank, LLMSRec.
- `F6E2-BLOCK-CLOSED-MODELS`: blocked_closed_source for GPT and Claude paper models.
- `F6E2-BLOCK-LLMSREC`: blocked_missing_prompt_evidence for LLMSRec strategy all local models.
- `F6E2-BLOCK-LASTFM-RERANK`: partial_manifest for LastFM reranking.
- `F6E2-BLOCK-TOKENIZER`: estimated_not_exact for prompt token estimates.
- `F6E2-BLOCK-WSL`: runtime_evidence_insufficient for Linux-only large-model runtime alternatives.

## Safe Next Action

Do not start full inference from F6E2. The next safe phase is an explicitly approved, append-only pilot after Ollama operational readiness is verified in the real user session.
