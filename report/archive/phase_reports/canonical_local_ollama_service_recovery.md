# Canonical Local Ollama Service Recovery

## 1. Scope and provenance

Phase 5B-P1B restores and verifies only the local Ollama loopback service. Branch `plan-d-extensions`, commit `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests`. The completed-but-blocked Phase 5B-P1 decision remains `backend_blocked` and is not overwritten.

## 2. Executive summary

The installed local Ollama executable was found and a single user-scoped `ollama serve` start restored loopback availability on `127.0.0.1:11434`. `/api/version`, `/api/tags`, and `/api/ps` were available. Exact tags `qwen2:7b-instruct` and `mistral:7b-instruct` were present. No inference endpoint was called.

## 3. Safety boundaries

No public web, external API, model pull, model create/copy/remove, package installation, environment repair, PATH change, registry change, service/task change, prompt execution, pilot resume, or scientific-contract modification was performed.

## 4. Phase 5B-P1 blocked state

Phase 5B-P1 artifacts exist and remain blocked with 60 sample rows, 120 planned requests, 0 attempted requests, and decision `backend_blocked`.

## 5. Ollama installation evidence

Resolved executable: `C:\Users\Bartolo\AppData\Local\Programs\Ollama\ollama.exe`; SHA256 `FFB61F3800D250A080155FE6243C87E6EFFA67B0686546252C965BC8CB736B77`; CLI version evidence `0.31.1`. `where.exe ollama` returned no PATH match, but PowerShell `Get-Command ollama` resolved the installed executable.

## 6. Existing process inspection

Before the start attempt, `Get-Process ollama` found no Ollama process. No Ollama Windows service or scheduled task was found. After the single start, PID `23508` is `ollama.exe` with command line `ollama.exe serve`.

## 7. Port 11434 inspection

Before startup there was no listening port on 11434. After startup, `Get-NetTCPConnection` reported `127.0.0.1:11434` in `Listen` state owned by PID `23508`.

## 8. Initial loopback findings

Initial probes to `http://127.0.0.1:11434/api/version` and `http://localhost:11434/api/version` failed before startup. This matched the Phase 5B-P1 `backend_blocked` condition.

## 9. Server-start procedure

One permitted user-scoped start attempt ran `ollama serve` from the installed executable. stdout and stderr were redirected to `C:\tmp\phase5b_p1b_ollama_stdout.log` and `C:\tmp\phase5b_p1b_ollama_stderr.log`. No service, scheduled task, PATH, registry or environment setting was changed.

## 10. Server-start result

The server became ready and reported version `0.31.1`. The stderr log reported listening on `127.0.0.1:11434`, model-list cache hydration with `models=2`, and no unused blobs removed.

## 11. Administrative endpoint verification

`/api/version` returned `0.31.1`; `/api/tags` returned both local model tags; `/api/ps` returned an empty model list, confirming no model was loaded by this recovery task.

## 12. Qwen2 tag verification

`qwen2:7b-instruct` is present exactly. Digest `dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755`; size `4431401491`; modified `2026-07-01T11:10:17.2004217+02:00`; currently not loaded by `/api/ps`; matches the Phase 5A required tag.

## 13. Mistral tag verification

`mistral:7b-instruct` is present exactly. Digest `6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf`; size `4372824384`; modified `2026-07-01T21:25:40.303487+02:00`; currently not loaded by `/api/ps`; matches the Phase 5A required tag.

## 14. Model digest evidence

Model manifest evidence: `[{"path": "C:\\Users\\Bartolo\\.ollama\\models\\manifests\\registry.ollama.ai\\library\\mistral\\7b-instruct", "sha256": "6577803AA9A036369E481D648A2BAEBB381EBC6E897F2BB9A766A2AA7BFBC1CF", "length": 857, "last_write": "2026-07-01T21:25:40.3034870+02:00"}, {"path": "C:\\Users\\Bartolo\\.ollama\\models\\manifests\\registry.ollama.ai\\library\\qwen2\\7b-instruct", "sha256": "DD314F039B9D54D5553002C906CE50C9FE7242F73F0680ABD04F01C8ECBD2755", "length": 1003, "last_write": "2026-07-01T11:10:17.2004217+02:00"}]`. Phase 5A inventory records required tags but not local digests, so digest identity is newly recorded here rather than asserted from Phase 5A.

## 15. Model immutability

Model directory `C:\Users\Bartolo\.ollama\models` was inspected read-only. Manifest hashes were recorded. Blob inspection was limited to count and aggregate size: `11` blobs, `8804225875` bytes total; no recursive multi-GB blob hashing or file copying was performed.

## 16. Configuration immutability

No process, user, or machine `OLLAMA_*` environment variables were present before startup, and none were created. The server log shows default effective values such as `OLLAMA_HOST=http://127.0.0.1:11434` and `OLLAMA_MODELS=C:\Users\Bartolo\.ollama\models`, but configuration was not edited.

## 17. Whether inference was avoided

No `/api/generate`, `/api/chat`, `/api/embed`, or model prompt endpoint was called. `/api/ps` reported no loaded models. No request or response JSONL was created.

## 18. Recovery decision

`recovery_status=service_ready_models_verified`. The loopback service is reachable and both exact local tags required by Phase 5A are present.

## 19. Required next action

`Phase 5B-P1C — resume the frozen 120-request canonical Qwen2 and Mistral pilot without rebuilding the sample`. This task does not execute that next phase.

## 20. Protected-input immutability

Protected hashes recorded for Phase 5A contracts, Phase 5B-P1 source, sample, request manifest, decision, and paper PDF: `{"paper/Beyond Utility Evaluating LLM As Recommander.pdf": "a143265ae42f027c079766b1c006c438d698897b732634d78a010eb0a62f0f23", "results/tables/canonical_llm_blocked_or_conditional.csv": "6739b89db6bd14cd75f160d6a8b4d84aa51a72c96e9f6706e7b7d7d507de7b04", "results/tables/canonical_llm_cohort_preflight.csv": "3741bda084defc0ff3a5ff1e3219bcba2cb9a0331ac26de0f90b83b9fb062c48", "results/tables/canonical_llm_execution_parameter_contract.csv": "981af1ed218ce00d919f3503ad8edf732ff7cb8c97b34394bc91dc1b7c00f7cf", "results/tables/canonical_llm_model_inventory.csv": "94887062c44bca434b0444ae0f8592fed44636ca9372caa86a8187aa6bb1ac4d", "results/tables/canonical_llm_parser_contract.csv": "7b66bab163cca76f0ca11fb69278791ace0effb3a7c312d34d7acc7f3733856b", "results/tables/canonical_llm_policy_options.csv": "1d2bcc40f51a87a73ab3def3ce33af5c0750d7bda675d511765d121abccd1642", "results/tables/canonical_llm_prompt_contract.csv": "5f0db988f0ea7bca1952f64f49ae5ed538f685e655148354772cd36d535391ed", "results/tables/canonical_llm_prompt_rendering_audit.csv": "06a6752bcb09bfad269cdcd836f91b36ed566957bf8faee8d89af32cc174a236", "results/tables/canonical_local_llm_pilot_decision.csv": "c02c98bdeb8aa0726d80c95da24106893b802ad8b6dcbd2c9bb2aa4e5d15e57f", "results/tables/canonical_local_llm_pilot_request_manifest.csv": "6b8daf36b0213b7ce00c779b78ca191895ded96f52d875abf143305abef072dc", "results/tables/canonical_local_llm_pilot_sample.csv": "9b7241479997ae2ef86d11760f5864688adb4a3b6b9b76143f03d7e1a27712b4", "src/experiments/run_canonical_local_llm_pilot.py": "a13974ce45a34b11d339e0ed0e3b17a462a3eac1bdc274662cdcf5016f4d46ca", "src/llm/canonical_local_llm_runner.py": "5d4222427b0d35abac7bdff1c72a797ca7567f42c82d2bd73600781c5b774124", "tests/test_canonical_local_llm_runner.py": "3f91e076f48070e4d8187beccab372ae71492279d078349acce58e054278cc25"}`. The recovery task created only separate Phase 5B-P1B infrastructure evidence.

## 21. Commands executed

- `git status --short --branch`
- `git branch --show-current`
- `git log -1 --format="%H %s"`
- `where.exe ollama`
- `Get-Command ollama; Get-FileHash ollama.exe`
- `ollama --version`
- `Get-ChildItem Env:OLLAMA* and User/Machine OLLAMA* inspection`
- `Get-Process ollama -ErrorAction SilentlyContinue`
- `Get-CimInstance Win32_Process filtered to ollama.exe`
- `Get-Service / Get-ScheduledTask filtered to Ollama`
- `Get-NetTCPConnection -LocalPort 11434; netstat -ano | Select-String :11434`
- `Invoke-RestMethod http://127.0.0.1:11434/api/version`
- `Invoke-RestMethod http://localhost:11434/api/version`
- `Start-Process ollama.exe serve with stdout/stderr redirected to C:\tmp`
- `Invoke-RestMethod http://127.0.0.1:11434/api/version`
- `Invoke-RestMethod http://127.0.0.1:11434/api/tags`
- `Invoke-RestMethod http://127.0.0.1:11434/api/ps`
- `ollama list`
- `model manifest file listing and SHA256 hashing`
- `blob count and aggregate byte-size inspection`
