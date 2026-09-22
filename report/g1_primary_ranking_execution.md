# G1 Primary Ranking Execution

## PRECONDITION
- Scientific base checkpoint: `5dfec10537a8acd1404c586323a3935add48bc2a`.
- Execution code commit: `3d7f3add4a86895335c7a2d71f4f2d5103bd127f`.
- Base checkpoint ancestor of execution commit: `true`.
- Decision: `G1_PRIMARY_RANKING_VALIDATED`.
- Validation failures: `0`.

## MODEL PREFLIGHT
model,model_tag,model_family,available,digest,g0_expected_digest,current_local_digest,digest_match,g0_model_identity_status,current_model_identity_status,metadata_access_status,metadata_error,ollama_base_url,ollama_host_env,size_bytes,modified_at,parameter_size,quantization_level,context_length,capabilities,ollama_version,ollama_version_error,ollama_ps_error,ollama_models_loaded,local_metadata_request_count,llm_generation_call,gpu,ram_gb,python,platform,status
qwen2:7b-instruct,qwen2:7b-instruct,QWEN,true,dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755,dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755,dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755,true,G0_PROVENANCE_AVAILABLE,MODEL_AVAILABLE_MATCHED,LOCAL_METADATA_OK,,http://127.0.0.1:11434,,4431401491,2026-07-01T11:10:17.2004217+02:00,7.6B,Q4_0,32768,"[""completion"",""tools""]",0.33.2,,,"[""gemma2:9b""]",3,false,"NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 595.79",31.714714050292969,3.12.10,Windows-11-10.0.26200-SP0,MODEL_AVAILABLE_MATCHED
mistral:7b-instruct,mistral:7b-instruct,MISTRAL,true,6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf,6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf,6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf,true,G0_PROVENANCE_AVAILABLE,MODEL_AVAILABLE_MATCHED,LOCAL_METADATA_OK,,http://127.0.0.1:11434,,4372824384,2026-07-01T21:25:40.303487+02:00,7.2B,Q4_K_M,32768,"[""completion"",""tools""]",0.33.2,,,"[""gemma2:9b""]",3,false,"NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 595.79",31.714714050292969,3.12.10,Windows-11-10.0.26200-SP0,MODEL_AVAILABLE_MATCHED
gemma2:9b,gemma2:9b,GEMMA,true,ff02c3702f322b9e075e9568332d96c0a7028002f1a5a056e0a6784320a4db0b,ff02c3702f322b9e075e9568332d96c0a7028002f1a5a056e0a6784320a4db0b,ff02c3702f322b9e075e9568332d96c0a7028002f1a5a056e0a6784320a4db0b,true,G0_PROVENANCE_AVAILABLE,MODEL_AVAILABLE_MATCHED,LOCAL_METADATA_OK,,http://127.0.0.1:11434,,5443152417,2026-08-24T15:32:42.1502412+02:00,9.2B,Q4_0,8192,"[""completion""]",0.33.2,,,"[""gemma2:9b""]",3,false,"NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 595.79",31.714714050292969,3.12.10,Windows-11-10.0.26200-SP0,MODEL_AVAILABLE_MATCHED

## REQUEST UNIVERSE
- Canonical primary-ranking requests: `29292`.
- Dataset request counts: `{"beauty":5940,"foodcom":6000,"lastfm":6000,"ml1m":6000,"sports":5352}`.
- Cells: `30`.
- Canonical logical order: `dataset_user_prompt_model`.
- Physical execution order: `model_dataset_prompt_user`.

## G0 REUSE
- Reused ranking requests: `36`.
- Identity mismatches: `0`.

## EXECUTION ACCOUNTING
- Terminal requests: `29292` / `29292`.
- New G1 terminal requests: `29256` / `29256`.
- Remaining: `0`.

## BEAUTY
- Terminal requests: `5940` / `5940`.
## SPORTS
- Terminal requests: `5352` / `5352`.
## ML1M
- Terminal requests: `6000` / `6000`.
## LASTFM
- Terminal requests: `6000` / `6000`.
## FOOD.COM
- Terminal requests: `6000` / `6000`.
## QWEN
- Terminal requests: `9764`.
## MISTRAL
- Terminal requests: `9764`.
## GEMMA
- Terminal requests: `9764`.
## BASE
- Terminal requests: `14646`.
## LLMRANK
- Terminal requests: `14646`.
## STRICT VALIDITY
- Strict valid: `29024`.
- Structurally invalid or nonterminal: `268`.
- Parse status counts: `{"duplicate_indices":30,"invalid_length":238,"parsed_valid":29024} `.
## UTILITY
- Utility rows: `90`; computed: `90`.
## BEYOND-UTILITY
- Beyond-utility rows: `420`.
- Computed rows: `414`.
## FOOD.COM WARM/COLD
- Diagnostic rows by scope: `{"ALL_USERS":24,"TRUE_ZERO_HISTORY_USERS":24,"WARM_USERS":24}`.
## RUNTIME
- Requests timed: `29292`; total seconds: `47308.962921600265`; throughput/hour: `2228.989888760661`.
## FAILURE / RETRY AUDIT
- Retries: `0`; backend failures: `0`; structurally invalid responses: `268`.
## RESUME
resume_mode,invocation_started_at_utc,invocation_completed_at_utc,planned_requests,g0_reused_requests,completed_new_g1_requests,completed_canonical_requests,first_run_llm_calls,resume_run_llm_calls,target_resume_run_calls,raw_response_hashes_unchanged,duplicate_request_rows,duplicate_response_rows,no_new_llm_calls_for_completed_requests,status,reason
true,2026-09-01T09:11:23.263Z,2026-09-01T09:11:29.604Z,29292,36,29256,29292,29256,0,0,true,0,0,true,PASS,Resume mode skipped all completed canonical G1 requests and left raw hashes unchanged.
## BUDGET INVARIANCE
- F0 conceptual budget is checked in `g1_primary_ranking_budget_invariance_manifest.json`.
## TESTS
- Focused G1 tests are in `tests/test_extension_g1_primary_ranking_execution.py`.
## FILES
- Raw dir: `results/llm/extension_canonical/g1_primary_ranking`.
- Request universe: `results/tables/g1_primary_ranking_request_universe.csv`.
- Report: `report/g1_primary_ranking_execution.md`.
## RETENTION / PROVENANCE
- Retention manifest: `results/manifests/g1_primary_ranking_retention_manifest.json`.
- Scientific foreground retained for Git includes the G1 execution/provenance manifests plus the completeness, utility, beyond-utility, complete-case, Food.com warm/cold, reliability, runtime, and validation tables.
- Raw and per-request evidence remains externalized: `results/llm/extension_canonical/g1_primary_ranking/**` and the bulk execution/audit CSVs listed in the retention manifest.
- External archive bundle: `archive_external/g1_primary_ranking/G1_primary_ranking_3d7f3add_external_evidence.zip`.
- External archive SHA256: `1da06322b966afde096681189bdc923ff7a34288ff899f94f564595fdd11ab8a`; entries: `29266`; uncompressed bytes: `796950141`.
## GIT STATUS
- Status at report render: `clean`.
## NEXT STAGE
- Do not start G2 history, G3 CandDif, or G4 reranking from this phase.

G1_PRIMARY_RANKING_VALIDATED
