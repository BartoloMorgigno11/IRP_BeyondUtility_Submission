# G0 Canonical Local LLM Canary

Decision: `G0_CANONICAL_LLM_CANARY_VALIDATED`

## Model Preflight

model,model_family,available,digest,size_bytes,modified_at,ollama_version,ollama_models_loaded,gpu,ram_gb,python,platform,status
qwen2:7b-instruct,QWEN,true,dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755,4431401491,2026-07-01T11:10:17.2004217+02:00,0.33.2,"[""gemma2:9b""]","NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 595.79",31.714714050292969,3.12.10,Windows-11-10.0.26200-SP0,AVAILABLE_LOCAL
mistral:7b-instruct,MISTRAL,true,6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf,4372824384,2026-07-01T21:25:40.303487+02:00,0.33.2,"[""gemma2:9b""]","NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 595.79",31.714714050292969,3.12.10,Windows-11-10.0.26200-SP0,AVAILABLE_LOCAL
gemma2:9b,GEMMA,true,ff02c3702f322b9e075e9568332d96c0a7028002f1a5a056e0a6784320a4db0b,5443152417,2026-08-24T15:32:42.1502412+02:00,0.33.2,"[""gemma2:9b""]","NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 595.79",31.714714050292969,3.12.10,Windows-11-10.0.26200-SP0,AVAILABLE_LOCAL

## Canary Users

logical_user_order,dataset,logical_user_group,pool_group_filter,user_id,selection_rule,selection_hash,selection_used_recommendation_success,selection_used_candidate_recall,ranking_candidate_hash,reranking_candidate_hash,ranking_candidate_source,reranking_candidate_source,full_history_length,visible_history_length,foodcom_history_class,history_representation,status,reason
1,beauty,PRIMARY,WARM_USERS,A11XAIFA10G7TS,min_sha256_over_protocol_renderable_final_users,00203ab2f1a8172fb172f3dcecbac3e3c01f5e40e17c1ab7b8649fcc1a184047,false,false,cbc439031d7be3c5f1cf71f6f53a285f96b51b63da8b95f2814810c388abee35,26d8af731f41e647294614d317a7a350052096cf0d14808372145a368d0b7d44,results/tables/plan_e_beauty_1000user_ranking_manifest.csv,F1BZQ-BA5E8D1214646491D9A9,11,10,,numbered_title_list,SELECTED,Deterministic final-user selection from renderable frozen inputs.
2,sports,PRIMARY,WARM_USERS,A13AT708PTPIM7,min_sha256_over_protocol_renderable_final_users,0006c08fbcb9a793262f66ae10c6b363ae74e2399e3ac3a1b18abe9a93d90d45,false,false,3eeaa74b5f792ab0507adbe9b2fb932c11b8156d40c26dfd96a9918205b5cef8,2d808dd16d715fc71b20d9857fdbe92f6ebf884814ae065a07e27f8fcec4777e,results/tables/plan_e_sports_1000user_ranking_manifest.csv,F1BZQ-CD6A5D84AE6922A332C6,4,4,,numbered_title_list,SELECTED,Deterministic final-user selection from renderable frozen inputs.
3,ml1m,PRIMARY,WARM_USERS,1846,min_sha256_over_protocol_renderable_final_users,00f100f27dd32612e509dc401ea3ec4a8444b68054cf0533bc31f9b9d2cb41bf,false,false,696e33ad4d16263f7fdcffc3b00c491e6953c1d961eb0faa02abb41ef1af36db,820006a94cb417540973a45f07e324fa9c9b19f3bc0f174d601030a066b1daf9,results/tables/plan_e_ml1m_1000user_ranking_manifest.csv,F1BZQ-E182CEBB4E0B7E0796DE,60,10,,numbered_title_list,SELECTED,Deterministic final-user selection from renderable frozen inputs.
4,lastfm,PRIMARY,WARM_USERS,780,min_sha256_over_protocol_renderable_final_users,002c753daedc04d5c12ac443c73a8d59eded93793fbfe380cc645a09843748d6,false,false,a4e5c0945cf9c106ecc264877ceb2a24a7226e71d29fadf61860987057bfedb9,eb8d185f1b1a22d37eb41bf9014afd0e96a4355963fe369a5e7e5897ad7d8c83,data/processed/finalization_f6c_lastfm_ranking_candidates.csv,F1BZQ-D178ABACC5CC5626E618,38,10,,numbered_title_list,SELECTED,Deterministic final-user selection from renderable frozen inputs.
5,foodcom,FOODCOM_WARM,WARM_USERS,1170119,min_sha256_over_protocol_renderable_final_users,006845a1f6b9e7afbee6b16f8dfbdb56b139312599c41ca03dcf91df5cd3235b,false,false,c1b76a7752dfee1f012aeac854a94b49a415513c532c2561f982a4a740a30b4e,6e7539c2a2201fcc97aad00d370dbae397f530ad723b011e01bcab892e78430c,results/tables/d0a_foodcom_primary1000_candidates.csv,F1BZQ-5DF2A707DEE17FA7C33A,2,2,WARM_USERS,numbered_title_list,SELECTED,Deterministic final-user selection from renderable frozen inputs.
6,foodcom,FOODCOM_TRUE_ZERO_HISTORY,TRUE_ZERO_HISTORY_USERS,2002347947,min_sha256_over_protocol_renderable_final_users,00156c8a71bed9fe813d273d3088484fa9beb99a0cf6f97aeb4a7f5065744330,false,false,98cdd162be561f4e9e2b2008ef853e06190f88c78cb20c7ed664cddddfecefc1,f4a3f8d4d8ac737014ae7f7f835dc819e1ea14ecfe415ffa221197091c6e2f08,results/tables/d0a_foodcom_primary1000_candidates.csv,F1BZQ-5AD21949296B3D3EAA84,0,0,TRUE_ZERO_HISTORY_USERS,(no visible user history),SELECTED,Deterministic final-user selection from renderable frozen inputs.

## Request Matrix

- Planned canonical requests: `72`.
- Tasks: `{"ranking":36,"reranking":36}`.
- Prompts: `{"BASE":36,"LLMRANK":36}`.
- Request ids are canonical extension request ids, not canary-only ids.

## Execution

- Backend successes by model: `{"gemma2:9b":24,"mistral:7b-instruct":24,"qwen2:7b-instruct":24}`.
- Terminal requests: `72` / `72`.

## Strict Parser

- Strict valid requests: `72`.
- Structurally invalid preserved requests: `0`.
- Invalid LLM content is not retried or repaired.

## Runtime

runtime_scope,model,task,prompt_strategy,dataset,observed_requests,mean_seconds,median_seconds,min_seconds,max_seconds,p90_seconds,p95_seconds,requests_per_hour,full_cell_request_count,approx_full_runtime_hours,status,reason
model_task,qwen2:7b-instruct,ranking,ALL,ALL,12,4.3047441499996539,4.2149754999845754,4.0782215999788605,5.3614799000206403,4.3698243599676063,4.8173995499877487,836.28663506059695,11764,14.066947272387758,DIAGNOSTIC_ONLY,Runtime estimate is diagnostic and does not alter the experiment universe.
model_task,mistral:7b-instruct,ranking,ALL,ALL,12,4.4076789583341451,4.3015746000164654,4.1786038000136605,5.5647451999830082,4.5125319899816532,4.9918899499811227,816.75640037100106,11764,14.403315351623023,DIAGNOSTIC_ONLY,Runtime estimate is diagnostic and does not alter the experiment universe.
model_task,gemma2:9b,ranking,ALL,ALL,12,7.8575077916660421,6.9252179000177421,6.5012615999439731,18.698464800021608,7.2372345299576413,12.394969029980702,458.16053836030437,11764,25.676589350322033,DIAGNOSTIC_ONLY,Runtime estimate is diagnostic and does not alter the experiment universe.
model_task,qwen2:7b-instruct,reranking,ALL,ALL,12,4.2233288333372911,4.1941605499887373,4.0738534000120126,4.4236747000250034,4.4158430500363473,4.4194460800325039,852.40816949488294,11764,13.800900109827747,DIAGNOSTIC_ONLY,Runtime estimate is diagnostic and does not alter the experiment universe.
model_task,mistral:7b-instruct,reranking,ALL,ALL,12,4.3043010833353037,4.2381981000071391,4.1396977999829687,4.6950813999865204,4.546728709974559,4.6258505899750162,836.37271889224883,11764,14.06549942898792,DIAGNOSTIC_ONLY,Runtime estimate is diagnostic and does not alter the experiment universe.
model_task,gemma2:9b,reranking,ALL,ALL,12,6.9152322500012815,6.8785370500118006,6.4971694999840111,7.2863123999559329,7.2784470499900626,7.2838134749763412,520.58989052744153,11764,22.597442274726411,DIAGNOSTIC_ONLY,Runtime estimate is diagnostic and does not alter the experiment universe.
model_task_prompt_dataset,qwen2:7b-instruct,ranking,BASE,beauty,1,5.3614799000206403,5.3614799000206403,5.3614799000206403,5.3614799000206403,5.3614799000206403,5.3614799000206403,671.4564014286691,990,1.4744069725056761,DIAGNOSTIC_ONLY,Runtime estimate is diagnostic and does not alter the experiment universe.
model_task_prompt_dataset,mistral:7b-instruct,ranking,BASE,beauty,1,5.5647451999830082,5.5647451999830082,5.5647451999830082,5.5647451999830082,5.5647451999830082,5.5647451999830082,646.92988998148428,990,1.5303049299953273,DIAGNOSTIC_ONLY,Runtime estimate is diagnostic and does not alter the experiment universe.
model_task_prompt_dataset,gemma2:9b,ranking,BASE,beauty,1,18.698464800021608,18.698464800021608,18.698464800021608,18.698464800021608,18.698464800021608,18.698464800021608,192.52917490829728,990,5.1420778200059418,DIAGNOSTIC_ONLY,Runtime estimate is diagnostic and does not alter the experiment universe.
model_task_prompt_dataset,qwen2:7b-instruct,ranking,LLMRANK,beauty,1,4.3480575000285171,4.3480575000285171,4.3480575000285171,4.3480575000285171,4.3480575000285171,4.3480575000285171,827.95593204008662,990,1.1957158125078422,DIAGNOSTIC_ONLY,Runtime estimate is diagnostic and does not alter the experiment universe.
model_task_prompt_dataset,mistral:7b-instruct,ranking,LLMRANK,beauty,1,4.5231901999795809,4.5231901999795809,4.5231901999795809,4.5231901999795809,4.5231901999795809,4.5231901999795809,795.89843469687639,990,1.2438773049943848,DIAGNOSTIC_ONLY,Runtime estimate is diagnostic and does not alter the experiment universe.
model_task_prompt_dataset,gemma2:9b,ranking,LLMRANK,beauty,1,7.0699484000215307,7.0699484000215307,7.0699484000215307,7.0699484000215307,7.0699484000215307,7.0699484000215307,509.19749286841142,990,1.9442358100059209,DIAGNOSTIC_ONLY,Runtime estimate is diagnostic and does not alter the experiment universe.

## Resume

resume_mode,invocation_started_at_utc,invocation_completed_at_utc,planned_requests,completed_canonical_requests,first_run_llm_calls,resume_run_llm_calls,target_resume_run_calls,raw_response_hashes_unchanged,duplicate_request_rows,duplicate_response_rows,no_new_llm_calls_for_completed_requests,status,reason
true,2026-08-30T15:16:18.859Z,2026-08-30T15:16:18.870Z,72,72,72,0,0,true,0,0,true,PASS,Resume mode skipped already completed canonical requests.

## Validation

- Checks: `40`.
- Failures: `0`.

No comparative scientific conclusion is drawn from this six-user canary.
