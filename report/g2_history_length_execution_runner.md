# G2 History-Length Execution Runner

## PRECONDITION
- Frozen protocol decision: `G2_NESTED_HISTORY_PROTOCOL_FROZEN`.
- Logical applicable requests: `88050`; G1 reuse: `18660`; future new calls: `69390`.

## RUNNER
- `src/extension/g2_history_length_execution.py` is a separated execution adapter over the frozen request universe.
- Dry run writes only pre-execution artifacts and performs zero generation calls.

## CANONICAL IDENTITY
- Identity is inherited from the frozen G2 request universe and includes dataset, user, model, prompt, H, history hash, candidate hash, target, generation contract hash, parser contract hash, and model digest.

## ANALYSIS MEMBERSHIP
- Requests can belong to PRIMARY_B1_GE10, LONG_HISTORY_B2_GE20, SECONDARY_FULL_CURVE_GE20, or FOODCOM_TRUE_ZERO_HISTORY_DIAGNOSTIC without changing canonical identity.

## G1 REUSE
- G1 reused requests are terminal reusable evidence and consume zero Ollama calls.

## REQUEST ACCOUNTING
By dataset:
dataset,logical_applicable_requests,g1_reusable_requests,new_call_budget_total,unresolved_new_requests
beauty,5658,1086,4572,4572
sports,5010,960,4050,4050
ml1m,35958,6000,29958,29958
lastfm,35022,5886,29136,29136
foodcom,6402,4728,1674,1674
By model:
model,logical_applicable_requests,g1_reusable_requests,new_call_budget_total,unresolved_new_requests
qwen2:7b-instruct,29350,6220,23130,23130
mistral:7b-instruct,29350,6220,23130,23130
gemma2:9b,29350,6220,23130,23130
By prompt:
prompt_strategy,logical_applicable_requests,g1_reusable_requests,new_call_budget_total,unresolved_new_requests
BASE,44025,9330,34695,34695
LLMRANK,44025,9330,34695,34695

## MODEL PREFLIGHT
model,model_family,expected_g1_digest,current_local_digest,digest_match,current_model_identity_status,metadata_access_status,llm_generation_call,status
qwen2:7b-instruct,QWEN,dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755,,not_checked,FROZEN_G1_DIGEST_REFERENCE_ONLY,DRY_RUN_NO_OLLAMA_METADATA_CALL,false,DRY_RUN_MODEL_CHECK_DEFERRED_UNTIL_EXECUTION
mistral:7b-instruct,MISTRAL,6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf,,not_checked,FROZEN_G1_DIGEST_REFERENCE_ONLY,DRY_RUN_NO_OLLAMA_METADATA_CALL,false,DRY_RUN_MODEL_CHECK_DEFERRED_UNTIL_EXECUTION
gemma2:9b,GEMMA,ff02c3702f322b9e075e9568332d96c0a7028002f1a5a056e0a6784320a4db0b,,not_checked,FROZEN_G1_DIGEST_REFERENCE_ONLY,DRY_RUN_NO_OLLAMA_METADATA_CALL,false,DRY_RUN_MODEL_CHECK_DEFERRED_UNTIL_EXECUTION

## EXECUTION ORDER
- Physical order: `model_dataset_prompt_history_user`; logical order: `dataset_user_history_prompt_model`.

## GENERATION CONTRACT
- `{"backend_retry_backoff_seconds":[5,15],"backend_retry_count":2,"format":"json","max_attempts":3,"num_predict":256,"scientific_retry_count":0,"stream":false,"structural_invalidity_retry_count":0,"temperature":0.0,"timeout_seconds":150}`

## RETRY / TERMINAL CONTRACT
- Terminal states: `STRICT_VALID;INVALID_FORMAT;INVALID_LENGTH;INVALID_DUPLICATE;INVALID_CANDIDATE;HALLUCINATION_OR_OUT_OF_CANDIDATE;BACKEND_FAILURE`. Backend failures retry up to the frozen backend retry contract; structural invalidity has zero scientific retry.

## RAW-FIRST PERSISTENCE
- Future execution records request intent, calls the backend, persists raw response JSON before parsing, then writes terminal ledger state.

## RESUME
- `--resume` skips G1 reuse and already-terminal G2 rows; interrupted non-terminal requests may be retried.

## ORPHAN POLICY
- Ambiguous terminal evidence blocks; only request-without-response is resumable without repair.

## RETENTION
- Foreground runner artifacts are Git-eligible. Raw responses and large per-request ledgers remain externalized evidence.

## DRY RUN
- Dry run constructs the full universe, reuse inventory, physical order, accounting, model reference rows, and validates zero inference calls.

## TESTS
- Focused tests are in `tests/test_extension_g2_history_length_execution.py`.

## EXECUTION COMMAND
```bash
python src/extension/g2_history_length_execution.py --config configs/g2_history_length_execution.yaml --allow-api-calls --resume --max-new-calls 5000
```

G2_EXECUTION_RUNNER_FROZEN
