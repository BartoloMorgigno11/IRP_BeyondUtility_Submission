# Phase F6E3B - Adapted Indexed Output-Contract Micro-Pilot

Decision: `f6e3b_adapted_output_contract_pilot_complete_gate_passed`.
Recommendation: `proceed_to_adapted_full_run`.

F6E3B reuses the frozen F6E3 logical instances and changes only the output contract to strict JSON with one-based candidate indices.

## Scope

- Included: Qwen2 Base, Mistral Base, Qwen2 LLMRank, Mistral LLMRank.
- Excluded: full run, LLMSRec, JSON/index standalone, Llama3, GPT, Claude, model pull, remote APIs.

## Execution

- Manifest rows: `32`.
- Attempted requests: `32`.
- Completed requests: `32`.
- Strict-valid responses: `31`.
- Raw files are F6E3B-specific and append-only.

## Cell Summary

- `mistral:7b-instruct` `paper_base_openp5_parametric_v1`: strict `8/8`, gate `pass`.
- `mistral:7b-instruct` `paper_llmrank_recency_parametric_v1`: strict `7/8`, gate `pass`.
- `qwen2:7b-instruct` `paper_base_openp5_parametric_v1`: strict `8/8`, gate `pass`.
- `qwen2:7b-instruct` `paper_llmrank_recency_parametric_v1`: strict `8/8`, gate `pass`.

## Runtime Gate

- `client_version`: `pass` (ollama version is 0.31.1).
- `process_snapshot`: `info` ({"Id":25176,"StartTime":"\/Date(1785063226894)\/","Path":"C:\\Users\\Bartolo\\AppData\\Local\\Programs\\Ollama\\ollama.exe"}).
- `api_version`: `pass` ({"version":"0.31.1"}).
- `api_tags`: `pass` ({"models":[{"capabilities":["completion","tools"],"details":{"context_length":32768,"embedding_length":4096,"families":["llama"],"family":"llama","format":"gguf).
- `digest_qwen2_7b_instruct`: `pass` (dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755).
- `digest_mistral_7b_instruct`: `pass` (6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf).

## Hashes

- Request manifest hash: `b7c62d543702f22f186eb08863f3e422ec88db43783f124c9162a8b87166c256`.
- Scientific content hash: `079a085837d1dade519d1e28e616a9ee2484450dd1c305c4e3261179b39889b1`.
- Qwen2 F6E3B raw hash: `6d70eca3edaeee1174e066404ccc8b51acc5f1886aa75ce7b185fcc2d58a96bc`.
- Mistral F6E3B raw hash: `aad02404ba238f1edce0b7558ae9cbedd148a021799f6b7b67bdb2a1bcec77ad`.

## Missing Or Blocked

- `F6E3B-BLOCK-JSON-INDEX-STANDALONE`: `not_executed_selected_scope`.
- `F6E3B-BLOCK-LLMSREC`: `not_executed_selected_scope`.
- `F6E3B-BLOCK-REMOTE-OR-LARGE-MODELS`: `not_authorized`.
- `F6E3B-BLOCK-MODEL-OUTPUTS`: `f6e3b_adapted_output_contract_pilot_failed_model_outputs`.
