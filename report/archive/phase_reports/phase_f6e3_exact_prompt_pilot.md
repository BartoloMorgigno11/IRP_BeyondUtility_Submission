# Phase F6E3 - Ollama Runtime Gate And Exact-Prompt Micro-Pilot

Decision: `f6e3_blocked_by_parser_validity`.

F6E3 materializes a corrective micro-pilot manifest from the descriptive F6E2 pilot design. It executes only base and LLMRank prompts on local Qwen2/Mistral when the Ollama gate passes.

## Scope

- Included: Qwen2 base, Mistral base, Qwen2 LLMRank, Mistral LLMRank.
- Excluded: JSON/index control, LLMSRec reconstructed, Llama 3 70B, GPT/Claude, full run.

## Runtime Gate

- `client_version`: `pass` (ollama version is 0.31.1).
- `process_snapshot`: `info` ({"Id":32120,"StartTime":"\/Date(1785052347468)\/","Path":"C:\\Users\\Bartolo\\AppData\\Local\\Programs\\Ollama\\ollama.exe"}).
- `api_version`: `pass` ({"version":"0.31.1"}).
- `api_tags`: `pass` ({"models":[{"capabilities":["completion","tools"],"details":{"context_length":32768,"embedding_length":4096,"families":["llama"],"family":"llama","format":"gguf).
- `digest_qwen2_7b_instruct`: `pass` (dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755).
- `digest_mistral_7b_instruct`: `pass` (6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf).

## Pilot Execution

- Request manifest rows: `32`.
- Attempted requests: `32`.
- Completed requests: `32`.
- Strict-valid responses: `0`.
- Raw paths are F6E3-specific and append-only.

## Hashes

- Request manifest hash: `f79be097fe09e4020c880edb663c31ea5ab5c1f73b7d019445651f5f17c21814`.
- Scientific content hash: `165636fcaf34eb9161471976d8586e20d9e5c681871c3dc67e84c16e6c428aa0`.
- Qwen2 F6E3 raw hash: `03d5a43d3d2f10c73e50c5086f1e3f6767051d2baff072ac777b3822bcc4305d`.
- Mistral F6E3 raw hash: `c8e57b9dfbde8ed3471ff45e1c272fbf898a48c7364402d40f49df3b60322e49`.

## Missing Or Blocked

- `F6E3-BLOCK-JSON-INDEX-NOT-SELECTED`: `not_executed_selected_scope`.
- `F6E3-BLOCK-LLMSREC-NOT-SELECTED`: `not_executed_selected_scope`.
- `F6E3-BLOCK-LLAMA3-70B`: `blocked_hardware`.
- `F6E3-BLOCK-PARSER`: `f6e3_blocked_by_parser_validity`.

## Full-Run Recommendation

`revise_parser_contract`; do not full-run until title-list parsing passes.
