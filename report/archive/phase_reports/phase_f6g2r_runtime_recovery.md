# F6G2R Runtime Recovery And Execution Preflight

Decision: `f6g2r_blocked_by_contract_conflict`
Recommendation: `inspect_output_contract_conflict`

## Runtime
- CPython: `3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]`
- Architecture: `64bit`
- Venv executable: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_f6g2_py312\Scripts\python.exe`
- Legacy `.venv`, `.venv_repro`, and `.venv_repro_system_site` were treated as immutable inputs.

## Execution Contract
- Frozen logical instances: `1859`
- Frozen requests: `7436`
- Frozen shards: `15`
- Candidate pool sizes: `20`
- F6G1 output lengths: `20`
- F6G2 requested/parser lengths: `20` / `20`
- Contract status: `blocked_by_top20_top10_conflict`

No Ollama generation, model pull, training, CUDA execution, API call, raw JSONL append, staging, or commit was performed.

## Scientific Hash Inputs
- Table hashes are recorded in `results/manifests/phase_f6g2r_runtime_recovery_freeze.json`.
- Report table count: `11`
