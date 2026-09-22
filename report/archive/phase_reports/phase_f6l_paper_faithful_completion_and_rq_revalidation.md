# Phase F6L Paper-Faithful Completion And RQ Revalidation

Decision token: `f6l_execution_matrix_partially_blocked_by_missing_official_artifacts`
Previous checkpoint token: `f6l_paper_faithful_execution_checkpoint_ready`

F6L has completed the final no-inference execution-authorization checkpoint. No Ollama start, LLM inference, API call, download, training, raw-output overwrite, staging, commit, cleanup, or `original_repo/` mutation was performed.

## Checkpoint

- Deviations found: Existing LastFM ranking uses 1859 users and JSON/index top-5 output; existing reranking is adapted with 0 raw rows and positive present only 592/1859.
- Paper-exact contracts recovered: 5-core; leave-one-out; 1000 users; K-S validation; ranking 1 positive + 19 negatives; reranking from BPRMF/GRU4Rec/SASRec/LightGCN; shared candidate order; top-10 title output; profile prompts recovered.
- Contracts still incomplete: LastFM exact sample/seed, LastFM BPRMF/GRU4Rec/SASRec/LightGCN top-K snapshots, final reranking pool identity/candidate ordering, LLMSRec demonstration orchestration, exact provider model versions, full title-normalization equivalence.
- Profile-generation matrix: Beauty/Sports official artifact inventory plus local Qwen2/Mistral profile_generation, history_only, profile_only, profile_plus_history; ML-1M/LastFM profile blocked until cohort contract.
- LastFM ranking matrix: Existing F6D4A reclassified adapted; new 1000-user K-S-valid sample planned for Qwen2/Mistral base and LLMRank; LLMSRec blocked.
- LastFM reranking matrix: Existing F6G1/F6G3 remains adapted/no-raw; paper-faithful reranking request count is 0 because official LastFM SOTA top-K snapshots and the resulting 20-item pool identity were not recovered.
- Initial aggregate request count: `8800`.
- Atomic breakdown request count after official-source recovery audit: `8800`.
- Estimated time: about 21000 seconds / 5.8 hours for authorized open-model checkpoint cells; reranking paper-faithful excluded as blocked.
- Estimated space: approximately 250-500 MB for raw JSONL and derived CSVs, measured during execution.
- Local models: qwen2:7b-instruct and mistral:7b-instruct from prior successful local artifacts; not rechecked by starting Ollama.
- Prompts and versions: template.txt, template_user_profile_generate.txt, template_profile.txt, template_profile_history.txt, llmrank_rf.txt, title-list top-10 parser; local JSON/index remains adapted.
- Paid/closed model blocked cells: GPT-4o-mini/GPT-4o/GPT-4/Claude/GLM closed/API cells are blocked unless exact official outputs are recovered.
- Official recovery outcome: No new download was performed: Beauty/Sports and Beauty SOTA official ZIPs are already present locally with hashes; official GitHub/author-linked sources expose no LastFM sample/seed, LastFM SOTA top-K snapshots, final LastFM reranking pool, or LLMSRec demonstration manifest.
- Final authorization: Authorized after explicit user approval: 8800 open-model requests; blocked: LastFM reranking, LastFM LLMSRec, closed/paid model cells; retained separately: adapted F6D4A/F6G1/F6G3 evidence.

## Atomic Request Composition

- Profile generation: `800` = Beauty/Sports `2` datasets x `200` users x Qwen2/Mistral `2` generators.
- History-only controls: `800` = Beauty/Sports `2` datasets x `200` users x Qwen2/Mistral `2` recommenders.
- Profile-only recommendations: `1600` = Beauty/Sports `2` datasets x `200` users x `2` profile generators x `2` recommenders.
- Profile-plus-history recommendations: `1600` = Beauty/Sports `2` datasets x `200` users x `2` profile generators x `2` recommenders.
- LastFM paper-protocol ranking: `4000` = `1000` users x Qwen2/Mistral `2` models x base/LLMRank `2` strategies.
- LastFM paper-faithful reranking: `0` = blocked because no paper-faithful 20-item reranking pools can be rendered.
- Total recalculated requests: `8800`.

## LastFM Reranking Zero Diagnosis

The immediate hard blocker is not a generic LastFM gap. The paper-faithful reranking prompt needs the concrete 20-item pool built from official BPRMF, GRU4Rec, SASRec, and LightGCN top-K recommendation snapshots on the same 1000-user cohort, with the official aggregation/order. The official notebook exposes the aggregation logic, but the official LastFM source-model top-K snapshots, exact sample/seed, and final candidate-pool artifact were not found. Therefore `paper_faithful_lastfm_reranking_request_count = 0`.

Hard blocker requirements:

- sample ufficiale di 1000 utenti
- seed
- BPRMF recommendations
- GRU4Rec recommendations
- SASRec recommendations
- LightGCN recommendations
- pool finale da 20 candidati
- candidate ordering

## Official Artifact Recovery

- Official artifacts already present locally with hashes: `3`.
- Official artifacts not found after source inspection: `2`.
- New downloads performed: `0`.
- Sources inspected: official GitHub repository, official GitHub `llm_results`, official GitHub `sota_results`, author README Dropbox links for Beauty/Sports, and author-linked GroupLens LastFM raw dataset page.
- The raw GroupLens HetRec LastFM dataset is not a paper-specific 1000-user sample, SOTA top-K snapshot, or final reranking pool, so it does not unblock paper-faithful reranking.

## Execution Command

```powershell
$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONPATH='.'; .\.venv_audit_recovery\Scripts\python.exe -B src\experiments\build_phase_f6l_paper_faithful_completion_and_rq_revalidation.py --config configs\phase_f6l_paper_faithful_completion_and_rq_revalidation.yaml --mode execute --allow-api-calls --local-only --checkpoint-id f6l_execution_matrix_partially_blocked_by_missing_official_artifacts --raw-output-dir results\llm\phase_f6l_execution_<run_id>
```

This is the only command skeleton authorized by the checkpoint contract. It must still be explicitly requested by the user before any inference; the current turn did not execute it.

## Request Count

- Profile generation: `800`.
- History-only controls: `800`.
- Profile-only recommendations: `1600`.
- Profile-plus-history recommendations: `1600`.
- LastFM ranking base and LLMRank on Qwen2/Mistral: `4000`.
- LastFM reranking: `0`.
- Total authorized future open-model requests after explicit user approval: `8800`.

## Authorization Classes

- Paper-protocol cells authorized after explicit approval: `28`.
- Adapted cells retained separately and not execution-authorized: `2`.
- Closed/paid model groups blocked: `1`.
- LLMSRec cells remain blocked because the official demonstration contract was not recovered.

## Evidence Hierarchy

Existing adapted results are preserved and reclassified, but they do not directly confirm or contradict paper numerical claims. New local Beauty/Sports open-model profile cells are `paper_protocol_reproduction_different_open_model`; new LastFM ranking cells are `paper_protocol_reproduction_new_sample`; LastFM reranking remains `blocked_missing_original_specification`.

## Validation

- Validation passed: `19`.
- Validation failed: `0`.

## Stop

`f6l_execution_matrix_partially_blocked_by_missing_official_artifacts`

<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_execute_preflight:start -->
## Authorized Execute Preflight

- Decision token: `f6l_execution_blocked_by_contract_runtime_or_protected_input_regression`.
- Planned requests: `8800`.
- Completed requests: `0`.
- Strict-valid requests: `0`.
- Failed requests: `0`.
- Residual requests: `8800`.
- Passing preflight checks: `15`.
- Failing preflight checks: `2`.
- Failing checks: per_request_manifest_exists_and_unique, lastfm_frozen_sample_artifact.
- Local model preflight: qwen2:7b-instruct=pass:dd314f039b9d; mistral:7b-instruct=pass:6577803aa9a0.

No local model inference was started. The raw output directory was not created because execution must stop before the first request when a frozen per-request manifest or frozen LastFM 1000-user sample is absent.

`f6l_execution_blocked_by_contract_runtime_or_protected_input_regression`
<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_execute_preflight:end -->

<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_contract_materialization:start -->
## Execution Contract Materialization

- Decision token: `f6l_execution_contract_materialization_blocked_by_missing_paper_specification`.
- LastFM sample materialized: `false`.
- LastFM candidate sets materialized: `false`.
- Open-model request manifest materialized: `false`.
- Aggregate authorized requests still valid: `8800`.
- Recovered K-S criterion: high-level paper statement only; not operational.
- Seed policy: author sample seed not recovered; F6C adapted candidate seed `42` is not promoted to F6L because the K-S gate is blocked.

Blocking definitions:

- ks_gate_operational_definition: Exact K-S feature/metric vector, threshold/alpha, and pre-inference gate procedure are not specified by the recovered paper/code/artifacts.
- lastfm_temporal_leave_one_out_order: Paper-faithful temporal ordering for LastFM leave-one-out cannot be certified from user_artists.dat; tag timestamps are not listening timestamps.

No inference, Ollama generation, remote API call, download, training, raw-output directory creation, staging, commit, cleanup, or `original_repo/` mutation was performed.

`f6l_execution_contract_materialization_blocked_by_missing_paper_specification`
<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_contract_materialization:end -->

<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_separated_checkpoints:start -->
## Separated Profile And LastFM Checkpoints

### Profile Checkpoint

- Requests materialized: `4800`.
- Manifest rows: `4800`.
- Unique request IDs: `4800`.
- Models: `qwen2:7b-instruct`, `mistral:7b-instruct`.
- Dependencies: profile-only and profile-plus-history rows point to the matching profile-generation parent request and are acyclic.
- Classification: `paper_protocol_reproduction_different_open_model`.
- Command after separate authorization: `$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONPATH='.'; .\.venv_audit_recovery\Scripts\python.exe -B src\experiments\build_phase_f6l_paper_faithful_completion_and_rq_revalidation.py --config configs\phase_f6l_paper_faithful_completion_and_rq_revalidation.yaml --mode execute-profile --allow-api-calls --local-only --checkpoint-id f6l_profile_execution_ready_lastfm_operationalization_checkpoint_ready --raw-output-dir results\llm\phase_f6l_profile_execution_<run_id>`

### LastFM Checkpoint

- Exact LastFM materialization remains blocked by missing paper specification.
- Local operationalization is preregistered as `paper_protocol_reproduction_under_explicit_operationalization`, not attributed to the authors.
- Full-cohort extension planned requests: `7436`.
- Full-cohort classification: `original_project_extension_full_cohort`.
- Author clarification draft created but not sent.

`f6l_profile_execution_ready_lastfm_operationalization_checkpoint_ready`
<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_separated_checkpoints:end -->

<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_profile_open_model_execution:start -->
## F6L Profile Open-Model Execution

- Decision token: `f6l_profile_open_model_execution_and_rq_revalidation_complete`.
- Run ID: `20260801_010000`.
- Raw output directory: `results/llm/phase_f6l_profile_execution_20260801_010000`.
- Preflight status: `pass`.
- Planned/completed/strict-valid/failed/residual requests: `4800`/`4800`/`807`/`0`/`0`.
- Retry count: `0`.
- Beauty profile deltas: HR@5 deltas mean=-0.018333;min=-0.050000;max=0.020000;n=12.
- Sports profile deltas: HR@5 deltas mean=-0.007500;min=-0.030000;max=0.000000;n=12.
- Qwen2 profile deltas: HR@5 deltas mean=-0.020000;min=-0.050000;max=0.000000;n=12.
- Mistral profile deltas: HR@5 deltas mean=-0.005833;min=-0.030000;max=0.020000;n=12.
- Runtime wall-clock seconds: `63401.901970`; throughput requests/hour: `272.547029`.
- Runtime resources: CPU `start=unavailable;end=unavailable`, RAM `start=unavailable;end=unavailable`, GPU `unavailable`, VRAM `unavailable`.
- Evidence classification: `paper_protocol_reproduction_different_open_model`.
- Closed/paid model limitation: no exact GPT-4o-mini/GPT/Claude/GLM evidence is claimed; no remote API was called.
- Profile RQ rows updated: `RQ6;RQ8;RQ9;RQ14`.

`f6l_profile_open_model_execution_and_rq_revalidation_complete`
<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_profile_open_model_execution:end -->

<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_lastfm_execution_contracts:start -->
## LastFM Execution Contracts

- Decision token: `f6l_lastfm_execution_contracts_ready_for_inference_authorization`.
- Exact paper replica: `blocked_missing_original_specification`.
- 1000-user local sample: `paper_protocol_reproduction_under_explicit_operationalization`.
- Full 1859-user local cohort: `original_project_extension_full_cohort`.
- Selected local K-S seed: `20260801`.
- K-S attempts: `1`; K-S tests in selected family: `8`.
- Operationalized sample users: `1000`; candidate rows: `20000`.
- Operationalized ranking request contracts: `4000`.
- Full-cohort ranking request contracts: `7436`.
- Operationalized reranking pool rows: `20000`; future reranking requests if separately authorized: `4000`.
- Full-cohort reranking pool rows: `37180`; future reranking requests if separately authorized: `7436`.
- Validation: `23` passed, `0` failed.
- No inference, Ollama generation, remote API, paid API, download, training, raw-output directory creation, staging, commit, cleanup, or `original_repo/` mutation was performed.

### Future Commands

- Revalidate/materialize these LastFM contracts only:
`$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONPATH='.'; .\.venv_audit_recovery\Scripts\python.exe -B src\experiments\build_phase_f6l_paper_faithful_completion_and_rq_revalidation.py --config configs\phase_f6l_paper_faithful_completion_and_rq_revalidation.yaml --mode lastfm-contracts`
- No inference command is recorded here because this turn was contract-only. A future runner must consume the CSV manifests only after explicit authorization.

`f6l_lastfm_execution_contracts_ready_for_inference_authorization`
<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_lastfm_execution_contracts:end -->

<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_lastfm_operationalized_ranking_execution:start -->
# Phase F6L LastFM Operationalized Ranking Execution

- Decision token: `f6l_lastfm_operationalized_ranking_execution_and_rq_revalidation_complete`.
- Run ID: `20260802_184502`.
- Raw output directory: `results/llm/phase_f6l_lastfm_operationalized_ranking_20260802_184502`.
- Evidence class: `paper_protocol_reproduction_under_explicit_operationalization`.
- Exact paper LastFM remains `blocked_missing_original_specification`.
- Full cohort and reranking were not executed.

## Metrics

- `qwen2:7b-instruct` `base/OpenP5 title-list` `HR@5` = `0`.
- `qwen2:7b-instruct` `base/OpenP5 title-list` `NDCG@5` = `0`.
- `qwen2:7b-instruct` `base/OpenP5 title-list` `validity_rate` = `0`.
- `qwen2:7b-instruct` `base/OpenP5 title-list` `hallucination_rate` = `1`.
- `qwen2:7b-instruct` `base/OpenP5 title-list` `duplicate_rate` = `0`.
- `qwen2:7b-instruct` `LLMRank recency` `HR@5` = `0`.
- `qwen2:7b-instruct` `LLMRank recency` `NDCG@5` = `0`.
- `qwen2:7b-instruct` `LLMRank recency` `validity_rate` = `0`.
- `qwen2:7b-instruct` `LLMRank recency` `hallucination_rate` = `0`.
- `qwen2:7b-instruct` `LLMRank recency` `duplicate_rate` = `0`.
- `mistral:7b-instruct` `base/OpenP5 title-list` `HR@5` = `0`.
- `mistral:7b-instruct` `base/OpenP5 title-list` `NDCG@5` = `0`.
- `mistral:7b-instruct` `base/OpenP5 title-list` `validity_rate` = `0`.
- `mistral:7b-instruct` `base/OpenP5 title-list` `hallucination_rate` = `0.05`.
- `mistral:7b-instruct` `base/OpenP5 title-list` `duplicate_rate` = `0`.
- `mistral:7b-instruct` `LLMRank recency` `HR@5` = `0`.
- `mistral:7b-instruct` `LLMRank recency` `NDCG@5` = `0`.
- `mistral:7b-instruct` `LLMRank recency` `validity_rate` = `0`.
- `mistral:7b-instruct` `LLMRank recency` `hallucination_rate` = `0`.
- `mistral:7b-instruct` `LLMRank recency` `duplicate_rate` = `0`.

## Limits

- This is not the original paper sample and not an exact paper numerical replica.
- Closed/remote models, LLMSRec, full cohort, and reranking remain outside this run.
<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_lastfm_operationalized_ranking_execution:end -->
