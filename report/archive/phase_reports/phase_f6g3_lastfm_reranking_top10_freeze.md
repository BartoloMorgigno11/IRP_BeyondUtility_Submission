# F6G3 LastFM Reranking Top-10 Freeze

## 1. Objective

Construct a deterministic corrective offline LastFM reranking freeze under `D_repository_derived_top10_paper_unspecified`.

Decision token: `f6g3_top10_freeze_complete_blocked_no_existing_raw_responses`.

## 2. Source Contract

- Candidate pool size: `20`
- Requested output length: `10`
- Parser expected length: `10`
- Metric cutoff K: `5`
- Paper output length: `paper_unspecified`
- Repository output length: `repository_derived_not_paper_specified`
- Classification: `repository_derived_adapted_protocol`

This phase does not claim paper-exact output-length replication.

## 3. Runtime And Prohibitions

- Runtime: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_audit_recovery\Scripts\python.exe`
- Runtime classification: `defensible_repaired_clone_exact_package_versions`
- Runtime caveat: repaired clone exact package tree, not exact original executable environment.
- No training, LLM inference, Ollama, API calls, downloads, package changes, staging, or commits occurred.

## 4. Source Artifacts

- `results/tables/exactness_frontier_by_component.csv`: role `exactness_frontier_components`, rows `192`, LastFM reranking rows `0`, status `supporting_evidence`
- `results/tables/numeric_cell_exactness_frontier.csv`: role `exactness_frontier_numeric_cells`, rows `187`, LastFM reranking rows `0`, status `supporting_evidence`
- `results/tables/phase_f6e4b_instance_results.csv`: role `existing_llm_instance_results_no_lastfm_reranking`, rows `30492`, LastFM reranking rows `0`, status `no_lastfm_reranking_raw_rows`
- `results/llm/phase_f6e4b_mistral_adapted_full_run_raw.jsonl`: role `existing_raw_llm_output_no_lastfm_reranking`, rows ``, LastFM reranking rows `0`, status `no_lastfm_reranking_raw_rows`
- `results/llm/phase_f6e4b_qwen2_adapted_full_run_raw.jsonl`: role `existing_raw_llm_output_no_lastfm_reranking`, rows ``, LastFM reranking rows `0`, status `no_lastfm_reranking_raw_rows`
- `results/tables/phase_f6g1_candidate_source_model_decisions.csv`: role `reranking_candidate_source_model_audit`, rows `7`, LastFM reranking rows `0`, status `legacy_top20_superseded`
- `results/tables/phase_f6g1_future_logical_instances.csv`: role `legacy_top20_logical_instances`, rows `1859`, LastFM reranking rows `1859`, status `legacy_top20_superseded`
- `results/tables/phase_f6g1_pool_instance_summary.csv`: role `candidate_pool_summary`, rows `1859`, LastFM reranking rows `0`, status `legacy_top20_superseded`
- `results/tables/phase_f6g1_pool_item_provenance.csv`: role `candidate_pool_item_provenance`, rows `37180`, LastFM reranking rows `0`, status `legacy_top20_superseded`
- `results/tables/phase_f6g1_future_request_manifest.csv`: role `legacy_top20_request_manifest`, rows `7436`, LastFM reranking rows `7436`, status `legacy_top20_superseded`
- `results/tables/phase_f6g1_future_shard_manifest.csv`: role `legacy_top20_shard_manifest`, rows `15`, LastFM reranking rows `0`, status `legacy_top20_superseded`
- `results/tables/phase_f6g2_output_length_contract_resolution.csv`: role `contract_resolution`, rows ``, LastFM reranking rows `0`, status `supporting_evidence`
- `results/tables/phase_f6g2_request_execution.csv`: role `blocked_f6g2_request_execution`, rows `7436`, LastFM reranking rows `7436`, status `supporting_evidence`

## 5. Legacy Supersession

F6G1 top-20 logical/request/shard manifests are superseded for execution. F6G3 reuses only candidate identity, cohort identity, user/task identifiers, candidate order, positive-position metadata, model/strategy identity, shard planning, and provenance. F6G2 remains historical blocked evidence because its canonical raw paths are absent.

## 6. Precondition Audit

- LastFM reranking logical instances: `1859`
- Legacy top-20 requests: `7436`
- Corrective top-10 requests: `7436`
- Existing raw rows reconciled: `0`
- Admitted top-10 rows: `0`
- Invalid or blocked rows retained: `7436`

No existing LastFM reranking raw response rows were found. F6E4B/F6E4C contain LastFM ranking rows, not LastFM reranking rows; F6G2 raw outputs are absent.

## 7. Candidate-Source Models

The F6G1 reconstructed reranking candidate pool uses adapted BPRMF, GRU4Rec, SASRec, and LightGCN source models with selected seeds recorded in `phase_f6g1_candidate_source_model_decisions.csv`. Random, MostPop, and BM25-like sources remain diagnostic-only.

## 8. Candidate-Pool Validation

All F6G1 logical instances preserve 20 unique candidate items. Positive items are present for 592 of 1,859 logical instances and absent for 1,267; absent positives remain in the operational denominator.

## 9. Top-10 Derivation

No raw response was admitted, so no canonical top-10 list was produced. The offline transformation for every row is `none_blocked_no_raw_response`. No output was repaired, reordered, supplemented, or generated.

## 10. Metrics

Utility metrics (`HR@5`, `NDCG@5`, mean positive rank, operational utility, conditional utility) are blank and blocked because no row was admitted to the top-10 freeze. Strict-valid and candidate-valid rates are both `0.000000` over the 7,436 expected requests.

Beyond-utility metrics are present as blocked rows with explicit formulas and denominators where applicable; unsupported values are blank, not zero.

## 11. Per-Cell Summary

- `mistral:7b-instruct` / `base_adapted_indexed`: expected `1859`, admitted `0`, blocked `1859`, strict-valid rate `0.000000`
- `mistral:7b-instruct` / `llmrank_adapted_indexed`: expected `1859`, admitted `0`, blocked `1859`, strict-valid rate `0.000000`
- `qwen2:7b-instruct` / `base_adapted_indexed`: expected `1859`, admitted `0`, blocked `1859`, strict-valid rate `0.000000`
- `qwen2:7b-instruct` / `llmrank_adapted_indexed`: expected `1859`, admitted `0`, blocked `1859`, strict-valid rate `0.000000`

## 12. Exactness Frontier

F6G3 addresses `0` exactness-frontier paper rows because the frontier tables contain no `LastFM` + `reranking` paper components or numeric cells. No classification was elevated. The maximum defensible classification for the operational freeze is repository-derived/adapted and blocked pending raw responses.

## 13. Threats To Validity

The candidate pools are reconstructed/adapted rather than paper-exact. The paper does not specify output length. No LastFM reranking raw responses are available, so execution-dependent metrics cannot be computed.

## 14. Deterministic Reproduction

Run from repository root:

```powershell
$env:PYTHONNOUSERSITE='1'
.venv_audit_recovery/Scripts/python.exe src/analysis/plan_f6g3_lastfm_reranking_top10_freeze.py --config configs/phase_f6g3_lastfm_reranking_top10_freeze.yaml
```

Focused verification command for this work package:

```powershell
$env:PYTHONNOUSERSITE='1'
.venv_audit_recovery/Scripts/python.exe -m pytest tests/test_phase_f6g3_lastfm_reranking_top10_freeze.py -q -p no:cacheprovider
```

Final focused-test result: `8 passed`. The determinism test runs the builder twice and compares all F6G3 output hashes, including the report, handoff, freeze manifest, protected-hash manifest, tables, and current context snapshot. The full test suite is excluded because existing tests include training, checkpoint, writer, inference, and network-adjacent paths that are outside this corrective offline freeze.

The deterministic output-hash registry is stored in `results/manifests/phase_f6g3_top10_freeze_manifest.json` under `output_hashes`; the manifest's own self hash is intentionally recorded as `self_reference_excluded_for_determinism`.

## 15. Protected Inputs

Protected input hashes were captured before and after the builder. Changed: `0`; missing: `0`.

## 16. Git State

- Branch: `plan-d-extensions`
- HEAD: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`
- Phase baseline rows: `1632` reconstructed from final status minus `22` F6G3 paths.
- Baseline counts: modified `16`, untracked `1605`.
- Final rows: `1654`
- Final counts: modified `16`, untracked `1627`.

## 17. Files Created

All F6G3 outputs are isolated under `configs/`, `src/`, `tests/`, `results/tables/`, `results/manifests/`, and `report/`.

## 18. Recommended Next Work Package

- `F6G3-NEXT-001`: Obtain explicit authorization for either locating an existing external exact raw-output archive or running a separate new inference phase; do not infer in F6G3.
- `F6G3-NEXT-002`: Keep candidate pools classified as adapted unless exact paper LastFM reranking pools are recovered.
- `F6G3-NEXT-003`: Use F6G3 report and manifest for future LastFM reranking claims.
