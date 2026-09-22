# Plan E CPU LightGCN Beauty and Sports Execution and Same-Cohort Baseline Comparison

## 1. Scope and provenance

- Branch: `plan-d-extensions`.
- Commit: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests`.
- Scope: CPU-only Beauty/Sports LightGCN execution plus same-cohort consolidation with Random, MostPop, BM25-like, BPRMF, GRU4Rec, SASRec and ML-1M Phase 3M LightGCN evidence.

## 2. Executive summary

- CPU LightGCN ranking rows: `3954`.
- CPU LightGCN metric rows: `8`.
- All-dataset LightGCN metric rows: `12`.
- Seven-baseline same-cohort metric rows: `84`.
- Paired comparison rows: `72`.

## 3. Scientific classification

The results are `adapted_paper_like_reconstructed`, not exact original-paper reproduction, because they use reconstructed Plan C data, Plan E2 strict cohorts and local LightGCN code.

## 4. Protected-input boundaries

The CPU runner verified protected hashes before and after execution and did not modify Phase 2A canonical candidate files, reconstructed interactions, prior neural rankings, historical Plan outputs or `original_repo/`.

## 5. Phase 3M CUDA pilot status

ML-1M remains `completed_with_determinism_warning`; its primary rankings and metrics are reused from `results/tables/plan_e_lightgcn_ml1m_pilot_rankings.csv` and `results/tables/plan_e_lightgcn_ml1m_pilot_metrics.csv`.

## 6. Phase 3M-D1 findings

The D1 diagnostic classified ML-1M CUDA LightGCN as `cuda_sparse_training_nondeterministic` and did not approve CUDA scale-up.

## 7. Phase 3M-D2 root-cause findings

The D2 decision classified the first true CUDA divergence as `cuda_sparse_forward_tensor_nondeterminism` at `layer_1_all_embeddings`; Adam was not the root cause.

## 8. Why CPU-only scale-up was selected

CPU-only scale-up was selected because D2 approved `approve_cpu_only_scale_up` while explicitly not approving the existing CUDA pipeline for new Beauty/Sports LightGCN runs.

## 9. CPU execution environment

- `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe`; Python `3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]`; torch `2.6.0+cu124`; CPU `Intel64 Family 6 Model 183 Stepping 1, GenuineIntel`.

## 10. CPU thread and determinism settings

The CPU runner forced one torch compute thread and one interop thread; manifest thread values are `1` and `1`.

## 11. Controlled configuration

Both CPU datasets used seed `42`, embedding dimension `64`, two propagation layers, Adam, batch size `4096`, five epochs, learning rate `0.001`, regularization `0.0` and train-universe negative sampling.

## 12. Beauty training graph

- Train edges `176139`, user nodes `22363`, item nodes `12092`, total nodes `34455`, directed edges `352278`.

## 13. Sports training graph

- Train edges `260739`, user nodes `35598`, item nodes `18347`, total nodes `53945`, directed edges `521478`.

## 14. Training-edge isolation

The graph summary records train-only interaction paths and hashes; no validation, test, held-out positive or candidate-derived edges were added.

## 15. Duplicate-edge policy

Beauty duplicate train edges: `0`. Sports duplicate train edges: `0`. Both graphs use deterministic user-item pair deduplication.

## 16. Bipartite mappings and offsets

Beauty node-map rows: `34455`. Sports node-map rows: `53945`. Item offsets equal the per-dataset user-node counts.

## 17. Graph normalization

Both graphs use `D^-1/2 A D^-1/2` and store coalesced `torch.sparse_coo` adjacency evidence with hash columns in `results/tables/plan_e_lightgcn_beauty_sports_cpu_graph_summary.csv`.

## 18. CPU sparse-operation validation

CPU sparse repeat checks passed for Beauty and Sports; `cpu_sparse_repeat_exact=true` in both graph-summary rows.

## 19. Negative-sampling contract

The manifest records one deterministic negative schedule hash per dataset; negatives are sampled from the train item universe while excluding the user's train positives.

## 20. Strict graph-compatible cohorts

Combined cohort rows: `4000`. Included rows: `3954`. Full cohort ID: `plan_e2_lightgcn_strict_connected_full`. Matched cohort ID: `plan_e2_lightgcn_strict_connected_matched`.

## 21. Beauty exclusions

Beauty excluded rows: `34`. These rows remain in the cohort audit but were not scored.

## 22. Sports exclusions

Sports excluded rows: `12`. These rows remain in the cohort audit but were not scored.

## 23. Beauty CPU primary training

- Duration `141.754` seconds; epochs `5`; final loss `0.460060831498`; status `completed_exact_cpu_repeat`.

## 24. Beauty CPU repeat training

- Duration `150.556` seconds; epochs `5`; final loss `0.460060831498`; status `completed_exact_cpu_repeat`.

## 25. Beauty exact determinism

Beauty determinism checks passed: `16` of `16`.

## 26. Sports CPU primary training

- Duration `279.553` seconds; epochs `5`; final loss `0.387453840114`; status `completed_exact_cpu_repeat`.

## 27. Sports CPU repeat training

- Duration `333.868` seconds; epochs `5`; final loss `0.387453840114`; status `completed_exact_cpu_repeat`.

## 28. Sports exact determinism

Sports determinism checks passed: `16` of `16`.

## 29. CPU runtime and memory

Combined history rows: `20`. Process RSS values are recorded per epoch in `results/tables/plan_e_lightgcn_beauty_sports_cpu_training_history.csv`.

## 30. Checkpoints and metadata

Primary checkpoints retained: `2`; repeat checkpoints retained: `0`.

## 31. Beauty ranking results

- Amazon Beauty ranking plan_e2_lightgcn_strict_connected_full: rows `983`, HR@5 `0.593082400814`, NDCG@5 `0.444716299372`.

## 32. Beauty reranking-candidate results

- Amazon Beauty reranking plan_e2_lightgcn_strict_connected_full: rows `983`, HR@5 `0.593082400814`, NDCG@5 `0.444716299372`.

## 33. Sports ranking results

- Amazon Sports ranking plan_e2_lightgcn_strict_connected_full: rows `994`, HR@5 `0.611670020121`, NDCG@5 `0.454488270841`.

## 34. Sports reranking-candidate results

- Amazon Sports reranking plan_e2_lightgcn_strict_connected_full: rows `994`, HR@5 `0.611670020121`, NDCG@5 `0.454488270841`.

## 35. Beauty full strict metrics

- Amazon Beauty ranking plan_e2_lightgcn_strict_connected_full: rows `983`, HR@5 `0.593082400814`, NDCG@5 `0.444716299372`. - Amazon Beauty reranking plan_e2_lightgcn_strict_connected_full: rows `983`, HR@5 `0.593082400814`, NDCG@5 `0.444716299372`.

## 36. Beauty matched strict metrics

- Amazon Beauty ranking plan_e2_lightgcn_strict_connected_matched: rows `977`, HR@5 `0.590583418628`, NDCG@5 `0.442573452176`. - Amazon Beauty reranking plan_e2_lightgcn_strict_connected_matched: rows `977`, HR@5 `0.590583418628`, NDCG@5 `0.442573452176`.

## 37. Sports full strict metrics

- Amazon Sports ranking plan_e2_lightgcn_strict_connected_full: rows `994`, HR@5 `0.611670020121`, NDCG@5 `0.454488270841`. - Amazon Sports reranking plan_e2_lightgcn_strict_connected_full: rows `994`, HR@5 `0.611670020121`, NDCG@5 `0.454488270841`.

## 38. Sports matched strict metrics

- Amazon Sports ranking plan_e2_lightgcn_strict_connected_matched: rows `907`, HR@5 `0.610804851158`, NDCG@5 `0.454440790785`. - Amazon Sports reranking plan_e2_lightgcn_strict_connected_matched: rows `907`, HR@5 `0.610804851158`, NDCG@5 `0.454440790785`.

## 39. Graph-degree findings

Degree summaries are recorded in the graph, ranking and metric tables. Mean candidate item degree is carried into all LightGCN metric rows.

## 40. ML-1M Phase 3M integration

ML-1M rows in `results/tables/plan_e_lightgcn_all_datasets_metrics.csv` exactly copy HR@5 and NDCG@5 from Phase 3M source metrics.

## 41. Mixed-device evidence boundary

Beauty/Sports are CPU exact-repeat evidence. ML-1M is retained CUDA primary evidence with a documented determinism warning and D2 root-cause boundary.

## 42. All-dataset LightGCN summary

All-dataset LightGCN metric rows: `12` across Beauty, Sports and ML-1M.

## 43. Seven-baseline same-cohort reconstruction

Seven baselines are consolidated on LightGCN strict connected full/matched cohorts: Random, MostPop, BM25-like, BPRMF, GRU4Rec, SASRec and LightGCN.

## 44. Random versus LightGCN

- Rows `12`; mean HR@5 delta `0.379026727465`; mean NDCG@5 delta `0.310749697615`; positive-rank wins/equal/losses `8445/577/2624`.

## 45. MostPop versus LightGCN

- Rows `12`; mean HR@5 delta `0.0798594801844`; mean NDCG@5 delta `0.0731663781252`; positive-rank wins/equal/losses `4856/4946/1844`.

## 46. BM25-like versus LightGCN

- Rows `12`; mean HR@5 delta `0.0904012328829`; mean NDCG@5 delta `0.068275959732`; positive-rank wins/equal/losses `5862/1398/4386`.

## 47. BPRMF versus LightGCN

- Rows `12`; mean HR@5 delta `0.0499075728633`; mean NDCG@5 delta `0.0360787253531`; positive-rank wins/equal/losses `3980/4068/3598`.

## 48. GRU4Rec versus LightGCN

- Rows `12`; mean HR@5 delta `-0.0562943637768`; mean NDCG@5 delta `-0.0591699548876`; positive-rank wins/equal/losses `3846/3176/4624`.

## 49. SASRec versus LightGCN

- Rows `12`; mean HR@5 delta `-0.113426922696`; mean NDCG@5 delta `-0.127085455482`; positive-rank wins/equal/losses `2868/3382/5396`.

## 50. Dataset-specific findings

Beauty strict rows: `1966`; Sports strict rows: `1988`. ML-1M strict rows: `1962` from Phase 3M.

## 51. Ranking versus reranking interpretation

Ranking and reranking use separate candidate-set IDs and are never averaged together in the generated tables.

## 52. Paired positive-rank findings

Every paired comparison records LightGCN better, equal and comparator better positive-rank row counts; counts sum to evaluated rows.

## 53. Candidate and row identity

The consolidation required exact row-key coverage and exact candidate-list equality before computing same-cohort metrics and paired comparisons.

## 54. Selection-bias implications

Metrics apply only to graph-compatible strict rows. Excluded Plan E2 rows are documented in the CPU cohort table, so results should not be interpreted as full Plan E2 coverage.

## 55. CUDA sparse-forward limitation

The ML-1M D2 table identifies CUDA sparse forward tensor nondeterminism; Beauty/Sports therefore do not use CUDA project-data sparse operations.

## 56. CPU determinism findings

Beauty and Sports both passed exact CPU checks for configuration, data hashes, graph hashes, negative schedule, model states, embeddings, scores, rankings, positive ranks, HR@5 and NDCG@5.

## 57. Deviations from the original paper

The phase remains adapted because it uses reconstructed datasets, Plan E2 strict cohorts, local LightGCN training and 20-candidate manifests rather than original paper artifacts.

## 58. Limitations

This phase does not run LastFM, does not build prompts, does not invoke LLMs and does not compute beyond-utility metrics.

## 59. Whether Phase 3N-CPU succeeded

Phase 3N-CPU succeeded: CPU Beauty/Sports LightGCN outputs, checkpoints, exact repeat evidence, all-dataset LightGCN summary and seven-baseline same-cohort comparisons were generated.

## 60. LightGCN completion assessment

LightGCN is complete for the adapted Plan E strict cohorts across Beauty, Sports and ML-1M, with mixed-device provenance explicitly documented.

## 61. Baseline-phase completion assessment

The same-cohort baseline phase is complete for seven baselines on full and matched LightGCN strict cohorts, with exact row/candidate identity checks.

## 62. Protected-input immutability

Validation rows and protected hash checks confirm the CPU runner did not alter protected inputs.

## 63. Commands executed

- `python src/experiments/run_plan_e_lightgcn_beauty_sports_cpu.py`
- `python src/experiments/build_plan_e_lightgcn_cpu_baseline_consolidation.py`
- Focused pytest commands are recorded in the final assistant response.

## 64. Recommended single next task

Phase 4A — canonical utility and beyond-utility metric contract audit and same-cohort consolidation preflight
