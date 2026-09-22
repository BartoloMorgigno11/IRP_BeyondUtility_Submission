# Plan E LightGCN Adapter and Graph-Construction Preflight Validation

## 1. Scope and provenance
- Branch: `plan-d-extensions`.
- Commit: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests`.
- Phase 3L adapter/preflight only; no LightGCN training, project-data propagation, scoring, rankings, metrics, plots or checkpoints were produced.
- Generated outputs are the eight `results/tables/plan_e_lightgcn_*.csv` files and this report.

## 2. Executive summary
- Plan E2 candidate verification passed for 6,000 rows across six 20-candidate manifests.
- Strict graph-compatible rows: Beauty 1966/2000, Sports 1988/2000, ML-1M 1962/2000.
- LastFM remains blocked because it has `split=observed` only and no Plan E2 candidate layer.
- Recommended future policy: `verified_train_graph_and_strict_connected_candidates`.

## 3. Safety boundaries
- The runner hashes protected inputs, writes only Phase 3L LightGCN audit artifacts, and uses synthetic sparse probes only.
- It does not call Plan D training, does not instantiate a project-data optimizer, and does not call project-data `final_embeddings` or candidate scoring.

## 4. Existing Plan D LightGCN implementation
- Model path: `src/baselines/plan_d_lightgcn.py`.
- Runner path: `src/experiments/run_plan_d_lightgcn.py`.
- Prior execution evidence remains in `results/tables/plan_d_lightgcn_training_summary.csv`; this preflight did not rerun it.

## 5. LightGCN architecture
- `LightGCNModel(num_users, num_items, embedding_dim, num_layers)` defines separate user and item embeddings initialized from a normal distribution with std 0.01.
- The graph is a sparse normalized bipartite adjacency.

## 6. Existing training interface
- `train_lightgcn` builds a normalized graph, creates an Adam optimizer, samples negatives and applies BPR loss.
- This preflight does not invoke that interface on project data.

## 7. Existing candidate-scoring interface
- `rank_lightgcn_candidates` propagates embeddings, dot-products aligned user/item embeddings, sorts by score, and writes recommendation rows in Plan D runs.
- This preflight audits candidate compatibility but does not call the scorer on project data.

## 8. Execution environment
- Python executable: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe`.
- Python version: `3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]`.
- Torch: `2.6.0+cu124`, torch CUDA runtime `12.4`.
- CUDA available: `true`; device: `NVIDIA GeForce RTX 4060 Laptop GPU`.

## 9. Graph-data inventory
- Amazon Beauty: total interactions 176139, train 176139, users 22363, items 12092, status `passed`.
- Amazon Sports: total interactions 260739, train 260739, users 35598, items 18347, status `passed`.
- ML-1M: total interactions 994169, train 994169, users 6040, items 3703, status `passed`.
- LastFM: total interactions 92834, train 0, users 0, items 0, status `blocked`.

## 10. Training-edge isolation
- Beauty, Sports and ML-1M graph audits use only `split=train` interactions. Validation/test/candidate/positive-derived edges are not inserted.
- LastFM has `split=observed` only, so it is blocked rather than reinterpreted as train.

## 11. Duplicate-edge policy
- Plan D data loading drops duplicate `user_id,item_id` train pairs before LightGCN graph construction.
- Amazon Beauty: duplicate train user-item pairs 0.
- Amazon Sports: duplicate train user-item pairs 0.
- ML-1M: duplicate train user-item pairs 0.

## 12. Bipartite node indexing
- Users and items use separate dense local indices; item global nodes are offset by `num_users`.
- Index details are in `results/tables/plan_e_lightgcn_index_audit.csv`.

## 13. User/item node offsets
- Amazon Beauty: item offset 22363, total nodes 34455.
- Amazon Sports: item offset 35598, total nodes 53945.
- ML-1M: item offset 6040, total nodes 9743.

## 14. Graph symmetrization
- Each train pair contributes user-to-item and item-to-user directed edges.

## 15. Self-loop policy
- No self-loops are added by the local Plan D LightGCN graph builder.

## 16. Degree calculations
- Amazon Beauty: user degree min/max/mean/median = 4/203/7.87635827036/5; item degree min/max/mean/median = 1/413/14.5665729408/8.
- Amazon Sports: user degree min/max/mean/median = 4/295/7.32454070453/5; item degree min/max/mean/median = 1/963/14.2115332207/8.
- ML-1M: user degree min/max/mean/median = 19/2313/164.597516556/95; item degree min/max/mean/median = 1/3403/268.476640562/123.

## 17. Normalized adjacency
- Formula: symmetric `D^-1/2 A D^-1/2`, with degrees calculated after symmetrization.
- Normalized adjacency hashes are recorded in `plan_e_lightgcn_normalization_audit.csv`.

## 18. Sparse tensor representation
- The existing implementation uses coalesced `torch.sparse_coo_tensor` with int64 indices and float32 weights.

## 19. Propagation-layer semantics
- Propagation recurrence is `E^(k+1) = A_norm E^k` through sparse matrix multiplication.

## 20. Layer-zero inclusion
- Layer zero is included in the final representation.

## 21. Layer aggregation
- Layer embeddings are averaged from layer 0 through `num_layers`.

## 22. User/item final embedding separation
- Final embeddings are split as users `final[:num_users]` and items `final[num_users:]`.

## 23. Candidate-scoring contract
- Candidate scoring is a dot product between propagated user and item embeddings; future execution must retain candidate order until score alignment is validated.

## 24. Beauty graph findings
- Amazon Beauty: 176139 train interactions, 176139 unique train pairs, 34455 graph nodes, 352278 directed edges, strict-compatible rows 1966/2000.

## 25. Sports graph findings
- Amazon Sports: 260739 train interactions, 260739 unique train pairs, 53945 graph nodes, 521478 directed edges, strict-compatible rows 1988/2000.

## 26. ML-1M graph findings
- ML-1M: 994169 train interactions, 994169 unique train pairs, 9743 graph nodes, 1988338 directed edges, strict-compatible rows 1962/2000.

## 27. LastFM status
- LastFM status is `blocked`: no split='train' interactions are available.

## 28. User connectivity findings
- All Plan E2 users for Beauty, Sports and ML-1M are mapped and graph-connected in the strict train graph.

## 29. Positive-item connectivity findings
- Positive-item connectivity is tracked per row in `plan_e_lightgcn_row_compatibility.csv`; rows with disconnected positives are excluded by strict policy.

## 30. Candidate-item connectivity findings
- Amazon Beauty: disconnected candidate occurrences across full Plan E2 rows = 34.
- Amazon Sports: disconnected candidate occurrences across full Plan E2 rows = 12.
- ML-1M: disconnected candidate occurrences across full Plan E2 rows = 38.

## 31. Isolated-embedding risk
- Any policy that allocates train-unseen candidates would leave zero-degree item embeddings at initialization except for layer-zero averaging, creating random score risk.

## 32. Plan E2 compatibility
- Row-level compatibility is in `results/tables/plan_e_lightgcn_row_compatibility.csv`.

## 33. Full-cohort retention
- Amazon Beauty ranking: 983/1000 retained (0.983).
- Amazon Beauty reranking: 983/1000 retained (0.983).
- Amazon Sports ranking: 994/1000 retained (0.994).
- Amazon Sports reranking: 994/1000 retained (0.994).
- ML-1M ranking: 981/1000 retained (0.981).
- ML-1M reranking: 981/1000 retained (0.981).

## 34. Matched-cohort retention
- Amazon Beauty ranking: 977/994 retained (0.982897384306).
- Amazon Beauty reranking: 977/994 retained (0.982897384306).
- Amazon Sports ranking: 907/913 retained (0.993428258488).
- Amazon Sports reranking: 907/913 retained (0.993428258488).
- ML-1M ranking: 981/1000 retained (0.981).
- ML-1M reranking: 981/1000 retained (0.981).

## 35. Sparse CPU findings
- Synthetic CPU sparse forward passed: `true`.

## 36. Sparse CUDA findings
- Synthetic CUDA sparse forward passed: `true`.

## 37. Deterministic sparse-operation findings
- CPU/CUDA synthetic max absolute difference: `9.313225746154785e-10`; warnings: `none`.

## 38. Resource and memory assessment
- Amazon Beauty: graph nodes 34455, directed edges 352278, estimated graph tensor bytes 33507000.
- Amazon Sports: graph nodes 53945, directed edges 521478, estimated graph tensor bytes 51859320.
- ML-1M: graph nodes 9743, directed edges 1988338, estimated graph tensor bytes 47249384.
- LastFM: graph nodes 0, directed edges 0, estimated graph tensor bytes 0.

## 39. Policy options evaluated
- Evaluated 10 policies A-J; exactly one is marked `recommended`.

## 40. Recommended future policy
- `verified_train_graph_and_strict_connected_candidates`: Safest no-leakage policy with no isolated random candidate embeddings and no candidate mutation.

## 41. Selection-bias implications
- Strict graph-connected filtering excludes rows with train-unseen candidates; comparisons must report the retained cohort and avoid mixing with full-cohort metrics.

## 42. Recommended first LightGCN pilot dataset
- ML-1M is the recommended first pilot dataset because it has the smallest graph and high strict retention.

## 43. Recommended first-pilot device
- Recommended first-pilot device: `cuda` based on synthetic sparse probe evidence.

## 44. Proposed pilot scope
- A future pilot should use the ML-1M strict graph-connected Plan E2 ranking and reranking rows only.

## 45. Pilot success criteria
- Stop-free completion, deterministic rerun hashes, no disconnected retained row, and no deviation from train-only graph construction.

## 46. Pilot stop conditions
- Stop on protected-input hash mismatch, CUDA sparse nondeterminism, non-finite normalized weights, disconnected retained candidate, candidate mutation, validation/test edge leakage, or checkpoint/ranking/metric output path ambiguity.

## 47. Tests and synthetic smoke probes
- Tests target synthetic adapter behavior and the existing cohort/candidate validation suites. Sparse probes are synthetic toy LightGCN forward passes only.

## 48. Protected-input immutability
- The runner verifies protected hashes before and after output generation.

## 49. Recommended single next task
- Phase 3M - controlled LightGCN ML-1M pilot configuration and execution.

## Evidence paths
- `results/tables/plan_e_lightgcn_preflight_manifest.csv`
- `results/tables/plan_e_lightgcn_graph_audit.csv`
- `results/tables/plan_e_lightgcn_normalization_audit.csv`
- `results/tables/plan_e_lightgcn_index_audit.csv`
- `results/tables/plan_e_lightgcn_row_compatibility.csv`
- `results/tables/plan_e_lightgcn_connectivity_impact.csv`
- `results/tables/plan_e_lightgcn_sparse_execution_audit.csv`
- `results/tables/plan_e_lightgcn_policy_options.csv`
