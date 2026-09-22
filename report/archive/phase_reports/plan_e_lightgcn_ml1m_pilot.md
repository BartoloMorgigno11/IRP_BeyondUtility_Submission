# Plan E LightGCN ML-1M Controlled Pilot

## 1. Scope and provenance
- Branch: `plan-d-extensions`.
- Commit: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests`.
- Phase 3M trained exactly one ML-1M LightGCN model family and scored only strict graph-compatible Plan E2 rows.

## 2. Executive summary
- Evaluated 1962 strict rows: 981 ranking and 981 reranking.
- Primary final loss: `0.329249883998`; repeat final loss: `0.32924988363`.
- Determinism status: `completed_with_determinism_warning`.

## 3. Scientific classification
- Classification: adapted / paper-like reconstructed, not exact paper reproduction.

## 4. Safety boundaries
- No web/API/download/environment changes. Only ML-1M LightGCN was trained.

## 5. Environment and hardware
- `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe`; torch `2.6.0+cu124`; CUDA `12.4`; device `NVIDIA GeForce RTX 4060 Laptop GPU`.

## 6. Configuration provenance
- Hyperparameters trace to the latest successful ML-1M Plan D LightGCN extended-subset row.

## 7. Final configuration
- embedding_dim=64, propagation_layers=2, batch_size=4096, epochs=5, lr=0.001, weight_decay=0.0, seed=42.

## 8. Existing LightGCN architecture
- Reused protected `LightGCNModel` with sparse propagation and dot-product scoring.

## 9. Authoritative ML-1M training interactions
- Train interactions: 994169; train users: 6040; train items: 3703.

## 10. Training-edge isolation
- Training graph used only `split=train` interactions.

## 11. Duplicate-edge policy
- Duplicate train pairs: 0; policy is deterministic deduplication.

## 12. User and item mappings
- Model node map contains 6,040 users and 3,703 items.

## 13. Bipartite node indices
- Users occupy 0..6039; items occupy 6040..9742.

## 14. Graph symmetrization
- Directed symmetric edges: 1988338.

## 15. Self-loop policy
- Self-loops: 0.

## 16. Degree construction
- User degree min/max/median: 19/2313/95.0; item degree min/max/median: 1/3403/123.0.

## 17. Normalized adjacency
- Normalized adjacency hash: `D9BFF25EC7A2CA0ECDBEBC363C14220D2D07D0EC1A0EBC2886A46DE61751F9EE`.

## 18. Sparse COO representation
- Coalesced sparse COO with float32 weights.

## 19. Sparse CUDA backend validation
- Synthetic CUDA probe and project sparse backend probe passed; project probe sum `6397.51416016`.

## 20. Propagation layers
- Propagation layers: 2.

## 21. Layer-zero inclusion
- Layer zero is included.

## 22. Layer aggregation
- Layer representations are averaged from layer 0 through L.

## 23. BPR training objective
- Reused Plan D BPR loss with one deterministic negative per positive.

## 24. Negative-sampling contract
- Negatives came from the train item universe and excluded user train positives.

## 25. Pilot-cohort construction
- Cohort rows: 2000; included: 1962; excluded: 38.

## 26. Excluded ML-1M rows
- 38 excluded rows contain graph-disconnected/train-unseen candidate items from Phase 3L evidence.

## 27. Training procedure
- Executed primary and same-seed repeat with identical graph, schedule, seed and CUDA deterministic settings.

## 28. Primary-run training history
- Completed 5 epochs; final total loss `0.329249883998`.

## 29. Same-seed repeat history
- Completed 5 epochs; final total loss `0.32924988363`.

## 30. Runtime and GPU-memory findings
- Primary duration `42.716` seconds; repeat duration `39.471` seconds.

## 31. Checkpoint and metadata
- One checkpoint retained: `results/models/plan_e_lightgcn_ml1m_pilot/model_state.pt`; SHA256 `F385C4BF57BCB3C0D8A6F41151AFB244E6C9E5120DE3DA49C54A1ABA8537C131`.

## 32. Final embedding propagation
- Final propagated embedding hash: `C2F9CD7FC7766F3E7E17DE8C086D1B0C14DB3A74F917D046D47CF3EF853148FD`.

## 33. Candidate-scoring contract
- Scoring used propagated user/item dot products and deterministic score ties.

## 34. Candidate-order and membership preservation
- Candidate arrays were preserved; ranked memberships equal candidate memberships.

## 35. Ranking results
- ranking: rows 981, HR@5 `0.666666666667`, NDCG@5 `0.467798353545`.

## 36. Reranking-candidate results
- reranking: rows 981, HR@5 `0.666666666667`, NDCG@5 `0.467798353545`.

## 37. Full strict-cohort metrics
- Full and matched cohorts are identical for ML-1M strict LightGCN rows; metrics are duplicated under both cohort IDs.

## 38. Matched strict-cohort metrics
- Matched strict-cohort rows: 981 ranking and 981 reranking.

## 39. Graph-degree findings
- Degree summaries are recorded in `plan_e_lightgcn_ml1m_pilot_metrics.csv` and `plan_e_lightgcn_ml1m_graph_summary.csv`.

## 40. Ranking versus reranking interpretation
- Ranking and reranking use different candidate orders/sets but the same strict graph-connected scoring policy.

## 41. Same-seed reproducibility
- Error-level determinism failures: 0.

## 42. Validation findings
- Validation rows: 41; error-level failures: 0.

## 43. Comparison boundary with existing baselines
- Same row keys as ML-1M BPRMF, GRU4Rec and SASRec pilot rankings, but no cross-baseline consolidation is performed here.

## 44. Sparse CUDA determinism limitation
- CUDA determinism is supported by this run's equality evidence; this remains hardware/environment-specific.

## 45. Deviations from the original paper
- Reconstructed data, local Plan D implementation, strict graph-connected subset and local environment differ from exact paper artifacts.

## 46. Limitations
- ML-1M-only pilot; Beauty/Sports/LastFM not executed; beyond-utility metrics not computed.

## 47. Whether the pilot succeeded
- Pilot success assessment: `completed_with_determinism_warning`.

## 48. Conditions for Beauty and Sports scale-up
- Scale-up requires the same protected graph, strict connected cohort, deterministic repeat and checkpoint controls.

## 49. Protected-input immutability
- Protected hashes were verified before and after execution.

## 50. Commands executed
- See final assistant response for command list; outputs are traceable to generated CSV files.

## 51. Recommended single next task
- Controlled CPU LightGCN ML-1M determinism diagnostic
