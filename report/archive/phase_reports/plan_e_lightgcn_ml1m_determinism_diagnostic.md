# Plan E LightGCN ML-1M Determinism and Loss-Accounting Diagnostic

## 1. Scope and provenance
- Branch: `plan-d-extensions`.
- Commit: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests`.
- Phase 3M-D1 diagnostic only; no Phase 3M output was rewritten.

## 2. Executive summary
- Classification: `cuda_sparse_training_nondeterministic`.
- Scale-up decision: `do_not_scale_up` on ``.
- CPU repeat exact: `true`; bounded CUDA model exact: `false`.

## 3. Phase 3M warning under investigation
- Phase 3M remains `completed_with_determinism_warning`; the warning was the epoch-loss sequence only.

## 4. Protected inputs
- Protected hashes verified before and after the diagnostic; configuration hash `643A5DE90EC9F457006DA96417327DBFE641A8072A4EE425AB382826F49E4FBB`.

## 5. Existing loss-accounting implementation
- BPR loss is `-logsigmoid(pos-neg).mean()` in torch float32. The protected loop appends `float(loss.detach().cpu().item())` after each optimizer step and records `float(np.mean(losses))`.

## 6. Epoch-loss reconciliation
- Loss rows: 5; maximum absolute Phase 3M epoch difference `1.2264330107036869e-09`.

## 7. Floating-point representation findings
- Maximum ULP distance: `22093453`.

## 8. Batch-mean versus weighted-mean semantics
- Phase 3M records an unweighted mean of batch means; the final partial batch has equal batch weight in that telemetry value.

## 9. CUDA bounded diagnostic design
- Two fresh CUDA runs used epoch 1, first 64 batches, fixed seed and fixed negative schedule, with no checkpoint or scoring.

## 10. CUDA batch-input identity
- Input batch hash sequence was identical across the two bounded CUDA executions.

## 11. CUDA per-batch loss findings
- Bounded CUDA per-batch loss sequence exact: `false`.

## 12. CUDA model-state findings
- First post-step model divergence: `0`.

## 13. CUDA optimizer-state findings
- Bounded CUDA optimizer exact: `false`.

## 14. First point of divergence, if any
- First bounded CUDA loss divergence: `61`.

## 15. CPU full-run design
- Two complete CPU runs used the same graph, mappings, schedule, batch size, epochs, optimizer, and seed.

## 16. CPU primary training
- Duration `612.850` seconds; final unweighted epoch loss `0.32924988583772763`.

## 17. CPU same-seed repeat
- Duration `613.738` seconds; final unweighted epoch loss `0.32924988583772763`.

## 18. CPU per-batch loss identity
- CPU batch-loss hash exact: `True`.

## 19. CPU epoch aggregation findings
- Python sum, math.fsum, NumPy float64 sum, torch float64 sum, unweighted mean and weighted mean were all compared from generated CPU history rows.

## 20. CPU model-state identity
- CPU final model-state exact: `true`.

## 21. CPU propagated-embedding identity
- CPU propagated embeddings exact: `true`.

## 22. CPU score and ranking identity
- CPU scores exact: `true`; CPU rankings exact: `true`.

## 23. CPU utility metrics
- CPU full ranking HR@5/NDCG@5 are in `results/tables/plan_e_lightgcn_ml1m_cpu_diagnostic_training_history.csv` and `results/tables/plan_e_lightgcn_ml1m_determinism_diagnostic.csv`; metric rows are generated in memory and summarized by hashes.

## 24. CPU versus CUDA comparison
- Ranked-row agreement rate: `1.0`; top-5 agreement rate: `1.0`.

## 25. Scientific reproducibility versus telemetry identity
- Phase 3M final states, embeddings, scores, ranks and metrics remain exact; epoch-loss telemetry is not bitwise exact.

## 26. Diagnostic classification
- `cuda_sparse_training_nondeterministic`.

## 27. Interpretation of the 1e-9 loss difference
- `bounded CUDA model-state divergence`.

## 28. Whether Phase 3M remains valid
- Phase 3M remains valid as `completed_with_determinism_warning`; scientifically reproducible outputs: `true`.

## 29. Approved loss tolerance, if any
- `not_approved`.

## 30. Scale-up decision
- `do_not_scale_up`.

## 31. Required wording for the final project report
- Phase 3M remains completed_with_determinism_warning. This diagnostic found exact CPU repeatability but bounded CUDA model and optimizer state divergence from the first optimizer step, so LightGCN CUDA scale-up is not approved without a focused follow-up diagnostic.

## 32. Limitations
- The bounded CUDA diagnostic covers only the first 64 epoch-1 batches; it is paired with full CPU evidence and Phase 3M full-run hashes.

## 33. Runtime and resource findings
- CPU primary/repeat durations: `612.850` / `613.738` seconds.

## 34. Protected-input immutability
- Training interactions `994169`; normalized adjacency hash `D9BFF25EC7A2CA0ECDBEBC363C14220D2D07D0EC1A0EBC2886A46DE61751F9EE`; protected hashes unchanged.

## 35. Commands executed
- `git status --short --branch`
- approved-interpreter pytest commands
- approved-interpreter diagnostic runner

## 36. Recommended single next task
- Phase 3M-D2 - isolate first-step CUDA LightGCN optimizer-state divergence on ML-1M with single-batch instrumentation
