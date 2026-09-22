# Plan E BPRMF ML-1M Controlled Pilot

## 1. Scope and provenance

- Experiment ID: `plan_e_bprmf_ml1m_pilot`.
- Branch: `plan-d-extensions`.
- Commit: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests`.

## 2. Executive summary

- Trained one BPRMF model family on ML-1M split=train interactions only.
- Scored 1962 strict train-known Plan E2 rows.
- Same-seed repeat determinism status: `passed`.
- Checkpoint SHA256: `3ED8A5AE9494EBA69B30E43A54E7F63E327167D462099DEB7D377B26DDFC39E0`.

## 3. Scientific classification

This is an adapted/paper-like reconstructed BPRMF pilot. It does not claim exact reproduction of the paper BPRMF baseline.

## 4. Safety boundaries

No APIs, web, downloads, package changes, environment changes, LLM inference, non-BPRMF model execution, candidate mutation, or historical artifact modification occurred.

## 5. Environment and hardware

Environment `repo_venv_cuda`: Python `3.12.10`, torch `2.6.0+cu124`, CUDA `12.4`, device `NVIDIA GeForce RTX 4060 Laptop GPU`. GPU memory before training: 7451181056 free / 8585216000 total bytes.

## 6. Configuration provenance

Hyperparameters follow the latest traceable Plan D ML-1M BPRMF extended-subset run in `results/tables/plan_d_bprmf_training_summary.csv`: seed 42, embedding_dim 64, lr 0.001, weight_decay 0.0, batch 4096, epochs 5, Adam, CUDA.

## 7. Final pilot configuration

Configuration path: `configs/plan_e_bprmf_ml1m_pilot.yaml`. The training-scope adaptation uses all ML-1M split=train rows because the Plan D 100000-row cap does not cover the approved strict Plan E2 cohort.

## 8. Authoritative training data

Training path: `data/reconstructed/plan_c/ml1m/interactions.csv`. Training rows: 994169; unique users: 6040; unique items: 3703; duplicate user-item rows: 0.

## 9. Leakage-prevention checks

Only `split=train` rows were fitted. Validation/test rows, Plan E2 positives, candidate positions, task labels and title data were not used for fitting.

## 10. Model-index mapping

Mapping hash: `3D5123D19BF5484503A72716345732362156EF74F8D66DD178DF22DBC29EE27C`. Model users: 981 scored pilot users, with 6040 train users in the model universe. Model item indices are zero-based and contain no padding or unknown row.

## 11. Negative-sampling contract

The Phase 3D sampler samples one negative per positive from the training item universe, excludes each user's train positives, uses a seeded NumPy generator, and uses no Plan E2 candidate data.

## 12. Pilot cohort construction

Policy: `exclude_rows_with_any_train_unseen_candidate`. Included rows: 1962 total, 981 ranking and 981 reranking.

## 13. Excluded ML-1M rows

38 ML-1M Plan E2 rows were excluded by the strict train-known policy, 19 ranking and 19 reranking. They are recorded in `results/tables/plan_e_bprmf_ml1m_pilot_cohort.csv`.

## 14. Model implementation

The pilot reuses `BPRMFModel` and `bpr_loss` from `src/baselines/plan_d_bprmf.py`; that file was not modified.

## 15. Training procedure

Two runs were executed: primary and same-seed repeat. Each used 5 epochs, batch size 4096, Adam, deterministic settings and the same seed.

## 16. Primary-run training history

Primary final mean BPR loss: 0.273663378915. Training history is in `results/tables/plan_e_bprmf_ml1m_pilot_training_history.csv`.

## 17. Same-seed repeat history

Repeat final mean BPR loss: 0.273663378915. The repeat did not retain a second permanent checkpoint.

## 18. Runtime and GPU-memory findings

Primary duration: 42.185s; repeat duration: 40.282s. Primary CUDA peak allocated: 14626816 bytes; reserved: 33554432 bytes.

## 19. Checkpoint and metadata

Checkpoint: `results/models/plan_e_bprmf_ml1m_pilot/model_state.pt`. Metadata: `results/models/plan_e_bprmf_ml1m_pilot/metadata.json`.

## 20. Candidate-scoring contract

Only 1,962 strict rows were scored. Scores were aligned to the original candidate arrays; ranking used descending score, lexical raw item ID, then original candidate position.

## 21. Candidate-order and membership preservation

Stored `candidate_items` arrays preserve canonical order. `ranked_items` contain exactly the same membership as the candidate list.

## 22. Ranking results

Ranking rows are stored in `results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv`.

## 23. Reranking-candidate results

Reranking rows use the reranking candidate set only and are reported separately from ranking rows.

## 24. Full strict-cohort metrics

- ranking plan_e2_bprmf_strict_train_known_full: rows=981, HR@5=0.728848114169, NDCG@5=0.518607480492, mean_rank=3.15392456677.
- ranking plan_e2_bprmf_strict_train_known_matched: rows=981, HR@5=0.728848114169, NDCG@5=0.518607480492, mean_rank=3.15392456677.
- reranking plan_e2_bprmf_strict_train_known_full: rows=981, HR@5=0.728848114169, NDCG@5=0.518607480492, mean_rank=3.15392456677.
- reranking plan_e2_bprmf_strict_train_known_matched: rows=981, HR@5=0.728848114169, NDCG@5=0.518607480492, mean_rank=3.15392456677.

## 25. Matched strict-cohort metrics

For ML-1M the full and matched strict cohorts have identical membership, but both cohort IDs are retained explicitly in the metrics table.

## 26. Ranking versus reranking interpretation

Ranking and reranking remain separate tasks. Similarity in candidate membership or model scores does not make them equivalent.

## 27. Comparison with non-neural Phase 3A baselines

Direct comparison with Phase 3A remains deferred until the traditional baseline rankings are filtered to exactly the same 981-row-per-task strict pilot cohort.

## 28. Same-seed reproducibility

Determinism rows are in `results/tables/plan_e_bprmf_ml1m_pilot_determinism.csv`. Error-level determinism failures: 0.

## 29. Validation findings

Validation rows are in `results/tables/plan_e_bprmf_ml1m_pilot_validation.csv`. Error-level validation failures: 0.

## 30. Deviations from the original paper

The implementation, preprocessing, candidate pools and strict row filtering are reconstructed/adapted local artifacts, not verified exact paper conditions.

## 31. Limitations

The run covers only ML-1M, only BPRMF, and only the strict train-known pilot cohort. Beauty, Sports, LastFM and other neural baselines were not executed.

## 32. Whether the pilot succeeded

Pilot success assessment: `succeeded`.

## 33. Conditions for Beauty and Sports scale-up

Scale-up should require the same strict train-known policy, protected-input hashing, deterministic repeat, and same-cohort consolidation with non-neural baselines.

## 34. Protected-input immutability

Protected hashes were verified before and after the run.

## 35. Commands executed

The run used `.venv\Scripts\python.exe -m pytest` for focused tests and `.venv\Scripts\python.exe src\experiments\run_plan_e_bprmf_ml1m_pilot.py` for the pilot.

## 36. Recommended single next task

Phase 3E - BPRMF Beauty and Sports controlled execution plus same-cohort baseline consolidation.
