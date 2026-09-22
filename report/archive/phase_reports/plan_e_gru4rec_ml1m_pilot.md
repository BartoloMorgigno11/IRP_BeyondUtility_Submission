# Plan E GRU4Rec ML-1M Controlled Pilot

## 1. Scope and provenance
Executed on branch `plan-d-extensions` at commit `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests` using configuration `configs/plan_e_gru4rec_ml1m_pilot.yaml` (`sha256=CC040BB51EBCE09CE0C7857B921036FAEB388EF248C0A852092B42385EFBD1B2`).

## 2. Executive summary
The pilot trained GRU4Rec on reconstructed ML-1M `split=train`, scored 1962 strict Plan E2 rows, and retained one primary checkpoint. The run remains classified as `adapted_paper_like_reconstructed`.

## 3. Scientific classification
This is an adapted paper-like reconstruction, not exact paper reproduction, because it uses reconstructed Plan C ML-1M data and local Plan D/Plan E code rather than verified original-paper artifacts.

## 4. Safety boundaries
Only ML-1M GRU4Rec was trained. Validation/test interactions, candidate labels, candidate positions, titles, LLMs, and other recommender families were not used for fitting.

## 5. Environment and hardware
`environment_id=repo_venv_cuda`, Python `3.12.10`, torch `2.6.0+cu124`, CUDA `12.4`, device `NVIDIA GeForce RTX 4060 Laptop GPU`.

## 6. Configuration provenance
Hyperparameters come from `results/tables/plan_d_gru4rec_training_summary.csv` row `Plan D ml1m gru4rec extended_subset completed 2026-06-30T14:24:45.150160+00:00` with config hash `2e552a1d9ab6`.

## 7. Final configuration
Embedding `64`, hidden `64`, layers `1`, dropout `0.2`, sequence length `50`, batch `512`, epochs `5`, learning rate `0.001`.

## 8. Authoritative interaction data
`data/reconstructed/plan_c/ml1m/interactions.csv` (`sha256=1871DE0C52382305B2AA54187300B68889B34551635E4991E6368E556B39D870`) supplied 994169 train rows, 0 validation rows, and 0 test rows; only train rows were used.

## 9. Split isolation and leakage prevention
The chronology builder filters `split=train` before ordering, indexing, pair construction, and model fitting. Evaluation positives were checked during scoring and were not injected into histories.

## 10. Chronology construction
Chronology hash `24C4153B4ABAAB170E59BCF8DE6E079A42AC9414A225816181B1071EE762AEAD` covers `user_id`, `item_id`, `timestamp`, and frozen `source_row_index`. It contains 6040 users and 3703 train-seen items.

## 11. Equal-timestamp handling
Equal timestamps are resolved by frozen source row order. There are 243758 equal-timestamp groups covering 770260 events.

## 12. Training-pair construction
All-prefix next-item construction produced 988129 pairs from train histories; the arrays hash is `52B8EBE3461B3ABDC580AAB294E98B7287C3CCEB0398BAFA24B95EC7ECBF928F`.

## 13. Item-token mapping
The item-token map has 3703 train-seen real item tokens; token 0 is reserved for padding. Map hash `6FAC77CE65025D80754503ACBD788ACA976CD2F9907E804B8773295D29B03AF1`.

## 14. Padding semantics
Sequences are left padded to length 50 with token 0. Real item IDs use dense train item index plus one, so padding cannot be scored as a candidate.

## 15. Truncation semantics
Histories longer than 50 keep the most recent 50 train events. In the strict pilot, 1374 scored rows were truncated.

## 16. Pilot-cohort construction
The cohort contains 2000 ML-1M Plan E2 rows, with 1962 included and 38 excluded under `strict_training_history_and_train_known_candidates`.

## 17. Excluded ML-1M rows
38 rows were excluded, all before scoring. The common reason is a strict sequential/train-seen compatibility failure recorded in `plan_e_gru4rec_ml1m_pilot_cohort.csv`.

## 18. GRU4Rec architecture
The model reuses the local Plan D `GRU4RecModel` with item embeddings, a GRU encoder, and dot-product candidate scoring from the final sequence representation.

## 19. Training procedure
Training used BPR pairwise loss, Adam, deterministic seeds, deterministic torch algorithms, and a deterministic negative sampler over train-seen item indices.

## 20. Primary-run training history
epoch 1: loss=0.312501; epoch 2: loss=0.222096; epoch 3: loss=0.182823; epoch 4: loss=0.158131; epoch 5: loss=0.141540

## 21. Same-seed repeat history
epoch 1: loss=0.312501; epoch 2: loss=0.222096; epoch 3: loss=0.182823; epoch 4: loss=0.158131; epoch 5: loss=0.141540

## 22. Runtime and GPU-memory findings
Primary runtime was 105.413s and repeat runtime was 104.287s. Peak CUDA fields are recorded per epoch in training history.

## 23. Checkpoint and metadata
Exactly one checkpoint was retained at `results/models/plan_e_gru4rec_ml1m_pilot/model_state.pt` (`sha256=1B81999CAE574DF883B98362CB2EFEB554EB7CBE275AD263FA179B6BE741D06D`), with metadata at `results/models/plan_e_gru4rec_ml1m_pilot/metadata.json`.

## 24. Evaluation-history construction
Evaluation histories are train-only user histories clipped to the most recent 50 items; no candidate membership or positive label is used to construct the history.

## 25. Candidate-scoring contract
The model scored only the supplied 20 candidates for each compatible row. Excluded rows were not scored.

## 26. Candidate-order and membership preservation
The `candidate_items` column preserves source order, while `ranked_items` is a permutation of the same 20 IDs after score sorting.

## 27. Ranking results
`evaluated_rows=981`, `HR@5=0.875637`, `NDCG@5=0.713518`, `mean_positive_rank_zero_based=1.712538`.

## 28. Reranking-candidate results
`evaluated_rows=981`, `HR@5=0.875637`, `NDCG@5=0.713518`, `mean_positive_rank_zero_based=1.712538`.

## 29. Full strict-cohort metrics
Ranking full cohort: `evaluated_rows=981`, `HR@5=0.875637`, `NDCG@5=0.713518`, `mean_positive_rank_zero_based=1.712538`. Reranking full cohort: `evaluated_rows=981`, `HR@5=0.875637`, `NDCG@5=0.713518`, `mean_positive_rank_zero_based=1.712538`.

## 30. Matched strict-cohort metrics
Ranking matched cohort: `evaluated_rows=981`, `HR@5=0.875637`, `NDCG@5=0.713518`, `mean_positive_rank_zero_based=1.712538`. Reranking matched cohort: `evaluated_rows=981`, `HR@5=0.875637`, `NDCG@5=0.713518`, `mean_positive_rank_zero_based=1.712538`.

## 31. History-length and truncation findings
Mean raw/effective history lengths are recorded in metrics. The full ranking truncation rate is `0.7003058103975535` and reranking truncation rate is `0.7003058103975535`.

## 32. Ranking versus reranking interpretation
Ranking and reranking use separate candidate-set IDs and should be interpreted as strict ML-1M Plan E2 candidate diagnostics, not full-catalog evaluation.

## 33. Same-seed reproducibility
13 of 13 determinism checks passed. Failed checks: [].

## 34. Validation findings
27 of 27 validation checks passed. Failed checks: [].

## 35. Comparison boundary with existing baselines
Do not compare these GRU4Rec metrics to existing baselines unless those baseline rankings are filtered to the same strict sequential ML-1M cohort and candidate rows.

## 36. Deviations from the original paper
The data, preprocessing lineage, model hyperparameters, and local training implementation are reconstructed/adapted. The pilot is evidence for a controlled local pipeline, not exact paper reproduction.

## 37. Limitations
The pilot covers ML-1M only, excludes 38 incompatible Plan E2 rows, and evaluates supplied 20-candidate lists rather than a full-catalog setting.

## 38. Whether the pilot succeeded
The pilot succeeded if all output validation checks passed and determinism checks did not fail. The machine-readable status is in the manifest and determinism table.

## 39. Conditions for Beauty and Sports scale-up
Scale-up requires the same strict train-history, train-seen candidate, chronology, and same-seed repeat controls on Beauty and Sports without altering existing manifests.

## 40. Protected-input immutability
Protected hashes were verified before and after execution. `original_repo/`, Plan D source/runner, Phase 3F adapter, BPRMF artifacts, candidate manifests, and reconstructed ML-1M data were not modified.

## 41. Commands executed
See final assistant response for the exact shell commands. The runner command was `.venv\Scripts\python.exe src/experiments/run_plan_e_gru4rec_ml1m_pilot.py`.

## 42. Recommended single next task
Phase 3H controlled GRU4Rec Beauty and Sports execution and same-cohort consolidation.
