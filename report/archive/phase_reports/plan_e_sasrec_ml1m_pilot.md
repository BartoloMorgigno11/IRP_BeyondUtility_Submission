# Plan E SASRec ML-1M Controlled Pilot

## 1. Scope and provenance
Executed on branch `plan-d-extensions` at commit `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests` using configuration `configs/plan_e_sasrec_ml1m_pilot.yaml` (`sha256=B49C0A19F40CB403A3F7F89625A306D9DD4D9F704D9BD3D49435B6C7F903C89F`).

## 2. Executive summary
The pilot trained SASRec on reconstructed ML-1M `split=train`, scored 1962 strict Plan E2 rows, and retained one primary checkpoint. The run remains classified as `adapted_paper_like_reconstructed`.

## 3. Scientific classification
This is an adapted paper-like reconstruction, not exact paper reproduction, because it uses reconstructed Plan C ML-1M data and local Plan D/Plan E code rather than verified original-paper artifacts.

## 4. Safety boundaries
Only ML-1M SASRec was trained. Validation/test interactions, candidate labels, candidate positions, titles, LLMs, and other recommender families were not used for fitting.

## 5. Environment and hardware
`environment_id=repo_venv_cuda`, Python `3.12.10`, torch `2.6.0+cu124`, CUDA `12.4`, device `NVIDIA GeForce RTX 4060 Laptop GPU`.

## 6. Configuration provenance
Hyperparameters come from `results/tables/plan_d_sasrec_training_summary.csv` row `Plan D ml1m sasrec extended_subset completed 2026-06-30T09:26:14.123481+00:00` with config hash `8864a614183d`.

## 7. Final configuration
Embedding `64`, hidden `64`, heads `2`, layers `2`, dropout `0.2`, sequence length `50`, batch `512`, epochs `5`, learning rate `0.001`.

## 8. SASRec architecture
The pilot reuses the local Plan D SASRec layers through a Phase 3J finite-mask wrapper: item embedding with padding index 0, learned absolute positional embeddings for positions 0..49, two Transformer encoder blocks, GELU feed-forward layers, final layer normalization, last non-padding representation selection, and dot-product candidate scoring.

## 9. Authoritative ML-1M interaction data
`data/reconstructed/plan_c/ml1m/interactions.csv` (`sha256=1871DE0C52382305B2AA54187300B68889B34551635E4991E6368E556B39D870`) supplied 994169 train rows, 0 validation rows, and 0 test rows; only train rows were used.

## 10. Split isolation and leakage prevention
The chronology builder filters `split=train` before ordering, indexing, pair construction, and model fitting. Evaluation positives were checked during scoring and were not injected into histories.

## 11. Chronology construction
Chronology hash `24C4153B4ABAAB170E59BCF8DE6E079A42AC9414A225816181B1071EE762AEAD` covers `user_id`, `item_id`, `timestamp`, and frozen `source_row_index`. It contains 6040 users and 3703 train-seen items.

## 12. Equal-timestamp handling
Equal timestamps are resolved by frozen source row order. There are 243758 equal-timestamp groups covering 770260 events.

## 13. Training-objective construction
All-prefix next-item construction produced 988129 supervised examples from train histories; the arrays hash is `52B8EBE3461B3ABDC580AAB294E98B7287C3CCEB0398BAFA24B95EC7ECBF928F`.

## 14. Positive and negative target semantics
Positive targets are the next train item in each user prefix. Negative targets are sampled from train-seen item indices excluding the user's train positives; padding token 0 is never a target.

## 15. Negative-sampling contract
The deterministic negative-sampling evidence hash is `31755A14E76DDA85C895455D1A445D7A9056EBC52CE6A660FCC9BB130BD47C1A` for the primary run and `31755A14E76DDA85C895455D1A445D7A9056EBC52CE6A660FCC9BB130BD47C1A` for the same-seed repeat.

## 16. Item-token mapping
The item-token map has 3703 train-seen real item tokens; token 0 is reserved for padding. Map hash `115817925F2B485C50DBF128E5372D2C11D7D9680ECC24418213C5D8B4ECE98D`.

## 17. Padding semantics
Sequences are left padded to length 50 with token 0. Real item IDs use dense train item index plus one, so padding cannot be scored as a candidate.

## 18. Positional-embedding semantics
The Plan D model uses absolute tensor positions `0..49` after left padding. This pilot preserves that contract and does not reconstruct paper-side absolute positions beyond the local tensor indices.

## 19. Causal-attention mask
The Transformer encoder receives a boolean causal component so future tensor positions are blocked while diagonal self-attention remains allowed.

## 20. Padding-attention mask
The Phase 3J wrapper combines the causal and padding-key masks and leaves each padding query's diagonal unmasked to avoid CUDA all-masked attention rows. Real-token queries still cannot attend to padding keys.

## 21. Final sequence representation
The scored user representation is selected from the last non-padding history position after the Transformer encoder and final layer normalization.

## 22. Pilot-cohort construction
The cohort contains 2000 ML-1M Plan E2 rows, with 1962 included and 38 excluded under `strict_causal_history_and_train_known_candidates`.

## 23. Excluded ML-1M rows
38 rows were excluded, all before scoring. The common reason is a strict sequential/train-seen compatibility failure recorded in `plan_e_sasrec_ml1m_pilot_cohort.csv`.

## 24. Training procedure
Training used BPR pairwise loss, Adam, deterministic seeds, deterministic torch algorithms, and a deterministic negative sampler over train-seen item indices.

## 25. Primary-run training history
epoch 1: loss=0.234130; epoch 2: loss=0.142567; epoch 3: loss=0.124041; epoch 4: loss=0.114610; epoch 5: loss=0.108095

## 26. Same-seed repeat history
epoch 1: loss=0.234130; epoch 2: loss=0.142567; epoch 3: loss=0.124041; epoch 4: loss=0.114610; epoch 5: loss=0.108095

## 27. Runtime and GPU-memory findings
Primary runtime was 524.177s and repeat runtime was 419.173s. Peak CUDA fields are recorded per epoch in training history.

## 28. Checkpoint and metadata
Exactly one checkpoint was retained at `results/models/plan_e_sasrec_ml1m_pilot/model_state.pt` (`sha256=0C1DDE87EF8D50AD909B81D81666DE5C5DFCB2B8B163B738FBC578A9A206657F`), with metadata at `results/models/plan_e_sasrec_ml1m_pilot/metadata.json`. The repeat run retained no checkpoint path.

## 29. Evaluation-history construction
Evaluation histories are train-only user histories clipped to the most recent 50 items; no candidate membership or positive label is used to construct the history.

## 30. Candidate-scoring contract
The model scored only the supplied 20 candidates for each compatible row. Excluded rows were not scored.

## 31. Candidate-order and membership preservation
The `candidate_items` column preserves source order, while `ranked_items` is a permutation of the same 20 IDs after score sorting.

## 32. Ranking results
`evaluated_rows=981`, `HR@5=0.904179`, `NDCG@5=0.767620`, `mean_positive_rank_zero_based=1.353721`.

## 33. Reranking-candidate results
`evaluated_rows=981`, `HR@5=0.904179`, `NDCG@5=0.767620`, `mean_positive_rank_zero_based=1.353721`.

## 34. Full strict-cohort metrics
Ranking full cohort: `evaluated_rows=981`, `HR@5=0.904179`, `NDCG@5=0.767620`, `mean_positive_rank_zero_based=1.353721`. Reranking full cohort: `evaluated_rows=981`, `HR@5=0.904179`, `NDCG@5=0.767620`, `mean_positive_rank_zero_based=1.353721`.

## 35. Matched strict-cohort metrics
Ranking matched cohort: `evaluated_rows=981`, `HR@5=0.904179`, `NDCG@5=0.767620`, `mean_positive_rank_zero_based=1.353721`. Reranking matched cohort: `evaluated_rows=981`, `HR@5=0.904179`, `NDCG@5=0.767620`, `mean_positive_rank_zero_based=1.353721`.

## 36. History-length and truncation findings
Histories longer than 50 keep the most recent 50 train events. In the strict pilot, 1374 scored rows were truncated and maximum effective history length was 50. Mean raw/effective lengths are recorded in metrics.

## 37. Ranking versus reranking interpretation
Ranking and reranking use separate candidate-set IDs and should be interpreted as strict ML-1M Plan E2 candidate diagnostics, not full-catalog evaluation.

## 38. Same-seed reproducibility
14 of 14 determinism checks passed. Failed checks: [].

## 39. Validation findings
27 of 27 validation checks passed. Failed checks: [].

## 40. Comparison boundary with existing baselines
Do not compare these SASRec metrics to existing baselines unless those baseline rankings are filtered to the same strict sequential ML-1M cohort and candidate rows.

## 41. Absolute-position reconstruction limitation
The local model uses tensor positions after left padding. The original paper's exact absolute-position construction is not independently verifiable from current repository artifacts.

## 42. Equal-timestamp reconstruction limitation
Equal timestamps are resolved by current persisted source row order. This is deterministic, but it remains a reconstruction when the original preprocessing order is not independently available.

## 43. Deviations from the original paper
The data, preprocessing lineage, model hyperparameters, and local training implementation are reconstructed/adapted. The pilot is evidence for a controlled local pipeline, not exact paper reproduction.

## 44. Limitations
The pilot covers ML-1M only, excludes 38 incompatible Plan E2 rows, and evaluates supplied 20-candidate lists rather than a full-catalog setting.

## 45. Whether the pilot succeeded
The pilot succeeded as a controlled adapted ML-1M execution if validation has no error-level failures and the manifest status is `completed`. Machine-readable evidence is in the manifest, validation, determinism, ranking, and metrics tables.

## 46. Conditions for Beauty and Sports scale-up
Scale-up requires the same strict train-history, train-seen candidate, chronology, and same-seed repeat controls on Beauty and Sports without altering existing manifests.

## 47. Protected-input immutability
Protected hashes were verified before and after execution. `original_repo/`, Plan D source/runner, Phase 3I adapter, BPRMF artifacts, candidate manifests, and reconstructed ML-1M data were not modified.

## 48. Commands executed
See final assistant response for the exact shell commands. The runner command was `.venv\Scripts\python.exe src/experiments/run_plan_e_sasrec_ml1m_pilot.py`.

## 49. Recommended single next task
Phase 3K — controlled SASRec Beauty and Sports execution and same-cohort consolidation.
