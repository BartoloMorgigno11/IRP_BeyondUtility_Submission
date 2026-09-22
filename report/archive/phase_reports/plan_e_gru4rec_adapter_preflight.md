# Plan E GRU4Rec Adapter and Sequential Preflight Validation

## 1. Scope and provenance

Phase 3F audited whether the local Plan D GRU4Rec implementation can be connected to the six Plan E2 candidate manifests without training or scoring project data. Branch `plan-d-extensions` at `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests` was used. Protected inputs are listed in `PROTECTED_HASHES` inside `src/experiments/build_plan_e_gru4rec_preflight.py`.

## 2. Executive summary

The preflight produced 7 manifest rows, 4 sequence-audit rows, 8 index-audit rows, 6000 row-compatibility rows, 12 history-impact rows and 8 policy rows. Beauty, Sports and ML-1M are ready only for a future policy-approved pilot; LastFM remains not applicable because it has no Plan E2 candidate rows and no train split.

## 3. Safety boundaries

No project-data GRU4Rec training, scoring, optimizer step, backward pass, checkpoint creation, recommendation file or metric file is performed by this runner. The project-data work is limited to reading CSV/JSON artifacts and writing the approved preflight outputs.

## 4. Existing Plan D GRU4Rec implementation

The audited implementation path is `src/baselines/plan_d_gru4rec.py`; the historical runner is `src/experiments/run_plan_d_gru4rec.py`. The preflight does not call the runner.

## 5. GRU4Rec model architecture

Plan D `GRU4RecModel` uses an item embedding table with `num_items + 1` rows and `padding_idx=0`, a GRU encoder, and dot-product candidate scoring against item embeddings. Real item tokens are raw dense item indices shifted by +1.

## 6. Existing training interface

Plan D creates prefix-to-next-item examples from each training sequence. The preflight records users with fewer than two train events as lacking a training pair under that contract.

## 7. Existing candidate-scoring interface

The local model can score candidate token tensors while preserving candidate order. Phase 3F only validates that future tensor order can be preserved; it does not create scores.

## 8. Environment verification

The CUDA-ready environment recorded in `results/tables/neural_environment_inventory.csv` is `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe` with Python `3.12.10`, torch `2.6.0+cu124`, CUDA available `true` and device `NVIDIA GeForce RTX 4060 Laptop GPU`.

## 9. Sequential-data inventory

The sequence audit covers Beauty, Sports, ML-1M and LastFM in `results/tables/plan_e_gru4rec_sequence_audit.csv`. Train interactions are 1431047 across the three Plan E2 datasets; LastFM contributes 0 train interactions in this preflight.

## 10. Chronology verification

Beauty, Sports and ML-1M have numeric timestamps and are sorted by user, timestamp and repository row-order tie key. Equal-timestamp user counts are recorded in the sequence audit rather than silently hidden.

## 11. Split and temporal-leakage checks

Beauty, Sports and ML-1M reconstructed interactions used here contain train rows only for this audit path. No validation/test event and no held-out positive is inserted into any evaluation history.

## 12. Training sequence construction

Histories are built from train-only interactions, prefix-target pairs require at least two training events, and user zero/one/two-or-more counts are recorded over unique Plan E2 users.

## 13. Padding and truncation semantics

The adapter applies left padding with token 0, keeps the most recent 50 items, and shifts real item indices by +1. History truncation counts are in `results/tables/plan_e_gru4rec_history_impact.csv`.

## 14. Item-index and output-space semantics

The recommended policy uses a train-seen item model index. Candidate items absent from train are flagged rather than assigned untrained output parameters in the main pilot cohort.

## 15. Beauty history findings

Full-cohort rows 2000; sequentially compatible rows 1966; rows with truncated histories 8; rows with train-unseen candidates 34.

## 16. Sports history findings

Full-cohort rows 2000; sequentially compatible rows 1988; rows with truncated histories 2; rows with train-unseen candidates 12.

## 17. ML-1M history findings

Full-cohort rows 2000; sequentially compatible rows 1962; rows with truncated histories 1394; rows with train-unseen candidates 38.

## 18. LastFM status

LastFM is represented in the manifest, sequence audit and index audit as not applicable or blocked. It has no Plan E2 candidate rows and its reconstructed interactions are not split into train rows.

## 19. Plan E2 user compatibility

All Plan E2 row compatibility details are in `results/tables/plan_e_gru4rec_row_compatibility.csv`. Full-cohort rows evaluated: 6000; matched-cohort rows evaluated: 5814.

## 20. Candidate item compatibility

Rows with unmapped candidate items: 0. Rows with train-unseen candidate items: 84.

## 21. Train-unseen candidate findings

Strict sequential compatibility excludes rows with train-unseen candidates. The total strict-compatible full rows are 5916 out of 6000.

## 22. History-length findings

Rows with truncated histories in full cohorts: 1404. Rows without a training pair in full cohorts: 0.

## 23. Users without valid training pairs

The preflight records training-pair availability per row and per candidate set. No missing-history or one-event row is treated as compatible under the recommended policy.

## 24. Candidate-order preservation

Candidate lists are decoded, validated as 20 unique raw IDs, and never sorted. Future score vectors must align by original candidate position.

## 25. Future score-alignment contract

The adapter exposes finite score-vector alignment helpers but no project-data scoring. A future runner must provide one finite score per original candidate item in the same order.

## 26. Full-cohort compatibility

plan_e2_paper_like_full evaluated rows 6000; compatible rows 5916; retention rate 0.986.

## 27. Matched-cohort compatibility

plan_e2_matched_comparison evaluated rows 5814; compatible rows 5730; retention rate 0.985552115583.

## 28. Policy options evaluated

The policy table evaluates policies A-H in `results/tables/plan_e_gru4rec_policy_options.csv`; exactly one row is marked recommended.

## 29. Recommended future policy

The recommended policy is `A: strict_training_history_and_train_known_candidates`. Rationale: It avoids leakage, candidate mutation and untrained candidate parameters.

## 30. Selection-bias implications

The strict policy does not preserve the full Phase 2 cohort when train-unseen candidates exist. Any future metric table must label the strict sequential cohort separately from full and matched Plan E2 cohorts.

## 31. Proposed first GRU4Rec pilot dataset

ML-1M is the proposed first pilot because it was the prior BPRMF pilot path and the report should validate the sequential adapter on one controlled dataset before Beauty/Sports expansion.

## 32. Proposed pilot scope

Use the ML-1M ranking and reranking Plan E2 rows that are sequentially compatible under policy A, with no candidate mutation and no validation/test history insertion.

## 33. Pilot success criteria

The pilot should stop unless all emitted rankings preserve 20-candidate membership and order, all scores are finite, no held-out positive is present in input history, and strict cohort membership is written explicitly.

## 34. Pilot stop conditions

Stop if protected input hashes change, if a user lacks train-only history, if candidates are reordered or replaced, if train-unseen candidates enter the main cohort, or if non-finite scores appear.

## 35. Required future configuration fields

A future pilot config should specify dataset, candidate_set_ids, max_sequence_length, minimum_history_length, strict_candidate_train_known, seed, output paths, protected-input hashes and an explicit no-api/no-download setting.

## 36. Tests and synthetic smoke probe

Focused tests were run with `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe -m pytest tests/test_plan_e_gru4rec_adapter.py`: 18 passed in 2.57s. The requested structural tests were run with `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe -m pytest tests/test_experiment_cohort_policy.py tests/test_dataset_candidate_set_validation.py`: 38 passed in 0.48s.

The explicit synthetic smoke probe used `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe`, torch `2.6.0+cu124`, CPU plus CUDA device `NVIDIA GeForce RTX 4060 Laptop GPU`. Synthetic cardinalities were 1 user and 3 items, padding index 0, sequence shape `[1, 3]`, sequence-length shape `[1]`, candidate tensor shape `[1, 3]`, CPU output shape `[1, 3]`, and CUDA output shape `[1, 3]`. No optimizer was created, no backward pass occurred, no project data was used, and no checkpoint was saved.

## 37. Protected-input immutability

The runner verifies protected hashes before writing outputs. The preflight was run twice with `C:\ProgramData\anaconda3\python.exe src\experiments\build_plan_e_gru4rec_preflight.py` under `PYTHONNOUSERSITE=1`, and all six generated CSV hashes matched exactly:

| File | SHA256 |
|---|---|
| `results/tables/plan_e_gru4rec_history_impact.csv` | `7333E6BB029582B456604A47D249F28BD3FB3379DCA4087271F6199C60AE5B33` |
| `results/tables/plan_e_gru4rec_index_audit.csv` | `7F3172E408CC135992913EEE8400B8E921442265141D82FFEC9E4941684CA318` |
| `results/tables/plan_e_gru4rec_policy_options.csv` | `B56093445E42A5092997BB5863BEDC1D72CE9732C6E2F11F94E5DA8882F17CE9` |
| `results/tables/plan_e_gru4rec_preflight_manifest.csv` | `4181EBBA00390FC9402DFD866BE32E7D128FEDA62267B06325E013E2663E138C` |
| `results/tables/plan_e_gru4rec_row_compatibility.csv` | `632FA890F05401CECA5A6B64212370CA0E80182D5B9A55918A9AC3C10630F6C1` |
| `results/tables/plan_e_gru4rec_sequence_audit.csv` | `9F97D98A658FEB5C35FEF9ED1F9610E764A87B5BFFFDAEACAD5D71AA5FEA5442` |

Independent output validation passed for exact schemas, 6,000 unique row-compatibility keys, JSON array fields, 12 history-impact rows, valid rates, exactly one recommended policy, and forbidden leakage/candidate-replacement policies. Protected input hashes checked after the build include `results/tables/ranking_candidate_sets.csv` = `640AD9C00B7D7C825AD10746A0E720C161D1BA9D75CC791A019DF7FE14C7A955`, `results/tables/positive_positions.csv` = `0C93DD55D5C96ED99FB7F1AFC8F5319E90F3092ECB6A48F2E9D46854B38CD6D2`, `src/baselines/plan_d_gru4rec.py` = `5E288F1917B8F9A3C037D5C3EEF3BE4C2320DF465B23370D94982F15F63F699D`, and `src/experiments/run_plan_d_gru4rec.py` = `42F3EECEB8BE58C94EBB353F5DD02BAE8EC55E9859EE6455470069E8206E40AC`.

## 38. Recommended single next task

Phase 3G - controlled GRU4Rec ML-1M pilot configuration and execution.
