# Plan E SASRec Adapter and Transformer-Sequence Preflight Validation

## 1. Scope and provenance

Phase 3I audited whether the current Plan D SASRec implementation can be safely connected to the six Plan E2 candidate manifests without training or scoring project data. Branch `plan-d-extensions` at `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests` was used.

## 2. Executive summary

The preflight produced 7 manifest rows, 4 sequence-audit rows, 4 mask/position rows, 8 index rows, 6000 row-compatibility rows, 12 history-impact rows and 10 policy rows. The recommended future policy is `A: strict_causal_history_and_train_known_candidates`.

## 3. Safety boundaries

No project-data SASRec training, scoring, optimizer step, backward pass, checkpoint creation, recommendation file, metric table or plot is performed by this runner.

## 4. Existing Plan D SASRec implementation

The audited implementation is `src/baselines/plan_d_sasrec.py`; the historical runner is `src/experiments/run_plan_d_sasrec.py`. Existing Plan D outputs are diagnostic/adapted and remain unchanged.

## 5. SASRec architecture

`SASRecModel` uses `nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)`, learned absolute positional embeddings of length `max_seq_len`, `nn.TransformerEncoderLayer` with GELU, feed-forward dimension `4 * embedding_dim`, dropout, post-norm encoder blocks, final `LayerNorm`, and dot-product candidate scoring.

## 6. Existing training interface

Plan D constructs all prefix-to-next-item examples. Each example feeds one fixed-length padded prefix and one positive and one sampled negative target into BPR loss over the final sequence representation.

## 7. Existing candidate-scoring interface

The model exposes `score_candidates(sequences, candidate_item_tokens)` and returns one score per supplied candidate token. Phase 3I does not invoke this method on project data.

## 8. Environment verification

Audit/build environment: `C:\ProgramData\anaconda3\python.exe`. Synthetic PyTorch environment: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe` with Python `3.12.10`, torch `2.6.0+cu124`, CUDA `12.4`, CUDA available `true`, device `NVIDIA GeForce RTX 4060 Laptop GPU`.

## 9. Sequential-data inventory

Train interactions across Beauty, Sports and ML-1M: `1431047`. LastFM is recorded as blocked/not applicable because it has no Plan E2 candidate layer and no `split=train` rows.

## 10. Chronology verification

Beauty, Sports and ML-1M use numeric timestamps sorted by user, timestamp, and frozen persisted source-row index. Equal-timestamp users are recorded rather than hidden.

## 11. Split and temporal-leakage checks

Only `split=train` rows are used for histories and objective counting. Validation/test counts are zero in the audited reconstructed Beauty, Sports and ML-1M interactions.

## 12. SASRec training-objective construction

A valid objective requires at least two train events. Plan E2 rows with valid SASRec objectives: `6000` of `6000`.

## 13. Target shifting and leakage audit

Prefix targets are shifted by one: prefix positions are strictly before the target. Padding targets are `0`, and detected target-leakage rows are `0`.

## 14. Padding semantics

Padding token is `0`; real item tokens are dense item indices plus `1`; histories are left-padded to length `50`.

## 15. Positional-embedding semantics

Plan D uses absolute tensor positions `0..seq_len-1`, including padding positions. The final representation is taken from the last non-padding tensor position.

## 16. Causal-attention mask

The causal mask is square boolean upper-triangular with `True` blocking future positions and diagonal self-attention allowed. Synthetic checks: `{"causal_mask_dtype": "bool", "causal_mask_shape": [50, 50], "diagonal_allowed": true, "left_padded_final_non_padding_position": 4, "padding_key_mask_true_blocks_padding": true, "smoke_probe_in_tests": "tests/test_plan_e_sasrec_adapter.py", "true_blocks_future": true}`.

## 17. Padding-attention mask

Plan D passes `src_key_padding_mask=sequences.eq(0)`, so padding keys are blocked. Padding queries are not explicitly neutralized, but padding positions are not selected as final representations or targets.

## 18. Final sequence-representation selection

The implementation computes the last non-padding position and selects that hidden state. Fully padded rows are invalid under the adapter.

## 19. Item-token and output-space semantics

The recommended strict policy builds a dense train-seen item token space for future execution and excludes train-unseen candidate rows from the main SASRec cohort.

## 20. Beauty history and objective findings

Full rows `2000`; strict-compatible rows `1966`; train-unseen rows `34`; truncated rows `8`.

## 21. Sports history and objective findings

Full rows `2000`; strict-compatible rows `1988`; train-unseen rows `12`; truncated rows `2`.

## 22. ML-1M history and objective findings

Full rows `2000`; strict-compatible rows `1962`; train-unseen rows `38`; truncated rows `1394`.

## 23. LastFM status

LastFM is explicit in manifest, sequence, mask and index outputs but has no Plan E2 candidate rows; it is not applicable for Phase 3J.

## 24. Plan E2 user compatibility

All 6,000 Plan E2 rows have user mappings and train-only histories; all have valid SASRec objectives under the current Plan D prefix-target contract.

## 25. Candidate-item compatibility

Rows with unmapped candidates: `0`. Rows with train-unseen candidates: `84`.

## 26. Train-unseen candidate findings

Train-unseen candidates account for `84` full rows and are the primary strict-policy exclusion.

## 27. History-length and truncation findings

Rows requiring history truncation under the full cohorts: `1404`.

## 28. Mask and positional-index findings

All Beauty, Sports and ML-1M rows have verified causal-mask, padding-mask and position-index status in `results/tables/plan_e_sasrec_row_compatibility.csv`.

## 29. Users without valid SASRec objectives

Rows without a valid SASRec objective: `0`.

## 30. Candidate-order preservation

Candidate JSON arrays are decoded for validation only; no row is sorted or rewritten, and candidate membership/order remain protected inputs.

## 31. Future candidate-scoring contract

A future runner must provide one finite score per original candidate token, in original candidate order, with no padding token and no candidate replacement.

## 32. Full-cohort compatibility

plan_e2_paper_like_full: evaluated rows `6000`, compatible rows `5916`, retention rate `0.986`.

## 33. Matched-cohort compatibility

plan_e2_matched_comparison: evaluated rows `5814`, compatible rows `5730`, retention rate `0.985552115583`.

## 34. Policy options evaluated

Policies A-J are recorded in `results/tables/plan_e_sasrec_policy_options.csv`; exactly one policy is recommended and all leakage/candidate-replacement/causal-mask/padding-as-unknown policies are forbidden.

## 35. Recommended future policy

Recommended policy: `strict_causal_history_and_train_known_candidates`. Rationale: This is the only policy that avoids leakage, candidate mutation, and random untrained candidate parameters.

## 36. Selection-bias implications

The strict policy excludes rows with train-unseen candidates, so future SASRec metrics must be labeled as strict sequential cohorts and compared only on same-row cohorts.

## 37. Proposed first SASRec pilot dataset

ML-1M is the recommended first pilot because Phase 3G already established the controlled sequential pilot pattern there and its Plan E2 full and matched memberships are identical.

## 38. Proposed pilot scope

Use ML-1M ranking and reranking Plan E2 rows that pass `strict_causal_history_and_train_known_candidates`; do one primary and one same-seed repeat only after approval.

## 39. Pilot success criteria

Candidate membership/order preserved, finite 20-score vectors, no validation/test history, no held-out positive in input history, deterministic primary/repeat hashes, and no protected-input mutation.

## 40. Pilot stop conditions

Stop on mask failure, padding collision, train-unseen candidate entering strict scoring, target leakage, non-finite score/loss, checkpoint path violation, protected hash change, or nondeterminism.

## 41. Required future configuration fields

Dataset, candidate-set IDs, max sequence length, padding/index policy, strict train-known flag, seed, deterministic CUDA settings, output paths, protected hashes, and explicit no-API/no-download flags.

## 42. Tests and synthetic smoke probe

`tests/test_plan_e_sasrec_adapter.py` imports the current Plan D model, performs CPU and conditional CUDA forward-only probes on synthetic sequences, verifies output and candidate-score shapes, creates no optimizer, performs no backward pass, uses no project data and saves no checkpoint.

## 43. Protected-input immutability

The builder verifies protected hashes before producing outputs. Final validation recomputes hashes for prior Phase artifacts, Plan D SASRec, BPRMF/GRU4Rec outputs and data files.

## 44. Recommended single next task

Phase 3J - controlled SASRec ML-1M pilot configuration and execution.
