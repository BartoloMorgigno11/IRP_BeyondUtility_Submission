# Plan E BPRMF Adapter and Preflight Validation

## 1. Scope and provenance

- Phase 3C audit and adapter preflight for BPRMF Plan E2 compatibility.
- Branch: `plan-d-extensions`.
- Commit: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests`.
- Generated tables: `results/tables/plan_e_bprmf_preflight_manifest.csv`, `plan_e_bprmf_mapping_audit.csv`, `plan_e_bprmf_row_compatibility.csv`, `plan_e_bprmf_cold_entity_impact.csv`, and `plan_e_bprmf_policy_options.csv`.

## 2. Executive summary

- Plan E2 input verification passed: 6,000 Plan E2 rows, six candidate-set IDs, 20 candidates per row, and one positive per row.
- Existing-model index compatibility rows: `6000` of `6000`.
- Strict train-known compatible rows: `5916` of `6000`.
- Rows with untrained-embedding risk: `84` of `6000`.
- Recommended future policy: `exclude_rows_with_any_train_unseen_candidate`.

## 3. Safety boundaries

No project-data training, project-data BPRMF scoring, neural inference, optimizer step, backward pass, recommendation ranking, metric computation, checkpoint creation, API call, web access, package installation, download, or environment modification was performed.

## 4. Authoritative inputs

Protected inputs and their expected hashes are checked by `src/experiments/build_plan_e_bprmf_preflight.py` before outputs are written. The main inputs are canonical Phase 2 candidates/cohorts, Phase 3A traditional baseline files, and Phase 3B neural feasibility files.

## 5. Existing Plan D BPRMF implementation

Local implementation path: `src/baselines/plan_d_bprmf.py`. Runner path: `src/experiments/run_plan_d_bprmf.py`. The adapter wraps this implementation contract and does not modify it.

## 6. BPRMF model and loss definition

`BPRMFModel` uses `nn.Embedding(num_users, embedding_dim)` and `nn.Embedding(num_items, embedding_dim)` with normally initialized embeddings. Scores are user-item dot products. The pairwise loss is `-log sigmoid(pos - neg)`.

## 7. Existing training and evaluation interfaces

Plan D `load_plan_c_bprmf_data` creates zero-based dense model indices from the union of train IDs and candidate IDs. `train_bprmf` creates the optimizer and trains. `rank_bprmf_candidates` scores candidate DataFrames after training. Phase 3C did not call training or project-data scoring.

## 8. CUDA environment verification

Environment ID: `repo_venv_cuda`. Python: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe`. Python version: `3.12.10`. Torch: `2.6.0+cu124`. CUDA available: `true`. Device: `["NVIDIA GeForce RTX 4060 Laptop GPU"]`.

## 9. Dataset training sources

- Amazon Beauty: `data/reconstructed/plan_c/amazon_beauty/interactions.csv`.
- Amazon Sports: `data/reconstructed/plan_c/amazon_sports/interactions.csv`.
- ML-1M: `data/reconstructed/plan_c/ml1m/interactions.csv`.
- LastFM is represented as blocked/not applicable for Plan E2.

## 10. User mapping audit

- Amazon Beauty user: mapping_coverage=1, train_seen_coverage=1, status=passed.
- Amazon Beauty item: mapping_coverage=1, train_seen_coverage=0.999285568483, status=warning.
- Amazon Sports user: mapping_coverage=1, train_seen_coverage=1, status=passed.
- Amazon Sports item: mapping_coverage=1, train_seen_coverage=0.999589153657, status=warning.
- ML-1M user: mapping_coverage=1, train_seen_coverage=1, status=passed.
- ML-1M item: mapping_coverage=1, train_seen_coverage=0.999187652315, status=warning.

## 11. Item mapping audit

The item mapping rows above show complete Plan E2 mapping coverage for Beauty, Sports and ML-1M. Item train-seen coverage is below 1.0 where Plan E2 includes mapped candidate items absent from train interactions.

## 12. Index-space and embedding-cardinality audit

Datamap internal IDs are explicit and one-based in the reconstructed artifacts. Plan D BPRMF itself uses zero-based dense indices with no reserved padding or unknown index. The adapter therefore treats datamap IDs as provenance/mapping evidence and prepares future model indices as zero-based dense indices from train-seen plus Plan E2 IDs.

## 13. Plan E2 candidate mapping

Every Plan E2 candidate row has 20 candidates and candidate order is preserved in `plan_e_bprmf_row_compatibility.csv`. Future score alignment is defined by raw candidate position, not by sorting.

## 14. Beauty compatibility

- Amazon Beauty: rows=2000, strict_train_known=1966, untrained_embedding_risk_rows=34, train_unseen_positive_rows=6, rows_with_any_train_unseen_candidate=34.

## 15. Sports compatibility

- Amazon Sports: rows=2000, strict_train_known=1988, untrained_embedding_risk_rows=12, train_unseen_positive_rows=4, rows_with_any_train_unseen_candidate=12.

## 16. ML-1M compatibility

- ML-1M: rows=2000, strict_train_known=1962, untrained_embedding_risk_rows=38, train_unseen_positive_rows=2, rows_with_any_train_unseen_candidate=38.

## 17. LastFM blocked status

LastFM has no Plan E2 candidate-set rows in the canonical Phase 2 artifacts. The manifest includes one explicit `not_applicable` row with a blocking reason.

## 18. Train-unseen user findings

Train-unseen user counts are recorded in `plan_e_bprmf_cold_entity_impact.csv`. No project-data user embedding was trained or scored in this phase.

## 19. Train-unseen positive-item findings

Rows with train-unseen positive items are recorded per dataset/task/cohort in `plan_e_bprmf_cold_entity_impact.csv` and per row in `plan_e_bprmf_row_compatibility.csv`.

## 20. Train-unseen candidate-item findings

Rows with any train-unseen candidate are the key source of untrained-embedding risk. Counts and candidate occurrence totals are in `plan_e_bprmf_cold_entity_impact.csv`.

## 21. Untrained-embedding risk

Under the current Plan D cardinality rule, mapped train-unseen candidate items can receive embedding rows but would remain at initialization unless they appear in train interactions. This is why strict train-known compatibility is reported separately from index compatibility.

## 22. Candidate-order preservation

The adapter decodes `candidate_items` JSON arrays, preserves their raw string order, and prepares future item-index payloads in the same order. No candidate sorting or membership change is performed.

## 23. Future score-alignment contract

Future score vectors must have length equal to the raw candidate list, all scores must be finite, and scores are aligned back to raw candidate IDs by zero-based candidate position.

## 24. Full-cohort compatibility

Full-cohort compatibility by dataset/task is summarized below:

- Amazon Beauty ranking plan_e2_paper_like_full: evaluated=1000, strict=983, retention=0.983.
- Amazon Beauty ranking plan_e2_matched_comparison: evaluated=994, strict=977, retention=0.982897384306.
- Amazon Beauty reranking plan_e2_paper_like_full: evaluated=1000, strict=983, retention=0.983.
- Amazon Beauty reranking plan_e2_matched_comparison: evaluated=994, strict=977, retention=0.982897384306.
- ML-1M ranking plan_e2_paper_like_full: evaluated=1000, strict=981, retention=0.981.
- ML-1M ranking plan_e2_matched_comparison: evaluated=1000, strict=981, retention=0.981.
- ML-1M reranking plan_e2_paper_like_full: evaluated=1000, strict=981, retention=0.981.
- ML-1M reranking plan_e2_matched_comparison: evaluated=1000, strict=981, retention=0.981.
- Amazon Sports ranking plan_e2_paper_like_full: evaluated=1000, strict=994, retention=0.994.
- Amazon Sports ranking plan_e2_matched_comparison: evaluated=913, strict=907, retention=0.993428258488.
- Amazon Sports reranking plan_e2_paper_like_full: evaluated=1000, strict=994, retention=0.994.
- Amazon Sports reranking plan_e2_matched_comparison: evaluated=913, strict=907, retention=0.993428258488.

## 25. Matched-cohort compatibility

The same table includes `plan_e2_matched_comparison` rows. Matched filtering must be applied from the same full-cohort prediction rows in any future pilot.

## 26. Policy options evaluated

Six policy options are recorded in `plan_e_bprmf_policy_options.csv`: retain randomly initialized unseen embeddings, exclude rows with any train-unseen candidate, exclude only train-unseen positives, deterministic fallback scores, remove/replace unseen candidates, and train with validation/test interactions.

## 27. Recommended future policy

Recommended policy: `exclude_rows_with_any_train_unseen_candidate`. Rationale: Prioritizes no leakage, no candidate mutation, clear interpretation and reproducible cohort comparison.

## 28. Selection-bias implications of strict compatibility filtering

Strict filtering reduces the evaluation cohort and can introduce selection bias toward users/items well covered in training. Any future comparison must report the strict cohort and apply it consistently to all compared methods.

## 29. Proposed ML-1M pilot scope

The safest next pilot scope is ML-1M BPRMF on Plan E2 ranking and reranking candidate sets, using the strict train-known policy and writing new pilot-specific outputs only after explicit Phase 3D approval.

## 30. Proposed pilot success criteria

The pilot should pass if it trains only on `split=train`, preserves Plan E2 candidate order, scores only approved rows, writes outputs to new Phase 3D paths, and produces finite scores for every retained candidate.

## 31. Pilot stop conditions

Stop on any protected-input mutation, candidate membership/order change, unmapped Plan E2 ID, non-finite loss or score, CUDA out-of-memory, checkpoint creation outside approved paths, or use of validation/test interactions in training.

## 32. Required future configuration fields

Future configuration should include dataset, candidate_set_ids, cohort policy, train data path, datamap path, output directory, seed, embedding dimension, epochs, batch size, device, and explicit approval to train.

## 33. Tests and synthetic smoke probe

`tests/test_plan_e_bprmf_adapter.py` contains synthetic-only mapping, compatibility, score-alignment, and BPRMF forward-pass tests. The smoke probe uses toy tensors only and performs no optimizer step or backward pass.

## 34. Protected-input immutability

Protected hashes are checked before generation and should be rechecked after generation. The preflight runner does not regenerate Phase 1, Phase 2, Phase 3A or Phase 3B artifacts.

## 35. Recommended single next task

Phase 3D - controlled BPRMF ML-1M pilot configuration and execution, only after explicit approval of the strict train-known compatibility policy.
