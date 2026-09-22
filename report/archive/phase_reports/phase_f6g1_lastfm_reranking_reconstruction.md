# Phase F6G1 - LastFM Re-ranking Reconstruction

## Initial repository state

Branch `plan-d-extensions`, HEAD `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`, staged files ``. Pre-existing tracked dirty files remain `M README.md
 M report/dataset_audit.md
 M report/notes.md`.

## Environment used

Interpreter `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_repro_system_site\Scripts\python.exe`; Python `3.12.10`; classification `reconstructed_venv_with_system_site_packages`.

## Frozen-input verification

Input changed=0, missing=0.

## Local source inventory

Rows: 8. Evidence remains local-only.

## Paper pool contract

The paper-exact LastFM pool contract remains incomplete for source top-K, union/order, seed identity and denominator.

## Original-repository pool contract

Original repository templates confirm candidate20 re-ranking prompts, but no executable LastFM pool builder was recovered.

## LastFM cohort audit

Audited users: 1859. No user is silently dropped.

## Traditional-prediction inventory

Inventory rows: 27. F6D1 candidate-only and F6D2 selected full-catalog predictions are separated.

## Seed-policy audit

F6D2 validation-selected seeds are frozen and used for the adapted pool; paper seed identity is unresolved.

## Candidate-source model decisions

BPRMF, GRU4Rec, SASRec and LightGCN are canonical adapted sources; Random, MostPop and BM25-like remain diagnostic/non-canonical for the pool.

## Canonical pool construction

Pool rows: 37180. Policy: round-robin union, first-occurrence dedupe, no positive insertion.

## Diagnostic pool variants

Positive-present population is diagnostic; all-user population is canonical for adapted future inference.

## Positive-item policy

Positive items are never inserted. Absent positives are retained and counted as utility failures in all-user operational utility.

## Previous 592/1859 pool reconciliation

Previous-pool comparison rows: 1859. Reuse is allowed only as adapted evidence.

## Pool-size results

Mean pool size=20; exact-target-size rate=1.

## Positive-present results

Positive-present count=592; rate=0.31845077998924154.

## Candidate-pool recall and upper bounds

All-user candidate-pool recall=0.31845077998924154; max HR@5=0.31845077998924154.

## Source-model overlap

Source contribution counts are recorded in `phase_f6g1_pool_quality_summary.csv`.

## Traditional-model within-pool metrics

Metric rows: 56; missing source scores are not imputed.

## Title/item reconciliation

Rows: 37180; candidate identity remains canonical item ID.

## Comparability gate

source-contract gate=warning, cohort-completeness gate=pass, prediction-completeness gate=pass, seed-policy gate=warning, candidate-pool-size gate=pass, duplicate-policy gate=pass, positive-policy gate=pass, candidate-pool-recall gate=warning, title/item-mapping gate=pass, no-leakage gate=pass, deterministic-reconstruction gate=pass, future-request-contract gate=pass

## Exactness classification

Overall pool is adapted/reconstructed, not paper-exact.

## Future logical-instance count

1859

## Future request count

7436

## Future shard count

15

## Implementation files

F6G1 config, analysis module, wrapper and tests were added.

## Tests

Validation passed=148, failed=0.

## Excluded unsafe tests

training;cuda;ollama;inference;network;downloads;api;unsafe_full_suite;tests/test_finalization_f6d2_lastfm_trainable_models.py

## Determinism

Builder outputs are designed for byte-stable scientific content; runtime metadata is separated in the manifest.

## Protected-input verification

Prior direct inputs changed=0.

## Freeze manifest

Scientific content hash `1318700448128bfcf9ebeab4928f9a71edd0dffe6679234f499bcb2845d96343`.

## Missing or blocked evidence

Rows: 5.

## Final decision

f6g1_lastfm_reranking_reconstruction_complete_with_adapted_pool

## Recommended next action

proceed_to_lastfm_reranking_llm_inference

## Final git state

Tracked dirty files: `M README.md
 M report/dataset_audit.md
 M report/notes.md`; staged files: ``.

## Files created or modified

Only F6G1 implementation and F6G1 artifacts are intended outputs.

## Remaining limitations

Paper-exact source K, union/order, seed identity, model snapshots and denominator remain unresolved.
