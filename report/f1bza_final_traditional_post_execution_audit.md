# F1BZA Final Traditional Post-Execution Audit

Decision token: `F1BZA_CORRECTION_REQUIRED`
Food.com history decision: `FOODCOM_HISTORY_VALID`
Reranking pool decision: `RERANKING_POOL_SOURCE_DEFECT`

## Food.com History

- Audited users: `1000`.
- Zero-history users: `727`.
- Zero-history reasons: `{"NONZERO_HISTORY_ALIGNED":261,"NONZERO_HISTORY_ALIGNED_SEQUENCE_TRUNCATED_TO_MAX_LEN":12,"TRUE_ZERO_HISTORY_RAW_SINGLE_INTERACTION_TARGET_ONLY":727}`.
- Summary rows: `5`.

Food.com zero-history users are true raw single-interaction users: the only chronological raw interaction is the frozen final target, so the final-training prefix i1..i[T-1] is empty. No smoothing, history redefinition, or target movement is applied.

## Reranking Pool Source

- Source audit rows: `20`.
- Source classes: `{"RERANKING_POOL_SOURCE_DEFECT":20}`.
- Tautological pool rows: `4882` / `4882`.

The F1BZ first-stage source is the frozen 1-positive/19-negative ranking candidate set. Since the positive is forced into that set and the per-model top-20 exhausts it, Candidate Recall@20 equals 1.0 by construction. This invalidates the reranking pools and readiness rows for future LLM reranking, but does not invalidate candidate20 ranking metrics.

## Full-Catalog Readiness

- Readiness cells: `20`.
- Conclusions: `{"BLOCKED":16,"FULL_CATALOG_REGENERATION_READY":4}`.

Existing checkpoints load for all 100 trainable runs. LastFM can regenerate non-tautological full-catalog first-stage scores from existing checkpoints because its raw catalog is covered by the model item universe. The other four datasets are blocked for raw full-catalog generation by item-universe coverage, not by checkpoint corruption.

## LastFM Lineage

lineage,source_universe,reranking_users,positive_present_rows,positive_present_rate,candidate_recall_at_20,selected_full_catalog_source_files,selected_models,source_evidence,source_hash,status
historical_f6d2_lastfm,FULL_CATALOG_PREDICTIONS,1859,,0.318450779989,0.318450779989,4,bprmf;gru4rec;lightgcn;sasrec,results/tables/finalization_f6d2_reranking_pool_validation.csv,3bffb09a0757fe84609e8026606378a315c9f7996adab871f8827c7af22d71e0,HISTORICAL_NON_TAUTOLOGICAL_REFERENCE
current_f1bz_lastfm,FROZEN_RANKING_CANDIDATE20,1000,1000,1,1,0,BPRMF;GRU4Rec;LightGCN;SASRec,results/tables/f1bz_candidate_recall.csv,4ea6407fde893fb00b6cdee39ee399852b8bbefdd019cb9d4ad5fdd34d989644,RERANKING_POOL_SOURCE_DEFECT
comparison,FULL_CATALOG_VS_CANDIDATE20,,,,,4,,F6D2 had non-tautological full-catalog predictions; F1BZ uses forced-positive candidate20.,,RERANKING_POOL_SOURCE_DEFECT

## Result Validity

- Classification counts: `{"INVALID":3,"REGENERATE_FROM_EXISTING_CHECKPOINT":4,"RETRAIN_REQUIRED":16,"VALID_AS_IS":5}`.
- Corrective status counts: `{"INVALID":2,"PARTIAL":1,"VALID_AS_IS":2}`.

Candidate20 final traditional metrics and deterministic baselines remain valid as-is. Reranking pools, Candidate Recall@20 as a reranking diagnostic, and reranking readiness require correction before any LLM reranking requests are executed.

## Validation

- Checks: `22`.
- Failures: `0`.
