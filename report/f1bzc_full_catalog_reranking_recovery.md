# F1BZC Full-Catalog Reranking Recovery

Decision token: `F1BZC_RERANKING_SOURCE_BLOCKED`

## Source-Seed Policy

- Rows: `20`.
- Frozen rule: all five preregistered seeds [42,43,44,45,46] are mean-aggregated.
- Final-test metrics are not used for source-seed selection.

## Full-Catalog Recovery

- Adapter statuses: `{"BLOCKED_SCIENTIFIC_ITEM_UNIVERSE_COVERAGE":16,"FULL_CATALOG_REGENERATION_READY":4}`.
- Execution statuses: `{"BLOCKED_SCIENTIFIC_ITEM_UNIVERSE_COVERAGE":16,"CORRECTED_F1BZC":4}`.

LastFM is the only dataset whose frozen F1BZ checkpoint item map covers the raw catalog. The other four datasets have raw catalog items absent from the checkpoint embedding table, so raw full-catalog scoring would require invented embeddings or retraining. F1BZC does neither.

## Adapted Pool Algorithm

Pseudocode:

```text
for model in [BPRMF, GRU4Rec, SASRec, LightGCN]:
  for item in that model's top20 full-catalog list:
    append item if not already present
    stop once the final pool has 20 items
```

- Contribution totals: `{"BPRMF":20000,"GRU4Rec":0,"LightGCN":0,"SASRec":0}`.
- Pool validation statuses: `{"BLOCKED":2,"PASS":4}`.

Applied exactly, the frozen F1BR policy is first-model dominated: BPRMF fills the full 20-item pool before the later models can contribute. F1BZC therefore blocks authoritative reranking-pool freeze instead of hiding a second protocol defect.

## Candidate Recall

dataset,source_policy,evaluation_N,positive_present_N,positive_absent_N,candidate_recall_at_20,warm_user_N,warm_positive_present_N,warm_candidate_recall_at_20,cold_user_N,cold_positive_present_N,cold_candidate_recall_at_20,candidate_recall_tautological,authoritative,status,reason
beauty,F1BZC_FULL_CATALOG_FIVE_SEED_MEAN,990,,,,,,,,,,not_evaluated,false,BLOCKED_SCIENTIFIC_ITEM_UNIVERSE_COVERAGE,No raw full-catalog pool could be produced without invented item embeddings.
sports,F1BZC_FULL_CATALOG_FIVE_SEED_MEAN,892,,,,,,,,,,not_evaluated,false,BLOCKED_SCIENTIFIC_ITEM_UNIVERSE_COVERAGE,No raw full-catalog pool could be produced without invented item embeddings.
ml1m,F1BZC_FULL_CATALOG_FIVE_SEED_MEAN,1000,,,,,,,,,,not_evaluated,false,BLOCKED_SCIENTIFIC_ITEM_UNIVERSE_COVERAGE,No raw full-catalog pool could be produced without invented item embeddings.
lastfm,F1BZC_FULL_CATALOG_FIVE_SEED_MEAN,1000,409,591,0.40899999999999997,1000,409,0.40899999999999997,0,,,false,false,BLOCKED_ADAPTED_POOL_SINGLE_MODEL_DOMINANCE,Recall is non-tautological but not authoritative because the adapted pool algorithm is single-model dominated.
foodcom,F1BZC_FULL_CATALOG_FIVE_SEED_MEAN,1000,,,,,,,,,,not_evaluated,false,BLOCKED_SCIENTIFIC_ITEM_UNIVERSE_COVERAGE,No raw full-catalog pool could be produced without invented item embeddings.

## Food.com Warm/Cold Diagnostic

- Rows: `183`.
- Food.com true zero-history users remain in the cohort; no interactions are invented or removed.

## Superseded Artifacts

- Superseded forensic artifacts: `7`.
- Old F1BZ first-stage rankings, reranking pools, candidate recall, and readiness remain preserved but are not authoritative downstream inputs.

## Readiness

- Readiness rows: `30`.
- Readiness statuses: `{"BLOCKED_ADAPTED_POOL_SINGLE_MODEL_DOMINANCE":6,"BLOCKED_FULL_CATALOG_ITEM_UNIVERSE_COVERAGE":24}`.
- Future requests unlocked: `0` / `29292`.

## Validation

- F1BZC audit checks: `22`.
- Failures: `0`.
- Full-catalog validation rows: `20`.
- Result validity classes: `{"BLOCKED_ADAPTED_POOL_SINGLE_MODEL_DOMINANCE":2,"BLOCKED_SCIENTIFIC_ITEM_UNIVERSE_COVERAGE":20,"CORRECTED_F1BZC":4,"SUPERSEDED_RERANKING_SOURCE_DEFECT":2,"VALID_AS_IS":3}`.
