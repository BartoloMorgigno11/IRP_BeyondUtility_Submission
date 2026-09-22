# Finalization F6D1 - LastFM deterministic baselines

## Decision
- Decisione: `finalization_f6d1_complete`.
- Scope: Random, MostPop train-only, BM25-like adapted; no neural training, no LLM inference.

## Inputs
- Candidate source: `data/processed/finalization_f6c_lastfm_ranking_candidates.csv`.
- Train source: `data/processed/finalization_f6c_lastfm_train.csv`.
- Catalog source: `data/processed/finalization_f6c_lastfm_catalog.csv`.

## Protocols
- Random uses explicit seed `42` and SHA-256 per user/candidate.
- MostPop uses train-only interaction counts.
- BM25 is classified as `bm25_like_adapted`, not exact BM25.

## Utility Metrics
- random_seeded HR@5: `0.251210328133`.
- random_seeded NDCG@5: `0.146834303071`.
- random_seeded mean_positive_rank: `10.5228617536`.
- random_seeded median_positive_rank: `11`.
- mostpop_train_only HR@5: `0.79020979021`.
- mostpop_train_only NDCG@5: `0.642689112268`.
- mostpop_train_only mean_positive_rank: `3.82840236686`.
- mostpop_train_only median_positive_rank: `2`.
- bm25_like_adapted HR@5: `0.917159763314`.
- bm25_like_adapted NDCG@5: `0.826437741943`.
- bm25_like_adapted mean_positive_rank: `2.14093598709`.
- bm25_like_adapted median_positive_rank: `1`.

## Validation
- Prediction validation pass/fail: `42/0`.
- F6D1 validation pass/fail: `33/0`.

## Completeness
- random_seeded: users `1859`, prediction rows `37180`, strict-valid users `1859`.
- mostpop_train_only: users `1859`, prediction rows `37180`, strict-valid users `1859`.
- bm25_like_adapted: users `1859`, prediction rows `37180`, strict-valid users `1859`.

## Reranking
- Reusable reranking sources: `0`.
- Reranking candidates were not generated because source evidence is unsupported.

## Missing Evidence And Coverage
- Active gaps before/after: `26/26`; partially resolved by F6D1: `4`.
- LastFM ranking execution coverage: `0/3 deterministic/non-neural local baselines` -> `3/3 Random, MostPop, BM25-like`.
- traditional-baseline execution coverage: `0/6 LastFM paper traditional families locally executed` -> `2/6 paper families locally adapted plus Random control`.
- dataset utility-ready coverage: `4/4 adapted data-ready, LastFM metrics absent` -> `4/4 adapted data-ready with LastFM deterministic utility metrics`.
- reranking-source coverage: `0 supported LastFM sources` -> `0 supported LastFM sources`.
- reranking-candidate readiness: `blocked` -> `blocked`.
- local numeric mapping coverage: `44/187` -> `48/187`.
- protocol-exact numeric replication: `0/187` -> `0/187`.
- active missing-evidence count: `26` -> `26`.

## Anomalies
- Anomalies recorded: `11`; high/critical: `4`.

## Outputs
- Predictions: `results/predictions/finalization_f6d1_lastfm_baseline_rankings.csv`.
- Tables: `13` finalization_f6d1 CSV files.
- Report: `report/finalization_f6d1_lastfm_baselines.md`.
