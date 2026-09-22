# Finalization F6D2 - LastFM trainable traditional models

Decisione: `finalization_f6d2_complete`.

F6D2 addestra controparti locali adattate per BPRMF, GRU4Rec, SASRec e LightGCN sullo split LastFM F6C. I risultati sono multi-seed locali e non sono repliche paper-exact.

## Ambiente e protocollo

- Branch atteso: `plan-d-extensions`.
- HEAD atteso: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`.
- Python: `1` per `PYTHONNOUSERSITE`; Torch `2.6.0+cu124`; CUDA available `True`; CUDA version `12.4`.
- Device configurato: `cpu`; runtime e memoria GPU nei CSV sono normalizzati per hash deterministici.
- Seed policy: `[42, 43, 44, 45, 46]`; run-count paper-like, identita seed non equivalente.

## Risultati aggregati candidate-set 20

- bprmf: HR@5=0.642280796127, NDCG@5=0.506184451135, mean_positive_rank=5.49521247983.
- gru4rec: HR@5=0.700484131254, NDCG@5=0.569194617951, mean_positive_rank=4.75664335664.
- sasrec: HR@5=0.732114039806, NDCG@5=0.602042642275, mean_positive_rank=4.38246369016.
- lightgcn: HR@5=0.710489510489, NDCG@5=0.579107790411, mean_positive_rank=4.59322216245.

## Reranking pool

- Pool paper-aligned: non creato, per policy union/source non recuperata.
- Pool adattato: `data/processed/finalization_f6d2_lastfm_reranking_candidates_adapted.csv` con 1859 utenti e 37180 righe.
- Positive inclusion rate nel pool adattato: 0.318450779989.

## Coverage e gap

- Local numeric mapping coverage: 56/187.
- Protocol-exact numeric replication: 0/187.
- Active missing-evidence count: 26 -> 26.

## Validazione

- Pass: 32; fail: 0.
- Checkpoint disponibili: 20/20.
- Candidate prediction rows: 743600.
- Full-catalog prediction rows: 743600.

## Artefatti

- model_protocol: `results/tables/finalization_f6d2_model_protocol.csv`
- smoke_test: `results/tables/finalization_f6d2_smoke_test.csv`
- execution_manifest: `results/tables/finalization_f6d2_execution_manifest.csv`
- seed_utility_metrics: `results/tables/finalization_f6d2_seed_utility_metrics.csv`
- aggregate_utility_metrics: `results/tables/finalization_f6d2_aggregate_utility_metrics.csv`
- training_curves: `results/tables/finalization_f6d2_training_curves.csv`
- prediction_validation: `results/tables/finalization_f6d2_prediction_validation.csv`
- reranking_policy_audit: `results/tables/finalization_f6d2_reranking_policy_audit.csv`
- reranking_source_selection: `results/tables/finalization_f6d2_reranking_source_selection.csv`
- reranking_pool_validation: `results/tables/finalization_f6d2_reranking_pool_validation.csv`
- lastfm_paper_alignment: `results/tables/finalization_f6d2_lastfm_paper_alignment.csv`
- baseline_comparison: `results/tables/finalization_f6d2_baseline_comparison.csv`
- lastfm_readiness_update: `results/tables/finalization_f6d2_lastfm_readiness_update.csv`
- missing_evidence_update: `results/tables/finalization_f6d2_missing_evidence_update.csv`
- coverage_impact: `results/tables/finalization_f6d2_coverage_impact.csv`
- anomalies: `results/tables/finalization_f6d2_anomalies.csv`
- validation: `results/tables/finalization_f6d2_validation.csv`
- candidate predictions: `results/predictions/finalization_f6d2_lastfm_candidate_rankings.csv`
- full-catalog predictions: `results/predictions/finalization_f6d2_lastfm_full_catalog_rankings.csv`
- adapted reranking pool: `data/processed/finalization_f6d2_lastfm_reranking_candidates_adapted.csv`
