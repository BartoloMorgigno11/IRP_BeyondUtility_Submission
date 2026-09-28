# RQ3 Traditional Scientific Freeze

## Scope

History-length sensitivity of traditional recommenders, retained as a local diagnostic extension. This is post-processing only; raw evidence already existed.

## History grid

`H = {1, 2, 3, 4, 5, 10, 15, 20}` across Beauty, Sports, ML-1M and LastFM.

## Models and H semantics

- **MostPop:** H-invariant train-split popularity control.
- **BM25-like/content:** H truncates the evaluation query history.
- **BPRMF:** retrained on `D'_H`.
- **GRU4Rec:** sequential retraining on `D'_H`.
- **SASRec:** sequential retraining on `D'_H`.
- **LightGCN:** retrained on `D'_H`.

GRU4Rec H1 and SASRec H1 are `NOT_APPLICABLE`: one interaction cannot form a next-item sequence example.

## Repair

Twelve historical Beauty/H5 aggregate keys (four trainable models × HR@5, NDCG@5 and MRR) had separate pilot `n=1` and remaining-seed `n=4` rows. Each was reconstructed deterministically from the existing seed set 42–46. No model execution, inference, recommendation generation or dataset reconstruction was performed.

Corrected aggregate SHA-256: `5c34f309df5dbd6d0c784b43342f0b1211603e8b5cbf73cf08cf087210171d3e`. Validation SHA-256: `b4be6e5b1ad1f8f4e36413b4062d12d1a2bd34c4d7147b5e506bc061153a302a`.

## Scientific status

This is `LOCAL_OPERATIONALIZATION_NOT_PAPER_EXACT`, not a paper-exact ReChorus reproduction. The traditional intervention is not semantically identical to the LLM GE10/GE20 history intervention: MostPop is invariant, BM25 changes query construction, and trainable models retrain on `D'_H`. It supersedes neither the older operational freeze nor its historical artifacts; it adds a separately provenance-tracked corrected post-processing freeze.
