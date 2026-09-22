# Plan C Final Report: Full-Paper Replication and Reconstruction Campaign

## Executive Summary

Plan C is the maximum-fidelity replication and reconstruction campaign for
*Beyond Utility: Evaluating LLM as Recommender*. It does not claim a fake full
reproduction. Instead, it documents what can be exactly audited, what remains
blocked, and what can be reconstructed from public raw data.

The main outcome is a defensible three-part result:

1. Exact Beauty/Sports paper reproduction remains blocked because the official
   processed artifacts are unavailable.
2. Public raw datasets were recovered, validated, and converted into Plan C
   reconstructed canonical datasets.
3. Deterministic baselines were evaluated on reconstructed datasets, producing
   real metrics that are clearly labeled as reconstructed-dataset results.

## Objective of the Project

The project objective was to evaluate how far the original paper can be
reproduced from the available repository, artifacts, and public data sources.
The work prioritizes reproducibility discipline over unsupported numerical
claims. Missing ground truth, candidate sets, or catalogs are treated as
blockers, not as opportunities to invent data.

## Original Paper Target and Intended Reproduction Scope

The target paper is *Beyond Utility: Evaluating LLM as Recommender*. The
intended Plan C scope was broad: datasets, candidate-set ranking tasks,
prompting strategies, LLM outputs, traditional baselines, utility metrics, and
beyond-utility evaluation dimensions.

The local campaign inspected the original repository, artifact references,
prompt/output folders, dataset expectations, and existing project reports.

## Exact-Reproduction Blockers

Exact Beauty/Sports reproduction remains blocked. The required official
processed files are still missing:

- `sequential_data.txt`;
- `negative_samples.txt`;
- `datamaps.json`;
- paper processed `meta.json`.

The official Dropbox processed-artifact links were manually tested and are not
recoverable in the current environment. Without these files, exact Beauty and
Sports HR@5/NDCG@5 and beyond-utility metrics cannot be claimed.

## Artifact Recovery Process

Plan C created a replication matrix, artifact manifest, validator, external
source ledger, manual download checklist, author-contact draft, and staging
inspector. This process distinguishes official original artifacts, raw public
datasets, reconstructed artifacts, local extensions, unavailable files, and
unknown sources.

This distinction is central: a public raw dataset is useful for reconstruction,
but it is not automatically equivalent to an author processed artifact.

## Raw Dataset Recovery and Validation

The following public raw sources were manually staged and validated:

- Amazon Beauty 5-core reviews and metadata;
- Amazon Sports and Outdoors 5-core reviews and metadata;
- MovieLens-1M;
- LastFM HetRec.

Schema checks confirmed that Amazon and ML-1M support chronological
leave-one-out reconstruction. LastFM provides user-artist interactions and
artist metadata, but lacks listening timestamps in `user_artists.dat`, so it
does not support the same chronological utility-evaluation protocol.

## Reconstruction Methodology

Reconstructed datasets are written under `data/reconstructed/plan_c/` and are
ignored by Git as generated data. For Amazon and ML-1M, the reconstruction
protocol is:

- sort interactions chronologically per user;
- keep users with at least two interactions;
- use the last interaction as ground truth;
- use earlier interactions as training/history;
- generate one positive plus fixed deterministic negatives per user;
- build canonical `interactions.csv`, `ground_truth.csv`, `candidates.csv`,
  and `catalog.csv`;
- write paper-like `sequential_data.txt`, `negative_samples.txt`,
  `datamaps.json`, and `meta.json` as reconstructed-format artifacts.

Final C5 reconstruction status:

| Dataset | Status | Train | Ground Truth | Candidates | Catalog | Candidate Universe |
|---|---:|---:|---:|---:|---:|---|
| ML-1M | completed | 994169 | 6040 | 604000 | 3883 | interaction_items |
| Amazon Beauty | completed | 176139 | 22363 | 2236300 | 259204 | interaction_items |
| Amazon Sports | completed | 260739 | 35598 | 3559800 | 532197 | interaction_items |
| LastFM | partial_extension_only | 92834 | 0 | 0 | 17632 | not applicable |

## Candidate Generation and Negative Sampling Correction

During C6 review, a methodological issue was identified: catalog-wide negative
sampling can make popularity baselines artificially strong when the metadata
catalog is much larger than the observed interaction item universe. This was
especially relevant for Amazon:

- Amazon Beauty observed interaction items: 12101; catalog rows: 259204.
- Amazon Sports observed interaction items: 18357; catalog rows: 532197.

The default was corrected to:

```yaml
negative_sampling:
  item_universe: interaction_items
```

This samples negatives from observed interaction items while excluding each
user's full interaction history. `catalog_items` remains available only as an
explicit experimental mode. Suspicious Amazon MostPop values produced before
this correction are superseded and are not treated as final.

## Baseline Evaluation Protocol

Plan C C6 evaluates deterministic non-LLM baselines:

- Random;
- MostPop;
- ContentBM25.

Metrics are computed only when reconstructed ground truth and candidates exist.
The primary metrics are HR@5 and NDCG@5, with Catalog Coverage@5 reported when
catalog data is available. Full recommendation outputs are generated artifacts
under `results/raw_outputs/` and ignored by Git.

## Results

Final consolidated results are stored in
`results/tables/plan_c_final_results_overview.csv`.

| Dataset | Baseline | Run Scope | HR@5 | NDCG@5 | Evaluated Users |
|---|---|---|---:|---:|---:|
| Amazon Beauty | Random | full | 0.051111210481599 | 0.0297061760161657 | 22363 |
| Amazon Beauty | MostPop | full | 0.1865581540938156 | 0.1218049913823627 | 22363 |
| Amazon Beauty | ContentBM25 | diagnostic_max_users_1000 | 0.317 | 0.2303098637007468 | 1000 |
| Amazon Sports | Random | full | 0.0487948761166357 | 0.0288092328987212 | 35598 |
| Amazon Sports | MostPop | full | 0.2109669082532726 | 0.1415364725903375 | 35598 |
| Amazon Sports | ContentBM25 | diagnostic_max_users_1000 | 0.262 | 0.182786145361599 | 1000 |
| ML-1M | Random | full | 0.0511589403973509 | 0.0284872079551804 | 6040 |
| ML-1M | MostPop | full | 0.3013245033112582 | 0.1989268049046611 | 6040 |
| ML-1M | ContentBM25 | full | 0.2375827814569536 | 0.1562358579730421 | 6040 |

MostPop is stronger than Random on all full evaluated datasets. ContentBM25 is
competitive, but Amazon ContentBM25 rows are diagnostic subset rows and should
not be interpreted as final full-dataset Amazon results.

## Diagnostic vs Full Results

Rows with `run_scope=full` are final within the reconstructed Plan C protocol.
Rows with `run_scope=diagnostic_max_users_1000` are runtime/stability checks.
They are useful for validating code paths but should not be presented as full
Amazon results.

## LastFM Limitation

LastFM is `extension_only`. It has reconstructed interactions and catalog
metadata, but no chronological ground truth or candidate sets. Therefore, no
LastFM HR@5/NDCG@5 metrics are written. This is intentional: computing utility
metrics without a valid split would be fake evaluation.

## What Can Be Claimed

The project can claim:

- the original repository and artifact surface were audited;
- exact Beauty/Sports reproduction is blocked by missing official artifacts;
- public raw datasets were recovered and schema-validated;
- Amazon Beauty, Amazon Sports, and ML-1M were reconstructed into canonical
  Plan C datasets;
- deterministic baselines produced real metrics on reconstructed datasets;
- candidate generation was corrected to use interaction-item negatives by
  default;
- LastFM was honestly blocked for utility metrics.

## What Cannot Be Claimed

The project cannot claim:

- exact reproduction of the original Beauty/Sports paper numbers;
- recovery of the official Beauty/Sports processed artifacts;
- equivalence between reconstructed public-raw datasets and author processed
  datasets;
- LLM/API reproduction of the paper's closed or hosted model outputs;
- full LastFM utility evaluation;
- final full Amazon ContentBM25 results unless full runs are executed later.

## Threats to Validity

Key threats are:

- missing official processed artifacts;
- possible mismatch between reconstructed preprocessing and author
  preprocessing;
- candidate-set evaluation sensitivity to negative sampling;
- model/version drift for any future LLM/API reproduction;
- diagnostic subset rows being misread as full results;
- LastFM lacking timestamps for chronological leave-one-out splitting.

## Reproducibility Checklist

Core commands:

```powershell
python -m pytest tests/
python src/data/reconstruct_plan_c_datasets.py --config configs/plan_c_reconstruction.yaml --datasets ml1m amazon_beauty amazon_sports lastfm
python src/experiments/run_plan_c_baselines.py --datasets ml1m --baselines random mostpop content_bm25
python src/experiments/run_plan_c_baselines.py --datasets amazon_beauty amazon_sports --baselines random mostpop
python src/experiments/summarize_plan_c_final_results.py
```

Generated final tables:

- `results/tables/plan_c_final_results_overview.csv`;
- `results/tables/plan_c_claims_and_limitations.csv`;
- `results/tables/plan_c_final_experiment_inventory.csv`.

Generated final plots:

- `results/plots/plan_c_final_hr_at_5_overview.png`;
- `results/plots/plan_c_final_ndcg_at_5_overview.png`.

## Suggested Oral-Exam Narrative

The strongest narrative is:

1. I first tried to reproduce the paper honestly from the official artifacts.
2. The exact Beauty/Sports reproduction is blocked because the official
   processed files are unavailable.
3. Instead of fabricating metrics, I documented the blocker and built a
   reconstruction track from public raw data.
4. I validated raw schemas, reconstructed canonical datasets, corrected a
   negative-sampling issue, and ran deterministic baselines.
5. The resulting metrics are real, but they are reconstructed-dataset metrics,
   not exact original paper results.

## Future Extensions

Future work should:

- contact the authors for official processed artifacts;
- reproduce paper prompt templates and LLM outputs only with explicit API/model
  approval;
- run full Amazon ContentBM25 if needed;
- add BPRMF, SASRec, LightGCN, and other traditional baselines;
- extend beyond-accuracy diagnostics on reconstructed outputs;
- investigate whether LastFM can support a defensible non-chronological
  protocol or requires author processed data.
