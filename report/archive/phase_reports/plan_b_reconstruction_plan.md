# Plan B Reconstruction Extension: MovieLens-1M End-to-End Evaluation

## 1. Title

Plan B Reconstruction Extension: MovieLens-1M End-to-End Evaluation

## 2. Purpose of Plan B

Plan A is closed as a readiness-aware reproducibility audit of *Beyond Utility:
Evaluating LLM as Recommender*. It documents that the available Beauty and
Sports artifacts can be normalized and audited, but that real numerical
reproduction is blocked because the original preprocessed files
`sequential_data.txt`, `negative_samples.txt`, `datamaps.json`, and
`meta.json` are unavailable from the official Dropbox links.

Plan B starts a separate reconstruction and extension track. Its purpose is to
build an end-to-end recommendation evaluation pipeline with real computable
metrics on a stable public dataset. Plan B does not replace, correct, or weaken
Plan A. It extends the project in a new direction while preserving Plan A's
claim: Beauty/Sports full numerical reproduction is currently blocked, and no
fake metrics should be written for those datasets.

## 3. Recommended Dataset

MovieLens-1M is the recommended first dataset for Plan B.

Reasons:

- It is a public and stable recommender-system benchmark.
- It provides explicit user-item-rating-timestamp interactions.
- It includes movie metadata, especially titles and genres.
- It is easier to canonicalize than Amazon review data because its core files
  have simpler tabular semantics.
- It is suitable for top-N recommendation experiments.
- It is compatible with a leave-one-out protocol where each user's latest
  interaction becomes the test target and earlier interactions become
  training/history.

MovieLens-1M should be used only after the user manually places the raw files
locally. This B0 phase does not download data and does not create processed
ML-1M files.

## 4. Why This Is Not an Exact Reproduction

The MovieLens-1M reconstruction does not recover the missing Beauty/Sports
preprocessed artifacts and does not guarantee equivalence with the original
paper's experiments.

This is a reconstruction/extension experiment. It is not an exact reproduction
of the original Beauty/Sports experiments. Any metrics computed later on
MovieLens-1M must be reported as Plan B reconstruction metrics, not as exact
paper reproduction results.

Important boundaries:

- Plan B does not make `sequential_data.txt`, `negative_samples.txt`,
  `datamaps.json`, or `meta.json` available for Beauty/Sports.
- Plan B does not validate the original paper's Beauty/Sports HR@5, NDCG@5,
  hallucination, coverage, candidate-bias, history-length, or MostPop numbers.
- Plan B can demonstrate that the project wrapper supports a complete real
  dataset pipeline once canonical files exist.

## 5. Required Canonical Files

Future Plan B phases should create these files only from real local raw
MovieLens-1M files:

| File | Expected role |
|---|---|
| `data/processed/ml1m/interactions.csv` | Canonical user-item interactions with `user_id`, `item_id`, `split`, and optional `timestamp` and `rating`. Train/history rows should be separated from test rows. |
| `data/processed/ml1m/ground_truth.csv` | Test target rows with `user_id` and `ground_truth_item_id`, optionally with `dataset` and `split`. This enables HR@5 and NDCG@5. |
| `data/processed/ml1m/candidates.csv` | Candidate pools with `user_id`, `candidate_item_id`, `candidate_position`, optional `is_ground_truth`, and optional `candidate_item_title`. This enables candidate-based ranking and position diagnostics. |
| `data/processed/ml1m/catalog.csv` | Movie catalog metadata with at least `item_id` and title fields, plus genres when available. This enables catalog coverage and item-text diagnostics. |
| `data/processed/ml1m/preparation_summary.csv` | Diagnostic summary of preprocessing decisions, counts, filters, seed, candidate policy, and output paths. This belongs under processed data because it documents dataset preparation, not metrics. |
| `configs/ml1m.yaml` | Dataset config pointing to the canonical ML-1M files and Plan B output paths. It must keep API calls disabled by default. |

These files are planned additions for future phases. They must not be created
with placeholder or synthetic data.

## 6. Proposed Preprocessing Protocol

A conservative ML-1M preprocessing protocol should:

1. Read raw MovieLens-1M `ratings.dat`.
2. Read raw MovieLens-1M `movies.dat`.
3. Optionally read `users.dat` only for documentation or future analysis.
4. Parse interactions as user, item, rating, and timestamp records.
5. Sort interactions by timestamp within each user.
6. Keep only users with enough interactions to support leave-one-out training
   and testing.
7. Use each retained user's last interaction as the test ground-truth item.
8. Use previous interactions as training/history interactions.
9. Generate negative samples from items the user has not interacted with.
10. Build candidate sets with one positive test item plus a fixed number of
    negative items.
11. Use a deterministic random seed, controlled by `configs/ml1m.yaml`.
12. Build the catalog from movie metadata, including title and genres.
13. Write a preparation summary that records all filters and candidate-sampling
    choices.

The exact minimum-interaction threshold, candidate count, and optional rating
filter should be chosen in B1 and documented in the config and preparation
summary before metrics are computed.

## 7. First Models to Run

Plan B should start with simple baselines:

- Random baseline.
- MostPop baseline.
- Optional BM25/content-based baseline using movie titles and genres.

Random and MostPop are appropriate first because they test the whole data,
recommendation-output, metric, provenance, and plotting pipeline without model
training or API calls. A BM25/content-based baseline can be added after the
basic path works because it uses available movie metadata and remains
lightweight.

BPRMF, SASRec, LightGCN, and LLM reranking should be deferred until the basic
pipeline works. Those models introduce training, larger dependency surfaces,
extra hyperparameters, and additional provenance requirements. LLM reranking
also introduces API or local-generation safety concerns and should remain
blocked unless a later prompt explicitly enables it with `--allow-api-calls`.

## 8. Metrics That Can Become Computable

Once real ML-1M canonical files exist, these metrics can become computable:

- HR@5.
- NDCG@5.
- Coverage@5.
- Catalog coverage diagnostics.
- Possibly popularity-based diagnostics.

These are real metrics only if they are computed from real canonical files.
They must not be placeholders, manually typed values, or copied from unrelated
sources. Until the ML-1M canonical files and recommendation outputs exist,
`results/metrics/` should remain unchanged for Plan B.

## 9. Required Future Code Additions

Future phases should add these files only when their corresponding work starts:

- `src/data/prepare_ml1m.py`
- `configs/ml1m.yaml`
- `src/baselines/random_baseline.py`
- Optional `src/baselines/bm25_baseline.py`
- `tests/test_prepare_ml1m.py`
- `tests/test_random_baseline.py`
- Optional `tests/test_bm25_baseline.py`

B0 creates none of these files. B0 only documents the feasibility plan.

## 10. Risks and Safeguards

Risks:

- ML-1M results may be misread as exact reproduction of the original
  Beauty/Sports experiments.
- Preprocessing choices such as rating filters, candidate count, and negative
  sampling can materially affect metrics.
- Candidate generation can accidentally include seen items or omit the positive
  target if not tested.
- Metric plots could be generated from diagnostics rather than real metric
  rows if output locations are not kept separate.

Safeguards:

- Do not claim exact paper reproduction.
- Keep Plan A unchanged.
- Use deterministic seeds.
- Document preprocessing choices.
- Keep raw data out of git.
- Avoid fake metrics.
- Write diagnostics to `results/tables/`.
- Write real metrics only to `results/metrics/`.
- Keep Plan B results clearly separated from Plan A readiness summaries.
- Keep Beauty/Sports configs unchanged unless a future task explicitly supplies
  real canonical Beauty/Sports artifacts.

## 11. Step-by-Step Execution Roadmap

1. B1 - Prepare MovieLens-1M canonical dataset.
   - Inputs: local `data/raw/ml1m/ratings.dat`, `movies.dat`, and optionally
     `users.dat`.
   - Outputs: canonical ML-1M processed files, `preparation_summary.csv`, and
     `configs/ml1m.yaml`.

2. B2 - Add Random baseline.
   - Inputs: `configs/ml1m.yaml`, canonical candidates, and ground truth.
   - Outputs: `src/baselines/random_baseline.py`, tests, and real random
     recommendation outputs under `results/raw_outputs/` only after execution.

3. B3 - Run MostPop + Random real metrics.
   - Inputs: canonical ML-1M interactions, candidates or user set, ground
     truth, and recommendation outputs.
   - Outputs: real HR@5 and NDCG@5 metric CSVs under `results/metrics/`, plus
     diagnostics under `results/tables/`.

4. B4 - Add BM25/content-based baseline.
   - Inputs: canonical catalog titles/genres, interactions or histories, and
     candidate sets.
   - Outputs: optional BM25 recommendation outputs and tests, with metrics only
     if real outputs are generated.

5. B5 - Run full reconstruction pipeline and plots.
   - Inputs: real metric CSVs from B3/B4 and canonical catalog.
   - Outputs: Plan B pipeline summaries, metric plots generated from real
     metric rows, and no diagnostic plots mislabeled as metric plots.

6. B6 - Update documentation with Plan B extension.
   - Inputs: generated Plan B canonical data summaries, metrics, plots, and
     known limitations.
   - Outputs: README/report updates that clearly label all ML-1M values as
     Plan B reconstruction metrics.

7. B7 - Optional advanced models.
   - Inputs: stable Plan B pipeline, documented compute budget, and explicit
     user approval for any training or API work.
   - Outputs: optional advanced-model recommendations and metrics, separated
     from simple-baseline reconstruction results.

## 12. Success Criteria

Plan B succeeds when:

- Canonical ML-1M files are created from real local raw files.
- `configs/ml1m.yaml` is valid and path-driven.
- MostPop and Random produce recommendation outputs.
- HR@5 and NDCG@5 are computed on ML-1M.
- Coverage@5 is computed if a catalog is available.
- Plots are generated from real metric rows.
- Plan B documentation clearly states that these are reconstruction metrics,
  not exact reproduction results.

## 13. Next Immediate Step

The original next step for B0 was for the user to manually download
MovieLens-1M and place `ratings.dat`, `movies.dat`, and optionally `users.dat`
under `data/raw/ml1m/` before running B1. That prerequisite has been satisfied
locally for the current Plan B run.

## 14. Current B1-B6 Completion Status

Plan B has now progressed from feasibility planning to a completed simple
baseline reconstruction track.

Completed phases:

| Phase | Status | Main outputs |
|---|---|---|
| B1 - Prepare MovieLens-1M canonical dataset | completed | `data/processed/ml1m/interactions.csv`, `ground_truth.csv`, `candidates.csv`, `catalog.csv`, `configs/ml1m.yaml` |
| B2 - Add Random baseline | completed | `results/metrics/ml1m_random_metrics.csv`, `results/tables/ml1m_random_baseline_summary.csv` |
| B3 - Add MostPop baseline | completed | `results/metrics/ml1m_mostpop_metrics.csv`, `results/tables/ml1m_mostpop_baseline_summary.csv` |
| B4 - Add ContentBM25 baseline | completed | `results/metrics/ml1m_content_bm25_metrics.csv`, `results/tables/ml1m_content_bm25_baseline_summary.csv` |
| B5 - Full reconstruction runner and plots | completed | `results/tables/ml1m_reconstruction_pipeline_summary.csv`, `results/plots/ml1m_baseline_*.png` |
| B6 - Final documentation consolidation | completed in documentation | README, technical report, results analysis, checklist, notes, and prompt log updates |

Current real Plan B metric results:

| Model | HR@5 | NDCG@5 |
|---|---:|---:|
| Random | 0.0514900662251655 | 0.0293715251496116 |
| MostPop | 0.320364238410596 | 0.2102806322093436 |
| ContentBM25 | 0.2508278145695364 | 0.1647567964031986 |

These remain MovieLens-1M reconstruction metrics. They do not recover the
missing Beauty/Sports preprocessed artifacts and must not be presented as exact
reproduction of the original paper's Beauty/Sports results.

The next optional phase is B7: advanced models or LLM reranking. B7 should only
be attempted with explicit scope, compute budget, and safeguards against API
calls or training unless the user requests them.
