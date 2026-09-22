# Reproducibility Checklist

This checklist is an evidence-based status record for the partial
reproducibility project around *Beyond Utility: Evaluating LLM as Recommender*.
It does not claim that the full paper was reproduced.

## 1. Project Identity

| Item | Status |
|---|---|
| Paper title | *Beyond Utility: Evaluating LLM as Recommender* |
| Project goal | Build a clean, reproducibility-oriented evaluation wrapper around the original repository and locally available artifacts, plus a separate ML-1M reconstruction track. |
| Reproduction type | Plan A: partial, readiness-aware reproducibility audit. Plan B: MovieLens-1M reconstruction/extension. |
| Datasets considered | Plan A: Beauty and Sports. Plan B: MovieLens-1M. |
| Reason for focus | Beauty/Sports have local saved LLM output archives; Beauty also has a local traditional-baseline archive. ML-1M is stable and reconstructible when raw files are manually supplied. |

## 2. Original Paper Scope

The original paper covers multiple datasets, multiple LLMs, traditional
recommendation baselines, ranking and re-ranking tasks, utility metrics, and
beyond-utility dimensions such as hallucination, candidate position bias,
history length sensitivity, and generation-involved behavior.

This project implements a constrained local wrapper around recoverable artifacts
instead of rerunning the complete original study.

Plan B is explicitly separate from this original-paper audit. It uses
MovieLens-1M to demonstrate an end-to-end evaluation pipeline with real metrics
once canonical files exist.

## 3. Current Reproduction Status

Status vocabulary: `reproduced`, `partially reproduced`,
`implemented but blocked`, `blocked`, `not attempted`, `not applicable`.

| Component | Status | Evidence | Notes |
|---|---|---|---|
| Original repository inspected | partially reproduced | `report/codebase_map.md`, `report/original_artifact_map.md` | Read-only inspection only; upstream code was not modified or executed for metrics. |
| Codebase mapped | reproduced | `report/codebase_map.md` | Repository structure, LLM outputs, SOTA outputs, scripts, and risks documented. |
| Configs created | reproduced | `configs/beauty.yaml`, `configs/sports.yaml` | Canonical data paths remain null because required artifacts are missing. |
| Dataset audit implemented | reproduced | `src/data/audit_datasets.py`, `results/tables/*_dataset_audit.csv`, `report/dataset_audit.md` | Audit reports unknown stats where canonical data is unavailable. |
| LLM outputs loaded/normalized | partially reproduced | `results/raw_outputs/beauty_llm_outputs_normalized.csv`, `results/raw_outputs/sports_llm_outputs_normalized.csv` | Text recommendations are normalized; item-id matching remains blocked without catalog mappings. |
| Baseline outputs loaded/normalized where available | partially reproduced | `results/raw_outputs/beauty_baseline_outputs_normalized.csv`, `results/tables/*_baseline_outputs_summary.csv` | Beauty BPRMF artifact is normalized; Sports baseline outputs are missing/not configured. |
| Utility metrics implemented | implemented but blocked | `src/evaluation/utility_metrics.py`, `tests/test_utility_metrics.py` | HR@5/NDCG@5 computation is blocked for current configs because ground truth is unavailable. |
| Hallucination metrics implemented | implemented but blocked | `src/evaluation/hallucination.py`, `tests/test_hallucination.py` | Real hallucination computation is blocked because catalog/candidates are unavailable. |
| Coverage extension implemented | implemented but blocked | `src/evaluation/coverage.py`, `src/experiments/run_extension_coverage.py`, `tests/test_coverage.py` | Real Catalog Coverage@5 is blocked because catalog paths are null. |
| Position-bias utilities implemented | implemented but blocked | `src/prompts/candidate_ordering.py`, `src/evaluation/position_bias.py`, `src/experiments/run_position_bias.py`, `tests/test_position_bias.py` | Real position-bias metrics are blocked because candidate sets are unavailable. |
| History-length utilities implemented | implemented but blocked | `src/prompts/history_length.py`, `src/experiments/run_history_length.py`, `tests/test_history_length.py` | Prompt variants are blocked because user histories/interactions are unavailable. |
| Cost/token utilities implemented | implemented but blocked | `src/evaluation/cost_metrics.py`, `configs/cost.yaml`, `tests/test_cost_metrics.py` | Real costs are unavailable because token/latency records and non-null prices are not configured. |
| Final pipeline implemented | partially reproduced | `src/experiments/run_pipeline.py`, `results/tables/*_final_summary.csv` | Pipeline records readiness and blockers; no final metrics CSV is written because no real metrics exist. |
| Plots implemented | partially reproduced | `src/experiments/make_plots.py`, `results/plots/*_readiness_status_counts.png` | Readiness plots exist; metric plots are skipped because metric rows are unavailable. |
| Quality review completed | reproduced | `report/quality_review.md` | Review documents fixed and deferred issues. |
| README completed | reproduced | `README.md` | README states partial reproduction and current blockers. |
| Plan B ML-1M canonical preparation | reproduced as reconstruction | `src/data/prepare_ml1m.py`, `configs/ml1m.yaml`, `results/tables/ml1m_dataset_preparation_summary.csv` | Raw files must be placed manually; no automatic download. |
| Plan B Random baseline | reproduced as reconstruction | `src/baselines/random_baseline.py`, `results/metrics/ml1m_random_metrics.csv` | Deterministic sanity-check baseline over candidate sets. |
| Plan B MostPop baseline | reproduced as reconstruction | `src/baselines/ml1m_mostpop_baseline.py`, `results/metrics/ml1m_mostpop_metrics.csv` | Popularity is computed only from training/history interactions. |
| Plan B ContentBM25 baseline | reproduced as reconstruction | `src/baselines/content_bm25_baseline.py`, `results/metrics/ml1m_content_bm25_metrics.csv` | Uses title/genre catalog text and training-only user profiles. |
| Plan B reconstruction runner and plots | reproduced as reconstruction | `src/experiments/run_ml1m_reconstruction.py`, `results/tables/ml1m_reconstruction_pipeline_summary.csv`, `results/plots/ml1m_baseline_*.png` | Generates comparison plots from real ML-1M metric rows. |

## 4. Dataset Artifact Checklist

### Beauty

| Artifact | Status | Evidence | Notes |
|---|---|---|---|
| LLM outputs | available | `original_repo/EvaLLMasRecommender/llm_results/beauty.zip` | Normalized under `results/raw_outputs/`. |
| Traditional baseline outputs | partially available | `original_repo/EvaLLMasRecommender/sota_results/beauty_his10.zip` | Configured BPRMF output normalized; full evaluation blocked by missing ground truth. |
| Preprocessed data archive | blocked | `report/original_artifact_map.md` | README-linked Dropbox item is unavailable/deleted. |
| `sequential_data.txt` | blocked | `report/original_artifact_map.md` | Not present locally; needed for histories and target items. |
| `negative_samples.txt` | blocked | `report/original_artifact_map.md` | Not present locally; needed for full candidate sets. |
| `datamaps.json` | blocked | `report/original_artifact_map.md` | Not present locally; needed for item-id/ASIN mappings. |
| `meta.json` | blocked | `report/original_artifact_map.md` | Not present locally; needed for catalog metadata and titles. |
| Ground truth | blocked | `configs/beauty.yaml`, `report/dataset_audit.md` | `ground_truth_path` is null. |
| Candidate sets | blocked | `configs/beauty.yaml`, `report/dataset_audit.md` | `candidates_path` is null; prompt `cand_list` exists only as sampled prompt data. |
| Catalog metadata | blocked | `configs/beauty.yaml`, `report/dataset_audit.md` | `catalog_path` is null. |
| Interaction histories | blocked | `configs/beauty.yaml`, `report/dataset_audit.md` | `interactions_path` is null. |

### Sports

| Artifact | Status | Evidence | Notes |
|---|---|---|---|
| LLM outputs | available | `original_repo/EvaLLMasRecommender/llm_results/sports.zip` | Normalized under `results/raw_outputs/`; one empty normalized recommendation row is skipped by readiness checks. |
| Traditional baseline outputs | blocked | `configs/sports.yaml`, `results/tables/sports_baseline_outputs_summary.csv` | No Sports SOTA output archive is present or configured. |
| Preprocessed data archive | blocked | `report/original_artifact_map.md` | README-linked Dropbox item is unavailable/deleted. |
| `sequential_data.txt` | blocked | `report/original_artifact_map.md` | Not present locally; needed for histories and target items. |
| `negative_samples.txt` | blocked | `report/original_artifact_map.md` | Not present locally; needed for full candidate sets. |
| `datamaps.json` | blocked | `report/original_artifact_map.md` | Not present locally; needed for item-id/ASIN mappings. |
| `meta.json` | blocked | `report/original_artifact_map.md` | Not present locally; needed for catalog metadata and titles. |
| Ground truth | blocked | `configs/sports.yaml`, `report/dataset_audit.md` | `ground_truth_path` is null. |
| Candidate sets | blocked | `configs/sports.yaml`, `report/dataset_audit.md` | `candidates_path` is null; prompt `cand_list` exists only as sampled prompt data. |
| Catalog metadata | blocked | `configs/sports.yaml`, `report/dataset_audit.md` | `catalog_path` is null. |
| Interaction histories | blocked | `configs/sports.yaml`, `report/dataset_audit.md` | `interactions_path` is null. |

## 5. Metric And Analysis Checklist

| Metric or analysis | Implementation status | Computation status | Required inputs | Current blocker | Output location if produced |
|---|---|---|---|---|---|
| HR@5 | implemented | blocked | Normalized recommendations and canonical ground truth | `ground_truth_path` is null for Beauty and Sports | No metric CSV written. |
| NDCG@5 | implemented | blocked | Normalized recommendations and canonical ground truth | `ground_truth_path` is null for Beauty and Sports | No metric CSV written. |
| Hallucination rate | implemented | blocked | LLM recommendations plus catalog or candidate reference | `catalog_path` and `candidates_path` are null | Diagnostic summaries under `results/tables/*_reproduction_summary.csv`. |
| Catalog Coverage@K | implemented | blocked | Recommendation item ids and canonical catalog | `catalog_path` is null | Diagnostic summaries under `results/tables/*_coverage_summary.csv`; no coverage metrics CSV. |
| Candidate Position Bias | implemented | blocked | Candidate sets and matched recommendation outputs | `candidates_path` is null | Diagnostic summaries under `results/tables/*_position_bias_summary.csv`; no position-bias metrics CSV. |
| History Length Sensitivity | implemented | blocked | User histories or interactions; later, model outputs and ground truth for metrics | `history_path`/`interactions_path` unavailable | Diagnostic summaries under `results/tables/*_history_length_summary.csv`; no prompt variants or metrics CSV. |
| Cost/token metrics | implemented | blocked | Token counts, latency logs, and explicit non-null pricing assumptions | Token/latency inputs missing; `configs/cost.yaml` prices are null | No real cost metrics CSV. |
| Readiness/status plots | implemented | produced | Existing readiness summary CSVs | Not blocked | `results/plots/beauty_readiness_status_counts.png`, `results/plots/sports_readiness_status_counts.png`. |

For Plan A Beauty/Sports, no fake, empty, or placeholder metric CSVs are used
as reproduced results. For Plan B ML-1M, real metric CSVs exist because
canonical ground truth, candidates, interactions, and recommendations exist.

## 5.1 Plan B ML-1M Metric Checklist

| Metric or analysis | Status | Evidence | Notes |
|---|---|---|---|
| Random HR@5 | computed | `results/metrics/ml1m_random_metrics.csv` | Value: `0.0514900662251655`. |
| Random NDCG@5 | computed | `results/metrics/ml1m_random_metrics.csv` | Value: `0.0293715251496116`. |
| MostPop HR@5 | computed | `results/metrics/ml1m_mostpop_metrics.csv` | Value: `0.320364238410596`; popularity uses training/history interactions only. |
| MostPop NDCG@5 | computed | `results/metrics/ml1m_mostpop_metrics.csv` | Value: `0.2102806322093436`. |
| ContentBM25 HR@5 | computed | `results/metrics/ml1m_content_bm25_metrics.csv` | Value: `0.2508278145695364`; profile uses training/history items only. |
| ContentBM25 NDCG@5 | computed | `results/metrics/ml1m_content_bm25_metrics.csv` | Value: `0.1647567964031986`. |
| Baseline comparison | produced | `results/tables/ml1m_baseline_comparison_summary.csv` | Contains six real available rows and no invented values. |
| ML-1M plots | produced | `results/plots/ml1m_baseline_hr_at_5.png`, `results/plots/ml1m_baseline_ndcg_at_5.png`, `results/plots/ml1m_baseline_comparison.png` | Generated only from real comparison rows. |

## 6. Baseline Checklist

| Baseline or output source | Status | Evidence | Notes |
|---|---|---|---|
| Beauty LLM output | partially reproduced | `results/raw_outputs/beauty_llm_outputs_normalized.csv` | Normalized from cached original output; utility still blocked by missing ground truth. |
| Sports LLM output | partially reproduced | `results/raw_outputs/sports_llm_outputs_normalized.csv` | Normalized from cached original output; utility still blocked by missing ground truth. |
| Beauty BPRMF output | partially reproduced | `results/raw_outputs/beauty_baseline_outputs_normalized.csv` | Existing baseline output normalized; evaluation blocked by missing ground truth. |
| Sports traditional baseline output | blocked | `results/tables/sports_baseline_outputs_summary.csv` | No local configured Sports baseline output exists. |
| MostPop | implemented but blocked | `src/baselines/mostpop.py`, `tests/test_mostpop.py` | Requires canonical `interactions_path`; not generated for current Beauty/Sports configs. |
| BPR, SASRec, LightGCN, GRU4Rec, other training | not attempted | `report/quality_review.md`, `README.md` | Traditional models were not rerun or trained. |
| Plan B Random | computed | `src/baselines/random_baseline.py` | Deterministic candidate-set baseline; no training. |
| Plan B MostPop | computed | `src/baselines/ml1m_mostpop_baseline.py` | Non-personalized popularity baseline; no held-out leakage in popularity counts. |
| Plan B ContentBM25 | computed | `src/baselines/content_bm25_baseline.py` | Lightweight content baseline using title/genre text; no external dependency or training. |

## 7. Reproducibility Blockers

- The external README-linked Beauty and Sports Dropbox preprocessed data links
  are unavailable/deleted.
- Canonical ground truth is missing.
- Canonical candidate sets are missing.
- Canonical catalog metadata is missing.
- Canonical interaction histories are missing.
- Sports traditional baseline outputs are missing/not configured.
- No API calls were performed.
- No model training was performed.
- No datasets were downloaded by the project pipeline.
- Plan B raw MovieLens-1M files must be downloaded manually by the user and
  placed under `data/raw/ml1m/`.
- Plan B does not recover or replace the missing Beauty/Sports preprocessed
  artifacts.

## 8. Evidence Files

Main evidence files for this checklist:

- `report/original_artifact_map.md`
- `results/tables/original_artifact_inventory.csv`
- `report/dataset_audit.md`
- `report/results_analysis.md`
- `report/quality_review.md`
- `README.md`
- `tests/`
- `results/tables/*_summary.csv`
- `results/raw_outputs/*_normalized.csv`
- `results/plots/*_readiness_status_counts.png`
- `results/metrics/ml1m_*_metrics.csv`
- `results/tables/ml1m_baseline_comparison_summary.csv`
- `results/tables/ml1m_reconstruction_pipeline_summary.csv`
- `results/plots/ml1m_baseline_*.png`

## 8.1 Plan B Reproduction Commands

Raw MovieLens-1M files must be placed manually:

```text
data/raw/ml1m/ratings.dat
data/raw/ml1m/movies.dat
data/raw/ml1m/users.dat
```

No automatic download is performed.

Prepare canonical ML-1M files:

```powershell
python src/data/prepare_ml1m.py --config configs/ml1m.yaml
```

Run the full Plan B pipeline:

```powershell
python src/experiments/run_ml1m_reconstruction.py --config configs/ml1m.yaml
```

Run tests:

```powershell
python -m pytest tests/
```

Expected Plan B metric outputs:

- `results/metrics/ml1m_random_metrics.csv`;
- `results/metrics/ml1m_mostpop_metrics.csv`;
- `results/metrics/ml1m_content_bm25_metrics.csv`.

Expected Plan B plot outputs:

- `results/plots/ml1m_baseline_hr_at_5.png`;
- `results/plots/ml1m_baseline_ndcg_at_5.png`;
- `results/plots/ml1m_baseline_comparison.png`.

Large generated recommendation CSVs under `results/raw_outputs/ml1m_*_recommendations.csv`
are ignored by Git and should not be committed.

## 8.2 Plan C Artifact Validation

Plan C starts a separate full-paper replication campaign. It does not replace
Plan A and does not relabel Plan B metrics as paper reproduction results.

Plan C artifact recovery files:

- `report/plan_c_full_replication_matrix.md`;
- `report/plan_c_artifact_recovery_protocol.md`;
- `data/artifact_manifests/plan_c_artifact_manifest_template.csv`;
- `data/artifact_manifests/plan_c_artifact_manifest.csv`;
- `results/tables/plan_c_replication_matrix.csv`;
- `results/tables/plan_c_required_artifacts.csv`;
- `results/tables/plan_c_models_metrics_tasks_inventory.csv`.

Each manifest row must classify artifacts as official, author-released,
third-party, reconstructed, extension-only, unavailable, or unknown. Each row
also records a reproducibility status such as `blocked_missing_artifact`,
`blocked_unavailable_url`, `requires_manual_download`, or `extension_only`.

Run the local-only validator:

```powershell
python src/audit/validate_plan_c_artifacts.py --manifest data/artifact_manifests/plan_c_artifact_manifest.csv
```

Optionally check SHA-256 values when expected hashes are recorded:

```powershell
python src/audit/validate_plan_c_artifacts.py --manifest data/artifact_manifests/plan_c_artifact_manifest.csv --hash
```

Validator outputs:

- `results/tables/plan_c_artifact_validation_summary.csv`;
- `results/tables/plan_c_artifact_validation_details.csv`.

The validator does not download files, call APIs, train models, compute
metrics, modify raw artifacts, or modify `original_repo/`. Newly recovered
artifacts should be manually placed outside `original_repo/`, preferably under
`data/raw/plan_c/<dataset>/`, then recorded in the manifest before rerunning
the validator.

## 8.3 Plan C External Source Checks

Plan C C2 adds a metadata-only source check and recovery ledger. This phase
does not download data and does not prove that an HTML page response contains a
valid processed artifact.

Run the URL checker:

```powershell
python src/audit/check_plan_c_source_urls.py --manifest data/artifact_manifests/plan_c_artifact_manifest.csv
```

Optional timeout override:

```powershell
python src/audit/check_plan_c_source_urls.py --manifest data/artifact_manifests/plan_c_artifact_manifest.csv --timeout 10
```

URL checker output:

- `results/tables/plan_c_source_url_check_results.csv`.

Source recovery outputs:

- `results/tables/plan_c_external_source_recovery_ledger.csv`;
- `report/plan_c_external_source_recovery_report.md`;
- `report/plan_c_manual_download_checklist.md`;
- `report/plan_c_author_contact_request.md`.

Interpretation rules:

- Official processed artifacts can support exact reproduction only after the
  real files are manually recovered and validated.
- Official raw datasets are reconstruction inputs, not proof of exact paper
  equivalence.
- Dropbox or Google Drive page reachability must not be treated as proof that
  a linked ZIP is valid or downloadable.
- `network_unverified` means the current check could not establish status; it
  must not be rewritten as unavailable without separate evidence.
- Newly recovered files should be placed under `data/raw/plan_c/<dataset>/`,
  kept out of Git, recorded in the manifest, and validated before any converter
  or metric step.

## 8.4 Plan C Manual Artifact Intake

Plan C C3 adds a safe staging workflow for manually downloaded or author-sent
artifacts. The staging directory is:

```text
data/external/plan_c/
```

Only `data/external/plan_c/.gitkeep` should be tracked. Real staged downloads
under that directory are ignored by Git.

Manually place candidate files in staging, then inspect them without extraction:

```powershell
python src/audit/inspect_plan_c_downloads.py --staging-dir data/external/plan_c --manifest data/artifact_manifests/plan_c_artifact_manifest.csv
```

Inspection outputs:

- `results/tables/plan_c_manual_download_inventory.csv`;
- `results/tables/plan_c_manual_download_archive_contents.csv`;
- `results/tables/plan_c_manual_download_artifact_matches.csv`;
- `results/tables/plan_c_manual_download_summary.csv`;
- `results/tables/plan_c_suggested_manifest_updates.csv`.

Rules for interpretation:

- Staged files are not automatically trusted.
- Archive inspection does not extract files and does not validate schemas.
- `official_original_candidate` rows require manual confirmation before any
  exact reproduction claim.
- `raw_dataset_candidate` rows are reconstruction inputs, not exact paper
  artifacts.
- Suggested manifest updates must be reviewed manually; the inspector never
  updates `data/artifact_manifests/plan_c_artifact_manifest.csv`.
- After a manual manifest edit, rerun the Plan C artifact validator before
  implementing converters, model runs, or metrics.

## 8.5 Plan C Raw Dataset Schema Validation

Plan C C4 validates staged raw public datasets as reconstruction inputs. It
does not create processed datasets, compute metrics, train models, or prove
paper-equivalent preprocessing.

Run:

```powershell
python src/audit/validate_plan_c_raw_datasets.py --staging-dir data/external/plan_c --sample-size 1000
```

Expected diagnostic outputs:

- `results/tables/plan_c_raw_dataset_validation_summary.csv`;
- `results/tables/plan_c_raw_dataset_schema_details.csv`;
- `results/tables/plan_c_raw_dataset_sample_stats.csv`;
- `results/tables/plan_c_reconstruction_readiness_summary.csv`.

C4 checks:

- Amazon Beauty/Sports review fields: `reviewerID`, `asin`, `overall`,
  `unixReviewTime`;
- Amazon Beauty/Sports metadata fields such as `asin`, `title`, `categories`,
  `description`, `price`, and `brand`;
- MovieLens-1M ZIP members: `ml-1m/ratings.dat`, `ml-1m/movies.dat`,
  `ml-1m/users.dat`;
- LastFM HetRec ZIP members: `user_artists.dat`, `artists.dat`, and
  `user_taggedartists.dat` when present.

Interpretation:

- Amazon Beauty, Amazon Sports, and ML-1M raw files can support controlled
  reconstruction when the C4 readiness table reports
  `reconstructible_from_public_raw`.
- LastFM may remain `extension_only` if chronological split support is only
  partial because `user_artists.dat` has weights but no listening timestamp.
- Exact Beauty/Sports reproduction remains blocked unless the author processed
  `sequential_data.txt`, `negative_samples.txt`, `datamaps.json`, and
  processed `meta.json` are recovered.

## 8.6 Plan C Raw-To-Canonical Reconstruction

Plan C C5 converts validated public raw inputs into reconstructed canonical
artifacts. These files are paper-like reconstruction artifacts, not official
original processed artifacts.

Run:

```powershell
python src/data/reconstruct_plan_c_datasets.py --config configs/plan_c_reconstruction.yaml
```

Optional dataset subset:

```powershell
python src/data/reconstruct_plan_c_datasets.py --config configs/plan_c_reconstruction.yaml --datasets amazon_beauty amazon_sports ml1m lastfm
```

Development-only diagnostic subset:

```powershell
python src/data/reconstruct_plan_c_datasets.py --config configs/plan_c_reconstruction.yaml --datasets amazon_beauty --max-users 1000
```

When `--max-users` is used, output summaries are marked
`diagnostic_subset=True` and should not be treated as final reconstruction
outputs.

Negative sampling default:

```yaml
negative_sampling:
  item_universe: interaction_items
```

This means candidate negatives are sampled from observed interaction item IDs,
excluding each user's full interaction history. This is the safer default for
leave-one-out candidate-set evaluation because metadata-only catalog items can
have zero training popularity and can inflate popularity baselines. The
alternative `catalog_items` mode is available only as an explicit experimental
setting and must be reported as such.

Development-only CSV/debug run without paper-like artifacts:

```powershell
python src/data/reconstruct_plan_c_datasets.py --config configs/plan_c_reconstruction.yaml --datasets ml1m --skip-paper-like-artifacts
```

Generated reconstructed datasets live under:

```text
data/reconstructed/plan_c/<dataset>/
```

This directory is ignored by Git except for `.gitkeep`.

Expected diagnostics:

- `results/tables/plan_c_reconstructed_dataset_summary.csv`;
- `results/tables/plan_c_reconstructed_artifact_inventory.csv`;
- `results/tables/plan_c_reconstructed_dataset_blockers.csv`;
- `results/tables/plan_c_reconstructed_dataset_stats.csv`.

The reconstruction summary and stats tables include candidate-universe
diagnostics:

- `candidate_item_universe`;
- `candidate_universe_size`;
- `catalog_size`;
- `interaction_item_count`;
- `users_with_insufficient_negatives`.

Interpretation:

- Amazon Beauty, Amazon Sports, and ML-1M can produce chronological
  leave-one-out reconstructed datasets if raw files are staged.
- LastFM remains `extension_only` unless exact timestamped interactions or the
  paper's processed protocol are recovered.
- The official Beauty/Sports Dropbox processed artifacts remain
  `blocked_unavailable_url` / `requires_author_contact` after manual testing.
- C5 does not train models, call APIs, compute metrics, or write metric CSVs.

## 8.7 Plan C Reconstructed Baseline Evaluation

Plan C C6 evaluates deterministic non-LLM baselines on reconstructed Plan C
canonical datasets. These are reconstructed-dataset baseline metrics, not exact
official Beauty/Sports paper reproduction results.

Default run:

```powershell
python src/experiments/run_plan_c_baselines.py
```

Requested validation runs:

```powershell
python src/experiments/run_plan_c_baselines.py --datasets ml1m --baselines random mostpop content_bm25
python src/experiments/run_plan_c_baselines.py --datasets amazon_beauty --baselines random mostpop --max-users 1000
```

The `--max-users` option is development-only. It marks metric, summary, and
recommendation rows with `diagnostic_subset=True` and must not be reported as a
final full-dataset result.

Expected real metric outputs:

- `results/metrics/plan_c_ml1m_random_metrics.csv`
- `results/metrics/plan_c_ml1m_mostpop_metrics.csv`
- `results/metrics/plan_c_ml1m_content_bm25_metrics.csv`
- `results/metrics/plan_c_amazon_beauty_random_metrics.csv`, when evaluated as
  a full non-diagnostic run
- `results/metrics/plan_c_amazon_beauty_mostpop_metrics.csv`, when evaluated as
  a full non-diagnostic run
- corresponding `plan_c_<dataset>_<baseline>_metrics.csv` files for any future
  full Amazon Sports or ContentBM25 runs.

Diagnostic subset metric outputs are intentionally separate:

```text
results/metrics/plan_c_<dataset>_<baseline>_diagnostic_max_users_<N>_metrics.csv
```

Full ranked diagnostic outputs are also separate and ignored by Git:

```text
results/raw_outputs/plan_c_<dataset>_<baseline>_diagnostic_max_users_<N>_recommendations.csv
```

Every C6 metric row includes:

- `diagnostic_subset`;
- `max_users`;
- `run_scope`;
- `evaluated_users`;
- `recommendation_rows`;
- `reconstruction_status`;
- `notes`.

Expected diagnostics and plots:

- `results/tables/plan_c_baseline_metrics_summary.csv`
- `results/tables/plan_c_baseline_comparison_summary.csv`
- `results/tables/plan_c_baseline_dataset_status.csv`
- `results/tables/plan_c_baseline_blockers.csv`
- `results/tables/plan_c_baseline_candidate_diagnostics.csv`
- `results/plots/plan_c_baseline_hr_at_5.png`
- `results/plots/plan_c_baseline_ndcg_at_5.png`
- `results/plots/plan_c_baseline_comparison.png`

Large full-ranked recommendation outputs are generated artifacts under
`results/raw_outputs/plan_c_*_recommendations.csv` and are ignored by Git.

Interpretation:

- Random is a deterministic sanity-check baseline.
- MostPop uses only reconstructed training/history interactions to compute item
  popularity.
- ContentBM25 builds user profiles only from reconstructed training/history
  interactions and catalog text.
- LastFM is skipped for utility metrics until valid ground truth and candidates
  exist.
- No APIs, downloads, LLM calls, or neural training are used by C6.
- Diagnostic subset rows are only runtime/stability checks. Use full
  `run_scope=full` rows for final baseline interpretation.
- Candidate diagnostics should be reviewed before interpreting popularity
  baselines. In particular, high
  `candidate_items_not_in_interaction_universe_rate` or high
  `zero_train_pop_candidate_rate` can indicate an overly easy candidate set.

## 8.8 Plan C Final Consolidation

Run:

```powershell
python src/experiments/summarize_plan_c_final_results.py
```

Expected final outputs:

- `results/tables/plan_c_final_results_overview.csv`;
- `results/tables/plan_c_claims_and_limitations.csv`;
- `results/tables/plan_c_final_experiment_inventory.csv`;
- `results/plots/plan_c_final_hr_at_5_overview.png`;
- `results/plots/plan_c_final_ndcg_at_5_overview.png`;
- `report/plan_c_final_report.md`;
- `report/oral_exam_plan_c_guide.md`.

Interpretation:

- `run_scope=full` rows are final within the reconstructed protocol.
- `run_scope=diagnostic_max_users_<N>` rows are diagnostic runtime/stability
  checks.
- Exact Beauty/Sports paper reproduction remains blocked.
- Plan C reconstructed metrics are real metrics, but not exact original paper
  metrics.

## 9. How To Complete A Fuller Reproduction

A fuller quantitative reproduction would require:

1. Obtain the deleted/unavailable preprocessed Beauty and Sports artifacts from
   the authors, or reconstruct equivalent artifacts from valid raw data in a
   separate documented phase.
2. Produce canonical project files for interactions, ground truth, candidates,
   and catalog metadata under `data/processed/`.
3. Configure `interactions_path`, `ground_truth_path`, `candidates_path`, and
   `catalog_path` in the dataset YAML files.
4. Rerun LLM output loading, baseline output loading, reproduction metrics,
   hallucination, coverage, position-bias, history-length, pipeline, and plot
   commands.
5. Optionally rerun traditional baselines or LLM calls only with explicit user
   approval, documented costs, and `--allow-api-calls` for any API step.

## 10. Final Statement

This project does not fabricate missing data or numerical results. Missing
artifacts and blocked metrics are explicitly documented. In its current state,
the repository is suitable as a controlled reproducibility engineering study:
it demonstrates artifact recovery, schema design, safe normalization, tested
metric implementations, readiness-aware orchestration, and honest reporting of
why full Beauty/Sports quantitative reproduction is not currently possible.
The separate ML-1M track demonstrates a real reconstruction pipeline, but those
values are not exact reproduction results for the original Beauty/Sports
experiments.
