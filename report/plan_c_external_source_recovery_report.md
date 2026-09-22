# Plan C External Source Recovery Report

## 1. Scope

This report records lightweight external source checking for Plan C. The goal is
to identify which sources appear reachable, which sources are only raw datasets,
which sources are supposed to contain processed paper artifacts, and which
blockers require author contact or reconstruction.

No datasets were downloaded. No remote content was saved. No APIs were called.
No models were trained. No metrics were computed. Files under `original_repo/`
were not modified.

The URL checker output is:

- `results/tables/plan_c_source_url_check_results.csv`

The recovery ledger is:

- `results/tables/plan_c_external_source_recovery_ledger.csv`

## 2. URL Check Summary

The checker read non-empty `source_url` values from
`data/artifact_manifests/plan_c_artifact_manifest.csv`, deduplicated them, and
performed metadata-only HTTP checks with a timeout.

Observed C2 status:

| Status | Count |
|---|---:|
| reachable | 5 |
| network_unverified | 1 |

Six unique URLs were checked:

| Source | Network status | Availability status | Notes |
|---|---|---|---|
| Official GitHub repository | checked | reachable | HTTP 200 HTML page. Useful for source-code provenance, not enough for metrics. |
| Beauty Dropbox processed ZIP URL | checked | reachable HTML artifact unverified | HTTP 200 returned HTML metadata. This does not prove the ZIP is downloadable and does not overturn the Plan A blocker. |
| Sports Dropbox processed ZIP URL | checked | reachable HTML artifact unverified | HTTP 200 returned HTML metadata. This does not prove the ZIP is downloadable and does not overturn the Plan A blocker. |
| Amazon 2014 raw data source | network unverified | network unverified | The C2 metadata request timed out. Availability is not inferred. |
| MovieLens-1M raw source | checked | reachable | HTTP 200 HTML page. Raw source is available as a manual source, not paper-equivalent by itself. |
| LastFM HetRec raw source | checked | reachable | HTTP 200 HTML page. Raw source is available as a manual source, not paper-equivalent by itself. |

## 3. Official Sources

The official upstream repository source exists:

- `https://github.com/JiangDeccc/EvaLLMasRecommender`

The local checkout under `original_repo/EvaLLMasRecommender/` contains:

- the original README;
- prompt templates;
- evaluation scripts;
- API wrapper scripts;
- Beauty and Sports saved LLM output ZIPs;
- Beauty SOTA output ZIP.

These artifacts are valuable provenance material, but they do not provide the
missing Beauty/Sports processed data files required for exact metrics.

## 4. Processed Paper Artifact Sources

The original README points to Dropbox URLs for processed Beauty and Sports
data. Those URLs are the only official processed artifact links found locally
for:

- `sequential_data.txt`
- `negative_samples.txt`
- `datamaps.json`
- `meta.json`

The C2 metadata check reached Dropbox HTML pages for both URLs. This is not
proof that the ZIP files are recoverable. It only means the web page responded.
Plan A documentation already records that manual browser/curl checks found the
Dropbox artifacts unavailable/deleted. Until a real ZIP is manually recovered
and validated, Beauty and Sports exact numerical reproduction remains blocked.

Recommended action:

1. The user may manually open the Dropbox links in a browser.
2. If a valid ZIP can be recovered, place it outside `original_repo/`, for
   example under `data/raw/plan_c/beauty/` or `data/raw/plan_c/sports/`.
3. Update `data/artifact_manifests/plan_c_artifact_manifest.csv` with local
   paths, expected sizes, and hashes if available.
4. Run:

```powershell
python src/audit/validate_plan_c_artifacts.py --manifest data/artifact_manifests/plan_c_artifact_manifest.csv
```

## 5. Raw Dataset Sources

The following URLs are raw dataset sources, not processed paper artifacts:

- Amazon Beauty/Sports 2014 raw data:
  `https://jmcauley.ucsd.edu/data/amazon/index_2014.html`
- MovieLens-1M:
  `https://grouplens.org/datasets/movielens/1m/`
- LastFM HetRec 2011:
  `https://grouplens.org/datasets/hetrec-2011/`

Raw public data does not automatically imply exact paper reproduction. The
paper's results depend on preprocessing choices and artifacts that raw data
does not provide directly:

- 5-core filtering details;
- chronological ordering;
- leave-one-out split construction;
- sampled user selection and K-S test policy;
- negative sampling seeds;
- candidate order and positive-item placement;
- item ID maps;
- title/category metadata normalization;
- traditional model training configs and seeds.

Therefore, raw data sources are reconstruction inputs unless the exact
author-preprocessed files or exact preprocessing protocol can be validated.

## 6. Manual Downloads Recommended

Manual download is recommended only when the user is ready to validate
artifacts and accept the relevant data terms.

Highest-priority exact reproduction attempt:

- Beauty processed ZIP from the official README Dropbox link;
- Sports processed ZIP from the official README Dropbox link.

Reconstruction inputs if exact processed files remain unavailable:

- Amazon 2014 Beauty/Sports raw data;
- MovieLens-1M raw files, already present locally for Plan B but not
  paper-equivalent by default;
- LastFM HetRec raw files.

No automatic download should be added to this project.

## 7. Validation After Manual Download

After manual placement, validate local paths and optional hashes:

```powershell
python src/audit/validate_plan_c_artifacts.py --manifest data/artifact_manifests/plan_c_artifact_manifest.csv
python src/audit/validate_plan_c_artifacts.py --manifest data/artifact_manifests/plan_c_artifact_manifest.csv --hash
```

Then a later Plan C phase should perform schema validation. Passing URL or file
existence checks is not enough to compute metrics.

## 8. Author Contact Blockers

The following blockers likely require author contact:

- renewed Beauty processed ZIP link or the processed files directly;
- renewed Sports processed ZIP link or the processed files directly;
- exact preprocessing scripts, random seeds, and sampled user lists;
- MovieLens-1M and LastFM paper-equivalent processed artifacts, if released;
- missing Sports SOTA outputs;
- SOTA outputs for MovieLens-1M and LastFM, if the paper expects them;
- clarification on whether the Dropbox links moved.

The draft message is saved at:

- `report/plan_c_author_contact_request.md`

## 9. Reconstruction-Ready Components

Components suitable for reconstruction rather than exact reproduction:

- MovieLens-1M: Plan B already demonstrates an extension pipeline, but a
  paper-equivalent Plan C protocol would need separate preprocessing and
  candidate-set validation.
- LastFM: raw source appears reachable, but no raw files are local and no
  paper-equivalent preprocessing exists yet.
- Amazon Beauty/Sports: raw reconstruction may be possible, but exact
  equivalence is uncertain without author preprocessing details.

Reconstruction metrics must remain clearly labeled as reconstruction or
extension metrics. They must not be merged with exact paper reproduction claims.

## 10. C4 Raw Dataset Validation Update

Plan C C4 validated the manually staged raw public files under
`data/external/plan_c/` using sampled schema checks only. The validator read
gzip and ZIP contents in place and did not extract archives permanently.

Validated raw inputs:

- `reviews_Beauty_5.json.gz`
- `meta_Beauty.json.gz`
- `reviews_Sports_and_Outdoors_5.json.gz`
- `meta_Sports_and_Outdoors.json.gz`
- `ml-1m.zip`
- `hetrec2011-lastfm-2k.zip`

Generated diagnostics:

- `results/tables/plan_c_raw_dataset_validation_summary.csv`
- `results/tables/plan_c_raw_dataset_schema_details.csv`
- `results/tables/plan_c_raw_dataset_sample_stats.csv`
- `results/tables/plan_c_reconstruction_readiness_summary.csv`

C4 readiness summary:

| Dataset | Raw reconstruction input | Sequential interactions | Item metadata | Chronological split | Status |
|---|---|---|---|---|---|
| Amazon Beauty | yes | yes | yes | yes | `reconstructible_from_public_raw` |
| Amazon Sports | yes | yes | yes | yes | `reconstructible_from_public_raw` |
| MovieLens-1M | yes | yes | yes | yes | `reconstructible_from_public_raw` |
| LastFM HetRec | yes | yes | yes | partial | `extension_only` |

This update does not recover the official Beauty/Sports processed artifacts.
`sequential_data.txt`, `negative_samples.txt`, `datamaps.json`, and the paper
processed `meta.json` remain missing for exact Beauty/Sports reproduction.

## 11. C5 Reconstruction Converter Update

The user manually confirmed that the official Beauty and Sports Dropbox links
do not provide recoverable processed downloads in the current environment.
Those official processed artifacts remain `blocked_unavailable_url` and require
author contact unless author-provided files are staged later.

Plan C C5 adds a separate reconstruction converter:

```powershell
python src/data/reconstruct_plan_c_datasets.py --config configs/plan_c_reconstruction.yaml
```

The converter writes reconstructed, paper-like canonical artifacts under:

```text
data/reconstructed/plan_c/<dataset>/
```

Those files are generated from public raw inputs and are explicitly marked as
reconstructed. They are not the official original processed artifacts and must
not be used to claim exact Beauty/Sports reproduction.

C5 reconstruction scope:

- Amazon Beauty: chronological leave-one-out reconstruction from raw Amazon
  reviews and metadata.
- Amazon Sports: chronological leave-one-out reconstruction from raw Amazon
  reviews and metadata.
- MovieLens-1M: chronological leave-one-out reconstruction written separately
  from Plan B outputs.
- LastFM: interaction/catalog reconstruction only, marked `extension_only`
  because `user_artists.dat` lacks listening timestamps.

Diagnostic outputs:

- `results/tables/plan_c_reconstructed_dataset_summary.csv`
- `results/tables/plan_c_reconstructed_artifact_inventory.csv`
- `results/tables/plan_c_reconstructed_dataset_blockers.csv`
- `results/tables/plan_c_reconstructed_dataset_stats.csv`

## 12. C6 Reconstructed Baseline Evaluation Update

Plan C C6 evaluates deterministic baselines on reconstructed canonical
datasets created by C5:

```powershell
python src/experiments/run_plan_c_baselines.py
```

The supported C6 baselines are:

- Random;
- MostPop;
- ContentBM25.

The supported utility metrics are HR@5 and NDCG@5. Catalog Coverage@5 is also
written when a reconstructed catalog is available.

C6 output locations:

- real metrics: `results/metrics/plan_c_<dataset>_<baseline>_metrics.csv`;
- diagnostics: `results/tables/plan_c_baseline_*.csv`;
- plots: `results/plots/plan_c_baseline*.png`;
- full ranked outputs:
  `results/raw_outputs/plan_c_<dataset>_<baseline>_recommendations.csv`.

These outputs are reconstructed-dataset baseline results. They do not recover
or replace the unavailable official Beauty/Sports processed artifacts. Exact
Beauty/Sports reproduction remains blocked by `blocked_unavailable_url` /
`requires_author_contact`.

LastFM remains excluded from utility metric evaluation because C5 marked it
`extension_only` and did not create valid ground truth or candidate sets.
