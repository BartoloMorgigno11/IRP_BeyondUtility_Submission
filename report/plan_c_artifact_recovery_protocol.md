# Plan C Artifact Recovery And Validation Protocol

## 1. Purpose

Plan C aims to maximize fidelity to *Beyond Utility: Evaluating LLM as
Recommender*. The first requirement is not new model code or new metrics. It is
artifact control.

The original paper's metrics depend on exact histories, held-out targets,
negative samples, candidate ordering, item-title mappings, categories, saved
LLM outputs, traditional baseline outputs, and model/version choices. If any of
those inputs are missing or reconstructed differently, later numbers may be
useful, but they are not exact reproduction evidence.

This protocol defines how artifacts must be acquired, classified, validated,
and documented before Plan C converters, model runs, or metric computation are
implemented.

Plan A and Plan B remain unchanged:

- Plan A is a closed readiness-aware audit of the original Beauty/Sports
  artifacts.
- Plan B is a completed MovieLens-1M reconstruction/extension track.
- Plan C is a separate full-paper replication campaign.

## 2. Artifact Classification

Every artifact in the Plan C manifest must have one `source_type`:

| Source type | Meaning |
|---|---|
| `official_original` | File shipped in the original repository checkout or paper bundle. |
| `official_dataset_raw` | Raw dataset from the official dataset host, manually placed locally. |
| `author_released_processed` | Processed data or outputs released by the paper authors. |
| `author_released_output` | Saved model outputs released by the paper authors. |
| `third_party_mirror` | Copy hosted outside the original authors or official dataset source. |
| `reconstructed_from_public_raw` | Artifact generated locally from public raw data using a documented reconstruction protocol. |
| `local_extension` | Artifact created by this project as Plan B or another extension. |
| `unavailable` | Artifact source is known but unavailable. |
| `unknown` | Artifact is required or suspected, but the source/protocol is not yet known. |

Every artifact must also have one `reproducibility_status`:

| Status | Meaning |
|---|---|
| `exact_ready` | Local artifact has passed validation and is sufficient for exact reproduction of its component. |
| `local_artifact_available` | Artifact exists locally but may still need downstream data or protocol validation. |
| `requires_manual_download` | User must manually place the artifact; Codex must not download it. |
| `reconstructible_from_public_raw` | A reconstruction is feasible from public raw data, but exact paper equivalence is not guaranteed. |
| `requires_author_contact` | Artifact likely requires author-provided files or clarification. |
| `requires_api` | Reproducing the artifact requires an LLM/API call with explicit approval. |
| `requires_training` | Reproducing the artifact requires model training or rerunning traditional recommenders. |
| `blocked_missing_artifact` | Artifact is required but not available locally. |
| `blocked_unavailable_url` | A known source URL is unavailable or deleted. |
| `blocked_license_or_terms` | The artifact cannot be used until licensing or terms are reviewed. |
| `blocked_unclear_protocol` | Exact file/protocol is not yet identifiable. |
| `extension_only` | Artifact belongs to a reconstruction/extension track, not exact paper reproduction. |

## 3. Dataset Requirements

### Amazon All Beauty

Required exact-reproduction artifacts:

- `sequential_data.txt`
- `negative_samples.txt`
- `datamaps.json`
- `meta.json`
- sampled user lists such as `sample.txt` and `sample_1k.txt`
- saved LLM outputs for each prompt/model condition
- traditional baseline outputs for BM25, MostPop, BPRMF, GRU4Rec, SASRec, and
  LightGCN where the paper reports them

The current checkout contains Beauty LLM outputs and a Beauty SOTA archive, but
the core processed data files are missing. Exact Beauty metrics remain blocked
until those files are recovered and validated.

### Sports & Outdoors

Required exact-reproduction artifacts mirror Beauty:

- `sequential_data.txt`
- `negative_samples.txt`
- `datamaps.json`
- `meta.json`
- sampled user lists
- saved LLM outputs
- traditional baseline outputs

The current checkout contains Sports LLM outputs but not the core processed
data files or a Sports SOTA archive. Exact Sports metrics remain blocked.

### MovieLens-1M

The paper includes ML-1M as an original dataset. Plan B already created a
MovieLens-1M reconstruction from manually supplied local raw files, but Plan B
is not automatically paper-equivalent.

For Plan C, ML-1M becomes exact only if the paper-equivalent preprocessing is
validated:

- raw `ratings.dat`, `movies.dat`, and optionally `users.dat`;
- paper-equivalent 5-core filtering;
- leave-one-out split details;
- sampled users;
- candidate sets and candidate order;
- saved LLM and traditional baseline outputs or reproducible rerun protocol.

Until those are fixed, Plan B ML-1M files must be classified as
`local_extension` and `extension_only`.

### LastFM

The original README references the HetRec 2011 LastFM source. Exact Plan C
LastFM replication needs the raw interaction and item metadata files, then a
validated paper-equivalent preprocessing protocol. The current checkout does
not contain LastFM raw or processed files, so LastFM is not ready.

## 4. Task Requirements

### Ranking

Exact ranking reproduction needs:

- chronological user histories;
- held-out positive test item per sampled user;
- negative samples drawn from items not in the user history;
- candidate list order for each prompt condition;
- item titles and item IDs mapped consistently;
- parsed model outputs;
- metric code using the paper's target/candidate definitions.

The paper's ranking setting uses one positive item plus randomly retrieved
negative items to simulate full-catalog ranking under LLM input-length limits.

### Re-Ranking

Exact re-ranking reproduction needs all ranking inputs plus traditional recall
model outputs. The paper states that re-ranking candidate pools are formed from
top-k outputs of BPRMF, GRU4Rec, SASRec, and LightGCN. If those outputs are not
available, reproducing re-ranking requires either author-provided recall files
or a later, explicit training phase.

## 5. Exact Reproduction Versus Reconstruction

An artifact supports exact reproduction only when:

1. the source is official or author-released;
2. the local file exists;
3. the expected size/hash matches, when known;
4. the schema is compatible with the paper's code/protocol;
5. the artifact maps to the same dataset, split, sample, candidate policy, and
   model/version condition.

An artifact is reconstruction when it is generated from public raw data after
the original processed artifact is unavailable. Reconstruction is valid only if
the process is deterministic and documented, but it should be labeled
`reconstructed_from_public_raw` or `extension_only` unless equivalence is
proven.

A component becomes `extension_only` when:

- it uses a different dataset protocol than the paper;
- it evaluates a substitute model;
- it uses different candidate sets, splits, sample users, or item mappings;
- it answers a project extension question rather than verifying a paper result.

## 6. Recording Source URLs

Source URLs should be recorded in
`data/artifact_manifests/plan_c_artifact_manifest.csv`, but Codex must not
download from them automatically.

Known local-source URLs from the upstream README and project reports are:

- Beauty processed ZIP Dropbox URL;
- Sports processed ZIP Dropbox URL;
- Amazon Beauty/Sports 2014 raw data URL;
- MovieLens-1M raw data URL;
- LastFM HetRec raw data URL.

If a URL is found in a local file, record it exactly as found. Do not invent
replacement URLs. If a link is unavailable, mark the artifact as
`blocked_unavailable_url` and document the blocker.

## 7. Manual Acquisition Procedure

When the user manually obtains an artifact:

1. Place it outside `original_repo/`; do not patch the upstream checkout.
2. Prefer `data/raw/plan_c/<dataset>/...` for author-released processed files
   and raw archives that should stay out of Git.
3. Update the manifest row's `local_path`, `found_locally`, `source_type`,
   `license_or_terms_notes`, and `last_checked`.
4. If the provider publishes file sizes or hashes, add them to
   `expected_size_bytes` and `expected_hash_sha256`.
5. Run the validator:

```powershell
python src/audit/validate_plan_c_artifacts.py --manifest data/artifact_manifests/plan_c_artifact_manifest.csv
```

6. If hashes are present and should be checked, run:

```powershell
python src/audit/validate_plan_c_artifacts.py --manifest data/artifact_manifests/plan_c_artifact_manifest.csv --hash
```

7. Store validator outputs under `results/tables/`; do not store validation
   diagnostics under `results/metrics/`.

## 8. Validation Rules

The validator checks local paths only. It must not download, modify raw files,
modify `original_repo/`, call APIs, train models, or compute metrics.

Validation should confirm:

- local path exists or is clearly missing;
- file size matches if `expected_size_bytes` is provided;
- SHA-256 matches if `expected_hash_sha256` is provided and `--hash` is used;
- empty expected size/hash fields do not cause false failures;
- missing `local_path` is reported as `missing_local_path`;
- directories are treated as found but not hashed.

Passing path validation is not the same as metric readiness. Schema validation,
converter validation, candidate-set validation, and metric validation are later
Plan C phases.

## 9. Handling Unavailable Links

If Dropbox, Google Drive, or another source link is unavailable:

1. Record the original URL in the manifest.
2. Set `reproducibility_status` to `blocked_unavailable_url` if the URL itself
   is known to be unavailable.
3. Set `blocker_reason` to the concrete observed blocker, such as deleted file,
   permission denied, unavailable share, or unknown.
4. Do not replace the URL with an unverified mirror.
5. If a third-party mirror is later used, classify it as `third_party_mirror`
   and document why it is or is not equivalent to the author release.

## 10. When To Reconstruct From Public Raw Data

Reconstruction is acceptable only after the exact artifact is unavailable or
not obtainable and the raw source is legitimate. A reconstruction phase must
document:

- raw data source and license/terms;
- filtering policy, including 5-core details;
- user/item ID mapping;
- chronological ordering and split policy;
- negative sampling policy and random seeds;
- sampled-user policy and any K-S test requirement;
- candidate ordering;
- output file schemas;
- differences from the paper.

Until this is done, reconstructed artifacts cannot support exact reproduction
claims.

## 11. Current C1 Decision

The current C1 phase creates the protocol, manifest, and validator only. It
does not recover missing Beauty/Sports data, does not convert datasets, does
not compute metrics, and does not rerun models.

The next safe Plan C phase is to validate any manually supplied artifacts. If
no new artifacts are supplied, the correct status remains blocked for exact
Beauty/Sports, paper-equivalent ML-1M, and LastFM replication.
