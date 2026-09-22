# Plan C Manual Artifact Intake Guide

## 1. Purpose

Plan C may require official processed archives, official raw datasets, or
author-sent files that are not available in the repository. Codex must not
download those files automatically. The user may manually place candidate
artifacts in:

```text
data/external/plan_c/
```

This staging area is for intake only. A file in staging is not automatically
trusted, validated, exact-ready, or suitable for metrics. It must be inspected,
classified, recorded in the artifact manifest, and validated before any later
converter or evaluation step.

## 2. Files The User May Place In Staging

Examples of acceptable manually supplied files:

- Beauty official processed archive, if the original Dropbox artifact can be
  recovered.
- Sports official processed archive, if the original Dropbox artifact can be
  recovered.
- Amazon 2014 Beauty/Sports raw data archives, if a later reconstruction phase
  is approved.
- LastFM HetRec 2011 archive, if a later reconstruction phase is approved.
- MovieLens-1M archive, if a separate Plan C copy is needed later.
- Author-sent artifacts such as processed files, candidate sets, datamaps,
  metadata, sampled user lists, baseline outputs, or checksums.

Do not place staged files under `original_repo/`. Do not commit downloaded
archives, extracted datasets, or large generated outputs.

## 3. Expected Artifact Examples

Possible exact-paper artifact candidates:

- `beauty.zip` containing `sequential_data.txt`, `negative_samples.txt`,
  `datamaps.json`, and `meta.json`.
- `sports.zip` containing `sequential_data.txt`, `negative_samples.txt`,
  `datamaps.json`, and `meta.json`.
- Author-sent files with checksums or explicit confirmation that they match the
  paper artifacts.

Possible reconstruction inputs:

- Amazon raw Beauty/Sports review and metadata archives.
- MovieLens-1M raw files such as `ratings.dat`, `movies.dat`, and `users.dat`.
- LastFM HetRec files such as `user_artists.dat` and `artists.dat`.

Raw public data does not recover the paper's processed artifacts by itself. It
is only a reconstruction input unless the exact preprocessing protocol,
candidate sets, mappings, sampled users, and seeds are recovered.

## 4. Intake Command

Inspect staged files without extracting archives:

```powershell
python src/audit/inspect_plan_c_downloads.py --staging-dir data/external/plan_c --manifest data/artifact_manifests/plan_c_artifact_manifest.csv
```

The inspector supports regular files, ZIP archives, and TAR archives supported
by Python's standard `tarfile` module. It lists archive members but does not
extract them.

## 5. Inspection Outputs

The command writes diagnostics under `results/tables/`:

- `plan_c_manual_download_inventory.csv`;
- `plan_c_manual_download_archive_contents.csv`;
- `plan_c_manual_download_artifact_matches.csv`;
- `plan_c_manual_download_summary.csv`;
- `plan_c_suggested_manifest_updates.csv`.

If staging is empty, the summary status is `no_staged_files` and no fake
artifact matches are created.

## 6. How To Interpret Results

`official_original_candidate` means expected processed filenames were found,
but exact reproduction still requires manual confirmation of source, file
structure, schema, checksums, and protocol.

`raw_dataset_candidate` means the file may support a later reconstruction
phase. It does not prove exact equivalence with the paper.

`baseline_output_candidate` and `llm_output_candidate` mean the file appears to
contain output-like artifacts. These still need mapping, prompt, candidate-set,
and evaluation-protocol validation before metrics can be computed.

`unknown` means the file was inventoried, but no known Plan C filename pattern
was detected.

## 7. Manifest Updates

The inspector does not update
`data/artifact_manifests/plan_c_artifact_manifest.csv` automatically. Instead,
it writes:

```text
results/tables/plan_c_suggested_manifest_updates.csv
```

Every suggested row has `requires_manual_confirmation=True`. The user or
researcher must review the source, license or terms, filename, expected size,
optional SHA-256 hash, and reproducibility status before editing the manifest.
No suggestion should be promoted directly to `exact_ready`.

## 8. Validation After Manual Manifest Update

After manually editing the manifest, run:

```powershell
python src/audit/validate_plan_c_artifacts.py --manifest data/artifact_manifests/plan_c_artifact_manifest.csv
```

If expected SHA-256 hashes are available:

```powershell
python src/audit/validate_plan_c_artifacts.py --manifest data/artifact_manifests/plan_c_artifact_manifest.csv --hash
```

Only after manifest validation and schema/protocol checks should a later Plan C
phase consider implementing converters or computing real metrics.

## 9. Raw Dataset Schema Validation

After raw public datasets are staged, run the C4 raw validation audit:

```powershell
python src/audit/validate_plan_c_raw_datasets.py --staging-dir data/external/plan_c --sample-size 1000
```

This command samples raw files and ZIP members without moving or permanently
extracting them. It writes:

- `results/tables/plan_c_raw_dataset_validation_summary.csv`;
- `results/tables/plan_c_raw_dataset_schema_details.csv`;
- `results/tables/plan_c_raw_dataset_sample_stats.csv`;
- `results/tables/plan_c_reconstruction_readiness_summary.csv`.

C4 validates whether raw inputs can support reconstruction. It does not create
canonical processed datasets and does not prove exact paper equivalence.

## 10. Raw-To-Canonical Reconstruction

After C4 confirms raw schema readiness, Plan C C5 can build reconstructed
canonical artifacts:

```powershell
python src/data/reconstruct_plan_c_datasets.py --config configs/plan_c_reconstruction.yaml
```

For development only, restrict a heavy dataset to a deterministic eligible-user
subset:

```powershell
python src/data/reconstruct_plan_c_datasets.py --config configs/plan_c_reconstruction.yaml --datasets amazon_beauty --max-users 1000
```

Outputs created with `--max-users` are marked `diagnostic_subset=True` and
must not be treated as final reconstruction outputs.

For debugging canonical CSV generation, paper-like files can be skipped:

```powershell
python src/data/reconstruct_plan_c_datasets.py --config configs/plan_c_reconstruction.yaml --datasets ml1m --skip-paper-like-artifacts
```

Outputs are written under:

```text
data/reconstructed/plan_c/<dataset>/
```

Expected files where reconstruction is safe:

- `interactions.csv`
- `ground_truth.csv`
- `candidates.csv`
- `catalog.csv`
- `datamaps.json`
- `meta.json`
- `sequential_data.txt`
- `negative_samples.txt`
- `README_RECONSTRUCTED.md`

These files are reconstructed artifacts from public raw inputs. They are not
the original paper's official preprocessed artifacts. LastFM may produce only
interaction/catalog artifacts when chronological leave-one-out is not supported
by the raw interaction file.

## 11. Safeguards

- Do not download files automatically.
- Do not extract archives by default.
- Do not modify `original_repo/`.
- Do not train models.
- Do not call APIs.
- Do not create fake data or fake metrics.
- Keep Plan A readiness results and Plan B reconstruction metrics unchanged.
- Keep staged real downloads ignored by Git.
