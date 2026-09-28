# External Archive Guide

Normal collaborator use should not require raw bulk artifacts outside Git. The committed foreground CSVs, manifests, reports, and deterministic packages are sufficient to inspect the final scientific conclusions.

## Policy

Keep these outside normal Git:

- raw G1 LLM responses
- raw G2 LLM responses
- raw G3 LLM responses
- raw G4 LLM responses
- `archive_external/*.zip`
- raw datasets where licensing or size requires local handling
- model weights and checkpoints
- local environments, caches, logs, and temporary files

Do not invent download URLs. If external evidence must be shared, use the existing archive path, size, SHA-256, and manifest lineage.

## Committed Foreground Packages

Large foreground CSV evidence that is useful for no-inference inspection is committed as deterministic gzip packages rather than normal-Git CSV blobs. The manifest is:

The current release inventory contains 51 packages.

```text
results/manifests/github_packaged_artifacts.json
```

Verify packages:

```powershell
.\.venv_audit_recovery\Scripts\python.exe scripts\restore_packaged_artifacts.py --verify-package
```

Restore and verify ignored CSV originals:

```powershell
.\.venv_audit_recovery\Scripts\python.exe scripts\restore_packaged_artifacts.py --restore
.\.venv_audit_recovery\Scripts\python.exe scripts\restore_packaged_artifacts.py --verify-restored
```

The restore script verifies package SHA-256 before extraction, refuses unsafe overwrites, and checks restored SHA-256 plus row counts. It performs no model calls, network calls, or metric computation.

## Externalized Raw LLM Evidence

The raw response trees are forensic evidence and remain local/external:

- `results/llm/extension_canonical/g1_primary_ranking/**`
- `results/llm/extension_canonical/g2_history_length/**`
- `results/llm/extension_canonical/g3_candidate_position/**`
- `results/llm/extension_canonical/g4_f1bzq_reranking/**`

Use committed inventories, manifests, and package-restored foreground CSVs before requesting raw response files.

## External Archive Bundles

Known local archive bundle paths:

- `archive_external/g1_primary_ranking/G1_primary_ranking_3d7f3add_external_evidence.zip`
- `archive_external/g2_history_length_execution/G2_history_length_8fb6dc9_external_evidence.zip`
- `archive_external/g3_candidate_position_execution/G3_candidate_position_06088e9_external_evidence.zip`
- `archive_external/g4_f1bzq_reranking_execution/G4_f1bzq_reranking_df8700c_external_evidence.zip`

These ZIP files are not normal Git artifacts. Verify against the relevant retention manifests before use.

## G1 Primary Ranking

G1 now has a committed foreground package for the large per-request CSV evidence:

- `results/tables/g1_primary_ranking_per_request_stats.csv.gz`
- `results/tables/g1_primary_ranking_strict_parsing.csv.gz`
- `results/tables/g1_primary_ranking_recommendations.csv.gz`
- `results/tables/g1_primary_ranking_request_universe.csv.gz`
- `results/tables/g1_primary_ranking_raw_response_inventory.csv.gz`
- `results/tables/g1_primary_ranking_execution_state.csv.gz`
- `results/tables/g1_primary_ranking_physical_execution_order.csv.gz`

The raw JSON response directory remains external. The package and committed manifests are sufficient for no-inference foreground inspection.

## G4 / F1BZQ

The final machine-readable authority and portability contracts are:

- `results/manifests/g4_f1bzq_canonical_provenance.json`
- `results/manifests/g4_f1bzq_reproducibility_contract.json`

`results/tables/g4_f1bzq_statistical_paired_contributions.csv.gz` is the
canonical byte-preserving delivery form of the logical paired-contributions
CSV. The canonical provenance manifest records both compressed and
decompressed SHA-256 values.

The G4 request, response, and event ledgers are classified as
`EXTERNAL_FROZEN_EXECUTION_EVIDENCE`. They are not Git dependencies for frozen
metric or statistical reproduction. Their external archive is identified by a
repository-relative logical path and expected SHA-256 in the canonical
provenance manifest.

## Verification

Use inventory rows and phase manifests to recover original paths and hashes. Do not delete unique scientific evidence unless a verified archive or external copy exists.
