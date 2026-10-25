# Data Availability and External Materials

This document distinguishes materials committed in this repository from
external inputs required for fresh execution. It does not assert redistribution
rights that are not documented in the repository.

## Materials included in the repository

- Frozen metric tables, manifests, reports, final figures, and deterministic
  gzip packages used for no-inference validation.
- Processed ML-1M files currently tracked under `data/processed/ml1m/`.
- Dataset-specific feasibility and provenance tables, including Food.com hash
  manifests under `results/manifests/` and `results/tables/`.
- The upstream code reference as a Git submodule at
  `original_repo/EvaLLMasRecommender/`, fixed to gitlink
  `1e12982ca5c32d271cec6d1cb723e071f8c6587c`.

The 51 package inventory in `results/manifests/github_packaged_artifacts.json`
is sufficient to restore the committed foreground CSV evidence required by the
frozen-result checks.

## Dataset provenance

| Dataset | Provenance recorded in this repository | Distribution status | Terms status |
|---|---|---|---|
| Amazon Beauty | Upstream README identifies the Amazon 2014 collection and its official UCSD source. | Raw data are not part of the curated release. | NOT VERIFIED here; consult the source terms before redistribution. |
| Amazon Sports | Upstream README identifies the Amazon 2014 collection and its official UCSD source. | Raw data are not part of the curated release. | NOT VERIFIED here; consult the source terms before redistribution. |
| MovieLens-1M | Upstream README identifies the GroupLens MovieLens-1M dataset. | Selected processed files are tracked; raw source material is not distributed as a release dependency. | NOT VERIFIED here; consult GroupLens terms before reuse or redistribution. |
| LastFM | Upstream README identifies the HetRec 2011/LastFM source through GroupLens. | Raw and reconstructed inputs are not part of the curated release. | NOT VERIFIED here; consult the source terms before reuse or redistribution. |
| Food.com | The repository records the raw archive filename, SHA-256, schema audit, and feasibility evidence in `results/manifests/d0a_foodcom_raw_archive_manifest.json`. | The raw archive and extracted raw CSVs are not part of the curated release. | The original download source and license are not sufficiently documented here; this must be resolved before public redistribution. |

The upstream README provides dataset landing pages for Amazon, MovieLens, and
LastFM. This repository preserves those provenance references but does not
replace the original dataset terms.

## Code provenance

The original framework is represented by the `EvaLLMasRecommender` submodule,
whose configured source is `https://github.com/JiangDeccc/EvaLLMasRecommender.git`.
The checked upstream snapshot contains no LICENSE, COPYING, or NOTICE file
that is verifiable from this repository. Its reuse terms therefore require
confirmation from the upstream project before assigning a license to this
submission or redistributing derived code more broadly.

The main repository also has no LICENSE file. No project-wide open-source
license can be inferred from the presence of public dependencies or an
upstream GitHub URL.

## Not distributed

The following remain local or external by design:

- raw datasets and reconstructed intermediate datasets;
- raw LLM response archives and external archive ZIPs;
- model weights, checkpoints, predictions, caches, local logs, and Python
  environments;
- resources needed only to rerun historical LLM inference or train models.

See `docs/external_archive_guide.md` for the external-evidence policy.

## Fresh execution

Fresh inference is outside the frozen-result validation scope. It may require
dataset access under the source terms, model weights and runtime availability,
hardware, and a local or external inference backend. The repository does not
claim that a fresh LLM run will reproduce historical outputs byte-for-byte.

## License review status

The release-validation dependencies are named and version-pinned in
`requirements-release-validation.txt`, but this repository does not include a
third-party license inventory. Before public release, confirm the licenses and
notice obligations for the upstream submodule, each distributed dependency,
and each dataset; only then choose an appropriate license for the project’s
original contributions.
