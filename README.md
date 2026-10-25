# Beyond Utility: Reproduction and Extension Package

## Overview

This repository contains the final reproducibility package for a local reproduction and extension study of:

**Beyond Utility: Evaluating LLM as Recommender**
WWW 2025 â€” arXiv `2411.00331`

The project reproduces and extends selected experimental questions from the original study under a controlled local protocol.

It does **not** claim paper-exact numerical replication.

Some elements of the original experimental setting were unavailable or could not be reproduced identically, including original candidate files and samples, exact model identities, and some protocol details. For this reason, the repository explicitly distinguishes evidence that is:

- exact;
- repository-derived;
- reconstructed;
- adapted;
- operationalized;
- diagnostic;
- blocked;
- extension evidence.

The original upstream implementation is preserved unchanged as a Git submodule under:

`original_repo/EvaLLMasRecommender/`

---

## Final Experimental State

The final frozen package contains:

- the completed local reproduction core;
- G1 primary-ranking experiments;
- G2 history-length sensitivity experiments;
- RQ3 traditional history-length scientific freeze;
- G3 candidate-position sensitivity experiments;
- G4 candidate-aware reranking experiments;
- utility and beyond-utility metrics;
- structural-validity evidence;
- statistical summaries and paired bootstrap analyses;
- final scientific figures;
- machine-readable tables;
- provenance and freeze manifests;

Important interpretation boundaries are preserved throughout the repository.

In particular:

- G1 versus G4 comparisons are descriptive because their candidate-generation conditions differ;
- G4 reranking performance is bounded by candidate recall;
- longer user histories do not produce a universally monotonic improvement;
- candidate ordering can materially affect local LLM recommender behavior;
- structural output validity and textual hallucination are treated as distinct phenomena;
- LLMSRec was not executed as a faithful paper-exact reproduction;
- fresh LLM execution is not claimed to be bitwise identical to the historical runs.

---

## Canonical Scientific Entry Points

The principal scientific documents are:

- Final scientific report:`report/final_scientific_report.md`
- Reproduction closure:`report/phase_f6m_reproduction_closure.md`
- G1 report:`report/g1_primary_ranking_execution.md`
- G2 synthesis:`report/g2_history_length_scientific_synthesis.md`
- RQ3 traditional freeze:`report/RQ3_TRADITIONAL_SCIENTIFIC_FREEZE.md`
- RQ3 traditional manifest:
  `results/rq3_traditional_scientific_freeze/freeze_manifest.json`
- G3 synthesis:`report/g3_candidate_position_scientific_synthesis.md`
- G4 synthesis:`report/g4_f1bzq_scientific_synthesis.md`
- Packaged-artifact manifest:`results/manifests/github_packaged_artifacts.json`
- Git provenance map:
  `report/GITHUB_PROVENANCE_MAP.md`
- Professor quickstart: `docs/PROFESSOR_QUICKSTART.md`
- Data availability and external-material policy: `docs/DATA_AVAILABILITY.md`

Canonical numerical values should be read from the machine-readable artifacts under `results/` rather than manually copied from prose reports.

---

## Upstream Repository

The original repository is referenced as a Git submodule.

After cloning this repository, initialize it with:

```powershell
git submodule update --init original_repo/EvaLLMasRecommender
```

The expected upstream gitlink is:

```text
1e12982ca5c32d271cec6d1cb723e071f8c6587c
```

Do **not** use:

```powershell
git submodule update --init --recursive
```

for the final validation workflow.

The referenced upstream snapshot contains a nested `ReChorus` gitlink but does not provide the nested `.gitmodules` metadata required for Git to resolve that dependency recursively.

`ReChorus` is not required for validation of the frozen results and final figures included in this submission.

Traditional-model retraining through that missing nested checkout is outside the frozen-result validation scope.

---

## Python Environments

Two dependency descriptions are provided.

### General project environment

`requirements.txt`

This file contains the general project dependencies using compatible lower bounds.

It is suitable for general development and inspection of the project, but it is **not** intended to guarantee byte-identical regeneration of frozen rendered figures across arbitrary future package versions.

### Frozen release-validation environment

`requirements-release-validation.txt`

This file records the core package versions used for deterministic validation of the final frozen figure package.

The final clean-room validation of commit
`77961e96e7cab529153cbc9b0c8352391e544822` was performed on 25 October
2026 with:

```text
Python      3.12.13
NumPy       2.5.0
pandas      3.0.3
PyYAML      6.0.3
Matplotlib  3.11.0
pytest      9.1.1
```

The historical development environment also used:

```text
PyTorch 2.6.0+cu124
```

PyTorch is not required for frozen-result figure validation.

Byte-identical rendering is environment-sensitive. In particular, the figure-regeneration tests should be executed using the frozen validation stack rather than arbitrary newer versions allowed by the general `requirements.txt`.

---

## Environment Setup

A standard Python virtual environment can be created with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

For general project dependencies:

```powershell
python -m pip install -r requirements.txt
```

For deterministic validation of the frozen figure package:

```powershell
python -m pip install -r requirements-release-validation.txt
```

A Conda environment definition is also provided in:

`environment.yml`

---

## Packaged Large Artifacts

Several large canonical CSV artifacts are stored as deterministic gzip packages to remain compatible with repository distribution constraints.

### Verify packaged artifacts

```powershell
python scripts/restore_packaged_artifacts.py --verify-package
```

### Restore packaged artifacts

```powershell
python scripts/restore_packaged_artifacts.py --restore
```

### Verify restored artifacts

```powershell
python scripts/restore_packaged_artifacts.py --verify-restored
```

The restoration procedure checks:

- gzip SHA-256;
- restored CSV SHA-256;
- expected row counts;
- unsafe overwrite conditions.

The restored oversized CSV files remain ignored by Git.

The release inventory contains **51 packaged artifacts**. The clean-room
validation of the commit above verified all 51 packages, restored all 51 CSV
artifacts, and verified every restored SHA-256 and row count.

---

## Release Validation

The following workflow constitutes the validation path for the frozen final submission.

Run all commands from the repository root.

### 1. Verify and restore packaged artifacts

```powershell
python scripts/restore_packaged_artifacts.py --verify-package
python scripts/restore_packaged_artifacts.py --restore
python scripts/restore_packaged_artifacts.py --verify-restored
```

Expected result:

```text
PASS 51 artifacts
```

### 2. Run the repository smoke test

```powershell
python src/experiments/exp00_smoke_test.py
```

The final clean-room validation produced:

```text
Summary: 34 passed, 0 failed, 0 warnings
Smoke test PASSED
```

### 3. Validate the frozen cross-RQ figure

Before the cross-RQ figure validation, validate the RQ3 traditional
history-length freeze:

```powershell
python -m pytest tests/test_rq3_traditional_scientific_freeze.py -q
```

Expected result:

```text
1 passed
```

This validates the corrected aggregate, the original per-seed evidence, and
the manifest-hashed provenance inputs without training, inference, or new
recommendation generation.

### 4. Validate the frozen cross-RQ figure

```powershell
python -m src.analysis.final_figures.cross_rq --mode validate
```

Expected result:

```text
PASS
```

### 5. Validate the global final-figure freeze

```powershell
python -m src.analysis.final_figures.global_freeze --mode validate
```

Expected role counts:

```text
MAIN_TEXT       12
APPENDIX         6
REPOSITORY_ONLY  7
EXCLUDED         1
```

### 6. Run the focused final-figure release tests

```powershell
python -m pytest `
  tests/test_final_rq2_heatmaps.py `
  tests/test_final_render_polish.py `
  tests/test_global_final_figure_freeze.py `
  -q
```

The final clean-room validation produced:

```text
11 passed
```

Matplotlib may emit `PendingDeprecationWarning` messages concerning `cmap.set_bad(...)`.

These warnings do not invalidate the frozen artifacts or the validation results.

On Windows systems where the default user temporary directory is protected,
pytest may be unable to create its fixture directory. In that case, preserve
the same focused test selection and place pytest temporary files under the
clone, for example:

```powershell
python -m pytest `
  tests/test_final_rq2_heatmaps.py `
  tests/test_final_render_polish.py `
  tests/test_global_final_figure_freeze.py `
  -q --basetemp=pytest-release-basetemp
```

---

## Clean-Room Validation Result

On 25 October 2026, commit
`77961e96e7cab529153cbc9b0c8352391e544822` was validated from a new,
independent GitHub clone using configured GitHub authentication. This confirms
authenticated cloning only; anonymous clone access was not validated.

The validated commit successfully passed:

```text
Git checkout                         PASS
Upstream submodule checkout          PASS
Packaged artifacts                  51/51 PASS
Restored artifacts                  51/51 PASS
Repository smoke test               34/34 PASS
RQ3 traditional freeze               1/1 PASS
Cross-RQ figure validation           PASS
Global final-figure freeze           PASS
Final figure release tests          11/11 PASS
```

The focused release tests completed with 30 non-blocking Matplotlib
`PendingDeprecationWarning` messages concerning `cmap.set_bad(...)`. The
validation confirmed that the frozen scientific evidence and final-figure
provenance can be inspected and validated without relying on the original
development working directory, LLM inference, recommender training, or model
weights.

---

## Scope of the Test Suite

The focused workflow described above is the acceptance path for the frozen final submission.

The complete directory:

`tests/`

contains a broader **historical development test suite**.

Some historical tests:

- reconstruct earlier experimental stages;
- depend on reconstructed or local datasets not distributed in the curated repository;
- depend on historical intermediate evidence;
- check Git ancestry or checkpoint conditions relevant to earlier development phases;
- exercise training or execution paths outside the frozen-result validation scope.

Therefore, running:

```powershell
python -m pytest tests
```

from the final curated clone is **not** the release acceptance criterion.

This distinction separates three different reproducibility goals:

1. **Frozen-result validation**Validation of the committed final tables, figures, manifests, and scientific evidence.
2. **Protocol or fresh-execution reproducibility**Re-execution of experiments when the required datasets, model runtimes, weights, and external resources are available.
3. **Historical development validation**
   Tests associated with earlier repository states, checkpoints, recovery phases, and intermediate experimental artifacts.

Failures caused exclusively by unavailable historical intermediate datasets or obsolete checkpoint guards do not imply corruption of the final frozen evidence.

---

## External and Non-Distributed Evidence

The following resources are intentionally excluded from normal Git distribution where appropriate:

- raw LLM response archives;
- external archive ZIP files;
- raw datasets;
- reconstructed historical intermediate datasets;
- model weights;
- local Python environments;
- execution caches;
- temporary files;
- local logs.

These resources are not required to inspect the final scientific conclusions or validate the committed frozen figure package.

Additional information is available in:

`docs/external_archive_guide.md`

Fresh inference or complete reconstruction of historical experimental stages may require additional external resources.

---

## Final Figures

The final figure freeze contains the following roles:

```text
12 MAIN_TEXT
 6 APPENDIX
 7 REPOSITORY_ONLY
 1 EXCLUDED
```

Each active frozen figure is associated with provenance information that may include:

- canonical source artifacts;
- source SHA-256 hashes;
- generator source;
- generator hash;
- machine-readable figure values;
- rendered artifact hashes;
- role classification;
- freeze metadata.

The global freeze is stored under:

`results/manifests/final_figures/`

The final figure validation scripts verify these provenance chains directly.

---

## Experimental Layers

### G1 â€” Primary Ranking

The G1 layer evaluates local LLM recommendation under the primary-ranking protocol across the frozen extension datasets and prompt configurations.

Its evidence includes utility, beyond-utility, parsing, recommendation, execution, and request-level artifacts.

### G2 â€” History-Length Sensitivity

The G2 layer studies the effect of controlled changes in user-history length while preserving the relevant candidate and target conditions.

The analysis distinguishes early gains from later plateaus or configuration-dependent oscillations rather than assuming that longer histories are always better.

### RQ3 â€” Traditional History-Length Diagnostic Extension

The RQ3 traditional freeze evaluates `H = {1, 2, 3, 4, 5, 10, 15, 20}` for
MostPop, BM25-like/content, BPRMF, GRU4Rec, SASRec, and LightGCN using HR@5,
NDCG@5, and MRR. It is explicitly classified as
`LOCAL_OPERATIONALIZATION_NOT_PAPER_EXACT` and is not semantically equivalent
to the LLM RQ3 GE10/GE20 intervention: MostPop is H-invariant, BM25-like/content
truncates query history, and the trainable models use `D'_H` retraining.

GRU4Rec H1 and SASRec H1 are `NOT_APPLICABLE` because a one-item history cannot
form a next-item sequence example. Twelve Beauty/H5 aggregate keys were repaired
deterministically from the already existing seed set 42â€“46. No new training,
inference, Ollama execution, recommendation generation, or model rerun occurred.

### G3 â€” Candidate-Position Sensitivity

The G3 layer investigates the effect of candidate ordering through controlled paired conditions.

The frozen statistical analysis evaluates paired differences using the documented bootstrap convention.

### G4 â€” Candidate-Aware Reranking

The G4 layer evaluates reranking over frozen candidate pools.

Candidate-stage recall is explicitly separated from conditional reranking utility.

G1-versus-G4 comparisons are therefore treated descriptively rather than as a controlled causal comparison.

---

## Scientific Interpretation

The repository is designed around a conservative reproducibility principle:

> Functional alignment and controlled local execution do not imply historical or paper-exact identity.

Exact numerical comparison is justified only when the relevant experimental dimensions align sufficiently, including:

- evaluation population;
- candidate universe;
- model identity;
- prompt contract;
- output contract;
- metric definition;
- aggregation procedure.

Where these conditions do not align, the repository preserves the distinction explicitly rather than presenting adapted local evidence as an exact replication.

---

## Repository Map

- `src/`Project implementation, evaluation, analysis, experiment, and utility code.
- `tests/`Release-validation tests and historical development tests.
- `configs/`Experiment and analysis configurations.
- `results/tables/`Canonical metric tables and packaged evidence.
- `results/manifests/`Provenance, artifact, protocol, and freeze manifests.
- `results/plots/`Frozen scientific figures.
- `report/`Scientific reports, provenance documentation, validation material, and manuscript source.
- `docs/`Reproducibility and external-archive documentation.
- `data/`Data workspace. Raw and reconstructed external data are generally not part of the curated Git distribution.
- `paper/`Reference paper material.
- `original_repo/`
  Upstream repository preserved as a Git submodule.

---

## Reproducibility Claim

This repository supports two principal levels of reproducibility.

### Frozen-result reproducibility

The committed scientific artifacts, selected packaged tables, final figures, and their provenance chains can be validated from the distributed repository using the documented release-validation workflow.

### Fresh-execution reproducibility

The experimental protocols and implementation are documented, but fresh execution can depend on:

- external datasets;
- model availability;
- model weights;
- runtime versions;
- hardware;
- nondeterministic LLM behavior;
- external inference services or local model runtimes.

Fresh execution is therefore not claimed to reproduce the historical inference responses byte-for-byte.

The project claims a **controlled local reproduction and extension with explicit provenance and documented limitations**, not historical identity and not paper-exact numerical replication.
