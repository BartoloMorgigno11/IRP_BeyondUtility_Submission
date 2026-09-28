# Reproducibility Guide

## Scope

This guide covers the GitHub-publishable full research repository. It includes the F6M local adapted reproduction closure and the frozen G1-G4 extension layers. It is not the future professor-curated submission branch.

## Scientific Status

- Reproduction: `LOCAL_ADAPTED_REPRODUCTION_COMPLETE_NOT_PAPER_EXACT`
- F6M: complete and frozen with documented limitations.
- G1: primary ranking complete and frozen.
- G2: history-length sensitivity complete and frozen.
- G3: candidate-position sensitivity complete and frozen.
- G4: F1BZQ reranking complete and frozen.

Do not describe the repository as a paper-exact replication. Local LLM rows use substitute model identities and local strict parsing/adapted protocols where the paper artifacts were unavailable.

## Environment

Create a fresh Python environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On this workstation, `.venv_audit_recovery` is the repaired local audit environment used for no-inference checks. It is ignored by Git and is not a portable project artifact.

## Submodule Initialization

The upstream repository is tracked as a submodule:

```powershell
git submodule update --init --recursive
```

Expected gitlink:

```text
1e12982ca5c32d271cec6d1cb723e071f8c6587c
```

Do not edit `original_repo/EvaLLMasRecommender/` unless a future task explicitly authorizes it.

## Packaged Evidence

Some large foreground CSV artifacts are represented in Git as deterministic `.csv.gz` files. Verify packages without restoring:

The current release inventory contains 38 packages.

```powershell
.\.venv_audit_recovery\Scripts\python.exe scripts\restore_packaged_artifacts.py --verify-package
```

Restore ignored CSV originals when byte-level inspection is needed:

```powershell
.\.venv_audit_recovery\Scripts\python.exe scripts\restore_packaged_artifacts.py --restore
.\.venv_audit_recovery\Scripts\python.exe scripts\restore_packaged_artifacts.py --verify-restored
```

The restore utility performs no network calls, no model calls, and no metric computation. It verifies packaged SHA-256, restored SHA-256, and row counts from `results/manifests/github_packaged_artifacts.json`.

## Offline F6M Closure Rebuild

The F6M builder remains offline-only:

```powershell
.\.venv_audit_recovery\Scripts\python.exe -B src\experiments\build_phase_f6m_reproduction_closure.py --config configs\phase_f6m_reproduction_closure.yaml
```

It reads existing CSV/JSON evidence and must not run LLM inference, remote APIs, downloads, recommender training, raw-output repair, deduplication, or extension experiments.

## Evidence Hierarchy

Use this order when answering collaborator questions:

1. Current generated CSV tables and manifests in `results/tables/` and `results/manifests/`.
2. `report/final_scientific_report.md`.
3. G1-G4 synthesis and validation reports.
4. `report/phase_f6m_reproduction_closure.md` for reproduction-only closure.
5. `report/GITHUB_PROVENANCE_MAP.md` for rewritten commit provenance.
6. Historical reports and archived notes, treated as historical context only.
7. Local ignored raw/bulk artifacts only when a manifest or inventory row requires forensic inspection.

## Safety Rules

- Do not modify `original_repo/`.
- Do not report adapted or operationalized evidence as exact paper numerical replication.
- Do not manually type final metric values into reports; read them from CSV.
- Do not commit raw datasets, model checkpoints, local runtimes, LLM raw response dumps, or external archive ZIPs.
- Do not rerun LLM inference or Ollama unless a future task explicitly authorizes it with the required flags.

## Known Broad-Suite Risk

Historical broad-suite guard failures are documented in the phase reports. Treat them as recorded historical-suite risks unless a focused current validation fails.
