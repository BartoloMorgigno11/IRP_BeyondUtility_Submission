# Professor Quickstart

This guide validates the committed frozen evidence. It does not rerun LLM
inference, recommender training, or model downloads.

## 1. Clone and initialise the upstream snapshot

Clone with the access method supplied for this repository, then initialise
only the documented upstream submodule:

```powershell
git clone https://github.com/BartoloMorgigno11/IRP_BeyondUtility_Submission.git
cd IRP_BeyondUtility_Submission
git submodule update --init original_repo/EvaLLMasRecommender
```

The expected upstream gitlink is
`1e12982ca5c32d271cec6d1cb723e071f8c6587c`. Do not use recursive submodule
initialisation: the upstream snapshot has a nested gitlink that is not needed
for frozen-result validation.

## 2. Create the release-validation environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-release-validation.txt
```

The clean-room validation of commit
`77961e96e7cab529153cbc9b0c8352391e544822` used Python 3.12.13, NumPy
2.5.0, pandas 3.0.3, PyYAML 6.0.3, Matplotlib 3.11.0, and pytest 9.1.1.

## 3. Restore packaged evidence

Run these commands from the repository root:

```powershell
python scripts/restore_packaged_artifacts.py --verify-package
python scripts/restore_packaged_artifacts.py --restore
python scripts/restore_packaged_artifacts.py --verify-restored
```

The manifest is `results/manifests/github_packaged_artifacts.json`. The
release contains 51 deterministic gzip packages. Restore creates ignored CSV
files, verifies package and restored SHA-256 values, and checks row counts.
It does not call models or recompute metrics.

## 4. Run the official frozen-result checks

```powershell
python src/experiments/exp00_smoke_test.py
python -m pytest tests/test_rq3_traditional_scientific_freeze.py -q
python -m src.analysis.final_figures.cross_rq --mode validate
python -m src.analysis.final_figures.global_freeze --mode validate
python -m pytest `
  tests/test_final_rq2_heatmaps.py `
  tests/test_final_render_polish.py `
  tests/test_global_final_figure_freeze.py `
  -q
```

Expected clean-room outcomes are 34/34 smoke checks, 1/1 RQ3 freeze test,
PASS for cross-RQ and global-freeze validation, and 11/11 focused tests.
Matplotlib can emit non-blocking `PendingDeprecationWarning` messages about
`cmap.set_bad(...)`.

If Windows protects the default pytest temporary directory, append
`--basetemp=pytest-release-basetemp` to the focused pytest command. This only
changes the location of temporary test files.

## 5. Find the scientific evidence

| Research question | Principal frozen evidence |
|---|---|
| RQ1 — utility | `results/final_figures/rq1/`, `report/g1_primary_ranking_execution.md` |
| RQ2 — beyond-utility | `results/final_figures/rq2/`, `results/tables/g1_primary_ranking_beyond_utility.csv` |
| RQ3 — history sensitivity | `results/final_figures/rq3/`, `report/g2_history_length_scientific_synthesis.md`, `results/rq3_traditional_scientific_freeze/` |
| RQ4 — candidate order | `results/final_figures/rq4/`, `report/g3_candidate_position_scientific_synthesis.md` |
| RQ5 — generation-involved profiles | `results/final_figures/rq5/`, `report/rq5_final_scientific_freeze.md` |
| RQ6 — output reliability | `results/final_figures/rq6/` |
| RQ7 — reranking | `results/final_figures/rq7/`, `report/g4_f1bzq_scientific_synthesis.md` |

Start with `report/final_scientific_report.md` for the scientific boundary:
this is an adapted local reproduction and extension, not a paper-exact
numerical replication.
