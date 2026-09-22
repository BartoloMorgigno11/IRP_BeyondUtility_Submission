# Repository Quality Review

Review date: 2026-06-28

Scope: `PROJECT_SPEC.md`, `AGENTS.md`, `CODEX_PROMPTS.md`, `README.md`,
configs, source code under `src/`, tests under `tests/`, reports under
`report/`, and generated CSVs/plots under `results/`. `original_repo/` was
checked only for git status and was not modified.

## Summary

- Full test suite: `124 passed, 2 skipped`.
- `results/metrics/` contains only `.gitkeep`; no empty or placeholder metric
  CSVs were found.
- No API calls, downloads, model training, or new metric computation were run.
- Generated diagnostic CSVs were refreshed after path/output-location fixes.
- `original_repo/` has no git status changes.

## Issues

| issue id | severity | file/location | description | suggested fix | status | reason if deferred |
|---|---|---|---|---|---|---|
| QR-001 | medium | `PROJECT_SPEC.md`, dataset audit output section | The spec still listed dataset audit CSVs under `results/metrics/`, while the project policy and actual audit phase use `results/tables/`. | Update the spec to `results/tables/<dataset>_dataset_audit.csv`. | fixed now |  |
| QR-002 | medium | `README.md`, Current Status | README still described the repository as Prompt 00 scaffold only, contradicting implemented modules, tests, generated diagnostics, and current blockers. | Replace the stale scaffold-only status with a concise current-status section that distinguishes diagnostics from real metrics. | fixed now |  |
| QR-003 | medium | `src/baselines/mostpop.py`, `tests/test_mostpop.py` | MostPop run summaries were written to `results/metrics/`, but they are diagnostic/run summaries rather than real metric values. | Write MostPop summaries to `results/tables/<dataset>_mostpop_summary.csv` and update tests. | fixed now |  |
| QR-004 | medium | `results/tables/*_dataset_audit.csv`, `*_llm_parsing_summary.csv`, `*_baseline_outputs_summary.csv`; generators in `src/data/audit_datasets.py`, `src/experiments/load_llm_outputs.py`, `src/baselines/load_baseline_outputs.py` | Generated diagnostic CSVs contained machine-local absolute Windows paths. These are not hardcoded in source, but they reduce portability and can leak local filesystem details when committed. | Store project-relative paths in generated diagnostic CSV fields and regenerate current outputs. | fixed now |  |
| QR-005 | medium | `src/experiments/run_pipeline.py`, `results/tables/*_final_summary.csv` | The pipeline treated `run_history_length.py` like other metric runners and recorded the non-written history-length metrics path as the phase path. That could be misread as a metrics artifact. | Select a returned `*_summary.csv` path as the phase path when a runner returns multiple paths; add a regression assertion. | fixed now |  |
| QR-006 | low | `tests/test_config.py` | The synthetic config fixture still modeled the old audit path under `results/metrics/`. | Update the fixture to use `results/tables/` for dataset audit outputs. | fixed now |  |
| QR-007 | medium | `results/tables/*_summary.csv` status columns | Status names are phase-specific: for example `parsed`, `found`, `computed`, `generated`, `ready`, `ready_with_warnings`, `blocked`, `missing`, and `not_configured`. They are understandable locally but not normalized across all summary CSVs. | Introduce a shared status vocabulary or a `status_family` column in a future cleanup. | deferred | This is a schema polish change across many existing summaries and is better handled before final documentation or in a dedicated status-normalization phase. |
| QR-008 | medium | `results/raw_outputs/beauty_baseline_outputs_normalized.csv` | The normalized Beauty baseline output is large, about 94 MB, and is already tracked by git. It may be inappropriate for normal commits unless the project intentionally commits generated normalized artifacts. | Decide whether large generated raw outputs should stay tracked, move to Git LFS/external storage, or be regenerated on demand. | deferred | Removing or untracking an existing generated artifact would be a repository policy decision and could disrupt current reproducibility evidence. |
| QR-009 | low | `report/results_analysis.md` | The Final Safe Pipeline section reflects the last dataset run, currently Sports, while separate Beauty and Sports final summary CSVs both exist. The plot section records both datasets. | In final analysis refinement, summarize both dataset final summaries in one report section. | deferred | This is documentation/report composition, not a code correctness issue. Prompt 23 is intended for results-analysis refinement. |
| QR-010 | low | CLI stdout/logs for several scripts | Some CLI messages print absolute paths even though persisted CSV fields now use relative paths. This is not a committed artifact issue but is less tidy for reproducibility logs. | Add a shared `_display_path` helper to all CLI print/log lines in a future cleanup. | deferred | Current persisted artifacts are portable; log display cleanup is lower priority and should not broaden this phase. |

## Checks With No Finding

- No `original_repo/` modifications were detected by `git status --short original_repo`.
- No Python source, configs, reports, or result CSVs contain absolute local
  paths after regeneration.
- No `.csv` files exist under `results/metrics/`; only `.gitkeep` is present.
- API-related scripts and runners either do not expose API behavior or reject
  configs where `api.allow_api_calls` or `experiment.allow_api_calls` is true.
- Implemented modules have synthetic test coverage under `tests/`.
- Current configs include required sections and keep API calls disabled.
- Readiness and diagnostic outputs are under `results/tables/`; plots are under
  `results/plots/`; normalized model outputs are under `results/raw_outputs/`.
- Current reports avoid claiming real HR@5, NDCG@5, hallucination, coverage,
  position-bias, history-length, or cost metric values.

## Commands Run

```powershell
rg -n "<absolute-path-pattern>" src tests configs report README.md PROJECT_SPEC.md AGENTS.md results -g !*.png
rg -n "requests|urllib|httpx|openai|anthropic|api_key|allow_api_calls|--allow-api-calls" src tests configs README.md report -g !*.png
rg -n "results[/\\]metrics|metrics_dir|_metrics\\.csv|summary\\.csv|dataset_audit\\.csv|parsing_summary|mostpop_summary" src tests configs report README.md PROJECT_SPEC.md CODEX_PROMPTS.md -g !*.png
python -m compileall src/data/audit_datasets.py src/experiments/load_llm_outputs.py src/baselines/load_baseline_outputs.py src/baselines/mostpop.py src/experiments/run_pipeline.py
pytest tests/test_config.py tests/test_dataset_audit.py tests/test_load_llm_outputs.py tests/test_load_baseline_outputs.py tests/test_mostpop.py tests/test_run_pipeline.py
python src/data/audit_datasets.py --config configs/beauty.yaml
python src/data/audit_datasets.py --config configs/sports.yaml
python src/experiments/load_llm_outputs.py --config configs/beauty.yaml --log-unmatched-limit 0
python src/experiments/load_llm_outputs.py --config configs/sports.yaml --log-unmatched-limit 0
python src/baselines/load_baseline_outputs.py --config configs/beauty.yaml
python src/baselines/load_baseline_outputs.py --config configs/sports.yaml
python src/experiments/run_pipeline.py --config configs/beauty.yaml
python src/experiments/run_pipeline.py --config configs/sports.yaml
python src/experiments/make_plots.py --config configs/beauty.yaml
python src/experiments/make_plots.py --config configs/sports.yaml
pytest tests/
```

## Current Commit Guidance

A commit is safe after reviewing the large tracked raw-output file policy.
The code and generated diagnostics are internally consistent, tests pass, and
`original_repo/` was not modified.
