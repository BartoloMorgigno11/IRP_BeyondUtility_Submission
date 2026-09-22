# Project Notes

## Current Canonical State

- Reproduction status: `LOCAL_ADAPTED_REPRODUCTION_COMPLETE_NOT_PAPER_EXACT`.
- F6M reproduction core: complete and frozen with documented limitations.
- Extension status: G1, G2, G3, and G4 complete and frozen.
- Current delivery branch: `delivery/full-research`.
- Pre-GitHub frozen scientific checkpoint: `cea96181cb78b249f5aadc490d6715d89e86ebde`.
- Canonical entrypoints: `README.md`, `report/final_scientific_report.md`, `report/phase_f6m_reproduction_closure.md`, G1-G4 phase reports, `report/GITHUB_PROVENANCE_MAP.md`, `results/manifests/github_packaged_artifacts.json`, and generated CSV tables under `results/tables/`.
- Historical notes below are retained for lineage and may describe states that were later superseded.

## 2026-09-06 - GitHub-Publishable Full Research Packaging

- Created immutable backup branch `backup/pre-github-cea9618` at `cea96181cb78b249f5aadc490d6715d89e86ebde`.
- Reconstructed the 37 unpublished scientific commits onto `delivery/full-research` from `origin/master` `72565a6ffc00dc9a2a52c7828656eb4a8a7b1647`.
- Replaced only GitHub-incompatible or near-limit foreground CSV blobs with deterministic gzip packages. The restored bytes match the original SHA-256 values recorded in `results/manifests/github_packaged_artifacts.json`.
- Added G1 foreground CSV packages for no-inference inspection without tracking raw LLM response files.
- Added `.gitmodules` for the upstream `original_repo/EvaLLMasRecommender` submodule while preserving gitlink `1e12982ca5c32d271cec6d1cb723e071f8c6587c`.
- Added restore/verify utility `scripts/restore_packaged_artifacts.py`.
- Added provenance maps `report/GITHUB_PROVENANCE_MAP.md` and `results/manifests/github_commit_provenance_map.csv`.
- No experiments, LLM inference, Ollama calls, API calls, metric regeneration, candidate-pool regeneration, raw dataset edits, model-weight edits, or upstream repository edits were performed.

## 2026-08-14 - GitHub-Readiness Cleanup Execution

- Added `.gitattributes`, expanded `.gitignore`, and generated `results/tables/repository_cleanup_inventory.csv` plus `report/github_readiness_cleanup_audit.md`.
- Updated the canonical collaborator entrypoints: `README.md`, `docs/artifact_catalog.md`, `docs/reproducibility_guide.md`, `docs/external_archive_guide.md`, and this notes header.
- Copied superseded foreground reports to `report/archive/phase_reports/` and old Plan C plots to `results/archive/plots/`; source and copied SHA-256 hashes were verified before accepting tracked deletions.
- Replaced operational workstation-specific paths in source/tests and current documentation with repo-relative commands or neutral test fixtures. Historical archive reports retain local paths as provenance evidence.
- Validation run: focused cleanup/metric/docs/F6M suite passed with `241 passed`; F6I2M hashing helper suite passed with `19 passed`; full test collection found `1954` tests.
- Known validation limit: the full F6I2K manifest-generation regression module was not completed because its in-memory build hashes protected bulk directories such as raw data, processed data, LLM outputs, models, and the transitional archive. The lightweight common-manifest API subset passed.
- Root `archive/`, bulk local results, duplicate paper convenience files, caches, environments, and logs are ignored/externalized rather than recursively deleted in this batch.

## 2026-06-26 - Prompt 00: Initial Setup and Project Scaffold

### Changes

- Created the initial project folder structure for `src/`, `configs/`, `data/`,
  `results/`, `tests/`, and `report/`.
- Added lightweight Python package markers under `src/`.
- Added initial setup files: `README.md`, `.gitignore`, `requirements.txt`, and
  `environment.yml`.
- Added a scaffold test that verifies the expected Prompt 00 directories exist.

### Commands Run

- `Get-Content -Path PROJECT_SPEC.md`
- `Get-Content -Path AGENTS.md`
- `Get-ChildItem -Force`
- `Select-String -Path CODEX_PROMPTS.md -Pattern "Phase|phase|PHASE|Initialize|scaffold" -Context 2,6`
- `Get-Content -Path CODEX_PROMPTS.md -TotalCount 65`
- `python --version` attempted, but local Python resolution is broken.
- `pytest tests/` attempted, but could not run because Python is not launchable in this shell.

### Outputs Generated

- No experiment outputs were generated.
- No numerical results were created.

### Known Issues

- The original repository has not been inspected or mapped yet.
- Dataset availability and formats have not been audited yet.
- Verification is blocked until Python is available: `python` resolves to the
  Windows Apps stub, and `.venv\Scripts\python.exe` points at a missing Store
  Python path.

### Next Step

- Run Prompt 01 to inspect `original_repo/` and create `report/codebase_map.md`.

## 2026-06-26 - Prompt 01: Original Repository Codebase Map

### Changes

- Inspected `original_repo/EvaLLMasRecommender` without modifying it.
- Created `report/codebase_map.md` documenting repository structure, available
  compressed LLM outputs, Beauty SOTA outputs, metrics, scripts, risks, and the
  recommended minimal reproduction path.

### Commands Run

- `Get-Content -Path PROJECT_SPEC.md`
- `Get-Content -Path AGENTS.md`
- `Get-ChildItem -Force original_repo\EvaLLMasRecommender`
- `rg --files original_repo\EvaLLMasRecommender`
- `Get-Content -Path original_repo\EvaLLMasRecommender\README.md`
- `Get-Content -Path original_repo\EvaLLMasRecommender\requirements.txt`
- `Get-Content -Path original_repo\EvaLLMasRecommender\eva\codes\llm_metrics.py`
- `Get-Content -Path original_repo\EvaLLMasRecommender\eva\codes\output_refine.py`
- `Get-Content -Path original_repo\EvaLLMasRecommender\eva\codes\zero_shot.py`
- `Get-Content -Path original_repo\EvaLLMasRecommender\eva\codes\llms.py`
- `Get-Content -Path original_repo\EvaLLMasRecommender\eva\codes\random_test.py`
- `Get-Content -Path original_repo\EvaLLMasRecommender\eva\codes\metrics.sh`
- `Get-Content -Path original_repo\EvaLLMasRecommender\eva\codes\random.sh`
- Read-only ZIP manifest and sample-record inspection for
  `llm_results/beauty.zip`, `llm_results/sports.zip`, and
  `sota_results/beauty_his10.zip`.

### Outputs Generated

- `report/codebase_map.md`

### Known Issues

- No tests were required because this phase added documentation only.
- The upstream preprocessed `data/` and `raw_data/` folders are not present in
  the inspected checkout.
- Sports SOTA output files are referenced by scripts but were not present.
- Direct upstream metric scripts are risky because they can write outputs into
  `original_repo/` and depend on missing data files.

### Next Step

- Run the dataset audit phase after locating or unpacking the expected
  preprocessed data files.

## 2026-06-26 - Prompt 02: Smoke Test Script

### Changes

- Added `src/experiments/exp00_smoke_test.py`.
- The script checks repository root execution, expected scaffold folders,
  `configs/`, `original_repo/EvaLLMasRecommender`, `paper/`, the paper PDF when
  available, `results/` subfolders, `report/`, key documentation files, and
  importability of `src` packages.
- Added an optional `--config` argument for CLI compatibility without requiring
  YAML configs before the config phase.

### Commands Run

- `Get-Content -Path PROJECT_SPEC.md`
- `Get-Content -Path AGENTS.md`
- `Get-Content -Path report\codebase_map.md`
- `python src/experiments/exp00_smoke_test.py`
- `pytest tests/`

### Outputs Generated

- No experiment outputs or numerical results were generated.
- Smoke test output reported `34 passed, 0 failed, 0 warnings`.

### Known Issues

- In the sandbox, `python` still fails to launch with a Windows login-session
  error. The smoke test and pytest both passed when rerun outside the sandbox.
- YAML dataset config files are not expected yet; they are part of the next
  phase.

### Next Step

- Create YAML config files for Beauty and Sports in the configuration phase.

## 2026-06-26 - Prompt 03: Dataset YAML Configs

### Changes

- Added `configs/beauty.yaml`.
- Added `configs/sports.yaml`.
- Kept unaudited dataset file paths as `null` with comments explaining they
  must be filled after dataset audit.
- Referenced only upstream ZIP paths that were already identified in
  `report/codebase_map.md`.
- Did not create optional ML-1M or cost placeholder configs to keep this phase
  focused on the minimum Beauty and Sports setup.

### Commands Run

- `Get-Content -Path PROJECT_SPEC.md`
- `Get-Content -Path AGENTS.md`
- `python -c "import yaml, pathlib; [print(p, yaml.safe_load(open(p, encoding='utf-8'))['dataset']['name']) for p in pathlib.Path('configs').glob('*.yaml')]"`
- `python src/experiments/exp00_smoke_test.py --config configs/beauty.yaml`

### Outputs Generated

- `configs/beauty.yaml`
- `configs/sports.yaml`

### Known Issues

- The actual interactions, catalog, candidates, and ground-truth files remain
  unresolved until dataset audit.
- Sports baseline outputs are still unknown in this checkout, so
  `baseline_outputs_path` is `null`.
- The Python validation commands required running outside the sandbox because
  sandboxed Python launch still fails in this environment.

### Next Step

- Implement dataset audit utilities that read these configs and report missing
  or available files without inventing dataset facts.

## 2026-06-26 - Configuration Infrastructure

### Changes

- Added `src/utils/config.py` with helpers to find the project root, load YAML
  configs relative to the root, validate experiment config structure, reject
  absolute configured paths, and resolve configured paths to absolute `Path`
  objects.
- Added `tests/test_config.py` for root discovery, YAML loading, schema
  validation, API-default safety, absolute-path rejection, and path resolution.
- Added `tests/conftest.py` so tests can import the local `src` package
  consistently.
- Updated `configs/beauty.yaml` and `configs/sports.yaml` with explicit
  `paths` and `api` sections while keeping API calls disabled by default.
- Did not add `configs/cost.yaml`; the cost config has a distinct schema and is
  better introduced with the later cost-metrics phase.

### Commands Run

- `Get-Content -Path PROJECT_SPEC.md`
- `Get-Content -Path AGENTS.md`
- `Get-Content -Path CODEX_PROMPTS.md`
- `Get-Content -Path report\codebase_map.md`
- `Get-Content -Path configs\beauty.yaml`
- `Get-Content -Path configs\sports.yaml`
- `Get-Content -Path src\experiments\exp00_smoke_test.py`
- `python -m compileall src\utils\config.py`
- `pytest tests/test_config.py`
- `pytest tests/`
- `python -c "from src.utils.config import load_yaml_config, find_project_root, resolve_configured_paths; root=find_project_root(); [print(path, load_yaml_config(path, root)['dataset']['name'], len(resolve_configured_paths(load_yaml_config(path, root), root))) for path in ('configs/beauty.yaml','configs/sports.yaml')]"`
- `python src/experiments/exp00_smoke_test.py --config configs/beauty.yaml`
- `python src/experiments/exp00_smoke_test.py --config configs/sports.yaml`

### Outputs Generated

- No experiment outputs or numerical results were generated.
- Config validation reported:
  - `configs/beauty.yaml beauty 32`
  - `configs/sports.yaml sports 31`
- `pytest tests/` reported 10 passing tests.

### Known Issues

- Dataset file paths for interactions, catalog, candidates, and ground truth
  remain `null` until the dataset audit phase.
- Sports baseline output remains unavailable in the current checkout.
- Python commands still require running outside the sandbox because sandboxed
  Python launch fails in this environment.

### Next Step

- Implement the dataset audit phase using `src/utils/config.py` to load and
  validate configs.

## 2026-06-26 - Dataset Audit Utilities

### Changes

- Added `src/data/load_data.py` with minimal safe helpers for reading JSON,
  CSV/TSV, text line counts, and ZIP members without extraction.
- Added `src/data/audit_datasets.py`, a root-runnable CLI accepting `--config`.
- The audit loads YAML through `src/utils/config.py`, checks configured paths,
  searches the original repository for likely canonical dataset files when
  config paths are `null`, summarizes configured LLM and baseline ZIP artifacts,
  writes CSV audit output, and regenerates `report/dataset_audit.md`.
- Added `tests/test_dataset_audit.py` for ZIP helper behavior and audit output
  expectations.

### Commands Run

- `Get-Content -Path PROJECT_SPEC.md`
- `Get-Content -Path AGENTS.md`
- `Get-Content -Path configs\beauty.yaml`
- `Get-Content -Path configs\sports.yaml`
- `Get-Content -Path src\utils\config.py`
- `python -m compileall src\data\load_data.py src\data\audit_datasets.py`
- `python src/data/audit_datasets.py --config configs/beauty.yaml`
- `python src/data/audit_datasets.py --config configs/sports.yaml`
- `python -m compileall src\data`
- `pytest tests/test_dataset_audit.py`
- `pytest tests/`

### Outputs Generated

- `results/tables/beauty_dataset_audit.csv`
- `results/tables/sports_dataset_audit.csv`
- `report/dataset_audit.md`

### Known Issues

- Canonical interactions, catalog, candidates, and ground-truth files are still
  not configured; full dataset user/item/interaction stats remain `unknown`.
- Candidate lists are available in configured LLM output prompt records, but
  these are not treated as full canonical dataset files.
- Sports baseline outputs remain unavailable in the current checkout.
- Python commands still require running outside the sandbox because sandboxed
  Python launch fails in this environment.

### Next Step

- Implement unified dataframe schema validators before metric implementation.

## 2026-06-26 - Dataset Audit Output Location Correction

### Changes

- Updated dataset audit CSV output paths from `results/metrics/` to
  `results/tables/`.
- Updated `configs/beauty.yaml` and `configs/sports.yaml` so
  `output.dataset_audit_path` points to `results/tables/`.
- Updated `src/data/audit_datasets.py` fallback output path and Markdown report
  scan location.
- Updated dataset audit tests to assert audit CSVs are configured under
  `results/tables/`.
- Reviewed `src/data/load_data.py`; it only contains safe generic JSON, CSV/TSV,
  text-line, and ZIP helpers, so no reduction was needed.
- Removed stale generated audit CSVs from `results/metrics/`.

### Commands Run

- `Get-Content -Path AGENTS.md`
- `Get-Content -Path PROJECT_SPEC.md`
- `Get-Content -Path report\codebase_map.md`
- `Get-Content -Path configs\beauty.yaml`
- `Get-Content -Path configs\sports.yaml`
- `Get-Content -Path src\utils\config.py`
- `Get-Content -Path src\data\audit_datasets.py`
- `Get-Content -Path src\data\load_data.py`
- `Get-Content -Path tests\test_dataset_audit.py`
- `python -m compileall src\data\audit_datasets.py src\data\load_data.py`
- `python src/data/audit_datasets.py --config configs/beauty.yaml`
- `python src/data/audit_datasets.py --config configs/sports.yaml`

### Outputs Generated

- `results/tables/beauty_dataset_audit.csv`
- `results/tables/sports_dataset_audit.csv`
- `report/dataset_audit.md`

### Known Issues

- Canonical interactions, catalog, candidates, and ground-truth paths remain
  unresolved until data preparation/audit inputs are available.
- No dataset statistics beyond available artifact-level audit facts were
  invented.

### Next Step

- Rerun the test suite, then continue to unified dataframe schema validators.

## 2026-06-27 - Dataframe Schema Validators

### Changes

- Added `src/data/schemas.py` with simple validators for interactions,
  candidates, recommendations, and ground-truth dataframes.
- Validators normalize column names by trimming whitespace and lowercasing,
  preserve extra columns after canonical columns, and return dataframe copies.
- Validators raise clear `ValueError` messages for missing required columns,
  missing required values, duplicate normalized column names, and
  recommendation rows without either an item id or item text.
- Added focused unit tests using synthetic data only.

### Commands Run

- `Get-Content -Path PROJECT_SPEC.md | Out-Null`
- `Get-Content -Path AGENTS.md | Out-Null`
- `python -m compileall src\data\schemas.py`
- `pytest tests\test_schemas.py`
- `pytest tests\`

### Outputs Generated

- No result CSVs, plots, or dataset-dependent outputs were generated.

### Known Issues

- The validators only enforce canonical dataframe shape and required values.
  They do not load real datasets, infer dataset-specific schemas, parse LLM
  outputs, compute metrics, or validate ranking semantics.

### Next Step

- Implement metric functions and tests when the metrics phase is requested.

## 2026-06-27 - Utility Metrics

### Changes

- Added `src/evaluation/utility_metrics.py` with `hit_rate_at_k`,
  `ndcg_at_k`, and `evaluate_utility_at_k`.
- Metrics support one or many relevant items per user, exclude users without
  ground truth from denominators, use `rank` ordering when present, fall back to
  input order otherwise, and handle empty recommendation dataframes.
- Duplicate recommendations do not add repeated hits or gain; their original
  rank positions are preserved.
- Added synthetic tests with exact expected values. The original repository
  metrics were not used.

### Commands Run

- `Get-Content -Path PROJECT_SPEC.md | Out-Null`
- `Get-Content -Path AGENTS.md | Out-Null`
- `python -m compileall src\evaluation\utility_metrics.py`
- `pytest tests\test_utility_metrics.py`
- `pytest tests\`

### Outputs Generated

- No result CSVs, plots, API outputs, or dataset-dependent outputs were
  generated.

### Known Issues

- Utility metrics currently evaluate exact item identifiers or exact text values
  only. Normalized, fuzzy, hallucination, coverage, baseline, and LLM-output
  parsing logic remain future phases.

### Next Step

- Implement output parsing or beyond-utility metrics only when explicitly
  requested.

## 2026-06-27 - LLM Output Parser

### Changes

- Added `src/evaluation/output_parser.py` to parse textual LLM recommendation
  outputs into ranked recommendation records.
- Supported numbered lists, bullet lists, comma-separated outputs, quoted item
  titles, explanatory text around lists, and catalog item IDs embedded in the
  output.
- Added text normalization that lowercases, strips punctuation, removes leading
  ranking/list markers, and collapses whitespace.
- Kept item-id, exact, normalized, fuzzy, and unmatched match types explicit.
  Unmatched parsed text is retained with missing `recommended_item_id` rather
  than silently converted into a catalog item.
- Added synthetic parser tests. Fuzzy matching is optional and uses
  `rapidfuzz` only when installed.

### Commands Run

- `Get-Content -Path PROJECT_SPEC.md | Out-Null`
- `Get-Content -Path AGENTS.md | Out-Null`
- `python -m compileall src\evaluation\output_parser.py`
- `pytest tests\test_output_parser.py`
- `pytest tests\`

### Outputs Generated

- No result CSVs, plots, API outputs, or dataset-dependent outputs were
  generated.

### Known Issues

- `rapidfuzz` is not installed in the current environment, so the positive
  fuzzy-matching test is skipped. The parser raises a clear `ImportError` if
  fuzzy matching is requested without `rapidfuzz`.
- Parser matching is catalog-driven; unmatched free text remains unmatched by
  design.

### Next Step

- Implement hallucination or coverage metrics only when explicitly requested.

## 2026-06-27 - Hallucination Metrics

### Changes

- Added `src/evaluation/hallucination.py` with `hallucination_rate`,
  `candidate_hallucination_rate`, and `catalog_hallucination_rate`.
- Hallucination summaries distinguish item-id, exact, normalized, fuzzy, and
  unmatched outputs.
- Fuzzy matches are never counted as exact matches. Summaries include both the
  standard unmatched-only hallucination rate and a stricter fuzzy-plus-unmatched
  rate for utility-safe reporting.
- Candidate hallucination supports user-specific candidate sets when `user_id`
  is present.
- Added synthetic tests for match-type counting, catalog matching, user-specific
  candidate matching, top-k handling, unknown match types, and optional fuzzy
  matching behavior.

### Commands Run

- `Get-Content -Path PROJECT_SPEC.md | Out-Null`
- `Get-Content -Path AGENTS.md | Out-Null`
- `python -m compileall src\evaluation\hallucination.py`
- `pytest tests\test_hallucination.py`
- `pytest tests\`

### Outputs Generated

- No result CSVs, plots, API outputs, or dataset-dependent outputs were
  generated.

### Known Issues

- `rapidfuzz` is not installed in the current environment, so positive fuzzy
  matching tests are skipped. Enabling fuzzy matching without `rapidfuzz` raises
  a clear `ImportError`.
- These metrics consume parsed recommendation records and candidate/catalog
  dataframes only; they do not load real datasets or parse original-repo files.

### Next Step

- Implement coverage metrics only when explicitly requested.

## 2026-06-27 - Cached LLM Output Loader

### Changes

- Added `src/experiments/load_llm_outputs.py` with a root-runnable CLI:
  `python src/experiments/load_llm_outputs.py --config configs/beauty.yaml`.
- The loader reads configured cached LLM JSON outputs from direct JSON files or
  ZIP members, parses them through `src/evaluation/output_parser.py`, and
  writes normalized recommendation records under `results/raw_outputs/`.
- The loader writes parsing diagnostics under `results/tables/` and creates
  clear diagnostic CSV rows when configured outputs are missing or unreadable.
- Added synthetic loader tests covering successful ZIP parsing and missing
  source-file diagnostics.
- Improved `parse_recommendation_items` for stringified list outputs with
  commas and apostrophes in item titles, matching the observed cached-output
  format.

### Commands Run

- `Get-Content -Path PROJECT_SPEC.md | Out-Null`
- `Get-Content -Path AGENTS.md | Out-Null`
- `Get-Content -Path report\codebase_map.md | Out-Null`
- `python -m compileall src\experiments\load_llm_outputs.py src\evaluation\output_parser.py`
- `pytest tests\test_load_llm_outputs.py tests\test_output_parser.py`
- `pytest tests\`
- `python src\experiments\load_llm_outputs.py --config configs\beauty.yaml --log-unmatched-limit 3`
- `python src\experiments\load_llm_outputs.py --config configs\sports.yaml --log-unmatched-limit 3`

### Outputs Generated

- `results/raw_outputs/beauty_llm_outputs_normalized.csv`
- `results/raw_outputs/sports_llm_outputs_normalized.csv`
- `results/tables/beauty_llm_parsing_summary.csv`
- `results/tables/sports_llm_parsing_summary.csv`

The generated summary CSVs report `catalog_not_configured`, so parsed
recommendations remain unmatched instead of being converted into item IDs.

### Known Issues

- Canonical catalog paths are still null in the current configs. Therefore
  normalized output files contain parsed recommendation text and provenance, but
  no validated catalog item IDs.
- No API calls were made and no fake outputs were created.

### Next Step

- Configure or derive audited catalog files before using parsed LLM outputs for
  exact utility evaluation.

## 2026-06-27 - Prompt 09 Output Location Correction

### Changes

- Updated `src/experiments/load_llm_outputs.py` so LLM normalized raw outputs
  remain under `results/raw_outputs/`, while LLM parsing summary CSVs are saved
  under `results/tables/`.
- Updated loader tests to expect parsing summaries in `results/tables/`.
- Removed stale generated LLM parsing summaries from `results/metrics/` after
  regenerating them in `results/tables/`.

### Commands Run

- `Get-Content -Path AGENTS.md | Out-Null`
- `Get-Content -Path PROJECT_SPEC.md | Out-Null`
- `Get-Content -Path src\experiments\load_llm_outputs.py | Out-Null`
- `Get-Content -Path configs\beauty.yaml | Out-Null`
- `Get-Content -Path configs\sports.yaml | Out-Null`
- `Get-Content -Path report\notes.md | Out-Null`
- `python -m compileall src\experiments\load_llm_outputs.py`
- `pytest tests\test_load_llm_outputs.py`
- `python src\experiments\load_llm_outputs.py --config configs\beauty.yaml --log-unmatched-limit 0`
- `python src\experiments\load_llm_outputs.py --config configs\sports.yaml --log-unmatched-limit 0`
- Removed stale `results\metrics\beauty_llm_parsing_summary.csv` and
  `results\metrics\sports_llm_parsing_summary.csv`
- `pytest tests\`

### Outputs Generated

- `results/raw_outputs/beauty_llm_outputs_normalized.csv`
- `results/raw_outputs/sports_llm_outputs_normalized.csv`
- `results/tables/beauty_llm_parsing_summary.csv`
- `results/tables/sports_llm_parsing_summary.csv`

### Known Issues

- Catalog paths remain null, so parsed recommendations remain unmatched rather
  than being converted into item IDs.
- No API calls were made and no fake outputs were created.

### Next Step

- Continue only with the next explicitly requested phase.

## 2026-06-29 - Results Analysis Refinement

### Changes

- Rewrote `report/results_analysis.md` into a final, exam-ready results
  analysis aligned with the README, technical report, reproducibility
  checklist, artifact map, quality review, configs, and generated CSV files.
- Reorganized the analysis around available generated outputs, Beauty and
  Sports dataset-level readiness, metric-level blockers, external artifact
  availability, plot interpretation, the value of readiness-aware reporting,
  and the distinction between reproduction and reconstruction.
- Removed the stale structure that emphasized only the last Sports pipeline run
  and replaced it with a two-dataset interpretation using existing summary CSV
  evidence.

### Commands Run

- `Get-Content -Raw PROJECT_SPEC.md`
- `Get-Content -Raw AGENTS.md`
- `Select-String -Path CODEX_PROMPTS.md -Pattern "Prompt 23" -Context 8,80`
- `Get-Content -Raw README.md`
- `Get-Content -Raw report/technical_report.md`
- `Get-Content -Raw report/reproducibility_checklist.md`
- `Get-Content -Raw report/original_artifact_map.md`
- `Get-Content -Raw report/quality_review.md`
- `Get-Content -Raw report/results_analysis.md`
- `Get-Content -Tail 180 report/notes.md`
- `Get-Content -Raw configs/beauty.yaml`
- `Get-Content -Raw configs/sports.yaml`
- `Get-ChildItem -Recurse -File results`
- `Get-Content -Raw results/tables/beauty_final_summary.csv`
- `Get-Content -Raw results/tables/sports_final_summary.csv`
- `Get-Content -Raw results/tables/beauty_plot_summary.csv`
- `Get-Content -Raw results/tables/sports_plot_summary.csv`
- `Get-ChildItem -Force results/metrics`
- `Import-Csv results/tables/beauty_final_summary.csv | Group-Object status`
- `Import-Csv results/tables/sports_final_summary.csv | Group-Object status`
- `Get-Content -Raw results/tables/beauty_llm_parsing_summary.csv`
- `Get-Content -Raw results/tables/sports_llm_parsing_summary.csv`
- `Get-Content -Raw results/tables/beauty_baseline_outputs_summary.csv`
- `Get-Content -Raw results/tables/sports_baseline_outputs_summary.csv`
- `Test-Path report/results_analysis.md`
- `Get-ChildItem -Force results/metrics`
- `git status --short original_repo`
- `rg -n "full paper was reproduced|fake metric|fabricated|blocked|not written|not computed|results/metrics|unavailable/deleted|reconstruction" report/results_analysis.md`
- `git status --short`

### Outputs Generated

- Updated `report/results_analysis.md`

No metrics, plots, raw outputs, API calls, downloads, model training, or
`original_repo/` modifications were produced in this phase.

### Known Issues

- Real metric computation remains blocked until canonical ground truth,
  candidates, catalog, and interactions are available for Beauty and Sports.
- A future reconstruction from raw data or related preprocessing sources would
  need explicit equivalence checks before being described as an exact
  reproduction.

### Next Step

- Continue only with the next explicitly requested phase.

## 2026-06-29 - Technical Report

### Changes

- Added `report/technical_report.md` as an exam-ready technical report for the
  readiness-aware reproducibility project.
- The report describes the original paper scope, project objectives,
  repository and pipeline design, available artifacts, implemented components,
  current diagnostic outputs, reproducibility blockers, limitations, future
  work, and command/evidence appendices.
- The report explicitly states that real HR@5, NDCG@5, Catalog Coverage@5,
  hallucination, MostPop, position-bias, history-length, and cost metrics are
  blocked for the current Beauty/Sports configs because canonical ground truth,
  catalog, candidates, and interactions remain unavailable.

### Commands Run

- `Get-Content -Raw PROJECT_SPEC.md`
- `Get-Content -Raw AGENTS.md`
- `Get-Content -Raw CODEX_PROMPTS.md`
- `Get-Content -Raw README.md`
- `Get-Content -Raw report/codebase_map.md`
- `Get-Content -Raw report/dataset_audit.md`
- `Get-Content -Raw report/results_analysis.md`
- `Get-Content -Raw report/quality_review.md`
- `Get-Content -Raw report/original_artifact_map.md`
- `Get-Content -Raw report/reproducibility_checklist.md`
- `Get-Content -Raw report/notes.md`
- `Get-Content -Raw configs/beauty.yaml`
- `Get-Content -Raw configs/sports.yaml`
- `rg --files -g "!*original_repo/**"`
- `Get-ChildItem -Recurse -File results`
- `Get-Content -TotalCount 8 results/tables/*_summary.csv`
- `python -m pytest tests/`

### Outputs Generated

- `report/technical_report.md`

No metrics, plots, raw outputs, API calls, downloads, model training, or
`original_repo/` modifications were produced in this phase.

The full test suite passed after the documentation update:
`124 passed, 2 skipped`.

### Known Issues

- The technical report remains a documentation artifact for a partial,
  readiness-aware reproduction. It does not add new data or unblock the missing
  canonical Beauty/Sports files.

### Next Step

- Continue only with the next explicitly requested phase.

## 2026-06-28 - Deep Original Artifact Mapping

### Changes

- Added `report/original_artifact_map.md`, a dataset-by-dataset map of local
  and README-linked original artifacts that could unlock real Beauty and Sports
  reproduction metrics after a future conversion phase.
- Added `results/tables/original_artifact_inventory.csv` with machine-readable
  artifact roles, local existence, expected formats, conversion routes,
  confidence, and risks.

### Commands Run

- Read project docs, configs, current readiness scripts, and original repo
  scripts with `Get-Content`, `Select-String`, and `rg`.
- Inspected local original ZIP archives with Python `zipfile` without
  extracting files.

### Outputs Generated

- `report/original_artifact_map.md`
- `results/tables/original_artifact_inventory.csv`

### Known Issues

- The core original preprocessed files `sequential_data.txt`,
  `negative_samples.txt`, `datamaps.json`, and `meta.json` are referenced by
  the original README and scripts but are not present locally for Beauty or
  Sports.
- Beauty LLM outputs, Sports LLM outputs, and Beauty SOTA outputs are local and
  readable. Sports SOTA outputs are referenced by original scripts but are not
  present locally.
- No configs were changed, no canonical CSVs were created, no metrics were
  computed, no APIs were called, no downloads were performed, and
  `original_repo/` was not modified.

### Next Step

- Manually provide the README-linked preprocessed data outside
  `original_repo/`, then implement a separate conversion phase that writes
  canonical CSVs under `data/processed/{dataset}/` and updates configs only
  after schema validation.

## 2026-06-28 - External Preprocessed Dataset Availability Blocker

### Changes

- Updated `report/original_artifact_map.md` with an external artifact
  availability check documenting that the README-linked Beauty and Sports
  Dropbox preprocessed-data items are unavailable/deleted.
- Updated `results/tables/original_artifact_inventory.csv` with an
  `availability_status` column and marked the README-linked preprocessed data
  entries as `unavailable/deleted`.
- Updated `report/results_analysis.md` to explain why HR@5, NDCG@5, Catalog
  Coverage@5, hallucination, MostPop, and history preparation remain blocked.

### Commands Run

- Read project docs, configs, reports, and the artifact inventory.
- Checked `data/raw/original_artifacts/` for invalid partial files.
- `python -m pytest tests/`

### Outputs Generated

- `report/original_artifact_map.md`
- `results/tables/original_artifact_inventory.csv`
- `report/results_analysis.md`
- `report/notes.md`

### Known Issues

- The unavailable README-linked archives mean `sequential_data.txt`,
  `negative_samples.txt`, `datamaps.json`, and `meta.json` cannot currently be
  recovered from the original links.
- `ground_truth_path`, `candidates_path`, `catalog_path`, and
  `interactions_path` remain unavailable unless artifacts are obtained from the
  authors or reconstructed from valid raw data.
- `data/raw/original_artifacts/` exists but is empty; no invalid partial
  HTML/error downloads were referenced as valid artifacts.
- No downloads, API calls, synthetic datasets, canonical CSVs, metric
  computation, config changes, or `original_repo/` modifications were
  performed.

### Next Step

- Defer Prompt 18.5B until the missing preprocessed artifacts are supplied by
  the authors or a separate raw-data reconstruction phase is explicitly
  requested.

## 2026-06-28 - Final README

### Changes

- Rewrote `README.md` as an exam-ready project overview for the partial,
  readiness-aware reproducibility study of *Beyond Utility: Evaluating LLM as
  Recommender*.
- Documented the project scope, partial reproduction status, unavailable
  external preprocessed datasets, repository structure, setup commands, main
  pipeline commands, current generated outputs, blocked metrics, implemented
  extensions, reproducibility policy, tests, limitations, extension path, and
  provenance policy.

### Commands Run

- Read project docs, reports, configs, result directory contents, and current
  repository structure.

### Outputs Generated

- `README.md`
- `report/notes.md`

### Known Issues

- The README intentionally does not claim real HR@5, NDCG@5, Coverage@5,
  hallucination, MostPop, position-bias, or history-length metric values.
- Real quantitative reproduction remains blocked until canonical ground truth,
  candidates, catalog, and interaction/history inputs are recovered and
  configured.
- No code changes, API calls, downloads, training, metrics computation, or
  `original_repo/` modifications were performed.

### Next Step

- Continue with the next documentation phase only when explicitly requested.

## 2026-06-28 - Reproducibility Checklist

### Changes

- Added `report/reproducibility_checklist.md` as an exam-ready checklist for
  the partial, readiness-aware reproducibility study.
- Documented project identity, original paper scope, current reproduction
  status, dataset artifacts, metric blockers, baseline status, evidence files,
  completion requirements, and the final integrity statement.

### Commands Run

- Read project docs, reports, configs, and current `results/` contents.

### Outputs Generated

- `report/reproducibility_checklist.md`
- `report/notes.md`

### Known Issues

- The checklist intentionally does not claim full paper reproduction or real
  HR@5, NDCG@5, Coverage@5, hallucination, MostPop, position-bias, or
  history-length metric computation.
- Real quantitative reproduction remains blocked until canonical interactions,
  ground truth, candidates, and catalog metadata are recovered and configured.
- No code changes, API calls, downloads, training, metric computation, result
  files, or `original_repo/` modifications were performed.

### Next Step

- Continue with the next explicitly requested documentation phase.

## 2026-06-28 - Repository Quality Review

### Changes

- Added `report/quality_review.md` with strict review findings, fixes applied,
  deferred risks, and verification commands.
- Updated `PROJECT_SPEC.md` so dataset audit outputs point to
  `results/tables/`.
- Updated `README.md` so the current status reflects implemented modules,
  readiness diagnostics, and blocked real metrics.
- Moved MostPop summaries from `results/metrics/` to `results/tables/` in code
  and tests.
- Updated generated diagnostic CSVs to store project-relative paths rather than
  machine-local absolute paths.
- Fixed final pipeline summary path selection for the history-length phase.

### Commands Run

- `python -m compileall src/data/audit_datasets.py src/experiments/load_llm_outputs.py src/baselines/load_baseline_outputs.py src/baselines/mostpop.py src/experiments/run_pipeline.py`
- `pytest tests/test_config.py tests/test_dataset_audit.py tests/test_load_llm_outputs.py tests/test_load_baseline_outputs.py tests/test_mostpop.py tests/test_run_pipeline.py`
- `python src/data/audit_datasets.py --config configs/beauty.yaml`
- `python src/data/audit_datasets.py --config configs/sports.yaml`
- `python src/experiments/load_llm_outputs.py --config configs/beauty.yaml --log-unmatched-limit 0`
- `python src/experiments/load_llm_outputs.py --config configs/sports.yaml --log-unmatched-limit 0`
- `python src/baselines/load_baseline_outputs.py --config configs/beauty.yaml`
- `python src/baselines/load_baseline_outputs.py --config configs/sports.yaml`
- `python src/experiments/run_pipeline.py --config configs/beauty.yaml`
- `python src/experiments/run_pipeline.py --config configs/sports.yaml`
- `python src/experiments/make_plots.py --config configs/beauty.yaml`
- `python src/experiments/make_plots.py --config configs/sports.yaml`
- `pytest tests/`

### Outputs Generated

- `report/quality_review.md`
- Refreshed `results/tables/*_dataset_audit.csv`
- Refreshed `results/tables/*_llm_parsing_summary.csv`
- Refreshed `results/tables/*_baseline_outputs_summary.csv`
- Refreshed `results/tables/*_final_summary.csv`
- Refreshed `results/tables/*_plot_summary.csv`
- Refreshed readiness plots under `results/plots/`

### Known Issues

- Summary status values remain phase-specific and should be normalized in a
  future cleanup if final reporting needs one shared status vocabulary.
- `results/raw_outputs/beauty_baseline_outputs_normalized.csv` is a tracked
  large generated file, about 94 MB; storage policy should be decided before
  final packaging.
- `report/results_analysis.md` still summarizes the final safe pipeline section
  for the latest pipeline run, currently Sports, while per-dataset CSVs exist
  for both Beauty and Sports.
- No fake metric values, API calls, downloads, model training, new metric
  computation, or `original_repo/` modifications were performed.

### Next Step

- Continue only with the next explicitly requested phase.

## 2026-06-27 - MostPop Baseline

### Changes

- Added `src/baselines/mostpop.py` with a root-runnable CLI:
  `python src/baselines/mostpop.py --config configs/beauty.yaml`.
- The baseline loads configured canonical interactions, computes global item
  popularity from `split == train` rows only, and recommends popular unseen
  items for each `split == test` user.
- Recommendations are written to `results/raw_outputs/<dataset>_mostpop_recommendations.csv`.
- Successful MostPop run summaries are written to
  `results/tables/<dataset>_mostpop_summary.csv`.
- Added synthetic tests verifying train-only popularity, seen-item exclusion,
  output paths, summary fields, and clear failure when `interactions_path` is
  not configured.

### Commands Run

- `Get-Content -Path PROJECT_SPEC.md | Out-Null`
- `Get-Content -Path AGENTS.md | Out-Null`
- `python -m compileall src\baselines\mostpop.py`
- `pytest tests\test_mostpop.py`
- `pytest tests\`
- `python src\baselines\mostpop.py --config configs\beauty.yaml`

### Outputs Generated

- No MostPop recommendation or summary CSVs were generated for the repository
  configs because `paths.interactions_path` is currently null.

### Known Issues

- Current Beauty and Sports configs do not yet point to canonical interactions
  files, so the CLI fails clearly before computing popularity or writing
  outputs.
- The baseline does not use test interactions to compute popularity.

### Next Step

- Configure audited canonical interactions before running the MostPop baseline
  on real project data.

## 2026-06-27 - Traditional Baseline Output Loader

### Changes

- Added `src/baselines/load_baseline_outputs.py` with a root-runnable CLI:
  `python src/baselines/load_baseline_outputs.py --config configs/beauty.yaml`.
- The loader uses YAML configs through `src/utils/config.py`, reads configured
  cached traditional/SOTA recommendation outputs, and never trains models.
- Found cached baseline outputs are normalized under `results/raw_outputs/`.
- Baseline discovery and diagnostic summaries are written under
  `results/tables/`, with explicit `found`, `missing`, and `not_configured`
  statuses.
- Added synthetic tests for loading a baseline CSV, preserving recommendation
  order, writing diagnostics when no files are configured/found, avoiding real
  `original_repo/` dependency, and not requiring `interactions_path`.

### Commands Run

- `Get-Content -Path PROJECT_SPEC.md | Out-Null`
- `Get-Content -Path AGENTS.md | Out-Null`
- `Get-Content -Path CODEX_PROMPTS.md | Out-Null`
- `Get-Content -Path report\codebase_map.md | Out-Null`
- `Get-Content -Path report\dataset_audit.md | Out-Null`
- `Get-Content -Path configs\beauty.yaml | Out-Null`
- `Get-Content -Path configs\sports.yaml | Out-Null`
- `Get-Content -Path src\utils\config.py | Out-Null`
- `Get-Content -Path src\data\schemas.py | Out-Null`
- `Get-Content -Path src\data\load_data.py | Out-Null`
- `Get-Content -Path src\baselines\mostpop.py | Out-Null`
- `Get-Content -Path tests\test_mostpop.py | Out-Null`
- `Get-Content -Path report\notes.md | Out-Null`
- Read-only inspection of `original_repo\EvaLLMasRecommender\sota_results\beauty_his10.zip`
- `python -m compileall src\baselines\load_baseline_outputs.py`
- `pytest tests\test_load_baseline_outputs.py`
- `pytest tests\`
- `python src\baselines\load_baseline_outputs.py --config configs\beauty.yaml`
- `python src\baselines\load_baseline_outputs.py --config configs\sports.yaml`

### Outputs Generated

- `results/raw_outputs/beauty_baseline_outputs_normalized.csv`
- `results/tables/beauty_baseline_outputs_summary.csv`
- `results/tables/sports_baseline_outputs_summary.csv`

The Beauty summary reports one configured cached `BPRMF` output found in
`beauty_his10.zip`. The Sports summary reports `not_configured` for configured
baseline paths and `missing` for original-repo search.

### Known Issues

- Sports traditional baseline outputs are still unavailable in the current
  checkout.
- MostPop was not run for Beauty or Sports because `interactions_path` remains
  null in both configs.
- No HR@K, NDCG@K, hallucination, coverage, position-bias, history-length,
  cost, or final experiment metrics were computed.
- No API calls, downloads, model training, or `original_repo/` modifications
  were performed.

### Next Step

- Continue only with the next explicitly requested phase.

## 2026-06-27 - Reproduction Runner

### Changes

- Added `src/experiments/run_reproduction.py` with a root-runnable CLI:
  `python src/experiments/run_reproduction.py --config configs/beauty.yaml`.
- The runner loads normalized LLM, traditional baseline, and MostPop outputs
  from `results/raw_outputs/` when available, validates canonical ground truth,
  and computes HR@5 and NDCG@5 only when required inputs exist.
- LLM hallucination metrics are computed only when configured candidates or
  catalog files exist and validate successfully.
- Reproduction metrics are written to `results/metrics/`; diagnostics and
  summaries are written to `results/tables/`; Markdown analysis is written to
  `report/results_analysis.md`.
- Added synthetic tests for successful utility/hallucination computation and
  missing-input diagnostics.

### Commands Run

- `python -m compileall src\experiments\run_reproduction.py`
- `pytest tests\test_run_reproduction.py`
- `pytest tests\`
- `python src\experiments\run_reproduction.py --config configs\beauty.yaml`

### Outputs Generated

- `results/metrics/beauty_reproduction_metrics.csv`
- `results/tables/beauty_reproduction_summary.csv`
- `report/results_analysis.md`

The Beauty run wrote a diagnostic summary and no metric values because
`paths.ground_truth_path` remains null in `configs/beauty.yaml`. Existing
normalized Beauty LLM and baseline outputs were detected; MostPop remains
missing because `interactions_path` is not configured.

### Known Issues

- No HR@5, NDCG@5, or hallucination values were computed for the real Beauty
  config because canonical ground truth is not configured.
- Hallucination evaluation for real configs will remain skipped until valid
  `paths.candidates_path` or `paths.catalog_path` is configured.
- No API calls, downloads, model training, fake outputs, or `original_repo/`
  modifications were performed.

### Next Step

- Configure canonical ground truth, and optionally catalog or candidates, before
  running reproduction metrics on real project data.

## 2026-06-27 - Prompt 12 Reproduction Pipeline Correction

### Changes

- Updated `src/experiments/run_reproduction.py` so diagnostic/readiness-only
  runs exit with status code 0.
- Missing or null `ground_truth_path` is reported in the readiness summary
  instead of treated as a fatal script error.
- Missing or null `candidates_path` and `catalog_path` are reported as
  hallucination readiness blockers when LLM outputs are present.
- Empty metric CSV files are no longer written when no real metric rows are
  computed; stale empty reproduction metric files are removed.
- Recommendation source validation failures are reported as blocked upstream
  readiness rows, allowing Beauty and Sports summaries to be produced.
- Updated `tests/test_run_reproduction.py` for exit-success diagnostic runs,
  no-empty-metrics behavior, and preserved synthetic metric computation.

### Commands Run

- `python -m compileall src\experiments\run_reproduction.py`
- `pytest tests\test_run_reproduction.py`
- `pytest tests\`
- `python src\experiments\run_reproduction.py --config configs\beauty.yaml`
- `python src\experiments\run_reproduction.py --config configs\sports.yaml`

### Outputs Generated

- `results/tables/beauty_reproduction_summary.csv`
- `results/tables/sports_reproduction_summary.csv`
- `report/results_analysis.md`

No reproduction metrics CSV was written for Beauty or Sports because the current
configs still do not provide canonical ground truth. The stale empty Beauty
metrics CSV from the previous diagnostic run was removed.

### Known Issues

- Beauty utility metrics remain blocked until `paths.ground_truth_path` is
  configured.
- Beauty hallucination metrics remain blocked until `paths.candidates_path` or
  `paths.catalog_path` is configured.
- Sports utility metrics remain blocked until ground truth and valid normalized
  recommendation outputs are available.
- No API calls, downloads, model training, fake outputs, or `original_repo/`
  modifications were performed.

### Next Step

- Complete only the next explicitly requested phase.

## 2026-06-27 - Sports LLM Readiness Correction

### Changes

- Updated `src/experiments/run_reproduction.py` so normalized recommendation
  files are usable when at least one row has a non-empty
  `recommended_item_id` or `recommended_item_text`.
- Rows with both recommendation fields empty are skipped and counted in the
  readiness message instead of blocking the whole file.
- Sports LLM outputs now report `ready_with_warnings` with one skipped empty
  row, while missing ground truth and missing catalog/candidates remain
  readiness blockers.
- Added focused tests for text-only LLM recommendations, mixed valid/empty
  rows, and completely empty recommendation files.

### Commands Run

- `python -m compileall src\experiments\run_reproduction.py`
- `pytest tests\test_run_reproduction.py`
- `pytest tests\`
- `python src\experiments\run_reproduction.py --config configs\sports.yaml`
- `python src\experiments\run_reproduction.py --config configs\beauty.yaml`

### Outputs Generated

- `results/tables/sports_reproduction_summary.csv`
- `results/tables/beauty_reproduction_summary.csv`
- `report/results_analysis.md`

No reproduction metrics CSV was written because canonical ground truth is still
not configured for either dataset.

### Known Issues

- Sports HR@5/NDCG@5 remain blocked until `paths.ground_truth_path` is
  configured.
- Sports hallucination remains blocked until valid `paths.candidates_path` or
  `paths.catalog_path` is configured.
- Sports traditional baseline and MostPop outputs remain unavailable/missing in
  the current configured readiness state.
- No API calls, downloads, model training, fake outputs, or `original_repo/`
  modifications were performed.

### Next Step

- Continue only with the next explicitly requested phase.

## 2026-06-27 - Catalog Coverage Extension

### Changes

- Added `src/evaluation/coverage.py` with `catalog_coverage_at_k` and
  `coverage_by_model`.
- The metric supports item IDs or string values, multiple users, top-K cutoffs,
  duplicate recommendation deduplication, empty recommendation inputs, and the
  documented empty-catalog policy of returning `0.0`.
- Added `src/experiments/run_extension_coverage.py`, a readiness-aware runner
  that computes real Catalog Coverage@5 only when `catalog_path` is configured
  to a readable catalog with `item_id` values and normalized recommendations
  contain comparable `recommended_item_id` values.
- Added `tests/test_coverage.py` with synthetic metric and runner tests.
- Updated `report/results_analysis.md` with a coverage-extension explanation
  and current blocker status.

### Commands Run

- `python -m compileall src\evaluation\coverage.py src\experiments\run_extension_coverage.py`
- `pytest tests\test_coverage.py`
- `pytest tests\`
- `python src\experiments\run_extension_coverage.py --config configs\beauty.yaml`
- `python src\experiments\run_extension_coverage.py --config configs\sports.yaml`

### Outputs Generated

- `results/tables/beauty_coverage_summary.csv`
- `results/tables/sports_coverage_summary.csv`
- `report/results_analysis.md`

No `results/metrics/*_coverage_metrics.csv` file was written because real
Coverage@5 is blocked for both current configs: `paths.catalog_path` is null.

### Known Issues

- Beauty and Sports Catalog Coverage@5 remain blocked until a real canonical
  catalog is configured.
- LLM recommendation text is not converted into item IDs by the coverage runner;
  normalized recommendations must contain valid `recommended_item_id` values
  to compute catalog coverage against an `item_id` catalog.
- No HR@5, NDCG@5, hallucination, position bias, history length, cost metrics,
  API calls, downloads, model training, fake outputs, or `original_repo/`
  modifications were performed.

### Next Step

- Continue only with the next explicitly requested phase.

## 2026-06-27 - Candidate Position Bias Utilities

### Changes

- Added `src/prompts/candidate_ordering.py` with deterministic candidate
  ordering variants: `original`, `random`, `reversed`,
  `positive_at_start`, and `positive_at_end`.
- Added `src/evaluation/position_bias.py` with selected candidate-position
  distribution and per-model/run summary helpers.
- Added `src/experiments/run_position_bias.py`, a readiness-aware runner that
  writes diagnostics when candidate lists or normalized recommendations are not
  available and writes metrics only when real matched inputs exist.
- Added `tests/test_position_bias.py` with synthetic ordering, metric, missing
  candidate, duplicate recommendation, grouping, and blocked-runner tests.
- Updated `report/results_analysis.md` with the current position-bias readiness
  status.

### Commands Run

- `python -m pytest tests/test_position_bias.py`
- `python -m pytest tests`
- `python src/experiments/run_position_bias.py --config configs/beauty.yaml`
- `python src/experiments/run_position_bias.py --config configs/sports.yaml`

### Outputs Generated

- `results/tables/beauty_position_bias_summary.csv`
- `results/tables/sports_position_bias_summary.csv`
- `report/results_analysis.md`

No `results/metrics/*_position_bias_metrics.csv` file was written because real
position-bias metrics are blocked for both current configs:
`paths.candidates_path` is null.

### Known Issues

- Beauty and Sports position-bias metrics remain blocked until real canonical
  candidate files are configured.
- Candidate ordering utilities only transform supplied candidates; they do not
  call APIs or generate LLM outputs.
- No HR@5, NDCG@5, coverage, hallucination, history length, cost metrics, API
  calls, downloads, model training, fake outputs, or `original_repo/`
  modifications were performed.

### Next Step

- Continue only with the next explicitly requested phase.

## 2026-06-27 - History Length Sensitivity Utilities

### Changes

- Added `src/prompts/history_length.py` with deterministic history truncation
  and prompt-input variant helpers for history lengths `0`, `1`, `2`, `5`,
  `10`, and `full`.
- Added `src/experiments/run_history_length.py`, a readiness-aware runner that
  prepares prompt-input variants only when configured history/interactions data
  is available and never calls APIs.
- Added `tests/test_history_length.py` with synthetic history truncation,
  ordering, shorter-history, empty-history, reproducibility, and runner tests.
- Updated `report/results_analysis.md` with the current history-length
  readiness status.

### Commands Run

- `python -m pytest tests/test_history_length.py`
- `python -m compileall src/prompts/history_length.py src/experiments/run_history_length.py`
- `python -m pytest tests`
- `python src/experiments/run_history_length.py --config configs/beauty.yaml`
- `python src/experiments/run_history_length.py --config configs/sports.yaml`

### Outputs Generated

- `results/tables/beauty_history_length_summary.csv`
- `results/tables/sports_history_length_summary.csv`
- `report/results_analysis.md`

No `results/raw_outputs/*_history_length_prompts.csv` file was written for the
current Beauty/Sports configs because neither `history_path` nor
`interactions_path` is configured. No
`results/metrics/*_history_length_metrics.csv` file was written.

### Known Issues

- Beauty and Sports history-length prompt preparation remains blocked until
  real canonical history/interactions files are configured.
- The runner prepares prompt-input variants only; it does not call APIs,
  generate model outputs, or compute HR@5/NDCG@5.
- No coverage, position bias, cost, final pipeline metrics, API calls,
  downloads, fake histories, fake outputs, or `original_repo/` modifications
  were performed.

### Next Step

- Continue only with the next explicitly requested phase.

## 2026-06-27 - Cost and Token Metric Utilities

### Changes

- Added `src/evaluation/cost_metrics.py` with safe utilities for recorded
  `prompt_tokens`, `completion_tokens`, `total_tokens`, `latency_seconds`, and
  explicit pricing assumptions.
- Added `tests/test_cost_metrics.py` with synthetic token, latency, pricing,
  missing-data, aggregation, and no-external-API dependency tests.
- Added `configs/cost.yaml` as a placeholder pricing schema with null prices
  and comments requiring manual price entry before real cost estimates.

### Commands Run

- `python -m pytest tests/test_cost_metrics.py`
- `python -m compileall src/evaluation/cost_metrics.py`
- `python -m pytest tests`

### Outputs Generated

- No experiment result CSVs were generated in this phase.

### Known Issues

- Real cost estimates remain unavailable until token counts, latency logs, and
  manually filled pricing assumptions are provided.
- `configs/cost.yaml` intentionally contains no real-world provider prices.
- No HR@5, NDCG@5, hallucination, coverage, position bias, history length,
  final pipeline metrics, API calls, downloads, fake prices, fake token counts,
  fake latency values, or `original_repo/` modifications were performed.

### Next Step

- Continue only with the next explicitly requested phase.

## 2026-06-27 - Final Safe Pipeline Orchestrator

### Changes

- Added `src/experiments/run_pipeline.py`, a root-runnable safe orchestrator
  for smoke checks, dataset audit verification, cached LLM output readiness,
  cached baseline readiness, MostPop gating, reproduction diagnostics, coverage,
  position bias, history length, and cost/token availability.
- Added `tests/test_run_pipeline.py` with synthetic temporary-project tests for
  final summaries, blocked diagnostics, CLI success, final-metrics gating,
  optional extension skips, API-disabled behavior, invalid configs, and
  `original_repo/` preservation.
- Updated `report/results_analysis.md` with the latest final pipeline summary.

### Commands Run

- `python -m pytest tests/test_run_pipeline.py`
- `python -m compileall src/experiments/run_pipeline.py`
- `python -m pytest tests`
- `python src/experiments/run_pipeline.py --config configs/beauty.yaml`
- `python src/experiments/run_pipeline.py --config configs/sports.yaml`

### Outputs Generated

- `results/tables/beauty_final_summary.csv`
- `results/tables/sports_final_summary.csv`
- `report/results_analysis.md`

No `results/metrics/*_final_results.csv` file was written because no
underlying phase computed real metric rows with the current configs.

### Known Issues

- Beauty and Sports utility metrics remain blocked until canonical
  `ground_truth_path` values are configured.
- Hallucination, coverage, and position bias remain blocked until canonical
  `catalog_path` and/or `candidates_path` values are configured.
- MostPop and history length remain blocked until canonical interaction/history
  files are configured.
- Cost estimates remain not configured because no token/latency records are
  configured and `configs/cost.yaml` prices are null.
- No plot generation, final README, final report, slide outline, API calls,
  downloads, model training, fake metrics, fake inputs, or `original_repo/`
  modifications were performed.

### Next Step

- Continue only with the next explicitly requested phase.

## 2026-06-28 - Plot Generation

### Changes

- Added `src/experiments/make_plots.py`, a root-runnable plot generator that
  reads existing CSVs only, writes plot summaries, and skips unavailable metric
  plots without inventing values.
- Added `tests/test_make_plots.py` with synthetic CSV tests for metric plot
  generation, missing metrics, empty metrics directories, deterministic output
  names, and readiness-only summaries.
- Updated `report/results_analysis.md` with a plot-generation section for
  Beauty and Sports.

### Commands Run

- `python -m compileall src/experiments/make_plots.py`
- `pytest tests/test_make_plots.py`
- `python src/experiments/make_plots.py --config configs/beauty.yaml`
- `python src/experiments/make_plots.py --config configs/sports.yaml`
- `pytest tests/`

### Outputs Generated

- `results/plots/beauty_readiness_status_counts.png`
- `results/plots/sports_readiness_status_counts.png`
- `results/tables/beauty_plot_summary.csv`
- `results/tables/sports_plot_summary.csv`
- `report/results_analysis.md`

### Known Issues

- HR@5, NDCG@5, Catalog Coverage@5, hallucination, position-bias, and
  history-length metric plots were skipped because `results/metrics/` contains
  no real metric CSV rows for the current Beauty/Sports configs.
- The generated plots are readiness/status-count diagnostics from existing
  final summary CSVs, not metric plots.
- No fake metric values, API calls, downloads, model training, new metric
  computation, or `original_repo/` modifications were performed.

### Next Step

- Continue only with the next explicitly requested phase.

## 2026-06-29 - Plan B Prompt B0: Reconstruction Feasibility Plan

### Changes

- Created `report/plan_b_reconstruction_plan.md` to start Plan B as a separate
  reconstruction/extension track.
- Framed MovieLens-1M as the recommended future dataset for an end-to-end
  pipeline with real computable metrics.
- Kept Plan A unchanged as a closed readiness-aware reproducibility audit for
  Beauty and Sports.

### Commands Run

- `Get-Content` was used to read the requested Plan A documentation, configs,
  and implementation files before writing the plan.

### Outputs Generated

- `report/plan_b_reconstruction_plan.md`

No metrics, ML-1M processed data, configs, code modules, API calls, downloads,
training runs, or synthetic results were generated.

### Known Issues

- Plan B cannot run until the user manually places real MovieLens-1M raw files
  under `data/raw/ml1m/`.
- Beauty and Sports remain blocked exactly as documented in Plan A because the
  original preprocessed artifacts are still unavailable.

### Next Step

- After the raw ML-1M files are placed locally, run B1 to prepare canonical
  `data/processed/ml1m/` files and `configs/ml1m.yaml`.

## 2026-06-29 - Plan B B1: MovieLens-1M Canonical Dataset Preparation

### Changes

- Added `src/data/prepare_ml1m.py`, a root-runnable preprocessing script for
  the separate Plan B MovieLens-1M reconstruction track.
- Added `configs/ml1m.yaml` with project-relative canonical paths, API calls
  disabled, deterministic seed `42`, and `99` negative samples per user.
- Added synthetic tests for chronological leave-one-out splitting, eligible
  user filtering, candidate construction, deterministic sampling, and missing
  raw-file diagnostics.
- Updated `README.md` with a short Plan B B1 section explaining manual raw data
  placement, no automatic download, generated canonical files, and the
  reconstruction-only framing.

### Commands Run

- `python -m compileall src/data/prepare_ml1m.py`
- `python -m pytest tests/test_prepare_ml1m.py tests/test_config.py`
- `python src/data/prepare_ml1m.py --config configs/ml1m.yaml`

### Outputs Generated

- `data/processed/ml1m/interactions.csv`
- `data/processed/ml1m/ground_truth.csv`
- `data/processed/ml1m/candidates.csv`
- `data/processed/ml1m/catalog.csv`
- `results/tables/ml1m_dataset_preparation_summary.csv`

The local raw ML-1M files were found under `data/raw/ml1m/`. The preparation
summary reports:

- raw ratings: `1000209`
- eligible users: `6040`
- train interactions: `994169`
- ground-truth rows: `6040`
- candidate rows: `604000`
- catalog rows: `3883`
- negative samples per user: `99`

### Known Issues

- No Plan B recommendation outputs or metric rows exist yet. HR@5, NDCG@5, and
  Coverage@5 must wait for a later phase that generates real recommendations.
- The generated ML-1M files are reconstruction artifacts, not exact
  Beauty/Sports reproduction artifacts.
- No API calls, downloads, heavy training, fake metrics, or `original_repo/`
  modifications were performed.

### Next Step

- Run B2 to add a deterministic Random baseline over the canonical ML-1M
  candidate sets, then run real metrics only after recommendations exist.

## 2026-06-29 - Plan B B2: Deterministic Random Baseline

### Changes

- Added `src/baselines/random_baseline.py`, a deterministic Random baseline
  for the separate Plan B MovieLens-1M reconstruction track.
- The baseline reads canonical ML-1M candidates and ground truth, assigns
  deterministic pseudo-random scores from `configs/ml1m.yaml`, ranks only
  candidate-set items, and computes HR@5 and NDCG@5 through the existing
  utility metric implementation.
- Added `tests/test_random_baseline.py` with synthetic fixtures covering
  deterministic ranking, seed changes, candidate-only outputs, consecutive
  ranks, controlled HR/NDCG values, missing-input diagnostics, and candidate
  metadata preservation.
- Updated `README.md` with the Plan B B2 Random baseline command and framing.

### Commands Run

- `python -m compileall src/baselines/random_baseline.py`
- `python -m pytest tests/test_random_baseline.py`
- `python src/baselines/random_baseline.py --config configs/ml1m.yaml`

### Outputs Generated

- `results/raw_outputs/ml1m_random_recommendations.csv`
- `results/metrics/ml1m_random_metrics.csv`
- `results/tables/ml1m_random_baseline_summary.csv`

The ML-1M candidate and ground-truth files were found. The Random baseline
summary reports:

- users evaluated: `6040`
- candidate rows: `604000`
- recommendation rows: `604000`
- metrics written: `True`
- seed: `42`

The real metric CSV reports:

- HR@5 (`hit_rate`, k=5): `0.05149006622516556`
- NDCG@5 (`ndcg`, k=5): `0.029371525149611694`

### Known Issues

- These are Plan B MovieLens-1M reconstruction metrics, not exact Plan A
  Beauty/Sports reproduction metrics.
- No MostPop Plan B metrics, coverage metrics, plots, or report integration
  were generated in this phase.
- No API calls, downloads, heavy training, fake metrics, fake recommendations,
  or `original_repo/` modifications were performed.

### Next Step

- Run B3 to add/run the next Plan B metric phase, likely MostPop plus Random
  reconstruction evaluation and plots from real metric rows.

## 2026-06-29 - Plan B B3: MovieLens-1M MostPop Baseline And Comparison

### Changes

- Added `src/baselines/ml1m_mostpop_baseline.py`, a candidate-set MostPop
  baseline for the separate Plan B MovieLens-1M reconstruction track.
- MostPop computes popularity from `data/processed/ml1m/interactions.csv`
  training/history rows only, reranks each user's canonical candidate set, and
  computes HR@5 and NDCG@5 through the existing utility metric implementation.
- Added `tests/test_ml1m_mostpop_baseline.py` with synthetic fixtures covering
  training-only popularity, held-out item leakage prevention, popularity
  ranking, deterministic item-id tie-breaking, candidate-only outputs,
  consecutive ranks, controlled HR/NDCG values, missing-input diagnostics, and
  comparison-table behavior.
- Updated `README.md` with the Plan B B3 MostPop command and framing.

### Commands Run

- `python -m compileall src/baselines/ml1m_mostpop_baseline.py`
- `python -m pytest tests/test_ml1m_mostpop_baseline.py`
- `python src/baselines/ml1m_mostpop_baseline.py --config configs/ml1m.yaml`

### Outputs Generated

- `results/raw_outputs/ml1m_mostpop_recommendations.csv`
- `results/metrics/ml1m_mostpop_metrics.csv`
- `results/tables/ml1m_mostpop_baseline_summary.csv`
- `results/tables/ml1m_baseline_comparison_summary.csv`

The ML-1M interactions, candidate, and ground-truth files were found. The
MostPop baseline summary reports:

- users evaluated: `6040`
- train interactions: `994169`
- candidate rows: `604000`
- recommendation rows: `604000`
- metrics written: `True`

The real MostPop metric CSV reports:

- HR@5 (`hit_rate`, k=5): `0.320364238410596`
- NDCG@5 (`ndcg`, k=5): `0.21028063220934365`

The baseline comparison table reports real values for:

- Random HR@5: `0.0514900662251655`
- Random NDCG@5: `0.0293715251496116`
- MostPop HR@5: `0.320364238410596`
- MostPop NDCG@5: `0.2102806322093436`

### Known Issues

- These are Plan B MovieLens-1M reconstruction metrics, not exact Plan A
  Beauty/Sports reproduction metrics.
- Full recommendation outputs under `results/raw_outputs/` are generated
  artifacts and may be too large for normal Git commits.
- No coverage metrics, plots, API calls, downloads, heavy training, fake
  metrics, fake recommendations, or `original_repo/` modifications were
  performed.

### Next Step

- Run the next Plan B phase to integrate the real Random and MostPop metric
  rows into reconstruction plots and documentation.

## 2026-06-29 - Plan B B4: MovieLens-1M ContentBM25 Baseline

### Changes

- Added `src/baselines/content_bm25_baseline.py`, a lightweight in-repository
  `ContentBM25` baseline for the separate Plan B MovieLens-1M reconstruction
  track.
- `ContentBM25` builds user term profiles only from
  `data/processed/ml1m/interactions.csv` training/history rows and uses ML-1M
  catalog title, genre, and normalized text fields as item content.
- Added `tests/test_content_bm25_baseline.py` with synthetic fixtures covering
  training-only profiles, held-out item leakage prevention, BM25 scoring,
  deterministic tie-breaking, candidate-only outputs, consecutive ranks,
  controlled HR/NDCG values, missing-input diagnostics, and comparison-table
  behavior.
- Updated `README.md` with the Plan B B4 content-baseline command and framing.

### Commands Run

- `python -m compileall src/baselines/content_bm25_baseline.py`
- `python -m pytest tests/test_content_bm25_baseline.py`
- `python src/baselines/content_bm25_baseline.py --config configs/ml1m.yaml`
- `python -m pytest tests/`

### Outputs Generated

- `results/raw_outputs/ml1m_content_bm25_recommendations.csv`
- `results/metrics/ml1m_content_bm25_metrics.csv`
- `results/tables/ml1m_content_bm25_baseline_summary.csv`
- `results/tables/ml1m_baseline_comparison_summary.csv`

The ML-1M interactions, candidate, ground-truth, and catalog files were found.
The ContentBM25 baseline summary reports:

- users evaluated: `6040`
- train interactions: `994169`
- candidate rows: `604000`
- recommendation rows: `604000`
- catalog rows: `3883`
- metrics written: `True`

The real ContentBM25 metric CSV reports:

- HR@5 (`hit_rate`, k=5): `0.2508278145695364`
- NDCG@5 (`ndcg`, k=5): `0.1647567964031986`

The refreshed baseline comparison table reports real values for:

- Random HR@5: `0.0514900662251655`
- Random NDCG@5: `0.0293715251496116`
- MostPop HR@5: `0.320364238410596`
- MostPop NDCG@5: `0.2102806322093436`
- ContentBM25 HR@5: `0.2508278145695364`
- ContentBM25 NDCG@5: `0.1647567964031986`

The full test suite result was `149 passed, 2 skipped`.

### Known Issues

- These are Plan B MovieLens-1M reconstruction metrics, not exact Plan A
  Beauty/Sports reproduction metrics.
- Full recommendation outputs under `results/raw_outputs/` are generated
  artifacts and may be too large for normal Git commits.
- No coverage metrics, plots, API calls, downloads, heavy training, fake
  metrics, fake recommendations, or `original_repo/` modifications were
  performed.

### Next Step

- Run B5 to generate Plan B reconstruction plots and update documentation from
  the real Random, MostPop, and ContentBM25 metric rows.

## 2026-06-29 - Plan B B5: MovieLens-1M Reconstruction Runner And Plots

### Changes

- Added `src/experiments/run_ml1m_reconstruction.py`, a safe Plan B ML-1M
  reconstruction runner.
- The runner verifies canonical ML-1M processed inputs, skips preprocessing
  when they already exist, refreshes Random, MostPop, and ContentBM25 metrics
  through the B2-B4 modules, rebuilds the comparison table from real metric
  CSVs, and generates plots from real comparison rows only.
- Added `tests/test_run_ml1m_reconstruction.py` with synthetic fixtures covering
  real-row comparison loading, missing metric diagnostics, graceful plot skips,
  missing-input summaries, no raw MovieLens dependency for unit tests, and
  preservation of `original_repo/` and Plan A config files.
- Updated `README.md` with the B5 runner command and output locations.

### Commands Run

- `python -m compileall src/experiments/run_ml1m_reconstruction.py`
- `python -m pytest tests/test_run_ml1m_reconstruction.py`
- `python -m pytest tests/`
- `python src/experiments/run_ml1m_reconstruction.py --config configs/ml1m.yaml`

### Outputs Generated

- `results/tables/ml1m_reconstruction_pipeline_summary.csv`
- `results/tables/ml1m_baseline_comparison_summary.csv`
- `results/plots/ml1m_baseline_hr_at_5.png`
- `results/plots/ml1m_baseline_ndcg_at_5.png`
- `results/plots/ml1m_baseline_comparison.png`

The runner found all canonical ML-1M processed inputs and skipped
preprocessing. It refreshed the existing Random, MostPop, and ContentBM25 metric
files from real recommendations and ground truth:

- `results/metrics/ml1m_random_metrics.csv`
- `results/metrics/ml1m_mostpop_metrics.csv`
- `results/metrics/ml1m_content_bm25_metrics.csv`

The final comparison table reports:

- Random HR@5: `0.0514900662251655`
- Random NDCG@5: `0.0293715251496116`
- MostPop HR@5: `0.320364238410596`
- MostPop NDCG@5: `0.2102806322093436`
- ContentBM25 HR@5: `0.2508278145695364`
- ContentBM25 NDCG@5: `0.1647567964031986`

The full test suite result before the runner execution was
`154 passed, 2 skipped`.

### Known Issues

- These remain Plan B MovieLens-1M reconstruction metrics, not exact Plan A
  Beauty/Sports reproduction metrics.
- Full recommendation outputs under `results/raw_outputs/` are generated
  artifacts and remain excluded from normal Git tracking.
- No API calls, downloads, heavy training, fake metrics, fake recommendations,
  or `original_repo/` modifications were performed.

### Next Step

- Run B6 to update the final documentation/report sections with Plan B
  reconstruction results and limitations.

## 2026-06-29 - Plan B B6: Final Documentation Consolidation

### Changes

- Updated `README.md` with the final Plan B metric table, output locations,
  and the explicit distinction between Plan A Beauty/Sports readiness and Plan
  B MovieLens-1M reconstruction.
- Updated `PROJECT_SPEC.md` so the final project scope reflects the two-track
  design: Plan A readiness-aware audit and Plan B reconstruction/extension.
- Updated `report/technical_report.md` with a dedicated Plan B results section,
  interpretation, limitations, and oral-exam framing.
- Rewrote `report/results_analysis.md` as a consolidated final analysis instead
  of a single auto-generated pipeline dump.
- Updated `report/reproducibility_checklist.md` with Plan B commands, expected
  metric outputs, expected plot outputs, and the generated-artifact policy for
  large recommendation CSVs.
- Updated `report/plan_b_reconstruction_plan.md` with B1-B6 completion status.
- Updated `CODEX_PROMPTS.md` with concise B0-B6 prompt records.

### Commands To Run

- `python -m pytest tests/`

### Documentation Position

- Plan A remains closed as a readiness-aware reproducibility audit.
- Beauty/Sports full numerical reproduction remains blocked by missing
  original preprocessed artifacts.
- Beauty/Sports `metric_rows=0` remains the correct honest outcome.
- Plan B results are real MovieLens-1M reconstruction metrics, not exact
  reproduction of the original Beauty/Sports experiments.

### Known Issues

- No code behavior was changed in B6.
- No APIs, downloads, training, fake metrics, empty metric CSVs, or
  `original_repo/` modifications were performed.

### Next Step

- Commit the stable Plan B reconstruction documentation and generated evidence,
  excluding ignored large recommendation CSVs.

## 2026-06-29 - Plan C C0: Full-Paper Replication Matrix

### Changes

- Started Plan C as a separate full-paper replication planning track.
- Created `report/plan_c_full_replication_matrix.md` with a dataset, task,
  prompting strategy, LLM model, traditional model, metric, and artifact matrix.
- Created machine-readable Plan C inventories under `results/tables/`.
- Kept Plan A and Plan B unchanged; Plan C does not claim exact reproduction.

### Commands Run

- Read existing project documentation, configs, and Plan A/Plan B reports.
- Inspected `original_repo/EvaLLMasRecommender` read-only.
- Inspected upstream README, templates, scripts, and ZIP manifests.
- Extracted targeted text from the local paper PDF with local Python/PyPDF2.

### Outputs Generated

- `report/plan_c_full_replication_matrix.md`
- `results/tables/plan_c_replication_matrix.csv`
- `results/tables/plan_c_required_artifacts.csv`
- `results/tables/plan_c_models_metrics_tasks_inventory.csv`

### Known Issues

- No full-paper numerical component is exact-ready from the current checkout.
- Beauty and Sports exact metrics remain blocked by missing preprocessed data.
- ML-1M and LastFM paper-equivalent replication require protocol validation
  beyond the completed Plan B reconstruction.
- Closed LLM reruns require APIs; open LLM reruns require local weights or a
  separately approved inference setup.

### Next Step

- Run Plan C C1 only after defining an artifact recovery and validation
  protocol for paper-equivalent processed files, sample users, candidate sets,
  SOTA outputs, and model-output provenance.

## 2026-06-29 - Plan C C1: Artifact Recovery And Validation Protocol

### Changes

- Created `report/plan_c_artifact_recovery_protocol.md` to define how Plan C
  artifacts should be classified, manually acquired, validated, and separated
  from reconstruction or extension artifacts.
- Added `data/artifact_manifests/plan_c_artifact_manifest_template.csv`.
- Added `data/artifact_manifests/plan_c_artifact_manifest.csv` with the
  initial local/missing artifact inventory for Beauty, Sports, ML-1M, LastFM,
  saved LLM outputs, saved SOTA outputs, prompt templates, and source URLs
  found in local files.
- Added `src/audit/validate_plan_c_artifacts.py`, a local-only manifest
  validator that checks paths, optional file sizes, optional SHA-256 hashes,
  and writes validation diagnostics under `results/tables/`.
- Added synthetic tests for existing files, missing files, size checks, hash
  checks, missing local paths, summary counts, and no-download behavior.

### Commands To Run

- `python -m pytest tests/`
- `python src/audit/validate_plan_c_artifacts.py --manifest data/artifact_manifests/plan_c_artifact_manifest.csv`

### Outputs Generated

- `report/plan_c_artifact_recovery_protocol.md`
- `data/artifact_manifests/plan_c_artifact_manifest_template.csv`
- `data/artifact_manifests/plan_c_artifact_manifest.csv`
- `src/audit/validate_plan_c_artifacts.py`
- `tests/test_validate_plan_c_artifacts.py`
- Expected validation outputs:
  - `results/tables/plan_c_artifact_validation_summary.csv`
  - `results/tables/plan_c_artifact_validation_details.csv`

### Known Issues

- The manifest records blockers; it does not resolve them.
- Beauty/Sports exact metrics remain blocked until the missing author
  preprocessed artifacts are recovered or a reconstruction protocol is approved.
- ML-1M Plan B files remain `extension_only` for paper replication.
- LastFM remains unavailable until manually supplied raw files and a
  paper-equivalent preprocessing protocol exists.

### Next Step

- If the user obtains any missing artifact manually, update the manifest
  `local_path`, source classification, size/hash expectations, and rerun the
  validator before implementing converters or metrics.

## 2026-06-29 - Plan C C2: External Source Checking And Recovery Ledger

### Changes

- Added `src/audit/check_plan_c_source_urls.py`, a metadata-only URL checker
  for non-empty source URLs in the Plan C artifact manifest.
- Added mocked tests for URL checking, HEAD/GET fallback behavior, network
  errors, empty URL skipping, duplicate URL grouping, and no-content-write
  behavior.
- Created `results/tables/plan_c_external_source_recovery_ledger.csv` to
  distinguish official processed artifacts, official raw datasets, local
  outputs, missing sources, reconstruction inputs, and author-contact blockers.
- Created `report/plan_c_external_source_recovery_report.md`,
  `report/plan_c_manual_download_checklist.md`, and
  `report/plan_c_author_contact_request.md`.

### Commands To Run

- `python -m pytest tests/`
- `python src/audit/check_plan_c_source_urls.py --manifest data/artifact_manifests/plan_c_artifact_manifest.csv`
- `python src/audit/validate_plan_c_artifacts.py --manifest data/artifact_manifests/plan_c_artifact_manifest.csv`

### Outputs Generated

- `src/audit/check_plan_c_source_urls.py`
- `tests/test_check_plan_c_source_urls.py`
- `results/tables/plan_c_source_url_check_results.csv`
- `results/tables/plan_c_external_source_recovery_ledger.csv`
- `report/plan_c_external_source_recovery_report.md`
- `report/plan_c_manual_download_checklist.md`
- `report/plan_c_author_contact_request.md`

### Known Issues

- Dropbox HTTP metadata responses do not prove that the Beauty/Sports processed
  ZIP files are downloadable or valid.
- The Amazon 2014 URL check may be network-unverified if the metadata request
  times out; this is not evidence that the source is unavailable.
- Raw public datasets remain reconstruction inputs unless paper-equivalent
  preprocessing, candidate sets, mappings, and seeds are recovered.

### Next Step

- Manually check the Beauty/Sports processed Dropbox links and, if they remain
  unavailable, contact the authors using
  `report/plan_c_author_contact_request.md` before implementing Plan C
  converters, model runs, or metrics.

## 2026-06-29 - Plan C C3: Manual Artifact Intake And Archive Inspection

### Changes

- Added the ignored manual staging area `data/external/plan_c/` with a tracked
  `.gitkeep` placeholder.
- Updated `.gitignore` so real staged Plan C downloads under
  `data/external/plan_c/` remain untracked while artifact manifests remain
  trackable.
- Added `src/audit/inspect_plan_c_downloads.py`, a local-only file and archive
  inspector that scans staging, lists ZIP/TAR members without extraction,
  detects known artifact filenames, and writes diagnostic tables.
- Added synthetic tests for empty staging, regular files, ZIP inspection,
  expected-file detection, deterministic dataset guesses, suggested manifest
  updates, no extraction, and no `original_repo/` mutation.
- Created `report/plan_c_manual_artifact_intake_guide.md`.

### Commands To Run

- `python -m pytest tests/`
- `python src/audit/inspect_plan_c_downloads.py --staging-dir data/external/plan_c --manifest data/artifact_manifests/plan_c_artifact_manifest.csv`
- `python src/audit/validate_plan_c_artifacts.py --manifest data/artifact_manifests/plan_c_artifact_manifest.csv`

### Outputs Generated

- `data/external/plan_c/.gitkeep`
- `src/audit/inspect_plan_c_downloads.py`
- `tests/test_inspect_plan_c_downloads.py`
- `report/plan_c_manual_artifact_intake_guide.md`
- Expected inspection outputs:
  - `results/tables/plan_c_manual_download_inventory.csv`
  - `results/tables/plan_c_manual_download_archive_contents.csv`
  - `results/tables/plan_c_manual_download_artifact_matches.csv`
  - `results/tables/plan_c_manual_download_summary.csv`
  - `results/tables/plan_c_suggested_manifest_updates.csv`

### Known Issues

- Staged files are only candidates until manually confirmed in the manifest
  and validated.
- Archive inspection lists member names but does not prove that the content is
  semantically correct or paper-equivalent.
- Raw public datasets remain reconstruction inputs unless exact paper
  preprocessing and candidate-set protocols are recovered.

### Next Step

- If the user manually places candidate archives in `data/external/plan_c/`,
  rerun the inspector, review suggested manifest updates manually, then rerun
  the Plan C artifact validator before implementing converters or metrics.

## 2026-06-29 - Plan C C4: Raw Dataset Schema Validation And Reconstruction Readiness

### Changes

- Added `src/audit/validate_plan_c_raw_datasets.py`, a local-only raw dataset
  validator for staged Plan C files under `data/external/plan_c/`.
- Added synthetic tests covering Amazon review JSON-gzip parsing, Amazon
  metadata parsing, MovieLens-1M ZIP parsing, LastFM ZIP parsing, missing-file
  diagnostics, sample limits, no permanent archive extraction, and no metric
  output creation.
- Generated raw validation diagnostics under `results/tables/`.
- Updated Plan C documentation to state that raw public data can support
  reconstruction audits but does not recover the original Beauty/Sports
  processed artifacts.

### Commands To Run

- `python -m pytest tests/`
- `python src/audit/validate_plan_c_raw_datasets.py --staging-dir data/external/plan_c --sample-size 1000`

### Outputs Generated

- `src/audit/validate_plan_c_raw_datasets.py`
- `tests/test_validate_plan_c_raw_datasets.py`
- `results/tables/plan_c_raw_dataset_validation_summary.csv`
- `results/tables/plan_c_raw_dataset_schema_details.csv`
- `results/tables/plan_c_raw_dataset_sample_stats.csv`
- `results/tables/plan_c_reconstruction_readiness_summary.csv`

### C4 Findings

- Amazon Beauty raw files are reconstruction-ready from sampled schema checks:
  reviews include user, item, rating, and timestamp fields; metadata includes
  item identifiers and catalog text fields.
- Amazon Sports raw files are reconstruction-ready from sampled schema checks
  under the same interpretation.
- MovieLens-1M raw ZIP is reconstruction-ready from sampled `ratings.dat`,
  `movies.dat`, and `users.dat` parsing.
- LastFM HetRec raw ZIP supports interaction and metadata reconstruction, but
  chronological split support is partial because `user_artists.dat` has weights
  without listening timestamps.

### Known Issues

- C4 sampled 1,000 rows per file/member by default; it is a schema/readiness
  audit, not a full converter or full-file statistical report.
- Exact Beauty/Sports reproduction remains blocked because the official
  `sequential_data.txt`, `negative_samples.txt`, `datamaps.json`, and processed
  `meta.json` artifacts are still not recovered.
- No processed datasets, metrics, model outputs, or training artifacts were
  created.

### Next Step

- Implement a separate Plan C reconstruction design for converting raw files
  only after deciding which dataset protocols, split rules, candidate sampling,
  and non-equivalence labels will be used.

## 2026-06-29 - Plan C C5: Raw-To-Canonical Reconstruction Converters

### Changes

- Added `configs/plan_c_reconstruction.yaml` for Plan C reconstruction inputs,
  output paths, seed, negative sampling, minimum interactions, enabled
  datasets, and dataset-specific raw filenames.
- Updated `.gitignore` so generated files under `data/reconstructed/plan_c/`
  remain untracked while `data/reconstructed/plan_c/.gitkeep` is trackable.
- Added `src/data/reconstruct_plan_c_datasets.py`, a config-driven converter
  that writes reconstructed canonical artifacts under
  `data/reconstructed/plan_c/<dataset>/`.
- Added synthetic tests for Amazon, ML-1M, LastFM, deterministic negatives,
  missing inputs, reconstructed artifact labels, and no metric output creation.
- Updated Plan C documentation to record the confirmed Beauty/Sports Dropbox
  blocker and the distinction between exact official artifacts and
  reconstructed public-raw artifacts.

### Commands To Run

- `python -m pytest tests/`
- `python src/data/reconstruct_plan_c_datasets.py --config configs/plan_c_reconstruction.yaml`
- Development-only subset:
  `python src/data/reconstruct_plan_c_datasets.py --config configs/plan_c_reconstruction.yaml --datasets amazon_beauty --max-users 1000`
- Development-only CSV/debug mode:
  `python src/data/reconstruct_plan_c_datasets.py --config configs/plan_c_reconstruction.yaml --datasets ml1m --skip-paper-like-artifacts`

### Intended Outputs

- `data/reconstructed/plan_c/<dataset>/interactions.csv`
- `data/reconstructed/plan_c/<dataset>/ground_truth.csv`, when a safe split is
  supported
- `data/reconstructed/plan_c/<dataset>/candidates.csv`, when a safe split is
  supported
- `data/reconstructed/plan_c/<dataset>/catalog.csv`
- `data/reconstructed/plan_c/<dataset>/datamaps.json`
- `data/reconstructed/plan_c/<dataset>/meta.json`
- `data/reconstructed/plan_c/<dataset>/sequential_data.txt`, when a safe split
  is supported
- `data/reconstructed/plan_c/<dataset>/negative_samples.txt`, when a safe split
  is supported
- `data/reconstructed/plan_c/<dataset>/README_RECONSTRUCTED.md`
- `results/tables/plan_c_reconstructed_dataset_summary.csv`
- `results/tables/plan_c_reconstructed_artifact_inventory.csv`
- `results/tables/plan_c_reconstructed_dataset_blockers.csv`
- `results/tables/plan_c_reconstructed_dataset_stats.csv`

### Known Issues

- The official Beauty/Sports processed Dropbox artifacts remain unrecovered
  after manual testing and still require author contact.
- C5 artifacts are marked `reconstructed_format`; they must not be reported as
  exact original paper artifacts.
- LastFM does not receive a chronological leave-one-out split in C5 because
  `user_artists.dat` lacks listening timestamps.
- No metric CSVs should be written in this phase.
- Outputs created with `--max-users` are marked `diagnostic_subset=True` and
  must not be treated as final reconstruction outputs.

### Next Step

- After the reconstruction command is run successfully, inspect the diagnostics
  and only then decide whether to implement Plan C baseline/evaluation runners
  over reconstructed datasets with explicit reconstruction labels.

### C5 Stability Fix

- Replaced the original per-user full-dataframe scan in `_build_sequential_lines`
  with a grouped implementation that sorts once and builds user sequences from
  grouped interactions.
- Replaced per-user full-catalog negative-pool construction with deterministic
  rejection sampling against each user's history set.
- Added progress logging for dataset start, raw loading, catalog building,
  split construction, candidate generation, canonical writing, paper-like
  artifact writing, summary writing, and dataset finish.
- Added `--max-users` as a development-only diagnostic subset option and
  `--skip-paper-like-artifacts` as a development-only CSV/debug option.
- Verified ML-1M full reconstruction and Amazon Beauty `--max-users 1000`
  diagnostic reconstruction after the fix.

## 2026-06-29 - Plan C C6: Baselines On Reconstructed Datasets

### Changes

- Added `src/experiments/run_plan_c_baselines.py` to evaluate deterministic
  non-LLM baselines on reconstructed Plan C canonical datasets.
- Added Random, MostPop, and ContentBM25 execution under Plan C output names
  without overwriting Plan B metric files.
- Added Plan C baseline diagnostics, comparison summaries, and plot generation
  under `results/tables/` and `results/plots/`.
- Updated `.gitignore` so large generated
  `results/raw_outputs/plan_c_*_recommendations.csv` files remain untracked.
- Added tests for deterministic ranking, metric output creation, missing-input
  blockers, LastFM blocking, diagnostic subsets, and Plan B metric isolation.

### Commands To Run

- `python -m pytest tests/`
- `python src/experiments/run_plan_c_baselines.py --datasets ml1m --baselines random mostpop content_bm25`
- Development-only subset:
  `python src/experiments/run_plan_c_baselines.py --datasets amazon_beauty --baselines random mostpop --max-users 1000`

### Intended Outputs

- `results/metrics/plan_c_<dataset>_<baseline>_metrics.csv`
- `results/tables/plan_c_baseline_metrics_summary.csv`
- `results/tables/plan_c_baseline_comparison_summary.csv`
- `results/tables/plan_c_baseline_dataset_status.csv`
- `results/tables/plan_c_baseline_blockers.csv`
- `results/plots/plan_c_baseline_hr_at_5.png`
- `results/plots/plan_c_baseline_ndcg_at_5.png`
- `results/plots/plan_c_baseline_comparison.png`

### Known Issues

- C6 metrics are real metrics on reconstructed public-raw Plan C datasets.
  They are not exact official paper reproduction metrics.
- Beauty/Sports exact reproduction remains blocked by unrecovered official
  processed artifacts.
- LastFM remains blocked for utility metrics because C5 did not create valid
  ground truth or candidates.
- `--max-users` is diagnostic-only and marks rows as
  `diagnostic_subset=True`.

### C6 Metadata/Filename Safety Fix

- Diagnostic `--max-users N` runs now write separate metric and recommendation
  filenames containing `diagnostic_max_users_<N>`.
- Full runs keep the normal filenames:
  `results/metrics/plan_c_<dataset>_<baseline>_metrics.csv`.
- Diagnostic runs write:
  `results/metrics/plan_c_<dataset>_<baseline>_diagnostic_max_users_<N>_metrics.csv`.
- Metric, dataset-status, blocker, and comparison rows now include
  `diagnostic_subset`, `max_users`, and `run_scope`.
- Diagnostic subset results are only runtime/stability validation evidence.
  Only full non-diagnostic rows should be treated as final Plan C baseline
  results.

### C5/C6 Candidate Universe Safety Fix

- Audited C5 candidate generation and confirmed that the previous implementation
  sampled negatives from `catalog["item_id"]`.
- This was risky for Amazon because the metadata catalog is much larger than the
  observed interaction item universe, which can create many zero-popularity
  negatives and inflate MostPop.
- Added `negative_sampling.item_universe` to
  `configs/plan_c_reconstruction.yaml`.
- The default is now `interaction_items`: sample negatives from observed
  interaction items, excluding each user's full history.
- `catalog_items` remains available only as an explicit experimental mode.
- C5 summaries now record candidate universe diagnostics:
  `candidate_item_universe`, `candidate_universe_size`, `catalog_size`,
  `interaction_item_count`, and `users_with_insufficient_negatives`.
- C6 baseline runs now write candidate diagnostics, including candidates outside
  the reconstructed interaction universe and zero-training-popularity candidate
  rate.
- Previous suspicious full Amazon MostPop metrics generated with catalog-based
  candidate negatives are not treated as final and should be removed or
  superseded by regenerated metrics after C5 reconstruction.

## 2026-06-30 - Plan C C7: Final Consolidation

### Changes

- Added `src/experiments/summarize_plan_c_final_results.py`.
- Added final Plan C synthesis tables:
  `plan_c_final_results_overview.csv`,
  `plan_c_claims_and_limitations.csv`, and
  `plan_c_final_experiment_inventory.csv`.
- Added final overview plots for HR@5 and NDCG@5.
- Added `report/plan_c_final_report.md`.
- Added Italian oral-exam guide:
  `report/oral_exam_plan_c_guide.md`.
- Updated README, project spec, technical report, results analysis,
  reproducibility checklist, and prompt log.

### Interpretation

- Exact Beauty/Sports reproduction remains blocked by unavailable official
  processed artifacts.
- Plan C reconstructed evaluation is complete for full ML-1M baselines and
  full Amazon Random/MostPop baselines.
- Amazon ContentBM25 rows are diagnostic subsets unless a future full run is
  executed.
- LastFM remains `extension_only` with no utility metrics.

### Command

```powershell
python src/experiments/summarize_plan_c_final_results.py
```

## 2026-06-30 - Plan D D2: Controlled BPRMF Baseline

### Changes

- Added `src/baselines/plan_d_bprmf.py` with a PyTorch BPRMF model,
  deterministic negative sampling, candidate-set ranking, utility metric rows,
  training summary upsert, and optional Plan D plots.
- Added `src/experiments/run_plan_d_bprmf.py`.
- Added synthetic tests:
  `tests/test_plan_d_bprmf.py` and `tests/test_run_plan_d_bprmf.py`.
- Updated dependency metadata to include PyTorch.
- Updated `.gitignore` so generated Plan D raw recommendation files and
  optional checkpoint/cache directories are not committed.

### Commands Run

```powershell
python -m pytest tests/test_plan_d_bprmf.py tests/test_run_plan_d_bprmf.py
python -m pytest tests/
python src/experiments/run_plan_d_bprmf.py --dataset ml1m --epochs 1 --max-users 100 --max-train-interactions 5000 --scope diagnostic_subset --skip-plots --verbose
python src/experiments/run_plan_d_bprmf.py --dataset amazon_beauty --epochs 1 --max-users 100 --max-train-interactions 5000 --scope diagnostic_subset --skip-plots --verbose
python src/experiments/run_plan_d_bprmf.py --dataset amazon_sports --epochs 1 --max-users 100 --max-train-interactions 5000 --scope diagnostic_subset --skip-plots --verbose
```

### Outputs Generated

- `results/metrics/plan_d_ml1m_bprmf_diagnostic_subset_metrics.csv`
- `results/metrics/plan_d_amazon_beauty_bprmf_diagnostic_subset_metrics.csv`
- `results/metrics/plan_d_amazon_sports_bprmf_diagnostic_subset_metrics.csv`
- `results/tables/plan_d_bprmf_training_summary.csv`
- Raw recommendation CSVs were generated for the diagnostic subset runs and are
  intentionally Git-ignored.

### Notes And Limitations

- Diagnostic subset BPRMF runs were executed for `ml1m`, `amazon_beauty`, and
  `amazon_sports` only:
  `max_users=100`, `max_train_interactions=5000`, and `epochs=1`.
- Negative sampling uses the reconstructed train/candidate item universe and
  excludes each user's training positives.
- Evaluation ranks only reconstructed Plan C candidate sets.
- CLI diagnostic scope defaults to capped runs when explicit limits are omitted.
- No Plan C outputs were rerun or modified.

## 2026-06-30 - Plan D D3: Controlled LightGCN Baseline

### Changes

- Added `src/baselines/plan_d_lightgcn.py` with a PyTorch LightGCN model,
  sparse normalized bipartite graph construction, deterministic negative
  sampling via the Plan D BPRMF sampler, candidate-set ranking, utility metric
  rows, training summary upsert, and optional Plan D plots.
- Added `src/experiments/run_plan_d_lightgcn.py`.
- Added synthetic tests:
  `tests/test_plan_d_lightgcn.py` and `tests/test_run_plan_d_lightgcn.py`.

### Commands Run

```powershell
python -m pytest tests/test_plan_d_lightgcn.py tests/test_run_plan_d_lightgcn.py
python -m pytest tests/
python src/experiments/run_plan_d_lightgcn.py --dataset ml1m --epochs 1 --max-users 100 --max-train-interactions 5000 --scope diagnostic_subset --skip-plots --verbose
```

### Outputs Generated

- `results/metrics/plan_d_ml1m_lightgcn_diagnostic_subset_metrics.csv`
- `results/tables/plan_d_lightgcn_training_summary.csv`
- `results/raw_outputs/plan_d_ml1m_lightgcn_diagnostic_subset_recommendations.csv`
  was generated and is intentionally Git-ignored.

### Notes And Limitations

- The executed LightGCN run is a diagnostic subset only:
  `max_users=100`, `max_train_interactions=5000`, and `epochs=1`.
- The graph is built only from reconstructed Plan C training interactions after
  diagnostic limits are applied.
- Graph propagation uses a sparse normalized bipartite adjacency, not a dense
  matrix.
- Evaluation ranks only reconstructed Plan C candidate sets.
- No Plan C outputs were rerun or modified.

## 2026-06-30 - Plan D D4: Controlled SASRec Baseline

### Changes

- Added `src/baselines/plan_d_sasrec.py` with a PyTorch SASRec-style model,
  item and positional embeddings, causal Transformer encoder blocks, left-padded
  sequence handling, prefix-to-next-item BPR training, candidate-set ranking,
  utility metric rows, training summary upsert, and optional Plan D plots.
- Added `src/experiments/run_plan_d_sasrec.py`.
- Added synthetic tests:
  `tests/test_plan_d_sasrec.py` and `tests/test_run_plan_d_sasrec.py`.

### Commands Run

```powershell
python -m pytest tests/test_plan_d_sasrec.py tests/test_run_plan_d_sasrec.py
python -m pytest tests/
python src/experiments/run_plan_d_sasrec.py --dataset ml1m --epochs 1 --max-users 100 --max-train-interactions 5000 --scope diagnostic_subset --skip-plots --verbose
python src/experiments/run_plan_d_sasrec.py --dataset amazon_beauty --epochs 1 --max-users 100 --max-train-interactions 5000 --scope diagnostic_subset --skip-plots --verbose
python src/experiments/run_plan_d_sasrec.py --dataset amazon_sports --epochs 1 --max-users 100 --max-train-interactions 5000 --scope diagnostic_subset --skip-plots --verbose
```

### Outputs Generated

- `results/metrics/plan_d_ml1m_sasrec_diagnostic_subset_metrics.csv`
- `results/metrics/plan_d_amazon_beauty_sasrec_diagnostic_subset_metrics.csv`
- `results/metrics/plan_d_amazon_sports_sasrec_diagnostic_subset_metrics.csv`
- `results/tables/plan_d_sasrec_training_summary.csv`
  contains rows for `ml1m`, `amazon_beauty`, and `amazon_sports` diagnostic
  runs.
- `results/raw_outputs/plan_d_ml1m_sasrec_diagnostic_subset_recommendations.csv`
  was generated and is intentionally Git-ignored.
- `results/raw_outputs/plan_d_amazon_beauty_sasrec_diagnostic_subset_recommendations.csv`
  was generated and is intentionally Git-ignored.
- `results/raw_outputs/plan_d_amazon_sports_sasrec_diagnostic_subset_recommendations.csv`
  was generated and is intentionally Git-ignored.

### Notes And Limitations

- The executed SASRec run is a diagnostic subset only:
  `max_users=100`, `max_train_interactions=5000`, and `epochs=1`.
- Training sequences use only reconstructed Plan C training interactions.
- Interactions are ordered by training timestamp when available, with input row
  order as a deterministic tie-breaker. If timestamps are unavailable, the
  existing interaction row order is used and reported in metric notes.
- Evaluation ranks only reconstructed Plan C candidate sets for users with
  non-empty training histories after diagnostic limits.
- The CUDA diagnostic run emitted PyTorch warnings that some Transformer/CuBLAS
  kernels are not fully deterministic unless `CUBLAS_WORKSPACE_CONFIG` is set
  before process start. The run still completed and wrote diagnostic outputs.
- No Plan C outputs were rerun or modified.

## 2026-06-30 - Plan D D4.6A: Controlled GRU4Rec Diagnostic Baseline

### Changes

- Added `src/baselines/plan_d_gru4rec.py` with a lightweight PyTorch
  GRU4Rec-style sequential baseline, deterministic negative sampling,
  candidate-set ranking, utility metric rows, training summary upsert, and
  optional Plan D plots.
- Added `src/experiments/run_plan_d_gru4rec.py`.
- Added synthetic tests:
  `tests/test_plan_d_gru4rec.py` and `tests/test_run_plan_d_gru4rec.py`.
- Updated the Plan D reproducibility matrix and feasibility ledger so GRU4Rec is
  marked `DIAGNOSTIC_ONLY` / `diagnostic`, not merely planned.

### Commands Run

```powershell
python -m pytest tests/test_plan_d_gru4rec.py tests/test_run_plan_d_gru4rec.py
python -m pytest tests/
python src/experiments/run_plan_d_gru4rec.py --dataset ml1m --epochs 1 --max-users 100 --max-train-interactions 5000 --scope diagnostic_subset --skip-plots --verbose
python src/experiments/run_plan_d_gru4rec.py --dataset amazon_beauty --epochs 1 --max-users 100 --max-train-interactions 5000 --scope diagnostic_subset --skip-plots --verbose
python src/experiments/run_plan_d_gru4rec.py --dataset amazon_sports --epochs 1 --max-users 100 --max-train-interactions 5000 --scope diagnostic_subset --skip-plots --verbose
```

### Outputs Generated

- `results/metrics/plan_d_ml1m_gru4rec_diagnostic_subset_metrics.csv`
- `results/metrics/plan_d_amazon_beauty_gru4rec_diagnostic_subset_metrics.csv`
- `results/metrics/plan_d_amazon_sports_gru4rec_diagnostic_subset_metrics.csv`
- `results/tables/plan_d_gru4rec_training_summary.csv`
- Raw recommendation CSVs were generated for the diagnostic subset runs and are
  intentionally Git-ignored.

### Notes And Limitations

- At D4.6A completion, GRU4Rec had diagnostic subset coverage for `ml1m`,
  `amazon_beauty`, and `amazon_sports` only.
- Extended_subset and full GRU4Rec runs had not yet been executed at the time
  of the D4.6A note; later D4.6B, D4.7H, and D4.7I phases added
  extended_subset coverage.
- Evaluation ranks only reconstructed Plan C candidate sets for users with
  non-empty training histories after diagnostic limits.
- No Plan C outputs were rerun or modified.

## 2026-06-30 - Plan D D4.7K: Extended Neural Baseline Metadata Update

### Changes

- Updated the Plan D reproducibility matrix, paper component matrix,
  feasibility ledger, next-action priorities, and notes after completed
  extended_subset neural baselines.
- BPRMF, LightGCN, SASRec, and GRU4Rec are documented as implemented with both
  diagnostic_subset and extended_subset coverage for the utility-evaluable
  reconstructed datasets: `ml1m`, `amazon_beauty`, and `amazon_sports`.
- LastFM is explicitly documented as
  `extension_only_no_valid_utility_evaluation`, not omitted.

### Generated Artifacts Referenced

- `results/tables/plan_d_global_extended_neural_baselines_summary.csv`
  contains 12 utility-evaluable extended_subset rows.
- `results/tables/plan_d_global_extended_dataset_model_coverage.csv`
  contains 16 dataset/model coverage rows, including 4 LastFM rows.
- `results/tables/plan_d_global_neural_diagnostic_vs_extended.csv`
  contains 48 diagnostic-vs-extended comparison rows.
- `report/plan_d_global_extended_neural_baselines.md`
  documents the global extended neural baseline summary.

### Empirical Summary

- ML-1M ranking: SASRec > GRU4Rec > LightGCN > BPRMF.
- Amazon Beauty ranking: GRU4Rec > LightGCN > BPRMF > SASRec.
- Amazon Sports ranking: GRU4Rec > LightGCN > BPRMF > SASRec.
- Global HR@5 top 5:
  1. `ml1m / sasrec` - HR@5 = `0.3333333333333333`
  2. `ml1m / gru4rec` - HR@5 = `0.2931547619047619`
  3. `ml1m / lightgcn` - HR@5 = `0.259`
  4. `ml1m / bprmf` - HR@5 = `0.243`
  5. `amazon_beauty / gru4rec` - HR@5 = `0.159`

### Interpretation And Limitations

- SASRec is strongest on ML-1M extended_subset.
- GRU4Rec is strongest on the reconstructed Amazon datasets.
- LightGCN consistently beats BPRMF on Amazon Beauty and Amazon Sports.
- SASRec performs poorly on the Amazon datasets despite strong ML-1M
  performance.
- The reconstructed Amazon subsets are sparse, with about 7k train interactions
  for 1000 users, which likely affects sequence-heavy transformer performance.
- These are reconstructed Plan D extended_subset results, not exact numerical
  reproduction of the original paper.
- Official preprocessed Amazon artifacts and exact paper candidate pools remain
  unavailable.
- No fake LastFM utility metrics were generated; LastFM is excluded from
  HR@5/NDCG@5 rankings because no valid comparable chronological
  split/candidate/ground-truth setup is available in the reconstructed project.
- LLM experiments remain blocked or conditional until a backend/API/model
  decision is made.

### Commands Run

```powershell
python -m pytest tests/test_plan_d_reproducibility_matrix.py
python -m pytest tests/
```

## 2026-07-01 - Plan D D5B: LLM Model Registry and Backend Feasibility Audit

### Changes

- Added a conservative registry for the seven paper LLMs:
  `gpt-3.5-turbo-0125`, `gpt-4o-mini-2024-07-18`,
  `gpt-4o-2024-08-06`, `claude-3-haiku-20240307`,
  `llama-3-70b-instruct`, `qwen-2-7b-instruct`, and
  `mistral-7b-instruct-v0.2`.
- Added a local backend audit that checks only local command availability,
  local Python package importability, and local torch/CUDA status.
- Added a D5C-D5J LLM task roadmap with all LLM execution phases still marked
  `pending_backend` or `conditional`.
- Updated the Plan D feasibility ledger and next-action priorities around the
  local Ollama/Qwen path.

### Commands Run

```powershell
python src/experiments/audit_plan_d_llm_backend.py
```

The first sandboxed Python launch failed before the script ran because the
sandbox could not start `python.exe`; the same local-only audit command was
rerun with elevated execution permissions. No network access, downloads, API
calls, model pulls, or LLM inference were performed.

### Outputs Generated

- `results/tables/plan_d_llm_model_registry.csv`
- `results/tables/plan_d_llm_backend_audit.csv`
- `results/tables/plan_d_llm_task_roadmap.csv`
- `report/plan_d_llm_model_registry_and_backend_audit.md`

### Local Audit Result

- `ollama` was not found on PATH, so the local Ollama backend remains
  `blocked_missing_backend`.
- `requests` is importable.
- `openai`, `anthropic`, `transformers`, and `llama_cpp` are not importable.
- `torch` is importable and `torch.cuda.is_available()` returned `true`.
- CUDA device name: `NVIDIA GeForce RTX 4060 Laptop GPU`.

### Notes And Limitations

- No exact LLM reproduction is claimed yet.
- No LLM recommendations or raw LLM output files were created.
- The recommended first local target is `qwen2:7b-instruct` through Ollama.
- The recommended second/fallback local target is
  `mistral:7b-instruct-v0.2-q4_0` or
  `mistral:7b-instruct-v0.2-q4_K_M`.
- `llama-3-70b-instruct` is documented as blocked for local RTX 4060 8GB
  execution and cloud/API-required for practical evaluation.
- OpenAI and Anthropic/Claude models remain conditional on explicit user
  approval, keys, availability, and budget.
- D5C cannot start until Ollama and at least `qwen2:7b-instruct` are available
  locally.

## 2026-07-01 - Plan D D5C0: Ollama and Qwen Backend Availability Recheck

### Changes

- Reran the local backend audit after the user manually installed Ollama and
  manually pulled `qwen2:7b-instruct`.
- Updated backend/reporting artifacts so the local Ollama/Qwen path is no
  longer marked `blocked_missing_backend`.
- Updated the LLM task roadmap so Qwen-related local phases now wait on the
  D5C smoke test rather than backend installation.
- Updated next-action priorities so the first active next step is
  `run_d5c_qwen2_smoke_test`.

### Commands Run

```powershell
python -m pytest tests/
ollama --version
ollama list
python src/experiments/audit_plan_d_llm_backend.py
```

### Local Audit Result

- Ollama version: `0.31.1`.
- `qwen2:7b-instruct` is present in `ollama list`.
- The local Ollama/Qwen backend is ready for the D5C smoke test.
- Mistral remains unavailable locally.

### Notes And Limitations

- No LLM inference was run.
- `ollama run` was not called.
- No Ollama HTTP API, OpenAI API, or Anthropic API call was made.
- No model download or pull was performed by Codex.
- No recommender training was run.
- No Plan C outputs, metrics CSVs, raw outputs, or LLM recommendation outputs
  were modified or created.
- No exact LLM reproduction or LLM recommendation metric is claimed yet.


## 2026-07-01 - Plan D D5C1: Qwen2 Ollama Smoke Test

### Changes

- Added a local Ollama client, strict JSON parser, and smoke-test runner for
  `qwen2:7b-instruct`.
- Ran exactly one local Qwen2/Ollama inference against a toy
  recommendation-style prompt with no real user or dataset data.
- Wrote a one-row smoke summary CSV and a smoke-test report.

### Commands Run

```powershell
python src/experiments/run_plan_d_qwen_smoke_test.py
```

### Outputs Generated

- `results/tables/plan_d_qwen2_ollama_smoke_test.csv`
- `report/plan_d_qwen2_ollama_smoke_test.md`

### Result

- response_received: `true`
- parse_success: `true`
- schema_valid: `true`
- selected_count: `3`
- candidate_validity_rate: `1.000000`
- duplicate_count: `0`
- hallucinated_count: `0`
- latency_seconds: `9.196662`
- status: `smoke_test_passed`

### Notes And Limitations

- This was one toy local smoke inference only.
- This is not a dataset ranking experiment.
- This is not an exact paper LLM reproduction.
- No external API, OpenAI API, or Anthropic API was called.
- No model was pulled or downloaded in this phase.
- No recommender model was trained.
- No Plan C output, source metric CSV, recommender raw output, or LLM
  recommendation output was modified or created.

## 2026-07-01 - Plan D D5D: Prompt Templates, Parser, and ML-1M Manifest

### Changes

- Added reusable LLM ranking prompt templates for `openp5_base`,
  `llmrank_recency`, and `llmsrec_icl_stub`.
- Added strict JSON parser utilities for future LLM ranking outputs.
- Prepared a deterministic no-inference ML-1M Qwen2 ranking manifest.
- Wrote the prompt-template registry and D5D report.

### Commands Run

```powershell
python src/experiments/prepare_plan_d_llm_ranking_diagnostic.py
```

### Outputs Generated

- `results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_manifest.csv`
- `results/tables/plan_d_llm_prompt_template_registry.csv`
- `report/plan_d_llm_prompt_templates_and_ranking_manifest.md`

### Result

- Manifest rows: `20`
- Ready rows: `20`
- Rows with positives in candidates: `20`
- Status: `completed_no_inference`

### Notes And Limitations

- No LLM inference was run.
- No Ollama HTTP API, OpenAI API, Anthropic API, or external API call was made.
- No model was pulled or downloaded.
- No recommender training, dataset ranking, or LastFM experiment was run.
- No Plan C output, source metric CSV, recommender raw output, or LLM output was modified or created.
- No LLM ranking reproduction is claimed yet.

## 2026-07-01 - Plan D D5E: Qwen2 ML-1M Ranking Diagnostic

### Changes

- Ran the bounded local Qwen2/Ollama ML-1M ranking diagnostic over the 20 D5D manifest rows.
- Parsed every response with the strict D5D ranking parser.
- Wrote per-instance, aggregate, metric-style, raw trace, and report artifacts.
- Updated the Plan D LLM roadmap, next actions, feasibility ledger, and reproducibility matrix.

### Commands Run

```powershell
python src/experiments/run_plan_d_qwen2_ml1m_ranking_diagnostic.py
```

### Outputs Generated

- `results/metrics/plan_d_qwen2_ml1m_ranking_diagnostic_metrics.csv`
- `results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_instance_summary.csv`
- `results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_summary.csv`
- `results/raw_outputs/plan_d_qwen2_ml1m_ranking_diagnostic_raw.jsonl` (Git-ignored raw local trace)
- `report/plan_d_qwen2_ml1m_ranking_diagnostic.md`

### Result

- status: `completed_diagnostic_subset`
- num_instances: `20`
- HR@5: `0.150000`
- NDCG@5: `0.100889`
- response_received_rate: `1.000000`
- parse_success_rate: `1.000000`
- schema_valid_rate: `1.000000`
- mean_candidate_validity_rate: `0.630000`
- hallucination_rate: `0.370000`
- duplicate_rate: `0.000000`
- exact_top_k_rate: `1.000000`
- empty_output_rate: `0.000000`
- mean_latency_seconds: `5.652650`
- total_runtime_seconds: `113.389987`

### Notes And Limitations

- This was local-only Ollama inference with `qwen2:7b-instruct`.
- This is not exact paper reproduction and not a full-scope LLM run.
- No OpenAI API, Anthropic API, or remote API was called.
- No model was pulled or downloaded in this phase.
- No recommender model was trained.
- No BPRMF, LightGCN, SASRec, GRU4Rec, or LastFM experiment was run.
- No Plan C output, source metric CSV, recommender raw output, or dataset reconstruction file was modified.

## 2026-07-01 - Plan D D5E.1: Qwen2 Output Validity and Hallucination Analysis

### Changes

- Analyzed D5E Qwen2 ML-1M ranking outputs without running new LLM inference.
- Wrote aggregate output-validity, valid-only utility, hallucination instance profile, and invalid item pattern tables.
- Updated Plan D LLM roadmap, next actions, feasibility ledger, and reproducibility matrix.

### Commands Run

```powershell
python src/experiments/analyze_plan_d_qwen2_output_validity.py
```

### Outputs Generated

- `results/tables/plan_d_qwen2_output_validity_breakdown.csv`
- `results/tables/plan_d_qwen2_valid_only_utility_summary.csv`
- `results/tables/plan_d_qwen2_hallucination_instance_profile.csv`
- `results/tables/plan_d_qwen2_invalid_item_patterns.csv`
- `report/plan_d_qwen2_output_validity_and_hallucination_analysis.md`

### Result

- response_received_rate: `1.000000`
- parse_success_rate: `1.000000`
- schema_valid_rate: `1.000000`
- exact_top_k_rate: `1.000000`
- mean_candidate_validity_rate: `0.630000`
- micro_candidate_validity_rate: `0.630000`
- hallucination_rate: `0.370000`
- duplicate_rate: `0.000000`
- instances_with_hallucinations: `8`
- valid_only_HR@5: `0.150000`
- valid_only_NDCG@5: `0.100889`

### Notes And Limitations

- No new LLM inference was run.
- No Ollama command, Ollama HTTP API, OpenAI API, Anthropic API, remote API, download, model pull, recommender training, or LastFM experiment occurred.
- No Plan C output, recommender metric CSV, D5E source metric CSV, recommender raw output, or raw LLM output was modified.
- Ignored raw JSONL status: available for compact parsed invalid-ID analysis.
- Qwen2 is reliable at response/parse/schema level in D5E but not fully reliable at candidate-adherence level.

## 2026-07-01 - Plan D D5F: Qwen2 Candidate Position Bias Diagnostic

### Changes

- Ran the D5F local Qwen2/Ollama candidate position-bias diagnostic over 20 ML-1M users and 3 deterministic candidate-order variants.
- Wrote manifest, instance summary, variant summary, pairwise deltas, selected-position distribution, metric-style CSV, raw ignored JSONL, and report artifacts.
- Used existing D5E original-order outputs as the reference; original order was not rerun.

### Commands Run

```powershell
python src/experiments/run_plan_d_qwen2_ml1m_candidate_position_bias_diagnostic.py
```

### Outputs Generated

- `results/metrics/plan_d_qwen2_ml1m_candidate_position_bias_metrics.csv`
- `results/tables/plan_d_qwen2_ml1m_candidate_position_bias_instance_summary.csv`
- `results/tables/plan_d_qwen2_ml1m_candidate_position_bias_variant_summary.csv`
- `results/tables/plan_d_qwen2_ml1m_candidate_position_bias_pairwise_delta.csv`
- `results/tables/plan_d_qwen2_ml1m_candidate_position_bias_selected_position_distribution.csv`
- `results/tables/plan_d_qwen2_ml1m_candidate_position_bias_manifest.csv`
- `results/raw_outputs/plan_d_qwen2_ml1m_candidate_position_bias_diagnostic_raw.jsonl` (Git-ignored)
- `report/plan_d_qwen2_ml1m_candidate_position_bias_diagnostic.md`

### Result

- Local Qwen2 requests: `60`
- Runtime seconds: `330.391387`
- Variants completed: `reverse_order, positive_first, positive_last`

### Notes And Limitations

- D5F is a diagnostic_subset robustness diagnostic, not exact reproduction and not full LLM reproduction.
- No OpenAI API call, Anthropic API call, external API call, download, model pull, recommender training, LastFM experiment, or Plan C modification occurred.

## 2026-07-01 - Plan D D5G: Qwen2 Strict Prompt And Post-Filter Diagnostic

### Changes

- Ran the D5G local Qwen2/Ollama strict candidate-ID-only prompt diagnostic over the same 20 ML-1M D5E instances.
- Added deterministic post-filter analyses for the original D5E outputs and the D5G strict-prompt outputs.
- Wrote manifest, instance summary, condition summary, pairwise deltas, post-filter profile, metric-style CSV, raw ignored JSONL, and report artifacts.

### Commands Run

```powershell
python src/experiments/run_plan_d_qwen2_ml1m_strict_prompt_diagnostic.py
```

### Outputs Generated

- `results/metrics/plan_d_qwen2_ml1m_strict_prompt_metrics.csv`
- `results/tables/plan_d_qwen2_ml1m_strict_prompt_instance_summary.csv`
- `results/tables/plan_d_qwen2_ml1m_strict_prompt_condition_summary.csv`
- `results/tables/plan_d_qwen2_ml1m_strict_prompt_pairwise_delta.csv`
- `results/tables/plan_d_qwen2_ml1m_strict_prompt_post_filter_profile.csv`
- `results/tables/plan_d_qwen2_ml1m_strict_prompt_manifest.csv`
- `results/raw_outputs/plan_d_qwen2_ml1m_strict_prompt_diagnostic_raw.jsonl` (Git-ignored)
- `report/plan_d_qwen2_ml1m_strict_prompt_diagnostic.md`

### Result

- Local Qwen2 requests: `20`
- Runtime seconds: `95.546373`
- Strict prompt HR@5: `0.550000`
- Strict prompt NDCG@5: `0.288479`
- Strict prompt hallucination rate: `0.000000`
- Strict post-filter hallucination rate: `0.000000`

### Notes And Limitations

- D5G is a diagnostic_subset prompt mitigation diagnostic, not exact reproduction and not full LLM reproduction.
- No OpenAI API call, Anthropic API call, external API call, download, model pull, recommender training, LastFM experiment, or Plan C modification occurred.

## 2026-07-01 - Plan D D5H: Mistral Local Availability And Smoke Test

### Changes

- Added D5H local Mistral availability detection and conservative smoke-test artifacts.
- Checked local Ollama model inventory through `ollama list`.
- Ran a smoke test only if the configured Mistral model was already locally available.

### Outputs Generated

- `results/metrics/plan_d_mistral_smoke_test_metrics.csv`
- `results/tables/plan_d_mistral_smoke_test_summary.csv`
- `report/plan_d_mistral_smoke_test.md`
- `results/raw_outputs/plan_d_mistral_smoke_test_raw.jsonl` (Git-ignored)

### Result

- Requested model: `mistral:7b-instruct`
- Selected model: `mistral:7b-instruct`
- Status: `smoke_test_passed`
- Local Ollama inference requests: `1`
- Model pull/download performed: `false`

### Notes And Limitations

- D5H is not exact reproduction and not full LLM reproduction.
- No OpenAI API call, Anthropic API call, external API call, download, model pull, recommender training, LastFM experiment, or Plan C modification occurred.

## 2026-07-01 - Plan D D5I: Qwen2 vs Mistral Strict-Prompt Diagnostic

### Changes

- Ran the D5I local Mistral/Ollama strict candidate-ID-only diagnostic over the same 20 ML-1M D5G instances.
- Reused D5G Qwen2 strict-prompt outputs as the Qwen2 reference; Qwen2 was not rerun.
- Wrote manifest, instance summary, model summary, pairwise deltas, agreement profile, metric-style CSV, raw ignored JSONL, and report artifacts.

### Outputs Generated

- `results/metrics/plan_d_qwen_vs_mistral_ml1m_diagnostic_metrics.csv`
- `results/tables/plan_d_qwen_vs_mistral_ml1m_instance_summary.csv`
- `results/tables/plan_d_qwen_vs_mistral_ml1m_model_summary.csv`
- `results/tables/plan_d_qwen_vs_mistral_ml1m_pairwise_delta.csv`
- `results/tables/plan_d_qwen_vs_mistral_ml1m_agreement_profile.csv`
- `results/tables/plan_d_qwen_vs_mistral_ml1m_manifest.csv`
- `results/raw_outputs/plan_d_qwen_vs_mistral_ml1m_diagnostic_raw.jsonl` (Git-ignored)
- `report/plan_d_qwen_vs_mistral_ml1m_diagnostic.md`

### Result

- New local Mistral requests: `20`
- Mistral runtime seconds: `89.110282`
- Qwen HR@5/NDCG@5: `0.550000` / `0.288479`
- Mistral HR@5/NDCG@5: `0.100000` / `0.044343`
- Qwen hallucination rate: `0.000000`
- Mistral hallucination rate: `0.000000`

### Notes And Limitations

- D5I is a diagnostic_subset local open-model comparison, not exact reproduction and not full LLM reproduction.
- No OpenAI API call, Anthropic API call, external API call, download, model pull, recommender training, LastFM experiment, or Plan C modification occurred.

## 2026-07-01 - Plan D D5J: Local LLM ML-1M Re-Ranking Diagnostic

### Changes

- Ran D5J local LLM re-ranking diagnostics for Qwen2 and Mistral over the same 20 ML-1M diagnostic instances.
- Added a no-inference `candidate_order_reference` condition using the first 5 original candidate IDs as the baseline top-5.
- Wrote manifest, instance summary, condition summary, pairwise deltas, agreement profile, position-shift profile, metric-style CSV, raw ignored JSONL, and report artifacts.

### Outputs Generated

- `results/metrics/plan_d_llm_ml1m_reranking_diagnostic_metrics.csv`
- `results/tables/plan_d_llm_ml1m_reranking_instance_summary.csv`
- `results/tables/plan_d_llm_ml1m_reranking_condition_summary.csv`
- `results/tables/plan_d_llm_ml1m_reranking_pairwise_delta.csv`
- `results/tables/plan_d_llm_ml1m_reranking_agreement_profile.csv`
- `results/tables/plan_d_llm_ml1m_reranking_position_shift_profile.csv`
- `results/tables/plan_d_llm_ml1m_reranking_manifest.csv`
- `results/raw_outputs/plan_d_llm_ml1m_reranking_diagnostic_raw.jsonl` (Git-ignored)
- `report/plan_d_llm_ml1m_reranking_diagnostic.md`

### Result

- New local LLM requests: `40`
- Qwen2 runtime seconds: `106.028203`
- Mistral runtime seconds: `95.398157`
- Reference HR@5/NDCG@5: `0.100000` / `0.056546`
- Qwen2 HR@5/NDCG@5: `0.050000` / `0.031546`
- Mistral HR@5/NDCG@5: `0.050000` / `0.021534`

### Notes And Limitations

- D5J is a diagnostic_subset local LLM re-ranking diagnostic, not exact reproduction and not full LLM reproduction.
- No OpenAI API call, Anthropic API call, external API call, download, model pull, recommender training, LastFM experiment, or Plan C modification occurred.

## 2026-07-01 - Plan D D5K: Local LLM ML-1M History Length Sensitivity

### Changes

- Ran D5K history-length diagnostics for Qwen2 and Mistral over the same 20 ML-1M diagnostic instances.
- Varied only the user-history window: most recent 5, 10, and 20 interactions.
- Kept candidate sets, positive items, top_k, temperature, backend, and strict candidate-ID-only JSON constraints fixed.
- Wrote D5K manifest, instance summary, condition summary, pairwise deltas, stability profile, metric-style CSV, ignored raw JSONL, and report.

### Outputs Generated

- `results/metrics/plan_d_llm_ml1m_history_length_sensitivity_metrics.csv`
- `results/tables/plan_d_llm_ml1m_history_length_instance_summary.csv`
- `results/tables/plan_d_llm_ml1m_history_length_condition_summary.csv`
- `results/tables/plan_d_llm_ml1m_history_length_pairwise_delta.csv`
- `results/tables/plan_d_llm_ml1m_history_length_stability_profile.csv`
- `results/tables/plan_d_llm_ml1m_history_length_manifest.csv`
- `results/raw_outputs/plan_d_llm_ml1m_history_length_sensitivity_raw.jsonl` (Git-ignored)
- `report/plan_d_llm_ml1m_history_length_sensitivity.md`

### Result

- New local LLM requests: `120`
- Qwen2 medium-history HR@5/NDCG@5: `0.450000` / `0.247602`
- Mistral medium-history HR@5/NDCG@5: `0.100000` / `0.040876`
- Qwen2 total runtime seconds: `288.704263`
- Mistral total runtime seconds: `267.844203`

### Notes And Limitations

- D5K is a diagnostic_subset local LLM history-length sensitivity diagnostic, not exact reproduction and not full LLM reproduction.
- No OpenAI API call, Anthropic API call, external API call, download, model pull, recommender training, LastFM experiment, or Plan C modification occurred.

## 2026-07-01 - Plan D D5L: Final Aggregation and Reproducibility Package

### Changes

- Built final Plan D aggregate tables, LLM diagnostic summary, blocked-items table, reproducibility status table, and repository hygiene audit.
- Created `report/plan_d_final_reproducibility_package.md`.
- Performed only non-destructive audit checks; no cleanup, deletion, inference, API call, download, model pull, training, or LastFM run occurred.

### Outputs Generated

- `results/tables/plan_d_final_experiment_index.csv`
- `results/tables/plan_d_final_metric_summary.csv`
- `results/tables/plan_d_final_llm_diagnostic_summary.csv`
- `results/tables/plan_d_final_reproducibility_status.csv`
- `results/tables/plan_d_final_blocked_items.csv`
- `results/tables/plan_d_repository_hygiene_audit.csv`
- `report/plan_d_final_reproducibility_package.md`

### Result

- D5E base prompt HR@5/NDCG@5: `0.150000` / `0.100889` with hallucination `0.370000`.
- D5G strict prompt HR@5/NDCG@5: `0.550000` / `0.288479` with hallucination `0.000000`.
- D5K Qwen2 medium-history HR@5/NDCG@5: `0.450000` / `0.247602`.

### Notes And Limitations

- D5L is aggregation/reporting and hygiene audit only.
- The project remains reconstructed reproduction plus diagnostic/extended local work, not exact reproduction and not full LLM reproduction.

## 2026-07-01 - Plan D D5M: Final Report Critical Review and Exam Readiness Audit

### Changes

- Built the D5M exam-readiness checklist, claim-safety audit, oral-defense key points, and concise review report.
- Reviewed the D5L final package, reproducibility matrix, notes, final tables, blocked-items table, and hygiene audit.
- No new inference, API call, download, model pull, recommender training, LastFM run, destructive cleanup, or metric recomputation occurred.

### Outputs Generated

- `results/tables/plan_d_exam_readiness_checklist.csv`
- `results/tables/plan_d_claim_safety_audit.csv`
- `results/tables/plan_d_oral_defense_key_points.csv`
- `report/plan_d_exam_readiness_review.md`

### Result

- Final oral positioning should be: reconstructed reproduction plus local diagnostic/extended subset analysis.
- D5E-D5K remain diagnostic_subset local LLM diagnostics, not exact or full LLM reproduction.
- The review flags stale D5H blocked wording in the matrix as a documentation wording issue to avoid in oral presentation.

## 2026-07-02 - Plan E E0: Paper-Condition Alignment and Scale-Up Feasibility Audit

### Changes

- Added deterministic Plan E0 audit builders for paper-condition alignment, missing conditions, exactness/feasibility classification, scale-up call estimates, local runtime estimates, API placeholder cost scenarios, and the recommended Plan E roadmap.
- Added `report/plan_e_paper_condition_alignment_audit.md`.
- Added focused tests for the E0 builders, build script boundaries, deterministic outputs, raw-output hygiene, and Plan C preservation.

### Commands Run

- `python src/experiments/build_plan_e_paper_condition_alignment_audit.py`
- `python -m pytest tests/test_plan_e_paper_condition_alignment.py -q`
- `python -m pytest tests/ -q`

The first sandboxed Python launches failed before Python started with a Windows logon-session error, so the same local commands were rerun with approved escalation. No model, API, download, training, or LastFM command was run.

### Outputs Generated

- `results/tables/plan_e_paper_condition_alignment_matrix.csv`
- `results/tables/plan_e_missing_conditions.csv`
- `results/tables/plan_e_exactness_feasibility.csv`
- `results/tables/plan_e_scaleup_call_estimates.csv`
- `results/tables/plan_e_local_runtime_estimates.csv`
- `results/tables/plan_e_api_cost_scenarios.csv`
- `results/tables/plan_e_recommended_roadmap.csv`
- `report/plan_e_paper_condition_alignment_audit.md`

### Result

- Focused E0 tests: `15 passed`.
- Full suite: `440 passed, 2 skipped`.
- E0 remains audit-only: no inference, no Ollama call, no Qwen2/Mistral call, no API call, no download, no model pull, no training, no LastFM run, no Plan C modification, and no destructive cleanup.

### Notes And Limitations

- Plan E0 estimates calls, local runtime, and API costs as planning assumptions only; API prices require current official verification in E6.
- Strict candidate-ID-only prompting remains a project extension, not a paper-exact prompt strategy.
- Local Qwen2/Mistral 20-user results remain diagnostic_subset results.
- Closed-source models remain conditional/API-dependent, Llama3-70B remains hardware/cloud/API-dependent, and LastFM remains extension_only_no_valid_utility_evaluation or blocked unless valid artifacts are found.
- Next recommended phase: E1 full-data inventory and refined cost/runtime estimator.

## 2026-07-02 - Plan E E1: Full-Data Inventory and Cost/Runtime Estimator

### Changes

- Added deterministic E1 inventory builders for repository artifacts, dataset availability, candidate/ranking/reranking manifests, metric artifacts, raw-output hygiene, scale-up readiness, local/paper-matrix call estimates, runtime estimates, and next actions.
- Added `report/plan_e_full_data_inventory.md`.
- Added focused tests for E1 generation, deterministic outputs, LastFM blocking, raw-output hygiene, Plan C preservation, and no-inference boundaries.

### Commands Run

- `python src/experiments/build_plan_e_full_data_inventory.py`
- `python -m pytest tests/test_plan_e_full_data_inventory.py -q`
- `python -m pytest tests/ -q`

The first sandboxed Python launches failed before Python started with a Windows logon-session error, so the same local commands were rerun with approved escalation. No inference, Ollama call, API call, download, model pull, training, or LastFM command was run.

### Outputs Generated

- `results/tables/plan_e_full_data_inventory_dataset_summary.csv`
- `results/tables/plan_e_full_data_inventory_artifact_summary.csv`
- `results/tables/plan_e_full_data_inventory_manifest_summary.csv`
- `results/tables/plan_e_full_data_inventory_metric_summary.csv`
- `results/tables/plan_e_full_data_inventory_raw_output_hygiene.csv`
- `results/tables/plan_e_full_data_inventory_scaleup_readiness.csv`
- `results/tables/plan_e_full_data_inventory_call_estimates.csv`
- `results/tables/plan_e_full_data_inventory_runtime_estimates.csv`
- `results/tables/plan_e_full_data_inventory_next_actions.csv`
- `report/plan_e_full_data_inventory.md`

### Result

- Dataset inventory rows: `4`.
- Manifest inventory rows: `10`.
- Metric inventory rows: `191`.
- Focused E1 tests: `14 passed`.
- Full suite: `454 passed, 2 skipped`.

### Notes And Limitations

- E1 is inventory and planning only; it does not complete 1000-user paper-like evaluation.
- ML-1M has existing 20-user D5 diagnostic manifests; ML-1M, Beauty, and Sports still need E2 paper-like 1000-user 20-candidate manifests.
- LastFM remains conditional/blocked unless valid split, candidate, and ground-truth artifacts are found.
- Local Qwen2/Mistral status remains local diagnostic, not paper-exact.
- Closed-source costs are not verified in E1; API pricing verification belongs to E6.
- Next recommended phase: E2 1000-user paper-like candidate pool construction.

## 2026-07-02 - Plan E E2: 1000-User Paper-Like Candidate Pool Construction

### Changes

- Added deterministic E2 builders for reconstructed paper-like 1000-user ranking and reranking candidate manifests.
- Constructed ML-1M, Amazon Beauty, and Amazon Sports manifests with 20 candidates per user, exactly 1 positive item, and exactly 19 negative items.
- Added dataset eligibility, source audit, manifest summary, validation, blocked-dataset, next-action tables, and `report/plan_e_paper_like_candidate_pool_construction.md`.
- Added focused tests for E2 generation, manifest invariants, LastFM blocking, deterministic outputs, raw-output hygiene, Plan C preservation, and no-inference boundaries.

### Commands Run

- `python src/experiments/build_plan_e_paper_like_candidate_pools.py`
- `python -m pytest tests/test_plan_e_paper_like_candidate_pools.py -q`

The first sandboxed Python launches failed before Python started with a Windows logon-session error, so the same local commands were rerun with approved escalation. No inference, Ollama call, API call, download, model pull, training, or LastFM command was run.

### Outputs Generated

- `results/tables/plan_e_paper_like_candidate_pool_dataset_eligibility.csv`
- `results/tables/plan_e_paper_like_candidate_pool_manifest_summary.csv`
- `results/tables/plan_e_paper_like_candidate_pool_validation.csv`
- `results/tables/plan_e_paper_like_candidate_pool_source_audit.csv`
- `results/tables/plan_e_paper_like_candidate_pool_blocked_datasets.csv`
- `results/tables/plan_e_paper_like_candidate_pool_next_actions.csv`
- `results/tables/plan_e_ml1m_1000user_ranking_manifest.csv`
- `results/tables/plan_e_ml1m_1000user_reranking_manifest.csv`
- `results/tables/plan_e_beauty_1000user_ranking_manifest.csv`
- `results/tables/plan_e_beauty_1000user_reranking_manifest.csv`
- `results/tables/plan_e_sports_1000user_ranking_manifest.csv`
- `results/tables/plan_e_sports_1000user_reranking_manifest.csv`
- `report/plan_e_paper_like_candidate_pool_construction.md`

### Result

- ML-1M selected users: `1000`.
- Amazon Beauty selected users: `1000`.
- Amazon Sports selected users: `1000`.
- Complete manifests: `6`.
- Validation failures: `0`.
- Focused E2 tests: `14 passed`.

### Notes And Limitations

- E2 constructs reconstructed paper-like candidate manifests, not LLM results and not exact reproduction.
- Generated manifests are near/reconstructed reproduction support artifacts unless exact original paper candidate files are verified.
- LastFM remains conditional/blocked because valid comparable candidate, split, and ground-truth artifacts are not present.
- Strict prompt remains a Plan D/E extension, not a paper-exact prompt strategy.
- Next recommended phase: E3 baseline alignment using the E2 manifests.

## 2026-07-02 - Plan E E3A: Baseline Alignment and No-Training Evaluation Audit

### Changes

- Added deterministic E3A baseline-alignment builders for E2 manifest inventory, source inventory, score availability, no-training metrics, missing-baseline actions, validation, and next actions.
- Computed HR@5/NDCG@5 only for no-training-safe methods on E2 manifests: `candidate_order_reference`, `random_deterministic_reference`, and `MostPop`.
- Represented all paper baselines: BM25, MostPop, BPRMF, GRU4Rec, SASRec, and LightGCN.
- Added `report/plan_e_baseline_alignment.md`.
- Added focused tests for E3A output generation, E2 manifest loading, no-training metrics, missing baselines, validation checks, raw-output hygiene, Plan C preservation, and no-inference boundaries.

### Commands Run

- `python src/experiments/build_plan_e_baseline_alignment.py`
- `python -m pytest tests/test_plan_e_baseline_alignment.py -q`

The first sandboxed Python launches failed before Python started with a Windows logon-session error, so the same local commands were rerun with approved escalation. No inference, Ollama call, Qwen2/Mistral call, API call, download, model pull, training, neural-baseline training, or LastFM command was run.

### Outputs Generated

- `results/tables/plan_e_baseline_alignment_manifest_inventory.csv`
- `results/tables/plan_e_baseline_alignment_source_inventory.csv`
- `results/tables/plan_e_baseline_alignment_score_availability.csv`
- `results/tables/plan_e_baseline_alignment_no_training_metrics.csv`
- `results/tables/plan_e_baseline_alignment_missing_baselines.csv`
- `results/tables/plan_e_baseline_alignment_validation.csv`
- `results/tables/plan_e_baseline_alignment_next_actions.csv`
- `report/plan_e_baseline_alignment.md`

### Result

- E2 manifests loaded: `6`.
- No-training metric rows: `18`.
- Missing baseline rows: `30`.
- Validation failures: `0`.
- Focused E3A tests: `13 passed`.

### Notes And Limitations

- E3A is not full E3 baseline completion; it is a no-training alignment/evaluation audit.
- Candidate-order and deterministic-random references are sanity checks, not paper baselines.
- MostPop is computed from available reconstructed train interactions without model training.
- BM25, BPRMF, GRU4Rec, SASRec, and LightGCN require aligned E2 scores or future approved scoring/training.
- D5 local LLM diagnostics are not mixed into E3A baseline metrics.
- LastFM remains blocked/conditional.
- No full exact reproduction is claimed.

## 2026-07-23 - Plan D Phase 6CLEAN: Repository Pruning and Delivery Freeze

### Changes

- Added a conservative Phase 6CLEAN pruning utility, CLI wrapper, tests, manifests, and report.
- Ran a verified dry-run over the repository on branch `plan-d-extensions` at HEAD `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`.
- Created the external archive root at `../IRP_BeyondUtility_Project_ARCHIVE_PRE_PRUNING/`.
- Deleted only dry-run-approved technical byproducts: `.pytest_cache`, `__pycache__`, and `.pyc` files.
- Added `docs/external_archive_guide.md` and linked it from `README.md`.

### Commands Run

- `python -B -m pytest -p no:cacheprovider tests/test_plan_d_phase_6clean_repository_pruning.py`
- `python -B src/experiments/build_plan_d_phase_6clean_repository_pruning.py --mode dry-run`
- `python -B src/experiments/build_plan_d_phase_6clean_repository_pruning.py --mode apply --dry-run-sha256 5ad501d6db49a856440577fe2b988abb552f4855bceabfc57f4df8007b6c6030`
- `python -B -m pytest -p no:cacheprovider tests/test_plan_d_phase_6clean_repository_pruning.py tests/test_plan_d_phase_6a_repository_consolidation.py tests/test_plan_d_phase_6b_readme_reproducibility.py tests/test_plan_d_phase_5e_paper_coverage_audit.py tests/test_plan_d_phase_5d_robustness_consolidation.py tests/test_dataset_candidate_set_validation.py tests/test_canonical_metric_contract.py tests/test_canonical_llm_contract.py tests/test_output_parser.py tests/test_plan_d_llm_prompt_templates.py`
- `python -B src/experiments/build_plan_d_phase_6clean_repository_pruning.py --mode validate-post-pruning`

No inference, Ollama call, Qwen2/Mistral call, API call, download, model pull, training, neural-baseline training, LastFM command, Git staging, Git commit, Git reset, Git clean, or checkout was run.

### Outputs Generated

- `src/analysis/plan_d_phase_6clean_repository_pruning.py`
- `src/experiments/build_plan_d_phase_6clean_repository_pruning.py`
- `tests/test_plan_d_phase_6clean_repository_pruning.py`
- `results/tables/plan_d_phase_6clean_*.csv`
- `results/tables/plan_d_phase_6clean_dry_run_plan.sha256`
- `report/plan_d_phase_6clean_repository_pruning.md`
- `docs/external_archive_guide.md`
- `../IRP_BeyondUtility_Project_ARCHIVE_PRE_PRUNING/README_ARCHIVE.md`
- `../IRP_BeyondUtility_Project_ARCHIVE_PRE_PRUNING/manifests/*.csv`

### Result

- Dry-run rows: `1245`.
- Keep rows: `1142`.
- Archive rows: `0`.
- Technical deletion rows: `103`.
- Removed technical/cache directories: `12`.
- Final validation failures: `0`.
- Targeted post-pruning tests: `278 passed, 1 skipped`.

### Notes And Limitations

- No scientific artifact was archived because every non-technical candidate was canonical, referenced, raw/irreplaceable, required by retained artifacts, or unresolved under the conservative policy.
- The external archive therefore contains manifests and README metadata, but no archived payload files under `files/`.
- The final report phase was not started.
- No scientific result, metric value, prompt, parser, Phase 5E accounting value, or frozen conclusion was changed.

## 2026-07-02 - Plan E E3B: BM25/BM25-like Scoring on E2 Manifests

### Changes

- Added deterministic E3B BM25-like scoring over the E2 reconstructed paper-like manifests.
- Built user queries from local reconstructed train/history interactions and item text from local reconstructed catalogs.
- Computed E2-aligned HR@5/NDCG@5 for `BM25_like_reconstructed` where local text/history evidence was available.
- Added BM25 source inventory, query/corpus audit, score table, metrics, validation, blocked/missing table, next actions, report, and focused tests.

### Commands Run

- `python src/experiments/build_plan_e_bm25_scoring.py`
- `python -m pytest tests/test_plan_e_bm25_scoring.py -q`
- `python -m pytest tests/ -q`

The first sandboxed Python launches failed before Python started with a Windows logon-session error. The E3B build and focused E3B tests were rerun with approved escalation. The full-suite rerun request was rejected by the approval system because the account hit its usage limit, so the full suite could not be completed in this turn.

No inference, Ollama call, Qwen2/Mistral call, API call, download, model pull, training, neural-baseline training, or LastFM command was run.

### Outputs Generated

- `results/tables/plan_e_bm25_source_inventory.csv`
- `results/tables/plan_e_bm25_query_corpus_audit.csv`
- `results/tables/plan_e_bm25_scores.csv`
- `results/tables/plan_e_bm25_metrics.csv`
- `results/tables/plan_e_bm25_validation.csv`
- `results/tables/plan_e_bm25_missing_or_blocked.csv`
- `results/tables/plan_e_bm25_next_actions.csv`
- `report/plan_e_bm25_scoring.md`

### Result

- Scorable ML-1M/Beauty/Sports dataset-task rows: `6`.
- BM25-like score rows: `120000`.
- BM25-like metric rows: `6`.
- Missing/blocked rows: `2`, both for LastFM.
- Validation failures: `0`.
- Focused E3B tests: `13 passed`.
- Full test suite: not completed because the escalated rerun was rejected after the sandbox launch failed.

### Notes And Limitations

- E3B is not full E3 baseline completion; it is no-training local text scoring.
- BM25 rows are classified as `BM25_like_reconstructed` / `reconstructed_BM25_like`, not exact BM25 reproduction.
- Exact paper BM25 configuration, corpus, preprocessing, query construction, and candidate scoring files were not verified.
- Candidate-order, deterministic-random, and MostPop metrics remain from E3A.
- Neural baselines remain for E3C/E3D and require aligned scores or future approved scoring/training.
- LastFM remains blocked/conditional.
- No full exact reproduction is claimed.

## 2026-07-27 - F6G2 Output-Length Contract Resolution

### Changes

- Resolved the frozen F6G2 top-20/top-10 conflict as `D_repository_derived_top10_paper_unspecified`.
- Added an explicit output-contract matrix for ranking and re-ranking across Base/OpenP5, LLMRank, and LLMSRec-style strategies.
- Classified F6G1/F6G2 full-20 request and shard manifests as superseded for execution, while preserving them unchanged.
- Documented that the paper is explicit about candidate pool size 20 and top-5 metrics, but does not explicitly specify returned output-list length.

### Commands Run

- `rg`/`Get-Content`/`tar -tf` inspections of paper references, original repository templates, original parser/evaluator code, released ZIP members, and F6 reports/manifests.
- Local structured ZIP/JSON inspection with `.venv_f6g2_py312\Scripts\python.exe` to summarize official released prompt/result files.
- arXiv HTML inspection for the paper text and appendix.

No inference, Ollama call, API call, model pull, download, training, raw JSONL append, F6E3B/F6E4A/F6E4B modification, F6G1/F6G2 modification, Git staging, commit, reset, clean, or checkout was run.

### Outputs Generated

- `results/tables/phase_f6g2_output_length_contract_resolution.csv`
- `report/phase_f6g2_output_length_contract_resolution.md`

### Result

- Authoritative executable output length: `10`.
- Candidate pool size: `20` for ranking and re-ranking.
- Parser expected length: `10`.
- Metric cutoff: `K=5`.
- Paper status: `paper_unspecified` for returned output-list length.
- Repository status: `repository_derived_not_paper_specified`.
- F6G1 reuse unchanged: `no`; a new corrective top-10 offline freeze is required before LastFM re-ranking inference.

### Notes And Limitations

- LLMRank remains explicitly flagged because its user template says to rank 20 candidates, while the official wrapper/parser/evaluator consumes a top-10 list.
- The corrective freeze should reuse only the F6G1 logical/candidate evidence as inputs, not the top-20 request/shard manifests themselves.
- No new numerical metric results were created.

## 2026-07-29 - Phase F6I1 Repository Retention Audit And Safe Cleanup Batch 1

### Changes

- Reconstructed a conservative repository dependency graph and file inventory for `phase_f6i1`.
- Classified every inventoried project-relevant path into one retention class.
- Generated the requested pre-deletion manifests before cleanup.
- Executed batch-1 cleanup only for Python bytecode cache files with retained source-file regeneration paths.

### Commands Run

- `PYTHONNOUSERSITE=1 .\.venv_audit_recovery\Scripts\python.exe -B src/experiments/build_phase_f6i1_repository_retention_audit.py --execute-cleanup --verify-determinism`
- `PYTHONNOUSERSITE=1 .\.venv_audit_recovery\Scripts\python.exe -B -m pytest tests/test_phase_f6i1_repository_retention_audit.py -q`

No inference, API call, download, model pull, recommender training, checkpoint mutation, raw data mutation, original repository mutation, Git staging, commit, reset, clean, or checkout was run.

### Outputs Generated

- `src/analysis/phase_f6i1_repository_retention_audit.py`
- `src/experiments/build_phase_f6i1_repository_retention_audit.py`
- `tests/test_phase_f6i1_repository_retention_audit.py`
- `results/tables/phase_f6i1_*.csv`
- `results/manifests/phase_f6i1_initial_capture.json`
- `report/phase_f6i1_repository_retention_audit_and_safe_cleanup_batch1.md`

### Result

- Final inventory rows: `1948`.
- Dependency edges: `45826`.
- Cumulative delete-now candidates: `152`.
- Files deleted: `152`.
- Blocking validation failures: `0`.

### Notes And Limitations

- Runtime classification remains `defensible_repaired_clone_exact_package_versions` and is not exact original runtime reproduction.
- Environment directories were retained as complete units and package files were not classified individually.
- Manual-review artifacts remain in place for later review.
- No scientific result, prompt, parser, metric implementation, raw response, prediction, checkpoint, candidate set, history, config, report, or original repository file was deleted.

## 2026-07-31 - Phase F6J Scientific Evidence Synthesis

### Changes

- Added the F6J research-question evidence matrix, scientific claim registry, paper reproduction versus extension matrix, and limitations/threats table.
- Added `report/phase_f6j_scientific_evidence_synthesis.md` with the final F6J decision token `f6j_scientific_evidence_synthesis_complete`.
- Updated `report/future_operations_context_handoff.md` so the current handoff points to F6J and records cleanup closure.
- Added `tests/test_phase_f6j_scientific_evidence_synthesis.py` for schema, traceability, status coherence, deterministic ordering, protected-input checks, no raw-output creation, cleanup-current recording, and stale-token rejection.
- Registered F6J as a terminal later scientific phase in the current cleanup guard. The guard also accepts the F6G3 runtime manifest rewrite only when its blocked/no-inference/no-API semantics remain intact.

### Commands Run

- F6J mandatory preflight: `140 passed, 1698 deselected`.
- F6J focused synthesis test: `11 passed`.
- Final current cleanup sweep after F6J handoff restoration: `151 passed, 1698 deselected`.
- Adjacent scientific evidence checks: F6F2 `25 passed`; F6F1 `13 passed`; F6E4C `10 passed`; F6G3 `8 passed`; F6D2 substantive `15 passed, 1 deselected`; F6D4A `18 passed`; F5B core `8 passed, 2 deselected`; F5E core `6 passed, 2 deselected`; F6I2J/F6I2K/F6I2KR/F6I2L guard subset `38 passed`.

### Outputs Generated

- `results/tables/phase_f6j_research_question_evidence_matrix.csv`
- `results/tables/phase_f6j_scientific_claim_registry.csv`
- `results/tables/phase_f6j_paper_reproduction_extension_matrix.csv`
- `results/tables/phase_f6j_limitations_and_threats.csv`
- `report/phase_f6j_scientific_evidence_synthesis.md`
- `tests/test_phase_f6j_scientific_evidence_synthesis.py`

### Result

- Decision: `f6j_scientific_evidence_synthesis_complete`.
- RQ statuses: supported `4`; contradicted `3`; mixed/context-dependent `4`; unresolved `3`.
- Cleanup is terminally closed for F6J.

### Notes And Limitations

- No training, inference, API calls, downloads, package mutation, raw-output repair, Git staging, or commit were performed.
- No `phase_f6j` raw LLM outputs were created.
- No `original_repo/` or `data/raw/` status changes were observed.
- F6D2, F5B, and F5E still contain historical protected-hash or validation-baseline checks tied to older phase snapshots; F6J used their non-stale substantive evidence checks and records this as a limitation.
- Cost and runtime remain unresolved for current official claims because F6J did not browse, call APIs, or run new measurements.

<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation:start -->
## 2026-07-31 - Phase F6L Paper-Faithful Completion Authorization Checkpoint

### Changes

- Extended the existing F6L checkpoint builder, wrapper, config, tests, exactness matrices, request manifest, runtime/cost table, RQ revalidation table, and checkpoint report.
- Added the atomic request breakdown, LastFM reranking prerequisite audit, official artifact recovery audit, and final execution authorization matrix.
- Reclassified existing LastFM ranking/reranking evidence without overwriting raw outputs.
- Recorded official free-download recovery attempts and local official ZIP hashes; no downloads were performed because no missing official LastFM reranking artifact was found.

### Commands Run

- F6L checkpoint builder with the canonical recovered runtime.
- Focused F6L authorization test: `15 passed`.
- No cleanup gate was run in this continuation turn, per the explicit `Nessun cleanup` constraint.

### Outputs Generated

- `configs/phase_f6l_paper_faithful_completion_and_rq_revalidation.yaml`
- `src/analysis/plan_f6l_paper_faithful_completion_and_rq_revalidation.py`
- `src/experiments/build_phase_f6l_paper_faithful_completion_and_rq_revalidation.py`
- `tests/test_phase_f6l_paper_faithful_completion_and_rq_revalidation.py`
- `results/tables/phase_f6l_*.csv`
- `results/manifests/phase_f6l_*.json`
- `report/phase_f6l_paper_faithful_completion_and_rq_revalidation.md`

### Result

- Decision: `f6l_execution_matrix_partially_blocked_by_missing_official_artifacts`.
- Recalculated future local request count after explicit approval: `8800`.
- No paper-faithful LastFM reranking inference is authorized at checkpoint; request count remains `0`.

### Notes And Limitations

- Closed/paid models remain blocked unless exact official outputs are recovered.
- A new deterministic K-S-valid LastFM sample is not the exact original sample unless author manifests are recovered.
- Local Qwen2/Mistral outputs, if later generated, will not be exact numerical reproduction.
- No inference, Ollama start, paid/API call, download, training, cleanup, staging, commit, raw-output overwrite, or `original_repo/` mutation was performed.
<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation:end -->

<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_execute_preflight:start -->
## 2026-08-01 - Phase F6L Authorized Execute Preflight

### Changes

- Added and ran the F6L execute-mode preflight after explicit user authorization.
- Verified the aggregate 8800-request authorization and local-only Ollama model availability.
- Blocked before inference because required frozen lower-level execution inputs are absent.

### Outputs Generated

- `results/tables/phase_f6l_execute_preflight.csv`
- `results/tables/phase_f6l_authorized_request_manifest_audit.csv`
- `results/tables/phase_f6l_ollama_local_model_preflight.csv`
- `results/tables/phase_f6l_execution_blocker.csv`
- `results/manifests/phase_f6l_execute_preflight_manifest.json`

### Result

- Decision: `f6l_execution_blocked_by_contract_runtime_or_protected_input_regression`.
- Requests planned/completed/strict-valid/failed/residual: `8800`/`0`/`0`/`0`/`8800`.
- Blockers: per_request_manifest_exists_and_unique, lastfm_frozen_sample_artifact.
<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_execute_preflight:end -->

<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_contract_materialization:start -->
## 2026-08-01 - Phase F6L Execution Contract Materialization Attempt

### Changes

- Added the F6L contract materialization audit and fail-closed blocker manifest.
- Did not create the LastFM F6L sample, candidate file, or open-model request manifest because the K-S gate is not operationally recoverable from verified sources.

### Outputs Generated

- `results/tables/phase_f6l_lastfm_contract_reconstruction_audit.csv`
- `results/tables/phase_f6l_execution_contract_materialization_blockers.csv`
- `results/tables/phase_f6l_execution_contract_materialization_summary.csv`
- `results/manifests/phase_f6l_execution_contract_materialization_manifest.json`

### Result

- Decision: `f6l_execution_contract_materialization_blocked_by_missing_paper_specification`.
- Blockers: ks_gate_operational_definition; lastfm_temporal_leave_one_out_order.
<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_contract_materialization:end -->

<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_separated_checkpoints:start -->
## 2026-08-01 - Phase F6L Separated Profile And LastFM Checkpoints

### Changes

- Materialized `phase_f6l_profile_request_manifest.csv` with 4800 profile-family requests derived from the F6L authorization matrix and official Beauty/Sports ZIP prompt artifacts.
- Added the author clarification request draft, LastFM missing-specification registry, local operationalization contract, sensitivity plan, full-cohort extension plan, RQ evidence hierarchy, and separated checkpoint summary.
- No LastFM sample/candidate/request manifest was materialized.

### Result

- Decision: `f6l_profile_execution_ready_lastfm_operationalization_checkpoint_ready`.
- Profile rows: `4800`.
- LastFM exact status: `exact_lastfm_blocked_operationalization_and_full_cohort_planned_not_executed`.
<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_separated_checkpoints:end -->

<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_profile_open_model_execution:start -->
## 2026-08-01 - Phase F6L Profile Open-Model Execution

### Changes

- Executed or preflight-blocked the 4800-row profile manifest under the existing F6L phase.
- Wrote profile execution state, parse rows, recommendation rows, metrics, differences, runtime resources, and profile RQ revalidation artifacts when execution reached consolidation.

### Result

- Decision: `f6l_profile_open_model_execution_and_rq_revalidation_complete`.
- Planned/completed/strict-valid/failed/residual: `4800`/`4800`/`807`/`0`/`0`.
- Raw output directory: `results/llm/phase_f6l_profile_execution_20260801_010000`.
<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_profile_open_model_execution:end -->

<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_lastfm_execution_contracts:start -->
## 2026-08-02 - Phase F6L LastFM Execution Contracts

### Changes

- Materialized LastFM-only local operationalized K-S sample tables, ranking request manifests, reranking pool readiness, validation, and manifest under the existing F6L phase.
- Preserved the evidence hierarchy: paper-exact blocked; 1000-user local operationalization; full-cohort extension.
- Did not run inference, train models, call APIs, alter `original_repo/`, alter YAML, alter cleanup/guard logic, stage, commit, or write raw LLM outputs.

### Outputs Generated

- `results/tables/phase_f6l_lastfm_local_pipeline_audit.csv`
- `results/tables/phase_f6l_lastfm_operationalized_seed_search.csv`
- `results/tables/phase_f6l_lastfm_operationalized_ks_results.csv`
- `results/tables/phase_f6l_lastfm_operationalized_sample_cohort.csv`
- `results/tables/phase_f6l_lastfm_operationalized_candidates.csv`
- `results/tables/phase_f6l_lastfm_operationalized_ranking_request_manifest.csv`
- `results/tables/phase_f6l_lastfm_full_cohort_ranking_request_manifest.csv`
- `results/tables/phase_f6l_lastfm_operationalized_reranking_pool.csv`
- `results/tables/phase_f6l_lastfm_full_cohort_reranking_pool.csv`
- `results/tables/phase_f6l_lastfm_reranking_readiness.csv`
- `results/tables/phase_f6l_lastfm_execution_contracts_validation.csv`
- `results/manifests/phase_f6l_lastfm_execution_contracts_manifest.json`

### Result

- Decision: `f6l_lastfm_execution_contracts_ready_for_inference_authorization`.
- Selected local K-S seed: `20260801`.
- Operationalized ranking requests: `4000`.
- Full-cohort ranking requests: `7436`.
<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_lastfm_execution_contracts:end -->

<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_lastfm_operationalized_ranking_execution:start -->
## 2026-08-02 - Phase F6L LastFM Operationalized Ranking Execution

### Changes

- Executed only the authorized LastFM operationalized ranking manifest when preflight and cleanup shard coverage passed.
- Wrote append-only raw request/response evidence and rebuilt metrics/RQ validation from raw evidence.
- Preserved paper-exact blocking and reported the run only as local operationalized evidence.

### Result

- Decision: `f6l_lastfm_operationalized_ranking_execution_and_rq_revalidation_complete`.
- Run ID: `20260802_184502`.
- Raw output directory: `results/llm/phase_f6l_lastfm_operationalized_ranking_20260802_184502`.
<!-- phase_f6l_paper_faithful_completion_and_rq_revalidation_lastfm_operationalized_ranking_execution:end -->

<!-- phase_f6l_lastfm_parser_forensics:start -->
## 2026-08-04 - Phase F6L LastFM Parser Forensics

- Performed zero-inference forensic reparse of immutable LastFM operationalized ranking raw outputs.
- Built V1 failure taxonomy, blind audit sample, conservative V2 parser diagnostics, holdout validation, complete-case metrics, and RQ correction.
- Decision: `f6l_lastfm_ranking_reparse_complete_with_conservative_parser_recovery`.
- V2 valid rows: `1264` / `4000`.
<!-- phase_f6l_lastfm_parser_forensics:end -->

## phase_f6l_lastfm_corrective_output_contract_pilot (2026-08-04T10:19:32Z)

- Decision token: `f6l_lastfm_corrective_output_contract_pilot_failed_redesign_required`
- Run id: `20260804_115330`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_pilot_20260804_115330`
- Scope: corrective V3 JSON/index pilot, 40 local Ollama requests maximum.
- Acceptance: 11/14 checks passed; strict-valid 33/40.
- Original raw unchanged: `AF1ECB11AABFE8BE793888FC244C67DB349C42CE18F8AF7B7060D47167C6E45C` -> `AF1ECB11AABFE8BE793888FC244C67DB349C42CE18F8AF7B7060D47167C6E45C`.
- Scientific boundary: validity diagnostics only; smoke HR/NDCG is not RQ evidence.

## phase_f6l_lastfm_corrective_output_contract_v4_pilot (2026-08-04T10:52:51Z)

- Decision token: `f6l_lastfm_corrective_output_contract_v4_pilot_passed_full_rerun_ready`
- Run id: `20260804_123438`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_pilot_20260804_123438`
- Completed/strict-valid: 40/40 of 40.
- Regression users: `640, 709, 1333, 1749`; holdout users: `223, 387, 2090, 208, 895, 609`.
- Scientific boundary: V4 validity only; smoke HR/NDCG is not RQ evidence.

## phase_f6l_lastfm_corrective_output_contract_v4_full_ranking (2026-08-04T13:02:32Z)

- Decision token: `f6l_lastfm_v4_constrained_full_ranking_stopped_by_contract_circuit_breaker`
- Run ID: `20260804_131959`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_full_ranking_20260804_131959`
- Planned/completed/strict-valid/failed/residual: `4000`/`291`/`290`/`1`/`3709`.
- Metrics rows: `0`; beyond-utility rows: `0`; RQ rows: `0`; acceptance pass: `8/13`.
- Scope boundary: no reranking, no full cohort, no LLMSRec, no remote APIs, no training, no download.

## phase_f6l_lastfm_corrective_output_contract_v4_full_ranking (2026-08-04T17:34:52Z)

- Decision token: `f6l_lastfm_v4_full_ranking_resume_stopped_by_systemic_contract_regression`
- Run ID: `20260804_131959`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_full_ranking_20260804_131959`
- Planned/completed/strict-valid/failed/residual: `4000`/`491`/`487`/`4`/`3509`.
- Metrics rows: `0`; beyond-utility rows: `0`; RQ rows: `0`; acceptance pass: `9/13`.
- Scope boundary: no reranking, no full cohort, no LLMSRec, no remote APIs, no training, no download.

## phase_f6l_lastfm_corrective_output_contract_v4_full_ranking (2026-08-05T07:09:10Z)

- Decision token: `f6l_lastfm_v4_full_ranking_partial_resumable_infrastructure_interruption`
- Run ID: `20260804_131959`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_full_ranking_20260804_131959`
- Planned/completed/strict-valid/failed/residual: `4000`/`3054`/`3020`/`34`/`946`.
- Metrics rows: `0`; beyond-utility rows: `0`; RQ rows: `0`; acceptance pass: `9/13`.
- Scope boundary: no reranking, no full cohort, no LLMSRec, no remote APIs, no training, no download.

## phase_f6l_lastfm_corrective_output_contract_v4_full_ranking (2026-08-05T07:54:31Z)

- Decision token: `f6l_lastfm_v4_full_ranking_partial_resumable_infrastructure_interruption`
- Run ID: `20260804_131959`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_full_ranking_20260804_131959`
- Planned/completed/strict-valid/failed/residual: `4000`/`3054`/`3020`/`34`/`946`.
- Metrics rows: `0`; beyond-utility rows: `0`; RQ rows: `0`; acceptance pass: `8/13`.
- Scope boundary: no reranking, no full cohort, no LLMSRec, no remote APIs, no training, no download.

## phase_f6l_lastfm_corrective_output_contract_v4_full_ranking (2026-08-05T08:08:27Z)

- Decision token: `f6l_lastfm_v4_full_ranking_partial_resumable_infrastructure_interruption`
- Run ID: `20260804_131959`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_full_ranking_20260804_131959`
- Planned/completed/strict-valid/failed/residual: `4000`/`3054`/`3020`/`34`/`946`.
- Metrics rows: `0`; beyond-utility rows: `0`; RQ rows: `0`; acceptance pass: `8/13`.
- Scope boundary: no reranking, no full cohort, no LLMSRec, no remote APIs, no training, no download.

## phase_f6l_lastfm_corrective_output_contract_v4_full_ranking (2026-08-05T08:22:43Z)

- Decision token: `f6l_lastfm_v4_full_ranking_partial_resumable_infrastructure_interruption`
- Run ID: `20260804_131959`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_full_ranking_20260804_131959`
- Planned/completed/strict-valid/failed/residual: `4000`/`3054`/`3020`/`34`/`946`.
- Metrics rows: `0`; beyond-utility rows: `0`; RQ rows: `0`; acceptance pass: `8/13`.
- Scope boundary: no reranking, no full cohort, no LLMSRec, no remote APIs, no training, no download.

## phase_f6l_lastfm_corrective_output_contract_v4_full_ranking (2026-08-05T16:11:15Z)

- Decision token: `f6l_lastfm_v4_full_ranking_partial_resumable_infrastructure_interruption`
- Run ID: `20260804_131959`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_full_ranking_20260804_131959`
- Planned/completed/strict-valid/failed/residual: `4000`/`3991`/`3942`/`49`/`9`.
- Metrics rows: `0`; beyond-utility rows: `0`; RQ rows: `0`; acceptance pass: `8/13`.
- Scope boundary: no reranking, no full cohort, no LLMSRec, no remote APIs, no training, no download.

## phase_f6l_lastfm_corrective_output_contract_v4_full_ranking (2026-08-05T18:52:45Z)

- Decision token: `f6l_lastfm_v4_full_ranking_complete_with_observed_duplicate_invalidity`
- Run ID: `20260804_131959`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_full_ranking_20260804_131959`
- Planned/completed/strict-valid/failed/residual: `4000`/`4000`/`3950`/`50`/`0`.
- Metrics rows: `28`; beyond-utility rows: `28`; RQ rows: `3`; acceptance pass: `12/13`.
- Scope boundary: no reranking, no full cohort, no LLMSRec, no remote APIs, no training, no download.

## phase_f6l_lastfm_corrective_output_contract_v4_full_ranking (2026-08-05T19:11:05Z)

- Decision token: `f6l_lastfm_v4_full_ranking_complete_with_observed_duplicate_invalidity`
- Run ID: `20260804_131959`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_full_ranking_20260804_131959`
- Planned/completed/strict-valid/failed/residual: `4000`/`4000`/`3950`/`50`/`0`.
- Metrics rows: `28`; beyond-utility rows: `28`; RQ rows: `3`; acceptance pass: `13/13`.
- Scope boundary: no reranking, no full cohort, no LLMSRec, no remote APIs, no training, no download.

<!-- phase_f6l_lastfm_v4_ranking_metrics_freeze:start -->
## Ranking Metrics Freeze Before Reranking

- Freeze state: `ranking_metrics_frozen_before_reranking`.
- Run ID: `20260804_131959`.
- Decision token: `f6l_lastfm_v4_ranking_metrics_consolidated_and_frozen_for_reranking`.
- Freeze SHA-256: `D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC`.
- Canonical metrics: `results/tables/phase_f6l_lastfm_v4_ranking_freeze_metrics.csv`.
- Canonical comparisons: `results/tables/phase_f6l_lastfm_v4_ranking_freeze_comparisons.csv`.
- Summary JSON: `results/manifests/phase_f6l_lastfm_v4_ranking_metrics_freeze_summary.json`.
- Scope boundary: no reranking, no new inference, no raw repair, no deduplication, no API call.
- Commands run: `.venv_audit_recovery\Scripts\python.exe -B src\analysis\phase_f6l_lastfm_v4_ranking_metrics_freeze.py --config configs\phase_f6l_paper_faithful_completion_and_rq_revalidation.yaml`.
- Tests passed: `24 passed` for `tests/test_phase_f6l_lastfm_v4_ranking_metrics_freeze.py`, `tests/test_phase_f6l_lastfm_corrective_output_contract_v4_pilot.py`, and `tests/test_phase_f6l_lastfm_parser_forensics.py` with `PYTHONNOUSERSITE=1`.
- Known unrelated test gap: `tests/test_phase_f6l_lastfm_execution_contracts.py` still returns `blocked_by_missing_local_evidence` in this workspace and is not a reranking authorization.
<!-- phase_f6l_lastfm_v4_ranking_metrics_freeze:end -->

## phase_f6l_lastfm_v4_reranking_full_run (2026-08-06T18:51:23Z)

- Decision token: `f6l_lastfm_v4_reranking_stopped_by_systemic_duplicate_regression`.
- Run ID: `20260806_034027`.
- Raw output dir: `results/llm/phase_f6l_lastfm_v4_reranking_full_run_20260806_034027`.
- Source pool: `results/tables/phase_f6l_lastfm_operationalized_reranking_pool.csv`; SHA-256 `D42BEB8EDC13070D0FB5CB0E27D4049B7849BC74ED1C12AA446C5B97DE9A9398`.
- Ranking baseline freeze hash remained `D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC`; reranking baseline audit passed `12/12`.
- Preflight passed `38/38`; canary passed.
- Planned/completed/strict-valid/failed/residual: `4000`/`2400`/`2352`/`48`/`1600`.
- Stop reason: closed 200-new-response duplicate-token window 12, responses `2201-2400`, had `11/200 = 0.055`, above the authorized `>0.05` systemic duplicate threshold.
- No additional reranking inference was launched after this stop, because continuing would violate the predeclared duplicate-token circuit breaker.
- Generated reranking artifacts include `results/tables/phase_f6l_lastfm_v4_reranking_full_run_state.csv`, `results/tables/phase_f6l_lastfm_v4_reranking_full_run_duplicate_windows.csv`, `results/tables/phase_f6l_lastfm_v4_reranking_full_run_metrics.csv`, `results/tables/phase_f6l_lastfm_v4_reranking_full_run_ranking_baseline_comparison.csv`, `results/manifests/phase_f6l_lastfm_v4_reranking_full_run_manifest.json`, and `report/phase_f6l_lastfm_v4_reranking_full_run.md`.
- Tests passed: `.venv_audit_recovery\Scripts\python.exe -B -m pytest tests/test_phase_f6l_lastfm_v4_reranking_full_run.py -q` returned `10 passed`.
- The historical `f6l_lastfm_execution_contracts_blocked_by_missing_local_evidence` gap remains classified as `historical_pre_existing_gap` and was not needed to define the reranking cohort, candidate pool, positive labels, or provenance.

## phase_f6l_lastfm_v4_reranking_post_breaker (2026-08-06T18:51:23Z)

- Decision token: `f6l_lastfm_v4_reranking_post_breaker_blocked_pending_evidence`.
- Audited run ID: `20260806_034027`; no inference, Ollama call, API call, detached process, raw overwrite, response repair, deduplication, or contract rewrite was performed.
- Final completed/strict-valid/structural-invalid/residual final-response requests: `2400`/`2352`/`48`/`1600`.
- Final invalid classification: all `48` structural invalids are `duplicate_token`; malformed JSON, unknown token, out-of-domain token, wrong length, mapping failure, and other structural errors are `0`.
- Circuit breaker remains contractually valid: closed window 12 over final response ordinals `2201-2400` has `11/200 = 0.055`, above `>0.05`; it is the first and only non-overlap 200-response window above 5%.
- Statistical interpretation: evidence supports a local/cell-specific duplicate-token spike in Mistral cells, not a global systemic regression claim.
- Blocking evidence: the exact residual final-response set is not manifest ordinal interval `2401-4000`; it is `48` earlier non-final timeout attempts plus never-attempted ordinals `2449-4000`.
- Baseline ranking audit passed again with freeze hash `D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC`; `original_repo/` status is clean.
- Generated: `results/tables/phase_f6l_lastfm_v4_reranking_post_breaker_cell_coverage.csv`, `results/tables/phase_f6l_lastfm_v4_reranking_post_breaker_failure_taxonomy.csv`, `results/tables/phase_f6l_lastfm_v4_reranking_post_breaker_window_analysis.csv`, `results/tables/phase_f6l_lastfm_v4_reranking_post_breaker_metrics.csv`, `results/tables/phase_f6l_lastfm_v4_reranking_post_breaker_statistical_assessment.csv`, `results/tables/phase_f6l_lastfm_v4_reranking_post_breaker_decision_matrix.csv`, `results/manifests/phase_f6l_lastfm_v4_reranking_post_breaker_summary.json`, `results/manifests/phase_f6l_lastfm_v4_reranking_post_breaker_hashes.json`, and `report/phase_f6l_lastfm_v4_reranking_post_breaker_forensic_consolidation.md`.
- V4.1 continuation proposal was not generated because the decision is `blocked_pending_evidence`; human review must decide the policy for the 48 non-final timeout attempts before any continuation authorization.
- Tests passed: `.venv_audit_recovery\Scripts\python.exe -B -m pytest tests/test_phase_f6l_lastfm_v4_reranking_post_breaker.py -q` returned `8 passed`.

## phase_f6l_lastfm_v4_reranking_continuation_v4_1_freeze (2026-08-07)

- Decision token: `f6l_lastfm_v4_reranking_post_breaker_versioned_continuation_ready`.
- Continuation ID: `phase_f6l_lastfm_v4_reranking_continuation_v4_1_20260807`.
- No inference, Ollama call, API call, detached process, raw overwrite, response repair, deduplication, or commit was performed.
- Policy resolved: a canonical request is completed only with a final response plus parser classification; non-final transport timeout attempts remain pending final-response requests.
- V4.1 target selection is `all_and_only_canonical_requests_without_final_response`, not manifest interval `2401-4000`.
- Continuation manifest rows: `1600`; `48` are `retry_after_non_final_transport_timeout`; `1552` are `first_attempt_never_attempted`; `prior_final_response_exists=false` for all rows.
- Raw artifact manifest hash preserved before/after freeze: `AB6449BD7BE8B8288A6FB0DB5FB3F3439F616662AE93FF572065120D2B6BE030`.
- Parser/schema/output contract hashes remain V4 frozen; ranking baseline freeze hash remains `D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC`; `original_repo/` is clean.
- Duplicate-token breaker V4.1 policy: stop on two consecutive closed 200-final-response windows in the same model/strategy cell above `0.05`, or on same-cell cumulative duplicate-token rate above `0.05` after at least `400` final responses; duplicate_token remains structural-invalid fail-strict with no repair, deduplication, or duplicate-token retry.
- Generated: `configs/phase_f6l_lastfm_v4_reranking_continuation_v4_1.yaml`, `results/tables/phase_f6l_lastfm_v4_reranking_continuation_v4_1_manifest.csv`, `results/tables/phase_f6l_lastfm_v4_reranking_continuation_v4_1_policy_audit.csv`, `results/manifests/phase_f6l_lastfm_v4_reranking_continuation_v4_1_summary.json`, `results/manifests/phase_f6l_lastfm_v4_reranking_continuation_v4_1_hashes.json`, and `report/phase_f6l_lastfm_v4_reranking_continuation_v4_1_freeze.md`.
- Tests passed: `.venv_audit_recovery\Scripts\python.exe -B -m pytest tests/test_phase_f6l_lastfm_v4_reranking_continuation_v4_1.py -q` returned `8 passed`.
- The continuation is ready for human review but not launched; a separate explicit authorization is still required before any new inference.

## phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution (2026-08-07T22:40:17Z)

- Decision token: `f6l_lastfm_v4_reranking_continuation_v4_1_complete`.
- Execution run ID: `20260807_111348`.
- Raw output dir: `results/llm/phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution_20260807_111348`.
- Execution path: the initial continuation process reached `1563/1600` final responses and stopped with `f6l_lastfm_v4_reranking_continuation_v4_1_partial_resumable_transport_timeout`; the idempotent `--resume` run completed the `37` residual final responses.
- One failed resume launch used the script path form and stopped immediately with `ModuleNotFoundError: No module named 'src'`; it wrote only the dedicated resume1 stderr log and did not change raw/state artifacts. The corrected module launch used `-m src.experiments.run_phase_f6l_lastfm_v4_reranking_continuation_v4_1`.
- Planned/continuation final/aggregate final responses: `1600`/`1600`/`4000`.
- Strict-valid/structural-invalid/duplicate-token invalid: `1571`/`29`/`29`.
- Raw continuation files: `1637`, consisting of `1600` final responses plus `37` preserved non-final transport timeout attempts.
- Final validation passed `10/10`: continuation manifest `1600`, aggregate unique IDs `4000`, original final responses `2400`, no original/continuation overlap, original raw preserved, parser/schema invariant, ranking freeze hash invariant, and `original_repo/` clean.
- Ranking baseline freeze hash remained `D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC`; no ranking value was regenerated or overwritten.
- Duplicate-window audit produced two individual closed windows above `0.05`, but no V4.1 breaker: they were not consecutive in the same cell and the cumulative same-cell duplicate-token rates remained below the `>0.05` stop threshold. The run therefore remains complete, not stopped.
- Generated final execution artifacts include `results/tables/phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution_fail_strict_metrics.csv`, `results/tables/phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution_ranking_vs_reranking_comparison.csv`, `results/tables/phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution_final_validation.csv`, `results/manifests/phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution_summary.json`, `results/manifests/phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution_hashes.json`, and `report/phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution_report.md`.
- Focused tests passed during the final resume: `13 passed` for `tests/test_phase_f6l_lastfm_v4_reranking_continuation_v4_1.py` and `tests/test_phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution.py`.
- The historical `f6l_lastfm_execution_contracts_blocked_by_missing_local_evidence` gap remains classified as `historical_pre_existing_gap`; it was not needed to define reranking cohort, candidate pool, positive labels, or provenance.
- Scope boundary: no baseline ranking modification, no original V4 raw overwrite, no `original_repo/` modification, no response repair, no deduplication, no new model, no training, no remote API.

## phase_f6m_reproduction_closure (2026-08-10T09:01:25Z)

- Decision token: `f6m_reproduction_closure_complete_with_documented_limitations`.
- Status: `REPRODUCIBILITY CORE - CLOSED`; `FUTURE EXTENSION - NOT STARTED` as a historical F6M-only statement, superseded for repository-wide status by later G1-G4 extension freezes.
- Commands run:
  - `.venv_audit_recovery\Scripts\python.exe -B -m pytest tests --collect-only -q -p no:cacheprovider`: `1949 tests collected`.
  - `.venv_audit_recovery\Scripts\python.exe -B -m pytest tests/test_phase_f6l_lastfm_v4_reranking_continuation_v4_1.py tests/test_phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution.py -q -p no:cacheprovider`: `13 passed`.
  - `.venv_audit_recovery\Scripts\python.exe -B -m pytest tests/test_phase_f6m_reproduction_closure.py -q -p no:cacheprovider --basetemp results/logs/phase_f6m_pytest_tmp`: `5 passed`.
  - `.venv_audit_recovery\Scripts\python.exe -B src\experiments\build_phase_f6m_reproduction_closure.py --config configs\phase_f6m_reproduction_closure.yaml`: emitted `results/manifests/phase_f6m_reproduction_closure_manifest.json`.
- Generated F6M artifacts: `report/phase_f6m_reproduction_closure.md`, `results/manifests/phase_f6m_reproduction_closure_manifest.json`, and `results/tables/phase_f6m_*.csv`.
- Documentation updated: `README.md`, `report/final_scientific_report.md`, and `report/future_operations_context_handoff.md` now use F6M as the final closure layer.
- LastFM ranking remains frozen with SHA-256 `D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC`.
- LastFM V4.1 reranking is complete with `4000/4000` aggregate final responses; reranking positive-present coverage is `314/1000 = 0.314` and diagnostic only.
- Known validation note: default Windows temp directory access is restricted in this workspace, so F6M focused tests use repo-local `--basetemp results/logs/phase_f6m_pytest_tmp`.
- Post-change validation:
  - `.venv_audit_recovery\Scripts\python.exe -B -m pytest tests/test_phase_f6m_reproduction_closure.py -q -p no:cacheprovider --basetemp results/logs/phase_f6m_pytest_tmp`: `5 passed`.
  - `.venv_audit_recovery\Scripts\python.exe -B -m pytest tests/test_phase_f6l_lastfm_v4_ranking_metrics_freeze.py tests/test_phase_f6l_lastfm_v4_reranking_continuation_v4_1.py tests/test_phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution.py -q -p no:cacheprovider --basetemp results/logs/phase_f6m_pytest_tmp_f6l`: `17 passed`.
  - `.venv_audit_recovery\Scripts\python.exe -B -m pytest tests/test_phase_f6j_scientific_evidence_synthesis.py tests/test_phase_f6k_scientific_gap_readiness.py tests/test_plan_d_phase_6b_readme_reproducibility.py tests/test_plan_d_phase_6c_final_scientific_report.py -q -p no:cacheprovider --basetemp results/logs/phase_f6m_pytest_tmp_docs`: `107 passed`.
  - `.venv_audit_recovery\Scripts\python.exe -B -m pytest tests --collect-only -q -p no:cacheprovider --basetemp results/logs/phase_f6m_pytest_tmp_collect`: `1954 tests collected`.
- Broad validation limitations:
  - `cleanup_current and not unsafe_external` timed out after 1200 seconds and showed historical guard failures; isolated first concrete failure is `tests/test_phase_f6i2g_first_python_consolidation_execution.py`, where the old F6I2G protected-input guard reports `f6i2g_failed_protected_input_change` after intentional F6M closure edits.
  - `not cleanup_current and not unsafe_external` timed out after 1200 seconds with additional historical failures; first concrete failure under `-x` is `tests/test_exactness_frontier_and_evidence_closure.py::test_required_schemas_and_exact_row_counts`, caused by `ModuleNotFoundError: No module named 'src'` in the older direct script path `src/analysis/build_exactness_frontier_and_evidence_closure.py`.
  - These broad-suite failures are not F6M metric or LastFM evidence failures; they are retained as residual historical-suite risks for the final commit record.
- Scope boundary: no LLM/API calls, no recommender training, no downloads, no full LastFM extension, no response repair, no deduplication, and no `original_repo/` modification.

## extension_e1_complete_metric_contract (2026-08-17)

- Decision token prepared: `EXTENSION_E1_COMPLETE_METRIC_CONTRACT_FROZEN`.
- Scope: post-F6M extension design evidence only; no experiment, training, LLM/API call, dataset download, preprocessing, raw-output repair, metric behavior edit, historical result rewrite, or `original_repo/` modification.
- Source inspection covered the local paper PDF hash `A143265AE42F027C079766B1C006C438D698897B732634D78A010EB0A62F0F23`, arXiv PDF text for formula recovery, official repository metric/prompt scripts, and local F3-F6M evidence artifacts.
- Generated artifacts: `report/extension_e1_complete_metric_contract.md`, `results/tables/extension_e1_metric_contract.csv`, `results/tables/extension_e1_four_dataset_coverage.csv`, `results/tables/extension_e1_fifth_dataset_requirements.csv`, `results/tables/extension_e1_inference_requirements.csv`, and `results/tables/extension_e1_dataset_selection_scorecard_template.csv`.
- Added generator: `src/analysis/extension_e1_metric_contract.py`.
- Added focused validation: `tests/test_extension_e1_metric_contract.py`.
- Key scientific boundaries: paper ItemCoverage is candidate-union coverage, not catalog coverage; paper hallucination is free-text/title matching and not local JSON/index strict validity; ILD remains local/adapted and must not replace paper OIC/Shannon diversity; calibration and candidate-recall metrics remain extension-only.
- Known unresolved details are explicitly recorded rather than reconciled silently: PopREO zero-denominator policy, Gini paper/code denominator conflict, paper hallucination indicator/prose mismatch, K-S gate alpha/metric-vector policy, and paper-only naming of the third diversity metric.
- Next step after validation: perform E2 fifth-dataset audit and selection against the frozen metric contract.

## extension_e2_fifth_dataset_selection (2026-08-17)

- Decision token prepared: `EXTENSION_E2_FIFTH_DATASET_SELECTED`.
- Scope: post-F6M extension design evidence only; no E3 acquisition, dataset download, preprocessing, training, LLM inference, API call, metric implementation change, frozen E1 artifact edit, or `original_repo/` modification.
- Recovery state before E2: branch `master`, HEAD `07e239ce1ede2807a40ea3064554febcf941e1db`, main worktree clean, `master...origin/master [ahead 5]`, nested `original_repo/EvaLLMasRecommender` clean on `main...origin/main`. No interrupted `extension_e2_*` artifact was found.
- Frozen E1 counts reused and confirmed: metric contract `29`, four-dataset coverage `26`, fifth-dataset requirements `23`, inference requirements `8`, scorecard criteria `25`.
- Candidate shortlist evaluated: Goodreads, BookCrossing / Book-Crossing, Yelp Open Dataset, Steam reviews/interactions, Food.com, and Google Restaurants / Google Local restaurant release.
- Primary dataset selected: `Food.com`, exact release `Food.com Recipes and Interactions, published 2019 Version v2`, using `RAW_recipes.csv`, `RAW_interactions.csv`, and official train/validation/test split files after authorized E3 acquisition.
- Backup dataset selected: `Steam`, exact release `UCSD Steam Video Game and Bundle Data`, using Version 2 review data plus item metadata after authorized E3 acquisition.
- Generated artifacts: `report/extension_e2_fifth_dataset_selection.md`, `results/tables/extension_e2_dataset_candidates.csv`, `results/tables/extension_e2_metric_compatibility.csv`, `results/tables/extension_e2_model_compatibility.csv`, `results/tables/extension_e2_dataset_scorecard.csv`, `results/tables/extension_e2_dataset_hard_gates.csv`, `results/tables/extension_e2_dataset_pareto.csv`, `results/tables/extension_e2_dataset_compute_estimates.csv`, and `results/tables/extension_e2_dataset_selection_decision.csv`.
- Added generator: `src/analysis/extension_e2_fifth_dataset_selection.py`.
- Added focused validation: `tests/test_extension_e2_fifth_dataset_selection.py`.
- Updated `.gitignore` with a narrow `results/tables/extension_e2_*.csv` exception so required E2 evidence tables are commit-visible like the frozen E1 tables.
- Commands run: `.\.venv_consolidation_candidate\Scripts\python.exe -B src\analysis\extension_e2_fifth_dataset_selection.py`; `.\.venv_consolidation_candidate\Scripts\python.exe -B -m pytest tests/test_extension_e2_fifth_dataset_selection.py -q -p no:cacheprovider --basetemp results/logs/extension_e2_pytest_tmp`.
- Tests passed: `6 passed`.
- Key scientific boundaries: Food.com can receive the complete applicable paper/local-reproduction evaluation before local extensions, but E3 must still document manual intake, checksums, exact user/history counts, duplicate policy, and no raw-data commit. All candidate pools, traditional outputs, LLM outputs, parser manifests, and extension artifacts remain future work.
- Next recommended action: perform E3 acquisition, preprocessing design, and integrity freeze for the selected fifth dataset. Do not execute E3 until explicitly authorized.

## extension_e3_foodcom_data_integrity implementation (2026-08-17)

- Decision token from real workspace run: `EXTENSION_E3_FOODCOM_ACQUISITION_BLOCKED`.
- Scope: E3 implementation and blocker evidence only; no dataset download, no Kaggle credentials/tooling, no E3 substitution dataset, no preprocessing of real Food.com data, no training, no model scoring, no LLM/API calls, no candidate-pool experiment, and no `original_repo/` modification.
- Added manual-staging config: `configs/foodcom_extension.yaml`.
- Added reusable E3 runner: `src/data/prepare_foodcom_extension.py`.
- Added targeted synthetic validation: `tests/test_extension_e3_foodcom_data_integrity.py`.
- Updated `.gitignore` so `data/processed/foodcom/` remains local-only and `results/tables/extension_e3_foodcom_*.csv` evidence tables are commit-visible.
- Real workspace acquisition state: required manual source directory `data/raw/foodcom/source/` was absent, so E3 stopped at acquisition without downloading or substituting data.
- Generated blocker evidence: `report/extension_e3_foodcom_data_integrity.md` and `results/tables/extension_e3_foodcom_*.csv`.
- Commands run: `PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 .\.venv_consolidation_candidate\Scripts\python.exe -B -m pytest tests/test_extension_e3_foodcom_data_integrity.py tests/test_config.py tests/test_schemas.py -q -p no:cacheprovider --basetemp results/logs/extension_e3_foodcom_tmp`; `PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 .\.venv_consolidation_candidate\Scripts\python.exe -B src\data\prepare_foodcom_extension.py --config configs\foodcom_extension.yaml`.
- Tests passed: `23 passed`.
- Next recommended action: manually stage the official UCSD-linked Kaggle archive `shuyangli94/food-com-recipes-and-user-interactions` under `data/raw/foodcom/source/`, then rerun E3.
## R2 official metric calculability audit

- Decision: `REPRODUCTION_R2_COMPLETE_METRIC_CALCULABILITY_AUDIT_COMPLETE`.
- Commands: `python src/analysis/reproduction_r2_official_metric_calculability.py --project-root .`.
- Tests: `tests/test_reproduction_r2_official_metric_calculability.py` passed `12 passed`; R1/E1 regression scope passed `9 passed` with repo-local `--basetemp`.
- Outputs: `results/tables/reproduction_r2_official_metric_calculability_master.csv`, `results/tables/reproduction_r2_official_metric_calculability_summary.csv`, `report/reproduction_r2_complete_official_metric_calculability.md`.
- Cells classified: 1891; UNKNOWN/UNINVESTIGATED/UNCLASSIFIED = 0/0/0.
- Known issues: paper values from `paper_atomic_contract.csv` retain their secondary structured source status; official data tree remains absent.
- Next step: R3 local-artifact metric calculability and reconstruction audit, without new inference.
## R3 complete local metric and beyond-utility audit

- Decision: `REPRODUCTION_R3_COMPLETE_LOCAL_METRIC_AUDIT_COMPLETE`.
- Commands: `python src/analysis/reproduction_r3_complete_local_metric_audit.py --project-root .`.
- Outputs: `results/tables/reproduction_r3_all_metric_cells_master.csv`, `results/tables/reproduction_r3_beyond_utility_master.csv`, `report/reproduction_r3_complete_local_metric_and_beyond_utility_audit.md`.
- Cells classified: 17042; UNKNOWN/UNINVESTIGATED/UNCLASSIFIED = 0/0/0.
- Known issues: R3 does not execute new missing cells; candidate-position diagnostics and constrained-output hallucination remain separated from paper-style claims.
- Next step: design the smallest controlled recovery phase for unresolved beyond-utility gaps.
## R4 three-way Beyond-Utility reconciliation

- Decision: `REPRODUCTION_R4_THREE_WAY_RECONCILIATION_COMPLETE`.
- Commands: `python src/experiments/build_reproduction_r4_three_way_reconciliation.py --config configs/reproduction_r4_three_way_reconciliation.yaml`.
- Outputs: `results/tables/reproduction_r4_three_way_reconciliation_master.csv`, `results/tables/reproduction_r4_gap_scientific_priority.csv`, `results/tables/reproduction_r4_recovery_request_count_validation.csv`, `report/reproduction_r4_three_way_beyond_utility_reconciliation.md`.
- Cells reconciled: 7834; R3 true gaps prioritized: 54.
- Recovery count status: `R4_RECOVERY_COUNT_STATUS = ESTIMATED_NOT_YET_DEDUPLICATED`; total proposed unique new requests `4400` is a planning estimate, not request-manifest proof.
- ML-1M CandDif: `ML1M_CANDDIF_P0_STATUS = IN_SCOPE_ALREADY_ACCOUNTED`; diagnostic rows exist, no paper-compatible local CandDif is promoted, and remaining CandDif gaps are `0`.
- Known issues: R4 plans but does not execute targeted recovery; global path remains protocol-limited.
- Next step: design the smallest controlled R5 recovery execution for P0/P1 only, if approved.

## R5A controlled P0/P1 recovery manifest

- Decision: `REPRODUCTION_R5A_RECOVERY_MANIFEST_COMPLETE`.
- Scope: deterministic manifest and validation only; no LLM/Ollama/API calls, training, downloads, Food.com work, R5 execution, or `original_repo/` modification.
- Outputs: `results/tables/reproduction_r5a_*.csv` and `report/reproduction_r5a_controlled_recovery_manifest.md`.
- Exact requests: raw `6000`, duplicates `1200`, unique new future requests `4800`.
- Generation split: profile-generation prerequisites `400`, downstream recommendation requests `800`, total generation-family requests `1200`.
- R4 `4400` remains a planning estimate, not request-manifest proof. R5A exact count includes ML-1M/LastFM profile prerequisites: delta `+400` (`9.09%`).
- Profile hash semantics: existing Beauty/Sports profile inputs have available hashes; ML-1M/LastFM future profile outputs use dependency request IDs and do not fabricate output hashes before inference.
- Manifest SHA256: `290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD`.
- Next step: execute R5 recovery only after explicit approval and only from the frozen manifest.

## R5B controlled CandDif recovery

- Decision: `REPRODUCTION_R5B_CANDDIF_RECOVERY_COMPLETE`.
- Scope: executed only frozen R5A `CANDDIF_RECOVERY` requests; no History/Generation recovery, ML-1M CandDif, external APIs, training, downloads, Food.com work, or `original_repo/` modification.
- Manifest SHA256: `290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD`.
- Outputs: `results/recovery/r5/*.json`, `results/tables/reproduction_r5b_*.csv`, and `report/reproduction_r5b_canddif_recovery.md`.
- Request accounting: total `1200`, succeeded `1200`, backend failed `0`, not executed `0`.
- Parser/metric policy: strict JSON index parser; invalid outputs are preserved and score zero under `FAIL_STRICT_ALL_MANIFEST_REQUESTS_INVALID_ZERO`.
- Validation: `20 PASS / 0 FAIL`.
- Evidence class: `PAPER_ALIGNED_ADAPTED_RECOVERY`; global status remains `CRITICAL_EXPERIMENTAL_PATH_PROTOCOL_LIMITED`.

## R5B forensic CandDif audit

- Decision: `R5B_MULTIPLE_PROTOCOL_FAILURES`.
- Scope: offline forensic audit only; no Ollama/API calls, no R5C execution, no raw output edits, no official R5B CSV overwrite, no commit, and no `original_repo/` modification.
- Outputs: `results/diagnostics/r5b/*.csv`, `results/diagnostics/r5b/*.jsonl`, `results/diagnostics/r5b/forensic_summary.json`, and `report/reproduction_r5b_forensic_audit.md`.
- Main finding: execution integrity passes, but the prompt never says `exactly 5`; `1132/1200` outputs are wrong length, all with `done_reason=stop`, and Qwen mostly returns near/full candidate-set rankings.
- Additional finding: `16` parse failures are incomplete Ollama responses with `done=false` that the runner accepted as `SUCCEEDED`.
- Metric reconstruction: official R5B HR/NDCG and all six CandDif values reproduce exactly under invalid-as-zero, but the results are not freeze-defensible scientifically.

## R5B corrected CandDif recovery

- Decision: `R5B_CORRECTED_RECOVERY_COMPLETE`.
- Manifest SHA256: `290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD`.
- Protocol hash: `9596F36E1B1DBC15B68679DDF32114E6AE09914B418FB31DC112072384B708D5`.
- Scope: new `R5B_CORRECTED` identity over the same 1200 frozen R5A CandDif rows; no R5C execution, commit, push, external API, download, training, or `original_repo/` modification.
- Outputs: `results/recovery/r5b_corrected/`, `results/diagnostics/r5b_corrected/`, `results/tables/reproduction_r5b_corrected_*.csv`, and `report/reproduction_r5b_corrected.md`.
- Request accounting: expected `1200`, accounted `1200`, valid `1200`, invalid `0`.
- Backend incomplete handling: incomplete Ollama responses are classified as `BACKEND_INCOMPLETE`, not `SUCCEEDED`.

## R5C History-Length recovery

- Decision: `R5C_HISTORY_LENGTH_RECOVERY_COMPLETE`.
- Manifest SHA256: `290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD`.
- Protocol hash: `30E4BCA9E1456EE9A7DE74A09FE03BF1A26EFCF758915052FD88F39AF9A836F2`.
- Scope: new `R5C_HISTORY_LENGTH` identity over 2400 frozen R5A `HISTORY_LENGTH_RECOVERY` rows; no CandDif, Generation, R5D, external API, download, training, push, or `original_repo/` modification.
- Outputs: `results/recovery/r5c_history_length/`, `results/diagnostics/r5c_history_length/`, `results/tables/reproduction_r5c_history_length_*.csv`, and `report/reproduction_r5c_history_length_recovery.md`.
- Request accounting: expected `2400`, accounted `2400`, valid `2385`, invalid `15`.
- Backend incomplete handling: incomplete Ollama responses are classified as `BACKEND_INCOMPLETE`, not `SUCCEEDED`.

## R5C invalid output forensic audit

- Decision: `R5C_SELECTIVE_RERUN_REQUIRED`.
- Scope: offline-only audit of 15 Qwen2/ML-1M R5C `OUT_OF_RANGE / unknown_indices` outputs; no LLM/Ollama calls, raw edits, R5C rerun, R5D, commit, push, or `original_repo/` modification.
- Case classification: exact offline mappable `15`, partial/ambiguous `0`, genuine invalid `0`.
- Scientific decision: item-ID to candidate-index mapping is deterministic but not allowed by the frozen R5C candidate-index protocol, so selective rerun is required.
- Outputs: `results/tables/reproduction_r5c_invalid_output_forensic_audit.csv`, `results/tables/reproduction_r5c_invalid_output_value_mapping.csv`, `results/tables/reproduction_r5c_selective_rerun_rows.csv`, `results/tables/reproduction_r5c_invalid_output_protocol_assessment.csv`, `results/diagnostics/r5c_history_length/invalid_output_forensic_summary.json`, and `report/reproduction_r5c_invalid_output_forensic_audit.md`.

## R5C selective rerun hard preflight

- Decision: `R5C_SELECTIVE_RERUN_NOT_ALLOWED_UNDER_FROZEN_PROTOCOL`.
- Scope: offline-only hard preflight for the proposed 15-row Qwen2/ML-1M selective rerun; no LLM/Ollama calls, canary, row execution, raw edits, R5D, commit, push, or `original_repo/` modification.
- Scientific decision: selective rerun under the same R5C protocol hash is not allowed because R5C froze `max_retries=0`, preserved terminal outputs, and did not predeclare invalid-only replacement semantics.
- Required remediation for an all-valid freeze: full R5C rerun under a new predeclared protocol identity; the 15-row file remains diagnostic-only under the current freeze.
- Outputs: `results/tables/reproduction_r5c_selective_rerun_hard_preflight.csv`, `results/diagnostics/r5c_history_length/selective_rerun_preflight_summary.json`, and `report/reproduction_r5c_selective_rerun_preflight.md`.

## R5C_CORRECTED History-Length recovery

- Decision: `R5C_CORRECTED_HISTORY_LENGTH_COMPLETE`.
- Manifest SHA256: `290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD`.
- Protocol hash: `3AAD4945AF443A7F50846BFD044C54FEFD5B77EDBD3725B1D1F1C62D829ECDEE`.
- Scope: new `R5C_CORRECTED` identity over the same 2400 frozen R5A `HISTORY_LENGTH_RECOVERY` rows; original R5C raw outputs are preserved.
- Fix: prompt output contract now explicitly requires candidate positions `1..N`, not dataset item IDs; parser/validator, candidate sets, histories, targets, seed policy, metrics, Ollama policy, and `max_retries=0` are preserved.
- Outputs: `results/recovery/r5c_history_length_corrected/`, `results/diagnostics/r5c_history_length_corrected/`, `results/tables/reproduction_r5c_history_length_corrected_*.csv`, and `report/reproduction_r5c_history_length_corrected.md`.
- Request accounting: expected `2400`, accounted `2400`, valid `2400`, invalid `0`.
- R5D was not executed.

## R5C_CORRECTED post-run validation

- Decision: `R5C_CORRECTED_HISTORY_LENGTH_COMPLETE`.
- Scope: offline post-run validation only; no LLM/Ollama calls, raw edits, R5D, commit, push, or `original_repo/` modification.
- Corrected protocol hash: `3AAD4945AF443A7F50846BFD044C54FEFD5B77EDBD3725B1D1F1C62D829ECDEE`.
- Final accounting: `2400/2400` valid, invalid `0`, out_of_range `0`.
- Qwen2 x ML-1M: `300/300` valid; original item-ID substitution failure eliminated: `True`.
- Outputs: `results/tables/reproduction_r5c_history_length_corrected_original_comparison.csv` and `results/diagnostics/r5c_history_length_corrected/postrun_validation_summary.json`.

## R5D Generation-Involved recovery

- Decision: `R5D_EXECUTION_FAILED`.
- Manifest SHA256: `290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD`.
- Protocol hash: `AA76CD3E936A969A57CDA2E9E54E22B73D6C1D3DB65C0795F86E181CEDFE3B1A`.
- Scope: new `R5D_GENERATION_INVOLVED_RECOVERY_V2` identity over 1200 frozen R5A `GENERATION_INVOLVED_RECOVERY` rows.
- Two-stage policy: ML-1M/LastFM profile-generation rows are executed before dependent downstream profile-plus-history ranking rows; Beauty/Sports reuse existing F6L profile context.
- Output contract: profile rows require strict short natural-language `user_profile` JSON; ranking rows require exactly 10 unique 1-based candidate positions, not item IDs.
- Outputs: `results/recovery/r5d_generation_involved_v2/`, `results/diagnostics/r5d_generation_involved_v2/`, `results/tables/reproduction_r5d_generation_involved_v2_*.csv`, and `report/reproduction_r5d_generation_involved_v2_recovery.md`.
- Request accounting: expected `1200`, accounted `600`, valid `597`, invalid `3`.
- No R5D successor phase was executed.

## R5D frozen-protocol completion

- Decision: `R5D_FROZEN_PROTOCOL_COMPLETION_VALID`.
- Protocol hash: `AA76CD3E936A969A57CDA2E9E54E22B73D6C1D3DB65C0795F86E181CEDFE3B1A`.
- Scope: resumed only the `600` R5D rows still marked `NOT_EXECUTED`; the 3 prior invalid terminal outputs and 597 prior valid terminal outputs were not reexecuted.
- Invalid policy: `FAIL_STRICT_ALL_MANIFEST_REQUESTS_INVALID_ZERO`; terminal invalid ranking outputs remain in the manifest denominator and receive zero HR/NDCG credit.
- Outputs: `results/recovery/r5d_generation_involved_v2/`, `results/diagnostics/r5d_generation_involved_v2/frozen_resume_*.csv`, `results/tables/reproduction_r5d_generation_involved_v2_*.csv`, `results/manifests/reproduction_r5d_generation_involved_v2_frozen_completion_candidate.json`, and `report/reproduction_r5d_generation_involved_v2_frozen_completion.md`.
- Request accounting: expected `1200`, accounted `1200`, valid `1195`, invalid `5`, retry rows `0`.
- Target gaps accounted under invalid policy: `4/4`; all-valid diagnostic: `2/4`.
- No R5D successor phase was executed.

## Final paper/official/local scientific reconciliation

- Decision: `LOCAL_ADAPTED_REPRODUCTION_COMPLETE_NOT_PAPER_EXACT`.
- Scope: offline final audit over R5B corrected, R5C corrected, and R5D frozen artifacts only; no LLM/Ollama/API calls and no raw output edits.
- Final accounting: `4800/4800` accounted, `4795/4800` valid, `5/4800` terminal invalid, retry rows `0`.
- Frozen commits: R5B `83b46454866ea61f3e05eec81d555be53e4c41e8`, R5C `98a93d9892d8e9b21c9cdfa981e0ebea0a987500`, R5D `ba1dbe8367216f7ea1044b2ff8db69318f8678ee`.
- Classification summary: paper-exact `0`, paper-aligned adapted `11`, directional-only `16`, divergent `1`, no-direct-comparison `16`.
- Final report: `report/reproduction_final_scientific_reconciliation.md`.

## F1B traditional recommender completion gate

- Decision: `F1B_TRADITIONAL_COMPLETION_BLOCKED`.
- Scope: five-dataset F0 traditional recommender surface over `Random`, `MostPop`, `BM25-like/content`, `BPRMF`, `GRU4Rec`, `SASRec`, and `LightGCN`.
- Finding: protocol-identical reusable traditional outputs are `0`; trainable recommenders are blocked before execution by unresolved seed/config lineage across original-paper seeds `0-4`, local adapted seeds `42-46`, and superseded Plan E single-seed cohorts.
- Outputs: `results/tables/f1b_*.csv`, `results/manifests/f1b_training_provenance.json`, `results/manifests/f1b_no_llm_inference.json`, and `report/f1b_traditional_model_completion.md`.
- Execution: no LLM/Ollama/API calls, no downloads, no network, no recommender training, no first-stage rankings, no reranking requests unlocked.

## F1BR traditional protocol recovery

- Decision: `F1BR_TRADITIONAL_PROTOCOL_RESOLVED`.
- Scope: protocol recovery only; no deterministic baseline execution, recommender training, LLM/Ollama/API calls, downloads, network, or commit.
- ReChorus: original repo gitlink recovered as `fb646d4eed13a8d81ae308965b08c816f40d1a57`; local ReChorus source/defaults are unavailable and not fabricated.
- Resolution: original-four model hyperparameters frozen from Appendix A.3; project-wide final seeds frozen as `[42,43,44,45,46]` with `LOCAL_REPRODUCIBLE_FIVE_SEED_POLICY`, not paper seed identity.
- Food.com: validation-only hyperparameter search frozen from original-four paper values, tuning seed `42`, selection metric `validation_ndcg_at_5`, no final-test tuning.
- Readiness: `15/15` deterministic rows `READY_TO_EXECUTE`; `20/20` trainable dataset-model rows `READY_TO_TRAIN`; exact full training budget `80` original-four final + `20` Food.com final + `40` Food.com tuning = `140`.

## 2026-08-31T08:23:28+00:00 - G1 primary ranking execution

- Decision: `G1_PRIMARY_RANKING_BLOCKED`.
- Invocation mode: `validate-only`.
- Planned canonical ranking requests: `29292`.
- G0 reused ranking requests: `36`.
- New G1 terminal requests: `0` / `29256`.
- Ollama calls in this invocation: `0`.
- No training, tuning, reranking, history-grid, CandDif, profile, pool-size sensitivity, external API, download, network download, commit, reset, clean, or checkout was requested by this phase.
- Added `configs/g1_primary_ranking_execution.yaml`, `src/extension/g1_primary_ranking_execution.py`, `tests/test_extension_g1_primary_ranking_execution.py`, and blocked-state report `report/g1_primary_ranking_execution.md`.
- Validate-only command: `.venv_audit_recovery\Scripts\python.exe src\extension\g1_primary_ranking_execution.py --config configs\g1_primary_ranking_execution.yaml --validate-only`; result `G1_PRIMARY_RANKING_BLOCKED`, `29292` planned, `36` G0 reused, `0` new terminal, `29256` remaining.
- Focused G1 tests: `.venv_audit_recovery\Scripts\python.exe -m pytest tests\test_extension_g1_primary_ranking_execution.py -q`; result `10 passed`, with a pytest cache permission warning only.
- Requested focused suite: `.venv_audit_recovery\Scripts\python.exe -m pytest tests\test_extension_g1_primary_ranking_execution.py tests\test_extension_g0_canonical_llm_canary.py tests\test_extension_f0_final_extension_freeze.py tests\test_extension_f1a_offline_metric_closure.py tests\test_extension_f1arr_popreo_dpd_recovery.py -q`; result `41 passed, 1 failed`. The failed row is the historical F0 live precondition `current_head_is_pre_f0_checkpoint`, because F0 expects `30d6c3e5fcb0e97451a4420829b88de10bb57213` while this G1 task runs at `5dfec10537a8acd1404c586323a3935add48bc2a`; tracked F0 budget/primary artifacts were unchanged.
- Known blockers before full G1 execution: worktree is not clean because the G1 implementation is uncommitted, local Ollama model preflight returned the three required models as unavailable, and the `29256` new local calls have not been executed. Next step is to review/commit or otherwise freeze the G1 implementation, make the exact G0/F0 local model digests available in Ollama, then resume with `--allow-api-calls` and optional `--max-new-calls`.

## 2026-08-31T08:45:53+00:00 - G1P local Ollama preflight recovery

- Decision: `G1P_LOCAL_PREFLIGHT_VALIDATED`.
- Root cause: G1 originally reused the G0 metadata helper that collapses any `/api/tags` metadata failure into `models=[]`, which was then reported as per-model `BLOCKED_MISSING_LOCAL_MODEL`; the implementation did not distinguish Ollama unreachable, metadata prohibited, model not installed, digest mismatch, and exact digest match.
- Fix: G1 now has loopback-only metadata preflight with explicit states `OLLAMA_UNREACHABLE`, `LOCAL_METADATA_CHECK_NOT_PERMITTED`, `MODEL_NOT_INSTALLED`, `MODEL_DIGEST_MISMATCH`, and `MODEL_AVAILABLE_MATCHED`; non-loopback endpoints and non-loopback redirects are rejected.
- Live validate-only model state from `results/tables/g1p_local_ollama_preflight.csv`: qwen2 `MODEL_AVAILABLE_MATCHED`, mistral `MODEL_AVAILABLE_MATCHED`, gemma2 `MODEL_AVAILABLE_MATCHED`; all current local digests match frozen G0 provenance.
- Validate-only command: `.venv_audit_recovery\Scripts\python.exe src\extension\g1_primary_ranking_execution.py --config configs\g1_primary_ranking_execution.yaml --validate-only`; result `G1_PRIMARY_RANKING_BLOCKED`, `29292` planned, `36` G0 reused, `0` new terminal, `29256` remaining, `0` generation calls in invocation.
- G1P tests: `.venv_audit_recovery\Scripts\python.exe -m pytest tests\test_extension_g1_primary_ranking_execution.py -q`; result `16 passed`, with a pytest cache permission warning only.
- Requested focused suite after G1P: `.venv_audit_recovery\Scripts\python.exe -m pytest tests\test_extension_g1_primary_ranking_execution.py tests\test_extension_g0_canonical_llm_canary.py tests\test_extension_f0_final_extension_freeze.py tests\test_extension_f1a_offline_metric_closure.py tests\test_extension_f1arr_popreo_dpd_recovery.py -q`; result `47 passed, 1 failed`. The sole failure remains `HISTORICAL_SNAPSHOT_NOT_CURRENT_G1_GATE` for F0 expecting `30d6c3e5fcb0e97451a4420829b88de10bb57213` while G1P runs at `5dfec10537a8acd1404c586323a3935add48bc2a`.
- Static G1 invariants after preflight fix: request universe `29292`, experiment cells `30`, G0 ranking reuse `36/36`, remaining new G1 calls `29256`, request-universe frame hash `84d55fbc3f7307f1b2a255d3ea588d09cf38a154ea199c4eaaed69a53fc5af8c`.
- Protected paths unchanged by git status: no `original_repo/`, `paper/`, `data/raw/`, `results/checkpoints/`, `results/models/`, G0 raw response, or F1BZQ authoritative pool modifications were observed.

## 2026-08-31T09:04:29+00:00 - G1F runner implementation freeze review

- Decision before commit: G1 runner code/config/tests are ready for explicit freeze; full G1 inference remains unstarted.
- HEAD semantics fixed: `configs/g1_primary_ranking_execution.yaml` now records `expected_base_checkpoint = 5dfec10537a8acd1404c586323a3935add48bc2a`; runtime validation records the current execution commit and requires the base checkpoint to be an ancestor rather than requiring current HEAD equality with the G0 base.
- Provenance added: G1 manifests now record `scientific_base_checkpoint`, `execution_code_commit`, `base_checkpoint_is_ancestor`, `working_tree_clean`, config hash, runner source hash, test source hash, and relevant frame hashes.
- Scheduler fixed: canonical logical order remains `dataset_user_prompt_model`; physical execution order is persisted separately as `model_dataset_prompt_user` in `results/tables/g1_primary_ranking_physical_execution_order.csv` and is used only for local call scheduling.
- Validate-only command: `.venv_audit_recovery\Scripts\python.exe src\extension\g1_primary_ranking_execution.py --config configs\g1_primary_ranking_execution.yaml --validate-only`; result `G1_PRIMARY_RANKING_BLOCKED`, `29292` planned, `36` G0 reused, `0` new terminal, `29256` remaining, `0` generation calls.
- Focused G1/G1P/G1F tests: `.venv_audit_recovery\Scripts\python.exe -m pytest tests\test_extension_g1_primary_ranking_execution.py -q`; result `20 passed`, with a pytest cache permission warning only.
- Protected paths unchanged by git status: no `original_repo/`, `paper/`, `data/raw/`, `results/checkpoints/`, `results/models/`, G0 raw response, or F1BZQ authoritative pool modifications were observed.

## 2026-08-31T08:45:58+00:00 - G1 primary ranking execution

- Decision: `G1_PRIMARY_RANKING_BLOCKED`.
- Invocation mode: `validate-only`.
- Planned canonical ranking requests: `29292`.
- G0 reused ranking requests: `36`.
- New G1 terminal requests: `0` / `29256`.
- Ollama calls in this invocation: `0`.
- No training, tuning, reranking, history-grid, CandDif, profile, pool-size sensitivity, external API, download, network download, commit, reset, clean, or checkout was requested by this phase.

## 2026-08-31T09:02:31+00:00 - G1 primary ranking execution

- Decision: `G1_PRIMARY_RANKING_BLOCKED`.
- Invocation mode: `validate-only`.
- Planned canonical ranking requests: `29292`.
- G0 reused ranking requests: `36`.
- New G1 terminal requests: `0` / `29256`.
- Ollama calls in this invocation: `0`.
- No training, tuning, reranking, history-grid, CandDif, profile, pool-size sensitivity, external API, download, network download, commit, reset, clean, or checkout was requested by this phase.

## 2026-08-31T09:04:37+00:00 - G1 primary ranking execution

- Decision: `G1_PRIMARY_RANKING_BLOCKED`.
- Invocation mode: `validate-only`.
- Planned canonical ranking requests: `29292`.
- G0 reused ranking requests: `36`.
- New G1 terminal requests: `0` / `29256`.
- Ollama calls in this invocation: `0`.
- No training, tuning, reranking, history-grid, CandDif, profile, pool-size sensitivity, external API, download, network download, commit, reset, clean, or checkout was requested by this phase.

## 2026-09-01T09:13:09+00:00 - G1 primary ranking execution

- Decision: `G1_PRIMARY_RANKING_VALIDATED`.
- Invocation mode: `execute-or-resume`.
- Planned canonical ranking requests: `29292`.
- G0 reused ranking requests: `36`.
- New G1 terminal requests: `29256` / `29256`.
- Ollama calls in this invocation: `0`.
- No training, tuning, reranking, history-grid, CandDif, profile, pool-size sensitivity, external API, download, network download, commit, reset, clean, or checkout was requested by this phase.

## 2026-09-01T11:20:28+00:00 - G1R scientific retention and external archive freeze

- Decision: `G1_RETENTION_VALIDATED`.
- Scope: retention/provenance only; no Ollama, inference, metric recomputation, G1 protocol edits, prompt edits, parser edits, raw response edits, downloads, training, tuning, commit, reset, clean, or checkout.
- Updated `.gitignore` with narrow exceptions for G1 foreground tables and G1 provenance manifests, plus an ignored `archive_external/` location for external evidence bundles.
- Created retention manifest `results/manifests/g1_primary_ranking_retention_manifest.json`, recording clone-readable foreground artifacts, externalized raw/per-request evidence, hash invariance, and archive recovery metadata.
- Created deterministic external archive `archive_external/g1_primary_ranking/G1_primary_ranking_3d7f3add_external_evidence.zip`; SHA256 `1da06322b966afde096681189bdc923ff7a34288ff899f94f564595fdd11ab8a`; entries `29266`; uncompressed bytes `796950141`.
- Updated `docs/external_archive_guide.md` and `report/g1_primary_ranking_execution.md` with the G1 retention split and restore/verification instructions.

## 2026-09-01T11:51:53Z - G2 history-length sensitivity protocol freeze

- Decision: `G2_HISTORY_LENGTH_PROTOCOL_FROZEN`.
- Scope: forensic protocol design, G1 reuse audit, prior-evidence rejection audit, request accounting, and retention planning only.
- Primary paired cohort: `2033` users with at least `20` train-history interactions.
- Planned applicable G2 requests: `77550`; exact G1 terminal reuse: `16560`; future new Ollama calls required: `60990`.
- Food.com true-zero H>0 not-applicable user conditions: `21810`.
- No inference, Ollama metadata, remote API calls, downloads, training, tuning, metric recomputation, G1 protocol edits, commit, reset, clean, checkout, G3, G4, or G5 actions were run.

## 2026-09-01T12:30:08Z - G2 nested history-length sensitivity protocol refreeze

- Decision: `G2_NESTED_HISTORY_PROTOCOL_FROZEN`.
- Scope: nested Design B protocol update, G1 reuse audit, prior-evidence rejection audit, request accounting, and retention planning only.
- B1 primary paired cohort: `2383` users with at least `10` train-history interactions.
- B2 long-history paired cohort: `2033` users with at least `20` train-history interactions.
- Planned applicable G2 requests: `88050`; exact G1 terminal reuse: `18660`; future new Ollama calls required: `69390`.
- Food.com true-zero H>0 not-applicable user conditions: `21810`.
- No inference, Ollama metadata, remote API calls, downloads, training, tuning, metric recomputation, G1 protocol edits, commit, reset, clean, checkout, G3, G4, or G5 actions were run.

## 2026-09-01T16:22:59Z - G2 history-length execution runner freeze

- Decision: `G2_EXECUTION_RUNNER_FROZEN`.
- Mode: `dry-run`; no Ollama generation or metadata calls were executed.
- Logical applicable requests: `88050`; G1 reusable: `18660`; future new calls: `69390`.
- Existing G2 terminal requests: `0`; unresolved new requests: `69390`.
- Created pre-execution runner artifacts for physical order, execution accounting, and exact G1 execution reuse.
- No G2B inference, G3/G4/G5, metrics, downloads, training, tuning, commit, or G0/G1 artifact edits were performed.

## 2026-09-01T16:26:43Z - G2 history-length execution runner freeze

- Decision: `G2_EXECUTION_RUNNER_FROZEN`.
- Mode: `dry-run`; no Ollama generation or metadata calls were executed.
- Logical applicable requests: `88050`; G1 reusable: `18660`; future new calls: `69390`.
- Existing G2 terminal requests: `0`; unresolved new requests: `69390`.
- Created pre-execution runner artifacts for physical order, execution accounting, and exact G1 execution reuse.
- No G2B inference, G3/G4/G5, metrics, downloads, training, tuning, commit, or G0/G1 artifact edits were performed.

## 2026-09-03T14:17:11Z - G2 history-length execution retention freeze

- Decision: `G2_EXECUTION_RETENTION_FROZEN`.
- Scope: retention/provenance only; no new inference, no Ollama calls, no G3, no protocol changes, no metric interpretation, no downloads, training, tuning, reset, clean, checkout, or push.
- Final accounting: logical `88050`; G1 exact zero-call reuse `18660`; G2 backend calls `69390`; strict valid `68274`; terminal invalid evidence `1116`; backend errors `0`.
- Integrity: orphan `0`; raw duplicates `0`; terminal duplicates `0`; ambiguous evidence `0`; preserved `2` harmless duplicate pre-send resume request records as provenance.
- Created tracked G2 evidence inventory `results/tables/g2_history_length_execution_evidence_inventory.csv` and retention manifest `results/manifests/g2_history_length_execution_retention_manifest.json`.
- Created ignored external archive `archive_external/g2_history_length_execution/G2_history_length_8fb6dc9_external_evidence.zip` with SHA256 `8d8f6192435b3c996306f1d2a8305ef70e3cdf263a1e23b8d6b466dfb1272a46`.
- Next safe step: G2 metric analysis from frozen evidence; do not start G3 until G2 readout is complete.

## 2026-09-03T14:38:12Z - G2 analytic universe freeze

- Reconstructed `88050` frozen logical G2 requests from retained G1/G2 evidence without Ollama calls or G3.
- Canonical split: G1 exact reuse `18660`; G2 terminal evidence `69390`; strict-valid total `86768`; invalid total `1282`; backend errors `0`.
- Denominator contract frozen for phase 1: utility uses planned/applicable users with invalid terminal responses counted as zero utility; beyond-utility uses strict-valid users plus metric-specific defined subsets.
- Outputs: `results/tables/g2_history_length_analysis_universe.csv`, `results/tables/g2_history_length_analysis_membership_universe.csv`, `results/tables/g2_history_length_analysis_cell_counts.csv`, `results/tables/g2_history_length_denominator_contract.csv`, `results/manifests/g2_history_length_analysis_universe_freeze.json`.
- Next safe step: compute G2 metrics from the frozen analytic universe only; do not start G3.

g2_history_length_analysis_universe_freeze

## 2026-09-03T18:20:07Z - G2 metric freeze

- Decision: `G2_METRICS_FROZEN`.
- Scope: computed G2 metrics from frozen analytic universe only; no Ollama calls, no G3, no protocol changes, no evidence edits.
- Final accounting: logical `88050`; G1 reuse `18660`; G2 terminal evidence `69390`; strict-valid `86768`; invalid `1282`.
- Cells computed: `396`; metric rows `5940`; undefined/blocked rows `984`.
- Denominators: utility planned/applicable with invalid-as-zero; beyond-utility strict-valid with metric-specific defined subsets.
- Outputs: `results/tables/g2_history_length_metric_results.csv`, `results/tables/g2_history_length_metric_denominators.csv`, `results/tables/g2_history_length_metric_validation.csv`, `results/tables/g2_history_length_metric_undefined.csv`, `results/manifests/g2_history_length_metrics_freeze.json`.
- Next safe step: descriptive G2 readout/statistical analysis from frozen metric tables; do not start G3 yet.

## 2026-09-03T18:47:57Z - G2 scientific readout

- Decision: `G2_DESCRIPTIVE_READOUT_COMPLETE_STATISTICS_NOT_FROZEN`.
- Scope: descriptive readout from frozen G2 metric/universe artifacts only; no Ollama calls, no G3, no protocol changes.
- Metric values `5940`; deltas `5670`; patterns `1440`.
- Frozen statistical plan rows `27` identified but not executed from cell-level frozen metric tables.
- Outputs: `results/tables/g2_history_length_readout_*.csv`, `results/manifests/g2_history_length_scientific_readout_manifest.json`, `report/g2_history_length_scientific_readout.md`.
- Next safe step: decide whether to freeze paired-user metric contributions/statistical execution, or move to report drafting from descriptive tables; do not start G3 yet.
## 2026-09-03T19:22:40Z - G2 frozen statistical analysis

- Decision: `G2_STATISTICAL_ANALYSIS_FROZEN`.
- Scope: executed frozen G2 paired-user bootstrap CI plan from retained evidence only; no Ollama calls, no G3, no protocol changes, no frozen evidence/universe/metric/readout edits.
- Frozen comparison coverage: `27/27`; statistical result rows `810`; paired-user contribution rows `122382`.
- Bootstrap: `10000` resamples, seed `20260901`, two-sided 95 percent percentile CI, paired mean delta.
- Primary CI decision counts: `{"NOT_SIGNIFICANT_CI_INCLUDES_ZERO":499,"SIGNIFICANT_NEGATIVE":67,"SIGNIFICANT_POSITIVE":244}`; corrected p-value decisions remain `NOT_DEFINED_NO_FROZEN_P_VALUE_METHOD`.
- Outputs: `results/tables/g2_history_length_statistical_paired_contributions.csv`, `results/tables/g2_history_length_statistical_results.csv`, `results/tables/g2_history_length_statistical_validation.csv`, `results/manifests/g2_history_length_statistical_analysis_manifest.json`, `report/g2_history_length_statistical_analysis.md`.
- Next safe step: integrate frozen G2 descriptive/statistical readout into the report; do not start G3 yet.
## 2026-09-03T19:30:50Z - G2 final scientific synthesis

- Decision: `G2_SCIENTIFIC_SYNTHESIS_FROZEN`.
- Scope: synthesized frozen descriptive and statistical G2 artifacts only; no Ollama calls, no G3, no new tests, no p-values, no protocol/evidence/metric/readout changes.
- Frozen comparison templates summarized: `27/27`; synthesis comparison rows `27`.
- PRIMARY H10-H0 descriptive `80/1/9` improved/equal/worsened; CI-supported `57` positive, `1` negative, `32` non-significant.
- LONG H20-H10 descriptive `27/4/59` improved/equal/worsened; CI-supported `3` positive, `18` negative, `69` non-significant.
- Outputs: `results/tables/g2_history_length_scientific_synthesis_*.csv`, `results/manifests/g2_history_length_scientific_synthesis_manifest.json`, `report/g2_history_length_scientific_synthesis.md`.
- Next safe step: integrate frozen G2 synthesis into the report; no relation rewrite or G3 yet.

## 2026-09-04T01:57:26+02:00 - Full report audit and update through frozen G2

- Scope: audited and updated the LaTeX manuscript under `report/latex/Reproduction_Study_of___Beyond_Utility__Evaluating_LLM_as_Recommender` using frozen reproduction, G1, and G2 artifacts only.
- Updated title/front matter, abstract, introduction contribution framing, extension validity threats, Extension, Overall Conclusions, and appendices.
- G1 integration: 29,292 logical requests; 30/30 primary-ranking cells; 90/90 utility rows; 414/420 beyond-utility rows computed; reliability totals and runtime copied from frozen G1 artifacts.
- G2 integration: 88,050 logical requests; 18,660 exact G1 reuses; 69,390 G2 terminal/backend evidence rows; 86,768 strict-valid analytic rows; 1,282 invalid rows; zero backend errors, missing, duplicate, or ambiguous terminal evidence.
- G2 synthesis integrated: PRIMARY H10-H0 descriptive `80/1/9`, CI-supported `57/1/32`; LONG H20-H10 descriptive `27/4/59`, CI-supported `3/18/69`; no p-values invented.
- Static LaTeX checks: 13/13 inputs present, 190 labels with no duplicates, zero missing references, 205 citation uses with zero missing BibTeX keys, and stale-status scan clean except explicit rejected universal-history claims.
- Build status: local PDF compilation blocked because no TeX engine (`pdflatex`, `bibtex`, `latexmk`, `tectonic`, `xelatex`, or `lualatex`) is available on PATH.
- Audit record: `report/latex/FULL_REPORT_AUDIT.md`.
- No Ollama calls, no new scientific analysis, no frozen scientific artifact edits, and no G3 execution.

## 2026-09-04T06:58:46Z - G3 candidate-position FIRST/RANDOM protocol freeze

- Decision: `G3_BASE_LLMRANK_FIRST_RANDOM_PROTOCOL_FROZEN`; runner decision: `G3_EXECUTION_RUNNER_FROZEN`.
- Scope: BASE + LLMRANK, FIRST vs RANDOM, LLMSRec excluded, `5 x 3 x 2 x 200 x 2 = 12000` logical requests.
- Cohort rule: `first_200_users_by_sort_key_from_frozen_G1_primary_candidate_users_after_candidate_contract_validation`.
- RANDOM algorithm/seed: `sha256_seeded_sort_rotate_if_positive_first` / `R5A_CANDDIF_RANDOM_POSITION_V1`.
- Reuse: `0`; future new Ollama calls: `12000`.
- No Ollama/API calls, G3 execution, G1/G2 evidence edits, downloads, training, tuning, or metric interpretation were run.

## 2026-09-04T13:39:34Z - G3 execution retention freeze

- Decision: `G3_EXECUTION_RETENTION_FROZEN` from `G3_FULL_EXECUTION_COMPLETED_AND_RECONCILED`.
- Evidence: `12000` raw files, `12000` terminal responses, `12000` request ledger rows, `36000` event rows.
- Accounting: `FIRST=6000`, `RANDOM=6000`, strict valid `11834`, invalid `166`, backend errors `0`, retries `0`, remaining `0`.
- Integrity: orphan/missing/duplicate/ambiguous checks passed with zero defects.
- External archive: `archive_external/g3_candidate_position_execution/G3_candidate_position_06088e9_external_evidence.zip` SHA256 `605c075af4a602d914d527d2ae555367abab76dcae95ede4bdff59a3b033696e`.
- No Ollama/API calls, metric computation, statistical analysis, protocol/config/runner changes, or evidence rewrites were performed.

## 2026-09-04T13:52:05Z - G3 analytic universe freeze

- Decision: `G3_ANALYTIC_UNIVERSE_FROZEN` from frozen retained G3 evidence.
- Universe: `12000` logical rows and `6000` FIRST/RANDOM pairs.
- Evidence resolution: `12000` raw files, `12000` terminal rows, strict valid `11834`, invalid `166`.
- Pairing/missing/orphan/duplicate/ambiguous defects: `0`.
- No Ollama/API calls, metric computation, statistical analysis, protocol/config/runner edits, or evidence rewrites were performed.
## 2026-09-04T14:09:57Z - G3 candidate-position metrics freeze

- Computed frozen G3 utility metrics from `results/tables/g3_candidate_position_analysis_universe.csv` only.
- Utility rows: `180`; CandDif rows: `60`; reliability rows: `25`.
- Reliability: strict-valid `11834`, invalid `166`, backend errors `0`.
- Validation: `26/26` PASS.
- No Ollama/API calls, bootstrap tests, G3 statistics, or downstream phases were run.
- Next safe step: frozen G3 statistical analysis over the paired universe and frozen metric artifacts.
## 2026-09-04T14:18:31Z - G3 frozen statistical analysis

- Decision: `G3_STATISTICAL_ANALYSIS_FROZEN`.
- Scope: executed the frozen G3 FIRST-vs-RANDOM paired bootstrap CI plan from frozen paired universe and metric artifacts only.
- Coverage: `90/90` comparisons; paired-user contribution rows `18000`; preserved pairs `6000`.
- Bootstrap: `10000` resamples, seed `20260901`, two-sided 95 percent percentile CI, paired mean delta `FIRST_minus_RANDOM`.
- Primary CI decision counts: `{"CI_SUPPORTED_NEGATIVE":1,"CI_SUPPORTED_POSITIVE":85,"NON_SIGNIFICANT_CI_INCLUDES_ZERO":4}`.
- Multiplicity/Holm: `NOT_APPLIED_NO_FROZEN_P_VALUE_METHOD_FOR_G3`; no p-values or corrected decisions were invented.
- CandDif HR/NDCG retained as descriptive cell-level position-bias measures only.
- No Ollama/API calls, protocol changes, evidence rewrites, new tests, or downstream phases were run.
- Next safe step: frozen G3 scientific readout/synthesis from frozen metrics and statistical artifacts.
## 2026-09-04T15:44:15Z - G3 final scientific synthesis

- Decision: `G3_SCIENTIFIC_SYNTHESIS_FROZEN`.
- Scope: synthesized frozen G3 FIRST-vs-RANDOM metric/statistical artifacts only; no inference, no new tests, no relation update, no G4.
- FIRST-vs-RANDOM CI decisions: `{"CI_SUPPORTED_NEGATIVE":1,"CI_SUPPORTED_POSITIVE":85,"NON_SIGNIFICANT_CI_INCLUDES_ZERO":4}` over `90/90` comparisons.
- Prompt interaction is descriptive matched-cell magnitude comparison, not a new inferential test.
- CandDif HR@5/NDCG@5 retained as descriptive position-bias measures only; no CandDif significance test computed.
- Reliability: FIRST invalid `96`; RANDOM invalid `70`; no structural-validity significance test applied.
- Multiplicity: Holm not applied because no frozen G3 p-value method exists.
- Next safe step: update report through frozen G3 results only.

## 2026-09-05T09:42:41+02:00 - Full report update and audit through frozen G3

- Scope: updated the canonical LaTeX manuscript under `report/latex/Reproduction_Study_of___Beyond_Utility__Evaluating_LLM_as_Recommender` using frozen G3 artifacts only.
- Added G3 candidate-position/order sensitivity to the Extension chapter: FIRST vs RANDOM, 5 datasets x 3 models x 2 prompts x 200 users, 12,000 calls, 6,000 FIRST and 6,000 RANDOM.
- Integrated frozen G3 execution accounting: strict-valid `11834`, invalid `166`, backend errors `0`, FIRST invalid `96`, RANDOM invalid `70`.
- Integrated frozen G3 bootstrap summary: `90/90` comparisons; overall `85/1/4`; BASE `45/0/0`; LLMRANK `40/1/4`; Qwen2 `30/0/0`; Mistral `25/1/4`; Gemma2 `30/0/0`.
- Documented exact Mistral LLMRANK exceptions, descriptive-only `CandDif_HR@5`/`CandDif_NDCG@5`, no CandDif significance test, no invented p-values, and Holm not applied.
- Updated stale manuscript status text so G1/G2/G3 are frozen extension evidence and G4 remains unfinished.
- Static LaTeX checks: 13/13 inputs present, 193 labels with no duplicates, zero missing references, 205 citation uses with zero missing BibTeX keys, and stale-status scan limited to explicit G4/no-CandDif-sig scope statements.
- Build status: local PDF compilation blocked because no TeX engine (`pdflatex`, `bibtex`, `latexmk`, `tectonic`, `xelatex`, or `lualatex`) is available on PATH.
- Audit record updated: `report/latex/FULL_REPORT_AUDIT.md`.
- No Ollama/API calls, no new scientific analysis, no G4, and no frozen scientific artifact edits were performed.

## 2026-09-05T08:26:45+00:00 - G4 F1BZQ reranking protocol freeze

- Decision: `G4_F1BZQ_RERANKING_PROTOCOL_FROZEN`.
- Invocation mode: `zero-call dry-run freeze`.
- Logical requests: `29292`.
- Response reuse: `0`.
- Future Ollama calls: `29292`.
- Existing terminal G4 evidence: `0`.
- Ollama/API calls in this invocation: `0`.
- No G1/G2/G3 artifacts, F1BZQ pool artifacts, original_repo files, paper files, raw data, training outputs, or superseded reranking pools were modified.

## 2026-09-06T03:06:05+00:00 - G4 F1BZQ reranking protocol freeze

- Decision: `G4_F1BZQ_RERANKING_PROTOCOL_FROZEN`.
- Invocation mode: `execute`.
- Logical requests: `29292`.
- Response reuse: `0`.
- Future Ollama calls: `29292`.
- Existing terminal G4 evidence: `24727`.
- Ollama/API calls in this invocation: `4565`.
- No G1/G2/G3 artifacts, F1BZQ pool artifacts, original_repo files, paper files, raw data, training outputs, or superseded reranking pools were modified.


## G4 Execution Retention Freeze - 2026-09-06
- Completed retention freeze for `G4_FULL_EXECUTION_COMPLETED_AND_RECONCILED` with zero Ollama/API calls and no metric/statistical computation.
- Verified `29292` raw response files, `29292` terminal response rows, `30` cells, `28986` strict-valid responses, `306` structural invalids, and `0` backend errors.
- Preserved two recovered request-only pre-response orphan incidents via recovery manifests and external archive provenance.
- Generated `results/tables/g4_f1bzq_execution_evidence_inventory.csv`, `results/manifests/g4_f1bzq_execution_retention_manifest.json`, `report/g4_f1bzq_execution_retention_freeze.md`, and `archive_external/g4_f1bzq_reranking_execution/G4_f1bzq_reranking_df8700c_external_evidence.zip` with SHA256 `dc32bb824475be37f13616d9d7f5080b5f7da8de5ccba11ce78cfccd39089cd8`.
- Next safe step: compute G4 metrics from frozen retained evidence only when explicitly requested.

## 2026-09-06T07:03:57Z - G4 analytic universe freeze

- Decision: `G4_ANALYTIC_UNIVERSE_FROZEN` from frozen retained G4 evidence only.
- Universe: `29292` logical rows, `4882` users, and `30` dataset x model x prompt cells.
- Evidence resolution: `29292` raw files, `29292` terminal rows, strict-valid `28986`, invalid `306`, backend errors `0`.
- Candidate-stage denominators: planned users `4882`, positive-present `743`, positive-absent `4139`.
- No Ollama/API calls, metric computation, statistical analysis, protocol/config/runner edits, F1BZQ-pool edits, or evidence rewrites were performed.

## g4_f1bzq_metrics_freeze - 2026-09-06T07:26:32Z

- Computed G4 F1BZQ metrics from the frozen analytic universe only.
- No Ollama/API calls; no bootstrap/statistical tests; no frozen evidence/pool/protocol/runner/universe edits.
- Metric result rows: 630; undefined/blocked rows: 510.
- Candidate recall by dataset: {"beauty":"0.049494949494949494","foodcom":"0.0040000000000000001","lastfm":"0.46899999999999997","ml1m":"0.193","sports":"0.031390134529147982"}.
- Validation passed: true.
- Next safe step: freeze G4 bootstrap/statistical comparison protocol and run statistics from frozen metric artifacts only.

## g4_f1bzq_statistical_analysis - 2026-09-06T07:49:30Z

- Executed the frozen G4 F1BZQ statistical comparison plan offline.
- No Ollama/API calls; no p-values; Holm not applied; no frozen evidence/protocol/universe/metric edits.
- Comparison templates covered: 75/75.
- Paired contribution rows: 145188; statistical result rows: 315.
- CI decision counts: {"CI_SUPPORTED_NEGATIVE":24,"CI_SUPPORTED_POSITIVE":29,"NON_SIGNIFICANT_CI_INCLUDES_ZERO":172}.
- Validation passed: true.
- Next safe step: perform final G4 scientific synthesis/report update from frozen metric and statistical artifacts only.

## g4_f1bzq_scientific_synthesis - 2026-09-06T12:42:20Z

- Built the final G4 F1BZQ scientific synthesis from frozen G4 analytic, metric, and statistical artifacts only.
- No Ollama/API calls, no new tests, no new statistics, no frozen artifact edits, and no final report rewrite.
- Candidate recall exact: {"beauty":"0.049494949494949494","foodcom":"0.0040000000000000001","lastfm":"0.46899999999999997","ml1m":"0.193","sports":"0.031390134529147982"}.
- Internal CI decisions: {"CI_SUPPORTED_NEGATIVE":24,"CI_SUPPORTED_POSITIVE":29,"NON_SIGNIFICANT_CI_INCLUDES_ZERO":172}.
- Validation passed: true.
- Next safe step: update the final report/relationship narrative from frozen G4 synthesis artifacts only when explicitly requested.

## final_full_latex_report_update_through_g4 - 2026-09-06T15:05:39+02:00

- Updated the full LaTeX manuscript from frozen repository artifacts only after G4 scientific synthesis freeze.
- Integrated G4 objective, dimensions, execution accounting, candidate recall, denominator semantics, CI decisions, prompt/model conclusions, and descriptive-only G1-vs-G4 policy.
- Finalized abstract, introduction contributions, discussion, threats to validity, extension chapter, overall conclusions, appendices, and `report/latex/FULL_REPORT_AUDIT.md`.
- Static checks passed: 13/13 inputs, 197 labels with 0 duplicates, 52 refs with 0 missing labels, 205 citation uses with 30 BibTeX entries and 0 missing citation keys.
- Local PDF build remains blocked because no TeX engine is available on PATH.
- No Ollama/API calls, no new experiments, no metric recomputation, no new statistical tests, and no frozen scientific artifacts were modified.


## RQ5 finalization: evidence import (2026-09-20)

Created freeze/rq5-finalization from 88c6987ffa7b0b246918686b720eb911896ebd48. Imported only selected RQ5 evidence into deterministic family archives with per-member SHA256 and provenance inventory. No inference, original-worktree changes or manuscript edits. Import command: python -B src/analysis/rq5_package_import.py --research-root <research-worktree> --submission-root <submission-worktree>. These source arguments are used only for one-time packaging; external reproduction uses the versioned package. Metric baseline: 41 passed, 2 optional rapidfuzz tests skipped; initial system-temp permission errors resolved using a worktree-local pytest basetemp. Next: revalidate raw evidence and derive history V2.

## RQ5 offline remediation and freeze (2026-09-20)

Implemented the isolated offline producer, exact matched-control/profile checks, immutable evidence verification, deterministic systematic profile reuse audit, strict-invalid-zero history V2 metrics and canonical-ID MostPop tie-breaking. Derived metric/statistical values and changes are recorded in machine-readable validation and migration manifests; no values were manually inserted into manuscript prose. Generated only five RQ5 figure families with current source/image hashes. Commands: rq5_final_freeze.py --mode recompute; --mode figures; --mode seal; selected pytest suite. Latest suite: 60 passed, 2 optional rapidfuzz skips, zero failures/errors. Earlier path-regex false positive and UTF-8 test decoding were corrected; they did not affect metrics. No inference, old artifact overwrites or changes to other RQs. Next: tracked-export clean-room validation and containing-commit attestation.

## RQ5 clean-room log byte integrity (2026-09-20)

Final attestation verification detected Windows CRLF translation in saved clean-room logs while their recorded hashes described normalized LF text. The assertion stopped before any attestation files were written. Changed the clean-room runner to save and hash the same UTF-8 bytes; scientific outputs remain unchanged. Resealed producer hashes and repeat the committed-package clean-room validation before attestation.

## RQ5 isolated clean-room validation and attestation (2026-09-20)

Validated the actual freeze-containing commit 27384ca2557112af180af24f856ade6139346045 using only its committed package export and an isolated Python runtime. All five steps passed: frozen evidence/hash validation, offline metric/bootstrap reproduction with byte comparison, figure regeneration, repeated hash validation, and 19 RQ5 tests (zero skips/failures/errors). Original worktree access and network connections were denied; no preexisting caches or external scientific bundles were used. All five saved log byte hashes were independently checked before recording this attestation. Recorded the clean-room result, hashed logs and containing-commit provenance in a subsequent metadata-only commit; the scientific manifest remains byte-identical to the validated commit. No inference, manuscript edits, protected branch changes or merge were performed. Next step, outside this task: review integration into submission/final.

## G1 metric contract and canonical provenance (2026-09-20)

Finalized the machine-readable G1 MRR and APLT alias contracts without changing any metric value or frozen implementation. MRR inspects all 10 strict-valid output positions; absent positives and invalid terminal responses contribute zero over the planned/applicable cell denominator. G1, G2, G3, G4 and RQ3 retain independent implementations whose fixed synthetic semantics are regression-tested. Canonicalized APLT@5/aplt_at_5 with upstream calACLT recorded only as an implementation alias. The corrected beyond-utility table hash is 2e5c774f552cb47f16622f2a989ce033ee05036ad814558926b40148e8857c5b; the prior hash remains historical provenance. The complete-case table contains utility metrics only and is classified as a noncanonical diagnostic. No inference, metric recomputation, manuscript edit or other-RQ change was performed.
