# CODEX_PROMPTS.md

# Safe prompt pack for VS Code + Codex

Project: **Reproducibility and Extension of “Beyond Utility: Evaluating LLM as Recommender”**

This file contains the updated, readiness-aware prompt pack to use with Codex in VS Code.

Use these prompts **one at a time**. Do **not** paste the whole file into Codex at once. After each prompt:

1. run the requested commands;
2. inspect the outputs;
3. check `git status --short`;
4. commit only when the phase is stable.

---

## Global rule before every prompt

Before implementing anything, Codex must read:

- `PROJECT_SPEC.md`
- `AGENTS.md`
- the current repository tree
- the files explicitly listed in each prompt

Codex must respect repository boundaries and must never modify `original_repo/` unless explicitly requested.

---

## Global safety rules

These rules apply to every prompt in this file.

1. Do not modify `original_repo/`.
2. Do not call APIs unless a prompt explicitly introduces `--allow-api-calls` and the flag is provided.
3. Do not download datasets.
4. Do not train heavy models unless explicitly requested later.
5. Do not invent missing data.
6. Do not invent ground truth.
7. Do not invent catalog items.
8. Do not invent candidate sets.
9. Do not invent item IDs.
10. Do not invent baseline outputs.
11. Do not invent LLM outputs.
12. Do not invent numerical results.
13. Write real metrics under `results/metrics/` only when all required inputs are available.
14. Write diagnostic/readiness summaries under `results/tables/`.
15. If a metric cannot be computed, mark it as `blocked`, `missing`, or `not_configured`; do not write zero as a placeholder.
16. Do not create empty metric CSV files under `results/metrics/`.
17. Keep every script executable from the project root.
18. Avoid absolute paths.
19. Use YAML configs for paths and settings.
20. Add tests using synthetic data unless the prompt explicitly says to use real project artifacts.
21. Preserve reproducibility: deterministic seeds, stable output names, clear logs.
22. Every generated CSV must include enough provenance to understand where it came from.
23. Every phase must update `report/notes.md` with a short, honest note about what changed and what remains blocked.

---

## Output folder policy

Use these folders consistently:

- `results/raw_outputs/`: normalized model outputs, prompt variants, raw generated artifacts that are inputs for later evaluation.
- `results/tables/`: audits, parsing summaries, readiness summaries, diagnostics, final summaries.
- `results/metrics/`: only real metric values computed from valid required inputs.
- `results/plots/`: generated plots from existing CSV results.
- `report/`: Markdown documentation, analysis, final report, presentation outline.

Important:

- Dataset audits go under `results/tables/`, not `results/metrics/`.
- Parsing summaries go under `results/tables/`, not `results/metrics/`.
- Readiness summaries go under `results/tables/`, not `results/metrics/`.
- Empty metrics CSVs must not be committed.

---

# Plan B prompt records - MovieLens-1M reconstruction track

These are concise records of the completed Plan B prompts. They are summaries,
not full prompt text.

| Prompt | Purpose | Main outputs |
|---|---|---|
| B0 | Define reconstruction feasibility plan without weakening Plan A. | `report/archive/phase_reports/plan_b_reconstruction_plan.md`; Plan B framed as ML-1M reconstruction/extension, not exact Beauty/Sports reproduction. |
| B1 | Prepare canonical MovieLens-1M data from manually supplied raw files. | `src/data/prepare_ml1m.py`, `configs/ml1m.yaml`, `data/processed/ml1m/*.csv`, `results/tables/ml1m_dataset_preparation_summary.csv`. |
| B2 | Add deterministic Random sanity-check baseline. | `src/baselines/random_baseline.py`, `results/metrics/ml1m_random_metrics.csv`, `results/tables/ml1m_random_baseline_summary.csv`. |
| B3 | Add training-only MostPop baseline and Random/MostPop comparison. | `src/baselines/ml1m_mostpop_baseline.py`, `results/metrics/ml1m_mostpop_metrics.csv`, `results/tables/ml1m_baseline_comparison_summary.csv`. |
| B4 | Add lightweight ContentBM25 baseline using ML-1M catalog title/genre text. | `src/baselines/content_bm25_baseline.py`, `results/metrics/ml1m_content_bm25_metrics.csv`, `results/tables/ml1m_content_bm25_baseline_summary.csv`. |
| B5 | Add full ML-1M reconstruction runner and plots. | `src/experiments/run_ml1m_reconstruction.py`, `results/tables/ml1m_reconstruction_pipeline_summary.csv`, `results/plots/ml1m_baseline_*.png`. |
| B6 | Consolidate final documentation for Plan A and Plan B. | README, project spec, technical report, results analysis, reproducibility checklist, Plan B plan, notes, and this prompt log. |

Plan B final metric rows are real MovieLens-1M reconstruction metrics:
Random HR@5 `0.0514900662251655`, Random NDCG@5 `0.0293715251496116`,
MostPop HR@5 `0.320364238410596`, MostPop NDCG@5 `0.2102806322093436`,
ContentBM25 HR@5 `0.2508278145695364`, and ContentBM25 NDCG@5
`0.1647567964031986`.

These values do not recover or reproduce the original Beauty/Sports numerical
results.

---

# Plan C prompt records - full-paper replication campaign

These are concise records of the Plan C prompts. They are summaries, not full
prompt text.

| Prompt | Purpose | Main outputs |
|---|---|---|
| C0 | Build a full replication matrix across paper datasets, tasks, prompt strategies, LLMs, traditional models, metrics, and artifacts. | `report/plan_c_full_replication_matrix.md`, `results/tables/plan_c_replication_matrix.csv`, `results/tables/plan_c_required_artifacts.csv`, `results/tables/plan_c_models_metrics_tasks_inventory.csv`. |
| C1 | Define artifact recovery and validation before converters, model runs, or metrics. | `report/plan_c_artifact_recovery_protocol.md`, `data/artifact_manifests/plan_c_artifact_manifest*.csv`, `src/audit/validate_plan_c_artifacts.py`, `results/tables/plan_c_artifact_validation_*.csv`. |
| C2 | Check external source URLs safely and create a recovery ledger, manual download checklist, and author-contact draft. | `src/audit/check_plan_c_source_urls.py`, `results/tables/plan_c_source_url_check_results.csv`, `results/tables/plan_c_external_source_recovery_ledger.csv`, `report/plan_c_external_source_recovery_report.md`, `report/plan_c_manual_download_checklist.md`, `report/plan_c_author_contact_request.md`. |
| C3 | Add a manual artifact staging workflow and inspect staged archives without extraction or manifest mutation. | `data/external/plan_c/.gitkeep`, `src/audit/inspect_plan_c_downloads.py`, `results/tables/plan_c_manual_download_*.csv`, `results/tables/plan_c_suggested_manifest_updates.csv`, `report/plan_c_manual_artifact_intake_guide.md`. |
| C4 | Validate staged raw public dataset schemas and classify reconstruction readiness without conversion or metrics. | `src/audit/validate_plan_c_raw_datasets.py`, `results/tables/plan_c_raw_dataset_*.csv`, `results/tables/plan_c_reconstruction_readiness_summary.csv`, `tests/test_validate_plan_c_raw_datasets.py`. |
| C5 | Convert validated public raw inputs into reconstructed canonical Plan C datasets without claiming exact paper artifact recovery. | `configs/plan_c_reconstruction.yaml`, `src/data/reconstruct_plan_c_datasets.py`, `data/reconstructed/plan_c/.gitkeep`, `results/tables/plan_c_reconstructed_dataset_*.csv`, `tests/test_reconstruct_plan_c_datasets.py`. |

C1 classifies artifacts as official, author-released, reconstructed,
extension-only, unavailable, or unknown, and validates only local paths,
optional sizes, and optional SHA-256 hashes. It does not download data, call
APIs, train models, compute metrics, or modify `original_repo/`.

C2 performs metadata-only source checks for manifest URLs. It records reachable,
unavailable, or network-unverified sources without downloading datasets, saving
remote content, calling APIs, training models, computing metrics, or modifying
`original_repo/`.

C3 scans manually staged files under `data/external/plan_c/`, inspects ZIP/TAR
member names without extraction, and writes suggested manifest updates that
always require manual confirmation. It does not trust staged files
automatically and does not create metrics.

C4 samples staged raw Amazon, MovieLens-1M, and LastFM files to verify schema
readiness for future reconstruction. It does not recover Beauty/Sports
processed paper artifacts, permanently extract archives, create processed
datasets, train models, or compute metrics.

C5 writes reconstructed canonical files under `data/reconstructed/plan_c/` from
staged public raw inputs. It supports `--datasets`, a development-only
`--max-users` diagnostic subset marker, and `--skip-paper-like-artifacts` for
debugging. Beauty/Sports official Dropbox artifacts remain blocked after
manual testing. Generated C5 files are labeled reconstructed, not exact, and no
metrics are written.

---

# Prompt 00 — Initial setup and project scaffold

```text
You are working on a reproducibility project for the paper "Beyond Utility: Evaluating LLM as Recommender".

Read PROJECT_SPEC.md and AGENTS.md first.

Task: create the initial project scaffold only. Do not implement metrics yet.

Create missing folders according to the expected structure:
- original_repo/
- paper/
- src/data/
- src/baselines/
- src/prompts/
- src/evaluation/
- src/experiments/
- src/utils/
- configs/
- data/raw/
- data/processed/
- data/samples/
- results/raw_outputs/
- results/metrics/
- results/tables/
- results/plots/
- tests/
- report/

Create or update:
- README.md
- .gitignore
- requirements.txt
- environment.yml
- report/notes.md

Also add minimal package markers where needed so imports from src work in tests.

Do not touch original_repo/.
Do not create fake results.
Do not add heavy dependencies.
Do not implement metrics yet.

After changes, report:
- files created or modified;
- commands to run;
- whether a commit is safe.
```

Success check:

```powershell
python --version
pip install -r requirements.txt
pytest tests\
```

Suggested commit:

```powershell
git add .
git commit -m "Initialize project scaffold"
git status --short
```

---

# Prompt 01 — Inspect original repository and create codebase map

```text
Read PROJECT_SPEC.md and AGENTS.md.

Inspect the repository under original_repo/EvaLLMasRecommender without modifying it.

Create report/codebase_map.md explaining:
- repository structure;
- dataset locations;
- LLM outputs;
- baseline outputs;
- metrics;
- ranking/re-ranking scripts;
- useful files;
- risky or expensive parts;
- scripts that may call APIs;
- scripts that may train models;
- recommended minimal reproduction path.

Do not modify original_repo/.
Do not run expensive training.
Do not call APIs.
If uncertain, explicitly mark uncertainty instead of guessing.

Update report/notes.md.

After changes, report:
- files created or modified;
- what original artifacts were found;
- what remains uncertain;
- commands to run;
- whether a commit is safe.
```

Success check:

```powershell
dir report\codebase_map.md
```

Suggested commit:

```powershell
git add report/codebase_map.md report/notes.md
git commit -m "Document original repository structure"
git status --short
```

---

# Prompt 02 — Smoke test script

```text
Read PROJECT_SPEC.md, AGENTS.md, and report/codebase_map.md.

Implement src/experiments/exp00_smoke_test.py.

The script must run from the repository root and check:
- expected folders;
- configs folder;
- original_repo/EvaLLMasRecommender;
- paper folder and paper PDF if available;
- Python imports from src;
- results subfolders;
- report folder;
- essential files such as README.md, PROJECT_SPEC.md, AGENTS.md.

The script should print/log a clear PASS/FAIL summary.
It should exit with non-zero status if critical checks fail.
Add minimal helper utilities only if needed under src/utils/.
Do not implement metrics yet.
Do not modify original_repo/.

Add or update tests only if useful.
Update report/notes.md.

After changes, report:
- files created or modified;
- commands to run;
- expected output;
- whether a commit is safe.
```

Success check:

```powershell
python src/experiments/exp00_smoke_test.py
pytest tests\
```

Suggested commit:

```powershell
git add src/experiments/exp00_smoke_test.py src/utils report/notes.md tests
git commit -m "Add project smoke test"
git status --short
```

---

# Prompt 03 — YAML configuration files and config loader

```text
Read PROJECT_SPEC.md, AGENTS.md, and report/codebase_map.md.

Create or update:
- configs/beauty.yaml
- configs/sports.yaml
- src/utils/config.py
- tests/test_config.py

Optionally create configs/ml1m.yaml and configs/cost.yaml only if useful as placeholders.

Each YAML must contain:
- dataset name;
- raw_dir;
- processed_dir;
- sample_dir;
- paths for interactions_path, catalog_path, candidates_path, ground_truth_path, llm_outputs_path, baseline_outputs_path;
- original_repo root;
- task ranking;
- k_values [5];
- seed 42;
- sample_size null;
- allow_api_calls false;
- models section;
- metrics section;
- output section.

Do not invent paths as facts.
If a path is uncertain, set it to null and add a comment explaining it must be filled after dataset audit.

src/utils/config.py must:
- load YAML safely;
- resolve paths relative to project root when appropriate;
- validate required sections;
- expose clear helper functions usable by later scripts.

Do not modify original_repo/.
Do not create fake results.
Update report/notes.md.

After changes, report:
- files created or modified;
- commands to run;
- expected output;
- whether a commit is safe.
```

Success check:

```powershell
python -c "import yaml, pathlib; [print(p, yaml.safe_load(open(p, encoding='utf-8'))['dataset']['name']) for p in pathlib.Path('configs').glob('*.yaml')]"
pytest tests/test_config.py
pytest tests\
```

Suggested commit:

```powershell
git add configs src/utils/config.py tests/test_config.py report/notes.md
git commit -m "Add configuration loading utilities"
git status --short
```

---

# Prompt 04 — Dataset audit utility

```text
Read PROJECT_SPEC.md, AGENTS.md, report/codebase_map.md, configs/beauty.yaml, configs/sports.yaml, and src/utils/config.py.

Implement:
- src/data/audit_datasets.py
- src/data/load_data.py with minimal safe loading helpers if needed
- tests/test_dataset_audit.py

The script must accept --config.
It must load YAML safely.
It must inspect configured dataset paths if present.
If paths are null, it may search inside original_repo/ for likely dataset files, but only for inspection.
It must not modify original_repo/.

The audit must report:
- configured paths;
- whether each path exists;
- likely files found inside original_repo/;
- basic stats when safely computable;
- unknown/null when stats cannot be computed;
- missing canonical interactions/catalog/candidates/ground_truth.

Save audit CSV to:
- results/tables/<dataset>_dataset_audit.csv

Write or update:
- report/dataset_audit.md
- report/notes.md

Do not invent statistics.
If a statistic cannot be computed, write null/unknown.
Do not write audits to results/metrics/.
Do not create fake processed datasets.

After changes, report:
- files created or modified;
- outputs generated;
- commands to run;
- expected output;
- whether a commit is safe.
```

Success check:

```powershell
python src/data/audit_datasets.py --config configs/beauty.yaml
python src/data/audit_datasets.py --config configs/sports.yaml
pytest tests/test_dataset_audit.py
pytest tests\
```

Suggested commit:

```powershell
git add src/data/audit_datasets.py src/data/load_data.py src/utils/config.py tests/test_dataset_audit.py
git add results/tables/*_dataset_audit.csv report/dataset_audit.md report/notes.md
git commit -m "Add dataset audit utilities"
git status --short
```

---

# Prompt 05 — Unified schemas

```text
Read PROJECT_SPEC.md, AGENTS.md, src/utils/config.py, and src/data/load_data.py.

Create src/data/schemas.py with simple schema validation helpers for:
- interactions;
- catalog/items;
- candidate sets;
- recommendations;
- parsed LLM outputs;
- ground truth;
- evaluation rows.

Each validator should:
- check required columns;
- normalize column names where reasonable;
- return a normalized dataframe or records;
- raise clear ValueError messages when schema is invalid;
- avoid dataset-specific paths;
- be easy to unit test.

Create tests/test_schemas.py with synthetic examples only.

Do not modify original_repo/.
Do not load real datasets in tests.
Do not invent project results.
Update report/notes.md.

After changes, report:
- files created or modified;
- commands to run;
- expected output;
- whether a commit is safe.
```

Success check:

```powershell
python -m compileall src/data/schemas.py
pytest tests/test_schemas.py
pytest tests\
```

Suggested commit:

```powershell
git add src/data/schemas.py tests/test_schemas.py report/notes.md
git commit -m "Add unified data schemas"
git status --short
```

---

# Prompt 06 — Utility metrics HR@K and NDCG@K

```text
Read PROJECT_SPEC.md, AGENTS.md, and src/data/schemas.py.

Implement src/evaluation/utility_metrics.py with:
- hit_rate_at_k;
- ndcg_at_k;
- evaluate_utility_at_k.

Requirements:
1. Support single relevant item per user.
2. Support multiple relevant items per user.
3. Handle empty recommendations.
4. Handle duplicated recommendations.
5. Handle users with no ground truth using a documented policy.
6. Do not treat missing ground truth as zero performance silently.
7. Support top-K cutoff.
8. Be deterministic and easy to test.
9. Do not depend on original repository code until our tests pass.
10. Do not read or write project result CSVs in this metric module.

Create tests/test_utility_metrics.py with synthetic examples and exact expected values.

Do not modify original_repo/.
Do not invent results.
Update report/notes.md.

After changes, report:
- files created or modified;
- commands to run;
- expected output;
- whether a commit is safe.
```

Success check:

```powershell
python -m compileall src/evaluation/utility_metrics.py
pytest tests/test_utility_metrics.py
pytest tests\
```

Suggested commit:

```powershell
git add src/evaluation/utility_metrics.py tests/test_utility_metrics.py report/notes.md
git commit -m "Add utility ranking metrics"
git status --short
```

---

# Prompt 07 — LLM output parser

```text
Read PROJECT_SPEC.md, AGENTS.md, and src/data/schemas.py.

Implement src/evaluation/output_parser.py to parse textual LLM recommendation outputs into ranked recommendation records.

Support:
- numbered lists;
- bullet lists;
- comma-separated items;
- quoted item titles;
- JSON-like lists;
- multiline outputs;
- outputs with extra explanatory text;
- outputs containing item IDs if present.

Implement normalization:
- lowercase;
- strip punctuation;
- collapse whitespace;
- remove ranking numbers.

Add optional fuzzy matching using rapidfuzz if available, but:
- keep exact, normalized, and fuzzy match types separate;
- keep fuzzy disabled by default;
- do not silently convert unmatched text into valid items.

Create tests/test_output_parser.py with synthetic noisy LLM outputs.

Do not modify original_repo/.
Do not invent item IDs.
Do not invent results.
Update report/notes.md.

After changes, report:
- files created or modified;
- commands to run;
- expected output;
- whether a commit is safe.
```

Success check:

```powershell
python -m compileall src/evaluation/output_parser.py
pytest tests/test_output_parser.py
pytest tests\
```

Suggested commit:

```powershell
git add src/evaluation/output_parser.py tests/test_output_parser.py report/notes.md
git commit -m "Add LLM output parser"
git status --short
```

---

# Prompt 08 — Hallucination metric

```text
Read PROJECT_SPEC.md, AGENTS.md, src/evaluation/output_parser.py, and src/data/schemas.py.

Implement src/evaluation/hallucination.py with:
- hallucination_rate;
- candidate_hallucination_rate;
- catalog_hallucination_rate;
- helper functions for user-level and aggregate hallucination summaries if useful.

Requirements:
1. Distinguish exact matches, normalized matches, fuzzy matches, and unmatched outputs.
2. Never improve utility by pretending fuzzy matches are exact.
3. Support candidate-set hallucination and catalog hallucination as separate concepts.
4. Handle empty recommendation lists.
5. Handle empty candidate/catalog references with clear documented behavior.
6. Do not read real datasets inside the metric module.
7. Do not write CSV files inside the metric module.

Create tests/test_hallucination.py with synthetic examples.

Do not modify original_repo/.
Do not invent catalog items.
Do not invent candidate items.
Do not invent results.
Update report/notes.md.

After changes, report:
- files created or modified;
- commands to run;
- expected output;
- whether a commit is safe.
```

Success check:

```powershell
python -m compileall src/evaluation/hallucination.py
pytest tests/test_hallucination.py
pytest tests\
```

Suggested commit:

```powershell
git add src/evaluation/hallucination.py tests/test_hallucination.py report/notes.md
git commit -m "Add hallucination metrics"
git status --short
```

---

# Prompt 09 — Load existing LLM outputs

```text
Read PROJECT_SPEC.md, AGENTS.md, report/codebase_map.md, report/dataset_audit.md, configs/beauty.yaml, configs/sports.yaml, src/utils/config.py, src/evaluation/output_parser.py, and src/data/schemas.py.

Implement src/experiments/load_llm_outputs.py.

The script must accept:
- --config
- --log-unmatched-limit, optional

Goal:
Locate existing LLM output files, parse them with src/evaluation/output_parser.py, and save normalized outputs for later evaluation.

Requirements:
1. Load YAML config safely.
2. Locate configured LLM output files if available.
3. If config paths are null, inspect original_repo/ according to report/codebase_map.md.
4. Never modify original_repo/.
5. Do not call APIs.
6. Do not create fake LLM outputs.
7. Parse outputs into normalized rows.
8. Preserve recommendation order.
9. Keep unmatched text explicit; do not invent item IDs.
10. If catalog_path is not configured, recommended_item_id may remain empty while recommended_item_text is preserved.
11. Save normalized outputs to results/raw_outputs/<dataset>_llm_outputs_normalized.csv.
12. Save parsing summary to results/tables/<dataset>_llm_parsing_summary.csv.
13. Do not save parsing summary to results/metrics/.
14. If no LLM outputs are found, produce a clear diagnostic summary instead of crashing without explanation.

Create tests/test_load_llm_outputs.py using synthetic temporary files only.

Update report/notes.md.

After changes, report:
- files created or modified;
- outputs generated;
- commands to run;
- expected output;
- whether a commit is safe.
```

Success check:

```powershell
python -m compileall src/experiments/load_llm_outputs.py
pytest tests/test_load_llm_outputs.py
python src/experiments/load_llm_outputs.py --config configs/beauty.yaml --log-unmatched-limit 0
python src/experiments/load_llm_outputs.py --config configs/sports.yaml --log-unmatched-limit 0
pytest tests\
```

Suggested commit:

```powershell
git add src/experiments/load_llm_outputs.py tests/test_load_llm_outputs.py report/notes.md
git add results/raw_outputs/*_llm_outputs_normalized.csv
git add results/tables/*_llm_parsing_summary.csv
git commit -m "Add LLM output loading pipeline"
git status --short
```

---

# Prompt 10 — MostPop baseline utility

```text
Read PROJECT_SPEC.md, AGENTS.md, report/dataset_audit.md, configs/beauty.yaml, configs/sports.yaml, src/utils/config.py, src/data/schemas.py, and src/data/load_data.py.

Implement src/baselines/mostpop.py.

Goal:
Implement a lightweight MostPop baseline utility over provided training interactions. Do not run it on real datasets unless interactions_path is configured and readable.

Requirements:
1. Load config safely.
2. If interactions_path is null or missing, exit with a clear controlled message.
3. Do not invent interactions.
4. Do not use test interactions to compute popularity.
5. Compute global item popularity from provided train interactions only.
6. Recommend top-K/top-N items for each test user when user set is available.
7. Exclude items already seen when possible.
8. Preserve deterministic tie-breaking.
9. Save recommendations to results/raw_outputs/<dataset>_mostpop_recommendations.csv only when real recommendations are generated from valid interactions.
10. Save summary to results/tables/<dataset>_mostpop_summary.csv.
11. Do not save MostPop summaries to results/metrics/.
12. Do not train heavy models.
13. Do not modify original_repo/.

Create tests/test_mostpop.py using synthetic interaction records only.

Update report/notes.md.

After changes, report:
- files created or modified;
- commands to run;
- expected behavior with current configs;
- whether a commit is safe.
```

Success check:

```powershell
python -m compileall src/baselines/mostpop.py
pytest tests/test_mostpop.py
pytest tests\
python src/baselines/mostpop.py --config configs/beauty.yaml
```

Expected current behavior:

- Tests pass.
- With current configs, MostPop should not generate real outputs if interactions_path is null.
- It should exit with a clear controlled message.

Suggested commit:

```powershell
git add src/baselines/mostpop.py tests/test_mostpop.py report/notes.md
git add results/raw_outputs/*_mostpop_recommendations.csv results/tables/*_mostpop_summary.csv
git commit -m "Add MostPop utility"
git status --short
```

If no MostPop outputs are generated, do not add non-existing result files.

---

# Prompt 11 — Load baseline outputs from original repo

```text
Read PROJECT_SPEC.md, AGENTS.md, report/codebase_map.md, report/dataset_audit.md, configs/beauty.yaml, configs/sports.yaml, src/utils/config.py, src/data/schemas.py, src/baselines/mostpop.py, and src/experiments/load_llm_outputs.py.

Implement src/baselines/load_baseline_outputs.py.

Goal:
Search configured paths and original_repo/ for existing traditional baseline or SOTA recommendation outputs, normalize them if found, or write a clear diagnostic summary if none are available.

Requirements:
1. Load YAML config safely.
2. Inspect configured baseline paths.
3. Inspect relevant original_repo/ locations identified in report/codebase_map.md and report/dataset_audit.md.
4. Never modify original_repo/.
5. Do not train BPR, SASRec, LightGCN, or other models.
6. Do not call APIs.
7. Do not download datasets.
8. Do not invent missing baseline outputs.
9. If baseline output files are found, normalize them under results/raw_outputs/<dataset>_baseline_outputs_normalized.csv.
10. If no baseline output files are found, write a diagnostic summary under results/tables/<dataset>_baseline_outputs_summary.csv.
11. The summary must clearly state found, missing, not_configured, or invalid status.
12. Do not calculate HR@K, NDCG@K, hallucination, coverage, position bias, history length, or cost metrics in this phase.
13. Do not create final experimental results.

Create tests/test_load_baseline_outputs.py using synthetic temporary files only.

Test cases must include:
- loading a synthetic baseline CSV;
- loading a synthetic baseline JSON if supported;
- preserving recommendation order;
- writing a diagnostic summary when no files are configured/found;
- not depending on real original_repo data;
- not requiring interactions_path.

Update report/notes.md.

After changes, report:
- files created or modified;
- outputs generated;
- commands to run;
- expected output;
- whether a commit is safe.
```

Success check:

```powershell
python -m compileall src/baselines/load_baseline_outputs.py
pytest tests/test_load_baseline_outputs.py
pytest tests\
python src/baselines/load_baseline_outputs.py --config configs/beauty.yaml
python src/baselines/load_baseline_outputs.py --config configs/sports.yaml
```

Suggested commit:

```powershell
git add src/baselines/load_baseline_outputs.py tests/test_load_baseline_outputs.py report/notes.md
git add results/raw_outputs/*_baseline_outputs_normalized.csv
git add results/tables/*_baseline_outputs_summary.csv
git commit -m "Add baseline output loader"
git status --short
```

---

# Prompt 12 — Reproduction pipeline with readiness checks

```text
Read PROJECT_SPEC.md, AGENTS.md, CODEX_PROMPTS.md, report/codebase_map.md, report/dataset_audit.md, configs/beauty.yaml, configs/sports.yaml, src/utils/config.py, src/data/schemas.py, src/evaluation/utility_metrics.py, src/evaluation/output_parser.py, src/evaluation/hallucination.py, src/experiments/load_llm_outputs.py, src/baselines/load_baseline_outputs.py, and src/baselines/mostpop.py.

Implement src/experiments/run_reproduction.py.

Goal:
Create a cautious and reproducible reproduction pipeline. The pipeline must combine available normalized LLM outputs and available normalized baseline outputs, but it must compute utility or hallucination metrics only when the required inputs are actually configured and available.

The script must be executable from the project root as:
- python src/experiments/run_reproduction.py --config configs/beauty.yaml
- python src/experiments/run_reproduction.py --config configs/sports.yaml

Requirements:
1. Load YAML config safely.
2. Check whether normalized LLM outputs exist under results/raw_outputs/.
3. A normalized recommendation file is usable if it has at least one row with non-empty recommended_item_id or non-empty recommended_item_text.
4. Do not require recommended_item_id when catalog matching is not configured.
5. If some rows have both recommended_item_id and recommended_item_text empty, skip/count them and mark ready_with_warnings if useful.
6. If all recommendation rows are empty, mark blocked/invalid.
7. Check whether normalized baseline outputs exist under results/raw_outputs/.
8. Check whether MostPop outputs exist under results/raw_outputs/.
9. Check whether ground_truth_path is configured and exists.
10. Check whether catalog_path or candidates_path is configured and exists.
11. If ground truth is missing or not configured, do not compute HR@5 or NDCG@5.
12. If catalog/candidates are missing or not configured, do not compute real hallucination rate.
13. If baseline outputs are missing for a dataset, do not invent, train, or download baseline outputs.
14. Clearly distinguish ready, ready_with_warnings, missing, not_configured, invalid, and blocked.
15. If all required inputs are available, compute HR@K and NDCG@K using src/evaluation/utility_metrics.py.
16. If catalog/candidates are available, compute hallucination metrics using src/evaluation/hallucination.py.
17. If inputs are incomplete, produce diagnostic outputs instead of fake metric values.
18. Do not write HR@5 = 0 or NDCG@5 = 0 because ground truth is missing.
19. Do not write an empty metrics CSV.
20. Write results/metrics/<dataset>_reproduction_metrics.csv only if at least one real metric has been computed from valid inputs.
21. Save readiness/diagnostic summary to results/tables/<dataset>_reproduction_summary.csv.
22. Update or create report/results_analysis.md.
23. Missing or null ground_truth_path must not be a fatal script error; the script should exit 0 when diagnostics are produced successfully.
24. Missing or null catalog_path/candidates_path must not be a fatal script error.
25. Return non-zero only for invalid config structure, unreadable required project files, or actual script errors.
26. Do not call APIs.
27. Do not download data.
28. Do not modify original_repo/.
29. Do not invent numerical results.

Expected current behavior:
- Beauty should find normalized LLM outputs.
- Beauty should find normalized BPRMF baseline outputs if already loaded.
- Sports should find normalized LLM outputs and may mark them ready_with_warnings if a small number of rows are empty.
- Sports may not find normalized baseline outputs.
- If ground truth and catalog/candidate paths are not configured, utility and hallucination evaluation must be blocked, not computed.
- No empty metrics CSV should be written.

Create tests/test_run_reproduction.py using synthetic temporary files/configs only.

Test cases must include:
- ready status when all required synthetic files exist;
- blocked utility evaluation when ground truth is missing;
- blocked hallucination evaluation when catalog/candidate files are missing;
- missing baseline outputs handled gracefully;
- no metric values produced when required inputs are unavailable;
- valid synthetic HR@K/NDCG@K computation only when synthetic ground truth is available;
- valid recommendation text with empty recommended_item_id is accepted as usable;
- partially empty recommendation rows do not block a whole file if valid rows exist.

Update report/notes.md.

After changes, report:
- files created or modified;
- generated outputs;
- commands to run;
- expected output;
- whether a commit is safe.
```

Success check:

```powershell
python -m compileall src/experiments/run_reproduction.py
pytest tests/test_run_reproduction.py
pytest tests\
python src/experiments/run_reproduction.py --config configs/beauty.yaml
python src/experiments/run_reproduction.py --config configs/sports.yaml
Get-Content results/tables/beauty_reproduction_summary.csv
Get-Content results/tables/sports_reproduction_summary.csv
```

Suggested commit:

```powershell
git add src/experiments/run_reproduction.py tests/test_run_reproduction.py
git add report/results_analysis.md report/notes.md
git add results/tables/*_reproduction_summary.csv
git add results/metrics/*_reproduction_metrics.csv
git add -u results/metrics
git commit -m "Add reproduction pipeline with readiness checks"
git status --short
```

If no real metrics were computed, do not add non-existing metric files and make sure empty metric CSVs are removed.

---

# Prompt 13 — Catalog Coverage@K extension

```text
Read PROJECT_SPEC.md, AGENTS.md, CODEX_PROMPTS.md, src/data/schemas.py, src/evaluation/utility_metrics.py, src/evaluation/hallucination.py, src/experiments/run_reproduction.py, configs/beauty.yaml, and configs/sports.yaml.

Implement only Prompt 13: Catalog Coverage@K extension.

Create:
- src/evaluation/coverage.py
- tests/test_coverage.py

Optionally create:
- src/experiments/run_extension_coverage.py

Goal:
Implement Catalog Coverage@K as the main project extension beyond the original reproduction metrics.

Definition:
Catalog Coverage@K measures the fraction of unique catalog items that appear at least once in the top-K recommendation lists across evaluated users.

Formula:
coverage@K = number_of_unique_recommended_items_in_top_k / number_of_items_in_catalog

Part A — Metric implementation:
1. Implement catalog_coverage_at_k.
2. Implement coverage_by_model.
3. Support recommendation lists as item IDs or strings.
4. Support multiple users.
5. Respect the top-K cutoff.
6. Deduplicate repeated recommendations across users.
7. Do not count duplicate recommendations multiple times.
8. Handle empty recommendation lists.
9. Handle empty catalog with documented behavior.
10. Keep implementation deterministic and easy to test.
11. Do not read real datasets inside src/evaluation/coverage.py.
12. Do not write CSV files inside src/evaluation/coverage.py.

Part B — Optional coverage runner:
If creating src/experiments/run_extension_coverage.py, make it readiness-aware.

The runner must:
1. Load YAML config using src/utils/config.py.
2. Check whether normalized recommendation outputs exist under results/raw_outputs/.
3. Check whether catalog_path is configured and points to a readable file.
4. If catalog_path is missing or null, do not compute Coverage@5.
5. If catalog_path is missing or null, write only a diagnostic summary under results/tables/.
6. Write results/metrics/<dataset>_coverage_metrics.csv only if real Coverage@5 values are computed from a valid catalog.
7. Do not use candidate sets as a catalog unless this is explicitly labeled as candidate_set_coverage@K, not catalog_coverage@K.
8. Do not invent catalog items.
9. Do not infer item IDs from text unless a valid catalog mapping is configured.
10. Do not call APIs.
11. Do not download data.
12. Do not modify original_repo/.
13. Do not compute HR@5, NDCG@5, hallucination, position bias, history length, or cost metrics in this phase.
14. Update report/results_analysis.md explaining why coverage can change the interpretation of utility metrics, but clearly state that real Coverage@5 is blocked when catalog_path is not configured.

Tests:
Create tests/test_coverage.py with synthetic catalogs and synthetic recommendation lists only.

Test cases must include:
- coverage@K equals 1.0 when all catalog items are recommended at least once;
- coverage@K equals 0.0 when no recommendations are present;
- partial catalog coverage;
- top-K cutoff changes the result;
- duplicate recommendations do not inflate coverage;
- multiple users are aggregated correctly;
- empty catalog behavior is documented and tested;
- per-model aggregation works.

If run_extension_coverage.py is implemented, add tests for:
- coverage metrics are written only when a valid synthetic catalog exists;
- diagnostic summary is written when catalog_path is missing;
- no metrics CSV is written when coverage is blocked.

After completing the phase, report:
- changed files;
- generated outputs;
- commands to run;
- expected outputs;
- whether a commit is safe.
```

Success check:

```powershell
python -m compileall src/evaluation/coverage.py
pytest tests/test_coverage.py
pytest tests\

# Only if run_extension_coverage.py exists:
python -m compileall src/experiments/run_extension_coverage.py
python src/experiments/run_extension_coverage.py --config configs/beauty.yaml
python src/experiments/run_extension_coverage.py --config configs/sports.yaml
```

Expected current behavior:

- Tests pass.
- With current configs, Coverage@5 is blocked because catalog_path is not configured.
- No real coverage metrics CSV is written under results/metrics/.
- Diagnostic summaries may be written under results/tables/.

Suggested commit:

```powershell
git add src/evaluation/coverage.py tests/test_coverage.py
git add src/experiments/run_extension_coverage.py
git add report/results_analysis.md report/notes.md
git add results/tables/ results/metrics/
git commit -m "Add catalog coverage extension"
git status --short
```

If src/experiments/run_extension_coverage.py was not created, do not add it.
If no real metrics were computed, make sure no empty coverage metrics CSV exists.

---

# Prompt 14 — Candidate position bias utilities

```text
Read PROJECT_SPEC.md, AGENTS.md, CODEX_PROMPTS.md, configs/beauty.yaml, configs/sports.yaml, src/data/schemas.py, src/evaluation/utility_metrics.py, src/experiments/run_reproduction.py, and report/results_analysis.md.

Implement only Prompt 14: candidate position bias utilities.

Create:
- src/prompts/candidate_ordering.py
- src/evaluation/position_bias.py
- tests/test_position_bias.py

Optionally create:
- src/experiments/run_position_bias.py

Goal:
Prepare a safe and testable analysis of candidate position bias. Candidate position bias measures whether recommended items are disproportionately selected from particular input candidate positions.

Candidate ordering variants:
- original
- random
- reversed
- positive_at_start
- positive_at_end

Part A — Utilities:
1. Implement deterministic candidate ordering transformations.
2. Implement selected_candidate_position_distribution.
3. Implement position_bias_summary helpers.
4. Support synthetic candidate lists and synthetic recommendation outputs.
5. Preserve original candidate item IDs/text.
6. Do not call APIs.
7. Do not generate LLM outputs.
8. Do not modify original_repo/.
9. Do not compute HR@5 or NDCG@5 unless valid ground truth exists.

Part B — Optional runner:
If creating src/experiments/run_position_bias.py, make it readiness-aware.

The runner must:
1. Load YAML config using src/utils/config.py.
2. Check whether candidates_path is configured and readable.
3. Check whether normalized recommendation outputs exist.
4. If candidates_path is missing or null, do not compute position bias.
5. If recommendation outputs are missing, do not compute position bias.
6. If required inputs are missing, write diagnostic summary under results/tables/.
7. Write results/metrics/<dataset>_position_bias_metrics.csv only if position bias is computed from valid inputs.
8. Do not call APIs unless --allow-api-calls is explicitly provided.
9. If --allow-api-calls is not provided, the script may only prepare candidate ordering variants, not send prompts.
10. Do not invent candidate lists, relevant items, or model outputs.
11. Do not modify original_repo/.

Tests:
Create tests/test_position_bias.py using synthetic candidates and synthetic recommendations only.

Test cases must include:
- reversed ordering works;
- random ordering is deterministic with seed;
- positive_at_start places the known positive item first in synthetic data;
- positive_at_end places the known positive item last in synthetic data;
- position distribution is computed correctly;
- duplicate recommendations do not inflate position counts;
- missing candidates are handled explicitly;
- no real metrics are produced when required inputs are unavailable.

After completing the phase, report:
- changed files;
- generated outputs;
- commands to run;
- expected outputs;
- whether a commit is safe.
```

Success check:

```powershell
python -m compileall src/prompts/candidate_ordering.py
python -m compileall src/evaluation/position_bias.py
pytest tests/test_position_bias.py
pytest tests\

# Only if run_position_bias.py exists:
python -m compileall src/experiments/run_position_bias.py
python src/experiments/run_position_bias.py --config configs/beauty.yaml
python src/experiments/run_position_bias.py --config configs/sports.yaml
```

Expected current behavior:

- Tests pass.
- With current configs, real position bias is blocked if candidates_path is not configured.
- Diagnostic summaries may be written under results/tables/.
- No fake position bias metrics are written.

Suggested commit:

```powershell
git add src/prompts/candidate_ordering.py src/evaluation/position_bias.py tests/test_position_bias.py
git add src/experiments/run_position_bias.py
git add results/tables/ results/metrics/
git add report/results_analysis.md report/notes.md
git commit -m "Add candidate position bias utilities"
git status --short
```

If src/experiments/run_position_bias.py was not created, do not add it.

---

# Prompt 15 — History length sensitivity utilities

```text
Read PROJECT_SPEC.md, AGENTS.md, CODEX_PROMPTS.md, configs/beauty.yaml, configs/sports.yaml, src/data/schemas.py, src/evaluation/utility_metrics.py, and src/experiments/run_reproduction.py.

Implement only Prompt 15: history length sensitivity utilities.

Create:
- src/prompts/history_length.py
- tests/test_history_length.py

Optionally create:
- src/experiments/run_history_length.py

Goal:
Prepare a safe analysis of how the amount of user history in the prompt affects recommendation behavior.

History lengths:
- 0
- 1
- 2
- 5
- 10
- full

Part A — Utilities:
1. Implement functions to truncate user histories to a requested length.
2. Implement functions to create history-length variants.
3. Preserve order of interactions where timestamps or ranks are available.
4. Support synthetic histories in tests.
5. Do not call APIs.
6. Do not generate LLM outputs.
7. Do not modify original_repo/.
8. Do not compute HR@5/NDCG@5 unless valid ground truth and outputs exist.

Part B — Optional runner:
If creating src/experiments/run_history_length.py, make it readiness-aware.

The runner must:
1. Load YAML config using src/utils/config.py.
2. Check whether interactions_path or a canonical user history file is configured and readable.
3. If history data is unavailable, write a diagnostic summary under results/tables/.
4. If history data is available, write prompt/input variants to results/raw_outputs/<dataset>_history_length_prompts.csv.
5. Do not call APIs unless --allow-api-calls is explicitly provided.
6. If --allow-api-calls is not provided, prepare prompts only.
7. Write results/metrics/<dataset>_history_length_metrics.csv only if valid model outputs and ground truth are available.
8. Do not invent histories, ground truth, or model outputs.
9. Do not download data.
10. Do not modify original_repo/.

Tests:
Create tests/test_history_length.py using synthetic user histories.

Test cases must include:
- length 0 returns empty history;
- length 1 returns most recent or latest-ranked item;
- length 2/5/10 truncates correctly;
- full preserves all history;
- users with shorter histories are handled;
- ordering is deterministic;
- prompt variant creation is reproducible.

After completing the phase, report:
- changed files;
- generated outputs;
- commands to run;
- expected outputs;
- whether a commit is safe.
```

Success check:

```powershell
python -m compileall src/prompts/history_length.py
pytest tests/test_history_length.py
pytest tests\

# Only if run_history_length.py exists:
python -m compileall src/experiments/run_history_length.py
python src/experiments/run_history_length.py --config configs/beauty.yaml
python src/experiments/run_history_length.py --config configs/sports.yaml
```

Expected current behavior:

- Tests pass.
- With current configs, real history-length experiments are likely blocked because interactions_path/history data is not configured.
- Diagnostic summaries may be written under results/tables/.
- No fake metrics are written.

Suggested commit:

```powershell
git add src/prompts/history_length.py tests/test_history_length.py
git add src/experiments/run_history_length.py
git add results/raw_outputs/ results/tables/ results/metrics/
git add report/results_analysis.md report/notes.md
git commit -m "Add history length sensitivity utilities"
git status --short
```

If src/experiments/run_history_length.py was not created, do not add it.

---

# Prompt 16 — Cost and token metrics

```text
Read PROJECT_SPEC.md, AGENTS.md, CODEX_PROMPTS.md, configs/beauty.yaml, configs/sports.yaml, and src/experiments/run_reproduction.py.

Implement only Prompt 16: cost and token metric utilities.

Create:
- src/evaluation/cost_metrics.py
- tests/test_cost_metrics.py
- configs/cost.yaml if missing

Goal:
Implement safe cost-related utilities for LLM-as-recommender evaluation.

Supported quantities:
- latency seconds;
- prompt tokens;
- completion tokens;
- total tokens;
- estimated cost per query;
- estimated cost per 1,000 users.

Requirements:
1. Do not call APIs.
2. Do not estimate tokens by calling external services.
3. If token counts are provided, aggregate them.
4. If token counts are missing, report token metrics as unavailable.
5. If model prices are configured in configs/cost.yaml, compute estimated costs.
6. If prices are missing, compute token and latency summaries only.
7. Do not invent prices.
8. Do not invent latency.
9. Do not invent token counts.
10. Keep model pricing assumptions explicit in configs/cost.yaml.
11. Do not modify original_repo/.
12. Do not write final experiment outputs in this phase unless a small diagnostic table is useful.

Tests:
Create tests/test_cost_metrics.py using synthetic token, latency, and pricing records.

Test cases must include:
- total token computation;
- cost per query;
- cost per 1,000 users;
- missing prices handled gracefully;
- missing token counts handled gracefully;
- aggregation by model;
- deterministic floating-point behavior with approximate assertions.

After completing the phase, report:
- changed files;
- commands to run;
- expected outputs;
- whether a commit is safe.
```

Success check:

```powershell
python -m compileall src/evaluation/cost_metrics.py
pytest tests/test_cost_metrics.py
pytest tests\
```

Suggested commit:

```powershell
git add src/evaluation/cost_metrics.py tests/test_cost_metrics.py configs/cost.yaml report/notes.md
git commit -m "Add cost metric utilities"
git status --short
```

---

# Prompt 17 — Final safe pipeline orchestrator

```text
Read PROJECT_SPEC.md, AGENTS.md, CODEX_PROMPTS.md, configs/beauty.yaml, configs/sports.yaml, src/experiments/exp00_smoke_test.py, src/data/audit_datasets.py, src/experiments/load_llm_outputs.py, src/baselines/load_baseline_outputs.py, src/baselines/mostpop.py, src/experiments/run_reproduction.py, and any extension runners that exist.

Implement only Prompt 17: final safe pipeline orchestrator.

Create:
- src/experiments/run_pipeline.py
- tests/test_run_pipeline.py

Goal:
Create a safe orchestrator that runs completed project phases in order and records which steps are ready, skipped, blocked, or completed.

The pipeline must be executable as:
- python src/experiments/run_pipeline.py --config configs/beauty.yaml
- python src/experiments/run_pipeline.py --config configs/sports.yaml

Required behavior:
1. Run or verify smoke checks.
2. Run dataset audit or verify existing audit outputs.
3. Run/load LLM outputs if configured or already available.
4. Load baseline outputs if configured or available.
5. Run MostPop only if interactions_path is configured and readable.
6. Run reproduction pipeline.
7. Run coverage extension if implemented.
8. Run position bias extension if implemented and inputs are available.
9. Run history length extension if implemented and inputs are available.
10. Never call APIs by default.
11. Never download data.
12. Never train heavy models.
13. Never modify original_repo/.
14. Do not invent missing inputs.
15. Do not write final metric values unless underlying steps computed real metrics.
16. Write final summary under results/tables/<dataset>_final_summary.csv.
17. Write final metrics under results/metrics/<dataset>_final_results.csv only if at least one real metric exists.
18. Update report/results_analysis.md with a clear explanation of completed, blocked, and missing components.

Tests:
Create tests/test_run_pipeline.py using synthetic temporary files and monkeypatching where needed.

Test cases must include:
- pipeline records blocked steps without crashing;
- no final metrics CSV is written when no real metrics exist;
- final summary is written;
- optional missing extension runners are handled gracefully;
- no API flag defaults to false;
- original_repo is not modified.

After completing the phase, report:
- changed files;
- generated outputs;
- commands to run;
- expected outputs;
- whether a commit is safe.
```

Success check:

```powershell
python -m compileall src/experiments/run_pipeline.py
pytest tests/test_run_pipeline.py
pytest tests\
python src/experiments/run_pipeline.py --config configs/beauty.yaml
python src/experiments/run_pipeline.py --config configs/sports.yaml
```

Expected current behavior:

- Pipeline exits 0.
- It writes final summaries under results/tables/.
- It writes final metrics only if real metrics were actually computed.
- With current missing ground truth/catalog, many evaluation steps should be blocked, not fake.

Suggested commit:

```powershell
git add src/experiments/run_pipeline.py tests/test_run_pipeline.py
git add results/tables/ results/metrics/
git add report/results_analysis.md report/notes.md
git commit -m "Add final safe pipeline orchestrator"
git status --short
```

---

# Prompt 18 — Plot generation

```text
Read PROJECT_SPEC.md, AGENTS.md, CODEX_PROMPTS.md, report/results_analysis.md, and the actual CSV files under results/metrics and results/tables.

Implement only Prompt 18: plot generation.

Create:
- src/experiments/make_plots.py
- tests/test_make_plots.py if practical

Goal:
Generate plots from existing result CSVs only.

Plot candidates:
- HR@5 by model, if HR@5 exists;
- NDCG@5 by model, if NDCG@5 exists;
- Coverage@5 by model, if Coverage@5 exists;
- hallucination rate by model, if hallucination metrics exist;
- NDCG@5 vs Coverage@5 scatter, if both metrics exist;
- position bias plot, if position bias metrics exist;
- history length plot, if history length metrics exist;
- readiness/blocker summary plot, if no real metrics exist.

Requirements:
1. Use matplotlib only.
2. Do not use seaborn.
3. One plot per figure.
4. Do not hardcode absolute paths.
5. Read existing CSVs only.
6. Do not invent metric values.
7. If a metric is unavailable, skip the plot and log the reason.
8. Save plots to results/plots/ with deterministic names.
9. Write a plot summary CSV or Markdown section explaining which plots were generated and which were skipped.
10. Do not modify original_repo/.
11. Do not call APIs.
12. Do not download data.

Tests:
If creating tests, use synthetic CSVs in tmp_path.

Test cases should include:
- plot is generated when a valid metric CSV exists;
- plot is skipped when metric is missing;
- no crash with empty metrics directory;
- deterministic output file names.

After completing the phase, report:
- changed files;
- generated plots;
- skipped plots;
- commands to run;
- expected outputs;
- whether a commit is safe.
```

Success check:

```powershell
python -m compileall src/experiments/make_plots.py
pytest tests\
python src/experiments/make_plots.py --config configs/beauty.yaml
python src/experiments/make_plots.py --config configs/sports.yaml
```

Expected current behavior:

- If no real metrics exist, metric plots are skipped.
- A readiness/blocker plot or summary may be generated if implemented.
- No fake plots are created from invented values.

Suggested commit:

```powershell
git add src/experiments/make_plots.py tests/test_make_plots.py
git add results/plots/ results/tables/
git add report/results_analysis.md report/notes.md
git commit -m "Add result plot generation"
git status --short
```

If tests/test_make_plots.py was not created, do not add it.

---

# Prompt 19 — Repository quality review

```text
Read PROJECT_SPEC.md, AGENTS.md, CODEX_PROMPTS.md, README.md, report/results_analysis.md, report/dataset_audit.md, configs/beauty.yaml, configs/sports.yaml, and the current repository files.

Implement only Prompt 19: quality review.

Create:
- report/archive/phase_reports/quality_review.md

Goal:
Perform a strict quality review of the repository before final documentation.

Check for:
1. Hardcoded absolute paths.
2. Accidental modifications to original_repo/.
3. Scripts that do not run from the project root.
4. Missing tests for implemented modules.
5. API calls possible without --allow-api-calls.
6. Fake results or placeholder metrics written under results/metrics/.
7. Inconsistent schemas between loaders and reproduction pipeline.
8. Missing config fields.
9. Unclear error messages.
10. README mismatch.
11. Results stored in the wrong folder, such as diagnostic summaries under results/metrics/.
12. Empty metric CSVs.
13. Large generated files that should not be committed.
14. Non-deterministic behavior without seed.
15. Report claims unsupported by CSV files.

Allowed fixes:
- Apply only safe, local fixes.
- Fix comments, paths, logs, documentation, tests, or small schema inconsistencies.
- Do not rewrite major pipeline logic unless a serious bug is found.
- Do not modify original_repo/.
- Do not invent results.

report/archive/phase_reports/quality_review.md must include:
- issue id;
- severity: low, medium, high;
- file/location;
- description;
- suggested fix;
- whether fixed now or deferred;
- reason if deferred.

After completing the phase, report:
- issues found;
- fixes applied;
- commands to run;
- whether a commit is safe.
```

Success check:

```powershell
pytest tests\
python src/experiments/exp00_smoke_test.py
```

Suggested commit:

```powershell
git add report/archive/phase_reports/quality_review.md
git add src tests README.md configs report/notes.md
git commit -m "Run repository quality review"
git status --short
```

Only add modified src/tests/README/config files if Codex actually changed them.

---

# Prompt 20 — Final README

```text
Read PROJECT_SPEC.md, AGENTS.md, CODEX_PROMPTS.md, README.md, report/results_analysis.md, report/dataset_audit.md, report/archive/phase_reports/quality_review.md if present, and actual repository files.

Implement only Prompt 20: final README.

Rewrite README.md as the final project README.

The README must include:
1. Project title.
2. Paper reference.
3. Project goal.
4. Scope of the partial reproducibility study.
5. What is reproduced.
6. What is extended.
7. What is not reproduced and why.
8. Repository structure.
9. Environment setup.
10. Exact command sequence.
11. Dataset preparation and configuration.
12. Explanation of configs.
13. Available metrics:
    - HR@5;
    - NDCG@5;
    - hallucination checks;
    - Catalog Coverage@K;
    - optional diagnostics.
14. Output locations:
    - results/raw_outputs/;
    - results/tables/;
    - results/metrics/;
    - results/plots/.
15. Readiness-aware behavior.
16. API/cost safety note.
17. Limitations.
18. Acknowledgment of the original paper and repository.
19. How to reproduce current project state.
20. How to extend the project when ground truth/catalog files become available.

Rules:
1. Do not include numerical results unless read from existing CSV files.
2. Clearly distinguish real computed metrics from diagnostic/readiness summaries.
3. Do not claim that full reproduction was achieved if ground truth/catalog is missing.
4. Do not oversell publication readiness.
5. Do not modify code unless necessary to keep README commands correct.

After completing the phase, report:
- changed files;
- commands to run;
- whether a commit is safe.
```

Success check:

```powershell
python src/experiments/exp00_smoke_test.py
pytest tests\
```

Suggested commit:

```powershell
git add README.md report/notes.md
git commit -m "Finalize README instructions"
git status --short
```

---

# Prompt 21 — Reproducibility checklist

```text
Read PROJECT_SPEC.md, AGENTS.md, README.md, report/results_analysis.md, report/dataset_audit.md, report/archive/phase_reports/quality_review.md if present, and generated CSVs under results/tables and results/metrics.

Implement only Prompt 21: reproducibility checklist.

Create:
- reproducibility_checklist.md

Goal:
Create an honest, exam-ready reproducibility checklist.

The checklist must cover:
1. Environment.
2. Python version.
3. Dependencies.
4. Repository structure.
5. Config files.
6. Random seeds.
7. Dataset availability.
8. Dataset audit.
9. Original repository inspection.
10. LLM outputs.
11. Baseline outputs.
12. MostPop availability.
13. Ground truth availability.
14. Catalog/candidate availability.
15. Metric implementations and tests.
16. Reproduction pipeline.
17. Coverage extension.
18. Optional bias/history/cost analyses if implemented.
19. Result provenance.
20. Plots.
21. Report.
22. Limitations.
23. What was reproduced.
24. What was approximated.
25. What was extended.
26. What could not be reproduced and why.
27. Commands needed to rerun the project.
28. Oral exam talking points.

Use checkboxes.

Rules:
1. Do not invent completed items.
2. Mark incomplete items honestly.
3. If a metric is blocked, state the exact missing prerequisite.
4. Do not modify original_repo/.
5. Do not create fake results.

After completing the phase, report:
- changed files;
- commands to run;
- whether a commit is safe.
```

Success check:

```powershell
dir reproducibility_checklist.md
```

Suggested commit:

```powershell
git add reproducibility_checklist.md report/notes.md
git commit -m "Add reproducibility checklist"
git status --short
```

---

# Prompt 22 — Technical report

```text
Read PROJECT_SPEC.md, AGENTS.md, README.md, report/results_analysis.md, report/dataset_audit.md, report/archive/phase_reports/quality_review.md if present, reproducibility_checklist.md if present, and generated CSVs under results/tables and results/metrics.

Implement only Prompt 22: final technical report draft.

Create:
- report/report.md

Goal:
Write an exam-ready technical report for the reproducibility project.

Required structure:
1. Title.
2. Abstract.
3. Introduction.
4. Background.
5. Original paper.
6. Reproducibility objective.
7. Scope and rationale for partial reproduction.
8. Experimental setup.
9. Repository and implementation design.
10. Dataset audit and data availability.
11. Reproduced components.
12. Blocked components and reasons.
13. Metrics:
    - HR@5;
    - NDCG@5;
    - hallucination;
    - Catalog Coverage@K extension;
    - optional diagnostics.
14. Results.
15. Extension: Catalog Coverage@K.
16. Discussion.
17. Threats to validity.
18. Reproducibility checklist summary.
19. Conclusion.
20. Appendix with commands.

Rules:
1. Do not invent numerical results.
2. Every numerical claim must come from generated CSVs.
3. Clearly separate:
   - original paper claims;
   - implemented reproduction pipeline;
   - actually computed results;
   - diagnostic/readiness findings;
   - project extension.
4. If HR@5/NDCG@5 are blocked due to missing ground truth, say so explicitly.
5. If Coverage@K is blocked due to missing catalog_path, say so explicitly.
6. Do not claim full reproduction if only partial reproduction/readiness was achieved.
7. Discuss why this is still a valid reproducibility engineering project.
8. Do not modify code.
9. Do not modify original_repo/.

After completing the phase, report:
- changed files;
- sections included;
- whether the report contains unsupported claims;
- commands to run;
- whether a commit is safe.
```

Success check:

```powershell
dir report\report.md
```

Suggested commit:

```powershell
git add report/report.md report/notes.md
git commit -m "Draft final technical report"
git status --short
```

---

# Prompt 23 — Results analysis refinement

```text
Read generated CSV files under results/metrics and results/tables, report/results_analysis.md, report/report.md, README.md, and reproducibility_checklist.md.

Implement only Prompt 23: results analysis refinement.

Update:
- report/results_analysis.md

Goal:
Improve the results analysis so that it is precise, honest, and useful for the oral exam.

The analysis must answer:
1. What was reproduced?
2. Which data artifacts were successfully recovered?
3. Which LLM outputs were normalized?
4. Which baseline outputs were normalized?
5. Which datasets are ready and which are blocked?
6. Why are HR@5 and NDCG@5 blocked or unavailable?
7. Why is hallucination evaluation blocked or unavailable?
8. Why is Coverage@K important?
9. Does Coverage@K change interpretation of utility metrics?
10. What does the current pipeline prove even without full metric computation?
11. What are the major reproducibility gaps?
12. What should be emphasized orally?
13. What would be needed to complete full reproduction?
14. Which parts are robustly tested?
15. Which parts are extensions beyond the original reproduction?

Rules:
1. Do not invent numbers.
2. If a metric is missing, say it is missing and explain why.
3. Use only generated CSVs and repository files as evidence.
4. Keep claims exam-ready.
5. Do not oversell publication readiness.
6. Do not modify code.
7. Do not modify original_repo/.

After completing the phase, report:
- changed files;
- key improvements;
- whether unsupported claims were avoided;
- whether a commit is safe.
```

Success check:

```powershell
dir report\results_analysis.md
```

Suggested commit:

```powershell
git add report/results_analysis.md report/notes.md
git commit -m "Refine results analysis"
git status --short
```

---

# Prompt 24 — Slide outline for 10-minute oral presentation

```text
Read PROJECT_SPEC.md, README.md, report/report.md, report/results_analysis.md, reproducibility_checklist.md, and generated CSVs under results/tables and results/metrics.

Implement only Prompt 24: slide outline for oral presentation.

Create:
- report/slides_outline.md

Goal:
Create an 8–10 slide outline for a 10-minute oral presentation.

Required storyline:
1. Motivation.
2. Original paper.
3. Why partial reproducibility.
4. Project architecture.
5. Data and artifact recovery.
6. Reproduction pipeline and readiness checks.
7. Reproduced/available outputs.
8. Extension: Catalog Coverage@K.
9. Limitations and blocked metrics.
10. Conclusion and future work.

For each slide include:
- title;
- 3–5 bullet points;
- what to say orally;
- expected time;
- supported evidence from project files or CSVs;
- warning if a claim depends on missing data.

Rules:
1. Do not include unsupported numerical results.
2. Do not claim full reproduction if metrics are blocked.
3. Emphasize engineering rigor, tests, artifact normalization, readiness checks, and honest limitations.
4. Make it suitable for a university exam.
5. Do not modify code.
6. Do not modify original_repo/.

After completing the phase, report:
- changed files;
- number of slides;
- whether unsupported numerical claims were avoided;
- whether a commit is safe.
```

Success check:

```powershell
dir report\slides_outline.md
```

Suggested commit:

```powershell
git add report/slides_outline.md report/notes.md
git commit -m "Add oral presentation outline"
git status --short
```

---

# Prompt 25 — Optional short paper plan

```text
Read PROJECT_SPEC.md, report/report.md, report/results_analysis.md, README.md, reproducibility_checklist.md, and generated CSVs under results/tables and results/metrics.

Implement only Prompt 25: optional short paper plan.

Create:
- report/short_paper_plan.md

Goal:
Frame a possible future short paper only if the project evidence is interesting enough.

The plan must include:
1. Possible title.
2. Abstract draft.
3. Contribution list.
4. Paper structure.
5. What is already supported by the current project.
6. What is not yet supported.
7. Additional experiments needed.
8. Required data to complete the study.
9. Potential venues/workshops to investigate later.
10. Why the current exam project is not yet a full publication.
11. Risks.
12. Missing evidence.
13. Ethical/reproducibility considerations.
14. Next steps.

Rules:
1. Be honest.
2. Do not oversell the project.
3. Do not invent results.
4. Do not claim publication readiness unless supported by real metrics.
5. Clearly state that missing ground truth/catalog can limit quantitative claims.
6. Do not modify code.
7. Do not modify original_repo/.

After completing the phase, report:
- changed files;
- whether the plan is optimistic or conservative;
- additional experiments required;
- whether a commit is safe.
```

Success check:

```powershell
dir report\short_paper_plan.md
```

Suggested commit:

```powershell
git add report/short_paper_plan.md report/notes.md
git commit -m "Add optional short paper plan"
git status --short
```

---

# Plan C Prompt C6 - Reconstructed baseline evaluation

Purpose:

- Evaluate deterministic non-LLM baselines on Plan C reconstructed canonical
  datasets.
- Keep Plan C outputs separate from Plan A readiness diagnostics and Plan B
  ML-1M extension outputs.
- Report real metrics only when reconstructed ground truth, candidates, and
  baseline rankings exist.

Implemented scope:

- Runner: `src/experiments/run_plan_c_baselines.py`.
- Baselines: Random, MostPop, ContentBM25.
- Metrics: HR@5, NDCG@5, and Catalog Coverage@5 when catalog data exists.
- Outputs:
  - `results/metrics/plan_c_<dataset>_<baseline>_metrics.csv`;
  - `results/tables/plan_c_baseline_*.csv`;
  - `results/plots/plan_c_baseline*.png`;
  - ignored raw rankings under
    `results/raw_outputs/plan_c_*_recommendations.csv`.

Constraints:

- Do not modify `original_repo/`.
- Do not download data, call APIs, train models, or fabricate metrics.
- Do not overwrite Plan B metric filenames.
- Mark `--max-users` runs as diagnostic subsets.
- Treat LastFM as blocked for utility metrics until valid ground truth and
  candidates exist.

Validation:

```powershell
python -m pytest tests/
python src/experiments/run_plan_c_baselines.py --datasets ml1m --baselines random mostpop content_bm25
python src/experiments/run_plan_c_baselines.py --datasets amazon_beauty --baselines random mostpop --max-users 1000
```

Follow-up safety fix:

- Diagnostic `--max-users N` runs must not write the same filenames as full
  runs.
- Full metrics use
  `results/metrics/plan_c_<dataset>_<baseline>_metrics.csv`.
- Diagnostic metrics use
  `results/metrics/plan_c_<dataset>_<baseline>_diagnostic_max_users_<N>_metrics.csv`.
- Metric, comparison, status, and blocker rows must include
  `diagnostic_subset`, `max_users`, `run_scope`, `evaluated_users`, and
  `recommendation_rows` where applicable.
- Diagnostic rows are stability/runtime checks only; final baseline claims
  should use full non-diagnostic rows.

Candidate-universe correction before commit:

- Audit C5 candidate generation before interpreting full Amazon MostPop
  metrics.
- The earlier sampler used catalog item IDs, which is risky for Amazon because
  metadata-only items can dominate the negative pool and have zero training
  popularity.
- Add `negative_sampling.item_universe`, defaulting to `interaction_items`.
- Supported values:
  - `interaction_items`: observed interaction item IDs, excluding each user's
    full history;
  - `catalog_items`: explicit experimental mode only.
- Add C5 summary diagnostics for candidate universe size and insufficient
  negative pools.
- Add C6 candidate diagnostics for candidates outside the reconstructed
  interaction universe and zero-training-popularity candidate rate.
- Supersede any suspicious Amazon MostPop metrics generated before the
  interaction-item candidate reconstruction.

---

# Plan C Prompt C7 - Final consolidation and exam-ready reporting

Purpose:

- Consolidate Plan C into final reports, tables, plots, and oral-exam guidance.
- Preserve the distinction between exact reproduction blockers and
  reconstructed-dataset evidence.

Implemented scope:

- Script: `src/experiments/summarize_plan_c_final_results.py`.
- Reports:
  - `report/plan_c_final_report.md`;
  - `report/oral_exam_plan_c_guide.md`.
- Tables:
  - `results/tables/plan_c_final_results_overview.csv`;
  - `results/tables/plan_c_claims_and_limitations.csv`;
  - `results/tables/plan_c_final_experiment_inventory.csv`.
- Plots:
  - `results/archive/plots/plan_c_final_hr_at_5_overview.png`;
  - `results/plots/plan_c_final_ndcg_at_5_overview.png`.

Constraints:

- Do not modify `original_repo/`.
- Do not download data, call APIs, train models, or fabricate metrics.
- Do not rewrite Plan A or Plan B metric evidence.
- State that exact Beauty/Sports reproduction remains blocked.

Validation:

```powershell
python -m pytest tests/
python src/experiments/summarize_plan_c_final_results.py
```
