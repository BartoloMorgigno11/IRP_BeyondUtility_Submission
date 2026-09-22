# AGENTS.md

# Instructions for Codex / AI coding agents

This repository implements a partial reproducibility and extension project for *Beyond Utility: Evaluating LLM as Recommender*.

The goal is to build a clean, modular, reproducible evaluation wrapper around the original repository, not to rewrite the original work or invent unsupported results.

## 1. Core operating principles

1. Do not modify `original_repo/` unless explicitly instructed.
2. All new source code must go under `src/`.
3. All tests must go under `tests/`.
4. All generated results must go under `results/`.
5. All report material must go under `report/`.
6. All dataset-dependent settings must be controlled through YAML configs under `configs/`.
7. Every script must be executable from the project root.
8. Do not hardcode absolute local paths.
9. Do not invent numerical results.
10. Do not manually type final metric values into reports unless they are read from generated CSV files.

## 2. Repository boundaries

### Allowed to modify

- `src/`
- `tests/`
- `configs/`
- `report/`
- `README.md`
- `PROJECT_SPEC.md`
- `AGENTS.md`
- `CODEX_PROMPTS.md`
- `reproducibility_checklist.md`
- `.gitignore`
- `requirements.txt`
- `environment.yml`

### Not allowed to modify by default

- `original_repo/`
- raw data files under `data/raw/`
- downloaded paper files under `paper/`
- committed results from previous runs, unless the current task explicitly regenerates them

If a change to `original_repo/` is unavoidable, first document why in `report/notes.md` and propose the smallest possible patch.

## 3. Required project structure

```text
IRP_BeyondUtility_Project/
├── original_repo/
│   └── EvaLLMasRecommender/
├── paper/
│   └── 2411.00331v1.pdf
├── src/
├── configs/
├── data/
├── results/
├── tests/
├── report/
├── README.md
├── PROJECT_SPEC.md
├── AGENTS.md
├── CODEX_PROMPTS.md
├── reproducibility_checklist.md
├── requirements.txt
├── environment.yml
└── .gitignore
```

Create missing folders when needed, but do not create unnecessary nested complexity.

## 4. Coding style

Use Python 3.10+, `pathlib.Path`, `argparse`, `yaml.safe_load`, `pandas`, `numpy`, and `logging`. Prefer simple, readable code over clever abstractions. Use type hints for public functions and docstrings for metric functions and parsers.

## 5. CLI conventions

Every experiment script must support:

```bash
python path/to/script.py --config configs/beauty.yaml
```

If a script may call an LLM/API, it must require:

```bash
--allow-api-calls
```

Without this flag, the script must fail safely or use cached/saved outputs.

Recommended common arguments: `--config`, `--output-dir`, `--sample-size`, `--seed`, `--run-id`, `--allow-api-calls`.

## 6. Config rules

All dataset-specific paths and experimental settings must be in YAML. Never require users to edit Python files to change paths or settings.

Minimal config shape:

```yaml
dataset:
  name: beauty
  raw_dir: data/raw/beauty
  processed_dir: data/processed/beauty
  sample_dir: data/samples/beauty
  catalog_path: null
  interactions_path: null
  candidates_path: null
  ground_truth_path: null

original_repo:
  root: original_repo/EvaLLMasRecommender

experiment:
  task: ranking
  k_values: [5]
  seed: 42
  sample_size: null
  allow_api_calls: false

models:
  llm_outputs: []
  baselines: [mostpop]

metrics:
  utility: [hr, ndcg]
  beyond_utility: [hallucination, coverage]
```

## 7. Data schema rules

Use canonical internal dataframes where possible.

Interactions: `user_id`, `item_id`, `split`, optional `timestamp`, `rating`.

Candidates: `user_id`, `candidate_item_id`, `candidate_position`, optional `is_ground_truth`, `candidate_item_title`.

Recommendations: `user_id`, `model_name`, `dataset`, `rank`, `recommended_item_id` or `recommended_item_text`, `run_id`, optional `raw_output`, `match_type`, `source_file`, `prompt_variant`, `candidate_ordering`, `history_length`.

Ground truth: `user_id`, `ground_truth_item_id`, optional `dataset`, `split`.

## 8. Metric rules

Every metric must be implemented in `src/evaluation/`, have a docstring, handle empty predictions, handle duplicated recommendations deterministically, have tests with known expected values, and return machine-readable outputs suitable for CSV.

Exact, normalized and fuzzy matching must be reported separately. Do not mix them silently.

## 9. Result provenance

Every experiment output must include dataset name, model name, task, k value, metric name, metric value, run id, timestamp, config path, source output file if applicable, and git commit hash if easy to retrieve.

Recommended CSV columns:

```text
run_id,dataset,task,model_name,prompt_variant,candidate_ordering,history_length,k,metric,value,source_file,config_path,timestamp
```

## 10. Output naming

Use deterministic file names, for example `results/metrics/beauty_reproduction_metrics.csv`, `results/tables/final_summary.csv`, `results/plots/beauty_hr_at_5.png`, `report/results_analysis.md`.

## 11. API and secret handling

Never commit API keys. Never create `.env` with real secrets. Use environment variables. Add `.env` to `.gitignore`. API scripts must not run unless `--allow-api-calls` is present. Prefer saved outputs and cached results.

## 12. Testing requirements

Run tests with:

```bash
pytest tests/
```

Minimum tests: HR@K, NDCG@K, output parser, hallucination, coverage.

Do not proceed to final pipeline until metric tests pass.

## 13. Documentation requirements

Update `report/notes.md` after every major phase with what changed, commands run, outputs generated, known issues and next step. Update `README.md` only after the pipeline is stable.

## 14. Plotting rules

Use matplotlib only. One chart per figure. No seaborn. Do not set custom colors unless explicitly requested. Save plots under `results/plots/`. Every plot must be generated from CSV results.

## 15. Quality review checklist

Before declaring a task complete, verify that scripts run from root, paths are config-driven, tests pass, outputs are saved correctly, no API calls happen by default, no numerical result is invented, no original repo files were modified accidentally, reproduced results and extensions are separated, and limitations are documented.

## 16. Commit discipline

After each stable phase, suggest a Git commit. Do not create commits automatically unless explicitly instructed by the user.

Suggested commit messages:

```text
git commit -m "Initialize project scaffold"
git commit -m "Add dataset audit utilities"
git commit -m "Implement utility metrics with tests"
git commit -m "Add LLM output parser and hallucination metrics"
git commit -m "Add MostPop baseline and reproduction pipeline"
git commit -m "Add coverage extension and plots"
git commit -m "Finalize report and reproducibility checklist"
```
