# Reproducibility Audit Current State

Audit timestamp: 2026-07-10T14:06:06.9014152+02:00  
Scope: audit-only current-state review of repository artifacts for the university reproducibility project based on *Beyond Utility: Evaluating LLM as Recommender*.

This audit did not attempt to improve, rerun, download, train, or infer any model. It records what is currently present, what has been executed previously, what can be validated in the current environment, and what remains blocked.

## Initial git status

Pre-write git state was checked immediately before creating the audit artifacts:

```text
git status --short --branch
## plan-d-extensions
?? "paper/Beyond Utility Evaluating LLM As Recommander.pdf"
```

```text
git branch --show-current
plan-d-extensions
```

```text
git log -1 --format="%H %s"
02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests
```

Existing recovery tags observed:

```text
plan-a-readiness-v1
plan-b-reconstruction-v1
plan-c-reconstruction-v1
```

The untracked PDF under `paper/` existed before this audit and was preserved.

## Repository areas inspected

Top-level repository areas inspected:

- `src/`
- `tests/`
- `configs/`
- `data/`
- `results/`
- `report/`
- `original_repo/EvaLLMasRecommender/`
- `README.md`
- `README_NEXT_STEPS.md`
- `PROJECT_SPEC.md`
- `AGENTS.md`
- `requirements.txt`
- `environment.yml`
- `.gitignore`

Specific evidence reviewed included:

- Local metric implementations in `src/evaluation/`
- Local baseline and analysis code in `src/baselines/`, `src/analysis/`, and `src/llm/`
- Dataset configs: `configs/beauty.yaml`, `configs/sports.yaml`, `configs/ml1m.yaml`, `configs/plan_c_reconstruction.yaml`, and `configs/cost.yaml`
- Current tests under `tests/`
- Generated metrics and tables under `results/metrics/` and `results/tables/`
- Phase reports under `report/`
- Upstream original repository README, requirements, prompt templates, LLM wrappers, metric code, zipped LLM outputs, and Beauty SOTA outputs

Original repository evidence was inspected read-only. A nested git status check on `original_repo/EvaLLMasRecommender` was blocked by Git safe-directory ownership protection, and no `safe.directory` change was made.

## Files created

Two files were created by this audit:

- `report/reproducibility_audit_current_state.md`
- `results/tables/reproducibility_component_matrix.csv`

## Files modified

No pre-existing tracked files were modified.

No files under `original_repo/` were modified. No raw data files were modified. No prior result files were modified. The pre-existing untracked PDF under `paper/` was not changed.

## Commands executed

All commands were read-only except the two `apply_patch` operations used to create the audit artifacts.

Repository status and provenance:

```powershell
git status --short --branch
git branch --show-current
git log -1 --format="%H %s"
git tag --list
Get-Date -Format o
git -C original_repo\EvaLLMasRecommender status --short --branch
```

Repository inventory and source inspection:

```powershell
Get-ChildItem -Force
rg --files
Get-Content README.md
Get-Content requirements.txt
Get-Content environment.yml
Get-Content configs\beauty.yaml
Get-Content configs\sports.yaml
Get-Content configs\ml1m.yaml
Get-Content configs\plan_c_reconstruction.yaml
Get-Content configs\cost.yaml
Get-Content original_repo\EvaLLMasRecommender\README.md
Get-Content original_repo\EvaLLMasRecommender\requirements.txt
```

Python environment investigation:

```powershell
where.exe python
where.exe python3
where.exe py
Get-Command python -All
python --version
.\.venv\Scripts\python.exe --version
Get-Content .venv\pyvenv.cfg
Get-ChildItem .venv\Scripts
py -0p
Get-Command py -All
where.exe conda
C:\ProgramData\anaconda3\python.exe --version
C:\ProgramData\anaconda3\Scripts\conda.exe env list
```

Test commands:

```powershell
$env:PYTHONNOUSERSITE='1'; C:\ProgramData\anaconda3\python.exe -m pytest tests/
$env:PYTHONNOUSERSITE='1'; C:\ProgramData\anaconda3\python.exe -m pytest -q tests/test_utility_metrics.py tests/test_output_parser.py tests/test_hallucination.py tests/test_coverage.py tests/test_position_bias.py tests/test_cost_metrics.py tests/test_history_length.py
$env:PYTHONNOUSERSITE='1'; C:\ProgramData\anaconda3\python.exe -m pytest -q tests/test_config.py tests/test_schemas.py tests/test_dataset_audit.py tests/test_load_llm_outputs.py tests/test_load_baseline_outputs.py tests/test_mostpop.py tests/test_random_baseline.py tests/test_ml1m_mostpop_baseline.py tests/test_content_bm25_baseline.py tests/test_make_plots.py tests/test_run_pipeline.py tests/test_run_reproduction.py tests/test_prepare_ml1m.py tests/test_run_ml1m_reconstruction.py
$env:PYTHONNOUSERSITE='1'; C:\ProgramData\anaconda3\python.exe -m pytest -q tests/test_check_plan_c_source_urls.py tests/test_inspect_plan_c_downloads.py tests/test_validate_plan_c_artifacts.py tests/test_validate_plan_c_raw_datasets.py tests/test_reconstruct_plan_c_datasets.py tests/test_run_plan_c_baselines.py tests/test_summarize_plan_c_final_results.py
$env:PYTHONNOUSERSITE='1'; C:\ProgramData\anaconda3\python.exe -m pytest -q tests/test_plan_d_beyond_utility_metrics.py tests/test_plan_d_exam_readiness_review.py tests/test_plan_d_final_aggregation.py tests/test_plan_d_history_length_sensitivity.py tests/test_plan_d_llm_model_registry.py tests/test_plan_d_llm_prompt_templates.py tests/test_plan_d_llm_ranking_manifest.py tests/test_plan_d_llm_reranking_diagnostic.py tests/test_plan_d_mistral_smoke_test.py tests/test_plan_d_qwen_smoke_test.py tests/test_plan_d_qwen_vs_mistral_diagnostic.py tests/test_plan_d_qwen2_candidate_position_bias_diagnostic.py tests/test_plan_d_qwen2_ml1m_ranking_diagnostic.py tests/test_plan_d_qwen2_output_validity_analysis.py tests/test_plan_d_qwen2_strict_prompt_diagnostic.py tests/test_plan_d_reproducibility_matrix.py
$env:PYTHONNOUSERSITE='1'; C:\ProgramData\anaconda3\python.exe -m pytest -q tests/test_plan_e_paper_condition_alignment.py
$env:PYTHONNOUSERSITE='1'; C:\ProgramData\anaconda3\python.exe -m pytest -q tests/test_plan_e_paper_like_candidate_pools.py
$env:PYTHONNOUSERSITE='1'; C:\ProgramData\anaconda3\python.exe -m pytest -q tests/test_plan_e_baseline_alignment.py
$env:PYTHONNOUSERSITE='1'; C:\ProgramData\anaconda3\python.exe -m pytest -q tests/test_plan_e_bm25_scoring.py
$env:PYTHONNOUSERSITE='1'; C:\ProgramData\anaconda3\python.exe -m pytest -q tests/test_plan_e_full_data_inventory.py
```

CSV validation commands:

```powershell
$rows = Import-Csv results\tables\reproducibility_component_matrix.csv; $rows.Count
$expected = @('component_id','phase','category','component','paper_expected_condition','repository_evidence','implementation_status','execution_status','alignment_status','classification','datasets','models_or_methods','sample_scale','metrics_or_outputs','tests','blocking_reason','main_gap','recommended_next_action','confidence','notes'); $actual = (Import-Csv results\tables\reproducibility_component_matrix.csv | Select-Object -First 1).PSObject.Properties.Name; Compare-Object $expected $actual
$allowed = @('reproduced','adapted','blocked','extension','not_attempted'); Import-Csv results\tables\reproducibility_component_matrix.csv | Where-Object { $allowed -notcontains $_.classification } | Select-Object component_id,classification
Import-Csv results\tables\reproducibility_component_matrix.csv | Where-Object { $_.component_id -eq '' -or $_.component -eq '' -or $_.classification -eq '' -or $_.recommended_next_action -eq '' } | Select-Object component_id,component,classification,recommended_next_action
Import-Csv results\tables\reproducibility_component_matrix.csv | Group-Object classification | Sort-Object Name | Select-Object Name,Count
```

## Python environment investigation

The default Python environment is not reproducible as configured:

- `where.exe python`, `where.exe python3`, and `where.exe py` did not find usable executables.
- `Get-Command python -All` resolved only a WindowsApps Python stub.
- `python --version` failed with a WindowsApps login-session error.
- `.venv\Scripts\python.exe --version` failed because the venv points to a missing WindowsApps Python 3.12 path.
- `.venv\pyvenv.cfg` points to `C:\Users\Bartolo\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe`.
- `py` was not available.

A usable Anaconda Python was found:

- `C:\ProgramData\anaconda3\python.exe --version` returned Python 3.12.4.
- With `PYTHONNOUSERSITE=1`, the Anaconda environment could import `pandas`, `numpy`, `yaml`, and `pytest`.
- Confirmed package versions: pandas 2.2.2, numpy 1.26.4, PyYAML 6.0.1, pytest 7.4.4.

Important blocker:

- `torch` is not installed in the working Anaconda environment.
- `requirements.txt` and `environment.yml` both declare torch, but the current executable used for validation cannot import it.
- Full test collection is therefore blocked for the torch-dependent Plan D neural tests.

## Test results

Full suite attempt:

```text
Command: $env:PYTHONNOUSERSITE='1'; C:\ProgramData\anaconda3\python.exe -m pytest tests/
Result: failed during collection
Collected: 441 items before collection errors
Errors: 8
Cause: ModuleNotFoundError: No module named 'torch'
```

Torch-blocked test modules:

- `tests/test_plan_d_bprmf.py`
- `tests/test_plan_d_gru4rec.py`
- `tests/test_plan_d_lightgcn.py`
- `tests/test_plan_d_sasrec.py`
- `tests/test_run_plan_d_bprmf.py`
- `tests/test_run_plan_d_gru4rec.py`
- `tests/test_run_plan_d_lightgcn.py`
- `tests/test_run_plan_d_sasrec.py`

Broad non-torch aggregate attempts excluding those files timed out at 120 seconds and again at 300 seconds. Individual bounded groups were then run and passed.

Bounded validation results:

| Group | Result |
| --- | --- |
| Core metrics, parsers, hallucination, coverage, position bias, cost, history length | 75 passed, 2 skipped in 1.60s |
| Config, schemas, data audit, loaders, non-neural baselines, plots, pipelines | 78 passed in 13.10s |
| Plan C source, validation, reconstruction, baselines, final summary | 53 passed in 6.31s |
| Plan D non-torch beyond utility and local LLM diagnostics | 142 passed in 12.28s |
| Plan E paper condition alignment | 15 passed in 10.55s |
| Plan E paper-like candidate pools | 14 passed in 235.48s |
| Plan E baseline alignment | 13 passed in 226.83s |
| Plan E BM25-like scoring | 13 passed in 342.56s |
| Plan E full data inventory | 14 passed in 223.28s |

Feasible bounded test total:

```text
417 passed, 2 skipped
```

This is not the same as a full-suite pass because the torch-dependent tests do not collect in the current environment.

## CSV validation results

The component matrix was validated after creation.

CSV path:

```text
results/tables/reproducibility_component_matrix.csv
```

Validation results:

- Row count: 96
- Required columns: present exactly
- Invalid `classification` values: none
- Missing critical fields checked: none
- Allowed classifications used: `adapted`, `blocked`, `extension`, `not_attempted`
- No row is classified as exact `reproduced`, because the audit did not find the official processed artifacts and exact candidate pools needed to justify that label.

Classification counts:

| Classification | Count |
| --- | ---: |
| adapted | 42 |
| blocked | 16 |
| extension | 35 |
| not_attempted | 3 |

## Key verified findings

Exact same-paper reproduction is currently blocked. The most important reason is that the exact official processed Beauty and Sports artifacts and original paper candidate pools are not present locally.

The repository has a strong reconstructed and extension pipeline:

- Plan C reconstructs public raw Amazon Beauty, Amazon Sports, ML1M, and partial LastFM artifacts.
- Plan C provides reconstructed 100-candidate pools for Beauty, Sports, and ML1M.
- Plan E provides paper-like reconstructed 20-candidate manifests for Beauty, Sports, and ML1M, with 1000 users per dataset and task.
- Plan E3B provides BM25-like scoring over those 20-candidate manifests.
- Plan D provides neural baseline diagnostics and local LLM diagnostics, but not paper-exact full-scale LLM reproduction.

Data availability verified:

- `data/reconstructed/plan_c/amazon_beauty`: interactions, ground truth, candidates, and catalog exist.
- `data/reconstructed/plan_c/amazon_sports`: interactions, ground truth, candidates, and catalog exist.
- `data/processed/ml1m` and `data/reconstructed/plan_c/ml1m`: interactions, ground truth, candidates, and catalog exist.
- `data/reconstructed/plan_c/lastfm`: observed interactions and catalog exist, but no ground truth or candidates.

Upstream artifacts verified read-only:

- `original_repo/EvaLLMasRecommender/llm_results/beauty.zip` contains Beauty LLM outputs for models including `glm4`, `gpt3.5`, `gpt4`, and `llama3`.
- `original_repo/EvaLLMasRecommender/llm_results/sports.zip` contains Sports LLM outputs for models including `glm4`, `gpt3.5`, `gpt4`, `llama2`, and `llama3`.
- `original_repo/EvaLLMasRecommender/sota_results/beauty_his10.zip` contains Beauty SOTA recommendation outputs including BPRMF, GRU4Rec, LightGCN, SASRec, and NRMS.
- No matching Sports SOTA archive was found.

Metric implementation verified:

- HR@K and NDCG@K are implemented and tested.
- Output parsing, hallucination, coverage, position bias, cost, and history-length utilities are implemented and tested.
- Beyond-utility extensions exist for catalog coverage, item coverage, popularity, novelty, long-tail rate, diversity proxy, duplicate rate, and candidate validity.

No numerical results were invented in this audit. Numeric findings in this report are either command outputs, row counts, test results, or summaries of existing CSV artifacts.

## Critical missing components

The following missing or incomplete components prevent an exact reproduction claim:

1. Official processed Beauty and Sports artifacts from the original paper condition.
2. Exact original 20-candidate pools and candidate ordering for paper users.
3. A complete local metric table generated from the upstream archived LLM outputs.
4. Exact prompt-to-output provenance for every upstream archived LLM file.
5. Exact paper BM25 implementation and text-processing path.
6. Exact paper-wide neural baseline reruns aligned to paper candidates.
7. Full LLM reproduction for GPT-4o-mini, GPT-4o, Claude 3 Haiku, Llama3-70B, Qwen2-7B, and Mistral-7B across paper-like datasets and tasks.
8. A working default Python environment with torch available.
9. A final consolidated reproduction table that separates exact, adapted, blocked, and extension results.

## Blocked components and exact reasons

Official Beauty and Sports reproduction:

- Blocked because the official processed artifacts and exact candidate pools are not present.
- Public raw reconstructions exist, but they cannot prove exact user sets, candidate negatives, or ordering.

LastFM utility evaluation:

- Blocked because current LastFM reconstruction has observed interactions and catalog only.
- There is no ground truth file and no candidate pool.

Full test suite:

- Blocked because `torch` is missing from the working Anaconda environment.
- Eight torch-dependent Plan D neural test modules fail during collection.

Default Python environment:

- Blocked because the default WindowsApps Python stub is not usable and `.venv` points to a missing WindowsApps Python executable.

Exact BM25 reproduction:

- Blocked because the exact original BM25 implementation and input protocol were not verified.
- Plan E3B is BM25-like reconstructed scoring, not certified exact BM25.

Full paper LLM reproduction:

- Blocked because no API calls are allowed by default and no complete locally audited metric table exists for all paper LLM conditions.
- Upstream saved outputs exist for some Beauty and Sports models, but they still need read-only ingestion, parsing, provenance mapping, and metric regeneration.

LLMSRec or ICL condition:

- Not attempted locally.
- The local prompt template module contains a documented stub, not an executed condition.

## Potentially misleading, stale or superseded claims

`README.md` has a stale test status:

- It states `207 passed, 2 skipped`.
- This audit found `417 passed, 2 skipped` across feasible bounded groups, plus a full-suite collection blocker caused by missing `torch`.
- The README was not updated because this task is audit-only.

`report/plan_e_full_data_inventory.md` is partly superseded:

- It states that no complete 1000-user paper-like 20-candidate manifests exist.
- Plan E2 manifests now exist for Beauty, Sports, and ML1M ranking and reranking.

`results/tables/plan_e_missing_conditions.csv` is partly superseded:

- It includes `MISS_1000_CANDIDATE_MANIFESTS`.
- That condition is stale because Plan E2 manifests now exist.

Some older Plan D summaries may not include Plan E3B:

- Plan D oral or readiness material is still useful, but should not be treated as the latest complete project state without adding Plan E2 and Plan E3B findings.

Any wording that implies exact paper reproduction for Beauty or Sports should be treated as too strong unless it explicitly says reconstructed or adapted.

## Recommended single next task

Generate a consolidated non-API Plan E results summary table from existing CSV artifacts, using the 1000-user paper-like 20-candidate manifests as the alignment point.

The task should:

- Use existing Plan E manifest, validation, MostPop/reference, and BM25-like CSV outputs.
- Clearly label every row as reconstructed paper-like, adapted, blocked, or extension.
- Avoid any API call, model pull, model training, or dataset download.
- Refresh the stale Plan E inventory and missing-condition claims after the table is generated.

This is the best next task because it creates the strongest current paper-aligned evidence without depending on unavailable official artifacts, torch installation, paid APIs, or local LLM inference.

## Final git status

Verified final git status after this audit artifact creation:

```text
## plan-d-extensions
?? "paper/Beyond Utility Evaluating LLM As Recommander.pdf"
?? report/reproducibility_audit_current_state.md
?? results/tables/reproducibility_component_matrix.csv
```

No commit was created.

## Safety confirmations

- No external APIs were called.
- No datasets were downloaded.
- No models were downloaded.
- No Ollama models were pulled.
- No training was started.
- No inference was started.
- No files under `original_repo/` were modified.
- No raw data files were modified.
- No existing result files were modified.
- No README or report files other than this new audit report were modified.
- The pre-existing untracked PDF `paper/Beyond Utility Evaluating LLM As Recommander.pdf` was preserved.
- No commit was created.
