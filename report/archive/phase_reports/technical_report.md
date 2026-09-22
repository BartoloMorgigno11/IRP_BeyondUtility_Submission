# A Readiness-Aware Reproducibility Study of Beyond Utility: Evaluating LLM as Recommender

## Abstract

This project studies the reproducibility of *Beyond Utility: Evaluating LLM as Recommender* from a constrained, readiness-aware perspective. The original paper evaluates large language models as recommender systems across multiple datasets, LLMs, traditional baselines, ranking and re-ranking tasks, and both utility and beyond-utility criteria. This repository does not claim to reproduce the full paper. Instead, it builds a clean evaluation wrapper around the original repository and the artifacts available in the current checkout.

The implemented work includes repository mapping, dataset auditing, configuration validation, cached LLM output normalization, cached baseline output normalization where available, tested metric utilities, readiness-aware experiment runners, plots from existing CSV files, a quality review, and reproducibility documentation. The main blocker is that the original README-linked preprocessed Beauty and Sports Dropbox artifacts are currently unavailable/deleted, so the canonical `sequential_data.txt`, `negative_samples.txt`, `datamaps.json`, and `meta.json` files could not be recovered from those links. Because those files provide ground truth, candidates, catalog metadata, and user histories, real HR@5, NDCG@5, Catalog Coverage@5, hallucination, MostPop, position-bias, and history-length metrics are not computed for the current Beauty and Sports configs. No missing metric values are fabricated.

The project therefore adds a separate Plan B reconstruction/extension track on MovieLens-1M. Plan B does not recover the missing Beauty/Sports artifacts and is not an exact reproduction of the original paper. It demonstrates that the project wrapper can run an end-to-end candidate-set recommendation evaluation when canonical data exist. MovieLens-1M is prepared with a deterministic leave-one-out protocol, and Random, MostPop, and ContentBM25 baselines are evaluated with real HR@5 and NDCG@5 metric rows.

## 1. Introduction

LLM-as-recommender systems are important because they combine recommendation, ranking, language generation, and interaction with natural-language prompts. Evaluating them only with top-k utility metrics can hide failures that are specific to generative systems, such as hallucinated items, sensitivity to prompt structure, and dependence on candidate ordering. These issues are directly relevant to Information Retrieval and Personalization because they affect ranking quality, robustness, transparency, and user trust.

Reproducibility is also central in recommender-system research. Reported metrics depend heavily on preprocessing, train/test splits, negative sampling, candidate pools, item metadata, and model-output parsing. If any of these artifacts are missing or ambiguous, reproducing a paper's numerical tables becomes unsafe. This project therefore uses a readiness-aware methodology: it implements the pipeline and metric tools, but computes real metrics only when the required inputs are present and validated.

The project attempts to reproduce the recoverable parts of the original study for Beauty and Sports: artifact inspection, cached LLM output loading, Beauty baseline output loading, and readiness diagnostics. It intentionally does not claim full quantitative reproduction, does not rerun LLM APIs, does not train traditional models, and does not convert missing data into fake ground truth. Separately, the MovieLens-1M reconstruction track demonstrates a complete evaluation workflow without weakening the Plan A conclusion.

## 2. Background And Original Paper

The original paper, *Beyond Utility: Evaluating LLM as Recommender*, studies LLMs used as recommendation models. It considers recommendation tasks such as ranking and re-ranking, compares LLM-based recommenders with traditional recommender baselines, and evaluates performance across utility and beyond-utility dimensions.

The paper scope is broader than this project. It includes four datasets, multiple LLMs, traditional baselines, saved prompt/output artifacts, and analyses beyond standard HR and NDCG. The beyond-utility dimensions are important because LLM recommenders may output invalid items, depend on candidate position, behave differently when user history length changes, and introduce generation-specific costs and failure modes.

The local original repository confirms this design. Its scripts include LLM prompt generation and API wrappers, output refinement, metric computation, and shell command lists for many experiment variants. The original metric script depends on preprocessed data files such as `sequential_data.txt`, `negative_samples.txt`, `datamaps.json`, and `meta.json`. Those files are not present in the current checkout, and the README-linked preprocessed archives are documented as unavailable/deleted.

## 3. Project Objectives

The objectives of this repository are practical and conservative:

- inspect and map the original repository without modifying it;
- identify which original artifacts are local, missing, or externally unavailable;
- create dataset configs that keep uncertain paths as `null`;
- normalize available Beauty and Sports cached LLM outputs;
- normalize available traditional baseline outputs where present;
- implement tested utility and beyond-utility metric utilities;
- create readiness-aware runners that produce diagnostics rather than fake metrics;
- document all blockers clearly;
- implement safe extensions for catalog coverage, candidate position bias, history length sensitivity, and cost/token analysis;
- implement a separate MovieLens-1M reconstruction track with real canonical data, simple baselines, and real HR@5/NDCG@5 metrics;
- prepare an exam-ready reproducibility package.

The project deliberately separates implementation readiness from quantitative reproduction. A metric implementation can be complete and tested on synthetic data while still being blocked on real Beauty/Sports artifacts.

## 4. Repository And Pipeline Design

The repository is organized around a project-local wrapper rather than direct modification of upstream code. Dataset-specific settings live in YAML configs under `configs/`. Source code lives under `src/`, tests under `tests/`, diagnostic outputs under `results/tables/`, real metric outputs under `results/metrics/`, normalized model outputs under `results/raw_outputs/`, plots under `results/plots/`, and documentation under `report/`. The upstream repository remains under `original_repo/`.

The configuration system in `src/utils/config.py` finds the project root, loads YAML safely, validates required sections, rejects hardcoded absolute paths, and resolves configured paths relative to the root. This keeps dataset-dependent choices out of Python code.

The dataset audit in `src/data/audit_datasets.py` inspects configured paths and searches the original repository for likely artifacts when paths are null. It writes audit CSVs under `results/tables/` and a Markdown summary under `report/dataset_audit.md`. Because the canonical dataset files are missing, dataset-level statistics such as user count, item count, interaction count, and catalog size remain unknown.

Schema helpers in `src/data/schemas.py` validate canonical dataframes for interactions, candidates, recommendations, and ground truth. The LLM parser in `src/evaluation/output_parser.py` converts noisy textual outputs into ranked recommendation records while preserving unmatched text and separating exact, normalized, fuzzy, and unmatched match types.

The baseline loader in `src/baselines/load_baseline_outputs.py` normalizes existing traditional baseline outputs when configured. `src/baselines/mostpop.py` implements a lightweight MostPop baseline over provided interactions only, but it does not generate Beauty/Sports recommendations while `interactions_path` is null.

The metric modules in `src/evaluation/` implement utility, hallucination, coverage, position-bias, and cost/token helpers. These modules are deterministic, testable, and independent from real dataset files. Experiment runners under `src/experiments/` orchestrate loading, reproduction diagnostics, coverage readiness, position-bias readiness, history-length readiness, the final safe pipeline, and plot generation.

The final Plan A pipeline in `src/experiments/run_pipeline.py` records whether each component is completed, ready, blocked, missing, or not configured. It writes dataset-specific final summaries under `results/tables/` and only writes final metric CSVs if an upstream step has produced real metric rows. With the current Beauty and Sports configs, no final metric CSV is written.

The Plan B pipeline in `src/experiments/run_ml1m_reconstruction.py` is intentionally separate. It checks whether canonical MovieLens-1M processed files exist, prepares them from local raw files only when needed, refreshes the Random, MostPop, and ContentBM25 baselines, rebuilds the comparison table from real metric CSVs, and generates ML-1M plots from real comparison rows only.

## 5. Available Artifacts

The local original repository contains cached LLM result archives for Beauty and Sports under `original_repo/EvaLLMasRecommender/llm_results/`. These archives include prompt and result JSON files. The project normalizes the configured GPT-3.5 ranking outputs for both datasets into `results/raw_outputs/`.

The local original repository also contains a Beauty traditional baseline archive under `original_repo/EvaLLMasRecommender/sota_results/beauty_his10.zip`. The project normalizes the configured Beauty BPRMF recommendation output into `results/raw_outputs/`. The Sports traditional baseline archive is not present locally and is reported as missing/not configured.

The key missing artifacts are the preprocessed Beauty and Sports data archives referenced by the original README. The project documentation records that both Dropbox links are unavailable/deleted. As a result, the files expected by original scripts cannot currently be recovered from those links:

- `sequential_data.txt`
- `negative_samples.txt`
- `datamaps.json`
- `meta.json`

These files are needed to construct canonical interactions, ground truth, candidates, catalog metadata, item-id/title mappings, and ASIN/item-id mappings. Without them, the available cached LLM and baseline recommendations cannot be evaluated safely against real targets.

## 6. Methods

The core method is readiness-aware reproduction. Instead of forcing every phase to produce metric rows, each runner checks whether the required inputs exist and are readable. If inputs are missing, the runner writes a diagnostic summary under `results/tables/` and exits successfully when the diagnostic output itself was produced correctly. Non-zero exits are reserved for invalid configs, unreadable required project files, or actual script errors.

The output policy is central to the methodology. Diagnostic and readiness files belong under `results/tables/`. Normalized cached outputs belong under `results/raw_outputs/`. Real computed metrics belong under `results/metrics/` only when valid underlying inputs are available. This prevents readiness artifacts from being mistaken for reproduced experimental results.

The project avoids fabricating missing information. It does not infer item IDs from free text without a validated catalog, does not treat missing ground truth as zero performance, does not use candidate sets as a full catalog unless explicitly labeled, and does not write empty metric CSVs as placeholders.

Tests use synthetic data to validate behavior independently of the missing original artifacts. The test suite covers config loading, schemas, output parsing, utility metrics, hallucination, coverage, position-bias helpers, history-length helpers, cost/token utilities, loaders, runners, pipeline behavior, and plot generation. The quality review records the full suite as passing at the time of that review.

## 7. Implemented Components

### 7.1 Data And Config Audit

The project includes safe YAML configs for Beauty and Sports. Canonical paths for interactions, catalog, candidates, and ground truth remain null because the corresponding artifacts are not available. The dataset audit documents which configured paths exist, which are missing, and which remain unknown. It also records that candidate lists are visible in cached LLM prompt/result artifacts, but those prompt-level candidates are not treated as complete canonical dataset files.

### 7.2 LLM Output Normalization

The LLM output loader reads configured cached JSON outputs from the original LLM ZIP archives without modifying `original_repo/`. It parses noisy recommendation text into ranked records, preserves raw output and provenance, and writes normalized recommendation files under `results/raw_outputs/`. Because no catalog is configured, the normalized LLM rows preserve recommendation text but do not claim validated item IDs.

### 7.3 Baseline Output Normalization

The baseline loader reads configured traditional/SOTA outputs when present. For Beauty, the configured BPRMF output in the local SOTA ZIP is normalized. For Sports, no traditional baseline output is configured or found locally, so a diagnostic summary is written instead. No traditional model is trained.

### 7.4 Utility Metrics

`src/evaluation/utility_metrics.py` implements HR@K, NDCG@K, and a combined evaluation helper. The implementation supports one or multiple relevant items per user, empty recommendations, duplicated recommendations, rank columns, and users with no ground truth. These functions are tested with synthetic examples. They are not run on the current Beauty/Sports artifacts because canonical ground truth is missing.

### 7.5 Hallucination Metrics

`src/evaluation/hallucination.py` implements hallucination summaries for recommendation outputs against candidate or catalog references. It distinguishes exact matches, normalized matches, fuzzy matches, and unmatched outputs. The metric never upgrades fuzzy matches into exact utility evidence. Real hallucination evaluation remains blocked because neither `catalog_path` nor `candidates_path` is configured for Beauty or Sports.

### 7.6 Catalog Coverage@K

`src/evaluation/coverage.py` implements Catalog Coverage@K, defined as the fraction of unique catalog items that appear at least once in the top-K recommendation lists across evaluated users. It supports item IDs or strings, multiple users, top-K truncation, duplicated recommendation handling, empty recommendations, and a documented empty-catalog behavior. The readiness-aware runner computes coverage only when a real catalog is configured. With the current configs, Coverage@5 is blocked because `catalog_path` is null.

### 7.7 Candidate Position Bias

`src/prompts/candidate_ordering.py` implements deterministic candidate ordering variants: original, random, reversed, positive at start, and positive at end. `src/evaluation/position_bias.py` computes selected-candidate position distributions and summaries from supplied candidates and recommendation outputs. The runner writes diagnostics when candidates or matched outputs are unavailable. Real position-bias analysis is blocked because `candidates_path` is null.

### 7.8 History Length Sensitivity

`src/prompts/history_length.py` implements deterministic truncation of user histories to lengths 0, 1, 2, 5, 10, and full, preserving chronological order when ordering fields exist. The runner can prepare prompt/input variants when history data is configured, but it does not call APIs by default. With current configs, history-length analysis is blocked because neither history data nor interactions are configured.

### 7.9 Cost And Token Metrics

`src/evaluation/cost_metrics.py` implements safe utilities for provided token counts, latency records, and explicit pricing assumptions. It can compute total tokens, aggregate latency and token summaries, and estimate costs when manually configured prices are present. `configs/cost.yaml` keeps prices null by default. No API calls or external token-estimation services are used.

### 7.10 Plots And Summaries

`src/experiments/make_plots.py` reads existing CSV files only. It skips HR@5, NDCG@5, Coverage@5, hallucination, position-bias, and history-length plots when those metric rows are unavailable. With the current outputs, it generates readiness/status-count plots for Beauty and Sports and writes plot summaries under `results/tables/`.

## 8. Results

The current project outputs are diagnostic and normalization artifacts, not reproduced metric tables. The following were actually produced:

- normalized Beauty and Sports LLM recommendation outputs under `results/raw_outputs/`;
- normalized Beauty BPRMF baseline recommendation output under `results/raw_outputs/`;
- dataset audit summaries, LLM parsing summaries, baseline summaries, reproduction readiness summaries, extension summaries, final pipeline summaries, plot summaries, and original artifact inventory under `results/tables/`;
- readiness/status plots under `results/plots/`;
- documentation reports under `report/`.

For Plan A Beauty/Sports, no real HR@5, NDCG@5, Catalog Coverage@5, hallucination, MostPop, candidate position-bias, history-length, or cost metric values are claimed. `results/metrics/` contains no real metric CSV for the current Beauty/Sports configs. This is intentional: the required canonical ground truth, catalog, candidates, and interactions are missing.

The final summaries show the expected readiness state. Beauty has available normalized LLM output and a normalized traditional baseline output, but MostPop, utility metrics, hallucination, coverage, position bias, and history-length analysis remain blocked or not configured. Sports has available normalized LLM output, no configured traditional baseline output, and the same missing canonical data blockers.

## 8.1 Plan B MovieLens-1M Reconstruction Results

Plan B is a reconstruction/extension experiment, not an exact reproduction of the original Beauty/Sports experiments. The user manually provides MovieLens-1M raw files under `data/raw/ml1m/`; the project does not download them. The preparation script reads `ratings.dat` and `movies.dat`, keeps original user and item identifiers, sorts interactions chronologically per user, keeps users with at least two interactions, and uses the last interaction as the held-out test item. Earlier interactions become training/history rows. Candidate sets contain exactly one positive held-out item and deterministic negative samples from items not interacted with by that user.

The canonical Plan B files are:

- `data/processed/ml1m/interactions.csv`;
- `data/processed/ml1m/ground_truth.csv`;
- `data/processed/ml1m/candidates.csv`;
- `data/processed/ml1m/catalog.csv`.

The implemented Plan B baselines are:

- Random: deterministic pseudo-random scoring of each user's candidate set, controlled by the config seed;
- MostPop: non-personalized popularity from training/history interactions only;
- ContentBM25: a lightweight content scorer using movie titles, genres, and normalized catalog text, with user profiles built only from training/history items.

The full runner command is:

```powershell
python src/experiments/run_ml1m_reconstruction.py --config configs/ml1m.yaml
```

The real Plan B results are:

| Model | HR@5 | NDCG@5 |
|---|---:|---:|
| Random | 0.0514900662251655 | 0.0293715251496116 |
| MostPop | 0.320364238410596 | 0.2102806322093436 |
| ContentBM25 | 0.2508278145695364 | 0.1647567964031986 |

These values are read from generated metric and comparison CSVs under `results/metrics/` and `results/tables/`. They are not copied from the original paper and they are not Beauty/Sports reproduction metrics.

The ranking is interpretable. Random is a sanity-check lower bound because it uses no preference, popularity, or metadata signal. ContentBM25 performs substantially better than Random, showing that simple lexical overlap between a user's history and movie title/genre metadata contains useful signal. MostPop performs best among the three simple baselines. This is plausible on MovieLens-1M because globally popular movies are widely watched and can be strong candidates in a fixed candidate-set evaluation.

There are important limitations. The ML-1M evaluation is candidate-set ranking, not full-catalog ranking: every user is evaluated over a sampled candidate list containing one positive item and fixed negatives. Metrics can change under a different negative-sampling policy, rating filter, candidate count, or full-catalog evaluation. The protocol is deterministic and documented, but it is not equivalent to the original Beauty/Sports preprocessing in the paper.

Plan B outputs are separated by type:

- real metrics: `results/metrics/ml1m_random_metrics.csv`, `results/metrics/ml1m_mostpop_metrics.csv`, `results/metrics/ml1m_content_bm25_metrics.csv`;
- diagnostics and comparisons: `results/tables/ml1m_dataset_preparation_summary.csv`, `results/tables/ml1m_reconstruction_pipeline_summary.csv`, `results/tables/ml1m_baseline_comparison_summary.csv`;
- plots: `results/plots/ml1m_baseline_hr_at_5.png`, `results/plots/ml1m_baseline_ndcg_at_5.png`, `results/plots/ml1m_baseline_comparison.png`;
- generated recommendation outputs: `results/raw_outputs/ml1m_*_recommendations.csv`, treated as large generated artifacts and ignored by Git.

## 9. Reproducibility Blockers

The main blocker is external artifact availability. The original README-linked preprocessed Beauty and Sports Dropbox archives are unavailable/deleted according to the documented availability check. Therefore, the project cannot recover `sequential_data.txt`, `negative_samples.txt`, `datamaps.json`, or `meta.json` from those links.

The consequences are concrete:

- no canonical ground truth for HR@5 and NDCG@5;
- no canonical candidate sets for candidate-level hallucination or position-bias analysis;
- no canonical catalog metadata for Catalog Coverage@5 or catalog-level hallucination;
- no canonical interaction histories for MostPop or history-length prompt variants;
- no validated title-to-item-id mapping for LLM text outputs;
- no Sports traditional baseline output artifact in the current checkout.

The project also intentionally avoids other expensive or unsafe paths. No APIs were called, no datasets were downloaded, no baseline models were trained, and no upstream files under `original_repo/` were modified.

## 10. Discussion

This is a partial reproduction, but it is still a useful reproducibility engineering result. The project shows exactly where the original paper can be inspected locally, which artifacts can be normalized, and which missing files prevent quantitative evaluation. That is preferable to reporting fabricated HR@5, NDCG@5, hallucination, or coverage values from incomplete inputs.

The project also exposes a real reproducibility issue: saved model outputs alone are not enough to reproduce recommender metrics. The target item, candidate pool, catalog, item mappings, and preprocessing decisions are part of the experimental evidence. When those artifacts disappear from external links, the numerical tables become difficult or impossible to verify without author assistance or a careful raw-data reconstruction.

Plan B adds complementary experimental evidence: once canonical data are available, the wrapper can produce real recommendation outputs, real metrics, comparisons, and plots. This is not evidence that the original Beauty/Sports numbers are reproduced. It is evidence that the reconstruction pipeline is operational and that the separation between readiness diagnostics and metric computation works in practice.

For an oral exam, the defensible claim is that the repository implements a safe, tested, reproducibility-oriented wrapper and demonstrates why full Beauty/Sports reproduction is blocked, then uses MovieLens-1M to show an end-to-end reconstruction. It connects directly to recommender-system evaluation topics: top-k ranking, candidate-set evaluation, negative sampling, catalog coverage, hallucination, prompt sensitivity, history truncation, baseline comparison, and provenance of evaluation data.

## 10.1 How To Present This At The Oral Exam

The key distinction is between reproducibility audit and reconstruction. Plan A asks whether the original Beauty/Sports experiments can be numerically reproduced from the available official repository artifacts. The answer is no for the current checkout, because the preprocessed data files that define targets, candidates, metadata, and histories are unavailable. Reporting `metric_rows=0` for Beauty/Sports is therefore a scientific result about artifact readiness, not a failure to run code.

Plan B asks a different question: can the project wrapper support a complete recommendation evaluation on a stable, reconstructible public dataset? MovieLens-1M answers that question positively. The main experimental evidence is the real Random, MostPop, and ContentBM25 HR@5/NDCG@5 table plus the generated plots. These results show the expected ordering of simple baselines and demonstrate that the metric pipeline works when canonical files exist.

This is scientifically honest because the project never converts missing Beauty/Sports data into fake labels, never treats diagnostics as metrics, and clearly labels ML-1M results as reconstruction metrics. It also links to course topics: leave-one-out splitting, negative sampling, top-N ranking, popularity baselines, content-based recommendation, candidate-set evaluation, metric provenance, and the difference between utility and beyond-utility evaluation.

## 11. Limitations

The project does not perform a full end-to-end reproduction of the paper. It does not compute real utility metrics for Beauty or Sports, does not rerun LLM APIs, does not retrain traditional recommenders, and does not reconstruct the missing Amazon preprocessing pipeline from raw data. MovieLens-1M Plan B is a candidate-set reconstruction, not full-catalog ranking and not original-paper equivalence.

The cached LLM outputs are normalized mostly as text because a validated catalog mapping is unavailable. The Beauty baseline output is normalized, but cannot be evaluated without canonical ground truth. Sports traditional baseline outputs are not available locally. The policy for large generated normalized outputs, especially the large Beauty baseline normalization file, should be decided later, for example by using external storage or Git LFS if necessary.

## 12. Future Work

A fuller Beauty/Sports reproduction should start by obtaining the deleted preprocessed artifacts from the authors or reconstructing equivalent files from valid raw Amazon data in a separate documented phase. The next data phase should generate canonical files under `data/processed/{dataset}/` for interactions, ground truth, candidates, and catalog metadata, then update `configs/beauty.yaml` and `configs/sports.yaml` to point to those files.

After those inputs are configured, the project can rerun real HR@5, NDCG@5, hallucination, Catalog Coverage@5, MostPop, position-bias, history-length, and plot commands. Future extensions could add additional datasets, compare more LLMs and baselines, run selected API calls only with explicit `--allow-api-calls`, and package large artifacts through an appropriate storage policy.

## 13. Conclusion

This repository is a controlled reproducibility engineering study of *Beyond Utility: Evaluating LLM as Recommender*. It does not reproduce the full paper numerically, but it implements the infrastructure needed for honest reproduction once missing artifacts are available. The main contribution is a safe, tested, readiness-aware Plan A pipeline that normalizes available artifacts, reports blockers precisely, avoids fake results, and separates diagnostic evidence from real metrics. The separate Plan B track demonstrates the same evaluation infrastructure on MovieLens-1M with real HR@5 and NDCG@5 metrics.

The project is suitable for an exam because it demonstrates technical implementation, artifact provenance, critical evaluation of reproducibility barriers, and safe beyond-utility extensions without overstating scientific results.

## 14. Plan C Final Consolidation

Plan C extends the project from a partial readiness audit and ML-1M extension
into a full-paper replication/reconstruction campaign. The final conclusion is
still not an exact paper reproduction: the official Beauty/Sports processed
artifacts remain unavailable. However, Plan C now provides a complete
reconstruction evidence chain:

- artifact matrix, manifest, source ledger, and validation protocol;
- raw dataset schema validation for Amazon Beauty, Amazon Sports, ML-1M, and
  LastFM;
- reconstructed canonical datasets for Amazon Beauty, Amazon Sports, and
  ML-1M;
- LastFM marked `extension_only`;
- corrected negative sampling with `interaction_items` as the default candidate
  universe;
- deterministic Random, MostPop, and ContentBM25 baseline metrics where valid
  ground truth and candidates exist;
- final claims and limitations in
  `results/tables/plan_c_claims_and_limitations.csv`.

The exam-ready Plan C summary is `report/plan_c_final_report.md`. The Italian
oral guide is `report/oral_exam_plan_c_guide.md`.

## Appendix A: Command Summary

Install and test:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest tests/
```

Run the main readiness pipeline:

```powershell
python src/experiments/exp00_smoke_test.py
python src/data/audit_datasets.py --config configs/beauty.yaml
python src/data/audit_datasets.py --config configs/sports.yaml
python src/experiments/load_llm_outputs.py --config configs/beauty.yaml
python src/experiments/load_llm_outputs.py --config configs/sports.yaml
python src/baselines/load_baseline_outputs.py --config configs/beauty.yaml
python src/baselines/load_baseline_outputs.py --config configs/sports.yaml
python src/experiments/run_reproduction.py --config configs/beauty.yaml
python src/experiments/run_reproduction.py --config configs/sports.yaml
python src/experiments/run_extension_coverage.py --config configs/beauty.yaml
python src/experiments/run_extension_coverage.py --config configs/sports.yaml
python src/experiments/run_position_bias.py --config configs/beauty.yaml
python src/experiments/run_position_bias.py --config configs/sports.yaml
python src/experiments/run_history_length.py --config configs/beauty.yaml
python src/experiments/run_history_length.py --config configs/sports.yaml
python src/experiments/run_pipeline.py --config configs/beauty.yaml
python src/experiments/run_pipeline.py --config configs/sports.yaml
python src/experiments/make_plots.py --config configs/beauty.yaml
python src/experiments/make_plots.py --config configs/sports.yaml
```

Run MostPop only after canonical interactions are configured:

```powershell
python src/baselines/mostpop.py --config configs/beauty.yaml
python src/baselines/mostpop.py --config configs/sports.yaml
```

Run the Plan B ML-1M reconstruction pipeline:

```powershell
python src/data/prepare_ml1m.py --config configs/ml1m.yaml
python src/experiments/run_ml1m_reconstruction.py --config configs/ml1m.yaml
```

## Appendix B: Evidence Files

The main evidence files for this report are:

- `README.md`
- `PROJECT_SPEC.md`
- `AGENTS.md`
- `CODEX_PROMPTS.md`
- `report/codebase_map.md`
- `report/dataset_audit.md`
- `report/original_artifact_map.md`
- `report/quality_review.md`
- `report/results_analysis.md`
- `report/reproducibility_checklist.md`
- `report/plan_b_reconstruction_plan.md`
- `results/tables/original_artifact_inventory.csv`
- `results/tables/*_summary.csv`
- `results/tables/*_dataset_audit.csv`
- `results/tables/ml1m_baseline_comparison_summary.csv`
- `results/metrics/ml1m_*_metrics.csv`
- `results/plots/*_readiness_status_counts.png`
- `results/plots/ml1m_baseline_*.png`
- `tests/`
