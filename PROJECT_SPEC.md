# PROJECT_SPEC.md

# Reproducibility and Extension of *Beyond Utility: Evaluating LLM as Recommender*

## 1. Project identity

**Working title:** Reproducibility and Extension of *Beyond Utility: Evaluating LLM as Recommender*  
**Alternative title:** Cost- and Coverage-Aware Reproducibility Study of LLM-based Recommender Evaluation  
**Course:** Information Retrieval and Personalization  
**Project type:** Reproducibility project with a small methodological extension  
**Main area:** Recommender Systems, LLM-as-Recommender, evaluation beyond utility  
**Primary implementation environment:** Visual Studio Code + Python + Git + ChatGPT Codex

## Current frozen state

This specification is historical planning context plus current boundaries. The repository is now frozen as a comprehensive full research repository: F6M closes the local adapted reproduction as `LOCAL_ADAPTED_REPRODUCTION_COMPLETE_NOT_PAPER_EXACT`, and G1-G4 close the adapted local extension through primary ranking, history-length sensitivity, candidate-position sensitivity, and F1BZQ reranking. These extension phases include Food.com and Gemma2 as local adapted evidence; they are not paper-exact replication.

The project has three clearly separated tracks. Plan A is a readiness-aware reproducibility audit of *Beyond Utility: Evaluating LLM as Recommender* on the original Beauty and Sports artifacts available in the upstream repository checkout. Plan B is a separate MovieLens-1M reconstruction/extension track that demonstrates an end-to-end recommendation evaluation pipeline on a stable public dataset. Plan C is a maximum-fidelity replication/reconstruction campaign that audits the full paper surface, validates public raw datasets, reconstructs canonical datasets, and evaluates deterministic baselines on reconstructed data. The goal is not to claim exact reproduction when evidence does not support it; Plan B and Plan C reconstructed metrics are not exact original Beauty/Sports paper metrics.

## 2. Motivation

The original paper argues that evaluating LLMs as recommenders only through utility/accuracy is insufficient. It proposes a multidimensional framework including conventional dimensions such as utility and novelty, plus LLM-related dimensions such as history length sensitivity, candidate position bias, generation-involved performance and hallucination.

This project is suitable for an Information Retrieval and Personalization exam because it connects ranking/re-ranking, top-N recommendation, recommender-system evaluation, traditional baselines versus LLM-based recommenders, beyond-accuracy metrics, robustness, bias, hallucination and reproducibility under realistic computational constraints.

## 3. Scope decision

### 3.1 Final Plan A scope

Plan A is closed as a readiness-aware partial reproducibility study.

- Task: inspect and normalize recoverable ranking artifacts for Beauty and Sports.
- Available artifacts: local Beauty/Sports cached LLM outputs and a local Beauty traditional baseline archive.
- Missing artifacts: the original preprocessed Beauty/Sports files `sequential_data.txt`, `negative_samples.txt`, `datamaps.json`, and `meta.json`.
- Outcome: audit, normalization, tested metrics, readiness summaries, and plots. Real Beauty/Sports HR@5, NDCG@5, hallucination, Catalog Coverage@5, MostPop, position-bias, and history-length metrics are blocked and not fabricated.
- Correct final status: Beauty/Sports final pipelines write readiness summaries with `metric_rows=0`.

### 3.2 Final Plan B scope

Plan B is a reconstruction/extension experiment on MovieLens-1M.

- Dataset: MovieLens-1M, manually provided under `data/raw/ml1m/`.
- Canonical files: `interactions.csv`, `ground_truth.csv`, `candidates.csv`, and `catalog.csv` under `data/processed/ml1m/`.
- Protocol: chronological leave-one-out split for eligible users, one positive held-out item per user, deterministic negative sampling, and candidate-set ranking.
- Baselines: deterministic Random, training-only MostPop, and ContentBM25 over movie title/genre text.
- Metrics: HR@5 and NDCG@5 computed from real ML-1M canonical files and recommendation outputs.
- Plots: ML-1M baseline comparison plots generated only from real comparison rows.

### 3.3 Final Plan C scope

Plan C is consolidated as a full-paper replication/reconstruction campaign.

- Exact original Beauty/Sports reproduction remains blocked by unavailable
  official processed artifacts.
- Public raw Amazon Beauty, Amazon Sports, ML-1M, and LastFM files were staged
  and validated.
- Amazon Beauty, Amazon Sports, and ML-1M were reconstructed into canonical
  Plan C datasets under `data/reconstructed/plan_c/`.
- LastFM remains `extension_only` because no valid chronological
  ground-truth/candidate protocol was produced.
- Negative sampling defaults to `interaction_items`, excluding each user's full
  history; catalog-wide negative sampling is experimental only.
- Plan C evaluates deterministic Random, MostPop, and ContentBM25 baselines
  where valid inputs exist.
- Final synthesis is stored in `report/plan_c_final_report.md` and
  `results/tables/plan_c_final_results_overview.csv`.

### 3.4 Publication-oriented version

This is optional and must not block the exam project.

- Add cost-per-query, latency and token usage.
- Add history length sensitivity: 0, 1, 2, 5, 10, full.
- Add prompt ablation against position bias.
- Add exact versus fuzzy hallucination matching.
- Add statistical robustness: confidence intervals or paired tests where meaningful.
- Frame the project as a short paper about constrained reproducibility and evaluation trade-offs.

## 4. Non-goals

The project does **not** aim to reproduce all seven LLMs, all six traditional models, all four datasets, or every table from the original paper. It does not fine-tune LLMs, run expensive API experiments by default, modify the original repository unless necessary, or claim numerical reproduction without logged scripts and CSV evidence.

## 5. Research questions

### RQ1 — Reproducibility

To what extent can a representative subset of the original paper’s results be reproduced under a constrained experimental setup?

### RQ2 — LLM vs traditional recommenders

How do LLM-based recommenders compare with traditional recommendation models when evaluated with both utility and beyond-utility metrics?

### RQ3 — Robustness and bias

How sensitive are LLM-based recommenders to candidate ordering and user history length?

### RQ4 — Extension

Does adding Catalog Coverage@K change the interpretation of recommendation quality compared with HR@5 and NDCG@5 alone?

### RQ5 — Optional publication-oriented question

How does the evaluation of LLM-based recommenders change when utility, hallucination, coverage and inference cost are considered together?

## 6. Datasets

Primary original datasets for Plan A: Beauty and Sports. Reconstruction dataset for Plan B: MovieLens-1M. Optional future datasets: LastFM or another public dataset with equivalent schema.

For every dataset used, the project must report number of users, number of items, number of interactions, split availability, candidate-list availability, ground-truth format, item metadata availability, average user history length, catalog size, sparsity if interaction matrix can be reconstructed, and any preprocessing difference from the original repository.

Dataset audit outputs:

- `results/tables/<dataset>_dataset_audit.csv`
- `report/dataset_audit.md`

## 7. Models and outputs

Preferred LLM-output order: load LLM outputs already available in the original repository; parse and evaluate saved output files; only if needed, run a small number of API or local LLM calls using explicit `--allow-api-calls`.

Minimum traditional baseline: MostPop implemented locally in `src/baselines/mostpop.py`.

Preferred additional baselines: baseline outputs from the original repository, BPRMF, SASRec, or LightGCN if available. The project should prioritize evaluating existing outputs over re-training heavy models.

## 8. Unified data schema

### Interaction record
Required fields: `user_id`, `item_id`, `split`. Optional fields: `timestamp`, `rating`.

### Candidate record
Required fields: `user_id`, `candidate_item_id`, `candidate_position`. Optional fields: `is_ground_truth`, `candidate_item_title`.

### Recommendation record
Required fields: `user_id`, `model_name`, `dataset`, `recommended_item_id` or `recommended_item_text`, `rank`, `run_id`. Optional fields: `raw_output`, `match_type`, `source_file`, `prompt_variant`, `candidate_ordering`, `history_length`.

### Ground-truth record
Required fields: `user_id`, `ground_truth_item_id`. Optional fields: `dataset`, `split`.

## 9. Metrics

### HR@K
Hit Rate at K is 1 for a user if at least one relevant ground-truth item appears in the top-K recommendations, otherwise 0. The final value is the mean over users.

### NDCG@K
NDCG@K rewards relevant items appearing higher in the ranked list. For one relevant item, if the relevant item is at rank `r <= K`, then `NDCG@K = 1 / log2(r + 1)`, otherwise 0. For multiple relevant items, use DCG divided by ideal DCG.

### Hallucination rate
For LLM outputs, hallucination rate is the fraction of generated recommendations that cannot be matched to any valid candidate item or catalog item, depending on the selected protocol. Support candidate-level and catalog-level hallucination. Report exact, normalized and fuzzy matching separately.

### Catalog Coverage@K
Catalog Coverage@K measures the fraction of the catalog that appears at least once in top-K recommendations across users: `Coverage@K = |unique recommended items in top-K| / |catalog items|`. This is the mandatory extension metric.

### Candidate position bias
Candidate position bias is evaluated by comparing utility metrics across candidate-ordering variants and, when possible, by analyzing the distribution of selected candidate input positions.

### History length sensitivity
History length sensitivity is evaluated by changing how many previous user interactions are included in the prompt: 0, 1, 2, 5, 10, full. Optional unless time allows.

### Cost metrics
Optional but recommended for publication orientation: prompt tokens, completion tokens, total tokens, latency seconds, estimated cost per query, estimated cost per 1,000 users.

## 10. Experiment matrix

### Plan A readiness matrix

| Dataset | Task | Models | Metrics |
|---|---|---|---|
| Beauty | Ranking | Cached LLM output + cached BPRMF output where available | Readiness diagnostics; HR@5/NDCG@5/hallucination/Coverage@5 blocked by missing canonical inputs |
| Sports | Ranking | Cached LLM output | Readiness diagnostics; HR@5/NDCG@5/hallucination/Coverage@5 blocked by missing canonical inputs |

### Plan B reconstruction matrix

| Dataset | Task | Models | Extra analysis |
|---|---|---|---|
| ML-1M | Candidate-set ranking | Random, MostPop, ContentBM25 | HR@5, NDCG@5, baseline comparison plots |

### Optional publication matrix

| Dataset | Task | Models | Extra analysis |
|---|---|---|---|
| ML-1M or LastFM | Ranking/Re-ranking | Selected subset | Coverage + cost + history sensitivity |

## 11. Required outputs

Code: `src/data/audit_datasets.py`, `src/data/load_data.py`, `src/data/schemas.py`, `src/data/prepare_ml1m.py`, `src/baselines/mostpop.py`, `src/baselines/random_baseline.py`, `src/baselines/ml1m_mostpop_baseline.py`, `src/baselines/content_bm25_baseline.py`, `src/evaluation/utility_metrics.py`, `src/evaluation/output_parser.py`, `src/evaluation/hallucination.py`, `src/evaluation/coverage.py`, `src/evaluation/position_bias.py`, `src/evaluation/cost_metrics.py`, `src/experiments/exp00_smoke_test.py`, `src/experiments/load_llm_outputs.py`, `src/experiments/run_reproduction.py`, `src/experiments/run_pipeline.py`, `src/experiments/make_plots.py`, `src/experiments/run_ml1m_reconstruction.py`.

Tests: `tests/test_utility_metrics.py`, `tests/test_output_parser.py`, `tests/test_hallucination.py`, `tests/test_coverage.py`.

Results: `results/metrics/*_metrics.csv`, `results/tables/*_summary.csv`, `results/plots/*.png`, `results/raw_outputs/`.

Reports: `report/codebase_map.md`, `report/notes.md`, `report/dataset_audit.md`, `report/results_analysis.md`, `report/archive/phase_reports/technical_report.md`, `report/archive/phase_reports/reproducibility_checklist.md`, `report/archive/phase_reports/plan_b_reconstruction_plan.md`.

Plan B final real results:

| Dataset | Model | HR@5 | NDCG@5 |
|---|---|---:|---:|
| ML-1M | Random | 0.0514900662251655 | 0.0293715251496116 |
| ML-1M | MostPop | 0.320364238410596 | 0.2102806322093436 |
| ML-1M | ContentBM25 | 0.2508278145695364 | 0.1647567964031986 |

These are Plan B reconstruction metrics only. They do not reproduce the original Beauty/Sports numerical results.

## 12. Reproducibility rules

All scripts must run from the project root. All paths must be relative or config-driven. All random processes must use fixed seeds. All experiment scripts must save the used config. No reported number may be typed manually without a source CSV. Every final table must be generated from saved results. API keys must never be committed. API calls must require `--allow-api-calls`. The original repository must not be modified unless explicitly requested. Any deviation from the original paper must be documented.

## 13. Success criteria by phase

A phase is complete only when the relevant script runs from the root, the output file is created in the expected folder, tests pass if code was added, `report/notes.md` is updated, and a Git commit is created with a meaningful message.

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Original repository is hard to run | Use it as data/output source and build a clean evaluation wrapper in `src/` |
| API costs become excessive | Default to saved outputs; require explicit API flag |
| Too many metrics/models | Complete minimum version first |
| Dataset paths are unclear | Run dataset audit before metrics |
| LLM output format is messy | Build robust parser with tests |
| Fuzzy matching inflates quality | Report exact and fuzzy separately |
| Results differ from original paper | Treat differences as reproducibility findings, not failures |
| Project becomes too broad | Freeze scope after minimum + one strong extension |

## 15. Final deliverables

Clean code in `src/`, tests in `tests/`, configs in `configs/`, results in `results/`, plots in `results/plots/`, final README, `PROJECT_SPEC.md`, `AGENTS.md`, `CODEX_PROMPTS.md`, `reproducibility_checklist.md`, `report/report.md`, `report/results_analysis.md`, `report/slides_outline.md`, a reproducible pipeline and a clear oral defense story.
