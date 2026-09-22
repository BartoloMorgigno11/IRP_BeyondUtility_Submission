# Traditional Baseline Consolidation and Alignment


## 1. Scope and provenance

- Phase: Phase 3A - Deterministic Non-Neural Baseline Consolidation.
- Branch: `plan-d-extensions`.
- Commit: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests`.
- Scope: deterministic non-neural baselines on the Phase 2C Plan E2 full and matched cohorts only.
- Safety: no APIs, web access, downloads, LLM inference, Ollama invocation, neural training or original_repo edits were required by this builder.
- Authoritative generated files are under `results/tables/` and this report.
- Phase 2 input hashes used by the build:
  - `results/tables/candidate_item_title_map.csv`: `5ff8a0b663e83d15830a6d86c8a2cfdbd9a3c7ce256f9ff496a7b5967b273e14`
  - `results/tables/dataset_statistics.csv`: `3b60324836d67ebfbc827b8962ac0eaa68f3a433259d3987fcdc2909643cd560`
  - `results/tables/experiment_cohort_membership.csv`: `d92be28d56d20897b25bd0e0d8387194930253aec62b957fe00aa230f7df0139`
  - `results/tables/experiment_cohort_summary.csv`: `40be65003a46815891ec35ee36b2a42db945e46759ee642f5e7ac4b5561f4b2d`
  - `results/tables/llm_title_exclusions.csv`: `eea097da89c98e96c8eea90bccf46a9180abcee2858e5de0a021025f90247081`
  - `results/tables/positive_positions.csv`: `0c93dd55d5c96ed99fb7f1afc8f5319e90f3092ecb6a48f2e9d46854b38cd6d2`
  - `results/tables/ranking_candidate_sets.csv`: `640ad9c00b7d7c825ad10746a0e720c161d1ba9d75cc791a019df7fe14c7a955`

## 2. Executive summary

- Completed execution units: 18; blocked manifest execution units: 6.
- Ranking rows emitted: 18000 (three completed baselines over 6,000 Plan E2 rows).
- Metric rows emitted: 36 (three baselines x six dataset/task candidate sets x two cohorts).
- Random and MostPop are paper-like reconstructed baselines because they run on reconstructed Plan E2 artifacts rather than verified exact paper artifacts.
- The executed BM25 family member is `bm25_like_e3b`; exact official BM25 remains blocked.
- Neural traditional baselines are intentionally deferred to Phase 3B and were not trained or executed.

## 3. Authoritative Phase 2 cohorts

- `ranking_candidate_sets.csv` contains 76,121 canonical candidate rows.
- The Plan E2 full cohort contributes 6,000 rows, exactly 1,000 per dataset/task candidate set.
- The matched comparison cohort contributes 5,814 rows and is identical to the strict-title LLM cohort.
- All Plan E2 rows used here contain 20 candidates and exactly one positive item.
- Cohort filtering is inherited from `results/tables/experiment_cohort_membership.csv`; this phase does not modify candidate lists or cohort flags.

## 4. Existing baseline inventory

- `src/baselines/random_baseline.py`: deterministic historical random baseline, useful pattern but not the authoritative Plan E2 output layer.
- `src/baselines/mostpop.py`: generic MostPop implementation over train interactions, not candidate-set specific by itself.
- `src/baselines/ml1m_mostpop_baseline.py`: Plan B ML-1M candidate-set MostPop evidence.
- `src/baselines/content_bm25_baseline.py`: content BM25-style historical implementation for older artifacts.
- `src/analysis/plan_e_baseline_alignment.py`: Plan E3A no-training Random/MostPop alignment evidence.
- `src/analysis/plan_e_bm25_scoring.py`: Plan E3B local BM25-like scoring evidence.
- `original_repo/EvaLLMasRecommender/eva/codes/random.sh` and `llm_metrics.py`: read-only official-reference evidence that names traditional baselines, but not sufficient to prove local exact BM25 alignment.

## 5. Historical Plan B baseline evidence

- Plan B-oriented implementations exist in `src/baselines/random_baseline.py`, `src/baselines/ml1m_mostpop_baseline.py`, and `src/baselines/content_bm25_baseline.py`.
- Those files are useful implementation evidence but are not authoritative Phase 3A outputs because they target older canonical/ML-1M artifacts rather than the Phase 2C Plan E2 cohort.

## 6. Historical Plan C baseline evidence

- Phase 3A uses reconstructed Plan C train interactions for MostPop counts.
- The training sources are `data/reconstructed/plan_c/amazon_beauty/interactions.csv`, `data/reconstructed/plan_c/amazon_sports/interactions.csv`, and `data/reconstructed/plan_c/ml1m/interactions.csv`.
- Candidate membership and positives still come only from the Phase 2C Plan E2 manifests.

## 7. Plan E3A alignment evidence

- Plan E3A already audited Random/MostPop-style no-training baselines in `src/analysis/plan_e_baseline_alignment.py`.
- Prior comparable full-cohort metrics are in `results/tables/plan_e_baseline_alignment_no_training_metrics.csv`; Phase 3A regenerates authoritative rankings plus full/matched metrics in the required schemas.

## 8. Plan E3B BM25-like evidence

- Plan E3B produced local BM25-like candidate scores in `results/tables/plan_e_bm25_scores.csv`.
- The stored variant is `local_bm25_like_k1_1.2_b_0.75` and the exactness class is reconstructed/BM25-like rather than exact official BM25.
- Prior BM25-like metrics are in `results/tables/plan_e_bm25_metrics.csv`; Phase 3A adapts that score table into canonical rankings and cohort-specific metrics.

## 9. Random baseline definition

- Canonical ID: `random_seeded`; global seed: `20260710`.
- Score derivation: SHA256 over global seed, dataset, task type, candidate-set ID, user ID and item ID.
- Positive item and relevance labels are excluded from ordering.

## 10. MostPop baseline definition

- Canonical ID: `mostpop_train_only`.
- Score definition: raw train-split interaction frequency per item.
- Validation/test interactions, held-out positives, and candidate occurrence frequencies are not used.
- Candidate items absent from the train counts receive score 0.

## 11. BM25 variant determination

- Executed canonical ID: `bm25_like_e3b`.
- It is an adapted BM25-like local variant, not exact official BM25.
- `bm25_exact_official` is represented as blocked in the manifest and blocked table because local evidence does not demonstrate exact paper equivalence.

## 12. Training-data provenance and leakage prevention

- MostPop uses only rows with `split=train` from the reconstructed Plan C interaction CSV for the matching dataset.
- Random has no training data.
- BM25-like uses prior Plan E3B scores whose query/content provenance is documented in Plan E3B artifacts; this phase loads rather than rewrites those scores.

## 13. Tie-breaking policies

- All completed baselines use: descending score; lexical item_id as string; original candidate position.
- Item IDs are compared as strings, preserving identifier semantics.
- Original candidate position is only the final fallback, and the positive label is never used for tie-breaking.

## 14. Full-cohort evaluation

| baseline | dataset | task | rows | HR@5 | NDCG@5 | mean rank | median rank |
|---|---|---|---:|---:|---:|---:|---:|
| `random_seeded` | Amazon Beauty | ranking | 1000 | 0.256 | 0.157981001577 | 9.468 | 10 |
| `random_seeded` | Amazon Beauty | reranking | 1000 | 0.246 | 0.142683522674 | 9.529 | 10 |
| `random_seeded` | Amazon Sports | ranking | 1000 | 0.239 | 0.140996381563 | 9.646 | 10 |
| `random_seeded` | Amazon Sports | reranking | 1000 | 0.222 | 0.129446675633 | 9.832 | 10 |
| `random_seeded` | ML-1M | ranking | 1000 | 0.266 | 0.154581706889 | 9.277 | 9 |
| `random_seeded` | ML-1M | reranking | 1000 | 0.241 | 0.143519403574 | 9.648 | 10 |
| `mostpop_train_only` | Amazon Beauty | ranking | 1000 | 0.48 | 0.325883987566 | 7.068 | 5 |
| `mostpop_train_only` | Amazon Beauty | reranking | 1000 | 0.48 | 0.325883987566 | 7.068 | 5 |
| `mostpop_train_only` | Amazon Sports | ranking | 1000 | 0.479 | 0.346300626572 | 7.01 | 5 |
| `mostpop_train_only` | Amazon Sports | reranking | 1000 | 0.479 | 0.346300626572 | 7.01 | 5 |
| `mostpop_train_only` | ML-1M | ranking | 1000 | 0.671 | 0.475901770019 | 3.717 | 2 |
| `mostpop_train_only` | ML-1M | reranking | 1000 | 0.671 | 0.475901770019 | 3.717 | 2 |
| `bm25_like_e3b` | Amazon Beauty | ranking | 1000 | 0.557 | 0.417611574729 | 5.541 | 3 |
| `bm25_like_e3b` | Amazon Beauty | reranking | 1000 | 0.557 | 0.417611574729 | 5.541 | 3 |
| `bm25_like_e3b` | Amazon Sports | ranking | 1000 | 0.515 | 0.373698112433 | 6.156 | 4 |
| `bm25_like_e3b` | Amazon Sports | reranking | 1000 | 0.515 | 0.373698112433 | 6.156 | 4 |
| `bm25_like_e3b` | ML-1M | ranking | 1000 | 0.522 | 0.364931424919 | 5.815 | 4 |
| `bm25_like_e3b` | ML-1M | reranking | 1000 | 0.522 | 0.364931424919 | 5.815 | 4 |

## 15. Matched-cohort evaluation

| baseline | dataset | task | rows | HR@5 | NDCG@5 | mean rank | median rank |
|---|---|---|---:|---:|---:|---:|---:|
| `random_seeded` | Amazon Beauty | ranking | 994 | 0.255533199195 | 0.157796852941 | 9.45875251509 | 10 |
| `random_seeded` | Amazon Beauty | reranking | 994 | 0.245472837022 | 0.14214956727 | 9.54828973843 | 10 |
| `random_seeded` | Amazon Sports | ranking | 913 | 0.237677984666 | 0.140262023798 | 9.63526834611 | 10 |
| `random_seeded` | Amazon Sports | reranking | 913 | 0.221248630887 | 0.12958358171 | 9.85213581599 | 10 |
| `random_seeded` | ML-1M | ranking | 1000 | 0.266 | 0.154581706889 | 9.277 | 9 |
| `random_seeded` | ML-1M | reranking | 1000 | 0.241 | 0.143519403574 | 9.648 | 10 |
| `mostpop_train_only` | Amazon Beauty | ranking | 994 | 0.478873239437 | 0.325777043515 | 7.08249496982 | 5 |
| `mostpop_train_only` | Amazon Beauty | reranking | 994 | 0.478873239437 | 0.325777043515 | 7.08249496982 | 5 |
| `mostpop_train_only` | Amazon Sports | ranking | 913 | 0.480832420591 | 0.344928430771 | 7.00547645126 | 5 |
| `mostpop_train_only` | Amazon Sports | reranking | 913 | 0.480832420591 | 0.344928430771 | 7.00547645126 | 5 |
| `mostpop_train_only` | ML-1M | ranking | 1000 | 0.671 | 0.475901770019 | 3.717 | 2 |
| `mostpop_train_only` | ML-1M | reranking | 1000 | 0.671 | 0.475901770019 | 3.717 | 2 |
| `bm25_like_e3b` | Amazon Beauty | ranking | 994 | 0.557344064386 | 0.41711426029 | 5.54929577465 | 3 |
| `bm25_like_e3b` | Amazon Beauty | reranking | 994 | 0.557344064386 | 0.41711426029 | 5.54929577465 | 3 |
| `bm25_like_e3b` | Amazon Sports | ranking | 913 | 0.524644030668 | 0.382264356757 | 5.99123767798 | 4 |
| `bm25_like_e3b` | Amazon Sports | reranking | 913 | 0.524644030668 | 0.382264356757 | 5.99123767798 | 4 |
| `bm25_like_e3b` | ML-1M | ranking | 1000 | 0.522 | 0.364931424919 | 5.815 | 4 |
| `bm25_like_e3b` | ML-1M | reranking | 1000 | 0.522 | 0.364931424919 | 5.815 | 4 |

## 16. Ranking results

| baseline | dataset | task | rows | HR@5 | NDCG@5 | mean rank | median rank |
|---|---|---|---:|---:|---:|---:|---:|
| `random_seeded` | Amazon Beauty | ranking | 1000 | 0.256 | 0.157981001577 | 9.468 | 10 |
| `random_seeded` | Amazon Sports | ranking | 1000 | 0.239 | 0.140996381563 | 9.646 | 10 |
| `random_seeded` | ML-1M | ranking | 1000 | 0.266 | 0.154581706889 | 9.277 | 9 |
| `mostpop_train_only` | Amazon Beauty | ranking | 1000 | 0.48 | 0.325883987566 | 7.068 | 5 |
| `mostpop_train_only` | Amazon Sports | ranking | 1000 | 0.479 | 0.346300626572 | 7.01 | 5 |
| `mostpop_train_only` | ML-1M | ranking | 1000 | 0.671 | 0.475901770019 | 3.717 | 2 |
| `bm25_like_e3b` | Amazon Beauty | ranking | 1000 | 0.557 | 0.417611574729 | 5.541 | 3 |
| `bm25_like_e3b` | Amazon Sports | ranking | 1000 | 0.515 | 0.373698112433 | 6.156 | 4 |
| `bm25_like_e3b` | ML-1M | ranking | 1000 | 0.522 | 0.364931424919 | 5.815 | 4 |

## 17. Reranking-candidate-set results

| baseline | dataset | task | rows | HR@5 | NDCG@5 | mean rank | median rank |
|---|---|---|---:|---:|---:|---:|---:|
| `random_seeded` | Amazon Beauty | reranking | 1000 | 0.246 | 0.142683522674 | 9.529 | 10 |
| `random_seeded` | Amazon Sports | reranking | 1000 | 0.222 | 0.129446675633 | 9.832 | 10 |
| `random_seeded` | ML-1M | reranking | 1000 | 0.241 | 0.143519403574 | 9.648 | 10 |
| `mostpop_train_only` | Amazon Beauty | reranking | 1000 | 0.48 | 0.325883987566 | 7.068 | 5 |
| `mostpop_train_only` | Amazon Sports | reranking | 1000 | 0.479 | 0.346300626572 | 7.01 | 5 |
| `mostpop_train_only` | ML-1M | reranking | 1000 | 0.671 | 0.475901770019 | 3.717 | 2 |
| `bm25_like_e3b` | Amazon Beauty | reranking | 1000 | 0.557 | 0.417611574729 | 5.541 | 3 |
| `bm25_like_e3b` | Amazon Sports | reranking | 1000 | 0.515 | 0.373698112433 | 6.156 | 4 |
| `bm25_like_e3b` | ML-1M | reranking | 1000 | 0.522 | 0.364931424919 | 5.815 | 4 |

## 18. Beauty findings

| baseline | dataset | task | rows | HR@5 | NDCG@5 | mean rank | median rank |
|---|---|---|---:|---:|---:|---:|---:|
| `random_seeded` | Amazon Beauty | ranking | 1000 | 0.256 | 0.157981001577 | 9.468 | 10 |
| `random_seeded` | Amazon Beauty | ranking | 994 | 0.255533199195 | 0.157796852941 | 9.45875251509 | 10 |
| `random_seeded` | Amazon Beauty | reranking | 1000 | 0.246 | 0.142683522674 | 9.529 | 10 |
| `random_seeded` | Amazon Beauty | reranking | 994 | 0.245472837022 | 0.14214956727 | 9.54828973843 | 10 |
| `mostpop_train_only` | Amazon Beauty | ranking | 1000 | 0.48 | 0.325883987566 | 7.068 | 5 |
| `mostpop_train_only` | Amazon Beauty | ranking | 994 | 0.478873239437 | 0.325777043515 | 7.08249496982 | 5 |
| `mostpop_train_only` | Amazon Beauty | reranking | 1000 | 0.48 | 0.325883987566 | 7.068 | 5 |
| `mostpop_train_only` | Amazon Beauty | reranking | 994 | 0.478873239437 | 0.325777043515 | 7.08249496982 | 5 |
| `bm25_like_e3b` | Amazon Beauty | ranking | 1000 | 0.557 | 0.417611574729 | 5.541 | 3 |
| `bm25_like_e3b` | Amazon Beauty | ranking | 994 | 0.557344064386 | 0.41711426029 | 5.54929577465 | 3 |
| `bm25_like_e3b` | Amazon Beauty | reranking | 1000 | 0.557 | 0.417611574729 | 5.541 | 3 |
| `bm25_like_e3b` | Amazon Beauty | reranking | 994 | 0.557344064386 | 0.41711426029 | 5.54929577465 | 3 |

## 19. Sports findings

| baseline | dataset | task | rows | HR@5 | NDCG@5 | mean rank | median rank |
|---|---|---|---:|---:|---:|---:|---:|
| `random_seeded` | Amazon Sports | ranking | 1000 | 0.239 | 0.140996381563 | 9.646 | 10 |
| `random_seeded` | Amazon Sports | ranking | 913 | 0.237677984666 | 0.140262023798 | 9.63526834611 | 10 |
| `random_seeded` | Amazon Sports | reranking | 1000 | 0.222 | 0.129446675633 | 9.832 | 10 |
| `random_seeded` | Amazon Sports | reranking | 913 | 0.221248630887 | 0.12958358171 | 9.85213581599 | 10 |
| `mostpop_train_only` | Amazon Sports | ranking | 1000 | 0.479 | 0.346300626572 | 7.01 | 5 |
| `mostpop_train_only` | Amazon Sports | ranking | 913 | 0.480832420591 | 0.344928430771 | 7.00547645126 | 5 |
| `mostpop_train_only` | Amazon Sports | reranking | 1000 | 0.479 | 0.346300626572 | 7.01 | 5 |
| `mostpop_train_only` | Amazon Sports | reranking | 913 | 0.480832420591 | 0.344928430771 | 7.00547645126 | 5 |
| `bm25_like_e3b` | Amazon Sports | ranking | 1000 | 0.515 | 0.373698112433 | 6.156 | 4 |
| `bm25_like_e3b` | Amazon Sports | ranking | 913 | 0.524644030668 | 0.382264356757 | 5.99123767798 | 4 |
| `bm25_like_e3b` | Amazon Sports | reranking | 1000 | 0.515 | 0.373698112433 | 6.156 | 4 |
| `bm25_like_e3b` | Amazon Sports | reranking | 913 | 0.524644030668 | 0.382264356757 | 5.99123767798 | 4 |

## 20. ML-1M findings

| baseline | dataset | task | rows | HR@5 | NDCG@5 | mean rank | median rank |
|---|---|---|---:|---:|---:|---:|---:|
| `random_seeded` | ML-1M | ranking | 1000 | 0.266 | 0.154581706889 | 9.277 | 9 |
| `random_seeded` | ML-1M | ranking | 1000 | 0.266 | 0.154581706889 | 9.277 | 9 |
| `random_seeded` | ML-1M | reranking | 1000 | 0.241 | 0.143519403574 | 9.648 | 10 |
| `random_seeded` | ML-1M | reranking | 1000 | 0.241 | 0.143519403574 | 9.648 | 10 |
| `mostpop_train_only` | ML-1M | ranking | 1000 | 0.671 | 0.475901770019 | 3.717 | 2 |
| `mostpop_train_only` | ML-1M | ranking | 1000 | 0.671 | 0.475901770019 | 3.717 | 2 |
| `mostpop_train_only` | ML-1M | reranking | 1000 | 0.671 | 0.475901770019 | 3.717 | 2 |
| `mostpop_train_only` | ML-1M | reranking | 1000 | 0.671 | 0.475901770019 | 3.717 | 2 |
| `bm25_like_e3b` | ML-1M | ranking | 1000 | 0.522 | 0.364931424919 | 5.815 | 4 |
| `bm25_like_e3b` | ML-1M | ranking | 1000 | 0.522 | 0.364931424919 | 5.815 | 4 |
| `bm25_like_e3b` | ML-1M | reranking | 1000 | 0.522 | 0.364931424919 | 5.815 | 4 |
| `bm25_like_e3b` | ML-1M | reranking | 1000 | 0.522 | 0.364931424919 | 5.815 | 4 |

## 21. Comparison with prior comparable outputs

- Random/MostPop prior Plan E3A and BM25-like prior Plan E3B outputs are comparable only for Plan E2 full 20-candidate dataset/task rows.
- MostPop and BM25-like full-cohort HR@5/NDCG@5 reconcile with the prior comparable Plan E3A/E3B metric evidence under the same candidate rows.
- Random is not expected to match the prior `random_deterministic_reference` exactly because Phase 3A uses the required `random_seeded` SHA256 derivation with global seed 20260710.
- Phase 3A writes row-level rankings, which E3A/E3B did not provide in the required canonical schema.
- Matched-cohort metrics are new filtered views over the same full-cohort rankings, so they are not separate stochastic reruns.

## 22. Incompatible historical results

- Plan B and Plan C 100-candidate or diagnostic-subset outputs are not mixed into Phase 3A.
- Plan D diagnostic rows are excluded by cohort validation.
- Historical files remain evidence but are not authoritative baseline rankings for future LLM comparisons.

## 23. Exact-paper alignment gaps

- Exact official BM25 cannot be claimed without verified document representation, query representation, preprocessing, parameters and aligned score artifacts.
- Random and MostPop run on reconstructed Plan E2 artifacts, so they are paper-like reconstructed rather than exact reproduced paper results.
- Neural traditional baselines require a separate environment and artifact feasibility audit.

## 24. Blocked and deferred traditional baselines

| baseline | category | rows |
|---|---|---:|
| `BPRMF` | intentionally_deferred | 6 |
| `GRU4Rec` | intentionally_deferred | 6 |
| `LightGCN` | intentionally_deferred | 6 |
| `SASRec` | intentionally_deferred | 6 |
| `bm25_exact_official` | artifact_blocked | 6 |

## 25. Canonical files for future metrics and LLM comparisons

- Run manifest: `results/tables/traditional_baseline_run_manifest.csv`.
- Row-level baseline rankings: `results/tables/traditional_baseline_rankings.csv`.
- Utility metrics: `results/tables/traditional_baselines_metrics.csv`.
- Validation checks: `results/tables/traditional_baseline_validation.csv`.
- Blocked/deferred baselines: `results/tables/traditional_baselines_blocked.csv`.

## 26. Test and validation commands

- Builder command: `PYTHONNOUSERSITE=1 C:\ProgramData\anaconda3\python.exe src/experiments/build_plan_e_non_neural_baselines.py`.
- Required targeted pytest command: `PYTHONNOUSERSITE=1 C:\ProgramData\anaconda3\python.exe -m pytest tests/test_plan_e_non_neural_baselines.py tests/test_experiment_cohort_policy.py tests/test_item_title_map_reconciliation.py tests/test_dataset_candidate_set_validation.py`.
- Generated CSV hashes from this build:
  - `results/tables/traditional_baselines_blocked.csv`: `db79a9363b4d4c3c12fbe0a5c038395f47c2ebd42d896386c52b97cb8fbf06b3`
  - `results/tables/traditional_baseline_run_manifest.csv`: `a33a4897b9ed5695fa738176d4000dc1c81d3cd2e66d99e87e0b18eb2771f93c`
  - `results/tables/traditional_baselines_metrics.csv`: `718f8b61d87118109654b94bd5dc8952ae24e407bf19def6b6bf4c9d4948d597`
  - `results/tables/traditional_baseline_rankings.csv`: `b9e99e14a70f7af60d602e32b5dfbbd5624493177b8dd9c9908fba0abad5e7ba`
  - `results/tables/traditional_baseline_validation.csv`: `caa405d7864a1c75be226cfb98ca62f695383b2bc604317ff87db81a95b7e663`
- Validation table error failures: 0.

## 27. Recommended single next task

Phase 3B neural-baseline environment and feasibility audit for BPRMF, GRU4Rec, SASRec and LightGCN.
