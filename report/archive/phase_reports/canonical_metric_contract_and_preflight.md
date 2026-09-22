# Canonical Utility and Beyond-Utility Metric Contract and Same-Cohort Preflight

## 1. Scope and provenance
- Branch: `plan-d-extensions`.
- Commit: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests`.
- Python executable: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe`.
- Python/pandas/NumPy/scipy/torch: `3.12.10`, `3.0.3`, `2.5.0`, `not_installed`, `2.6.0+cu124`.
- Protected input files hashed: `39`.

## 2. Executive summary
- Metric contracts written: `13`; status counts: `{"blocked_ambiguous_definition":2,"diagnostic_only":2,"ready_adapted":2,"ready_reconstructed":5,"structural_invariant":2}`.
- Same-cohort preflight rows: `84` and all rows have status `ready`.
- Applicability rows: `924`; status counts: `{"blocked":168,"ready":420,"ready_with_adaptation":168,"structural_invariant":168}`.
- Blocked/ambiguous issue rows: `7`.

## 3. Safety boundaries
- Phase 4A read existing artifacts only, recomputed HR@5/NDCG@5 only for reconciliation, and wrote only the requested Phase 4A outputs.

## 4. Completed seven-baseline layer
- Baselines in same-cohort table: `7`.
- Datasets: `3`; tasks: `2`; cohorts: `2`.

## 5. Canonical same-cohort definition
- Full cohort ID: `plan_e2_lightgcn_strict_connected_full`.
- Matched cohort ID: `plan_e2_lightgcn_strict_connected_matched`.

## 6. Paper metric inventory
- The local paper PDF was hashed, but no local PDF text extractor is installed. Paper-level claims are therefore tied to the PDF artifact plus official repository code and local codebase-map evidence.

## 7. Official-repository metric inventory
- Official metric evidence is `original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py`, including `calHR`, `calNDCG`, `calARP`, `calACLT`, `calItemCoverage`, `calSelfInfo`, `calMeanShannon`, `calMeanJaccard`, and `calHallucination`.

## 8. Local metric implementation inventory
- Inventory rows: `17`. Local sources include utility, coverage, hallucination, and Plan D beyond-utility implementations.

## 9. Conflicting metric definitions
- Conflicts are recorded for smoothed novelty, popularity-bias semantics, catalog versus candidate-pool coverage, duplicate denominator handling, and exact ILD representation.

## 10. Utility metric contract
- HR@5 and NDCG@5 are ready_reconstructed same-cohort utility metrics with top-5 cutoff and one positive item per row.

## 11. HR@5 definition
- HR@5 is `mean(positive_rank_zero_based < 5)` over rows.

## 12. NDCG@5 definition
- NDCG@5 is `1/log2(rank_one_based+1)` for positive ranks in the top five, otherwise `0`, averaged over rows.

## 13. Positive-rank semantics
- Stored positive ranks are zero-based in `positive_rank_zero_based`; one-based ranks are derived by adding one.

## 14. Utility metric reconciliation
- Reconciled rows: `84` of `84`.

## 15. Novelty definition
- Novelty@5 is ready_adapted as train-only self-information without invented smoothing; zero train probability blocks final use.

## 16. Catalog coverage definition
- Catalog Coverage@5 is adapted: unique valid top-5 recommended items divided by the train-seen catalog.

## 17. Item coverage definition
- Item Coverage@5 reconstructs official candidate-pool coverage as unique top-5 items divided by the strict same-cohort candidate union.

## 18. Long-tail rate definition
- Long-tail Rate@5 reconstructs official ACLT using the strict less-than 0.8 quantile of train item frequencies.

## 19. Average popularity definition
- Average Popularity@5 reconstructs official ARP as mean raw train interaction count over top-5 positions.

## 20. Popularity-bias definition
- Popularity Bias@5 remains blocked because source evidence conflicts and it must not duplicate Average Popularity@5.

## 21. Intra-list diversity definition
- Exact ILD remains blocked because the exact representation and similarity source are not frozen; category and title-token adaptations are feasible but explicitly adapted.

## 22. Duplicate-rate definition
- Duplicate Rate@5 is a structural invariant for deterministic candidate permutations, expected `0` when rankings are valid.

## 23. Candidate-validity definition
- Candidate Validity Rate is a structural invariant for candidate-constrained deterministic baselines, expected `1` when top-5 items are candidate members.

## 24. Aggregation-level contract
- Utility metrics are row-level macro means; popularity/novelty/long-tail are top-5 position means; coverage metrics are cohort-level unique-item ratios.

## 25. Top-5 extraction contract
- Top-5 extractability passed for `84` of `84` preflight units.

## 26. Duplicate and invalid-item handling
- Ranking membership passed for `84` of `84` preflight units.

## 27. Missing-data handling
- Missing popularity or metadata is not imputed. Blocked/adapted statuses identify metrics needing those inputs.

## 28. Train-only popularity sources
- Recommended train-only popularity rows: `3`.

## 29. Popularity input findings
- Popularity audit rows: `15`. Recommended sources are leakage-free train interaction counts only.

## 30. Catalog and item-universe findings
- Universe audit rows: `15` covering five universe candidates for each dataset.

## 31. Coverage-denominator decision
- Catalog coverage uses train_seen_catalog; item coverage uses strict_cohort_candidate_union.

## 32. Novelty smoothing decision
- No smoothing is approved. Plan D add-one smoothing is recorded as a conflict/adaptation.

## 33. Long-tail threshold evidence
- Recommended long-tail policies: `3`.

## 34. Popularity-bias history source
- No history-normalized popularity-bias source is frozen; the metric is blocked pending a source-backed formula.

## 35. ML-1M similarity and genre feasibility
- Similarity feasibility rows: `3`; ready/adapted rows: `2`; blocked rows: `1`.

## 36. Beauty similarity and metadata feasibility
- Similarity feasibility rows: `3`; ready/adapted rows: `2`; blocked rows: `1`.

## 37. Sports similarity and metadata feasibility
- Similarity feasibility rows: `3`; ready/adapted rows: `2`; blocked rows: `1`.

## 38. Title-token ILD adaptation assessment
- Canonical title-token Jaccard is feasible as an adapted source, but it is not recommended as exact ILD.

## 39. Structural invariants for deterministic baselines
- Structural invariant applicability rows: `168`.

## 40. Candidate validity applicability
- Candidate-validity rows in applicability matrix: `84`.

## 41. Duplicate-rate applicability
- Duplicate-rate rows in applicability matrix: `84`.

## 42. Full strict cohort preflight
- Observed rows by dataset/task: `{"Amazon Beauty/ranking":983,"Amazon Beauty/reranking":983,"Amazon Sports/ranking":994,"Amazon Sports/reranking":994,"ML-1M/ranking":981,"ML-1M/reranking":981}`.

## 43. Matched strict cohort preflight
- Observed rows by dataset/task: `{"Amazon Beauty/ranking":977,"Amazon Beauty/reranking":977,"Amazon Sports/ranking":907,"Amazon Sports/reranking":907,"ML-1M/ranking":981,"ML-1M/reranking":981}`.

## 44. Ranking task preflight
- `ranking` preflight rows: `42`; ready rows: `42`.

## 45. Reranking task preflight
- `reranking` preflight rows: `42`; ready rows: `42`.

## 46. Seven-baseline applicability matrix
- Applicability matrix contains `924` rows for 11 non-diagnostic metrics, 7 baselines, 3 datasets, 2 tasks, and 2 cohorts.

## 47. Exact metrics ready for computation
- Exact-ready contracts: `0`.

## 48. Reconstructed metrics ready for computation
- Reconstructed-ready contracts: `5`.

## 49. Adapted metrics requiring explicit labeling
- Adapted-ready contracts: `2`.

## 50. Blocked metrics
- Blocked contract rows: `2`; issue rows: `7`.

## 51. Selection-bias implications
- Final metrics must be reported only on the frozen LightGCN-compatible strict cohorts; no hidden intersections are allowed.

## 52. Mixed-device LightGCN provenance
- Beauty and Sports LightGCN source rows are CPU exact-repeat outputs; ML-1M uses the documented existing CUDA primary output with sparse-forward nondeterminism provenance.

## 53. Metric reproducibility requirements
- Phase 4B must reuse these contract, popularity, universe, long-tail, similarity, applicability, and blocked-policy tables without changing candidate or cohort membership.

## 54. Output schemas proposed for final computation
- Final Phase 4B should write a new long metric table that references metric_id, baseline_id, dataset, task_type, cohort_id, ranking_source_path, contract hash, input hashes, value, and status.

## 55. Final metric-computation stop conditions
- Stop if protected inputs change, utility reconciliation fails, a blocked metric is requested as final, or a required source has missing popularity/metadata without an approved adapted policy.

## 56. Tests
- Required tests are documented in the final command log and validate formulas, preflight helpers, and existing Phase 2/3 policy utilities.

## 57. Determinism validation
- Generated CSV hashes are deterministic when the builder is rerun under identical inputs; output file hashes are recorded for `10` CSVs.

## 58. Protected-input immutability
- Protected input hashes unchanged within the builder run: `true`.

## 59. Commands executed
- See final response for the shell command log. The builder itself writes only Phase 4A artifacts.

## 60. Recommended single next task
Phase 4B — canonical utility and beyond-utility metric computation for seven baselines on frozen same cohorts
