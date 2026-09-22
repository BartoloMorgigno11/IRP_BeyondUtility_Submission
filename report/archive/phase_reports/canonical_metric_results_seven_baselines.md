# Canonical Utility and Beyond-Utility Results for Seven Baselines

## 1. Scope and provenance
- Branch: `plan-d-extensions`.
- Commit: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests`.
- Environment: `python=3.12.10;pandas=3.0.3;numpy=2.5.0;scipy=not_required_not_imported;torch=not_required_not_imported;executable=C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe;cwd=C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project;cpu_only=true`.

## 2. Executive summary
- Long-form result rows: `1092`; wide rows: `84`; status counts: `{"blocked_ambiguous_definition":168,"computed_adapted":168,"computed_reconstructed":252,"diagnostic_computed":168,"recomputed_reconciled":168,"structural_invariant_verified":168}`.
- Blocked-result rows: `168`. Popularity bias and ILD remain non-numeric.

## 3. Scientific classification
- Utility and selected beyond-utility metrics are reconstructed or adapted according to the Phase 4A contract, not exact original-paper reproduction.

## 4. Protected-input boundaries
- Existing rankings, cohorts, candidates, interactions, datamaps, title maps, checkpoints, paper files, and `original_repo/` were read only.

## 5. Frozen Phase 4A contract
- Contract source: `results/tables/canonical_metric_contract.csv`; all 13 contract hashes validated before computation.

## 6. Canonical baseline layer
- Baselines: `random_seeded, mostpop_train_only, bm25_like_e3b, bprmf_reconstructed, gru4rec_reconstructed, sasrec_reconstructed, lightgcn_reconstructed`.

## 7. Canonical datasets, tasks and cohorts
- Datasets: `Amazon Beauty, Amazon Sports, ML-1M`; tasks: `ranking`, `reranking`; cohorts: `plan_e2_lightgcn_strict_connected_full`, `plan_e2_lightgcn_strict_connected_matched`.

## 8. Ranking-source validation
- Canonical ranking rows: `41412`; top-5 exposures: `207060`.

## 9. Train-only popularity construction
- Train item rows by dataset: `{"Amazon Beauty":12092,"Amazon Sports":18347,"ML-1M":3703}`.

## 10. Item-popularity distributions
- Popularity summary: `{"Amazon Beauty":{"items":12092,"max_count":413,"total_train_interactions":176139},"Amazon Sports":{"items":18347,"max_count":963,"total_train_interactions":260739},"ML-1M":{"items":3703,"max_count":3403,"total_train_interactions":994169}}`.

## 11. Long-tail threshold construction
- Long-tail thresholds and tail counts: `{"Amazon Beauty":{"tail_items":9595,"threshold":"17"},"Amazon Sports":{"tail_items":14557,"threshold":"17"},"ML-1M":{"tail_items":2962,"threshold":"425.60000000000002"}}`.

## 12. Top-5 exposure extraction
- Exactly five top-5 exposures were generated for each canonical ranking row.

## 13. HR@5 results
- `hr_at_5` computed rows: `84`; min `0.22050716648291069`; max `0.90417940876656477`.

## 14. NDCG@5 results
- `ndcg_at_5` computed rows: `84`; min `0.12920411777926305`; max `0.7676203092393179`.

## 15. Mean positive-rank results
- `mean_positive_rank_zero_based` computed rows: `84`; min `1.3537206931702344`; max `9.8566703417861081`.

## 16. Median positive-rank results
- `median_positive_rank_zero_based` computed rows: `84`; min `0`; max `10`.

## 17. Novelty@5 contract
- Novelty uses train-only self-information in bits with no smoothing.

## 18. Novelty@5 results
- `novelty_at_5` computed rows: `84`; min `10.646133241080385`; max `14.744500017904512`.

## 19. Catalog-coverage contract
- Catalog coverage uses the train-seen catalog denominator.

## 20. Catalog-coverage results
- `catalog_coverage_at_5` computed rows: `84`; min `0.17087262222706709`; max `0.73480961382662702`.

## 21. Item-coverage contract
- Item coverage uses the strict same-cohort candidate-union denominator.

## 22. Item-coverage results
- `item_coverage_at_5` computed rows: `84`; min `0.27303605643616097`; max `0.73779826464208242`.

## 23. Long-tail-rate contract
- Long-tail membership uses train counts strictly below the frozen 0.8 quantile threshold.

## 24. Long-tail-rate results
- `long_tail_rate_at_5` computed rows: `84`; min `0.21388329979879278`; max `0.8055045871559634`.

## 25. Average-popularity contract
- Average popularity uses raw train interaction counts and is not popularity bias.

## 26. Average-popularity results
- `average_popularity_at_5` computed rows: `84`; min `14.859316427783904`; max `741.91498470948011`.

## 27. Duplicate-rate verification
- Duplicate Rate@5 verified as structural invariant `0` for all 84 cells.

## 28. Candidate-validity verification
- Candidate Validity Rate verified as structural invariant `1` for all 84 cells.

## 29. Popularity-bias blocked status
- `popularity_bias_at_5` has 84 blocked rows with empty numeric values.

## 30. ILD blocked status
- `intra_list_diversity_at_5` has 84 blocked rows with empty numeric values.

## 31. Beauty results
- `Amazon Beauty` rows for selected numeric metrics: `112`.

## 32. Sports results
- `Amazon Sports` rows for selected numeric metrics: `112`.

## 33. ML-1M results
- `ML-1M` rows for selected numeric metrics: `112`.

## 34. Random findings
- `random_seeded` computed/verified numeric rows: `132` across datasets, tasks and cohorts.

## 35. MostPop findings
- `mostpop_train_only` computed/verified numeric rows: `132` across datasets, tasks and cohorts.

## 36. BM25-like findings
- `bm25_like_e3b` computed/verified numeric rows: `132` across datasets, tasks and cohorts.

## 37. BPRMF findings
- `bprmf_reconstructed` computed/verified numeric rows: `132` across datasets, tasks and cohorts.

## 38. GRU4Rec findings
- `gru4rec_reconstructed` computed/verified numeric rows: `132` across datasets, tasks and cohorts.

## 39. SASRec findings
- `sasrec_reconstructed` computed/verified numeric rows: `132` across datasets, tasks and cohorts.

## 40. LightGCN findings
- `lightgcn_reconstructed` computed/verified numeric rows: `132` across datasets, tasks and cohorts.

## 41. Ranking versus reranking findings
- HR@5 cells by task: `{"ranking":42,"reranking":42}`.

## 42. Full versus matched cohort findings
- Full rows by dataset: `{"Amazon Beauty":"983","Amazon Sports":"994","ML-1M":"981"}`; matched rows by dataset: `{"Amazon Beauty":"977","Amazon Sports":"907","ML-1M":"981"}`.

## 43. Utility versus novelty trade-offs
- The matrix reports utility and novelty separately; no composite score or global best baseline is claimed.

## 44. Utility versus popularity findings
- Average popularity is reported separately from HR/NDCG and does not replace blocked popularity bias.

## 45. Utility versus coverage findings
- Catalog and item coverage use distinct denominators and are not averaged per row.

## 46. Structural invariants
- Duplicate and candidate-validity values are verified structural checks, not comparative quality wins.

## 47. Selection-bias implications
- All result cells use the frozen LightGCN-compatible strict cohorts; matched rows are filters of the same strict-full rows.

## 48. Mixed-device LightGCN provenance
- LightGCN source provenance remains mixed as documented before Phase 4B; no LightGCN computation was rerun.

## 49. Exact, reconstructed and adapted result labels
- Computation statuses preserve `recomputed_reconciled`, `diagnostic_computed`, `computed_reconstructed`, `computed_adapted`, and `structural_invariant_verified` distinctions.

## 50. Blocked metric interpretation
- Blocked metric rows are placeholders for transparency and must not be interpreted as zero values.

## 51. Long-form result matrix
- Path: `results/tables/canonical_metric_results_long.csv`; rows: `1092`.

## 52. Wide-form result matrix
- Path: `results/tables/canonical_metric_results_wide.csv`; rows: `84`.

## 53. Validation findings
- Validation rows: `46`; error-level failures: `0`.

## 54. Determinism findings
- Output hashes recorded for `9` generated CSVs before validation/report finalization.

## 55. Deviations from the original paper
- Catalog coverage and novelty are explicitly adapted where Phase 4A required adapted labels; popularity bias and ILD are blocked.

## 56. Limitations
- Results depend on reconstructed same-cohort artifacts and do not resolve blocked exact paper definitions.

## 57. Whether Phase 4B succeeded
- Phase 4B succeeded if validation remains passed and deterministic hashes match across the required rerun.

## 58. Metric-phase completion assessment
- The metric computation phase is complete for approved metrics and transparent for blocked metrics.

## 59. Protected-input immutability
- Protected hashes are recomputed inside the builder; generated outputs are separate Phase 4B artifacts.

## 60. Commands executed
- See final response for the complete command list.

## 61. Recommended single next task
Phase 5A — canonical prompt, parsing and LLM execution contract audit for paper-aligned ranking and reranking
