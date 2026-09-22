# Plan E SASRec Beauty and Sports Execution and Same-Cohort Baseline Comparison

## 1. Scope and provenance

- Branch: `plan-d-extensions`.
- Commit: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests`.
- Scope: controlled Beauty/Sports SASRec execution and same-cohort consolidation with Random, MostPop, BM25-like, BPRMF, GRU4Rec and ML-1M SASRec evidence.

## 2. Executive summary

- Beauty/Sports SASRec ranking rows: `3954`.
- All-dataset SASRec metric rows: `12`.
- Same-cohort six-baseline metric rows: `72`.
- Paired comparison rows: `60`.
- Warning inventory rows: `1`.

## 3. Scientific classification

The runs are `adapted_paper_like_reconstructed`, not exact paper reproduction, because they use reconstructed Plan C data, Plan E2 cohorts and local SASRec code.

## 4. Protected inputs and safety boundaries

The training runner verified protected hashes before and after each dataset. It did not modify Phase 1-3J artifacts, BPRMF or GRU4Rec outputs, candidate manifests, reconstructed interactions, datamaps, title maps, or `original_repo/`.

## 5. CUDA interpreter recovery context

The run uses the recovered repository venv CUDA interpreter after earlier WindowsApps alias failures.

## 6. Absolute interpreter invocation policy

The manifest records `python_executable=C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe` and `PYTHONNOUSERSITE=1` was required by the runner.

## 7. Controlled configuration

Both datasets use seed `42`, embedding and hidden size `64`, two attention heads, two Transformer blocks, max sequence length `50`, batch size `512`, learning rate `0.001`, five epochs and deterministic CUDA algorithms.

## 8. Execution environment

`environment_id=repo_venv_cuda`, Python `3.12.10`, torch `2.6.0+cu124`, CUDA `12.4`, device `NVIDIA GeForce RTX 4060 Laptop GPU`.

## 9. Warning inventory

Warning rows: `1`; blocking warning rows: `0`.

## 10. Beauty training data

`training_interactions=176139`, `training_users=22363`, `training_items=12092`.

## 11. Sports training data

`training_interactions=260739`, `training_users=35598`, `training_items=18347`.

## 12. Beauty chronology and timestamp ties

Equal timestamp groups from sequence summary total `33104` with `100787` affected events.

## 13. Sports chronology and timestamp ties

Equal timestamp groups from sequence summary total `51364` with `159538` affected events.

## 14. SASRec objective construction

Beauty examples: `153776`. Sports examples: `225141`. Both use all-prefix next-item objectives from train-only histories.

## 15. Negative-sampling contract

The retained manifest hashes `negative_sampling_evidence_sha256` for primary and repeat runs; negatives are sampled by the deterministic Plan D user-positive exclusion sampler.

## 16. Item-token mappings

Beauty token rows: `12092`. Sports token rows: `18347`. Token `0` is absent from real item rows.

## 17. Padding semantics

Left padding uses index `0`; real item tokens are dense train indices plus one.

## 18. Absolute positional embeddings

Position embedding cardinality is `50`, matching the maximum sequence length.

## 19. Causal and padding masks

Strict cohort rows require `causal_mask_verified=true` and `padding_mask_verified=true`; scoring rows also passed finite-score validation.

## 20. Strict SASRec cohorts

Beauty scored rows: `1966`. Sports scored rows: `1988`. Combined excluded rows: `46`.

## 21. Beauty exclusions

Beauty excluded rows: `34`; exclusion reasons are from `results/tables/plan_e_sasrec_beauty_sports_cohort.csv`.

## 22. Sports exclusions

Sports excluded rows: `12`; exclusion reasons are from `results/tables/plan_e_sasrec_beauty_sports_cohort.csv`.

## 23. Beauty primary training

Run `plan_e_sasrec_beauty_primary` completed in `78.186` seconds with checkpoint `results/models/plan_e_sasrec_beauty/model_state.pt`.

## 24. Beauty same-seed repeat

Run `plan_e_sasrec_beauty_same_seed_repeat` completed in `77.156` seconds and retained no checkpoint.

## 25. Sports primary training

Run `plan_e_sasrec_sports_primary` completed in `115.159` seconds with checkpoint `results/models/plan_e_sasrec_sports/model_state.pt`.

## 26. Sports same-seed repeat

Run `plan_e_sasrec_sports_same_seed_repeat` completed in `109.365` seconds and retained no checkpoint.

## 27. Runtime and GPU memory

Combined training-history rows: `20`. Peak reserved bytes by dataset are Beauty `478150656` and Sports `478150656`.

## 28. Checkpoints and metadata

Primary checkpoints retained: `2`; repeat checkpoints retained: `0`.

## 29. Beauty ranking results

`evaluated_rows=983`, `HR@5=0.627670`, `NDCG@5=0.473619`, `mean_positive_rank_zero_based=4.693795`.

## 30. Beauty reranking-candidate results

`evaluated_rows=983`, `HR@5=0.627670`, `NDCG@5=0.473619`, `mean_positive_rank_zero_based=4.693795`.

## 31. Sports ranking results

`evaluated_rows=994`, `HR@5=0.675050`, `NDCG@5=0.504323`, `mean_positive_rank_zero_based=3.744467`.

## 32. Sports reranking-candidate results

`evaluated_rows=994`, `HR@5=0.675050`, `NDCG@5=0.504323`, `mean_positive_rank_zero_based=3.744467`.

## 33. Beauty full strict metrics

Ranking: `evaluated_rows=983`, `HR@5=0.627670`, `NDCG@5=0.473619`, `mean_positive_rank_zero_based=4.693795`. Reranking: `evaluated_rows=983`, `HR@5=0.627670`, `NDCG@5=0.473619`, `mean_positive_rank_zero_based=4.693795`.

## 34. Beauty matched strict metrics

Ranking: `evaluated_rows=977`, `HR@5=0.625384`, `NDCG@5=0.470764`, `mean_positive_rank_zero_based=4.721597`. Reranking: `evaluated_rows=977`, `HR@5=0.625384`, `NDCG@5=0.470764`, `mean_positive_rank_zero_based=4.721597`.

## 35. Sports full strict metrics

Ranking: `evaluated_rows=994`, `HR@5=0.675050`, `NDCG@5=0.504323`, `mean_positive_rank_zero_based=3.744467`. Reranking: `evaluated_rows=994`, `HR@5=0.675050`, `NDCG@5=0.504323`, `mean_positive_rank_zero_based=3.744467`.

## 36. Sports matched strict metrics

Ranking: `evaluated_rows=907`, `HR@5=0.683572`, `NDCG@5=0.510382`, `mean_positive_rank_zero_based=3.692393`. Reranking: `evaluated_rows=907`, `HR@5=0.683572`, `NDCG@5=0.510382`, `mean_positive_rank_zero_based=3.692393`.

## 37. ML-1M Phase 3J integration

ML-1M rows in the all-dataset table: `4`. HR@5 and NDCG@5 were checked for exact carry-over from Phase 3J.

## 38. All-dataset SASRec summary

All-dataset rows: `12` across datasets `Amazon Beauty, Amazon Sports, ML-1M`.

## 39. History and truncation comparison

Beauty strict rows with truncation: `8`. Sports strict rows with truncation: `2`.

## 40. Same-cohort six-baseline reconstruction

Same-cohort rows: `72` for six baselines, three datasets, two tasks and two SASRec cohorts.

## 41. Random versus SASRec

Rows: `12`; mean HR difference SASRec minus comparator `0.492454`; mean NDCG difference `0.437835`.

## 42. MostPop versus SASRec

Rows: `12`; mean HR difference SASRec minus comparator `0.193286`; mean NDCG difference `0.200252`.

## 43. BM25-like versus SASRec

Rows: `12`; mean HR difference SASRec minus comparator `0.203828`; mean NDCG difference `0.195361`.

## 44. BPRMF versus SASRec

Rows: `12`; mean HR difference SASRec minus comparator `0.163334`; mean NDCG difference `0.163164`.

## 45. GRU4Rec versus SASRec

Rows: `12`; mean HR difference SASRec minus comparator `0.057133`; mean NDCG difference `0.067916`.

## 46. Dataset-specific findings

Beauty same-cohort rows: `24`. Sports same-cohort rows: `24`. ML-1M same-cohort rows: `24`.

## 47. Ranking versus reranking interpretation

Ranking rows in comparison: `30`. Reranking rows in comparison: `30`.

## 48. Paired positive-rank findings

Across paired rows, SASRec-better counts sum to `33703`, comparator-better counts sum to `12224`, and equal counts sum to `12303`.

## 49. Candidate and row identity validation

All comparison rows have `exact_row_key_match=true` and `exact_candidate_list_match=true`.

## 50. Determinism findings

Determinism rows: `28`; failed rows: `0`.

## 51. Warning interpretation

The warning inventory records the reproducible focused-test odd-head Transformer warning and classifies it as low risk for Phase 3K project-data runs, which use two attention heads.

## 52. WindowsApps interpreter limitation

Relative `python` and WindowsApps alias execution remain outside this phase's allowed execution policy.

## 53. Absolute-position reconstruction limitation

Absolute positions are reconstructed from left-padded train histories; this is adapted evidence, not verified original-paper SASRec internals.

## 54. Equal-timestamp reconstruction limitation

Equal timestamp groups are ordered by frozen source row index; this is deterministic but reconstructed.

## 55. Deviations from the original paper

The run uses reconstructed data, local SASRec, strict 20-candidate Plan E2 cohorts and same-cohort adapted comparisons.

## 56. Limitations

No LightGCN, LLM inference, API call, prompt generation, or significance testing was performed.

## 57. Whether Phase 3K succeeded

Phase 3K succeeded if all generated validation rows pass and no blocking warnings are present. Current validation failed rows: `0`.

## 58. SASRec completion assessment

Beauty, Sports and ML-1M now have SASRec adapted Plan E2 strict-cohort metrics in `results/tables/plan_e_sasrec_all_datasets_metrics.csv`.

## 59. Conditions for beginning LightGCN

The next phase can begin only if these artifacts remain unchanged and the same candidate/cohort identity policy is reused.

## 60. Protected-input immutability

Protected input immutability was checked in the runner and final validation; Phase 2A candidate files were read-only for this phase.

## 61. Commands executed

Commands are recorded in the final response; generated CSVs contain executable, config and source-path provenance.

## 62. Recommended single next task

Phase 3L — LightGCN Plan E2 adapter and graph-construction preflight validation
