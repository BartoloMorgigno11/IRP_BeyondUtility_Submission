# Plan E BPRMF Beauty and Sports Execution and Same-Cohort Baseline Comparison

## 1. Scope and provenance

- Branch: `plan-d-extensions`.
- Commit: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests`.
- Scope: Beauty and Sports BPRMF execution plus same-cohort consolidation with existing Random, MostPop and BM25-like rankings.

## 2. Executive summary

- Beauty and Sports primary and same-seed repeat BPRMF runs completed.
- New Beauty/Sports BPRMF ranking rows: `3954` from `results/tables/plan_e_bprmf_beauty_sports_rankings.csv`.
- Same-cohort baseline metric rows: `48`.
- Paired comparison rows: `36`.

## 3. Scientific classification

The BPRMF results are `adapted_paper_like_reconstructed`, not exact paper reproduction.

## 4. Protected inputs and safety boundaries

Protected hashes were checked before and after training. Key hashes:

- `src/baselines/plan_d_bprmf.py`: `ADDFE475C9FE718A4CC8C6E051C9FF7C93FAE33431E5531C04D35203865034B8`.
- `src/baselines/plan_e_bprmf_adapter.py`: `DEAEA8B4DC9A809C89E4547964329B745C5E460A8E375D1618096CBE0866AE8F`.
- `src/baselines/plan_e_bprmf_training.py`: `08C1E571E5140778964468953395B97153E8DF1201E2A4F004F54842AE3F8090`.
- `results/tables/ranking_candidate_sets.csv`: `640AD9C00B7D7C825AD10746A0E720C161D1BA9D75CC791A019DF7FE14C7A955`.
- `results/tables/traditional_baseline_rankings.csv`: `B9E99E14A70F7AF60D602E32B5DFBBD5624493177B8DD9C9908FBA0ABAD5E7BA`.
- `data/reconstructed/plan_c/amazon_beauty/interactions.csv`: `D2A7801A024349063CFF7C455E872EED6E8EB0132324ED8B710BE1FF9A0E9478`.
- `data/reconstructed/plan_c/amazon_sports/interactions.csv`: `E7BC05D829B83DCD286E3FE1ACBCCB61341A8AB709F4260F9889633552829B40`.

## 5. Controlled configuration policy

Beauty and Sports reused the Phase 3D seed, embedding dimension, optimizer, learning rate, batch size, epochs, deterministic CUDA settings, negative-sampling policy and tie-breaking rule.

## 6. Execution environment

Execution used `repo_venv_cuda`; environment details are recorded in `results/tables/plan_e_bprmf_beauty_sports_run_manifest.csv`.

## 7. Beauty training data

Training rows: `176139`; users: `22363`; items: `12092`; source: `data/reconstructed/plan_c/amazon_beauty/interactions.csv`.

## 8. Sports training data

Training rows: `260739`; users: `35598`; items: `18347`; source: `data/reconstructed/plan_c/amazon_sports/interactions.csv`.

## 9. Mapping construction

Combined Beauty/Sports mapping rows: `88400` in `results/tables/plan_e_bprmf_beauty_sports_model_id_map.csv`.

## 10. Strict train-known cohorts

Beauty scored rows: `1966`. Sports scored rows: `1988`.

## 11. Beauty excluded rows

Beauty excluded rows: `34` from the combined cohort table.

## 12. Sports excluded rows

Sports excluded rows: `12` from the combined cohort table.

## 13. Negative-sampling contract

The validation table records the bounded deterministic negative-sampling check for each dataset.

## 14. Beauty primary training

Primary duration `14.114672800` seconds; final loss `0.6162866611372341`.

## 15. Beauty same-seed repeat

Repeat duration `12.049424200` seconds; final loss `0.6162866611372341`.

## 16. Sports primary training

Primary duration `9.697159100` seconds; final loss `0.6361491745337844`.

## 17. Sports same-seed repeat

Repeat duration `10.605653100` seconds; final loss `0.6361491745337844`.

## 18. Runtime and GPU-memory findings

Runtime and `gpu_free_before_bytes` are recorded per run in the run manifest; CUDA peak allocation/reservation is recorded per epoch in training history.

## 19. Checkpoints and metadata

Beauty checkpoint: `results/models/plan_e_bprmf_beauty/model_state.pt`. Sports checkpoint: `results/models/plan_e_bprmf_sports/model_state.pt`. Repeat checkpoints were not retained.

## 20. Beauty ranking results

- Amazon Beauty ranking plan_e2_bprmf_strict_train_known_full: rows=983, HR@5=0.473041709054, NDCG@5=0.350632776189, mean_rank=6.68056968464.
- Amazon Beauty ranking plan_e2_bprmf_strict_train_known_matched: rows=977, HR@5=0.473899692938, NDCG@5=0.35073901637, mean_rank=6.68065506653.

## 21. Beauty reranking-candidate results

- Amazon Beauty reranking plan_e2_bprmf_strict_train_known_full: rows=983, HR@5=0.473041709054, NDCG@5=0.350632776189, mean_rank=6.68056968464.
- Amazon Beauty reranking plan_e2_bprmf_strict_train_known_matched: rows=977, HR@5=0.473899692938, NDCG@5=0.35073901637, mean_rank=6.68065506653.

## 22. Sports ranking results

- Amazon Sports ranking plan_e2_bprmf_strict_train_known_full: rows=994, HR@5=0.516096579477, NDCG@5=0.387193161909, mean_rank=5.99698189135.
- Amazon Sports ranking plan_e2_bprmf_strict_train_known_matched: rows=907, HR@5=0.519294377067, NDCG@5=0.389563252693, mean_rank=5.90297684675.

## 23. Sports reranking-candidate results

- Amazon Sports reranking plan_e2_bprmf_strict_train_known_full: rows=994, HR@5=0.516096579477, NDCG@5=0.387193161909, mean_rank=5.99698189135.
- Amazon Sports reranking plan_e2_bprmf_strict_train_known_matched: rows=907, HR@5=0.519294377067, NDCG@5=0.389563252693, mean_rank=5.90297684675.

## 24. Beauty full strict-cohort metrics

- Amazon Beauty ranking plan_e2_bprmf_strict_train_known_full: rows=983, HR@5=0.473041709054, NDCG@5=0.350632776189, mean_rank=6.68056968464.
- Amazon Beauty reranking plan_e2_bprmf_strict_train_known_full: rows=983, HR@5=0.473041709054, NDCG@5=0.350632776189, mean_rank=6.68056968464.

## 25. Beauty matched strict-cohort metrics

- Amazon Beauty ranking plan_e2_bprmf_strict_train_known_matched: rows=977, HR@5=0.473899692938, NDCG@5=0.35073901637, mean_rank=6.68065506653.
- Amazon Beauty reranking plan_e2_bprmf_strict_train_known_matched: rows=977, HR@5=0.473899692938, NDCG@5=0.35073901637, mean_rank=6.68065506653.

## 26. Sports full strict-cohort metrics

- Amazon Sports ranking plan_e2_bprmf_strict_train_known_full: rows=994, HR@5=0.516096579477, NDCG@5=0.387193161909, mean_rank=5.99698189135.
- Amazon Sports reranking plan_e2_bprmf_strict_train_known_full: rows=994, HR@5=0.516096579477, NDCG@5=0.387193161909, mean_rank=5.99698189135.

## 27. Sports matched strict-cohort metrics

- Amazon Sports ranking plan_e2_bprmf_strict_train_known_matched: rows=907, HR@5=0.519294377067, NDCG@5=0.389563252693, mean_rank=5.90297684675.
- Amazon Sports reranking plan_e2_bprmf_strict_train_known_matched: rows=907, HR@5=0.519294377067, NDCG@5=0.389563252693, mean_rank=5.90297684675.

## 28. ML-1M Phase 3D integration

ML-1M primary metrics were copied from `results/tables/plan_e_bprmf_ml1m_pilot_metrics.csv` into the all-dataset summary without retraining.

## 29. All-dataset BPRMF summary

- Amazon Beauty ranking plan_e2_bprmf_strict_train_known_full: rows=983, HR@5=0.473041709054, NDCG@5=0.350632776189, mean_rank=6.68056968464.
- Amazon Beauty ranking plan_e2_bprmf_strict_train_known_matched: rows=977, HR@5=0.473899692938, NDCG@5=0.35073901637, mean_rank=6.68065506653.
- Amazon Beauty reranking plan_e2_bprmf_strict_train_known_full: rows=983, HR@5=0.473041709054, NDCG@5=0.350632776189, mean_rank=6.68056968464.
- Amazon Beauty reranking plan_e2_bprmf_strict_train_known_matched: rows=977, HR@5=0.473899692938, NDCG@5=0.35073901637, mean_rank=6.68065506653.
- Amazon Sports ranking plan_e2_bprmf_strict_train_known_full: rows=994, HR@5=0.516096579477, NDCG@5=0.387193161909, mean_rank=5.99698189135.
- Amazon Sports ranking plan_e2_bprmf_strict_train_known_matched: rows=907, HR@5=0.519294377067, NDCG@5=0.389563252693, mean_rank=5.90297684675.
- Amazon Sports reranking plan_e2_bprmf_strict_train_known_full: rows=994, HR@5=0.516096579477, NDCG@5=0.387193161909, mean_rank=5.99698189135.
- Amazon Sports reranking plan_e2_bprmf_strict_train_known_matched: rows=907, HR@5=0.519294377067, NDCG@5=0.389563252693, mean_rank=5.90297684675.
- ML-1M ranking plan_e2_bprmf_strict_train_known_full: rows=981, HR@5=0.728848114169, NDCG@5=0.518607480492, mean_rank=3.15392456677.
- ML-1M ranking plan_e2_bprmf_strict_train_known_matched: rows=981, HR@5=0.728848114169, NDCG@5=0.518607480492, mean_rank=3.15392456677.
- ML-1M reranking plan_e2_bprmf_strict_train_known_full: rows=981, HR@5=0.728848114169, NDCG@5=0.518607480492, mean_rank=3.15392456677.
- ML-1M reranking plan_e2_bprmf_strict_train_known_matched: rows=981, HR@5=0.728848114169, NDCG@5=0.518607480492, mean_rank=3.15392456677.

## 30. Same-cohort non-neural metric reconstruction

Existing Phase 3A rankings were filtered to the exact BPRMF row keys; no Random, MostPop or BM25-like scoring was rerun.

## 31. Random versus BPRMF

- Amazon Beauty ranking plan_e2_bprmf_strict_train_known_full: HR diff=0.220752797558, NDCG diff=0.194768062971, BPRMF rank win rate=0.622583926755.
- Amazon Beauty ranking plan_e2_bprmf_strict_train_known_matched: HR diff=0.222108495394, NDCG diff=0.195074652666, BPRMF rank win rate=0.622313203685.
- Amazon Beauty reranking plan_e2_bprmf_strict_train_known_full: HR diff=0.225839267548, NDCG diff=0.206910612093, BPRMF rank win rate=0.6002034588.
- Amazon Beauty reranking plan_e2_bprmf_strict_train_known_matched: HR diff=0.227226202661, NDCG diff=0.207553720056, BPRMF rank win rate=0.601842374616.
- Amazon Sports ranking plan_e2_bprmf_strict_train_known_full: HR diff=0.276659959759, NDCG diff=0.246351731765, BPRMF rank win rate=0.65291750503.
- Amazon Sports ranking plan_e2_bprmf_strict_train_known_matched: HR diff=0.281146637266, NDCG diff=0.249475901284, BPRMF rank win rate=0.656008820287.
- Amazon Sports reranking plan_e2_bprmf_strict_train_known_full: HR diff=0.29476861167, NDCG diff=0.25798904413, BPRMF rank win rate=0.669014084507.
- Amazon Sports reranking plan_e2_bprmf_strict_train_known_matched: HR diff=0.298787210584, NDCG diff=0.260244589473, BPRMF rank win rate=0.674751929438.
- ML-1M ranking plan_e2_bprmf_strict_train_known_full: HR diff=0.461773700306, NDCG diff=0.363018033224, BPRMF rank win rate=0.769622833843.
- ML-1M ranking plan_e2_bprmf_strict_train_known_matched: HR diff=0.461773700306, NDCG diff=0.363018033224, BPRMF rank win rate=0.769622833843.
- ML-1M reranking plan_e2_bprmf_strict_train_known_full: HR diff=0.489296636086, NDCG diff=0.375823643127, BPRMF rank win rate=0.803261977574.
- ML-1M reranking plan_e2_bprmf_strict_train_known_matched: HR diff=0.489296636086, NDCG diff=0.375823643127, BPRMF rank win rate=0.803261977574.

## 32. MostPop versus BPRMF

- Amazon Beauty ranking plan_e2_bprmf_strict_train_known_full: HR diff=-0.00712105798576, NDCG diff=0.0241210486832, BPRMF rank win rate=0.507629704985.
- Amazon Beauty ranking plan_e2_bprmf_strict_train_known_matched: HR diff=-0.00511770726714, NDCG diff=0.0243322386564, BPRMF rank win rate=0.508700102354.
- Amazon Beauty reranking plan_e2_bprmf_strict_train_known_full: HR diff=-0.00712105798576, NDCG diff=0.0241210486832, BPRMF rank win rate=0.507629704985.
- Amazon Beauty reranking plan_e2_bprmf_strict_train_known_matched: HR diff=-0.00511770726714, NDCG diff=0.0243322386564, BPRMF rank win rate=0.508700102354.
- Amazon Sports ranking plan_e2_bprmf_strict_train_known_full: HR diff=0.0362173038229, NDCG diff=0.0403112438291, BPRMF rank win rate=0.51509054326.
- Amazon Sports ranking plan_e2_bprmf_strict_train_known_matched: HR diff=0.0374862183021, NDCG diff=0.0440068499437, BPRMF rank win rate=0.518191841235.
- Amazon Sports reranking plan_e2_bprmf_strict_train_known_full: HR diff=0.0362173038229, NDCG diff=0.0403112438291, BPRMF rank win rate=0.51509054326.
- Amazon Sports reranking plan_e2_bprmf_strict_train_known_matched: HR diff=0.0374862183021, NDCG diff=0.0440068499437, BPRMF rank win rate=0.518191841235.
- ML-1M ranking plan_e2_bprmf_strict_train_known_full: HR diff=0.059123343527, NDCG diff=0.0448772677603, BPRMF rank win rate=0.38124362895.
- ML-1M ranking plan_e2_bprmf_strict_train_known_matched: HR diff=0.059123343527, NDCG diff=0.0448772677603, BPRMF rank win rate=0.38124362895.
- ML-1M reranking plan_e2_bprmf_strict_train_known_full: HR diff=0.059123343527, NDCG diff=0.0448772677603, BPRMF rank win rate=0.38124362895.
- ML-1M reranking plan_e2_bprmf_strict_train_known_matched: HR diff=0.059123343527, NDCG diff=0.0448772677603, BPRMF rank win rate=0.38124362895.

## 33. BM25-like versus BPRMF

- Amazon Beauty ranking plan_e2_bprmf_strict_train_known_full: HR diff=-0.0834181078332, NDCG diff=-0.0657962321746, BPRMF rank win rate=0.386571719227.
- Amazon Beauty ranking plan_e2_bprmf_strict_train_known_matched: HR diff=-0.0829068577277, NDCG diff=-0.0651767617479, BPRMF rank win rate=0.386898669396.
- Amazon Beauty reranking plan_e2_bprmf_strict_train_known_full: HR diff=-0.0834181078332, NDCG diff=-0.0657962321746, BPRMF rank win rate=0.386571719227.
- Amazon Beauty reranking plan_e2_bprmf_strict_train_known_matched: HR diff=-0.0829068577277, NDCG diff=-0.0651767617479, BPRMF rank win rate=0.386898669396.
- Amazon Sports ranking plan_e2_bprmf_strict_train_known_full: HR diff=0, NDCG diff=0.0126345506158, BPRMF rank win rate=0.4476861167.
- Amazon Sports ranking plan_e2_bprmf_strict_train_known_matched: HR diff=-0.00661521499449, NDCG diff=0.00629918994527, BPRMF rank win rate=0.446527012128.
- Amazon Sports reranking plan_e2_bprmf_strict_train_known_full: HR diff=0, NDCG diff=0.0126345506158, BPRMF rank win rate=0.4476861167.
- Amazon Sports reranking plan_e2_bprmf_strict_train_known_matched: HR diff=-0.00661521499449, NDCG diff=0.00629918994527, BPRMF rank win rate=0.446527012128.
- ML-1M ranking plan_e2_bprmf_strict_train_known_full: HR diff=0.207951070336, NDCG diff=0.152611329817, BPRMF rank win rate=0.58002038736.
- ML-1M ranking plan_e2_bprmf_strict_train_known_matched: HR diff=0.207951070336, NDCG diff=0.152611329817, BPRMF rank win rate=0.58002038736.
- ML-1M reranking plan_e2_bprmf_strict_train_known_full: HR diff=0.207951070336, NDCG diff=0.152611329817, BPRMF rank win rate=0.58002038736.
- ML-1M reranking plan_e2_bprmf_strict_train_known_matched: HR diff=0.207951070336, NDCG diff=0.152611329817, BPRMF rank win rate=0.58002038736.

## 34. Dataset-specific comparison

Dataset-specific rows are in `results/tables/plan_e_bprmf_non_neural_comparison.csv`; every row has exact row-key and candidate-list match flags set to `true`.

## 35. Ranking versus reranking-candidate interpretation

Ranking and reranking candidate-set IDs remain separate in every output table.

## 36. Positive-rank paired comparison

Paired win/loss/equal counts sum to evaluated rows for all `36` comparison rows.

## 37. Candidate and row identity validation

The consolidation runner requires exact `(candidate_set_id, user_id)` membership and exact candidate-list equality before computing comparisons.

## 38. Determinism findings

Determinism rows: `22`; error-level failures: `0`.

## 39. Deviations from the original paper

The datasets and candidate sets are reconstructed Plan E2 artifacts and the BPRMF implementation is a local Plan D implementation.

## 40. Limitations

The task covers BPRMF only for Beauty and Sports; GRU4Rec, SASRec and LightGCN remain unexecuted.

## 41. Whether Phase 3E succeeded

Phase 3E succeeded if the CSV validation and focused tests pass; generated validation rows report `0` error-level failures.

## 42. Traditional-baseline completion status

Random, MostPop, BM25-like and BPRMF now have same-cohort metrics for Beauty, Sports and ML-1M.

## 43. Conditions for beginning GRU4Rec

Begin only after preserving these Phase 3E artifacts and applying a separate adapter/preflight phase for GRU4Rec.

## 44. Protected-input immutability

Protected inputs remained unchanged during the recorded run.

## 45. Commands executed

- `git status --short --branch`
- `git branch --show-current`
- `git log -1 --format="%H %s"`
- `.venv\Scripts\python.exe -m pytest -q tests/test_plan_e_bprmf_baseline_consolidation.py tests/test_plan_e_bprmf_training.py tests/test_plan_e_bprmf_adapter.py tests/test_experiment_cohort_policy.py`
- `.venv\Scripts\python.exe src/experiments/run_plan_e_bprmf_beauty_sports.py`
- `C:\ProgramData\anaconda3\python.exe src/experiments/build_plan_e_bprmf_baseline_consolidation.py`

## 46. Recommended single next task

Phase 3F - GRU4Rec Plan E2 adapter and preflight validation.
