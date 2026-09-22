# Plan E GRU4Rec Beauty and Sports Execution and Same-Cohort Baseline Comparison

## 1. Scope and provenance

- Branch: `plan-d-extensions`.
- Commit: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests`.
- Scope: controlled Beauty/Sports GRU4Rec execution and same-cohort consolidation with Random, MostPop, BM25-like, BPRMF and ML-1M GRU4Rec evidence.

## 2. Executive summary

- Beauty/Sports GRU4Rec ranking rows: `3954`.
- All-dataset GRU4Rec metric rows: `12`.
- Same-cohort baseline metric rows: `60`.
- Paired comparison rows: `48`.
- Classification: `adapted_paper_like_reconstructed`.

## 3. Scientific classification

The runs are adapted paper-like reconstructed experiments, not exact paper reproduction, because they use Plan E2 reconstructed cohorts and a local GRU4Rec implementation.

## 4. Protected inputs and safety boundaries

Protected hashes were checked by the training runner and are repeated here for key files:

- `src/baselines/plan_d_gru4rec.py`: `5E288F1917B8F9A3C037D5C3EEF3BE4C2320DF465B23370D94982F15F63F699D`.
- `src/experiments/run_plan_d_gru4rec.py`: `42F3EECEB8BE58C94EBB353F5DD02BAE8EC55E9859EE6455470069E8206E40AC`.
- `src/baselines/plan_e_gru4rec_adapter.py`: `F3972AEDE7A085FD7D247826D0EB12B097A5F9EBB0E6F6ADA6143B4EE8C84DCD`.
- `src/baselines/plan_e_gru4rec_training.py`: `6E12301AA324ECADEA55B1AD8B1F7BD2E8D545A8C7A0114BEAD5AA45A6AC868F`.
- `src/experiments/run_plan_e_gru4rec_ml1m_pilot.py`: `422DE06F12C44667C8FDF93FA5A2A0EA84F9FD14A10D390822AE74C2607999EF`.
- `results/tables/ranking_candidate_sets.csv`: `640AD9C00B7D7C825AD10746A0E720C161D1BA9D75CC791A019DF7FE14C7A955`.
- `results/tables/positive_positions.csv`: `0C93DD55D5C96ED99FB7F1AFC8F5319E90F3092ECB6A48F2E9D46854B38CD6D2`.
- `results/tables/plan_e_gru4rec_row_compatibility.csv`: `632FA890F05401CECA5A6B64212370CA0E80182D5B9A55918A9AC3C10630F6C1`.
- `results/tables/plan_e_gru4rec_ml1m_pilot_rankings.csv`: `AF9674672647F2892AFDC7018DD3261ED1B9A215CDFCA105C686022457635B94`.
- `results/tables/traditional_baseline_rankings.csv`: `B9E99E14A70F7AF60D602E32B5DFBBD5624493177B8DD9C9908FBA0ABAD5E7BA`.
- `results/tables/plan_e_bprmf_beauty_sports_rankings.csv`: `428A1176C4BB903EC94E1704221ED19C5CFB74206A6446390B31AD41C516285A`.
- `results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv`: `A8C43DCB6D88D6A4726F36C5EBFFBADA54589C88A7F0988446A5439D083E547B`.
- `data/reconstructed/plan_c/amazon_beauty/interactions.csv`: `D2A7801A024349063CFF7C455E872EED6E8EB0132324ED8B710BE1FF9A0E9478`.
- `data/reconstructed/plan_c/amazon_sports/interactions.csv`: `E7BC05D829B83DCD286E3FE1ACBCCB61341A8AB709F4260F9889633552829B40`.

## 5. Controlled configuration policy

Beauty and Sports reused the Phase 3G seed `42`, embedding dimension `64`, hidden size `64`, one GRU layer, configured dropout `0.2`, sequence length `50`, Adam learning rate `0.001`, batch size `512`, five epochs, deterministic CUDA settings and HR/NDCG cutoff `5`.

## 6. Effective dropout semantics

The Plan D model passes recurrent dropout to PyTorch GRU, which is inactive when `num_layers=1`; the model also applies the configured embedding dropout in `GRU4RecModel.encode` during training. The architecture and configuration were not changed.

## 7. Execution environment

Executable `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe`; Python `3.12.10`; torch `2.6.0+cu124`; CUDA `12.4`; GPU `NVIDIA GeForce RTX 4060 Laptop GPU`.

## 8. Beauty training data

Training interactions: `176139`; training pairs: `153776`; source: `data/reconstructed/plan_c/amazon_beauty/interactions.csv`; data hash: `D2A7801A024349063CFF7C455E872EED6E8EB0132324ED8B710BE1FF9A0E9478`.

## 9. Sports training data

Training interactions: `260739`; training pairs: `225141`; source: `data/reconstructed/plan_c/amazon_sports/interactions.csv`; data hash: `E7BC05D829B83DCD286E3FE1ACBCCB61341A8AB709F4260F9889633552829B40`.

## 10. Beauty chronology and timestamp ties

Unique train users: `22363`; train items: `12092`; chronology hash: `54BE5113B37ED66C74DA141F4A8C2C6C8E2130FCB89F0F2F9C97F0D4B0FC997C`; users in sequence summary: `22363`.

## 11. Sports chronology and timestamp ties

Unique train users: `35598`; train items: `18347`; chronology hash: `599038AD7DADDCE5BD0BD144BD5F3697257992924CA5362A0D999AAD61626F8C`; users in sequence summary: `35598`.

## 12. Training-pair construction

Beauty training pairs: `153776`. Sports training pairs: `225141`. Both use all-prefix next-item pairs from `split=train` only.

## 13. Token mappings

Beauty token rows: `12092`. Sports token rows: `18347`. Token `0` is reserved for padding and real tokens are contiguous from `1`.

## 14. Padding and truncation

Beauty truncated strict rows: `8`. Sports truncated strict rows: `2`. Histories keep the most recent 50 train events.

## 15. Strict sequential cohorts

Beauty scored rows: `1966`; Sports scored rows: `1988`; combined excluded rows: `46`.

## 16. Beauty exclusions

Beauty excluded rows: `34`.

## 17. Sports exclusions

Sports excluded rows: `12`.

## 18. Beauty primary training

Primary duration `46.679330500075594` seconds; final loss `0.39515848304346152`; checkpoint `results/models/plan_e_gru4rec_beauty/model_state.pt`.

## 19. Beauty same-seed repeat

Repeat duration `49.638995400047861` seconds; final loss `0.39515848304346152`; checkpoint fields are empty.

## 20. Sports primary training

Primary duration `70.957285799901001` seconds; final loss `0.37604971548373051`; checkpoint `results/models/plan_e_gru4rec_sports/model_state.pt`.

## 21. Sports same-seed repeat

Repeat duration `69.032997000031173` seconds; final loss `0.37604971548373051`; checkpoint fields are empty.

## 22. Runtime and GPU memory

Primary total duration `117.637` seconds; repeat total duration `118.672` seconds; peak allocated `257491968` bytes; peak reserved `367001600` bytes.

## 23. Checkpoints and metadata

Beauty checkpoint: `results/models/plan_e_gru4rec_beauty/model_state.pt`. Sports checkpoint: `results/models/plan_e_gru4rec_sports/model_state.pt`. Repeat checkpoints were not retained.

## 24. Beauty ranking results

- Amazon Beauty ranking plan_e2_gru4rec_strict_sequential_full: rows=983, HR@5=0.547304170905, NDCG@5=0.387098748547, mean_rank=5.55442522889.
- Amazon Beauty ranking plan_e2_gru4rec_strict_sequential_matched: rows=977, HR@5=0.545547594678, NDCG@5=0.38536365124, mean_rank=5.57523029683.

## 25. Beauty reranking-candidate results

- Amazon Beauty reranking plan_e2_gru4rec_strict_sequential_full: rows=983, HR@5=0.547304170905, NDCG@5=0.387098748547, mean_rank=5.55442522889.
- Amazon Beauty reranking plan_e2_gru4rec_strict_sequential_matched: rows=977, HR@5=0.545547594678, NDCG@5=0.38536365124, mean_rank=5.57523029683.

## 26. Sports ranking results

- Amazon Sports ranking plan_e2_gru4rec_strict_sequential_full: rows=994, HR@5=0.61569416499, NDCG@5=0.442963541313, mean_rank=4.44668008048.
- Amazon Sports ranking plan_e2_gru4rec_strict_sequential_matched: rows=907, HR@5=0.617420066152, NDCG@5=0.444374240118, mean_rank=4.44983461963.

## 27. Sports reranking-candidate results

- Amazon Sports reranking plan_e2_gru4rec_strict_sequential_full: rows=994, HR@5=0.61569416499, NDCG@5=0.442963541313, mean_rank=4.44668008048.
- Amazon Sports reranking plan_e2_gru4rec_strict_sequential_matched: rows=907, HR@5=0.617420066152, NDCG@5=0.444374240118, mean_rank=4.44983461963.

## 28. Beauty full strict metrics

- Amazon Beauty ranking plan_e2_gru4rec_strict_sequential_full: rows=983, HR@5=0.547304170905, NDCG@5=0.387098748547, mean_rank=5.55442522889.
- Amazon Beauty reranking plan_e2_gru4rec_strict_sequential_full: rows=983, HR@5=0.547304170905, NDCG@5=0.387098748547, mean_rank=5.55442522889.

## 29. Beauty matched strict metrics

- Amazon Beauty ranking plan_e2_gru4rec_strict_sequential_matched: rows=977, HR@5=0.545547594678, NDCG@5=0.38536365124, mean_rank=5.57523029683.
- Amazon Beauty reranking plan_e2_gru4rec_strict_sequential_matched: rows=977, HR@5=0.545547594678, NDCG@5=0.38536365124, mean_rank=5.57523029683.

## 30. Sports full strict metrics

- Amazon Sports ranking plan_e2_gru4rec_strict_sequential_full: rows=994, HR@5=0.61569416499, NDCG@5=0.442963541313, mean_rank=4.44668008048.
- Amazon Sports reranking plan_e2_gru4rec_strict_sequential_full: rows=994, HR@5=0.61569416499, NDCG@5=0.442963541313, mean_rank=4.44668008048.

## 31. Sports matched strict metrics

- Amazon Sports ranking plan_e2_gru4rec_strict_sequential_matched: rows=907, HR@5=0.617420066152, NDCG@5=0.444374240118, mean_rank=4.44983461963.
- Amazon Sports reranking plan_e2_gru4rec_strict_sequential_matched: rows=907, HR@5=0.617420066152, NDCG@5=0.444374240118, mean_rank=4.44983461963.

## 32. ML-1M Phase 3G integration

ML-1M primary metrics were copied from `results/tables/plan_e_gru4rec_ml1m_pilot_metrics.csv`; no ML-1M GRU4Rec training was rerun in Phase 3H.

## 33. All-dataset GRU4Rec summary

- Amazon Beauty ranking plan_e2_gru4rec_strict_sequential_full: rows=983, HR@5=0.547304170905, NDCG@5=0.387098748547, mean_rank=5.55442522889.
- Amazon Beauty ranking plan_e2_gru4rec_strict_sequential_matched: rows=977, HR@5=0.545547594678, NDCG@5=0.38536365124, mean_rank=5.57523029683.
- Amazon Beauty reranking plan_e2_gru4rec_strict_sequential_full: rows=983, HR@5=0.547304170905, NDCG@5=0.387098748547, mean_rank=5.55442522889.
- Amazon Beauty reranking plan_e2_gru4rec_strict_sequential_matched: rows=977, HR@5=0.545547594678, NDCG@5=0.38536365124, mean_rank=5.57523029683.
- Amazon Sports ranking plan_e2_gru4rec_strict_sequential_full: rows=994, HR@5=0.61569416499, NDCG@5=0.442963541313, mean_rank=4.44668008048.
- Amazon Sports ranking plan_e2_gru4rec_strict_sequential_matched: rows=907, HR@5=0.617420066152, NDCG@5=0.444374240118, mean_rank=4.44983461963.
- Amazon Sports reranking plan_e2_gru4rec_strict_sequential_full: rows=994, HR@5=0.61569416499, NDCG@5=0.442963541313, mean_rank=4.44668008048.
- Amazon Sports reranking plan_e2_gru4rec_strict_sequential_matched: rows=907, HR@5=0.617420066152, NDCG@5=0.444374240118, mean_rank=4.44983461963.
- ML-1M ranking plan_e2_gru4rec_strict_sequential_full: rows=981, HR@5=0.875637104995, NDCG@5=0.713517534186, mean_rank=1.7125382263.
- ML-1M ranking plan_e2_gru4rec_strict_sequential_matched: rows=981, HR@5=0.875637104995, NDCG@5=0.713517534186, mean_rank=1.7125382263.
- ML-1M reranking plan_e2_gru4rec_strict_sequential_full: rows=981, HR@5=0.875637104995, NDCG@5=0.713517534186, mean_rank=1.7125382263.
- ML-1M reranking plan_e2_gru4rec_strict_sequential_matched: rows=981, HR@5=0.875637104995, NDCG@5=0.713517534186, mean_rank=1.7125382263.

## 34. History-length and truncation comparison

- Amazon Beauty: mean raw history=7.51226219321, mean effective history=7.46736381328, truncation rate=0.00408167090279.
- Amazon Sports: mean raw history=7.34370445385, mean effective history=7.33948730975, truncation rate=0.00105428602486.
- ML-1M: mean raw history=179.260958206, mean effective history=44.6931702345, truncation rate=0.700305810398.

## 35. Same-cohort prior-baseline reconstruction

Existing Random, MostPop, BM25-like and BPRMF rankings were read and filtered to exact GRU4Rec strict row keys; no prior baseline was rerun.

## 36. Random versus GRU4Rec

- Amazon Beauty ranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.29501525941, NDCG diff=0.23123403533, GRU4Rec rank win rate=0.662258392675.
- Amazon Beauty ranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.293756397134, NDCG diff=0.229699287535, GRU4Rec rank win rate=0.660184237462.
- Amazon Beauty reranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.3001017294, NDCG diff=0.243376584452, GRU4Rec rank win rate=0.665310274669.
- Amazon Beauty reranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.298874104401, NDCG diff=0.242178354926, GRU4Rec rank win rate=0.665301944729.
- Amazon Sports ranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.376257545272, NDCG diff=0.302122111169, GRU4Rec rank win rate=0.725352112676.
- Amazon Sports ranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.379272326351, NDCG diff=0.304286888709, GRU4Rec rank win rate=0.727673649394.
- Amazon Sports reranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.394366197183, NDCG diff=0.313759423534, GRU4Rec rank win rate=0.746478873239.
- Amazon Sports reranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.396912899669, NDCG diff=0.315055576898, GRU4Rec rank win rate=0.747519294377.
- ML-1M ranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.608562691131, NDCG diff=0.557928086917, GRU4Rec rank win rate=0.850152905199.
- ML-1M ranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.608562691131, NDCG diff=0.557928086917, GRU4Rec rank win rate=0.850152905199.
- ML-1M reranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.636085626911, NDCG diff=0.570733696821, GRU4Rec rank win rate=0.870540265036.
- ML-1M reranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.636085626911, NDCG diff=0.570733696821, GRU4Rec rank win rate=0.870540265036.

## 37. MostPop versus GRU4Rec

- Amazon Beauty ranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.0671414038657, NDCG diff=0.0605870210417, GRU4Rec rank win rate=0.544252288911.
- Amazon Beauty ranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.0665301944729, NDCG diff=0.0589568735259, GRU4Rec rank win rate=0.542476970317.
- Amazon Beauty reranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.0671414038657, NDCG diff=0.0605870210417, GRU4Rec rank win rate=0.544252288911.
- Amazon Beauty reranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.0665301944729, NDCG diff=0.0589568735259, GRU4Rec rank win rate=0.542476970317.
- Amazon Sports ranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.135814889336, NDCG diff=0.0960816232328, GRU4Rec rank win rate=0.590543259557.
- Amazon Sports ranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.135611907387, NDCG diff=0.0988178373685, GRU4Rec rank win rate=0.595369349504.
- Amazon Sports reranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.135814889336, NDCG diff=0.0960816232328, GRU4Rec rank win rate=0.590543259557.
- Amazon Sports reranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.135611907387, NDCG diff=0.0988178373685, GRU4Rec rank win rate=0.595369349504.
- ML-1M ranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.205912334353, NDCG diff=0.239787321454, GRU4Rec rank win rate=0.59123343527.
- ML-1M ranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.205912334353, NDCG diff=0.239787321454, GRU4Rec rank win rate=0.59123343527.
- ML-1M reranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.205912334353, NDCG diff=0.239787321454, GRU4Rec rank win rate=0.59123343527.
- ML-1M reranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.205912334353, NDCG diff=0.239787321454, GRU4Rec rank win rate=0.59123343527.

## 38. BM25-like versus GRU4Rec

- Amazon Beauty ranking plan_e2_gru4rec_strict_sequential_full: HR diff=-0.00915564598169, NDCG diff=-0.0293302598162, GRU4Rec rank win rate=0.426246185148.
- Amazon Beauty ranking plan_e2_gru4rec_strict_sequential_matched: HR diff=-0.0112589559877, NDCG diff=-0.0305521268783, GRU4Rec rank win rate=0.425793244626.
- Amazon Beauty reranking plan_e2_gru4rec_strict_sequential_full: HR diff=-0.00915564598169, NDCG diff=-0.0293302598162, GRU4Rec rank win rate=0.426246185148.
- Amazon Beauty reranking plan_e2_gru4rec_strict_sequential_matched: HR diff=-0.0112589559877, NDCG diff=-0.0305521268783, GRU4Rec rank win rate=0.425793244626.
- Amazon Sports ranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.0995975855131, NDCG diff=0.0684049300195, GRU4Rec rank win rate=0.527162977867.
- Amazon Sports ranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.0915104740904, NDCG diff=0.0611101773701, GRU4Rec rank win rate=0.519294377067.
- Amazon Sports reranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.0995975855131, NDCG diff=0.0684049300195, GRU4Rec rank win rate=0.527162977867.
- Amazon Sports reranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.0915104740904, NDCG diff=0.0611101773701, GRU4Rec rank win rate=0.519294377067.
- ML-1M ranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.354740061162, NDCG diff=0.347521383511, GRU4Rec rank win rate=0.691131498471.
- ML-1M ranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.354740061162, NDCG diff=0.347521383511, GRU4Rec rank win rate=0.691131498471.
- ML-1M reranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.354740061162, NDCG diff=0.347521383511, GRU4Rec rank win rate=0.691131498471.
- ML-1M reranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.354740061162, NDCG diff=0.347521383511, GRU4Rec rank win rate=0.691131498471.

## 39. BPRMF versus GRU4Rec

- Amazon Beauty ranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.0742624618515, NDCG diff=0.0364659723584, GRU4Rec rank win rate=0.427263479145.
- Amazon Beauty ranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.07164790174, NDCG diff=0.0346246348695, GRU4Rec rank win rate=0.425793244626.
- Amazon Beauty reranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.0742624618515, NDCG diff=0.0364659723584, GRU4Rec rank win rate=0.427263479145.
- Amazon Beauty reranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.07164790174, NDCG diff=0.0346246348695, GRU4Rec rank win rate=0.425793244626.
- Amazon Sports ranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.0995975855131, NDCG diff=0.0557703794037, GRU4Rec rank win rate=0.440643863179.
- Amazon Sports ranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.0981256890849, NDCG diff=0.0548109874248, GRU4Rec rank win rate=0.437706725469.
- Amazon Sports reranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.0995975855131, NDCG diff=0.0557703794037, GRU4Rec rank win rate=0.440643863179.
- Amazon Sports reranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.0981256890849, NDCG diff=0.0548109874248, GRU4Rec rank win rate=0.437706725469.
- ML-1M ranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.146788990826, NDCG diff=0.194910053694, GRU4Rec rank win rate=0.539245667686.
- ML-1M ranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.146788990826, NDCG diff=0.194910053694, GRU4Rec rank win rate=0.539245667686.
- ML-1M reranking plan_e2_gru4rec_strict_sequential_full: HR diff=0.146788990826, NDCG diff=0.194910053694, GRU4Rec rank win rate=0.539245667686.
- ML-1M reranking plan_e2_gru4rec_strict_sequential_matched: HR diff=0.146788990826, NDCG diff=0.194910053694, GRU4Rec rank win rate=0.539245667686.

## 40. Dataset-specific findings

- Amazon Beauty: comparison rows=16, mean HR difference=0.106630138142, mean NDCG difference=0.0754995307778.
- Amazon Sports: comparison rows=16, mean HR difference=0.179207826895, mean NDCG difference=0.131575991872.
- ML-1M: comparison rows=16, mean HR difference=0.33244138634, mean NDCG difference=0.336637412632.

## 41. Ranking versus reranking interpretation

Ranking and reranking candidate-set IDs remain separate in all output tables. The Beauty/Sports candidate rows are 20-candidate Plan E2 rows only.

## 42. Paired positive-rank findings

Paired win/equal/loss counts sum to evaluated rows for all `48` comparison rows.

## 43. Candidate and row identity validation

The consolidation helper requires exact row-key equality and exact candidate-list equality before computing comparisons.

## 44. Determinism findings

Determinism checks: `26`; failed checks: `0`.

## 45. Timestamp-order reconstruction limitation

Chronology uses timestamp ascending with original source-row index as the approved deterministic tie rule; this reconstructs a deterministic order but cannot prove the original paper's exact hidden preprocessing order.

## 46. Deviations from the original paper

The datasets, candidates and splits are reconstructed Plan E2 artifacts; the model is a local GRU4Rec implementation; the evaluated strict cohorts exclude rows whose histories or candidates violate the approved sequential policy.

## 47. Limitations

This phase does not train SASRec, LightGCN, BPRMF or ML-1M GRU4Rec; it does not compute beyond-utility metrics; it does not run statistical significance tests.

## 48. Whether Phase 3H succeeded

Phase 3H succeeded if this report, the 12-row all-dataset GRU4Rec metric table, the 60-row same-cohort metric table, the 48-row paired comparison table, and the Beauty/Sports validation and determinism tables pass validation.

## 49. GRU4Rec completion assessment

GRU4Rec now has controlled Plan E2-compatible results for Beauty, Sports and ML-1M across ranking and reranking candidate sets, with full and matched strict cohorts.

## 50. Conditions for beginning SASRec

Begin SASRec only after preserving the Phase 3H artifacts and applying a separate adapter/preflight validation phase.

## 51. Protected-input immutability

Protected inputs were read only by the consolidation runner. Final hash checks should match the recorded hashes in the training manifest and final command log.

## 52. Commands executed

- `git status --short --branch`
- `git branch --show-current`
- `git log -1 --format="%H %s"`
- `.venv\Scripts\python.exe -m pytest tests/test_plan_e_gru4rec_baseline_consolidation.py tests/test_plan_e_gru4rec_training.py tests/test_plan_e_gru4rec_adapter.py tests/test_experiment_cohort_policy.py`
- `.venv\Scripts\python.exe src/experiments/run_plan_e_gru4rec_beauty_sports.py`
- `.venv\Scripts\python.exe src/experiments/build_plan_e_gru4rec_baseline_consolidation.py`

## 53. Recommended single next task

Phase 3I — SASRec Plan E2 adapter and preflight validation.
