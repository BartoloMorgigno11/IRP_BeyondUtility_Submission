# F6L LastFM V4 Reranking Full Run

Decision token: `f6l_lastfm_v4_reranking_stopped_by_systemic_duplicate_regression`.

Description: LastFM sample reranking reproduction under explicit local operationalization.
Classification: `paper_protocol_reproduction_under_explicit_operationalization`.

## Source

- Canonical pool: `results/tables/phase_f6l_lastfm_operationalized_reranking_pool.csv`
- Pool file SHA256: `D42BEB8EDC13070D0FB5CB0E27D4049B7849BC74ED1C12AA446C5B97DE9A9398`
- Pool content hash: `7D7F6718C2E92D9EEEAE2B930E37E3E831EDFA3F1FA0DA019AFA65AF955332CE`
- Candidate definition: 20-item per-user reranking pool reconstructed by local BPRMF/GRU4Rec/SASRec/LightGCN round-robin union, with original candidate order preserved and C01-C20 assigned by candidate_position.
- Positive coverage: `314/1000` users; absent `686`.
- Ranking baseline freeze hash: `D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC`.
- Ranking baseline audit: `results/tables/phase_f6l_lastfm_v4_reranking_full_run_ranking_baseline_audit.csv`.

## Exclusions

- Full-cohort 1859-user reranking pool is excluded.
- F6D3 planned full-cohort reranking manifest is excluded.
- Ranking V4 raw outputs and candidate rows are not used as reranking inputs.
- Ranking-vs-reranking comparisons use frozen fail-strict ranking values only; complete-case ranking is secondary.

## Execution

- Run id: `20260806_034027`
- Raw output dir: `results/llm/phase_f6l_lastfm_v4_reranking_full_run_20260806_034027`
- Planned: `4000`; completed: `2400`; strict-valid: `2352`; structural-invalid: `48`; retry: `0`.
- Parser hash: `5527024992F87984078F46F15491F1E8CB6EB7DF6659559416EBD67F8A098854`
- Schema hash: `266F765149A0C46DAFAB8D4B1C9A0ECAF63986E806E6866F74FA2E75D04321E3`
- Manifest hash: `CA4C12F26558B42143C430235D1F1DC37EDB9547B7DC0FD97ED9947434BDE6B9`

No paper-exact LastFM reranking claim is made.
