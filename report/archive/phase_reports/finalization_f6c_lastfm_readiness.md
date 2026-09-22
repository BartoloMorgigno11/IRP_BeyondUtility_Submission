# 1. Executive summary
- Decisione: `finalization_f6c_complete`.
- LastFM overall readiness: `partially_ready`; ranking: `ready_adapted`; reranking: `blocked`.

# 2. Perche LastFM e parte del paper
- F6B classifica LastFM come dataset originale: `results/tables/finalization_f6b_paper_component_inventory.csv`.

# 3. Sorgenti locali
- Sorgenti inventariate: `16`; archivio primario: `data/external/plan_c/hetrec2011-lastfm-2k.zip`.

# 4. Protocollo del paper
- Elementi protocollo inventariati: `24`; timestamp listening: `blocked_exact`.

# 5. Confronto paper/upstream/locale
- Paper/upstream richiedono leave-one-out; F6C materializza una policy adattata, non temporale.

# 6. Audit delle interazioni
- Raw: `92834` interazioni, `1892` utenti, `17632` item.
- Final selected: `71355` interazioni, `1859` utenti, `2823` item.

# 7. Preprocessing
- Policy primaria: iterative 5-core, dedup user-item, weight retained as implicit signal.

# 8. K-core
- Min degree finale user/item: `5/5`.

# 9. Item identity e metadati
- Name coverage finale: `100.000000`; tag coverage finale: `99.645767`.

# 10. Split
- Train/validation/test users: `1859/1859/1859`.
- Split validation pass/fail: `16/0`.

# 11. Coorte
- Ranking cohort rows: `1859`; reranking cohort rows: `1859`.

# 12. Candidate-set protocol
- Ranking candidate sets: `1859`; candidate rows: `37180`; candidate count: `20`.

# 13. Ranking readiness
- `ready_adapted`: inputs are ready for later offline scoring, not paper-exact.

# 14. Reranking readiness
- `blocked`: no traceable LastFM traditional ranking source exists.

# 15. Risultati gia esistenti
- Existing LastFM result inventory rows: `40`; no local prediction-list result is reusable.

# 16. Stato utility
- Utility inputs for ranking are `ready_adapted`; metrics are not computed in F6C.

# 17. Stato beyond-utility
- Catalog and partial tag features are available; prediction-list metrics require later model outputs.

# 18. Missing evidence
- F6B rows audited: `44`; partially resolved by F6C: `2`.

# 19. Coverage impact
- dataset evaluation-ready coverage: `3/4` -> `4/4 adapted local data readiness`.
- task readiness coverage: `ranking/reranking blocked for LastFM` -> `ranking ready_adapted; reranking blocked`.
- LastFM preprocessing coverage: `partial interaction/catalog only` -> `5-core adapted preprocessing materialized`.
- LastFM split coverage: `missing` -> `adapted validation/test split materialized`.
- LastFM ranking-candidate coverage: `missing` -> `20-candidate adapted ranking set materialized`.
- LastFM reranking readiness: `blocked` -> `blocked`.
- local numeric mapping coverage: `44/187` -> `44/187`.
- protocol-exact numeric replication: `0/187` -> `0/187`.
- missing-evidence count: `27 active gaps` -> `2 partially resolved by F6C; exact gaps remain`.

# 20. Piano di esecuzione successivo
- 1. deterministic baseline smoke test.
- 2. Random and MostPop.
- 3. BM25.
- 4. BPRMF.
- 5. GRU4Rec.
- 6. SASRec.
- 7. LightGCN.
- 8. Qwen2.
- 9. Mistral.
- 10. beyond-utility metrics.
- 11. paper-observation verification.

# 21. Anomalie e limitazioni
- Anomalie registrate: `14`; high/critical: `3`.

# 22. Test
- Validazione F6C: `31` pass, `0` fail.

# 23. Protezione degli input
- `checked=1144;changed=0`.

# 24. Decisione finale
- `finalization_f6c_complete`.

# 25. Prossima fase
- F6D: baseline smoke test, Random/MostPop/BM25, poi modelli addestrati e LLM solo con autorizzazione separata.
