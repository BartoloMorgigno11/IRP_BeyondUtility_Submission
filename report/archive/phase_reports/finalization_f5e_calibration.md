# 1. Executive summary
- Decisione: `finalization_f5e_complete`.
- Calibration e consolidata come metrica locale extension-only, non come componente del paper.

# 2. Stato iniziale
- Dopo F5D restava ambigua solo `calibration`.

# 3. Definizioni trovate
- `paper_calibration_search` da `paper`: no paper calibration formula recovered from local searchable artifacts
- `upstream_absence_audit` da `upstream`: upstream metrics file exposes HR/NDCG/ARP/ACLT/coverage/serendipity/Shannon/Jaccard/hallucination, not calibration
- `f4_blocked_metric` da `canonical`: F4 blocked calibration because no source-backed definition and target distribution were available
- `extension_jsd_profile_distribution` da `local`: JSD(P_train_profile,Q_top5_recommendations), log base 2
- `extension_one_minus_jsd` da `local`: 1 - JSD(P_train_profile,Q_top5_recommendations)
- `extension_hellinger_sensitivity` da `local`: sqrt(0.5 * sum((sqrt(P_i)-sqrt(Q_i))^2))
- `score_probability_calibration_not_applicable` da `unknown`: not applicable: no probabilistic model scores are calibrated against observed probabilities

# 4. Presenza o assenza nel paper
- Paper calibration term recovered: `false`; formula disponibile: `false`.

# 5. Presenza o assenza upstream
- Upstream contains calibration: `false`.

# 6. Recommendation vs probability calibration
- Recommendation calibration confronta distribuzioni di feature profilo/raccomandazioni; probability calibration calibra score/probabilita e resta fuori scope.

# 7. Prerequisiti
- Righe prerequisiti: `40`.

# 8. Feature policy
- F5E riusa le feature primarie F5B: categorie Amazon e generi ML-1M.

# 9. User-profile policy
- Profilo train-only, weighting uniforme, multi-label fractional, nessun recency/rating weighting, nessun test leakage.

# 10. Recommendation-distribution policy
- Top-5, rank weighting uniforme, duplicati mantenuti come esposizioni, invalid/missing feature esclusi e contati.

# 11. Divergence policy
- Primaria: JSD base 2. Sensitivity: Hellinger distance.

# 12. Definizione matematica
- `M = 0.5 * (P + Q)`; `JSD = 0.5*KL(P||M) + 0.5*KL(Q||M)`; `score = 1 - JSD`.

# 13. Dataset e prediction source valutabili
- Prediction source inventariate: `128`.
- Valutabili: `126`.
- Dataset bloccato: LastFM.

# 14. Feature e profile coverage
- Coverage rows: `126`; valid-user rate medio: `0.97280656087532014`.

# 15. Risultati
- Risultati totali: `252`; reportable: `92`.
- Primo valore reportable JSD: `0.48342826968167735` per `Amazon Beauty` `ranking` `Mistral 7B Instruct`.

# 16. Distribuzioni per utenti
- Sono riportati quantili, perfect-alignment rate, high-divergence rate e bucket di lunghezza history.

# 17. Sensitivity analysis
- Righe sensitivity: `126`; Hellinger non sostituisce JSD.

# 18. Confronti validi
- Confronti validi: `24`.

# 19. Riconciliazione
- Righe riconciliazione: `3`.

# 20. Trade-off con le altre metriche
- Trade-off rows: `46`; descrittivi, non causali.

# 21. Stato scientifico
- Stato: `resolved_extension_only`; paper equivalence: `not_paper_metric`.

# 22. Claim supportati
- `F5E-CLAIM-001`: F5E computes extension-only Calibration@5 with JSD; first reportable divergence is 0.48342826968167735.
- `F5E-CLAIM-002`: F5E separates recommendation/profile distribution calibration from score/probability calibration.
- `F5E-CLAIM-003`: Within a same-protocol row, higher calibration score means closer profile-distribution alignment; top observed reportable score is 0.706562407996425.
- `F5E-CLAIM-004`: Hellinger is reported as sensitivity only because its scale differs from JSD.
- `F5E-CLAIM-005`: F5E valid comparisons require the same dataset, task, cohort, candidate set, feature policy and divergence.
- `F5E-CLAIM-006`: F5E reports feature coverage and valid-user rate for every computed source.

# 23. Claim non supportati
- Non e supportato dichiarare calibration come metrica del paper; non sono supportati confronti cross-dataset come se le feature avessero identica semantica.

# 24. Impatto sulla coverage
- exact reproduction coverage: `12.328766999999999 -> 12.328766999999999` (0).
- reconstructed-or-better coverage: `34.246575 -> 34.246575` (0).
- maximum defensible coverage: `68.493150999999997 -> 68.493150999999997` (0).
- exact numeric replication coverage: `0 -> 0` (0).
- consolidated metric count: `12 -> 13` (1).
- ambiguous metric count: `1 -> 0` (-1).
- missing evidence count: `38 -> 37` (-1).
- safe claim count: `20 -> 26` (6).
- extension metric count: `0 -> 1` (1).

# 25. Anomalie e limitazioni
- Anomalie registrate: `14`.

# 26. Test
- Test automatici F5E e regressioni sicure sono previsti dal protocollo di esecuzione.

# 27. Protezione degli input
- `checked=1144;changed=0`.

# 28. Decisione finale
- `finalization_f5e_complete`.

# 29. Prossimo passo consigliato
- Usare F5E come estensione locale citabile solo con policy e coverage; non incrementare la replica del paper.
