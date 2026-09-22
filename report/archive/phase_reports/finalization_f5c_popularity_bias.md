# Finalization F5C Popularity Bias Resolution

## 1. Executive summary
- Decisione: `finalization_f5c_complete`.
- `popularity_bias_at_5` e risolto come alias superseded: non viene prodotto alcun valore numerico con quel nome.
- Metriche numeriche F5C: `7`; righe reportable: `322`.

## 2. Stato di partenza F5B/F5C
- Fonti prediction ereditate da F5B: `127`.
- Fonti computabili ARP/APLT: `126`.
- Nessun file F2/F3/F4/F5A/F5B viene modificato da F5C.

## 3. Problema semantico
- Il nome `popularity_bias_at_5` mescolava livello medio di popolarita, esposizione long-tail, shift rispetto al profilo utente e PopREO.
- F5C vieta l'uso del nome ambiguo come metrica numerica e richiede nomi espliciti.

## 4. Evidenza upstream
- `calARP` misura la media delle frequenze train degli item raccomandati.
- `calACLT` misura la quota di item train-seen sotto la soglia q80.
- `calPopREO` misura dispersione di hit-rate per gruppi di popolarita del target, ma manca la policy per gruppi vuoti e denominatore zero.

## 5. Inventario metriche
- `arp_at_5`: `resolved_functional`.
- `average_recommendation_popularity_at_5`: `resolved_functional`.
- `normalized_average_recommendation_popularity_at_5`: `resolved_extension_only`.
- `aplt_at_5`: `resolved_functional`.
- `head_exposure_at_5`: `resolved_extension_only`.
- `popularity_shift_at_5`: `resolved_extension_only`.
- `popularity_lift_at_5`: `resolved_extension_only`.
- `popreo_at_5`: `blocked_missing_denominator_policy`.
- `popularity_bias_at_5`: `superseded_alias`.

## 6. Policy train-only
- `Amazon Beauty`: `active`, source `data/reconstructed/plan_c/amazon_beauty/interactions.csv`, threshold `17`.
- `Amazon Sports`: `active`, source `data/reconstructed/plan_c/amazon_sports/interactions.csv`, threshold `17`.
- `ML-1M`: `active`, source `data/processed/ml1m/interactions.csv`, threshold `425.40000000000009`.
- `LastFM`: `blocked`, source `data/reconstructed/plan_c/lastfm/interactions.csv`, threshold ``.

## 7. Statistiche item popularity
- `Amazon Beauty`: train-seen `12092`, catalog `259204`, head `2497`, tail `9595`, gini `0.97741230479379793`.
- `Amazon Sports`: train-seen `18347`, catalog `532197`, head `3790`, tail `14557`, gini `0.98308466557917185`.
- `ML-1M`: train-seen `3704`, catalog `3883`, head `741`, tail `2963`, gini `0.65049958276838415`.

## 8. Head, mid e tail
- Tail: `train_seen_count < q80`; head: `train_seen_count >= q80`.
- Mid non e definito dalla funzione upstream `calACLT`; F5C non introduce una soglia arbitraria.

## 9. Fonti recommendation list
- Fonti totali: `127`; fonti primarie: `46`; pilot: `32`; diagnostic: `43`.

## 10. ARP@5
- Primo valore reportable: `15.912727272727274`.
- Formula: media train-count degli item top-5.

## 11. Average Recommendation Popularity@5
- Primo valore reportable: `15.912727272727274`.
- E semanticamente equivalente ad ARP nella policy F5C, ma non e chiamata popularity bias.

## 12. APLT/ACLT@5
- Primo valore reportable: `0.771919191919192`.
- Cold/unseen item non sono long-tail member perche la membership upstream nasce da `freq_dict`.

## 13. Head Exposure@5
- Primo valore reportable: `0.22686868686868689`.
- E una estensione diagnostica locale sotto la stessa soglia q80.

## 14. Popularity shift e lift
- Primo shift reportable: `-27.816617218994367`.
- Primo lift reportable: `0.60290722296671695`.
- Righe summary profilo: `378`.

## 15. PopREO
- PopREO e identificato come target-popularity hit fairness.
- F5C non calcola PopREO perche mancano policy locali per gruppi vuoti e denominatore zero.

## 16. Popularity bias alias
- Stato: `superseded_alias`.
- Sostituito da: `arp_at_5;average_recommendation_popularity_at_5;aplt_at_5;head_exposure_at_5;popularity_shift_at_5;popularity_lift_at_5;popreo_at_5`.

## 17. Confronti validi
- Confronti same dataset/task/cohort/candidate/metric: `168`.

## 18. Riconciliazione
- Righe reconciliation: `147`.
- Valori con coorte/candidate set/denominatore diverso non sono conflitti numerici.

## 19. LastFM
- LastFM resta bloccato per F5C: split `observed`, nessuna policy train-only equivalente e nessuna fonte finale F5B.

## 20. Impatto copertura
- `exact reproduction coverage`: `12.328766999999999` -> `12.328766999999999` change `0`.
- `reconstructed-or-better coverage`: `34.246575` -> `34.246575` change `0`.
- `maximum defensible coverage`: `68.493150999999997` -> `68.493150999999997` change `0`.
- `exact numeric replication coverage`: `0` -> `0` change `0`.
- `consolidated metric count`: `8` -> `11` change `3`.
- `ambiguous metric count`: `3` -> `2` change `-1`.
- `missing evidence count`: `40` -> `39` change `-1`.
- `safe report claim count`: `10` -> `15` change `5`.

## 21. Claim update
- `F5C-CLAIM-001`: Popularity bias is resolved as a superseded alias, not as a numeric metric.
- `F5C-CLAIM-002`: ARP/Average Recommendation Popularity@5 is computable offline for reportable sources across Amazon Beauty;Amazon Sports;ML-1M.
- `F5C-CLAIM-003`: APLT/ACLT@5 is computed with the recovered q80 train-count threshold and reported separately from ARP.
- `F5C-CLAIM-004`: PopREO is identified as target-popularity hit-rate fairness and remains numerically blocked.
- `F5C-CLAIM-005`: Popularity shift and lift are available as local extension diagnostics against each user's train profile.

## 22. Anomalie
- Anomalie registrate: `5`.
- `F5C-ANOM-001` `medium`: The label popularity_bias_at_5 aggregated distinct semantic families.
- `F5C-ANOM-002` `medium`: PopREO formula is recoverable, but empty target-group and zero-denominator handling is not specified.
- `F5C-ANOM-003` `info`: Upstream ACLT defines a binary q80 tail/head split; no mid group is defined.
- `F5C-ANOM-004` `info`: 1 prediction-source groups cannot compute popularity metrics.
- `F5C-ANOM-005` `info`: LastFM has observed interactions but no train split under the F5C policy and no final F5B source.

## 23. Validazione
- Pass: `30`; fail: `0`.

## 24. Input protetti
- `checked=1144;changed=0`.

## 25. Determinismo
- Il builder serializza CSV ordinati e id stabili; la seconda esecuzione deve produrre gli stessi hash.

## 26. Decisione finale
- `finalization_f5c_complete`.

## 27. File prodotti e next action
- `results/tables/finalization_f5c_popularity_metric_inventory.csv`
- `results/tables/finalization_f5c_popularity_definition_comparison.csv`
- `results/tables/finalization_f5c_item_popularity_policy.csv`
- `results/tables/finalization_f5c_item_popularity_statistics.csv`
- `results/tables/finalization_f5c_prediction_source_inventory.csv`
- `results/tables/finalization_f5c_popularity_results.csv`
- `results/tables/finalization_f5c_item_group_exposure.csv`
- `results/tables/finalization_f5c_user_popularity_shift_summary.csv`
- `results/tables/finalization_f5c_valid_popularity_comparisons.csv`
- `results/tables/finalization_f5c_popularity_reconciliation.csv`
- `results/tables/finalization_f5c_popularity_bias_resolution.csv`
- `results/tables/finalization_f5c_metric_status_update.csv`
- `results/tables/finalization_f5c_claim_update.csv`
- `results/tables/finalization_f5c_coverage_impact.csv`
- `results/tables/finalization_f5c_anomalies.csv`
- `results/tables/finalization_f5c_validation.csv`
- `report/finalization_f5c_popularity_bias.md`
- Prossimo passo: citare solo metriche esplicite; recuperare policy PopREO prima di qualunque valore numerico PopREO.
