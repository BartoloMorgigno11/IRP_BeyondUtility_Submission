# Finalization F5D Serendipity

# 1. Executive summary
- Decisione: `finalization_f5d_complete`.
- Serendipity@5 e risolta come metrica adattata: binary relevance times primitive-baseline miss.
- Risultati reportable: `46`.

# 2. Stato iniziale della metrica
- F4 marcava `serendipity_at_5` come blocked_missing_inputs per mancanza del vettore hit MostPop allineato.

# 3. Definizioni trovate
- `paper_metric_name_unextracted_formula` da `paper`: paper exact prose unavailable; upstream defines serendipity via target hit and MostPop hit vectors
- `upstream_hit_minus_pop_hit_intersection` da `upstream`: mean(hit_results - hit_results_pop * hit_results)
- `f4_blocked_missing_inputs` da `canonical`: F4 recognized calSerendipity but lacked aligned popularity-baseline hit vectors
- `f5d_mostpop_train_only_primitive` da `local`: mean(binary_hit_at_5 * (1 - mostpop_train_only_hit_at_5))
- `f5d_random_seeded_primitive_sensitivity` da `local`: mean(binary_hit_at_5 * (1 - random_seeded_hit_at_5))

# 4. Definizione del paper
- Il nome paper e presente, ma la formula exact non e machine-extracted localmente.

# 5. Implementazione upstream
- `calSerendipity(hit_results, hit_results_pop) = mean(hit_results - hit_results_pop * hit_results)`.

# 6. Implementazione locale
- F5D usa `positive_item_id` per relevance e `mostpop_train_only` allineato per unexpectedness.

# 7. Differenza tra novelty, unexpectedness, ILD e serendipity
- Novelty non equivale a serendipity; misura self-information/popolarita.
- Diversity/ILD non equivale a serendipity; misura dissimilarita fra item raccomandati.
- Unexpectedness e una componente rispetto a una reference; da sola non e serendipity.

# 8. Prerequisiti
- Righe prerequisiti: `88`.

# 9. Primitive recommender policy
- Primaria: MostPop train-only locale da `results/tables/traditional_baseline_rankings.csv`.
- Sensitivity: Random seeded; non sostituisce la policy primaria.

# 10. Relevance policy
- Rilevanza binaria: il positivo ground-truth appare nella top-5.

# 11. Unexpectedness policy
- Unexpectedness primaria: `1 - mostpop_train_only_hit_at_5`.

# 12. Policy serendipity congelata
- Formula: `mean(binary_hit_at_5 * (1 - primitive_hit_at_5))`.
- Candidate-set evaluation con un solo positivo puo produrre molti zeri.

# 13. Dataset e prediction source valutabili
- Prediction source inventariate: `127`.
- Valutabili: `126`.
- Dataset bloccato: LastFM.

# 14. Componenti della metrica
- Componenti: relevance_rate, unexpectedness, primitive_list_overlap, final_serendipity.

# 15. Risultati
- Primo valore reportable: `0.14646464646464646` per `Amazon Beauty` `ranking` `Mistral 7B Instruct`.

# 16. Distribuzione per utenti
- Prima quota zero serendipity: `0.85353535353535348`.
- Prima quota con item serendipitous: `0.14646464646464646`.

# 17. Sensitivity analysis
- Righe sensitivity: `126`.

# 18. Confronti validi
- Confronti validi: `24`.

# 19. Riconciliazione
- Righe reconciliation: `15`.

# 20. Trade-off con utility e altre metriche
- Righe tradeoff: `46`. Sono descrittive, non causali.

# 21. Stato scientifico
- Stato: `resolved_adapted`.

# 22. Claim supportati
- `F5D-CLAIM-001`: F5D computes adapted Serendipity@5 with the recovered upstream hit-vector formula; first reportable value is 0.14646464646464646.
- `F5D-CLAIM-002`: Novelty, ILD, unexpectedness, and serendipity are separate; F5D serendipity does not use self-information or pairwise diversity.
- `F5D-CLAIM-003`: The first primary source has zero-serendipity user rate 0.85353535353535348.
- `F5D-CLAIM-004`: Changing the primitive from MostPop to Random is reported as sensitivity only and does not replace the primary policy.
- `F5D-CLAIM-005`: F5D valid comparisons are restricted to the same dataset, task, cohort, candidate set, cutoff, and policy.

# 23. Claim non supportati
- Nessun claim exact paper per serendipity.
- Nessun confronto cross-dataset o mixed-policy.

# 24. Impatto sulla copertura
- `exact reproduction coverage`: `12.328766999999999` -> `12.328766999999999` change `0`.
- `reconstructed-or-better coverage`: `34.246575` -> `34.246575` change `0`.
- `maximum defensible coverage`: `68.493150999999997` -> `68.493150999999997` change `0`.
- `exact numeric replication coverage`: `0` -> `0` change `0`.
- `consolidated metric count`: `11` -> `12` change `1`.
- `ambiguous metric count`: `2` -> `1` change `-1`.
- `missing evidence count`: `39` -> `38` change `-1`.
- `safe report claim count`: `15` -> `20` change `5`.

# 25. Anomalie e limitazioni
- `F5D-ANOM-001` `medium`: Serendipity was previously at risk of being confused with novelty.
- `F5D-ANOM-002` `medium`: Serendipity was previously at risk of being confused with ILD/diversity.
- `F5D-ANOM-003` `medium`: Upstream formula requires a primitive hit vector but the exact paper primitive run is not available.
- `F5D-ANOM-004` `low`: One-positive candidate rows make binary relevance sparse.
- `F5D-ANOM-005` `info`: Upstream serendipity does not document item-feature representation because it does not use one.
- `F5D-ANOM-006` `info`: LastFM lacks a final aligned F5D prediction/primitive policy.
- `F5D-ANOM-007` `info`: Pilot and diagnostic sources are computable but non-primary when flagged.
- `F5D-ANOM-008` `info`: 1 prediction sources are blocked for serendipity.
- `F5D-ANOM-009` `info`: Some rows have no relevant item in top-5.

# 26. Test
- Test F5D e regressioni sicure sono eseguite offline con `PYTHONNOUSERSITE=1`.

# 27. Protezione degli input
- `checked=1144;changed=0`.

# 28. Decisione finale
- `finalization_f5d_complete`.

# 29. Prossimo passo consigliato
- Recuperare la definizione/pipeline exact del paper e il primitive baseline originale se si vuole passare da adapted a exact.
