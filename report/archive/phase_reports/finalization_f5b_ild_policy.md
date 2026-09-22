# Finalization F5B ILD Policy

## 1. Executive summary
- Decisione: `finalization_f5b_complete`.
- ILD@5 e risolta come `resolved_adapted`, non come replica exact del paper.
- Risultati ILD cosine calcolati: `46` primari; `52` reportable.

## 2. Stato precedente della metrica
- F3/F4 bloccavano `intra_list_diversity_at_5` per rappresentazione item e distanza non congelate.
- F5B congela una policy locale adattata e conserva il blocco exact.

## 3. Fonti delle feature item
- Fonti inventariate: `13`.
- `Amazon Beauty` `brand`: `data/reconstructed/plan_c/amazon_beauty/catalog.csv` coverage `0.4901120353080971`.
- `Amazon Beauty` `category_path`: `data/reconstructed/plan_c/amazon_beauty/catalog.csv` coverage `1`.
- `Amazon Beauty` `description`: `data/reconstructed/plan_c/amazon_beauty/catalog.csv` coverage `0.89607799262357057`.
- `Amazon Beauty` `title`: `data/reconstructed/plan_c/amazon_beauty/catalog.csv` coverage `0.99828706347124274`.
- `Amazon Sports` `brand`: `data/reconstructed/plan_c/amazon_sports/catalog.csv` coverage `0.28385165643549287`.
- `Amazon Sports` `category_path`: `data/reconstructed/plan_c/amazon_sports/catalog.csv` coverage `1`.
- `Amazon Sports` `description`: `data/reconstructed/plan_c/amazon_sports/catalog.csv` coverage `0.74281516055144992`.
- `Amazon Sports` `title`: `data/reconstructed/plan_c/amazon_sports/catalog.csv` coverage `0.99568580807482943`.
- `LastFM` `artist`: `data/reconstructed/plan_c/lastfm/catalog.csv` coverage `1`.
- `LastFM` `category`: `data/reconstructed/plan_c/lastfm/catalog.csv` coverage `1`.
- `ML-1M` `genre`: `data/processed/ml1m/catalog.csv` coverage `1`.
- `ML-1M` `genre`: `data/raw/ml1m/movies.dat` coverage `1`.
- `ML-1M` `category`: `data/reconstructed/plan_c/ml1m/catalog.csv` coverage `1`.

## 4. Copertura dei metadati
- `Amazon Beauty` `F5B-ILD-COSINE-amazon-beauty`: catalog `1`, recommendation `1`, valid-user `0.99994389587073607`.
- `Amazon Beauty` `F5B-ILD-JACCARD-amazon-beauty`: catalog `1`, recommendation `1`, valid-user `0.99994389587073607`.
- `Amazon Sports` `F5B-ILD-COSINE-amazon-sports`: catalog `1`, recommendation `1`, valid-user `0.99982876712328772`.
- `Amazon Sports` `F5B-ILD-JACCARD-amazon-sports`: catalog `1`, recommendation `1`, valid-user `0.99982876712328772`.
- `ML-1M` `F5B-ILD-COSINE-ml-1m`: catalog `1`, recommendation `1`, valid-user `0.99780000000000002`.
- `ML-1M` `F5B-ILD-JACCARD-ml-1m`: catalog `1`, recommendation `1`, valid-user `0.99780000000000002`.

## 5. Policy primaria congelata
- Multi-hot categorical feature tokens; cosine similarity; dissimilarity `1 - cosine_similarity`; top-k 5.
- Coppie non ordinate tra item utilizzabili; macro-media sugli utenti validi.
- Duplicati mantenuti; coppia dello stesso item con dissimilarita zero.
- Item invalidi o senza feature sono esclusi dalle coppie e conteggiati.

## 6. Policy di sensibilita
- Stessi token, stesse regole, Jaccard similarity; non sostituisce la policy primaria.

## 7. Definizione matematica
- `ILD@k(u) = 2 / (m * (m - 1)) * sum_{a < b}(1 - sim(i_a, i_b))`, con `m >= 2` item utilizzabili.
- `ILD@k` e la media sugli utenti validi.

## 8. Fonti delle recommendation list
- Fonti/gruppi inventariati: `127`; computabili: `84`; primari: `46`.

## 9. Dataset e modelli valutabili
- `Amazon Beauty` `ranking` `Mistral 7B Instruct` da `results/tables/canonical_local_llm_full_matrix_rankings.csv`.
- `Amazon Beauty` `ranking` `Qwen2 7B Instruct` da `results/tables/canonical_local_llm_full_matrix_rankings.csv`.
- `Amazon Beauty` `reranking` `Mistral 7B Instruct` da `results/tables/canonical_local_llm_full_matrix_rankings.csv`.
- `Amazon Beauty` `reranking` `Qwen2 7B Instruct` da `results/tables/canonical_local_llm_full_matrix_rankings.csv`.
- `Amazon Sports` `ranking` `Mistral 7B Instruct` da `results/tables/canonical_local_llm_full_matrix_rankings.csv`.
- `Amazon Sports` `ranking` `Qwen2 7B Instruct` da `results/tables/canonical_local_llm_full_matrix_rankings.csv`.
- `Amazon Sports` `reranking` `Mistral 7B Instruct` da `results/tables/canonical_local_llm_full_matrix_rankings.csv`.
- `Amazon Sports` `reranking` `Qwen2 7B Instruct` da `results/tables/canonical_local_llm_full_matrix_rankings.csv`.
- `ML-1M` `ranking` `Mistral 7B Instruct` da `results/tables/canonical_local_llm_full_matrix_rankings.csv`.
- `ML-1M` `ranking` `Qwen2 7B Instruct` da `results/tables/canonical_local_llm_full_matrix_rankings.csv`.
- `ML-1M` `reranking` `Mistral 7B Instruct` da `results/tables/canonical_local_llm_full_matrix_rankings.csv`.
- `ML-1M` `reranking` `Qwen2 7B Instruct` da `results/tables/canonical_local_llm_full_matrix_rankings.csv`.
- `Amazon Beauty` `reranking` `Mistral 7B Instruct` da `results/tables/canonical_local_llm_hybrid_holdout_rankings.csv`.
- `Amazon Beauty` `reranking` `Qwen2 7B Instruct` da `results/tables/canonical_local_llm_hybrid_holdout_rankings.csv`.
- `Amazon Sports` `reranking` `Mistral 7B Instruct` da `results/tables/canonical_local_llm_hybrid_holdout_rankings.csv`.
- `Amazon Sports` `reranking` `Qwen2 7B Instruct` da `results/tables/canonical_local_llm_hybrid_holdout_rankings.csv`.
- `ML-1M` `reranking` `Mistral 7B Instruct` da `results/tables/canonical_local_llm_hybrid_holdout_rankings.csv`.
- `ML-1M` `reranking` `Qwen2 7B Instruct` da `results/tables/canonical_local_llm_hybrid_holdout_rankings.csv`.
- `Amazon Beauty` `ranking` `Mistral 7B Instruct` da `results/tables/canonical_local_llm_pilot_resume_rankings.csv`.
- `Amazon Beauty` `ranking` `Qwen2 7B Instruct` da `results/tables/canonical_local_llm_pilot_resume_rankings.csv`.
- `Amazon Beauty` `reranking` `Mistral 7B Instruct` da `results/tables/canonical_local_llm_pilot_resume_rankings.csv`.
- `Amazon Beauty` `reranking` `Qwen2 7B Instruct` da `results/tables/canonical_local_llm_pilot_resume_rankings.csv`.
- `Amazon Sports` `ranking` `Mistral 7B Instruct` da `results/tables/canonical_local_llm_pilot_resume_rankings.csv`.
- `Amazon Sports` `ranking` `Qwen2 7B Instruct` da `results/tables/canonical_local_llm_pilot_resume_rankings.csv`.
- `Amazon Sports` `reranking` `Mistral 7B Instruct` da `results/tables/canonical_local_llm_pilot_resume_rankings.csv`.
- `Amazon Sports` `reranking` `Qwen2 7B Instruct` da `results/tables/canonical_local_llm_pilot_resume_rankings.csv`.
- `ML-1M` `ranking` `Mistral 7B Instruct` da `results/tables/canonical_local_llm_pilot_resume_rankings.csv`.
- `ML-1M` `ranking` `Qwen2 7B Instruct` da `results/tables/canonical_local_llm_pilot_resume_rankings.csv`.
- `ML-1M` `reranking` `Mistral 7B Instruct` da `results/tables/canonical_local_llm_pilot_resume_rankings.csv`.
- `ML-1M` `reranking` `Qwen2 7B Instruct` da `results/tables/canonical_local_llm_pilot_resume_rankings.csv`.

## 10. Risultati ILD@5
- `Amazon Beauty` `ranking` `Mistral 7B Instruct`: ILD `0.65701326982218666`, valid users `990/990`, reportable `true`.
- `Amazon Beauty` `ranking` `Qwen2 7B Instruct`: ILD `0.66021094847713913`, valid users `990/990`, reportable `true`.
- `Amazon Beauty` `ranking` `bm25_like_e3b`: ILD `0.6344527939526996`, valid users `1000/1000`, reportable `true`.
- `Amazon Beauty` `ranking` `bprmf_plan_e_beauty_strict_train_known`: ILD `0.6622896074350082`, valid users `983/983`, reportable `true`.
- `Amazon Beauty` `ranking` `gru4rec_reconstructed_beauty_strict_sequential`: ILD `0.65934486096750311`, valid users `983/983`, reportable `true`.
- `Amazon Beauty` `ranking` `lightgcn_reconstructed_beauty_strict_connected_cpu`: ILD `0.64511333123487291`, valid users `983/983`, reportable `true`.
- `Amazon Beauty` `ranking` `mostpop_train_only`: ILD `0.66831857142459783`, valid users `1000/1000`, reportable `true`.
- `Amazon Beauty` `ranking` `random_seeded`: ILD `0.66628459388352124`, valid users `1000/1000`, reportable `true`.
- `Amazon Beauty` `ranking` `sasrec_reconstructed_beauty_strict_causal`: ILD `0.65492005725072766`, valid users `983/983`, reportable `true`.
- `Amazon Beauty` `reranking` `Mistral 7B Instruct`: ILD `0.66348929228473952`, valid users `10/10`, reportable `true`.
- `Amazon Beauty` `reranking` `Mistral 7B Instruct`: ILD `0.66423291051830713`, valid users `990/990`, reportable `true`.
- `Amazon Beauty` `reranking` `Qwen2 7B Instruct`: ILD `0.70300131679396993`, valid users `4/10`, reportable `true`.
- `Amazon Beauty` `reranking` `Qwen2 7B Instruct`: ILD `0.66439677649705642`, valid users `989/990`, reportable `true`.
- `Amazon Beauty` `reranking` `bm25_like_e3b`: ILD `0.6344527939526996`, valid users `1000/1000`, reportable `true`.
- `Amazon Beauty` `reranking` `bprmf_plan_e_beauty_strict_train_known`: ILD `0.6622896074350082`, valid users `983/983`, reportable `true`.
- `Amazon Beauty` `reranking` `gru4rec_reconstructed_beauty_strict_sequential`: ILD `0.65934486096750311`, valid users `983/983`, reportable `true`.
- `Amazon Beauty` `reranking` `lightgcn_reconstructed_beauty_strict_connected_cpu`: ILD `0.64511333123487291`, valid users `983/983`, reportable `true`.
- `Amazon Beauty` `reranking` `mostpop_train_only`: ILD `0.66831857142459783`, valid users `1000/1000`, reportable `true`.
- `Amazon Beauty` `reranking` `random_seeded`: ILD `0.66784780189814619`, valid users `1000/1000`, reportable `true`.
- `Amazon Beauty` `reranking` `sasrec_reconstructed_beauty_strict_causal`: ILD `0.65492005725072766`, valid users `983/983`, reportable `true`.
- `Amazon Sports` `ranking` `Mistral 7B Instruct`: ILD `0.69316872222249526`, valid users `892/892`, reportable `true`.
- `Amazon Sports` `ranking` `Qwen2 7B Instruct`: ILD `0.6888144968069817`, valid users `892/892`, reportable `true`.
- `Amazon Sports` `ranking` `bm25_like_e3b`: ILD `0.65883692857975729`, valid users `1000/1000`, reportable `true`.
- `Amazon Sports` `ranking` `bprmf_plan_e_sports_strict_train_known`: ILD `0.64645141829450292`, valid users `994/994`, reportable `true`.
- `Amazon Sports` `ranking` `gru4rec_reconstructed_sports_strict_sequential`: ILD `0.62992739567729883`, valid users `994/994`, reportable `true`.
- `Amazon Sports` `ranking` `lightgcn_reconstructed_sports_strict_connected_cpu`: ILD `0.59295263984474011`, valid users `994/994`, reportable `true`.
- `Amazon Sports` `ranking` `mostpop_train_only`: ILD `0.68054725106208891`, valid users `1000/1000`, reportable `true`.
- `Amazon Sports` `ranking` `random_seeded`: ILD `0.6932151105164166`, valid users `1000/1000`, reportable `true`.
- `Amazon Sports` `ranking` `sasrec_reconstructed_sports_strict_causal`: ILD `0.61726042485938715`, valid users `994/994`, reportable `true`.
- `Amazon Sports` `reranking` `Mistral 7B Instruct`: ILD `0.66097958112282262`, valid users `9/10`, reportable `true`.
- `Amazon Sports` `reranking` `Mistral 7B Instruct`: ILD `0.68518127881697333`, valid users `892/892`, reportable `true`.
- `Amazon Sports` `reranking` `Qwen2 7B Instruct`: ILD `0.64593081058744095`, valid users `5/10`, reportable `true`.
- `Amazon Sports` `reranking` `Qwen2 7B Instruct`: ILD `0.68980994113078709`, valid users `889/892`, reportable `true`.
- `Amazon Sports` `reranking` `bm25_like_e3b`: ILD `0.65883692857975729`, valid users `1000/1000`, reportable `true`.
- `Amazon Sports` `reranking` `bprmf_plan_e_sports_strict_train_known`: ILD `0.64645141829450292`, valid users `994/994`, reportable `true`.
- `Amazon Sports` `reranking` `gru4rec_reconstructed_sports_strict_sequential`: ILD `0.62992739567729883`, valid users `994/994`, reportable `true`.
- `Amazon Sports` `reranking` `lightgcn_reconstructed_sports_strict_connected_cpu`: ILD `0.59295263984474011`, valid users `994/994`, reportable `true`.
- `Amazon Sports` `reranking` `mostpop_train_only`: ILD `0.68054725106208891`, valid users `1000/1000`, reportable `true`.
- `Amazon Sports` `reranking` `random_seeded`: ILD `0.69452096115507034`, valid users `1000/1000`, reportable `true`.
- `Amazon Sports` `reranking` `sasrec_reconstructed_sports_strict_causal`: ILD `0.61726042485938715`, valid users `994/994`, reportable `true`.

## 11. Copertura delle feature
- Ogni risultato riporta feature coverage delle raccomandazioni e valid-user rate; i valori cross-dataset non sono direttamente comparabili perche la rappresentazione item cambia.

## 12. Confronti validi
- Confronti validi prodotti: `27`.

## 13. Sensitivity analysis
- Righe sensitivity: `84`. Jaccard e diagnostica; la policy primaria resta cosine.

## 14. Riconciliazione con valori precedenti
- Righe di riconciliazione: `84`; i valori precedenti erano bloccati o non protocol-equivalent.

## 15. Stato scientifico della metrica
- Stato F5B: `resolved_adapted`; paper equivalence `adapted_not_exact`.

## 16. Claim supportati
- `F3-CLAIM-005`: The high local structural validity evidence depends on the indexed JSON prompt/parser contract.
- `F3-CLAIM-009`: Local reranking utility is generally lower than local ranking utility for Qwen2 and Mistral in the generated evidence.
- `F5B-CLAIM-001`: ILD@5 is now resolved only as an adapted local metric under the frozen F5B categorical-feature cosine policy.
- `F5B-CLAIM-002`: ILD model comparisons are valid only inside the same dataset, task, cohort, candidate set, top-k, feature policy and missing/duplicate policy.
- `F5B-CLAIM-003`: The primary feature policy has measurable catalog and recommendation feature coverage for the computable local datasets.
- `F5B-CLAIM-004`: Jaccard sensitivity is a diagnostic robustness view and does not replace the primary cosine ILD result.

## 17. Claim non supportati
- Non e supportato alcun claim di replica exact paper per ILD.
- Non e supportato un confronto diretto tra dataset diversi.

## 18. Impatto sulla copertura
- `exact reproduction coverage`: `12.328767` -> `12.328767` change `0`.
- `reconstructed-or-better coverage`: `34.246575` -> `34.246575` change `0`.
- `maximum defensible coverage`: `68.493151` -> `68.493151` change `0`.
- `exact numeric replication coverage`: `0.0` -> `0.0` change `0`.
- `consolidated metric count`: `7` -> `8` change `1`.
- `ambiguous metric count`: `4` -> `3` change `-1`.
- `missing evidence count`: `41` -> `40` change `-1`.
- `safe claim count`: `6` -> `10` change `4`.

## 19. Anomalie e limitazioni
- Anomalie registrate: `56`.
- `F5B-ANOM-001` `medium`: Exact paper ILD representation and distance function remain unrecovered.
- `F5B-ANOM-002` `medium`: LastFM has local artist metadata but no final saved recommendation lists or structured tag policy.
- `F5B-ANOM-003` `info`: Previous ILD rows are blocked or blank rather than numeric.
- `F5B-ANOM-079d92299657` `info`: brand coverage is sparse and less semantically central than category path
- `F5B-ANOM-170b77d23588` `low`: derived duplicate canonical metric input; source-specific ranking files are used to avoid double counting
- `F5B-ANOM-33daea14ff3a` `low`: derived duplicate canonical metric input; source-specific ranking files are used to avoid double counting
- `F5B-ANOM-3e402590e06e` `low`: derived duplicate canonical metric input; source-specific ranking files are used to avoid double counting
- `F5B-ANOM-44caad04e019` `low`: derived duplicate canonical metric input; source-specific ranking files are used to avoid double counting
- `F5B-ANOM-4a86408df448` `low`: derived duplicate canonical metric input; source-specific ranking files are used to avoid double counting
- `F5B-ANOM-5191826a620f` `info`: free-text fallback excluded because structured categorical metadata is available
- `F5B-ANOM-52de57181362` `low`: derived duplicate canonical metric input; source-specific ranking files are used to avoid double counting
- `F5B-ANOM-5453f2056447` `low`: derived duplicate canonical metric input; source-specific ranking files are used to avoid double counting

## 20. Test
- Test F5B: `tests/test_finalization_f5b_ild_policy.py`.
- Regressioni sicure F3/F4/F5A-FIX/canonical metrics/coverage/candidate validation da eseguire offline.

## 21. Protezione degli input
- Input protetti: `checked=1144;changed=0`.

## 22. Decisione finale
- `finalization_f5b_complete`.

## 23. Prossimo passo consigliato
- Usare F5B come evidence adattata locale; recuperare la feature policy esatta del paper solo con evidenza sorgente separata.
