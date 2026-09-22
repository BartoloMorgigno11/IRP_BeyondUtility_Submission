# Finalization F4 Metric Resolution And Missing-Evidence Triage

## 1. Executive summary
- Decisione finale: `finalization_f4_complete`.
- Metriche F3 analizzate: `17`.
- Metriche problematiche F3: `calibration; catalog_coverage_at_5; intra_list_diversity_at_5; popularity_bias_at_5; serendipity_at_5`.
- Risultati offline/adapted ricomputati: `96`.
- Evidenze mancanti triaged: `42`.

## 2. Input e fonti
- Fonti principali: tabelle F3, canonical metric contract, canonical metric results, LLM beyond-utility metrics, upstream `llm_metrics.py`.
- Il PDF locale esiste, ma F4 registra un'anomalia per assenza di parser PDF nella `.venv`; nessuna formula viene inventata dal paper.

## 3. Metriche problematiche identificate
- `calibration`: `blocked_missing_definition`.
- `catalog_coverage_at_5`: `resolved_adapted`.
- `intra_list_diversity_at_5`: `blocked_paper_ambiguity`.
- `popularity_bias_at_5`: `blocked_paper_ambiguity`.
- `serendipity_at_5`: `blocked_missing_inputs`.

## 4. Definizioni del paper
- Le definizioni paper non sono machine-extracted in F4; i campi paper restano conservativi e puntano agli artefatti strutturati esistenti.

## 5. Definizioni upstream
- Upstream evidence principale: `original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py` con `calARP`, `calPopREO`, `calItemCoverage`, `calSerendipity`, `calHallucination`, `calMeanShannon`, `calMeanJaccard`.

## 6. Definizioni locali
- Local evidence principale: `src/metrics/canonical_metric_contract.py`, `src/metrics/canonical_metric_computation.py`, `src/evaluation/coverage.py`, `src/evaluation/hallucination.py`, `src/evaluation/position_bias.py`.

## 7. Confronto delle definizioni
- Righe comparison: `238`.

## 8. Metriche risolte
- `catalog_coverage_at_5` e risolta come metrica adapted train-seen catalog coverage, non come exact paper ItemCoverage.

## 9. Metriche ancora bloccate
- `calibration`: No paper, upstream, or local final formula is available in frozen artifacts.
- `intra_list_diversity_at_5`: Exact item representation and distance function are not recoverable from frozen local evidence.
- `popularity_bias_at_5`: Paper terminology is not disambiguated from ARP/average popularity and PopREO fairness evidence.
- `serendipity_at_5`: Upstream calSerendipity exists, but the aligned popularity-baseline hit vector is not frozen.

## 10. Risultati offline ricomputati
- Righe: `96`; tutte da component counts o metric numerator/denominator gia salvati.

## 11. Riconciliazione con risultati precedenti
- Righe riconciliate: `96`.

## 12. Triage delle 42 evidenze mancanti
- Resolved: `1`; ready_for_f5: `17`; conditional: `16`; blocked: `8`.

## 13. Esperimenti candidati per F5
- 1. Machine-verify paper numeric result cells from local PDF
- 2. Freeze and compute adapted ILD feature policy
- 3. Resolve popularity-bias semantics against ARP and PopREO
- 4. Assess LastFM and official processed data feasibility
- 5. Triage prompt-family paper components with cached or approved local execution

## 14. Claim aggiornati
- Claim sicuri dopo F4: `6`; non sicuri: `3`.

## 15. Impatto sulla copertura
- `exact reproduction coverage`: `12.328767` -> `12.328767`.
- `reconstructed-or-better coverage`: `34.246575` -> `34.246575`.
- `maximum defensible coverage`: `68.493151` -> `68.493151`.
- `exact numerical coverage`: `0.0` -> `0.0`.
- `consolidated metric count`: `6` -> `7`.
- `ambiguous metric count`: `5` -> `4`.
- `safe claim count`: `6` -> `6`.
- `missing evidence count`: `42` -> `41`.

## 16. Anomalie
- Anomalie registrate: `2`.

## 17. Test e validazioni
- Validation pass/fail: `25` / `0`.

## 18. Protezione degli input
- Protected hash status: `checked=1144;changed=0`.

## 19. Decisione finale
- `finalization_f4_complete`.

## 20. Prossima fase consigliata
- F5 dovrebbe iniziare da estrazione/validazione delle celle numeriche paper, poi ILD adapted e pop-bias semantic audit.

Classi: exact paper metric richiede equivalenza completa; functional reproduction ricostruisce il comportamento; adapted metric usa una policy locale esplicita; extension-only non e paper-aligned; unresolved metric resta bloccata.
