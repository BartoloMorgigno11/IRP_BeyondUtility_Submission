# Finalization F3 Canonical Scientific Consolidation

## 1. Executive summary
- Decisione finale: `finalization_f3_complete`.
- Evidenze F2 lette: `314`.
- Risultati canonici F3: `314`.
- Risultati primari/reportable: `288`.
- Confronti validi: `228`.
- Coverage exact/reconstructed/max/exact numeric: `12.328767`, `34.246575`, `68.493151`, `0.0`.

## 2. Input utilizzati
- F2 registry resolved to `results/tables/finalization_f2_canonical_result_registry.csv`; the requested alias `results/tables/finalization_f2_canonical_registry.csv` was not the generated filename.
- Input primari: F2 registry, F2 conflicts, Phase 5E coverage, canonical LLM metrics, dataset/cohort summaries and Plan E metric tables.
- Report usati solo come fonti interpretative; i valori numerici provengono da CSV.

## 3. Schema canonico
- `finalization_f3_canonical_scientific_results.csv` conserva una riga per ogni evidenza F2 e aggiunge classe di riproduzione, status artefatto, gruppo comparabile e flag di reportabilita.

## 4. Regole di classificazione
- Exact reproduction richiede equivalenza su dataset, split, coorte, candidate set, modello, prompt, parametri, metrica, cutoff e protocollo.
- In assenza di equivalenza completa, F3 usa classi prudenti: reconstructed, adapted, extension_only, diagnostic_only o legacy_only.

## 5. Regole di comparabilita
- Un gruppo comparabile coincide su dataset, task, coorte, candidate set, candidate count, popolazione, strict/full, metrica, cutoff, prompt/policy e denominator policy.
- MostPop ML-1M Plan B e Plan E2 rimangono gruppi separati.

## 6. Risultati primari
- Primary/reportable rows: `288`.
- Pilot, diagnostic e legacy non sono primary.

## 7. Confronti scientificamente validi
- F3 produce `228` confronti pairwise, tutti entro lo stesso comparable group.

## 8. Matrice riassuntiva
- La matrice primaria deriva solo da `is_primary_result=true` e `is_reportable=true`.

## 9. Allineamento al paper
- Component/result-cell rows: `84`.
- Exact numerical replication rows: `0`.
- F3 non aumenta automaticamente le percentuali Phase 5E.

## 10. Stato delle metriche
- Metriche consolidate: `6`.
- Popularity bias e intra-list diversity restano non consolidate.

## 11. Evidenze mancanti
- Missing evidence rows: `42`.

## 12. Claim scientifici supportati
- Safe claims: `6`.

## 13. Claim non supportati
- Unsafe claims: `3`.

## 14. Risultati esclusi
- Non-reportable rows: `0`.
- Le esclusioni hanno motivazione in `exclusion_reason`.

## 15. Limitazioni
- La replica numerica esatta del paper resta 0/11 celle.
- I modelli LLM locali sono sostituti/adattamenti, non equivalenti completi ai setup originali.

## 16. Validazioni
- Validation pass/fail: `22` / `0`.

## 17. Test
- Test F3 e regressioni selezionate devono essere eseguiti con `.venv/Scripts/python.exe`, `PYTHONNOUSERSITE=1` e senza training/inference.

## 18. Protezione degli input
- Protected input status: `checked=1144;changed=0`.

## 19. Decisione finale
- `finalization_f3_complete`.

## 20. Prossime azioni
- Revisionare le tabelle F3 prima di usarle nella relazione finale o in una presentazione.
- Non dichiarare il progetto una replica completa.
