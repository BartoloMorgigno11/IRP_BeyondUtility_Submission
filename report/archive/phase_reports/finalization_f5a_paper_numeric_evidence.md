# Finalization F5A Paper Numeric Evidence

## 1. Executive summary
- Decisione finale: `finalization_f5a_complete`.
- Celle target: `11`; valori machine-verified: `0`.
- Exact numeric replications: `0`.

## 2. Paper autoritativo
- `paper/2411.00331v1.pdf` sha256 `a143265ae42f027c079766b1c006c438d698897b732634d78a010eb0a62f0f23`.

## 3. Versioni locali del paper
- PDF locali inventariati: `2`; i due file locali sono byte-identical se condividono lo stesso SHA-256.

## 4. Strumenti di estrazione disponibili
- Tool inventariati: `13`; selected parser: `python_stdlib_raw_pdf_metadata_scan` per metadata only.

## 5. Metodo di estrazione
- Le 11 celle sono identificate da `results/tables/plan_d_phase_5e_paper_result_cells.csv`; senza parser tabellare, i valori restano unresolved.

## 6. Le 11 celle numeriche target
- Target IDs: `PRC-0073DE5853; PRC-195BD94B8B; PRC-3B1816C463; PRC-3C2F4B3551; PRC-88936A0830; PRC-941ACB79D8; PRC-97892A414B; PRC-B3B613E7D6; PRC-C9EE65DFB5; PRC-D388B98E99; PRC-E92F01D27A`.

## 7. Valori originali recuperati
- Valori recuperati e machine-verified: `0`.

## 8. Celle non recuperate
- Celle non recuperate: `11`.

## 9. Mapping con i risultati locali
- Celle con local counterpart ID: `6`.

## 10. Verifica dell'equivalenza di protocollo
- Protocol exact/functionally/adapted/incompatible: `0` / `0` / `0` / `6`.

## 11. Confronto numerico
- Printed precision match: `0`; exact raw match: `0`.

## 12. Celle esattamente replicate
- `0`.

## 13. Celle funzionalmente replicate
- `0`.

## 14. Coincidenze numeriche non equivalenti
- `0`.

## 15. Impatto sulla copertura
- `exact numerical result-cell coverage`: `0.0` -> `0.0`.
- `verified_paper_numeric_cell_extraction_coverage`: `0.0` -> `0.0`.
- `mapped local result-cell coverage`: `54.545455` -> `54.545455`.
- `protocol-exact mapped-cell coverage`: `0.0` -> `0.0`.
- `functional numeric replication coverage`: `0.0` -> `0.0`.
- `exact reproduction component coverage`: `12.328767` -> `12.328767`.
- `reconstructed-or-better component coverage`: `34.246575` -> `34.246575`.
- `maximum defensible component coverage`: `68.493151` -> `68.493151`.

## 16. Claim aggiornati
- Claim rows: `12`.

## 17. Anomalie
- Anomalie: `5`.

## 18. Test e validazioni
- Validation pass/fail: `28` / `0`.

## 19. Protezione degli input
- `checked=1144;changed=0`.

## 20. Decisione finale
- `finalization_f5a_complete`.

## 21. Prossimo passo consigliato
- Procedere con estrazione manuale double-entry o approvare un parser PDF locale in F5B.

Recuperare il numero del paper non significa averlo replicato; ottenere lo stesso numero non significa replicare il protocollo; una replica numerica esatta richiede valore verificato e protocollo exact.
