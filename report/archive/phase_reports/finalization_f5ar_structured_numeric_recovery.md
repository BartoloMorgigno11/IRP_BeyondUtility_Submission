# Finalization F5A-R Structured Numeric Recovery

## 1. Executive summary
- Decisione: `finalization_f5ar_complete`.
- DOCX: `paper/traduzione_italiana_beyond_utility_llm_recommender.docx`; valid ZIP: `true`; document.xml: `true`.
- Il DOCX e' trattato solo come fonte secondaria strutturata; il PDF resta autoritativo.

## 2. Metodo DOCX
- Parsing via `zipfile` e `xml.etree.ElementTree`, leggendo WordprocessingML da `word/document.xml` o relazione officeDocument.

## 3. Inventario tabelle
- Tabelle DOCX trovate: `6`; celle tabella estratte: `289`.

## 4. Mapping delle 11 celle F5A
- Celle target: `11`; con candidato: `3`; univoche: `0`.

## 5. Recovery secondaria
- Nessuna cella e' dichiarata PDF-verified sul solo DOCX.

## 6. Pacchetto manuale PDF
- Compilare opzionalmente `results/manual/finalization_f5ar_pdf_confirmations.csv` con colonne: `paper_numeric_cell_id,pdf_value_confirmed,confirmed_pdf_value_text,confirmed_by,confirmation_date,review_notes`.
- Una conferma e' ignorata se ID, valore numerico, revisore o data non sono validi.

## 7. Celle da controllare nel PDF
- `PRC-0073DE5853`: table ``, row ``, column ``, DOCX value ``. Controllare visivamente nel PDF autoritativo tabella, riga, colonna e valore stampato.
- `PRC-195BD94B8B`: table `3`, row `LLMRank (Llama3-70b)`, column `Beauty NDCG`, DOCX value `0,3954`. Controllare visivamente nel PDF autoritativo tabella, riga, colonna e valore stampato.
- `PRC-3B1816C463`: table `5`, row `Popularity ↑`, column `SASRec`, DOCX value `0,0259`. Controllare visivamente nel PDF autoritativo tabella, riga, colonna e valore stampato.
- `PRC-3C2F4B3551`: table ``, row ``, column ``, DOCX value ``. Controllare visivamente nel PDF autoritativo tabella, riga, colonna e valore stampato.
- `PRC-88936A0830`: table ``, row ``, column ``, DOCX value ``. Controllare visivamente nel PDF autoritativo tabella, riga, colonna e valore stampato.
- `PRC-941ACB79D8`: table `3`, row `SASRec`, column `Beauty HR`, DOCX value `0,647`. Controllare visivamente nel PDF autoritativo tabella, riga, colonna e valore stampato.
- `PRC-97892A414B`: table ``, row ``, column ``, DOCX value ``. Controllare visivamente nel PDF autoritativo tabella, riga, colonna e valore stampato.
- `PRC-B3B613E7D6`: table ``, row ``, column ``, DOCX value ``. Controllare visivamente nel PDF autoritativo tabella, riga, colonna e valore stampato.
- `PRC-C9EE65DFB5`: table ``, row ``, column ``, DOCX value ``. Controllare visivamente nel PDF autoritativo tabella, riga, colonna e valore stampato.
- `PRC-D388B98E99`: table ``, row ``, column ``, DOCX value ``. Controllare visivamente nel PDF autoritativo tabella, riga, colonna e valore stampato.
- `PRC-E92F01D27A`: table ``, row ``, column ``, DOCX value ``. Controllare visivamente nel PDF autoritativo tabella, riga, colonna e valore stampato.

## 8. Confronto locale preliminare
- Righe confronto preliminare: `11`; claim di replica permessi: `0`.

## 9. Coverage
- `secondary_structured_recovery_coverage`: `0.0` -> `0.0`.
- `pending_manual_review_coverage`: `0.0` -> `100.000000`.
- `verified_paper_numeric_cell_extraction_coverage`: `0.0` -> `0.0`.
- `exact_numerical_coverage`: `0.0` -> `0.0`.
- `functional_numerical_coverage`: `0.0` -> `0.0`.

## 10. Anomalie
- Anomalie registrate: `3`.

## 11. Validazione
- Validation pass/fail: `18` / `0`.

## 12. Decisione finale
- `finalization_f5ar_complete`.

Il recupero del numero dal DOCX non e' una verifica PDF; una replica numerica richiede PDF confermato e protocollo equivalente.
