# Finalization F5A-FIX Atomic Paper Cells

## 1. Executive summary
- Decisione: `finalization_f5af_complete`.
- Vecchi target superseded: `11`; model identity: `5`; celle atomiche: `187`.

## 2. Perche gli 11 target precedenti non erano celle
- Cinque righe erano componenti di inventario modello, quindi non risultati numerici.
- Sei righe erano famiglie metriche aggregate: ogni metrica attraversa molte intersezioni tabella/riga/colonna.

## 3. Componenti e celle atomiche
- Paper component: elemento del paper da coprire; model inventory: presenza di un modello; metric family: insieme di celle; atomic numeric result cell: singola intersezione riga-colonna.

## 4. Target superseded
- Tutti gli `11` target precedenti hanno status `superseded_non_atomic_numeric_inventory`.

## 5. Inventario atomico
- Tabelle incluse: 3, 4, 5. Celle atomiche reali trovate: `187`.
- PDF-verified: `0`; secondary-source verified: `187`.

## 6. Mapping locale
- Celle con mapping locale: `44`.
- Protocol-exact: `0`.

## 7. Denominatore corretto
- La copertura numerica precedente `0/11` non era ben definita perche il denominatore conteneva model identity e metric families, non celle atomiche.

## 8. Copertura corretta
- `paper model inventory coverage`: `5/5` = `100.000000`.
- `atomic paper value extraction coverage`: `187/187` = `100.000000`.
- `atomic local mapping coverage`: `44/187` = `23.529412`.
- `protocol-exact mapping coverage`: `0/187` = `0.0`.
- `printed numeric match coverage`: `0/187` = `0.0`.
- `exact numeric replication coverage`: `0/187` = `0.0`.

## 9. Anomalie
- Anomalie registrate: `5`.

## 10. Conservazione audit storici
- F5A e F5A-R restano immutati come audit storici; F5A-FIX produce solo un inventario correttivo separato.

## 11. Validazione
- Validation pass/fail: `12` / `0`.

## 12. Decisione finale
- `finalization_f5af_complete`.
