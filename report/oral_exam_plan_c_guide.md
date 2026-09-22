# Guida Orale Plan C

## Versione Breve: 2 Minuti

Il progetto parte dal paper *Beyond Utility: Evaluating LLM as Recommender* e
prova a capire quanto sia realmente riproducibile.

La risposta onesta e' divisa in tre parti. Primo: la riproduzione esatta su
Beauty e Sports e' bloccata, perche' mancano gli artefatti preprocessati
ufficiali: `sequential_data.txt`, `negative_samples.txt`, `datamaps.json` e il
`meta.json` preprocessato del paper. I link Dropbox ufficiali non sono
recuperabili nell'ambiente attuale.

Secondo: invece di inventare risultati, ho costruito una pipeline di
ricostruzione da dataset pubblici: Amazon Beauty, Amazon Sports, MovieLens-1M e
LastFM. Beauty, Sports e ML-1M sono stati convertiti in file canonici con split
leave-one-out e candidate set. LastFM resta extension-only perche' non ha
timestamp adatti allo split cronologico.

Terzo: ho eseguito baseline deterministiche, Random, MostPop e ContentBM25,
ottenendo metriche reali sui dataset ricostruiti. Queste metriche non sono la
riproduzione esatta del paper, ma una valutazione ricostruita e documentata.

## Versione Media: 5 Minuti

Il progetto ha tre piani.

Plan A e' l'audit di riproducibilita'. Qui il punto centrale e' che il progetto
non forza numeri falsi: se mancano ground truth, candidate set e metadati
ufficiali, le metriche originali Beauty/Sports non vengono calcolate.

Plan B dimostra una pipeline end-to-end su ML-1M come estensione separata.
Serve a mostrare che il codice di valutazione funziona su un dataset stabile e
ricostruibile.

Plan C e' il tentativo piu' ambizioso: ho creato una matrice di replica,
manifest degli artefatti, controlli sulle sorgenti, validazione dei dataset raw,
conversione raw-to-canonical e baseline evaluation.

La parte metodologicamente piu' importante e' la generazione dei candidati. In
una prima versione, il negative sampling poteva usare l'intero catalogo
metadata. Per Amazon questo e' rischioso, perche' il catalogo e' molto piu'
grande degli item osservati nelle interazioni. Molti negativi avrebbero
popolarita' zero, rendendo MostPop troppo forte. Ho quindi corretto il default:
i negativi sono campionati dagli item osservati nelle interazioni, escludendo
tutta la history dell'utente.

Risultato: le metriche Amazon MostPop sono diventate piu' plausibili. Per
esempio HR@5 passa da valori sospetti sopra 0.89/0.94 a circa 0.187 per Beauty
e 0.211 per Sports nel protocollo ricostruito.

## Versione Dettagliata: 10 Minuti

1. Obiettivo: verificare se il paper e' riproducibile e, se non lo e', costruire
   una ricostruzione scientificamente onesta.
2. Audit degli artefatti: ho controllato repository originale, report locali,
   link, output salvati e file attesi.
3. Blocco principale: gli artefatti preprocessati ufficiali Beauty/Sports non
   sono disponibili.
4. Recupero dati raw: Amazon Beauty/Sports 5-core, ML-1M e LastFM HetRec sono
   stati validati.
5. Ricostruzione: Beauty, Sports e ML-1M sono stati convertiti in
   `interactions.csv`, `ground_truth.csv`, `candidates.csv`, `catalog.csv` e
   artefatti paper-like marcati come ricostruiti.
6. LastFM: resta limitato a interazioni/catalogo, perche' manca un timestamp
   utile allo split cronologico.
7. Baseline: Random come lower bound, MostPop come baseline di popolarita',
   ContentBM25 come baseline content-based su titolo/generi/testo catalogo.
8. Metriche: HR@5, NDCG@5 e Catalog Coverage@5.
9. Correzione negative sampling: default `interaction_items`, non catalogo
   intero.
10. Conclusione: non e' una riproduzione esatta del paper, ma una pipeline di
    riproducibilita' e ricostruzione completa, con limiti espliciti.

## Domande Probabili e Risposte Ideali

**Domanda: Perche' non avete semplicemente riprodotto i numeri del paper?**

Risposta: perche' mancano gli artefatti preprocessati ufficiali necessari. Senza
ground truth, candidate set, datamaps e metadata preprocessati, riprodurre i
numeri sarebbe inventare risultati.

**Domanda: Allora il progetto e' fallito?**

Risposta: no. Il risultato scientifico e' che la riproduzione esatta e'
bloccata, e il blocco e' documentato. In piu' ho costruito una ricostruzione
controllata da dati raw pubblici.

**Domanda: Perche' MovieLens-1M?**

Risposta: e' stabile, pubblico, contiene rating con timestamp e metadati film,
quindi permette split cronologico, candidate generation e metriche reali.

**Domanda: Come giustifichi la ricostruzione da dati raw pubblici?**

Risposta: la giustifico come ricostruzione, non come riproduzione esatta. Il
report separa chiaramente official processed artifacts, raw public data e
reconstructed artifacts.

**Domanda: Che problema c'era nel negative sampling?**

Risposta: campionare negativi dall'intero catalogo Amazon puo' inserire molti
item mai osservati nelle interazioni. Questi item hanno popolarita' zero, quindi
MostPop puo' sembrare artificialmente forte. Il default corretto usa solo item
osservati nelle interazioni.

**Domanda: Perche' LastFM e' extension-only?**

Risposta: il file `user_artists.dat` contiene interazioni utente-artista con
peso, ma non timestamp di ascolto. Senza timestamp non posso fare lo stesso
leave-one-out cronologico. Quindi non genero metriche HR/NDCG finte.

**Domanda: Perche' non avete eseguito gli LLM/API?**

Risposta: per vincoli di costo, accesso, drift dei modelli e riproducibilita'.
Prima bisogna avere dataset e candidati validi. Inoltre le API richiederebbero
un consenso esplicito e non garantirebbero lo stesso modello/versione del paper.

## Collegamenti ai Temi del Corso

- Recommender systems: ranking top-N, candidate set, leave-one-out.
- Metriche: HR@5, NDCG@5, coverage.
- Beyond accuracy: il paper motiva metriche oltre la sola utilita'.
- Riproducibilita': audit di artefatti, source ledger, manifest, blocker.
- Dataset: differenza tra raw data, processed artifacts e reconstruction.
- Popularity baseline: MostPop come baseline forte ma sensibile al negative
  sampling.
- Content-based recommendation: ContentBM25 usa testo del catalogo e history
  dell'utente.
- Etica e trasparenza: non inventare metriche, separare claim esatti e claim
  ricostruiti, dichiarare limitazioni.

## Frase Finale Consigliata

"Il contributo principale non e' dire che abbiamo riprodotto tutto, ma mostrare
con precisione cosa e' riproducibile, cosa e' bloccato, e come costruire una
ricostruzione controllata senza falsificare i risultati."
