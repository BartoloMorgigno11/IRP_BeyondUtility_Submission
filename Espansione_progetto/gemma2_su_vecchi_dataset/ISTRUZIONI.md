# ISTRUZIONI — Gemma 2 su Amazon Beauty e Sports

> Questi script testano **Gemma 2 9B** (il nuovo modello) sui dataset
> Amazon Beauty e Amazon Sports, già usati nel paper originale.

---

## Prerequisiti

- **Ollama già installato** ✅ (lo hai già)
- **gemma2:9b già scaricato** — verifica con `ollama list`. Se non c'è: `ollama pull gemma2:9b`
- **Virtual environment attivo**: `source .venv/bin/activate`

---

## ⚠️ File dati necessari

Questi dataset richiedono i file JSON grezzi di Amazon Reviews 2018.

**Controlla se li hai già:**
```bash
ls data/raw/beauty/
# Deve mostrare: reviews_Beauty.json  meta_Beauty.json

ls data/raw/sports/
# Deve mostrare: reviews_Sports.json  meta_Sports.json
```

**Se non li hai**, scaricali da questi link (scegli "5-core" e "metadata"):
- Beauty reviews: https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/AMAZON_FASHION.json.gz
  - Nota: cercare specificamente "All_Beauty" o "Beauty" 5-core su [questa pagina](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
- In alternativa, **contatta Vito** — lui li può fornire direttamente.

Una volta scaricati, mettili in:
```
data/raw/beauty/reviews_Beauty.json
data/raw/beauty/meta_Beauty.json
data/raw/sports/reviews_Sports.json
data/raw/sports/meta_Sports.json
```

---

## OPZIONE A — Esegui tutto con un comando

```bash
cd /percorso/a/IRP_BeyondUtility_Project
source .venv/bin/activate

# Terminale 1: avvia Ollama
ollama serve

# Terminale 2: esegui tutto
bash Espansione_progetto/gemma2_su_vecchi_dataset/run_all_gemma2.sh
```

---

## OPZIONE B — Esegui passo per passo

### Step 1 — Prepara i dataset (non richiede Ollama)

```bash
python Espansione_progetto/gemma2_su_vecchi_dataset/0_prepara_beauty_sports.py \
    --datasets both
```

> Questo può richiedere **5-15 minuti** perché legge file JSON da ~500 MB ciascuno.
> Genera i file in `data/processed/beauty/` e `data/processed/sports/`.

Per solo uno dei due:
```bash
python Espansione_progetto/gemma2_su_vecchi_dataset/0_prepara_beauty_sports.py \
    --datasets beauty
```

---

### Step 2 — Esegui Gemma 2 su Beauty

```bash
python Espansione_progetto/gemma2_su_vecchi_dataset/1_run_gemma2_beauty_sports.py \
    --dataset beauty \
    --allow-api-calls
```

**Resume** (se viene interrotto):
```bash
python Espansione_progetto/gemma2_su_vecchi_dataset/1_run_gemma2_beauty_sports.py \
    --dataset beauty \
    --allow-api-calls \
    --resume
```

---

### Step 3 — Esegui Gemma 2 su Sports

```bash
python Espansione_progetto/gemma2_su_vecchi_dataset/1_run_gemma2_beauty_sports.py \
    --dataset sports \
    --allow-api-calls
```

---

## File generati

| File | Contenuto |
|---|---|
| `data/processed/beauty/` | Dataset Beauty processato (4 CSV) |
| `data/processed/sports/` | Dataset Sports processato (4 CSV) |
| `results/raw_outputs/beauty_gemma2_recommendations.csv` | Raccomandazioni Beauty |
| `results/raw_outputs/sports_gemma2_recommendations.csv` | Raccomandazioni Sports |
| `results/metrics/beauty_gemma2_metrics.csv` | HR@5, NDCG@5, Coverage, Hallucination su Beauty |
| `results/metrics/sports_gemma2_metrics.csv` | HR@5, NDCG@5, Coverage, Hallucination su Sports |

---

## Tempi stimati

| Fase | Tempo stimato |
|---|---|
| Preparazione dati (entrambi) | 10-20 min (lettura JSON ~1 GB) |
| Gemma 2 su Beauty (500 utenti, GPU) | 30-50 min |
| Gemma 2 su Sports (500 utenti, GPU) | 30-50 min |

---

## Troubleshooting

### "File non trovato: data/raw/beauty/..."
→ Devi procurarti i file raw JSON. Vedi la sezione "File dati necessari" sopra.

### "Ollama non raggiungibile"
→ Avvia Ollama: `ollama serve` (in un terminale separato)

### "Modello 'gemma2:9b' non trovato"
→ Scarica: `ollama pull gemma2:9b`

### "No module named 'pandas'"
→ Attiva il venv: `source .venv/bin/activate`

### Lo script viene interrotto a metà
→ Riavvia con `--resume`, riprende da dove si era fermato.
