#!/usr/bin/env bash
# =============================================================================
#  run_all_gemma2.sh — Esegui Gemma 2 su Beauty e Sports
# =============================================================================
#
# Esegue in sequenza:
#   1. Preparazione dataset Beauty e Sports (da file JSON raw)
#   2. Gemma 2 9B su Beauty
#   3. Gemma 2 9B su Sports
#
# USO (dalla ROOT del progetto):
#   bash Espansione_progetto/gemma2_su_vecchi_dataset/run_all_gemma2.sh
#
# PREREQUISITI:
#   - Ollama avviato: ollama serve (in un altro terminale)
#   - gemma2:9b scaricato: ollama pull gemma2:9b
#   - Virtual environment attivo: source .venv/bin/activate
#   - File raw JSON in data/raw/beauty/ e data/raw/sports/
#     (vedi ISTRUZIONI.md se non ce li hai)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo ""
echo "============================================================"
echo "  Gemma 2 — Beauty e Sports"
echo "============================================================"
echo "  Root: $PROJECT_ROOT"
echo ""

cd "$PROJECT_ROOT"

# --------------------------------------------------------------------------
# STEP 1: Prepara i dataset
# --------------------------------------------------------------------------
echo "[STEP 1/3] Preparazione dataset Beauty e Sports ..."
python Espansione_progetto/gemma2_su_vecchi_dataset/0_prepara_beauty_sports.py \
    --datasets both
echo "  OK — dataset pronti."
echo ""

# --------------------------------------------------------------------------
# STEP 2: Gemma 2 su Beauty
# --------------------------------------------------------------------------
echo "[STEP 2/3] Gemma 2 su Amazon Beauty ..."
python Espansione_progetto/gemma2_su_vecchi_dataset/1_run_gemma2_beauty_sports.py \
    --dataset beauty \
    --allow-api-calls
echo "  OK — Beauty completato."
echo ""

# --------------------------------------------------------------------------
# STEP 3: Gemma 2 su Sports
# --------------------------------------------------------------------------
echo "[STEP 3/3] Gemma 2 su Amazon Sports ..."
python Espansione_progetto/gemma2_su_vecchi_dataset/1_run_gemma2_beauty_sports.py \
    --dataset sports \
    --allow-api-calls
echo "  OK — Sports completato."
echo ""

echo "============================================================"
echo "  Tutti gli esperimenti completati!"
echo "  Metriche:"
echo "    results/metrics/beauty_gemma2_metrics.csv"
echo "    results/metrics/sports_gemma2_metrics.csv"
echo "============================================================"
echo ""
