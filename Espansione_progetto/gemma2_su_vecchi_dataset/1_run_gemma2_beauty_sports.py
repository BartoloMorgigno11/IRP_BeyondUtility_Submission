"""Esegue Gemma 2 9B (via Ollama) come recommender su Beauty e/o Sports.

Questo script:
1. Verifica che i dati processati esistano (altrimenti chiede di eseguire lo step 0)
2. Chiama Gemma 2 tramite Ollama per ogni utente
3. Salva raccomandazioni e metriche

Esegui dalla root del progetto:

    python Espansione_progetto/gemma2_su_vecchi_dataset/1_run_gemma2_beauty_sports.py \\
        --dataset beauty --allow-api-calls

    python Espansione_progetto/gemma2_su_vecchi_dataset/1_run_gemma2_beauty_sports.py \\
        --dataset sports --allow-api-calls

    python Espansione_progetto/gemma2_su_vecchi_dataset/1_run_gemma2_beauty_sports.py \\
        --dataset both --allow-api-calls

REQUISITI:
  - Ollama installato e avviato:  ollama serve
  - Modello scaricato:            ollama pull gemma2:9b
  - Virtual environment attivo:  source .venv/bin/activate
  - Dati processati presenti:    python .../0_prepara_beauty_sports.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = logging.getLogger("run_gemma2_beauty_sports")

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma2:9b"
TOP_K = 5
HISTORY_LIMIT = 10
TIMEOUT_SECONDS = 120.0

OUTPUT_COLUMNS = [
    "user_id", "model_name", "dataset", "rank",
    "recommended_item_id", "recommended_item_title",
    "run_id", "raw_output", "parse_status", "latency_seconds",
    "timestamp",
]

METRICS_COLUMNS = [
    "run_id", "dataset", "task", "model_name",
    "metric", "k", "value", "config_path", "timestamp",
]

# Dataset configs: percorsi relativi alla root del progetto
DATASET_CONFIGS = {
    "beauty": {
        "processed_dir": "data/processed/beauty",
        "dataset_label": "Amazon Beauty",
        "metrics_file": "results/metrics/beauty_gemma2_metrics.csv",
        "recs_file": "results/raw_outputs/beauty_gemma2_recommendations.csv",
    },
    "sports": {
        "processed_dir": "data/processed/sports",
        "dataset_label": "Amazon Sports",
        "metrics_file": "results/metrics/sports_gemma2_metrics.csv",
        "recs_file": "results/raw_outputs/sports_gemma2_recommendations.csv",
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Esegui Gemma 2 via Ollama su Beauty e/o Sports."
    )
    parser.add_argument(
        "--dataset", choices=["beauty", "sports", "both"],
        default="both",
        help="Dataset su cui eseguire (default: both)",
    )
    parser.add_argument(
        "--allow-api-calls", action="store_true",
        help="Flag obbligatorio per abilitare le chiamate LLM (sicurezza)",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Modello Ollama (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Limita a N utenti per testing veloce",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Riprende da dove si era fermato (skip utenti già processati)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def _ollama_generate(model: str, prompt: str, timeout: float = TIMEOUT_SECONDS) -> dict[str, Any]:
    """Invia una richiesta al server Ollama locale."""
    import urllib.request

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except Exception as exc:
        return {"ok": False, "error": str(exc), "latency": time.perf_counter() - started, "text": ""}

    latency = time.perf_counter() - started
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return {"ok": False, "error": "invalid JSON from Ollama", "latency": latency, "text": ""}

    text = parsed.get("response", "")
    if not isinstance(text, str):
        return {"ok": False, "error": "missing response field", "latency": latency, "text": ""}
    return {"ok": True, "error": None, "latency": latency, "text": text.strip()}


def check_ollama(model: str) -> None:
    """Verifica che Ollama sia avviato e il modello disponibile."""
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:11434/api/version", timeout=5) as r:
            version = json.loads(r.read())
        LOGGER.info("Ollama version: %s", version.get("version", "?"))
    except Exception as exc:
        raise RuntimeError(
            f"Ollama non raggiungibile a {OLLAMA_URL}.\n"
            "Avvia Ollama: ollama serve"
        ) from exc

    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
            tags = json.loads(r.read())
        available = [m["name"] for m in tags.get("models", [])]
        model_base = model.split(":")[0]
        if not any(model_base in m for m in available):
            raise RuntimeError(
                f"Modello '{model}' non trovato in Ollama.\n"
                f"Disponibili: {available}\n"
                f"Esegui: ollama pull {model}"
            )
    except RuntimeError:
        raise
    except Exception as exc:
        LOGGER.warning("Impossibile verificare la lista modelli: %s — continuo.", exc)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    user_id: str, history: list[str], candidates: list[tuple[str, str]],
    top_k: int, domain_label: str
) -> str:
    history_text = "\n".join(f"- {t}" for t in history) if history else "- (no history available)"
    candidate_text = "\n".join(f"- {iid}: {title}" for iid, title in candidates)
    placeholders = ", ".join(f'"item_id_{i}"' for i in range(1, top_k + 1))
    schema = '{\n  "ranked_item_ids": [' + placeholders + '],\n  "status": "ok"\n}'

    return f"""You are evaluating candidate items for a {domain_label} recommender system.
Use the user's purchase/interaction history and the candidate list to rank the top {top_k} items for user `{user_id}`.

User history (chronological order, most recent last):
{history_text}

Candidate items (ID: Title):
{candidate_text}

Instructions:
- Return only valid JSON, nothing else.
- Rank exactly {top_k} item IDs from the candidate list above.
- Use ONLY item IDs that appear in the candidate list.
- Do NOT invent or hallucinate item IDs.
- Do NOT add explanations.

Required JSON schema:
{schema}
"""


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

def _parse_response(text: str, valid_ids: set[str], top_k: int) -> tuple[list[str], str]:
    try:
        data = json.loads(text)
        ranked = data.get("ranked_item_ids", [])
        if isinstance(ranked, list):
            valid = [str(r) for r in ranked if str(r) in valid_ids]
            if len(valid) >= 1:
                return valid[:top_k], "exact"
    except (json.JSONDecodeError, AttributeError):
        pass

    found = [cid for cid in valid_ids if cid in text]
    if found:
        return found[:top_k], "partial"

    return [], "failed"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(
    recs_df: pd.DataFrame, ground_truth_df: pd.DataFrame,
    catalog_size: int, top_k: int,
    dataset_name: str, model_name: str, run_id: str, config_path: str,
) -> pd.DataFrame:
    gt_map = dict(zip(ground_truth_df["user_id"], ground_truth_df["ground_truth_item_id"]))
    users = recs_df["user_id"].unique()
    hits, ndcgs, unique_recs = [], [], set()

    for user_id in users:
        user_recs = recs_df[recs_df["user_id"] == user_id].sort_values("rank")
        gt_item = gt_map.get(user_id)
        rec_ids = user_recs["recommended_item_id"].tolist()[:top_k]

        hit = int(gt_item in rec_ids) if gt_item else 0
        hits.append(hit)

        ndcg = 0.0
        if gt_item and gt_item in rec_ids:
            rank = rec_ids.index(gt_item) + 1
            ndcg = 1.0 / math.log2(rank + 1)
        ndcgs.append(ndcg)
        unique_recs.update(rec_ids)

    hr = sum(hits) / len(hits) if hits else 0.0
    ndcg_val = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    coverage = len(unique_recs) / catalog_size if catalog_size > 0 else 0.0

    failed_rows = recs_df[
        recs_df["recommended_item_id"].isna() |
        (recs_df["recommended_item_id"] == "")
    ]
    total_slots = len(users) * top_k
    hallucination = len(failed_rows) / total_slots if total_slots > 0 else 0.0

    now = datetime.now(timezone.utc).isoformat()
    rows = [
        {"run_id": run_id, "dataset": dataset_name, "task": "ranking", "model_name": model_name,
         "metric": "hr", "k": top_k, "value": hr, "config_path": config_path, "timestamp": now},
        {"run_id": run_id, "dataset": dataset_name, "task": "ranking", "model_name": model_name,
         "metric": "ndcg", "k": top_k, "value": ndcg_val, "config_path": config_path, "timestamp": now},
        {"run_id": run_id, "dataset": dataset_name, "task": "ranking", "model_name": model_name,
         "metric": "catalog_coverage", "k": top_k, "value": coverage, "config_path": config_path, "timestamp": now},
        {"run_id": run_id, "dataset": dataset_name, "task": "ranking", "model_name": model_name,
         "metric": "hallucination_rate", "k": top_k, "value": hallucination, "config_path": config_path, "timestamp": now},
    ]
    return pd.DataFrame(rows, columns=METRICS_COLUMNS)


# ---------------------------------------------------------------------------
# Checkpoint write
# ---------------------------------------------------------------------------

def _checkpoint_write(rows: list[dict[str, Any]], path: Path, already_done: set[str]) -> None:
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if path.is_file():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Single dataset runner
# ---------------------------------------------------------------------------

def run_gemma2_on_dataset(
    dataset_name: str,
    model: str,
    sample_size: int | None,
    resume: bool,
) -> None:
    cfg = DATASET_CONFIGS[dataset_name]
    processed_dir = PROJECT_ROOT / cfg["processed_dir"]
    recs_path = PROJECT_ROOT / cfg["recs_file"]
    metrics_path = PROJECT_ROOT / cfg["metrics_file"]
    domain_label = cfg["dataset_label"]

    # Verifica file processati
    required_files = [
        processed_dir / "interactions.csv",
        processed_dir / "candidates.csv",
        processed_dir / "ground_truth.csv",
        processed_dir / "catalog.csv",
    ]
    for p in required_files:
        if not p.is_file():
            LOGGER.error("File non trovato: %s", p)
            LOGGER.error("Esegui prima: python Espansione_progetto/gemma2_su_vecchi_dataset/0_prepara_beauty_sports.py --datasets %s", dataset_name)
            return

    # Crea cartelle output
    recs_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = f"{dataset_name}_gemma2_{model.replace(':', '_').replace('.', '_')}"

    LOGGER.info("=== Gemma 2 — %s ===", domain_label)
    LOGGER.info("Model:       %s", model)
    LOGGER.info("Sample:      %s", sample_size or "all")
    LOGGER.info("Output recs: %s", recs_path)

    # Carica dati
    interactions = pd.read_csv(processed_dir / "interactions.csv")
    candidates   = pd.read_csv(processed_dir / "candidates.csv")
    ground_truth = pd.read_csv(processed_dir / "ground_truth.csv")
    catalog      = pd.read_csv(processed_dir / "catalog.csv")
    catalog_size = len(catalog)

    title_map = dict(zip(catalog["item_id"].astype(str), catalog["title"].astype(str)))

    # Costruisce history per utente
    train_interactions = interactions[interactions["split"] == "train"].copy()
    train_interactions = train_interactions.sort_values(["user_id", "timestamp"])
    user_history: dict[str, list[str]] = (
        train_interactions.groupby("user_id")["item_id"]
        .apply(lambda x: [title_map.get(str(i), str(i)) for i in x.tolist()[-HISTORY_LIMIT:]])
        .to_dict()
    )

    # Utenti da processare
    eligible_users = sorted(ground_truth["user_id"].unique())
    if sample_size is not None and sample_size < len(eligible_users):
        eligible_users = eligible_users[:sample_size]

    # Resume: skip già processati
    already_done: set[str] = set()
    if resume and recs_path.is_file():
        existing = pd.read_csv(recs_path)
        already_done = set(existing["user_id"].unique())
        LOGGER.info("Resume: %d utenti già processati, skip.", len(already_done))
        eligible_users = [u for u in eligible_users if u not in already_done]

    LOGGER.info("Utenti da processare: %d", len(eligible_users))
    if not eligible_users:
        LOGGER.info("Nessun utente da processare. Esiting.")
        return

    candidates["item_id_str"] = candidates["candidate_item_id"].astype(str)
    user_candidates = candidates.groupby("user_id")

    all_rows: list[dict[str, Any]] = []
    total = len(eligible_users)
    failed_users = 0

    for i, user_id in enumerate(eligible_users, 1):
        if i % 10 == 0 or i == 1 or i == total:
            LOGGER.info("[%d/%d] Processing user %s ...", i, total, user_id)

        if user_id not in user_candidates.groups:
            LOGGER.warning("User %s: nessun candidate — skip.", user_id)
            failed_users += 1
            continue

        user_cand_df = user_candidates.get_group(user_id).sort_values("candidate_position")
        cand_pairs = [
            (str(row["candidate_item_id"]), str(row["candidate_item_title"]))
            for _, row in user_cand_df.iterrows()
        ]
        valid_ids = {pair[0] for pair in cand_pairs}
        history = user_history.get(str(user_id), [])

        prompt = _build_prompt(str(user_id), history, cand_pairs, TOP_K, domain_label)
        result = _ollama_generate(model, prompt, timeout=TIMEOUT_SECONDS)
        now = datetime.now(timezone.utc).isoformat()

        if not result["ok"]:
            LOGGER.warning("User %s: Ollama error — %s", user_id, result["error"])
            failed_users += 1
            all_rows.append({
                "user_id": user_id, "model_name": "gemma2", "dataset": dataset_name,
                "rank": None, "recommended_item_id": None, "recommended_item_title": None,
                "run_id": run_id, "raw_output": result["error"],
                "parse_status": "ollama_error", "latency_seconds": result["latency"],
                "timestamp": now,
            })
            continue

        ranked_ids, parse_status = _parse_response(result["text"], valid_ids, TOP_K)

        if not ranked_ids:
            LOGGER.warning("User %s: parse fallito. Raw: %s", user_id, result["text"][:200])
            failed_users += 1
            all_rows.append({
                "user_id": user_id, "model_name": "gemma2", "dataset": dataset_name,
                "rank": None, "recommended_item_id": None, "recommended_item_title": None,
                "run_id": run_id, "raw_output": result["text"][:500],
                "parse_status": "failed", "latency_seconds": result["latency"],
                "timestamp": now,
            })
            continue

        for rank, item_id in enumerate(ranked_ids, 1):
            all_rows.append({
                "user_id": user_id,
                "model_name": "gemma2",
                "dataset": dataset_name,
                "rank": rank,
                "recommended_item_id": item_id,
                "recommended_item_title": title_map.get(item_id, item_id),
                "run_id": run_id,
                "raw_output": result["text"][:200] if rank == 1 else "",
                "parse_status": parse_status,
                "latency_seconds": result["latency"] if rank == 1 else 0.0,
                "timestamp": now,
            })

        # Checkpoint ogni 10 utenti
        if i % 10 == 0:
            _checkpoint_write(all_rows, recs_path, already_done)
            all_rows = []

    if all_rows:
        _checkpoint_write(all_rows, recs_path, already_done)

    LOGGER.info("Inferenza completata. Utenti falliti: %d / %d", failed_users, total)

    # Calcola metriche
    if not recs_path.is_file():
        LOGGER.error("File raccomandazioni non trovato — metriche saltate.")
        return

    recs_df = pd.read_csv(recs_path)
    recs_df = recs_df.dropna(subset=["recommended_item_id"])

    if recs_df.empty:
        LOGGER.error("Nessuna raccomandazione valida — metriche saltate.")
        return

    LOGGER.info("Calcolando metriche su %d righe ...", len(recs_df))
    metrics_df = _compute_metrics(
        recs_df, ground_truth, catalog_size, TOP_K,
        dataset_name=dataset_name, model_name="gemma2",
        run_id=run_id, config_path=str(dataset_name),
    )
    metrics_df.to_csv(metrics_path, index=False)
    LOGGER.info("Metriche salvate in %s", metrics_path)

    print("\n" + "=" * 60)
    print(f"  Gemma 2 — {domain_label} — Results")
    print("=" * 60)
    for _, row in metrics_df.iterrows():
        print(f"  {row['metric']:25s} @{row['k']} = {row['value']:.4f}")
    print("=" * 60)
    print(f"\n  Raccomandazioni: {recs_path}")
    print(f"  Metriche:        {metrics_path}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)

    if not args.allow_api_calls:
        print("ERRORE: il flag --allow-api-calls è obbligatorio per eseguire chiamate LLM.")
        print("       È un meccanismo di sicurezza per evitare chiamate accidentali.")
        sys.exit(1)

    model = args.model
    sample_size = args.sample_size

    # Verifica Ollama (una sola volta)
    check_ollama(model)

    # Determina quali dataset eseguire
    datasets_to_run = ["beauty", "sports"] if args.dataset == "both" else [args.dataset]

    for ds in datasets_to_run:
        run_gemma2_on_dataset(
            dataset_name=ds,
            model=model,
            sample_size=sample_size,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
