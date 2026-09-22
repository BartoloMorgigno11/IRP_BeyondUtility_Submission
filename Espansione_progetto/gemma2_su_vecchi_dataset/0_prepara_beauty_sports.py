"""Prepara i dataset Amazon Beauty e Sports per l'esperimento con Gemma 2.

Legge i file JSON grezzi di Amazon Reviews 2018 (formato Amazon Reviews)
e produce i quattro file canonici CSV:
  - interactions.csv
  - catalog.csv
  - candidates.csv
  - ground_truth.csv

Esegui dalla root del progetto:

    python Espansione_progetto/gemma2_su_vecchi_dataset/0_prepara_beauty_sports.py

PREREQUISITI — File raw necessari:
  data/raw/beauty/reviews_Beauty.json       (~0.5 GB)
  data/raw/beauty/meta_Beauty.json          (~0.1 GB)
  data/raw/sports/reviews_Sports.json       (~0.5 GB)
  data/raw/sports/meta_Sports.json          (~0.1 GB)

Se non hai questi file, vedi ISTRUZIONI.md per il download.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = logging.getLogger("prepara_beauty_sports")

INTERACTIONS_COLUMNS = ["user_id", "item_id", "split", "timestamp", "rating"]
CATALOG_COLUMNS = ["item_id", "title", "categories", "normalized_text"]
CANDIDATES_COLUMNS = [
    "user_id", "candidate_item_id", "candidate_position",
    "is_ground_truth", "candidate_item_title",
]
GROUND_TRUTH_COLUMNS = ["user_id", "ground_truth_item_id", "dataset", "split"]

MIN_INTERACTIONS = 5
NEGATIVES_PER_USER = 99
SEED = 42
SAMPLE_SIZE = 500  # Per un run veloce; imposta None per il dataset completo


# ---------------------------------------------------------------------------
# Readers for Amazon Reviews JSON lines
# ---------------------------------------------------------------------------

def _clean_text(value: Any) -> str:
    if not value:
        return ""
    text = str(value).strip()
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return text


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r" +", " ", text).strip()
    return text


def _read_reviews(path: Path) -> pd.DataFrame:
    LOGGER.info("Leggendo reviews da %s ...", path)
    rows: list[dict[str, Any]] = []
    skipped = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            reviewer_id = record.get("reviewerID", "")
            asin = record.get("asin", "")
            overall = record.get("overall")
            ts = record.get("unixReviewTime")
            if not reviewer_id or not asin or overall is None or ts is None:
                skipped += 1
                continue
            rows.append({
                "user_id": str(reviewer_id).strip(),
                "item_id": str(asin).strip(),
                "rating": float(overall),
                "timestamp": int(ts),
            })
    LOGGER.info("  Caricati %d reviews (%d skipped).", len(rows), skipped)
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def _read_metadata(path: Path) -> pd.DataFrame:
    LOGGER.info("Leggendo metadata da %s ...", path)
    rows: list[dict[str, Any]] = []
    skipped = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            asin = record.get("asin", "")
            if not asin:
                skipped += 1
                continue
            title = _clean_text(record.get("title", ""))
            cats_raw = record.get("category", record.get("categories", []))
            if isinstance(cats_raw, list):
                categories = " | ".join(str(c).strip() for c in cats_raw if c)
            else:
                categories = str(cats_raw).strip()
            normalized = _normalize_text(" ".join([title, categories]))
            rows.append({
                "item_id": str(asin).strip(),
                "title": title,
                "categories": categories,
                "normalized_text": normalized,
            })
    LOGGER.info("  Caricati %d catalog entries (%d skipped).", len(rows), skipped)
    df = pd.DataFrame(rows, columns=CATALOG_COLUMNS)
    return df.drop_duplicates("item_id", keep="first").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _apply_kcore(df: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    prev_len = -1
    iteration = 0
    while len(df) != prev_len:
        prev_len = len(df)
        iteration += 1
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index
        df = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]
    LOGGER.info("  k-core (%d iter): %d interazioni, %d utenti, %d item",
                iteration, len(df), df["user_id"].nunique(), df["item_id"].nunique())
    return df.reset_index(drop=True)


def _chronological_leave_one_out(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    last_idx = df.groupby("user_id").tail(1).index
    test_mask = df.index.isin(last_idx)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()
    train_df["split"] = "train"
    test_df["split"] = "test"
    return train_df, test_df


def _build_candidates(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    dataset_name: str,
    negatives_per_user: int = NEGATIVES_PER_USER,
    seed: int = SEED,
    sample_size: int | None = SAMPLE_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    all_item_ids = catalog_df["item_id"].tolist()
    title_map = dict(zip(catalog_df["item_id"], catalog_df["title"]))
    user_history: dict[str, set[str]] = (
        train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    )
    eligible_users = sorted(test_df["user_id"].unique())

    if sample_size is not None and sample_size < len(eligible_users):
        rng_sample = np.random.default_rng(seed)
        idx = rng_sample.choice(len(eligible_users), size=sample_size, replace=False)
        eligible_users = [eligible_users[i] for i in sorted(idx)]

    LOGGER.info("  Costruendo candidates per %d utenti ...", len(eligible_users))
    candidate_rows: list[dict[str, Any]] = []
    ground_truth_rows: list[dict[str, Any]] = []
    skipped = 0

    for user_id in eligible_users:
        gt_rows = test_df[test_df["user_id"] == user_id]
        if gt_rows.empty:
            skipped += 1
            continue
        gt_item = str(gt_rows.iloc[0]["item_id"])
        seen = user_history.get(user_id, set()) | {gt_item}
        negatives_pool = [iid for iid in all_item_ids if iid not in seen]
        if len(negatives_pool) < negatives_per_user:
            skipped += 1
            continue
        neg_indices = rng.choice(len(negatives_pool), size=negatives_per_user, replace=False)
        neg_items = [negatives_pool[i] for i in neg_indices]
        candidates = [gt_item] + neg_items
        shuffle_order = rng.permutation(len(candidates))
        candidates = [candidates[i] for i in shuffle_order]

        for pos, cand_item in enumerate(candidates):
            candidate_rows.append({
                "user_id": user_id,
                "candidate_item_id": cand_item,
                "candidate_position": pos,
                "is_ground_truth": (cand_item == gt_item),
                "candidate_item_title": title_map.get(cand_item, cand_item),
            })
        ground_truth_rows.append({
            "user_id": user_id,
            "ground_truth_item_id": gt_item,
            "dataset": dataset_name,
            "split": "test",
        })

    LOGGER.info("  Skipped: %d utenti | OK: %d utenti", skipped, len(ground_truth_rows))
    return (
        pd.DataFrame(candidate_rows, columns=CANDIDATES_COLUMNS),
        pd.DataFrame(ground_truth_rows, columns=GROUND_TRUTH_COLUMNS),
    )


# ---------------------------------------------------------------------------
# Main preparation for one dataset
# ---------------------------------------------------------------------------

def prepare_dataset(
    dataset_name: str,
    reviews_path: Path,
    metadata_path: Path,
    output_dir: Path,
    sample_size: int | None = SAMPLE_SIZE,
) -> None:
    """Prepara i 4 file canonici per un dataset Amazon Reviews."""
    LOGGER.info("=== Preparazione dataset: %s ===", dataset_name)

    if not reviews_path.is_file():
        LOGGER.error("File reviews non trovato: %s", reviews_path)
        LOGGER.error("Vedi ISTRUZIONI.md per il download.")
        return
    if not metadata_path.is_file():
        LOGGER.error("File metadata non trovato: %s", metadata_path)
        LOGGER.error("Vedi ISTRUZIONI.md per il download.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Leggi raw data
    reviews_df = _read_reviews(reviews_path)
    catalog_raw = _read_metadata(metadata_path)

    # 2. k-core filtering
    reviews_df = _apply_kcore(reviews_df, MIN_INTERACTIONS)

    # 3. Catalogo
    interaction_items = set(reviews_df["item_id"].unique())
    catalog_df = catalog_raw[catalog_raw["item_id"].isin(interaction_items)].copy()
    missing_items = interaction_items - set(catalog_df["item_id"])
    if missing_items:
        LOGGER.warning("%d item senza metadata — uso item_id come titolo.", len(missing_items))
        missing_rows = pd.DataFrame({
            "item_id": list(missing_items),
            "title": list(missing_items),
            "categories": "",
            "normalized_text": "",
        })
        catalog_df = pd.concat([catalog_df, missing_rows], ignore_index=True)
    catalog_df = catalog_df.reset_index(drop=True)

    # 4. Split temporale leave-one-out
    train_df, test_df = _chronological_leave_one_out(reviews_df)

    # 5. Interactions.csv
    interactions_df = pd.concat([
        train_df[["user_id", "item_id", "timestamp", "rating"]].assign(split="train"),
        test_df[["user_id", "item_id", "timestamp", "rating"]].assign(split="test"),
    ], ignore_index=True)[INTERACTIONS_COLUMNS]

    # 6. Candidates e ground truth
    candidates_df, ground_truth_df = _build_candidates(
        test_df, train_df, catalog_df, dataset_name,
        sample_size=sample_size,
    )

    # 7. Salva
    for df, filename in [
        (interactions_df, "interactions.csv"),
        (catalog_df[CATALOG_COLUMNS], "catalog.csv"),
        (candidates_df, "candidates.csv"),
        (ground_truth_df, "ground_truth.csv"),
    ]:
        out = output_dir / filename
        df.to_csv(out, index=False)
        LOGGER.info("Scritto: %s (%d righe)", out, len(df))

    print(f"\n  {dataset_name.upper()} — Preparazione completata")
    print(f"    Utenti con candidates: {len(ground_truth_df)}")
    print(f"    Catalog items:         {len(catalog_df)}")
    print(f"    Output dir:            {output_dir}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara i dataset Beauty e Sports per l'esperimento con Gemma 2."
    )
    parser.add_argument(
        "--datasets", nargs="+",
        choices=["beauty", "sports", "both"],
        default=["both"],
        help="Dataset da preparare (default: both)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=SAMPLE_SIZE,
        help=f"Utenti da campionare (default: {SAMPLE_SIZE}; usa 0 per tutti)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    sample_size = args.sample_size if args.sample_size > 0 else None

    datasets_to_run = args.datasets
    if "both" in datasets_to_run:
        datasets_to_run = ["beauty", "sports"]

    print("\n" + "=" * 60)
    print("  Preparazione dataset per gemma2 (Beauty & Sports)")
    print("=" * 60)

    dataset_configs = {
        "beauty": {
            "reviews": PROJECT_ROOT / "data/raw/beauty/reviews_Beauty.json",
            "meta":    PROJECT_ROOT / "data/raw/beauty/meta_Beauty.json",
            "output":  PROJECT_ROOT / "data/processed/beauty",
        },
        "sports": {
            "reviews": PROJECT_ROOT / "data/raw/sports/reviews_Sports.json",
            "meta":    PROJECT_ROOT / "data/raw/sports/meta_Sports.json",
            "output":  PROJECT_ROOT / "data/processed/sports",
        },
    }

    for ds_name in datasets_to_run:
        cfg = dataset_configs[ds_name]
        prepare_dataset(
            dataset_name=ds_name,
            reviews_path=cfg["reviews"],
            metadata_path=cfg["meta"],
            output_dir=cfg["output"],
            sample_size=sample_size,
        )

    print("=" * 60)
    print("  Preparazione completata!")
    print("  I dataset sono pronti in data/processed/{beauty,sports}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
