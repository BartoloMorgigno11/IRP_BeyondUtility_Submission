from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.llm.plan_d_prompt_templates import PromptItem, build_ranking_prompt


PLAN = "Plan D"
PHASE = "D5E_qwen2_ml1m_ranking_diagnostic"
DATASET = "ml1m"
MODEL = "qwen2:7b-instruct"
BACKEND = "ollama"
PROMPT_STRATEGY = "openp5_base"
TOP_K = 5
DEFAULT_USERS = 20
DEFAULT_CANDIDATE_POOL_SIZE = 20

MANIFEST_COLUMNS = [
    "plan",
    "phase",
    "dataset",
    "user_id",
    "instance_id",
    "prompt_strategy",
    "model",
    "backend",
    "top_k",
    "candidate_pool_size",
    "history_length",
    "candidate_count",
    "positive_item_id",
    "positive_in_candidates",
    "history_preview",
    "candidate_ids",
    "candidate_titles_preview",
    "prompt_char_length",
    "manifest_status",
    "notes",
]


@dataclass(frozen=True)
class ManifestSummary:
    row_count: int
    ready_rows: int
    positives_present_rows: int
    output_path: Path
    notes: str


def build_ml1m_qwen_manifest(
    *,
    data_root: Path,
    output_path: Path,
    requested_users: int = DEFAULT_USERS,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
    top_k: int = TOP_K,
) -> ManifestSummary:
    """Build the deterministic ML-1M no-inference manifest for future Qwen ranking."""

    dataset_dir = data_root / DATASET
    interactions = _read_csv(dataset_dir / "interactions.csv")
    candidates = _read_csv(dataset_dir / "candidates.csv")
    ground_truth = _read_csv(dataset_dir / "ground_truth.csv")
    catalog = _read_csv(dataset_dir / "catalog.csv")

    catalog_by_item = _catalog_map(catalog)
    train = interactions.loc[interactions["split"].astype(str).str.lower() == "train"].copy()
    train = _stable_sort_history(train)
    ground_truth_by_user = {
        str(row["user_id"]): str(row["ground_truth_item_id"]) for _, row in ground_truth.iterrows()
    }
    histories_by_user = {
        str(user_id): group.copy()
        for user_id, group in train.groupby(train["user_id"].astype(str), sort=False)
    }
    candidates_by_user = {
        str(user_id): _stable_sort_candidates(group.copy())
        for user_id, group in candidates.groupby(candidates["user_id"].astype(str), sort=False)
    }

    rows: list[dict[str, str]] = []
    for user_id in _sorted_user_ids(list(candidates_by_user)):
        if len(rows) >= requested_users:
            break
        full_user_candidates = candidates_by_user[user_id]
        if len(full_user_candidates) < candidate_pool_size:
            continue

        history = histories_by_user.get(user_id)
        if history is None:
            continue
        if history.empty:
            continue

        positive_item_id = ground_truth_by_user.get(user_id, "")
        full_candidate_ids = full_user_candidates["candidate_item_id"].astype(str).tolist()
        positive_in_full_candidates = positive_item_id in set(full_candidate_ids)
        gt_flags = full_user_candidates.get("is_ground_truth")
        positive_flag_count = _positive_flag_count(gt_flags) if gt_flags is not None else int(positive_in_full_candidates)
        if positive_flag_count != 1 or not positive_in_full_candidates:
            continue

        user_candidates = _select_diagnostic_candidate_pool(
            full_user_candidates,
            positive_item_id=positive_item_id,
            candidate_pool_size=candidate_pool_size,
        )
        if len(user_candidates) != candidate_pool_size:
            continue
        candidate_ids = user_candidates["candidate_item_id"].astype(str).tolist()

        history_items = [_history_prompt_item(row, catalog_by_item) for _, row in history.iterrows()]
        candidate_items = [_candidate_prompt_item(row, catalog_by_item) for _, row in user_candidates.iterrows()]
        prompt = build_ranking_prompt(
            user_id=user_id,
            history_items=history_items,
            candidate_items=candidate_items,
            top_k=top_k,
            prompt_strategy=PROMPT_STRATEGY,
        )
        rows.append(
            {
                "plan": PLAN,
                "phase": PHASE,
                "dataset": DATASET,
                "user_id": user_id,
                "instance_id": f"{PHASE}_{DATASET}_{len(rows) + 1:02d}_user_{user_id}",
                "prompt_strategy": PROMPT_STRATEGY,
                "model": MODEL,
                "backend": BACKEND,
                "top_k": str(top_k),
                "candidate_pool_size": str(candidate_pool_size),
                "history_length": str(len(history_items)),
                "candidate_count": str(len(candidate_ids)),
                "positive_item_id": positive_item_id,
                "positive_in_candidates": "true",
                "history_preview": _preview_items(history_items, limit=5),
                "candidate_ids": json.dumps(candidate_ids, separators=(",", ":")),
                "candidate_titles_preview": _preview_items(candidate_items, limit=5),
                "prompt_char_length": str(len(prompt)),
                "manifest_status": "ready_for_llm_inference",
                "notes": (
                    "Prepared from reconstructed Plan C ML-1M candidates; 20-item diagnostic pool retains "
                    "the positive target and earliest reconstructed negatives; no LLM inference was run."
                ),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    frame.to_csv(output_path, index=False)
    ready_rows = int((frame["manifest_status"] == "ready_for_llm_inference").sum()) if not frame.empty else 0
    positives_present_rows = int((frame["positive_in_candidates"] == "true").sum()) if not frame.empty else 0
    notes = (
        "prepared_full_20_user_manifest"
        if ready_rows == requested_users
        else f"partial_manifest_ready_rows_{ready_rows}_of_{requested_users}"
    )
    return ManifestSummary(
        row_count=len(frame),
        ready_rows=ready_rows,
        positives_present_rows=positives_present_rows,
        output_path=output_path,
        notes=notes,
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required manifest input: {path}")
    return pd.read_csv(path, dtype="string", keep_default_na=False)


def _catalog_map(catalog: pd.DataFrame) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for _, row in catalog.iterrows():
        item_id = str(row.get("item_id", "")).strip()
        if not item_id:
            continue
        title = str(row.get("title", "")).strip() or item_id
        categories = str(row.get("categories", "")).strip()
        result[item_id] = {"title": title, "metadata": categories}
    return result


def _stable_sort_history(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["__timestamp_sort"] = pd.to_numeric(result.get("timestamp", -1), errors="coerce").fillna(-1)
    result["__item_sort"] = result["item_id"].map(_id_sort_key)
    return result.sort_values(["user_id", "__timestamp_sort", "__item_sort"], kind="mergesort").drop(
        columns=["__timestamp_sort", "__item_sort"]
    )


def _stable_sort_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["__candidate_position"] = pd.to_numeric(result.get("candidate_position", 0), errors="coerce").fillna(0)
    result["__candidate_sort"] = result["candidate_item_id"].map(_id_sort_key)
    return result.sort_values(["__candidate_position", "__candidate_sort"], kind="mergesort").drop(
        columns=["__candidate_position", "__candidate_sort"]
    )


def _select_diagnostic_candidate_pool(
    user_candidates: pd.DataFrame,
    *,
    positive_item_id: str,
    candidate_pool_size: int,
) -> pd.DataFrame:
    positive_rows = user_candidates.loc[user_candidates["candidate_item_id"].astype(str) == positive_item_id]
    if len(positive_rows) != 1:
        return user_candidates.head(0).copy()
    negatives = user_candidates.loc[user_candidates["candidate_item_id"].astype(str) != positive_item_id].head(
        candidate_pool_size - 1
    )
    selected = pd.concat([positive_rows, negatives], ignore_index=True)
    return _stable_sort_candidates(selected)


def _sorted_user_ids(user_ids: list[str]) -> list[str]:
    return sorted((str(user_id) for user_id in user_ids), key=_id_sort_key)


def _id_sort_key(value: object) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def _history_prompt_item(row: pd.Series, catalog_by_item: dict[str, dict[str, str]]) -> PromptItem:
    item_id = str(row["item_id"])
    meta = catalog_by_item.get(item_id, {})
    return PromptItem(
        item_id=item_id,
        title=meta.get("title", item_id),
        metadata=meta.get("metadata", ""),
    )


def _candidate_prompt_item(row: pd.Series, catalog_by_item: dict[str, dict[str, str]]) -> PromptItem:
    item_id = str(row["candidate_item_id"])
    title = str(row.get("candidate_item_title", "")).strip()
    meta = catalog_by_item.get(item_id, {})
    return PromptItem(
        item_id=item_id,
        title=title or meta.get("title", item_id),
        metadata=meta.get("metadata", ""),
    )


def _preview_items(items: list[PromptItem], *, limit: int) -> str:
    preview = [f"{item.item_id}:{item.title}" for item in items[:limit]]
    return " | ".join(preview)


def _positive_flag_count(series: pd.Series) -> int:
    normalized = series.astype(str).str.strip().str.lower()
    return int(normalized.isin({"true", "1", "yes", "y"}).sum())
