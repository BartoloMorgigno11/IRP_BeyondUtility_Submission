"""Deterministic random baseline for the Plan B MovieLens-1M track.

Run from the repository root:

    python src/baselines/random_baseline.py --config configs/ml1m.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import read_table
from src.data.schemas import validate_candidates, validate_ground_truth
from src.evaluation.utility_metrics import evaluate_utility_at_k
from src.utils.config import ConfigError, find_project_root, load_yaml_config, resolve_project_path


LOGGER = logging.getLogger("random_baseline")
MODEL_NAME = "random"

RECOMMENDATION_COLUMNS = [
    "dataset",
    "model",
    "model_name",
    "user_id",
    "item_id",
    "recommended_item_id",
    "score",
    "rank",
    "run_id",
    "is_positive",
    "candidate_position",
    "config_path",
    "candidates_path",
    "ground_truth_path",
    "timestamp",
]

METRIC_COLUMNS = [
    "dataset",
    "model",
    "model_name",
    "metric",
    "k",
    "value",
    "num_users",
    "num_recommendation_rows",
    "status",
    "run_id",
    "config_path",
    "timestamp",
]

SUMMARY_COLUMNS = [
    "dataset",
    "model",
    "status",
    "users_evaluated",
    "candidate_rows",
    "recommendation_rows",
    "metrics_written",
    "seed",
    "notes",
    "candidates_path",
    "ground_truth_path",
    "recommendations_path",
    "metrics_path",
    "timestamp",
]


class RandomBaselineBlocked(ValueError):
    """Raised when the Random baseline cannot produce real outputs."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic Random baseline over canonical candidates."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Project-relative YAML config path.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id stored in output rows.",
    )
    return parser.parse_args(argv)


def run_random_baseline(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    run_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path, bool]:
    """Run the deterministic Random baseline and compute HR/NDCG when valid."""

    root = Path(project_root).resolve() if project_root is not None else find_project_root()
    config = load_yaml_config(config_path, project_root=root)
    _ensure_api_calls_disabled(config)

    dataset = str(config["dataset"]["name"])
    seed = int(config["experiment"]["seed"])
    k = max(int(value) for value in config["experiment"]["k_values"])
    run_id = run_id or f"{dataset}_{MODEL_NAME}_seed{seed}"
    timestamp = datetime.now(timezone.utc).isoformat()
    config_label = str(config_path)

    raw_outputs_dir = _configured_output_dir(config, root, "raw_outputs_dir", "results/raw_outputs")
    metrics_dir = _configured_output_dir(config, root, "metrics_dir", "results/metrics")
    tables_dir = _configured_output_dir(config, root, "tables_dir", "results/tables")
    recommendations_path = raw_outputs_dir / f"{dataset}_random_recommendations.csv"
    metrics_path = metrics_dir / f"{dataset}_random_metrics.csv"
    summary_path = tables_dir / f"{dataset}_random_baseline_summary.csv"

    candidates_path = _configured_input_path(config, root, "candidates_path")
    ground_truth_path = _configured_input_path(config, root, "ground_truth_path")
    missing_inputs = _missing_inputs(
        {
            "candidates_path": candidates_path,
            "ground_truth_path": ground_truth_path,
        },
        root,
    )
    if missing_inputs:
        summary = _build_summary(
            dataset=dataset,
            status="missing_input",
            users_evaluated=0,
            candidate_rows=0,
            recommendation_rows=0,
            metrics_written=False,
            seed=seed,
            notes="Missing required input file(s): " + "; ".join(missing_inputs),
            candidates_path=candidates_path,
            ground_truth_path=ground_truth_path,
            recommendations_path=recommendations_path,
            metrics_path=metrics_path,
            timestamp=timestamp,
            root=root,
        )
        _write_summary(summary, summary_path)
        LOGGER.warning("Random baseline skipped; %s", summary.loc[0, "notes"])
        return (
            pd.DataFrame(columns=RECOMMENDATION_COLUMNS),
            pd.DataFrame(columns=METRIC_COLUMNS),
            summary,
            recommendations_path,
            metrics_path,
            summary_path,
            False,
        )

    candidates = _read_candidates(candidates_path)
    ground_truth = _read_ground_truth(ground_truth_path)
    if candidates.empty or ground_truth.empty:
        notes = "Candidates and ground truth must both contain at least one row."
        summary = _build_summary(
            dataset=dataset,
            status="blocked",
            users_evaluated=0,
            candidate_rows=len(candidates),
            recommendation_rows=0,
            metrics_written=False,
            seed=seed,
            notes=notes,
            candidates_path=candidates_path,
            ground_truth_path=ground_truth_path,
            recommendations_path=recommendations_path,
            metrics_path=metrics_path,
            timestamp=timestamp,
            root=root,
        )
        _write_summary(summary, summary_path)
        LOGGER.warning("Random baseline blocked; %s", notes)
        return (
            pd.DataFrame(columns=RECOMMENDATION_COLUMNS),
            pd.DataFrame(columns=METRIC_COLUMNS),
            summary,
            recommendations_path,
            metrics_path,
            summary_path,
            False,
        )

    recommendations = rank_candidates_random(
        candidates,
        dataset=dataset,
        seed=seed,
        run_id=run_id,
        config_path=config_label,
        candidates_path=_display_path(candidates_path, root),
        ground_truth_path=_display_path(ground_truth_path, root),
        timestamp=timestamp,
    )
    metrics = random_metric_rows(
        recommendations,
        ground_truth,
        dataset=dataset,
        k=k,
        run_id=run_id,
        config_path=config_label,
        timestamp=timestamp,
    )
    metrics_written = not metrics.empty
    summary = _build_summary(
        dataset=dataset,
        status="completed" if metrics_written else "blocked",
        users_evaluated=int(metrics["num_users"].max()) if metrics_written else 0,
        candidate_rows=len(candidates),
        recommendation_rows=len(recommendations),
        metrics_written=metrics_written,
        seed=seed,
        notes=(
            f"Computed deterministic Random HR@{k} and NDCG@{k}."
            if metrics_written
            else "No metric rows were computed."
        ),
        candidates_path=candidates_path,
        ground_truth_path=ground_truth_path,
        recommendations_path=recommendations_path,
        metrics_path=metrics_path,
        timestamp=timestamp,
        root=root,
    )

    raw_outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(recommendations_path, index=False)
    if metrics_written:
        metrics.to_csv(metrics_path, index=False)
    _write_summary(summary, summary_path)
    LOGGER.info("Saved Random recommendations to %s", _display_path(recommendations_path, root))
    if metrics_written:
        LOGGER.info("Saved Random metrics to %s", _display_path(metrics_path, root))
    LOGGER.info("Saved Random summary to %s", _display_path(summary_path, root))
    return (
        recommendations,
        metrics,
        summary,
        recommendations_path,
        metrics_path,
        summary_path,
        metrics_written,
    )


def rank_candidates_random(
    candidates: pd.DataFrame,
    *,
    dataset: str,
    seed: int,
    run_id: str,
    config_path: str,
    candidates_path: str,
    ground_truth_path: str,
    timestamp: str,
) -> pd.DataFrame:
    """Assign deterministic pseudo-random scores and rank candidates by user."""

    frame = validate_candidates(candidates).copy()
    _coerce_columns_to_string(frame, ("user_id", "candidate_item_id"))
    frame["score"] = [
        _deterministic_score(seed, user_id, item_id)
        for user_id, item_id in zip(frame["user_id"], frame["candidate_item_id"])
    ]
    frame["__item_sort"] = frame["candidate_item_id"].astype(str)
    frame = frame.sort_values(
        ["user_id", "score", "__item_sort"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    frame["rank"] = frame.groupby("user_id", sort=False).cumcount() + 1
    is_positive = (
        _boolean_series(frame["is_ground_truth"])
        if "is_ground_truth" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    recommendations = pd.DataFrame(
        {
            "dataset": dataset,
            "model": MODEL_NAME,
            "model_name": MODEL_NAME,
            "user_id": frame["user_id"],
            "item_id": frame["candidate_item_id"],
            "recommended_item_id": frame["candidate_item_id"],
            "score": frame["score"],
            "rank": frame["rank"].astype(int),
            "run_id": run_id,
            "is_positive": is_positive,
            "candidate_position": frame.get("candidate_position", pd.NA),
            "config_path": config_path,
            "candidates_path": candidates_path,
            "ground_truth_path": ground_truth_path,
            "timestamp": timestamp,
        },
        columns=RECOMMENDATION_COLUMNS,
    )
    return recommendations


def random_metric_rows(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    dataset: str,
    k: int,
    run_id: str,
    config_path: str,
    timestamp: str,
) -> pd.DataFrame:
    """Compute CSV-ready HR@K and NDCG@K rows for Random recommendations."""

    if recommendations.empty or ground_truth.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    values = evaluate_utility_at_k(
        recommendations,
        ground_truth,
        k,
        recommendation_col="recommended_item_id",
    )
    if int(values["user_count"]) == 0:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    rows = []
    for metric_name, value_key in (("hit_rate", "hit_rate"), ("ndcg", "ndcg")):
        rows.append(
            {
                "dataset": dataset,
                "model": MODEL_NAME,
                "model_name": MODEL_NAME,
                "metric": metric_name,
                "k": k,
                "value": float(values[value_key]),
                "num_users": int(values["user_count"]),
                "num_recommendation_rows": int(len(recommendations)),
                "status": "computed",
                "run_id": run_id,
                "config_path": config_path,
                "timestamp": timestamp,
            }
        )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def _read_candidates(path: Path) -> pd.DataFrame:
    candidates = validate_candidates(read_table(path))
    _coerce_columns_to_string(candidates, ("user_id", "candidate_item_id"))
    return candidates


def _read_ground_truth(path: Path) -> pd.DataFrame:
    ground_truth = validate_ground_truth(read_table(path))
    _coerce_columns_to_string(ground_truth, ("user_id", "ground_truth_item_id"))
    return ground_truth


def _coerce_columns_to_string(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].astype("string").str.strip()


def _boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def _deterministic_score(seed: int, user_id: Any, item_id: Any) -> float:
    payload = f"{seed}|{user_id}|{item_id}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def _configured_input_path(config: dict[str, Any], root: Path, key: str) -> Path | None:
    value = config.get("paths", {}).get(key) or config.get("dataset", {}).get(key)
    return resolve_project_path(value, root) if value is not None else None


def _missing_inputs(paths: dict[str, Path | None], root: Path) -> list[str]:
    missing = []
    for name, path in paths.items():
        if path is None:
            missing.append(f"{name}=not_configured")
        elif not path.is_file():
            missing.append(f"{name}={_display_path(path, root)}")
    return missing


def _configured_output_dir(
    config: dict[str, Any],
    root: Path,
    key: str,
    fallback: str,
) -> Path:
    value = config.get("output", {}).get(key, fallback)
    resolved = resolve_project_path(value, root)
    return resolved if resolved is not None else root / fallback


def _build_summary(
    *,
    dataset: str,
    status: str,
    users_evaluated: int,
    candidate_rows: int,
    recommendation_rows: int,
    metrics_written: bool,
    seed: int,
    notes: str,
    candidates_path: Path | None,
    ground_truth_path: Path | None,
    recommendations_path: Path,
    metrics_path: Path,
    timestamp: str,
    root: Path,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "model": MODEL_NAME,
                "status": status,
                "users_evaluated": int(users_evaluated),
                "candidate_rows": int(candidate_rows),
                "recommendation_rows": int(recommendation_rows),
                "metrics_written": bool(metrics_written),
                "seed": int(seed),
                "notes": notes,
                "candidates_path": _display_path(candidates_path, root) if candidates_path is not None else "",
                "ground_truth_path": _display_path(ground_truth_path, root) if ground_truth_path is not None else "",
                "recommendations_path": _display_path(recommendations_path, root),
                "metrics_path": _display_path(metrics_path, root),
                "timestamp": timestamp,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def _write_summary(summary: pd.DataFrame, summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _ensure_api_calls_disabled(config: dict[str, Any]) -> None:
    if config.get("api", {}).get("allow_api_calls"):
        raise ConfigError("api.allow_api_calls must be false; Random baseline never calls APIs.")
    if config.get("experiment", {}).get("allow_api_calls"):
        raise ConfigError(
            "experiment.allow_api_calls must be false; Random baseline never calls APIs."
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args(argv)
    try:
        root = find_project_root()
        recommendations, metrics, summary, recommendations_path, metrics_path, summary_path, metrics_written = (
            run_random_baseline(args.config, project_root=root, run_id=args.run_id)
        )
    except (ConfigError, RandomBaselineBlocked, ValueError, OSError, pd.errors.ParserError) as exc:
        LOGGER.error("%s", exc)
        return 2

    print(f"status={summary.loc[0, 'status']}")
    print(f"recommendations={_display_path(recommendations_path, root) if not recommendations.empty else 'not_written'}")
    print(f"metrics={_display_path(metrics_path, root) if metrics_written else 'not_written'}")
    print(f"summary={_display_path(summary_path, root)}")
    print(f"recommendation_rows={len(recommendations)}")
    print(f"metric_rows={len(metrics)}")
    if not metrics.empty:
        for _, row in metrics.iterrows():
            print(f"{row['metric']}@{int(row['k'])}={float(row['value'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
