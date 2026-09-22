"""Candidate-set MostPop baseline for the Plan B MovieLens-1M track.

Run from the repository root:

    python src/baselines/ml1m_mostpop_baseline.py --config configs/ml1m.yaml
"""

from __future__ import annotations

import argparse
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
from src.data.schemas import validate_candidates, validate_ground_truth, validate_interactions
from src.evaluation.utility_metrics import evaluate_utility_at_k
from src.utils.config import ConfigError, find_project_root, load_yaml_config, resolve_project_path


LOGGER = logging.getLogger("ml1m_mostpop_baseline")
MODEL_NAME = "mostpop"

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
    "interactions_path",
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
    "train_interactions",
    "candidate_rows",
    "recommendation_rows",
    "metrics_written",
    "notes",
    "interactions_path",
    "candidates_path",
    "ground_truth_path",
    "recommendations_path",
    "metrics_path",
    "comparison_path",
    "timestamp",
]

COMPARISON_COLUMNS = [
    "dataset",
    "model",
    "metric",
    "k",
    "value",
    "status",
    "source_file",
    "notes",
    "timestamp",
]

EXPECTED_COMPARISON_MODELS = ("random", "mostpop")
EXPECTED_COMPARISON_METRICS = ("hit_rate", "ndcg")


class MostPopCandidateBlocked(ValueError):
    """Raised when the Plan B MostPop baseline cannot produce real outputs."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MostPop over canonical ML-1M candidate sets."
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


def run_mostpop_candidate_baseline(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    run_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path, Path, bool]:
    """Run candidate-set MostPop and write metrics plus comparison diagnostics."""

    root = Path(project_root).resolve() if project_root is not None else find_project_root()
    config = load_yaml_config(config_path, project_root=root)
    _ensure_api_calls_disabled(config)

    dataset = str(config["dataset"]["name"])
    k = max(int(value) for value in config["experiment"]["k_values"])
    run_id = run_id or f"{dataset}_{MODEL_NAME}_train_popularity"
    timestamp = datetime.now(timezone.utc).isoformat()
    config_label = str(config_path)

    raw_outputs_dir = _configured_output_dir(config, root, "raw_outputs_dir", "results/raw_outputs")
    metrics_dir = _configured_output_dir(config, root, "metrics_dir", "results/metrics")
    tables_dir = _configured_output_dir(config, root, "tables_dir", "results/tables")
    recommendations_path = raw_outputs_dir / f"{dataset}_mostpop_recommendations.csv"
    metrics_path = metrics_dir / f"{dataset}_mostpop_metrics.csv"
    summary_path = tables_dir / f"{dataset}_mostpop_baseline_summary.csv"
    comparison_path = tables_dir / f"{dataset}_baseline_comparison_summary.csv"

    interactions_path = _configured_input_path(config, root, "interactions_path")
    candidates_path = _configured_input_path(config, root, "candidates_path")
    ground_truth_path = _configured_input_path(config, root, "ground_truth_path")
    missing_inputs = _missing_inputs(
        {
            "interactions_path": interactions_path,
            "candidates_path": candidates_path,
            "ground_truth_path": ground_truth_path,
        },
        root,
    )
    if missing_inputs:
        notes = "Missing required input file(s): " + "; ".join(missing_inputs)
        summary = _build_summary(
            dataset=dataset,
            status="missing_input",
            users_evaluated=0,
            train_interactions=0,
            candidate_rows=0,
            recommendation_rows=0,
            metrics_written=False,
            notes=notes,
            interactions_path=interactions_path,
            candidates_path=candidates_path,
            ground_truth_path=ground_truth_path,
            recommendations_path=recommendations_path,
            metrics_path=metrics_path,
            comparison_path=comparison_path,
            timestamp=timestamp,
            root=root,
        )
        _write_summary(summary, summary_path)
        comparison = build_baseline_comparison(
            dataset=dataset,
            metrics_dir=metrics_dir,
            comparison_path=comparison_path,
            k=k,
            timestamp=timestamp,
            root=root,
        )
        LOGGER.warning("MostPop baseline skipped; %s", notes)
        return (
            pd.DataFrame(columns=RECOMMENDATION_COLUMNS),
            pd.DataFrame(columns=METRIC_COLUMNS),
            summary,
            comparison,
            recommendations_path,
            metrics_path,
            summary_path,
            comparison_path,
            False,
        )

    interactions = _read_interactions(interactions_path)
    candidates = _read_candidates(candidates_path)
    ground_truth = _read_ground_truth(ground_truth_path)
    train_interactions = _train_interactions(interactions)
    if train_interactions.empty or candidates.empty or ground_truth.empty:
        notes = "Train interactions, candidates, and ground truth must all contain at least one row."
        summary = _build_summary(
            dataset=dataset,
            status="blocked",
            users_evaluated=0,
            train_interactions=len(train_interactions),
            candidate_rows=len(candidates),
            recommendation_rows=0,
            metrics_written=False,
            notes=notes,
            interactions_path=interactions_path,
            candidates_path=candidates_path,
            ground_truth_path=ground_truth_path,
            recommendations_path=recommendations_path,
            metrics_path=metrics_path,
            comparison_path=comparison_path,
            timestamp=timestamp,
            root=root,
        )
        _write_summary(summary, summary_path)
        comparison = build_baseline_comparison(
            dataset=dataset,
            metrics_dir=metrics_dir,
            comparison_path=comparison_path,
            k=k,
            timestamp=timestamp,
            root=root,
        )
        LOGGER.warning("MostPop baseline blocked; %s", notes)
        return (
            pd.DataFrame(columns=RECOMMENDATION_COLUMNS),
            pd.DataFrame(columns=METRIC_COLUMNS),
            summary,
            comparison,
            recommendations_path,
            metrics_path,
            summary_path,
            comparison_path,
            False,
        )

    popularity = compute_train_popularity(interactions)
    recommendations = rank_candidates_mostpop(
        candidates,
        popularity,
        dataset=dataset,
        run_id=run_id,
        config_path=config_label,
        interactions_path=_display_path(interactions_path, root),
        candidates_path=_display_path(candidates_path, root),
        ground_truth_path=_display_path(ground_truth_path, root),
        timestamp=timestamp,
    )
    metrics = mostpop_metric_rows(
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
        train_interactions=len(train_interactions),
        candidate_rows=len(candidates),
        recommendation_rows=len(recommendations),
        metrics_written=metrics_written,
        notes=(
            f"Computed MostPop HR@{k} and NDCG@{k} from training-only popularity."
            if metrics_written
            else "No metric rows were computed."
        ),
        interactions_path=interactions_path,
        candidates_path=candidates_path,
        ground_truth_path=ground_truth_path,
        recommendations_path=recommendations_path,
        metrics_path=metrics_path,
        comparison_path=comparison_path,
        timestamp=timestamp,
        root=root,
    )

    raw_outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(recommendations_path, index=False)
    if metrics_written:
        metrics.to_csv(metrics_path, index=False)
    _write_summary(summary, summary_path)
    comparison = build_baseline_comparison(
        dataset=dataset,
        metrics_dir=metrics_dir,
        comparison_path=comparison_path,
        k=k,
        timestamp=timestamp,
        root=root,
    )
    LOGGER.info("Saved MostPop recommendations to %s", _display_path(recommendations_path, root))
    if metrics_written:
        LOGGER.info("Saved MostPop metrics to %s", _display_path(metrics_path, root))
    LOGGER.info("Saved MostPop summary to %s", _display_path(summary_path, root))
    LOGGER.info("Saved baseline comparison to %s", _display_path(comparison_path, root))
    return (
        recommendations,
        metrics,
        summary,
        comparison,
        recommendations_path,
        metrics_path,
        summary_path,
        comparison_path,
        metrics_written,
    )


def compute_train_popularity(interactions: pd.DataFrame) -> pd.DataFrame:
    """Count item popularity from training/history interactions only."""

    frame = validate_interactions(interactions).copy()
    _coerce_columns_to_string(frame, ("user_id", "item_id", "split"))
    train = _train_interactions(frame)
    popularity = (
        train.groupby("item_id", sort=False)
        .size()
        .reset_index(name="score")
    )
    popularity["score"] = popularity["score"].astype(int)
    return popularity.sort_values(
        ["score", "item_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def rank_candidates_mostpop(
    candidates: pd.DataFrame,
    popularity: pd.DataFrame,
    *,
    dataset: str,
    run_id: str,
    config_path: str,
    interactions_path: str,
    candidates_path: str,
    ground_truth_path: str,
    timestamp: str,
) -> pd.DataFrame:
    """Rank each user's candidates by training popularity and item id."""

    frame = validate_candidates(candidates).copy()
    _coerce_columns_to_string(frame, ("user_id", "candidate_item_id"))
    popularity_frame = popularity.copy()
    _coerce_columns_to_string(popularity_frame, ("item_id",))
    if "score" not in popularity_frame.columns:
        raise ValueError("popularity dataframe must contain a score column")

    frame = frame.merge(
        popularity_frame.loc[:, ["item_id", "score"]],
        left_on="candidate_item_id",
        right_on="item_id",
        how="left",
    )
    frame["score"] = frame["score"].fillna(0).astype(int)
    sort_parts = _item_sort_columns(frame["candidate_item_id"])
    for column, values in sort_parts.items():
        frame[column] = values
    frame = frame.sort_values(
        [
            "user_id",
            "score",
            "__item_sort_group",
            "__item_sort_number",
            "__item_sort_text",
        ],
        ascending=[True, False, True, True, True],
        kind="mergesort",
    )
    frame["rank"] = frame.groupby("user_id", sort=False).cumcount() + 1
    is_positive = (
        _boolean_series(frame["is_ground_truth"])
        if "is_ground_truth" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    return pd.DataFrame(
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
            "interactions_path": interactions_path,
            "candidates_path": candidates_path,
            "ground_truth_path": ground_truth_path,
            "timestamp": timestamp,
        },
        columns=RECOMMENDATION_COLUMNS,
    )


def mostpop_metric_rows(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    dataset: str,
    k: int,
    run_id: str,
    config_path: str,
    timestamp: str,
) -> pd.DataFrame:
    """Compute CSV-ready HR@K and NDCG@K rows for MostPop recommendations."""

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


def build_baseline_comparison(
    *,
    dataset: str,
    metrics_dir: Path,
    comparison_path: Path,
    k: int,
    timestamp: str,
    root: Path,
) -> pd.DataFrame:
    """Write a comparison table from real Random and MostPop metric files."""

    rows: list[dict[str, Any]] = []
    for model in EXPECTED_COMPARISON_MODELS:
        metric_file = metrics_dir / f"{dataset}_{model}_metrics.csv"
        metric_frame = _read_metric_file(metric_file)
        for metric in EXPECTED_COMPARISON_METRICS:
            matched = _matching_metric_row(metric_frame, model=model, metric=metric, k=k)
            if matched is None:
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "metric": metric,
                        "k": k,
                        "value": pd.NA,
                        "status": "missing",
                        "source_file": _display_path(metric_file, root),
                        "notes": "Metric row not available; no value invented.",
                        "timestamp": timestamp,
                    }
                )
            else:
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "metric": metric,
                        "k": int(matched["k"]),
                        "value": float(matched["value"]),
                        "status": "available",
                        "source_file": _display_path(metric_file, root),
                        "notes": "Real metric row loaded from generated metrics CSV.",
                        "timestamp": timestamp,
                    }
                )
    comparison = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)
    return comparison


def _matching_metric_row(
    metrics: pd.DataFrame,
    *,
    model: str,
    metric: str,
    k: int,
) -> pd.Series | None:
    if metrics.empty:
        return None
    normalized = metrics.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    required = {"model", "metric", "k", "value"}
    if not required.issubset(normalized.columns):
        return None
    matches = normalized.loc[
        (normalized["model"].astype("string") == model)
        & (normalized["metric"].astype("string") == metric)
        & (pd.to_numeric(normalized["k"], errors="coerce") == k)
        & (normalized["value"].notna())
    ]
    if matches.empty:
        return None
    return matches.iloc[0]


def _read_metric_file(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _read_interactions(path: Path) -> pd.DataFrame:
    interactions = validate_interactions(read_table(path))
    _coerce_columns_to_string(interactions, ("user_id", "item_id", "split"))
    return interactions


def _read_candidates(path: Path) -> pd.DataFrame:
    candidates = validate_candidates(read_table(path))
    _coerce_columns_to_string(candidates, ("user_id", "candidate_item_id"))
    return candidates


def _read_ground_truth(path: Path) -> pd.DataFrame:
    ground_truth = validate_ground_truth(read_table(path))
    _coerce_columns_to_string(ground_truth, ("user_id", "ground_truth_item_id"))
    return ground_truth


def _train_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    split = interactions["split"].astype("string").str.strip().str.lower()
    return interactions.loc[split == "train"].copy()


def _item_sort_columns(values: pd.Series) -> dict[str, list[Any]]:
    groups: list[int] = []
    numbers: list[int] = []
    texts: list[str] = []
    for value in values.astype("string"):
        text = str(value)
        try:
            number = int(text)
        except ValueError:
            groups.append(1)
            numbers.append(0)
            texts.append(text)
        else:
            groups.append(0)
            numbers.append(number)
            texts.append("")
    return {
        "__item_sort_group": groups,
        "__item_sort_number": numbers,
        "__item_sort_text": texts,
    }


def _coerce_columns_to_string(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].astype("string").str.strip()


def _boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


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
    train_interactions: int,
    candidate_rows: int,
    recommendation_rows: int,
    metrics_written: bool,
    notes: str,
    interactions_path: Path | None,
    candidates_path: Path | None,
    ground_truth_path: Path | None,
    recommendations_path: Path,
    metrics_path: Path,
    comparison_path: Path,
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
                "train_interactions": int(train_interactions),
                "candidate_rows": int(candidate_rows),
                "recommendation_rows": int(recommendation_rows),
                "metrics_written": bool(metrics_written),
                "notes": notes,
                "interactions_path": _display_path(interactions_path, root) if interactions_path is not None else "",
                "candidates_path": _display_path(candidates_path, root) if candidates_path is not None else "",
                "ground_truth_path": _display_path(ground_truth_path, root) if ground_truth_path is not None else "",
                "recommendations_path": _display_path(recommendations_path, root),
                "metrics_path": _display_path(metrics_path, root),
                "comparison_path": _display_path(comparison_path, root),
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
        raise ConfigError("api.allow_api_calls must be false; MostPop baseline never calls APIs.")
    if config.get("experiment", {}).get("allow_api_calls"):
        raise ConfigError(
            "experiment.allow_api_calls must be false; MostPop baseline never calls APIs."
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args(argv)
    try:
        root = find_project_root()
        recommendations, metrics, summary, comparison, recommendations_path, metrics_path, summary_path, comparison_path, metrics_written = (
            run_mostpop_candidate_baseline(args.config, project_root=root, run_id=args.run_id)
        )
    except (ConfigError, MostPopCandidateBlocked, ValueError, OSError, pd.errors.ParserError) as exc:
        LOGGER.error("%s", exc)
        return 2

    print(f"status={summary.loc[0, 'status']}")
    print(f"recommendations={_display_path(recommendations_path, root) if not recommendations.empty else 'not_written'}")
    print(f"metrics={_display_path(metrics_path, root) if metrics_written else 'not_written'}")
    print(f"summary={_display_path(summary_path, root)}")
    print(f"comparison={_display_path(comparison_path, root)}")
    print(f"recommendation_rows={len(recommendations)}")
    print(f"metric_rows={len(metrics)}")
    print(f"comparison_rows={len(comparison)}")
    if not metrics.empty:
        for _, row in metrics.iterrows():
            print(f"{row['metric']}@{int(row['k'])}={float(row['value'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
