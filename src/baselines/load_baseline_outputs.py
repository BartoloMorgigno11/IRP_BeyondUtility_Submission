"""Load cached traditional baseline recommendation outputs.

Run from the repository root:

    python src/baselines/load_baseline_outputs.py --config configs/beauty.yaml
"""

from __future__ import annotations

import argparse
import ast
import logging
import re
import sys
import zipfile
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schemas import validate_recommendations
from src.utils.config import ConfigError, find_project_root, load_yaml_config, resolve_project_path


LOGGER = logging.getLogger("load_baseline_outputs")

RAW_OUTPUT_COLUMNS = [
    "user_id",
    "model_name",
    "dataset",
    "rank",
    "recommended_item_id",
    "run_id",
    "source_file",
    "source_archive_member",
    "source_row_index",
    "history_length",
    "seed",
]

SUMMARY_COLUMNS = [
    "dataset",
    "model_name",
    "run_id",
    "source_file",
    "archive_member",
    "source_kind",
    "status",
    "message",
    "source_file_exists",
    "archive_member_exists",
    "records_total",
    "recommendations_normalized",
    "output_path",
]

BASELINE_FILE_RE = re.compile(
    r"rec-(?P<model>[^-]+)-.*?(?:history(?P<history>\d+))?.*?(?:seed(?P<seed>\d+))?",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class BaselineSource:
    """One candidate cached baseline output source."""

    model_name: str
    source_path: Path
    source_file: str
    archive_member: str | None
    source_kind: str
    run_id: str
    history_length: int | None = None
    seed: int | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize cached traditional baseline recommendation outputs."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Project-relative YAML config path.",
    )
    return parser.parse_args(argv)


def load_baseline_outputs(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Find and normalize cached traditional baseline outputs for one config."""

    root = Path(project_root).resolve() if project_root is not None else find_project_root()
    config = load_yaml_config(config_path, project_root=root)
    _ensure_api_calls_disabled(config)

    dataset = str(config["dataset"]["name"])
    raw_outputs_dir = _configured_output_dir(
        config,
        root,
        "raw_outputs_dir",
        "results/raw_outputs",
    )
    tables_dir = _configured_output_dir(config, root, "tables_dir", "results/tables")
    raw_outputs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    normalized_path = raw_outputs_dir / f"{dataset}_baseline_outputs_normalized.csv"
    summary_path = tables_dir / f"{dataset}_baseline_outputs_summary.csv"
    normalized_path_label = normalized_path.relative_to(root)

    sources, diagnostics = discover_baseline_sources(config, root, dataset)
    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = [dict(row) for row in diagnostics]

    for source in sources:
        normalized, summary_row = _load_one_source(
            source,
            dataset=dataset,
            output_path=normalized_path_label,
        )
        frames.append(normalized)
        summary_rows.append(summary_row)

    normalized_outputs = _normalize_output_columns(
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)

    if not normalized_outputs.empty:
        normalized_outputs.to_csv(normalized_path, index=False)
        LOGGER.info("Saved normalized baseline outputs to %s", normalized_path)
    else:
        LOGGER.warning("No cached baseline recommendation outputs were normalized.")

    summary.to_csv(summary_path, index=False)
    LOGGER.info("Saved baseline output summary to %s", summary_path)
    return normalized_outputs, summary, normalized_path, summary_path


def discover_baseline_sources(
    config: dict[str, Any],
    project_root: Path,
    dataset: str,
) -> tuple[list[BaselineSource], list[dict[str, Any]]]:
    """Discover configured and original-repo baseline output sources."""

    sources: list[BaselineSource] = []
    diagnostics: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()

    configured_entries = _configured_baseline_entries(config)
    if not configured_entries:
        diagnostics.append(
            _diagnostic_row(
                dataset=dataset,
                source_kind="models.baseline_outputs",
                status="not_configured",
                message="No models.baseline_outputs entries are configured.",
            )
        )

    for entry in configured_entries:
        source = _source_from_config_entry(entry, project_root, dataset)
        key = _source_key(source)
        if key not in seen:
            seen.add(key)
            sources.append(source)

    if configured_entries:
        return sources, diagnostics

    baseline_outputs_path = (
        config.get("paths", {}).get("baseline_outputs_path")
        or config.get("dataset", {}).get("baseline_outputs_path")
    )
    if baseline_outputs_path is None:
        diagnostics.append(
            _diagnostic_row(
                dataset=dataset,
                source_kind="paths.baseline_outputs_path",
                status="not_configured",
                message="No paths.baseline_outputs_path is configured.",
            )
        )
    else:
        configured_path_sources = _discover_from_path_value(
            baseline_outputs_path,
            project_root,
            dataset,
            source_kind="paths.baseline_outputs_path",
            diagnostics=diagnostics,
        )
        for source in configured_path_sources:
            key = _source_key(source)
            if key not in seen:
                seen.add(key)
                sources.append(source)

    original_sources = _discover_from_original_repo(config, project_root, dataset)
    if original_sources:
        for source in original_sources:
            key = _source_key(source)
            if key not in seen:
                seen.add(key)
                sources.append(source)
    else:
        diagnostics.append(
            _diagnostic_row(
                dataset=dataset,
                source_kind="original_repo_search",
                status="missing",
                message="No relevant traditional baseline outputs found under original_repo.",
            )
        )

    return sources, diagnostics


def _load_one_source(
    source: BaselineSource,
    *,
    dataset: str,
    output_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_exists = source.source_path.is_file()
    if not source_exists:
        return pd.DataFrame(), _diagnostic_row(
            dataset=dataset,
            model_name=source.model_name,
            run_id=source.run_id,
            source_file=source.source_file,
            archive_member=source.archive_member,
            source_kind=source.source_kind,
            status="missing",
            message=f"Configured source file does not exist: {source.source_file}",
            source_file_exists=False,
            archive_member_exists=False,
            output_path=output_path,
        )

    try:
        table, archive_member_exists = _read_baseline_table(source)
    except Exception as exc:  # pragma: no cover - diagnostic path.
        return pd.DataFrame(), _diagnostic_row(
            dataset=dataset,
            model_name=source.model_name,
            run_id=source.run_id,
            source_file=source.source_file,
            archive_member=source.archive_member,
            source_kind=source.source_kind,
            status="read_error",
            message=str(exc),
            source_file_exists=source_exists,
            archive_member_exists=False,
            output_path=output_path,
        )

    try:
        normalized = normalize_baseline_recommendations(table, source, dataset)
    except ValueError as exc:
        return pd.DataFrame(), _diagnostic_row(
            dataset=dataset,
            model_name=source.model_name,
            run_id=source.run_id,
            source_file=source.source_file,
            archive_member=source.archive_member,
            source_kind=source.source_kind,
            status="unsupported_schema",
            message=str(exc),
            source_file_exists=source_exists,
            archive_member_exists=archive_member_exists,
            records_total=len(table),
            output_path=output_path,
        )

    return normalized, _diagnostic_row(
        dataset=dataset,
        model_name=source.model_name,
        run_id=source.run_id,
        source_file=source.source_file,
        archive_member=source.archive_member,
        source_kind=source.source_kind,
        status="found",
        message=f"Normalized {len(normalized)} recommendation rows.",
        source_file_exists=source_exists,
        archive_member_exists=archive_member_exists,
        records_total=len(table),
        recommendations_normalized=len(normalized),
        output_path=output_path,
    )


def normalize_baseline_recommendations(
    table: pd.DataFrame,
    source: BaselineSource,
    dataset: str,
) -> pd.DataFrame:
    """Normalize one cached baseline table into recommendation rows."""

    if not isinstance(table, pd.DataFrame):
        raise ValueError("Baseline output must be a pandas DataFrame.")
    if table.empty:
        return pd.DataFrame(columns=RAW_OUTPUT_COLUMNS)

    normalized = table.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    if "user_id" not in normalized.columns:
        raise ValueError("Baseline output is missing required user_id column.")

    if "rec_items" in normalized.columns:
        rows = _rows_from_rec_items(normalized, source, dataset)
    else:
        rows = _rows_from_long_format(normalized, source, dataset)

    output = pd.DataFrame(rows, columns=RAW_OUTPUT_COLUMNS)
    if not output.empty:
        validate_recommendations(output)
    return output


def _rows_from_rec_items(
    table: pd.DataFrame,
    source: BaselineSource,
    dataset: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_row_index, row in table.iterrows():
        user_id = row["user_id"]
        for rank, item_id in enumerate(_parse_recommended_items(row["rec_items"]), start=1):
            rows.append(
                _recommendation_row(
                    user_id=user_id,
                    rank=rank,
                    item_id=item_id,
                    source=source,
                    dataset=dataset,
                    source_row_index=int(source_row_index),
                )
            )
    return rows


def _rows_from_long_format(
    table: pd.DataFrame,
    source: BaselineSource,
    dataset: str,
) -> list[dict[str, Any]]:
    item_column = _first_present(
        table.columns,
        ("recommended_item_id", "item_id", "rec_item", "candidate_item_id"),
    )
    if item_column is None:
        raise ValueError(
            "Baseline output must contain rec_items or a recommended item column."
        )

    ranked = table.copy()
    ranked["__source_row_index"] = range(len(ranked))
    if "rank" in ranked.columns:
        ranked["__rank_value"] = pd.to_numeric(ranked["rank"], errors="coerce")
        ranked["__rank_missing"] = ranked["__rank_value"].isna()
        ranked = ranked.sort_values(
            ["user_id", "__rank_missing", "__rank_value", "__source_row_index"],
            kind="mergesort",
        )
        ranks = ranked["__rank_value"]
    else:
        ranked = ranked.sort_values(["user_id", "__source_row_index"], kind="mergesort")
        ranks = ranked.groupby("user_id", sort=False).cumcount() + 1

    rows: list[dict[str, Any]] = []
    for (_, row), rank in zip(ranked.iterrows(), ranks):
        if _is_missing_value(row[item_column]):
            continue
        rows.append(
            _recommendation_row(
                user_id=row["user_id"],
                rank=int(rank) if not pd.isna(rank) else len(rows) + 1,
                item_id=row[item_column],
                source=source,
                dataset=dataset,
                source_row_index=int(row["__source_row_index"]),
            )
        )
    return rows


def _recommendation_row(
    *,
    user_id: Any,
    rank: int,
    item_id: Any,
    source: BaselineSource,
    dataset: str,
    source_row_index: int,
) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "model_name": source.model_name,
        "dataset": dataset,
        "rank": rank,
        "recommended_item_id": item_id,
        "run_id": source.run_id,
        "source_file": source.source_file,
        "source_archive_member": source.archive_member,
        "source_row_index": source_row_index,
        "history_length": source.history_length,
        "seed": source.seed,
    }


def _parse_recommended_items(value: Any) -> list[Any]:
    if _is_missing_value(value):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)

    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None

    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    if parsed is not None and not isinstance(parsed, (dict, set)):
        return [parsed]

    text = text.strip("[]")
    if not text:
        return []
    return [part.strip().strip("'\"") for part in text.split(",") if part.strip()]


def _read_baseline_table(source: BaselineSource) -> tuple[pd.DataFrame, bool]:
    if source.source_path.suffix.lower() == ".zip":
        if source.archive_member is None:
            raise ValueError("ZIP baseline source requires an archive_member.")
        with zipfile.ZipFile(source.source_path) as archive:
            if source.archive_member not in archive.namelist():
                raise ValueError(
                    f"ZIP archive does not contain member: {source.archive_member}"
                )
            with archive.open(source.archive_member) as raw_file:
                table = pd.read_csv(TextIOWrapper(raw_file, encoding="utf-8"), sep=None, engine="python")
        return table, True

    table = pd.read_csv(source.source_path, sep=None, engine="python")
    return table, source.archive_member is None


def _configured_baseline_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    entries = config.get("models", {}).get("baseline_outputs", [])
    return [dict(entry) for entry in entries if isinstance(entry, dict)]


def _source_from_config_entry(
    entry: dict[str, Any],
    project_root: Path,
    dataset: str,
) -> BaselineSource:
    source_file = entry["source_file"]
    source_path = resolve_project_path(source_file, project_root)
    if source_path is None:
        raise ConfigError("models.baseline_outputs.source_file must not be null")

    archive_member = entry.get("archive_member")
    inferred = _infer_metadata(entry.get("model_name"), archive_member or str(source_file))
    model_name = str(entry.get("model_name") or inferred["model_name"])
    history_length = entry.get("history_length", inferred["history_length"])
    seed = entry.get("seed", inferred["seed"])
    run_id = str(entry.get("run_id") or _default_run_id(dataset, model_name, archive_member or source_file))
    return BaselineSource(
        model_name=model_name,
        source_path=source_path,
        source_file=str(source_file),
        archive_member=archive_member,
        source_kind="models.baseline_outputs",
        run_id=run_id,
        history_length=_to_optional_int(history_length),
        seed=_to_optional_int(seed),
    )


def _discover_from_path_value(
    path_value: str,
    project_root: Path,
    dataset: str,
    *,
    source_kind: str,
    diagnostics: list[dict[str, Any]],
) -> list[BaselineSource]:
    path = resolve_project_path(path_value, project_root)
    if path is None or not path.exists():
        diagnostics.append(
            _diagnostic_row(
                dataset=dataset,
                source_file=path_value,
                source_kind=source_kind,
                status="missing",
                message=f"Configured baseline output path does not exist: {path_value}",
                source_file_exists=False,
            )
        )
        return []

    if path.is_file():
        return _discover_from_file(path, project_root, dataset, source_kind)

    sources: list[BaselineSource] = []
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            sources.extend(_discover_from_file(file_path, project_root, dataset, source_kind))
    return sources


def _discover_from_original_repo(
    config: dict[str, Any],
    project_root: Path,
    dataset: str,
) -> list[BaselineSource]:
    roots: list[Path] = []
    for path_value in (
        config.get("original_repo", {}).get("sota_results_dir"),
        "original_repo/EvaLLMasRecommender/sota_results",
    ):
        try:
            path = resolve_project_path(path_value, project_root)
        except ConfigError:
            continue
        if path is not None and path.is_dir() and path not in roots:
            roots.append(path)

    sources: list[BaselineSource] = []
    for root in roots:
        for file_path in sorted(root.rglob("*")):
            if file_path.is_file():
                sources.extend(
                    _discover_from_file(
                        file_path,
                        project_root,
                        dataset,
                        source_kind="original_repo_search",
                    )
                )
    return sources


def _discover_from_file(
    path: Path,
    project_root: Path,
    dataset: str,
    source_kind: str,
) -> list[BaselineSource]:
    suffix = path.suffix.lower()
    if suffix not in {".zip", ".csv", ".tsv", ".txt"}:
        return []

    relative_path = _relative_to_root(path, project_root)
    if suffix == ".zip":
        if dataset.lower() not in relative_path.lower():
            return []
        with zipfile.ZipFile(path) as archive:
            members = [
                member
                for member in archive.namelist()
                if _is_relevant_baseline_member(member, dataset)
            ]
        return [
            _source_from_discovered_member(
                path,
                relative_path,
                member,
                dataset,
                source_kind,
            )
            for member in members
        ]

    if not _is_relevant_baseline_member(relative_path, dataset):
        return []
    return [
        _source_from_discovered_member(
            path,
            relative_path,
            None,
            dataset,
            source_kind,
        )
    ]


def _source_from_discovered_member(
    path: Path,
    source_file: str,
    archive_member: str | None,
    dataset: str,
    source_kind: str,
) -> BaselineSource:
    label = archive_member or source_file
    inferred = _infer_metadata(None, label)
    model_name = inferred["model_name"]
    return BaselineSource(
        model_name=model_name,
        source_path=path,
        source_file=source_file,
        archive_member=archive_member,
        source_kind=source_kind,
        run_id=_default_run_id(dataset, model_name, label),
        history_length=inferred["history_length"],
        seed=inferred["seed"],
    )


def _is_relevant_baseline_member(name: str, dataset: str) -> bool:
    lower = name.lower()
    return (
        dataset.lower() in lower
        and lower.endswith((".csv", ".tsv", ".txt"))
        and ("rec-" in lower or "sota" in lower or "baseline" in lower)
    )


def _infer_metadata(
    model_name: Any,
    label: str,
) -> dict[str, Any]:
    match = BASELINE_FILE_RE.search(label)
    inferred_model = str(model_name) if model_name else None
    history_length = None
    seed = None
    if match is not None:
        inferred_model = inferred_model or match.group("model")
        history = match.group("history")
        seed_text = match.group("seed")
        history_length = int(history) if history is not None else None
        seed = int(seed_text) if seed_text is not None else None
    return {
        "model_name": inferred_model or "baseline",
        "history_length": history_length,
        "seed": seed,
    }


def _default_run_id(dataset: str, model_name: str, label: str) -> str:
    stem = Path(label).stem
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return f"{dataset}_{model_name}_{cleaned}"


def _source_key(source: BaselineSource) -> tuple[str, str | None]:
    return (str(source.source_path.resolve()), source.archive_member)


def _configured_output_dir(
    config: dict[str, Any],
    project_root: Path,
    key: str,
    fallback: str,
) -> Path:
    value = config.get("output", {}).get(key, fallback)
    resolved = resolve_project_path(value, project_root)
    return resolved if resolved is not None else project_root / fallback


def _diagnostic_row(
    *,
    dataset: str,
    source_kind: str,
    status: str,
    message: str,
    model_name: str | None = None,
    run_id: str | None = None,
    source_file: str | None = None,
    archive_member: str | None = None,
    source_file_exists: bool = False,
    archive_member_exists: bool = False,
    records_total: int = 0,
    recommendations_normalized: int = 0,
    output_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "model_name": model_name,
        "run_id": run_id,
        "source_file": source_file,
        "archive_member": archive_member,
        "source_kind": source_kind,
        "status": status,
        "message": message,
        "source_file_exists": source_file_exists,
        "archive_member_exists": archive_member_exists,
        "records_total": records_total,
        "recommendations_normalized": recommendations_normalized,
        "output_path": str(output_path) if output_path is not None else None,
    }


def _normalize_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=RAW_OUTPUT_COLUMNS)
    ordered = [column for column in RAW_OUTPUT_COLUMNS if column in frame.columns]
    extras = [column for column in frame.columns if column not in ordered]
    return frame.loc[:, [*ordered, *extras]]


def _first_present(columns: pd.Index, names: tuple[str, ...]) -> str | None:
    available = set(columns)
    for name in names:
        if name in available:
            return name
    return None


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except ValueError:
        return False
    return str(value).strip() == ""


def _to_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _relative_to_root(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root))
    except ValueError:
        return str(path)


def _ensure_api_calls_disabled(config: dict[str, Any]) -> None:
    if config.get("api", {}).get("allow_api_calls"):
        raise ConfigError("api.allow_api_calls must be false; this loader never calls APIs.")
    if config.get("experiment", {}).get("allow_api_calls"):
        raise ConfigError(
            "experiment.allow_api_calls must be false; this loader never calls APIs."
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args(argv)

    try:
        normalized, summary, normalized_path, summary_path = load_baseline_outputs(args.config)
    except (ConfigError, ValueError, zipfile.BadZipFile, OSError) as exc:
        LOGGER.error("%s", exc)
        return 2

    print(f"normalized_outputs={normalized_path if not normalized.empty else 'not_written'}")
    print(f"summary={summary_path}")
    print(f"recommendations_normalized={len(normalized)}")
    print(f"statuses={','.join(summary['status'].astype(str).tolist())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
