"""Load cached LLM outputs and normalize parsed recommendations.

Run from the repository root:

    python src/experiments/load_llm_outputs.py --config configs/beauty.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import read_json, read_json_from_zip, read_table
from src.evaluation.output_parser import MATCH_UNMATCHED, parse_llm_output
from src.utils.config import ConfigError, find_project_root, load_yaml_config, resolve_project_path


LOGGER = logging.getLogger("load_llm_outputs")

NORMALIZED_OUTPUT_COLUMNS = [
    "user_id",
    "model_name",
    "dataset",
    "rank",
    "recommended_item_id",
    "recommended_item_text",
    "run_id",
    "raw_output",
    "match_type",
    "match_score",
    "catalog_item_title",
    "source_file",
    "source_archive_member",
    "source_record_index",
    "target_item_text",
    "template",
    "candidate_item_ids",
    "prompt_variant",
    "candidate_ordering",
    "history_length",
]

SUMMARY_COLUMNS = [
    "dataset",
    "model_name",
    "run_id",
    "source_file",
    "archive_member",
    "status",
    "message",
    "source_file_exists",
    "archive_member_exists",
    "records_total",
    "records_with_result",
    "records_without_result",
    "non_mapping_records",
    "parsed_recommendations",
    "item_id_matches",
    "exact_matches",
    "normalized_matches",
    "fuzzy_matches",
    "unmatched",
    "catalog_status",
    "normalized_output_path",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize cached LLM recommendation outputs from a YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Project-relative YAML config path.",
    )
    parser.add_argument(
        "--log-unmatched-limit",
        type=int,
        default=20,
        help="Maximum unmatched recommendation examples to log.",
    )
    return parser.parse_args(argv)


def load_llm_outputs(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    log_unmatched_limit: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Load configured cached LLM outputs and write normalized CSV artifacts."""

    root = Path(project_root).resolve() if project_root is not None else find_project_root()
    config = load_yaml_config(config_path, project_root=root)
    _ensure_api_calls_disabled(config)

    dataset = str(config["dataset"]["name"])
    raw_outputs_dir = _configured_output_dir(config, root, "raw_outputs_dir", "results/raw_outputs")
    tables_dir = _configured_output_dir(config, root, "tables_dir", "results/tables")
    raw_outputs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    normalized_output_path = raw_outputs_dir / f"{dataset}_llm_outputs_normalized.csv"
    summary_output_path = tables_dir / f"{dataset}_llm_parsing_summary.csv"
    normalized_output_label = normalized_output_path.relative_to(root)

    catalog, catalog_status = _load_optional_catalog(config, root)
    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    entries = list(_configured_llm_outputs(config))
    if not entries:
        summary_rows.append(
            _diagnostic_row(
                dataset=dataset,
                status="no_configured_llm_outputs",
                message="No models.llm_outputs entries are configured.",
                normalized_output_path=normalized_output_label,
                catalog_status=catalog_status,
            )
        )

    for entry in entries:
        parsed_frame, summary_row = _load_one_configured_output(
            entry,
            dataset=dataset,
            project_root=root,
            catalog=catalog,
            catalog_status=catalog_status,
            normalized_output_path=normalized_output_label,
        )
        frames.append(parsed_frame)
        summary_rows.append(summary_row)

    normalized_outputs = _normalize_output_columns(
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)

    normalized_outputs.to_csv(normalized_output_path, index=False)
    summary.to_csv(summary_output_path, index=False)
    _log_unmatched_outputs(normalized_outputs, log_unmatched_limit)

    LOGGER.info("Saved normalized outputs to %s", normalized_output_path)
    LOGGER.info("Saved parsing summary to %s", summary_output_path)
    return normalized_outputs, summary, normalized_output_path, summary_output_path


def _load_one_configured_output(
    entry: dict[str, Any],
    *,
    dataset: str,
    project_root: Path,
    catalog: pd.DataFrame | None,
    catalog_status: str,
    normalized_output_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model_name = str(entry.get("model_name", "unknown_model"))
    run_id = str(entry.get("run_id", "unknown_run"))
    source_value = entry.get("source_file")
    archive_member = entry.get("archive_member")
    source_path = resolve_project_path(source_value, project_root) if source_value else None
    source_file_exists = bool(source_path and source_path.is_file())

    if source_path is None:
        return pd.DataFrame(), _diagnostic_row(
            dataset=dataset,
            model_name=model_name,
            run_id=run_id,
            source_file=source_value,
            archive_member=archive_member,
            status="source_file_not_configured",
            message="LLM output entry has no source_file.",
            source_file_exists=False,
            archive_member_exists=False,
            normalized_output_path=normalized_output_path,
            catalog_status=catalog_status,
        )

    if not source_path.is_file():
        return pd.DataFrame(), _diagnostic_row(
            dataset=dataset,
            model_name=model_name,
            run_id=run_id,
            source_file=source_value,
            archive_member=archive_member,
            status="source_file_missing",
            message=f"Configured source_file does not exist: {source_value}",
            source_file_exists=False,
            archive_member_exists=False,
            normalized_output_path=normalized_output_path,
            catalog_status=catalog_status,
        )

    payload, archive_member_exists, read_error = _read_configured_payload(
        source_path,
        archive_member,
    )
    if read_error is not None:
        return pd.DataFrame(), _diagnostic_row(
            dataset=dataset,
            model_name=model_name,
            run_id=run_id,
            source_file=source_value,
            archive_member=archive_member,
            status=read_error[0],
            message=read_error[1],
            source_file_exists=source_file_exists,
            archive_member_exists=archive_member_exists,
            normalized_output_path=normalized_output_path,
            catalog_status=catalog_status,
        )

    records, schema_message = _records_from_payload(payload)
    if records is None:
        return pd.DataFrame(), _diagnostic_row(
            dataset=dataset,
            model_name=model_name,
            run_id=run_id,
            source_file=source_value,
            archive_member=archive_member,
            status="unsupported_json_schema",
            message=schema_message,
            source_file_exists=source_file_exists,
            archive_member_exists=archive_member_exists,
            normalized_output_path=normalized_output_path,
            catalog_status=catalog_status,
        )

    parsed_frames: list[pd.DataFrame] = []
    stats = {
        "records_total": len(records),
        "records_with_result": 0,
        "records_without_result": 0,
        "non_mapping_records": 0,
    }

    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            stats["non_mapping_records"] += 1
            continue

        result_value = _extract_result_value(record)
        if _is_missing_value(result_value):
            stats["records_without_result"] += 1
            continue

        stats["records_with_result"] += 1
        output_text = _output_text_from_result(result_value)
        parsed = parse_llm_output(
            output_text,
            user_id=_extract_user_id(record),
            dataset=dataset,
            model_name=model_name,
            run_id=run_id,
            catalog=catalog,
            source_file=str(source_value),
            prompt_variant=entry.get("prompt_variant"),
            candidate_ordering=entry.get("candidate_ordering"),
            history_length=entry.get("history_length"),
        )
        if parsed.empty:
            continue

        parsed["source_archive_member"] = archive_member
        parsed["source_record_index"] = record_index
        parsed["target_item_text"] = record.get("target")
        parsed["template"] = record.get("template")
        parsed["candidate_item_ids"] = _serialize_field(record.get("cand_list"))
        parsed_frames.append(parsed)

    parsed_output = _normalize_output_columns(
        pd.concat(parsed_frames, ignore_index=True) if parsed_frames else pd.DataFrame()
    )
    summary_row = _summary_row_from_parsed(
        parsed_output,
        dataset=dataset,
        model_name=model_name,
        run_id=run_id,
        source_file=str(source_value),
        archive_member=archive_member,
        source_file_exists=source_file_exists,
        archive_member_exists=archive_member_exists,
        normalized_output_path=normalized_output_path,
        catalog_status=catalog_status,
        stats=stats,
    )
    return parsed_output, summary_row


def _read_configured_payload(
    source_path: Path,
    archive_member: str | None,
) -> tuple[Any, bool, tuple[str, str] | None]:
    if source_path.suffix.lower() == ".zip":
        if not archive_member:
            return None, False, (
                "archive_member_not_configured",
                "ZIP source_file requires archive_member.",
            )
        try:
            with zipfile.ZipFile(source_path) as archive:
                member_exists = archive_member in archive.namelist()
            if not member_exists:
                return None, False, (
                    "archive_member_missing",
                    f"ZIP archive does not contain configured member: {archive_member}",
                )
            return read_json_from_zip(source_path, archive_member), True, None
        except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
            return None, False, ("read_error", str(exc))

    try:
        return read_json(source_path), archive_member is None, None
    except (OSError, json.JSONDecodeError) as exc:
        return None, archive_member is None, ("read_error", str(exc))


def _records_from_payload(payload: Any) -> tuple[list[Any] | None, str]:
    if isinstance(payload, list):
        return payload, "Loaded top-level JSON list."
    if isinstance(payload, dict):
        for key in ("records", "data", "outputs", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value, f"Loaded JSON list from key: {key}."
    return None, "Expected a JSON list or a mapping containing a list under records/data/outputs/results."


def _summary_row_from_parsed(
    parsed: pd.DataFrame,
    *,
    dataset: str,
    model_name: str,
    run_id: str,
    source_file: str,
    archive_member: str | None,
    source_file_exists: bool,
    archive_member_exists: bool,
    normalized_output_path: Path,
    catalog_status: str,
    stats: dict[str, int],
) -> dict[str, Any]:
    match_counts = (
        parsed["match_type"].value_counts().to_dict()
        if "match_type" in parsed.columns
        else {}
    )
    parsed_count = int(len(parsed))
    status = "parsed" if parsed_count else "no_parsed_recommendations"
    message = (
        f"Parsed {parsed_count} recommendation rows from {stats['records_with_result']} "
        "records with result."
    )

    return _diagnostic_row(
        dataset=dataset,
        model_name=model_name,
        run_id=run_id,
        source_file=source_file,
        archive_member=archive_member,
        status=status,
        message=message,
        source_file_exists=source_file_exists,
        archive_member_exists=archive_member_exists,
        records_total=stats["records_total"],
        records_with_result=stats["records_with_result"],
        records_without_result=stats["records_without_result"],
        non_mapping_records=stats["non_mapping_records"],
        parsed_recommendations=parsed_count,
        item_id_matches=int(match_counts.get("item_id", 0)),
        exact_matches=int(match_counts.get("exact", 0)),
        normalized_matches=int(match_counts.get("normalized", 0)),
        fuzzy_matches=int(match_counts.get("fuzzy", 0)),
        unmatched=int(match_counts.get(MATCH_UNMATCHED, 0)),
        catalog_status=catalog_status,
        normalized_output_path=normalized_output_path,
    )


def _diagnostic_row(
    *,
    dataset: str,
    model_name: str | None = None,
    run_id: str | None = None,
    source_file: Any = None,
    archive_member: Any = None,
    status: str,
    message: str,
    source_file_exists: bool = False,
    archive_member_exists: bool = False,
    records_total: int = 0,
    records_with_result: int = 0,
    records_without_result: int = 0,
    non_mapping_records: int = 0,
    parsed_recommendations: int = 0,
    item_id_matches: int = 0,
    exact_matches: int = 0,
    normalized_matches: int = 0,
    fuzzy_matches: int = 0,
    unmatched: int = 0,
    catalog_status: str = "catalog_not_configured",
    normalized_output_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "model_name": model_name,
        "run_id": run_id,
        "source_file": source_file,
        "archive_member": archive_member,
        "status": status,
        "message": message,
        "source_file_exists": source_file_exists,
        "archive_member_exists": archive_member_exists,
        "records_total": records_total,
        "records_with_result": records_with_result,
        "records_without_result": records_without_result,
        "non_mapping_records": non_mapping_records,
        "parsed_recommendations": parsed_recommendations,
        "item_id_matches": item_id_matches,
        "exact_matches": exact_matches,
        "normalized_matches": normalized_matches,
        "fuzzy_matches": fuzzy_matches,
        "unmatched": unmatched,
        "catalog_status": catalog_status,
        "normalized_output_path": str(normalized_output_path) if normalized_output_path else None,
    }


def _configured_llm_outputs(config: dict[str, Any]) -> list[dict[str, Any]]:
    llm_outputs = config.get("models", {}).get("llm_outputs", [])
    return [dict(entry) for entry in llm_outputs if isinstance(entry, dict)]


def _configured_output_dir(
    config: dict[str, Any],
    project_root: Path,
    key: str,
    fallback: str,
) -> Path:
    value = config.get("output", {}).get(key, fallback)
    resolved = resolve_project_path(value, project_root)
    if resolved is None:
        return project_root / fallback
    return resolved


def _load_optional_catalog(
    config: dict[str, Any],
    project_root: Path,
) -> tuple[pd.DataFrame | None, str]:
    catalog_value = (
        config.get("paths", {}).get("catalog_path")
        or config.get("dataset", {}).get("catalog_path")
    )
    if catalog_value is None:
        return None, "catalog_not_configured"

    catalog_path = resolve_project_path(catalog_value, project_root)
    if catalog_path is None or not catalog_path.is_file():
        return None, f"catalog_missing:{catalog_value}"

    try:
        if catalog_path.suffix.lower() == ".json":
            data = read_json(catalog_path)
            if isinstance(data, list):
                return pd.DataFrame(data), f"catalog_loaded:{catalog_value}"
            if isinstance(data, dict):
                return pd.DataFrame(data.values()), f"catalog_loaded:{catalog_value}"
            return None, f"catalog_unsupported_json:{catalog_value}"
        return read_table(catalog_path), f"catalog_loaded:{catalog_value}"
    except Exception as exc:  # pragma: no cover - diagnostic path.
        return None, f"catalog_read_error:{catalog_value}:{exc}"


def _ensure_api_calls_disabled(config: dict[str, Any]) -> None:
    if config.get("api", {}).get("allow_api_calls"):
        raise ConfigError("api.allow_api_calls must be false; this loader never calls APIs.")
    if config.get("experiment", {}).get("allow_api_calls"):
        raise ConfigError(
            "experiment.allow_api_calls must be false; this loader never calls APIs."
        )


def _extract_result_value(record: dict[str, Any]) -> Any:
    for key in ("result", "output", "response", "raw_output"):
        if key in record:
            return record[key]
    return None


def _extract_user_id(record: dict[str, Any]) -> Any:
    for key in ("user", "user_id"):
        if key in record:
            return record[key]
    return pd.NA


def _output_text_from_result(result_value: Any) -> str:
    if isinstance(result_value, list):
        return "\n".join(f"{index}. {item}" for index, item in enumerate(result_value, start=1))
    if isinstance(result_value, dict):
        return json.dumps(result_value, ensure_ascii=True)
    return str(result_value)


def _serialize_field(value: Any) -> Any:
    if _is_missing_value(value):
        return pd.NA
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True)
    return value


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _normalize_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=NORMALIZED_OUTPUT_COLUMNS)
    ordered = [column for column in NORMALIZED_OUTPUT_COLUMNS if column in frame.columns]
    extras = [column for column in frame.columns if column not in ordered]
    return frame.loc[:, [*ordered, *extras]]


def _log_unmatched_outputs(frame: pd.DataFrame, limit: int) -> None:
    if frame.empty or "match_type" not in frame.columns:
        LOGGER.warning("No parsed LLM recommendation rows were produced.")
        return

    unmatched = frame.loc[frame["match_type"] == MATCH_UNMATCHED]
    if unmatched.empty:
        LOGGER.info("No unmatched parsed LLM outputs.")
        return

    LOGGER.warning(
        "Unmatched parsed LLM outputs: %s of %s rows.",
        len(unmatched),
        len(frame),
    )
    if limit <= 0:
        return

    for _, row in unmatched.head(limit).iterrows():
        LOGGER.warning(
            "unmatched dataset=%s model=%s user_id=%s rank=%s text=%r source=%s member=%s",
            row.get("dataset"),
            row.get("model_name"),
            row.get("user_id"),
            row.get("rank"),
            row.get("recommended_item_text"),
            row.get("source_file"),
            row.get("source_archive_member"),
        )

    remaining = len(unmatched) - min(limit, len(unmatched))
    if remaining > 0:
        LOGGER.warning("Suppressed %s additional unmatched output log entries.", remaining)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args(argv)

    try:
        normalized, summary, normalized_path, summary_path = load_llm_outputs(
            args.config,
            log_unmatched_limit=args.log_unmatched_limit,
        )
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return 2

    LOGGER.info("Parsed %s recommendation rows.", len(normalized))
    LOGGER.info("Summary statuses: %s", ", ".join(summary["status"].astype(str).tolist()))
    print(f"normalized_outputs={normalized_path}")
    print(f"parsing_summary={summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
