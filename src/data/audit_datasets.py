"""Audit configured dataset paths and available upstream artifacts.

Run from the repository root:

    python src/data/audit_datasets.py --config configs/beauty.yaml
"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import (  # noqa: E402
    count_nonempty_lines,
    list_zip_members,
    read_csv_from_zip,
    read_json,
    read_json_from_zip,
    read_table,
    zip_member_exists,
)
from src.utils.config import find_project_root, load_yaml_config, resolve_project_path  # noqa: E402


LOGGER = logging.getLogger("audit_datasets")

CANONICAL_DATA_KEYS = (
    "interactions_path",
    "catalog_path",
    "candidates_path",
    "ground_truth_path",
)
PATH_KEYS = (
    "raw_dir",
    "processed_dir",
    "sample_dir",
    "interactions_path",
    "catalog_path",
    "candidates_path",
    "ground_truth_path",
    "llm_outputs_path",
    "baseline_outputs_path",
)
SEARCH_PATTERNS = {
    "interactions_path": ("sequential_data", "interactions", "ratings", "reviews"),
    "catalog_path": ("meta.json", "datamaps.json", "catalog", "item", "items"),
    "candidates_path": ("negative_samples", "candidate", "candidates"),
    "ground_truth_path": ("ground_truth", "test", "target"),
}
CORE_STATS = (
    "user_count",
    "item_count",
    "interaction_count",
    "split_availability",
    "candidate_list_availability",
    "ground_truth_availability",
    "item_metadata_availability",
    "average_user_history_length",
    "catalog_size",
    "sparsity",
)


@dataclass
class AuditContext:
    dataset: str
    config_path: Path
    project_root: Path
    config: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit dataset path availability and basic stats.")
    parser.add_argument("--config", required=True, type=Path, help="Path to a YAML config from the project root.")
    return parser.parse_args()


def audit_dataset(config_path: Path) -> pd.DataFrame:
    project_root = find_project_root()
    config = load_yaml_config(config_path, project_root=project_root)
    context = AuditContext(
        dataset=str(config["dataset"]["name"]),
        config_path=config_path,
        project_root=project_root,
        config=config,
    )

    rows: list[dict[str, Any]] = []
    computed_stats: dict[str, Any] = {}

    _audit_configured_paths(context, rows)
    _audit_original_repo_search(context, rows)
    _audit_llm_outputs(context, rows, computed_stats)
    _audit_baseline_outputs(context, rows)
    _audit_canonical_files(context, rows, computed_stats)
    _add_core_stats(context, rows, computed_stats)

    df = pd.DataFrame(rows, columns=_audit_columns())
    return df


def save_audit_outputs(df: pd.DataFrame, config: dict[str, Any], project_root: Path) -> tuple[Path, Path]:
    dataset = str(config["dataset"]["name"])
    configured_output = config["output"].get("dataset_audit_path")
    output_path = resolve_project_path(configured_output, project_root)
    if output_path is None:
        output_path = project_root / "results" / "tables" / f"{dataset}_dataset_audit.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    report_path = project_root / "report" / "dataset_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown_report(project_root, report_path)
    return output_path, report_path


def _audit_configured_paths(context: AuditContext, rows: list[dict[str, Any]]) -> None:
    for key in PATH_KEYS:
        _add_path_row(
            context,
            rows,
            category="configured_path",
            item=key,
            config_key=f"paths.{key}",
            configured_value=context.config["paths"].get(key),
        )

    for key, value in context.config["original_repo"].items():
        _add_path_row(
            context,
            rows,
            category="original_repo_path",
            item=key,
            config_key=f"original_repo.{key}",
            configured_value=value,
        )


def _audit_original_repo_search(context: AuditContext, rows: list[dict[str, Any]]) -> None:
    original_root_value = context.config["original_repo"].get("root")
    original_root = resolve_project_path(original_root_value, context.project_root)
    if original_root is None or not original_root.exists():
        for key in CANONICAL_DATA_KEYS:
            if context.config["paths"].get(key) is None:
                _add_row(
                    context,
                    rows,
                    category="search",
                    item=key,
                    config_key=f"paths.{key}",
                    configured_value=None,
                    resolved_path=None,
                    exists=False,
                    status="not_searched",
                    statistic="search_match_count",
                    value=0,
                    note="Original repository root is missing.",
                )
        return

    for key in CANONICAL_DATA_KEYS:
        if context.config["paths"].get(key) is not None:
            continue
        matches = _search_original_repo(original_root, SEARCH_PATTERNS[key], context.dataset)
        if not matches:
            _add_row(
                context,
                rows,
                category="search",
                item=key,
                config_key=f"paths.{key}",
                configured_value=None,
                resolved_path=None,
                exists=False,
                status="not_found",
                statistic="search_match_count",
                value=0,
                note="No likely canonical dataset file found by name search.",
            )
            continue

        for match in matches[:20]:
            _add_row(
                context,
                rows,
                category="search",
                item=key,
                config_key=f"paths.{key}",
                configured_value=None,
                resolved_path=match,
                exists=True,
                status="possible_match",
                statistic="search_match",
                value=match,
                note="Candidate found by name search; validate schema before using as canonical data.",
            )
        if len(matches) > 20:
            _add_row(
                context,
                rows,
                category="search",
                item=key,
                config_key=f"paths.{key}",
                configured_value=None,
                resolved_path=None,
                exists=True,
                status="truncated",
                statistic="additional_search_matches",
                value=len(matches) - 20,
                note="Only the first 20 search matches are listed.",
            )


def _audit_llm_outputs(
    context: AuditContext,
    rows: list[dict[str, Any]],
    computed_stats: dict[str, Any],
) -> None:
    zip_path = _resolve_optional_path(context, context.config["paths"].get("llm_outputs_path"))
    if zip_path is None or not zip_path.is_file():
        return

    try:
        members = list_zip_members(zip_path)
    except zipfile.BadZipFile:
        _add_stat_row(context, rows, "llm_outputs", "zip_valid", False, "Configured LLM output is not a valid ZIP.")
        return

    _add_stat_row(context, rows, "llm_outputs", "zip_member_count", len(members))
    _add_stat_row(context, rows, "llm_outputs", "json_file_count", _count_suffix(members, ".json"))
    _add_stat_row(context, rows, "llm_outputs", "csv_file_count", _count_suffix(members, ".csv"))
    _add_stat_row(context, rows, "llm_outputs", "txt_file_count", _count_suffix(members, ".txt"))

    for index, model_config in enumerate(context.config["models"].get("llm_outputs", [])):
        member = model_config.get("archive_member")
        item = f"models.llm_outputs[{index}]"
        if not isinstance(member, str):
            continue
        member_exists = zip_member_exists(zip_path, member)
        _add_row(
            context,
            rows,
            category="llm_output",
            item=item,
            config_key=f"{item}.archive_member",
            configured_value=member,
            resolved_path=f"{zip_path}!{member}",
            exists=member_exists,
            status="found" if member_exists else "missing",
            statistic="archive_member_exists",
            value=member_exists,
            note="Configured LLM output archive member.",
        )
        if not member_exists or not member.lower().endswith(".json"):
            continue
        try:
            records = read_json_from_zip(zip_path, member)
        except Exception as exc:
            _add_stat_row(context, rows, item, "llm_record_read_error", "unknown", str(exc))
            continue
        _summarize_llm_records(context, rows, item, records, computed_stats)


def _audit_baseline_outputs(context: AuditContext, rows: list[dict[str, Any]]) -> None:
    zip_path = _resolve_optional_path(context, context.config["paths"].get("baseline_outputs_path"))
    if zip_path is None:
        _add_stat_row(context, rows, "baseline_outputs", "baseline_outputs_path", "not_configured")
        return
    if not zip_path.is_file():
        _add_stat_row(context, rows, "baseline_outputs", "baseline_outputs_path", "missing")
        return

    try:
        members = list_zip_members(zip_path)
    except zipfile.BadZipFile:
        _add_stat_row(context, rows, "baseline_outputs", "zip_valid", False, "Configured baseline output is not a valid ZIP.")
        return

    _add_stat_row(context, rows, "baseline_outputs", "zip_member_count", len(members))
    _add_stat_row(context, rows, "baseline_outputs", "csv_file_count", _count_suffix(members, ".csv"))

    for index, model_config in enumerate(context.config["models"].get("baseline_outputs", [])):
        member = model_config.get("archive_member")
        item = f"models.baseline_outputs[{index}]"
        if not isinstance(member, str):
            continue
        member_exists = zip_member_exists(zip_path, member)
        _add_row(
            context,
            rows,
            category="baseline_output",
            item=item,
            config_key=f"{item}.archive_member",
            configured_value=member,
            resolved_path=f"{zip_path}!{member}",
            exists=member_exists,
            status="found" if member_exists else "missing",
            statistic="archive_member_exists",
            value=member_exists,
            note="Configured baseline output archive member.",
        )
        if not member_exists:
            continue
        try:
            table = read_csv_from_zip(zip_path, member, sep="\t")
        except Exception as exc:
            _add_stat_row(context, rows, item, "baseline_read_error", "unknown", str(exc))
            continue
        _summarize_baseline_table(context, rows, item, table)


def _audit_canonical_files(
    context: AuditContext,
    rows: list[dict[str, Any]],
    computed_stats: dict[str, Any],
) -> None:
    for key in CANONICAL_DATA_KEYS:
        path = _resolve_optional_path(context, context.config["paths"].get(key))
        if path is None:
            continue
        if not path.is_file():
            continue
        try:
            if path.suffix.lower() in {".csv", ".tsv"}:
                table = read_table(path)
                _summarize_canonical_table(context, rows, key, table, computed_stats)
            elif path.suffix.lower() in {".json", ".jsonl"}:
                _summarize_json_file(context, rows, key, path, computed_stats)
            else:
                _add_stat_row(context, rows, key, "nonempty_line_count", count_nonempty_lines(path))
        except Exception as exc:
            _add_stat_row(context, rows, key, "read_error", "unknown", str(exc))


def _summarize_llm_records(
    context: AuditContext,
    rows: list[dict[str, Any]],
    item: str,
    records: Any,
    computed_stats: dict[str, Any],
) -> None:
    if not isinstance(records, list):
        _add_stat_row(context, rows, item, "llm_record_count", "unknown", "JSON root is not a list.")
        return

    users = [record.get("user") for record in records if isinstance(record, dict) and record.get("user") is not None]
    candidate_lengths = []
    for record in records:
        if not isinstance(record, dict) or "cand_list" not in record:
            continue
        cand_list = _parse_candidate_list(record["cand_list"])
        if cand_list is not None:
            candidate_lengths.append(len(cand_list))
    target_count = sum(1 for record in records if isinstance(record, dict) and record.get("target") is not None)

    _add_stat_row(context, rows, item, "llm_record_count", len(records))
    _add_stat_row(context, rows, item, "llm_unique_user_count", len(set(users)) if users else "unknown")
    _add_stat_row(context, rows, item, "llm_records_with_target", target_count)
    if candidate_lengths:
        _add_stat_row(context, rows, item, "candidate_list_length_min", min(candidate_lengths))
        _add_stat_row(context, rows, item, "candidate_list_length_max", max(candidate_lengths))
        _add_stat_row(context, rows, item, "candidate_list_length_mean", float(np.mean(candidate_lengths)))
        computed_stats["candidate_list_availability"] = "available_in_configured_llm_outputs"


def _summarize_baseline_table(context: AuditContext, rows: list[dict[str, Any]], item: str, table: pd.DataFrame) -> None:
    _add_stat_row(context, rows, item, "baseline_row_count", len(table))
    if "user_id" in table.columns:
        _add_stat_row(context, rows, item, "baseline_unique_user_count", int(table["user_id"].nunique()))
    if "rec_items" in table.columns:
        lengths = []
        for value in table["rec_items"].dropna():
            try:
                parsed = ast.literal_eval(str(value))
            except (SyntaxError, ValueError):
                continue
            if isinstance(parsed, list):
                lengths.append(len(parsed))
        if lengths:
            _add_stat_row(context, rows, item, "baseline_rec_items_length_min", min(lengths))
            _add_stat_row(context, rows, item, "baseline_rec_items_length_max", max(lengths))
            _add_stat_row(context, rows, item, "baseline_rec_items_length_mean", float(np.mean(lengths)))


def _summarize_canonical_table(
    context: AuditContext,
    rows: list[dict[str, Any]],
    key: str,
    table: pd.DataFrame,
    computed_stats: dict[str, Any],
) -> None:
    _add_stat_row(context, rows, key, "row_count", len(table))
    if key == "interactions_path":
        computed_stats["interaction_count"] = len(table)
        if "user_id" in table.columns:
            computed_stats["user_count"] = int(table["user_id"].nunique())
            _add_stat_row(context, rows, key, "user_count", computed_stats["user_count"])
        if "item_id" in table.columns:
            computed_stats["item_count"] = int(table["item_id"].nunique())
            _add_stat_row(context, rows, key, "item_count", computed_stats["item_count"])
        if "split" in table.columns:
            split_values = sorted(str(value) for value in table["split"].dropna().unique())
            computed_stats["split_availability"] = ",".join(split_values) if split_values else "unknown"
            _add_stat_row(context, rows, key, "split_values", computed_stats["split_availability"])
    elif key == "catalog_path":
        computed_stats["catalog_size"] = len(table)
        computed_stats["item_metadata_availability"] = "available"
    elif key == "candidates_path":
        computed_stats["candidate_list_availability"] = "available"
    elif key == "ground_truth_path":
        computed_stats["ground_truth_availability"] = "available"


def _summarize_json_file(
    context: AuditContext,
    rows: list[dict[str, Any]],
    key: str,
    path: Path,
    computed_stats: dict[str, Any],
) -> None:
    if path.suffix.lower() == ".jsonl":
        line_count = count_nonempty_lines(path)
        _add_stat_row(context, rows, key, "jsonl_record_count", line_count)
        if key == "catalog_path":
            computed_stats["catalog_size"] = line_count
            computed_stats["item_metadata_availability"] = "available"
        return

    data = read_json(path)
    if isinstance(data, list):
        _add_stat_row(context, rows, key, "json_record_count", len(data))
        if key == "catalog_path":
            computed_stats["catalog_size"] = len(data)
            computed_stats["item_metadata_availability"] = "available"
    elif isinstance(data, dict):
        _add_stat_row(context, rows, key, "json_key_count", len(data))
    else:
        _add_stat_row(context, rows, key, "json_record_count", "unknown", "JSON root is not a list or object.")


def _add_core_stats(context: AuditContext, rows: list[dict[str, Any]], computed_stats: dict[str, Any]) -> None:
    if "candidate_list_availability" not in computed_stats:
        configured = context.config["paths"].get("candidates_path")
        computed_stats["candidate_list_availability"] = "configured" if configured else "unknown"
    if "ground_truth_availability" not in computed_stats:
        configured = context.config["paths"].get("ground_truth_path")
        computed_stats["ground_truth_availability"] = "configured" if configured else "unknown"
    if "item_metadata_availability" not in computed_stats:
        configured = context.config["paths"].get("catalog_path")
        computed_stats["item_metadata_availability"] = "configured" if configured else "unknown"
    if "split_availability" not in computed_stats:
        computed_stats["split_availability"] = "unknown"

    for stat in CORE_STATS:
        value = computed_stats.get(stat, "unknown")
        _add_row(
            context,
            rows,
            category="core_dataset_stat",
            item=stat,
            config_key=None,
            configured_value=None,
            resolved_path=None,
            exists=None,
            status="computed" if value != "unknown" else "unknown",
            statistic=stat,
            value=value,
            note="Computed only from available configured files; unknown means insufficient canonical data.",
        )


def _add_path_row(
    context: AuditContext,
    rows: list[dict[str, Any]],
    category: str,
    item: str,
    config_key: str,
    configured_value: Any,
) -> None:
    resolved = _resolve_optional_path(context, configured_value)
    exists = resolved.exists() if resolved is not None else False
    if configured_value is None:
        status = "not_configured"
        note = "Path is null in config; fill after dataset audit if needed."
    elif exists:
        status = "found"
        note = "Configured path exists."
    else:
        status = "missing"
        note = "Configured path does not exist."

    _add_row(
        context,
        rows,
        category=category,
        item=item,
        config_key=config_key,
        configured_value=configured_value,
        resolved_path=resolved,
        exists=exists,
        status=status,
        statistic="path_status",
        value=status,
        note=note,
    )
    if resolved is not None and exists:
        size = resolved.stat().st_size if resolved.is_file() else None
        _add_row(
            context,
            rows,
            category=category,
            item=item,
            config_key=config_key,
            configured_value=configured_value,
            resolved_path=resolved,
            exists=exists,
            status=status,
            statistic="size_bytes",
            value=size if size is not None else "not_applicable",
            note="File size in bytes, or not_applicable for directories.",
        )


def _add_stat_row(
    context: AuditContext,
    rows: list[dict[str, Any]],
    item: str,
    statistic: str,
    value: Any,
    note: str = "",
) -> None:
    _add_row(
        context,
        rows,
        category="inferred_stat",
        item=item,
        config_key=None,
        configured_value=None,
        resolved_path=None,
        exists=None,
        status="computed" if value != "unknown" else "unknown",
        statistic=statistic,
        value=value,
        note=note,
    )


def _add_row(
    context: AuditContext,
    rows: list[dict[str, Any]],
    *,
    category: str,
    item: str,
    config_key: str | None,
    configured_value: Any,
    resolved_path: Any,
    exists: bool | None,
    status: str,
    statistic: str,
    value: Any,
    note: str,
) -> None:
    rows.append(
        {
            "dataset": context.dataset,
            "config_path": _format_audit_value(context.config_path, context.project_root),
            "category": category,
            "item": item,
            "config_key": config_key,
            "configured_value": _format_audit_value(configured_value, context.project_root),
            "resolved_path": _format_audit_value(resolved_path, context.project_root),
            "exists": exists,
            "status": status,
            "statistic": statistic,
            "value": _format_audit_value(value, context.project_root),
            "note": note,
        }
    )


def _resolve_optional_path(context: AuditContext, value: Any) -> Path | None:
    if value is None:
        return None
    return resolve_project_path(value, context.project_root)


def _parse_candidate_list(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
        return parsed if isinstance(parsed, list) else None
    return None


def _search_original_repo(original_root: Path, patterns: tuple[str, ...], dataset: str) -> list[str]:
    matches: list[str] = []
    lower_patterns = tuple(pattern.lower() for pattern in patterns)
    dataset_lower = dataset.lower()

    for path in original_root.rglob("*"):
        if ".git" in path.parts:
            continue
        lower_path = str(path).lower()
        if dataset_lower not in lower_path:
            continue
        if path.is_file() and any(pattern in path.name.lower() for pattern in lower_patterns):
            matches.append(str(path))

    for zip_path in original_root.rglob("*.zip"):
        if ".git" in zip_path.parts:
            continue
        try:
            members = list_zip_members(zip_path)
        except zipfile.BadZipFile:
            continue
        for member in members:
            member_name = str(member["name"])
            lower_member = member_name.lower()
            if dataset_lower in lower_member and any(pattern in lower_member for pattern in lower_patterns):
                matches.append(f"{zip_path}!{member_name}")

    return sorted(set(matches))


def _count_suffix(members: list[dict[str, Any]], suffix: str) -> int:
    return sum(1 for member in members if str(member["name"]).lower().endswith(suffix))


def _format_audit_value(value: Any, project_root: Path) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return _relative_path_text(value, project_root)
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, str):
            return _relative_embedded_path_text(value, project_root)
        return value
    return _relative_embedded_path_text(str(value), project_root)


def _relative_embedded_path_text(value: str, project_root: Path) -> str:
    path_part, separator, archive_member = value.partition("!")
    path = Path(path_part)
    if not path.is_absolute():
        return value
    relative = _relative_path_text(path, project_root)
    if relative == str(path):
        return value
    return f"{relative}{separator}{archive_member}" if separator else relative


def _relative_path_text(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root))
    except ValueError:
        return str(path)


def _audit_columns() -> list[str]:
    return [
        "dataset",
        "config_path",
        "category",
        "item",
        "config_key",
        "configured_value",
        "resolved_path",
        "exists",
        "status",
        "statistic",
        "value",
        "note",
    ]


def _write_markdown_report(project_root: Path, report_path: Path) -> None:
    audit_files = sorted((project_root / "results" / "tables").glob("*_dataset_audit.csv"))
    lines = [
        "# Dataset Audit",
        "",
        "This report is generated from `results/tables/*_dataset_audit.csv`.",
        "Unknown values indicate insufficient configured canonical data; they are not inferred manually.",
        "",
    ]

    if not audit_files:
        lines.extend(["No dataset audit CSV files are available yet.", ""])
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return

    for audit_file in audit_files:
        df = pd.read_csv(audit_file)
        dataset = str(df["dataset"].iloc[0]) if not df.empty else audit_file.stem.replace("_dataset_audit", "")
        lines.extend([f"## {dataset}", ""])

        path_rows = df[df["statistic"] == "path_status"]
        found = int((path_rows["status"] == "found").sum()) if not path_rows.empty else 0
        missing = int((path_rows["status"] == "missing").sum()) if not path_rows.empty else 0
        not_configured = int((path_rows["status"] == "not_configured").sum()) if not path_rows.empty else 0
        lines.extend(
            [
                f"- Audit CSV: `{audit_file.relative_to(project_root)}`",
                f"- Configured paths found: {found}",
                f"- Configured paths missing: {missing}",
                f"- Paths not configured: {not_configured}",
            ]
        )

        core_rows = df[df["category"] == "core_dataset_stat"]
        if not core_rows.empty:
            lines.extend(["", "| Statistic | Value | Status |", "|---|---:|---|"])
            for _, row in core_rows.iterrows():
                lines.append(f"| {row['statistic']} | {row['value']} | {row['status']} |")

        search_rows = df[df["category"] == "search"]
        possible_matches = int((search_rows["status"] == "possible_match").sum()) if not search_rows.empty else 0
        lines.extend(["", f"- Possible original-repo search matches: {possible_matches}", ""])

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    project_root = find_project_root()
    config = load_yaml_config(args.config, project_root=project_root)
    df = audit_dataset(args.config)
    output_path, report_path = save_audit_outputs(df, config, project_root)

    missing_count = int((df["status"] == "missing").sum())
    unknown_count = int((df["status"] == "unknown").sum())
    LOGGER.info("Dataset audit complete: %s", config["dataset"]["name"])
    LOGGER.info("Rows written: %s", len(df))
    LOGGER.info("Missing configured paths: %s", missing_count)
    LOGGER.info("Unknown core stats: %s", unknown_count)
    LOGGER.info("CSV: %s", output_path)
    LOGGER.info("Report: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
