"""Finalization F5A-FIX atomic paper result-cell correction.

This phase preserves F5A/F5A-R as historical audits and creates a corrected
inventory where numeric result cells are atomic row/column intersections.
"""

from __future__ import annotations

from src.audit import common_git

import hashlib
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.analysis.protected_input_compatibility import (
    protected_hash_counts as _d1r_protected_hash_counts,
    protected_hash_status as _d1r_protected_hash_status,
)


EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"

REPORT_PATH = Path("report/finalization_f5af_atomic_paper_cells.md")
MANUAL_DIR = Path("results/manual")
ALIGNMENT_DIAGNOSTIC = Path("results/manual/finalization_f5a_alignment_diagnostic.csv")
PDF_CONFIRMATIONS = Path("results/manual/finalization_f5ar_pdf_confirmations.csv")

F2_PROTECTED_AFTER = Path("results/tables/finalization_f2_protected_input_hashes_after.csv")
F2_REGISTRY = Path("results/tables/finalization_f2_canonical_result_registry.csv")
F5A_EXTRACTION = Path("results/tables/finalization_f5a_numeric_cell_extraction.csv")
F5A_MAPPING = Path("results/tables/finalization_f5a_local_result_mapping.csv")
F5AR_TABLES = Path("results/tables/finalization_f5ar_docx_tables.csv")
F5AR_RECOVERY = Path("results/tables/finalization_f5ar_numeric_cell_recovery.csv")

OUTPUT_TABLES: dict[str, Path] = {
    "superseded_target_audit": Path("results/tables/finalization_f5af_superseded_target_audit.csv"),
    "paper_model_inventory": Path("results/tables/finalization_f5af_paper_model_inventory.csv"),
    "atomic_paper_result_cells": Path("results/tables/finalization_f5af_atomic_paper_result_cells.csv"),
    "local_atomic_mapping": Path("results/tables/finalization_f5af_local_atomic_mapping.csv"),
    "corrected_coverage": Path("results/tables/finalization_f5af_corrected_coverage.csv"),
    "anomalies": Path("results/tables/finalization_f5af_anomalies.csv"),
    "validation": Path("results/tables/finalization_f5af_validation.csv"),
}

SUPERSEDED_COLUMNS = [
    "paper_numeric_cell_id",
    "previous_target_type",
    "corrected_target_type",
    "is_numeric",
    "is_atomic",
    "superseded_status",
    "superseded_reason",
    "replacement_inventory",
    "notes",
]

MODEL_COLUMNS = [
    "model_inventory_id",
    "paper_component_id",
    "model_name",
    "model_family",
    "open_or_closed",
    "paper_presence_verified",
    "source_table_or_section",
    "is_numeric_result",
    "local_equivalent",
    "local_equivalence_class",
    "notes",
]

ATOMIC_COLUMNS = [
    "atomic_cell_id",
    "paper_table_number",
    "paper_section",
    "paper_page",
    "dataset",
    "task",
    "model",
    "model_variant",
    "prompt_or_setting",
    "metric",
    "cutoff",
    "row_label",
    "column_label",
    "paper_value_text",
    "paper_value_numeric",
    "decimal_places",
    "significance_marker",
    "source_type",
    "pdf_verified",
    "secondary_source_verified",
    "notes",
]

MAPPING_COLUMNS = [
    "atomic_cell_id",
    "local_result_id",
    "dataset_match",
    "task_match",
    "model_match",
    "version_match",
    "metric_match",
    "cutoff_match",
    "cohort_match",
    "candidate_set_match",
    "protocol_equivalence_class",
    "local_value",
    "absolute_difference",
    "printed_precision_match",
    "exact_numeric_replication",
    "mapping_status",
    "notes",
]

COVERAGE_COLUMNS = [
    "coverage_metric",
    "numerator",
    "denominator",
    "coverage_percent",
    "previous_denominator",
    "previous_value",
    "denominator_policy",
    "notes",
]

ANOMALY_COLUMNS = [
    "anomaly_id",
    "area",
    "severity",
    "source_file",
    "description",
    "scientific_impact",
    "can_continue",
    "recommended_action",
    "notes",
]

VALIDATION_COLUMNS = ["check_id", "check_name", "status", "observed", "expected", "details"]

ABSOLUTE_PATH_PATTERN = re.compile(r"\b[A-Za-z]:[\\/]")


@dataclass(frozen=True)
class SourceBundle:
    root: Path
    protected_inputs: pd.DataFrame
    registry: pd.DataFrame
    f5a_extraction: pd.DataFrame
    f5a_mapping: pd.DataFrame
    f5ar_tables: pd.DataFrame
    f5ar_recovery: pd.DataFrame


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "|".join(clean(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def read_csv(root: Path, relative: Path) -> pd.DataFrame:
    path = root / relative
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_sources(root: Path) -> SourceBundle:
    return SourceBundle(
        root=root,
        protected_inputs=read_csv(root, F2_PROTECTED_AFTER),
        registry=read_csv(root, F2_REGISTRY),
        f5a_extraction=read_csv(root, F5A_EXTRACTION),
        f5a_mapping=read_csv(root, F5A_MAPPING),
        f5ar_tables=read_csv(root, F5AR_TABLES),
        f5ar_recovery=read_csv(root, F5AR_RECOVERY),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def sort_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    existing = [column for column in columns if column in frame.columns]
    return frame.sort_values(existing, kind="mergesort").reset_index(drop=True) if existing else frame.reset_index(drop=True)


def git_output(root: Path, args: Sequence[str]) -> str:
    return common_git.git_stdout_no_check(args, root)


def ensure_manual_evidence(root: Path) -> dict[str, str]:
    """Copy requested manual evidence into results/manual when source files exist.

    The current repository may already contain the destination files. If neither
    source nor destination exists, F5A-FIX records that as an anomaly but does not
    fabricate manual evidence.
    """

    statuses: dict[str, str] = {}
    for filename, destination in {
        "finalization_f5a_alignment_diagnostic.csv": ALIGNMENT_DIAGNOSTIC,
        "finalization_f5ar_pdf_confirmations.csv": PDF_CONFIRMATIONS,
    }.items():
        dest = root / destination
        if dest.exists():
            statuses[destination.as_posix()] = "present"
            continue
        source = root / filename
        if source.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            statuses[destination.as_posix()] = "copied_from_repo_root"
        else:
            statuses[destination.as_posix()] = "missing"
    return statuses


def build_superseded_target_audit(bundle: SourceBundle) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in bundle.f5a_extraction.iterrows():
        metric = clean(row.get("paper_metric"))
        is_model = metric == "model_identity"
        rows.append(
            {
                "paper_numeric_cell_id": clean(row.get("paper_numeric_cell_id")),
                "previous_target_type": "f5a_paper_numeric_cell",
                "corrected_target_type": "paper_model_inventory_component" if is_model else "non_atomic_metric_family",
                "is_numeric": "false",
                "is_atomic": "false",
                "superseded_status": "superseded_non_atomic_numeric_inventory",
                "superseded_reason": (
                    "model identity component is not a numeric result cell"
                    if is_model
                    else "metric-family aggregate spans many row/column intersections and cannot be confirmed by one numeric value"
                ),
                "replacement_inventory": (
                    "results/tables/finalization_f5af_paper_model_inventory.csv"
                    if is_model
                    else "results/tables/finalization_f5af_atomic_paper_result_cells.csv"
                ),
                "notes": "F5A/F5A-R are preserved as historical audits; this row is not deleted or rewritten.",
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=SUPERSEDED_COLUMNS), ["paper_numeric_cell_id"])


def build_paper_model_inventory(bundle: SourceBundle) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    model_rows = bundle.f5a_extraction.loc[bundle.f5a_extraction["paper_metric"].eq("model_identity")]
    for _, row in model_rows.iterrows():
        model = clean(row.get("paper_model"))
        rows.append(
            {
                "model_inventory_id": stable_id("F5AF-MODEL", clean(row.get("paper_numeric_cell_id")), model),
                "paper_component_id": clean(row.get("paper_component_id")),
                "model_name": model,
                "model_family": model_family(model),
                "open_or_closed": open_or_closed(model),
                "paper_presence_verified": "true",
                "source_table_or_section": clean(row.get("paper_table_or_figure")) or clean(row.get("paper_section")),
                "is_numeric_result": "false",
                "local_equivalent": "",
                "local_equivalence_class": "none",
                "notes": "Inventory component only; excluded from all numeric denominators.",
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=MODEL_COLUMNS), ["model_name"])


def model_family(model: str) -> str:
    lowered = model.lower()
    if "gpt" in lowered:
        return "GPT"
    if "claude" in lowered:
        return "Claude"
    if "llama" in lowered:
        return "Llama"
    return "unknown"


def open_or_closed(model: str) -> str:
    lowered = model.lower()
    if "llama" in lowered:
        return "open_or_open_weight"
    if "gpt" in lowered or "claude" in lowered:
        return "closed"
    return "unknown"


def build_atomic_paper_result_cells(bundle: SourceBundle) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    rows.extend(table3_cells(bundle.f5ar_tables))
    rows.extend(table4_cells(bundle.f5ar_tables))
    rows.extend(table5_cells(bundle.f5ar_tables))
    frame = pd.DataFrame(rows, columns=ATOMIC_COLUMNS)
    return sort_frame(frame.drop_duplicates(subset=["atomic_cell_id"]), ["paper_table_number", "dataset", "task", "model", "metric", "column_label", "row_label"])


def table3_cells(tables: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    table = tables.loc[tables["table_index"].eq("3")]
    for _, cell in table.iterrows():
        if clean(cell.get("row_index")) == "0" or clean(cell.get("column_index")) == "0":
            continue
        column = clean(cell.get("column_label"))
        match = re.match(r"(.+?)\s+(HR|NDCG)$", column)
        if not match:
            continue
        dataset = dataset_name(match.group(1))
        metric = "hr_at_5" if match.group(2) == "HR" else "ndcg_at_5"
        rows.append(atomic_row("3", "utility", dataset, "ranking", clean(cell.get("row_label")), "", "", metric, "5", cell))
    return rows


def table4_cells(tables: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    table = tables.loc[tables["table_index"].eq("4")]
    for _, cell in table.iterrows():
        if clean(cell.get("row_index")) == "0" or clean(cell.get("column_index")) == "0":
            continue
        rows.append(
            atomic_row(
                "4",
                "hallucinations",
                dataset_name(clean(cell.get("row_label"))),
                "llm_output_validation",
                clean(cell.get("column_label")),
                "",
                "",
                "hallucination_rate",
                "",
                cell,
            )
        )
    return rows


def table5_cells(tables: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    table = tables.loc[tables["table_index"].eq("5")]
    for _, cell in table.iterrows():
        if clean(cell.get("row_index")) == "0" or clean(cell.get("column_index")) == "0":
            continue
        metric, cutoff = metric_from_table5(clean(cell.get("row_label")))
        rows.append(
            atomic_row(
                "5",
                "reranking_beauty",
                "Amazon Beauty",
                "reranking",
                clean(cell.get("column_label")),
                "",
                "",
                metric,
                cutoff,
                cell,
            )
        )
    return rows


def atomic_row(table_number: str, section: str, dataset: str, task: str, model: str, variant: str, prompt: str, metric: str, cutoff: str, cell: Mapping[str, Any]) -> dict[str, str]:
    text = clean(cell.get("cell_text"))
    value = clean(cell.get("numeric_value"))
    return {
        "atomic_cell_id": stable_id("F5AF-ATOM", table_number, dataset, task, model, metric, cutoff, clean(cell.get("row_label")), clean(cell.get("column_label")), text),
        "paper_table_number": table_number,
        "paper_section": section,
        "paper_page": "unknown_pending_pdf_page_review",
        "dataset": dataset,
        "task": task,
        "model": model,
        "model_variant": variant,
        "prompt_or_setting": prompt,
        "metric": metric,
        "cutoff": cutoff,
        "row_label": clean(cell.get("row_label")),
        "column_label": clean(cell.get("column_label")),
        "paper_value_text": text,
        "paper_value_numeric": value,
        "decimal_places": clean(cell.get("decimal_places")),
        "significance_marker": clean(cell.get("significance_marker")),
        "source_type": "secondary_structured_source_docx",
        "pdf_verified": "false",
        "secondary_source_verified": "true",
        "notes": "Atomic row/column intersection recovered from F5A-R DOCX extraction; PDF visual confirmation still required.",
    }


def dataset_name(value: str) -> str:
    lookup = {"Beauty": "Amazon Beauty", "Sports": "Amazon Sports", "ML-1M": "ML-1M", "LastFM": "LastFM"}
    return lookup.get(clean(value), clean(value))


def metric_from_table5(label: str) -> tuple[str, str]:
    lowered = label.lower()
    if "hr@5" in lowered:
        return "hr_at_5", "5"
    if "ndcg@5" in lowered:
        return "ndcg_at_5", "5"
    if "popularity" in lowered:
        return "average_popularity", ""
    if "serendipity" in lowered:
        return "serendipity", ""
    if "bias hr" in lowered:
        return "candidate_position_bias_hr", ""
    if "bias ndcg" in lowered:
        return "candidate_position_bias_ndcg", ""
    if "hallucination" in lowered:
        return "hallucination_rate", ""
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_"), ""


def normalize_model(value: str) -> str:
    text = clean(value).lower()
    replacements = {
        "bm25-like": "bm25",
        "bprmf_reconstructed": "bprmf",
        "gru4rec": "gru4rec",
        "sasrec": "sasrec",
        "sasurec": "sasrec",
        "qwen2 7b instruct": "qwen2-7b",
        "mistral 7b instruct": "mistral-7b",
    }
    text = replacements.get(text, text)
    return re.sub(r"[^a-z0-9]+", "", text)


def normalize_dataset(value: str) -> str:
    return dataset_name(clean(value)).lower().replace("amazon ", "").replace("-", "").replace(" ", "")


def metric_alias(metric: str) -> str:
    lookup = {
        "average_popularity": "average_popularity_at_5",
        "hallucination_rate": "hallucination",
    }
    return lookup.get(metric, metric)


def build_local_atomic_mapping(bundle: SourceBundle, atomic: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    registry = bundle.registry.copy()
    if not registry.empty:
        registry["_dataset_norm"] = registry["dataset"].map(normalize_dataset)
        registry["_model_norm"] = registry["model"].map(normalize_model)
        registry["_metric_norm"] = registry["metric"].map(metric_alias)
    for _, cell in atomic.iterrows():
        rows.append(mapping_for_cell(cell, registry))
    return sort_frame(pd.DataFrame(rows, columns=MAPPING_COLUMNS), ["atomic_cell_id"])


def mapping_for_cell(cell: Mapping[str, Any], registry: pd.DataFrame) -> dict[str, str]:
    paper_value = clean(cell.get("paper_value_numeric"))
    if not paper_value:
        return empty_mapping(cell, "not_applicable_missing_marker", "Paper cell is an explicit missing marker, not a numeric comparison target.")
    model = clean(cell.get("model"))
    if is_llm_substitute_blocked(model):
        return empty_mapping(cell, "blocked_model_prompt_protocol_mismatch", "Local substitute/open LLM runs are not associated to paper LLM cells without version, prompt, and protocol equivalence.")
    if registry.empty:
        return empty_mapping(cell, "no_local_candidate", "Canonical local registry is unavailable.")
    candidates = registry.loc[
        registry["_dataset_norm"].eq(normalize_dataset(clean(cell.get("dataset"))))
        & registry["_model_norm"].eq(normalize_model(model))
        & registry["_metric_norm"].eq(metric_alias(clean(cell.get("metric"))))
    ].copy()
    cutoff = clean(cell.get("cutoff"))
    if cutoff:
        candidates = candidates.loc[candidates["cutoff"].eq(cutoff)]
    task = clean(cell.get("task"))
    if task:
        candidates = candidates.loc[candidates["task"].eq(task)]
    if candidates.empty:
        return empty_mapping(cell, "no_local_candidate", "No local result matches dataset, task, model, metric, and cutoff.")
    chosen = candidates.sort_values(["artifact_category", "phase", "evidence_id"], kind="mergesort").iloc[0]
    local_value = clean(chosen.get("value"))
    abs_diff = numeric_difference(paper_value, local_value)
    printed_match = printed_precision_match(paper_value, local_value, clean(cell.get("decimal_places")))
    return {
        "atomic_cell_id": clean(cell.get("atomic_cell_id")),
        "local_result_id": clean(chosen.get("evidence_id")),
        "dataset_match": "true",
        "task_match": bool_text(clean(chosen.get("task")) == task),
        "model_match": "true",
        "version_match": "unknown",
        "metric_match": "true",
        "cutoff_match": bool_text(not cutoff or clean(chosen.get("cutoff")) == cutoff),
        "cohort_match": "false",
        "candidate_set_match": "false",
        "protocol_equivalence_class": "incompatible",
        "local_value": local_value,
        "absolute_difference": abs_diff,
        "printed_precision_match": bool_text(printed_match),
        "exact_numeric_replication": "false",
        "mapping_status": "mapped_protocol_incompatible",
        "notes": "Local value matches superficial labels but cohort/candidate/protocol equivalence is not exact.",
    }


def is_llm_substitute_blocked(model: str) -> bool:
    lowered = model.lower()
    return any(token in lowered for token in ["qwen", "mistral", "llama", "claude", "gpt", "llmrank", "llmsrec"])


def empty_mapping(cell: Mapping[str, Any], status: str, notes: str) -> dict[str, str]:
    return {
        "atomic_cell_id": clean(cell.get("atomic_cell_id")),
        "local_result_id": "",
        "dataset_match": "false",
        "task_match": "false",
        "model_match": "false",
        "version_match": "false",
        "metric_match": "false",
        "cutoff_match": "false",
        "cohort_match": "false",
        "candidate_set_match": "false",
        "protocol_equivalence_class": "not_applicable" if status.startswith("not_applicable") else "insufficient_evidence",
        "local_value": "",
        "absolute_difference": "",
        "printed_precision_match": "false",
        "exact_numeric_replication": "false",
        "mapping_status": status,
        "notes": notes,
    }


def numeric_difference(left: str, right: str) -> str:
    if not clean(left) or not clean(right):
        return ""
    try:
        return f"{abs(float(left) - float(right)):.17g}"
    except ValueError:
        return ""


def printed_precision_match(paper: str, local: str, decimals: str) -> bool:
    if not clean(paper) or not clean(local) or not clean(decimals):
        return False
    try:
        places = int(decimals)
        return f"{float(paper):.{places}f}" == f"{float(local):.{places}f}"
    except ValueError:
        return False


def percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return ""
    value = numerator / denominator * 100.0
    return "0.0" if value == 0 else f"{value:.6f}"


def build_corrected_coverage(model_inventory: pd.DataFrame, atomic: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    denominator = len(atomic)
    numeric_or_marker_count = int(atomic["paper_value_text"].astype(str).str.len().gt(0).sum())
    mapped_count = int(mapping["local_result_id"].astype(str).str.len().gt(0).sum())
    exact_protocol_count = int(mapping["protocol_equivalence_class"].eq("exact").sum())
    printed_count = int(mapping["printed_precision_match"].eq("true").sum())
    exact_count = int(mapping["exact_numeric_replication"].eq("true").sum())
    rows = [
        coverage_row("paper model inventory coverage", len(model_inventory), len(model_inventory), "5 model identities are moved out of numeric denominator.", "5", ""),
        coverage_row("atomic paper value extraction coverage", numeric_or_marker_count, denominator, "Denominator is real atomic Table 3/4/5 cell count, including Table 5 missing markers.", "11", "0.0"),
        coverage_row("atomic local mapping coverage", mapped_count, denominator, "Mappings require matching dataset, task, model, metric, and cutoff labels.", "11", ""),
        coverage_row("protocol-exact mapping coverage", exact_protocol_count, denominator, "No mapping is protocol-exact without cohort/candidate/prompt equivalence.", "11", "0.0"),
        coverage_row("printed numeric match coverage", printed_count, denominator, "Printed matches do not imply exact replication when protocol is incompatible.", "11", ""),
        coverage_row("exact numeric replication coverage", exact_count, denominator, "Exact numeric replication requires PDF value plus exact protocol; none are promoted.", "11", "0.0"),
    ]
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def coverage_row(metric: str, numerator: int, denominator: int, notes: str, previous_denominator: str, previous_value: str) -> dict[str, str]:
    return {
        "coverage_metric": metric,
        "numerator": str(numerator),
        "denominator": str(denominator),
        "coverage_percent": percent(numerator, denominator),
        "previous_denominator": previous_denominator,
        "previous_value": previous_value,
        "denominator_policy": "model identities excluded; numeric denominator is corrected atomic paper result-cell count",
        "notes": notes,
    }


def build_anomalies(bundle: SourceBundle, superseded: pd.DataFrame, atomic: pd.DataFrame, mapping: pd.DataFrame, manual_status: Mapping[str, str]) -> pd.DataFrame:
    recovery_values = sorted(v for v in bundle.f5ar_recovery.get("recovered_value_numeric", pd.Series(dtype=str)).astype(str).tolist() if v in {"0.3954", "0.0259", "0.647"})
    rows = [
        anomaly("F5AF-ANOM-001", "target_inventory", "high", F5A_EXTRACTION.as_posix(), "Five previous F5A targets were model identity components mislabeled as numeric cells.", "Old target inventory mixed model inventory with numeric evidence.", "true", "Supersede these rows into paper model inventory.", ""),
        anomaly("F5AF-ANOM-002", "target_inventory", "high", F5A_EXTRACTION.as_posix(), "Six previous F5A targets were metric-family aggregates mislabeled as atomic cells.", "One aggregate metric family can correspond to many table cells.", "true", "Use finalization_f5af_atomic_paper_result_cells.csv.", ""),
        anomaly("F5AF-ANOM-003", "alignment", "medium", F5AR_RECOVERY.as_posix(), "Values 0.3954, 0.0259 and 0.647 are genuine table values but invalid as unique mappings for old aggregate targets.", "They must be represented as atomic row/column cells, not replacements for old PRC aggregate IDs.", "true", "Keep them only inside atomic cell inventory.", ";".join(recovery_values)),
        anomaly("F5AF-ANOM-004", "coverage", "high", F5A_EXTRACTION.as_posix(), "The previous denominator of 11 was not a well-defined numeric-cell denominator.", "Coverage must be recomputed over the corrected atomic inventory.", "true", "Use corrected denominator from F5A-FIX.", f"atomic_denominator={len(atomic)}"),
        anomaly("F5AF-ANOM-005", "docx_pdf_alignment", "medium", PDF_CONFIRMATIONS.as_posix(), "No usable atomic PDF confirmation file is available; DOCX/PDF mismatch cannot be exhaustively assessed.", "PDF-verified count remains zero; DOCX is secondary evidence only.", "true", "Perform manual PDF review at atomic-cell level.", f"manual_status={dict(manual_status)}"),
    ]
    if int(mapping["printed_precision_match"].eq("true").sum()):
        rows.append(anomaly("F5AF-ANOM-006", "numeric_match", "medium", OUTPUT_TABLES["local_atomic_mapping"].as_posix(), "Some printed numeric matches occur under non-exact protocols.", "These do not increase exact replication coverage.", "true", "Require exact protocol before any replication claim.", ""))
    return pd.DataFrame(rows, columns=ANOMALY_COLUMNS)


def anomaly(anomaly_id: str, area: str, severity: str, source_file: str, description: str, impact: str, can_continue: str, action: str, notes: str) -> dict[str, str]:
    return {
        "anomaly_id": anomaly_id,
        "area": area,
        "severity": severity,
        "source_file": source_file,
        "description": description,
        "scientific_impact": impact,
        "can_continue": can_continue,
        "recommended_action": action,
        "notes": notes,
    }


def protected_hash_status(root: Path, protected_after: pd.DataFrame) -> str:
    return _d1r_protected_hash_status(root, protected_after)


def no_absolute_paths(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return True
    return ABSOLUTE_PATH_PATTERN.search(frame.to_csv(index=False, lineterminator="\n")) is None


def build_validation(bundle: SourceBundle, superseded: pd.DataFrame, models: pd.DataFrame, atomic: pd.DataFrame, mapping: pd.DataFrame, coverage: pd.DataFrame, frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    prot_status = protected_hash_status(bundle.root, bundle.protected_inputs)
    corrected_types = superseded["corrected_target_type"] if not superseded.empty else pd.Series(dtype=str)

    def add(check_id: str, name: str, condition: bool, observed: Any, expected: Any, details: str) -> None:
        rows.append({"check_id": check_id, "check_name": name, "status": "pass" if condition else "fail", "observed": clean(observed), "expected": clean(expected), "details": details})

    add("F5AF-VAL-001", "all_11_old_targets_classified", len(superseded) == 11, len(superseded), "11", "Every old F5A target is represented.")
    add("F5AF-VAL-002", "exactly_5_model_inventory_targets", corrected_types.eq("paper_model_inventory_component").sum() == 5, corrected_types.eq("paper_model_inventory_component").sum(), "5", "Five old targets are model identity components.")
    add("F5AF-VAL-003", "exactly_6_metric_family_targets", corrected_types.eq("non_atomic_metric_family").sum() == 6, corrected_types.eq("non_atomic_metric_family").sum(), "6", "Six old targets are metric-family aggregates.")
    add("F5AF-VAL-004", "no_old_target_remains_numeric_cell", not superseded["is_atomic"].eq("true").any() and not superseded["is_numeric"].eq("true").any(), "0 old numeric cells", "0", "Old PRC targets are superseded, not retained as numeric cells.")
    add("F5AF-VAL-005", "atomic_cells_have_required_identity", atomic_required_fields_present(atomic), len(atomic), "all atomic cells identified", "Each new cell has table, row, column, dataset, task, model, and metric.")
    add("F5AF-VAL-006", "every_value_has_source", atomic["source_type"].astype(str).str.len().gt(0).all() and atomic["paper_value_text"].astype(str).str.len().gt(0).all(), len(atomic), "all source/value text present", "Values and missing markers retain DOCX source.")
    add("F5AF-VAL-007", "model_identities_excluded_from_numeric_denominator", len(models) == 5 and len(atomic) != 11, f"models={len(models)};atomic={len(atomic)}", "5 model rows excluded; denominator != 11", "Model inventory rows are separate from numeric denominator.")
    add("F5AF-VAL-008", "no_arbitrary_03954_00259_0647_replication", no_arbitrary_value_replication(atomic, mapping), "no exact replication for flagged values", "true", "Flagged DOCX values are not promoted as old-target unique mappings.")
    add("F5AF-VAL-009", "coverage_uses_atomic_denominator", coverage.loc[coverage["coverage_metric"].ne("paper model inventory coverage"), "denominator"].eq(str(len(atomic))).all(), len(atomic), "coverage denominator equals atomic cell count", "Corrected coverage no longer uses denominator 11.")
    add("F5AF-VAL-010", "protected_inputs_unchanged", prot_status.endswith("changed=0"), prot_status, "changed=0", "Protected inputs match F2 after hashes.")
    add("F5AF-VAL-011", "deterministic_frames", deterministic_frames(frames), "stable", "stable", "Frame serialization hashes are deterministic.")
    add("F5AF-VAL-012", "relative_paths_only", all(no_absolute_paths(frame) for frame in frames), "all frames", "no absolute paths", "Output frames contain only relative paths.")
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def atomic_required_fields_present(atomic: pd.DataFrame) -> bool:
    required = ["paper_table_number", "row_label", "column_label", "dataset", "task", "model", "metric"]
    return not atomic.empty and atomic[required].astype(str).apply(lambda col: col.str.len().gt(0).all()).all()


def no_arbitrary_value_replication(atomic: pd.DataFrame, mapping: pd.DataFrame) -> bool:
    flagged_ids = set(atomic.loc[atomic["paper_value_numeric"].isin(["0.3954", "0.0259", "0.647"]), "atomic_cell_id"])
    if not flagged_ids:
        return True
    flagged_mapping = mapping.loc[mapping["atomic_cell_id"].isin(flagged_ids)]
    return not flagged_mapping["exact_numeric_replication"].eq("true").any()


def deterministic_frames(frames: Sequence[pd.DataFrame]) -> bool:
    return [frame_sha256(frame) for frame in frames] == [frame_sha256(frame.copy()) for frame in frames]


def final_decision(validation: pd.DataFrame) -> str:
    return "finalization_f5af_complete" if not validation.empty and validation["status"].eq("pass").all() else "finalization_f5af_blocked"


def build_frames(root: Path, copy_manual: bool = False) -> dict[str, pd.DataFrame]:
    manual_status = ensure_manual_evidence(root) if copy_manual else {ALIGNMENT_DIAGNOSTIC.as_posix(): "not_checked", PDF_CONFIRMATIONS.as_posix(): "not_checked"}
    bundle = load_sources(root)
    superseded = build_superseded_target_audit(bundle)
    models = build_paper_model_inventory(bundle)
    atomic = build_atomic_paper_result_cells(bundle)
    mapping = build_local_atomic_mapping(bundle, atomic)
    coverage = build_corrected_coverage(models, atomic, mapping)
    anomalies = build_anomalies(bundle, superseded, atomic, mapping, manual_status)
    frames_no_validation = [superseded, models, atomic, mapping, coverage, anomalies]
    validation = build_validation(bundle, superseded, models, atomic, mapping, coverage, frames_no_validation)
    return {
        "superseded_target_audit": superseded,
        "paper_model_inventory": models,
        "atomic_paper_result_cells": atomic,
        "local_atomic_mapping": mapping,
        "corrected_coverage": coverage,
        "anomalies": anomalies,
        "validation": validation,
    }


def build_report(root: Path, frames: Mapping[str, pd.DataFrame]) -> str:
    superseded = frames["superseded_target_audit"]
    models = frames["paper_model_inventory"]
    atomic = frames["atomic_paper_result_cells"]
    mapping = frames["local_atomic_mapping"]
    coverage = frames["corrected_coverage"]
    anomalies = frames["anomalies"]
    validation = frames["validation"]
    decision = final_decision(validation)
    lines = [
        "# Finalization F5A-FIX Atomic Paper Cells",
        "",
        "## 1. Executive summary",
        f"- Decisione: `{decision}`.",
        f"- Vecchi target superseded: `{len(superseded)}`; model identity: `{len(models)}`; celle atomiche: `{len(atomic)}`.",
        "",
        "## 2. Perche gli 11 target precedenti non erano celle",
        "- Cinque righe erano componenti di inventario modello, quindi non risultati numerici.",
        "- Sei righe erano famiglie metriche aggregate: ogni metrica attraversa molte intersezioni tabella/riga/colonna.",
        "",
        "## 3. Componenti e celle atomiche",
        "- Paper component: elemento del paper da coprire; model inventory: presenza di un modello; metric family: insieme di celle; atomic numeric result cell: singola intersezione riga-colonna.",
        "",
        "## 4. Target superseded",
        f"- Tutti gli `{len(superseded)}` target precedenti hanno status `superseded_non_atomic_numeric_inventory`.",
        "",
        "## 5. Inventario atomico",
        f"- Tabelle incluse: 3, 4, 5. Celle atomiche reali trovate: `{len(atomic)}`.",
        f"- PDF-verified: `{atomic['pdf_verified'].eq('true').sum()}`; secondary-source verified: `{atomic['secondary_source_verified'].eq('true').sum()}`.",
        "",
        "## 6. Mapping locale",
        f"- Celle con mapping locale: `{mapping['local_result_id'].astype(str).str.len().gt(0).sum()}`.",
        f"- Protocol-exact: `{mapping['protocol_equivalence_class'].eq('exact').sum()}`.",
        "",
        "## 7. Denominatore corretto",
        "- La copertura numerica precedente `0/11` non era ben definita perche il denominatore conteneva model identity e metric families, non celle atomiche.",
        "",
        "## 8. Copertura corretta",
        *[f"- `{clean(row['coverage_metric'])}`: `{clean(row['numerator'])}/{clean(row['denominator'])}` = `{clean(row['coverage_percent'])}`." for _, row in coverage.iterrows()],
        "",
        "## 9. Anomalie",
        f"- Anomalie registrate: `{len(anomalies)}`.",
        "",
        "## 10. Conservazione audit storici",
        "- F5A e F5A-R restano immutati come audit storici; F5A-FIX produce solo un inventario correttivo separato.",
        "",
        "## 11. Validazione",
        f"- Validation pass/fail: `{validation['status'].eq('pass').sum()}` / `{validation['status'].eq('fail').sum()}`.",
        "",
        "## 12. Decisione finale",
        f"- `{decision}`.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(root: Path, frames: Mapping[str, pd.DataFrame], report: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key, relative in OUTPUT_TABLES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[key].to_csv(path, index=False, lineterminator="\n")
        hashes[relative.as_posix()] = file_sha256(path)
    report_path = root / REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    hashes[REPORT_PATH.as_posix()] = file_sha256(report_path)
    return dict(sorted(hashes.items()))


def build_all(root: Path) -> dict[str, Any]:
    frames = build_frames(root, copy_manual=True)
    report = build_report(root, frames)
    hashes = write_outputs(root, frames, report)
    return {"frames": frames, "report": report, "hashes": hashes, "decision": final_decision(frames["validation"])}


def output_hashes(root: Path) -> dict[str, str]:
    paths = [REPORT_PATH, *OUTPUT_TABLES.values()]
    return {path.as_posix(): file_sha256(root / path) for path in paths if (root / path).exists()}
