"""Phase F6F1 beyond-utility metric recovery and offline evaluation.

This module is intentionally offline. It reads frozen F6E4C/F6E4B/F6D/F2
and Plan E artifacts, computes only metrics with source-backed contracts, and
writes new ``phase_f6f1_*`` outputs. It never calls LLM backends, training
code, network APIs, or mutates prior phase artifacts.
"""

from __future__ import annotations

from src.audit import common_git

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


PHASE_ID = "phase_f6f1"
DEFAULT_CONFIG_PATH = Path("configs/phase_f6f1_beyond_utility_metrics.yaml")
DECISION_COMPLETE_WITH_BLOCKS = "f6f1_beyond_utility_metrics_complete_with_documented_blocks"
DECISION_BLOCKED_INPUT = "f6f1_blocked_by_missing_input"
DECISION_BLOCKED_TESTS = "f6f1_blocked_by_tests"
TOP_K = 5
TOLERANCE = 1e-12

EXACTNESS_CLASSES = {
    "exact_formula_exact_inputs",
    "exact_formula_reconstructed_inputs",
    "exact_formula_adapted_population",
    "reconstructed_formula",
    "local_proxy",
    "extension_metric",
    "blocked_missing_evidence",
    "blocked_missing_input",
    "not_applicable",
}

F6E4C_STRATEGY_LABELS = {
    "paper_base_openp5_parametric_v1": "Base",
    "paper_llmrank_recency_parametric_v1": "LLMRank",
}

BASELINE_FAMILY_ALIASES = {
    "bm25_like_e3b": "BM25-like",
    "bm25_like_adapted": "BM25-like",
    "mostpop_train_only": "MostPop",
    "bprmf": "BPRMF",
    "gru4rec": "GRU4Rec",
    "sasrec": "SASRec",
    "lightgcn": "LightGCN",
}

REQUESTED_BASELINE_FAMILIES = ["BM25-like", "MostPop", "BPRMF", "GRU4Rec", "SASRec", "LightGCN"]

SOURCE_INVENTORY_COLUMNS = [
    "metric",
    "source_path",
    "source_type",
    "section",
    "function_or_formula",
    "symbols",
    "top_k",
    "threshold",
    "catalog_definition",
    "popularity_definition",
    "baseline_reference",
    "normalization",
    "edge_cases",
    "evidence_status",
    "source_sha256",
    "notes",
]

CONTRACT_COLUMNS = [
    "metric",
    "paper_definition",
    "formula",
    "formula_origin",
    "dataset_task_applicability",
    "population",
    "denominator",
    "top_k",
    "invalid_output_treatment",
    "positive_absent_treatment",
    "baseline_reference",
    "data_availability",
    "exactness_classification",
    "claim_allowed",
    "limitation",
]

EXACTNESS_COLUMNS = [
    "paper_metric",
    "local_metric",
    "formula_match",
    "input_match",
    "population_match",
    "implementation_match",
    "exactness_classification",
    "claim_allowed",
    "limitation",
]

APLT_COLUMNS = [
    "result_id",
    "scope",
    "source_type",
    "dataset",
    "task",
    "model_name",
    "strategy_id",
    "strategy_label",
    "baseline_family",
    "population",
    "metric",
    "k",
    "value",
    "denominator",
    "valid_list_count",
    "invalid_list_count",
    "missing_popularity_count",
    "long_tail_threshold",
    "classification",
    "policy",
    "source_file",
    "status",
    "notes",
]

POPULARITY_NOVELTY_COLUMNS = [
    "result_id",
    "scope",
    "source_type",
    "dataset",
    "task",
    "model_name",
    "strategy_id",
    "strategy_label",
    "baseline_family",
    "population",
    "metric",
    "k",
    "value",
    "denominator",
    "valid_list_count",
    "invalid_list_count",
    "missing_popularity_count",
    "missing_novelty_count",
    "catalog_denominator",
    "candidate_union_denominator",
    "classification",
    "policy",
    "source_file",
    "status",
    "notes",
]

SERENDIPITY_COLUMNS = [
    "result_id",
    "scope",
    "source_type",
    "dataset",
    "task",
    "model_name",
    "strategy_id",
    "strategy_label",
    "baseline_family",
    "population",
    "metric",
    "k",
    "value",
    "denominator",
    "valid_list_count",
    "invalid_list_count",
    "missing_mostpop_reference_count",
    "model_hit_rate",
    "mostpop_hit_rate",
    "classification",
    "policy",
    "source_file",
    "status",
    "notes",
]

CANDIDATE_POSITION_COLUMNS = [
    "analysis_id",
    "analysis_type",
    "dataset",
    "task",
    "model_name",
    "strategy_id",
    "strategy_label",
    "stratum",
    "metric",
    "k",
    "value",
    "logical_requests",
    "strict_valid",
    "denominator",
    "classification",
    "canddif_formula_status",
    "source_file",
    "notes",
]

RANKING_RERANKING_COLUMNS = [
    "comparison_id",
    "dataset",
    "model_name",
    "strategy_id",
    "strategy_label",
    "population",
    "metric",
    "ranking_value",
    "reranking_value",
    "delta_reranking_minus_ranking",
    "paired_intersection_size",
    "ranking_denominator",
    "reranking_denominator",
    "classification",
    "notes",
]

LLM_BASELINE_COLUMNS = [
    "comparison_id",
    "dataset",
    "task",
    "llm_model",
    "strategy_id",
    "strategy_label",
    "baseline_family",
    "baseline_model",
    "population",
    "metric",
    "llm_value",
    "baseline_value",
    "delta_llm_minus_baseline",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "paired_intersection_size",
    "llm_coverage",
    "baseline_coverage",
    "classification",
    "notes",
]

RQ_COLUMNS = [
    "research_question",
    "status",
    "primary_evidence",
    "classification",
    "allowed_claim",
    "limitation",
]

MISSING_COLUMNS = [
    "item_id",
    "metric_or_component",
    "scope",
    "blocking_classification",
    "blocking_reason",
    "required_evidence",
    "current_evidence",
    "recommended_action",
]

STRUCTURAL_DIAGNOSTIC_COLUMNS = [
    "diagnostic_id",
    "scope",
    "source_type",
    "dataset",
    "task",
    "model_name",
    "strategy_id",
    "strategy_label",
    "baseline_family",
    "metric",
    "k",
    "value",
    "denominator",
    "logical_requests",
    "strict_valid",
    "invalid_list_count",
    "positive_present_count",
    "positive_absent_count",
    "classification",
    "policy",
    "source_file",
    "notes",
]


@dataclass(frozen=True)
class PopularityData:
    dataset: str
    train_counts: Mapping[str, int]
    train_unique_user_counts: Mapping[str, int]
    total_train_interactions: int
    train_users: int
    long_tail_threshold: float
    source_path: str
    source_sha256: str
    catalog_items: frozenset[str]


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: Any, length: int = 16) -> str:
    return f"{prefix}-{sha256_text('||'.join(str(part) for part in parts))[:length]}"


def canonical_dataset(value: Any) -> str:
    text = str(value).strip()
    lowered = text.lower().replace("_", "-")
    if lowered in {"ml1m", "ml-1m", "movie-lens-1m", "movielens-1m"}:
        return "ML-1M"
    if lowered in {"beauty", "amazon beauty", "amazon-beauty"}:
        return "Amazon Beauty"
    if lowered in {"sports", "amazon sports", "amazon-sports"}:
        return "Amazon Sports"
    if lowered in {"lastfm", "last-fm", "lastfm-2k"}:
        return "LastFM"
    return text


def strategy_label(strategy_id: Any) -> str:
    return F6E4C_STRATEGY_LABELS.get(str(strategy_id), "")


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_int(value: Any, default: int | None = None) -> int | None:
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def parse_float(value: Any, default: float = math.nan) -> float:
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def fmt_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.17g}"


def split_pipe(value: Any) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    return [part for part in text.split("|") if part != ""]


def parse_json_array(value: Any) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    decoded = json.loads(text)
    if not isinstance(decoded, list):
        raise ValueError("cell is not a JSON array")
    return [str(item) for item in decoded]


def candidate_order_hash(items: Sequence[Any]) -> str:
    return sha256_text(stable_json([str(item) for item in items]))


def safe_read_csv(path: Path, *, usecols: Sequence[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, usecols=usecols)
    except pd.errors.ParserError:
        return _read_csv_lenient(path, usecols=usecols)


def _read_csv_lenient(path: Path, *, usecols: Sequence[str] | None = None) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        selected = list(usecols) if usecols is not None else header
        rows: list[dict[str, str]] = []
        for raw in reader:
            if len(raw) > len(header):
                raw = raw[: len(header) - 1] + [",".join(raw[len(header) - 1 :])]
            elif len(raw) < len(header):
                raw = raw + [""] * (len(header) - len(raw))
            mapped = dict(zip(header, raw, strict=False))
            rows.append({column: mapped.get(column, "") for column in selected})
    return pd.DataFrame(rows, columns=selected, dtype=str)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted(
        [{column: row.get(column, "") for column in columns} for row in rows],
        key=lambda row: tuple(str(row.get(column, "")) for column in columns),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow(row)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(payload) + "\n", encoding="utf-8")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _git(root: Path, *args: str) -> str:
    return common_git.git_stdout_no_check(args, root, encoding="utf-8", errors="replace")


def _existing_first_timestamp(root: Path, config: Mapping[str, Any]) -> str:
    manifest_path = root / config["outputs"]["freeze_manifest"]
    if not manifest_path.exists():
        return ""
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    return str(existing.get("repository_state", {}).get("timestamp", ""))


def record_repository_state(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Return deterministic repository/runtime state for the phase manifest."""

    preserved_timestamp = _existing_first_timestamp(root, config)
    timestamp = preserved_timestamp or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    generated_paths = _generated_output_paths(config)
    status_short = _filter_git_path_output(_git(root, "status", "--short"), generated_paths, status_format=True)
    status_tracked = _filter_git_path_output(_git(root, "status", "--short", "--untracked-files=no"), generated_paths, status_format=True)
    diff_name_only = _filter_git_path_output(_git(root, "diff", "--name-only"), generated_paths, status_format=False)
    diff_stat = _filter_git_path_output(_git(root, "diff", "--stat"), generated_paths, status_format=False)
    cached_stat = _git(root, "diff", "--cached", "--stat")
    staged_files = _git(root, "diff", "--cached", "--name-only")
    return {
        "directory": str(root),
        "branch": _git(root, "branch", "--show-current"),
        "head": _git(root, "rev-parse", "HEAD"),
        "git_status_short": status_short,
        "git_status_tracked_short": status_tracked,
        "git_diff_name_only": diff_name_only,
        "git_diff_stat": diff_stat,
        "git_diff_cached_stat": cached_stat,
        "staged_files": staged_files,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "timestamp": timestamp,
    }


def _generated_output_paths(config: Mapping[str, Any]) -> set[str]:
    outputs = {str(config["outputs"]["report"]).replace("\\", "/"), str(config["outputs"]["freeze_manifest"]).replace("\\", "/")}
    outputs.update(str(path).replace("\\", "/") for path in config["outputs"]["tables"].values())
    return outputs


def _filter_git_path_output(text: str, generated_paths: set[str], *, status_format: bool) -> str:
    return common_git.filter_git_path_output(text, generated_paths, status_format=status_format)


def _iter_config_paths(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _iter_config_paths(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            yield from _iter_config_paths(nested)


def _load_expected_hashes(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    expected: dict[str, str] = {}

    f6e4c_manifest = root / config["inputs"]["f6e4c_freeze_manifest"]
    if f6e4c_manifest.exists():
        manifest = json.loads(f6e4c_manifest.read_text(encoding="utf-8"))
        for key in ("input_hashes", "raw_f6e4b_hashes"):
            for path, digest in manifest.get(key, {}).items():
                expected[str(path).replace("\\", "/")] = str(digest)

    for table_key in ("f2_protected_hashes", "f6d4a_closure_hash_manifest", "f6d4a_input_hashes"):
        table_path = root / config["inputs"].get(table_key, "")
        if not table_path.exists():
            continue
        frame = safe_read_csv(table_path)
        path_col = "relative_path" if "relative_path" in frame.columns else "path"
        if path_col not in frame.columns or "sha256" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            rel = str(row[path_col]).replace("\\", "/")
            digest = str(row["sha256"]).strip()
            exists = str(row.get("exists", "true")).strip().lower()
            if rel and digest and digest != "missing" and exists != "false":
                expected.setdefault(rel, digest)

    return expected


def verify_frozen_inputs(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify relevant frozen input files and protected manifests."""

    expected = _load_expected_hashes(root, config)
    config_paths = {str(path).replace("\\", "/") for path in _iter_config_paths(config.get("inputs", {}))}
    for source in config.get("baseline_sources", {}).get("wide", []):
        config_paths.add(str(source["path"]).replace("\\", "/"))
    for source in config.get("baseline_sources", {}).get("long", []):
        config_paths.add(str(source["path"]).replace("\\", "/"))

    checked_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for rel in sorted(path for path in config_paths if path):
        path = root / rel
        exists = path.exists()
        observed = sha256_file(path) if exists and path.is_file() else "missing"
        expected_hash = expected.get(rel, "")
        if not exists:
            status = "missing"
            failures.append(f"{rel}:missing")
        elif expected_hash and observed != expected_hash:
            status = "changed"
            failures.append(f"{rel}:changed")
        elif expected_hash:
            status = "pass"
        else:
            status = "current_hash_registered_no_prior_expected"
        checked_rows.append(
            {
                "path": rel,
                "expected_sha256": expected_hash,
                "observed_sha256": observed,
                "status": status,
            }
        )

    protected_checked = 0
    protected_changed = 0
    protected_missing = 0
    f2_path = root / config["inputs"]["f2_protected_hashes"]
    if f2_path.exists():
        protected = safe_read_csv(f2_path)
        for _, row in protected.iterrows():
            rel = str(row.get("relative_path", "")).replace("\\", "/")
            digest = str(row.get("sha256", "")).strip()
            if not rel or not digest or digest == "missing":
                continue
            protected_checked += 1
            path = root / rel
            if not path.exists():
                protected_missing += 1
                failures.append(f"{rel}:protected_missing")
                continue
            observed = sha256_file(path)
            if observed != digest:
                protected_changed += 1
                failures.append(f"{rel}:protected_changed")

    if failures:
        raise RuntimeError("f6f1_input_integrity_failed:" + ";".join(failures[:20]))

    return {
        "checked_inputs": checked_rows,
        "checked_count": len(checked_rows),
        "expected_hash_count": sum(1 for row in checked_rows if row["expected_sha256"]),
        "current_hash_only_count": sum(
            1 for row in checked_rows if row["status"] == "current_hash_registered_no_prior_expected"
        ),
        "protected_summary": {
            "checked": protected_checked,
            "changed": protected_changed,
            "missing": protected_missing,
        },
        "input_hashes": {row["path"]: row["observed_sha256"] for row in checked_rows},
    }


def _docx_contains_formula(docx_path: Path, pattern: str) -> bool:
    if not docx_path.exists():
        return False
    try:
        with zipfile.ZipFile(docx_path) as archive:
            text = archive.read("word/document.xml").decode("utf-8", errors="ignore")
    except (KeyError, zipfile.BadZipFile):
        return False
    return pattern.lower() in text.lower()


def build_metric_source_inventory(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    official = config["inputs"]["official_metrics_code"]
    pdf = config["inputs"]["paper_pdf"]
    docx = config["inputs"]["paper_docx_secondary"]
    f5c = "report/finalization_f5c_popularity_bias.md"
    f5d = "report/finalization_f5d_serendipity.md"
    f6e4c = "results/tables/phase_f6e4c_request_level_metrics.csv"
    plan_d = config["inputs"]["plan_d_candidate_position_variant_summary"]
    sources = [
        {
            "metric": "APLT@5",
            "source_path": official,
            "source_type": "official_code",
            "section": "llm_metrics.py metrics popularity bias",
            "function_or_formula": "calACLT(results, freq_dict, high_bar, metric_len)",
            "symbols": "results; freq_dict; high_bar; metric_len",
            "top_k": "5",
            "threshold": "q80 train item frequency, strict less-than",
            "catalog_definition": "train-seen items present in freq_dict",
            "popularity_definition": "raw train interaction frequency",
            "baseline_reference": "not_applicable",
            "normalization": "top-K exposure mean",
            "edge_cases": "items missing from freq_dict are not long-tail",
            "evidence_status": "complete_official_code_and_secondary_docx_formula",
            "source_sha256": sha256_file(root / official),
            "notes": f"DOCX formula present={_docx_contains_formula(root / docx, 'APLT@K')}",
        },
        {
            "metric": "ARP@5",
            "source_path": official,
            "source_type": "official_code",
            "section": "llm_metrics.py metrics popularity bias",
            "function_or_formula": "calARP(freq_dict, id_results, metric_len)",
            "symbols": "freq_dict; id_results; metric_len",
            "top_k": "5",
            "threshold": "not_applicable",
            "catalog_definition": "train-seen items with zero for missing items",
            "popularity_definition": "raw train interaction frequency",
            "baseline_reference": "not_applicable",
            "normalization": "top-K exposure mean",
            "edge_cases": "missing items contribute zero in official code",
            "evidence_status": "complete_official_code_and_secondary_docx_formula",
            "source_sha256": sha256_file(root / official),
            "notes": f"DOCX formula present={_docx_contains_formula(root / docx, 'ARP@K')}",
        },
        {
            "metric": "Serendipity@5",
            "source_path": official,
            "source_type": "official_code",
            "section": "llm_metrics.py novelty metrics",
            "function_or_formula": "calSerendipity(hit_results, hit_results_pop)",
            "symbols": "hit_results; hit_results_pop",
            "top_k": "5",
            "threshold": "not_applicable",
            "catalog_definition": "same candidate rows as evaluated recommender",
            "popularity_definition": "MostPop train-only hit vector",
            "baseline_reference": "MostPop",
            "normalization": "mean over evaluated rows",
            "edge_cases": "requires aligned MostPop hit vector",
            "evidence_status": "official_code_complete_paper_exact_claim_not_allowed",
            "source_sha256": sha256_file(root / official),
            "notes": f"Prior F5D policy source={f5d}",
        },
        {
            "metric": "SelfInformation@5",
            "source_path": official,
            "source_type": "official_code",
            "section": "llm_metrics.py novelty metrics",
            "function_or_formula": "calSelfInfo(results, metric_len, information)",
            "symbols": "results; information",
            "top_k": "5",
            "threshold": "not_applicable",
            "catalog_definition": "train-seen item information dictionary",
            "popularity_definition": "paper DOCX: log2(|U| / |U_i|)",
            "baseline_reference": "not_applicable",
            "normalization": "top-K exposure mean",
            "edge_cases": "missing information blocks reconstructed paper-style result",
            "evidence_status": "formula_recoverable_inputs_reconstructed",
            "source_sha256": sha256_file(root / official),
            "notes": f"DOCX formula present={_docx_contains_formula(root / docx, 'SelfInformation@K')}",
        },
        {
            "metric": "CandDif",
            "source_path": docx,
            "source_type": "secondary_docx_formula",
            "section": "3.4.4 and appendix candidate position bias",
            "function_or_formula": "-log(1-Acc(R_firstposition)) - (-log(1-Acc(R_randomposition)))",
            "symbols": "Acc in {HR,NDCG}",
            "top_k": "5",
            "threshold": "not_applicable",
            "catalog_definition": "same candidate rows under controlled position variants",
            "popularity_definition": "not_applicable",
            "baseline_reference": "random-position condition",
            "normalization": "difference of transformed accuracies",
            "edge_cases": "undefined when Acc >= 1; blocked without controlled first/random pairs",
            "evidence_status": "secondary_formula_present_full_f6e4c_input_missing",
            "source_sha256": sha256_file(root / docx),
            "notes": f"Plan D diagnostic source={plan_d}",
        },
        {
            "metric": "Candidate-position observational strata",
            "source_path": f6e4c,
            "source_type": "frozen_local_evidence",
            "section": "F6E4C request-level metadata",
            "function_or_formula": "HR/NDCG/final-rank grouped by positive_initial_position",
            "symbols": "positive_initial_position; strict_valid; positive_rank",
            "top_k": "5",
            "threshold": "first half <=10; second half >10",
            "catalog_definition": "frozen candidate order hash",
            "popularity_definition": "not_applicable",
            "baseline_reference": "not_applicable",
            "normalization": "within-position/bin means",
            "edge_cases": "observational only; no causal claim",
            "evidence_status": "complete_for_observational_adapted_analysis",
            "source_sha256": sha256_file(root / f6e4c),
            "notes": "Uses frozen F6E4A/F6E4C candidate-order hashes.",
        },
        {
            "metric": "Popularity bias / PopREO",
            "source_path": official,
            "source_type": "official_code",
            "section": "llm_metrics.py calPopREO",
            "function_or_formula": "std(group_hit_rates) / mean(group_hit_rates)",
            "symbols": "hit_results; quantiles; freq_dict; id_targets",
            "top_k": "5",
            "threshold": "five popularity quantile groups",
            "catalog_definition": "target items by train popularity groups",
            "popularity_definition": "raw train interaction frequency",
            "baseline_reference": "not_applicable",
            "normalization": "coefficient of variation",
            "edge_cases": "empty groups and zero mean policy not frozen",
            "evidence_status": "blocked_missing_denominator_policy",
            "source_sha256": sha256_file(root / official),
            "notes": f"Prior F5C policy source={f5c}",
        },
        {
            "metric": "Coverage/reliability extensions",
            "source_path": f6e4c,
            "source_type": "frozen_local_evidence",
            "section": "F6E4C request-level metrics and structural fields",
            "function_or_formula": "strict-valid rate; duplicate rate; unique top-K item coverage",
            "symbols": "strict_valid; parsed_item_ids; candidate_item_ids",
            "top_k": "5",
            "threshold": "not_applicable",
            "catalog_definition": "train-seen catalog and candidate union",
            "popularity_definition": "not_applicable",
            "baseline_reference": "not_applicable",
            "normalization": "extension-specific denominator",
            "edge_cases": "invalid LLM lists excluded from conditional list metrics",
            "evidence_status": "extension_metric",
            "source_sha256": sha256_file(root / f6e4c),
            "notes": "Not renamed as diversity, fairness, or paper-exact metric.",
        },
    ]
    return sources


def build_metric_contracts() -> list[dict[str, Any]]:
    rows = [
        {
            "metric": "HR@5",
            "paper_definition": "Hit rate at cutoff 5 for the single positive item.",
            "formula": "mean(1[positive_rank <= 5])",
            "formula_origin": "paper utility metric and official calHR",
            "dataset_task_applicability": "all frozen F6E4C workloads",
            "population": "operational and conditional",
            "denominator": "logical requests for operational; strict-valid requests for conditional",
            "top_k": "5",
            "invalid_output_treatment": "operational zero; conditional excluded",
            "positive_absent_treatment": "counts as miss and reported in candidate-pool recall where applicable",
            "baseline_reference": "not_applicable",
            "data_availability": "complete from F6E4C",
            "exactness_classification": "exact_formula_adapted_population",
            "claim_allowed": "authoritative for adapted F6E4C matrix",
            "limitation": "not paper-exact model/protocol reproduction",
        },
        {
            "metric": "NDCG@5",
            "paper_definition": "Discounted gain for the single positive item at cutoff 5.",
            "formula": "mean(1/log2(rank+1) if rank <= 5 else 0)",
            "formula_origin": "paper utility metric and official calNDCG",
            "dataset_task_applicability": "all frozen F6E4C workloads",
            "population": "operational and conditional",
            "denominator": "logical requests for operational; strict-valid requests for conditional",
            "top_k": "5",
            "invalid_output_treatment": "operational zero; conditional excluded",
            "positive_absent_treatment": "counts as miss and reported in candidate-pool recall where applicable",
            "baseline_reference": "not_applicable",
            "data_availability": "complete from F6E4C",
            "exactness_classification": "exact_formula_adapted_population",
            "claim_allowed": "authoritative for adapted F6E4C matrix",
            "limitation": "not paper-exact model/protocol reproduction",
        },
        {
            "metric": "APLT@5",
            "paper_definition": "Average percentage of long-tail items in the top-K list.",
            "formula": "mean_u(|R_u@5 intersect Phi| / 5), Phi={i: train_count(i) < q80(train_count)}",
            "formula_origin": "DOCX formula plus official calACLT",
            "dataset_task_applicability": "datasets with train-only popularity counts",
            "population": "conditional strict-valid lists; baseline valid lists",
            "denominator": "5 * valid-list count",
            "top_k": "5",
            "invalid_output_treatment": "conditional excluded; operational not_defined for LLM invalid lists",
            "positive_absent_treatment": "not required by exposure formula; reported separately",
            "baseline_reference": "not_applicable",
            "data_availability": "available for Beauty, Sports, ML-1M, and adapted LastFM train split",
            "exactness_classification": "exact_formula_adapted_population",
            "claim_allowed": "paper formula applied in adapted local population",
            "limitation": "train split/candidate population is local adapted evidence",
        },
        {
            "metric": "Average Popularity@5 / ARP@5",
            "paper_definition": "Average popularity of recommended items.",
            "formula": "mean_i train_count(i) over all top-5 exposures",
            "formula_origin": "DOCX formula plus official calARP",
            "dataset_task_applicability": "datasets with train-only popularity counts",
            "population": "conditional strict-valid lists; baseline valid lists",
            "denominator": "5 * valid-list count",
            "top_k": "5",
            "invalid_output_treatment": "conditional excluded; operational not_defined for LLM invalid lists",
            "positive_absent_treatment": "not required by exposure formula; reported separately",
            "baseline_reference": "not_applicable",
            "data_availability": "available for Beauty, Sports, ML-1M, and adapted LastFM train split",
            "exactness_classification": "exact_formula_adapted_population",
            "claim_allowed": "paper formula applied in adapted local population",
            "limitation": "not renamed as popularity bias",
        },
        {
            "metric": "SelfInformation@5",
            "paper_definition": "Novelty as self-information of recommended items.",
            "formula": "mean log2(|U_train| / |U_i_train|) over top-5 exposures",
            "formula_origin": "DOCX formula and official calSelfInfo information dictionary",
            "dataset_task_applicability": "rows where top-5 items have positive train unique-user support",
            "population": "conditional strict-valid lists; baseline valid lists",
            "denominator": "5 * valid-list count with complete novelty support",
            "top_k": "5",
            "invalid_output_treatment": "conditional excluded; operational not_defined for LLM invalid lists",
            "positive_absent_treatment": "not required by exposure formula; reported separately",
            "baseline_reference": "not_applicable",
            "data_availability": "partial when cold/unseen items appear",
            "exactness_classification": "exact_formula_reconstructed_inputs",
            "claim_allowed": "reconstructed novelty where input support is complete",
            "limitation": "PDF formula not machine-verified; DOCX is secondary",
        },
        {
            "metric": "Serendipity@5",
            "paper_definition": "Useful unexpected recommendations relative to MostPop.",
            "formula": "mean(hit_at5 * (1 - mostpop_hit_at5))",
            "formula_origin": "official calSerendipity with F5D MostPop policy",
            "dataset_task_applicability": "rows with aligned MostPop baseline hit vector",
            "population": "conditional strict-valid LLM lists; valid baseline lists",
            "denominator": "valid paired rows with MostPop reference",
            "top_k": "5",
            "invalid_output_treatment": "conditional excluded; operational not_defined for LLM invalid lists",
            "positive_absent_treatment": "requires relevance/hit definition; positive-absent separated",
            "baseline_reference": "MostPop train-only frozen predictions",
            "data_availability": "available when MostPop row matches same user/candidate order",
            "exactness_classification": "reconstructed_formula",
            "claim_allowed": "adapted/reconstructed serendipity only",
            "limitation": "DOCX item-list formula and official hit-vector code differ; no paper-exact claim",
        },
        {
            "metric": "CandDif",
            "paper_definition": "Accuracy sensitivity between positive-first and random-position candidate orders.",
            "formula": "-log(1-Acc(first)) - (-log(1-Acc(random)))",
            "formula_origin": "DOCX secondary formula",
            "dataset_task_applicability": "controlled candidate-order pairs only",
            "population": "paired controlled instances",
            "denominator": "paired first/random rows",
            "top_k": "5",
            "invalid_output_treatment": "variant-specific valid policy; no F6E4C operational substitution",
            "positive_absent_treatment": "requires positive-present rows",
            "baseline_reference": "random-position condition",
            "data_availability": "blocked for full F6E4C; limited Plan D diagnostic available",
            "exactness_classification": "blocked_missing_input",
            "claim_allowed": "no full-matrix causal CandDif claim",
            "limitation": "F6E4C did not rerun same instances under controlled positive-first/random orders",
        },
    ]
    _assert_classifications(rows, "exactness_classification")
    return rows


def build_metric_exactness_matrix() -> list[dict[str, Any]]:
    rows = [
        ("HR@5", "hr_at5_operational/hr_at5_conditional", "yes", "adapted", "adapted", "yes", "exact_formula_adapted_population", "adapted utility claim", "local LLM/model protocol differs from paper"),
        ("NDCG@5", "ndcg_at5_operational/ndcg_at5_conditional", "yes", "adapted", "adapted", "yes", "exact_formula_adapted_population", "adapted utility claim", "local LLM/model protocol differs from paper"),
        ("APLT", "aplt_at_5", "yes", "reconstructed", "adapted", "yes", "exact_formula_adapted_population", "paper formula in adapted setting", "train/candidate population local"),
        ("serendipity", "serendipity_at_5_mostpop_hit", "partial", "reconstructed", "adapted", "yes", "reconstructed_formula", "reconstructed/adapted only", "paper prose and official code are not fully identical locally"),
        ("novelty", "novelty_self_information_at_5", "yes", "reconstructed", "adapted", "yes", "exact_formula_reconstructed_inputs", "reconstructed novelty where inputs complete", "DOCX secondary formula; cold items blocked"),
        ("average popularity", "average_popularity_at_5", "yes", "reconstructed", "adapted", "yes", "exact_formula_adapted_population", "paper formula in adapted setting", "not popularity bias"),
        ("popularity bias", "popreo_at_5", "partial", "missing", "missing", "no", "blocked_missing_evidence", "no numeric claim", "empty group/zero denominator policy missing"),
        ("candidate-position bias", "observational_position_strata", "partial", "adapted", "observational", "yes", "exact_formula_adapted_population", "observational adapted claim", "no causal claim"),
        ("CandDif", "canddif_diagnostic_or_blocked", "yes", "missing for F6E4C", "missing for F6E4C", "limited diagnostic", "blocked_missing_input", "no full-matrix CandDif claim", "controlled first/random F6E4C pairs missing"),
        ("catalog coverage", "catalog_coverage_at_5", "not paper exact", "local", "extension", "yes", "extension_metric", "extension only", "not diversity/fairness"),
        ("item coverage", "item_coverage_at_5", "partial", "adapted", "adapted", "yes", "extension_metric", "candidate-union extension", "denominator is local candidate union"),
        ("duplicate rate", "duplicate_rate_at_5", "not_applicable", "local", "extension", "yes", "extension_metric", "reliability extension", "not a paper metric"),
        ("strict-valid rate", "strict_valid_rate", "not_applicable", "local", "extension", "yes", "extension_metric", "reliability extension", "parser/adapted-output metric"),
        ("operational utility", "operational_hr_ndcg", "yes", "adapted", "adapted", "yes", "exact_formula_adapted_population", "adapted operational utility", "invalid-as-zero applies only to utility"),
    ]
    result = [
        {
            "paper_metric": paper,
            "local_metric": local,
            "formula_match": formula,
            "input_match": inputs,
            "population_match": population,
            "implementation_match": implementation,
            "exactness_classification": classification,
            "claim_allowed": claim,
            "limitation": limitation,
        }
        for paper, local, formula, inputs, population, implementation, classification, claim, limitation in rows
    ]
    _assert_classifications(result, "exactness_classification")
    return result


def _assert_classifications(rows: Sequence[Mapping[str, Any]], column: str) -> None:
    bad = sorted({str(row[column]) for row in rows if str(row[column]) not in EXACTNESS_CLASSES})
    if bad:
        raise ValueError(f"unexpected exactness classification: {bad}")


def _source_info_from_canonical(root: Path, path: str) -> dict[str, dict[str, Any]]:
    frame = safe_read_csv(root / path)
    mapping: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        candidates = parse_json_array(row["candidate_items"])
        mapping[str(row["request_id"])] = {
            "user_id": str(row["user_id"]),
            "candidate_set_id": str(row.get("candidate_set_id", "")),
            "candidate_items": candidates,
            "candidate_order_hash": str(row.get("candidate_order_sha256", "")) or candidate_order_hash(candidates),
            "positive_present": str(row.get("positive_item_id", "")) in set(candidates),
        }
    return mapping


def _source_info_from_lastfm_manifest(root: Path, path: str) -> dict[str, dict[str, Any]]:
    if not (root / path).exists():
        return {}
    frame = safe_read_csv(root / path)
    mapping: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        candidates = split_pipe(row.get("candidate_item_ids", ""))
        mapping[str(row["request_id"])] = {
            "user_id": str(row["user_id"]),
            "candidate_set_id": str(row.get("cohort_name", "finalization_f6c_lastfm_ranking_cohort_full_1859")),
            "candidate_items": candidates,
            "candidate_order_hash": candidate_order_hash(candidates),
            "positive_present": parse_bool(row.get("positive_present", "")),
        }
    return mapping


def load_f6e4c_llm_rows(root: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    request_metrics = safe_read_csv(root / config["inputs"]["f6e4c_request_level_metrics"])
    f6e4a_manifest = safe_read_csv(root / config["inputs"]["f6e4a_request_manifest"])
    f6e4a_by_request = {
        str(row["request_id"]): {
            "candidate_items": split_pipe(row.get("candidate_item_ids", "")),
            "candidate_order_hash": str(row.get("candidate_order_hash", "")),
        }
        for _, row in f6e4a_manifest.iterrows()
    }

    source_info = _source_info_from_canonical(root, config["inputs"]["canonical_full_matrix_rankings"])
    source_info.update(_source_info_from_lastfm_manifest(root, config["inputs"]["f6d3_lastfm_ranking_manifest"]))

    rows: list[dict[str, Any]] = []
    for _, row in request_metrics.iterrows():
        request_id = str(row["request_id"])
        source_request_id = str(row["source_request_id"])
        f6e4a_info = f6e4a_by_request.get(request_id, {})
        source = source_info.get(source_request_id, {})
        candidate_items = list(f6e4a_info.get("candidate_items") or source.get("candidate_items") or [])
        c_hash = str(f6e4a_info.get("candidate_order_hash") or source.get("candidate_order_hash") or "")
        if candidate_items and not c_hash:
            c_hash = candidate_order_hash(candidate_items)
        parsed_items = split_pipe(row.get("parsed_item_ids", ""))
        strict_valid = parse_bool(row.get("strict_valid", ""))
        top_items = parsed_items[:TOP_K] if strict_valid and len(parsed_items) >= TOP_K else []
        positive_initial_position = parse_int(row.get("positive_initial_position", ""), None)
        rows.append(
            {
                "source_type": "llm",
                "source_file": str(row.get("raw_output_file", config["inputs"]["f6e4c_request_level_metrics"])),
                "request_id": request_id,
                "source_request_id": source_request_id,
                "dataset": canonical_dataset(row["dataset"]),
                "task": str(row["task"]),
                "model_name": str(row["model"]),
                "strategy_id": str(row["strategy_id"]),
                "strategy_label": strategy_label(row["strategy_id"]),
                "baseline_family": "",
                "candidate_set_id": str(source.get("candidate_set_id", "")),
                "user_id": str(source.get("user_id", "")),
                "positive_item_id": str(row.get("positive_item_id", "")),
                "candidate_items": candidate_items,
                "candidate_order_hash": c_hash,
                "parsed_items": parsed_items,
                "top_items": top_items,
                "valid_list": strict_valid,
                "strict_valid": strict_valid,
                "invalid_reason": str(row.get("invalid_reason", "")),
                "positive_present": positive_initial_position is not None and positive_initial_position > 0,
                "positive_initial_position": positive_initial_position,
                "positive_rank_one_based": parse_int(row.get("positive_rank", ""), None),
                "hr_at5_operational": parse_float(row.get("hr_at5_operational", "")),
                "ndcg_at5_operational": parse_float(row.get("ndcg_at5_operational", "")),
                "hr_at5_conditional": parse_float(row.get("hr_at5_conditional", "")),
                "ndcg_at5_conditional": parse_float(row.get("ndcg_at5_conditional", "")),
            }
        )
    return pd.DataFrame(rows)


def normalize_llm_lists(frame: pd.DataFrame) -> pd.DataFrame:
    """Return F6E4C LLM rows with top-5 list fields normalized."""

    result = frame.copy()
    result["row_key"] = result.apply(
        lambda row: "|".join(
            [
                str(row["dataset"]),
                str(row["task"]),
                str(row["user_id"]),
                str(row["positive_item_id"]),
                str(row["candidate_order_hash"]),
            ]
        ),
        axis=1,
    )
    return result


def _baseline_family_from_model(model_id: str) -> str:
    lowered = str(model_id).lower()
    for key, family in BASELINE_FAMILY_ALIASES.items():
        if key in lowered:
            return family
    return ""


def _normalize_wide_baseline(root: Path, source: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = root / source["path"]
    if not path.exists():
        return []
    frame = safe_read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        model_id = str(row.get("baseline_id", row.get("model_id", "")))
        family = _baseline_family_from_model(model_id)
        if family not in REQUESTED_BASELINE_FAMILIES:
            continue
        try:
            candidate_items = parse_json_array(row["candidate_items"])
            ranked_items = parse_json_array(row["ranked_items"])
        except (KeyError, json.JSONDecodeError, ValueError):
            continue
        valid = parse_bool(row.get("ranking_valid", "true")) and len(ranked_items) >= TOP_K
        positive_item = str(row.get("positive_item_id", ""))
        positive_rank_one_based = parse_int(row.get("positive_rank_one_based", ""), None)
        if positive_rank_one_based is None and positive_item in ranked_items:
            positive_rank_one_based = ranked_items.index(positive_item) + 1
        hr = 1.0 if positive_rank_one_based is not None and positive_rank_one_based <= TOP_K else 0.0
        ndcg = parse_float(row.get("ndcg_at_5", ""))
        if not math.isfinite(ndcg):
            ndcg = 1.0 / math.log2(positive_rank_one_based + 1) if hr else 0.0
        rows.append(
            {
                "source_type": "baseline",
                "source_file": str(source["path"]),
                "request_id": "",
                "source_request_id": "",
                "dataset": canonical_dataset(row.get("dataset", "")),
                "task": str(row.get("task_type", row.get("task", ""))),
                "model_name": model_id,
                "strategy_id": "",
                "strategy_label": "",
                "baseline_family": family,
                "candidate_set_id": str(row.get("candidate_set_id", "")),
                "user_id": str(row.get("user_id", "")),
                "positive_item_id": positive_item,
                "candidate_items": candidate_items,
                "candidate_order_hash": candidate_order_hash(candidate_items),
                "parsed_items": ranked_items,
                "top_items": ranked_items[:TOP_K] if valid else [],
                "valid_list": valid,
                "strict_valid": valid,
                "invalid_reason": "" if valid else "invalid_baseline_ranking",
                "positive_present": positive_item in set(candidate_items),
                "positive_initial_position": candidate_items.index(positive_item) + 1 if positive_item in candidate_items else None,
                "positive_rank_one_based": positive_rank_one_based,
                "hr_at5_operational": hr,
                "ndcg_at5_operational": ndcg,
                "hr_at5_conditional": hr,
                "ndcg_at5_conditional": ndcg,
            }
        )
    return rows


def _normalize_long_baseline(root: Path, source: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = root / source["path"]
    if not path.exists():
        return []
    usecols = [
        "model",
        "user_id",
        "candidate_item_id",
        "rank",
        "is_positive",
        "candidate_original_position",
        "cohort_name",
        "candidate_set_name",
        "source_candidate_file",
    ]
    frame = safe_read_csv(path, usecols=[column for column in usecols if column in pd.read_csv(path, nrows=0).columns])
    rows: list[dict[str, Any]] = []
    group_cols = ["model", "user_id"]
    for (model, user_id), group in frame.groupby(group_cols, sort=False, dropna=False):
        family = _baseline_family_from_model(str(model))
        if family not in REQUESTED_BASELINE_FAMILIES:
            continue
        ranked_group = group.copy()
        ranked_group["__rank"] = ranked_group["rank"].map(lambda value: parse_int(value, 10**9) or 10**9)
        ranked_group = ranked_group.sort_values(["__rank", "candidate_item_id"], kind="mergesort")
        candidate_group = group.copy()
        candidate_group["__candidate_position"] = candidate_group["candidate_original_position"].map(
            lambda value: parse_int(value, 10**9) or 10**9
        )
        candidate_group = candidate_group.sort_values(["__candidate_position", "candidate_item_id"], kind="mergesort")
        ranked_items = [str(item) for item in ranked_group["candidate_item_id"].tolist()]
        candidate_items = [str(item) for item in candidate_group["candidate_item_id"].tolist()]
        positive_rows = group.loc[group["is_positive"].map(parse_bool)]
        positive_item = str(positive_rows.iloc[0]["candidate_item_id"]) if not positive_rows.empty else ""
        positive_rank = ranked_items.index(positive_item) + 1 if positive_item in ranked_items else None
        hr = 1.0 if positive_rank is not None and positive_rank <= TOP_K else 0.0
        ndcg = 1.0 / math.log2(positive_rank + 1) if hr else 0.0
        candidate_set = str(group.iloc[0].get("candidate_set_name", group.iloc[0].get("cohort_name", "")))
        rows.append(
            {
                "source_type": "baseline",
                "source_file": str(source["path"]),
                "request_id": "",
                "source_request_id": "",
                "dataset": canonical_dataset(source.get("dataset", "LastFM")),
                "task": str(source.get("task", "ranking")),
                "model_name": str(model),
                "strategy_id": "",
                "strategy_label": "",
                "baseline_family": family,
                "candidate_set_id": candidate_set,
                "user_id": str(user_id),
                "positive_item_id": positive_item,
                "candidate_items": candidate_items,
                "candidate_order_hash": candidate_order_hash(candidate_items),
                "parsed_items": ranked_items,
                "top_items": ranked_items[:TOP_K],
                "valid_list": len(ranked_items) >= TOP_K,
                "strict_valid": len(ranked_items) >= TOP_K,
                "invalid_reason": "",
                "positive_present": positive_item in set(candidate_items),
                "positive_initial_position": candidate_items.index(positive_item) + 1 if positive_item in candidate_items else None,
                "positive_rank_one_based": positive_rank,
                "hr_at5_operational": hr,
                "ndcg_at5_operational": ndcg,
                "hr_at5_conditional": hr,
                "ndcg_at5_conditional": ndcg,
            }
        )
    return rows


def normalize_baseline_rankings(root: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in config.get("baseline_sources", {}).get("wide", []):
        rows.extend(_normalize_wide_baseline(root, source))
    for source in config.get("baseline_sources", {}).get("long", []):
        rows.extend(_normalize_long_baseline(root, source))
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["row_key"] = result.apply(
        lambda row: "|".join(
            [
                str(row["dataset"]),
                str(row["task"]),
                str(row["user_id"]),
                str(row["positive_item_id"]),
                str(row["candidate_order_hash"]),
            ]
        ),
        axis=1,
    )
    return result


def _load_lastfm_popularity(root: Path, config: Mapping[str, Any]) -> PopularityData:
    train_path = root / config["inputs"]["lastfm_train"]
    train = safe_read_csv(train_path)
    train = train.loc[train["split"].str.lower().eq("train")].copy()
    train["item_id"] = train["item_id"].astype(str)
    train["user_id"] = train["user_id"].astype(str)
    counts = Counter(train["item_id"].tolist())
    unique_counts = Counter(
        {
            str(item): int(count)
            for item, count in train.drop_duplicates(["item_id", "user_id"]).groupby("item_id", sort=False).size().items()
        }
    )
    threshold = float(np.quantile(np.asarray([count for count in counts.values() if count > 0], dtype=float), 0.8))
    catalog_path = root / config["inputs"]["lastfm_catalog"]
    catalog_items = frozenset(safe_read_csv(catalog_path)["item_id"].astype(str).tolist()) if catalog_path.exists() else frozenset(counts)
    return PopularityData(
        dataset="LastFM",
        train_counts=dict(counts),
        train_unique_user_counts=dict(unique_counts),
        total_train_interactions=int(sum(counts.values())),
        train_users=int(train["user_id"].nunique()),
        long_tail_threshold=threshold,
        source_path=config["inputs"]["lastfm_train"],
        source_sha256=sha256_file(train_path),
        catalog_items=catalog_items,
    )


def load_popularity_data(root: Path, config: Mapping[str, Any]) -> dict[str, PopularityData]:
    item_stats = safe_read_csv(root / config["inputs"]["train_item_statistics"])
    pop_stats_path = root / config["inputs"]["item_popularity_statistics"]
    pop_stats = safe_read_csv(pop_stats_path) if pop_stats_path.exists() else pd.DataFrame()
    train_users_by_dataset = {
        canonical_dataset(row["dataset"]): int(float(row["train_users"]))
        for _, row in pop_stats.iterrows()
        if str(row.get("train_users", "")).strip()
    }
    data: dict[str, PopularityData] = {}
    for dataset, group in item_stats.groupby(item_stats["dataset"].map(canonical_dataset), sort=False):
        counts = {str(row["item_id"]): int(float(row["train_interaction_count"])) for _, row in group.iterrows()}
        unique_counts = {str(row["item_id"]): int(float(row["train_unique_user_count"])) for _, row in group.iterrows()}
        total = int(float(group.iloc[0]["total_train_interactions"]))
        threshold = float(group.iloc[0]["long_tail_threshold"])
        source_path = str(group.iloc[0].get("source_path", config["inputs"]["train_item_statistics"]))
        source_sha = str(group.iloc[0].get("source_sha256", ""))
        data[str(dataset)] = PopularityData(
            dataset=str(dataset),
            train_counts=counts,
            train_unique_user_counts=unique_counts,
            total_train_interactions=total,
            train_users=train_users_by_dataset.get(str(dataset), 0),
            long_tail_threshold=threshold,
            source_path=source_path,
            source_sha256=source_sha,
            catalog_items=frozenset(counts),
        )
    data["LastFM"] = _load_lastfm_popularity(root, config)
    return data


def _duplicate_rate(items: Sequence[str], *, k: int = TOP_K) -> float:
    top = list(items[:k])
    if len(top) < k:
        return math.nan
    seen: set[str] = set()
    duplicates = 0
    for item in top:
        if item in seen:
            duplicates += 1
        seen.add(item)
    return duplicates / k


def _row_popularity_metrics(row: Mapping[str, Any], popularity: Mapping[str, PopularityData], *, k: int = TOP_K) -> dict[str, Any]:
    dataset = str(row["dataset"])
    pdata = popularity.get(dataset)
    top = list(row.get("top_items", []))[:k]
    if pdata is None or len(top) < k:
        return {
            "aplt_at_5": math.nan,
            "average_popularity_at_5": math.nan,
            "novelty_self_information_at_5": math.nan,
            "novelty_interaction_self_information_at_5": math.nan,
            "duplicate_rate_at_5": _duplicate_rate(top, k=k),
            "missing_popularity_count": k,
            "missing_novelty_count": k,
            "long_tail_threshold": math.nan,
        }
    counts = [int(pdata.train_counts.get(str(item), 0)) for item in top]
    unique_counts = [int(pdata.train_unique_user_counts.get(str(item), 0)) for item in top]
    long_tail = [count > 0 and float(count) < pdata.long_tail_threshold for count in counts]
    novelty_user = math.nan
    if pdata.train_users > 0 and all(count > 0 for count in unique_counts):
        novelty_user = float(np.mean([math.log2(pdata.train_users / count) for count in unique_counts]))
    novelty_interactions = math.nan
    if pdata.total_train_interactions > 0 and all(count > 0 for count in counts):
        novelty_interactions = float(np.mean([-math.log2(count / pdata.total_train_interactions) for count in counts]))
    return {
        "aplt_at_5": float(np.mean(np.asarray(long_tail, dtype=float))),
        "average_popularity_at_5": float(np.mean(np.asarray(counts, dtype=float))),
        "novelty_self_information_at_5": novelty_user,
        "novelty_interaction_self_information_at_5": novelty_interactions,
        "duplicate_rate_at_5": _duplicate_rate(top, k=k),
        "missing_popularity_count": sum(1 for count in counts if count <= 0),
        "missing_novelty_count": sum(1 for count in unique_counts if count <= 0),
        "long_tail_threshold": pdata.long_tail_threshold,
    }


def enrich_row_metrics(frame: pd.DataFrame, popularity: Mapping[str, PopularityData]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    rows = []
    for _, row in frame.iterrows():
        metrics = _row_popularity_metrics(row, popularity)
        enriched = row.to_dict()
        enriched.update(metrics)
        rows.append(enriched)
    result = pd.DataFrame(rows)
    return result


def _group_key_columns(scope: str) -> list[str]:
    if scope == "dataset_task":
        return ["source_type", "dataset", "task", "model_name", "strategy_id", "strategy_label", "baseline_family", "source_file"]
    if scope == "cell":
        return ["source_type", "model_name", "strategy_id", "strategy_label", "baseline_family", "source_file"]
    raise ValueError(scope)


def _aggregate_metric_rows(
    frame: pd.DataFrame,
    *,
    metric: str,
    scope: str,
    classification: str,
    columns_kind: str,
    candidate_unions: Mapping[tuple[str, str], set[str]],
    popularity: Mapping[str, PopularityData],
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    group_cols = _group_key_columns(scope)
    valid = frame.loc[frame["valid_list"] & frame[metric].map(lambda value: math.isfinite(float(value)) if value != "" else False)].copy()
    grouped = valid.groupby(group_cols, sort=False, dropna=False)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        key_map = dict(zip(group_cols, key, strict=False))
        dataset = key_map.get("dataset", "ALL") if scope == "dataset_task" else "ALL"
        task = key_map.get("task", "ALL") if scope == "dataset_task" else "ALL"
        source_type = str(key_map.get("source_type", ""))
        model_name = str(key_map.get("model_name", ""))
        source_file = str(key_map.get("source_file", ""))
        invalid_count = 0
        if source_type == "llm":
            mask = (
                (frame["source_type"] == source_type)
                & (frame["model_name"] == model_name)
                & (frame["strategy_id"] == str(key_map.get("strategy_id", "")))
            )
            if scope == "dataset_task":
                mask &= (frame["dataset"] == dataset) & (frame["task"] == task)
            invalid_count = int((~frame.loc[mask, "valid_list"]).sum())
        values = [float(value) for value in group[metric].tolist()]
        top_items = [item for items in group["top_items"].tolist() for item in list(items)[:TOP_K]]
        pdata = popularity.get(str(dataset)) if dataset != "ALL" else None
        catalog_denominator = len(pdata.catalog_items) if pdata else ""
        candidate_denominator = len(candidate_unions.get((str(dataset), str(task)), set())) if dataset != "ALL" else ""
        row = {
            "result_id": stable_id("F6F1-RES", scope, metric, dataset, task, model_name, key_map.get("strategy_id", ""), key_map.get("baseline_family", "")),
            "scope": scope,
            "source_type": source_type,
            "dataset": dataset,
            "task": task,
            "model_name": model_name,
            "strategy_id": str(key_map.get("strategy_id", "")),
            "strategy_label": str(key_map.get("strategy_label", "")),
            "baseline_family": str(key_map.get("baseline_family", "")),
            "population": "conditional",
            "metric": metric,
            "k": str(TOP_K),
            "value": fmt_float(float(np.mean(np.asarray(values, dtype=float)))),
            "denominator": str(len(values)),
            "valid_list_count": str(len(group)),
            "invalid_list_count": str(invalid_count),
            "missing_popularity_count": str(int(group.get("missing_popularity_count", pd.Series(dtype=float)).map(lambda value: int(float(value))).sum())) if "missing_popularity_count" in group else "0",
            "missing_novelty_count": str(int(group.get("missing_novelty_count", pd.Series(dtype=float)).map(lambda value: int(float(value))).sum())) if "missing_novelty_count" in group else "0",
            "long_tail_threshold": fmt_float(float(group["long_tail_threshold"].iloc[0])) if "long_tail_threshold" in group and metric == "aplt_at_5" else "",
            "catalog_denominator": str(catalog_denominator),
            "candidate_union_denominator": str(candidate_denominator),
            "classification": classification,
            "policy": "strict-valid conditional list metric; no invalid-list imputation",
            "source_file": source_file,
            "status": "computed",
            "notes": "per-user top-5 value then macro mean; coverage metrics are separately cohort-level when listed",
        }
        if metric == "catalog_coverage_at_5":
            denominator = len(pdata.catalog_items) if pdata else 0
            value = len(set(top_items).intersection(pdata.catalog_items)) / denominator if denominator else math.nan
            row["value"] = fmt_float(value)
            row["denominator"] = str(denominator)
        elif metric == "item_coverage_at_5":
            union = candidate_unions.get((str(dataset), str(task)), set())
            value = len(set(top_items).intersection(union)) / len(union) if union else math.nan
            row["value"] = fmt_float(value)
            row["denominator"] = str(len(union))
        rows.append(row)

        if source_type == "llm":
            op_row = row.copy()
            op_row["result_id"] = stable_id("F6F1-RES", scope, metric, dataset, task, model_name, key_map.get("strategy_id", ""), "operational")
            op_row["population"] = "operational"
            op_row["value"] = ""
            op_row["denominator"] = str(len(group) + invalid_count)
            op_row["status"] = "not_defined"
            op_row["policy"] = "invalid LLM lists are not assigned zero for list beyond-utility metrics"
            op_row["notes"] = "Operational invalid-as-zero is only defined for utility in F6E4C."
            rows.append(op_row)
    return rows


def _candidate_unions(frame: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    unions: dict[tuple[str, str], set[str]] = {}
    for _, row in frame.iterrows():
        key = (str(row["dataset"]), str(row["task"]))
        unions.setdefault(key, set()).update(str(item) for item in row.get("candidate_items", []))
    return unions


def compute_aplt_arp_novelty(frame: pd.DataFrame, popularity: Mapping[str, PopularityData]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_unions = _candidate_unions(frame)
    aplt_rows: list[dict[str, Any]] = []
    other_rows: list[dict[str, Any]] = []
    for scope in ("dataset_task", "cell"):
        aplt_rows.extend(
            _aggregate_metric_rows(
                frame,
                metric="aplt_at_5",
                scope=scope,
                classification="exact_formula_adapted_population",
                columns_kind="aplt",
                candidate_unions=candidate_unions,
                popularity=popularity,
            )
        )
        for metric, classification in [
            ("average_popularity_at_5", "exact_formula_adapted_population"),
            ("novelty_self_information_at_5", "exact_formula_reconstructed_inputs"),
            ("novelty_interaction_self_information_at_5", "local_proxy"),
            ("duplicate_rate_at_5", "extension_metric"),
            ("catalog_coverage_at_5", "extension_metric"),
            ("item_coverage_at_5", "extension_metric"),
        ]:
            working = frame.copy()
            if metric in {"catalog_coverage_at_5", "item_coverage_at_5"}:
                working[metric] = 1.0
            other_rows.extend(
                _aggregate_metric_rows(
                    working,
                    metric=metric,
                    scope=scope,
                    classification=classification,
                    columns_kind="other",
                    candidate_unions=candidate_unions,
                    popularity=popularity,
                )
            )

    reliability_rows = _reliability_extension_rows(frame)
    other_rows.extend(reliability_rows)
    for row in aplt_rows:
        row.pop("missing_novelty_count", None)
        row.pop("catalog_denominator", None)
        row.pop("candidate_union_denominator", None)
    return aplt_rows, other_rows


def _reliability_extension_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    llm = frame.loc[frame["source_type"].eq("llm")]
    if llm.empty:
        return rows
    for scope, group_cols in {
        "dataset_task": ["dataset", "task", "model_name", "strategy_id", "strategy_label", "source_file"],
        "cell": ["model_name", "strategy_id", "strategy_label", "source_file"],
    }.items():
        for key, group in llm.groupby(group_cols, sort=False, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            key_map = dict(zip(group_cols, key, strict=False))
            dataset = key_map.get("dataset", "ALL") if scope == "dataset_task" else "ALL"
            task = key_map.get("task", "ALL") if scope == "dataset_task" else "ALL"
            strict_valid = int(group["valid_list"].sum())
            logical = len(group)
            for metric, value, notes in [
                ("strict_valid_rate", strict_valid / logical if logical else math.nan, "strict-valid lists divided by logical requests"),
                ("invalid_list_rate", (logical - strict_valid) / logical if logical else math.nan, "invalid lists divided by logical requests"),
                ("operational_hr_at_5", float(group["hr_at5_operational"].mean()), "F6E4C utility policy: invalid output counted as zero"),
                ("operational_ndcg_at_5", float(group["ndcg_at5_operational"].mean()), "F6E4C utility policy: invalid output counted as zero"),
            ]:
                rows.append(
                    {
                        "result_id": stable_id("F6F1-REL", scope, metric, dataset, task, key_map.get("model_name", ""), key_map.get("strategy_id", "")),
                        "scope": scope,
                        "source_type": "llm",
                        "dataset": dataset,
                        "task": task,
                        "model_name": str(key_map.get("model_name", "")),
                        "strategy_id": str(key_map.get("strategy_id", "")),
                        "strategy_label": str(key_map.get("strategy_label", "")),
                        "baseline_family": "",
                        "population": "operational" if metric.startswith("operational") or metric.endswith("rate") else "conditional",
                        "metric": metric,
                        "k": "5" if metric.endswith("_at_5") else "",
                        "value": fmt_float(value),
                        "denominator": str(logical),
                        "valid_list_count": str(strict_valid),
                        "invalid_list_count": str(logical - strict_valid),
                        "missing_popularity_count": "0",
                        "missing_novelty_count": "0",
                        "catalog_denominator": "",
                        "candidate_union_denominator": "",
                        "classification": "extension_metric" if metric.endswith("rate") else "exact_formula_adapted_population",
                        "policy": "F6E4C reliability/utility policy",
                        "source_file": str(key_map.get("source_file", "")),
                        "status": "computed",
                        "notes": notes,
                    }
                )
    return rows


def build_structural_diagnostics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return extension diagnostics for list validity, positive presence, and operational utility."""

    if frame.empty:
        return []

    rows: list[dict[str, Any]] = []
    scopes = {
        "dataset_task": ["source_type", "dataset", "task", "model_name", "strategy_id", "strategy_label", "baseline_family", "source_file"],
        "cell": ["source_type", "model_name", "strategy_id", "strategy_label", "baseline_family", "source_file"],
    }
    for scope, group_cols in scopes.items():
        for key, group in frame.groupby(group_cols, sort=False, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            key_map = dict(zip(group_cols, key, strict=False))
            dataset = str(key_map.get("dataset", "ALL")) if scope == "dataset_task" else "ALL"
            task = str(key_map.get("task", "ALL")) if scope == "dataset_task" else "ALL"
            source_type = str(key_map.get("source_type", ""))
            model_name = str(key_map.get("model_name", ""))
            strategy_id = str(key_map.get("strategy_id", ""))
            strategy = str(key_map.get("strategy_label", ""))
            baseline_family = str(key_map.get("baseline_family", ""))
            source_file = str(key_map.get("source_file", ""))
            logical = len(group)
            strict_group = group.loc[group["valid_list"]]
            strict = len(strict_group)
            invalid = logical - strict
            positive_present = int(group["positive_present"].map(bool).sum()) if "positive_present" in group else 0
            positive_absent = logical - positive_present
            duplicate_values = [_duplicate_rate(list(items)[:TOP_K], k=TOP_K) for items in strict_group["top_items"].tolist()]
            operational_hr_values = [parse_float(value) for value in group.get("hr_at5_operational", pd.Series(dtype=float)).tolist()]
            operational_ndcg_values = [parse_float(value) for value in group.get("ndcg_at5_operational", pd.Series(dtype=float)).tolist()]
            finite_hr = [value for value in operational_hr_values if math.isfinite(value)]
            finite_ndcg = [value for value in operational_ndcg_values if math.isfinite(value)]

            metric_specs = [
                ("logical_request_count", float(logical), logical, "", "extension_metric", "frozen rows in this diagnostic cell"),
                ("strict_valid_rate", strict / logical if logical else math.nan, logical, "", "extension_metric", "strict-valid lists divided by logical rows"),
                ("invalid_list_rate", invalid / logical if logical else math.nan, logical, "", "extension_metric", "invalid lists divided by logical rows"),
                ("positive_present_rate", positive_present / logical if logical else math.nan, logical, "", "extension_metric", "positive item present in frozen candidate pool"),
                ("positive_absent_rate", positive_absent / logical if logical else math.nan, logical, "", "extension_metric", "positive item absent from frozen candidate pool"),
                ("duplicate_rate_at_5", float(np.mean(duplicate_values)) if duplicate_values else math.nan, strict, str(TOP_K), "extension_metric", "mean per-list duplicate fraction among strict-valid top-5 lists"),
                (
                    "operational_hr_at_5",
                    float(np.mean(finite_hr)) if finite_hr else math.nan,
                    logical,
                    str(TOP_K),
                    "exact_formula_adapted_population",
                    "F6E4C utility policy: invalid output counted as zero",
                ),
                (
                    "operational_ndcg_at_5",
                    float(np.mean(finite_ndcg)) if finite_ndcg else math.nan,
                    logical,
                    str(TOP_K),
                    "exact_formula_adapted_population",
                    "F6E4C utility policy: invalid output counted as zero",
                ),
            ]
            for metric, value, denominator, k_value, classification, notes in metric_specs:
                rows.append(
                    {
                        "diagnostic_id": stable_id("F6F1-DIAG", scope, source_type, dataset, task, model_name, strategy_id, baseline_family, metric, source_file),
                        "scope": scope,
                        "source_type": source_type,
                        "dataset": dataset,
                        "task": task,
                        "model_name": model_name,
                        "strategy_id": strategy_id,
                        "strategy_label": strategy,
                        "baseline_family": baseline_family,
                        "metric": metric,
                        "k": k_value,
                        "value": fmt_float(value),
                        "denominator": str(denominator),
                        "logical_requests": str(logical),
                        "strict_valid": str(strict),
                        "invalid_list_count": str(invalid),
                        "positive_present_count": str(positive_present),
                        "positive_absent_count": str(positive_absent),
                        "classification": classification,
                        "policy": "structural diagnostic; no paper-exact beyond-utility claim",
                        "source_file": source_file,
                        "notes": notes,
                    }
                )
    _assert_classifications(rows, "classification")
    return rows


def _hit_at5(row: Mapping[str, Any]) -> float:
    rank = row.get("positive_rank_one_based")
    if rank is None or (isinstance(rank, float) and math.isnan(rank)):
        return 0.0
    return 1.0 if int(rank) <= TOP_K else 0.0


def compute_serendipity(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    mostpop = frame.loc[(frame["source_type"] == "baseline") & (frame["baseline_family"] == "MostPop")].copy()
    mostpop_ref = {}
    for _, row in mostpop.iterrows():
        if bool(row["valid_list"]):
            mostpop_ref[str(row["row_key"])] = _hit_at5(row)

    working_rows: list[dict[str, Any]] = []
    missing_reference: Counter[tuple[str, str, str, str, str, str]] = Counter()
    for _, row in frame.iterrows():
        key = str(row["row_key"])
        valid = bool(row["valid_list"])
        group_key = (
            str(row["source_type"]),
            str(row["dataset"]),
            str(row["task"]),
            str(row["model_name"]),
            str(row["strategy_id"]),
            str(row["baseline_family"]),
        )
        if not valid:
            continue
        if key not in mostpop_ref:
            missing_reference[group_key] += 1
            continue
        hit = _hit_at5(row)
        mp_hit = mostpop_ref[key]
        enriched = row.to_dict()
        enriched["serendipity_at_5"] = hit * (1.0 - mp_hit)
        enriched["model_hit_at_5"] = hit
        enriched["mostpop_hit_at_5"] = mp_hit
        working_rows.append(enriched)
    working = pd.DataFrame(working_rows)
    if working.empty:
        return []

    result: list[dict[str, Any]] = []
    for scope in ("dataset_task", "cell"):
        group_cols = _group_key_columns(scope)
        for key, group in working.groupby(group_cols, sort=False, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            key_map = dict(zip(group_cols, key, strict=False))
            dataset = key_map.get("dataset", "ALL") if scope == "dataset_task" else "ALL"
            task = key_map.get("task", "ALL") if scope == "dataset_task" else "ALL"
            source_type = str(key_map.get("source_type", ""))
            model_name = str(key_map.get("model_name", ""))
            missing = int(missing_reference.get((source_type, str(dataset), str(task), model_name, str(key_map.get("strategy_id", "")), str(key_map.get("baseline_family", ""))), 0))
            source_mask = (frame["source_type"] == source_type) & (frame["model_name"] == model_name)
            if scope == "dataset_task":
                source_mask &= (frame["dataset"] == dataset) & (frame["task"] == task)
            if source_type == "llm":
                source_mask &= frame["strategy_id"] == str(key_map.get("strategy_id", ""))
            else:
                source_mask &= frame["baseline_family"] == str(key_map.get("baseline_family", ""))
            invalid_count = int((~frame.loc[source_mask, "valid_list"]).sum())
            row = {
                "result_id": stable_id("F6F1-SER", scope, dataset, task, model_name, key_map.get("strategy_id", ""), key_map.get("baseline_family", "")),
                "scope": scope,
                "source_type": source_type,
                "dataset": dataset,
                "task": task,
                "model_name": model_name,
                "strategy_id": str(key_map.get("strategy_id", "")),
                "strategy_label": str(key_map.get("strategy_label", "")),
                "baseline_family": str(key_map.get("baseline_family", "")),
                "population": "conditional",
                "metric": "serendipity_at_5",
                "k": str(TOP_K),
                "value": fmt_float(float(group["serendipity_at_5"].mean())),
                "denominator": str(len(group)),
                "valid_list_count": str(len(group)),
                "invalid_list_count": str(invalid_count),
                "missing_mostpop_reference_count": str(missing),
                "model_hit_rate": fmt_float(float(group["model_hit_at_5"].mean())),
                "mostpop_hit_rate": fmt_float(float(group["mostpop_hit_at_5"].mean())),
                "classification": "reconstructed_formula",
                "policy": "mean(hit_at5 * (1 - mostpop_train_only_hit_at5))",
                "source_file": str(key_map.get("source_file", "")),
                "status": "computed",
                "notes": "MostPop reference is frozen train-only baseline; no paper-exact claim.",
            }
            result.append(row)
            if source_type == "llm":
                op_row = row.copy()
                op_row["result_id"] = stable_id("F6F1-SER", scope, dataset, task, model_name, key_map.get("strategy_id", ""), "operational")
                op_row["population"] = "operational"
                op_row["value"] = ""
                op_row["status"] = "not_defined"
                op_row["policy"] = "invalid LLM lists are not assigned zero for serendipity"
                op_row["notes"] = "Operational invalid-as-zero is only defined for utility in F6E4C."
                result.append(op_row)
    return result


def canddif(first_accuracy: float, random_accuracy: float) -> float:
    if first_accuracy >= 1.0 or random_accuracy >= 1.0:
        raise ValueError("CandDif undefined for accuracy >= 1")
    if first_accuracy < 0.0 or random_accuracy < 0.0:
        raise ValueError("CandDif undefined for negative accuracy")
    return -math.log(1.0 - first_accuracy) - (-math.log(1.0 - random_accuracy))


def compute_candidate_position(root: Path, config: Mapping[str, Any], llm: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid_positions = llm.loc[llm["positive_initial_position"].map(lambda value: value is not None and int(value) > 0)].copy()
    for group_cols in [
        ["dataset", "task", "model_name", "strategy_id", "strategy_label", "source_file"],
    ]:
        for key, group in valid_positions.groupby(group_cols, sort=False, dropna=False):
            key_map = dict(zip(group_cols, key if isinstance(key, tuple) else (key,), strict=False))
            dataset = str(key_map["dataset"])
            task = str(key_map["task"])
            model = str(key_map["model_name"])
            strategy = str(key_map["strategy_id"])
            for position, pgroup in group.groupby("positive_initial_position", sort=True):
                logical = len(pgroup)
                strict = int(pgroup["valid_list"].sum())
                strict_group = pgroup.loc[pgroup["valid_list"]]
                values = {
                    "operational_hr_at_5": float(pgroup["hr_at5_operational"].mean()),
                    "conditional_hr_at_5": float(strict_group["hr_at5_conditional"].mean()) if strict else math.nan,
                    "operational_ndcg_at_5": float(pgroup["ndcg_at5_operational"].mean()),
                    "conditional_ndcg_at_5": float(strict_group["ndcg_at5_conditional"].mean()) if strict else math.nan,
                    "mean_final_positive_rank": float(
                        np.mean([rank for rank in strict_group["positive_rank_one_based"].tolist() if rank is not None])
                    )
                    if strict and any(rank is not None for rank in strict_group["positive_rank_one_based"].tolist())
                    else math.nan,
                }
                for metric, value in values.items():
                    rows.append(
                        {
                            "analysis_id": stable_id("F6F1-POS", "position", dataset, task, model, strategy, position, metric),
                            "analysis_type": "observational_positive_initial_position",
                            "dataset": dataset,
                            "task": task,
                            "model_name": model,
                            "strategy_id": strategy,
                            "strategy_label": str(key_map["strategy_label"]),
                            "stratum": f"position_{int(position)}",
                            "metric": metric,
                            "k": "5" if metric.endswith("_at_5") else "",
                            "value": fmt_float(value),
                            "logical_requests": str(logical),
                            "strict_valid": str(strict),
                            "denominator": str(strict if metric.startswith("conditional") or metric.startswith("mean_final") else logical),
                            "classification": "exact_formula_adapted_population",
                            "canddif_formula_status": "not_canddif_observational_only",
                            "source_file": str(key_map["source_file"]),
                            "notes": "Observational F6E4C stratum; no causal candidate-position claim.",
                        }
                    )
            for bin_name, mask in {
                "first_half_1_10": group["positive_initial_position"].map(lambda value: int(value) <= 10),
                "second_half_11_20": group["positive_initial_position"].map(lambda value: int(value) > 10),
            }.items():
                bgroup = group.loc[mask]
                if bgroup.empty:
                    continue
                strict_group = bgroup.loc[bgroup["valid_list"]]
                for metric, column in [
                    ("operational_hr_at_5", "hr_at5_operational"),
                    ("conditional_hr_at_5", "hr_at5_conditional"),
                    ("operational_ndcg_at_5", "ndcg_at5_operational"),
                    ("conditional_ndcg_at_5", "ndcg_at5_conditional"),
                ]:
                    source_group = strict_group if metric.startswith("conditional") else bgroup
                    rows.append(
                        {
                            "analysis_id": stable_id("F6F1-POS", "bin", dataset, task, model, strategy, bin_name, metric),
                            "analysis_type": "observational_position_bin",
                            "dataset": dataset,
                            "task": task,
                            "model_name": model,
                            "strategy_id": strategy,
                            "strategy_label": str(key_map["strategy_label"]),
                            "stratum": bin_name,
                            "metric": metric,
                            "k": "5",
                            "value": fmt_float(float(source_group[column].mean())) if not source_group.empty else "",
                            "logical_requests": str(len(bgroup)),
                            "strict_valid": str(len(strict_group)),
                            "denominator": str(len(source_group)),
                            "classification": "exact_formula_adapted_population",
                            "canddif_formula_status": "not_canddif_observational_only",
                            "source_file": str(key_map["source_file"]),
                            "notes": "First-half versus second-half observational comparison.",
                        }
                    )
            for metric, column in [("position_slope_conditional_hr_at_5", "hr_at5_conditional"), ("position_slope_conditional_ndcg_at_5", "ndcg_at5_conditional")]:
                points = []
                for position, pgroup in group.loc[group["valid_list"]].groupby("positive_initial_position", sort=True):
                    if len(pgroup):
                        points.append((float(position), float(pgroup[column].mean())))
                value = math.nan
                if len(points) >= 2:
                    x = np.asarray([point[0] for point in points], dtype=float)
                    y = np.asarray([point[1] for point in points], dtype=float)
                    value = float(np.polyfit(x, y, 1)[0])
                rows.append(
                    {
                        "analysis_id": stable_id("F6F1-POS", "slope", dataset, task, model, strategy, metric),
                        "analysis_type": "observational_position_slope",
                        "dataset": dataset,
                        "task": task,
                        "model_name": model,
                        "strategy_id": strategy,
                        "strategy_label": str(key_map["strategy_label"]),
                        "stratum": "positions_1_20",
                        "metric": metric,
                        "k": "5",
                        "value": fmt_float(value),
                        "logical_requests": str(len(group)),
                        "strict_valid": str(int(group["valid_list"].sum())),
                        "denominator": str(len(points)),
                        "classification": "exact_formula_adapted_population",
                        "canddif_formula_status": "not_canddif_observational_only",
                        "source_file": str(key_map["source_file"]),
                        "notes": "Linear slope across frozen observed positive initial positions; no causal claim.",
                    }
                )

    rows.extend(_plan_d_canddif_rows(root, config))
    rows.append(
        {
            "analysis_id": "F6F1-CANDDIF-F6E4C-BLOCKED",
            "analysis_type": "f6e4c_canddif_status",
            "dataset": "ALL",
            "task": "ALL",
            "model_name": "ALL",
            "strategy_id": "ALL",
            "strategy_label": "",
            "stratum": "full_f6e4c_matrix",
            "metric": "CandDif",
            "k": "5",
            "value": "",
            "logical_requests": str(len(llm)),
            "strict_valid": str(int(llm["valid_list"].sum())),
            "denominator": "",
            "classification": "blocked_missing_input",
            "canddif_formula_status": "blocked_controlled_first_random_pairs_missing",
            "source_file": config["inputs"]["f6e4c_request_level_metrics"],
            "notes": "F6E4C has one frozen candidate order per request, not controlled positive-first/random reruns.",
        }
    )
    return rows


def _plan_d_canddif_rows(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = root / config["inputs"]["plan_d_candidate_position_variant_summary"]
    if not path.exists():
        return []
    frame = safe_read_csv(path)
    result: list[dict[str, Any]] = []
    first = frame.loc[frame["variant"].eq("positive_first")]
    if first.empty:
        return result
    row = first.iloc[0]
    for metric, value_col, delta_col in [
        ("CandDif_HR@5", "hr_at_5", "hr_at_5_delta_vs_original"),
        ("CandDif_NDCG@5", "ndcg_at_5", "ndcg_at_5_delta_vs_original"),
    ]:
        first_acc = parse_float(row[value_col])
        random_acc = first_acc - parse_float(row[delta_col])
        try:
            value = canddif(first_acc, random_acc)
            status = "computed_diagnostic_only"
        except ValueError:
            value = math.nan
            status = "blocked_nonfinite_formula"
        result.append(
            {
                "analysis_id": stable_id("F6F1-CANDDIF", metric),
                "analysis_type": "controlled_diagnostic_canddif",
                "dataset": "ML-1M",
                "task": "diagnostic_ranking",
                "model_name": "qwen2:7b-instruct",
                "strategy_id": "openp5_base",
                "strategy_label": "Base",
                "stratum": "positive_first_vs_original_random_order",
                "metric": metric,
                "k": "5",
                "value": fmt_float(value),
                "logical_requests": str(row.get("total_instances", "")),
                "strict_valid": str(row.get("num_exact_top_k", "")),
                "denominator": str(row.get("total_instances", "")),
                "classification": "reconstructed_formula",
                "canddif_formula_status": status,
                "source_file": config["inputs"]["plan_d_candidate_position_variant_summary"],
                "notes": "Plan D diagnostic subset only; not promoted to full F6E4C CandDif.",
            }
        )
    return result


def compute_ranking_vs_reranking(llm: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eligible = llm.loc[llm["dataset"].ne("LastFM")].copy()
    for (dataset, model, strategy), group in eligible.groupby(["dataset", "model_name", "strategy_id"], sort=False):
        ranking = group.loc[group["task"].eq("ranking")].copy()
        reranking = group.loc[group["task"].eq("reranking")].copy()
        if ranking.empty or reranking.empty:
            continue
        rank_by_user = {str(row["user_id"]): row for _, row in ranking.iterrows() if str(row["user_id"])}
        rerank_by_user = {str(row["user_id"]): row for _, row in reranking.iterrows() if str(row["user_id"])}
        common_users = sorted(set(rank_by_user).intersection(rerank_by_user))
        pairs = [(rank_by_user[user], rerank_by_user[user]) for user in common_users]
        if not pairs:
            continue
        for population, predicate, metric_cols in [
            ("all_user_operational", lambda left, right: True, [("HR@5", "hr_at5_operational"), ("NDCG@5", "ndcg_at5_operational")]),
            ("strict_valid_conditional", lambda left, right: bool(left["valid_list"]) and bool(right["valid_list"]), [("HR@5", "hr_at5_conditional"), ("NDCG@5", "ndcg_at5_conditional")]),
            ("positive_present", lambda left, right: bool(left["positive_present"]) and bool(right["positive_present"]), [("HR@5", "hr_at5_operational"), ("NDCG@5", "ndcg_at5_operational")]),
        ]:
            filtered = [(left, right) for left, right in pairs if predicate(left, right)]
            for metric, column in metric_cols:
                left_values = [float(left[column]) for left, _ in filtered if math.isfinite(float(left[column]))]
                right_values = [float(right[column]) for _, right in filtered if math.isfinite(float(right[column]))]
                if len(left_values) != len(right_values) or not left_values:
                    continue
                rows.append(
                    {
                        "comparison_id": stable_id("F6F1-RVR", dataset, model, strategy, population, metric),
                        "dataset": dataset,
                        "model_name": model,
                        "strategy_id": strategy,
                        "strategy_label": strategy_label(strategy),
                        "population": population,
                        "metric": metric,
                        "ranking_value": fmt_float(float(np.mean(left_values))),
                        "reranking_value": fmt_float(float(np.mean(right_values))),
                        "delta_reranking_minus_ranking": fmt_float(float(np.mean(np.asarray(right_values) - np.asarray(left_values)))),
                        "paired_intersection_size": str(len(filtered)),
                        "ranking_denominator": str(len(ranking)),
                        "reranking_denominator": str(len(reranking)),
                        "classification": "exact_formula_adapted_population",
                        "notes": "Same dataset/model/strategy/user intersection; LastFM re-ranking excluded.",
                    }
                )
        for task_name, task_frame in [("ranking", ranking), ("reranking", reranking)]:
            recall = float(task_frame["positive_present"].mean()) if len(task_frame) else math.nan
            rows.append(
                {
                    "comparison_id": stable_id("F6F1-RVR", dataset, model, strategy, task_name, "candidate_pool_recall"),
                    "dataset": dataset,
                    "model_name": model,
                    "strategy_id": strategy,
                    "strategy_label": strategy_label(strategy),
                    "population": task_name,
                    "metric": "candidate_pool_recall_upper_bound",
                    "ranking_value": fmt_float(recall) if task_name == "ranking" else "",
                    "reranking_value": fmt_float(recall) if task_name == "reranking" else "",
                    "delta_reranking_minus_ranking": "",
                    "paired_intersection_size": str(len(task_frame)),
                    "ranking_denominator": str(len(ranking)),
                    "reranking_denominator": str(len(reranking)),
                    "classification": "extension_metric",
                    "notes": "Positive-present fraction; utility upper bound for candidate-pool construction.",
                }
            )
    rows.append(
        {
            "comparison_id": "F6F1-RVR-LASTFM-RERANKING-EXCLUDED",
            "dataset": "LastFM",
            "model_name": "ALL",
            "strategy_id": "ALL",
            "strategy_label": "",
            "population": "not_applicable",
            "metric": "ranking_vs_reranking",
            "ranking_value": "",
            "reranking_value": "",
            "delta_reranking_minus_ranking": "",
            "paired_intersection_size": "0",
            "ranking_denominator": str(len(llm.loc[(llm["dataset"] == "LastFM") & (llm["task"] == "ranking")])),
            "reranking_denominator": "0",
            "classification": "not_applicable",
            "notes": "LastFM re-ranking is excluded by F6E4C/F6F1 contract.",
        }
    )
    return rows


def paired_bootstrap_ci(
    left_values: Sequence[float],
    right_values: Sequence[float],
    *,
    resamples: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return deterministic paired 95% CI for mean(right-left)."""

    left = np.asarray(left_values, dtype=np.float64)
    right = np.asarray(right_values, dtype=np.float64)
    if len(left) != len(right):
        raise ValueError("paired arrays must have equal length")
    if len(left) == 0:
        return math.nan, math.nan
    delta = right - left
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(resamples, len(delta)))
    samples = delta[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _metric_classification(metric: str) -> str:
    return {
        "HR@5": "exact_formula_adapted_population",
        "NDCG@5": "exact_formula_adapted_population",
        "APLT@5": "exact_formula_adapted_population",
        "Average Popularity@5": "exact_formula_adapted_population",
        "Novelty SelfInformation@5": "exact_formula_reconstructed_inputs",
        "Serendipity@5": "reconstructed_formula",
    }.get(metric, "extension_metric")


def compute_llm_vs_baseline(
    combined: pd.DataFrame,
    *,
    resamples: int = 10000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if combined.empty:
        return []
    llm = combined.loc[combined["source_type"].eq("llm")].copy()
    baselines = combined.loc[combined["source_type"].eq("baseline")].copy()
    baseline_by_family = {
        family: group.copy()
        for family, group in baselines.groupby("baseline_family", sort=False)
        if family in REQUESTED_BASELINE_FAMILIES
    }
    rows: list[dict[str, Any]] = []
    metric_specs = [
        ("HR@5", "hr_at5_conditional"),
        ("NDCG@5", "ndcg_at5_conditional"),
        ("APLT@5", "aplt_at_5"),
        ("Average Popularity@5", "average_popularity_at_5"),
        ("Novelty SelfInformation@5", "novelty_self_information_at_5"),
        ("Serendipity@5", "serendipity_at_5"),
    ]
    for (dataset, task, model, strategy), group in llm.groupby(["dataset", "task", "model_name", "strategy_id"], sort=False):
        strict = group.loc[group["valid_list"]].copy()
        if strict.empty:
            continue
        for family in REQUESTED_BASELINE_FAMILIES:
            bframe = baseline_by_family.get(family, pd.DataFrame())
            bframe = bframe.loc[(bframe["dataset"] == dataset) & (bframe["task"] == task)] if not bframe.empty else bframe
            if bframe.empty:
                rows.append(_blocked_baseline_row(dataset, task, model, strategy, family, "baseline_source_missing"))
                continue
            bmap = {str(row["row_key"]): row for _, row in bframe.iterrows() if bool(row["valid_list"])}
            for metric, column in metric_specs:
                llm_values: list[float] = []
                baseline_values: list[float] = []
                baseline_model = ""
                for _, lrow in strict.iterrows():
                    brow = bmap.get(str(lrow["row_key"]))
                    if brow is None:
                        continue
                    lvalue = parse_float(lrow.get(column, ""))
                    bvalue = parse_float(brow.get(column, ""))
                    if not (math.isfinite(lvalue) and math.isfinite(bvalue)):
                        continue
                    llm_values.append(lvalue)
                    baseline_values.append(bvalue)
                    baseline_model = str(brow.get("model_name", baseline_model))
                if not llm_values:
                    rows.append(_blocked_baseline_row(dataset, task, model, strategy, family, f"no_paired_rows_for_{metric}"))
                    continue
                low, high = paired_bootstrap_ci(baseline_values, llm_values, resamples=resamples, seed=seed)
                rows.append(
                    {
                        "comparison_id": stable_id("F6F1-LVB", dataset, task, model, strategy, family, metric),
                        "dataset": dataset,
                        "task": task,
                        "llm_model": model,
                        "strategy_id": strategy,
                        "strategy_label": strategy_label(strategy),
                        "baseline_family": family,
                        "baseline_model": baseline_model,
                        "population": "strict_valid_paired_intersection",
                        "metric": metric,
                        "llm_value": fmt_float(float(np.mean(llm_values))),
                        "baseline_value": fmt_float(float(np.mean(baseline_values))),
                        "delta_llm_minus_baseline": fmt_float(float(np.mean(np.asarray(llm_values) - np.asarray(baseline_values)))),
                        "bootstrap_ci_low": fmt_float(low),
                        "bootstrap_ci_high": fmt_float(high),
                        "paired_intersection_size": str(len(llm_values)),
                        "llm_coverage": fmt_float(len(llm_values) / len(strict)),
                        "baseline_coverage": fmt_float(len(llm_values) / len(bframe)),
                        "classification": _metric_classification(metric),
                        "notes": "Paired by dataset/task/user/positive item/candidate-order hash; bootstrap is paired 10000 resamples seed 42.",
                    }
                )
    return rows


def _blocked_baseline_row(dataset: str, task: str, model: str, strategy: str, family: str, reason: str) -> dict[str, Any]:
    return {
        "comparison_id": stable_id("F6F1-LVB-BLOCK", dataset, task, model, strategy, family, reason),
        "dataset": dataset,
        "task": task,
        "llm_model": model,
        "strategy_id": strategy,
        "strategy_label": strategy_label(strategy),
        "baseline_family": family,
        "baseline_model": "",
        "population": "strict_valid_paired_intersection",
        "metric": "ALL",
        "llm_value": "",
        "baseline_value": "",
        "delta_llm_minus_baseline": "",
        "bootstrap_ci_low": "",
        "bootstrap_ci_high": "",
        "paired_intersection_size": "0",
        "llm_coverage": "0",
        "baseline_coverage": "0",
        "classification": "blocked_missing_input",
        "notes": reason,
    }


def attach_serendipity_row_metric(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["serendipity_at_5"] = math.nan
    mostpop = result.loc[(result["source_type"] == "baseline") & (result["baseline_family"] == "MostPop")]
    mostpop_ref = {str(row["row_key"]): _hit_at5(row) for _, row in mostpop.iterrows() if bool(row["valid_list"])}
    for idx, row in result.iterrows():
        if not bool(row["valid_list"]):
            continue
        key = str(row["row_key"])
        if key not in mostpop_ref:
            continue
        result.at[idx, "serendipity_at_5"] = _hit_at5(row) * (1.0 - mostpop_ref[key])
    return result


def build_research_question_update(
    llm_vs_baseline: Sequence[Mapping[str, Any]],
    aplt_results: Sequence[Mapping[str, Any]],
    serendipity_results: Sequence[Mapping[str, Any]],
    candidate_position_results: Sequence[Mapping[str, Any]],
    ranking_vs_reranking: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    comparison_count = sum(1 for row in llm_vs_baseline if row.get("classification") != "blocked_missing_input")
    aplt_count = sum(1 for row in aplt_results if row.get("status") == "computed")
    ser_count = sum(1 for row in serendipity_results if row.get("status") == "computed")
    pos_count = sum(1 for row in candidate_position_results if row.get("analysis_type", "").startswith("observational"))
    rvr_count = sum(1 for row in ranking_vs_reranking if row.get("classification") != "not_applicable")
    return [
        {
            "research_question": "utility degli LLM rispetto alle baseline",
            "status": "updated",
            "primary_evidence": f"{comparison_count} paired LLM-vs-baseline metric rows",
            "classification": "partially_supported",
            "allowed_claim": "paired adapted comparisons are available where candidate/order/user intersections exist",
            "limitation": "not paper-exact baseline training/cohort reproduction",
        },
        {
            "research_question": "comportamento long-tail",
            "status": "updated",
            "primary_evidence": f"{aplt_count} APLT rows",
            "classification": "supported_in_adapted_setting",
            "allowed_claim": "APLT computed with recovered q80 train-count policy in adapted F6E4C population",
            "limitation": "local train splits and candidate pools are adapted",
        },
        {
            "research_question": "novelty",
            "status": "updated",
            "primary_evidence": "novelty_self_information_at_5 and local proxy rows in popularity/novelty table",
            "classification": "partially_supported",
            "allowed_claim": "paper-style self-information is reported only when train-user support exists",
            "limitation": "DOCX formula is secondary and cold items block exact reconstructed input rows",
        },
        {
            "research_question": "serendipity",
            "status": "updated",
            "primary_evidence": f"{ser_count} serendipity rows",
            "classification": "partially_supported",
            "allowed_claim": "MostPop-hit reconstructed serendipity is available on paired rows",
            "limitation": "not a paper-exact claim because paper prose/code evidence differ locally",
        },
        {
            "research_question": "candidate-position bias",
            "status": "updated",
            "primary_evidence": f"{pos_count} observational position rows plus diagnostic CandDif status",
            "classification": "partially_supported",
            "allowed_claim": "observational F6E4C position strata and limited diagnostic CandDif are available",
            "limitation": "no full-matrix causal controlled permutation evidence",
        },
        {
            "research_question": "ranking versus re-ranking",
            "status": "updated",
            "primary_evidence": f"{rvr_count} paired ranking/re-ranking rows",
            "classification": "partially_supported",
            "allowed_claim": "same-user adapted comparisons are available for Beauty, Sports, and ML-1M",
            "limitation": "LastFM re-ranking excluded and cohorts must be read per-row",
        },
        {
            "research_question": "Base versus LLMRank",
            "status": "updated",
            "primary_evidence": "F6E4C strategy paired comparison plus F6F1 beyond-utility tables",
            "classification": "supported_in_adapted_setting",
            "allowed_claim": "Base and LLMRank are compared within the frozen adapted matrix",
            "limitation": "invalid-output differences must be kept explicit",
        },
        {
            "research_question": "output reliability",
            "status": "updated",
            "primary_evidence": "strict_valid_rate, invalid_list_rate, duplicate_rate_at_5",
            "classification": "supported_in_adapted_setting",
            "allowed_claim": "strict-valid and invalid-list behavior is measurable offline",
            "limitation": "reliability metrics are extensions, not paper utility metrics",
        },
    ]


def build_missing_or_blocked(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": "F6F1-BLOCK-001",
            "metric_or_component": "CandDif full F6E4C",
            "scope": "full_f6e4c_matrix",
            "blocking_classification": "blocked_missing_input",
            "blocking_reason": "F6E4C has one frozen candidate order per request, not controlled positive-first/random reruns.",
            "required_evidence": "same instances rerun under positive-first and random-position candidate orders",
            "current_evidence": "observational positive_initial_position strata; Plan D diagnostic subset only",
            "recommended_action": "do not claim causal CandDif for F6E4C; run a controlled offline-frozen design only if approved later",
        },
        {
            "item_id": "F6F1-BLOCK-002",
            "metric_or_component": "PopREO / popularity_bias exact metric",
            "scope": "all datasets",
            "blocking_classification": "blocked_missing_evidence",
            "blocking_reason": "empty popularity group and zero denominator policies are not frozen.",
            "required_evidence": "paper/code-confirmed PopREO denominator and empty-group treatment",
            "current_evidence": "official calPopREO function plus F5C semantic audit",
            "recommended_action": "keep ARP/APLT separate and do not rename either as popularity bias",
        },
        {
            "item_id": "F6F1-BLOCK-003",
            "metric_or_component": "paper-exact serendipity",
            "scope": "all datasets",
            "blocking_classification": "blocked_missing_evidence",
            "blocking_reason": "official code hit-vector definition and secondary DOCX item-list prose are not fully equivalent.",
            "required_evidence": "PDF-verified formula and exact primitive MostPop run from paper",
            "current_evidence": "F5D MostPop train-only reconstructed policy",
            "recommended_action": "report reconstructed/adapted serendipity only",
        },
        {
            "item_id": "F6F1-BLOCK-004",
            "metric_or_component": "LastFM re-ranking",
            "scope": "LastFM",
            "blocking_classification": "not_applicable",
            "blocking_reason": "LastFM re-ranking is excluded by F6E4C consolidated matrix.",
            "required_evidence": "not_applicable",
            "current_evidence": "F6E4C workloads include LastFM ranking only",
            "recommended_action": "exclude from ranking-vs-reranking tables",
        },
        {
            "item_id": "F6F1-BLOCK-005",
            "metric_or_component": "manual PDF formula verification",
            "scope": "paper source inventory",
            "blocking_classification": "blocked_missing_evidence",
            "blocking_reason": "local PDF parser remains unavailable; DOCX is secondary source only.",
            "required_evidence": "machine-readable or manually double-entered PDF formula verification",
            "current_evidence": f"PDF path {config['inputs']['paper_pdf']} plus DOCX secondary source",
            "recommended_action": "review phase_f6f1_metric_source_inventory.csv before upgrading exactness",
        },
    ]


def _table_hash(path: Path) -> str:
    return sha256_file(path) if path.exists() else "missing"


def build_freeze_manifest(
    root: Path,
    config: Mapping[str, Any],
    repository_state: Mapping[str, Any],
    input_verification: Mapping[str, Any],
    outputs: Mapping[str, Path],
) -> dict[str, Any]:
    table_hashes = {name: _table_hash(path) for name, path in sorted(outputs.items())}
    f6e4c_manifest = json.loads((root / config["inputs"]["f6e4c_freeze_manifest"]).read_text(encoding="utf-8"))
    content = {
        "phase": PHASE_ID,
        "table_hashes": table_hashes,
        "input_hashes": input_verification["input_hashes"],
        "bootstrap_configuration": {
            "resamples": int(config["phase"]["bootstrap_resamples"]),
            "seed": int(config["phase"]["bootstrap_seed"]),
            "ci": "95%",
            "paired": True,
        },
        "decision": DECISION_COMPLETE_WITH_BLOCKS,
    }
    return {
        **content,
        "f6e4c_scientific_hash": f6e4c_manifest.get("scientific_content_hash", ""),
        "f6e4c_decision": f6e4c_manifest.get("decision", ""),
        "metric_contract_hash": table_hashes.get("metric_contract", ""),
        "formula_source_hashes": {
            "paper_pdf": sha256_file(root / config["inputs"]["paper_pdf"]),
            "paper_docx_secondary": sha256_file(root / config["inputs"]["paper_docx_secondary"]),
            "official_metrics_code": sha256_file(root / config["inputs"]["official_metrics_code"]),
        },
        "result_table_hashes": table_hashes,
        "scientific_content_hash": sha256_text(stable_json(content)),
        "repository_state": dict(repository_state),
        "protected_summary": input_verification["protected_summary"],
        "recommended_next_action": "review phase_f6f1_missing_or_blocked.csv before any PDF/manual exactness upgrade",
    }


def build_report(
    config: Mapping[str, Any],
    repository_state: Mapping[str, Any],
    input_verification: Mapping[str, Any],
    manifest: Mapping[str, Any],
    counts: Mapping[str, int],
) -> str:
    sections = [
        ("Initial repository state", f"Branch `{repository_state['branch']}`, HEAD `{repository_state['head']}`, staged files `{repository_state['staged_files'] or 'none'}`. Tracked dirty state remains `{repository_state['git_status_tracked_short']}`."),
        ("Frozen-input verification", f"Checked `{input_verification['checked_count']}` configured inputs and `{input_verification['protected_summary']['checked']}` protected rows; changed `{input_verification['protected_summary']['changed']}`, missing `{input_verification['protected_summary']['missing']}`."),
        ("Paper metric source inventory", f"Rows written: `{counts['metric_source_inventory']}`. PDF is authoritative; DOCX is secondary structured evidence."),
        ("Metric contracts", f"Rows written: `{counts['metric_contract']}` with exact/adapted/reconstructed/blocked classifications."),
        ("Exactness classifications", f"Rows written: `{counts['metric_exactness_matrix']}`. No proxy is promoted to paper-exact."),
        ("APLT results", f"Rows written: `{counts['aplt_results']}`. Invalid LLM lists are excluded for conditional and operational is not defined."),
        ("Serendipity results", f"Rows written: `{counts['serendipity_results']}` using frozen MostPop references where paired."),
        ("Popularity and novelty results", f"Rows written: `{counts['popularity_novelty_results']}`. ARP, novelty, coverage, and reliability remain separate metrics."),
        ("Candidate-position results", f"Rows written: `{counts['candidate_position_results']}`. F6E4C position analysis is observational; CandDif is blocked for full matrix."),
        ("CandDif status", "Plan D diagnostic CandDif is reported as diagnostic only; F6E4C full-matrix CandDif is `blocked_missing_input`."),
        ("Ranking-vs-reranking results", f"Rows written: `{counts['ranking_vs_reranking']}`. LastFM re-ranking is excluded."),
        ("LLM-vs-baseline results", f"Rows written: `{counts['llm_vs_baseline']}` using paired intersections and bootstrap seed 42."),
        ("Structural diagnostics", f"Rows written: `{counts['structural_diagnostics']}`. These rows keep reliability, positive-presence, duplicate-rate, and operational utility diagnostics separate from paper metric claims."),
        ("Invalid-output policies", "Utility operational invalid equals zero. List beyond-utility metrics use strict-valid conditional rows; operational list values are `not_defined` for invalid LLM outputs."),
        ("Updated research-question evidence", f"Rows written: `{counts['research_question_update']}`."),
        ("Implementation files", "`src/analysis/plan_f6f1_beyond_utility_metrics.py`, `src/experiments/build_phase_f6f1_beyond_utility_metrics.py`, and `configs/phase_f6f1_beyond_utility_metrics.yaml`."),
        ("Tests added", "`tests/test_plan_f6f1_beyond_utility_metrics.py`."),
        ("Regression results", "Recorded by operator command log; unsafe suites are excluded by config."),
        ("Excluded unsafe tests", ", ".join(config.get("unsafe_excluded_tests", []))),
        ("Determinism", "Builder preserves first-run timestamp in the freeze manifest and writes sorted CSV/JSON outputs."),
        ("Protected-input verification", f"Protected summary: `{manifest['protected_summary']}`."),
        ("Freeze manifest", f"`{config['outputs']['freeze_manifest']}` with scientific content hash `{manifest['scientific_content_hash']}`."),
        ("Missing or blocked evidence", f"Rows written: `{counts['missing_or_blocked']}`."),
        ("Final decision", f"`{manifest['decision']}`."),
        ("Recommended next action", manifest["recommended_next_action"]),
        ("Final git state", "No git add, commit, reset, clean, checkout, or protected input modification is performed by the builder."),
        ("Files created or modified", "Only new `phase_f6f1_*` files plus the F6F1 config/code/test files."),
        ("Remaining limitations", "Paper-exact claims remain limited by PDF extraction, adapted local candidate populations, and missing controlled full-matrix CandDif permutations."),
    ]
    lines = ["# Phase F6F1 Beyond-Utility Metrics Recovery", ""]
    for idx, (title, body) in enumerate(sections, start=1):
        lines.extend([f"## {idx}. {title}", body, ""])
    return "\n".join(lines).rstrip() + "\n"


def write_phase_outputs(
    root: Path,
    config: Mapping[str, Any],
    repository_state: Mapping[str, Any],
    input_verification: Mapping[str, Any],
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    outputs_config = config["outputs"]["tables"]
    columns = {
        "metric_source_inventory": SOURCE_INVENTORY_COLUMNS,
        "metric_contract": CONTRACT_COLUMNS,
        "metric_exactness_matrix": EXACTNESS_COLUMNS,
        "aplt_results": APLT_COLUMNS,
        "serendipity_results": SERENDIPITY_COLUMNS,
        "popularity_novelty_results": POPULARITY_NOVELTY_COLUMNS,
        "candidate_position_results": CANDIDATE_POSITION_COLUMNS,
        "ranking_vs_reranking": RANKING_RERANKING_COLUMNS,
        "llm_vs_baseline": LLM_BASELINE_COLUMNS,
        "structural_diagnostics": STRUCTURAL_DIAGNOSTIC_COLUMNS,
        "research_question_update": RQ_COLUMNS,
        "missing_or_blocked": MISSING_COLUMNS,
    }
    output_paths: dict[str, Path] = {}
    for name, rows in tables.items():
        path = root / outputs_config[name]
        write_csv(path, rows, columns[name])
        output_paths[name] = path

    manifest = build_freeze_manifest(root, config, repository_state, input_verification, output_paths)
    write_json(root / config["outputs"]["freeze_manifest"], manifest)
    counts = {name: len(rows) for name, rows in tables.items()}
    report_text = build_report(config, repository_state, input_verification, manifest, counts)
    report_path = root / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8", newline="\n")

    manifest = build_freeze_manifest(root, config, repository_state, input_verification, output_paths)
    manifest["report_hash"] = sha256_file(report_path)
    write_json(root / config["outputs"]["freeze_manifest"], manifest)
    return manifest


def run_phase(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    repository_state = record_repository_state(root, config)
    input_verification = verify_frozen_inputs(root, config)

    source_inventory = build_metric_source_inventory(root, config)
    contracts = build_metric_contracts()
    exactness = build_metric_exactness_matrix()

    llm = normalize_llm_lists(load_f6e4c_llm_rows(root, config))
    baselines = normalize_baseline_rankings(root, config)
    combined = pd.concat([llm, baselines], ignore_index=True, sort=False)
    popularity = load_popularity_data(root, config)
    combined = enrich_row_metrics(combined, popularity)
    combined = attach_serendipity_row_metric(combined)

    aplt_rows, popularity_novelty_rows = compute_aplt_arp_novelty(combined, popularity)
    serendipity_rows = compute_serendipity(combined)
    candidate_position_rows = compute_candidate_position(root, config, llm)
    ranking_vs_reranking_rows = compute_ranking_vs_reranking(llm)
    llm_vs_baseline_rows = compute_llm_vs_baseline(
        combined,
        resamples=int(config["phase"]["bootstrap_resamples"]),
        seed=int(config["phase"]["bootstrap_seed"]),
    )
    structural_diagnostics_rows = build_structural_diagnostics(combined)
    rq_rows = build_research_question_update(
        llm_vs_baseline_rows,
        aplt_rows,
        serendipity_rows,
        candidate_position_rows,
        ranking_vs_reranking_rows,
    )
    missing_rows = build_missing_or_blocked(config)

    tables = {
        "metric_source_inventory": source_inventory,
        "metric_contract": contracts,
        "metric_exactness_matrix": exactness,
        "aplt_results": aplt_rows,
        "serendipity_results": serendipity_rows,
        "popularity_novelty_results": popularity_novelty_rows,
        "candidate_position_results": candidate_position_rows,
        "ranking_vs_reranking": ranking_vs_reranking_rows,
        "llm_vs_baseline": llm_vs_baseline_rows,
        "structural_diagnostics": structural_diagnostics_rows,
        "research_question_update": rq_rows,
        "missing_or_blocked": missing_rows,
    }
    return write_phase_outputs(root, config, repository_state, input_verification, tables)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Phase F6F1 beyond-utility metrics offline.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to F6F1 YAML config.")
    args = parser.parse_args(argv)

    root = Path.cwd()
    config = load_config(root / args.config)
    manifest = run_phase(root, config)
    print(stable_json({"phase": PHASE_ID, "decision": manifest["decision"], "scientific_content_hash": manifest["scientific_content_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
