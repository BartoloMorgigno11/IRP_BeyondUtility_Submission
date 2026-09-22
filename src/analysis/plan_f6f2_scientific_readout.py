"""Phase F6F2 scientific readout and claim freeze.

This module is offline-only. It reads frozen F6F1, F6F1R, and F6E4C
artifacts, summarizes the numeric evidence, freezes qualified claims, and
defines the next-phase requirements without modifying prior phase artifacts.
"""

from __future__ import annotations

from src.audit import common_git

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


PHASE_ID = "phase_f6f2"
DEFAULT_CONFIG_PATH = Path("configs/phase_f6f2_scientific_readout.yaml")
EXPECTED_F6F1_HASH = "4bf9391b835dad47c80f41b9d41b130f64ac3f4776561cc19bf0a3d044ffb37c"
DECISION_COMPLETE_WITH_BLOCKS = "f6f2_scientific_readout_complete_with_documented_blocks"
RECOMMENDATION = "proceed_to_lastfm_reranking_reconstruction"
PRE_EXISTING_DIRTY = ["README.md", "report/dataset_audit.md", "report/notes.md"]
F6F2_TEST_SUITES = [
    ("F6F2 tests", ["tests/test_plan_f6f2_scientific_readout.py"]),
    ("F6F1 tests", ["tests/test_plan_f6f1_beyond_utility_metrics.py"]),
    ("F6F1R tests", ["tests/test_plan_f6f1r_environment_recovery.py"]),
    ("F6E4C tests", ["tests/test_plan_f6e4c_scientific_consolidation.py"]),
    ("protected input guard", ["tests/test_finalization_f6d4a_protected_input_guard.py"]),
    ("minimal parser contract", ["tests/test_canonical_llm_contract.py"]),
]


INPUT_INTEGRITY_COLUMNS = [
    "artifact",
    "path",
    "phase",
    "expected_sha256",
    "actual_sha256",
    "status",
    "changed",
    "missing",
    "notes",
]
METRIC_STATUS_COLUMNS = [
    "paper_metric",
    "local_implementation",
    "formula_status",
    "input_status",
    "population_status",
    "implementation_status",
    "final_exactness_classification",
    "datasets_covered",
    "tasks_covered",
    "models_covered",
    "baselines_covered",
    "numeric_result_exists",
    "paper_claim_use",
    "principal_limitation",
]
KEY_FINDING_COLUMNS = [
    "finding_id",
    "question",
    "source_table",
    "metric",
    "population",
    "scope",
    "n_rows",
    "mean_value",
    "mean_delta",
    "min_delta",
    "max_delta",
    "positive_delta_count",
    "negative_delta_count",
    "zero_delta_count",
    "supported_ci_count",
    "direction",
    "strength",
    "evidence_classification",
    "allowed_interpretation",
    "limitation",
]
READOUT_COLUMNS = [
    "readout_id",
    "readout_type",
    "source_table",
    "scope",
    "dataset",
    "task",
    "model_name",
    "strategy_id",
    "strategy_label",
    "baseline_family",
    "comparator",
    "population",
    "metric",
    "value",
    "comparison_value",
    "delta",
    "ci_low",
    "ci_high",
    "paired_intersection_size",
    "denominator",
    "exactness_classification",
    "statistical_status",
    "direction",
    "strength",
    "allowed_interpretation",
    "limitation",
    "notes",
]
RQ_COLUMNS = [
    "rq_id",
    "paper_claim",
    "local_question",
    "evidence_tables",
    "models",
    "datasets",
    "tasks",
    "metrics",
    "exactness_level",
    "result_direction",
    "final_evidence_classification",
    "allowed_conclusion",
    "prohibited_conclusion",
    "principal_limitation",
    "next_evidence_required",
]
CLAIM_COLUMNS = [
    "claim_id",
    "claim_text",
    "evidence",
    "evidence_level",
    "exactness_classification",
    "statistical_status",
    "allowed_wording",
    "prohibited_wording",
    "limitation",
    "scope",
]
GAP_COLUMNS = [
    "gap_id",
    "gap",
    "paper_or_extension_scope",
    "current_status",
    "blocking_reason",
    "required_inputs",
    "estimated_computational_class",
    "dependency_on_phase",
    "recommended_order",
]
NEXT_PHASE_COLUMNS = [
    "requirement_id",
    "component",
    "requirement",
    "policy",
    "source_or_input",
    "reuse_status",
    "new_inference_required",
    "exactness_classification",
    "stop_condition",
]
MISSING_COLUMNS = [
    "item_id",
    "component",
    "status",
    "classification",
    "reason",
    "required_evidence",
    "current_evidence",
    "recommended_action",
]


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


def stable_id(prefix: str, *parts: Any, length: int = 14) -> str:
    return f"{prefix}-{sha256_text('||'.join(str(part) for part in parts))[:length]}"


def fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.17g}"


def parse_float(value: Any) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        [{column: "" if row.get(column, "") is None else str(row.get(column, "")) for column in columns} for row in rows],
        key=lambda row: tuple(row[column] for column in columns),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(payload) + "\n", encoding="utf-8")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(args: Sequence[str], cwd: Path, timeout: int = 120) -> tuple[int, str, str, float]:
    started = time.perf_counter()
    completed = subprocess.run(
        list(args),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return completed.returncode, completed.stdout, completed.stderr, time.perf_counter() - started


def git(root: Path, *args: str) -> str:
    return common_git.git_stdout_no_check(args, root, timeout=60, empty_on_failure=True)


def existing_manifest(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    path = root / config["outputs"]["freeze_manifest"]
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except json.JSONDecodeError:
        return {}


def generated_paths(config: Mapping[str, Any]) -> list[str]:
    paths = [config["outputs"]["report"], config["outputs"]["freeze_manifest"]]
    paths.extend(config["outputs"]["tables"].values())
    paths.extend(
        [
            "configs/phase_f6f2_scientific_readout.yaml",
            "src/analysis/plan_f6f2_scientific_readout.py",
            "src/experiments/build_phase_f6f2_scientific_readout.py",
            "tests/test_plan_f6f2_scientific_readout.py",
        ]
    )
    return [str(path).replace("\\", "/").rstrip("/") for path in paths]


def filter_generated_status(text: str, config: Mapping[str, Any]) -> str:
    generated = generated_paths(config)
    kept: list[str] = []
    for line in text.splitlines():
        path = line[3:].strip().strip('"').replace("\\", "/") if len(line) > 3 else line
        if any(path == item or path.startswith(item + "/") for item in generated):
            continue
        kept.append(line)
    return "\n".join(kept)


def record_repository_state(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    previous = existing_manifest(root, config)
    return {
        "repository_path": str(root),
        "branch": git(root, "branch", "--show-current"),
        "head": git(root, "rev-parse", "HEAD"),
        "git_status_short": filter_generated_status(git(root, "status", "--short"), config),
        "git_status_tracked_short": git(root, "status", "--short", "--untracked-files=no"),
        "git_diff_name_only": git(root, "diff", "--name-only"),
        "git_diff_stat": git(root, "diff", "--stat"),
        "git_diff_cached_stat": git(root, "diff", "--cached", "--stat"),
        "staged_files": git(root, "diff", "--cached", "--name-only"),
        "timestamp": previous.get("runtime_metadata", {}).get("timestamp") or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "operating_system": platform.platform(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "environment_classification": config["phase"]["environment_classification"],
    }


def verify_interpreter(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    expected = (root / config["phase"]["interpreter_path"]).resolve(strict=False)
    actual = Path(sys.executable).resolve(strict=False)
    starts = actual.exists()
    imports_ok = True
    import_error = ""
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import pytest  # noqa: F401
        import yaml as _yaml  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised in blocked envs
        imports_ok = False
        import_error = f"{type(exc).__name__}: {exc}"
    status = "pass" if starts and imports_ok and str(actual).lower() == str(expected).lower() else "fail"
    return {
        "expected_interpreter": str(expected),
        "actual_interpreter": str(actual),
        "status": status,
        "imports_ok": "true" if imports_ok else "false",
        "import_error": import_error,
        "python_version": platform.python_version(),
    }


def required_input_paths(root: Path, config: Mapping[str, Any]) -> list[tuple[str, str, Path, str]]:
    result: list[tuple[str, str, Path, str]] = []
    for key, rel_path in config["inputs"].items():
        path = root / rel_path
        phase = "F6F1" if "f6f1_" in key else "F6F1R" if "f6f1r_" in key else "F6E4C" if "f6e4c_" in key else "input"
        result.append((key, phase, path, ""))
    f6f1_manifest_path = root / config["inputs"]["f6f1_manifest"]
    if f6f1_manifest_path.exists():
        manifest = load_json(f6f1_manifest_path)
        for name, digest in manifest.get("result_table_hashes", {}).items():
            f6f1_config = load_config(root / "configs/phase_f6f1_beyond_utility_metrics.yaml")
            table_path = f6f1_config["outputs"]["tables"].get(name)
            if table_path:
                result.append((f"f6f1_result_{name}", "F6F1", root / table_path, digest))
        result.append(("f6f1_report_hash", "F6F1", root / "report/phase_f6f1_beyond_utility_metrics.md", manifest.get("report_hash", "")))
    f6f1r_manifest_path = root / config["inputs"]["f6f1r_manifest"]
    if f6f1r_manifest_path.exists():
        manifest = load_json(f6f1r_manifest_path)
        f6f1r_paths = {
            "broken_venv_audit": "results/tables/phase_f6f1r_broken_venv_audit.csv",
            "cross_environment_comparison": "results/tables/phase_f6f1r_cross_environment_comparison.csv",
            "dependency_contract": "results/tables/phase_f6f1r_dependency_contract.csv",
            "environment_gate": "results/tables/phase_f6f1r_environment_gate.csv",
            "environment_variables": "results/environment/phase_f6f1r_environment_variables_redacted.json",
            "input_integrity": "results/tables/phase_f6f1r_input_integrity.csv",
            "interpreter_identity": "results/tables/phase_f6f1r_interpreter_identity.csv",
            "missing_or_blocked": "results/tables/phase_f6f1r_missing_or_blocked.csv",
            "pip_freeze": "results/environment/phase_f6f1r_pip_freeze.txt",
            "pip_list": "results/environment/phase_f6f1r_pip_list.json",
            "python_identity": "results/environment/phase_f6f1r_python_identity.json",
            "python_inventory": "results/tables/phase_f6f1r_python_inventory.csv",
            "report": "report/phase_f6f1r_environment_recovery.md",
            "repository_state": "results/tables/phase_f6f1r_repository_state.csv",
            "test_results": "results/tables/phase_f6f1r_test_results.csv",
        }
        for name, digest in manifest.get("artifact_hashes", {}).items():
            if name in f6f1r_paths:
                result.append((f"f6f1r_artifact_{name}", "F6F1R", root / f6f1r_paths[name], digest))
    f6e4c_manifest_path = root / config["inputs"]["f6e4c_manifest"]
    if f6e4c_manifest_path.exists():
        manifest = load_json(f6e4c_manifest_path)
        for rel_path, digest in manifest.get("input_hashes", {}).items():
            path = root / rel_path
            if path.exists():
                result.append((f"f6e4c_direct_input_{Path(rel_path).name}", "F6E4C", path, digest))
    seen: set[str] = set()
    unique: list[tuple[str, str, Path, str]] = []
    for item in result:
        key = str(item[2].resolve(strict=False)).lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def verify_input_integrity(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact, phase, path, expected_hash in required_input_paths(root, config):
        exists = path.exists()
        actual = sha256_file(path) if exists and path.is_file() else ""
        missing = not exists
        changed = bool(expected_hash and actual and expected_hash.lower() != actual.lower())
        rows.append(
            {
                "artifact": artifact,
                "path": str(path.relative_to(root) if str(path).startswith(str(root)) else path),
                "phase": phase,
                "expected_sha256": expected_hash,
                "actual_sha256": actual,
                "status": "pass" if not missing and not changed else "fail",
                "changed": "true" if changed else "false",
                "missing": "true" if missing else "false",
                "notes": "",
            }
        )
    f6f1 = load_json(root / config["inputs"]["f6f1_manifest"])
    f6f1r = load_json(root / config["inputs"]["f6f1r_manifest"])
    f6e4c = load_json(root / config["inputs"]["f6e4c_manifest"])
    if f6f1.get("scientific_content_hash") != config["phase"]["expected_f6f1_scientific_hash"]:
        raise ValueError("F6F1 scientific content hash mismatch")
    if f6f1.get("decision") != config["phase"]["expected_f6f1_decision"]:
        raise ValueError("F6F1 decision mismatch")
    if f6f1r.get("decision") != config["phase"]["expected_f6f1r_decision"]:
        raise ValueError("F6F1R decision mismatch")
    if f6f1r.get("frozen_f6f1_scientific_content_hash") != f6f1r.get("isolated_f6f1_scientific_content_hash"):
        raise ValueError("F6F1R scientific comparison mismatch")
    if f6e4c.get("decision") != config["phase"]["expected_f6e4c_decision"]:
        raise ValueError("F6E4C decision mismatch")
    return rows


def load_tables(root: Path, config: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    tables: dict[str, list[dict[str, str]]] = {}
    for key, rel_path in config["inputs"].items():
        if str(rel_path).endswith(".csv"):
            tables[key] = read_csv(root / rel_path)
    return tables


def validate_columns(root: Path, config: Mapping[str, Any]) -> None:
    required = {
        "f6f1_metric_exactness_matrix": ["paper_metric", "local_metric", "exactness_classification"],
        "f6f1_aplt_results": ["dataset", "task", "model_name", "population", "metric", "value", "classification"],
        "f6f1_serendipity_results": ["dataset", "task", "model_name", "population", "metric", "value", "classification"],
        "f6f1_popularity_novelty_results": ["dataset", "task", "model_name", "population", "metric", "value", "classification"],
        "f6f1_candidate_position_results": ["analysis_type", "dataset", "task", "model_name", "metric", "value", "classification"],
        "f6f1_ranking_vs_reranking": ["dataset", "model_name", "population", "metric", "delta_reranking_minus_ranking", "classification"],
        "f6f1_llm_vs_baseline": ["dataset", "task", "llm_model", "baseline_family", "metric", "delta_llm_minus_baseline", "classification"],
        "f6e4c_strategy_paired_comparison": ["comparison_type", "population", "dataset", "task", "model", "metric", "delta_right_minus_left"],
        "f6e4c_model_paired_comparison": ["comparison_type", "population", "dataset", "task", "model", "metric", "delta_right_minus_left"],
    }
    for key, columns in required.items():
        path = root / config["inputs"][key]
        present = set(csv_columns(path))
        missing = [column for column in columns if column not in present]
        if missing:
            raise ValueError(f"{key} missing columns: {missing}")


def unique_join(rows: Sequence[Mapping[str, str]], column: str) -> str:
    values = sorted({str(row.get(column, "")) for row in rows if str(row.get(column, "")).strip()})
    return "|".join(values)


def has_numeric(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> str:
    for row in rows:
        for column in columns:
            if parse_float(row.get(column, "")) is not None:
                return "true"
    return "false"


def rows_for_metric(metric: str, tables: Mapping[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    lowered = metric.lower()
    if "aplt" in lowered:
        return tables["f6f1_aplt_results"]
    if "serendipity" in lowered:
        return tables["f6f1_serendipity_results"]
    if "novelty" in lowered or "popularity" in lowered or "coverage" in lowered or "duplicate" in lowered or "strict-valid" in lowered or "operational" in lowered:
        rows = list(tables["f6f1_popularity_novelty_results"]) + list(tables["f6f1_structural_diagnostics"])
        if "average popularity" in lowered:
            return [row for row in rows if row.get("metric") == "average_popularity_at_5"]
        if "novelty" in lowered:
            return [row for row in rows if "novelty" in row.get("metric", "")]
        if "catalog coverage" in lowered:
            return [row for row in rows if row.get("metric") == "catalog_coverage_at_5"]
        if "item coverage" in lowered:
            return [row for row in rows if row.get("metric") == "item_coverage_at_5"]
        if "duplicate" in lowered:
            return [row for row in rows if row.get("metric") == "duplicate_rate_at_5"]
        if "strict-valid" in lowered:
            return [row for row in rows if row.get("metric") == "strict_valid_rate"]
        if "operational" in lowered:
            return [row for row in rows if row.get("metric") in {"operational_hr_at_5", "operational_ndcg_at_5"}]
        return rows
    if "candidate" in lowered or "canddif" in lowered:
        return tables["f6f1_candidate_position_results"]
    if metric in {"HR@5", "NDCG@5"}:
        return [row for row in tables["f6f1_llm_vs_baseline"] if row.get("metric") == metric]
    return []


def build_metric_status(tables: Mapping[str, list[dict[str, str]]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    allowed = set(config["allowed_exactness_classifications"])
    rows: list[dict[str, Any]] = []
    for exact in tables["f6f1_metric_exactness_matrix"]:
        classification = exact["exactness_classification"]
        if classification not in allowed:
            raise ValueError(f"unexpected exactness classification: {classification}")
        metric_rows = rows_for_metric(exact["paper_metric"], tables)
        if classification.startswith("blocked") and has_numeric(metric_rows, ["value", "delta_llm_minus_baseline"]) == "true" and exact["paper_metric"] in {"popularity bias", "CandDif"}:
            paper_use = "blocked_or_diagnostic_only"
        elif classification in {"extension_metric", "local_proxy"}:
            paper_use = "no_paper_claim_extension_only"
        elif classification.startswith("blocked"):
            paper_use = "no_numeric_paper_claim"
        else:
            paper_use = "qualified_claim_allowed"
        rows.append(
            {
                "paper_metric": exact["paper_metric"],
                "local_implementation": exact["local_metric"],
                "formula_status": exact["formula_match"],
                "input_status": exact["input_match"],
                "population_status": exact["population_match"],
                "implementation_status": exact["implementation_match"],
                "final_exactness_classification": classification,
                "datasets_covered": unique_join(metric_rows, "dataset"),
                "tasks_covered": unique_join(metric_rows, "task"),
                "models_covered": unique_join(metric_rows, "model_name") or unique_join(metric_rows, "llm_model"),
                "baselines_covered": unique_join(metric_rows, "baseline_family"),
                "numeric_result_exists": has_numeric(metric_rows, ["value", "delta_llm_minus_baseline", "delta_reranking_minus_ranking"]),
                "paper_claim_use": paper_use,
                "principal_limitation": exact["limitation"],
            }
        )
    return rows


def ci_status(low: Any, high: Any, delta: Any) -> str:
    lo = parse_float(low)
    hi = parse_float(high)
    de = parse_float(delta)
    if lo is None or hi is None:
        return "not_applicable"
    if lo > 0 and hi > 0:
        return "positive_ci_excludes_zero"
    if lo < 0 and hi < 0:
        return "negative_ci_excludes_zero"
    if de is not None and de == 0:
        return "zero_delta"
    return "ci_crosses_zero"


def direction_from_delta(delta: Any) -> str:
    value = parse_float(delta)
    if value is None:
        return "not_numeric"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def strength_from_delta(delta: Any) -> str:
    value = parse_float(delta)
    if value is None:
        return "not_numeric"
    magnitude = abs(value)
    if magnitude >= 0.10:
        return "large"
    if magnitude >= 0.03:
        return "moderate"
    if magnitude > 0:
        return "small"
    return "zero"


def readout_row(
    prefix: str,
    readout_type: str,
    source_table: str,
    row: Mapping[str, str],
    *,
    comparator: str = "",
    value: str = "",
    comparison_value: str = "",
    delta: str = "",
    ci_low: str = "",
    ci_high: str = "",
    exactness: str = "",
    allowed: str = "",
    limitation: str = "",
) -> dict[str, Any]:
    model_name = row.get("model_name", row.get("llm_model", row.get("model", "")))
    baseline = row.get("baseline_family", "")
    metric = row.get("metric", "")
    actual_delta = delta or row.get("delta_llm_minus_baseline", row.get("delta_reranking_minus_ranking", row.get("delta_right_minus_left", "")))
    return {
        "readout_id": stable_id(prefix, source_table, readout_type, row.get("result_id", row.get("comparison_id", row.get("analysis_id", ""))), metric, comparator),
        "readout_type": readout_type,
        "source_table": source_table,
        "scope": row.get("scope", ""),
        "dataset": row.get("dataset", ""),
        "task": row.get("task", ""),
        "model_name": model_name,
        "strategy_id": row.get("strategy_id", ""),
        "strategy_label": row.get("strategy_label", ""),
        "baseline_family": baseline,
        "comparator": comparator or row.get("baseline_model", row.get("right_label", "")),
        "population": row.get("population", ""),
        "metric": metric,
        "value": value or row.get("value", row.get("llm_value", row.get("left_mean", ""))),
        "comparison_value": comparison_value or row.get("baseline_value", row.get("reranking_value", row.get("right_mean", ""))),
        "delta": actual_delta,
        "ci_low": ci_low or row.get("bootstrap_ci_low", ""),
        "ci_high": ci_high or row.get("bootstrap_ci_high", ""),
        "paired_intersection_size": row.get("paired_intersection_size", row.get("n_strict_intersection", "")),
        "denominator": row.get("denominator", row.get("ranking_denominator", "")),
        "exactness_classification": exactness or row.get("classification", row.get("exactness_classification", "")),
        "statistical_status": ci_status(ci_low or row.get("bootstrap_ci_low", ""), ci_high or row.get("bootstrap_ci_high", ""), actual_delta),
        "direction": direction_from_delta(actual_delta) if actual_delta else ("value_reported" if value or row.get("value") else "not_numeric"),
        "strength": strength_from_delta(actual_delta) if actual_delta else "not_applicable",
        "allowed_interpretation": allowed,
        "limitation": limitation,
        "notes": row.get("notes", ""),
    }


def build_aplt_readout(tables: Mapping[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    rows = [
        readout_row("F6F2-APLT", "cell_value", "phase_f6f1_aplt_results.csv", row, allowed="Qualified adapted long-tail claim.", limitation=row.get("notes", ""))
        for row in tables["f6f1_aplt_results"]
    ]
    rows.extend(
        readout_row("F6F2-APLT", "paired_llm_vs_baseline_delta", "phase_f6f1_llm_vs_baseline.csv", row, comparator=row.get("baseline_family", ""), allowed="LLM minus baseline APLT on paired strict-valid intersections.", limitation=row.get("notes", ""))
        for row in tables["f6f1_llm_vs_baseline"]
        if row.get("metric") == "APLT@5"
    )
    return rows


def build_serendipity_readout(tables: Mapping[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    rows = [
        readout_row("F6F2-SER", "cell_value", "phase_f6f1_serendipity_results.csv", row, allowed="Reconstructed MostPop-hit serendipity claim only.", limitation="Paper-exact serendipity remains blocked.")
        for row in tables["f6f1_serendipity_results"]
    ]
    rows.extend(
        readout_row("F6F2-SER", "paired_llm_vs_baseline_delta", "phase_f6f1_llm_vs_baseline.csv", row, comparator=row.get("baseline_family", ""), allowed="LLM minus baseline reconstructed serendipity on paired rows.", limitation=row.get("notes", ""))
        for row in tables["f6f1_llm_vs_baseline"]
        if row.get("metric") == "Serendipity@5"
    )
    return rows


def build_popularity_novelty_readout(tables: Mapping[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    allowed_by_metric = {
        "average_popularity_at_5": "Average raw train-count popularity; lower means less popular.",
        "novelty_self_information_at_5": "Paper-style reconstructed self-information novelty; higher means more novel.",
        "novelty_interaction_self_information_at_5": "Local interaction-count novelty proxy; not a paper-exact result.",
        "catalog_coverage_at_5": "Extension coverage diagnostic.",
        "item_coverage_at_5": "Extension candidate-union coverage diagnostic.",
        "duplicate_rate_at_5": "Extension reliability diagnostic; lower is better.",
        "strict_valid_rate": "Extension output reliability diagnostic; higher is better.",
        "invalid_list_rate": "Extension output reliability diagnostic; lower is better.",
        "operational_hr_at_5": "Utility diagnostic using F6E4C invalid-as-zero policy.",
        "operational_ndcg_at_5": "Utility diagnostic using F6E4C invalid-as-zero policy.",
    }
    rows = [
        readout_row("F6F2-POP", "cell_value", "phase_f6f1_popularity_novelty_results.csv", row, allowed=allowed_by_metric.get(row.get("metric", ""), "Qualified metric-specific claim."), limitation=row.get("notes", ""))
        for row in tables["f6f1_popularity_novelty_results"]
    ]
    rows.extend(
        readout_row("F6F2-STRUCT", "structural_diagnostic", "phase_f6f1_structural_diagnostics.csv", row, allowed=allowed_by_metric.get(row.get("metric", ""), "Structural diagnostic only."), limitation=row.get("notes", ""))
        for row in tables["f6f1_structural_diagnostics"]
    )
    rows.extend(
        readout_row("F6F2-POP", "paired_llm_vs_baseline_delta", "phase_f6f1_llm_vs_baseline.csv", row, comparator=row.get("baseline_family", ""), allowed="Paired strict-valid LLM minus baseline comparison.", limitation=row.get("notes", ""))
        for row in tables["f6f1_llm_vs_baseline"]
        if row.get("metric") in {"Average Popularity@5", "Novelty SelfInformation@5"}
    )
    return rows


def build_candidate_position_readout(tables: Mapping[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in tables["f6f1_candidate_position_results"]:
        allowed = "Observational position-stratum evidence only; no causal candidate-position claim."
        if "canddif" in row.get("analysis_type", "").lower():
            allowed = "Diagnostic CandDif only; full F6E4C controlled CandDif remains blocked."
        rows.append(readout_row("F6F2-POS", row.get("analysis_type", ""), "phase_f6f1_candidate_position_results.csv", row, value=row.get("value", ""), allowed=allowed, limitation=row.get("canddif_formula_status", "")))
    return rows


def build_ranking_reranking_readout(tables: Mapping[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    return [
        readout_row("F6F2-RVR", "ranking_vs_reranking_delta", "phase_f6f1_ranking_vs_reranking.csv", row, comparator="reranking", allowed="Adapted same-user ranking-vs-reranking comparison; LastFM reranking excluded.", limitation=row.get("notes", ""))
        for row in tables["f6f1_ranking_vs_reranking"]
    ]


def build_llm_baseline_readout(tables: Mapping[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    return [
        readout_row("F6F2-LVB", "paired_llm_vs_baseline_delta", "phase_f6f1_llm_vs_baseline.csv", row, comparator=row.get("baseline_family", ""), allowed="Paired adapted LLM minus baseline comparison only.", limitation=row.get("notes", ""))
        for row in tables["f6f1_llm_vs_baseline"]
    ]


def build_strategy_model_readout(tables: Mapping[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_key, source_name, readout_type in [
        ("f6e4c_strategy_paired_comparison", "phase_f6e4c_strategy_paired_comparison.csv", "base_vs_llmrank"),
        ("f6e4c_model_paired_comparison", "phase_f6e4c_model_paired_comparison.csv", "qwen2_vs_mistral"),
    ]:
        for row in tables[source_key]:
            rows.append(
                readout_row(
                    "F6F2-STRAT",
                    readout_type,
                    source_name,
                    row,
                    comparator=row.get("right_label", ""),
                    value=row.get("left_mean", ""),
                    comparison_value=row.get("right_mean", ""),
                    delta=row.get("delta_right_minus_left", ""),
                    ci_low=row.get("bootstrap_ci_low", ""),
                    ci_high=row.get("bootstrap_ci_high", ""),
                    exactness="exact_formula_adapted_population",
                    allowed="Adapted paired utility/reliability/runtime comparison.",
                    limitation=row.get("notes", ""),
                )
            )
    return rows


def numeric_summary(rows: Sequence[Mapping[str, str]], delta_column: str) -> dict[str, Any]:
    values = [parse_float(row.get(delta_column, "")) for row in rows]
    numbers = [value for value in values if value is not None]
    positives = sum(1 for value in numbers if value > 0)
    negatives = sum(1 for value in numbers if value < 0)
    zeros = sum(1 for value in numbers if value == 0)
    supported = sum(1 for row in rows if ci_status(row.get("bootstrap_ci_low", ""), row.get("bootstrap_ci_high", ""), row.get(delta_column, "")) in {"positive_ci_excludes_zero", "negative_ci_excludes_zero"})
    return {
        "n": len(numbers),
        "mean": sum(numbers) / len(numbers) if numbers else None,
        "min": min(numbers) if numbers else None,
        "max": max(numbers) if numbers else None,
        "positive": positives,
        "negative": negatives,
        "zero": zeros,
        "supported": supported,
    }


def dominant_direction(summary: Mapping[str, Any]) -> str:
    if summary["positive"] > summary["negative"]:
        return "mostly_positive"
    if summary["negative"] > summary["positive"]:
        return "mostly_negative"
    if summary["positive"] == summary["negative"] == 0:
        return "no_numeric_delta"
    return "mixed"


def finding_row(finding_id: str, question: str, source: str, metric: str, population: str, scope: str, summary: Mapping[str, Any], classification: str, interpretation: str, limitation: str) -> dict[str, Any]:
    return {
        "finding_id": finding_id,
        "question": question,
        "source_table": source,
        "metric": metric,
        "population": population,
        "scope": scope,
        "n_rows": summary["n"],
        "mean_value": "",
        "mean_delta": fmt(summary["mean"]),
        "min_delta": fmt(summary["min"]),
        "max_delta": fmt(summary["max"]),
        "positive_delta_count": summary["positive"],
        "negative_delta_count": summary["negative"],
        "zero_delta_count": summary["zero"],
        "supported_ci_count": summary["supported"],
        "direction": dominant_direction(summary),
        "strength": strength_from_delta(summary["mean"]),
        "evidence_classification": classification,
        "allowed_interpretation": interpretation,
        "limitation": limitation,
    }


def build_key_numeric_findings(tables: Mapping[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    llm = [row for row in tables["f6f1_llm_vs_baseline"] if row.get("classification") != "blocked_missing_input"]
    rvr = [row for row in tables["f6f1_ranking_vs_reranking"] if row.get("classification") != "not_applicable"]
    strategy = tables["f6e4c_strategy_paired_comparison"]
    model = tables["f6e4c_model_paired_comparison"]
    findings = [
        finding_row("F6F2-FIND-001", "Are LLMs less accurate than traditional baselines?", "phase_f6f1_llm_vs_baseline.csv", "HR@5", "strict_valid_paired_intersection", "all available dataset/task cells", numeric_summary([row for row in llm if row.get("metric") == "HR@5"], "delta_llm_minus_baseline"), "supported_in_adapted_setting", "Negative deltas mean the LLM is below the paired baseline on HR@5.", "Adapted local candidate/order intersections only."),
        finding_row("F6F2-FIND-002", "Are LLMs less accurate than traditional baselines?", "phase_f6f1_llm_vs_baseline.csv", "NDCG@5", "strict_valid_paired_intersection", "all available dataset/task cells", numeric_summary([row for row in llm if row.get("metric") == "NDCG@5"], "delta_llm_minus_baseline"), "supported_in_adapted_setting", "Negative deltas mean the LLM is below the paired baseline on NDCG@5.", "Adapted local candidate/order intersections only."),
        finding_row("F6F2-FIND-003", "Do LLMs recommend more long-tail items?", "phase_f6f1_llm_vs_baseline.csv", "APLT@5", "strict_valid_paired_intersection", "all available dataset/task cells", numeric_summary([row for row in llm if row.get("metric") == "APLT@5"], "delta_llm_minus_baseline"), "partially_supported", "Positive deltas mean LLM top-5 lists contain more q80 long-tail items than the paired baseline.", "APLT is exact formula in adapted local population."),
        finding_row("F6F2-FIND-004", "Are LLMs more novel?", "phase_f6f1_llm_vs_baseline.csv", "Novelty SelfInformation@5", "strict_valid_paired_intersection", "all available dataset/task cells", numeric_summary([row for row in llm if row.get("metric") == "Novelty SelfInformation@5"], "delta_llm_minus_baseline"), "partially_supported", "Positive deltas mean higher reconstructed self-information novelty.", "Novelty uses reconstructed unique-user inputs."),
        finding_row("F6F2-FIND-005", "Are LLMs more serendipitous?", "phase_f6f1_llm_vs_baseline.csv", "Serendipity@5", "strict_valid_paired_intersection", "paired MostPop-reference rows", numeric_summary([row for row in llm if row.get("metric") == "Serendipity@5"], "delta_llm_minus_baseline"), "partially_supported", "Positive deltas mean higher reconstructed MostPop-hit serendipity.", "Paper-exact serendipity remains blocked."),
        finding_row("F6F2-FIND-006", "Is re-ranking better than ranking?", "phase_f6f1_ranking_vs_reranking.csv", "HR@5", "paired ranking/reranking", "Beauty, Sports, ML-1M; LastFM excluded", numeric_summary([row for row in rvr if row.get("metric") == "HR@5"], "delta_reranking_minus_ranking"), "partially_supported", "Positive deltas favor re-ranking.", "Candidate-pool recall and positive-present rates must be kept separate from utility."),
        finding_row("F6F2-FIND-007", "Is LLMRank better than Base?", "phase_f6e4c_strategy_paired_comparison.csv", "hr_at5", "paired F6E4C strategy comparison", "all F6E4C cells", numeric_summary([row for row in strategy if row.get("metric") == "hr_at5"], "delta_right_minus_left"), "partially_supported", "Positive deltas favor the right-label strategy.", "Core adapter differences and invalid-output rates remain part of interpretation."),
        finding_row("F6F2-FIND-008", "Does model family affect utility?", "phase_f6e4c_model_paired_comparison.csv", "hr_at5", "paired F6E4C model comparison", "all F6E4C cells", numeric_summary([row for row in model if row.get("metric") == "hr_at5"], "delta_right_minus_left"), "partially_supported", "Nonzero deltas show model-family utility differences in the adapted matrix.", "No universal model superiority claim is allowed."),
    ]
    for row in findings:
        if row["finding_id"] == "F6F2-FIND-003" and row["direction"] == "mostly_positive":
            row["evidence_classification"] = "supported_in_adapted_setting"
        if row["finding_id"] in {"F6F2-FIND-005", "F6F2-FIND-006", "F6F2-FIND-007"} and row["direction"] == "mostly_negative":
            row["evidence_classification"] = "opposite_result_in_adapted_setting"
    return findings


def rq(row_id: str, claim: str, local: str, tables: str, models: str, datasets: str, tasks: str, metrics: str, exactness: str, direction: str, classification: str, allowed: str, prohibited: str, limitation: str, next_evidence: str) -> dict[str, Any]:
    return {
        "rq_id": row_id,
        "paper_claim": claim,
        "local_question": local,
        "evidence_tables": tables,
        "models": models,
        "datasets": datasets,
        "tasks": tasks,
        "metrics": metrics,
        "exactness_level": exactness,
        "result_direction": direction,
        "final_evidence_classification": classification,
        "allowed_conclusion": allowed,
        "prohibited_conclusion": prohibited,
        "principal_limitation": limitation,
        "next_evidence_required": next_evidence,
    }


def build_research_question_closure(findings: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    direction_by_id = {row["finding_id"]: row["direction"] for row in findings}
    classification_by_id = {row["finding_id"]: row["evidence_classification"] for row in findings}
    return [
        rq("RQ1", "LLMs are weaker than traditional recommenders on utility.", "LLM utility versus traditional recommenders.", "phase_f6f1_llm_vs_baseline.csv", "Qwen2, Mistral", "Amazon Beauty, Amazon Sports, ML-1M, LastFM where paired", "ranking, reranking where available", "HR@5, NDCG@5", "exact_formula_adapted_population", f"HR {direction_by_id.get('F6F2-FIND-001','')}; NDCG {direction_by_id.get('F6F2-FIND-002','')}", "supported_in_adapted_setting", "In the adapted paired setting, LLM utility is generally below traditional baselines.", "Do not claim paper-exact universal underperformance.", "Local candidate/order intersections and reconstructed baselines.", "Paper-exact candidate pools and model snapshots."),
        rq("RQ2", "Domain and dataset properties affect outcomes.", "Domain-knowledge or dataset-domain effect.", "phase_f6e4c_dataset_task_summary.csv; phase_f6f1_llm_vs_baseline.csv", "Qwen2, Mistral", "Beauty, Sports, ML-1M, LastFM", "ranking, reranking", "HR@5, NDCG@5, reliability", "exact_formula_adapted_population", "dataset-dependent", "partially_supported", "Dataset/task differences are visible in the adapted matrix.", "Do not claim domain knowledge was isolated causally.", "No controlled domain-only intervention.", "Controlled domain-factor experiment."),
        rq("RQ3", "LLMs recommend more long-tail items.", "Long-tail/APLT behavior.", "phase_f6f1_aplt_results.csv; phase_f6f1_llm_vs_baseline.csv", "Qwen2, Mistral", "available paired datasets", "ranking, reranking", "APLT@5", "exact_formula_adapted_population", direction_by_id.get("F6F2-FIND-003", ""), classification_by_id.get("F6F2-FIND-003", "partially_supported"), "APLT supports a qualified adapted long-tail readout where deltas are positive.", "Do not rename APLT as popularity bias.", "Adapted train-population and candidate-pool definitions.", "Paper-exact data snapshots."),
        rq("RQ4", "LLMs are more novel.", "Novelty behavior.", "phase_f6f1_popularity_novelty_results.csv; phase_f6f1_llm_vs_baseline.csv", "Qwen2, Mistral", "available paired datasets", "ranking, reranking", "Novelty SelfInformation@5", "exact_formula_reconstructed_inputs", direction_by_id.get("F6F2-FIND-004", ""), "partially_supported", "Novelty is reported only for reconstructed self-information inputs.", "Do not treat local interaction-count novelty as the paper metric.", "Cold/missing popularity support may block exact rows.", "PDF-verified novelty formula and exact train-user counts."),
        rq("RQ5", "LLMs are more serendipitous.", "Serendipity behavior.", "phase_f6f1_serendipity_results.csv; phase_f6f1_llm_vs_baseline.csv", "Qwen2, Mistral", "paired MostPop-reference datasets", "ranking, reranking where reference exists", "Serendipity@5", "reconstructed_formula", direction_by_id.get("F6F2-FIND-005", ""), classification_by_id.get("F6F2-FIND-005", "partially_supported"), "Only reconstructed MostPop-hit serendipity can be claimed.", "Do not call the serendipity result paper-exact.", "Paper prose and official-code primitive are not fully reconciled.", "PDF/manual formula verification."),
        rq("RQ6", "Short-history and cold-start behavior matters.", "Short-history or cold-start behavior.", "phase_f6e4c_history_length_analysis.csv", "Qwen2, Mistral", "F6E4C datasets", "ranking, reranking", "HR@5, NDCG@5 by history stratum", "exact_formula_adapted_population", "adapted strata only", "partially_supported", "History-stratum observations exist in F6E4C.", "Do not claim L=0 cold-start replication.", "No controlled L=0 profile-only condition.", "Controlled history truncation and L=0 cohort."),
        rq("RQ7", "Candidate position bias affects LLM recommendations.", "Candidate-position bias.", "phase_f6f1_candidate_position_results.csv", "Qwen2, Mistral", "F6E4C datasets", "ranking, reranking where present", "HR@5, NDCG@5, final rank by initial position", "exact_formula_adapted_population plus blocked CandDif", "observational trend", "partially_supported", "Position-stratum evidence is observational.", "Do not claim causal CandDif for F6E4C.", "No full-matrix controlled positive-first/random reruns.", "Controlled candidate-order permutations."),
        rq("RQ8", "Profiles improve recommendation effectiveness.", "Profile-generation effectiveness.", "not available in F6F1", "not available", "not available", "not available", "profile-only/profile-plus-history utility", "not_applicable", "not_tested", "not_tested", "No F6F1 evidence supports profile-generation claims.", "Do not claim profile generation was reproduced.", "Profile-only and profile-plus-history workflows were not run.", "Profile generation experiment with frozen prompts."),
        rq("RQ9", "LLM outputs may hallucinate or be unreliable.", "Hallucination and output reliability.", "phase_f6f1_structural_diagnostics.csv; phase_f6e4c_invalid_output_audit.csv", "Qwen2, Mistral", "F6E4C datasets", "ranking, reranking", "strict_valid_rate, invalid_list_rate, duplicate_rate_at_5", "extension_metric", "measured reliability differences", "supported_in_adapted_setting", "Output reliability is measurable and must be separated from utility.", "Do not claim hallucination was eliminated.", "Reliability diagnostics are extensions.", "Manual hallucination taxonomy or exact paper parser audit."),
        rq("RQ10", "LLMs perform better in re-ranking than ranking.", "Ranking versus re-ranking.", "phase_f6f1_ranking_vs_reranking.csv", "Qwen2, Mistral", "Beauty, Sports, ML-1M", "ranking, reranking", "HR@5, NDCG@5, candidate-pool recall", "exact_formula_adapted_population", direction_by_id.get("F6F2-FIND-006", ""), classification_by_id.get("F6F2-FIND-006", "partially_supported"), "Re-ranking evidence is adapted and excludes LastFM.", "Do not treat candidate-pool recall as model utility.", "LastFM reranking missing.", "F6G1 LastFM reranking reconstruction."),
        rq("RQ11", "LLMRank improves over Base.", "Base versus LLMRank.", "phase_f6e4c_strategy_paired_comparison.csv; F6F1 readouts", "Qwen2, Mistral", "F6E4C datasets", "ranking, reranking", "HR@5, NDCG@5, APLT, novelty, reliability", "exact_formula_adapted_population plus extensions", direction_by_id.get("F6F2-FIND-007", ""), classification_by_id.get("F6F2-FIND-007", "partially_supported"), "Base-vs-LLMRank differs by model/dataset/task.", "Do not claim LLMRank is always better or worse.", "Adapter and invalid-output effects must remain visible.", "Controlled adapter ablation."),
        rq("RQ12", "Structured indexed output adaptation is effective.", "Structured indexed output adaptation.", "phase_f6e4c_cell_summary.csv; phase_f6f1_structural_diagnostics.csv", "Qwen2, Mistral", "F6E4C datasets", "ranking, reranking", "strict_valid_rate, invalid_list_rate", "extension_metric", "supported parser reliability", "supported_in_adapted_setting", "Structured indexed output enables offline parsing and reliability measurement.", "Do not claim all hallucination was eliminated.", "This is an adaptation, not the paper's exact raw-output protocol.", "Exact paper parser and raw-output protocol replication."),
    ]


def claim(claim_id: str, text: str, evidence: str, level: str, exactness: str, stat: str, allowed: str, prohibited: str, limitation: str, scope: str) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "claim_text": text,
        "evidence": evidence,
        "evidence_level": level,
        "exactness_classification": exactness,
        "statistical_status": stat,
        "allowed_wording": allowed,
        "prohibited_wording": prohibited,
        "limitation": limitation,
        "scope": scope,
    }


def build_claim_registry(findings: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        claim("F6F2-CLAIM-001", "Adapted LLM utility is generally below paired traditional baselines.", "phase_f6f2_llm_baseline_readout.csv", "paired numeric evidence", "exact_formula_adapted_population", "bootstrap CIs reported per row", "In our adapted local setting, LLM utility is generally below paired traditional baselines.", "We exactly replicated every paper utility result.", "Candidate/order intersections are local and adapted.", "F6F1 paired intersections"),
        claim("F6F2-CLAIM-002", "APLT long-tail behavior is recoverable and must be interpreted separately from popularity bias.", "phase_f6f2_aplt_readout.csv", "source-backed metric evidence", "exact_formula_adapted_population", "bootstrap CIs reported for paired deltas", "The result supports a qualified adapted long-tail readout.", "LLMs eliminate popularity bias.", "APLT is not PopREO/popularity bias.", "F6F1 APLT"),
        claim("F6F2-CLAIM-003", "Novelty self-information is reconstructed where train-user inputs exist.", "phase_f6f2_popularity_novelty_readout.csv", "reconstructed-input evidence", "exact_formula_reconstructed_inputs", "bootstrap CIs reported for paired deltas", "The metric uses the paper formula with reconstructed inputs.", "Coverage proves novelty.", "Local interaction-count novelty remains a proxy.", "F6F1 novelty"),
        claim("F6F2-CLAIM-004", "Serendipity is reconstructed with a MostPop-hit reference, not paper-exact.", "phase_f6f2_serendipity_readout.csv", "reconstructed-formula evidence", "reconstructed_formula", "bootstrap CIs reported for paired deltas", "This is reconstructed serendipity under the frozen MostPop-hit policy.", "The paper-exact serendipity result was fully reproduced.", "Paper prose and code evidence remain unresolved.", "F6F1 serendipity"),
        claim("F6F2-CLAIM-005", "Candidate-position evidence is observational; full CandDif is blocked.", "phase_f6f2_candidate_position_readout.csv", "observational strata plus diagnostic block", "exact_formula_adapted_population", "not_applicable", "This result is observational rather than causal.", "Candidate position causally determines the result.", "No full controlled positive-first/random reruns.", "F6F1 candidate position"),
        claim("F6F2-CLAIM-006", "Ranking-vs-reranking evidence is adapted and excludes LastFM reranking.", "phase_f6f2_ranking_reranking_readout.csv", "paired adapted comparison", "exact_formula_adapted_population", "row-level deltas reported", "The paper finding is only partially supported in the adapted setting.", "Re-ranking is universally better.", "LastFM reranking is missing.", "Beauty/Sports/ML-1M"),
        claim("F6F2-CLAIM-007", "Base-vs-LLMRank effects vary by model, dataset, and task.", "phase_f6f2_strategy_model_readout.csv", "paired F6E4C evidence", "exact_formula_adapted_population", "bootstrap CIs reported per row", "LLMRank behavior is cell-dependent in this local setting.", "LLMRank is always worse.", "Adapter and invalid-output conflicts remain visible.", "F6E4C/F6F1"),
        claim("F6F2-CLAIM-008", "Qwen2 and Mistral differ, but no universal model superiority claim is supported.", "phase_f6f2_strategy_model_readout.csv", "paired F6E4C evidence", "exact_formula_adapted_population", "bootstrap CIs reported per row", "Model-family effects are observed in the adapted matrix.", "Mistral is always better.", "Mistral LLMRank LastFM conflict requires qualification.", "F6E4C/F6F1"),
        claim("F6F2-CLAIM-009", "Output reliability is a measured extension, not a paper-exact utility result.", "phase_f6f2_popularity_novelty_readout.csv", "structural diagnostic evidence", "extension_metric", "not_applicable", "Strict-validity and invalid-list rates are reliability diagnostics.", "Hallucination was eliminated.", "No manual hallucination taxonomy is frozen.", "F6F1 structural diagnostics"),
        claim("F6F2-CLAIM-010", "The F6F1 scientific outputs are reproducible across the recovered environment.", "phase_f6f1r_environment_freeze.json", "cross-environment byte/scientific comparison", "not_applicable", "byte-identical CSVs; runtime metadata differs", "F6F1 scientific CSVs are byte-identical across environments.", "The fallback environment is fully isolated.", "Environment uses system-site packages.", "F6F1R"),
    ]


def validate_claims(rows: Sequence[Mapping[str, Any]], prohibited: Sequence[str]) -> None:
    banned = [phrase.lower() for phrase in prohibited]
    for row in rows:
        checked = f"{row.get('claim_text','')} {row.get('allowed_wording','')}".lower()
        for phrase in banned:
            if phrase in checked:
                raise ValueError(f"unsupported prohibited wording in claim {row.get('claim_id')}: {phrase}")


def build_remaining_gaps() -> list[dict[str, Any]]:
    raw = [
        ("GAP-001", "LastFM re-ranking", "paper scope", "blocked", "No frozen LastFM reranking LLM matrix.", "LastFM reranking cohort and candidate pools", "offline_light", "F6G1", 1),
        ("GAP-002", "controlled history truncation", "paper scope", "not_tested", "No controlled truncation sweep.", "history-stratified reruns", "inference_required", "future phase", 4),
        ("GAP-003", "L=0 cold-start condition", "paper scope", "not_tested", "No profile-only zero-history cohort.", "L=0 prompts and cohorts", "inference_required", "future phase", 5),
        ("GAP-004", "retraining baselines by history length", "paper scope", "not_tested", "Plan E rankings are not retrained by history stratum.", "history-specific training splits", "training_required", "future phase", 6),
        ("GAP-005", "exact LLMSRec orchestration", "paper scope", "blocked", "Original orchestration not fully runnable locally.", "exact scripts/model snapshots", "blocked_unavailable", "manual audit", 9),
        ("GAP-006", "profile-only evaluation", "paper scope", "not_tested", "No profile-only prompts were run.", "profile generation outputs", "inference_required", "future phase", 7),
        ("GAP-007", "profile-plus-history evaluation", "paper scope", "not_tested", "No profile-plus-history prompts were run.", "profile/history prompts", "inference_required", "future phase", 8),
        ("GAP-008", "Mistral exact version uncertainty", "environment/model scope", "documented", "Local model snapshot cannot be proven paper-exact.", "model digest/version evidence", "blocked_unavailable", "manual audit", 10),
        ("GAP-009", "exact candidate permutations for position bias", "paper scope", "blocked", "F6E4C has one frozen order per request.", "controlled first/random reruns", "inference_required", "future phase", 3),
        ("GAP-010", "exact paper dataset snapshots", "paper scope", "blocked", "Original exact data snapshots unavailable.", "paper snapshot hashes", "blocked_unavailable", "data audit", 11),
        ("GAP-011", "exact model snapshots", "paper scope", "blocked", "Exact paper LLM weights/endpoints unavailable.", "model IDs/digests/API versions", "external_api_required", "manual audit", 12),
        ("GAP-012", "fairness/diversity appendix metrics", "appendix scope", "not_tested", "Not part of F6F1 supported metrics.", "appendix formulas and inputs", "offline_light", "future phase", 13),
        ("GAP-013", "fifth dataset course extension", "extension scope", "not_tested", "No course dataset artifacts are frozen.", "course data and candidates", "offline_heavy", "future phase", 14),
        ("GAP-014", "modern local model extension", "extension scope", "not_tested", "No modern model inference was run.", "approved local model and outputs", "inference_required", "future phase", 15),
    ]
    return [
        {
            "gap_id": item[0],
            "gap": item[1],
            "paper_or_extension_scope": item[2],
            "current_status": item[3],
            "blocking_reason": item[4],
            "required_inputs": item[5],
            "estimated_computational_class": item[6],
            "dependency_on_phase": item[7],
            "recommended_order": item[8],
        }
        for item in raw
    ]


def build_next_phase_requirements() -> list[dict[str, Any]]:
    raw = [
        ("F6G1-REQ-001", "required LastFM user cohort", "Use the frozen F6D/F6C LastFM ranking cohort as the starting user set.", "No cohort expansion without a new manifest.", "finalization_f6c_lastfm_ranking_cohort", "reuse_existing", "false", "exact_formula_adapted_population", "Stop if cohort hashes differ."),
        ("F6G1-REQ-002", "positive-item policy", "Preserve each user's frozen positive item and record positive-present status.", "Positive-absent rows stay separated.", "F6D3/F6D4A manifests", "reuse_existing", "false", "exact_formula_adapted_population", "Stop if positive IDs cannot be reconciled."),
        ("F6G1-REQ-003", "re-ranking candidate-pool construction", "Construct LastFM reranking pools from frozen ranking outputs without changing training data.", "Candidate-pool provenance hash required.", "F6D1/F6D2 rankings", "reuse_possible", "false", "reconstructed_formula", "Stop if duplicate or missing item policy is ambiguous."),
        ("F6G1-REQ-004", "source traditional models", "Use MostPop, BM25-like, BPRMF, GRU4Rec, SASRec, and LightGCN where frozen scores exist.", "Do not retrain in F6G1 unless explicitly approved.", "Plan E and F6D predictions", "reuse_possible", "false", "exact_formula_adapted_population", "Stop if model score coverage is insufficient."),
        ("F6G1-REQ-005", "candidate-pool size", "Declare fixed reranking pool size before running any LLM step.", "No post-hoc pool resizing.", "F6G1 config", "new_config_only", "false", "not_applicable", "Stop if pool size cannot be matched across models."),
        ("F6G1-REQ-006", "duplicate handling", "Deduplicate by item_id after preserving first occurrence and report removed duplicates.", "Never silently drop positives.", "candidate construction table", "new_artifact", "false", "not_applicable", "Stop if positive item is removed."),
        ("F6G1-REQ-007", "positive-present, recall, upper bound", "Report positive-present rate, candidate-pool recall, and utility upper bound before any utility claim.", "Do not treat candidate recall as model quality.", "candidate construction table", "new_artifact", "false", "extension_metric", "Stop if upper bound is below claimed HR."),
        ("F6G1-REQ-008", "train/test boundaries", "Reuse frozen train/test split boundaries from LastFM finalization.", "No leakage from test positives into train popularity.", "finalization_f6c_lastfm_train/test", "reuse_existing", "false", "not_applicable", "Stop if split hashes differ."),
        ("F6G1-REQ-009", "title/item reconciliation", "Map item IDs to titles using frozen LastFM catalog and record unresolved titles.", "No fuzzy title substitution without audit row.", "finalization_f6c_lastfm_catalog", "reuse_existing", "false", "not_applicable", "Stop if title coverage blocks prompt rendering."),
        ("F6G1-REQ-010", "comparison with paper", "Classify reranking evidence as adapted unless exact paper pools/model snapshots are proven.", "No paper-exact claim by default.", "paper/original_repo/F6G1 artifacts", "manual_evidence_needed", "false", "reconstructed_formula", "Stop if exactness classification conflicts."),
        ("F6G1-REQ-011", "reuse of F6D2 scores", "Reuse F6D2 trainable model scores when candidate coverage is sufficient.", "New traditional inference only if coverage gate fails.", "finalization_f6d2_lastfm_candidate_rankings", "reuse_possible", "false", "exact_formula_adapted_population", "Stop if score rows cannot cover candidate pool."),
        ("F6G1-REQ-012", "new LLM inference", "No new LLM inference in the comparability gate; later inference requires explicit phase approval.", "F6G1 should end before fresh LLM calls unless redirected.", "none", "not_in_f6g1_gate", "eventually_true", "not_applicable", "Stop before Ollama/API invocation."),
    ]
    return [
        {
            "requirement_id": item[0],
            "component": item[1],
            "requirement": item[2],
            "policy": item[3],
            "source_or_input": item[4],
            "reuse_status": item[5],
            "new_inference_required": item[6],
            "exactness_classification": item[7],
            "stop_condition": item[8],
        }
        for item in raw
    ]


def build_missing_or_blocked(tables: Mapping[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    rows = [
        {
            "item_id": row["item_id"],
            "component": row["metric_or_component"],
            "status": "blocked_or_documented",
            "classification": row["blocking_classification"],
            "reason": row["blocking_reason"],
            "required_evidence": row["required_evidence"],
            "current_evidence": row["current_evidence"],
            "recommended_action": row["recommended_action"],
        }
        for row in tables["f6f1_missing_or_blocked"]
    ]
    rows.append(
        {
            "item_id": "F6F2-BLOCK-001",
            "component": "fully isolated Python dependency rebuild",
            "status": "documented",
            "classification": "environment_limitation",
            "reason": "F6F1R validated a reconstructed venv with system-site packages, not a fully isolated wheel-based rebuild.",
            "required_evidence": "local wheel/cache bundle for all required packages",
            "current_evidence": "F6F1R scientific outputs are identical across environments",
            "recommended_action": "keep current environment frozen unless dependency-bundle reconstruction is requested",
        }
    )
    return rows


def mean_value(rows: Sequence[Mapping[str, str]], column: str) -> str:
    values = [parse_float(row.get(column, "")) for row in rows]
    numbers = [value for value in values if value is not None]
    return fmt(sum(numbers) / len(numbers)) if numbers else ""


def validate_research_questions(rows: Sequence[Mapping[str, Any]], allowed: Sequence[str]) -> None:
    ids = [row["rq_id"] for row in rows]
    if ids != [f"RQ{idx}" for idx in range(1, 13)]:
        raise ValueError("research-question closure must contain RQ1-RQ12 exactly")
    allowed_set = set(allowed)
    for row in rows:
        if row["final_evidence_classification"] not in allowed_set:
            raise ValueError(f"invalid RQ classification: {row['final_evidence_classification']}")


def validate_no_blocked_metric_promoted(metric_status: Sequence[Mapping[str, Any]]) -> None:
    for row in metric_status:
        if str(row["final_exactness_classification"]).startswith("blocked") and row["paper_claim_use"] == "qualified_claim_allowed":
            raise ValueError(f"blocked metric promoted to claim: {row['paper_metric']}")


def build_report(
    repository: Mapping[str, Any],
    interpreter: Mapping[str, str],
    input_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    counts: Mapping[str, int],
    findings: Sequence[Mapping[str, Any]],
) -> str:
    changed = sum(1 for row in input_rows if row["changed"] == "true")
    missing = sum(1 for row in input_rows if row["missing"] == "true")
    finding_summary = "; ".join(f"{row['finding_id']} {row['metric']} {row['direction']} mean_delta={row['mean_delta']}" for row in findings[:8])
    sections = [
        ("Initial repository state", f"Branch `{repository['branch']}`, HEAD `{repository['head']}`, staged files `{repository['staged_files'] or 'none'}`. Tracked dirty files remain `{repository['git_status_tracked_short']}`."),
        ("Environment used", f"`{interpreter['actual_interpreter']}` with classification `{manifest['environment_classification']}`."),
        ("Frozen-input verification", f"Rows `{counts['input_integrity']}`; changed `{changed}`, missing `{missing}`."),
        ("F6F1 scientific-hash verification", f"`{manifest['f6f1_scientific_hash']}`."),
        ("Metric exactness readout", f"Rows `{counts['metric_status']}`; no blocked metric is promoted to exact."),
        ("APLT findings", f"Rows `{counts['aplt_readout']}`. APLT uses the F6F1 q80 train-frequency policy."),
        ("Serendipity findings", f"Rows `{counts['serendipity_readout']}`. Serendipity remains reconstructed MostPop-hit evidence."),
        ("Popularity and novelty findings", f"Rows `{counts['popularity_novelty_readout']}`. Novelty, average popularity, APLT, and coverage stay separate."),
        ("Candidate-position findings", f"Rows `{counts['candidate_position_readout']}`. Position evidence is observational; full CandDif remains blocked."),
        ("CandDif status", "Only diagnostic or blocked CandDif evidence is allowed; no causal claim is made."),
        ("Ranking versus re-ranking findings", f"Rows `{counts['ranking_reranking_readout']}`; LastFM re-ranking remains excluded."),
        ("LLM versus baseline findings", f"Rows `{counts['llm_baseline_readout']}` with paired intersections and bootstrap CIs from F6F1."),
        ("Base versus LLMRank findings", "F6E4C paired strategy rows are reported in the strategy/model readout."),
        ("Qwen2 versus Mistral findings", "F6E4C paired model rows are reported with no universal superiority claim."),
        ("Research-question closure", f"Rows `{counts['research_question_closure']}` with exactly one classification per RQ."),
        ("Claim registry", f"Rows `{counts['claim_registry']}` with allowed and prohibited wording."),
        ("Remaining replication gaps", f"Rows `{counts['remaining_replication_gaps']}`."),
        ("F6G1 requirements", f"Rows `{counts['next_phase_requirements']}`."),
        ("Implementation files", "`src/analysis/plan_f6f2_scientific_readout.py`, `src/experiments/build_phase_f6f2_scientific_readout.py`, `configs/phase_f6f2_scientific_readout.yaml`."),
        ("Tests", stable_json(manifest.get("validation_summary", {}))),
        ("Excluded unsafe tests", ", ".join(manifest.get("unsafe_excluded_tests", []))),
        ("Determinism", "F6F2 writes sorted CSV/JSON outputs and separates runtime metadata from scientific content."),
        ("Protected-input verification", f"Input invariance changed `{changed}`, missing `{missing}`."),
        ("Scientific freeze manifest", "`results/manifests/phase_f6f2_scientific_readout_freeze.json`."),
        ("Missing or blocked evidence", f"Rows `{counts['missing_or_blocked']}`."),
        ("Final decision", manifest["decision"]),
        ("Recommended next action", manifest["recommendation"]),
        ("Final git state", "No staging, commit, reset, clean, checkout, or frozen-output mutation is performed."),
        ("Files created or modified", "Only new `phase_f6f2_*` outputs plus F6F2 config/source/test files."),
        ("Remaining limitations", "F6F2 is a scientific readout over frozen adapted evidence; it does not create new predictions or repair blocked exact paper evidence."),
        ("Key numeric summary", finding_summary),
    ]
    lines = ["# Phase F6F2 Scientific Readout", ""]
    for idx, (title, body) in enumerate(sections, start=1):
        lines.extend([f"## {idx}. {title}", body, ""])
    return "\n".join(lines).rstrip() + "\n"


def run_validation_suites(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    suites: list[dict[str, Any]] = []
    for name, files in F6F2_TEST_SUITES:
        command = [sys.executable, "-m", "pytest", *files, "-q"]
        code, out, err, _duration = run_command(command, root, timeout=360)
        text = out + "\n" + err
        passed = parse_count(text, "passed")
        failed = parse_count(text, "failed")
        skipped = parse_count(text, "skipped")
        suites.append(
            {
                "suite": name,
                "status": "pass" if code == 0 else "fail",
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "command": " ".join(command),
            }
        )
    return {
        "suites": suites,
        "total_passed": sum(row["passed"] for row in suites),
        "total_failed": sum(row["failed"] for row in suites),
        "total_skipped": sum(row["skipped"] for row in suites),
        "unsafe_exclusions": list(config.get("unsafe_excluded_tests", [])),
    }


def parse_count(text: str, word: str) -> int:
    match = re.search(rf"(\d+)\s+{word}", text)
    return int(match.group(1)) if match else 0


def table_hashes(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    return {name: sha256_file(root / path) for name, path in sorted(config["outputs"]["tables"].items())}


def build_manifest(
    root: Path,
    config: Mapping[str, Any],
    repository: Mapping[str, Any],
    interpreter: Mapping[str, str],
    input_rows: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    f6f1 = load_json(root / config["inputs"]["f6f1_manifest"])
    f6f1r = load_json(root / config["inputs"]["f6f1r_manifest"])
    f6e4c = load_json(root / config["inputs"]["f6e4c_manifest"])
    hashes = table_hashes(root, config)
    scientific = {
        "phase": PHASE_ID,
        "f6f1_scientific_hash": f6f1["scientific_content_hash"],
        "f6f1r_scientific_comparison_status": "identical" if f6f1r.get("frozen_f6f1_scientific_content_hash") == f6f1r.get("isolated_f6f1_scientific_content_hash") else "mismatch",
        "f6e4c_scientific_hash": f6e4c.get("scientific_content_hash", ""),
        "metric_status_hash": hashes["metric_status"],
        "numeric_findings_hash": hashes["key_numeric_findings"],
        "rq_closure_hash": hashes["research_question_closure"],
        "claim_registry_hash": hashes["claim_registry"],
        "remaining_gap_hash": hashes["remaining_replication_gaps"],
        "next_phase_requirements_hash": hashes["next_phase_requirements"],
        "input_integrity_hash": hashes["input_integrity"],
        "decision": config["phase"]["decision_complete_with_blocks"],
        "recommendation": config["phase"]["recommendation"],
    }
    return {
        **scientific,
        "scientific_content_hash": sha256_text(stable_json(scientific)),
        "direct_input_hashes": {row["path"]: row["actual_sha256"] for row in input_rows},
        "table_hashes": hashes,
        "f6f1_decision": f6f1.get("decision", ""),
        "f6f1r_decision": f6f1r.get("decision", ""),
        "f6e4c_decision": f6e4c.get("decision", ""),
        "environment_classification": config["phase"]["environment_classification"],
        "interpreter_gate": dict(interpreter),
        "validation_summary": validation,
        "unsafe_excluded_tests": list(config.get("unsafe_excluded_tests", [])),
        "input_invariance": {
            "changed": sum(1 for row in input_rows if row["changed"] == "true"),
            "missing": sum(1 for row in input_rows if row["missing"] == "true"),
        },
        "runtime_metadata": {
            "timestamp": repository["timestamp"],
            "repository_state": dict(repository),
            "python": sys.executable,
            "platform": platform.platform(),
        },
    }


def run_phase(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    repository = record_repository_state(root, config)
    interpreter = verify_interpreter(root, config)
    if interpreter["status"] != "pass":
        raise RuntimeError(f"F6F2 interpreter gate failed: {interpreter}")
    validate_columns(root, config)
    input_rows = verify_input_integrity(root, config)
    tables = load_tables(root, config)

    metric_status = build_metric_status(tables, config)
    validate_no_blocked_metric_promoted(metric_status)
    aplt = build_aplt_readout(tables)
    serendipity = build_serendipity_readout(tables)
    popularity = build_popularity_novelty_readout(tables)
    candidate = build_candidate_position_readout(tables)
    ranking = build_ranking_reranking_readout(tables)
    llm_baseline = build_llm_baseline_readout(tables)
    strategy_model = build_strategy_model_readout(tables)
    findings = build_key_numeric_findings(tables)
    rq_rows = build_research_question_closure(findings)
    validate_research_questions(rq_rows, config["allowed_research_question_classifications"])
    claims = build_claim_registry(findings)
    validate_claims(claims, config["prohibited_wording"])
    gaps = build_remaining_gaps()
    next_phase = build_next_phase_requirements()
    missing = build_missing_or_blocked(tables)

    table_rows = {
        "input_integrity": input_rows,
        "metric_status": metric_status,
        "key_numeric_findings": findings,
        "aplt_readout": aplt,
        "serendipity_readout": serendipity,
        "popularity_novelty_readout": popularity,
        "candidate_position_readout": candidate,
        "ranking_reranking_readout": ranking,
        "llm_baseline_readout": llm_baseline,
        "strategy_model_readout": strategy_model,
        "research_question_closure": rq_rows,
        "claim_registry": claims,
        "remaining_replication_gaps": gaps,
        "next_phase_requirements": next_phase,
        "missing_or_blocked": missing,
    }
    table_columns = {
        "input_integrity": INPUT_INTEGRITY_COLUMNS,
        "metric_status": METRIC_STATUS_COLUMNS,
        "key_numeric_findings": KEY_FINDING_COLUMNS,
        "aplt_readout": READOUT_COLUMNS,
        "serendipity_readout": READOUT_COLUMNS,
        "popularity_novelty_readout": READOUT_COLUMNS,
        "candidate_position_readout": READOUT_COLUMNS,
        "ranking_reranking_readout": READOUT_COLUMNS,
        "llm_baseline_readout": READOUT_COLUMNS,
        "strategy_model_readout": READOUT_COLUMNS,
        "research_question_closure": RQ_COLUMNS,
        "claim_registry": CLAIM_COLUMNS,
        "remaining_replication_gaps": GAP_COLUMNS,
        "next_phase_requirements": NEXT_PHASE_COLUMNS,
        "missing_or_blocked": MISSING_COLUMNS,
    }
    for name, rows in table_rows.items():
        write_csv(root / config["outputs"]["tables"][name], rows, table_columns[name])

    validation = run_validation_suites(root, config)
    manifest = build_manifest(root, config, repository, interpreter, input_rows, validation)
    counts = {name: len(rows) for name, rows in table_rows.items()}
    report = build_report(repository, interpreter, input_rows, manifest, counts, findings)
    report_path = root / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    manifest["report_hash"] = sha256_file(report_path)
    write_json(root / config["outputs"]["freeze_manifest"], manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Phase F6F2 scientific readout artifacts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to F6F2 YAML config.")
    args = parser.parse_args(argv)
    root = Path.cwd()
    config = load_config(root / args.config)
    manifest = run_phase(root, config)
    print(stable_json({"phase": PHASE_ID, "decision": manifest["decision"], "recommendation": manifest["recommendation"], "scientific_content_hash": manifest["scientific_content_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
