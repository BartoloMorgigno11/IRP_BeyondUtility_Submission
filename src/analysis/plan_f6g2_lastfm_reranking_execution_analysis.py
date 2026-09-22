"""Phase F6G2 contract-gated LastFM re-ranking execution freeze.

This module is offline by default. It reads the frozen F6G1 request
manifests and prior top-10 adapter evidence, reconciles the request contract,
and writes deterministic F6G2 freeze artifacts. Current local evidence blocks
LLM generation before any Ollama call because F6G1 froze a full top-20 output
contract while the approved reranking adapter evidence is top-10.
"""

from __future__ import annotations

from src.audit import common_git

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import site
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PHASE_ID = "phase_f6g2"
DEFAULT_CONFIG_PATH = Path("configs/phase_f6g2_lastfm_reranking_execution.yaml")
PRE_EXISTING_DIRTY = ["README.md", "report/dataset_audit.md", "report/notes.md"]

DECISION_COMPLETE = "f6g2_adapted_lastfm_reranking_complete"
DECISION_COMPLETE_INVALID = "f6g2_adapted_lastfm_reranking_complete_with_invalid_outputs"
DECISION_CONTRACT = "f6g2_blocked_by_request_contract_conflict"
DECISION_INPUTS = "f6g2_blocked_by_input_invariance"
DECISION_MODEL = "f6g2_blocked_by_model_identity"
DECISION_RUNTIME = "f6g2_blocked_by_runtime"
DECISION_RAW = "f6g2_blocked_by_raw_integrity"
DECISION_TESTS = "f6g2_blocked_by_tests"
DECISION_INCOMPLETE = "f6g2_incomplete_missing_requests"

RECOMMEND_CONTRACT = "keep_lastfm_reranking_execution_frozen"
RECOMMEND_COMPLETE = "proceed_to_f6g3_scientific_consolidation"
RECOMMEND_RESUME = "resume_missing_requests"
RECOMMEND_RUNTIME = "repair_runtime_before_resume"


REPOSITORY_COLUMNS = [
    "phase",
    "repository_path",
    "branch",
    "head",
    "expected_branch",
    "expected_head",
    "branch_status",
    "head_status",
    "staged_files",
    "tracked_dirty_files",
    "pre_existing_tracked_dirty_files",
    "generated_paths_filtered",
    "timestamp",
    "python_executable",
    "python_version",
    "environment_classification",
    "status",
    "notes",
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
RUNTIME_GATE_COLUMNS = [
    "gate_id",
    "gate_name",
    "requirement",
    "observed_evidence",
    "status",
    "blocking",
    "execution_allowed",
    "ollama_called",
    "notes",
]
MODEL_IDENTITY_COLUMNS = [
    "model",
    "expected_digest_source",
    "expected_digest",
    "actual_digest",
    "identity_status",
    "classification",
    "ollama_checked",
    "generation_allowed",
    "notes",
]
PROGRESS_COLUMNS = [
    "progress_id",
    "mode",
    "expected_requests",
    "raw_completed_requests",
    "strict_valid_requests",
    "invalid_requests",
    "missing_requests",
    "blocked_requests",
    "current_shard_id",
    "status",
    "requests_per_minute",
    "eta_seconds",
    "last_request_id",
    "timestamp",
    "notes",
]
SHARD_STATUS_COLUMNS = [
    "shard_id",
    "shard_index",
    "model",
    "strategy",
    "expected_request_count",
    "completed_request_count",
    "missing_request_count",
    "blocked_request_count",
    "status",
    "first_request_id",
    "last_request_id",
    "notes",
]
REQUEST_EXECUTION_COLUMNS = [
    "request_id",
    "logical_instance_id",
    "shard_id",
    "shard_index",
    "request_order",
    "dataset",
    "task",
    "user_id",
    "model",
    "strategy",
    "candidate_pool_size",
    "requested_ranking_length",
    "parser_expected_length",
    "evaluation_cutoff_k",
    "candidate_order_hash",
    "positive_item_id",
    "positive_present",
    "positive_initial_position",
    "execution_status",
    "transport_status",
    "strict_valid",
    "invalid_reason",
    "raw_output_file",
    "raw_record_sha256",
    "latency_seconds",
    "prompt_eval_count",
    "eval_count",
    "notes",
]
PARSER_VALIDITY_COLUMNS = [
    "request_id",
    "model",
    "strategy",
    "parse_status",
    "json_parse_valid",
    "schema_valid",
    "strict_valid",
    "parsed_count",
    "duplicate_count",
    "unknown_index_count",
    "extra_text_detected",
    "parsed_indices",
    "mapped_item_ids",
    "positive_final_rank",
    "hr_at_1",
    "hr_at_5",
    "hr_at_10",
    "ndcg_at_5",
    "ndcg_at_10",
    "population_role",
    "notes",
]
INVALID_AUDIT_COLUMNS = [
    "audit_id",
    "model",
    "strategy",
    "invalid_reason",
    "count",
    "population",
    "retry_allowed",
    "repair_allowed",
    "completed_semantically_invalid",
    "notes",
]
CELL_SUMMARY_COLUMNS = [
    "summary_id",
    "dataset",
    "task",
    "model",
    "strategy",
    "population",
    "expected_requests",
    "completed_requests",
    "transport_success",
    "strict_valid",
    "invalid_outputs",
    "positive_present_users",
    "hr_at_1",
    "hr_at_5",
    "hr_at_10",
    "ndcg_at_5",
    "ndcg_at_10",
    "status",
    "notes",
]
POPULATION_SUMMARY_COLUMNS = [
    "summary_id",
    "dataset",
    "task",
    "population",
    "expected_requests",
    "completed_requests",
    "transport_success",
    "strict_valid",
    "invalid_outputs",
    "positive_present_users",
    "hr_at_1",
    "hr_at_5",
    "hr_at_10",
    "ndcg_at_5",
    "ndcg_at_10",
    "status",
    "notes",
]
DATASET_TASK_SUMMARY_COLUMNS = [
    "summary_id",
    "dataset",
    "task",
    "logical_instances",
    "requests",
    "shards",
    "candidate_pool_size",
    "requested_ranking_length",
    "parser_expected_length",
    "completed_requests",
    "strict_valid_requests",
    "blocked_requests",
    "status",
    "notes",
]
EXECUTION_GATE_COLUMNS = [
    "gate_id",
    "gate_name",
    "requirement",
    "observed_evidence",
    "status",
    "impact",
    "decision_if_failed",
    "recommendation_if_failed",
    "notes",
]
MISSING_COLUMNS = [
    "block_id",
    "component",
    "status",
    "blocking_reason",
    "current_evidence",
    "required_evidence_or_action",
    "decision",
    "recommendation",
    "notes",
]

TABLE_COLUMNS = {
    "repository_state": REPOSITORY_COLUMNS,
    "input_integrity": INPUT_INTEGRITY_COLUMNS,
    "runtime_gate": RUNTIME_GATE_COLUMNS,
    "model_identity": MODEL_IDENTITY_COLUMNS,
    "progress": PROGRESS_COLUMNS,
    "shard_status": SHARD_STATUS_COLUMNS,
    "request_execution": REQUEST_EXECUTION_COLUMNS,
    "parser_validity": PARSER_VALIDITY_COLUMNS,
    "invalid_output_audit": INVALID_AUDIT_COLUMNS,
    "cell_execution_summary": CELL_SUMMARY_COLUMNS,
    "population_execution_summary": POPULATION_SUMMARY_COLUMNS,
    "dataset_task_execution_summary": DATASET_TASK_SUMMARY_COLUMNS,
    "execution_gate": EXECUTION_GATE_COLUMNS,
    "missing_or_blocked": MISSING_COLUMNS,
}


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


def bool_text(value: Any) -> str:
    if isinstance(value, str):
        return "true" if value.strip().lower() == "true" else "false"
    return "true" if bool(value) else "false"


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


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def split_pipe(value: Any) -> list[str]:
    return [part for part in str(value or "").split("|") if part != ""]


def pipe_join(values: Iterable[Any]) -> str:
    return "|".join(str(value) for value in values)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: "" if value is None else str(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: "" if row.get(column, "") is None else str(row.get(column, "")) for column in columns})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(payload) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError:
        local_site = Path.cwd() / ".venv" / "Lib" / "site-packages"
        if local_site.exists():
            site.addsitedir(str(local_site))
        try:
            import yaml
        except ModuleNotFoundError:
            return json.loads(path.read_text(encoding="utf-8"))
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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
    paths.extend(config.get("outputs", {}).get("raw", {}).values())
    paths.extend(
        [
            "configs/phase_f6g2_lastfm_reranking_execution.yaml",
            "src/analysis/plan_f6g2_lastfm_reranking_execution_analysis.py",
            "src/llm/plan_f6g2_lastfm_reranking_runner.py",
            "src/experiments/run_phase_f6g2_lastfm_reranking.py",
            "tests/test_plan_f6g2_lastfm_reranking_execution.py",
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
    timestamp = previous.get("runtime_metadata", {}).get("timestamp") or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    branch = git(root, "branch", "--show-current")
    head = git(root, "rev-parse", "HEAD")
    expected_branch = str(config["phase"].get("expected_branch", ""))
    expected_head = str(config["phase"].get("expected_head", ""))
    filtered = filter_generated_status(git(root, "status", "--short"), config)
    branch_ok = not expected_branch or branch == expected_branch
    head_ok = not expected_head or head == expected_head
    return {
        "phase": PHASE_ID,
        "repository_path": str(root),
        "branch": branch,
        "head": head,
        "expected_branch": expected_branch,
        "expected_head": expected_head,
        "branch_status": "pass" if branch_ok else "fail",
        "head_status": "pass" if head_ok else "fail",
        "staged_files": git(root, "diff", "--cached", "--name-only"),
        "tracked_dirty_files": git(root, "status", "--short", "--untracked-files=no"),
        "pre_existing_tracked_dirty_files": ";".join(PRE_EXISTING_DIRTY),
        "generated_paths_filtered": bool_text(True),
        "timestamp": timestamp,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "environment_classification": config["phase"]["environment_classification"],
        "status": "pass" if branch_ok and head_ok else "fail",
        "notes": "Repository state records existing dirty tracked files but filters F6G2 generated paths from untracked noise.",
    }


def verify_interpreter(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    expected = (root / config["phase"]["interpreter_path"]).resolve(strict=False)
    actual = Path(sys.executable).resolve(strict=False)
    imports_ok = True
    import_error = ""
    try:
        import pytest  # noqa: F401
        import yaml as _yaml  # noqa: F401
    except Exception as exc:  # pragma: no cover
        imports_ok = False
        import_error = f"{type(exc).__name__}: {exc}"
    status = "pass" if actual.exists() and imports_ok and str(actual).lower() == str(expected).lower() else "fail"
    return {
        "expected_interpreter": str(expected),
        "actual_interpreter": str(actual),
        "status": status,
        "imports_ok": bool_text(imports_ok),
        "import_error": import_error,
        "python_version": platform.python_version(),
        "sys_prefix": sys.prefix,
        "sys_base_prefix": sys.base_prefix,
    }


def verify_input_integrity(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact, spec in config["input_hashes"].items():
        path = root / spec["path"]
        exists = path.exists()
        actual = sha256_file(path) if exists and path.is_file() else ""
        expected = str(spec.get("sha256", ""))
        changed = bool(expected and actual and expected.lower() != actual.lower())
        rows.append(
            {
                "artifact": artifact,
                "path": spec["path"],
                "phase": spec.get("phase", ""),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": "pass" if exists and not changed else "fail",
                "changed": bool_text(changed),
                "missing": bool_text(not exists),
                "notes": "",
            }
        )
    return rows


def assert_no_input_failures(rows: Sequence[Mapping[str, Any]]) -> None:
    failures = [row for row in rows if row["status"] != "pass"]
    if failures:
        detail = "; ".join(f"{row['artifact']}:{row['status']}" for row in failures[:5])
        raise RuntimeError(f"F6G2 input integrity gate failed: {detail}")


def validate_output_paths(config: Mapping[str, Any]) -> None:
    allowed_prefixes = ("results/tables/phase_f6g2_", "results/manifests/phase_f6g2_", "report/phase_f6g2_")
    paths = [config["outputs"]["report"], config["outputs"]["freeze_manifest"], *config["outputs"]["tables"].values()]
    for path in paths:
        normalized = str(path).replace("\\", "/")
        if not normalized.startswith(allowed_prefixes):
            raise ValueError(f"F6G2 output path is outside phase namespace: {path}")
    for path in config.get("outputs", {}).get("raw", {}).values():
        normalized = str(path).replace("\\", "/")
        if not normalized.startswith("results/raw/phase_f6g2_"):
            raise ValueError(f"F6G2 raw path is outside phase namespace: {path}")


def load_frames(root: Path, config: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    def path(name: str) -> Path:
        return root / config["input_hashes"][name]["path"]

    return {
        "logical": read_csv(path("f6g1_future_logical_instances")),
        "requests": read_csv(path("f6g1_future_request_manifest")),
        "shards": read_csv(path("f6g1_future_shard_manifest")),
        "pool_contract": read_csv(path("f6g1_paper_pool_contract")),
        "f6e3b_requests": read_csv(path("f6e3b_request_manifest")),
        "f6e4a_requests": read_csv(path("f6e4a_request_manifest")),
        "protocol_decision": read_csv(path("reranking_protocol_decision")),
        "top10_prompt": read_csv(path("top10_prompt_contract")),
        "top10_parser": read_csv(path("top10_parser_contract")),
        "full20_prompt": read_csv(path("full20_prompt_v2_contract")),
    }


def verify_phase_manifests(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    f6g1 = load_json(root / config["input_hashes"]["f6g1_manifest"]["path"])
    f6f2 = load_json(root / config["input_hashes"]["f6f2_manifest"]["path"])
    f6f1 = load_json(root / config["input_hashes"]["f6f1_manifest"]["path"])
    f6e4c = load_json(root / config["input_hashes"]["f6e4c_manifest"]["path"])
    checks = {
        "f6g1_decision": f6g1.get("decision", "") == config["phase"]["expected_f6g1_decision"],
        "f6g1_scientific_hash": f6g1.get("scientific_content_hash", "") == config["phase"]["expected_f6g1_scientific_hash"],
        "f6f2_decision": f6f2.get("decision", "") == config["phase"]["expected_f6f2_decision"],
        "f6f2_scientific_hash": f6f2.get("scientific_content_hash", "") == config["phase"]["expected_f6f2_scientific_hash"],
        "f6f1_decision": f6f1.get("decision", "") == config["phase"]["expected_f6f1_decision"],
        "f6f1_scientific_hash": f6f1.get("scientific_content_hash", "") == config["phase"]["expected_f6f1_scientific_hash"],
        "f6e4c_decision": f6e4c.get("decision", "") == config["phase"]["expected_f6e4c_decision"],
        "f6e4c_scientific_hash": f6e4c.get("scientific_content_hash", "") == config["phase"]["expected_f6e4c_scientific_hash"],
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "f6g1_scientific_hash": f6g1.get("scientific_content_hash", ""),
        "f6f2_scientific_hash": f6f2.get("scientific_content_hash", ""),
        "f6f1_scientific_hash": f6f1.get("scientific_content_hash", ""),
        "f6e4c_scientific_hash": f6e4c.get("scientific_content_hash", ""),
        "f6g1_decision": f6g1.get("decision", ""),
        "f6f2_decision": f6f2.get("decision", ""),
        "f6f1_decision": f6f1.get("decision", ""),
        "f6e4c_decision": f6e4c.get("decision", ""),
        "f6g1_future_request_manifest_hash": f6g1.get("future_request_manifest_hash", ""),
        "f6g1_future_logical_instance_hash": f6g1.get("future_logical_instance_hash", ""),
        "f6g1_future_shard_manifest_hash": f6g1.get("future_shard_manifest_hash", ""),
    }


def unique_sorted(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values if str(value) != ""})


def expected_schema_lengths(schema_texts: Iterable[Any]) -> list[str]:
    lengths: set[str] = set()
    for text in schema_texts:
        try:
            schema = json.loads(str(text))
        except json.JSONDecodeError:
            continue
        ranking = schema.get("properties", {}).get("ranking", {}) if isinstance(schema, dict) else {}
        min_items = ranking.get("minItems")
        max_items = ranking.get("maxItems")
        if min_items == max_items and min_items is not None:
            lengths.add(str(min_items))
    return sorted(lengths, key=lambda item: parse_int(item))


def column_values(rows: Sequence[Mapping[str, Any]], column: str) -> list[str]:
    return [str(row.get(column, "")) for row in rows]


def contract_reconciliation(frames: Mapping[str, Sequence[Mapping[str, Any]]], config: Mapping[str, Any]) -> dict[str, Any]:
    requests = frames["requests"]
    logical = frames["logical"]
    candidate_pool_sizes = unique_sorted(len(split_pipe(row.get("candidate_item_ids", ""))) for row in requests)
    logical_output_lengths = unique_sorted(column_values(logical, "expected_output_length"))
    request_output_lengths = unique_sorted(column_values(requests, "required_output_length"))
    schema_lengths = expected_schema_lengths(column_values(requests, "expected_json_schema"))
    parser_contracts = unique_sorted(column_values(requests, "parser_contract"))

    top10_parser_counts = unique_sorted(column_values(frames["top10_parser"], "expected_output_count"))
    top10_prompt_counts = unique_sorted(column_values(frames["top10_prompt"], "requested_output_count"))
    e3b_contracts = unique_sorted(column_values(frames["f6e3b_requests"], "output_contract_id"))
    e4a_schemas = unique_sorted(column_values(frames["f6e4a_requests"], "expected_schema"))
    protocol = dict(frames["protocol_decision"][0]) if frames["protocol_decision"] else {}
    required_length = str(config["execution_contract"]["required_requested_ranking_length"])
    required_parser_length = str(config["execution_contract"]["required_parser_expected_length"])
    required_pool_size = str(config["execution_contract"]["candidate_pool_size"])
    blocked_contracts = {str(value) for value in config["execution_contract"].get("blocked_f6g1_parser_contracts", [])}

    candidate_pool_ok = candidate_pool_sizes == [required_pool_size]
    f6g1_length_ok = request_output_lengths == [required_length] and logical_output_lengths == [required_length]
    f6g1_parser_ok = schema_lengths == [required_parser_length]
    f6g1_contract_ok = not any(contract_id in blocked_contracts for contract_id in parser_contracts)
    top10_evidence_ok = top10_parser_counts == [required_parser_length] and top10_prompt_counts == [required_length]
    blocked = not (candidate_pool_ok and f6g1_length_ok and f6g1_parser_ok and f6g1_contract_ok and top10_evidence_ok)
    if blocked:
        reason = (
            "F6G1 future manifests explicitly require candidate pool size "
            f"{';'.join(candidate_pool_sizes)}, output length "
            f"{';'.join(request_output_lengths or logical_output_lengths or schema_lengths)} with parser "
            f"{';'.join(parser_contracts)}, while approved reranking adapter evidence requires top-{required_length}."
        )
    else:
        reason = "F6G1 request contract matches approved top-10 reranking adapter evidence."
    return {
        "candidate_pool_sizes": candidate_pool_sizes,
        "logical_output_lengths": logical_output_lengths,
        "request_output_lengths": request_output_lengths,
        "parser_schema_lengths": schema_lengths,
        "parser_contracts": parser_contracts,
        "top10_parser_expected_counts": top10_parser_counts,
        "top10_prompt_requested_counts": top10_prompt_counts,
        "f6e3b_output_contracts": e3b_contracts,
        "f6e4a_expected_schemas": e4a_schemas[:3],
        "protocol_selected_option": protocol.get("selected_protocol_option_id", ""),
        "protocol_recommended_decision": protocol.get("recommended_decision", ""),
        "required_requested_ranking_length": required_length,
        "required_parser_expected_length": required_parser_length,
        "required_candidate_pool_size": required_pool_size,
        "status": "blocked" if blocked else "pass",
        "blocking_reason": reason,
        "execution_allowed": not blocked,
    }


def raw_path_for_model(config: Mapping[str, Any], model: str) -> str:
    key = "qwen2" if str(model).startswith("qwen2") else "mistral" if str(model).startswith("mistral") else str(model)
    return str(config.get("outputs", {}).get("raw", {}).get(key, ""))


def raw_file_stats(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for key, rel in sorted(config.get("outputs", {}).get("raw", {}).items()):
        path = root / rel
        line_count = 0
        if path.exists():
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                line_count = sum(1 for line in handle if line.strip())
        stats[key] = {
            "path": rel,
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
            "line_count": line_count,
        }
    return stats


def load_raw_records(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str], list[dict[str, Any]]]:
    by_request: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    errors: list[dict[str, Any]] = []
    for _key, rel in sorted(config.get("outputs", {}).get("raw", {}).items()):
        path = root / rel
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError as exc:
                    errors.append({"path": rel, "line_number": line_number, "error": f"json_decode_error:{exc.msg}"})
                    continue
                request_id = str(record.get("request_id", ""))
                if not request_id:
                    errors.append({"path": rel, "line_number": line_number, "error": "missing_request_id"})
                    continue
                if request_id in by_request:
                    duplicates.append(request_id)
                by_request[request_id] = {**record, "_raw_file": rel, "_line_number": line_number}
    return by_request, sorted(set(duplicates)), errors


def plan_resume(expected_request_ids: Sequence[str], raw_records: Mapping[str, Mapping[str, Any]]) -> list[str]:
    seen = set(raw_records)
    return [request_id for request_id in expected_request_ids if request_id not in seen]


def parse_status(
    json_parse_valid: bool,
    schema_valid: bool,
    strict_valid: bool,
    indices: Sequence[int],
    status: str,
    duplicate_count: int,
    unknown_count: int,
    extra_text: bool,
) -> dict[str, Any]:
    return {
        "json_parse_valid": json_parse_valid,
        "schema_valid": schema_valid,
        "strict_valid": strict_valid,
        "output_indices": tuple(indices),
        "output_length": len(indices),
        "duplicate_count": duplicate_count,
        "unknown_index_count": unknown_count,
        "parse_status": status,
        "extra_text_detected": extra_text,
    }


def parse_strict_indexed_response(
    response_text: str,
    *,
    candidate_count: int = 20,
    expected_count: int = 10,
    ranking_key: str = "ranking",
    require_complete_permutation: bool = False,
) -> dict[str, Any]:
    text = str(response_text or "").strip()
    if not text:
        return parse_status(False, False, False, (), "empty_response", 0, 0, False)
    extra_text = not (text.startswith("{") and text.endswith("}")) or "```" in text
    if extra_text:
        return parse_status(False, False, False, (), "extra_text_or_code_fence", 0, 0, True)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return parse_status(False, False, False, (), "json_parse_failed", 0, 0, extra_text)
    if not isinstance(parsed, dict):
        return parse_status(True, False, False, (), "invalid_schema", 0, 0, extra_text)
    keys = set(parsed)
    if keys != {ranking_key}:
        status = f"missing_{ranking_key}_key" if ranking_key not in keys else "extra_keys"
        return parse_status(True, False, False, (), status, 0, 0, extra_text)
    values = parsed.get(ranking_key)
    if not isinstance(values, list) or not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return parse_status(True, False, False, (), "invalid_schema", 0, 0, extra_text)
    indices = tuple(int(value) for value in values)
    duplicate_count = len(indices) - len(set(indices))
    unknown_count = sum(1 for value in indices if value < 1 or value > candidate_count)
    if len(indices) != expected_count:
        return parse_status(True, True, False, indices, "invalid_length", duplicate_count, unknown_count, extra_text)
    if duplicate_count:
        return parse_status(True, True, False, indices, "duplicate_indices", duplicate_count, unknown_count, extra_text)
    if unknown_count:
        return parse_status(True, True, False, indices, "out_of_range_indices", duplicate_count, unknown_count, extra_text)
    if require_complete_permutation and set(indices) != set(range(1, candidate_count + 1)):
        return parse_status(True, True, False, indices, "not_complete_permutation", duplicate_count, unknown_count, extra_text)
    return parse_status(True, True, True, indices, "parsed_valid", 0, 0, extra_text)


def map_indices(indices: Sequence[int], candidate_ids: Sequence[str]) -> list[str]:
    return [str(candidate_ids[int(index) - 1]) for index in indices if 1 <= int(index) <= len(candidate_ids)]


def metric_values(mapped_ids: Sequence[str], positive_item_id: str, positive_present: bool, strict_valid: bool) -> dict[str, Any]:
    rank = ""
    values = {"hr_at_1": 0.0, "hr_at_5": 0.0, "hr_at_10": 0.0, "ndcg_at_5": 0.0, "ndcg_at_10": 0.0}
    if strict_valid and positive_present and positive_item_id in mapped_ids:
        rank_int = list(mapped_ids).index(positive_item_id) + 1
        rank = str(rank_int)
        for k in (1, 5, 10):
            if rank_int <= k:
                values[f"hr_at_{k}"] = 1.0
        if rank_int <= 5:
            values["ndcg_at_5"] = 1.0 / math.log2(rank_int + 1)
        if rank_int <= 10:
            values["ndcg_at_10"] = 1.0 / math.log2(rank_int + 1)
    return {"positive_final_rank": rank, **values}


def request_shard_lookup(requests: Sequence[Mapping[str, Any]], shards: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    offset = 0
    ordered_shards = sorted(shards, key=lambda row: parse_int(row.get("shard_index", 0)))
    request_list = list(requests)
    for shard in ordered_shards:
        count = parse_int(shard.get("request_count", 0))
        for request in request_list[offset : offset + count]:
            request_id = str(request.get("request_id", ""))
            lookup[request_id] = {
                "shard_id": str(shard.get("shard_id", "")),
                "shard_index": str(shard.get("shard_index", "")),
            }
        offset += count
    return lookup


def build_repository_state_table(repository: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(repository)]


def build_runtime_gate(
    repository: Mapping[str, Any],
    interpreter: Mapping[str, str],
    input_rows: Sequence[Mapping[str, Any]],
    phase_checks: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    input_ok = all(row["status"] == "pass" for row in input_rows)
    repo_ok = repository["status"] == "pass"
    gates = [
        ("repository_state", "repository gate", "branch and HEAD match config", f"branch={repository['branch']};head={repository['head']}", "pass" if repo_ok else "fail", repo_ok),
        ("interpreter", "interpreter gate", "run with configured reproducibility interpreter", f"actual={interpreter['actual_interpreter']};status={interpreter['status']}", interpreter["status"], interpreter["status"] == "pass"),
        ("input_integrity", "input integrity gate", "all direct frozen inputs hash-match", f"changed={sum(1 for row in input_rows if row['changed'] == 'true')};missing={sum(1 for row in input_rows if row['missing'] == 'true')}", "pass" if input_ok else "fail", input_ok),
        ("phase_manifest", "F6G1/F6F2/F6F1/F6E4C gate", "prior decisions and scientific hashes match config", f"status={phase_checks['status']}", phase_checks["status"], phase_checks["status"] == "pass"),
        ("request_contract", "request contract reconciliation gate", "F6G1 requested length and parser length must match approved top-10 contract", contract["blocking_reason"], contract["status"], bool(contract["execution_allowed"])),
        ("local_ollama_generation", "local Ollama generation gate", "generation allowed only after contract gate passes", "not called; blocked before inference", "skipped" if not contract["execution_allowed"] else "pending", bool(contract["execution_allowed"])),
    ]
    rows: list[dict[str, Any]] = []
    for key, name, requirement, evidence, status, allowed in gates:
        blocking = status in {"fail", "blocked"}
        rows.append(
            {
                "gate_id": stable_id("F6G2-RUNTIME", key),
                "gate_name": name,
                "requirement": requirement,
                "observed_evidence": evidence,
                "status": status,
                "blocking": bool_text(blocking),
                "execution_allowed": bool_text(allowed and contract["execution_allowed"]),
                "ollama_called": bool_text(False),
                "notes": "No model generation is attempted by the blocked freeze.",
            }
        )
    return rows


def build_model_identity(config: Mapping[str, Any], contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in config["runtime"]["models"]:
        rows.append(
            {
                "model": model,
                "expected_digest_source": "not_checked_until_request_contract_gate_passes",
                "expected_digest": "",
                "actual_digest": "",
                "identity_status": "not_checked_contract_gate_blocked" if not contract["execution_allowed"] else "pending_runtime_check",
                "classification": config["runtime"]["model_equivalence_classification"],
                "ollama_checked": bool_text(False),
                "generation_allowed": bool_text(False),
                "notes": "Digest is intentionally not guessed and Ollama is not queried while the request contract is blocked.",
            }
        )
    return rows


def build_request_execution(
    frames: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    raw_records: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    requests = list(frames["requests"])
    lookup = request_shard_lookup(requests, frames["shards"])
    rows: list[dict[str, Any]] = []
    cutoffs = pipe_join(config["execution_contract"]["evaluation_cutoffs"])
    for order, row in enumerate(requests, start=1):
        request_id = str(row["request_id"])
        shard = lookup.get(request_id, {"shard_id": "", "shard_index": ""})
        candidate_pool_size = len(split_pipe(row.get("candidate_item_ids", "")))
        requested_length = str(row.get("required_output_length", ""))
        parser_lengths = expected_schema_lengths([row.get("expected_json_schema", "")])
        parser_length = parser_lengths[0] if parser_lengths else requested_length
        raw = raw_records.get(request_id)
        if not contract["execution_allowed"]:
            execution_status = "blocked_pre_inference_contract_conflict"
            transport_status = "not_started"
            strict_valid = "false"
            invalid_reason = "request_contract_conflict"
        elif raw is None:
            execution_status = "missing_raw_response"
            transport_status = "missing"
            strict_valid = "false"
            invalid_reason = "missing_raw_response"
        else:
            transport_status = str(raw.get("transport_status", "success"))
            execution_status = "completed" if transport_status == "success" else "transport_failed_resumable"
            strict_valid = ""
            invalid_reason = ""
        rows.append(
            {
                "request_id": request_id,
                "logical_instance_id": row.get("logical_instance_id", ""),
                "shard_id": shard["shard_id"],
                "shard_index": shard["shard_index"],
                "request_order": str(order),
                "dataset": row.get("dataset", ""),
                "task": row.get("task", ""),
                "user_id": row.get("user_id", ""),
                "model": row.get("model", ""),
                "strategy": row.get("strategy", ""),
                "candidate_pool_size": str(candidate_pool_size),
                "requested_ranking_length": requested_length,
                "parser_expected_length": parser_length,
                "evaluation_cutoff_k": cutoffs,
                "candidate_order_hash": row.get("candidate_order_hash", ""),
                "positive_item_id": row.get("positive_item_id", ""),
                "positive_present": row.get("positive_present", ""),
                "positive_initial_position": row.get("positive_initial_position", ""),
                "execution_status": execution_status,
                "transport_status": transport_status,
                "strict_valid": strict_valid,
                "invalid_reason": invalid_reason,
                "raw_output_file": raw_path_for_model(config, str(row.get("model", ""))),
                "raw_record_sha256": sha256_text(stable_json(raw)) if raw else "",
                "latency_seconds": str(raw.get("duration_seconds", "")) if raw else "",
                "prompt_eval_count": str(raw.get("prompt_eval_count", "")) if raw else "",
                "eval_count": str(raw.get("eval_count", "")) if raw else "",
                "notes": "F6G2 blocked before inference; no raw response exists." if not contract["execution_allowed"] else "",
            }
        )
    return rows


def build_parser_validity(
    frames: Mapping[str, Sequence[Mapping[str, Any]]],
    contract: Mapping[str, Any],
    raw_records: Mapping[str, Mapping[str, Any]],
    *,
    ranking_key: str = "ranking",
) -> list[dict[str, Any]]:
    if not raw_records:
        return []
    rows: list[dict[str, Any]] = []
    request_by_id = {str(row["request_id"]): row for row in frames["requests"]}
    expected_count = parse_int(contract["required_parser_expected_length"])
    for request_id, raw in sorted(raw_records.items()):
        request = request_by_id.get(request_id)
        if request is None:
            continue
        candidate_ids = split_pipe(request.get("candidate_item_ids", ""))
        parse = parse_strict_indexed_response(
            str(raw.get("raw_response", "")),
            candidate_count=len(candidate_ids),
            expected_count=expected_count,
            ranking_key=ranking_key,
        )
        mapped = map_indices(parse["output_indices"], candidate_ids)
        positive_present = str(request.get("positive_present", "")).lower() == "true"
        metrics = metric_values(mapped, str(request.get("positive_item_id", "")), positive_present, bool(parse["strict_valid"]))
        rows.append(
            {
                "request_id": request_id,
                "model": request.get("model", ""),
                "strategy": request.get("strategy", ""),
                "parse_status": parse["parse_status"],
                "json_parse_valid": bool_text(parse["json_parse_valid"]),
                "schema_valid": bool_text(parse["schema_valid"]),
                "strict_valid": bool_text(parse["strict_valid"]),
                "parsed_count": str(parse["output_length"]),
                "duplicate_count": str(parse["duplicate_count"]),
                "unknown_index_count": str(parse["unknown_index_count"]),
                "extra_text_detected": bool_text(parse["extra_text_detected"]),
                "parsed_indices": pipe_join(parse["output_indices"]),
                "mapped_item_ids": pipe_join(mapped),
                "positive_final_rank": metrics["positive_final_rank"],
                "hr_at_1": fmt(metrics["hr_at_1"]),
                "hr_at_5": fmt(metrics["hr_at_5"]),
                "hr_at_10": fmt(metrics["hr_at_10"]),
                "ndcg_at_5": fmt(metrics["ndcg_at_5"]),
                "ndcg_at_10": fmt(metrics["ndcg_at_10"]),
                "population_role": "positive_present" if positive_present else "positive_absent",
                "notes": "Strict parser only; no repair, truncation or salvage.",
            }
        )
    return rows


def build_invalid_output_audit(parser_rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    invalid = [row for row in parser_rows if row["strict_valid"] != "true"]
    if not invalid:
        return [
            {
                "audit_id": stable_id("F6G2-INVALID", "no_execution" if not contract["execution_allowed"] else "none"),
                "model": "",
                "strategy": "",
                "invalid_reason": "not_applicable_no_execution" if not contract["execution_allowed"] else "none",
                "count": "0",
                "population": "all",
                "retry_allowed": bool_text(False),
                "repair_allowed": bool_text(False),
                "completed_semantically_invalid": bool_text(False),
                "notes": "No semantic output was produced because F6G2 stopped before inference.",
            }
        ]
    counts = Counter((row["model"], row["strategy"], row["parse_status"]) for row in invalid)
    return [
        {
            "audit_id": stable_id("F6G2-INVALID", model, strategy, reason),
            "model": model,
            "strategy": strategy,
            "invalid_reason": reason,
            "count": str(count),
            "population": "strict_parser_outputs",
            "retry_allowed": bool_text(False),
            "repair_allowed": bool_text(False),
            "completed_semantically_invalid": bool_text(True),
            "notes": "Semantic invalidity is preserved and is not retried or repaired.",
        }
        for (model, strategy, reason), count in sorted(counts.items())
    ]


def build_progress(
    mode: str,
    repository: Mapping[str, Any],
    request_rows: Sequence[Mapping[str, Any]],
    parser_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    raw_records: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected = len(request_rows)
    completed = sum(1 for row in request_rows if row["execution_status"] == "completed")
    strict = sum(1 for row in parser_rows if row["strict_valid"] == "true")
    invalid = sum(1 for row in parser_rows if row["strict_valid"] == "false")
    blocked = expected if not contract["execution_allowed"] else 0
    missing = len(plan_resume([str(row["request_id"]) for row in request_rows], raw_records))
    return [
        {
            "progress_id": stable_id("F6G2-PROGRESS", mode),
            "mode": mode,
            "expected_requests": str(expected),
            "raw_completed_requests": str(completed),
            "strict_valid_requests": str(strict),
            "invalid_requests": str(invalid),
            "missing_requests": str(missing),
            "blocked_requests": str(blocked),
            "current_shard_id": "",
            "status": "blocked_request_contract_conflict" if not contract["execution_allowed"] else "ready_or_incomplete",
            "requests_per_minute": "",
            "eta_seconds": "",
            "last_request_id": "",
            "timestamp": repository["timestamp"],
            "notes": "Progress is atomic and deterministic for the blocked freeze; no raw append occurred.",
        }
    ]


def build_shard_status(
    frames: Mapping[str, Sequence[Mapping[str, Any]]],
    request_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_shard = Counter(row["shard_id"] for row in request_rows if row.get("execution_status") == "completed")
    rows: list[dict[str, Any]] = []
    for shard in sorted(frames["shards"], key=lambda row: parse_int(row.get("shard_index", 0))):
        shard_id = str(shard["shard_id"])
        expected = parse_int(shard.get("request_count", 0))
        completed = by_shard.get(shard_id, 0)
        blocked = expected if not contract["execution_allowed"] else 0
        missing = expected - completed
        rows.append(
            {
                "shard_id": shard_id,
                "shard_index": shard.get("shard_index", ""),
                "model": shard.get("model", ""),
                "strategy": shard.get("strategy", ""),
                "expected_request_count": str(expected),
                "completed_request_count": str(completed),
                "missing_request_count": str(missing),
                "blocked_request_count": str(blocked),
                "status": "blocked_integrity" if not contract["execution_allowed"] else ("completed" if completed == expected else "pending"),
                "first_request_id": shard.get("first_request_id", ""),
                "last_request_id": shard.get("last_request_id", ""),
                "notes": "Blocked before inference by request-contract conflict." if not contract["execution_allowed"] else "",
            }
        )
    return rows


def _mean_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> str:
    values = [float(row[metric]) for row in rows if str(row.get(metric, "")) != ""]
    return fmt(sum(values) / len(values)) if values else ""


def _summary_status(contract: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    if not contract["execution_allowed"]:
        return "blocked_request_contract_conflict"
    return "complete" if rows and all(row.get("execution_status") == "completed" for row in rows) else "incomplete"


def build_cell_execution_summary(
    request_rows: Sequence[Mapping[str, Any]],
    parser_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    parser_by_request = {row["request_id"]: row for row in parser_rows}
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in request_rows:
        groups.setdefault((row["dataset"], row["task"], row["model"], row["strategy"]), []).append(row)
    populations = [
        ("all_user_operational", lambda request: True, False),
        ("all_user_strict_valid_conditional", lambda request: True, True),
        ("positive_present_operational_diagnostic", lambda request: request["positive_present"] == "true", False),
        ("positive_present_strict_valid_conditional_diagnostic", lambda request: request["positive_present"] == "true", True),
    ]
    for (dataset, task, model, strategy), group in sorted(groups.items()):
        for population, predicate, strict_only in populations:
            eligible = [row for row in group if predicate(row)]
            parsed = [parser_by_request[row["request_id"]] for row in eligible if row["request_id"] in parser_by_request]
            if strict_only:
                parsed = [row for row in parsed if row["strict_valid"] == "true"]
            rows.append(
                {
                    "summary_id": stable_id("F6G2-CELL", dataset, task, model, strategy, population),
                    "dataset": dataset,
                    "task": task,
                    "model": model,
                    "strategy": strategy,
                    "population": population,
                    "expected_requests": str(len(eligible)),
                    "completed_requests": str(sum(1 for row in eligible if row["execution_status"] == "completed")),
                    "transport_success": str(sum(1 for row in eligible if row["transport_status"] == "success")),
                    "strict_valid": str(sum(1 for row in parsed if row["strict_valid"] == "true")),
                    "invalid_outputs": str(sum(1 for row in parsed if row["strict_valid"] == "false")),
                    "positive_present_users": str(sum(1 for row in eligible if row["positive_present"] == "true")),
                    "hr_at_1": _mean_metric(parsed, "hr_at_1"),
                    "hr_at_5": _mean_metric(parsed, "hr_at_5"),
                    "hr_at_10": _mean_metric(parsed, "hr_at_10"),
                    "ndcg_at_5": _mean_metric(parsed, "ndcg_at_5"),
                    "ndcg_at_10": _mean_metric(parsed, "ndcg_at_10"),
                    "status": _summary_status(contract, eligible),
                    "notes": "Population is not merged with any other denominator.",
                }
            )
    return rows


def build_population_execution_summary(
    cell_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in cell_rows:
        groups.setdefault((row["dataset"], row["task"], row["population"]), []).append(row)
    rows: list[dict[str, Any]] = []
    for (dataset, task, population), group in sorted(groups.items()):
        rows.append(
            {
                "summary_id": stable_id("F6G2-POP", dataset, task, population),
                "dataset": dataset,
                "task": task,
                "population": population,
                "expected_requests": str(sum(parse_int(row["expected_requests"]) for row in group)),
                "completed_requests": str(sum(parse_int(row["completed_requests"]) for row in group)),
                "transport_success": str(sum(parse_int(row["transport_success"]) for row in group)),
                "strict_valid": str(sum(parse_int(row["strict_valid"]) for row in group)),
                "invalid_outputs": str(sum(parse_int(row["invalid_outputs"]) for row in group)),
                "positive_present_users": str(sum(parse_int(row["positive_present_users"]) for row in group)),
                "hr_at_1": "",
                "hr_at_5": "",
                "hr_at_10": "",
                "ndcg_at_5": "",
                "ndcg_at_10": "",
                "status": "blocked_request_contract_conflict" if not contract["execution_allowed"] else "summarized",
                "notes": "Aggregated across model/strategy cells; metric means are intentionally omitted here to avoid denominator mixing.",
            }
        )
    return rows


def build_dataset_task_execution_summary(
    frames: Mapping[str, Sequence[Mapping[str, Any]]],
    request_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    dataset = str(frames["requests"][0].get("dataset", "")) if frames["requests"] else ""
    task = str(frames["requests"][0].get("task", "")) if frames["requests"] else ""
    return [
        {
            "summary_id": stable_id("F6G2-DATASET", dataset, task),
            "dataset": dataset,
            "task": task,
            "logical_instances": str(len({str(row.get("logical_instance_id", "")) for row in frames["logical"]})),
            "requests": str(len(frames["requests"])),
            "shards": str(len(frames["shards"])),
            "candidate_pool_size": pipe_join(contract["candidate_pool_sizes"]),
            "requested_ranking_length": pipe_join(contract["request_output_lengths"]),
            "parser_expected_length": pipe_join(contract["parser_schema_lengths"]),
            "completed_requests": str(sum(1 for row in request_rows if row["execution_status"] == "completed")),
            "strict_valid_requests": str(sum(1 for row in request_rows if row["strict_valid"] == "true")),
            "blocked_requests": str(sum(1 for row in request_rows if row["execution_status"].startswith("blocked_"))),
            "status": "blocked_request_contract_conflict" if not contract["execution_allowed"] else "execution_ready",
            "notes": contract["blocking_reason"],
        }
    ]


def build_execution_gate(
    config: Mapping[str, Any],
    repository: Mapping[str, Any],
    interpreter: Mapping[str, str],
    input_rows: Sequence[Mapping[str, Any]],
    phase_checks: Mapping[str, Any],
    frames: Mapping[str, Sequence[Mapping[str, Any]]],
    contract: Mapping[str, Any],
    raw_duplicates: Sequence[str],
    raw_errors: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    input_ok = all(row["status"] == "pass" for row in input_rows)
    expected_request_count = int(config["execution_contract"]["expected_requests"])
    expected_shard_count = int(config["execution_contract"]["expected_shards"])
    expected = len(frames["requests"]) == expected_request_count
    unique_request_count = len({str(row.get("request_id", "")) for row in frames["requests"]})
    request_unique = unique_request_count == len(frames["requests"])
    rows = [
        ("repository", "repository state", "branch/head match", f"branch_status={repository['branch_status']};head_status={repository['head_status']}", repository["status"], DECISION_RUNTIME, RECOMMEND_RUNTIME, ""),
        ("interpreter", "interpreter identity", "configured interpreter is active", interpreter["actual_interpreter"], interpreter["status"], DECISION_RUNTIME, RECOMMEND_RUNTIME, ""),
        ("input_integrity", "frozen input hashes", "all direct inputs unchanged", f"pass={input_ok}", "pass" if input_ok else "fail", DECISION_INPUTS, RECOMMEND_CONTRACT, ""),
        ("phase_scientific", "prior phase scientific hashes", "F6G1/F6F2/F6F1/F6E4C match expected decisions and hashes", f"status={phase_checks['status']}", phase_checks["status"], DECISION_INPUTS, RECOMMEND_CONTRACT, ""),
        ("request_uniqueness", "request uniqueness", "every frozen F6G1 request ID is unique", f"unique={unique_request_count};rows={len(frames['requests'])}", "pass" if request_unique else "fail", DECISION_INPUTS, RECOMMEND_CONTRACT, ""),
        ("request_count", "request count", f"F6G1 request manifest has {expected_request_count} rows", f"rows={len(frames['requests'])}", "pass" if expected else "fail", DECISION_INPUTS, RECOMMEND_CONTRACT, ""),
        ("shard_count", "shard count", f"F6G1 shard manifest has {expected_shard_count} shards", f"rows={len(frames['shards'])}", "pass" if len(frames["shards"]) == expected_shard_count else "fail", DECISION_INPUTS, RECOMMEND_CONTRACT, ""),
        ("contract", "request contract reconciliation", "candidate_pool_size=20 but requested/parser length must be approved top-10", contract["blocking_reason"], contract["status"], DECISION_CONTRACT, RECOMMEND_CONTRACT, "This is the hard pre-inference gate."),
        ("raw_integrity", "raw JSONL uniqueness", "existing raw records, if any, have unique request IDs and valid JSONL", f"duplicates={len(raw_duplicates)};errors={len(raw_errors)}", "pass" if not raw_duplicates and not raw_errors else "fail", DECISION_RAW, RECOMMEND_RUNTIME, ""),
        ("model_identity", "model identity", "local model digest check happens only after contract gate passes", "not checked because contract gate is blocked", "skipped" if not contract["execution_allowed"] else "pending", DECISION_MODEL, RECOMMEND_RUNTIME, ""),
        ("raw_append", "raw append permission", "no raw writes while contract is blocked", "no raw append attempted", "pass", DECISION_RAW, RECOMMEND_RUNTIME, ""),
    ]
    return [
        {
            "gate_id": stable_id("F6G2-GATE", key),
            "gate_name": name,
            "requirement": requirement,
            "observed_evidence": evidence,
            "status": status,
            "impact": "blocking" if status in {"fail", "blocked"} else "nonblocking",
            "decision_if_failed": decision,
            "recommendation_if_failed": recommendation,
            "notes": notes,
        }
        for key, name, requirement, evidence, status, decision, recommendation, notes in rows
    ]


def build_missing_or_blocked(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    if contract["execution_allowed"]:
        return []
    return [
        {
            "block_id": stable_id("F6G2-BLOCK", "request_contract_conflict"),
            "component": "request_contract",
            "status": "blocked",
            "blocking_reason": contract["blocking_reason"],
            "current_evidence": f"F6G1 lengths={pipe_join(contract['request_output_lengths'])}; F6G1 parser={pipe_join(contract['parser_contracts'])}; top10 evidence={pipe_join(contract['top10_parser_expected_counts'])}",
            "required_evidence_or_action": "Create a corrective offline freeze that preserves F6G1 but provides a new top-10 request/parser contract for LastFM re-ranking.",
            "decision": DECISION_CONTRACT,
            "recommendation": RECOMMEND_CONTRACT,
            "notes": "No inference, no raw append and no prompt rewrite are allowed in F6G2 while this conflict remains.",
        },
        {
            "block_id": stable_id("F6G2-BLOCK", "model_identity_skipped"),
            "component": "model_identity",
            "status": "blocked_by_contract_gate",
            "blocking_reason": "Model identity is not queried because generation is already blocked.",
            "current_evidence": "Ollama was not called.",
            "required_evidence_or_action": "After a corrected request contract exists, verify local model digests without pulling models.",
            "decision": DECISION_CONTRACT,
            "recommendation": RECOMMEND_CONTRACT,
            "notes": "Digest values are not guessed.",
        },
    ]


def run_validation_suites(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if not config.get("validation", {}).get("run_suites", False):
        return {"suites": [], "total_passed": 0, "total_failed": 0, "total_skipped": 0, "unsafe_exclusions": list(config.get("validation", {}).get("unsafe_exclusions", []))}
    suites: list[dict[str, Any]] = []
    for suite in config["validation"]["suites"]:
        command = [sys.executable, "-m", "pytest", *suite["files"], "-q"]
        code, out, err, _duration = run_command(command, root, timeout=900)
        text = out + "\n" + err
        suites.append(
            {
                "suite": suite["name"],
                "status": "pass" if code == 0 else "fail",
                "passed": parse_pytest_count(text, "passed"),
                "failed": parse_pytest_count(text, "failed"),
                "skipped": parse_pytest_count(text, "skipped"),
                "command": " ".join(command),
            }
        )
    return {
        "suites": suites,
        "total_passed": sum(row["passed"] for row in suites),
        "total_failed": sum(row["failed"] for row in suites),
        "total_skipped": sum(row["skipped"] for row in suites),
        "unsafe_exclusions": list(config["validation"].get("unsafe_exclusions", [])),
    }


def skipped_validation_summary(config: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "suites": [
            {
                "suite": "validation skipped",
                "status": "blocked",
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "command": reason,
            }
        ],
        "total_passed": 0,
        "total_failed": 0,
        "total_skipped": 0,
        "unsafe_exclusions": list(config.get("validation", {}).get("unsafe_exclusions", [])),
    }


def parse_pytest_count(text: str, word: str) -> int:
    import re

    pattern = re.compile(rf"(\d+)\s+{word}")
    return sum(int(match.group(1)) for match in pattern.finditer(text))


def table_hashes(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    return {name: sha256_file(root / path) for name, path in sorted(config["outputs"]["tables"].items())}


def raw_counts(raw_records: Mapping[str, Mapping[str, Any]], parser_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    transports = Counter(str(row.get("transport_status", "success")) for row in raw_records.values())
    strict_valid = sum(1 for row in parser_rows if row["strict_valid"] == "true")
    invalid = [row for row in parser_rows if row["strict_valid"] != "true"]
    return {
        "raw_record_count": len(raw_records),
        "transport_counts": dict(sorted(transports.items())),
        "strict_valid_count": strict_valid,
        "invalid_count": len(invalid),
        "invalid_reason_distribution": dict(sorted(Counter(row["parse_status"] for row in invalid).items())),
    }


def completed_shard_count(request_rows: Sequence[Mapping[str, Any]]) -> int:
    by_shard: dict[str, list[Mapping[str, Any]]] = {}
    for row in request_rows:
        by_shard.setdefault(str(row.get("shard_id", "")), []).append(row)
    return sum(1 for rows in by_shard.values() if rows and all(row.get("execution_status") == "completed" for row in rows))


def determine_decision(
    input_rows: Sequence[Mapping[str, Any]],
    phase_checks: Mapping[str, Any],
    interpreter: Mapping[str, str],
    contract: Mapping[str, Any],
    raw_duplicates: Sequence[str],
    raw_errors: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    request_rows: Sequence[Mapping[str, Any]],
    parser_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    if any(row["status"] != "pass" for row in input_rows) or phase_checks["status"] != "pass":
        return DECISION_INPUTS, RECOMMEND_CONTRACT
    if interpreter["status"] != "pass":
        return DECISION_RUNTIME, RECOMMEND_RUNTIME
    if int(validation.get("total_failed", 0)) > 0:
        return DECISION_TESTS, RECOMMEND_RUNTIME
    if raw_duplicates or raw_errors:
        return DECISION_RAW, RECOMMEND_RUNTIME
    if not contract["execution_allowed"]:
        return DECISION_CONTRACT, RECOMMEND_CONTRACT
    completed = sum(1 for row in request_rows if row["execution_status"] == "completed")
    if completed != len(request_rows):
        return DECISION_INCOMPLETE, RECOMMEND_RESUME
    invalid = [row for row in parser_rows if row["strict_valid"] != "true"]
    if invalid:
        return DECISION_COMPLETE_INVALID, RECOMMEND_COMPLETE
    return DECISION_COMPLETE, RECOMMEND_COMPLETE


def build_report(
    repository: Mapping[str, Any],
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    counts: Mapping[str, int],
) -> str:
    sections = [
        ("Initial Repository State", f"Branch `{repository['branch']}`, HEAD `{repository['head']}`, staged files `{repository['staged_files']}`. Protected dirty tracked files remain `{repository['tracked_dirty_files']}`."),
        ("Frozen Inputs", f"Changed inputs: `{manifest['input_invariance']['changed']}`; missing inputs: `{manifest['input_invariance']['missing']}`."),
        ("Contract Reconciliation", contract["blocking_reason"]),
        ("Candidate Pool Size", f"F6G1 candidate pool size evidence: `{pipe_join(contract['candidate_pool_sizes'])}`."),
        ("Requested Ranking Length", f"F6G1 requested length: `{pipe_join(contract['request_output_lengths'])}`; approved required length: `{contract['required_requested_ranking_length']}`."),
        ("Parser Expected Length", f"F6G1 parser schema length: `{pipe_join(contract['parser_schema_lengths'])}`; approved parser length: `{contract['required_parser_expected_length']}`."),
        ("Execution Status", "No Ollama generation, prompt rendering, retries or raw JSONL appends were attempted."),
        ("Shard Status", f"Shard rows written: `{counts['shard_status']}`; all shards are blocked before inference."),
        ("Request Status", f"Request rows written: `{counts['request_execution']}`; each request is marked `blocked_pre_inference_contract_conflict`."),
        ("Raw Outputs", "F6G2 raw files are configured but are not created by this blocked freeze."),
        ("Population Summaries", "All-user, strict-valid, positive-present operational and positive-present strict-valid populations are materialized separately with blocked status."),
        ("Tests", f"Validation passed `{manifest['validation_summary']['total_passed']}`, failed `{manifest['validation_summary']['total_failed']}`."),
        ("Final Decision", manifest["decision"]),
        ("Recommendation", manifest["recommended_next_action"]),
        ("Scientific Content Hash", manifest["scientific_content_hash"]),
    ]
    lines = ["# Phase F6G2 - Contract-Gated LastFM Re-ranking Execution Freeze", ""]
    for title, body in sections:
        lines.extend([f"## {title}", "", str(body), ""])
    return "\n".join(lines).rstrip() + "\n"


def build_manifest(
    root: Path,
    config: Mapping[str, Any],
    repository: Mapping[str, Any],
    interpreter: Mapping[str, str],
    phase_checks: Mapping[str, Any],
    input_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    raw_records: Mapping[str, Mapping[str, Any]],
    raw_duplicates: Sequence[str],
    raw_errors: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    request_rows: Sequence[Mapping[str, Any]],
    parser_rows: Sequence[Mapping[str, Any]],
    counts: Mapping[str, int],
) -> dict[str, Any]:
    hashes = table_hashes(root, config)
    decision, recommendation = determine_decision(input_rows, phase_checks, interpreter, contract, raw_duplicates, raw_errors, validation, request_rows, parser_rows)
    raw_stat = raw_file_stats(root, config)
    raw_summary = raw_counts(raw_records, parser_rows)
    scientific = {
        "phase": PHASE_ID,
        "f6g1_scientific_hash": phase_checks["f6g1_scientific_hash"],
        "f6f2_scientific_hash": phase_checks["f6f2_scientific_hash"],
        "f6f1_scientific_hash": phase_checks["f6f1_scientific_hash"],
        "f6e4c_scientific_hash": phase_checks["f6e4c_scientific_hash"],
        "f6g1_future_logical_instance_hash": phase_checks["f6g1_future_logical_instance_hash"],
        "f6g1_future_request_manifest_hash": phase_checks["f6g1_future_request_manifest_hash"],
        "f6g1_future_shard_manifest_hash": phase_checks["f6g1_future_shard_manifest_hash"],
        "direct_input_hashes": {row["path"]: row["actual_sha256"] for row in input_rows},
        "contract_reconciliation": dict(contract),
        "raw_file_stats": raw_stat,
        "raw_uniqueness": {"duplicate_request_ids": list(raw_duplicates), "jsonl_errors": list(raw_errors)},
        "completeness": {
            "expected_requests": len(request_rows),
            "completed_requests": sum(1 for row in request_rows if row["execution_status"] == "completed"),
            "blocked_requests": sum(1 for row in request_rows if str(row["execution_status"]).startswith("blocked_")),
            "missing_requests": len(plan_resume([str(row["request_id"]) for row in request_rows], raw_records)),
            "expected_shards": int(config["execution_contract"]["expected_shards"]),
            "completed_shards": completed_shard_count(request_rows),
        },
        "transport_counts": raw_summary["transport_counts"],
        "strict_valid_count": raw_summary["strict_valid_count"],
        "invalid_count": raw_summary["invalid_count"],
        "invalid_reason_distribution": raw_summary["invalid_reason_distribution"],
        "table_counts": dict(counts),
        "table_hashes": hashes,
        "protected_input_result": {
            "changed": sum(1 for row in input_rows if row["changed"] == "true"),
            "missing": sum(1 for row in input_rows if row["missing"] == "true"),
        },
        "decision": decision,
        "recommended_next_action": recommendation,
    }
    scientific_hash = sha256_text(stable_json(scientific))
    previous = existing_manifest(root, config)
    return {
        **scientific,
        "scientific_content_hash": scientific_hash,
        "phase_checks": dict(phase_checks),
        "interpreter_gate": dict(interpreter),
        "validation_summary": validation,
        "unsafe_excluded_tests": list(config.get("validation", {}).get("unsafe_exclusions", [])),
        "input_invariance": {
            "changed": sum(1 for row in input_rows if row["changed"] == "true"),
            "missing": sum(1 for row in input_rows if row["missing"] == "true"),
        },
        "runtime_metadata": {
            "timestamp": repository["timestamp"],
            "repository_state": dict(repository),
            "python": sys.executable,
            "platform": platform.platform(),
            "analysis_only_determinism_result": "stable_with_previous_manifest" if previous.get("scientific_content_hash") == scientific_hash else "first_run_or_scientific_change",
            "runtime_metadata_separated_from_scientific_hash": True,
        },
    }


def run_phase(root: Path, config: Mapping[str, Any], *, mode: str = "preflight-only") -> dict[str, Any]:
    validate_output_paths(config)
    repository = record_repository_state(root, config)
    interpreter = verify_interpreter(root, config)
    input_rows = verify_input_integrity(root, config)
    assert_no_input_failures(input_rows)
    phase_checks = verify_phase_manifests(root, config)
    if phase_checks["status"] != "pass":
        raise RuntimeError(f"F6G2 frozen phase scientific gate failed: {phase_checks}")

    frames = load_frames(root, config)
    contract = contract_reconciliation(frames, config)
    raw_records, raw_duplicates, raw_errors = load_raw_records(root, config)
    request_rows = build_request_execution(frames, config, contract, raw_records)
    parser_rows = build_parser_validity(frames, contract, raw_records)
    invalid_rows = build_invalid_output_audit(parser_rows, contract)
    cell_rows = build_cell_execution_summary(request_rows, parser_rows, contract)
    population_rows = build_population_execution_summary(cell_rows, contract)

    table_rows = {
        "repository_state": build_repository_state_table(repository),
        "input_integrity": input_rows,
        "runtime_gate": build_runtime_gate(repository, interpreter, input_rows, phase_checks, contract),
        "model_identity": build_model_identity(config, contract),
        "progress": build_progress(mode, repository, request_rows, parser_rows, contract, raw_records),
        "shard_status": build_shard_status(frames, request_rows, contract),
        "request_execution": request_rows,
        "parser_validity": parser_rows,
        "invalid_output_audit": invalid_rows,
        "cell_execution_summary": cell_rows,
        "population_execution_summary": population_rows,
        "dataset_task_execution_summary": build_dataset_task_execution_summary(frames, request_rows, contract),
        "execution_gate": build_execution_gate(config, repository, interpreter, input_rows, phase_checks, frames, contract, raw_duplicates, raw_errors),
        "missing_or_blocked": build_missing_or_blocked(contract),
    }
    for name, rows in table_rows.items():
        write_csv(root / config["outputs"]["tables"][name], rows, TABLE_COLUMNS[name])

    if interpreter["status"] == "pass":
        validation = run_validation_suites(root, config)
    else:
        validation = skipped_validation_summary(config, "configured interpreter gate failed; safe suites not run")
    counts = {name: len(rows) for name, rows in table_rows.items()}
    manifest = build_manifest(root, config, repository, interpreter, phase_checks, input_rows, contract, raw_records, raw_duplicates, raw_errors, validation, request_rows, parser_rows, counts)
    report = build_report(repository, manifest, contract, counts)
    report_path = root / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    manifest["report_hash"] = sha256_file(report_path)
    write_json(root / config["outputs"]["freeze_manifest"], manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Phase F6G2 contract-gated LastFM re-ranking execution freeze artifacts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to F6G2 YAML config.")
    parser.add_argument("--preflight-only", action="store_true", help="Run gates and write blocked/preflight artifacts.")
    parser.add_argument("--integrity-only", action="store_true", help="Run input/raw integrity gates and write artifacts.")
    parser.add_argument("--analysis-only", action="store_true", help="Analyze existing raw records without model calls.")
    args = parser.parse_args(argv)
    mode = "analysis-only" if args.analysis_only else "integrity-only" if args.integrity_only else "preflight-only"
    root = Path.cwd()
    config = load_config(root / args.config)
    manifest = run_phase(root, config, mode=mode)
    print(
        stable_json(
            {
                "phase": PHASE_ID,
                "decision": manifest["decision"],
                "recommendation": manifest["recommended_next_action"],
                "scientific_content_hash": manifest["scientific_content_hash"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
