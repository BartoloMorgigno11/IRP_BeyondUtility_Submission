"""Local-only profile execution for F6L.

This module is deliberately scoped to the 4800 Beauty/Sports profile-family
requests frozen in ``phase_f6l_profile_request_manifest.csv``. It does not
materialize or execute LastFM cells.
"""

from __future__ import annotations

import ast
import json
import math
import re
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.analysis import plan_f6l_paper_faithful_completion_and_rq_revalidation as base


DECISION_PROFILE_EXECUTION_COMPLETE = "f6l_profile_open_model_execution_and_rq_revalidation_complete"
DECISION_PROFILE_EXECUTION_PARTIAL = "f6l_profile_execution_partial_resumable"
DECISION_PROFILE_EXECUTION_BLOCKED = "f6l_profile_execution_blocked_by_contract_runtime_or_protected_input_regression"
OLLAMA_GENERATE_URL = "http://127.0.0.1:11434/api/generate"

PROFILE_EXECUTION_TABLES = {
    "profile_execution_preflight": base.TABLES_DIR / "phase_f6l_profile_execution_preflight.csv",
    "profile_execution_state": base.TABLES_DIR / "phase_f6l_profile_execution_state.csv",
    "profile_parse_results": base.TABLES_DIR / "phase_f6l_profile_parse_results.csv",
    "profile_recommendation_rows": base.TABLES_DIR / "phase_f6l_profile_recommendation_rows.csv",
    "profile_metrics": base.TABLES_DIR / "phase_f6l_profile_metrics.csv",
    "profile_metric_differences": base.TABLES_DIR / "phase_f6l_profile_metric_differences.csv",
    "profile_runtime_resources": base.TABLES_DIR / "phase_f6l_profile_runtime_resources.csv",
    "profile_rq_revalidation": base.TABLES_DIR / "phase_f6l_profile_rq_revalidation.csv",
}

PROFILE_EXECUTION_MANIFEST_PATH = base.MANIFESTS_DIR / "phase_f6l_profile_execution_manifest.json"

PROFILE_EXECUTION_FIELDS = {
    "profile_execution_preflight": [
        "check_id",
        "check_name",
        "required",
        "observed",
        "status",
        "severity",
        "evidence",
        "notes",
    ],
    "profile_execution_state": [
        "request_id",
        "run_id",
        "status",
        "strict_valid",
        "dataset",
        "user_id",
        "model_name",
        "model_digest",
        "profile_generator_name",
        "profile_generator_digest",
        "recommender_model_name",
        "input_variant",
        "dependency_request_id",
        "experiment_cell_id",
        "prompting_strategy",
        "rendered_prompt_hash",
        "actual_raw_output_relative_path",
        "started_at",
        "finished_at",
        "wall_clock_seconds",
        "retry_count",
        "error",
        "parser_status",
        "parsed_count",
        "valid_item_count",
        "invalid_item_count",
        "hallucination_count",
        "duplicate_count",
        "hit_at_5",
        "ndcg_at_5",
        "prompt_tokens",
        "completion_tokens",
        "total_duration_ns",
        "load_duration_ns",
        "prompt_eval_duration_ns",
        "eval_duration_ns",
        "raw_response_sha256",
        "raw_response_bytes",
        "source_lineage",
        "input_contract_hash",
    ],
    "profile_parse_results": [
        "request_id",
        "run_id",
        "dataset",
        "user_id",
        "model_name",
        "profile_generator_name",
        "recommender_model_name",
        "input_variant",
        "dependency_request_id",
        "candidate_list_hash",
        "positive_item_id",
        "positive_candidate_position",
        "parser_contract",
        "parsed_titles_json",
        "parsed_item_ids_json",
        "invalid_titles_json",
        "parser_status",
        "strict_valid",
        "hit_at_5",
        "ndcg_at_5",
        "duplicate_count",
        "hallucination_count",
        "parsing_time_seconds",
        "source_file",
        "evidence_classification",
    ],
    "profile_recommendation_rows": [
        "request_id",
        "run_id",
        "dataset",
        "user_id",
        "model_name",
        "profile_generator_name",
        "recommender_model_name",
        "input_variant",
        "rank",
        "recommended_item_id",
        "recommended_item_title",
        "match_status",
        "is_positive",
        "source_file",
        "evidence_classification",
    ],
    "profile_metrics": [
        "metric_id",
        "run_id",
        "dataset",
        "profile_generator",
        "recommender_model",
        "input_variant",
        "metric",
        "value",
        "numerator",
        "denominator",
        "status",
        "evidence_classification",
        "source_file",
        "notes",
    ],
    "profile_metric_differences": [
        "comparison_id",
        "run_id",
        "dataset",
        "profile_generator",
        "recommender_model",
        "comparison",
        "metric",
        "baseline_value",
        "treatment_value",
        "absolute_difference",
        "relative_difference",
        "paired_users",
        "status",
        "evidence_classification",
        "source_file",
        "notes",
    ],
    "profile_runtime_resources": [
        "runtime_id",
        "run_id",
        "scope",
        "model",
        "request_count",
        "completed_requests",
        "strict_valid_requests",
        "failed_requests",
        "residual_requests",
        "retry_count",
        "wall_clock_seconds",
        "latency_mean",
        "latency_median",
        "latency_p90",
        "latency_p95",
        "latency_p99",
        "throughput_requests_per_hour",
        "prompt_tokens",
        "completion_tokens",
        "raw_output_bytes",
        "parsing_time_seconds",
        "consolidation_time_seconds",
        "cpu",
        "ram",
        "gpu",
        "vram",
        "marginal_api_cost",
        "hardware_cost",
        "energy_cost",
        "total_cost_claim",
        "source_file",
        "notes",
    ],
    "profile_rq_revalidation": [
        "rq_id",
        "previous_status",
        "paper_claim",
        "beauty_evidence",
        "sports_evidence",
        "qwen2_evidence",
        "mistral_evidence",
        "profile_only",
        "profile_plus_history",
        "dataset_consistency",
        "model_consistency",
        "new_status",
        "motivation",
        "limits",
        "source_file",
    ],
}

_SOURCE_ROW_CACHE: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pct(values: Sequence[float], percentile: float) -> str:
    if not values:
        return ""
    ordered = sorted(values)
    index = int(math.ceil((percentile / 100.0) * len(ordered))) - 1
    return f"{ordered[max(0, min(index, len(ordered) - 1))]:.6f}"


def mean(values: Sequence[float]) -> str:
    return f"{statistics.mean(values):.6f}" if values else ""


def median(values: Sequence[float]) -> str:
    return f"{statistics.median(values):.6f}" if values else ""


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def source_rows_by_user(dataset: str, variant: str) -> dict[str, Mapping[str, Any]]:
    key = (dataset, variant)
    if key not in _SOURCE_ROW_CACHE:
        rows = base.zip_json_rows(dataset, base.PROFILE_SOURCE_ENTRIES[variant])
        _SOURCE_ROW_CACHE[key] = {str(row["user"]): row for row in rows}
    return _SOURCE_ROW_CACHE[key]


def source_row_for_manifest(row: Mapping[str, str]) -> Mapping[str, Any]:
    variant = row["input_variant"]
    by_user = source_rows_by_user(row["dataset"], variant)
    source = by_user.get(str(row["user_id"]))
    if source is None:
        raise KeyError(f"missing source row for {row['request_id']} user={row['user_id']} variant={variant}")
    return source


def candidate_ids_and_titles(source_row: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    candidate_ids = base.parse_candidate_ids(str(source_row["cand_list"])) if source_row.get("cand_list") else []
    candidate_titles = base.parse_prompt_candidate_titles(str(source_row.get("prompt", "")))
    return candidate_ids, candidate_titles


def profile_render_recipe(source_prompt: str, dependency_request_id: str) -> dict[str, str]:
    return {
        "source_prompt_hash": base.sha256_text(source_prompt),
        "dependency_request_id": dependency_request_id,
        "profile_placeholder": "replace author-profile exemplar with local parent profile output after strict-valid parent completion",
    }


def quote_profile(profile_text: str) -> str:
    compact = " ".join(profile_text.strip().split())
    return repr(compact)


def replace_profile_text(prompt: str, profile_text: str) -> str:
    quoted = quote_profile(profile_text)
    patterns = [
        r"(user profile goes like:\s*)(.*?)(\. Please select)",
        r"(has been profiled as:\s*)(.*?)(\. Please select)",
        r"('s user profile is:\s*)(.*?)(\. Please select)",
        r"('s user profile:\s*)(.*?)(,\s*please select)",
        r"(Here is the user profile of .*?:\s*)(.*?)(\. Please select)",
    ]
    for pattern in patterns:
        def replacement(match: re.Match[str]) -> str:
            return f"{match.group(1)}{quoted}{match.group(3)}"

        updated, count = re.subn(pattern, replacement, prompt, count=1, flags=re.DOTALL)
        if count:
            return updated
    raise ValueError("could not locate profile text placeholder in dependent prompt")


def render_prompt(row: Mapping[str, str], parent_profiles: Mapping[str, str]) -> str:
    source = source_row_for_manifest(row)
    prompt = str(source.get("prompt", ""))
    dependency = row.get("dependency_request_id", "")
    if not dependency:
        return prompt
    profile_text = parent_profiles.get(dependency, "")
    if not profile_text.strip():
        raise ValueError(f"parent profile is not strict-valid for {row['request_id']}: {dependency}")
    return replace_profile_text(prompt, profile_text)


def request_order_key(row: Mapping[str, str]) -> tuple[int, str]:
    order = {
        "profile_generation": 0,
        "history_only": 1,
        "profile_only": 2,
        "profile_plus_history": 3,
    }
    return order.get(row.get("input_variant", ""), 99), row.get("request_id", "")


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()


def model_digest_rows(ollama_payload: Mapping[str, Any], error: str = "") -> list[dict[str, str]]:
    rows = base.ollama_model_preflight_rows(ollama_payload, error)
    for row in rows:
        expected = base.MODEL_DIGESTS[row["model"]]
        digest_match = row["digest"] == expected
        row["status"] = "pass" if row["available"] == "true" and digest_match else "fail"
        row["notes"] = (
            "Local Ollama tag and digest match the frozen F6L contract."
            if row["status"] == "pass"
            else f"Expected digest {expected}; observed {row['digest'] or 'missing'}. {error}".strip()
        )
    return rows


def preflight_row(index: int, name: str, required: str, observed: str, ok: bool, evidence: str, notes: str) -> dict[str, str]:
    return {
        "check_id": f"F6L-PROFILE-PREFLIGHT-{index:03d}",
        "check_name": name,
        "required": required,
        "observed": observed,
        "status": "pass" if ok else "fail",
        "severity": "error",
        "evidence": evidence,
        "notes": notes,
    }


def validate_manifest_inputs(rows: Sequence[Mapping[str, str]]) -> list[str]:
    issues: list[str] = []
    for row in rows:
        try:
            source = source_row_for_manifest(row)
            prompt = str(source.get("prompt", ""))
            template_path = base.resolve_project_path(Path(row["prompt_template_path"]))
            if not template_path.exists() or base.file_sha256(template_path) != row["prompt_template_hash"]:
                issues.append(f"{row['request_id']}:prompt_template_hash")
            if row["dependency_request_id"]:
                expected_render_hash = base.sha256_text(base.canonical_json(profile_render_recipe(prompt, row["dependency_request_id"])))
                if expected_render_hash != row["rendered_prompt_hash"]:
                    issues.append(f"{row['request_id']}:deferred_render_recipe_hash")
            elif base.sha256_text(prompt) != row["rendered_prompt_hash"]:
                issues.append(f"{row['request_id']}:rendered_prompt_hash")
            if row["input_variant"] != "profile_generation":
                candidate_ids, candidate_titles = candidate_ids_and_titles(source)
                candidate_hash = base.sha256_text(base.canonical_json(candidate_ids))
                positive_id, positive_position = base.positive_candidate(source)
                if len(candidate_ids) != 20 or len(candidate_titles) != 20:
                    issues.append(f"{row['request_id']}:candidate20")
                if candidate_hash != row["candidate_list_hash"]:
                    issues.append(f"{row['request_id']}:candidate_hash")
                if positive_id != row["positive_item_id"] or positive_position != row["positive_candidate_position"]:
                    issues.append(f"{row['request_id']}:positive_mapping")
            if row["input_variant"] == "profile_generation":
                if row["output_length"] != "profile_text" or row["parser_contract"] != "free_text_profile_retained":
                    issues.append(f"{row['request_id']}:profile_parser_contract")
            elif row["output_length"] != "10" or row["parser_contract"] != "paper_title_list_top10_parser":
                issues.append(f"{row['request_id']}:recommendation_parser_contract")
        except Exception as exc:
            issues.append(f"{row.get('request_id', 'unknown')}:{type(exc).__name__}:{exc}")
        if len(issues) > 20:
            break
    return issues


def raw_dir_resume_status(raw_output_dir: Path, rows: Sequence[Mapping[str, str]], manifest_hash: str) -> tuple[bool, str]:
    if not raw_output_dir.exists():
        return True, "new_non_existing"
    manifest_path = raw_output_dir / "profile_execution_manifest.json"
    if not manifest_path.exists():
        return False, "exists_without_profile_execution_manifest"
    payload = base.read_json_file(manifest_path)
    request_ids = {row["request_id"] for row in rows}
    prior_ids = set(payload.get("request_ids", [])) if isinstance(payload.get("request_ids", []), list) else set()
    same_manifest = payload.get("profile_manifest_sha256") == manifest_hash
    same_checkpoint = payload.get("checkpoint_id") == base.DECISION_SEPARATED_PROFILE_READY
    if same_manifest and same_checkpoint and prior_ids.issubset(request_ids):
        return True, "valid_resume_directory"
    return False, "exists_but_not_matching_f6l_profile_resume_contract"


def build_profile_preflight(args: Any, ollama_payload: Mapping[str, Any] | None = None, ollama_error: str = "") -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    base.require_runtime()
    rows = base.read_csv_rows(base.PROFILE_REQUEST_MANIFEST_PATH)
    checkpoint = base.read_json_file(base.SEPARATED_CHECKPOINT_MANIFEST_PATH)
    checkpoint_hashes = checkpoint.get("output_hashes", {}) if isinstance(checkpoint.get("output_hashes", {}), Mapping) else {}
    manifest_hash = base.safe_file_sha256(base.PROFILE_REQUEST_MANIFEST_PATH)
    expected_manifest_hash = str(checkpoint_hashes.get(base.rel(base.PROFILE_REQUEST_MANIFEST_PATH), ""))
    raw_output_dir = base.resolve_project_path(args.raw_output_dir)
    raw_root = (base.PROJECT_ROOT / "results" / "llm").resolve()
    protected_checkpoint = base.read_json_file(base.PROTECTED_HASHES_PATH)
    current_hashes = base.hash_protected_inputs()
    regressions = base.protected_hash_regressions(current_hashes, protected_checkpoint) if protected_checkpoint else ["missing_protected_hash_checkpoint"]
    manifest_checks = base.validate_profile_manifest(rows)
    manifest_input_issues = validate_manifest_inputs(rows) if rows else ["manifest_missing"]
    if ollama_payload is None and not ollama_error:
        ollama_payload, ollama_error = base.fetch_local_ollama_tags()
    model_rows = model_digest_rows(ollama_payload or {}, ollama_error)
    model_ok = all(row["status"] == "pass" for row in model_rows)
    staged = base.git_output(["diff", "--cached", "--name-only"])
    original_repo_status = base.git_output(["status", "--porcelain", "--", "original_repo"])
    raw_dir_ok, raw_dir_status = raw_dir_resume_status(raw_output_dir, rows, manifest_hash)
    variants = Counter(row.get("input_variant", "") for row in rows)
    datasets = Counter(row.get("dataset", "") for row in rows)
    models = Counter(row.get("model_name", "") for row in rows)
    dependencies = [row.get("dependency_request_id", "") for row in rows if row.get("dependency_request_id", "")]

    checks = [
        (
            "checkpoint_id_matches_profile_ready_token",
            base.DECISION_SEPARATED_PROFILE_READY,
            args.checkpoint_id,
            args.checkpoint_id == base.DECISION_SEPARATED_PROFILE_READY,
            "CLI --checkpoint-id",
            "Execution must be tied to the separated F6L profile checkpoint.",
        ),
        (
            "manifest_present_and_byte_identical_to_checkpoint",
            f"{base.rel(base.PROFILE_REQUEST_MANIFEST_PATH)} sha256={expected_manifest_hash}",
            f"exists={base.PROFILE_REQUEST_MANIFEST_PATH.exists()};sha256={manifest_hash}",
            base.PROFILE_REQUEST_MANIFEST_PATH.exists() and manifest_hash == expected_manifest_hash,
            f"{base.rel(base.SEPARATED_CHECKPOINT_MANIFEST_PATH)};{base.rel(base.PROFILE_REQUEST_MANIFEST_PATH)}",
            "The frozen 4800-row manifest must not change between checkpoint and execution.",
        ),
        (
            "manifest_rows_and_unique_request_ids",
            "4800 rows and 4800 unique request_id values",
            f"rows={len(rows)};unique={len({row.get('request_id', '') for row in rows})}",
            len(rows) == 4800 and len({row.get("request_id", "") for row in rows}) == 4800,
            base.rel(base.PROFILE_REQUEST_MANIFEST_PATH),
            "Only rows in this manifest are authorized.",
        ),
        (
            "family_dataset_model_counts",
            "profile_generation=800;history_only=800;profile_only=1600;profile_plus_history=1600;beauty=2400;sports=2400;qwen2=2400;mistral=2400",
            f"variants={dict(variants)};datasets={dict(datasets)};models={dict(models)}",
            variants == {"profile_generation": 800, "history_only": 800, "profile_only": 1600, "profile_plus_history": 1600}
            and datasets == {"beauty": 2400, "sports": 2400}
            and models == {"qwen2:7b-instruct": 2400, "mistral:7b-instruct": 2400},
            base.rel(base.PROFILE_REQUEST_MANIFEST_PATH),
            "The profile execution scope excludes LastFM and every unapproved model.",
        ),
        (
            "manifest_dependency_graph_valid_acyclic",
            "all dependency_request_id values resolve to one profile_generation parent and no cycles",
            ";".join(f"{name}={observed}" for name, observed, _, _ in manifest_checks if name.startswith("profile_dependencies")),
            all(ok for name, _, ok, _ in manifest_checks if name.startswith("profile_dependencies")) and len(dependencies) == 3200,
            base.rel(base.PROFILE_REQUEST_MANIFEST_PATH),
            "Dependent profile prompts can execute only after strict-valid parent profiles.",
        ),
        (
            "prompt_candidate_positive_contracts_frozen",
            "prompt template hashes, rendered hashes/recipes, candidate hashes, positive item mapping, output length, parser",
            "ok" if not manifest_input_issues else ";".join(manifest_input_issues[:8]),
            not manifest_input_issues,
            "original_repo/EvaLLMasRecommender/llm_results/*.zip;original_repo/EvaLLMasRecommender/eva/templates/*.txt",
            "This reads official artifacts and does not modify them.",
        ),
        (
            "local_only_ollama_endpoint",
            "localhost or 127.0.0.1 only for tags and generation",
            f"tags={base.OLLAMA_TAGS_URL};generate={OLLAMA_GENERATE_URL};allow_api_calls={args.allow_api_calls};local_only={args.local_only}",
            bool(args.allow_api_calls and args.local_only and base.OLLAMA_TAGS_URL.startswith("http://127.0.0.1:") and OLLAMA_GENERATE_URL.startswith("http://127.0.0.1:")),
            "CLI flags and local endpoint constants",
            "--allow-api-calls is interpreted only as permission for local Ollama.",
        ),
        (
            "local_model_digests_match",
            "Qwen2 and Mistral digests exactly match the frozen contract",
            ";".join(f"{row['model']}={row['status']}:{row['digest']}" for row in model_rows),
            model_ok,
            base.OLLAMA_TAGS_URL,
            "No remote model or paid API is permitted.",
        ),
        (
            "raw_output_dir_new_or_valid_resume_under_results_llm",
            "new non-existing path below results/llm, or valid matching resumable F6L profile directory",
            f"{raw_dir_status}:{base.rel(raw_output_dir) if base.path_is_under(raw_output_dir, base.PROJECT_ROOT) else raw_output_dir}",
            base.path_is_under(raw_output_dir, raw_root) and raw_dir_ok,
            "results/llm",
            "Raw responses are append-only; unrelated existing directories are rejected.",
        ),
        (
            "protected_inputs_and_23_f6i2p_yaml_unchanged",
            "no protected hash regressions and 23 protected F6I2P YAML configs",
            f"status={current_hashes.get('status')};f6i2p={current_hashes.get('f6i2p_yaml_count')};regressions={';'.join(regressions[:6]) if regressions else 'none'}",
            current_hashes.get("status") == "pass" and current_hashes.get("f6i2p_yaml_count") == 23 and not regressions,
            base.rel(base.PROTECTED_HASHES_PATH),
            "Execution cannot start after protected input mutation.",
        ),
        (
            "git_staged_empty",
            "no staged paths",
            staged or "empty",
            staged == "",
            "git diff --cached --name-only",
            "The phase must not stage changes automatically.",
        ),
        (
            "original_repo_unmodified",
            "no original_repo working-tree changes",
            original_repo_status or "empty",
            original_repo_status == "",
            "git status --porcelain -- original_repo",
            "Execution reads official artifacts but must not mutate original_repo/.",
        ),
    ]
    preflight_rows = [preflight_row(index, *check) for index, check in enumerate(checks, start=1)]
    manifest = {
        "phase_id": base.PHASE_ID,
        "phase_name": base.PHASE_NAME,
        "checkpoint_id": args.checkpoint_id,
        "profile_manifest": base.rel(base.PROFILE_REQUEST_MANIFEST_PATH),
        "profile_manifest_sha256": manifest_hash,
        "expected_profile_manifest_sha256": expected_manifest_hash,
        "planned_requests": len(rows),
        "unique_request_ids": len({row.get("request_id", "") for row in rows}),
        "counts_by_input_variant": dict(variants),
        "counts_by_dataset": dict(datasets),
        "counts_by_model": dict(models),
        "raw_output_dir": base.rel(raw_output_dir) if base.path_is_under(raw_output_dir, base.PROJECT_ROOT) else str(raw_output_dir),
        "raw_output_dir_status": raw_dir_status,
        "blocking_checks": [row["check_name"] for row in preflight_rows if row["status"] != "pass"],
        "inference_performed": False,
        "remote_api_calls_performed": False,
        "lastfm_requests_performed": False,
        "downloads_performed": False,
        "training_performed": False,
        "model_preflight": model_rows,
        "created_at": utc_now(),
    }
    return preflight_rows, model_rows, manifest


def write_blocked_preflight(preflight_rows: Sequence[Mapping[str, str]], manifest: Mapping[str, Any]) -> None:
    base.write_csv(PROFILE_EXECUTION_TABLES["profile_execution_preflight"], preflight_rows, PROFILE_EXECUTION_FIELDS["profile_execution_preflight"])
    blocked_manifest = {
        **manifest,
        "decision_token": DECISION_PROFILE_EXECUTION_BLOCKED,
        "output_hashes": {
            base.rel(PROFILE_EXECUTION_TABLES["profile_execution_preflight"]): base.safe_file_sha256(PROFILE_EXECUTION_TABLES["profile_execution_preflight"])
        },
    }
    base.write_json(PROFILE_EXECUTION_MANIFEST_PATH, blocked_manifest)
    update_profile_report_handoff_notes(blocked_manifest, [], [], [], [], [], [], preflight_rows)


def extract_titles_from_response(response_text: str) -> list[str]:
    text = response_text.strip()
    if not text:
        return []
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if 0 <= bracket_start < bracket_end:
        candidate = text[bracket_start : bracket_end + 1]
        try:
            value = ast.literal_eval(candidate)
            if isinstance(value, (list, tuple)):
                return [str(item).strip() for item in value if str(item).strip()]
        except (SyntaxError, ValueError):
            pass
    quoted = re.findall(r"['\"]([^'\"]{2,300})['\"]", text)
    if quoted:
        return [item.strip() for item in quoted if item.strip()]
    lines: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[\).:-])\s*", "", line).strip()
        cleaned = cleaned.strip("`'\" ")
        if cleaned:
            lines.append(cleaned)
    return lines


def parse_recommendation_response(
    response_text: str,
    candidate_ids: Sequence[str],
    candidate_titles: Sequence[str],
    positive_item_id: str,
) -> dict[str, Any]:
    parsed_titles = extract_titles_from_response(response_text)
    title_to_id: dict[str, str] = {}
    title_to_title: dict[str, str] = {}
    for item_id, title in zip(candidate_ids, candidate_titles):
        normalized = base.normalized_title(title)
        title_to_id[normalized] = item_id
        title_to_title[normalized] = title

    parsed_ids: list[str] = []
    canonical_titles: list[str] = []
    invalid_titles: list[str] = []
    for title in parsed_titles[:10]:
        normalized = base.normalized_title(title)
        item_id = title_to_id.get(normalized)
        if item_id is None:
            parsed_ids.append("")
            canonical_titles.append(title)
            invalid_titles.append(title)
        else:
            parsed_ids.append(item_id)
            canonical_titles.append(title_to_title[normalized])

    valid_ids = [item_id for item_id in parsed_ids if item_id]
    duplicate_count = len(valid_ids) - len(set(valid_ids))
    invalid_count = len([item_id for item_id in parsed_ids if not item_id])
    strict_valid = len(parsed_titles) == 10 and len(parsed_ids) == 10 and invalid_count == 0 and duplicate_count == 0
    hit_rank = 0
    for index, item_id in enumerate(parsed_ids[:5], start=1):
        if item_id == positive_item_id:
            hit_rank = index
            break
    ndcg = 1.0 / math.log2(hit_rank + 1) if hit_rank else 0.0
    return {
        "parsed_titles": canonical_titles,
        "parsed_item_ids": parsed_ids,
        "invalid_titles": invalid_titles,
        "parser_status": "strict_valid" if strict_valid else "parsed_with_contract_violation",
        "strict_valid": strict_valid,
        "hit_at_5": 1 if hit_rank else 0,
        "ndcg_at_5": ndcg,
        "duplicate_count": duplicate_count,
        "hallucination_count": invalid_count,
        "valid_item_count": len(valid_ids),
        "invalid_item_count": invalid_count,
    }


def parse_profile_response(response_text: str) -> dict[str, Any]:
    strict_valid = bool(response_text.strip())
    return {
        "parser_status": "strict_valid" if strict_valid else "empty_profile_text",
        "strict_valid": strict_valid,
        "parsed_titles": [],
        "parsed_item_ids": [],
        "invalid_titles": [],
        "hit_at_5": "",
        "ndcg_at_5": "",
        "duplicate_count": 0,
        "hallucination_count": 0,
        "valid_item_count": 0,
        "invalid_item_count": 0,
    }


def ollama_generate(model_name: str, prompt: str, generation_parameters: Mapping[str, Any]) -> dict[str, Any]:
    if not OLLAMA_GENERATE_URL.startswith("http://127.0.0.1:"):
        raise RuntimeError("blocked_non_local_ollama_generate_endpoint")
    options = {
        "temperature": generation_parameters.get("temperature", 0),
        "seed": generation_parameters.get("seed", 0),
    }
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    timeout_seconds = int(generation_parameters.get("timeout_seconds", 120) or 120)
    request = urllib.request.Request(
        OLLAMA_GENERATE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        decoded = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(decoded)
    if not isinstance(parsed, dict):
        raise RuntimeError("ollama_generate_response_not_object")
    return parsed


def resource_snapshot() -> dict[str, str]:
    try:
        import psutil  # type: ignore

        process = psutil.Process()
        memory = process.memory_info()
        return {
            "cpu": f"{process.cpu_percent(interval=None):.3f}",
            "ram": str(memory.rss),
            "gpu": "unavailable",
            "vram": "unavailable",
        }
    except Exception:
        return {"cpu": "unavailable", "ram": "unavailable", "gpu": "unavailable", "vram": "unavailable"}


def state_row_from_record(record: Mapping[str, Any]) -> dict[str, str]:
    row = record["manifest_row"]
    parse = record.get("parse", {})
    response_text = str(record.get("ollama_response", {}).get("response", "")) if isinstance(record.get("ollama_response"), Mapping) else ""
    status = str(record.get("status", ""))
    raw_path = str(record.get("raw_output_relative_path", ""))
    return {
        "request_id": row["request_id"],
        "run_id": str(record.get("run_id", "")),
        "status": status,
        "strict_valid": "true" if parse.get("strict_valid") else "false",
        "dataset": row["dataset"],
        "user_id": row["user_id"],
        "model_name": row["model_name"],
        "model_digest": row["model_digest"],
        "profile_generator_name": row["profile_generator_name"],
        "profile_generator_digest": row["profile_generator_digest"],
        "recommender_model_name": row["recommender_model_name"],
        "input_variant": row["input_variant"],
        "dependency_request_id": row["dependency_request_id"],
        "experiment_cell_id": row["experiment_cell_id"],
        "prompting_strategy": row["prompting_strategy"],
        "rendered_prompt_hash": str(record.get("rendered_prompt_hash", "")),
        "actual_raw_output_relative_path": raw_path,
        "started_at": str(record.get("started_at", "")),
        "finished_at": str(record.get("finished_at", "")),
        "wall_clock_seconds": f"{safe_float(record.get('wall_clock_seconds', 0.0)):.6f}",
        "retry_count": str(record.get("retry_count", 0)),
        "error": str(record.get("error", "")),
        "parser_status": str(parse.get("parser_status", "")),
        "parsed_count": str(len(parse.get("parsed_titles", [])) if isinstance(parse.get("parsed_titles", []), list) else 0),
        "valid_item_count": str(parse.get("valid_item_count", "")),
        "invalid_item_count": str(parse.get("invalid_item_count", "")),
        "hallucination_count": str(parse.get("hallucination_count", "")),
        "duplicate_count": str(parse.get("duplicate_count", "")),
        "hit_at_5": str(parse.get("hit_at_5", "")),
        "ndcg_at_5": f"{safe_float(parse.get('ndcg_at_5', 0.0)):.6f}" if parse.get("ndcg_at_5", "") != "" else "",
        "prompt_tokens": str(record.get("ollama_response", {}).get("prompt_eval_count", "")) if isinstance(record.get("ollama_response"), Mapping) else "",
        "completion_tokens": str(record.get("ollama_response", {}).get("eval_count", "")) if isinstance(record.get("ollama_response"), Mapping) else "",
        "total_duration_ns": str(record.get("ollama_response", {}).get("total_duration", "")) if isinstance(record.get("ollama_response"), Mapping) else "",
        "load_duration_ns": str(record.get("ollama_response", {}).get("load_duration", "")) if isinstance(record.get("ollama_response"), Mapping) else "",
        "prompt_eval_duration_ns": str(record.get("ollama_response", {}).get("prompt_eval_duration", "")) if isinstance(record.get("ollama_response"), Mapping) else "",
        "eval_duration_ns": str(record.get("ollama_response", {}).get("eval_duration", "")) if isinstance(record.get("ollama_response"), Mapping) else "",
        "raw_response_sha256": base.sha256_text(response_text) if response_text else "",
        "raw_response_bytes": str(len(response_text.encode("utf-8"))) if response_text else "0",
        "source_lineage": row["input_lineage"],
        "input_contract_hash": row["request_contract_hash"],
    }


def blocked_dependency_record(row: Mapping[str, str], run_id: str, raw_output_dir: Path) -> dict[str, Any]:
    now = utc_now()
    return {
        "run_id": run_id,
        "status": "blocked_dependency_not_strict_valid",
        "strict_valid": False,
        "manifest_row": dict(row),
        "rendered_prompt_hash": "",
        "raw_output_relative_path": "",
        "started_at": now,
        "finished_at": now,
        "wall_clock_seconds": 0.0,
        "retry_count": 0,
        "error": f"dependency_not_strict_valid:{row['dependency_request_id']}",
        "parse": parse_profile_response("") if row["input_variant"] == "profile_generation" else {
            "parser_status": "blocked_dependency_not_strict_valid",
            "strict_valid": False,
            "parsed_titles": [],
            "parsed_item_ids": [],
            "invalid_titles": [],
            "hit_at_5": 0,
            "ndcg_at_5": 0.0,
            "duplicate_count": 0,
            "hallucination_count": 0,
            "valid_item_count": 0,
            "invalid_item_count": 0,
        },
    }


def execute_one(row: Mapping[str, str], run_id: str, raw_output_dir: Path, parent_profiles: Mapping[str, str]) -> dict[str, Any]:
    prompt = render_prompt(row, parent_profiles)
    rendered_prompt_hash = base.sha256_text(prompt)
    source = source_row_for_manifest(row)
    generation_parameters = json.loads(row["generation_parameters_json"])
    candidate_ids, candidate_titles = candidate_ids_and_titles(source) if row["input_variant"] != "profile_generation" else ([], [])
    started = utc_now()
    start_perf = time.perf_counter()
    error = ""
    ollama_response: dict[str, Any] = {}
    try:
        ollama_response = ollama_generate(row["model_name"], prompt, generation_parameters)
        status = "completed"
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    wall_clock = time.perf_counter() - start_perf
    finished = utc_now()
    response_text = str(ollama_response.get("response", "")) if ollama_response else ""
    parse_start = time.perf_counter()
    if status == "completed" and row["input_variant"] == "profile_generation":
        parsed = parse_profile_response(response_text)
    elif status == "completed":
        parsed = parse_recommendation_response(response_text, candidate_ids, candidate_titles, row["positive_item_id"])
    else:
        parsed = {
            "parser_status": "not_parsed_failed_request",
            "strict_valid": False,
            "parsed_titles": [],
            "parsed_item_ids": [],
            "invalid_titles": [],
            "hit_at_5": 0 if row["input_variant"] != "profile_generation" else "",
            "ndcg_at_5": 0.0 if row["input_variant"] != "profile_generation" else "",
            "duplicate_count": 0,
            "hallucination_count": 0,
            "valid_item_count": 0,
            "invalid_item_count": 0,
        }
    parsing_time = time.perf_counter() - parse_start
    raw_path = raw_output_dir / f"{row['request_id']}.json"
    record = {
        "run_id": run_id,
        "phase_id": base.PHASE_ID,
        "phase_name": base.PHASE_NAME,
        "status": status,
        "manifest_row": dict(row),
        "rendered_prompt": prompt,
        "rendered_prompt_hash": rendered_prompt_hash,
        "generation_parameters": generation_parameters,
        "candidate_item_ids": candidate_ids,
        "candidate_item_titles": candidate_titles,
        "ollama_response": ollama_response,
        "raw_response": response_text,
        "started_at": started,
        "finished_at": finished,
        "wall_clock_seconds": wall_clock,
        "retry_count": 0,
        "error": error,
        "parse": {**parsed, "parsing_time_seconds": parsing_time},
        "strict_valid": bool(parsed.get("strict_valid")),
        "raw_output_relative_path": base.rel(raw_path),
        "evidence_classification": "paper_protocol_reproduction_different_open_model",
    }
    if raw_path.exists():
        raise FileExistsError(f"refusing to overwrite existing raw request file: {base.rel(raw_path)}")
    atomic_write_json(raw_path, record)
    append_jsonl(raw_output_dir / "raw_responses.jsonl", record)
    return record


def load_existing_records(raw_output_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in read_jsonl_records(raw_output_dir / "raw_responses.jsonl"):
        row = record.get("manifest_row", {})
        if isinstance(row, Mapping) and row.get("request_id"):
            records[str(row["request_id"])] = record
    for raw_path in raw_output_dir.glob("F6L-PROFILE-REQ-*.json"):
        payload = base.read_json_file(raw_path)
        row = payload.get("manifest_row", {})
        if isinstance(row, Mapping) and row.get("request_id") and str(row["request_id"]) not in records:
            records[str(row["request_id"])] = payload
    return records


def parse_rows_from_records(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    parse_rows: list[dict[str, str]] = []
    recommendation_rows: list[dict[str, str]] = []
    for record in records:
        row = record.get("manifest_row", {})
        parse = record.get("parse", {})
        if not isinstance(row, Mapping) or row.get("input_variant") == "profile_generation":
            continue
        if not isinstance(parse, Mapping):
            continue
        parsed_titles = parse.get("parsed_titles", [])
        parsed_ids = parse.get("parsed_item_ids", [])
        if not isinstance(parsed_titles, list):
            parsed_titles = []
        if not isinstance(parsed_ids, list):
            parsed_ids = []
        parse_rows.append(
            {
                "request_id": str(row["request_id"]),
                "run_id": str(record.get("run_id", "")),
                "dataset": str(row["dataset"]),
                "user_id": str(row["user_id"]),
                "model_name": str(row["model_name"]),
                "profile_generator_name": str(row["profile_generator_name"]),
                "recommender_model_name": str(row["recommender_model_name"]),
                "input_variant": str(row["input_variant"]),
                "dependency_request_id": str(row["dependency_request_id"]),
                "candidate_list_hash": str(row["candidate_list_hash"]),
                "positive_item_id": str(row["positive_item_id"]),
                "positive_candidate_position": str(row["positive_candidate_position"]),
                "parser_contract": str(row["parser_contract"]),
                "parsed_titles_json": base.canonical_json(parsed_titles),
                "parsed_item_ids_json": base.canonical_json(parsed_ids),
                "invalid_titles_json": base.canonical_json(parse.get("invalid_titles", [])),
                "parser_status": str(parse.get("parser_status", "")),
                "strict_valid": "true" if parse.get("strict_valid") else "false",
                "hit_at_5": str(parse.get("hit_at_5", "0")),
                "ndcg_at_5": f"{safe_float(parse.get('ndcg_at_5', 0.0)):.6f}",
                "duplicate_count": str(parse.get("duplicate_count", "0")),
                "hallucination_count": str(parse.get("hallucination_count", "0")),
                "parsing_time_seconds": f"{safe_float(parse.get('parsing_time_seconds', 0.0)):.6f}",
                "source_file": str(record.get("raw_output_relative_path", "")),
                "evidence_classification": "paper_protocol_reproduction_different_open_model",
            }
        )
        for rank, (item_id, title) in enumerate(zip(parsed_ids[:10], parsed_titles[:10]), start=1):
            recommendation_rows.append(
                {
                    "request_id": str(row["request_id"]),
                    "run_id": str(record.get("run_id", "")),
                    "dataset": str(row["dataset"]),
                    "user_id": str(row["user_id"]),
                    "model_name": str(row["model_name"]),
                    "profile_generator_name": str(row["profile_generator_name"]),
                    "recommender_model_name": str(row["recommender_model_name"]),
                    "input_variant": str(row["input_variant"]),
                    "rank": str(rank),
                    "recommended_item_id": str(item_id),
                    "recommended_item_title": str(title),
                    "match_status": "candidate_match" if item_id else "invalid_item",
                    "is_positive": "true" if item_id and item_id == row["positive_item_id"] else "false",
                    "source_file": str(record.get("raw_output_relative_path", "")),
                    "evidence_classification": "paper_protocol_reproduction_different_open_model",
                }
            )
    return parse_rows, recommendation_rows


def group_key(row: Mapping[str, str]) -> tuple[str, str, str, str]:
    profile_generator = row["profile_generator_name"] or "none"
    recommender = row["recommender_model_name"] or row["model_name"]
    return row["dataset"], profile_generator, recommender, row["input_variant"]


def metric_row(run_id: str, key: tuple[str, str, str, str], metric: str, value: str, numerator: str, denominator: str, status: str, source_file: str, notes: str) -> dict[str, str]:
    dataset, profile_generator, recommender, input_variant = key
    payload = [run_id, dataset, profile_generator, recommender, input_variant, metric]
    return {
        "metric_id": base.sha256_text(base.canonical_json(payload))[:16],
        "run_id": run_id,
        "dataset": dataset,
        "profile_generator": profile_generator,
        "recommender_model": recommender,
        "input_variant": input_variant,
        "metric": metric,
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "status": status,
        "evidence_classification": "paper_protocol_reproduction_different_open_model",
        "source_file": source_file,
        "notes": notes,
    }


def aggregate_profile_metrics(run_id: str, parse_rows: Sequence[Mapping[str, str]], recommendation_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in parse_rows:
        groups[group_key(row)].append(row)
    rec_by_key: dict[tuple[str, str, str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in recommendation_rows:
        rec_by_key[group_key(row)].append(row)
    rows: list[dict[str, str]] = []
    source_file = base.rel(PROFILE_EXECUTION_TABLES["profile_parse_results"])
    all_candidates_by_key: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    manifest_rows = base.read_csv_rows(base.PROFILE_REQUEST_MANIFEST_PATH)
    for row in manifest_rows:
        if row["input_variant"] == "profile_generation":
            continue
        try:
            source = source_row_for_manifest(row)
            candidate_ids, _ = candidate_ids_and_titles(source)
            all_candidates_by_key[group_key(row)].update(candidate_ids)
        except Exception:
            pass

    for key in sorted(groups):
        items = groups[key]
        denominator = len(items)
        strict_count = sum(1 for row in items if row["strict_valid"] == "true")
        hit_sum = sum(int(row["hit_at_5"] or 0) for row in items)
        ndcg_sum = sum(safe_float(row["ndcg_at_5"]) for row in items)
        hallucination_sum = sum(int(row["hallucination_count"] or 0) for row in items)
        duplicate_sum = sum(int(row["duplicate_count"] or 0) for row in items)
        parsed_count = sum(len(json.loads(row["parsed_titles_json"])) for row in items)
        valid_recs = [row for row in rec_by_key.get(key, []) if row["match_status"] == "candidate_match"]
        unique_recommended = {row["recommended_item_id"] for row in valid_recs if row["recommended_item_id"]}
        candidate_universe = all_candidates_by_key.get(key, set())

        rows.extend(
            [
                metric_row(run_id, key, "HR@5", f"{hit_sum / denominator:.6f}" if denominator else "", str(hit_sum), str(denominator), "computed", source_file, "Invalid or failed parses contribute zero hits."),
                metric_row(run_id, key, "NDCG@5", f"{ndcg_sum / denominator:.6f}" if denominator else "", f"{ndcg_sum:.6f}", str(denominator), "computed", source_file, "Invalid or failed parses contribute zero gain."),
                metric_row(run_id, key, "validity_rate", f"{strict_count / denominator:.6f}" if denominator else "", str(strict_count), str(denominator), "computed", source_file, "Strict validity requires exactly 10 candidate titles and no duplicates."),
                metric_row(run_id, key, "hallucination_rate", f"{hallucination_sum / parsed_count:.6f}" if parsed_count else "", str(hallucination_sum), str(parsed_count), "computed", source_file, "Hallucination counts parsed titles not matched to the candidate list."),
                metric_row(run_id, key, "duplicate_rate", f"{duplicate_sum / parsed_count:.6f}" if parsed_count else "", str(duplicate_sum), str(parsed_count), "computed", source_file, "Duplicate rate is duplicate matched items divided by parsed titles."),
                metric_row(run_id, key, "coverage", f"{len(unique_recommended) / len(candidate_universe):.6f}" if candidate_universe else "", str(len(unique_recommended)), str(len(candidate_universe)), "computed", source_file, "Coverage is unique valid recommended candidate IDs over the group candidate universe."),
                metric_row(run_id, key, "APLT", "", "", "", "unsupported_missing_train_popularity", source_file, "Train-set item popularity is not frozen in the profile execution manifest."),
                metric_row(run_id, key, "popularity_bias", "", "", "", "unsupported_missing_train_popularity", source_file, "Popularity bias needs a paper-faithful train popularity table."),
                metric_row(run_id, key, "ILD_diversity", "", "", "", "unsupported_missing_item_similarity_or_category", source_file, "ILD needs item similarity/category metadata not present in the profile request contract."),
                metric_row(run_id, key, "calibration", "", "", "", "unsupported_missing_user_item_attribute_distribution", source_file, "Calibration needs comparable user preference and item attribute distributions."),
            ]
        )
    return rows


def comparable_key(row: Mapping[str, str]) -> tuple[str, str, str, str]:
    return row["dataset"], row["user_id"], row["recommender_model_name"] or row["model_name"], row["candidate_list_hash"]


def profile_comparable_key(row: Mapping[str, str]) -> tuple[str, str, str, str, str]:
    return (*comparable_key(row), row["profile_generator_name"] or "none")


def aggregate_metric_differences(run_id: str, parse_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    by_history: dict[tuple[str, str, str, str], Mapping[str, str]] = {}
    by_profile_variant: dict[str, dict[tuple[str, str, str, str, str], Mapping[str, str]]] = defaultdict(dict)
    for row in parse_rows:
        if row["input_variant"] == "history_only":
            by_history[comparable_key(row)] = row
        elif row["input_variant"] in {"profile_only", "profile_plus_history"}:
            by_profile_variant[row["input_variant"]][profile_comparable_key(row)] = row
    rows: list[dict[str, str]] = []
    comparisons = [
        ("profile_only", "history_only", "profile_only_vs_history_only"),
        ("profile_plus_history", "history_only", "profile_plus_history_vs_history_only"),
        ("profile_plus_history", "profile_only", "profile_plus_history_vs_profile_only"),
    ]
    source_file = base.rel(PROFILE_EXECUTION_TABLES["profile_parse_results"])
    for treatment, baseline, label in comparisons:
        for treatment_key, treatment_row in sorted(by_profile_variant.get(treatment, {}).items()):
            if baseline == "history_only":
                baseline_row = by_history.get(treatment_key[:4])
            else:
                baseline_row = by_profile_variant.get(baseline, {}).get(treatment_key)
            if baseline_row is None:
                continue
            dataset, _, recommender, _, _ = treatment_key
            profile_generator = treatment_row["profile_generator_name"] or "none"
            for metric in ["HR@5", "NDCG@5"]:
                treatment_value = safe_float(treatment_row["hit_at_5"] if metric == "HR@5" else treatment_row["ndcg_at_5"])
                baseline_value = safe_float(baseline_row["hit_at_5"] if metric == "HR@5" else baseline_row["ndcg_at_5"])
                diff = treatment_value - baseline_value
                rel_diff = "" if baseline_value == 0 else f"{diff / baseline_value:.6f}"
                rows.append(
                    {
                        "comparison_id": base.sha256_text(base.canonical_json([run_id, dataset, profile_generator, recommender, label, metric, treatment_key]))[:16],
                        "run_id": run_id,
                        "dataset": dataset,
                        "profile_generator": profile_generator,
                        "recommender_model": recommender,
                        "comparison": label,
                        "metric": metric,
                        "baseline_value": f"{baseline_value:.6f}",
                        "treatment_value": f"{treatment_value:.6f}",
                        "absolute_difference": f"{diff:.6f}",
                        "relative_difference": rel_diff,
                        "paired_users": "1",
                        "status": "computed_pair",
                        "evidence_classification": "paper_protocol_reproduction_different_open_model",
                        "source_file": source_file,
                        "notes": "Pair matched on dataset, user, recommender, candidate-list hash, profile generator where applicable, decoding contract, parser, and cutoff.",
                    }
                )
    return collapse_difference_rows(rows, run_id)


def collapse_difference_rows(rows: Sequence[Mapping[str, str]], run_id: str) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["profile_generator"], row["recommender_model"], row["comparison"], row["metric"])].append(row)
    collapsed: list[dict[str, str]] = []
    for key in sorted(grouped):
        dataset, profile_generator, recommender, comparison, metric = key
        items = grouped[key]
        baseline_values = [safe_float(row["baseline_value"]) for row in items]
        treatment_values = [safe_float(row["treatment_value"]) for row in items]
        baseline_value = statistics.mean(baseline_values) if baseline_values else 0.0
        treatment_value = statistics.mean(treatment_values) if treatment_values else 0.0
        diff = treatment_value - baseline_value
        collapsed.append(
            {
                "comparison_id": base.sha256_text(base.canonical_json([run_id, key]))[:16],
                "run_id": run_id,
                "dataset": dataset,
                "profile_generator": profile_generator,
                "recommender_model": recommender,
                "comparison": comparison,
                "metric": metric,
                "baseline_value": f"{baseline_value:.6f}",
                "treatment_value": f"{treatment_value:.6f}",
                "absolute_difference": f"{diff:.6f}",
                "relative_difference": "" if baseline_value == 0 else f"{diff / baseline_value:.6f}",
                "paired_users": str(len(items)),
                "status": "computed",
                "evidence_classification": "paper_protocol_reproduction_different_open_model",
                "source_file": base.rel(PROFILE_EXECUTION_TABLES["profile_parse_results"]),
                "notes": "Aggregated only after disaggregating by dataset, profile generator, recommender model, comparison, and metric.",
            }
        )
    return collapsed


def runtime_rows(run_id: str, state_rows: Sequence[Mapping[str, str]], start_time: float, parsing_time: float, consolidation_time: float, resource_start: Mapping[str, str], resource_end: Mapping[str, str], raw_output_dir: Path) -> list[dict[str, str]]:
    latencies = [safe_float(row["wall_clock_seconds"]) for row in state_rows if row["status"] == "completed"]
    completed = sum(1 for row in state_rows if row["status"] == "completed")
    strict = sum(1 for row in state_rows if row["strict_valid"] == "true")
    failed = sum(1 for row in state_rows if row["status"] == "failed")
    blocked = sum(1 for row in state_rows if row["status"].startswith("blocked"))
    planned = 4800
    residual = planned - completed - failed - blocked
    wall_clock = time.perf_counter() - start_time
    prompt_tokens = sum(int(row["prompt_tokens"] or 0) for row in state_rows if str(row["prompt_tokens"] or "").isdigit())
    completion_tokens = sum(int(row["completion_tokens"] or 0) for row in state_rows if str(row["completion_tokens"] or "").isdigit())
    raw_bytes = sum(int(row["raw_response_bytes"] or 0) for row in state_rows if str(row["raw_response_bytes"] or "").isdigit())
    return [
        {
            "runtime_id": f"F6L-PROFILE-EXEC-{base.sha256_text(run_id)[:12]}",
            "run_id": run_id,
            "scope": "profile_open_model_execution",
            "model": "qwen2:7b-instruct;mistral:7b-instruct",
            "request_count": str(planned),
            "completed_requests": str(completed),
            "strict_valid_requests": str(strict),
            "failed_requests": str(failed),
            "residual_requests": str(max(0, residual)),
            "retry_count": "0",
            "wall_clock_seconds": f"{wall_clock:.6f}",
            "latency_mean": mean(latencies),
            "latency_median": median(latencies),
            "latency_p90": pct(latencies, 90),
            "latency_p95": pct(latencies, 95),
            "latency_p99": pct(latencies, 99),
            "throughput_requests_per_hour": f"{completed / wall_clock * 3600:.6f}" if wall_clock > 0 else "",
            "prompt_tokens": str(prompt_tokens) if prompt_tokens else "",
            "completion_tokens": str(completion_tokens) if completion_tokens else "",
            "raw_output_bytes": str(raw_bytes),
            "parsing_time_seconds": f"{parsing_time:.6f}",
            "consolidation_time_seconds": f"{consolidation_time:.6f}",
            "cpu": f"start={resource_start.get('cpu')};end={resource_end.get('cpu')}",
            "ram": f"start={resource_start.get('ram')};end={resource_end.get('ram')}",
            "gpu": "unavailable",
            "vram": "unavailable",
            "marginal_api_cost": "0",
            "hardware_cost": "not_monetized",
            "energy_cost": "unavailable",
            "total_cost_claim": "not_claimed_zero_hardware_energy_not_monetized",
            "source_file": base.rel(raw_output_dir / "raw_responses.jsonl"),
            "notes": "Local Ollama execution only; marginal paid API cost is zero but total cost is not claimed as zero.",
        }
    ]


def profile_generation_result_rows(metrics: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in metrics:
        rows.append(
            {
                "result_id": row["metric_id"],
                "dataset": row["dataset"],
                "profile_generator": "" if row["profile_generator"] == "none" else row["profile_generator"],
                "recommender_model": row["recommender_model"],
                "mode": row["input_variant"],
                "evidence_classification": row["evidence_classification"],
                "request_count": row["denominator"],
                "metric": row["metric"],
                "value": row["value"],
                "status": row["status"],
                "source_file": row["source_file"],
                "notes": row["notes"],
            }
        )
    return rows


def update_runtime_cost_results(profile_runtime: Sequence[Mapping[str, str]]) -> None:
    existing = [
        row
        for row in base.read_csv_rows(base.OUTPUT_TABLES["runtime_cost_results"])
        if not row.get("runtime_id", "").startswith("F6L-PROFILE-EXEC-")
    ]
    mapped: list[dict[str, str]] = []
    for row in profile_runtime:
        mapped.append(
            {
                "runtime_id": row["runtime_id"],
                "scope": row["scope"],
                "model": row["model"],
                "task": "profile_generation_and_recommendation",
                "request_count": row["request_count"],
                "observed_or_estimated": "observed_profile_execution",
                "wall_clock_seconds": row["wall_clock_seconds"],
                "latency_mean": row["latency_mean"],
                "latency_p50": row["latency_median"],
                "latency_p95": row["latency_p95"],
                "throughput_requests_per_hour": row["throughput_requests_per_hour"],
                "successes": row["completed_requests"],
                "failures": row["failed_requests"],
                "retries": row["retry_count"],
                "cpu": row["cpu"],
                "ram": row["ram"],
                "gpu": row["gpu"],
                "vram": row["vram"],
                "input_tokens": row["prompt_tokens"],
                "output_tokens": row["completion_tokens"],
                "raw_output_bytes": row["raw_output_bytes"],
                "parsing_time_seconds": row["parsing_time_seconds"],
                "consolidation_time_seconds": row["consolidation_time_seconds"],
                "marginal_api_cost": row["marginal_api_cost"],
                "total_cost_claim": row["total_cost_claim"],
                "evidence_classification": "paper_protocol_reproduction_different_open_model",
                "source_file": row["source_file"],
                "notes": row["notes"],
            }
        )
    base.write_csv(base.OUTPUT_TABLES["runtime_cost_results"], existing + mapped, base.TABLE_FIELDS["runtime_cost_results"])


def summarize_difference(diffs: Sequence[Mapping[str, str]], dataset: str = "", model_prefix: str = "", profile_variant: str = "") -> str:
    selected = []
    for row in diffs:
        if dataset and row["dataset"] != dataset:
            continue
        if model_prefix and not row["recommender_model"].startswith(model_prefix):
            continue
        if profile_variant and not row["comparison"].startswith(profile_variant):
            continue
        if row["metric"] == "HR@5":
            selected.append(float(row["absolute_difference"]))
    if not selected:
        return "unavailable"
    return f"HR@5 deltas mean={statistics.mean(selected):.6f};min={min(selected):.6f};max={max(selected):.6f};n={len(selected)}"


def determine_profile_status(diffs: Sequence[Mapping[str, str]]) -> tuple[str, str, str]:
    profile_diffs = [float(row["absolute_difference"]) for row in diffs if row["metric"] == "HR@5" and row["comparison"] in {"profile_only_vs_history_only", "profile_plus_history_vs_history_only"}]
    if not profile_diffs:
        return "unresolved_for_original_closed_model", "unavailable", "unavailable"
    positive = [value for value in profile_diffs if value > 0]
    negative = [value for value in profile_diffs if value < 0]
    by_dataset = defaultdict(list)
    by_model = defaultdict(list)
    for row in diffs:
        if row["metric"] == "HR@5":
            by_dataset[row["dataset"]].append(float(row["absolute_difference"]))
            by_model[row["recommender_model"]].append(float(row["absolute_difference"]))
    dataset_signs = {dataset: statistics.mean(values) > 0 for dataset, values in by_dataset.items() if values}
    model_signs = {model: statistics.mean(values) > 0 for model, values in by_model.items() if values}
    if positive and negative:
        if len(set(dataset_signs.values())) > 1:
            return "mixed_dataset_dependent", str(dataset_signs), str(model_signs)
        return "mixed_model_dependent", str(dataset_signs), str(model_signs)
    if positive:
        return "supported_with_open_model_protocol_evidence", str(dataset_signs), str(model_signs)
    return "contradicted_with_open_model_protocol_evidence", str(dataset_signs), str(model_signs)


def build_rq_rows(metrics: Sequence[Mapping[str, str]], diffs: Sequence[Mapping[str, str]], runtime: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    status, dataset_consistency, model_consistency = determine_profile_status(diffs)
    qwen = summarize_difference(diffs, model_prefix="qwen2")
    mistral = summarize_difference(diffs, model_prefix="mistral")
    beauty = summarize_difference(diffs, dataset="beauty")
    sports = summarize_difference(diffs, dataset="sports")
    profile_only = summarize_difference(diffs, profile_variant="profile_only")
    profile_plus = summarize_difference(diffs, profile_variant="profile_plus_history")
    source = f"{base.rel(PROFILE_EXECUTION_TABLES['profile_metrics'])};{base.rel(PROFILE_EXECUTION_TABLES['profile_metric_differences'])}"
    hallucination_values = [
        float(row["value"])
        for row in metrics
        if row["metric"] == "hallucination_rate" and row["status"] == "computed" and row["value"] != ""
    ]
    hallucination_status = (
        "supported_with_open_model_protocol_evidence"
        if hallucination_values and max(hallucination_values) <= 0.05
        else "mixed_model_dependent"
        if hallucination_values
        else "unresolved_for_original_closed_model"
    )
    return [
        {
            "rq_id": "RQ6",
            "previous_status": "mixed_model_dependent",
            "paper_claim": "Short histories and alternative user evidence can affect recommendation.",
            "beauty_evidence": beauty,
            "sports_evidence": sports,
            "qwen2_evidence": qwen,
            "mistral_evidence": mistral,
            "profile_only": profile_only,
            "profile_plus_history": profile_plus,
            "dataset_consistency": dataset_consistency,
            "model_consistency": model_consistency,
            "new_status": status,
            "motivation": "Profile-family open-model cells are compared against homogeneous history-only controls.",
            "limits": "This updates only Beauty/Sports local open models and does not establish closed-model transfer.",
            "source_file": source,
        },
        {
            "rq_id": "RQ8",
            "previous_status": "unresolved_missing_original_artifact",
            "paper_claim": "LLM-generated profiles capture useful user patterns.",
            "beauty_evidence": beauty,
            "sports_evidence": sports,
            "qwen2_evidence": qwen,
            "mistral_evidence": mistral,
            "profile_only": profile_only,
            "profile_plus_history": profile_plus,
            "dataset_consistency": dataset_consistency,
            "model_consistency": model_consistency,
            "new_status": status,
            "motivation": "New Qwen2/Mistral profiles and matched profile recommendation prompts were executed under the frozen paper-title-list contract.",
            "limits": "Evidence class is paper_protocol_reproduction_different_open_model, not exact GPT-4o-mini or closed-model reproduction.",
            "source_file": source,
        },
        {
            "rq_id": "RQ9",
            "previous_status": "supported_only_in_adapted_extension",
            "paper_claim": "Hallucination is generally low except some models.",
            "beauty_evidence": "computed from title-list parser",
            "sports_evidence": "computed from title-list parser",
            "qwen2_evidence": "hallucination rates disaggregated in profile metrics",
            "mistral_evidence": "hallucination rates disaggregated in profile metrics",
            "profile_only": "included",
            "profile_plus_history": "included",
            "dataset_consistency": "see disaggregated metric rows",
            "model_consistency": "see disaggregated metric rows",
            "new_status": hallucination_status,
            "motivation": "The profile execution uses the paper title-list candidate parser rather than adapted JSON/index outputs.",
            "limits": "No claim is made for closed/paid models or LastFM cells.",
            "source_file": base.rel(PROFILE_EXECUTION_TABLES["profile_metrics"]),
        },
        {
            "rq_id": "RQ14",
            "previous_status": "supported_only_in_adapted_extension",
            "paper_claim": "Runtime and costs need current characterization.",
            "beauty_evidence": "included in profile runtime totals",
            "sports_evidence": "included in profile runtime totals",
            "qwen2_evidence": "local Ollama token/latency telemetry when exposed",
            "mistral_evidence": "local Ollama token/latency telemetry when exposed",
            "profile_only": "included",
            "profile_plus_history": "included",
            "dataset_consistency": "not_applicable_runtime_characterization",
            "model_consistency": "not_applicable_runtime_characterization",
            "new_status": "supported_with_open_model_protocol_evidence" if runtime else "unresolved_for_original_closed_model",
            "motivation": "Observed local profile-execution telemetry replaces checkpoint estimates for this subset.",
            "limits": "Marginal paid API cost is zero; hardware and energy are not monetized, so total cost is not claimed as zero.",
            "source_file": base.rel(PROFILE_EXECUTION_TABLES["profile_runtime_resources"]),
        },
    ]


def update_rq_revalidation_table(profile_rq_rows: Sequence[Mapping[str, str]]) -> None:
    profile_by_id = {row["rq_id"]: row for row in profile_rq_rows}
    existing = base.read_csv_rows(base.OUTPUT_TABLES["rq_revalidation"])
    updated: list[dict[str, str]] = []
    for row in existing:
        profile = profile_by_id.get(row["rq_id"])
        if profile is None:
            updated.append(row)
            continue
        if row["rq_id"] in {"RQ6", "RQ8"}:
            evidence = f"Beauty: {profile['beauty_evidence']}; Sports: {profile['sports_evidence']}; Qwen2: {profile['qwen2_evidence']}; Mistral: {profile['mistral_evidence']}"
            row = {
                **row,
                "paper_protocol_new_output_evidence": evidence,
                "models_covered": "Qwen2;Mistral",
                "datasets_covered": "Beauty;Sports",
                "limits": profile["limits"],
                "new_status": profile["new_status"],
                "variation_reason": profile["motivation"],
            }
        elif row["rq_id"] == "RQ9":
            row = {
                **row,
                "paper_protocol_new_output_evidence": "Profile title-list validity and hallucination rates computed for Beauty/Sports Qwen2/Mistral.",
                "models_covered": "Qwen2;Mistral",
                "datasets_covered": "Beauty;Sports",
                "limits": profile["limits"],
                "new_status": profile["new_status"],
                "variation_reason": profile["motivation"],
            }
        elif row["rq_id"] == "RQ14":
            row = {
                **row,
                "paper_protocol_new_output_evidence": "Observed profile execution runtime and local marginal API cost recorded.",
                "models_covered": "Qwen2;Mistral",
                "datasets_covered": "Beauty;Sports",
                "limits": profile["limits"],
                "new_status": profile["new_status"],
                "variation_reason": profile["motivation"],
            }
        updated.append(row)
    base.write_csv(base.OUTPUT_TABLES["rq_revalidation"], updated, base.TABLE_FIELDS["rq_revalidation"])


def build_profile_report_section(manifest: Mapping[str, Any], runtime: Sequence[Mapping[str, str]], rq_rows: Sequence[Mapping[str, str]], metrics: Sequence[Mapping[str, str]], diffs: Sequence[Mapping[str, str]], preflight_rows: Sequence[Mapping[str, str]]) -> str:
    runtime_row = runtime[0] if runtime else {}
    failures = [row for row in preflight_rows if row["status"] != "pass"]
    beauty = summarize_difference(diffs, dataset="beauty")
    sports = summarize_difference(diffs, dataset="sports")
    qwen = summarize_difference(diffs, model_prefix="qwen2")
    mistral = summarize_difference(diffs, model_prefix="mistral")
    return "\n".join(
        [
            "## F6L Profile Open-Model Execution",
            "",
            f"- Decision token: `{manifest.get('decision_token', '')}`.",
            f"- Run ID: `{manifest.get('run_id', '')}`.",
            f"- Raw output directory: `{manifest.get('raw_output_dir', '')}`.",
            f"- Preflight status: `{'pass' if not failures else 'fail'}`.",
            f"- Planned/completed/strict-valid/failed/residual requests: `{manifest.get('planned_requests', 0)}`/`{manifest.get('completed_requests', 0)}`/`{manifest.get('strict_valid_requests', 0)}`/`{manifest.get('failed_requests', 0)}`/`{manifest.get('residual_requests', 0)}`.",
            f"- Retry count: `{manifest.get('retry_count', 0)}`.",
            f"- Beauty profile deltas: {beauty}.",
            f"- Sports profile deltas: {sports}.",
            f"- Qwen2 profile deltas: {qwen}.",
            f"- Mistral profile deltas: {mistral}.",
            f"- Runtime wall-clock seconds: `{runtime_row.get('wall_clock_seconds', '')}`; throughput requests/hour: `{runtime_row.get('throughput_requests_per_hour', '')}`.",
            f"- Runtime resources: CPU `{runtime_row.get('cpu', '')}`, RAM `{runtime_row.get('ram', '')}`, GPU `{runtime_row.get('gpu', '')}`, VRAM `{runtime_row.get('vram', '')}`.",
            "- Evidence classification: `paper_protocol_reproduction_different_open_model`.",
            "- Closed/paid model limitation: no exact GPT-4o-mini/GPT/Claude/GLM evidence is claimed; no remote API was called.",
            f"- Profile RQ rows updated: `{';'.join(row['rq_id'] for row in rq_rows) if rq_rows else 'none'}`.",
            "",
            f"`{manifest.get('decision_token', '')}`",
            "",
        ]
    )


def update_profile_report_handoff_notes(
    manifest: Mapping[str, Any],
    runtime: Sequence[Mapping[str, str]],
    rq_rows: Sequence[Mapping[str, str]],
    metrics: Sequence[Mapping[str, str]],
    diffs: Sequence[Mapping[str, str]],
    parse_rows: Sequence[Mapping[str, str]],
    recommendation_rows: Sequence[Mapping[str, str]],
    preflight_rows: Sequence[Mapping[str, str]],
) -> None:
    marker = f"{base.PHASE_NAME}_profile_open_model_execution"
    base.update_marked_section(base.REPORT_PATH, marker, build_profile_report_section(manifest, runtime, rq_rows, metrics, diffs, preflight_rows))
    handoff = "\n".join(
        [
            "## F6L Profile Open-Model Execution",
            "",
            f"- Decision token: `{manifest.get('decision_token', '')}`.",
            f"- Run ID: `{manifest.get('run_id', '')}`.",
            f"- Raw output directory: `{manifest.get('raw_output_dir', '')}`.",
            f"- Planned/completed/strict-valid/failed/residual: `{manifest.get('planned_requests', 0)}`/`{manifest.get('completed_requests', 0)}`/`{manifest.get('strict_valid_requests', 0)}`/`{manifest.get('failed_requests', 0)}`/`{manifest.get('residual_requests', 0)}`.",
            "- LastFM, remote API, paid models, downloads, training, staging, and original_repo mutation remain out of scope.",
        ]
    )
    notes = "\n".join(
        [
            "## 2026-08-01 - Phase F6L Profile Open-Model Execution",
            "",
            "### Changes",
            "",
            "- Executed or preflight-blocked the 4800-row profile manifest under the existing F6L phase.",
            "- Wrote profile execution state, parse rows, recommendation rows, metrics, differences, runtime resources, and profile RQ revalidation artifacts when execution reached consolidation.",
            "",
            "### Result",
            "",
            f"- Decision: `{manifest.get('decision_token', '')}`.",
            f"- Planned/completed/strict-valid/failed/residual: `{manifest.get('planned_requests', 0)}`/`{manifest.get('completed_requests', 0)}`/`{manifest.get('strict_valid_requests', 0)}`/`{manifest.get('failed_requests', 0)}`/`{manifest.get('residual_requests', 0)}`.",
            f"- Raw output directory: `{manifest.get('raw_output_dir', '')}`.",
        ]
    )
    base.update_marked_section(base.HANDOFF_PATH, marker, handoff)
    base.update_marked_section(base.NOTES_PATH, marker, notes)


def write_run_manifest(raw_output_dir: Path, manifest: Mapping[str, Any]) -> None:
    base.write_json(raw_output_dir / "profile_execution_manifest.json", manifest)
    base.write_json(PROFILE_EXECUTION_MANIFEST_PATH, manifest)


def consolidate_records(
    run_id: str,
    raw_output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    start_time: float,
    resource_start: Mapping[str, str],
    preflight_rows: Sequence[Mapping[str, str]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    consolidation_start = time.perf_counter()
    state_rows = [state_row_from_record(record) for record in records]
    parsing_time = sum(safe_float(record.get("parse", {}).get("parsing_time_seconds", 0.0)) for record in records if isinstance(record.get("parse"), Mapping))
    parse_rows, recommendation_rows = parse_rows_from_records(records)
    metrics = aggregate_profile_metrics(run_id, parse_rows, recommendation_rows)
    diffs = aggregate_metric_differences(run_id, parse_rows)
    resource_end = resource_snapshot()
    consolidation_time = time.perf_counter() - consolidation_start
    runtime = runtime_rows(run_id, state_rows, start_time, parsing_time, consolidation_time, resource_start, resource_end, raw_output_dir)
    rq_rows = build_rq_rows(metrics, diffs, runtime)

    base.write_csv(PROFILE_EXECUTION_TABLES["profile_execution_state"], state_rows, PROFILE_EXECUTION_FIELDS["profile_execution_state"])
    base.write_csv(PROFILE_EXECUTION_TABLES["profile_parse_results"], parse_rows, PROFILE_EXECUTION_FIELDS["profile_parse_results"])
    base.write_csv(PROFILE_EXECUTION_TABLES["profile_recommendation_rows"], recommendation_rows, PROFILE_EXECUTION_FIELDS["profile_recommendation_rows"])
    base.write_csv(PROFILE_EXECUTION_TABLES["profile_metrics"], metrics, PROFILE_EXECUTION_FIELDS["profile_metrics"])
    base.write_csv(PROFILE_EXECUTION_TABLES["profile_metric_differences"], diffs, PROFILE_EXECUTION_FIELDS["profile_metric_differences"])
    base.write_csv(PROFILE_EXECUTION_TABLES["profile_runtime_resources"], runtime, PROFILE_EXECUTION_FIELDS["profile_runtime_resources"])
    base.write_csv(PROFILE_EXECUTION_TABLES["profile_rq_revalidation"], rq_rows, PROFILE_EXECUTION_FIELDS["profile_rq_revalidation"])
    base.write_csv(base.OUTPUT_TABLES["profile_generation_results"], profile_generation_result_rows(metrics), base.TABLE_FIELDS["profile_generation_results"])
    update_runtime_cost_results(runtime)
    update_rq_revalidation_table(rq_rows)

    completed = sum(1 for row in state_rows if row["status"] == "completed")
    strict = sum(1 for row in state_rows if row["strict_valid"] == "true")
    failed = sum(1 for row in state_rows if row["status"] == "failed")
    blocked = sum(1 for row in state_rows if row["status"].startswith("blocked"))
    planned = int(manifest.get("planned_requests", 4800))
    residual = max(0, planned - completed - failed - blocked)
    decision = DECISION_PROFILE_EXECUTION_COMPLETE if completed + failed + blocked == planned else DECISION_PROFILE_EXECUTION_PARTIAL
    output_hashes = {
        base.rel(path): base.safe_file_sha256(path)
        for path in PROFILE_EXECUTION_TABLES.values()
        if path.exists()
    }
    output_hashes[base.rel(base.OUTPUT_TABLES["profile_generation_results"])] = base.safe_file_sha256(base.OUTPUT_TABLES["profile_generation_results"])
    output_hashes[base.rel(base.OUTPUT_TABLES["runtime_cost_results"])] = base.safe_file_sha256(base.OUTPUT_TABLES["runtime_cost_results"])
    output_hashes[base.rel(base.OUTPUT_TABLES["rq_revalidation"])] = base.safe_file_sha256(base.OUTPUT_TABLES["rq_revalidation"])
    final_manifest = {
        **manifest,
        "decision_token": decision,
        "run_id": run_id,
        "planned_requests": planned,
        "completed_requests": completed,
        "strict_valid_requests": strict,
        "failed_requests": failed,
        "blocked_dependency_requests": blocked,
        "residual_requests": residual,
        "retry_count": 0,
        "inference_performed": completed > 0 or failed > 0,
        "remote_api_calls_performed": False,
        "lastfm_requests_performed": False,
        "downloads_performed": False,
        "training_performed": False,
        "tables": {name: base.rel(path) for name, path in PROFILE_EXECUTION_TABLES.items()},
        "updated_tables": {
            "profile_generation_results": base.rel(base.OUTPUT_TABLES["profile_generation_results"]),
            "runtime_cost_results": base.rel(base.OUTPUT_TABLES["runtime_cost_results"]),
            "rq_revalidation": base.rel(base.OUTPUT_TABLES["rq_revalidation"]),
        },
        "output_hashes": output_hashes,
        "raw_jsonl_sha256": base.safe_file_sha256(raw_output_dir / "raw_responses.jsonl"),
        "raw_output_bytes": runtime[0]["raw_output_bytes"] if runtime else "0",
        "finalized_at": utc_now(),
    }
    write_run_manifest(raw_output_dir, final_manifest)
    update_profile_report_handoff_notes(final_manifest, runtime, rq_rows, metrics, diffs, parse_rows, recommendation_rows, preflight_rows)
    return final_manifest


def execute_profile_mode(args: Any) -> int:
    missing = []
    if not args.allow_api_calls:
        missing.append("--allow-api-calls")
    if not args.local_only:
        missing.append("--local-only")
    if not args.checkpoint_id:
        missing.append("--checkpoint-id")
    if not args.raw_output_dir:
        missing.append("--raw-output-dir")
    if missing:
        print(json.dumps({"phase_id": base.PHASE_ID, "mode": args.mode, "decision": "blocked_missing_execution_flags", "missing": missing}, sort_keys=True))
        return 2

    preflight_rows, _, manifest = build_profile_preflight(args)
    if any(row["status"] != "pass" for row in preflight_rows):
        write_blocked_preflight(preflight_rows, manifest)
        print(json.dumps({**manifest, "decision_token": DECISION_PROFILE_EXECUTION_BLOCKED}, sort_keys=True))
        return 1

    run_id = Path(args.raw_output_dir).name.replace("phase_f6l_profile_execution_", "")
    raw_output_dir = base.resolve_project_path(args.raw_output_dir)
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    base.write_csv(PROFILE_EXECUTION_TABLES["profile_execution_preflight"], preflight_rows, PROFILE_EXECUTION_FIELDS["profile_execution_preflight"])
    rows = sorted(base.read_csv_rows(base.PROFILE_REQUEST_MANIFEST_PATH), key=request_order_key)
    request_ids = [row["request_id"] for row in rows]
    start_time = time.perf_counter()
    resource_start = resource_snapshot()
    run_manifest = {
        **manifest,
        "decision_token": DECISION_PROFILE_EXECUTION_PARTIAL,
        "run_id": run_id,
        "request_ids": request_ids,
        "inference_performed": False,
        "started_at": utc_now(),
    }
    write_run_manifest(raw_output_dir, run_manifest)

    existing_by_id = load_existing_records(raw_output_dir)
    records: dict[str, dict[str, Any]] = dict(existing_by_id)
    parent_profiles: dict[str, str] = {
        request_id: str(record.get("raw_response", ""))
        for request_id, record in records.items()
        if record.get("manifest_row", {}).get("input_variant") == "profile_generation" and record.get("strict_valid")
    }
    checkpoint_dir = raw_output_dir / "checkpoints"

    for index, row in enumerate(rows, start=1):
        request_id = row["request_id"]
        prior = records.get(request_id)
        if prior and (prior.get("strict_valid") or prior.get("status") in {"completed", "failed", "blocked_dependency_not_strict_valid"}):
            if row["input_variant"] == "profile_generation" and prior.get("strict_valid"):
                parent_profiles[request_id] = str(prior.get("raw_response", ""))
            continue
        dependency = row.get("dependency_request_id", "")
        if dependency and dependency not in parent_profiles:
            record = blocked_dependency_record(row, run_id, raw_output_dir)
            records[request_id] = record
            append_jsonl(raw_output_dir / "raw_responses.jsonl", record)
            continue
        try:
            record = execute_one(row, run_id, raw_output_dir, parent_profiles)
        except FileExistsError as exc:
            record = {
                "run_id": run_id,
                "status": "failed",
                "strict_valid": False,
                "manifest_row": dict(row),
                "rendered_prompt_hash": "",
                "raw_output_relative_path": "",
                "started_at": utc_now(),
                "finished_at": utc_now(),
                "wall_clock_seconds": 0.0,
                "retry_count": 0,
                "error": f"{type(exc).__name__}: {exc}",
                "parse": {"parser_status": "raw_overwrite_refused", "strict_valid": False},
            }
            append_jsonl(raw_output_dir / "raw_responses.jsonl", record)
        records[request_id] = record
        if row["input_variant"] == "profile_generation" and record.get("strict_valid"):
            parent_profiles[request_id] = str(record.get("raw_response", ""))
        if index % 50 == 0:
            checkpoint_payload = {
                **run_manifest,
                "checkpoint_index": index,
                "completed_records": len(records),
                "strict_valid_records": sum(1 for item in records.values() if item.get("strict_valid")),
                "updated_at": utc_now(),
            }
            atomic_write_json(checkpoint_dir / f"checkpoint_{index:04d}.json", checkpoint_payload)

    final_manifest = consolidate_records(
        run_id,
        raw_output_dir,
        [records[request_id] for request_id in request_ids if request_id in records],
        start_time,
        resource_start,
        preflight_rows,
        run_manifest,
    )
    print(json.dumps(final_manifest, sort_keys=True))
    return 0 if final_manifest["decision_token"] == DECISION_PROFILE_EXECUTION_COMPLETE else 1
