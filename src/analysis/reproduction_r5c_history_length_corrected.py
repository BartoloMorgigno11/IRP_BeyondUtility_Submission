"""R5C_CORRECTED History-Length full rerun.

This is a new protocol identity over the same 2400 frozen R5A
HISTORY_LENGTH_RECOVERY rows. It preserves the original R5C raw namespace and
changes only the output contract wording so candidate positions are
unambiguous. Retries remain disabled.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import reproduction_r5c_history_length_recovery as base
from src.analysis.reproduction_r5b_canddif_recovery import (
    EXPECTED_R5A_MANIFEST_SHA256,
    FAIL_STRICT_DENOMINATOR_POLICY,
    candidate_item_ids,
    canonical_json,
    git_stdout,
    load_json,
    parse_expected_count,
    protected_hash_check,
    read_csv,
    rel_path,
    sha256_file,
    sha256_text,
    write_csv,
    write_json_atomic,
)
from src.llm.canonical_llm_contract import parse_strict_index_response


PHASE = "R5C_CORRECTED"
DEFAULT_CONFIG_PATH = Path("configs") / "reproduction_r5c_history_length_corrected.yaml"
NOTES_PATH = Path("report") / "notes.md"
R5C_ORIGINAL_PROTOCOL_HASH = "30E4BCA9E1456EE9A7DE74A09FE03BF1A26EFCF758915052FD88F39AF9A836F2"
R5C_PROMPT_TEMPLATE_ID = "r5c_corrected_history_length_json_index_exact5_v2"
R5C_PARSER_CONTRACT = "r5c_ranked_index_json_top5_v1"
EVIDENCE_CLASS = "PAPER_ALIGNED_ADAPTED_HISTORY_LENGTH_RECOVERY_CORRECTED"
GLOBAL_STATUS = "CRITICAL_EXPERIMENTAL_PATH_R5C_CORRECTED_PENDING_FREEZE"

DECISION_COMPLETE = "R5C_CORRECTED_HISTORY_LENGTH_COMPLETE"
DECISION_PREFLIGHT_FAILED = "R5C_CORRECTED_STATIC_PREFLIGHT_FAILED"
DECISION_CONTEXT_FAILED = "R5C_CORRECTED_CONTEXT_LENGTH_CHECK_FAILED"
DECISION_CANARY_FAILED = "R5C_CORRECTED_CANARY_FAILED"
DECISION_EARLY_FAILED = "R5C_CORRECTED_EARLY_RUN_FAILED"
DECISION_EXECUTION_FAILED = "R5C_CORRECTED_EXECUTION_FAILED"
DECISION_SCIENTIFIC_FAILED = "R5C_CORRECTED_SCIENTIFIC_VALIDITY_FAILED"
DECISION_PREFLIGHT_READY = "R5C_CORRECTED_PREFLIGHT_READY"
DECISION_CANARY_PASS = "R5C_CORRECTED_CANARY_PASS"
DECISION_EARLY_PASS = "R5C_CORRECTED_EARLY_HEALTH_PASS"

ORIGINAL_R5C_NAMESPACE = "results/recovery/r5c_history_length/"


@dataclass(frozen=True)
class CorrectedBuildResult:
    decision: str
    frames: dict[str, pd.DataFrame]
    report: str
    manifest_sha256: str
    protocol_hash: str
    executed_this_run: int
    full_run_authorized: bool
    blocked_reason: str


def corrected_instruction(expected_count: int = 5, candidate_count: int = 20) -> str:
    return (
        "Use only the train-history and candidate context below. "
        f"Return exactly {expected_count} 1-based candidate indices. "
        f"Valid outputs are candidate POSITIONS 1..{candidate_count}, not dataset item IDs. "
        "Use only integer indices corresponding to positions in the candidate list supplied in this prompt. "
        "Do not return item IDs. "
        f"Return only valid JSON with key ranked_candidate_indices containing EXACTLY {expected_count} "
        f"unique integers ordered from rank 1 to rank {expected_count}: "
        '{"ranked_candidate_indices":[2,5,1,7,4]}. '
        f"Do not return more than {expected_count} indices. "
        f"Do not return fewer than {expected_count} indices. "
        "Do not include rationale, explanation, prose, Markdown, or extra keys."
    )


def render_corrected_prompt(old_prompt: str, expected_count: int = 5, candidate_count: int = 20) -> str:
    old_sentence = (
        "Use only the train-history and candidate context below. "
        "Return JSON with key ranked_candidate_indices containing unique 1-based candidate indices."
    )
    instruction = corrected_instruction(expected_count, candidate_count)
    if old_sentence in old_prompt:
        return old_prompt.replace(old_sentence, instruction, 1)
    marker = "\nUser history:\n"
    if marker not in old_prompt:
        raise base.R5CHistoryLengthError("cannot locate User history block in frozen history-length prompt")
    head, tail = old_prompt.split(marker, 1)
    head_lines = head.rstrip().splitlines()
    if len(head_lines) < 3:
        raise base.R5CHistoryLengthError("unexpected frozen history-length prompt header")
    corrected_head = "\n".join(head_lines[:3] + [instruction])
    return corrected_head + marker + tail


def corrected_request_id(original_request_id: str) -> str:
    suffix = sha256_text(f"{PHASE}|{original_request_id}")[:16]
    return f"R5C-CORR-{suffix}"


def corrected_request_key(row: Mapping[str, Any]) -> str:
    fields = [
        "phase",
        "r5a_request_id",
        "recovery_family",
        "dataset",
        "model_name",
        "task",
        "experimental_condition",
        "history_length",
        "history_item_count",
        "user_id",
        "candidate_list_hash",
        "history_hash",
        "prompt_template_id",
        "parser_contract",
        "expected_output_schema",
        "prompt_hash",
    ]
    return canonical_json({field: row.get(field, "") for field in fields})


def corrected_material_context_hash(row: Mapping[str, Any]) -> str:
    return sha256_text(
        canonical_json(
            {
                "phase": PHASE,
                "r5a_request_id": row.get("r5a_request_id", ""),
                "dataset": row.get("dataset", ""),
                "user_id": row.get("user_id", ""),
                "model_name": row.get("model_name", ""),
                "experimental_condition": row.get("experimental_condition", ""),
                "history_length": row.get("history_length", ""),
                "history_item_count": row.get("history_item_count", ""),
                "history_hash": row.get("history_hash", ""),
                "candidate_list_hash": row.get("candidate_list_hash", ""),
                "unordered_candidate_item_set_hash": row.get("unordered_candidate_item_set_hash", ""),
                "positive_item_id": row.get("positive_item_id", ""),
                "prompt_hash": row.get("prompt_hash", ""),
                "parser_contract": row.get("parser_contract", ""),
                "expected_output_schema": row.get("expected_output_schema", ""),
            }
        )
    )


def load_corrected_manifest(
    project_root: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    frozen = config.get("frozen_r5a") or {}
    path = base.project_path(project_root, frozen["manifest"])
    if not path.exists():
        raise FileNotFoundError(path)
    manifest_sha = sha256_file(path)
    expected_sha = str(frozen.get("manifest_sha256", EXPECTED_R5A_MANIFEST_SHA256)).upper()
    if manifest_sha != expected_sha:
        raise base.R5CHistoryLengthError(f"R5A manifest SHA mismatch observed={manifest_sha} expected={expected_sha}")
    manifest = read_csv(path)
    expected_rows = int(frozen.get("manifest_rows", 4800))
    if len(manifest) != expected_rows:
        raise base.R5CHistoryLengthError(f"R5A manifest row count changed: {len(manifest)} != {expected_rows}")

    scope = config.get("scope") or {}
    family = str(scope.get("recovery_family", "HISTORY_LENGTH_RECOVERY"))
    history = manifest.loc[manifest["recovery_family"].eq(family)].copy()
    selected_rows = int(scope.get("selected_rows", 2400))
    if len(history) != selected_rows:
        raise base.R5CHistoryLengthError(f"R5C_CORRECTED history subset row count changed: {len(history)} != {selected_rows}")
    if not history["task"].eq("ranking").all():
        raise base.R5CHistoryLengthError("R5C_CORRECTED history subset contains non-ranking rows")
    if history["recovery_family"].isin(set(scope.get("forbidden_recovery_families") or [])).any():
        raise base.R5CHistoryLengthError("R5C_CORRECTED history subset contains forbidden recovery families")
    if history["task"].isin(set(scope.get("forbidden_tasks") or [])).any():
        raise base.R5CHistoryLengthError("R5C_CORRECTED history subset contains forbidden tasks")
    if set(history["dataset"]) != set(scope.get("datasets") or []):
        raise base.R5CHistoryLengthError(f"R5C_CORRECTED datasets changed: {sorted(set(history['dataset']))}")
    if set(history["model_name"]) != set(scope.get("models") or []):
        raise base.R5CHistoryLengthError(f"R5C_CORRECTED models changed: {sorted(set(history['model_name']))}")

    namespace = base.r5c_namespace(project_root, config)
    expected_count = base.expected_k(config)
    original_request_ids = history["request_id"].astype(str).tolist()
    history["phase"] = PHASE
    history["r5a_request_id"] = original_request_ids
    history["original_output_destination"] = history["output_destination"].astype(str)
    history["request_id"] = [corrected_request_id(value) for value in original_request_ids]
    history["prompt_template_id"] = R5C_PROMPT_TEMPLATE_ID
    history["prompt_template_source"] = "src/analysis/reproduction_r5c_history_length_corrected.py"
    history["parser_contract"] = R5C_PARSER_CONTRACT
    history["expected_output_schema"] = canonical_json({"ranked_candidate_indices": list(range(1, expected_count + 1))})
    history["prompt_render"] = [
        render_corrected_prompt(str(row["prompt_render"]), expected_count, int(row["candidate_count"]))
        for row in history.to_dict("records")
    ]
    history["prompt_hash"] = history["prompt_render"].map(sha256_text)
    history["output_destination"] = [
        base.normalize_rel(Path((config.get("outputs") or {})["result_namespace"]) / f"{request_id}.json")
        for request_id in history["request_id"].astype(str)
    ]
    history["request_key_sha256"] = [sha256_text(corrected_request_key(row)) for row in history.to_dict("records")]
    history["material_context_hash"] = [corrected_material_context_hash(row) for row in history.to_dict("records")]

    if not history["request_id"].is_unique:
        raise base.R5CHistoryLengthError("R5C_CORRECTED request IDs are not unique")
    if not history["output_destination"].is_unique:
        raise base.R5CHistoryLengthError("R5C_CORRECTED output destinations are not unique")
    for destination in history["output_destination"].astype(str):
        normalized = base.normalize_rel(destination)
        resolved = base.project_path(project_root, destination).resolve()
        if namespace.resolve() not in [resolved, *resolved.parents]:
            raise base.R5CHistoryLengthError(f"R5C_CORRECTED output path outside namespace: {destination}")
        if normalized.startswith(base.ORIGINAL_R5_NAMESPACE):
            raise base.R5CHistoryLengthError(f"R5C_CORRECTED output path overlaps original R5 namespace: {destination}")
        if normalized.startswith(base.R5B_CORRECTED_NAMESPACE):
            raise base.R5CHistoryLengthError(f"R5C_CORRECTED output path overlaps corrected R5B namespace: {destination}")
        if normalized.startswith(ORIGINAL_R5C_NAMESPACE):
            raise base.R5CHistoryLengthError(f"R5C_CORRECTED output path overlaps original R5C namespace: {destination}")
    return manifest, history.reset_index(drop=True), manifest_sha


def _file_hash_or_missing(path: Path) -> str:
    return sha256_file(path) if path.exists() else "MISSING"


def protocol_metadata_corrected(
    project_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    history: pd.DataFrame,
    manifest_sha: str,
    group_validation: pd.DataFrame,
    runtime_tags: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[str, pd.DataFrame]:
    module_path = Path(__file__).resolve()
    parser_path = project_root / "src" / "llm" / "canonical_llm_contract.py"
    manifest_builder = project_root / "src" / "analysis" / "reproduction_r5a_controlled_recovery_manifest.py"
    prompt_hashes = sorted(history["prompt_hash"].astype(str).unique())
    prompt_family_hash = sha256_text(canonical_json(prompt_hashes))
    group_hash = sha256_text(canonical_json(group_validation.fillna("").to_dict("records")))
    generation_config = {
        "endpoint": str((config.get("runtime") or {}).get("endpoint", "")),
        "format": str((config.get("runtime") or {}).get("format", "json")),
        "stream": str((config.get("runtime") or {}).get("stream", False)).lower(),
        "temperature": str((config.get("runtime") or {}).get("temperature", 0)),
        "max_retries": str((config.get("runtime") or {}).get("max_retries", 0)),
        "timeout_seconds": str((config.get("runtime") or {}).get("timeout_seconds", 120)),
        "num_predict_values": sorted(history["max_new_tokens"].astype(str).unique()),
        "seed_values": sorted(history["seed"].astype(str).unique()),
    }
    models = []
    for model in sorted(set(history["model_name"].astype(str))):
        details = (runtime_tags or {}).get(model, {})
        models.append(
            {
                "model_name": model,
                "digest": str(details.get("digest", "")),
                "details": details.get("details", {}) if isinstance(details.get("details"), dict) else {},
            }
        )
    payload = {
        "phase": PHASE,
        "manifest_sha256": manifest_sha,
        "selected_rows": len(history),
        "expected_k": base.expected_k(config),
        "prompt_template_id": R5C_PROMPT_TEMPLATE_ID,
        "prompt_family_hash": prompt_family_hash,
        "parser_contract": R5C_PARSER_CONTRACT,
        "parser_hash": _file_hash_or_missing(parser_path),
        "runner_hash": _file_hash_or_missing(module_path),
        "config_hash": _file_hash_or_missing(config_path),
        "history_candidate_construction_hash": _file_hash_or_missing(manifest_builder),
        "metric_hash": _file_hash_or_missing(project_root / "src" / "analysis" / "reproduction_r5c_history_length_recovery.py"),
        "group_validation_hash": group_hash,
        "generation_config": generation_config,
        "models": models,
    }
    protocol_hash = sha256_text(canonical_json(payload))
    rows = [
        {"component": "manifest", "value": manifest_sha, "sha256": manifest_sha},
        {"component": "prompt_family", "value": R5C_PROMPT_TEMPLATE_ID, "sha256": prompt_family_hash},
        {"component": "parser", "value": R5C_PARSER_CONTRACT, "sha256": payload["parser_hash"]},
        {"component": "runner", "value": base.normalize_rel(module_path), "sha256": payload["runner_hash"]},
        {"component": "config", "value": base.normalize_rel(config_path), "sha256": payload["config_hash"]},
        {"component": "history_candidate_construction", "value": "R5A frozen history-length rows", "sha256": payload["history_candidate_construction_hash"]},
        {"component": "group_validation", "value": "same candidates and positive item across history lengths", "sha256": group_hash},
        {"component": "metrics", "value": "reuse R5C HR@5/NDCG@5 implementation", "sha256": payload["metric_hash"]},
        {"component": "generation_config", "value": canonical_json(generation_config), "sha256": sha256_text(canonical_json(generation_config))},
        {"component": "protocol", "value": "R5C_CORRECTED_PROTOCOL_PREFLIGHT_READY", "sha256": protocol_hash},
    ]
    for model in models:
        rows.append({"component": "model_identifier", "value": model["model_name"], "sha256": str(model.get("digest", ""))})
    return protocol_hash, pd.DataFrame(rows)


def protected_artifact_integrity(project_root: Path) -> pd.DataFrame:
    inherited = protected_hash_check(project_root)
    if inherited.empty:
        return inherited
    frame = inherited.copy()
    hash_ok = frame["expected_sha256"].astype(str).str.upper().eq(frame["observed_sha256"].astype(str).str.upper())
    clean_status = frame["git_status"].astype(str).eq("")
    exists = frame["exists"].astype(str).eq("true")
    frame["hash_unchanged"] = [base.bool_text(value) for value in (exists & hash_ok & clean_status).tolist()]
    frame["protection_status"] = [
        "PROTECTED_UNCHANGED" if value else "PROTECTED_CONFLICT"
        for value in (exists & hash_ok & clean_status).tolist()
    ]
    return frame


def build_static_preflight_corrected(
    project_root: Path,
    config: Mapping[str, Any],
    manifest: pd.DataFrame,
    history: pd.DataFrame,
    manifest_sha: str,
    group_validation: pd.DataFrame,
    context_preflight: pd.DataFrame,
    protocol_hash: str,
) -> pd.DataFrame:
    scope = config.get("scope") or {}
    runtime = config.get("runtime") or {}
    outputs = config.get("outputs") or {}
    namespace = base.normalize_rel(outputs["result_namespace"]).rstrip("/") + "/"
    old_outputs = set(history["original_output_destination"].astype(str).map(base.normalize_rel))
    new_outputs = set(history["output_destination"].astype(str).map(base.normalize_rel))
    prompts = history["prompt_render"].astype(str)
    original_repo_status = git_stdout(["-C", "original_repo/EvaLLMasRecommender", "status", "--short"], project_root)
    head = git_stdout(["rev-parse", "HEAD"], project_root)
    protected = protected_artifact_integrity(project_root)
    r5b_freeze = base.project_path(project_root, ((config.get("frozen_r5b_corrected") or {}).get("freeze_manifest", "")))
    r5b_protocol = ""
    if r5b_freeze.exists():
        payload = load_json(r5b_freeze)
        r5b_protocol = str(payload.get("r5b_corrected_protocol_hash", "") or payload.get("protocol_hash", ""))
    expected_by_dataset = base.configured_conditions_by_dataset(config)
    cell_counts = history.groupby(["dataset", "model_name", "experimental_condition"], sort=True).size()
    group_counts = history.groupby(["dataset", "model_name", "user_id"], sort=True).size()
    out_of_scope = manifest.loc[~manifest["recovery_family"].eq("HISTORY_LENGTH_RECOVERY")]
    original_r5c_raw = project_root / "results" / "recovery" / "r5c_history_length"
    original_r5c_count = len(list(original_r5c_raw.glob("*.json"))) if original_r5c_raw.exists() else 0
    checks: list[tuple[str, str, bool, str, str]] = [
        ("STATIC-001", "manifest SHA", manifest_sha == EXPECTED_R5A_MANIFEST_SHA256, manifest_sha, EXPECTED_R5A_MANIFEST_SHA256),
        ("STATIC-002", "repository HEAD is R5B corrected freeze", head == str((config.get("frozen_r5b_corrected") or {}).get("freeze_commit", base.R5B_CORRECTED_FREEZE_COMMIT)), head, str((config.get("frozen_r5b_corrected") or {}).get("freeze_commit", base.R5B_CORRECTED_FREEZE_COMMIT))),
        ("STATIC-003", "R5B corrected freeze manifest exists", r5b_freeze.exists(), base.normalize_rel(r5b_freeze), "exists"),
        ("STATIC-004", "R5B corrected protocol hash", r5b_protocol == base.R5B_CORRECTED_PROTOCOL_HASH, r5b_protocol, base.R5B_CORRECTED_PROTOCOL_HASH),
        ("STATIC-005", "manifest rows", len(manifest) == int((config.get("frozen_r5a") or {}).get("manifest_rows", 4800)), str(len(manifest)), "4800"),
        ("STATIC-006", "selected history rows", len(history) == int(scope.get("selected_rows", 2400)), str(len(history)), "2400"),
        ("STATIC-007", "out-of-scope rows remain unselected", len(out_of_scope) == 2400, str(len(out_of_scope)), "2400"),
        ("STATIC-008", "no CandDif rows selected", not history["recovery_family"].eq("CANDDIF_RECOVERY").any(), str(int(history["recovery_family"].eq("CANDDIF_RECOVERY").sum())), "0"),
        ("STATIC-009", "no Generation rows selected", not history["recovery_family"].eq("GENERATION_INVOLVED_RECOVERY").any(), str(int(history["recovery_family"].eq("GENERATION_INVOLVED_RECOVERY").sum())), "0"),
        ("STATIC-010", "ranking task only", history["task"].eq("ranking").all(), str(sorted(set(history["task"]))), "ranking"),
        ("STATIC-011", "datasets", set(history["dataset"]) == set(scope.get("datasets") or []), str(sorted(set(history["dataset"]))), str(sorted(scope.get("datasets") or []))),
        ("STATIC-012", "models", set(history["model_name"]) == set(scope.get("models") or []), str(sorted(set(history["model_name"]))), str(sorted(scope.get("models") or []))),
        ("STATIC-013", "expected cell count", len(cell_counts) == int(scope.get("expected_cells", 24)) and cell_counts.eq(100).all(), f"{len(cell_counts)} cells; min={int(cell_counts.min())}; max={int(cell_counts.max())}", "24 cells of 100"),
        ("STATIC-014", "expected group count", len(group_counts) == int(scope.get("expected_groups", 800)) and group_counts.eq(3).all(), f"{len(group_counts)} groups; min={int(group_counts.min())}; max={int(group_counts.max())}", "800 groups of 3"),
        ("STATIC-015", "conditions by dataset", all(set(history.loc[history["dataset"].eq(dataset), "experimental_condition"].astype(str)) == conditions for dataset, conditions in expected_by_dataset.items()), str({dataset: sorted(set(history.loc[history["dataset"].eq(dataset), "experimental_condition"].astype(str))) for dataset in expected_by_dataset}), str({k: sorted(v) for k, v in expected_by_dataset.items()})),
        ("STATIC-016", "expected K is 5", history.apply(parse_expected_count, axis=1).eq(5).all(), str(sorted(set(history.apply(parse_expected_count, axis=1)))), "[5]"),
        ("STATIC-017", "candidate count 20", history["candidate_count"].astype(int).eq(20).all(), str(sorted(set(history["candidate_count"].astype(str)))), "20"),
        ("STATIC-018", "group validation", len(group_validation) == int(scope.get("expected_groups", 800)) and group_validation["group_valid"].eq("true").all(), f"{int(group_validation['group_valid'].eq('true').sum())}/{len(group_validation)}", "800/800"),
        ("STATIC-019", "corrected prompt says EXACTLY 5", prompts.str.contains("EXACTLY 5", regex=False).all(), str(int(prompts.str.contains("EXACTLY 5", regex=False).sum())), str(len(history))),
        ("STATIC-020", "corrected prompt states positions 1..N", prompts.str.contains("candidate POSITIONS 1..20", regex=False).all(), str(int(prompts.str.contains("candidate POSITIONS 1..20", regex=False).sum())), str(len(history))),
        ("STATIC-021", "corrected prompt rejects item IDs", prompts.str.contains("not dataset item IDs", regex=False).all() and prompts.str.contains("Do not return item IDs", regex=False).all(), "checked", "all prompts"),
        ("STATIC-022", "parser contract unchanged strict index parser", set(history["parser_contract"]) == {R5C_PARSER_CONTRACT}, str(sorted(set(history["parser_contract"]))), R5C_PARSER_CONTRACT),
        ("STATIC-023", "local endpoint only", str(runtime.get("endpoint", "")).startswith("http://127.0.0.1") or str(runtime.get("endpoint", "")).startswith("http://localhost"), str(runtime.get("endpoint", "")), "localhost Ollama"),
        ("STATIC-024", "format json stream false", str(runtime.get("format", "")) == "json" and bool(runtime.get("stream", False)) is False, f"format={runtime.get('format')};stream={runtime.get('stream')}", "format=json;stream=false"),
        ("STATIC-025", "max_retries remains zero", int(runtime.get("max_retries", 0)) == 0, str(runtime.get("max_retries", 0)), "0"),
        ("STATIC-026", "output namespace R5C corrected", namespace == "results/recovery/r5c_history_length_corrected/", namespace, "results/recovery/r5c_history_length_corrected/"),
        ("STATIC-027", "no output path overlap with original R5A destinations", old_outputs.isdisjoint(new_outputs), str(len(old_outputs.intersection(new_outputs))), "0"),
        ("STATIC-028", "no output path overlap with original R5C namespace", not any(path.startswith(ORIGINAL_R5C_NAMESPACE) for path in new_outputs), "checked", "none"),
        ("STATIC-029", "original R5C raw preserved", original_r5c_count == 2400, str(original_r5c_count), "2400"),
        ("STATIC-030", "original repo clean", original_repo_status == "", original_repo_status, "empty"),
        ("STATIC-031", "protected artifacts unchanged", protected.empty or protected["protection_status"].eq("PROTECTED_UNCHANGED").all(), f"{int(protected['protection_status'].eq('PROTECTED_UNCHANGED').sum())}/{len(protected)}", f"{len(protected)}/{len(protected)}"),
        ("STATIC-032", "disk free guard", base.shutil.disk_usage(project_root).free >= int(runtime.get("disk_free_min_bytes", 0)), str(base.shutil.disk_usage(project_root).free), f">={runtime.get('disk_free_min_bytes', 0)}"),
        ("STATIC-033", "context envelope", context_preflight["status"].eq("pass").all(), f"{int(context_preflight['status'].eq('pass').sum())}/{len(context_preflight)}", "all pass"),
        ("STATIC-034", "protocol hash present and new", len(protocol_hash) == 64 and protocol_hash != R5C_ORIGINAL_PROTOCOL_HASH, protocol_hash, "64-char SHA256 different from original R5C"),
    ]
    return pd.DataFrame(
        [
            {"check_id": check_id, "check_name": name, "status": "pass" if ok else "fail", "observed": observed, "expected": expected}
            for check_id, name, ok, observed, expected in checks
        ]
    )


def _ml1m_item_id_case_row(history: pd.DataFrame) -> tuple[dict[str, Any], list[int]]:
    ml1m = history.loc[history["dataset"].eq("ML-1M")]
    for row in ml1m.to_dict("records"):
        ids: list[int] = []
        for value in candidate_item_ids(row):
            if str(value).isdigit() and int(value) > int(row["candidate_count"]):
                ids.append(int(value))
            if len(ids) == 5:
                return row, ids
    raise base.R5CHistoryLengthError("cannot find ML-1M synthetic row with five numeric item IDs outside 1..N")


def build_synthetic_tests_corrected(history: pd.DataFrame) -> pd.DataFrame:
    synthetic = _ORIGINAL_BUILD_SYNTHETIC_TESTS(history).copy()
    row, item_ids = _ml1m_item_id_case_row(history)
    text = canonical_json({"ranked_candidate_indices": item_ids})
    parsed = parse_strict_index_response(
        text,
        candidate_items=candidate_item_ids(row),
        expected_count=5,
        require_full_permutation=False,
    )
    extra = pd.DataFrame(
        [
            {
                "case_id": "ml1m_item_ids_in_indices_field",
                "expected_valid": "false",
                "observed_valid": base.bool_text(bool(parsed.parse_valid)),
                "expected_status": "unknown_indices",
                "observed_status": parsed.status,
                "output_length": int(parsed.output_length),
                "duplicate_count": int(parsed.duplicate_count),
                "unknown_item_count": int(parsed.unknown_item_count),
                "pass": base.bool_text((not parsed.parse_valid) and parsed.status == "unknown_indices"),
                "notes": f"actual ML-1M item IDs {item_ids} must not be accepted as candidate positions",
            }
        ]
    )
    return pd.concat([synthetic, extra], ignore_index=True)


def _parse_raw_values(raw: str) -> list[int]:
    try:
        payload = json.loads(str(raw).strip())
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    values = payload.get("ranked_candidate_indices")
    if not isinstance(values, list):
        return []
    parsed: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            return []
        parsed.append(int(value))
    return parsed


_ORIGINAL_BUILD_SYNTHETIC_TESTS = base.build_synthetic_tests
_ORIGINAL_STATUS_FOR_PAYLOAD = base.status_for_payload
_ORIGINAL_BUILD_VALIDATION = base.build_validation
_ORIGINAL_WRITE_OUTPUTS = base.write_outputs
_ORIGINAL_RENDER_REPORT = base.render_report


def status_for_payload_corrected(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    manifest_sha: str,
    protocol_hash: str,
    path: Path,
) -> dict[str, Any]:
    record = _ORIGINAL_STATUS_FOR_PAYLOAD(row, payload, manifest_sha, protocol_hash, path)
    values = _parse_raw_values(str(payload.get("raw_response_text", "")))
    candidate_count = int(row.get("candidate_count", 0) or 0)
    item_ids = {str(value) for value in candidate_item_ids(row)}
    out_of_range_item_ids = [value for value in values if not 1 <= value <= candidate_count and str(value) in item_ids]
    record.update(
        {
            "ranked_candidate_indices_json": canonical_json(values),
            "ranked_item_ids_json": canonical_json(payload.get("ranked_item_ids", [])),
            "raw_response_text": str(payload.get("raw_response_text", "")),
            "values_within_1_to_n": base.bool_text(bool(values) and all(1 <= value <= candidate_count for value in values)),
            "dataset_item_id_substitution_count": len(out_of_range_item_ids),
            "dataset_item_id_substitution_values": "|".join(str(value) for value in out_of_range_item_ids),
        }
    )
    return record


def canary_pass_corrected(
    status: pd.DataFrame,
    canary_ids: set[str],
    config: Mapping[str, Any],
) -> tuple[bool, str, pd.DataFrame]:
    canary = status.loc[status["request_id"].isin(canary_ids)].copy()
    table = canary[
        [
            "model_name",
            "dataset",
            "experimental_condition",
            "history_length",
            "backend_complete",
            "output_length",
            "parse_valid",
            "validity_class",
            "parse_status",
            "values_within_1_to_n",
            "dataset_item_id_substitution_count",
            "ranked_candidate_indices_json",
            "raw_response_text",
            "request_id",
            "r5a_request_id",
        ]
    ].rename(
        columns={
            "model_name": "Model",
            "dataset": "Dataset",
            "experimental_condition": "Condition",
            "history_length": "HistoryLength",
            "backend_complete": "Complete",
            "output_length": "Parsed K",
            "parse_valid": "Valid",
            "validity_class": "Failure",
            "values_within_1_to_n": "Within1ToN",
            "dataset_item_id_substitution_count": "ItemIdSubstitutionCount",
            "ranked_candidate_indices_json": "ParsedValues",
            "raw_response_text": "Raw",
        }
    )
    expected_cells = int((config.get("scope") or {}).get("expected_cells", 24))
    if len(canary) != expected_cells:
        return False, f"canary accounted {len(canary)} != {expected_cells}", table
    required = {
        ("qwen2:7b-instruct", "ML-1M", "HISTORY_LENGTH_5"),
        ("qwen2:7b-instruct", "ML-1M", "HISTORY_LENGTH_10"),
        ("qwen2:7b-instruct", "ML-1M", "HISTORY_LENGTH_20"),
    }
    observed = set(zip(canary["model_name"], canary["dataset"], canary["experimental_condition"]))
    missing = sorted(required.difference(observed))
    if missing:
        return False, f"canary missing required Qwen2 ML-1M cells {missing}", table
    failed = canary.loc[~canary["parse_valid"].eq("true")]
    if not failed.empty:
        return False, f"canary invalid outputs {len(failed)}", table
    if not canary["output_length"].astype(str).eq("5").all():
        return False, "canary did not return exactly K=5 everywhere", table
    if not canary["values_within_1_to_n"].eq("true").all():
        return False, "canary has values outside 1..N", table
    if int(canary["dataset_item_id_substitution_count"].astype(int).sum()) > 0:
        return False, "canary repeated item-ID substitution", table
    health = base.health_summary(canary, label="canary")
    ok, reason = health_pass_corrected(health, config)
    return ok, reason, table


def health_pass_corrected(health: pd.DataFrame, config: Mapping[str, Any]) -> tuple[bool, str]:
    if health.empty:
        return False, "no terminal outputs"
    overall = health.loc[health["scope"].eq("overall")].iloc[0]
    if int(overall["backend_incomplete"]) > 0:
        return False, "backend incomplete present"
    if int(overall["backend_failed"]) > 0:
        return False, "backend failed present"
    if int(overall["wrong_length"]) > 0:
        return False, "wrong_length present"
    if int(overall["parse_failure"]) > 0:
        return False, "parse_failure present"
    if int(overall["out_of_range"]) > 0:
        return False, "out_of_range present"
    if int(overall["duplicate"]) > 0:
        return False, "duplicate present"
    qwen_ml1m = health.loc[
        health["scope"].eq("model_dataset_condition")
        & health["key"].astype(str).str.startswith("qwen2:7b-instruct|ML-1M|")
    ]
    if not qwen_ml1m.empty and int(qwen_ml1m["out_of_range"].astype(int).sum()) > 0:
        return False, "Qwen2 ML-1M out_of_range item-ID substitution pattern present"
    if float(overall["validity_rate"]) < 1.0:
        return False, f"overall validity {overall['validity_rate']} < 1.0"
    return True, ""


def build_validation_corrected(
    static_preflight: pd.DataFrame,
    synthetic_tests: pd.DataFrame,
    context_preflight: pd.DataFrame,
    canary_table: pd.DataFrame,
    early_health: pd.DataFrame,
    status: pd.DataFrame,
    group_validation: pd.DataFrame,
    *,
    full_run_authorized: bool,
    decision: str,
) -> pd.DataFrame:
    validation = _ORIGINAL_BUILD_VALIDATION(
        static_preflight,
        synthetic_tests,
        context_preflight,
        canary_table,
        early_health,
        status,
        group_validation,
        full_run_authorized=full_run_authorized,
        decision=decision,
    )
    terminal = status.loc[~status["execution_status"].eq("NOT_EXECUTED")]
    extras = [
        ("VAL-CORR-010", "canary values are candidate positions", canary_table.empty or canary_table["Within1ToN"].eq("true").all(), "checked", "all true"),
        ("VAL-CORR-011", "canary has no item-ID substitution", canary_table.empty or int(canary_table["ItemIdSubstitutionCount"].astype(int).sum()) == 0, "checked", "0"),
        ("VAL-CORR-012", "max retries zero in terminal outputs", terminal.empty or terminal["retry_count"].astype(int).eq(0).all(), "checked", "0"),
        ("VAL-CORR-013", "no out_of_range when complete", decision != DECISION_COMPLETE or int(terminal["validity_class"].eq("OUT_OF_RANGE").sum()) == 0, str(int(terminal["validity_class"].eq("OUT_OF_RANGE").sum())), "0"),
    ]
    return pd.concat(
        [
            validation,
            pd.DataFrame(
                [
                    {"check_id": check_id, "check_name": name, "status": "pass" if ok else "fail", "observed": observed, "expected": expected}
                    for check_id, name, ok, observed, expected in extras
                ]
            ),
        ],
        ignore_index=True,
    )


def write_freeze_candidate_corrected(
    project_root: Path,
    config: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    manifest_sha: str,
    protocol_hash: str,
    decision: str,
) -> None:
    status = frames["request_execution_status"]
    terminal = status.loc[~status["execution_status"].eq("NOT_EXECUTED")]
    namespace = base.r5c_namespace(project_root, config)
    raw_paths = sorted(namespace.glob("*.json")) if namespace.exists() else []
    aggregate_hash = sha256_text(canonical_json([sha256_file(path) for path in raw_paths]))
    valid = int(terminal["parse_valid"].eq("true").sum())
    payload = {
        "freeze_metadata_version": "r5c_history_length_corrected_freeze_candidate_v1",
        "phase": PHASE,
        "decision": decision,
        "created_utc": base.utc_now(),
        "r5a_manifest_sha256": manifest_sha,
        "protocol_hash": protocol_hash,
        "original_r5c_protocol_hash": R5C_ORIGINAL_PROTOCOL_HASH,
        "raw_namespace": base.normalize_rel((config.get("outputs") or {})["result_namespace"]),
        "original_r5c_namespace_preserved": ORIGINAL_R5C_NAMESPACE.rstrip("/"),
        "raw_count": len(raw_paths),
        "raw_aggregate_sha256": aggregate_hash,
        "status_counts": {
            "requests_expected": len(status),
            "requests_accounted": len(terminal),
            "valid": valid,
            "invalid": int(len(terminal) - valid),
            "wrong_length": int(terminal["validity_class"].eq("WRONG_LENGTH").sum()),
            "parse_failure": int(terminal["validity_class"].eq("PARSE_FAILURE").sum()),
            "out_of_range": int(terminal["validity_class"].eq("OUT_OF_RANGE").sum()),
            "backend_incomplete": int(terminal["validity_class"].eq("BACKEND_INCOMPLETE").sum()),
        },
        "paper_exact": False,
        "freeze_ready": bool(decision == DECISION_COMPLETE and len(terminal) == len(status) and valid == len(status)),
    }
    destination = base.project_path(project_root, (config.get("outputs") or {})["freeze_candidate"])
    write_json_atomic(payload, destination)


def build_original_comparison(project_root: Path, corrected_status: pd.DataFrame) -> pd.DataFrame:
    original_path = project_root / "results" / "tables" / "reproduction_r5c_history_length_request_execution_status.csv"
    if not original_path.exists():
        return pd.DataFrame()
    original = read_csv(original_path)
    rows: list[dict[str, Any]] = []
    for label, frame in [("R5C_ORIGINAL", original), ("R5C_CORRECTED", corrected_status)]:
        terminal = frame.loc[~frame["execution_status"].eq("NOT_EXECUTED")]
        rows.append(
            {
                "scope": "overall",
                "protocol": label,
                "key": "all",
                "accounted": len(terminal),
                "valid": int(terminal["parse_valid"].eq("true").sum()),
                "invalid": int(len(terminal) - terminal["parse_valid"].eq("true").sum()),
                "out_of_range": int(terminal["validity_class"].eq("OUT_OF_RANGE").sum()),
                "wrong_length": int(terminal["validity_class"].eq("WRONG_LENGTH").sum()),
                "parse_failure": int(terminal["validity_class"].eq("PARSE_FAILURE").sum()),
                "backend_incomplete": int(terminal["validity_class"].eq("BACKEND_INCOMPLETE").sum()),
            }
        )
        qwen_ml1m = terminal.loc[terminal["model_name"].eq("qwen2:7b-instruct") & terminal["dataset"].eq("ML-1M")]
        rows.append(
            {
                "scope": "model_dataset",
                "protocol": label,
                "key": "qwen2:7b-instruct|ML-1M",
                "accounted": len(qwen_ml1m),
                "valid": int(qwen_ml1m["parse_valid"].eq("true").sum()),
                "invalid": int(len(qwen_ml1m) - qwen_ml1m["parse_valid"].eq("true").sum()),
                "out_of_range": int(qwen_ml1m["validity_class"].eq("OUT_OF_RANGE").sum()),
                "wrong_length": int(qwen_ml1m["validity_class"].eq("WRONG_LENGTH").sum()),
                "parse_failure": int(qwen_ml1m["validity_class"].eq("PARSE_FAILURE").sum()),
                "backend_incomplete": int(qwen_ml1m["validity_class"].eq("BACKEND_INCOMPLETE").sum()),
            }
        )
        for history_length, group in qwen_ml1m.groupby("history_length", sort=True):
            rows.append(
                {
                    "scope": "qwen2_ml1m_history_length",
                    "protocol": label,
                    "key": str(history_length),
                    "accounted": len(group),
                    "valid": int(group["parse_valid"].eq("true").sum()),
                    "invalid": int(len(group) - group["parse_valid"].eq("true").sum()),
                    "out_of_range": int(group["validity_class"].eq("OUT_OF_RANGE").sum()),
                    "wrong_length": int(group["validity_class"].eq("WRONG_LENGTH").sum()),
                    "parse_failure": int(group["validity_class"].eq("PARSE_FAILURE").sum()),
                    "backend_incomplete": int(group["validity_class"].eq("BACKEND_INCOMPLETE").sum()),
                }
            )
    return pd.DataFrame(rows)


def write_outputs_corrected(
    project_root: Path,
    config: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    report: str,
    manifest_sha: str,
    protocol_hash: str,
    decision: str,
) -> None:
    _ORIGINAL_WRITE_OUTPUTS(project_root, config, frames, report, manifest_sha, protocol_hash, decision)
    comparison = build_original_comparison(project_root, frames["request_execution_status"])
    if not comparison.empty:
        write_csv(comparison, base.project_path(project_root, (config.get("outputs") or {})["original_comparison"]))


def render_report_corrected(
    frames: Mapping[str, pd.DataFrame],
    decision: str,
    manifest_sha: str,
    protocol_hash: str,
    full_run_authorized: bool,
    blocked_reason: str,
    executed_this_run: int,
) -> str:
    base_report = _ORIGINAL_RENDER_REPORT(frames, decision, manifest_sha, protocol_hash, full_run_authorized, blocked_reason, executed_this_run)
    return base_report.replace("# R5C History-Length Recovery", "# R5C_CORRECTED History-Length Recovery").replace(
        "R5C executes only frozen R5A",
        "R5C_CORRECTED executes only frozen R5A",
    ).replace(
        "R5C is a local adapted history-length recovery",
        "R5C_CORRECTED is a local adapted history-length recovery",
    )


def update_notes_corrected(project_root: Path, decision: str, manifest_sha: str, protocol_hash: str, status: pd.DataFrame) -> None:
    notes_path = project_root / NOTES_PATH
    terminal = status.loc[~status["execution_status"].eq("NOT_EXECUTED")]
    section = f"""

## R5C_CORRECTED History-Length recovery

- Decision: `{decision}`.
- Manifest SHA256: `{manifest_sha}`.
- Protocol hash: `{protocol_hash}`.
- Scope: new `R5C_CORRECTED` identity over the same 2400 frozen R5A `HISTORY_LENGTH_RECOVERY` rows; original R5C raw outputs are preserved.
- Fix: prompt output contract now explicitly requires candidate positions `1..N`, not dataset item IDs; parser/validator, candidate sets, histories, targets, seed policy, metrics, Ollama policy, and `max_retries=0` are preserved.
- Outputs: `results/recovery/r5c_history_length_corrected/`, `results/diagnostics/r5c_history_length_corrected/`, `results/tables/reproduction_r5c_history_length_corrected_*.csv`, and `report/reproduction_r5c_history_length_corrected.md`.
- Request accounting: expected `2400`, accounted `{len(terminal)}`, valid `{int(terminal["parse_valid"].eq("true").sum())}`, invalid `{int(len(terminal) - terminal["parse_valid"].eq("true").sum())}`.
- R5D was not executed.
"""
    existing = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    marker = "## R5C_CORRECTED History-Length recovery"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + section
    else:
        existing = existing.rstrip() + section
    notes_path.write_text(existing.lstrip() + "\n", encoding="utf-8")


@contextmanager
def corrected_protocol_context() -> Iterator[None]:
    patches: dict[str, Any] = {
        "PHASE": PHASE,
        "DEFAULT_CONFIG_PATH": DEFAULT_CONFIG_PATH,
        "R5C_PROMPT_TEMPLATE_ID": R5C_PROMPT_TEMPLATE_ID,
        "R5C_PARSER_CONTRACT": R5C_PARSER_CONTRACT,
        "EVIDENCE_CLASS": EVIDENCE_CLASS,
        "GLOBAL_STATUS": GLOBAL_STATUS,
        "DECISION_COMPLETE": DECISION_COMPLETE,
        "DECISION_PREFLIGHT_FAILED": DECISION_PREFLIGHT_FAILED,
        "DECISION_CONTEXT_FAILED": DECISION_CONTEXT_FAILED,
        "DECISION_CANARY_FAILED": DECISION_CANARY_FAILED,
        "DECISION_EARLY_FAILED": DECISION_EARLY_FAILED,
        "DECISION_EXECUTION_FAILED": DECISION_EXECUTION_FAILED,
        "DECISION_SCIENTIFIC_FAILED": DECISION_SCIENTIFIC_FAILED,
        "DECISION_PREFLIGHT_READY": DECISION_PREFLIGHT_READY,
        "DECISION_CANARY_PASS": DECISION_CANARY_PASS,
        "DECISION_EARLY_PASS": DECISION_EARLY_PASS,
        "corrected_instruction": corrected_instruction,
        "render_corrected_prompt": render_corrected_prompt,
        "r5c_request_id": corrected_request_id,
        "r5c_request_key": corrected_request_key,
        "r5c_material_context_hash": corrected_material_context_hash,
        "load_r5c_manifest": load_corrected_manifest,
        "protocol_metadata": protocol_metadata_corrected,
        "build_static_preflight": build_static_preflight_corrected,
        "build_synthetic_tests": build_synthetic_tests_corrected,
        "status_for_payload": status_for_payload_corrected,
        "canary_pass": canary_pass_corrected,
        "health_pass": health_pass_corrected,
        "build_validation": build_validation_corrected,
        "write_freeze_candidate": write_freeze_candidate_corrected,
        "write_outputs": write_outputs_corrected,
        "render_report": render_report_corrected,
        "update_notes": update_notes_corrected,
    }
    old = {name: getattr(base, name) for name in patches}
    for name, value in patches.items():
        setattr(base, name, value)
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(base, name, value)


def load_config(path: Path | None = None) -> dict[str, Any]:
    return base.load_config(path or DEFAULT_CONFIG_PATH)


def load_r5c_manifest(project_root: Path, config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    return load_corrected_manifest(project_root, config)


def build_r5c_history_length_corrected(
    project_root: Path,
    config_path: Path | None = None,
    *,
    write: bool = True,
    run_gated: bool = False,
    allow_api_calls: bool = False,
    allow_local_ollama: bool = False,
    stop_after: str = "full",
) -> CorrectedBuildResult:
    actual_config_path = config_path or DEFAULT_CONFIG_PATH
    with corrected_protocol_context():
        result = base.build_r5c_history_length(
            project_root,
            config_path=actual_config_path,
            write=write,
            run_gated=run_gated,
            allow_api_calls=allow_api_calls,
            allow_local_ollama=allow_local_ollama,
            stop_after=stop_after,
        )
    return CorrectedBuildResult(
        decision=result.decision,
        frames=result.frames,
        report=result.report,
        manifest_sha256=result.manifest_sha256,
        protocol_hash=result.protocol_hash,
        executed_this_run=result.executed_this_run,
        full_run_authorized=result.full_run_authorized,
        blocked_reason=result.blocked_reason,
    )


def _print_summary(result: CorrectedBuildResult) -> None:
    status = result.frames["request_execution_status"]
    terminal = status.loc[~status["execution_status"].eq("NOT_EXECUTED")]
    static = "PASS" if result.frames["static_preflight"]["status"].eq("pass").all() else "FAIL"
    synthetic = "PASS" if result.frames["synthetic_tests"]["pass"].eq("true").all() else "FAIL"
    context = "PASS" if result.frames["context_preflight"]["status"].eq("pass").all() else "FAIL"
    print(f"FINAL_DECISION={result.decision}")
    print(f"MANIFEST_SHA_VERIFIED={base.bool_text(result.manifest_sha256 == EXPECTED_R5A_MANIFEST_SHA256)}")
    print(f"CORRECTED_PROTOCOL_HASH={result.protocol_hash}")
    print(f"STATIC_PREFLIGHT={static}")
    print(f"SYNTHETIC={synthetic}")
    print(f"CONTEXT_PREFLIGHT={context}")
    print(f"FULL RUN AUTHORIZED: {'YES' if result.full_run_authorized else 'NO'}")
    print("REQUESTS_EXPECTED=2400")
    print(f"REQUESTS_ACCOUNTED={len(terminal)}")
    print(f"VALID={int(terminal['parse_valid'].eq('true').sum())}")
    print(f"INVALID={int(len(terminal) - terminal['parse_valid'].eq('true').sum())}")
    print(f"OUT_OF_RANGE={int(terminal['validity_class'].eq('OUT_OF_RANGE').sum())}")
    print(f"WRONG_LENGTH={int(terminal['validity_class'].eq('WRONG_LENGTH').sum())}")
    print(f"PARSE_FAILURE={int(terminal['validity_class'].eq('PARSE_FAILURE').sum())}")
    print(f"BACKEND_INCOMPLETE={int(terminal['validity_class'].eq('BACKEND_INCOMPLETE').sum())}")
    print(f"EXECUTED_THIS_RUN={result.executed_this_run}")
    if result.blocked_reason:
        print(f"blocked_reason={result.blocked_reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--run-gated", action="store_true")
    parser.add_argument("--stop-after", choices=["canary", "early", "full"], default="full")
    parser.add_argument("--allow-api-calls", action="store_true", help="Required before any local LLM/Ollama inference call.")
    parser.add_argument("--allow-local-ollama", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    project_root = Path(args.project_root or config.get("project_root", ".")).resolve()
    try:
        result = build_r5c_history_length_corrected(
            project_root,
            config_path=config_path,
            write=not args.no_write,
            run_gated=args.run_gated,
            allow_api_calls=args.allow_api_calls,
            allow_local_ollama=args.allow_local_ollama,
            stop_after=args.stop_after,
        )
    except base.R5CHistoryLengthError as exc:
        print(DECISION_EXECUTION_FAILED)
        print(f"blocked_reason={exc}")
        return 2
    _print_summary(result)
    return 0 if result.decision in {DECISION_COMPLETE, DECISION_PREFLIGHT_READY, DECISION_CANARY_PASS, DECISION_EARLY_PASS} else 1


if __name__ == "__main__":
    raise SystemExit(main())
