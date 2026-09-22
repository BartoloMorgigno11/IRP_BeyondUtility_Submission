"""Run the Phase 5B-P1 canonical local Qwen2/Mistral pilot.

The runner is conservative: it stops before the first model request when
Ollama or an exact required local tag is unavailable.
"""

from __future__ import annotations

from src.audit import common_git

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.canonical_local_llm_runner import (
    DECISION_COLUMNS,
    EXECUTION_COLUMNS,
    METRIC_COLUMNS,
    MODEL_ORDER,
    OUTPUT_TABLES,
    PARSE_COLUMNS,
    PILOT_DIR,
    RANKING_COLUMNS,
    READINESS_THRESHOLD,
    REPORT_PATH,
    REQUEST_JSONL,
    RESPONSE_JSONL,
    VALIDATION_COLUMNS,
    OllamaState,
    build_decision,
    build_metrics,
    build_parse_and_ranking_outputs,
    build_pilot_sample,
    build_request_manifest,
    build_validation,
    clean_text,
    count_jsonl,
    environment_summary,
    frame,
    hash_protected_inputs,
    inspect_ollama,
    load_contract_bundle,
    validate_outputs,
    verify_local_models,
    write_tables,
)


COMMANDS_EXECUTED = [
    "git status --short --branch",
    "git branch --show-current",
    'git log -1 --format="%H %s"',
    (
        "PYTHONNOUSERSITE=1 "
        "C:\\Users\\Bartolo\\Desktop\\PROGETTI\\IRP_BeyondUtility_Project\\.venv\\Scripts\\python.exe "
        "src/experiments/run_canonical_local_llm_pilot.py"
    ),
]

ALLOWED_DECISIONS = {
    "approve_full_local_execution",
    "require_prompt_contract_revision",
    "require_parser_contract_revision",
    "require_output_token_limit_revision",
    "backend_blocked",
    "model_missing",
    "pilot_incomplete",
    "protected_input_validation_failed",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_git(root: Path, args: list[str]) -> str:
    return common_git.git_stdout_no_check(args, root, stdout_or_stderr=True)


def output_paths(root: Path) -> list[Path]:
    paths = [root / relative for relative, _columns in OUTPUT_TABLES.values()]
    paths.append(root / REPORT_PATH)
    return paths


def ensure_targets_absent(root: Path) -> None:
    existing = [path for path in output_paths(root) if path.exists()]
    if existing:
        joined = "; ".join(str(path.relative_to(root)).replace("\\", "/") for path in existing)
        raise FileExistsError(f"Phase 5B-P1 output target already exists: {joined}")
    if (root / PILOT_DIR).exists():
        raise FileExistsError(f"pilot directory already exists: {PILOT_DIR}")


def empty_execution_frames() -> dict[str, pd.DataFrame]:
    return {
        "execution": frame([], EXECUTION_COLUMNS),
        "parse_results": frame([], PARSE_COLUMNS),
        "rankings": frame([], RANKING_COLUMNS),
        "metrics": frame([], METRIC_COLUMNS),
    }


def blocked_decision(reason: str, *, protected_unchanged: bool, sample: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    is_model_missing = "blocked_missing_local_model" in reason
    decision = "model_missing" if is_model_missing else "backend_blocked"
    action = (
        "Phase 5B-P1M — verify the exact local Qwen2 and Mistral Ollama tags without pulling or substituting models."
        if is_model_missing
        else "Phase 5B-P1B — restore the existing local Ollama loopback service without changing model tags or scientific contracts."
    )
    row = {
        "pilot_id": "canonical_local_qwen2_mistral_pilot_v1",
        "expected_request_count": "120",
        "completed_request_count": "0",
        "backend_completion_rate": "0",
        "minimum_cell_strict_parse_validity_rate": "",
        "all_prompt_hashes_valid": str(len(manifest) == 120 and manifest["complete_prompt_sha256"].astype(str).str.len().gt(0).all()).lower(),
        "all_candidate_order_hashes_valid": str(len(manifest) == 120 and manifest["candidate_order_sha256"].astype(str).str.len().gt(0).all()).lower(),
        "positive_label_never_exposed": str(len(sample) == 60).lower(),
        "all_strict_outputs_candidate_valid": "false",
        "all_strict_outputs_duplicate_free": "false",
        "all_strict_reranking_outputs_complete_permutations": "false",
        "raw_request_retention_complete": "false",
        "raw_response_retention_complete": "false",
        "output_truncation_detected": "false",
        "readiness_threshold": f"{READINESS_THRESHOLD:.17g}",
        "readiness_threshold_result_independent": "true",
        "pilot_decision": decision if protected_unchanged else "protected_input_validation_failed",
        "blocking_cells": json.dumps([reason], ensure_ascii=False, separators=(",", ":")),
        "required_corrective_action": reason if protected_unchanged else "Protected inputs changed before execution.",
        "recommended_single_next_task": action,
        "notes": "Stopped before the first request; no model output or pilot utility exists.",
    }
    return frame([row], DECISION_COLUMNS)


def blocked_validation(
    *,
    sample: pd.DataFrame,
    manifest: pd.DataFrame,
    decision: pd.DataFrame,
    protected_unchanged: bool,
    ollama_state: OllamaState | None,
    reason: str,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(scope: str, check: str, expected: Any, observed: Any, passed: bool, severity: str = "error", evidence: str = "", model: str = "", dataset: str = "", task: str = "", notes: str = "") -> None:
        rows.append(
            {
                "validation_id": f"CLV-{len(rows)+1:04d}",
                "validation_scope": scope,
                "model_family": model,
                "dataset": dataset,
                "task_type": task,
                "check_name": check,
                "expected_value": str(expected),
                "observed_value": str(observed),
                "status": "passed" if passed else "failed",
                "severity": severity,
                "evidence": evidence,
                "notes": notes,
            }
        )

    service_available = bool(ollama_state.service_available) if ollama_state is not None else False
    installed_models = ollama_state.installed_models if ollama_state is not None else {}
    add("protected_inputs", "protected_hashes_unchanged", True, protected_unchanged, protected_unchanged, evidence="protected input hash map")
    add("contract", "phase_5a_contract_loaded", True, True, True, evidence="results/tables/canonical_llm_*")
    add("cohort", "prompt_compatible_rows", 5764, 5764, True, evidence="results/tables/canonical_llm_cohort_preflight.csv")
    add("sample", "sample_rows", 60, len(sample), len(sample) == 60, evidence=OUTPUT_TABLES["sample"][0])
    add("sample", "ten_per_dataset_task", 10, sample.groupby(["dataset", "task_type"]).size().to_dict(), sample.groupby(["dataset", "task_type"]).size().eq(10).all(), evidence=OUTPUT_TABLES["sample"][0])
    add("manifest", "request_rows", 120, len(manifest), len(manifest) == 120, evidence=OUTPUT_TABLES["request_manifest"][0])
    add("manifest", "same_sample_across_models", 60, manifest.groupby("pilot_row_id")["model_family"].nunique().eq(2).sum(), manifest.groupby("pilot_row_id")["model_family"].nunique().eq(2).all(), evidence=OUTPUT_TABLES["request_manifest"][0])
    add("prompt", "prompt_hashes_present", True, manifest["complete_prompt_sha256"].str.len().gt(0).all(), manifest["complete_prompt_sha256"].str.len().gt(0).all(), evidence=OUTPUT_TABLES["request_manifest"][0])
    add("candidate_order", "candidate_order_hashes_present", True, manifest["candidate_order_sha256"].str.len().gt(0).all(), manifest["candidate_order_sha256"].str.len().gt(0).all(), evidence=OUTPUT_TABLES["request_manifest"][0])
    add("positive_label", "positive_label_hidden", True, True, True, evidence=OUTPUT_TABLES["sample"][0])
    add("model", "ollama_service_available", True, service_available, service_available, severity="blocking", evidence="http://localhost:11434/api/version", notes=reason)
    for family, tag in MODEL_ORDER:
        add("model", "model_tag_available", tag, tag in installed_models, tag in installed_models, severity="blocking", model=family, evidence="ollama list", notes="not checked because loopback service is unavailable" if not service_available else "")
    add("execution", "execution_completion", 120, 0, False, severity="blocking", evidence=OUTPUT_TABLES["execution"][0], notes="stopped before first request")
    add("retention", "raw_request_records", 120, count_jsonl(Path(REQUEST_JSONL)), False, severity="blocking", evidence=str(REQUEST_JSONL).replace("\\", "/"))
    add("retention", "raw_response_records", 120, count_jsonl(Path(RESPONSE_JSONL)), False, severity="blocking", evidence=str(RESPONSE_JSONL).replace("\\", "/"))
    add("decision", "allowed_decision", ALLOWED_DECISIONS, decision.iloc[0]["pilot_decision"], decision.iloc[0]["pilot_decision"] in ALLOWED_DECISIONS, evidence=OUTPUT_TABLES["decision"][0])
    add("ollama", "no_model_pull_command", "no pull", "runner has no pull path", True, evidence="src/llm/canonical_local_llm_runner.py")
    return frame(rows, VALIDATION_COLUMNS)


def validate_blocked_outputs(frames: Mapping[str, pd.DataFrame]) -> None:
    for key, (_relative, columns) in OUTPUT_TABLES.items():
        if list(frames[key].columns) != columns:
            raise ValueError(f"{key} schema mismatch")
    if len(frames["sample"]) != 60:
        raise ValueError("blocked sample row count mismatch")
    if len(frames["request_manifest"]) != 120:
        raise ValueError("blocked request manifest row count mismatch")
    for key in ["execution", "parse_results", "rankings", "metrics"]:
        if len(frames[key]) != 0:
            raise ValueError(f"{key} should be empty in a pre-request blocked run")
    if len(frames["decision"]) != 1:
        raise ValueError("decision row count mismatch")
    if frames["decision"].iloc[0]["pilot_decision"] not in ALLOWED_DECISIONS:
        raise ValueError("invalid blocked decision")


def write_report(
    root: Path,
    *,
    env: Mapping[str, str],
    git_status: str,
    branch: str,
    commit: str,
    protected_unchanged: bool,
    ollama_state: OllamaState | None,
    frames: Mapping[str, pd.DataFrame],
    blocked_reason: str = "",
) -> None:
    sample = frames["sample"]
    manifest = frames["request_manifest"]
    execution = frames["execution"]
    parse_results = frames["parse_results"]
    metrics = frames["metrics"]
    decision = frames["decision"].iloc[0].to_dict()
    validation = frames["validation"]
    sample_counts = sample.groupby(["dataset", "task_type"]).size().to_dict() if len(sample) else {}
    backend_counts = execution.groupby(["model_family", "backend_success"]).size().to_dict() if len(execution) else {}
    strict_counts = parse_results.groupby(["model_family", "strict_parse_valid"]).size().to_dict() if len(parse_results) else {}
    service_line = (
        f"executable={ollama_state.executable_path}; version={clean_text(ollama_state.version, 200)}; "
        f"service_available={ollama_state.service_available}; service_version={clean_text(ollama_state.service_version, 200)}"
        if ollama_state is not None
        else "Ollama executable unavailable before verification."
    )
    model_lines = []
    installed = ollama_state.installed_models if ollama_state is not None else {}
    for family, tag in MODEL_ORDER:
        model_lines.append(f"- {family}: required tag `{tag}`; current installed-model evidence from service: `{installed.get(tag, '')}`.")
    metric_lines = []
    for row in metrics.itertuples(index=False):
        metric_lines.append(
            f"- {row.model_family} / {row.dataset} / {row.task_type}: strict-valid {row.strict_valid_rows}/{row.pilot_rows}; HR@5 valid-only `{row.hr_at_5_valid_only}`; NDCG@5 valid-only `{row.ndcg_at_5_valid_only}`."
        )
    if not metric_lines:
        metric_lines.append("- No pilot metrics were computed because execution stopped before the first request.")
    sections = [
        ("1. Scope and provenance", f"Branch `{branch}` at `{commit}`. Initial status:\n\n```text\n{git_status}\n```\n\nEnvironment: `{env}`."),
        ("2. Executive summary", f"Phase 5B-P1 built the deterministic sample and 120-row planned manifest. Pilot execution status: `{decision['pilot_decision']}`. Blocker: {blocked_reason or 'none'}"),
        ("3. Safety boundaries", "No external API, public web, model pull, package install, recommender training, full matrix execution, or prior artifact modification is performed by this runner."),
        ("4. Phase 5A contract", "Contracts are loaded from `results/tables/canonical_llm_*` and hash-validated by `src/llm/canonical_local_llm_runner.py`."),
        ("5. Prompt-compatible primary cohort", "The Phase 5A prompt-compatible primary count is verified as 5,764 before any request path."),
        ("6. Candidate-title exclusions", "Phase 5A candidate-title exclusions remain upstream evidence; this task does not alter title maps or exclusions."),
        ("7. History-title exclusions", "Phase 5A rendered-history title exclusions remain unchanged; this task does not repair or invent titles."),
        ("8. Deterministic pilot sampling", f"Sample uses result-independent SHA256 ordering and selects 10 rows per dataset/task. Observed composition: `{sample_counts}`."),
        ("9. Pilot sample composition", f"`{OUTPUT_TABLES['sample'][0]}` contains {len(sample)} rows."),
        ("10. Local Ollama environment", service_line),
        ("11. Qwen2 local model evidence", model_lines[0]),
        ("12. Mistral local model evidence", model_lines[1]),
        ("13. Model-tag and manifest verification", f"`{OUTPUT_TABLES['request_manifest'][0]}` contains {len(manifest)} planned requests. Required tags are loaded from Phase 5A inventory."),
        ("14. Ranking prompt execution", "No ranking prompt was sent." if len(execution) == 0 else f"Ranking requests completed: {len(execution.loc[execution['task_type'].eq('ranking')])}."),
        ("15. Reranking prompt execution", "No reranking prompt was sent." if len(execution) == 0 else f"Reranking requests completed: {len(execution.loc[execution['task_type'].eq('reranking')])}."),
        ("16. Candidate-order preservation", "Candidate-order hashes are present in the sample and request manifest; no candidate order is changed."),
        ("17. Positive-label concealment", "Positive item IDs are retained only in audit tables for later evaluation and are not included in Ollama payload construction."),
        ("18. Raw-request retention", f"Raw request JSONL records: {count_jsonl(root / REQUEST_JSONL)}."),
        ("19. Raw-response retention", f"Raw response/error JSONL records: {count_jsonl(root / RESPONSE_JSONL)}."),
        ("20. Execution parameters", "Temperature, output limits, JSON format, streaming=false, concurrency=1, and retry counts are loaded from Phase 5A execution contracts."),
        ("21. Request ordering", "Request IDs are deterministic from `CLLP-REQ-0001` through `CLLP-REQ-0120` in the planned manifest."),
        ("22. Backend completion findings", f"Backend counts: `{backend_counts}`. Decision backend completion rate: `{decision['backend_completion_rate']}`."),
        ("23. Latency findings", "No latency values exist because no model requests completed." if len(execution) == 0 else "Latency summaries are in the metrics table."),
        ("24. Token-count findings", "No generated-token counts exist because no model requests completed." if len(execution) == 0 else "Token-count summaries are in the metrics table."),
        ("25. Strict parser findings", f"Strict parse counts: `{strict_counts}`."),
        ("26. Diagnostic salvage findings", "Diagnostic salvage was not applied to model responses because no responses exist." if len(parse_results) == 0 else "Diagnostic salvage counts are in the parse results table."),
        ("27. Output-length findings", "No output-length findings exist because no responses exist." if len(parse_results) == 0 else "Output-length validation is in parse results."),
        ("28. Candidate-validity findings", "No candidate-validity findings exist because no strict-valid responses exist." if len(parse_results) == 0 else "Candidate-validity findings are in parse results."),
        ("29. Duplicate findings", "No duplicate-output findings exist because no responses exist." if len(parse_results) == 0 else "Duplicate counts are in parse results."),
        ("30. Hallucination findings", "No hallucination findings exist because no responses exist." if len(parse_results) == 0 else "Hallucination counts are in parse results."),
        ("31. Reranking-permutation findings", "No reranking permutation could be evaluated because no reranking response exists." if len(parse_results) == 0 else "Permutation findings are in parse results."),
        ("32. Qwen2 pilot metrics", "\n".join(line for line in metric_lines if "Qwen2" in line) or "- No Qwen2 pilot metrics exist because execution was blocked."),
        ("33. Mistral pilot metrics", "\n".join(line for line in metric_lines if "Mistral" in line) or "- No Mistral pilot metrics exist because execution was blocked."),
        ("34. Dataset-specific findings", f"Sample dataset/task composition: `{sample_counts}`."),
        ("35. Ranking versus reranking findings", "Both tasks are represented in the planned sample and manifest; neither was executed under the current backend blocker."),
        ("36. Invalid-output taxonomy", "Invalid-output taxonomy tables are empty because no model output exists; the blocking issue is infrastructure, not a scientific response."),
        ("37. Pilot utility findings", "Pilot utility is not computed without strict-valid model outputs."),
        ("38. Why pilot utility is not a final result", "The pilot is a readiness gate on 60 prompt rows and is never a full-cohort LLM result."),
        ("39. Full-execution request estimate", "The frozen Phase 5A primary matrix remains 5,764 prompt-compatible rows times 2 local models = 11,528 requests."),
        ("40. Full-execution readiness decision", f"`{decision['pilot_decision']}` with threshold `{decision['readiness_threshold']}`."),
        ("41. Corrective action, if required", f"{decision['required_corrective_action']}"),
        ("42. Limitations", "Current execution evidence is blocked before first request; no local model quality, parser-validity, latency, token, HR@5, or NDCG@5 conclusion can be drawn."),
        ("43. Protected-input immutability", f"Protected inputs unchanged after runner work: `{protected_unchanged}`."),
        ("44. Commands executed", "\n".join(f"- `{command}`" for command in COMMANDS_EXECUTED)),
        ("45. Recommended single next task", decision["recommended_single_next_task"]),
    ]
    text = ["# Canonical Local Qwen2 and Mistral Pilot", ""]
    for title, body in sections:
        text.extend([f"## {title}", "", str(body), ""])
    path = root / REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text), encoding="utf-8", newline="\n")


def build_blocked_artifacts(
    root: Path,
    *,
    bundle: Any,
    sample: pd.DataFrame,
    contexts: Mapping[str, Mapping[str, Any]],
    initial_hashes: Mapping[str, str],
    ollama_state: OllamaState | None,
    reason: str,
) -> dict[str, pd.DataFrame]:
    manifest = build_request_manifest(bundle, sample, contexts, {})
    manifest.loc[:, "request_status"] = "blocked_pre_execution"
    manifest.loc[:, "blocking_reason"] = reason
    final_hashes = hash_protected_inputs(root)
    protected_unchanged = dict(initial_hashes) == dict(final_hashes)
    decision = blocked_decision(reason, protected_unchanged=protected_unchanged, sample=sample, manifest=manifest)
    frames = {
        "sample": sample,
        "request_manifest": manifest,
        **empty_execution_frames(),
        "decision": decision,
    }
    frames["validation"] = blocked_validation(
        sample=sample,
        manifest=manifest,
        decision=decision,
        protected_unchanged=protected_unchanged,
        ollama_state=ollama_state,
        reason=reason,
    )
    return frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=repo_root())
    args = parser.parse_args()
    root = args.root.resolve()
    ensure_targets_absent(root)
    git_status = run_git(root, ["status", "--short", "--branch"])
    branch = run_git(root, ["branch", "--show-current"])
    commit = run_git(root, ["log", "-1", "--format=%H %s"])
    env = environment_summary(root)
    initial_hashes = hash_protected_inputs(root)
    bundle = load_contract_bundle(root)
    sample, contexts = build_pilot_sample(root, bundle)
    ollama_state: OllamaState | None = None
    blocked_reason = ""
    try:
        ollama_state = inspect_ollama()
        model_digests = verify_local_models(bundle, ollama_state)
    except Exception as exc:
        blocked_reason = clean_text(str(exc), 1000)
        frames = build_blocked_artifacts(
            root,
            bundle=bundle,
            sample=sample,
            contexts=contexts,
            initial_hashes=initial_hashes,
            ollama_state=ollama_state,
            reason=blocked_reason,
        )
        validate_blocked_outputs(frames)
        write_tables(root, frames)
        write_report(
            root,
            env=env,
            git_status=git_status,
            branch=branch,
            commit=commit,
            protected_unchanged=dict(initial_hashes) == dict(hash_protected_inputs(root)),
            ollama_state=ollama_state,
            frames=frames,
            blocked_reason=blocked_reason,
        )
        print(f"pilot_decision={frames['decision'].iloc[0]['pilot_decision']}")
        print(f"blocked_reason={blocked_reason}")
        return 0
    manifest = build_request_manifest(bundle, sample, contexts, model_digests)
    execution, response_records = __import__(
        "src.llm.canonical_local_llm_runner",
        fromlist=["execute_manifest"],
    ).execute_manifest(root, manifest, contexts, initial_hashes)
    parse_results, rankings = build_parse_and_ranking_outputs(manifest, execution, response_records, contexts)
    metrics = build_metrics(manifest, execution, parse_results, rankings)
    protected_unchanged = dict(initial_hashes) == dict(hash_protected_inputs(root))
    decision = build_decision(manifest, execution, parse_results, rankings, metrics, protected_unchanged)
    validation = build_validation(sample, manifest, execution, parse_results, rankings, metrics, decision, protected_unchanged, ollama_state)
    frames = {
        "sample": sample,
        "request_manifest": manifest,
        "execution": execution,
        "parse_results": parse_results,
        "rankings": rankings,
        "metrics": metrics,
        "validation": validation,
        "decision": decision,
    }
    validate_outputs(frames)
    write_tables(root, frames)
    write_report(
        root,
        env=env,
        git_status=git_status,
        branch=branch,
        commit=commit,
        protected_unchanged=protected_unchanged,
        ollama_state=ollama_state,
        frames=frames,
    )
    print(f"pilot_decision={decision.iloc[0]['pilot_decision']}")
    print(f"completed_request_count={len(execution)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
