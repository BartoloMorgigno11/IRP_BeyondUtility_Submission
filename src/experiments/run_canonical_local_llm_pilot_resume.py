"""Run Phase 5B-P1C resume of the frozen canonical local LLM pilot."""

from __future__ import annotations

from src.audit import common_git

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.canonical_local_llm_pilot_resume import (
    EVENT_JSONL,
    REQUEST_JSONL,
    RESPONSE_JSONL,
    RESUME_DIR,
    OUTPUT_TABLES,
    REPORT_PATH,
    build_all_result_tables,
    build_decision,
    build_provenance,
    build_validation,
    environment_summary,
    hash_protected_inputs,
    load_resume_inputs,
    pending_requests_in_order,
    read_jsonl,
    read_table,
    state_summary,
    utc_now,
    validate_output_frames,
    verify_ollama_admin,
    write_report,
    write_tables,
    execute_pending_requests,
)


COMMANDS_EXECUTED = [
    "git status --short --branch",
    "git branch --show-current",
    'git log -1 --format="%H %s"',
    (
        "PYTHONNOUSERSITE=1 "
        "C:\\Users\\Bartolo\\Desktop\\PROGETTI\\IRP_BeyondUtility_Project\\.venv\\Scripts\\python.exe "
        "-m pytest tests\\test_canonical_local_llm_pilot_resume.py "
        "tests\\test_canonical_local_llm_runner.py "
        "tests\\test_canonical_llm_contract.py "
        "tests\\test_experiment_cohort_policy.py "
        "tests\\test_dataset_candidate_set_validation.py"
    ),
    (
        "PYTHONNOUSERSITE=1 "
        "C:\\Users\\Bartolo\\Desktop\\PROGETTI\\IRP_BeyondUtility_Project\\.venv\\Scripts\\python.exe "
        "src\\experiments\\run_canonical_local_llm_pilot_resume.py"
    ),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_git(root: Path, args: list[str]) -> str:
    return common_git.git_stdout_no_check(args, root, stdout_or_stderr=True)


def expected_python(root: Path) -> Path:
    return root / ".venv" / "Scripts" / "python.exe"


def assert_required_interpreter(root: Path) -> None:
    observed = Path(sys.executable).resolve()
    expected = expected_python(root).resolve()
    if os.path.normcase(str(observed)) != os.path.normcase(str(expected)):
        raise RuntimeError(f"wrong Python executable: observed={observed}; expected={expected}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("PYTHONNOUSERSITE must be set to 1")


def ensure_resume_directory_valid(root: Path) -> None:
    directory = root / RESUME_DIR
    if not directory.exists():
        return
    allowed = {REQUEST_JSONL.name, RESPONSE_JSONL.name, EVENT_JSONL.name}
    unexpected = sorted(path.name for path in directory.iterdir() if path.is_file() and path.name not in allowed)
    if unexpected:
        raise RuntimeError(f"unexpected files in resume directory: {unexpected}")


def output_paths(root: Path) -> list[Path]:
    return [root / relative for relative, _columns in OUTPUT_TABLES.values()] + [root / REPORT_PATH]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repo_root(), help="Repository root; defaults to this checkout.")
    args = parser.parse_args(argv)
    root = args.root.resolve()

    assert_required_interpreter(root)
    ensure_resume_directory_valid(root)

    print(run_git(root, ["status", "--short", "--branch"]), flush=True)
    print(run_git(root, ["branch", "--show-current"]), flush=True)
    print(run_git(root, ["log", "-1", "--format=%H %s"]), flush=True)

    started_at = utc_now()
    initial_hashes = hash_protected_inputs(root)
    inputs = load_resume_inputs(root)
    if initial_hashes != inputs.protected_hashes:
        raise RuntimeError("protected_input_validation_failed: initial hash drift during input load")

    initial_request_records = read_jsonl(root / REQUEST_JSONL)
    initial_response_records = read_jsonl(root / RESPONSE_JSONL)
    initial_summary = state_summary(inputs.manifest, initial_request_records, initial_response_records)
    initial_request_count = len(initial_request_records)
    initial_response_count = len(initial_response_records)
    provenance_summary = initial_summary
    provenance_request_count = initial_request_count
    provenance_response_count = initial_response_count
    provenance_started_at = started_at
    existing_provenance_path = root / OUTPUT_TABLES["provenance"][0]
    if existing_provenance_path.exists():
        existing_provenance = read_table(root, OUTPUT_TABLES["provenance"][0])
        if len(existing_provenance) == 1:
            existing_row = existing_provenance.iloc[0].to_dict()
            provenance_request_count = int(existing_row.get("initial_request_jsonl_records", initial_request_count) or 0)
            provenance_response_count = int(existing_row.get("initial_response_jsonl_records", initial_response_count) or 0)
            provenance_summary = {
                "completed": int(existing_row.get("initial_completed_requests", initial_summary.get("completed", 0)) or 0),
                "not_started": int(existing_row.get("initial_pending_requests", initial_summary.get("not_started", 0)) or 0),
                "ambiguous_interrupted": int(existing_row.get("initial_ambiguous_requests", initial_summary.get("ambiguous_interrupted", 0)) or 0),
                "invalid_duplicate_evidence": initial_summary.get("invalid_duplicate_evidence", 0),
            }
            provenance_started_at = existing_row.get("execution_started_at_utc", started_at) or started_at

    pending = pending_requests_in_order(inputs.manifest, initial_request_records, initial_response_records)
    print(
        "Initial resume state: "
        f"completed={initial_summary.get('completed', 0)} "
        f"pending={initial_summary.get('not_started', 0)} "
        f"ambiguous={initial_summary.get('ambiguous_interrupted', 0)} "
        f"invalid={initial_summary.get('invalid_duplicate_evidence', 0)}",
        flush=True,
    )
    print(f"Pending frozen requests before execution: {len(pending)}", flush=True)

    admin = verify_ollama_admin(root, inputs.recovery)
    print(f"Ollama server version: {admin.server_version}", flush=True)

    execute_pending_requests(root, inputs, admin)

    result_frames = build_all_result_tables(root, inputs)
    protected_unchanged = hash_protected_inputs(root) == initial_hashes
    digest_ok = True
    decision = build_decision(
        result_frames["request_state"],
        result_frames["execution"],
        result_frames["parse_results"],
        result_frames["rankings"],
        result_frames["metrics"],
        protected_unchanged=protected_unchanged,
        digest_ok=digest_ok,
    )
    validation = build_validation(
        result_frames["request_state"],
        result_frames["execution"],
        result_frames["parse_results"],
        result_frames["rankings"],
        result_frames["metrics"],
        decision,
        protected_unchanged,
        digest_ok,
    )
    completed_at = utc_now()
    decision_row = decision.iloc[0].to_dict()
    provenance = build_provenance(
        root,
        inputs,
        admin,
        provenance_summary,
        provenance_request_count,
        provenance_response_count,
        provenance_started_at,
        completed_at,
        "completed" if decision_row["completed_request_count"] == "120" else "blocked",
        "" if decision_row["pilot_decision"] == "approve_full_local_execution" else decision_row["required_corrective_action"],
    )

    frames = {
        "provenance": provenance,
        **result_frames,
        "validation": validation,
        "decision": decision,
    }
    validate_output_frames(frames)
    write_tables(root, frames)
    write_report(root, frames, environment_summary(root), COMMANDS_EXECUTED)

    print(f"Decision: {decision_row['pilot_decision']}", flush=True)
    print(f"Completed requests: {decision_row['completed_request_count']}", flush=True)
    print(f"Protected inputs unchanged: {protected_unchanged}", flush=True)
    print("Wrote Phase 5B-P1C outputs:", flush=True)
    for path in output_paths(root):
        print(f"- {path.relative_to(root)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
