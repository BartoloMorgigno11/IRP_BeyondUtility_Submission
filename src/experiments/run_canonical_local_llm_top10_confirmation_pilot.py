"""Run/resume the Phase 5B-P1G top-10 reranking confirmation pilot."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.canonical_local_llm_top10_confirmation_runner import (  # noqa: E402
    EVENT_JSONL,
    REQUEST_JSONL,
    RESPONSE_JSONL,
    build_all_result_tables,
    environment_summary,
    execute_pending_requests,
    hash_protected_inputs,
    load_inputs,
    output_targets,
    pending_requests_in_order,
    read_jsonl,
    state_summary,
    utc_now,
    validate_written_outputs,
    verify_ollama_admin,
    write_report,
    write_tables,
)


COMMANDS_EXECUTED = [
    "git status --short --branch",
    "git branch --show-current",
    'git log -1 --format=\"%H %s\"',
    "$env:PYTHONNOUSERSITE='1'; & 'C:\\Users\\Bartolo\\Desktop\\PROGETTI\\IRP_BeyondUtility_Project\\.venv\\Scripts\\python.exe' -m pytest tests\\test_canonical_local_llm_top10_confirmation_runner.py ...",
    "$env:PYTHONNOUSERSITE='1'; & 'C:\\Users\\Bartolo\\Desktop\\PROGETTI\\IRP_BeyondUtility_Project\\.venv\\Scripts\\python.exe' src\\experiments\\run_canonical_local_llm_top10_confirmation_pilot.py",
]


def _require_environment(root: Path) -> None:
    expected_python = (root / ".venv" / "Scripts" / "python.exe").resolve()
    actual_python = Path(sys.executable).resolve()
    if actual_python != expected_python:
        raise RuntimeError(f"expected repo-local venv interpreter {expected_python}, got {actual_python}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("PYTHONNOUSERSITE=1 is required")


def main() -> int:
    _require_environment(ROOT)
    inputs = load_inputs(ROOT)
    before_hashes = dict(inputs.protected_hashes)
    if hash_protected_inputs(ROOT) != before_hashes:
        raise RuntimeError("protected inputs changed before top10 confirmation execution")
    request_records = read_jsonl(ROOT / REQUEST_JSONL)
    response_records = read_jsonl(ROOT / RESPONSE_JSONL)
    initial_request_count = len(request_records)
    initial_response_count = len(response_records)
    initial_summary = state_summary(inputs.manifest, request_records, response_records)
    if initial_summary.get("ambiguous_interrupted", 0) or initial_summary.get("invalid_duplicate_evidence", 0):
        raise RuntimeError(f"unsafe initial resume state: {initial_summary}")
    pending_requests_in_order(inputs.manifest, request_records, response_records)
    started_at = utc_now()
    admin = verify_ollama_admin(ROOT, inputs.recovery)
    execute_pending_requests(ROOT, inputs)
    completed_at = utc_now()
    protected_unchanged = hash_protected_inputs(ROOT) == before_hashes
    frames = build_all_result_tables(
        ROOT,
        inputs,
        admin,
        started_at,
        completed_at,
        initial_summary,
        initial_request_count,
        initial_response_count,
        protected_unchanged,
    )
    write_tables(ROOT, frames)
    write_report(ROOT, frames, environment_summary(ROOT), COMMANDS_EXECUTED)
    validate_written_outputs(ROOT, frames)
    if hash_protected_inputs(ROOT) != before_hashes:
        raise RuntimeError("protected inputs changed during Phase 5B-P1G top10 confirmation")
    decision = frames["decision"].iloc[0].to_dict()
    print("canonical_local_llm_top10_confirmation_pilot")
    print(f"completed={decision['completed_request_count']}")
    print(f"backend_completion_rate={decision['backend_completion_rate']}")
    print(f"minimum_cell_strict_parse_validity_rate={decision['minimum_cell_strict_parse_validity_rate']}")
    print(f"cells_passing_threshold={decision['cells_passing_threshold']}/{decision['total_readiness_cells']}")
    print(f"decision={decision['pilot_decision']}")
    print(f"raw_requests={REQUEST_JSONL.as_posix()}")
    print(f"raw_responses={RESPONSE_JSONL.as_posix()}")
    print(f"runner_events={EVENT_JSONL.as_posix()}")
    for path in output_targets(ROOT):
        print(path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
