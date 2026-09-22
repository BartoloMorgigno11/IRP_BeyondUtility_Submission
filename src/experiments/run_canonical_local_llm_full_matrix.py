"""Run Phase 5B-P2 full canonical local LLM matrix chunks."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.canonical_local_llm_full_matrix_runner import (
    EXPECTED_REQUESTS,
    REQUEST_JSONL,
    RESPONSE_JSONL,
    build_all_result_tables,
    execute_pending_requests,
    load_inputs,
    pending_requests_in_order,
    read_jsonl,
    state_summary,
    utc_now,
    validate_output_files,
    verify_ollama_admin,
    write_report,
    write_tables,
)


REQUIRED_PYTHON = Path(os.environ.get("CANONICAL_LLM_REQUIRED_PYTHON", str(ROOT / ".venv" / "Scripts" / "python.exe")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-requests", type=int, default=256, help="Maximum new pending requests to execute in this invocation.")
    parser.add_argument("--report-only", action="store_true", help="Rebuild derived tables and report without sending new requests.")
    return parser.parse_args()


def assert_required_environment() -> None:
    if Path(sys.executable).resolve() != REQUIRED_PYTHON.resolve():
        raise RuntimeError(f"Phase 5B-P2 must use {REQUIRED_PYTHON}; observed {sys.executable}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("PYTHONNOUSERSITE=1 is required")


def main() -> int:
    assert_required_environment()
    args = parse_args()
    root = Path.cwd()
    inputs = load_inputs(root)
    admin = verify_ollama_admin(root, inputs.recovery)
    request_records_before = read_jsonl(root / REQUEST_JSONL)
    response_records_before = read_jsonl(root / RESPONSE_JSONL)
    initial_summary = state_summary(inputs.manifest, request_records_before, response_records_before)
    started_at = utc_now()
    max_new = 0 if args.report_only else int(args.max_new_requests)
    sent = execute_pending_requests(root, inputs, max_new)
    completed_at = utc_now()
    frames = build_all_result_tables(
        root,
        inputs,
        admin,
        started_at,
        completed_at,
        initial_summary,
        len(request_records_before),
        len(response_records_before),
        max_new,
        sent,
    )
    write_tables(root, frames)
    command_prefix = f"$env:PYTHONNOUSERSITE='1'; & '{sys.executable}'"
    write_report(
        root,
        frames,
        [
            f"{command_prefix} src/experiments/run_canonical_local_llm_full_matrix.py --max-new-requests 256",
            f"{command_prefix} src/experiments/run_canonical_local_llm_full_matrix.py --report-only",
        ],
    )
    validate_output_files(root)
    request_records_after = read_jsonl(root / REQUEST_JSONL)
    response_records_after = read_jsonl(root / RESPONSE_JSONL)
    pending = pending_requests_in_order(inputs.manifest, request_records_after, response_records_after)
    decision = frames["decision"].iloc[0]
    print(f"sent_this_invocation={sent}", flush=True)
    print(f"completed={decision['completed_request_count']}", flush=True)
    print(f"pending={len(pending)}", flush=True)
    print(f"decision={decision['execution_decision']}", flush=True)
    if len(request_records_after) > EXPECTED_REQUESTS or len(response_records_after) > EXPECTED_REQUESTS:
        raise RuntimeError("raw evidence exceeds full manifest size")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
