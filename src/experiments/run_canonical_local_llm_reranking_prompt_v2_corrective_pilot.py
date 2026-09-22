"""Run/resume the Phase 5B-P1D reranking prompt-v2 corrective pilot."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.canonical_local_llm_reranking_prompt_v2_runner import (  # noqa: E402
    EVENT_JSONL,
    RAW_DIR,
    REQUEST_JSONL,
    RESPONSE_JSONL,
    build_all_result_tables,
    environment_summary,
    execute_pending_requests,
    hash_protected_inputs,
    load_inputs,
    output_targets,
    read_jsonl,
    state_summary,
    validate_output_frames,
    verify_ollama_admin,
    write_report,
    write_tables,
    utc_now,
)


COMMANDS_EXECUTED = [
    "git status --short --branch",
    "git branch --show-current",
    'git log -1 --format=\"%H %s\"',
    "$env:PYTHONNOUSERSITE='1'; & '.venv\\Scripts\\python.exe' -m pytest tests\\test_canonical_local_llm_reranking_prompt_v2_runner.py tests\\test_canonical_local_llm_pilot_invalid_output_audit.py tests\\test_canonical_local_llm_pilot_resume.py tests\\test_canonical_local_llm_runner.py tests\\test_canonical_llm_contract.py tests\\test_experiment_cohort_policy.py tests\\test_dataset_candidate_set_validation.py -q",
    "$env:PYTHONNOUSERSITE='1'; & '.venv\\Scripts\\python.exe' src\\experiments\\run_canonical_local_llm_reranking_prompt_v2_corrective_pilot.py",
]


def _require_environment(root: Path) -> None:
    expected_python = (root / ".venv" / "Scripts" / "python.exe").resolve()
    actual_python = Path(sys.executable).resolve()
    if actual_python != expected_python:
        raise RuntimeError(f"expected repo-local venv interpreter {expected_python}, got {actual_python}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("PYTHONNOUSERSITE=1 is required")


def _validate_raw_counts(root: Path) -> None:
    request_records = read_jsonl(root / REQUEST_JSONL)
    response_records = read_jsonl(root / RESPONSE_JSONL)
    event_records = read_jsonl(root / EVENT_JSONL)
    if len(request_records) != 60:
        raise RuntimeError(f"request JSONL count mismatch: {len(request_records)}")
    if len(response_records) != 60:
        raise RuntimeError(f"response JSONL count mismatch: {len(response_records)}")
    if len(event_records) != 120:
        raise RuntimeError(f"event JSONL count mismatch: {len(event_records)}")
    ids = [str(record.get("corrective_request_id", record.get("request_id", ""))) for record in request_records]
    if len(set(ids)) != 60:
        raise RuntimeError("request JSONL contains duplicate corrective IDs")
    response_ids = [str(record.get("corrective_request_id", record.get("request_id", ""))) for record in response_records]
    if sorted(response_ids) != sorted(ids):
        raise RuntimeError("response JSONL IDs do not match request JSONL IDs")


def main() -> int:
    _require_environment(ROOT)
    inputs = load_inputs(ROOT)
    before_hashes = dict(inputs.protected_hashes)
    if hash_protected_inputs(ROOT) != before_hashes:
        raise RuntimeError("protected inputs changed before execution")
    request_records = read_jsonl(ROOT / REQUEST_JSONL)
    response_records = read_jsonl(ROOT / RESPONSE_JSONL)
    initial_request_count = len(request_records)
    initial_response_count = len(response_records)
    initial_summary = state_summary(inputs.corrective_manifest, request_records, response_records)
    if initial_summary.get("ambiguous_interrupted", 0) or initial_summary.get("invalid_duplicate_evidence", 0):
        raise RuntimeError(f"unsafe initial resume state: {initial_summary}")
    started_at = utc_now()
    admin = verify_ollama_admin(ROOT, inputs.recovery)
    execute_pending_requests(ROOT, inputs)
    completed_at = utc_now()
    frames = build_all_result_tables(
        ROOT,
        inputs,
        admin,
        started_at,
        completed_at,
        initial_summary,
        initial_request_count,
        initial_response_count,
    )
    write_tables(ROOT, frames)
    write_report(ROOT, frames, environment_summary(ROOT), COMMANDS_EXECUTED)
    validate_output_frames(frames)
    _validate_raw_counts(ROOT)
    if hash_protected_inputs(ROOT) != before_hashes:
        raise RuntimeError("protected inputs changed during P1D")
    for path in output_targets(ROOT):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty output: {path}")
    decision = frames["decision"].iloc[0].to_dict()
    print("canonical_local_llm_reranking_prompt_v2_corrective_pilot")
    print(f"raw_dir={RAW_DIR.as_posix()}")
    print(f"completed={decision['completed_request_count']}")
    print(f"backend_completion_rate={decision['backend_completion_rate']}")
    print(f"minimum_cell_strict_parse_validity_rate={decision['minimum_cell_strict_parse_validity_rate']}")
    print(f"decision={decision['pilot_decision']}")
    for path in output_targets(ROOT):
        print(path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
