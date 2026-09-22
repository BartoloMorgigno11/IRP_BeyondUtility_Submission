"""Build the Phase 5B-P1F final reranking protocol decision audit.

This script is offline-only. It performs no inference and invokes no Ollama
endpoint, including administrative endpoints.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.canonical_local_llm_final_reranking_protocol_audit import (  # noqa: E402
    build_all_frames,
    hash_protected_inputs,
    output_paths,
    validate_written_outputs,
    write_report,
    write_tables,
)


COMMANDS_EXECUTED = [
    "git status --short --branch",
    "git branch --show-current",
    'git log -1 --format=\"%H %s\"',
    "$env:PYTHONNOUSERSITE='1'; & '.venv\\Scripts\\python.exe' src\\experiments\\build_canonical_local_llm_final_reranking_protocol_audit.py",
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
    before_hashes = hash_protected_inputs(ROOT)
    protected_hashes, frames = build_all_frames(ROOT)
    if before_hashes != protected_hashes:
        raise RuntimeError("protected input hashes changed during evidence loading")
    decision = frames["final_decision"].iloc[0]
    if decision["holdout_failure_preserved"] != "true":
        raise RuntimeError("holdout failure was not preserved")
    if decision["utility_used_for_decision"] != "false":
        raise RuntimeError("utility influenced the protocol decision")
    if decision["prompt_v3_created"] != "false":
        raise RuntimeError("prompt v3 creation is forbidden")
    if decision["full20_paper_exact"] != "false":
        raise RuntimeError("full20 cannot be claimed paper exact without evidence")
    if decision["recommended_decision"] == "approve_paper_aligned_topk_reranking_confirmation_pilot":
        if decision["official_reranking_output_count_verified"] != "true":
            raise RuntimeError("top-K confirmation cannot be approved without official/paper evidence")
    write_tables(ROOT, frames)
    write_report(ROOT, frames, protected_hashes, COMMANDS_EXECUTED)
    output_hashes = validate_written_outputs(ROOT, frames)
    after_hashes = hash_protected_inputs(ROOT)
    if before_hashes != after_hashes:
        changed = sorted(
            path for path in set(before_hashes) | set(after_hashes) if before_hashes.get(path) != after_hashes.get(path)
        )
        raise RuntimeError(f"protected input hashes changed: {changed}")
    print("canonical_local_llm_final_reranking_protocol_audit")
    print(f"decision={decision['recommended_decision']}")
    print(f"selected_protocol_option_id={decision['selected_protocol_option_id']}")
    print(f"confirmation_pilot_required={decision['confirmation_pilot_required']}")
    print(f"protected_hashes={len(protected_hashes)}")
    print(f"output_hashes={len(output_hashes)}")
    for path in output_paths(ROOT):
        print(path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

