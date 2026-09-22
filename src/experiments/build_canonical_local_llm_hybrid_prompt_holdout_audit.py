"""Build the offline Phase 5B-P1E hybrid prompt and holdout audit.

This script performs no inference and invokes no Ollama endpoint.
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

from src.analysis.canonical_local_llm_hybrid_prompt_holdout_audit import (  # noqa: E402
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
    "$env:PYTHONNOUSERSITE='1'; & '.venv\\Scripts\\python.exe' src\\experiments\\build_canonical_local_llm_hybrid_prompt_holdout_audit.py",
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
    bundle, frames = build_all_frames(ROOT)
    if before_hashes != bundle.protected_hashes:
        raise RuntimeError("protected input hashes changed during evidence loading")
    write_tables(ROOT, frames)
    write_report(ROOT, bundle, frames, COMMANDS_EXECUTED)
    validate_written_outputs(ROOT, frames)
    after_hashes = hash_protected_inputs(ROOT)
    if before_hashes != after_hashes:
        changed = sorted(
            path for path in set(before_hashes) | set(after_hashes) if before_hashes.get(path) != after_hashes.get(path)
        )
        raise RuntimeError(f"protected input hashes changed: {changed}")
    decision = frames["decision"].iloc[0].to_dict()
    print("canonical_local_llm_hybrid_prompt_holdout_audit")
    print(f"decision={decision['recommended_decision']}")
    print(f"holdout_sample_rows={decision['holdout_sample_rows']}")
    print(f"holdout_request_rows={decision['holdout_request_rows']}")
    print(f"common_prompt_v3_required={decision['common_prompt_v3_required']}")
    for path in output_paths(ROOT, frames):
        print(path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
