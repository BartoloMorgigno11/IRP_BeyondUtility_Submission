"""Build the Phase 5B-P1G top-10 reranking confirmation preflight.

This runner is offline-only. It creates contract, sample, manifest, validation,
and report artifacts without invoking Ollama or creating request/response JSONL.
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

from src.analysis.canonical_local_llm_top10_confirmation_preflight import (  # noqa: E402
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
    "$env:PYTHONNOUSERSITE='1'; .\\.venv\\Scripts\\python.exe -c \"import sys, platform, pandas, numpy; ...\"",
    "$env:PYTHONNOUSERSITE='1'; .\\.venv\\Scripts\\python.exe src\\experiments\\build_canonical_local_llm_top10_confirmation_preflight.py",
    "$env:PYTHONNOUSERSITE='1'; .\\.venv\\Scripts\\python.exe -m pytest tests\\test_canonical_local_llm_top10_confirmation_preflight.py ...",
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
        changed = sorted(
            path for path in set(before_hashes) | set(protected_hashes) if before_hashes.get(path) != protected_hashes.get(path)
        )
        raise RuntimeError(f"protected input hashes changed during evidence loading: {changed}")
    write_tables(ROOT, frames)
    write_report(ROOT, frames, protected_hashes, COMMANDS_EXECUTED)
    output_hashes = validate_written_outputs(ROOT, frames)
    after_hashes = hash_protected_inputs(ROOT)
    if before_hashes != after_hashes:
        changed = sorted(
            path for path in set(before_hashes) | set(after_hashes) if before_hashes.get(path) != after_hashes.get(path)
        )
        raise RuntimeError(f"protected input hashes changed during preflight generation: {changed}")
    decision = frames["preflight_decision"].iloc[0]
    print("canonical_local_llm_top10_confirmation_preflight")
    print(f"decision={decision['preflight_decision']}")
    print(f"prompt_compatible_rows={decision['prompt_compatible_rows']}")
    print(f"two_model_complete_request_count={decision['two_model_complete_request_count']}")
    print(f"confirmation_sample_rows={decision['confirmation_sample_rows']}")
    print(f"confirmation_request_rows={decision['confirmation_request_rows']}")
    print(f"protected_hashes={len(protected_hashes)}")
    print(f"output_hashes={len(output_hashes)}")
    for path in output_paths(ROOT):
        print(path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
