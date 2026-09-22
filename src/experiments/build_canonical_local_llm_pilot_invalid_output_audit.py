"""Build the offline Phase 5B-P1C invalid-output prompt audit.

This script performs no model calls and invokes no Ollama endpoint.
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

from src.analysis.canonical_local_llm_pilot_invalid_output_audit import (  # noqa: E402
    output_paths,
    build_all_frames,
    hash_paths,
    validate_frames,
    write_report,
    write_tables,
)


COMMANDS_EXECUTED = [
    "git status --short --branch",
    "git branch --show-current",
    'git log -1 --format="%H %s"',
    "$env:PYTHONNOUSERSITE='1'; & '.venv\\Scripts\\python.exe' src\\experiments\\build_canonical_local_llm_pilot_invalid_output_audit.py",
]


def _require_clean_environment(root: Path) -> None:
    expected_python = (root / ".venv" / "Scripts" / "python.exe").resolve()
    actual_python = Path(sys.executable).resolve()
    if actual_python != expected_python:
        raise RuntimeError(f"expected repo-local venv interpreter {expected_python}, got {actual_python}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("PYTHONNOUSERSITE=1 is required")


def _validate_written_outputs(root: Path, frames: dict[str, object]) -> None:
    validate_frames(frames)
    for path in output_paths(root, frames):
        if not path.is_file():
            raise RuntimeError(f"expected output not written: {path}")
        if path.stat().st_size == 0:
            raise RuntimeError(f"empty output written: {path}")
    decision = frames["decision"].iloc[0]
    if decision["recommended_decision"] == "approve_reranking_prompt_v2_corrective_pilot":
        if frames["parser_assessment"]["parser_revision_recommended"].eq("true").any():
            raise RuntimeError("prompt v2 cannot be approved when a parser defect is detected")
        if frames["prompt_v2_contract"].empty:
            raise RuntimeError("approved corrective pilot requires a prompt-v2 contract")
        if len(frames["corrective_manifest"]) != 60:
            raise RuntimeError("approved corrective pilot requires exactly 60 planned reranking rows")
    report_text = (root / "report" / "canonical_local_llm_pilot_invalid_output_and_prompt_corrective_audit.md").read_text(encoding="utf-8")
    section_count = sum(1 for line in report_text.splitlines() if line.startswith("## "))
    if section_count != 37:
        raise RuntimeError(f"report section count mismatch: {section_count}")
    invalid_response_texts = frames["invalid_response_audit"].get("raw_response_text", [])
    for raw_text in invalid_response_texts:
        if raw_text and str(raw_text) in report_text:
            raise RuntimeError("full raw response leaked into report")


def main() -> int:
    _require_clean_environment(ROOT)
    before_hashes = hash_paths(ROOT)
    bundle, frames = build_all_frames(ROOT)
    write_tables(ROOT, frames)
    write_report(ROOT, bundle, frames, COMMANDS_EXECUTED)
    after_hashes = hash_paths(ROOT)
    if before_hashes != after_hashes:
        changed = sorted(path for path in set(before_hashes) | set(after_hashes) if before_hashes.get(path) != after_hashes.get(path))
        raise RuntimeError(f"protected input hashes changed: {changed}")
    _validate_written_outputs(ROOT, frames)
    decision = frames["decision"].iloc[0].to_dict()
    print("canonical_local_llm_pilot_invalid_output_audit")
    print(f"raw_requests={decision['authoritative_total_requests']}")
    print(f"strict_valid={decision['authoritative_strict_valid_count']}")
    print(f"strict_invalid={decision['authoritative_strict_invalid_count']}")
    print(f"decision={decision['recommended_decision']}")
    print(f"corrective_manifest_rows={decision['corrective_pilot_manifest_rows']}")
    for path in output_paths(ROOT, frames):
        print(path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
