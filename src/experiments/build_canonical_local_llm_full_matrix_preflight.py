"""Build Phase 5B-P2 full local LLM matrix preflight artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.canonical_local_llm_full_matrix_preflight import (
    build_preflight_frames,
    validate_written_preflight,
    write_preflight_frames,
)


def main() -> int:
    root = Path.cwd()
    frames = build_preflight_frames(root)
    write_preflight_frames(root, frames)
    hashes = validate_written_preflight(root)
    for path, digest in hashes.items():
        print(f"{path} {digest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
