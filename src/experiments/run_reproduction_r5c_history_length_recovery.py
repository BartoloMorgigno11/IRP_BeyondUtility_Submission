"""CLI wrapper for the R5C History-Length recovery runner."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.reproduction_r5c_history_length_recovery import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
