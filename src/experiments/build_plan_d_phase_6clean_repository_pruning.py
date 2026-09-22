"""CLI wrapper for Phase 6CLEAN repository pruning."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.plan_d_phase_6clean_repository_pruning import main


if __name__ == "__main__":
    raise SystemExit(main())
