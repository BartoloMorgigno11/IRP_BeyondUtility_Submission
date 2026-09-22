"""CLI wrapper for Phase F6I1 repository retention audit."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.phase_f6i1_repository_retention_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
