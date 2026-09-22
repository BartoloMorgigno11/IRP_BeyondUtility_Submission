"""CLI wrapper for Phase F6I2A consolidation design."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.plan_f6i2a_substantive_repository_consolidation_design import main


if __name__ == "__main__":
    raise SystemExit(main())
