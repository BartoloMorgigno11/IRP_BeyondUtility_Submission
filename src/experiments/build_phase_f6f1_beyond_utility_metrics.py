"""CLI entrypoint for Phase F6F1 beyond-utility metrics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.plan_f6f1_beyond_utility_metrics import main


if __name__ == "__main__":
    raise SystemExit(main())
