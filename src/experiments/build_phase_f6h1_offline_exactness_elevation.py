"""CLI wrapper for the F6H1 offline exactness-elevation builder."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_f6h1_offline_exactness_elevation import main


if __name__ == "__main__":
    main()
