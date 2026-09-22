"""CLI wrapper for the F6I2N repository cleanup readiness review."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_f6i2n_repository_cleanup_after_explicit_approval import main


if __name__ == "__main__":
    raise SystemExit(main())
