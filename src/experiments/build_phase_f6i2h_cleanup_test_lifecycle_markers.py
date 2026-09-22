"""CLI wrapper for the F6I2H cleanup test lifecycle builder."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_f6i2h_cleanup_test_lifecycle_markers import main


if __name__ == "__main__":
    main()
