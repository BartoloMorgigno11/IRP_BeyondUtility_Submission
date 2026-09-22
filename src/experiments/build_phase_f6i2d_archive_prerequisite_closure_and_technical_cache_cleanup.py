"""CLI wrapper for the F6I2D archive prerequisite closure builder."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_f6i2d_archive_prerequisite_closure_and_technical_cache_cleanup import main


if __name__ == "__main__":
    main()
