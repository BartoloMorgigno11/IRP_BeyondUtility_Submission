"""CLI wrapper for the F6I2F post-archive readiness review builder."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_f6i2f_post_archive_consolidation_readiness_review import main


if __name__ == "__main__":
    main()
