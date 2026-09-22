"""CLI wrapper for the F6L paper-faithful checkpoint builder."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_f6l_paper_faithful_completion_and_rq_revalidation import main


if __name__ == "__main__":
    raise SystemExit(main())
