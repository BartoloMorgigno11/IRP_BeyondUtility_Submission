"""CLI wrapper for the F6G3 LastFM reranking top-10 freeze builder."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_f6g3_lastfm_reranking_top10_freeze import main


if __name__ == "__main__":
    main()
