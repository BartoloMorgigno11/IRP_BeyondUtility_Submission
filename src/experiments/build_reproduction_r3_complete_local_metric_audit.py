"""CLI wrapper for the R3 complete local metric audit."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.reproduction_r3_complete_local_metric_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
