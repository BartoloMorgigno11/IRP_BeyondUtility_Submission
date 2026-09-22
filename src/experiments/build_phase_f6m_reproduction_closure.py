"""CLI wrapper for the F6M reproduction closure builder."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.phase_f6m_reproduction_closure import main


if __name__ == "__main__":
    main()
