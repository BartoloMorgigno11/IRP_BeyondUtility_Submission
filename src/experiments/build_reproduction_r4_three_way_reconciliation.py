"""CLI wrapper for the R4 three-way reconciliation builder."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.reproduction_r4_three_way_reconciliation import build_r4


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reproduction_r4_three_way_reconciliation.yaml")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    project_root = Path(args.project_root or config.get("project_root", ".")).resolve()
    result = build_r4(project_root, write=not args.no_write)
    frames = result["frames"]
    validation = frames["validation"]
    print(f"decision={result['decision']}")
    print(f"three_way_rows={len(frames['three_way_master'])}")
    print(f"beyond_summary_rows={len(frames['beyond_summary'])}")
    print(f"gap_priority_rows={len(frames['gap_priority'])}")
    print(f"proposed_recovery_rows={len(frames['proposed_recovery_experiments'])}")
    print(f"validation_pass={int((validation['status'] == 'pass').sum())}")
    print(f"validation_fail={int((validation['status'] == 'fail').sum())}")
    return 0 if result["decision"] == "REPRODUCTION_R4_THREE_WAY_RECONCILIATION_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
