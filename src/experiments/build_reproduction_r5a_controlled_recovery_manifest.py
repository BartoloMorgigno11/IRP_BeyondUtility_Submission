"""CLI wrapper for the R5A controlled recovery manifest builder."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.reproduction_r5a_controlled_recovery_manifest import (  # noqa: E402
    DECISION_COMPLETE,
    build_r5a,
)


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reproduction_r5a_controlled_recovery_manifest.yaml")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    project_root = Path(args.project_root or config.get("project_root", ".")).resolve()
    result = build_r5a(project_root, config_path=config_path, write=not args.no_write)
    frames = result["frames"]
    validation = frames["validation"]
    print(f"decision={result['decision']}")
    print(f"manifest_rows={len(frames['recovery_request_manifest'])}")
    print(f"raw_rows={len(frames['raw_planned_requests'])}")
    print(f"manifest_sha256={result['manifest_sha256']}")
    print(f"validation_pass={int(validation['status'].eq('pass').sum())}")
    print(f"validation_fail={int(validation['status'].eq('fail').sum())}")
    return 0 if result["decision"] == DECISION_COMPLETE else 1


if __name__ == "__main__":
    raise SystemExit(main())
