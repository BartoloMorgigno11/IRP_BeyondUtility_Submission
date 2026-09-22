"""Summarize existing Plan D ML-1M extended neural baseline metrics.

Run from the repository root:

    python src/experiments/summarize_plan_d_extended_neural.py

The script reads existing metric CSVs only. It does not train models, run
inference, modify raw outputs, download data, or call APIs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_d_extended_neural_summary import write_outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Plan D ML-1M extended neural baselines.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root containing results/metrics and report/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = write_outputs(project_root=args.project_root)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
