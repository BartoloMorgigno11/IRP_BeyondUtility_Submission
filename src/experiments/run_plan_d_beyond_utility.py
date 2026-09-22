"""Run Plan D beyond-utility evaluation on existing Plan C recommendations.

Run from the repository root:

    python src/experiments/run_plan_d_beyond_utility.py

This script reads existing ``results/raw_outputs/plan_c_*_recommendations.csv``
files and reconstructed Plan C canonical datasets. It does not rerun Plan C
baselines, download data, train models, call APIs, or write Plan C outputs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_d_beyond_utility import run_plan_d_beyond_utility


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Plan D beyond-utility metrics from existing Plan C outputs."
    )
    parser.add_argument("--cutoff", type=int, default=5, help="Evaluation cutoff. D1 supports 5.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/reconstructed/plan_c"),
        help="Project-relative reconstructed Plan C data root.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Project-relative results root.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args(argv)
    result = run_plan_d_beyond_utility(
        cutoff=args.cutoff,
        data_root=args.data_root,
        results_root=args.results_root,
        project_root=PROJECT_ROOT,
    )
    metrics = result["metrics"]
    status = result["status"]
    logging.info("Plan D status rows: %s", len(status))
    logging.info("Plan D metric rows: %s", len(metrics))
    if result["metrics_path"] is not None:
        logging.info("Saved metrics to %s", result["metrics_path"].relative_to(PROJECT_ROOT))
    logging.info("Saved status to %s", result["status_path"].relative_to(PROJECT_ROOT))
    if result["comparison_path"] is not None:
        logging.info("Saved comparison to %s", result["comparison_path"].relative_to(PROJECT_ROOT))
    for plot_path in result["plot_paths"]:
        logging.info("Saved plot to %s", plot_path.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

