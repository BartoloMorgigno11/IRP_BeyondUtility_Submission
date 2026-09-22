from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_e_baseline_alignment import (
    PHASE_ID,
    PHASE_TITLE,
    PlanEBaselineAlignmentPaths,
    build_all_tables,
    write_baseline_alignment_outputs,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Plan E3A baseline alignment and no-training metrics.")
    parser.add_argument(
        "--config",
        default=None,
        help="Accepted for project CLI consistency; E3A reads E2 manifests and existing local artifacts.",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    tables = build_all_tables(PROJECT_ROOT)
    paths = PlanEBaselineAlignmentPaths()
    write_baseline_alignment_outputs(tables, paths)

    metrics = tables["no_training_metrics"]
    missing = tables["missing_baselines"]
    validation = tables["validation"]
    print(f"{PHASE_ID} {PHASE_TITLE}")
    print("No-training audit: no inference, no Ollama calls, no API calls, no downloads, no model pulls, no training, no LastFM run.")
    print(f"manifest_inventory_rows: {len(tables['manifest_inventory'])}")
    print(f"source_inventory_rows: {len(tables['source_inventory'])}")
    print(f"score_availability_rows: {len(tables['score_availability'])}")
    print(f"no_training_metric_rows: {len(metrics)}")
    print(f"missing_baseline_rows: {len(missing)}")
    print(f"validation_rows: {len(validation)}")
    print(f"validation_failures: {int((validation['passed'] != 'true').sum())}")
    print(f"report: {paths.report.as_posix()}")
    LOGGER.info("Wrote %s %s outputs", PHASE_ID, PHASE_TITLE)


if __name__ == "__main__":
    main()
