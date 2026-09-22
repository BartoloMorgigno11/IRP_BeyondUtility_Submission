from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_e_full_data_inventory import (
    PHASE_ID,
    PHASE_TITLE,
    PlanEInventoryPaths,
    build_all_tables,
    write_inventory_outputs,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Plan E1 full-data inventory and runtime estimator.")
    parser.add_argument(
        "--config",
        default=None,
        help="Accepted for project CLI consistency; E1 inventories repository artifacts and does not run experiments.",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    tables = build_all_tables(PROJECT_ROOT)
    paths = PlanEInventoryPaths()
    write_inventory_outputs(tables, paths)

    print(f"{PHASE_ID} {PHASE_TITLE}")
    print("Audit-only: no inference, no Ollama calls, no API calls, no downloads, no model pulls, no training, no LastFM run.")
    print(f"dataset_summary_rows: {len(tables['dataset_summary'])}")
    print(f"artifact_summary_rows: {len(tables['artifact_summary'])}")
    print(f"manifest_summary_rows: {len(tables['manifest_summary'])}")
    print(f"metric_summary_rows: {len(tables['metric_summary'])}")
    print(f"raw_output_hygiene_rows: {len(tables['raw_output_hygiene'])}")
    print(f"scaleup_readiness_rows: {len(tables['scaleup_readiness'])}")
    print(f"call_estimate_rows: {len(tables['call_estimates'])}")
    print(f"runtime_estimate_rows: {len(tables['runtime_estimates'])}")
    print(f"next_action_rows: {len(tables['next_actions'])}")
    print(f"report: {paths.report.as_posix()}")
    LOGGER.info("Wrote %s %s outputs", PHASE_ID, PHASE_TITLE)


if __name__ == "__main__":
    main()
