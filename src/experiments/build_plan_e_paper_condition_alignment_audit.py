from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_e_paper_condition_alignment import (
    PHASE_ID,
    PHASE_TITLE,
    PlanEAuditPaths,
    build_all_tables,
    write_audit_outputs,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Plan E0 paper-condition alignment and scale-up feasibility audit."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Accepted for project CLI consistency; Plan E0 is config-independent and does not read dataset settings.",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    tables = build_all_tables(PROJECT_ROOT)
    paths = PlanEAuditPaths()
    write_audit_outputs(tables, paths)

    summary = {
        "alignment_rows": len(tables["alignment_matrix"]),
        "missing_condition_rows": len(tables["missing_conditions"]),
        "exactness_rows": len(tables["exactness_feasibility"]),
        "scaleup_scenarios": len(tables["scaleup_call_estimates"]),
        "runtime_rows": len(tables["local_runtime_estimates"]),
        "api_cost_rows": len(tables["api_cost_scenarios"]),
        "roadmap_rows": len(tables["recommended_roadmap"]),
    }
    LOGGER.info("Wrote %s %s outputs", PHASE_ID, PHASE_TITLE)
    print(f"{PHASE_ID} {PHASE_TITLE}")
    print("Audit-only: no inference, no Ollama calls, no API calls, no downloads, no model pulls, no training, no LastFM run.")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"report: {paths.report.as_posix()}")


if __name__ == "__main__":
    main()
