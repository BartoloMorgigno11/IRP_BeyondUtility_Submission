from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_e_paper_like_candidate_pools import (
    PHASE_ID,
    PHASE_TITLE,
    PlanECandidatePoolPaths,
    build_candidate_pool_outputs,
    write_candidate_pool_outputs,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Plan E2 paper-like candidate-pool manifests.")
    parser.add_argument(
        "--config",
        default=None,
        help="Accepted for project CLI consistency; E2 discovers local repository candidate sources.",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    paths = PlanECandidatePoolPaths()
    tables = build_candidate_pool_outputs(PROJECT_ROOT, paths)
    write_candidate_pool_outputs(tables, paths)

    eligibility = tables["dataset_eligibility"]
    manifest_summary = tables["manifest_summary"]
    validation = tables["validation"]
    complete = int((manifest_summary["exact_1000_user_target_met"] == "true").sum())
    failures = int((validation["passed"] != "true").sum())

    print(f"{PHASE_ID} {PHASE_TITLE}")
    print("Manifest-only: no inference, no Ollama calls, no API calls, no downloads, no model pulls, no training, no LastFM run.")
    print(f"dataset_eligibility_rows: {len(eligibility)}")
    print(f"manifest_summary_rows: {len(manifest_summary)}")
    print(f"complete_1000_user_manifests: {complete}")
    print(f"validation_rows: {len(validation)}")
    print(f"validation_failures: {failures}")
    print(f"report: {paths.report.as_posix()}")
    LOGGER.info("Wrote %s %s outputs", PHASE_ID, PHASE_TITLE)


if __name__ == "__main__":
    main()
