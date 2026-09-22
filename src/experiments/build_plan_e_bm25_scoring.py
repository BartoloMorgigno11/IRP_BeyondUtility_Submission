from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_e_bm25_scoring import (  # noqa: E402
    PHASE_ID,
    PHASE_TITLE,
    PlanEBM25ScoringPaths,
    build_all_tables,
    write_bm25_scoring_outputs,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Plan E3B BM25/BM25-like scoring outputs.")
    parser.add_argument(
        "--config",
        default=None,
        help="Accepted for project CLI consistency; E3B reads E2 manifests and local reconstructed artifacts.",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    tables = build_all_tables(PROJECT_ROOT)
    paths = PlanEBM25ScoringPaths()
    write_bm25_scoring_outputs(tables, paths)

    audit = tables["query_corpus_audit"]
    scores = tables["scores"]
    metrics = tables["metrics"]
    validation = tables["validation"]
    scored_rows = int((audit["can_score"] == "true").sum()) if not audit.empty else 0
    print(f"{PHASE_ID} {PHASE_TITLE}")
    print("BM25-like audit: no inference, no Ollama calls, no API calls, no downloads, no model pulls, no training, no neural training, no LastFM run.")
    print(f"source_inventory_rows: {len(tables['source_inventory'])}")
    print(f"query_corpus_audit_rows: {len(audit)}")
    print(f"scorable_dataset_task_rows: {scored_rows}")
    print(f"bm25_score_rows: {len(scores)}")
    print(f"bm25_metric_rows: {len(metrics)}")
    print(f"missing_or_blocked_rows: {len(tables['missing_or_blocked'])}")
    print(f"validation_rows: {len(validation)}")
    print(f"validation_failures: {int((validation['passed'] != 'true').sum())}")
    print(f"report: {paths.report.as_posix()}")
    LOGGER.info("Wrote %s %s outputs", PHASE_ID, PHASE_TITLE)


if __name__ == "__main__":
    main()
