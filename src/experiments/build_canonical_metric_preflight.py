"""Build the Phase 4A canonical metric contract and preflight artifacts.

Run from the project root:

    python src/experiments/build_canonical_metric_preflight.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.canonical_metric_preflight import OUTPUTS, build_all


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase 4A metric-contract audit and same-cohort preflight outputs."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root. Defaults to the detected project root.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve()
    outputs, initial_hashes, final_hashes, output_hashes, _ = build_all(root)
    unchanged = initial_hashes == final_hashes
    print(f"metric_contract_rows={len(outputs['contract'])}")
    print(f"implementation_inventory_rows={len(outputs['inventory'])}")
    print(f"same_cohort_preflight_rows={len(outputs['preflight'])}")
    print(f"applicability_rows={len(outputs['applicability'])}")
    print(f"blocked_or_ambiguous_rows={len(outputs['blocked'])}")
    print(f"protected_inputs_hashed={len(initial_hashes)}")
    print(f"protected_inputs_unchanged={str(unchanged).lower()}")
    for relative_path in [path for key, path in OUTPUTS.items() if key != "report"]:
        print(f"sha256 {relative_path} {output_hashes[relative_path]}")
    print(f"report_path={OUTPUTS['report']}")
    return 0 if unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
