"""Build Finalization F4 metric resolution outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.finalization_f4_metric_resolution import (  # noqa: E402
    EXPECTED_BRANCH,
    EXPECTED_HEAD,
    build_all,
    git_output,
    output_hashes,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Finalization F4 metric resolution package.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ml1m.yaml"),
        help="Accepted for project CLI consistency; F4 reads frozen evidence tables.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve()
    result = build_all(root)
    frames = result["frames"]
    inventory = frames["metric_resolution_inventory"]
    triage = frames["missing_evidence_triage"]
    claims = frames["claim_update"]
    coverage = frames["coverage_impact"]
    problematic = inventory.loc[inventory["f3_blocking_issue"].astype(str).str.len().gt(0)]
    blocked = inventory.loc[inventory["resolution_decision"].astype(str).str.startswith("blocked")]
    print(f"expected_branch={EXPECTED_BRANCH}")
    print(f"observed_branch={git_output(root, ['branch', '--show-current'])}")
    print(f"expected_head={EXPECTED_HEAD}")
    print(f"observed_head={git_output(root, ['rev-parse', 'HEAD'])}")
    print(f"decision={result['decision']}")
    print(f"metrics_analyzed={len(inventory)}")
    print(f"problematic_metrics={';'.join(sorted(problematic['f3_metric_definition_id'].tolist()))}")
    print(f"resolved_exact={inventory['resolution_decision'].eq('resolved_exact').sum()}")
    print(f"resolved_functional={inventory['resolution_decision'].eq('resolved_functional').sum()}")
    print(f"resolved_adapted={inventory['resolution_decision'].eq('resolved_adapted').sum()}")
    print(f"resolved_extension_only={inventory['resolution_decision'].eq('resolved_extension_only').sum()}")
    print(f"blocked_metrics={len(blocked)}")
    print(f"resolved_metric_results={len(frames['resolved_metric_results'])}")
    print(f"reconciled_results={frames['metric_reconciliation']['is_reconciled'].eq('true').sum()}")
    print(f"missing_evidence_analyzed={len(triage)}")
    print(f"missing_resolved={triage['resolution_status'].eq('resolved').sum()}")
    print(f"missing_ready_for_f5={triage['resolution_status'].eq('ready_for_f5').sum()}")
    print(f"missing_conditional={triage['resolution_status'].eq('conditional').sum()}")
    print(f"missing_blocked={triage['resolution_status'].eq('blocked').sum()}")
    print(f"f5_candidates={len(frames['f5_experiment_candidates'])}")
    print(f"safe_claims={claims['is_safe_for_report'].eq('true').sum()}")
    print(f"unsafe_claims={claims['is_safe_for_report'].eq('false').sum()}")
    for _, row in coverage.iterrows():
        print(f"coverage {row['coverage_metric']} {row['f3_value']}->{row['f4_value']} change={row['absolute_change']}")
    print(f"anomalies={len(frames['anomalies'])}")
    print(f"validation_pass={frames['validation']['status'].eq('pass').sum()}")
    print(f"validation_fail={frames['validation']['status'].eq('fail').sum()}")
    for path, digest in result["hashes"].items():
        print(f"sha256 {path} {digest}")
    second_hashes = output_hashes(root)
    print(f"second_hashes_match={str(second_hashes == result['hashes']).lower()}")
    return 0 if result["decision"] == "finalization_f4_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
