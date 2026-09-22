"""Build Finalization F3 canonical scientific consolidation outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.finalization_f3_canonical_scientific_consolidation import (  # noqa: E402
    EXPECTED_BRANCH,
    EXPECTED_HEAD,
    build_all,
    git_output,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Finalization F3 canonical scientific consolidation.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=Path("configs/ml1m.yaml"), help="Accepted for CLI consistency; F3 reads frozen evidence.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve()
    result = build_all(root)
    frames = result["frames"]
    print(f"expected_branch={EXPECTED_BRANCH}")
    print(f"observed_branch={git_output(root, ['branch', '--show-current'])}")
    print(f"expected_head={EXPECTED_HEAD}")
    print(f"observed_head={git_output(root, ['rev-parse', 'HEAD'])}")
    print(f"decision={result['decision']}")
    print(f"f2_rows={len(result['bundle'].f2_registry)}")
    print(f"canonical_results={len(frames['canonical_results'])}")
    print(f"primary_results={frames['canonical_results']['is_primary_result'].eq('true').sum()}")
    print(f"reportable_results={frames['canonical_results']['is_reportable'].eq('true').sum()}")
    print(f"comparable_groups={frames['canonical_results']['comparable_group_id'].nunique()}")
    print(f"valid_comparisons={len(frames['valid_comparisons'])}")
    print(f"paper_alignment_rows={len(frames['paper_alignment'])}")
    print(f"metric_status_rows={len(frames['metric_status'])}")
    print(f"missing_evidence_rows={len(frames['missing_evidence'])}")
    print(f"safe_claims={frames['claim_registry']['is_safe_for_report'].eq('true').sum()}")
    print(f"unsafe_claims={frames['claim_registry']['is_safe_for_report'].eq('false').sum()}")
    print(f"validation_pass={frames['validation']['status'].eq('pass').sum()}")
    print(f"validation_fail={frames['validation']['status'].eq('fail').sum()}")
    for path, digest in result["hashes"].items():
        print(f"sha256 {path} {digest}")
    return 0 if result["decision"] == "finalization_f3_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

