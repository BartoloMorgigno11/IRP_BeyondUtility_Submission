"""CLI wrapper for Phase 6C final scientific report generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_d_phase_6c_final_scientific_report import (  # noqa: E402
    phase_6c_decision,
    verify_output_hash_manifests,
    write_phase6c_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase 6C final scientific report")
    parser.add_argument("--hash-label", choices=["first", "second"], default="first")
    args = parser.parse_args()
    root = Path.cwd()
    result = write_phase6c_outputs(root, args.hash_label)
    print(f"decision={result['decision']}")
    print(f"hash_label={result['hash_label']}")
    print(f"protected_count={result['protected_count']}")
    print(f"section_count={result['section_count']}")
    print(f"word_count={result['word_count']}")
    print(f"claim_count={result['claim_count']}")
    print(f"quantitative_claim_count={result['quantitative_claim_count']}")
    print(f"deterministic_hashes_match={verify_output_hash_manifests(root)}")
    print(f"final_decision={phase_6c_decision(root)}")
    return 0 if result["decision"] == "final_unified_scientific_report_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
