r"""Build offline Phase 5D robustness consolidation artifacts.

Run from the repository root with the authorized interpreter, for example:

    $env:PYTHONNOUSERSITE = "1"
    .\.venv\Scripts\python.exe src\experiments\build_plan_d_phase_5d_robustness_consolidation.py

The builder reads existing artifacts only. It does not call Ollama, APIs,
recommender training, downloads, package installation, staging, or commits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_d_phase_5d_robustness_consolidation import (  # noqa: E402
    INITIAL_PROTECTED_MANIFEST_PATH,
    build_all,
    phase_5d_decision,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline Phase 5D LLM robustness consolidation.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=None, help="Accepted for project CLI consistency; not used by this offline consolidation.")
    parser.add_argument("--initial-protected-manifest", type=Path, default=INITIAL_PROTECTED_MANIFEST_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve()
    outputs, hashes = build_all(root, args.initial_protected_manifest)
    validation = outputs["validation"]
    decision = phase_5d_decision(validation)
    print(f"source_inventory_rows={len(outputs['source_inventory'])}")
    print(f"experiment_registry_rows={len(outputs['experiment_registry'])}")
    print(f"evidence_claim_rows={len(outputs['evidence_claims'])}")
    print(f"robustness_summary_rows={len(outputs['robustness_summary'])}")
    print(f"validation_pass={validation['status'].eq('pass').sum()}")
    print(f"validation_fail={validation['status'].eq('fail').sum()}")
    print(f"phase_5d_decision={decision}")
    for path, digest in hashes.items():
        print(f"sha256 {path} {digest}")
    return 0 if decision == "canonical_existing_robustness_evidence_consolidation_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
