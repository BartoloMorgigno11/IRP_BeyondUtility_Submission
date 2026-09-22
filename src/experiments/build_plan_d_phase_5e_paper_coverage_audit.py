r"""Build offline Phase 5E final paper coverage audit artifacts.

Run from the repository root with the authorized interpreter:

    $env:PYTHONNOUSERSITE = "1"
    .\.venv\Scripts\python.exe src\experiments\build_plan_d_phase_5e_paper_coverage_audit.py

The builder reads existing artifacts only. It does not call Ollama, APIs,
network resources, LLM inference, recommender training/inference, package
installation, staging, or commits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_d_phase_5e_paper_coverage_audit import (  # noqa: E402
    INITIAL_PROTECTED_MANIFEST_PATH,
    build_all,
    phase_5e_decision,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline Phase 5E paper coverage audit.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=None, help="Accepted for project CLI consistency; not used by this offline audit.")
    parser.add_argument("--initial-protected-manifest", type=Path, default=INITIAL_PROTECTED_MANIFEST_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve()
    outputs, hashes = build_all(root, args.initial_protected_manifest)
    validation = outputs["validation"]
    metrics = outputs["coverage_metrics"].set_index("metric_name")["value"].to_dict()
    coverage = outputs["component_coverage"]
    result_cells = outputs["paper_result_cells"]
    print(f"paper_component_rows={len(outputs['paper_component_registry'])}")
    print(f"applicable_component_count={metrics['applicable_component_count']}")
    print(f"reproduced_exact_count={metrics['reproduced_exact_count']}")
    print(f"reconstructed_count={metrics['reconstructed_count']}")
    print(f"adapted_count={metrics['adapted_count']}")
    print(f"blocked_count={metrics['blocked_count']}")
    print(f"extension_count={metrics['extension_only_count']}")
    print(f"paper_result_cell_count={len(result_cells)}")
    print(f"validation_pass={validation['status'].eq('pass').sum()}")
    print(f"validation_fail={validation['status'].eq('fail').sum()}")
    print(f"strict_exact_reproduction_micro_pct={metrics['strict_exact_reproduction_micro_pct']}")
    print(f"maximum_defensible_micro_coverage_pct={metrics['maximum_defensible_micro_coverage_pct']}")
    print(f"phase_5e_decision={phase_5e_decision(validation)}")
    print("status_counts=" + coverage["primary_coverage_status"].value_counts().sort_index().to_json())
    for path, digest in hashes.items():
        print(f"sha256 {path} {digest}")
    return 0 if phase_5e_decision(validation) == "maximum_defensible_paper_coverage_audit_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
