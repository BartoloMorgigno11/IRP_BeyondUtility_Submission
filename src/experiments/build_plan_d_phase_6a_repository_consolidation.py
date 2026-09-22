r"""Build Phase 6A repository evidence freeze and consolidation artifacts.

Run from the repository root with:

    $env:PYTHONNOUSERSITE = "1"
    .\.venv\Scripts\python.exe src\experiments\build_plan_d_phase_6a_repository_consolidation.py

The builder reads existing artifacts and writes only Phase 6A-specific
registries, manifests, hashes, and the consolidation blueprint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_d_phase_6a_repository_consolidation import (  # noqa: E402
    EXPECTED_BRANCH,
    EXPECTED_HEAD,
    build_all,
    phase_6a_decision,
    write_output_hash_manifest,
    write_protected_manifest_only,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline Phase 6A repository consolidation blueprint.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=None, help="Accepted for project CLI consistency; not used.")
    parser.add_argument("--protected-manifest-only", choices=["initial", "final"], default=None)
    parser.add_argument("--output-hash-manifest", choices=["first", "second"], default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve()
    if args.protected_manifest_only:
        manifest = write_protected_manifest_only(root, args.protected_manifest_only)
        print(f"phase_6a_protected_manifest={args.protected_manifest_only}")
        print(f"protected_input_count={len(manifest)}")
        return 0
    outputs, hashes = build_all(root, output_hash_label=args.output_hash_manifest)
    validation = outputs["validation"]
    artifact_registry = outputs["artifact_registry"]
    plan_e = outputs["plan_e_reconciliation"]
    phase5e = outputs["phase_5e_accounting"]
    print(f"expected_branch={EXPECTED_BRANCH}")
    print(f"expected_head={EXPECTED_HEAD}")
    print(f"total_artifacts={len(artifact_registry)}")
    print(f"plan_e_artifacts={len(plan_e)}")
    print(f"entrypoints={len(outputs['reproducibility_entrypoints'])}")
    print(f"validation_pass={validation['status'].eq('pass').sum()}")
    print(f"validation_fail={validation['status'].eq('fail').sum()}")
    print("phase5e_status_distribution=" + phase5e[phase5e["record_type"].eq("status_count")].set_index("status_or_metric")["count"].to_json())
    print("lifecycle_counts=" + artifact_registry["lifecycle_class"].value_counts().sort_index().to_json())
    print("disposition_counts=" + artifact_registry["final_delivery_disposition"].value_counts().sort_index().to_json())
    print(f"phase_6a_decision={phase_6a_decision(validation)}")
    for path, digest in sorted(hashes.items()):
        print(f"sha256 {path} {digest}")
    if args.output_hash_manifest:
        write_output_hash_manifest(root, args.output_hash_manifest)
        print(f"output_hash_manifest={args.output_hash_manifest}")
    return 0 if phase_6a_decision(validation) == "final_repository_evidence_freeze_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

