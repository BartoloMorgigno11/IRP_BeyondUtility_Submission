r"""Build Phase 6B README and reproducibility documentation package.

Run from the repository root:

    $env:PYTHONNOUSERSITE = "1"
    .\.venv\Scripts\python.exe src\experiments\build_plan_d_phase_6b_readme_reproducibility.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_d_phase_6b_readme_reproducibility import (  # noqa: E402
    build_all,
    compare_output_hashes,
    phase_6b_decision,
    write_output_hash_manifest,
    write_protected_manifest_only,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 6B README reproducibility package.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=None, help="Accepted for CLI consistency; not used.")
    parser.add_argument("--protected-manifest-only", choices=["initial", "final"], default=None)
    parser.add_argument("--output-hash-manifest", choices=["first", "second"], default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve()
    if args.protected_manifest_only:
        manifest = write_protected_manifest_only(root, args.protected_manifest_only)
        print(f"phase_6b_protected_manifest={args.protected_manifest_only}")
        print(f"protected_input_count={len(manifest)}")
        return 0

    outputs, hashes = build_all(root, output_hash_label=args.output_hash_manifest)
    validation = outputs["validation"]
    safe_commands = outputs["safe_command_registry"]
    excluded = outputs["excluded_entrypoints"]
    claim_audit = outputs["readme_claim_audit"]
    print(f"safe_command_count={len(safe_commands)}")
    print(f"excluded_entrypoint_count={len(excluded)}")
    print(f"claim_audit_rows={len(claim_audit)}")
    print(f"validation_pass={validation['status'].eq('pass').sum()}")
    print(f"validation_fail={validation['status'].eq('fail').sum()}")
    print(f"phase_6b_decision={phase_6b_decision(validation)}")
    for path, digest in sorted(hashes.items()):
        print(f"sha256 {path} {digest}")
    if args.output_hash_manifest:
        write_output_hash_manifest(root, args.output_hash_manifest)
        print(f"output_hash_manifest={args.output_hash_manifest}")
    print(f"output_hashes_identical={str(compare_output_hashes(root)).lower()}")
    return 0 if phase_6b_decision(validation) == "final_readme_and_reproducibility_package_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

