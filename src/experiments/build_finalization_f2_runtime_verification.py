"""CLI wrapper for Finalization F2 runtime verification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.finalization_f2_runtime_verification import (  # noqa: E402
    F2OutputExistsError,
    build_all,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Finalization F2 runtime verification outputs.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=Path("configs/ml1m.yaml"))
    parser.add_argument("--skip-tests", action="store_true", help="Build inventories and hashes without running safe tests.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve()
    try:
        result = build_all(root, config_path=args.config, run_tests=not args.skip_tests)
    except F2OutputExistsError as exc:
        print(f"decision=blocked_existing_f2_outputs")
        print(str(exc))
        return 2
    decision = result["decision"]
    frames = result["frames"]
    print(f"decision={decision}")
    print(f"environment_rows={len(frames['environment_inventory'])}")
    print(f"test_inventory_rows={len(frames['test_inventory'])}")
    print(f"test_result_rows={len(frames['test_results'])}")
    print(f"protected_hash_rows={len(frames['hashes_before'])}")
    print(f"hash_changed_rows={frames['hash_validation']['status'].ne('unchanged').sum()}")
    print(f"canonical_registry_rows={len(frames['canonical_registry'])}")
    print(f"conflict_rows={len(frames['result_conflicts'])}")
    return 0 if decision == "finalization_f2_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

