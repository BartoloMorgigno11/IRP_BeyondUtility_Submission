"""Build Phase F6E1 prompt-strategy recovery artifacts."""

from __future__ import annotations

from src.audit import common_git

import argparse
import subprocess
from pathlib import Path
from typing import Sequence

from src.llm.plan_f6e1_prompt_strategies import (
    EXPECTED_BRANCH,
    EXPECTED_HEAD,
    build_all,
    output_hashes,
    protected_input_summary,
)


def git_output(root: Path, args: Sequence[str]) -> str:
    return common_git.git_stdout_no_check(args, root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build F6E1 prompt strategy recovery artifacts.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output root used by tests. Defaults to the repository root.",
    )
    parser.add_argument(
        "--allow-unexpected-git",
        action="store_true",
        help="Allow build when branch/HEAD differ from the frozen audit state.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    branch = git_output(root, ["branch", "--show-current"])
    head = git_output(root, ["rev-parse", "HEAD"])
    if not args.allow_unexpected_git:
        if branch != EXPECTED_BRANCH:
            raise SystemExit(f"Unexpected branch: observed={branch} expected={EXPECTED_BRANCH}")
        if head != EXPECTED_HEAD:
            raise SystemExit(f"Unexpected HEAD: observed={head} expected={EXPECTED_HEAD}")

    before = output_hashes(args.output_root or root)
    result = build_all(root, args.output_root)
    after = output_hashes(args.output_root or root)
    protected = protected_input_summary(root)

    print(f"branch={branch}")
    print(f"head={head}")
    print(f"decision={result['decision']}")
    print(f"source_inventory_rows={len(result['frames']['prompt_source_inventory'])}")
    print(f"strategy_rows={len(result['frames']['prompt_strategy_spec'])}")
    print(f"placeholder_rows={len(result['frames']['placeholder_contract'])}")
    print(f"parser_rows={len(result['frames']['parser_contract'])}")
    print(f"model_strategy_rows={len(result['frames']['model_strategy_matrix'])}")
    print(f"missing_or_blocked_rows={len(result['frames']['missing_or_blocked'])}")
    print(f"scientific_content_hash={result['manifest']['scientific_content_hash']}")
    print(f"f2_protected_checked={protected['checked']}")
    print(f"f2_protected_missing={protected['missing']}")
    print(f"f2_protected_changed={protected['changed']}")
    print(f"second_hashes_match={str((not before) or before == after).lower()}")
    for path, digest in result["hashes"].items():
        print(f"sha256:{path}={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
