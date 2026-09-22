"""CLI entrypoint for the Phase F6E2 offline runtime preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.plan_f6e2_runtime_preflight import build_and_write


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic F6E2 runtime preflight artifacts.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root. Defaults to current directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hashes = build_and_write(args.root)
    print(json.dumps({"created_or_updated": hashes}, ensure_ascii=True, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

