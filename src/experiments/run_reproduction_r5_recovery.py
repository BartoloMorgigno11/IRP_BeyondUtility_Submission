"""Dry-run checker for a future R5 recovery manifest.

This script intentionally does not execute recovery requests. It exists so the
R5A manifest can be inspected with the same entry point that a future approved
runner would extend.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "request_id",
    "recovery_family",
    "dataset",
    "model_name",
    "task",
    "user_id",
    "experimental_condition",
    "candidate_list_hash",
    "history_hash",
    "prompt_hash",
    "material_context_hash",
    "output_destination",
    "execution_allowed_in_r5a",
    "inference_decision",
}


def dry_run_manifest(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = pd.read_csv(path, keep_default_na=False, low_memory=False)
    missing = REQUIRED_COLUMNS.difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing required columns: {sorted(missing)}")
    if not manifest["request_id"].is_unique:
        raise ValueError("manifest request_id values are not unique")
    execution_allowed = manifest["execution_allowed_in_r5a"].astype(str).str.lower()
    if not execution_allowed.eq("false").all():
        raise ValueError("R5A manifest unexpectedly permits execution")
    return {
        "manifest_rows": len(manifest),
        "profile_generation_rows": int(manifest["task"].eq("profile_generation").sum()),
        "ranking_rows": int(manifest["task"].eq("ranking").sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="results/tables/reproduction_r5a_recovery_request_manifest.csv",
    )
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-api-calls", action="store_true")
    args = parser.parse_args()

    if args.execute:
        if not args.allow_api_calls:
            print("R5 execution requires --allow-api-calls and explicit future implementation.")
            return 2
        print("R5 execution is not implemented in R5A; dry-run manifest validation only.")
        return 2

    summary = dry_run_manifest(Path(args.manifest))
    print("dry_run=true")
    for key, value in summary.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
