"""Build the formal R5D freeze manifest from existing frozen artifacts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PROTOCOL_HASH = "AA76CD3E936A969A57CDA2E9E54E22B73D6C1D3DB65C0795F86E181CEDFE3B1A"
INVALID_POLICY = "FAIL_STRICT_ALL_MANIFEST_REQUESTS_INVALID_ZERO"

RAW_NAMESPACE = ROOT / "results" / "recovery" / "r5d_generation_involved_v2"
DIAGNOSTICS_DIR = ROOT / "results" / "diagnostics" / "r5d_generation_involved_v2"
TABLE_DIR = ROOT / "results" / "tables"
REPORTS = [
    ROOT / "report" / "reproduction_r5d_generation_involved_v2_recovery.md",
    ROOT / "report" / "reproduction_r5d_generation_involved_v2_frozen_completion.md",
]
SOURCES = [
    ROOT / "configs" / "reproduction_r5d_generation_involved_recovery.yaml",
    ROOT / "src" / "analysis" / "reproduction_r5d_generation_involved_recovery.py",
    ROOT / "src" / "analysis" / "reproduction_r5d_frozen_resume.py",
    ROOT / "src" / "analysis" / "reproduction_r5d_freeze_manifest.py",
    ROOT / "src" / "experiments" / "run_reproduction_r5d_generation_involved_recovery.py",
    ROOT / "tests" / "test_reproduction_r5d_generation_involved_recovery.py",
]
FREEZE_PATH = ROOT / "results" / "manifests" / "reproduction_r5d_generation_involved_v2_freeze.json"
COMPLETION_CANDIDATE = ROOT / "results" / "manifests" / "reproduction_r5d_generation_involved_v2_frozen_completion_candidate.json"


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def hash_inventory(paths: list[Path]) -> dict[str, str]:
    return {rel(path): sha256_file(path) for path in sorted(paths) if path.exists()}


def r5d_tables() -> list[Path]:
    return sorted(TABLE_DIR.glob("reproduction_r5d_generation_involved_v2_*.csv"))


def main() -> int:
    status_path = TABLE_DIR / "reproduction_r5d_generation_involved_v2_request_execution_status.csv"
    status = pd.read_csv(status_path)
    terminal = status.loc[~status["execution_status"].astype(str).eq("NOT_EXECUTED")]
    valid = int(terminal["parse_valid"].astype(str).str.lower().eq("true").sum())
    raw_paths = sorted(RAW_NAMESPACE.glob("*.json"))
    protocol_values = sorted(set(terminal["protocol_hash"].astype(str)))
    retries = int(terminal["retry_count"].astype(int).ne(0).sum())
    invalid = int(len(terminal) - valid)
    missing = int(status["execution_status"].astype(str).eq("NOT_EXECUTED").sum())
    completion = json.loads(COMPLETION_CANDIDATE.read_text(encoding="utf-8"))
    diagnostics = sorted(path for path in DIAGNOSTICS_DIR.glob("*") if path.is_file())
    payload: dict[str, Any] = {
        "freeze_metadata_version": "r5d_generation_involved_v2_freeze_v1",
        "decision": "R5D_FROZEN_PROTOCOL_COMPLETION_VALID",
        "created_from_head": git_output(["rev-parse", "HEAD"]),
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "protocol_hash_values_in_terminal_outputs": protocol_values,
        "raw_namespace": rel(RAW_NAMESPACE),
        "raw_count": len(raw_paths),
        "requests_expected": len(status),
        "requests_accounted": len(terminal),
        "valid": valid,
        "terminal_invalid": invalid,
        "missing": missing,
        "unexpected": max(0, len(raw_paths) - len(status)),
        "unauthorized_retries": retries,
        "duplicate_request_ids": int(status["request_id"].astype(str).duplicated().sum()),
        "duplicate_output_destinations": int(status["output_destination"].astype(str).duplicated().sum()),
        "invalid_policy": INVALID_POLICY,
        "target_gaps_accounted_under_invalid_policy": completion.get("target_gaps_accounted_under_invalid_policy"),
        "target_gaps_all_valid": completion.get("target_gaps_all_valid"),
        "target_gaps_total": completion.get("target_gaps_total"),
        "paper_exact": False,
        "freeze_ready": True,
        "raw_aggregate_sha256": completion.get("raw_aggregate_sha256"),
        "raw_file_hashes": hash_inventory(raw_paths),
        "table_hashes": hash_inventory(r5d_tables()),
        "diagnostic_hashes": hash_inventory(diagnostics),
        "report_hashes": hash_inventory(REPORTS),
        "source_hashes": hash_inventory(SOURCES),
        "completion_candidate_hash": sha256_file(COMPLETION_CANDIDATE),
    }
    checks = [
        payload["protocol_hash_values_in_terminal_outputs"] == [EXPECTED_PROTOCOL_HASH],
        payload["raw_count"] == 1200,
        payload["requests_expected"] == 1200,
        payload["requests_accounted"] == 1200,
        payload["valid"] == 1195,
        payload["terminal_invalid"] == 5,
        payload["missing"] == 0,
        payload["unexpected"] == 0,
        payload["unauthorized_retries"] == 0,
        payload["duplicate_request_ids"] == 0,
        payload["duplicate_output_destinations"] == 0,
        payload["invalid_policy"] == INVALID_POLICY,
        payload["target_gaps_accounted_under_invalid_policy"] == 4,
    ]
    payload["validation_pass"] = all(checks)
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"R5D_FREEZE_VALIDATION_PASS={payload['validation_pass']}")
    print(f"R5D_FREEZE_MANIFEST={rel(FREEZE_PATH)}")
    return 0 if payload["validation_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
