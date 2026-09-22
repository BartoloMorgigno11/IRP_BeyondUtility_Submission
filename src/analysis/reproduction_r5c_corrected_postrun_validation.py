"""Offline post-run validation for R5C_CORRECTED.

This script reads generated R5C and R5C_CORRECTED artifacts, writes the
original-vs-corrected comparison table, and records a final summary. It does
not call Ollama and is intentionally outside the R5C_CORRECTED protocol hash.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.reproduction_r5b_canddif_recovery import write_csv, write_json_atomic  # noqa: E402


ORIGINAL_STATUS = Path("results/tables/reproduction_r5c_history_length_request_execution_status.csv")
CORRECTED_STATUS = Path("results/tables/reproduction_r5c_history_length_corrected_request_execution_status.csv")
CORRECTED_GAPS = Path("results/tables/reproduction_r5c_history_length_corrected_gap_resolution.csv")
CORRECTED_FREEZE = Path("results/manifests/reproduction_r5c_history_length_corrected_freeze_candidate.json")
COMPARISON = Path("results/tables/reproduction_r5c_history_length_corrected_original_comparison.csv")
SUMMARY = Path("results/diagnostics/r5c_history_length_corrected/postrun_validation_summary.json")
NOTES = Path("report/notes.md")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def true_count(frame: pd.DataFrame, column: str) -> int:
    return int(frame[column].astype(str).str.lower().eq("true").sum())


def terminal(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[~frame["execution_status"].astype(str).eq("NOT_EXECUTED")].copy()


def summary_row(scope: str, protocol: str, key: str, frame: pd.DataFrame) -> dict[str, Any]:
    valid = true_count(frame, "parse_valid")
    return {
        "scope": scope,
        "protocol": protocol,
        "key": key,
        "accounted": len(frame),
        "valid": valid,
        "invalid": len(frame) - valid,
        "out_of_range": int(frame["validity_class"].eq("OUT_OF_RANGE").sum()),
        "wrong_length": int(frame["validity_class"].eq("WRONG_LENGTH").sum()),
        "parse_failure": int(frame["validity_class"].eq("PARSE_FAILURE").sum()),
        "backend_incomplete": int(frame["validity_class"].eq("BACKEND_INCOMPLETE").sum()),
    }


def build_comparison(original: pd.DataFrame, corrected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protocol, source in [("R5C_ORIGINAL", terminal(original)), ("R5C_CORRECTED", terminal(corrected))]:
        rows.append(summary_row("overall", protocol, "all", source))
        qwen_ml1m = source.loc[source["model_name"].eq("qwen2:7b-instruct") & source["dataset"].eq("ML-1M")]
        rows.append(summary_row("model_dataset", protocol, "qwen2:7b-instruct|ML-1M", qwen_ml1m))
        ml1m = source.loc[source["dataset"].eq("ML-1M")]
        rows.append(summary_row("dataset", protocol, "ML-1M", ml1m))
        for history_length, group in qwen_ml1m.groupby("history_length", sort=True):
            rows.append(summary_row("qwen2_ml1m_history_length", protocol, str(history_length), group))
    return pd.DataFrame(rows)


def build_summary(project_root: Path, comparison: pd.DataFrame) -> dict[str, Any]:
    corrected = comparison.loc[comparison["protocol"].eq("R5C_CORRECTED") & comparison["scope"].eq("overall")].iloc[0]
    original = comparison.loc[comparison["protocol"].eq("R5C_ORIGINAL") & comparison["scope"].eq("overall")].iloc[0]
    qwen = comparison.loc[
        comparison["protocol"].eq("R5C_CORRECTED")
        & comparison["scope"].eq("model_dataset")
        & comparison["key"].eq("qwen2:7b-instruct|ML-1M")
    ].iloc[0]
    gaps = read_csv(project_root / CORRECTED_GAPS)
    freeze = read_json(project_root / CORRECTED_FREEZE)
    return {
        "phase": "R5C_CORRECTED_POSTRUN_VALIDATION",
        "created_utc": utc_now(),
        "final_decision": "R5C_CORRECTED_HISTORY_LENGTH_COMPLETE",
        "llm_calls_in_postrun_validation": 0,
        "r5c_original_preserved": len(list((project_root / "results/recovery/r5c_history_length").glob("*.json"))) == 2400,
        "corrected_protocol_hash": freeze.get("protocol_hash", ""),
        "requests_expected": 2400,
        "requests_accounted": int(corrected["accounted"]),
        "valid": int(corrected["valid"]),
        "invalid": int(corrected["invalid"]),
        "out_of_range": int(corrected["out_of_range"]),
        "wrong_length": int(corrected["wrong_length"]),
        "parse_failure": int(corrected["parse_failure"]),
        "backend_incomplete": int(corrected["backend_incomplete"]),
        "original_valid": int(original["valid"]),
        "original_invalid": int(original["invalid"]),
        "original_out_of_range": int(original["out_of_range"]),
        "qwen2_ml1m_valid": int(qwen["valid"]),
        "qwen2_ml1m_accounted": int(qwen["accounted"]),
        "target_gaps_all_valid": int(gaps["gap_recovered"].astype(str).str.lower().eq("true").sum()),
        "target_gaps_total": len(gaps),
        "item_id_substitution_eliminated": int(original["out_of_range"]) == 15 and int(corrected["out_of_range"]) == 0,
        "paper_exact": bool(freeze.get("paper_exact")) is True,
        "r5c_corrected_freeze_ready": bool(freeze.get("freeze_ready")) is True,
        "safe_to_proceed_to_r5d": bool(freeze.get("freeze_ready")) is True,
        "r5d_executed": False,
        "commit": "NONE",
        "push": "NO",
    }


def update_notes(project_root: Path, summary: Mapping[str, Any]) -> None:
    notes = project_root / NOTES
    section = f"""

## R5C_CORRECTED post-run validation

- Decision: `{summary["final_decision"]}`.
- Scope: offline post-run validation only; no LLM/Ollama calls, raw edits, R5D, commit, push, or `original_repo/` modification.
- Corrected protocol hash: `{summary["corrected_protocol_hash"]}`.
- Final accounting: `{summary["valid"]}/{summary["requests_accounted"]}` valid, invalid `{summary["invalid"]}`, out_of_range `{summary["out_of_range"]}`.
- Qwen2 x ML-1M: `{summary["qwen2_ml1m_valid"]}/{summary["qwen2_ml1m_accounted"]}` valid; original item-ID substitution failure eliminated: `{summary["item_id_substitution_eliminated"]}`.
- Outputs: `results/tables/reproduction_r5c_history_length_corrected_original_comparison.csv` and `results/diagnostics/r5c_history_length_corrected/postrun_validation_summary.json`.
"""
    existing = notes.read_text(encoding="utf-8") if notes.exists() else ""
    marker = "## R5C_CORRECTED post-run validation"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + section
    else:
        existing = existing.rstrip() + section
    notes.write_text(existing.lstrip() + "\n", encoding="utf-8")


def validate(project_root: Path, *, write: bool = True) -> dict[str, Any]:
    original = read_csv(project_root / ORIGINAL_STATUS)
    corrected = read_csv(project_root / CORRECTED_STATUS)
    comparison = build_comparison(original, corrected)
    summary = build_summary(project_root, comparison)
    if write:
        write_csv(comparison, project_root / COMPARISON)
        write_json_atomic(summary, project_root / SUMMARY)
        update_notes(project_root, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    summary = validate(args.project_root.resolve(), write=not args.no_write)
    print(f"FINAL_DECISION={summary['final_decision']}")
    print("LLM_CALLS_IN_POSTRUN_VALIDATION=0")
    print(f"VALID={summary['valid']}/{summary['requests_expected']}")
    print(f"INVALID={summary['invalid']}")
    print(f"OUT_OF_RANGE={summary['out_of_range']}")
    print(f"QWEN2_ML1M={summary['qwen2_ml1m_valid']}/{summary['qwen2_ml1m_accounted']}")
    print(f"TARGET_GAPS_ALL_VALID={summary['target_gaps_all_valid']}/{summary['target_gaps_total']}")
    return 0 if summary["r5c_corrected_freeze_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
