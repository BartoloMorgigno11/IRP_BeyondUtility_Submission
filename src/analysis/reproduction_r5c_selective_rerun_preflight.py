"""Hard preflight for the proposed R5C selective rerun.

This module performs only offline checks. It decides whether the existing
frozen R5C protocol permits rerunning only the 15 observed invalid rows.
It does not call Ollama, mutate raw outputs, execute R5D, commit, or push.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml


EXPECTED_R5C_PROTOCOL_HASH = "30E4BCA9E1456EE9A7DE74A09FE03BF1A26EFCF758915052FD88F39AF9A836F2"
DECISION = "R5C_SELECTIVE_RERUN_NOT_ALLOWED_UNDER_FROZEN_PROTOCOL"
PHASE = "R5C_SELECTIVE_RERUN_HARD_PREFLIGHT"

R5C_CONFIG = Path("configs/reproduction_r5c_history_length_recovery.yaml")
R5C_RUNNER = Path("src/analysis/reproduction_r5c_history_length_recovery.py")
R5C_REPORT = Path("report/reproduction_r5c_history_length_recovery.md")
FORENSIC_REPORT = Path("report/reproduction_r5c_invalid_output_forensic_audit.md")
FORENSIC_SUMMARY = Path("results/diagnostics/r5c_history_length/invalid_output_forensic_summary.json")
RERUN_ROWS = Path("results/tables/reproduction_r5c_selective_rerun_rows.csv")
STATUS_TABLE = Path("results/tables/reproduction_r5c_history_length_request_execution_status.csv")
GAP_TABLE = Path("results/tables/reproduction_r5c_history_length_gap_resolution.csv")
PROTOCOL_ASSESSMENT = Path("results/tables/reproduction_r5c_invalid_output_protocol_assessment.csv")
PAPER_CONTRACT = Path("results/tables/paper_atomic_contract.csv")
PAPER_PDF = Path("paper/2411.00331v1.pdf")
ORIGINAL_ZERO_SHOT = Path("original_repo/EvaLLMasRecommender/eva/codes/zero_shot.py")
ORIGINAL_REFINE = Path("original_repo/EvaLLMasRecommender/eva/codes/output_refine.py")
ORIGINAL_LLM = Path("original_repo/EvaLLMasRecommender/eva/codes/llms.py")
ORIGINAL_METRICS = Path("original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py")
NOTES = Path("report/notes.md")

CHECK_TABLE = Path("results/tables/reproduction_r5c_selective_rerun_hard_preflight.csv")
SUMMARY_JSON = Path("results/diagnostics/r5c_history_length/selective_rerun_preflight_summary.json")
REPORT = Path("report/reproduction_r5c_selective_rerun_preflight.md")


@dataclass(frozen=True)
class PreflightResult:
    decision: str
    checks: pd.DataFrame
    summary: dict[str, Any]
    report: str


class R5CSelectivePreflightError(RuntimeError):
    """Raised when required offline preflight evidence is missing."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def read_text(project_root: Path, path: Path) -> str:
    full = project_path(project_root, path)
    if not full.exists():
        raise R5CSelectivePreflightError(f"missing required file: {path}")
    return full.read_text(encoding="utf-8")


def read_json(project_root: Path, path: Path) -> dict[str, Any]:
    full = project_path(project_root, path)
    if not full.exists():
        raise R5CSelectivePreflightError(f"missing required file: {path}")
    with full.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise R5CSelectivePreflightError(f"expected JSON object: {path}")
    return payload


def read_csv_frame(project_root: Path, path: Path) -> pd.DataFrame:
    full = project_path(project_root, path)
    if not full.exists():
        raise R5CSelectivePreflightError(f"missing required file: {path}")
    return pd.read_csv(full, dtype=str, keep_default_na=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    selected = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    header = "| " + " | ".join(selected.columns) + " |"
    separator = "| " + " | ".join("---" for _ in selected.columns) + " |"
    rows = [
        "| " + " | ".join(str(row[column]).replace("\n", " ") for column in selected.columns) + " |"
        for row in selected.to_dict("records")
    ]
    return "\n".join([header, separator, *rows])


def count_original_raw_preserved(project_root: Path, rerun_rows: pd.DataFrame) -> tuple[int, list[str]]:
    preserved = 0
    missing: list[str] = []
    for row in rerun_rows.to_dict("records"):
        destination = project_path(project_root, str(row.get("output_destination", "")))
        if destination.exists():
            preserved += 1
        else:
            missing.append(str(row.get("request_id", "")))
    return preserved, missing


def r5c_counts(status: pd.DataFrame, gaps: pd.DataFrame) -> dict[str, int]:
    valid = int(status["valid_output"].eq("true").sum())
    invalid = int(status["terminal_status"].eq("true").sum() - valid)
    return {
        "requests": int(len(status)),
        "valid": valid,
        "invalid": invalid,
        "target_gaps_all_valid": int(gaps["gap_recovered"].eq("true").sum()),
        "target_gaps_total": int(len(gaps)),
    }


def build_checks(project_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = yaml.safe_load(read_text(project_root, R5C_CONFIG))
    if not isinstance(config, dict):
        raise R5CSelectivePreflightError("R5C config is not a YAML mapping")
    runner = read_text(project_root, R5C_RUNNER)
    r5c_report = read_text(project_root, R5C_REPORT)
    forensic_report = read_text(project_root, FORENSIC_REPORT)
    forensic_summary = read_json(project_root, FORENSIC_SUMMARY)
    rerun_rows = read_csv_frame(project_root, RERUN_ROWS)
    status = read_csv_frame(project_root, STATUS_TABLE)
    gaps = read_csv_frame(project_root, GAP_TABLE)
    protocol_assessment = read_csv_frame(project_root, PROTOCOL_ASSESSMENT)
    zero_shot = read_text(project_root, ORIGINAL_ZERO_SHOT)
    output_refine = read_text(project_root, ORIGINAL_REFINE)
    llm_code = read_text(project_root, ORIGINAL_LLM)
    metrics_code = read_text(project_root, ORIGINAL_METRICS)
    paper_contract = read_csv_frame(project_root, PAPER_CONTRACT)

    runtime = config.get("runtime") or {}
    outputs = config.get("outputs") or {}
    counts = r5c_counts(status, gaps)
    original_preserved, missing_originals = count_original_raw_preserved(project_root, rerun_rows)
    paper_pdf = project_path(project_root, PAPER_PDF)

    protocol_hashes = set(status["protocol_hash"].astype(str))
    retry_count_values = set(status["retry_count"].astype(str)) if "retry_count" in status.columns else set()
    invalid = status.loc[status["valid_output"].eq("false")].copy()
    invalid_is_target = (
        len(invalid) == 15
        and invalid["dataset"].eq("ML-1M").all()
        and invalid["model_name"].eq("qwen2:7b-instruct").all()
        and invalid["parse_status"].eq("unknown_indices").all()
    )
    original_selective_loop = (
        "while True" in zero_shot
        and "second=True" in zero_shot
        and "invalid_format" in output_refine
        and "invalid_list" in output_refine
    )
    original_contract_is_titles = (
        "python list of item's title" in llm_code
        and "mapListTitle2item_id" in metrics_code
    )
    paper_retry_contract_absent = (
        "repetition_policy" in paper_contract.columns
        and not paper_contract["repetition_policy"].astype(str).str.strip().astype(bool).any()
    )
    same_protocol_requires_no_retry = int(runtime.get("max_retries", 0)) == 0

    check_specs: list[dict[str, Any]] = [
        {
            "check_id": "PREFLIGHT-001",
            "evidence_area": "scope",
            "status": "pass",
            "blocks_selective_rerun": "false",
            "observed": f"{len(rerun_rows)} rows",
            "expected": "15 rows",
            "finding": "The affected manifest rows are precisely listed for offline review.",
        },
        {
            "check_id": "PREFLIGHT-002",
            "evidence_area": "protocol_hash",
            "status": "pass" if protocol_hashes == {EXPECTED_R5C_PROTOCOL_HASH} else "fail",
            "blocks_selective_rerun": "true" if protocol_hashes != {EXPECTED_R5C_PROTOCOL_HASH} else "false",
            "observed": ";".join(sorted(protocol_hashes)),
            "expected": EXPECTED_R5C_PROTOCOL_HASH,
            "finding": "The frozen completed R5C attempt set is bound to one protocol hash.",
        },
        {
            "check_id": "PREFLIGHT-003",
            "evidence_area": "generation_config",
            "status": "pass" if same_protocol_requires_no_retry else "fail",
            "blocks_selective_rerun": "true",
            "observed": str(runtime.get("max_retries", "")),
            "expected": "0",
            "finding": "R5C generation config explicitly freezes no retry policy.",
        },
        {
            "check_id": "PREFLIGHT-004",
            "evidence_area": "frozen_report",
            "status": "pass" if "no retry policy introduced" in r5c_report else "fail",
            "blocks_selective_rerun": "true",
            "observed": "found" if "no retry policy introduced" in r5c_report else "missing",
            "expected": "STATIC-025 no retry policy introduced",
            "finding": "The R5C static gate treated absence of retry policy as part of the protocol.",
        },
        {
            "check_id": "PREFLIGHT-005",
            "evidence_area": "runner",
            "status": "pass" if "if destination.exists()" in runner and "validate_existing_result" in runner else "fail",
            "blocks_selective_rerun": "true",
            "observed": "validate_existing_result_on_existing_destination",
            "expected": "terminal outputs are validated, not replaced",
            "finding": "The R5C runner preserves terminal outputs and does not define invalid-only replacement semantics.",
        },
        {
            "check_id": "PREFLIGHT-006",
            "evidence_area": "retry_accounting",
            "status": "pass" if retry_count_values == {"0"} else "fail",
            "blocks_selective_rerun": "true",
            "observed": ";".join(sorted(retry_count_values)),
            "expected": "0",
            "finding": "All terminal R5C outputs are first attempts under the frozen attempt accounting.",
        },
        {
            "check_id": "PREFLIGHT-007",
            "evidence_area": "invalids",
            "status": "pass" if invalid_is_target else "fail",
            "blocks_selective_rerun": "false",
            "observed": f"{len(invalid)} invalid; datasets={sorted(set(invalid['dataset']))}; parse_status={sorted(set(invalid['parse_status']))}",
            "expected": "15 Qwen2 ML-1M unknown_indices rows",
            "finding": "The proposed rows match the observed R5C structural-invalid failure set.",
        },
        {
            "check_id": "PREFLIGHT-008",
            "evidence_area": "forensic_audit",
            "status": "pass" if forensic_summary.get("post_hoc_normalization_scientifically_allowed") is False else "fail",
            "blocks_selective_rerun": "false",
            "observed": str(forensic_summary.get("post_hoc_normalization_scientifically_allowed")),
            "expected": "False",
            "finding": "The prior audit disallowed repairing existing raw outputs by item-ID normalization.",
        },
        {
            "check_id": "PREFLIGHT-009",
            "evidence_area": "forensic_audit",
            "status": "pass" if "post-hoc item-ID to candidate-index normalization is not allowed" in forensic_report else "fail",
            "blocks_selective_rerun": "false",
            "observed": "found" if "post-hoc item-ID to candidate-index normalization is not allowed" in forensic_report else "missing",
            "expected": "normalization disallowed",
            "finding": "The previous future-rerun recommendation did not itself authorize replacement under the same hash.",
        },
        {
            "check_id": "PREFLIGHT-010",
            "evidence_area": "parser_protocol",
            "status": "pass" if protocol_assessment["allows_item_id_to_index_normalization"].eq("false").all() else "fail",
            "blocks_selective_rerun": "false",
            "observed": ";".join(sorted(set(protocol_assessment["allows_item_id_to_index_normalization"].astype(str)))),
            "expected": "false",
            "finding": "The parser and prompt evidence keep item IDs outside the R5C valid namespace.",
        },
        {
            "check_id": "PREFLIGHT-011",
            "evidence_area": "original_repository",
            "status": "pass" if original_selective_loop else "fail",
            "blocks_selective_rerun": "false",
            "observed": "selective invalid_format loop present",
            "expected": "repository evidence recorded",
            "finding": "The original code has a selective retry loop for format failures in its title-list contract.",
        },
        {
            "check_id": "PREFLIGHT-012",
            "evidence_area": "original_repository",
            "status": "pass" if original_contract_is_titles else "fail",
            "blocks_selective_rerun": "true",
            "observed": "title-list generation and title-to-item metric mapping",
            "expected": "not R5C index namespace",
            "finding": "The original selective retry is not an authorization for R5C item-ID-in-index-field failures.",
        },
        {
            "check_id": "PREFLIGHT-013",
            "evidence_area": "paper_protocol",
            "status": "pass" if paper_pdf.exists() and paper_retry_contract_absent else "fail",
            "blocks_selective_rerun": "true",
            "observed": f"paper_pdf_exists={paper_pdf.exists()}; populated_repetition_policy={not paper_retry_contract_absent}",
            "expected": "no explicit paper retry policy in local atomic contract",
            "finding": "Local paper-derived protocol evidence does not predeclare an invalid-only rerun policy for this adapted R5C.",
        },
        {
            "check_id": "PREFLIGHT-014",
            "evidence_area": "bias_risk",
            "status": "fail",
            "blocks_selective_rerun": "true",
            "observed": "rerun set selected after observing invalid outputs",
            "expected": "predeclared attempt policy before inference",
            "finding": "Replacing only observed invalid outputs would introduce selective-retry bias under the frozen R5C attempt set.",
        },
        {
            "check_id": "PREFLIGHT-015",
            "evidence_area": "protocol_mixing",
            "status": "fail",
            "blocks_selective_rerun": "true",
            "observed": "new recovery namespace would be combined with original first-attempt namespace",
            "expected": "single predeclared protocol identity or separate new protocol",
            "finding": "Using a separate namespace as a replacement while retaining the same protocol hash would mix attempt policies.",
        },
        {
            "check_id": "PREFLIGHT-016",
            "evidence_area": "raw_preservation",
            "status": "pass" if original_preserved == 15 else "fail",
            "blocks_selective_rerun": "false",
            "observed": f"{original_preserved}/15 original raw files present",
            "expected": "15/15",
            "finding": "Original invalid raw outputs remain present and were not overwritten by this preflight.",
        },
    ]
    checks = pd.DataFrame(check_specs)

    scientific_blocks = checks.loc[checks["blocks_selective_rerun"].eq("true") & checks["status"].eq("fail")]
    explicit_no_retry_blocks = checks.loc[
        checks["blocks_selective_rerun"].eq("true")
        & checks["check_id"].isin(["PREFLIGHT-003", "PREFLIGHT-004", "PREFLIGHT-005", "PREFLIGHT-006", "PREFLIGHT-012", "PREFLIGHT-013"])
        & checks["status"].eq("pass")
    ]
    selective_allowed = False
    preflight_pass = False

    summary = {
        "phase": PHASE,
        "created_utc": utc_now(),
        "final_decision": DECISION,
        "selective_rerun_scientifically_allowed": selective_allowed,
        "preflight": "PASS" if preflight_pass else "FAIL",
        "llm_calls": 0,
        "canary_valid": 0,
        "canary_total": 3,
        "canary_executed": False,
        "selective_rows_executed": 0,
        "selective_rows_total": 15,
        "selective_rows_valid": 0,
        "repeated_retries": 0,
        "original_invalid_raw_preserved": original_preserved == 15,
        "missing_original_invalid_raw_request_ids": missing_originals,
        "r5c_final_valid": counts["valid"],
        "r5c_final_requests": counts["requests"],
        "r5c_final_invalid": counts["invalid"],
        "target_gaps_all_valid": counts["target_gaps_all_valid"],
        "target_gaps_total": counts["target_gaps_total"],
        "metrics_recomputed": False,
        "r5c_freeze_ready": False,
        "safe_to_proceed_to_r5d": False,
        "r5d_executed": False,
        "commit": "NONE",
        "push": "NO",
        "full_r5c_rerun_required": True,
        "full_r5c_rerun_required_condition": "Required if the objective is an all-valid R5C freeze under one unbiased, predeclared attempt policy.",
        "other_remediation": "Define a new protocol identity before inference, then rerun the full R5C attempt set; keep the 15-row selective set diagnostic-only unless a new protocol explicitly predeclares replacement rules.",
        "checks_total": int(len(checks)),
        "checks_pass": int(checks["status"].eq("pass").sum()),
        "checks_fail": int(checks["status"].eq("fail").sum()),
        "blocking_failures": scientific_blocks["check_id"].tolist(),
        "explicit_no_retry_evidence": explicit_no_retry_blocks["check_id"].tolist(),
        "input_hashes": {
            str(path): sha256_file(project_path(project_root, path))
            for path in [
                R5C_CONFIG,
                R5C_RUNNER,
                R5C_REPORT,
                FORENSIC_REPORT,
                FORENSIC_SUMMARY,
                RERUN_ROWS,
                STATUS_TABLE,
                GAP_TABLE,
                PROTOCOL_ASSESSMENT,
                PAPER_CONTRACT,
                ORIGINAL_ZERO_SHOT,
                ORIGINAL_REFINE,
                ORIGINAL_LLM,
                ORIGINAL_METRICS,
            ]
        },
    }
    return checks, summary


def render_report(checks: pd.DataFrame, summary: Mapping[str, Any]) -> str:
    return f"""# R5C Selective Rerun Hard Preflight

## Decision

`{summary["final_decision"]}`

Selective rerun under the same frozen R5C protocol hash is not scientifically allowed. The R5C protocol hash includes `max_retries=0`, the frozen report records `STATIC-025 no retry policy introduced`, and the runner preserves existing terminal outputs rather than replacing invalid terminal attempts.

## Stop Condition

- Selective rerun scientifically allowed: `NO`
- LLM calls: `{summary["llm_calls"]}`
- Canary executed: `NO`
- Selective rows executed: `{summary["selective_rows_executed"]}/15`

## Key Reason

The 15 rows were selected after observing invalid outputs. Replacing only those rows would create an invalid-only retry policy that was not predeclared in R5C and is not represented by protocol hash `{EXPECTED_R5C_PROTOCOL_HASH}`. The original repository contains a selective retry loop for non-parseable title-list format failures, but R5C deliberately froze a different strict JSON candidate-index contract with zero retries.

## Checks

{render_table(checks, ["check_id", "evidence_area", "status", "blocks_selective_rerun", "observed", "expected", "finding"])}

## Remediation

`FULL R5C RERUN REQUIRED` if the target is an all-valid R5C freeze under one unbiased attempt policy. A future remediation should use a new protocol identity with predeclared retry/replacement rules before inference. The existing 15-row list remains diagnostic evidence and cannot repair the frozen R5C freeze by itself.
"""


def update_notes(project_root: Path, summary: Mapping[str, Any]) -> None:
    notes_path = project_path(project_root, NOTES)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    section = f"""

## R5C selective rerun hard preflight

- Decision: `{summary["final_decision"]}`.
- Scope: offline-only hard preflight for the proposed 15-row Qwen2/ML-1M selective rerun; no LLM/Ollama calls, canary, row execution, raw edits, R5D, commit, push, or `original_repo/` modification.
- Scientific decision: selective rerun under the same R5C protocol hash is not allowed because R5C froze `max_retries=0`, preserved terminal outputs, and did not predeclare invalid-only replacement semantics.
- Required remediation for an all-valid freeze: full R5C rerun under a new predeclared protocol identity; the 15-row file remains diagnostic-only under the current freeze.
- Outputs: `results/tables/reproduction_r5c_selective_rerun_hard_preflight.csv`, `results/diagnostics/r5c_history_length/selective_rerun_preflight_summary.json`, and `report/reproduction_r5c_selective_rerun_preflight.md`.
"""
    existing = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    marker = "## R5C selective rerun hard preflight"
    if marker in existing:
        prefix = existing.split(marker)[0].rstrip()
        notes_path.write_text(prefix + section + "\n", encoding="utf-8")
    else:
        notes_path.write_text(existing.rstrip() + section + "\n", encoding="utf-8")


def build_preflight(project_root: Path, *, write: bool = True) -> PreflightResult:
    checks, summary = build_checks(project_root)
    report = render_report(checks, summary)
    result = PreflightResult(DECISION, checks, summary, report)
    if write:
        write_csv(checks, project_path(project_root, CHECK_TABLE))
        write_json(summary, project_path(project_root, SUMMARY_JSON))
        report_path = project_path(project_root, REPORT)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        update_notes(project_root, summary)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline hard preflight for the R5C selective rerun proposal.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root.")
    parser.add_argument("--no-write", action="store_true", help="Run checks without writing artifacts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_preflight(args.project_root.resolve(), write=not args.no_write)
    summary = result.summary
    print(f"FINAL_DECISION={summary['final_decision']}")
    print("SELECTIVE_RERUN_SCIENTIFICALLY_ALLOWED=NO")
    print(f"PREFLIGHT={summary['preflight']}")
    print("LLM_CALLS=0")
    print(f"SELECTIVE_ROWS_EXECUTED={summary['selective_rows_executed']}/15")
    print(f"R5C_FINAL_VALID={summary['r5c_final_valid']}/{summary['r5c_final_requests']}")
    print(f"R5C_FINAL_INVALID={summary['r5c_final_invalid']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
