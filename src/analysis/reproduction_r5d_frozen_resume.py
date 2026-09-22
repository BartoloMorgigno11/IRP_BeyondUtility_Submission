"""Complete never-executed R5D frozen-protocol rows.

This script is deliberately outside the R5D protocol hash. It does not change
prompting, parsing, validation, model config, or output destinations. It only
checks that a partially completed frozen R5D run can be resumed without retrying
terminal rows, then executes rows still marked NOT_EXECUTED.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import reproduction_r5d_generation_involved_recovery as r5d


EXPECTED_PROTOCOL_HASH = "AA76CD3E936A969A57CDA2E9E54E22B73D6C1D3DB65C0795F86E181CEDFE3B1A"
ORIGINAL_INVALID_REQUEST_IDS = {
    "R5D-RANK-59C06DDCFD78BFF4",
    "R5D-RANK-E491A3315449174D",
    "R5D-RANK-167AC5F3AB797DBA",
}
RESUME_DECISION = "R5D_FROZEN_PROTOCOL_COMPLETION_VALID"
INVALID_POLICY = "FAIL_STRICT_ALL_MANIFEST_REQUESTS_INVALID_ZERO"


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def terminal_rows(status: pd.DataFrame) -> pd.DataFrame:
    return status.loc[~status["execution_status"].eq("NOT_EXECUTED")].copy()


def missing_rows(status: pd.DataFrame) -> pd.DataFrame:
    return status.loc[status["execution_status"].eq("NOT_EXECUTED")].copy()


def summarize_status(status: pd.DataFrame, raw_count: int) -> dict[str, Any]:
    terminal = terminal_rows(status)
    valid = int(truthy(terminal["parse_valid"]).sum())
    return {
        "requests_expected": int(len(status)),
        "requests_accounted": int(len(terminal)),
        "valid": valid,
        "invalid": int(len(terminal) - valid),
        "missing": int(len(status) - len(terminal)),
        "raw_count": int(raw_count),
        "retry_nonzero": int(terminal["retry_count"].astype(int).ne(0).sum()) if not terminal.empty else 0,
        "backend_incomplete": int(terminal["validity_class"].eq("BACKEND_INCOMPLETE").sum()) if not terminal.empty else 0,
        "backend_failed": int(terminal["execution_status"].eq("BACKEND_FAILED").sum()) if not terminal.empty else 0,
        "wrong_length": int(terminal["validity_class"].eq("WRONG_LENGTH").sum()) if not terminal.empty else 0,
        "out_of_range": int(terminal["validity_class"].eq("OUT_OF_RANGE").sum()) if not terminal.empty else 0,
        "duplicate": int(terminal["validity_class"].eq("DUPLICATE").sum()) if not terminal.empty else 0,
        "parse_failure": int(terminal["validity_class"].eq("PARSE_FAILURE").sum()) if not terminal.empty else 0,
        "item_id_substitution_values": int(terminal["dataset_item_id_substitution_count"].astype(int).sum()) if not terminal.empty else 0,
    }


def raw_count(project_root: Path, config: Mapping[str, Any]) -> int:
    namespace = r5d.result_namespace(project_root, config)
    return len(list(namespace.glob("*.json"))) if namespace.exists() else 0


def protocol_hash_from_freeze(project_root: Path, config: Mapping[str, Any]) -> str:
    freeze_path = r5d.diagnostics_dir(project_root, config) / "protocol_freeze.csv"
    if not freeze_path.exists():
        return ""
    freeze = pd.read_csv(freeze_path)
    protocol = freeze.loc[freeze["component"].astype(str).eq("protocol")]
    return str(protocol.iloc[0]["sha256"]) if not protocol.empty else ""


def compute_runtime_protocol(
    project_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    frame: pd.DataFrame,
    manifest_sha: str,
    dependency_validation: pd.DataFrame,
) -> tuple[str, pd.DataFrame, bool, str, dict[str, dict[str, Any]]]:
    runtime_available, runtime_error, runtime_tags = r5d.r5c.validate_runtime_available(
        config,
        (config.get("scope") or {}).get("models", []),
    )
    protocol_hash, protocol_freeze = r5d.protocol_metadata(
        project_root,
        config_path,
        config,
        frame,
        manifest_sha,
        dependency_validation,
        runtime_tags,
    )
    return protocol_hash, protocol_freeze, runtime_available, runtime_error, runtime_tags


def build_resume_preflight(
    project_root: Path,
    config: Mapping[str, Any],
    frame: pd.DataFrame,
    status: pd.DataFrame,
    *,
    manifest_sha: str,
    computed_protocol_hash: str,
    frozen_protocol_hash: str,
    runtime_available: bool,
    runtime_error: str,
) -> pd.DataFrame:
    terminal = terminal_rows(status)
    missing = missing_rows(status)
    invalid = terminal.loc[~truthy(terminal["parse_valid"])]
    original_invalid_present = set(invalid["request_id"].astype(str)) == ORIGINAL_INVALID_REQUEST_IDS
    original_invalid_not_missing = not ORIGINAL_INVALID_REQUEST_IDS.intersection(set(missing["request_id"].astype(str)))
    missing_destinations = [r5d.project_path(project_root, value) for value in missing["output_destination"].astype(str)]
    missing_outputs_absent = all(not path.exists() for path in missing_destinations)
    terminal_outputs_present = terminal["result_file_exists"].astype(str).str.lower().eq("true").all()
    terminal_retry_zero = terminal["retry_count"].astype(int).eq(0).all()
    output_unique = frame["output_destination"].astype(str).is_unique
    request_unique = frame["request_id"].astype(str).is_unique
    missing_all_ranking = missing["task"].astype(str).eq("ranking").all() if not missing.empty else False
    profile_status = status.loc[status["task"].eq("profile_generation")]
    profile_all_done_valid = (
        len(profile_status) == 400
        and profile_status["execution_status"].ne("NOT_EXECUTED").all()
        and truthy(profile_status["parse_valid"]).all()
    )
    dependency_ids = set(status.loc[truthy(status["parse_valid"]), "request_id"].astype(str))
    missing_dependency_ok = True
    for row in missing.to_dict("records"):
        dataset = str(row["dataset"])
        dep = str(row.get("r5d_profile_dependency_request_id", ""))
        if dataset in {"ML-1M", "LastFM"} and dep not in dependency_ids:
            missing_dependency_ok = False
            break
    checks: list[dict[str, Any]] = [
        {
            "check_id": "RESUME-001",
            "check_name": "manifest row count unchanged",
            "status": len(frame) == 1200,
            "observed": len(frame),
            "expected": 1200,
        },
        {
            "check_id": "RESUME-002",
            "check_name": "r5a manifest sha unchanged",
            "status": manifest_sha == r5d.EXPECTED_R5A_MANIFEST_SHA256,
            "observed": manifest_sha,
            "expected": r5d.EXPECTED_R5A_MANIFEST_SHA256,
        },
        {
            "check_id": "RESUME-003",
            "check_name": "stored frozen protocol hash unchanged",
            "status": frozen_protocol_hash == EXPECTED_PROTOCOL_HASH,
            "observed": frozen_protocol_hash,
            "expected": EXPECTED_PROTOCOL_HASH,
        },
        {
            "check_id": "RESUME-004",
            "check_name": "computed runtime protocol hash unchanged",
            "status": computed_protocol_hash == EXPECTED_PROTOCOL_HASH,
            "observed": computed_protocol_hash,
            "expected": EXPECTED_PROTOCOL_HASH,
        },
        {
            "check_id": "RESUME-005",
            "check_name": "runtime models available with frozen identifiers",
            "status": runtime_available,
            "observed": runtime_error or "available",
            "expected": "available",
        },
        {
            "check_id": "RESUME-006",
            "check_name": "completed terminal rows before resume",
            "status": len(terminal) == 600,
            "observed": len(terminal),
            "expected": 600,
        },
        {
            "check_id": "RESUME-007",
            "check_name": "remaining rows before resume",
            "status": len(missing) == 600,
            "observed": len(missing),
            "expected": 600,
        },
        {
            "check_id": "RESUME-008",
            "check_name": "terminal retry count zero",
            "status": terminal_retry_zero,
            "observed": int(terminal["retry_count"].astype(int).sum()) if not terminal.empty else 0,
            "expected": 0,
        },
        {
            "check_id": "RESUME-009",
            "check_name": "original invalids terminal and unchanged",
            "status": original_invalid_present,
            "observed": ";".join(sorted(invalid["request_id"].astype(str))),
            "expected": ";".join(sorted(ORIGINAL_INVALID_REQUEST_IDS)),
        },
        {
            "check_id": "RESUME-010",
            "check_name": "original invalids excluded from missing set",
            "status": original_invalid_not_missing,
            "observed": len(ORIGINAL_INVALID_REQUEST_IDS.intersection(set(missing["request_id"].astype(str)))),
            "expected": 0,
        },
        {
            "check_id": "RESUME-011",
            "check_name": "missing output files absent",
            "status": missing_outputs_absent,
            "observed": int(sum(path.exists() for path in missing_destinations)),
            "expected": 0,
        },
        {
            "check_id": "RESUME-012",
            "check_name": "terminal output files present",
            "status": terminal_outputs_present,
            "observed": int(terminal["result_file_exists"].astype(str).str.lower().eq("true").sum()),
            "expected": len(terminal),
        },
        {
            "check_id": "RESUME-013",
            "check_name": "request ids and output destinations unique",
            "status": request_unique and output_unique,
            "observed": f"requests_unique={request_unique};outputs_unique={output_unique}",
            "expected": "true;true",
        },
        {
            "check_id": "RESUME-014",
            "check_name": "missing rows are downstream ranking rows",
            "status": missing_all_ranking,
            "observed": ",".join(sorted(set(missing["task"].astype(str)))) if not missing.empty else "",
            "expected": "ranking",
        },
        {
            "check_id": "RESUME-015",
            "check_name": "profile-generation dependencies complete",
            "status": profile_all_done_valid and missing_dependency_ok,
            "observed": f"profiles_valid={int(truthy(profile_status['parse_valid']).sum())}/{len(profile_status)};missing_dependencies_ok={missing_dependency_ok}",
            "expected": "400/400;true",
        },
        {
            "check_id": "RESUME-016",
            "check_name": "invalid-output policy is manifest-denominator zero credit",
            "status": r5d.FAIL_STRICT_DENOMINATOR_POLICY == INVALID_POLICY,
            "observed": r5d.FAIL_STRICT_DENOMINATOR_POLICY,
            "expected": INVALID_POLICY,
        },
        {
            "check_id": "RESUME-017",
            "check_name": "max retries frozen to zero",
            "status": int((config.get("runtime") or {}).get("max_retries", -1)) == 0,
            "observed": int((config.get("runtime") or {}).get("max_retries", -1)),
            "expected": 0,
        },
    ]
    return pd.DataFrame(
        [
            {
                "check_id": check["check_id"],
                "check_name": check["check_name"],
                "status": "pass" if bool(check["status"]) else "fail",
                "observed": check["observed"],
                "expected": check["expected"],
            }
            for check in checks
        ]
    )


def rows_from_status(frame: pd.DataFrame, status_subset: pd.DataFrame) -> list[dict[str, Any]]:
    ids = set(status_subset["request_id"].astype(str))
    return [row for row in frame.to_dict("records") if str(row["request_id"]) in ids]


def chunked(rows: Sequence[Mapping[str, Any]], size: int) -> list[list[Mapping[str, Any]]]:
    return [list(rows[index : index + size]) for index in range(0, len(rows), size)]


def new_systematic_failure(
    status: pd.DataFrame,
    executed_ids: set[str],
    *,
    min_repeated_pattern: int = 3,
) -> tuple[bool, str]:
    new_terminal = status.loc[status["request_id"].astype(str).isin(executed_ids)]
    new_invalid = new_terminal.loc[~truthy(new_terminal["parse_valid"])]
    if new_invalid.empty:
        return False, ""
    if new_invalid["validity_class"].eq("BACKEND_INCOMPLETE").any():
        return True, "new backend incomplete output"
    if new_invalid["execution_status"].eq("BACKEND_FAILED").any():
        return True, "new backend failed output"
    grouping = ["task", "model_name", "dataset", "experimental_condition", "validity_class"]
    counts = new_invalid.groupby(grouping, dropna=False).size().reset_index(name="n")
    repeated = counts.loc[counts["n"].ge(min_repeated_pattern)]
    if repeated.empty:
        return False, ""
    row = repeated.iloc[0].to_dict()
    return (
        True,
        "new repeated invalid pattern "
        f"{row['task']}|{row['model_name']}|{row['dataset']}|{row['experimental_condition']}|"
        f"{row['validity_class']} n={row['n']}",
    )


def checkpoint_record(
    label: str,
    status: pd.DataFrame,
    executed_ids: set[str],
    raw_files: int,
) -> dict[str, Any]:
    summary = summarize_status(status, raw_files)
    new_terminal = status.loc[status["request_id"].astype(str).isin(executed_ids)]
    new_valid = int(truthy(new_terminal["parse_valid"]).sum())
    return {
        "checkpoint": label,
        **summary,
        "new_rows_terminal": len(new_terminal),
        "new_rows_valid": new_valid,
        "new_rows_invalid": int(len(new_terminal) - new_valid),
    }


def write_completion_artifacts(
    project_root: Path,
    config: Mapping[str, Any],
    frame: pd.DataFrame,
    manifest_sha: str,
    protocol_hash: str,
    protocol_freeze: pd.DataFrame,
    runtime_available: bool,
    runtime_error: str,
    runtime_tags: Mapping[str, Mapping[str, Any]],
    preflight: pd.DataFrame,
    checkpoints: pd.DataFrame,
    executed_this_run: int,
    decision: str,
) -> dict[str, Any]:
    status = r5d.build_status(project_root, frame, manifest_sha, protocol_hash)
    dependency = r5d.build_dependency_validation(frame)
    scope_summary = r5d.build_scope_summary(frame)
    metrics = r5d.build_metrics_by_condition(status)
    generation_results = r5d.build_generation_results(metrics)
    conditional = r5d.build_conditional_valid_metrics(status)
    dataset_model_summary = r5d.build_dataset_model_summary(status)
    gaps = r5d.build_gap_resolution(frame, status)
    runtime = r5d.build_runtime_provenance(
        project_root,
        config,
        status,
        runtime_available,
        runtime_error,
        runtime_tags,
        protocol_hash,
    )
    validation = build_resume_validation(status, preflight, gaps, decision)

    outputs = r5d.configured_outputs(config)
    frames = {
        "execution_manifest": r5d.build_execution_manifest(frame, manifest_sha, protocol_hash),
        "request_execution_status": status,
        "dependency_validation": dependency,
        "output_validity": r5d.build_output_validity(status),
        "metrics_by_condition": metrics,
        "generation_results": generation_results,
        "conditional_valid_metrics": conditional,
        "dataset_model_summary": dataset_model_summary,
        "gap_resolution": gaps,
        "runtime_provenance": runtime,
        "validation": validation,
        "scope_summary": scope_summary,
    }
    for key, value in frames.items():
        r5d.write_csv(value, project_root / outputs[key])

    diag = r5d.diagnostics_dir(project_root, config)
    diag.mkdir(parents=True, exist_ok=True)
    r5d.write_csv(protocol_freeze, diag / "protocol_freeze.csv")
    r5d.write_csv(preflight, diag / "frozen_resume_preflight.csv")
    r5d.write_csv(checkpoints, diag / "frozen_resume_checkpoints.csv")

    terminal = terminal_rows(status)
    valid = int(truthy(terminal["parse_valid"]).sum())
    gaps_policy_recovered = int(gaps["requests_accounted"].astype(int).eq(gaps["requests_expected"].astype(int)).sum())
    gaps_all_valid = int(gaps["gap_recovered"].eq("true").sum())
    raw_files = raw_count(project_root, config)
    freeze_ready = (
        decision == RESUME_DECISION
        and len(status) == 1200
        and len(terminal) == 1200
        and raw_files == 1200
        and int(terminal["retry_count"].astype(int).sum()) == 0
        and protocol_hash == EXPECTED_PROTOCOL_HASH
    )
    completion = {
        "freeze_metadata_version": "r5d_generation_involved_frozen_completion_candidate_v1",
        "phase": r5d.PHASE,
        "decision": decision,
        "created_utc": r5d.r5c.utc_now(),
        "r5a_manifest_sha256": manifest_sha,
        "protocol_hash": protocol_hash,
        "raw_namespace": r5d.normalize_rel((config.get("outputs") or {})["result_namespace"]),
        "raw_count": raw_files,
        "raw_aggregate_sha256": r5d.sha256_text(
            r5d.canonical_json(
                [
                    r5d.sha256_file(path)
                    for path in sorted(r5d.result_namespace(project_root, config).glob("*.json"))
                ]
            )
        ),
        "requests_expected": len(status),
        "requests_accounted": len(terminal),
        "valid": valid,
        "invalid": int(len(terminal) - valid),
        "invalid_policy": INVALID_POLICY,
        "invalid_policy_interpretation": "terminal invalid outputs remain in the manifest denominator and receive zero HR/NDCG credit",
        "target_gaps_accounted_under_invalid_policy": gaps_policy_recovered,
        "target_gaps_all_valid": gaps_all_valid,
        "target_gaps_total": len(gaps),
        "executed_this_resume": executed_this_run,
        "retried_rows": 0,
        "paper_exact": False,
        "freeze_ready_under_invalid_policy": freeze_ready,
        "all_valid_freeze_ready": False,
    }
    r5d.write_json_atomic(
        completion,
        project_root / "results" / "manifests" / "reproduction_r5d_generation_involved_v2_frozen_completion_candidate.json",
    )
    summary = {
        **completion,
        "status_counts": summarize_status(status, raw_files),
        "invalid_by_cell": invalid_by_cell(status),
        "failure_reasons": failure_reasons(status),
    }
    r5d.write_json_atomic(summary, diag / "frozen_resume_summary.json")
    report = render_completion_report(summary, metrics, gaps, preflight, checkpoints, validation)
    report_path = project_root / "report" / "reproduction_r5d_generation_involved_v2_frozen_completion.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return summary


def build_resume_validation(
    status: pd.DataFrame,
    preflight: pd.DataFrame,
    gaps: pd.DataFrame,
    decision: str,
) -> pd.DataFrame:
    terminal = terminal_rows(status)
    valid = int(truthy(terminal["parse_valid"]).sum())
    checks = [
        ("RESVAL-001", "offline resume preflight pass", preflight["status"].eq("pass").all(), f"{int(preflight['status'].eq('pass').sum())}/{len(preflight)}", f"{len(preflight)}/{len(preflight)}"),
        ("RESVAL-002", "all manifest rows accounted", len(terminal) == len(status) == 1200, f"{len(terminal)}/{len(status)}", "1200/1200"),
        ("RESVAL-003", "one raw per manifest row", len(terminal) == 1200, len(terminal), 1200),
        ("RESVAL-004", "no retries", terminal["retry_count"].astype(int).eq(0).all(), int(terminal["retry_count"].astype(int).sum()), 0),
        ("RESVAL-005", "original invalids retained", ORIGINAL_INVALID_REQUEST_IDS.issubset(set(terminal.loc[~truthy(terminal['parse_valid']), 'request_id'].astype(str))), "checked", "retained"),
        ("RESVAL-006", "invalids allowed by denominator policy", r5d.FAIL_STRICT_DENOMINATOR_POLICY == INVALID_POLICY, r5d.FAIL_STRICT_DENOMINATOR_POLICY, INVALID_POLICY),
        ("RESVAL-007", "target gaps accounted", gaps["requests_accounted"].astype(int).eq(gaps["requests_expected"].astype(int)).all(), f"{int(gaps['requests_accounted'].astype(int).eq(gaps['requests_expected'].astype(int)).sum())}/{len(gaps)}", f"{len(gaps)}/{len(gaps)}"),
        ("RESVAL-008", "decision matches frozen completion", decision == RESUME_DECISION, decision, RESUME_DECISION),
        ("RESVAL-009", "validity is not forced to all-valid", valid < len(terminal), f"{valid}/{len(terminal)}", "invalids terminal"),
    ]
    return pd.DataFrame(
        [
            {
                "check_id": check_id,
                "check_name": name,
                "status": "pass" if bool(ok) else "fail",
                "observed": observed,
                "expected": expected,
            }
            for check_id, name, ok, observed, expected in checks
        ]
    )


def invalid_by_cell(status: pd.DataFrame) -> list[dict[str, Any]]:
    terminal = terminal_rows(status)
    invalid = terminal.loc[~truthy(terminal["parse_valid"])]
    if invalid.empty:
        return []
    rows = []
    for key, group in invalid.groupby(["model_name", "dataset", "experimental_condition", "validity_class"], sort=True):
        model, dataset, condition, failure = key
        rows.append(
            {
                "model_name": model,
                "dataset": dataset,
                "experimental_condition": condition,
                "validity_class": failure,
                "n": int(len(group)),
                "request_ids": sorted(group["request_id"].astype(str).tolist()),
            }
        )
    return rows


def failure_reasons(status: pd.DataFrame) -> dict[str, int]:
    terminal = terminal_rows(status)
    invalid = terminal.loc[~truthy(terminal["parse_valid"])]
    if invalid.empty:
        return {}
    return {str(key): int(value) for key, value in invalid["validity_class"].value_counts().sort_index().items()}


def render_completion_report(
    summary: Mapping[str, Any],
    metrics: pd.DataFrame,
    gaps: pd.DataFrame,
    preflight: pd.DataFrame,
    checkpoints: pd.DataFrame,
    validation: pd.DataFrame,
) -> str:
    terminal = int(summary["requests_accounted"])
    valid = int(summary["valid"])
    invalid = int(summary["invalid"])
    return f"""# R5D Frozen-Protocol Completion

Decision: `{summary['decision']}`

Protocol hash: `{summary['protocol_hash']}`

## Accounting

- Requests expected: `1200`
- Requests accounted: `{terminal}`
- Valid: `{valid}`
- Invalid: `{invalid}`
- Invalid policy: `{summary['invalid_policy']}`
- Executed in frozen resume: `{summary['executed_this_resume']}`
- Retried rows: `0`

## Invalid Outputs

```json
{json.dumps(summary['invalid_by_cell'], indent=2)}
```

## Target Gaps

- Accounted under invalid-output policy: `{summary['target_gaps_accounted_under_invalid_policy']}/{summary['target_gaps_total']}`
- All-valid diagnostic: `{summary['target_gaps_all_valid']}/{summary['target_gaps_total']}`

{r5d.render_markdown_table(gaps, ['gap_id', 'dataset', 'requests_expected', 'requests_accounted', 'valid', 'invalid', 'gap_recovered'])}

## Metrics

{r5d.render_markdown_table(metrics, ['dataset', 'model_name', 'experimental_condition', 'n_manifest', 'n_valid', 'n_invalid', 'HR@5', 'NDCG@5'])}

## Resume Preflight

{r5d.render_markdown_table(preflight, ['check_id', 'check_name', 'status', 'observed', 'expected'])}

## Checkpoints

{r5d.render_markdown_table(checkpoints, ['checkpoint', 'requests_accounted', 'valid', 'invalid', 'missing', 'new_rows_terminal', 'new_rows_valid', 'new_rows_invalid'])}

## Validation

{r5d.render_markdown_table(validation, ['check_id', 'check_name', 'status', 'observed', 'expected'])}

## Interpretation

The three known invalid R5D outputs were not retried or repaired. They remain
terminal manifest rows and are counted in the denominator with zero HR/NDCG
credit under `{summary['invalid_policy']}`. This completion therefore freezes a
one-attempt-per-row R5D run, not an all-valid run.
"""


def run_resume(
    project_root: Path,
    config_path: Path,
    *,
    allow_api_calls: bool,
    allow_local_ollama: bool,
    batch_size: int,
) -> dict[str, Any]:
    config = r5d.load_config(config_path)
    _, frame, manifest_sha = r5d.load_r5d_manifest(project_root, config)
    dependency = r5d.build_dependency_validation(frame)
    protocol_hash, protocol_freeze, runtime_available, runtime_error, runtime_tags = compute_runtime_protocol(
        project_root,
        config_path,
        config,
        frame,
        manifest_sha,
        dependency,
    )
    status = r5d.build_status(project_root, frame, manifest_sha, EXPECTED_PROTOCOL_HASH)
    frozen_protocol_hash = protocol_hash_from_freeze(project_root, config)
    preflight = build_resume_preflight(
        project_root,
        config,
        frame,
        status,
        manifest_sha=manifest_sha,
        computed_protocol_hash=protocol_hash,
        frozen_protocol_hash=frozen_protocol_hash,
        runtime_available=runtime_available,
        runtime_error=runtime_error,
    )
    diag = r5d.diagnostics_dir(project_root, config)
    diag.mkdir(parents=True, exist_ok=True)
    r5d.write_csv(preflight, diag / "frozen_resume_preflight.csv")
    if not preflight["status"].eq("pass").all():
        return {
            "decision": "R5D_FROZEN_RESUME_NOT_ALLOWED",
            "llm_calls": 0,
            "preflight_pass": False,
            "preflight": preflight.to_dict("records"),
        }
    if not allow_api_calls or not allow_local_ollama:
        return {
            "decision": "R5D_FROZEN_RESUME_ALLOWED_NOT_EXECUTED",
            "llm_calls": 0,
            "preflight_pass": True,
            "preflight": preflight.to_dict("records"),
        }

    missing = missing_rows(status)
    rows = rows_from_status(frame, missing)
    executed_ids: set[str] = set()
    checkpoints: list[dict[str, Any]] = [
        checkpoint_record("before_resume", status, executed_ids, raw_count(project_root, config))
    ]
    executed = 0
    for index, batch in enumerate(chunked(rows, batch_size), start=1):
        batch_ids = {str(row["request_id"]) for row in batch}
        overlap = batch_ids.intersection(set(terminal_rows(status)["request_id"].astype(str)))
        if overlap:
            raise r5d.R5DError(f"resume batch overlaps terminal rows: {sorted(overlap)[:5]}")
        executed += r5d.execute_selected_requests(
            project_root,
            config,
            batch,
            manifest_sha,
            EXPECTED_PROTOCOL_HASH,
            allow_api_calls=allow_api_calls,
            allow_local_ollama=allow_local_ollama,
            gate_stage=f"frozen_resume_batch_{index:03d}",
        )
        executed_ids.update(batch_ids)
        status = r5d.build_status(project_root, frame, manifest_sha, EXPECTED_PROTOCOL_HASH)
        checkpoints.append(
            checkpoint_record(
                f"after_batch_{index:03d}",
                status,
                executed_ids,
                raw_count(project_root, config),
            )
        )
        breaker, reason = new_systematic_failure(status, executed_ids)
        checkpoint_frame = pd.DataFrame(checkpoints)
        r5d.write_csv(checkpoint_frame, diag / "frozen_resume_checkpoints.csv")
        if breaker:
            summary = write_completion_artifacts(
                project_root,
                config,
                frame,
                manifest_sha,
                EXPECTED_PROTOCOL_HASH,
                protocol_freeze,
                runtime_available,
                runtime_error,
                runtime_tags,
                preflight,
                checkpoint_frame,
                executed,
                "R5D_FROZEN_RESUME_CIRCUIT_BREAKER",
            )
            summary["circuit_breaker_reason"] = reason
            summary["llm_calls"] = executed
            r5d.write_json_atomic(summary, diag / "frozen_resume_summary.json")
            return summary
        print(
            f"checkpoint=after_batch_{index:03d} executed={executed} "
            f"accounted={checkpoints[-1]['requests_accounted']} "
            f"valid={checkpoints[-1]['valid']} invalid={checkpoints[-1]['invalid']}",
            flush=True,
        )

    final_status = r5d.build_status(project_root, frame, manifest_sha, EXPECTED_PROTOCOL_HASH)
    final_terminal = terminal_rows(final_status)
    final_raw = raw_count(project_root, config)
    decision = (
        RESUME_DECISION
        if len(final_status) == 1200
        and len(final_terminal) == 1200
        and final_raw == 1200
        and int(final_terminal["retry_count"].astype(int).sum()) == 0
        else "R5D_FULL_CORRECTED_RERUN_REQUIRED"
    )
    summary = write_completion_artifacts(
        project_root,
        config,
        frame,
        manifest_sha,
        EXPECTED_PROTOCOL_HASH,
        protocol_freeze,
        runtime_available,
        runtime_error,
        runtime_tags,
        preflight,
        pd.DataFrame(checkpoints),
        executed,
        decision,
    )
    summary["llm_calls"] = executed
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default=str(r5d.DEFAULT_CONFIG_PATH))
    parser.add_argument("--allow-api-calls", action="store_true")
    parser.add_argument("--allow-local-ollama", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    summary = run_resume(
        project_root,
        config_path,
        allow_api_calls=args.allow_api_calls,
        allow_local_ollama=args.allow_local_ollama,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary.get("decision") in {RESUME_DECISION, "R5D_FROZEN_RESUME_ALLOWED_NOT_EXECUTED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
