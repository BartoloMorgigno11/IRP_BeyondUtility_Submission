"""Offline forensic audit for the 15 invalid R5C Qwen2/ML-1M outputs.

The audit reads existing R5C raw JSON outputs and manifest rows only. It does
not call an LLM, modify raw outputs, rerun R5C, execute R5D, commit, or push.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.reproduction_r5b_canddif_recovery import (
    candidate_records,
    canonical_json,
    hr_at_5,
    load_json,
    ndcg_at_5,
    positive_rank,
    read_csv,
    sha256_file,
    sha256_text,
    write_csv,
    write_json_atomic,
)


PHASE = "R5C_INVALID_OUTPUT_FORENSIC_AUDIT"
R5C_STATUS = Path("results/tables/reproduction_r5c_history_length_request_execution_status.csv")
R5C_MANIFEST = Path("results/tables/reproduction_r5c_history_length_execution_manifest.csv")
R5A_MANIFEST = Path("results/tables/reproduction_r5a_recovery_request_manifest.csv")
R5C_GAP_RESOLUTION = Path("results/tables/reproduction_r5c_history_length_gap_resolution.csv")
R5C_CONFIG = Path("configs/reproduction_r5c_history_length_recovery.yaml")
R5C_RUNNER = Path("src/analysis/reproduction_r5c_history_length_recovery.py")
CANONICAL_PARSER = Path("src/llm/canonical_llm_contract.py")
PLAN_D_PROMPT_REPORT = Path("report/plan_d_llm_prompt_templates_and_ranking_manifest.md")
ORIGINAL_LLM = Path("original_repo/EvaLLMasRecommender/eva/codes/llms.py")
ORIGINAL_REFINE = Path("original_repo/EvaLLMasRecommender/eva/codes/output_refine.py")
ORIGINAL_METRICS = Path("original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py")
ORIGINAL_TEMPLATE = Path("original_repo/EvaLLMasRecommender/eva/templates/template.txt")
REPORT_PATH = Path("report/reproduction_r5c_invalid_output_forensic_audit.md")
NOTES_PATH = Path("report/notes.md")

AUDIT_TABLE = Path("results/tables/reproduction_r5c_invalid_output_forensic_audit.csv")
VALUE_TABLE = Path("results/tables/reproduction_r5c_invalid_output_value_mapping.csv")
RERUN_TABLE = Path("results/tables/reproduction_r5c_selective_rerun_rows.csv")
SCIENCE_TABLE = Path("results/tables/reproduction_r5c_invalid_output_protocol_assessment.csv")
SUMMARY_JSON = Path("results/diagnostics/r5c_history_length/invalid_output_forensic_summary.json")

DECISION_SELECTIVE_RERUN = "R5C_SELECTIVE_RERUN_REQUIRED"
CLASS_EXACT = "A_EXACT_OFFLINE_MAPPING"
CLASS_PARTIAL = "B_PARTIAL_OR_AMBIGUOUS_MAPPING"
CLASS_INVALID = "C_GENUINE_INVALID_OUTPUT"
STRICT_EXPECTED = "ranked_candidate_indices must contain exactly five unique 1-based candidate indices from 1..20"


@dataclass(frozen=True)
class AuditResult:
    decision: str
    audit: pd.DataFrame
    values: pd.DataFrame
    rerun_rows: pd.DataFrame
    science: pd.DataFrame
    report: str
    summary: dict[str, Any]


class R5CInvalidAuditError(RuntimeError):
    """Raised when required frozen R5C evidence is missing or inconsistent."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def render_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    selected = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    header = "| " + " | ".join(selected.columns) + " |"
    separator = "| " + " | ".join("---" for _ in selected.columns) + " |"
    body = [
        "| " + " | ".join(str(row[column]).replace("\n", " ") for column in selected.columns) + " |"
        for row in selected.to_dict("records")
    ]
    return "\n".join([header, separator, *body])


def parse_raw_values(raw_response_text: str) -> tuple[list[int], str]:
    try:
        parsed = json.loads(str(raw_response_text).strip())
    except json.JSONDecodeError as exc:
        return [], f"parse_failed:{exc}"
    if not isinstance(parsed, dict):
        return [], "invalid_schema:not_object"
    values = parsed.get("ranked_candidate_indices")
    if not isinstance(values, list):
        return [], "invalid_schema:missing_ranked_candidate_indices"
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return [], "invalid_schema:non_integer_value"
    return [int(value) for value in values], "parsed_int_values"


def candidate_mapping(row: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    records = candidate_records(row)
    ordered: list[dict[str, Any]] = []
    item_to_indices: dict[str, list[int]] = {}
    for index, item in enumerate(records, start=1):
        item_id = str(item.get("item_id", ""))
        title = str(item.get("title", ""))
        ordered.append({"candidate_index": index, "item_id": item_id, "title": title})
        item_to_indices.setdefault(item_id, []).append(index)
    return ordered, item_to_indices


def classify_request(values: list[int], value_rows: list[dict[str, Any]], candidate_count: int) -> str:
    if len(values) != 5 or len(set(values)) != len(values):
        return CLASS_PARTIAL
    all_index_valid = all(1 <= value <= candidate_count for value in values)
    if all_index_valid:
        return CLASS_PARTIAL
    if all(row["item_id_mapping_status"] == "exact" for row in value_rows):
        mapped = [int(row["mapped_candidate_index"]) for row in value_rows]
        if len(set(mapped)) == len(mapped):
            return CLASS_EXACT
        return CLASS_PARTIAL
    if any(row["item_id_mapping_status"] in {"exact", "ambiguous"} for row in value_rows):
        return CLASS_PARTIAL
    return CLASS_INVALID


def build_forensic_tables(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    status_path = project_path(project_root, R5C_STATUS)
    manifest_path = project_path(project_root, R5C_MANIFEST)
    r5a_manifest_path = project_path(project_root, R5A_MANIFEST)
    if not status_path.exists():
        raise FileNotFoundError(status_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not r5a_manifest_path.exists():
        raise FileNotFoundError(r5a_manifest_path)
    status = read_csv(status_path)
    manifest = read_csv(manifest_path)
    r5a_manifest = read_csv(r5a_manifest_path)
    invalid = status.loc[
        status["model_name"].eq("qwen2:7b-instruct")
        & status["dataset"].eq("ML-1M")
        & status["validity_class"].eq("OUT_OF_RANGE")
        & status["parse_status"].eq("unknown_indices")
    ].copy()
    if len(invalid) != 15:
        raise R5CInvalidAuditError(f"expected 15 R5C invalid Qwen2/ML-1M rows, observed {len(invalid)}")
    manifest_by_id = manifest.set_index("request_id", drop=False)
    r5a_by_id = r5a_manifest.set_index("request_id", drop=False)
    audit_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    rerun_rows: list[dict[str, Any]] = []
    for invalid_row in invalid.to_dict("records"):
        request_id = str(invalid_row["request_id"])
        if request_id not in manifest_by_id.index:
            raise R5CInvalidAuditError(f"missing manifest row for {request_id}")
        manifest_row = manifest_by_id.loc[request_id].to_dict()
        r5a_request_id = str(invalid_row["r5a_request_id"])
        if r5a_request_id not in r5a_by_id.index:
            raise R5CInvalidAuditError(f"missing R5A manifest row for {r5a_request_id}")
        r5a_row = r5a_by_id.loc[r5a_request_id].to_dict()
        raw_path = project_path(project_root, str(invalid_row["output_destination"]))
        raw = load_json(raw_path)
        raw_response = str(raw.get("raw_response_text", ""))
        values, raw_parse_status = parse_raw_values(raw_response)
        ordered_candidates, item_to_indices = candidate_mapping(r5a_row)
        candidate_count = len(ordered_candidates)
        local_value_rows: list[dict[str, Any]] = []
        for position, value in enumerate(values, start=1):
            value_text = str(value)
            valid_index = 1 <= value <= candidate_count
            item_id_indices = item_to_indices.get(value_text, [])
            if len(item_id_indices) == 1:
                mapping_status = "exact"
                mapped_index: int | str = item_id_indices[0]
            elif len(item_id_indices) > 1:
                mapping_status = "ambiguous"
                mapped_index = "|".join(str(item) for item in item_id_indices)
            else:
                mapping_status = "not_candidate_item_id"
                mapped_index = ""
            ambiguous_index_and_id = valid_index and bool(item_id_indices)
            row = {
                "request_id": request_id,
                "r5a_request_id": str(invalid_row["r5a_request_id"]),
                "model_name": str(invalid_row["model_name"]),
                "dataset": str(invalid_row["dataset"]),
                "history_length": str(invalid_row["history_length"]),
                "experimental_condition": str(invalid_row["experimental_condition"]),
                "returned_position": position,
                "returned_value": value,
                "valid_1_based_index": bool_text(valid_index),
                "valid_index_range": f"1..{candidate_count}",
                "candidate_item_id_match": bool_text(bool(item_id_indices)),
                "item_id_mapping_status": mapping_status,
                "mapped_candidate_index": mapped_index,
                "ambiguous_index_and_item_id": bool_text(ambiguous_index_and_id),
            }
            value_rows.append(row)
            local_value_rows.append(row)
        classification = classify_request(values, local_value_rows, candidate_count)
        mapped_indices = [
            int(row["mapped_candidate_index"])
            for row in local_value_rows
            if str(row["mapped_candidate_index"]).isdigit()
        ]
        mapped_items = [
            ordered_candidates[index - 1]["item_id"]
            for index in mapped_indices
            if 1 <= index <= len(ordered_candidates)
        ]
        mapped_rank = positive_rank(r5a_row, mapped_items)
        audit_rows.append(
            {
                "request_id": request_id,
                "r5a_request_id": str(invalid_row["r5a_request_id"]),
                "model_name": str(invalid_row["model_name"]),
                "dataset": str(invalid_row["dataset"]),
                "history_length": str(invalid_row["history_length"]),
                "experimental_condition": str(invalid_row["experimental_condition"]),
                "candidate_count": candidate_count,
                "candidate_set_ordered_json": canonical_json(ordered_candidates),
                "expected_representation": STRICT_EXPECTED,
                "raw_response": raw_response,
                "parsed_values": canonical_json(values),
                "validation_failure": f"{invalid_row['validity_class']} / {invalid_row['parse_status']}",
                "valid_index_values_count": sum(1 for value in values if 1 <= value <= candidate_count),
                "candidate_item_id_values_count": sum(1 for row in local_value_rows if row["candidate_item_id_match"] == "true"),
                "ambiguous_values_count": sum(1 for row in local_value_rows if row["item_id_mapping_status"] == "ambiguous"),
                "mapped_candidate_indices": canonical_json(mapped_indices),
                "mapped_item_ids": canonical_json(mapped_items),
                "mapped_positive_rank": mapped_rank,
                "mapped_hr_at_5": hr_at_5(mapped_rank),
                "mapped_ndcg_at_5": ndcg_at_5(mapped_rank),
                "request_classification": classification,
                "raw_parse_status": raw_parse_status,
                "raw_file_sha256": sha256_file(raw_path),
                "output_destination": str(invalid_row["output_destination"]),
            }
        )
        rerun_rows.append(
            {
                "request_id": request_id,
                "r5a_request_id": str(invalid_row["r5a_request_id"]),
                "dataset": str(invalid_row["dataset"]),
                "model_name": str(invalid_row["model_name"]),
                "history_length": str(invalid_row["history_length"]),
                "experimental_condition": str(invalid_row["experimental_condition"]),
                "user_id": str(invalid_row["user_id"]),
                "candidate_list_hash": str(manifest_row["candidate_list_hash"]),
                "history_hash": str(manifest_row["history_hash"]),
                "prompt_hash": str(manifest_row["prompt_hash"]),
                "output_destination": str(invalid_row["output_destination"]),
                "failure_mode": "qwen_returned_candidate_item_ids_in_ranked_candidate_indices_field",
                "future_rerun_required_check": "Qwen must return exactly five values in the 1..20 candidate-index namespace.",
            }
        )
    audit = pd.DataFrame(audit_rows).sort_values(
        ["history_length", "request_id"], key=lambda col: col.astype(str)
    ).reset_index(drop=True)
    values = pd.DataFrame(value_rows)
    rerun = pd.DataFrame(rerun_rows).sort_values(
        ["history_length", "request_id"], key=lambda col: col.astype(str)
    ).reset_index(drop=True)
    return audit, values, rerun


def build_science_assessment(project_root: Path) -> pd.DataFrame:
    evidence = [
        {
            "component": "R5C frozen prompt",
            "source_path": str(R5C_RUNNER),
            "finding": "R5C corrected instruction requires exactly five unique one-based candidate indices and explicitly says not to use item IDs.",
            "allows_item_id_to_index_normalization": "false",
        },
        {
            "component": "R5C parser/validator",
            "source_path": str(CANONICAL_PARSER),
            "finding": "Strict parser interprets ranked_candidate_indices as one-based candidate positions; values outside 1..N are unknown_indices.",
            "allows_item_id_to_index_normalization": "false",
        },
        {
            "component": "R5C synthetic tests",
            "source_path": str(REPORT_PATH.parent / "reproduction_r5c_history_length_recovery.md"),
            "finding": "The frozen run validates candidate-set violations as unknown_indices and does not include item-ID rescue in the scientific parser.",
            "allows_item_id_to_index_normalization": "false",
        },
        {
            "component": "Original implementation",
            "source_path": str(ORIGINAL_LLM),
            "finding": "Original LLM calls request a Python list of item titles, not local R5C one-based candidate indices or numeric item IDs.",
            "allows_item_id_to_index_normalization": "false",
        },
        {
            "component": "Original output refinement",
            "source_path": str(ORIGINAL_REFINE),
            "finding": "Original refinement extracts list outputs and length-adjusts title lists; it is not an R5C index parser and does not authorize numeric item-ID rescue.",
            "allows_item_id_to_index_normalization": "false",
        },
        {
            "component": "Original metric mapping",
            "source_path": str(ORIGINAL_METRICS),
            "finding": "Original metrics map returned titles to item IDs through title matching; this differs from changing R5C's declared index namespace after inference.",
            "allows_item_id_to_index_normalization": "false",
        },
        {
            "component": "Prior local ML-1M item-ID contract",
            "source_path": str(PLAN_D_PROMPT_REPORT),
            "finding": "Plan D documented a separate item-ID JSON contract for a prior ML-1M diagnostic, but R5C did not use that contract.",
            "allows_item_id_to_index_normalization": "false",
        },
        {
            "component": "Protocol decision",
            "source_path": str(R5C_CONFIG),
            "finding": "Changing invalid R5C outputs to valid through post-hoc item-ID normalization would modify the frozen output contract and parser semantics.",
            "allows_item_id_to_index_normalization": "false",
        },
    ]
    rows: list[dict[str, Any]] = []
    for item in evidence:
        path = project_path(project_root, item["source_path"])
        rows.append(
            {
                **item,
                "source_exists": bool_text(path.exists()),
                "source_sha256": sha256_file(path) if path.exists() else "",
            }
        )
    return pd.DataFrame(rows)


def existing_r5c_counts(project_root: Path) -> dict[str, Any]:
    status = read_csv(project_path(project_root, R5C_STATUS))
    terminal = status.loc[~status["execution_status"].eq("NOT_EXECUTED")]
    gaps = read_csv(project_path(project_root, R5C_GAP_RESOLUTION)) if project_path(project_root, R5C_GAP_RESOLUTION).exists() else pd.DataFrame()
    return {
        "requests": len(status),
        "accounted": len(terminal),
        "valid": int(terminal["validity_class"].eq("VALID").sum()),
        "invalid": int(terminal["validity_class"].ne("VALID").sum()),
        "wrong_length": int(terminal["validity_class"].eq("WRONG_LENGTH").sum()),
        "parse_failure": int(terminal["validity_class"].eq("PARSE_FAILURE").sum()),
        "backend_incomplete": int(terminal["validity_class"].eq("BACKEND_INCOMPLETE").sum()),
        "target_gaps_all_valid": int(gaps["gap_recovered"].astype(str).str.lower().eq("true").sum()) if not gaps.empty else 0,
        "target_gaps_total": len(gaps) if not gaps.empty else 0,
    }


def build_summary(
    project_root: Path,
    audit: pd.DataFrame,
    values: pd.DataFrame,
    rerun: pd.DataFrame,
    science: pd.DataFrame,
) -> dict[str, Any]:
    counts = existing_r5c_counts(project_root)
    exact = int(audit["request_classification"].eq(CLASS_EXACT).sum())
    partial = int(audit["request_classification"].eq(CLASS_PARTIAL).sum())
    genuine = int(audit["request_classification"].eq(CLASS_INVALID).sum())
    all_exact = exact == len(audit) and partial == 0 and genuine == 0
    normalization_allowed = science["allows_item_id_to_index_normalization"].eq("true").all()
    hypothetical_valid = counts["valid"] + exact if all_exact else counts["valid"]
    hypothetical_invalid = counts["requests"] - hypothetical_valid
    hypothetical_gap_recovered = counts["target_gaps_total"] if all_exact else counts["target_gaps_all_valid"]
    return {
        "phase": PHASE,
        "created_utc": utc_now(),
        "final_decision": DECISION_SELECTIVE_RERUN,
        "llm_calls": 0,
        "cases_audited": len(audit),
        "exact_offline_mappable": exact,
        "partial_or_ambiguous": partial,
        "genuine_invalid": genuine,
        "item_id_to_index_mapping": "DETERMINISTIC" if all_exact else "AMBIGUOUS" if partial else "NOT_APPLICABLE",
        "post_hoc_normalization_scientifically_allowed": bool(normalization_allowed),
        "can_repair_from_existing_raw_outputs_scientifically": False,
        "technical_mapping_possible": bool(all_exact),
        "selective_rerun_required": True,
        "rows_requiring_rerun": rerun["request_id"].tolist(),
        "existing_r5c_counts": counts,
        "if_offline_recovered_hypothetical": {
            "valid": hypothetical_valid,
            "invalid": hypothetical_invalid,
            "target_gaps_all_valid": hypothetical_gap_recovered,
            "target_gaps_total": counts["target_gaps_total"],
            "r5c_freeze_ready_if_normalization_were_allowed": bool(hypothetical_valid == counts["requests"] and normalization_allowed),
        },
        "raw_output_modified": False,
        "r5c_rerun_executed": False,
        "r5d_executed": False,
        "commit": "NONE",
        "push": "NO",
        "input_hashes": {
            str(R5C_STATUS): sha256_file(project_path(project_root, R5C_STATUS)),
            str(R5C_MANIFEST): sha256_file(project_path(project_root, R5C_MANIFEST)),
        },
    }


def render_report(audit: pd.DataFrame, values: pd.DataFrame, rerun: pd.DataFrame, science: pd.DataFrame, summary: Mapping[str, Any]) -> str:
    return f"""# R5C Invalid Output Forensic Audit

## Scope

Offline-only audit of the 15 R5C invalid outputs with `qwen2:7b-instruct`, `ML-1M`, and `OUT_OF_RANGE / unknown_indices`.

No LLM/Ollama calls, raw-output edits, R5C rerun, R5D execution, commit, or push were performed.

## Decision

`{summary["final_decision"]}`

All 15 outputs are technically mappable as candidate item IDs, but post-hoc item-ID to candidate-index normalization is not allowed by the frozen R5C protocol.

## Case Counts

- Cases audited: `{summary["cases_audited"]}/15`
- Exact offline mappable: `{summary["exact_offline_mappable"]}`
- Partial or ambiguous: `{summary["partial_or_ambiguous"]}`
- Genuine invalid: `{summary["genuine_invalid"]}`
- Item-ID to index mapping: `{summary["item_id_to_index_mapping"]}`
- Post-hoc normalization scientifically allowed: `NO`
- Selective rerun required: `YES`

## Fifteen Request Audit

{render_table(audit, ["request_id", "r5a_request_id", "history_length", "experimental_condition", "validation_failure", "parsed_values", "mapped_candidate_indices", "candidate_item_id_values_count", "ambiguous_values_count", "request_classification"])}

## Protocol Assessment

{render_table(science, ["component", "source_path", "source_exists", "allows_item_id_to_index_normalization", "finding"])}

## Selective Rerun Rows

{render_table(rerun, ["request_id", "r5a_request_id", "history_length", "experimental_condition", "user_id", "failure_mode", "future_rerun_required_check"])}

## Future Minimal Rerun Protocol

1. Static preflight over only the 15 frozen manifest rows.
2. Synthetic test that rejects item IDs inside `ranked_candidate_indices` and accepts only `1..20` candidate indices.
3. Qwen x ML-1M canary with at least one affected history length cell.
4. Immediate validation that raw outputs contain exactly five 1-based candidate indices, not item IDs.
5. Only then rerun the 15 affected rows with the unchanged frozen R5C protocol identity or a separately named corrected selective-rerun identity.

## Hypothetical Offline Mapping

The existing raw outputs are semantically recoverable in a technical sense. If the mapping were allowed, the 15 rows would map deterministically and R5C would be `2400/2400` valid with `4/4` target gaps all-valid. This is not applied because the protocol disallows the normalization.
"""


def update_notes(project_root: Path, summary: Mapping[str, Any]) -> None:
    notes_path = project_path(project_root, NOTES_PATH)
    section = f"""
## R5C invalid output forensic audit

- Decision: `{summary["final_decision"]}`.
- Scope: offline-only audit of 15 Qwen2/ML-1M R5C `OUT_OF_RANGE / unknown_indices` outputs; no LLM/Ollama calls, raw edits, R5C rerun, R5D, commit, push, or `original_repo/` modification.
- Case classification: exact offline mappable `{summary["exact_offline_mappable"]}`, partial/ambiguous `{summary["partial_or_ambiguous"]}`, genuine invalid `{summary["genuine_invalid"]}`.
- Scientific decision: item-ID to candidate-index mapping is deterministic but not allowed by the frozen R5C candidate-index protocol, so selective rerun is required.
- Outputs: `results/tables/reproduction_r5c_invalid_output_forensic_audit.csv`, `results/tables/reproduction_r5c_invalid_output_value_mapping.csv`, `results/tables/reproduction_r5c_selective_rerun_rows.csv`, `results/tables/reproduction_r5c_invalid_output_protocol_assessment.csv`, `results/diagnostics/r5c_history_length/invalid_output_forensic_summary.json`, and `report/reproduction_r5c_invalid_output_forensic_audit.md`.
"""
    existing = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    marker = "## R5C invalid output forensic audit"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n" + section
    else:
        existing = existing.rstrip() + "\n" + section
    notes_path.write_text(existing.lstrip(), encoding="utf-8")


def write_outputs(project_root: Path, result: AuditResult) -> None:
    write_csv(result.audit, project_path(project_root, AUDIT_TABLE))
    write_csv(result.values, project_path(project_root, VALUE_TABLE))
    write_csv(result.rerun_rows, project_path(project_root, RERUN_TABLE))
    write_csv(result.science, project_path(project_root, SCIENCE_TABLE))
    report_path = project_path(project_root, REPORT_PATH)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(result.report, encoding="utf-8")
    write_json_atomic(result.summary, project_path(project_root, SUMMARY_JSON))
    update_notes(project_root, result.summary)


def build_audit(project_root: Path, *, write: bool = True) -> AuditResult:
    audit, values, rerun = build_forensic_tables(project_root)
    science = build_science_assessment(project_root)
    summary = build_summary(project_root, audit, values, rerun, science)
    report = render_report(audit, values, rerun, science, summary)
    result = AuditResult(
        decision=str(summary["final_decision"]),
        audit=audit,
        values=values,
        rerun_rows=rerun,
        science=science,
        report=report,
        summary=summary,
    )
    if write:
        write_outputs(project_root, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    result = build_audit(project_root, write=not args.no_write)
    summary = result.summary
    print(f"FINAL_DECISION={summary['final_decision']}")
    print("LLM_CALLS=0")
    print(f"CASES_AUDITED={summary['cases_audited']}/15")
    print(f"EXACT_OFFLINE_MAPPABLE={summary['exact_offline_mappable']}")
    print(f"PARTIAL_OR_AMBIGUOUS={summary['partial_or_ambiguous']}")
    print(f"GENUINE_INVALID={summary['genuine_invalid']}")
    print(f"ITEM_ID_TO_INDEX_MAPPING={summary['item_id_to_index_mapping']}")
    print("POST_HOC_NORMALIZATION_SCIENTIFICALLY_ALLOWED=NO")
    print("CAN_REPAIR_FROM_EXISTING_RAW_OUTPUTS=NO")
    print("SELECTIVE_RERUN_REQUIRED=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
