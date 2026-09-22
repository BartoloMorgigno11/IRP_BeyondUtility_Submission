"""Offline forensic diagnostics for R5B CandDif recovery.

This script is deliberately read-only with respect to the frozen R5B artifacts:
it reads the R5A manifest, R5B CSV summaries, and R5B raw JSON outputs, then
writes independent diagnostic tables under results/diagnostics/r5b.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.canonical_llm_contract import (  # noqa: E402
    parse_diagnostic_salvage_response,
    parse_strict_index_response,
)


OUT = ROOT / "results" / "diagnostics" / "r5b"
EXPECTED_MANIFEST_SHA = "290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD"
EXPECTED_HEAD = "98c28c5364c40e571765a3b50d4dab8affe80743"
FAIL_STRICT_POLICY = "FAIL_STRICT_ALL_MANIFEST_REQUESTS_INVALID_ZERO"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize(row.get(key, "")) for key in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def serialize(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps({str(key): item for key, item in value.items()}, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, float):
        return repr(value)
    return "" if value is None else str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def as_bool_text(value: bool) -> str:
    return "YES" if value else "NO"


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value: Any) -> int:
    try:
        if value == "":
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def candidate_records(row: dict[str, str]) -> list[dict[str, Any]]:
    return json.loads(row["candidate_list"])


def candidate_ids(row: dict[str, str]) -> list[str]:
    return [str(item["item_id"]) for item in candidate_records(row)]


def history_records(row: dict[str, str]) -> list[dict[str, Any]]:
    return json.loads(row["history_items"])


def expected_count(row: dict[str, str]) -> int:
    try:
        parsed = json.loads(row.get("expected_output_schema", ""))
    except json.JSONDecodeError:
        return 5
    values = parsed.get("ranked_candidate_indices") if isinstance(parsed, dict) else None
    return len(values) if isinstance(values, list) and values else 5


def positive_rank(row: dict[str, str], indices: list[int]) -> int:
    ids = candidate_ids(row)
    positive = str(row["positive_item_id"])
    mapped = [ids[index - 1] for index in indices if 1 <= index <= len(ids)]
    for rank, item_id in enumerate(mapped, start=1):
        if str(item_id) == positive:
            return rank
    return 0


def hr(rank: int) -> float:
    return 1.0 if 1 <= rank <= 5 else 0.0


def ndcg(rank: int) -> float:
    return 1.0 / math.log2(rank + 1) if 1 <= rank <= 5 else 0.0


def canddif_value(first_acc: float, random_acc: float) -> tuple[float, str]:
    numerator = 1.0 - random_acc
    denominator = 1.0 - first_acc
    if numerator <= 0 and denominator <= 0:
        return 0.0, "BOTH_ERROR_RATES_ZERO_RETURNED_0"
    if numerator <= 0:
        return math.log(1e-12 / max(denominator, 1e-12)), "NUMERATOR_CLAMPED_EPSILON_1E-12"
    if denominator <= 0:
        return math.log(numerator / 1e-12), "DENOMINATOR_CLAMPED_EPSILON_1E-12"
    return math.log(numerator / denominator), "NONE"


def canonical_candidate_set_hash(ids: list[str]) -> str:
    return sha256_text(json.dumps(sorted(ids), ensure_ascii=True, separators=(",", ":")))


def random_order_expected(first_row: dict[str, str], random_row: dict[str, str]) -> bool:
    candidates = candidate_records(first_row)
    seed = str(random_row["candidate_order_seed"])
    dataset = str(random_row["dataset"])
    user_id = str(random_row["user_id"])
    expected = sorted(
        candidates,
        key=lambda item: sha256_text(f"{seed}|{dataset}|{user_id}|{item.get('item_id', '')}|{item.get('title', '')}"),
    )
    if expected and bool(expected[0].get("is_positive")) and len(expected) > 1:
        expected = expected[1:] + expected[:1]
    return [str(item["item_id"]) for item in expected] == candidate_ids(random_row)


def root_cause(row: dict[str, Any]) -> str:
    status = row["parse_status"]
    if status == "parsed_valid":
        return "VALID"
    if status == "parse_failed":
        if row.get("ollama_done") is False:
            return "RUNNER_ACCEPTED_INCOMPLETE_OLLAMA_DONE_FALSE"
        return "PARSE_FAILURE"
    if status != "invalid_length":
        return "OTHER_STRUCTURAL_INVALIDITY"
    length = int(row["parsed_length"])
    expected = int(row["expected_length"])
    if length < expected:
        return "MODEL_PRODUCED_UNDER_K"
    if length == 10:
        return "MODEL_PRODUCED_TOP10_ORIGINAL_CONTRACT"
    if length == 20:
        return "MODEL_PRODUCED_FULL_CANDIDATE_SET"
    if length > 20:
        return "MODEL_PRODUCED_MORE_THAN_CANDIDATE_SET"
    if 11 <= length <= 19:
        return "MODEL_PRODUCED_NEAR_FULL_CANDIDATE_SET"
    if 6 <= length <= 9:
        return "MODEL_PRODUCED_OVER_K_SHORT_LIST"
    return "UNKNOWN_WRONG_LENGTH"


def format_family(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return "empty"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        if text.startswith("{"):
            return "truncated_or_invalid_json_object"
        if text.startswith("["):
            return "json_array_or_invalid_array"
        return "non_json_text"
    if isinstance(parsed, dict):
        if isinstance(parsed.get("ranked_candidate_indices"), list):
            return "json_object_ranked_candidate_indices"
        return "json_object_other_schema"
    if isinstance(parsed, list):
        return "json_array"
    return type(parsed).__name__


def duplicate_count(values: list[Any]) -> int:
    return max(0, len(values) - len(set(map(str, values))))


def unknown_count(values: list[Any], candidate_count: int) -> int:
    return sum(1 for value in values if not isinstance(value, int) or isinstance(value, bool) or value < 1 or value > candidate_count)


@dataclass(frozen=True)
class Data:
    manifest: list[dict[str, str]]
    canddif: list[dict[str, str]]
    status: list[dict[str, str]]
    metrics: list[dict[str, str]]
    canddif_results: list[dict[str, str]]
    payload_by_request: dict[str, dict[str, Any]]


def load_data() -> Data:
    manifest = read_csv(ROOT / "results" / "tables" / "reproduction_r5a_recovery_request_manifest.csv")
    canddif = [row for row in manifest if row["recovery_family"] == "CANDDIF_RECOVERY"]
    status = read_csv(ROOT / "results" / "tables" / "reproduction_r5b_request_execution_status.csv")
    metrics = read_csv(ROOT / "results" / "tables" / "reproduction_r5b_metrics_by_condition.csv")
    canddif_results = read_csv(ROOT / "results" / "tables" / "reproduction_r5b_canddif_results.csv")
    payload_by_request = {}
    for row in canddif:
        path = ROOT / row["output_destination"]
        if path.exists():
            payload_by_request[row["request_id"]] = json.loads(path.read_text(encoding="utf-8"))
    return Data(manifest, canddif, status, metrics, canddif_results, payload_by_request)


def build_request_level(data: Data) -> list[dict[str, Any]]:
    status_by_request = {row["request_id"]: row for row in data.status}
    rows: list[dict[str, Any]] = []
    for manifest_row in data.canddif:
        request_id = manifest_row["request_id"]
        status = status_by_request[request_id]
        payload = data.payload_by_request.get(request_id, {})
        raw = str(payload.get("raw_response_text", ""))
        indices = payload.get("ranked_candidate_indices", [])
        if not isinstance(indices, list):
            indices = []
        expected = int(payload.get("expected_output_count", expected_count(manifest_row)) or expected_count(manifest_row))
        parsed_length = int(payload.get("output_length", 0) or 0)
        first5 = indices[:5]
        top5_salvage_valid = (
            len(first5) == 5
            and all(isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 20 for value in first5)
            and len(set(first5)) == 5
        )
        top5_rank = positive_rank(manifest_row, first5) if top5_salvage_valid else 0
        ollama = payload.get("ollama_raw_response", {})
        row = {
            "request_id": request_id,
            "dataset": manifest_row["dataset"],
            "model_name": manifest_row["model_name"],
            "condition": manifest_row["experimental_condition"],
            "user_id": manifest_row["user_id"],
            "candidate_count": safe_int(manifest_row["candidate_count"]),
            "positive_candidate_position": safe_int(manifest_row["positive_candidate_position"]),
            "expected_length": expected,
            "parsed_length": parsed_length,
            "length_difference": parsed_length - expected,
            "parse_valid": status["parse_valid"],
            "parse_status": status["parse_status"],
            "parse_failure_category": status["parse_failure_category"],
            "saved_hr_at_5": safe_float(status["hr_at_5"]),
            "saved_ndcg_at_5": safe_float(status["ndcg_at_5"]),
            "saved_positive_rank": safe_int(status["positive_rank"]),
            "raw_response_text": raw,
            "raw_response_preview": raw.replace("\n", "\\n")[:240],
            "format_family": format_family(raw),
            "indices": indices,
            "first5_indices": first5,
            "duplicate_count": safe_int(payload.get("duplicate_count", duplicate_count(indices))),
            "unknown_index_count": safe_int(payload.get("unknown_item_count", unknown_count(indices, 20))),
            "ollama_done": ollama.get("done"),
            "ollama_done_reason": ollama.get("done_reason", ""),
            "ollama_eval_count": ollama.get("eval_count", ""),
            "ollama_prompt_eval_count": ollama.get("prompt_eval_count", ""),
            "ollama_total_duration": ollama.get("total_duration", ""),
            "top5_salvage_valid": top5_salvage_valid,
            "top5_salvage_positive_rank": top5_rank,
            "top5_salvage_hr_at_5": hr(top5_rank) if top5_salvage_valid else 0.0,
            "top5_salvage_ndcg_at_5": ndcg(top5_rank) if top5_salvage_valid else 0.0,
            "prompt_has_exactly_5": "exactly 5" in manifest_row["prompt_render"].lower(),
            "prompt_has_top_5": "top 5" in manifest_row["prompt_render"].lower() or "top-5" in manifest_row["prompt_render"].lower(),
            "prompt_has_json_key": "ranked_candidate_indices" in manifest_row["prompt_render"],
            "prompt_has_unique": "unique" in manifest_row["prompt_render"].lower(),
            "prompt_forbids_rationale": "do not explain" in manifest_row["prompt_render"].lower()
            or "no rationale" in manifest_row["prompt_render"].lower(),
        }
        row["root_cause"] = root_cause(row)
        rows.append(row)
    return rows


def grouped_length_distribution(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    filtered = [row for row in rows if row["parse_status"] == "invalid_length"]
    group_counts = Counter(tuple(row[field] for field in group_fields) for row in filtered)
    length_counts = Counter((tuple(row[field] for field in group_fields), row["parsed_length"]) for row in filtered)
    out_rows: list[dict[str, Any]] = []
    for (group_key, parsed_length), count in sorted(length_counts.items(), key=lambda item: (item[0][0], item[0][1])):
        group_total = group_counts[group_key]
        result = {field: group_key[index] for index, field in enumerate(group_fields)}
        result.update(
            {
                "parsed_length": parsed_length,
                "count": count,
                "percentage_of_group_wrong_length": count / group_total if group_total else 0.0,
                "wrong_length_group_total": group_total,
            }
        )
        out_rows.append(result)
    return out_rows


def validity_breakdowns(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    groups = sorted(set(tuple(row[field] for field in group_fields) for row in rows))
    out_rows: list[dict[str, Any]] = []
    for group_key in groups:
        group = [row for row in rows if tuple(row[field] for field in group_fields) == group_key]
        counts = Counter(row["parse_status"] for row in group)
        result = {field: group_key[index] for index, field in enumerate(group_fields)}
        total = len(group)
        result.update(
            {
                "requests": total,
                "valid": counts["parsed_valid"],
                "invalid": total - counts["parsed_valid"],
                "wrong_length": counts["invalid_length"],
                "parse_failed": counts["parse_failed"],
                "valid_rate": counts["parsed_valid"] / total if total else 0.0,
                "invalid_rate": (total - counts["parsed_valid"]) / total if total else 0.0,
            }
        )
        out_rows.append(result)
    return out_rows


def prompt_audit(data: Data) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in data.canddif:
        key = (row["dataset"], row["model_name"], row["experimental_condition"], row["prompt_hash"])
        if key in seen:
            continue
        seen.add(key)
        prompt = row["prompt_render"]
        rows.append(
            {
                "dataset": row["dataset"],
                "model_name": row["model_name"],
                "condition": row["experimental_condition"],
                "prompt_template_id": row["prompt_template_id"],
                "prompt_hash": row["prompt_hash"],
                "candidate_count": row["candidate_count"],
                "expected_output_schema": row["expected_output_schema"],
                "contains_return_json": "Return JSON" in prompt,
                "contains_ranked_candidate_indices": "ranked_candidate_indices" in prompt,
                "contains_unique": "unique" in prompt.lower(),
                "contains_exactly_5": "exactly 5" in prompt.lower(),
                "contains_top_5": "top 5" in prompt.lower() or "top-5" in prompt.lower(),
                "contains_no_explanation": "do not explain" in prompt.lower() or "no explanation" in prompt.lower(),
                "contains_no_rationale": "no rationale" in prompt.lower(),
                "contains_do_not_use_item_ids": "do not use item ids" in prompt.lower(),
                "mentions_20_candidates": "20" in prompt,
                "prompt_line_count": len(prompt.splitlines()),
                "prompt_characters": len(prompt),
                "first_220_chars": prompt.replace("\n", "\\n")[:220],
            }
        )
    return rows


def ollama_config_audit(data: Data) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = sorted(set((row["model_name"], row["temperature"], row["seed"], row["max_new_tokens"]) for row in data.canddif))
    for model, temperature, seed, max_new_tokens in grouped:
        rows.append(
            {
                "model": model,
                "api": "/api/generate",
                "temperature": temperature,
                "seed": seed,
                "num_predict": max_new_tokens,
                "stream": "false",
                "format": "json",
                "format_is_schema": "NO",
                "system_prompt_sent": "NO",
                "raw_mode": "not configured",
                "top_p": "not configured",
                "top_k": "not configured",
                "stop_sequences": "not configured",
                "context_size": "not configured in request",
                "template": "ollama model default template",
                "timeout_seconds": "120",
                "audit_note": "R5B uses JSON mode but not a JSON schema; the request does not encode expected list length.",
            }
        )
    return rows


def metric_reconstruction(rows: list[dict[str, Any]], reported_metrics: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reported_by_group = {
        (row["dataset"], row["model_name"], row["experimental_condition"]): row for row in reported_metrics
    }
    groups = sorted(set((row["dataset"], row["model_name"], row["condition"]) for row in rows))
    denom_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    for group_key in groups:
        group = [row for row in rows if (row["dataset"], row["model_name"], row["condition"]) == group_key]
        requests = len(group)
        valid_group = [row for row in group if row["parse_valid"] == "true"]
        hits = sum(safe_float(row["saved_hr_at_5"]) for row in group)
        ndcg_sum = sum(safe_float(row["saved_ndcg_at_5"]) for row in group)
        hr_e2e = hits / requests if requests else 0.0
        ndcg_e2e = ndcg_sum / requests if requests else 0.0
        hr_cond = sum(safe_float(row["saved_hr_at_5"]) for row in valid_group) / len(valid_group) if valid_group else 0.0
        ndcg_cond = sum(safe_float(row["saved_ndcg_at_5"]) for row in valid_group) / len(valid_group) if valid_group else 0.0
        reported = reported_by_group[group_key]
        base = {
            "dataset": group_key[0],
            "model_name": group_key[1],
            "condition": group_key[2],
            "requests": requests,
            "valid_outputs": len(valid_group),
            "invalid_outputs": requests - len(valid_group),
            "hits": hits,
            "hr_numerator": hits,
            "hr_denominator": requests,
            "ndcg_sum": ndcg_sum,
            "ndcg_denominator": requests,
            "recomputed_hr": hr_e2e,
            "reported_hr": safe_float(reported["hr_at_5"]),
            "hr_delta": hr_e2e - safe_float(reported["hr_at_5"]),
            "recomputed_ndcg": ndcg_e2e,
            "reported_ndcg": safe_float(reported["ndcg_at_5"]),
            "ndcg_delta": ndcg_e2e - safe_float(reported["ndcg_at_5"]),
            "denominator_policy": FAIL_STRICT_POLICY,
        }
        denom_rows.append(base)
        diag_rows.append(
            {
                **{key: base[key] for key in ["dataset", "model_name", "condition", "requests", "valid_outputs", "invalid_outputs"]},
                "hr_end_to_end": hr_e2e,
                "ndcg_end_to_end": ndcg_e2e,
                "hr_conditional_valid": hr_cond,
                "ndcg_conditional_valid": ndcg_cond,
                "conditional_valid_warning": "N_VALID_SMALL_DO_NOT_INTERPRET_AS_ROBUST" if len(valid_group) < 30 else "",
            }
        )
    return denom_rows, diag_rows


def canddif_reconstruction(
    denom_rows: list[dict[str, Any]],
    diagnostic_metrics: list[dict[str, Any]],
    reported_canddif: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_group = {(row["dataset"], row["model_name"], row["condition"]): row for row in denom_rows}
    diag_by_group = {(row["dataset"], row["model_name"], row["condition"]): row for row in diagnostic_metrics}
    reported_by_group = {(row["dataset"], row["model_name"]): row for row in reported_canddif}
    out_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    for dataset, model in sorted(set((row["dataset"], row["model_name"]) for row in denom_rows)):
        first = by_group[(dataset, model, "CANDDIF_FIRST")]
        random = by_group[(dataset, model, "CANDDIF_RANDOM")]
        reported = reported_by_group[(dataset, model)]
        c_hr, s_hr = canddif_value(first["recomputed_hr"], random["recomputed_hr"])
        c_ndcg, s_ndcg = canddif_value(first["recomputed_ndcg"], random["recomputed_ndcg"])
        out_rows.append(
            {
                "dataset": dataset,
                "model_name": model,
                "reported_CandDif_HR": safe_float(reported["CandDif_HR"]),
                "recomputed_CandDif_HR": c_hr,
                "delta_CandDif_HR": c_hr - safe_float(reported["CandDif_HR"]),
                "reported_CandDif_NDCG": safe_float(reported["CandDif_NDCG"]),
                "recomputed_CandDif_NDCG": c_ndcg,
                "delta_CandDif_NDCG": c_ndcg - safe_float(reported["CandDif_NDCG"]),
                "formula": "log((1-Acc_random)/(1-Acc_first))",
                "hr_safety": s_hr,
                "ndcg_safety": s_ndcg,
            }
        )
        first_diag = diag_by_group[(dataset, model, "CANDDIF_FIRST")]
        random_diag = diag_by_group[(dataset, model, "CANDDIF_RANDOM")]
        e2e_hr, _ = canddif_value(first_diag["hr_end_to_end"], random_diag["hr_end_to_end"])
        e2e_ndcg, _ = canddif_value(first_diag["ndcg_end_to_end"], random_diag["ndcg_end_to_end"])
        cond_hr, _ = canddif_value(first_diag["hr_conditional_valid"], random_diag["hr_conditional_valid"])
        cond_ndcg, _ = canddif_value(first_diag["ndcg_conditional_valid"], random_diag["ndcg_conditional_valid"])
        diag_rows.append(
            {
                "dataset": dataset,
                "model_name": model,
                "first_valid": first_diag["valid_outputs"],
                "random_valid": random_diag["valid_outputs"],
                "CandDif_HR_end_to_end": e2e_hr,
                "CandDif_NDCG_end_to_end": e2e_ndcg,
                "CandDif_HR_conditional_valid": cond_hr,
                "CandDif_NDCG_conditional_valid": cond_ndcg,
                "conditional_valid_warning": "USES_ONLY_VALID_OUTPUTS_N_CAN_BE_TINY",
            }
        )
    return out_rows, diag_rows


def independent_pair_integrity(data: Data) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in data.canddif:
        grouped.setdefault((row["dataset"], row["model_name"], row["user_id"]), []).append(row)
    for index, (key, group) in enumerate(sorted(grouped.items()), start=1):
        first_rows = [row for row in group if row["experimental_condition"] == "CANDDIF_FIRST"]
        random_rows = [row for row in group if row["experimental_condition"] == "CANDDIF_RANDOM"]
        first = first_rows[0] if first_rows else {}
        random = random_rows[0] if random_rows else {}
        first_ids = candidate_ids(first) if first else []
        random_ids = candidate_ids(random) if random else []
        same_set = sorted(first_ids) == sorted(random_ids)
        row = {
            "pair_id": f"R5B-FORENSIC-PAIR-{index:04d}",
            "dataset": key[0],
            "model_name": key[1],
            "user_id": key[2],
            "first_request_id": first.get("request_id", ""),
            "random_request_id": random.get("request_id", ""),
            "row_count": len(group),
            "has_exact_two_rows": len(group) == 2,
            "same_dataset": first.get("dataset") == random.get("dataset") == key[0],
            "same_user": first.get("user_id") == random.get("user_id") == key[2],
            "same_history_hash": first.get("history_hash") == random.get("history_hash"),
            "same_positive": first.get("positive_item_id") == random.get("positive_item_id"),
            "same_candidate_set_hash": canonical_candidate_set_hash(first_ids) == canonical_candidate_set_hash(random_ids),
            "same_candidate_set_as_set": same_set,
            "different_order": first_ids != random_ids,
            "first_positive_position": first.get("positive_candidate_position", ""),
            "random_positive_position": random.get("positive_candidate_position", ""),
            "first_positive_first": first.get("positive_candidate_position") == "1",
            "random_positive_not_first": random.get("positive_candidate_position") != "1",
            "random_order_matches_seed": random_order_expected(first, random) if first and random else False,
            "first_candidate_hash": canonical_candidate_set_hash(first_ids),
            "random_candidate_hash": canonical_candidate_set_hash(random_ids),
        }
        row["pair_valid_independent"] = all(
            [
                row["has_exact_two_rows"],
                row["same_dataset"],
                row["same_user"],
                row["same_history_hash"],
                row["same_positive"],
                row["same_candidate_set_hash"],
                row["same_candidate_set_as_set"],
                row["different_order"],
                row["first_positive_first"],
                row["random_positive_not_first"],
                row["random_order_matches_seed"],
            ]
        )
        rows.append(row)
    summary = [
        {
            "pairs": len(rows),
            "valid_pairs": sum(row["pair_valid_independent"] for row in rows),
            "invalid_pairs": sum(not row["pair_valid_independent"] for row in rows),
            "same_candidate_set_failures": sum(not row["same_candidate_set_hash"] for row in rows),
            "random_order_failures": sum(not row["random_order_matches_seed"] for row in rows),
            "duplicate_pair_keys": len(data.canddif) - len(rows) * 2,
        }
    ]
    return rows, summary


def raw_completeness(data: Data) -> list[dict[str, Any]]:
    expected_ids = [row["request_id"] for row in data.canddif]
    raw_ids = []
    corrupted = 0
    empty = 0
    for path in sorted((ROOT / "results" / "recovery" / "r5").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_ids.append(str(payload.get("request_id", "")))
            if not str(payload.get("raw_response_text", "")):
                empty += 1
        except Exception:
            corrupted += 1
    expected_counter = Counter(expected_ids)
    raw_counter = Counter(raw_ids)
    missing = sorted([request_id for request_id in expected_counter if raw_counter[request_id] == 0])
    unexpected = sorted([request_id for request_id in raw_counter if request_id not in expected_counter and request_id])
    duplicates = sorted([request_id for request_id, count in raw_counter.items() if count > 1 and request_id])
    return [
        {
            "unique_expected_ids": len(expected_counter),
            "unique_raw_output_ids": len(raw_counter),
            "raw_json_files": len(raw_ids),
            "missing_count": len(missing),
            "duplicate_count": len(duplicates),
            "unexpected_count": len(unexpected),
            "corrupted_count": corrupted,
            "empty_raw_response_count": empty,
            "accounting_pass": len(expected_counter) == 1200
            and len(raw_counter) == 1200
            and len(raw_ids) == 1200
            and not missing
            and not duplicates
            and not unexpected
            and corrupted == 0,
            "missing_ids": missing[:20],
            "duplicate_ids": duplicates[:20],
            "unexpected_ids": unexpected[:20],
        }
    ]


def smoke_reuse(data: Data) -> list[dict[str, Any]]:
    selected_ids: set[str] = set()
    by_model: dict[str, list[dict[str, str]]] = {}
    for row in data.canddif:
        by_model.setdefault(row["model_name"], []).append(row)
    for model in sorted(by_model):
        first_group_key = sorted(set((row["dataset"], row["user_id"]) for row in by_model[model]))[0]
        pair = [
            row
            for row in by_model[model]
            if (row["dataset"], row["user_id"]) == first_group_key
        ]
        selected_ids.update(row["request_id"] for row in pair)
    rows: list[dict[str, Any]] = []
    for row in data.canddif:
        if row["request_id"] not in selected_ids:
            continue
        payload = data.payload_by_request.get(row["request_id"], {})
        rows.append(
            {
                "request_id": row["request_id"],
                "dataset": row["dataset"],
                "model_name": row["model_name"],
                "condition": row["experimental_condition"],
                "same_prompt_hash": payload.get("prompt_hash") == row["prompt_hash"],
                "same_model_identifier": payload.get("model_name") == row["model_name"],
                "same_manifest_row": payload.get("request_id") == row["request_id"],
                "same_params": str(payload.get("temperature")) == row["temperature"]
                and str(payload.get("seed")) == row["seed"]
                and str(payload.get("max_new_tokens")) == row["max_new_tokens"],
                "result_file": row["output_destination"],
                "parse_status": payload.get("parse_status", ""),
            }
        )
    return rows


def select_forensic_samples(rows: list[dict[str, Any]], data: Data) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_ids: list[str] = []
    for dataset in ["Beauty", "Sports", "LastFM"]:
        for model in ["mistral:7b-instruct", "qwen2:7b-instruct"]:
            candidates = [
                row for row in rows
                if row["dataset"] == dataset
                and row["model_name"] == model
                and row["parse_valid"] != "true"
            ]
            selected_ids.extend(row["request_id"] for row in candidates[:5])
    valid_controls = [row for row in rows if row["parse_valid"] == "true"]
    selected_ids.extend(row["request_id"] for row in valid_controls[:10])
    selected_ids = list(dict.fromkeys(selected_ids))
    manifest_by_request = {row["request_id"]: row for row in data.canddif}
    sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for request_id in selected_ids:
        row = next(item for item in rows if item["request_id"] == request_id)
        manifest_row = manifest_by_request[request_id]
        payload = data.payload_by_request[request_id]
        candidates = candidate_records(manifest_row)
        normalized_ranking = [
            candidates[index - 1]["item_id"]
            for index in row["indices"]
            if isinstance(index, int) and not isinstance(index, bool) and 1 <= index <= len(candidates)
        ]
        classification = "MODEL_NONCOMPLIANCE"
        if row["parse_status"] == "parse_failed" and row["ollama_done"] is False:
            classification = "TRUNCATION"
        elif row["parse_status"] == "parse_failed":
            classification = "PARSER_FAILURE"
        elif row["parse_status"] == "invalid_length":
            classification = "PROMPT_AMBIGUITY" if row["parsed_length"] >= 5 else "MODEL_NONCOMPLIANCE"
        sample = {
            "request_id": request_id,
            "dataset": row["dataset"],
            "model_name": row["model_name"],
            "condition": row["condition"],
            "candidate_set": candidates,
            "expected_k": row["expected_length"],
            "raw_response": payload.get("raw_response_text", ""),
            "parser_extraction": row["indices"],
            "normalized_ranking": normalized_ranking,
            "parsed_length": row["parsed_length"],
            "validator_result": row["parse_valid"],
            "invalid_reason": row["parse_status"],
            "failure_classification": classification,
        }
        sample_rows.append(sample)
        summary_rows.append(
            {
                "request_id": request_id,
                "dataset": row["dataset"],
                "model_name": row["model_name"],
                "condition": row["condition"],
                "expected_k": row["expected_length"],
                "parsed_length": row["parsed_length"],
                "parse_valid": row["parse_valid"],
                "parse_status": row["parse_status"],
                "failure_classification": classification,
                "raw_response_preview": row["raw_response_preview"],
            }
        )
    return summary_rows, sample_rows


def qwen_diagnostics(rows: list[dict[str, Any]], data: Data) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    qwen_rows = [row for row in rows if row["model_name"] == "qwen2:7b-instruct"]
    strata: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    for dataset in ["Beauty", "Sports", "LastFM"]:
        for condition in ["CANDDIF_FIRST", "CANDDIF_RANDOM"]:
            group = [row for row in qwen_rows if row["dataset"] == dataset and row["condition"] == condition]
            selected_ids.extend(row["request_id"] for row in group[:9])
    selected_ids = selected_ids[:54]
    manifest_by_request = {row["request_id"]: row for row in data.canddif}
    for request_id in selected_ids:
        row = next(item for item in rows if item["request_id"] == request_id)
        payload = data.payload_by_request[request_id]
        strata.append(
            {
                "request_id": request_id,
                "dataset": row["dataset"],
                "condition": row["condition"],
                "parse_status": row["parse_status"],
                "expected_length": row["expected_length"],
                "parsed_length": row["parsed_length"],
                "format_family": row["format_family"],
                "has_numbering": False,
                "has_rationale": False,
                "is_json": row["format_family"].startswith("json_object"),
                "candidate_set_complete": row["parsed_length"] == safe_int(manifest_by_request[request_id]["candidate_count"]),
                "near_complete": 11 <= row["parsed_length"] <= 20,
                "done_reason": row["ollama_done_reason"],
                "raw_response": payload.get("raw_response_text", ""),
            }
        )
    summary: list[dict[str, Any]] = []
    for key in sorted(set((row["dataset"], row["condition"]) for row in qwen_rows)):
        group = [row for row in qwen_rows if (row["dataset"], row["condition"]) == key]
        total = len(group)
        summary.append(
            {
                "dataset": key[0],
                "condition": key[1],
                "requests": total,
                "valid": sum(row["parse_valid"] == "true" for row in group),
                "invalid": sum(row["parse_valid"] != "true" for row in group),
                "full_20": sum(row["parsed_length"] == 20 for row in group),
                "near_full_11_20": sum(11 <= row["parsed_length"] <= 20 for row in group),
                "under_k": sum(row["parse_status"] == "invalid_length" and row["parsed_length"] < row["expected_length"] for row in group),
                "top10": sum(row["parsed_length"] == 10 for row in group),
                "over_candidate_count": sum(row["parsed_length"] > 20 for row in group),
                "dominant_pattern": Counter(root_cause(row) for row in group).most_common(1)[0][0],
            }
        )
    return summary, strata


def parser_synthetic_tests() -> list[dict[str, Any]]:
    candidate_items = [str(index) for index in range(1, 21)]
    cases = [
        ("perfect_json", '{"ranked_candidate_indices":[1,2,3,4,5]}', True, [1, 2, 3, 4, 5]),
        ("numbered_lines", "1. 342\n2. 177\n3. 901", False, []),
        ("json_array", "[1,2,3,4,5]", False, []),
        ("csv_indices", "1,2,3,4,5", False, []),
        ("text_plus_json", 'Here is my ranking: {"ranked_candidate_indices":[1,2,3,4,5]}', False, []),
        ("json_string_indices", '{"ranked_candidate_indices":["1","2","3","4","5"]}', False, []),
        ("wrong_key_ranking", '{"ranking":[1,2,3,4,5]}', False, []),
        ("multiline_indices", "1\n2\n3\n4\n5", False, []),
        ("k_minus_1", '{"ranked_candidate_indices":[1,2,3,4]}', False, [1, 2, 3, 4]),
        ("k_plus_1", '{"ranked_candidate_indices":[1,2,3,4,5,6]}', False, [1, 2, 3, 4, 5, 6]),
        ("two_k_due_numbering", '{"ranked_candidate_indices":[1,342,2,177,3,901,4,111,5,222]}', False, [1, 342, 2, 177, 3, 901, 4, 111, 5, 222]),
        ("duplicate", '{"ranked_candidate_indices":[1,1,2,3,4]}', False, [1, 1, 2, 3, 4]),
        ("unknown_index", '{"ranked_candidate_indices":[1,2,3,4,21]}', False, [1, 2, 3, 4, 21]),
        ("trailing_text", '{"ranked_candidate_indices":[1,2,3,4,5]} thanks', False, []),
        ("leading_text", 'ranking {"ranked_candidate_indices":[1,2,3,4,5]}', False, []),
        ("code_fence", '```json\n{"ranked_candidate_indices":[1,2,3,4,5]}\n```', False, []),
        ("empty", "", False, []),
        ("null_value", '{"ranked_candidate_indices":null}', False, []),
        ("rationale_json_extra_key", '{"reason":"because 1 and 2 are relevant","ranked_candidate_indices":[1,2,3,4,5]}', True, [1, 2, 3, 4, 5]),
        ("boolean_index", '{"ranked_candidate_indices":[true,2,3,4,5]}', False, []),
        ("float_index", '{"ranked_candidate_indices":[1.0,2,3,4,5]}', False, []),
        ("negative_index", '{"ranked_candidate_indices":[-1,2,3,4,5]}', False, [-1, 2, 3, 4, 5]),
    ]
    rows: list[dict[str, Any]] = []
    for case_id, text, expected_valid, expected_indices in cases:
        strict = parse_strict_index_response(text, candidate_items=candidate_items, expected_count=5)
        salvage = parse_diagnostic_salvage_response(text, candidate_items=candidate_items, expected_count=5)
        rows.append(
            {
                "case_id": case_id,
                "input": text.replace("\n", "\\n"),
                "expected_contract_valid": expected_valid,
                "expected_indices": expected_indices,
                "actual_valid": strict.parse_valid,
                "actual_status": strict.status,
                "actual_indices": list(strict.output_indices),
                "actual_length": strict.output_length,
                "strict_pass": strict.parse_valid == expected_valid
                and (not expected_indices or list(strict.output_indices) == expected_indices),
                "diagnostic_salvage_valid": salvage.parse_valid,
                "diagnostic_salvage_status": salvage.status,
                "diagnostic_salvage_indices": list(salvage.output_indices),
                "transformations": list(salvage.transformations),
            }
        )
    return rows


def manual_pair_reconstruction(rows: list[dict[str, Any]], data: Data) -> list[dict[str, Any]]:
    manifest_by_request = {row["request_id"]: row for row in data.canddif}
    row_by_request = {row["request_id"]: row for row in rows}
    pair_keys = sorted(set((row["dataset"], row["model_name"], row["user_id"]) for row in data.canddif))
    selected: list[tuple[str, str, str]] = []
    for dataset in ["Beauty", "Sports", "LastFM"]:
        for model in ["mistral:7b-instruct", "qwen2:7b-instruct"]:
            selected.extend([key for key in pair_keys if key[0] == dataset and key[1] == model][:4])
    selected = selected[:20]
    out_rows: list[dict[str, Any]] = []
    for index, key in enumerate(selected, start=1):
        pair_manifest = [row for row in data.canddif if (row["dataset"], row["model_name"], row["user_id"]) == key]
        for condition in ["CANDDIF_FIRST", "CANDDIF_RANDOM"]:
            manifest_row = next(row for row in pair_manifest if row["experimental_condition"] == condition)
            forensic = row_by_request[manifest_row["request_id"]]
            rank = positive_rank(manifest_row, forensic["indices"]) if forensic["parse_valid"] == "true" else 0
            out_rows.append(
                {
                    "manual_pair_id": f"R5B-MANUAL-PAIR-{index:02d}",
                    "request_id": manifest_row["request_id"],
                    "dataset": key[0],
                    "model_name": key[1],
                    "user_id": key[2],
                    "condition": condition,
                    "ground_truth_item_id": manifest_row["positive_item_id"],
                    "candidate_set_hash": canonical_candidate_set_hash(candidate_ids(manifest_row)),
                    "expected_k": forensic["expected_length"],
                    "raw_response_preview": forensic["raw_response_preview"],
                    "manual_parser_indices": forensic["indices"],
                    "manual_parse_valid": forensic["parse_valid"],
                    "manual_positive_rank": rank,
                    "manual_hr_at_5": hr(rank) if forensic["parse_valid"] == "true" else 0.0,
                    "saved_hr_at_5": forensic["saved_hr_at_5"],
                    "hr_delta": (hr(rank) if forensic["parse_valid"] == "true" else 0.0) - forensic["saved_hr_at_5"],
                    "manual_ndcg_at_5": ndcg(rank) if forensic["parse_valid"] == "true" else 0.0,
                    "saved_ndcg_at_5": forensic["saved_ndcg_at_5"],
                    "ndcg_delta": (ndcg(rank) if forensic["parse_valid"] == "true" else 0.0) - forensic["saved_ndcg_at_5"],
                }
            )
    return out_rows


def original_protocol_comparison() -> list[dict[str, Any]]:
    return [
        {
            "component": "Prompt",
            "original_or_paper": "System message asks for 10 items from candidate list as a Python list of item titles; templates also say top 10.",
            "r5b": "Single Ollama generate prompt asks JSON key ranked_candidate_indices with unique 1-based candidate indices, but does not say exactly 5.",
            "equivalent": "NO",
            "impact": "R5B is adapted and its prompt is ambiguous about output length.",
        },
        {
            "component": "Parser",
            "original_or_paper": "Regex extracts bracketed list; parses title strings; length >10 is truncated and length <10 is padded with N/A after being counted invalid.",
            "r5b": "Strict JSON object parser; integer indices only; any length other than 5 invalid; no truncation or padding.",
            "equivalent": "NO",
            "impact": "R5B preserves invalids as failures, but is not original parser behavior.",
        },
        {
            "component": "Validator",
            "original_or_paper": "Title-to-item mapping can map unknown titles to 0; hallucination counted downstream.",
            "r5b": "Index range, duplicate, exact length, and candidate mapping checked before utility scoring.",
            "equivalent": "NO",
            "impact": "R5B uses stricter structural validation.",
        },
        {
            "component": "Invalid handling",
            "original_or_paper": "During API collection, invalid outputs are retried; during metrics, malformed length may be truncated or padded if parsing succeeds.",
            "r5b": "No scientific retry; invalid outputs retained and scored HR/NDCG zero.",
            "equivalent": "NO",
            "impact": "R5B end-to-end reliability is measurable but not paper-exact.",
        },
        {
            "component": "HR denominator",
            "original_or_paper": "Mean over sampled users after read_llm_results succeeds; invalid collection may have been retried.",
            "r5b": "All 100 manifest requests per dataset/model/condition in denominator.",
            "equivalent": "PARTIAL",
            "impact": "Mathematically clear, but harsher than an output-retry paper flow.",
        },
        {
            "component": "NDCG denominator",
            "original_or_paper": "Mean over sampled users after parser/read success.",
            "r5b": "All manifest requests per condition in denominator with invalid=0.",
            "equivalent": "PARTIAL",
            "impact": "No denominator bug found, but protocol differs.",
        },
        {
            "component": "CandDif",
            "original_or_paper": "-log(1-Acc(first)) - (-log(1-Acc(random))) = log((1-Acc_random)/(1-Acc_first)).",
            "r5b": "Same formula applied to HR@5 and NDCG@5 aggregates.",
            "equivalent": "YES_FOR_FORMULA",
            "impact": "Formula reconstructs exactly; input accuracies are protocol-limited.",
        },
    ]


def build_root_cause_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    invalid = [row for row in rows if row["parse_valid"] != "true"]
    total = len(rows)
    counts = Counter(row["root_cause"] for row in invalid)
    label_map = {
        "MODEL_PRODUCED_FULL_CANDIDATE_SET": "Model produced full 20-candidate ranking",
        "MODEL_PRODUCED_NEAR_FULL_CANDIDATE_SET": "Model produced near-full candidate ranking (11-19 IDs)",
        "MODEL_PRODUCED_UNDER_K": "Model produced fewer than K IDs",
        "MODEL_PRODUCED_OVER_K_SHORT_LIST": "Model produced over-K short list (6-9 IDs)",
        "MODEL_PRODUCED_TOP10_ORIGINAL_CONTRACT": "Model produced 10 IDs, matching original top-10 habit",
        "MODEL_PRODUCED_MORE_THAN_CANDIDATE_SET": "Model produced more than 20 indices",
        "RUNNER_ACCEPTED_INCOMPLETE_OLLAMA_DONE_FALSE": "Runner accepted incomplete Ollama response with done=false",
        "PARSE_FAILURE": "Parse failure",
        "OTHER_STRUCTURAL_INVALIDITY": "Other structural invalidity",
        "UNKNOWN_WRONG_LENGTH": "Unknown wrong length",
    }
    return [
        {
            "root_cause": label_map.get(key, key),
            "technical_code": key,
            "count": count,
            "percent_invalid": count / len(invalid) if invalid else 0.0,
            "percent_total": count / total if total else 0.0,
        }
        for key, count in counts.most_common()
    ]


def artifact_integrity(data: Data, pair_summary: list[dict[str, Any]], raw_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_path = ROOT / "results" / "tables" / "reproduction_r5a_recovery_request_manifest.csv"
    selected = data.canddif
    counts = Counter((row["recovery_family"], row["experimental_condition"]) for row in selected)
    datasets = Counter(row["dataset"] for row in selected)
    families = Counter(row["recovery_family"] for row in data.manifest)
    return [
        {
            "check": "git_head",
            "observed": git(["rev-parse", "HEAD"]),
            "expected": EXPECTED_HEAD,
            "pass": git(["rev-parse", "HEAD"]) == EXPECTED_HEAD,
        },
        {
            "check": "manifest_sha256",
            "observed": sha256_file(manifest_path),
            "expected": EXPECTED_MANIFEST_SHA,
            "pass": sha256_file(manifest_path) == EXPECTED_MANIFEST_SHA,
        },
        {
            "check": "manifest_total_rows",
            "observed": len(data.manifest),
            "expected": 4800,
            "pass": len(data.manifest) == 4800,
        },
        {
            "check": "canddif_selected_rows",
            "observed": len(selected),
            "expected": 1200,
            "pass": len(selected) == 1200,
        },
        {
            "check": "first_random_counts",
            "observed": dict(counts),
            "expected": {"CANDDIF_FIRST": 600, "CANDDIF_RANDOM": 600},
            "pass": counts[("CANDDIF_RECOVERY", "CANDDIF_FIRST")] == 600
            and counts[("CANDDIF_RECOVERY", "CANDDIF_RANDOM")] == 600,
        },
        {
            "check": "dataset_counts",
            "observed": dict(datasets),
            "expected": {"Beauty": 400, "Sports": 400, "LastFM": 400},
            "pass": datasets == Counter({"Beauty": 400, "Sports": 400, "LastFM": 400}),
        },
        {
            "check": "out_of_scope_r5b_families",
            "observed": {key: families[key] for key in sorted(families)},
            "expected": "R5B selected CANDDIF only; R5A manifest still contains all families",
            "pass": all(row["recovery_family"] == "CANDDIF_RECOVERY" for row in selected)
            and not any(row["dataset"] == "ML-1M" for row in selected),
        },
        {
            "check": "pair_integrity",
            "observed": pair_summary[0],
            "expected": {"pairs": 600, "valid_pairs": 600, "invalid_pairs": 0},
            "pass": pair_summary[0]["pairs"] == 600 and pair_summary[0]["invalid_pairs"] == 0,
        },
        {
            "check": "raw_output_accounting",
            "observed": raw_summary[0],
            "expected": "1200 expected and 1200 accounted",
            "pass": bool(raw_summary[0]["accounting_pass"]),
        },
    ]


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    request_rows = build_request_level(data)
    pair_rows, pair_summary = independent_pair_integrity(data)
    raw_summary = raw_completeness(data)
    denom_rows, diagnostic_metrics = metric_reconstruction(request_rows, data.metrics)
    canddif_rows, diagnostic_canddif = canddif_reconstruction(denom_rows, diagnostic_metrics, data.canddif_results)
    sample_summary, sample_jsonl = select_forensic_samples(request_rows, data)
    qwen_summary, qwen_jsonl = qwen_diagnostics(request_rows, data)
    manual_pairs = manual_pair_reconstruction(request_rows, data)

    write_csv(OUT / "request_level_forensics.csv", request_rows)
    write_csv(OUT / "wrong_length_distribution.csv", grouped_length_distribution(request_rows, []))
    write_csv(OUT / "wrong_length_by_dataset.csv", grouped_length_distribution(request_rows, ["dataset"]))
    write_csv(OUT / "wrong_length_by_model.csv", grouped_length_distribution(request_rows, ["model_name"]))
    write_csv(OUT / "wrong_length_by_condition.csv", grouped_length_distribution(request_rows, ["condition"]))
    write_csv(OUT / "wrong_length_by_dataset_model.csv", grouped_length_distribution(request_rows, ["dataset", "model_name"]))
    write_csv(OUT / "wrong_length_by_model_condition.csv", grouped_length_distribution(request_rows, ["model_name", "condition"]))
    write_csv(
        OUT / "wrong_length_by_dataset_model_condition.csv",
        grouped_length_distribution(request_rows, ["dataset", "model_name", "condition"]),
    )
    write_csv(OUT / "validity_by_dataset_model_condition.csv", validity_breakdowns(request_rows, ["dataset", "model_name", "condition"]))
    write_csv(OUT / "validity_by_dataset_model.csv", validity_breakdowns(request_rows, ["dataset", "model_name"]))
    write_csv(OUT / "prompt_audit.csv", prompt_audit(data))
    write_csv(OUT / "ollama_config_audit.csv", ollama_config_audit(data))
    write_csv(OUT / "metric_denominator_reconstruction.csv", denom_rows)
    write_csv(OUT / "diagnostic_metrics_end_to_end_vs_conditional_valid.csv", diagnostic_metrics)
    write_csv(OUT / "independent_canddif_reconstruction.csv", canddif_rows)
    write_csv(OUT / "diagnostic_canddif_end_to_end_vs_conditional_valid.csv", diagnostic_canddif)
    write_csv(OUT / "pair_integrity_independent.csv", pair_rows)
    write_csv(OUT / "pair_integrity_summary.csv", pair_summary)
    write_csv(OUT / "raw_output_completeness.csv", raw_summary)
    write_csv(OUT / "smoke_reuse_verification.csv", smoke_reuse(data))
    write_csv(OUT / "manual_pair_reconstruction.csv", manual_pairs)
    write_csv(OUT / "forensic_sample_summary.csv", sample_summary)
    write_jsonl(OUT / "forensic_samples.jsonl", sample_jsonl)
    write_csv(OUT / "qwen_format_summary.csv", qwen_summary)
    write_jsonl(OUT / "qwen_54_samples.jsonl", qwen_jsonl)
    write_csv(OUT / "parser_synthetic_tests.csv", parser_synthetic_tests())
    write_csv(OUT / "original_protocol_comparison.csv", original_protocol_comparison())
    write_csv(OUT / "root_cause_attribution.csv", build_root_cause_summary(request_rows))
    write_csv(OUT / "artifact_integrity.csv", artifact_integrity(data, pair_summary, raw_summary))

    summary = {
        "requests": len(request_rows),
        "valid": sum(row["parse_valid"] == "true" for row in request_rows),
        "invalid": sum(row["parse_valid"] != "true" for row in request_rows),
        "wrong_length": sum(row["parse_status"] == "invalid_length" for row in request_rows),
        "parse_failed": sum(row["parse_status"] == "parse_failed" for row in request_rows),
        "wrong_length_done_stop": sum(
            row["parse_status"] == "invalid_length" and row["ollama_done"] is True and row["ollama_done_reason"] == "stop"
            for row in request_rows
        ),
        "parse_failed_done_false": sum(
            row["parse_status"] == "parse_failed" and row["ollama_done"] is False for row in request_rows
        ),
        "top5_salvage_possible_including_valid": sum(row["top5_salvage_valid"] for row in request_rows),
        "top5_salvage_possible_invalid_only": sum(
            row["top5_salvage_valid"] and row["parse_valid"] != "true" for row in request_rows
        ),
        "qwen_valid": sum(row["model_name"] == "qwen2:7b-instruct" and row["parse_valid"] == "true" for row in request_rows),
        "qwen_invalid": sum(row["model_name"] == "qwen2:7b-instruct" and row["parse_valid"] != "true" for row in request_rows),
        "mistral_valid": sum(row["model_name"] == "mistral:7b-instruct" and row["parse_valid"] == "true" for row in request_rows),
        "mistral_invalid": sum(row["model_name"] == "mistral:7b-instruct" and row["parse_valid"] != "true" for row in request_rows),
        "dominant_root_causes": build_root_cause_summary(request_rows),
    }
    write_json(OUT / "forensic_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
