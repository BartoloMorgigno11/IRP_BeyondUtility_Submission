"""Corrective V3 output-contract pilot for F6L LastFM ranking.

This module is intentionally separate from the prior V1/V2 parser forensics.
It builds a deterministic 40-request local-only pilot around JSON candidate
indices, executes only that pilot when explicitly authorized, and records
contract-validity diagnostics without updating scientific RQs.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.analysis import plan_f6l_paper_faithful_completion_and_rq_revalidation as f6l


PILOT_ID = "phase_f6l_lastfm_corrective_output_contract_pilot"
PILOT_CLASSIFICATION = "corrective_local_output_contract_pilot"
ORIGINAL_RUN_ID = "20260802_184502"
ORIGINAL_RAW_DIR = f6l.PROJECT_ROOT / "results" / "llm" / f"phase_f6l_lastfm_operationalized_ranking_{ORIGINAL_RUN_ID}"
PARSER_VERSION = "phase_f6l_lastfm_candidate_index_json_parser_v3"
OUTPUT_CONTRACT = "candidate_index_json_top10_v3"
DECISION_PASS = "f6l_lastfm_corrective_output_contract_pilot_passed_full_rerun_ready"
DECISION_FAILED = "f6l_lastfm_corrective_output_contract_pilot_failed_redesign_required"
DECISION_PREFLIGHT_BLOCKED = "f6l_lastfm_corrective_output_contract_pilot_blocked_by_preflight_regression"

TABLES_DIR = f6l.PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = f6l.PROJECT_ROOT / "results" / "manifests"
REPORT_PATH = f6l.PROJECT_ROOT / "report" / f"{PILOT_ID}.md"
NOTES_PATH = f6l.PROJECT_ROOT / "report" / "notes.md"
HANDOFF_PATH = f6l.PROJECT_ROOT / "report" / "future_operations_context_handoff.md"

PILOT_MANIFEST_PATH = TABLES_DIR / f"{PILOT_ID}_manifest.csv"
PILOT_SELECTION_PATH = TABLES_DIR / f"{PILOT_ID}_sample_selection.csv"
PILOT_PREFLIGHT_PATH = TABLES_DIR / f"{PILOT_ID}_preflight.csv"
PILOT_STATE_PATH = TABLES_DIR / f"{PILOT_ID}_state.csv"
PILOT_EXECUTION_PATH = TABLES_DIR / f"{PILOT_ID}_execution.csv"
PILOT_PARSE_RESULTS_PATH = TABLES_DIR / f"{PILOT_ID}_parse_results.csv"
PILOT_RECOMMENDATIONS_PATH = TABLES_DIR / f"{PILOT_ID}_recommendation_rows.csv"
PILOT_VALIDITY_PATH = TABLES_DIR / f"{PILOT_ID}_validity.csv"
PILOT_FAILURE_TAXONOMY_PATH = TABLES_DIR / f"{PILOT_ID}_failure_taxonomy.csv"
PILOT_ACCEPTANCE_PATH = TABLES_DIR / f"{PILOT_ID}_acceptance.csv"
PILOT_SMOKE_METRICS_PATH = TABLES_DIR / f"{PILOT_ID}_smoke_metrics.csv"
FULL_RUN_PLAN_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_full_run_plan.csv"
PILOT_JSON_MANIFEST_PATH = MANIFESTS_DIR / f"{PILOT_ID}_manifest.json"

PILOT_USER_COUNTS = {"short": 4, "medium": 3, "long": 3}
PILOT_REQUEST_COUNT = 40
PILOT_USER_COUNT = 10
STRATEGY_ORDER = ["base/OpenP5 title-list", "LLMRank recency"]
MODEL_ORDER = ["qwen2:7b-instruct", "mistral:7b-instruct"]
GENERATION_PARAMETERS = {
    "temperature": 0,
    "seed": 0,
    "stream": False,
    "response_format": "json_object",
    "timeout_seconds": 120,
    "scientific_retry_count": 0,
    "infrastructure_retry_count": 0,
}
OUTPUT_CONTRACT_RULES = {
    "format": "exactly_one_json_object",
    "schema": {"ranking": "list[int]"},
    "ranking_length": 10,
    "index_min": 1,
    "index_max": 20,
    "duplicates_allowed": False,
    "extra_text_allowed": False,
    "extra_keys_allowed": False,
    "markdown_allowed": False,
    "titles_allowed_in_output": False,
    "indices_refer_only_to_candidate_list": True,
    "positive_item_metadata_allowed_in_prompt": False,
    "example": {"ranking": [3, 17, 1, 9, 6, 12, 4, 20, 8, 11]},
}
FORBIDDEN_PROMPT_TOKENS = (
    "positive_item",
    "positive item",
    "positive_position",
    "positive position",
    "ground_truth",
    "ground truth",
    "is_positive",
    "test_item",
    "test item",
    "validation_item",
    "validation item",
    "hit rate",
    "hr@",
    "ndcg",
)
SELECTION_INPUT_FIELDS = (
    "user_id",
    "sample_ordinal",
    "train_history_count",
    "history_hash",
    "selected_seed",
)
SELECTION_BLIND_FORBIDDEN_FIELDS = (
    "validation_item_id",
    "test_item_id",
    "test_item_title",
    "positive_item",
    "positive_position",
    "is_positive",
    "v1",
    "v2",
    "raw_output",
    "hr",
    "ndcg",
)

MANIFEST_FIELDS = [
    "request_id",
    "pilot_id",
    "user_id",
    "history_length",
    "history_stratum",
    "model_name",
    "model_digest",
    "prompting_strategy",
    "candidate_list",
    "candidate_index_mapping",
    "candidate_list_hash",
    "prompt_template",
    "prompt_template_hash",
    "rendered_prompt_hash",
    "parser_version",
    "output_contract",
    "output_contract_hash",
    "generation_parameters",
    "expected_raw_output_path",
    "evidence_class",
    "lineage",
    "request_contract_hash",
]

SELECTION_FIELDS = [
    "pilot_id",
    "user_id",
    "sample_ordinal",
    "history_length",
    "history_stratum",
    "selection_seed",
    "history_quantile_policy",
    "stratum_rank",
    "selection_key_hash",
    "selection_input_fields",
    "selection_blind_to",
    "evidence_class",
    "lineage",
]

PREFLIGHT_FIELDS = ["check_id", "check_name", "status", "observed", "expected", "severity", "evidence", "notes"]
STATE_FIELDS = [
    "run_id",
    "stage",
    "decision_token",
    "planned",
    "completed",
    "strict_valid",
    "failed",
    "residual",
    "duplicate_output",
    "out_of_range",
    "malformed_json",
    "ambiguous_mapping",
    "extra_candidate",
    "percent_complete",
    "last_request_id",
    "last_timestamp",
    "throughput_requests_per_hour",
    "eta_seconds",
    "raw_output_dir",
    "notes",
]
EXECUTION_FIELDS = [
    "request_id",
    "pilot_id",
    "user_id",
    "model_name",
    "model_digest",
    "prompting_strategy",
    "attempt_index",
    "started_at",
    "completed_at",
    "duration_seconds",
    "backend_success",
    "response_nonempty",
    "done_reason",
    "prompt_eval_count",
    "eval_count",
    "load_duration_nanoseconds",
    "prompt_eval_duration_nanoseconds",
    "eval_duration_nanoseconds",
    "total_duration_nanoseconds",
    "raw_response_sha256",
    "backend_error_type",
    "backend_error_message",
    "response_jsonl_recorded",
    "raw_json_path",
    "execution_status",
    "notes",
]
PARSE_FIELDS = [
    "request_id",
    "pilot_id",
    "user_id",
    "model_name",
    "model_digest",
    "prompting_strategy",
    "parser_version",
    "parser_source_hash",
    "output_contract",
    "output_contract_hash",
    "prompt_template_hash",
    "rendered_prompt_hash",
    "candidate_list_hash",
    "candidate_index_mapping_hash",
    "mapping_hash",
    "raw_response_sha256",
    "strict_valid",
    "failure_reason",
    "ranking_indices",
    "ranked_item_ids",
    "ranked_titles",
    "output_count",
    "duplicate_output_count",
    "out_of_range_count",
    "malformed_json_count",
    "ambiguous_mapping_count",
    "extra_candidate_count",
    "request_contract_hash",
    "hash_coherence",
    "evidence_class",
    "notes",
]
RECOMMENDATION_FIELDS = [
    "request_id",
    "pilot_id",
    "user_id",
    "model_name",
    "model_digest",
    "prompting_strategy",
    "rank",
    "candidate_index",
    "recommended_item_id",
    "recommended_item_title",
    "candidate_list_hash",
    "candidate_index_mapping_hash",
    "parser_version",
    "evidence_class",
    "notes",
]
VALIDITY_FIELDS = [
    "pilot_id",
    "run_id",
    "model_name",
    "prompting_strategy",
    "planned",
    "completed",
    "strict_valid",
    "failed",
    "strict_valid_rate",
    "duplicate_output_count",
    "out_of_range_count",
    "malformed_json_count",
    "ambiguous_mapping_count",
    "extra_candidate_count",
    "residual",
    "evidence_class",
    "notes",
]
FAILURE_FIELDS = ["pilot_id", "run_id", "failure_reason", "count", "evidence_class", "notes"]
ACCEPTANCE_FIELDS = ["pilot_id", "run_id", "criterion", "observed", "expected", "status", "evidence", "notes"]
SMOKE_FIELDS = [
    "pilot_id",
    "run_id",
    "model_name",
    "prompting_strategy",
    "metric",
    "value",
    "evaluated_requests",
    "evidence_class",
    "notes",
]
FULL_RUN_PLAN_FIELDS = ["plan_id", "pilot_run_id", "future_run_id_placeholder", "status", "future_command", "request_count", "notes"]


@dataclass(frozen=True)
class PilotBuild:
    run_id: str
    raw_output_dir: Path
    manifest_rows: list[dict[str, str]]
    selection_rows: list[dict[str, str]]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ParserV3Result:
    strict_valid: bool
    failure_reason: str
    ranking_indices: list[int]
    ranked_item_ids: list[str]
    ranked_titles: list[str]
    output_count: int
    duplicate_output_count: int
    out_of_range_count: int
    malformed_json_count: int
    ambiguous_mapping_count: int
    extra_candidate_count: int
    parsed_payload: dict[str, Any] | None


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def json_list(values: Iterable[Any]) -> str:
    return json.dumps(list(values), ensure_ascii=True, separators=(",", ":"))


def format_float(value: float, digits: int = 12) -> str:
    if math.isnan(value):
        return ""
    text = f"{value:.{digits}g}"
    return "0" if text == "-0" else text


def stable_hash(*parts: object) -> str:
    return f6l.sha256_text("|".join(str(part) for part in parts))


def stable_request_id(index: int, user_id: str, model_name: str, strategy: str) -> str:
    suffix = stable_hash(PILOT_ID, user_id, model_name, strategy)[:10]
    return f"F6L-LASTFM-V3-PILOT-{index:04d}-{suffix}"


def local_raw_output_dir(run_id: str) -> Path:
    return f6l.PROJECT_ROOT / "results" / "llm" / f"{PILOT_ID}_{run_id}"


def output_contract_hash() -> str:
    return f6l.sha256_text(canonical_json(OUTPUT_CONTRACT_RULES))


def prompt_template_name(strategy: str) -> str:
    if strategy == "base/OpenP5 title-list":
        return f"{PILOT_ID}_v3_base_openp5_index_json"
    if strategy == "LLMRank recency":
        return f"{PILOT_ID}_v3_llmrank_recency_index_json"
    raise ValueError(f"unsupported pilot strategy: {strategy}")


def prompt_template_text(strategy: str) -> str:
    shared = "\n".join(
        [
            "You are ranking LastFM candidate artists/items for a recommender task.",
            "Use the user's history and the candidate list only.",
            "Candidate indices are stable identifiers and refer only to the numbered candidate list in this prompt.",
            "Return exactly one JSON object and nothing else.",
            'The required schema is {{"ranking":[3,17,1,9,6,12,4,20,8,11]}}.',
            "The ranking value must contain exactly 10 distinct integers.",
            "Each integer must be between 1 and 20.",
            "Do not output titles, item ids, markdown, comments, explanations, or additional keys.",
            "",
            "User history, earlier to later:",
            "{history_block}",
            "",
            "Candidate list:",
            "{candidate_block}",
            "",
            "Task: rank the 10 candidates most likely to be the user's next preferred artist/item, from most likely to least likely.",
        ]
    )
    if strategy == "base/OpenP5 title-list":
        return shared
    if strategy == "LLMRank recency":
        return "\n".join(
            [
                shared,
                "Recency cue: give appropriate attention to the later items in the history while still using the full history.",
                "Final answer must be only the JSON object.",
            ]
        )
    raise ValueError(f"unsupported pilot strategy: {strategy}")


def prompt_template_hash(strategy: str) -> str:
    return f6l.sha256_text(prompt_template_text(strategy))


def parser_source_hash() -> str:
    return f6l.sha256_text(inspect.getsource(parse_v3_response))


def candidate_index_mapping(candidate_rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, str]]:
    rows = sorted(candidate_rows, key=lambda row: int(row["candidate_position"]))
    return {
        str(index): {
            "candidate_index": str(index),
            "item_id": str(row["candidate_item_id"]),
            "title": str(row["candidate_item_title"]),
        }
        for index, row in enumerate(rows, start=1)
    }


def candidate_index_mapping_hash(candidate_rows: Sequence[Mapping[str, str]]) -> str:
    return f6l.sha256_text(canonical_json(candidate_index_mapping(candidate_rows)))


def render_candidate_block(candidate_rows: Sequence[Mapping[str, str]]) -> str:
    lines = []
    for index, row in enumerate(sorted(candidate_rows, key=lambda item: int(item["candidate_position"])), start=1):
        lines.append(f"{index}. item_id={row['candidate_item_id']} | title={row['candidate_item_title']}")
    return "\n".join(lines)


def render_history_block(sample_row: Mapping[str, str]) -> str:
    titles = f6l.parse_json_list(sample_row.get("history_item_titles", "[]"))
    if not titles:
        return "(empty history)"
    return "\n".join(f"{index}. {title}" for index, title in enumerate(titles, start=1))


def render_v3_prompt(sample_row: Mapping[str, str], candidate_rows: Sequence[Mapping[str, str]], strategy: str) -> str:
    prompt = prompt_template_text(strategy).format(
        history_block=render_history_block(sample_row),
        candidate_block=render_candidate_block(candidate_rows),
    )
    lowered = prompt.casefold()
    forbidden = [token for token in FORBIDDEN_PROMPT_TOKENS if token in lowered]
    if forbidden:
        raise ValueError(f"forbidden prompt token(s): {forbidden}")
    return prompt


def parse_v3_response(text: str, candidate_rows: Sequence[Mapping[str, str]]) -> ParserV3Result:
    """Parse the V3 strict candidate-index JSON response contract.

    The accepted output is exactly one JSON object with only a ``ranking`` key,
    where the value is a list of 10 distinct integer indices in the inclusive
    range [1, 20]. No title, fuzzy, positive item, HR, or NDCG information is
    used.
    """
    stripped = (text or "").strip()
    if not stripped:
        return ParserV3Result(False, "empty_response", [], [], [], 0, 0, 0, 1, 0, 0, None)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return ParserV3Result(False, "malformed_json", [], [], [], 0, 0, 0, 1, 0, 0, None)
    if not isinstance(payload, dict):
        return ParserV3Result(False, "incompatible_json_structure", [], [], [], 0, 0, 0, 0, 0, 0, None)
    if set(payload) != {"ranking"}:
        reason = "extra_or_missing_json_keys"
        return ParserV3Result(False, reason, [], [], [], 0, 0, 0, 0, 0, 0, payload)
    ranking = payload.get("ranking")
    if not isinstance(ranking, list):
        return ParserV3Result(False, "ranking_not_list", [], [], [], 0, 0, 0, 0, 0, 0, payload)
    output_count = len(ranking)
    extra_candidate_count = max(0, output_count - 10)
    if output_count != 10:
        return ParserV3Result(False, "wrong_ranking_length", [], [], [], output_count, 0, 0, 0, 0, extra_candidate_count, payload)
    non_integer = [item for item in ranking if not isinstance(item, int) or isinstance(item, bool)]
    if non_integer:
        return ParserV3Result(False, "non_integer_index", [], [], [], output_count, 0, 0, 0, 0, 0, payload)
    indices = [int(item) for item in ranking]
    out_of_range = [item for item in indices if item < 1 or item > 20]
    if out_of_range:
        return ParserV3Result(False, "out_of_range_index", indices, [], [], output_count, 0, len(out_of_range), 0, 0, 0, payload)
    duplicate_count = output_count - len(set(indices))
    if duplicate_count:
        return ParserV3Result(False, "duplicate_index", indices, [], [], output_count, duplicate_count, 0, 0, 0, 0, payload)
    mapping = candidate_index_mapping(candidate_rows)
    if {int(index) for index in mapping} != set(range(1, 21)):
        return ParserV3Result(False, "ambiguous_candidate_index_mapping", indices, [], [], output_count, 0, 0, 0, 1, 0, payload)
    ranked_item_ids = [mapping[str(index)]["item_id"] for index in indices]
    ranked_titles = [mapping[str(index)]["title"] for index in indices]
    return ParserV3Result(True, "", indices, ranked_item_ids, ranked_titles, output_count, 0, 0, 0, 0, 0, payload)


def sample_rows_by_user() -> dict[str, dict[str, str]]:
    rows = f6l.read_csv_rows(f6l.LASTFM_OPERATIONALIZED_SAMPLE_COHORT_PATH)
    return {str(row["user_id"]): row for row in rows}


def select_pilot_users(sample_rows: Sequence[Mapping[str, str]] | None = None) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = [dict(row) for row in (sample_rows if sample_rows is not None else f6l.read_csv_rows(f6l.LASTFM_OPERATIONALIZED_SAMPLE_COHORT_PATH))]
    rows.sort(key=lambda row: int(row["sample_ordinal"]))
    if len(rows) != 1000:
        raise ValueError(f"expected 1000 operationalized users, found {len(rows)}")
    base_seed = rows[0].get("selected_seed", "20260801")
    selection_seed = f6l.sha256_text(f"{PILOT_ID}|{base_seed}|history_stratified_blind_v1")[:16]
    history_lengths = sorted(int(row["train_history_count"]) for row in rows)
    q1 = history_lengths[math.ceil(len(history_lengths) / 3) - 1]
    q2 = history_lengths[math.ceil(2 * len(history_lengths) / 3) - 1]

    def stratum_for(length: int) -> str:
        if length <= q1:
            return "short"
        if length <= q2:
            return "medium"
        return "long"

    by_stratum: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        history_length = int(row["train_history_count"])
        stratum = stratum_for(history_length)
        key = stable_hash(selection_seed, stratum, row["user_id"], row["sample_ordinal"])
        by_stratum[stratum].append({**row, "_history_stratum": stratum, "_selection_key_hash": key})
    selected: list[dict[str, str]] = []
    for stratum in ["short", "medium", "long"]:
        candidates = sorted(by_stratum[stratum], key=lambda row: (row["_selection_key_hash"], int(row["user_id"])))
        needed = PILOT_USER_COUNTS[stratum]
        if len(candidates) < needed:
            raise ValueError(f"not enough users in stratum {stratum}: {len(candidates)} < {needed}")
        selected.extend(candidates[:needed])
    selected.sort(key=lambda row: (["short", "medium", "long"].index(row["_history_stratum"]), row["_selection_key_hash"]))
    meta = {
        "selection_seed": selection_seed,
        "base_sample_seed": base_seed,
        "history_quantile_policy": f"tertiles_over_1000_train_history_count:q1_le_{q1}:q2_le_{q2}",
        "history_quantile_thresholds": {"short_max": q1, "medium_max": q2},
        "selection_input_fields": list(SELECTION_INPUT_FIELDS),
        "selection_blind_to": list(SELECTION_BLIND_FORBIDDEN_FIELDS),
    }
    return selected, meta


def build_selection_rows(selected: Sequence[Mapping[str, str]], meta: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    ranks: Counter[str] = Counter()
    for row in selected:
        stratum = str(row["_history_stratum"])
        ranks[stratum] += 1
        rows.append(
            {
                "pilot_id": PILOT_ID,
                "user_id": row["user_id"],
                "sample_ordinal": row["sample_ordinal"],
                "history_length": row["train_history_count"],
                "history_stratum": stratum,
                "selection_seed": str(meta["selection_seed"]),
                "history_quantile_policy": str(meta["history_quantile_policy"]),
                "stratum_rank": str(ranks[stratum]),
                "selection_key_hash": row["_selection_key_hash"],
                "selection_input_fields": json_list(SELECTION_INPUT_FIELDS),
                "selection_blind_to": json_list(SELECTION_BLIND_FORBIDDEN_FIELDS),
                "evidence_class": PILOT_CLASSIFICATION,
                "lineage": f6l.rel(f6l.LASTFM_OPERATIONALIZED_SAMPLE_COHORT_PATH),
            }
        )
    return rows


def manifest_row_for_request(
    *,
    index: int,
    run_id: str,
    raw_output_dir: Path,
    sample_row: Mapping[str, str],
    candidate_rows: Sequence[Mapping[str, str]],
    model_name: str,
    strategy: str,
) -> dict[str, str]:
    prompt = render_v3_prompt(sample_row, candidate_rows, strategy)
    candidate_ids = [str(row["candidate_item_id"]) for row in sorted(candidate_rows, key=lambda item: int(item["candidate_position"]))]
    mapping = candidate_index_mapping(candidate_rows)
    mapping_hash = f6l.sha256_text(canonical_json(mapping))
    row = {
        "request_id": stable_request_id(index, sample_row["user_id"], model_name, strategy),
        "pilot_id": PILOT_ID,
        "user_id": sample_row["user_id"],
        "history_length": sample_row["train_history_count"],
        "history_stratum": str(sample_row["_history_stratum"]),
        "model_name": model_name,
        "model_digest": f6l.MODEL_DIGESTS[model_name],
        "prompting_strategy": strategy,
        "candidate_list": json_list(candidate_ids),
        "candidate_index_mapping": canonical_json(mapping),
        "candidate_list_hash": str(candidate_rows[0]["candidate_list_hash"]),
        "prompt_template": prompt_template_name(strategy),
        "prompt_template_hash": prompt_template_hash(strategy),
        "rendered_prompt_hash": f6l.sha256_text(prompt),
        "parser_version": PARSER_VERSION,
        "output_contract": OUTPUT_CONTRACT,
        "output_contract_hash": output_contract_hash(),
        "generation_parameters": canonical_json(GENERATION_PARAMETERS),
        "expected_raw_output_path": f6l.rel(raw_output_dir / "responses" / f"{stable_request_id(index, sample_row['user_id'], model_name, strategy)}.json"),
        "evidence_class": PILOT_CLASSIFICATION,
        "lineage": ";".join(
            [
                f6l.rel(f6l.LASTFM_OPERATIONALIZED_SAMPLE_COHORT_PATH),
                f6l.rel(f6l.LASTFM_OPERATIONALIZED_CANDIDATES_PATH),
                "parser_v3_separate_from_v1_v2",
                f"candidate_index_mapping_hash={mapping_hash}",
            ]
        ),
        "request_contract_hash": "",
    }
    contract_payload = {
        key: row[key]
        for key in [
            "pilot_id",
            "user_id",
            "history_length",
            "history_stratum",
            "model_name",
            "model_digest",
            "prompting_strategy",
            "candidate_list_hash",
            "prompt_template_hash",
            "rendered_prompt_hash",
            "parser_version",
            "output_contract_hash",
            "generation_parameters",
        ]
    }
    contract_payload["candidate_index_mapping_hash"] = mapping_hash
    row["request_contract_hash"] = f6l.sha256_text(canonical_json(contract_payload))
    return row


def build_pilot(run_id: str | None = None, *, raw_output_dir: Path | None = None, write_outputs: bool = True) -> PilotBuild:
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_output_dir = raw_output_dir or local_raw_output_dir(run_id)
    selected, selection_meta = select_pilot_users()
    selection_rows = build_selection_rows(selected, selection_meta)
    candidates_by_user = f6l.candidate_rows_for_operationalized_sample()
    rows: list[dict[str, str]] = []
    request_index = 0
    for sample_row in selected:
        user_candidates = candidates_by_user[str(sample_row["user_id"])]
        for model_name in MODEL_ORDER:
            for strategy in STRATEGY_ORDER:
                request_index += 1
                rows.append(
                    manifest_row_for_request(
                        index=request_index,
                        run_id=run_id,
                        raw_output_dir=raw_output_dir,
                        sample_row=sample_row,
                        candidate_rows=user_candidates,
                        model_name=model_name,
                        strategy=strategy,
                    )
                )
    manifest = {
        "pilot_id": PILOT_ID,
        "run_id": run_id,
        "created_at": utc_now(),
        "classification": PILOT_CLASSIFICATION,
        "original_run_preserved_as": "prompt_parser_interface_fragility",
        "original_run_id": ORIGINAL_RUN_ID,
        "request_count": len(rows),
        "user_count": len({row["user_id"] for row in rows}),
        "models": {model: f6l.MODEL_DIGESTS[model] for model in MODEL_ORDER},
        "strategies": STRATEGY_ORDER,
        "parser_version": PARSER_VERSION,
        "parser_source_hash": parser_source_hash(),
        "output_contract": OUTPUT_CONTRACT,
        "output_contract_hash": output_contract_hash(),
        "generation_parameters": GENERATION_PARAMETERS,
        "selection": selection_meta,
        "manifest_hash": f6l.sha256_text(canonical_json(rows)),
        "inference_limit": PILOT_REQUEST_COUNT,
        "full_cohort_execution_performed": False,
        "reranking_performed": False,
        "remote_api_calls_performed": False,
        "training_or_downloads_performed": False,
    }
    if write_outputs:
        f6l.write_csv(PILOT_MANIFEST_PATH, rows, MANIFEST_FIELDS)
        f6l.write_csv(PILOT_SELECTION_PATH, selection_rows, SELECTION_FIELDS)
        f6l.write_json(PILOT_JSON_MANIFEST_PATH, manifest)
    return PilotBuild(run_id, raw_output_dir, rows, selection_rows, manifest)


def raw_hash_map(raw_output_dir: Path) -> dict[str, str]:
    paths = f6l.raw_response_paths(raw_output_dir)
    files = [paths["requests"], paths["responses"]]
    if paths["individual"].exists():
        files.extend(sorted(path for path in paths["individual"].glob("*.json") if path.is_file()))
    return {f6l.rel(path): f6l.safe_file_sha256(path) for path in files if path.exists()}


def original_raw_hash_map_sha() -> str:
    return f6l.sha256_text(canonical_json(raw_hash_map(ORIGINAL_RAW_DIR)))


def original_raw_expected_sha() -> str:
    rows = f6l.read_csv_rows(TABLES_DIR / "phase_f6l_lastfm_reparse_raw_protection.csv")
    if not rows:
        return ""
    return rows[0].get("raw_hash_map_after_sha256", "")


def run_focused_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        "results/logs/phase_f6l_lastfm_corrective_output_contract_pilot_pytest",
        "-q",
        "tests/test_phase_f6l_lastfm_corrective_output_contract_pilot.py",
        "tests/test_phase_f6l_lastfm_execution_contracts.py",
        "tests/test_phase_f6l_lastfm_parser_forensics.py",
        "tests/test_phase_f6l_paper_faithful_completion_and_rq_revalidation.py",
    ]
    started = utc_now()
    start_clock = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=f6l.PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout_tail": "\n".join(completed.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(completed.stderr.splitlines()[-20:]),
        "stdout_sha256": f6l.sha256_text(completed.stdout),
        "stderr_sha256": f6l.sha256_text(completed.stderr),
        "started_at": started,
        "completed_at": utc_now(),
        "duration_seconds": time.perf_counter() - start_clock,
    }


def validate_pilot_manifest(rows: Sequence[Mapping[str, str]]) -> list[tuple[str, bool, str, str, str]]:
    checks: list[tuple[str, bool, str, str, str]] = []
    request_ids = [row["request_id"] for row in rows]
    users = {row["user_id"] for row in rows}
    per_user = Counter(row["user_id"] for row in rows)
    models = {row["model_name"] for row in rows}
    strategies = {row["prompting_strategy"] for row in rows}
    checks.append(("manifest_rows_40", len(rows) == 40, str(len(rows)), "40", f6l.rel(PILOT_MANIFEST_PATH)))
    checks.append(("unique_request_ids_40", len(set(request_ids)) == 40, str(len(set(request_ids))), "40", f6l.rel(PILOT_MANIFEST_PATH)))
    checks.append(("exactly_10_users", len(users) == 10, str(len(users)), "10", f6l.rel(PILOT_SELECTION_PATH)))
    checks.append(("four_requests_per_user", set(per_user.values()) == {4}, ",".join(str(v) for v in sorted(set(per_user.values()))), "4", f6l.rel(PILOT_MANIFEST_PATH)))
    checks.append(("two_models_only", models == set(MODEL_ORDER), ";".join(sorted(models)), ";".join(MODEL_ORDER), f6l.rel(PILOT_MANIFEST_PATH)))
    checks.append(("two_strategies_only", strategies == set(STRATEGY_ORDER), ";".join(sorted(strategies)), ";".join(STRATEGY_ORDER), f6l.rel(PILOT_MANIFEST_PATH)))
    checks.append(("no_llmsrec", not any("LLMSRec" in row["prompting_strategy"] for row in rows), "none" if not any("LLMSRec" in row["prompting_strategy"] for row in rows) else "present", "none", f6l.rel(PILOT_MANIFEST_PATH)))
    checks.append(("no_full_cohort", len(users) == 10 and len(rows) == PILOT_REQUEST_COUNT, f"users={len(users)};requests={len(rows)}", "users=10;requests=40", f6l.rel(PILOT_MANIFEST_PATH)))
    checks.append(("no_reranking", all("rerank" not in row["prompting_strategy"].casefold() for row in rows), "ranking_only", "ranking_only", f6l.rel(PILOT_MANIFEST_PATH)))
    checks.append(("candidate_lists_same_across_cells", all(len({row["candidate_list_hash"] for row in rows if row["user_id"] == user}) == 1 for user in users), "pass", "one candidate hash per user", f6l.rel(PILOT_MANIFEST_PATH)))
    stratum_counts = Counter(row["history_stratum"] for row in rows)
    checks.append(("history_strata_4_3_3_users", {key: stratum_counts[key] // 4 for key in PILOT_USER_COUNTS} == PILOT_USER_COUNTS, canonical_json({key: stratum_counts[key] // 4 for key in PILOT_USER_COUNTS}), canonical_json(PILOT_USER_COUNTS), f6l.rel(PILOT_SELECTION_PATH)))
    generation_ok = all(json.loads(row["generation_parameters"]) == GENERATION_PARAMETERS for row in rows)
    checks.append(("generation_parameters_v3_json_local_contract", generation_ok, "pass" if generation_ok else "fail", canonical_json(GENERATION_PARAMETERS), f6l.rel(PILOT_MANIFEST_PATH)))
    prompt_ok = True
    prompt_failures: list[str] = []
    sample_map = sample_rows_by_user()
    candidates_by_user = f6l.candidate_rows_for_operationalized_sample()
    for row in rows:
        try:
            prompt = render_v3_prompt(sample_map[row["user_id"]], candidates_by_user[row["user_id"]], row["prompting_strategy"])
        except Exception as exc:
            prompt_ok = False
            prompt_failures.append(f"{row['request_id']}:{exc}")
            continue
        if f6l.sha256_text(prompt) != row["rendered_prompt_hash"]:
            prompt_ok = False
            prompt_failures.append(f"{row['request_id']}:hash_mismatch")
    checks.append(("no_positive_or_metric_tokens_in_prompt", prompt_ok, ";".join(prompt_failures[:3]) if prompt_failures else "none", "none", "render_v3_prompt"))
    selection_blind = not any(field in SELECTION_INPUT_FIELDS for field in SELECTION_BLIND_FORBIDDEN_FIELDS)
    checks.append(("selection_blind_to_positive_parser_and_metric_outcomes", selection_blind, json_list(SELECTION_INPUT_FIELDS), "no positive/parser/raw/HR/NDCG fields", f6l.rel(PILOT_SELECTION_PATH)))
    return checks


def validate_preflight(
    build: PilotBuild,
    *,
    raw_output_dir: Path,
    resume: bool,
    allow_api_calls: bool,
    local_only: bool,
    endpoint: str,
    profile_hashes_before: Mapping[str, str],
    run_tests: bool,
) -> tuple[list[dict[str, str]], dict[str, Any], f6l.LocalOllamaState | None]:
    rows: list[dict[str, str]] = []
    test_result: dict[str, Any] = {}
    ollama_state: f6l.LocalOllamaState | None = None

    def add(name: str, passed: bool, observed: str, expected: str, evidence: str, notes: str, severity: str = "error") -> None:
        rows.append(
            {
                "check_id": f"F6L-LASTFM-V3-PREFLIGHT-{len(rows) + 1:02d}",
                "check_name": name,
                "status": "pass" if passed else "fail",
                "observed": observed,
                "expected": expected,
                "severity": severity,
                "evidence": evidence,
                "notes": notes,
            }
        )

    for name, passed, observed, expected, evidence in validate_pilot_manifest(build.manifest_rows):
        add(name, passed, observed, expected, evidence, "Pilot manifest contract check.")

    raw_dir_exists = raw_output_dir.exists()
    raw_dir_ok = (resume and raw_dir_exists) or ((not resume) and (not raw_dir_exists))
    add("raw_output_dir_append_only_policy", raw_dir_ok, f"exists={raw_dir_exists};resume={resume};dir={f6l.rel(raw_output_dir) if raw_output_dir.is_relative_to(f6l.PROJECT_ROOT) else raw_output_dir}", "new nonexistent dir unless explicit resume", "CLI args", "No overwrite of existing pilot raw output is allowed.")

    add("allow_api_calls_explicit_for_local_inference", allow_api_calls, str(allow_api_calls).lower(), "true", "CLI args", "Repository API convention remains explicit even for local Ollama.")
    add("local_only_flag", local_only, str(local_only).lower(), "true", "CLI args", "Remote providers are not authorized.")
    endpoint_ok = endpoint in {f6l.OLLAMA_LOCAL_GENERATE_URL, "http://127.0.0.1:11434/api/generate", "http://localhost:11434/api/generate"}
    add("endpoint_loopback_only", endpoint_ok, endpoint, f6l.OLLAMA_LOCAL_GENERATE_URL, "CLI args", "Only localhost Ollama endpoint is authorized.")
    add("no_remote_model_names", set(row["model_name"] for row in build.manifest_rows) == set(MODEL_ORDER), ";".join(sorted({row["model_name"] for row in build.manifest_rows})), ";".join(MODEL_ORDER), f6l.rel(PILOT_MANIFEST_PATH), "Only Qwen2 and Mistral local open models are included.")
    add("no_training_downloads_or_original_repo_edits_requested", True, "false", "false", f6l.rel(PILOT_MANIFEST_PATH), "This script has no training/download/original_repo write path.")
    original_status = f6l.git_output(["status", "--short", "original_repo"])
    add("original_repo_clean", original_status == "", original_status or "none", "none", "git status --short original_repo", "The original repository remains read-only.")
    staged = f6l.git_output(["diff", "--cached", "--name-only"])
    add("git_staging_empty", staged == "", staged or "none", "none", "git diff --cached --name-only", "No commit or staging is authorized.")
    protected = f6l.hash_protected_inputs()
    add("protected_inputs_pass", protected.get("status") == "pass", str(protected.get("status")), "pass", f6l.rel(f6l.PROTECTED_HASHES_PATH), "Protected inputs must remain intact.")
    add("twenty_three_yaml_invariant", protected.get("f6i2p_yaml_count") == 23, str(protected.get("f6i2p_yaml_count")), "23", f6l.rel(f6l.PROTECTED_HASHES_PATH), "No protected YAML mutation is authorized.")
    current_original_sha = original_raw_hash_map_sha()
    expected_original_sha = original_raw_expected_sha()
    add("original_4000_raw_unchanged", current_original_sha == expected_original_sha and bool(expected_original_sha), current_original_sha or "missing", expected_original_sha or "recorded raw protection sha", f6l.rel(ORIGINAL_RAW_DIR), "The 4000-output run is preserved as prompt/parser interface fragility evidence.")
    profile_after = f6l.profile_raw_hashes()
    add("profile_raw_unchanged", profile_after == dict(profile_hashes_before), f"before={len(profile_hashes_before)};after={len(profile_after)}", "same hash map", "results/llm/phase_f6l_profile_execution_*", "Pilot must not touch profile raw outputs.")

    if run_tests:
        test_result = run_focused_tests()
        add(
            "focused_f6l_tests_pass",
            test_result["exit_code"] == 0,
            f"exit={test_result['exit_code']};stdout_tail={test_result['stdout_tail']}",
            "exit=0",
            "pytest focused F6L shard",
            "Pilot inference is blocked unless focused F6L tests pass first.",
        )
    else:
        add("focused_f6l_tests_pass", True, "skipped_by_unit_test", "pass", "unit-test-only code path", "Execution path always runs this check.", severity="info")

    if all(row["status"] == "pass" for row in rows):
        ollama_state = f6l.start_local_ollama_if_needed(build.run_id)
        digest_ok = ollama_state.service_available and all(
            ollama_state.model_digests.get(model, "").lower() == f6l.MODEL_DIGESTS[model].lower()
            for model in MODEL_ORDER
        )
        add(
            "local_ollama_digests_match",
            digest_ok,
            canonical_json({model: ollama_state.model_digests.get(model, "") for model in MODEL_ORDER}),
            canonical_json({model: f6l.MODEL_DIGESTS[model] for model in MODEL_ORDER}),
            f6l.OLLAMA_LOCAL_TAGS_URL,
            f"start_attempted={ollama_state.start_attempted};pid={ollama_state.started_pid};error={ollama_state.error}",
        )
    else:
        add("local_ollama_digests_match", False, "not_checked_due_prior_preflight_failure", "local Qwen2/Mistral digests available", f6l.OLLAMA_LOCAL_TAGS_URL, "Ollama was not started because an earlier preflight gate failed.")

    f6l.write_csv(PILOT_PREFLIGHT_PATH, rows, PREFLIGHT_FIELDS)
    return rows, test_result, ollama_state


def execute_local_ollama_json_request(model: str, prompt: str, generation_parameters: Mapping[str, Any], endpoint: str) -> f6l.LocalOllamaRequestResult:
    started = utc_now()
    start_clock = time.perf_counter()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": float(generation_parameters.get("temperature", 0)),
            "seed": int(generation_parameters.get("seed", 0)),
        },
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=int(generation_parameters.get("timeout_seconds", 120))) as response:
            raw_body = response.read().decode("utf-8")
        raw = json.loads(raw_body)
        response_text = str(raw.get("response", ""))
        success = bool(response_text.strip()) and bool(raw.get("done", True))
        return f6l.LocalOllamaRequestResult(success, response_text, raw, "", "", started, utc_now(), time.perf_counter() - start_clock)
    except urllib.error.HTTPError as exc:
        return f6l.LocalOllamaRequestResult(False, "", {}, "HTTPError", f"{exc.code}:{exc.reason}", started, utc_now(), time.perf_counter() - start_clock)
    except urllib.error.URLError as exc:
        return f6l.LocalOllamaRequestResult(False, "", {}, "URLError", str(exc.reason), started, utc_now(), time.perf_counter() - start_clock)
    except TimeoutError as exc:
        return f6l.LocalOllamaRequestResult(False, "", {}, "TimeoutError", str(exc), started, utc_now(), time.perf_counter() - start_clock)
    except Exception as exc:
        return f6l.LocalOllamaRequestResult(False, "", {}, type(exc).__name__, str(exc), started, utc_now(), time.perf_counter() - start_clock)


def parse_row_to_dict(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "request_id": row["request_id"],
        "pilot_id": row["pilot_id"],
        "user_id": row["user_id"],
        "model_name": row["model_name"],
        "model_digest": row["model_digest"],
        "prompting_strategy": row["prompting_strategy"],
        "candidate_list_hash": row["candidate_list_hash"],
        "prompt_template_hash": row["prompt_template_hash"],
        "rendered_prompt_hash": row["rendered_prompt_hash"],
        "parser_version": row["parser_version"],
        "output_contract": row["output_contract"],
        "output_contract_hash": row["output_contract_hash"],
        "generation_parameters": json.loads(row["generation_parameters"]),
        "candidate_index_mapping": json.loads(row["candidate_index_mapping"]),
        "request_contract_hash": row["request_contract_hash"],
    }


def raw_record_for_response(
    *,
    row: Mapping[str, str],
    prompt: str,
    candidate_rows: Sequence[Mapping[str, str]],
    result: f6l.LocalOllamaRequestResult,
    parse: ParserV3Result,
    raw_json_path: Path,
    attempt_index: int,
) -> dict[str, Any]:
    mapping = candidate_index_mapping(candidate_rows)
    return {
        **parse_row_to_dict(row),
        "prompt": prompt,
        "candidate_index_mapping_hash": f6l.sha256_text(canonical_json(mapping)),
        "raw_output_text": result.response_text,
        "raw_backend_response": result.raw_backend_response,
        "parser_output": {
            "strict_valid": parse.strict_valid,
            "failure_reason": parse.failure_reason,
            "ranking_indices": parse.ranking_indices,
            "ranked_item_ids": parse.ranked_item_ids,
            "ranked_titles": parse.ranked_titles,
            "output_count": parse.output_count,
            "duplicate_output_count": parse.duplicate_output_count,
            "out_of_range_count": parse.out_of_range_count,
            "malformed_json_count": parse.malformed_json_count,
            "ambiguous_mapping_count": parse.ambiguous_mapping_count,
            "extra_candidate_count": parse.extra_candidate_count,
            "parsed_payload": parse.parsed_payload,
        },
        "parser_source_hash": parser_source_hash(),
        "backend_success": result.backend_success,
        "backend_error_type": result.backend_error_type,
        "backend_error_message": result.backend_error_message,
        "started_at": result.started_at,
        "completed_at": result.completed_at,
        "duration_seconds": result.duration_seconds,
        "raw_response_sha256": f6l.sha256_text(result.response_text),
        "attempt_index": attempt_index,
        "is_final_attempt": True,
        "raw_json_path": f6l.rel(raw_json_path),
        "evidence_class": PILOT_CLASSIFICATION,
    }


def write_state(
    *,
    run_id: str,
    raw_output_dir: Path,
    stage: str,
    decision_token: str,
    started_clock: float,
    last_request_id: str,
    parse_rows: Sequence[Mapping[str, str]],
    completed: int,
    notes: str,
) -> None:
    strict_valid = sum(1 for row in parse_rows if row.get("strict_valid") == "true")
    failed = completed - strict_valid
    residual = PILOT_REQUEST_COUNT - completed
    elapsed_hours = max((time.perf_counter() - started_clock) / 3600.0, 1e-9)
    throughput = completed / elapsed_hours if completed else 0.0
    eta = residual / throughput * 3600.0 if throughput and residual else 0.0
    duplicate_output = sum(int(row.get("duplicate_output_count") or 0) for row in parse_rows)
    out_of_range = sum(int(row.get("out_of_range_count") or 0) for row in parse_rows)
    malformed = sum(int(row.get("malformed_json_count") or 0) for row in parse_rows)
    ambiguous = sum(int(row.get("ambiguous_mapping_count") or 0) for row in parse_rows)
    extra = sum(int(row.get("extra_candidate_count") or 0) for row in parse_rows)
    rows = [
        {
            "run_id": run_id,
            "stage": stage,
            "decision_token": decision_token,
            "planned": str(PILOT_REQUEST_COUNT),
            "completed": str(completed),
            "strict_valid": str(strict_valid),
            "failed": str(failed),
            "residual": str(residual),
            "duplicate_output": str(duplicate_output),
            "out_of_range": str(out_of_range),
            "malformed_json": str(malformed),
            "ambiguous_mapping": str(ambiguous),
            "extra_candidate": str(extra),
            "percent_complete": format_float(completed / PILOT_REQUEST_COUNT if PILOT_REQUEST_COUNT else 0.0),
            "last_request_id": last_request_id,
            "last_timestamp": utc_now(),
            "throughput_requests_per_hour": format_float(throughput),
            "eta_seconds": format_float(eta),
            "raw_output_dir": f6l.rel(raw_output_dir),
            "notes": notes,
        }
    ]
    f6l.write_csv(PILOT_STATE_PATH, rows, STATE_FIELDS)


def existing_final_records(raw_output_dir: Path) -> dict[str, dict[str, Any]]:
    return f6l.final_response_records(raw_output_dir)


def execute_pilot_requests(
    build: PilotBuild,
    *,
    raw_output_dir: Path,
    endpoint: str,
    resume: bool,
) -> None:
    paths = f6l.raw_response_paths(raw_output_dir)
    paths["individual"].mkdir(parents=True, exist_ok=True)
    if not paths["command"].exists():
        paths["command"].write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    f6l.write_json(
        paths["manifest"],
        {
            **build.manifest,
            "raw_output_dir": f6l.rel(raw_output_dir),
            "started_at": utc_now(),
            "endpoint": endpoint,
            "resume": resume,
        },
    )
    sample_map = sample_rows_by_user()
    candidates_by_user = f6l.candidate_rows_for_operationalized_sample()
    finals = existing_final_records(raw_output_dir) if resume else {}
    started_clock = time.perf_counter()
    parse_rows: list[dict[str, str]] = []
    completed = 0
    for row in build.manifest_rows:
        if row["request_id"] in finals:
            completed += 1
            continue
        prompt = render_v3_prompt(sample_map[row["user_id"]], candidates_by_user[row["user_id"]], row["prompting_strategy"])
        request_record = {
            **parse_row_to_dict(row),
            "prompt": prompt,
            "endpoint": endpoint,
            "queued_at": utc_now(),
            "expected_raw_output_path": row["expected_raw_output_path"],
            "evidence_class": PILOT_CLASSIFICATION,
        }
        raw_json_path = paths["individual"] / f"{row['request_id']}.json"
        if raw_json_path.exists():
            raise RuntimeError(f"refusing to overwrite existing raw response without final JSONL resume record: {raw_json_path}")
        f6l.append_jsonl_fsync(paths["requests"], request_record)
        result = execute_local_ollama_json_request(row["model_name"], prompt, GENERATION_PARAMETERS, endpoint)
        parse = parse_v3_response(result.response_text, candidates_by_user[row["user_id"]])
        raw_record = raw_record_for_response(
            row=row,
            prompt=prompt,
            candidate_rows=candidates_by_user[row["user_id"]],
            result=result,
            parse=parse,
            raw_json_path=raw_json_path,
            attempt_index=1,
        )
        raw_json_path.write_text(json.dumps(raw_record, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        f6l.append_jsonl_fsync(paths["responses"], raw_record)
        completed += 1
        parse_rows.append(parse_result_row(row, raw_record, candidates_by_user[row["user_id"]]))
        write_state(
            run_id=build.run_id,
            raw_output_dir=raw_output_dir,
            stage="executing",
            decision_token=DECISION_FAILED,
            started_clock=started_clock,
            last_request_id=row["request_id"],
            parse_rows=parse_rows,
            completed=completed,
            notes="observable incremental state; final decision assigned during consolidation",
        )
        print(
            f"[{utc_now()}] completed {completed}/{PILOT_REQUEST_COUNT} "
            f"strict_valid={sum(1 for item in parse_rows if item['strict_valid'] == 'true')} "
            f"last={row['request_id']} failure={parse.failure_reason or 'none'}",
            flush=True,
        )


def parse_result_row(row: Mapping[str, str], raw_record: Mapping[str, Any], candidate_rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    parse = raw_record.get("parser_output", {})
    mapping_hash = candidate_index_mapping_hash(candidate_rows)
    hash_coherence = (
        raw_record.get("rendered_prompt_hash") == row["rendered_prompt_hash"]
        and raw_record.get("candidate_list_hash") == row["candidate_list_hash"]
        and raw_record.get("candidate_index_mapping_hash") == mapping_hash
        and raw_record.get("request_contract_hash") == row["request_contract_hash"]
    )
    return {
        "request_id": row["request_id"],
        "pilot_id": PILOT_ID,
        "user_id": row["user_id"],
        "model_name": row["model_name"],
        "model_digest": row["model_digest"],
        "prompting_strategy": row["prompting_strategy"],
        "parser_version": PARSER_VERSION,
        "parser_source_hash": str(raw_record.get("parser_source_hash", parser_source_hash())),
        "output_contract": OUTPUT_CONTRACT,
        "output_contract_hash": row["output_contract_hash"],
        "prompt_template_hash": row["prompt_template_hash"],
        "rendered_prompt_hash": row["rendered_prompt_hash"],
        "candidate_list_hash": row["candidate_list_hash"],
        "candidate_index_mapping_hash": mapping_hash,
        "mapping_hash": mapping_hash,
        "raw_response_sha256": str(raw_record.get("raw_response_sha256", "")),
        "strict_valid": "true" if parse.get("strict_valid") else "false",
        "failure_reason": str(parse.get("failure_reason", "")),
        "ranking_indices": json_list(parse.get("ranking_indices", [])),
        "ranked_item_ids": json_list(parse.get("ranked_item_ids", [])),
        "ranked_titles": json_list(parse.get("ranked_titles", [])),
        "output_count": str(parse.get("output_count", 0)),
        "duplicate_output_count": str(parse.get("duplicate_output_count", 0)),
        "out_of_range_count": str(parse.get("out_of_range_count", 0)),
        "malformed_json_count": str(parse.get("malformed_json_count", 0)),
        "ambiguous_mapping_count": str(parse.get("ambiguous_mapping_count", 0)),
        "extra_candidate_count": str(parse.get("extra_candidate_count", 0)),
        "request_contract_hash": row["request_contract_hash"],
        "hash_coherence": "true" if hash_coherence else "false",
        "evidence_class": PILOT_CLASSIFICATION,
        "notes": "pilot validity only; no fuzzy/relevance/positive metric used by parser",
    }


def execution_row(row: Mapping[str, str], raw_record: Mapping[str, Any]) -> dict[str, str]:
    backend = raw_record.get("raw_backend_response", {})
    return {
        "request_id": row["request_id"],
        "pilot_id": PILOT_ID,
        "user_id": row["user_id"],
        "model_name": row["model_name"],
        "model_digest": row["model_digest"],
        "prompting_strategy": row["prompting_strategy"],
        "attempt_index": str(raw_record.get("attempt_index", 1)),
        "started_at": str(raw_record.get("started_at", "")),
        "completed_at": str(raw_record.get("completed_at", "")),
        "duration_seconds": format_float(float(raw_record.get("duration_seconds", 0.0))),
        "backend_success": "true" if raw_record.get("backend_success") else "false",
        "response_nonempty": "true" if str(raw_record.get("raw_output_text", "")).strip() else "false",
        "done_reason": str(backend.get("done_reason", "")),
        "prompt_eval_count": str(backend.get("prompt_eval_count", "")),
        "eval_count": str(backend.get("eval_count", "")),
        "load_duration_nanoseconds": str(backend.get("load_duration", "")),
        "prompt_eval_duration_nanoseconds": str(backend.get("prompt_eval_duration", "")),
        "eval_duration_nanoseconds": str(backend.get("eval_duration", "")),
        "total_duration_nanoseconds": str(backend.get("total_duration", "")),
        "raw_response_sha256": str(raw_record.get("raw_response_sha256", "")),
        "backend_error_type": str(raw_record.get("backend_error_type", "")),
        "backend_error_message": str(raw_record.get("backend_error_message", "")),
        "response_jsonl_recorded": "true",
        "raw_json_path": str(raw_record.get("raw_json_path", "")),
        "execution_status": "completed" if raw_record.get("backend_success") else "backend_failed",
        "notes": "single local-only attempt; no scientific retry",
    }


def recommendation_rows_from_parse(row: Mapping[str, str], parse_row: Mapping[str, str]) -> list[dict[str, str]]:
    if parse_row["strict_valid"] != "true":
        return []
    indices = [int(item) for item in json.loads(parse_row["ranking_indices"])]
    item_ids = [str(item) for item in json.loads(parse_row["ranked_item_ids"])]
    titles = [str(item) for item in json.loads(parse_row["ranked_titles"])]
    return [
        {
            "request_id": row["request_id"],
            "pilot_id": PILOT_ID,
            "user_id": row["user_id"],
            "model_name": row["model_name"],
            "model_digest": row["model_digest"],
            "prompting_strategy": row["prompting_strategy"],
            "rank": str(rank),
            "candidate_index": str(index),
            "recommended_item_id": item_id,
            "recommended_item_title": title,
            "candidate_list_hash": row["candidate_list_hash"],
            "candidate_index_mapping_hash": parse_row["candidate_index_mapping_hash"],
            "parser_version": PARSER_VERSION,
            "evidence_class": PILOT_CLASSIFICATION,
            "notes": "index mapped deterministically to frozen candidate list",
        }
        for rank, (index, item_id, title) in enumerate(zip(indices, item_ids, titles), start=1)
    ]


def consolidate_outputs(build: PilotBuild, *, raw_output_dir: Path, preflight_rows: Sequence[Mapping[str, str]], profile_hashes_before: Mapping[str, str], original_raw_sha_before: str) -> tuple[str, dict[str, Any]]:
    raw_records = existing_final_records(raw_output_dir)
    candidates_by_user = f6l.candidate_rows_for_operationalized_sample()
    manifest_by_id = {row["request_id"]: row for row in build.manifest_rows}
    execution_rows: list[dict[str, str]] = []
    parse_rows: list[dict[str, str]] = []
    rec_rows: list[dict[str, str]] = []
    for request_id, row in manifest_by_id.items():
        raw_record = raw_records.get(request_id)
        if not raw_record:
            continue
        execution_rows.append(execution_row(row, raw_record))
        parse_row = parse_result_row(row, raw_record, candidates_by_user[row["user_id"]])
        parse_rows.append(parse_row)
        rec_rows.extend(recommendation_rows_from_parse(row, parse_row))

    completed = len(raw_records)
    strict_valid = sum(1 for row in parse_rows if row["strict_valid"] == "true")
    failed = completed - strict_valid
    residual = PILOT_REQUEST_COUNT - completed
    duplicate_output = sum(int(row["duplicate_output_count"] or 0) for row in parse_rows)
    out_of_range = sum(int(row["out_of_range_count"] or 0) for row in parse_rows)
    malformed_json = sum(int(row["malformed_json_count"] or 0) for row in parse_rows)
    ambiguous_mapping = sum(int(row["ambiguous_mapping_count"] or 0) for row in parse_rows)
    extra_candidate = sum(int(row["extra_candidate_count"] or 0) for row in parse_rows)
    hash_coherence_pass = bool(parse_rows) and all(row["hash_coherence"] == "true" for row in parse_rows)
    request_counts, response_counts = f6l.request_record_counts(raw_output_dir)
    duplicate_request_records = sum(max(0, count - 1) for count in response_counts.values())
    original_raw_sha_after = original_raw_hash_map_sha()
    profile_hashes_after = f6l.profile_raw_hashes()
    raw_original_unchanged = original_raw_sha_before == original_raw_sha_after == original_raw_expected_sha()
    profile_unchanged = dict(profile_hashes_before) == profile_hashes_after

    if (
        completed == PILOT_REQUEST_COUNT
        and strict_valid == PILOT_REQUEST_COUNT
        and failed == 0
        and residual == 0
        and duplicate_output == 0
        and out_of_range == 0
        and malformed_json == 0
        and ambiguous_mapping == 0
        and extra_candidate == 0
        and duplicate_request_records == 0
        and hash_coherence_pass
        and raw_original_unchanged
        and profile_unchanged
        and all(row["status"] == "pass" for row in preflight_rows)
    ):
        decision = DECISION_PASS
    else:
        decision = DECISION_FAILED

    validity_rows = build_validity_rows(build.run_id, parse_rows, completed)
    failure_rows = build_failure_rows(build.run_id, parse_rows)
    smoke_rows = build_smoke_metrics(build.run_id, parse_rows, build.manifest_rows)
    acceptance_rows = build_acceptance_rows(
        build.run_id,
        {
            "completed": completed,
            "strict_valid": strict_valid,
            "failed": failed,
            "residual": residual,
            "duplicate_output": duplicate_output,
            "out_of_range": out_of_range,
            "malformed_json": malformed_json,
            "ambiguous_mapping": ambiguous_mapping,
            "extra_candidate": extra_candidate,
            "duplicate_request_records": duplicate_request_records,
            "hash_coherence_pass": hash_coherence_pass,
            "original_raw_unchanged": raw_original_unchanged,
            "profile_raw_unchanged": profile_unchanged,
            "preflight_pass": all(row["status"] == "pass" for row in preflight_rows),
        },
    )

    f6l.write_csv(PILOT_EXECUTION_PATH, execution_rows, EXECUTION_FIELDS)
    f6l.write_csv(PILOT_PARSE_RESULTS_PATH, parse_rows, PARSE_FIELDS)
    f6l.write_csv(PILOT_RECOMMENDATIONS_PATH, rec_rows, RECOMMENDATION_FIELDS)
    f6l.write_csv(PILOT_VALIDITY_PATH, validity_rows, VALIDITY_FIELDS)
    f6l.write_csv(PILOT_FAILURE_TAXONOMY_PATH, failure_rows, FAILURE_FIELDS)
    f6l.write_csv(PILOT_ACCEPTANCE_PATH, acceptance_rows, ACCEPTANCE_FIELDS)
    f6l.write_csv(PILOT_SMOKE_METRICS_PATH, smoke_rows, SMOKE_FIELDS)
    if decision == DECISION_PASS:
        f6l.write_csv(FULL_RUN_PLAN_PATH, build_full_run_plan_rows(build.run_id), FULL_RUN_PLAN_FIELDS)
    write_state(
        run_id=build.run_id,
        raw_output_dir=raw_output_dir,
        stage="complete",
        decision_token=decision,
        started_clock=time.perf_counter(),
        last_request_id=max(raw_records) if raw_records else "",
        parse_rows=parse_rows,
        completed=completed,
        notes="terminal pilot consolidation",
    )
    output_hashes = {
        f6l.rel(path): f6l.safe_file_sha256(path)
        for path in [
            PILOT_MANIFEST_PATH,
            PILOT_SELECTION_PATH,
            PILOT_PREFLIGHT_PATH,
            PILOT_STATE_PATH,
            PILOT_EXECUTION_PATH,
            PILOT_PARSE_RESULTS_PATH,
            PILOT_RECOMMENDATIONS_PATH,
            PILOT_VALIDITY_PATH,
            PILOT_FAILURE_TAXONOMY_PATH,
            PILOT_ACCEPTANCE_PATH,
            PILOT_SMOKE_METRICS_PATH,
            FULL_RUN_PLAN_PATH,
        ]
        if path.exists()
    }
    manifest = {
        **build.manifest,
        "decision_token": decision,
        "raw_output_dir": f6l.rel(raw_output_dir),
        "completed": completed,
        "strict_valid": strict_valid,
        "failed": failed,
        "residual": residual,
        "acceptance": {row["criterion"]: row["status"] for row in acceptance_rows},
        "original_raw_hash_before": original_raw_sha_before,
        "original_raw_hash_after": original_raw_sha_after,
        "profile_raw_count_before": len(profile_hashes_before),
        "profile_raw_count_after": len(profile_hashes_after),
        "output_hashes": output_hashes,
    }
    f6l.write_json(PILOT_JSON_MANIFEST_PATH, manifest)
    write_report_and_notes(build, manifest, acceptance_rows, validity_rows, failure_rows, smoke_rows)
    return decision, manifest


def build_validity_rows(run_id: str, parse_rows: Sequence[Mapping[str, str]], completed: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    by_cell: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in parse_rows:
        by_cell[(row["model_name"], row["prompting_strategy"])].append(row)
    for model in MODEL_ORDER:
        for strategy in STRATEGY_ORDER:
            cell_rows = by_cell.get((model, strategy), [])
            strict = sum(1 for row in cell_rows if row["strict_valid"] == "true")
            cell_completed = len(cell_rows)
            rows.append(
                {
                    "pilot_id": PILOT_ID,
                    "run_id": run_id,
                    "model_name": model,
                    "prompting_strategy": strategy,
                    "planned": str(PILOT_USER_COUNT),
                    "completed": str(cell_completed),
                    "strict_valid": str(strict),
                    "failed": str(cell_completed - strict),
                    "strict_valid_rate": format_float(strict / cell_completed if cell_completed else 0.0),
                    "duplicate_output_count": str(sum(int(row["duplicate_output_count"] or 0) for row in cell_rows)),
                    "out_of_range_count": str(sum(int(row["out_of_range_count"] or 0) for row in cell_rows)),
                    "malformed_json_count": str(sum(int(row["malformed_json_count"] or 0) for row in cell_rows)),
                    "ambiguous_mapping_count": str(sum(int(row["ambiguous_mapping_count"] or 0) for row in cell_rows)),
                    "extra_candidate_count": str(sum(int(row["extra_candidate_count"] or 0) for row in cell_rows)),
                    "residual": str(PILOT_USER_COUNT - cell_completed),
                    "evidence_class": PILOT_CLASSIFICATION,
                    "notes": "validity diagnostic only; no scientific model comparison",
                }
            )
    rows.append(
        {
            "pilot_id": PILOT_ID,
            "run_id": run_id,
            "model_name": "__all__",
            "prompting_strategy": "__all__",
            "planned": str(PILOT_REQUEST_COUNT),
            "completed": str(completed),
            "strict_valid": str(sum(1 for row in parse_rows if row["strict_valid"] == "true")),
            "failed": str(completed - sum(1 for row in parse_rows if row["strict_valid"] == "true")),
            "strict_valid_rate": format_float(sum(1 for row in parse_rows if row["strict_valid"] == "true") / completed if completed else 0.0),
            "duplicate_output_count": str(sum(int(row["duplicate_output_count"] or 0) for row in parse_rows)),
            "out_of_range_count": str(sum(int(row["out_of_range_count"] or 0) for row in parse_rows)),
            "malformed_json_count": str(sum(int(row["malformed_json_count"] or 0) for row in parse_rows)),
            "ambiguous_mapping_count": str(sum(int(row["ambiguous_mapping_count"] or 0) for row in parse_rows)),
            "extra_candidate_count": str(sum(int(row["extra_candidate_count"] or 0) for row in parse_rows)),
            "residual": str(PILOT_REQUEST_COUNT - completed),
            "evidence_class": PILOT_CLASSIFICATION,
            "notes": "aggregate V3 strict-validity diagnostic",
        }
    )
    return rows


def build_failure_rows(run_id: str, parse_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    counts = Counter(row["failure_reason"] or "strict_valid" for row in parse_rows)
    if not counts:
        counts["no_completed_requests"] = 0
    return [
        {
            "pilot_id": PILOT_ID,
            "run_id": run_id,
            "failure_reason": reason,
            "count": str(count),
            "evidence_class": PILOT_CLASSIFICATION,
            "notes": "V3 parser failure taxonomy; strict_valid is not a failure",
        }
        for reason, count in sorted(counts.items())
    ]


def build_smoke_metrics(run_id: str, parse_rows: Sequence[Mapping[str, str]], manifest_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    manifest_by_id = {row["request_id"]: row for row in manifest_rows}
    candidates_by_user = f6l.candidate_rows_for_operationalized_sample()
    rows: list[dict[str, str]] = []
    by_cell: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in parse_rows:
        by_cell[(row["model_name"], row["prompting_strategy"])].append(row)
    for (model, strategy), cell_rows in sorted(by_cell.items()):
        hits: list[float] = []
        ndcgs: list[float] = []
        for row in cell_rows:
            if row["strict_valid"] != "true":
                continue
            manifest = manifest_by_id[row["request_id"]]
            positive_candidates = [candidate for candidate in candidates_by_user[row["user_id"]] if candidate.get("is_positive") == "true"]
            positive_item = positive_candidates[0]["candidate_item_id"] if positive_candidates else ""
            ranked = [str(item) for item in json.loads(row["ranked_item_ids"])]
            rank = ranked.index(positive_item) + 1 if positive_item in ranked else None
            hit = 1.0 if rank is not None and rank <= 5 else 0.0
            ndcg = 1.0 / math.log2(rank + 1) if rank is not None and rank <= 5 else 0.0
            hits.append(hit)
            ndcgs.append(ndcg)
        for metric, values in [("HR@5", hits), ("NDCG@5", ndcgs)]:
            rows.append(
                {
                    "pilot_id": PILOT_ID,
                    "run_id": run_id,
                    "model_name": model,
                    "prompting_strategy": strategy,
                    "metric": metric,
                    "value": format_float(sum(values) / len(values) if values else math.nan),
                    "evaluated_requests": str(len(values)),
                    "evidence_class": "pilot_only_not_scientific_evidence",
                    "notes": "technical smoke check only; do not update RQs or compare models on 10 users",
                }
            )
    return rows


def build_acceptance_rows(run_id: str, values: Mapping[str, Any]) -> list[dict[str, str]]:
    criteria = [
        ("completed", values["completed"], PILOT_REQUEST_COUNT),
        ("strict_valid_v3", values["strict_valid"], PILOT_REQUEST_COUNT),
        ("failed", values["failed"], 0),
        ("residual", values["residual"], 0),
        ("duplicate_output", values["duplicate_output"], 0),
        ("out_of_range", values["out_of_range"], 0),
        ("malformed_json", values["malformed_json"], 0),
        ("ambiguous_mapping", values["ambiguous_mapping"], 0),
        ("extra_candidate", values["extra_candidate"], 0),
        ("duplicate_request_records", values["duplicate_request_records"], 0),
        ("request_candidate_hash_coherence", "pass" if values["hash_coherence_pass"] else "fail", "pass"),
        ("original_raw_unchanged", "pass" if values["original_raw_unchanged"] else "fail", "pass"),
        ("profile_raw_unchanged", "pass" if values["profile_raw_unchanged"] else "fail", "pass"),
        ("preflight_pass", "pass" if values["preflight_pass"] else "fail", "pass"),
    ]
    rows = []
    for index, (criterion, observed, expected) in enumerate(criteria, start=1):
        rows.append(
            {
                "pilot_id": PILOT_ID,
                "run_id": run_id,
                "criterion": criterion,
                "observed": str(observed),
                "expected": str(expected),
                "status": "pass" if str(observed) == str(expected) else "fail",
                "evidence": f6l.rel(PILOT_PARSE_RESULTS_PATH) if criterion not in {"original_raw_unchanged", "profile_raw_unchanged", "preflight_pass"} else f6l.rel(PILOT_PREFLIGHT_PATH),
                "notes": "preregistered acceptance threshold; not relaxed after outputs",
            }
        )
    return rows


def build_full_run_plan_rows(pilot_run_id: str) -> list[dict[str, str]]:
    future_run_id = "YYYYMMDD_HHMMSS"
    command = " ".join(
        [
            "$env:PYTHONNOUSERSITE='1';",
            "$env:PYTHONDONTWRITEBYTECODE='1';",
            "$env:PYTHONPATH='.';",
            str(f6l.RECOVERED_INTERPRETER),
            "-B",
            "src/experiments/build_phase_f6l_lastfm_corrective_output_contract_pilot.py",
            "--mode",
            "execute-full-v3",
            "--run-id",
            future_run_id,
            "--allow-api-calls",
            "--local-only",
            "--confirm-full-v3-rerun",
        ]
    )
    return [
        {
            "plan_id": "F6L-LASTFM-V3-FULL-RUN-PLAN-001",
            "pilot_run_id": pilot_run_id,
            "future_run_id_placeholder": future_run_id,
            "status": "prepared_not_executed",
            "future_command": command,
            "request_count": "4000",
            "notes": "Future operation must rerun all 1000 users x 2 models x 2 strategies with V3 in a new raw dir; do not mix with V1/V2/original outputs.",
        }
    ]


def write_report_and_notes(
    build: PilotBuild,
    manifest: Mapping[str, Any],
    acceptance_rows: Sequence[Mapping[str, str]],
    validity_rows: Sequence[Mapping[str, str]],
    failure_rows: Sequence[Mapping[str, str]],
    smoke_rows: Sequence[Mapping[str, str]],
) -> None:
    acceptance_pass = sum(1 for row in acceptance_rows if row["status"] == "pass")
    acceptance_total = len(acceptance_rows)
    aggregate = next((row for row in validity_rows if row["model_name"] == "__all__"), {})
    report = "\n".join(
        [
            f"# {PILOT_ID}",
            "",
            f"- Decision token: `{manifest['decision_token']}`",
            f"- Run id: `{build.run_id}`",
            f"- Raw output dir: `{manifest['raw_output_dir']}`",
            f"- Scope: 40-request corrective local output-contract pilot only.",
            f"- Original 4000-output run remains separate as `prompt_parser_interface_fragility` evidence.",
            f"- V3 strict-valid: {aggregate.get('strict_valid', '0')}/{aggregate.get('completed', '0')}",
            f"- Acceptance checks: {acceptance_pass}/{acceptance_total} pass.",
            f"- Parser: `{PARSER_VERSION}` hash `{manifest['parser_source_hash']}`.",
            f"- Contract hash: `{manifest['output_contract_hash']}`.",
            f"- Full 4000-run command prepared: {'yes, not executed' if manifest['decision_token'] == DECISION_PASS else 'no'}",
            "",
            "## Output Contract",
            "",
            "The pilot prompts require exactly one JSON object of the form `{\"ranking\":[3,17,1,9,6,12,4,20,8,11]}`. The ranking list must contain exactly 10 distinct integers in the inclusive range 1-20. Titles, item ids, markdown, explanations, extra keys, and extra text are rejected.",
            "",
            "## Evidence Boundary",
            "",
            "Validity and contract diagnostics are pilot evidence. HR@5/NDCG@5 rows are technical smoke checks only and are marked `pilot_only_not_scientific_evidence`; they do not update research-question conclusions.",
            "",
            "## Files",
            "",
            f"- Manifest: `{f6l.rel(PILOT_MANIFEST_PATH)}`",
            f"- Preflight: `{f6l.rel(PILOT_PREFLIGHT_PATH)}`",
            f"- Parse results: `{f6l.rel(PILOT_PARSE_RESULTS_PATH)}`",
            f"- Acceptance: `{f6l.rel(PILOT_ACCEPTANCE_PATH)}`",
            f"- Smoke metrics: `{f6l.rel(PILOT_SMOKE_METRICS_PATH)}`",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    note = "\n".join(
        [
            "",
            f"## {PILOT_ID} ({utc_now()})",
            "",
            f"- Decision token: `{manifest['decision_token']}`",
            f"- Run id: `{build.run_id}`",
            f"- Raw output dir: `{manifest['raw_output_dir']}`",
            f"- Scope: corrective V3 JSON/index pilot, 40 local Ollama requests maximum.",
            f"- Acceptance: {acceptance_pass}/{acceptance_total} checks passed; strict-valid {aggregate.get('strict_valid', '0')}/{aggregate.get('completed', '0')}.",
            f"- Original raw unchanged: `{manifest['original_raw_hash_before']}` -> `{manifest['original_raw_hash_after']}`.",
            "- Scientific boundary: validity diagnostics only; smoke HR/NDCG is not RQ evidence.",
        ]
    )
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTES_PATH.open("a", encoding="utf-8") as handle:
        handle.write(note + "\n")
    HANDOFF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HANDOFF_PATH.open("a", encoding="utf-8") as handle:
        handle.write(note + "\n")


def run_pilot(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.mode == "execute-full-v3":
        raise SystemExit("Full 4000-request V3 execution is prepared but not authorized in this pilot-only turn.")
    if not args.allow_api_calls:
        raise SystemExit("--allow-api-calls is required before local Ollama inference")
    if not args.local_only:
        raise SystemExit("--local-only is required; remote APIs are not authorized")
    endpoint = str(args.ollama_endpoint)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_output_dir = Path(args.raw_output_dir) if args.raw_output_dir else local_raw_output_dir(run_id)
    if not raw_output_dir.is_absolute():
        raw_output_dir = f6l.PROJECT_ROOT / raw_output_dir
    build = build_pilot(run_id=run_id, raw_output_dir=raw_output_dir, write_outputs=True)
    profile_hashes_before = f6l.profile_raw_hashes()
    original_sha_before = original_raw_hash_map_sha()
    preflight_rows, test_result, ollama_state = validate_preflight(
        build,
        raw_output_dir=raw_output_dir,
        resume=args.resume,
        allow_api_calls=args.allow_api_calls,
        local_only=args.local_only,
        endpoint=endpoint,
        profile_hashes_before=profile_hashes_before,
        run_tests=True,
    )
    if not all(row["status"] == "pass" for row in preflight_rows):
        decision = DECISION_PREFLIGHT_BLOCKED
        write_state(
            run_id=build.run_id,
            raw_output_dir=raw_output_dir,
            stage="blocked_preflight",
            decision_token=decision,
            started_clock=time.perf_counter(),
            last_request_id="",
            parse_rows=[],
            completed=0,
            notes="preflight failed; no inference executed",
        )
        manifest = {
            **build.manifest,
            "decision_token": decision,
            "raw_output_dir": f6l.rel(raw_output_dir),
            "focused_tests": test_result,
            "preflight_failures": [row for row in preflight_rows if row["status"] != "pass"],
        }
        f6l.write_json(PILOT_JSON_MANIFEST_PATH, manifest)
        write_report_and_notes(build, manifest, build_acceptance_rows(build.run_id, {"completed": 0, "strict_valid": 0, "failed": 0, "residual": PILOT_REQUEST_COUNT, "duplicate_output": 0, "out_of_range": 0, "malformed_json": 0, "ambiguous_mapping": 0, "extra_candidate": 0, "duplicate_request_records": 0, "hash_coherence_pass": False, "original_raw_unchanged": original_sha_before == original_raw_expected_sha(), "profile_raw_unchanged": True, "preflight_pass": False}), [], [], [])
        return decision, manifest
    execute_pilot_requests(build, raw_output_dir=raw_output_dir, endpoint=endpoint, resume=args.resume)
    decision, manifest = consolidate_outputs(
        build,
        raw_output_dir=raw_output_dir,
        preflight_rows=preflight_rows,
        profile_hashes_before=profile_hashes_before,
        original_raw_sha_before=original_sha_before,
    )
    return decision, manifest


def build_only(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    run_id = args.run_id or "planned"
    raw_output_dir = Path(args.raw_output_dir) if args.raw_output_dir else local_raw_output_dir(run_id)
    if not raw_output_dir.is_absolute():
        raw_output_dir = f6l.PROJECT_ROOT / raw_output_dir
    build = build_pilot(run_id=run_id, raw_output_dir=raw_output_dir, write_outputs=True)
    preflight_rows, _, _ = validate_preflight(
        build,
        raw_output_dir=raw_output_dir,
        resume=args.resume,
        allow_api_calls=args.allow_api_calls,
        local_only=args.local_only,
        endpoint=str(args.ollama_endpoint),
        profile_hashes_before=f6l.profile_raw_hashes(),
        run_tests=False,
    )
    decision = "pilot_manifest_and_preflight_prepared_no_inference"
    manifest = {**build.manifest, "decision_token": decision, "preflight_status": "pass" if all(row["status"] == "pass" for row in preflight_rows) else "fail"}
    f6l.write_json(PILOT_JSON_MANIFEST_PATH, manifest)
    return decision, manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["plan", "execute", "execute-full-v3"], default="plan")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--raw-output-dir", default="")
    parser.add_argument("--allow-api-calls", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-full-v3-rerun", action="store_true")
    parser.add_argument("--ollama-endpoint", default=f6l.OLLAMA_LOCAL_GENERATE_URL)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "plan":
        decision, manifest = build_only(args)
    else:
        decision, manifest = run_pilot(args)
    print(json.dumps({"decision_token": decision, "run_id": manifest.get("run_id"), "raw_output_dir": manifest.get("raw_output_dir", "")}, sort_keys=True))
    return 0 if decision not in {DECISION_PREFLIGHT_BLOCKED, DECISION_FAILED} else 1


if __name__ == "__main__":
    raise SystemExit(main())
