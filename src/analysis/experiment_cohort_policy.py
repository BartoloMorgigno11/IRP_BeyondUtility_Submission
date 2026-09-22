from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

from src.analysis.item_title_map_reconciliation import (
    cleanup_title,
    decode_candidate_items,
    preserve_item_id,
)


CANDIDATE_ITEM_TITLE_MAP_COLUMNS = [
    "dataset",
    "item_id",
    "canonical_title",
    "title_status",
    "selected_source_path",
    "in_twenty_candidate_sets",
    "in_one_hundred_candidate_sets",
    "candidate_set_count",
    "candidate_occurrence_count",
    "positive_occurrence_count",
    "llm_usable",
    "notes",
]

EXPERIMENT_COHORT_MEMBERSHIP_COLUMNS = [
    "candidate_set_id",
    "source_path",
    "source_plan",
    "dataset",
    "task_type",
    "user_id",
    "positive_item_id",
    "candidate_count",
    "stored_positive_position",
    "computed_positive_position_zero_based",
    "row_validation_status",
    "positive_title_available",
    "all_candidate_titles_available",
    "missing_title_item_count",
    "missing_title_item_ids",
    "eligible_plan_e2_paper_like_full",
    "eligible_plan_e2_llm_strict_title",
    "eligible_plan_e2_matched_comparison",
    "eligible_legacy_reconstructed_100_candidate",
    "eligible_plan_d_diagnostic_20_candidate",
    "exclusion_reasons",
    "notes",
]

LLM_TITLE_EXCLUSIONS_COLUMNS = [
    "exclusion_id",
    "candidate_set_id",
    "source_path",
    "source_plan",
    "dataset",
    "task_type",
    "user_id",
    "positive_item_id",
    "candidate_count",
    "positive_title_missing",
    "missing_title_item_count",
    "missing_title_item_ids",
    "missing_positive_item_id",
    "exclusion_reason",
    "policy_action",
    "notes",
]

EXPERIMENT_COHORT_SUMMARY_COLUMNS = [
    "cohort_id",
    "dataset",
    "task_type",
    "candidate_set_id",
    "source_plan",
    "source_path",
    "total_source_rows",
    "eligible_rows",
    "excluded_rows",
    "eligibility_rate",
    "rows_missing_any_title",
    "rows_missing_positive_title",
    "unique_missing_title_items",
    "positive_position_min_zero_based",
    "positive_position_max_zero_based",
    "positive_position_mean_zero_based",
    "positive_position_distribution",
    "notes",
]

ALLOWED_COHORT_IDS = [
    "plan_e2_paper_like_full",
    "plan_e2_llm_strict_title",
    "plan_e2_matched_comparison",
    "legacy_reconstructed_100_candidate",
    "plan_d_diagnostic_20_candidate",
]

MEMBERSHIP_BOOLEAN_COLUMNS = [
    "positive_title_available",
    "all_candidate_titles_available",
    "eligible_plan_e2_paper_like_full",
    "eligible_plan_e2_llm_strict_title",
    "eligible_plan_e2_matched_comparison",
    "eligible_legacy_reconstructed_100_candidate",
    "eligible_plan_d_diagnostic_20_candidate",
]

COHORT_FIELD_BY_ID = {
    "plan_e2_paper_like_full": "eligible_plan_e2_paper_like_full",
    "plan_e2_llm_strict_title": "eligible_plan_e2_llm_strict_title",
    "plan_e2_matched_comparison": "eligible_plan_e2_matched_comparison",
    "legacy_reconstructed_100_candidate": "eligible_legacy_reconstructed_100_candidate",
    "plan_d_diagnostic_20_candidate": "eligible_plan_d_diagnostic_20_candidate",
}


def bool_str(value: bool) -> str:
    return "true" if value else "false"


def parse_bool(value: Any) -> bool:
    return str(value).strip().casefold() == "true"


def int_value(value: Any, default: int = 0) -> int:
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def normalize_source_plan(value: Any) -> str:
    text = str(value).strip().casefold().replace("-", " ")
    text = " ".join(text.split())
    if text in {"plan b", "b"}:
        return "plan_b"
    if text in {"plan c", "c"}:
        return "plan_c"
    if text in {"plan d", "d"}:
        return "plan_d"
    if text in {"plan e", "plan e2", "e", "e2"}:
        return "plan_e2"
    return "unknown"


def classify_task_type(
    candidate_set_id: Any,
    source_path: Any = "",
    source_plan: Any = "",
) -> str:
    plan = normalize_source_plan(source_plan)
    haystack = f"{candidate_set_id} {source_path}".casefold()
    if plan == "plan_d":
        if "reranking" in haystack:
            return "diagnostic_reranking"
        if "ranking" in haystack or "candidate_position_bias" in haystack or "diagnostic" in haystack:
            return "diagnostic_ranking"
        return "unknown"
    if plan == "plan_e2":
        if "reranking" in haystack:
            return "reranking"
        if "ranking" in haystack:
            return "ranking"
        return "unknown"
    if plan in {"plan_b", "plan_c"}:
        if "candidate" in haystack or "candidates" in haystack:
            return "legacy_ranking"
        return "unknown"
    return "unknown"


def is_structurally_valid(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("row_validation_status", "")).strip() in {"valid", "valid_with_warnings"}
        and int_value(row.get("positive_occurrence_count")) == 1
        and str(row.get("positive_in_candidates", "")).strip().casefold() == "true"
    )


def ground_truth_not_invalid(row: Mapping[str, Any]) -> bool:
    value = str(row.get("ground_truth_match", "")).strip().casefold()
    return value not in {"false", "invalid", "0"}


def is_title_usable(title_row: Mapping[str, Any] | None) -> bool:
    if not title_row:
        return False
    return (
        str(title_row.get("title_status", "")).startswith("resolved")
        and cleanup_title(title_row.get("canonical_title", "")) != ""
    )


def missing_title_item_ids(
    candidate_row: Mapping[str, Any],
    title_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[str]:
    dataset = str(candidate_row["dataset"])
    missing: list[str] = []
    seen: set[str] = set()
    for item_id in decode_candidate_items(candidate_row["candidate_items"]):
        key = (dataset, item_id)
        if item_id in seen:
            continue
        seen.add(item_id)
        if not is_title_usable(title_by_key.get(key)):
            missing.append(item_id)
    return missing


def positive_title_available(
    candidate_row: Mapping[str, Any],
    title_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> bool:
    key = (str(candidate_row["dataset"]), preserve_item_id(candidate_row["positive_item_id"]))
    return is_title_usable(title_by_key.get(key))


def eligible_plan_e2_paper_like_full(row: Mapping[str, Any]) -> bool:
    return (
        normalize_source_plan(row.get("source_plan")) == "plan_e2"
        and int_value(row.get("candidate_count")) == 20
        and is_structurally_valid(row)
        and ground_truth_not_invalid(row)
    )


def eligible_legacy_reconstructed_100_candidate(row: Mapping[str, Any]) -> bool:
    return (
        normalize_source_plan(row.get("source_plan")) in {"plan_b", "plan_c"}
        and int_value(row.get("candidate_count")) == 100
        and is_structurally_valid(row)
    )


def eligible_plan_d_diagnostic_20_candidate(row: Mapping[str, Any]) -> bool:
    return (
        normalize_source_plan(row.get("source_plan")) == "plan_d"
        and int_value(row.get("candidate_count")) == 20
        and is_structurally_valid(row)
    )


def calculate_membership(
    candidate_row: Mapping[str, Any],
    title_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    missing_ids = missing_title_item_ids(candidate_row, title_by_key)
    positive_available = positive_title_available(candidate_row, title_by_key)
    full = eligible_plan_e2_paper_like_full(candidate_row)
    strict = full and positive_available and not missing_ids
    matched = strict
    legacy = eligible_legacy_reconstructed_100_candidate(candidate_row)
    diagnostic = eligible_plan_d_diagnostic_20_candidate(candidate_row)
    task_type = classify_task_type(
        candidate_row.get("candidate_set_id", ""),
        candidate_row.get("source_path", ""),
        candidate_row.get("source_plan", ""),
    )
    return {
        "task_type": task_type,
        "positive_title_available": positive_available,
        "all_candidate_titles_available": not missing_ids,
        "missing_title_item_count": len(missing_ids),
        "missing_title_item_ids": missing_ids,
        "eligible_plan_e2_paper_like_full": full,
        "eligible_plan_e2_llm_strict_title": strict,
        "eligible_plan_e2_matched_comparison": matched,
        "eligible_legacy_reconstructed_100_candidate": legacy,
        "eligible_plan_d_diagnostic_20_candidate": diagnostic,
    }


def exclusion_reasons(
    candidate_row: Mapping[str, Any],
    membership: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    plan = normalize_source_plan(candidate_row.get("source_plan"))
    if plan != "plan_e2":
        reasons.append("not_plan_e2_source")
    if int_value(candidate_row.get("candidate_count")) != 20:
        reasons.append("candidate_count_not_20")
    if str(candidate_row.get("row_validation_status", "")).strip() not in {"valid", "valid_with_warnings"}:
        reasons.append("row_not_structurally_valid")
    if int_value(candidate_row.get("positive_occurrence_count")) != 1:
        reasons.append("positive_not_once")
    if str(candidate_row.get("positive_in_candidates", "")).strip().casefold() != "true":
        reasons.append("positive_not_in_candidates")
    if not ground_truth_not_invalid(candidate_row):
        reasons.append("ground_truth_invalid")
    if membership.get("eligible_plan_e2_paper_like_full"):
        if not membership.get("positive_title_available"):
            reasons.append("missing_positive_title")
        if int(membership.get("missing_title_item_count", 0)) > 0:
            reasons.append("missing_candidate_titles")
    return reasons


def membership_row(
    candidate_row: Mapping[str, Any],
    title_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    membership = calculate_membership(candidate_row, title_by_key)
    reasons = exclusion_reasons(candidate_row, membership)
    notes = membership_notes(membership)
    return {
        "candidate_set_id": candidate_row["candidate_set_id"],
        "source_path": candidate_row["source_path"],
        "source_plan": candidate_row["source_plan"],
        "dataset": candidate_row["dataset"],
        "task_type": membership["task_type"],
        "user_id": preserve_item_id(candidate_row["user_id"]),
        "positive_item_id": preserve_item_id(candidate_row["positive_item_id"]),
        "candidate_count": str(candidate_row["candidate_count"]),
        "stored_positive_position": str(candidate_row.get("stored_positive_position", "")),
        "computed_positive_position_zero_based": str(
            candidate_row.get("computed_positive_position_zero_based", "")
        ),
        "row_validation_status": candidate_row["row_validation_status"],
        "positive_title_available": bool_str(bool(membership["positive_title_available"])),
        "all_candidate_titles_available": bool_str(bool(membership["all_candidate_titles_available"])),
        "missing_title_item_count": membership["missing_title_item_count"],
        "missing_title_item_ids": json.dumps(membership["missing_title_item_ids"], ensure_ascii=False),
        "eligible_plan_e2_paper_like_full": bool_str(bool(membership["eligible_plan_e2_paper_like_full"])),
        "eligible_plan_e2_llm_strict_title": bool_str(
            bool(membership["eligible_plan_e2_llm_strict_title"])
        ),
        "eligible_plan_e2_matched_comparison": bool_str(
            bool(membership["eligible_plan_e2_matched_comparison"])
        ),
        "eligible_legacy_reconstructed_100_candidate": bool_str(
            bool(membership["eligible_legacy_reconstructed_100_candidate"])
        ),
        "eligible_plan_d_diagnostic_20_candidate": bool_str(
            bool(membership["eligible_plan_d_diagnostic_20_candidate"])
        ),
        "exclusion_reasons": ";".join(reasons),
        "notes": notes,
    }


def membership_notes(membership: Mapping[str, Any]) -> str:
    if membership.get("eligible_plan_e2_llm_strict_title"):
        return "eligible_plan_e2_full_strict_and_matched"
    if membership.get("eligible_plan_e2_paper_like_full"):
        return "eligible_plan_e2_full_only_due_title_gap"
    if membership.get("eligible_legacy_reconstructed_100_candidate"):
        return "legacy_100_candidate_cohort_only"
    if membership.get("eligible_plan_d_diagnostic_20_candidate"):
        return "plan_d_diagnostic_only"
    return "not_eligible_for_defined_cohorts"


def sort_key_for_id(value: Any) -> tuple[int, Any]:
    text = preserve_item_id(value)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def sort_membership_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in sorted(
            rows,
            key=lambda row: (
                row["candidate_set_id"],
                row["dataset"],
                sort_key_for_id(row["user_id"]),
            ),
        )
    ]


def aggregate_candidate_item_usage(
    candidate_rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    usage: dict[tuple[str, str], dict[str, Any]] = {}
    for row in candidate_rows:
        dataset = str(row["dataset"])
        candidate_set_id = str(row["candidate_set_id"])
        candidate_count = int_value(row["candidate_count"])
        for item_id in decode_candidate_items(row["candidate_items"]):
            key = (dataset, item_id)
            entry = usage.setdefault(
                key,
                {
                    "dataset": dataset,
                    "item_id": item_id,
                    "candidate_set_ids": set(),
                    "candidate_occurrence_count": 0,
                    "positive_occurrence_count": 0,
                    "in_twenty_candidate_sets": False,
                    "in_one_hundred_candidate_sets": False,
                },
            )
            entry["candidate_set_ids"].add(candidate_set_id)
            entry["candidate_occurrence_count"] += 1
            if candidate_count == 20:
                entry["in_twenty_candidate_sets"] = True
            if candidate_count == 100:
                entry["in_one_hundred_candidate_sets"] = True
        positive_id = preserve_item_id(row["positive_item_id"])
        if positive_id:
            key = (dataset, positive_id)
            entry = usage.setdefault(
                key,
                {
                    "dataset": dataset,
                    "item_id": positive_id,
                    "candidate_set_ids": set(),
                    "candidate_occurrence_count": 0,
                    "positive_occurrence_count": 0,
                    "in_twenty_candidate_sets": False,
                    "in_one_hundred_candidate_sets": False,
                },
            )
            entry["positive_occurrence_count"] += 1
    return usage


def candidate_only_title_map_rows(
    usage: Mapping[tuple[str, str], Mapping[str, Any]],
    title_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(usage, key=lambda item: (item[0], sort_key_for_id(item[1]))):
        entry = usage[key]
        title_row = title_by_key.get(key)
        if title_row is None:
            raise ValueError(f"candidate item missing from title map: {key}")
        usable = is_title_usable(title_row)
        rows.append(
            {
                "dataset": entry["dataset"],
                "item_id": entry["item_id"],
                "canonical_title": cleanup_title(title_row.get("canonical_title", "")),
                "title_status": title_row.get("title_status", ""),
                "selected_source_path": title_row.get("selected_source_path", ""),
                "in_twenty_candidate_sets": bool_str(bool(entry["in_twenty_candidate_sets"])),
                "in_one_hundred_candidate_sets": bool_str(bool(entry["in_one_hundred_candidate_sets"])),
                "candidate_set_count": len(entry["candidate_set_ids"]),
                "candidate_occurrence_count": entry["candidate_occurrence_count"],
                "positive_occurrence_count": entry["positive_occurrence_count"],
                "llm_usable": bool_str(usable),
                "notes": "resolved_for_llm" if usable else "unresolved_title_not_llm_usable",
            }
        )
    return rows


def title_exclusion_rows(membership_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for membership in sort_membership_rows(
        row
        for row in membership_rows
        if parse_bool(row["eligible_plan_e2_paper_like_full"])
        and not parse_bool(row["eligible_plan_e2_llm_strict_title"])
        and int_value(row["missing_title_item_count"]) > 0
    ):
        positive_missing = not parse_bool(membership["positive_title_available"])
        missing_positive_item_id = membership["positive_item_id"] if positive_missing else ""
        rows.append(
            {
                "exclusion_id": f"LTE-{len(rows) + 1:06d}",
                "candidate_set_id": membership["candidate_set_id"],
                "source_path": membership["source_path"],
                "source_plan": membership["source_plan"],
                "dataset": membership["dataset"],
                "task_type": membership["task_type"],
                "user_id": membership["user_id"],
                "positive_item_id": membership["positive_item_id"],
                "candidate_count": membership["candidate_count"],
                "positive_title_missing": bool_str(positive_missing),
                "missing_title_item_count": membership["missing_title_item_count"],
                "missing_title_item_ids": membership["missing_title_item_ids"],
                "missing_positive_item_id": missing_positive_item_id,
                "exclusion_reason": membership["exclusion_reasons"],
                "policy_action": "exclude_from_llm_and_matched_cohorts_keep_in_full_baseline_cohort",
                "notes": "candidate list and order preserved; row excluded only by title policy",
            }
        )
    return rows


def cohort_summary_rows(membership_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in membership_rows:
        grouped[str(row["candidate_set_id"])].append(row)

    rows: list[dict[str, Any]] = []
    for candidate_set_id in sorted(grouped):
        source_rows = grouped[candidate_set_id]
        first = source_rows[0]
        applicable_cohorts = applicable_cohorts_for_source(first)
        for cohort_id in applicable_cohorts:
            field = COHORT_FIELD_BY_ID[cohort_id]
            eligible_rows = [row for row in source_rows if parse_bool(row[field])]
            missing_items: set[str] = set()
            rows_missing_any_title = 0
            rows_missing_positive_title = 0
            for row in source_rows:
                missing_ids = json.loads(str(row["missing_title_item_ids"]))
                if missing_ids:
                    rows_missing_any_title += 1
                    missing_items.update(missing_ids)
                if not parse_bool(row["positive_title_available"]):
                    rows_missing_positive_title += 1
            positions = [
                int_value(row["computed_positive_position_zero_based"])
                for row in eligible_rows
                if str(row["computed_positive_position_zero_based"]).strip() != ""
            ]
            distribution = Counter(positions)
            total = len(source_rows)
            eligible = len(eligible_rows)
            rows.append(
                {
                    "cohort_id": cohort_id,
                    "dataset": first["dataset"],
                    "task_type": first["task_type"],
                    "candidate_set_id": candidate_set_id,
                    "source_plan": first["source_plan"],
                    "source_path": first["source_path"],
                    "total_source_rows": total,
                    "eligible_rows": eligible,
                    "excluded_rows": total - eligible,
                    "eligibility_rate": safe_divide(eligible, total),
                    "rows_missing_any_title": rows_missing_any_title,
                    "rows_missing_positive_title": rows_missing_positive_title,
                    "unique_missing_title_items": len(missing_items),
                    "positive_position_min_zero_based": min(positions) if positions else "",
                    "positive_position_max_zero_based": max(positions) if positions else "",
                    "positive_position_mean_zero_based": safe_divide(sum(positions), len(positions))
                    if positions
                    else "",
                    "positive_position_distribution": json.dumps(
                        {str(key): distribution[key] for key in sorted(distribution)},
                        sort_keys=True,
                    ),
                    "notes": summary_note(cohort_id),
                }
            )
    order = {cohort_id: index for index, cohort_id in enumerate(ALLOWED_COHORT_IDS)}
    return sorted(
        rows,
        key=lambda row: (
            order[row["cohort_id"]],
            row["dataset"],
            row["task_type"],
            row["candidate_set_id"],
        ),
    )


def applicable_cohorts_for_source(row: Mapping[str, Any]) -> list[str]:
    plan = normalize_source_plan(row.get("source_plan"))
    candidate_count = int_value(row.get("candidate_count"))
    if plan == "plan_e2" and candidate_count == 20:
        return [
            "plan_e2_paper_like_full",
            "plan_e2_llm_strict_title",
            "plan_e2_matched_comparison",
        ]
    if plan in {"plan_b", "plan_c"} and candidate_count == 100:
        return ["legacy_reconstructed_100_candidate"]
    if plan == "plan_d" and candidate_count == 20:
        return ["plan_d_diagnostic_20_candidate"]
    return []


def summary_note(cohort_id: str) -> str:
    if cohort_id == "plan_e2_paper_like_full":
        return "baseline full cohort; titles not required"
    if cohort_id == "plan_e2_llm_strict_title":
        return "LLM-approved cohort; all candidate and positive titles resolved"
    if cohort_id == "plan_e2_matched_comparison":
        return "traditional baseline comparison cohort matched exactly to strict-title LLM cohort"
    if cohort_id == "legacy_reconstructed_100_candidate":
        return "legacy reconstructed one-hundred-candidate cohort; do not mix with Plan E2"
    if cohort_id == "plan_d_diagnostic_20_candidate":
        return "diagnostic cohort only; excluded from main paper-like results"
    return ""


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
