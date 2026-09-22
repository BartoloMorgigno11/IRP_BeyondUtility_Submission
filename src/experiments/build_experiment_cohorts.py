from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.experiment_cohort_policy import (
    ALLOWED_COHORT_IDS,
    CANDIDATE_ITEM_TITLE_MAP_COLUMNS,
    EXPERIMENT_COHORT_MEMBERSHIP_COLUMNS,
    EXPERIMENT_COHORT_SUMMARY_COLUMNS,
    LLM_TITLE_EXCLUSIONS_COLUMNS,
    MEMBERSHIP_BOOLEAN_COLUMNS,
    aggregate_candidate_item_usage,
    bool_str,
    candidate_only_title_map_rows,
    classify_task_type,
    cohort_summary_rows,
    int_value,
    parse_bool,
    sort_membership_rows,
    title_exclusion_rows,
)
from src.analysis.item_title_map_reconciliation import (
    ITEM_TITLE_COVERAGE_COLUMNS,
    ITEM_TITLE_MAP_COLUMNS,
    PHASE_2A_CANDIDATE_COLUMNS,
    cleanup_title,
    decode_candidate_items,
    preserve_item_id,
)
from src.audit.common_io import (
    read_csv_rows as _read_csv_rows,
    write_csv_rows as _write_csv_rows,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_CANDIDATE_SETS = PROJECT_ROOT / "results" / "tables" / "ranking_candidate_sets.csv"
DEFAULT_POSITIONS = PROJECT_ROOT / "results" / "tables" / "positive_positions.csv"
DEFAULT_DATASET_STATS = PROJECT_ROOT / "results" / "tables" / "dataset_statistics.csv"
DEFAULT_ITEM_TITLE_MAP = PROJECT_ROOT / "results" / "tables" / "item_title_map.csv"
DEFAULT_ITEM_TITLE_ISSUES = PROJECT_ROOT / "results" / "tables" / "item_title_mapping_issues.csv"
DEFAULT_ITEM_TITLE_COVERAGE = PROJECT_ROOT / "results" / "tables" / "item_title_coverage_summary.csv"

DEFAULT_CANDIDATE_TITLE_MAP_OUTPUT = PROJECT_ROOT / "results" / "tables" / "candidate_item_title_map.csv"
DEFAULT_MEMBERSHIP_OUTPUT = PROJECT_ROOT / "results" / "tables" / "experiment_cohort_membership.csv"
DEFAULT_EXCLUSIONS_OUTPUT = PROJECT_ROOT / "results" / "tables" / "llm_title_exclusions.csv"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "results" / "tables" / "experiment_cohort_summary.csv"
DEFAULT_REPORT_OUTPUT = PROJECT_ROOT / "report" / "experiment_cohort_policy.md"

INPUT_PATHS = [
    DEFAULT_CANDIDATE_SETS,
    DEFAULT_POSITIONS,
    DEFAULT_DATASET_STATS,
    DEFAULT_ITEM_TITLE_MAP,
    DEFAULT_ITEM_TITLE_ISSUES,
    DEFAULT_ITEM_TITLE_COVERAGE,
]

POSITIVE_POSITION_COLUMNS = [
    "candidate_set_id",
    "source_path",
    "source_plan",
    "dataset",
    "user_id",
    "positive_item_id",
    "candidate_count",
    "stored_positive_position",
    "stored_position_base",
    "computed_positive_position_zero_based",
    "computed_positive_position_one_based",
    "position_matches_stored",
    "position_validation_status",
    "notes",
]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    candidate_path = resolve_path(args.candidate_sets)
    input_hashes = {display_path(path): sha256_file(path) for path in INPUT_PATHS}
    input_sizes = {display_path(path): path.stat().st_size for path in INPUT_PATHS}

    candidate_rows = read_csv_rows(candidate_path)
    positions_rows = read_csv_rows(DEFAULT_POSITIONS)
    coverage_rows = read_csv_rows(DEFAULT_ITEM_TITLE_COVERAGE)
    input_verification = verify_inputs(candidate_rows, positions_rows, coverage_rows)

    usage = aggregate_candidate_item_usage(candidate_rows)
    title_by_key = load_candidate_title_rows(DEFAULT_ITEM_TITLE_MAP, set(usage))
    if set(title_by_key) != set(usage):
        missing = sorted(set(usage).difference(title_by_key))[:5]
        raise ValueError(f"candidate items missing from item_title_map.csv: {missing}")

    candidate_title_rows = candidate_only_title_map_rows(usage, title_by_key)
    membership_rows = sort_membership_rows(
        membership_row_with_titles(row, title_by_key) for row in candidate_rows
    )
    exclusion_rows = title_exclusion_rows(membership_rows)
    summary_rows = cohort_summary_rows(membership_rows)

    validation_messages = validate_outputs(
        candidate_rows=candidate_rows,
        candidate_title_rows=candidate_title_rows,
        membership_rows=membership_rows,
        exclusion_rows=exclusion_rows,
        summary_rows=summary_rows,
    )

    write_csv(DEFAULT_CANDIDATE_TITLE_MAP_OUTPUT, CANDIDATE_ITEM_TITLE_MAP_COLUMNS, candidate_title_rows)
    write_csv(DEFAULT_MEMBERSHIP_OUTPUT, EXPERIMENT_COHORT_MEMBERSHIP_COLUMNS, membership_rows)
    write_csv(DEFAULT_EXCLUSIONS_OUTPUT, LLM_TITLE_EXCLUSIONS_COLUMNS, exclusion_rows)
    write_csv(DEFAULT_SUMMARY_OUTPUT, EXPERIMENT_COHORT_SUMMARY_COLUMNS, summary_rows)
    write_report(
        report_path=DEFAULT_REPORT_OUTPUT,
        input_hashes=input_hashes,
        input_sizes=input_sizes,
        input_verification=input_verification,
        candidate_title_rows=candidate_title_rows,
        membership_rows=membership_rows,
        exclusion_rows=exclusion_rows,
        summary_rows=summary_rows,
        validation_messages=validation_messages,
    )

    print(f"candidate rows: {len(candidate_rows)}")
    print(f"candidate-only title rows: {len(candidate_title_rows)}")
    print(f"membership rows: {len(membership_rows)}")
    print(f"title exclusions: {len(exclusion_rows)}")
    print(f"summary rows: {len(summary_rows)}")
    for cohort_id in ALLOWED_COHORT_IDS:
        eligible = sum(
            int(row["eligible_rows"])
            for row in summary_rows
            if row["cohort_id"] == cohort_id
        )
        print(f"{cohort_id}: {eligible}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase 2C experiment cohorts and title-exclusion policy artifacts."
    )
    parser.add_argument(
        "--candidate-sets",
        default=str(DEFAULT_CANDIDATE_SETS),
        help="Path to Phase 2A ranking candidate sets.",
    )
    return parser.parse_args(argv)


def membership_row_with_titles(
    candidate_row: Mapping[str, Any],
    title_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    from src.analysis.experiment_cohort_policy import membership_row

    return membership_row(candidate_row, title_by_key)


def verify_inputs(
    candidate_rows: list[dict[str, str]],
    positions_rows: list[dict[str, str]],
    coverage_rows: list[dict[str, str]],
) -> dict[str, Any]:
    if not candidate_rows:
        raise ValueError("ranking_candidate_sets.csv is empty")
    if list(candidate_rows[0].keys()) != PHASE_2A_CANDIDATE_COLUMNS:
        raise ValueError("ranking_candidate_sets.csv schema does not match Phase 2A")
    if positions_rows and list(positions_rows[0].keys()) != POSITIVE_POSITION_COLUMNS:
        raise ValueError("positive_positions.csv schema does not match Phase 2A")
    if coverage_rows and list(coverage_rows[0].keys()) != ITEM_TITLE_COVERAGE_COLUMNS:
        raise ValueError("item_title_coverage_summary.csv schema does not match Phase 2B")

    candidate_set_ids: set[str] = set()
    candidate_count_distribution: Counter[int] = Counter()
    row_status_distribution: Counter[str] = Counter()
    candidate_keys: set[tuple[str, str]] = set()
    membership_keys: set[tuple[str, str]] = set()
    for row in candidate_rows:
        candidate_set_ids.add(row["candidate_set_id"])
        membership_key = (row["candidate_set_id"], preserve_item_id(row["user_id"]))
        if membership_key in membership_keys:
            raise ValueError(f"duplicate candidate_set_id/user_id row: {membership_key}")
        membership_keys.add(membership_key)
        candidate_items = decode_candidate_items(row["candidate_items"])
        if len(candidate_items) != int_value(row["candidate_count"]):
            raise ValueError("candidate_count does not match decoded candidate_items")
        candidate_count_distribution[len(candidate_items)] += 1
        row_status_distribution[row["row_validation_status"]] += 1
        for item_id in candidate_items:
            candidate_keys.add((row["dataset"], item_id))

    if len(candidate_rows) != 76121:
        raise ValueError(f"expected 76121 canonical candidate rows; found {len(candidate_rows)}")
    if len(candidate_set_ids) != 14:
        raise ValueError(f"expected 14 candidate-set IDs; found {len(candidate_set_ids)}")
    if len(positions_rows) != len(candidate_rows):
        raise ValueError("positive_positions.csv row count does not match candidate rows")

    position_keys = {
        (row["candidate_set_id"], preserve_item_id(row["user_id"]))
        for row in positions_rows
    }
    if position_keys != membership_keys:
        raise ValueError("positive_positions.csv keys do not match candidate rows")

    return {
        "candidate_rows": len(candidate_rows),
        "candidate_set_ids": len(candidate_set_ids),
        "candidate_count_distribution": dict(sorted(candidate_count_distribution.items())),
        "row_status_distribution": dict(sorted(row_status_distribution.items())),
        "unique_candidate_items": len(candidate_keys),
        "positive_position_rows": len(positions_rows),
        "coverage_rows": len(coverage_rows),
    }


def load_candidate_title_rows(
    item_title_path: Path,
    candidate_keys: set[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    title_by_key: dict[tuple[str, str], dict[str, str]] = {}
    with item_title_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ITEM_TITLE_MAP_COLUMNS:
            raise ValueError("item_title_map.csv schema does not match Phase 2B")
        for row in reader:
            key = (row["dataset"], preserve_item_id(row["item_id"]))
            if key in candidate_keys:
                title_by_key[key] = row
    return title_by_key


def validate_outputs(
    *,
    candidate_rows: list[dict[str, str]],
    candidate_title_rows: list[dict[str, Any]],
    membership_rows: list[dict[str, Any]],
    exclusion_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> list[str]:
    messages: list[str] = []

    candidate_keys: set[tuple[str, str]] = set()
    for row in candidate_rows:
        for item_id in decode_candidate_items(row["candidate_items"]):
            candidate_keys.add((row["dataset"], item_id))

    title_keys = [(row["dataset"], preserve_item_id(row["item_id"])) for row in candidate_title_rows]
    if len(title_keys) != len(set(title_keys)):
        raise ValueError("candidate_item_title_map.csv contains duplicate item keys")
    if set(title_keys) != candidate_keys:
        raise ValueError("candidate_item_title_map.csv keys do not equal candidate items")
    for row in candidate_title_rows:
        if row["canonical_title"] in {"Unknown", "Missing title"}:
            raise ValueError("placeholder title found in candidate_item_title_map.csv")
        llm_usable = parse_bool(row["llm_usable"])
        expected = str(row["title_status"]).startswith("resolved") and cleanup_title(row["canonical_title"]) != ""
        if llm_usable != expected:
            raise ValueError("llm_usable does not agree with title status and title value")
        if not llm_usable and row["canonical_title"]:
            raise ValueError("unusable title row has non-empty canonical_title")

    if len(membership_rows) != len(candidate_rows):
        raise ValueError("experiment_cohort_membership.csv row count mismatch")
    membership_keys = [(row["candidate_set_id"], row["user_id"]) for row in membership_rows]
    if len(membership_keys) != len(set(membership_keys)):
        raise ValueError("membership rows contain duplicate candidate_set_id/user_id keys")
    for row in membership_rows:
        json.loads(str(row["missing_title_item_ids"]))
        for column in MEMBERSHIP_BOOLEAN_COLUMNS:
            if str(row[column]) not in {"true", "false"}:
                raise ValueError(f"invalid boolean value in {column}: {row[column]}")
        if parse_bool(row["eligible_plan_e2_llm_strict_title"]) and not parse_bool(
            row["eligible_plan_e2_paper_like_full"]
        ):
            raise ValueError("strict-title row is not full-cohort eligible")
        if parse_bool(row["eligible_plan_e2_llm_strict_title"]) != parse_bool(
            row["eligible_plan_e2_matched_comparison"]
        ):
            raise ValueError("matched-comparison membership differs from strict-title membership")
        if row["source_plan"] == "Plan D" and (
            parse_bool(row["eligible_plan_e2_paper_like_full"])
            or parse_bool(row["eligible_plan_e2_llm_strict_title"])
            or parse_bool(row["eligible_plan_e2_matched_comparison"])
        ):
            raise ValueError("Plan D row entered a Plan E2 cohort")
        if int_value(row["candidate_count"]) == 100 and (
            parse_bool(row["eligible_plan_e2_paper_like_full"])
            or parse_bool(row["eligible_plan_e2_llm_strict_title"])
            or parse_bool(row["eligible_plan_e2_matched_comparison"])
        ):
            raise ValueError("100-candidate row entered a Plan E2 cohort")

    exclusion_ids = [row["exclusion_id"] for row in exclusion_rows]
    if len(exclusion_ids) != len(set(exclusion_ids)):
        raise ValueError("llm_title_exclusions.csv contains duplicate exclusion IDs")
    membership_by_key = {
        (row["candidate_set_id"], row["user_id"]): row
        for row in membership_rows
    }
    expected_exclusions = {
        (row["candidate_set_id"], row["user_id"])
        for row in membership_rows
        if parse_bool(row["eligible_plan_e2_paper_like_full"])
        and not parse_bool(row["eligible_plan_e2_llm_strict_title"])
        and int_value(row["missing_title_item_count"]) > 0
    }
    actual_exclusions = set()
    for row in exclusion_rows:
        key = (row["candidate_set_id"], row["user_id"])
        actual_exclusions.add(key)
        membership = membership_by_key[key]
        if row["source_plan"] != "Plan E2":
            raise ValueError("non-Plan E2 row found in llm_title_exclusions.csv")
        if not parse_bool(membership["eligible_plan_e2_paper_like_full"]):
            raise ValueError("title exclusion row is not full-cohort eligible")
        if parse_bool(membership["eligible_plan_e2_llm_strict_title"]) or parse_bool(
            membership["eligible_plan_e2_matched_comparison"]
        ):
            raise ValueError("title exclusion row is strict or matched eligible")
        if int_value(row["missing_title_item_count"]) <= 0:
            raise ValueError("title exclusion row has no missing title")
        json.loads(str(row["missing_title_item_ids"]))
    if expected_exclusions != actual_exclusions:
        raise ValueError("llm_title_exclusions.csv does not match strict-title exclusions")

    summary_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    strict_counts: dict[str, int] = {}
    matched_counts: dict[str, int] = {}
    for row in summary_rows:
        if row["cohort_id"] not in ALLOWED_COHORT_IDS:
            raise ValueError(f"invalid cohort_id: {row['cohort_id']}")
        key = (row["cohort_id"], row["candidate_set_id"])
        summary_by_key[key] = row
        rate = float(row["eligibility_rate"])
        if rate < 0 or rate > 1:
            raise ValueError("eligibility rate outside [0, 1]")
        total = int(row["total_source_rows"])
        eligible = int(row["eligible_rows"])
        excluded = int(row["excluded_rows"])
        if eligible + excluded != total:
            raise ValueError("summary eligible/excluded counts do not sum to total")
        distribution = json.loads(str(row["positive_position_distribution"]))
        if sum(int(value) for value in distribution.values()) != eligible:
            raise ValueError("positive-position distribution does not sum to eligible rows")
        if row["cohort_id"] == "plan_e2_llm_strict_title":
            strict_counts[row["candidate_set_id"]] = eligible
        if row["cohort_id"] == "plan_e2_matched_comparison":
            matched_counts[row["candidate_set_id"]] = eligible
    if strict_counts != matched_counts:
        raise ValueError("matched summary counts differ from strict-title summary counts")

    for candidate_set_id, source_rows in group_by(membership_rows, "candidate_set_id").items():
        for cohort_id in ALLOWED_COHORT_IDS:
            field = {
                "plan_e2_paper_like_full": "eligible_plan_e2_paper_like_full",
                "plan_e2_llm_strict_title": "eligible_plan_e2_llm_strict_title",
                "plan_e2_matched_comparison": "eligible_plan_e2_matched_comparison",
                "legacy_reconstructed_100_candidate": "eligible_legacy_reconstructed_100_candidate",
                "plan_d_diagnostic_20_candidate": "eligible_plan_d_diagnostic_20_candidate",
            }[cohort_id]
            eligible = sum(1 for row in source_rows if parse_bool(row[field]))
            summary = summary_by_key.get((cohort_id, candidate_set_id))
            if summary is None:
                if eligible != 0:
                    raise ValueError(f"missing summary row for {cohort_id}/{candidate_set_id}")
            elif int(summary["eligible_rows"]) != eligible:
                raise ValueError(f"summary count mismatch for {cohort_id}/{candidate_set_id}")

    messages.append(f"validated {len(candidate_title_rows)} candidate_item_title_map rows")
    messages.append(f"validated {len(membership_rows)} experiment_cohort_membership rows")
    messages.append(f"validated {len(exclusion_rows)} llm_title_exclusions rows")
    messages.append(f"validated {len(summary_rows)} experiment_cohort_summary rows")
    return messages


def write_report(
    *,
    report_path: Path,
    input_hashes: dict[str, str],
    input_sizes: dict[str, int],
    input_verification: dict[str, Any],
    candidate_title_rows: list[dict[str, Any]],
    membership_rows: list[dict[str, Any]],
    exclusion_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    validation_messages: list[str],
) -> None:
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    task_table = task_classification_table(membership_rows)
    cohort_totals = cohort_totals_from_summary(summary_rows)
    exclusion_by_dataset = count_by(exclusion_rows, "dataset")
    positive_exclusion_by_dataset = Counter(
        row["dataset"] for row in exclusion_rows if row["positive_title_missing"] == "true"
    )
    missing_item_counts = unique_missing_items_by_dataset(exclusion_rows)
    plan_e_position_rows = plan_e_position_comparison(summary_rows)

    lines: list[str] = []
    lines.append("# Experiment Cohort and Title-Exclusion Policy")
    lines.append("")
    lines.append("## 1. Scope and provenance")
    lines.append("")
    lines.append(f"- Audit timestamp: {now}")
    lines.append("- Phase: 2C experiment cohort definition and title-gap exclusion policy")
    lines.append("- Python executable: `C:\\ProgramData\\anaconda3\\python.exe` with `PYTHONNOUSERSITE=1`")
    lines.append("- No recommender, baseline, LLM, training, scoring, or metric experiment was run.")
    lines.append("")
    lines.append("## 2. Executive summary")
    lines.append("")
    lines.append(f"- Candidate-only title map rows: {len(candidate_title_rows)}.")
    lines.append(f"- Cohort membership rows: {len(membership_rows)}.")
    lines.append(f"- LLM title exclusions: {len(exclusion_rows)}.")
    lines.append(f"- Plan E2 full baseline eligible rows: {cohort_totals['plan_e2_paper_like_full']}.")
    lines.append(f"- Plan E2 strict-title LLM rows: {cohort_totals['plan_e2_llm_strict_title']}.")
    lines.append(f"- Plan E2 matched comparison rows: {cohort_totals['plan_e2_matched_comparison']}.")
    lines.append(f"- Legacy one-hundred-candidate rows: {cohort_totals['legacy_reconstructed_100_candidate']}.")
    lines.append(f"- Plan D diagnostic rows: {cohort_totals['plan_d_diagnostic_20_candidate']}.")
    lines.append("")
    lines.append("## 3. Why unresolved titles are not repaired")
    lines.append("")
    lines.append("- Fabricating titles, substituting ASINs, using descriptions/categories, deleting candidates, replacing candidates, or changing order would alter the experimental condition.")
    lines.append("- Traditional baselines do not require text titles, so structurally valid full-cohort rows can remain available for baseline-only reporting.")
    lines.append("- LLM prompts require resolved human-readable titles; unresolved title rows are therefore excluded from strict-title and matched cohorts rather than repaired.")
    lines.append("")
    lines.append("## 4. Input integrity verification")
    lines.append("")
    lines.extend(markdown_kv_table(input_verification))
    lines.append("")
    lines.append("Input hashes:")
    for path, digest in input_hashes.items():
        lines.append(f"- `{path}`: `{digest}`")
    lines.append("")
    lines.append("## 5. Candidate-set families")
    lines.append("")
    lines.extend(markdown_table(
        ["source_plan", "candidate_sets", "rows"],
        source_plan_family_rows(membership_rows),
    ))
    lines.append("")
    lines.append("## 6. Task-type classification")
    lines.append("")
    lines.extend(markdown_table(
        ["candidate_set_id", "dataset", "source_plan", "task_type", "rows"],
        task_table,
    ))
    lines.append("")
    lines.append("## 7. Authoritative cohort definitions")
    lines.append("")
    lines.append("- `plan_e2_paper_like_full`: Plan E2, 20 candidates, structurally valid, positive present exactly once, ground truth not invalid; titles not required.")
    lines.append("- `plan_e2_llm_strict_title`: full Plan E2 eligibility plus resolved positive and candidate titles.")
    lines.append("- `plan_e2_matched_comparison`: exactly identical membership to strict-title LLM cohort.")
    lines.append("- `legacy_reconstructed_100_candidate`: Plan B or Plan C 100-candidate structurally valid rows.")
    lines.append("- `plan_d_diagnostic_20_candidate`: Plan D 20-candidate structurally valid rows, diagnostic only.")
    lines.append("")
    lines.append("## 8. Plan E2 full baseline cohort")
    lines.append("")
    lines.extend(summary_lines_for_cohort(summary_rows, "plan_e2_paper_like_full"))
    lines.append("")
    lines.append("## 9. Plan E2 strict-title LLM cohort")
    lines.append("")
    lines.extend(summary_lines_for_cohort(summary_rows, "plan_e2_llm_strict_title"))
    lines.append("")
    lines.append("## 10. Plan E2 matched comparison cohort")
    lines.append("")
    lines.extend(summary_lines_for_cohort(summary_rows, "plan_e2_matched_comparison"))
    lines.append("")
    lines.append("## 11. Legacy one-hundred-candidate cohort")
    lines.append("")
    lines.extend(summary_lines_for_cohort(summary_rows, "legacy_reconstructed_100_candidate"))
    lines.append("")
    lines.append("## 12. Plan D diagnostic cohort")
    lines.append("")
    lines.extend(summary_lines_for_cohort(summary_rows, "plan_d_diagnostic_20_candidate"))
    lines.append("")
    lines.append("## 13. Beauty exclusions")
    lines.append("")
    lines.append(f"- Beauty title exclusions: {exclusion_by_dataset.get('Amazon Beauty', 0)} rows.")
    lines.append(f"- Unique missing Beauty title items among exclusions: {missing_item_counts.get('Amazon Beauty', 0)}.")
    lines.append("")
    lines.append("## 14. Sports exclusions")
    lines.append("")
    lines.append(f"- Sports title exclusions: {exclusion_by_dataset.get('Amazon Sports', 0)} rows.")
    lines.append(f"- Unique missing Sports title items among exclusions: {missing_item_counts.get('Amazon Sports', 0)}.")
    lines.append("")
    lines.append("## 15. ML-1M exclusions")
    lines.append("")
    lines.append(f"- ML-1M title exclusions: {exclusion_by_dataset.get('ML-1M', 0)} rows.")
    lines.append("- ML-1M Plan E2 rows have complete candidate and positive title coverage.")
    lines.append("")
    lines.append("## 16. Positive-title exclusions")
    lines.append("")
    for dataset in ["Amazon Beauty", "Amazon Sports", "ML-1M"]:
        lines.append(f"- {dataset}: {positive_exclusion_by_dataset.get(dataset, 0)} rows have unresolved positive titles.")
    lines.append("")
    lines.append("## 17. Negative-candidate-title exclusions")
    lines.append("")
    for dataset in ["Amazon Beauty", "Amazon Sports", "ML-1M"]:
        negative_only = exclusion_by_dataset.get(dataset, 0) - positive_exclusion_by_dataset.get(dataset, 0)
        lines.append(f"- {dataset}: {negative_only} title-excluded rows have only non-positive candidate title gaps.")
    lines.append("")
    lines.append("## 18. Cohort-size and retention analysis")
    lines.append("")
    lines.extend(markdown_table(
        ["cohort_id", "eligible_rows"],
        [[cohort_id, cohort_totals[cohort_id]] for cohort_id in ALLOWED_COHORT_IDS],
    ))
    lines.append("")
    lines.append("## 19. Positive-position distribution comparison")
    lines.append("")
    lines.extend(markdown_table(
        ["dataset", "task_type", "candidate_set_id", "full_rows", "strict_rows", "full_distribution", "strict_distribution"],
        plan_e_position_rows,
    ))
    lines.append("")
    lines.append("## 20. Potential selection bias introduced by title exclusions")
    lines.append("")
    lines.append("- Beauty and Sports strict-title cohorts exclude rows with unresolved local metadata; this can change user/item composition and positive-position distributions.")
    lines.append("- ML-1M has no Plan E2 title exclusions, so no title-gap selection filter is introduced for ML-1M in this phase.")
    lines.append("- High retention does not prove absence of bias; future reports must label full-baseline and strict-title/matched cohorts separately.")
    lines.append("")
    lines.append("## 21. Rules for future traditional-baseline evaluation")
    lines.append("")
    lines.append("- Baselines may be run on `plan_e2_paper_like_full` for full-cohort baseline-only reporting.")
    lines.append("- Baselines must also be run on `plan_e2_matched_comparison` before direct comparison with LLM results.")
    lines.append("")
    lines.append("## 22. Rules for future LLM evaluation")
    lines.append("")
    lines.append("- LLM prompt construction may use only `plan_e2_llm_strict_title` rows.")
    lines.append("- Rows listed in `llm_title_exclusions.csv` must not be repaired or silently included.")
    lines.append("")
    lines.append("## 23. Rules for direct baseline-versus-LLM comparisons")
    lines.append("")
    lines.append("- Direct comparisons must use `plan_e2_matched_comparison`, whose membership is identical to the strict-title LLM cohort.")
    lines.append("- Full-cohort baseline numbers may be reported only as a separately labeled baseline-only condition.")
    lines.append("")
    lines.append("## 24. Canonical files to use in later phases")
    lines.append("")
    lines.append("- `results/tables/experiment_cohort_membership.csv`")
    lines.append("- `results/tables/experiment_cohort_summary.csv`")
    lines.append("- `results/tables/llm_title_exclusions.csv`")
    lines.append("- `results/tables/candidate_item_title_map.csv`")
    lines.append("")
    lines.append("## 25. Files that must not be mixed")
    lines.append("")
    lines.append("- Do not mix Plan B/C 100-candidate rows with Plan E2 20-candidate paper-like rows.")
    lines.append("- Do not mix Plan D diagnostic rows with main paper-like results.")
    lines.append("- Do not compare Plan E2 full baseline rows directly against strict-title LLM rows without matched-cohort recomputation.")
    lines.append("")
    lines.append("## 26. File-size and repository-management note")
    lines.append("")
    lines.append(f"- Full item-title map size: {input_sizes['results/tables/item_title_map.csv']} bytes.")
    lines.append(f"- Candidate-only title map size: {DEFAULT_CANDIDATE_TITLE_MAP_OUTPUT.stat().st_size if DEFAULT_CANDIDATE_TITLE_MAP_OUTPUT.exists() else 'written after report'} bytes.")
    lines.append("- No file was deleted, compressed, or replaced in this task.")
    lines.append("")
    lines.append("## 27. Tests and validation commands")
    lines.append("")
    lines.append("- `$env:PYTHONNOUSERSITE='1'; & 'C:\\ProgramData\\anaconda3\\python.exe' src/experiments/build_experiment_cohorts.py`")
    lines.append("- `$env:PYTHONNOUSERSITE='1'; & 'C:\\ProgramData\\anaconda3\\python.exe' -m pytest tests/test_experiment_cohort_policy.py tests/test_item_title_map_reconciliation.py tests/test_dataset_candidate_set_validation.py`")
    for message in validation_messages:
        lines.append(f"- {message}")
    lines.append("")
    lines.append("## 28. Recommended single next task")
    lines.append("")
    lines.append("Phase 3 traditional-baseline consolidation on both the Plan E2 full cohort and the Plan E2 matched comparison cohort.")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def source_plan_family_rows(rows: list[dict[str, Any]]) -> list[list[Any]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for row in rows:
        grouped[row["source_plan"]].add(row["candidate_set_id"])
        counts[row["source_plan"]] += 1
    return [[plan, len(grouped[plan]), counts[plan]] for plan in sorted(grouped)]


def task_classification_table(rows: list[dict[str, Any]]) -> list[list[Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["candidate_set_id"]].append(row)
    output: list[list[Any]] = []
    for candidate_set_id in sorted(grouped):
        first = grouped[candidate_set_id][0]
        output.append(
            [
                candidate_set_id,
                first["dataset"],
                first["source_plan"],
                first["task_type"],
                len(grouped[candidate_set_id]),
            ]
        )
    return output


def cohort_totals_from_summary(summary_rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    totals = {cohort_id: 0 for cohort_id in ALLOWED_COHORT_IDS}
    for row in summary_rows:
        totals[row["cohort_id"]] += int(row["eligible_rows"])
    return totals


def summary_lines_for_cohort(summary_rows: list[dict[str, Any]], cohort_id: str) -> list[str]:
    rows = [row for row in summary_rows if row["cohort_id"] == cohort_id]
    if not rows:
        return ["No applicable candidate sets."]
    return markdown_table(
        ["dataset", "task_type", "candidate_set_id", "total", "eligible", "excluded", "rate"],
        [
            [
                row["dataset"],
                row["task_type"],
                row["candidate_set_id"],
                row["total_source_rows"],
                row["eligible_rows"],
                row["excluded_rows"],
                f"{float(row['eligibility_rate']):.6f}",
            ]
            for row in rows
        ],
    )


def plan_e_position_comparison(summary_rows: list[dict[str, Any]]) -> list[list[Any]]:
    full = {
        row["candidate_set_id"]: row
        for row in summary_rows
        if row["cohort_id"] == "plan_e2_paper_like_full"
    }
    strict = {
        row["candidate_set_id"]: row
        for row in summary_rows
        if row["cohort_id"] == "plan_e2_llm_strict_title"
    }
    output: list[list[Any]] = []
    for candidate_set_id in sorted(full):
        full_row = full[candidate_set_id]
        strict_row = strict[candidate_set_id]
        output.append(
            [
                full_row["dataset"],
                full_row["task_type"],
                candidate_set_id,
                full_row["eligible_rows"],
                strict_row["eligible_rows"],
                full_row["positive_position_distribution"],
                strict_row["positive_position_distribution"],
            ]
        )
    return output


def unique_missing_items_by_dataset(exclusion_rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    items: dict[str, set[str]] = defaultdict(set)
    for row in exclusion_rows:
        items[row["dataset"]].update(json.loads(str(row["missing_title_item_ids"])))
    return {dataset: len(values) for dataset, values in items.items()}


def count_by(rows: Iterable[Mapping[str, Any]], field: str) -> Counter[str]:
    return Counter(str(row[field]) for row in rows)


def group_by(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return grouped


def markdown_kv_table(values: Mapping[str, Any]) -> list[str]:
    return markdown_table(
        ["field", "value"],
        [[key, json.dumps(value, sort_keys=True) if isinstance(value, dict) else value] for key, value in values.items()],
    )


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return ["No rows."]
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(str(value) for value in row) + " |")
    return output


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    return _read_csv_rows(path)


def write_csv(path: Path, columns: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    _write_csv_rows(path, columns, rows, extrasaction="ignore")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
