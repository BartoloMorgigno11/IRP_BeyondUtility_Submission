"""Forensic reparse of F6L LastFM operationalized ranking raw outputs.

This module performs no inference. It reads the immutable raw outputs from the
authorized F6L LastFM operationalized ranking run and builds parser failure
taxonomy, a conservative V2 parser audit, complete-case metrics, and reports.
"""

from __future__ import annotations

import argparse
import csv
import html
import inspect
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.analysis import plan_f6l_paper_faithful_completion_and_rq_revalidation as f6l


DECISION_RECOVERY_COMPLETE = "f6l_lastfm_ranking_reparse_complete_with_conservative_parser_recovery"
DECISION_AMBIGUOUS_MAPPING = "f6l_lastfm_ranking_evaluation_blocked_by_ambiguous_output_mapping"
DECISION_RERUN_REQUIRED = "f6l_lastfm_ranking_rerun_required_due_prompt_output_contract_failure"
DECISION_CONTRACT_DEFECT = "f6l_lastfm_ranking_contract_defect_requires_corrective_pilot"

RUN_ID = "20260802_184502"
RAW_OUTPUT_DIR = f6l.PROJECT_ROOT / "results" / "llm" / f"phase_f6l_lastfm_operationalized_ranking_{RUN_ID}"
AUDIT_SEED = "phase_f6l_lastfm_parser_forensics_blind_audit_seed_v1"
PARSER_V2_VERSION = "phase_f6l_lastfm_conservative_title_rank_parser_v2"

TABLES_DIR = f6l.PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = f6l.PROJECT_ROOT / "results" / "manifests"
REPORT_PATH = f6l.PROJECT_ROOT / "report" / "phase_f6l_lastfm_parser_forensics.md"

TAXONOMY_PATH = TABLES_DIR / "phase_f6l_lastfm_parser_failure_taxonomy.csv"
SUMMARY_PATH = TABLES_DIR / "phase_f6l_lastfm_parser_failure_summary.csv"
AUDIT_SAMPLE_PATH = TABLES_DIR / "phase_f6l_lastfm_parser_blind_audit_sample.csv"
V1_CONTRACT_PATH = TABLES_DIR / "phase_f6l_lastfm_parser_v1_contract.csv"
V2_CONTRACT_PATH = TABLES_DIR / "phase_f6l_lastfm_parser_v2_contract.csv"
HOLDOUT_PATH = TABLES_DIR / "phase_f6l_lastfm_parser_v2_holdout_validation.csv"
REPARSE_V1_PATH = TABLES_DIR / "phase_f6l_lastfm_reparse_v1_results.csv"
REPARSE_V2_PATH = TABLES_DIR / "phase_f6l_lastfm_reparse_v2_results.csv"
V1_V2_COMPARISON_PATH = TABLES_DIR / "phase_f6l_lastfm_reparse_v1_v2_comparison.csv"
V2_METRICS_PATH = TABLES_DIR / "phase_f6l_lastfm_reparse_v2_complete_case_metrics.csv"
V2_VALIDITY_PATH = TABLES_DIR / "phase_f6l_lastfm_reparse_v2_validity_coverage.csv"
FAIL_STRICT_PATH = TABLES_DIR / "phase_f6l_lastfm_reparse_fail_strict_sensitivity.csv"
BOUNDS_PATH = TABLES_DIR / "phase_f6l_lastfm_reparse_bounds.csv"
RQ_CORRECTION_PATH = TABLES_DIR / "phase_f6l_lastfm_reparse_rq_correction.csv"
RAW_PROTECTION_PATH = TABLES_DIR / "phase_f6l_lastfm_reparse_raw_protection.csv"
MANIFEST_PATH = MANIFESTS_DIR / "phase_f6l_lastfm_parser_forensics_manifest.json"

PARSER_V2_RULES = [
    "NFKC Unicode normalization, HTML unescape, case folding, whitespace normalization.",
    "Mojibake repair aliases are considered only when they map uniquely to one candidate title.",
    "Markdown emphasis, wrapping quotes, list bullets, and numeric list prefixes are removed.",
    "Candidate IDs are accepted only when the leading list item is exactly one authorized candidate ID.",
    "JSON/list outputs are accepted only when all selected entries map uniquely to candidates.",
    "Numbered or bullet blocks are accepted only when a clear ordered candidate block exists.",
    "For a clear ordered block longer than 10, only the first 10 are evaluated as the top-10 prefix.",
    "Ambiguous candidate mappings, duplicate top-10 items, hallucinated top-10 items, or fewer than 10 mapped items remain invalid.",
    "No fuzzy matching, relevance labels, positive item, HR, NDCG, or desired utility result is used by the parser.",
]


@dataclass(frozen=True)
class ParserV2Result:
    conservative_valid: bool
    ranked_item_ids: list[str]
    ranked_titles: list[str]
    parsed_items: list[str]
    extraction_rule: str
    failure_reason: str
    output_count: int
    candidate_match_count: int
    exact_match_count: int
    normalized_match_count: int
    hallucinated_items: list[str]
    ambiguous_items: list[str]
    duplicate_items: list[str]
    parse_exception: str
    block_count: int


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean_cell(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def json_list(values: Iterable[Any]) -> str:
    return json.dumps([str(value) for value in values], ensure_ascii=True, separators=(",", ":"))


def parse_json_list(value: str) -> list[str]:
    try:
        payload = json.loads(value or "[]")
        if isinstance(payload, list):
            return [str(item) for item in payload]
    except Exception:
        return []
    return []


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean_cell(row.get(field, "")) for field in fieldnames})
    temp.replace(path)


def stable_id(prefix: str, *parts: object) -> str:
    return f"{prefix}-{f6l.sha256_text('|'.join(str(part) for part in parts))[:16]}"


def format_float(value: float, digits: int = 12) -> str:
    if math.isnan(value):
        return ""
    text = f"{value:.{digits}g}"
    return "0" if text == "-0" else text


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def raw_paths() -> dict[str, Path]:
    return f6l.raw_response_paths(RAW_OUTPUT_DIR)


def raw_file_hashes() -> dict[str, str]:
    paths = raw_paths()
    response_files = sorted(paths["individual"].glob("*.json"))
    payload = {
        f6l.rel(paths["requests"]): f6l.safe_file_sha256(paths["requests"]),
        f6l.rel(paths["responses"]): f6l.safe_file_sha256(paths["responses"]),
        **{f6l.rel(path): f6l.safe_file_sha256(path) for path in response_files},
    }
    return payload


def load_raw_records() -> list[dict[str, Any]]:
    records = f6l.read_jsonl(raw_paths()["responses"])
    finals = [dict(record) for record in records if str(record.get("is_final_attempt", "true")).lower() == "true"]
    finals.sort(key=lambda row: row.get("request_id", ""))
    return finals


def normalize_text(value: str) -> str:
    text = html.unescape(str(value or ""))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def punctuation_light_key(value: str) -> str:
    text = normalize_text(value)
    text = re.sub(r"^[\"'“”‘’`*_\\s]+|[\"'“”‘’`*_\\s]+$", "", text)
    text = re.sub(r"\s*\([^)]{1,80}\)\s*$", "", text)
    text = re.sub(r"[^\w&]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def mojibake_variants(value: str) -> set[str]:
    variants = {str(value or "")}
    text = str(value or "")
    if any(marker in text for marker in ["Ã", "Â", "â"]):
        try:
            repaired = text.encode("latin1").decode("utf-8")
            if repaired:
                variants.add(repaired)
        except Exception:
            pass
    return variants


def strip_markdown(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^\s*[*_`]+|[*_`]+\s*$", "", text)
    text = re.sub(r"^\s*\*\*(.*?)\*\*\s*$", r"\1", text)
    text = re.sub(r"^\s*__(.*?)__\s*$", r"\1", text)
    return text.strip()


def title_aliases(title: str) -> set[str]:
    aliases: set[str] = set()
    for variant in mojibake_variants(title):
        cleaned = strip_markdown(variant)
        aliases.add(normalize_text(cleaned))
        aliases.add(punctuation_light_key(cleaned))
    return {alias for alias in aliases if alias}


def build_candidate_lookup(candidate_rows: Sequence[Mapping[str, str]]) -> tuple[dict[str, list[Mapping[str, str]]], dict[str, Mapping[str, str]]]:
    aliases: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    by_id = {row["candidate_item_id"]: row for row in candidate_rows}
    for row in candidate_rows:
        for alias in title_aliases(row["candidate_item_title"]):
            aliases[alias].append(row)
    return dict(aliases), by_id


def lookup_candidate(value: str, aliases: Mapping[str, list[Mapping[str, str]]], by_id: Mapping[str, Mapping[str, str]]) -> tuple[str, str, str]:
    segment = strip_markdown(value)
    segment = re.sub(r"^\s*[\"']|[\"']\s*$", "", segment).strip()
    if re.fullmatch(r"\d+", segment) and segment in by_id:
        row = by_id[segment]
        return "mapped", row["candidate_item_id"], row["candidate_item_title"]
    keys = [normalize_text(segment), punctuation_light_key(segment)]
    matches: list[Mapping[str, str]] = []
    seen: set[str] = set()
    for key in keys:
        for row in aliases.get(key, []):
            if row["candidate_item_id"] not in seen:
                matches.append(row)
                seen.add(row["candidate_item_id"])
    if len(matches) == 1:
        row = matches[0]
        return "mapped", row["candidate_item_id"], row["candidate_item_title"]
    if len(matches) > 1:
        return "ambiguous", "", segment
    return "unmapped", "", segment


def exact_title_mentions(text: str, candidate_rows: Sequence[Mapping[str, str]]) -> set[str]:
    return {row["candidate_item_id"] for row in candidate_rows if row["candidate_item_title"] and row["candidate_item_title"] in text}


def normalized_title_mentions(text: str, candidate_rows: Sequence[Mapping[str, str]]) -> set[str]:
    normalized = f" {punctuation_light_key(text)} "
    found: set[str] = set()
    for row in candidate_rows:
        for alias in title_aliases(row["candidate_item_title"]):
            if alias and f" {alias} " in normalized:
                found.add(row["candidate_item_id"])
                break
    return found


def candidate_id_mentions(text: str, candidate_rows: Sequence[Mapping[str, str]]) -> set[str]:
    tokens = set(re.findall(r"\b\d+\b", text))
    candidate_ids = {row["candidate_item_id"] for row in candidate_rows}
    return tokens & candidate_ids


def bracketed_list_candidates(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    for match in re.findall(r"\[[\s\S]*?\]", text):
        try:
            payload = json.loads(match)
        except Exception:
            try:
                payload = json.loads(match.replace("'", '"'))
            except Exception:
                continue
        if isinstance(payload, list):
            blocks.append([str(item) for item in payload])
    return blocks


def list_line_segment(line: str) -> tuple[bool, str]:
    text = line.strip()
    if not text:
        return False, ""
    text = re.sub(r"^\s*(?:>\s*)+", "", text)
    patterns = [
        r"^\s*\*\*\s*(\d{1,3})[\).\:-]\s*(.*?)\s*\*\*\s*$",
        r"^\s*(\d{1,3})[\).\:-]\s+(.*)$",
        r"^\s*[-*+]\s+(.*)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            segment = match.group(match.lastindex or 1)
            return True, segment.strip()
    return False, ""


def leading_title_segment(segment: str) -> str:
    text = strip_markdown(segment)
    text = re.sub(r"^\s*[\"']|[\"']\s*$", "", text).strip()
    for delimiter in [" - ", " -- ", " – ", " — ", ": "]:
        if delimiter in text:
            text = text.split(delimiter, 1)[0].strip()
            break
    text = re.sub(r"\s*\([^)]{1,80}\)\s*$", "", text).strip()
    text = strip_markdown(text)
    return text


def list_blocks(text: str) -> list[list[dict[str, str]]]:
    blocks: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    for line in text.splitlines():
        is_list, segment = list_line_segment(line)
        if is_list:
            current.append({"line": line, "segment": leading_title_segment(segment)})
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def parse_v2_response(raw_text: str, candidate_rows: Sequence[Mapping[str, str]], expected_count: int = 10) -> ParserV2Result:
    text = str(raw_text or "")
    aliases, by_id = build_candidate_lookup(candidate_rows)
    exact = exact_title_mentions(text, candidate_rows)
    normalized = normalized_title_mentions(text, candidate_rows)
    parse_exception = ""
    candidates: list[dict[str, Any]] = []

    def evaluate_items(items: Sequence[str], rule: str, block_count: int) -> ParserV2Result:
        mapped_ids: list[str] = []
        mapped_titles: list[str] = []
        parsed_items: list[str] = []
        hallucinated: list[str] = []
        ambiguous: list[str] = []
        for item in items[:expected_count]:
            parsed_items.append(str(item))
            status, item_id, title = lookup_candidate(str(item), aliases, by_id)
            if status == "mapped":
                mapped_ids.append(item_id)
                mapped_titles.append(title)
            elif status == "ambiguous":
                ambiguous.append(title)
            else:
                hallucinated.append(str(item))
        duplicate_items = sorted(item for item, count in Counter(mapped_ids).items() if count > 1)
        valid = (
            len(parsed_items) >= expected_count
            and len(mapped_ids) == expected_count
            and not hallucinated
            and not ambiguous
            and not duplicate_items
        )
        failure = ""
        if not valid:
            if len(parsed_items) < expected_count:
                failure = "fewer_than_10_clear_ranked_items"
            elif ambiguous:
                failure = "ambiguous_candidate_mapping"
            elif hallucinated:
                failure = "hallucinated_or_unmapped_top10_item"
            elif duplicate_items:
                failure = "duplicate_candidate_in_top10"
            else:
                failure = "unknown_v2_invalid"
        return ParserV2Result(
            conservative_valid=valid,
            ranked_item_ids=mapped_ids[:expected_count],
            ranked_titles=mapped_titles[:expected_count],
            parsed_items=list(parsed_items[:expected_count]),
            extraction_rule=rule,
            failure_reason=failure,
            output_count=len(items),
            candidate_match_count=len(mapped_ids),
            exact_match_count=len(exact),
            normalized_match_count=len(normalized),
            hallucinated_items=hallucinated,
            ambiguous_items=ambiguous,
            duplicate_items=duplicate_items,
            parse_exception=parse_exception,
            block_count=block_count,
        )

    try:
        for block in bracketed_list_candidates(text):
            candidates.append({"rule": "json_or_bracketed_list", "items": block, "block_count": 1})
    except Exception as exc:
        parse_exception = type(exc).__name__

    blocks = list_blocks(text)
    for block in blocks:
        mapped_or_invalid: list[str] = []
        mapped_count = 0
        for entry in block:
            segment = entry["segment"]
            status, _, _ = lookup_candidate(segment, aliases, by_id)
            if status == "mapped":
                mapped_or_invalid.append(segment)
                mapped_count += 1
            elif segment:
                mapped_or_invalid.append(segment)
        if len(mapped_or_invalid) >= expected_count and mapped_count > 0:
            rule = "numbered_or_bullet_candidate_rank_top10"
            if len(mapped_or_invalid) > expected_count:
                rule = "numbered_or_bullet_candidate_rank_prefix_top10"
            candidates.append({"rule": rule, "items": mapped_or_invalid, "block_count": len(blocks)})

    valid_results = [evaluate_items(candidate["items"], candidate["rule"], int(candidate["block_count"])) for candidate in candidates]
    valid_results = [result for result in valid_results if result.conservative_valid]
    if valid_results:
        signatures = {tuple(result.ranked_item_ids) for result in valid_results}
        if len(signatures) == 1:
            return valid_results[-1]
        return ParserV2Result(
            conservative_valid=False,
            ranked_item_ids=[],
            ranked_titles=[],
            parsed_items=[],
            extraction_rule="multiple_conflicting_valid_blocks",
            failure_reason="ambiguous_multiple_valid_rankings",
            output_count=0,
            candidate_match_count=0,
            exact_match_count=len(exact),
            normalized_match_count=len(normalized),
            hallucinated_items=[],
            ambiguous_items=["multiple_conflicting_valid_blocks"],
            duplicate_items=[],
            parse_exception=parse_exception,
            block_count=len(blocks),
        )

    if candidates:
        evaluated = [evaluate_items(candidate["items"], candidate["rule"], int(candidate["block_count"])) for candidate in candidates]
        evaluated.sort(key=lambda item: (item.candidate_match_count, -len(item.hallucinated_items), -len(item.ambiguous_items)), reverse=True)
        return evaluated[0]

    failure = "no_structured_candidate_rank_block"
    if len(normalized) == 0:
        failure = "no_candidate_titles_recognizable"
    return ParserV2Result(
        conservative_valid=False,
        ranked_item_ids=[],
        ranked_titles=[],
        parsed_items=[],
        extraction_rule="none",
        failure_reason=failure,
        output_count=0,
        candidate_match_count=0,
        exact_match_count=len(exact),
        normalized_match_count=len(normalized),
        hallucinated_items=[],
        ambiguous_items=[],
        duplicate_items=[],
        parse_exception=parse_exception,
        block_count=len(blocks),
    )


def parser_v2_hash() -> str:
    source = inspect.getsource(parse_v2_response) + "\n" + inspect.getsource(lookup_candidate) + "\n" + canonical_json(PARSER_V2_RULES)
    return f6l.sha256_text(source)


def v1_failure_category(v1: Mapping[str, Any], text: str) -> str:
    if v1.get("strict_valid"):
        return "v1_strict_valid"
    reason = str(v1.get("failure_reason", ""))
    if not str(text or "").strip():
        return "empty_response"
    if "missing_or_multiple_bracketed_lists" in reason:
        return "textual_ranking_not_single_bracketed_list"
    if "JSONDecodeError" in reason or "title_list_parse_failed" in reason:
        return "bracketed_list_parse_exception"
    if not v1.get("output_count_valid", False):
        return "wrong_number_of_items_for_v1"
    if not v1.get("candidate_validity", False):
        return "candidate_title_mapping_failure"
    if not v1.get("uniqueness_valid", True):
        return "duplicate_candidate_failure"
    return "other_v1_strict_failure"


def format_signature(text: str) -> str:
    numbered = len([line for line in text.splitlines() if re.match(r"^\s*\d{1,3}[\).\:-]\s+", line)])
    bullets = len([line for line in text.splitlines() if re.match(r"^\s*[-*+]\s+", line)])
    markdown = bool(re.search(r"\*\*|__", text))
    bracketed = len(re.findall(r"\[[\s\S]*?\]", text))
    intro = bool(re.search(r"(?i)\b(to determine|based on|here are|analysis|rank)\b", text[:600]))
    return f"numbered={numbered};bullets={bullets};markdown={str(markdown).lower()};bracketed_lists={bracketed};introductory_text={str(intro).lower()}"


def redacted_structural_excerpt(text: str, candidate_rows: Sequence[Mapping[str, str]], max_lines: int = 8) -> str:
    redacted = str(text or "")
    for row in sorted(candidate_rows, key=lambda item: len(item["candidate_item_title"]), reverse=True):
        title = row["candidate_item_title"]
        if title:
            redacted = redacted.replace(title, "[CANDIDATE_TITLE]")
            for variant in mojibake_variants(title):
                redacted = redacted.replace(variant, "[CANDIDATE_TITLE]")
    redacted = re.sub(r"[A-Za-z0-9][A-Za-z0-9 '&./$-]{2,}", "[TEXT]", redacted)
    return "\\n".join(redacted.splitlines()[:max_lines])


def output_length_bucket(char_count: int) -> str:
    if char_count < 250:
        return "short_lt_250"
    if char_count < 1000:
        return "medium_250_999"
    if char_count < 2500:
        return "long_1000_2499"
    return "very_long_ge_2500"


def deterministic_score(*parts: str) -> str:
    return f6l.sha256_text("|".join(parts))


def select_audit_sample(rows: Sequence[Mapping[str, str]], max_per_cell_category: int = 2, max_total: int = 40) -> set[str]:
    groups: dict[tuple[str, str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["model_name"], row["prompting_strategy"], row["v1_failure_category"])].append(row)
    selected: list[str] = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda row: deterministic_score(AUDIT_SEED, row["request_id"]))
        selected.extend(row["request_id"] for row in ordered[:max_per_cell_category])
    selected = sorted(set(selected), key=lambda request_id: deterministic_score(AUDIT_SEED, request_id))[:max_total]
    return set(selected)


def positive_rank(ranked_item_ids: Sequence[str], positive_item: str) -> int | None:
    try:
        return list(ranked_item_ids).index(positive_item) + 1
    except ValueError:
        return None


def ndcg_at_5(rank: int | None) -> float:
    if rank is None or rank > 5:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def build_v2_metrics(v2_rows: Sequence[Mapping[str, str]], manifest_rows: Sequence[Mapping[str, str]], candidate_universe: set[str]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    manifest_by_id = {row["request_id"]: row for row in manifest_rows}
    metrics: list[dict[str, str]] = []
    validity: list[dict[str, str]] = []
    sensitivity: list[dict[str, str]] = []
    bounds: list[dict[str, str]] = []
    for model in f6l.MODEL_DIGESTS:
        for strategy in f6l.LASTFM_RANKING_STRATEGIES:
            cell_rows = [row for row in v2_rows if row["model_name"] == model and row["prompting_strategy"] == strategy]
            valid_rows = [row for row in cell_rows if row["conservative_valid_v2"] == "true"]
            total = len(cell_rows)
            valid_count = len(valid_rows)
            invalid_count = total - valid_count
            hits: list[float] = []
            ndcgs: list[float] = []
            ranks: list[float] = []
            top5_items: list[str] = []
            for row in valid_rows:
                manifest = manifest_by_id[row["request_id"]]
                ranked = parse_json_list(row["ranked_item_ids_v2"])
                rank = positive_rank(ranked, manifest["positive_item"])
                hits.append(1.0 if rank is not None and rank <= 5 else 0.0)
                ndcgs.append(ndcg_at_5(rank))
                ranks.append(float(rank) if rank is not None else 20.0)
                top5_items.extend(ranked[:5])
            metric_values = {
                "HR@5": sum(hits) / len(hits) if hits else math.nan,
                "NDCG@5": sum(ndcgs) / len(ndcgs) if ndcgs else math.nan,
                "mean_positive_rank": sum(ranks) / len(ranks) if ranks else math.nan,
            }
            for metric, value in metric_values.items():
                metrics.append(
                    {
                        "metric_id": stable_id("F6L-LASTFM-V2-METRIC", model, strategy, metric),
                        "sample_id": "phase_f6l_lastfm_operationalized_1000",
                        "dataset": "LastFM",
                        "task": "ranking",
                        "model_name": model,
                        "model_digest": f6l.MODEL_DIGESTS[model],
                        "prompting_strategy": strategy,
                        "evidence_class": "paper_protocol_reproduction_under_explicit_operationalization",
                        "parser_version": PARSER_V2_VERSION,
                        "metric": metric,
                        "k": "5" if metric != "mean_positive_rank" else "",
                        "value": format_float(value) if not math.isnan(value) else "",
                        "valid_v2_rows": str(valid_count),
                        "total_rows": str(total),
                        "metric_population": "complete_case_v2_valid_only",
                        "source_file": f6l.rel(REPARSE_V2_PATH),
                        "notes": "Invalid V2 outputs are excluded from this complete-case metric, not imputed as zero.",
                    }
                )
                fail_strict = 0.0 if metric in {"HR@5", "NDCG@5"} else 20.0
                sensitivity.append(
                    {
                        "sensitivity_id": stable_id("F6L-LASTFM-V2-SENS", model, strategy, metric),
                        "model_name": model,
                        "prompting_strategy": strategy,
                        "metric": metric,
                        "fail_strict_value": format_float(fail_strict),
                        "complete_case_value": format_float(value) if not math.isnan(value) else "",
                        "delta_complete_case_minus_fail_strict": format_float(value - fail_strict) if not math.isnan(value) else "",
                        "valid_v2_rows": str(valid_count),
                        "invalid_v2_rows": str(invalid_count),
                        "notes": "Fail-strict values are sensitivity analysis only, not the main scientific complete-case estimate.",
                    }
                )
                if metric == "HR@5":
                    lower = sum(hits) / total if total else 0.0
                    upper = (sum(hits) + invalid_count) / total if total else 0.0
                elif metric == "NDCG@5":
                    lower = sum(ndcgs) / total if total else 0.0
                    upper = (sum(ndcgs) + invalid_count) / total if total else 0.0
                else:
                    lower = 1.0 if valid_count == total and ranks else 1.0
                    upper = 20.0
                bounds.append(
                    {
                        "bounds_id": stable_id("F6L-LASTFM-V2-BOUNDS", model, strategy, metric),
                        "model_name": model,
                        "prompting_strategy": strategy,
                        "metric": metric,
                        "lower_bound_invalid_unfavorable": format_float(lower),
                        "complete_case_observed": format_float(value) if not math.isnan(value) else "",
                        "upper_bound_invalid_favorable": format_float(upper),
                        "valid_v2_rows": str(valid_count),
                        "total_rows": str(total),
                        "notes": "Bounds show missingness sensitivity over invalid V2 rows.",
                    }
                )
            validity.append(
                {
                    "coverage_id": stable_id("F6L-LASTFM-V2-COVER", model, strategy),
                    "model_name": model,
                    "prompting_strategy": strategy,
                    "total_requests": str(total),
                    "strict_valid_v1": str(sum(1 for row in cell_rows if row["strict_valid_v1"] == "true")),
                    "conservative_valid_v2": str(valid_count),
                    "recovered_by_v2": str(sum(1 for row in cell_rows if row["recovered_by_v2"] == "true")),
                    "still_invalid_v2": str(invalid_count),
                    "recovery_rate": format_float((valid_count / total) if total else 0.0),
                    "validity_rate_v2": format_float((valid_count / total) if total else 0.0),
                    "complete_case_unique_top5_items": str(len(set(top5_items))),
                    "complete_case_coverage_top5": format_float((len(set(top5_items)) / len(candidate_universe)) if candidate_universe and top5_items else 0.0),
                    "candidate_universe": str(len(candidate_universe)),
                    "notes": "Coverage is computed only over V2-valid top-5 recommendations.",
                }
            )
    return metrics, validity, sensitivity, bounds


def build_contract_rows(parser_hash: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    v1 = [
        ("input_expected", "single bracketed list of title strings"),
        ("delimiter_contract", "exactly one text span matching square brackets"),
        ("item_count", "exactly 10 parsed titles"),
        ("matching", "candidate title lookup after V1 title_results_map/html unescape/casefold"),
        ("extra_text", "allowed only if it does not introduce additional bracketed spans; numbered prose without bracketed list is invalid"),
        ("non_candidate", "any unmapped parsed title invalidates strict output"),
        ("duplicates", "duplicate mapped candidate IDs invalidate strict output"),
    ]
    v2 = [(f"rule_{index:02d}", rule) for index, rule in enumerate(PARSER_V2_RULES, start=1)]
    return (
        [{"parser_version": "paper_title_list_top10_parser_v1", "contract_key": key, "contract_value": value, "notes": "Diagnostic only; V1 is not weakened or replaced."} for key, value in v1],
        [{"parser_version": PARSER_V2_VERSION, "contract_key": key, "contract_value": value, "parser_code_hash": parser_hash, "notes": "Frozen before holdout application; no relevance fields used."} for key, value in v2],
    )


def build_report(manifest: Mapping[str, Any], summary_rows: Sequence[Mapping[str, str]], validity_rows: Sequence[Mapping[str, str]], rq_rows: Sequence[Mapping[str, str]]) -> str:
    dominant = sorted(summary_rows, key=lambda row: int(row["request_count"]), reverse=True)[:8]
    dominant_lines = [
        f"- `{row['model_name']}` `{row['prompting_strategy']}` `{row['failure_category']}`: `{row['request_count']}`."
        for row in dominant
    ]
    validity_lines = [
        f"- `{row['model_name']}` `{row['prompting_strategy']}`: V2 valid `{row['conservative_valid_v2']}/{row['total_requests']}`; recovery `{row['recovery_rate']}`."
        for row in validity_rows
    ]
    rq_lines = [f"- `{row['rq_id']}`: `{row['new_status']}` - {row['motivation']}" for row in rq_rows]
    return "\n".join(
        [
            "# Phase F6L LastFM Parser Forensics",
            "",
            f"- Decision token: `{manifest['decision_token']}`.",
            f"- Run ID audited: `{RUN_ID}`.",
            "- No inference, reranking, full-cohort, API calls, downloads, training, or raw-output mutation was performed.",
            f"- Parser V2 hash: `{manifest['parser_v2_hash']}`.",
            f"- Raw protection: `{manifest['raw_protection_status']}`.",
            "",
            "## Dominant Failure Taxonomy",
            "",
            *dominant_lines,
            "",
            "## Conservative V2 Recovery",
            "",
            *validity_lines,
            "",
            "## Blind Audit Sample",
            "",
            f"- Deterministic audit sample rows: `{manifest['audit_sample_size']}`.",
            f"- Holdout rows: `{manifest['holdout_size']}`.",
            "- Reported examples are structural/redacted and keyed by raw response hash; positive item, positive position, HR, and NDCG were not used to define parser rules.",
            "",
            "## RQ Correction",
            "",
            *rq_lines,
            "",
            "## Artifacts",
            "",
            f"- `{f6l.rel(TAXONOMY_PATH)}`",
            f"- `{f6l.rel(SUMMARY_PATH)}`",
            f"- `{f6l.rel(REPARSE_V2_PATH)}`",
            f"- `{f6l.rel(V2_METRICS_PATH)}`",
            f"- `{f6l.rel(MANIFEST_PATH)}`",
        ]
    )


def update_notes_and_handoff(decision_token: str, manifest: Mapping[str, Any]) -> None:
    section = "phase_f6l_lastfm_parser_forensics"
    notes = "\n".join(
        [
            "## 2026-08-04 - Phase F6L LastFM Parser Forensics",
            "",
            "- Performed zero-inference forensic reparse of immutable LastFM operationalized ranking raw outputs.",
            "- Built V1 failure taxonomy, blind audit sample, conservative V2 parser diagnostics, holdout validation, complete-case metrics, and RQ correction.",
            f"- Decision: `{decision_token}`.",
            f"- V2 valid rows: `{manifest['v2_valid_total']}` / `4000`.",
        ]
    )
    handoff = "\n".join(
        [
            "## F6L LastFM Parser Forensics",
            "",
            f"- Decision token: `{decision_token}`.",
            f"- Run audited: `{RUN_ID}`.",
            f"- Parser V2 hash: `{manifest['parser_v2_hash']}`.",
            f"- Raw protection status: `{manifest['raw_protection_status']}`.",
            "- No rerun is authorized or executed by this phase.",
        ]
    )
    f6l.update_marked_section(f6l.NOTES_PATH, section, notes)
    f6l.update_marked_section(f6l.HANDOFF_PATH, section, handoff)


def run_forensics(args: argparse.Namespace) -> int:
    if args.raw_output_dir:
        raw_dir = f6l.resolve_project_path(args.raw_output_dir)
        if raw_dir != RAW_OUTPUT_DIR:
            raise ValueError(f"Only the authorized raw directory is allowed: {f6l.rel(RAW_OUTPUT_DIR)}")
    before_hashes = raw_file_hashes()
    records = load_raw_records()
    if len(records) != 4000:
        raise ValueError(f"Expected 4000 final raw records, found {len(records)}")
    manifest_rows = f6l.manifest_rows_for_operationalized_ranking()
    manifest_by_id = {row["request_id"]: row for row in manifest_rows}
    candidate_by_user = f6l.candidate_rows_for_operationalized_sample()
    parser_hash = parser_v2_hash()
    freeze_time = utc_now()

    taxonomy_rows: list[dict[str, str]] = []
    v1_rows: list[dict[str, str]] = []
    interim_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        request_id = str(record["request_id"])
        request = manifest_by_id[request_id]
        candidates = candidate_by_user[request["user_id"]]
        text = str(record.get("output_top10_text", "") or record.get("raw_backend_response", {}).get("response", ""))
        v1 = f6l.parse_paper_title_list_response(text, candidates, expected_count=10)
        exact_mentions = exact_title_mentions(text, candidates)
        normalized_mentions = normalized_title_mentions(text, candidates)
        ids = candidate_id_mentions(text, candidates)
        category = v1_failure_category(v1, text)
        parse_exception = ""
        if "title_list_parse_failed:" in str(v1.get("failure_reason", "")):
            parse_exception = str(v1.get("failure_reason", "")).split(":", 1)[-1]
        base = {
            "request_id": request_id,
            "sample_id": request["sample_id"],
            "user_id": request["user_id"],
            "model_name": request["model_name"],
            "model_digest": request["model_digest"],
            "prompting_strategy": request["prompting_strategy"],
            "raw_response_sha256": str(record.get("raw_response_sha256", "")),
            "output_char_length": str(len(text)),
            "output_line_count": str(len(text.splitlines())),
            "output_length_bucket": output_length_bucket(len(text)),
            "done_reason": str(record.get("raw_backend_response", {}).get("done_reason", "")),
            "truncated_flag": clean_cell(str(record.get("raw_backend_response", {}).get("done_reason", "")).lower() == "length"),
            "markdown_present": clean_cell(bool(re.search(r"\*\*|__", text))),
            "numbered_lines": str(len([line for line in text.splitlines() if re.match(r"^\s*\d{1,3}[\).\:-]\s+", line)])),
            "bullet_lines": str(len([line for line in text.splitlines() if re.match(r"^\s*[-*+]\s+", line)])),
            "bracketed_list_count": str(len(re.findall(r"\[[\s\S]*?\]", text))),
            "candidate_id_mentions": str(len(ids)),
            "candidate_title_recognizable_count": str(len(normalized_mentions)),
            "exact_match_count": str(len(exact_mentions)),
            "normalized_match_count": str(len(normalized_mentions)),
            "v1_failure_category": category,
            "v1_failure_detail": str(v1.get("failure_reason", "")),
            "v1_json_list_detected": clean_cell(v1.get("json_list_detected", False)),
            "v1_parse_exception": parse_exception,
            "v1_output_count": str(len(v1.get("parsed_titles", []))),
            "v1_hallucinated_count": str(v1.get("hallucinated_count", 0)),
            "v1_duplicate_count": str(v1.get("duplicate_count", 0)),
            "strict_valid_v1": clean_cell(v1.get("strict_valid", False)),
        }
        taxonomy_rows.append(dict(base))
        v1_rows.append(
            {
                **{key: base[key] for key in ["request_id", "sample_id", "user_id", "model_name", "model_digest", "prompting_strategy", "raw_response_sha256"]},
                "parser_version": "paper_title_list_top10_parser_v1",
                "strict_valid_v1": clean_cell(v1.get("strict_valid", False)),
                "json_list_detected": clean_cell(v1.get("json_list_detected", False)),
                "schema_valid": clean_cell(v1.get("schema_valid", False)),
                "output_count_valid": clean_cell(v1.get("output_count_valid", False)),
                "title_mapping_valid": clean_cell(v1.get("title_mapping_valid", False)),
                "uniqueness_valid": clean_cell(v1.get("uniqueness_valid", False)),
                "parsed_titles_v1": json_list(v1.get("parsed_titles", [])),
                "ranked_item_ids_v1": json_list(v1.get("ranked_item_ids", [])),
                "hallucinated_count_v1": str(v1.get("hallucinated_count", 0)),
                "duplicate_count_v1": str(v1.get("duplicate_count", 0)),
                "invalid_items_v1": json_list(v1.get("invalid_items", [])),
                "failure_reason_v1": str(v1.get("failure_reason", "")),
            }
        )
        interim_by_id[request_id] = {"record": record, "request": request, "text": text, "v1": v1, "base": base}

    audit_ids = select_audit_sample(taxonomy_rows)
    for row in taxonomy_rows:
        row["audit_split"] = "audit_sample" if row["request_id"] in audit_ids else "validation_holdout"

    v2_rows: list[dict[str, str]] = []
    comparison_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, str]] = []
    for request_id, bundle in interim_by_id.items():
        request = bundle["request"]
        candidates = candidate_by_user[request["user_id"]]
        text = bundle["text"]
        v1 = bundle["v1"]
        v2 = parse_v2_response(text, candidates, expected_count=10)
        split = "audit_sample" if request_id in audit_ids else "validation_holdout"
        recovered = bool((not v1.get("strict_valid")) and v2.conservative_valid)
        still_invalid = not v2.conservative_valid
        v2_rows.append(
            {
                "request_id": request_id,
                "sample_id": request["sample_id"],
                "user_id": request["user_id"],
                "model_name": request["model_name"],
                "model_digest": request["model_digest"],
                "prompting_strategy": request["prompting_strategy"],
                "raw_response_sha256": str(bundle["record"].get("raw_response_sha256", "")),
                "parser_version": PARSER_V2_VERSION,
                "parser_v2_hash": parser_hash,
                "audit_split": split,
                "strict_valid_v1": clean_cell(v1.get("strict_valid", False)),
                "conservative_valid_v2": clean_cell(v2.conservative_valid),
                "recovered_by_v2": clean_cell(recovered),
                "still_invalid_v2": clean_cell(still_invalid),
                "extraction_rule": v2.extraction_rule,
                "failure_reason_v2": v2.failure_reason,
                "output_count_v2": str(v2.output_count),
                "candidate_match_count_v2": str(v2.candidate_match_count),
                "exact_match_count": str(v2.exact_match_count),
                "normalized_match_count": str(v2.normalized_match_count),
                "ranked_item_ids_v2": json_list(v2.ranked_item_ids),
                "ranked_titles_v2": json_list(v2.ranked_titles),
                "parsed_items_v2": json_list(v2.parsed_items),
                "hallucinated_count_v2": str(len(v2.hallucinated_items)),
                "duplicate_count_v2": str(len(v2.duplicate_items)),
                "ambiguous_mapping_count_v2": str(len(v2.ambiguous_items)),
                "hallucinated_items_hash": f6l.sha256_text(json_list(v2.hallucinated_items)) if v2.hallucinated_items else "",
                "duplicate_items_hash": f6l.sha256_text(json_list(v2.duplicate_items)) if v2.duplicate_items else "",
                "ambiguous_items_hash": f6l.sha256_text(json_list(v2.ambiguous_items)) if v2.ambiguous_items else "",
                "parse_exception_v2": v2.parse_exception,
                "block_count_v2": str(v2.block_count),
            }
        )
        comparison_rows.append(
            {
                "request_id": request_id,
                "model_name": request["model_name"],
                "prompting_strategy": request["prompting_strategy"],
                "strict_valid_v1": clean_cell(v1.get("strict_valid", False)),
                "conservative_valid_v2": clean_cell(v2.conservative_valid),
                "recovered_by_v2": clean_cell(recovered),
                "still_invalid_v2": clean_cell(still_invalid),
                "v1_failure_category": bundle["base"]["v1_failure_category"],
                "v1_failure_detail": str(v1.get("failure_reason", "")),
                "v2_extraction_rule": v2.extraction_rule,
                "v2_failure_reason": v2.failure_reason,
                "audit_split": split,
            }
        )
        if split == "audit_sample":
            audit_rows.append(
                {
                    "request_id": request_id,
                    "model_name": request["model_name"],
                    "prompting_strategy": request["prompting_strategy"],
                    "v1_failure_category": bundle["base"]["v1_failure_category"],
                    "raw_response_sha256": str(bundle["record"].get("raw_response_sha256", "")),
                    "format_signature": format_signature(text),
                    "redacted_excerpt_sha256": f6l.sha256_text(redacted_structural_excerpt(text, candidates)),
                    "redacted_structural_excerpt": redacted_structural_excerpt(text, candidates),
                    "audit_policy": "blind_to_positive_item_positive_position_hr_ndcg",
                }
            )

    v2_by_id = {row["request_id"]: row for row in v2_rows}
    for row in taxonomy_rows:
        v2 = v2_by_id[row["request_id"]]
        row.update(
            {
                "v2_extraction_rule": v2["extraction_rule"],
                "v2_failure_category": v2["failure_reason_v2"] or "v2_valid",
                "v2_output_count": v2["output_count_v2"],
                "v2_candidate_match_count": v2["candidate_match_count_v2"],
                "v2_hallucinated_count": v2["hallucinated_count_v2"],
                "v2_duplicate_count": v2["duplicate_count_v2"],
                "v2_ambiguous_mapping_count": v2["ambiguous_mapping_count_v2"],
                "conservative_valid_v2": v2["conservative_valid_v2"],
                "recovered_by_v2": v2["recovered_by_v2"],
                "still_invalid_v2": v2["still_invalid_v2"],
            }
        )

    summary_rows: list[dict[str, str]] = []
    summary_groups: dict[tuple[str, str, str, str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in taxonomy_rows:
        key = (row["model_name"], row["prompting_strategy"], row["audit_split"], row["v1_failure_category"], row["output_length_bucket"])
        summary_groups[key].append(row)
    for (model, strategy, split, category, bucket), rows in sorted(summary_groups.items()):
        count = len(rows)
        summary_rows.append(
            {
                "summary_id": stable_id("F6L-LASTFM-PARSE-SUM", model, strategy, split, category, bucket),
                "model_name": model,
                "prompting_strategy": strategy,
                "audit_split": split,
                "failure_category": category,
                "output_length_bucket": bucket,
                "request_count": str(count),
                "rate_within_cell": format_float(count / max(1, len([row for row in taxonomy_rows if row["model_name"] == model and row["prompting_strategy"] == strategy and row["audit_split"] == split]))),
                "mean_candidate_title_recognizable": format_float(statistics.mean(int(row["candidate_title_recognizable_count"]) for row in rows)),
                "mean_exact_match_count": format_float(statistics.mean(int(row["exact_match_count"]) for row in rows)),
                "mean_normalized_match_count": format_float(statistics.mean(int(row["normalized_match_count"]) for row in rows)),
                "total_hallucinated_v2": str(sum(int(row["v2_hallucinated_count"]) for row in rows)),
                "total_duplicates_v2": str(sum(int(row["v2_duplicate_count"]) for row in rows)),
                "parse_exception_count_v1": str(sum(1 for row in rows if row["v1_parse_exception"])),
                "notes": "Summary is blind to positive item and utility metrics.",
            }
        )

    holdout_rows: list[dict[str, str]] = []
    for split in ["audit_sample", "validation_holdout", "all"]:
        split_rows = v2_rows if split == "all" else [row for row in v2_rows if row["audit_split"] == split]
        for model in f6l.MODEL_DIGESTS:
            for strategy in f6l.LASTFM_RANKING_STRATEGIES:
                cell = [row for row in split_rows if row["model_name"] == model and row["prompting_strategy"] == strategy]
                if not cell:
                    continue
                valid = sum(1 for row in cell if row["conservative_valid_v2"] == "true")
                holdout_rows.append(
                    {
                        "validation_id": stable_id("F6L-LASTFM-V2-HOLDOUT", split, model, strategy),
                        "parser_version": PARSER_V2_VERSION,
                        "parser_v2_hash": parser_hash,
                        "freeze_timestamp": freeze_time,
                        "split": split,
                        "model_name": model,
                        "prompting_strategy": strategy,
                        "rows": str(len(cell)),
                        "conservative_valid_v2": str(valid),
                        "recovered_by_v2": str(sum(1 for row in cell if row["recovered_by_v2"] == "true")),
                        "recovery_rate": format_float(valid / len(cell)),
                        "still_invalid_v2": str(len(cell) - valid),
                        "notes": "Parser rules were frozen before holdout scoring and do not use relevance fields.",
                    }
                )

    candidate_universe = {row["candidate_item_id"] for rows in candidate_by_user.values() for row in rows}
    metrics_rows, validity_rows, sensitivity_rows, bounds_rows = build_v2_metrics(v2_rows, manifest_rows, candidate_universe)
    valid_total = sum(1 for row in v2_rows if row["conservative_valid_v2"] == "true")
    ambiguous_total = sum(int(row["ambiguous_mapping_count_v2"]) for row in v2_rows)
    hallucination_total = sum(int(row["hallucinated_count_v2"]) for row in v2_rows)
    if valid_total >= 400:
        decision = DECISION_RECOVERY_COMPLETE
    elif valid_total > 0 or ambiguous_total > 0:
        decision = DECISION_AMBIGUOUS_MAPPING
    elif any("prompt" in row["v1_failure_detail"].casefold() for row in taxonomy_rows):
        decision = DECISION_CONTRACT_DEFECT
    else:
        decision = DECISION_RERUN_REQUIRED

    rq_rows = [
        {
            "rq_id": "F6L-LASTFM-RQ-UTILITY",
            "previous_status": "mixed_model_dependent",
            "new_status": "unresolved_for_original_paper_sample",
            "motivation": "The raw run completed but V2 did not recover a scientifically usable complete-case set under conservative parsing." if valid_total < 400 else "V2 recovered enough complete cases for local-only complete-case estimates.",
            "limits": "No utility claim is made from fail-strict zero rows; original paper sample and closed models remain unresolved.",
        },
        {
            "rq_id": "F6L-LASTFM-RQ-PROMPT",
            "previous_status": "supported_under_local_operationalization",
            "new_status": "unresolved_for_original_paper_sample" if valid_total < 400 else "mixed_prompt_dependent",
            "motivation": "Base/OpenP5 vs LLMRank cannot be judged from invalid outputs except as output-contract behavior." if valid_total < 400 else "Only V2-valid complete cases can compare prompts.",
            "limits": "Prompt comparison is separated from utility and validity/coverage.",
        },
        {
            "rq_id": "F6L-LASTFM-RQ-BEYOND-UTILITY",
            "previous_status": "mixed_model_dependent",
            "new_status": "unresolved_for_original_paper_sample",
            "motivation": "Beyond-utility metrics require valid recommendation rows; invalid rows are not imputed in the main scientific table.",
            "limits": "Report validity/coverage and missingness bounds separately.",
        },
    ]

    raw_after = raw_file_hashes()
    raw_protection = {
        "check_id": "F6L-LASTFM-RAW-PROTECT-001",
        "run_id": RUN_ID,
        "raw_output_dir": f6l.rel(RAW_OUTPUT_DIR),
        "raw_file_count_before": str(len(before_hashes)),
        "raw_file_count_after": str(len(raw_after)),
        "raw_hash_map_before_sha256": f6l.sha256_text(canonical_json(before_hashes)),
        "raw_hash_map_after_sha256": f6l.sha256_text(canonical_json(raw_after)),
        "status": "pass" if before_hashes == raw_after else "fail",
        "notes": "Forensic reparse reads immutable raw outputs and writes only derived tables/report/manifest.",
    }

    v1_contract, v2_contract = build_contract_rows(parser_hash)
    outputs: list[tuple[Path, list[dict[str, str]], list[str]]] = [
        (TAXONOMY_PATH, taxonomy_rows, TAXONOMY_FIELDS),
        (SUMMARY_PATH, summary_rows, SUMMARY_FIELDS),
        (AUDIT_SAMPLE_PATH, audit_rows, AUDIT_FIELDS),
        (V1_CONTRACT_PATH, v1_contract, V1_CONTRACT_FIELDS),
        (V2_CONTRACT_PATH, v2_contract, V2_CONTRACT_FIELDS),
        (HOLDOUT_PATH, holdout_rows, HOLDOUT_FIELDS),
        (REPARSE_V1_PATH, v1_rows, V1_RESULT_FIELDS),
        (REPARSE_V2_PATH, v2_rows, V2_RESULT_FIELDS),
        (V1_V2_COMPARISON_PATH, comparison_rows, COMPARISON_FIELDS),
        (V2_METRICS_PATH, metrics_rows, METRICS_FIELDS),
        (V2_VALIDITY_PATH, validity_rows, VALIDITY_FIELDS),
        (FAIL_STRICT_PATH, sensitivity_rows, SENSITIVITY_FIELDS),
        (BOUNDS_PATH, bounds_rows, BOUNDS_FIELDS),
        (RQ_CORRECTION_PATH, rq_rows, RQ_FIELDS),
        (RAW_PROTECTION_PATH, [raw_protection], RAW_PROTECTION_FIELDS),
    ]
    for path, rows, fields in outputs:
        write_csv(path, rows, fields)

    manifest = {
        "phase_id": "phase_f6l",
        "run_id": RUN_ID,
        "decision_token": decision,
        "created_at": utc_now(),
        "inference_performed": False,
        "ollama_generation_performed": False,
        "api_calls_performed": False,
        "training_performed": False,
        "reranking_performed": False,
        "full_cohort_performed": False,
        "raw_output_modified": False,
        "parser_v2_version": PARSER_V2_VERSION,
        "parser_v2_hash": parser_hash,
        "parser_v2_freeze_timestamp": freeze_time,
        "parser_v2_rules": PARSER_V2_RULES,
        "audit_sample_seed": AUDIT_SEED,
        "audit_sample_size": len(audit_rows),
        "holdout_size": len(v2_rows) - len(audit_rows),
        "v1_strict_valid_total": sum(1 for row in v1_rows if row["strict_valid_v1"] == "true"),
        "v2_valid_total": valid_total,
        "v2_still_invalid_total": len(v2_rows) - valid_total,
        "v2_recovery_rate": format_float(valid_total / len(v2_rows)),
        "v2_hallucinated_item_total": hallucination_total,
        "v2_ambiguous_mapping_total": ambiguous_total,
        "raw_protection_status": raw_protection["status"],
        "output_hashes": {f6l.rel(path): f6l.safe_file_sha256(path) for path, _, _ in outputs},
    }
    f6l.write_json(MANIFEST_PATH, manifest)
    REPORT_PATH.write_text(build_report(manifest, summary_rows, validity_rows, rq_rows), encoding="utf-8")
    manifest["output_hashes"][f6l.rel(REPORT_PATH)] = f6l.safe_file_sha256(REPORT_PATH)
    f6l.write_json(MANIFEST_PATH, manifest)
    update_notes_and_handoff(decision, manifest)
    print(canonical_json({"decision_token": decision, "v2_valid_total": valid_total, "still_invalid": len(v2_rows) - valid_total, "raw_protection": raw_protection["status"]}))
    return 0


TAXONOMY_FIELDS = [
    "request_id",
    "sample_id",
    "user_id",
    "model_name",
    "model_digest",
    "prompting_strategy",
    "raw_response_sha256",
    "output_char_length",
    "output_line_count",
    "output_length_bucket",
    "done_reason",
    "truncated_flag",
    "markdown_present",
    "numbered_lines",
    "bullet_lines",
    "bracketed_list_count",
    "candidate_id_mentions",
    "candidate_title_recognizable_count",
    "exact_match_count",
    "normalized_match_count",
    "v1_failure_category",
    "v1_failure_detail",
    "v1_json_list_detected",
    "v1_parse_exception",
    "v1_output_count",
    "v1_hallucinated_count",
    "v1_duplicate_count",
    "strict_valid_v1",
    "audit_split",
    "v2_extraction_rule",
    "v2_failure_category",
    "v2_output_count",
    "v2_candidate_match_count",
    "v2_hallucinated_count",
    "v2_duplicate_count",
    "v2_ambiguous_mapping_count",
    "conservative_valid_v2",
    "recovered_by_v2",
    "still_invalid_v2",
]

SUMMARY_FIELDS = [
    "summary_id",
    "model_name",
    "prompting_strategy",
    "audit_split",
    "failure_category",
    "output_length_bucket",
    "request_count",
    "rate_within_cell",
    "mean_candidate_title_recognizable",
    "mean_exact_match_count",
    "mean_normalized_match_count",
    "total_hallucinated_v2",
    "total_duplicates_v2",
    "parse_exception_count_v1",
    "notes",
]

AUDIT_FIELDS = [
    "request_id",
    "model_name",
    "prompting_strategy",
    "v1_failure_category",
    "raw_response_sha256",
    "format_signature",
    "redacted_excerpt_sha256",
    "redacted_structural_excerpt",
    "audit_policy",
]

V1_CONTRACT_FIELDS = ["parser_version", "contract_key", "contract_value", "notes"]
V2_CONTRACT_FIELDS = ["parser_version", "contract_key", "contract_value", "parser_code_hash", "notes"]
HOLDOUT_FIELDS = ["validation_id", "parser_version", "parser_v2_hash", "freeze_timestamp", "split", "model_name", "prompting_strategy", "rows", "conservative_valid_v2", "recovered_by_v2", "recovery_rate", "still_invalid_v2", "notes"]
V1_RESULT_FIELDS = ["request_id", "sample_id", "user_id", "model_name", "model_digest", "prompting_strategy", "raw_response_sha256", "parser_version", "strict_valid_v1", "json_list_detected", "schema_valid", "output_count_valid", "title_mapping_valid", "uniqueness_valid", "parsed_titles_v1", "ranked_item_ids_v1", "hallucinated_count_v1", "duplicate_count_v1", "invalid_items_v1", "failure_reason_v1"]
V2_RESULT_FIELDS = ["request_id", "sample_id", "user_id", "model_name", "model_digest", "prompting_strategy", "raw_response_sha256", "parser_version", "parser_v2_hash", "audit_split", "strict_valid_v1", "conservative_valid_v2", "recovered_by_v2", "still_invalid_v2", "extraction_rule", "failure_reason_v2", "output_count_v2", "candidate_match_count_v2", "exact_match_count", "normalized_match_count", "ranked_item_ids_v2", "ranked_titles_v2", "parsed_items_v2", "hallucinated_count_v2", "duplicate_count_v2", "ambiguous_mapping_count_v2", "hallucinated_items_hash", "duplicate_items_hash", "ambiguous_items_hash", "parse_exception_v2", "block_count_v2"]
COMPARISON_FIELDS = ["request_id", "model_name", "prompting_strategy", "strict_valid_v1", "conservative_valid_v2", "recovered_by_v2", "still_invalid_v2", "v1_failure_category", "v1_failure_detail", "v2_extraction_rule", "v2_failure_reason", "audit_split"]
METRICS_FIELDS = ["metric_id", "sample_id", "dataset", "task", "model_name", "model_digest", "prompting_strategy", "evidence_class", "parser_version", "metric", "k", "value", "valid_v2_rows", "total_rows", "metric_population", "source_file", "notes"]
VALIDITY_FIELDS = ["coverage_id", "model_name", "prompting_strategy", "total_requests", "strict_valid_v1", "conservative_valid_v2", "recovered_by_v2", "still_invalid_v2", "recovery_rate", "validity_rate_v2", "complete_case_unique_top5_items", "complete_case_coverage_top5", "candidate_universe", "notes"]
SENSITIVITY_FIELDS = ["sensitivity_id", "model_name", "prompting_strategy", "metric", "fail_strict_value", "complete_case_value", "delta_complete_case_minus_fail_strict", "valid_v2_rows", "invalid_v2_rows", "notes"]
BOUNDS_FIELDS = ["bounds_id", "model_name", "prompting_strategy", "metric", "lower_bound_invalid_unfavorable", "complete_case_observed", "upper_bound_invalid_favorable", "valid_v2_rows", "total_rows", "notes"]
RQ_FIELDS = ["rq_id", "previous_status", "new_status", "motivation", "limits"]
RAW_PROTECTION_FIELDS = ["check_id", "run_id", "raw_output_dir", "raw_file_count_before", "raw_file_count_after", "raw_hash_map_before_sha256", "raw_hash_map_after_sha256", "status", "notes"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=f6l.CONFIG_PATH)
    parser.add_argument("--raw-output-dir", type=Path, default=RAW_OUTPUT_DIR)
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args(argv)
    f6l.require_runtime()
    if args.run_id != RUN_ID:
        raise ValueError(f"Only run_id {RUN_ID} is authorized for this forensic reparse")
    return run_forensics(args)


if __name__ == "__main__":
    raise SystemExit(main())
