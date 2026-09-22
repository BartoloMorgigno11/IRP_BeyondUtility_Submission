"""Offline F6E3A parser recovery audit.

This module performs a deterministic, read-only analysis of the frozen F6E3
raw responses. It does not import Ollama clients, open sockets, or mutate any
prior F6E1/F6E2/F6E3 artifacts.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import math
import re
import string
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, Sequence


PHASE_ID = "phase_f6e3a"
FROZEN_TIMESTAMP_UTC = "2026-07-26T00:00:00Z"

QWEN2_MODEL = "qwen2:7b-instruct"
MISTRAL_MODEL = "mistral:7b-instruct"
BASE_STRATEGY_ID = "paper_base_openp5_parametric_v1"
LLMRANK_STRATEGY_ID = "paper_llmrank_recency_parametric_v1"

F6E3_REQUEST_MANIFEST = Path("results/tables/phase_f6e3_request_manifest.csv")
F6E3_INSTANCE_RESULTS = Path("results/tables/phase_f6e3_instance_results.csv")
F6E3_PARSER_VALIDITY = Path("results/tables/phase_f6e3_parser_validity.csv")
F6E1_FREEZE = Path("results/manifests/phase_f6e1_prompt_freeze.json")
F6E2_FREEZE = Path("results/manifests/phase_f6e2_preflight_freeze.json")
F6E3_FREEZE = Path("results/manifests/phase_f6e3_pilot_freeze.json")
F6E2_PILOT_DESIGN = Path("results/tables/phase_f6e2_pilot_design.csv")
F6E1_PROMPT_MODULE = Path("src/llm/plan_f6e1_prompt_strategies.py")
LOCAL_F6E3_MODULE = Path("src/analysis/plan_f6e3_exact_prompt_pilot.py")
LOCAL_OUTPUT_PARSER = Path("src/evaluation/output_parser.py")
UPSTREAM_LLM_METRICS = Path("original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py")
F2_PROTECTED_VALIDATION = Path("results/tables/finalization_f2_protected_input_validation.csv")
F6D4A_INPUT_HASHES = Path("results/tables/finalization_f6d4a_input_hashes.csv")
F6D4A_PROTECTED_VERIFICATION = Path("results/tables/finalization_f6d4a_closure_protected_input_verification.csv")
F6D3_LASTFM_MANIFEST = Path("results/tables/finalization_f6d3_lastfm_ranking_llm_manifest.csv")

RAW_PATHS = {
    QWEN2_MODEL: Path("results/llm/phase_f6e3_qwen2_exact_prompt_pilot_raw.jsonl"),
    MISTRAL_MODEL: Path("results/llm/phase_f6e3_mistral_exact_prompt_pilot_raw.jsonl"),
}
EXPECTED_RAW_HASHES = {
    RAW_PATHS[QWEN2_MODEL].as_posix(): "03d5a43d3d2f10c73e50c5086f1e3f6767051d2baff072ac777b3822bcc4305d",
    RAW_PATHS[MISTRAL_MODEL].as_posix(): "c8e57b9dfbde8ed3471ff45e1c272fbf898a48c7364402d40f49df3b60322e49",
}

REPORT_PATH = Path("report/phase_f6e3a_parser_recovery_audit.md")
MANIFEST_PATH = Path("results/manifests/phase_f6e3a_parser_recovery_freeze.json")
OUTPUT_TABLES = {
    "output_format_inventory": Path("results/tables/phase_f6e3a_output_format_inventory.csv"),
    "prompt_output_contract": Path("results/tables/phase_f6e3a_prompt_output_contract.csv"),
    "upstream_parser_audit": Path("results/tables/phase_f6e3a_upstream_parser_audit.csv"),
    "parser_ladder_results": Path("results/tables/phase_f6e3a_parser_ladder_results.csv"),
    "candidate_title_ambiguity": Path("results/tables/phase_f6e3a_candidate_title_ambiguity.csv"),
    "hallucination_audit": Path("results/tables/phase_f6e3a_hallucination_audit.csv"),
    "cell_summary": Path("results/tables/phase_f6e3a_cell_summary.csv"),
    "root_cause": Path("results/tables/phase_f6e3a_root_cause.csv"),
    "gate_reassessment": Path("results/tables/phase_f6e3a_gate_reassessment.csv"),
    "missing_or_blocked": Path("results/tables/phase_f6e3a_missing_or_blocked.csv"),
}

OUTPUT_FORMAT_COLUMNS = [
    "record_id",
    "request_id",
    "source_request_id",
    "model",
    "strategy_id",
    "dataset",
    "task",
    "pilot_stratum",
    "candidate_count",
    "serialized_title_count",
    "prompt_title_count",
    "response_length_chars",
    "response_line_count",
    "done_reason",
    "has_numbered_list",
    "has_bullets",
    "has_json",
    "has_python_list",
    "has_quotes",
    "has_reasoning_prose",
    "apparent_recommendation_count",
    "representation_type",
    "observed_categories",
    "current_parse_status",
    "current_strict_valid",
    "current_error",
    "raw_response_sha256",
    "example_excerpt",
]
PROMPT_CONTRACT_COLUMNS = [
    "contract_id",
    "strategy_id",
    "requirement_id",
    "requirement_text",
    "source_contract",
    "rendered_prompt_contains_requirement",
    "observed_compliant_count",
    "observed_noncompliant_count",
    "observed_behavior",
    "contract_gap",
    "notes",
]
UPSTREAM_AUDIT_COLUMNS = [
    "audit_id",
    "source_file",
    "function",
    "call_path",
    "input_contract",
    "output_contract",
    "normalization",
    "fallback",
    "error_handling",
    "local_difference",
    "replay_classification",
    "source_sha256",
]
PARSER_LADDER_COLUMNS = [
    "ladder_id",
    "request_id",
    "model",
    "strategy_id",
    "dataset",
    "parser_level",
    "parser_id",
    "replay_classification",
    "parse_valid",
    "strict_valid",
    "recovered_valid",
    "output_length",
    "mapped_count",
    "unmatched_count",
    "duplicate_count",
    "hallucination_count",
    "ambiguous_count",
    "incomplete",
    "candidate_matches",
    "unmatched_items",
    "duplicates",
    "failure_reason",
    "notes",
]
CANDIDATE_AMBIGUITY_COLUMNS = [
    "request_id",
    "model",
    "strategy_id",
    "dataset",
    "candidate_count",
    "serialized_title_count",
    "prompt_title_count",
    "serialization_delimiter_collision",
    "exact_duplicate_titles",
    "normalized_duplicate_titles",
    "empty_titles",
    "titles_with_year",
    "titles_with_parentheses",
    "titles_with_slash",
    "titles_with_apostrophe",
    "titles_with_unicode",
    "titles_with_punctuation",
    "titles_with_numeric_prefix",
    "non_injective_title_map",
    "title_to_candidate_mapping",
    "ambiguous_keys",
    "notes",
]
HALLUCINATION_COLUMNS = [
    "request_id",
    "model",
    "strategy_id",
    "dataset",
    "format_extra_text",
    "title_normalization_difference",
    "candidate_alias",
    "ambiguous_candidate",
    "genuine_out_of_set_title",
    "generated_explanation_only",
    "duplicate_candidate",
    "omitted_candidate",
    "insufficient_output_length",
    "hallucination_status",
    "evidence",
]
CELL_SUMMARY_COLUMNS = [
    "cell_id",
    "model",
    "strategy_id",
    "logical_requests",
    "current_strict_valid",
    "upstream_valid",
    "normalized_valid",
    "permissive_recoverable",
    "genuine_hallucinations",
    "partial_rankings",
    "duplicate_rankings",
    "ambiguous_mappings",
    "output_format_distribution",
    "root_cause_distribution",
    "gate_population",
    "notes",
]
ROOT_CAUSE_COLUMNS = [
    "request_id",
    "model",
    "strategy_id",
    "dataset",
    "primary_cause",
    "evidence",
    "confidence",
    "safe_correction",
    "whether_new_inference_is_required",
]
GATE_REASSESSMENT_COLUMNS = [
    "gate_id",
    "population",
    "eligible_strict_count",
    "total_requests",
    "strict_valid_rate",
    "threshold_source",
    "gate_status",
    "decision",
    "notes",
]
MISSING_COLUMNS = [
    "block_id",
    "status",
    "severity",
    "affected_scope",
    "evidence",
    "notes",
]

TABLE_COLUMNS = {
    "output_format_inventory": OUTPUT_FORMAT_COLUMNS,
    "prompt_output_contract": PROMPT_CONTRACT_COLUMNS,
    "upstream_parser_audit": UPSTREAM_AUDIT_COLUMNS,
    "parser_ladder_results": PARSER_LADDER_COLUMNS,
    "candidate_title_ambiguity": CANDIDATE_AMBIGUITY_COLUMNS,
    "hallucination_audit": HALLUCINATION_COLUMNS,
    "cell_summary": CELL_SUMMARY_COLUMNS,
    "root_cause": ROOT_CAUSE_COLUMNS,
    "gate_reassessment": GATE_REASSESSMENT_COLUMNS,
    "missing_or_blocked": MISSING_COLUMNS,
}


@dataclass(frozen=True)
class CandidateContext:
    candidate_ids: tuple[str, ...]
    serialized_titles: tuple[str, ...]
    prompt_titles: tuple[str, ...]
    analysis_titles: tuple[str, ...]
    title_source: str
    serialization_delimiter_collision: bool


@dataclass(frozen=True)
class ParseOutcome:
    parser_level: str
    parser_id: str
    replay_classification: str
    parse_valid: bool
    strict_valid: bool
    recovered_valid: bool
    items: tuple[str, ...]
    matched_ids: tuple[str, ...]
    matched_titles: tuple[str, ...]
    unmatched_items: tuple[str, ...]
    duplicate_ids: tuple[str, ...]
    ambiguous_items: tuple[str, ...]
    hallucination_count: int
    failure_reason: str
    notes: str


@dataclass(frozen=True)
class BuildOutputs:
    tables: dict[str, list[dict[str, str]]]
    manifest: dict[str, Any]
    report: str


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256_or_missing(root: Path, relative_path: Path) -> str:
    path = root / relative_path
    if not path.exists() or not path.is_file():
        return "missing"
    return sha256_file(path)


def stable_id(prefix: str, *parts: Any, length: int = 16) -> str:
    return f"{prefix}-{sha256_text('||'.join(str(part) for part in parts))[:length]}"


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def fmt_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return ""
    return f"{numerator / denominator:.6f}".rstrip("0").rstrip(".")


def join_pipe(values: Sequence[Any]) -> str:
    return "|".join(str(value) for value in values)


def split_pipe(value: Any) -> list[str]:
    return [part for part in str(value or "").split("|") if part != ""]


def read_csv_rows(root: Path, relative_path: Path) -> list[dict[str, str]]:
    path = root / relative_path
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def csv_text(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue()


def verify_frozen_raw_hashes(root: Path) -> dict[str, str]:
    observed = {relative_path: file_sha256_or_missing(root, Path(relative_path)) for relative_path in EXPECTED_RAW_HASHES}
    mismatches = [
        f"{relative_path}:expected={EXPECTED_RAW_HASHES[relative_path]}:observed={observed[relative_path]}"
        for relative_path in sorted(EXPECTED_RAW_HASHES)
        if observed[relative_path] != EXPECTED_RAW_HASHES[relative_path]
    ]
    if mismatches:
        raise RuntimeError("f6e3a_raw_hash_mismatch:" + ";".join(mismatches))
    return observed


def load_final_raw_by_request(root: Path) -> dict[str, dict[str, Any]]:
    verify_frozen_raw_hashes(root)
    records: dict[str, dict[str, Any]] = {}
    for raw_path in RAW_PATHS.values():
        for record in read_jsonl(root / raw_path):
            if record.get("is_final_attempt") is True:
                records[str(record["request_id"])] = record
    return records


def parse_candidate_titles_from_prompt(prompt: str) -> list[str]:
    candidates: list[list[str]] = []
    for match in re.finditer(r"\[[\s\S]*?\]", str(prompt or "")):
        try:
            parsed = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            continue
        if isinstance(parsed, list) and all(not isinstance(value, (dict, list, tuple)) for value in parsed):
            values = [html.unescape(str(value).strip()) for value in parsed]
            if len(values) >= 10:
                candidates.append(values)
    twenty = [values for values in candidates if len(values) == 20]
    if twenty:
        return twenty[-1]

    numbered = []
    if "Candidates:" in prompt:
        after = prompt.split("Candidates:", 1)[1]
        for line in after.splitlines():
            match = re.match(r"^\s*(\d+)\s*[\.\)\]:-]\s*(.+?)\s*$", line)
            if match:
                numbered.append(html.unescape(match.group(2).strip()))
    return numbered if len(numbered) == 20 else []


def candidate_context(row: Mapping[str, str]) -> CandidateContext:
    candidate_ids = tuple(split_pipe(row.get("candidate_item_ids", "")))
    serialized_titles = tuple(split_pipe(row.get("candidate_item_titles", "")))
    prompt_titles = tuple(parse_candidate_titles_from_prompt(row.get("prompt", "")))
    use_prompt = len(prompt_titles) == len(candidate_ids) and bool(prompt_titles)
    analysis_titles = prompt_titles if use_prompt else serialized_titles
    delimiter_collision = len(serialized_titles) != int(row.get("candidate_count", "0") or 0)
    return CandidateContext(
        candidate_ids=candidate_ids,
        serialized_titles=serialized_titles,
        prompt_titles=prompt_titles,
        analysis_titles=tuple(analysis_titles),
        title_source="prompt_literal" if use_prompt else "serialized_pipe",
        serialization_delimiter_collision=delimiter_collision,
    )


RANKED_LINE_RE = re.compile(r"^\s*(?:[-*]\s*)?(?:rank\s*)?\d+\s*[\.\)\]:-]\s*(.+?)\s*$", re.IGNORECASE)
BULLET_LINE_RE = re.compile(r"^\s*[-*]\s+(.+?)\s*$")


def strip_outer_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text


def strip_markdown_wrappers(value: str) -> str:
    text = value.strip()
    while len(text) >= 4 and text.startswith("**") and text.endswith("**"):
        text = text[2:-2].strip()
    return text


def f6e3_snapshot_extract_items(response_text: str, *, max_items: int = 10) -> list[str]:
    numbered: list[str] = []
    for line in str(response_text or "").splitlines():
        match = RANKED_LINE_RE.match(line)
        if match:
            value = strip_outer_quotes(match.group(1).strip())
            if value:
                numbered.append(value)
                if len(numbered) >= max_items:
                    break
    if numbered:
        return numbered
    literal = parse_python_list_literal_items(response_text)
    if literal:
        return literal[:max_items]
    quoted = [strip_outer_quotes(match) for match in re.findall(r"['\"]([^'\"]{2,})['\"]", str(response_text or ""))]
    if quoted:
        return [value for value in quoted if value][:max_items]
    bullets = []
    for line in str(response_text or "").splitlines():
        match = BULLET_LINE_RE.match(line)
        if match:
            bullets.append(match.group(1).strip())
    return bullets[:max_items]


def parse_python_list_literal_items(response_text: str) -> list[str]:
    matches = re.findall(r"\[[\s\S]*\]", str(response_text or ""))
    if len(matches) != 1:
        return []
    text = matches[0]
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(value[0] if isinstance(value, list) and value else value).strip() for value in parsed]


def upstream_results_map(title_str: Any) -> str:
    if isinstance(title_str, list):
        title_str = title_str[0] if title_str else ""
    text = str(title_str)
    text = text.replace("\\'", "'")
    text = text.replace("Â®", "&reg;")
    text = text.replace("Ã¨", "&egrave;")
    text = text.replace("â˜…", "&#9733;")
    return text


def upstream_parse_items_non_llama(response_text: str) -> tuple[list[str], str]:
    matches = re.findall(r"\[[\s\S]*\]", str(response_text or ""))
    if len(matches) != 1:
        return [], "invalid_format"
    result_str = matches[0]
    try:
        result_str = re.sub(r"'[,]*[ \n\r]+#.*$", "',", result_str, flags=re.MULTILINE)
        result_str = re.sub(r"'[ \n\r]*,[ \n\r]*'", '", "', result_str.replace('"', "'"))
        result_str = re.sub(r"'[ \n,]*\]", '"]', result_str)
        result_str = re.sub(r"\[[ \n]*'", '["', result_str)
        result_str = result_str.replace("['", '["').replace("\\", "\\\\")
        parsed = json.loads(result_str)
    except Exception:
        return [], "json_parse_failed"
    if not isinstance(parsed, list):
        return [], "invalid_schema"
    return [upstream_results_map(value) for value in parsed], "parsed"


def normalize_peripheral(value: str) -> str:
    text = html.unescape(str(value or ""))
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"^\s*(?:[-*]\s*)?(?:rank\s*)?\d+\s*[\.\)\]:-]\s*", "", text, flags=re.IGNORECASE)
    text = strip_outer_quotes(text.strip())
    text = strip_markdown_wrappers(text)
    text = text.strip()
    peripheral = " \t\r\n\"'`*" + string.punctuation
    text = text.strip(peripheral)
    text = re.sub(r"\s+", " ", text)
    return text.casefold().strip()


def normalize_diagnostic(value: str) -> str:
    text = normalize_peripheral(value)
    text = text.translate(str.maketrans({character: " " for character in string.punctuation}))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_exact_index(candidate_ids: Sequence[str], candidate_titles: Sequence[str]) -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    for item_id, title in zip(candidate_ids, candidate_titles):
        index[str(title)] = (str(item_id), str(title))
    return index


def build_multi_index(candidate_ids: Sequence[str], candidate_titles: Sequence[str], normalizer) -> dict[str, list[tuple[str, str]]]:
    index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for item_id, title in zip(candidate_ids, candidate_titles):
        key = normalizer(str(title))
        if key:
            index[key].append((str(item_id), str(title)))
    return dict(index)


def map_items(
    items: Sequence[str],
    candidate_ids: Sequence[str],
    candidate_titles: Sequence[str],
    *,
    mode: str,
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    exact = build_exact_index(candidate_ids, candidate_titles)
    normalized = build_multi_index(candidate_ids, candidate_titles, normalize_peripheral)
    diagnostic = build_multi_index(candidate_ids, candidate_titles, normalize_diagnostic)
    id_index = {str(item_id): (str(item_id), str(title)) for item_id, title in zip(candidate_ids, candidate_titles)}

    matched_ids: list[str] = []
    matched_titles: list[str] = []
    unmatched: list[str] = []
    ambiguous: list[str] = []
    match_types: list[str] = []

    for raw_item in items:
        item = html.unescape(str(raw_item or "").strip())
        resolved: tuple[str, str] | None = None
        match_type = ""

        if mode in {"current_local_strict", "upstream", "normalized"}:
            if item in exact:
                resolved = exact[item]
                match_type = "exact_title"
            else:
                key = normalize_peripheral(item)
                matches = normalized.get(key, [])
                if len(matches) == 1:
                    resolved = matches[0]
                    match_type = "normalized_title"
                elif len(matches) > 1:
                    ambiguous.append(item)
        if resolved is None and mode == "upstream":
            mapped = upstream_map_item(item, exact)
            if mapped is not None:
                resolved = mapped
                match_type = "upstream_regex_title"
        if resolved is None and mode == "permissive":
            resolved, match_type, ambiguous_hit = permissive_map_item(item, candidate_ids, candidate_titles, exact, normalized, diagnostic, id_index)
            if ambiguous_hit:
                ambiguous.append(item)
        if resolved is None:
            unmatched.append(item)
            continue
        matched_ids.append(resolved[0])
        matched_titles.append(resolved[1])
        match_types.append(match_type)

    return matched_ids, matched_titles, unmatched, ambiguous, match_types


def upstream_map_item(item: str, exact: Mapping[str, tuple[str, str]]) -> tuple[str, str] | None:
    if item in exact:
        return exact[item]
    pattern = re.escape(item)
    pattern = re.sub(r"&(?!amp;)", "&amp;", pattern)
    pattern = pattern.replace("'", "(?:'|&quot;)").replace("&quot;", "(?:'|&quot;)")
    compiled = re.compile(pattern, re.IGNORECASE)
    for title, resolved in exact.items():
        match = compiled.search(title)
        if match and match.group() in exact:
            return exact[match.group()]
    return None


def permissive_candidates(item: str) -> list[str]:
    text = str(item or "").strip()
    candidates: list[str] = []
    bold = re.search(r"\*\*(.+?)\*\*", text)
    if bold:
        candidates.append(bold.group(1).strip())
    quoted = re.match(r'^\s*["\'](.+?)["\']', text)
    if quoted:
        candidates.append(quoted.group(1).strip())
    for separator in [" - ", " -- ", " -", " - ", ":", " because ", " as "]:
        if separator in text:
            prefix = text.split(separator, 1)[0].strip()
            if prefix:
                candidates.append(prefix)
    candidates.append(text)
    cleaned: list[str] = []
    for candidate in candidates:
        candidate = strip_outer_quotes(strip_markdown_wrappers(candidate.strip()))
        if candidate and candidate not in cleaned:
            cleaned.append(candidate)
    return cleaned


def permissive_map_item(
    item: str,
    candidate_ids: Sequence[str],
    candidate_titles: Sequence[str],
    exact: Mapping[str, tuple[str, str]],
    normalized: Mapping[str, list[tuple[str, str]]],
    diagnostic: Mapping[str, list[tuple[str, str]]],
    id_index: Mapping[str, tuple[str, str]],
) -> tuple[tuple[str, str] | None, str, bool]:
    text = str(item or "").strip()
    for candidate in permissive_candidates(text):
        if candidate in exact:
            return exact[candidate], "permissive_exact_title", False
        key = normalize_peripheral(candidate)
        matches = normalized.get(key, [])
        if len(matches) == 1:
            return matches[0], "permissive_normalized_title", False
        if len(matches) > 1:
            return None, "", True
        diag_key = normalize_diagnostic(candidate)
        matches = diagnostic.get(diag_key, [])
        if len(matches) == 1:
            return matches[0], "permissive_punctuation_title", False
        if len(matches) > 1:
            return None, "", True
        if len(diag_key) >= 4 and not diag_key.isdigit():
            contained = []
            for title_key, values in diagnostic.items():
                if diag_key and title_key and (diag_key in title_key or title_key in diag_key):
                    contained.extend(values)
            unique = unique_by_id(contained)
            if len(unique) == 1:
                return unique[0], "permissive_unique_substring", False
            if len(unique) > 1:
                return None, "", True
    if text in id_index:
        return id_index[text], "item_id", False
    id_match = re.fullmatch(r"(?:item\s*)?([A-Za-z0-9_:-]{2,})", text, flags=re.IGNORECASE)
    if id_match and id_match.group(1) in id_index:
        return id_index[id_match.group(1)], "item_id", False
    idx_match = re.fullmatch(r"(?:candidate|index|item)?\s*#?\s*(\d{1,2})", text, flags=re.IGNORECASE)
    if idx_match:
        index = int(idx_match.group(1))
        if 1 <= index <= len(candidate_ids):
            return (str(candidate_ids[index - 1]), str(candidate_titles[index - 1])), "one_based_index", False
    return None, "", False


def unique_by_id(values: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for item_id, title in values:
        if item_id not in seen:
            unique.append((item_id, title))
            seen.add(item_id)
    return unique


def parse_with_level(level: str, response_text: str, candidate_ids: Sequence[str], candidate_titles: Sequence[str]) -> ParseOutcome:
    if level == "current_local_strict":
        items = tuple(f6e3_snapshot_extract_items(response_text, max_items=10))
        mode = "current_local_strict"
        parser_id = "official_title_list_top10_parser_local_strict_v1_snapshot"
        replay = "local_snapshot"
        initial_failure = "" if items else "parse_failed"
    elif level == "upstream_exact_or_functional":
        parsed, status = upstream_parse_items_non_llama(response_text)
        items = tuple(parsed[:10])
        mode = "upstream"
        parser_id = "upstream_llm_metrics_non_llama_functional_adapter"
        replay = "upstream_adapter_functional"
        initial_failure = "" if status == "parsed" else status
    elif level == "deterministic_normalized_title_match":
        items = tuple(f6e3_snapshot_extract_items(response_text, max_items=10))
        mode = "normalized"
        parser_id = "deterministic_normalized_title_match_v1"
        replay = "diagnostic_strict_compatible"
        initial_failure = "" if items else "parse_failed"
    elif level == "diagnostic_permissive_recovery":
        items = tuple(f6e3_snapshot_extract_items(response_text, max_items=10))
        mode = "permissive"
        parser_id = "diagnostic_permissive_recovery_v1"
        replay = "diagnostic_only"
        initial_failure = "" if items else "parse_failed"
    else:
        raise ValueError(f"unknown_parser_level:{level}")

    matched_ids, matched_titles, unmatched, ambiguous, match_types = map_items(items, candidate_ids, candidate_titles, mode=mode)
    duplicate_ids = sorted(item_id for item_id, count in Counter(matched_ids).items() if count > 1)
    incomplete = len(items) < 10
    strict_valid = len(items) == 10 and len(matched_ids) == 10 and not duplicate_ids and not unmatched and not ambiguous
    recovered_valid = level == "diagnostic_permissive_recovery" and strict_valid
    if level == "diagnostic_permissive_recovery":
        strict_valid = False
    parse_valid = bool(items) if level != "upstream_exact_or_functional" else initial_failure == ""
    hallucination_count = 0 if ambiguous else len(unmatched)
    failure_parts = []
    if initial_failure:
        failure_parts.append(initial_failure)
    if incomplete:
        failure_parts.append("incomplete_output_length")
    if unmatched:
        failure_parts.append("unmatched_items")
    if ambiguous:
        failure_parts.append("ambiguous_items")
    if duplicate_ids:
        failure_parts.append("duplicate_items")
    if not failure_parts and not strict_valid and not recovered_valid:
        failure_parts.append("not_strict")
    return ParseOutcome(
        parser_level=level,
        parser_id=parser_id,
        replay_classification=replay,
        parse_valid=parse_valid,
        strict_valid=strict_valid,
        recovered_valid=recovered_valid,
        items=items,
        matched_ids=tuple(matched_ids),
        matched_titles=tuple(matched_titles),
        unmatched_items=tuple(unmatched),
        duplicate_ids=tuple(duplicate_ids),
        ambiguous_items=tuple(ambiguous),
        hallucination_count=hallucination_count,
        failure_reason=";".join(failure_parts),
        notes=";".join(match_types[:5]),
    )


def observed_format(response_text: str, current_items: Sequence[str], row: Mapping[str, str], current: Mapping[str, str], context: CandidateContext) -> dict[str, str]:
    text = str(response_text or "")
    lines = [line for line in text.splitlines() if line.strip()]
    has_numbered = any(RANKED_LINE_RE.match(line) for line in lines)
    has_bullets = any(BULLET_LINE_RE.match(line) for line in lines)
    has_json = bool(re.search(r"\{[\s\S]*\}", text))
    has_python_list = bool(re.search(r"\[[\s\S]*\]", text))
    has_quotes = '"' in text or "'" in text
    reasoning_markers = ["because", "based on", "considering", "likely", "interest", "here", "recommend"]
    has_reasoning = any(marker in text.casefold() for marker in reasoning_markers)
    categories: list[str] = []
    if has_numbered:
        categories.append("numbered_candidate_titles")
    if has_bullets:
        categories.append("bullet_candidate_titles")
    if has_quotes:
        categories.append("quoted_candidate_titles")
    if has_python_list:
        categories.append("python_list_or_bracketed_text")
    if has_json:
        categories.append("json_like")
    if has_reasoning:
        categories.append("mixed_titles_and_reasoning")
    if len(current_items) < 10:
        categories.append("partial_ranking")
    if context.serialization_delimiter_collision:
        categories.append("candidate_metadata_serialization_delimiter_collision")
    if not categories:
        categories.append("other_observed_format")
    if any(re.fullmatch(r"\d{1,2}", item.strip()) for item in current_items):
        representation = "position"
    elif any(item in context.candidate_ids for item in current_items):
        representation = "item_id"
    elif current_items:
        representation = "text_mixed" if has_reasoning else "title"
    else:
        representation = "not_recognizable"
    return {
        "record_id": stable_id("F6E3A-FMT", row.get("request_id", "")),
        "request_id": row.get("request_id", ""),
        "source_request_id": row.get("source_request_id", ""),
        "model": row.get("model", ""),
        "strategy_id": row.get("strategy_id", ""),
        "dataset": row.get("dataset", ""),
        "task": row.get("task", ""),
        "pilot_stratum": row.get("pilot_stratum", ""),
        "candidate_count": row.get("candidate_count", ""),
        "serialized_title_count": str(len(context.serialized_titles)),
        "prompt_title_count": str(len(context.prompt_titles)),
        "response_length_chars": str(len(text)),
        "response_line_count": str(len(lines)),
        "done_reason": str(current.get("done_reason", "")),
        "has_numbered_list": bool_text(has_numbered),
        "has_bullets": bool_text(has_bullets),
        "has_json": bool_text(has_json),
        "has_python_list": bool_text(has_python_list),
        "has_quotes": bool_text(has_quotes),
        "has_reasoning_prose": bool_text(has_reasoning),
        "apparent_recommendation_count": str(len(current_items)),
        "representation_type": representation,
        "observed_categories": join_pipe(categories),
        "current_parse_status": current.get("parse_status", ""),
        "current_strict_valid": current.get("strict_valid", ""),
        "current_error": current.get("failure_reason", current.get("parse_status", "")),
        "raw_response_sha256": sha256_text(text),
        "example_excerpt": text[:240].replace("\n", " "),
    }


def candidate_ambiguity_row(row: Mapping[str, str], context: CandidateContext) -> dict[str, str]:
    titles = list(context.analysis_titles)
    exact_counts = Counter(titles)
    normalized_counts = Counter(normalize_peripheral(title) for title in titles if normalize_peripheral(title))
    exact_dupes = sorted(title for title, count in exact_counts.items() if count > 1)
    normalized_dupes = sorted(title for title, count in normalized_counts.items() if count > 1)
    empty_titles = sum(1 for title in titles if not str(title).strip())
    ambiguous_keys = normalized_dupes
    notes = []
    if context.serialization_delimiter_collision:
        notes.append("pipe-delimited candidate_item_titles is non-canonical for at least one title")
    if context.title_source == "prompt_literal":
        notes.append("analysis title map recovered from rendered prompt literal")
    return {
        "request_id": row.get("request_id", ""),
        "model": row.get("model", ""),
        "strategy_id": row.get("strategy_id", ""),
        "dataset": row.get("dataset", ""),
        "candidate_count": row.get("candidate_count", ""),
        "serialized_title_count": str(len(context.serialized_titles)),
        "prompt_title_count": str(len(context.prompt_titles)),
        "serialization_delimiter_collision": bool_text(context.serialization_delimiter_collision),
        "exact_duplicate_titles": str(len(exact_dupes)),
        "normalized_duplicate_titles": str(len(normalized_dupes)),
        "empty_titles": str(empty_titles),
        "titles_with_year": str(sum(1 for title in titles if re.search(r"\((?:19|20)\d{2}\)", title))),
        "titles_with_parentheses": str(sum(1 for title in titles if "(" in title or ")" in title)),
        "titles_with_slash": str(sum(1 for title in titles if "/" in title)),
        "titles_with_apostrophe": str(sum(1 for title in titles if "'" in title or "’" in title)),
        "titles_with_unicode": str(sum(1 for title in titles if any(ord(ch) > 127 for ch in title))),
        "titles_with_punctuation": str(sum(1 for title in titles if any(ch in string.punctuation for ch in title))),
        "titles_with_numeric_prefix": str(sum(1 for title in titles if re.match(r"^\s*\d+[\.\)]", title))),
        "non_injective_title_map": bool_text(bool(exact_dupes or normalized_dupes)),
        "title_to_candidate_mapping": "ambiguous" if ambiguous_keys else ("impossible" if len(titles) != len(context.candidate_ids) else "unique"),
        "ambiguous_keys": join_pipe(ambiguous_keys[:10]),
        "notes": "; ".join(notes),
    }


def ladder_row(row: Mapping[str, str], outcome: ParseOutcome) -> dict[str, str]:
    return {
        "ladder_id": stable_id("F6E3A-LAD", row.get("request_id", ""), outcome.parser_level),
        "request_id": row.get("request_id", ""),
        "model": row.get("model", ""),
        "strategy_id": row.get("strategy_id", ""),
        "dataset": row.get("dataset", ""),
        "parser_level": outcome.parser_level,
        "parser_id": outcome.parser_id,
        "replay_classification": outcome.replay_classification,
        "parse_valid": bool_text(outcome.parse_valid),
        "strict_valid": bool_text(outcome.strict_valid),
        "recovered_valid": bool_text(outcome.recovered_valid),
        "output_length": str(len(outcome.items)),
        "mapped_count": str(len(outcome.matched_ids)),
        "unmatched_count": str(len(outcome.unmatched_items)),
        "duplicate_count": str(len(outcome.duplicate_ids)),
        "hallucination_count": str(outcome.hallucination_count),
        "ambiguous_count": str(len(outcome.ambiguous_items)),
        "incomplete": bool_text(len(outcome.items) < 10),
        "candidate_matches": join_pipe(outcome.matched_ids),
        "unmatched_items": join_pipe(item[:120] for item in outcome.unmatched_items),
        "duplicates": join_pipe(outcome.duplicate_ids),
        "failure_reason": outcome.failure_reason,
        "notes": outcome.notes,
    }


def hallucination_row(row: Mapping[str, str], fmt_row: Mapping[str, str], l2: ParseOutcome, l3: ParseOutcome) -> dict[str, str]:
    format_extra = fmt_row["has_reasoning_prose"] == "true" or any(" - " in item for item in l2.items)
    normalization_difference = len(l2.matched_ids) > len(parse_by_exact_only(l2.items, row))
    alias_count = max(0, len(l3.matched_ids) - len(l2.matched_ids))
    ambiguous = len(l2.ambiguous_items) + len(l3.ambiguous_items)
    genuine = len(l3.unmatched_items) if len(l3.items) >= 10 else 0
    insufficient = len(l2.items) < 10 or fmt_row.get("done_reason") == "length"
    status = "genuine_candidate_hallucination" if genuine else "not_hallucination_format_or_incomplete"
    if ambiguous:
        status = "ambiguous_candidate_mapping"
    return {
        "request_id": row.get("request_id", ""),
        "model": row.get("model", ""),
        "strategy_id": row.get("strategy_id", ""),
        "dataset": row.get("dataset", ""),
        "format_extra_text": bool_text(format_extra),
        "title_normalization_difference": str(int(normalization_difference)),
        "candidate_alias": str(alias_count),
        "ambiguous_candidate": str(ambiguous),
        "genuine_out_of_set_title": str(genuine),
        "generated_explanation_only": bool_text(format_extra and not l3.matched_ids),
        "duplicate_candidate": str(len(l2.duplicate_ids) + len(l3.duplicate_ids)),
        "omitted_candidate": str(max(0, 10 - len(l2.items))),
        "insufficient_output_length": bool_text(insufficient),
        "hallucination_status": status,
        "evidence": l2.failure_reason + ("; " + l3.failure_reason if l3.failure_reason else ""),
    }


def parse_by_exact_only(items: Sequence[str], row: Mapping[str, str]) -> list[str]:
    context = candidate_context(row)
    exact = build_exact_index(context.candidate_ids, context.analysis_titles)
    return [exact[item][0] for item in items if item in exact]


def prompt_contract_rows(request_rows: Sequence[Mapping[str, str]], format_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    by_strategy = defaultdict(list)
    for row in request_rows:
        by_strategy[row["strategy_id"]].append(row)
    formats_by_strategy = defaultdict(list)
    for row in format_rows:
        formats_by_strategy[row["strategy_id"]].append(row)
    requirements = [
        ("output_python_list", "Output as a python list of item titles.", "python list", "source_upstream_system_prompt"),
        ("top10_count", "Return exactly 10 recommended items.", "top 10|10 items|10 candidate", "rendered_prompt_or_contract"),
        ("no_explanation", "Do not explain the reason or include any other words.", "do not explain|without explanation|no explanation", "source_upstream_system_prompt"),
        ("candidate_set_only", "Choose items from the candidate list only.", "candidate", "rendered_prompt_or_contract"),
        ("unique_items", "Return unique candidate items.", "unique", "strict_local_contract"),
        ("ordered_descending", "Rank from most to least likely.", "descending|rank", "rendered_prompt_or_contract"),
    ]
    for strategy_id in [BASE_STRATEGY_ID, LLMRANK_STRATEGY_ID]:
        prompts = [row.get("prompt", "") for row in by_strategy.get(strategy_id, [])]
        joined_prompt = "\n".join(prompts).casefold()
        observed = formats_by_strategy.get(strategy_id, [])
        for req_id, text, pattern, source in requirements:
            contains = bool(re.search(pattern, joined_prompt, flags=re.IGNORECASE))
            if req_id == "output_python_list":
                compliant = sum(1 for row in observed if row["has_python_list"] == "true")
            elif req_id == "top10_count":
                compliant = sum(1 for row in observed if int(row["apparent_recommendation_count"] or 0) == 10)
            elif req_id == "no_explanation":
                compliant = sum(1 for row in observed if row["has_reasoning_prose"] == "false")
            elif req_id == "candidate_set_only":
                compliant = sum(1 for row in observed if "other_observed_format" not in row["observed_categories"])
            elif req_id == "unique_items":
                compliant = len(observed)
            else:
                compliant = sum(1 for row in observed if row["has_numbered_list"] == "true" or row["has_python_list"] == "true")
            rows.append(
                {
                    "contract_id": stable_id("F6E3A-CON", strategy_id, req_id),
                    "strategy_id": strategy_id,
                    "requirement_id": req_id,
                    "requirement_text": text,
                    "source_contract": source,
                    "rendered_prompt_contains_requirement": bool_text(contains),
                    "observed_compliant_count": str(compliant),
                    "observed_noncompliant_count": str(len(observed) - compliant),
                    "observed_behavior": observed_behavior_summary(observed, req_id),
                    "contract_gap": bool_text(not contains and source == "source_upstream_system_prompt"),
                    "notes": "Upstream system prompt requirement is not necessarily present in rendered F6E3 prompt." if source == "source_upstream_system_prompt" else "",
                }
            )
    return rows


def observed_behavior_summary(format_rows: Sequence[Mapping[str, str]], requirement_id: str) -> str:
    if not format_rows:
        return "no_outputs"
    if requirement_id == "output_python_list":
        return f"python_list={sum(1 for row in format_rows if row['has_python_list'] == 'true')};numbered={sum(1 for row in format_rows if row['has_numbered_list'] == 'true')}"
    if requirement_id == "no_explanation":
        return f"reasoning={sum(1 for row in format_rows if row['has_reasoning_prose'] == 'true')}"
    if requirement_id == "top10_count":
        counts = Counter(row["apparent_recommendation_count"] for row in format_rows)
        return stable_json(dict(sorted(counts.items())))
    return stable_json(dict(sorted(Counter(row["observed_categories"] for row in format_rows).items())))


def upstream_parser_audit_rows(root: Path) -> list[dict[str, str]]:
    source_hash = file_sha256_or_missing(root, UPSTREAM_LLM_METRICS)
    return [
        {
            "audit_id": "F6E3A-UP-001",
            "source_file": UPSTREAM_LLM_METRICS.as_posix(),
            "function": "read_llm_results",
            "call_path": "read_llm_results -> regex bracket extraction -> json.loads -> resultsMap -> mapListTitle2item_id",
            "input_contract": "JSON file containing result, target and user fields; non-llama branch expects exactly one bracketed python-style list.",
            "output_contract": "list of 10 titles mapped to item ids, with zero used for unmapped titles.",
            "normalization": "quote repair, backslash escaping, resultsMap replacements for selected escaped characters.",
            "fallback": "non-llama branch has no numbered-list fallback; llama branch has separate numbered-line regex.",
            "error_handling": "invalid result format raises ValueError at batch level; invalid length is truncated or padded before mapping.",
            "local_difference": "F6E3 local parser accepts numbered lines but maps only exact/normalized full candidate titles.",
            "replay_classification": "upstream_adapter_functional",
            "source_sha256": source_hash,
        },
        {
            "audit_id": "F6E3A-UP-002",
            "source_file": UPSTREAM_LLM_METRICS.as_posix(),
            "function": "mapTitle2itemid/create_pattern/resultsMap/calHallucination",
            "call_path": "title string -> resultsMap -> exact title lookup -> regex-compatible lookup -> zero for missing -> hallucination if zero",
            "input_contract": "candidate or metadata title string",
            "output_contract": "integer item id or zero",
            "normalization": "limited HTML/entity and quote compatibility; no semantic fuzzy matching.",
            "fallback": "regex pattern with apostrophe/entity handling against known title corpus.",
            "error_handling": "unmapped items become zero and later count as hallucination.",
            "local_difference": "F6E3 scoped mapping reports unmapped candidate titles before metric computation; F6E3A keeps hallucination separate from format failures.",
            "replay_classification": "upstream_adapter_functional",
            "source_sha256": source_hash,
        },
    ]


def root_cause_row(
    row: Mapping[str, str],
    fmt_row: Mapping[str, str],
    context: CandidateContext,
    outcomes: Mapping[str, ParseOutcome],
    hallucination: Mapping[str, str],
) -> dict[str, str]:
    l0 = outcomes["current_local_strict"]
    l1 = outcomes["upstream_exact_or_functional"]
    l2 = outcomes["deterministic_normalized_title_match"]
    l3 = outcomes["diagnostic_permissive_recovery"]
    if l1.strict_valid or l2.strict_valid:
        cause = "local_parser_too_strict"
        correction = "repeat_analysis_only_with_corrected_parser"
        inference = "false"
        confidence = "high"
        evidence = "strict-compatible parser recovers 10 unique in-candidate titles"
    elif l3.ambiguous_items:
        cause = "candidate_title_ambiguity"
        correction = "repair_candidate_metadata_first"
        inference = "false"
        confidence = "medium"
        evidence = "permissive parser encountered ambiguous candidate mapping"
    elif hallucination["genuine_out_of_set_title"] != "0":
        cause = "genuine_candidate_hallucination"
        correction = "keep_exact_prompt_cells_blocked"
        inference = "true"
        confidence = "medium"
        evidence = "permissive parser still leaves out-of-candidate titles"
    elif len(l0.items) < 10 or fmt_row.get("done_reason") == "length":
        cause = "incomplete_model_output"
        correction = "run_adapted_output_contract_pilot"
        inference = "true"
        confidence = "high"
        evidence = f"output_length={len(l0.items)};done_reason={fmt_row.get('done_reason')}"
    elif fmt_row.get("has_reasoning_prose") == "true":
        cause = "model_ignored_output_contract"
        correction = "run_adapted_output_contract_pilot"
        inference = "true"
        confidence = "high"
        evidence = "model returned prose/explanations rather than a strict top-10 title list"
    elif not l1.parse_valid and fmt_row.get("has_python_list") == "false":
        cause = "prompt_output_underspecified"
        correction = "run_adapted_output_contract_pilot"
        inference = "true"
        confidence = "medium"
        evidence = "upstream non-llama parser expects python list but rendered output is not a python list"
    elif context.serialization_delimiter_collision and len(context.serialized_titles) != len(context.candidate_ids):
        cause = "candidate_metadata_problem"
        correction = "repair_candidate_metadata_serialization_for_derived_tables"
        inference = "false"
        confidence = "high"
        evidence = "candidate_item_titles pipe serialization does not preserve exactly 20 titles"
    else:
        cause = "mixed_or_multiple_causes"
        correction = "keep_exact_prompt_cells_blocked"
        inference = "indeterminate"
        confidence = "medium"
        evidence = f"l0={l0.failure_reason};l1={l1.failure_reason};l2={l2.failure_reason};l3={l3.failure_reason}"
    return {
        "request_id": row.get("request_id", ""),
        "model": row.get("model", ""),
        "strategy_id": row.get("strategy_id", ""),
        "dataset": row.get("dataset", ""),
        "primary_cause": cause,
        "evidence": evidence,
        "confidence": confidence,
        "safe_correction": correction,
        "whether_new_inference_is_required": inference,
    }


def cell_summary_rows(
    request_rows: Sequence[Mapping[str, str]],
    ladder_rows: Sequence[Mapping[str, str]],
    format_rows: Sequence[Mapping[str, str]],
    hallucination_rows: Sequence[Mapping[str, str]],
    root_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    ladder_by_request_level = {(row["request_id"], row["parser_level"]): row for row in ladder_rows}
    format_by_request = {row["request_id"]: row for row in format_rows}
    hallucination_by_request = {row["request_id"]: row for row in hallucination_rows}
    root_by_request = {row["request_id"]: row for row in root_rows}
    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in request_rows:
        grouped[(row["model"], row["strategy_id"])].append(row)
    rows: list[dict[str, str]] = []
    for (model, strategy_id), members in sorted(grouped.items()):
        request_ids = [row["request_id"] for row in members]
        def count_level(level: str, column: str, expected: str = "true") -> int:
            return sum(1 for request_id in request_ids if ladder_by_request_level[(request_id, level)][column] == expected)

        output_distribution = Counter(format_by_request[request_id]["observed_categories"] for request_id in request_ids)
        root_distribution = Counter(root_by_request[request_id]["primary_cause"] for request_id in request_ids)
        rows.append(
            {
                "cell_id": stable_id("F6E3A-CELL", model, strategy_id),
                "model": model,
                "strategy_id": strategy_id,
                "logical_requests": str(len(members)),
                "current_strict_valid": str(count_level("current_local_strict", "strict_valid")),
                "upstream_valid": str(count_level("upstream_exact_or_functional", "strict_valid")),
                "normalized_valid": str(count_level("deterministic_normalized_title_match", "strict_valid")),
                "permissive_recoverable": str(count_level("diagnostic_permissive_recovery", "recovered_valid")),
                "genuine_hallucinations": str(sum(int(hallucination_by_request[request_id]["genuine_out_of_set_title"]) for request_id in request_ids)),
                "partial_rankings": str(sum(1 for request_id in request_ids if int(ladder_by_request_level[(request_id, "current_local_strict")]["output_length"]) < 10)),
                "duplicate_rankings": str(sum(1 for request_id in request_ids if int(ladder_by_request_level[(request_id, "diagnostic_permissive_recovery")]["duplicate_count"]) > 0)),
                "ambiguous_mappings": str(sum(1 for request_id in request_ids if int(ladder_by_request_level[(request_id, "diagnostic_permissive_recovery")]["ambiguous_count"]) > 0)),
                "output_format_distribution": stable_json(dict(sorted(output_distribution.items()))),
                "root_cause_distribution": stable_json(dict(sorted(root_distribution.items()))),
                "gate_population": "upstream_or_normalized_strict_only",
                "notes": "Pilot cell summary; no permissive diagnostic recovery is counted as strict.",
            }
        )
    return rows


def gate_reassessment_rows(ladder_rows: Sequence[Mapping[str, str]], total_requests: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    populations = [
        ("current_local_original", "current_local_strict", "strict_valid"),
        ("upstream_compatible_strict", "upstream_exact_or_functional", "strict_valid"),
        ("normalized_strict_compatible", "deterministic_normalized_title_match", "strict_valid"),
        ("permissive_diagnostic_not_gate_eligible", "diagnostic_permissive_recovery", "recovered_valid"),
    ]
    for population, level, column in populations:
        eligible = sum(1 for row in ladder_rows if row["parser_level"] == level and row[column] == "true")
        gate_eligible = not population.startswith("permissive")
        pass_gate = gate_eligible and eligible == total_requests
        rows.append(
            {
                "gate_id": stable_id("F6E3A-GATE", population),
                "population": population,
                "eligible_strict_count": str(eligible),
                "total_requests": str(total_requests),
                "strict_valid_rate": fmt_ratio(eligible, total_requests),
                "threshold_source": "F6E2/F6E3 frozen parser gate: strict_invalid=0",
                "gate_status": "pass" if pass_gate else ("not_gate_eligible" if not gate_eligible else "fail"),
                "decision": "eligible_for_gate" if gate_eligible else "diagnostic_only",
                "notes": "Permissive recovery cannot pass the scientific gate." if not gate_eligible else "",
            }
        )
    return rows


def final_decision(root_rows: Sequence[Mapping[str, str]], gate_rows: Sequence[Mapping[str, str]], candidate_rows: Sequence[Mapping[str, str]]) -> str:
    gate_pass = any(row["population"] in {"upstream_compatible_strict", "normalized_strict_compatible"} and row["gate_status"] == "pass" for row in gate_rows)
    if gate_pass:
        return "f6e3a_parser_recovery_complete_gate_passed"
    if any(row["title_to_candidate_mapping"] == "ambiguous" for row in candidate_rows):
        return "f6e3a_parser_recovery_complete_with_ambiguous_mapping_blocks"
    if any(row["title_to_candidate_mapping"] == "impossible" for row in candidate_rows):
        return "f6e3a_blocked_candidate_data_inconsistency"
    return "f6e3a_parser_recovery_complete_gate_failed_model_outputs"


def recommendation_for_decision(decision: str) -> str:
    if decision == "f6e3a_parser_recovery_complete_gate_passed":
        return "proceed_to_full_run"
    if decision == "f6e3a_parser_recovery_complete_with_ambiguous_mapping_blocks":
        return "repair_candidate_metadata_first"
    if decision == "f6e3a_blocked_candidate_data_inconsistency":
        return "repair_candidate_metadata_first"
    if decision == "f6e3a_parser_recovery_complete_gate_failed_model_outputs":
        return "run_adapted_output_contract_pilot"
    return "keep_exact_prompt_cells_blocked"


def missing_or_blocked_rows(decision: str, root_rows: Sequence[Mapping[str, str]], candidate_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows = [
        {
            "block_id": "F6E3A-BLOCK-NO-NEW-INFERENCE",
            "status": "not_executed_by_policy",
            "severity": "info",
            "affected_scope": "Ollama generate/version/tags",
            "evidence": "F6E3A offline plan",
            "notes": "No runtime gate or generation endpoint is called in this phase.",
        }
    ]
    if any(row["serialization_delimiter_collision"] == "true" for row in candidate_rows):
        rows.append(
            {
                "block_id": "F6E3A-BLOCK-CANDIDATE-SERIALIZATION",
                "status": "documented_candidate_metadata_problem",
                "severity": "high",
                "affected_scope": "phase_f6e3_request_manifest.csv candidate_item_titles delimiter",
                "evidence": "candidate_title_ambiguity table",
                "notes": "Pipe-delimited title serialization is unsafe for titles containing pipe characters; prompt literals are used for diagnostic replay.",
            }
        )
    if decision == "f6e3a_parser_recovery_complete_gate_failed_model_outputs":
        rows.append(
            {
                "block_id": "F6E3A-BLOCK-PARSER-GATE",
                "status": decision,
                "severity": "high",
                "affected_scope": "exact-prompt Base/LLMRank pilot",
                "evidence": "phase_f6e3a_gate_reassessment.csv",
                "notes": "No strict-compatible parser population reaches 32/32.",
            }
        )
    if any(row["title_to_candidate_mapping"] == "ambiguous" for row in candidate_rows):
        rows.append(
            {
                "block_id": "F6E3A-BLOCK-AMBIGUOUS-TITLES",
                "status": "ambiguous_mapping_blocks",
                "severity": "medium",
                "affected_scope": "candidate title maps",
                "evidence": "phase_f6e3a_candidate_title_ambiguity.csv",
                "notes": "Ambiguous title maps cannot be resolved arbitrarily.",
            }
        )
    return rows


def input_hashes(root: Path, request_rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    paths = [
        F6E3_REQUEST_MANIFEST,
        F6E3_INSTANCE_RESULTS,
        F6E3_PARSER_VALIDITY,
        F6E1_FREEZE,
        F6E2_FREEZE,
        F6E3_FREEZE,
        F6E2_PILOT_DESIGN,
        F6E1_PROMPT_MODULE,
        LOCAL_F6E3_MODULE,
        LOCAL_OUTPUT_PARSER,
        UPSTREAM_LLM_METRICS,
        F2_PROTECTED_VALIDATION,
        F6D4A_INPUT_HASHES,
        F6D4A_PROTECTED_VERIFICATION,
        F6D3_LASTFM_MANIFEST,
    ]
    paths.extend(Path(row["source_manifest_path"]) for row in request_rows if row.get("source_manifest_path"))
    paths.extend(RAW_PATHS.values())
    unique = sorted({path.as_posix() for path in paths})
    return {path: file_sha256_or_missing(root, Path(path)) for path in unique}


def table_hashes(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, str]:
    return {name: sha256_text(csv_text(rows, TABLE_COLUMNS[name])) for name, rows in sorted(tables.items())}


def protected_summary(root: Path) -> dict[str, Any]:
    f2_rows = read_csv_rows(root, F2_PROTECTED_VALIDATION)
    f2_changed = sum(1 for row in f2_rows if row.get("status") != "unchanged")
    f6d4a_rows = read_csv_rows(root, F6D4A_PROTECTED_VERIFICATION)
    summary = {
        "f2_checked": len(f2_rows),
        "f2_changed": f2_changed,
        "f6d4a_groups_checked": len(f6d4a_rows),
        "f6d4a_total_changed": sum(int(row.get("changed", "0") or 0) for row in f6d4a_rows),
    }
    return summary


def build_all(root: Path) -> BuildOutputs:
    raw_hash = verify_frozen_raw_hashes(root)
    request_rows = read_csv_rows(root, F6E3_REQUEST_MANIFEST)
    if len(request_rows) != 32:
        raise RuntimeError(f"f6e3a_expected_32_requests_observed_{len(request_rows)}")
    current_parser_rows = {row["request_id"]: row for row in read_csv_rows(root, F6E3_PARSER_VALIDITY)}
    raw_by_request = load_final_raw_by_request(root)
    if len(raw_by_request) != 32:
        raise RuntimeError(f"f6e3a_expected_32_raw_observed_{len(raw_by_request)}")

    output_rows: list[dict[str, str]] = []
    candidate_rows: list[dict[str, str]] = []
    ladder_rows: list[dict[str, str]] = []
    hallucination_rows: list[dict[str, str]] = []
    root_rows: list[dict[str, str]] = []

    for row in request_rows:
        request_id = row["request_id"]
        raw = raw_by_request[request_id]
        response_text = str(raw.get("raw_response", ""))
        context = candidate_context(row)
        current_items = f6e3_snapshot_extract_items(response_text, max_items=10)
        current = dict(current_parser_rows.get(request_id, {}))
        current["done_reason"] = str(raw.get("done_reason", ""))
        fmt_row = observed_format(response_text, current_items, row, current, context)
        output_rows.append(fmt_row)
        candidate_rows.append(candidate_ambiguity_row(row, context))

        outcomes: dict[str, ParseOutcome] = {}
        for level in [
            "current_local_strict",
            "upstream_exact_or_functional",
            "deterministic_normalized_title_match",
            "diagnostic_permissive_recovery",
        ]:
            titles_for_level = context.serialized_titles if level == "current_local_strict" else context.analysis_titles
            outcomes[level] = parse_with_level(level, response_text, context.candidate_ids, titles_for_level)
        for outcome in outcomes.values():
            ladder_rows.append(ladder_row(row, outcome))
        hall_row = hallucination_row(row, fmt_row, outcomes["deterministic_normalized_title_match"], outcomes["diagnostic_permissive_recovery"])
        hallucination_rows.append(hall_row)
        root_rows.append(root_cause_row(row, fmt_row, context, outcomes, hall_row))

    prompt_rows = prompt_contract_rows(request_rows, output_rows)
    upstream_rows = upstream_parser_audit_rows(root)
    cell_rows = cell_summary_rows(request_rows, ladder_rows, output_rows, hallucination_rows, root_rows)
    gate_rows = gate_reassessment_rows(ladder_rows, len(request_rows))
    decision = final_decision(root_rows, gate_rows, candidate_rows)
    missing_rows = missing_or_blocked_rows(decision, root_rows, candidate_rows)

    tables = {
        "output_format_inventory": output_rows,
        "prompt_output_contract": prompt_rows,
        "upstream_parser_audit": upstream_rows,
        "parser_ladder_results": ladder_rows,
        "candidate_title_ambiguity": candidate_rows,
        "hallucination_audit": hallucination_rows,
        "cell_summary": cell_rows,
        "root_cause": root_rows,
        "gate_reassessment": gate_rows,
        "missing_or_blocked": missing_rows,
    }
    hashes = table_hashes(tables)
    inputs = input_hashes(root, request_rows)
    protected = protected_summary(root)
    content = {
        "decision": decision,
        "input_hashes": inputs,
        "raw_hashes": raw_hash,
        "table_hashes": hashes,
        "protected_summary": protected,
        "recommendation": recommendation_for_decision(decision),
    }
    manifest = {
        "phase": PHASE_ID,
        "decision": decision,
        "frozen_timestamp_utc": FROZEN_TIMESTAMP_UTC,
        "raw_f6e3_hashes": raw_hash,
        "prompt_freeze_hash": inputs.get(F6E1_FREEZE.as_posix(), "missing"),
        "preflight_freeze_hash": inputs.get(F6E2_FREEZE.as_posix(), "missing"),
        "pilot_freeze_hash": inputs.get(F6E3_FREEZE.as_posix(), "missing"),
        "parser_source_hashes": {
            LOCAL_F6E3_MODULE.as_posix(): inputs.get(LOCAL_F6E3_MODULE.as_posix(), "missing"),
            LOCAL_OUTPUT_PARSER.as_posix(): inputs.get(LOCAL_OUTPUT_PARSER.as_posix(), "missing"),
        },
        "upstream_parser_hashes": {
            UPSTREAM_LLM_METRICS.as_posix(): inputs.get(UPSTREAM_LLM_METRICS.as_posix(), "missing"),
        },
        "candidate_data_hashes": {
            path: value
            for path, value in inputs.items()
            if path.endswith("_manifest.csv") or path == F6E3_REQUEST_MANIFEST.as_posix()
        },
        "input_hashes": inputs,
        "table_hashes": hashes,
        "scientific_content_hash": sha256_text(stable_json(content)),
        "no_new_requests_sent": True,
        "network_calls_made": 0,
        "ollama_calls_made": 0,
        "protected_summary": protected,
        "recommendation": recommendation_for_decision(decision),
    }
    report = build_report(manifest, tables)
    return BuildOutputs(tables=tables, manifest=manifest, report=report)


def build_report(manifest: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping[str, str]]]) -> str:
    total = len(tables["output_format_inventory"])
    strict_current = sum(1 for row in tables["parser_ladder_results"] if row["parser_level"] == "current_local_strict" and row["strict_valid"] == "true")
    strict_upstream = sum(1 for row in tables["parser_ladder_results"] if row["parser_level"] == "upstream_exact_or_functional" and row["strict_valid"] == "true")
    strict_norm = sum(1 for row in tables["parser_ladder_results"] if row["parser_level"] == "deterministic_normalized_title_match" and row["strict_valid"] == "true")
    permissive = sum(1 for row in tables["parser_ladder_results"] if row["parser_level"] == "diagnostic_permissive_recovery" and row["recovered_valid"] == "true")
    causes = Counter(row["primary_cause"] for row in tables["root_cause"])
    sections = [
        ("Initial repository state", "Captured in the operator response to keep builder output deterministic; F6E3A builder itself does not call git."),
        ("Frozen-input verification", f"F6E3 raw hashes matched frozen values. Protected summary: {stable_json(manifest['protected_summary'])}."),
        ("Raw-output inventory", f"Frozen F6E3 responses analyzed: {total}; no new requests sent."),
        ("Observed format classification", stable_json(dict(sorted(Counter(row["observed_categories"] for row in tables["output_format_inventory"]).items())))),
        ("Prompt-output contract audit", "Rendered prompts were compared against upstream system-output requirements and observed response formats."),
        ("Upstream parser discovery", "Upstream evidence is original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py; replay classification is upstream_adapter_functional."),
        ("Upstream parser replay", f"Upstream-compatible strict valid: {strict_upstream}/{total}."),
        ("Parser ladder", f"Current strict {strict_current}/{total}; normalized strict {strict_norm}/{total}; permissive diagnostic recovered {permissive}/{total}."),
        ("Candidate-title ambiguity", stable_json(dict(sorted(Counter(row["title_to_candidate_mapping"] for row in tables["candidate_title_ambiguity"]).items())))),
        ("Hallucination audit", stable_json(dict(sorted(Counter(row["hallucination_status"] for row in tables["hallucination_audit"]).items())))),
        ("Qwen2 Base analysis", cell_text(tables["cell_summary"], QWEN2_MODEL, BASE_STRATEGY_ID)),
        ("Mistral Base analysis", cell_text(tables["cell_summary"], MISTRAL_MODEL, BASE_STRATEGY_ID)),
        ("Qwen2 LLMRank analysis", cell_text(tables["cell_summary"], QWEN2_MODEL, LLMRANK_STRATEGY_ID)),
        ("Mistral LLMRank analysis", cell_text(tables["cell_summary"], MISTRAL_MODEL, LLMRANK_STRATEGY_ID)),
        ("Root-cause attribution", stable_json(dict(sorted(causes.items())))),
        ("Parser correction", "No F6E3 parser correction is applied by F6E3A unless strict-compatible recovery is demonstrated; permissive recovery remains diagnostic."),
        ("Tests added", "tests/test_plan_f6e3a_parser_recovery_audit.py covers offline purity, parser ladder, raw invariance and deterministic writes."),
        ("Analysis-only regeneration", "F6E3A writes only phase_f6e3a_* report, tables and manifest."),
        ("Gate reassessment", stable_json({row["population"]: row["gate_status"] for row in tables["gate_reassessment"]})),
        ("Determinism", "Builder uses frozen timestamps and sorted/hash-stable table construction."),
        ("Targeted tests", "Recorded in operator response after execution."),
        ("Full-suite result", "Recorded in operator response after execution."),
        ("Protected-input verification", stable_json(manifest["protected_summary"])),
        ("Parser-recovery manifest", f"Scientific content hash: {manifest['scientific_content_hash']}."),
        ("Missing or blocked evidence", stable_json(dict(sorted(Counter(row["status"] for row in tables["missing_or_blocked"]).items())))),
        ("Final decision", f"`{manifest['decision']}`."),
        ("Recommended next action", f"`{manifest['recommendation']}`."),
        ("Final git state", "Captured in operator response; no staging or commit is performed by the builder."),
        ("Files created or modified", "F6E3A creates only report/phase_f6e3a_parser_recovery_audit.md, results/tables/phase_f6e3a_*.csv, and results/manifests/phase_f6e3a_parser_recovery_freeze.json."),
        ("Remaining limitations", "No upstream batch execution with original data paths is performed; the adapter is functional and candidate-scoped for frozen F6E3 rows."),
    ]
    lines = ["# Phase F6E3A - Exact-Prompt Output Contract And Parser Recovery Audit", "", f"Decision: `{manifest['decision']}`.", ""]
    for index, (title, body) in enumerate(sections, start=1):
        lines.extend([f"## {index}. {title}", "", body, ""])
    return "\n".join(lines)


def cell_text(rows: Sequence[Mapping[str, str]], model: str, strategy_id: str) -> str:
    for row in rows:
        if row["model"] == model and row["strategy_id"] == strategy_id:
            return (
                f"logical={row['logical_requests']}; current={row['current_strict_valid']}; "
                f"upstream={row['upstream_valid']}; normalized={row['normalized_valid']}; "
                f"permissive={row['permissive_recoverable']}; causes={row['root_cause_distribution']}"
            )
    return "cell_missing"


def write_outputs(outputs: BuildOutputs, root: Path) -> dict[str, str]:
    written: dict[str, str] = {}
    for table_name, relative_path in OUTPUT_TABLES.items():
        write_csv(root / relative_path, outputs.tables[table_name], TABLE_COLUMNS[table_name])
        written[relative_path.as_posix()] = sha256_file(root / relative_path)
    (root / MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST_PATH).write_text(json.dumps(outputs.manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    written[MANIFEST_PATH.as_posix()] = sha256_file(root / MANIFEST_PATH)
    (root / REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / REPORT_PATH).write_text(outputs.report, encoding="utf-8")
    written[REPORT_PATH.as_posix()] = sha256_file(root / REPORT_PATH)
    return written


def run(root: Path) -> dict[str, str]:
    return write_outputs(build_all(root), root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build offline F6E3A parser recovery audit artifacts.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    written = run(args.root)
    print(json.dumps({"written": written}, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
