"""Parse textual LLM recommendation outputs into ranked records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re
import string

import pandas as pd

try:  # pragma: no cover - exercised only when rapidfuzz is installed.
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover - optional dependency.
    fuzz = None
    process = None


DEFAULT_ITEM_ID_COLUMN = "item_id"
DEFAULT_ITEM_TITLE_COLUMN = "item_title"
FALLBACK_TITLE_COLUMNS = ("candidate_item_title", "title", "name")
MATCH_ITEM_ID = "item_id"
MATCH_EXACT = "exact"
MATCH_NORMALIZED = "normalized"
MATCH_FUZZY = "fuzzy"
MATCH_UNMATCHED = "unmatched"

_LIST_ITEM_RE = re.compile(
    r"^\s*(?:[-*•]\s+|(?:rank\s*)?\d+\s*[\.\)\]:\-–—]\s+)(?P<item>.+)$",
    flags=re.IGNORECASE,
)
_RANK_PREFIX_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:(?:rank|item)\s*)?\d+\s*[\.\)\]:\-–—]\s*",
    flags=re.IGNORECASE,
)
_LEADING_FILLER_RE = re.compile(
    r"^\s*(?:and|then|also|finally|recommendation|recommendations)\s*[:\-]?\s+",
    flags=re.IGNORECASE,
)
_HEADER_PREFIX_RE = re.compile(
    r"\b(?:recommend|recommendations|items|choices|top|list)\b",
    flags=re.IGNORECASE,
)
_QUOTED_RE = re.compile(
    r'"([^"]+)"|' + r"'([^']+)'" + r"|“([^”]+)”|‘([^’]+)’"
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


@dataclass(frozen=True)
class MatchResult:
    """Catalog matching result for one parsed recommendation string."""

    item_id: Any
    item_title: Any
    match_type: str
    match_score: float | None


@dataclass(frozen=True)
class _CatalogEntry:
    item_id: Any
    item_title: Any


@dataclass(frozen=True)
class _CatalogIndex:
    entries: tuple[_CatalogEntry, ...]
    id_exact: dict[str, _CatalogEntry]
    id_lower: dict[str, _CatalogEntry | None]
    title_exact: dict[str, _CatalogEntry]
    title_normalized: dict[str, _CatalogEntry]


def normalize_recommendation_text(text: object) -> str:
    """Normalize recommendation text for matching.

    Normalization lowercases text, removes leading ranking/list markers,
    replaces punctuation with spaces, collapses whitespace, and trims the
    result. This function does not perform fuzzy matching.
    """

    if text is None or pd.isna(text):
        return ""

    value = _remove_ranking_prefix(str(text)).lower()
    punctuation_table = str.maketrans(
        {character: " " for character in string.punctuation}
    )
    value = value.translate(punctuation_table)
    value = re.sub(r"[“”‘’•–—]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def parse_recommendation_items(
    output_text: object,
    *,
    max_items: int | None = None,
) -> list[str]:
    """Extract ranked recommendation strings from noisy LLM output text."""

    if output_text is None or pd.isna(output_text):
        return []

    text = str(output_text).strip()
    if not text:
        return []

    candidates: list[str] = []
    for line in text.splitlines():
        match = _LIST_ITEM_RE.match(line.strip())
        if match:
            candidates.extend(_items_from_segment(match.group("item")))

    if not candidates:
        candidates.extend(_items_from_list_literal_text(text))

    if not candidates:
        quoted_items = [_clean_candidate(match[0]) for match in _quoted_parts(text)]
        candidates.extend(quoted_items)

    if not candidates:
        candidates.extend(_items_from_segment(text, allow_header=True))

    cleaned: list[str] = []
    for candidate in candidates:
        value = _clean_candidate(candidate)
        if value:
            cleaned.append(value)
        if max_items is not None and len(cleaned) >= max_items:
            break

    return cleaned


def parse_llm_output(
    output_text: object,
    *,
    user_id: Any,
    dataset: str,
    model_name: str,
    run_id: str,
    catalog: pd.DataFrame | None = None,
    max_items: int | None = None,
    item_id_col: str = DEFAULT_ITEM_ID_COLUMN,
    item_title_col: str = DEFAULT_ITEM_TITLE_COLUMN,
    enable_fuzzy: bool = False,
    fuzzy_threshold: float = 90.0,
    source_file: str | None = None,
    prompt_variant: str | None = None,
    candidate_ordering: str | None = None,
    history_length: int | None = None,
) -> pd.DataFrame:
    """Parse one LLM output string into canonical recommendation records.

    If a catalog is supplied, parsed text is matched by item id, exact title,
    normalized title, and optionally fuzzy title matching. Unmatched parsed text
    is retained with ``match_type == "unmatched"`` and a missing
    ``recommended_item_id``; it is not converted into a valid catalog item.
    """

    parsed_items = parse_recommendation_items(output_text, max_items=max_items)
    catalog_index = _build_catalog_index(
        catalog,
        item_id_col=item_id_col,
        item_title_col=item_title_col,
    )

    rows: list[dict[str, Any]] = []
    for rank, item_text in enumerate(parsed_items, start=1):
        match = match_catalog_item(
            item_text,
            catalog_index=catalog_index,
            enable_fuzzy=enable_fuzzy,
            fuzzy_threshold=fuzzy_threshold,
        )
        row: dict[str, Any] = {
            "user_id": user_id,
            "model_name": model_name,
            "dataset": dataset,
            "rank": rank,
            "recommended_item_id": match.item_id,
            "recommended_item_text": item_text,
            "run_id": run_id,
            "raw_output": output_text,
            "match_type": match.match_type,
            "match_score": match.match_score,
            "catalog_item_title": match.item_title,
        }
        if source_file is not None:
            row["source_file"] = source_file
        if prompt_variant is not None:
            row["prompt_variant"] = prompt_variant
        if candidate_ordering is not None:
            row["candidate_ordering"] = candidate_ordering
        if history_length is not None:
            row["history_length"] = history_length
        rows.append(row)

    return pd.DataFrame(rows, columns=_output_columns(rows))


def match_catalog_item(
    candidate_text: object,
    *,
    catalog: pd.DataFrame | None = None,
    catalog_index: _CatalogIndex | None = None,
    item_id_col: str = DEFAULT_ITEM_ID_COLUMN,
    item_title_col: str = DEFAULT_ITEM_TITLE_COLUMN,
    enable_fuzzy: bool = False,
    fuzzy_threshold: float = 90.0,
) -> MatchResult:
    """Match one parsed recommendation string against a catalog."""

    if catalog_index is None:
        catalog_index = _build_catalog_index(
            catalog,
            item_id_col=item_id_col,
            item_title_col=item_title_col,
        )

    if catalog_index is None:
        return MatchResult(pd.NA, pd.NA, MATCH_UNMATCHED, None)

    raw_text = _clean_candidate(candidate_text)
    title_text = _title_text_for_matching(raw_text)

    item_id_match = _match_by_item_id(raw_text, catalog_index)
    if item_id_match is not None:
        return MatchResult(
            item_id_match.item_id,
            item_id_match.item_title,
            MATCH_ITEM_ID,
            100.0,
        )

    exact_match = catalog_index.title_exact.get(title_text)
    if exact_match is not None:
        return MatchResult(
            exact_match.item_id,
            exact_match.item_title,
            MATCH_EXACT,
            100.0,
        )

    normalized = normalize_recommendation_text(title_text)
    normalized_match = catalog_index.title_normalized.get(normalized)
    if normalized_match is not None:
        return MatchResult(
            normalized_match.item_id,
            normalized_match.item_title,
            MATCH_NORMALIZED,
            100.0,
        )

    if enable_fuzzy:
        fuzzy_match = _match_by_fuzzy_title(
            normalized,
            catalog_index,
            fuzzy_threshold=fuzzy_threshold,
        )
        if fuzzy_match is not None:
            entry, score = fuzzy_match
            return MatchResult(entry.item_id, entry.item_title, MATCH_FUZZY, score)

    return MatchResult(pd.NA, pd.NA, MATCH_UNMATCHED, None)


def _items_from_list_literal_text(text: str) -> list[str]:
    stripped = text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return []

    inner = stripped[1:-1].strip()
    if not inner:
        return []

    # Upstream cached outputs are often stringified Python lists. Product titles
    # can contain commas and apostrophes, so split only on quote-comma-quote
    # boundaries rather than every comma or every quote.
    parts = re.split(r"(?<=[\"'])\s*,\s*(?=[\"'])", inner)
    if len(parts) <= 1:
        return []

    return [_clean_candidate(_strip_outer_quotes(part)) for part in parts]


def _items_from_segment(segment: str, *, allow_header: bool = False) -> list[str]:
    segment = segment.strip()
    if not segment:
        return []

    quoted_parts = _quoted_parts(segment)
    if len(quoted_parts) > 1:
        return [part[0] for part in quoted_parts]
    if len(quoted_parts) == 1:
        return [_clean_candidate(segment)]

    if allow_header:
        segment = _strip_explanatory_header(segment)

    if "," in segment:
        return [_clean_candidate(part) for part in segment.split(",")]

    return [_clean_candidate(segment)]


def _quoted_parts(text: str) -> list[tuple[str, int, int]]:
    parts: list[tuple[str, int, int]] = []
    for match in _QUOTED_RE.finditer(text):
        value = next(group for group in match.groups() if group is not None)
        parts.append((value, match.start(), match.end()))
    return parts


def _strip_outer_quotes(text: str) -> str:
    value = text.strip()
    quote_pairs = (("'", "'"), ('"', '"'), ("â€œ", "â€"), ("â€˜", "â€™"))
    for left, right in quote_pairs:
        if value.startswith(left) and value.endswith(right):
            return value[len(left) : -len(right)].strip()
    return value


def _clean_candidate(candidate: object) -> str:
    if candidate is None or pd.isna(candidate):
        return ""

    text = _remove_ranking_prefix(str(candidate).strip())
    text = _strip_explanatory_header(text)
    text = _LEADING_FILLER_RE.sub("", text)
    text = text.strip()

    quoted_parts = _quoted_parts(text)
    if len(quoted_parts) == 1:
        quoted, start, _ = quoted_parts[0]
        id_prefix = _identifier_from_prefix(text[:start])
        return f"{id_prefix} {quoted}".strip() if id_prefix else quoted.strip()

    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n\"'“”‘’.,;:")


def _remove_ranking_prefix(text: str) -> str:
    return _RANK_PREFIX_RE.sub("", text)


def _strip_explanatory_header(text: str) -> str:
    if ":" not in text:
        return text

    prefix, suffix = text.split(":", 1)
    if _HEADER_PREFIX_RE.search(prefix):
        return suffix.strip()
    return text


def _identifier_from_prefix(prefix: str) -> str | None:
    tokens = [
        token
        for token in _TOKEN_RE.findall(prefix)
        if token.lower() not in {"item", "id", "asin", "sku"}
    ]
    return tokens[-1] if tokens else None


def _title_text_for_matching(text: str) -> str:
    quoted_parts = _quoted_parts(text)
    if len(quoted_parts) == 1:
        return quoted_parts[0][0].strip()
    return text.strip()


def _build_catalog_index(
    catalog: pd.DataFrame | None,
    *,
    item_id_col: str,
    item_title_col: str,
) -> _CatalogIndex | None:
    if catalog is None:
        return None
    if not isinstance(catalog, pd.DataFrame):
        raise ValueError("catalog must be a pandas DataFrame")

    normalized = catalog.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    item_id_col = str(item_id_col).strip().lower()
    item_title_col = _resolve_title_column(
        normalized,
        str(item_title_col).strip().lower(),
    )

    if item_id_col not in normalized.columns and item_title_col is None:
        raise ValueError(
            "catalog must contain an item id column or an item title column"
        )

    entries: list[_CatalogEntry] = []
    for _, row in normalized.iterrows():
        item_id = row[item_id_col] if item_id_col in normalized.columns else pd.NA
        item_title = row[item_title_col] if item_title_col is not None else pd.NA
        if _is_missing_scalar(item_id) and _is_missing_scalar(item_title):
            continue
        entries.append(_CatalogEntry(item_id=item_id, item_title=item_title))

    id_exact: dict[str, _CatalogEntry] = {}
    id_lower_values: dict[str, list[_CatalogEntry]] = {}
    title_exact: dict[str, _CatalogEntry] = {}
    title_normalized: dict[str, _CatalogEntry] = {}

    for entry in entries:
        if not _is_missing_scalar(entry.item_id):
            item_id = str(entry.item_id).strip()
            id_exact.setdefault(item_id, entry)
            id_lower_values.setdefault(item_id.lower(), []).append(entry)

        if not _is_missing_scalar(entry.item_title):
            title = str(entry.item_title).strip()
            title_exact.setdefault(title, entry)
            normalized_title = normalize_recommendation_text(title)
            if normalized_title:
                title_normalized.setdefault(normalized_title, entry)

    id_lower = {
        item_id: matches[0] if len(matches) == 1 else None
        for item_id, matches in id_lower_values.items()
    }
    return _CatalogIndex(
        entries=tuple(entries),
        id_exact=id_exact,
        id_lower=id_lower,
        title_exact=title_exact,
        title_normalized=title_normalized,
    )


def _resolve_title_column(catalog: pd.DataFrame, requested: str) -> str | None:
    if requested in catalog.columns:
        return requested
    for fallback in FALLBACK_TITLE_COLUMNS:
        if fallback in catalog.columns:
            return fallback
    return None


def _match_by_item_id(
    candidate_text: str,
    catalog_index: _CatalogIndex,
) -> _CatalogEntry | None:
    candidate = candidate_text.strip()
    exact = catalog_index.id_exact.get(candidate)
    if exact is not None:
        return exact

    lower = catalog_index.id_lower.get(candidate.lower())
    if lower is not None:
        return lower

    for token in _TOKEN_RE.findall(candidate):
        exact = catalog_index.id_exact.get(token)
        if exact is not None:
            return exact
        lower = catalog_index.id_lower.get(token.lower())
        if lower is not None:
            return lower

    return None


def _match_by_fuzzy_title(
    normalized_text: str,
    catalog_index: _CatalogIndex,
    *,
    fuzzy_threshold: float,
) -> tuple[_CatalogEntry, float] | None:
    if not normalized_text:
        return None
    if fuzz is None or process is None:
        raise ImportError(
            "rapidfuzz is required for fuzzy matching. Install rapidfuzz or "
            "call with enable_fuzzy=False."
        )

    choices = list(catalog_index.title_normalized.keys())
    if not choices:
        return None

    result = process.extractOne(
        normalized_text,
        choices,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=fuzzy_threshold,
    )
    if result is None:
        return None

    matched_title, score, _ = result
    return catalog_index.title_normalized[matched_title], float(score)


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except ValueError:
        return False


def _output_columns(rows: list[dict[str, Any]]) -> list[str]:
    base_columns = [
        "user_id",
        "model_name",
        "dataset",
        "rank",
        "recommended_item_id",
        "recommended_item_text",
        "run_id",
        "raw_output",
        "match_type",
        "match_score",
        "catalog_item_title",
    ]
    extra_columns = [
        column
        for row in rows
        for column in row
        if column not in base_columns
    ]
    return [*base_columns, *dict.fromkeys(extra_columns)]
