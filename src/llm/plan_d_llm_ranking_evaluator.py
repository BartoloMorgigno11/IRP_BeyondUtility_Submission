from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.llm.plan_d_llm_ranking_parser import RankingParseResult, parse_llm_ranking_response
from src.llm.plan_d_prompt_templates import PromptItem, build_ranking_prompt


PLAN = "Plan D"
PHASE = "D5E_qwen2_ml1m_ranking_diagnostic"
DATASET = "ml1m"
SCOPE = "diagnostic_subset"
BACKEND = "ollama"
MODEL = "qwen2:7b-instruct"
PROMPT_STRATEGY = "openp5_base"
TEMPERATURE = 0.0
TOP_K = 5
CANDIDATE_POOL_SIZE = 20
METHOD = "qwen2_ollama_openp5"

INSTANCE_SUMMARY_COLUMNS = [
    "plan",
    "phase",
    "dataset",
    "scope",
    "user_id",
    "instance_id",
    "backend",
    "model",
    "prompt_strategy",
    "temperature",
    "top_k",
    "candidate_pool_size",
    "positive_item_id",
    "response_received",
    "parse_success",
    "schema_valid",
    "selected_count",
    "candidate_validity_rate",
    "duplicate_count",
    "hallucinated_count",
    "positive_rank",
    "hit_at_5",
    "ndcg_at_5",
    "latency_seconds",
    "parse_status",
    "status",
    "notes",
]

AGGREGATE_SUMMARY_COLUMNS = [
    "plan",
    "phase",
    "dataset",
    "scope",
    "backend",
    "model",
    "prompt_strategy",
    "temperature",
    "num_instances",
    "num_responses_received",
    "response_received_rate",
    "parse_success_rate",
    "schema_valid_rate",
    "mean_candidate_validity_rate",
    "hallucination_rate",
    "duplicate_rate",
    "exact_top_k_rate",
    "empty_output_rate",
    "hit_rate_at_5",
    "ndcg_at_5",
    "mean_latency_seconds",
    "total_runtime_seconds",
    "raw_output_path",
    "status",
    "notes",
]

METRIC_COLUMNS = [
    "plan",
    "phase",
    "dataset",
    "scope",
    "method",
    "backend",
    "model",
    "prompt_strategy",
    "metric",
    "k",
    "value",
    "num_instances",
    "notes",
]


@dataclass(frozen=True)
class PromptPayload:
    """Prompt and evaluation metadata for one D5E manifest row."""

    user_id: str
    instance_id: str
    positive_item_id: str
    candidate_ids: list[str]
    prompt: str


@dataclass(frozen=True)
class InstanceEvaluation:
    """Per-instance D5E evaluation row derived from one local LLM response."""

    plan: str
    phase: str
    dataset: str
    scope: str
    user_id: str
    instance_id: str
    backend: str
    model: str
    prompt_strategy: str
    temperature: float
    top_k: int
    candidate_pool_size: int
    positive_item_id: str
    response_received: bool
    parse_success: bool
    schema_valid: bool
    selected_count: int
    candidate_validity_rate: float
    duplicate_count: int
    hallucinated_count: int
    positive_rank: int | None
    hit_at_5: float
    ndcg_at_5: float
    latency_seconds: float
    parse_status: str
    status: str
    notes: str

    def row(self) -> dict[str, str]:
        return {
            "plan": self.plan,
            "phase": self.phase,
            "dataset": self.dataset,
            "scope": self.scope,
            "user_id": self.user_id,
            "instance_id": self.instance_id,
            "backend": self.backend,
            "model": self.model,
            "prompt_strategy": self.prompt_strategy,
            "temperature": _format_float(self.temperature),
            "top_k": str(self.top_k),
            "candidate_pool_size": str(self.candidate_pool_size),
            "positive_item_id": self.positive_item_id,
            "response_received": _bool_text(self.response_received),
            "parse_success": _bool_text(self.parse_success),
            "schema_valid": _bool_text(self.schema_valid),
            "selected_count": str(self.selected_count),
            "candidate_validity_rate": _format_float(self.candidate_validity_rate),
            "duplicate_count": str(self.duplicate_count),
            "hallucinated_count": str(self.hallucinated_count),
            "positive_rank": "" if self.positive_rank is None else str(self.positive_rank),
            "hit_at_5": _format_float(self.hit_at_5),
            "ndcg_at_5": _format_float(self.ndcg_at_5),
            "latency_seconds": _format_float(self.latency_seconds),
            "parse_status": self.parse_status,
            "status": self.status,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class DiagnosticAggregate:
    """Aggregate D5E diagnostic metrics."""

    num_instances: int
    num_responses_received: int
    response_received_rate: float
    parse_success_rate: float
    schema_valid_rate: float
    mean_candidate_validity_rate: float
    hallucination_rate: float
    duplicate_rate: float
    exact_top_k_rate: float
    empty_output_rate: float
    hit_rate_at_5: float
    ndcg_at_5: float
    mean_latency_seconds: float
    total_runtime_seconds: float
    raw_output_path: str
    status: str
    notes: str

    def summary_row(self) -> dict[str, str]:
        return {
            "plan": PLAN,
            "phase": PHASE,
            "dataset": DATASET,
            "scope": SCOPE,
            "backend": BACKEND,
            "model": MODEL,
            "prompt_strategy": PROMPT_STRATEGY,
            "temperature": _format_float(TEMPERATURE),
            "num_instances": str(self.num_instances),
            "num_responses_received": str(self.num_responses_received),
            "response_received_rate": _format_float(self.response_received_rate),
            "parse_success_rate": _format_float(self.parse_success_rate),
            "schema_valid_rate": _format_float(self.schema_valid_rate),
            "mean_candidate_validity_rate": _format_float(self.mean_candidate_validity_rate),
            "hallucination_rate": _format_float(self.hallucination_rate),
            "duplicate_rate": _format_float(self.duplicate_rate),
            "exact_top_k_rate": _format_float(self.exact_top_k_rate),
            "empty_output_rate": _format_float(self.empty_output_rate),
            "hit_rate_at_5": _format_float(self.hit_rate_at_5),
            "ndcg_at_5": _format_float(self.ndcg_at_5),
            "mean_latency_seconds": _format_float(self.mean_latency_seconds),
            "total_runtime_seconds": _format_float(self.total_runtime_seconds),
            "raw_output_path": self.raw_output_path,
            "status": self.status,
            "notes": self.notes,
        }

    def metric_rows(self) -> list[dict[str, str]]:
        metrics = [
            ("hit_rate", "5", self.hit_rate_at_5),
            ("ndcg", "5", self.ndcg_at_5),
            ("response_received_rate", "", self.response_received_rate),
            ("parse_success_rate", "", self.parse_success_rate),
            ("schema_valid_rate", "", self.schema_valid_rate),
            ("mean_candidate_validity_rate", "", self.mean_candidate_validity_rate),
            ("hallucination_rate", "", self.hallucination_rate),
            ("duplicate_rate", "", self.duplicate_rate),
            ("exact_top_k_rate", "", self.exact_top_k_rate),
            ("empty_output_rate", "", self.empty_output_rate),
            ("mean_latency_seconds", "", self.mean_latency_seconds),
            ("total_runtime_seconds", "", self.total_runtime_seconds),
        ]
        return [
            {
                "plan": PLAN,
                "phase": PHASE,
                "dataset": DATASET,
                "scope": SCOPE,
                "method": METHOD,
                "backend": BACKEND,
                "model": MODEL,
                "prompt_strategy": PROMPT_STRATEGY,
                "metric": metric,
                "k": k,
                "value": _format_float(value),
                "num_instances": str(self.num_instances),
                "notes": self.notes,
            }
            for metric, k, value in metrics
        ]


class ManifestPromptBuilder:
    """Rebuild D5E prompts from the manifest and local reconstructed ML-1M files."""

    def __init__(self, data_root: Path):
        dataset_dir = data_root / DATASET
        interactions = _read_csv(dataset_dir / "interactions.csv")
        candidates = _read_csv(dataset_dir / "candidates.csv")
        catalog = _read_csv(dataset_dir / "catalog.csv")

        self.catalog_by_item = _catalog_map(catalog)
        train = interactions.loc[interactions["split"].astype(str).str.lower() == "train"].copy()
        train = _stable_sort_history(train)
        self.histories_by_user = {
            str(user_id): group.copy()
            for user_id, group in train.groupby(train["user_id"].astype(str), sort=False)
        }
        self.candidates_by_user = {
            str(user_id): group.copy()
            for user_id, group in candidates.groupby(candidates["user_id"].astype(str), sort=False)
        }

    def build_from_manifest_row(self, row: pd.Series) -> PromptPayload:
        user_id = str(row["user_id"])
        candidate_ids = _parse_candidate_ids(str(row["candidate_ids"]))
        history = self.histories_by_user.get(user_id)
        if history is None or history.empty:
            raise ValueError(f"Manifest user {user_id} has no reconstructed train history")

        user_candidates = self.candidates_by_user.get(user_id)
        if user_candidates is None or user_candidates.empty:
            raise ValueError(f"Manifest user {user_id} has no reconstructed candidate rows")

        candidate_rows = _candidate_rows_in_manifest_order(user_candidates, candidate_ids)
        if len(candidate_rows) != len(candidate_ids):
            raise ValueError(f"Manifest user {user_id} has candidate IDs missing from reconstructed candidates")

        history_items = [_history_prompt_item(item_row, self.catalog_by_item) for _, item_row in history.iterrows()]
        candidate_items = [
            _candidate_prompt_item(item_row, self.catalog_by_item) for _, item_row in candidate_rows.iterrows()
        ]
        top_k = int(str(row["top_k"]))
        prompt = build_ranking_prompt(
            user_id=user_id,
            history_items=history_items,
            candidate_items=candidate_items,
            top_k=top_k,
            prompt_strategy=str(row["prompt_strategy"]),
        )
        return PromptPayload(
            user_id=user_id,
            instance_id=str(row["instance_id"]),
            positive_item_id=str(row["positive_item_id"]),
            candidate_ids=candidate_ids,
            prompt=prompt,
        )


def load_and_validate_manifest(path: Path) -> pd.DataFrame:
    """Load and strictly validate the D5D manifest for the bounded D5E run."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing D5E manifest: {path}")
    manifest = pd.read_csv(path, dtype=str).fillna("")
    errors: list[str] = []
    if len(manifest) != 20:
        errors.append(f"expected exactly 20 manifest rows, found {len(manifest)}")
    _expect_single_value(manifest, "plan", PLAN, errors)
    _expect_single_value(manifest, "phase", PHASE, errors)
    _expect_single_value(manifest, "dataset", DATASET, errors)
    _expect_single_value(manifest, "model", MODEL, errors)
    _expect_single_value(manifest, "backend", BACKEND, errors)
    _expect_single_value(manifest, "prompt_strategy", PROMPT_STRATEGY, errors)
    _expect_single_value(manifest, "candidate_count", str(CANDIDATE_POOL_SIZE), errors)
    _expect_single_value(manifest, "candidate_pool_size", str(CANDIDATE_POOL_SIZE), errors)
    _expect_single_value(manifest, "top_k", str(TOP_K), errors)
    _expect_single_value(manifest, "positive_in_candidates", "true", errors, lower=True)
    _expect_single_value(manifest, "manifest_status", "ready_for_llm_inference", errors)
    for _, row in manifest.iterrows():
        candidate_ids = _parse_candidate_ids(str(row.get("candidate_ids", "")))
        if len(candidate_ids) != CANDIDATE_POOL_SIZE:
            errors.append(f"{row.get('instance_id', '<missing>')}: expected 20 candidate IDs")
        if str(row.get("positive_item_id", "")) not in set(candidate_ids):
            errors.append(f"{row.get('instance_id', '<missing>')}: positive item is not in candidate IDs")
    if errors:
        raise ValueError("Invalid D5E manifest: " + "; ".join(errors))
    return manifest


def evaluate_response(
    *,
    payload: PromptPayload,
    response_received: bool,
    raw_response_text: str,
    latency_seconds: float,
    error_message: str | None,
) -> tuple[InstanceEvaluation, RankingParseResult]:
    """Parse and score one D5E local LLM response."""

    if response_received:
        parsed = parse_llm_ranking_response(
            raw_response_text,
            candidate_ids=set(payload.candidate_ids),
            expected_top_k=TOP_K,
        )
    else:
        parsed = parse_llm_ranking_response("", candidate_ids=set(payload.candidate_ids), expected_top_k=TOP_K)

    positive_rank = _positive_rank(
        ranked_item_ids=parsed.ranked_item_ids,
        positive_item_id=payload.positive_item_id,
        top_k=TOP_K,
    )
    hit_at_5 = 1.0 if positive_rank is not None else 0.0
    ndcg_at_5 = 1.0 / math.log2(positive_rank + 1) if positive_rank is not None else 0.0
    status = _instance_status(response_received=response_received, parsed=parsed)
    notes = _instance_notes(error_message=error_message, parsed=parsed)
    evaluation = InstanceEvaluation(
        plan=PLAN,
        phase=PHASE,
        dataset=DATASET,
        scope=SCOPE,
        user_id=payload.user_id,
        instance_id=payload.instance_id,
        backend=BACKEND,
        model=MODEL,
        prompt_strategy=PROMPT_STRATEGY,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        candidate_pool_size=CANDIDATE_POOL_SIZE,
        positive_item_id=payload.positive_item_id,
        response_received=response_received,
        parse_success=parsed.parse_success,
        schema_valid=parsed.schema_valid,
        selected_count=parsed.selected_count,
        candidate_validity_rate=parsed.candidate_validity_rate,
        duplicate_count=parsed.duplicate_count,
        hallucinated_count=parsed.hallucinated_count,
        positive_rank=positive_rank,
        hit_at_5=hit_at_5,
        ndcg_at_5=ndcg_at_5,
        latency_seconds=latency_seconds,
        parse_status=parsed.status,
        status=status,
        notes=notes,
    )
    return evaluation, parsed


def aggregate_evaluations(
    evaluations: list[InstanceEvaluation],
    *,
    total_runtime_seconds: float,
    raw_output_path: str,
) -> DiagnosticAggregate:
    """Aggregate D5E instance metrics using deterministic rate definitions."""

    if not evaluations:
        return DiagnosticAggregate(
            num_instances=0,
            num_responses_received=0,
            response_received_rate=0.0,
            parse_success_rate=0.0,
            schema_valid_rate=0.0,
            mean_candidate_validity_rate=0.0,
            hallucination_rate=0.0,
            duplicate_rate=0.0,
            exact_top_k_rate=0.0,
            empty_output_rate=0.0,
            hit_rate_at_5=0.0,
            ndcg_at_5=0.0,
            mean_latency_seconds=0.0,
            total_runtime_seconds=total_runtime_seconds,
            raw_output_path=raw_output_path,
            status="failed_diagnostic_no_instances",
            notes="No manifest instances were evaluated.",
        )

    num_instances = len(evaluations)
    response_received_count = sum(1 for item in evaluations if item.response_received)
    total_selected = sum(item.selected_count for item in evaluations)
    total_hallucinated = sum(item.hallucinated_count for item in evaluations)
    total_duplicates = sum(item.duplicate_count for item in evaluations)
    status = _aggregate_status(evaluations)
    notes = _aggregate_notes(evaluations, status)
    return DiagnosticAggregate(
        num_instances=num_instances,
        num_responses_received=response_received_count,
        response_received_rate=response_received_count / num_instances,
        parse_success_rate=sum(1 for item in evaluations if item.parse_success) / num_instances,
        schema_valid_rate=sum(1 for item in evaluations if item.schema_valid) / num_instances,
        mean_candidate_validity_rate=sum(item.candidate_validity_rate for item in evaluations) / num_instances,
        hallucination_rate=(total_hallucinated / total_selected) if total_selected else 0.0,
        duplicate_rate=(total_duplicates / total_selected) if total_selected else 0.0,
        exact_top_k_rate=sum(1 for item in evaluations if item.selected_count == TOP_K) / num_instances,
        empty_output_rate=sum(1 for item in evaluations if item.selected_count == 0) / num_instances,
        hit_rate_at_5=sum(item.hit_at_5 for item in evaluations) / num_instances,
        ndcg_at_5=sum(item.ndcg_at_5 for item in evaluations) / num_instances,
        mean_latency_seconds=sum(item.latency_seconds for item in evaluations) / num_instances,
        total_runtime_seconds=total_runtime_seconds,
        raw_output_path=raw_output_path,
        status=status,
        notes=notes,
    )


def raw_trace_row(
    *,
    payload: PromptPayload,
    raw_response_text: str,
    parsed_result: RankingParseResult,
    latency_seconds: float,
    error_message: str | None,
) -> dict[str, Any]:
    """Return one JSONL-safe raw trace row without storing full prompts."""

    return {
        "instance_id": payload.instance_id,
        "user_id": payload.user_id,
        "model": MODEL,
        "backend": BACKEND,
        "prompt_strategy": PROMPT_STRATEGY,
        "raw_response_text": raw_response_text,
        "parsed_result": asdict(parsed_result),
        "latency_seconds": latency_seconds,
        "error_message": error_message or "",
    }


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required local artifact: {path}")
    return pd.read_csv(path, dtype="string", keep_default_na=False)


def _expect_single_value(
    frame: pd.DataFrame,
    column: str,
    expected: str,
    errors: list[str],
    *,
    lower: bool = False,
) -> None:
    if column not in frame.columns:
        errors.append(f"missing column {column}")
        return
    values = frame[column].astype(str).str.strip()
    if lower:
        values = values.str.lower()
        expected = expected.lower()
    observed = set(values.tolist())
    if observed != {expected}:
        errors.append(f"{column} expected {expected!r}, observed {sorted(observed)!r}")


def _parse_candidate_ids(encoded: str) -> list[str]:
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid candidate_ids JSON: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("candidate_ids must be a JSON list of strings")
    return value


def _catalog_map(catalog: pd.DataFrame) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for _, row in catalog.iterrows():
        item_id = str(row.get("item_id", "")).strip()
        if not item_id:
            continue
        result[item_id] = {
            "title": str(row.get("title", "")).strip() or item_id,
            "metadata": str(row.get("categories", "")).strip(),
        }
    return result


def _stable_sort_history(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["__timestamp_sort"] = pd.to_numeric(result.get("timestamp", -1), errors="coerce").fillna(-1)
    result["__item_sort"] = result["item_id"].map(_id_sort_key)
    return result.sort_values(["user_id", "__timestamp_sort", "__item_sort"], kind="mergesort").drop(
        columns=["__timestamp_sort", "__item_sort"]
    )


def _candidate_rows_in_manifest_order(user_candidates: pd.DataFrame, candidate_ids: list[str]) -> pd.DataFrame:
    by_item_id = {
        str(row["candidate_item_id"]): row
        for _, row in user_candidates.iterrows()
    }
    rows = [by_item_id[item_id] for item_id in candidate_ids if item_id in by_item_id]
    return pd.DataFrame(rows)


def _history_prompt_item(row: pd.Series, catalog_by_item: dict[str, dict[str, str]]) -> PromptItem:
    item_id = str(row["item_id"])
    meta = catalog_by_item.get(item_id, {})
    return PromptItem(
        item_id=item_id,
        title=meta.get("title", item_id),
        metadata=meta.get("metadata", ""),
    )


def _candidate_prompt_item(row: pd.Series, catalog_by_item: dict[str, dict[str, str]]) -> PromptItem:
    item_id = str(row["candidate_item_id"])
    title = str(row.get("candidate_item_title", "")).strip()
    meta = catalog_by_item.get(item_id, {})
    return PromptItem(
        item_id=item_id,
        title=title or meta.get("title", item_id),
        metadata=meta.get("metadata", ""),
    )


def _positive_rank(*, ranked_item_ids: list[str], positive_item_id: str, top_k: int) -> int | None:
    for index, item_id in enumerate(ranked_item_ids[:top_k], start=1):
        if item_id == positive_item_id:
            return index
    return None


def _instance_status(*, response_received: bool, parsed: RankingParseResult) -> str:
    if not response_received:
        return "failed_backend"
    if parsed.status == "parsed_valid":
        return "completed"
    if parsed.status == "parsed_with_hallucinations":
        return "completed_with_hallucinations"
    if parsed.status == "parsed_with_duplicates":
        return "completed_with_duplicates"
    if parsed.status == "parsed_invalid_schema":
        return "completed_invalid_schema"
    if parsed.status == "empty_response":
        return "failed_empty_response"
    return "failed_parse"


def _instance_notes(*, error_message: str | None, parsed: RankingParseResult) -> str:
    if error_message:
        return f"Local Ollama request failed: {error_message}"
    if parsed.error_message:
        return parsed.error_message
    if parsed.hallucinated_count > 0:
        return "Parsed response contains item IDs outside the candidate set."
    if parsed.duplicate_count > 0:
        return "Parsed response contains duplicate item IDs."
    return "Parsed response evaluated without manual correction."


def _aggregate_status(evaluations: list[InstanceEvaluation]) -> str:
    if all(not item.response_received for item in evaluations):
        return "failed_diagnostic_backend"
    if any(not item.response_received for item in evaluations):
        return "completed_with_instance_failures"
    return "completed_diagnostic_subset"


def _aggregate_notes(evaluations: list[InstanceEvaluation], status: str) -> str:
    if status == "failed_diagnostic_backend":
        return "No local Ollama responses were received; no outputs were fabricated."
    if status == "completed_with_instance_failures":
        failed = sum(1 for item in evaluations if not item.response_received)
        return f"Diagnostic processed {len(evaluations)} manifest rows with {failed} backend failures."
    return (
        "Diagnostic processed all 20 D5D manifest rows with local Qwen2/Ollama inference; "
        "not exact paper reproduction."
    )


def _id_sort_key(value: object) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _format_float(value: float) -> str:
    return f"{float(value):.6f}"
