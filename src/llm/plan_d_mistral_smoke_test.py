from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any

from src.llm.plan_d_llm_ranking_parser import RankingParseResult, parse_llm_ranking_response


PLAN = "Plan D"
PHASE = "D5H_mistral_local_availability_smoke_test"
DATASET = "synthetic_toy"
SCOPE = "smoke_test"
BACKEND = "ollama"
DEFAULT_MODEL = "mistral:7b-instruct"
PROMPT_STRATEGY = "strict_candidate_id_only_smoke"
METHOD = "mistral_ollama_smoke_test"
TEMPERATURE = 0.0
TOP_K = 3
CANDIDATE_IDS = ["101", "102", "103", "104", "105"]

SUMMARY_COLUMNS = [
    "plan",
    "phase",
    "classification",
    "backend",
    "requested_model",
    "selected_model",
    "model_available",
    "mistral_candidates",
    "ollama_list_checked",
    "local_ollama_used",
    "local_request_count",
    "response_received",
    "parse_success",
    "schema_valid",
    "exact_top_k",
    "candidate_validity_rate",
    "hallucination_rate",
    "duplicate_rate",
    "latency_seconds",
    "status",
    "raw_output_path",
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
    "metric",
    "k",
    "value",
    "num_instances",
    "status",
    "notes",
]


@dataclass(frozen=True)
class LocalModelInventory:
    """Parsed local Ollama model inventory."""

    command_available: bool
    model_names: list[str]
    raw_output: str
    error_message: str


@dataclass(frozen=True)
class SmokeEvaluation:
    """Compact D5H Mistral smoke-test result."""

    requested_model: str
    selected_model: str
    model_available: bool
    mistral_candidates: list[str]
    ollama_list_checked: bool
    local_ollama_used: bool
    local_request_count: int
    response_received: bool
    parse_success: bool
    schema_valid: bool
    exact_top_k: bool
    candidate_validity_rate: float
    hallucination_rate: float
    duplicate_rate: float
    latency_seconds: float
    status: str
    raw_output_path: str
    notes: str


def parse_ollama_list_output(output: str) -> list[str]:
    """Parse model names from `ollama list` text output."""

    model_names: list[str] = []
    for line in str(output or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        first = stripped.split()[0]
        if first.lower() == "name":
            continue
        model_names.append(first)
    return model_names


def detect_mistral_candidates(model_names: list[str]) -> list[str]:
    """Return local model names that look Mistral-compatible."""

    return sorted(name for name in model_names if "mistral" in name.lower())


def select_local_mistral_model(model_names: list[str], requested_model: str) -> str | None:
    """Select the requested local Mistral model only when it is already installed."""

    return requested_model if requested_model in set(model_names) else None


def check_local_ollama_models() -> LocalModelInventory:
    """Run the local `ollama list` command without downloading or pulling models."""

    try:
        completed = subprocess.run(
            ["ollama", "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except FileNotFoundError:
        return LocalModelInventory(False, [], "", "ollama command not found")
    except subprocess.TimeoutExpired:
        return LocalModelInventory(False, [], "", "ollama list timed out")
    except Exception as exc:  # pragma: no cover - local environment dependent
        return LocalModelInventory(False, [], "", str(exc))

    raw_output = completed.stdout or ""
    error_message = (completed.stderr or "").strip()
    if completed.returncode != 0:
        return LocalModelInventory(False, [], raw_output, error_message or f"ollama list exited {completed.returncode}")
    return LocalModelInventory(True, parse_ollama_list_output(raw_output), raw_output, "")


def build_mistral_smoke_prompt(candidate_ids: list[str] | None = None, top_k: int = TOP_K) -> str:
    """Build a tiny strict JSON candidate-ID-only smoke-test prompt."""

    ids = list(candidate_ids or CANDIDATE_IDS)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if top_k > len(ids):
        raise ValueError("top_k cannot exceed candidate count")
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_ids must be unique")
    allowed_ids = json.dumps(ids, separators=(",", ":"))
    return f"""You are running a local model availability smoke test.

Allowed candidate IDs, copy from this list only:
{allowed_ids}

Rules:
- Return only valid JSON.
- Select exactly {top_k} item IDs.
- Every selected ID must be copied exactly from the allowed candidate ID list.
- Do not invent IDs.
- Do not output item titles.
- Do not output explanations.
- If uncertain, still choose from the provided candidate IDs only.

Required JSON object:
ranked_item_ids must be an array of exactly {top_k} strings copied from the allowed list; status must be "ok".
"""


def evaluate_smoke_response(raw_response_text: str, *, response_received: bool) -> tuple[RankingParseResult, dict[str, float | bool]]:
    """Parse and score one tiny Mistral smoke response."""

    parsed = parse_llm_ranking_response(
        raw_response_text if response_received else "",
        candidate_ids=set(CANDIDATE_IDS),
        expected_top_k=TOP_K,
    )
    total_selected = parsed.selected_count
    hallucination_rate = parsed.hallucinated_count / total_selected if total_selected else 0.0
    duplicate_rate = parsed.duplicate_count / total_selected if total_selected else 0.0
    metrics: dict[str, float | bool] = {
        "response_received": response_received,
        "parse_success": parsed.parse_success,
        "schema_valid": parsed.schema_valid,
        "exact_top_k": parsed.selected_count == TOP_K,
        "candidate_validity_rate": parsed.candidate_validity_rate,
        "hallucination_rate": hallucination_rate,
        "duplicate_rate": duplicate_rate,
    }
    return parsed, metrics


def blocked_evaluation(
    *,
    requested_model: str,
    inventory: LocalModelInventory,
    raw_output_path: str = "",
) -> SmokeEvaluation:
    """Build a blocked D5H result without fabricating inference metrics."""

    candidates = detect_mistral_candidates(inventory.model_names)
    if inventory.command_available:
        notes = (
            f"Requested model `{requested_model}` is not available locally. "
            "No Mistral inference was run. Manual command if desired later: ollama pull mistral:7b-instruct."
        )
        if candidates:
            notes += f" Possible local Mistral candidates detected: {', '.join(candidates)}."
    else:
        notes = (
            f"Could not inspect local Ollama models: {inventory.error_message}. "
            "No Mistral inference was run and no model pull/download was performed."
        )
    return SmokeEvaluation(
        requested_model=requested_model,
        selected_model="",
        model_available=False,
        mistral_candidates=candidates,
        ollama_list_checked=inventory.command_available,
        local_ollama_used=False,
        local_request_count=0,
        response_received=False,
        parse_success=False,
        schema_valid=False,
        exact_top_k=False,
        candidate_validity_rate=0.0,
        hallucination_rate=0.0,
        duplicate_rate=0.0,
        latency_seconds=0.0,
        status="blocked_model_not_available",
        raw_output_path=raw_output_path,
        notes=notes,
    )


def evaluation_summary_row(evaluation: SmokeEvaluation) -> dict[str, str]:
    """Return the single-row committed D5H summary table."""

    blocked = evaluation.status == "blocked_model_not_available"
    return {
        "plan": PLAN,
        "phase": PHASE,
        "classification": "local_model_availability_smoke_test",
        "backend": BACKEND,
        "requested_model": evaluation.requested_model,
        "selected_model": evaluation.selected_model,
        "model_available": _bool_text(evaluation.model_available),
        "mistral_candidates": json.dumps(evaluation.mistral_candidates, separators=(",", ":")),
        "ollama_list_checked": _bool_text(evaluation.ollama_list_checked),
        "local_ollama_used": _bool_text(evaluation.local_ollama_used),
        "local_request_count": str(evaluation.local_request_count),
        "response_received": "" if blocked else _bool_text(evaluation.response_received),
        "parse_success": "" if blocked else _bool_text(evaluation.parse_success),
        "schema_valid": "" if blocked else _bool_text(evaluation.schema_valid),
        "exact_top_k": "" if blocked else _bool_text(evaluation.exact_top_k),
        "candidate_validity_rate": "" if blocked else _format_float(evaluation.candidate_validity_rate),
        "hallucination_rate": "" if blocked else _format_float(evaluation.hallucination_rate),
        "duplicate_rate": "" if blocked else _format_float(evaluation.duplicate_rate),
        "latency_seconds": "" if blocked else _format_float(evaluation.latency_seconds),
        "status": evaluation.status,
        "raw_output_path": evaluation.raw_output_path,
        "notes": evaluation.notes,
    }


def metric_rows(evaluation: SmokeEvaluation) -> list[dict[str, str]]:
    """Return D5H metric rows; blocked rows keep values blank to avoid fake metrics."""

    if evaluation.status == "blocked_model_not_available":
        metrics = [
            "response_received",
            "parse_success",
            "schema_valid",
            "exact_top_k",
            "candidate_validity_rate",
            "hallucination_rate",
            "duplicate_rate",
            "latency_seconds",
        ]
        return [_metric_row(evaluation, metric, "", "", "No smoke-test inference was run.") for metric in metrics]

    values = {
        "response_received": 1.0 if evaluation.response_received else 0.0,
        "parse_success": 1.0 if evaluation.parse_success else 0.0,
        "schema_valid": 1.0 if evaluation.schema_valid else 0.0,
        "exact_top_k": 1.0 if evaluation.exact_top_k else 0.0,
        "candidate_validity_rate": evaluation.candidate_validity_rate,
        "hallucination_rate": evaluation.hallucination_rate,
        "duplicate_rate": evaluation.duplicate_rate,
        "latency_seconds": evaluation.latency_seconds,
    }
    return [
        _metric_row(evaluation, metric, "3" if metric in {"exact_top_k"} else "", _format_float(value), evaluation.notes)
        for metric, value in values.items()
    ]


def raw_smoke_trace_row(
    *,
    evaluation: SmokeEvaluation,
    raw_response_text: str,
    parsed_result: Any,
) -> dict[str, Any]:
    """Return one JSONL-safe D5H raw trace row without storing secrets or prompts."""

    return {
        "phase": PHASE,
        "backend": BACKEND,
        "model": evaluation.selected_model,
        "prompt_strategy": PROMPT_STRATEGY,
        "raw_response_text": raw_response_text,
        "parsed_result": parsed_result,
        "latency_seconds": evaluation.latency_seconds,
        "status": evaluation.status,
    }


def _metric_row(evaluation: SmokeEvaluation, metric: str, k: str, value: str, notes: str) -> dict[str, str]:
    return {
        "plan": PLAN,
        "phase": PHASE,
        "dataset": DATASET,
        "scope": SCOPE,
        "method": METHOD,
        "backend": BACKEND,
        "model": evaluation.selected_model or evaluation.requested_model,
        "metric": metric,
        "k": k,
        "value": value,
        "num_instances": "0" if evaluation.status == "blocked_model_not_available" else "1",
        "status": evaluation.status,
        "notes": notes,
    }


def _format_float(value: float) -> str:
    return f"{float(value):.6f}"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"
