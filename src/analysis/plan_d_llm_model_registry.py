from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORT_DIR = PROJECT_ROOT / "report"

MODEL_REGISTRY_PATH = RESULTS_TABLES_DIR / "plan_d_llm_model_registry.csv"
BACKEND_AUDIT_PATH = RESULTS_TABLES_DIR / "plan_d_llm_backend_audit.csv"
TASK_ROADMAP_PATH = RESULTS_TABLES_DIR / "plan_d_llm_task_roadmap.csv"
REPORT_PATH = REPORT_DIR / "plan_d_llm_model_registry_and_backend_audit.md"
FEASIBILITY_LEDGER_PATH = RESULTS_TABLES_DIR / "plan_d_feasibility_ledger.csv"
NEXT_ACTIONS_PATH = RESULTS_TABLES_DIR / "plan_d_next_actions_priority.csv"

LOGGER = logging.getLogger(__name__)

MODEL_REGISTRY_COLUMNS = [
    "paper_model",
    "model_family",
    "source_type",
    "paper_model_exact_id",
    "recommended_backend",
    "recommended_local_model",
    "recommended_api_model",
    "exact_model_availability",
    "local_feasibility_on_rtx4060_8gb",
    "api_required",
    "reproduction_level",
    "priority",
    "status",
    "planned_use",
    "notes",
]

BACKEND_AUDIT_COLUMNS = [
    "backend",
    "check_name",
    "command_or_import",
    "available",
    "version_or_result",
    "status",
    "notes",
]

TASK_ROADMAP_COLUMNS = [
    "phase",
    "task",
    "dataset",
    "model",
    "backend",
    "users",
    "candidate_pool_size",
    "prompt_strategy",
    "output_format",
    "metrics",
    "prerequisites",
    "status",
    "notes",
]

LEDGER_COLUMNS = [
    "item_id",
    "area",
    "target",
    "feasibility_status",
    "required_resources",
    "estimated_effort",
    "risk_level",
    "external_dependency",
    "recommended_decision",
    "notes",
]

NEXT_ACTION_COLUMNS = [
    "priority_rank",
    "action_id",
    "action_title",
    "related_component",
    "why_it_matters",
    "expected_output",
    "prerequisites",
    "risk",
    "recommended_order",
    "notes",
]

QWEN_MODEL = "qwen2:7b-instruct"
MISTRAL_MODELS = (
    "mistral:7b-instruct-v0.2-q4_0",
    "mistral:7b-instruct-v0.2-q4_K_M",
)


@dataclass(frozen=True)
class BackendContext:
    ollama_exists: bool
    ollama_models: tuple[str, ...]
    qwen_available: bool
    mistral_available: bool
    torch_importable: bool
    cuda_available: bool
    cuda_device_name: str


def model_registry_rows() -> list[dict[str, str]]:
    """Return the conservative D5B registry for the seven paper LLMs."""

    return [
        {
            "paper_model": "gpt-3.5-turbo-0125",
            "model_family": "openai_gpt",
            "source_type": "closed_api",
            "paper_model_exact_id": "gpt-3.5-turbo-0125",
            "recommended_backend": "openai_api",
            "recommended_local_model": "none",
            "recommended_api_model": "gpt-3.5-turbo-0125 if available, otherwise none",
            "exact_model_availability": "conditional_verify_with_api_model_list",
            "local_feasibility_on_rtx4060_8gb": "not_local",
            "api_required": "true",
            "reproduction_level": "exact_only_if_api_model_available",
            "priority": "low",
            "status": "conditional_legacy_api",
            "planned_use": "document_only_unless_api_approved",
            "notes": (
                "Legacy/conditional; do not use without explicit user approval "
                "and API availability verification."
            ),
        },
        {
            "paper_model": "gpt-4o-mini-2024-07-18",
            "model_family": "openai_gpt",
            "source_type": "closed_api",
            "paper_model_exact_id": "gpt-4o-mini-2024-07-18",
            "recommended_backend": "openai_api",
            "recommended_local_model": "none",
            "recommended_api_model": "gpt-4o-mini-2024-07-18",
            "exact_model_availability": "documented_api_snapshot_but_requires_key_and_budget",
            "local_feasibility_on_rtx4060_8gb": "not_local",
            "api_required": "true",
            "reproduction_level": "exact_api_conditional",
            "priority": "medium_high",
            "status": "feasible_with_api_budget",
            "planned_use": "optional_small_api_diagnostic_if_approved",
            "notes": "Best first closed-source candidate if user approves API budget.",
        },
        {
            "paper_model": "gpt-4o-2024-08-06",
            "model_family": "openai_gpt",
            "source_type": "closed_api",
            "paper_model_exact_id": "gpt-4o-2024-08-06",
            "recommended_backend": "openai_api",
            "recommended_local_model": "none",
            "recommended_api_model": "gpt-4o-2024-08-06",
            "exact_model_availability": "documented_api_snapshot_but_requires_key_and_budget",
            "local_feasibility_on_rtx4060_8gb": "not_local",
            "api_required": "true",
            "reproduction_level": "exact_api_conditional",
            "priority": "medium",
            "status": "feasible_with_api_budget",
            "planned_use": "optional_small_api_diagnostic_if_approved",
            "notes": (
                "Stronger but more expensive closed-source candidate; only for "
                "small diagnostic subset if approved."
            ),
        },
        {
            "paper_model": "claude-3-haiku-20240307",
            "model_family": "anthropic_claude",
            "source_type": "closed_api",
            "paper_model_exact_id": "claude-3-haiku-20240307",
            "recommended_backend": "anthropic_api",
            "recommended_local_model": "none",
            "recommended_api_model": "none_exact",
            "exact_model_availability": "retired",
            "local_feasibility_on_rtx4060_8gb": "not_local",
            "api_required": "true",
            "reproduction_level": "blocked_exact_replacement_only",
            "priority": "low",
            "status": "blocked_exact_model_retired",
            "planned_use": "document_only_or_non_exact_proxy_if_later_approved",
            "notes": (
                "Exact model retired; possible future proxy with current Claude "
                "Haiku only if explicitly approved and clearly marked non-exact."
            ),
        },
        {
            "paper_model": "llama-3-70b-instruct",
            "model_family": "meta_llama",
            "source_type": "open_weight_large",
            "paper_model_exact_id": "llama-3-70b-instruct",
            "recommended_backend": "cloud_or_api",
            "recommended_local_model": "none_for_8gb_vram",
            "recommended_api_model": "provider_dependent",
            "exact_model_availability": "open_weight_but_not_local_feasible_on_8gb",
            "local_feasibility_on_rtx4060_8gb": "not_feasible",
            "api_required": "true_for_practical_run",
            "reproduction_level": "blocked_local_conditional_cloud",
            "priority": "low_medium",
            "status": "blocked_local_hardware",
            "planned_use": "document_only_unless_cloud_or_api_approved",
            "notes": (
                "70B model is not realistic on RTX 4060 8GB; document as "
                "cloud/API-required."
            ),
        },
        {
            "paper_model": "qwen-2-7b-instruct",
            "model_family": "qwen",
            "source_type": "open_local",
            "paper_model_exact_id": "qwen-2-7b-instruct",
            "recommended_backend": "ollama",
            "recommended_local_model": QWEN_MODEL,
            "recommended_api_model": "none",
            "exact_model_availability": "near_exact_or_exact_family_available_locally_if_pulled",
            "local_feasibility_on_rtx4060_8gb": "feasible_quantized",
            "api_required": "false",
            "reproduction_level": "local_open_model_near_exact",
            "priority": "high",
            "status": "recommended_first_local_model",
            "planned_use": "D5C ranking diagnostic, then position bias/reranking if successful",
            "notes": "First model to try locally after user installs Ollama and pulls the model.",
        },
        {
            "paper_model": "mistral-7b-instruct-v0.2",
            "model_family": "mistral",
            "source_type": "open_local",
            "paper_model_exact_id": "mistral-7b-instruct-v0.2",
            "recommended_backend": "ollama",
            "recommended_local_model": (
                "mistral:7b-instruct-v0.2-q4_0 or "
                "mistral:7b-instruct-v0.2-q4_K_M"
            ),
            "recommended_api_model": "none",
            "exact_model_availability": "near_exact_or_exact_family_available_locally_if_pulled",
            "local_feasibility_on_rtx4060_8gb": "feasible_quantized",
            "api_required": "false",
            "reproduction_level": "local_open_model_near_exact",
            "priority": "high",
            "status": "recommended_second_local_model",
            "planned_use": "D5D or D5F ranking diagnostic after Qwen smoke test",
            "notes": (
                "Second local model; useful for comparing hallucination/output "
                "validity against Qwen."
            ),
        },
    ]


def audit_backend_rows() -> tuple[list[dict[str, str]], BackendContext]:
    """Inspect local backend availability without downloads, HTTP calls, or inference."""

    rows: list[dict[str, str]] = []
    ollama_path = shutil.which("ollama")
    ollama_exists = ollama_path is not None
    rows.append(
        {
            "backend": "ollama",
            "check_name": "command_exists",
            "command_or_import": "ollama",
            "available": _bool_text(ollama_exists),
            "version_or_result": "found_on_path" if ollama_exists else "not_found_on_path",
            "status": "available" if ollama_exists else "blocked_missing_backend",
            "notes": (
                "Ollama command is available on PATH."
                if ollama_exists
                else "Ollama command is missing; D5C cannot start until Ollama and qwen2 are available."
            ),
        }
    )

    ollama_models: tuple[str, ...] = ()
    if ollama_exists:
        version_result = _run_local_command(["ollama", "--version"])
        rows.append(
            {
                "backend": "ollama",
                "check_name": "version",
                "command_or_import": "ollama --version",
                "available": _bool_text(version_result.returncode == 0),
                "version_or_result": version_result.output,
                "status": "available" if version_result.returncode == 0 else "conditional",
                "notes": "Local command only; no model run, pull, download, or API call was performed.",
            }
        )
        list_result = _run_local_command(["ollama", "list"])
        if list_result.returncode == 0:
            ollama_models = _parse_ollama_model_names(list_result.output)
            qwen_available = QWEN_MODEL.lower() in {model.lower() for model in ollama_models}
            mistral_available = _has_mistral_model(ollama_models)
            if qwen_available:
                installed_status = "available"
                installed_notes = "Required first local model is installed for D5C smoke testing."
            else:
                installed_status = "conditional"
                installed_notes = (
                    "Ollama is available, but qwen2:7b-instruct is not listed. "
                    "Recommended manual commands only: ollama pull qwen2:7b-instruct; "
                    "ollama pull mistral:7b-instruct-v0.2-q4_0; "
                    "ollama pull mistral:7b-instruct-v0.2-q4_K_M."
                )
            rows.append(
                {
                    "backend": "ollama",
                    "check_name": "installed_models",
                    "command_or_import": "ollama list",
                    "available": "true",
                    "version_or_result": _models_result_text(ollama_models),
                    "status": installed_status,
                    "notes": installed_notes,
                }
            )
        else:
            rows.append(
                {
                    "backend": "ollama",
                    "check_name": "installed_models",
                    "command_or_import": "ollama list",
                    "available": "false",
                    "version_or_result": list_result.output,
                    "status": "conditional",
                    "notes": "Ollama exists, but the local model list could not be read; D5C remains pending_backend.",
                }
            )
    else:
        rows.extend(
            [
                {
                    "backend": "ollama",
                    "check_name": "version",
                    "command_or_import": "ollama --version",
                    "available": "false",
                    "version_or_result": "not_checked",
                    "status": "blocked_missing_backend",
                    "notes": "Skipped because Ollama command is missing.",
                },
                {
                    "backend": "ollama",
                    "check_name": "installed_models",
                    "command_or_import": "ollama list",
                    "available": "false",
                    "version_or_result": "not_checked",
                    "status": "blocked_missing_backend",
                    "notes": "Skipped because Ollama command is missing; no model pull was attempted.",
                },
            ]
        )

    rows.extend(_python_import_rows())
    torch_rows, torch_importable, cuda_available, cuda_device_name = _torch_rows()
    rows.extend(torch_rows)

    qwen_available = QWEN_MODEL.lower() in {model.lower() for model in ollama_models}
    mistral_available = _has_mistral_model(ollama_models)
    rows.append(
        {
            "backend": "local_gpu",
            "check_name": "rtx4060_8gb_assumption",
            "command_or_import": "planning_assumption",
            "available": "true",
            "version_or_result": (
                "RTX 4060 8GB planning target; "
                f"actual_cuda_device={cuda_device_name if cuda_available else 'not_available'}"
            ),
            "status": "conditional",
            "notes": (
                "Feasibility labels are conservative for the project target GPU; "
                "7B quantized local models are planned, 70B local execution is blocked."
            ),
        }
    )

    return rows, BackendContext(
        ollama_exists=ollama_exists,
        ollama_models=ollama_models,
        qwen_available=qwen_available,
        mistral_available=mistral_available,
        torch_importable=torch_importable,
        cuda_available=cuda_available,
        cuda_device_name=cuda_device_name,
    )


def task_roadmap_rows() -> list[dict[str, str]]:
    """Return the D5C-D5J task roadmap without running any LLM task."""

    return [
        {
            "phase": "D5C",
            "task": "ollama_smoke_test",
            "dataset": "none",
            "model": QWEN_MODEL,
            "backend": "ollama",
            "users": "0",
            "candidate_pool_size": "none",
            "prompt_strategy": "simple_json",
            "output_format": "JSON",
            "metrics": "parse_success, latency",
            "prerequisites": "Ollama installed and qwen2 pulled",
            "status": "pending_backend",
            "notes": "No LLM inference is run in D5B; this is the first future local smoke test.",
        },
        {
            "phase": "D5D",
            "task": "llm_ranking_diagnostic",
            "dataset": "ml1m",
            "model": QWEN_MODEL,
            "backend": "ollama",
            "users": "20",
            "candidate_pool_size": "20",
            "prompt_strategy": "openp5_base",
            "output_format": "strict_json",
            "metrics": (
                "HR@5,NDCG@5,candidate_validity_rate,duplicate_rate,"
                "hallucination_rate,parse_failure_rate,runtime"
            ),
            "prerequisites": "D5C passed",
            "status": "pending_backend",
            "notes": "Small diagnostic only; no exact paper LLM metric claim.",
        },
        {
            "phase": "D5E",
            "task": "llm_ranking_diagnostic",
            "dataset": "ml1m",
            "model": "mistral:7b-instruct-v0.2-q4_0",
            "backend": "ollama",
            "users": "20",
            "candidate_pool_size": "20",
            "prompt_strategy": "openp5_base",
            "output_format": "strict_json",
            "metrics": "same metrics",
            "prerequisites": "D5C and Mistral pulled",
            "status": "pending_backend",
            "notes": "Second local model after Qwen path is stable.",
        },
        {
            "phase": "D5F",
            "task": "candidate_position_bias",
            "dataset": "ml1m",
            "model": QWEN_MODEL,
            "backend": "ollama",
            "users": "20",
            "candidate_pool_size": "20",
            "prompt_strategy": "openp5_base",
            "output_format": "strict_json",
            "metrics": "HR@5_by_position,position_bias_delta,top_position_selection_rate",
            "prerequisites": "D5D passed",
            "status": "pending_backend",
            "notes": "Requires real parsed LLM outputs; no synthetic outputs permitted.",
        },
        {
            "phase": "D5G",
            "task": "llm_reranking_diagnostic",
            "dataset": "ml1m",
            "model": QWEN_MODEL,
            "backend": "ollama",
            "users": "20",
            "candidate_pool_size": "20",
            "prompt_strategy": "openp5_base",
            "output_format": "strict_json",
            "metrics": "HR@5,NDCG@5,rerank_delta,validity,hallucination",
            "prerequisites": "D5D passed and baseline top20 candidates available",
            "status": "pending_backend",
            "notes": "Candidate-pool policy must be fixed before execution.",
        },
        {
            "phase": "D5H",
            "task": "output_validity_hallucination_report",
            "dataset": "ml1m",
            "model": "qwen2+mistral",
            "backend": "ollama",
            "users": "20",
            "candidate_pool_size": "20",
            "prompt_strategy": "openp5_base",
            "output_format": "strict_json",
            "metrics": (
                "valid_json_rate,candidate_validity_rate,hallucination_rate,"
                "duplicate_rate,empty_output_rate"
            ),
            "prerequisites": "D5D-D5G outputs",
            "status": "pending_backend",
            "notes": "Report only after real outputs exist.",
        },
        {
            "phase": "D5I",
            "task": "optional_openai_api_diagnostic",
            "dataset": "ml1m",
            "model": "gpt-4o-mini-2024-07-18",
            "backend": "openai_api",
            "users": "20",
            "candidate_pool_size": "20",
            "prompt_strategy": "openp5_base",
            "output_format": "strict_json",
            "metrics": "same metrics plus estimated_cost",
            "prerequisites": "explicit user approval and API key",
            "status": "conditional",
            "notes": "No OpenAI API calls were made in D5B.",
        },
        {
            "phase": "D5J",
            "task": "final_llm_blocker_or_results_integration",
            "dataset": "all",
            "model": "all_attempted",
            "backend": "mixed",
            "users": "n/a",
            "candidate_pool_size": "n/a",
            "prompt_strategy": "n/a",
            "output_format": "n/a",
            "metrics": "report integration",
            "prerequisites": "previous phases",
            "status": "pending",
            "notes": "Integrate real LLM results or an explicit backend blocker.",
        },
    ]


def write_all_outputs() -> None:
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    registry = _frame(model_registry_rows(), MODEL_REGISTRY_COLUMNS)
    audit_rows, context = audit_backend_rows()
    audit = _frame(audit_rows, BACKEND_AUDIT_COLUMNS)
    roadmap = _frame(task_roadmap_rows(), TASK_ROADMAP_COLUMNS)

    registry.to_csv(MODEL_REGISTRY_PATH, index=False)
    audit.to_csv(BACKEND_AUDIT_PATH, index=False)
    roadmap.to_csv(TASK_ROADMAP_PATH, index=False)

    _write_report(context)
    _upsert_feasibility_ledger(context)
    _write_next_actions(context)

    LOGGER.info("Wrote %s", MODEL_REGISTRY_PATH)
    LOGGER.info("Wrote %s", BACKEND_AUDIT_PATH)
    LOGGER.info("Wrote %s", TASK_ROADMAP_PATH)
    LOGGER.info("Wrote %s", REPORT_PATH)


def _write_report(context: BackendContext) -> None:
    if not context.ollama_exists:
        audit_summary = (
            "Ollama command was not found on PATH. The local Ollama backend is "
            "`blocked_missing_backend`, and D5C cannot start until Ollama is "
            "installed manually and `qwen2:7b-instruct` is pulled by the user."
        )
        next_step = (
            "Install Ollama manually, then rerun the D5B backend audit or proceed "
            "to D5C only after manually pulling `qwen2:7b-instruct`."
        )
    elif not context.qwen_available:
        audit_summary = (
            "Ollama command is available, but `qwen2:7b-instruct` was not found "
            "in `ollama list`. The backend is available but D5C remains "
            "`pending_backend` until the first local model is present."
        )
        next_step = (
            "The user may manually run `ollama pull qwen2:7b-instruct`; Mistral "
            "pull commands remain optional recommendations and were not executed."
        )
    else:
        audit_summary = (
            "Ollama command is available and `qwen2:7b-instruct` appears in the "
            "local model list. D5C can proceed as a smoke test if the user approves "
            "the next execution phase."
        )
        next_step = "Proceed to the D5C Qwen smoke test."

    mistral_status = (
        "available in `ollama list`"
        if context.mistral_available
        else "not found in `ollama list` or not checked because Ollama is missing"
    )
    cuda_status = (
        f"available (`{context.cuda_device_name}`)"
        if context.cuda_available
        else "not available or torch/CUDA could not be used"
    )

    text = f"""# Plan D LLM Model Registry and Backend Feasibility Audit

D5B is a documentation and local audit phase only. No LLM inference was run, no LLM recommendations were generated, and no API calls/downloads were performed.

The goal is methodological traceability before any LLM experiment. The registry records which paper LLMs can be reproduced exactly, reconstructed with a local open model, approximated with a proxy, or only documented as blocked/conditional.

## Paper LLMs Covered

- `gpt-3.5-turbo-0125`
- `gpt-4o-mini-2024-07-18`
- `gpt-4o-2024-08-06`
- `claude-3-haiku-20240307`
- `llama-3-70b-instruct`
- `qwen-2-7b-instruct`
- `mistral-7b-instruct-v0.2`

## Reproduction Levels

Exact reproduction means using the same model identifier or a verifiably still-available API snapshot. Reconstructed local reproduction means using an open-weight family/model path that is close enough for a clearly labeled diagnostic, without claiming exact numerical reproduction. Proxy reproduction means a replacement model is used because the exact model is unavailable; this must be labeled non-exact.

No exact LLM reproduction is claimed in D5B.

## Recommended Local Path

1. Use Ollama as the local backend.
2. Start with `qwen2:7b-instruct`.
3. Add `mistral:7b-instruct-v0.2-q4_0` or `mistral:7b-instruct-v0.2-q4_K_M` as the second local model.

Qwen is the first local target because it is a 7B open local model, feasible in quantized form on the RTX 4060 8GB planning target, and gives the project the shortest path to D5C smoke testing and D5D ranking diagnostics. Mistral is the second target because it is another feasible 7B instruct model and is useful for comparing hallucination and output-validity behavior against Qwen.

The following commands are recommendations only and were not executed:

```powershell
ollama pull qwen2:7b-instruct
ollama pull mistral:7b-instruct-v0.2-q4_0
ollama pull mistral:7b-instruct-v0.2-q4_K_M
```

`llama-3-70b-instruct` is not feasible locally on an RTX 4060 8GB target. Even though it is open weight, practical evaluation requires cloud hardware or a provider API, so it is documented as `blocked_local_hardware` / cloud-or-API required.

## Closed-Source API Path

- `gpt-4o-mini-2024-07-18` is the best first optional OpenAI API model if the user explicitly approves API access and budget.
- `gpt-4o-2024-08-06` is stronger but more expensive, so it should be limited to a small diagnostic subset if approved.
- `gpt-3.5-turbo-0125` is a legacy model whose availability must be verified with an API model-list check before any exact claim.
- `claude-3-haiku-20240307` is retired for exact reproduction; a current Claude Haiku replacement would be a non-exact proxy and would require explicit approval.

No OpenAI API calls and no Anthropic/Claude API calls were made in D5B.

## Local Backend Audit Result

{audit_summary}

Additional local audit facts:

- Mistral local model status: {mistral_status}.
- Torch importable: `{str(context.torch_importable).lower()}`.
- CUDA status from torch: {cuda_status}.
- LastFM is not required here; D5B only audits LLM model/backend feasibility and does not run LastFM experiments.

D5C cannot start unless Ollama and at least `qwen2:7b-instruct` are available locally.

## Recommended Next Step

{next_step}

If Ollama is missing, install Ollama manually, then rerun D5B backend audit or proceed to D5C smoke test after pulling Qwen. If Ollama exists but Qwen is missing, the user may manually run `ollama pull qwen2:7b-instruct`. If Qwen exists, proceed to the D5C smoke test.

## Documentation Claims

- No exact LLM reproduction is claimed yet.
- Local open-model reproduction is planned.
- Closed API reproduction is conditional.
- Retired or unavailable models are documented honestly.
- The LLM blocker will be documented if the backend remains unavailable.

## Generated Artifacts

- `results/tables/plan_d_llm_model_registry.csv`
- `results/tables/plan_d_llm_backend_audit.csv`
- `results/tables/plan_d_llm_task_roadmap.csv`
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def _upsert_feasibility_ledger(context: BackendContext) -> None:
    frame = _read_or_empty(FEASIBILITY_LEDGER_PATH, LEDGER_COLUMNS)
    rows = _ledger_rows(context)
    updated = _upsert_rows(frame, rows, "item_id", LEDGER_COLUMNS)
    updated.to_csv(FEASIBILITY_LEDGER_PATH, index=False)


def _ledger_rows(context: BackendContext) -> list[dict[str, str]]:
    local_status = "feasible_with_backend" if context.qwen_available else "blocked_missing_backend"
    qwen_status = "feasible_with_backend" if context.qwen_available else "blocked_missing_backend"
    mistral_status = "feasible_with_backend" if context.mistral_available else "blocked_missing_backend"
    ollama_note = (
        "Ollama and qwen2 are available; D5C can proceed if the user approves."
        if context.qwen_available
        else "D5C cannot start until Ollama and qwen2:7b-instruct are available locally."
    )
    return [
        {
            "item_id": "local_llm_backend",
            "area": "llm_backend",
            "target": "Local LLM runtime for D5",
            "feasibility_status": local_status,
            "required_resources": "Ollama plus qwen2:7b-instruct for the first local smoke test",
            "estimated_effort": "low to medium",
            "risk_level": "high",
            "external_dependency": "yes",
            "recommended_decision": "Prefer local Ollama path before any API-backed LLM experiment.",
            "notes": f"D5B audit: {ollama_note} No inference, downloads, or API calls were performed.",
        },
        {
            "item_id": "ollama_backend",
            "area": "llm_backend",
            "target": "Ollama local serving",
            "feasibility_status": local_status,
            "required_resources": "Installed Ollama binary and qwen2:7b-instruct local model",
            "estimated_effort": "low",
            "risk_level": "medium",
            "external_dependency": "yes",
            "recommended_decision": "Use for D5C only after the user manually installs Ollama and pulls qwen2.",
            "notes": f"D5B audit: {ollama_note}",
        },
        {
            "item_id": "local_ollama_backend",
            "area": "llm_backend",
            "target": "Ollama backend with first local Qwen model",
            "feasibility_status": local_status,
            "required_resources": "ollama command plus qwen2:7b-instruct in ollama list",
            "estimated_effort": "low",
            "risk_level": "medium",
            "external_dependency": "yes",
            "recommended_decision": "Gate D5C on this row becoming feasible_with_backend.",
            "notes": f"D5B audit: {ollama_note}",
        },
        {
            "item_id": "closed_api_llms",
            "area": "llm_backend",
            "target": "GPT/Claude-style remote models",
            "feasibility_status": "feasible_with_api_budget",
            "required_resources": "API keys, budget, explicit allow-api-calls policy, and user approval",
            "estimated_effort": "medium",
            "risk_level": "high",
            "external_dependency": "yes",
            "recommended_decision": "Keep conditional; do not call APIs unless the user explicitly changes scope.",
            "notes": "D5B made no OpenAI or Anthropic calls. GPT-4o-mini/GPT-4o are conditional; Claude exact model is retired.",
        },
        {
            "item_id": "paper_llm_gpt_3_5_turbo_0125",
            "area": "llm_model",
            "target": "gpt-3.5-turbo-0125",
            "feasibility_status": "not_recommended",
            "required_resources": "OpenAI API key, budget, user approval, and model-list availability verification",
            "estimated_effort": "low if API approved",
            "risk_level": "high",
            "external_dependency": "yes",
            "recommended_decision": "Document as conditional legacy API; do not run in local-only D5.",
            "notes": "registry_status=conditional_legacy_api; reproduction_level=exact_only_if_api_model_available.",
        },
        {
            "item_id": "paper_llm_gpt_4o_mini_2024_07_18",
            "area": "llm_model",
            "target": "gpt-4o-mini-2024-07-18",
            "feasibility_status": "feasible_with_api_budget",
            "required_resources": "OpenAI API key, budget, and explicit user approval",
            "estimated_effort": "low if API approved",
            "risk_level": "high",
            "external_dependency": "yes",
            "recommended_decision": "Best first optional closed-source diagnostic if API budget is approved.",
            "notes": "registry_status=feasible_with_api_budget; no API call was performed in D5B.",
        },
        {
            "item_id": "paper_llm_gpt_4o_2024_08_06",
            "area": "llm_model",
            "target": "gpt-4o-2024-08-06",
            "feasibility_status": "feasible_with_api_budget",
            "required_resources": "OpenAI API key, budget, and explicit user approval",
            "estimated_effort": "low if API approved",
            "risk_level": "high",
            "external_dependency": "yes",
            "recommended_decision": "Use only for a small diagnostic subset if approved.",
            "notes": "registry_status=feasible_with_api_budget; stronger but more expensive optional API path.",
        },
        {
            "item_id": "paper_llm_claude_3_haiku_20240307",
            "area": "llm_model",
            "target": "claude-3-haiku-20240307",
            "feasibility_status": "not_recommended",
            "required_resources": "Anthropic API key plus non-exact replacement approval",
            "estimated_effort": "low if proxy approved",
            "risk_level": "high",
            "external_dependency": "yes",
            "recommended_decision": "Do not claim exact reproduction; exact model is retired.",
            "notes": "registry_status=blocked_exact_model_retired; any current Claude Haiku would be a non-exact proxy.",
        },
        {
            "item_id": "paper_llm_llama_3_70b_instruct",
            "area": "llm_model",
            "target": "llama-3-70b-instruct",
            "feasibility_status": "not_recommended",
            "required_resources": "Cloud GPU or provider API; RTX 4060 8GB is not sufficient",
            "estimated_effort": "medium to high",
            "risk_level": "high",
            "external_dependency": "yes",
            "recommended_decision": "Document as blocked locally; do not attempt on the target laptop GPU.",
            "notes": "registry_status=blocked_local_hardware; cloud_or_api_required for practical evaluation.",
        },
        {
            "item_id": "paper_llm_qwen_2_7b_instruct",
            "area": "llm_model",
            "target": "qwen-2-7b-instruct",
            "feasibility_status": qwen_status,
            "required_resources": "Ollama and qwen2:7b-instruct pulled locally",
            "estimated_effort": "low after backend setup",
            "risk_level": "medium",
            "external_dependency": "yes",
            "recommended_decision": "Use as the first local D5C/D5D model after backend is ready.",
            "notes": "registry_status=recommended_first_local_model; planned_use=D5C smoke test, then ranking/position/reranking diagnostics.",
        },
        {
            "item_id": "paper_llm_mistral_7b_instruct_v0_2",
            "area": "llm_model",
            "target": "mistral-7b-instruct-v0.2",
            "feasibility_status": mistral_status,
            "required_resources": "Ollama and a quantized mistral-7b-instruct-v0.2 model pulled locally",
            "estimated_effort": "low after backend setup",
            "risk_level": "medium",
            "external_dependency": "yes",
            "recommended_decision": "Use as the second local comparison model after Qwen smoke test.",
            "notes": "registry_status=recommended_second_local_model; compare hallucination/output validity against Qwen.",
        },
    ]


def _write_next_actions(context: BackendContext) -> None:
    ollama_note = (
        "Current audit: Ollama command exists."
        if context.ollama_exists
        else "Current audit: Ollama command is missing."
    )
    qwen_note = (
        "Current audit: qwen2 is available, so this may already be satisfied."
        if context.qwen_available
        else "Current audit: qwen2 is not available for D5C."
    )
    mistral_note = (
        "Current audit: a Mistral v0.2 quantized model is available."
        if context.mistral_available
        else "Current audit: Mistral v0.2 quantized model is not available."
    )
    rows = [
        {
            "priority_rank": "1",
            "action_id": "install_or_confirm_ollama_backend",
            "action_title": "Install or confirm Ollama backend",
            "related_component": "local_ollama_backend",
            "why_it_matters": "D5C-D5H local LLM diagnostics require a real local backend.",
            "expected_output": "Ollama available on PATH and documented in the backend audit.",
            "prerequisites": "Manual user installation if missing; no automated install in this project.",
            "risk": "high",
            "recommended_order": "next",
            "notes": ollama_note,
        },
        {
            "priority_rank": "2",
            "action_id": "pull_qwen2_7b_instruct",
            "action_title": "Pull qwen2 7B instruct",
            "related_component": "paper_llm_qwen_2_7b_instruct",
            "why_it_matters": "Qwen is the first local model target for D5C smoke testing.",
            "expected_output": "qwen2:7b-instruct appears in ollama list.",
            "prerequisites": "Ollama installed; manual user pull only.",
            "risk": "medium",
            "recommended_order": "after Ollama",
            "notes": qwen_note,
        },
        {
            "priority_rank": "3",
            "action_id": "run_d5c_qwen2_smoke_test",
            "action_title": "Run D5C Qwen smoke test",
            "related_component": "ollama_smoke_test",
            "why_it_matters": "The project needs a minimal parser/latency check before ranking diagnostics.",
            "expected_output": "D5C smoke-test audit rows and no fabricated recommendation outputs.",
            "prerequisites": "Ollama installed and qwen2:7b-instruct available.",
            "risk": "medium",
            "recommended_order": "after qwen2 pull",
            "notes": "Do not run until backend is ready and the task scope is approved.",
        },
        {
            "priority_rank": "4",
            "action_id": "implement_llm_prompt_templates_and_parser",
            "action_title": "Implement LLM prompt templates and parser",
            "related_component": "llm_ranking; hallucination",
            "why_it_matters": "Strict JSON parsing is required before any real LLM outputs are evaluated.",
            "expected_output": "Prompt templates, parser tests, and validation rules.",
            "prerequisites": "D5C design confirmed.",
            "risk": "medium",
            "recommended_order": "before D5D",
            "notes": "No LLM inference is needed to implement or test parser fixtures.",
        },
        {
            "priority_rank": "5",
            "action_id": "run_qwen2_ml1m_ranking_diagnostic_20_users",
            "action_title": "Run Qwen ML-1M ranking diagnostic",
            "related_component": "llm_ranking",
            "why_it_matters": "This is the smallest real local LLM ranking experiment.",
            "expected_output": "20-user ML-1M ranking diagnostics with strict validity and hallucination metrics.",
            "prerequisites": "D5C passed and parser ready.",
            "risk": "high",
            "recommended_order": "after parser",
            "notes": "Use real outputs only; do not fabricate raw outputs.",
        },
        {
            "priority_rank": "6",
            "action_id": "pull_mistral_7b_instruct_v02_quantized",
            "action_title": "Pull Mistral 7B instruct v0.2 quantized",
            "related_component": "paper_llm_mistral_7b_instruct_v0_2",
            "why_it_matters": "Mistral is the second local model for output-validity comparison.",
            "expected_output": "A recommended Mistral v0.2 quantized model appears in ollama list.",
            "prerequisites": "Ollama installed; manual user pull only.",
            "risk": "medium",
            "recommended_order": "after Qwen path works",
            "notes": mistral_note,
        },
        {
            "priority_rank": "7",
            "action_id": "run_mistral_ml1m_ranking_diagnostic_20_users",
            "action_title": "Run Mistral ML-1M ranking diagnostic",
            "related_component": "llm_ranking; paper_llm_mistral_7b_instruct_v0_2",
            "why_it_matters": "A second local model helps compare validity and hallucination behavior.",
            "expected_output": "20-user Mistral ML-1M ranking diagnostic metrics.",
            "prerequisites": "D5C passed, parser ready, Mistral pulled.",
            "risk": "high",
            "recommended_order": "after Qwen D5D",
            "notes": "Keep as diagnostic; no exact paper reproduction claim.",
        },
        {
            "priority_rank": "8",
            "action_id": "run_candidate_position_bias_diagnostic",
            "action_title": "Run candidate position-bias diagnostic",
            "related_component": "position_bias",
            "why_it_matters": "Candidate position bias is meaningful only with real LLM selections.",
            "expected_output": "Position-bias metrics from real parsed LLM outputs.",
            "prerequisites": "D5D passed.",
            "risk": "medium",
            "recommended_order": "after Qwen ranking diagnostic",
            "notes": "Do not compute from fake or empty outputs.",
        },
        {
            "priority_rank": "9",
            "action_id": "run_llm_reranking_diagnostic",
            "action_title": "Run LLM reranking diagnostic",
            "related_component": "llm_reranking",
            "why_it_matters": "Reranking is closer to the paper task but needs a stable candidate-pool policy.",
            "expected_output": "Reranking diagnostics with HR@5, NDCG@5, validity, and hallucination metrics.",
            "prerequisites": "D5D passed and baseline top20 candidate source fixed.",
            "risk": "high",
            "recommended_order": "after position-bias diagnostic",
            "notes": "Keep as diagnostic_subset unless official candidate pools are recovered.",
        },
        {
            "priority_rank": "10",
            "action_id": "integrate_llm_results_or_blocker_into_final_report",
            "action_title": "Integrate LLM results or blocker into final report",
            "related_component": "final_report; local_llm_backend; api_availability",
            "why_it_matters": "The final report must close the LLM loop with real results or a documented blocker.",
            "expected_output": "Final LLM section grounded in generated CSVs and audit artifacts.",
            "prerequisites": "D5C-D5H outputs or explicit backend blocker.",
            "risk": "low",
            "recommended_order": "after attempted LLM path or blocker decision",
            "notes": "Do not manually invent metric values.",
        },
    ]
    _frame(rows, NEXT_ACTION_COLUMNS).to_csv(NEXT_ACTIONS_PATH, index=False)


def _python_import_rows() -> list[dict[str, str]]:
    checks = [
        ("requests", "requests_import", "requests"),
        ("openai", "openai_import", "openai"),
        ("anthropic", "anthropic_import", "anthropic"),
        ("transformers", "transformers_import", "transformers"),
        ("llama_cpp", "llama_cpp_import", "llama-cpp-python"),
    ]
    rows: list[dict[str, str]] = []
    for module_name, check_name, distribution_name in checks:
        available = importlib.util.find_spec(module_name) is not None
        rows.append(
            {
                "backend": "python",
                "check_name": check_name,
                "command_or_import": f"import {module_name}",
                "available": _bool_text(available),
                "version_or_result": _package_version(distribution_name) if available else "not_importable",
                "status": "available" if available else "missing",
                "notes": "Importability check only; no HTTP request or API call was made.",
            }
        )
    return rows


def _torch_rows() -> tuple[list[dict[str, str]], bool, bool, str]:
    rows: list[dict[str, str]] = []
    if importlib.util.find_spec("torch") is None:
        rows.extend(
            [
                {
                    "backend": "torch",
                    "check_name": "torch_import",
                    "command_or_import": "import torch",
                    "available": "false",
                    "version_or_result": "not_importable",
                    "status": "missing",
                    "notes": "Torch is not importable; CUDA checks are marked not_applicable.",
                },
                {
                    "backend": "torch",
                    "check_name": "cuda_available",
                    "command_or_import": "torch.cuda.is_available()",
                    "available": "false",
                    "version_or_result": "not_available",
                    "status": "not_applicable",
                    "notes": "Not checked because torch is missing.",
                },
                {
                    "backend": "torch",
                    "check_name": "cuda_device_name",
                    "command_or_import": "torch.cuda.get_device_name(0)",
                    "available": "false",
                    "version_or_result": "not_available",
                    "status": "not_applicable",
                    "notes": "Not checked because torch is missing.",
                },
            ]
        )
        return rows, False, False, "not_available"

    try:
        import torch

        torch_version = getattr(torch, "__version__", "imported")
        cuda_available = bool(torch.cuda.is_available())
        cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else "not_available"
        rows.extend(
            [
                {
                    "backend": "torch",
                    "check_name": "torch_import",
                    "command_or_import": "import torch",
                    "available": "true",
                    "version_or_result": str(torch_version),
                    "status": "available",
                    "notes": "Imported locally for CUDA availability inspection only.",
                },
                {
                    "backend": "torch",
                    "check_name": "cuda_available",
                    "command_or_import": "torch.cuda.is_available()",
                    "available": _bool_text(cuda_available),
                    "version_or_result": _bool_text(cuda_available),
                    "status": "available" if cuda_available else "missing",
                    "notes": "Local CUDA availability check only; no model inference was run.",
                },
                {
                    "backend": "torch",
                    "check_name": "cuda_device_name",
                    "command_or_import": "torch.cuda.get_device_name(0)",
                    "available": _bool_text(cuda_available),
                    "version_or_result": cuda_device_name,
                    "status": "available" if cuda_available else "not_applicable",
                    "notes": "Device name recorded only when CUDA is available.",
                },
            ]
        )
        return rows, True, cuda_available, cuda_device_name
    except Exception as exc:  # pragma: no cover - depends on local torch installation
        message = _clean_text(str(exc), 240)
        rows.extend(
            [
                {
                    "backend": "torch",
                    "check_name": "torch_import",
                    "command_or_import": "import torch",
                    "available": "false",
                    "version_or_result": message,
                    "status": "missing",
                    "notes": "Torch import raised locally; CUDA checks are not_applicable.",
                },
                {
                    "backend": "torch",
                    "check_name": "cuda_available",
                    "command_or_import": "torch.cuda.is_available()",
                    "available": "false",
                    "version_or_result": "not_available",
                    "status": "not_applicable",
                    "notes": "Not checked because torch import failed.",
                },
                {
                    "backend": "torch",
                    "check_name": "cuda_device_name",
                    "command_or_import": "torch.cuda.get_device_name(0)",
                    "available": "false",
                    "version_or_result": "not_available",
                    "status": "not_applicable",
                    "notes": "Not checked because torch import failed.",
                },
            ]
        )
        return rows, False, False, "not_available"


@dataclass(frozen=True)
class LocalCommandResult:
    returncode: int
    output: str


def _run_local_command(args: list[str]) -> LocalCommandResult:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=12,
        )
    except Exception as exc:  # pragma: no cover - depends on local executable state
        return LocalCommandResult(returncode=1, output=_clean_text(str(exc), 300))

    output = completed.stdout.strip() or completed.stderr.strip()
    return LocalCommandResult(
        returncode=completed.returncode,
        output=_clean_text(output or "no_output", 500),
    )


def _parse_ollama_model_names(output: str) -> tuple[str, ...]:
    names: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        first = stripped.split()[0]
        if first.upper() == "NAME":
            continue
        names.append(first)
    return tuple(sorted(set(names), key=str.lower))


def _has_mistral_model(models: Iterable[str]) -> bool:
    lowered = {model.lower() for model in models}
    recommended = {model.lower() for model in MISTRAL_MODELS}
    return bool(lowered & recommended) or any(
        model.startswith("mistral:7b-instruct-v0.2") for model in lowered
    )


def _models_result_text(models: tuple[str, ...]) -> str:
    if not models:
        return "no_models_listed"
    return ";".join(models)


def _package_version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "importable_version_unknown"


def _read_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype=str).fillna("")
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame[columns]


def _upsert_rows(
    frame: pd.DataFrame,
    rows: list[dict[str, str]],
    key_column: str,
    columns: list[str],
) -> pd.DataFrame:
    updated = frame.copy()
    for raw_row in rows:
        row = {column: str(raw_row.get(column, "")) for column in columns}
        key = row[key_column]
        mask = updated[key_column] == key
        if mask.any():
            updated.loc[mask, columns] = [row[column] for column in columns]
        else:
            updated = pd.concat([updated, pd.DataFrame([row], columns=columns)], ignore_index=True)
    return updated[columns]


def _frame(rows: list[dict[str, str]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{column: str(row.get(column, "")) for column in columns} for row in rows],
        columns=columns,
    )


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _clean_text(value: str, limit: int) -> str:
    cleaned = " ".join(str(value).split())
    return cleaned[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write the Plan D D5B LLM model registry and local backend audit. "
            "This command does not run inference, download models, or call APIs."
        )
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    write_all_outputs()


if __name__ == "__main__":
    main()
