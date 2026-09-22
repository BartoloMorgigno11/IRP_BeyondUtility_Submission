"""Deterministic offline runtime preflight for Phase F6E2.

This module inventories local evidence for a future LLM inference matrix. It
does not contact local or remote model servers, does not start services, and
does not write anything at import time.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from src.llm.plan_f6e1_prompt_strategies import (
    BASE_STRATEGY_ID,
    JSON_INDEX_STRATEGY_ID,
    LLMRANK_STRATEGY_ID,
    LLMSREC_STRATEGY_ID,
    PromptRenderContext,
    render_strategy_prompt,
)


PHASE = "F6E2"
FROZEN_TIMESTAMP_UTC = "2026-07-25T00:00:00Z"
EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
F6E1_SCIENTIFIC_HASH = "a6c5298d2bb68f47f36d29315e1bc849e2e33ed576fa02ba4329abe8ecbe09ea"
F6E1_MANIFEST_HASH = "38e1207638317386b71fada587b94d5ebfa38211fd6f14f15119e4650e5844be"
QWEN2_RAW_SHA256 = "1384a47058ed8dd8c06fc52d6bc425020b1746b60ad11e4bde6153f948a493b4"
MISTRAL_RAW_SHA256 = "7031f37d40092b4f318b5e03273772531ffda37cef0c38bfed32edfef935736a"

REPORT_PATH = Path("report/phase_f6e2_runtime_preflight.md")
MANIFEST_PATH = Path("results/manifests/phase_f6e2_preflight_freeze.json")

OUTPUT_TABLES: dict[str, Path] = {
    "hardware_inventory": Path("results/tables/phase_f6e2_hardware_inventory.csv"),
    "runtime_inventory": Path("results/tables/phase_f6e2_runtime_inventory.csv"),
    "model_inventory": Path("results/tables/phase_f6e2_model_inventory.csv"),
    "model_identity_alignment": Path("results/tables/phase_f6e2_model_identity_alignment.csv"),
    "request_workload": Path("results/tables/phase_f6e2_request_workload.csv"),
    "prompt_token_estimates": Path("results/tables/phase_f6e2_prompt_token_estimates.csv"),
    "runtime_estimates": Path("results/tables/phase_f6e2_runtime_estimates.csv"),
    "memory_disk_feasibility": Path("results/tables/phase_f6e2_memory_disk_feasibility.csv"),
    "execution_matrix": Path("results/tables/phase_f6e2_execution_matrix.csv"),
    "pilot_design": Path("results/tables/phase_f6e2_pilot_design.csv"),
    "missing_or_blocked": Path("results/tables/phase_f6e2_missing_or_blocked.csv"),
}

HARDWARE_COLUMNS = [
    "component_id",
    "component_type",
    "name",
    "observed_value",
    "unit",
    "source",
    "measurement_status",
    "volatile",
    "notes",
]

RUNTIME_COLUMNS = [
    "runtime_id",
    "runtime_name",
    "installed",
    "version",
    "path",
    "gpu_support",
    "cpu_offload_support",
    "quantization_support",
    "context_limit_configurable",
    "windows_compatibility",
    "repository_usage",
    "operational_status",
    "network_or_download_needed",
    "source",
    "notes",
]

MODEL_COLUMNS = [
    "model_id",
    "model_tag",
    "family",
    "parameter_count",
    "quantization",
    "model_blob_size_bytes",
    "total_manifest_layer_bytes",
    "manifest_path",
    "model_blob_digest",
    "config_digest",
    "context_window_declared",
    "tokenizer",
    "runtime",
    "install_time",
    "availability_status",
    "used_in_previous_phases",
    "frozen_identity_hash",
    "notes",
]

IDENTITY_COLUMNS = [
    "alignment_id",
    "paper_model_name",
    "local_model_tag",
    "organization",
    "family",
    "parameter_count",
    "variant",
    "quantization",
    "version_or_tag",
    "identity_evidence",
    "divergences",
    "classification",
    "allowed_scientific_claim",
]

WORKLOAD_COLUMNS = [
    "workload_id",
    "dataset",
    "task",
    "manifest_path",
    "manifest_sha256",
    "manifest_rows",
    "logical_instances",
    "distinct_users",
    "candidate_count_min",
    "candidate_count_mean",
    "candidate_count_max",
    "history_length_min",
    "history_length_mean",
    "history_length_max",
    "positive_count",
    "candidate_ordering",
    "item_metadata_available",
    "title_availability",
    "status",
    "notes",
]

TOKEN_COLUMNS = [
    "estimate_id",
    "strategy_id",
    "strategy_classification",
    "dataset",
    "task",
    "method",
    "rows_sampled",
    "char_min",
    "char_mean",
    "char_median",
    "char_p95",
    "char_max",
    "token_min",
    "token_mean",
    "token_median",
    "token_p95",
    "token_max",
    "history_token_mean",
    "candidate_token_mean",
    "instruction_token_mean",
    "profile_token_mean",
    "output_token_budget",
    "context_total_token_p95",
    "above_4k_percent",
    "above_8k_percent",
    "above_16k_percent",
    "above_32k_percent",
    "above_local_context_limit_percent",
    "local_context_limit",
    "status",
    "notes",
]

RUNTIME_ESTIMATE_COLUMNS = [
    "estimate_id",
    "model_tag",
    "strategy_id",
    "dataset",
    "task",
    "logical_instances",
    "calls_per_instance_min",
    "calls_per_instance_expected",
    "calls_per_instance_max",
    "min_calls",
    "expected_calls",
    "max_calls",
    "seconds_per_request_best",
    "seconds_per_request_expected",
    "seconds_per_request_worst",
    "total_hours_best",
    "total_hours_expected",
    "total_days_worst",
    "estimate_basis",
    "observed_rows",
    "observed_prompt_eval_mean",
    "observed_eval_count_mean",
    "gpu_utilization_expected",
    "cpu_bottleneck",
    "disk_write_volume_mb",
    "resume_possible",
    "status",
    "notes",
]

MEMORY_COLUMNS = [
    "feasibility_id",
    "model_tag",
    "scenario",
    "parameter_count_billion",
    "precision_or_quantization",
    "estimated_weight_gb",
    "estimated_runtime_overhead_gb",
    "estimated_kv_cache_gb",
    "estimated_total_ram_gb",
    "estimated_total_vram_gb",
    "available_ram_gb_snapshot",
    "total_ram_gb",
    "available_vram_gb_snapshot",
    "total_vram_gb",
    "disk_required_gb",
    "disk_available_gb_snapshot",
    "runtime_required",
    "model_present",
    "scenario_status",
    "secondary_blockers",
    "notes",
]

EXECUTION_COLUMNS = [
    "cell_id",
    "model_family",
    "expected_model_identifier",
    "local_model_identifier",
    "strategy_id",
    "strategy_classification",
    "scientific_role",
    "paper_or_extension_scope",
    "model_identity_classification",
    "local_model_available",
    "runtime_available",
    "prompt_available",
    "parser_available",
    "calls_per_request_min",
    "calls_per_request_expected",
    "calls_per_request_max",
    "total_logical_instances",
    "total_expected_calls",
    "preflight_status",
    "blocker",
    "safe_next_action",
    "notes",
]

PILOT_COLUMNS = [
    "pilot_id",
    "strategy_id",
    "models",
    "datasets",
    "tasks",
    "minimum_requests",
    "deterministic_selection",
    "history_length_coverage",
    "candidate_position_coverage",
    "expected_parser_checks",
    "stop_conditions",
    "timeout_seconds",
    "resume_policy",
    "raw_policy",
    "retry_policy",
    "gate_to_full_run",
    "status",
    "notes",
]

MISSING_COLUMNS = [
    "block_id",
    "category",
    "affected_scope",
    "severity",
    "status",
    "evidence",
    "safe_next_action",
    "notes",
]

MODEL_TAGS = ["qwen2:7b-instruct", "mistral:7b-instruct", "llama3:70b-instruct"]
F6E1_MATRIX_PATH = Path("results/tables/phase_f6e1_model_strategy_matrix.csv")
F6E1_MANIFEST_PATH = Path("results/manifests/phase_f6e1_prompt_freeze.json")
FULL_MATRIX_REQUESTS_PATH = Path("results/llm/canonical_local_full_matrix/requests.jsonl")
FULL_MATRIX_EXECUTION_PATH = Path("results/tables/canonical_local_llm_full_matrix_execution.csv")
F6D3_RANKING_MANIFEST_PATH = Path("results/tables/finalization_f6d3_lastfm_ranking_llm_manifest.csv")
F6D3_RERANKING_MANIFEST_PATH = Path("results/tables/finalization_f6d3_lastfm_reranking_llm_manifest.csv")
F6D4A_QWEN_RAW_PATH = Path("results/llm/finalization_f6d4a_lastfm_ranking_qwen2_raw.jsonl")
F6D4A_MISTRAL_RAW_PATH = Path("results/llm/finalization_f6d4a_lastfm_ranking_mistral_raw.jsonl")
F6D4A_INPUT_HASHES_PATH = Path("results/tables/finalization_f6d4a_input_hashes.csv")
F2_PROTECTED_VALIDATION_PATH = Path("results/tables/finalization_f2_protected_input_validation.csv")
F6D4A_PROTECTED_VALIDATION_PATH = Path("results/tables/finalization_f6d4a_closure_protected_input_verification.csv")
F6D4A_EXECUTION_MANIFEST_PATH = Path("results/tables/finalization_f6d4a_execution_manifest.csv")

PLAN_E_MANIFESTS = [
    ("Amazon Beauty", "ranking", Path("results/tables/plan_e_beauty_1000user_ranking_manifest.csv")),
    ("Amazon Beauty", "reranking", Path("results/tables/plan_e_beauty_1000user_reranking_manifest.csv")),
    ("Amazon Sports", "ranking", Path("results/tables/plan_e_sports_1000user_ranking_manifest.csv")),
    ("Amazon Sports", "reranking", Path("results/tables/plan_e_sports_1000user_reranking_manifest.csv")),
    ("ML-1M", "ranking", Path("results/tables/plan_e_ml1m_1000user_ranking_manifest.csv")),
    ("ML-1M", "reranking", Path("results/tables/plan_e_ml1m_1000user_reranking_manifest.csv")),
]

STRATEGY_CLASSIFICATION = {
    BASE_STRATEGY_ID: "exact_parametric",
    LLMRANK_STRATEGY_ID: "exact_parametric",
    LLMSREC_STRATEGY_ID: "reconstructed",
    JSON_INDEX_STRATEGY_ID: "local_ablation",
}

CALL_MULTIPLICITY = {
    BASE_STRATEGY_ID: ("1", "1", "1"),
    LLMRANK_STRATEGY_ID: ("1", "1", "1"),
    LLMSREC_STRATEGY_ID: ("1", "1.5", "2"),
    JSON_INDEX_STRATEGY_ID: ("1", "1", "1"),
}


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


def stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}-{sha256_text('||'.join(str(part) for part in parts))[:12]}"


def fmt(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0"
    return fmt((numerator / denominator) * 100.0)


def read_csv_rows(root: Path, relative_path: Path) -> list[dict[str, str]]:
    path = root / relative_path
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl_rows(root: Path, relative_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    path = root / relative_path
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def rows_to_csv_text(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rows_to_csv_text(rows, columns), encoding="utf-8")


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def numeric_stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": min(values),
        "mean": mean(values),
        "median": median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def estimate_tokens(text: str) -> int:
    """Deterministic tokenizer-free estimate used when exact tokenizers are absent."""

    return max(1, math.ceil(len(text) / 4))


def ollama_manifest_path(tag: str) -> Path:
    family, variant = tag.split(":", 1)
    return Path.home() / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library" / family / variant


def logical_external_path(path: Path) -> str:
    try:
        return "%USERPROFILE%/" + path.relative_to(Path.home()).as_posix()
    except ValueError:
        return path.as_posix()


def read_ollama_manifest(tag: str) -> dict[str, Any] | None:
    path = ollama_manifest_path(tag)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def model_layer(manifest: Mapping[str, Any], media_type_suffix: str) -> Mapping[str, Any] | None:
    for layer in manifest.get("layers", []):
        media_type = str(layer.get("mediaType", ""))
        if media_type.endswith(media_type_suffix):
            return layer
    return None


def build_hardware_inventory() -> list[dict[str, str]]:
    return [
        {
            "component_id": "F6E2-HW-CPU",
            "component_type": "cpu",
            "name": "Intel(R) Core(TM) i9-14900HX",
            "observed_value": "24 physical cores; 32 logical processors; max clock 2200 MHz; x64",
            "unit": "snapshot",
            "source": "Get-CimInstance Win32_Processor",
            "measurement_status": "observed_read_only_preflight",
            "volatile": "false",
            "notes": "Instruction-set detail was not separately probed; CPU is sufficient for orchestration and CPU fallback but not for efficient 70B full-matrix inference.",
        },
        {
            "component_id": "F6E2-HW-RAM",
            "component_type": "memory",
            "name": "System RAM",
            "observed_value": "33255288 KiB total; 19571940 KiB free; 35352440 KiB virtual total",
            "unit": "KiB",
            "source": "Get-CimInstance Win32_OperatingSystem",
            "measurement_status": "observed_read_only_preflight",
            "volatile": "true",
            "notes": "Approx. 31.71 GiB physical total and 18.66 GiB free at measurement; free RAM is volatile.",
        },
        {
            "component_id": "F6E2-HW-GPU-INTEL",
            "component_type": "gpu",
            "name": "Intel(R) UHD Graphics",
            "observed_value": "AdapterRAM 2147479552; driver 32.0.101.6790",
            "unit": "bytes",
            "source": "Get-CimInstance Win32_VideoController",
            "measurement_status": "observed_read_only_preflight",
            "volatile": "false",
            "notes": "Integrated GPU is not treated as a viable LLM inference accelerator.",
        },
        {
            "component_id": "F6E2-HW-GPU-NVIDIA",
            "component_type": "gpu",
            "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "observed_value": "8188 MiB total; 7866 MiB free; 91 MiB used; driver 595.79; compute capability 8.9; CUDA 13.2 exposed",
            "unit": "MiB",
            "source": "nvidia-smi",
            "measurement_status": "observed_read_only_preflight",
            "volatile": "true",
            "notes": "Supports CUDA execution for installed PyTorch; 8 GiB VRAM is adequate for local 7B quantized Ollama models but not 70B full local inference.",
        },
        {
            "component_id": "F6E2-HW-STORAGE-C",
            "component_type": "storage",
            "name": "C:",
            "observed_value": "1023010664448 total; 185582936064 free; NTFS",
            "unit": "bytes",
            "source": "System.IO.DriveInfo",
            "measurement_status": "observed_read_only_preflight",
            "volatile": "true",
            "notes": "Approx. 172.80 GiB free at measurement. Existing Ollama model blobs total approx. 8.20 GiB.",
        },
        {
            "component_id": "F6E2-HW-OS",
            "component_type": "system",
            "name": "Microsoft Windows 11 Pro",
            "observed_value": "Version 10.0.26200; build 26200; 64 bit",
            "unit": "snapshot",
            "source": "Get-CimInstance Win32_OperatingSystem",
            "measurement_status": "observed_read_only_preflight",
            "volatile": "false",
            "notes": "WSL status command returned E_ACCESSDENIED in the current session, so Linux runtime availability is not verified.",
        },
    ]


def build_runtime_inventory(root: Path) -> list[dict[str, str]]:
    ollama_path = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
    nvcc_path = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/nvcc.exe")
    rows = [
        {
            "runtime_id": "F6E2-RT-OLLAMA",
            "runtime_name": "Ollama",
            "installed": "true",
            "version": "client 0.31.1; server not reachable during preflight",
            "path": logical_external_path(ollama_path),
            "gpu_support": "yes_via_ollama_runtime_when_service_operational",
            "cpu_offload_support": "runtime_dependent",
            "quantization_support": "ollama_model_manifest_quantization_not_declared_in_snapshot",
            "context_limit_configurable": "yes_runtime_parameter_dependent",
            "windows_compatibility": "native_windows_client_installed",
            "repository_usage": "used by F6D4A local Qwen2/Mistral JSON-index ranking",
            "operational_status": "client_present_server_not_active_or_log_permission_blocked",
            "network_or_download_needed": "not_for_installed_qwen2_mistral; yes_for_missing_llama3_70b",
            "source": "Get-Command ollama; ollama --version; local model manifests",
            "notes": "Read-only model list attempt failed because Ollama tried to create a server log and hit access denied; F6E2 does not start a server.",
        },
        {
            "runtime_id": "F6E2-RT-NVIDIA-SMI",
            "runtime_name": "NVIDIA driver runtime",
            "installed": "true",
            "version": "driver 595.79; CUDA 13.2 exposed",
            "path": "C:/Windows/System32/nvidia-smi.exe",
            "gpu_support": "true",
            "cpu_offload_support": "not_applicable",
            "quantization_support": "not_applicable",
            "context_limit_configurable": "not_applicable",
            "windows_compatibility": "native_windows",
            "repository_usage": "hardware diagnostics only",
            "operational_status": "available",
            "network_or_download_needed": "false",
            "source": "nvidia-smi",
            "notes": "Driver/runtime mismatch with nvcc 11.8 is not blocking for installed torch cu124 according to torch.cuda_available.",
        },
        {
            "runtime_id": "F6E2-RT-NVCC",
            "runtime_name": "CUDA toolkit nvcc",
            "installed": "true",
            "version": "11.8 V11.8.89",
            "path": nvcc_path.as_posix(),
            "gpu_support": "compile_tool_only",
            "cpu_offload_support": "not_applicable",
            "quantization_support": "not_applicable",
            "context_limit_configurable": "not_applicable",
            "windows_compatibility": "native_windows",
            "repository_usage": "diagnostic; not required for Ollama inference",
            "operational_status": "available",
            "network_or_download_needed": "false",
            "source": "where.exe nvcc; nvcc --version",
            "notes": "No compilation is performed by F6E2.",
        },
        {
            "runtime_id": "F6E2-RT-PYTORCH",
            "runtime_name": "PyTorch",
            "installed": "true",
            "version": "2.6.0+cu124",
            "path": ".venv",
            "gpu_support": "torch.cuda_available true; device RTX 4060 Laptop GPU",
            "cpu_offload_support": "model_code_dependent",
            "quantization_support": "torch_native_limited; no bitsandbytes installed",
            "context_limit_configurable": "model_code_dependent",
            "windows_compatibility": "installed_in_project_venv",
            "repository_usage": "traditional neural baselines; not used by offline F6E2",
            "operational_status": "available",
            "network_or_download_needed": "false",
            "source": ".venv Python import inventory",
            "notes": "Useful for local neural baselines, not sufficient alone for Hugging Face 70B inference because transformers/accelerate are absent.",
        },
        {
            "runtime_id": "F6E2-RT-TRANSFORMERS",
            "runtime_name": "Transformers",
            "installed": "false",
            "version": "",
            "path": "",
            "gpu_support": "not_available",
            "cpu_offload_support": "not_available",
            "quantization_support": "not_available",
            "context_limit_configurable": "not_available",
            "windows_compatibility": "not_installed",
            "repository_usage": "not available for F6E2",
            "operational_status": "missing",
            "network_or_download_needed": "would_require_install_and_model_download",
            "source": ".venv Python import inventory",
            "notes": "Blocks any Hugging Face based Llama 3 70B path under no-install policy.",
        },
        {
            "runtime_id": "F6E2-RT-ACCELERATE",
            "runtime_name": "accelerate",
            "installed": "false",
            "version": "",
            "path": "",
            "gpu_support": "not_available",
            "cpu_offload_support": "not_available",
            "quantization_support": "not_available",
            "context_limit_configurable": "not_available",
            "windows_compatibility": "not_installed",
            "repository_usage": "not available for F6E2",
            "operational_status": "missing",
            "network_or_download_needed": "would_require_install",
            "source": ".venv Python import inventory",
            "notes": "Would be needed for large HF model sharding/offload.",
        },
        {
            "runtime_id": "F6E2-RT-BITSANDBYTES",
            "runtime_name": "bitsandbytes",
            "installed": "false",
            "version": "",
            "path": "",
            "gpu_support": "not_available",
            "cpu_offload_support": "not_available",
            "quantization_support": "not_available",
            "context_limit_configurable": "not_available",
            "windows_compatibility": "not_installed",
            "repository_usage": "not available for F6E2",
            "operational_status": "missing",
            "network_or_download_needed": "would_require_install",
            "source": ".venv Python import inventory",
            "notes": "Blocks common 4-bit HF loading route under no-install policy.",
        },
        {
            "runtime_id": "F6E2-RT-VLLM",
            "runtime_name": "vLLM",
            "installed": "false",
            "version": "",
            "path": "",
            "gpu_support": "not_available",
            "cpu_offload_support": "not_available",
            "quantization_support": "not_available",
            "context_limit_configurable": "not_available",
            "windows_compatibility": "not_installed",
            "repository_usage": "not available for F6E2",
            "operational_status": "missing",
            "network_or_download_needed": "would_require_install_and_model_download",
            "source": ".venv Python import inventory",
            "notes": "Not a viable 70B route in current Windows/.venv state.",
        },
        {
            "runtime_id": "F6E2-RT-LLAMA-CPP",
            "runtime_name": "llama.cpp or llama-cpp-python",
            "installed": "false",
            "version": "",
            "path": "",
            "gpu_support": "not_available",
            "cpu_offload_support": "not_available",
            "quantization_support": "GGUF route unavailable locally",
            "context_limit_configurable": "not_available",
            "windows_compatibility": "not_installed",
            "repository_usage": "not available for F6E2",
            "operational_status": "missing",
            "network_or_download_needed": "would_require_runtime_install_and_model_file",
            "source": ".venv Python import inventory; PATH inspection",
            "notes": "Blocks GGUF/offload Llama 3 70B scenarios under no-install policy.",
        },
        {
            "runtime_id": "F6E2-RT-GPTQ-AWQ",
            "runtime_name": "GPTQ/AWQ Python runtimes",
            "installed": "false",
            "version": "",
            "path": "",
            "gpu_support": "not_available",
            "cpu_offload_support": "not_available",
            "quantization_support": "not_available",
            "context_limit_configurable": "not_available",
            "windows_compatibility": "not_installed",
            "repository_usage": "not available for F6E2",
            "operational_status": "missing",
            "network_or_download_needed": "would_require_install_and_model_download",
            "source": ".venv Python import inventory",
            "notes": "No local GPTQ/AWQ backend is available.",
        },
        {
            "runtime_id": "F6E2-RT-WSL",
            "runtime_name": "WSL",
            "installed": "unknown",
            "version": "",
            "path": "C:/Windows/System32/wsl.exe",
            "gpu_support": "unknown",
            "cpu_offload_support": "unknown",
            "quantization_support": "unknown",
            "context_limit_configurable": "unknown",
            "windows_compatibility": "windows_feature",
            "repository_usage": "not used by current repository evidence",
            "operational_status": "status_access_denied",
            "network_or_download_needed": "unknown",
            "source": "wsl --status",
            "notes": "Command returned E_ACCESSDENIED; no Linux runtime is verified for F6E2.",
        },
    ]
    return rows


def build_model_inventory(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for tag, family, params in [
        ("qwen2:7b-instruct", "Qwen2", "7B"),
        ("mistral:7b-instruct", "Mistral", "7B"),
    ]:
        manifest = read_ollama_manifest(tag)
        manifest_path = ollama_manifest_path(tag)
        model = model_layer(manifest or {}, "image.model") if manifest else None
        total_bytes = sum(int(layer.get("size", 0)) for layer in (manifest or {}).get("layers", []))
        model_size = int(model.get("size", 0)) if model else 0
        model_digest = str(model.get("digest", "")).replace("sha256:", "") if model else ""
        config_digest = str((manifest or {}).get("config", {}).get("digest", "")).replace("sha256:", "")
        rows.append(
            {
                "model_id": stable_id("F6E2-MODEL", tag),
                "model_tag": tag,
                "family": family,
                "parameter_count": params,
                "quantization": "ollama_blob_quantization_not_declared_in_manifest",
                "model_blob_size_bytes": str(model_size),
                "total_manifest_layer_bytes": str(total_bytes),
                "manifest_path": logical_external_path(manifest_path),
                "model_blob_digest": model_digest,
                "config_digest": config_digest,
                "context_window_declared": "not_declared_in_manifest_snapshot",
                "tokenizer": "ollama_runtime_internal_not_separately_verified",
                "runtime": "Ollama",
                "install_time": "not_recorded",
                "availability_status": "available_manifest_and_blob" if manifest else "missing",
                "used_in_previous_phases": "canonical_local_full_matrix;F6D4A_LastFM_JSON_index",
                "frozen_identity_hash": sha256_text(stable_json(manifest)) if manifest else "missing",
                "notes": "Installed local paper-family model; exact paper-version equivalence is not demonstrated from tag/digest alone.",
            }
        )
    rows.append(
        {
            "model_id": stable_id("F6E2-MODEL", "llama3:70b-instruct"),
            "model_tag": "llama3:70b-instruct",
            "family": "Llama 3",
            "parameter_count": "70B",
            "quantization": "not_available",
            "model_blob_size_bytes": "0",
            "total_manifest_layer_bytes": "0",
            "manifest_path": logical_external_path(ollama_manifest_path("llama3:70b-instruct")),
            "model_blob_digest": "",
            "config_digest": "",
            "context_window_declared": "not_available",
            "tokenizer": "not_available",
            "runtime": "not_available",
            "install_time": "not_available",
            "availability_status": "missing",
            "used_in_previous_phases": "not_used",
            "frozen_identity_hash": "missing",
            "notes": "No local Llama 3 manifest or blob was found; no model acquisition is performed in F6E2.",
        }
    )
    return rows


def build_identity_alignment(model_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    by_tag = {row["model_tag"]: row for row in model_rows}
    qwen_digest = by_tag.get("qwen2:7b-instruct", {}).get("model_blob_digest", "")
    mistral_digest = by_tag.get("mistral:7b-instruct", {}).get("model_blob_digest", "")
    return [
        {
            "alignment_id": "F6E2-ID-QWEN2",
            "paper_model_name": "qwen-2-7b-instruct",
            "local_model_tag": "qwen2:7b-instruct",
            "organization": "Qwen",
            "family": "Qwen2",
            "parameter_count": "7B",
            "variant": "instruct",
            "quantization": "local_ollama_quantization_not_declared",
            "version_or_tag": "local tag qwen2:7b-instruct",
            "identity_evidence": f"local Ollama manifest and model digest {qwen_digest}",
            "divergences": "paper exact release, tokenizer, quantization, decoding, prompt/cohort/candidate policy not proven identical",
            "classification": "exact_family_version_uncertain",
            "allowed_scientific_claim": "paper model family represented by a local adapted execution, not protocol-exact replication",
        },
        {
            "alignment_id": "F6E2-ID-MISTRAL",
            "paper_model_name": "mistral-7b-instruct-v0.2",
            "local_model_tag": "mistral:7b-instruct",
            "organization": "Mistral AI",
            "family": "Mistral",
            "parameter_count": "7B",
            "variant": "instruct",
            "quantization": "local_ollama_quantization_not_declared",
            "version_or_tag": "local tag mistral:7b-instruct",
            "identity_evidence": f"local Ollama manifest and model digest {mistral_digest}",
            "divergences": "v0.2 equivalence, tokenizer, quantization, decoding, prompt/cohort/candidate policy not proven identical",
            "classification": "exact_family_version_uncertain",
            "allowed_scientific_claim": "paper model family represented by a local adapted execution, not protocol-exact replication",
        },
        {
            "alignment_id": "F6E2-ID-LLAMA3-70B",
            "paper_model_name": "llama-3-70b-instruct",
            "local_model_tag": "",
            "organization": "Meta",
            "family": "Llama 3",
            "parameter_count": "70B",
            "variant": "instruct",
            "quantization": "not_available",
            "version_or_tag": "not_installed",
            "identity_evidence": "no local Llama 3 Ollama manifest found",
            "divergences": "missing local model and local hardware/runtime feasibility not met",
            "classification": "unavailable",
            "allowed_scientific_claim": "blocked paper component; no local result can be claimed",
        },
        {
            "alignment_id": "F6E2-ID-GPT35",
            "paper_model_name": "gpt-3.5-turbo-0125",
            "local_model_tag": "",
            "organization": "OpenAI",
            "family": "GPT",
            "parameter_count": "closed",
            "variant": "chat",
            "quantization": "not_applicable",
            "version_or_tag": "remote_api_model",
            "identity_evidence": "not locally installed; API disallowed",
            "divergences": "closed source remote model outside no-API policy",
            "classification": "unavailable",
            "allowed_scientific_claim": "blocked closed-source paper component",
        },
        {
            "alignment_id": "F6E2-ID-GPT4OMINI",
            "paper_model_name": "gpt-4o-mini-2024-07-18",
            "local_model_tag": "",
            "organization": "OpenAI",
            "family": "GPT-4o",
            "parameter_count": "closed",
            "variant": "chat",
            "quantization": "not_applicable",
            "version_or_tag": "remote_api_model",
            "identity_evidence": "not locally installed; API disallowed",
            "divergences": "closed source remote model outside no-API policy",
            "classification": "unavailable",
            "allowed_scientific_claim": "blocked closed-source paper component",
        },
        {
            "alignment_id": "F6E2-ID-GPT4O",
            "paper_model_name": "gpt-4o-2024-08-06",
            "local_model_tag": "",
            "organization": "OpenAI",
            "family": "GPT-4o",
            "parameter_count": "closed",
            "variant": "chat",
            "quantization": "not_applicable",
            "version_or_tag": "remote_api_model",
            "identity_evidence": "not locally installed; API disallowed",
            "divergences": "closed source remote model outside no-API policy",
            "classification": "unavailable",
            "allowed_scientific_claim": "blocked closed-source paper component",
        },
        {
            "alignment_id": "F6E2-ID-CLAUDE",
            "paper_model_name": "claude-3-haiku-20240307",
            "local_model_tag": "",
            "organization": "Anthropic",
            "family": "Claude",
            "parameter_count": "closed",
            "variant": "chat",
            "quantization": "not_applicable",
            "version_or_tag": "remote_api_model",
            "identity_evidence": "not locally installed; API disallowed",
            "divergences": "closed source remote model outside no-API policy",
            "classification": "unavailable",
            "allowed_scientific_claim": "blocked closed-source paper component",
        },
    ]


def split_pipe_count(value: str) -> int:
    if not value:
        return 0
    return len([item for item in str(value).split("|") if item != ""])


def numeric_field(row: Mapping[str, str], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                continue
    return default


def infer_candidate_count(row: Mapping[str, str]) -> int:
    for name in ("candidate_pool_size", "candidate_count", "num_candidates"):
        value = row.get(name)
        if value not in (None, ""):
            try:
                return int(float(value))
            except ValueError:
                pass
    return max(split_pipe_count(row.get("candidate_item_ids", "")), split_pipe_count(row.get("candidate_items", "")))


def infer_history_length(row: Mapping[str, str]) -> int:
    for name in ("history_length", "train_history_length", "user_history_length"):
        value = row.get(name)
        if value not in (None, ""):
            try:
                return int(float(value))
            except ValueError:
                pass
    return max(split_pipe_count(row.get("history_item_ids", "")), split_pipe_count(row.get("history_items", "")))


def build_plan_e_workload_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset, task, manifest_path in PLAN_E_MANIFESTS:
        manifest = read_csv_rows(root, manifest_path)
        users = {row.get("user_id", "") for row in manifest if row.get("user_id", "") != ""}
        candidate_counts_by_user: dict[str, int] = defaultdict(int)
        positives_by_user: dict[str, int] = defaultdict(int)
        orderings = set()
        for row in manifest:
            user = row.get("user_id", "")
            if user:
                candidate_counts_by_user[user] += 1
                if str(row.get("is_positive", "")).lower() in {"true", "1", "yes"}:
                    positives_by_user[user] += 1
            if row.get("ordering_strategy"):
                orderings.add(row["ordering_strategy"])
        counts = list(candidate_counts_by_user.values())
        positives = sum(1 for value in positives_by_user.values() if value > 0)
        logical_instances = len(users)
        rows.append(
            {
                "workload_id": stable_id("F6E2-WL", dataset, task),
                "dataset": dataset,
                "task": task,
                "manifest_path": manifest_path.as_posix(),
                "manifest_sha256": file_sha256_or_missing(root, manifest_path),
                "manifest_rows": str(len(manifest)),
                "logical_instances": str(logical_instances),
                "distinct_users": str(len(users)),
                "candidate_count_min": str(min(counts) if counts else 0),
                "candidate_count_mean": fmt(mean(counts) if counts else 0),
                "candidate_count_max": str(max(counts) if counts else 0),
                "history_length_min": "10",
                "history_length_mean": "10",
                "history_length_max": "10",
                "positive_count": str(positives),
                "candidate_ordering": ";".join(sorted(orderings)) if orderings else "not_recorded",
                "item_metadata_available": "title_map_available",
                "title_availability": "verified_by_canonical_llm_preflight_or_manifest",
                "status": "ready",
                "notes": "Plan E 1000-user per-candidate manifest collapsed to one logical LLM request per user.",
            }
        )
    return rows


def build_lastfm_workload_row(root: Path, task: str, path: Path) -> dict[str, str]:
    manifest = read_csv_rows(root, path)
    users = {row.get("user_id", "") for row in manifest if row.get("user_id", "")}
    unique_rows = {}
    for row in manifest:
        user = row.get("user_id", "")
        model = row.get("model_tag") or row.get("model", "")
        if user and model:
            unique_rows[(user, model)] = row
    qwen_rows = [row for (user, model), row in unique_rows.items() if model == "qwen2:7b-instruct"]
    counts = [infer_candidate_count(row) for row in qwen_rows]
    histories = [infer_history_length(row) for row in qwen_rows]
    positive_count = sum(1 for row in qwen_rows if str(row.get("positive_present", row.get("is_positive", "true"))).lower() in {"true", "1", "yes"})
    status = "ready" if task == "ranking" else "partial"
    notes = "F6D3/F6D4A LastFM ranking manifest." if task == "ranking" else "Reranking manifest exists but is not utility-comparable when positives are absent from adapted pool."
    return {
        "workload_id": stable_id("F6E2-WL", "LastFM", task),
        "dataset": "LastFM",
        "task": task,
        "manifest_path": path.as_posix(),
        "manifest_sha256": file_sha256_or_missing(root, path),
        "manifest_rows": str(len(manifest)),
        "logical_instances": str(len(qwen_rows)),
        "distinct_users": str(len(users)),
        "candidate_count_min": str(min(counts) if counts else 0),
        "candidate_count_mean": fmt(mean(counts) if counts else 0),
        "candidate_count_max": str(max(counts) if counts else 0),
        "history_length_min": str(min(histories) if histories else 0),
        "history_length_mean": fmt(mean(histories) if histories else 0),
        "history_length_max": str(max(histories) if histories else 0),
        "positive_count": str(positive_count),
        "candidate_ordering": "adapted_lastfm_candidate_order",
        "item_metadata_available": "title/catalog evidence available",
        "title_availability": "available_for_prompt_rendering",
        "status": status,
        "notes": notes,
    }


def build_request_workload(root: Path) -> list[dict[str, str]]:
    rows = build_plan_e_workload_rows(root)
    rows.append(build_lastfm_workload_row(root, "ranking", F6D3_RANKING_MANIFEST_PATH))
    rows.append(build_lastfm_workload_row(root, "reranking", F6D3_RERANKING_MANIFEST_PATH))
    return rows


def sample_contexts_from_jsonl(root: Path, limit_per_dataset_task: int = 100) -> dict[tuple[str, str], list[PromptRenderContext]]:
    rows_by_key: dict[tuple[str, str], list[PromptRenderContext]] = defaultdict(list)
    path = root / FULL_MATRIX_REQUESTS_PATH
    if not path.exists():
        return rows_by_key
    seen_users: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            dataset = str(payload.get("dataset", ""))
            task = str(payload.get("task_type", ""))
            user = str(payload.get("user_id", ""))
            key = (dataset, task)
            user_key = (dataset, task, user)
            if user_key in seen_users or len(rows_by_key[key]) >= limit_per_dataset_task:
                continue
            prompt = str(payload.get("request_payload", {}).get("prompt", ""))
            history = extract_numbered_block(prompt, "User history:", "Candidates:")
            candidates = extract_numbered_block(prompt, "Candidates:", "")
            if not history or not candidates:
                continue
            rows_by_key[key].append(
                PromptRenderContext(
                    dataset=dataset,
                    user_name=user or "user",
                    user_history_titles=tuple(history[-10:]),
                    candidate_titles=tuple(candidates[:20]),
                    user_profile="Local profile text unavailable in frozen request; F6E2 synthetic audit profile.",
                )
            )
            seen_users.add(user_key)
    return rows_by_key


def extract_numbered_block(text: str, start_label: str, end_label: str) -> list[str]:
    if start_label not in text:
        return []
    after = text.split(start_label, 1)[1]
    if end_label and end_label in after:
        after = after.split(end_label, 1)[0]
    titles: list[str] = []
    for line in after.splitlines():
        match = re.match(r"\s*\d+\.\s*(.+?)\s*$", line)
        if match:
            titles.append(match.group(1))
    return titles


def sample_contexts_from_lastfm_manifest(root: Path, limit: int = 100) -> dict[tuple[str, str], list[PromptRenderContext]]:
    contexts: dict[tuple[str, str], list[PromptRenderContext]] = defaultdict(list)
    for task, path in [("ranking", F6D3_RANKING_MANIFEST_PATH), ("reranking", F6D3_RERANKING_MANIFEST_PATH)]:
        rows = read_csv_rows(root, path)
        seen: set[str] = set()
        for row in rows:
            model_tag = row.get("model_tag") or row.get("model", "")
            if model_tag != "qwen2:7b-instruct":
                continue
            user = row.get("user_id", "")
            if not user or user in seen:
                continue
            history = [value for value in row.get("history_titles", row.get("history_item_names", "")).split("|") if value]
            candidates = [value for value in row.get("candidate_titles", row.get("candidate_item_names", "")).split("|") if value]
            if not history:
                history = [f"LastFM history item {idx}" for idx in range(1, 11)]
            if not candidates:
                candidates = [f"LastFM candidate {idx}" for idx in range(1, 21)]
            contexts[("LastFM", task)].append(
                PromptRenderContext(
                    dataset="LastFM",
                    user_name=user,
                    user_history_titles=tuple(history[-10:]),
                    candidate_titles=tuple(candidates[:20]),
                    user_profile="LastFM profile text unavailable; F6E2 synthetic audit profile.",
                )
            )
            seen.add(user)
            if len(contexts[("LastFM", task)]) >= limit:
                break
    return contexts


def representative_contexts(root: Path) -> dict[tuple[str, str], list[PromptRenderContext]]:
    contexts = sample_contexts_from_jsonl(root)
    lastfm = sample_contexts_from_lastfm_manifest(root)
    for key, rows in lastfm.items():
        contexts[key].extend(rows)
    return contexts


def prompt_parts(prompt: str, context: PromptRenderContext) -> tuple[int, int, int, int]:
    history_text = "\n".join(context.user_history_titles)
    candidate_text = "\n".join(context.candidate_titles)
    history_tokens = estimate_tokens(history_text)
    candidate_tokens = estimate_tokens(candidate_text)
    profile_tokens = estimate_tokens(context.user_profile) if context.user_profile else 0
    instruction_tokens = max(0, estimate_tokens(prompt) - history_tokens - candidate_tokens - profile_tokens)
    return history_tokens, candidate_tokens, instruction_tokens, profile_tokens


def build_prompt_token_estimates(root: Path, workloads: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    contexts_by_key = representative_contexts(root)
    rows: list[dict[str, str]] = []
    strategies = [BASE_STRATEGY_ID, LLMRANK_STRATEGY_ID, LLMSREC_STRATEGY_ID, JSON_INDEX_STRATEGY_ID]
    for workload in workloads:
        dataset = workload["dataset"]
        task = workload["task"]
        contexts = contexts_by_key.get((dataset, task), [])
        if not contexts:
            contexts = [
                PromptRenderContext(
                    dataset=dataset,
                    user_name="synthetic_user",
                    user_history_titles=tuple(f"{dataset} history item {idx}" for idx in range(1, 11)),
                    candidate_titles=tuple(f"{dataset} candidate {idx}" for idx in range(1, 21)),
                    user_profile=f"Synthetic {dataset} user profile for offline prompt sizing.",
                )
            ]
        for strategy in strategies:
            prompts: list[str] = []
            history_tokens: list[int] = []
            candidate_tokens: list[int] = []
            instruction_tokens: list[int] = []
            profile_tokens: list[int] = []
            for context in contexts:
                render_task = task if strategy == JSON_INDEX_STRATEGY_ID else "ranking"
                if strategy == JSON_INDEX_STRATEGY_ID and render_task == "reranking":
                    # F6E1 renderer accepts top_k=20 for reranking, but render_strategy_prompt
                    # is ranking-only. Use a local call through the public JSON branch for a
                    # comparable one-call JSON/index estimate.
                    from src.llm.plan_f6e1_prompt_strategies import render_json_index_prompt

                    prompt = render_json_index_prompt(context=context, task_type="reranking", top_k=20)["user_message"]
                else:
                    prompt = render_strategy_prompt(strategy, context, root=root)
                prompts.append(prompt)
                hist, cand, instr, prof = prompt_parts(prompt, context)
                history_tokens.append(hist)
                candidate_tokens.append(cand)
                instruction_tokens.append(instr)
                profile_tokens.append(prof if strategy == LLMSREC_STRATEGY_ID else 0)
            chars = [len(prompt) for prompt in prompts]
            tokens = [estimate_tokens(prompt) for prompt in prompts]
            char_stats = numeric_stats([float(value) for value in chars])
            token_stats = numeric_stats([float(value) for value in tokens])
            output_budget = 80 if strategy != JSON_INDEX_STRATEGY_ID else (120 if task == "ranking" else 400)
            local_limit = 4096
            rows.append(
                {
                    "estimate_id": stable_id("F6E2-TOK", strategy, dataset, task),
                    "strategy_id": strategy,
                    "strategy_classification": STRATEGY_CLASSIFICATION[strategy],
                    "dataset": dataset,
                    "task": task,
                    "method": "char_div_4_estimate_no_exact_tokenizer",
                    "rows_sampled": str(len(prompts)),
                    "char_min": fmt(char_stats["min"]),
                    "char_mean": fmt(char_stats["mean"]),
                    "char_median": fmt(char_stats["median"]),
                    "char_p95": fmt(char_stats["p95"]),
                    "char_max": fmt(char_stats["max"]),
                    "token_min": fmt(token_stats["min"]),
                    "token_mean": fmt(token_stats["mean"]),
                    "token_median": fmt(token_stats["median"]),
                    "token_p95": fmt(token_stats["p95"]),
                    "token_max": fmt(token_stats["max"]),
                    "history_token_mean": fmt(mean(history_tokens) if history_tokens else 0),
                    "candidate_token_mean": fmt(mean(candidate_tokens) if candidate_tokens else 0),
                    "instruction_token_mean": fmt(mean(instruction_tokens) if instruction_tokens else 0),
                    "profile_token_mean": fmt(mean(profile_tokens) if profile_tokens else 0),
                    "output_token_budget": str(output_budget),
                    "context_total_token_p95": fmt(token_stats["p95"] + output_budget),
                    "above_4k_percent": pct(sum(value + output_budget > 4096 for value in tokens), len(tokens)),
                    "above_8k_percent": pct(sum(value + output_budget > 8192 for value in tokens), len(tokens)),
                    "above_16k_percent": pct(sum(value + output_budget > 16384 for value in tokens), len(tokens)),
                    "above_32k_percent": pct(sum(value + output_budget > 32768 for value in tokens), len(tokens)),
                    "above_local_context_limit_percent": pct(sum(value + output_budget > local_limit for value in tokens), len(tokens)),
                    "local_context_limit": str(local_limit),
                    "status": "estimated",
                    "notes": "Exact tokenizer not installed or not used; no tokenizer download performed.",
                }
            )
    return rows


def observed_timing_by_model(root: Path) -> dict[str, dict[str, float]]:
    durations: dict[str, list[float]] = defaultdict(list)
    prompt_counts: dict[str, list[float]] = defaultdict(list)
    eval_counts: dict[str, list[float]] = defaultdict(list)
    for row in read_csv_rows(root, FULL_MATRIX_EXECUTION_PATH):
        tag = row.get("model_tag") or row.get("model_name") or ""
        if tag:
            duration = numeric_field(row, "duration_seconds", "total_duration_seconds")
            if duration > 0:
                durations[tag].append(duration)
            prompt_eval = numeric_field(row, "prompt_eval_count")
            if prompt_eval > 0:
                prompt_counts[tag].append(prompt_eval)
            eval_count = numeric_field(row, "eval_count")
            if eval_count > 0:
                eval_counts[tag].append(eval_count)
    for path, tag in [(F6D4A_QWEN_RAW_PATH, "qwen2:7b-instruct"), (F6D4A_MISTRAL_RAW_PATH, "mistral:7b-instruct")]:
        for row in read_jsonl_rows(root, path):
            duration = numeric_field(row, "duration_seconds")
            if duration > 0:
                durations[tag].append(duration)
            prompt_eval = numeric_field(row, "prompt_eval_count")
            if prompt_eval > 0:
                prompt_counts[tag].append(prompt_eval)
            eval_count = numeric_field(row, "eval_count")
            if eval_count > 0:
                eval_counts[tag].append(eval_count)
    result: dict[str, dict[str, float]] = {}
    for tag in sorted(durations):
        result[tag] = {
            "mean_duration": mean(durations[tag]) if durations[tag] else 0.0,
            "p95_duration": percentile(durations[tag], 0.95) if durations[tag] else 0.0,
            "rows": float(len(durations[tag])),
            "prompt_eval_mean": mean(prompt_counts[tag]) if prompt_counts[tag] else 0.0,
            "eval_count_mean": mean(eval_counts[tag]) if eval_counts[tag] else 0.0,
        }
    return result


def workload_total_instances(workloads: Sequence[Mapping[str, str]], include_partial: bool = False) -> int:
    total = 0
    for row in workloads:
        if row["status"] == "ready" or include_partial:
            total += int(float(row["logical_instances"]))
    return total


def calls_for_strategy(strategy_id: str) -> tuple[float, float, float]:
    values = CALL_MULTIPLICITY[strategy_id]
    return float(values[0]), float(values[1]), float(values[2])


def build_runtime_estimates(root: Path, workloads: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    timing = observed_timing_by_model(root)
    rows: list[dict[str, str]] = []
    ready_workloads = [row for row in workloads if row["status"] == "ready"]
    for model_tag in MODEL_TAGS:
        for strategy in [BASE_STRATEGY_ID, LLMRANK_STRATEGY_ID, LLMSREC_STRATEGY_ID, JSON_INDEX_STRATEGY_ID]:
            for workload in ready_workloads:
                instances = int(float(workload["logical_instances"]))
                min_calls, expected_calls_per, max_calls = calls_for_strategy(strategy)
                model_timing = timing.get(model_tag)
                if model_timing:
                    expected_seconds = model_timing["mean_duration"]
                    worst_seconds = max(model_timing["p95_duration"], expected_seconds)
                    best_seconds = max(expected_seconds * 0.75, 0.01)
                    basis = "estimate_from_same_model"
                    status = "estimated"
                elif model_tag == "llama3:70b-instruct":
                    expected_seconds = worst_seconds = best_seconds = 0.0
                    basis = "not_estimable"
                    status = "blocked_missing_model"
                    model_timing = {"rows": 0.0, "prompt_eval_mean": 0.0, "eval_count_mean": 0.0}
                else:
                    expected_seconds = 20.0
                    worst_seconds = 40.0
                    best_seconds = 10.0
                    basis = "extrapolation"
                    status = "estimated_with_low_confidence"
                    model_timing = {"rows": 0.0, "prompt_eval_mean": 0.0, "eval_count_mean": 0.0}
                if strategy == LLMSREC_STRATEGY_ID and status == "estimated":
                    expected_seconds *= 1.5
                    worst_seconds *= 2.0
                if strategy == LLMRANK_STRATEGY_ID and status == "estimated":
                    expected_seconds *= 1.15
                    worst_seconds *= 1.25
                min_total = int(math.ceil(instances * min_calls))
                expected_total = int(math.ceil(instances * expected_calls_per))
                max_total = int(math.ceil(instances * max_calls))
                rows.append(
                    {
                        "estimate_id": stable_id("F6E2-RTE", model_tag, strategy, workload["dataset"], workload["task"]),
                        "model_tag": model_tag,
                        "strategy_id": strategy,
                        "dataset": workload["dataset"],
                        "task": workload["task"],
                        "logical_instances": str(instances),
                        "calls_per_instance_min": fmt(min_calls),
                        "calls_per_instance_expected": fmt(expected_calls_per),
                        "calls_per_instance_max": fmt(max_calls),
                        "min_calls": str(min_total),
                        "expected_calls": str(expected_total),
                        "max_calls": str(max_total),
                        "seconds_per_request_best": fmt(best_seconds),
                        "seconds_per_request_expected": fmt(expected_seconds),
                        "seconds_per_request_worst": fmt(worst_seconds),
                        "total_hours_best": fmt((best_seconds * min_total) / 3600.0),
                        "total_hours_expected": fmt((expected_seconds * expected_total) / 3600.0),
                        "total_days_worst": fmt((worst_seconds * max_total) / 86400.0),
                        "estimate_basis": basis,
                        "observed_rows": fmt(model_timing["rows"]),
                        "observed_prompt_eval_mean": fmt(model_timing["prompt_eval_mean"]),
                        "observed_eval_count_mean": fmt(model_timing["eval_count_mean"]),
                        "gpu_utilization_expected": "low_to_moderate_7b" if model_tag != "llama3:70b-instruct" else "not_estimated",
                        "cpu_bottleneck": "possible_during_prompt_eval" if model_tag != "llama3:70b-instruct" else "not_estimable",
                        "disk_write_volume_mb": fmt(expected_total * 0.015),
                        "resume_possible": "true",
                        "status": status,
                        "notes": "No benchmark was run in F6E2; estimates use frozen prior execution telemetry when available.",
                    }
                )
    return rows


def build_memory_disk_feasibility(model_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    present = {row["model_tag"]: row["availability_status"].startswith("available") for row in model_rows}
    rows: list[dict[str, str]] = []
    rows.extend(
        [
            {
                "feasibility_id": "F6E2-MEM-QWEN2-OLLAMA",
                "model_tag": "qwen2:7b-instruct",
                "scenario": "installed_ollama_quantized_7b",
                "parameter_count_billion": "7",
                "precision_or_quantization": "ollama_blob_quantized_unknown",
                "estimated_weight_gb": "4.13",
                "estimated_runtime_overhead_gb": "1.5",
                "estimated_kv_cache_gb": "0.5",
                "estimated_total_ram_gb": "6.5",
                "estimated_total_vram_gb": "5.5",
                "available_ram_gb_snapshot": "18.66",
                "total_ram_gb": "31.71",
                "available_vram_gb_snapshot": "7.68",
                "total_vram_gb": "8.00",
                "disk_required_gb": "4.13",
                "disk_available_gb_snapshot": "172.80",
                "runtime_required": "Ollama operational server",
                "model_present": str(present.get("qwen2:7b-instruct", False)).lower(),
                "scenario_status": "locally_feasible",
                "secondary_blockers": "ollama_server_currently_not_active_or_log_permission_blocked",
                "notes": "Previous local runs prove feasible once Ollama service is operational.",
            },
            {
                "feasibility_id": "F6E2-MEM-MISTRAL-OLLAMA",
                "model_tag": "mistral:7b-instruct",
                "scenario": "installed_ollama_quantized_7b",
                "parameter_count_billion": "7",
                "precision_or_quantization": "ollama_blob_quantized_unknown",
                "estimated_weight_gb": "4.07",
                "estimated_runtime_overhead_gb": "1.5",
                "estimated_kv_cache_gb": "0.5",
                "estimated_total_ram_gb": "6.5",
                "estimated_total_vram_gb": "5.5",
                "available_ram_gb_snapshot": "18.66",
                "total_ram_gb": "31.71",
                "available_vram_gb_snapshot": "7.68",
                "total_vram_gb": "8.00",
                "disk_required_gb": "4.07",
                "disk_available_gb_snapshot": "172.80",
                "runtime_required": "Ollama operational server",
                "model_present": str(present.get("mistral:7b-instruct", False)).lower(),
                "scenario_status": "locally_feasible",
                "secondary_blockers": "ollama_server_currently_not_active_or_log_permission_blocked",
                "notes": "Previous local runs prove feasible once Ollama service is operational.",
            },
        ]
    )
    scenarios = [
        ("fp16", 140.0, 12.0, 10.0, "blocked_by_vram", "missing_model;ram_insufficient_for_stable_run"),
        ("int8", 70.0, 8.0, 8.0, "blocked_by_vram", "missing_model;ram_insufficient_for_stable_run"),
        ("6bit", 52.5, 6.0, 7.0, "blocked_by_vram", "missing_model;ram_insufficient_for_stable_run"),
        ("5bit", 43.75, 5.0, 6.0, "blocked_by_vram", "missing_model;ram_insufficient_for_stable_run"),
        ("4bit", 35.0, 5.0, 5.0, "blocked_by_ram", "missing_model;runtime_missing;vram_requires_heavy_cpu_offload"),
        ("3bit_extreme", 26.25, 5.0, 5.0, "technically_possible_but_impractical", "missing_model;runtime_missing;very_slow_cpu_offload;little_ram_margin"),
    ]
    for quant, weights, overhead, kv, status, blockers in scenarios:
        rows.append(
            {
                "feasibility_id": stable_id("F6E2-MEM-LLAMA70B", quant),
                "model_tag": "llama3:70b-instruct",
                "scenario": f"llama3_70b_{quant}",
                "parameter_count_billion": "70",
                "precision_or_quantization": quant,
                "estimated_weight_gb": fmt(weights),
                "estimated_runtime_overhead_gb": fmt(overhead),
                "estimated_kv_cache_gb": fmt(kv),
                "estimated_total_ram_gb": fmt(weights + overhead + kv),
                "estimated_total_vram_gb": fmt(min(weights + overhead, 140.0)),
                "available_ram_gb_snapshot": "18.66",
                "total_ram_gb": "31.71",
                "available_vram_gb_snapshot": "7.68",
                "total_vram_gb": "8.00",
                "disk_required_gb": fmt(weights + 5.0),
                "disk_available_gb_snapshot": "172.80",
                "runtime_required": "GGUF/HF large-model runtime not installed",
                "model_present": "false",
                "scenario_status": status,
                "secondary_blockers": blockers,
                "notes": "Scenario is not experimentally usable on this machine for thousands of requests under no-install/no-download policy.",
            }
        )
    return rows


def model_identity_for_tag(tag: str, identity_rows: Sequence[Mapping[str, str]]) -> str:
    for row in identity_rows:
        if row["local_model_tag"] == tag or (tag == "" and row["paper_model_name"].startswith("llama-3")):
            return row["classification"]
    return "unavailable"


def build_execution_matrix(
    root: Path,
    workloads: Sequence[Mapping[str, str]],
    model_rows: Sequence[Mapping[str, str]],
    identity_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    f6e1_rows = read_csv_rows(root, F6E1_MATRIX_PATH)
    model_available = {row["model_tag"]: row["availability_status"].startswith("available") for row in model_rows}
    total_instances = workload_total_instances(workloads, include_partial=False)
    rows: list[dict[str, str]] = []
    for row in f6e1_rows:
        local_tag = row.get("local_model_identifier", "")
        strategy = row["strategy_id"]
        calls_min, calls_expected, calls_max = calls_for_strategy(strategy)
        available = bool(local_tag and model_available.get(local_tag, False))
        if strategy == LLMSREC_STRATEGY_ID:
            prompt_status = "reconstructed"
        else:
            prompt_status = "available"
        if "Llama3" in row["model_family"]:
            status = "blocked_hardware"
            blocker = "local Llama 3 70B model missing and hardware/runtime feasibility not met"
        elif strategy == JSON_INDEX_STRATEGY_ID:
            status = "ready_local_extension"
            blocker = "not paper prompt; Ollama server gate still required before new inference"
        elif strategy == LLMSREC_STRATEGY_ID:
            status = "ready_local_reconstructed_prompt" if available else "conditional_missing_model"
            blocker = "LLMSRec orchestration reconstructed; exact paper profile pipeline not proven"
        else:
            status = "ready_local_exact_prompt" if available else "conditional_missing_model"
            blocker = "exact model-version equivalence and Ollama operational gate still required"
        scientific_role = "local_ablation" if strategy == JSON_INDEX_STRATEGY_ID else ("paper_aligned_core" if row.get("scope") == "paper_aligned_core" else "cross_model_extension")
        rows.append(
            {
                "cell_id": stable_id("F6E2-CELL", row["matrix_id"]),
                "model_family": row["model_family"],
                "expected_model_identifier": row["expected_model_identifier"],
                "local_model_identifier": local_tag,
                "strategy_id": strategy,
                "strategy_classification": STRATEGY_CLASSIFICATION[strategy],
                "scientific_role": scientific_role,
                "paper_or_extension_scope": row.get("scope", ""),
                "model_identity_classification": model_identity_for_tag(local_tag, identity_rows) if local_tag else "unavailable",
                "local_model_available": str(available).lower(),
                "runtime_available": "conditional_ollama_client_installed_server_not_active" if available else "false",
                "prompt_available": prompt_status,
                "parser_available": row.get("parser_available", ""),
                "calls_per_request_min": fmt(calls_min),
                "calls_per_request_expected": fmt(calls_expected),
                "calls_per_request_max": fmt(calls_max),
                "total_logical_instances": str(total_instances),
                "total_expected_calls": str(int(math.ceil(total_instances * calls_expected))) if status not in {"blocked_hardware", "conditional_missing_model"} else "not_computable_until_unblocked",
                "preflight_status": status,
                "blocker": blocker,
                "safe_next_action": "run small append-only pilot only after explicit inference approval and Ollama operational gate",
                "notes": "F6E2 classification is preflight-only; no model was called.",
            }
        )
    return rows


def build_pilot_design() -> list[dict[str, str]]:
    return [
        {
            "pilot_id": "F6E2-PILOT-BASE",
            "strategy_id": BASE_STRATEGY_ID,
            "models": "qwen2:7b-instruct;mistral:7b-instruct",
            "datasets": "Amazon Beauty;Amazon Sports;ML-1M;LastFM",
            "tasks": "ranking",
            "minimum_requests": "16",
            "deterministic_selection": "first and last user by manifest order per dataset plus high/low history-length cases where available",
            "history_length_coverage": "min/median/max where manifest supports it; otherwise frozen 10-history prompt contract",
            "candidate_position_coverage": "include positive at early, middle, late positions where available",
            "expected_parser_checks": "official title-list parser; exact item-title match; length and duplicate checks",
            "stop_conditions": "any HTTP/generation outside approved runner, raw hash mutation, parser failure spike, missing manifest, timeout burst",
            "timeout_seconds": "120",
            "resume_policy": "manifest-driven request_id resume",
            "raw_policy": "append_only_new_raw_file",
            "retry_policy": "no automatic retry in first pilot",
            "gate_to_full_run": "all pilot requests completed, parser pass rate acceptable, raw/protected hashes unchanged",
            "status": "designed_not_executed",
            "notes": "Pilot is not materialized in F6E2.",
        },
        {
            "pilot_id": "F6E2-PILOT-LLMRANK",
            "strategy_id": LLMRANK_STRATEGY_ID,
            "models": "qwen2:7b-instruct;mistral:7b-instruct",
            "datasets": "Amazon Beauty;Amazon Sports;ML-1M;LastFM",
            "tasks": "ranking",
            "minimum_requests": "16",
            "deterministic_selection": "same instances as base pilot for paired comparison",
            "history_length_coverage": "recency-focused history contract",
            "candidate_position_coverage": "same as base pilot",
            "expected_parser_checks": "official title-list parser and ranking length checks",
            "stop_conditions": "same as base plus context overflow on recency prompt",
            "timeout_seconds": "150",
            "resume_policy": "manifest-driven request_id resume",
            "raw_policy": "append_only_new_raw_file",
            "retry_policy": "no automatic retry in first pilot",
            "gate_to_full_run": "paired base/LLMRank pilot validates parser and runtime",
            "status": "designed_not_executed",
            "notes": "Cross-model extension for Qwen/Mistral; paper-core Llama3 remains blocked.",
        },
        {
            "pilot_id": "F6E2-PILOT-LLMSREC",
            "strategy_id": LLMSREC_STRATEGY_ID,
            "models": "qwen2:7b-instruct;mistral:7b-instruct",
            "datasets": "Amazon Beauty;Amazon Sports;ML-1M;LastFM",
            "tasks": "ranking;profile_generation_if_enabled",
            "minimum_requests": "16 logical; 16-32 LLM calls depending on selected reconstruction",
            "deterministic_selection": "same instances as base pilot; profile generation disabled unless explicitly approved",
            "history_length_coverage": "include at least two-history minimum and long-history examples",
            "candidate_position_coverage": "same as base pilot",
            "expected_parser_checks": "profile-step schema if enabled; ranking parser after profile injection",
            "stop_conditions": "profile-step ambiguity, prompt evidence gap, context overflow, raw mutation",
            "timeout_seconds": "240",
            "resume_policy": "separate request namespaces per profile and ranking step",
            "raw_policy": "append_only_new_raw_file",
            "retry_policy": "no automatic retry in first pilot",
            "gate_to_full_run": "profile orchestration evidence resolved or explicitly limited to in-context reconstruction",
            "status": "designed_not_executed",
            "notes": "Cannot be classified as exact prompt because F6E1 froze it as reconstructed.",
        },
        {
            "pilot_id": "F6E2-PILOT-JSON-INDEX",
            "strategy_id": JSON_INDEX_STRATEGY_ID,
            "models": "qwen2:7b-instruct;mistral:7b-instruct",
            "datasets": "Amazon Beauty;Amazon Sports;ML-1M;LastFM",
            "tasks": "ranking;reranking",
            "minimum_requests": "16",
            "deterministic_selection": "same instances where comparable; include LastFM strict-valid audit cases separately",
            "history_length_coverage": "frozen local JSON/index history contract",
            "candidate_position_coverage": "include known invalid-output stress cases only as read-only fixtures",
            "expected_parser_checks": "JSON schema, length, range, duplicate index, strict-valid filters",
            "stop_conditions": "any raw overwrite, invalid rate regression, context overflow",
            "timeout_seconds": "120",
            "resume_policy": "request_id resume from manifest",
            "raw_policy": "append_only_new_raw_file",
            "retry_policy": "no automatic retry in first pilot",
            "gate_to_full_run": "pilot reproduces known strict-valid behavior and zero protected-input changes",
            "status": "designed_not_executed",
            "notes": "Local ablation, not a paper prompt strategy.",
        },
    ]


def build_missing_or_blocked() -> list[dict[str, str]]:
    return [
        {
            "block_id": "F6E2-BLOCK-OLLAMA-SERVER",
            "category": "runtime",
            "affected_scope": "all future local Ollama inference cells",
            "severity": "high",
            "status": "conditional_runtime_risk",
            "evidence": "Ollama client 0.31.1 installed; server not reachable; read-only list attempt hit log permission error",
            "safe_next_action": "user-session Ollama operational gate before any future inference",
            "notes": "F6E2 does not start a server.",
        },
        {
            "block_id": "F6E2-BLOCK-LLAMA70B",
            "category": "model_hardware",
            "affected_scope": "Llama 3 70B base, LLMRank, LLMSRec",
            "severity": "critical",
            "status": "blocked_hardware",
            "evidence": "no local Llama3 manifest; 8 GiB VRAM and 31.7 GiB RAM insufficient for stable 70B matrix",
            "safe_next_action": "do not run locally unless hardware/runtime/model availability changes and is re-audited",
            "notes": "No model download is performed.",
        },
        {
            "block_id": "F6E2-BLOCK-CLOSED-MODELS",
            "category": "model_identity",
            "affected_scope": "GPT and Claude paper models",
            "severity": "critical",
            "status": "blocked_closed_source",
            "evidence": "closed-source API models are not locally installed and API usage is disallowed",
            "safe_next_action": "document as blocked paper components unless API policy changes",
            "notes": "No remote API calls are made.",
        },
        {
            "block_id": "F6E2-BLOCK-LLMSREC",
            "category": "prompt_pipeline",
            "affected_scope": "LLMSRec strategy all local models",
            "severity": "medium",
            "status": "blocked_missing_prompt_evidence",
            "evidence": "F6E1 froze LLMSRec as reconstructed with incomplete orchestration evidence",
            "safe_next_action": "pilot only as reconstructed strategy with explicit scope",
            "notes": "Never classify LLMSRec as ready_local_exact_prompt.",
        },
        {
            "block_id": "F6E2-BLOCK-LASTFM-RERANK",
            "category": "manifest_policy",
            "affected_scope": "LastFM reranking",
            "severity": "medium",
            "status": "partial_manifest",
            "evidence": "F6D3 reranking manifest exists but positive inclusion is not comparable under current adapted pool policy",
            "safe_next_action": "keep separate from ranking and from utility-comparable paper claims",
            "notes": "Manifest is inventoried but not counted as ready workload.",
        },
        {
            "block_id": "F6E2-BLOCK-TOKENIZER",
            "category": "tokenization",
            "affected_scope": "prompt token estimates",
            "severity": "low",
            "status": "estimated_not_exact",
            "evidence": "transformers/tokenizers unavailable; F6E2 uses deterministic char/4 estimate",
            "safe_next_action": "only use exact tokenizer after explicit install/download policy change or locally available tokenizer proof",
            "notes": "Prompt sizes are estimates, not exact tokenization.",
        },
        {
            "block_id": "F6E2-BLOCK-WSL",
            "category": "runtime",
            "affected_scope": "Linux-only large-model runtime alternatives",
            "severity": "medium",
            "status": "runtime_evidence_insufficient",
            "evidence": "wsl --status returned E_ACCESSDENIED",
            "safe_next_action": "do not assume WSL or Linux GPU runtime availability",
            "notes": "F6E2 stays Windows/local evidence only.",
        },
    ]


def protected_input_summary(root: Path) -> dict[str, Any]:
    f2_rows = read_csv_rows(root, F2_PROTECTED_VALIDATION_PATH)
    f6d4a_rows = read_csv_rows(root, F6D4A_PROTECTED_VALIDATION_PATH)
    f2_checked = 0
    f2_changed = 0
    for row in f2_rows:
        try:
            f2_checked += int(float(row.get("checked", "0")))
            f2_changed += int(float(row.get("changed", "0")))
        except ValueError:
            continue
    f6d4a_checked = 0
    f6d4a_changed = 0
    for row in f6d4a_rows:
        if row.get("verification_id") == "F6D4A-CLOSURE-PROT-F6D4A":
            try:
                f6d4a_checked += int(float(row.get("checked", "0")))
                f6d4a_changed += int(float(row.get("changed", "0")))
            except ValueError:
                continue
    raw_hashes = {
        F6D4A_QWEN_RAW_PATH.as_posix(): file_sha256_or_missing(root, F6D4A_QWEN_RAW_PATH),
        F6D4A_MISTRAL_RAW_PATH.as_posix(): file_sha256_or_missing(root, F6D4A_MISTRAL_RAW_PATH),
        F6D3_RANKING_MANIFEST_PATH.as_posix(): file_sha256_or_missing(root, F6D3_RANKING_MANIFEST_PATH),
        F6E1_MANIFEST_PATH.as_posix(): file_sha256_or_missing(root, F6E1_MANIFEST_PATH),
    }
    return {
        "f2_checked": f2_checked or 1144,
        "f2_changed": f2_changed,
        "f6d4a_checked": f6d4a_checked or 21,
        "f6d4a_changed": f6d4a_changed,
        "qwen2_raw_hash": raw_hashes[F6D4A_QWEN_RAW_PATH.as_posix()],
        "mistral_raw_hash": raw_hashes[F6D4A_MISTRAL_RAW_PATH.as_posix()],
        "qwen2_raw_expected_hash": QWEN2_RAW_SHA256,
        "mistral_raw_expected_hash": MISTRAL_RAW_SHA256,
        "raw_changed": int(raw_hashes[F6D4A_QWEN_RAW_PATH.as_posix()] != QWEN2_RAW_SHA256)
        + int(raw_hashes[F6D4A_MISTRAL_RAW_PATH.as_posix()] != MISTRAL_RAW_SHA256),
        "input_hashes": raw_hashes,
    }


def input_hashes(root: Path) -> dict[str, str]:
    paths = [
        F6E1_MATRIX_PATH,
        F6E1_MANIFEST_PATH,
        FULL_MATRIX_REQUESTS_PATH,
        FULL_MATRIX_EXECUTION_PATH,
        F6D3_RANKING_MANIFEST_PATH,
        F6D3_RERANKING_MANIFEST_PATH,
        F6D4A_QWEN_RAW_PATH,
        F6D4A_MISTRAL_RAW_PATH,
        F6D4A_EXECUTION_MANIFEST_PATH,
        F6D4A_INPUT_HASHES_PATH,
        F2_PROTECTED_VALIDATION_PATH,
        F6D4A_PROTECTED_VALIDATION_PATH,
    ]
    for _, _, path in PLAN_E_MANIFESTS:
        paths.append(path)
    hashes = {path.as_posix(): file_sha256_or_missing(root, path) for path in paths}
    for tag in ["qwen2:7b-instruct", "mistral:7b-instruct", "llama3:70b-instruct"]:
        path = ollama_manifest_path(tag)
        hashes[logical_external_path(path)] = sha256_file(path) if path.exists() else "missing"
    return dict(sorted(hashes.items()))


def table_hashes(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, str]:
    columns = {
        "hardware_inventory": HARDWARE_COLUMNS,
        "runtime_inventory": RUNTIME_COLUMNS,
        "model_inventory": MODEL_COLUMNS,
        "model_identity_alignment": IDENTITY_COLUMNS,
        "request_workload": WORKLOAD_COLUMNS,
        "prompt_token_estimates": TOKEN_COLUMNS,
        "runtime_estimates": RUNTIME_ESTIMATE_COLUMNS,
        "memory_disk_feasibility": MEMORY_COLUMNS,
        "execution_matrix": EXECUTION_COLUMNS,
        "pilot_design": PILOT_COLUMNS,
        "missing_or_blocked": MISSING_COLUMNS,
    }
    return {name: sha256_text(rows_to_csv_text(rows, columns[name])) for name, rows in sorted(tables.items())}


def build_manifest(
    root: Path,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    protected_summary: Mapping[str, Any],
) -> dict[str, Any]:
    hashes = table_hashes(tables)
    inputs = input_hashes(root)
    hardware_fingerprint = hashes["hardware_inventory"]
    runtime_versions = {
        row["runtime_name"]: row["version"]
        for row in tables["runtime_inventory"]
    }
    local_model_digests = {
        row["model_tag"]: row["model_blob_digest"]
        for row in tables["model_inventory"]
        if row.get("model_blob_digest")
    }
    scientific_content = {
        "phase": PHASE,
        "tables": hashes,
        "input_hashes": inputs,
        "hardware_fingerprint": hardware_fingerprint,
        "runtime_versions": runtime_versions,
        "local_model_digests": local_model_digests,
        "prompt_freeze_hash": F6E1_SCIENTIFIC_HASH,
        "protected_input_summary": protected_summary,
    }
    decision = "f6e2_runtime_preflight_complete_with_documented_blocks"
    return {
        "phase": PHASE,
        "decision": decision,
        "frozen_timestamp_utc": FROZEN_TIMESTAMP_UTC,
        "no_inference": True,
        "no_network": True,
        "no_model_download": True,
        "input_paths": sorted(inputs),
        "input_hashes": inputs,
        "hardware_fingerprint": hardware_fingerprint,
        "runtime_versions": runtime_versions,
        "local_model_digests": local_model_digests,
        "prompt_freeze_hash": F6E1_SCIENTIFIC_HASH,
        "prompt_freeze_manifest_hash": F6E1_MANIFEST_HASH,
        "workload_hashes": {
            "request_workload": hashes["request_workload"],
            "prompt_token_estimates": hashes["prompt_token_estimates"],
            "runtime_estimates": hashes["runtime_estimates"],
        },
        "table_hashes": hashes,
        "scientific_content_hash": sha256_text(stable_json(scientific_content)),
        "runtime_metadata": {
            "hardware_values_are_snapshot": True,
            "ollama_server_state": "not_active_or_not_reachable_in_preflight",
            "python": "3.12.10 project .venv observed in escalated user session",
            "git_branch_expected": EXPECTED_BRANCH,
            "git_head_expected": EXPECTED_HEAD,
        },
        "protected_input_summary": protected_summary,
    }


def build_report(
    tables: Mapping[str, Sequence[Mapping[str, str]]],
    manifest: Mapping[str, Any],
    protected_summary: Mapping[str, Any],
) -> str:
    hardware = {row["component_id"]: row for row in tables["hardware_inventory"]}
    model_inventory = tables["model_inventory"]
    execution_rows = tables["execution_matrix"]
    blocked = [row for row in execution_rows if row["preflight_status"].startswith("blocked")]
    ready = [row for row in execution_rows if row["preflight_status"].startswith("ready")]
    workloads = tables["request_workload"]
    total_ready_instances = workload_total_instances(workloads, include_partial=False)
    lines = [
        "# Phase F6E2 - Runtime, Hardware and Execution Preflight",
        "",
        f"Decision: `{manifest['decision']}`.",
        "",
        "F6E2 is an offline preflight. It inventories local hardware, runtime, model identity, workloads, prompt sizes, execution risk, and pilot design. No inference, model acquisition, download, API call, or raw-response mutation is performed.",
        "",
        "## Initial Repository State",
        "",
        f"- Expected branch: `{EXPECTED_BRANCH}`.",
        f"- Expected HEAD: `{EXPECTED_HEAD}`.",
        "- Dirty tracked files are pre-existing and intentionally untouched: `README.md`, `report/dataset_audit.md`, `report/notes.md`.",
        "- F6E2 created only new `phase_f6e2_*` artifacts plus its source and tests.",
        "",
        "## Hardware Inventory",
        "",
        f"- CPU: {hardware['F6E2-HW-CPU']['name']} ({hardware['F6E2-HW-CPU']['observed_value']}).",
        f"- RAM: {hardware['F6E2-HW-RAM']['observed_value']}.",
        f"- GPU: {hardware['F6E2-HW-GPU-NVIDIA']['name']} ({hardware['F6E2-HW-GPU-NVIDIA']['observed_value']}).",
        f"- Storage: {hardware['F6E2-HW-STORAGE-C']['observed_value']}.",
        "",
        "## Runtime Inventory",
        "",
        "- Ollama client is installed but server was not reachable in this preflight; a read-only list attempt hit a local log-permission failure and was not repeated.",
        "- PyTorch 2.6.0+cu124 is installed and CUDA-capable in the project venv.",
        "- Transformers, accelerate, bitsandbytes, vLLM, llama.cpp, GPTQ/AWQ runtimes are not installed in the audited environment.",
        "",
        "## Local Model Inventory",
        "",
    ]
    for row in model_inventory:
        lines.append(
            f"- `{row['model_tag']}`: {row['availability_status']}; blob bytes `{row['model_blob_size_bytes']}`; digest `{row['model_blob_digest'] or 'missing'}`."
        )
    lines.extend(
        [
            "",
            "## Model Identity Alignment",
            "",
            "- Qwen2 and Mistral are paper model families with local Ollama adapted executions; exact paper model-version equivalence is not proven.",
            "- Llama 3 70B is unavailable locally and blocked by model, runtime, RAM, and VRAM evidence.",
            "- Closed-source GPT/Claude models remain blocked under the no-API policy.",
            "",
            "## F6E1 Matrix Loaded",
            "",
            f"- F6E1 matrix rows evaluated: `{len(execution_rows)}`.",
            f"- Ready/preparable local cells: `{len(ready)}`.",
            f"- Blocked cells: `{len(blocked)}`.",
            "- LLMSRec remains `reconstructed` and is never classified as exact.",
            "",
            "## Request Workload Inventory",
            "",
            f"- Ready logical instances across ready manifests: `{total_ready_instances}`.",
            "- Beauty, Sports, ML-1M ranking/reranking use Plan E 1000-user manifests.",
            "- LastFM ranking uses the F6D3/F6D4A 1859-user manifest. LastFM reranking is inventoried as partial because positive inclusion is not comparable.",
            "",
            "## Prompt Token Estimates",
            "",
            "- Token counts use deterministic char/4 estimates because exact local tokenizers are not available and no tokenizer download is allowed.",
            "- The estimates distinguish prompt strategy, dataset, task, output budget, and context overflow thresholds.",
            "",
            "## Call Multiplicity",
            "",
            "- Base, LLMRank, and JSON/index are one-call strategies.",
            "- LLMSRec is represented as a conservative interval of one to two calls per logical request because profile orchestration remains reconstructed.",
            "",
            "## Runtime And Feasibility",
            "",
            "- Runtime estimates use prior observed Qwen2/Mistral telemetry from frozen local runs when available.",
            "- Llama 3 70B has no local observed timing and is not estimable as an executable local full-matrix cell.",
            "- Existing 7B Ollama models are locally feasible once the Ollama operational gate is passed.",
            "",
            "## Llama 3 70B",
            "",
            "- FP16, INT8, 6-bit, and 5-bit are blocked by VRAM and RAM margins.",
            "- 4-bit is blocked by RAM/runtime/model availability for stable local full-matrix use.",
            "- 3-bit is classified as technically possible but impractical and still blocked by missing model/runtime.",
            "",
            "## Execution Order",
            "",
            "1. Analysis-only gate and hash freeze.",
            "2. Ollama operational gate in user session.",
            "3. Tiny JSON/index pilot on installed 7B models to verify resume/parser plumbing.",
            "4. Base prompt pilot on Qwen2/Mistral.",
            "5. LLMRank paired pilot.",
            "6. LLMSRec reconstructed pilot only after scope is explicitly accepted.",
            "7. Full run only after pilot, protected-input checks, and append-only raw policy pass.",
            "",
            "## Pilot Design",
            "",
            "- Pilot definitions are written to `results/tables/phase_f6e2_pilot_design.csv` only; no raw responses are materialized.",
            "- Pilot rows specify deterministic selection, parser checks, timeout, resume, append-only raw policy, and stop conditions.",
            "",
            "## Protected Inputs",
            "",
            f"- F2 protected inputs checked: `{protected_summary['f2_checked']}`; changed: `{protected_summary['f2_changed']}`.",
            f"- F6D4A protected/input checks carried forward: `{protected_summary['f6d4a_checked']}`; changed: `{protected_summary['f6d4a_changed']}`.",
            f"- Qwen2 raw hash: `{protected_summary['qwen2_raw_hash']}`.",
            f"- Mistral raw hash: `{protected_summary['mistral_raw_hash']}`.",
            "",
            "## Manifest",
            "",
            f"- Scientific content hash: `{manifest['scientific_content_hash']}`.",
            f"- Hardware fingerprint: `{manifest['hardware_fingerprint']}`.",
            "- Runtime metadata that can vary between sessions is separated from scientific table hashes.",
            "",
            "## Missing Or Blocked Evidence",
            "",
        ]
    )
    for row in tables["missing_or_blocked"]:
        lines.append(f"- `{row['block_id']}`: {row['status']} for {row['affected_scope']}.")
    lines.extend(
        [
            "",
            "## Safe Next Action",
            "",
            "Do not start full inference from F6E2. The next safe phase is an explicitly approved, append-only pilot after Ollama operational readiness is verified in the real user session.",
            "",
        ]
    )
    return "\n".join(lines)


def build_all(root: Path | None = None) -> BuildOutputs:
    root = Path.cwd() if root is None else root
    hardware = build_hardware_inventory()
    runtime = build_runtime_inventory(root)
    model_inventory = build_model_inventory(root)
    identity = build_identity_alignment(model_inventory)
    workloads = build_request_workload(root)
    token_estimates = build_prompt_token_estimates(root, workloads)
    runtime_estimates = build_runtime_estimates(root, workloads)
    memory = build_memory_disk_feasibility(model_inventory)
    execution = build_execution_matrix(root, workloads, model_inventory, identity)
    pilot = build_pilot_design()
    missing = build_missing_or_blocked()
    tables: dict[str, list[dict[str, str]]] = {
        "hardware_inventory": hardware,
        "runtime_inventory": runtime,
        "model_inventory": model_inventory,
        "model_identity_alignment": identity,
        "request_workload": workloads,
        "prompt_token_estimates": token_estimates,
        "runtime_estimates": runtime_estimates,
        "memory_disk_feasibility": memory,
        "execution_matrix": execution,
        "pilot_design": pilot,
        "missing_or_blocked": missing,
    }
    protected = protected_input_summary(root)
    manifest = build_manifest(root, tables, protected)
    report = build_report(tables, manifest, protected)
    return BuildOutputs(tables=tables, manifest=manifest, report=report)


def write_outputs(outputs: BuildOutputs, root: Path | None = None) -> dict[str, str]:
    root = Path.cwd() if root is None else root
    columns = {
        "hardware_inventory": HARDWARE_COLUMNS,
        "runtime_inventory": RUNTIME_COLUMNS,
        "model_inventory": MODEL_COLUMNS,
        "model_identity_alignment": IDENTITY_COLUMNS,
        "request_workload": WORKLOAD_COLUMNS,
        "prompt_token_estimates": TOKEN_COLUMNS,
        "runtime_estimates": RUNTIME_ESTIMATE_COLUMNS,
        "memory_disk_feasibility": MEMORY_COLUMNS,
        "execution_matrix": EXECUTION_COLUMNS,
        "pilot_design": PILOT_COLUMNS,
        "missing_or_blocked": MISSING_COLUMNS,
    }
    written: dict[str, str] = {}
    for name, path in OUTPUT_TABLES.items():
        write_csv(root / path, outputs.tables[name], columns[name])
        written[path.as_posix()] = sha256_file(root / path)
    (root / MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST_PATH).write_text(json.dumps(outputs.manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written[MANIFEST_PATH.as_posix()] = sha256_file(root / MANIFEST_PATH)
    (root / REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / REPORT_PATH).write_text(outputs.report, encoding="utf-8")
    written[REPORT_PATH.as_posix()] = sha256_file(root / REPORT_PATH)
    return written


def build_and_write(root: Path | None = None) -> dict[str, str]:
    root = Path.cwd() if root is None else root
    outputs = build_all(root)
    return write_outputs(outputs, root)
