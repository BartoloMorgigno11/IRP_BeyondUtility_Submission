"""Utilities for the Plan E LightGCN ML-1M determinism diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ALLOWED_CLASSIFICATIONS = {
    "cuda_training_and_logging_bitwise_deterministic",
    "cuda_model_path_deterministic_loss_logging_not_bitwise",
    "cuda_model_path_numerically_deterministic_with_tolerance",
    "cuda_sparse_training_nondeterministic",
    "cpu_exact_determinism_cuda_logging_warning",
    "cpu_exact_determinism_cuda_model_warning",
    "cpu_determinism_failed",
    "diagnostic_incomplete_due_to_runtime",
    "diagnostic_blocked",
    "root_cause_unresolved",
}

ALLOWED_SCALE_UP_DECISIONS = {
    "approve_cuda_scale_up_with_documented_loss_telemetry_tolerance",
    "approve_cpu_only_scale_up",
    "require_additional_diagnostic",
    "do_not_scale_up",
}


@dataclass(frozen=True)
class LossAggregates:
    """Per-epoch aggregate loss summaries for a sequence of batch means."""

    batch_count: int
    training_example_count: int
    python_sum_loss: float
    math_fsum_loss: float
    numpy_float64_sum_loss: float
    torch_float64_sum_loss: float
    unweighted_batch_mean: float
    batch_size_weighted_mean: float
    minimum_batch_loss: float
    maximum_batch_loss: float
    first_batch_loss: float
    last_batch_loss: float
    batch_loss_sequence_sha256: str
    batch_size_sequence_sha256: str


def canonical_float(value: float | str | Decimal) -> str:
    """Return a deterministic decimal representation for a finite float."""

    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite float cannot be serialized: {value!r}")
    return format(number, ".17g")


def float_hex(value: float | str | Decimal) -> str:
    """Return the Python hexadecimal representation of a finite float."""

    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite float has no diagnostic hex form: {value!r}")
    return number.hex()


def _ordered_uint64(value: float) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    if bits & (1 << 63):
        return (~bits) & ((1 << 64) - 1)
    return bits | (1 << 63)


def ulp_distance(left: float | str | Decimal, right: float | str | Decimal) -> int:
    """Return an integer ULP-oriented distance for two finite float64 values."""

    left_float = float(left)
    right_float = float(right)
    if not math.isfinite(left_float) or not math.isfinite(right_float):
        raise ValueError("ULP distance requires finite values")
    return abs(_ordered_uint64(left_float) - _ordered_uint64(right_float))


def relative_difference(left: float | str | Decimal, right: float | str | Decimal) -> float:
    """Return a symmetric relative difference, handling zero denominators."""

    left_float = float(left)
    right_float = float(right)
    absolute = abs(left_float - right_float)
    denominator = max(abs(left_float), abs(right_float))
    return 0.0 if denominator == 0.0 else absolute / denominator


def rounded_equal(left: float | str | Decimal, right: float | str | Decimal, decimals: int) -> bool:
    """Return whether two floats match after decimal rounding."""

    return round(float(left), decimals) == round(float(right), decimals)


def numeric_sha256(payload: Any) -> str:
    """Hash a JSON-serializable payload with stable separators and key ordering."""

    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def hash_int_array(values: np.ndarray | Sequence[int]) -> str:
    """Hash integer array contents with dtype and shape metadata."""

    array = np.asarray(values, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest().upper()


def hash_batch_losses(losses: Sequence[float]) -> str:
    """Hash canonical serialized per-batch loss values."""

    return numeric_sha256([canonical_float(value) for value in losses])


def hash_batch_sizes(batch_sizes: Sequence[int]) -> str:
    """Hash per-batch sizes deterministically."""

    return numeric_sha256([int(value) for value in batch_sizes])


def aggregate_batch_losses(losses: Sequence[float], batch_sizes: Sequence[int]) -> LossAggregates:
    """Return all required loss aggregations for one epoch."""

    if len(losses) == 0:
        raise ValueError("cannot aggregate an empty batch-loss sequence")
    if len(losses) != len(batch_sizes):
        raise ValueError("loss and batch-size sequences have different lengths")
    if any(size <= 0 for size in batch_sizes):
        raise ValueError("batch sizes must be positive")
    loss_array = np.asarray([float(value) for value in losses], dtype=np.float64)
    size_array = np.asarray([int(value) for value in batch_sizes], dtype=np.float64)
    if not np.isfinite(loss_array).all():
        raise ValueError("non-finite batch loss encountered")
    training_examples = int(size_array.sum())
    weighted_sum = float(np.dot(loss_array, size_array))
    return LossAggregates(
        batch_count=len(losses),
        training_example_count=training_examples,
        python_sum_loss=float(sum(float(value) for value in losses)),
        math_fsum_loss=float(math.fsum(float(value) for value in losses)),
        numpy_float64_sum_loss=float(np.sum(loss_array, dtype=np.float64)),
        torch_float64_sum_loss=float(torch.as_tensor(loss_array, dtype=torch.float64).sum().item()),
        unweighted_batch_mean=float(np.mean(loss_array)),
        batch_size_weighted_mean=weighted_sum / float(training_examples),
        minimum_batch_loss=float(np.min(loss_array)),
        maximum_batch_loss=float(np.max(loss_array)),
        first_batch_loss=float(loss_array[0]),
        last_batch_loss=float(loss_array[-1]),
        batch_loss_sequence_sha256=hash_batch_losses(loss_array.tolist()),
        batch_size_sequence_sha256=hash_batch_sizes([int(value) for value in batch_sizes]),
    )


def tensor_state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash a tensor state dict without relying on torch serialization."""

    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest().upper()


def optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
    """Hash optimizer state tensors and scalar hyperparameters deterministically."""

    state = optimizer.state_dict()
    digest = hashlib.sha256()
    digest.update(json.dumps(state.get("param_groups", []), sort_keys=True, default=str).encode("utf-8"))
    for parameter_id in sorted(state.get("state", {}), key=lambda value: str(value)):
        digest.update(str(parameter_id).encode("utf-8"))
        parameter_state = state["state"][parameter_id]
        for key in sorted(parameter_state):
            value = parameter_state[key]
            digest.update(str(key).encode("utf-8"))
            if torch.is_tensor(value):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tuple(tensor.shape)).encode("utf-8"))
                digest.update(str(tensor.dtype).encode("utf-8"))
                digest.update(tensor.numpy().tobytes())
            else:
                digest.update(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest().upper()


def first_divergent_index(left: Sequence[Any], right: Sequence[Any]) -> int | None:
    """Return the first zero-based index where two sequences differ."""

    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if left_value != right_value:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def all_status_passed(rows: Sequence[Mapping[str, Any]], *, scope: str | None = None) -> bool:
    """Return whether all rows in an optional diagnostic scope passed."""

    selected = [row for row in rows if scope is None or row.get("diagnostic_scope") == scope]
    return bool(selected) and all(row.get("status") in {"passed", "not_applicable"} for row in selected)


def classify_diagnostic(
    *,
    cuda_state_exact: bool,
    cuda_optimizer_exact: bool,
    cuda_loss_exact: bool,
    phase3m_epoch_loss_exact: bool,
    cpu_repeat_exact: bool,
    cpu_final_state_exact: bool,
    cpu_embeddings_exact: bool,
    cpu_scores_exact: bool,
    cpu_rankings_exact: bool,
    phase3m_final_outputs_exact: bool,
    completed: bool = True,
) -> str:
    """Classify the diagnostic using the Phase 3M-D1 decision rules."""

    if not completed:
        return "diagnostic_incomplete_due_to_runtime"
    if not cpu_repeat_exact:
        return "cpu_determinism_failed"
    if not cuda_state_exact or not cuda_optimizer_exact:
        return "cuda_sparse_training_nondeterministic"
    if (
        cpu_final_state_exact
        and cpu_embeddings_exact
        and cpu_scores_exact
        and cpu_rankings_exact
        and phase3m_final_outputs_exact
        and (not cuda_loss_exact or not phase3m_epoch_loss_exact)
    ):
        return "cpu_exact_determinism_cuda_logging_warning"
    if cuda_state_exact and cuda_optimizer_exact and not cuda_loss_exact:
        return "cuda_model_path_deterministic_loss_logging_not_bitwise"
    if cuda_state_exact and cuda_optimizer_exact and cuda_loss_exact and cpu_repeat_exact:
        return "cuda_training_and_logging_bitwise_deterministic"
    return "root_cause_unresolved"


def validate_scale_up_decision(
    *,
    classification: str,
    phase3m_final_outputs_exact: bool,
    cuda_state_exact: bool,
    cuda_optimizer_exact: bool,
    cpu_repeat_exact: bool,
    no_error_failures: bool,
    tolerance: str,
) -> tuple[str, str]:
    """Return the approved scale-up decision and device."""

    if classification not in ALLOWED_CLASSIFICATIONS:
        raise ValueError(f"unknown diagnostic classification: {classification}")
    if (
        phase3m_final_outputs_exact
        and cuda_state_exact
        and cuda_optimizer_exact
        and cpu_repeat_exact
        and no_error_failures
        and classification
        in {
            "cpu_exact_determinism_cuda_logging_warning",
            "cuda_model_path_deterministic_loss_logging_not_bitwise",
            "cuda_training_and_logging_bitwise_deterministic",
        }
    ):
        if not tolerance:
            raise ValueError("CUDA scale-up approval requires a non-empty tolerance")
        return "approve_cuda_scale_up_with_documented_loss_telemetry_tolerance", "cuda"
    if cpu_repeat_exact and no_error_failures:
        return "approve_cpu_only_scale_up", "cpu"
    if classification in {"diagnostic_blocked", "diagnostic_incomplete_due_to_runtime", "root_cause_unresolved"}:
        return "require_additional_diagnostic", ""
    return "do_not_scale_up", ""


def cross_device_metric_row(
    *,
    comparison_scope: str,
    metric: str,
    cuda_value: float | str,
    cpu_value: float | str,
    equality_expected: str,
    status: str,
    interpretation: str,
    notes: str = "",
    agreement_rate: float | str = "",
) -> dict[str, Any]:
    """Create one CPU/CUDA comparison row."""

    cuda_number = None if cuda_value == "" else float(cuda_value)
    cpu_number = None if cpu_value == "" else float(cpu_value)
    if cuda_number is None or cpu_number is None:
        absolute = ""
        relative = ""
    else:
        absolute = abs(cuda_number - cpu_number)
        relative = relative_difference(cuda_number, cpu_number)
    return {
        "comparison_scope": comparison_scope,
        "metric": metric,
        "cuda_value": cuda_value,
        "cpu_value": cpu_value,
        "absolute_difference": absolute,
        "relative_difference": relative,
        "agreement_rate": agreement_rate,
        "equality_expected": equality_expected,
        "status": status,
        "interpretation": interpretation,
        "notes": notes,
    }
