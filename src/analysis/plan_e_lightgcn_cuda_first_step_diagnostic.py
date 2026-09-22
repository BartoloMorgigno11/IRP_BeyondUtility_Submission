"""Tensor-level diagnostics for the Plan E LightGCN CUDA first step.

The helpers in this module deliberately avoid ``torch.save`` and optimizer
parameter IDs when hashing state.  They are used by the Phase 3M-D2 runner and
are covered by synthetic tests only.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch


TENSOR_HASH_METHOD = "sha256_semantic_name_dtype_shape_byte_order_cpu_contiguous_bytes"
MODEL_HASH_METHOD = "sha256_sorted_named_parameter_tensor_content_hashes"
OPTIMIZER_HASH_METHOD = "sha256_parameter_name_mapped_optimizer_state"

ALLOWED_CLASSIFICATIONS = {
    "d1_state_hash_serialization_artifact",
    "d1_optimizer_hash_parameter_id_artifact",
    "cuda_sparse_forward_tensor_nondeterminism",
    "cuda_sparse_backward_gradient_nondeterminism",
    "cuda_default_adam_update_nondeterminism",
    "cuda_foreach_adam_update_nondeterminism",
    "cuda_optimizer_state_only_nondeterminism",
    "mixed_cuda_forward_backward_optimizer_nondeterminism",
    "no_first_step_divergence_reproduced",
    "root_cause_unresolved",
    "diagnostic_blocked",
}

ALLOWED_SCALE_UP_DECISIONS = {
    "approve_existing_cuda_pipeline_scale_up",
    "require_corrected_ml1m_rerun_with_foreach_false",
    "approve_cpu_only_scale_up",
    "require_sparse_backward_followup",
    "require_optimizer_followup",
    "do_not_scale_up",
    "diagnostic_incomplete",
}

STAGE_ORDER = [
    "batch_users",
    "batch_positive_items",
    "batch_negative_items",
    "adjacency_indices",
    "adjacency_values",
    "initial_user_embedding_parameter",
    "initial_item_embedding_parameter",
    "layer_0_all_embeddings",
    "layer_1_all_embeddings",
    "layer_2_all_embeddings",
    "final_all_embeddings",
    "final_user_embeddings",
    "final_item_embeddings",
    "batch_user_embeddings",
    "batch_positive_embeddings",
    "batch_negative_embeddings",
    "positive_scores",
    "negative_scores",
    "bpr_loss_tensor",
    "regularization_loss_tensor",
    "total_loss_tensor",
    "user_embedding_gradient",
    "item_embedding_gradient",
    "post_step_user_embedding_parameter",
    "post_step_item_embedding_parameter",
    "user_step",
    "user_exp_avg",
    "user_exp_avg_sq",
    "item_step",
    "item_exp_avg",
    "item_exp_avg_sq",
]

FORWARD_STAGE_NAMES = {
    "layer_0_all_embeddings",
    "layer_1_all_embeddings",
    "layer_2_all_embeddings",
    "final_all_embeddings",
    "final_user_embeddings",
    "final_item_embeddings",
    "batch_user_embeddings",
    "batch_positive_embeddings",
    "batch_negative_embeddings",
    "positive_scores",
    "negative_scores",
}

LOSS_STAGE_NAMES = {"bpr_loss_tensor", "regularization_loss_tensor", "total_loss_tensor"}
GRADIENT_STAGE_NAMES = {"user_embedding_gradient", "item_embedding_gradient"}
POST_STEP_PARAMETER_STAGE_NAMES = {"post_step_user_embedding_parameter", "post_step_item_embedding_parameter"}
OPTIMIZER_STAGE_NAMES = {
    "user_step",
    "user_exp_avg",
    "user_exp_avg_sq",
    "item_step",
    "item_exp_avg",
    "item_exp_avg_sq",
}


@dataclass(frozen=True)
class TensorComparison:
    """Direct tensor comparison evidence suitable for CSV serialization."""

    tensor_dtype: str
    tensor_shape: str
    element_count: int
    left_content_sha256: str
    right_content_sha256: str
    torch_equal: bool
    mismatching_element_count: int
    mismatch_fraction: float
    maximum_absolute_difference: float | str
    mean_absolute_difference: float | str
    maximum_relative_difference: float | str
    first_mismatch_flat_index: int | str
    left_first_mismatch_value: str
    right_first_mismatch_value: str
    all_values_finite: bool
    status: str
    interpretation: str


def canonical_int_array_sha256(values: np.ndarray | Sequence[int]) -> str:
    """Hash an integer array in the same shape-aware form used by D1."""

    array = np.asarray(values, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest().upper()


def tensor_content_sha256(name: str, tensor: torch.Tensor) -> str:
    """Return a device-independent SHA256 over semantic tensor contents."""

    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(TENSOR_HASH_METHOD.encode("utf-8"))
    digest.update(b"\0name=")
    digest.update(str(name).encode("utf-8"))
    digest.update(b"\0dtype=")
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(b"\0shape=")
    digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0byte_order=")
    digest.update(sys.byteorder.encode("ascii"))
    digest.update(b"\0payload=")
    digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest().upper()


def stable_json_sha256(payload: Any) -> str:
    """Hash JSON-serializable data with stable key ordering."""

    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def named_tensor_hashes(named_tensors: Mapping[str, torch.Tensor]) -> dict[str, str]:
    """Hash each tensor by sorted semantic name."""

    return {name: tensor_content_sha256(name, named_tensors[name]) for name in sorted(named_tensors)}


def combined_named_tensor_sha256(named_tensors: Mapping[str, torch.Tensor]) -> str:
    """Hash a named tensor collection independent of dictionary order."""

    digest = hashlib.sha256()
    digest.update(MODEL_HASH_METHOD.encode("utf-8"))
    for name in sorted(named_tensors):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor_content_sha256(name, named_tensors[name]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def clone_named_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone model parameters to CPU using semantic parameter names."""

    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}


def clone_named_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    """Clone gradients to CPU while preserving missing-gradient information."""

    gradients: dict[str, torch.Tensor | None] = {}
    for name, parameter in model.named_parameters():
        gradients[name] = None if parameter.grad is None else parameter.grad.detach().cpu().clone()
    return gradients


def inject_named_gradients(
    model: torch.nn.Module,
    gradients: Mapping[str, torch.Tensor | None],
    *,
    device: torch.device,
) -> None:
    """Inject cloned gradients into a model by parameter name."""

    by_name = dict(model.named_parameters())
    for name, parameter in by_name.items():
        gradient = gradients.get(name)
        parameter.grad = None if gradient is None else gradient.detach().clone().to(device=device, dtype=parameter.dtype)


def legacy_state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Reproduce the Phase 3M/D1 named state hash for audit comparison."""

    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest().upper()


def canonical_model_state_sha256(model: torch.nn.Module) -> str:
    """Hash model parameters by semantic name and tensor content."""

    return combined_named_tensor_sha256(clone_named_parameters(model))


def _parameter_name_by_object(model: torch.nn.Module) -> dict[torch.nn.Parameter, str]:
    return {parameter: name for name, parameter in model.named_parameters()}


def optimizer_state_by_parameter_name(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
) -> dict[str, dict[str, Any]]:
    """Return optimizer state keyed by model parameter name, not parameter ID."""

    names = _parameter_name_by_object(model)
    mapped: dict[str, dict[str, Any]] = {}
    for parameter, state in optimizer.state.items():
        name = names.get(parameter)
        if name is None:
            continue
        mapped[name] = {str(field): value for field, value in state.items()}
    return mapped


def _scalar_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return {"type": "bool", "value": bool(value)}
    if isinstance(value, int):
        return {"type": "int", "value": int(value)}
    if isinstance(value, float):
        return {"type": "float", "value": format(float(value), ".17g")}
    return {"type": type(value).__name__, "value": str(value)}


def optimizer_state_field_sha256(parameter_name: str, field_name: str, value: Any) -> str:
    """Hash one optimizer state field using a semantic parameter name."""

    digest = hashlib.sha256()
    digest.update(OPTIMIZER_HASH_METHOD.encode("utf-8"))
    digest.update(b"\0parameter=")
    digest.update(parameter_name.encode("utf-8"))
    digest.update(b"\0field=")
    digest.update(field_name.encode("utf-8"))
    if torch.is_tensor(value):
        digest.update(b"\0tensor_hash=")
        digest.update(tensor_content_sha256(f"{parameter_name}.{field_name}", value).encode("ascii"))
    else:
        digest.update(b"\0scalar=")
        digest.update(json.dumps(_scalar_payload(value), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest().upper()


def canonical_optimizer_state_sha256(optimizer: torch.optim.Optimizer, model: torch.nn.Module) -> str:
    """Hash optimizer state by parameter name and sorted field name."""

    mapped = optimizer_state_by_parameter_name(optimizer, model)
    digest = hashlib.sha256()
    digest.update(OPTIMIZER_HASH_METHOD.encode("utf-8"))
    for parameter_name in sorted(mapped):
        for field_name in sorted(mapped[parameter_name]):
            digest.update(parameter_name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(field_name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(
                optimizer_state_field_sha256(parameter_name, field_name, mapped[parameter_name][field_name]).encode(
                    "ascii"
                )
            )
            digest.update(b"\n")
    return digest.hexdigest().upper()


def _tensor_shape_text(tensor: torch.Tensor) -> str:
    return json.dumps(list(tensor.shape), separators=(",", ":"))


def _first_mismatch(left: torch.Tensor, right: torch.Tensor) -> tuple[int | str, str, str]:
    if left.shape != right.shape:
        return "", "", ""
    mismatch = torch.ne(left.reshape(-1), right.reshape(-1))
    if not bool(mismatch.any().item()):
        return "", "", ""
    index = int(torch.nonzero(mismatch, as_tuple=False)[0].item())
    left_value = left.reshape(-1)[index].item()
    right_value = right.reshape(-1)[index].item()
    return index, _format_scalar(left_value), _format_scalar(right_value)


def _format_scalar(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)):
            return format(float(value), ".17g")
        return str(float(value))
    return str(value)


def _all_finite(tensor: torch.Tensor) -> bool:
    if torch.is_floating_point(tensor) or torch.is_complex(tensor):
        return bool(torch.isfinite(tensor).all().item())
    return True


def compare_tensors(name: str, left: torch.Tensor | None, right: torch.Tensor | None) -> TensorComparison:
    """Compare two tensors directly and report exact mismatch statistics."""

    if left is None or right is None:
        return TensorComparison(
            tensor_dtype="",
            tensor_shape="",
            element_count=0,
            left_content_sha256="",
            right_content_sha256="",
            torch_equal=False,
            mismatching_element_count=0,
            mismatch_fraction=0.0,
            maximum_absolute_difference="",
            mean_absolute_difference="",
            maximum_relative_difference="",
            first_mismatch_flat_index="",
            left_first_mismatch_value="",
            right_first_mismatch_value="",
            all_values_finite=False,
            status="not_applicable",
            interpretation="tensor missing from one or both sides",
        )

    left_cpu = left.detach().cpu().contiguous()
    right_cpu = right.detach().cpu().contiguous()
    left_hash = tensor_content_sha256(name, left_cpu)
    right_hash = tensor_content_sha256(name, right_cpu)
    dtype_text = str(left_cpu.dtype) if left_cpu.dtype == right_cpu.dtype else f"{left_cpu.dtype}|{right_cpu.dtype}"
    shape_text = _tensor_shape_text(left_cpu) if left_cpu.shape == right_cpu.shape else f"{list(left_cpu.shape)}|{list(right_cpu.shape)}"
    element_count = int(left_cpu.numel()) if left_cpu.shape == right_cpu.shape else 0
    all_finite = _all_finite(left_cpu) and _all_finite(right_cpu)

    if left_cpu.shape != right_cpu.shape or left_cpu.dtype != right_cpu.dtype:
        return TensorComparison(
            tensor_dtype=dtype_text,
            tensor_shape=shape_text,
            element_count=element_count,
            left_content_sha256=left_hash,
            right_content_sha256=right_hash,
            torch_equal=False,
            mismatching_element_count=element_count,
            mismatch_fraction=1.0 if element_count else 0.0,
            maximum_absolute_difference="",
            mean_absolute_difference="",
            maximum_relative_difference="",
            first_mismatch_flat_index="",
            left_first_mismatch_value="",
            right_first_mismatch_value="",
            all_values_finite=all_finite,
            status="different",
            interpretation="shape or dtype differs",
        )

    torch_equal = bool(torch.equal(left_cpu, right_cpu))
    mismatch_mask = torch.ne(left_cpu.reshape(-1), right_cpu.reshape(-1))
    mismatches = int(mismatch_mask.sum().item())
    mismatch_fraction = 0.0 if element_count == 0 else float(mismatches) / float(element_count)
    first_index, first_left, first_right = _first_mismatch(left_cpu, right_cpu)

    max_abs: float | str = ""
    mean_abs: float | str = ""
    max_relative: float | str = ""
    if torch.is_floating_point(left_cpu) or torch.is_complex(left_cpu):
        left_float = left_cpu.to(torch.float64)
        right_float = right_cpu.to(torch.float64)
        diff = torch.abs(left_float - right_float)
        if diff.numel():
            max_abs = float(diff.max().item())
            mean_abs = float(diff.mean().item())
            denominator = torch.maximum(torch.abs(left_float), torch.abs(right_float))
            valid = denominator > 0
            relative = torch.zeros_like(diff)
            relative[valid] = diff[valid] / denominator[valid]
            if diff.numel():
                max_relative = float(relative.max().item())
    else:
        left_float = left_cpu.to(torch.float64)
        right_float = right_cpu.to(torch.float64)
        diff = torch.abs(left_float - right_float)
        if diff.numel():
            max_abs = float(diff.max().item())
            mean_abs = float(diff.mean().item())
            max_relative = 0.0 if mismatches == 0 else ""

    status = "exact" if torch_equal and left_hash == right_hash else "different"
    if status == "exact":
        interpretation = "tensor contents are exactly equal"
    elif mismatches == 0:
        interpretation = "content hash differs without element mismatch"
    elif torch.is_floating_point(left_cpu):
        interpretation = "floating tensor contents differ bitwise"
    else:
        interpretation = "integer or non-floating tensor contents differ"

    return TensorComparison(
        tensor_dtype=dtype_text,
        tensor_shape=shape_text,
        element_count=element_count,
        left_content_sha256=left_hash,
        right_content_sha256=right_hash,
        torch_equal=torch_equal,
        mismatching_element_count=mismatches,
        mismatch_fraction=mismatch_fraction,
        maximum_absolute_difference=max_abs,
        mean_absolute_difference=mean_abs,
        maximum_relative_difference=max_relative,
        first_mismatch_flat_index=first_index,
        left_first_mismatch_value=first_left,
        right_first_mismatch_value=first_right,
        all_values_finite=all_finite,
        status=status,
        interpretation=interpretation,
    )


def comparison_to_row(comparison: TensorComparison) -> dict[str, Any]:
    """Convert comparison evidence to a mutable row dictionary."""

    return {
        "tensor_dtype": comparison.tensor_dtype,
        "tensor_shape": comparison.tensor_shape,
        "element_count": comparison.element_count,
        "left_content_sha256": comparison.left_content_sha256,
        "right_content_sha256": comparison.right_content_sha256,
        "torch_equal": comparison.torch_equal,
        "mismatching_element_count": comparison.mismatching_element_count,
        "mismatch_fraction": comparison.mismatch_fraction,
        "maximum_absolute_difference": comparison.maximum_absolute_difference,
        "mean_absolute_difference": comparison.mean_absolute_difference,
        "maximum_relative_difference": comparison.maximum_relative_difference,
        "first_mismatch_flat_index": comparison.first_mismatch_flat_index,
        "left_first_mismatch_value": comparison.left_first_mismatch_value,
        "right_first_mismatch_value": comparison.right_first_mismatch_value,
        "all_values_finite": comparison.all_values_finite,
        "status": comparison.status,
        "interpretation": comparison.interpretation,
    }


def first_divergence_stage(stage_rows: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """Return the earliest stage marked different and its interpretation."""

    by_stage = {str(row["stage_name"]): row for row in stage_rows}
    for stage_name in STAGE_ORDER:
        row = by_stage.get(stage_name)
        if row and row.get("status") == "different":
            return stage_name, str(row.get("interpretation", ""))
    return "none", "all applicable stage tensors are exact"


def stages_exact(stage_rows: Sequence[Mapping[str, Any]], names: set[str]) -> bool | str:
    """Return exact/not-applicable status for a named stage group."""

    selected = [row for row in stage_rows if row.get("stage_name") in names]
    applicable = [row for row in selected if row.get("status") != "not_applicable"]
    if not applicable:
        return "not_applicable"
    return all(row.get("status") == "exact" for row in applicable)


def max_observed_absolute_difference(stage_rows: Sequence[Mapping[str, Any]]) -> float:
    """Return the maximum populated absolute difference across stage rows."""

    values: list[float] = []
    for row in stage_rows:
        value = row.get("maximum_absolute_difference", "")
        if value == "" or value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return max(values) if values else 0.0


def classify_diagnostic(
    *,
    original_pipeline_first_divergence: str,
    forward_only_exact: bool | str,
    forward_backward_gradient_exact: bool | str,
    shared_default_post_step_exact: bool | str,
    shared_default_optimizer_exact: bool | str,
    shared_foreach_false_post_step_exact: bool | str,
    original_pipeline_all_exact: bool,
    legacy_hash_artifact_detected: bool,
    optimizer_parameter_id_artifact_detected: bool,
    blocked: bool = False,
) -> str:
    """Classify D2 evidence using the allowed Phase 3M-D2 labels."""

    if blocked:
        return "diagnostic_blocked"
    if original_pipeline_all_exact and legacy_hash_artifact_detected:
        return "d1_state_hash_serialization_artifact"
    if original_pipeline_all_exact and optimizer_parameter_id_artifact_detected:
        return "d1_optimizer_hash_parameter_id_artifact"
    if original_pipeline_all_exact:
        return "no_first_step_divergence_reproduced"
    if forward_only_exact is False:
        return "cuda_sparse_forward_tensor_nondeterminism"
    if forward_backward_gradient_exact is False:
        return "cuda_sparse_backward_gradient_nondeterminism"
    if shared_default_post_step_exact is False:
        if shared_foreach_false_post_step_exact is True:
            return "cuda_foreach_adam_update_nondeterminism"
        return "cuda_default_adam_update_nondeterminism"
    if shared_default_post_step_exact is True and shared_default_optimizer_exact is False:
        return "cuda_optimizer_state_only_nondeterminism"
    if original_pipeline_first_divergence != "none":
        return "mixed_cuda_forward_backward_optimizer_nondeterminism"
    return "root_cause_unresolved"


def validate_scale_up_decision(
    *,
    classification: str,
    original_pipeline_all_exact: bool,
    forward_exact: bool | str,
    gradients_exact: bool | str,
    default_adam_exact: bool | str,
    foreach_false_adam_exact: bool | str,
    diagnostic_complete: bool,
) -> str:
    """Return the D2 scale-up decision for the observed classification."""

    if classification not in ALLOWED_CLASSIFICATIONS:
        raise ValueError(f"unknown classification: {classification}")
    if not diagnostic_complete or classification in {"diagnostic_blocked", "root_cause_unresolved"}:
        return "diagnostic_incomplete" if not diagnostic_complete else "do_not_scale_up"
    if original_pipeline_all_exact and classification in {
        "d1_state_hash_serialization_artifact",
        "d1_optimizer_hash_parameter_id_artifact",
        "no_first_step_divergence_reproduced",
    }:
        return "approve_existing_cuda_pipeline_scale_up"
    if classification == "cuda_sparse_forward_tensor_nondeterminism":
        return "approve_cpu_only_scale_up"
    if gradients_exact is True and default_adam_exact is False and foreach_false_adam_exact is True:
        return "require_corrected_ml1m_rerun_with_foreach_false"
    if forward_exact is False or gradients_exact is False:
        return "require_sparse_backward_followup"
    if default_adam_exact is False:
        return "require_optimizer_followup"
    return "do_not_scale_up"


def optimizer_defaults_summary(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Extract the Adam defaults required by the D2 manifest/report."""

    defaults = dict(optimizer.defaults)
    betas = defaults.get("betas", ("", ""))
    def display(value: Any) -> Any:
        return "None" if value is None else value

    return {
        "optimizer_class": type(optimizer).__name__,
        "optimizer_foreach": display(defaults.get("foreach", "")),
        "optimizer_fused": display(defaults.get("fused", "")),
        "optimizer_capturable": display(defaults.get("capturable", "")),
        "optimizer_differentiable": display(defaults.get("differentiable", "")),
        "optimizer_amsgrad": display(defaults.get("amsgrad", "")),
        "learning_rate": display(defaults.get("lr", "")),
        "beta1": betas[0] if isinstance(betas, tuple) else "",
        "beta2": betas[1] if isinstance(betas, tuple) else "",
        "epsilon": display(defaults.get("eps", "")),
        "weight_decay": display(defaults.get("weight_decay", "")),
    }
