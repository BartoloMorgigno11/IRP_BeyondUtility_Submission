from __future__ import annotations

import pytest
import torch

from src.analysis.plan_e_lightgcn_cuda_first_step_diagnostic import (
    canonical_model_state_sha256,
    canonical_optimizer_state_sha256,
    classify_diagnostic,
    clone_named_gradients,
    combined_named_tensor_sha256,
    compare_tensors,
    inject_named_gradients,
    named_tensor_hashes,
    optimizer_state_by_parameter_name,
    tensor_content_sha256,
    validate_scale_up_decision,
)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.user_embedding = torch.nn.Embedding(2, 3)
        self.item_embedding = torch.nn.Embedding(2, 3)

    def forward(self) -> torch.Tensor:
        return (self.user_embedding.weight * self.item_embedding.weight).sum()


def test_tensor_content_hash_is_device_independent_when_content_matches() -> None:
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    clone = tensor.clone()
    assert tensor_content_sha256("x", tensor) == tensor_content_sha256("x", clone)


def test_tensor_content_hash_includes_dtype_and_shape() -> None:
    base = torch.arange(4, dtype=torch.float32)
    same_bytes_other_shape = base.reshape(2, 2)
    other_dtype = base.to(torch.float64)
    assert tensor_content_sha256("x", base) != tensor_content_sha256("x", same_bytes_other_shape)
    assert tensor_content_sha256("x", base) != tensor_content_sha256("x", other_dtype)


def test_named_tensor_hashing_is_dictionary_order_independent() -> None:
    left = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
    right = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    assert named_tensor_hashes(left) == named_tensor_hashes(right)
    assert combined_named_tensor_sha256(left) == combined_named_tensor_sha256(right)


def test_model_state_hash_sorts_parameter_names() -> None:
    torch.manual_seed(7)
    model = TinyModel()
    first = canonical_model_state_sha256(model)
    second = canonical_model_state_sha256(model)
    assert first == second


def _stepped_optimizer(model: TinyModel) -> torch.optim.Optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = model()
    loss.backward()
    optimizer.step()
    return optimizer


def test_optimizer_hash_is_parameter_id_independent() -> None:
    torch.manual_seed(11)
    first = TinyModel()
    state = {name: tensor.detach().clone() for name, tensor in first.state_dict().items()}
    second = TinyModel()
    second.load_state_dict(state)
    optimizer_a = _stepped_optimizer(first)
    optimizer_b = _stepped_optimizer(second)
    assert canonical_optimizer_state_sha256(optimizer_a, first) == canonical_optimizer_state_sha256(optimizer_b, second)
    assert set(optimizer_state_by_parameter_name(optimizer_a, first)) == {
        "user_embedding.weight",
        "item_embedding.weight",
    }


def test_optimizer_hash_changes_for_genuine_moment_difference() -> None:
    torch.manual_seed(13)
    first = TinyModel()
    second = TinyModel()
    second.load_state_dict(first.state_dict())
    optimizer_a = _stepped_optimizer(first)
    optimizer_b = _stepped_optimizer(second)
    mapped = optimizer_state_by_parameter_name(optimizer_b, second)
    mapped["user_embedding.weight"]["exp_avg"].add_(1.0)
    assert canonical_optimizer_state_sha256(optimizer_a, first) != canonical_optimizer_state_sha256(optimizer_b, second)


def test_optimizer_scalar_state_is_handled() -> None:
    model = TinyModel()
    optimizer = _stepped_optimizer(model)
    mapped = optimizer_state_by_parameter_name(optimizer, model)
    assert "step" in mapped["user_embedding.weight"]
    assert canonical_optimizer_state_sha256(optimizer, model)


def test_compare_exact_tensors() -> None:
    comparison = compare_tensors("x", torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))
    assert comparison.status == "exact"
    assert comparison.torch_equal is True
    assert comparison.mismatching_element_count == 0


def test_compare_one_element_mismatch() -> None:
    comparison = compare_tensors("x", torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 5.0, 3.0]))
    assert comparison.status == "different"
    assert comparison.first_mismatch_flat_index == 1
    assert comparison.mismatching_element_count == 1
    assert comparison.mismatch_fraction == pytest.approx(1 / 3)
    assert comparison.maximum_absolute_difference == pytest.approx(3.0)
    assert comparison.mean_absolute_difference == pytest.approx(1.0)
    assert comparison.maximum_relative_difference == pytest.approx(0.6)


def test_compare_integer_exactness() -> None:
    comparison = compare_tensors("x", torch.tensor([1, 2]), torch.tensor([1, 3]))
    assert comparison.status == "different"
    assert comparison.maximum_absolute_difference == pytest.approx(1.0)
    assert comparison.maximum_relative_difference == ""


def test_non_finite_detection() -> None:
    comparison = compare_tensors("x", torch.tensor([float("nan")]), torch.tensor([float("nan")]))
    assert comparison.all_values_finite is False
    assert comparison.status == "different"


def test_exact_gradient_cloning_and_injection() -> None:
    source = TinyModel()
    source.forward().backward()
    gradients = clone_named_gradients(source)
    target = TinyModel()
    inject_named_gradients(target, gradients, device=torch.device("cpu"))
    for name, parameter in target.named_parameters():
        assert parameter.grad is not None
        assert torch.equal(parameter.grad, gradients[name])


def test_missing_gradient_detection() -> None:
    model = TinyModel()
    gradients = clone_named_gradients(model)
    assert gradients["user_embedding.weight"] is None


def test_hash_artifact_classification() -> None:
    assert (
        classify_diagnostic(
            original_pipeline_first_divergence="none",
            forward_only_exact=True,
            forward_backward_gradient_exact=True,
            shared_default_post_step_exact=True,
            shared_default_optimizer_exact=True,
            shared_foreach_false_post_step_exact=True,
            original_pipeline_all_exact=True,
            legacy_hash_artifact_detected=True,
            optimizer_parameter_id_artifact_detected=False,
        )
        == "d1_state_hash_serialization_artifact"
    )


def test_sparse_forward_classification() -> None:
    assert (
        classify_diagnostic(
            original_pipeline_first_divergence="layer_1_all_embeddings",
            forward_only_exact=False,
            forward_backward_gradient_exact=False,
            shared_default_post_step_exact="not_applicable",
            shared_default_optimizer_exact="not_applicable",
            shared_foreach_false_post_step_exact="not_applicable",
            original_pipeline_all_exact=False,
            legacy_hash_artifact_detected=False,
            optimizer_parameter_id_artifact_detected=False,
        )
        == "cuda_sparse_forward_tensor_nondeterminism"
    )


def test_sparse_backward_classification() -> None:
    assert (
        classify_diagnostic(
            original_pipeline_first_divergence="user_embedding_gradient",
            forward_only_exact=True,
            forward_backward_gradient_exact=False,
            shared_default_post_step_exact="not_applicable",
            shared_default_optimizer_exact="not_applicable",
            shared_foreach_false_post_step_exact="not_applicable",
            original_pipeline_all_exact=False,
            legacy_hash_artifact_detected=False,
            optimizer_parameter_id_artifact_detected=False,
        )
        == "cuda_sparse_backward_gradient_nondeterminism"
    )


def test_default_adam_classification() -> None:
    assert (
        classify_diagnostic(
            original_pipeline_first_divergence="post_step_user_embedding_parameter",
            forward_only_exact=True,
            forward_backward_gradient_exact=True,
            shared_default_post_step_exact=False,
            shared_default_optimizer_exact=False,
            shared_foreach_false_post_step_exact=False,
            original_pipeline_all_exact=False,
            legacy_hash_artifact_detected=False,
            optimizer_parameter_id_artifact_detected=False,
        )
        == "cuda_default_adam_update_nondeterminism"
    )


def test_foreach_adam_classification() -> None:
    assert (
        classify_diagnostic(
            original_pipeline_first_divergence="post_step_user_embedding_parameter",
            forward_only_exact=True,
            forward_backward_gradient_exact=True,
            shared_default_post_step_exact=False,
            shared_default_optimizer_exact=False,
            shared_foreach_false_post_step_exact=True,
            original_pipeline_all_exact=False,
            legacy_hash_artifact_detected=False,
            optimizer_parameter_id_artifact_detected=False,
        )
        == "cuda_foreach_adam_update_nondeterminism"
    )


def test_optimizer_state_only_classification() -> None:
    assert (
        classify_diagnostic(
            original_pipeline_first_divergence="user_step",
            forward_only_exact=True,
            forward_backward_gradient_exact=True,
            shared_default_post_step_exact=True,
            shared_default_optimizer_exact=False,
            shared_foreach_false_post_step_exact=True,
            original_pipeline_all_exact=False,
            legacy_hash_artifact_detected=False,
            optimizer_parameter_id_artifact_detected=False,
        )
        == "cuda_optimizer_state_only_nondeterminism"
    )


def test_unresolved_classification() -> None:
    assert (
        classify_diagnostic(
            original_pipeline_first_divergence="none",
            forward_only_exact="not_applicable",
            forward_backward_gradient_exact="not_applicable",
            shared_default_post_step_exact="not_applicable",
            shared_default_optimizer_exact="not_applicable",
            shared_foreach_false_post_step_exact="not_applicable",
            original_pipeline_all_exact=False,
            legacy_hash_artifact_detected=False,
            optimizer_parameter_id_artifact_detected=False,
        )
        == "root_cause_unresolved"
    )


def test_decision_approves_existing_cuda_only_for_exact_pipeline() -> None:
    assert (
        validate_scale_up_decision(
            classification="no_first_step_divergence_reproduced",
            original_pipeline_all_exact=True,
            forward_exact=True,
            gradients_exact=True,
            default_adam_exact=True,
            foreach_false_adam_exact=True,
            diagnostic_complete=True,
        )
        == "approve_existing_cuda_pipeline_scale_up"
    )


def test_decision_requires_corrected_ml1m_when_foreach_false_resolves() -> None:
    assert (
        validate_scale_up_decision(
            classification="cuda_foreach_adam_update_nondeterminism",
            original_pipeline_all_exact=False,
            forward_exact=True,
            gradients_exact=True,
            default_adam_exact=False,
            foreach_false_adam_exact=True,
            diagnostic_complete=True,
        )
        == "require_corrected_ml1m_rerun_with_foreach_false"
    )


def test_decision_rejects_sparse_backward_divergence() -> None:
    assert (
        validate_scale_up_decision(
            classification="cuda_sparse_backward_gradient_nondeterminism",
            original_pipeline_all_exact=False,
            forward_exact=True,
            gradients_exact=False,
            default_adam_exact="not_applicable",
            foreach_false_adam_exact="not_applicable",
            diagnostic_complete=True,
        )
        == "require_sparse_backward_followup"
    )


def test_decision_rejects_ambiguous_result() -> None:
    assert (
        validate_scale_up_decision(
            classification="root_cause_unresolved",
            original_pipeline_all_exact=False,
            forward_exact="not_applicable",
            gradients_exact="not_applicable",
            default_adam_exact="not_applicable",
            foreach_false_adam_exact="not_applicable",
            diagnostic_complete=True,
        )
        == "do_not_scale_up"
    )

