from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.analysis.plan_e_lightgcn_determinism_diagnostic import (
    aggregate_batch_losses,
    canonical_float,
    classify_diagnostic,
    cross_device_metric_row,
    first_divergent_index,
    float_hex,
    hash_batch_losses,
    hash_int_array,
    optimizer_state_sha256,
    relative_difference,
    rounded_equal,
    tensor_state_sha256,
    ulp_distance,
    validate_scale_up_decision,
)


def test_canonical_float_serialization_and_hex_are_stable() -> None:
    assert canonical_float(0.1) == "0.10000000000000001"
    assert float_hex(0.5) == "0x1.0000000000000p-1"
    with pytest.raises(ValueError):
        canonical_float(float("nan"))


def test_ulp_distance_and_exact_equality_detection() -> None:
    left = 1.0
    right = np.nextafter(1.0, 2.0)
    assert ulp_distance(left, right) == 1
    assert ulp_distance(left, left) == 0
    assert float_hex(left) != float_hex(right)


def test_relative_difference_handles_zero() -> None:
    assert relative_difference(0.0, 0.0) == 0.0
    assert relative_difference(0.0, 2.0) == 1.0


def test_rounded_equality_at_configured_precisions() -> None:
    assert rounded_equal(1.000000004, 1.0, 8)
    assert not rounded_equal(1.000000004, 1.0, 10)


def test_loss_aggregation_methods_and_weighted_mean() -> None:
    losses = [0.1, 0.2, 0.3]
    sizes = [2, 2, 6]
    aggregates = aggregate_batch_losses(losses, sizes)
    assert aggregates.batch_count == 3
    assert aggregates.training_example_count == 10
    assert aggregates.python_sum_loss == pytest.approx(sum(losses))
    assert aggregates.math_fsum_loss == pytest.approx(math.fsum(losses))
    assert aggregates.numpy_float64_sum_loss == pytest.approx(np.asarray(losses, dtype=np.float64).sum())
    assert aggregates.torch_float64_sum_loss == pytest.approx(torch.as_tensor(losses, dtype=torch.float64).sum().item())
    assert aggregates.unweighted_batch_mean == pytest.approx(0.2)
    assert aggregates.batch_size_weighted_mean == pytest.approx(0.24)


def test_batch_loss_sequence_hash_uses_canonical_serialization() -> None:
    assert hash_batch_losses([0.1, 0.2]) == hash_batch_losses([float("0.1"), float("0.2")])
    assert hash_batch_losses([0.1, 0.2]) != hash_batch_losses([0.1, 0.20000000000000004])


def test_batch_input_hash_preserves_order_and_values() -> None:
    assert hash_int_array(np.asarray([1, 2, 3], dtype=np.int64)) == hash_int_array([1, 2, 3])
    assert hash_int_array([1, 2, 3]) != hash_int_array([3, 2, 1])


def test_optimizer_state_hash_changes_after_step() -> None:
    model = torch.nn.Embedding(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = optimizer_state_sha256(optimizer)
    loss = model(torch.tensor([0, 1])).sum()
    loss.backward()
    optimizer.step()
    after = optimizer_state_sha256(optimizer)
    assert before != after
    assert after == optimizer_state_sha256(optimizer)


def test_model_state_step_comparison_hashes() -> None:
    model = torch.nn.Linear(2, 1)
    before = tensor_state_sha256(model.state_dict())
    with torch.no_grad():
        model.weight.add_(1.0)
    after = tensor_state_sha256(model.state_dict())
    assert before != after


def test_first_divergent_batch_detection() -> None:
    assert first_divergent_index(["a", "b"], ["a", "c"]) == 1
    assert first_divergent_index(["a"], ["a"]) is None
    assert first_divergent_index(["a"], ["a", "b"]) == 1


def test_exact_cpu_diagnostic_classification() -> None:
    classification = classify_diagnostic(
        cuda_state_exact=True,
        cuda_optimizer_exact=True,
        cuda_loss_exact=True,
        phase3m_epoch_loss_exact=False,
        cpu_repeat_exact=True,
        cpu_final_state_exact=True,
        cpu_embeddings_exact=True,
        cpu_scores_exact=True,
        cpu_rankings_exact=True,
        phase3m_final_outputs_exact=True,
    )
    assert classification == "cpu_exact_determinism_cuda_logging_warning"


def test_cuda_logging_only_classification() -> None:
    classification = classify_diagnostic(
        cuda_state_exact=True,
        cuda_optimizer_exact=True,
        cuda_loss_exact=False,
        phase3m_epoch_loss_exact=True,
        cpu_repeat_exact=True,
        cpu_final_state_exact=True,
        cpu_embeddings_exact=True,
        cpu_scores_exact=True,
        cpu_rankings_exact=True,
        phase3m_final_outputs_exact=True,
    )
    assert classification == "cpu_exact_determinism_cuda_logging_warning"


def test_true_model_divergence_classification() -> None:
    classification = classify_diagnostic(
        cuda_state_exact=False,
        cuda_optimizer_exact=True,
        cuda_loss_exact=False,
        phase3m_epoch_loss_exact=False,
        cpu_repeat_exact=True,
        cpu_final_state_exact=True,
        cpu_embeddings_exact=True,
        cpu_scores_exact=True,
        cpu_rankings_exact=True,
        phase3m_final_outputs_exact=True,
    )
    assert classification == "cuda_sparse_training_nondeterministic"


def test_cpu_determinism_failure_classification() -> None:
    classification = classify_diagnostic(
        cuda_state_exact=True,
        cuda_optimizer_exact=True,
        cuda_loss_exact=True,
        phase3m_epoch_loss_exact=True,
        cpu_repeat_exact=False,
        cpu_final_state_exact=False,
        cpu_embeddings_exact=False,
        cpu_scores_exact=False,
        cpu_rankings_exact=False,
        phase3m_final_outputs_exact=True,
    )
    assert classification == "cpu_determinism_failed"


def test_scale_up_approval_rules_require_tolerance() -> None:
    with pytest.raises(ValueError):
        validate_scale_up_decision(
            classification="cpu_exact_determinism_cuda_logging_warning",
            phase3m_final_outputs_exact=True,
            cuda_state_exact=True,
            cuda_optimizer_exact=True,
            cpu_repeat_exact=True,
            no_error_failures=True,
            tolerance="",
        )
    decision, device = validate_scale_up_decision(
        classification="cpu_exact_determinism_cuda_logging_warning",
        phase3m_final_outputs_exact=True,
        cuda_state_exact=True,
        cuda_optimizer_exact=True,
        cpu_repeat_exact=True,
        no_error_failures=True,
        tolerance="1e-8 absolute",
    )
    assert decision == "approve_cuda_scale_up_with_documented_loss_telemetry_tolerance"
    assert device == "cuda"


def test_rejection_of_ambiguous_decision() -> None:
    with pytest.raises(ValueError):
        validate_scale_up_decision(
            classification="not_a_classification",
            phase3m_final_outputs_exact=True,
            cuda_state_exact=True,
            cuda_optimizer_exact=True,
            cpu_repeat_exact=True,
            no_error_failures=True,
            tolerance="1e-8",
        )


def test_cross_device_equality_is_not_expected() -> None:
    row = cross_device_metric_row(
        comparison_scope="state",
        metric="model-state hash equality",
        cuda_value="",
        cpu_value="",
        equality_expected="false",
        status="not_applicable",
        interpretation="state hashes are device-specific",
    )
    assert row["status"] == "not_applicable"
    assert row["equality_expected"] == "false"


def test_source_frames_not_mutated_by_aggregation() -> None:
    frame = pd.DataFrame({"loss": [0.1, 0.2], "size": [2, 2]})
    before = frame.copy(deep=True)
    aggregate_batch_losses(frame["loss"].tolist(), frame["size"].tolist())
    pd.testing.assert_frame_equal(frame, before)
