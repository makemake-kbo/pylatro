"""Packed diagnostic transfers preserve buckets, scores, and singleton rules."""

import pytest
import torch

from pylatro_agent.training.ppo_optimization import _outcome_metric_inputs_cpu, _outcome_metric_rows


@pytest.mark.parametrize("rows", [0, 1, 7, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_packed_metrics_match_reference(rows, dtype):
    generator = torch.Generator().manual_seed(19)
    probabilities = torch.softmax(torch.randn(rows, 9, generator=generator, dtype=dtype), dim=-1)
    values = {"outcome_probabilities": probabilities, "win_prob": probabilities[:, -1]}
    sampled = {
        "terminal_outcome_target": torch.arange(rows) % 9,
        "current_antes": torch.arange(rows) % 8 + 1,
        "cross_rollout_flags": (torch.arange(rows) % 3 == 0).float(),
    }
    expected = _outcome_metric_rows(values, sampled)
    host_values, host_sampled = _outcome_metric_inputs_cpu(values, sampled)
    actual = _outcome_metric_rows(host_values, host_sampled)
    assert actual.keys() == expected.keys()
    for key, (expected_values, expected_mask) in expected.items():
        actual_values, actual_mask = actual[key]
        torch.testing.assert_close(actual_values, expected_values, atol=0, rtol=0)
        if expected_mask is None:
            assert actual_mask is None
        else:
            torch.testing.assert_close(actual_mask, expected_mask)
        assert actual_values.device.type == "cpu"
        assert not actual_values.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_metric_buckets_are_computed_on_cpu():
    probabilities = torch.softmax(torch.randn(32, 9, device="cuda"), dim=-1)
    values = {"outcome_probabilities": probabilities, "win_prob": probabilities[:, -1]}
    sampled = {
        "terminal_outcome_target": torch.arange(32, device="cuda") % 9,
        "current_antes": torch.arange(32, device="cuda") % 8 + 1,
        "cross_rollout_flags": torch.zeros(32, device="cuda"),
    }
    expected = _outcome_metric_rows(
        {k: v.cpu() for k, v in values.items()},
        {k: v.cpu() for k, v in sampled.items()},
    )
    actual = _outcome_metric_rows(values, sampled)
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key][0].device.type == "cpu"
        torch.testing.assert_close(actual[key][0], expected[key][0])


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="requires CUDA",
            ),
        ),
    ],
)
def test_rollout_transfer_preserves_statistics_and_cpu_views(device):
    import numpy as np

    from pylatro_agent.training.ppo_rollout import _rollout_statistics_numpy

    tensors = tuple(torch.arange(6, dtype=torch.float32, device=device) / (index + 1) for index in range(4))
    survival = torch.arange(48, dtype=torch.float32, device=device).reshape(6, 8) / 48
    expected = [tensor.cpu().numpy() for tensor in (*tensors, survival)]
    actual = _rollout_statistics_numpy(*tensors, survival)
    for reference, result in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(result, reference)
        assert result.dtype == reference.dtype
        if device == "cpu":
            assert np.shares_memory(reference, result)


def test_packing_normalizes_bucket_fields_before_float_conversion():
    probabilities = torch.full((2, 9), 1 / 9)
    values = {"outcome_probabilities": probabilities, "win_prob": probabilities[:, -1]}
    sampled = {
        "terminal_outcome_target": torch.tensor([0, 1]),
        "current_antes": torch.tensor([1.999999999, 2.0], dtype=torch.float64),
        "cross_rollout_flags": torch.tensor([0.50000000001, 0.5], dtype=torch.float64),
    }
    expected = _outcome_metric_rows(values, sampled)
    actual = _outcome_metric_rows(*_outcome_metric_inputs_cpu(values, sampled))
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key][0], expected[key][0], atol=0, rtol=0)
        if expected[key][1] is not None:
            torch.testing.assert_close(actual[key][1], expected[key][1])
