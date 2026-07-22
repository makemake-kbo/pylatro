"""Tests for the HL-Gauss categorical value head and advantage clipping."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pylatro_agent.constants import MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM
from pylatro_agent.training.rollout_buffer import RolloutBuffer
from pylatro_agent.value_head import HL_GAUSS_SIGMA_RATIO, ValueHead, hl_gauss_projection


def _make_head(bins: int = 51, v_min: float = -8.0, v_max: float = 12.0) -> ValueHead:
    return ValueHead(d_model=32, value_bins=bins, value_v_min=v_min, value_v_max=v_max)


def _forward(head: ValueHead, batch: int = 4, seq: int = 6) -> dict[str, torch.Tensor]:
    backbone_out = torch.randn(batch, seq, 32)
    padding_mask = torch.ones(batch, seq)
    return head(backbone_out, padding_mask)


class TestHlGaussProjection:
    def test_rows_are_distributions(self) -> None:
        head = _make_head()
        targets = torch.tensor([-8.0, -3.7, 0.0, 6.42, 12.0])
        probs = hl_gauss_projection(targets, head.bin_edges, head.hl_gauss_sigma)
        assert probs.shape == (5, 51)
        assert torch.all(probs >= 0)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-6)

    def test_expectation_recovers_target(self) -> None:
        head = _make_head()
        # Away from the edges the Gaussian smear is symmetric, so the
        # projected histogram's mean must match the scalar target.
        targets = torch.linspace(-6.0, 10.0, 33)
        probs = hl_gauss_projection(targets, head.bin_edges, head.hl_gauss_sigma)
        means = (probs * head.bin_centers).sum(dim=-1)
        assert torch.allclose(means, targets, atol=1e-3)

    def test_out_of_range_target_degrades_to_edge_bin(self) -> None:
        head = _make_head()
        probs = hl_gauss_projection(
            torch.tensor([-50.0, 50.0]), head.bin_edges, head.hl_gauss_sigma
        )
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-6)
        assert probs[0].argmax().item() == 0
        assert probs[1].argmax().item() == 50

    def test_smear_spans_multiple_bins(self) -> None:
        head = _make_head()
        probs = hl_gauss_projection(
            torch.tensor([2.0]), head.bin_edges, head.hl_gauss_sigma
        )
        # sigma = 0.75 * bin width should spread meaningful mass over ~3 bins,
        # the dense-gradient property HL-Gauss has over two-hot.
        assert int((probs[0] > 0.05).sum()) >= 3


class TestCategoricalValueHead:
    def test_scalar_mode_unchanged(self) -> None:
        head = ValueHead(d_model=32)
        out = _forward(head)
        assert out["expected_score"].shape == (4,)
        assert "expected_score_logits" not in out
        assert head.expected_score.weight.shape == (1, 32)

    def test_scalar_parameter_order_preserves_optimizer_resume_layout(self) -> None:
        names = [name for name, _ in ValueHead(d_model=32).named_parameters()]
        assert names.index("expected_score.weight") < names.index("ante_survival.weight")

    def test_categorical_outputs(self) -> None:
        head = _make_head()
        out = _forward(head)
        assert out["expected_score"].shape == (4,)
        assert out["expected_score_logits"].shape == (4, 51)
        # The histogram mean is bounded by the atom grid by construction.
        assert torch.all(out["expected_score"] >= -8.0)
        assert torch.all(out["expected_score"] <= 12.0)

    def test_scalar_matches_logit_histogram_mean(self) -> None:
        head = _make_head()
        out = _forward(head)
        manual = (out["expected_score_logits"].softmax(-1) * head.bin_centers).sum(-1)
        assert torch.allclose(out["expected_score"], manual, atol=1e-6)

    def test_sigma_ratio(self) -> None:
        head = _make_head()
        width = 20.0 / 50
        assert head.hl_gauss_sigma == pytest.approx(HL_GAUSS_SIGMA_RATIO * width)

    def test_bin_buffers_not_in_state_dict(self) -> None:
        # Derived from config; persisting them would block scalar->categorical
        # checkpoint loads on a spurious missing-buffer mismatch.
        head = _make_head()
        assert "bin_centers" not in head.state_dict()
        assert "bin_edges" not in head.state_dict()

    def test_scalar_checkpoint_loads_with_head_skipped(self) -> None:
        scalar_sd = ValueHead(d_model=32).state_dict()
        cat = _make_head()
        model_sd = cat.state_dict()
        compatible = {
            k: v for k, v in scalar_sd.items() if k in model_sd and model_sd[k].shape == v.shape
        }
        skipped = set(scalar_sd) - set(compatible)
        # Only the expected_score tensors differ; everything else migrates.
        assert skipped == {"expected_score.weight", "expected_score.bias"}
        cat.load_state_dict(compatible, strict=False)

    def test_invalid_configs_raise(self) -> None:
        with pytest.raises(ValueError):
            ValueHead(d_model=32, value_bins=1)
        with pytest.raises(ValueError):
            ValueHead(d_model=32, value_bins=51, value_v_min=5.0, value_v_max=-5.0)

    def test_cross_entropy_gradient_is_bounded(self) -> None:
        # The HL-Gauss selling point: a wildly wrong prediction produces a
        # bounded logit gradient (softmax - target), unlike MSE whose gradient
        # grows with the error.
        head = _make_head()
        logits = torch.zeros(1, 51, requires_grad=True)
        target = hl_gauss_projection(
            torch.tensor([11.5]), head.bin_edges, head.hl_gauss_sigma
        )
        loss = -(target * torch.log_softmax(logits, dim=-1)).sum()
        loss.backward()
        assert logits.grad is not None
        assert torch.all(logits.grad.abs() <= 1.0)


def _dummy_obs(num_envs: int) -> dict[str, np.ndarray]:
    return {
        "tokens": np.zeros((num_envs, MAX_SEQ_LEN, TOKEN_DIM), dtype=np.int16),
        "token_types": np.zeros((num_envs, MAX_SEQ_LEN), dtype=np.int8),
        "scalars": np.zeros((num_envs, SCALAR_DIM), dtype=np.float32),
        "attention_mask": np.ones((num_envs, MAX_SEQ_LEN), dtype=np.int8),
        "action_mask": np.ones((num_envs, NUM_ACTIONS), dtype=np.float32),
    }


class TestAdvantageClip:
    def _filled_buffer(self, rewards: list[float]) -> RolloutBuffer:
        buffer = RolloutBuffer(
            num_envs=1, rollout_length=len(rewards), gamma=0.99, gae_lambda=0.95
        )
        for step, reward in enumerate(rewards):
            buffer.add_batch(
                step=step,
                obs=_dummy_obs(num_envs=1),
                actions=np.array([0], dtype=np.int64),
                rewards=np.array([reward], dtype=np.float32),
                values=np.array([0.0], dtype=np.float32),
                log_probs=np.array([0.0], dtype=np.float32),
                terminated=np.array([step == len(rewards) - 1]),
                truncated=np.array([False]),
            )
        buffer.compute_returns_and_advantages(last_values=np.array([0.0]))
        return buffer

    def test_clip_clamps_outliers(self) -> None:
        # One huge terminal among near-zero shaping -> a >2 sigma outlier.
        buffer = self._filled_buffer([0.01] * 30 + [10.0])
        buffer.normalize_advantages(clip_sigma=2.0)
        assert float(np.abs(buffer.advantages).max()) <= 2.0 + 1e-6

    def test_clip_disabled_by_default(self) -> None:
        buffer = self._filled_buffer([0.01] * 30 + [10.0])
        buffer.normalize_advantages()
        assert float(np.abs(buffer.advantages).max()) > 2.0

    def test_clip_leaves_bulk_untouched(self) -> None:
        rewards = [0.01] * 30 + [10.0]
        clipped = self._filled_buffer(rewards)
        plain = self._filled_buffer(rewards)
        clipped.normalize_advantages(clip_sigma=4.0)
        plain.normalize_advantages()
        inliers = np.abs(plain.advantages) <= 4.0
        np.testing.assert_allclose(
            clipped.advantages[inliers.flatten()], plain.advantages[inliers.flatten()]
        )


def test_ppo_config_rejects_negative_advantage_clip() -> None:
    from pylatro_agent.training.ppo import PPOConfig, _validate_ppo_config

    with pytest.raises(ValueError, match="advantage_clip_sigma"):
        _validate_ppo_config(PPOConfig(advantage_clip_sigma=-1.0))
