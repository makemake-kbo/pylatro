"""Regression tests for PPO rollout bookkeeping."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pylatro_agent.constants import MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM
from pylatro_agent.training.rollout_buffer import RolloutBuffer


def _dummy_obs(num_envs: int) -> dict[str, np.ndarray]:
    return {
        "tokens": np.zeros((num_envs, MAX_SEQ_LEN, TOKEN_DIM), dtype=np.int16),
        "token_types": np.zeros((num_envs, MAX_SEQ_LEN), dtype=np.int8),
        "scalars": np.zeros((num_envs, SCALAR_DIM), dtype=np.float32),
        "attention_mask": np.ones((num_envs, MAX_SEQ_LEN), dtype=np.int8),
        "action_mask": np.ones((num_envs, NUM_ACTIONS), dtype=np.float32),
    }


def test_rollout_buffer_long_horizon_defaults() -> None:
    buffer = RolloutBuffer(num_envs=1, rollout_length=1)
    assert buffer.gamma == 0.997
    assert buffer.gae_lambda == 0.97


def test_rollout_buffer_bootstraps_across_truncation() -> None:
    buffer = RolloutBuffer(num_envs=1, rollout_length=1, gamma=0.99, gae_lambda=0.95)
    buffer.add_batch(
        step=0,
        obs=_dummy_obs(num_envs=1),
        actions=np.array([0], dtype=np.int64),
        rewards=np.array([1.0], dtype=np.float32),
        values=np.array([0.2], dtype=np.float32),
        log_probs=np.array([0.0], dtype=np.float32),
        terminated=np.array([False]),
        truncated=np.array([True]),
        bootstrap_values=np.array([0.5], dtype=np.float32),
    )

    buffer.compute_returns_and_advantages(last_values=np.array([0.5], dtype=np.float32))

    np.testing.assert_allclose(buffer.advantages[0], 1.295, rtol=1e-6)
    np.testing.assert_allclose(buffer.returns[0], 1.495, rtol=1e-6)


def test_rollout_buffer_stops_bootstrap_on_termination() -> None:
    buffer = RolloutBuffer(num_envs=1, rollout_length=1, gamma=0.99, gae_lambda=0.95)
    buffer.add_batch(
        step=0,
        obs=_dummy_obs(num_envs=1),
        actions=np.array([0], dtype=np.int64),
        rewards=np.array([1.0], dtype=np.float32),
        values=np.array([0.2], dtype=np.float32),
        log_probs=np.array([0.0], dtype=np.float32),
        terminated=np.array([True]),
        truncated=np.array([False]),
    )

    buffer.compute_returns_and_advantages(last_values=np.array([0.5], dtype=np.float32))

    np.testing.assert_allclose(buffer.advantages[0], 0.8, rtol=1e-6)
    np.testing.assert_allclose(buffer.returns[0], 1.0, rtol=1e-6)


def test_rollout_buffer_labels_completed_episode_outcomes() -> None:
    buffer = RolloutBuffer(num_envs=1, rollout_length=3)

    buffer.set_episode_outcome(
        env_idx=0, start_step=1, end_step=2, won=True, final_ante=4
    )

    np.testing.assert_array_equal(buffer.terminal_outcome_targets, [0, 8, 8])
    np.testing.assert_array_equal(buffer.terminal_outcome_masks, [0.0, 1.0, 1.0])

    loss_buffer = RolloutBuffer(num_envs=1, rollout_length=2)
    loss_buffer.set_episode_outcome(
        env_idx=0, start_step=0, end_step=1, won=False, final_ante=3
    )
    np.testing.assert_array_equal(loss_buffer.terminal_outcome_targets, [2, 2])
    np.testing.assert_array_equal(loss_buffer.terminal_outcome_masks, [1.0, 1.0])

    stalled_buffer = RolloutBuffer(num_envs=1, rollout_length=2)
    np.testing.assert_array_equal(stalled_buffer.terminal_outcome_masks, [0.0, 0.0])


def test_rollout_buffer_truncation_does_not_leak_gae_across_episode_boundary() -> None:
    buffer = RolloutBuffer(num_envs=1, rollout_length=3, gamma=0.99, gae_lambda=0.95)
    buffer.add_batch(
        step=0,
        obs=_dummy_obs(num_envs=1),
        actions=np.array([0], dtype=np.int64),
        rewards=np.array([0.0], dtype=np.float32),
        values=np.array([0.1], dtype=np.float32),
        log_probs=np.array([0.0], dtype=np.float32),
        terminated=np.array([False]),
        truncated=np.array([False]),
    )
    buffer.add_batch(
        step=1,
        obs=_dummy_obs(num_envs=1),
        actions=np.array([0], dtype=np.int64),
        rewards=np.array([1.0], dtype=np.float32),
        values=np.array([0.2], dtype=np.float32),
        log_probs=np.array([0.0], dtype=np.float32),
        terminated=np.array([False]),
        truncated=np.array([True]),
        bootstrap_values=np.array([0.5], dtype=np.float32),
    )
    # This transition belongs to the next episode and must not influence the
    # truncated step's advantage.
    buffer.add_batch(
        step=2,
        obs=_dummy_obs(num_envs=1),
        actions=np.array([0], dtype=np.int64),
        rewards=np.array([0.0], dtype=np.float32),
        values=np.array([10.0], dtype=np.float32),
        log_probs=np.array([0.0], dtype=np.float32),
        terminated=np.array([False]),
        truncated=np.array([False]),
    )

    buffer.compute_returns_and_advantages(last_values=np.array([0.0], dtype=np.float32))

    np.testing.assert_allclose(buffer.advantages[0], 1.3159475, rtol=1e-6)
    np.testing.assert_allclose(buffer.advantages[1], 1.295, rtol=1e-6)


def test_microbatches_preserve_exact_logical_batch_and_partial_weight() -> None:
    buffer = RolloutBuffer(num_envs=1, rollout_length=352)
    buffer._step_counts[0] = 352

    batches = buffer.get_batches(
        352,
        torch.device("cpu"),
        micro_batch_size=64,
    )

    assert [batch["actions"].numel() for batch in batches] == [64, 64, 64, 64, 64, 32]
    assert [bool(batch["_logical_group_start"]) for batch in batches] == [True, False, False, False, False, False]
    assert [bool(batch["_logical_group_end"]) for batch in batches] == [False, False, False, False, False, True]
    assert sum(float(batch["_loss_weight"]) for batch in batches) == pytest.approx(1.0)
    assert float(batches[-1]["_loss_weight"]) == pytest.approx(32 / 352)
