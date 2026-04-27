"""Regression tests for PPO rollout bookkeeping."""

from __future__ import annotations

import numpy as np

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


def test_teacher_actions_default_sentinel() -> None:
    buffer = RolloutBuffer(num_envs=2, rollout_length=2)
    # Default fill is -1 across the whole buffer.
    assert (buffer.teacher_actions == -1).all()


def test_teacher_actions_round_trip_through_add_batch() -> None:
    import torch

    buffer = RolloutBuffer(num_envs=2, rollout_length=1, gamma=0.99, gae_lambda=0.95)
    teacher = np.array([7, 13], dtype=np.int64)
    buffer.add_batch(
        step=0,
        obs=_dummy_obs(num_envs=2),
        actions=np.array([0, 0], dtype=np.int64),
        rewards=np.array([0.0, 0.0], dtype=np.float32),
        values=np.array([0.0, 0.0], dtype=np.float32),
        log_probs=np.array([0.0, 0.0], dtype=np.float32),
        terminated=np.array([False, False]),
        truncated=np.array([False, False]),
        teacher_actions=teacher,
    )

    # teacher_actions stored at the per-env flat indices
    assert buffer.teacher_actions[0] == 7
    assert buffer.teacher_actions[1] == 13

    buffer.compute_returns_and_advantages(last_values=np.array([0.0, 0.0], dtype=np.float32))
    batches = buffer.get_batches(batch_size=4, device=torch.device("cpu"))
    assert len(batches) == 1
    teacher_tensor = batches[0]["teacher_actions"]
    assert teacher_tensor.dtype == torch.int64
    # Order may be shuffled; just check the multiset.
    assert sorted(teacher_tensor.tolist()) == [7, 13]


def test_teacher_actions_omitted_defaults_to_sentinel() -> None:
    buffer = RolloutBuffer(num_envs=2, rollout_length=1, gamma=0.99, gae_lambda=0.95)
    buffer.add_batch(
        step=0,
        obs=_dummy_obs(num_envs=2),
        actions=np.array([0, 0], dtype=np.int64),
        rewards=np.array([0.0, 0.0], dtype=np.float32),
        values=np.array([0.0, 0.0], dtype=np.float32),
        log_probs=np.array([0.0, 0.0], dtype=np.float32),
        terminated=np.array([False, False]),
        truncated=np.array([False, False]),
    )
    assert (buffer.teacher_actions[:2] == -1).all()


def test_distill_weights_default_to_one() -> None:
    buffer = RolloutBuffer(num_envs=2, rollout_length=2)
    assert (buffer.distill_weights == 1.0).all()


def test_distill_weights_round_trip_through_add_batch() -> None:
    import torch

    buffer = RolloutBuffer(num_envs=2, rollout_length=1, gamma=0.99, gae_lambda=0.95)
    weights = np.array([1.0, 3.0], dtype=np.float32)
    buffer.add_batch(
        step=0,
        obs=_dummy_obs(num_envs=2),
        actions=np.array([0, 0], dtype=np.int64),
        rewards=np.array([0.0, 0.0], dtype=np.float32),
        values=np.array([0.0, 0.0], dtype=np.float32),
        log_probs=np.array([0.0, 0.0], dtype=np.float32),
        terminated=np.array([False, False]),
        truncated=np.array([False, False]),
        distill_weights=weights,
    )

    assert buffer.distill_weights[0] == 1.0
    assert buffer.distill_weights[1] == 3.0

    buffer.compute_returns_and_advantages(last_values=np.array([0.0, 0.0], dtype=np.float32))
    batches = buffer.get_batches(batch_size=4, device=torch.device("cpu"))
    assert len(batches) == 1
    weight_tensor = batches[0]["distill_weights"]
    assert weight_tensor.dtype == torch.float32
    assert sorted(weight_tensor.tolist()) == [1.0, 3.0]
