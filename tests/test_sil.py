"""Tests for self-imitation win-episode buffering."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pylatro_agent.constants import MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM
from pylatro_agent.training.sil import SILEpisodeTracker, WinEpisodeBuffer


def _make_obs(num_envs: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "tokens": rng.integers(0, 100, size=(num_envs, MAX_SEQ_LEN, TOKEN_DIM), dtype=np.int64),
        "token_types": rng.integers(0, 8, size=(num_envs, MAX_SEQ_LEN), dtype=np.int64),
        "scalars": rng.random((num_envs, SCALAR_DIM)).astype(np.float32),
        "attention_mask": rng.integers(0, 2, size=(num_envs, MAX_SEQ_LEN), dtype=np.int64),
        "action_mask": rng.integers(0, 2, size=(num_envs, NUM_ACTIONS)).astype(np.float32),
    }


def _run_episode(
    tracker: SILEpisodeTracker,
    buffer: WinEpisodeBuffer,
    env_idx: int,
    steps: int,
    won: bool,
    rng: np.random.Generator,
    num_envs: int = 2,
) -> list[tuple[dict, int]]:
    recorded = []
    for _ in range(steps):
        obs = _make_obs(num_envs, rng)
        actions = rng.integers(0, NUM_ACTIONS, size=num_envs)
        rewards = rng.random(num_envs).astype(np.float32)
        tracker.record_step(obs, actions, rewards)
        recorded.append((obs, int(actions[env_idx])))
    tracker.finish_episode(env_idx, won=won, buffer=buffer)
    return recorded


def test_tracker_stores_winning_episode_and_drops_losses() -> None:
    rng = np.random.default_rng(0)
    buffer = WinEpisodeBuffer(capacity_episodes=4)
    tracker = SILEpisodeTracker(num_envs=1)

    _run_episode(tracker, buffer, env_idx=0, steps=5, won=False, rng=rng, num_envs=1)
    assert buffer.num_episodes == 0
    assert buffer.num_transitions == 0

    recorded = _run_episode(tracker, buffer, env_idx=0, steps=3, won=True, rng=rng, num_envs=1)
    assert buffer.num_episodes == 1
    assert buffer.num_transitions == 3

    episode = buffer._episodes[0]
    for t, (obs, action) in enumerate(recorded):
        assert episode["actions"][t] == action
        np.testing.assert_array_equal(episode["tokens"][t], obs["tokens"][0].astype(np.int16))
        unpacked = np.unpackbits(episode["action_mask_packed"][t], count=NUM_ACTIONS)
        np.testing.assert_array_equal(unpacked.astype(np.float32), obs["action_mask"][0])


def test_tracker_separates_envs_across_shared_steps() -> None:
    # Both envs record every step; finishing env 0 must not consume env 1's
    # transitions, and env 1's episode keeps accumulating afterwards.
    rng = np.random.default_rng(1)
    buffer = WinEpisodeBuffer(capacity_episodes=4)
    tracker = SILEpisodeTracker(num_envs=2)

    for _ in range(4):
        tracker.record_step(
            _make_obs(2, rng), rng.integers(0, NUM_ACTIONS, size=2), rng.random(2).astype(np.float32)
        )
    tracker.finish_episode(0, won=True, buffer=buffer)
    assert buffer.num_episodes == 1
    assert buffer.num_transitions == 4

    for _ in range(2):
        tracker.record_step(
            _make_obs(2, rng), rng.integers(0, NUM_ACTIONS, size=2), rng.random(2).astype(np.float32)
        )
    tracker.finish_episode(1, won=True, buffer=buffer)
    assert buffer.num_episodes == 2
    assert buffer.num_transitions == 4 + 6


def test_tracker_drops_overflow_episodes() -> None:
    rng = np.random.default_rng(2)
    buffer = WinEpisodeBuffer(capacity_episodes=4)
    tracker = SILEpisodeTracker(num_envs=1, max_episode_steps=3)

    _run_episode(tracker, buffer, env_idx=0, steps=5, won=True, rng=rng, num_envs=1)
    assert buffer.num_episodes == 0

    # Overflow state resets at episode end; the next short win is kept.
    _run_episode(tracker, buffer, env_idx=0, steps=2, won=True, rng=rng, num_envs=1)
    assert buffer.num_episodes == 1
    assert buffer.num_transitions == 2


def test_buffer_fifo_eviction() -> None:
    rng = np.random.default_rng(3)
    buffer = WinEpisodeBuffer(capacity_episodes=2)
    tracker = SILEpisodeTracker(num_envs=1)

    for steps in (2, 3, 4):
        _run_episode(tracker, buffer, env_idx=0, steps=steps, won=True, rng=rng, num_envs=1)

    assert buffer.num_episodes == 2
    assert buffer.num_transitions == 3 + 4
    assert buffer.episodes_added_total == 3


def test_buffer_sample_batch_shapes_and_dtypes() -> None:
    rng = np.random.default_rng(4)
    buffer = WinEpisodeBuffer(capacity_episodes=4, seed=0)
    tracker = SILEpisodeTracker(num_envs=1)
    _run_episode(tracker, buffer, env_idx=0, steps=4, won=True, rng=rng, num_envs=1)

    batch = buffer.sample(8, device=torch.device("cpu"))
    assert batch["tokens"].shape == (8, MAX_SEQ_LEN, TOKEN_DIM)
    assert batch["tokens"].dtype == torch.int64
    assert batch["token_types"].shape == (8, MAX_SEQ_LEN)
    assert batch["scalars"].shape == (8, SCALAR_DIM)
    assert batch["scalars"].dtype == torch.float32
    assert batch["attention_mask"].shape == (8, MAX_SEQ_LEN)
    assert batch["action_mask"].shape == (8, NUM_ACTIONS)
    assert batch["action_mask"].dtype == torch.float32
    assert set(batch["action_mask"].unique().tolist()) <= {0.0, 1.0}
    assert batch["actions"].shape == (8,)
    assert batch["actions"].dtype == torch.int64
    assert batch["returns"].shape == (8,)
    assert batch["returns"].dtype == torch.float32


def test_finish_episode_computes_discounted_returns() -> None:
    rng = np.random.default_rng(6)
    buffer = WinEpisodeBuffer(capacity_episodes=2)
    tracker = SILEpisodeTracker(num_envs=1, gamma=0.5)

    for reward in (1.0, 0.0, 2.0):
        tracker.record_step(
            _make_obs(1, rng),
            rng.integers(0, NUM_ACTIONS, size=1),
            np.asarray([reward], dtype=np.float32),
        )
    tracker.finish_episode(0, won=True, buffer=buffer)

    # Return-to-go at gamma 0.5: [1 + 0.5*(0 + 0.5*2), 0 + 0.5*2, 2].
    np.testing.assert_allclose(buffer._episodes[0]["returns"], [1.5, 1.0, 2.0])


def test_buffer_sample_empty_raises() -> None:
    buffer = WinEpisodeBuffer(capacity_episodes=2)
    with pytest.raises(ValueError):
        buffer.sample(4, device=torch.device("cpu"))


def test_config_validation() -> None:
    from pylatro_agent.training.ppo import PPOConfig, _validate_ppo_config

    _validate_ppo_config(PPOConfig(sil_coeff=0.1))
    with pytest.raises(ValueError):
        _validate_ppo_config(PPOConfig(sil_coeff=-0.1))
    with pytest.raises(ValueError):
        _validate_ppo_config(PPOConfig(sil_buffer_episodes=0))
    with pytest.raises(ValueError):
        _validate_ppo_config(PPOConfig(sil_batch_size=0))
    with pytest.raises(ValueError):
        _validate_ppo_config(PPOConfig(sil_min_episodes=0))


def test_sample_sil_loss_gating() -> None:
    """_sample_sil_loss returns None when disabled or under-filled (no forward runs)."""
    from pylatro_agent.training.ppo import PPOConfig, _sample_sil_loss

    rng = np.random.default_rng(5)
    buffer = WinEpisodeBuffer(capacity_episodes=4, seed=0)
    tracker = SILEpisodeTracker(num_envs=1)
    _run_episode(tracker, buffer, env_idx=0, steps=1, won=True, rng=rng, num_envs=1)

    # The gating paths return before any model call, so no model is needed.
    device = torch.device("cpu")
    off = PPOConfig(sil_coeff=0.0, sil_min_episodes=1)
    assert _sample_sil_loss(None, None, off, device) is None
    assert _sample_sil_loss(None, buffer, off, device) is None

    underfilled = PPOConfig(sil_coeff=0.2, sil_min_episodes=2)
    assert _sample_sil_loss(None, buffer, underfilled, device) is None


def test_sample_sil_loss_advantage_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    """The NLL weight is min((R - V)+ / clip, 1): zero when the critic already
    predicts the buffered return, full at a shortfall of sil_advantage_clip."""
    from pylatro_agent.training import ppo as ppo_module
    from pylatro_agent.training.ppo import PPOConfig, _sample_sil_loss

    rng = np.random.default_rng(7)
    buffer = WinEpisodeBuffer(capacity_episodes=4, seed=0)
    tracker = SILEpisodeTracker(num_envs=1, gamma=1.0)
    _run_episode(tracker, buffer, env_idx=0, steps=4, won=True, rng=rng, num_envs=1)

    class _ConstDist:
        """log_prob is a constant -1.0, so the ungated NLL mean is exactly 1.0."""

        def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
            return torch.full((actions.shape[0],), -1.0)

    shortfall = {"value": 0.0}

    def fake_grammar(model, batch, temperature):
        values = batch["returns"] - shortfall["value"]
        return _ConstDist(), {"expected_score": values}

    monkeypatch.setattr(ppo_module, "_grammar_distribution", fake_grammar)
    device = torch.device("cpu")
    config = PPOConfig(sil_coeff=0.2, sil_min_episodes=1, sil_advantage_clip=3.0)

    # Critic already predicts every buffered return: the gate closes fully.
    loss, advantage_mean, gate_mean = _sample_sil_loss(None, buffer, config, device)
    assert loss.item() == pytest.approx(0.0)
    assert advantage_mean == pytest.approx(0.0)
    assert gate_mean == pytest.approx(0.0)

    # Shortfall of exactly the clip: full behavior-cloning weight.
    shortfall["value"] = 3.0
    loss, advantage_mean, gate_mean = _sample_sil_loss(None, buffer, config, device)
    assert loss.item() == pytest.approx(1.0)
    assert advantage_mean == pytest.approx(3.0)
    assert gate_mean == pytest.approx(1.0)

    # Half the clip: half weight.
    shortfall["value"] = 1.5
    loss, advantage_mean, gate_mean = _sample_sil_loss(None, buffer, config, device)
    assert loss.item() == pytest.approx(0.5)
    assert advantage_mean == pytest.approx(1.5)
    assert gate_mean == pytest.approx(0.5)

    # Gating disabled: plain NLL regardless of the critic, no metrics.
    ungated = PPOConfig(
        sil_coeff=0.2, sil_min_episodes=1, sil_advantage_gating=False
    )
    shortfall["value"] = 0.0
    loss, advantage_mean, gate_mean = _sample_sil_loss(None, buffer, ungated, device)
    assert loss.item() == pytest.approx(1.0)
    assert advantage_mean is None
    assert gate_mean is None
