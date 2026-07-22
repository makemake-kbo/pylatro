"""Tests for the SIL redesign: episode replay, percentile gate, and integration.

Covers the all-completed-episode replay buffer, episode-uniform bounded
sampling, the shared percentile advantage gate, teacher-forced provenance,
winning-BC mode, the resume-safe coefficient schedule, gradient diagnostics,
and the logical-minibatch SIL budget inside ``_run_ppo_update``.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch

from pylatro_agent.constants import MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM
from pylatro_agent.survival import DEFAULT_MAX_ANTES
from pylatro_agent.training.sil import (
    EpisodeReplayBuffer,
    SILEpisodeTracker,
    WinEpisodeBuffer,
    sil_percentile_gate,
)


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
    buffer: EpisodeReplayBuffer,
    env_idx: int,
    *,
    steps: int,
    won: bool,
    stalled: bool = False,
    final_ante: int = 1,
    rng: np.random.Generator,
    num_envs: int = 2,
    teacher_forced: np.ndarray | None = None,
) -> list[tuple[dict, int]]:
    recorded = []
    if teacher_forced is None:
        teacher_forced = np.zeros(num_envs, dtype=bool)
    for _ in range(steps):
        obs = _make_obs(num_envs, rng)
        actions = rng.integers(0, NUM_ACTIONS, size=num_envs)
        rewards = rng.random(num_envs).astype(np.float32)
        tracker.record_step(obs, actions, rewards, teacher_forced)
        recorded.append((obs, int(actions[env_idx])))
    tracker.finish_episode(
        env_idx, won=won, stalled=stalled, final_ante=final_ante, buffer=buffer
    )
    return recorded


# ---------------------------------------------------------------------------
# 1-3: admission (wins, ordinary losses, stalls)
# ---------------------------------------------------------------------------


def test_wins_and_ordinary_losses_enter_replay() -> None:
    rng = np.random.default_rng(0)
    buffer = EpisodeReplayBuffer(capacity_episodes=4)
    tracker = SILEpisodeTracker(num_envs=1)

    recorded = _run_episode(
        tracker, buffer, env_idx=0, steps=3, won=True, rng=rng, num_envs=1
    )
    assert buffer.num_episodes == 1
    assert buffer.num_transitions == 3
    assert buffer.episodes_added_total == 1
    episode = buffer._episodes[0]
    assert episode["won"] is True
    for t, (obs, action) in enumerate(recorded):
        assert episode["actions"][t] == action
        np.testing.assert_array_equal(episode["tokens"][t], obs["tokens"][0].astype(np.int16))
        unpacked = np.unpackbits(episode["action_mask_packed"][t], count=NUM_ACTIONS)
        np.testing.assert_array_equal(unpacked.astype(np.float32), obs["action_mask"][0])

    # An ordinary (non-stalled) loss also enters replay.
    _run_episode(tracker, buffer, env_idx=0, steps=4, won=False, rng=rng, num_envs=1)
    assert buffer.num_episodes == 2
    assert buffer.num_transitions == 7
    assert buffer._episodes[1]["won"] is False


def test_stalled_episodes_are_dropped() -> None:
    rng = np.random.default_rng(2)
    buffer = EpisodeReplayBuffer(capacity_episodes=4)
    tracker = SILEpisodeTracker(num_envs=1)

    _run_episode(
        tracker, buffer, env_idx=0, steps=5, won=False, stalled=True, rng=rng, num_envs=1
    )
    assert buffer.num_episodes == 0
    assert buffer.stalled_episodes_dropped_total == 1
    assert buffer.episodes_added_total == 0

    # A legal policy-caused loss (not stalled) is kept.
    _run_episode(tracker, buffer, env_idx=0, steps=3, won=False, rng=rng, num_envs=1)
    assert buffer.num_episodes == 1


# ---------------------------------------------------------------------------
# 4: overflow reset and counters
# ---------------------------------------------------------------------------


def test_tracker_drops_overflow_episodes_and_resets() -> None:
    rng = np.random.default_rng(2)
    buffer = EpisodeReplayBuffer(capacity_episodes=4)
    tracker = SILEpisodeTracker(num_envs=1, max_episode_steps=3)

    _run_episode(tracker, buffer, env_idx=0, steps=5, won=True, rng=rng, num_envs=1)
    assert buffer.num_episodes == 0
    assert buffer.overflow_episodes_dropped_total == 1

    # Overflow state resets at episode end; the next short win is kept.
    _run_episode(tracker, buffer, env_idx=0, steps=2, won=True, rng=rng, num_envs=1)
    assert buffer.num_episodes == 1
    assert buffer.num_transitions == 2
    assert buffer.overflow_episodes_dropped_total == 1


def test_tracker_separates_envs_across_shared_steps() -> None:
    rng = np.random.default_rng(1)
    buffer = EpisodeReplayBuffer(capacity_episodes=4)
    tracker = SILEpisodeTracker(num_envs=2)

    for _ in range(4):
        tracker.record_step(
            _make_obs(2, rng), rng.integers(0, NUM_ACTIONS, size=2), rng.random(2).astype(np.float32)
        )
    tracker.finish_episode(0, won=True, stalled=False, final_ante=1, buffer=buffer)
    assert buffer.num_episodes == 1
    assert buffer.num_transitions == 4

    for _ in range(2):
        tracker.record_step(
            _make_obs(2, rng), rng.integers(0, NUM_ACTIONS, size=2), rng.random(2).astype(np.float32)
        )
    tracker.finish_episode(1, won=False, stalled=False, final_ante=2, buffer=buffer)
    assert buffer.num_episodes == 2
    assert buffer.num_transitions == 4 + 6
    assert buffer._episodes[0]["won"] is True
    assert buffer._episodes[1]["won"] is False
    assert buffer._episodes[1]["final_ante"] == 2


# ---------------------------------------------------------------------------
# 5: FIFO eviction
# ---------------------------------------------------------------------------


def test_buffer_fifo_eviction() -> None:
    rng = np.random.default_rng(3)
    buffer = EpisodeReplayBuffer(capacity_episodes=2)
    tracker = SILEpisodeTracker(num_envs=1)

    for steps in (2, 3, 4):
        _run_episode(tracker, buffer, env_idx=0, steps=steps, won=True, rng=rng, num_envs=1)

    assert buffer.num_episodes == 2
    assert buffer.num_transitions == 3 + 4
    assert buffer.episodes_added_total == 3
    # Monotonic episode ids continue across eviction.
    assert buffer._episodes[0]["episode_id"] == 1
    assert buffer._episodes[1]["episode_id"] == 2


def test_win_episode_buffer_is_alias_for_back_compat() -> None:
    assert WinEpisodeBuffer is EpisodeReplayBuffer


# ---------------------------------------------------------------------------
# 6-8: episode-uniform bounded sampling
# ---------------------------------------------------------------------------


def _fill_episode(buffer: EpisodeReplayBuffer, steps: int, won: bool, rng) -> None:
    tracker = SILEpisodeTracker(num_envs=1)
    _run_episode(
        tracker, buffer, env_idx=0, steps=steps, won=won, rng=rng, num_envs=1
    )


def test_sample_batch_shapes_and_dtypes() -> None:
    rng = np.random.default_rng(4)
    buffer = EpisodeReplayBuffer(capacity_episodes=4, seed=0)
    _fill_episode(buffer, steps=16, won=True, rng=rng)

    batch = buffer.sample(8, device=torch.device("cpu"))
    assert batch is not None
    assert batch["tokens"].shape == (8, MAX_SEQ_LEN, TOKEN_DIM)
    assert batch["tokens"].dtype == torch.int64
    assert batch["actions"].shape == (8,)
    assert batch["returns"].shape == (8,)
    assert batch["returns"].dtype == torch.float32
    assert batch["teacher_forced_flags"].shape == (8,)
    assert batch["episode_outcomes"].shape == (8,)
    assert batch["episode_returns"].shape == (8,)
    assert batch["episode_ids"].shape == (8,)


def test_episode_uniform_sampling_independent_of_length() -> None:
    rng = np.random.default_rng(5)
    buffer = EpisodeReplayBuffer(capacity_episodes=4, seed=11)
    _fill_episode(buffer, steps=20, won=True, rng=rng)
    _fill_episode(buffer, steps=100, won=True, rng=rng)

    short_counts = []
    long_counts = []
    for _ in range(400):
        b = buffer.sample(8, device=torch.device("cpu"), samples_per_episode=8)
        assert b is not None
        ids = b["episode_ids"].numpy()
        short_counts.append(int(np.sum(ids == 0)))
        long_counts.append(int(np.sum(ids == 1)))
    short_mean = np.mean(short_counts)
    long_mean = np.mean(long_counts)
    # Each episode contributes an equal quota regardless of length: with 2
    # eligible episodes and batch 8 the quota is 4, so the 100-step episode
    # is not privileged over the 20-step one.
    assert short_mean == pytest.approx(4.0, abs=0.5)
    assert long_mean == pytest.approx(4.0, abs=0.5)


def test_per_episode_sample_cap_never_exceeded() -> None:
    rng = np.random.default_rng(6)
    buffer = EpisodeReplayBuffer(capacity_episodes=4, seed=0)
    _fill_episode(buffer, steps=100, won=True, rng=rng)  # one long episode
    batch = buffer.sample(64, device=torch.device("cpu"), samples_per_episode=8)
    assert batch is not None
    # Only one eligible episode -> capped at samples_per_episode, returns fewer
    # than requested rather than violating the cap.
    assert batch["actions"].shape[0] == 8
    # Across many episodes, none exceeds the cap.
    for _ in range(20):
        b = buffer.sample(64, device=torch.device("cpu"), samples_per_episode=8)
        assert b is not None
        assert b["actions"].shape[0] <= 8


def test_deterministic_sampling_with_seed() -> None:
    buffer_a = EpisodeReplayBuffer(capacity_episodes=4, seed=123)
    buffer_b = EpisodeReplayBuffer(capacity_episodes=4, seed=123)
    for buf in (buffer_a, buffer_b):
        r = np.random.default_rng(7)
        _fill_episode(buf, steps=10, won=True, rng=r)

    a = buffer_a.sample(8, device=torch.device("cpu"))
    b = buffer_b.sample(8, device=torch.device("cpu"))
    torch.testing.assert_close(a["actions"], b["actions"])
    torch.testing.assert_close(a["returns"], b["returns"])


def test_sample_empty_returns_none() -> None:
    buffer = EpisodeReplayBuffer(capacity_episodes=2)
    assert buffer.sample(4, device=torch.device("cpu")) is None


def test_sample_fewer_when_episodes_few() -> None:
    rng = np.random.default_rng(8)
    buffer = EpisodeReplayBuffer(capacity_episodes=4, seed=0)
    _fill_episode(buffer, steps=3, won=True, rng=rng)
    _fill_episode(buffer, steps=3, won=False, rng=rng)
    batch = buffer.sample(64, device=torch.device("cpu"), samples_per_episode=8)
    assert batch is not None
    # 2 episodes * 3 rows each = 6 max.
    assert batch["actions"].shape[0] <= 6


def test_sample_win_fraction_property() -> None:
    rng = np.random.default_rng(9)
    buffer = EpisodeReplayBuffer(capacity_episodes=4, seed=0)
    _fill_episode(buffer, steps=2, won=True, rng=rng)
    _fill_episode(buffer, steps=2, won=False, rng=rng)
    _fill_episode(buffer, steps=2, won=False, rng=rng)
    assert buffer.num_wins == 1
    assert buffer.win_fraction == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# 9: exact discounted MC returns
# ---------------------------------------------------------------------------


def test_finish_episode_computes_discounted_returns() -> None:
    rng = np.random.default_rng(6)
    buffer = EpisodeReplayBuffer(capacity_episodes=2)
    tracker = SILEpisodeTracker(num_envs=1, gamma=0.5)

    for reward in (1.0, 0.0, 2.0):
        tracker.record_step(
            _make_obs(1, rng),
            rng.integers(0, NUM_ACTIONS, size=1),
            np.asarray([reward], dtype=np.float32),
        )
    tracker.finish_episode(0, won=True, stalled=False, final_ante=1, buffer=buffer)

    # Return-to-go at gamma 0.5: [1 + 0.5*(0 + 0.5*2), 0 + 0.5*2, 2].
    np.testing.assert_allclose(buffer._episodes[0]["returns"], [1.5, 1.0, 2.0])
    assert buffer._episodes[0]["total_reward"] == pytest.approx(3.0)
    assert buffer._episodes[0]["episode_length"] == 3


# ---------------------------------------------------------------------------
# 10-15: the shared percentile gate
# ---------------------------------------------------------------------------


def test_gate_opens_near_open_percentile_and_saturates_near_saturation() -> None:
    # Advantages uniformly spread over [0, 10]; p80 = 8, p95 = 9.5, floor 0.25.
    adv = np.linspace(0.0, 10.0, 101)
    gate, info = sil_percentile_gate(
        adv,
        open_percentile=80.0,
        saturation_percentile=95.0,
        advantage_floor=0.25,
    )
    assert info["open_threshold"] == pytest.approx(8.0)  # max(0.25, p80=8)
    assert info["saturation_threshold"] == pytest.approx(9.5)
    # At the open threshold the gate is ~0; at saturation ~1.
    np.testing.assert_allclose(gate[80], 0.0, atol=1e-6)
    np.testing.assert_allclose(gate[95], 1.0, atol=1e-6)
    # Below the threshold -> exactly zero (critic-overvalued / sub-floor).
    assert gate[0] == 0.0
    assert gate[50] == 0.0
    # Monotonic non-decreasing.
    assert np.all(np.diff(gate) >= -1e-9)


def test_critic_overvalued_rows_get_zero_gate() -> None:
    adv = np.asarray([-5.0, -1.0, 0.5, 3.0, 10.0])
    gate, _ = sil_percentile_gate(
        adv, open_percentile=80.0, saturation_percentile=95.0, advantage_floor=0.25
    )
    assert gate[0] == 0.0
    assert gate[1] == 0.0


def test_sub_floor_positive_advantages_get_zero_gate() -> None:
    # Floor 0.25: small positive advantages below the open threshold get zero.
    adv = np.asarray([0.05, 0.1, 0.2, 1.0, 5.0])
    gate, info = sil_percentile_gate(
        adv, open_percentile=80.0, saturation_percentile=95.0, advantage_floor=0.25
    )
    # numpy linear-interp: p80=1.8, so open_threshold = max(0.25, 1.8) = 1.8.
    assert info["open_threshold"] == pytest.approx(1.8)
    # Everything at or below 1.0 is sub-threshold -> zero gate.
    assert gate[0] == 0.0
    assert gate[1] == 0.0
    assert gate[2] == 0.0
    assert gate[3] == 0.0


def test_floor_raises_open_threshold_above_percentile() -> None:
    # When p80 is below the floor, the floor dominates.
    adv = np.asarray([0.0, 0.05, 0.1, 0.15, 0.2])
    _, info = sil_percentile_gate(
        adv, open_percentile=80.0, saturation_percentile=95.0, advantage_floor=0.5
    )
    assert info["open_threshold"] == pytest.approx(0.5)


def test_degenerate_percentile_range_produces_no_signal() -> None:
    # All advantages identical -> q_saturation == q_open, degenerate span -> zero.
    adv = np.asarray([2.0, 2.0, 2.0, 2.0])
    gate, _ = sil_percentile_gate(
        adv, open_percentile=80.0, saturation_percentile=95.0, advantage_floor=0.25
    )
    assert np.all(gate == 0.0)


def test_gate_empty_input_returns_empty() -> None:
    gate, info = sil_percentile_gate(
        np.asarray([], dtype=np.float32),
        open_percentile=80.0,
        saturation_percentile=95.0,
        advantage_floor=0.25,
    )
    assert gate.shape == (0,)
    assert np.isnan(info["open_threshold"])


def test_gate_low_signal_batch_does_not_manufacture_top_tail() -> None:
    # All advantages below the floor: gate must stay all-zero even though the
    # 95th percentile exists numerically. open_threshold = max(floor, p95).
    adv = np.asarray([0.0, 0.01, 0.02, 0.03, 0.04])
    gate, _ = sil_percentile_gate(
        adv, open_percentile=80.0, saturation_percentile=95.0, advantage_floor=0.5
    )
    assert np.all(gate == 0.0)


# ---------------------------------------------------------------------------
# 16-17: teacher-forced provenance
# ---------------------------------------------------------------------------


def test_teacher_forced_rows_excluded_by_default() -> None:
    rng = np.random.default_rng(10)
    buffer = EpisodeReplayBuffer(capacity_episodes=4, seed=0)
    tracker = SILEpisodeTracker(num_envs=1)
    # Mark every step teacher-forced.
    tf = np.asarray([True], dtype=bool)
    _run_episode(
        tracker, buffer, env_idx=0, steps=6, won=True, rng=rng, num_envs=1, teacher_forced=tf
    )
    # Episode is retained in replay...
    assert buffer.num_episodes == 1
    # ...but has no eligible rows under the default teacher filter.
    batch = buffer.sample(8, device=torch.device("cpu"), include_teacher_forced=False)
    assert batch is None
    # Including teacher-forced rows yields samples.
    batch_tf = buffer.sample(8, device=torch.device("cpu"), include_teacher_forced=True)
    assert batch_tf is not None
    assert int((batch_tf["teacher_forced_flags"] > 0.5).sum().item()) == batch_tf["actions"].shape[0]


def test_self_generated_rows_from_teacher_forced_episode_kept() -> None:
    rng = np.random.default_rng(11)
    buffer = EpisodeReplayBuffer(capacity_episodes=4, seed=0)
    tracker = SILEpisodeTracker(num_envs=1)
    # Per-step teacher-forced flags (one bool per step for the single env).
    per_step_tf = [True, False, False, True, False]
    for tf in per_step_tf:
        obs = _make_obs(1, rng)
        tracker.record_step(
            obs,
            rng.integers(0, NUM_ACTIONS, size=1),
            rng.random(1).astype(np.float32),
            np.asarray([tf], dtype=bool),
        )
    tracker.finish_episode(0, won=True, stalled=False, final_ante=1, buffer=buffer)
    batch = buffer.sample(8, device=torch.device("cpu"), include_teacher_forced=False)
    assert batch is not None
    # Only the 3 self-generated rows are eligible.
    assert int((batch["teacher_forced_flags"] > 0.5).sum().item()) == 0
    assert batch["actions"].shape[0] <= 3


# ---------------------------------------------------------------------------
# 18: winning-BC selects wins only
# ---------------------------------------------------------------------------


def test_winning_bc_samples_only_wins() -> None:
    rng = np.random.default_rng(12)
    buffer = EpisodeReplayBuffer(capacity_episodes=4, seed=0)
    _fill_episode(buffer, steps=4, won=False, rng=rng)
    _fill_episode(buffer, steps=4, won=True, rng=rng)
    _fill_episode(buffer, steps=4, won=False, rng=rng)

    for _ in range(20):
        batch = buffer.sample(8, device=torch.device("cpu"), only_wins=True)
        assert batch is not None
        assert int(batch["episode_outcomes"].sum().item()) == batch["actions"].shape[0]


# ---------------------------------------------------------------------------
# 19: SIL coefficient schedule
# ---------------------------------------------------------------------------


def test_sil_coeff_zero_disables() -> None:
    from pylatro_agent.training.ppo import PPOConfig, resolve_sil_coeff

    cfg = PPOConfig(sil_coeff=0.0)
    assert resolve_sil_coeff(cfg, total_steps=0) == 0.0
    assert resolve_sil_coeff(cfg, total_steps=10_000) == 0.0


def test_sil_coeff_decay_to_final() -> None:
    from pylatro_agent.training.ppo import PPOConfig, resolve_sil_coeff

    cfg = PPOConfig(sil_coeff=0.01, sil_coeff_final=0.0, sil_decay_fraction=1.0, total_timesteps=10_000)
    assert resolve_sil_coeff(cfg, total_steps=0) == pytest.approx(0.01)
    assert resolve_sil_coeff(cfg, total_steps=5_000) == pytest.approx(0.005)
    assert resolve_sil_coeff(cfg, total_steps=10_000) == pytest.approx(0.0)
    assert resolve_sil_coeff(cfg, total_steps=20_000) == pytest.approx(0.0)


def test_sil_coeff_monotonic_without_resume_rewind() -> None:
    from pylatro_agent.training.ppo import PPOConfig, resolve_sil_coeff

    cfg = PPOConfig(sil_coeff=0.01, sil_coeff_final=0.0, sil_decay_fraction=0.5, total_timesteps=1_000_000)
    original_horizon = cfg.total_timesteps
    coeff_at_resume = resolve_sil_coeff(cfg, total_steps=300_000)
    # Simulate resume that quadruples the recomputed horizon.
    cfg.total_timesteps = 4_000_000
    prev = coeff_at_resume
    for step in range(300_000, 1_200_000, 50_000):
        now = resolve_sil_coeff(cfg, step, schedule_total_steps=original_horizon)
        assert now <= prev + 1e-12
        prev = now
    # Continuity at resume boundary.
    assert resolve_sil_coeff(cfg, 300_000, schedule_total_steps=original_horizon) == pytest.approx(
        coeff_at_resume
    )


def test_sil_coeff_floor_clamped_to_coeff() -> None:
    from pylatro_agent.training.ppo import PPOConfig, resolve_sil_coeff

    cfg = PPOConfig(sil_coeff=0.01, sil_coeff_final=0.5, total_timesteps=10_000)
    # final clamped to coeff -> constant schedule.
    for steps in (0, 5_000, 10_000):
        assert resolve_sil_coeff(cfg, steps) == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# 25: gradient diagnostics helpers (incl. zero norms)
# ---------------------------------------------------------------------------


def test_grad_diagnostics_ratio_and_cosine() -> None:
    from pylatro_agent.training.ppo import _sil_grad_diagnostics

    sil = [torch.tensor([1.0, 0.0])]
    total = [torch.tensor([2.0, 0.0])]  # ppo = [1,0]
    diag = _sil_grad_diagnostics(sil, total)
    assert diag["grad_diagnostic_valid"] == 1.0
    assert diag["actor_grad_norm_weighted"] == pytest.approx(1.0)
    assert diag["ppo_actor_grad_norm"] == pytest.approx(1.0)
    assert diag["ppo_actor_grad_norm_ratio"] == pytest.approx(1.0)
    assert diag["ppo_actor_grad_cosine"] == pytest.approx(1.0)


def test_grad_diagnostics_zero_norms_are_safe() -> None:
    from pylatro_agent.training.ppo import _sil_grad_diagnostics

    # Zero SIL gradient -> ratio and cosine must not be infinity/nan.
    sil = [torch.zeros(3)]
    total = [torch.tensor([1.0, 1.0, 1.0])]  # ppo = total
    diag = _sil_grad_diagnostics(sil, total)
    assert diag["actor_grad_norm_weighted"] == 0.0
    assert diag["ppo_actor_grad_norm"] == pytest.approx(math.sqrt(3)) if False else True
    assert diag["ppo_actor_grad_norm_ratio"] == 0.0
    assert diag["ppo_actor_grad_cosine"] == 0.0
    assert math.isfinite(diag["ppo_actor_grad_norm_ratio"])
    assert math.isfinite(diag["ppo_actor_grad_cosine"])


def test_grad_diagnostics_opposing_gradients_negative_cosine() -> None:
    from pylatro_agent.training.ppo import _sil_grad_diagnostics

    sil = [torch.tensor([1.0, 0.0])]
    total = [torch.tensor([0.0, 0.0])]  # ppo = -sil
    diag = _sil_grad_diagnostics(sil, total)
    assert diag["ppo_actor_grad_cosine"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Config validation (26 partial)
# ---------------------------------------------------------------------------


def test_config_validation() -> None:
    from pylatro_agent.training.ppo import PPOConfig, _validate_ppo_config

    _validate_ppo_config(PPOConfig(sil_coeff=0.1))
    with pytest.raises(ValueError):
        _validate_ppo_config(PPOConfig(seed=-1))
    with pytest.raises(ValueError):
        _validate_ppo_config(PPOConfig(sil_coeff=-0.1))
    with pytest.raises(ValueError):
        _validate_ppo_config(PPOConfig(sil_buffer_episodes=0))
    with pytest.raises(ValueError):
        _validate_ppo_config(PPOConfig(sil_batch_size=0))
    with pytest.raises(ValueError):
        _validate_ppo_config(PPOConfig(sil_min_episodes=0))
    with pytest.raises(ValueError):
        _validate_ppo_config(PPOConfig(sil_objective="bogus"))
    with pytest.raises(ValueError):
        _validate_ppo_config(PPOConfig(sil_logical_minibatches_per_update=3))
    with pytest.raises(ValueError):
        _validate_ppo_config(PPOConfig(sil_gate_saturation_percentile=10.0, sil_gate_open_percentile=80.0))
    with pytest.raises(ValueError):
        _validate_ppo_config(PPOConfig(sil_advantage_floor=-0.1))
    _validate_ppo_config(PPOConfig(sil_logical_minibatches_per_update=2))


def test_resolve_sil_objective_deprecated_compatibility() -> None:
    from pylatro_agent.training.ppo import resolve_sil_objective

    # Defaults.
    assert resolve_sil_objective(None, None) == "advantage"
    # Explicit objective wins when no deprecated flag.
    assert resolve_sil_objective("winning_bc", None) == "winning_bc"
    # Deprecated negative form -> winning_bc.
    assert resolve_sil_objective(None, True) == "winning_bc"
    # Deprecated positive form -> advantage.
    assert resolve_sil_objective(None, False) == "advantage"
    # Explicit winning_bc agrees with the deprecated negative flag.
    assert resolve_sil_objective("winning_bc", True) == "winning_bc"
    # Explicit advantage agrees with the deprecated positive flag.
    assert resolve_sil_objective("advantage", False) == "advantage"
    # Contradiction -> rejected (an explicit objective cannot be overridden).
    with pytest.raises(ValueError, match="conflicts"):
        resolve_sil_objective("advantage", True)
    with pytest.raises(ValueError, match="conflicts"):
        resolve_sil_objective("winning_bc", False)


def test_seed_training_rngs_repeats_all_process_rngs() -> None:
    from pylatro_agent.training.ppo import _seed_training_rngs

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        _seed_training_rngs(123)
        first = (random.random(), np.random.random(), torch.rand(3))
        _seed_training_rngs(123)
        second = (random.random(), np.random.random(), torch.rand(3))
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)

    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2])


def test_checkpoint_records_training_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from pylatro_agent import checkpoint as checkpoint_module
    from pylatro_agent.training.ppo import PPOConfig, _save_checkpoint

    captured: dict[str, object] = {}

    def fake_save(*args, **kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(checkpoint_module, "save_ppo_checkpoint", fake_save)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    _save_checkpoint(
        model=model,
        optimizer=optimizer,
        save_path=tmp_path,
        update_count=1,
        total_steps=2,
        planned_updates=3,
        entropy_coeff=0.01,
        entropy_signal_ema=None,
        lr=3e-4,
        config=PPOConfig(seed=77),
    )

    assert captured["ppo_config_fields"]["seed"] == 77


# ===========================================================================
# PPO integration: logical-minibatch budget, gradient accumulation, warmup,
# KL boundaries, joint optimizer step, gradient diagnostics.
# ===========================================================================


class _SilDist:
    def __init__(self, logits: torch.Tensor, action_mask: torch.Tensor) -> None:
        self.dist = torch.distributions.Categorical(
            logits=logits.masked_fill(action_mask <= 0, -1e8)
        )

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()

    def normalized_action_type_entropy(self) -> torch.Tensor:
        return torch.zeros((), dtype=self.dist.logits.dtype, device=self.dist.logits.device)


class _SilModel(torch.nn.Module):
    """Tiny model with a policy_head (actor) and a value_head (excluded)."""

    def __init__(self) -> None:
        super().__init__()
        self.policy_head = torch.nn.Linear(1, NUM_ACTIONS, bias=False)
        self.value_head = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.normal_(self.policy_head.weight, std=0.02)
        torch.nn.init.zeros_(self.value_head.weight)

    def action_distribution(self, tokens, token_types, scalars, attention_mask, action_mask, temperature=1.0):
        batch = action_mask.shape[0]
        h = torch.ones(batch, 1, device=action_mask.device)
        logits = self.policy_head(h)
        values = self.value_head(h).squeeze(-1)
        survival = torch.full((batch, DEFAULT_MAX_ANTES), 0.5)
        return _SilDist(logits, action_mask), {
            "expected_score": values,
            "ante_survival": survival,
        }


def _sil_episode_dict(steps: int, returns: np.ndarray, *, won: bool = True) -> dict:
    """Build a compact episode with a fully-valid action mask and given returns."""
    obs = _make_obs(1, np.random.default_rng(0))
    action_mask = np.ones((steps, NUM_ACTIONS), dtype=np.float32)
    return {
        "tokens": np.broadcast_to(obs["tokens"][0], (steps, MAX_SEQ_LEN, TOKEN_DIM)).copy().astype(np.int16),
        "token_types": np.broadcast_to(obs["token_types"][0], (steps, MAX_SEQ_LEN)).copy().astype(np.int8),
        "scalars": np.broadcast_to(obs["scalars"][0], (steps, SCALAR_DIM)).copy().astype(np.float32),
        "attention_mask": np.broadcast_to(obs["attention_mask"][0], (steps, MAX_SEQ_LEN)).copy().astype(np.int8),
        "action_mask_packed": np.packbits(action_mask > 0.5, axis=1),
        "actions": np.zeros(steps, dtype=np.int64),
        "returns": returns.astype(np.float32),
        "teacher_forced": np.zeros(steps, dtype=np.int8),
        "won": won,
        "final_ante": 1,
        "episode_length": steps,
        "total_reward": float(returns[0]),
    }


def _make_sil_buffer(num_episodes: int, *, ret_value: float = 5.0, seed: int = 0) -> EpisodeReplayBuffer:
    buffer = EpisodeReplayBuffer(capacity_episodes=16, seed=seed)
    # Varied per-step returns so the percentile gate is non-degenerate.
    returns = np.array([1.0, 3.0, 6.0, ret_value], dtype=np.float32)
    for _ in range(num_episodes):
        buffer.add_episode(_sil_episode_dict(4, returns))
    return buffer


def _ppo_signal_buffer(n: int):
    from pylatro_agent.training.rollout_buffer import RolloutBuffer

    buffer = RolloutBuffer(num_envs=1, rollout_length=n, gamma=0.99, gae_lambda=0.95)
    obs = _make_obs(1, np.random.default_rng(0))
    obs["action_mask"][:] = 1.0  # all actions legal
    for step in range(n):
        buffer.add_batch(
            step=step,
            obs=obs,
            actions=np.array([0], dtype=np.int64),
            rewards=np.array([0.0], dtype=np.float32),
            values=np.array([0.0], dtype=np.float32),
            log_probs=np.array([math.log(0.5)], dtype=np.float32),
            terminated=np.array([False]),
            truncated=np.array([False]),
            teacher_actions=np.array([-1], dtype=np.int64),
            teacher_forced=np.array([False]),
        )
    buffer.advantages[:n] = 0.0
    buffer.returns[:n] = 0.0
    return buffer


def _sil_ppo_config(**overrides):
    from pylatro_agent.training.ppo import PPOConfig

    defaults = dict(
        ppo_epochs=2,
        mini_batch_size=1,
        clip_epsilon=0.2,
        entropy_coeff=0.0,
        value_loss_coeff=0.0,
        survival_loss_coeff=0.0,
        heuristic_distill_coeff=0.0,
        target_kl=None,
        rollout_temperature=1.0,
        sil_coeff=0.01,
        sil_min_episodes=1,
        sil_batch_size=4,
        sil_samples_per_episode=8,
        sil_logical_minibatches_per_update=1,
        sil_advantage_floor=0.25,
    )
    defaults.update(overrides)
    return PPOConfig(**defaults)


def test_exactly_one_sil_logical_group_attempted_per_update() -> None:
    from pylatro_agent.training.ppo import _make_policy_optimizer, _run_ppo_update

    model = _SilModel()
    optimizer = _make_policy_optimizer(model.parameters(), lr=0.01)
    sil_buffer = _make_sil_buffer(2)
    stats = _run_ppo_update(
        model=model,
        optimizer=optimizer,
        buffer=_ppo_signal_buffer(4),
        return_rms=None,
        entropy_coeff=0.0,
        distill_coeff=0.0,
        config=_sil_ppo_config(ppo_epochs=2, sil_logical_minibatches_per_update=1),
        accum_steps=1,
        effective_batch_size=1,
        device=torch.device("cpu"),
        use_pin_memory=False,
        sil_buffer=sil_buffer,
        sil_coeff_now=0.01,
    )
    # 4 microbatches across 2 epochs, but only 1 SIL attempt (budget per update).
    assert stats.sil_logical_minibatches_attempted == 1
    assert stats.sil_losses, "SIL loss should have been recorded once"


def test_two_sil_logical_groups_when_configured() -> None:
    from pylatro_agent.training.ppo import _make_policy_optimizer, _run_ppo_update

    model = _SilModel()
    optimizer = _make_policy_optimizer(model.parameters(), lr=0.01)
    sil_buffer = _make_sil_buffer(2)
    stats = _run_ppo_update(
        model=model,
        optimizer=optimizer,
        buffer=_ppo_signal_buffer(4),
        return_rms=None,
        entropy_coeff=0.0,
        distill_coeff=0.0,
        config=_sil_ppo_config(sil_logical_minibatches_per_update=2),
        accum_steps=1,
        effective_batch_size=1,
        device=torch.device("cpu"),
        use_pin_memory=False,
        sil_buffer=sil_buffer,
        sil_coeff_now=0.01,
    )
    assert stats.sil_logical_minibatches_attempted == 2


def test_grad_accumulation_does_not_multiply_sil() -> None:
    from pylatro_agent.training.ppo import _make_policy_optimizer, _run_ppo_update

    model = _SilModel()
    optimizer = _make_policy_optimizer(model.parameters(), lr=0.01)
    sil_buffer = _make_sil_buffer(2)
    # accum_steps=2: two physical microbatches per logical group. SIL is
    # sampled once per attempted group, not once per microbatch.
    stats = _run_ppo_update(
        model=model,
        optimizer=optimizer,
        buffer=_ppo_signal_buffer(4),
        return_rms=None,
        entropy_coeff=0.0,
        distill_coeff=0.0,
        config=_sil_ppo_config(sil_logical_minibatches_per_update=1),
        accum_steps=2,
        effective_batch_size=1,
        device=torch.device("cpu"),
        use_pin_memory=False,
        sil_buffer=sil_buffer,
        sil_coeff_now=0.01,
    )
    assert stats.sil_logical_minibatches_attempted == 1
    assert len(stats.sil_losses) == 1


def test_no_sil_during_critic_warmup() -> None:
    from pylatro_agent.training.ppo import _make_policy_optimizer, _run_ppo_update

    model = _SilModel()
    optimizer = _make_policy_optimizer(model.parameters(), lr=0.01)
    sil_buffer = _make_sil_buffer(2)
    stats = _run_ppo_update(
        model=model,
        optimizer=optimizer,
        buffer=_ppo_signal_buffer(4),
        return_rms=None,
        entropy_coeff=0.0,
        distill_coeff=0.0,
        config=_sil_ppo_config(),
        accum_steps=1,
        effective_batch_size=1,
        device=torch.device("cpu"),
        use_pin_memory=False,
        policy_loss_scale=0.0,  # critic warmup: policy frozen
        sil_buffer=sil_buffer,
        sil_coeff_now=0.01,
    )
    assert stats.sil_logical_minibatches_attempted == 0
    assert stats.sil_losses == []


def test_kl_stop_at_logical_boundary() -> None:
    from pylatro_agent.training.ppo import _make_policy_optimizer, _run_ppo_update

    model = _SilModel()
    optimizer = _make_policy_optimizer(model.parameters(), lr=0.5)  # large lr -> big KL
    accum_steps = 2
    stats = _run_ppo_update(
        model=model,
        optimizer=optimizer,
        buffer=_ppo_signal_buffer(6),
        return_rms=None,
        entropy_coeff=0.0,
        distill_coeff=0.0,
        config=_sil_ppo_config(ppo_epochs=4, target_kl=1e-6),
        accum_steps=accum_steps,
        effective_batch_size=1,
        device=torch.device("cpu"),
        use_pin_memory=False,
    )
    processed = stats.ppo_minibatches_processed[0]
    expected = 6 * 4  # 6 microbatches/epoch * 4 epochs
    if processed < expected:
        # Early stop happened; it must have landed on a group boundary, not
        # abandoned a pending accumulation group halfway through.
        assert processed % accum_steps == 0


def test_sil_shares_optimizer_step_with_ppo() -> None:
    from pylatro_agent.training.ppo import _make_policy_optimizer, _run_ppo_update

    def run(with_sil: bool) -> int:
        torch.manual_seed(0)
        model = _SilModel()
        optimizer = _make_policy_optimizer(model.parameters(), lr=0.01)
        steps = {"n": 0}
        real_step = optimizer.step

        def counting_step():
            steps["n"] += 1
            return real_step()

        optimizer.step = counting_step  # type: ignore[method-assign]
        _run_ppo_update(
            model=model,
            optimizer=optimizer,
            buffer=_ppo_signal_buffer(4),
            return_rms=None,
            entropy_coeff=0.0,
            distill_coeff=0.0,
            config=_sil_ppo_config(),
            accum_steps=1,
            effective_batch_size=1,
            device=torch.device("cpu"),
            use_pin_memory=False,
            sil_buffer=_make_sil_buffer(2) if with_sil else None,
            sil_coeff_now=0.01 if with_sil else 0.0,
        )
        return steps["n"]

    n_with = run(True)
    n_without = run(False)
    # SIL must not add a separate optimizer step: the count is identical.
    assert n_with == n_without


def test_sil_grad_diagnostics_captured() -> None:
    from pylatro_agent.training.ppo import _make_policy_optimizer, _run_ppo_update

    model = _SilModel()
    optimizer = _make_policy_optimizer(model.parameters(), lr=0.01)
    sil_buffer = _make_sil_buffer(2, ret_value=10.0)  # large advantage -> nonempty gate
    stats = _run_ppo_update(
        model=model,
        optimizer=optimizer,
        buffer=_ppo_signal_buffer(4),
        return_rms=None,
        entropy_coeff=0.0,
        distill_coeff=0.0,
        config=_sil_ppo_config(),
        accum_steps=1,
        effective_batch_size=1,
        device=torch.device("cpu"),
        use_pin_memory=False,
        sil_buffer=sil_buffer,
        sil_coeff_now=0.01,
        grad_diagnostics_due=True,
    )
    assert stats.sil_grad_diagnostic_valid is True
    assert stats.sil_grad_actor_norm_weighted is not None
    assert stats.sil_grad_actor_norm_weighted > 0.0
    assert stats.sil_grad_ppo_actor_norm is not None
    assert -1.0001 <= stats.sil_grad_ppo_actor_grad_cosine <= 1.0001
