from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pylatro_agent.reward import RewardConfig, reward_checkpoint_metadata

MODULE_PATH = Path(__file__).parents[1] / "tools" / "offline_critic_probe.py"
SPEC = importlib.util.spec_from_file_location("offline_critic_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_discounted_returns() -> None:
    result = probe._discounted_returns([1.0, 2.0, 3.0], gamma=0.5)
    np.testing.assert_allclose(result, [2.75, 3.5, 3.0])


def test_lambda_returns_terminal_episode() -> None:
    rewards = np.asarray([1.0, 2.0], dtype=np.float32)
    values = np.asarray([0.5, 0.25], dtype=np.float32)
    episode_ids = np.asarray([0, 0], dtype=np.int64)
    result = probe._lambda_returns(rewards, values, episode_ids, gamma=0.9, gae_lambda=1.0)
    np.testing.assert_allclose(result, [2.8, 2.0], rtol=1e-6)


def test_split_episodes_keeps_whole_episodes_and_both_outcomes() -> None:
    episode_ids = np.repeat(np.arange(8), 3)
    episode_wins = {index: index >= 4 for index in range(8)}
    train, validation = probe._split_episodes(
        episode_ids, episode_wins, validation_fraction=0.25, seed=4
    )
    train_episodes = set(episode_ids[train])
    validation_episodes = set(episode_ids[validation])
    assert train_episodes.isdisjoint(validation_episodes)
    assert {episode_wins[index] for index in validation_episodes} == {False, True}


def test_sil_gate_metrics_use_shared_training_gate() -> None:
    metrics = probe._sil_gate_metrics(
        mc_returns=np.asarray([0.0, 1.0, 2.0, 4.0]),
        predictions=np.zeros(4),
        wins=np.ones(4, dtype=np.bool_),
        episode_ids=np.asarray([0, 0, 0, 0]),
        open_percentile=80.0,
        saturation_percentile=95.0,
        advantage_floor=0.25,
    )
    # raw advantages [0,1,2,4]; p80=2.8 -> open_threshold 2.8; only adv 4 gates.
    agg = metrics["aggregate"]
    assert agg["advantage_mean"] == pytest.approx(1.75)
    assert agg["open_threshold"] == pytest.approx(2.8)
    assert agg["gate_mean"] == pytest.approx(0.25)
    assert agg["gate_positive_fraction"] == pytest.approx(0.25)
    assert agg["gate_saturation_fraction"] == pytest.approx(0.25)
    assert agg["noise_floor_rejected_fraction"] == pytest.approx(0.75)
    assert metrics["win"]["states"] == 4
    assert metrics["loss"]["states"] == 0


def test_sil_gate_metrics_win_and_loss_subsets() -> None:
    metrics = probe._sil_gate_metrics(
        mc_returns=np.asarray([0.0, 5.0, 1.0, 6.0]),
        predictions=np.zeros(4),
        wins=np.asarray([True, True, False, False]),
        episode_ids=np.asarray([0, 0, 1, 1]),
        open_percentile=80.0,
        saturation_percentile=95.0,
        advantage_floor=0.25,
    )
    assert metrics["aggregate"]["states"] == 4
    assert metrics["win"]["states"] == 2
    assert metrics["loss"]["states"] == 2
    # Gate mass from wins fraction is computed over the aggregate gate.
    assert "gate_weight_from_wins_fraction" in metrics
    assert metrics["unique_episodes_with_gate_mass"] >= 1


def test_sil_gate_metrics_degenerate_range_no_signal() -> None:
    metrics = probe._sil_gate_metrics(
        mc_returns=np.asarray([2.0, 2.0, 2.0, 2.0]),
        predictions=np.zeros(4),
        wins=np.ones(4, dtype=np.bool_),
        episode_ids=np.asarray([0, 0, 0, 0]),
        open_percentile=80.0,
        saturation_percentile=95.0,
        advantage_floor=0.25,
    )
    assert metrics["aggregate"]["gate_mean"] == pytest.approx(0.0)
    assert metrics["aggregate"]["gate_positive_fraction"] == pytest.approx(0.0)


def test_target_stats_report_return_distribution() -> None:
    stats = probe._target_stats(np.asarray([-9.0, 0.0, 13.0, 14.0]))
    assert stats["min"] == -9.0
    assert stats["p50"] == pytest.approx(6.5)
    assert stats["max"] == 14.0


def test_checkpoint_reward_config_restores_all_shaping_settings() -> None:
    expected = RewardConfig(
        gamma=0.991,
        potential_win_ante=4,
        dense_reward_scale=0.25,
        enable_planet_match_rewards=True,
        planet_unmatched_use_penalty_coeff=0.4,
        enable_score_build_potential=True,
    )
    payload = reward_checkpoint_metadata(expected)

    actual = probe._checkpoint_reward_config(payload, checkpoint="probe.pt")

    assert actual == expected


def test_checkpoint_reward_config_rejects_task_changing_overrides() -> None:
    payload = reward_checkpoint_metadata(
        RewardConfig(gamma=0.991, potential_win_ante=4, dense_reward_scale=0.25)
    )

    with pytest.raises(ValueError, match=r"--gamma=.*does not match"):
        probe._checkpoint_reward_config(payload, gamma_override=0.997)
    with pytest.raises(ValueError, match=r"--win-ante=.*does not match"):
        probe._checkpoint_reward_config(payload, win_ante_override=8)


def test_checkpoint_reward_config_rejects_missing_metadata() -> None:
    with pytest.raises(RuntimeError, match="complete reward_config"):
        probe._checkpoint_reward_config({})


def test_critic_metrics_detect_calibration_skill_and_positive_return_ev() -> None:
    probabilities = np.full((4, 9), 0.0, dtype=np.float32)
    probabilities[0, [0, 8]] = [0.9, 0.1]
    probabilities[1, [0, 8]] = [0.8, 0.2]
    probabilities[2, [0, 8]] = [0.2, 0.8]
    probabilities[3, [0, 8]] = [0.1, 0.9]
    returns = np.asarray([-5.0, -4.0, 8.0, 10.0], dtype=np.float32)
    predictions = returns + np.asarray([0.2, -0.1, 0.1, -0.2], dtype=np.float32)
    terminal_values = np.asarray([-4.5, -4.5, 9.0, 9.0], dtype=np.float32)
    dataset = {
        "outcome_probabilities": probabilities,
        "outcome_targets": np.asarray([0, 0, 8, 8], dtype=np.int64),
        "mc_returns": returns,
        "expected_returns": predictions,
        "terminal_values": terminal_values,
        "return_residuals": predictions - terminal_values,
    }

    metrics = probe._critic_metrics(dataset, np.arange(4))

    assert metrics["outcome_brier"] < metrics["outcome_climatology_brier"]
    assert metrics["derived_win_brier"] < metrics["derived_win_climatology_brier"]
    assert metrics["derived_win_brier_skill"] > 0.0
    assert metrics["explained_variance"] > 0.0
