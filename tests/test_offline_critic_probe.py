from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

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


def test_target_stats_report_hl_support_clipping() -> None:
    stats = probe._target_stats(np.asarray([-9.0, 0.0, 13.0, 14.0]))
    assert stats["below_hl_support_fraction"] == pytest.approx(0.25)
    assert stats["above_hl_support_fraction"] == pytest.approx(0.5)
