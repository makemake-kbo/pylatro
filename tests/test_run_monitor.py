from __future__ import annotations

import math

from pylatro_agent.training.run_monitor import MonitorPolicy, eta_seconds, quality_stop_reason


def points(values, start=1, stride=1):
    return [(float(i * 60), start + i * stride, v) for i, v in enumerate(values)]


def test_sparse_wins_and_diagnostic_nan_are_not_failure():
    series = {"eval/win_rate": points([0] * 100, stride=10), "ppo/explained_variance": points([math.nan])}
    assert quality_stop_reason("ppo", series, MonitorPolicy()) is None


def test_nonfinite_core_metric_stops():
    assert quality_stop_reason("ppo", {"ppo/policy_loss": points([math.inf])}, MonitorPolicy())


def test_eval_regression_needs_three_distinct_consecutive_evals():
    p = MonitorPolicy()
    assert quality_stop_reason("ppo", {"eval/win_rate": points([0.04, 0.01, 0.01], stride=10)}, p) is None
    assert quality_stop_reason("ppo", {"eval/win_rate": points([0.04, 0.01, 0.01, 0.01], stride=10)}, p)
    assert quality_stop_reason("ppo", {"eval/win_rate": points([0.04, 0.01, 0.03, 0.01], stride=10)}, p) is None
    assert quality_stop_reason("ppo", {"eval/win_rate": [(0, 10, 0.04), (1, 20, 0.01)] * 3}, p) is None
    assert quality_stop_reason("ppo", {"eval/win_rate": points([0.01, 0, 0, 0], stride=10)}, p) is None


def test_stalls_and_kl_require_consecutive_updates():
    p = MonitorPolicy()
    assert quality_stop_reason("ppo", {"ppo/approx_kl": points([0.3, 0.3])}, p) is None
    assert quality_stop_reason("ppo", {"ppo/approx_kl": points([0.3] * 3)}, p)
    assert quality_stop_reason("ppo", {"ppo/approx_kl": points([0.3] * 3, stride=2)}, p) is None
    assert quality_stop_reason("ppo", {"recent_100/stall_rate": points([0.9] * 10)}, p) is None
    assert quality_stop_reason("ppo", {"recent_100/stall_rate": points([0.9] * 10, start=20)}, p)


def test_pt_needs_epoch_loss_and_accuracy_regression():
    p = MonitorPolicy()
    series = {"epoch/loss": points([2, 3.5, 4]), "epoch/accuracy": points([0.6, 0.4, 0.4])}
    assert quality_stop_reason("supervised", series, p)
    series["epoch/accuracy"] = points([0.6, 0.65, 0.7])
    assert quality_stop_reason("supervised", series, p) is None


def test_eta_uses_observed_wall_clock_and_finished_target():
    assert eta_seconds([], 1000) is None
    assert eta_seconds([(0, 10, 0), (100, 20, 0)], 30) == 100
    assert eta_seconds([(0, 10, 0), (100, 20, 0)], 20) == 0


def test_ppo_stop_request_saves_full_checkpoint_at_update_boundary(tmp_path, monkeypatch):
    import torch

    from pylatro_agent.agent import AgentConfig
    from pylatro_agent.archive import ArchiveConfig
    from pylatro_agent.reward import RewardConfig
    from pylatro_agent.training import ppo
    from pylatro_agent.training.ppo_config import PPOConfig

    request = tmp_path / "stop.json"
    request.write_text('{"reason": "test"}')

    def unexpected_eval(*args, **kwargs):
        raise AssertionError("stop-requested update must not start another evaluation")

    monkeypatch.setattr(ppo, "evaluate_model", unexpected_eval)
    ppo.train_ppo(
        PPOConfig(
            num_envs=1,
            rollout_length=4,
            total_updates=3,
            ppo_epochs=1,
            mini_batch_size=4,
            micro_batch_size=4,
            async_envs=False,
            eval_interval=1,
            eval_games=1,
            checkpoint_interval=100,
            device="cpu",
            stop_request_path=str(request),
            archive_config=ArchiveConfig(capacity_per_bucket=1),
            reward_config=RewardConfig(objective="milestone"),
            win_ante=8,
            save_dir=str(tmp_path / "checkpoints"),
            log_dir=str(tmp_path / "events"),
        ),
        agent_config=AgentConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, dropout=0),
    )
    checkpoint = torch.load(tmp_path / "checkpoints/ppo_monitor_stop.pt", weights_only=False)
    assert checkpoint["update_count"] == 1
    assert checkpoint["total_steps"] == 4
    assert checkpoint["monitor_stop_requested"] is True
    assert "optimizer_state_dict" in checkpoint and "archive_states" in checkpoint
    assert (tmp_path / "checkpoints/ppo_latest.pt").is_file()
