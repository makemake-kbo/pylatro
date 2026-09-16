"""Checkpoint selection and strict resume use the current completed update."""

from dataclasses import replace

import pytest
import torch

from pylatro import load_game_data
from pylatro_agent.agent import AgentConfig
from pylatro_agent.training import ppo


@pytest.mark.parametrize("adaptive_entropy", [False, True])
def test_regression_stop_preserves_best_checkpoint_and_resumes_current_state(
    tmp_path, monkeypatch, adaptive_entropy,
):
    evaluations = []

    def evaluate(*args, update_count, **kwargs):
        evaluations.append(update_count)
        return 0.75 if update_count == 1 else 0.25

    monkeypatch.setattr(ppo, "evaluate_model", evaluate)
    config = ppo.PPOConfig(
        num_envs=1, rollout_length=4, total_updates=5, ppo_epochs=1,
        mini_batch_size=4, micro_batch_size=3, async_envs=False,
        hand_ar_mixture_eps=0, adaptive_entropy=adaptive_entropy,
        terminal_replay_updates_per_ppo_update=0,
        eval_games=1, eval_interval=1, checkpoint_interval=1,
        eval_regression_tolerance=0.1, eval_regression_patience=1,
        risk_forecast_log=False,
        save_dir=str(tmp_path / "checkpoints"), log_dir=str(tmp_path / "logs"),
    )
    architecture = AgentConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32, dropout=0)

    def load(name):
        return torch.load(tmp_path / "checkpoints" / name, weights_only=False)

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        ppo.train_ppo(config, architecture, data=load_game_data())
        best = load("ppo_best_eval.pt")
        stopped = load("ppo_regression_stop.pt")
        latest = load("ppo_latest.pt")
        assert evaluations == [0, 1, 2]
        assert load("ppo_update1.pt")["best_eval_win_rate"] is None
        assert best["update_count"] == best["best_eval_update"] == 1
        assert best["best_eval_win_rate"] == 0.75
        for saved in (stopped, latest):
            assert saved["update_count"] == 2
            assert saved["total_steps"] == 8
            assert saved["planned_updates"] == 5
            assert saved["best_eval_update"] == 1
            assert saved["best_eval_win_rate"] == 0.75
            assert saved["eval_regression_streak"] == 1
            assert saved["schedule_total_steps"] == 20
            for key, value in stopped["state_dict"].items():
                torch.testing.assert_close(saved["state_dict"][key], value, rtol=0, atol=0)

        # A relative resume target starts from the stopped update, keeps the
        # original annealing horizon, and does not repeat baseline evaluation.
        ppo.train_ppo(
            replace(config), replace(architecture), data=load_game_data(),
            resume_path=str(tmp_path / "checkpoints" / "ppo_latest.pt"), additional_updates=1,
        )
        resumed = load("ppo_latest.pt")
        assert evaluations == [0, 1, 2, 3]
        assert resumed["update_count"] == resumed["planned_updates"] == 3
        assert resumed["total_steps"] == 12
        assert resumed["schedule_total_steps"] == 20
        assert resumed["best_eval_update"] == 1
        assert load("ppo_best_eval.pt")["update_count"] == 1
        if adaptive_entropy:
            assert resumed["alpha_optimizer_state_dict"]["state"][0]["step"].item() == 3
    finally:
        torch.set_num_threads(threads)
