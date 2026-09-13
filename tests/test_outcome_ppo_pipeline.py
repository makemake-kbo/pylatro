import json
from dataclasses import asdict

import pytest
import torch

from pylatro import load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.checkpoint import load_checkpoint_payload
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.outcome_study import outcome_reward
from pylatro_agent.training import ppo
from pylatro_agent.training.ppo_checkpoint import _save_checkpoint
from pylatro_agent.training.ppo_config import PPOConfig
from pylatro_agent.training.ppo_evaluation import validate_resume_evaluation_seeds
from pylatro_agent.vocab import build_vocab


def test_ppo_excludes_entire_evaluation_panel_and_confirmation(monkeypatch, tmp_path):
    config = PPOConfig(
        num_envs=1,
        rollout_length=2,
        total_updates=1,
        eval_games=1,
        eval_sampled_games=0,
        eval_seeds=[190000, 190001],
        device="cpu",
        save_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
    )
    config.training_excluded_seeds = [200000, 200001]

    class ReachedEnvironment(Exception):
        pass

    def inspect(*args, **kwargs):
        assert set(kwargs["excluded_seeds"]) == {190000, 190001, 200000, 200001}
        raise ReachedEnvironment

    monkeypatch.setattr(ppo, "_make_vectorized_envs", inspect)
    with pytest.raises(ReachedEnvironment):
        ppo.train_ppo(config, AgentConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32), data=load_game_data())


def test_resume_rejects_changed_reservations_outside_evaluated_prefix():
    config = PPOConfig(eval_games=1, eval_seeds=[190000, 190001])
    config.training_excluded_seeds = [200000]
    saved = asdict(config)
    saved["training_excluded_seeds"] = [200000]
    config.training_excluded_seeds = [200001]
    with pytest.raises(ValueError, match="reservation"):
        validate_resume_evaluation_seeds(config, saved)


def test_full_checkpoint_preserves_deck_and_complete_reservations(tmp_path):
    config = PPOConfig(deck_key="b_blue", eval_games=1, eval_seeds=[190000, 190001], training_excluded_seeds=[200000])
    model = BalatroAgent(AgentConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32), build_vocab(load_game_data()))
    path = _save_checkpoint(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        save_path=tmp_path,
        update_count=0,
        total_steps=0,
        planned_updates=1,
        entropy_coeff=0.001,
        entropy_signal_ema=None,
        lr=0.001,
        agent_config=model.config,
        config=config,
    )
    saved = load_checkpoint_payload(path, "cpu")["ppo_config_fields"]
    assert saved.get("deck_key") == "b_blue"
    assert saved.get("training_excluded_seeds") == [200000]
    validate_resume_evaluation_seeds(config, saved)


def test_combined_outcome_ppo_runs_and_resumes_without_teacher(monkeypatch, tmp_path):
    def reject_teacher(*args, **kwargs):
        raise AssertionError("Outcome-only PPO called a teacher")

    monkeypatch.setattr(HeuristicAgent, "select_action", reject_teacher)
    config = PPOConfig(
        num_envs=2,
        rollout_length=16,
        total_updates=2,
        ppo_epochs=1,
        mini_batch_size=8,
        micro_batch_size=4,
        eval_games=1,
        eval_sampled_games=0,
        eval_seeds=[190000, 190001],
        training_excluded_seeds=[200000],
        eval_before_training=False,
        eval_workers=0,
        eval_cpu_threads=1,
        max_no_progress_steps=4,
        async_envs=False,
        reward_config=outcome_reward(),
        gamma=1,
        gae_lambda=1,
        deck_key="b_blue",
        hand_ar_mixture_eps=0.5,
        terminal_replay_updates_per_ppo_update=0,
        return_path_coeff=0,
        save_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
        risk_forecast_log=False,
        training_episode_log=True,
    )
    architecture = AgentConfig(
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        dropout=0,
        hand_ar_mixture_eps=0.5,
        action_head_variant="entity_sale_subset_residual_v1",
    )
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        ppo.train_ppo(config, architecture, data=load_game_data())
        latest = tmp_path / "checkpoints/ppo_latest.pt"
        first = load_checkpoint_payload(latest, "cpu")
        assert first["update_count"] == 2
        assert first["agent_config"]["win_only_value"] is True
        assert first["reward_config"] == asdict(outcome_reward())
        assert first["training_reserved_seeds"] == [190000, 190001, 200000]
        config.total_updates = 3
        ppo.train_ppo(config, architecture, resume_path=str(latest), data=load_game_data())
        saved = load_checkpoint_payload(latest, "cpu")
        assert saved["update_count"] == 3
        assert all(torch.isfinite(tensor).all() for tensor in saved["state_dict"].values())
        episodes = [json.loads(line) for line in (tmp_path / "logs/training_episodes.jsonl").read_text().splitlines()]
        assert episodes
        assert all(row["seed"] not in {190000, 190001, 200000} for row in episodes)
        assert all(row["reward"] == 10 * row["won"] for row in episodes)
        assert all(row["deck"] == "b_blue" for row in episodes)
    finally:
        torch.set_num_threads(threads)
