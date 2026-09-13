import copy
from dataclasses import asdict

import pytest
import torch
from torch import nn

from pylatro import load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.checkpoint import load_checkpoint_payload, save_checkpoint
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.outcome_study import actor_digest, outcome_reward
from pylatro_agent.reward import reward_checkpoint_metadata
from pylatro_agent.training import ppo
from pylatro_agent.training.ppo_config import PPOConfig
from pylatro_agent.vocab import build_vocab
from tools.outcome_critic_intervention import COMPLETE_CRITIC_PROTOCOL, complete_episode_critic_updates
from tools.train_outcome_ppo import same_state_tree


def test_actor_step_preserves_nonzero_critic_adam_moments_and_restores_scope(monkeypatch):
    model = nn.Module()
    model.actor = nn.Linear(2, 2)
    model.value_head = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()
    before = copy.deepcopy(model.value_head.state_dict())
    moments = {name: copy.deepcopy(optimizer.state[parameter])
               for name, parameter in model.value_head.named_parameters()}

    def step(**kwargs):
        assert kwargs["config"].outcome_loss_coeff == 0
        assert not any(parameter.requires_grad for parameter in model.value_head.parameters())
        optimizer.zero_grad()
        (model.actor(torch.ones(2, 2)).square().mean() + 0 * model.value_head(torch.ones(2, 2)).sum()).backward()
        optimizer.step()

    monkeypatch.setattr(ppo, "_run_ppo_update", step)
    original_save = ppo._save_checkpoint
    config = PPOConfig(value_loss_coeff=0., gamma=1., outcome_loss_coeff=.1)
    with complete_episode_critic_updates():
        ppo._run_ppo_update(model=model, optimizer=optimizer, config=config)
    assert config.outcome_loss_coeff == .1
    assert same_state_tree(before, model.value_head.state_dict())
    for name, parameter in model.value_head.named_parameters():
        assert same_state_tree(moments[name], optimizer.state[parameter])
        assert parameter.requires_grad
    assert ppo._run_ppo_update is step and ppo._save_checkpoint is original_save
    with pytest.raises(ValueError, match="gamma one"), complete_episode_critic_updates():
        ppo._run_ppo_update(model=model, config=PPOConfig(gamma=.99))
    assert ppo._run_ppo_update is step and ppo._save_checkpoint is original_save


def test_real_ppo_updates_record_complete_critic_protocol_and_keep_actor_teacher_free(monkeypatch, tmp_path):
    def reject_teacher(*args, **kwargs):
        raise AssertionError("PPO requested a teacher action")

    monkeypatch.setattr(HeuristicAgent, "select_action", reject_teacher)
    architecture = AgentConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32, dropout=0,
                               hand_ar_mixture_eps=.5, win_only_value=True,
                               action_head_variant="entity_sale_subset_residual_v1")
    initial = BalatroAgent(architecture, build_vocab(load_game_data()))
    for parameter in initial.value_head.return_residual.parameters():
        parameter.data.zero_()
    source = tmp_path / "initial.pt"
    save_checkpoint(initial, source, extra=dict(agent_config=asdict(architecture),
                                               **reward_checkpoint_metadata(outcome_reward())))
    config = PPOConfig(
        device="cpu", num_envs=2, rollout_length=32, total_updates=2, ppo_epochs=1,
        mini_batch_size=8, micro_batch_size=4, eval_games=1, eval_sampled_games=0,
        eval_seeds=[190000], training_excluded_seeds=[200000], eval_before_training=False,
        eval_workers=0, eval_cpu_threads=1, max_no_progress_steps=16, async_envs=False,
        reward_config=outcome_reward(), gamma=1., gae_lambda=1., deck_key="b_blue",
        hand_ar_mixture_eps=.5, value_loss_coeff=0, outcome_loss_coeff=.1,
        terminal_replay_updates_per_ppo_update=30, terminal_replay_batch_size=256,
        terminal_replay_min_episodes=1, terminal_replay_row_uniform=True,
        return_path_coeff=0, save_dir=str(tmp_path / "checkpoints"), log_dir=str(tmp_path / "logs"),
        risk_forecast_log=False,
    )
    real_update = ppo._run_ppo_update
    checked = []

    def inspect(**kwargs):
        model, optimizer = kwargs["model"], kwargs["optimizer"]
        before = copy.deepcopy(model.value_head.state_dict())
        moments = {name: copy.deepcopy(optimizer.state.get(parameter, {}))
                   for name, parameter in model.value_head.named_parameters()}
        result = real_update(**kwargs)
        assert same_state_tree(before, model.value_head.state_dict())
        assert all(same_state_tree(moments[name], optimizer.state.get(parameter, {}))
                   for name, parameter in model.value_head.named_parameters())
        checked.append(True)
        return result

    monkeypatch.setattr(ppo, "_run_ppo_update", inspect)
    with complete_episode_critic_updates():
        ppo.train_ppo(config, architecture, pretrained_path=str(source), data=load_game_data())
    assert len(checked) == 2
    saved = load_checkpoint_payload(tmp_path / "checkpoints/ppo_latest.pt", "cpu")
    assert saved["outcome_study_critic_protocol"] == COMPLETE_CRITIC_PROTOCOL
    final = BalatroAgent(architecture, build_vocab(load_game_data()))
    final.load_state_dict(saved["state_dict"])
    assert actor_digest(final) != actor_digest(initial)
    assert all(torch.count_nonzero(parameter) == 0 for parameter in final.value_head.return_residual.parameters())
    assert ppo._run_ppo_update is inspect
