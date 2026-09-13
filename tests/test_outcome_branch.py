import copy
import json
from dataclasses import asdict, replace

import pytest
import torch

from pylatro import load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.checkpoint import load_checkpoint_payload, load_ppo_resume_payload
from pylatro_agent.outcome_study import outcome_reward
from pylatro_agent.training.ppo_checkpoint import _save_checkpoint
from pylatro_agent.training.ppo_config import PPOConfig, _validate_resume_provenance
from pylatro_agent.training.ppo_optimization import _restore_policy_optimizer_state
from pylatro_agent.transfer import sha256
from pylatro_agent.value_head import outcome_nll, return_huber_loss
from pylatro_agent.vocab import build_vocab
from tools.outcome_critic_intervention import COMPLETE_CRITIC_PROTOCOL
from tools.run_outcome_pilots import checkpoint_at_update, completed_checkpoint
from tools.train_outcome_ppo import branch_checkpoint, same_state_tree, validate_branch_recipe


def test_lr_branch_preserves_parent_weights_moments_rng_and_strict_resume(tmp_path):
    config = PPOConfig(lr=1e-5, gamma=1., gae_lambda=1., reward_config=outcome_reward(), deck_key="b_blue",
                       ppo_run_uuid="parent", ppo_source_sha256="a" * 64, ppo_recipe_id="gae100",
                       eval_games=1, eval_seeds=[190000, 190001], training_excluded_seeds=[200000])
    architecture = AgentConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32, win_only_value=True)
    model = BalatroAgent(architecture, build_vocab(load_game_data()))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    sum(parameter.sum() for parameter in model.parameters()).backward()
    optimizer.step()
    source = _save_checkpoint(model=model, optimizer=optimizer, save_path=tmp_path,
                              update_count=20, total_steps=163840, planned_updates=20, entropy_coeff=.001,
                              entropy_signal_ema=.15, lr=config.lr, agent_config=architecture, config=config,
                              extra={"schedule_total_steps": 163840})
    parent_hash = sha256(source)
    changed = replace(config, lr=3e-5, ppo_run_uuid="child", ppo_recipe_id="gae100_lr3e5")
    destination = tmp_path / "branch.pt"
    evidence = branch_checkpoint(source, destination, changed)
    assert sha256(source) == parent_hash
    assert evidence["intervention"] == {"lr": {"from": 1e-5, "to": 3e-5}}
    original = load_checkpoint_payload(source, "cpu")
    branched = load_ppo_resume_payload(destination, "cpu", active_reward_config=outcome_reward(), active_win_ante=8)
    _validate_resume_provenance(changed, branched)
    assert same_state_tree(original["state_dict"], branched["state_dict"])
    assert same_state_tree(original["optimizer_state_dict"], branched["optimizer_state_dict"])
    assert same_state_tree(original["rng_states"], branched["rng_states"])
    restored = torch.optim.Adam(model.parameters(), lr=changed.lr)
    _restore_policy_optimizer_state(restored, branched, changed, torch.device("cpu"))
    assert restored.param_groups[0]["lr"] == 3e-5
    assert same_state_tree(original["optimizer_state_dict"]["state"], restored.state_dict()["state"])
    with pytest.raises(ValueError, match="Undeclared"):
        branch_checkpoint(source, tmp_path / "bad_gamma.pt", replace(changed, gamma=.99))
    with pytest.raises(ValueError, match="different core"):
        branch_checkpoint(source, tmp_path / "bad_source.pt", replace(changed, ppo_source_sha256="b" * 64))
    with pytest.raises(FileExistsError):
        branch_checkpoint(source, destination, changed)


def test_lr_branch_selects_fixed_parent_after_control_has_advanced(tmp_path):
    paths = {}
    for update in (20, 35):
        directory = tmp_path / f"leg_from0_to{update}"
        directory.mkdir()
        checkpoint = directory / "checkpoint.pt"
        checkpoint.write_bytes(f"checkpoint {update}".encode())
        paths[update] = checkpoint
        (directory / "manifest.json").write_text(json.dumps(dict(
            status="complete", update_count=update, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        )))
    assert completed_checkpoint(tmp_path) == (35, paths[35])
    assert checkpoint_at_update(tmp_path, 20) == paths[20]
    with pytest.raises(ValueError, match="Expected one"):
        checkpoint_at_update(tmp_path, 21)


def test_lr_branch_validates_fields_outside_the_checkpoint_subset():
    config = PPOConfig(ppo_source_sha256="a" * 64, lr=1e-5, value_loss_coeff=.25)
    recipe = dict(source_core_sha256="a" * 64, ppo_config=asdict(config))
    validate_branch_recipe(recipe, replace(config, lr=3e-5, save_dir="a-new-directory"))
    with pytest.raises(ValueError, match="complete recipe"):
        validate_branch_recipe(recipe, replace(config, lr=3e-5, value_loss_coeff=.1))


def test_terminal_critic_resets_return_moments_and_stays_zero_during_updates(tmp_path):
    config = PPOConfig(lr=1e-5, gamma=1., gae_lambda=1., reward_config=outcome_reward(), deck_key="b_blue",
                       ppo_run_uuid="parent", ppo_source_sha256="a" * 64, ppo_recipe_id="gae100")
    architecture = AgentConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32, win_only_value=True)
    model = BalatroAgent(architecture, build_vocab(load_game_data()))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    sum(parameter.sum() for parameter in model.parameters()).backward()
    optimizer.step()
    source = _save_checkpoint(model=model, optimizer=optimizer, save_path=tmp_path,
                              update_count=20, total_steps=163840, planned_updates=20, entropy_coeff=.001,
                              entropy_signal_ema=.15, lr=config.lr, agent_config=architecture, config=config)
    source_hash = sha256(source)
    # Merely zeroing the layer is insufficient: existing Adam momentum moves it.
    naive = copy.deepcopy(model)
    naive_optimizer = torch.optim.Adam(naive.parameters(), lr=config.lr)
    naive_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    naive_optimizer.zero_grad()
    for parameter in naive.value_head.return_residual.parameters():
        parameter.data.zero_()
        parameter.grad = torch.zeros_like(parameter)
    naive_optimizer.step()
    assert any(torch.count_nonzero(parameter) for parameter in naive.value_head.return_residual.parameters())

    changed = replace(config, value_loss_coeff=0., ppo_run_uuid="terminal", ppo_recipe_id="gae100_terminal")
    recipe = dict(ppo_config=asdict(config), source_core_sha256="a" * 64)
    validate_branch_recipe(recipe, changed, "terminal_critic")
    with pytest.raises(ValueError, match="complete recipe"):
        validate_branch_recipe(recipe, replace(changed, lr=3e-5), "terminal_critic")
    destination = tmp_path / "terminal.pt"
    branch_checkpoint(source, destination, changed, "terminal_critic", asdict(config))
    payload = load_checkpoint_payload(destination, "cpu")
    model.load_state_dict(payload["state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    for _ in range(3):
        optimizer.zero_grad()
        values = model.value_head(torch.randn(32, 1, 16), torch.ones(32, 1))
        loss = changed.value_loss_coeff * return_huber_loss(values, torch.full((32,), 10.))
        loss += changed.outcome_loss_coeff * outcome_nll(values["outcome_probabilities"], torch.tensor([0, 8] * 16))
        loss.backward()
        optimizer.step()
        assert all(torch.count_nonzero(parameter) == 0 for parameter in model.value_head.return_residual.parameters())
        assert torch.equal(values["expected_return"], values["terminal_value"])
    assert sha256(source) == source_hash
    complete = replace(changed, ppo_run_uuid="complete", ppo_recipe_id="gae100_complete",
                       terminal_replay_updates_per_ppo_update=30)
    validate_branch_recipe(recipe, complete, "complete_critic")
    complete_path = tmp_path / "complete.pt"
    branch_checkpoint(source, complete_path, complete, "complete_critic", asdict(config))
    full = load_checkpoint_payload(complete_path, "cpu")
    assert full["outcome_study_critic_protocol"] == COMPLETE_CRITIC_PROTOCOL
    assert full["outcome_study_branch"]["kind"] == "complete_critic"
    assert sha256(source) == source_hash
