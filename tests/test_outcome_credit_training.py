"""Outcome-directed PPO targets, objective selection, and exact continuation."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import test_complete_cohort as fixtures
import torch
from test_complete_cohort import cohort_config, tiny_model
from test_complete_cohort_training import install_workers

from pylatro_agent.checkpoint import load_checkpoint_payload
from pylatro_agent.training import outcome_credit, ppo_optimization
from pylatro_agent.training.cohort_config import validate_cohort_resume
from pylatro_agent.training.cohort_training import initial_cohort_state
from pylatro_agent.training.ppo import train_ppo
from pylatro_agent.training.ppo_checkpoint import _load_checkpoint_strict
from pylatro_agent.training.ppo_config import _validate_ppo_config
from tools.train_outcome_ppo import same_state_tree

cpu_threads = fixtures.cpu_threads
game_data = fixtures.game_data
observation = fixtures.observation


def outcome_config(**kwargs):
    return cohort_config(
        return_value_mode="terminal_only",
        advantage_estimator="complete_episode_mc",
        value_loss_coeff=0,
        **kwargs,
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"collection_mode": "fixed_steps", "outcome_label_mode": "completion_window"}, "require complete_cohort"),
        ({"return_value_mode": "unknown"}, "return_value_mode"),
        ({"advantage_estimator": "unknown"}, "advantage_estimator"),
        ({"return_value_mode": "terminal_plus_residual"}, "requires terminal_only"),
        ({"value_loss_coeff": 0.25}, "zero residual loss"),
        ({"outcome_loss_coeff": 0}, "positive outcome loss"),
        ({"outcome_loss_coeff": float("nan")}, "positive outcome loss"),
        ({"outcome_loss_coeff": float("inf")}, "positive outcome loss"),
        ({"outcome_label_mode": "completion_window"}, "complete_episode"),
        ({"gamma": 0.99}, "gamma=1"),
        ({"milestone_final_scale": 0.5}, "fixed reward"),
    ],
)
def test_outcome_credit_rejects_incoherent_recipe(overrides, message):
    with pytest.raises(ValueError, match=message):
        _validate_ppo_config(replace(outcome_config(), **overrides))


def test_outcome_credit_rejects_dense_reward():
    config = outcome_config()
    config.reward_config = replace(config.reward_config, milestone_reward_budget=1)
    with pytest.raises(ValueError, match="terminal-only"):
        _validate_ppo_config(config)


def test_old_cohort_recipe_restores_only_historical_credit_defaults():
    config = cohort_config()
    saved = asdict(config)
    del saved["return_value_mode"], saved["advantage_estimator"]
    payload = {
        "ppo_config_fields": dict(saved),
        "cohort_ppo_config": dict(saved),
        "cohort_state": initial_cohort_state(config),
    }
    validate_cohort_resume(config, payload)
    with pytest.raises(ValueError, match="changed on resume"):
        validate_cohort_resume(outcome_config(), payload)
    # Compatibility must not default an unrelated missing field.
    del payload["cohort_ppo_config"]["gae_lambda"]
    with pytest.raises(ValueError, match="schema"):
        validate_cohort_resume(config, payload)


def test_production_outcome_credit_skips_wrong_loss_and_resumes_exactly(monkeypatch, observation, game_data, tmp_path):
    workers = install_workers(monkeypatch, observation)

    def reject_huber(*_args, **_kwargs):
        raise AssertionError("terminal-only PPO invoked the mismatched residual objective")

    monkeypatch.setattr(ppo_optimization, "return_huber_loss", reject_huber)
    architecture = tiny_model(game_data).config
    config = outcome_config(save_dir=str(tmp_path / "full/checkpoints"), log_dir=str(tmp_path / "full/logs"))
    train_ppo(config, copy.deepcopy(architecture), data=game_data)
    final = load_checkpoint_payload(tmp_path / "full/checkpoints/ppo_latest.pt", "cpu")
    initial = load_checkpoint_payload(tmp_path / "full/checkpoints/ppo_update0.pt", "cpu")
    split = replace(
        config, total_updates=1, save_dir=str(tmp_path / "split/checkpoints"), log_dir=str(tmp_path / "split/logs")
    )
    train_ppo(split, copy.deepcopy(architecture), data=game_data)
    first = tmp_path / "split/checkpoints/ppo_latest.pt"
    split.total_updates = 2
    train_ppo(split, resume_path=str(first), data=game_data)
    resumed = load_checkpoint_payload(first, "cpu")
    for key in ("state_dict", "optimizer_state_dict", "rng_states", "cohort_state", "total_steps", "update_count"):
        assert same_state_tree(final[key], resumed[key]), key
    assert final["agent_config"]["return_value_mode"] == "terminal_only"
    assert final["ppo_config_fields"]["advantage_estimator"] == "complete_episode_mc"
    assert final["cohort_ppo_config"]["value_loss_coeff"] == 0
    for name, tensor in initial["state_dict"].items():
        if name.startswith(("value_head.pool_proj.", "value_head.return_residual.")):
            assert same_state_tree(tensor, final["state_dict"][name]), name
    for update in (1, 2):
        folder = tmp_path / f"full/logs/cohorts/update_{update}"
        stats = json.loads((folder / "update.json").read_text())
        assert stats["optimizer"]["return_loss_enabled"] is False
        assert all(loss == 0 for loss in stats["optimizer"]["return_hubers"])
        assert stats["buffer"]["credit_targets"]["return_target_mse_to_completed_return"] == 0
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["ppo_config"]["return_value_mode"] == "terminal_only"
        for episode in manifest["episodes"]:
            with np.load(folder / episode["path"]) as arrays:
                assert ((arrays["values"] >= 0) & (arrays["values"] <= 10)).all()
    assert all(worker.closed for worker in workers)


def test_resume_cannot_silently_change_agent_value_semantics(monkeypatch, observation, game_data, tmp_path):
    install_workers(monkeypatch, observation)
    config = outcome_config(total_updates=1, save_dir=str(tmp_path / "checkpoints"), log_dir=str(tmp_path / "logs"))
    train_ppo(config, tiny_model(game_data).config, data=game_data)
    with pytest.raises(ValueError, match="return_value_mode changed"):
        _load_checkpoint_strict(tiny_model(game_data), str(tmp_path / "checkpoints/ppo_latest.pt"), torch.device("cpu"))
    legacy_architecture = tiny_model(game_data).config
    with pytest.raises(ValueError, match="architecture differs"):
        train_ppo(
            replace(config, total_updates=2),
            legacy_architecture,
            resume_path=str(tmp_path / "checkpoints/ppo_latest.pt"),
            data=game_data,
        )


def test_migrated_training_retains_provenance_and_rejects_source_drift(monkeypatch, observation, game_data, tmp_path):
    install_workers(monkeypatch, observation)
    monkeypatch.setattr(outcome_credit, "source_hashes", lambda: {"fixture.py": "a" * 64})
    config = cohort_config(
        total_updates=1, save_dir=str(tmp_path / "parent/checkpoints"), log_dir=str(tmp_path / "parent/logs")
    )
    train_ppo(config, tiny_model(game_data).config, data=game_data)
    branch = tmp_path / "branch/initial.pt"
    outcome_credit.prepare_outcome_credit_branch(tmp_path / "parent/checkpoints/ppo_latest.pt", branch)
    initial = load_checkpoint_payload(branch, "cpu")
    recipe, _ = outcome_credit._parent_configuration(initial)
    recipe.total_updates = 2
    train_ppo(recipe, resume_path=str(branch), data=game_data)
    latest = tmp_path / "branch/checkpoints/ppo_latest.pt"
    trained = load_checkpoint_payload(latest, "cpu")
    assert trained["outcome_credit_branch"] == initial["outcome_credit_branch"]
    assert trained["cohort_state"]["seed_cursor"] == 4
    monkeypatch.setattr(outcome_credit, "source_hashes", lambda: {"fixture.py": "b" * 64})

    def reject_workers(*_args, **_kwargs):
        raise AssertionError("A changed-source checkpoint reached environment creation")

    monkeypatch.setattr("pylatro_agent.training.cohort_training.EvaluationEnvs", reject_workers)
    with pytest.raises(ValueError, match="source"):
        train_ppo(replace(recipe, total_updates=3), resume_path=str(latest), data=game_data)


def test_additional_updates_persist_effective_recipe(monkeypatch, observation, game_data, tmp_path):
    install_workers(monkeypatch, observation)
    config = outcome_config(total_updates=1, save_dir=str(tmp_path / "checkpoints"), log_dir=str(tmp_path / "logs"))
    train_ppo(config, tiny_model(game_data).config, data=game_data)
    latest = tmp_path / "checkpoints/ppo_latest.pt"
    train_ppo(config, resume_path=str(latest), additional_updates=1, data=game_data)
    saved = load_checkpoint_payload(latest, "cpu")
    assert saved["update_count"] == saved["planned_updates"] == saved["cohort_ppo_config"]["total_updates"] == 2
