"""Bounded production PPO updates, failure atomicity, and exact CPU resume."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace

import pytest
import test_complete_cohort as fixtures
import torch
from test_complete_cohort import (
    ScriptedEnvs,
    cohort_config,
    tiny_model,
)

from pylatro_agent.checkpoint import load_checkpoint_payload
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.training import cohort_training
from pylatro_agent.training.cohort_config import validate_cohort_resume
from pylatro_agent.training.ppo import train_ppo
from pylatro_agent.training.ppo_checkpoint import _save_checkpoint
from pylatro_agent.training.ppo_config import PPOConfig
from tools.train_outcome_ppo import same_state_tree

cpu_threads = fixtures.cpu_threads
game_data = fixtures.game_data
observation = fixtures.observation


def install_workers(monkeypatch, observation, *, failure=None):
    workers = []

    def create(_workers, **kwargs):
        assert kwargs["enable_teacher"] is False
        worker = ScriptedEnvs(observation, failure=failure)
        workers.append(worker)
        return worker

    monkeypatch.setattr(cohort_training, "EvaluationEnvs", create)
    return workers


def test_production_cohort_ppo_exact_cpu_resume(monkeypatch, observation, game_data, tmp_path):
    workers = install_workers(monkeypatch, observation)

    def reject(*args, **kwargs):
        raise AssertionError("Complete-cohort PPO called a teacher")

    monkeypatch.setattr(HeuristicAgent, "select_action", reject)
    architecture = tiny_model(game_data).config
    config = cohort_config(save_dir=str(tmp_path / "full/checkpoints"), log_dir=str(tmp_path / "full/logs"))
    train_ppo(config, copy.deepcopy(architecture), data=game_data)
    uninterrupted = load_checkpoint_payload(tmp_path / "full/checkpoints/ppo_latest.pt", "cpu")
    split = replace(
        config, total_updates=1, save_dir=str(tmp_path / "split/checkpoints"), log_dir=str(tmp_path / "split/logs")
    )
    train_ppo(split, copy.deepcopy(architecture), data=game_data)
    first_path = tmp_path / "split/checkpoints/ppo_latest.pt"
    first = load_checkpoint_payload(first_path, "cpu")
    assert first["update_count"] == 1
    assert first["total_steps"] == 9
    assert first["cohort_state"]["seed_cursor"] == 2
    split.total_updates = 2
    train_ppo(split, resume_path=str(first_path), data=game_data)
    resumed = load_checkpoint_payload(first_path, "cpu")
    for key in ("state_dict", "optimizer_state_dict", "rng_states", "cohort_state", "total_steps", "update_count"):
        assert same_state_tree(uninterrupted[key], resumed[key]), key
    assert resumed["total_steps"] == 18  # Not updates * 8192, nor sampled optimizer rows.
    assert resumed["cohort_state"]["optimizer_examples"] == 32
    assert resumed["cohort_state"]["slot_counters"] == [4, 14]
    assert workers[2].resets == split.cohort_seeds[2:4]
    assert all(worker.closed for worker in workers)
    assert resumed["cohort_ppo_config"]["cohort_seeds"] == split.cohort_seeds
    assert resumed["ppo_config_fields"]["collection_mode"] == "complete_cohort"


def test_failed_cohort_never_reaches_optimizer_or_commits_rows(monkeypatch, observation, game_data, tmp_path):
    workers = install_workers(monkeypatch, observation, failure="stall")

    def reject(*args, **kwargs):
        raise AssertionError("Optimization ran on an incomplete cohort")

    monkeypatch.setattr(cohort_training, "_run_ppo_update", reject)
    config = cohort_config(save_dir=str(tmp_path / "checkpoints"), log_dir=str(tmp_path / "logs"))
    with pytest.raises(RuntimeError, match="incomplete"):
        train_ppo(config, tiny_model(game_data).config, data=game_data)
    saved = load_checkpoint_payload(tmp_path / "checkpoints/ppo_latest.pt", "cpu")
    assert saved["update_count"] == 0
    assert saved["cohort_state"]["seed_cursor"] == 0
    assert saved["total_steps"] == 0
    assert (tmp_path / "logs/failure.json").exists()
    assert workers[0].closed


def test_real_games_complete_one_cpu_ppo_update_without_a_teacher(monkeypatch, game_data, tmp_path):
    def reject(*args, **kwargs):
        raise AssertionError("Complete-cohort PPO called a teacher")

    monkeypatch.setattr(HeuristicAgent, "select_action", reject)
    config = cohort_config(
        total_updates=1,
        cohort_max_episode_steps=256,
        max_no_progress_steps=32,
        save_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
    )
    train_ppo(config, tiny_model(game_data).config, data=game_data)
    saved = load_checkpoint_payload(tmp_path / "checkpoints/ppo_latest.pt", "cpu")
    assert saved["update_count"] == 1 and saved["cohort_state"]["seed_cursor"] == 2
    assert 0 < saved["total_steps"] <= 512
    assert saved["cohort_state"]["collected_transitions"] == saved["total_steps"]
    assert saved["cohort_state"]["optimizer_examples"] == 16


@pytest.mark.parametrize("field", ["optimizer_examples", "collected_transitions"])
def test_resume_rejects_inconsistent_committed_counts(field):
    config = cohort_config()
    state = cohort_training.initial_cohort_state(config)
    state[field] = 1
    with pytest.raises(ValueError, match="counts disagree"):
        cohort_training._validate_state(state, config, 0, 1)


@pytest.mark.parametrize(
    "change",
    [
        dict(cohort_seeds=[1, 2]),
        dict(outcome_label_mode="completion_window"),
        dict(cohort_games=4),
        dict(rollout_length=8),
        dict(device="mps"),
        dict(hand_ar_mixture_eps=0.1),
    ],
)
def test_cohort_resume_rejects_training_configuration_changes(change):
    config = cohort_config()
    payload = {
        "ppo_config_fields": asdict(config),
        "cohort_ppo_config": asdict(config),
        "cohort_state": cohort_training.initial_cohort_state(config),
    }
    with pytest.raises(ValueError, match="changed"):
        validate_cohort_resume(replace(config, **change), payload)


def test_legacy_resume_requires_explicit_collection_branch(game_data, tmp_path):
    model = tiny_model(game_data)
    config = PPOConfig()
    path = _save_checkpoint(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        save_path=tmp_path,
        update_count=1,
        total_steps=8,
        planned_updates=2,
        entropy_coeff=0.01,
        entropy_signal_ema=None,
        lr=1e-5,
        agent_config=model.config,
        config=config,
    )
    payload = load_checkpoint_payload(path, "cpu")
    # A checkpoint predating the new fields still loads on the legacy path.
    del payload["ppo_config_fields"]["collection_mode"]
    validate_cohort_resume(PPOConfig(), payload)
    with pytest.raises(ValueError, match="explicit"):
        validate_cohort_resume(cohort_config(), payload)
    payload["ppo_config_fields"]["collection_mode"] = "complete_cohort"
    with pytest.raises(ValueError, match="explicit"):
        validate_cohort_resume(PPOConfig(), payload)


def test_resume_rejects_missing_or_wrong_cohort_state():
    config = cohort_config()
    payload = {"ppo_config_fields": asdict(config), "cohort_ppo_config": asdict(config)}
    with pytest.raises(ValueError, match="metadata"):
        validate_cohort_resume(config, payload)
    payload["cohort_state"] = {"version": 99}
    with pytest.raises(ValueError, match="metadata"):
        validate_cohort_resume(config, payload)


def test_insufficient_seed_plan_fails_before_workers(monkeypatch, game_data, tmp_path):
    def reject(*args, **kwargs):
        raise AssertionError("Workers started with an insufficient seed plan")

    monkeypatch.setattr(cohort_training, "EvaluationEnvs", reject)
    config = cohort_config(
        cohort_seeds=[7000000, 7000001],
        total_updates=2,
        save_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
    )
    with pytest.raises(ValueError, match="Insufficient"):
        train_ppo(config, tiny_model(game_data).config, data=game_data)
