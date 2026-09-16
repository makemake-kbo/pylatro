"""Checkpoint semantics and preparation-only migration preserve training state."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, replace

import numpy as np
import pytest
import test_complete_cohort as fixtures
import torch

from pylatro_agent import checkpoint
from pylatro_agent.agent import AgentConfig
from pylatro_agent.outcome_study import outcome_reward
from pylatro_agent.reward import DEFAULT_REWARD_CONFIG, reward_checkpoint_metadata
from pylatro_agent.training import outcome_credit
from pylatro_agent.training.cohort_config import validate_cohort_resume
from pylatro_agent.training.cohort_training import initial_cohort_state
from pylatro_agent.training.outcome_credit import prepare_outcome_credit_branch, validate_outcome_credit_sources
from pylatro_agent.training.ppo_checkpoint import _save_checkpoint
from pylatro_agent.training.ppo_config import _effective_reward_config
from pylatro_agent.training.ppo_evaluation import training_seed_reservations
from pylatro_agent.training.ppo_observations import _single_obs_to_batch
from pylatro_agent.training.ppo_policy import _grammar_distribution
from pylatro_agent.transfer import load_policy, restore_configuration, sha256
from pylatro_agent.vocab import build_vocab
from tools.prepare_outcome_credit import main
from tools.train_outcome_ppo import same_state_tree

cpu_threads = fixtures.cpu_threads
game_data = fixtures.game_data
observation = fixtures.observation


@pytest.fixture
def parent(game_data, tmp_path, monkeypatch):
    # Exercise accelerator RNG preservation as opaque bytes without using MPS.
    monkeypatch.setattr(
        checkpoint,
        "capture_rng_states",
        lambda: {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_mps": torch.arange(32, dtype=torch.uint8),
        },
    )
    monkeypatch.setattr(outcome_credit, "source_hashes", lambda: {"fixture.py": "a" * 64})
    model = fixtures.tiny_model(game_data)
    config = fixtures.cohort_config()
    config.reward_config = _effective_reward_config(config)
    model._training_reserved_seeds = training_seed_reservations(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    sum(parameter.sum() for parameter in model.parameters()).backward()
    optimizer.step()
    state = initial_cohort_state(config)
    state.update(seed_cursor=2, slot_counters=[2, 7], optimizer_examples=16, collected_transitions=9)
    return _save_checkpoint(
        model=model,
        optimizer=optimizer,
        save_path=tmp_path,
        update_count=1,
        total_steps=9,
        planned_updates=2,
        entropy_coeff=config.entropy_coeff,
        entropy_signal_ema=None,
        lr=config.lr,
        agent_config=model.config,
        config=config,
        extra={"cohort_state": state, "cohort_ppo_config": asdict(config)},
    )


@pytest.mark.parametrize(
    "missing", [(), ("return_value_mode",), ("action_head_variant",), ("return_value_mode", "action_head_variant")]
)
def test_known_agent_schemas_restore_legacy_semantics_without_mutation(missing):
    payload = {"agent_config": asdict(AgentConfig()), **reward_checkpoint_metadata(DEFAULT_REWARD_CONFIG)}
    for key in missing:
        del payload["agent_config"][key]
    before = copy.deepcopy(payload)
    config, _ = restore_configuration(payload)
    assert config.return_value_mode == "terminal_plus_residual"
    assert config.action_head_variant == "legacy_v1"
    assert payload == before
    payload["agent_config"].pop("d_model")
    with pytest.raises(ValueError, match="configuration fields"):
        restore_configuration(payload)


def test_explicit_terminal_mode_survives_configuration_restore():
    config = AgentConfig(win_only_value=True, return_value_mode="terminal_only")
    payload = {"agent_config": asdict(config), **reward_checkpoint_metadata(outcome_reward())}
    assert restore_configuration(payload)[0] == config
    payload["agent_config"]["return_value_mode"] = "unknown"
    with pytest.raises(ValueError, match="return_value_mode"):
        restore_configuration(payload)


@pytest.mark.parametrize(
    "reward",
    [
        DEFAULT_REWARD_CONFIG,
        replace(outcome_reward(), gamma=0.99),
        replace(outcome_reward(), milestone_reward_budget=1),
    ],
)
def test_terminal_value_checkpoint_rejects_incompatible_reward_meaning(reward):
    payload = {
        "agent_config": asdict(AgentConfig(win_only_value=True, return_value_mode="terminal_only")),
        **reward_checkpoint_metadata(reward),
    }
    with pytest.raises(ValueError, match="terminal-only checkpoint rewards"):
        restore_configuration(payload)


@pytest.mark.parametrize("metadata", ["ppo_config_fields", "cohort_ppo_config"])
def test_restore_rejects_cross_metadata_value_semantics(metadata):
    payload = {
        "agent_config": asdict(AgentConfig(win_only_value=True)),
        **reward_checkpoint_metadata(DEFAULT_REWARD_CONFIG),
        metadata: {"return_value_mode": "terminal_only"},
    }
    with pytest.raises(ValueError, match="return_value_mode disagrees"):
        restore_configuration(payload)
    # A deleted new-mode field cannot disguise a checkpoint as a legacy one.
    del payload["agent_config"]["return_value_mode"]
    with pytest.raises(ValueError, match="return_value_mode disagrees"):
        restore_configuration(payload)


@pytest.mark.parametrize("legacy", [False, True])
def test_branch_preserves_full_state_seed_plan_and_records_every_override(parent, tmp_path, legacy):
    if legacy:
        archived = checkpoint.load_checkpoint_payload(parent, "cpu")
        archived["agent_config"].pop("return_value_mode")
        for field in ("return_value_mode", "advantage_estimator"):
            archived["cohort_ppo_config"].pop(field)
            archived["ppo_config_fields"].pop(field)
        torch.save(archived, parent)
    old_hash = sha256(parent)
    old = checkpoint.load_checkpoint_payload(parent, "cpu")
    destination = tmp_path / "new/initial.pt"
    result = prepare_outcome_credit_branch(parent, destination, parent_sha256=old_hash)
    new = checkpoint.load_checkpoint_payload(destination, "cpu")
    for key in outcome_credit._PRESERVED:
        assert same_state_tree(old[key], new[key]), key
    assert sha256(parent) == old_hash
    assert new["cohort_ppo_config"]["cohort_seeds"] == old["cohort_ppo_config"]["cohort_seeds"]
    assert result["preserved_consumed_seed_count"] == 2
    assert result["remaining_predeclared_games"] == 10
    assert result["agent_overrides"] == {
        "return_value_mode": {"from": "terminal_plus_residual", "to": "terminal_only"},
    }
    config, _ = outcome_credit._parent_configuration(new)
    assert config.advantage_estimator == "complete_episode_mc"
    assert config.return_value_mode == "terminal_only"
    assert config.value_loss_coeff == 0
    assert config.outcome_label_mode == "complete_episode"
    validate_cohort_resume(config, new)
    assert not result["training_started"] and not result["accelerator_budget_assigned"]
    assert result["source_hashes"] == {"fixture.py": "a" * 64}
    assert (len(result["configuration_migrations"]) == 3) if legacy else not result["configuration_migrations"]
    assert not (destination.parent / "checkpoints").exists()
    assert not (destination.parent / "logs").exists()
    before_hash = sha256(destination)
    with pytest.raises(FileExistsError):
        prepare_outcome_credit_branch(parent, destination)
    assert sha256(destination) == before_hash


def test_evaluation_restores_mode_and_keeps_actor_probabilities(parent, game_data, observation, tmp_path):
    destination = tmp_path / "branch/initial.pt"
    prepare_outcome_credit_branch(parent, destination)
    vocab = build_vocab(game_data)
    old_model, _, _ = load_policy(parent, vocab, torch.device("cpu"))
    new_model, _, payload = load_policy(destination, vocab, torch.device("cpu"))
    assert payload["configuration_migrations"] == []
    assert new_model.config.return_value_mode == "terminal_only"
    batch = _single_obs_to_batch(observation, torch.device("cpu"))
    with torch.inference_mode():
        old_dist, _old_value = _grammar_distribution(old_model, batch)
        new_dist, new_value = _grammar_distribution(new_model, batch)
        torch.testing.assert_close(old_dist.action_type_probs, new_dist.action_type_probs, rtol=0, atol=0)
        for action in np.flatnonzero(observation["action_mask"]):
            selected = torch.tensor([action])
            torch.testing.assert_close(old_dist.log_prob(selected), new_dist.log_prob(selected), rtol=0, atol=0)
        torch.testing.assert_close(new_value["expected_return"], 10 * new_value["win_prob"], rtol=0, atol=0)


def test_load_policy_reports_both_archived_configuration_migrations(parent, game_data):
    payload = checkpoint.load_checkpoint_payload(parent, "cpu")
    payload["agent_config"].pop("action_head_variant")
    payload["agent_config"].pop("return_value_mode")
    torch.save(payload, parent)
    model, _, restored = load_policy(parent, build_vocab(game_data), torch.device("cpu"))
    assert model.config.return_value_mode == "terminal_plus_residual"
    assert {item["field"] for item in restored["configuration_migrations"]} == {
        "agent_config.action_head_variant",
        "agent_config.return_value_mode",
    }


@pytest.mark.parametrize("corruption", ["counter", "recipe", "mode", "schema", "rng"])
def test_branch_rejects_inconsistent_parent_without_output(parent, tmp_path, corruption):
    payload = checkpoint.load_checkpoint_payload(parent, "cpu")
    if corruption == "counter":
        payload["cohort_state"]["seed_cursor"] = 4
    elif corruption == "recipe":
        payload["ppo_config_fields"]["gamma"] = 0.99
    elif corruption == "mode":
        payload["agent_config"]["return_value_mode"] = "terminal_only"
    elif corruption == "schema":
        payload["cohort_ppo_config"].pop("lr")
    else:
        payload["rng_states"] = {}
    torch.save(payload, parent)
    destination = tmp_path / "invalid/initial.pt"
    with pytest.raises(ValueError):
        prepare_outcome_credit_branch(parent, destination)
    assert not destination.exists()


def test_parent_hash_mismatch_does_not_create_branch(parent, tmp_path):
    destination = tmp_path / "invalid/initial.pt"
    with pytest.raises(ValueError, match="checksum"):
        prepare_outcome_credit_branch(parent, destination, parent_sha256="0" * 64)
    assert not destination.exists()


def test_branch_normalizes_historical_additional_update_plan(parent, tmp_path):
    saved = checkpoint.load_checkpoint_payload(parent, "cpu")
    saved["planned_updates"] = 3
    torch.save(saved, parent)
    destination = tmp_path / "extended/initial.pt"
    result = prepare_outcome_credit_branch(parent, destination)
    prepared = checkpoint.load_checkpoint_payload(destination, "cpu")
    assert prepared["planned_updates"] == prepared["cohort_ppo_config"]["total_updates"] == 3
    assert result["ppo_overrides"]["total_updates"] == {"from": 2, "to": 3}
    assert same_state_tree(saved["cohort_state"], prepared["cohort_state"])


def test_branch_does_not_point_training_back_at_existing_runtime_directories(parent, tmp_path):
    destination = tmp_path / "old-run/initial.pt"
    (destination.parent / "logs").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="runtime directory already exists"):
        prepare_outcome_credit_branch(parent, destination)
    assert not destination.exists()


def test_cli_only_prepares_checkpoint(parent, tmp_path, capsys):
    destination = tmp_path / "cli/initial.pt"
    main(["--parent", str(parent), "--destination", str(destination)])
    result = json.loads(capsys.readouterr().out)
    assert result["checkpoint"] == str(destination)
    assert not result["training_started"] and not result["accelerator_budget_assigned"]
    assert list(destination.parent.iterdir()) == [destination]


def test_prepared_branch_refuses_changed_source_code(parent, tmp_path, monkeypatch):
    destination = tmp_path / "source-pinned/initial.pt"
    prepare_outcome_credit_branch(parent, destination)
    payload = checkpoint.load_checkpoint_payload(destination, "cpu")
    validate_outcome_credit_sources(payload)
    monkeypatch.setattr(outcome_credit, "source_hashes", lambda: {"fixture.py": "b" * 64})
    with pytest.raises(ValueError, match="source changed after preparation"):
        validate_outcome_credit_sources(payload)


@pytest.mark.parametrize(
    "metadata,key",
    [
        ("outcome_credit_branch", "source_sha256"),
        ("ppo_run_provenance", "source_sha256"),
        ("cohort_ppo_config", "ppo_source_sha256"),
        ("ppo_config_fields", "ppo_source_sha256"),
    ],
)
def test_branch_source_digest_must_agree_across_snapshots(parent, tmp_path, metadata, key):
    destination = tmp_path / "digest/initial.pt"
    prepare_outcome_credit_branch(parent, destination)
    payload = checkpoint.load_checkpoint_payload(destination, "cpu")
    payload[metadata][key] = "0" * 64
    with pytest.raises(ValueError, match="source digest disagrees"):
        validate_outcome_credit_sources(payload)


def test_old_cohort_source_validation_is_unchanged(parent, monkeypatch):
    def reject():
        raise AssertionError("Legacy cohort source pinning was unexpectedly invoked")

    monkeypatch.setattr(outcome_credit, "source_hashes", reject)
    validate_outcome_credit_sources(checkpoint.load_checkpoint_payload(parent, "cpu"))
