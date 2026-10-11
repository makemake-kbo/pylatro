"""Continuation preserves state and overrides only the explicit pilot gate."""

import copy
import pickle

import numpy as np
import pytest
import torch

from pylatro_agent.search_ppo_study import atomic
from pylatro_agent.training.search_guidance import RECIPE
from pylatro_agent.training.search_ppo import PublicPPOConfig, validate_resume
from tools.continue_search_guided_ppo import (
    TRAINING_SECONDS,
    branch_payload,
    can_start_update,
    clock,
    continuation_config,
    paired_result,
    reserve,
)
from tools.train_outcome_ppo import same_state_tree


def test_branch_preserves_adam_policy_label_rng_and_current_slot_counters():
    from dataclasses import asdict

    from pylatro_agent.outcome_study import outcome_reward

    config = PublicPPOConfig(reward_config=outcome_reward(), search_guidance=RECIPE.metadata(), total_updates=1)
    parent = dict(
        state_dict={"weight": torch.tensor([1.5])},
        optimizer_state_dict={"state": {0: {"exp_avg": torch.tensor([0.3]), "step": torch.tensor(17.0)}}},
        rng_states={
            "torch_cpu": torch.Generator().manual_seed(5).get_state(),
            "torch_mps": torch.Generator().manual_seed(8).get_state(),
            "numpy": np.random.RandomState(7).get_state(),
        },
        observation_mode=config.observation_mode,
        public_ppo_config=asdict(config),
        update_count=143,
        total_steps=23456,
        cohort_state={"slot_counters": [999] * config.num_envs},
        search_ppo_study=dict(
            completed_updates=1,
            slot_counters=[1234] * config.num_envs,
            guidance_rng=np.random.RandomState(9).get_state(),
            evidence={"old": "hash"},
        ),
    )
    before = pickle.dumps(parent)
    continued = continuation_config(parent, list(range(100000, 104096)))
    identity = {"parent_sha256": "test-parent"}
    result = branch_payload(parent, continued, identity)
    for key in ("state_dict", "optimizer_state_dict", "rng_states", "update_count", "total_steps"):
        assert same_state_tree(result[key], parent[key])
    assert same_state_tree(result["search_ppo_study"]["guidance_rng"], parent["search_ppo_study"]["guidance_rng"])
    assert result["search_ppo_study"]["slot_counters"] == [1234] * config.num_envs
    assert result["search_ppo_study"]["seed_cursor"] == result["search_ppo_study"]["completed_updates"] == 0
    assert pickle.dumps(parent) == before
    old, new = asdict(config), asdict(continued)
    assert {k for k in old if old[k] != new[k]} == {"total_updates", "cohort_seeds"}
    validate_resume(result, identity, continued)
    changed = copy.deepcopy(continued)
    changed.search_guidance["coefficient"] = 0.03
    with pytest.raises(ValueError):
        validate_resume(result, identity, changed)


def test_fresh_seeds_exclude_all_previous_reservations():
    first = reserve(set())
    previous = {s for panel in first.values() for s in panel}
    second = reserve(previous)
    flat = [s for panel in second.values() for s in panel]
    assert len(flat) == len(set(flat)) and not set(flat) & previous
    assert second == reserve(previous)
    assert len(second["evaluation"]) == 256


def test_four_hour_clock_cannot_reset_and_budget_uses_runtime_only(tmp_path):
    original = clock(tmp_path, now=1000)
    assert original["training_deadline"] == 1000 + TRAINING_SECONDS
    assert clock(tmp_path, now=2000) == original
    assert can_start_update(0, 1000, [100, 200], 300)
    assert not can_start_update(700, 1000, [100, 200], 300)
    atomic(tmp_path / "clock.json", {k: v + 1 for k, v in original.items()})
    with pytest.raises(ValueError, match="Frozen artifact"):
        clock(tmp_path)


def test_paired_single_run_interval_and_missing_results():
    positive = paired_result([1] * 256, [0] * 256)
    assert positive["improvement"] and positive["ci95"] == [1.0, 1.0]
    tied = paired_result([1, 0] * 128, [1, 0] * 128)
    assert tied["gain"] == 0 and not tied["improvement"]
    assert paired_result([0] * 256, [1] * 256)["gain"] == -1
    with pytest.raises(ValueError):
        paired_result([1], [])
