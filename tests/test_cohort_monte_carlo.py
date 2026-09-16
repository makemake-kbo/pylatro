"""Complete-game credit must follow observed rewards, not critic bootstrap errors."""

from dataclasses import replace

import numpy as np
import pytest

from pylatro_agent.constants import CURRENT_ANTE_SCALAR_INDEX, NUM_ACTIONS
from pylatro_agent.training.cohort import (
    CohortBuffer,
    CohortEpisode,
    complete_episode_monte_carlo,
    sampling_configuration,
)
from pylatro_agent.training.ppo_config import PPOConfig


def config(**changes):
    defaults = PPOConfig(
        collection_mode="complete_cohort",
        outcome_label_mode="complete_episode",
        advantage_estimator="complete_episode_mc",
        return_value_mode="terminal_only",
        value_loss_coeff=0,
        cohort_training_rows=6,
        rollout_length=4,
        gamma=1,
        gae_lambda=0.97,
        advantage_clip_sigma=0,
    )
    return replace(defaults, **changes)


def episode(settings, *, seed=10, won=True, length=3, offset=0, baseline=2.0):
    result = CohortEpisode(seed=seed, slot=0, policy_version=3, policy_sha256="frozen-collector")
    result.sampling_configuration = sampling_configuration(settings)
    scalars = np.zeros(CURRENT_ANTE_SCALAR_INDEX + 1, np.float32)
    scalars[CURRENT_ANTE_SCALAR_INDEX] = 1
    action_mask = np.zeros(NUM_ACTIONS, np.int8)
    action_mask[0] = 1
    for step in range(length):
        result.observations.append({"scalars": scalars, "action_mask": action_mask})
        result.actions.append(0)
        result.rewards.append(10.0 if won and step == length - 1 else 0.0)
        result.values.append(baseline)
        result.log_probs.append(0.0)
        result.slot_steps.append(offset + step)
    result.outcome = {
        "terminated": True,
        "truncated": False,
        "stalled": False,
        "won": won,
        "final_ante": 9 if won else 1,
    }
    return result


def buffer(settings, episodes):
    size = sum(len(e.actions) for e in episodes)
    return CohortBuffer(episodes, replace(settings, cohort_training_rows=size), draw_indices=np.arange(size))


def test_terminal_reward_reaches_early_decisions_without_trace_decay():
    rewards = np.zeros(600, np.float32)
    rewards[-1] = 10
    advantages, returns = complete_episode_monte_carlo(rewards, np.full(600, 2.0))
    np.testing.assert_array_equal(returns, np.full(600, 10.0))
    np.testing.assert_array_equal(advantages, np.full(600, 8.0))


def test_reward_to_go_uses_actual_rewards_and_excludes_past_rewards():
    advantages, returns = complete_episode_monte_carlo([1, -2, 3, 10], [2, 4, 6, 8])
    np.testing.assert_array_equal(returns, [12, 11, 13, 10])
    np.testing.assert_array_equal(advantages, [10, 7, 7, 2])


def test_baseline_perturbation_changes_only_advantages():
    rewards, baseline = np.array([0, 0, 10]), np.array([1, 2, 3])
    advantages, returns = complete_episode_monte_carlo(rewards, baseline)
    shifted_advantages, shifted_returns = complete_episode_monte_carlo(rewards, baseline + 25)
    np.testing.assert_array_equal(returns, shifted_returns)
    np.testing.assert_array_equal(shifted_advantages, advantages - 25)
    np.testing.assert_array_equal(rewards, [0, 0, 10])
    np.testing.assert_array_equal(baseline, [1, 2, 3])


def test_future_value_error_cannot_reverse_earlier_action_credit():
    rewards = [0, 0, 10]
    advantages, returns = complete_episode_monte_carlo(rewards, [2, 2, 2])
    corrupted_advantages, corrupted_returns = complete_episode_monte_carlo(rewards, [2, -1000, 1000])
    assert advantages[0] == corrupted_advantages[0] == 8
    np.testing.assert_array_equal(returns, corrupted_returns)


@pytest.mark.parametrize("fragment_length", [1, 4, 256])
@pytest.mark.parametrize("offset", [0, 3, 1000])
def test_slot_and_fragment_boundaries_do_not_change_complete_episode_credit(fragment_length, offset):
    reference_config = config()
    reference = buffer(reference_config, [episode(reference_config, length=7)])
    settings = config(rollout_length=fragment_length, gae_lambda=0.0)
    shifted = buffer(settings, [episode(settings, length=7, offset=offset)])
    for key in ("returns", "completed_returns", "raw_advantages", "advantages"):
        np.testing.assert_array_equal(shifted.arrays[key], reference.arrays[key])


def test_complete_returns_do_not_leak_between_games():
    settings = config()
    episodes = [episode(settings, seed=10, won=False), episode(settings, seed=11, won=True)]
    result = buffer(settings, episodes)
    np.testing.assert_array_equal(result.arrays["returns"], [0, 0, 0, 10, 10, 10])
    np.testing.assert_array_equal(result.arrays["raw_advantages"], [-2, -2, -2, 8, 8, 8])
    reversed_result = buffer(settings, episodes[::-1])
    for seed in (10, 11):
        for key in ("returns", "raw_advantages", "advantages"):
            np.testing.assert_array_equal(
                result.arrays[key][result.arrays["seed"] == seed],
                reversed_result.arrays[key][reversed_result.arrays["seed"] == seed],
            )


def test_buffer_keeps_collected_baseline_and_credit_fixed_for_optimizer_passes():
    settings = config()
    collected = episode(settings)
    result = buffer(settings, [collected])
    first_batch = result.materialize([0, 1, 2], "cpu")
    collected.values[:] = [100, 200, 300]
    second_batch = result.materialize([0, 1, 2], "cpu")
    for key in ("returns", "advantages"):
        np.testing.assert_array_equal(first_batch[key].numpy(), second_batch[key].numpy())
    np.testing.assert_array_equal(result.arrays["values"], [2, 2, 2])
    np.testing.assert_array_equal(result.arrays["raw_advantages"], [8, 8, 8])


def test_legacy_fragment_targets_remain_unchanged_and_error_is_reported():
    settings = config(advantage_estimator="fragment_gae", gae_lambda=1)
    collected = episode(settings, offset=2)
    collected.values[:] = [1, 2, 3]
    result = buffer(settings, [collected])
    np.testing.assert_array_equal(result.arrays["returns"], [3, 3, 10])
    np.testing.assert_array_equal(result.arrays["raw_advantages"], [2, 1, 7])
    np.testing.assert_array_equal(result.arrays["completed_returns"], [10, 10, 10])
    metrics = result.diagnostics()
    assert metrics["advantage_estimator"] == "fragment_gae"
    assert metrics["credit_targets"]["return_target_bias_to_completed_return"] == pytest.approx(-14 / 3)
    assert metrics["credit_targets"]["return_target_mse_to_completed_return"] == pytest.approx(98 / 3)


def test_diagnostics_compare_unnormalized_values_with_actual_returns():
    settings = config()
    result = buffer(settings, [episode(settings)])
    metrics = result.diagnostics()
    assert metrics["advantage_estimator"] == "complete_episode_mc"
    assert metrics["credit_targets"] == {
        "completed_return_mean": 10.0,
        "critic_bias_to_completed_return": -8.0,
        "critic_mse_to_completed_return": 64.0,
        "return_target_bias_to_completed_return": 0.0,
        "return_target_mse_to_completed_return": 0.0,
        "raw_advantage_mean": 8.0,
        "raw_advantage_std": 0.0,
    }
    np.testing.assert_array_equal(result.arrays["advantages"], [0, 0, 0])


@pytest.mark.parametrize(
    ("rewards", "values"),
    [([], []), ([1], []), ([[1]], [[2]]), ([np.nan], [1]), ([1], [np.inf])],
)
def test_monte_carlo_rejects_incomplete_or_nonfinite_vectors(rewards, values):
    with pytest.raises(ValueError):
        complete_episode_monte_carlo(rewards, values)


def test_monte_carlo_rejects_discounted_objective():
    with pytest.raises(ValueError, match="gamma=1"):
        complete_episode_monte_carlo([0, 10], [1, 2], gamma=0.99)


@pytest.mark.parametrize("failure", ["unterminated", "truncated", "stalled"])
def test_monte_carlo_buffer_rejects_censored_episode(failure):
    settings = config()
    collected = episode(settings)
    if failure == "unterminated":
        collected.outcome["terminated"] = False
    else:
        collected.outcome[failure] = True
    with pytest.raises(ValueError, match="Censored outcomes"):
        buffer(settings, [collected])
