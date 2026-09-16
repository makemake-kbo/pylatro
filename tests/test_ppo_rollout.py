"""Observation transport and episode boundaries in the PPO collector."""

import json
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium.vector.vector_env import AutoresetMode

from pylatro import load_game_data
from pylatro_agent.constants import ActionRange
from pylatro_agent.env import BalatroEnv
from pylatro_agent.training import ppo_rollout
from pylatro_agent.training.ppo_config import PPOConfig
from pylatro_agent.training.ppo_observations import (
    _obs_dicts_to_batch,
    _ObsBuffer,
    _observation_buffer_batch,
    _single_obs_to_batch,
)
from pylatro_agent.training.sil import EpisodeReplayBuffer
from pylatro_agent.vocab import build_vocab


def test_async_environment_factories_preserve_worker_configuration():
    data = load_game_data()
    env = ppo_rollout._make_vectorized_envs(
        2,
        data,
        build_vocab(data),
        use_async=True,
        env_seed_base=42,
        deck_key="b_blue",
        max_no_progress_steps=3,
    )
    try:
        env.reset()
        assert env.metadata["autoreset_mode"] == AutoresetMode.SAME_STEP
        assert env.get_attr("_deck_key") == ("b_blue", "b_blue")
        assert env.get_attr("_max_steps") == (3, 3)
        assert env.get_attr("_archive_index") == (0, 1)
        _, _, terminated, truncated, infos = env.step(np.array([ActionRange.BLIND_PLAY] * 2))
        assert not terminated.any() and not truncated.any()
        np.testing.assert_array_equal(infos["origin_seed"], ["42", "43"])
        np.testing.assert_array_equal(infos["teacher_action"], [-1, -1])
    finally:
        env.close()


def test_all_observation_batch_paths_preserve_environment_contract():
    env = BalatroEnv(seed=42, enable_teacher=False)
    try:
        obs, _ = env.reset()
    finally:
        env.close()
    device = torch.device("cpu")
    source = {key: np.full_like(array, index + 1) for index, (key, array) in enumerate(obs.items())}
    single = _single_obs_to_batch(source, device)
    stacked = _obs_dicts_to_batch([source, source], device)
    staging = _ObsBuffer(2, device)
    staging.update({key: np.stack([value, value]) for key, value in source.items()})
    staged = _observation_buffer_batch(staging)
    assert single.keys() == stacked.keys() == staged.keys() == obs.keys()
    floating = {"scalars", "action_mask", "history_event_features", "history_omitted"}
    for key, array in source.items():
        expected = torch.tensor(array, dtype=torch.float32 if key in floating else torch.long)
        torch.testing.assert_close(single[key][0], expected, rtol=0, atol=0)
        torch.testing.assert_close(stacked[key], expected.expand(2, *expected.shape), rtol=0, atol=0)
        torch.testing.assert_close(staged[key], stacked[key], rtol=0, atol=0)
        array.fill(0)
        torch.testing.assert_close(single[key][0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("ending", ["win", "truncation", "stall"])
def test_episode_crosses_rollouts_and_uses_final_info_and_observation(monkeypatch, ending):
    config = PPOConfig(num_envs=1, rollout_length=1, gamma=1, gae_lambda=1)
    obs_buf = _ObsBuffer(1, torch.device("cpu"))

    def observation(value):
        obs = {key: np.zeros_like(array) for key, array in obs_buf.as_numpy_dict().items()}
        obs["scalars"][0, 0] = value
        obs["scalars"][0, 2] = 1
        obs["action_mask"][0, ActionRange.BLIND_PLAY] = 1
        return obs

    def predictions(model, batch, **kwargs):
        return {"expected_return": batch["scalars"][:, 0].clone(), "ante_survival": torch.ones(1, 8)}

    distribution = SimpleNamespace(
        sample=lambda: torch.tensor([ActionRange.BLIND_PLAY]),
        log_prob=lambda actions: torch.tensor([-0.25]),
        action_type_probs=torch.ones(1, 1),
    )
    monkeypatch.setattr(
        ppo_rollout,
        "_grammar_distribution",
        lambda model, batch, **kwargs: (
            distribution,
            predictions(model, batch),
        ),
    )
    monkeypatch.setattr(ppo_rollout, "_critic_predictions", predictions)
    final_obs = {key: array[0] for key, array in observation(7).items()}
    transitions = iter(
        [
            (observation(3), np.array([1.0]), np.array([False]), np.array([False]), {}),
            (
                observation(99),
                np.array([4.0]),
                np.array([ending == "win"]),
                np.array([ending != "win"]),
                {
                    "reward_total": np.array([-999.0]),
                    "won": np.array([False]),
                    "ante": np.array([1]),
                    "final_info": {
                        "reward_total": np.array([4.0]),
                        "reward_terminal": np.array([4.0]),
                        "won": np.array([ending == "win"]),
                        "stalled": np.array([ending == "stall"]),
                        "ante": np.array([3]),
                        "origin_seed": np.array([42]),
                    },
                    "final_obs": [final_obs],
                },
            ),
        ]
    )
    env = SimpleNamespace(step=lambda actions: next(transitions))
    state = ppo_rollout.RolloutState(config)
    replay = EpisodeReplayBuffer(capacity_episodes=4)
    episode_file = StringIO()
    obs_buf.update(observation(2))

    first, _, first_steps = ppo_rollout.collect_rollout(
        torch.nn.Identity(),
        env,
        obs_buf,
        config,
        state,
        replay,
        update_count=0,
    )
    assert first_steps == 1
    np.testing.assert_array_equal(first.returns, [4])
    np.testing.assert_array_equal(first.terminal_outcome_masks, [0])
    assert replay.num_episodes == 0

    second, metrics, second_steps = ppo_rollout.collect_rollout(
        torch.nn.Identity(),
        env,
        obs_buf,
        config,
        state,
        replay,
        update_count=1,
        training_episode_file=episode_file,
    )
    assert second_steps == 1
    np.testing.assert_array_equal(first.scalars[:, 0], [2])
    np.testing.assert_array_equal(second.scalars[:, 0], [3])
    np.testing.assert_array_equal(second.returns, [11 if ending == "truncation" else 4])
    np.testing.assert_array_equal(second.terminal_outcome_masks, [ending != "stall"])
    assert metrics.completed_episode_rewards == [5]
    assert metrics.completed_episode_lengths == [2]
    assert metrics.completed_episode_antes == [3]
    assert state.episode_rewards[0] == state.episode_lengths[0] == 0
    row = json.loads(episode_file.getvalue())
    assert row["seed"] == 42 and row["reward"] == 5 and row["steps"] == 2
    assert row["won"] == (ending == "win")
    if ending == "stall":
        assert replay.num_episodes == 0
    else:
        episode = replay._episodes[0]
        np.testing.assert_array_equal(episode["scalars"][:, 0], [2, 3])
        np.testing.assert_array_equal(episode["returns"], [5, 4])
        np.testing.assert_array_equal(episode["policy_versions"], [0, 1])
