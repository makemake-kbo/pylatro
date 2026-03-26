"""Smoke tests for the Gymnasium environment."""

from __future__ import annotations

import numpy as np
import pytest

from pylatro import load_game_data
from pylatro_agent.constants import NUM_ACTIONS
from pylatro_agent.env import BalatroEnv
from pylatro_agent.vocab import build_vocab


@pytest.fixture(scope="module")
def game_data():
    return load_game_data()


@pytest.fixture(scope="module")
def vocab(game_data):
    return build_vocab(game_data)


def test_env_reset(game_data, vocab):
    env = BalatroEnv(seed=42, data=game_data, vocab=vocab)
    obs, info = env.reset()

    assert "tokens" in obs
    assert "token_types" in obs
    assert "scalars" in obs
    assert "attention_mask" in obs
    assert "action_mask" in obs
    assert "selected_cards" in obs
    assert obs["tokens"].shape == (160, 12)
    assert obs["action_mask"].shape == (NUM_ACTIONS,)
    assert info["sub_phase"] == "blind_select"


def test_env_action_mask_has_valid_actions(game_data, vocab):
    env = BalatroEnv(seed=42, data=game_data, vocab=vocab)
    obs, _ = env.reset()
    mask = obs["action_mask"]
    assert mask.sum() > 0, "Must have at least one valid action"


def test_env_random_rollout(game_data, vocab):
    """Run random valid actions and ensure no crashes."""
    env = BalatroEnv(seed=42, data=game_data, vocab=vocab, max_steps=500)
    obs, _ = env.reset()
    steps = 0

    for _ in range(500):
        mask = obs["action_mask"]
        valid = np.where(mask == 1)[0]
        if len(valid) == 0:
            break
        action = np.random.choice(valid)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break

    assert steps > 0, "Should have taken at least one step"


def test_env_multiple_resets(game_data, vocab):
    """Ensure environment can be reset multiple times."""
    env = BalatroEnv(seed=1, data=game_data, vocab=vocab, max_steps=100)
    for seed in range(3):
        obs, info = env.reset(seed=seed)
        mask = obs["action_mask"]
        assert mask.sum() > 0

        for _ in range(10):
            valid = np.where(mask == 1)[0]
            if len(valid) == 0:
                break
            action = np.random.choice(valid)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
            mask = obs["action_mask"]
