"""Smoke tests for the Gymnasium environment."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.vector.vector_env import AutoresetMode

from pylatro import load_game_data, populate_shop
from pylatro_agent.constants import NUM_ACTIONS, ActionRange, SubPhase
from pylatro_agent.env import BalatroEnv
from pylatro_agent.training.ppo import _make_vectorized_envs
from pylatro_agent.vocab import build_vocab
from pylatro_cli.controller import GamePhase


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
        obs, _reward, terminated, truncated, _info = env.step(action)
        steps += 1
        if terminated or truncated:
            break

    assert steps > 0, "Should have taken at least one step"


def test_env_multiple_resets(game_data, vocab):
    """Ensure environment can be reset multiple times."""
    env = BalatroEnv(seed=1, data=game_data, vocab=vocab, max_steps=100)
    for seed in range(3):
        obs, _info = env.reset(seed=seed)
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


def test_env_constructor_seed_is_only_used_for_first_reset(game_data, vocab):
    env = BalatroEnv(seed=7, data=game_data, vocab=vocab, max_steps=100)

    _obs1, _ = env.reset()
    first_seed = env.state.seed

    _obs2, _ = env.reset()
    second_seed = env.state.seed

    assert first_seed == "7"
    assert second_seed != first_seed


def test_env_shop_buy_opens_booster_pack_without_index_error(game_data, vocab):
    env = BalatroEnv(seed=42, data=game_data, vocab=vocab)
    env.reset()

    assert env.state is not None
    env.state.dollars = 100
    populate_shop(env.state)
    env._controller.phase = GamePhase.SHOP
    env._sub_phase = SubPhase.SHOP

    obs = env._obs_to_dict(env._build_obs())
    booster_offset = len(env.state.shop.cards) + len(env.state.shop.vouchers)
    action = ActionRange.SHOP_BUY_START + booster_offset

    assert obs["action_mask"][action] == 1

    _, _, terminated, truncated, info = env.step(action)

    assert not terminated
    assert not truncated
    assert "error" not in info
    assert env.state.pack is not None
    assert env._sub_phase == SubPhase.BOOSTER_PACK


def test_env_play_candidate_executes_directly(game_data, vocab):
    env = BalatroEnv(seed=42, data=game_data, vocab=vocab, max_steps=3)
    env.reset()

    _, _, terminated, truncated, info = env.step(ActionRange.BLIND_PLAY)
    assert not terminated
    assert not truncated
    assert info["progress_made"]

    play_action = _first_valid(
        env.action_masks(),
        ActionRange.PLAY_CANDIDATE_START,
        ActionRange.PLAY_CANDIDATE_END,
    )
    _, _, terminated, truncated, info = env.step(play_action)
    assert not terminated
    assert not truncated
    assert info["progress_made"]
    assert env._sub_phase == SubPhase.CHOOSE_ACTION


def test_env_play_candidate_reports_progress(game_data, vocab):
    env = BalatroEnv(seed=42, data=game_data, vocab=vocab, max_steps=3)
    env.reset()

    _, _, _, _, _ = env.step(ActionRange.BLIND_PLAY)
    play_action = _first_valid(
        env.action_masks(),
        ActionRange.PLAY_CANDIDATE_START,
        ActionRange.PLAY_CANDIDATE_END,
    )
    _, _, terminated, truncated, info = env.step(play_action)
    assert not terminated
    assert not truncated
    assert info["progress_made"]
    assert info["steps_since_progress"] == 0
    assert not info["stalled"]


def _first_valid(mask: np.ndarray, start: int, end: int) -> int:
    valid = np.where(mask[start:end + 1] == 1)[0]
    assert len(valid) > 0
    return start + int(valid[0])


def test_vector_env_uses_same_step_autoreset(game_data, vocab):
    vec_env = _make_vectorized_envs(1, game_data, vocab, use_async=False)

    try:
        assert vec_env.metadata["autoreset_mode"] == AutoresetMode.SAME_STEP
    finally:
        vec_env.close()
