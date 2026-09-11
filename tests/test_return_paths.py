"""Archive ancestry is verified and learned only through separate actor BC."""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest
import torch

from pylatro import load_game_data
from pylatro_agent.archive import ArchiveConfig, observation_fingerprint
from pylatro_agent.constants import TOKENIZER_VERSION, ActionRange, SubPhase
from pylatro_agent.env import BalatroEnv
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.reward import RewardConfig
from pylatro_agent.training.return_paths import ReturnPathReplay
from pylatro_agent.vocab import build_vocab
from pylatro_cli.controller import GamePhase


@pytest.fixture
def prefix():
    data = load_game_data()
    vocab = build_vocab(data)
    env = BalatroEnv(
        data=data,
        vocab=vocab,
        seed=42,
        enable_teacher=False,
        archive_config=ArchiveConfig(min_ante=1, return_probability=1),
        reward_config=RewardConfig(objective="milestone"),
    )
    env.reset()
    teacher = HeuristicAgent()
    for _ in range(100):
        action = teacher.select_action(
            env.state, env._sub_phase, env.action_masks(), round_score=env._controller.round_score
        )
        obs, _, done, truncated, _ = env.step(action)
        assert not done and not truncated
        if env._sub_phase == SubPhase.SHOP:
            break
    assert env._sub_phase == SubPhase.SHOP
    recipe = {
        "tokenizer_version": TOKENIZER_VERSION,
        "seed": "42",
        "stake": 1,
        "deck_key": "b_red",
        "actions": list(env._action_lineage),
        "boundary_fingerprint": observation_fingerprint(obs),
        "won": True,
        "win_ante": 8,
    }
    yield data, vocab, env, recipe
    env.close()


def test_return_prefix_reconstruction_is_budgeted_verified_and_resumable(prefix):
    data, vocab, _, recipe = prefix
    replay = ReturnPathReplay(data, vocab, capacity=2, seed=4)
    assert replay.enqueue(json.dumps(recipe).encode())
    replay.advance(1)
    assert replay.rebuilt_steps == 1 and not replay.ready
    saved = deepcopy(replay.state_dict())
    resumed = ReturnPathReplay(data, vocab, capacity=2, seed=999)
    resumed.load_state_dict(saved)
    resumed.advance(len(recipe["actions"]))
    assert len(resumed.ready) == 1 and resumed.rejected == 0
    batch = resumed.sample(32, torch.device("cpu"))
    assert batch is not None
    assert int(ActionRange.BLIND_PLAY) in batch["actions"].tolist()
    assert batch["scalars"][:, 2].eq(1).all()  # early, pre-archive actions
    assert "returns" not in batch and "terminal_outcome_target" not in batch
    assert not resumed.enqueue(json.dumps(recipe).encode())  # deduplicate ancestry
    second = ReturnPathReplay(data, vocab, capacity=2)
    second.load_state_dict(deepcopy(resumed.state_dict()))
    left = resumed.sample(8, torch.device("cpu"))
    right = second.sample(8, torch.device("cpu"))
    for key in left:
        torch.testing.assert_close(left[key], right[key])
    replay.close()
    resumed.close()
    second.close()


def test_reconstruction_rejects_mismatched_boundary_and_reserved_seeds(prefix):
    data, vocab, _, recipe = prefix
    reserved = ReturnPathReplay(data, vocab, excluded_seeds=(42,))
    assert not reserved.enqueue(json.dumps(recipe).encode())
    recipe["boundary_fingerprint"] = "0" * 64
    replay = ReturnPathReplay(data, vocab)
    replay.enqueue(json.dumps(recipe).encode())
    replay.advance(100)
    assert replay.rejected == 1 and not replay.ready
    assert replay.sample(8, torch.device("cpu")) is None


def test_worker_emits_only_the_ancestral_prefix_after_a_win(prefix, monkeypatch):
    # The prefix is a real teacher trajectory. Only the terminal win is forced
    # here to test IPC/lineage wiring, not to claim full-run competence.
    _, _, env, expected = prefix
    _, info = env.reset()
    assert info["archive_start"]
    assert env._archive_prefix_length == len(expected["actions"])

    def finish(_action):
        env._controller.phase = GamePhase.GAME_WON
        env.state.won = True

    monkeypatch.setattr(env, "_execute_action", finish)
    _, _, done, _, info = env.step(int(ActionRange.SHOP_LEAVE))
    assert done and info["won"]
    recipe = json.loads(info["winning_return_path"])
    assert recipe["actions"] == expected["actions"]
    assert recipe["boundary_fingerprint"] == expected["boundary_fingerprint"]
    assert env._action_lineage == [*expected["actions"], int(ActionRange.SHOP_LEAVE)]


def test_snapshots_preserve_action_lineage_without_aliasing(prefix):
    data, _, env, _ = prefix
    snapshot = env.snapshot()
    restored = BalatroEnv(data=data, seed=2, enable_teacher=False)
    restored.restore_snapshot(snapshot)
    assert restored._action_lineage == env._action_lineage
    restored._action_lineage.append(0)
    assert snapshot["action_lineage"] == env._action_lineage
    assert len(restored._action_lineage) == len(env._action_lineage) + 1
    np.testing.assert_array_equal(snapshot["controller"].state.seed, env.state.seed)
    restored.close()
