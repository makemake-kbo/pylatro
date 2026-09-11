"""Decision-critical observation semantics agree between PT and PPO."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pylatro import load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.constants import ECONOMY_SCALAR_START, TOKEN_DIM, SubPhase
from pylatro_agent.env import BalatroEnv
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.reward import RewardConfig
from pylatro_agent.tokenizer import Tokenizer, sign_log
from pylatro_agent.training.fast_generate import _build_obs
from pylatro_agent.training.fast_runner import FastRunner
from pylatro_agent.vocab import build_vocab


def test_teacher_and_gym_observations_agree_on_real_trajectory():
    data = load_game_data()
    vocab = build_vocab(data)
    env = BalatroEnv(
        data=data, vocab=vocab, seed=42, enable_teacher=False, reward_config=RewardConfig(objective="milestone")
    )
    runner = FastRunner(42, data)
    obs, _ = env.reset()
    teacher = HeuristicAgent()
    phases = set()
    try:
        for _ in range(100):
            phases.add(runner.sub_phase)
            expected = _build_obs(runner, Tokenizer(vocab))
            for key in obs:
                np.testing.assert_allclose(obs[key], expected[key], atol=1e-6, rtol=0, err_msg=key)
            action = teacher.select_action(
                runner.state, runner.sub_phase, runner.compute_mask(), round_score=runner.round_score
            )
            runner.step(action)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        assert {SubPhase.BLIND_SELECT, SubPhase.CHOOSE_ACTION, SubPhase.SHOP} <= phases
    finally:
        env.close()


def test_economy_exposes_cost_without_prescribing_reroll_strategy():
    env = BalatroEnv(seed=42, enable_teacher=False)
    env.reset()
    env.state.dollars = 30
    env.state.current_round.reroll_cost = 5
    cheap = env._tokenizer.tokenize(env.state, SubPhase.SHOP).scalars
    env.state.current_round.reroll_cost = 17
    expensive = env._tokenizer.tokenize(env.state, SubPhase.SHOP).scalars
    offset = ECONOMY_SCALAR_START
    assert cheap[offset] == pytest.approx(sign_log(5))
    assert expensive[offset] == pytest.approx(sign_log(17))
    assert cheap[offset + 2] == pytest.approx(sign_log(25))
    assert expensive[offset + 2] == pytest.approx(sign_log(13))
    assert cheap[offset + 3] == expensive[offset + 3]  # same interest before spending
    assert cheap[offset + 4] > expensive[offset + 4]  # different interest after spending
    env.close()


@pytest.mark.parametrize("column", [6, 7, 8, 10])
def test_card_state_fields_reach_the_embedding(column):
    torch.manual_seed(0)
    model = BalatroAgent(AgentConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64), build_vocab(load_game_data()))
    tokens = torch.zeros(1, 1, TOKEN_DIM, dtype=torch.long)
    initial = model.embedding.deck_emb(tokens)
    tokens[:, :, column] = 1
    assert not torch.equal(initial, model.embedding.deck_emb(tokens))


def test_training_seed_reservation_covers_initial_and_explicit_resets():
    env = BalatroEnv(seed=42, excluded_seeds=(42, 10000), enable_teacher=False)
    env.reset()
    assert env.state.seed not in {"42", "10000"}
    with pytest.raises(ValueError, match="reserved"):
        env.reset(seed=10000)
    env.close()
