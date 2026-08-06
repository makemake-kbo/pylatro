"""Tests for the structured agent forward pass."""

from __future__ import annotations

import pytest
import torch

from pylatro import load_game_data
from pylatro_agent.action_grammar import NUM_GRAMMAR_ACTIONS, ActionGrammarDistribution
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.constants import MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM
from pylatro_agent.training.ppo import _grammar_distribution
from pylatro_agent.vocab import build_vocab


@pytest.fixture(scope="module")
def model() -> BalatroAgent:
    vocab = build_vocab(load_game_data())
    config = AgentConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0)
    return BalatroAgent(config, vocab)


def _inputs(batch_size: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros(batch_size, MAX_SEQ_LEN, TOKEN_DIM, dtype=torch.long),
        torch.zeros(batch_size, MAX_SEQ_LEN, dtype=torch.long),
        torch.zeros(batch_size, SCALAR_DIM, dtype=torch.float32),
        torch.ones(batch_size, MAX_SEQ_LEN, dtype=torch.long),
        torch.ones(batch_size, NUM_ACTIONS, dtype=torch.float32),
    )


def test_forward_returns_structured_distribution_and_values(model: BalatroAgent) -> None:
    distribution, values = model(*_inputs(2))

    assert isinstance(distribution, ActionGrammarDistribution)
    assert distribution.action_type_probs.shape == (2, NUM_GRAMMAR_ACTIONS)
    assert values["win_prob"].shape == (2,)
    assert values["expected_score"].shape == (2,)
    assert values["ante_survival"].shape == (2, 8)


def test_distribution_samples_only_valid_actions(model: BalatroAgent) -> None:
    tokens, token_types, scalars, attention_mask, action_mask = _inputs(64)
    action_mask.zero_()
    action_mask[:, 0] = 1
    action_mask[:, 1] = 1

    distribution, _ = model(
        tokens,
        token_types,
        scalars,
        attention_mask,
        action_mask,
    )
    actions = distribution.sample()

    assert actions.shape == (64,)
    assert set(actions.tolist()) <= {0, 1}
    assert distribution.log_prob(actions).shape == (64,)


def test_named_distribution_entry_point_matches_forward_contract(
    model: BalatroAgent,
) -> None:
    distribution, values = model.action_distribution(*_inputs(3))

    assert isinstance(distribution, ActionGrammarDistribution)
    assert distribution.sample().shape == (3,)
    assert values["expected_score"].shape == (3,)


def test_data_parallel_path_uses_tensor_outputs_and_rebuilds_distribution(
    model: BalatroAgent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapped = torch.nn.DataParallel(model)
    tokens, token_types, scalars, attention_mask, action_mask = _inputs(2)
    batch = {
        "tokens": tokens,
        "token_types": token_types,
        "scalars": scalars,
        "attention_mask": attention_mask,
        "action_mask": action_mask,
    }

    def fail_if_unwrapped(*_args, **_kwargs):
        raise AssertionError("DataParallel path bypassed model.forward")

    monkeypatch.setattr(model, "action_distribution", fail_if_unwrapped)
    distribution, values = _grammar_distribution(wrapped, batch)

    assert isinstance(distribution, ActionGrammarDistribution)
    assert distribution.action_type_probs.shape == (2, NUM_GRAMMAR_ACTIONS)
    assert values["expected_score"].shape == (2,)


def test_removed_flat_heads_are_not_registered(model: BalatroAgent) -> None:
    assert not hasattr(model, "blind_head")
    assert not hasattr(model, "choose_head")
    assert not hasattr(model, "shop_head")
    assert not hasattr(model, "booster_head")
