"""Tests for the agent forward pass."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pylatro import load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.constants import MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM, SubPhase
from pylatro_agent.vocab import build_vocab


@pytest.fixture(scope="module")
def game_data():
    return load_game_data()


@pytest.fixture(scope="module")
def vocab(game_data):
    return build_vocab(game_data)


@pytest.fixture(scope="module")
def model(vocab):
    config = AgentConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0)
    return BalatroAgent(config, vocab)


def test_forward_shapes(model):
    batch_size = 2
    tokens = torch.zeros(batch_size, MAX_SEQ_LEN, TOKEN_DIM, dtype=torch.long)
    token_types = torch.zeros(batch_size, MAX_SEQ_LEN, dtype=torch.long)
    scalars = torch.zeros(batch_size, SCALAR_DIM, dtype=torch.float32)
    attention_mask = torch.ones(batch_size, MAX_SEQ_LEN, dtype=torch.long)
    action_mask = torch.ones(batch_size, NUM_ACTIONS, dtype=torch.float32)

    dist, value_dict = model(
        tokens, token_types, scalars, attention_mask, action_mask,
        sub_phase=SubPhase.BLIND_SELECT,
    )

    assert dist.probs.shape == (batch_size, NUM_ACTIONS)
    assert value_dict["win_prob"].shape == (batch_size,)
    assert value_dict["expected_score"].shape == (batch_size,)
    assert value_dict["ante_survival"].shape == (batch_size, 8)


def test_forward_valid_distribution(model):
    batch_size = 1
    tokens = torch.zeros(batch_size, MAX_SEQ_LEN, TOKEN_DIM, dtype=torch.long)
    token_types = torch.zeros(batch_size, MAX_SEQ_LEN, dtype=torch.long)
    scalars = torch.zeros(batch_size, SCALAR_DIM, dtype=torch.float32)
    attention_mask = torch.ones(batch_size, MAX_SEQ_LEN, dtype=torch.long)

    # Only allow blind select actions
    action_mask = torch.zeros(batch_size, NUM_ACTIONS, dtype=torch.float32)
    action_mask[0, 0] = 1  # BLIND_PLAY
    action_mask[0, 1] = 1  # BLIND_SKIP

    dist, _ = model(
        tokens, token_types, scalars, attention_mask, action_mask,
        sub_phase=SubPhase.BLIND_SELECT,
    )

    probs = dist.probs[0]
    # Only masked actions should have non-zero probability
    assert probs[0] > 0
    assert probs[1] > 0
    assert probs[2:].sum() < 1e-5
    assert abs(probs.sum().item() - 1.0) < 1e-4


def test_forward_sample(model):
    batch_size = 4
    tokens = torch.zeros(batch_size, MAX_SEQ_LEN, TOKEN_DIM, dtype=torch.long)
    token_types = torch.zeros(batch_size, MAX_SEQ_LEN, dtype=torch.long)
    scalars = torch.zeros(batch_size, SCALAR_DIM, dtype=torch.float32)
    attention_mask = torch.ones(batch_size, MAX_SEQ_LEN, dtype=torch.long)
    action_mask = torch.ones(batch_size, NUM_ACTIONS, dtype=torch.float32)

    dist, _ = model(
        tokens, token_types, scalars, attention_mask, action_mask,
        sub_phase=SubPhase.CHOOSE_ACTION,
    )

    actions = dist.sample()
    assert actions.shape == (batch_size,)
    log_probs = dist.log_prob(actions)
    assert log_probs.shape == (batch_size,)


def test_parameter_count(vocab):
    config = AgentConfig()  # Full-size model
    model = BalatroAgent(config, vocab)
    params = model.count_parameters()
    assert 5_000_000 < params < 20_000_000, f"Expected ~10M params, got {params:,}"


def test_mixed_subphase_batch(model):
    """Test batch with different sub-phases."""
    batch_size = 3
    tokens = torch.zeros(batch_size, MAX_SEQ_LEN, TOKEN_DIM, dtype=torch.long)
    token_types = torch.zeros(batch_size, MAX_SEQ_LEN, dtype=torch.long)
    scalars = torch.zeros(batch_size, SCALAR_DIM, dtype=torch.float32)
    attention_mask = torch.ones(batch_size, MAX_SEQ_LEN, dtype=torch.long)
    action_mask = torch.ones(batch_size, NUM_ACTIONS, dtype=torch.float32)

    sub_phases = [SubPhase.BLIND_SELECT, SubPhase.CHOOSE_ACTION, SubPhase.SHOP]

    dist, value_dict = model(
        tokens, token_types, scalars, attention_mask, action_mask,
        sub_phase=sub_phases,
    )

    assert dist.probs.shape == (batch_size, NUM_ACTIONS)
    actions = dist.sample()
    assert actions.shape == (batch_size,)
