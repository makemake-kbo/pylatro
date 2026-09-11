"""Critic-only inference preserves values/gradients and excludes actor work."""

from unittest.mock import patch

import pytest
import torch

from pylatro import load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.env import BalatroEnv
from pylatro_agent.training.ppo_observations import _obs_dicts_to_batch
from pylatro_agent.training.ppo_policy import _critic_predictions, _grammar_distribution
from pylatro_agent.vocab import build_vocab


@pytest.mark.parametrize("parallel", [False, True])
def test_critic_forward_matches_full_values_and_terminal_gradients(parallel):
    data = load_game_data()
    vocab = build_vocab(data)
    torch.manual_seed(731)
    base = BalatroAgent(AgentConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64), vocab).eval()
    model = torch.nn.DataParallel(base) if parallel else base
    env = BalatroEnv(seed=1701, data=data, vocab=vocab, enable_teacher=False)
    try:
        obs, _ = env.reset()
    finally:
        env.close()
    batch = _obs_dicts_to_batch([obs, obs], torch.device("cpu"))
    params = [*base.value_head.outcome_proj.parameters(), *base.value_head.ante_survival.parameters()]
    _, full = _grammar_distribution(model, batch)
    reference_gradients = torch.autograd.grad(-full["outcome_probabilities"][:, 0].log().mean(), params)
    with patch.object(base.action_grammar_head, "forward", side_effect=AssertionError("unused policy head")):
        values = _critic_predictions(model, {key: value for key, value in batch.items() if key != "action_mask"})
    for key in full:
        torch.testing.assert_close(values[key], full[key], atol=1e-6, rtol=1e-5)
    gradients = torch.autograd.grad(-values["outcome_probabilities"][:, 0].log().mean(), params, retain_graph=True)
    for actual, expected in zip(gradients, reference_gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    encoder_parameters = [*base.embedding.parameters(), *base.backbone.parameters()]
    encoder_grads = torch.autograd.grad(
        values["outcome_probabilities"][:, 0].sum(),
        encoder_parameters,
        allow_unused=True,
    )
    assert all(grad is None for grad in encoder_grads)
