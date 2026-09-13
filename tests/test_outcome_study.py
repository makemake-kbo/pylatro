from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pylatro_agent.outcome_study import (
    actor_digest,
    complete_game_targets,
    critic_from_pooled,
    outcome_reward,
    pooled_critic_inputs,
    study_protocol,
)
from pylatro_agent.reward import default_reward_components
from pylatro_agent.value_head import ValueHead, outcome_nll


@pytest.mark.parametrize("won", [False, True])
@pytest.mark.parametrize("boss", [0, 1, 7])
def test_outcome_only_reward_ignores_strategic_and_milestone_bonuses(won, boss):
    config = outcome_reward()
    state = SimpleNamespace(win_ante=8)
    previous = {"dollars": 0}
    current = {"dollars": 100, "boss_cleared_ante": boss, "milestone_scale": 1.0}
    live = default_reward_components(state, previous, current, False, won, config)
    terminal = default_reward_components(state, previous, current, True, won, config)
    assert all(value == 0 for value in live.values())
    assert terminal["total"] == terminal["terminal"] == 10 * won
    assert all(value == 0 for key, value in terminal.items() if key not in {"total", "terminal"})


def test_complete_game_labels_credit_every_executed_action():
    returns, targets = complete_game_targets(True, 9, 205)
    np.testing.assert_array_equal(returns, np.full(205, 10))
    np.testing.assert_array_equal(targets, np.full(205, 8))
    returns, targets = complete_game_targets(False, 4, 151)
    assert not returns.any()
    assert (targets == 3).all()
    with pytest.raises(ValueError):
        complete_game_targets(False, 9, 50)


def test_cached_masked_pooling_preserves_production_critic_predictions():
    torch.manual_seed(19)
    head = ValueHead(16, win_only=True)
    encoded = torch.randn(4, 10, 16)
    mask = torch.tensor([[1] * 10, [1, 0] * 5, [0] * 10, [1] + [0] * 9])
    antes = torch.tensor([1, 2, 3, 8])
    original = head(encoded, mask, antes, torch.full_like(antes, 8))
    cached = critic_from_pooled(head, pooled_critic_inputs((encoded, mask), {}), antes)
    for key in original:
        torch.testing.assert_close(original[key], cached[key], rtol=0, atol=0)


def test_critic_fitting_cannot_change_frozen_actor():
    torch.manual_seed(8)
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(3, 16)
    model.actor = torch.nn.Linear(16, 4)
    model.value_head = ValueHead(16, win_only=True)
    before = actor_digest(model)
    features = model.encoder(torch.randn(32, 3)).detach()
    optimizer = torch.optim.Adam(model.value_head.parameters(), lr=0.01)
    for _ in range(3):
        values = critic_from_pooled(model.value_head, features, torch.ones(32))
        loss = outcome_nll(values["outcome_probabilities"], torch.arange(32) % 9)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert actor_digest(model) == before
    assert all(p.grad is None for p in model.encoder.parameters())


def test_study_panels_and_new_budget_are_separate():
    protocol = study_protocol()
    panels = [set(values) for name, values in protocol.items() if name.endswith("_seeds") and name != "sampling_seeds"]
    assert sum(map(len, panels)) == len(set().union(*panels))
    assert protocol["accelerator_limit_seconds"] == 43200
    assert not protocol["teacher_action_supervision"]
    assert not protocol["heuristic_reward_bonuses"]
