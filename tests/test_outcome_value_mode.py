"""The win-only critic baseline must follow its directly supervised mean."""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from pylatro import load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.constants import (
    CURRENT_ANTE_SCALAR_INDEX,
    MAX_SEQ_LEN,
    NUM_ACTIONS,
    SCALAR_DIM,
    TOKEN_DIM,
    WIN_ANTE_SCALAR_INDEX,
)
from pylatro_agent.value_head import ValueHead, outcome_nll, return_huber_loss
from pylatro_agent.vocab import build_vocab


def _fixed_state_head(probability: float, mode: str) -> ValueHead:
    head = ValueHead(8, win_only=True, return_value_mode=mode).double()
    with torch.no_grad():
        head.ante_survival.weight.zero_()
        head.ante_survival.bias.fill_(torch.logit(torch.tensor(probability, dtype=torch.float64)))
        head.return_residual.weight.zero_()
        head.return_residual.bias.fill_(123.0)
    return head


def _fixed_state_outputs(head: ValueHead) -> dict[str, torch.Tensor]:
    return head(
        torch.zeros(5, 1, 8, dtype=torch.float64),
        torch.ones(5, 1, dtype=torch.float64),
        torch.full((5,), 8),
        torch.full((5,), 8),
    )


@pytest.mark.parametrize("probability", [0.2, 0.4, 0.8])
def test_outcome_objective_moves_terminal_baseline_toward_conditional_mean(probability: float) -> None:
    # One observable state, with two wins and three losses: E[return] = 4.
    # The derivative wrt the final survival logit must be p(win) - 0.4.
    head = _fixed_state_head(probability, "terminal_only")
    outputs = _fixed_state_outputs(head)
    outcome_nll(outputs["outcome_probabilities"], torch.tensor([8, 8, 7, 7, 7])).backward()

    assert head.ante_survival.bias.grad is not None
    assert float(head.ante_survival.bias.grad[-1]) == pytest.approx(probability - 0.4, abs=1e-14)
    torch.testing.assert_close(outputs["expected_return"], torch.full((5,), 10 * probability, dtype=torch.float64))
    assert outputs["expected_return"] is outputs["terminal_value"]
    assert outputs["expected_score"] is outputs["expected_return"]
    assert not outputs["return_residual"].requires_grad
    assert torch.count_nonzero(outputs["return_residual"]) == 0
    assert all(parameter.grad is None for parameter in head.return_residual.parameters())
    assert all(parameter.grad is None for parameter in head.pool_proj.parameters())


def test_legacy_huber_correction_moves_away_from_correct_binary_return_mean() -> None:
    head = _fixed_state_head(0.4, "terminal_plus_residual")
    with torch.no_grad():
        head.return_residual.bias.zero_()
    outputs = _fixed_state_outputs(head)
    torch.testing.assert_close(outputs["expected_return"], torch.full((5,), 4.0, dtype=torch.float64))

    return_huber_loss(outputs, torch.tensor([10.0, 10.0, 0.0, 0.0, 0.0], dtype=torch.float64)).backward()

    # Huber estimates a robust location: gradient descent lowers the already
    # correct mean here. This objective mismatch motivates the explicit mode.
    assert head.return_residual.bias.grad is not None
    assert float(head.return_residual.bias.grad.item()) == pytest.approx(0.2)


def test_terminal_only_never_evaluates_residual_tower(monkeypatch: pytest.MonkeyPatch) -> None:
    head = ValueHead(8, win_only=True, return_value_mode="terminal_only")

    def fail_if_evaluated(*_args, **_kwargs):
        raise AssertionError("terminal-only critic evaluated its unused residual tower")

    monkeypatch.setattr(head.pool_proj, "forward", fail_if_evaluated)
    monkeypatch.setattr(head.return_residual, "forward", fail_if_evaluated)
    outputs = head(torch.randn(4, 2, 8), torch.ones(4, 2), torch.tensor([1, 2, 6, 8]))
    torch.testing.assert_close(outputs["expected_return"], 10 * outputs["win_prob"])
    assert bool(((outputs["expected_return"] >= 0) & (outputs["expected_return"] <= 10)).all())


def test_terminal_only_preserves_legacy_parameter_names_and_optimizer_state() -> None:
    legacy = _fixed_state_head(0.4, "terminal_plus_residual")
    optimizer = torch.optim.AdamW(legacy.parameters(), lr=0.01, weight_decay=0.1)
    outputs = _fixed_state_outputs(legacy)
    loss = outcome_nll(outputs["outcome_probabilities"], torch.tensor([8, 8, 7, 7, 7]))
    loss = loss + return_huber_loss(outputs, torch.zeros(5, dtype=torch.float64))
    loss.backward()
    optimizer.step()

    migrated = _fixed_state_head(0.4, "terminal_only")
    assert tuple(dict(legacy.named_parameters())) == tuple(dict(migrated.named_parameters()))
    migrated.load_state_dict(legacy.state_dict(), strict=True)
    resumed_optimizer = torch.optim.AdamW(migrated.parameters(), lr=0.01, weight_decay=0.1)
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    residual_before = {name: value.detach().clone() for name, value in migrated.return_residual.named_parameters()}
    residual_steps = {
        name: resumed_optimizer.state[parameter]["step"].clone()
        for name, parameter in migrated.return_residual.named_parameters()
    }
    resumed_optimizer.zero_grad(set_to_none=True)
    outputs = _fixed_state_outputs(migrated)
    outcome_nll(outputs["outcome_probabilities"], torch.tensor([8, 8, 7, 7, 7])).backward()
    resumed_optimizer.step()

    for name, parameter in migrated.return_residual.named_parameters():
        torch.testing.assert_close(parameter, residual_before[name], rtol=0, atol=0)
        torch.testing.assert_close(resumed_optimizer.state[parameter]["step"], residual_steps[name], rtol=0, atol=0)


def test_agent_config_persists_explicit_value_mode_and_defaults_legacy_metadata() -> None:
    config = AgentConfig(win_only_value=True, return_value_mode="terminal_only")
    assert AgentConfig(**asdict(config)) == config
    legacy_metadata = asdict(config)
    del legacy_metadata["return_value_mode"]
    assert AgentConfig(**legacy_metadata).return_value_mode == "terminal_plus_residual"


@pytest.mark.parametrize("mode", ["bogus", "terminal_only"])
def test_invalid_value_semantics_are_rejected_at_construction(mode: str) -> None:
    with pytest.raises(ValueError, match="return_value_mode"):
        AgentConfig(return_value_mode=mode)
    with pytest.raises(ValueError, match="return_value_mode"):
        ValueHead(8, return_value_mode=mode)


def test_policy_and_critic_only_paths_share_terminal_baseline() -> None:
    config = AgentConfig(
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        dropout=0,
        win_only_value=True,
        return_value_mode="terminal_only",
    )
    model = BalatroAgent(config, build_vocab(load_game_data())).eval()
    scalars = torch.zeros(1, SCALAR_DIM)
    scalars[:, CURRENT_ANTE_SCALAR_INDEX] = 1
    scalars[:, WIN_ANTE_SCALAR_INDEX] = 8
    inputs = (
        torch.zeros(1, MAX_SEQ_LEN, TOKEN_DIM, dtype=torch.long),
        torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long),
        scalars,
        torch.ones(1, MAX_SEQ_LEN, dtype=torch.long),
    )
    with torch.no_grad():
        _, full = model(*inputs, torch.ones(1, NUM_ACTIONS))
        _, critic_only = model(*inputs, critic_only=True)

    assert model.value_head.return_value_mode == "terminal_only"
    for key in full:
        torch.testing.assert_close(full[key], critic_only[key], rtol=0, atol=0)
    torch.testing.assert_close(full["expected_return"], 10 * full["win_prob"])
