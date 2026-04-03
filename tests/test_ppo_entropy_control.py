import math

import pytest
import torch

from pylatro_agent.training.ppo import (
    _entropy_alpha_loss,
    _make_alpha_optimizer,
    _make_policy_optimizer,
    _mean_normalized_entropy,
    _smoothed_entropy_signal,
)


def test_smoothed_entropy_signal_uses_current_value_first() -> None:
    assert _smoothed_entropy_signal(None, 0.3, 0.9) == 0.3


def test_smoothed_entropy_signal_applies_ema() -> None:
    assert _smoothed_entropy_signal(0.2, 0.5, 0.8) == 0.26


def test_alpha_loss_gradient_decreases_alpha_when_entropy_above_target() -> None:
    log_alpha = torch.tensor(math.log(0.01), dtype=torch.float32, requires_grad=True)
    loss = _entropy_alpha_loss(log_alpha, entropy_signal=0.4, target_entropy=0.25)
    loss.backward()

    assert log_alpha.grad is not None
    assert log_alpha.grad.item() > 0.0


def test_alpha_loss_gradient_increases_alpha_when_entropy_below_target() -> None:
    log_alpha = torch.tensor(math.log(0.01), dtype=torch.float32, requires_grad=True)
    loss = _entropy_alpha_loss(log_alpha, entropy_signal=0.1, target_entropy=0.25)
    loss.backward()

    assert log_alpha.grad is not None
    assert log_alpha.grad.item() < 0.0


def test_alpha_optimizer_step_decreases_alpha_when_entropy_above_target() -> None:
    log_alpha = torch.tensor(math.log(0.01), dtype=torch.float32, requires_grad=True)
    optimizer = _make_alpha_optimizer(log_alpha, lr=3e-4)
    before = log_alpha.exp().item()

    loss = _entropy_alpha_loss(log_alpha, entropy_signal=0.4, target_entropy=0.25)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    after = log_alpha.exp().item()
    assert after < before


def test_alpha_optimizer_step_is_stable_when_entropy_matches_target() -> None:
    log_alpha = torch.tensor(math.log(0.01), dtype=torch.float32, requires_grad=True)
    optimizer = _make_alpha_optimizer(log_alpha, lr=3e-4)
    before = log_alpha.exp().item()

    loss = _entropy_alpha_loss(log_alpha, entropy_signal=0.25, target_entropy=0.25)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    after = log_alpha.exp().item()
    assert after == before


def test_policy_optimizer_uses_adam_without_weight_decay() -> None:
    param = torch.nn.Parameter(torch.tensor(0.0))

    optimizer = _make_policy_optimizer([param], lr=1e-3)

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["weight_decay"] == pytest.approx(0.0)


def test_mean_normalized_entropy_is_one_for_uniform_binary_policy() -> None:
    probs = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    entropy = torch.distributions.Categorical(probs=probs).entropy()
    action_mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    normalized = _mean_normalized_entropy(entropy, action_mask)

    assert normalized.item() == pytest.approx(1.0)


def test_mean_normalized_entropy_is_zero_when_only_one_action_is_valid() -> None:
    probs = torch.tensor([[1.0]], dtype=torch.float32)
    entropy = torch.distributions.Categorical(probs=probs).entropy()
    action_mask = torch.tensor([[1.0]], dtype=torch.float32)

    normalized = _mean_normalized_entropy(entropy, action_mask)

    assert normalized.item() == pytest.approx(0.0)
