import math

import torch

from pylatro_agent.training.ppo import _entropy_alpha_loss, _smoothed_entropy_signal


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
