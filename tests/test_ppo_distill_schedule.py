"""Tests for the PPO heuristic-distillation coefficient schedule.

These lock in the fix for the regression where
``--heuristic-distill-coeff 0.0`` still left distillation active because
``heuristic_distill_min`` defaulted to ``0.03``. With the fix, the user-facing
contract is:

* ``heuristic_distill_coeff <= 0`` disables distillation completely; the
  runtime coefficient is exactly ``0.0`` regardless of ``heuristic_distill_min``.
* Otherwise the coefficient decays linearly from ``heuristic_distill_coeff``
  toward ``min(heuristic_distill_min, heuristic_distill_coeff)`` over
  ``distill_decay_fraction`` of training, never falling below the floor.
"""

from __future__ import annotations

import pytest

from pylatro_agent.training.ppo import PPOConfig, resolve_distill_coeff


def _config(
    *,
    coeff: float,
    floor: float,
    decay_fraction: float | None = None,
    total_timesteps: int = 10_000,
) -> PPOConfig:
    return PPOConfig(
        heuristic_distill_coeff=coeff,
        heuristic_distill_min=floor,
        distill_decay_fraction=decay_fraction,
        total_timesteps=total_timesteps,
    )


def test_zero_coeff_disables_distillation_with_default_min() -> None:
    """coeff=0.0 must beat the default heuristic_distill_min=0.03 floor."""
    cfg = _config(coeff=0.0, floor=0.03)
    assert resolve_distill_coeff(cfg, total_steps=0) == 0.0
    assert resolve_distill_coeff(cfg, total_steps=5_000) == 0.0
    assert resolve_distill_coeff(cfg, total_steps=10_000) == 0.0


def test_zero_coeff_disables_distillation_with_zero_min() -> None:
    cfg = _config(coeff=0.0, floor=0.0)
    assert resolve_distill_coeff(cfg, total_steps=0) == 0.0
    assert resolve_distill_coeff(cfg, total_steps=10_000) == 0.0


def test_negative_coeff_disables_distillation() -> None:
    """Negative coefficients are treated the same as zero."""
    cfg = _config(coeff=-1.0, floor=0.03)
    assert resolve_distill_coeff(cfg, total_steps=0) == 0.0


def test_normal_decay_never_below_floor() -> None:
    cfg = _config(coeff=0.1, floor=0.03)
    start = resolve_distill_coeff(cfg, total_steps=0)
    mid = resolve_distill_coeff(cfg, total_steps=5_000)
    end = resolve_distill_coeff(cfg, total_steps=10_000)
    after = resolve_distill_coeff(cfg, total_steps=20_000)

    assert start == pytest.approx(0.1)
    assert mid == pytest.approx(0.065)
    assert end == pytest.approx(0.03)
    assert after == pytest.approx(0.03)


def test_floor_equal_to_coeff_is_constant() -> None:
    cfg = _config(coeff=0.03, floor=0.03)
    for steps in (0, 100, 5_000, 10_000, 100_000):
        assert resolve_distill_coeff(cfg, total_steps=steps) == pytest.approx(0.03)


def test_floor_above_coeff_is_clamped_to_coeff(caplog) -> None:
    """If min > coeff we clamp the floor to coeff rather than raising the start."""
    cfg = _config(coeff=0.01, floor=0.03)
    # First call may emit a warning; either way the schedule must be constant 0.01.
    caplog.set_level("WARNING", logger="pylatro_agent.training.ppo")
    for steps in (0, 100, 5_000, 10_000):
        assert resolve_distill_coeff(cfg, total_steps=steps) == pytest.approx(0.01)
    # Subsequent calls should not re-warn (idempotent).
    resolve_distill_coeff(cfg, total_steps=6_000)
    warn_records = [r for r in caplog.records if r.levelname == "WARNING"]
    assert 0 <= len(warn_records) <= 1


def test_distill_decay_fraction_shortens_decay_window() -> None:
    cfg = _config(coeff=0.1, floor=0.0, decay_fraction=0.5, total_timesteps=10_000)
    assert resolve_distill_coeff(cfg, total_steps=0) == pytest.approx(0.1)
    # decay_fraction=0.5 means full decay completes at step 5_000.
    assert resolve_distill_coeff(cfg, total_steps=5_000) == pytest.approx(0.0)
    assert resolve_distill_coeff(cfg, total_steps=10_000) == pytest.approx(0.0)
