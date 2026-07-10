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

from pylatro_agent.training.ppo import (
    PPOConfig,
    _resolve_schedule_total_steps,
    resolve_distill_coeff,
)


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


# --- Schedule pinning across resume -----------------------------------------
#
# Resume recomputes config.total_timesteps as planned_updates * steps_per_update
# with the CURRENT num_envs, so growing 8->32 envs stretches the anneal horizon
# and moves an already-decayed coefficient backward. These tests lock in the
# fix: the horizon captured at the original run's start (schedule_total_steps)
# is persisted in checkpoints and pins the schedule on resume.


def test_pinned_horizon_prevents_resume_rewind_live_run_numbers() -> None:
    """Reproduce the u3000 8->32-env resume: 0.0 must not climb back to ~0.164."""
    cfg = _config(coeff=0.3, floor=0.0, decay_fraction=0.5, total_timesteps=6_144_000)
    original_horizon = cfg.total_timesteps
    # End of the original 8-env leg (update 3000): fully decayed.
    assert resolve_distill_coeff(cfg, total_steps=6_144_000) == pytest.approx(0.0)

    # Resume with 32 envs and +316 updates: total_timesteps becomes 3316 * 8192.
    cfg.total_timesteps = 3_316 * 32 * 256
    # Legacy behavior (no pinned horizon) rewinds the schedule to ~0.164.
    assert resolve_distill_coeff(cfg, total_steps=6_144_000) == pytest.approx(0.164, abs=1e-3)
    # Pinned horizon keeps the coefficient at its decayed value.
    assert (
        resolve_distill_coeff(cfg, 6_144_000, schedule_total_steps=original_horizon)
        == pytest.approx(0.0)
    )


def test_pinned_horizon_is_monotonic_and_continuous_across_resume() -> None:
    cfg = _config(coeff=0.3, floor=0.0, decay_fraction=0.5, total_timesteps=1_000_000)
    original_horizon = cfg.total_timesteps
    resume_step = 300_000
    coeff_at_resume = resolve_distill_coeff(cfg, total_steps=resume_step)

    # Simulate a resume that quadruples the env count / update target.
    cfg.total_timesteps = 4_000_000
    prev = coeff_at_resume
    for step in range(resume_step, 1_200_000, 50_000):
        now = resolve_distill_coeff(cfg, step, schedule_total_steps=original_horizon)
        assert now <= prev + 1e-12, f"coefficient increased at step {step}"
        prev = now
    # Continuity at the resume boundary: same step, same coefficient.
    assert (
        resolve_distill_coeff(cfg, resume_step, schedule_total_steps=original_horizon)
        == pytest.approx(coeff_at_resume)
    )


def test_resolve_schedule_total_steps_fresh_run_uses_config() -> None:
    cfg = _config(coeff=0.3, floor=0.0, total_timesteps=123_456)
    assert _resolve_schedule_total_steps(cfg, resume_state=None) == 123_456


def test_resolve_schedule_total_steps_resume_restores_saved_horizon() -> None:
    cfg = _config(coeff=0.3, floor=0.0, total_timesteps=27_164_672)
    resume_state = {"schedule_total_steps": 6_144_000}
    assert _resolve_schedule_total_steps(cfg, resume_state) == 6_144_000


def test_resolve_schedule_total_steps_reset_flag_reanchors() -> None:
    cfg = _config(coeff=0.3, floor=0.0, total_timesteps=27_164_672)
    cfg.reset_schedules = True
    resume_state = {"schedule_total_steps": 6_144_000}
    assert _resolve_schedule_total_steps(cfg, resume_state) == 27_164_672


def test_resolve_schedule_total_steps_legacy_checkpoint_warns_and_reanchors(caplog) -> None:
    cfg = _config(coeff=0.3, floor=0.0, total_timesteps=27_164_672)
    caplog.set_level("WARNING", logger="pylatro_agent.training.ppo")
    assert _resolve_schedule_total_steps(cfg, resume_state={}) == 27_164_672
    assert any("REWIND" in record.getMessage() for record in caplog.records)
