"""Tests for the teacher-action reachability masking in PPO distillation.

Background: ``ActionGrammarDistribution.log_prob`` returns a ``-1e8`` floor
for hand actions the structured policy cannot represent in its candidate-hand
slots. Without the masking in ``_safe_distill_loss`` a single such teacher
action could push ``ppo/distill_loss`` into the 17M-21M range and dominate
the entire PPO update.

These tests exercise the masking through the pure helper functions so they
do not need the full model or env.
"""

from __future__ import annotations

import math

import pytest
import torch

from pylatro_agent.constants import NUM_ACTIONS
from pylatro_agent.training.ppo import (
    _safe_distill_loss,
    _teacher_reachability_mask,
)


def _make_inputs(
    *,
    teacher: list[int],
    teacher_lp: list[float],
    action_mask_rows: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the (teacher, teacher_lp, action_mask) triplet the helpers consume."""
    teacher_t = torch.tensor(teacher, dtype=torch.long)
    teacher_lp_t = torch.tensor(teacher_lp, dtype=torch.float32)
    action_mask_t = torch.tensor(action_mask_rows, dtype=torch.float32)
    assert action_mask_t.shape == (len(teacher), NUM_ACTIONS)
    return teacher_t, teacher_lp_t, action_mask_t


def test_valid_reachable_teacher_contributes_to_loss() -> None:
    teacher, teacher_lp, action_mask = _make_inputs(
        teacher=[5],
        teacher_lp=[math.log(0.5)],
        action_mask_rows=[[0] * NUM_ACTIONS],
    )
    action_mask[0, 5] = 1.0

    reachable = _teacher_reachability_mask(teacher, teacher_lp, action_mask)
    assert reachable.tolist() == [1.0]

    distill_weights = torch.tensor([1.0])
    loss, _ = _safe_distill_loss(teacher_lp, reachable, distill_weights)
    assert torch.isfinite(loss)
    # -log(0.5) == log(2) ~= 0.693
    assert loss.item() == pytest.approx(math.log(2.0), rel=1e-5)


def test_teacher_action_invalid_under_action_mask_is_ignored() -> None:
    """teacher=5 with action_mask[5]=0 should be marked unreachable."""
    teacher, teacher_lp, action_mask = _make_inputs(
        teacher=[5],
        teacher_lp=[math.log(0.5)],
        action_mask_rows=[[0] * NUM_ACTIONS],
    )
    # action_mask[5] stays 0 -> teacher action not legal this step.

    reachable = _teacher_reachability_mask(teacher, teacher_lp, action_mask)
    assert reachable.tolist() == [0.0]


def test_minus_1e8_log_prob_is_ignored() -> None:
    """The classic -1e8 floor must be filtered out."""
    teacher, teacher_lp, action_mask = _make_inputs(
        teacher=[5],
        teacher_lp=[-1e8],
        action_mask_rows=[[0] * NUM_ACTIONS],
    )
    action_mask[0, 5] = 1.0  # legal in the env, but not reachable by the policy

    reachable = _teacher_reachability_mask(teacher, teacher_lp, action_mask)
    assert reachable.tolist() == [0.0]

    distill_weights = torch.tensor([1.0])
    loss, _ = _safe_distill_loss(teacher_lp, reachable, distill_weights)
    # valid_count clamps to 1.0, but the masked-out row contributes 0.
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0)


def test_nan_or_inf_log_prob_is_ignored() -> None:
    teacher, teacher_lp, action_mask = _make_inputs(
        teacher=[5, 6],
        teacher_lp=[float("nan"), float("inf")],
        action_mask_rows=[[0] * NUM_ACTIONS, [0] * NUM_ACTIONS],
    )
    action_mask[0, 5] = 1.0
    action_mask[1, 6] = 1.0

    reachable = _teacher_reachability_mask(teacher, teacher_lp, action_mask)
    assert reachable.tolist() == [0.0, 0.0]


def test_all_teacher_actions_ignored_yields_finite_zero_loss() -> None:
    """The historical failure mode: every label dropped -> must NOT be NaN."""
    teacher, teacher_lp, action_mask = _make_inputs(
        teacher=[5, 6],
        teacher_lp=[-1e8, -1e8],
        action_mask_rows=[[0] * NUM_ACTIONS, [0] * NUM_ACTIONS],
    )
    action_mask[0, 5] = 1.0
    action_mask[1, 6] = 1.0

    reachable = _teacher_reachability_mask(teacher, teacher_lp, action_mask)
    assert reachable.sum().item() == 0.0

    distill_weights = torch.tensor([1.0, 1.0])
    loss, _ = _safe_distill_loss(teacher_lp, reachable, distill_weights)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0)


def test_weighted_distill_loss_is_finite_under_floor_contamination() -> None:
    """Mixed batch: one reachable, one -1e8 floor. Loss must stay finite and small."""
    teacher, teacher_lp, action_mask = _make_inputs(
        teacher=[5, 6],
        teacher_lp=[math.log(0.25), -1e8],
        action_mask_rows=[[0] * NUM_ACTIONS, [0] * NUM_ACTIONS],
    )
    action_mask[0, 5] = 1.0
    action_mask[1, 6] = 1.0  # legal in env but unreachable in policy distribution

    reachable = _teacher_reachability_mask(teacher, teacher_lp, action_mask)
    assert reachable.tolist() == [1.0, 0.0]
    assert (reachable.sum() / max(reachable.sum().item(), 1.0)).item() == 1.0

    distill_weights = torch.tensor([1.0, 1.0])
    loss, _ = _safe_distill_loss(teacher_lp, reachable, distill_weights)
    assert torch.isfinite(loss)
    # Only the reachable row contributes, and -log(0.25) ~= 1.386.
    assert loss.item() == pytest.approx(-math.log(0.25), rel=1e-5)
    # Sanity bound: must be far below the historical 17M values.
    assert loss.item() < 100.0


def test_reachability_uses_num_actions_upper_bound() -> None:
    """teacher = NUM_ACTIONS or larger must be treated as out-of-range."""
    teacher = torch.tensor([NUM_ACTIONS], dtype=torch.long)
    teacher_lp = torch.tensor([math.log(0.5)], dtype=torch.float32)
    action_mask = torch.zeros(1, NUM_ACTIONS, dtype=torch.float32)

    reachable = _teacher_reachability_mask(teacher, teacher_lp, action_mask)
    assert reachable.tolist() == [0.0]


def test_loss_clamps_low_log_prob_at_negative_fifty() -> None:
    """A reachable teacher with very low probability contributes a bounded loss."""
    teacher = torch.tensor([5], dtype=torch.long)
    teacher_lp = torch.tensor([-200.0], dtype=torch.float32)
    action_mask = torch.zeros(1, NUM_ACTIONS, dtype=torch.float32)
    action_mask[0, 5] = 1.0

    reachable = _teacher_reachability_mask(teacher, teacher_lp, action_mask)
    assert reachable.tolist() == [1.0]  # finite and above -1e7

    distill_weights = torch.tensor([1.0])
    loss, _ = _safe_distill_loss(teacher_lp, reachable, distill_weights)
    assert torch.isfinite(loss)
    # Clamped at -50, so the NLL contribution is exactly 50.0.
    assert loss.item() == pytest.approx(50.0)
