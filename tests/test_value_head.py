"""Tests for the v8 conditional-outcome critic and PPO advantage clipping."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pylatro_agent.constants import MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM
from pylatro_agent.reward import outcome_value
from pylatro_agent.survival import (
    DEFAULT_MAX_ANTES,
    hazard_outcome_probabilities,
    terminal_outcome_class,
)
from pylatro_agent.training.rollout_buffer import RolloutBuffer
from pylatro_agent.value_head import (
    ValueHead,
    outcome_nll,
    return_huber_loss,
    terminal_outcome_utilities,
)


def _forward(
    head: ValueHead,
    *,
    current_antes: torch.Tensor,
    win_antes: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch = current_antes.numel()
    backbone_out = torch.randn(batch, 6, 32)
    attention_mask = torch.ones(batch, 6)
    return head(backbone_out, attention_mask, current_antes, win_antes)


def test_hazard_outcomes_are_normalized_and_mask_impossible_antes() -> None:
    hazards = torch.full((4, DEFAULT_MAX_ANTES), 0.5)
    current = torch.tensor([1, 2, 4, 7])
    targets = torch.tensor([8, 4, 4, 8])

    probabilities = hazard_outcome_probabilities(hazards, current, targets)

    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(4))
    for row, (current_ante, win_ante) in enumerate(zip(current, targets, strict=True)):
        assert torch.count_nonzero(probabilities[row, : current_ante - 1]) == 0
        assert torch.count_nonzero(
            probabilities[row, win_ante:DEFAULT_MAX_ANTES]
        ) == 0


@pytest.mark.parametrize("bad_ante", [0, DEFAULT_MAX_ANTES + 1])
def test_hazard_outcomes_reject_antes_outside_critic_horizon(bad_ante: int) -> None:
    hazards = torch.full((1, DEFAULT_MAX_ANTES), 0.5)

    with pytest.raises(ValueError, match="current_antes"):
        hazard_outcome_probabilities(
            hazards,
            torch.tensor([bad_ante]),
            torch.tensor([DEFAULT_MAX_ANTES]),
        )
    with pytest.raises(ValueError, match="win_antes"):
        hazard_outcome_probabilities(
            hazards,
            torch.tensor([1]),
            torch.tensor([bad_ante]),
        )
    with pytest.raises(ValueError, match="final_ante"):
        terminal_outcome_class(won=False, final_ante=bad_ante)
    with pytest.raises(ValueError, match="win_antes"):
        terminal_outcome_utilities(torch.tensor([bad_ante]))


def test_win_probability_is_final_outcome_mass_and_aliases_are_exact() -> None:
    head = ValueHead(d_model=32)
    outputs = _forward(
        head,
        current_antes=torch.tensor([1, 3, 5]),
        win_antes=torch.tensor([4, 6, 8]),
    )

    assert outputs["expected_score"] is outputs["expected_return"]
    torch.testing.assert_close(
        outputs["win_prob"], outputs["outcome_probabilities"][:, -1]
    )
    torch.testing.assert_close(
        outputs["expected_return"],
        outputs["terminal_value"] + outputs["return_residual"],
    )


def test_terminal_utilities_match_reward_outcomes_for_every_ante() -> None:
    win_antes = torch.arange(1, DEFAULT_MAX_ANTES + 1)
    utilities = terminal_outcome_utilities(win_antes)

    for row, win_ante in enumerate(win_antes.tolist()):
        for death_ante in range(1, DEFAULT_MAX_ANTES + 1):
            assert utilities[row, death_ante - 1].item() == pytest.approx(
                outcome_value(won=False, ante=death_ante, win_ante=win_ante)
            )
        assert utilities[row, -1].item() == pytest.approx(
            outcome_value(won=True, ante=win_ante, win_ante=win_ante)
        )


def test_return_huber_updates_residual_but_not_hazard_output() -> None:
    head = ValueHead(d_model=32)
    outputs = _forward(
        head,
        current_antes=torch.tensor([1, 2, 3]),
        win_antes=torch.tensor([4, 5, 8]),
    )

    return_huber_loss(outputs, torch.tensor([4.0, -2.0, 1.0])).backward()

    assert head.return_residual.weight.grad is not None
    assert torch.count_nonzero(head.return_residual.weight.grad) > 0
    assert head.ante_survival.weight.grad is None
    assert head.ante_survival.bias.grad is None


def test_outcome_nll_updates_hazards_but_not_residual() -> None:
    head = ValueHead(d_model=32)
    outputs = _forward(
        head,
        current_antes=torch.tensor([1, 2, 3]),
        win_antes=torch.tensor([4, 5, 8]),
    )

    outcome_nll(
        outputs["outcome_probabilities"],
        torch.tensor([0, 4, DEFAULT_MAX_ANTES]),
    ).backward()

    assert head.ante_survival.weight.grad is not None
    assert torch.count_nonzero(head.ante_survival.weight.grad) > 0
    assert head.return_residual.weight.grad is None
    assert head.return_residual.bias.grad is None


def test_outcome_nll_uses_valid_label_denominator() -> None:
    probabilities = torch.tensor(
        [[0.8, 0.2], [0.1, 0.9], [0.25, 0.75]], dtype=torch.float32
    )
    targets = torch.tensor([0, 0, 1])
    mask = torch.tensor([1.0, 0.0, 1.0])

    actual = outcome_nll(probabilities, targets, mask)
    expected = (-torch.log(torch.tensor(0.8)) - torch.log(torch.tensor(0.75))) / 2
    torch.testing.assert_close(actual, expected)
    assert outcome_nll(probabilities, targets, torch.zeros(3)).item() == 0.0


def _dummy_obs(num_envs: int) -> dict[str, np.ndarray]:
    return {
        "tokens": np.zeros((num_envs, MAX_SEQ_LEN, TOKEN_DIM), dtype=np.int16),
        "token_types": np.zeros((num_envs, MAX_SEQ_LEN), dtype=np.int8),
        "scalars": np.zeros((num_envs, SCALAR_DIM), dtype=np.float32),
        "attention_mask": np.ones((num_envs, MAX_SEQ_LEN), dtype=np.int8),
        "action_mask": np.ones((num_envs, NUM_ACTIONS), dtype=np.float32),
    }


def _filled_buffer(rewards: list[float]) -> RolloutBuffer:
    buffer = RolloutBuffer(
        num_envs=1, rollout_length=len(rewards), gamma=0.99, gae_lambda=0.95
    )
    for step, reward in enumerate(rewards):
        buffer.add_batch(
            step=step,
            obs=_dummy_obs(num_envs=1),
            actions=np.array([0], dtype=np.int64),
            rewards=np.array([reward], dtype=np.float32),
            values=np.array([0.0], dtype=np.float32),
            log_probs=np.array([0.0], dtype=np.float32),
            terminated=np.array([step == len(rewards) - 1]),
            truncated=np.array([False]),
        )
    buffer.compute_returns_and_advantages(last_values=np.array([0.0]))
    return buffer


def test_advantage_clip_clamps_outliers() -> None:
    buffer = _filled_buffer([0.01] * 30 + [10.0])
    buffer.normalize_advantages(clip_sigma=2.0)
    assert float(np.abs(buffer.advantages).max()) <= 2.0 + 1e-6


def test_ppo_config_rejects_negative_advantage_clip() -> None:
    from pylatro_agent.training.ppo import PPOConfig, _validate_ppo_config

    with pytest.raises(ValueError, match="advantage_clip_sigma"):
        _validate_ppo_config(PPOConfig(advantage_clip_sigma=-1.0))


@pytest.mark.parametrize("win_ante", [0, DEFAULT_MAX_ANTES + 1])
def test_ppo_config_rejects_win_ante_outside_critic_horizon(win_ante: int) -> None:
    from pylatro_agent.training.ppo import PPOConfig, _validate_ppo_config

    with pytest.raises(ValueError, match="win_ante must be between 1 and 8"):
        _validate_ppo_config(PPOConfig(win_ante=win_ante))
