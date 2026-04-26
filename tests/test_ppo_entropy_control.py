import math

import numpy as np
import pytest
import torch

from pylatro_agent.constants import NUM_ACTIONS, ActionRange
from pylatro_agent.training.ppo import (
    _entropy_alpha_loss,
    _extract_step_info_value,
    _make_alpha_optimizer,
    _make_policy_optimizer,
    _masked_kl_divergence,
    _mean_normalized_action_type_entropy,
    _mean_normalized_entropy,
    _mean_valid_action_type_count,
    _record_action_diagnostics,
    _RolloutMetrics,
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


def test_mean_normalized_action_type_entropy_is_one_for_uniform_type_mass() -> None:
    probs = torch.zeros((1, NUM_ACTIONS), dtype=torch.float32)
    mask = torch.zeros((1, NUM_ACTIONS), dtype=torch.float32)
    probs[0, ActionRange.SHOP_BUY_START] = 0.25
    probs[0, ActionRange.SHOP_BUY_START + 1] = 0.25
    probs[0, ActionRange.SHOP_LEAVE] = 0.5
    mask[0, ActionRange.SHOP_BUY_START] = 1.0
    mask[0, ActionRange.SHOP_BUY_START + 1] = 1.0
    mask[0, ActionRange.SHOP_LEAVE] = 1.0

    normalized = _mean_normalized_action_type_entropy(probs, mask)

    assert normalized.item() == pytest.approx(1.0)


def test_mean_normalized_action_type_entropy_is_zero_with_one_valid_type() -> None:
    probs = torch.zeros((1, NUM_ACTIONS), dtype=torch.float32)
    mask = torch.zeros((1, NUM_ACTIONS), dtype=torch.float32)
    probs[0, ActionRange.PLAY_SUBSET_START] = 0.6
    probs[0, ActionRange.PLAY_SUBSET_START + 1] = 0.4
    mask[0, ActionRange.PLAY_SUBSET_START] = 1.0
    mask[0, ActionRange.PLAY_SUBSET_START + 1] = 1.0

    normalized = _mean_normalized_action_type_entropy(probs, mask)

    assert normalized.item() == pytest.approx(0.0)


def test_mean_valid_action_type_count_counts_distinct_types() -> None:
    mask = torch.zeros((2, NUM_ACTIONS), dtype=torch.float32)
    mask[0, ActionRange.SHOP_BUY_START] = 1.0
    mask[0, ActionRange.SHOP_REROLL] = 1.0
    mask[0, ActionRange.SHOP_LEAVE] = 1.0
    mask[1, ActionRange.PLAY_SUBSET_START] = 1.0
    mask[1, ActionRange.PLAY_SUBSET_START + 1] = 1.0
    mask[1, ActionRange.DISCARD_SUBSET_START] = 1.0

    mean_count = _mean_valid_action_type_count(mask)

    assert mean_count == pytest.approx(2.5)


def test_masked_kl_divergence_is_zero_for_identical_logits() -> None:
    logits = torch.tensor([[2.0, 0.0, -4.0]], dtype=torch.float32)
    mask = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)

    kl = _masked_kl_divergence(logits, logits, mask)

    assert torch.isfinite(kl)
    assert kl.item() == pytest.approx(0.0)


def test_masked_kl_divergence_stays_finite_for_extreme_valid_logits() -> None:
    policy_logits = torch.tensor([[0.0, -120.0]], dtype=torch.float32)
    reference_logits = torch.tensor([[-120.0, 0.0]], dtype=torch.float32)
    mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    kl = _masked_kl_divergence(policy_logits, reference_logits, mask)

    assert torch.isfinite(kl)
    assert kl.item() > 0.0


def test_extract_step_info_value_prefers_final_info_for_done_envs() -> None:
    infos = {
        "reward_total": np.array([0.25, 0.5], dtype=np.float32),
        "_reward_total": np.array([True, True]),
        "final_info": {
            "reward_total": np.array([-9.0, 0.0], dtype=np.float32),
            "_reward_total": np.array([True, False]),
        },
    }

    assert _extract_step_info_value(infos, "reward_total", 0, done=True) == pytest.approx(-9.0)
    assert _extract_step_info_value(infos, "reward_total", 1, done=True) == pytest.approx(0.5)


def test_extract_step_info_value_uses_live_info_for_nonterminal_steps() -> None:
    infos = {
        "progress_made": np.array([True, False]),
        "_progress_made": np.array([True, True]),
    }

    assert _extract_step_info_value(infos, "progress_made", 0, done=False) is True
    assert _extract_step_info_value(infos, "progress_made", 1, done=False) is False


def test_record_action_diagnostics_aggregates_hand_and_planet_signals() -> None:
    rm = _RolloutMetrics()
    infos = {
        "hand_play_observed": np.array([True]),
        "_hand_play_observed": np.array([True]),
        "hand_play_in_candidates": np.array([True]),
        "_hand_play_in_candidates": np.array([True]),
        "hand_play_top1": np.array([False]),
        "_hand_play_top1": np.array([True]),
        "hand_play_top3": np.array([True]),
        "_hand_play_top3": np.array([True]),
        "hand_play_candidate_value_ratio": np.array([0.75], dtype=np.float32),
        "_hand_play_candidate_value_ratio": np.array([True]),
        "hand_play_chosen_hand": np.array(["Pair"], dtype=object),
        "_hand_play_chosen_hand": np.array([True]),
        "hand_play_best_hand": np.array(["Flush"], dtype=object),
        "_hand_play_best_hand": np.array([True]),
        "planet_use_observed": np.array([True]),
        "_planet_use_observed": np.array([True]),
        "planet_use_played_hand": np.array([True]),
        "_planet_use_played_hand": np.array([True]),
        "planet_use_main_hand_match": np.array([False]),
        "_planet_use_main_hand_match": np.array([True]),
        "planet_use_key": np.array(["c_pluto"], dtype=object),
        "_planet_use_key": np.array([True]),
    }

    _record_action_diagnostics(rm, infos, 0, done=False)

    assert rm.hand_play_in_candidates == [1.0]
    assert rm.hand_play_top1 == [0.0]
    assert rm.hand_play_top3 == [1.0]
    assert rm.hand_play_value_ratios == pytest.approx([0.75])
    assert rm.hand_chosen_counts["Pair"] == 1
    assert rm.hand_best_counts["Flush"] == 1
    assert rm.planet_use_played_hand == [1.0]
    assert rm.planet_use_main_hand_match == [0.0]
    assert rm.planet_use_key_counts["c_pluto"] == 1
