import math

import numpy as np
import pytest
import torch

from pylatro_agent.constants import MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM, TOKENIZER_VERSION, ActionRange
from pylatro_agent.reward import RewardConfig
from pylatro_agent.survival import DEFAULT_MAX_ANTES
from pylatro_agent.training.ppo import (
    PPOConfig,
    _effective_reward_config,
    _entropy_alpha_loss,
    _extract_step_info_value,
    _load_checkpoint_compatible,
    _make_alpha_optimizer,
    _make_policy_optimizer,
    _mean_valid_action_type_count,
    _per_state_normalized_entropy,
    _ppo_terminal_flags,
    _record_action_diagnostics,
    _RolloutMetrics,
    _run_ppo_update,
    _smoothed_entropy_signal,
    _validate_ppo_config,
)
from pylatro_agent.training.rollout_buffer import RolloutBuffer


class _TinyPpoDistribution:
    def __init__(self, logits: torch.Tensor, action_mask: torch.Tensor, temperature: float = 1.0) -> None:
        masked_logits = logits / temperature
        self.dist = torch.distributions.Categorical(logits=masked_logits.masked_fill(action_mask <= 0, -1e8))

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()

    def normalized_action_type_entropy(self) -> torch.Tensor:
        return torch.zeros((), dtype=self.dist.logits.dtype, device=self.dist.logits.device)


class _TinyPpoModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(NUM_ACTIONS))
        self.value = torch.nn.Parameter(torch.zeros(()))

    def action_distribution(
        self,
        tokens: torch.Tensor,
        token_types: torch.Tensor,
        scalars: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[_TinyPpoDistribution, dict[str, torch.Tensor]]:
        batch = action_mask.shape[0]
        logits = self.logits.unsqueeze(0).expand(batch, -1)
        values = self.value.expand(batch)
        survival = torch.full((batch, DEFAULT_MAX_ANTES), 0.5, dtype=logits.dtype, device=logits.device)
        return _TinyPpoDistribution(logits, action_mask, temperature), {
            "expected_score": values,
            "ante_survival": survival,
        }


class _TinyDecoupledCriticModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = torch.nn.Linear(1, 1, bias=False)
        self.policy_head = torch.nn.Linear(1, NUM_ACTIONS, bias=False)
        self.value_head = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.constant_(self.trunk.weight, 1.0)
        torch.nn.init.zeros_(self.policy_head.weight)
        torch.nn.init.zeros_(self.value_head.weight)

    def action_distribution(
        self,
        tokens: torch.Tensor,
        token_types: torch.Tensor,
        scalars: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[_TinyPpoDistribution, dict[str, torch.Tensor]]:
        batch = action_mask.shape[0]
        h = self.trunk(torch.ones(batch, 1, device=action_mask.device))
        logits = self.policy_head(h)
        values = self.value_head(h).squeeze(-1)
        survival = torch.full((batch, DEFAULT_MAX_ANTES), 0.5, dtype=logits.dtype, device=logits.device)
        return _TinyPpoDistribution(logits, action_mask, temperature), {
            "expected_score": values,
            "ante_survival": survival,
        }


def _dummy_obs(num_envs: int) -> dict[str, np.ndarray]:
    action_mask = np.zeros((num_envs, NUM_ACTIONS), dtype=np.float32)
    action_mask[:, int(ActionRange.SHOP_REROLL)] = 1.0
    action_mask[:, int(ActionRange.SHOP_LEAVE)] = 1.0
    return {
        "tokens": np.zeros((num_envs, MAX_SEQ_LEN, TOKEN_DIM), dtype=np.int16),
        "token_types": np.zeros((num_envs, MAX_SEQ_LEN), dtype=np.int8),
        "scalars": np.zeros((num_envs, SCALAR_DIM), dtype=np.float32),
        "attention_mask": np.ones((num_envs, MAX_SEQ_LEN), dtype=np.int8),
        "action_mask": action_mask,
    }


def _make_signal_buffer(
    *,
    actions: list[int],
    advantages: list[float],
) -> RolloutBuffer:
    buffer = RolloutBuffer(num_envs=1, rollout_length=len(actions), gamma=0.99, gae_lambda=0.95)
    obs = _dummy_obs(num_envs=1)
    old_log_prob = np.array([math.log(0.5)], dtype=np.float32)
    for step, action in enumerate(actions):
        buffer.add_batch(
            step=step,
            obs=obs,
            actions=np.array([action], dtype=np.int64),
            rewards=np.array([0.0], dtype=np.float32),
            values=np.array([0.0], dtype=np.float32),
            log_probs=old_log_prob,
            terminated=np.array([False]),
            truncated=np.array([False]),
        )
    buffer.advantages[: len(actions)] = np.asarray(advantages, dtype=np.float32)
    buffer.returns[: len(actions)] = 0.0
    return buffer


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


def test_ppo_update_increases_probability_of_positive_advantage_action() -> None:
    good_action = int(ActionRange.SHOP_LEAVE)
    bad_action = int(ActionRange.SHOP_REROLL)
    model = _TinyPpoModel()
    optimizer = _make_policy_optimizer(model.parameters(), lr=0.1)
    buffer = _make_signal_buffer(
        actions=[good_action, bad_action],
        advantages=[1.0, -1.0],
    )
    config = PPOConfig(
        ppo_epochs=8,
        mini_batch_size=2,
        clip_epsilon=0.2,
        entropy_coeff=0.0,
        value_loss_coeff=0.0,
        survival_loss_coeff=0.0,
        target_kl=None,
        rollout_temperature=1.0,
    )

    obs = _dummy_obs(num_envs=1)
    action_mask = torch.as_tensor(obs["action_mask"])
    with torch.no_grad():
        before = torch.softmax(model.logits.masked_fill(action_mask[0] <= 0, -1e8), dim=-1)[good_action].item()

    stats = _run_ppo_update(
        model=model,
        optimizer=optimizer,
        buffer=buffer,
        return_rms=None,
        entropy_coeff=0.0,
        config=config,
        accum_steps=1,
        effective_batch_size=2,
        device=torch.device("cpu"),
        use_pin_memory=False,
    )

    with torch.no_grad():
        probs = torch.softmax(model.logits.masked_fill(action_mask[0] <= 0, -1e8), dim=-1)
    assert probs[good_action].item() > before
    assert probs[good_action].item() > probs[bad_action].item()
    assert stats.approx_kls
    assert min(stats.approx_kls) >= 0.0


def test_critic_updates_trunk_false_routes_value_loss_to_value_head_only() -> None:
    action = int(ActionRange.SHOP_LEAVE)
    model = _TinyDecoupledCriticModel()
    # Nonzero value head so a critic backward through the trunk would leave a
    # visible gradient — with the zero init the trunk assertion holds vacuously
    # under any critic_updates_trunk setting.
    torch.nn.init.constant_(model.value_head.weight, 0.5)
    optimizer = _make_policy_optimizer(model.parameters(), lr=0.1)
    buffer = _make_signal_buffer(actions=[action, action], advantages=[0.0, 0.0])
    buffer.returns[:2] = 10.0
    config = PPOConfig(
        ppo_epochs=1,
        mini_batch_size=2,
        clip_epsilon=0.2,
        entropy_coeff=0.0,
        value_loss_coeff=1.0,
        survival_loss_coeff=0.0,
        target_kl=None,
        rollout_temperature=1.0,
        critic_updates_trunk=False,
    )

    trunk_before = model.trunk.weight.detach().clone()
    value_before = model.value_head.weight.detach().clone()
    _run_ppo_update(
        model=model,
        optimizer=optimizer,
        buffer=buffer,
        return_rms=None,
        entropy_coeff=0.0,
        config=config,
        accum_steps=1,
        effective_batch_size=2,
        device=torch.device("cpu"),
        use_pin_memory=False,
    )

    assert torch.equal(model.trunk.weight.detach(), trunk_before)
    assert not torch.allclose(model.value_head.weight.detach(), value_before)


def test_load_checkpoint_compatible_rejects_architecture_mismatch(tmp_path) -> None:
    model = torch.nn.Linear(2, 2)
    ckpt = tmp_path / "bad.pt"
    torch.save(
        {
            "tokenizer_version": TOKENIZER_VERSION,
            "state_dict": {
                "weight": torch.zeros(3, 2),
                "bias": torch.zeros(3),
            },
        },
        ckpt,
    )

    with pytest.raises(RuntimeError, match="architecture-incompatible"):
        _load_checkpoint_compatible(model, str(ckpt), torch.device("cpu"))


def test_ppo_config_rejects_nonpositive_rollout_temperature() -> None:
    with pytest.raises(ValueError, match="rollout_temperature"):
        _validate_ppo_config(PPOConfig(rollout_temperature=0.0))


def test_ppo_config_rejects_nonpositive_target_kl_when_set() -> None:
    with pytest.raises(ValueError, match="target_kl"):
        _validate_ppo_config(PPOConfig(target_kl=0.0))


def test_ppo_config_allows_disabled_target_kl() -> None:
    _validate_ppo_config(PPOConfig(target_kl=None))


def test_ppo_config_rejects_negative_counterfactual_interval() -> None:
    with pytest.raises(ValueError, match="counterfactual_diagnostic_interval"):
        _validate_ppo_config(PPOConfig(counterfactual_diagnostic_interval=-1))


def test_mean_normalized_entropy_is_one_for_uniform_binary_policy() -> None:
    probs = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    entropy = torch.distributions.Categorical(probs=probs).entropy()
    action_mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    normalized = _per_state_normalized_entropy(entropy, action_mask).mean()

    assert normalized.item() == pytest.approx(1.0)


def test_mean_normalized_entropy_is_zero_when_only_one_action_is_valid() -> None:
    probs = torch.tensor([[1.0]], dtype=torch.float32)
    entropy = torch.distributions.Categorical(probs=probs).entropy()
    action_mask = torch.tensor([[1.0]], dtype=torch.float32)

    normalized = _per_state_normalized_entropy(entropy, action_mask).mean()

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


def test_effective_reward_config_pins_potential_gamma_without_mutating_input() -> None:
    original = RewardConfig(gamma=0.9)
    effective = _effective_reward_config(PPOConfig(gamma=0.997, reward_config=original))

    assert effective.gamma == pytest.approx(0.997)
    assert original.gamma == pytest.approx(0.9)


def test_stall_is_absorbing_but_ordinary_truncation_bootstraps_from_final_value() -> None:
    terminated = np.array([False, False])
    truncated = np.array([True, True])
    infos = {
        # SAME_STEP live info belongs to the reset observations.
        "stalled": np.array([False, False]),
        "_stalled": np.array([True, True]),
        # The completed episodes' flags live under final_info.
        "final_info": {
            "stalled": np.array([True, False]),
            "_stalled": np.array([True, True]),
        },
    }

    ppo_terminated, ppo_truncated, stalled = _ppo_terminal_flags(
        terminated,
        truncated,
        infos,
    )

    np.testing.assert_array_equal(stalled, [True, False])
    np.testing.assert_array_equal(ppo_terminated, [True, False])
    np.testing.assert_array_equal(ppo_truncated, [False, True])
    # The environment-facing Gymnasium flags remain truncations for both.
    np.testing.assert_array_equal(terminated, [False, False])
    np.testing.assert_array_equal(truncated, [True, True])

    buffer = RolloutBuffer(num_envs=2, rollout_length=1, gamma=0.997, gae_lambda=0.97)
    buffer.add_batch(
        step=0,
        obs=_dummy_obs(num_envs=2),
        actions=np.array([ActionRange.SHOP_LEAVE, ActionRange.SHOP_LEAVE], dtype=np.int64),
        rewards=np.array([1.0, 1.0], dtype=np.float32),
        values=np.array([2.0, 2.0], dtype=np.float32),
        log_probs=np.array([0.0, 0.0], dtype=np.float32),
        terminated=ppo_terminated,
        truncated=ppo_truncated,
        bootstrap_values=np.array([10.0, 10.0], dtype=np.float32),
    )
    buffer.compute_returns_and_advantages(last_values=np.array([99.0, 99.0]))

    assert buffer.returns[0] == pytest.approx(1.0)
    assert buffer.returns[1] == pytest.approx(1.0 + 0.997 * 10.0)


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


def test_record_action_diagnostics_aggregates_joker_build_and_counterfactual_signals() -> None:
    rm = _RolloutMetrics()
    infos = {
        "shop_joker_offer_observed": np.array([True]),
        "shop_offered_joker_emitted_count": np.array([1]),
        "shop_offered_joker_0_id": np.array(["j_hologram"], dtype=object),
        "shop_bought_joker_id": np.array(["j_hologram"], dtype=object),
        "shop_sold_joker_id": np.array(["j_joker"], dtype=object),
        "joker_roster_changed": np.array([True]),
        "joker_acquired_count": np.array([1]),
        "joker_removed_count": np.array([1]),
        "joker_turnover_count": np.array([2]),
        "joker_churn_count": np.array([1]),
        "joker_replacement_event": np.array([True]),
        "joker_acquired_emitted_count": np.array([1]),
        "joker_acquired_0_id": np.array(["j_hologram"], dtype=object),
        "joker_removed_emitted_count": np.array([1]),
        "joker_removed_0_id": np.array(["j_joker"], dtype=object),
        "build_diagnostics_observed": np.array([True]),
        "build_pre_estimated_score": np.array([100.0]),
        "build_post_estimated_score": np.array([150.0]),
        "build_pre_required_score": np.array([120.0]),
        "build_post_required_score": np.array([120.0]),
        "build_pre_readiness": np.array([0.8]),
        "build_post_readiness": np.array([1.25]),
        "build_pre_score_gain_ratio": np.array([1.5]),
        "build_post_score_gain_ratio": np.array([2.0]),
        "build_pre_modeled_fraction": np.array([1.0]),
        "build_post_modeled_fraction": np.array([1.0]),
        "build_estimated_score_delta": np.array([50.0]),
        "build_post_joker_emitted_count": np.array([1]),
        "build_post_joker_0_id": np.array(["j_hologram"], dtype=object),
        "build_post_joker_0_marginal_ratio": np.array([1.5]),
        "build_post_joker_0_modeled_fraction": np.array([1.0]),
        "potential_pre_total": np.array([0.4]),
        "potential_post_total": np.array([0.6]),
        "potential_delta_total": np.array([0.2]),
        "hologram_scaling_count": np.array([1]),
        "hologram_x_mult_delta": np.array([0.25]),
        "hologram_build_score_delta": np.array([50.0]),
        "counterfactual_call": np.array([True]),
        "counterfactual_failure": np.array([False]),
        "counterfactual_focal_joker_id": np.array(["j_hologram"], dtype=object),
        "counterfactual_representative_vs_realized_abs_log_ratio_gap": np.array([0.1]),
        "counterfactual_representative_vs_realized_log_ratio_gap": np.array([-0.1]),
    }

    _record_action_diagnostics(rm, infos, 0, done=False)

    assert rm.shop_offered_joker_counts["j_hologram"] == 1
    assert rm.shop_bought_joker_counts["j_hologram"] == 1
    assert rm.shop_sold_joker_counts["j_joker"] == 1
    assert rm.joker_acquired_id_counts["j_hologram"] == 1
    assert rm.joker_removed_id_counts["j_joker"] == 1
    assert rm.joker_churn_count == 1
    assert rm.joker_marginal_ratios["j_hologram"] == pytest.approx([1.5])
    assert rm.build_values["build_post_readiness"] == pytest.approx([1.25])
    assert rm.potential_values["post_total"] == pytest.approx([0.6])
    assert rm.hologram_x_mult_deltas == pytest.approx([0.25])
    assert rm.counterfactual_calls == 1
    assert rm.counterfactual_failures == 0
    assert rm.counterfactual_representative_realized_abs_gaps == pytest.approx([0.1])


def test_on_policy_advantage_diagnostics_populated() -> None:
    good_action = int(ActionRange.SHOP_LEAVE)
    bad_action = int(ActionRange.SHOP_REROLL)
    model = _TinyPpoModel()
    optimizer = _make_policy_optimizer(model.parameters(), lr=0.01)
    buffer = _make_signal_buffer(
        actions=[good_action, bad_action],
        advantages=[1.0, -1.0],
    )
    config = PPOConfig(
        ppo_epochs=1,
        mini_batch_size=2,
        clip_epsilon=0.2,
        entropy_coeff=0.0,
        value_loss_coeff=0.0,
        survival_loss_coeff=0.0,
        target_kl=None,
        rollout_temperature=1.0,
    )

    stats = _run_ppo_update(
        model=model,
        optimizer=optimizer,
        buffer=buffer,
        return_rms=None,
        entropy_coeff=0.0,
        config=config,
        accum_steps=1,
        effective_batch_size=2,
        device=torch.device("cpu"),
        use_pin_memory=False,
    )

    assert stats.on_policy_advantage_means
    assert stats.on_policy_advantage_stds
    assert stats.on_policy_positive_advantage_fractions
    assert stats.on_policy_return_means
    assert stats.on_policy_fractions[0] == 1.0
    assert stats.on_policy_positive_advantage_fractions[0] == pytest.approx(0.5)


def _freeze_test_config(**overrides) -> PPOConfig:
    base = dict(
        ppo_epochs=1,
        mini_batch_size=2,
        clip_epsilon=0.2,
        entropy_coeff=0.0,
        value_loss_coeff=1.0,
        survival_loss_coeff=0.0,
        target_kl=None,
        rollout_temperature=1.0,
    )
    base.update(overrides)
    return PPOConfig(**base)


def test_critic_warmup_freeze_keeps_policy_bitwise_identical() -> None:
    """policy_loss_scale=0 must be a TRUE freeze.

    Two historical leaks: (1) with critic_updates_trunk=True the critic loss
    backpropagated through the shared trunk and drifted the policy logits;
    (2) the zero-scaled policy backward left zero-valued grads on policy
    params, so restored Adam momentum kept moving them. Both must be dead:
    policy-producing params stay bit-identical and their Adam state untouched,
    while the value head still trains.
    """
    action = int(ActionRange.SHOP_LEAVE)
    model = _TinyDecoupledCriticModel()
    # Non-zero value head so the critic loss reaches the trunk if leaked.
    torch.nn.init.constant_(model.value_head.weight, 0.5)
    optimizer = _make_policy_optimizer(model.parameters(), lr=0.1)
    config = _freeze_test_config(critic_updates_trunk=True)

    # 1) Unfrozen update with signal: builds Adam momentum on policy params.
    warm_buffer = _make_signal_buffer(actions=[action, action], advantages=[1.0, 1.0])
    warm_buffer.returns[:2] = 5.0
    _run_ppo_update(
        model=model,
        optimizer=optimizer,
        buffer=warm_buffer,
        return_rms=None,
        entropy_coeff=0.0,
        config=config,
        accum_steps=1,
        effective_batch_size=2,
        device=torch.device("cpu"),
        use_pin_memory=False,
        policy_loss_scale=1.0,
    )
    policy_state = optimizer.state.get(model.policy_head.weight)
    assert policy_state, "warm update should create Adam state on policy params"
    steps_before = int(policy_state["step"])
    exp_avg_before = policy_state["exp_avg"].clone()
    trunk_before = model.trunk.weight.detach().clone()
    policy_before = model.policy_head.weight.detach().clone()
    value_before = model.value_head.weight.detach().clone()

    # 2) Frozen update with a large value error that would move the trunk if
    #    the critic loss leaked past the value head.
    frozen_buffer = _make_signal_buffer(actions=[action, action], advantages=[1.0, -1.0])
    frozen_buffer.returns[:2] = 10.0
    _run_ppo_update(
        model=model,
        optimizer=optimizer,
        buffer=frozen_buffer,
        return_rms=None,
        entropy_coeff=0.0,
        config=config,
        accum_steps=1,
        effective_batch_size=2,
        device=torch.device("cpu"),
        use_pin_memory=False,
        policy_loss_scale=0.0,
    )

    assert torch.equal(model.policy_head.weight.detach(), policy_before)
    assert torch.equal(model.trunk.weight.detach(), trunk_before)
    assert not torch.allclose(model.value_head.weight.detach(), value_before)
    policy_state_after = optimizer.state[model.policy_head.weight]
    assert int(policy_state_after["step"]) == steps_before
    assert torch.equal(policy_state_after["exp_avg"], exp_avg_before)


