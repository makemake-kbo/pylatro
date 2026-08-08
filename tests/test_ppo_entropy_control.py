import math

import numpy as np
import pytest
import torch

from pylatro_agent.constants import MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM, TOKENIZER_VERSION, ActionRange
from pylatro_agent.reward import RewardConfig
from pylatro_agent.survival import DEFAULT_MAX_ANTES
from pylatro_agent.training.ppo import (
    PPOConfig,
    _actor_transition,
    _critic_warmup_decision,
    _effective_reward_config,
    _entropy_alpha_loss,
    _extract_step_info_value,
    _load_checkpoint_compatible,
    _make_alpha_optimizer,
    _make_policy_optimizer,
    _mean_valid_action_type_count,
    _next_blind_clear_outcome,
    _next_eval_regression_streak,
    _per_state_normalized_entropy,
    _physical_minibatch_count,
    _policy_temperature_for_scalars,
    _ppo_terminal_flags,
    _record_action_diagnostics,
    _RolloutMetrics,
    _run_ppo_update,
    _sample_weighted_mean,
    _set_optimizer_lr_for_phase,
    _smoothed_entropy_signal,
    _validate_ppo_config,
    _write_action_behavior_metrics,
    _write_ante1_metrics,
    _write_risk_calibration_metrics,
    _write_rollout_episode_metrics,
    _write_terminal_loss_metrics,
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


class _TinyAuxPpoModel(_TinyPpoModel):
    def __init__(self) -> None:
        super().__init__()
        self.survival_logits = torch.nn.Parameter(torch.linspace(-0.8, 0.6, DEFAULT_MAX_ANTES))
        self.win_logit = torch.nn.Parameter(torch.tensor(-0.35))

    def action_distribution(
        self,
        tokens: torch.Tensor,
        token_types: torch.Tensor,
        scalars: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[_TinyPpoDistribution, dict[str, torch.Tensor]]:
        del tokens, token_types, scalars, attention_mask
        batch = action_mask.shape[0]
        logits = self.logits.unsqueeze(0).expand(batch, -1)
        return _TinyPpoDistribution(logits, action_mask, temperature), {
            "expected_score": self.value.expand(batch),
            "ante_survival": self.survival_logits.sigmoid().unsqueeze(0).expand(batch, -1),
            "win_prob": self.win_logit.sigmoid().expand(batch),
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


def test_sample_weighted_metrics_are_invariant_to_partial_microbatch_segmentation() -> None:
    full_batch_mean = 32.0 / 352.0
    segmented = _sample_weighted_mean(
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [64, 64, 64, 64, 64, 32],
    )

    assert segmented == pytest.approx(full_batch_mean)
    assert _sample_weighted_mean([full_batch_mean], [352]) == pytest.approx(segmented)


def test_physical_minibatch_count_includes_each_logical_tail() -> None:
    assert _physical_minibatch_count(352, 352, 64) == 6
    assert _physical_minibatch_count(704, 352, 64) == 12
    assert _physical_minibatch_count(752, 352, 64) == 13


def _make_auxiliary_mask_buffer(*, all_zero: bool = False) -> RolloutBuffer:
    sample_count = 11
    action = int(ActionRange.SHOP_LEAVE)
    buffer = _make_signal_buffer(
        actions=[action] * sample_count,
        advantages=[0.0] * sample_count,
    )
    if all_zero:
        return buffer

    survival_mask = np.zeros((sample_count, DEFAULT_MAX_ANTES), dtype=np.float32)
    survival_mask[:4] = 1.0
    survival_mask[4:8, :2] = 1.0
    survival_mask[8:, 0] = 1.0
    survival_target = np.indices(survival_mask.shape).sum(axis=0) % 2
    buffer.ante_survival_masks[:sample_count] = survival_mask
    buffer.ante_survival_targets[:sample_count] = survival_target.astype(np.float32)

    buffer.win_probability_masks[:sample_count] = np.array(
        [1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1],
        dtype=np.float32,
    )
    buffer.win_probability_targets[:sample_count] = np.array(
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        dtype=np.float32,
    )
    return buffer


def _run_auxiliary_update(*, effective_batch_size: int, accum_steps: int, all_zero: bool = False):
    model = _TinyAuxPpoModel()
    optimizer = _make_policy_optimizer(model.parameters(), lr=0.01)
    config = PPOConfig(
        ppo_epochs=1,
        mini_batch_size=11,
        clip_epsilon=0.2,
        entropy_coeff=0.0,
        value_loss_coeff=0.0,
        survival_loss_coeff=1.0,
        win_probability_loss_coeff=1.0,
        critic_updates_trunk=True,
        max_grad_norm=100.0,
        target_kl=None,
        rollout_temperature=1.0,
    )
    captured_grads: dict[str, list[torch.Tensor]] = {name: [] for name, _param in model.named_parameters()}
    handles = []
    for name, parameter in model.named_parameters():
        handles.append(
            parameter.register_hook(lambda grad, key=name: captured_grads[key].append(grad.detach().clone()))
        )

    np.random.seed(12345)
    stats = _run_ppo_update(
        model=model,
        optimizer=optimizer,
        buffer=_make_auxiliary_mask_buffer(all_zero=all_zero),
        return_rms=None,
        entropy_coeff=0.0,
        config=config,
        accum_steps=accum_steps,
        effective_batch_size=effective_batch_size,
        device=torch.device("cpu"),
        use_pin_memory=False,
    )
    for handle in handles:
        handle.remove()
    grad_sums = {
        name: torch.stack(grads).sum(dim=0) if grads else torch.zeros_like(parameter)
        for (name, parameter), grads in zip(model.named_parameters(), captured_grads.values(), strict=True)
    }
    return model, stats, grad_sums


def test_auxiliary_losses_match_unsplit_logical_batch_with_uneven_masks() -> None:
    full_model, full_stats, full_grads = _run_auxiliary_update(effective_batch_size=11, accum_steps=1)
    split_model, split_stats, split_grads = _run_auxiliary_update(effective_batch_size=4, accum_steps=3)

    assert split_stats.sample_counts == [4, 4, 3]
    assert len(set(split_stats.survival_valid_counts)) > 1
    assert len(set(split_stats.win_probability_valid_counts)) > 1
    for name, full_parameter in full_model.named_parameters():
        torch.testing.assert_close(split_model.state_dict()[name], full_parameter, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(split_grads[name], full_grads[name], rtol=1e-6, atol=1e-7)

    full_survival = _sample_weighted_mean(full_stats.survival_losses, full_stats.survival_valid_counts)
    split_survival = _sample_weighted_mean(split_stats.survival_losses, split_stats.survival_valid_counts)
    full_win = _sample_weighted_mean(full_stats.win_probability_losses, full_stats.win_probability_valid_counts)
    split_win = _sample_weighted_mean(split_stats.win_probability_losses, split_stats.win_probability_valid_counts)
    assert split_survival == pytest.approx(full_survival, rel=1e-7, abs=1e-8)
    assert split_win == pytest.approx(full_win, rel=1e-7, abs=1e-8)


def test_auxiliary_losses_are_safe_with_all_zero_valid_masks() -> None:
    model, stats, grads = _run_auxiliary_update(effective_batch_size=4, accum_steps=3, all_zero=True)

    assert stats.survival_valid_counts == [0.0, 0.0, 0.0]
    assert stats.win_probability_valid_counts == [0.0, 0.0, 0.0]
    assert _sample_weighted_mean(stats.survival_losses, stats.survival_valid_counts) == 0.0
    assert _sample_weighted_mean(stats.win_probability_losses, stats.win_probability_valid_counts) == 0.0
    assert torch.equal(model.survival_logits, torch.linspace(-0.8, 0.6, DEFAULT_MAX_ANTES))
    assert torch.equal(model.win_logit, torch.tensor(-0.35))
    assert torch.count_nonzero(grads["survival_logits"]) == 0
    assert torch.count_nonzero(grads["win_logit"]) == 0


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


def test_ppo_config_rejects_invalid_danger_temperature_controls() -> None:
    with pytest.raises(ValueError, match="danger_rollout_temperature"):
        _validate_ppo_config(PPOConfig(danger_rollout_temperature=0.0))
    with pytest.raises(ValueError, match="danger_death_probability_threshold"):
        _validate_ppo_config(PPOConfig(danger_death_probability_threshold=1.1))


def test_policy_temperature_sharpens_only_active_ante1_or_danger_rows() -> None:
    scalars = torch.zeros(5, SCALAR_DIM)
    scalars[:, 2] = torch.tensor([1.0, 2.0, 2.0, 1.0, 2.0])
    scalars[:, 7] = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0])
    scalars[:, 12] = torch.tensor([0.0, 0.8, 0.1, 0.9, 0.9])
    config = PPOConfig(
        rollout_temperature=1.0,
        danger_rollout_temperature=0.85,
        danger_death_probability_threshold=0.35,
    )

    temperatures = _policy_temperature_for_scalars(scalars, config)

    assert isinstance(temperatures, torch.Tensor)
    assert temperatures.tolist() == pytest.approx([0.85, 0.85, 1.0, 1.0, 1.0])


def test_ppo_config_rejects_nonpositive_target_kl_when_set() -> None:
    with pytest.raises(ValueError, match="target_kl"):
        _validate_ppo_config(PPOConfig(target_kl=0.0))


def test_ppo_config_allows_disabled_target_kl() -> None:
    _validate_ppo_config(PPOConfig(target_kl=None))


def test_ppo_config_uses_short_no_progress_backstop() -> None:
    assert PPOConfig().max_no_progress_steps == 32


@pytest.mark.parametrize("tolerance", [0.0, -0.1, 1.1])
def test_ppo_config_rejects_invalid_eval_regression_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="eval_regression_tolerance"):
        _validate_ppo_config(PPOConfig(eval_regression_tolerance=tolerance))


def test_ppo_config_rejects_nonpositive_eval_regression_patience() -> None:
    with pytest.raises(ValueError, match="eval_regression_patience"):
        _validate_ppo_config(PPOConfig(eval_regression_patience=0))


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (PPOConfig(critic_warmup_updates=-1), "critic_warmup_updates"),
        (PPOConfig(critic_warmup_lr=0.0), "critic_warmup_lr"),
        (
            PPOConfig(critic_warmup_updates=2, critic_warmup_min_ev=0.4, critic_warmup_ev_window=1),
            "critic_warmup_ev_window",
        ),
        (
            PPOConfig(critic_warmup_updates=4, critic_warmup_max_updates=3),
            "critic_warmup_max_updates",
        ),
        (PPOConfig(actor_ramp_updates=-1), "actor_ramp_updates"),
        (PPOConfig(actor_ramp_start_clip_fraction=0.0), "actor_ramp_start_clip_fraction"),
        (PPOConfig(ppo_run_uuid="12345678-1234-5678-9234-567812345678"), "must be set together"),
        (
            PPOConfig(
                ppo_run_uuid="not-a-uuid",
                ppo_source_sha256="a" * 64,
                ppo_recipe_id="recipe",
            ),
            "valid UUID",
        ),
        (
            PPOConfig(
                ppo_run_uuid="12345678-1234-5678-9234-567812345678",
                ppo_source_sha256="bad",
                ppo_recipe_id="recipe",
            ),
            "64-character",
        ),
    ],
)
def test_ppo_config_rejects_invalid_safe_unfreeze_controls(config: PPOConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_ppo_config(config)


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


def test_rollout_episode_metrics_are_not_lifetime_averages() -> None:
    class _Writer:
        def __init__(self) -> None:
            self.scalars: dict[str, tuple[float, int]] = {}

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            self.scalars[tag] = (value, step)

    rm = _RolloutMetrics(
        completed_episode_rewards=[-5.0, -3.0],
        completed_episode_lengths=[40, 80],
        completed_episode_wins=[0.0, 1.0],
        completed_episode_stalls=[1.0, 0.0],
        completed_episode_antes=[2, 4],
        completed_episode_tarot_uses=[1, 3],
    )
    writer = _Writer()

    _write_rollout_episode_metrics(writer, rm, step=7)

    assert writer.scalars == {
        "rollout/episode_reward_mean": (-4.0, 7),
        "rollout/episode_length_mean": (60.0, 7),
        "rollout/win_rate": (0.5, 7),
        "rollout/stall_rate": (0.5, 7),
        "rollout/final_ante_mean": (3.0, 7),
        "rollout/mean_ante_reached": (3.0, 7),
        "rollout/tarot_uses_per_completed_episode_mean": (2.0, 7),
    }


def test_rollout_episode_metrics_skip_empty_rollouts() -> None:
    class _Writer:
        def add_scalar(self, *_args) -> None:
            raise AssertionError("empty rollouts must not emit episode metrics")

    _write_rollout_episode_metrics(_Writer(), _RolloutMetrics(), step=1)


def test_action_behavior_metrics_expose_macro_collapse_and_joker_move_loop() -> None:
    class _Writer:
        def __init__(self) -> None:
            self.scalars: dict[str, tuple[float, int]] = {}

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            self.scalars[tag] = (value, step)

    rm = _RolloutMetrics(
        action_type_counts={"move_joker": 80, "shop_leave": 20},
        joker_move_rewards=[0.0, -0.1, 0.02],
        steps_since_progress=[0.0, 1.0, 2.0, 32.0],
    )
    writer = _Writer()

    _write_action_behavior_metrics(writer, rm, step=11)

    assert writer.scalars["actions/type/move_joker_fraction"] == (0.8, 11)
    assert writer.scalars["actions/type/shop_leave_fraction"] == (0.2, 11)
    assert writer.scalars["actions/type/play_subset_fraction"] == (0.0, 11)
    assert writer.scalars["actions/move_joker_non_improving_fraction"] == pytest.approx((2 / 3, 11))
    assert writer.scalars["actions/move_joker_reward_mean"] == pytest.approx((-0.08 / 3, 11))
    assert writer.scalars["rollout/no_progress_streak_p95"] == pytest.approx(
        (np.percentile(rm.steps_since_progress, 95), 11)
    )
    assert writer.scalars["rollout/no_progress_streak_max"] == (32.0, 11)


def test_eval_regression_streak_requires_consecutive_material_drops() -> None:
    streak = _next_eval_regression_streak(
        win_rate=0.50,
        best_win_rate=0.64,
        current_streak=0,
        tolerance=0.10,
    )
    assert streak == 1
    streak = _next_eval_regression_streak(
        win_rate=0.48,
        best_win_rate=0.64,
        current_streak=streak,
        tolerance=0.10,
    )
    assert streak == 2
    assert (
        _next_eval_regression_streak(
            win_rate=0.60,
            best_win_rate=0.64,
            current_streak=streak,
            tolerance=0.10,
        )
        == 0
    )


def test_eval_regression_boundary_is_robust_to_float_roundoff() -> None:
    assert 0.87 - 0.77 < 0.10
    assert (
        _next_eval_regression_streak(
            win_rate=0.77,
            best_win_rate=0.87,
            current_streak=0,
            tolerance=0.10,
        )
        == 1
    )


def test_critic_warmup_requires_minimum_count_and_complete_rolling_ev_window() -> None:
    config = PPOConfig(
        critic_warmup_updates=4,
        critic_warmup_min_ev=0.4,
        critic_warmup_ev_window=3,
        critic_warmup_max_updates=8,
    )

    single_spike = _critic_warmup_decision(
        config,
        completed_updates=4,
        ev_history=[0.9],
    )
    assert single_spike.active
    assert not single_spike.ready

    too_early = _critic_warmup_decision(
        config,
        completed_updates=3,
        ev_history=[0.41, 0.42, 0.43],
    )
    assert too_early.active
    assert not too_early.ready

    ready = _critic_warmup_decision(
        config,
        completed_updates=4,
        ev_history=[0.37, 0.41, 0.42],
    )
    assert ready.ready
    assert not ready.active
    assert ready.rolling_ev == pytest.approx(0.4)


def test_critic_warmup_rejects_nonfinite_or_weak_window_and_fails_closed_at_cap() -> None:
    config = PPOConfig(
        critic_warmup_updates=2,
        critic_warmup_min_ev=0.4,
        critic_warmup_ev_window=3,
        critic_warmup_max_updates=6,
    )

    assert not _critic_warmup_decision(
        config,
        completed_updates=5,
        ev_history=[0.5, float("nan"), 0.7],
    ).ready
    exhausted = _critic_warmup_decision(
        config,
        completed_updates=6,
        ev_history=[0.2, 0.3, 0.39],
    )
    assert exhausted.exhausted
    assert not exhausted.active
    assert not exhausted.ready

    # Readiness wins at the boundary; a healthy critic is not stopped merely
    # because it cleared the gate on the last allowed update.
    boundary_ready = _critic_warmup_decision(
        config,
        completed_updates=6,
        ev_history=[0.39, 0.40, 0.41],
    )
    assert boundary_ready.ready
    assert not boundary_ready.exhausted


def test_count_only_critic_warmup_still_honors_minimum_update_count() -> None:
    config = PPOConfig(
        critic_warmup_updates=3,
        critic_warmup_min_ev=0.0,
        critic_warmup_ev_window=1,
    )
    assert _critic_warmup_decision(config, completed_updates=2, ev_history=[]).active
    assert _critic_warmup_decision(config, completed_updates=3, ev_history=[]).ready


def test_actor_transition_ramps_clip_and_protects_trunk_for_full_window() -> None:
    config = PPOConfig(
        clip_epsilon=0.1,
        actor_ramp_updates=3,
        actor_ramp_start_clip_fraction=0.5,
    )
    assert _actor_transition(config, actor_updates_completed=0) == pytest.approx((0.05, 0.0, True))
    assert _actor_transition(config, actor_updates_completed=1) == pytest.approx((0.075, 0.5, True))
    assert _actor_transition(config, actor_updates_completed=2) == pytest.approx((0.1, 1.0, True))
    assert _actor_transition(config, actor_updates_completed=3) == pytest.approx((0.1, 1.0, False))


def test_terminal_loss_metrics_are_compact_and_numeric() -> None:
    class _Writer:
        def __init__(self) -> None:
            self.scalars: dict[str, tuple[float, int]] = {}

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            self.scalars[tag] = (value, step)

    rm = _RolloutMetrics(
        completed_episode_rewards=[1.0, 2.0, 3.0, 4.0],
        completed_episode_stalls=[0.0, 0.0, 0.0, 1.0],
        terminal_loss_antes=[2, 3, 3],
        terminal_loss_score_ratios=[0.25, 0.5, 0.75],
        terminal_loss_blind_counts={"small": 1, "big": 0, "boss": 2},
        terminal_boss_loss_counts={"bl_hook": 2},
        terminal_loss_last_play_top1=[1.0, 0.0],
        terminal_loss_last_play_value_ratios=[1.0, 0.5],
    )
    writer = _Writer()

    _write_terminal_loss_metrics(writer, rm, step=9, win_ante=4)

    assert writer.scalars["terminal/loss_count"] == (3.0, 9)
    assert writer.scalars["terminal/completed_episode_count"] == (4.0, 9)
    assert writer.scalars["terminal/stall_episode_count"] == (1.0, 9)
    assert writer.scalars["terminal/nonstall_completed_episode_count"] == (3.0, 9)
    assert writer.scalars["terminal/loss_ante/denominator_count"] == (3.0, 9)
    assert writer.scalars["terminal/loss_ante/1_count"] == (0.0, 9)
    assert writer.scalars["terminal/ante1_death_per_nonstall_completed_episode"] == (0.0, 9)
    assert writer.scalars["terminal/loss_ante_mean"] == pytest.approx((8 / 3, 9))
    assert writer.scalars["terminal/loss_ante/1_fraction"] == (0.0, 9)
    assert writer.scalars["terminal/loss_ante/2_fraction"] == pytest.approx((1 / 3, 9))
    assert writer.scalars["terminal/loss_ante/3_fraction"] == pytest.approx((2 / 3, 9))
    assert writer.scalars["terminal/loss_ante/4_fraction"] == (0.0, 9)
    assert writer.scalars["terminal/loss_blind/small_fraction"] == pytest.approx((1 / 3, 9))
    assert writer.scalars["terminal/loss_blind/boss_fraction"] == pytest.approx((2 / 3, 9))
    assert writer.scalars["terminal/loss_score_ratio_mean"] == (0.5, 9)
    assert writer.scalars["terminal/loss_score_ratio_p50"] == (0.5, 9)
    assert writer.scalars["terminal/loss_last_play/top1_fraction"] == (0.5, 9)
    assert writer.scalars["terminal/loss_last_play/legal_top1_fraction"] == (0.5, 9)
    assert writer.scalars["terminal/loss_last_play/value_ratio_mean"] == (0.75, 9)
    assert writer.scalars["terminal/loss_last_play/legal_candidate_value_ratio_mean"] == (0.75, 9)
    assert writer.scalars["terminal/boss_loss/bl_hook_count"] == (2.0, 9)


def test_terminal_loss_metrics_emit_zero_safe_denominators_without_completions() -> None:
    class _Writer:
        def __init__(self) -> None:
            self.scalars: dict[str, tuple[float, int]] = {}

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            self.scalars[tag] = (value, step)

    writer = _Writer()
    _write_terminal_loss_metrics(writer, _RolloutMetrics(), step=3, win_ante=4)

    assert writer.scalars["terminal/completed_episode_count"] == (0.0, 3)
    assert writer.scalars["terminal/nonstall_completed_episode_count"] == (0.0, 3)
    assert writer.scalars["terminal/loss_count"] == (0.0, 3)
    assert writer.scalars["terminal/loss_ante/denominator_count"] == (0.0, 3)
    assert writer.scalars["terminal/loss_ante/1_count"] == (0.0, 3)
    assert writer.scalars["terminal/loss_ante/1_fraction"] == (0.0, 3)
    assert writer.scalars["terminal/ante1_death_per_nonstall_completed_episode"] == (0.0, 3)


def test_terminal_ante1_death_rate_excludes_stalls_from_denominator() -> None:
    class _Writer:
        def __init__(self) -> None:
            self.scalars: dict[str, tuple[float, int]] = {}

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            self.scalars[tag] = (value, step)

    rm = _RolloutMetrics(
        completed_episode_rewards=[0.0, 0.0, 0.0, 0.0],
        completed_episode_stalls=[0.0, 1.0, 0.0, 0.0],
        terminal_loss_antes=[1, 2],
    )
    writer = _Writer()

    _write_terminal_loss_metrics(writer, rm, step=5, win_ante=4)

    assert writer.scalars["terminal/nonstall_completed_episode_count"] == (3.0, 5)
    assert writer.scalars["terminal/loss_ante/1_fraction"] == (0.5, 5)
    assert writer.scalars["terminal/ante1_death_per_nonstall_completed_episode"] == pytest.approx((1 / 3, 5))


def test_terminal_ante1_death_rate_is_zero_with_completions_but_no_losses() -> None:
    class _Writer:
        def __init__(self) -> None:
            self.scalars: dict[str, tuple[float, int]] = {}

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            self.scalars[tag] = (value, step)

    rm = _RolloutMetrics(
        completed_episode_rewards=[1.0, 1.0],
        completed_episode_stalls=[0.0, 0.0],
    )
    writer = _Writer()

    _write_terminal_loss_metrics(writer, rm, step=6, win_ante=4)

    assert writer.scalars["terminal/nonstall_completed_episode_count"] == (2.0, 6)
    assert writer.scalars["terminal/loss_count"] == (0.0, 6)
    assert writer.scalars["terminal/loss_ante/1_fraction"] == (0.0, 6)
    assert writer.scalars["terminal/ante1_death_per_nonstall_completed_episode"] == (0.0, 6)


def test_ante1_metrics_are_scoped_and_publish_mean_denominators() -> None:
    class _Writer:
        def __init__(self) -> None:
            self.scalars: dict[str, tuple[float, int]] = {}

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            self.scalars[tag] = (value, step)

    rm = _RolloutMetrics(
        terminal_loss_antes=[1, 2],
        ante1_death_blind_counts={"small": 1},
        ante1_blind_clear_counts={"small": 2, "big": 1},
        ante1_clear_hands_used=[1.0, 2.0, 3.0],
        ante1_clear_hands_unused=[3.0, 2.0, 1.0],
        ante1_clear_discards_used=[0.0, 1.0, 2.0],
        ante1_play_count=4,
        ante1_play_hand_counts={"Pair": 3, "High Card": 1},
        ante1_play_realized_to_remaining_target=[0.5, 1.5],
        ante1_conservative_chosen_best_ratios=[0.5, 1.0],
        ante1_one_hand_clear_proxy_observed=4,
        ante1_one_hand_clear_proxy_available=2,
        ante1_one_hand_clear_proxy_chosen=1,
        ante1_one_hand_clear_proxy_missed=1,
    )
    writer = _Writer()

    _write_ante1_metrics(writer, rm, step=8)

    assert writer.scalars["ante1/death/count"] == (1.0, 8)
    assert writer.scalars["ante1/blind/small/death_count"] == (1.0, 8)
    assert writer.scalars["ante1/blind/small/clear_count"] == (2.0, 8)
    assert writer.scalars["ante1/clear/count"] == (3.0, 8)
    assert writer.scalars["ante1/clear/hands_used_mean"] == (2.0, 8)
    assert writer.scalars["ante1/clear/hands_unused_mean"] == (2.0, 8)
    assert writer.scalars["ante1/clear/discards_used_mean"] == (1.0, 8)
    assert writer.scalars["ante1/play/realized_progress_count"] == (2.0, 8)
    assert writer.scalars["ante1/play/realized_score_to_remaining_target_mean"] == (1.0, 8)
    assert writer.scalars["ante1/play/conservative_proxy_comparison_count"] == (2.0, 8)
    assert writer.scalars["ante1/play/conservative_chosen_best_ratio_mean"] == (0.75, 8)
    assert writer.scalars["ante1/one_hand_clear_proxy/opportunity_count"] == (4.0, 8)
    assert writer.scalars["ante1/one_hand_clear_proxy/missed_count"] == (1.0, 8)
    assert writer.scalars["ante1/hand_type/denominator_count"] == (4.0, 8)
    assert writer.scalars["ante1/hand_type/pair_share"] == (0.75, 8)
    assert writer.scalars["ante1/hand_type/high_card_share"] == (0.25, 8)


def test_risk_calibration_metrics_report_false_safe_ante1_deaths() -> None:
    class _Writer:
        def __init__(self) -> None:
            self.scalars: dict[str, tuple[float, int]] = {}

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            self.scalars[tag] = (value, step)

    rm = _RolloutMetrics(
        risk_shop_death_predictions=[0.2, 0.8],
        risk_shop_death_outcomes=[1.0, 0.0],
        risk_shop_death_briers=[0.64, 0.64],
        risk_shop_raw_death_predictions=[0.7, 0.9],
        risk_shop_raw_death_briers=[0.49, 0.81],
        risk_ante1_false_safe_deaths=[1.0, 0.0],
    )
    writer = _Writer()

    _write_risk_calibration_metrics(writer, rm, step=7)

    assert writer.scalars["strategy/risk/shop_death_brier"] == pytest.approx((0.64, 7))
    assert writer.scalars["strategy/risk/predicted_death_mean"] == pytest.approx((0.5, 7))
    assert writer.scalars["strategy/risk/actual_death_rate"] == pytest.approx((0.5, 7))
    assert writer.scalars["strategy/risk/death_auc"] == pytest.approx((0.0, 7))
    assert writer.scalars["strategy/risk/raw_shop_death_brier"] == pytest.approx((0.65, 7))
    assert writer.scalars["strategy/risk/raw_predicted_death_mean"] == pytest.approx((0.8, 7))
    assert writer.scalars["strategy/risk/ante1_false_safe_death_fraction"] == pytest.approx((0.5, 7))


def test_next_blind_calibration_does_not_charge_a_later_same_ante_death() -> None:
    assert _next_blind_clear_outcome(
        shop_ante=1,
        shop_blind_index=1,
        final_ante=1,
        won=False,
        terminal_blind="Boss",
    ) == pytest.approx(1.0)
    assert _next_blind_clear_outcome(
        shop_ante=1,
        shop_blind_index=2,
        final_ante=1,
        won=False,
        terminal_blind="Boss",
    ) == pytest.approx(0.0)


def test_record_action_diagnostics_aggregates_hand_and_planet_signals() -> None:
    rm = _RolloutMetrics()
    infos = {
        "ante1_play_observed": np.array([True]),
        "ante1_play_hand": np.array(["Pair"], dtype=object),
        "ante1_play_realized_to_remaining_target": np.array([0.8]),
        "ante1_conservative_chosen_best_ratio": np.array([0.7]),
        "ante1_one_hand_clear_proxy_observed": np.array([True]),
        "ante1_one_hand_clear_proxy_available": np.array([True]),
        "ante1_one_hand_clear_proxy_chosen": np.array([False]),
        "ante1_one_hand_clear_proxy_missed": np.array([True]),
        "ante1_blind_cleared": np.array([True]),
        "ante1_blind_clear_type": np.array(["small"], dtype=object),
        "ante1_blind_clear_hands_used": np.array([2]),
        "ante1_blind_clear_hands_unused": np.array([2]),
        "ante1_blind_clear_discards_used": np.array([1]),
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
        "pack_claim_seal": np.array(["Blue"], dtype=object),
        "purple_seal_tarot_generated_count": np.array([1]),
        "blue_seal_planet_generated_count": np.array([1]),
    }

    _record_action_diagnostics(rm, infos, 0, done=False)

    assert rm.ante1_play_count == 1
    assert rm.ante1_play_hand_counts["Pair"] == 1
    assert rm.ante1_play_realized_to_remaining_target == pytest.approx([0.8])
    assert rm.ante1_conservative_chosen_best_ratios == pytest.approx([0.7])
    assert rm.ante1_one_hand_clear_proxy_available == 1
    assert rm.ante1_one_hand_clear_proxy_chosen == 0
    assert rm.ante1_one_hand_clear_proxy_missed == 1
    assert rm.ante1_blind_clear_counts["small"] == 1
    assert rm.ante1_clear_hands_used == [2.0]
    assert rm.ante1_clear_hands_unused == [2.0]
    assert rm.ante1_clear_discards_used == [1.0]
    assert rm.hand_play_in_candidates == [1.0]
    assert rm.hand_play_top1 == [0.0]
    assert rm.hand_play_top3 == [1.0]
    assert rm.hand_play_value_ratios == pytest.approx([0.75])
    assert rm.hand_chosen_counts["Pair"] == 1
    assert rm.hand_best_counts["Flush"] == 1
    assert rm.planet_use_played_hand == [1.0]
    assert rm.planet_use_main_hand_match == [0.0]
    assert rm.planet_use_key_counts["c_pluto"] == 1
    assert rm.pack_claim_seal_counts["Blue"] == 1
    assert rm.purple_seal_tarots_generated == 1
    assert rm.blue_seal_planets_generated == 1


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
        "hand_plan_post_type": np.array(["Pair"], dtype=object),
        "hand_plan_post_reliability": np.array([0.9]),
        "hand_plan_post_readiness": np.array([1.1]),
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
        "potential_post_seal_value": np.array([0.5]),
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
    assert rm.potential_values["post_seal_value"] == pytest.approx([0.5])
    assert rm.hand_plan_type_counts["Pair"] == 1
    assert rm.hand_plan_reliability == pytest.approx([0.9])
    assert rm.hand_plan_readiness == pytest.approx([1.1])
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
    config = _freeze_test_config(
        lr=0.1,
        critic_updates_trunk=True,
        critic_warmup_updates=1,
        critic_warmup_lr=0.2,
    )
    optimizer = _make_policy_optimizer(model.parameters(), lr=config.lr)

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
    frozen_lr = _set_optimizer_lr_for_phase(optimizer, config, in_critic_warmup=True)
    stats = _run_ppo_update(
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
    assert frozen_lr == pytest.approx(0.2)
    assert stats.actual_lr == pytest.approx(0.2)
    policy_state_after = optimizer.state[model.policy_head.weight]
    assert int(policy_state_after["step"]) == steps_before
    assert torch.equal(policy_state_after["exp_avg"], exp_avg_before)


def test_protected_actor_ramp_keeps_critic_loss_out_of_policy_trunk() -> None:
    """The ramp may train PPO, but critic error must remain value-head-only."""

    action = int(ActionRange.SHOP_LEAVE)
    model = _TinyDecoupledCriticModel()
    torch.nn.init.constant_(model.value_head.weight, 0.5)
    optimizer = _make_policy_optimizer(model.parameters(), lr=0.1)
    config = _freeze_test_config(critic_updates_trunk=True)
    buffer = _make_signal_buffer(actions=[action, action], advantages=[0.0, 0.0])
    buffer.returns[:2] = 10.0
    trunk_before = model.trunk.weight.detach().clone()
    policy_before = model.policy_head.weight.detach().clone()
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
        policy_loss_scale=1.0,
        clip_epsilon=0.05,
        protect_actor_from_critic=True,
    )

    assert torch.equal(model.policy_head.weight.detach(), policy_before)
    assert torch.equal(model.trunk.weight.detach(), trunk_before)
    assert not torch.allclose(model.value_head.weight.detach(), value_before)
