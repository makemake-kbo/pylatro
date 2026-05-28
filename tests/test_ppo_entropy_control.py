import math

import numpy as np
import pytest
import torch

from pylatro_agent.constants import MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM, TOKENIZER_VERSION, ActionRange
from pylatro_agent.survival import DEFAULT_MAX_ANTES
from pylatro_agent.training.ppo import (
    PPOConfig,
    _entropy_alpha_loss,
    _extract_step_info_value,
    _load_checkpoint_compatible,
    _make_alpha_optimizer,
    _make_policy_optimizer,
    _masked_kl_divergence,
    _mean_normalized_action_type_entropy,
    _mean_normalized_entropy,
    _mean_valid_action_type_count,
    _record_action_diagnostics,
    _RolloutMetrics,
    _run_dagger_bc_update,
    _run_ppo_update,
    _scheduled_teacher_rollout_prob,
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
    teacher_actions: list[int] | None = None,
    teacher_forced: list[bool] | None = None,
) -> RolloutBuffer:
    buffer = RolloutBuffer(num_envs=1, rollout_length=len(actions), gamma=0.99, gae_lambda=0.95)
    obs = _dummy_obs(num_envs=1)
    old_log_prob = np.array([math.log(0.5)], dtype=np.float32)
    for step, action in enumerate(actions):
        teacher = None
        if teacher_actions is not None:
            teacher = np.array([teacher_actions[step]], dtype=np.int64)
        forced = None
        if teacher_forced is not None:
            forced = np.array([teacher_forced[step]], dtype=np.bool_)
        buffer.add_batch(
            step=step,
            obs=obs,
            actions=np.array([action], dtype=np.int64),
            rewards=np.array([0.0], dtype=np.float32),
            values=np.array([0.0], dtype=np.float32),
            log_probs=old_log_prob,
            terminated=np.array([False]),
            truncated=np.array([False]),
            teacher_actions=teacher,
            teacher_forced=forced,
        )
    buffer.advantages[:len(actions)] = np.asarray(advantages, dtype=np.float32)
    buffer.returns[:len(actions)] = 0.0
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
        teacher_actions=[-1, -1],
    )
    config = PPOConfig(
        ppo_epochs=8,
        mini_batch_size=2,
        clip_epsilon=0.2,
        entropy_coeff=0.0,
        value_loss_coeff=0.0,
        survival_loss_coeff=0.0,
        heuristic_distill_coeff=0.0,
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
        distill_coeff=0.0,
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


def test_heuristic_distillation_increases_teacher_action_probability_without_advantage() -> None:
    teacher_action = int(ActionRange.SHOP_LEAVE)
    sampled_action = int(ActionRange.SHOP_REROLL)
    model = _TinyPpoModel()
    optimizer = _make_policy_optimizer(model.parameters(), lr=0.1)
    buffer = _make_signal_buffer(
        actions=[sampled_action, sampled_action],
        advantages=[0.0, 0.0],
        teacher_actions=[teacher_action, teacher_action],
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
        before = torch.softmax(model.logits.masked_fill(action_mask[0] <= 0, -1e8), dim=-1)[teacher_action].item()

    stats = _run_ppo_update(
        model=model,
        optimizer=optimizer,
        buffer=buffer,
        return_rms=None,
        entropy_coeff=0.0,
        distill_coeff=1.0,
        config=config,
        accum_steps=1,
        effective_batch_size=2,
        device=torch.device("cpu"),
        use_pin_memory=False,
    )

    with torch.no_grad():
        probs = torch.softmax(model.logits.masked_fill(action_mask[0] <= 0, -1e8), dim=-1)
    assert probs[teacher_action].item() > before
    assert probs[teacher_action].item() > probs[sampled_action].item()
    assert stats.distill_losses[0] > stats.distill_losses[-1]


def test_teacher_forced_samples_do_not_drive_ppo_policy_loss() -> None:
    good_action = int(ActionRange.SHOP_LEAVE)
    bad_action = int(ActionRange.SHOP_REROLL)
    model = _TinyPpoModel()
    optimizer = _make_policy_optimizer(model.parameters(), lr=0.1)
    buffer = _make_signal_buffer(
        actions=[good_action, bad_action],
        advantages=[1.0, -1.0],
        teacher_actions=[-1, -1],
        teacher_forced=[True, True],
    )
    config = PPOConfig(
        ppo_epochs=8,
        mini_batch_size=2,
        clip_epsilon=0.2,
        entropy_coeff=0.0,
        value_loss_coeff=0.0,
        survival_loss_coeff=0.0,
        heuristic_distill_coeff=0.0,
        target_kl=None,
        rollout_temperature=1.0,
    )

    before = model.logits.detach().clone()
    stats = _run_ppo_update(
        model=model,
        optimizer=optimizer,
        buffer=buffer,
        return_rms=None,
        entropy_coeff=0.0,
        distill_coeff=0.0,
        config=config,
        accum_steps=1,
        effective_batch_size=2,
        device=torch.device("cpu"),
        use_pin_memory=False,
    )

    assert torch.allclose(model.logits.detach(), before)
    assert stats.on_policy_fractions
    assert max(stats.on_policy_fractions) == 0.0


def test_dagger_bc_update_increases_teacher_action_probability() -> None:
    teacher_action = int(ActionRange.SHOP_LEAVE)
    sampled_action = int(ActionRange.SHOP_REROLL)
    model = _TinyPpoModel()
    optimizer = _make_policy_optimizer(model.parameters(), lr=0.1)
    buffer = _make_signal_buffer(
        actions=[sampled_action, sampled_action],
        advantages=[0.0, 0.0],
        teacher_actions=[teacher_action, teacher_action],
        teacher_forced=[True, True],
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
        dagger_bc_epochs=4,
        dagger_bc_coeff=1.0,
        dagger_bc_lr_mult=3.0,
    )

    obs = _dummy_obs(num_envs=1)
    action_mask = torch.as_tensor(obs["action_mask"])
    with torch.no_grad():
        before = torch.softmax(model.logits.masked_fill(action_mask[0] <= 0, -1e8), dim=-1)[teacher_action].item()

    losses, _matches = _run_dagger_bc_update(
        model=model,
        optimizer=optimizer,
        buffer=buffer,
        coeff=1.0,
        config=config,
        accum_steps=1,
        effective_batch_size=2,
        device=torch.device("cpu"),
        use_pin_memory=False,
    )

    with torch.no_grad():
        probs = torch.softmax(model.logits.masked_fill(action_mask[0] <= 0, -1e8), dim=-1)
    assert probs[teacher_action].item() > before
    assert probs[teacher_action].item() > probs[sampled_action].item()
    assert losses[0] > losses[-1]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_ppo_value_loss_updates_value_head_not_shared_trunk() -> None:
    action = int(ActionRange.SHOP_LEAVE)
    model = _TinyDecoupledCriticModel()
    optimizer = _make_policy_optimizer(model.parameters(), lr=0.1)
    buffer = _make_signal_buffer(actions=[action, action], advantages=[0.0, 0.0], teacher_actions=[-1, -1])
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
    )

    trunk_before = model.trunk.weight.detach().clone()
    value_before = model.value_head.weight.detach().clone()
    _run_ppo_update(
        model=model,
        optimizer=optimizer,
        buffer=buffer,
        return_rms=None,
        entropy_coeff=0.0,
        distill_coeff=0.0,
        config=config,
        accum_steps=1,
        effective_batch_size=2,
        device=torch.device("cpu"),
        use_pin_memory=False,
    )

    assert torch.allclose(model.trunk.weight.detach(), trunk_before)
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


def test_ppo_config_rejects_teacher_rollout_prob_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="teacher_rollout_prob"):
        _validate_ppo_config(PPOConfig(teacher_rollout_prob=1.1))


def test_teacher_rollout_schedule_warms_up_then_decays() -> None:
    config = PPOConfig(
        teacher_rollout_prob=1.0,
        teacher_rollout_final_prob=0.25,
        teacher_rollout_warmup_fraction=0.2,
        teacher_rollout_decay_fraction=0.5,
    )

    assert _scheduled_teacher_rollout_prob(config, 0.1) == pytest.approx(1.0)
    assert _scheduled_teacher_rollout_prob(config, 0.45) == pytest.approx(0.625)
    assert _scheduled_teacher_rollout_prob(config, 0.9) == pytest.approx(0.25)


def test_ppo_config_rejects_nonpositive_dagger_lr_multiplier() -> None:
    with pytest.raises(ValueError, match="dagger_bc_lr_mult"):
        _validate_ppo_config(PPOConfig(dagger_bc_lr_mult=0.0))


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


def test_dagger_before_ppo_inflates_kl_beyond_target() -> None:
    """Regression test: DAgger BC updates the model before PPO runs,
    making old_log_probs stale and inflating the KL ratio.

    When DAgger runs first, the model changes and PPO's importance
    ratio diverges from 1.0, causing approx_kl >> target_kl and
    early stopping after 1 mini-batch. With the fix (PPO first),
    KL stays near 0 because old_log_probs are fresh."""
    teacher_action = int(ActionRange.SHOP_LEAVE)
    sampled_action = int(ActionRange.SHOP_REROLL)
    model = _TinyPpoModel()
    optimizer = _make_policy_optimizer(model.parameters(), lr=0.1)
    buffer = _make_signal_buffer(
        actions=[sampled_action, sampled_action],
        advantages=[1.0, -1.0],
        teacher_actions=[teacher_action, teacher_action],
        teacher_forced=[False, False],
    )
    config = PPOConfig(
        ppo_epochs=4,
        mini_batch_size=2,
        clip_epsilon=0.2,
        entropy_coeff=0.0,
        value_loss_coeff=0.0,
        survival_loss_coeff=0.0,
        heuristic_distill_coeff=0.0,
        target_kl=0.03,
        rollout_temperature=1.0,
        dagger_bc_epochs=4,
        dagger_bc_coeff=1.0,
        dagger_bc_lr_mult=3.0,
    )
    device = torch.device("cpu")

    ppo_stats = _run_ppo_update(
        model=model,
        optimizer=optimizer,
        buffer=buffer,
        return_rms=None,
        entropy_coeff=0.0,
        distill_coeff=0.0,
        config=config,
        accum_steps=1,
        effective_batch_size=2,
        device=device,
        use_pin_memory=False,
    )

    assert ppo_stats.approx_kls, "PPO should produce at least one KL measurement"
    mean_kl = float(np.mean(ppo_stats.approx_kls))
    assert mean_kl < 0.1, (
        f"PPO KL after running BEFORE DAgger should be small, got {mean_kl:.4f}. "
        "If DAgger ran first, old_log_probs would be stale and KL would be huge."
    )
    assert ppo_stats.ppo_minibatches_processed
    total_minibatches = int(np.sum(ppo_stats.ppo_minibatches_processed))
    assert total_minibatches >= 2, (
        f"PPO should process multiple mini-batches when KL is controlled, "
        f"got {total_minibatches}"
    )

    _run_dagger_bc_update(
        model=model,
        optimizer=optimizer,
        buffer=buffer,
        coeff=config.dagger_bc_coeff,
        config=config,
        accum_steps=1,
        effective_batch_size=2,
        device=device,
        use_pin_memory=False,
    )

    obs = _dummy_obs(num_envs=1)
    action_mask = torch.as_tensor(obs["action_mask"])
    with torch.no_grad():
        probs = torch.softmax(model.logits.masked_fill(action_mask[0] <= 0, -1e8), dim=-1)
    assert probs[teacher_action].item() > probs[sampled_action].item()


def test_on_policy_advantage_diagnostics_populated() -> None:
    good_action = int(ActionRange.SHOP_LEAVE)
    bad_action = int(ActionRange.SHOP_REROLL)
    model = _TinyPpoModel()
    optimizer = _make_policy_optimizer(model.parameters(), lr=0.01)
    buffer = _make_signal_buffer(
        actions=[good_action, bad_action],
        advantages=[1.0, -1.0],
        teacher_actions=[-1, -1],
        teacher_forced=[False, False],
    )
    config = PPOConfig(
        ppo_epochs=1,
        mini_batch_size=2,
        clip_epsilon=0.2,
        entropy_coeff=0.0,
        value_loss_coeff=0.0,
        survival_loss_coeff=0.0,
        heuristic_distill_coeff=0.0,
        target_kl=None,
        rollout_temperature=1.0,
    )

    stats = _run_ppo_update(
        model=model,
        optimizer=optimizer,
        buffer=buffer,
        return_rms=None,
        entropy_coeff=0.0,
        distill_coeff=0.0,
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
