"""PPO optimizer updates, KL rollback, entropy, and replay auxiliary losses."""

from __future__ import annotations

import copy
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from ..action_grammar import ActionGrammarDistribution
from ..agent import BalatroAgent
from ..value_head import outcome_nll, return_huber_loss
from .ppo_policy import (
    _ACTION_ID_TO_TYPE_INDEX,
    _ACTION_TYPES,
    _critic_predictions,
    _grammar_distribution,
    _policy_temperature_for_scalars,
    _unwrap_model,
)
from .sil import (
    EpisodeReplayBuffer,
    sil_percentile_gate,
)

if TYPE_CHECKING:
    from .ppo_config import (
        PPOConfig,
    )
    from .rollout_buffer import RolloutBuffer

logger = logging.getLogger(__name__)


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer state tensors to ``device`` after ``load_state_dict``."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _apply_lr_override(optimizer: torch.optim.Optimizer, new_lr: float, checkpoint_lr: float | None) -> None:
    """Set each param group's LR to ``new_lr``, warning if it differs from the checkpoint."""
    for group in optimizer.param_groups:
        group["lr"] = new_lr
    if checkpoint_lr is not None and abs(checkpoint_lr - new_lr) > 1e-12:
        logger.warning(
            "Overriding optimizer LR from checkpoint %.2e to CLI %.2e. "
            "Adam moments are preserved; only the step LR changes.",
            checkpoint_lr,
            new_lr,
        )


def _restore_policy_optimizer_state(
    optimizer: torch.optim.Optimizer,
    resume_state: dict,
    config: PPOConfig,
    device: torch.device,
) -> float:
    """Strictly restore Adam state and apply the active run LR."""

    optimizer_state = resume_state.get("optimizer_state_dict")
    if not isinstance(optimizer_state, dict):
        raise RuntimeError("Strict resume checkpoint lacks optimizer_state_dict")

    optimizer.load_state_dict(optimizer_state)
    _optimizer_to(optimizer, device)
    _apply_lr_override(optimizer, config.lr, resume_state.get("lr"))
    return float(config.lr)


@dataclass
class _UpdateStats:
    policy_losses: list[float]
    return_hubers: list[float]
    outcome_nlls: list[float]
    outcome_briers: list[float]
    derived_win_briers: list[float]
    outcome_valid_counts: list[float]
    terminal_value_means: list[float]
    return_residual_means: list[float]
    expected_return_means: list[float]
    entropies: list[float]
    normalized_entropies: list[float]
    action_type_entropies: list[float]
    clip_fracs: list[float]
    approx_kls: list[float]
    valid_action_counts: list[float]
    valid_action_type_counts: list[float]
    sample_counts: list[int] = field(default_factory=list)
    on_policy_fractions: list[float] = field(default_factory=list)
    on_policy_advantage_means: list[float] = field(default_factory=list)
    on_policy_advantage_stds: list[float] = field(default_factory=list)
    on_policy_positive_advantage_fractions: list[float] = field(default_factory=list)
    on_policy_return_means: list[float] = field(default_factory=list)
    ppo_minibatches_processed: list[int] = field(default_factory=list)
    ppo_samples_processed: int = 0
    running_kl: float = 0.0
    full_kl: float = 0.0
    kl_rollback: bool = False
    stop_reason: str = "none"
    actual_lr: float = 0.0
    sil_losses: list[float] = field(default_factory=list)
    sil_losses_weighted: list[float] = field(default_factory=list)
    sil_advantage_means: list[float] = field(default_factory=list)
    sil_gate_means: list[float] = field(default_factory=list)
    sil_gate_saturation_fractions: list[float] = field(default_factory=list)
    sil_advantage_p50s: list[float] = field(default_factory=list)
    sil_advantage_p95s: list[float] = field(default_factory=list)
    sil_advantage_p99s: list[float] = field(default_factory=list)
    sil_advantage_p80s: list[float] = field(default_factory=list)
    sil_gate_open_thresholds: list[float] = field(default_factory=list)
    sil_gate_saturation_thresholds: list[float] = field(default_factory=list)
    sil_gate_positive_fractions: list[float] = field(default_factory=list)
    sil_noise_floor_rejected_fractions: list[float] = field(default_factory=list)
    sil_gate_weight_from_wins_fractions: list[float] = field(default_factory=list)
    sil_samples_counts: list[int] = field(default_factory=list)
    sil_unique_episodes_sampleds: list[int] = field(default_factory=list)
    sil_max_samples_from_one_episodes: list[int] = field(default_factory=list)
    sil_sample_win_fractions: list[float] = field(default_factory=list)
    sil_logical_minibatches_attempted: int = 0
    sil_logical_minibatches_applied: int = 0
    sil_grad_actor_norm_weighted: float | None = None
    sil_grad_ppo_actor_norm: float | None = None
    sil_grad_ppo_actor_norm_ratio: float | None = None
    sil_grad_ppo_actor_grad_cosine: float | None = None
    sil_grad_diagnostic_valid: bool = False
    return_path_loss: float = 0.0
    return_path_samples: int = 0


@dataclass
class _PPOUpdateSnapshot:
    """Exact pre-update state used to reject an unsafe PPO update."""

    model: dict
    optimizer: dict


def _clone_state_to_cpu(value):
    """Recursively clone checkpoint-like state without retaining GPU storage."""
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _clone_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _snapshot_ppo_update(model: nn.Module, optimizer: Adam) -> _PPOUpdateSnapshot:
    """Clone model/optimizer state to CPU before a rollback-protected update."""
    return _PPOUpdateSnapshot(
        model=_clone_state_to_cpu(model.state_dict()),
        optimizer=_clone_state_to_cpu(optimizer.state_dict()),
    )


def _restore_ppo_update(
    model: nn.Module,
    optimizer: Adam,
    snapshot: _PPOUpdateSnapshot,
) -> None:
    """Restore every parameter, buffer, Adam moment, and Adam step exactly."""
    model.load_state_dict(snapshot.model)
    optimizer.load_state_dict(snapshot.optimizer)
    device = next(model.parameters()).device
    _optimizer_to(optimizer, device)
    optimizer.zero_grad(set_to_none=True)


def _sample_weighted_mean(values: list[float], sample_counts: list[int | float]) -> float:
    """Aggregate per-microbatch means with their actual denominator counts."""
    if not values:
        return 0.0
    if len(values) != len(sample_counts) or not sample_counts:
        return float(np.mean(values))
    weights = np.asarray(sample_counts, dtype=np.float64)
    if float(weights.sum()) <= 0.0:
        return 0.0
    return float(np.average(np.asarray(values, dtype=np.float64), weights=weights))


def _logical_mask_weights(
    batches: list[dict[str, torch.Tensor]],
    mask_key: str,
    accum_steps: int,
) -> tuple[list[float], list[float]]:
    """Return valid-target fractions within each logical optimizer group.

    Outcome NLL is a mean over valid terminal labels, not rows. Its accumulated
    gradient must therefore be weighted by each microbatch's share of the
    logical group's valid labels. All-zero groups receive zero weights,
    avoiding both divide-by-zero and accidental auxiliary gradients.
    """
    valid_counts = [float(batch[mask_key].detach().sum().item()) for batch in batches]
    weights = [0.0] * len(batches)
    group_start = 0
    for index, batch in enumerate(batches):
        if "_logical_group_end" in batch:
            group_end = bool(batch["_logical_group_end"].item())
        else:
            group_end = ((index + 1) % max(accum_steps, 1) == 0) or index + 1 == len(batches)
        if not group_end:
            continue
        group_total = sum(valid_counts[group_start : index + 1])
        if group_total > 0.0:
            for group_index in range(group_start, index + 1):
                weights[group_index] = valid_counts[group_index] / group_total
        group_start = index + 1
    return weights, valid_counts


def _physical_minibatch_count(total_samples: int, logical_batch_size: int, micro_batch_size: int) -> int:
    """Count physical forwards when each logical batch has its own tail."""
    if total_samples <= 0:
        return 0
    logical = max(int(logical_batch_size), 1)
    micro = max(min(int(micro_batch_size), logical), 1)
    full_groups, remainder = divmod(int(total_samples), logical)
    count = full_groups * math.ceil(logical / micro)
    if remainder:
        count += math.ceil(remainder / micro)
    return count


def _value_head_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return PPO critic parameters whose losses should not update the shared trunk."""
    value_head = getattr(_unwrap_model(model), "value_head", None)
    if value_head is None:
        return []
    return [param for param in value_head.parameters() if param.requires_grad]


def _accumulate_critic_grads_into_value_head(critic_loss: torch.Tensor, value_params: list[nn.Parameter]) -> None:
    """Backprop ``critic_loss`` onto the value-head parameters only.

    Uses ``torch.autograd.grad`` instead of ``backward`` so trunk/policy
    parameters never receive a ``.grad`` tensor — Adam skips grad-None params
    entirely, leaving both their weights and their optimizer moments untouched.
    """
    critic_grads = torch.autograd.grad(
        critic_loss,
        value_params,
        allow_unused=True,
    )
    for param, grad in zip(value_params, critic_grads, strict=True):
        if grad is None:
            continue
        if param.grad is None:
            param.grad = grad.detach()
        else:
            param.grad.add_(grad.detach())


@dataclass
class _WeightedKL:
    """Sample-weighted KL accumulator (minibatches may have different sizes)."""

    total: float = 0.0
    samples: int = 0

    def add(self, mean_kl: float, samples: int) -> None:
        if samples > 0:
            self.total += float(mean_kl) * samples
            self.samples += samples

    @property
    def mean(self) -> float:
        return self.total / self.samples if self.samples else 0.0


def _evaluate_rollout_kl(
    model: nn.Module,
    buffer: RolloutBuffer,
    batch_size: int,
    device: torch.device,
    use_pin_memory: bool,
    config: PPOConfig,
) -> tuple[float, float, int]:
    """Evaluate current-vs-rollout KL over all on-policy samples.

    The buffer API shuffles with NumPy, so preserve its RNG state: a diagnostic
    pass must not change the order of the following training epoch.
    """
    numpy_state = np.random.get_state()
    try:
        batches = buffer.get_batches(batch_size, device, pin_memory=use_pin_memory)
    finally:
        np.random.set_state(numpy_state)

    aggregate = _WeightedKL()
    minibatch_max = 0.0
    with torch.no_grad():
        for batch in batches:
            dist, _ = _grammar_distribution(
                model,
                batch,
                temperature=_policy_temperature_for_scalars(batch["scalars"], config),
            )
            sample_count = int(batch["actions"].numel())
            new_log_probs = dist.log_prob(batch["actions"])
            log_ratio = new_log_probs - batch["old_log_probs"]
            ratio = torch.exp(log_ratio)
            kl = ((ratio - 1.0) - log_ratio).mean().item()
            aggregate.add(kl, sample_count)
            minibatch_max = max(minibatch_max, kl)
    return aggregate.mean, minibatch_max, aggregate.samples


def _run_ppo_update(
    model: nn.Module,
    optimizer: Adam,
    buffer: RolloutBuffer,
    entropy_coeff: float,
    config: PPOConfig,
    accum_steps: int,
    effective_batch_size: int,
    device: torch.device,
    use_pin_memory: bool,
    sil_buffer: EpisodeReplayBuffer | None = None,
    sil_coeff_now: float = 0.0,
    grad_diagnostics_due: bool = False,
    return_path_replay=None,
) -> _UpdateStats:
    """Run `config.ppo_epochs` passes over the buffer and apply PPO updates.

    Keeps dropout disabled so the PPO ratio compares the same policy function
    that collected `old_log_probs`. Caller is responsible for the entropy-alpha
    step and logging.

    SIL is folded into the *same* backward/optimizer step as PPO. It is
    attempted on at most ``config.sil_logical_minibatches_per_update`` logical
    optimizer groups per update (one accumulated optimizer step == one logical
    group, regardless of gradient accumulation). One SIL batch is sampled per
    attempted group and contributes exactly one aggregate gradient via a
    separate full-weight backward at the group start; PPO and SIL then share
    the subsequent clip + ``optimizer.step()``. KL early stopping always halts
    at a logical-group boundary so a pending accumulation group is never
    abandoned halfway through.
    """
    model.eval()
    effective_clip_epsilon = float(config.clip_epsilon)
    stats = _UpdateStats(
        policy_losses=[],
        return_hubers=[],
        outcome_nlls=[],
        outcome_briers=[],
        derived_win_briers=[],
        outcome_valid_counts=[],
        terminal_value_means=[],
        return_residual_means=[],
        expected_return_means=[],
        entropies=[],
        normalized_entropies=[],
        action_type_entropies=[],
        clip_fracs=[],
        approx_kls=[],
        valid_action_counts=[],
        valid_action_type_counts=[],
        on_policy_fractions=[],
    )
    stats.actual_lr = float(optimizer.param_groups[0]["lr"])

    # A hard KL limit is a rejection criterion, not a warning. Snapshot before
    # the first gradient is computed so a rejected update restores both weights
    # and Adam's moments/step counters.
    rollback_snapshot = _snapshot_ppo_update(model, optimizer) if config.target_kl_max is not None else None
    n_rollout_samples = len(buffer._flat_returns)
    if accum_steps > 1:
        minibatches_per_epoch = max(
            1,
            _physical_minibatch_count(
                n_rollout_samples,
                config.mini_batch_size,
                effective_batch_size,
            ),
        )
    else:
        minibatches_per_epoch = max(1, math.ceil(n_rollout_samples / effective_batch_size))
    expected_minibatches = minibatches_per_epoch * config.ppo_epochs
    min_soft_stop_fraction = config.min_minibatch_fraction if config.min_minibatch_fraction is not None else 0.0
    min_soft_stop_minibatches = math.ceil(expected_minibatches * min_soft_stop_fraction)
    running_kl = _WeightedKL()

    # SIL logical-group budget. Per update (not per epoch). Attempted groups
    # count even when their gate is empty, so training does not keep resampling
    # until it finds a favorable replay batch.
    sil_budget = config.sil_logical_minibatches_per_update if sil_coeff_now > 0.0 else 0
    sil_attempted_groups = 0
    sil_applied_groups = 0
    # Gradient diagnostics: captured once, on the first group where SIL is
    # applied, by snapshotting actor .grad right after the SIL backward and
    # again before the group's global clip. PPO actor grad = total - SIL.
    grad_diag_attempted = False
    sil_grad_snapshot: list[torch.Tensor | None] | None = None

    minibatches_processed = 0
    return_path_attempted = False
    # Existing two-way SIL/PPO gradient attribution excludes a third actor loss.
    grad_diagnostics_due = grad_diagnostics_due and return_path_replay is None
    for _ppo_epoch in range(config.ppo_epochs):
        batches = buffer.get_batches(
            config.mini_batch_size if accum_steps > 1 else effective_batch_size,
            device,
            pin_memory=use_pin_memory,
            micro_batch_size=effective_batch_size if accum_steps > 1 else None,
        )
        outcome_loss_weights, outcome_valid_counts = _logical_mask_weights(
            batches,
            "terminal_outcome_mask",
            accum_steps,
        )
        optimizer.zero_grad()
        for i, batch in enumerate(batches):
            if "_logical_group_start" in batch:
                is_group_start = bool(batch["_logical_group_start"].item())
                is_step_boundary = bool(batch["_logical_group_end"].item())
                loss_weight = batch["_loss_weight"]
            else:
                # Backward-compatible path for lightweight test buffers.
                is_group_start = i % accum_steps == 0
                is_step_boundary = ((i + 1) % accum_steps == 0) or ((i + 1) == len(batches))
                loss_weight = torch.as_tensor(1.0 / accum_steps, device=device)
            outcome_loss_weight = torch.as_tensor(outcome_loss_weights[i], device=device)

            if is_group_start and not return_path_attempted and return_path_replay is not None:
                return_path_attempted = True
                prefix_batch = return_path_replay.sample(
                    config.return_path_batch_size, device,
                    samples_per_episode=config.return_path_samples_per_episode,
                ) if config.return_path_coeff > 0 else None
                if prefix_batch is not None:
                    prefix_dist, _ = _grammar_distribution(
                        model, prefix_batch,
                        temperature=_policy_temperature_for_scalars(prefix_batch["scalars"], config),
                    )
                    prefix_loss = -prefix_dist.log_prob(prefix_batch["actions"]).mean()
                    if not torch.isfinite(prefix_loss):
                        raise RuntimeError("Nonfinite return-path imitation loss")
                    (config.return_path_coeff * prefix_loss).backward()
                    stats.return_path_loss = float(prefix_loss.detach())
                    stats.return_path_samples = int(prefix_batch["actions"].numel())
                    del prefix_batch, prefix_dist, prefix_loss

            # --- SIL auxiliary loss (one batch per attempted logical group) ---
            # Sampled and backwarded at the group start so the whole group
            # receives exactly one aggregate SIL contribution; the coefficient
            # is the runtime decayed value, not the static config field.
            if is_group_start and sil_attempted_groups < sil_budget and sil_buffer is not None:
                sil_attempted_groups += 1
                sil_result = _compute_sil_group_loss(model, sil_buffer, config, device)
                stats.sil_logical_minibatches_attempted = sil_attempted_groups
                if sil_result.loss is not None:
                    capture_grads = grad_diagnostics_due and not grad_diag_attempted and sil_applied_groups == 0
                    grad_diag_attempted = capture_grads or grad_diag_attempted
                    (sil_coeff_now * sil_result.loss).backward()
                    sil_applied_groups += 1
                    stats.sil_logical_minibatches_applied = sil_applied_groups
                    stats.sil_losses.append(sil_result.loss.item())
                    stats.sil_losses_weighted.append(sil_coeff_now * sil_result.loss.item())
                    for key, stat_list in (
                        ("advantage_mean", stats.sil_advantage_means),
                        ("gate_mean", stats.sil_gate_means),
                        ("gate_saturation_fraction", stats.sil_gate_saturation_fractions),
                        ("advantage_p50", stats.sil_advantage_p50s),
                        ("advantage_p80", stats.sil_advantage_p80s),
                        ("advantage_p95", stats.sil_advantage_p95s),
                        ("gate_open_threshold", stats.sil_gate_open_thresholds),
                        ("gate_saturation_threshold", stats.sil_gate_saturation_thresholds),
                        ("gate_positive_fraction", stats.sil_gate_positive_fractions),
                        ("noise_floor_rejected_fraction", stats.sil_noise_floor_rejected_fractions),
                        ("gate_weight_from_wins_fraction", stats.sil_gate_weight_from_wins_fractions),
                        ("samples", stats.sil_samples_counts),
                        ("unique_episodes_sampled", stats.sil_unique_episodes_sampleds),
                        ("max_samples_from_one_episode", stats.sil_max_samples_from_one_episodes),
                        ("sample_win_fraction", stats.sil_sample_win_fractions),
                    ):
                        if key in sil_result.diagnostics:
                            value = sil_result.diagnostics[key]
                            stat_list.append(value if isinstance(value, int) else float(value))
                    if "advantage_p99" in sil_result.diagnostics:
                        stats.sil_advantage_p99s.append(sil_result.diagnostics["advantage_p99"])
                    if capture_grads:
                        sil_grad_snapshot = _snapshot_actor_grads(model)

            dist, value_dict = _grammar_distribution(
                model,
                batch,
                temperature=_policy_temperature_for_scalars(batch["scalars"], config),
            )

            new_log_probs = dist.log_prob(batch["actions"])
            entropy_per_state = dist.entropy()
            entropy = entropy_per_state.mean()
            valid_action_counts = batch["action_mask"].sum(dim=-1)
            normalized_entropy_per_state = _per_state_normalized_entropy(entropy_per_state, batch["action_mask"])
            normalized_entropy = normalized_entropy_per_state.mean()
            normalized_action_type_entropy = dist.normalized_action_type_entropy()
            if isinstance(dist, ActionGrammarDistribution):
                # The grammar has already reduced the flat mask by action type.
                # Exclude its sampling fallback on rows with no legal actions.
                valid_action_type_count_mean = (dist.macro_mask.sum(dim=-1) * (valid_action_counts > 0)).float().mean()
            else:
                valid_action_type_count_mean = torch.as_tensor(
                    _mean_valid_action_type_count(batch["action_mask"]), device=device
                )

            # Policy loss (clipped PPO). Advantages are already normalized
            # once over the full rollout in buffer.normalize_advantages();
            # per-mini-batch normalization would let rare terminals
            # dominate their batch and crush others to noise.
            on_policy_samples = batch["actions"].numel()
            log_ratio = new_log_probs - batch["old_log_probs"]
            ratio = torch.exp(log_ratio)
            advantages = batch["advantages"]
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(
                    ratio,
                    1 - effective_clip_epsilon,
                    1 + effective_clip_epsilon,
                )
                * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # The composed return is terminal utility plus a learned return
            # correction. Regress only the correction: the detached terminal
            # value prevents stochastic return targets from distorting outcome
            # calibration.
            returns_target = batch["returns"]
            return_loss = return_huber_loss(value_dict, returns_target)

            terminal_nll = outcome_nll(
                value_dict["outcome_probabilities"],
                batch["terminal_outcome_target"],
                batch["terminal_outcome_mask"],
            )
            with torch.no_grad():
                outcome_mask = batch["terminal_outcome_mask"].float()
                outcome_denominator = outcome_mask.sum().clamp(min=1.0)
                one_hot_outcome = F.one_hot(
                    batch["terminal_outcome_target"].long(),
                    num_classes=value_dict["outcome_probabilities"].shape[1],
                ).to(dtype=value_dict["outcome_probabilities"].dtype)
                outcome_brier_per_row = (value_dict["outcome_probabilities"] - one_hot_outcome).square().sum(dim=-1)
                outcome_brier = (outcome_brier_per_row * outcome_mask).sum() / outcome_denominator
                win_target = (
                    batch["terminal_outcome_target"] == value_dict["outcome_probabilities"].shape[1] - 1
                ).float()
                derived_win_brier = (
                    (value_dict["win_prob"] - win_target).square() * outcome_mask
                ).sum() / outcome_denominator

            # Row-normalized value loss and valid-target-normalized auxiliary
            # losses have different logical denominators. Weight each by its
            # own share rather than applying the row fraction to the entire
            # critic objective.
            weighted_critic_loss = (
                config.value_loss_coeff * return_loss * loss_weight
                + config.outcome_loss_coeff * terminal_nll * outcome_loss_weight
            )

            policy_objective_loss = policy_loss - entropy_coeff * (
                normalized_entropy + config.action_type_entropy_scale * normalized_action_type_entropy
            )
            total_loss = policy_objective_loss * loss_weight + weighted_critic_loss
            total_loss.backward()

            # Step every accum_steps micro-batches (or on last batch). This is
            # a logical optimizer-group boundary: PPO and SIL share this single
            # clip + optimizer.step().
            if is_step_boundary:
                if sil_grad_snapshot is not None:
                    # Gradient diagnostics from the same logical group: PPO
                    # actor grad = total (PPO + SIL) - SIL snapshot, measured
                    # before global clipping and without mutating .grad.
                    total_snapshot = _snapshot_actor_grads(model)
                    diag = _sil_grad_diagnostics(sil_grad_snapshot, total_snapshot)
                    stats.sil_grad_actor_norm_weighted = diag["actor_grad_norm_weighted"]
                    stats.sil_grad_ppo_actor_norm = diag["ppo_actor_grad_norm"]
                    stats.sil_grad_ppo_actor_norm_ratio = diag["ppo_actor_grad_norm_ratio"]
                    stats.sil_grad_ppo_actor_grad_cosine = diag["ppo_actor_grad_cosine"]
                    stats.sil_grad_diagnostic_valid = bool(diag["grad_diagnostic_valid"])
                    sil_grad_snapshot = None
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm, error_if_nonfinite=config.precision == "bf16"
                )
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                # One device-to-host transfer for telemetry, rather than one
                # synchronization for every scalar. KL is still checked after
                # each minibatch, at the same optimizer boundary as before.
                metrics = {
                    "clip_fracs": ((ratio - 1.0).abs() > effective_clip_epsilon).float().mean(),
                    "approx_kls": ((ratio - 1.0) - log_ratio).mean(),
                    "valid_action_counts": valid_action_counts.float().mean(),
                    "valid_action_type_counts": valid_action_type_count_mean,
                    "on_policy_advantage_means": advantages.mean(),
                    "on_policy_advantage_stds": advantages.std(unbiased=False),
                    "on_policy_positive_advantage_fractions": (advantages > 0).float().mean(),
                    "on_policy_return_means": batch["returns"].mean(),
                    "policy_losses": policy_loss,
                    "return_hubers": return_loss,
                    "outcome_nlls": terminal_nll,
                    "outcome_briers": outcome_brier,
                    "derived_win_briers": derived_win_brier,
                    "terminal_value_means": value_dict["terminal_value"].mean(),
                    "return_residual_means": value_dict["return_residual"].mean(),
                    "expected_return_means": value_dict["expected_return"].mean(),
                    "entropies": entropy,
                    "normalized_entropies": normalized_entropy,
                    "action_type_entropies": normalized_action_type_entropy,
                }
                for key, value in zip(
                    metrics, torch.stack(list(metrics.values())).detach().cpu().tolist(), strict=True
                ):
                    getattr(stats, key).append(value)
                approx_kl = stats.approx_kls[-1]
            stats.outcome_valid_counts.append(outcome_valid_counts[i])
            stats.on_policy_fractions.append(1.0)
            minibatches_processed += 1
            stats.sample_counts.append(on_policy_samples)
            stats.ppo_samples_processed += on_policy_samples
            running_kl.add(approx_kl, on_policy_samples)
            stats.running_kl = running_kl.mean

            # A hard minibatch breach rejects the entire PPO update. This check
            # is intentionally independent of the soft-stop minimum fraction.
            if rollback_snapshot is not None and config.target_kl_max is not None and approx_kl > config.target_kl_max:
                logger.warning(
                    "Rejecting PPO update: minibatch KL %.5f exceeded hard limit %.5f",
                    approx_kl,
                    config.target_kl_max,
                )
                _restore_ppo_update(model, optimizer, rollback_snapshot)
                stats.kl_rollback = True
                stats.stop_reason = "hard_kl"
                break
        if stats.kl_rollback:
            break

        # Soft stopping uses a full-rollout, sample-weighted measurement at an
        # epoch boundary. A noisy individual minibatch can therefore neither
        # terminate an update nor bias the aggregate merely by being small.
        full_kl, full_kl_max, _ = _evaluate_rollout_kl(
            model,
            buffer,
            effective_batch_size,
            device,
            use_pin_memory,
            config,
        )
        stats.full_kl = full_kl
        if rollback_snapshot is not None and config.target_kl_max is not None and full_kl_max > config.target_kl_max:
            logger.warning(
                "Rejecting PPO update: full-rollout minibatch KL %.5f exceeded hard limit %.5f",
                full_kl_max,
                config.target_kl_max,
            )
            _restore_ppo_update(model, optimizer, rollback_snapshot)
            stats.kl_rollback = True
            stats.stop_reason = "hard_kl"
            break
        if (
            config.target_kl is not None
            and full_kl > config.target_kl
            and minibatches_processed >= min_soft_stop_minibatches
        ):
            logger.debug(
                "Stopping remaining PPO epochs: full-rollout KL %.5f exceeded target %.5f after %d/%d minibatches",
                full_kl,
                config.target_kl,
                minibatches_processed,
                expected_minibatches,
            )
            stats.stop_reason = "soft_kl"
            break

    stats.ppo_minibatches_processed.append(minibatches_processed)
    return stats


@dataclass
class _SILGroupResult:
    """Outcome of one SIL logical-group loss computation."""

    loss: torch.Tensor | None
    diagnostics: dict[str, float]


@dataclass
class _TerminalReplayResult:
    """Metrics from head-only completed-episode terminal supervision."""

    updates_applied: int
    samples: int
    diagnostics: dict[str, float]


def _terminal_aux_parameters(
    model: nn.Module,
) -> list[nn.Parameter]:
    """Return the outcome tower's parameters, excluding the return path.

    ``outcome_proj`` and ``ante_survival`` feed only the hazards, so replay can
    train both without disturbing ``pool_proj``/``return_residual`` or the
    shared trunk.
    """

    value_head = getattr(_unwrap_model(model), "value_head", None)
    if value_head is None:
        return []
    return [
        parameter
        for name, parameter in value_head.named_parameters()
        if parameter.requires_grad and name.startswith(("outcome_proj.", "ante_survival."))
    ]


@torch.no_grad()
def _outcome_metric_inputs_cpu(value_dict, sampled):
    """Transfer the small diagnostic payload once, before bucket reductions.

    Critic probabilities are FP32 in production. Ante/outcome labels are small
    exact integers, so packing them alongside probabilities preserves labels.
    This avoids GPU synchronization for every bucket, mask, and scalar metric.
    """
    probabilities = value_dict["outcome_probabilities"]
    classes = probabilities.shape[1]
    packed = torch.cat([
        probabilities,
        value_dict["win_prob"].reshape(-1, 1),
        sampled["terminal_outcome_target"].to(probabilities.dtype).reshape(-1, 1),
        sampled["current_antes"].long().to(probabilities.dtype).reshape(-1, 1),
        (sampled["cross_rollout_flags"] > 0.5).to(probabilities.dtype).reshape(-1, 1),
    ], dim=1).cpu()
    return (
        {"outcome_probabilities": packed[:, :classes], "win_prob": packed[:, classes]},
        {
            "terminal_outcome_target": packed[:, classes + 1].long(),
            "current_antes": packed[:, classes + 2].long(),
            "cross_rollout_flags": packed[:, classes + 3],
        },
    )


def _outcome_metric_rows(
    value_dict: dict[str, torch.Tensor],
    sampled: dict[str, torch.Tensor],
) -> dict[str, tuple[torch.Tensor, torch.Tensor | None]]:
    """Build per-row outcome scores plus the buckets they are reported over.

    Returns ``{metric_key: (values, mask)}`` so training and holdout passes
    share one definition. The climatology reference is the bucket's own
    empirical outcome rate corrected by ``n / (n - 1)``: scoring an empirical
    rate on the same rows it was fitted on understates its Brier by exactly a
    factor ``(1 - 1/n)``, which on a 32-row bucket is a 3-point handicap
    charged to climatology and credited to the model.
    """

    if value_dict["outcome_probabilities"].device.type != "cpu":
        value_dict, sampled = _outcome_metric_inputs_cpu(value_dict, sampled)

    outcome_probabilities = value_dict["outcome_probabilities"]
    outcome_targets = sampled["terminal_outcome_target"]
    num_classes = outcome_probabilities.shape[1]
    selected = outcome_probabilities.gather(1, outcome_targets.unsqueeze(1)).squeeze(1)
    outcome_nll_per_row = -selected.clamp_min(1e-7).log()
    one_hot_outcome = F.one_hot(outcome_targets, num_classes=num_classes).to(dtype=outcome_probabilities.dtype)
    outcome_brier = (outcome_probabilities - one_hot_outcome).square().sum(dim=-1)
    win_targets = (outcome_targets == num_classes - 1).float()
    derived_win_brier = (value_dict["win_prob"] - win_targets).square()

    cross_rollout = sampled["cross_rollout_flags"] > 0.5
    current_antes = sampled["current_antes"].long()
    buckets: list[tuple[str, torch.Tensor | None]] = [
        ("", None),
        ("cross_rollout/", cross_rollout),
        ("same_rollout/", ~cross_rollout),
        *[(f"ante_{ante}/", current_antes == int(ante)) for ante in current_antes.unique().tolist()],
    ]

    rows: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
    for prefix, bucket_mask in buckets:
        if bucket_mask is not None and not bool(bucket_mask.any()):
            continue
        bucket_outcomes = one_hot_outcome if bucket_mask is None else one_hot_outcome[bucket_mask]
        rows[prefix + "outcome_nll"] = (outcome_nll_per_row, bucket_mask)
        rows[prefix + "outcome_brier"] = (outcome_brier, bucket_mask)
        rows[prefix + "derived_win_brier"] = (derived_win_brier, bucket_mask)
        bucket_size = int(bucket_outcomes.shape[0])
        if bucket_size < 2:
            # A one-row empirical climatology fits itself perfectly and the
            # n / (n - 1) correction is undefined. Report the model's scores for
            # the row but publish no reference, so no skill is derived from it.
            continue
        bucket_climatology = bucket_outcomes.mean(dim=0, keepdim=True)
        unbiased_scale = bucket_size / (bucket_size - 1)
        rows[prefix + "outcome_climatology_brier"] = (
            (bucket_climatology - bucket_outcomes).square().sum(dim=-1) * unbiased_scale,
            None,
        )
    return rows


def _add_brier_skills(diagnostics: dict[str, float]) -> None:
    """Derive ``*_brier_skill`` for every Brier/climatology pair in place."""

    for metric_name, metric_value in list(diagnostics.items()):
        if not metric_name.endswith("outcome_brier"):
            continue
        prefix = metric_name[: -len("outcome_brier")]
        climatology_value = diagnostics.get(prefix + "outcome_climatology_brier")
        if climatology_value is not None and climatology_value > 0.0:
            diagnostics[prefix + "outcome_brier_skill"] = 1.0 - metric_value / climatology_value


def _run_terminal_replay_updates(
    model: nn.Module,
    optimizer: Adam,
    episode_buffer: EpisodeReplayBuffer,
    config: PPOConfig,
    device: torch.device,
) -> _TerminalReplayResult:
    """Train terminal heads from complete recent episodes without actor credit.

    Complete Monte Carlo outcomes supervise the normalized distribution over
    death at each remaining Ante plus reaching the target. Only the value
    head's outcome tower receives gradients; replay cannot update the
    policy/shared trunk, shared pooling, or return residual.

    Replay training draws exclude the replay-held-out episodes. These episodes
    HAVE been trained on by PPO, so ``replay_holdout/*`` is explicitly not a
    generalization metric. Independent critic evaluation uses reserved fresh
    seeds in ppo_evaluation and never inserts its rows into any training buffer.
    """

    diagnostics: dict[str, float] = {
        "buffer_episodes": float(episode_buffer.num_episodes),
        "buffer_transitions": float(episode_buffer.num_transitions),
        "buffer_win_fraction": float(episode_buffer.win_fraction),
        "buffer_holdout_episodes": float(episode_buffer.num_holdout_episodes),
        "label_coverage": float(episode_buffer.labeled_transition_fraction),
        "cross_rollout_transition_fraction": float(episode_buffer.cross_rollout_transition_fraction),
        "episodes_added_total": float(episode_buffer.episodes_added_total),
        "stalled_episodes_dropped_total": float(episode_buffer.stalled_episodes_dropped_total),
        "overflow_episodes_dropped_total": float(episode_buffer.overflow_episodes_dropped_total),
    }
    no_update = _TerminalReplayResult(0, 0, diagnostics)
    if config.terminal_replay_updates_per_ppo_update <= 0:
        return no_update
    if episode_buffer.num_episodes < config.terminal_replay_min_episodes:
        return no_update

    terminal_params = _terminal_aux_parameters(model)
    if not terminal_params or config.outcome_loss_coeff <= 0.0:
        return no_update

    metric_sums: defaultdict[str, float] = defaultdict(float)
    metric_counts: defaultdict[str, int] = defaultdict(int)

    def record_metric(
        key: str,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        detached = values.detach()
        if mask is not None:
            detached = detached[mask]
        if detached.numel() == 0:
            return
        metric_sums[key] += float(detached.float().sum().item())
        metric_counts[key] += int(detached.numel())

    def forward_outcomes(sampled: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        value_dict = _critic_predictions(
            model,
            sampled,
            temperature=_policy_temperature_for_scalars(sampled["scalars"], config),
        )
        return value_dict

    updates_applied = 0
    samples_seen = 0
    model.eval()
    for _ in range(config.terminal_replay_updates_per_ppo_update):
        sampled = episode_buffer.sample(
            config.terminal_replay_batch_size,
            device,
            samples_per_episode=config.terminal_replay_samples_per_episode,
            include_teacher_forced=True,
            row_uniform=config.terminal_replay_row_uniform,
            include_action_mask=not isinstance(_unwrap_model(model), BalatroAgent),
        )
        if sampled is None:
            break

        optimizer.zero_grad()
        value_dict = forward_outcomes(sampled)
        if value_dict.get("outcome_probabilities") is None:
            optimizer.zero_grad()
            break
        outcome_targets = sampled["terminal_outcome_target"]
        selected_outcome_probability = (
            value_dict["outcome_probabilities"]
            .gather(
                1,
                outcome_targets.unsqueeze(1),
            )
            .squeeze(1)
        )
        outcome_loss = -selected_outcome_probability.clamp_min(1e-7).log().mean()

        loss = config.outcome_loss_coeff * outcome_loss
        _accumulate_critic_grads_into_value_head(loss, terminal_params)
        nn.utils.clip_grad_norm_(terminal_params, config.max_grad_norm, error_if_nonfinite=config.precision == "bf16")
        optimizer.step()
        optimizer.zero_grad()
        updates_applied += 1
        samples_seen += int(outcome_targets.numel())

        with torch.no_grad():
            for key, (values, mask) in _outcome_metric_rows(value_dict, sampled).items():
                record_metric(key, values, mask)

    if updates_applied and episode_buffer.num_holdout_episodes > 0:
        holdout = episode_buffer.sample(
            config.terminal_replay_holdout_batch_size,
            device,
            samples_per_episode=config.terminal_replay_samples_per_episode,
            include_teacher_forced=True,
            holdout=True,
            row_uniform=config.terminal_replay_row_uniform,
            include_action_mask=not isinstance(_unwrap_model(model), BalatroAgent),
        )
        if holdout is not None:
            with torch.no_grad():
                holdout_values = forward_outcomes(holdout)
                if holdout_values.get("outcome_probabilities") is not None:
                    diagnostics["replay_holdout/samples"] = float(holdout["terminal_outcome_target"].numel())
                    for key, (values, mask) in _outcome_metric_rows(holdout_values, holdout).items():
                        record_metric("replay_holdout/" + key, values, mask)

    for key, total in metric_sums.items():
        diagnostics[key] = total / metric_counts[key]
    _add_brier_skills(diagnostics)
    diagnostics["updates_applied"] = float(updates_applied)
    diagnostics["samples"] = float(samples_seen)
    return _TerminalReplayResult(updates_applied, samples_seen, diagnostics)


def _compute_sil_group_loss(
    model: nn.Module,
    sil_buffer: EpisodeReplayBuffer | None,
    config: PPOConfig,
    device: torch.device,
) -> _SILGroupResult:
    """Sample one episode-uniform SIL batch and compute the gated actor loss.

    Used once per attempted logical optimizer group (not per physical
    microbatch). The loss is the gate-weighted NLL of the buffered actions; the
    coefficient is applied by the caller when folding it into the combined
    backward. SIL stays actor-only: the critic value enters only as a detached
    gate, so this never adds a value term.

    For ``sil_objective == "advantage"`` the gate is the shared percentile gate
    (:func:`sil_percentile_gate`) over current-critic MC advantages; for
    ``"winning_bc"`` the gate is 1 for every otherwise-valid winning row (plain
    behavior cloning), serving as a matched control.

    The loss reduction denominator is ALL otherwise-valid sampled rows
    (finite log-prob + teacher filter), including valid zero-gate rows, so gate
    sparsity reduces total SIL pressure rather than concentrating it.

    Returns a result whose ``loss`` is None when SIL is disabled, the buffer is
    short of ``sil_min_episodes``, sampling yields no eligible row, or no row is
    reachable. ``diagnostics`` always carries buffer counters for logging.
    """
    diagnostics: dict[str, float] = {
        "buffer_episodes": float(sil_buffer.num_episodes if sil_buffer else 0),
        "buffer_transitions": float(sil_buffer.num_transitions if sil_buffer else 0),
        "buffer_win_fraction": float(sil_buffer.win_fraction if sil_buffer else 0.0),
        "episodes_added_total": float(sil_buffer.episodes_added_total if sil_buffer else 0),
        "stalled_episodes_dropped_total": float(sil_buffer.stalled_episodes_dropped_total if sil_buffer else 0),
        "overflow_episodes_dropped_total": float(sil_buffer.overflow_episodes_dropped_total if sil_buffer else 0),
    }
    empty = _SILGroupResult(loss=None, diagnostics=diagnostics)
    # The caller (_run_ppo_update) gates on the runtime decayed coefficient
    # (sil_coeff_now) before attempting SIL; this helper never reads the static
    # initial coefficient so the runtime value is the sole authority.
    if sil_buffer is None:
        return empty
    if sil_buffer.num_episodes < config.sil_min_episodes:
        return empty

    winning_bc = config.sil_objective == "winning_bc"
    sampled = sil_buffer.sample(
        config.sil_batch_size,
        device,
        samples_per_episode=config.sil_samples_per_episode,
        only_wins=winning_bc,
    )
    if sampled is None:
        return empty

    n_sampled = int(sampled["actions"].shape[0])
    diagnostics["samples"] = float(n_sampled)
    if "episode_ids" in sampled:
        episode_ids_np = sampled["episode_ids"].detach().cpu().numpy()
        unique_ids, counts = np.unique(episode_ids_np, return_counts=True)
        diagnostics["unique_episodes_sampled"] = float(len(unique_ids))
        diagnostics["max_samples_from_one_episode"] = float(int(counts.max()) if counts.size else 0)
    outcomes = sampled.get("episode_outcomes")
    if outcomes is not None:
        diagnostics["sample_win_fraction"] = float(outcomes.mean().item())

    dist, value_dict = _grammar_distribution(
        model,
        sampled,
        temperature=_policy_temperature_for_scalars(sampled["scalars"], config),
    )
    log_probs = dist.log_prob(sampled["actions"])
    finite = (log_probs > -1e7).float()

    # Both stored returns and the composed critic prediction are in raw reward
    # units, so the SIL gate has one stable interpretation.
    v_raw = value_dict["expected_return"].detach()
    raw_advantage = sampled["returns"] - v_raw

    if winning_bc:
        gate = finite.clone()
        advantage_for_stats = raw_advantage
    else:
        finite_mask = finite.bool()
        eligible_adv = raw_advantage[finite_mask].detach().float().cpu().numpy()
        gate_np, gate_info = sil_percentile_gate(
            eligible_adv,
            open_percentile=config.sil_gate_open_percentile,
            saturation_percentile=config.sil_gate_saturation_percentile,
            advantage_floor=config.sil_advantage_floor,
        )
        gate = torch.zeros_like(finite)
        gate[finite_mask] = torch.as_tensor(gate_np, dtype=gate.dtype, device=device)
        advantage_for_stats = raw_advantage
        diagnostics["gate_open_threshold"] = float(gate_info["open_threshold"])
        diagnostics["gate_saturation_threshold"] = float(gate_info["saturation_threshold"])

    denom = finite.sum()
    if denom.item() <= 0:
        return empty
    loss = -(log_probs * gate * finite).sum() / denom

    with torch.no_grad():
        finite_mask = finite.bool()
        adv_np = advantage_for_stats[finite_mask].detach().float().cpu().numpy()
        gate_np = gate[finite_mask].detach().float().cpu().numpy()
        diagnostics["advantage_mean"] = float(np.mean(adv_np)) if adv_np.size else 0.0
        diagnostics["gate_mean"] = float(np.mean(gate_np)) if gate_np.size else 0.0
        if adv_np.size:
            diagnostics["advantage_p50"] = float(np.percentile(adv_np, 50.0))
            diagnostics["advantage_p80"] = float(np.percentile(adv_np, 80.0))
            diagnostics["advantage_p95"] = float(np.percentile(adv_np, 95.0))
        if gate_np.size:
            diagnostics["gate_positive_fraction"] = float(np.mean(gate_np > 0.0))
            diagnostics["gate_saturation_fraction"] = float(np.mean(gate_np >= 1.0))
            diagnostics["noise_floor_rejected_fraction"] = float(np.mean(gate_np == 0.0))
        if outcomes is not None and gate_np.size:
            win_flags = outcomes[finite_mask].detach().float().cpu().numpy()
            gate_mass = float(gate_np.sum())
            if gate_mass > 0.0:
                diagnostics["gate_weight_from_wins_fraction"] = float((gate_np * win_flags).sum() / gate_mass)
            else:
                diagnostics["gate_weight_from_wins_fraction"] = 0.0

    return _SILGroupResult(loss=loss, diagnostics=diagnostics)


def _snapshot_actor_grads(model: nn.Module) -> list[torch.Tensor | None]:
    """Clone current ``.grad`` for actor parameters (None preserved)."""
    snapshot: list[torch.Tensor | None] = []
    for name, param in model.named_parameters():
        if name.startswith(("value_head.", "module.value_head.")):
            continue
        snapshot.append(param.grad.detach().clone() if param.grad is not None else None)
    return snapshot


def _sil_grad_diagnostics(
    sil_grads: list[torch.Tensor | None], total_grads: list[torch.Tensor | None]
) -> dict[str, float]:
    """Weighted SIL vs PPO actor-gradient norms, ratio, and cosine similarity.

    ``sil_grads`` is the snapshot taken right after the SIL backward;
    ``total_grads`` is the snapshot taken after the group's PPO backwards but
    before global clipping. PPO actor gradient = total - SIL. Handles zero and
    non-finite norms safely: a zero component norm yields ratio/cosine 0.0
    rather than infinity or a misleading value.
    """
    sil_sq = 0.0
    ppo_sq = 0.0
    dot = 0.0
    for sg, tg in zip(sil_grads, total_grads, strict=True):
        if sg is None and tg is None:
            continue
        if sg is not None:
            s = sg.detach().float().reshape(-1)
            sil_sq += float(s.pow(2).sum().item())
            p = (tg.detach().float().reshape(-1) - s) if tg is not None else -s
            ppo_sq += float(p.pow(2).sum().item())
            dot += float((s * p).sum().item())
        else:
            p = tg.detach().float().reshape(-1)
            ppo_sq += float(p.pow(2).sum().item())
    eps = 1e-12
    sil_norm = math.sqrt(sil_sq)
    ppo_norm = math.sqrt(ppo_sq)
    ratio = 0.0 if ppo_norm <= eps else sil_norm / ppo_norm
    cosine = 0.0 if (sil_norm <= eps or ppo_norm <= eps) else dot / (sil_norm * ppo_norm)
    if not (math.isfinite(sil_norm) and math.isfinite(ppo_norm)):
        return {
            "actor_grad_norm_weighted": 0.0,
            "ppo_actor_grad_norm": 0.0,
            "ppo_actor_grad_norm_ratio": 0.0,
            "ppo_actor_grad_cosine": 0.0,
            "grad_diagnostic_valid": 0.0,
        }
    return {
        "actor_grad_norm_weighted": sil_norm,
        "ppo_actor_grad_norm": ppo_norm,
        "ppo_actor_grad_norm_ratio": ratio,
        "ppo_actor_grad_cosine": cosine,
        "grad_diagnostic_valid": 1.0,
    }


def _smoothed_entropy_signal(previous: float | None, current: float, beta: float) -> float:
    """Update the controller's entropy signal with an EMA."""
    if previous is None:
        return current
    return beta * previous + (1.0 - beta) * current


def _entropy_alpha_loss(log_alpha: torch.Tensor, entropy_signal: float, target_entropy: float) -> torch.Tensor:
    """Return the alpha loss for adaptive entropy tuning."""
    signal = torch.tensor(entropy_signal, dtype=torch.float32, device=log_alpha.device)
    return log_alpha.exp() * (signal - target_entropy)


def _make_alpha_optimizer(log_alpha: torch.Tensor, lr: float) -> Adam:
    """Build the entropy-coefficient optimizer without weight decay.

    `log_alpha` is usually negative. AdamW's decoupled weight decay pushes
    negative parameters toward zero, which increases alpha even when the
    entropy error is zero. Plain Adam avoids that bias.
    """
    return Adam([log_alpha], lr=lr)


def _make_policy_optimizer(parameters, lr: float) -> Adam:
    """Build the PPO optimizer without weight decay so RL updates do not flatten the policy prior."""
    return Adam(parameters, lr=lr)


def _per_state_normalized_entropy(entropy_per_state: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Return each state's entropy normalized by the log of its valid-action count."""
    valid_action_counts = action_mask.sum(dim=-1).to(entropy_per_state.dtype)
    max_entropy = torch.log(valid_action_counts.clamp_min(2.0))
    return torch.where(
        valid_action_counts > 1.0,
        entropy_per_state / max_entropy,
        torch.zeros_like(entropy_per_state),
    )


def _mean_valid_action_type_count(action_mask: torch.Tensor) -> float:
    """Return the mean number of valid action types in a batch."""
    type_index = _ACTION_ID_TO_TYPE_INDEX.to(device=action_mask.device)
    expanded_type_index = type_index.unsqueeze(0).expand(action_mask.shape[0], -1)
    valid_action_mask = (action_mask > 0).to(action_mask.dtype)
    valid_type_counts = torch.zeros(
        action_mask.shape[0],
        len(_ACTION_TYPES),
        dtype=action_mask.dtype,
        device=action_mask.device,
    )
    valid_type_counts.scatter_add_(1, expanded_type_index, valid_action_mask)
    return float((valid_type_counts > 0).sum(dim=-1).float().mean().item())
