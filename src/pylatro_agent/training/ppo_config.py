"""PPO configuration, validation, provenance, and resume-safe schedules."""

from __future__ import annotations

import logging
import math
import random
import uuid
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..reward import DEFAULT_REWARD_CONFIG, RewardConfig
from ..survival import validate_critic_win_ante

if TYPE_CHECKING:
    from ..archive import ArchiveConfig

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    # Collection and archive replay.
    archive_config: ArchiveConfig | None = None
    # Actor-only BC of verified archive prefixes; zero disables. Never enters PPO ratios or GAE.
    return_path_coeff: float = 0.02
    return_path_capacity: int = 32
    return_path_batch_size: int = 64
    return_path_samples_per_episode: int = 8
    return_path_rebuild_steps: int = 64
    milestone_final_scale: float = 0.0
    milestone_decay_fraction: float = 0.8
    num_envs: int = 32
    deck_key: str = "b_red"
    seed: int = 0
    rollout_length: int = 256
    # fixed_steps uses fragment GAE; complete_cohort collects whole games.
    collection_mode: str = "fixed_steps"
    cohort_games: int = 64
    cohort_training_rows: int = 8192
    cohort_max_episode_steps: int = 2048
    cohort_workers: int = 4
    cohort_seeds: list[int] | None = None
    outcome_label_mode: str = "completion_window"
    # Checkpointed critic/advantage semantics; changing them requires an explicit branch.
    return_value_mode: str = "terminal_plus_residual"
    advantage_estimator: str = "fragment_gae"
    # Training budget and PPO objective.
    total_timesteps: int = 1_000_000
    # When supplied, sets the absolute update target and derives total_timesteps.
    total_updates: int | None = None
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    gamma: float = 0.997
    gae_lambda: float = 0.97
    clip_epsilon: float = 0.1
    # Soft stop at logical batch boundaries; target_kl_max rolls back the entire update.
    target_kl: float | None = 0.05
    # Diagnostic threshold only. min_minibatch_fraction delays soft KL stopping.
    target_kl_p95: float | None = None
    target_kl_max: float | None = None
    min_minibatch_fraction: float | None = None
    entropy_coeff: float = 0.01
    adaptive_entropy: bool = False
    target_entropy: float = 0.15
    alpha_lr: float = 0.01
    alpha_min: float = 0.001
    alpha_max: float = 0.05
    entropy_ema_beta: float = 0.6
    action_type_entropy_scale: float = 0.0
    value_loss_coeff: float = 0.25
    max_grad_norm: float = 0.5
    lr: float = 5e-05
    device: str = "cpu"
    # None preserves checkpoint precision on resume; an explicit value overrides it.
    precision: str | None = None
    # Persistence and evaluation.
    save_dir: str = "checkpoints/ppo"
    log_dir: str = "runs/ppo"
    # Checked at update boundaries; save a full checkpoint before stopping.
    stop_request_path: str | None = None
    eval_interval: int = 5
    log_interval: int = 10
    checkpoint_interval: int = 50
    eval_games: int = 50
    # Reserved seed subset evaluated with sampled actions for on-policy critic metrics.
    eval_sampled_games: int = 25
    eval_before_training: bool = True
    # Stop after consecutive material regressions from the best eval; None disables.
    eval_regression_tolerance: float | None = None
    eval_regression_patience: int = 2
    # Temporarily move the model for evaluation, restoring it afterward.
    eval_device: str | None = None
    # Games advanced per inference batch; eval_workers=0 runs engines in-process.
    eval_batch_size: int = 32
    eval_workers: int = 4
    # 0 selects by model width/CPU affinity; restore training threads after evaluation.
    eval_cpu_threads: int = 0
    # Training/evaluation victory target; None means Ante 8.
    win_ante: int | None = None
    stake: int = 1
    max_no_progress_steps: int = 32
    # Physical forward/backward size; accumulate into exact logical mini_batch_size groups.
    micro_batch_size: int = 64
    async_envs: bool = True
    # Shared by rollout sampling, PPO loss, and bootstrap to keep likelihood ratios consistent.
    rollout_temperature: float = 1.0
    # Optional sharpening for Ante-1/dangerous active hands; other phases keep the base temperature.
    danger_rollout_temperature: float | None = None
    danger_death_probability_threshold: float = 0.35
    # Categorical terminal-outcome supervision; stalled/censored episodes remain masked.
    outcome_loss_coeff: float = 0.1
    # Completed-episode replay trains only the outcome tower, independent of SIL.
    terminal_replay_batch_size: int = 256
    terminal_replay_min_episodes: int = 8
    terminal_replay_samples_per_episode: int = 8
    terminal_replay_updates_per_ppo_update: int = 4
    # Sample transitions uniformly instead of giving every episode equal weight.
    terminal_replay_row_uniform: bool = True
    # Excluded from replay/SIL but already seen by PPO; this measures replay fitting only.
    terminal_replay_holdout_fraction: float = 0.1
    terminal_replay_holdout_batch_size: int = 512
    # None uses default potential rewards; PPO pins gamma and victory target.
    reward_config: RewardConfig | None = None
    # Replay every Nth play with one joker removed for marginal diagnostics; 0 disables.
    counterfactual_diagnostic_interval: int = 0
    # Append resolved forecasts to risk_forecasts.jsonl for risk calibration.
    risk_forecast_log: bool = True
    # Append actual training seeds and complete outcomes to training_episodes.jsonl.
    training_episode_log: bool = False
    # None uses 10000 + game index. Training excludes the entire list, including unused eval tails.
    eval_seeds: list[int] | None = None
    # Additional panels reserved from training.
    training_excluded_seeds: list[int] | None = None
    # Autoregressive hand/discard mixture weight; 0 selects candidate-only support.
    hand_ar_mixture_eps: float = 0.1
    # Optional run identity: all three fields required together and matched on strict resume.
    ppo_run_uuid: str | None = None
    ppo_source_sha256: str | None = None
    ppo_recipe_id: str | None = None
    # Discard saved best-eval selection state on resume.
    reset_best_eval: bool = False
    # Re-anchor annealing to this leg instead of the saved horizon.
    reset_schedules: bool = False
    # Self-imitation shares PPO optimizer steps and KL guards; 0 disables actor replay.
    sil_coeff: float = 0.0
    # Shared completed-episode capacity for terminal replay and SIL (checkpoint/CLI name retained).
    sil_buffer_episodes: int = 256
    # Transitions per SIL sample, once per attempted logical optimizer group.
    sil_batch_size: int = 64
    # Minimum completed episodes; winning_bc additionally requires sampled wins.
    sil_min_episodes: int = 8
    # Clamp globally normalized advantages; 0 disables.
    advantage_clip_sigma: float = 4.0
    # advantage uses critic-relative percentile gating; winning_bc clones only wins with unit weights.
    sil_objective: str = "advantage"
    # Raw-reward advantage floor below which the SIL gate stays closed.
    sil_advantage_floor: float = 0.25
    # Percentiles at which the gate opens and saturates.
    sil_gate_open_percentile: float = 80.0
    sil_gate_saturation_percentile: float = 95.0
    # Maximum transitions from one episode in a SIL/calibration sample.
    sil_samples_per_episode: int = 8
    # Attempted groups per update (not epoch); empty gates still consume the budget.
    sil_logical_minibatches_per_update: int = 1
    # Decay to this coefficient over sil_decay_fraction of the pinned schedule horizon.
    sil_coeff_final: float = 0.0
    sil_decay_fraction: float = 1.0
    # Updates between actor-gradient ratio/cosine diagnostics; no adaptive control.
    sil_grad_diagnostics_interval: int = 10


def _ppo_run_provenance(config: PPOConfig) -> dict[str, str] | None:
    values = (config.ppo_run_uuid, config.ppo_source_sha256, config.ppo_recipe_id)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("ppo_run_uuid, ppo_source_sha256, and ppo_recipe_id must be set together")
    return {
        "run_uuid": str(config.ppo_run_uuid),
        "source_sha256": str(config.ppo_source_sha256).lower(),
        "recipe_id": str(config.ppo_recipe_id),
    }


def _validate_resume_provenance(config: PPOConfig, resume_state: dict | None) -> None:
    """Require exact active/saved run identity before strict resume proceeds."""

    if resume_state is None:
        return
    active = _ppo_run_provenance(config)
    saved = resume_state.get("ppo_run_provenance")
    if active is None and saved is None:
        return
    if active is None:
        raise RuntimeError(
            "Strict resume checkpoint is provenance-bound; pass its explicit PPO run UUID, source SHA256, "
            "and recipe identity."
        )
    if not isinstance(saved, dict):
        raise RuntimeError("Strict resume checkpoint lacks required ppo_run_provenance")
    normalized_saved = {
        "run_uuid": str(saved.get("run_uuid", "")),
        "source_sha256": str(saved.get("source_sha256", "")).lower(),
        "recipe_id": str(saved.get("recipe_id", "")),
    }
    if normalized_saved != active:
        raise RuntimeError(f"Strict resume PPO run provenance mismatch: saved={normalized_saved!r}, active={active!r}")


def _seed_training_rngs(seed: int) -> None:
    """Seed process RNGs used by model initialization and PPO sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_ppo_config(config: PPOConfig) -> None:
    """Raise ValueError for invalid combinations; warn on risky ones."""
    from .cohort_config import validate_cohort_config

    validate_cohort_config(config)
    if config.precision not in (None, "fp32", "bf16"):
        raise ValueError("precision must be 'fp32', 'bf16', or None")
    validate_critic_win_ante(config.win_ante if config.win_ante is not None else 8)
    if not math.isfinite(config.return_path_coeff) or config.return_path_coeff < 0:
        raise ValueError("return_path_coeff must be finite and non-negative")
    if min(config.return_path_capacity, config.return_path_batch_size,
           config.return_path_samples_per_episode, config.return_path_rebuild_steps) < 1:
        raise ValueError("Return-path replay capacities and budgets must be positive")
    if config.archive_config is not None and (config.win_ante or 8) != 8:
        raise ValueError("Archive training requires win_ante=8")
    if (
        config.reward_config is not None
        and config.reward_config.objective == "milestone"
        and (config.win_ante or 8) != 8
    ):
        raise ValueError("Milestone rewards require win_ante=8")
    if not 0 <= config.milestone_final_scale <= 1 or not 0 < config.milestone_decay_fraction <= 1:
        raise ValueError("Invalid milestone annealing schedule")
    if not 0 <= config.seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    for name in ("log_interval", "checkpoint_interval", "eval_interval"):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.eval_games < 1 or config.eval_sampled_games < 0:
        raise ValueError("eval_games must be positive and eval_sampled_games non-negative")
    if config.eval_cpu_threads < 0:
        raise ValueError("eval_cpu_threads must be non-negative")
    if config.eval_workers < 0:
        raise ValueError("eval_workers must be non-negative")
    if config.eval_regression_tolerance is not None and not 0.0 < config.eval_regression_tolerance <= 1.0:
        raise ValueError("eval_regression_tolerance must be in (0, 1] when set")
    if config.eval_regression_patience <= 0:
        raise ValueError("eval_regression_patience must be positive")
    provenance = _ppo_run_provenance(config)
    if provenance is not None:
        try:
            parsed_uuid = uuid.UUID(provenance["run_uuid"])
        except ValueError as exc:
            raise ValueError("ppo_run_uuid must be a valid UUID") from exc
        if str(parsed_uuid) != provenance["run_uuid"]:
            raise ValueError("ppo_run_uuid must use canonical lowercase UUID form")
        source_sha256 = provenance["source_sha256"]
        if len(source_sha256) != 64 or any(character not in "0123456789abcdef" for character in source_sha256):
            raise ValueError("ppo_source_sha256 must be a 64-character hexadecimal SHA256")
        if not provenance["recipe_id"].strip():
            raise ValueError("ppo_recipe_id must be non-empty")
    for name in ("ppo_epochs", "mini_batch_size", "micro_batch_size", "rollout_length", "max_no_progress_steps"):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.counterfactual_diagnostic_interval < 0:
        raise ValueError("counterfactual_diagnostic_interval must be non-negative")
    if not np.isfinite(config.lr) or config.lr <= 0.0:
        raise ValueError("lr must be finite and positive")
    if config.rollout_temperature <= 0.0:
        raise ValueError("rollout_temperature must be positive")
    if config.danger_rollout_temperature is not None and config.danger_rollout_temperature <= 0.0:
        raise ValueError("danger_rollout_temperature must be positive when set")
    if not 0.0 <= config.danger_death_probability_threshold <= 1.0:
        raise ValueError("danger_death_probability_threshold must be between 0 and 1")
    if config.entropy_coeff < 0.0:
        raise ValueError("entropy_coeff must be non-negative")
    if config.outcome_loss_coeff < 0.0:
        raise ValueError("outcome_loss_coeff must be non-negative")
    for name in ("terminal_replay_batch_size", "terminal_replay_min_episodes", "terminal_replay_samples_per_episode"):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.terminal_replay_updates_per_ppo_update < 0:
        raise ValueError("terminal_replay_updates_per_ppo_update must be non-negative")
    if config.terminal_replay_holdout_batch_size <= 0:
        raise ValueError("terminal_replay_holdout_batch_size must be positive")
    if not 0.0 <= config.terminal_replay_holdout_fraction < 1.0:
        raise ValueError("terminal_replay_holdout_fraction must be in [0, 1)")
    if config.target_kl is not None and config.target_kl <= 0.0:
        raise ValueError("target_kl must be positive when set")
    if config.target_kl_p95 is not None and config.target_kl_p95 <= 0.0:
        raise ValueError("target_kl_p95 must be positive when set")
    if config.target_kl_max is not None and config.target_kl_max <= 0.0:
        raise ValueError("target_kl_max must be positive when set")
    if config.min_minibatch_fraction is not None and not 0.0 <= config.min_minibatch_fraction <= 1.0:
        raise ValueError("min_minibatch_fraction must be between 0 and 1")
    if config.action_type_entropy_scale < 0.0:
        raise ValueError("action_type_entropy_scale must be non-negative")
    if config.sil_coeff < 0.0:
        raise ValueError("sil_coeff must be non-negative")
    for name in ("sil_buffer_episodes", "sil_batch_size", "sil_min_episodes"):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.sil_objective not in ("advantage", "winning_bc"):
        raise ValueError(f"sil_objective must be 'advantage' or 'winning_bc', got {config.sil_objective!r}")
    if config.sil_advantage_floor < 0.0:
        raise ValueError("sil_advantage_floor must be non-negative")
    if not 0.0 <= config.sil_gate_open_percentile <= 100.0:
        raise ValueError("sil_gate_open_percentile must be between 0 and 100")
    if not 0.0 <= config.sil_gate_saturation_percentile <= 100.0:
        raise ValueError("sil_gate_saturation_percentile must be between 0 and 100")
    if config.sil_gate_saturation_percentile < config.sil_gate_open_percentile:
        raise ValueError("sil_gate_saturation_percentile must be >= sil_gate_open_percentile")
    if config.sil_samples_per_episode <= 0:
        raise ValueError("sil_samples_per_episode must be positive")
    if config.sil_logical_minibatches_per_update not in (1, 2):
        raise ValueError("sil_logical_minibatches_per_update must be 1 or 2")
    if config.sil_coeff_final < 0.0:
        raise ValueError("sil_coeff_final must be non-negative")
    if not 0.0 <= config.sil_decay_fraction <= 1.0:
        raise ValueError("sil_decay_fraction must be between 0 and 1")
    if config.sil_grad_diagnostics_interval <= 0:
        raise ValueError("sil_grad_diagnostics_interval must be positive")
    if config.advantage_clip_sigma < 0.0:
        raise ValueError("advantage_clip_sigma must be non-negative (0 disables)")
    if config.adaptive_entropy:
        if config.entropy_coeff <= 0.0:
            raise ValueError("entropy_coeff must be positive when adaptive entropy is enabled")
        if config.alpha_lr <= 0.0:
            raise ValueError("alpha_lr must be positive when adaptive entropy is enabled")
        if config.alpha_min <= 0.0:
            raise ValueError("alpha_min must be positive when adaptive entropy is enabled")
        if config.alpha_max < config.alpha_min:
            raise ValueError("alpha_max must be greater than or equal to alpha_min")
        if not 0.0 <= config.target_entropy <= 1.0:
            raise ValueError("target_entropy must be between 0 and 1 when using normalized entropy")
        if not 0.0 <= config.entropy_ema_beta < 1.0:
            raise ValueError("entropy_ema_beta must be in [0, 1)")
        if config.target_entropy < 0.1:
            logger.warning(
                "target_entropy=%.3f is a very low normalized entropy target; "
                "it will aggressively push the policy toward near-deterministic behavior.",
                config.target_entropy,
            )


def resolve_sil_coeff(
    config: PPOConfig,
    total_steps: int,
    schedule_total_steps: int | None = None,
) -> float:
    """Return the runtime SIL coefficient at ``total_steps``.

    * ``sil_coeff <= 0`` -> SIL is fully disabled (returns 0.0). This is the
      exact no-SIL switch.
    * Otherwise the coefficient linearly decays from ``sil_coeff`` toward
      ``sil_coeff_final`` (clamped to not exceed ``sil_coeff``) over
      ``sil_decay_fraction`` of the pinned schedule horizon, then stays at the
      final value.
    * ``schedule_total_steps`` pins the anneal horizon independently of
      ``config.total_timesteps`` so resume with a different env count does not
      rewind an already-decayed coefficient. The train loop passes the same
      pinned horizon used by the training run.
    """
    if config.sil_coeff <= 0.0:
        return 0.0
    final = min(float(config.sil_coeff_final), float(config.sil_coeff))
    decay_fraction = min(max(float(config.sil_decay_fraction), 0.0), 1.0)
    horizon = schedule_total_steps if schedule_total_steps is not None else config.total_timesteps
    decay_steps = max(1, int(horizon * decay_fraction))
    progress = min(1.0, total_steps / decay_steps)
    return max(final, config.sil_coeff - (config.sil_coeff - final) * progress)


def _resolve_schedule_total_steps(config: PPOConfig, resume_state: dict | None) -> int:
    """Return the anneal horizon (in env steps) for fraction-of-training schedules.

    Fresh runs anchor to the run's own ``total_timesteps``. Resumed runs restore
    the horizon persisted in the checkpoint so schedules continue exactly where
    they left off — resume recomputes ``config.total_timesteps`` from the current
    ``num_envs``/update target, and annealing against the recomputed value moves
    already-decayed coefficients backward. ``reset_schedules`` opts back into
    re-anchoring; checkpoints from before this field existed re-anchor with a
    loud warning because the original horizon is unknowable.
    """
    if resume_state is None:
        return config.total_timesteps
    if config.reset_schedules:
        logger.warning(
            "reset_schedules: re-anchoring anneal schedules to this leg's "
            "horizon of %d steps. Already-"
            "decayed coefficients will climb back toward their start values.",
            config.total_timesteps,
        )
        return config.total_timesteps
    saved = resume_state.get("schedule_total_steps")
    if saved is not None:
        if int(saved) != config.total_timesteps:
            logger.info(
                "Anneal schedules pinned to the original horizon of %d steps "
                "(this leg's recomputed total_timesteps is %d). Pass "
                "--reset-schedules to re-anchor intentionally.",
                int(saved),
                config.total_timesteps,
            )
        return int(saved)
    logger.warning(
        "Checkpoint predates schedule_total_steps; anneal schedules re-derive "
        "from this leg's total_timesteps=%d. If num_envs or the update target "
        "grew, previously-decayed coefficients will rewind.",
        config.total_timesteps,
    )
    return config.total_timesteps


def resolve_milestone_scale(config: PPOConfig, total_steps: int, schedule_total_steps: int) -> float:
    """Anneal a bounded auxiliary objective on the checkpointed step horizon."""
    duration = max(1.0, schedule_total_steps * config.milestone_decay_fraction)
    fraction = min(max(total_steps / duration, 0.0), 1.0)
    return 1.0 + fraction * (config.milestone_final_scale - 1.0)


def _effective_reward_config(config: PPOConfig) -> RewardConfig:
    """Return reward targets aligned with PPO discounting and victory Ante."""
    base = config.reward_config if config.reward_config is not None else DEFAULT_REWARD_CONFIG
    return replace(
        base,
        gamma=config.gamma,
        potential_win_ante=config.win_ante or 8,
    )
