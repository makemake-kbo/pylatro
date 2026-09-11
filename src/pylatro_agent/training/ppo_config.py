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
    archive_config: ArchiveConfig | None = None
    # Separate BC of verified early prefixes behind archive wins. Never feeds
    # PPO ratios/GAE or the terminal critic. Zero is the exact no-return-BC control.
    return_path_coeff: float = 0.02
    return_path_capacity: int = 32
    return_path_batch_size: int = 64
    return_path_samples_per_episode: int = 8
    return_path_rebuild_steps: int = 64
    milestone_final_scale: float = 0.0
    milestone_decay_fraction: float = 0.8
    num_envs: int = 32
    seed: int = 0
    rollout_length: int = 256
    total_timesteps: int = 1_000_000
    # When set, overrides total_timesteps: the run trains for exactly this many
    # PPO updates and total_timesteps is derived as total_updates * steps_per_update.
    # Useful for "N more updates" semantics without doing the timesteps math.
    total_updates: int | None = None
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    gamma: float = 0.997
    gae_lambda: float = 0.97
    clip_epsilon: float = 0.1  # PPO clip range; tighter than the usual 0.2
    target_kl: float | None = (
        0.05  # Phase 4: raised from 0.03; chronic KL-stop means the step size is wrong, not the trust region
    )
    # Optional trust-region guards. P95 remains diagnostic; max rejects and
    # rolls back the complete PPO update. The minimum fraction prevents a soft
    # target-KL stop until enough minibatches have been processed.
    target_kl_p95: float | None = None
    target_kl_max: float | None = None
    min_minibatch_fraction: float | None = None
    entropy_coeff: float = (
        0.01  # Phase 4: fixed at 0.01 (was 0.001); mixture eps is now the primary exploration mechanism
    )
    adaptive_entropy: bool = False
    target_entropy: float = 0.15
    alpha_lr: float = 1e-2
    alpha_min: float = 0.001
    alpha_max: float = 0.05
    entropy_ema_beta: float = 0.6
    action_type_entropy_scale: float = 0.0
    value_loss_coeff: float = 0.25
    max_grad_norm: float = 0.5
    lr: float = 5e-5  # Phase 4: lowered from 1e-4; chronic KL-stop at lr=1e-4 means the step size was too large
    device: str = "cpu"
    # None preserves saved model precision on resume (FP32 for older files).
    # A supplied value is an explicit override, logged by the trainer.
    precision: str | None = None
    save_dir: str = "checkpoints/ppo"
    log_dir: str = "runs/ppo"
    # Optional operator/watchdog request, checked only at completed-update
    # boundaries. Save a full resume checkpoint before honoring the stop.
    stop_request_path: str | None = None
    eval_interval: int = 5
    log_interval: int = 10
    checkpoint_interval: int = 50
    # Wins are rare even for the heuristic (~1% at ante 8, ~12% at ante 5,
    # ~39% at ante 4). 10 eval games can't measure rare-event winrate; bump
    # the default so eval/win_rate has signal to compare against.
    eval_games: int = 50
    # Reuse a subset of the reserved seeds with sampled policy actions to
    # evaluate the critic against the policy whose outcomes it predicts.
    eval_sampled_games: int = 25
    eval_before_training: bool = True
    # Stop cleanly when deterministic evaluation regresses materially from the
    # best checkpoint for consecutive evals. None disables the guard.
    eval_regression_tolerance: float | None = None
    eval_regression_patience: int = 2
    # Optional override for the device used during eval. When set (e.g. "cpu"),
    # eval games run on that device instead of the training device. Useful for
    # long MPS runs where the eval loop's serial forward-pass allocations
    # otherwise compete with idle AsyncVectorEnv workers for unified memory and
    # can trip the OS memory-pressure killer. The model is moved to the eval
    # device for the duration of evaluate_model and moved back afterward.
    eval_device: str | None = None
    # Games advanced in lockstep per eval forward pass. Eval used to run one
    # game at a time with batch-size-1 forwards and cost more wall clock than
    # the training between evals; batching amortizes the policy forward over
    # many games without changing any per-seed result.
    eval_batch_size: int = 32
    # Curriculum: cap the run's victory threshold below the engine default
    # (8). Heuristic-teacher win rates by ante are ~39% at 4, ~12% at 5,
    # ~2% at 6. Set to None for the standard ante-8 victory condition.
    win_ante: int | None = None
    # Stake (difficulty tier, 1-8) the training/eval envs run at. The self-play
    # curriculum ramps this; on its own PPO trains at the base stake.
    stake: int = 1
    # Keep self-induced action loops close enough for terminal stall credit to
    # reach the responsible decisions through GAE. Immediate idle penalties
    # remain the primary defense; this is the bounded backstop.
    max_no_progress_steps: int = 32
    # Physical samples per forward/backward pass. Logical ``mini_batch_size``
    # gradients are accumulated exactly across these chunks on both single-
    # and multi-GPU runs.
    micro_batch_size: int = 64
    async_envs: bool = True  # Use multiprocess envs (AsyncVectorEnv)
    # Sharpens the on-policy distribution for both rollout sampling and PPO loss
    # computation. The BC-pretrained policy at temperature=1 has chosen_action_prob ≈ 0.5
    # over ~250 valid actions per state, which means a 30-step sampled episode has ~0.5^30
    # probability of even matching its own greedy trajectory, sampled rollouts essentially
    # never win and PPO sees no positive advantage to lock onto. Sharpening the distribution
    # by a fixed factor at all sites (rollout, train forward, truncation bootstrap) keeps
    # PPO consistent, old_log_probs and new_log_probs are computed under the same
    # distribution, while letting the agent take competent actions in rollouts. Set to 1.0
    # to disable; lower for more deterministic behavior.
    rollout_temperature: float = (
        1.0  # Phase 4: was 0.7; the sharpening crutch now only suppresses exploration post-Phase-1 BC
    )
    # Optional state-dependent sharpening for active hand decisions. Ante 1 is
    # always protected; later antes use the explicit immediate-danger signal.
    # Shop / pack / blind-select states retain rollout_temperature so Joker and
    # consumable search remain exploratory.
    danger_rollout_temperature: float | None = None
    danger_death_probability_threshold: float = 0.35
    # Complete terminal outcomes supervise the conditional hazard model with
    # one categorical likelihood. Stalled/censored episodes remain masked.
    outcome_loss_coeff: float = 0.10
    # Always-on completed-episode replay for terminal-critic supervision.
    # Episode assembly is independent of SIL and crosses rollout boundaries.
    # Replay updates touch only the outcome tower (hazard projection plus its
    # output layer): the actor, shared trunk, pooling, and return residual are
    # unchanged.
    #
    # Volume matters here. At 32 envs x 256 steps a rollout is ~8k rows, so the
    # old 1x128 replay contributed under 2% of the outcome gradient while the
    # buffer turned over every ~15 updates, evicting ~98% of stored transitions
    # unsampled. 4x256 raises replay to ~1k rows per update against the same
    # rollout, at the cost of four extra forward passes.
    terminal_replay_batch_size: int = 256
    terminal_replay_min_episodes: int = 8
    terminal_replay_samples_per_episode: int = 8
    terminal_replay_updates_per_ppo_update: int = 4
    # Draw replay rows uniformly over transitions rather than over episodes.
    # Episode-uniform draws weight a 20-step Ante-1 death like a 300-step
    # Ante-5 win, starving the deep states the critic is worst at.
    terminal_replay_row_uniform: bool = True
    # Replay-only split: excluded from replay and SIL, but already seen by PPO.
    # It measures replay fitting, NOT independent critic generalization.
    # Truly separate critic metrics come from reserved fresh evaluation seeds
    # under eval/sampled/critic/*. Legacy config names remain load/CLI compatible.
    terminal_replay_holdout_fraction: float = 0.1
    terminal_replay_holdout_batch_size: int = 512
    # Optional reward settings threaded through BalatroEnv. None uses the
    # default potential-based reward configuration.
    reward_config: RewardConfig | None = None
    # Sampled exact joker-marginal validation. 0 disables. A positive N copies
    # one pre-play RunState and replays the same selected cards with one focal
    # joker removed every Nth actual play. Disabled by default because the state
    # copy is deliberately bounded but still material in many-env training.
    counterfactual_diagnostic_interval: int = 0
    # Append every resolved shop-leave risk forecast (predictions + realized
    # next-blind outcome) to <log_dir>/risk_forecasts.jsonl. These pairs are
    # the input to tools/fit_risk_calibration.py, which refits the analytic
    # death-probability Platt constants in pylatro_agent.risk.
    risk_forecast_log: bool = True
    # Fixed, versioned seed list for the in-training eval pass. Passing a
    # stable list (pylatro_agent.eval.EVAL_SEEDS_V1) makes every checkpoint's
    # eval reproducible and pairable across runs via the McNemar / paired
    # bootstrap harness in pylatro_agent.eval. None preserves the historical
    # 10000 + game_idx seeds.
    eval_seeds: list[int] | None = None
    # Mixture weight on the autoregressive hand/discard head. 0.0
    # selects candidate-only support; 0.1 (default for PPO)
    # gives the policy full support over every legal hand play while keeping
    # the candidate head dominant. The AR head is by this point a competent
    # proposal distribution (trained at eps=0.5 during BC), not noise.
    hand_ar_mixture_eps: float = 0.1
    # Optional cryptographic run identity. Production launchers provide all
    # three fields; every checkpoint then persists them and strict resume
    # requires an exact match before any environment is created.
    ppo_run_uuid: str | None = None
    ppo_source_sha256: str | None = None
    ppo_recipe_id: str | None = None
    # Discard the resumed checkpoint's best_eval_win_rate so ppo_best_eval.pt
    # selection restarts from scratch. Required when the eval task changes
    # (e.g. a --win-ante bump), otherwise no best-eval checkpoint is ever
    # written until the harder task beats the old task's record.
    reset_best_eval: bool = False
    # Re-anchor fraction-of-training anneal schedules to this leg's recomputed
    # total_timesteps instead of the horizon persisted in the checkpoint.
    reset_schedules: bool = False
    # Self-imitation (SIL-as-BC) on the agent's own winning episodes. Wins at
    # high win-ante targets are too sparse for on-policy PPO (a handful per
    # update); replaying complete winning trajectories as a behavior-cloning
    # term multiplies the win-signal density without touching the reward
    # function — only genuine wins enter the buffer, so there is nothing to
    # farm. The NLL term is added to the PPO minibatch objective (single
    # backward per minibatch) so the coefficient trades off directly against
    # the policy/entropy terms; a separate optimizer pass would let
    # Adam's gradient renormalization largely cancel the coefficient. Riding
    # inside the PPO loop also puts SIL movement under the target-kl guard.
    # 0.0 disables. The buffer is in-memory only; it refills over the first
    # ~buffer/wins-per-update updates after a resume.
    sil_coeff: float = 0.0
    # Legacy name retained for CLI/checkpoint compatibility. This is now the
    # shared completed-episode replay capacity used by the always-on terminal
    # critic and by SIL when SIL is enabled. Wins + ordinary losses are stored.
    sil_buffer_episodes: int = 256
    sil_batch_size: int = 64  # transitions sampled per PPO micro-batch
    # Skip SIL until the replay buffer holds this many completed episodes. Wins
    # and ordinary (non-stalled) losses both count: the buffer stores both now,
    # and it is the advantage gate below — not this threshold — that keeps an
    # early, win-poor buffer from driving the actor. Under "winning_bc" the
    # win-only sampler simply returns nothing until wins accumulate, so SIL is
    # still a no-op until then even though this counts all episodes.
    sil_min_episodes: int = 8
    # Clamp on globally-normalized advantages, in standard deviations. The
    # advantage distribution is heavy-tailed (kurtosis ~4.8 measured at
    # win_ante=5): near-terminal coin-flip states reach 6 sigma and the top 1%
    # of states carry ~20% of sum(adv^2), so a handful of aleatoric outcomes
    # dominate each update's policy gradient without registering in mean KL.
    # 0 disables.
    advantage_clip_sigma: float = 4.0
    # --- SIL redesign ---
    # SIL objective. "advantage" samples all valid completed episodes (wins and
    # losses), computes a current-critic MC advantage, and applies the robust
    # percentile gate below. "winning_bc" samples only winning episodes and
    # uses a unit gate (plain behavior cloning of wins), serving as a matched
    # control against advantage SIL. ``--sil-coeff 0`` remains the exact no-SIL
    # switch regardless of objective.
    sil_objective: str = "advantage"
    # Absolute advantage floor (raw reward units). Sub-floor positive advantages
    # receive zero gate weight, and the percentile gate only opens when the
    # open-percentile exceeds this floor.
    sil_advantage_floor: float = 0.25
    # Percentiles of the eligible raw-advantage distribution at which the gate
    # opens (80th) and saturates (95th).
    sil_gate_open_percentile: float = 80.0
    sil_gate_saturation_percentile: float = 95.0
    # Maximum transitions one episode may contribute to a SIL / calibration
    # batch. Episode-uniform sampling plus this cap means a long episode is not
    # privileged over a short one.
    sil_samples_per_episode: int = 8
    # Number of logical PPO optimizer minibatches (accumulated optimizer steps)
    # per update that may attempt SIL. 1 or 2. The budget is per update, not per
    # PPO epoch, and attempted groups count even when their gate is empty so
    # training does not keep resampling for a favorable batch.
    sil_logical_minibatches_per_update: int = 1
    # Resume-safe linear decay for the SIL coefficient. Decays monotonically
    # from ``sil_coeff`` to ``sil_coeff_final`` over ``sil_decay_fraction`` of
    # the pinned schedule horizon; never rewinds on resume.
    sil_coeff_final: float = 0.0
    sil_decay_fraction: float = 1.0
    # Interval (in updates) at which weighted SIL / PPO actor-gradient ratio and
    # cosine similarity are logged. Diagnostic only; no adaptive control.
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
    if config.log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if config.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if config.eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    if config.eval_games < 1 or config.eval_sampled_games < 0:
        raise ValueError("eval_games must be positive and eval_sampled_games non-negative")
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
    if config.ppo_epochs <= 0:
        raise ValueError("ppo_epochs must be positive")
    if config.mini_batch_size <= 0:
        raise ValueError("mini_batch_size must be positive")
    if config.micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    if config.rollout_length <= 0:
        raise ValueError("rollout_length must be positive")
    if config.max_no_progress_steps <= 0:
        raise ValueError("max_no_progress_steps must be positive")
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
    if config.terminal_replay_batch_size <= 0:
        raise ValueError("terminal_replay_batch_size must be positive")
    if config.terminal_replay_min_episodes <= 0:
        raise ValueError("terminal_replay_min_episodes must be positive")
    if config.terminal_replay_samples_per_episode <= 0:
        raise ValueError("terminal_replay_samples_per_episode must be positive")
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
    if config.sil_buffer_episodes <= 0:
        raise ValueError("sil_buffer_episodes must be positive")
    if config.sil_batch_size <= 0:
        raise ValueError("sil_batch_size must be positive")
    if config.sil_min_episodes <= 0:
        raise ValueError("sil_min_episodes must be positive")
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
