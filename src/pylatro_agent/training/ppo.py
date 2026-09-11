"""PPO training orchestration and backwards-compatible public entry points."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pylatro import GameData, load_game_data

from ..agent import AgentConfig, BalatroAgent
from ..vocab import build_vocab
from .ppo_checkpoint import (
    _load_checkpoint_strict,
    _load_state_dict_into_model,
    _mirror_latest_checkpoint,
    _save_checkpoint,
)
from .ppo_checkpoint import (
    load_actor_transfer as load_actor_transfer,
)
from .ppo_config import (
    PPOConfig as PPOConfig,
)
from .ppo_config import (
    _effective_reward_config,
    _resolve_schedule_total_steps,
    _seed_training_rngs,
    _validate_ppo_config,
    _validate_resume_provenance,
)
from .ppo_config import (
    resolve_milestone_scale as resolve_milestone_scale,
)
from .ppo_config import (
    resolve_sil_coeff as resolve_sil_coeff,
)
from .ppo_evaluation import (
    _next_eval_regression_streak,
    evaluation_rank,
    evaluation_seed_list,
    validate_resume_evaluation_seeds,
)
from .ppo_evaluation import (
    evaluate_model as evaluate_model,
)
from .ppo_evaluation import (
    run_seed_evaluation as run_seed_evaluation,
)
from .ppo_metrics import (
    _safe_mean,
    write_rollout_metrics,
)
from .ppo_observations import (
    _ObsBuffer,
)
from .ppo_optimization import (
    _entropy_alpha_loss,
    _make_alpha_optimizer,
    _make_policy_optimizer,
    _optimizer_to,
    _physical_minibatch_count,
    _restore_policy_optimizer_state,
    _run_ppo_update,
    _run_terminal_replay_updates,
    _sample_weighted_mean,
    _smoothed_entropy_signal,
)
from .ppo_rollout import (
    RolloutState,
    _make_vectorized_envs,
    collect_rollout,
)
from .return_paths import ReturnPathReplay
from .sil import (
    EpisodeReplayBuffer,
)

logger = logging.getLogger(__name__)


def train_ppo(
    config: PPOConfig,
    agent_config: AgentConfig | None = None,
    pretrained_path: str | None = None,
    resume_path: str | None = None,
    additional_updates: int | None = None,
    data: GameData | None = None,
    actor_transfer_path: str | None = None,
) -> BalatroAgent:
    """Run PPO training with vectorized environments.

    Resume semantics:

    * ``pretrained_path``: weights-only init. Optimizer, counters, and entropy
      controller all start fresh at update 0.
    * ``resume_path``: strict resume. Loads optimizer state, update counter,
      total_steps, entropy controller, and RNG. Requires a
      full PPO checkpoint (``checkpoint_format == "ppo_full"``); a weights-only
      checkpoint raises a clear error pointing at ``--pretrained``.
    * ``additional_updates``: only meaningful with ``resume_path``. Runs that
      many more updates *after* the checkpoint's saved update count.
    """
    _validate_ppo_config(config)
    # Work with a private RewardConfig copy and pin its potential discount to
    # the PPO return discount before resume validation or environment creation.
    config.reward_config = _effective_reward_config(config)
    _seed_training_rngs(config.seed)
    if data is None:
        data = load_game_data()
    vocab = build_vocab(data)
    if agent_config is None:
        agent_config = AgentConfig()
    # Apply the PPO-phase mixture weight so the AR head provides targeted
    # exploration alongside the candidate head (Phase 1.2). The model reads
    # this from its config at every action_distribution() call.
    agent_config.hand_ar_mixture_eps = config.hand_ar_mixture_eps
    agent_config.win_only_value = config.reward_config.objective == "milestone"

    device = torch.device(config.device)
    use_pin_memory = device.type == "cuda"
    model = BalatroAgent(agent_config, vocab).to(device)

    if sum(bool(path) for path in (resume_path, pretrained_path, actor_transfer_path)) > 1:
        raise ValueError("Choose one of resume, pretrained, or actor transfer")
    if actor_transfer_path and (config.win_ante or 8) != 8:
        raise ValueError("Actor transfer initializes the fixed Ante-8 task")

    if additional_updates is not None and not resume_path:
        raise ValueError("--additional-updates requires --resume PATH")

    # === Strict resume: load full PPO state before optimizer creation ===
    resume_state: dict | None = None
    if resume_path:
        from ..checkpoint import load_ppo_resume_payload, restore_rng_states

        resume_state = load_ppo_resume_payload(
            resume_path,
            device,
            active_reward_config=config.reward_config,
            active_win_ante=config.win_ante,
        )
        _load_state_dict_into_model(model, resume_state["state_dict"], resume_path)
        if resume_state.get("actor_transfer"):
            model._actor_transfer_metadata = resume_state["actor_transfer"]
        saved_config = resume_state.get("ppo_config_fields") or {}
        validate_resume_evaluation_seeds(config, saved_config)
        model._training_reserved_seeds = resume_state.get("training_reserved_seeds")
        archive_config = asdict(config.archive_config) if config.archive_config else None
        if saved_config.get("archive_config") != archive_config:
            raise ValueError("Archive configuration changed on resume; use the saved archive settings")
        for key in ("milestone_final_scale", "milestone_decay_fraction"):
            if key in saved_config and saved_config[key] != getattr(config, key):
                raise ValueError(f"{key} changed on resume; keep the saved reward schedule")
        logger.info(
            "Resumed model weights from %s (update_count=%d, total_steps=%d)",
            resume_path,
            resume_state.get("update_count", 0),
            resume_state.get("total_steps", 0),
        )
        restore_rng_states(resume_state.get("rng_states", {}))
    elif pretrained_path:
        _load_checkpoint_strict(
            model,
            pretrained_path,
            device,
            active_reward_config=config.reward_config,
        )
        logger.info("Loaded pretrained model from %s", pretrained_path)
    elif actor_transfer_path:
        load_actor_transfer(model, actor_transfer_path, device)
    _validate_resume_provenance(config, resume_state)
    eval_panel = evaluation_seed_list(config.eval_games, config.eval_seeds)
    known_reserved = getattr(model, "_training_reserved_seeds", eval_panel)
    model._training_reserved_seeds = (
        sorted(set(known_reserved) & set(eval_panel)) if known_reserved is not None else None
    )
    if model._training_reserved_seeds != sorted(eval_panel):
        logger.warning("Evaluation seeds are excluded from this PPO run, but complete source-training "
                       "disjointness is unverified; do not claim these metrics are wholly unseen-data results")
    if resume_state is not None and not config.reset_best_eval and "best_eval_rank" in resume_state:
        model._best_eval_rank = tuple(resume_state["best_eval_rank"])
    saved_precision = (resume_state.get("agent_config") or {}).get("precision", "fp32") if resume_state else None
    requested_precision = config.precision
    agent_config.precision = requested_precision or saved_precision or agent_config.precision
    config.precision = agent_config.precision
    if saved_precision is not None and requested_precision is not None and requested_precision != saved_precision:
        logger.warning("Explicit compute precision change on resume: %s -> %s", saved_precision, requested_precision)
    logger.info(
        "Transformer precision: %s (CUDA only); parameters, RL heads, and optimizer remain FP32", config.precision
    )
    from ..precision import backbone_autocast

    with backbone_autocast(config.precision, device):
        pass  # Validate saved settings and hardware before starting env workers.
    if config.precision == "bf16" and device.type != "cuda":
        logger.warning("BF16 acceleration is CUDA-only; this %s run computes in FP32", device.type)
    if config.danger_rollout_temperature is None:
        logger.info(
            "Rollout temperature: %.3f (applied to rollout, training, and bootstrap)",
            config.rollout_temperature,
        )
    else:
        logger.info(
            "Rollout temperature: base=%.3f, active-hand danger/Ante-1=%.3f "
            "(death_probability>=%.2f; shop temperature unchanged)",
            config.rollout_temperature,
            min(config.danger_rollout_temperature, config.rollout_temperature),
            config.danger_death_probability_threshold,
        )
    logger.info(
        "Danger policy conditioning: direct features enabled; unsafe-shop leave logit penalty=%.2f",
        max(float(agent_config.danger_shop_leave_logit_penalty), 0.0),
    )
    # Discount-horizon diagnostic for explicit low-gamma overrides. At
    # gamma=0.99 a terminal reward is discounted to ~0.22 at 150 steps and
    # ~0.05 at 300, making early-game economy decisions nearly invisible.
    # The default 0.997 retains ~0.41 at 300 steps.
    effective_win_ante = config.win_ante if config.win_ante is not None else 8
    if effective_win_ante >= 6 and config.gamma < 0.995:
        logger.warning(
            "gamma=%.4f with win_ante=%d: terminal reward is discounted to "
            "%.3f at ~300 steps. Consider --gamma 0.997 for long-horizon runs "
            "(Phase 3.3).",
            config.gamma,
            effective_win_ante,
            config.gamma**300,
        )
    use_multi_gpu = config.device == "cuda" and torch.cuda.device_count() > 1
    if use_multi_gpu:
        model = nn.DataParallel(model)
    effective_batch_size = max(1, min(config.micro_batch_size, config.mini_batch_size))
    accum_steps = max(1, math.ceil(config.mini_batch_size / effective_batch_size))
    if accum_steps > 1:
        logger.info(
            "PPO gradient accumulation: up to %d physical microbatches per exact logical batch "
            "(logical=%d, physical<=%d, DataParallel=%s).",
            accum_steps,
            config.mini_batch_size,
            effective_batch_size,
            use_multi_gpu,
        )

    optimizer = _make_policy_optimizer(model.parameters(), config.lr)
    if resume_state is not None:
        active_lr = _restore_policy_optimizer_state(
            optimizer,
            resume_state,
            config,
            device,
        )
        logger.info(
            "Restored PPO optimizer state (Adam moments + counters) from checkpoint at active LR %.2e.",
            active_lr,
        )

    # Create vectorized environments. Each env replays a deterministic
    # game-seed stream from its constructor seed, so resuming with plain
    # 0..N-1 seeds replays the same opening game library every leg and the
    # critic overfits to it (novel seeds then produce systematically noisy
    # advantages — the env-count-change collapse). Offset the seeds by the
    # resumed update counter so every leg trains on fresh streams. A large
    # config-seed stride keeps neighboring experiment seeds from sharing all
    # but one environment stream. Seed 0 preserves the historical stream.
    env_seed_base = config.seed * 10_000_019
    if resume_state is not None:
        env_seed_base += int(resume_state.get("update_count", 0)) * 1_000_003
    logger.info("Env seed base for this leg: %d", env_seed_base)
    vec_env = _make_vectorized_envs(
        config.num_envs,
        data,
        vocab,
        stake=config.stake,
        max_no_progress_steps=config.max_no_progress_steps,
        use_async=config.async_envs,
        win_ante=config.win_ante,
        reward_config=config.reward_config,
        env_seed_base=env_seed_base,
        counterfactual_diagnostic_interval=config.counterfactual_diagnostic_interval,
        archive_config=config.archive_config,
        excluded_seeds=tuple(evaluation_seed_list(config.eval_games, config.eval_seeds)),
    )
    if resume_state is not None and config.archive_config is not None:
        archive_states = resume_state.get("archive_states")
        if not isinstance(archive_states, list) or len(archive_states) != config.num_envs:
            vec_env.close()
            raise ValueError("Archive resume requires saved archives and the original number of environments")
        vec_env.call("load_archive_states", archive_states)
    obs_dict, _reset_info = vec_env.reset()
    # Pre-allocate obs tensors for batched inference
    obs_buf = _ObsBuffer(config.num_envs, device)
    obs_buf.update(obs_dict)

    from torch.utils.tensorboard import SummaryWriter

    save_path = Path(config.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(config.log_dir)

    steps_per_update = config.num_envs * config.rollout_length
    planned_updates = max(1, math.ceil(config.total_timesteps / steps_per_update))
    if config.total_updates is not None:
        # --updates N semantics: run exactly N updates, deriving timesteps from it.
        planned_updates = config.total_updates
        config.total_timesteps = planned_updates * steps_per_update
    if config.total_timesteps % steps_per_update != 0:
        logger.info(
            "PPO total_timesteps=%d is not divisible by steps_per_update=%d; "
            "the final update will overshoot to %d total env steps.",
            config.total_timesteps,
            steps_per_update,
            planned_updates * steps_per_update,
        )
    if planned_updates < 20:
        logger.warning(
            "PPO is configured for only %d policy updates (%d steps/update). "
            "This is usually too few to debug or improve a weak prior; "
            "lower rollout_length or raise total_timesteps.",
            planned_updates,
            steps_per_update,
        )
    logger.info(
        "Starting PPO training: total_timesteps=%d, steps_per_update=%d, planned_updates=%d, "
        "ppo_epochs=%d, lr=%.2e, entropy_coeff=%.5f, adaptive_entropy=%s, "
        "target_entropy=%.3f, alpha_lr=%.2e, action_type_entropy_scale=%.2f, "
        "target_kl=%s, max_no_progress_steps=%d, log_interval=%d, checkpoint_interval=%d, eval_interval=%d",
        config.total_timesteps,
        steps_per_update,
        planned_updates,
        config.ppo_epochs,
        config.lr,
        config.entropy_coeff,
        config.adaptive_entropy,
        config.target_entropy,
        config.alpha_lr,
        config.action_type_entropy_scale,
        "None" if config.target_kl is None else f"{config.target_kl:.5f}",
        config.max_no_progress_steps,
        config.log_interval,
        config.checkpoint_interval,
        config.eval_interval,
    )

    total_steps = 0
    update_count = 0
    entropy_coeff = config.entropy_coeff
    if config.adaptive_entropy:
        # Adaptive entropy coefficient driven by normalized entropy.
        log_alpha = torch.tensor(
            np.log(config.entropy_coeff),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        alpha_optimizer = _make_alpha_optimizer(log_alpha, config.alpha_lr)
    else:
        log_alpha = None
        alpha_optimizer = None
    entropy_signal_ema: float | None = None

    # === Restore mutable training state from the resume checkpoint ===
    if resume_state is not None:
        update_count = int(resume_state.get("update_count", 0))
        total_steps = int(resume_state.get("total_steps", 0))
        saved_entropy_coeff = resume_state.get("entropy_coeff")
        if config.adaptive_entropy:
            # Adaptive mode: the controller state in the checkpoint is the
            # source of truth; the CLI value is only the initial seed.
            if saved_entropy_coeff is not None:
                entropy_coeff = float(saved_entropy_coeff)
        elif saved_entropy_coeff is not None and abs(float(saved_entropy_coeff) - entropy_coeff) > 1e-12:
            # Fixed-coefficient mode: the CLI value is authoritative, mirroring
            # _apply_lr_override, silently restoring the checkpoint's coeff
            # would make --entropy-coeff a no-op on resume.
            logger.warning(
                "Overriding entropy_coeff from checkpoint %.5f to CLI %.5f.",
                float(saved_entropy_coeff),
                entropy_coeff,
            )
        saved_ema = resume_state.get("entropy_signal_ema")
        if saved_ema is not None:
            entropy_signal_ema = float(saved_ema)
        if config.adaptive_entropy:
            assert log_alpha is not None
            assert alpha_optimizer is not None
            saved_log_alpha = resume_state.get("log_alpha")
            if saved_log_alpha is not None:
                with torch.no_grad():
                    log_alpha.copy_(saved_log_alpha.to(device))
                if "alpha_optimizer_state_dict" in resume_state:
                    alpha_optimizer.load_state_dict(resume_state["alpha_optimizer_state_dict"])
                    _optimizer_to(alpha_optimizer, device)
                logger.info(
                    "Restored adaptive-entropy controller: entropy_coeff=%.5f, signal_ema=%s",
                    entropy_coeff,
                    "None" if entropy_signal_ema is None else f"{entropy_signal_ema:.4f}",
                )
        # Resolve the target update count for this resumed run, in priority order:
        #   1. --additional-updates N  -> relative: saved_count + N
        #   2. --updates N (--total-updates) -> absolute target N
        #   3. checkpoint's saved planned_updates (continue the original plan)
        if additional_updates is not None:
            planned_updates = update_count + additional_updates
            config.total_timesteps = planned_updates * steps_per_update
            logger.info(
                "Resuming from update %d: running %d additional updates (target %d).",
                update_count,
                additional_updates,
                planned_updates,
            )
        elif config.total_updates is not None:
            planned_updates = config.total_updates
            config.total_timesteps = planned_updates * steps_per_update
            logger.info(
                "Resuming from update %d toward absolute target %d updates (--updates).",
                update_count,
                planned_updates,
            )
        else:
            saved_planned = resume_state.get("planned_updates")
            if saved_planned is not None and saved_planned > update_count:
                planned_updates = int(saved_planned)
                config.total_timesteps = planned_updates * steps_per_update
            logger.info(
                "Resuming from update %d toward planned_updates=%d.",
                update_count,
                planned_updates,
            )
    # Anneal horizon for fraction-of-training schedules. Resolved after resume
    # finalizes
    # total_timesteps, persisted in every checkpoint so later legs keep the
    # original horizon regardless of num_envs/update-target changes.
    schedule_total_steps = _resolve_schedule_total_steps(config, resume_state)
    rollout_state = RolloutState(config)
    # Phase 4: consecutive-minibatch-fraction tracker for the chronic KL-stop alert.
    _low_minibatch_streak = 0
    # Phase 6: rolling win_rate/ep_reward history for the correlation acceptance
    # criterion (corr > 0.5). Currently ~0/negative because dense shaping is
    # farmable independent of winning.
    # Best-eval tracking for ppo_best_eval.pt selection. On resume, carry the
    # saved best forward so the resumed run keeps the prior best unless it beats it.
    best_eval_win_rate: float | None = None
    best_eval_update: int | None = None
    eval_regression_streak = 0
    early_stop_requested = False
    if resume_state is not None and not config.reset_best_eval:
        best_eval_win_rate = resume_state.get("best_eval_win_rate")
        best_eval_update = resume_state.get("best_eval_update")
    elif resume_state is not None:
        logger.info(
            "reset_best_eval: discarding resumed best_eval_win_rate=%s (update %s); "
            "best-eval selection restarts from scratch",
            resume_state.get("best_eval_win_rate"),
            resume_state.get("best_eval_update"),
        )
    # Always-on complete-episode assembly. The legacy SIL replay capacity is
    # reused so terminal critic replay and optional SIL share one compact copy
    # of recent observations. SIL still sees no buffer when its coefficient is
    # disabled, preserving its exact actor-loss switch.
    episode_buffer = EpisodeReplayBuffer(
        config.sil_buffer_episodes,
        seed=config.seed,
        holdout_fraction=config.terminal_replay_holdout_fraction,
    )
    sil_buffer = episode_buffer if config.sil_coeff > 0.0 else None
    return_path_replay = None
    risk_forecast_path = Path(config.log_dir) / "risk_forecasts.jsonl"
    risk_forecast_file = None
    try:
        if config.archive_config is not None and config.return_path_coeff > 0:
            return_path_replay = ReturnPathReplay(
                data, vocab, capacity=config.return_path_capacity, seed=config.seed,
                excluded_seeds=tuple(eval_panel),
            )
            if resume_state is not None:
                saved_paths = resume_state.get("return_path_replay")
                if saved_paths is None:
                    raise ValueError("Return-path resume requires its saved replay state; use explicit actor transfer")
                return_path_replay.load_state_dict(saved_paths)
            (model.module if isinstance(model, nn.DataParallel) else model)._return_path_replay = return_path_replay
        if config.risk_forecast_log:
            risk_forecast_path.parent.mkdir(parents=True, exist_ok=True)
            # Long-lived append handle; closed in this function's finally block.
            risk_forecast_file = open(risk_forecast_path, "a", buffering=1)  # noqa: SIM115

        initial_stop_pending = bool(config.stop_request_path and Path(config.stop_request_path).is_file())
        if update_count == 0 and config.eval_before_training and not initial_stop_pending:
            baseline_device = torch.device(config.eval_device) if config.eval_device else device
            try:
                evaluate_model(
                    model.to(baseline_device), data, vocab, config.eval_games, baseline_device,
                    max_no_progress_steps=config.max_no_progress_steps, win_ante=config.win_ante,
                    temperature=config.rollout_temperature, stake=config.stake, seeds=config.eval_seeds,
                    eval_batch_size=config.eval_batch_size, reward_config=config.reward_config,
                    results_path=Path(config.log_dir) / "eval_outcomes.jsonl", writer=writer, update_count=0,
                    sampled_games=config.eval_sampled_games, policy_config=config, sampling_seed=config.seed + 31013,
                )
            finally:
                if baseline_device != device:
                    model.to(device)
                if baseline_device.type == "mps":
                    torch.mps.empty_cache()
            writer.flush()
        while update_count < planned_updates:
            if config.reward_config.objective == "milestone":
                milestone_scale = resolve_milestone_scale(config, total_steps, schedule_total_steps)
                vec_env.call("set_milestone_scale", milestone_scale)
                writer.add_scalar("curriculum/milestone_scale", milestone_scale, update_count + 1)
            buffer, rm, collected_steps = collect_rollout(
                model,
                vec_env,
                obs_buf,
                config,
                rollout_state,
                episode_buffer,
                update_count=update_count,
                risk_forecast_file=risk_forecast_file,
                return_path_replay=return_path_replay,
            )
            total_steps += collected_steps
            if return_path_replay is not None:
                return_path_replay.advance(config.return_path_rebuild_steps)
                for name, value in return_path_replay.metrics().items():
                    writer.add_scalar("return_path/" + name, value, update_count + 1)

            write_rollout_metrics(
                writer,
                rm,
                buffer,
                rollout_state.history,
                win_ante=effective_win_ante,
                step=update_count + 1,
            )

            # SIL coefficient uses the same pinned schedule horizon so it decays
            # monotonically to sil_coeff_final without rewinding on resume.
            sil_coeff_now = resolve_sil_coeff(config, total_steps, schedule_total_steps=schedule_total_steps)
            sil_grad_diagnostics_due = (
                sil_coeff_now > 0.0
                and config.sil_grad_diagnostics_interval > 0
                and (update_count + 1) % config.sil_grad_diagnostics_interval == 0
            )

            update_stats = _run_ppo_update(
                model=model,
                optimizer=optimizer,
                buffer=buffer,
                entropy_coeff=entropy_coeff,
                config=config,
                accum_steps=accum_steps,
                effective_batch_size=effective_batch_size,
                device=device,
                use_pin_memory=use_pin_memory,
                sil_buffer=sil_buffer,
                sil_coeff_now=sil_coeff_now,
                return_path_replay=return_path_replay,
                grad_diagnostics_due=sil_grad_diagnostics_due,
            )

            terminal_replay_result = _run_terminal_replay_updates(
                model,
                optimizer,
                episode_buffer,
                config,
                device,
            )

            update_policy_losses = update_stats.policy_losses
            update_entropies = update_stats.entropies
            update_normalized_entropies = update_stats.normalized_entropies
            update_action_type_entropies = update_stats.action_type_entropies
            update_clip_fracs = update_stats.clip_fracs
            update_approx_kls = update_stats.approx_kls
            update_valid_action_counts = update_stats.valid_action_counts
            update_valid_action_type_counts = update_stats.valid_action_type_counts

            # Track the normalized entropy signal every update, even with fixed entropy.
            mean_normalized_entropy = _sample_weighted_mean(
                update_normalized_entropies,
                update_stats.sample_counts,
            )
            entropy_signal_ema = _smoothed_entropy_signal(
                entropy_signal_ema,
                mean_normalized_entropy,
                config.entropy_ema_beta,
            )
            if config.adaptive_entropy and not update_stats.kl_rollback:
                assert log_alpha is not None
                assert alpha_optimizer is not None
                alpha_loss = _entropy_alpha_loss(log_alpha, entropy_signal_ema, config.target_entropy)
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()
                with torch.no_grad():
                    log_alpha.clamp_(np.log(config.alpha_min), np.log(config.alpha_max))
                entropy_coeff = log_alpha.exp().item()
            update_count += 1

            # TensorBoard logging
            weighted_mean = partial(
                _sample_weighted_mean,
                sample_counts=update_stats.sample_counts,
            )

            mean_entropy = weighted_mean(update_entropies)
            mean_policy_loss = weighted_mean(update_policy_losses)
            mean_return_huber = weighted_mean(update_stats.return_hubers)
            mean_outcome_nll = _sample_weighted_mean(
                update_stats.outcome_nlls,
                update_stats.outcome_valid_counts,
            )
            mean_outcome_brier = _sample_weighted_mean(
                update_stats.outcome_briers,
                update_stats.outcome_valid_counts,
            )
            mean_derived_win_brier = _sample_weighted_mean(
                update_stats.derived_win_briers,
                update_stats.outcome_valid_counts,
            )
            mean_action_type_entropy = weighted_mean(update_action_type_entropies)
            mean_clip_fraction = weighted_mean(update_clip_fracs)
            mean_approx_kl = update_stats.running_kl
            mean_valid_action_count = weighted_mean(update_valid_action_counts)
            mean_valid_action_type_count = weighted_mean(update_valid_action_type_counts)
            writer.add_scalar("ppo/policy_loss", mean_policy_loss, update_count)
            writer.add_scalar("ppo/entropy_raw", mean_entropy, update_count)
            writer.add_scalar("critic/outcome_nll", mean_outcome_nll, update_count)
            writer.add_scalar("critic/outcome_brier", mean_outcome_brier, update_count)
            writer.add_scalar("critic/derived_win_brier", mean_derived_win_brier, update_count)
            writer.add_scalar("critic/return_huber", mean_return_huber, update_count)
            writer.add_scalar(
                "critic/terminal_value_mean",
                weighted_mean(update_stats.terminal_value_means),
                update_count,
            )
            writer.add_scalar(
                "critic/return_residual_mean",
                weighted_mean(update_stats.return_residual_means),
                update_count,
            )
            writer.add_scalar(
                "critic/expected_return_mean",
                weighted_mean(update_stats.expected_return_means),
                update_count,
            )
            writer.add_scalar("ppo/entropy_normalized", mean_normalized_entropy, update_count)
            writer.add_scalar(
                "ppo/action_type_entropy_normalized",
                mean_action_type_entropy,
                update_count,
            )
            writer.add_scalar("ppo/clip_fraction", mean_clip_fraction, update_count)
            writer.add_scalar("ppo/approx_kl", mean_approx_kl, update_count)
            writer.add_scalar(
                "ppo/kl_rollback_event",
                1.0 if update_stats.kl_rollback else 0.0,
                update_count,
            )
            writer.add_scalar("ppo/actual_lr", update_stats.actual_lr, update_count)
            writer.add_scalar("ppo/effective_lr", config.lr, update_count)
            writer.add_scalar("ppo/configured_actor_lr", config.lr, update_count)
            for metric_name, metric_value in terminal_replay_result.diagnostics.items():
                if math.isfinite(metric_value):
                    writer.add_scalar(
                        f"terminal_replay/{metric_name}",
                        metric_value,
                        update_count,
                    )
            if sil_buffer is not None:
                writer.add_scalar("sil/buffer_win_fraction", float(sil_buffer.win_fraction), update_count)
                writer.add_scalar("sil/coeff", float(sil_coeff_now), update_count)
                if update_stats.sil_losses_weighted:
                    writer.add_scalar(
                        "sil/loss_weighted",
                        float(np.mean(update_stats.sil_losses_weighted)),
                        update_count,
                    )
                if update_stats.sil_gate_means:
                    writer.add_scalar(
                        "sil/gate_mean",
                        float(np.mean(update_stats.sil_gate_means)),
                        update_count,
                    )
                if update_stats.sil_grad_diagnostic_valid:
                    writer.add_scalar(
                        "sil/ppo_actor_grad_norm_ratio",
                        float(update_stats.sil_grad_ppo_actor_norm_ratio),
                        update_count,
                    )
            minibatches_processed = int(np.sum(update_stats.ppo_minibatches_processed))
            # Target-KL early stopping can halt an update after far fewer minibatches
            # than expected; without this it is invisible. Expected = full passes over
            # the rollout for every PPO epoch.
            n_samples = len(buffer._flat_returns) if len(buffer._flat_returns) > 0 else 0
            minibatches_expected = 0
            minibatch_fraction = 1.0
            if n_samples > 0:
                if accum_steps > 1:
                    minibatches_per_epoch = max(
                        1,
                        _physical_minibatch_count(
                            n_samples,
                            config.mini_batch_size,
                            effective_batch_size,
                        ),
                    )
                else:
                    minibatches_per_epoch = max(1, math.ceil(n_samples / effective_batch_size))
                minibatches_expected = minibatches_per_epoch * config.ppo_epochs
                minibatch_fraction = min(1.0, minibatches_processed / minibatches_expected)
                writer.add_scalar("ppo/minibatch_fraction", minibatch_fraction, update_count)
            # Trust-region extrema feed the console diagnostic warnings below.
            kl_p95 = float(np.percentile(update_approx_kls, 95)) if update_approx_kls else 0.0
            kl_max = float(np.max(update_approx_kls)) if update_approx_kls else 0.0
            clip_frac_max = float(np.max(update_clip_fracs)) if update_clip_fracs else 0.0
            writer.add_scalar(
                "ppo/valid_action_type_count_mean",
                mean_valid_action_type_count,
                update_count,
            )
            writer.add_scalar("ppo/entropy_coeff", entropy_coeff, update_count)
            # Release the rollout buffer before the eval/checkpoint memory peak.
            # The buffer holds the full rollout's observation/action_mask arrays
            # (the largest host allocation in the loop) and is not read again until
            # the next iteration reallocates it. Holding it alive through eval
            # (which spins up `eval_games` fresh envs) and the checkpoint save
            # stacks two large allocations on top of it. On a long MPS run that
            # peak is enough to trip the OS memory-pressure killer, which SIGKILLs
            # the main process (largest footprint) with no Python traceback and
            # leaves every AsyncVectorEnv worker dying on EOFError/BrokenPipe.
            # Free it here and return cached device memory to the OS so the peak
            # does not accumulate update over update.
            del buffer
            if device.type == "mps":
                torch.mps.empty_cache()

            if rollout_state.history.rewards:
                recent = rollout_state.history.rewards[-100:]
                recent_wins = rollout_state.history.wins[-100:]
                recent_stalls = rollout_state.history.stalls[-100:]
                recent_reward_mean = float(np.mean(recent))
                recent_length_mean = float(np.mean(rollout_state.history.lengths[-100:]))
                recent_win_rate = float(np.mean(recent_wins))
                recent_stall_rate = float(np.mean(recent_stalls))
            else:
                recent_reward_mean = float("nan")
                recent_length_mean = float("nan")
                recent_win_rate = float("nan")
                recent_stall_rate = float("nan")

            monitor_stop = bool(config.stop_request_path and Path(config.stop_request_path).is_file())
            should_checkpoint = (
                update_count % config.checkpoint_interval == 0 or update_count == planned_updates or monitor_stop
            )
            for origin, wins in rollout_state.history.origin_wins.items():
                if wins:
                    writer.add_scalar(f"curriculum/{origin}_win_rate", float(np.mean(wins[-100:])), update_count)
                    writer.add_scalar(f"curriculum/{origin}_episodes", len(wins), update_count)
            for ante, outcomes in rollout_state.history.continuation_survival.items():
                if outcomes:
                    writer.add_scalar(f"curriculum/survive_ante{ante}", float(np.mean(outcomes[-100:])), update_count)
            if config.archive_config is not None:
                metrics = vec_env.call("archive_metrics")
                for key in metrics[0]:
                    writer.add_scalar(f"archive/{key}", sum(row[key] for row in metrics), update_count)
            if should_checkpoint:
                # Make sure all preceding scalars
                # land on disk before the checkpoint save, which is itself
                # a long sync that could crash if memory is tight.
                writer.flush()
                checkpoint_path = _save_checkpoint(
                    vector_env=vec_env,
                    model=model,
                    optimizer=optimizer,
                    save_path=save_path,
                    update_count=update_count,
                    total_steps=total_steps,
                    planned_updates=planned_updates,
                    entropy_coeff=entropy_coeff,
                    entropy_signal_ema=entropy_signal_ema,
                    lr=config.lr,
                    log_alpha=log_alpha,
                    alpha_optimizer=alpha_optimizer,
                    agent_config=agent_config,
                    config=config,
                    filename="ppo_monitor_stop.pt" if monitor_stop else None,
                    extra={
                        "best_eval_win_rate": best_eval_win_rate,
                        "best_eval_update": best_eval_update,
                        "schedule_total_steps": schedule_total_steps,
                        "monitor_stop_requested": monitor_stop,
                    },
                )
                # Always mirror the latest checkpoint so resume/eval always has
                # a stable path without guessing the highest update number.
                _mirror_latest_checkpoint(checkpoint_path, save_path)
                logger.info("Saved checkpoint: %s", checkpoint_path)

            if monitor_stop:
                early_stop_requested = True
                logger.warning("Monitor stop request honored at update %d; resume checkpoint saved.", update_count)

            eval_win_rate: float | None = None
            writer.add_scalar("return_path/loss", update_stats.return_path_loss, update_count)
            writer.add_scalar("return_path/samples", update_stats.return_path_samples, update_count)
            writer.add_scalar("return_path/loss_weighted",
                              config.return_path_coeff * update_stats.return_path_loss, update_count)
            should_eval = not monitor_stop and (
                update_count % config.eval_interval == 0 or update_count == planned_updates
            )
            if should_eval:
                eval_summary: dict = {}
                # Flush before eval so any preceding scalars survive a
                # parent-process crash during the (potentially long) eval.
                writer.flush()
                # Optionally run eval on a different device to keep MPS
                # memory pressure down. The model is temporarily moved to
                # the eval device and moved back when eval finishes.
                eval_device = device
                if config.eval_device:
                    eval_device = torch.device(config.eval_device)
                    model_to_eval = model.to(eval_device)
                else:
                    model_to_eval = model
                try:
                    win_rate = evaluate_model(
                        model_to_eval,
                        data,
                        vocab,
                        config.eval_games,
                        eval_device,
                        max_no_progress_steps=config.max_no_progress_steps,
                        win_ante=config.win_ante,
                        temperature=config.rollout_temperature,
                        seeds=config.eval_seeds,
                        eval_batch_size=config.eval_batch_size,
                        stake=config.stake,
                        reward_config=config.reward_config,
                        results_path=Path(config.log_dir) / "eval_outcomes.jsonl",
                        writer=writer,
                        update_count=update_count,
                        sampled_games=config.eval_sampled_games,
                        policy_config=config,
                        sampling_seed=config.seed + 31013,
                        summary_out=eval_summary,
                    )
                finally:
                    if config.eval_device and eval_device != device:
                        # Move model back to the training device for the next
                        # rollout. Failure to do this leaves the optimizer's
                        # param groups pointing at the wrong device.
                        model.to(device)
                    # Force-release any large eval-side allocations before the
                    # next training rollout; this is a no-op on CPU but
                    # materially reduces MPS high-water-mark usage.
                    if eval_device.type == "mps":
                        torch.mps.empty_cache()
                    del model_to_eval
                writer.add_scalar("eval/win_rate", win_rate, update_count)
                writer.flush()
                eval_win_rate = win_rate
                # Track the best eval win rate and snapshot ppo_best_eval.pt so
                # checkpoint selection does not rely on reading every event file.
                rank = evaluation_rank(win_rate, eval_summary)
                base_model = model.module if isinstance(model, nn.DataParallel) else model
                previous_rank = getattr(base_model, "_best_eval_rank", None)
                if previous_rank is None and resume_state is not None and not config.reset_best_eval:
                    previous_rank = resume_state.get("best_eval_rank")
                if previous_rank is None and best_eval_win_rate is not None:
                    previous_rank = (best_eval_win_rate, -1.0, -1.0, -1.0)
                is_best = previous_rank is None or rank > tuple(previous_rank)
                if is_best:
                    base_model._best_eval_rank = rank
                    best_eval_win_rate = win_rate
                    best_eval_update = update_count
                    best_path = _save_checkpoint(
                        vector_env=vec_env,
                        model=model,
                        optimizer=optimizer,
                        save_path=save_path,
                        update_count=update_count,
                        total_steps=total_steps,
                        planned_updates=planned_updates,
                        entropy_coeff=entropy_coeff,
                        entropy_signal_ema=entropy_signal_ema,
                        lr=config.lr,
                        log_alpha=log_alpha,
                        alpha_optimizer=alpha_optimizer,
                        agent_config=agent_config,
                        config=config,
                        filename="ppo_best_eval.pt",
                        extra={
                            "best_eval_win_rate": best_eval_win_rate,
                            "best_eval_update": best_eval_update,
                            "schedule_total_steps": schedule_total_steps,
                        },
                    )
                    logger.info(
                        "New best eval (win rate primary, fresh progression tie-break) "
                        "win_rate=%.3f at update %d -> %s",
                        win_rate,
                        update_count,
                        best_path,
                    )
                eval_regression_streak = _next_eval_regression_streak(
                    win_rate=win_rate,
                    best_win_rate=best_eval_win_rate,
                    current_streak=eval_regression_streak,
                    tolerance=config.eval_regression_tolerance,
                )
                eval_best = best_eval_win_rate if best_eval_win_rate is not None else win_rate
                regression_from_best = max(float(eval_best) - win_rate, 0.0)
                writer.add_scalar("eval/regression_from_best", regression_from_best, update_count)
                writer.add_scalar("eval/regression_streak", float(eval_regression_streak), update_count)
                regression_early_stop = (
                    config.eval_regression_tolerance is not None
                    and eval_regression_streak >= config.eval_regression_patience
                )
                writer.add_scalar("eval/regression_early_stop", float(regression_early_stop), update_count)
                if regression_early_stop:
                    early_stop_requested = True
                    writer.flush()
                    regression_stop_path = _save_checkpoint(
                        vector_env=vec_env,
                        model=model,
                        optimizer=optimizer,
                        save_path=save_path,
                        update_count=update_count,
                        total_steps=total_steps,
                        planned_updates=planned_updates,
                        entropy_coeff=entropy_coeff,
                        entropy_signal_ema=entropy_signal_ema,
                        lr=config.lr,
                        log_alpha=log_alpha,
                        alpha_optimizer=alpha_optimizer,
                        agent_config=agent_config,
                        config=config,
                        filename="ppo_regression_stop.pt",
                        extra={
                            "best_eval_win_rate": best_eval_win_rate,
                            "best_eval_update": best_eval_update,
                            "schedule_total_steps": schedule_total_steps,
                            "eval_regression_streak": eval_regression_streak,
                        },
                    )
                    _mirror_latest_checkpoint(regression_stop_path, save_path)
                    logger.error(
                        "Stopping PPO after %d consecutive eval regressions: "
                        "win_rate=%.3f, best=%.3f at update %s, tolerance=%.3f. "
                        "Best checkpoint remains ppo_best_eval.pt; exact resume checkpoint=%s.",
                        eval_regression_streak,
                        win_rate,
                        best_eval_win_rate,
                        best_eval_update,
                        config.eval_regression_tolerance,
                        regression_stop_path,
                    )
                # eval runs `eval_games` full games in-process; release the
                # forward-pass allocations it cached before the next rollout.
                if device.type == "mps":
                    torch.mps.empty_cache()

            should_log_progress = (
                update_count % config.log_interval == 0 or should_eval or update_count == planned_updates
            )
            if should_log_progress:
                progress = (
                    f"Update {update_count}/{planned_updates}, steps {total_steps}/{config.total_timesteps}: "
                    f"policy_loss={mean_policy_loss:.4f}, "
                    f"return_huber={mean_return_huber:.4f}, "
                    f"entropy={mean_entropy:.4f}, "
                    f"entropy_signal={entropy_signal_ema:.4f}, "
                    f"entropy_coeff={entropy_coeff:.5f}, "
                    f"clip_fraction={mean_clip_fraction:.4f}, "
                    f"approx_kl={mean_approx_kl:.5f}, "
                    f"valid_actions={mean_valid_action_count:.1f}, "
                    f"ep_reward_mean={recent_reward_mean:.3f}, "
                    f"ep_length_mean={recent_length_mean:.1f}, "
                    f"rollout_win_rate={recent_win_rate:.3f}, "
                    f"rollout_stall_rate={recent_stall_rate:.3f}"
                )
                if eval_win_rate is not None:
                    progress += f", eval_win_rate={eval_win_rate:.3f}"
                logger.info(progress)

            # === Diagnostic warnings ===
            # Surface unhealthy PPO dynamics on the console so they are not
            # buried in TensorBoard. Each warning is rate-limited by
            # log_interval (the same gate as the progress line above) so the
            # log stays readable when a condition is persistently true.
            if update_stats.ppo_minibatches_processed:
                minibatches_done = int(np.sum(update_stats.ppo_minibatches_processed))
                # minibatches_expected and minibatch_fraction were captured
                # above, before `del buffer`. The buffer is released before
                # eval/checkpoint to bound MPS memory peak.
                if minibatches_expected > 0 and minibatch_fraction < 0.5:
                    logger.warning(
                        "PPO update %d only processed %d/%d expected minibatches "
                        "(%.0f%%); target_kl is stopping updates early. Consider "
                        "raising --target-kl or lowering --ppo-epochs.",
                        update_count,
                        minibatches_done,
                        minibatches_expected,
                        minibatch_fraction * 100.0,
                    )

                # Phase 4: alert if minibatch_fraction < 0.8 for 10 consecutive
                # updates, chronic KL-stop means the step size (lr) is wrong,
                # not the trust region. Prefer fixing lr over reverting target_kl.
                if minibatches_expected > 0 and minibatch_fraction < 0.8:
                    _low_minibatch_streak += 1
                else:
                    _low_minibatch_streak = 0
                if _low_minibatch_streak >= 10:
                    logger.warning(
                        "ppo/minibatch_fraction < 0.8 for %d consecutive updates "
                        "(current=%.2f at update %d). Chronic KL-stop means the step "
                        "size is wrong: reduce --lr (currently %.2e) before touching "
                        "--target-kl.",
                        _low_minibatch_streak,
                        minibatch_fraction,
                        update_count,
                        config.lr,
                    )

            # === Trust-region diagnostic warnings (Phase 3) ===
            # Surface dangerous KL spikes and premature early-stopping. The
            # hard thresholds can be tightened via the CLI guards below.
            if should_log_progress:
                if kl_max > 0.25:
                    logger.warning(
                        "ppo/stop_reason_kl_max=%.4f (>0.25) at update %d; a minibatch "
                        "moved the policy far outside the trust region. Lower "
                        "--target-kl / --lr or --ppo-epochs.",
                        kl_max,
                        update_count,
                    )
                if kl_p95 > 0.10:
                    logger.warning(
                        "ppo/stop_reason_kl_p95=%.4f (>0.10) at update %d; the 95th "
                        "percentile minibatch KL is unsafe. Consider a tighter --target-kl.",
                        kl_p95,
                        update_count,
                    )
                if minibatches_expected > 0 and minibatch_fraction < 0.25:
                    logger.warning(
                        "ppo/minibatch_fraction=%.2f (<0.25) at update %d; target_kl "
                        "is halting updates almost immediately.",
                        minibatch_fraction,
                        update_count,
                    )
                if clip_frac_max > 0.30:
                    logger.warning(
                        "ppo/clip_fraction_max=%.3f (>0.30) at update %d; many "
                        "minibatches are hitting the PPO clip, a sign of large "
                        "policy moves.",
                        clip_frac_max,
                        update_count,
                    )
                # Honor explicit CLI guards if provided (tighter than the defaults).
                if config.target_kl_max is not None and kl_max > config.target_kl_max:
                    logger.warning(
                        "--target-kl-max=%.3f exceeded: ppo/stop_reason_kl_max=%.4f at update %d.",
                        config.target_kl_max,
                        kl_max,
                        update_count,
                    )
                if config.target_kl_p95 is not None and kl_p95 > config.target_kl_p95:
                    logger.warning(
                        "--target-kl-p95=%.3f exceeded: ppo/stop_reason_kl_p95=%.4f at update %d.",
                        config.target_kl_p95,
                        kl_p95,
                        update_count,
                    )
                if (
                    config.min_minibatch_fraction is not None
                    and minibatches_expected > 0
                    and minibatch_fraction < config.min_minibatch_fraction
                ):
                    logger.warning(
                        "--min-minibatch-fraction=%.2f not met: ppo/minibatch_fraction=%.2f at update %d.",
                        config.min_minibatch_fraction,
                        minibatch_fraction,
                        update_count,
                    )

            # === Exploration-pressure warnings (Phase 4) ===
            mean_normalized_entropy_update = mean_normalized_entropy
            mean_chosen_action_prob = _safe_mean(rm.chosen_action_probs)
            mean_action_type_entropy_update = mean_action_type_entropy
            if should_log_progress:
                if mean_normalized_entropy_update < 0.10:
                    logger.warning(
                        "ppo/entropy_normalized=%.3f (<0.10) at update %d; the policy "
                        "is near-deterministic. Raise --entropy-coeff / --target-entropy "
                        "or enable --adaptive-entropy.",
                        mean_normalized_entropy_update,
                        update_count,
                    )
                if rm.chosen_action_probs and mean_chosen_action_prob > 0.80:
                    logger.warning(
                        "debug/chosen_action_prob_mean=%.3f (>0.80) at update %d; the "
                        "sampled action is almost always the greedy one.",
                        mean_chosen_action_prob,
                        update_count,
                    )
                if mean_action_type_entropy_update < 0.20:
                    logger.warning(
                        "ppo/action_type_entropy_normalized=%.3f (<0.20) at update %d; "
                        "the policy has collapsed onto a single action family.",
                        mean_action_type_entropy_update,
                        update_count,
                    )

            if early_stop_requested:
                break

    finally:
        if return_path_replay is not None:
            return_path_replay.close()
        # Always close the vector env and TensorBoard writer, even on
        # KeyboardInterrupt, OOM, or any unhandled exception in the training
        # loop. Without this, an early parent-process exit leaves every
        # AsyncVectorEnv worker alive long enough to flood the log with
        # EOFError/BrokenPipe traceback cascades that mask the real cause.
        #
        # If you are debugging a crash where the visible trace is just
        # AsyncVectorEnv EOFError/BrokenPipe, the real exception is usually
        # above it in the log (or in this finally block). Re-running with
        # --sync-envs surfaces env-worker errors directly in the parent.
        try:
            vec_env.close()
        except Exception:
            logger.exception("Failed to close vector env cleanly")
        if writer is not None:
            try:
                writer.flush()
                writer.close()
            except Exception:
                logger.exception("Failed to close TensorBoard writer cleanly")
        if risk_forecast_file is not None:
            try:
                risk_forecast_file.close()
            except Exception:
                logger.exception("Failed to close risk forecast log cleanly")
    logger.info(
        "PPO training complete after %d updates and %d env steps.",
        update_count,
        total_steps,
    )
    return model
