"""PPO training loop with vectorized environments."""

import copy
import json
import logging
import math
import random
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.vector.vector_env import AutoresetMode
from torch.optim import Adam

from pylatro import GameData, load_game_data

from ..action import ActionType, decode_action
from ..action_grammar import ActionGrammarDistribution, ActionGrammarOutput
from ..agent import AgentConfig, BalatroAgent
from ..archive import ArchiveConfig
from ..constants import (
    HISTORY_EVENT_DIM,
    HISTORY_FEATURE_DIM,
    HISTORY_MAX_CARDS,
    HISTORY_MAX_JOKERS,
    HISTORY_MAX_PLAYS,
    HISTORY_OMITTED_DIM,
    HISTORY_ROUNDS,
    MAX_SEQ_LEN,
    META_START,
    NUM_ACTIONS,
    POKER_HAND_NAMES,
    SCALAR_DIM,
    TOKEN_DIM,
)
from ..diagnostics import (
    MAX_DIAGNOSTIC_EVENTS,
    MAX_DIAGNOSTIC_JOKERS,
    MAX_DIAGNOSTIC_SHOP_JOKERS,
)
from ..env import BalatroEnv
from ..reward import DEFAULT_REWARD_CONFIG, REWARD_INFO_KEYS, RewardConfig
from ..risk import uncalibrate_analytic_death_probability
from ..survival import validate_critic_win_ante
from ..value_head import outcome_nll, return_huber_loss
from ..vocab import Vocab, build_vocab
from .rollout_buffer import RolloutBuffer
from .sil import (
    EpisodeReplayBuffer,
    EpisodeTracker,
    sil_percentile_gate,
)

logger = logging.getLogger(__name__)
_MISSING = object()
_HISTORY_SIGNATURE_CACHE: dict[type, bool] = {}
_BLIND_INDEX = {"small": 0, "big": 1, "boss": 2}


def _next_blind_clear_outcome(
    *,
    shop_ante: int,
    shop_blind_index: int,
    final_ante: int,
    won: bool,
    terminal_blind: str,
) -> float:
    """Return whether the blind forecast at shop leave was subsequently cleared."""

    if won or final_ante > shop_ante:
        return 1.0
    if final_ante < shop_ante:
        return 0.0
    terminal_index = _BLIND_INDEX.get(str(terminal_blind).lower(), shop_blind_index)
    return float(terminal_index > shop_blind_index)


def _binary_roc_auc(predictions: list[float], outcomes: list[float]) -> float | None:
    """Return tie-aware ROC AUC, or None when only one outcome class exists."""

    prediction_array = np.asarray(predictions, dtype=np.float64)
    outcome_array = np.asarray(outcomes, dtype=np.float64) > 0.5
    positive_count = int(outcome_array.sum())
    negative_count = int(outcome_array.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None

    order = np.argsort(prediction_array, kind="mergesort")
    sorted_predictions = prediction_array[order]
    ranks = np.empty(prediction_array.size, dtype=np.float64)
    start = 0
    while start < prediction_array.size:
        end = start + 1
        while end < prediction_array.size and sorted_predictions[end] == sorted_predictions[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(ranks[outcome_array].sum())
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)


_ACTION_TYPES = tuple(ActionType)
_ACTION_TYPE_TO_INDEX = {action_type: idx for idx, action_type in enumerate(_ACTION_TYPES)}
_ACTION_ID_TO_TYPE_INDEX = torch.tensor(
    [_ACTION_TYPE_TO_INDEX[decode_action(action_id).action_type] for action_id in range(NUM_ACTIONS)],
    dtype=torch.long,
)


def _load_state_dict_into_model(model: nn.Module, state_dict: dict, checkpoint_path: str) -> None:
    """Strictly load weights, allowing only a DataParallel prefix change."""
    has_module_prefix = any(k.startswith("module.") for k in state_dict)
    is_wrapped = isinstance(model, nn.DataParallel)

    if has_module_prefix and not is_wrapped:
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    elif not has_module_prefix and is_wrapped:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is architecture-incompatible and does not "
            "exactly match the configured model. Use the source model dimensions "
            "or a matching checkpoint."
        ) from exc


def _load_checkpoint_strict(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    *,
    active_reward_config: RewardConfig | None = None,
) -> None:
    """Load current-schema weights, optionally requiring matching rewards."""
    from ..checkpoint import load_checkpoint_payload
    from ..reward import reward_config_fingerprint

    payload = load_checkpoint_payload(checkpoint_path, device)
    if active_reward_config is not None:
        saved_fingerprint = payload.get("reward_fingerprint")
        active_fingerprint = reward_config_fingerprint(active_reward_config)
        if saved_fingerprint and saved_fingerprint != active_fingerprint:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} was trained with a different reward "
                "fingerprint. Use matching reward settings, or --actor-transfer "
                "to start a new Ante-8 PPO run with a fresh critic and optimizer."
            )
        if not saved_fingerprint:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} has no reward fingerprint, so loading its "
                "critic cannot be verified. Use a matching checkpoint or "
                "--actor-transfer to start a new Ante-8 PPO run with a fresh critic."
            )
    _load_state_dict_into_model(model, payload["state_dict"], checkpoint_path)


def load_actor_transfer(model: nn.Module, path: str, device: torch.device) -> None:
    """Explicit weights-only migration; never restore old reward/optimizer state.

    v12 appends Joker features, retaining the first 12 token fields. Its new
    zero adapter makes compatible v11 actor transfer behavior-preserving.
    Older observation/action schemas require a separate migration.
    """
    import hashlib

    from ..schema import ACTOR_TRANSFER_SCHEMAS

    payload = torch.load(path, map_location=device, weights_only=False)
    version = payload.get("tokenizer_version")
    if version not in ACTOR_TRANSFER_SCHEMAS or payload.get("tokenizer_semantics") != ACTOR_TRANSFER_SCHEMAS[version]:
        raise ValueError("Actor transfer supports tokenizer v11 or the current schema; older schemas need migration")
    saved = {key.removeprefix("module."): value for key, value in payload["state_dict"].items()}
    base_model = _unwrap_model(model)
    current = base_model.state_dict()
    # Discover the module's registered path rather than assuming a naming alias.
    adapters = {key for key in current if key.endswith("state_proj.weight") and "joker" in key}
    for key, value in current.items():
        if key.startswith("value_head."):
            continue
        if key not in saved:
            if version == 11 and key in adapters:
                current[key] = torch.zeros_like(value)
                continue
            raise ValueError(f"Actor transfer missing {key}; architecture must match the source")
        if value.shape != saved[key].shape:
            raise ValueError(f"Actor transfer shape mismatch for {key}; use source model dimensions")
        current[key] = saved[key]
    extra_keys = set(saved) - set(current)
    if any(not key.startswith("value_head.") for key in extra_keys):
        raise ValueError(f"Actor transfer contains incompatible actor parameters: {sorted(extra_keys)}")
    saved_goal = int((payload.get("ppo_config_fields") or {}).get("win_ante")
                     or (payload.get("reward_config") or {}).get("potential_win_ante") or 8)
    if not 1 <= saved_goal <= 8:
        raise ValueError("Actor transfer requires a source target Ante between 1 and 8")
    if saved_goal != 8:
        for key, value in current.items():
            if key.endswith("win_ante_emb.weight"):
                value = value.clone()
                value[8] = value[saved_goal]
                current[key] = value
    base_model.load_state_dict(current, strict=True)
    with open(path, "rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    base_model._actor_transfer_metadata = {
        "path": str(path), "sha256": digest, "tokenizer_version": version,
        "source_win_ante": saved_goal, "critic_reset": True,
    }
    logger.info("Transferred actor from %s (tokenizer %s); critic and optimizer start fresh", path, version)


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
    config: "PPOConfig",
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


def _make_env(
    seed: int,
    stake: int,
    data: GameData,
    vocab: Vocab,
    max_no_progress_steps: int,
    win_ante: int | None,
    reward_config: RewardConfig | None = None,
    counterfactual_diagnostic_interval: int = 0,
    archive_config: ArchiveConfig | None = None,
    archive_index: int = 0,
):
    """Factory for creating a BalatroEnv (used by vectorized env wrappers)."""

    def _thunk():
        return BalatroEnv(
            seed=seed,
            stake=stake,
            data=data,
            vocab=vocab,
            max_steps=max_no_progress_steps,
            win_ante=win_ante,
            reward_config=reward_config,
            enable_teacher=False,
            counterfactual_diagnostic_interval=counterfactual_diagnostic_interval,
            archive_config=archive_config,
            archive_index=archive_index,
        )

    return _thunk


def _make_vectorized_envs(
    num_envs: int,
    data: GameData,
    vocab: Vocab,
    stake: int = 1,
    max_no_progress_steps: int = 256,
    use_async: bool = True,
    win_ante: int | None = None,
    reward_config: RewardConfig | None = None,
    env_seed_base: int = 0,
    counterfactual_diagnostic_interval: int = 0,
    archive_config: ArchiveConfig | None = None,
):
    """Create a gymnasium VectorEnv (async for multiprocess, sync for single-process)."""
    import gymnasium

    env_fns = [
        _make_env(
            env_seed_base + i,
            stake,
            data,
            vocab,
            max_no_progress_steps,
            win_ante,
            reward_config,
            counterfactual_diagnostic_interval=counterfactual_diagnostic_interval,
            archive_config=archive_config,
            archive_index=i,
        )
        for i in range(num_envs)
    ]

    if use_async and num_envs > 1:
        return gymnasium.vector.AsyncVectorEnv(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    else:
        return gymnasium.vector.SyncVectorEnv(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)


@dataclass
class PPOConfig:
    archive_config: ArchiveConfig | None = None
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
    save_dir: str = "checkpoints/ppo"
    log_dir: str = "runs/ppo"
    eval_interval: int = 5
    log_interval: int = 10
    checkpoint_interval: int = 50
    # Wins are rare even for the heuristic (~1% at ante 8, ~12% at ante 5,
    # ~39% at ante 4). 10 eval games can't measure rare-event winrate; bump
    # the default so eval/win_rate has signal to compare against.
    eval_games: int = 50
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
    # Fraction of completed episodes withheld from every training sampler and
    # scored no-grad instead. This is the only critic number in the run that
    # measures generalization rather than fit. 0 disables the split.
    #
    # The split is enforced buffer-wide, so SIL gives up this fraction of its
    # episodes too. That is deliberate: SIL updates the actor and shared trunk,
    # so an episode SIL trained on is no longer held out from the critic that
    # reads that trunk.
    terminal_replay_holdout_fraction: float = 0.1
    terminal_replay_holdout_batch_size: int = 512
    # Optional reward settings threaded through BalatroEnv. None uses the
    # default potential-based reward configuration.
    reward_config: "RewardConfig | None" = None
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


@dataclass
class _RolloutMetrics:
    """Per-rollout accumulators populated during the step loop."""

    action_type_counts: Counter = field(default_factory=Counter)
    hand_chosen_counts: Counter = field(default_factory=Counter)
    hand_best_counts: Counter = field(default_factory=Counter)
    planet_use_key_counts: Counter = field(default_factory=Counter)
    planet_claim_key_counts: Counter = field(default_factory=Counter)
    consumable_use_set_counts: Counter = field(default_factory=Counter)
    consumable_claim_set_counts: Counter = field(default_factory=Counter)
    consumable_buy_set_counts: Counter = field(default_factory=Counter)
    consumable_offered_counts: Counter = field(default_factory=Counter)
    consumable_claimable_counts: Counter = field(default_factory=Counter)
    consumable_eligible_offer_opportunities: Counter = field(default_factory=Counter)
    consumable_inventory_full_blocked_counts: Counter = field(default_factory=Counter)
    consumable_acquired_counts: Counter = field(default_factory=Counter)
    consumable_exact_use_counts: Counter = field(default_factory=Counter)
    consumable_pack_auto_use_counts: Counter = field(default_factory=Counter)
    consumable_sold_counts: Counter = field(default_factory=Counter)
    consumable_overwritten_counts: Counter = field(default_factory=Counter)
    consumable_expired_counts: Counter = field(default_factory=Counter)
    consumable_owned_states: Counter = field(default_factory=Counter)
    consumable_legal_use_opportunities: Counter = field(default_factory=Counter)
    pack_claim_seal_counts: Counter = field(default_factory=Counter)
    pack_offered_seal_counts: Counter = field(default_factory=Counter)
    pack_skip_state_counts: Counter = field(default_factory=Counter)
    step_rewards: list[float] = field(default_factory=list)
    progress_flags: list[float] = field(default_factory=list)
    steps_since_progress: list[float] = field(default_factory=list)
    chosen_action_probs: list[float] = field(default_factory=list)
    max_action_type_probs: list[float] = field(default_factory=list)
    done_flags: list[float] = field(default_factory=list)
    terminated_flags: list[float] = field(default_factory=list)
    truncated_flags: list[float] = field(default_factory=list)
    completed_episode_rewards: list[float] = field(default_factory=list)
    completed_episode_lengths: list[int] = field(default_factory=list)
    completed_episode_wins: list[float] = field(default_factory=list)
    completed_episode_stalls: list[float] = field(default_factory=list)
    completed_episode_antes: list[int] = field(default_factory=list)
    completed_episode_tarot_uses: list[int] = field(default_factory=list)
    terminal_loss_antes: list[int] = field(default_factory=list)
    terminal_loss_score_ratios: list[float] = field(default_factory=list)
    terminal_loss_dollars: list[float] = field(default_factory=list)
    terminal_loss_cash_ge_10: list[float] = field(default_factory=list)
    terminal_loss_joker_full_weak: list[float] = field(default_factory=list)
    terminal_loss_blind_counts: Counter = field(default_factory=Counter)
    ante1_death_blind_counts: Counter = field(default_factory=Counter)
    terminal_boss_loss_counts: Counter = field(default_factory=Counter)
    terminal_loss_last_play_top1: list[float] = field(default_factory=list)
    terminal_loss_last_play_value_ratios: list[float] = field(default_factory=list)
    ante1_blind_clear_counts: Counter = field(default_factory=Counter)
    ante1_clear_hands_used: list[float] = field(default_factory=list)
    ante1_clear_hands_unused: list[float] = field(default_factory=list)
    ante1_clear_discards_used: list[float] = field(default_factory=list)
    ante1_play_count: int = 0
    ante1_play_hand_counts: Counter = field(default_factory=Counter)
    ante1_play_realized_to_remaining_target: list[float] = field(default_factory=list)
    ante1_conservative_chosen_best_ratios: list[float] = field(default_factory=list)
    ante1_one_hand_clear_proxy_observed: int = 0
    ante1_one_hand_clear_proxy_available: int = 0
    ante1_one_hand_clear_proxy_chosen: int = 0
    ante1_one_hand_clear_proxy_missed: int = 0
    reward_component_values: defaultdict = field(default_factory=lambda: defaultdict(list))
    pre_choose_action_flags: list[float] = field(default_factory=list)
    hand_play_observed: list[float] = field(default_factory=list)
    hand_play_in_candidates: list[float] = field(default_factory=list)
    hand_play_top1: list[float] = field(default_factory=list)
    hand_play_top3: list[float] = field(default_factory=list)
    hand_play_value_ratios: list[float] = field(default_factory=list)
    hand_play_not_in_candidates: list[float] = field(default_factory=list)
    planet_use_observed: list[float] = field(default_factory=list)
    planet_use_played_hand: list[float] = field(default_factory=list)
    planet_use_play_share: list[float] = field(default_factory=list)
    planet_use_main_hand_match: list[float] = field(default_factory=list)
    planet_claim_observed: list[float] = field(default_factory=list)
    planet_claim_played_hand: list[float] = field(default_factory=list)
    planet_claim_play_share: list[float] = field(default_factory=list)
    planet_claim_main_hand_match: list[float] = field(default_factory=list)
    planet_pack_skip: list[float] = field(default_factory=list)
    planet_use_alignment_counts: Counter = field(default_factory=Counter)
    planet_use_hand_counts: Counter = field(default_factory=Counter)
    planet_active_plan_owned: int = 0
    planet_active_plan_legal: int = 0
    planet_active_plan_uses: int = 0
    tarot_use_family_counts: Counter = field(default_factory=Counter)
    tarot_fix_reliability_deltas: list[float] = field(default_factory=list)
    attributable_cash_payouts: list[float] = field(default_factory=list)
    gold_cards_created: int = 0
    held_gold_payout_dollars: int = 0
    shop_offered_joker_counts: Counter = field(default_factory=Counter)
    shop_bought_joker_counts: Counter = field(default_factory=Counter)
    shop_sold_joker_counts: Counter = field(default_factory=Counter)
    clear_probabilities: list[float] = field(default_factory=list)
    immediate_death_probabilities: list[float] = field(default_factory=list)
    shop_leave_flags: list[float] = field(default_factory=list)
    shop_unsafe_leave_flags: list[float] = field(default_factory=list)
    shop_unsafe_can_reroll_flags: list[float] = field(default_factory=list)
    shop_missed_upgrade_flags: list[float] = field(default_factory=list)
    shop_leave_full_weak_flags: list[float] = field(default_factory=list)
    shop_best_upgrade_deltas: list[float] = field(default_factory=list)
    shop_survival_predictions: list[float] = field(default_factory=list)
    shop_survival_outcomes: list[float] = field(default_factory=list)
    shop_survival_briers: list[float] = field(default_factory=list)
    risk_shop_death_predictions: list[float] = field(default_factory=list)
    risk_shop_death_outcomes: list[float] = field(default_factory=list)
    risk_shop_death_briers: list[float] = field(default_factory=list)
    risk_shop_raw_death_predictions: list[float] = field(default_factory=list)
    risk_shop_raw_death_briers: list[float] = field(default_factory=list)
    risk_ante1_false_safe_deaths: list[float] = field(default_factory=list)
    joker_marginal_ratios: defaultdict = field(default_factory=lambda: defaultdict(list))
    joker_modeled_fractions: defaultdict = field(default_factory=lambda: defaultdict(list))
    build_values: defaultdict = field(default_factory=lambda: defaultdict(list))
    potential_values: defaultdict = field(default_factory=lambda: defaultdict(list))
    hand_plan_type_counts: Counter = field(default_factory=Counter)
    hand_plan_reliability: list[float] = field(default_factory=list)
    hand_plan_readiness: list[float] = field(default_factory=list)
    purple_seal_tarots_generated: int = 0
    blue_seal_planets_generated: int = 0
    purple_seals_activated: int = 0
    blue_seals_activated: int = 0
    hologram_scaling_counts: list[float] = field(default_factory=list)
    hologram_x_mult_deltas: list[float] = field(default_factory=list)
    hologram_build_score_deltas: list[float] = field(default_factory=list)
    joker_acquired_count: int = 0
    joker_removed_count: int = 0
    joker_turnover_count: int = 0
    joker_churn_count: int = 0
    joker_replacement_events: int = 0
    joker_acquired_id_counts: Counter = field(default_factory=Counter)
    joker_removed_id_counts: Counter = field(default_factory=Counter)
    counterfactual_calls: int = 0
    counterfactual_failures: int = 0
    counterfactual_representative_realized_abs_gaps: list[float] = field(default_factory=list)
    counterfactual_representative_realized_signed_gaps: list[float] = field(default_factory=list)
    counterfactual_focal_counts: Counter = field(default_factory=Counter)
    play_subset_count: int = 0
    discard_subset_count: int = 0
    joker_order_money_flags: list[float] = field(default_factory=list)
    joker_order_dollars_gained: list[float] = field(default_factory=list)
    joker_order_chips_forgone: list[float] = field(default_factory=list)
    joker_order_clear_flags: list[float] = field(default_factory=list)


def _write_rollout_episode_metrics(writer, rm: _RolloutMetrics, step: int) -> None:
    """Log episode outcomes completed in this rollout, never lifetime means."""
    if not rm.completed_episode_rewards:
        return
    metrics = {
        "episode_reward_mean": rm.completed_episode_rewards,
        "episode_length_mean": rm.completed_episode_lengths,
        "win_rate": rm.completed_episode_wins,
        "stall_rate": rm.completed_episode_stalls,
        "final_ante_mean": rm.completed_episode_antes,
        # Retain the clearer alias introduced for ante diagnostics.
        "mean_ante_reached": rm.completed_episode_antes,
    }
    if rm.completed_episode_tarot_uses:
        metrics["tarot_uses_per_completed_episode_mean"] = rm.completed_episode_tarot_uses
    for tag, values in metrics.items():
        writer.add_scalar(f"rollout/{tag}", float(np.mean(values)), step)


def _write_consumable_strategy_metrics(
    writer,
    rm: _RolloutMetrics,
    step: int,
) -> None:
    """Write bounded Planet/Tarot funnel metrics with explicit denominators."""

    steps_per_thousand = max(len(rm.step_rewards) / 1000.0, 1e-9)
    for consumable_set in ("Planet", "Tarot"):
        tag = consumable_set.lower()
        selected = (
            rm.consumable_buy_set_counts[consumable_set]
            + rm.consumable_claim_set_counts[consumable_set]
        )
        inventory_uses = max(
            rm.consumable_exact_use_counts[consumable_set]
            - rm.consumable_pack_auto_use_counts[consumable_set],
            0,
        )
        counts = {
            "offered": rm.consumable_offered_counts[consumable_set],
            "claimable": rm.consumable_claimable_counts[consumable_set],
            "acquired": rm.consumable_acquired_counts[consumable_set],
            "used": rm.consumable_exact_use_counts[consumable_set],
            "pack_auto_used": rm.consumable_pack_auto_use_counts[consumable_set],
            "sold": rm.consumable_sold_counts[consumable_set],
            "overwritten": rm.consumable_overwritten_counts[consumable_set],
            "expired": rm.consumable_expired_counts[consumable_set],
            "inventory_full_blocked": rm.consumable_inventory_full_blocked_counts[
                consumable_set
            ],
        }
        for name, count in counts.items():
            writer.add_scalar(
                f"strategy/consumables/{tag}/{name}_per_1k_steps",
                count / steps_per_thousand,
                step,
            )
        eligible = rm.consumable_eligible_offer_opportunities[consumable_set]
        if eligible:
            writer.add_scalar(
                f"strategy/consumables/{tag}/claim_rate_given_eligible_offer",
                selected / eligible,
                step,
            )
        legal = rm.consumable_legal_use_opportunities[consumable_set]
        if legal:
            writer.add_scalar(
                f"strategy/consumables/{tag}/use_rate_given_owned_legal",
                inventory_uses / legal,
                step,
            )
        offered = rm.consumable_offered_counts[consumable_set]
        if offered:
            writer.add_scalar(
                f"strategy/consumables/{tag}/inventory_full_blocked_offer_rate",
                rm.consumable_inventory_full_blocked_counts[consumable_set] / offered,
                step,
            )
        writer.add_scalar(
            f"strategy/consumables/{tag}/owned_state_fraction",
            rm.consumable_owned_states[consumable_set] / max(len(rm.step_rewards), 1),
            step,
        )

    planet_uses = sum(rm.planet_use_alignment_counts.values())
    if planet_uses:
        writer.add_scalar(
            "strategy/planets/matched_use_rate",
            rm.planet_use_alignment_counts["matched"] / planet_uses,
            step,
        )
        writer.add_scalar(
            "strategy/planets/unmatched_use_rate",
            rm.planet_use_alignment_counts["unmatched"] / planet_uses,
            step,
        )
    for hand_name in POKER_HAND_NAMES:
        count = rm.planet_use_hand_counts[hand_name]
        if count:
            hand_tag = hand_name.lower().replace(" ", "_")
            writer.add_scalar(
                f"strategy/planets/uses_by_hand/{hand_tag}_per_1k_steps",
                count / steps_per_thousand,
                step,
            )
    writer.add_scalar(
        "strategy/planets/active_plan_owned_states_per_1k_steps",
        rm.planet_active_plan_owned / steps_per_thousand,
        step,
    )
    if rm.planet_active_plan_legal:
        writer.add_scalar(
            "strategy/planets/active_plan_use_rate_given_legal",
            rm.planet_active_plan_uses / rm.planet_active_plan_legal,
            step,
        )

    tarot_uses = sum(rm.tarot_use_family_counts.values())
    for family in (
        "cash",
        "gold",
        "deck_cut",
        "rank_fix",
        "suit_fix",
        "creation",
        "joker",
        "enhancement",
    ):
        if tarot_uses:
            writer.add_scalar(
                f"strategy/tarots/use_family/{family}_share",
                rm.tarot_use_family_counts[family] / tarot_uses,
                step,
            )
    if rm.tarot_fix_reliability_deltas:
        writer.add_scalar(
            "strategy/tarots/fix_reliability_delta_mean",
            float(np.mean(rm.tarot_fix_reliability_deltas)),
            step,
        )
    if rm.attributable_cash_payouts:
        writer.add_scalar(
            "strategy/tarots/attributable_cash_per_use_mean",
            float(np.mean(rm.attributable_cash_payouts)),
            step,
        )
    writer.add_scalar(
        "strategy/gold/cards_created_per_1k_steps",
        rm.gold_cards_created / steps_per_thousand,
        step,
    )
    writer.add_scalar(
        "strategy/gold/payout_dollars_per_1k_steps",
        rm.held_gold_payout_dollars / steps_per_thousand,
        step,
    )
    for seal in ("Blue", "Purple"):
        writer.add_scalar(
            f"strategy/seals/offered/{seal.lower()}_per_1k_steps",
            rm.pack_offered_seal_counts[seal] / steps_per_thousand,
            step,
        )
        writer.add_scalar(
            f"strategy/seals/activated/{seal.lower()}_per_1k_steps",
            (
                rm.blue_seals_activated
                if seal == "Blue"
                else rm.purple_seals_activated
            )
            / steps_per_thousand,
            step,
        )


def _write_action_behavior_metrics(writer, rm: _RolloutMetrics, step: int) -> None:
    """Write compact action-family and no-progress-loop diagnostics."""

    action_total = sum(rm.action_type_counts.values())
    if action_total:
        for action_type in _ACTION_TYPES:
            writer.add_scalar(
                f"actions/type/{action_type.value}_fraction",
                rm.action_type_counts.get(action_type.value, 0) / action_total,
                step,
            )
    if rm.joker_order_money_flags:
        # Money mode is only ever taken on a play the harness proved would
        # clear the blind anyway, so chips_forgone is the price of that cash
        # and should stay small relative to the target it still met.
        writer.add_scalar(
            "joker_order/money_objective_fraction",
            float(np.mean(rm.joker_order_money_flags)),
            step,
        )
        writer.add_scalar(
            "joker_order/clearing_play_fraction",
            float(np.mean(rm.joker_order_clear_flags)),
            step,
        )
        writer.add_scalar(
            "joker_order/dollars_gained_per_play",
            float(np.mean(rm.joker_order_dollars_gained)),
            step,
        )
        writer.add_scalar(
            "joker_order/chips_forgone_per_play",
            float(np.mean(rm.joker_order_chips_forgone)),
            step,
        )
    if rm.steps_since_progress:
        writer.add_scalar(
            "rollout/no_progress_streak_p95",
            float(np.percentile(rm.steps_since_progress, 95)),
            step,
        )
        writer.add_scalar(
            "rollout/no_progress_streak_max",
            float(np.max(rm.steps_since_progress)),
            step,
        )


def _write_terminal_loss_metrics(writer, rm: _RolloutMetrics, step: int, *, win_ante: int) -> None:
    """Write a compact, numeric diagnosis of non-stall episode losses."""
    loss_count = len(rm.terminal_loss_antes)
    completed_count = len(rm.completed_episode_rewards)
    stall_count = int(sum(rm.completed_episode_stalls))
    nonstall_completed_count = max(completed_count - stall_count, 0)
    ante1_death_count = rm.terminal_loss_antes.count(1)

    writer.add_scalar("terminal/completed_episode_count", float(completed_count), step)
    writer.add_scalar("terminal/stall_episode_count", float(stall_count), step)
    writer.add_scalar(
        "terminal/nonstall_completed_episode_count",
        float(nonstall_completed_count),
        step,
    )
    writer.add_scalar("terminal/loss_count", float(loss_count), step)
    writer.add_scalar("terminal/loss_ante/denominator_count", float(loss_count), step)
    writer.add_scalar("terminal/loss_ante/1_count", float(ante1_death_count), step)
    writer.add_scalar(
        "terminal/ante1_death_per_nonstall_completed_episode",
        (ante1_death_count / nonstall_completed_count)
        if nonstall_completed_count
        else 0.0,
        step,
    )

    if loss_count:
        writer.add_scalar("terminal/loss_ante_mean", float(np.mean(rm.terminal_loss_antes)), step)
    for ante in range(1, win_ante + 1):
        fraction = rm.terminal_loss_antes.count(ante) / loss_count if loss_count else 0.0
        writer.add_scalar(f"terminal/loss_ante/{ante}_fraction", fraction, step)
    writer.add_scalar("terminal/loss_blind/denominator_count", float(loss_count), step)
    for blind in ("small", "big", "boss"):
        fraction = rm.terminal_loss_blind_counts[blind] / loss_count if loss_count else 0.0
        writer.add_scalar(f"terminal/loss_blind/{blind}_fraction", fraction, step)

    if rm.terminal_loss_score_ratios:
        writer.add_scalar(
            "terminal/loss_score_ratio_mean",
            float(np.mean(rm.terminal_loss_score_ratios)),
            step,
        )
        writer.add_scalar(
            "terminal/loss_score_ratio_p50",
            float(np.median(rm.terminal_loss_score_ratios)),
            step,
        )
    if rm.terminal_loss_dollars:
        writer.add_scalar("terminal/loss_cash_mean", float(np.mean(rm.terminal_loss_dollars)), step)
        writer.add_scalar(
            "terminal/loss_cash_ge_10_fraction",
            float(np.mean(rm.terminal_loss_cash_ge_10)),
            step,
        )
        writer.add_scalar(
            "terminal/loss_joker_full_weak_fraction",
            float(np.mean(rm.terminal_loss_joker_full_weak)),
            step,
        )
    if rm.terminal_loss_last_play_top1:
        legal_top1_fraction = float(np.mean(rm.terminal_loss_last_play_top1))
        writer.add_scalar(
            "terminal/loss_last_play/top1_fraction",
            legal_top1_fraction,
            step,
        )
        writer.add_scalar(
            "terminal/loss_last_play/legal_top1_fraction",
            legal_top1_fraction,
            step,
        )
    if rm.terminal_loss_last_play_value_ratios:
        legal_value_ratio = float(np.mean(rm.terminal_loss_last_play_value_ratios))
        writer.add_scalar(
            "terminal/loss_last_play/value_ratio_mean",
            legal_value_ratio,
            step,
        )
        writer.add_scalar(
            "terminal/loss_last_play/legal_candidate_value_ratio_mean",
            legal_value_ratio,
            step,
        )
    for boss_key, count in rm.terminal_boss_loss_counts.items():
        writer.add_scalar(f"terminal/boss_loss/{boss_key}_count", float(count), step)


def _write_ante1_metrics(writer, rm: _RolloutMetrics, step: int) -> None:
    """Write bounded Ante-1 scoring telemetry with explicit denominators."""

    ante1_death_count = rm.terminal_loss_antes.count(1)
    writer.add_scalar("ante1/death/count", float(ante1_death_count), step)
    for blind in ("small", "big", "boss"):
        writer.add_scalar(
            f"ante1/blind/{blind}/death_count",
            float(rm.ante1_death_blind_counts.get(blind, 0)),
            step,
        )
        writer.add_scalar(
            f"ante1/blind/{blind}/clear_count",
            float(rm.ante1_blind_clear_counts.get(blind, 0)),
            step,
        )

    clear_count = sum(rm.ante1_blind_clear_counts.values())
    writer.add_scalar("ante1/clear/count", float(clear_count), step)
    writer.add_scalar(
        "ante1/clear/hands_used_mean",
        float(np.mean(rm.ante1_clear_hands_used))
        if rm.ante1_clear_hands_used
        else 0.0,
        step,
    )
    writer.add_scalar(
        "ante1/clear/hands_unused_mean",
        float(np.mean(rm.ante1_clear_hands_unused))
        if rm.ante1_clear_hands_unused
        else 0.0,
        step,
    )
    writer.add_scalar(
        "ante1/clear/discards_used_mean",
        float(np.mean(rm.ante1_clear_discards_used))
        if rm.ante1_clear_discards_used
        else 0.0,
        step,
    )

    writer.add_scalar("ante1/play/count", float(rm.ante1_play_count), step)
    realized_count = len(rm.ante1_play_realized_to_remaining_target)
    writer.add_scalar(
        "ante1/play/realized_progress_count",
        float(realized_count),
        step,
    )
    writer.add_scalar(
        "ante1/play/realized_score_to_remaining_target_mean",
        float(np.mean(rm.ante1_play_realized_to_remaining_target))
        if realized_count
        else 0.0,
        step,
    )
    comparison_count = len(rm.ante1_conservative_chosen_best_ratios)
    writer.add_scalar(
        "ante1/play/conservative_proxy_comparison_count",
        float(comparison_count),
        step,
    )
    writer.add_scalar(
        "ante1/play/conservative_chosen_best_ratio_mean",
        float(np.mean(rm.ante1_conservative_chosen_best_ratios))
        if comparison_count
        else 0.0,
        step,
    )

    writer.add_scalar(
        "ante1/one_hand_clear_proxy/opportunity_count",
        float(rm.ante1_one_hand_clear_proxy_observed),
        step,
    )
    for outcome, count in (
        ("available", rm.ante1_one_hand_clear_proxy_available),
        ("chosen", rm.ante1_one_hand_clear_proxy_chosen),
        ("missed", rm.ante1_one_hand_clear_proxy_missed),
    ):
        writer.add_scalar(
            f"ante1/one_hand_clear_proxy/{outcome}_count",
            float(count),
            step,
        )

    hand_total = sum(rm.ante1_play_hand_counts.values())
    writer.add_scalar("ante1/hand_type/denominator_count", float(hand_total), step)
    for hand_name in POKER_HAND_NAMES:
        tag_name = hand_name.lower().replace(" ", "_")
        writer.add_scalar(
            f"ante1/hand_type/{tag_name}_share",
            (rm.ante1_play_hand_counts.get(hand_name, 0) / hand_total)
            if hand_total
            else 0.0,
            step,
        )


def _write_risk_calibration_metrics(writer, rm: _RolloutMetrics, step: int) -> None:
    """Write compact calibration of shop danger estimates against survival."""

    if rm.risk_shop_death_briers:
        writer.add_scalar(
            "strategy/risk/shop_death_brier",
            float(np.mean(rm.risk_shop_death_briers)),
            step,
        )
        writer.add_scalar(
            "strategy/risk/predicted_death_mean",
            float(np.mean(rm.risk_shop_death_predictions)),
            step,
        )
        writer.add_scalar(
            "strategy/risk/actual_death_rate",
            float(np.mean(rm.risk_shop_death_outcomes)),
            step,
        )
        auc = _binary_roc_auc(rm.risk_shop_death_predictions, rm.risk_shop_death_outcomes)
        if auc is not None:
            writer.add_scalar("strategy/risk/death_auc", auc, step)
    if rm.risk_shop_raw_death_briers:
        writer.add_scalar(
            "strategy/risk/raw_shop_death_brier",
            float(np.mean(rm.risk_shop_raw_death_briers)),
            step,
        )
        writer.add_scalar(
            "strategy/risk/raw_predicted_death_mean",
            float(np.mean(rm.risk_shop_raw_death_predictions)),
            step,
        )
    if rm.risk_ante1_false_safe_deaths:
        writer.add_scalar(
            "strategy/risk/ante1_false_safe_death_fraction",
            float(np.mean(rm.risk_ante1_false_safe_deaths)),
            step,
        )


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model when wrapped for multi-GPU training."""
    return model.module if isinstance(model, nn.DataParallel) else model


def _grammar_distribution(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    temperature: float | torch.Tensor = 1.0,
):
    import inspect

    base_model = _unwrap_model(model)
    model_type = type(base_model)
    supports_history = _HISTORY_SIGNATURE_CACHE.get(model_type)
    if supports_history is None:
        parameters = inspect.signature(base_model.action_distribution).parameters
        supports_history = "history_events" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        _HISTORY_SIGNATURE_CACHE[model_type] = supports_history
    history_kwargs = {
        key: batch[key]
        for key in (
            "history_events",
            "history_event_features",
            "history_cards",
            "history_card_mask",
            "history_jokers",
            "history_joker_mask",
            "history_event_mask",
            "history_round_mask",
            "history_omitted",
        )
        if supports_history and key in batch
    }
    if isinstance(model, nn.DataParallel) and isinstance(base_model, BalatroAgent):
        raw_output, value_dict = model(
            batch["tokens"],
            batch["token_types"],
            batch["scalars"],
            batch["attention_mask"],
            batch["action_mask"],
            temperature=temperature,
            return_raw_outputs=True,
            **history_kwargs,
        )
        grammar_output = ActionGrammarOutput(**raw_output)
        return (
            ActionGrammarDistribution(
                grammar_output,
                batch["action_mask"],
                temperature=temperature,
                tokens=batch["tokens"],
                hand_ar_mixture_eps=base_model.config.hand_ar_mixture_eps,
            ),
            value_dict,
        )
    return base_model.action_distribution(
        batch["tokens"],
        batch["token_types"],
        batch["scalars"],
        batch["attention_mask"],
        batch["action_mask"],
        temperature=temperature,
        **history_kwargs,
    )


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


def _observation_buffer_batch(obs_buf) -> dict[str, torch.Tensor]:
    """Expose an observation buffer through the common model-batch interface."""
    return {
        "tokens": obs_buf.tokens,
        "token_types": obs_buf.token_types,
        "scalars": obs_buf.scalars,
        "attention_mask": obs_buf.attention_mask,
        "action_mask": obs_buf.action_mask,
        "history_events": obs_buf.history_events,
        "history_event_features": obs_buf.history_event_features,
        "history_cards": obs_buf.history_cards,
        "history_card_mask": obs_buf.history_card_mask,
        "history_jokers": obs_buf.history_jokers,
        "history_joker_mask": obs_buf.history_joker_mask,
        "history_event_mask": obs_buf.history_event_mask,
        "history_round_mask": obs_buf.history_round_mask,
        "history_omitted": obs_buf.history_omitted,
    }


def _policy_temperature_for_scalars(
    scalars: torch.Tensor,
    config: PPOConfig,
) -> float | torch.Tensor:
    """Return the on-policy temperature for each observation row.

    Only active hand-play states are sharpened. Ante 1 is protected regardless
    of the analytic risk estimate; later hands are sharpened when immediate
    death probability crosses the configured threshold. Because this function
    is used for rollout collection, PPO minibatches, SIL, and KL diagnostics,
    old and new log-probabilities remain distributions over the same policy.
    """

    danger_temperature = config.danger_rollout_temperature
    if danger_temperature is None:
        return config.rollout_temperature

    base = torch.full(
        (scalars.shape[0],),
        float(config.rollout_temperature),
        dtype=scalars.dtype,
        device=scalars.device,
    )
    # Tokenizer scalar layout: ante=2, sub_phase=7 (CHOOSE_ACTION=1),
    # immediate_death_probability=12.
    active_hand = scalars[:, 7].round().eq(1)
    opening_ante = scalars[:, 2] <= 1.0
    immediate_danger = scalars[:, 12] >= float(config.danger_death_probability_threshold)
    sharpen = active_hand & (opening_ante | immediate_danger)
    danger = torch.full_like(base, min(float(danger_temperature), float(config.rollout_temperature)))
    return torch.where(sharpen, danger, base)


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


def _seed_training_rngs(seed: int) -> None:
    """Seed process RNGs used by model initialization and PPO sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_ppo_config(config: PPOConfig) -> None:
    """Raise ValueError for invalid combinations; warn on risky ones."""
    validate_critic_win_ante(config.win_ante if config.win_ante is not None else 8)
    if config.archive_config is not None and (config.win_ante or 8) != 8:
        raise ValueError("Archive training requires win_ante=8")
    if (config.reward_config is not None and config.reward_config.objective == "milestone"
            and (config.win_ante or 8) != 8):
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
    config: "PPOConfig",
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


def _resolve_schedule_total_steps(config: "PPOConfig", resume_state: dict | None) -> int:
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
    sil_buffer: "EpisodeReplayBuffer | None" = None,
    sil_coeff_now: float = 0.0,
    grad_diagnostics_due: bool = False,
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
    rollback_snapshot = (
        _snapshot_ppo_update(model, optimizer) if config.target_kl_max is not None else None
    )
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
            valid_action_type_count_mean = _mean_valid_action_type_count(batch["action_mask"])

            # Policy loss (clipped PPO). Advantages are already normalized
            # once over the full rollout in buffer.normalize_advantages();
            # per-mini-batch normalization would let rare terminals
            # dominate their batch and crush others to noise.
            on_policy_count = torch.as_tensor(float(batch["actions"].numel()), device=device)
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
                outcome_brier_per_row = (
                    value_dict["outcome_probabilities"] - one_hot_outcome
                ).square().sum(dim=-1)
                outcome_brier = (
                    outcome_brier_per_row * outcome_mask
                ).sum() / outcome_denominator
                win_target = (
                    batch["terminal_outcome_target"]
                    == value_dict["outcome_probabilities"].shape[1] - 1
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
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                on_policy_fraction = 1.0
                clip_frac = ((ratio - 1.0).abs() > effective_clip_epsilon).float().mean().item()
                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                valid_action_count_mean = valid_action_counts.float().mean().item()
                stats.on_policy_advantage_means.append(advantages.mean().item())
                stats.on_policy_advantage_stds.append(advantages.std(unbiased=False).item())
                stats.on_policy_positive_advantage_fractions.append((advantages > 0).float().mean().item())
                stats.on_policy_return_means.append(batch["returns"].mean().item())
            stats.policy_losses.append(policy_loss.item())
            stats.return_hubers.append(return_loss.item())
            stats.outcome_nlls.append(terminal_nll.item())
            stats.outcome_briers.append(outcome_brier.item())
            stats.derived_win_briers.append(derived_win_brier.item())
            stats.outcome_valid_counts.append(outcome_valid_counts[i])
            stats.terminal_value_means.append(value_dict["terminal_value"].mean().item())
            stats.return_residual_means.append(value_dict["return_residual"].mean().item())
            stats.expected_return_means.append(value_dict["expected_return"].mean().item())
            stats.entropies.append(entropy.item())
            stats.normalized_entropies.append(normalized_entropy.item())
            stats.action_type_entropies.append(normalized_action_type_entropy.item())
            stats.clip_fracs.append(clip_frac)
            stats.approx_kls.append(approx_kl)
            stats.valid_action_counts.append(valid_action_count_mean)
            stats.valid_action_type_counts.append(valid_action_type_count_mean)
            stats.on_policy_fractions.append(on_policy_fraction)
            minibatches_processed += 1
            on_policy_samples = int(on_policy_count.item())
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
        if parameter.requires_grad
        and name.startswith(("outcome_proj.", "ante_survival."))
    ]


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

    outcome_probabilities = value_dict["outcome_probabilities"]
    outcome_targets = sampled["terminal_outcome_target"]
    num_classes = outcome_probabilities.shape[1]
    selected = outcome_probabilities.gather(1, outcome_targets.unsqueeze(1)).squeeze(1)
    outcome_nll_per_row = -selected.clamp_min(1e-7).log()
    one_hot_outcome = F.one_hot(outcome_targets, num_classes=num_classes).to(
        dtype=outcome_probabilities.dtype
    )
    outcome_brier = (outcome_probabilities - one_hot_outcome).square().sum(dim=-1)
    win_targets = (outcome_targets == num_classes - 1).float()
    derived_win_brier = (value_dict["win_prob"] - win_targets).square()

    cross_rollout = sampled["cross_rollout_flags"] > 0.5
    current_antes = sampled["current_antes"].long()
    buckets: list[tuple[str, torch.Tensor | None]] = [
        ("", None),
        ("cross_rollout/", cross_rollout),
        ("same_rollout/", ~cross_rollout),
        *[
            (f"ante_{ante}/", current_antes == int(ante))
            for ante in current_antes.unique().tolist()
        ],
    ]

    rows: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
    for prefix, bucket_mask in buckets:
        if bucket_mask is not None and not bool(bucket_mask.any()):
            continue
        bucket_outcomes = (
            one_hot_outcome if bucket_mask is None else one_hot_outcome[bucket_mask]
        )
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
            diagnostics[prefix + "outcome_brier_skill"] = (
                1.0 - metric_value / climatology_value
            )


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

    Training draws exclude the withheld episodes entirely. After the training
    iterations a no-grad pass scores a batch drawn only from those withheld
    episodes, so ``holdout/*`` reports fit on episodes no optimizer step has
    ever seen while the unprefixed metrics report fit on the training half.
    """

    diagnostics: dict[str, float] = {
        "buffer_episodes": float(episode_buffer.num_episodes),
        "buffer_transitions": float(episode_buffer.num_transitions),
        "buffer_win_fraction": float(episode_buffer.win_fraction),
        "buffer_holdout_episodes": float(episode_buffer.num_holdout_episodes),
        "label_coverage": float(episode_buffer.labeled_transition_fraction),
        "cross_rollout_transition_fraction": float(
            episode_buffer.cross_rollout_transition_fraction
        ),
        "episodes_added_total": float(episode_buffer.episodes_added_total),
        "stalled_episodes_dropped_total": float(
            episode_buffer.stalled_episodes_dropped_total
        ),
        "overflow_episodes_dropped_total": float(
            episode_buffer.overflow_episodes_dropped_total
        ),
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
        _, value_dict = _grammar_distribution(
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
        )
        if sampled is None:
            break

        optimizer.zero_grad()
        value_dict = forward_outcomes(sampled)
        if value_dict.get("outcome_probabilities") is None:
            optimizer.zero_grad()
            break
        outcome_targets = sampled["terminal_outcome_target"]
        selected_outcome_probability = value_dict["outcome_probabilities"].gather(
            1,
            outcome_targets.unsqueeze(1),
        ).squeeze(1)
        outcome_loss = -selected_outcome_probability.clamp_min(1e-7).log().mean()

        loss = config.outcome_loss_coeff * outcome_loss
        _accumulate_critic_grads_into_value_head(loss, terminal_params)
        nn.utils.clip_grad_norm_(terminal_params, config.max_grad_norm)
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
        )
        if holdout is not None:
            with torch.no_grad():
                holdout_values = forward_outcomes(holdout)
                if holdout_values.get("outcome_probabilities") is not None:
                    diagnostics["holdout/samples"] = float(
                        holdout["terminal_outcome_target"].numel()
                    )
                    for key, (values, mask) in _outcome_metric_rows(
                        holdout_values, holdout
                    ).items():
                        record_metric("holdout/" + key, values, mask)

    for key, total in metric_sums.items():
        diagnostics[key] = total / metric_counts[key]
    _add_brier_skills(diagnostics)
    diagnostics["updates_applied"] = float(updates_applied)
    diagnostics["samples"] = float(samples_seen)
    return _TerminalReplayResult(updates_applied, samples_seen, diagnostics)

def _compute_sil_group_loss(
    model: nn.Module,
    sil_buffer: "EpisodeReplayBuffer | None",
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


def _save_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    save_path: Path,
    update_count: int,
    total_steps: int,
    planned_updates: int,
    entropy_coeff: float,
    entropy_signal_ema: float | None,
    lr: float,
    log_alpha: torch.Tensor | None = None,
    alpha_optimizer: torch.optim.Optimizer | None = None,
    agent_config: "AgentConfig | None" = None,
    config: "PPOConfig | None" = None,
    filename: str | None = None,
    extra: dict | None = None,
    vector_env=None,
) -> Path:
    """Persist a full PPO resume checkpoint and return its path."""
    from ..checkpoint import save_ppo_checkpoint

    checkpoint_path = save_path / (filename or f"ppo_update{update_count}.pt")
    # Snapshot the PPOConfig fields that affect resume correctness so a future
    # resume can warn on mismatched hyperparameters (kept lightweight: only
    # scalar scheduling fields, not the full dataclass).
    config_fields: dict[str, object] = {}
    if config is not None:
        for key in (
            "seed",
            "ppo_epochs",
            "mini_batch_size",
            "micro_batch_size",
            "lr",
            "clip_epsilon",
            "gae_lambda",
            "rollout_temperature",
            "danger_rollout_temperature",
            "danger_death_probability_threshold",
            "action_type_entropy_scale",
            "max_no_progress_steps",
            "eval_regression_tolerance",
            "eval_regression_patience",
            "ppo_run_uuid",
            "ppo_source_sha256",
            "ppo_recipe_id",
            "gamma",
            "win_ante",
            "target_entropy",
            "entropy_ema_beta",
            "adaptive_entropy",
            "milestone_final_scale",
            "milestone_decay_fraction",
        ):
            config_fields[key] = getattr(config, key)
        config_fields["archive_config"] = asdict(config.archive_config) if config.archive_config else None
    active_reward_config = _effective_reward_config(config) if config is not None else DEFAULT_REWARD_CONFIG
    checkpoint_extra = dict(extra or {})
    transfer_metadata = getattr(_unwrap_model(model), "_actor_transfer_metadata", None)
    if transfer_metadata is not None:
        checkpoint_extra["actor_transfer"] = transfer_metadata
    if config is not None and config.archive_config is not None:
        if vector_env is None:
            raise ValueError("Archive checkpoints require the training environments")
        checkpoint_extra["archive_states"] = list(vector_env.call("archive_state_dict"))
    checkpoint_extra["ppo_active_lr"] = float(optimizer.param_groups[0]["lr"])
    if config is not None:
        provenance = _ppo_run_provenance(config)
        if provenance is not None:
            checkpoint_extra["ppo_run_provenance"] = provenance
    save_ppo_checkpoint(
        _unwrap_model(model),
        checkpoint_path,
        optimizer=optimizer,
        update_count=update_count,
        total_steps=total_steps,
        planned_updates=planned_updates,
        entropy_coeff=entropy_coeff,
        entropy_signal_ema=entropy_signal_ema,
        lr=lr,
        log_alpha=log_alpha,
        alpha_optimizer=alpha_optimizer,
        agent_config=agent_config,
        reward_config=active_reward_config,
        ppo_config_fields=config_fields,
        extra=checkpoint_extra,
    )
    return checkpoint_path


def _mirror_latest_checkpoint(checkpoint_path: Path, save_path: Path) -> Path:
    """Mirror a committed checkpoint to the stable strict-resume path."""

    latest_path = save_path / "ppo_latest.pt"
    try:
        latest_path.unlink(missing_ok=True)
        shutil.copy2(checkpoint_path, latest_path)
    except OSError:
        logger.warning("Could not mirror latest checkpoint to %s", latest_path)
    return latest_path


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


def _extract_vector_info_value(info_dict: dict, key: str, env_idx: int, default=None):
    """Read one env's value from Gymnasium's vector-info dict-of-arrays format."""
    values = info_dict.get(key)
    if values is None:
        return default

    mask = info_dict.get(f"_{key}")
    if mask is not None and not bool(mask[env_idx]):
        return default

    value = values[env_idx]
    return value.item() if isinstance(value, np.generic) else value


def _extract_step_info_value(info_dict: dict, key: str, env_idx: int, *, done: bool, default=None):
    """Read the just-finished step's info, preferring final_info on autoresets."""
    if done:
        final_info = info_dict.get("final_info")
        if isinstance(final_info, dict):
            final_value = _extract_vector_info_value(final_info, key, env_idx, _MISSING)
            if final_value is not _MISSING:
                return final_value

    value = _extract_vector_info_value(info_dict, key, env_idx, _MISSING)
    return default if value is _MISSING else value


def _extract_step_count(info_dict: dict, key: str, env_idx: int, *, done: bool) -> int:
    return int(_extract_step_info_value(info_dict, key, env_idx, done=done, default=0) or 0)


def _extract_step_flag(info_dict: dict, key: str, env_idx: int, *, done: bool) -> bool:
    return bool(_extract_step_info_value(info_dict, key, env_idx, done=done, default=False))


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


def _ppo_terminal_flags(
    terminated: np.ndarray,
    truncated: np.ndarray,
    infos: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map external Gymnasium endings to PPO bootstrap semantics.

    No-progress stalls stay ``truncated=True`` at the environment boundary, but
    they carry a terminal loss reward and are absorbing for value targets. Other
    truncations keep Gymnasium's bootstrap-from-final-observation behavior.
    """
    dones = terminated | truncated
    stalled = np.asarray(
        [
            bool(
                _extract_step_info_value(
                    infos,
                    "stalled",
                    env_idx,
                    done=bool(dones[env_idx]),
                    default=False,
                )
            )
            for env_idx in range(len(dones))
        ],
        dtype=np.bool_,
    )
    ppo_terminated = np.asarray(terminated, dtype=np.bool_) | stalled
    ppo_truncated = np.asarray(truncated, dtype=np.bool_) & ~ppo_terminated
    return ppo_terminated, ppo_truncated, stalled


def _record_action_diagnostics(rm: _RolloutMetrics, infos: dict, env_idx: int, *, done: bool) -> None:
    """Aggregate optional env-provided decision-quality diagnostics."""
    if _extract_step_info_value(
        infos,
        "ante1_play_observed",
        env_idx,
        done=done,
        default=False,
    ):
        rm.ante1_play_count += 1
        hand_name = _extract_step_info_value(
            infos,
            "ante1_play_hand",
            env_idx,
            done=done,
            default="",
        )
        if hand_name:
            rm.ante1_play_hand_counts[str(hand_name)] += 1
        realized_progress = _extract_step_info_value(
            infos,
            "ante1_play_realized_to_remaining_target",
            env_idx,
            done=done,
            default=None,
        )
        if realized_progress is not None:
            rm.ante1_play_realized_to_remaining_target.append(float(realized_progress))
        conservative_ratio = _extract_step_info_value(
            infos,
            "ante1_conservative_chosen_best_ratio",
            env_idx,
            done=done,
            default=None,
        )
        if conservative_ratio is not None:
            rm.ante1_conservative_chosen_best_ratios.append(float(conservative_ratio))
        if _extract_step_info_value(
            infos,
            "ante1_one_hand_clear_proxy_observed",
            env_idx,
            done=done,
            default=False,
        ):
            rm.ante1_one_hand_clear_proxy_observed += 1
            rm.ante1_one_hand_clear_proxy_available += int(
                bool(
                    _extract_step_info_value(
                        infos,
                        "ante1_one_hand_clear_proxy_available",
                        env_idx,
                        done=done,
                        default=False,
                    )
                )
            )
            rm.ante1_one_hand_clear_proxy_chosen += int(
                bool(
                    _extract_step_info_value(
                        infos,
                        "ante1_one_hand_clear_proxy_chosen",
                        env_idx,
                        done=done,
                        default=False,
                    )
                )
            )
            rm.ante1_one_hand_clear_proxy_missed += int(
                bool(
                    _extract_step_info_value(
                        infos,
                        "ante1_one_hand_clear_proxy_missed",
                        env_idx,
                        done=done,
                        default=False,
                    )
                )
            )
    if _extract_step_info_value(
        infos,
        "ante1_blind_cleared",
        env_idx,
        done=done,
        default=False,
    ):
        blind = str(
            _extract_step_info_value(
                infos,
                "ante1_blind_clear_type",
                env_idx,
                done=done,
                default="",
            )
            or ""
        ).lower()
        rm.ante1_blind_clear_counts[blind] += 1
        rm.ante1_clear_hands_used.append(
            float(
                _extract_step_info_value(
                    infos,
                    "ante1_blind_clear_hands_used",
                    env_idx,
                    done=done,
                    default=0,
                )
            )
        )
        rm.ante1_clear_hands_unused.append(
            float(
                _extract_step_info_value(
                    infos,
                    "ante1_blind_clear_hands_unused",
                    env_idx,
                    done=done,
                    default=0,
                )
            )
        )
        rm.ante1_clear_discards_used.append(
            float(
                _extract_step_info_value(
                    infos,
                    "ante1_blind_clear_discards_used",
                    env_idx,
                    done=done,
                    default=0,
                )
            )
        )
    if _extract_step_info_value(
        infos,
        "joker_replacement_sequence",
        env_idx,
        done=done,
        default=False,
    ):
        rm.joker_replacement_events += 1
    for info_key, counter in (
        ("consumable_use_set", rm.consumable_use_set_counts),
        ("pack_claim_set", rm.consumable_claim_set_counts),
        ("shop_bought_consumable_set", rm.consumable_buy_set_counts),
    ):
        consumable_set = _extract_step_info_value(infos, info_key, env_idx, done=done, default="")
        if consumable_set:
            counter[str(consumable_set)] += 1
    for consumable_set in ("Planet", "Tarot"):
        tag = consumable_set.lower()
        for info_key, counter in (
            (f"consumable_{tag}_offered_count", rm.consumable_offered_counts),
            (f"consumable_{tag}_claimable_count", rm.consumable_claimable_counts),
            (
                f"consumable_{tag}_inventory_full_blocked_count",
                rm.consumable_inventory_full_blocked_counts,
            ),
            (f"strategic_{tag}_acquired", rm.consumable_acquired_counts),
            (f"strategic_{tag}_uses", rm.consumable_exact_use_counts),
            (f"strategic_{tag}_pack_auto_uses", rm.consumable_pack_auto_use_counts),
            (f"strategic_{tag}_sold", rm.consumable_sold_counts),
            (f"strategic_{tag}_overwritten", rm.consumable_overwritten_counts),
            (f"strategic_{tag}_expired", rm.consumable_expired_counts),
        ):
            counter[consumable_set] += _extract_step_count(
                infos, info_key, env_idx, done=done
            )
        rm.consumable_owned_states[consumable_set] += int(
            _extract_step_flag(
                infos, f"consumable_{tag}_owned_state", env_idx, done=done
            )
        )
        rm.consumable_legal_use_opportunities[consumable_set] += int(
            _extract_step_flag(
                infos,
                f"consumable_{tag}_legal_use_opportunity",
                env_idx,
                done=done,
            )
        )
        rm.consumable_eligible_offer_opportunities[consumable_set] += int(
            _extract_step_flag(
                infos,
                f"consumable_{tag}_eligible_offer_opportunity",
                env_idx,
                done=done,
            )
        )

    rm.planet_active_plan_owned += int(
        _extract_step_flag(
            infos, "consumable_planet_active_plan_owned", env_idx, done=done
        )
    )
    rm.planet_active_plan_legal += int(
        _extract_step_flag(
            infos, "consumable_planet_active_plan_legal", env_idx, done=done
        )
    )

    exact_planet_uses = _extract_step_count(
        infos, "strategic_planet_uses", env_idx, done=done
    )
    planet_pack_auto_uses = _extract_step_count(
        infos, "strategic_planet_pack_auto_uses", env_idx, done=done
    )
    if exact_planet_uses:
        prefix = (
            "planet_use"
            if _extract_step_flag(infos, "planet_use_observed", env_idx, done=done)
            else "planet_claim"
        )
        supported = _extract_step_flag(
            infos, f"{prefix}_plan_supported", env_idx, done=done
        )
        alignment = "matched" if supported else "unmatched"
        rm.planet_use_alignment_counts[alignment] += exact_planet_uses
        hand_type = str(
            _extract_step_info_value(
                infos,
                f"{prefix}_hand_type",
                env_idx,
                done=done,
                default="",
            )
            or ""
        )
        if hand_type:
            rm.planet_use_hand_counts[hand_type] += exact_planet_uses
        if _extract_step_flag(
            infos, f"{prefix}_active_plan_match", env_idx, done=done
        ):
            rm.planet_active_plan_uses += max(
                exact_planet_uses - planet_pack_auto_uses,
                0,
            )

    tarot_uses = _extract_step_count(infos, "strategic_tarot_uses", env_idx, done=done)
    tarot_family = str(
        _extract_step_info_value(
            infos,
            "strategic_tarot_family",
            env_idx,
            done=done,
            default="",
        )
        or ""
    )
    if tarot_uses and tarot_family:
        rm.tarot_use_family_counts[tarot_family] += tarot_uses
        if tarot_family in {"deck_cut", "rank_fix", "suit_fix"}:
            rm.tarot_fix_reliability_deltas.append(
                float(
                    _extract_step_info_value(
                        infos,
                        "strategic_tarot_fix_reliability_delta",
                        env_idx,
                        done=done,
                        default=0.0,
                    )
                )
            )
        if tarot_family == "cash":
            rm.attributable_cash_payouts.append(
                float(
                    _extract_step_info_value(
                        infos,
                        "strategic_attributable_cash_payout",
                        env_idx,
                        done=done,
                        default=0.0,
                    )
                )
            )

    rm.gold_cards_created += _extract_step_count(
        infos, "strategic_gold_created_tarot", env_idx, done=done
    ) + _extract_step_count(
        infos, "strategic_gold_created_midas", env_idx, done=done
    )
    rm.held_gold_payout_dollars += _extract_step_count(
        infos, "strategic_held_gold_payout", env_idx, done=done
    )
    claimed_seal = _extract_step_info_value(infos, "pack_claim_seal", env_idx, done=done, default="")
    if claimed_seal:
        rm.pack_claim_seal_counts[str(claimed_seal)] += 1
    for seal in ("Blue", "Purple"):
        rm.pack_offered_seal_counts[seal] += _extract_step_count(
            infos, f"seal_{seal.lower()}_offered_count", env_idx, done=done
        )
    rm.blue_seals_activated += _extract_step_count(
        infos, "strategic_blue_seals_activated", env_idx, done=done
    )
    rm.purple_seals_activated += _extract_step_count(
        infos, "strategic_purple_seals_activated", env_idx, done=done
    )
    if _extract_step_info_value(infos, "shop_leave_observed", env_idx, done=done, default=False):
        rm.shop_leave_flags.append(1.0)
        rm.shop_unsafe_leave_flags.append(
            float(_extract_step_info_value(infos, "shop_unsafe_leave", env_idx, done=done, default=False))
        )
        rm.shop_unsafe_can_reroll_flags.append(
            float(
                _extract_step_info_value(
                    infos,
                    "shop_unsafe_can_reroll",
                    env_idx,
                    done=done,
                    default=False,
                )
            )
        )
        rm.shop_missed_upgrade_flags.append(
            float(
                _extract_step_info_value(
                    infos,
                    "shop_missed_confident_upgrade",
                    env_idx,
                    done=done,
                    default=False,
                )
            )
        )
        rm.shop_leave_full_weak_flags.append(
            float(
                _extract_step_info_value(
                    infos,
                    "shop_leave_joker_full_weak",
                    env_idx,
                    done=done,
                    default=False,
                )
            )
        )
        upgrade_delta = _extract_step_info_value(
            infos,
            "shop_best_confident_upgrade_delta",
            env_idx,
            done=done,
            default=None,
        )
        if upgrade_delta is not None:
            rm.shop_best_upgrade_deltas.append(float(upgrade_delta))
    rm.purple_seal_tarots_generated += int(
        _extract_step_info_value(
            infos,
            "strategic_purple_tarots_generated",
            env_idx,
            done=done,
            default=_extract_step_info_value(
                infos,
                "purple_seal_tarot_generated_count",
                env_idx,
                done=done,
                default=0,
            ),
        )
    )
    rm.blue_seal_planets_generated += int(
        _extract_step_info_value(
            infos,
            "strategic_blue_planets_generated",
            env_idx,
            done=done,
            default=_extract_step_info_value(
                infos,
                "blue_seal_planet_generated_count",
                env_idx,
                done=done,
                default=0,
            ),
        )
    )
    if _extract_step_info_value(infos, "hand_play_observed", env_idx, done=done, default=False):
        rm.hand_play_observed.append(1.0)
        not_in_candidates = bool(
            _extract_step_info_value(infos, "hand_play_not_in_candidates", env_idx, done=done, default=False)
        )
        in_candidates = bool(
            _extract_step_info_value(infos, "hand_play_in_candidates", env_idx, done=done, default=False)
        )
        rm.hand_play_not_in_candidates.append(float(not_in_candidates))
        rm.hand_play_in_candidates.append(float(in_candidates))

        if in_candidates:
            rm.hand_play_top1.append(
                float(
                    _extract_step_info_value(
                        infos,
                        "hand_play_legal_top1",
                        env_idx,
                        done=done,
                        default=_extract_step_info_value(
                            infos,
                            "hand_play_top1",
                            env_idx,
                            done=done,
                            default=False,
                        ),
                    )
                )
            )
            rm.hand_play_top3.append(
                float(
                    _extract_step_info_value(
                        infos,
                        "hand_play_top3",
                        env_idx,
                        done=done,
                        default=False,
                    )
                )
            )
            value_ratio = _extract_step_info_value(
                infos,
                "hand_play_legal_candidate_value_ratio",
                env_idx,
                done=done,
                default=_extract_step_info_value(
                    infos,
                    "hand_play_candidate_value_ratio",
                    env_idx,
                    done=done,
                    default=None,
                ),
            )
            if value_ratio is not None:
                rm.hand_play_value_ratios.append(float(value_ratio))
            chosen_hand = _extract_step_info_value(infos, "hand_play_chosen_hand", env_idx, done=done, default="")
            if chosen_hand:
                rm.hand_chosen_counts[str(chosen_hand)] += 1

        best_hand = _extract_step_info_value(infos, "hand_play_best_hand", env_idx, done=done, default="")
        if best_hand:
            rm.hand_best_counts[str(best_hand)] += 1

    if _extract_step_info_value(infos, "planet_use_observed", env_idx, done=done, default=False):
        rm.planet_use_observed.append(1.0)
        rm.planet_use_played_hand.append(
            float(
                _extract_step_info_value(
                    infos,
                    "planet_use_played_hand",
                    env_idx,
                    done=done,
                    default=False,
                )
            )
        )
        rm.planet_use_play_share.append(
            float(
                _extract_step_info_value(
                    infos,
                    "planet_use_play_share",
                    env_idx,
                    done=done,
                    default=0.0,
                )
            )
        )
        rm.planet_use_main_hand_match.append(
            float(
                _extract_step_info_value(
                    infos,
                    "planet_use_main_hand_match",
                    env_idx,
                    done=done,
                    default=False,
                )
            )
        )
        planet_key = _extract_step_info_value(infos, "planet_use_key", env_idx, done=done, default="")
        if planet_key:
            rm.planet_use_key_counts[str(planet_key)] += 1

    if _extract_step_info_value(infos, "planet_claim_observed", env_idx, done=done, default=False):
        rm.planet_claim_observed.append(1.0)
        rm.planet_claim_played_hand.append(
            float(
                _extract_step_info_value(
                    infos,
                    "planet_claim_played_hand",
                    env_idx,
                    done=done,
                    default=False,
                )
            )
        )
        rm.planet_claim_play_share.append(
            float(
                _extract_step_info_value(
                    infos,
                    "planet_claim_play_share",
                    env_idx,
                    done=done,
                    default=0.0,
                )
            )
        )
        rm.planet_claim_main_hand_match.append(
            float(
                _extract_step_info_value(
                    infos,
                    "planet_claim_main_hand_match",
                    env_idx,
                    done=done,
                    default=False,
                )
            )
        )
        planet_key = _extract_step_info_value(infos, "planet_claim_key", env_idx, done=done, default="")
        if planet_key:
            rm.planet_claim_key_counts[str(planet_key)] += 1

    if _extract_step_info_value(infos, "planet_pack_skip", env_idx, done=done, default=None) is not None:
        rm.planet_pack_skip.append(
            float(
                _extract_step_info_value(
                    infos,
                    "planet_pack_skip",
                    env_idx,
                    done=done,
                    default=False,
                )
            )
        )
        pack_state_name = _extract_step_info_value(infos, "pack_skip_state_name", env_idx, done=done, default="")
        if pack_state_name:
            rm.pack_skip_state_counts[str(pack_state_name)] += 1

    if _extract_step_info_value(infos, "shop_joker_offer_observed", env_idx, done=done, default=False):
        offered_count = int(
            _extract_step_info_value(
                infos,
                "shop_offered_joker_emitted_count",
                env_idx,
                done=done,
                default=0,
            )
        )
        for index in range(min(offered_count, MAX_DIAGNOSTIC_SHOP_JOKERS)):
            center = _extract_step_info_value(
                infos,
                f"shop_offered_joker_{index}_id",
                env_idx,
                done=done,
                default="",
            )
            if center:
                rm.shop_offered_joker_counts[str(center)] += 1

    for info_key, counter in (
        ("shop_bought_joker_id", rm.shop_bought_joker_counts),
        ("shop_sold_joker_id", rm.shop_sold_joker_counts),
    ):
        center = _extract_step_info_value(infos, info_key, env_idx, done=done, default="")
        if center:
            counter[str(center)] += 1

    if _extract_step_info_value(infos, "joker_roster_changed", env_idx, done=done, default=False):
        rm.joker_acquired_count += int(
            _extract_step_info_value(infos, "joker_acquired_count", env_idx, done=done, default=0)
        )
        rm.joker_removed_count += int(
            _extract_step_info_value(infos, "joker_removed_count", env_idx, done=done, default=0)
        )
        rm.joker_turnover_count += int(
            _extract_step_info_value(infos, "joker_turnover_count", env_idx, done=done, default=0)
        )
        rm.joker_churn_count += int(_extract_step_info_value(infos, "joker_churn_count", env_idx, done=done, default=0))
        rm.joker_replacement_events += int(
            bool(_extract_step_info_value(infos, "joker_replacement_event", env_idx, done=done, default=False))
        )
        for prefix, counter in (
            ("joker_acquired", rm.joker_acquired_id_counts),
            ("joker_removed", rm.joker_removed_id_counts),
        ):
            emitted = int(_extract_step_info_value(infos, f"{prefix}_emitted_count", env_idx, done=done, default=0))
            for index in range(min(emitted, MAX_DIAGNOSTIC_EVENTS)):
                center = _extract_step_info_value(
                    infos,
                    f"{prefix}_{index}_id",
                    env_idx,
                    done=done,
                    default="",
                )
                if center:
                    counter[str(center)] += 1

    if _extract_step_info_value(infos, "build_diagnostics_observed", env_idx, done=done, default=False):
        plan_type = _extract_step_info_value(infos, "hand_plan_post_type", env_idx, done=done, default="")
        if plan_type:
            rm.hand_plan_type_counts[str(plan_type)] += 1
        plan_reliability = _extract_step_info_value(
            infos,
            "hand_plan_post_reliability",
            env_idx,
            done=done,
            default=None,
        )
        if plan_reliability is not None:
            rm.hand_plan_reliability.append(float(plan_reliability))
        plan_readiness = _extract_step_info_value(
            infos,
            "hand_plan_post_readiness",
            env_idx,
            done=done,
            default=None,
        )
        if plan_readiness is not None:
            rm.hand_plan_readiness.append(float(plan_readiness))
        for prefix in ("build_pre", "build_post"):
            for metric in (
                "estimated_score",
                "required_score",
                "readiness",
                "score_gain_ratio",
                "modeled_fraction",
            ):
                value = _extract_step_info_value(infos, f"{prefix}_{metric}", env_idx, done=done, default=None)
                if value is not None:
                    rm.build_values[f"{prefix}_{metric}"].append(float(value))
        score_delta = _extract_step_info_value(infos, "build_estimated_score_delta", env_idx, done=done, default=None)
        if score_delta is not None:
            rm.build_values["estimated_score_delta"].append(float(score_delta))

        emitted = int(
            _extract_step_info_value(
                infos,
                "build_post_joker_emitted_count",
                env_idx,
                done=done,
                default=0,
            )
        )
        for index in range(min(emitted, MAX_DIAGNOSTIC_JOKERS)):
            center = _extract_step_info_value(
                infos,
                f"build_post_joker_{index}_id",
                env_idx,
                done=done,
                default="",
            )
            ratio = _extract_step_info_value(
                infos,
                f"build_post_joker_{index}_marginal_ratio",
                env_idx,
                done=done,
                default=None,
            )
            modeled = _extract_step_info_value(
                infos,
                f"build_post_joker_{index}_modeled_fraction",
                env_idx,
                done=done,
                default=None,
            )
            if center and ratio is not None:
                rm.joker_marginal_ratios[str(center)].append(float(ratio))
            if center and modeled is not None:
                rm.joker_modeled_fractions[str(center)].append(float(modeled))

        for timing in ("pre", "post", "delta"):
            for component in (
                "blind_progress",
                "ante_progress",
                "realized_build_quality",
                "scaling_option_value",
                "readiness",
                "economy",
                "tarot_option_value",
                "planet_option_value",
                "seal_value",
                "joker_search_option",
                "standard_pack_search_option",
                "total",
            ):
                value = _extract_step_info_value(
                    infos,
                    f"potential_{timing}_{component}",
                    env_idx,
                    done=done,
                    default=None,
                )
                if value is not None:
                    rm.potential_values[f"{timing}_{component}"].append(float(value))

    hologram_count = _extract_step_info_value(infos, "hologram_scaling_count", env_idx, done=done, default=None)
    if hologram_count is not None:
        rm.hologram_scaling_counts.append(float(hologram_count))
        rm.hologram_x_mult_deltas.append(
            float(_extract_step_info_value(infos, "hologram_x_mult_delta", env_idx, done=done, default=0.0))
        )
        rm.hologram_build_score_deltas.append(
            float(
                _extract_step_info_value(
                    infos,
                    "hologram_build_score_delta",
                    env_idx,
                    done=done,
                    default=0.0,
                )
            )
        )

    if _extract_step_info_value(infos, "counterfactual_call", env_idx, done=done, default=False):
        rm.counterfactual_calls += 1
        failure = bool(_extract_step_info_value(infos, "counterfactual_failure", env_idx, done=done, default=False))
        rm.counterfactual_failures += int(failure)
        focal = _extract_step_info_value(infos, "counterfactual_focal_joker_id", env_idx, done=done, default="")
        if focal:
            rm.counterfactual_focal_counts[str(focal)] += 1
        if not failure:
            abs_gap = _extract_step_info_value(
                infos,
                "counterfactual_representative_vs_realized_abs_log_ratio_gap",
                env_idx,
                done=done,
                default=None,
            )
            signed_gap = _extract_step_info_value(
                infos,
                "counterfactual_representative_vs_realized_log_ratio_gap",
                env_idx,
                done=done,
                default=None,
            )
            if abs_gap is not None:
                rm.counterfactual_representative_realized_abs_gaps.append(float(abs_gap))
            if signed_gap is not None:
                rm.counterfactual_representative_realized_signed_gaps.append(float(signed_gap))


def _safe_mean(values: list[float]) -> float:
    """Return the mean of a list or NaN when empty."""
    return float(np.mean(values)) if values else float("nan")


def _next_eval_regression_streak(
    *,
    win_rate: float,
    best_win_rate: float | None,
    current_streak: int,
    tolerance: float | None,
) -> int:
    """Advance or clear the consecutive material-eval-regression count."""

    if tolerance is None or best_win_rate is None:
        return 0
    regression = best_win_rate - win_rate
    # Eval rates are ratios of integer game counts, so a nominal boundary such
    # as 0.87 - 0.77 can land just below 0.10 in binary floating point. Treat
    # numerically equal boundaries as material regressions.
    if regression > tolerance or math.isclose(regression, tolerance, rel_tol=1e-9, abs_tol=1e-12):
        return current_streak + 1
    return 0


def _sanitize_tag_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value) or "unknown"


def _action_type_name(action_id: int) -> str:
    """Map a flat action id to a stable TensorBoard-friendly action type name."""
    return decode_action(int(action_id)).action_type.value


def _obs_dicts_to_batch(obs_list: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    """Stack a list of single-env observations into a model batch."""
    return {
        "tokens": torch.tensor(np.stack([obs["tokens"] for obs in obs_list]), dtype=torch.long, device=device),
        "token_types": torch.tensor(
            np.stack([obs["token_types"] for obs in obs_list]),
            dtype=torch.long,
            device=device,
        ),
        "scalars": torch.tensor(np.stack([obs["scalars"] for obs in obs_list]), dtype=torch.float32, device=device),
        "attention_mask": torch.tensor(
            np.stack([obs["attention_mask"] for obs in obs_list]),
            dtype=torch.long,
            device=device,
        ),
        "action_mask": torch.tensor(
            np.stack([obs["action_mask"] for obs in obs_list]),
            dtype=torch.float32,
            device=device,
        ),
        "history_events": torch.tensor(
            np.stack([obs["history_events"] for obs in obs_list]), dtype=torch.long, device=device
        ),
        "history_event_features": torch.tensor(
            np.stack([obs["history_event_features"] for obs in obs_list]), dtype=torch.float32, device=device
        ),
        "history_cards": torch.tensor(
            np.stack([obs["history_cards"] for obs in obs_list]), dtype=torch.long, device=device
        ),
        "history_card_mask": torch.tensor(
            np.stack([obs["history_card_mask"] for obs in obs_list]), dtype=torch.long, device=device
        ),
        "history_jokers": torch.tensor(
            np.stack([obs["history_jokers"] for obs in obs_list]), dtype=torch.long, device=device
        ),
        "history_joker_mask": torch.tensor(
            np.stack([obs["history_joker_mask"] for obs in obs_list]), dtype=torch.long, device=device
        ),
        "history_event_mask": torch.tensor(
            np.stack([obs["history_event_mask"] for obs in obs_list]), dtype=torch.long, device=device
        ),
        "history_round_mask": torch.tensor(
            np.stack([obs["history_round_mask"] for obs in obs_list]), dtype=torch.long, device=device
        ),
        "history_omitted": torch.tensor(
            np.stack([obs["history_omitted"] for obs in obs_list]), dtype=torch.float32, device=device
        ),
    }


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
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_wins: list[bool] = []
    origin_wins: dict[str, list[float]] = {"fresh": [], "archive": []}
    continuation_survival: dict[int, list[float]] = {ante: [] for ante in range(1, 9)}
    episode_stalls: list[bool] = []
    episode_antes: list[int] = []
    episode_tarot_uses: list[int] = []
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
    episode_tracker = EpisodeTracker(config.num_envs, gamma=config.gamma)
    sil_buffer = episode_buffer if config.sil_coeff > 0.0 else None
    # Per-env accumulators (vectorized envs auto-reset, so we track manually)
    env_ep_reward = np.zeros(config.num_envs, dtype=np.float64)
    env_ep_length = np.zeros(config.num_envs, dtype=np.int64)
    # Calibrate the survival prediction shown at the most recent shop leave.
    # These persist across rollout boundaries because an episode often spans
    # several PPO updates.
    # Every shop-leave forecast is queued and resolved at episode end against
    # the realized outcome. The retired single-slot version kept only the LAST
    # shop leave per episode, a sample selected by imminent death: a perfectly
    # calibrated model looks "optimistic" on it. Resolving the full queue makes
    # critic/shop_survival_* and strategy/risk/* unbiased over all shop leaves
    # and feeds tools/fit_risk_calibration.py with usable pairs.
    env_pending_shop_forecasts: list[list[dict]] = [[] for _ in range(config.num_envs)]
    _MAX_PENDING_SHOP_FORECASTS = 128
    risk_forecast_path = Path(config.log_dir) / "risk_forecasts.jsonl"
    risk_forecast_file = None
    if config.risk_forecast_log:
        risk_forecast_path.parent.mkdir(parents=True, exist_ok=True)
        # Long-lived append handle; closed in this function's finally block.
        risk_forecast_file = open(risk_forecast_path, "a", buffering=1)  # noqa: SIM115
    # Per-env start step within the current rollout for the current episode.
    # Reset to 0 at each rollout, advanced past every `done` step so the
    # buffer can retroactively fill categorical outcome labels for completed
    # non-stalled episodes only.
    env_episode_start_step = np.zeros(config.num_envs, dtype=np.int64)

    try:
        while update_count < planned_updates:
            if config.reward_config.objective == "milestone":
                milestone_scale = resolve_milestone_scale(config, total_steps, schedule_total_steps)
                vec_env.call("set_milestone_scale", milestone_scale)
                writer.add_scalar("curriculum/milestone_scale", milestone_scale, update_count + 1)
            buffer = RolloutBuffer(
                num_envs=config.num_envs,
                rollout_length=config.rollout_length,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )
            env_episode_start_step[:] = 0
            rm = _RolloutMetrics()
            # === Collect rollouts (vectorized) ===
            model.eval()
            for step in range(config.rollout_length):
                with torch.no_grad():
                    dist, value_dict = _grammar_distribution(
                        model,
                        _observation_buffer_batch(obs_buf),
                        temperature=_policy_temperature_for_scalars(obs_buf.scalars, config),
                    )
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)
                    values = value_dict["expected_return"]
                    chosen_action_probs = dist.selected_prob(actions)
                    # Cheap macro-concentration telemetry. This is explicitly
                    # not the maximum flat-action probability; computing the
                    # exact hand-mixture mode on every rollout step would add
                    # substantial evaluation-only work to training.
                    max_action_type_probs = dist.action_type_probs.max(dim=-1).values

                actions_np = actions.cpu().numpy()
                log_probs_np = log_probs.cpu().numpy()
                values_np = values.cpu().numpy()
                chosen_action_probs_np = chosen_action_probs.cpu().numpy()
                max_action_type_probs_np = max_action_type_probs.cpu().numpy()
                pre_scalars = obs_buf._np_scalars
                pre_tokens = obs_buf._np_tokens
                rm.clear_probabilities.extend(pre_scalars[:, 11].astype(np.float64).tolist())
                rm.immediate_death_probabilities.extend(pre_scalars[:, 12].astype(np.float64).tolist())
                survival_np = value_dict["ante_survival"].cpu().numpy()
                for env_idx, action_id in enumerate(actions_np):
                    if _action_type_name(int(action_id)) != ActionType.SHOP_LEAVE.value:
                        continue
                    shop_ante = max(round(float(pre_scalars[env_idx, 2])), 1)
                    survival_index = min(shop_ante - 1, survival_np.shape[1] - 1)
                    calibrated_death = 1.0 - float(pre_scalars[env_idx, 11])
                    raw_death = uncalibrate_analytic_death_probability(calibrated_death)
                    pending = env_pending_shop_forecasts[env_idx]
                    if len(pending) < _MAX_PENDING_SHOP_FORECASTS:
                        pending.append(
                            {
                                "shop_ante": int(shop_ante),
                                "shop_blind_index": int(pre_tokens[env_idx, META_START + 3, 0]) // 100,
                                "survival_pred": float(survival_np[env_idx, survival_index]),
                                "analytic_clear_pred": float(pre_scalars[env_idx, 11]),
                                "raw_analytic_clear_pred": 1.0 - raw_death,
                            }
                        )

                # Step all envs at once
                next_obs_dict, rewards, terminated, truncated, infos = vec_env.step(actions_np)
                dones = terminated | truncated
                ppo_terminated, ppo_truncated, stalled_flags = _ppo_terminal_flags(
                    terminated,
                    truncated,
                    infos,
                )
                terminal_rewards_np = np.asarray(
                    [
                        float(
                            _extract_step_info_value(
                                infos,
                                "reward_terminal",
                                env_idx,
                                done=bool(dones[env_idx]),
                                default=0.0,
                            )
                            or 0.0
                        )
                        for env_idx in range(config.num_envs)
                    ],
                    dtype=np.float32,
                )
                reported_rewards_np = np.asarray(
                    [
                        float(
                            _extract_step_info_value(
                                infos,
                                "reward_total",
                                env_idx,
                                done=bool(dones[env_idx]),
                                default=rewards[env_idx],
                            )
                        )
                        for env_idx in range(config.num_envs)
                    ],
                    dtype=np.float32,
                )
                if not np.allclose(reported_rewards_np, rewards, rtol=0.0, atol=1e-6):
                    max_error = float(np.max(np.abs(reported_rewards_np - rewards)))
                    raise RuntimeError(
                        "Environment reward_total does not reconstruct the rollout reward "
                        f"(max absolute error {max_error:.3g})"
                    )
                bootstrap_values_np = np.zeros(config.num_envs, dtype=np.float32)
                if np.any(ppo_truncated) and "final_obs" in infos:
                    final_obs_arr = infos["final_obs"]
                    truncated_indices = [idx for idx in np.where(ppo_truncated)[0] if final_obs_arr[idx] is not None]
                    if truncated_indices:
                        final_obs_batch = _obs_dicts_to_batch([final_obs_arr[idx] for idx in truncated_indices], device)
                        with torch.no_grad():
                            _, truncated_value_dict = _grammar_distribution(
                                model,
                                final_obs_batch,
                                temperature=_policy_temperature_for_scalars(final_obs_batch["scalars"], config),
                            )
                        truncated_vals = truncated_value_dict["expected_return"].cpu().numpy()
                        bootstrap_values_np[np.asarray(truncated_indices, dtype=np.int64)] = truncated_vals

                # Store transition (using pre-step obs from obs_buf)
                buffer.add_batch(
                    step=step,
                    obs=obs_buf.as_numpy_dict(),
                    actions=actions_np,
                    rewards=rewards.astype(np.float32),
                    values=values_np,
                    log_probs=log_probs_np,
                    terminated=ppo_terminated,
                    truncated=ppo_truncated,
                    bootstrap_values=bootstrap_values_np,
                )
                # Same pre-step observations the rollout buffer stores. The
                # tracker clones compact arrays before the shared _ObsBuffer is
                # overwritten and persists episode prefixes across updates.
                episode_tracker.record_step(
                    obs_buf.as_numpy_dict(),
                    actions_np,
                    rewards,
                    terminal_rewards=terminal_rewards_np,
                    behavior_log_probs=log_probs_np,
                    policy_version=update_count,
                )

                # Track per-env episode stats
                env_ep_reward += rewards
                env_ep_length += 1
                rm.step_rewards.extend(rewards.astype(np.float64).tolist())
                rm.chosen_action_probs.extend(chosen_action_probs_np.astype(np.float64).tolist())
                rm.max_action_type_probs.extend(max_action_type_probs_np.astype(np.float64).tolist())
                rm.done_flags.extend(dones.astype(np.float64).tolist())
                rm.terminated_flags.extend(terminated.astype(np.float64).tolist())
                rm.truncated_flags.extend(truncated.astype(np.float64).tolist())
                for action_id in actions_np:
                    rm.action_type_counts[_action_type_name(int(action_id))] += 1
                for env_idx in range(config.num_envs):
                    step_done = bool(dones[env_idx])
                    pre_sub_phase = _extract_step_info_value(
                        infos,
                        "pre_sub_phase",
                        env_idx,
                        done=step_done,
                        default="",
                    )
                    action_type_name = _action_type_name(int(actions_np[env_idx]))

                    in_choose_action = pre_sub_phase == "choose_action"
                    rm.pre_choose_action_flags.append(float(in_choose_action))
                    if action_type_name == ActionType.PLAY_SUBSET.value:
                        rm.play_subset_count += 1
                    elif action_type_name == ActionType.DISCARD_SUBSET.value:
                        rm.discard_subset_count += 1

                    rm.progress_flags.append(
                        float(
                            bool(
                                _extract_step_info_value(
                                    infos,
                                    "progress_made",
                                    env_idx,
                                    done=step_done,
                                    default=False,
                                )
                            )
                        )
                    )
                    rm.steps_since_progress.append(
                        float(
                            _extract_step_info_value(infos, "steps_since_progress", env_idx, done=step_done, default=0)
                        )
                    )
                    if action_type_name == ActionType.PLAY_SUBSET.value:
                        objective = _extract_step_info_value(
                            infos, "joker_order_objective", env_idx, done=step_done, default=""
                        )
                        if objective:
                            rm.joker_order_money_flags.append(float(str(objective) == "money"))
                            rm.joker_order_clear_flags.append(
                                float(
                                    bool(
                                        _extract_step_info_value(
                                            infos, "joker_order_clears", env_idx, done=step_done, default=False
                                        )
                                    )
                                )
                            )
                            rm.joker_order_dollars_gained.append(
                                float(
                                    _extract_step_info_value(
                                        infos, "joker_order_dollars_gained", env_idx, done=step_done, default=0.0
                                    )
                                    or 0.0
                                )
                            )
                            rm.joker_order_chips_forgone.append(
                                float(
                                    _extract_step_info_value(
                                        infos, "joker_order_chips_forgone", env_idx, done=step_done, default=0.0
                                    )
                                    or 0.0
                                )
                            )
                    for component_name in REWARD_INFO_KEYS:
                        component_value = _extract_step_info_value(
                            infos,
                            component_name,
                            env_idx,
                            done=step_done,
                            default=None,
                        )
                        if component_value is not None:
                            rm.reward_component_values[component_name].append(float(component_value))

                    _record_action_diagnostics(rm, infos, env_idx, done=step_done)

                # Handle completed episodes (vectorized envs auto-reset)
                for i in np.where(dones)[0]:
                    ep_reward = float(env_ep_reward[i])
                    ep_length = int(env_ep_length[i])
                    ep_won = bool(_extract_step_info_value(infos, "won", i, done=True, default=False))
                    from_archive = bool(_extract_step_info_value(infos, "archive_start", i, done=True, default=False))
                    start_ante = int(_extract_step_info_value(infos, "start_ante", i, done=True, default=1))
                    origin_wins["archive" if from_archive else "fresh"].append(float(ep_won))
                    ep_stalled = bool(stalled_flags[i])
                    ep_ante = int(_extract_step_info_value(infos, "ante", i, done=True, default=1))
                    if not ep_stalled:
                        for ante in range(start_ante, min(ep_ante, 8) + 1):
                            continuation_survival[ante].append(float(ep_won or ep_ante > ante))
                    ep_terminal_blind = str(
                        _extract_step_info_value(
                            infos,
                            "blind_on_deck",
                            i,
                            done=True,
                            default="",
                        )
                        or ""
                    ).lower()
                    ep_tarot_uses = int(
                        _extract_step_info_value(
                            infos,
                            "tarot_usage_total",
                            i,
                            done=True,
                            default=0,
                        )
                        or 0
                    )
                    episode_rewards.append(ep_reward)
                    episode_lengths.append(ep_length)
                    episode_wins.append(ep_won)
                    episode_stalls.append(ep_stalled)
                    episode_antes.append(ep_ante)
                    episode_tarot_uses.append(ep_tarot_uses)
                    rm.completed_episode_rewards.append(ep_reward)
                    rm.completed_episode_lengths.append(ep_length)
                    rm.completed_episode_wins.append(float(ep_won))
                    rm.completed_episode_stalls.append(float(ep_stalled))
                    rm.completed_episode_antes.append(ep_ante)
                    rm.completed_episode_tarot_uses.append(ep_tarot_uses)
                    if not ep_won and not ep_stalled:
                        rm.terminal_loss_antes.append(ep_ante)
                        terminal_dollars = float(
                            _extract_step_info_value(infos, "dollars", i, done=True, default=0.0) or 0.0
                        )
                        terminal_full_weak = bool(
                            _extract_step_info_value(
                                infos,
                                "joker_full",
                                i,
                                done=True,
                                default=False,
                            )
                            and _extract_step_info_value(
                                infos,
                                "weak_confident_joker",
                                i,
                                done=True,
                                default=False,
                            )
                        )
                        rm.terminal_loss_dollars.append(terminal_dollars)
                        rm.terminal_loss_cash_ge_10.append(float(terminal_dollars >= 10.0))
                        rm.terminal_loss_joker_full_weak.append(float(terminal_full_weak))
                        end_blind = ep_terminal_blind
                        rm.terminal_loss_blind_counts[end_blind] += 1
                        if ep_ante == 1:
                            rm.ante1_death_blind_counts[end_blind] += 1
                        if end_blind == "boss":
                            boss_key = str(_extract_step_info_value(infos, "boss_key", i, done=True, default="") or "")
                            if boss_key:
                                rm.terminal_boss_loss_counts[boss_key] += 1
                        blind_target = float(
                            _extract_step_info_value(infos, "blind_target", i, done=True, default=0.0) or 0.0
                        )
                        round_score = float(
                            _extract_step_info_value(infos, "round_score", i, done=True, default=0.0) or 0.0
                        )
                        if blind_target > 0.0:
                            rm.terminal_loss_score_ratios.append(round_score / blind_target)
                        last_play_top1 = _extract_step_info_value(
                            infos,
                            "hand_play_legal_top1",
                            i,
                            done=True,
                            default=_extract_step_info_value(
                                infos,
                                "hand_play_top1",
                                i,
                                done=True,
                                default=None,
                            ),
                        )
                        if last_play_top1 is not None:
                            rm.terminal_loss_last_play_top1.append(float(bool(last_play_top1)))
                        last_play_value_ratio = _extract_step_info_value(
                            infos,
                            "hand_play_legal_candidate_value_ratio",
                            i,
                            done=True,
                            default=_extract_step_info_value(
                                infos,
                                "hand_play_candidate_value_ratio",
                                i,
                                done=True,
                                default=None,
                            ),
                        )
                        if last_play_value_ratio is not None:
                            rm.terminal_loss_last_play_value_ratios.append(float(last_play_value_ratio))
                    if env_pending_shop_forecasts[i]:
                        if not ep_stalled:
                            terminal_blind = str(
                                _extract_step_info_value(
                                    infos,
                                    "blind_on_deck",
                                    i,
                                    done=True,
                                    default="",
                                )
                                or ""
                            )
                            for forecast in env_pending_shop_forecasts[i]:
                                shop_ante = int(forecast["shop_ante"])
                                ante_outcome = float(ep_won or ep_ante > shop_ante)
                                prediction = float(forecast["survival_pred"])
                                rm.shop_survival_predictions.append(prediction)
                                rm.shop_survival_outcomes.append(ante_outcome)
                                rm.shop_survival_briers.append((prediction - ante_outcome) ** 2)
                                blind_outcome = _next_blind_clear_outcome(
                                    shop_ante=shop_ante,
                                    shop_blind_index=int(forecast["shop_blind_index"]),
                                    final_ante=ep_ante,
                                    won=ep_won,
                                    terminal_blind=terminal_blind,
                                )
                                predicted_death = 1.0 - float(forecast["analytic_clear_pred"])
                                death_outcome = 1.0 - blind_outcome
                                rm.risk_shop_death_predictions.append(predicted_death)
                                rm.risk_shop_death_outcomes.append(death_outcome)
                                rm.risk_shop_death_briers.append((predicted_death - death_outcome) ** 2)
                                raw_predicted_death = 1.0 - float(forecast["raw_analytic_clear_pred"])
                                rm.risk_shop_raw_death_predictions.append(raw_predicted_death)
                                rm.risk_shop_raw_death_briers.append((raw_predicted_death - death_outcome) ** 2)
                                if shop_ante == 1 and death_outcome > 0.5:
                                    rm.risk_ante1_false_safe_deaths.append(float(predicted_death <= 0.35))
                                if risk_forecast_file is not None:
                                    risk_forecast_file.write(
                                        json.dumps(
                                            {
                                                "update": int(update_count),
                                                "shop_ante": shop_ante,
                                                "shop_blind_index": int(forecast["shop_blind_index"]),
                                                "survival_pred": prediction,
                                                "calibrated_death_pred": predicted_death,
                                                "raw_death_pred": raw_predicted_death,
                                                "next_blind_death": death_outcome,
                                                "ante_survived": ante_outcome,
                                                "won": bool(ep_won),
                                                "final_ante": int(ep_ante),
                                            }
                                        )
                                        + "\n"
                                    )
                        env_pending_shop_forecasts[i].clear()
                    # Insert wins and ordinary completed losses for terminal
                    # critic replay. Infrastructure/no-progress stalls are
                    # deliberately excluded because their final-Ante outcome
                    # is censored rather than a policy loss.
                    episode_tracker.finish_episode(
                        int(i),
                        won=ep_won,
                        stalled=ep_stalled,
                        final_ante=ep_ante,
                        win_ante=effective_win_ante,
                        terminal_blind=ep_terminal_blind,
                        buffer=episode_buffer,
                    )
                    # Stalled episodes are censored, so their outcome mask stays
                    # zero. Complete wins and losses get one categorical label.
                    if not ep_stalled:
                        buffer.set_episode_outcome(
                            env_idx=int(i),
                            start_step=int(env_episode_start_step[i]),
                            end_step=step,
                            won=ep_won,
                            final_ante=ep_ante,
                        )
                    env_episode_start_step[i] = step + 1
                    env_ep_reward[i] = 0.0
                    env_ep_length[i] = 0

                # Update obs buffer with new observations
                obs_buf.update(next_obs_dict)
                total_steps += config.num_envs

            # Bootstrap values for GAE
            with torch.no_grad():
                _, value_dict = _grammar_distribution(
                    model,
                    _observation_buffer_batch(obs_buf),
                    temperature=_policy_temperature_for_scalars(obs_buf.scalars, config),
                )
                last_values = value_dict["expected_return"].cpu().numpy()

            buffer.compute_returns_and_advantages(last_values=last_values)
            buffer.normalize_advantages(clip_sigma=config.advantage_clip_sigma)

            # Explained variance of the composed critic on this rollout:
            # 1 - Var(returns - values) / Var(returns).
            flat_returns = buffer._flat_returns
            explained_variance = float("nan")
            if len(flat_returns) > 1:
                var_returns = float(np.var(flat_returns))
                if var_returns > 1e-9:
                    explained_variance = 1.0 - float(np.var(flat_returns - buffer._flat_values)) / var_returns
            writer.add_scalar("ppo/explained_variance", explained_variance, update_count + 1)
            # Which rollout rows the main-loop outcome NLL can actually train
            # on, and how much shallower they are than the rows it drops.
            for label_key, label_value in buffer.outcome_label_diagnostics().items():
                writer.add_scalar(
                    f"critic/rollout_labels/{label_key}",
                    label_value,
                    update_count + 1,
                )
            rollout_step = update_count + 1
            writer.add_scalar("rollout/step_reward_mean", float(np.mean(rm.step_rewards)), rollout_step)
            writer.add_scalar("rollout/reward_mean", float(np.mean(rm.step_rewards)), rollout_step)
            writer.add_scalar("rollout/chosen_action_prob_mean", float(np.mean(rm.chosen_action_probs)), rollout_step)
            writer.add_scalar(
                "rollout/max_action_type_prob_mean",
                float(np.mean(rm.max_action_type_probs)),
                rollout_step,
            )
            writer.add_scalar("rollout/done_rate", float(np.mean(rm.done_flags)), rollout_step)
            writer.add_scalar("rollout/progress_rate", float(np.mean(rm.progress_flags)), rollout_step)
            writer.add_scalar("rollout/completed_episodes", float(np.sum(rm.done_flags)), rollout_step)
            _write_rollout_episode_metrics(writer, rm, rollout_step)
            _write_action_behavior_metrics(writer, rm, rollout_step)
            _write_terminal_loss_metrics(writer, rm, rollout_step, win_ante=effective_win_ante)
            _write_ante1_metrics(writer, rm, rollout_step)
            if rm.clear_probabilities:
                writer.add_scalar(
                    "strategy/risk/clear_probability_mean",
                    float(np.mean(rm.clear_probabilities)),
                    rollout_step,
                )
                writer.add_scalar(
                    "strategy/risk/immediate_death_probability_mean",
                    float(np.mean(rm.immediate_death_probabilities)),
                    rollout_step,
                )
            if rm.shop_leave_flags:
                writer.add_scalar(
                    "shop/unsafe_leave_fraction",
                    float(np.mean(rm.shop_unsafe_leave_flags)),
                    rollout_step,
                )
                writer.add_scalar(
                    "shop/unsafe_can_reroll_fraction",
                    float(np.mean(rm.shop_unsafe_can_reroll_flags)),
                    rollout_step,
                )
                writer.add_scalar(
                    "shop/missed_confident_upgrade_fraction",
                    float(np.mean(rm.shop_missed_upgrade_flags)),
                    rollout_step,
                )
                writer.add_scalar(
                    "shop/full_weak_leave_fraction",
                    float(np.mean(rm.shop_leave_full_weak_flags)),
                    rollout_step,
                )
            if rm.shop_best_upgrade_deltas:
                writer.add_scalar(
                    "shop/best_confident_upgrade_delta_mean",
                    float(np.mean(rm.shop_best_upgrade_deltas)),
                    rollout_step,
                )
            if rm.shop_survival_briers:
                writer.add_scalar(
                    "critic/shop_survival_brier",
                    float(np.mean(rm.shop_survival_briers)),
                    rollout_step,
                )
                writer.add_scalar(
                    "critic/shop_survival_prediction_mean",
                    float(np.mean(rm.shop_survival_predictions)),
                    rollout_step,
                )
                writer.add_scalar(
                    "critic/shop_survival_outcome_mean",
                    float(np.mean(rm.shop_survival_outcomes)),
                    rollout_step,
                )
            _write_risk_calibration_metrics(writer, rm, rollout_step)
            if episode_rewards:
                writer.add_scalar(
                    "recent_100/episode_reward_mean", float(np.mean(episode_rewards[-100:])), rollout_step
                )
                writer.add_scalar(
                    "recent_100/episode_length_mean", float(np.mean(episode_lengths[-100:])), rollout_step
                )
                writer.add_scalar("recent_100/win_rate", float(np.mean(episode_wins[-100:])), rollout_step)
                writer.add_scalar("recent_100/stall_rate", float(np.mean(episode_stalls[-100:])), rollout_step)
                writer.add_scalar("recent_100/final_ante_mean", float(np.mean(episode_antes[-100:])), rollout_step)
                writer.add_scalar(
                    "recent_100/tarot_uses_per_completed_episode_mean",
                    float(np.mean(episode_tarot_uses[-100:])),
                    rollout_step,
                )
            hand_total = sum(rm.hand_chosen_counts.values())
            if hand_total:
                for hand_name, count in rm.hand_chosen_counts.items():
                    tag_name = str(hand_name).lower().replace(" ", "_")
                    writer.add_scalar(f"rollout/hands_played/{tag_name}", count / hand_total, rollout_step)
            steps_per_thousand = max(len(rm.step_rewards) / 1000.0, 1e-9)
            writer.add_scalar(
                "shop/joker_offers_per_1k_steps",
                sum(rm.shop_offered_joker_counts.values()) / steps_per_thousand,
                rollout_step,
            )
            writer.add_scalar(
                "shop/joker_buys_per_1k_steps",
                sum(rm.shop_bought_joker_counts.values()) / steps_per_thousand,
                rollout_step,
            )
            writer.add_scalar(
                "shop/joker_sells_per_1k_steps",
                sum(rm.shop_sold_joker_counts.values()) / steps_per_thousand,
                rollout_step,
            )
            completed_episodes = len(rm.completed_episode_rewards)
            if completed_episodes:
                writer.add_scalar(
                    "joker/replacements_per_episode",
                    rm.joker_replacement_events / completed_episodes,
                    rollout_step,
                )
                writer.add_scalar(
                    "joker/churn_per_episode",
                    rm.joker_churn_count / completed_episodes,
                    rollout_step,
                )
            for consumable_set in ("Planet", "Tarot"):
                tag_name = consumable_set.lower()
                writer.add_scalar(
                    f"rollout/shop_buys/{tag_name}_per_1k_steps",
                    rm.consumable_buy_set_counts[consumable_set] / steps_per_thousand,
                    rollout_step,
                )
                writer.add_scalar(
                    f"rollout/pack_claims/{tag_name}_per_1k_steps",
                    rm.consumable_claim_set_counts[consumable_set] / steps_per_thousand,
                    rollout_step,
                )
                writer.add_scalar(
                    f"rollout/uses/{tag_name}_per_1k_steps",
                    rm.consumable_use_set_counts[consumable_set] / steps_per_thousand,
                    rollout_step,
                )
            for seal in ("Blue", "Purple", "Gold", "Red"):
                writer.add_scalar(
                    f"strategy/seals/claims/{seal.lower()}_per_1k_steps",
                    rm.pack_claim_seal_counts[seal] / steps_per_thousand,
                    rollout_step,
                )
            writer.add_scalar(
                "strategy/seals/purple_tarots_generated_per_1k_steps",
                rm.purple_seal_tarots_generated / steps_per_thousand,
                rollout_step,
            )
            writer.add_scalar(
                "strategy/seals/blue_planets_generated_per_1k_steps",
                rm.blue_seal_planets_generated / steps_per_thousand,
                rollout_step,
            )
            _write_consumable_strategy_metrics(writer, rm, rollout_step)
            plan_total = sum(rm.hand_plan_type_counts.values())
            if plan_total:
                for hand_name, count in rm.hand_plan_type_counts.items():
                    tag_name = hand_name.lower().replace(" ", "_")
                    writer.add_scalar(
                        f"strategy/hand_plan/{tag_name}_share",
                        count / plan_total,
                        rollout_step,
                    )
            if rm.hand_plan_reliability:
                writer.add_scalar(
                    "strategy/hand_plan/reliability_mean",
                    float(np.mean(rm.hand_plan_reliability)),
                    rollout_step,
                )
            if rm.hand_plan_readiness:
                writer.add_scalar(
                    "strategy/hand_plan/readiness_mean",
                    float(np.mean(rm.hand_plan_readiness)),
                    rollout_step,
                )
            for component in (
                "economy",
                "tarot_option_value",
                "planet_option_value",
                "seal_value",
                "joker_search_option",
                "standard_pack_search_option",
            ):
                values = rm.potential_values.get(f"post_{component}")
                if values:
                    writer.add_scalar(
                        f"strategy/potential/{component}_mean",
                        float(np.mean(values)),
                        rollout_step,
                    )
            for component_name, values in rm.reward_component_values.items():
                if values:
                    writer.add_scalar(
                        f"rollout/reward_components/{component_name}",
                        float(np.mean(values)),
                        rollout_step,
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

            if episode_rewards:
                recent = episode_rewards[-100:]
                recent_wins = episode_wins[-100:]
                recent_stalls = episode_stalls[-100:]
                recent_reward_mean = float(np.mean(recent))
                recent_length_mean = float(np.mean(episode_lengths[-100:]))
                recent_win_rate = float(np.mean(recent_wins))
                recent_stall_rate = float(np.mean(recent_stalls))
            else:
                recent_reward_mean = float("nan")
                recent_length_mean = float("nan")
                recent_win_rate = float("nan")
                recent_stall_rate = float("nan")

            should_checkpoint = update_count % config.checkpoint_interval == 0 or update_count == planned_updates
            for origin, wins in origin_wins.items():
                if wins:
                    writer.add_scalar(f"curriculum/{origin}_win_rate", float(np.mean(wins[-100:])), update_count)
                    writer.add_scalar(f"curriculum/{origin}_episodes", len(wins), update_count)
            for ante, outcomes in continuation_survival.items():
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
                    extra={
                        "best_eval_win_rate": best_eval_win_rate,
                        "best_eval_update": best_eval_update,
                        "schedule_total_steps": schedule_total_steps,
                    },
                )
                # Always mirror the latest checkpoint so resume/eval always has
                # a stable path without guessing the highest update number.
                _mirror_latest_checkpoint(checkpoint_path, save_path)
                logger.info("Saved checkpoint: %s", checkpoint_path)

            eval_win_rate: float | None = None
            should_eval = update_count % config.eval_interval == 0 or update_count == planned_updates
            if should_eval:
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
                is_best = best_eval_win_rate is None or win_rate > best_eval_win_rate
                if is_best:
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
                        "New best eval win_rate=%.3f at update %d -> %s",
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


class _ObsBuffer:
    """Pre-allocated GPU/device tensors for batched observations.

    Avoids re-creating tensors every step by writing into existing storage.
    """

    def __init__(self, num_envs: int, device: torch.device) -> None:
        self.num_envs = num_envs
        self.device = device
        self.tokens = torch.zeros(num_envs, MAX_SEQ_LEN, TOKEN_DIM, dtype=torch.long, device=device)
        self.token_types = torch.zeros(num_envs, MAX_SEQ_LEN, dtype=torch.long, device=device)
        self.scalars = torch.zeros(num_envs, SCALAR_DIM, dtype=torch.float32, device=device)
        self.attention_mask = torch.zeros(num_envs, MAX_SEQ_LEN, dtype=torch.long, device=device)
        self.action_mask = torch.zeros(num_envs, NUM_ACTIONS, dtype=torch.float32, device=device)
        self.history_events = torch.zeros(
            num_envs, HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_EVENT_DIM, dtype=torch.long, device=device
        )
        self.history_event_features = torch.zeros(
            num_envs,
            HISTORY_ROUNDS,
            HISTORY_MAX_PLAYS,
            HISTORY_FEATURE_DIM,
            dtype=torch.float32,
            device=device,
        )
        self.history_cards = torch.zeros(
            num_envs,
            HISTORY_ROUNDS,
            HISTORY_MAX_PLAYS,
            HISTORY_MAX_CARDS,
            TOKEN_DIM,
            dtype=torch.long,
            device=device,
        )
        self.history_card_mask = torch.zeros(
            num_envs, HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_CARDS, dtype=torch.long, device=device
        )
        self.history_jokers = torch.zeros(
            num_envs, HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_JOKERS, dtype=torch.long, device=device
        )
        self.history_joker_mask = torch.zeros_like(self.history_jokers)
        self.history_event_mask = torch.zeros(
            num_envs, HISTORY_ROUNDS, HISTORY_MAX_PLAYS, dtype=torch.long, device=device
        )
        self.history_round_mask = torch.zeros(num_envs, HISTORY_ROUNDS, dtype=torch.long, device=device)
        self.history_omitted = torch.zeros(
            num_envs, HISTORY_ROUNDS, HISTORY_OMITTED_DIM, dtype=torch.float32, device=device
        )

        # Numpy views for writing from env output (CPU side)
        self._np_tokens = np.zeros((num_envs, MAX_SEQ_LEN, TOKEN_DIM), dtype=np.int64)
        self._np_token_types = np.zeros((num_envs, MAX_SEQ_LEN), dtype=np.int64)
        self._np_scalars = np.zeros((num_envs, SCALAR_DIM), dtype=np.float32)
        self._np_attention_mask = np.zeros((num_envs, MAX_SEQ_LEN), dtype=np.int64)
        self._np_action_mask = np.zeros((num_envs, NUM_ACTIONS), dtype=np.float32)
        self._np_history_events = np.zeros(
            (num_envs, HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_EVENT_DIM), dtype=np.int64
        )
        self._np_history_event_features = np.zeros(
            (num_envs, HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_FEATURE_DIM), dtype=np.float32
        )
        self._np_history_cards = np.zeros(
            (num_envs, HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_CARDS, TOKEN_DIM), dtype=np.int64
        )
        self._np_history_card_mask = np.zeros(
            (num_envs, HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_CARDS), dtype=np.int64
        )
        self._np_history_jokers = np.zeros(
            (num_envs, HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_JOKERS), dtype=np.int64
        )
        self._np_history_joker_mask = np.zeros_like(self._np_history_jokers)
        self._np_history_event_mask = np.zeros((num_envs, HISTORY_ROUNDS, HISTORY_MAX_PLAYS), dtype=np.int64)
        self._np_history_round_mask = np.zeros((num_envs, HISTORY_ROUNDS), dtype=np.int64)
        self._np_history_omitted = np.zeros((num_envs, HISTORY_ROUNDS, HISTORY_OMITTED_DIM), dtype=np.float32)

    def update(self, obs_dict: dict) -> None:
        """Copy vectorized env output into pre-allocated tensors."""
        np.copyto(self._np_tokens, obs_dict["tokens"])
        np.copyto(self._np_token_types, obs_dict["token_types"])
        np.copyto(self._np_scalars, obs_dict["scalars"])
        np.copyto(self._np_attention_mask, obs_dict["attention_mask"])
        np.copyto(self._np_action_mask, obs_dict["action_mask"])
        for target, key in (
            (self._np_history_events, "history_events"),
            (self._np_history_event_features, "history_event_features"),
            (self._np_history_cards, "history_cards"),
            (self._np_history_card_mask, "history_card_mask"),
            (self._np_history_jokers, "history_jokers"),
            (self._np_history_joker_mask, "history_joker_mask"),
            (self._np_history_event_mask, "history_event_mask"),
            (self._np_history_round_mask, "history_round_mask"),
            (self._np_history_omitted, "history_omitted"),
        ):
            np.copyto(target, obs_dict[key])

        self.tokens.copy_(torch.from_numpy(self._np_tokens))
        self.token_types.copy_(torch.from_numpy(self._np_token_types))
        self.scalars.copy_(torch.from_numpy(self._np_scalars))
        self.attention_mask.copy_(torch.from_numpy(self._np_attention_mask))
        self.action_mask.copy_(torch.from_numpy(self._np_action_mask))
        self.history_events.copy_(torch.from_numpy(self._np_history_events))
        self.history_event_features.copy_(torch.from_numpy(self._np_history_event_features))
        self.history_cards.copy_(torch.from_numpy(self._np_history_cards))
        self.history_card_mask.copy_(torch.from_numpy(self._np_history_card_mask))
        self.history_jokers.copy_(torch.from_numpy(self._np_history_jokers))
        self.history_joker_mask.copy_(torch.from_numpy(self._np_history_joker_mask))
        self.history_event_mask.copy_(torch.from_numpy(self._np_history_event_mask))
        self.history_round_mask.copy_(torch.from_numpy(self._np_history_round_mask))
        self.history_omitted.copy_(torch.from_numpy(self._np_history_omitted))

    def as_numpy_dict(self) -> dict:
        """Return current numpy arrays (for storing in rollout buffer)."""
        return {
            "tokens": self._np_tokens,
            "token_types": self._np_token_types,
            "scalars": self._np_scalars,
            "attention_mask": self._np_attention_mask,
            "action_mask": self._np_action_mask,
            "history_events": self._np_history_events,
            "history_event_features": self._np_history_event_features,
            "history_cards": self._np_history_cards,
            "history_card_mask": self._np_history_card_mask,
            "history_jokers": self._np_history_jokers,
            "history_joker_mask": self._np_history_joker_mask,
            "history_event_mask": self._np_history_event_mask,
            "history_round_mask": self._np_history_round_mask,
            "history_omitted": self._np_history_omitted,
        }


def evaluate_model(
    model: BalatroAgent,
    data: GameData,
    vocab: Vocab,
    num_games: int,
    device: torch.device,
    max_no_progress_steps: int = 256,
    win_ante: int | None = None,
    temperature: float = 1.0,
    stake: int = 1,
    seeds: list[int] | None = None,
    eval_batch_size: int = 32,
) -> float:
    """Evaluate model win rate with greedy action selection over num_games.

    When ``seeds`` is provided it is used verbatim (and truncated to
    ``num_games``); otherwise the historical ``10000 + game_idx`` seeds are
    generated. Passing the versioned :data:`pylatro_agent.eval.EVAL_SEEDS_V1`
    list makes every checkpoint's eval reproducible and pairable across runs.

    Games are played in lockstep batches of ``eval_batch_size`` so one forward
    pass serves many games (see :func:`run_seed_evaluation`). Per-seed results
    are unchanged: the engine is deterministic given its seed and greedy action
    selection does not depend on what the other games in the batch are doing.
    """
    seed_list = list(seeds[:num_games]) if seeds is not None else [10000 + i for i in range(num_games)]
    outcomes = run_seed_evaluation(
        model,
        data,
        vocab,
        seed_list,
        device,
        max_no_progress_steps=max_no_progress_steps,
        win_ante=win_ante,
        temperature=temperature,
        stake=stake,
        greedy=True,
        batch_size=eval_batch_size,
    )
    wins = sum(1 for outcome in outcomes if outcome["won"])
    return wins / max(len(outcomes), 1)


def run_seed_evaluation(
    model: BalatroAgent,
    data: GameData,
    vocab: Vocab,
    seeds: list[int],
    device: torch.device,
    *,
    max_no_progress_steps: int = 256,
    win_ante: int | None = None,
    temperature: float = 1.0,
    stake: int = 1,
    greedy: bool = True,
    batch_size: int = 32,
) -> list[dict]:
    """Play every seed and return its outcome, batching the policy forward pass.

    The retired implementation played games strictly serially with batch-size-1
    forwards. Eval then cost more wall clock than the training it was measuring
    (measured on the Ante-5 runs: 60-95 minutes per 400-game eval against ~54
    minutes of training per 25-update interval). Here up to ``batch_size`` games
    advance in lockstep and share one forward pass, so the same GPU/CPU work
    serves many games at once. Env stepping stays in-process and serial, so the
    speedup tracks the forward-pass share of per-step cost.

    Determinism: each seed constructs its own env and greedy selection is a
    per-row argmax, so per-seed outcomes do not depend on batch composition or
    on which slot a seed lands in. Sampled evaluation (``greedy=False``) draws
    from the global torch RNG and is reproducible only under a fixed seed and a
    fixed ``batch_size``.

    Memory safety: runs under ``torch.inference_mode`` (no autograd graph) and
    drains the MPS cache periodically so an eval-heavy run cannot jetsam idle
    AsyncVectorEnv workers.
    """
    model.eval()
    if not seeds:
        return []
    slot_count = max(1, min(int(batch_size), len(seeds)))

    results_by_seed: dict[int, dict] = {}
    pending = list(seeds)
    # Each live slot is (seed, env, obs). Slots advance in lockstep; a finished
    # slot immediately picks up the next pending seed so the batch stays full.
    slots: list[tuple[int, BalatroEnv, dict]] = []

    def _start(seed: int) -> tuple[int, BalatroEnv, dict]:
        env = BalatroEnv(
            seed=seed,
            data=data,
            vocab=vocab,
            stake=stake,
            max_steps=max_no_progress_steps,
            win_ante=win_ante,
            # Eval never reads teacher labels; the heuristic teacher would
            # otherwise run twice per step of every eval game.
            enable_teacher=False,
        )
        obs, _ = env.reset(seed=seed)
        return seed, env, obs

    with torch.inference_mode():
        while pending and len(slots) < slot_count:
            slots.append(_start(pending.pop(0)))

        steps_since_drain = 0
        while slots:
            batch = _obs_dicts_to_batch([obs for _seed, _env, obs in slots], device)
            dist, _value = _grammar_distribution(model, batch, temperature=temperature)
            actions = (dist.mode() if greedy else dist.sample()).cpu().numpy()
            # Drop per-step inference tensors immediately; otherwise they sit on
            # the MPS allocator until the whole eval finishes.
            del batch, dist

            next_slots: list[tuple[int, BalatroEnv, dict]] = []
            for slot_index, (seed, env, _obs) in enumerate(slots):
                obs, _reward, terminated, truncated, info = env.step(int(actions[slot_index]))
                if terminated or truncated:
                    results_by_seed[seed] = {
                        "seed": seed,
                        "won": bool(info.get("won", False)),
                        "max_ante": int(info.get("ante", 1) or 1),
                        "round_score": int(info.get("round_score", 0) or 0),
                        "stalled": bool(info.get("stalled", False)),
                    }
                    if pending:
                        next_slots.append(_start(pending.pop(0)))
                else:
                    next_slots.append((seed, env, obs))
            slots = next_slots

            steps_since_drain += 1
            if device.type == "mps" and steps_since_drain >= 200:
                torch.mps.empty_cache()
                steps_since_drain = 0

    # Preserve caller seed order regardless of completion order.
    return [results_by_seed[seed] for seed in seeds if seed in results_by_seed]


def _single_obs_to_batch(obs: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.tensor(obs["tokens"], dtype=torch.long, device=device).unsqueeze(0),
        "token_types": torch.tensor(obs["token_types"], dtype=torch.long, device=device).unsqueeze(0),
        "scalars": torch.tensor(obs["scalars"], dtype=torch.float32, device=device).unsqueeze(0),
        "attention_mask": torch.tensor(obs["attention_mask"], dtype=torch.long, device=device).unsqueeze(0),
        "action_mask": torch.tensor(obs["action_mask"], dtype=torch.float32, device=device).unsqueeze(0),
        "history_events": torch.tensor(obs["history_events"], dtype=torch.long, device=device).unsqueeze(0),
        "history_event_features": torch.tensor(
            obs["history_event_features"], dtype=torch.float32, device=device
        ).unsqueeze(0),
        "history_cards": torch.tensor(obs["history_cards"], dtype=torch.long, device=device).unsqueeze(0),
        "history_card_mask": torch.tensor(obs["history_card_mask"], dtype=torch.long, device=device).unsqueeze(0),
        "history_jokers": torch.tensor(obs["history_jokers"], dtype=torch.long, device=device).unsqueeze(0),
        "history_joker_mask": torch.tensor(obs["history_joker_mask"], dtype=torch.long, device=device).unsqueeze(0),
        "history_event_mask": torch.tensor(obs["history_event_mask"], dtype=torch.long, device=device).unsqueeze(0),
        "history_round_mask": torch.tensor(obs["history_round_mask"], dtype=torch.long, device=device).unsqueeze(0),
        "history_omitted": torch.tensor(obs["history_omitted"], dtype=torch.float32, device=device).unsqueeze(0),
    }
