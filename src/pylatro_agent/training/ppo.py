"""PPO training loop with vectorized environments."""

import copy
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.vector.vector_env import AutoresetMode
from torch.optim import Adam

from pylatro import GameData, load_game_data

from ..action import ActionType, decode_action
from ..agent import AgentConfig, BalatroAgent
from ..constants import (
    HISTORY_EVENT_DIM,
    HISTORY_FEATURE_DIM,
    HISTORY_MAX_CARDS,
    HISTORY_MAX_JOKERS,
    HISTORY_MAX_PLAYS,
    HISTORY_OMITTED_DIM,
    HISTORY_ROUNDS,
    MAX_SEQ_LEN,
    NUM_ACTIONS,
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
from ..survival import compute_ante_survival_targets
from ..value_head import hl_gauss_projection
from ..vocab import Vocab, build_vocab
from .rollout_buffer import RolloutBuffer
from .sil import (
    EpisodeReplayBuffer,
    SILEpisodeTracker,
    sil_percentile_gate,
)

logger = logging.getLogger(__name__)
_MISSING = object()
_HISTORY_SIGNATURE_CACHE: dict[type, bool] = {}


class RunningMeanStd:
    """Welford's online algorithm for tracking mean/variance of a stream."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var + 1e-8))

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


_ACTION_TYPES = tuple(ActionType)
_ACTION_TYPE_TO_INDEX = {action_type: idx for idx, action_type in enumerate(_ACTION_TYPES)}
_ACTION_ID_TO_TYPE_INDEX = torch.tensor(
    [_ACTION_TYPE_TO_INDEX[decode_action(action_id).action_type] for action_id in range(NUM_ACTIONS)],
    dtype=torch.long,
)


def _load_state_dict_into_model(model: nn.Module, state_dict: dict, checkpoint_path: str) -> None:
    """Load a raw ``state_dict`` into ``model``, handling DataParallel prefix
    mismatch and minor head shape drift. Shared by weights-only init and resume.
    """
    has_module_prefix = any(k.startswith("module.") for k in state_dict)
    is_wrapped = isinstance(model, nn.DataParallel)

    if has_module_prefix and not is_wrapped:
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    elif not has_module_prefix and is_wrapped:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}

    model_state = model.state_dict()
    compatible = {
        key: value for key, value in state_dict.items() if key in model_state and model_state[key].shape == value.shape
    }
    missing = sorted(set(model_state) - set(compatible))
    skipped = sorted(set(state_dict) - set(compatible))
    compatible_params = sum(value.numel() for value in compatible.values())
    model_params = sum(value.numel() for value in model_state.values())
    compatible_fraction = compatible_params / max(model_params, 1)
    if compatible_fraction < 0.8:
        examples = ", ".join(skipped[:5])
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is architecture-incompatible with the current PPO model: "
            f"only {compatible_fraction:.1%} of model parameters have matching tensor shapes. "
            "Pass the matching --d-model/--n-layers/--n-heads/--d-ff values for this checkpoint or retrain with the "
            f"current architecture. Example skipped tensors: {examples}"
        )
    model.load_state_dict(compatible, strict=False)
    if missing:
        logger.warning("Checkpoint missing %d params after compatibility filter", len(missing))
    if skipped:
        logger.warning("Checkpoint skipped %d incompatible params", len(skipped))


def _load_checkpoint_compatible(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    *,
    reinit_value_head: bool = False,
    active_reward_config: RewardConfig | None = None,
) -> None:
    """Load checkpoint, handling DataParallel prefix mismatch and minor head shape drift.

    When ``reinit_value_head`` is set, the value head keeps its random
    initialization. This is required when loading weights without matching
    reward metadata or after an intentional reward configuration change.
    """
    from ..checkpoint import load_checkpoint_payload
    from ..reward import reward_config_fingerprint

    payload = load_checkpoint_payload(checkpoint_path, device)
    if active_reward_config is not None:
        saved_fingerprint = payload.get("reward_fingerprint")
        active_fingerprint = reward_config_fingerprint(active_reward_config)
        if saved_fingerprint and saved_fingerprint != active_fingerprint and not reinit_value_head:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} was trained with a different reward "
                "fingerprint. Loading its value head through --pretrained would restore "
                "a stale critic. Pass --reinit-value-head --critic-warmup-updates 15 "
                "--critic-warmup-min-ev 0."
            )
        if not saved_fingerprint and not reinit_value_head:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} has no reward fingerprint, so loading its "
                "value head cannot verify that critic targets match the active reward model. "
                "Pass --reinit-value-head --critic-warmup-updates 15 "
                "--critic-warmup-min-ev 0."
            )

    state_dict = payload["state_dict"]
    if reinit_value_head:
        # Match both bare and DataParallel-prefixed keys, the module. prefix is
        # only stripped later, inside _load_state_dict_into_model.
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith(("value_head.", "module.value_head."))}
        logger.info(
            "Reinitializing value head (reinit_value_head=True); %d tensors loaded, value_head.* skipped.",
            len(state_dict),
        )
    _load_state_dict_into_model(model, state_dict, checkpoint_path)


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


def _make_env(
    seed: int,
    stake: int,
    data: GameData,
    vocab: Vocab,
    max_no_progress_steps: int,
    win_ante: int | None,
    reward_config: RewardConfig | None = None,
    counterfactual_diagnostic_interval: int = 0,
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
        )
        for i in range(num_envs)
    ]

    if use_async and num_envs > 1:
        return gymnasium.vector.AsyncVectorEnv(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    else:
        return gymnasium.vector.SyncVectorEnv(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)


@dataclass
class PPOConfig:
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
    # Optional override for the device used during eval. When set (e.g. "cpu"),
    # eval games run on that device instead of the training device. Useful for
    # long MPS runs where the eval loop's serial forward-pass allocations
    # otherwise compete with idle AsyncVectorEnv workers for unified memory and
    # can trip the OS memory-pressure killer. The model is moved to the eval
    # device for the duration of evaluate_model and moved back afterward.
    eval_device: str | None = None
    # Curriculum: cap the run's victory threshold below the engine default
    # (8). Heuristic-teacher win rates by ante are ~39% at 4, ~12% at 5,
    # ~2% at 6. Set to None for the standard ante-8 victory condition.
    win_ante: int | None = None
    # Stake (difficulty tier, 1-8) the training/eval envs run at. The self-play
    # curriculum ramps this; on its own PPO trains at the base stake.
    stake: int = 1
    max_no_progress_steps: int = 256
    micro_batch_size: int = 64  # Physical batch per forward pass (DataParallel grad accum)
    normalize_returns: bool = False  # BC pretraining supervises expected_score on raw ±10-ish
    # returns; turning on running-mean/std normalization here causes a one-rollout GAE
    # corruption window the first time rms.std deviates from 1, which is enough to wreck a
    # pretrained policy. Keep the value head in raw reward space.
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
    # Weight on the ante_survival auxiliary BCE loss. Small by default ,
    # the head is useful for analysis and as an auxiliary learning signal,
    # but it shouldn't meaningfully pull the policy optimization.
    survival_loss_coeff: float = 0.05
    # When True, the critic loss is included in the main backward pass so
    # value gradients flow through the shared trunk (the PPO default for
    # shared-backbone models). With the flag off, a 2-layer ValueHead must fit
    # returns from features it cannot
    # influence, leaving ppo/value_loss stuck at 20-35 (RMSE ~5 on a ±10
    # return scale). Value-gradient-through-trunk is controlled by
    # value_loss_coeff=0.25 initially; halve it if policy KL becomes erratic.
    critic_updates_trunk: bool = True
    # Optional reward settings threaded through BalatroEnv. None uses the
    # default potential-based reward configuration.
    reward_config: "RewardConfig | None" = None
    # Sampled exact joker-marginal validation. 0 disables. A positive N copies
    # one pre-play RunState and replays the same selected cards with one focal
    # joker removed every Nth actual play. Disabled by default because the state
    # copy is deliberately bounded but still material in many-env training.
    counterfactual_diagnostic_interval: int = 0
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
    # Phase 3.2: frozen-policy critic warmup. For the first ``critic_warmup_updates``
    # PPO updates, collect rollouts with the (sampling) policy but train *only*
    # the critic (+survival head) on GAE returns; policy/entropy losses
    # are multiplied by 0. After a reward-function change, advantages are garbage
    # until the critic tracks the new return distribution; warming it up
    # on-policy removes the window in which PPO earnestly optimizes noise.
    # Unfreezing is gated on explained_variance > critic_warmup_min_ev (0 = no gate).
    critic_warmup_updates: int = 0
    critic_warmup_min_ev: float = 0.7
    # Phase 3.2: reinitialize the value head when loading a pretrained checkpoint.
    # Required after any reward-function change so the critic doesn't start from a
    # stale return mapping. Pair with critic_warmup_updates to warm the fresh head.
    reinit_value_head: bool = False
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
    sil_buffer_episodes: int = 256  # FIFO capacity. Wins + ordinary losses are
    # both stored now (32 envs * rollout 256 / ~100-step episodes churn ~80
    # completed episodes per update), so 256 keeps roughly three updates of
    # history while staying bounded.
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
    value_losses: list[float]
    # MSE between the scalar value and the return target, in whatever units the
    # target is in (raw returns, or normalized when normalize_returns is on),
    # logged in both head modes. Under the HL-Gauss head value_losses holds
    # cross-entropy (nats), so value_mse is the MSE proxy that stays comparable
    # to historical MSE runs — and it is genuinely raw-scale there, since
    # normalize_returns is incompatible with the HL-Gauss head.
    value_mses: list[float]
    survival_losses: list[float]
    entropies: list[float]
    normalized_entropies: list[float]
    action_type_entropies: list[float]
    clip_fracs: list[float]
    approx_kls: list[float]
    valid_action_counts: list[float]
    valid_action_type_counts: list[float]
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
    pack_skip_state_counts: Counter = field(default_factory=Counter)
    step_rewards: list[float] = field(default_factory=list)
    progress_flags: list[float] = field(default_factory=list)
    steps_since_progress: list[float] = field(default_factory=list)
    chosen_action_probs: list[float] = field(default_factory=list)
    max_action_probs: list[float] = field(default_factory=list)
    done_flags: list[float] = field(default_factory=list)
    terminated_flags: list[float] = field(default_factory=list)
    truncated_flags: list[float] = field(default_factory=list)
    completed_episode_antes: list[int] = field(default_factory=list)
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
    shop_offered_joker_counts: Counter = field(default_factory=Counter)
    shop_bought_joker_counts: Counter = field(default_factory=Counter)
    shop_sold_joker_counts: Counter = field(default_factory=Counter)
    joker_marginal_ratios: defaultdict = field(default_factory=lambda: defaultdict(list))
    joker_modeled_fractions: defaultdict = field(default_factory=lambda: defaultdict(list))
    build_values: defaultdict = field(default_factory=lambda: defaultdict(list))
    potential_values: defaultdict = field(default_factory=lambda: defaultdict(list))
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


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model when wrapped for multi-GPU training."""
    return model.module if isinstance(model, nn.DataParallel) else model


def _grammar_distribution(model: nn.Module, batch: dict[str, torch.Tensor], temperature: float = 1.0):
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
    return base_model.action_distribution(
        batch["tokens"],
        batch["token_types"],
        batch["scalars"],
        batch["attention_mask"],
        batch["action_mask"],
        temperature=temperature,
        **history_kwargs,
    )


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
    if not 0 <= config.seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    if config.log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if config.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if config.eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    if config.ppo_epochs <= 0:
        raise ValueError("ppo_epochs must be positive")
    if config.rollout_length <= 0:
        raise ValueError("rollout_length must be positive")
    if config.max_no_progress_steps <= 0:
        raise ValueError("max_no_progress_steps must be positive")
    if config.counterfactual_diagnostic_interval < 0:
        raise ValueError("counterfactual_diagnostic_interval must be non-negative")
    if config.lr <= 0.0:
        raise ValueError("lr must be positive")
    if config.rollout_temperature <= 0.0:
        raise ValueError("rollout_temperature must be positive")
    if config.entropy_coeff < 0.0:
        raise ValueError("entropy_coeff must be non-negative")
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
    temperature: float,
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
            dist, _ = _grammar_distribution(model, batch, temperature=temperature)
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
    return_rms: "RunningMeanStd | None",
    entropy_coeff: float,
    config: PPOConfig,
    accum_steps: int,
    effective_batch_size: int,
    device: torch.device,
    use_pin_memory: bool,
    policy_loss_scale: float = 1.0,
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
    policy_frozen = policy_loss_scale <= 0.0
    stats = _UpdateStats(
        policy_losses=[],
        value_losses=[],
        value_mses=[],
        survival_losses=[],
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
        _snapshot_ppo_update(model, optimizer) if config.target_kl_max is not None and not policy_frozen else None
    )
    n_rollout_samples = len(buffer._flat_returns)
    minibatches_per_epoch = max(1, math.ceil(n_rollout_samples / effective_batch_size))
    expected_minibatches = minibatches_per_epoch * config.ppo_epochs
    min_soft_stop_fraction = config.min_minibatch_fraction if config.min_minibatch_fraction is not None else 0.0
    min_soft_stop_minibatches = math.ceil(expected_minibatches * min_soft_stop_fraction)
    running_kl = _WeightedKL()

    # SIL logical-group budget. Per update (not per epoch). Attempted groups
    # count even when their gate is empty, so training does not keep resampling
    # until it finds a favorable replay batch. Critic warmup (policy_frozen)
    # produces no SIL actor gradient.
    sil_budget = config.sil_logical_minibatches_per_update if (sil_coeff_now > 0.0 and not policy_frozen) else 0
    sil_attempted_groups = 0
    sil_applied_groups = 0
    # Gradient diagnostics: captured once, on the first group where SIL is
    # applied, by snapshotting actor .grad right after the SIL backward and
    # again before the group's global clip. PPO actor grad = total - SIL.
    grad_diag_attempted = False
    sil_grad_snapshot: list[torch.Tensor | None] | None = None

    minibatches_processed = 0
    for _ppo_epoch in range(config.ppo_epochs):
        batches = buffer.get_batches(effective_batch_size, device, pin_memory=use_pin_memory)
        optimizer.zero_grad()
        for i, batch in enumerate(batches):
            is_group_start = i % accum_steps == 0
            is_step_boundary = ((i + 1) % accum_steps == 0) or ((i + 1) == len(batches))

            # --- SIL auxiliary loss (one batch per attempted logical group) ---
            # Sampled and backwarded at the group start so the whole group
            # receives exactly one aggregate SIL contribution; the coefficient
            # is the runtime decayed value, not the static config field.
            if is_group_start and sil_attempted_groups < sil_budget and sil_buffer is not None:
                sil_attempted_groups += 1
                sil_result = _compute_sil_group_loss(model, sil_buffer, config, device, return_rms)
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

            dist, value_dict = _grammar_distribution(model, batch, temperature=config.rollout_temperature)

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
            surr2 = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (normalize targets so critic trains in unit-variance space)
            returns_target = batch["returns"]
            if return_rms is not None:
                returns_target = (returns_target - return_rms.mean) / return_rms.std
            value_logits = value_dict.get("expected_score_logits")
            if value_logits is not None:
                # HL-Gauss categorical head: cross-entropy against the
                # Gaussian-smeared projection of the scalar target. Bounded
                # per-bin gradients where MSE against bimodal near-terminal
                # returns produces large alternating-sign errors.
                value_head = _unwrap_model(model).value_head
                with torch.no_grad():
                    target_probs = hl_gauss_projection(returns_target, value_head.bin_edges, value_head.hl_gauss_sigma)
                value_loss = -(target_probs * F.log_softmax(value_logits, dim=-1)).sum(dim=-1).mean()
            else:
                value_loss = F.mse_loss(value_dict["expected_score"], returns_target)
            with torch.no_grad():
                value_mse = F.mse_loss(value_dict["expected_score"], returns_target)

            # Ante-survival aux loss (BCE masked by observed antes).
            surv_mask = batch["ante_survival_mask"]
            surv_bce = F.binary_cross_entropy(
                value_dict["ante_survival"],
                batch["ante_survival_target"],
                reduction="none",
            )
            survival_loss = (surv_bce * surv_mask).sum() / surv_mask.sum().clamp(min=1.0)

            critic_loss = config.value_loss_coeff * value_loss + config.survival_loss_coeff * survival_loss

            if policy_frozen:
                # Phase 3.2 critic warmup: a TRUE policy freeze. Never backward
                # the policy objective — a zero-scaled backward still leaves
                # zero-valued (not None) grads on policy params, and Adam then
                # moves them with its restored momentum. The critic loss IS
                # backwarded through the full graph because that is what frees
                # the trunk's saved activations each micro-batch: computing
                # value-head grads with autograd.grad leaves the previous
                # minibatch's attention buffers (tens of GiB at batch 512)
                # alive into the next forward and OOMs MPS. Every gradient it
                # writes outside the value head is dropped before the optimizer
                # step, so policy params keep grad=None and Adam skips them
                # (critic_updates_trunk is deliberately ignored while frozen:
                # trunk updates from the value loss drift the policy logits).
                value_params = _value_head_parameters(model)
                critic_loss.div(accum_steps).backward()
                value_param_ids = {id(param) for param in value_params}
                for param in model.parameters():
                    if id(param) not in value_param_ids:
                        param.grad = None
                if not value_params and not getattr(_run_ppo_update, "_frozen_no_value_head_warned", False):
                    logger.warning(
                        "Critic warmup with no value_head module: nothing trains while the policy is frozen."
                    )
                    _run_ppo_update._frozen_no_value_head_warned = True  # type: ignore[attr-defined]
            else:
                policy_objective_loss = policy_loss - entropy_coeff * (
                    normalized_entropy + config.action_type_entropy_scale * normalized_action_type_entropy
                )
                # SIL is no longer folded into per-microbatch policy_objective_loss:
                # it is sampled once per attempted logical group (see the group
                # start block above) and backwarded there with full weight, so
                # one logical group receives exactly one aggregate SIL
                # contribution while sharing this microbatch's clip + step.
                policy_objective_loss = policy_objective_loss * policy_loss_scale

                if config.critic_updates_trunk:
                    total_loss = (policy_objective_loss + critic_loss).div(accum_steps)
                    total_loss.backward()
                else:
                    value_params = _value_head_parameters(model)
                    policy_objective_loss.div(accum_steps).backward(retain_graph=bool(value_params))
                    if value_params:
                        _accumulate_critic_grads_into_value_head(critic_loss.div(accum_steps), value_params)

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
                clip_frac = ((ratio - 1.0).abs() > config.clip_epsilon).float().mean().item()
                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                valid_action_count_mean = valid_action_counts.float().mean().item()
                stats.on_policy_advantage_means.append(advantages.mean().item())
                stats.on_policy_advantage_stds.append(advantages.std().item())
                stats.on_policy_positive_advantage_fractions.append((advantages > 0).float().mean().item())
                stats.on_policy_return_means.append(batch["returns"].mean().item())
            stats.policy_losses.append(policy_loss.item())
            stats.value_losses.append(value_loss.item())
            stats.value_mses.append(value_mse.item())
            stats.survival_losses.append(survival_loss.item())
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
            config.rollout_temperature,
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


def _value_in_raw_units(values: torch.Tensor, return_rms: "RunningMeanStd | None") -> torch.Tensor:
    """Convert a critic prediction back to raw reward units when returns are normalized."""
    if return_rms is None:
        return values.detach()
    mean = torch.as_tensor(return_rms.mean, dtype=values.dtype, device=values.device)
    std = torch.as_tensor(return_rms.std, dtype=values.dtype, device=values.device)
    return (values.detach() * std + mean).detach()


@dataclass
class _SILGroupResult:
    """Outcome of one SIL logical-group loss computation."""

    loss: torch.Tensor | None
    diagnostics: dict[str, float]


def _compute_sil_group_loss(
    model: nn.Module,
    sil_buffer: "EpisodeReplayBuffer | None",
    config: PPOConfig,
    device: torch.device,
    return_rms: "RunningMeanStd | None",
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

    dist, value_dict = _grammar_distribution(model, sampled, temperature=config.rollout_temperature)
    log_probs = dist.log_prob(sampled["actions"])
    finite = (log_probs > -1e7).float()

    # Raw advantages in reward units. The stored returns are raw shaped MC
    # return-to-go; convert the current critic prediction back to raw units
    # when return normalization is enabled so the subtraction is consistent.
    v_raw = _value_in_raw_units(value_dict["expected_score"], return_rms)
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
    return_rms: "RunningMeanStd | None" = None,
    agent_config: "AgentConfig | None" = None,
    config: "PPOConfig | None" = None,
    filename: str | None = None,
    extra: dict | None = None,
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
            "clip_epsilon",
            "gae_lambda",
            "rollout_temperature",
            "action_type_entropy_scale",
            "gamma",
            "win_ante",
            "target_entropy",
            "entropy_ema_beta",
            "adaptive_entropy",
        ):
            config_fields[key] = getattr(config, key)
    active_reward_config = _effective_reward_config(config) if config is not None else DEFAULT_REWARD_CONFIG
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
        return_rms=return_rms,
        agent_config=agent_config,
        reward_config=active_reward_config,
        ppo_config_fields=config_fields,
        extra=extra,
    )
    return checkpoint_path


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


def _effective_reward_config(config: PPOConfig) -> RewardConfig:
    """Return an isolated reward config whose potential discount matches PPO."""
    base = config.reward_config if config.reward_config is not None else DEFAULT_REWARD_CONFIG
    return replace(base, gamma=config.gamma)


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
    for info_key, counter in (
        ("consumable_use_set", rm.consumable_use_set_counts),
        ("pack_claim_set", rm.consumable_claim_set_counts),
        ("shop_bought_consumable_set", rm.consumable_buy_set_counts),
    ):
        consumable_set = _extract_step_info_value(infos, info_key, env_idx, done=done, default="")
        if consumable_set:
            counter[str(consumable_set)] += 1
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
                        "hand_play_top1",
                        env_idx,
                        done=done,
                        default=False,
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
                "hand_play_candidate_value_ratio",
                env_idx,
                done=done,
                default=None,
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
) -> BalatroAgent:
    """Run PPO training with vectorized environments.

    Resume semantics:

    * ``pretrained_path``: weights-only init. Optimizer, counters, and entropy
      controller all start fresh at update 0.
    * ``resume_path``: strict resume. Loads optimizer state, update counter,
      total_steps, entropy controller, RNG, and return normalization. Requires a
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

    device = torch.device(config.device)
    use_pin_memory = device.type == "cuda"
    model = BalatroAgent(agent_config, vocab).to(device)

    if config.normalize_returns and agent_config.value_bins > 0:
        raise ValueError(
            "normalize_returns is incompatible with the HL-Gauss value head: "
            "the bin grid spans raw return units, so unit-variance targets "
            "would collapse onto a few central bins. Disable one of the two."
        )

    if resume_path and pretrained_path:
        raise ValueError(
            "Pass either --pretrained or --resume, not both. --pretrained is "
            "weights-only init; --resume restores optimizer/counters/RNG."
        )
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
        logger.info(
            "Resumed model weights from %s (update_count=%d, total_steps=%d)",
            resume_path,
            resume_state.get("update_count", 0),
            resume_state.get("total_steps", 0),
        )
        restore_rng_states(resume_state.get("rng_states", {}))
    elif pretrained_path:
        _load_checkpoint_compatible(
            model,
            pretrained_path,
            device,
            reinit_value_head=config.reinit_value_head,
            active_reward_config=config.reward_config,
        )
        if config.reinit_value_head:
            logger.info(
                "Loaded pretrained model from %s with value head reinitialized "
                "(reward function changed; critic will warm up from scratch).",
                pretrained_path,
            )
        else:
            logger.info(f"Loaded pretrained model from {pretrained_path}")
    if resume_path or pretrained_path:
        # Crossing scalar/categorical head shapes cannot preserve the critic.
        # A strict resume is also unsafe because Adam restores its tensor state
        # by parameter position and those tensors have incompatible shapes.
        loaded_bins = 0
        loaded_cfg: dict = {}
        if resume_state is not None:
            loaded_cfg = resume_state.get("agent_config") or {}
        else:
            from ..checkpoint import load_checkpoint_payload

            loaded_cfg = load_checkpoint_payload(pretrained_path, "cpu").get("agent_config") or {}
        loaded_bins = int(loaded_cfg.get("value_bins", 0) or 0)
        if resume_path and loaded_bins != agent_config.value_bins:
            raise ValueError(
                "Cannot --resume across value-head architectures "
                f"(checkpoint value_bins={loaded_bins}, model value_bins={agent_config.value_bins}): "
                "the saved Adam state is shape-incompatible. Use --pretrained "
                "--reinit-value-head with critic warmup instead."
            )
        # Same bin count but a shifted atom grid silently rescales the value
        # head: bin_centers/bin_edges are non-persistent buffers rebuilt from
        # config on load, while the position-matched Adam state and logits are
        # restored as-is. Refuse the resume rather than corrupt the value scale.
        if resume_path and loaded_bins > 0 and loaded_bins == agent_config.value_bins:
            loaded_v_min = float(loaded_cfg.get("value_v_min", agent_config.value_v_min))
            loaded_v_max = float(loaded_cfg.get("value_v_max", agent_config.value_v_max))
            if (loaded_v_min != agent_config.value_v_min) or (loaded_v_max != agent_config.value_v_max):
                raise ValueError(
                    "Cannot --resume across value-head atom ranges "
                    f"(checkpoint [{loaded_v_min}, {loaded_v_max}], "
                    f"model [{agent_config.value_v_min}, {agent_config.value_v_max}]): "
                    "the bin grid is rebuilt from config so the restored logits "
                    "and Adam state would map onto a different value scale. Match "
                    "value_v_min/value_v_max, or use --pretrained --reinit-value-head."
                )
        if loaded_bins != agent_config.value_bins and config.critic_warmup_updates <= 0:
            raise ValueError(
                "The value head is freshly initialized (checkpoint has "
                f"value_bins={loaded_bins}, model has {agent_config.value_bins}) "
                "but critic warmup is disabled. Pass --critic-warmup-updates 15 "
                "--critic-warmup-min-ev 0 so the random head converges before "
                "the policy trains on its advantages."
            )
    logger.info("Rollout temperature: %.3f (applied to rollout, training, and bootstrap)", config.rollout_temperature)
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
        accum_steps = max(1, config.mini_batch_size // config.micro_batch_size)
        effective_batch_size = config.micro_batch_size
        logger.info(f"DataParallel: grad accum {accum_steps} steps, micro_batch={effective_batch_size}")
    else:
        accum_steps = 1
        effective_batch_size = config.mini_batch_size

    optimizer = _make_policy_optimizer(model.parameters(), config.lr)
    if resume_state is not None and "optimizer_state_dict" in resume_state:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        _optimizer_to(optimizer, device)
        _apply_lr_override(optimizer, config.lr, resume_state.get("lr"))
        logger.info("Restored PPO optimizer state (Adam moments + counters) from checkpoint.")

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
    )
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
    return_rms = RunningMeanStd() if config.normalize_returns else None
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
        if config.normalize_returns and "return_rms" in resume_state:
            rms = resume_state["return_rms"]
            return_rms = RunningMeanStd()
            return_rms.mean = float(rms["mean"])
            return_rms.var = float(rms["var"])
            return_rms.count = float(rms["count"])
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
    episode_stalls: list[bool] = []
    episode_antes: list[int] = []
    # Boss on deck + blind being fought when the episode ended (loss diagnosis:
    # The Hook alone caused 59% of greedy ante-1 deaths pre-surcharge).
    episode_end_bosses: list[str] = []
    episode_end_blinds: list[str] = []
    # Phase 4: consecutive-minibatch-fraction tracker for the chronic KL-stop alert.
    _low_minibatch_streak = 0
    # Phase 3.2: latch set once explained variance clears critic_warmup_min_ev;
    # the policy stays frozen until then (metric-gated unfreeze, not count-gated).
    _critic_warmup_ev_cleared = False
    # Warmup windows are relative to the start of THIS leg: update_count
    # resumes at the checkpoint's absolute counter, and comparing it against
    # critic_warmup_updates directly would instantly trip the 4x give-up cap
    # on any resumed run, silently disabling the warmup.
    _leg_start_update = update_count
    # Phase 6: rolling win_rate/ep_reward history for the correlation acceptance
    # criterion (corr > 0.5). Currently ~0/negative because dense shaping is
    # farmable independent of winning.
    # Best-eval tracking for ppo_best_eval.pt selection. On resume, carry the
    # saved best forward so the resumed run keeps the prior best unless it beats it.
    best_eval_win_rate: float | None = None
    best_eval_update: int | None = None
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
    # Self-imitation: bounded replay of all completed non-stalled episodes
    # (wins and ordinary losses) plus a per-env tracker that assembles them
    # across rollout boundaries (episodes are ~100 steps and routinely outlive
    # a single rollout). Only created when SIL is enabled (sil_coeff > 0); the
    # runtime decayed coefficient (resolve_sil_coeff) may still hit zero near
    # the end of training, in which case _run_ppo_update skips replay work.
    sil_buffer: EpisodeReplayBuffer | None = None
    sil_tracker: SILEpisodeTracker | None = None
    if config.sil_coeff > 0.0:
        sil_buffer = EpisodeReplayBuffer(config.sil_buffer_episodes, seed=config.seed)
        sil_tracker = SILEpisodeTracker(config.num_envs, gamma=config.gamma)
    # Per-env accumulators (vectorized envs auto-reset, so we track manually)
    env_ep_reward = np.zeros(config.num_envs, dtype=np.float64)
    env_ep_length = np.zeros(config.num_envs, dtype=np.int64)
    # Per-env start step within the current rollout for the current episode.
    # Reset to 0 at each rollout, advanced past every `done` step so the
    # buffer can retroactively fill ante-survival targets for completed
    # episodes only.
    env_episode_start_step = np.zeros(config.num_envs, dtype=np.int64)

    try:
        while update_count < planned_updates:
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
                    dist, value_dict = _unwrap_model(model).action_distribution(
                        obs_buf.tokens,
                        obs_buf.token_types,
                        obs_buf.scalars,
                        obs_buf.attention_mask,
                        obs_buf.action_mask,
                        history_events=obs_buf.history_events,
                        history_event_features=obs_buf.history_event_features,
                        history_cards=obs_buf.history_cards,
                        history_card_mask=obs_buf.history_card_mask,
                        history_jokers=obs_buf.history_jokers,
                        history_joker_mask=obs_buf.history_joker_mask,
                        history_event_mask=obs_buf.history_event_mask,
                        history_round_mask=obs_buf.history_round_mask,
                        history_omitted=obs_buf.history_omitted,
                        temperature=config.rollout_temperature,
                    )
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)
                    values = value_dict["expected_score"]
                    chosen_action_probs = dist.selected_prob(actions)
                    max_action_probs = dist.max_prob()

                actions_np = actions.cpu().numpy()
                log_probs_np = log_probs.cpu().numpy()
                values_np = values.cpu().numpy()
                if return_rms is not None:
                    values_np = return_rms.denormalize(values_np)
                chosen_action_probs_np = chosen_action_probs.cpu().numpy()
                max_action_probs_np = max_action_probs.cpu().numpy()

                # Step all envs at once
                next_obs_dict, rewards, terminated, truncated, infos = vec_env.step(actions_np)
                dones = terminated | truncated
                ppo_terminated, ppo_truncated, stalled_flags = _ppo_terminal_flags(
                    terminated,
                    truncated,
                    infos,
                )
                bootstrap_values_np = np.zeros(config.num_envs, dtype=np.float32)
                if np.any(ppo_truncated) and "final_obs" in infos:
                    final_obs_arr = infos["final_obs"]
                    truncated_indices = [idx for idx in np.where(ppo_truncated)[0] if final_obs_arr[idx] is not None]
                    if truncated_indices:
                        final_obs_batch = _obs_dicts_to_batch([final_obs_arr[idx] for idx in truncated_indices], device)
                        with torch.no_grad():
                            _, truncated_value_dict = _grammar_distribution(
                                model, final_obs_batch, temperature=config.rollout_temperature
                            )
                        truncated_vals = truncated_value_dict["expected_score"].cpu().numpy()
                        if return_rms is not None:
                            truncated_vals = return_rms.denormalize(truncated_vals)
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
                if sil_tracker is not None:
                    # Same pre-step obs the rollout buffer stores; the tracker
                    # copies compactly so the shared _ObsBuffer arrays are safe
                    # to overwrite next step. Rewards feed the return-to-go
                    # used for SIL advantage gating.
                    sil_tracker.record_step(obs_buf.as_numpy_dict(), actions_np, rewards)

                # Track per-env episode stats
                env_ep_reward += rewards
                env_ep_length += 1
                rm.step_rewards.extend(rewards.astype(np.float64).tolist())
                rm.chosen_action_probs.extend(chosen_action_probs_np.astype(np.float64).tolist())
                rm.max_action_probs.extend(max_action_probs_np.astype(np.float64).tolist())
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
                    episode_rewards.append(float(env_ep_reward[i]))
                    episode_lengths.append(int(env_ep_length[i]))
                    ep_won = bool(_extract_step_info_value(infos, "won", i, done=True, default=False))
                    ep_stalled = bool(stalled_flags[i])
                    ep_ante = int(_extract_step_info_value(infos, "ante", i, done=True, default=1))
                    episode_wins.append(ep_won)
                    episode_stalls.append(ep_stalled)
                    episode_antes.append(ep_ante)
                    rm.completed_episode_antes.append(ep_ante)
                    episode_end_bosses.append(
                        str(_extract_step_info_value(infos, "boss_key", i, done=True, default="") or "")
                    )
                    episode_end_blinds.append(
                        str(_extract_step_info_value(infos, "blind_on_deck", i, done=True, default="") or "")
                    )
                    if sil_tracker is not None and sil_buffer is not None:
                        # Insert wins and ordinary completed losses; drop only
                        # episodes the environment itself flags as
                        # infrastructure/no-progress stalls (a legal
                        # policy-caused loss is kept).
                        sil_tracker.finish_episode(
                            int(i),
                            won=ep_won,
                            stalled=ep_stalled,
                            final_ante=ep_ante,
                            buffer=sil_buffer,
                        )
                    # Fill ante-survival targets for every step in this episode.
                    # Truncated-by-stall episodes have no conclusive outcome on
                    # their final ante, so leave mask=0 and skip the fill.
                    if not ep_stalled:
                        surv_target, surv_mask = compute_ante_survival_targets(ep_ante, ep_won)
                        buffer.set_episode_survival(
                            env_idx=int(i),
                            start_step=int(env_episode_start_step[i]),
                            end_step=step,
                            target=surv_target,
                            mask=surv_mask,
                        )
                    env_episode_start_step[i] = step + 1
                    env_ep_reward[i] = 0.0
                    env_ep_length[i] = 0

                # Update obs buffer with new observations
                obs_buf.update(next_obs_dict)
                total_steps += config.num_envs

            # Bootstrap values for GAE
            with torch.no_grad():
                _, value_dict = _unwrap_model(model).action_distribution(
                    obs_buf.tokens,
                    obs_buf.token_types,
                    obs_buf.scalars,
                    obs_buf.attention_mask,
                    obs_buf.action_mask,
                    history_events=obs_buf.history_events,
                    history_event_features=obs_buf.history_event_features,
                    history_cards=obs_buf.history_cards,
                    history_card_mask=obs_buf.history_card_mask,
                    history_jokers=obs_buf.history_jokers,
                    history_joker_mask=obs_buf.history_joker_mask,
                    history_event_mask=obs_buf.history_event_mask,
                    history_round_mask=obs_buf.history_round_mask,
                    history_omitted=obs_buf.history_omitted,
                    temperature=config.rollout_temperature,
                )
                last_values = value_dict["expected_score"].cpu().numpy()
                if return_rms is not None:
                    last_values = return_rms.denormalize(last_values)

            buffer.compute_returns_and_advantages(last_values=last_values)
            buffer.normalize_advantages(clip_sigma=config.advantage_clip_sigma)
            if return_rms is not None:
                return_rms.update(buffer._flat_returns)

            # Phase 3.2: explained variance of the critic on this rollout.
            # 1 - Var(returns - values) / Var(returns). ~0.6 currently (derived
            # from ppo/value_loss ~25 vs returns_std ~9.5); gate critic-warmup
            # unfreezing on this exceeding critic_warmup_min_ev (>0.7).
            flat_returns = buffer._flat_returns
            explained_variance = float("nan")
            if len(flat_returns) > 1:
                var_returns = float(np.var(flat_returns))
                if var_returns > 1e-9:
                    explained_variance = 1.0 - float(np.var(flat_returns - buffer._flat_values)) / var_returns
            writer.add_scalar("ppo/explained_variance", explained_variance, update_count + 1)
            rollout_step = update_count + 1
            writer.add_scalar("rollout/step_reward_mean", float(np.mean(rm.step_rewards)), rollout_step)
            writer.add_scalar("rollout/reward_mean", float(np.mean(rm.step_rewards)), rollout_step)
            writer.add_scalar("rollout/chosen_action_prob_mean", float(np.mean(rm.chosen_action_probs)), rollout_step)
            writer.add_scalar("rollout/max_action_prob_mean", float(np.mean(rm.max_action_probs)), rollout_step)
            writer.add_scalar("rollout/done_rate", float(np.mean(rm.done_flags)), rollout_step)
            writer.add_scalar("rollout/progress_rate", float(np.mean(rm.progress_flags)), rollout_step)
            writer.add_scalar("rollout/completed_episodes", float(np.sum(rm.done_flags)), rollout_step)
            if rm.completed_episode_antes:
                writer.add_scalar(
                    "rollout/mean_ante_reached",
                    float(np.mean(rm.completed_episode_antes)),
                    rollout_step,
                )
            if episode_rewards:
                writer.add_scalar("rollout/episode_reward_mean", float(np.mean(episode_rewards)), rollout_step)
                writer.add_scalar("rollout/episode_length_mean", float(np.mean(episode_lengths)), rollout_step)
                writer.add_scalar("rollout/win_rate", float(np.mean(episode_wins)), rollout_step)
                writer.add_scalar("rollout/stall_rate", float(np.mean(episode_stalls)), rollout_step)
                writer.add_scalar("rollout/final_ante_mean", float(np.mean(episode_antes)), rollout_step)
            hand_total = sum(rm.hand_chosen_counts.values())
            if hand_total:
                for hand_name, count in rm.hand_chosen_counts.items():
                    tag_name = str(hand_name).lower().replace(" ", "_")
                    writer.add_scalar(f"rollout/hands_played/{tag_name}", count / hand_total, rollout_step)
            steps_per_thousand = max(len(rm.step_rewards) / 1000.0, 1e-9)
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
            for component_name, values in rm.reward_component_values.items():
                if values:
                    writer.add_scalar(
                        f"rollout/reward_components/{component_name}",
                        float(np.mean(values)),
                        rollout_step,
                    )

            # Phase 3.2: determine whether this update is in critic-warmup
            # (policy frozen, only critic + survival train). Unfreezing is
            # gated on the EV metric, not the update count: the policy stays
            # frozen past critic_warmup_updates until EV clears
            # critic_warmup_min_ev (latched, a later noisy dip does not
            # re-freeze). A 4x hard cap prevents an unreachable gate from
            # freezing the policy forever. With critic_warmup_min_ev <= 0 the
            # warmup is purely count-based.
            in_critic_warmup = False
            if config.critic_warmup_updates > 0 and not _critic_warmup_ev_cleared:
                leg_update = update_count - _leg_start_update
                if config.critic_warmup_min_ev <= 0.0:
                    in_critic_warmup = leg_update < config.critic_warmup_updates
                elif (
                    leg_update > 0
                    and np.isfinite(explained_variance)
                    and explained_variance >= config.critic_warmup_min_ev
                ):
                    _critic_warmup_ev_cleared = True
                elif leg_update >= 4 * config.critic_warmup_updates:
                    _critic_warmup_ev_cleared = True
                    logger.warning(
                        "Critic warmup EV gate (%.2f) not reached after %d updates "
                        "(4x critic_warmup_updates cap); unfreezing anyway. EV=%.3f. "
                        "Advantages may be noisy, consider a longer warmup or a "
                        "higher value_loss_coeff.",
                        config.critic_warmup_min_ev,
                        update_count,
                        explained_variance,
                    )
                else:
                    in_critic_warmup = True
            if in_critic_warmup:
                writer.add_scalar("ppo/critic_warmup_active", 1.0, update_count + 1)
                logger.info(
                    "Critic warmup active at update %d (EV=%.3f, gate=%.2f); policy frozen.",
                    update_count + 1,
                    explained_variance,
                    config.critic_warmup_min_ev,
                )

            # SIL coefficient uses the same pinned schedule horizon so it decays
            # monotonically to sil_coeff_final without rewinding on resume.
            sil_coeff_now = resolve_sil_coeff(config, total_steps, schedule_total_steps=schedule_total_steps)
            # Critic warmup must produce no SIL actor gradient; force the
            # runtime coefficient to zero while the policy is frozen regardless
            # of the schedule.
            if in_critic_warmup:
                sil_coeff_now = 0.0
            sil_grad_diagnostics_due = (
                sil_coeff_now > 0.0
                and config.sil_grad_diagnostics_interval > 0
                and (update_count + 1) % config.sil_grad_diagnostics_interval == 0
            )

            update_stats = _run_ppo_update(
                model=model,
                optimizer=optimizer,
                buffer=buffer,
                return_rms=return_rms,
                entropy_coeff=entropy_coeff,
                config=config,
                accum_steps=accum_steps,
                effective_batch_size=effective_batch_size,
                device=device,
                use_pin_memory=use_pin_memory,
                policy_loss_scale=0.0 if in_critic_warmup else 1.0,
                sil_buffer=sil_buffer,
                sil_coeff_now=sil_coeff_now,
                grad_diagnostics_due=sil_grad_diagnostics_due,
            )

            update_policy_losses = update_stats.policy_losses
            update_value_losses = update_stats.value_losses
            update_entropies = update_stats.entropies
            update_normalized_entropies = update_stats.normalized_entropies
            update_action_type_entropies = update_stats.action_type_entropies
            update_clip_fracs = update_stats.clip_fracs
            update_approx_kls = update_stats.approx_kls
            update_valid_action_counts = update_stats.valid_action_counts
            update_valid_action_type_counts = update_stats.valid_action_type_counts

            # Track the normalized entropy signal every update, even with fixed entropy.
            mean_normalized_entropy = float(np.mean(update_normalized_entropies))
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
            mean_entropy = float(np.mean(update_entropies))
            writer.add_scalar("ppo/policy_loss", np.mean(update_policy_losses), update_count)
            writer.add_scalar("ppo/value_loss", np.mean(update_value_losses), update_count)
            writer.add_scalar("ppo/entropy_raw", mean_entropy, update_count)
            # Return-unit MSE regardless of head mode (raw under HL-Gauss, where
            # returns are never normalized); under HL-Gauss value_loss is
            # cross-entropy so this is the run-over-run comparable series.
            writer.add_scalar("ppo/value_mse", np.mean(update_stats.value_mses), update_count)
            writer.add_scalar("ppo/entropy_normalized", np.mean(update_normalized_entropies), update_count)
            writer.add_scalar("ppo/action_type_entropy_normalized", np.mean(update_action_type_entropies), update_count)
            writer.add_scalar("ppo/clip_fraction", np.mean(update_clip_fracs), update_count)
            writer.add_scalar("ppo/approx_kl", update_stats.running_kl, update_count)
            writer.add_scalar(
                "ppo/kl_rollback_event",
                1.0 if update_stats.kl_rollback else 0.0,
                update_count,
            )
            writer.add_scalar("ppo/actual_lr", update_stats.actual_lr, update_count)
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
                minibatches_per_epoch = max(1, math.ceil(n_samples / effective_batch_size))
                minibatches_expected = minibatches_per_epoch * config.ppo_epochs
                minibatch_fraction = min(1.0, minibatches_processed / minibatches_expected)
                writer.add_scalar("ppo/minibatch_fraction", minibatch_fraction, update_count)
            # Trust-region extrema feed the console diagnostic warnings below.
            kl_p95 = float(np.percentile(update_approx_kls, 95)) if update_approx_kls else 0.0
            kl_max = float(np.max(update_approx_kls)) if update_approx_kls else 0.0
            clip_frac_max = float(np.max(update_clip_fracs)) if update_clip_fracs else 0.0
            writer.add_scalar(
                "ppo/valid_action_type_count_mean", np.mean(update_valid_action_type_counts), update_count
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
            if should_checkpoint:
                # Make sure all preceding scalars
                # land on disk before the checkpoint save, which is itself
                # a long sync that could crash if memory is tight.
                writer.flush()
                checkpoint_path = _save_checkpoint(
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
                    return_rms=return_rms,
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
                latest_path = save_path / "ppo_latest.pt"
                try:
                    latest_path.unlink(missing_ok=True)
                    import shutil

                    shutil.copy2(checkpoint_path, latest_path)
                except OSError:
                    logger.warning("Could not mirror latest checkpoint to %s", latest_path)
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
                        return_rms=return_rms,
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
                    f"policy_loss={np.mean(update_policy_losses):.4f}, "
                    f"value_loss={np.mean(update_value_losses):.4f}, "
                    f"entropy={mean_entropy:.4f}, "
                    f"entropy_signal={entropy_signal_ema:.4f}, "
                    f"entropy_coeff={entropy_coeff:.5f}, "
                    f"clip_fraction={np.mean(update_clip_fracs):.4f}, "
                    f"approx_kl={np.mean(update_approx_kls):.5f}, "
                    f"valid_actions={np.mean(update_valid_action_counts):.1f}, "
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
            mean_policy_loss = float(np.mean(update_policy_losses))
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
            mean_normalized_entropy_update = float(np.mean(update_normalized_entropies))
            mean_chosen_action_prob = _safe_mean(rm.chosen_action_probs)
            mean_action_type_entropy_update = float(np.mean(update_action_type_entropies))
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
        self._np_history_event_mask = np.zeros(
            (num_envs, HISTORY_ROUNDS, HISTORY_MAX_PLAYS), dtype=np.int64
        )
        self._np_history_round_mask = np.zeros((num_envs, HISTORY_ROUNDS), dtype=np.int64)
        self._np_history_omitted = np.zeros(
            (num_envs, HISTORY_ROUNDS, HISTORY_OMITTED_DIM), dtype=np.float32
        )

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
) -> float:
    """Evaluate model win rate with greedy action selection over num_games.

    When ``seeds`` is provided it is used verbatim (and truncated to
    ``num_games``); otherwise the historical ``10000 + game_idx`` seeds are
    generated. Passing the versioned :data:`pylatro_agent.eval.EVAL_SEEDS_V1`
    list makes every checkpoint's eval reproducible and pairable across runs.

    Memory safety:

    * Runs under ``torch.inference_mode`` so eval never builds an autograd
      graph. Categorical sampling on MPS otherwise leaks intermediate
      tensors that accumulate across ``num_games``.
    * Drains the MPS cache every 50 games so peak unified-memory usage
      stays bounded; without this an eval-heavy run can jetsam the idle
      AsyncVectorEnv workers (manifesting as EOFError/BrokenPipe on the
      next rollout step).
    """
    model.eval()
    wins = 0
    seed_list = seeds[:num_games] if seeds is not None else None

    with torch.inference_mode():
        for game_idx in range(num_games):
            # Eval runs serially in-process; each game caches MPS forward-pass
            # allocations that are only released when the whole eval finishes. Over
            # a large `num_games` that ramp builds enough unified-memory pressure to
            # let the OS jetsam an idle AsyncVectorEnv worker (-> EOFError/BrokenPipe
            # on the next rollout step). Drain the cache periodically to cap the peak.
            if device.type == "mps" and game_idx > 0 and game_idx % 50 == 0:
                torch.mps.empty_cache()

            env_seed = seed_list[game_idx] if seed_list is not None else 10000 + game_idx
            env = BalatroEnv(
                seed=env_seed,
                data=data,
                vocab=vocab,
                stake=stake,
                max_steps=max_no_progress_steps,
                win_ante=win_ante,
                # Greedy eval never reads teacher labels; the heuristic teacher
                # would otherwise run twice per step of every eval game.
                enable_teacher=False,
            )
            obs, _ = env.reset()
            done = False

            while not done:
                batch = _single_obs_to_batch(obs, device)
                dist, _ = _grammar_distribution(model, batch, temperature=temperature)
                action = dist.mode().item()
                # Drop the per-step inference tensors immediately; otherwise
                # they sit on the MPS allocator until the end of the game.
                del batch, dist
                obs, _reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            if info.get("won", False):
                wins += 1

    return wins / max(num_games, 1)


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
        "history_card_mask": torch.tensor(
            obs["history_card_mask"], dtype=torch.long, device=device
        ).unsqueeze(0),
        "history_jokers": torch.tensor(obs["history_jokers"], dtype=torch.long, device=device).unsqueeze(0),
        "history_joker_mask": torch.tensor(
            obs["history_joker_mask"], dtype=torch.long, device=device
        ).unsqueeze(0),
        "history_event_mask": torch.tensor(
            obs["history_event_mask"], dtype=torch.long, device=device
        ).unsqueeze(0),
        "history_round_mask": torch.tensor(
            obs["history_round_mask"], dtype=torch.long, device=device
        ).unsqueeze(0),
        "history_omitted": torch.tensor(
            obs["history_omitted"], dtype=torch.float32, device=device
        ).unsqueeze(0),
    }
