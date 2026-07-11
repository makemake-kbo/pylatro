"""Phase 2: PPO training loop with vectorized environments."""

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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
from ..constants import MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM
from ..env import BalatroEnv
from ..reward import _COMPONENT_GROUP, REWARD_INFO_KEYS, RewardConfig
from ..survival import compute_ante_survival_targets
from ..vocab import Vocab, build_vocab
from .rollout_buffer import RolloutBuffer
from .sil import SILEpisodeTracker, WinEpisodeBuffer

logger = logging.getLogger(__name__)
_MISSING = object()


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

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean
_ACTION_TYPES = tuple(ActionType)
_ACTION_TYPE_TO_INDEX = {action_type: idx for idx, action_type in enumerate(_ACTION_TYPES)}
_ACTION_ID_TO_TYPE_INDEX = torch.tensor(
    [_ACTION_TYPE_TO_INDEX[decode_action(action_id).action_type] for action_id in range(NUM_ACTIONS)],
    dtype=torch.long,
)

# Coarse action families used to break down *unreachable* teacher labels, so a
# distillation run can tell whether the heuristic teacher is unreachable because
# it picks hand subsets the structured policy can't represent, shop/pack items,
# or blind selections.
_ACTION_FAMILY_NAMES = ("play_subset", "use_consumable", "shop", "pack", "blind")
_ACTION_TYPE_TO_FAMILY: dict[ActionType, int] = {
    ActionType.BLIND_PLAY: 4,
    ActionType.BLIND_SKIP: 4,
    ActionType.BLIND_REROLL: 4,
    ActionType.PLAY_SUBSET: 0,
    ActionType.DISCARD_SUBSET: 0,
    ActionType.USE_CONSUMABLE_NO_TARGET: 1,
    ActionType.USE_CONSUMABLE_HAND_SUBSET: 1,
    ActionType.USE_CONSUMABLE_JOKER: 1,
    ActionType.SHOP_BUY: 2,
    ActionType.SHOP_REROLL: 2,
    ActionType.SHOP_SELL_JOKER: 2,
    ActionType.SHOP_SELL_CONSUMABLE: 2,
    ActionType.SHOP_LEAVE: 2,
    ActionType.PACK_CLAIM: 3,
    ActionType.PACK_SKIP: 3,
}
_ACTION_ID_TO_FAMILY = torch.tensor(
    [_ACTION_TYPE_TO_FAMILY[decode_action(action_id).action_type] for action_id in range(NUM_ACTIONS)],
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
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
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
) -> None:
    """Load checkpoint, handling DataParallel prefix mismatch and minor head shape drift.

    When ``reinit_value_head`` is set, the value head weights are *not* loaded ,
    they keep their random init. Use this after a reward-function change (Phase
    2.4): a critic regressing returns from a deleted reward function produces
    systematically wrong advantages until it re-converges, during which PPO can
    destroy the BC policy. Reinitializing forces the critic to learn the new
    return from scratch rather than unlearning a stale mapping.
    """
    from ..checkpoint import load_checkpoint_payload
    state_dict = load_checkpoint_payload(checkpoint_path, device)["state_dict"]
    if reinit_value_head:
        # Match both bare and DataParallel-prefixed keys, the module. prefix is
        # only stripped later, inside _load_state_dict_into_model.
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith(("value_head.", "module.value_head."))
        }
        logger.info(
            "Reinitializing value head (reinit_value_head=True); %d tensors loaded, "
            "value_head.* skipped.",
            len(state_dict),
        )
    _load_state_dict_into_model(model, state_dict, checkpoint_path)


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer state tensors to ``device`` after ``load_state_dict``."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _apply_lr_override(
    optimizer: torch.optim.Optimizer, new_lr: float, checkpoint_lr: float | None
) -> None:
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
    enable_teacher: bool = True,
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
            enable_teacher=enable_teacher,
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
    enable_teacher: bool = True,
):
    """Create a gymnasium VectorEnv (async for multiprocess, sync for single-process)."""
    import gymnasium

    env_fns = [
        _make_env(
            env_seed_base + i, stake, data, vocab, max_no_progress_steps,
            win_ante, reward_config, enable_teacher=enable_teacher,
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
    rollout_length: int = 256
    total_timesteps: int = 1_000_000
    # When set, overrides total_timesteps: the run trains for exactly this many
    # PPO updates and total_timesteps is derived as total_updates * steps_per_update.
    # Useful for "N more updates" semantics without doing the timesteps math.
    total_updates: int | None = None
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.1  # PPO clip range; tighter than the usual 0.2
    target_kl: float | None = 0.05  # Phase 4: raised from 0.03; chronic KL-stop means the step size is wrong, not the trust region
    # Optional trust-region diagnostic guards (do not stop training on their
    # own; they log ppo/stop_reason_* and emit console warnings when breached).
    target_kl_p95: float | None = None
    target_kl_max: float | None = None
    min_minibatch_fraction: float | None = None
    entropy_coeff: float = 0.01  # Phase 4: fixed at 0.01 (was 0.001); mixture eps is now the primary exploration mechanism
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
    # Continuous distillation from the heuristic teacher. At each step the
    # env emits HeuristicAgent.select_action(...) into info["teacher_action"];
    # PPO adds a NLL term -log pi(a_teacher | s) under the structured
    # ActionGrammarDistribution to its loss. Replaces the previous
    # frozen-reference KL anchor, denser per-state supervision and no extra
    # forward pass through a second model.
    heuristic_distill_coeff: float = 0.3
    heuristic_distill_min: float = 0.0  # Phase 4: was 0.03; a nonzero floor anchors the policy to the heuristic forever
    # During curriculum smoke/training, execute the heuristic action with this
    # probability when the env exposes one. The transition still stores the
    # policy log-prob of that action, so PPO and distillation both train on the
    # successful trajectory instead of waiting for a sampled policy to stumble
    # into sparse wins. Keep at 0.0 for strictly on-policy PPO.
    teacher_rollout_prob: float = 0.0
    # Optional linear schedule for teacher_rollout_prob. When final_prob is
    # set, keep teacher_rollout_prob fixed through warmup_fraction, then
    # linearly anneal it to final_prob over decay_fraction of training.
    teacher_rollout_final_prob: float | None = None
    teacher_rollout_warmup_fraction: float = 0.0
    teacher_rollout_decay_fraction: float = 1.0
    # Sharpens the on-policy distribution for both rollout sampling and PPO loss
    # computation. The BC-pretrained policy at temperature=1 has chosen_action_prob ≈ 0.5
    # over ~250 valid actions per state, which means a 30-step sampled episode has ~0.5^30
    # probability of even matching its own greedy trajectory, sampled rollouts essentially
    # never win and PPO sees no positive advantage to lock onto. Sharpening the distribution
    # by a fixed factor at all sites (rollout, train forward, truncation bootstrap) keeps
    # PPO consistent, old_log_probs and new_log_probs are computed under the same
    # distribution, while letting the agent take competent actions in rollouts. Set to 1.0
    # to disable; lower for more deterministic behavior.
    rollout_temperature: float = 1.0  # Phase 4: was 0.7; the sharpening crutch now only suppresses exploration post-Phase-1 BC
    # Weight on the ante_survival auxiliary BCE loss. Small by default ,
    # the head is useful for analysis and as an auxiliary learning signal,
    # but it shouldn't meaningfully pull the policy optimization.
    survival_loss_coeff: float = 0.05
    # Optional DAgger-style behavior cloning pass over the freshly collected
    # online states before each PPO update. This uses teacher labels only and
    # does not treat teacher-forced actions as on-policy PPO samples.
    dagger_bc_epochs: int = 0
    dagger_bc_coeff: float = 1.0
    dagger_bc_lr_mult: float = 1.0
    # When True, the critic loss is included in the main backward pass so
    # value gradients flow through the shared trunk (the PPO default for
    # shared-backbone models). Phase 3.1 flips this from False: with the flag
    # off, a 2-layer ValueHead must fit returns from features it cannot
    # influence, leaving ppo/value_loss stuck at 20-35 (RMSE ~5 on a ±10
    # return scale). Value-gradient-through-trunk is controlled by
    # value_loss_coeff=0.25 initially; halve it if policy KL becomes erratic.
    critic_updates_trunk: bool = True
    # Linear decay schedule for heuristic distillation. When set, the
    # distill coefficient linearly decays from heuristic_distill_coeff
    # to heuristic_distill_min over this fraction of total training.
    # After the decay completes, distillation is permanently at min.
    # Set to None to disable decay (keep constant distill weight).
    # To run a true no-teacher fine-tune phase, also set
    # heuristic_distill_min=0.0.
    # Phase 4: distill decays to 0 over half of training (was constant with a 0.03
    # floor). The teacher is a good warmup prior and a terrible ceiling.
    distill_decay_fraction: float | None = 0.5
    # Optional reward shaping override. Threaded through BalatroEnv to
    # default_reward_components. Use PPO_SPARSE_CONFIG to ablate away
    # heuristic-derived dense components (hand-candidate top1/top3,
    # planet match, shop reroll, etc.) while keeping terminal and
    # progress signals. Defaults to None (DEFAULT_REWARD_CONFIG).
    reward_config: "RewardConfig | None" = None
    # Fixed, versioned seed list for the in-training eval pass. Passing a
    # stable list (pylatro_agent.eval.EVAL_SEEDS_V1) makes every checkpoint's
    # eval reproducible and pairable across runs via the McNemar / paired
    # bootstrap harness in pylatro_agent.eval. None preserves the historical
    # 10000 + game_idx seeds.
    eval_seeds: list[int] | None = None
    # Mixture weight on the autoregressive hand/discard head (Phase 1). 0.0
    # reproduces the historical candidate-only support; 0.1 (default for PPO)
    # gives the policy full support over every legal hand play while keeping
    # the candidate head dominant. The AR head is by this point a competent
    # proposal distribution (trained at eps=0.5 during BC), not noise.
    hand_ar_mixture_eps: float = 0.1
    # Phase 3.2: frozen-policy critic warmup. For the first ``critic_warmup_updates``
    # PPO updates, collect rollouts with the (sampling) policy but train *only*
    # the critic (+survival head) on GAE returns; policy/distill/entropy losses
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
    # Re-anchor the fraction-of-training anneal schedules (heuristic distill,
    # teacher-rollout prob) to THIS leg's recomputed total_timesteps instead of
    # the horizon persisted in the checkpoint. Without this flag, resumed runs
    # keep the original horizon so an already-decayed coefficient can never
    # climb back when num_envs or the update target grows.
    reset_schedules: bool = False
    # Self-imitation (SIL-as-BC) on the agent's own winning episodes. Wins at
    # high win-ante targets are too sparse for on-policy PPO (a handful per
    # update); replaying complete winning trajectories as a behavior-cloning
    # term multiplies the win-signal density without touching the reward
    # function — only genuine wins enter the buffer, so there is nothing to
    # farm. The NLL term is added to the PPO minibatch objective (single
    # backward per minibatch) so the coefficient trades off directly against
    # the policy/entropy/distill terms; a separate optimizer pass would let
    # Adam's gradient renormalization largely cancel the coefficient. Riding
    # inside the PPO loop also puts SIL movement under the target-kl guard.
    # 0.0 disables. The buffer is in-memory only; it refills over the first
    # ~buffer/wins-per-update updates after a resume.
    sil_coeff: float = 0.0
    sil_buffer_episodes: int = 64  # FIFO capacity (~50MB at typical episode lengths)
    sil_batch_size: int = 64  # transitions sampled per PPO micro-batch
    sil_min_episodes: int = 8  # skip SIL until the buffer holds this many wins
    # Advantage gating (Oh et al. 2018): weight each sampled transition's NLL
    # by min((R - V)+ / sil_advantage_clip, 1) instead of imitating uniformly.
    # Transitions the critic already values correctly contribute nothing, so
    # the term self-decays as wins become routine; without it, abundant wins
    # (32 envs at ~30% win rate churn the FIFO every ~2 updates) turn SIL into
    # near-on-policy self-cloning that entrenches the current mode.
    # sil_advantage_clip is the return shortfall (in reward units) at which a
    # transition reaches full BC weight, so sil_coeff keeps its ungated meaning
    # as the maximum per-transition pull.
    sil_advantage_gating: bool = True
    sil_advantage_clip: float = 3.0


@dataclass
class _UpdateStats:
    policy_losses: list[float]
    value_losses: list[float]
    survival_losses: list[float]
    entropies: list[float]
    normalized_entropies: list[float]
    action_type_entropies: list[float]
    clip_fracs: list[float]
    approx_kls: list[float]
    valid_action_counts: list[float]
    valid_action_type_counts: list[float]
    distill_losses: list[float]
    distill_losses_weighted: list[float] = field(default_factory=list)
    teacher_match_fractions: list[float] = field(default_factory=list)
    distill_weight_means: list[float] = field(default_factory=list)
    teacher_reachable_fractions: list[float] = field(default_factory=list)
    teacher_unreachable_fractions: list[float] = field(default_factory=list)
    teacher_lp_mean_reachable: list[float] = field(default_factory=list)
    # ``label_present`` is the fraction of all steps where the env emitted a
    # teacher label. ``action_mask_valid`` is, of those labels, how many are
    # legal under the step's action mask; ``reachable`` is the subset that also
    # has a finite policy log-prob above the -1e8 floor.
    teacher_label_present_fractions: list[float] = field(default_factory=list)
    teacher_action_mask_valid_fractions: list[float] = field(default_factory=list)
    # Breakdown of unreachable (present-but-not-reachable) teacher labels by
    # action family, so distillation diagnostics can tell hand-subset
    # mismatches apart from shop/pack/blind unreachability.
    teacher_unreachable_family_fractions: dict[str, list[float]] = field(default_factory=dict)
    on_policy_fractions: list[float] = field(default_factory=list)
    on_policy_advantage_means: list[float] = field(default_factory=list)
    on_policy_advantage_stds: list[float] = field(default_factory=list)
    on_policy_positive_advantage_fractions: list[float] = field(default_factory=list)
    on_policy_return_means: list[float] = field(default_factory=list)
    ppo_minibatches_processed: list[int] = field(default_factory=list)
    sil_losses: list[float] = field(default_factory=list)
    sil_advantage_means: list[float] = field(default_factory=list)
    sil_gate_means: list[float] = field(default_factory=list)


@dataclass
class _RolloutMetrics:
    """Per-rollout accumulators populated during the step loop."""
    action_type_counts: Counter = field(default_factory=Counter)
    hand_chosen_counts: Counter = field(default_factory=Counter)
    hand_best_counts: Counter = field(default_factory=Counter)
    planet_use_key_counts: Counter = field(default_factory=Counter)
    planet_claim_key_counts: Counter = field(default_factory=Counter)
    pack_skip_state_counts: Counter = field(default_factory=Counter)
    step_rewards: list[float] = field(default_factory=list)
    progress_flags: list[float] = field(default_factory=list)
    steps_since_progress: list[float] = field(default_factory=list)
    chosen_action_probs: list[float] = field(default_factory=list)
    max_action_probs: list[float] = field(default_factory=list)
    teacher_rollout_used: list[float] = field(default_factory=list)
    done_flags: list[float] = field(default_factory=list)
    terminated_flags: list[float] = field(default_factory=list)
    truncated_flags: list[float] = field(default_factory=list)
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
    play_subset_count: int = 0
    discard_subset_count: int = 0


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model when wrapped for multi-GPU training."""
    return model.module if isinstance(model, nn.DataParallel) else model


def _grammar_distribution(model: nn.Module, batch: dict[str, torch.Tensor], temperature: float = 1.0):
    base_model = _unwrap_model(model)
    return base_model.action_distribution(
        batch["tokens"],
        batch["token_types"],
        batch["scalars"],
        batch["attention_mask"],
        batch["action_mask"],
        temperature=temperature,
    )


def _value_head_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return PPO critic parameters whose losses should not update the shared trunk."""
    value_head = getattr(_unwrap_model(model), "value_head", None)
    if value_head is None:
        return []
    return [param for param in value_head.parameters() if param.requires_grad]


def _accumulate_critic_grads_into_value_head(
    critic_loss: torch.Tensor, value_params: list[nn.Parameter]
) -> None:
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


def _validate_ppo_config(config: PPOConfig) -> None:
    """Raise ValueError for invalid combinations; warn on risky ones."""
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
    if config.lr <= 0.0:
        raise ValueError("lr must be positive")
    if config.rollout_temperature <= 0.0:
        raise ValueError("rollout_temperature must be positive")
    if config.entropy_coeff < 0.0:
        raise ValueError("entropy_coeff must be non-negative")
    if config.target_kl is not None and config.target_kl <= 0.0:
        raise ValueError("target_kl must be positive when set")
    if not 0.0 <= config.teacher_rollout_prob <= 1.0:
        raise ValueError("teacher_rollout_prob must be between 0 and 1")
    if config.teacher_rollout_final_prob is not None and not 0.0 <= config.teacher_rollout_final_prob <= 1.0:
        raise ValueError("teacher_rollout_final_prob must be between 0 and 1")
    if not 0.0 <= config.teacher_rollout_warmup_fraction <= 1.0:
        raise ValueError("teacher_rollout_warmup_fraction must be between 0 and 1")
    if not 0.0 <= config.teacher_rollout_decay_fraction <= 1.0:
        raise ValueError("teacher_rollout_decay_fraction must be between 0 and 1")
    if config.action_type_entropy_scale < 0.0:
        raise ValueError("action_type_entropy_scale must be non-negative")
    if config.dagger_bc_epochs < 0:
        raise ValueError("dagger_bc_epochs must be non-negative")
    if config.dagger_bc_coeff < 0.0:
        raise ValueError("dagger_bc_coeff must be non-negative")
    if config.dagger_bc_lr_mult <= 0.0:
        raise ValueError("dagger_bc_lr_mult must be positive")
    if config.sil_coeff < 0.0:
        raise ValueError("sil_coeff must be non-negative")
    if config.sil_buffer_episodes <= 0:
        raise ValueError("sil_buffer_episodes must be positive")
    if config.sil_batch_size <= 0:
        raise ValueError("sil_batch_size must be positive")
    if config.sil_min_episodes <= 0:
        raise ValueError("sil_min_episodes must be positive")
    if config.sil_advantage_clip <= 0.0:
        raise ValueError("sil_advantage_clip must be positive")
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


def resolve_distill_coeff(
    config: "PPOConfig",
    total_steps: int,
    schedule_total_steps: int | None = None,
) -> float:
    """Return the heuristic-teacher distillation coefficient at ``total_steps``.

    Semantics:

    * ``heuristic_distill_coeff <= 0`` -> distillation is fully disabled (returns 0.0).
      This makes ``--heuristic-distill-coeff 0.0`` an explicit "turn it off" switch
      regardless of ``--heuristic-distill-min``.
    * Otherwise the coefficient linearly decays from ``heuristic_distill_coeff``
      toward a floor over ``distill_decay_fraction`` of training (defaulting to
      the full run when ``None``).
    * The floor is clamped to ``min(heuristic_distill_min, heuristic_distill_coeff)``
      so a misconfigured floor that exceeds the start value cannot silently pin
      distillation above the requested starting coefficient; a warning is logged
      the first time that condition is seen.
    * ``schedule_total_steps`` pins the anneal horizon independently of
      ``config.total_timesteps``. Resume recomputes ``total_timesteps`` from the
      current ``num_envs``/update target, so annealing against it rewinds an
      already-decayed coefficient whenever those grow (a resumed 8->32-env run
      revived a fully-decayed teacher at coeff ~0.164). The train loop passes
      the horizon captured at the original run's start; ``None`` preserves the
      legacy behavior for callers without a pinned horizon.
    """
    if config.heuristic_distill_coeff <= 0.0:
        return 0.0

    floor = min(config.heuristic_distill_min, config.heuristic_distill_coeff)
    if floor < config.heuristic_distill_min:
        # Only log on the first call so we don't spam the training loop.
        if not getattr(resolve_distill_coeff, "_floor_clamp_warned", False):
            logger.warning(
                "heuristic_distill_min=%.4f exceeds heuristic_distill_coeff=%.4f; "
                "clamping the distillation floor to %.4f. Set --heuristic-distill-min "
                "<= --heuristic-distill-coeff to remove this warning.",
                config.heuristic_distill_min,
                config.heuristic_distill_coeff,
                floor,
            )
            resolve_distill_coeff._floor_clamp_warned = True  # type: ignore[attr-defined]

    decay_fraction = (
        config.distill_decay_fraction
        if config.distill_decay_fraction is not None
        else 1.0
    )
    horizon = (
        schedule_total_steps
        if schedule_total_steps is not None
        else config.total_timesteps
    )
    decay_steps = max(1, int(horizon * decay_fraction))
    progress = min(1.0, total_steps / decay_steps)

    return max(
        floor,
        config.heuristic_distill_coeff
        - (config.heuristic_distill_coeff - floor) * progress,
    )


def _resolve_schedule_total_steps(
    config: "PPOConfig", resume_state: dict | None
) -> int:
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
            "reset_schedules: re-anchoring anneal schedules (distill coeff, "
            "teacher-rollout prob) to this leg's horizon of %d steps. Already-"
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
        "grew, previously-decayed coefficients (heuristic distill, teacher-"
        "rollout prob) will REWIND — pass --heuristic-distill-coeff 0.0 unless "
        "teacher re-anchoring is intended.",
        config.total_timesteps,
    )
    return config.total_timesteps


def _teacher_reachability_mask(
    teacher: torch.Tensor,
    teacher_lp: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Return a float tensor marking which teacher actions can safely drive distillation.

    A teacher action is "reachable" when ALL of:
      * the env actually emitted one (``teacher >= 0``);
      * the slot itself is legal under the current step's action_mask;
      * the policy distribution reports a finite log-prob above the
        ``-1e8`` floor used by ``ActionGrammarDistribution.log_prob`` for
        out-of-candidates slots.

    The third condition is what protects us from the historical blow-up where
    the heuristic teacher picked a valid raw hand action that the structured
    policy cannot represent in its candidate-hand slots, ``log_prob`` returned
    ``-1e8``, and the resulting NLL term dominated the PPO loss with values in
    the 17M-21M range.
    """
    teacher_in_range = (teacher >= 0) & (teacher < NUM_ACTIONS)
    teacher_safe = teacher.clamp(0, NUM_ACTIONS - 1)
    teacher_valid_under_mask = teacher_in_range & (
        action_mask.gather(1, teacher_safe.unsqueeze(-1)).squeeze(-1).bool()
    )
    teacher_reachable = (
        teacher_valid_under_mask
        & torch.isfinite(teacher_lp)
        & (teacher_lp > -1e7)
    )
    return teacher_reachable.float()


def _safe_distill_loss(
    teacher_lp: torch.Tensor,
    reachable: torch.Tensor,
    distill_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a reachability-masked distillation NLL plus the weighted version.

    The raw teacher log-prob is clamped at ``-50`` before negation so even an
    extremely unlikely-but-reachable teacher action contributes a bounded
    gradient (~= 50) instead of an unbounded one. Without this clamp a single
    very-low-probability teacher action could still spike the loss into the
    tens of millions when the structured distribution is sharply concentrated
    elsewhere.

    Returns ``(distill_loss, distill_loss_weighted)`` where the weighted variant
    folds in the runtime ``distill_coeff``-scaled weight for diagnostics only.
    """
    valid_count = reachable.sum().clamp(min=1.0)
    teacher_lp_for_loss = teacher_lp.clamp_min(-50.0)
    distill_loss = -(teacher_lp_for_loss * reachable * distill_weights).sum() / valid_count
    distill_loss_weighted = distill_loss  # caller scales by distill_coeff for the loss term
    return distill_loss, distill_loss_weighted


def _run_ppo_update(
    model: nn.Module,
    optimizer: Adam,
    buffer: RolloutBuffer,
    return_rms: "RunningMeanStd | None",
    entropy_coeff: float,
    distill_coeff: float,
    config: PPOConfig,
    accum_steps: int,
    effective_batch_size: int,
    device: torch.device,
    use_pin_memory: bool,
    policy_loss_scale: float = 1.0,
    sil_buffer: "WinEpisodeBuffer | None" = None,
) -> _UpdateStats:
    """Run `config.ppo_epochs` passes over the buffer and apply PPO updates.

    Keeps dropout disabled so the PPO ratio compares the same policy function
    that collected `old_log_probs`. Caller is responsible for the entropy-alpha
    step and logging.
    """
    model.eval()
    stats = _UpdateStats(
        policy_losses=[], value_losses=[], survival_losses=[],
        entropies=[], normalized_entropies=[], action_type_entropies=[],
        clip_fracs=[], approx_kls=[],
        valid_action_counts=[], valid_action_type_counts=[],
        distill_losses=[], teacher_match_fractions=[], distill_weight_means=[],
        on_policy_fractions=[],
    )

    stop_update = False
    minibatches_processed = 0
    for _ppo_epoch in range(config.ppo_epochs):
        if stop_update:
            break
        batches = buffer.get_batches(effective_batch_size, device, pin_memory=use_pin_memory)
        optimizer.zero_grad()
        for i, batch in enumerate(batches):
            dist, value_dict = _grammar_distribution(
                model, batch, temperature=config.rollout_temperature
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
            on_policy = ~batch["teacher_forced"].bool()
            on_policy_count = on_policy.float().sum()
            log_ratio = new_log_probs - batch["old_log_probs"]
            ratio = torch.exp(log_ratio)
            advantages = batch["advantages"]
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantages
            ppo_loss_per_state = -torch.min(surr1, surr2)
            if on_policy_count.item() > 0:
                policy_loss = (ppo_loss_per_state * on_policy.float()).sum() / on_policy_count
            else:
                # differentiable zero: keeps params in the graph when a minibatch has no on-policy rows
                policy_loss = new_log_probs.sum() * 0.0

            # Value loss (normalize targets so critic trains in unit-variance space)
            returns_target = batch["returns"]
            if return_rms is not None:
                returns_target = (returns_target - return_rms.mean) / return_rms.std
            value_loss = F.mse_loss(value_dict["expected_score"], returns_target)

            # Ante-survival aux loss (BCE masked by observed antes).
            surv_mask = batch["ante_survival_mask"]
            surv_bce = F.binary_cross_entropy(
                value_dict["ante_survival"],
                batch["ante_survival_target"],
                reduction="none",
            )
            survival_loss = (surv_bce * surv_mask).sum() / surv_mask.sum().clamp(min=1.0)

            policy_frozen = policy_loss_scale <= 0.0

            # Heuristic distillation: NLL of the teacher action under the
            # current structured distribution. Replaces the frozen-reference
            # KL anchor; per-state supervision rather than a snapshot
            # comparison. Sentinel -1 means "no valid teacher this step".
            #
            # Reachability masking: the heuristic teacher occasionally picks
            # a hand action that is legal in the raw env but is not one of
            # the candidate-hand slots the structured distribution emits.
            # ActionGrammarDistribution.log_prob returns the -1e8 floor for
            # those slots; naively including them in the NLL average produced
            # distill_loss values in the 17M-21M range that dominated PPO.
            # We drop those rows from the loss and log the skipped fraction
            # so the silent-label-drop is visible in TensorBoard.
            #
            # The teacher log_prob is a full second grammar evaluation per
            # minibatch, so it only runs when distillation can actually
            # contribute to this update's objective.
            distill_loss: torch.Tensor | None = None
            teacher_match = 0.0
            distill_weight_mean = 0.0
            teacher_label_present_fraction = 0.0
            teacher_action_mask_valid_fraction = 0.0
            teacher_reachable_fraction = 0.0
            teacher_unreachable_fraction = 0.0
            teacher_lp_mean_reachable = 0.0
            unreachable_family_fracs: dict[str, float] = {
                fam_name: 0.0 for fam_name in _ACTION_FAMILY_NAMES
            }
            distill_loss_weighted_value = 0.0
            if distill_coeff > 0.0 and not policy_frozen:
                teacher = batch["teacher_actions"]
                teacher_safe = teacher.clamp(0, NUM_ACTIONS - 1)
                teacher_lp = dist.log_prob(teacher_safe)
                reachable = _teacher_reachability_mask(
                    teacher, teacher_lp, batch["action_mask"]
                )
                reachable_count = reachable.sum().clamp(min=1.0)
                distill_weights = batch["distill_weights"]
                # Per-step weighting: regret signals (out-of-candidates,
                # mismatched planet use) up-weight teacher NLL on those steps
                # so distillation pressure concentrates where the policy
                # diverged from the heuristic on a high-stakes decision.
                distill_loss, _ = _safe_distill_loss(teacher_lp, reachable, distill_weights)
                with torch.no_grad():
                    policy_choice = dist.mode() if hasattr(dist, "mode") else batch["actions"]
                    teacher_match = (
                        ((policy_choice == teacher_safe).float() * reachable).sum()
                        / reachable_count
                    ).item()
                    distill_weight_mean = (
                        (distill_weights * reachable).sum() / reachable_count
                    ).item()

                    # Diagnostics: how often did the teacher produce a label
                    # at all, how often was it reachable, and what was the
                    # mean reachable log-prob. The unreachable fraction is the
                    # key metric for spotting -1e8 floor contamination.
                    teacher_present_bool = teacher >= 0
                    teacher_present = teacher_present_bool.float()
                    teacher_present_count = teacher_present.sum().clamp(min=1.0)
                    # "label present" = fraction of ALL steps with a teacher label.
                    teacher_label_present_fraction = (
                        teacher_present.sum() / max(1.0, float(teacher.numel()))
                    ).item()
                    # "action mask valid" = of the present labels, how many are
                    # legal under the current step's action_mask.
                    teacher_safe_t = teacher.clamp(0, NUM_ACTIONS - 1)
                    teacher_mask_valid = teacher_present_bool & batch["action_mask"].gather(
                        1, teacher_safe_t.unsqueeze(-1)
                    ).squeeze(-1).bool()
                    teacher_action_mask_valid_fraction = (
                        teacher_mask_valid.float().sum() / teacher_present_count
                    ).item()
                    teacher_reachable_fraction = (
                        reachable.sum() / teacher_present_count
                    ).item()
                    teacher_unreachable_fraction = 1.0 - teacher_reachable_fraction
                    teacher_lp_mean_reachable = (
                        (teacher_lp * reachable).sum() / reachable_count
                    ).item()
                    # Break the *unreachable* (present-but-not-reachable) labels
                    # down by action family so distillation diagnostics can tell
                    # hand-subset mismatches from shop/pack/blind unreachability.
                    unreachable_mask = teacher_present_bool & (~reachable.bool())
                    unreachable_denom = unreachable_mask.float().sum().clamp(min=1.0)
                    family_ids = _ACTION_ID_TO_FAMILY.to(device=teacher.device)[teacher_safe_t]
                    for fam_idx, fam_name in enumerate(_ACTION_FAMILY_NAMES):
                        unreachable_family_fracs[fam_name] = (
                            (unreachable_mask & (family_ids == fam_idx)).float().sum()
                            / unreachable_denom
                        ).item()
                    distill_loss_weighted_value = (
                        float(distill_loss.item()) * distill_coeff
                    )

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
                        "Critic warmup with no value_head module: nothing trains "
                        "while the policy is frozen."
                    )
                    _run_ppo_update._frozen_no_value_head_warned = True  # type: ignore[attr-defined]
            else:
                policy_objective_loss = (
                    policy_loss
                    - entropy_coeff
                    * (normalized_entropy + config.action_type_entropy_scale * normalized_action_type_entropy)
                )
                if distill_loss is not None:
                    policy_objective_loss = policy_objective_loss + distill_coeff * distill_loss
                # Self-imitation term: advantage-gated NLL of the agent's own
                # buffered winning actions, sharing this micro-batch's backward
                # and the target-kl guard.
                sil_result = _sample_sil_loss(model, sil_buffer, config, device)
                if sil_result is not None:
                    sil_loss, sil_advantage_mean, sil_gate_mean = sil_result
                    policy_objective_loss = policy_objective_loss + config.sil_coeff * sil_loss
                    stats.sil_losses.append(sil_loss.item())
                    if sil_advantage_mean is not None:
                        stats.sil_advantage_means.append(sil_advantage_mean)
                    if sil_gate_mean is not None:
                        stats.sil_gate_means.append(sil_gate_mean)
                policy_objective_loss = policy_objective_loss * policy_loss_scale

                if config.critic_updates_trunk:
                    total_loss = (
                        policy_objective_loss + critic_loss
                    ).div(accum_steps)
                    total_loss.backward()
                else:
                    value_params = _value_head_parameters(model)
                    policy_objective_loss.div(accum_steps).backward(retain_graph=bool(value_params))
                    if value_params:
                        _accumulate_critic_grads_into_value_head(
                            critic_loss.div(accum_steps), value_params
                        )

            # Step every accum_steps micro-batches (or on last batch)
            if (i + 1) % accum_steps == 0 or (i + 1) == len(batches):
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                on_policy_fraction = (on_policy_count / max(1, int(on_policy.numel()))).item()
                if on_policy_count.item() > 0:
                    on_policy_float = on_policy.float()
                    clip_frac = (
                        (((ratio - 1.0).abs() > config.clip_epsilon).float() * on_policy_float).sum()
                        / on_policy_count
                    ).item()
                    approx_kl = ((((ratio - 1.0) - log_ratio) * on_policy_float).sum() / on_policy_count).item()
                else:
                    clip_frac = 0.0
                    approx_kl = 0.0
                valid_action_count_mean = valid_action_counts.float().mean().item()
                if on_policy_count.item() > 0:
                    on_policy_adv = advantages[on_policy]
                    stats.on_policy_advantage_means.append(on_policy_adv.mean().item())
                    stats.on_policy_advantage_stds.append(on_policy_adv.std().item())
                    stats.on_policy_positive_advantage_fractions.append(
                        (on_policy_adv > 0).float().mean().item()
                    )
                    on_policy_returns = batch["returns"][on_policy]
                    stats.on_policy_return_means.append(on_policy_returns.mean().item())
                else:
                    stats.on_policy_advantage_means.append(0.0)
                    stats.on_policy_advantage_stds.append(0.0)
                    stats.on_policy_positive_advantage_fractions.append(0.0)
                    stats.on_policy_return_means.append(0.0)
            stats.policy_losses.append(policy_loss.item())
            stats.value_losses.append(value_loss.item())
            stats.survival_losses.append(survival_loss.item())
            stats.entropies.append(entropy.item())
            stats.normalized_entropies.append(normalized_entropy.item())
            stats.action_type_entropies.append(normalized_action_type_entropy.item())
            stats.clip_fracs.append(clip_frac)
            stats.approx_kls.append(approx_kl)
            stats.valid_action_counts.append(valid_action_count_mean)
            stats.valid_action_type_counts.append(valid_action_type_count_mean)
            stats.distill_losses.append(0.0 if distill_loss is None else distill_loss.item())
            stats.distill_losses_weighted.append(distill_loss_weighted_value)
            stats.teacher_match_fractions.append(teacher_match)
            stats.distill_weight_means.append(distill_weight_mean)
            stats.teacher_label_present_fractions.append(teacher_label_present_fraction)
            stats.teacher_action_mask_valid_fractions.append(teacher_action_mask_valid_fraction)
            stats.teacher_reachable_fractions.append(teacher_reachable_fraction)
            stats.teacher_unreachable_fractions.append(teacher_unreachable_fraction)
            stats.teacher_lp_mean_reachable.append(teacher_lp_mean_reachable)
            for fam_name, frac in unreachable_family_fracs.items():
                stats.teacher_unreachable_family_fractions.setdefault(fam_name, []).append(frac)
            stats.on_policy_fractions.append(on_policy_fraction)
            minibatches_processed += 1

            if config.target_kl is not None and approx_kl > config.target_kl:
                logger.debug(
                    "Stopping PPO update early: approx_kl=%.5f exceeded target_kl=%.5f",
                    approx_kl,
                    config.target_kl,
                )
                stop_update = True
                break

    stats.ppo_minibatches_processed.append(minibatches_processed)
    return stats


def _run_dagger_bc_update(
    model: nn.Module,
    optimizer: Adam,
    buffer: RolloutBuffer,
    coeff: float,
    config: PPOConfig,
    accum_steps: int,
    effective_batch_size: int,
    device: torch.device,
    use_pin_memory: bool,
) -> tuple[list[float], list[float]]:
    """Run online behavior cloning over the rollout states using teacher labels."""
    if config.dagger_bc_epochs <= 0 or coeff <= 0.0:
        return [], []

    model.eval()
    losses: list[float] = []
    teacher_matches: list[float] = []
    original_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    try:
        if config.dagger_bc_lr_mult != 1.0:
            for group, lr in zip(optimizer.param_groups, original_lrs, strict=True):
                group["lr"] = lr * config.dagger_bc_lr_mult

        for _epoch in range(config.dagger_bc_epochs):
            batches = buffer.get_batches(effective_batch_size, device, pin_memory=use_pin_memory)
            optimizer.zero_grad()
            for i, batch in enumerate(batches):
                teacher = batch["teacher_actions"]
                teacher_present = (teacher >= 0).float()
                if teacher_present.sum().item() <= 0:
                    continue

                dist, _value_dict = _grammar_distribution(
                    model,
                    batch,
                    temperature=config.rollout_temperature,
                )
                teacher_safe = teacher.clamp(0, NUM_ACTIONS - 1)
                teacher_lp = dist.log_prob(teacher_safe)
                # Mirror the PPO update's reachability mask so DAgger BC
                # also ignores -1e8 floor rows from candidate-hand mismatches.
                reachable = _teacher_reachability_mask(
                    teacher, teacher_lp, batch["action_mask"]
                )
                if reachable.sum().item() <= 0:
                    continue
                distill_weights = batch["distill_weights"]
                bc_loss, _ = _safe_distill_loss(teacher_lp, reachable, distill_weights)
                (coeff * bc_loss).div(accum_steps).backward()

                if (i + 1) % accum_steps == 0 or (i + 1) == len(batches):
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                with torch.no_grad():
                    reachable_count = reachable.sum().clamp(min=1.0)
                    if hasattr(dist, "mode"):
                        mode = dist.mode()
                        teacher_match = (
                            ((mode == teacher_safe).float() * reachable).sum()
                            / reachable_count
                        ).item()
                    else:
                        teacher_match = float("nan")
                losses.append(bc_loss.item())
                teacher_matches.append(teacher_match)
    finally:
        for group, lr in zip(optimizer.param_groups, original_lrs, strict=True):
            group["lr"] = lr

    return losses, teacher_matches


def _sample_sil_loss(
    model: nn.Module,
    sil_buffer: "WinEpisodeBuffer | None",
    config: PPOConfig,
    device: torch.device,
) -> "tuple[torch.Tensor, float | None, float | None] | None":
    """Advantage-gated NLL of buffered winning actions, or None.

    Returns ``(loss, advantage_mean, gate_mean)``; the metrics are None when
    gating is disabled. Sampled fresh per PPO micro-batch and added to the
    combined objective there — a separate backward/step would let Adam's
    gradient renormalization largely cancel sil_coeff. Non-finite log-probs
    (candidate-hand support drift, same failure mode the teacher
    reachability mask guards) are dropped from the mean rather than
    poisoning the gradient. Returns None when SIL is disabled, the buffer
    is short of sil_min_episodes, or no sampled row is reachable.

    With sil_advantage_gating each transition's NLL is weighted by
    min((R - V)+ / sil_advantage_clip, 1): the critic's shortfall on the
    stored return-to-go decides how hard to imitate, so well-predicted
    (routine) wins contribute nothing and the term self-decays instead of
    behavior-cloning the policy's average recent win. V is detached — the
    gate must not train the critic toward pretending wins are expected.
    """
    if sil_buffer is None or config.sil_coeff <= 0.0:
        return None
    if sil_buffer.num_episodes < config.sil_min_episodes:
        return None
    batch = sil_buffer.sample(config.sil_batch_size, device)
    dist, value_dict = _grammar_distribution(
        model,
        batch,
        temperature=config.rollout_temperature,
    )
    log_probs = dist.log_prob(batch["actions"])
    finite = (log_probs > -1e7).float()
    denom = finite.sum()
    if denom.item() <= 0:
        return None
    if not config.sil_advantage_gating:
        return -(log_probs * finite).sum() / denom, None, None
    advantage = (batch["returns"] - value_dict["expected_score"].detach()).clamp_min(0.0)
    gate = (advantage / config.sil_advantage_clip).clamp_max(1.0)
    loss = -(log_probs * gate * finite).sum() / denom
    advantage_mean = ((advantage * finite).sum() / denom).item()
    gate_mean = ((gate * finite).sum() / denom).item()
    return loss, advantage_mean, gate_mean


def _scheduled_teacher_rollout_prob(config: PPOConfig, progress: float) -> float:
    """Return the teacher-forcing probability for the current training progress."""
    start = float(config.teacher_rollout_prob)
    final = config.teacher_rollout_final_prob
    if final is None:
        return start

    progress = min(max(float(progress), 0.0), 1.0)
    warmup = float(config.teacher_rollout_warmup_fraction)
    if progress <= warmup:
        return start

    decay = float(config.teacher_rollout_decay_fraction)
    if decay <= 0.0:
        return float(final)

    anneal_progress = min(max((progress - warmup) / decay, 0.0), 1.0)
    return start + (float(final) - start) * anneal_progress


def _write_rollout_scalars(writer, update_count: int, rm: "_RolloutMetrics") -> None:
    """Write the rollout-collected metrics for one update to TensorBoard."""
    writer.add_scalar("debug/chosen_action_prob_mean", _safe_mean(rm.chosen_action_probs), update_count)
    # This is the max ACTION-TYPE probability (over the grammar's action-type
    # distribution), not a true flat-action max.
    writer.add_scalar("debug/max_action_type_prob_mean", _safe_mean(rm.max_action_probs), update_count)
    writer.add_scalar("debug/teacher_rollout_used_fraction", _safe_mean(rm.teacher_rollout_used), update_count)
    writer.add_scalar("debug/step_reward_mean", _safe_mean(rm.step_rewards), update_count)
    writer.add_scalar("rollout/progress_rate", _safe_mean(rm.progress_flags), update_count)
    writer.add_scalar("rollout/steps_since_progress_mean", _safe_mean(rm.steps_since_progress), update_count)
    if rm.steps_since_progress:
        writer.add_scalar(
            "rollout/steps_since_progress_max",
            float(np.max(rm.steps_since_progress)),
            update_count,
        )
    writer.add_scalar("rollout/done_rate", _safe_mean(rm.done_flags), update_count)
    writer.add_scalar("rollout/terminated_rate", _safe_mean(rm.terminated_flags), update_count)
    writer.add_scalar("rollout/truncated_rate", _safe_mean(rm.truncated_flags), update_count)
    writer.add_scalar("subphase/choose_action_fraction", _safe_mean(rm.pre_choose_action_flags), update_count)
    total_play_discard = rm.play_subset_count + rm.discard_subset_count
    if total_play_discard > 0:
        writer.add_scalar("hand_choice/play_fraction", rm.play_subset_count / total_play_discard, update_count)
        writer.add_scalar("hand_choice/discard_fraction", rm.discard_subset_count / total_play_discard, update_count)
    else:
        writer.add_scalar("hand_choice/play_fraction", float("nan"), update_count)
        writer.add_scalar("hand_choice/discard_fraction", float("nan"), update_count)
    total_action_count = sum(rm.action_type_counts.values())
    if total_action_count > 0:
        for action_type in ActionType:
            writer.add_scalar(
                f"actions/{action_type.value}_fraction",
                rm.action_type_counts[action_type.value] / total_action_count,
                update_count,
            )

    writer.add_scalar("hand/play_observed_count", float(len(rm.hand_play_observed)), update_count)
    writer.add_scalar("hand/not_in_candidates_fraction", _safe_mean(rm.hand_play_not_in_candidates), update_count)
    writer.add_scalar("hand/in_candidates_fraction", _safe_mean(rm.hand_play_in_candidates), update_count)
    writer.add_scalar("hand/top1_match_fraction", _safe_mean(rm.hand_play_top1), update_count)
    writer.add_scalar("hand/top3_match_fraction", _safe_mean(rm.hand_play_top3), update_count)
    writer.add_scalar("hand/candidate_value_ratio_mean", _safe_mean(rm.hand_play_value_ratios), update_count)
    _write_counter_fractions(writer, "hand/chosen", rm.hand_chosen_counts, update_count)
    _write_counter_fractions(writer, "hand/best", rm.hand_best_counts, update_count)

    writer.add_scalar("planet/use_count", float(len(rm.planet_use_observed)), update_count)
    writer.add_scalar("planet/use_played_hand_fraction", _safe_mean(rm.planet_use_played_hand), update_count)
    writer.add_scalar("planet/use_play_share_mean", _safe_mean(rm.planet_use_play_share), update_count)
    writer.add_scalar("planet/use_main_hand_match_fraction", _safe_mean(rm.planet_use_main_hand_match), update_count)
    # Unmatched / not-played fractions expose the planet-churn pathology
    # directly: a rising unmatched fraction with falling match fraction means
    # the policy is engaging with planets that do not align with its build.
    use_unmatched = [0.0 if v else 1.0 for v in rm.planet_use_main_hand_match]
    claim_unmatched = [0.0 if v else 1.0 for v in rm.planet_claim_main_hand_match]
    use_not_played = [0.0 if v else 1.0 for v in rm.planet_use_played_hand]
    claim_not_played = [0.0 if v else 1.0 for v in rm.planet_claim_played_hand]
    writer.add_scalar("planet/use_unmatched_fraction", _safe_mean(use_unmatched), update_count)
    writer.add_scalar("planet/claim_unmatched_fraction", _safe_mean(claim_unmatched), update_count)
    writer.add_scalar("planet/use_not_played_hand_fraction", _safe_mean(use_not_played), update_count)
    writer.add_scalar("planet/claim_not_played_hand_fraction", _safe_mean(claim_not_played), update_count)
    episode_count = max(int(sum(rm.done_flags)), 1)
    writer.add_scalar("planet/use_count_per_episode", float(len(rm.planet_use_observed)) / episode_count, update_count)
    writer.add_scalar(
        "planet/claim_count_per_episode",
        float(len(rm.planet_claim_observed)) / episode_count,
        update_count,
    )
    _write_counter_fractions(writer, "planet/use_key", rm.planet_use_key_counts, update_count)

    writer.add_scalar("planet/claim_count", float(len(rm.planet_claim_observed)), update_count)
    writer.add_scalar("planet/claim_played_hand_fraction", _safe_mean(rm.planet_claim_played_hand), update_count)
    writer.add_scalar("planet/claim_play_share_mean", _safe_mean(rm.planet_claim_play_share), update_count)
    writer.add_scalar(
        "planet/claim_main_hand_match_fraction",
        _safe_mean(rm.planet_claim_main_hand_match),
        update_count,
    )
    _write_counter_fractions(writer, "planet/claim_key", rm.planet_claim_key_counts, update_count)

    writer.add_scalar("pack/planet_skip_fraction", _safe_mean(rm.planet_pack_skip), update_count)
    _write_counter_fractions(writer, "pack/skip_state", rm.pack_skip_state_counts, update_count)

    for component_name, component_values in rm.reward_component_values.items():
        writer.add_scalar(
            f"reward/{component_name.removeprefix('reward_')}_mean",
            _safe_mean(component_values),
            update_count,
        )

    # Per-step dense shaping total = total reward minus the terminal component.
    # Means are linear over the same per-step samples, so this is the mean of
    # (total - terminal). Lets us confirm the dense_reward_scale ablation drops
    # dense shaping roughly proportionally while reward/terminal_mean (logged
    # once by the per-component loop above) is unchanged.
    total_vals = rm.reward_component_values.get("reward_total")
    terminal_vals = rm.reward_component_values.get("reward_terminal")
    if total_vals:
        dense_total_mean = _safe_mean(total_vals) - (
            _safe_mean(terminal_vals) if terminal_vals else 0.0
        )
        writer.add_scalar("reward/dense_total_mean", dense_total_mean, update_count)

    # Episode-level fraction of return magnitude contributed by the terminal
    # outcome: abs(terminal_sum) / (abs(terminal_sum) + abs(dense_sum)). This is
    # the quantity Phase 2 targets (> 0.5). Per-step means (logged above) cannot
    # answer "how much of what the agent optimizes is winning", because terminal
    # rewards fire on ~1 step while dense shaping fires on every step; the
    # episode-level magnitude ratio is the correct measure. ~0.13 in recover_v3.
    if total_vals and terminal_vals:
        terminal_sum = float(np.sum(np.abs(terminal_vals)))
        dense_sum = float(np.sum(np.abs(np.asarray(total_vals)))) - terminal_sum
        denom = terminal_sum + dense_sum
        if denom > 1e-9:
            writer.add_scalar(
                "reward/terminal_fraction_of_return",
                terminal_sum / denom,
                update_count,
            )

    # Reward group totals, surface how much shaping comes from already-solved
    # local hand play vs strategic shop/joker/economy signal, so a run can be
    # diagnosed when local play drowns out the strategic loop.
    group_sums: defaultdict = defaultdict(float)
    for name, vals in rm.reward_component_values.items():
        component = name.removeprefix("reward_")
        if component == "total":
            continue
        group = _COMPONENT_GROUP.get(component)
        if group is None:
            continue
        group_sums[group] += _safe_mean(vals)
    for group, value in group_sums.items():
        writer.add_scalar(f"reward/group_{group}_total_mean", value, update_count)


def _write_counter_fractions(writer, prefix: str, counter: Counter, update_count: int) -> None:
    total = sum(counter.values())
    if total <= 0:
        return
    for key, count in counter.items():
        writer.add_scalar(f"{prefix}/{_sanitize_tag_part(str(key))}_fraction", count / total, update_count)


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
            "ppo_epochs", "mini_batch_size", "clip_epsilon", "gae_lambda",
            "rollout_temperature", "action_type_entropy_scale", "gamma",
            "target_entropy", "entropy_ema_beta", "adaptive_entropy",
        ):
            config_fields[key] = getattr(config, key)
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


# Retained for unit tests; production uses _per_state_normalized_entropy(...).mean() (~450-451).
def _mean_normalized_entropy(entropy_per_state: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Return the batch mean of per-state normalized entropy."""
    return _per_state_normalized_entropy(entropy_per_state, action_mask).mean()


# Retained for unit tests; production uses the approx-KL estimator ((ratio-1)-log_ratio) (~554).
def _masked_kl_divergence(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Return mean KL(policy || reference) over the valid-action support.

    `torch.distributions.kl_divergence(Categorical, Categorical)` can report
    `inf` when a valid tail action underflows to zero probability in the
    reference policy. Computing KL from masked log-softmax values keeps the
    valid support aligned and stays finite for extreme-yet-valid logits.
    """
    masked_policy_logits = policy_logits.masked_fill(action_mask == 0, -1e8)
    masked_reference_logits = reference_logits.masked_fill(action_mask == 0, -1e8)
    log_policy = F.log_softmax(masked_policy_logits, dim=-1)
    log_reference = F.log_softmax(masked_reference_logits, dim=-1)
    policy_probs = log_policy.exp()
    valid_mask = (action_mask > 0).to(log_policy.dtype)
    kl_per_state = (policy_probs * (log_policy - log_reference) * valid_mask).sum(dim=-1)
    return kl_per_state.mean()


# Retained for unit tests; production uses dist.normalized_action_type_entropy() (~452).
def _mean_normalized_action_type_entropy(action_probs: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Return mean entropy over action types, normalized by valid type count.

    This complements flat action entropy in highly imbalanced spaces where a
    few action families contain most of the valid actions.
    """
    type_index = _ACTION_ID_TO_TYPE_INDEX.to(device=action_probs.device)
    expanded_type_index = type_index.unsqueeze(0).expand(action_probs.shape[0], -1)

    type_probs = torch.zeros(
        action_probs.shape[0],
        len(_ACTION_TYPES),
        dtype=action_probs.dtype,
        device=action_probs.device,
    )
    type_probs.scatter_add_(1, expanded_type_index, action_probs)

    valid_action_mask = (action_mask > 0).to(action_probs.dtype)
    valid_type_counts = torch.zeros_like(type_probs)
    valid_type_counts.scatter_add_(1, expanded_type_index, valid_action_mask)
    valid_type_counts = (valid_type_counts > 0).sum(dim=-1).to(action_probs.dtype)

    safe_type_probs = type_probs.clamp_min(1e-12)
    type_entropy = -(safe_type_probs * safe_type_probs.log()).sum(dim=-1)
    max_type_entropy = torch.log(valid_type_counts.clamp_min(2.0))
    normalized_type_entropy = torch.where(
        valid_type_counts > 1.0,
        type_entropy / max_type_entropy,
        torch.zeros_like(type_entropy),
    )
    return normalized_type_entropy.mean()


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


_REGRET_DISTILL_WEIGHT_MAX = 3.0


def _compute_distill_weight(infos: dict, env_idx: int, *, done: bool) -> float:
    """Return a per-step distillation weight in [1.0, _REGRET_DISTILL_WEIGHT_MAX].

    Weights >1 amplify the teacher-NLL gradient on steps where the agent
    diverged from the heuristic on a high-stakes decision: an
    out-of-candidates hand subset, a low-ratio in-candidates play, or a
    mismatched planet use. Steps without diagnostics keep weight 1.0.
    """
    weight = 1.0
    action_type = _extract_step_info_value(infos, "action_type", env_idx, done=done, default="")
    if action_type == "play_subset":
        if bool(_extract_step_info_value(infos, "hand_play_not_in_candidates", env_idx, done=done, default=False)):
            weight = max(weight, _REGRET_DISTILL_WEIGHT_MAX)
        else:
            ratio = _extract_step_info_value(
                infos, "hand_play_candidate_value_ratio", env_idx, done=done, default=None
            )
            if ratio is not None:
                gap = max(0.0, 1.0 - float(ratio))
                weight = max(weight, 1.0 + (_REGRET_DISTILL_WEIGHT_MAX - 1.0) * gap)
    planet_used = bool(
        _extract_step_info_value(infos, "planet_use_observed", env_idx, done=done, default=False)
    )
    if planet_used:
        match = bool(
            _extract_step_info_value(infos, "planet_use_main_hand_match", env_idx, done=done, default=False)
        )
        if not match:
            weight = max(weight, _REGRET_DISTILL_WEIGHT_MAX)
    return weight


def _record_action_diagnostics(rm: _RolloutMetrics, infos: dict, env_idx: int, *, done: bool) -> None:
    """Aggregate optional env-provided decision-quality diagnostics."""
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
            rm.hand_play_top1.append(float(_extract_step_info_value(
                infos,
                "hand_play_top1",
                env_idx,
                done=done,
                default=False,
            )))
            rm.hand_play_top3.append(float(_extract_step_info_value(
                infos,
                "hand_play_top3",
                env_idx,
                done=done,
                default=False,
            )))
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
        rm.planet_use_played_hand.append(float(_extract_step_info_value(
            infos,
            "planet_use_played_hand",
            env_idx,
            done=done,
            default=False,
        )))
        rm.planet_use_play_share.append(float(_extract_step_info_value(
            infos,
            "planet_use_play_share",
            env_idx,
            done=done,
            default=0.0,
        )))
        rm.planet_use_main_hand_match.append(float(_extract_step_info_value(
            infos,
            "planet_use_main_hand_match",
            env_idx,
            done=done,
            default=False,
        )))
        planet_key = _extract_step_info_value(infos, "planet_use_key", env_idx, done=done, default="")
        if planet_key:
            rm.planet_use_key_counts[str(planet_key)] += 1

    if _extract_step_info_value(infos, "planet_claim_observed", env_idx, done=done, default=False):
        rm.planet_claim_observed.append(1.0)
        rm.planet_claim_played_hand.append(float(_extract_step_info_value(
            infos,
            "planet_claim_played_hand",
            env_idx,
            done=done,
            default=False,
        )))
        rm.planet_claim_play_share.append(float(_extract_step_info_value(
            infos,
            "planet_claim_play_share",
            env_idx,
            done=done,
            default=0.0,
        )))
        rm.planet_claim_main_hand_match.append(float(_extract_step_info_value(
            infos,
            "planet_claim_main_hand_match",
            env_idx,
            done=done,
            default=False,
        )))
        planet_key = _extract_step_info_value(infos, "planet_claim_key", env_idx, done=done, default="")
        if planet_key:
            rm.planet_claim_key_counts[str(planet_key)] += 1

    if _extract_step_info_value(infos, "planet_pack_skip", env_idx, done=done, default=None) is not None:
        rm.planet_pack_skip.append(float(_extract_step_info_value(
            infos,
            "planet_pack_skip",
            env_idx,
            done=done,
            default=False,
        )))
        pack_state_name = _extract_step_info_value(infos, "pack_skip_state_name", env_idx, done=done, default="")
        if pack_state_name:
            rm.pack_skip_state_counts[str(pack_state_name)] += 1


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
            np.stack([obs["token_types"] for obs in obs_list]), dtype=torch.long, device=device,
        ),
        "scalars": torch.tensor(np.stack([obs["scalars"] for obs in obs_list]), dtype=torch.float32, device=device),
        "attention_mask": torch.tensor(
            np.stack([obs["attention_mask"] for obs in obs_list]), dtype=torch.long, device=device,
        ),
        "action_mask": torch.tensor(
            np.stack([obs["action_mask"] for obs in obs_list]), dtype=torch.float32, device=device,
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

    _validate_ppo_config(config)

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
        resume_state = load_ppo_resume_payload(resume_path, device)
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
            model, pretrained_path, device, reinit_value_head=config.reinit_value_head
        )
        if config.reinit_value_head:
            logger.info(
                "Loaded pretrained model from %s with value head reinitialized "
                "(reward function changed; critic will warm up from scratch).",
                pretrained_path,
            )
        else:
            logger.info(f"Loaded pretrained model from {pretrained_path}")
    if config.heuristic_distill_coeff > 0.0:
        floor = min(config.heuristic_distill_min, config.heuristic_distill_coeff)
        logger.info(
            "Heuristic distillation active (coeff=%.4f, floor=%.4f)",
            config.heuristic_distill_coeff,
            floor,
        )
    else:
        logger.info(
            "Heuristic distillation disabled (coeff=%.4f <= 0); "
            "ppo/distill_coeff will be 0.0 for the whole run.",
            config.heuristic_distill_coeff,
        )
    logger.info("Rollout temperature: %.3f (applied to rollout, training, and bootstrap)",
                config.rollout_temperature)
    # Phase 3.3: discount-horizon diagnostic. At gamma=0.99 a terminal reward
    # is discounted to ~0.22 at 150 steps, ~0.05 at 300, the early-game shop
    # economy decisions that matter most for winning are nearly invisible. For
    # win_ante >= 6 (long episodes), 0.997 is recommended (0.997^300 ~ 0.41).
    effective_win_ante = config.win_ante if config.win_ante is not None else 8
    if effective_win_ante >= 6 and config.gamma < 0.995:
        logger.warning(
            "gamma=%.4f with win_ante=%d: terminal reward is discounted to "
            "%.3f at ~300 steps. Consider --gamma 0.997 for long-horizon runs "
            "(Phase 3.3).",
            config.gamma,
            effective_win_ante,
            config.gamma ** 300,
        )
    if config.teacher_rollout_prob > 0.0:
        logger.info("Teacher-guided rollout probability: %.3f", config.teacher_rollout_prob)
    if config.teacher_rollout_final_prob is not None:
        logger.info(
            "Teacher-guided rollout schedule: %.3f -> %.3f after warmup %.2f over decay %.2f",
            config.teacher_rollout_prob,
            config.teacher_rollout_final_prob,
            config.teacher_rollout_warmup_fraction,
            config.teacher_rollout_decay_fraction,
        )
    if config.dagger_bc_epochs > 0 and config.dagger_bc_coeff > 0.0:
        logger.info(
            "Online DAgger BC active (epochs=%d, coeff=%.4f, lr_mult=%.2f)",
            config.dagger_bc_epochs,
            config.dagger_bc_coeff,
            config.dagger_bc_lr_mult,
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

    # Sync gamma into the reward config so potential-based shaping (Phase 2)
    # telescopes under the same discount the value function regresses.
    if config.reward_config is not None:
        config.reward_config.gamma = config.gamma

    # Create vectorized environments. Each env replays a deterministic
    # game-seed stream from its constructor seed, so resuming with plain
    # 0..N-1 seeds replays the same opening game library every leg and the
    # critic overfits to it (novel seeds then produce systematically noisy
    # advantages — the env-count-change collapse). Offset the seeds by the
    # resumed update counter so every leg trains on fresh streams; fresh
    # runs keep base 0 and are byte-identical to prior behavior.
    env_seed_base = 0
    if resume_state is not None:
        env_seed_base = int(resume_state.get("update_count", 0)) * 1_000_003
        logger.info("Env seed base for this leg: %d", env_seed_base)
    # The heuristic teacher runs twice per env step; skip it entirely when no
    # training objective consumes its labels (distillation, DAgger BC, or
    # teacher-forced rollouts). Envs then emit the -1 "no teacher" sentinel.
    teacher_needed = (
        config.heuristic_distill_coeff > 0.0
        or (config.dagger_bc_epochs > 0 and config.dagger_bc_coeff > 0.0)
        or config.teacher_rollout_prob > 0.0
        or (config.teacher_rollout_final_prob or 0.0) > 0.0
    )
    if not teacher_needed:
        logger.info(
            "Heuristic teacher disabled in training envs (no distill/DAgger/"
            "teacher-forcing consumers); teacher metrics will read 0/-1."
        )
    vec_env = _make_vectorized_envs(
        config.num_envs, data, vocab,
        stake=config.stake,
        max_no_progress_steps=config.max_no_progress_steps,
        use_async=config.async_envs,
        win_ante=config.win_ante,
        reward_config=config.reward_config,
        env_seed_base=env_seed_base,
        enable_teacher=teacher_needed,
    )
    obs_dict, reset_info = vec_env.reset()
    # Pre-allocate obs tensors for batched inference
    obs_buf = _ObsBuffer(config.num_envs, device)
    obs_buf.update(obs_dict)
    teacher_action_buf = np.asarray(
        [
            int(_extract_vector_info_value(reset_info, "teacher_action", idx, -1))
            for idx in range(config.num_envs)
        ],
        dtype=np.int64,
    )

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
        elif (
            saved_entropy_coeff is not None
            and abs(float(saved_entropy_coeff) - entropy_coeff) > 1e-12
        ):
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
    # Anneal horizon for fraction-of-training schedules (distill coeff,
    # teacher-rollout prob). Resolved AFTER the resume block finalizes
    # total_timesteps, persisted in every checkpoint so later legs keep the
    # original horizon regardless of num_envs/update-target changes.
    schedule_total_steps = _resolve_schedule_total_steps(config, resume_state)
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_wins: list[bool] = []
    episode_stalls: list[bool] = []
    episode_antes: list[int] = []
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
    _rolling_win_rates: list[float] = []
    _rolling_ep_rewards: list[float] = []
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
    # Self-imitation: FIFO buffer of complete winning episodes plus a per-env
    # tracker that assembles them across rollout boundaries (episodes are
    # ~100 steps and routinely outlive a single rollout).
    sil_buffer: WinEpisodeBuffer | None = None
    sil_tracker: SILEpisodeTracker | None = None
    if config.sil_coeff > 0.0:
        sil_buffer = WinEpisodeBuffer(config.sil_buffer_episodes)
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
            rollout_progress = total_steps / max(1, schedule_total_steps)
            teacher_rollout_prob_now = _scheduled_teacher_rollout_prob(config, rollout_progress)

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
                        temperature=config.rollout_temperature,
                    )
                    actions = dist.sample()
                    use_teacher = torch.zeros(config.num_envs, dtype=torch.bool, device=device)
                    if teacher_rollout_prob_now > 0.0:
                        teacher_actions_t = torch.as_tensor(teacher_action_buf, dtype=torch.long, device=device)
                        teacher_valid = (
                            (teacher_actions_t >= 0)
                            & (teacher_actions_t < NUM_ACTIONS)
                            & obs_buf.action_mask.gather(
                                1,
                                teacher_actions_t.clamp(0, NUM_ACTIONS - 1).unsqueeze(-1),
                            ).squeeze(-1).bool()
                        )
                        use_teacher = (
                            torch.rand(config.num_envs, device=device) < teacher_rollout_prob_now
                        ) & teacher_valid
                        actions = torch.where(use_teacher, teacher_actions_t, actions)
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
                use_teacher_np = use_teacher.cpu().numpy()

                # Step all envs at once
                next_obs_dict, rewards, terminated, truncated, infos = vec_env.step(actions_np)
                dones = terminated | truncated
                bootstrap_values_np = np.zeros(config.num_envs, dtype=np.float32)
                teacher_actions_np = np.asarray(
                    [
                        int(
                            _extract_step_info_value(
                                infos, "teacher_action", env_idx, done=bool(dones[env_idx]), default=-1
                            )
                        )
                        for env_idx in range(config.num_envs)
                    ],
                    dtype=np.int64,
                )
                distill_weights_np = np.asarray(
                    [
                        _compute_distill_weight(infos, env_idx, done=bool(dones[env_idx]))
                        for env_idx in range(config.num_envs)
                    ],
                    dtype=np.float32,
                )

                if np.any(truncated) and "final_obs" in infos:
                    final_obs_arr = infos["final_obs"]
                    truncated_indices = [idx for idx in np.where(truncated)[0] if final_obs_arr[idx] is not None]
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
                    terminated=terminated,
                    truncated=truncated,
                    bootstrap_values=bootstrap_values_np,
                    teacher_actions=teacher_actions_np,
                    distill_weights=distill_weights_np,
                    teacher_forced=use_teacher_np,
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
                rm.teacher_rollout_used.extend(use_teacher_np.astype(np.float64).tolist())
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
                        float(_extract_step_info_value(infos, "steps_since_progress", env_idx, done=step_done, default=0))
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
                    ep_stalled = bool(_extract_step_info_value(infos, "stalled", i, done=True, default=False))
                    ep_ante = int(_extract_step_info_value(infos, "ante", i, done=True, default=1))
                    episode_wins.append(ep_won)
                    episode_stalls.append(ep_stalled)
                    episode_antes.append(ep_ante)
                    if sil_tracker is not None and sil_buffer is not None:
                        sil_tracker.finish_episode(
                            int(i), won=ep_won and not ep_stalled, buffer=sil_buffer
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
                next_teacher_actions = np.asarray(
                    [
                        int(_extract_step_info_value(
                            infos,
                            "next_teacher_action",
                            env_idx,
                            done=bool(dones[env_idx]),
                            default=-1,
                        ))
                        for env_idx in range(config.num_envs)
                    ],
                    dtype=np.int64,
                )
                if np.any(dones):
                    for done_idx in np.where(dones)[0]:
                        reset_teacher = _extract_vector_info_value(
                            infos,
                            "teacher_action",
                            int(done_idx),
                            _MISSING,
                        )
                        if reset_teacher is not _MISSING:
                            next_teacher_actions[int(done_idx)] = int(reset_teacher)
                teacher_action_buf = next_teacher_actions
                total_steps += config.num_envs

            # Bootstrap values for GAE
            with torch.no_grad():
                _, value_dict = _unwrap_model(model).action_distribution(
                    obs_buf.tokens,
                    obs_buf.token_types,
                    obs_buf.scalars,
                    obs_buf.attention_mask,
                    obs_buf.action_mask,
                    temperature=config.rollout_temperature,
                )
                last_values = value_dict["expected_score"].cpu().numpy()
                if return_rms is not None:
                    last_values = return_rms.denormalize(last_values)

            buffer.compute_returns_and_advantages(last_values=last_values)
            buffer.normalize_advantages()
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
                    explained_variance = 1.0 - float(
                        np.var(flat_returns - buffer._flat_values)
                    ) / var_returns
            writer.add_scalar("ppo/explained_variance", explained_variance, update_count + 1)

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

            # Resolve the distillation coefficient via the shared helper so the
            # schedule semantics (zero-disables, floor clamp, decay window) are
            # consistent across the train loop, the test suite, and any future
            # call sites. See resolve_distill_coeff for the full contract.
            distill_coeff_now = resolve_distill_coeff(
                config, total_steps, schedule_total_steps=schedule_total_steps
            )

            update_stats = _run_ppo_update(
                model=model,
                optimizer=optimizer,
                buffer=buffer,
                return_rms=return_rms,
                entropy_coeff=entropy_coeff,
                distill_coeff=distill_coeff_now,
                config=config,
                accum_steps=accum_steps,
                effective_batch_size=effective_batch_size,
                device=device,
                use_pin_memory=use_pin_memory,
                policy_loss_scale=0.0 if in_critic_warmup else 1.0,
                sil_buffer=sil_buffer,
            )

            dagger_losses, dagger_teacher_match_fractions = _run_dagger_bc_update(
                model=model,
                optimizer=optimizer,
                buffer=buffer,
                # DAgger BC trains the policy, so it must respect the critic
                # warmup freeze or the "frozen" policy drifts anyway.
                coeff=0.0 if in_critic_warmup else config.dagger_bc_coeff,
                config=config,
                accum_steps=accum_steps,
                effective_batch_size=effective_batch_size,
                device=device,
                use_pin_memory=use_pin_memory,
            )

            update_policy_losses = update_stats.policy_losses
            update_value_losses = update_stats.value_losses
            update_survival_losses = update_stats.survival_losses
            update_entropies = update_stats.entropies
            update_normalized_entropies = update_stats.normalized_entropies
            update_action_type_entropies = update_stats.action_type_entropies
            update_clip_fracs = update_stats.clip_fracs
            update_approx_kls = update_stats.approx_kls
            update_valid_action_counts = update_stats.valid_action_counts
            update_valid_action_type_counts = update_stats.valid_action_type_counts
            update_distill_losses = update_stats.distill_losses
            update_teacher_match_fractions = update_stats.teacher_match_fractions
            update_distill_weight_means = update_stats.distill_weight_means
            update_on_policy_fractions = update_stats.on_policy_fractions

            # Track the normalized entropy signal every update, even with fixed entropy.
            mean_normalized_entropy = float(np.mean(update_normalized_entropies))
            entropy_signal_ema = _smoothed_entropy_signal(
                entropy_signal_ema,
                mean_normalized_entropy,
                config.entropy_ema_beta,
            )
            if config.adaptive_entropy:
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
            writer.add_scalar("ppo/survival_loss", np.mean(update_survival_losses), update_count)
            writer.add_scalar("ppo/entropy", mean_entropy, update_count)
            writer.add_scalar("ppo/entropy_normalized", np.mean(update_normalized_entropies), update_count)
            writer.add_scalar("ppo/action_type_entropy_normalized", np.mean(update_action_type_entropies), update_count)
            writer.add_scalar("ppo/clip_fraction", np.mean(update_clip_fracs), update_count)
            writer.add_scalar("ppo/approx_kl", np.mean(update_approx_kls), update_count)
            writer.add_scalar("ppo/distill_loss", float(np.mean(update_distill_losses)), update_count)
            writer.add_scalar("ppo/distill_coeff", float(distill_coeff_now), update_count)
            if sil_buffer is not None:
                writer.add_scalar("sil/buffer_episodes", float(sil_buffer.num_episodes), update_count)
                writer.add_scalar("sil/buffer_transitions", float(sil_buffer.num_transitions), update_count)
                writer.add_scalar(
                    "sil/episodes_added_total", float(sil_buffer.episodes_added_total), update_count
                )
                if update_stats.sil_losses:
                    writer.add_scalar("sil/loss", float(np.mean(update_stats.sil_losses)), update_count)
                if update_stats.sil_advantage_means:
                    writer.add_scalar(
                        "sil/advantage_mean",
                        float(np.mean(update_stats.sil_advantage_means)),
                        update_count,
                    )
                if update_stats.sil_gate_means:
                    writer.add_scalar(
                        "sil/gate_mean",
                        float(np.mean(update_stats.sil_gate_means)),
                        update_count,
                    )
            if update_stats.distill_losses_weighted:
                writer.add_scalar(
                    "ppo/distill_loss_weighted",
                    float(np.mean(update_stats.distill_losses_weighted)),
                    update_count,
                )
            if update_stats.teacher_label_present_fractions:
                writer.add_scalar(
                    "ppo/teacher_label_present_fraction",
                    float(np.mean(update_stats.teacher_label_present_fractions)),
                    update_count,
                )
                writer.add_scalar(
                    "ppo/teacher_action_mask_valid_fraction",
                    float(np.mean(update_stats.teacher_action_mask_valid_fractions)),
                    update_count,
                )
                writer.add_scalar(
                    "ppo/teacher_reachable_fraction",
                    float(np.mean(update_stats.teacher_reachable_fractions)),
                    update_count,
                )
                writer.add_scalar(
                    "ppo/teacher_unreachable_fraction",
                    float(np.mean(update_stats.teacher_unreachable_fractions)),
                    update_count,
                )
                # Unreachable-by-family breakdown.
                for fam_name in _ACTION_FAMILY_NAMES:
                    fam_vals = update_stats.teacher_unreachable_family_fractions.get(fam_name)
                    if fam_vals:
                        writer.add_scalar(
                            f"ppo/teacher_unreachable/{fam_name}_fraction",
                            float(np.mean(fam_vals)),
                            update_count,
                        )
                writer.add_scalar(
                    "ppo/teacher_lp_mean_reachable",
                    float(np.mean(update_stats.teacher_lp_mean_reachable)),
                    update_count,
                )
            writer.add_scalar("ppo/on_policy_fraction", float(np.mean(update_on_policy_fractions)), update_count)
            writer.add_scalar(
                "ppo/on_policy_advantage_mean",
                float(np.mean(update_stats.on_policy_advantage_means)),
                update_count,
            )
            writer.add_scalar(
                "ppo/on_policy_advantage_std",
                float(np.mean(update_stats.on_policy_advantage_stds)),
                update_count,
            )
            writer.add_scalar(
                "ppo/on_policy_positive_advantage_fraction",
                float(np.mean(update_stats.on_policy_positive_advantage_fractions)),
                update_count,
            )
            writer.add_scalar(
                "ppo/on_policy_return_mean",
                float(np.mean(update_stats.on_policy_return_means)),
                update_count,
            )
            minibatches_processed = int(np.sum(update_stats.ppo_minibatches_processed))
            writer.add_scalar("ppo/minibatches_processed", minibatches_processed, update_count)
            # Target-KL early stopping can halt an update after far fewer minibatches
            # than expected; without this it is invisible. Expected = full passes over
            # the rollout for every PPO epoch.
            n_samples = len(buffer._flat_returns) if len(buffer._flat_returns) > 0 else 0
            minibatches_expected = 0
            minibatch_fraction = 1.0
            if n_samples > 0:
                minibatches_per_epoch = max(1, math.ceil(n_samples / effective_batch_size))
                minibatches_expected = minibatches_per_epoch * config.ppo_epochs
                writer.add_scalar("ppo/minibatches_expected", minibatches_expected, update_count)
                minibatch_fraction = min(1.0, minibatches_processed / minibatches_expected)
                writer.add_scalar("ppo/minibatch_fraction", minibatch_fraction, update_count)
                writer.add_scalar(
                    "ppo/early_stop_fraction",
                    1.0 if minibatches_processed < minibatches_expected else 0.0,
                    update_count,
                )
            # Trust-region statistics: capture once so both the TensorBoard
            # ppo/stop_reason_* scalars and the console diagnostic warnings
            # below use identical values for this update.
            kl_p95 = float(np.percentile(update_approx_kls, 95)) if update_approx_kls else 0.0
            kl_max = float(np.max(update_approx_kls)) if update_approx_kls else 0.0
            clip_frac_max = float(np.max(update_clip_fracs)) if update_clip_fracs else 0.0
            writer.add_scalar("ppo/stop_reason_kl_p95", kl_p95, update_count)
            writer.add_scalar("ppo/stop_reason_kl_max", kl_max, update_count)
            if update_approx_kls and config.target_kl is not None:
                writer.add_scalar(
                    "ppo/early_stop_kl",
                    1.0 if kl_max > config.target_kl else 0.0,
                    update_count,
                )
            if update_clip_fracs:
                writer.add_scalar("ppo/clip_fraction_max", clip_frac_max, update_count)
            writer.add_scalar("debug/teacher_rollout_prob", teacher_rollout_prob_now, update_count)
            writer.add_scalar(
                "dagger/bc_loss",
                float(np.mean(dagger_losses)) if dagger_losses else float("nan"),
                update_count,
            )
            writer.add_scalar(
                "dagger/teacher_match_fraction",
                float(np.nanmean(dagger_teacher_match_fractions)) if dagger_teacher_match_fractions else float("nan"),
                update_count,
            )
            writer.add_scalar(
                "ppo/teacher_match_fraction",
                float(np.mean(update_teacher_match_fractions)),
                update_count,
            )
            writer.add_scalar(
                "ppo/distill_weight_mean",
                float(np.mean(update_distill_weight_means)),
                update_count,
            )
            writer.add_scalar("ppo/valid_action_count_mean", np.mean(update_valid_action_counts), update_count)
            writer.add_scalar("ppo/valid_action_type_count_mean", np.mean(update_valid_action_type_counts), update_count)
            writer.add_scalar("ppo/entropy_coeff", entropy_coeff, update_count)
            # Entropy-controller saturation flags: 1.0 when entropy_coeff is
            # pinned at its min/max clamp, so an adaptive run can be diagnosed
            # when the controller has run out of room (e.g. permanently at max
            # because the policy is too deterministic even at max entropy).
            writer.add_scalar(
                "ppo/entropy_coeff_at_min",
                1.0 if config.adaptive_entropy and entropy_coeff <= config.alpha_min + 1e-9 else 0.0,
                update_count,
            )
            writer.add_scalar(
                "ppo/entropy_coeff_at_max",
                1.0 if config.adaptive_entropy and entropy_coeff >= config.alpha_max - 1e-9 else 0.0,
                update_count,
            )
            writer.add_scalar("ppo/entropy_bonus", entropy_coeff * mean_normalized_entropy, update_count)
            writer.add_scalar(
                "ppo/action_type_entropy_bonus",
                entropy_coeff * config.action_type_entropy_scale * np.mean(update_action_type_entropies),
                update_count,
            )
            writer.add_scalar(
                "ppo/entropy_bonus_total",
                entropy_coeff
                * (mean_normalized_entropy + config.action_type_entropy_scale * np.mean(update_action_type_entropies)),
                update_count,
            )
            writer.add_scalar("ppo/entropy_bonus_raw", entropy_coeff * mean_entropy, update_count)
            writer.add_scalar("ppo/entropy_signal", entropy_signal_ema, update_count)
            writer.add_scalar("ppo/entropy_target_error", entropy_signal_ema - config.target_entropy, update_count)
            writer.add_scalar("ppo/total_steps", total_steps, update_count)
            dense_scale = config.reward_config.dense_reward_scale if config.reward_config is not None else 1.0
            writer.add_scalar("reward/dense_scale", float(dense_scale), update_count)
            _write_rollout_scalars(writer, update_count, rm)

            if return_rms is not None:
                writer.add_scalar("ppo/return_norm_mean", return_rms.mean, update_count)
                writer.add_scalar("ppo/return_norm_std", return_rms.std, update_count)

            flat_returns = buffer._flat_returns
            if len(flat_returns) > 0:
                writer.add_scalar("debug/returns_mean", float(np.mean(flat_returns)), update_count)
                writer.add_scalar("debug/returns_std", float(np.std(flat_returns)), update_count)
                flat_adv = buffer._flat_advantages
                writer.add_scalar("debug/advantages_mean", float(np.mean(flat_adv)), update_count)
                writer.add_scalar("debug/advantages_std", float(np.std(flat_adv)), update_count)

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
                writer.add_scalar("rollout/ep_reward_mean", np.mean(recent), update_count)
                writer.add_scalar("rollout/ep_length_mean", np.mean(episode_lengths[-100:]), update_count)
                writer.add_scalar("rollout/win_rate", np.mean(recent_wins), update_count)
                writer.add_scalar("rollout/stall_rate", np.mean(recent_stalls), update_count)
                writer.add_scalar("rollout/episodes_total", len(episode_rewards), update_count)
                # Where episodes end: mean ante reached, plus the loss-only
                # distribution (wins excluded so the "wall" ante stands out).
                recent_antes = episode_antes[-100:]
                writer.add_scalar("rollout/ante_reached_mean", float(np.mean(recent_antes)), update_count)
                loss_antes = [a for a, w in zip(recent_antes, recent_wins) if not w]
                if loss_antes:
                    for bucket in range(1, 9):
                        writer.add_scalar(
                            f"rollout/loss_ante/{bucket}_fraction",
                            float(np.mean([a == bucket for a in loss_antes])),
                            update_count,
                        )
                recent_reward_mean = float(np.mean(recent))
                recent_length_mean = float(np.mean(episode_lengths[-100:]))
                recent_win_rate = float(np.mean(recent_wins))
                recent_stall_rate = float(np.mean(recent_stalls))
                # Phase 6: track rolling win_rate/ep_reward for the correlation
                # acceptance criterion (they are currently decoupled because
                # dense shaping is farmable independent of winning).
                _rolling_win_rates.append(recent_win_rate)
                _rolling_ep_rewards.append(recent_reward_mean)
                if len(_rolling_win_rates) > 100:
                    _rolling_win_rates = _rolling_win_rates[-100:]
                    _rolling_ep_rewards = _rolling_ep_rewards[-100:]
                if len(_rolling_win_rates) >= 5:
                    corr = float(np.corrcoef(_rolling_win_rates, _rolling_ep_rewards)[0, 1])
                    writer.add_scalar("rollout/win_rate_ep_reward_corr", corr, update_count)
            else:
                recent_reward_mean = float("nan")
                recent_length_mean = float("nan")
                recent_win_rate = float("nan")
                recent_stall_rate = float("nan")

            should_checkpoint = update_count % config.checkpoint_interval == 0 or update_count == planned_updates
            if should_checkpoint:
                # Make sure all preceding scalars (rollout, distill, etc.)
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
                writer.add_scalar("eval/best_win_rate", best_eval_win_rate, update_count)
                writer.add_scalar("eval/best_update", best_eval_update, update_count)
                writer.add_scalar("checkpoint/is_best_eval", 1.0 if is_best else 0.0, update_count)
                # eval runs `eval_games` full games in-process; release the
                # forward-pass allocations it cached before the next rollout.
                if device.type == "mps":
                    torch.mps.empty_cache()

            should_log_progress = update_count % config.log_interval == 0 or should_eval or update_count == planned_updates
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
            if (
                update_stats.distill_losses_weighted
                and mean_policy_loss > 0.0
            ):
                mean_distill_weighted = float(
                    np.mean(update_stats.distill_losses_weighted)
                )
                if mean_distill_weighted > 10.0 * mean_policy_loss:
                    logger.warning(
                        "ppo/distill_loss_weighted=%.4f is >10x ppo/policy_loss=%.4f "
                        "at update %d; distillation is dominating the policy gradient. "
                        "Check ppo/teacher_unreachable_fraction and consider lowering "
                        "--heuristic-distill-coeff.",
                        mean_distill_weighted,
                        mean_policy_loss,
                        update_count,
                    )

            if update_stats.teacher_reachable_fractions:
                mean_reachable = float(
                    np.mean(update_stats.teacher_reachable_fractions)
                )
                # Only warn when there are actually teacher labels in the
                # batch (i.e. teacher_label_present_fraction > 0.1). A batch
                # with no teacher labels trivially has reachability 0.
                mean_teacher_present = float(
                    np.mean(update_stats.teacher_label_present_fractions)
                )
                if mean_teacher_present > 0.1 and mean_reachable < 0.95:
                    logger.warning(
                        "ppo/teacher_reachable_fraction=%.3f (<0.95) at update %d; "
                        "%.1f%% of teacher labels were dropped as unreachable. "
                        "Inspect hand/not_in_candidates_fraction to see if the "
                        "structured policy is missing hand slots the heuristic uses.",
                        mean_reachable,
                        update_count,
                        (1.0 - mean_reachable) * 100.0,
                    )

            if update_stats.ppo_minibatches_processed:
                minibatches_done = int(
                    np.sum(update_stats.ppo_minibatches_processed)
                )
                # minibatches_expected and minibatch_fraction were captured
                # above, before `del buffer`. The buffer is released before
                # eval/checkpoint to bound MPS memory peak.
                if (
                    minibatches_expected > 0
                    and minibatch_fraction < 0.5
                ):
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

        # Numpy views for writing from env output (CPU side)
        self._np_tokens = np.zeros((num_envs, MAX_SEQ_LEN, TOKEN_DIM), dtype=np.int64)
        self._np_token_types = np.zeros((num_envs, MAX_SEQ_LEN), dtype=np.int64)
        self._np_scalars = np.zeros((num_envs, SCALAR_DIM), dtype=np.float32)
        self._np_attention_mask = np.zeros((num_envs, MAX_SEQ_LEN), dtype=np.int64)
        self._np_action_mask = np.zeros((num_envs, NUM_ACTIONS), dtype=np.float32)

    def update(self, obs_dict: dict) -> None:
        """Copy vectorized env output into pre-allocated tensors."""
        np.copyto(self._np_tokens, obs_dict["tokens"])
        np.copyto(self._np_token_types, obs_dict["token_types"])
        np.copyto(self._np_scalars, obs_dict["scalars"])
        np.copyto(self._np_attention_mask, obs_dict["attention_mask"])
        np.copyto(self._np_action_mask, obs_dict["action_mask"])

        self.tokens.copy_(torch.from_numpy(self._np_tokens))
        self.token_types.copy_(torch.from_numpy(self._np_token_types))
        self.scalars.copy_(torch.from_numpy(self._np_scalars))
        self.attention_mask.copy_(torch.from_numpy(self._np_attention_mask))
        self.action_mask.copy_(torch.from_numpy(self._np_action_mask))

    def as_numpy_dict(self) -> dict:
        """Return current numpy arrays (for storing in rollout buffer)."""
        return {
            "tokens": self._np_tokens,
            "token_types": self._np_token_types,
            "scalars": self._np_scalars,
            "attention_mask": self._np_attention_mask,
            "action_mask": self._np_action_mask,
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
    }
