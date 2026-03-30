"""Phase 2: PPO training loop with vectorized environments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from pylatro import GameData, load_game_data

from ..agent import AgentConfig, BalatroAgent
from ..constants import MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM
from ..distributions import MaskedCategorical
from ..env import BalatroEnv
from ..vocab import Vocab, build_vocab
from .rollout_buffer import RolloutBuffer

logger = logging.getLogger(__name__)


def _load_checkpoint_compatible(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """Load checkpoint, handling DataParallel prefix mismatch."""
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    is_wrapped = isinstance(model, nn.DataParallel)

    if has_module_prefix and not is_wrapped:
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    elif not has_module_prefix and is_wrapped:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)


def _make_env(seed: int, stake: int, data: GameData, vocab: Vocab):
    """Factory for creating a BalatroEnv (used by vectorized env wrappers)."""
    def _thunk():
        return BalatroEnv(seed=seed, stake=stake, data=data, vocab=vocab)
    return _thunk


def _make_vectorized_envs(
    num_envs: int,
    data: GameData,
    vocab: Vocab,
    stake: int = 1,
    use_async: bool = True,
):
    """Create a gymnasium VectorEnv (async for multiprocess, sync for single-process)."""
    import gymnasium

    env_fns = [_make_env(i, stake, data, vocab) for i in range(num_envs)]

    if use_async and num_envs > 1:
        return gymnasium.vector.AsyncVectorEnv(env_fns)
    else:
        return gymnasium.vector.SyncVectorEnv(env_fns)


@dataclass
class PPOConfig:
    num_envs: int = 32
    rollout_length: int = 2048
    total_timesteps: int = 1_000_000
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.1
    entropy_coeff: float = 0.01
    target_entropy: float = 0.15
    alpha_lr: float = 3e-4
    alpha_min: float = 0.001
    alpha_max: float = 0.03
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 2e-5
    device: str = "cpu"
    save_dir: str = "checkpoints/ppo"
    log_dir: str = "runs/ppo"
    eval_interval: int = 50
    eval_games: int = 10
    async_envs: bool = True  # Use multiprocess envs (AsyncVectorEnv)


def train_ppo(
    config: PPOConfig,
    agent_config: AgentConfig | None = None,
    pretrained_path: str | None = None,
    data: GameData | None = None,
) -> BalatroAgent:
    """Run PPO training with vectorized environments."""
    if data is None:
        data = load_game_data()
    vocab = build_vocab(data)
    if agent_config is None:
        agent_config = AgentConfig()

    device = torch.device(config.device)
    use_pin_memory = device.type == "cuda"
    model = BalatroAgent(agent_config, vocab).to(device)

    if pretrained_path:
        _load_checkpoint_compatible(model, pretrained_path, device)
        logger.info(f"Loaded pretrained model from {pretrained_path}")

    if config.device == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=config.lr)

    # Create vectorized environments
    vec_env = _make_vectorized_envs(
        config.num_envs, data, vocab,
        use_async=config.async_envs,
    )
    obs_dict, _ = vec_env.reset()

    # Pre-allocate obs tensors for batched inference
    obs_buf = _ObsBuffer(config.num_envs, device)
    obs_buf.update(obs_dict)

    from torch.utils.tensorboard import SummaryWriter

    save_path = Path(config.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(config.log_dir)

    total_steps = 0
    update_count = 0
    # Adaptive entropy coefficient (SAC-style)
    log_alpha = torch.tensor(np.log(config.entropy_coeff), dtype=torch.float32, requires_grad=True)
    alpha_optimizer = AdamW([log_alpha], lr=config.alpha_lr)
    entropy_coeff = config.entropy_coeff
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_wins: list[bool] = []
    # Per-env accumulators (vectorized envs auto-reset, so we track manually)
    env_ep_reward = np.zeros(config.num_envs, dtype=np.float64)
    env_ep_length = np.zeros(config.num_envs, dtype=np.int64)

    while total_steps < config.total_timesteps:
        buffer = RolloutBuffer(
            num_envs=config.num_envs,
            rollout_length=config.rollout_length,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        # === Collect rollouts (vectorized) ===
        model.eval()
        for step in range(config.rollout_length):
            with torch.no_grad():
                logits, value_dict = model(
                    obs_buf.tokens, obs_buf.token_types, obs_buf.scalars,
                    obs_buf.attention_mask, obs_buf.action_mask,
                )
                dist = MaskedCategorical(logits, obs_buf.action_mask)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                values = value_dict["expected_score"]

            actions_np = actions.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()
            values_np = values.cpu().numpy()

            # Step all envs at once
            next_obs_dict, rewards, terminated, truncated, infos = vec_env.step(actions_np)
            dones = terminated | truncated

            # Store transition (using pre-step obs from obs_buf)
            buffer.add_batch(
                step=step,
                obs=obs_buf.as_numpy_dict(),
                actions=actions_np,
                rewards=rewards.astype(np.float32),
                values=values_np,
                log_probs=log_probs_np,
                dones=dones,
            )

            # Track per-env episode stats
            env_ep_reward += rewards
            env_ep_length += 1

            # Handle completed episodes (vectorized envs auto-reset)
            for i in np.where(dones)[0]:
                episode_rewards.append(float(env_ep_reward[i]))
                episode_lengths.append(int(env_ep_length[i]))
                # final_info is in infos for auto-reset envs
                final_info = infos.get("final_info", [None] * config.num_envs)
                if final_info[i] is not None:
                    episode_wins.append(final_info[i].get("won", False))
                else:
                    episode_wins.append(infos.get("won", [False] * config.num_envs)[i] if "won" in infos else False)
                env_ep_reward[i] = 0.0
                env_ep_length[i] = 0

            # Update obs buffer with new observations
            obs_buf.update(next_obs_dict)
            total_steps += config.num_envs

        # Bootstrap values for GAE
        with torch.no_grad():
            _, value_dict = model(
                obs_buf.tokens, obs_buf.token_types, obs_buf.scalars,
                obs_buf.attention_mask, obs_buf.action_mask,
            )
            last_values = value_dict["expected_score"].cpu().numpy()

        buffer.compute_returns_and_advantages(last_values=last_values)

        # === PPO update ===
        model.train()
        update_policy_losses = []
        update_value_losses = []
        update_entropies = []
        update_clip_fracs = []

        for ppo_epoch in range(config.ppo_epochs):
            batches = buffer.get_batches(config.mini_batch_size, device, pin_memory=use_pin_memory)
            for batch in batches:
                logits, value_dict = model(
                    batch["tokens"], batch["token_types"], batch["scalars"],
                    batch["attention_mask"], batch["action_mask"],
                )
                dist = MaskedCategorical(logits, batch["action_mask"])

                new_log_probs = dist.log_prob(batch["actions"])
                entropy = dist.entropy().mean()

                # Policy loss (clipped PPO)
                ratio = torch.exp(new_log_probs - batch["old_log_probs"])
                advantages = batch["advantages"]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(value_dict["expected_score"], batch["returns"])

                alpha = log_alpha.exp().detach()
                loss = policy_loss + config.value_loss_coeff * value_loss - alpha * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > config.clip_epsilon).float().mean().item()
                update_policy_losses.append(policy_loss.item())
                update_value_losses.append(value_loss.item())
                update_entropies.append(entropy.item())
                update_clip_fracs.append(clip_frac)

        # Adaptive entropy: adjust alpha toward target entropy
        mean_entropy = np.mean(update_entropies)
        alpha_loss = -(log_alpha * (mean_entropy - config.target_entropy))
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()
        with torch.no_grad():
            log_alpha.clamp_(np.log(config.alpha_min), np.log(config.alpha_max))
        entropy_coeff = log_alpha.exp().item()
        update_count += 1

        # Save checkpoint every 10 updates
        if update_count % 10 == 0:
            save_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(save_model.state_dict(), save_path / f"ppo_update{update_count}.pt")

        # TensorBoard logging
        writer.add_scalar("ppo/policy_loss", np.mean(update_policy_losses), update_count)
        writer.add_scalar("ppo/value_loss", np.mean(update_value_losses), update_count)
        writer.add_scalar("ppo/entropy", np.mean(update_entropies), update_count)
        writer.add_scalar("ppo/clip_fraction", np.mean(update_clip_fracs), update_count)
        writer.add_scalar("ppo/entropy_coeff", entropy_coeff, update_count)
        writer.add_scalar("ppo/total_steps", total_steps, update_count)

        flat_returns = buffer._flat_returns
        if len(flat_returns) > 0:
            writer.add_scalar("debug/returns_mean", float(np.mean(flat_returns)), update_count)
            writer.add_scalar("debug/returns_std", float(np.std(flat_returns)), update_count)
            flat_adv = buffer._flat_advantages
            writer.add_scalar("debug/advantages_mean", float(np.mean(flat_adv)), update_count)
            writer.add_scalar("debug/advantages_std", float(np.std(flat_adv)), update_count)

        if episode_rewards:
            recent = episode_rewards[-100:]
            recent_wins = episode_wins[-100:]
            writer.add_scalar("rollout/ep_reward_mean", np.mean(recent), update_count)
            writer.add_scalar("rollout/ep_length_mean", np.mean(episode_lengths[-100:]), update_count)
            writer.add_scalar("rollout/win_rate", np.mean(recent_wins), update_count)
            writer.add_scalar("rollout/episodes_total", len(episode_rewards), update_count)

        if update_count % config.eval_interval == 0:
            win_rate = evaluate_model(model, data, vocab, config.eval_games, device)
            writer.add_scalar("eval/win_rate", win_rate, update_count)
            logger.info(
                f"Update {update_count}, steps {total_steps}: "
                f"win_rate={win_rate:.3f}, entropy_coeff={entropy_coeff:.5f}"
            )
            save_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(save_model.state_dict(), save_path / f"ppo_update{update_count}.pt")

    vec_env.close()
    writer.close()
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
) -> float:
    """Evaluate model win rate over num_games."""
    model.eval()
    wins = 0

    for game_idx in range(num_games):
        env = BalatroEnv(seed=10000 + game_idx, data=data, vocab=vocab)
        obs, _ = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                batch = _single_obs_to_batch(obs, device)
                logits, _ = model(
                    batch["tokens"], batch["token_types"], batch["scalars"],
                    batch["attention_mask"], batch["action_mask"],
                )
                dist = MaskedCategorical(logits, batch["action_mask"])
                action = dist.sample().item()

            obs, reward, terminated, truncated, info = env.step(action)
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
