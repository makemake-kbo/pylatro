"""Phase 2: PPO training loop."""

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
from ..env import BalatroEnv
from ..vocab import Vocab, build_vocab
from .rollout_buffer import RolloutBuffer

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    num_envs: int = 32
    rollout_length: int = 512
    total_timesteps: int = 1_000_000
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    entropy_decay: float = 0.9999
    entropy_floor: float = 0.003
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 1.0
    lr: float = 1e-4
    device: str = "cpu"
    save_dir: str = "checkpoints/ppo"
    log_dir: str = "runs/ppo"
    eval_interval: int = 50
    eval_games: int = 10


def train_ppo(
    config: PPOConfig,
    agent_config: AgentConfig | None = None,
    pretrained_path: str | None = None,
    data: GameData | None = None,
) -> BalatroAgent:
    """Run PPO training."""
    if data is None:
        data = load_game_data()
    vocab = build_vocab(data)
    if agent_config is None:
        agent_config = AgentConfig()

    device = torch.device(config.device)
    model = BalatroAgent(agent_config, vocab).to(device)

    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
        logger.info(f"Loaded pretrained model from {pretrained_path}")

    optimizer = AdamW(model.parameters(), lr=config.lr)

    # Create environments
    envs = [BalatroEnv(seed=i, data=data, vocab=vocab) for i in range(config.num_envs)]
    obs_list = []
    for env in envs:
        obs, _ = env.reset()
        obs_list.append(obs)

    from torch.utils.tensorboard import SummaryWriter

    save_path = Path(config.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(config.log_dir)

    total_steps = 0
    update_count = 0
    entropy_coeff = config.entropy_coeff
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_wins: list[bool] = []
    # Track per-env episode accumulators
    env_ep_reward = [0.0] * config.num_envs
    env_ep_length = [0] * config.num_envs

    while total_steps < config.total_timesteps:
        buffer = RolloutBuffer(num_envs=config.num_envs, gamma=config.gamma, gae_lambda=config.gae_lambda)

        # Collect rollouts
        model.eval()
        for step in range(config.rollout_length):
            with torch.no_grad():
                batch = _obs_list_to_batch(obs_list, device)
                dist, value_dict = model(
                    batch["tokens"], batch["token_types"], batch["scalars"],
                    batch["attention_mask"], batch["action_mask"],
                )

                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                values = value_dict["expected_score"]

            for i, env in enumerate(envs):
                action = actions[i].item()
                value = values[i].item()
                log_prob = log_probs[i].item()

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                buffer.add(i, obs_list[i], action, reward, value, log_prob, done)
                env_ep_reward[i] += reward
                env_ep_length[i] += 1

                if done:
                    episode_rewards.append(env_ep_reward[i])
                    episode_lengths.append(env_ep_length[i])
                    episode_wins.append(info.get("won", False))
                    env_ep_reward[i] = 0.0
                    env_ep_length[i] = 0
                    obs, _ = env.reset(seed=total_steps + i)

                obs_list[i] = obs
                total_steps += 1

        # Compute per-env last values for GAE bootstrap
        with torch.no_grad():
            batch = _obs_list_to_batch(obs_list, device)
            _, value_dict = model(
                batch["tokens"], batch["token_types"], batch["scalars"],
                batch["attention_mask"], batch["action_mask"],
            )
            last_values = value_dict["expected_score"].cpu().numpy().tolist()

        buffer.compute_returns_and_advantages(last_values=last_values)

        # PPO update
        model.train()
        update_policy_losses = []
        update_value_losses = []
        update_entropies = []
        update_clip_fracs = []

        for ppo_epoch in range(config.ppo_epochs):
            batches = buffer.get_batches(config.mini_batch_size, device)
            for batch in batches:
                dist, value_dict = model(
                    batch["tokens"], batch["token_types"], batch["scalars"],
                    batch["attention_mask"], batch["action_mask"],
                )

                new_log_probs = dist.log_prob(batch["actions"])
                entropy = dist.entropy().mean()

                # Policy loss (clipped PPO)
                ratio = torch.exp(new_log_probs - batch["old_log_probs"])
                advantages = batch["advantages"]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss — use unbounded expected_score head for PPO value
                value_loss = F.mse_loss(value_dict["expected_score"], batch["returns"])

                loss = policy_loss + config.value_loss_coeff * value_loss - entropy_coeff * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                # Track for logging
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > config.clip_epsilon).float().mean().item()
                update_policy_losses.append(policy_loss.item())
                update_value_losses.append(value_loss.item())
                update_entropies.append(entropy.item())
                update_clip_fracs.append(clip_frac)

        entropy_coeff = max(entropy_coeff * config.entropy_decay, config.entropy_floor)
        update_count += 1

        # TensorBoard: per-update metrics
        writer.add_scalar("ppo/policy_loss", np.mean(update_policy_losses), update_count)
        writer.add_scalar("ppo/value_loss", np.mean(update_value_losses), update_count)
        writer.add_scalar("ppo/entropy", np.mean(update_entropies), update_count)
        writer.add_scalar("ppo/clip_fraction", np.mean(update_clip_fracs), update_count)
        writer.add_scalar("ppo/entropy_coeff", entropy_coeff, update_count)
        writer.add_scalar("ppo/total_steps", total_steps, update_count)

        # TensorBoard: episode stats (from completed episodes this rollout)
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
            torch.save(model.state_dict(), save_path / f"ppo_update{update_count}.pt")

    writer.close()
    return model


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
                dist, _ = model(
                    batch["tokens"], batch["token_types"], batch["scalars"],
                    batch["attention_mask"], batch["action_mask"],
                )
                action = dist.sample().item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if info.get("won", False):
            wins += 1

    return wins / max(num_games, 1)


def _obs_list_to_batch(obs_list: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.tensor(np.array([o["tokens"] for o in obs_list]), dtype=torch.long, device=device),
        "token_types": torch.tensor(np.array([o["token_types"] for o in obs_list]), dtype=torch.long, device=device),
        "scalars": torch.tensor(np.array([o["scalars"] for o in obs_list]), dtype=torch.float32, device=device),
        "attention_mask": torch.tensor(np.array([o["attention_mask"] for o in obs_list]), dtype=torch.long, device=device),
        "action_mask": torch.tensor(np.array([o["action_mask"] for o in obs_list]), dtype=torch.float32, device=device),
    }


def _single_obs_to_batch(obs: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.tensor(obs["tokens"], dtype=torch.long, device=device).unsqueeze(0),
        "token_types": torch.tensor(obs["token_types"], dtype=torch.long, device=device).unsqueeze(0),
        "scalars": torch.tensor(obs["scalars"], dtype=torch.float32, device=device).unsqueeze(0),
        "attention_mask": torch.tensor(obs["attention_mask"], dtype=torch.long, device=device).unsqueeze(0),
        "action_mask": torch.tensor(obs["action_mask"], dtype=torch.float32, device=device).unsqueeze(0),
    }
