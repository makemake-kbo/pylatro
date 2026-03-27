"""Experience storage for PPO training with correct per-env GAE."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class EnvTrajectory:
    """Stores one environment's trajectory within a rollout."""
    tokens: list[np.ndarray] = field(default_factory=list)
    token_types: list[np.ndarray] = field(default_factory=list)
    scalars: list[np.ndarray] = field(default_factory=list)
    attention_masks: list[np.ndarray] = field(default_factory=list)
    action_masks: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    def add(self, obs: dict, action: int, reward: float, value: float, log_prob: float, done: bool) -> None:
        self.tokens.append(obs["tokens"])
        self.token_types.append(obs["token_types"])
        self.scalars.append(obs["scalars"])
        self.attention_masks.append(obs["attention_mask"])
        self.action_masks.append(obs["action_mask"])
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.actions)


@dataclass
class RolloutBuffer:
    """Stores per-env trajectories and computes GAE correctly per environment."""

    num_envs: int = 1
    gamma: float = 0.995
    gae_lambda: float = 0.95

    envs: list[EnvTrajectory] = field(default_factory=list)

    # Flattened after compute_returns_and_advantages
    _flat_tokens: list[np.ndarray] = field(default_factory=list)
    _flat_token_types: list[np.ndarray] = field(default_factory=list)
    _flat_scalars: list[np.ndarray] = field(default_factory=list)
    _flat_attention_masks: list[np.ndarray] = field(default_factory=list)
    _flat_action_masks: list[np.ndarray] = field(default_factory=list)
    _flat_actions: list[int] = field(default_factory=list)
    _flat_log_probs: list[float] = field(default_factory=list)
    _flat_advantages: np.ndarray | None = None
    _flat_returns: np.ndarray | None = None

    def __post_init__(self):
        if not self.envs:
            self.envs = [EnvTrajectory() for _ in range(self.num_envs)]

    def add(self, env_idx: int, obs: dict, action: int, reward: float, value: float, log_prob: float, done: bool) -> None:
        self.envs[env_idx].add(obs, action, reward, value, log_prob, done)

    def compute_returns_and_advantages(self, last_values: list[float]) -> None:
        """Compute GAE advantages per env, then flatten for batching.

        Args:
            last_values: bootstrap value for each env (one per env).
        """
        all_advantages = []
        all_returns = []

        for env_idx, traj in enumerate(self.envs):
            n = len(traj.rewards)
            if n == 0:
                continue

            advantages = np.zeros(n, dtype=np.float32)
            last_gae = 0.0

            for t in reversed(range(n)):
                if t == n - 1:
                    next_value = last_values[env_idx]
                else:
                    next_value = traj.values[t + 1]

                next_non_terminal = 1.0 - float(traj.dones[t])
                delta = traj.rewards[t] + self.gamma * next_value * next_non_terminal - traj.values[t]
                last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
                advantages[t] = last_gae

            returns = advantages + np.array(traj.values, dtype=np.float32)

            # Flatten into combined lists
            self._flat_tokens.extend(traj.tokens)
            self._flat_token_types.extend(traj.token_types)
            self._flat_scalars.extend(traj.scalars)
            self._flat_attention_masks.extend(traj.attention_masks)
            self._flat_action_masks.extend(traj.action_masks)
            self._flat_actions.extend(traj.actions)
            self._flat_log_probs.extend(traj.log_probs)
            all_advantages.append(advantages)
            all_returns.append(returns)

        self._flat_advantages = np.concatenate(all_advantages) if all_advantages else np.array([], dtype=np.float32)
        self._flat_returns = np.concatenate(all_returns) if all_returns else np.array([], dtype=np.float32)

    def get_batches(self, batch_size: int, device: torch.device) -> list[dict[str, torch.Tensor]]:
        """Yield shuffled mini-batches as tensors."""
        n = len(self._flat_actions)
        if n == 0:
            return []
        indices = np.random.permutation(n)

        batches = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            batch = {
                "tokens": torch.tensor(np.array([self._flat_tokens[i] for i in idx]), dtype=torch.int64, device=device),
                "token_types": torch.tensor(np.array([self._flat_token_types[i] for i in idx]), dtype=torch.long, device=device),
                "scalars": torch.tensor(np.array([self._flat_scalars[i] for i in idx]), dtype=torch.float32, device=device),
                "attention_mask": torch.tensor(np.array([self._flat_attention_masks[i] for i in idx]), dtype=torch.long, device=device),
                "action_mask": torch.tensor(np.array([self._flat_action_masks[i] for i in idx]), dtype=torch.float32, device=device),
                "actions": torch.tensor([self._flat_actions[i] for i in idx], dtype=torch.long, device=device),
                "old_log_probs": torch.tensor([self._flat_log_probs[i] for i in idx], dtype=torch.float32, device=device),
                "advantages": torch.tensor(self._flat_advantages[idx], dtype=torch.float32, device=device),
                "returns": torch.tensor(self._flat_returns[idx], dtype=torch.float32, device=device),
            }
            batches.append(batch)
        return batches

    def total_transitions(self) -> int:
        return sum(len(t) for t in self.envs)

    def clear(self) -> None:
        self.envs = [EnvTrajectory() for _ in range(self.num_envs)]
        self._flat_tokens.clear()
        self._flat_token_types.clear()
        self._flat_scalars.clear()
        self._flat_attention_masks.clear()
        self._flat_action_masks.clear()
        self._flat_actions.clear()
        self._flat_log_probs.clear()
        self._flat_advantages = None
        self._flat_returns = None
