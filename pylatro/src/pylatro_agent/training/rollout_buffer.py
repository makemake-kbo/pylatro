"""Experience storage for PPO training."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..constants import MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM


@dataclass
class RolloutBuffer:
    """Stores trajectories for PPO updates."""

    buffer_size: int = 8192
    gamma: float = 0.995
    gae_lambda: float = 0.95

    # Storage
    tokens: list[np.ndarray] = field(default_factory=list)
    token_types: list[np.ndarray] = field(default_factory=list)
    scalars: list[np.ndarray] = field(default_factory=list)
    attention_masks: list[np.ndarray] = field(default_factory=list)
    action_masks: list[np.ndarray] = field(default_factory=list)
    selected_cards: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    # Computed
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None

    def add(
        self,
        obs: dict,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        self.tokens.append(obs["tokens"])
        self.token_types.append(obs["token_types"])
        self.scalars.append(obs["scalars"])
        self.attention_masks.append(obs["attention_mask"])
        self.action_masks.append(obs["action_mask"])
        self.selected_cards.append(obs["selected_cards"])
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value: float = 0.0) -> None:
        """Compute GAE advantages and discounted returns."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])

            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + np.array(self.values, dtype=np.float32)

    def get_batches(self, batch_size: int, device: torch.device) -> list[dict[str, torch.Tensor]]:
        """Yield mini-batches as tensors."""
        n = len(self.actions)
        indices = np.random.permutation(n)

        batches = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            batch = {
                "tokens": torch.tensor(np.array([self.tokens[i] for i in idx]), dtype=torch.int64, device=device),
                "token_types": torch.tensor(np.array([self.token_types[i] for i in idx]), dtype=torch.long, device=device),
                "scalars": torch.tensor(np.array([self.scalars[i] for i in idx]), dtype=torch.float32, device=device),
                "attention_mask": torch.tensor(np.array([self.attention_masks[i] for i in idx]), dtype=torch.long, device=device),
                "action_mask": torch.tensor(np.array([self.action_masks[i] for i in idx]), dtype=torch.float32, device=device),
                "actions": torch.tensor([self.actions[i] for i in idx], dtype=torch.long, device=device),
                "old_log_probs": torch.tensor([self.log_probs[i] for i in idx], dtype=torch.float32, device=device),
                "advantages": torch.tensor(self.advantages[idx], dtype=torch.float32, device=device),
                "returns": torch.tensor(self.returns[idx], dtype=torch.float32, device=device),
            }
            batches.append(batch)
        return batches

    def clear(self) -> None:
        self.tokens.clear()
        self.token_types.clear()
        self.scalars.clear()
        self.attention_masks.clear()
        self.action_masks.clear()
        self.selected_cards.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.advantages = None
        self.returns = None

    def __len__(self) -> int:
        return len(self.actions)
