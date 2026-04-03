"""Experience storage for PPO training with pre-allocated arrays and correct per-env GAE."""

from __future__ import annotations

import numpy as np
import torch

from ..constants import MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM


class RolloutBuffer:
    """Pre-allocated rollout storage with per-env GAE computation.

    All observation/action data is stored in flat pre-allocated numpy arrays
    indexed by (env_idx * rollout_length + step). This avoids per-step Python
    list appends and makes batch construction a single numpy fancy-index.
    """

    def __init__(
        self,
        num_envs: int,
        rollout_length: int,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
    ) -> None:
        self.num_envs = num_envs
        self.rollout_length = rollout_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.total_size = num_envs * rollout_length

        # Pre-allocate observation arrays — shape: (total, ...)
        self.tokens = np.zeros((self.total_size, MAX_SEQ_LEN, TOKEN_DIM), dtype=np.int16)
        self.token_types = np.zeros((self.total_size, MAX_SEQ_LEN), dtype=np.int8)
        self.scalars = np.zeros((self.total_size, SCALAR_DIM), dtype=np.float32)
        self.attention_masks = np.zeros((self.total_size, MAX_SEQ_LEN), dtype=np.int8)
        self.action_masks = np.zeros((self.total_size, NUM_ACTIONS), dtype=np.float32)

        # Pre-allocate action/value arrays — shape: (total,)
        self.actions = np.zeros(self.total_size, dtype=np.int64)
        self.rewards = np.zeros(self.total_size, dtype=np.float32)
        self.values = np.zeros(self.total_size, dtype=np.float32)
        self.log_probs = np.zeros(self.total_size, dtype=np.float32)
        self.terminated = np.zeros(self.total_size, dtype=np.bool_)
        self.truncated = np.zeros(self.total_size, dtype=np.bool_)
        self.bootstrap_values = np.zeros(self.total_size, dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros(self.total_size, dtype=np.float32)
        self.returns = np.zeros(self.total_size, dtype=np.float32)

        # Write pointer per env
        self._step_counts = np.zeros(num_envs, dtype=np.int64)

    def _index(self, env_idx: int, step: int) -> int:
        return env_idx * self.rollout_length + step

    def add(
        self,
        env_idx: int,
        obs: dict,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        terminated: bool,
        truncated: bool,
        bootstrap_value: float = 0.0,
    ) -> None:
        """Store one transition for one environment."""
        step = self._step_counts[env_idx]
        idx = self._index(env_idx, step)

        self.tokens[idx] = obs["tokens"]
        self.token_types[idx] = obs["token_types"]
        self.scalars[idx] = obs["scalars"]
        self.attention_masks[idx] = obs["attention_mask"]
        self.action_masks[idx] = obs["action_mask"]
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.terminated[idx] = terminated
        self.truncated[idx] = truncated
        self.bootstrap_values[idx] = bootstrap_value

        self._step_counts[env_idx] = step + 1

    def add_batch(
        self,
        step: int,
        obs: dict,
        actions: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        bootstrap_values: np.ndarray | None = None,
    ) -> None:
        """Store one timestep for all environments at once (vectorized)."""
        indices = np.arange(self.num_envs) * self.rollout_length + step
        if bootstrap_values is None:
            bootstrap_values = np.zeros(self.num_envs, dtype=np.float32)

        self.tokens[indices] = obs["tokens"]
        self.token_types[indices] = obs["token_types"]
        self.scalars[indices] = obs["scalars"]
        self.attention_masks[indices] = obs["attention_mask"]
        self.action_masks[indices] = obs["action_mask"]
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.values[indices] = values
        self.log_probs[indices] = log_probs
        self.terminated[indices] = terminated
        self.truncated[indices] = truncated
        self.bootstrap_values[indices] = bootstrap_values

        self._step_counts[:] = step + 1

    def compute_returns_and_advantages(self, last_values: np.ndarray | list[float]) -> None:
        """Compute GAE advantages per env, storing into pre-allocated arrays.

        Args:
            last_values: bootstrap value for each env (length num_envs).
        """
        last_values = np.asarray(last_values, dtype=np.float32)

        for env_idx in range(self.num_envs):
            start = env_idx * self.rollout_length
            n = int(self._step_counts[env_idx]) if self._step_counts[env_idx] > 0 else self.rollout_length
            end = start + n

            env_rewards = self.rewards[start:end]
            env_values = self.values[start:end]
            env_terminated = self.terminated[start:end]
            env_truncated = self.truncated[start:end]
            env_bootstrap_values = self.bootstrap_values[start:end]

            last_gae = 0.0
            for t in reversed(range(n)):
                idx = start + t
                if env_terminated[t]:
                    next_value = 0.0
                    bootstrap_mask = 0.0
                    gae_continue_mask = 0.0
                elif env_truncated[t]:
                    next_value = env_bootstrap_values[t]
                    bootstrap_mask = 1.0
                    # Truncation ends the episode but still bootstraps from the
                    # final observation; do not leak GAE across the reset boundary.
                    gae_continue_mask = 0.0
                else:
                    if t == n - 1:
                        next_value = last_values[env_idx]
                    else:
                        next_value = env_values[t + 1]
                    bootstrap_mask = 1.0
                    gae_continue_mask = 1.0

                delta = env_rewards[t] + self.gamma * next_value * bootstrap_mask - env_values[t]
                last_gae = delta + self.gamma * self.gae_lambda * gae_continue_mask * last_gae
                self.advantages[idx] = last_gae

            self.returns[start:end] = self.advantages[start:end] + env_values[:n]

    def get_batches(
        self,
        batch_size: int,
        device: torch.device,
        pin_memory: bool = False,
    ) -> list[dict[str, torch.Tensor]]:
        """Return shuffled mini-batches as tensors.

        Uses numpy fancy indexing on pre-allocated arrays — no Python list
        comprehensions over individual transitions.
        """
        # Determine valid range (in case envs didn't all fill rollout_length)
        valid_indices = []
        for env_idx in range(self.num_envs):
            start = env_idx * self.rollout_length
            n = int(self._step_counts[env_idx]) if self._step_counts[env_idx] > 0 else self.rollout_length
            valid_indices.append(np.arange(start, start + n))
        all_indices = np.concatenate(valid_indices)
        n = len(all_indices)
        if n == 0:
            return []

        shuffled = np.random.permutation(all_indices)

        batches = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = shuffled[start:end]

            batch = {
                "tokens": torch.as_tensor(self.tokens[idx].astype(np.int64), device=device),
                "token_types": torch.as_tensor(self.token_types[idx].astype(np.int64), device=device),
                "scalars": torch.as_tensor(self.scalars[idx], device=device),
                "attention_mask": torch.as_tensor(self.attention_masks[idx].astype(np.int64), device=device),
                "action_mask": torch.as_tensor(self.action_masks[idx], device=device),
                "actions": torch.as_tensor(self.actions[idx], device=device),
                "old_log_probs": torch.as_tensor(self.log_probs[idx], device=device),
                "advantages": torch.as_tensor(self.advantages[idx], device=device),
                "returns": torch.as_tensor(self.returns[idx], device=device),
            }

            if pin_memory and device.type == "cpu":
                batch = {k: v.pin_memory() for k, v in batch.items()}

            batches.append(batch)
        return batches

    def total_transitions(self) -> int:
        return int(self._step_counts.sum())

    @property
    def _flat_returns(self) -> np.ndarray:
        """Compatibility property for logging."""
        valid = []
        for env_idx in range(self.num_envs):
            start = env_idx * self.rollout_length
            n = int(self._step_counts[env_idx]) if self._step_counts[env_idx] > 0 else self.rollout_length
            valid.append(self.returns[start:start + n])
        return np.concatenate(valid) if valid else np.array([], dtype=np.float32)

    @property
    def _flat_advantages(self) -> np.ndarray:
        """Compatibility property for logging."""
        valid = []
        for env_idx in range(self.num_envs):
            start = env_idx * self.rollout_length
            n = int(self._step_counts[env_idx]) if self._step_counts[env_idx] > 0 else self.rollout_length
            valid.append(self.advantages[start:start + n])
        return np.concatenate(valid) if valid else np.array([], dtype=np.float32)
