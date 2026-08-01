"""Experience storage for PPO training with pre-allocated arrays and correct per-env GAE."""

from __future__ import annotations

import numpy as np
import torch

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
from ..survival import DEFAULT_MAX_ANTES


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
        gamma: float = 0.997,
        gae_lambda: float = 0.97,
    ) -> None:
        self.num_envs = num_envs
        self.rollout_length = rollout_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.total_size = num_envs * rollout_length

        # Pre-allocate observation arrays, shape: (total, ...)
        self.tokens = np.zeros((self.total_size, MAX_SEQ_LEN, TOKEN_DIM), dtype=np.int16)
        self.token_types = np.zeros((self.total_size, MAX_SEQ_LEN), dtype=np.int8)
        self.scalars = np.zeros((self.total_size, SCALAR_DIM), dtype=np.float32)
        self.attention_masks = np.zeros((self.total_size, MAX_SEQ_LEN), dtype=np.int8)
        self.action_masks = np.zeros((self.total_size, NUM_ACTIONS), dtype=np.float32)
        self.history_events = np.zeros(
            (self.total_size, HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_EVENT_DIM), dtype=np.int16
        )
        self.history_event_features = np.zeros(
            (self.total_size, HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_FEATURE_DIM), dtype=np.float32
        )
        self.history_cards = np.zeros(
            (
                self.total_size,
                HISTORY_ROUNDS,
                HISTORY_MAX_PLAYS,
                HISTORY_MAX_CARDS,
                TOKEN_DIM,
            ),
            dtype=np.int16,
        )
        self.history_card_masks = np.zeros(
            (self.total_size, HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_CARDS), dtype=np.int8
        )
        self.history_jokers = np.zeros(
            (self.total_size, HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_JOKERS), dtype=np.int16
        )
        self.history_joker_masks = np.zeros(
            (self.total_size, HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_JOKERS), dtype=np.int8
        )
        self.history_event_masks = np.zeros(
            (self.total_size, HISTORY_ROUNDS, HISTORY_MAX_PLAYS), dtype=np.int8
        )
        self.history_round_masks = np.zeros((self.total_size, HISTORY_ROUNDS), dtype=np.int8)
        self.history_omitted = np.zeros(
            (self.total_size, HISTORY_ROUNDS, HISTORY_OMITTED_DIM), dtype=np.float32
        )

        # Pre-allocate action/value arrays, shape: (total,)
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

        # Per-step ante-survival aux targets, filled retroactively at
        # episode-end by `set_episode_survival`. Defaults to mask=0 so
        # unfinished episodes contribute nothing to the survival loss.
        self.ante_survival_targets = np.zeros((self.total_size, DEFAULT_MAX_ANTES), dtype=np.float32)
        self.ante_survival_masks = np.zeros((self.total_size, DEFAULT_MAX_ANTES), dtype=np.float32)

        # Write pointer per env
        self._step_counts = np.zeros(num_envs, dtype=np.int64)

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
        for target, key in (
            (self.history_events, "history_events"),
            (self.history_event_features, "history_event_features"),
            (self.history_cards, "history_cards"),
            (self.history_card_masks, "history_card_mask"),
            (self.history_jokers, "history_jokers"),
            (self.history_joker_masks, "history_joker_mask"),
            (self.history_event_masks, "history_event_mask"),
            (self.history_round_masks, "history_round_mask"),
            (self.history_omitted, "history_omitted"),
        ):
            if key in obs:
                target[indices] = obs[key]
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.values[indices] = values
        self.log_probs[indices] = log_probs
        self.terminated[indices] = terminated
        self.truncated[indices] = truncated
        self.bootstrap_values[indices] = bootstrap_values

        self._step_counts[:] = step + 1

    def set_episode_survival(
        self,
        env_idx: int,
        start_step: int,
        end_step: int,
        target: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        """Fill [start_step, end_step] inclusive of env_idx with survival target + mask."""
        base = env_idx * self.rollout_length
        lo = base + start_step
        hi = base + end_step + 1
        self.ante_survival_targets[lo:hi] = target
        self.ante_survival_masks[lo:hi] = mask

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
                    next_value = last_values[env_idx] if t == n - 1 else env_values[t + 1]
                    bootstrap_mask = 1.0
                    gae_continue_mask = 1.0

                delta = env_rewards[t] + self.gamma * next_value * bootstrap_mask - env_values[t]
                last_gae = delta + self.gamma * self.gae_lambda * gae_continue_mask * last_gae
                self.advantages[idx] = last_gae

            self.returns[start:end] = self.advantages[start:end] + env_values[:n]

    def normalize_advantages(self, eps: float = 1e-8, clip_sigma: float = 0.0) -> None:
        """Normalize advantages across the full rollout (not per mini-batch).

        Per-mini-batch normalization lets rare high-magnitude transitions
        (e.g. terminal rewards in a sparse-win setting) dominate only the
        batch they land in, while batches without terminals see shaping
        noise blown up to unit variance. Normalizing once over all valid
        transitions keeps the relative scale of wins/losses vs shaping
        consistent across every mini-batch.

        ``clip_sigma`` > 0 clamps the normalized advantages to that many
        standard deviations. Near-terminal coin-flip states carry mostly
        aleatoric outcome noise in their advantages (measured: the top 1%
        of |adv| states, median 4 steps from terminal, carried ~20% of
        sum(adv^2) at 4.8 excess kurtosis); the clamp caps their gradient
        share without touching the bulk of the distribution.
        """
        valid: list[np.ndarray] = []
        for env_idx in range(self.num_envs):
            start = env_idx * self.rollout_length
            n = int(self._step_counts[env_idx]) if self._step_counts[env_idx] > 0 else self.rollout_length
            valid.append(self.advantages[start:start + n])
        if not valid:
            return
        flat = np.concatenate(valid)
        if flat.size == 0:
            return
        mean = float(flat.mean())
        std = float(flat.std())
        scale = 1.0 / (std + eps)
        for env_idx in range(self.num_envs):
            start = env_idx * self.rollout_length
            n = int(self._step_counts[env_idx]) if self._step_counts[env_idx] > 0 else self.rollout_length
            normalized = (self.advantages[start:start + n] - mean) * scale
            if clip_sigma > 0.0:
                np.clip(normalized, -clip_sigma, clip_sigma, out=normalized)
            self.advantages[start:start + n] = normalized

    def get_batches(
        self,
        batch_size: int,
        device: torch.device,
        pin_memory: bool = False,
    ) -> list[dict[str, torch.Tensor]]:
        """Return shuffled mini-batches as tensors.

        Uses numpy fancy indexing on pre-allocated arrays, no Python list
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
                "history_events": torch.as_tensor(self.history_events[idx].astype(np.int64), device=device),
                "history_event_features": torch.as_tensor(self.history_event_features[idx], device=device),
                "history_cards": torch.as_tensor(self.history_cards[idx].astype(np.int64), device=device),
                "history_card_mask": torch.as_tensor(
                    self.history_card_masks[idx].astype(np.int64), device=device
                ),
                "history_jokers": torch.as_tensor(self.history_jokers[idx].astype(np.int64), device=device),
                "history_joker_mask": torch.as_tensor(
                    self.history_joker_masks[idx].astype(np.int64), device=device
                ),
                "history_event_mask": torch.as_tensor(
                    self.history_event_masks[idx].astype(np.int64), device=device
                ),
                "history_round_mask": torch.as_tensor(
                    self.history_round_masks[idx].astype(np.int64), device=device
                ),
                "history_omitted": torch.as_tensor(self.history_omitted[idx], device=device),
                "actions": torch.as_tensor(self.actions[idx], device=device),
                "old_log_probs": torch.as_tensor(self.log_probs[idx], device=device),
                "advantages": torch.as_tensor(self.advantages[idx], device=device),
                "returns": torch.as_tensor(self.returns[idx], device=device),
                "ante_survival_target": torch.as_tensor(self.ante_survival_targets[idx], device=device),
                "ante_survival_mask": torch.as_tensor(self.ante_survival_masks[idx], device=device),
            }

            if pin_memory and device.type == "cpu":
                batch = {k: v.pin_memory() for k, v in batch.items()}

            batches.append(batch)
        return batches

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

    @property
    def _flat_values(self) -> np.ndarray:
        """Valid-step value predictions, aligned index-for-index with _flat_returns."""
        valid = []
        for env_idx in range(self.num_envs):
            start = env_idx * self.rollout_length
            n = int(self._step_counts[env_idx]) if self._step_counts[env_idx] > 0 else self.rollout_length
            valid.append(self.values[start:start + n])
        return np.concatenate(valid) if valid else np.array([], dtype=np.float32)
