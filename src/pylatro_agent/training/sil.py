"""Self-imitation learning (SIL) on the agent's own winning episodes.

At high win-ante targets PPO sees only a handful of wins per update, and
each win teaches for exactly one update before its rollout is discarded.
SIL keeps a FIFO buffer of complete winning trajectories and replays them
with a behavior-cloning loss alongside PPO, multiplying the effective
density of the win signal without touching the reward function: nothing
enters the buffer except a genuine win, so there is no shaping to farm.

Episodes routinely span rollout boundaries (episode length ~100 steps vs
rollout_length per env), so transitions are accumulated per env in
``SILEpisodeTracker`` across updates rather than sliced out of the
per-update ``RolloutBuffer``.
"""

from __future__ import annotations

import numpy as np
import torch

from ..constants import NUM_ACTIONS

_MASK_PACKED_BYTES = (NUM_ACTIONS + 7) // 8


class WinEpisodeBuffer:
    """FIFO buffer of complete winning episodes, stored compactly.

    Observations use the RolloutBuffer's compact dtypes (int16 tokens, int8
    masks); the flat action mask (NUM_ACTIONS float32) is bit-packed to 1/32
    of its rollout-buffer size so a default-sized buffer stays tens of MB.
    """

    def __init__(self, capacity_episodes: int, seed: int | None = None) -> None:
        if capacity_episodes <= 0:
            raise ValueError("capacity_episodes must be positive")
        self.capacity_episodes = capacity_episodes
        self._episodes: list[dict[str, np.ndarray]] = []
        self._num_transitions = 0
        self._rng = np.random.default_rng(seed)
        self.episodes_added_total = 0

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def num_transitions(self) -> int:
        return self._num_transitions

    def add_episode(self, episode: dict[str, np.ndarray]) -> None:
        steps = int(episode["actions"].shape[0])
        if steps == 0:
            return
        self._episodes.append(episode)
        self._num_transitions += steps
        self.episodes_added_total += 1
        while len(self._episodes) > self.capacity_episodes:
            evicted = self._episodes.pop(0)
            self._num_transitions -= int(evicted["actions"].shape[0])

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        """Sample transitions uniformly (with replacement) across the buffer.

        Returns a model batch with the same keys/dtypes as
        ``RolloutBuffer.get_batches`` (minus the PPO-only fields).
        """
        if self._num_transitions == 0:
            raise ValueError("cannot sample from an empty WinEpisodeBuffer")
        lengths = np.asarray([ep["actions"].shape[0] for ep in self._episodes], dtype=np.int64)
        cumulative = np.cumsum(lengths)
        flat_idx = self._rng.integers(0, self._num_transitions, size=batch_size)
        ep_idx = np.searchsorted(cumulative, flat_idx, side="right")
        step_idx = flat_idx - (cumulative[ep_idx] - lengths[ep_idx])
        picks = list(zip(ep_idx, step_idx, strict=True))

        tokens = np.stack([self._episodes[e]["tokens"][t] for e, t in picks])
        token_types = np.stack([self._episodes[e]["token_types"][t] for e, t in picks])
        scalars = np.stack([self._episodes[e]["scalars"][t] for e, t in picks])
        attention_mask = np.stack([self._episodes[e]["attention_mask"][t] for e, t in picks])
        packed = np.stack([self._episodes[e]["action_mask_packed"][t] for e, t in picks])
        action_mask = np.unpackbits(packed, axis=1, count=NUM_ACTIONS).astype(np.float32)
        actions = np.asarray([self._episodes[e]["actions"][t] for e, t in picks], dtype=np.int64)

        return {
            "tokens": torch.as_tensor(tokens.astype(np.int64), device=device),
            "token_types": torch.as_tensor(token_types.astype(np.int64), device=device),
            "scalars": torch.as_tensor(scalars, device=device),
            "attention_mask": torch.as_tensor(attention_mask.astype(np.int64), device=device),
            "action_mask": torch.as_tensor(action_mask, device=device),
            "actions": torch.as_tensor(actions, device=device),
        }


class SILEpisodeTracker:
    """Accumulate per-env transitions across rollouts; emit complete wins.

    ``record_step`` must be called once per vector-env step with the
    pre-step observations (the same arrays handed to the rollout buffer);
    ``finish_episode`` at each done. Episodes longer than
    ``max_episode_steps`` are dropped rather than growing unbounded.
    """

    def __init__(self, num_envs: int, max_episode_steps: int = 2048) -> None:
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self._steps: list[list[tuple]] = [[] for _ in range(num_envs)]
        self._overflowed = [False] * num_envs

    def record_step(self, obs: dict[str, np.ndarray], actions: np.ndarray) -> None:
        for i in range(self.num_envs):
            if self._overflowed[i]:
                continue
            if len(self._steps[i]) >= self.max_episode_steps:
                self._overflowed[i] = True
                self._steps[i] = []
                continue
            self._steps[i].append(
                (
                    obs["tokens"][i].astype(np.int16),
                    obs["token_types"][i].astype(np.int8),
                    obs["scalars"][i].astype(np.float32),
                    obs["attention_mask"][i].astype(np.int8),
                    np.packbits(obs["action_mask"][i] > 0.5),
                    int(actions[i]),
                )
            )

    def finish_episode(self, env_idx: int, won: bool, buffer: WinEpisodeBuffer) -> None:
        steps = self._steps[env_idx]
        overflowed = self._overflowed[env_idx]
        self._steps[env_idx] = []
        self._overflowed[env_idx] = False
        if not won or overflowed or not steps:
            return
        buffer.add_episode(
            {
                "tokens": np.stack([s[0] for s in steps]),
                "token_types": np.stack([s[1] for s in steps]),
                "scalars": np.stack([s[2] for s in steps]),
                "attention_mask": np.stack([s[3] for s in steps]),
                "action_mask_packed": np.stack([s[4] for s in steps]),
                "actions": np.asarray([s[5] for s in steps], dtype=np.int64),
            }
        )
