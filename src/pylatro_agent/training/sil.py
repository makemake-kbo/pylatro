"""Self-imitation learning (SIL) auxiliary actor loss.

SIL is a small, bounded, critic-gated auxiliary actor loss trained alongside
PPO in the same backward/optimizer step. The earlier design cloned every
winning trajectory with an advantage gate that, in practice, saturated at full
weight and turned SIL into near-on-policy self-cloning of the average recent
win. This module implements the redesign:

* an :class:`EpisodeReplayBuffer` that keeps *all* completed non-stalled
  episodes (wins and ordinary losses), not just wins;
* episode-uniform replay sampling with a per-episode transition cap, so a long
  episode is no more likely to be replayed than a short one;
* a single shared percentile advantage gate (:func:`sil_percentile_gate`) used
  by both PPO training and the offline critic probe, so the two cannot drift.

The SIL loss is actor-only: it never adds a value term. ``sil_coeff == 0`` is
the exact no-SIL switch and preserves the historical PPO path.

Episodes routinely span rollout boundaries (episode length ~100 steps vs
rollout_length per env), so transitions are accumulated per env in
:class:`SILEpisodeTracker` across updates rather than sliced out of the
per-update :class:`~pylatro_agent.training.rollout_buffer.RolloutBuffer`.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..constants import NUM_ACTIONS

_MASK_PACKED_BYTES = (NUM_ACTIONS + 7) // 8


def sil_percentile_gate(
    raw_advantages: np.ndarray,
    *,
    open_percentile: float,
    saturation_percentile: float,
    advantage_floor: float,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute SIL gate weights from raw (reward-unit) advantages.

    This is the single source of truth for the SIL gate; PPO training and the
    offline critic probe both call it so their thresholds cannot diverge.

    For each otherwise-eligible sampled transition the raw advantage is
    ``R_mc - stop_gradient(V_current)`` in raw reward units. The gate is then::

        q_open       = percentile(adv, open_percentile)
        q_saturation = percentile(adv, saturation_percentile)
        open_threshold = max(advantage_floor, q_open)

        if q_saturation <= open_threshold + epsilon:
            gate = zeros          # degenerate range: no forced signal
        else:
            gate = clip((adv - open_threshold)
                        / (q_saturation - open_threshold), 0, 1)

    Consequences:

    * critic-overvalued transitions (negative advantage) receive zero weight;
    * sub-floor positive advantages receive zero weight;
    * the gate opens around ``open_percentile`` *only when that percentile
      exceeds the absolute floor*;
    * the gate reaches 1 around ``saturation_percentile``;
    * a batch with no advantage above the floor does not manufacture an active
      top tail (degenerate ranges produce no SIL signal);
    * percentiles are computed over ALL supplied advantages, not only the
      already-positive tail.

    Args:
        raw_advantages: 1-D array of raw advantages for every otherwise-eligible
            row (teacher-forced and non-finite-log-prob rows are filtered out
            by the caller before this call).

    Returns:
        ``(gate, info)`` where ``gate`` is a float32 array the same length as
        the input and ``info`` carries ``open_threshold``, ``saturation_threshold``,
        ``q_open`` and ``q_saturation`` for diagnostics.
    """
    adv = np.asarray(raw_advantages, dtype=np.float64).reshape(-1)
    if adv.size == 0:
        info = {
            "open_threshold": float("nan"),
            "saturation_threshold": float("nan"),
            "q_open": float("nan"),
            "q_saturation": float("nan"),
        }
        return np.zeros(0, dtype=np.float32), info
    q_open = float(np.percentile(adv, open_percentile))
    q_saturation = float(np.percentile(adv, saturation_percentile))
    open_threshold = max(float(advantage_floor), q_open)
    if q_saturation <= open_threshold + epsilon:
        gate = np.zeros(adv.shape, dtype=np.float32)
    else:
        span = q_saturation - open_threshold
        gate = np.clip((adv - open_threshold) / span, 0.0, 1.0).astype(np.float32)
    info = {
        "open_threshold": float(open_threshold),
        "saturation_threshold": float(q_saturation),
        "q_open": q_open,
        "q_saturation": q_saturation,
    }
    return gate, info


class EpisodeReplayBuffer:
    """Bounded FIFO buffer of completed non-stalled episodes (wins and losses).

    Stores every completed episode the environment reports as non-stalled: wins
    *and* ordinary losses. Infrastructure/no-progress stalls (as identified by
    the environment metadata) are never inserted, so a legal policy-caused loss
    stays in replay while a hung env episode does not.

    Observations use the RolloutBuffer's compact dtypes (int16 tokens, int8
    masks); the flat action mask (NUM_ACTIONS float32) is bit-packed to 1/32
    of its rollout-buffer size so a default-sized buffer stays tens of MB.

    Sampling is *episode-uniform* with a per-episode transition cap: each
    eligible episode is equally likely to be selected, and no episode
    contributes more than ``samples_per_episode`` transitions to one batch. A
    long episode is therefore not privileged over a short one merely because it
    contains more transitions.
    """

    # Compact observation/mask fields stored per transition.
    _COMPACT_FIELDS = ("tokens", "token_types", "scalars", "attention_mask", "action_mask_packed")

    def __init__(self, capacity_episodes: int, seed: int | None = None) -> None:
        if capacity_episodes <= 0:
            raise ValueError("capacity_episodes must be positive")
        self.capacity_episodes = capacity_episodes
        self._episodes: list[dict] = []
        self._num_transitions = 0
        self._rng = np.random.default_rng(seed)
        # Monotonic episode id assigned in insertion order.
        self._next_episode_id = 0
        self.episodes_added_total = 0
        self.stalled_episodes_dropped_total = 0
        self.overflow_episodes_dropped_total = 0

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def num_transitions(self) -> int:
        return self._num_transitions

    @property
    def num_wins(self) -> int:
        return int(sum(1 for ep in self._episodes if ep["won"]))

    @property
    def win_fraction(self) -> float:
        if not self._episodes:
            return 0.0
        return self.num_wins / len(self._episodes)

    def add_episode(self, episode: dict) -> None:
        """Insert a completed episode dict (already validated as non-stalled).

        Caller-filtered teacher-forced rows stay in the episode; they are only
        excluded from sampling/loss when ``include_teacher_forced`` is False.
        """
        steps = int(episode["actions"].shape[0])
        if steps == 0:
            return
        episode = dict(episode)
        episode.setdefault("teacher_forced", np.zeros(steps, dtype=np.int8))
        episode["episode_id"] = self._next_episode_id
        self._next_episode_id += 1
        self._episodes.append(episode)
        self._num_transitions += steps
        self.episodes_added_total += 1
        while len(self._episodes) > self.capacity_episodes:
            evicted = self._episodes.pop(0)
            self._num_transitions -= int(evicted["actions"].shape[0])

    def _eligible_row_indices(self, episode: dict, *, include_teacher_forced: bool) -> np.ndarray:
        """Rows usable for SIL loss / gate calibration under the teacher filter."""
        if include_teacher_forced:
            return np.arange(episode["actions"].shape[0], dtype=np.int64)
        teacher = np.asarray(episode.get("teacher_forced"), dtype=np.int8)
        return np.flatnonzero(teacher == 0).astype(np.int64)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        *,
        samples_per_episode: int = 8,
        only_wins: bool = False,
        include_teacher_forced: bool = False,
    ) -> dict[str, torch.Tensor] | None:
        """Sample an episode-uniform batch with a per-episode transition cap.

        Selects eligible episodes uniformly at random, gives selected episodes
        near-equal transition quotas, and samples transitions uniformly without
        replacement inside each episode. Never exceeds ``samples_per_episode``
        rows from one episode. Returns fewer than ``batch_size`` rows rather
        than violating the cap, and returns ``None`` when no eligible episode
        has any eligible row.

        For advantage SIL all non-stalled completed episodes are eligible; for
        winning behavior cloning only winning episodes are eligible
        (``only_wins=True``).

        The returned dict carries the model batch keys plus metadata arrays:
        ``episode_ids``, ``episode_outcomes`` (1.0 win / 0.0 loss),
        ``teacher_forced_flags``, and ``episode_returns`` (each row's episode
        total shaped return) so callers can test and log where gate mass lands.
        """
        if samples_per_episode <= 0:
            raise ValueError("samples_per_episode must be positive")
        eligible: list[tuple[dict, np.ndarray]] = []
        for ep in self._episodes:
            if only_wins and not ep["won"]:
                continue
            rows = self._eligible_row_indices(ep, include_teacher_forced=include_teacher_forced)
            if rows.size > 0:
                eligible.append((ep, rows))
        if not eligible:
            return None

        order = self._rng.permutation(len(eligible))
        selected = [eligible[i] for i in order]
        n = len(selected)
        # Near-equal quota per selected episode, capped at samples_per_episode.
        quota = min(samples_per_episode, max(1, math.ceil(batch_size / n)))
        picks: list[tuple[dict, int]] = []
        for ep, rows in selected:
            k = min(quota, rows.size)
            chosen = self._rng.choice(rows, size=k, replace=False)
            for r in chosen:
                picks.append((ep, int(r)))
            if len(picks) >= batch_size:
                break
        if len(picks) > batch_size:
            # Random sub-sample so truncation does not systematically drop the
            # last-selected episode; keeps near-equal contributions.
            idx = self._rng.permutation(len(picks))[:batch_size]
            picks = [picks[i] for i in idx]

        return self._materialize_batch(picks, device)

    def _materialize_batch(self, picks: list[tuple[dict, int]], device: torch.device) -> dict[str, torch.Tensor]:
        tokens = np.stack([ep["tokens"][t] for ep, t in picks])
        token_types = np.stack([ep["token_types"][t] for ep, t in picks])
        scalars = np.stack([ep["scalars"][t] for ep, t in picks])
        attention_mask = np.stack([ep["attention_mask"][t] for ep, t in picks])
        packed = np.stack([ep["action_mask_packed"][t] for ep, t in picks])
        action_mask = np.unpackbits(packed, axis=1, count=NUM_ACTIONS).astype(np.float32)
        actions = np.asarray([ep["actions"][t] for ep, t in picks], dtype=np.int64)
        returns = np.asarray([ep["returns"][t] for ep, t in picks], dtype=np.float32)
        teacher = np.asarray(
            [float(np.asarray(ep.get("teacher_forced"))[t]) for ep, t in picks],
            dtype=np.float32,
        )
        episode_ids = np.asarray([float(ep["episode_id"]) for ep, t in picks], dtype=np.float32)
        outcomes = np.asarray([1.0 if ep["won"] else 0.0 for ep, t in picks], dtype=np.float32)
        episode_returns = np.asarray([float(ep["total_reward"]) for ep, t in picks], dtype=np.float32)
        return {
            "tokens": torch.as_tensor(tokens.astype(np.int64), device=device),
            "token_types": torch.as_tensor(token_types.astype(np.int64), device=device),
            "scalars": torch.as_tensor(scalars, device=device),
            "attention_mask": torch.as_tensor(attention_mask.astype(np.int64), device=device),
            "action_mask": torch.as_tensor(action_mask, device=device),
            "actions": torch.as_tensor(actions, device=device),
            "returns": torch.as_tensor(returns, device=device),
            "teacher_forced_flags": torch.as_tensor(teacher, device=device),
            "episode_ids": torch.as_tensor(episode_ids, device=device),
            "episode_outcomes": torch.as_tensor(outcomes, device=device),
            "episode_returns": torch.as_tensor(episode_returns, device=device),
        }


class SILEpisodeTracker:
    """Accumulate per-env transitions across rollouts; emit complete episodes.

    ``record_step`` must be called once per vector-env step with the pre-step
    observations (the same arrays handed to the rollout buffer), the executed
    actions, the rewards, and the per-env teacher-forced mask. The executed
    action may have been teacher-forced; that provenance is recorded so SIL can
    exclude teacher-forced rows from the actor loss by default.
    ``finish_episode`` at each done.

    Episodes longer than ``max_episode_steps`` are dropped (overflow) rather
    than growing unbounded. On completion, wins and ordinary losses are inserted
    into the replay buffer; episodes the environment flags as infrastructure
    stalls are dropped. ``gamma`` must match the PPO discount so the stored
    return-to-go is comparable to the critic's value predictions.
    """

    def __init__(self, num_envs: int, max_episode_steps: int = 2048, gamma: float = 1.0) -> None:
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.gamma = float(gamma)
        self._steps: list[list[tuple]] = [[] for _ in range(num_envs)]
        self._overflowed = [False] * num_envs

    def record_step(
        self,
        obs: dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        teacher_forced: np.ndarray | None = None,
    ) -> None:
        if teacher_forced is None:
            teacher_forced = np.zeros(self.num_envs, dtype=bool)
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
                    float(rewards[i]),
                    bool(teacher_forced[i]),
                )
            )

    def finish_episode(
        self,
        env_idx: int,
        *,
        won: bool,
        stalled: bool,
        final_ante: int = 0,
        buffer: EpisodeReplayBuffer,
    ) -> None:
        steps = self._steps[env_idx]
        overflowed = self._overflowed[env_idx]
        self._steps[env_idx] = []
        self._overflowed[env_idx] = False
        # Always reset tracking after a completion or drop. Overflow and stall
        # episodes are dropped: a legal policy loss is only dropped when the
        # environment itself identifies an infrastructure/no-progress stall.
        if overflowed:
            buffer.overflow_episodes_dropped_total += 1
            return
        if stalled:
            buffer.stalled_episodes_dropped_total += 1
            return
        if not steps:
            return
        returns = np.empty(len(steps), dtype=np.float32)
        acc = 0.0
        total_reward = 0.0
        for t in range(len(steps) - 1, -1, -1):
            r = steps[t][6]
            total_reward += r
            acc = r + self.gamma * acc
            returns[t] = acc
        buffer.add_episode(
            {
                "tokens": np.stack([s[0] for s in steps]),
                "token_types": np.stack([s[1] for s in steps]),
                "scalars": np.stack([s[2] for s in steps]),
                "attention_mask": np.stack([s[3] for s in steps]),
                "action_mask_packed": np.stack([s[4] for s in steps]),
                "actions": np.asarray([s[5] for s in steps], dtype=np.int64),
                "returns": returns,
                "teacher_forced": np.asarray([s[7] for s in steps], dtype=np.int8),
                "won": bool(won),
                "final_ante": int(final_ante),
                "episode_length": len(steps),
                "total_reward": float(total_reward),
            }
        )
