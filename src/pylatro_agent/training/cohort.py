"""Frozen-policy complete cohorts and compact, lazy PPO training batches.

The outcome label and GAE boundary are deliberately independent. References
and the limits of the experiment are in docs/transfer/complete_cohort.md.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from ..constants import CURRENT_ANTE_SCALAR_INDEX, NUM_ACTIONS
from ..survival import terminal_outcome_class
from .ppo_observations import _obs_dicts_to_batch
from .ppo_policy import _grammar_distribution, _policy_temperature_for_scalars

FLOAT_OBSERVATIONS = {"scalars", "history_event_features", "history_omitted"}


def policy_digest(model):
    digest = hashlib.sha256()
    if getattr(model, "config", None) is not None:
        digest.update(json.dumps(asdict(model.config), sort_keys=True, allow_nan=False).encode())
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sampling_configuration(config):
    return {
        key: getattr(config, key)
        for key in (
            "hand_ar_mixture_eps",
            "rollout_temperature",
            "danger_rollout_temperature",
            "danger_death_probability_threshold",
        )
    }


@contextmanager
def frozen_collection_policy(model):
    """PPO Algorithm 1: collect under pi_old, then optimize; no mid-game lag."""
    parameters = [(p, p.requires_grad) for p in model.parameters()]
    modes = [(module, module.training) for module in model.modules()]
    before = policy_digest(model)
    model.eval()
    for parameter, _ in parameters:
        parameter.requires_grad_(False)
    try:
        with torch.no_grad():
            yield before
    finally:
        for parameter, enabled in parameters:
            parameter.requires_grad_(enabled)
        for module, mode in modes:
            module.training = mode
        if policy_digest(model) != before:
            raise RuntimeError("Policy parameters, buffers, or configuration changed during complete-cohort collection")


class CohortFailure(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class ObservationStore:
    """Copy observations immediately, compact on CPU, concatenate in chunks."""

    def __init__(self, chunk_size=256):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self.pending = []
        self.chunks = []
        self.ends = []
        self.size = 0
        self.schema = None

    def append(self, obs):
        schema = {key: (np.asarray(value).shape, np.asarray(value).dtype.str) for key, value in obs.items()}
        if self.schema is not None and schema != self.schema:
            raise ValueError("Observation schema changed within an episode")
        if not np.isin(obs["action_mask"], [0, 1]).all():
            raise ValueError("Action masks must be binary before bit packing")
        copied = {}
        for key, value in obs.items():
            array = np.asarray(value)
            if key == "action_mask":
                copied[key] = np.packbits(array)
            else:
                # Preserve the environment's dtype and every history field.
                copied[key] = array.copy()
        self.schema = schema
        self.pending.append(copied)
        self.size += 1
        if len(self.pending) == self.chunk_size:
            self.flush()

    def flush(self):
        if self.pending:
            self.chunks.append({key: np.stack([row[key] for row in self.pending]) for key in self.pending[0]})
            self.ends.append(self.size)
            self.pending.clear()

    def take(self, indices):
        self.flush()
        indices = np.asarray(indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise IndexError("Observation row outside written range")
        selected = np.searchsorted(self.ends, indices, side="right")
        result = {}
        for chunk_id in np.unique(selected):
            positions = np.flatnonzero(selected == chunk_id)
            start = 0 if chunk_id == 0 else self.ends[chunk_id - 1]
            rows = indices[positions] - start
            for key, array in self.chunks[chunk_id].items():
                if key not in result:
                    result[key] = np.empty((len(indices), *array.shape[1:]), dtype=array.dtype)
                result[key][positions] = array[rows]
        if "action_mask" in result:
            result["action_mask"] = np.unpackbits(result["action_mask"], axis=-1, count=NUM_ACTIONS).astype(np.int8)
        return result


@dataclass
class CohortEpisode:
    seed: int
    slot: int
    policy_version: int
    policy_sha256: str
    observations: ObservationStore = field(default_factory=ObservationStore)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    values: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    slot_steps: list = field(default_factory=list)
    outcome: dict | None = None
    final_observation: dict | None = None
    reward_info: list = field(default_factory=list)
    sampling_configuration: dict = field(default_factory=dict)


def completion_mask(slot_steps, fragment_length):
    """Rubin's selection concern applied here: availability depends on the future end."""
    steps = np.asarray(slot_steps, dtype=np.int64)
    if len(steps) == 0 or fragment_length < 1 or np.any(np.diff(steps) != 1):
        raise ValueError("Completion mask requires contiguous nonempty decision counters")
    return (steps // fragment_length == steps[-1] // fragment_length).astype(np.float32)


def fragment_gae(rewards, values, slot_steps, *, gamma, gae_lambda, fragment_length):
    """PPO Eqs. 10-12 / Pardo: retain bootstrap, end trace at artificial cuts.

    Called only for a complete normally terminated episode. Collection has
    already evaluated the next state's value, including at fragment boundaries.
    """
    rewards, values = np.asarray(rewards, np.float32), np.asarray(values, np.float32)
    counters = np.asarray(slot_steps)
    if len(rewards) == 0 or len(rewards) != len(values) or len(counters) != len(values):
        raise ValueError("GAE requires aligned nonempty complete episode arrays")
    advantages = np.zeros_like(values)
    trace = 0.0
    for t in reversed(range(len(values))):
        terminal = t == len(values) - 1
        next_value = 0.0 if terminal else values[t + 1]
        boundary = (int(counters[t]) + 1) % fragment_length == 0
        delta = rewards[t] + gamma * next_value - values[t]
        trace = delta + (0.0 if terminal or boundary else gamma * gae_lambda * trace)
        advantages[t] = trace
    return advantages, advantages + values


def collect_complete_cohort(model, envs, config, seeds, slot_counters, *, policy_version, chunk_size=256):
    """Finish a predeclared cohort; no replacement seeds or optimizer calls."""
    seeds = list(seeds)
    if len(seeds) != config.cohort_games or len(set(seeds)) != len(seeds):
        raise ValueError("Cohort must contain exactly its declared distinct seeds")
    counters = list(slot_counters)  # Commit only after a successful update.
    if len(counters) != config.num_envs:
        raise ValueError("Wrong number of logical slot counters")
    assignments = [seeds[slot :: config.num_envs] for slot in range(config.num_envs)]
    offsets = [0] * config.num_envs
    episodes = {}
    active = {}
    observations = {}
    device = next(model.parameters()).device
    settings = sampling_configuration(config)
    if model.config.hand_ar_mixture_eps != config.hand_ar_mixture_eps:
        raise ValueError("Model and collector hand-policy mixtures differ")
    diagnostics = {"seeds": seeds, "policy_version": policy_version, "completed_seeds": [], "steps": 0}

    def start(slot, digest):
        seed = assignments[slot][offsets[slot]]
        offsets[slot] += 1
        obs, _phase = envs.reset(seed)
        active[slot] = seed
        observations[slot] = obs
        episodes[seed] = CohortEpisode(seed, slot, policy_version, digest, ObservationStore(chunk_size))
        episodes[seed].sampling_configuration = dict(settings)

    try:
        with frozen_collection_policy(model) as digest:
            diagnostics["policy_sha256"] = digest
            for slot in range(config.num_envs):
                start(slot, digest)
            while active:
                slots = sorted(active)
                batch = _obs_dicts_to_batch([observations[slot] for slot in slots], device)
                dist, predictions = _grammar_distribution(
                    model,
                    batch,
                    temperature=_policy_temperature_for_scalars(batch["scalars"], config),
                )
                actions = dist.sample()
                logs = dist.log_prob(actions)
                values = predictions["expected_return"]
                if not all(torch.isfinite(t).all() for t in (logs, *predictions.values())):
                    raise RuntimeError("Non-finite policy or critic output during collection")
                actions_np = actions.cpu().numpy()
                logs_np, values_np = logs.cpu().numpy(), values.cpu().numpy()
                requests = []
                for i, slot in enumerate(slots):
                    seed = active[slot]
                    action = int(actions_np[i])
                    obs = observations[slot]
                    if not 0 <= action < NUM_ACTIONS or not obs["action_mask"][action]:
                        raise RuntimeError(f"Illegal action {action}, seed {seed}, step {len(episodes[seed].actions)}")
                    episode = episodes[seed]
                    episode.observations.append(obs)
                    episode.actions.append(action)
                    episode.values.append(float(values_np[i]))
                    episode.log_probs.append(float(logs_np[i]))
                    episode.slot_steps.append(counters[slot])
                    counters[slot] += 1
                    requests.append((seed, action))
                results = envs.step_training(requests)
                if set(results) != {seed for seed, _ in requests}:
                    raise RuntimeError("Missing or unexpected worker result")
                for slot in slots:
                    seed = active[slot]
                    obs, reward, terminated, truncated, info, _seconds = results[seed]
                    episode = episodes[seed]
                    episode.rewards.append(float(reward))
                    episode.reward_info.append({k: v for k, v in info.items() if k.startswith("reward_")})
                    diagnostics["steps"] += 1
                    if not np.isfinite(reward):
                        raise RuntimeError(f"Non-finite reward for seed {seed}")
                    if truncated or info.get("stalled"):
                        raise RuntimeError(f"Censored or stalled game {seed}; the entire cohort is incomplete")
                    expected_reward = 10.0 if terminated and info["won"] else 0.0
                    if not np.isclose(reward, expected_reward, atol=1e-6):
                        raise RuntimeError(f"Unexpected non-terminal-only reward for seed {seed}")
                    if terminated:
                        episode.outcome = {
                            "seed": seed,
                            "won": bool(info["won"]),
                            "final_ante": int(info["ante"]),
                            "terminated": True,
                            "truncated": False,
                            "stalled": False,
                            "steps": len(episode.actions),
                            "cleared_antes": info.get("cleared_antes", []),
                        }
                        if not np.isclose(sum(episode.rewards), 10.0 * episode.outcome["won"], atol=1e-6):
                            raise RuntimeError(f"Unexpected non-terminal-only reward for seed {seed}")
                        episode.final_observation = {k: np.array(v, copy=True) for k, v in obs.items()}
                        episode.observations.flush()
                        diagnostics["completed_seeds"].append(seed)
                        del active[slot]
                        if offsets[slot] < len(assignments[slot]):
                            start(slot, digest)
                    else:
                        if len(episode.actions) >= config.cohort_max_episode_steps:
                            raise RuntimeError(f"Episode step limit reached for seed {seed}; no replacement game")
                        observations[slot] = obs
            if set(diagnostics["completed_seeds"]) != set(seeds):
                raise RuntimeError("Incomplete cohort")
            if sampling_configuration(config) != settings:
                raise RuntimeError("Sampling configuration changed during collection")
        return [episodes[seed] for seed in seeds], counters
    except Exception as exc:
        diagnostics["unfinished"] = {
            str(seed): len(episode.actions) for seed, episode in episodes.items() if episode.outcome is None
        }
        raise CohortFailure(str(exc), diagnostics) from exc


class CohortBuffer:
    """All transitions remain eligible; PPO draws a fixed physical exposure."""

    def __init__(self, episodes, config, *, draw_indices=None):
        if not episodes or any(e.outcome is None or not e.actions for e in episodes):
            raise ValueError("PPO requires a nonempty completely observed cohort")
        if len({e.seed for e in episodes}) != len(episodes):
            raise ValueError("Duplicate game seed in cohort")
        if len({(e.policy_sha256, e.policy_version) for e in episodes}) != 1:
            raise ValueError("Mixed collection policies in cohort")
        if any(e.sampling_configuration != sampling_configuration(config) for e in episodes):
            raise ValueError("Cohort sampling configuration differs from optimization")
        self.episodes = episodes
        arrays = {
            k: []
            for k in (
                "actions",
                "old_log_probs",
                "values",
                "returns",
                "advantages",
                "terminal_outcome_target",
                "completion_mask",
                "seed",
                "step",
                "ante",
            )
        }
        lengths = []
        for episode in episodes:
            episode.observations.flush()
            length = len(episode.actions)
            if episode.observations.size != length or any(
                len(v) != length
                for v in (
                    episode.log_probs,
                    episode.rewards,
                    episode.values,
                    episode.slot_steps,
                )
            ):
                raise ValueError("Missing or misaligned episode decisions")
            outcome = episode.outcome
            if not outcome["terminated"] or outcome.get("truncated") or outcome.get("stalled"):
                raise ValueError("Censored outcomes cannot train complete-cohort PPO")
            advantages, returns = fragment_gae(
                episode.rewards,
                episode.values,
                episode.slot_steps,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                fragment_length=config.rollout_length,
            )
            values = {
                "actions": np.asarray(episode.actions, np.int64),
                "old_log_probs": np.asarray(episode.log_probs, np.float32),
                "values": np.asarray(episode.values, np.float32),
                "returns": returns,
                "advantages": advantages,
                "terminal_outcome_target": np.full(
                    length, terminal_outcome_class(won=outcome["won"], final_ante=outcome["final_ante"]), np.int64
                ),
                "completion_mask": completion_mask(episode.slot_steps, config.rollout_length),
                "seed": np.full(length, episode.seed, np.int64),
                "step": np.arange(length),
                "ante": np.concatenate(
                    [chunk["scalars"][:, CURRENT_ANTE_SCALAR_INDEX] for chunk in episode.observations.chunks]
                ),
            }
            for key in arrays:
                arrays[key].append(values[key])
            lengths.append(length)
        self.arrays = {key: np.concatenate(parts) for key, parts in arrays.items()}
        self.size = sum(lengths)
        self.ends = np.cumsum(lengths)
        advantages = self.arrays["advantages"]
        self.arrays["advantages"] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if config.advantage_clip_sigma > 0:
            np.clip(
                self.arrays["advantages"],
                -config.advantage_clip_sigma,
                config.advantage_clip_sigma,
                out=self.arrays["advantages"],
            )
        self.arrays["terminal_outcome_mask"] = (
            np.ones(self.size, np.float32)
            if config.outcome_label_mode == "complete_episode"
            else self.arrays["completion_mask"].copy()
        )
        self.draw_indices = (
            np.random.randint(self.size, size=config.cohort_training_rows)
            if draw_indices is None
            else np.asarray(draw_indices, np.int64).copy()
        )
        if (
            len(self.draw_indices) != config.cohort_training_rows
            or np.any(self.draw_indices < 0)
            or np.any(self.draw_indices >= self.size)
        ):
            raise ValueError("Invalid cohort training row draw")

    @property
    def _flat_returns(self):
        return self.arrays["returns"][self.draw_indices]

    def get_batches(self, batch_size, device, pin_memory=False, micro_batch_size=None):
        # The permutation is planned eagerly on CPU so KL diagnostics can save
        # and restore the NumPy generator exactly as in the legacy buffer.
        order = self.draw_indices[np.random.permutation(len(self.draw_indices))]
        return CohortBatches(self, order, batch_size, device, pin_memory, micro_batch_size)

    def materialize(self, indices, device, pin_memory=False):
        indices = np.asarray(indices, np.int64)
        owners = np.searchsorted(self.ends, indices, side="right")
        obs = {}
        for owner in np.unique(owners):
            positions = np.flatnonzero(owners == owner)
            start = 0 if owner == 0 else self.ends[owner - 1]
            data = self.episodes[owner].observations.take(indices[positions] - start)
            for key, array in data.items():
                if key not in obs:
                    obs[key] = np.empty((len(indices), *array.shape[1:]), dtype=array.dtype)
                obs[key][positions] = array
        result = {}
        for key, array in obs.items():
            dtype = torch.float32 if key in FLOAT_OBSERVATIONS or key == "action_mask" else torch.long
            tensor = torch.as_tensor(array).to(dtype=dtype)
            if pin_memory:
                tensor = tensor.pin_memory()
            result[key] = tensor.to(device, non_blocking=pin_memory)
        for key in (
            "actions",
            "old_log_probs",
            "returns",
            "advantages",
            "terminal_outcome_target",
            "terminal_outcome_mask",
        ):
            result[key] = torch.as_tensor(self.arrays[key][indices], device=device)
        return result

    def diagnostics(self):
        mask = self.arrays["terminal_outcome_mask"]
        starts = self.arrays["step"] == 0
        wins = self.arrays["terminal_outcome_target"] == 8
        coverage = {}
        groups = {
            "all": np.ones(self.size, bool),
            "start": starts,
            "win": wins,
            "loss": ~wins,
            "winning_start": starts & wins,
            "losing_start": starts & ~wins,
        }
        groups.update({f"ante_{a}": self.arrays["ante"] == a for a in range(1, 9)})
        groups.update(
            {
                f"position_{lo}_{hi}": (self.arrays["step"] >= lo) & (self.arrays["step"] < hi)
                for lo, hi in ((0, 64), (64, 128), (128, 256), (256, 2048))
            }
        )
        for key, group in groups.items():
            coverage[key] = {"rows": int(group.sum()), "labels": int(mask[group].sum())}
        return {
            "collected_transitions": self.size,
            "games": len(self.episodes),
            "label_coverage": coverage,
            "drawn_rows": len(self.draw_indices),
            "unique_drawn_rows": len(set(self.draw_indices)),
            "duplicate_draw_fraction": 1 - len(set(self.draw_indices)) / len(self.draw_indices),
            "draw_sha256": hashlib.sha256(self.draw_indices.tobytes()).hexdigest(),
            "drawn_labels": int(mask[self.draw_indices].sum()),
        }


class CohortBatches(Sequence):
    """CPU batch plan with on-demand tensor creation, including mask metadata."""

    def __init__(self, buffer, order, logical_size, device, pin_memory, physical_size):
        self.buffer, self.order = buffer, order
        self.device, self.pin_memory = device, pin_memory
        physical_size = min(logical_size, physical_size or logical_size)
        self.plan = []
        for lo in range(0, len(order), logical_size):
            hi = min(lo + logical_size, len(order))
            for start in range(lo, hi, physical_size):
                end = min(start + physical_size, hi)
                self.plan.append((start, end, start == lo, end == hi, (end - start) / (hi - lo)))

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        start, end, first, last, weight = self.plan[index]
        batch = self.buffer.materialize(self.order[start:end], self.device, self.pin_memory)
        batch.update(
            {
                "_logical_group_start": torch.tensor(first),
                "_logical_group_end": torch.tensor(last),
                "_loss_weight": torch.tensor(weight, dtype=torch.float32, device=self.device),
            }
        )
        return batch

    def logical_mask_weights(self, key):
        masks = self.buffer.arrays[key]
        counts = [float(masks[self.order[lo:hi]].sum()) for lo, hi, *_ in self.plan]
        weights = [0.0] * len(counts)
        start = 0
        for i, plan in enumerate(self.plan):
            if plan[3]:
                total = sum(counts[start : i + 1])
                if total:
                    weights[start : i + 1] = [count / total for count in counts[start : i + 1]]
                start = i + 1
        return weights, counts
