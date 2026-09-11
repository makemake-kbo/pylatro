"""Bounded reconstruction/replay of the early paths behind archive wins.

Only verified winning prefixes enter this actor-only BC store. They never enter
the PPO rollout, GAE, value regression, terminal replay, or validation panels.
The worker sends a compact seed/action recipe; rebuilding is CPU-step budgeted.
"""

from __future__ import annotations

import json
import pickle
import zlib
from collections import deque

import numpy as np
import torch

from ..archive import MAX_RETURN_PATH_ACTIONS, observation_fingerprint
from ..constants import NUM_ACTIONS, TOKENIZER_VERSION
from ..env import BalatroEnv
from ..reward import RewardConfig
from .ppo_observations import _obs_dicts_to_batch


def _encode_observation(obs: dict) -> bytes:
    compact = {key: value for key, value in obs.items() if key != "action_mask"}
    compact["action_mask_packed"] = np.packbits(obs["action_mask"] > 0)
    return zlib.compress(pickle.dumps(compact, protocol=pickle.HIGHEST_PROTOCOL), level=1)


def _decode_observation(payload: bytes) -> dict:
    # Only our own in-process/checkpoint-generated observations are accepted.
    obs = pickle.loads(zlib.decompress(payload))
    obs["action_mask"] = np.unpackbits(obs.pop("action_mask_packed"), count=NUM_ACTIONS).astype(np.int8)
    return obs


class ReturnPathReplay:
    def __init__(self, data, vocab, *, capacity: int = 32, seed: int = 0, excluded_seeds: tuple[int, ...] = ()):
        if capacity < 1:
            raise ValueError("Return-path capacity must be positive")
        self.data, self.vocab, self.capacity = data, vocab, capacity
        self.excluded_seeds = frozenset(str(value) for value in excluded_seeds)
        self.rng = np.random.default_rng(seed)
        self.pending: deque[dict] = deque()
        self.ready: list[dict] = []
        self.active: dict | None = None
        self.env: BalatroEnv | None = None
        self.obs: dict | None = None
        self.encoded: list[bytes] = []
        self.rebuilt_steps = 0
        self.rejected = 0
        self.completed = 0

    @staticmethod
    def _key(recipe: dict) -> tuple:
        return recipe["seed"], recipe["boundary_fingerprint"]

    def enqueue(self, payload: bytes) -> bool:
        """Queue a genuine archive win; reject stale schemas/evaluation seeds."""
        try:
            recipe = json.loads(payload)
            if (
                recipe["tokenizer_version"] != TOKENIZER_VERSION
                or recipe.get("won") is not True
                or recipe["win_ante"] != 8
                or str(recipe["seed"]) in self.excluded_seeds
                or not 0 < len(recipe["actions"]) <= MAX_RETURN_PATH_ACTIONS
                or any(not isinstance(action, int) or not 0 <= action < NUM_ACTIONS for action in recipe["actions"])
                or len(recipe["boundary_fingerprint"]) != 64
            ):
                raise ValueError("Invalid return-path recipe")
            int(recipe["seed"])
        except (ValueError, KeyError, TypeError):
            self.rejected += 1
            return False
        known = [row["recipe"] for row in self.ready] + list(self.pending)
        if self.active is not None:
            known.append(self.active)
        if any(self._key(other) == self._key(recipe) for other in known):
            return False
        if len(self.pending) >= self.capacity:
            self.pending.popleft()
            self.rejected += 1
        self.pending.append(recipe)
        return True

    def _finish(self, *, valid: bool) -> None:
        if valid:
            self.ready.append({"recipe": self.active, "observations": self.encoded})
            self.ready = self.ready[-self.capacity :]
            self.completed += 1
        else:
            self.rejected += 1
        if self.env is not None:
            self.env.close()
        self.active, self.env, self.obs = None, None, None
        self.encoded = []

    def advance(self, step_budget: int) -> None:
        """Reconstruct at most step_budget simulator actions, across updates."""
        for _ in range(step_budget):
            if self.active is None:
                if not self.pending:
                    break
                self.active = self.pending.popleft()
                self.env = BalatroEnv(
                    data=self.data,
                    vocab=self.vocab,
                    seed=int(self.active["seed"]),
                    stake=self.active["stake"],
                    deck_key=self.active["deck_key"],
                    win_ante=8,
                    enable_teacher=False,
                    reward_config=RewardConfig(objective="milestone"),
                )
                self.obs, _ = self.env.reset(seed=int(self.active["seed"]))
            action = self.active["actions"][len(self.encoded)]
            if not self.obs["action_mask"][action]:
                self._finish(valid=False)
                continue
            self.encoded.append(_encode_observation(self.obs))
            self.obs, _, terminated, truncated, info = self.env.step(action)
            self.rebuilt_steps += 1
            if terminated or truncated or info.get("error"):
                self._finish(valid=False)
            elif len(self.encoded) == len(self.active["actions"]):
                self._finish(valid=observation_fingerprint(self.obs) == self.active["boundary_fingerprint"])

    def sample(self, batch_size: int, device: torch.device, *, samples_per_episode: int = 8) -> dict | None:
        if not self.ready:
            return None
        picks = []
        quota = min(samples_per_episode, max(1, (batch_size + len(self.ready) - 1) // len(self.ready)))
        for index in self.rng.permutation(len(self.ready)):
            episode = self.ready[index]
            for row in self.rng.choice(
                len(episode["observations"]), size=min(quota, len(episode["observations"])), replace=False
            ):
                picks.append((episode, int(row)))
            if len(picks) >= batch_size:
                break
        picks = picks[:batch_size]
        batch = _obs_dicts_to_batch([_decode_observation(ep["observations"][row]) for ep, row in picks], device)
        batch["actions"] = torch.tensor([ep["recipe"]["actions"][row] for ep, row in picks], device=device)
        return batch

    def state_dict(self) -> dict:
        # An in-flight reconstruction restarts from its recipe after resume;
        # already verified BC data and sampler RNG are preserved exactly.
        pending = ([self.active] if self.active is not None else []) + list(self.pending)
        return {
            "tokenizer_version": TOKENIZER_VERSION,
            "capacity": self.capacity,
            "ready": self.ready,
            "pending": pending,
            "rng": self.rng.bit_generator.state,
            "rebuilt_steps": self.rebuilt_steps,
            "rejected": self.rejected,
            "completed": self.completed,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["tokenizer_version"] != TOKENIZER_VERSION or state["capacity"] != self.capacity:
            raise ValueError("Return-path replay schema/capacity mismatch")
        if len(state["ready"]) > self.capacity or len(state["pending"]) > self.capacity + 1:
            raise ValueError("Return-path replay exceeds capacity")
        recipes = [row["recipe"] for row in state["ready"]] + state["pending"]
        if any(str(recipe["seed"]) in self.excluded_seeds for recipe in recipes):
            raise ValueError("Return-path replay contains evaluation seeds")
        self.close()
        self.active, self.env, self.obs = None, None, None
        self.encoded = []
        self.ready, self.pending = state["ready"], deque(state["pending"])
        self.rng.bit_generator.state = state["rng"]
        self.rebuilt_steps, self.rejected, self.completed = (
            state[name] for name in ("rebuilt_steps", "rejected", "completed")
        )

    def metrics(self) -> dict[str, float]:
        return {
            "verified_prefixes": float(len(self.ready)),
            "pending_prefixes": float(len(self.pending)),
            "rebuilding": float(self.active is not None),
            "rejected": float(self.rejected),
            "completed_total": float(self.completed),
            "rebuild_steps_total": float(self.rebuilt_steps),
            "stored_bytes": float(sum(len(obs) for ep in self.ready for obs in ep["observations"])),
        }

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
