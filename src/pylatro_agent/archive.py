"""Bounded archives of reachable simulator states for return-and-explore training.

After a return the current policy generates an entirely new PPO continuation.
Snapshots also retain bounded action lineage for a separate winning-prefix
imitation objective; those historical actions never enter the PPO surrogate.
Buckets are sampled uniformly over Ante and phase.
"""

from __future__ import annotations

import hashlib
import io
import pickle
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

ARCHIVE_VERSION = 2
MAX_RETURN_PATH_ACTIONS = 2048


def observation_fingerprint(obs: dict[str, np.ndarray]) -> str:
    """Verify deterministic prefix reconstruction before any imitation loss."""
    digest = hashlib.sha256()
    for key, array in sorted(obs.items()):
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ArchiveConfig:
    return_probability: float = 0.5
    min_ante: int = 4
    max_ante: int = 8
    capacity_per_bucket: int = 8

    def __post_init__(self) -> None:
        if not 0 <= self.return_probability <= 1:
            raise ValueError("archive return_probability must be between 0 and 1")
        if not 1 <= self.min_ante <= self.max_ante <= 8:
            raise ValueError("archive requires 1 <= min_ante <= max_ante <= 8")
        if self.capacity_per_bucket < 1:
            raise ValueError("archive capacity_per_bucket must be positive")


def pack_snapshot(snapshot: dict[str, Any], data: Any) -> bytes:
    """Serialize an owned snapshot without duplicating immutable game data."""

    class Writer(pickle.Pickler):
        def persistent_id(self, obj):
            return "game_data" if obj is data else None

    stream = io.BytesIO()
    Writer(stream, protocol=pickle.HIGHEST_PROTOCOL).dump(snapshot)
    return stream.getvalue()


def unpack_snapshot(payload: bytes, data: Any) -> dict[str, Any]:
    """Read snapshots from our own archive/checkpoints (trusted pickle only)."""

    class Reader(pickle.Unpickler):
        def persistent_load(self, pid):
            if pid != "game_data":
                raise ValueError(f"Unknown archive resource: {pid!r}")
            return data

    return Reader(io.BytesIO(payload)).load()


class StateArchive:
    def __init__(self, config: ArchiveConfig, *, seed: int | None = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.buckets: dict[tuple[int, str], list[tuple[str, bytes]]] = {}
        self.seen: dict[tuple[int, str], int] = {}
        self.fresh_starts = 0
        self.returns = 0

    def add(self, *, ante: int, phase: str, identity: str, payload: bytes) -> bool:
        if not self.config.min_ante <= ante <= self.config.max_ante:
            return False
        key = (ante, phase)
        bucket = self.buckets.setdefault(key, [])
        # Repeated visits to the same seed/boundary do not fill the archive
        # with copies of one lucky opening. Keep the latest reachable variant.
        for i, (saved_identity, _) in enumerate(bucket):
            if saved_identity == identity:
                bucket[i] = (identity, payload)
                return True
        self.seen[key] = self.seen.get(key, 0) + 1
        if len(bucket) < self.config.capacity_per_bucket:
            bucket.append((identity, payload))
            return True
        index = int(self.rng.integers(self.seen[key]))
        if index < len(bucket):
            bucket[index] = (identity, payload)
            return True
        return False

    def choose(self, *, force_fresh: bool = False) -> bytes | None:
        keys = sorted(key for key, bucket in self.buckets.items() if bucket)
        if force_fresh or not keys or self.rng.random() >= self.config.return_probability:
            self.fresh_starts += 1
            return None
        antes = sorted({ante for ante, _ in keys})
        ante = antes[int(self.rng.integers(len(antes)))]
        keys = [key for key in keys if key[0] == ante]
        bucket = self.buckets[keys[int(self.rng.integers(len(keys)))]]
        self.returns += 1
        return bucket[int(self.rng.integers(len(bucket)))][1]

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": ARCHIVE_VERSION,
            "config": asdict(self.config),
            "buckets": self.buckets,
            "seen": self.seen,
            "rng": self.rng.bit_generator.state,
            "fresh_starts": self.fresh_starts,
            "returns": self.returns,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != ARCHIVE_VERSION or state.get("config") != asdict(self.config):
            raise ValueError("Archive version/config mismatch; use the original archive settings on resume")
        self.buckets = state["buckets"]
        self.seen = state["seen"]
        if any(len(bucket) > self.config.capacity_per_bucket for bucket in self.buckets.values()):
            raise ValueError("Archive bucket exceeds configured capacity")
        self.rng.bit_generator.state = state["rng"]
        self.fresh_starts = int(state["fresh_starts"])
        self.returns = int(state["returns"])

    def metrics(self) -> dict[str, float]:
        result = {
            "states": float(sum(map(len, self.buckets.values()))),
            "returns": float(self.returns),
            "fresh_starts": float(self.fresh_starts),
        }
        for ante in range(self.config.min_ante, self.config.max_ante + 1):
            result[f"states_ante{ante}"] = float(
                sum(len(bucket) for (a, _), bucket in self.buckets.items() if a == ante)
            )
        return result
