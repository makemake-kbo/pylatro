"""Frozen protocol primitives for the public search-guided PPO study."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tarfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

NAMESPACE = "SEARCH-GUIDED-PPO-20261004-V1"
PANEL_SIZES = dict(
    pilot=64,
    timing=8,
    exploratory_training=1280,
    confirmation_training=1280,
    diagnostic=64,
    development=96,
    confirmation=256,
)
TRAINING_RNG = dict(pilot=613, exploratory=1729, confirmation=2718)


def encode(value):
    def native(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)

    return json.dumps(value, default=native, sort_keys=True, allow_nan=False, indent=2) + "\n"


def fingerprint(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as stream:
        stream.write(encode(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    sync_directory(path.parent)


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != json.loads(encode(value)):
            raise ValueError(f"Frozen artifact changed: {path}")
    else:
        atomic(path, value)


def save_torch(path, payload):
    import torch

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    sync_directory(path.parent)


@contextmanager
def exclusive(out):
    import fcntl

    out.mkdir(parents=True, exist_ok=True)
    with (out / "owner.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def start_clock(out, *, now=None):
    path = out / "clock.json"
    if path.exists():
        clock = read(path)
        if clock["deadline"] - clock["started"] != 86400:
            raise ValueError("Invalid immutable deadline")
        immutable(out / "execution.json", dict(clock_sha256=fingerprint(clock)))
        return clock
    started = time.time() if now is None else now
    clock = dict(started=started, deadline=started + 86400, seconds=86400)
    immutable(path, clock)
    immutable(out / "execution.json", dict(clock_sha256=fingerprint(clock)))
    return clock


def check_deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Original 24-hour wall deadline reached")


def metadata_name(path):
    path = Path(path)
    name = path.name.lower()
    return path.suffix in {".json", ".jsonl", ".yaml", ".yml", ".md"} and (
        any(
            w in name
            for w in (
                "seed",
                "manifest",
                "protocol",
                "config",
                "generation",
                "reservation",
                "registry",
                "exclusion",
                "metadata",
                "panel",
            )
        )
        and not any(w in name for w in ("outcome", "result", "trajectory", "label", "games"))
    )


def metadata_integers(blob):
    # A conservative superset: every integer token in metadata, including
    # numeric strings and unused reservations. Never inspect outcome files.
    return {int(m) for m in re.findall(rb"(?<![A-Za-z0-9])\d+(?![A-Za-z0-9])", blob)}


def audit_metadata(repo, out):
    records, exclusions = [], set()

    def consume(name, blob):
        if NAMESPACE.encode() in blob:
            raise ValueError(f"Namespace already reserved: {name}")
        values = metadata_integers(blob)
        exclusions.update(values)
        records.append(
            dict(path=name, sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob), integer_tokens=len(values))
        )

    for folder in ("eval", "datasets", "incumbents", "runs", "docs", "cache", "data", "archives"):
        for path in sorted((repo / folder).rglob("*")):
            if path.is_file() and out not in path.parents and metadata_name(path):
                consume(str(path.relative_to(repo)), path.read_bytes())
    archives = []
    for path in sorted((repo / "archives").rglob("*.tar.gz")):
        count = 0
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                if member.isfile() and metadata_name(member.name):
                    consume(f"{path.relative_to(repo)}::{member.name}", archive.extractfile(member).read())
                    count += 1
        archives.append(dict(path=str(path.relative_to(repo)), sha256=digest(path), metadata_files=count))
    return dict(
        metadata=records,
        archives=archives,
        exclusion_count=len(exclusions),
        rule="Exclude every integer token in retained metadata, including invalidated and unused reservations; "
        "stream archived metadata only; never open reserved outcomes.",
    ), exclusions


def seed_panels(exclusions=()):
    used = set(exclusions)
    panels = {}
    # Cython's executor accepts signed 32-bit integers. Large fresh values
    # avoid broad historical low-integer ranges; every candidate is audited.
    for name, count in PANEL_SIZES.items():
        panel, cursor = [], 0
        while len(panel) < count:
            hashed = hashlib.sha256(f"{NAMESPACE}:{name}:{cursor}".encode()).digest()
            seed = 1_000_000_000 + int.from_bytes(hashed[:8], "big") % 1_147_483_647
            cursor += 1
            if seed not in used:
                panel.append(seed)
                used.add(seed)
        panels[name] = panel
    assert_separation(panels, exclusions)
    return panels


def assert_separation(panels, exclusions=()):
    flat = [seed for panel in panels.values() for seed in panel]
    if (
        any(type(s) is not int or not 0 <= s < 2**31 for s in flat)
        or len(set(flat)) != len(flat)
        or set(flat) & set(exclusions)
    ):
        raise ValueError("Game seed reservation collision or invalid integer")


def select_duration(timing, remaining):
    """Only runtime and coverage may enter the duration decision; never wins.

    Evaluation timing is measured wall time per game at the same four-worker
    limit, including startup (no assumed parallel speedup). Include four training arms,
    five greedy+sampled confirmation models, base+two assisted models and
    two early-only models, plus 64 fixed-state diagnostic games and labels.
    """
    keys = {
        "ordinary_update",
        "guided_update",
        "greedy_game",
        "sampled_game",
        "assisted_game",
        "label_state",
        "diagnostic_inference_state",
    }
    if set(timing) != keys or any(not math.isfinite(v) or v < 0 for v in timing.values()):
        raise ValueError("Duration selection requires runtimes only")
    evaluation = (
        3 * 96 * (timing["greedy_game"] + timing["sampled_game"] + timing["assisted_game"])
        + 5 * 256 * (timing["greedy_game"] + timing["sampled_game"])
        + 3 * 256 * timing["assisted_game"]
        + 2 * 256 * timing["greedy_game"]
    )
    diagnostics = (
        64 * timing["greedy_game"] + 256 * timing["label_state"] + 9 * 256 * timing["diagnostic_inference_state"]
    )
    projections = {
        n: 1.25 * (2 * n * (timing["ordinary_update"] + timing["guided_update"]) + evaluation + diagnostics)
        for n in (20, 12, 8)
    }
    chosen = next((n for n in (20, 12, 8) if projections[n] <= remaining), None)
    return dict(
        updates=chosen, projections_seconds=projections, remaining_seconds=remaining, margin=0.25, timing=timing
    )


def paired_claim(candidates, controls):
    candidates, controls = np.asarray(candidates, float), np.asarray(controls, float)
    if candidates.ndim != 2 or candidates.shape != controls.shape or candidates.shape[0] != 2:
        raise ValueError("Require matched game results for two trained runs")
    differences = candidates - controls
    rng = np.random.default_rng(1729)
    # All model outcomes of each game travel together in every bootstrap draw.
    indices = rng.integers(candidates.shape[1], size=(10000, candidates.shape[1]))
    bootstrap = differences.mean(axis=0)[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    gains = differences.mean(axis=1)
    gain = float(gains.mean())
    return dict(
        mean_gain=gain,
        per_run_gains=gains.tolist(),
        ci95=[float(low), float(high)],
        qualifies=bool(gain >= 0.05 and low > 0 and np.all(gains > 0)),
        bootstrap_draws=10000,
        bootstrap_seed=1729,
        uncertainty="Paired across game seeds; conditional on these two trained runs.",
    )
