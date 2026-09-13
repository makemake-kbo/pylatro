"""Persisted accelerator limits for bounded, resumable experiment jobs."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import signal
import time
from pathlib import Path

import torch

from .learning_bottleneck import atomic_json


@contextlib.contextmanager
def accelerator_job(root: Path, name: str, estimate_seconds: float):
    """Charge wall time, including failures; admit jobs only with a 20% margin."""
    root = Path(root)
    with (root / "accelerator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / "compute.json"
        ledger = json.loads(path.read_text())
        if any(j["status"] == "running" for j in ledger["jobs"]):
            raise RuntimeError("Inspect and account for the existing accelerator job before resuming")
        remaining = ledger["limit_seconds"] - sum(j["elapsed_seconds"] for j in ledger["jobs"])
        if estimate_seconds <= 0 or 1.2 * estimate_seconds + 5 > remaining:
            raise TimeoutError(f"{name} does not fit the remaining {remaining:.1f} accelerator seconds")
        job = dict(name=name, pid=os.getpid(), status="running", started_unix=time.time(), elapsed_seconds=0)
        ledger["jobs"].append(job)
        atomic_json(path, ledger)
        started = time.monotonic()

        def deadline(signum, frame):
            raise TimeoutError(f"Accelerator budget reached during {name}")

        previous = signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, remaining - 5)
        try:
            yield
            torch.mps.synchronize()
            job["status"] = "complete"
        except BaseException as exc:
            job.update(status="failed", error=repr(exc))
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            job["elapsed_seconds"] = time.monotonic() - started
            ledger["elapsed_seconds"] = sum(j["elapsed_seconds"] for j in ledger["jobs"])
            atomic_json(path, ledger)
