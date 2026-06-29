"""Checkpoint save/load with tokenizer-version guard.

Legacy checkpoints were raw `state_dict()` tensors. New checkpoints are a
dict with a `tokenizer_version` field plus the state dict, so that loading
an old model against a newer observation format fails loudly instead of
silently mis-aligning.

Full PPO checkpoints additionally carry optimizer state, update counters,
entropy-controller state, RNG states, and config snapshots so that a run can
be *resumed* (not just re-initialized). They are marked with
``checkpoint_format == "ppo_full"``.
"""

from __future__ import annotations

import dataclasses
import logging
import pickle
import random
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from .constants import TOKENIZER_VERSION

logger = logging.getLogger(__name__)

# Marker distinguishing a full PPO resume checkpoint from a weights-only one.
PPO_CHECKPOINT_FORMAT = "ppo_full"


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    payload = {
        "tokenizer_version": TOKENIZER_VERSION,
        "state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint_payload(
    path: str | Path,
    device: torch.device | str,
) -> dict[str, Any]:
    """Load a checkpoint and verify its tokenizer_version matches.

    Accepts both the new dict format and legacy raw state_dicts (warning
    logged for the latter — no version can be verified).

    Loads with ``weights_only=True`` first (the safe default for arbitrary
    checkpoints). Full PPO resume checkpoints carry optimizer/NumPy RNG state
    that is not weights-only-loadable, so on a weights-only failure we retry
    with ``weights_only=False``. This is safe because such files are only ever
    written by our own training code via :func:`save_ppo_checkpoint`.
    """
    try:
        blob = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        # PyTorch raises UnpicklingError / pickle errors for non-weights-only
        # content (NumPy RNG state, optimizer state). Retry without the guard.
        if "weights_only" in str(exc) or isinstance(exc, (pickle.UnpicklingError,)):
            blob = torch.load(path, map_location=device, weights_only=False)
        else:
            raise
    if isinstance(blob, dict) and "state_dict" in blob:
        saved_version = blob.get("tokenizer_version")
        if saved_version != TOKENIZER_VERSION:
            raise RuntimeError(
                f"Checkpoint {path} was saved with tokenizer_version="
                f"{saved_version!r}, but current TOKENIZER_VERSION="
                f"{TOKENIZER_VERSION}. Observation format has changed; "
                "retrain or pin the tokenizer version."
            )
        return blob
    logger.warning(
        "Checkpoint %s has no tokenizer_version stamp (legacy format); "
        "loading without version check. Re-save to lock in the current version.",
        path,
    )
    return {"state_dict": blob, "tokenizer_version": None}


def capture_rng_states() -> dict[str, Any]:
    """Snapshot Python, NumPy, and Torch RNG states for deterministic resume."""
    states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_states(states: dict[str, Any]) -> None:
    """Restore RNG states captured by :func:`capture_rng_states`.

    Missing keys are skipped so a checkpoint saved on CPU can resume on a
    CUDA host (and vice-versa) without error.
    """
    if not states:
        return
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch_cpu" in states:
        torch.set_rng_state(states["torch_cpu"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def save_ppo_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    optimizer: torch.optim.Optimizer,
    update_count: int,
    total_steps: int,
    planned_updates: int,
    entropy_coeff: float,
    entropy_signal_ema: float | None,
    lr: float,
    log_alpha: torch.Tensor | None = None,
    alpha_optimizer: torch.optim.Optimizer | None = None,
    return_rms: Any = None,
    agent_config: Any = None,
    ppo_config_fields: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist a *full* PPO checkpoint that supports strict ``--resume``.

    Stores model weights, optimizer state, update counters, entropy-controller
    state, RNG states, and config snapshots. A checkpoint written here is
    marked ``checkpoint_format == "ppo_full"`` so :func:`load_ppo_resume_payload`
    can distinguish it from weights-only checkpoints.
    """
    path = Path(path)
    payload: dict[str, Any] = {
        "tokenizer_version": TOKENIZER_VERSION,
        "checkpoint_format": PPO_CHECKPOINT_FORMAT,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "update_count": int(update_count),
        "total_steps": int(total_steps),
        "planned_updates": int(planned_updates),
        "entropy_coeff": float(entropy_coeff),
        "entropy_signal_ema": (
            float(entropy_signal_ema) if entropy_signal_ema is not None else None
        ),
        "lr": float(lr),
        "rng_states": capture_rng_states(),
    }
    if log_alpha is not None:
        payload["log_alpha"] = log_alpha.detach().cpu()
    if alpha_optimizer is not None:
        payload["alpha_optimizer_state_dict"] = alpha_optimizer.state_dict()
    if return_rms is not None:
        payload["return_rms"] = {
            "mean": float(return_rms.mean),
            "var": float(return_rms.var),
            "count": float(return_rms.count),
        }
    if agent_config is not None:
        if dataclasses.is_dataclass(agent_config):
            payload["agent_config"] = dataclasses.asdict(cast("Any", agent_config))
        else:
            payload["agent_config"] = agent_config
    if ppo_config_fields:
        payload["ppo_config_fields"] = ppo_config_fields
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def is_ppo_full_checkpoint(path: str | Path, device: torch.device | str) -> bool:
    """Return True if ``path`` points at a full PPO resume checkpoint.

    Full PPO checkpoints store optimizer state, RNG state, and config snapshots,
    so they are loaded with ``weights_only=False``. They are only ever written by
    :func:`save_ppo_checkpoint` (our own training code), so this is safe.
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    return isinstance(blob, dict) and blob.get("checkpoint_format") == PPO_CHECKPOINT_FORMAT


def load_ppo_resume_payload(
    path: str | Path,
    device: torch.device | str,
) -> dict[str, Any]:
    """Load a full PPO checkpoint, raising clearly for weights-only files.

    Raises ``RuntimeError`` if the checkpoint lacks the optimizer/counter state
    required for a strict resume (i.e. it is a weights-only checkpoint).

    Loaded with ``weights_only=False`` because full PPO checkpoints carry
    optimizer state, NumPy/Python RNG state, and config snapshots. These files
    are only written by :func:`save_ppo_checkpoint` (our own training code).
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    if not (isinstance(blob, dict) and blob.get("checkpoint_format") == PPO_CHECKPOINT_FORMAT):
        raise RuntimeError(
            f"Checkpoint {path} is a weights-only checkpoint; "
            "use --pretrained or resume from a full PPO checkpoint."
        )
    saved_version = blob.get("tokenizer_version")
    if saved_version != TOKENIZER_VERSION:
        raise RuntimeError(
            f"Checkpoint {path} was saved with tokenizer_version="
            f"{saved_version!r}, but current TOKENIZER_VERSION="
            f"{TOKENIZER_VERSION}. Observation format has changed; "
            "retrain or pin the tokenizer version."
        )
    return blob
