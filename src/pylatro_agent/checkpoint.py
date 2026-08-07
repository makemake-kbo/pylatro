"""Checkpoint save/load with a tokenizer-version guard.

Every supported checkpoint contains a ``tokenizer_version`` and state dict so
an observation-schema mismatch fails loudly instead of silently mis-aligning.

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

from .constants import TOKENIZER_SEMANTICS, TOKENIZER_VERSION

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
        "tokenizer_semantics": TOKENIZER_SEMANTICS,
        "state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    payload["tokenizer_version"] = TOKENIZER_VERSION
    payload["tokenizer_semantics"] = TOKENIZER_SEMANTICS
    torch.save(payload, path)
    return path


def load_checkpoint_payload(
    path: str | Path,
    device: torch.device | str,
    *,
    allow_compatible_tokenizer: bool = False,
) -> dict[str, Any]:
    """Load a checkpoint and verify its tokenizer version.

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
    if not isinstance(blob, dict) or "state_dict" not in blob:
        raise RuntimeError(
            f"Checkpoint {path} has no versioned payload. Raw state dicts are unsupported; "
            "retrain with the current tokenizer."
        )
    saved_version = blob.get("tokenizer_version")
    append_only_compatible = (
        allow_compatible_tokenizer
        and TOKENIZER_VERSION == 7
        and saved_version in {5, 6}
    )
    if saved_version != TOKENIZER_VERSION and not append_only_compatible:
        raise RuntimeError(
            f"Checkpoint {path} was saved with tokenizer_version="
            f"{saved_version!r}, but current TOKENIZER_VERSION="
            f"{TOKENIZER_VERSION}. Observation format has changed; "
            "retrain or pin the tokenizer version."
        )
    if append_only_compatible:
        logger.warning(
            "Loading tokenizer_version=%s checkpoint into append-only tokenizer_version=7; "
            "new risk/strategy adapters remain freshly zero-initialized.",
            saved_version,
        )
    _validate_tokenizer_semantics(blob, path, allow_append_only_compatible=append_only_compatible)
    return blob


def _validate_tokenizer_semantics(
    blob: dict[str, Any],
    path: str | Path,
    *,
    allow_append_only_compatible: bool = False,
) -> None:
    """Reject known shape-compatible tokenizer semantic mismatches.

    Old weights-only v6 files contain no semantic marker and generally cannot
    be distinguished. Full PPO checkpoints from the faulty 100fbf0 generation
    can be identified by reward_model_version=11 and are rejected explicitly.
    """
    saved_version = blob.get("tokenizer_version")
    saved_semantics = blob.get("tokenizer_semantics")
    compatible_v6_semantics = (
        allow_append_only_compatible
        and saved_version == 6
        and saved_semantics == "v6_projected_score_blind_ratio"
    )
    if saved_semantics is not None and saved_semantics != TOKENIZER_SEMANTICS and not compatible_v6_semantics:
        raise RuntimeError(
            f"Checkpoint {path} uses tokenizer_semantics={saved_semantics!r}, but the active "
            f"semantics are {TOKENIZER_SEMANTICS!r}. Candidate token meanings are incompatible."
        )
    if saved_version == 6 and saved_semantics is None and blob.get("reward_model_version") == 11:
        raise RuntimeError(
            f"Checkpoint {path} is a known faulty tokenizer-v6/reward-model-11 artifact from "
            "the 100fbf0 candidate-field remap. Use a bef33f6/6060673-era checkpoint instead."
        )


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
        # torch.load(map_location=device) may have moved the saved CPU state
        # to an accelerator; set_rng_state requires a CPU ByteTensor.
        torch.set_rng_state(states["torch_cpu"].to(device="cpu", dtype=torch.uint8))
    if "torch_cuda" in states and torch.cuda.is_available():
        # Checkpoints loaded with map_location="cuda" move these tensors onto
        # the accelerator, but set_rng_state_all requires CPU ByteTensors.
        cuda_states = [state.to(device="cpu", dtype=torch.uint8) for state in states["torch_cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


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
    reward_config: Any = None,
    ppo_config_fields: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist a *full* PPO checkpoint that supports strict ``--resume``.

    Stores model weights, optimizer state, update counters, entropy-controller
    state, RNG states, and config snapshots. A checkpoint written here is
    marked ``checkpoint_format == "ppo_full"`` so :func:`load_ppo_resume_payload`
    can distinguish it from weights-only checkpoints.
    """
    from .reward import RewardConfig, reward_checkpoint_metadata

    if not isinstance(reward_config, RewardConfig):
        raise ValueError("save_ppo_checkpoint requires the active RewardConfig")

    path = Path(path)
    payload: dict[str, Any] = {
        "tokenizer_version": TOKENIZER_VERSION,
        "tokenizer_semantics": TOKENIZER_SEMANTICS,
        "checkpoint_format": PPO_CHECKPOINT_FORMAT,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "update_count": int(update_count),
        "total_steps": int(total_steps),
        "planned_updates": int(planned_updates),
        "entropy_coeff": float(entropy_coeff),
        "entropy_signal_ema": (float(entropy_signal_ema) if entropy_signal_ema is not None else None),
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
    # Reserved reward metadata is always derived from the active configuration;
    # callers cannot override it through ``extra``.
    payload.update(reward_checkpoint_metadata(reward_config))
    payload["tokenizer_version"] = TOKENIZER_VERSION
    payload["tokenizer_semantics"] = TOKENIZER_SEMANTICS
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
    *,
    active_reward_config: Any,
    active_win_ante: int | None = None,
) -> dict[str, Any]:
    """Load a full PPO checkpoint, raising clearly for weights-only files.

    Raises ``RuntimeError`` if the checkpoint lacks the optimizer/counter state
    required for a strict resume, or if its reward fingerprint does not match
    the active RewardConfig and reward-model version.

    Loaded with ``weights_only=False`` because full PPO checkpoints carry
    optimizer state, NumPy/Python RNG state, and config snapshots. These files
    are only written by :func:`save_ppo_checkpoint` (our own training code).
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    if not (isinstance(blob, dict) and blob.get("checkpoint_format") == PPO_CHECKPOINT_FORMAT):
        raise RuntimeError(
            f"Checkpoint {path} is a weights-only checkpoint; use --pretrained or resume from a full PPO checkpoint."
        )
    saved_version = blob.get("tokenizer_version")
    if saved_version != TOKENIZER_VERSION:
        raise RuntimeError(
            f"Checkpoint {path} was saved with tokenizer_version="
            f"{saved_version!r}, but current TOKENIZER_VERSION="
            f"{TOKENIZER_VERSION}. Observation format has changed; "
            "retrain or pin the tokenizer version."
        )
    _validate_tokenizer_semantics(blob, path)

    from .reward import REWARD_MODEL_VERSION, reward_config_fingerprint

    saved_fingerprint = blob.get("reward_fingerprint")
    if not saved_fingerprint:
        raise RuntimeError(
            f"Checkpoint {path} has no reward fingerprint, so strict --resume cannot "
            "verify that its critic targets match the active reward model. Load it "
            "through --pretrained instead, with --reinit-value-head "
            "--critic-warmup-updates 15 --critic-warmup-min-ev 0."
        )
    active_fingerprint = reward_config_fingerprint(active_reward_config)
    if saved_fingerprint != active_fingerprint:
        raise RuntimeError(
            f"Checkpoint {path} reward fingerprint mismatch "
            f"(saved={str(saved_fingerprint)[:12]}, active={active_fingerprint[:12]}, "
            f"saved_version={blob.get('reward_model_version')!r}, "
            f"active_version={REWARD_MODEL_VERSION}). Strict --resume would restore "
            "a critic and Adam state trained on different return targets. Use "
            "--pretrained instead, with --reinit-value-head "
            "--critic-warmup-updates 15 --critic-warmup-min-ev 0."
        )

    saved_fields = blob.get("ppo_config_fields") or {}
    if "win_ante" in saved_fields:
        saved_win_ante = int(saved_fields.get("win_ante") or 8)
        effective_active_win_ante = int(active_win_ante or 8)
        if saved_win_ante != effective_active_win_ante:
            raise RuntimeError(
                f"Checkpoint {path} win_ante mismatch "
                f"(saved={saved_win_ante}, active={effective_active_win_ante}). "
                "Strict --resume would restore critic and optimizer state from a "
                "different terminal task. Use --pretrained --reinit-value-head "
                "--critic-warmup-updates 15 --critic-warmup-min-ev 0 instead."
            )
    return blob
