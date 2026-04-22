"""Checkpoint save/load with tokenizer-version guard.

Legacy checkpoints were raw `state_dict()` tensors. New checkpoints are a
dict with a `tokenizer_version` field plus the state dict, so that loading
an old model against a newer observation format fails loudly instead of
silently mis-aligning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .constants import TOKENIZER_VERSION

logger = logging.getLogger(__name__)


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
    """
    blob = torch.load(path, map_location=device, weights_only=True)
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
