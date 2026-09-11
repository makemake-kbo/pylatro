"""Explicit transformer compute precision; parameters and RL heads stay FP32."""

from __future__ import annotations

from contextlib import nullcontext
from functools import cache

import torch


@cache
def _supports_native_bf16(device_index: int) -> bool:
    with torch.cuda.device(device_index):
        return torch.cuda.is_bf16_supported(including_emulation=False)


def backbone_autocast(precision: str, device: torch.device):
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be 'fp32' or 'bf16'")
    # Checkpoints remain usable for CPU/MPS evaluation. Their FP32 parameters
    # need no conversion, and the CUDA-only acceleration is simply inactive.
    if precision == "fp32" or device.type != "cuda":
        return nullcontext()
    index = device.index if device.index is not None else torch.cuda.current_device()
    if not _supports_native_bf16(index):
        raise RuntimeError("BF16 compute requires native CUDA BF16 support; choose precision='fp32'")
    # Scoped inside model.forward so DataParallel worker threads use it too.
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
