"""Observation staging, batching, and vector-environment info handling."""

from __future__ import annotations

import numpy as np
import torch

from ..constants import (
    HISTORY_EVENT_DIM,
    HISTORY_FEATURE_DIM,
    HISTORY_MAX_CARDS,
    HISTORY_MAX_JOKERS,
    HISTORY_MAX_PLAYS,
    HISTORY_OMITTED_DIM,
    HISTORY_ROUNDS,
    MAX_SEQ_LEN,
    NUM_ACTIONS,
    SCALAR_DIM,
    TOKEN_DIM,
)

_MISSING = object()

# Shapes exclude the batch axis. Keep the model's observation contract in one
# place so staging and one-off batches always use the same keys and dtypes.
_OBSERVATION_SPEC = {
    "tokens": ((MAX_SEQ_LEN, TOKEN_DIM), torch.long),
    "token_types": ((MAX_SEQ_LEN,), torch.long),
    "scalars": ((SCALAR_DIM,), torch.float32),
    "attention_mask": ((MAX_SEQ_LEN,), torch.long),
    "action_mask": ((NUM_ACTIONS,), torch.float32),
    "history_events": ((HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_EVENT_DIM), torch.long),
    "history_event_features": ((HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_FEATURE_DIM), torch.float32),
    "history_cards": ((HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_CARDS, TOKEN_DIM), torch.long),
    "history_card_mask": ((HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_CARDS), torch.long),
    "history_jokers": ((HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_JOKERS), torch.long),
    "history_joker_mask": ((HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_JOKERS), torch.long),
    "history_event_mask": ((HISTORY_ROUNDS, HISTORY_MAX_PLAYS), torch.long),
    "history_round_mask": ((HISTORY_ROUNDS,), torch.long),
    "history_omitted": ((HISTORY_ROUNDS, HISTORY_OMITTED_DIM), torch.float32),
}


def _observation_buffer_batch(obs_buf) -> dict[str, torch.Tensor]:
    """Expose an observation buffer through the common model-batch interface."""
    return {key: getattr(obs_buf, key) for key in _OBSERVATION_SPEC}


def _extract_vector_info_value(info_dict: dict, key: str, env_idx: int, default=None):
    """Read one env's value from Gymnasium's vector-info dict-of-arrays format."""
    values = info_dict.get(key)
    if values is None:
        return default

    mask = info_dict.get(f"_{key}")
    if mask is not None and not bool(mask[env_idx]):
        return default

    value = values[env_idx]
    return value.item() if isinstance(value, np.generic) else value


def _extract_step_info_value(info_dict: dict, key: str, env_idx: int, *, done: bool, default=None):
    """Read the just-finished step's info, preferring final_info on autoresets."""
    if done:
        final_info = info_dict.get("final_info")
        if isinstance(final_info, dict):
            final_value = _extract_vector_info_value(final_info, key, env_idx, _MISSING)
            if final_value is not _MISSING:
                return final_value

    value = _extract_vector_info_value(info_dict, key, env_idx, _MISSING)
    return default if value is _MISSING else value


def _extract_step_count(info_dict: dict, key: str, env_idx: int, *, done: bool) -> int:
    return int(_extract_step_info_value(info_dict, key, env_idx, done=done, default=0) or 0)


def _extract_step_flag(info_dict: dict, key: str, env_idx: int, *, done: bool) -> bool:
    return bool(_extract_step_info_value(info_dict, key, env_idx, done=done, default=False))


def _ppo_terminal_flags(
    terminated: np.ndarray,
    truncated: np.ndarray,
    infos: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map external Gymnasium endings to PPO bootstrap semantics.

    No-progress stalls stay ``truncated=True`` at the environment boundary, but
    they carry a terminal loss reward and are absorbing for value targets. Other
    truncations keep Gymnasium's bootstrap-from-final-observation behavior.
    """
    dones = terminated | truncated
    stalled = np.asarray(
        [_extract_step_flag(infos, "stalled", env_idx, done=bool(done)) for env_idx, done in enumerate(dones)],
        dtype=np.bool_,
    )
    ppo_terminated = np.asarray(terminated, dtype=np.bool_) | stalled
    ppo_truncated = np.asarray(truncated, dtype=np.bool_) & ~ppo_terminated
    return ppo_terminated, ppo_truncated, stalled


def _obs_dicts_to_batch(obs_list: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    """Stack a list of single-env observations into a model batch."""
    return {
        key: torch.tensor(np.stack([obs[key] for obs in obs_list]), dtype=dtype, device=device)
        for key, (_, dtype) in _OBSERVATION_SPEC.items()
    }


class _ObsBuffer:
    """Pre-allocated GPU/device tensors for batched observations.

    Avoids re-creating tensors every step by writing into existing storage.
    """

    tokens: torch.Tensor
    token_types: torch.Tensor
    scalars: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    history_events: torch.Tensor
    history_event_features: torch.Tensor
    history_cards: torch.Tensor
    history_card_mask: torch.Tensor
    history_jokers: torch.Tensor
    history_joker_mask: torch.Tensor
    history_event_mask: torch.Tensor
    history_round_mask: torch.Tensor
    history_omitted: torch.Tensor

    def __init__(self, num_envs: int, device: torch.device) -> None:
        self.num_envs = num_envs
        self.device = device
        for key, (shape, dtype) in _OBSERVATION_SPEC.items():
            setattr(self, key, torch.zeros((num_envs, *shape), dtype=dtype, device=device))

        # CPU tensors and NumPy arrays share storage. CUDA uses persistent
        # pinned staging, so copies can be enqueued without a sync per field.
        self._host_tensors = {}
        for key, target in _observation_buffer_batch(self).items():
            host = (
                target
                if device.type == "cpu"
                else torch.zeros_like(target, device="cpu", pin_memory=device.type == "cuda")
            )
            self._host_tensors[key] = host
            setattr(self, f"_np_{key}", host.numpy())
        self._copy_complete = torch.cuda.Event() if device.type == "cuda" else None
        self._copy_pending = False

    def update(self, obs_dict: dict) -> None:
        """Stage observations on the current device stream.

        Wait before reusing pinned source storage: a second update must not
        mutate host memory while a previous asynchronous copy is still reading.
        Consumers must use the same stream, or explicitly wait on that stream.
        """
        if self._copy_pending:
            self._copy_complete.synchronize()
        for key in self._host_tensors:
            np.copyto(getattr(self, f"_np_{key}"), obs_dict[key])
        if self.device.type != "cpu":
            for key, host in self._host_tensors.items():
                getattr(self, key).copy_(host, non_blocking=self.device.type == "cuda")
        if self._copy_complete is not None:
            self._copy_complete.record(torch.cuda.current_stream(self.device))
            self._copy_pending = True

    def as_numpy_dict(self) -> dict:
        """Return current numpy arrays (for storing in rollout buffer)."""
        return {key: getattr(self, f"_np_{key}") for key in _OBSERVATION_SPEC}


def _single_obs_to_batch(obs: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.tensor(obs[key], dtype=dtype, device=device).unsqueeze(0)
        for key, (_, dtype) in _OBSERVATION_SPEC.items()
    }
