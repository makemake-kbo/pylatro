"""Observation staging, batching, and vector-environment info handling."""

from __future__ import annotations

import logging

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

logger = logging.getLogger(__name__)


_MISSING = object()


def _observation_buffer_batch(obs_buf) -> dict[str, torch.Tensor]:
    """Expose an observation buffer through the common model-batch interface."""
    return {
        "tokens": obs_buf.tokens,
        "token_types": obs_buf.token_types,
        "scalars": obs_buf.scalars,
        "attention_mask": obs_buf.attention_mask,
        "action_mask": obs_buf.action_mask,
        "history_events": obs_buf.history_events,
        "history_event_features": obs_buf.history_event_features,
        "history_cards": obs_buf.history_cards,
        "history_card_mask": obs_buf.history_card_mask,
        "history_jokers": obs_buf.history_jokers,
        "history_joker_mask": obs_buf.history_joker_mask,
        "history_event_mask": obs_buf.history_event_mask,
        "history_round_mask": obs_buf.history_round_mask,
        "history_omitted": obs_buf.history_omitted,
    }


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
        [
            bool(
                _extract_step_info_value(
                    infos,
                    "stalled",
                    env_idx,
                    done=bool(dones[env_idx]),
                    default=False,
                )
            )
            for env_idx in range(len(dones))
        ],
        dtype=np.bool_,
    )
    ppo_terminated = np.asarray(terminated, dtype=np.bool_) | stalled
    ppo_truncated = np.asarray(truncated, dtype=np.bool_) & ~ppo_terminated
    return ppo_terminated, ppo_truncated, stalled


def _obs_dicts_to_batch(obs_list: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    """Stack a list of single-env observations into a model batch."""
    return {
        "tokens": torch.tensor(np.stack([obs["tokens"] for obs in obs_list]), dtype=torch.long, device=device),
        "token_types": torch.tensor(
            np.stack([obs["token_types"] for obs in obs_list]),
            dtype=torch.long,
            device=device,
        ),
        "scalars": torch.tensor(np.stack([obs["scalars"] for obs in obs_list]), dtype=torch.float32, device=device),
        "attention_mask": torch.tensor(
            np.stack([obs["attention_mask"] for obs in obs_list]),
            dtype=torch.long,
            device=device,
        ),
        "action_mask": torch.tensor(
            np.stack([obs["action_mask"] for obs in obs_list]),
            dtype=torch.float32,
            device=device,
        ),
        "history_events": torch.tensor(
            np.stack([obs["history_events"] for obs in obs_list]), dtype=torch.long, device=device
        ),
        "history_event_features": torch.tensor(
            np.stack([obs["history_event_features"] for obs in obs_list]), dtype=torch.float32, device=device
        ),
        "history_cards": torch.tensor(
            np.stack([obs["history_cards"] for obs in obs_list]), dtype=torch.long, device=device
        ),
        "history_card_mask": torch.tensor(
            np.stack([obs["history_card_mask"] for obs in obs_list]), dtype=torch.long, device=device
        ),
        "history_jokers": torch.tensor(
            np.stack([obs["history_jokers"] for obs in obs_list]), dtype=torch.long, device=device
        ),
        "history_joker_mask": torch.tensor(
            np.stack([obs["history_joker_mask"] for obs in obs_list]), dtype=torch.long, device=device
        ),
        "history_event_mask": torch.tensor(
            np.stack([obs["history_event_mask"] for obs in obs_list]), dtype=torch.long, device=device
        ),
        "history_round_mask": torch.tensor(
            np.stack([obs["history_round_mask"] for obs in obs_list]), dtype=torch.long, device=device
        ),
        "history_omitted": torch.tensor(
            np.stack([obs["history_omitted"] for obs in obs_list]), dtype=torch.float32, device=device
        ),
    }


class _ObsBuffer:
    """Pre-allocated GPU/device tensors for batched observations.

    Avoids re-creating tensors every step by writing into existing storage.
    """

    def __init__(self, num_envs: int, device: torch.device) -> None:
        self.num_envs = num_envs
        self.device = device
        self.tokens = torch.zeros(num_envs, MAX_SEQ_LEN, TOKEN_DIM, dtype=torch.long, device=device)
        self.token_types = torch.zeros(num_envs, MAX_SEQ_LEN, dtype=torch.long, device=device)
        self.scalars = torch.zeros(num_envs, SCALAR_DIM, dtype=torch.float32, device=device)
        self.attention_mask = torch.zeros(num_envs, MAX_SEQ_LEN, dtype=torch.long, device=device)
        self.action_mask = torch.zeros(num_envs, NUM_ACTIONS, dtype=torch.float32, device=device)
        self.history_events = torch.zeros(
            num_envs, HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_EVENT_DIM, dtype=torch.long, device=device
        )
        self.history_event_features = torch.zeros(
            num_envs,
            HISTORY_ROUNDS,
            HISTORY_MAX_PLAYS,
            HISTORY_FEATURE_DIM,
            dtype=torch.float32,
            device=device,
        )
        self.history_cards = torch.zeros(
            num_envs,
            HISTORY_ROUNDS,
            HISTORY_MAX_PLAYS,
            HISTORY_MAX_CARDS,
            TOKEN_DIM,
            dtype=torch.long,
            device=device,
        )
        self.history_card_mask = torch.zeros(
            num_envs, HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_CARDS, dtype=torch.long, device=device
        )
        self.history_jokers = torch.zeros(
            num_envs, HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_JOKERS, dtype=torch.long, device=device
        )
        self.history_joker_mask = torch.zeros_like(self.history_jokers)
        self.history_event_mask = torch.zeros(
            num_envs, HISTORY_ROUNDS, HISTORY_MAX_PLAYS, dtype=torch.long, device=device
        )
        self.history_round_mask = torch.zeros(num_envs, HISTORY_ROUNDS, dtype=torch.long, device=device)
        self.history_omitted = torch.zeros(
            num_envs, HISTORY_ROUNDS, HISTORY_OMITTED_DIM, dtype=torch.float32, device=device
        )

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
        return {
            "tokens": self._np_tokens,
            "token_types": self._np_token_types,
            "scalars": self._np_scalars,
            "attention_mask": self._np_attention_mask,
            "action_mask": self._np_action_mask,
            "history_events": self._np_history_events,
            "history_event_features": self._np_history_event_features,
            "history_cards": self._np_history_cards,
            "history_card_mask": self._np_history_card_mask,
            "history_jokers": self._np_history_jokers,
            "history_joker_mask": self._np_history_joker_mask,
            "history_event_mask": self._np_history_event_mask,
            "history_round_mask": self._np_history_round_mask,
            "history_omitted": self._np_history_omitted,
        }


def _single_obs_to_batch(obs: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.tensor(obs["tokens"], dtype=torch.long, device=device).unsqueeze(0),
        "token_types": torch.tensor(obs["token_types"], dtype=torch.long, device=device).unsqueeze(0),
        "scalars": torch.tensor(obs["scalars"], dtype=torch.float32, device=device).unsqueeze(0),
        "attention_mask": torch.tensor(obs["attention_mask"], dtype=torch.long, device=device).unsqueeze(0),
        "action_mask": torch.tensor(obs["action_mask"], dtype=torch.float32, device=device).unsqueeze(0),
        "history_events": torch.tensor(obs["history_events"], dtype=torch.long, device=device).unsqueeze(0),
        "history_event_features": torch.tensor(
            obs["history_event_features"], dtype=torch.float32, device=device
        ).unsqueeze(0),
        "history_cards": torch.tensor(obs["history_cards"], dtype=torch.long, device=device).unsqueeze(0),
        "history_card_mask": torch.tensor(obs["history_card_mask"], dtype=torch.long, device=device).unsqueeze(0),
        "history_jokers": torch.tensor(obs["history_jokers"], dtype=torch.long, device=device).unsqueeze(0),
        "history_joker_mask": torch.tensor(obs["history_joker_mask"], dtype=torch.long, device=device).unsqueeze(0),
        "history_event_mask": torch.tensor(obs["history_event_mask"], dtype=torch.long, device=device).unsqueeze(0),
        "history_round_mask": torch.tensor(obs["history_round_mask"], dtype=torch.long, device=device).unsqueeze(0),
        "history_omitted": torch.tensor(obs["history_omitted"], dtype=torch.float32, device=device).unsqueeze(0),
    }
