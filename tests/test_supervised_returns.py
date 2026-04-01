from __future__ import annotations

import numpy as np
import pytest
import torch

from pylatro_agent.constants import MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM
from pylatro_agent.training.supervised import _collate_batch, _discounted_returns


def _dummy_obs() -> dict[str, np.ndarray]:
    return {
        "tokens": np.zeros((MAX_SEQ_LEN, TOKEN_DIM), dtype=np.int16),
        "token_types": np.zeros((MAX_SEQ_LEN,), dtype=np.int8),
        "scalars": np.zeros((SCALAR_DIM,), dtype=np.float32),
        "attention_mask": np.ones((MAX_SEQ_LEN,), dtype=np.int8),
        "action_mask": np.ones((NUM_ACTIONS,), dtype=np.int8),
    }


def test_discounted_returns_tracks_reward_to_go() -> None:
    returns = _discounted_returns([1.0, 0.5, -2.0], gamma=0.9)

    assert returns == [pytest.approx(-0.17), pytest.approx(-1.3), pytest.approx(-2.0)]


def test_collate_batch_prefers_recorded_return_target() -> None:
    batch = _collate_batch(
        [
            {
                "obs": _dummy_obs(),
                "action": 0,
                "won": False,
                "max_ante": 4,
                "return_target": -0.25,
            }
        ],
        torch.device("cpu"),
    )

    assert batch["value_target"].tolist() == [-0.25]


def test_collate_batch_falls_back_to_legacy_value_target() -> None:
    batch = _collate_batch(
        [
            {
                "obs": _dummy_obs(),
                "action": 0,
                "won": False,
                "max_ante": 3,
            }
        ],
        torch.device("cpu"),
    )

    assert batch["value_target"].tolist() == [-7.0]
