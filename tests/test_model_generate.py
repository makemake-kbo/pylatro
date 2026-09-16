from __future__ import annotations

import numpy as np

from pylatro import load_game_data
from pylatro_agent.training.model_generate import (
    _make_vectorized_envs,
    _snapshot_vector_observations,
)
from pylatro_agent.vocab import build_vocab


def test_snapshot_vector_observations_preserves_compact_dtypes() -> None:
    source = {
        "tokens": np.arange(24, dtype=np.int16).reshape(2, 3, 4),
        "action_mask": np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8),
        "scalars": np.arange(6, dtype=np.float32).reshape(2, 3),
    }

    snapshots = _snapshot_vector_observations(source)

    assert len(snapshots) == 2
    for index, snapshot in enumerate(snapshots):
        for key, values in source.items():
            np.testing.assert_array_equal(snapshot[key], values[index])
            assert snapshot[key].dtype == values.dtype
            assert not np.shares_memory(snapshot[key], values)


def test_model_generation_env_disables_unused_teacher() -> None:
    data = load_game_data()
    env = _make_vectorized_envs(
        1,
        data,
        build_vocab(data),
        max_no_progress_steps=100,
        win_ante=8,
        use_async=False,
    )
    try:
        assert env.envs[0]._teacher is None
    finally:
        env.close()
