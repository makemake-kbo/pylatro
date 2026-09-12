"""Worker wire packing must be lossless for every legal-mask bit."""

import pickle

import numpy as np
import pytest

from pylatro_agent.constants import NUM_ACTIONS
from pylatro_agent.env import BalatroEnv
from pylatro_agent.training.evaluation_envs import _pack_observation, _unpack_observation


@pytest.mark.parametrize("mask_kind", ["empty", "full", "random", "last"])
def test_observation_wire_round_trip(mask_kind):
    env = BalatroEnv(seed=1701, enable_teacher=False)
    try:
        obs, _ = env.reset()
    finally:
        env.close()
    if mask_kind == "empty":
        obs["action_mask"].fill(0)
    elif mask_kind == "full":
        obs["action_mask"].fill(1)
    elif mask_kind == "last":
        obs["action_mask"].fill(0)
        obs["action_mask"][-1] = 1
    else:
        obs["action_mask"][:] = np.random.default_rng(73).integers(0, 2, size=NUM_ACTIONS, dtype=np.int8)
    before = {key: value.copy() for key, value in obs.items()}
    packed = _pack_observation(obs)
    assert packed["action_mask"].nbytes == (NUM_ACTIONS + 7) // 8
    assert len(pickle.dumps(packed, protocol=pickle.DEFAULT_PROTOCOL)) < len(
        pickle.dumps(obs, protocol=pickle.DEFAULT_PROTOCOL)
    )
    actual = _unpack_observation(pickle.loads(pickle.dumps(packed, protocol=pickle.DEFAULT_PROTOCOL)))
    assert actual.keys() == obs.keys()
    for key in obs:
        np.testing.assert_array_equal(obs[key], before[key])
        np.testing.assert_array_equal(actual[key], before[key])
        assert actual[key].dtype == before[key].dtype
        assert actual[key].shape == before[key].shape
