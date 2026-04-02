from __future__ import annotations

from types import SimpleNamespace

import pytest

from pylatro_agent.reward import default_reward


def _dummy_state(ante: int = 1, interest_cap: int = 25) -> SimpleNamespace:
    return SimpleNamespace(
        round_resets=SimpleNamespace(ante=ante),
        interest_cap=interest_cap,
    )


def test_default_reward_rewards_round_score_progress() -> None:
    state = _dummy_state()
    prev_info = {
        "ante": 1,
        "round_score": 100,
        "blind_target": 800,
    }
    curr_info = {
        "round_score": 500,
        "blind_target": 800,
        "progress_made": True,
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    expected = 0.5 * ((500 / 800) - (100 / 800)) - 0.001
    assert reward == pytest.approx(expected)


def test_default_reward_penalizes_idle_step_more_than_progress_step() -> None:
    state = _dummy_state()
    prev_info = {
        "ante": 1,
        "round_score": 0,
        "blind_target": 300,
    }
    curr_info = {
        "round_score": 0,
        "blind_target": 300,
        "progress_made": False,
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    assert reward == pytest.approx(-0.01)
