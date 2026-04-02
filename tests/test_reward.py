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

    expected = 1.5 * ((500 / 800) - (100 / 800)) - 0.001
    assert reward == pytest.approx(expected)


def test_default_reward_gives_idle_grace_before_ramping_penalty() -> None:
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
        "steps_since_progress": 3,
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    assert reward == pytest.approx(-0.001)


def test_default_reward_ramps_idle_penalty_after_grace_window() -> None:
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
        "steps_since_progress": 20,
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    assert reward == pytest.approx(-(0.001 + (20 - 8) * 0.0003))


def test_default_reward_stalled_terminal_is_less_harsh_than_true_loss() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1}
    curr_info = {"stalled": True}

    reward = default_reward(state, prev_info, curr_info, terminated=True, won=False)

    assert reward == pytest.approx(-6.0)
