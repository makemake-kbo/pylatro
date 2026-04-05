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

    expected = (500 / 800) - (100 / 800)
    assert reward == pytest.approx(expected)


def test_default_reward_blind_clear_bonus_stays_modest() -> None:
    state = _dummy_state()
    prev_info = {
        "ante": 1,
        "round_score": 0,
        "blind_target": 400,
    }
    curr_info = {
        "round_score": 400,
        "blind_target": 400,
        "progress_made": True,
        "blind_just_beaten": True,
        "hands_left": 3,
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    assert reward == pytest.approx(1.0 + 0.5 + 0.15)


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

    assert reward == pytest.approx(-(0.001 + (20 - 8) * 0.0005))


def test_default_reward_caps_idle_penalty() -> None:
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
        "steps_since_progress": 200,
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    assert reward == pytest.approx(-0.02)


def test_default_reward_stalled_terminal_is_harsher_than_true_loss() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1}
    ordinary_loss = default_reward(state, prev_info, {"stalled": False}, terminated=True, won=False)
    stalled_loss = default_reward(state, prev_info, {"stalled": True}, terminated=True, won=False)

    assert ordinary_loss == pytest.approx(-10.0)
    assert stalled_loss == pytest.approx(-11.5)
    assert stalled_loss < ordinary_loss


def test_default_reward_stalled_truncation_uses_stall_penalty() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1}

    reward = default_reward(state, prev_info, {"stalled": True}, terminated=False, won=False)

    assert reward == pytest.approx(-11.5)
