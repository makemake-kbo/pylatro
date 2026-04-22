from __future__ import annotations

from types import SimpleNamespace

import pytest

from pylatro_agent.reward import REWARD_SCALE, default_reward


def _dummy_state(ante: int = 1, interest_cap: int = 25, win_ante: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        round_resets=SimpleNamespace(ante=ante),
        interest_cap=interest_cap,
        win_ante=win_ante,
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

    expected = ((500 / 800) - (100 / 800)) * 0.25
    assert reward == pytest.approx(expected * REWARD_SCALE)


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

    assert reward == pytest.approx(REWARD_SCALE * (0.25 + 0.5 + 0.15))


def test_default_reward_penalizes_spending_resources_without_relieving_pressure() -> None:
    state = _dummy_state()
    prev_info = {
        "round_score": 0,
        "blind_target": 400,
        "hands_left": 4,
        "discards_left": 2,
        "phase": "hand_play",
        "progress_made": True,
    }
    curr_info = {
        "round_score": 0,
        "blind_target": 400,
        "hands_left": 4,
        "discards_left": 1,
        "phase": "hand_play",
        "progress_made": True,
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    expected = 1.5 * ((1.0 / (4 + 0.5 * 2)) - (1.0 / (4 + 0.5 * 1)))
    assert reward == pytest.approx(expected * REWARD_SCALE)
    assert reward < 0.0


def test_default_reward_rewards_relieving_pressure_during_hand_play() -> None:
    state = _dummy_state()
    prev_info = {
        "round_score": 0,
        "blind_target": 400,
        "hands_left": 4,
        "discards_left": 2,
        "phase": "hand_play",
        "progress_made": True,
    }
    curr_info = {
        "round_score": 160,
        "blind_target": 400,
        "hands_left": 3,
        "discards_left": 2,
        "phase": "hand_play",
        "progress_made": True,
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    expected_score_progress = 0.25 * (160 / 400)
    expected_pressure_progress = 1.5 * ((1.0 / (4 + 0.5 * 2)) - ((1.0 - 160 / 400) / (3 + 0.5 * 2)))
    assert reward == pytest.approx(REWARD_SCALE * (expected_score_progress + expected_pressure_progress))
    assert reward > REWARD_SCALE * expected_score_progress


def test_default_reward_treats_blind_clear_as_full_pressure_relief() -> None:
    state = _dummy_state()
    prev_info = {
        "round_score": 300,
        "blind_target": 400,
        "hands_left": 2,
        "discards_left": 1,
        "phase": "hand_play",
        "progress_made": True,
    }
    curr_info = {
        "round_score": 450,
        "blind_target": 400,
        "hands_left": 1,
        "discards_left": 1,
        "phase": "shop",
        "progress_made": True,
        "blind_just_beaten": True,
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    expected_score_progress = 0.25 * ((400 / 400) - (300 / 400))
    expected_pressure_progress = 1.5 * ((0.25) / (2 + 0.5 * 1))
    expected_blind_clear = 0.5 + 0.05
    assert reward == pytest.approx(REWARD_SCALE * (expected_score_progress + expected_pressure_progress + expected_blind_clear))


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

    assert reward == pytest.approx(-0.001 * REWARD_SCALE)


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

    assert reward == pytest.approx(REWARD_SCALE * -(0.001 + (20 - 8) * 0.0005))


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

    assert reward == pytest.approx(-0.02 * REWARD_SCALE)


def test_default_reward_rewards_targeted_consumable_use() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1, "round_score": 0, "blind_target": 300}
    for action_type in ("use_consumable_hand_subset", "use_consumable_joker"):
        curr_info = {
            "round_score": 0,
            "blind_target": 300,
            "progress_made": True,
            "action_type": action_type,
        }
        reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)
        assert reward == pytest.approx(0.1 * REWARD_SCALE)


def test_default_reward_does_not_reward_notarget_consumable_use() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1, "round_score": 0, "blind_target": 300}
    curr_info = {
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "use_consumable_no_target",
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    assert reward == pytest.approx(0.0)


def test_default_reward_rewards_shop_reroll() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1, "round_score": 0, "blind_target": 300}
    curr_info = {
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "shop_reroll",
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    assert reward == pytest.approx(0.08 * REWARD_SCALE)


def test_default_reward_penalizes_shop_sell_joker() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1, "round_score": 0, "blind_target": 300}
    curr_info = {
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "shop_sell_joker",
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    assert reward == pytest.approx(-0.05 * REWARD_SCALE)


def test_default_reward_penalizes_shop_sell_consumable() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1, "round_score": 0, "blind_target": 300}
    curr_info = {
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "shop_sell_consumable",
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    assert reward == pytest.approx(-0.05 * REWARD_SCALE)


def test_default_reward_stalled_terminal_is_harsher_than_true_loss() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1}
    ordinary_loss = default_reward(state, prev_info, {"stalled": False}, terminated=True, won=False)
    stalled_loss = default_reward(state, prev_info, {"stalled": True}, terminated=True, won=False)

    # Ante 1 death on an 8-ante win target: base -30, unfinished antes 7 * 3.
    assert ordinary_loss == pytest.approx(-51.0 * REWARD_SCALE)
    assert stalled_loss == pytest.approx(-56.0 * REWARD_SCALE)
    assert stalled_loss < ordinary_loss


def test_default_reward_loss_penalty_scales_with_unfinished_antes() -> None:
    prev_info = {"ante": 1}

    early_loss = default_reward(_dummy_state(ante=1), prev_info, {"stalled": False}, terminated=True, won=False)
    later_loss = default_reward(_dummy_state(ante=6), prev_info, {"stalled": False}, terminated=True, won=False)

    # Dying earlier must strictly hurt more than dying deeper in the run.
    assert early_loss == pytest.approx(REWARD_SCALE * (-30.0 - 3.0 * 7))
    assert later_loss == pytest.approx(REWARD_SCALE * (-30.0 - 3.0 * 2))
    assert early_loss < later_loss


def test_default_reward_stalled_truncation_uses_stall_penalty() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1}

    reward = default_reward(state, prev_info, {"stalled": True}, terminated=False, won=False)

    assert reward == pytest.approx(-56.0 * REWARD_SCALE)
