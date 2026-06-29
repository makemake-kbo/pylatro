from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from pylatro_agent.reward import (
    BLIND_CLEAR_REWARD,
    CONSUMABLE_TARGETED_USE_REWARD,
    HAND_SUBSET_BONUS_SCALE,
    HAND_TOP1_BONUS,
    HAND_TOP3_BONUS,
    HANDS_LEFT_BONUS_SCALE,
    PLANET_FOOL_OVERWRITE_PENALTY,
    PLANET_MATCH_BONUS,
    PLANET_PLAYED_HAND_BONUS,
    PLANET_SKIP_PENALTY,
    PRESSURE_PROGRESS_SCALE,
    PRETRAIN_STALL_EXTRA_PENALTY,
    PRETRAIN_WIN_VALUE,
    PPO_SPARSE_CONFIG,
    REWARD_SCALE,
    RewardConfig,
    SCORE_PROGRESS_SCALE,
    SHOP_REROLL_REWARD,
    SHOP_SELL_PENALTY,
    STANDARD_OVERFULL_CARD_BASE_PENALTY,
    STANDARD_OVERFULL_CARD_EXPONENT,
    TAROT_SKIP_FIXING_PENALTY,
    default_reward,
    default_reward_components,
    pretraining_outcome_value,
)


def _dummy_state(ante: int = 1, interest_cap: int = 25, win_ante: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        round_resets=SimpleNamespace(ante=ante),
        interest_cap=interest_cap,
        win_ante=win_ante,
    )


def _card(rank: str, suit: str, center_key: str = "c_base", seal: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        rank=rank,
        suit=suit,
        center_key=center_key,
        seal=seal,
        destroyed=False,
    )


def _pack_detail(
    rank: str,
    suit: str,
    center_key: str = "c_base",
    seal: str = "",
    edition: dict | None = None,
) -> dict:
    return {
        "rank": rank,
        "suit": suit,
        "center_key": center_key,
        "seal": seal,
        "edition": edition or {},
    }


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

    expected = ((500 / 800) - (100 / 800)) * SCORE_PROGRESS_SCALE
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

    assert reward == pytest.approx(
        REWARD_SCALE * (SCORE_PROGRESS_SCALE + BLIND_CLEAR_REWARD + 3 * HANDS_LEFT_BONUS_SCALE)
    )


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

    expected = PRESSURE_PROGRESS_SCALE * ((1.0 / (4 + 0.5 * 2)) - (1.0 / (4 + 0.5 * 1)))
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

    expected_score_progress = SCORE_PROGRESS_SCALE * (160 / 400)
    expected_pressure_progress = PRESSURE_PROGRESS_SCALE * (
        (1.0 / (4 + 0.5 * 2)) - ((1.0 - 160 / 400) / (3 + 0.5 * 2))
    )
    assert reward == pytest.approx(REWARD_SCALE * (expected_score_progress + expected_pressure_progress))
    assert reward > REWARD_SCALE * expected_score_progress


def test_default_reward_separates_good_hand_play_from_noisy_play() -> None:
    state = _dummy_state()
    prev_info = {
        "round_score": 0,
        "blind_target": 400,
        "hands_left": 4,
        "discards_left": 2,
        "phase": "hand_play",
        "progress_made": True,
    }
    good_curr_info = {
        "round_score": 160,
        "blind_target": 400,
        "hands_left": 3,
        "discards_left": 2,
        "phase": "hand_play",
        "progress_made": True,
        "action_type": "play_subset",
        "hand_play_candidate_value_ratio": 1.0,
    }
    noisy_curr_info = {
        "round_score": 0,
        "blind_target": 400,
        "hands_left": 3,
        "discards_left": 2,
        "phase": "hand_play",
        "progress_made": True,
        "action_type": "play_subset",
        "hand_play_not_in_candidates": True,
        "hand_play_candidate_value_ratio": 0.0,
    }

    good_reward = default_reward(state, prev_info, good_curr_info, terminated=False, won=False)
    noisy_reward = default_reward(state, prev_info, noisy_curr_info, terminated=False, won=False)

    assert good_reward > 0.0
    assert noisy_reward < 0.0
    expected_good = (
        SCORE_PROGRESS_SCALE * (160 / 400)
        + PRESSURE_PROGRESS_SCALE * ((1.0 / (4 + 0.5 * 2)) - ((1.0 - 160 / 400) / (3 + 0.5 * 2)))
        + HAND_SUBSET_BONUS_SCALE
    )
    expected_noisy = PRESSURE_PROGRESS_SCALE * (
        (1.0 / (4 + 0.5 * 2)) - (1.0 / (3 + 0.5 * 2))
    )
    assert good_reward - noisy_reward == pytest.approx(REWARD_SCALE * (expected_good - expected_noisy))


def test_dense_reward_components_are_large_but_below_win_outcome() -> None:
    state = _dummy_state(ante=8)
    prev_info = {
        "ante": 7,
        "round_score": 0,
        "blind_target": 400,
        "hands_left": 4,
        "discards_left": 2,
        "phase": "hand_play",
        "dollars": 25,
    }
    curr_info = {
        "round_score": 400,
        "blind_target": 400,
        "hands_left": 4,
        "discards_left": 2,
        "phase": "shop",
        "progress_made": True,
        "blind_just_beaten": True,
        "action_type": "play_subset",
        "hand_play_candidate_value_ratio": 1.0,
    }

    dense_total = default_reward(state, prev_info, curr_info, terminated=False, won=False)
    early_loss = abs(default_reward(_dummy_state(ante=1), {"ante": 1}, {"ante": 1}, terminated=True, won=False))
    win_reward = default_reward(_dummy_state(ante=8), {"ante": 8}, {"ante": 8}, terminated=True, won=True)

    assert dense_total > 0.0
    assert early_loss >= 9.0
    assert win_reward == pytest.approx(PRETRAIN_WIN_VALUE)
    assert dense_total > early_loss
    assert dense_total < win_reward


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

    expected_score_progress = SCORE_PROGRESS_SCALE * ((400 / 400) - (300 / 400))
    expected_pressure_progress = PRESSURE_PROGRESS_SCALE * ((0.25) / (2 + 0.5 * 1))
    expected_blind_clear = BLIND_CLEAR_REWARD + HANDS_LEFT_BONUS_SCALE
    assert reward == pytest.approx(
        REWARD_SCALE * (expected_score_progress + expected_pressure_progress + expected_blind_clear)
    )


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
        assert reward == pytest.approx(CONSUMABLE_TARGETED_USE_REWARD * REWARD_SCALE)


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


def test_hand_subset_bonus_scales_with_value_ratio() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1, "round_score": 0, "blind_target": 300}
    curr_info = {
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "play_subset",
        "hand_play_candidate_value_ratio": 0.5,
    }
    components = default_reward_components(state, prev_info, curr_info, terminated=False, won=False)
    expected = HAND_SUBSET_BONUS_SCALE * 0.5 * REWARD_SCALE
    assert components["hand_subset_bonus"] == pytest.approx(expected)


def test_hand_top_bonuses_reward_specific_heuristic_quality() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1, "round_score": 0, "blind_target": 300}
    base_curr_info = {
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "play_subset",
        "hand_play_candidate_value_ratio": 1.0,
    }

    top1 = default_reward_components(
        state,
        prev_info,
        {**base_curr_info, "hand_play_top1": True, "hand_play_top3": True},
        terminated=False,
        won=False,
    )
    top3 = default_reward_components(
        state,
        prev_info,
        {**base_curr_info, "hand_play_top1": False, "hand_play_top3": True},
        terminated=False,
        won=False,
    )

    assert top1["hand_top1_bonus"] == pytest.approx(HAND_TOP1_BONUS * REWARD_SCALE)
    assert top1["hand_top3_bonus"] == pytest.approx(0.0)
    assert top3["hand_top1_bonus"] == pytest.approx(0.0)
    assert top3["hand_top3_bonus"] == pytest.approx(HAND_TOP3_BONUS * REWARD_SCALE)
    assert top1["total"] > top3["total"]


def test_hand_subset_bonus_zero_when_not_in_candidates() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1, "round_score": 0, "blind_target": 300}
    curr_info = {
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "play_subset",
        "hand_play_not_in_candidates": True,
        "hand_play_candidate_value_ratio": 0.0,
    }
    components = default_reward_components(state, prev_info, curr_info, terminated=False, won=False)
    assert components["hand_subset_bonus"] == pytest.approx(0.0)


def test_hand_subset_bonus_zero_for_non_play_actions() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1, "round_score": 0, "blind_target": 300}
    curr_info = {
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "shop_buy",
        "hand_play_candidate_value_ratio": 1.0,
    }
    components = default_reward_components(state, prev_info, curr_info, terminated=False, won=False)
    assert components["hand_subset_bonus"] == pytest.approx(0.0)


def test_planet_match_bonus_rewards_correct_hand_type() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1, "round_score": 0, "blind_target": 300}
    curr_info = {
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "use_consumable_no_target",
        "planet_use_observed": True,
        "planet_use_main_hand_match": True,
    }
    components = default_reward_components(state, prev_info, curr_info, terminated=False, won=False)
    expected = PLANET_MATCH_BONUS * REWARD_SCALE
    assert components["planet_match_bonus"] == pytest.approx(expected)


def test_planet_claim_rewards_played_and_main_hand_alignment() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1, "round_score": 0, "blind_target": 300}
    curr_info = {
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "pack_claim",
        "planet_claim_observed": True,
        "planet_claim_played_hand": True,
        "planet_claim_main_hand_match": True,
    }

    components = default_reward_components(state, prev_info, curr_info, terminated=False, won=False)

    assert components["planet_played_hand_bonus"] == pytest.approx(PLANET_PLAYED_HAND_BONUS * REWARD_SCALE)
    assert components["planet_match_bonus"] == pytest.approx(PLANET_MATCH_BONUS * REWARD_SCALE)


def test_planet_match_bonus_zero_when_mismatch() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1, "round_score": 0, "blind_target": 300}
    curr_info = {
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "use_consumable_no_target",
        "planet_use_observed": True,
        "planet_use_main_hand_match": False,
    }
    components = default_reward_components(state, prev_info, curr_info, terminated=False, won=False)
    assert components["planet_match_bonus"] == pytest.approx(0.0)


def test_planet_unmatched_use_penalty_applies_only_when_enabled() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1, "round_score": 0, "blind_target": 300}
    curr_info = {
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "use_consumable_no_target",
        "planet_use_observed": True,
        "planet_use_main_hand_match": False,
    }

    # Default config: penalty disabled (coeff 0.0), so no unmatched penalty.
    default_components = default_reward_components(state, prev_info, curr_info, terminated=False, won=False)
    assert default_components["planet_unmatched_use_penalty"] == pytest.approx(0.0)

    # With the penalty enabled, an unmatched planet use incurs the penalty.
    config = RewardConfig(planet_unmatched_use_penalty_coeff=0.3)
    components = default_reward_components(state, prev_info, curr_info, terminated=False, won=False, config=config)
    assert components["planet_unmatched_use_penalty"] == pytest.approx(-0.3 * REWARD_SCALE)


def test_planet_unmatched_claim_penalty_applies_only_when_enabled() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1, "round_score": 0, "blind_target": 300}
    curr_info = {
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "pack_claim",
        "planet_claim_observed": True,
        "planet_claim_main_hand_match": False,
    }
    default_components = default_reward_components(state, prev_info, curr_info, terminated=False, won=False)
    assert default_components["planet_unmatched_claim_penalty"] == pytest.approx(0.0)

    config = RewardConfig(planet_unmatched_claim_penalty_coeff=0.25)
    components = default_reward_components(state, prev_info, curr_info, terminated=False, won=False, config=config)
    assert components["planet_unmatched_claim_penalty"] == pytest.approx(-0.25 * REWARD_SCALE)


def test_progression_reward_scale_scales_progress_components() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1, "round_score": 0, "blind_target": 300}
    curr_info = {
        "round_score": 300,
        "blind_target": 300,
        "progress_made": True,
        "blind_just_beaten": True,
        "hands_left": 2,
    }
    default_components = default_reward_components(
        state, prev_info, curr_info, terminated=False, won=False
    )
    config = RewardConfig(progression_reward_scale=0.5)
    scaled_components = default_reward_components(
        state, prev_info, curr_info, terminated=False, won=False, config=config
    )

    assert scaled_components["blind_clear"] == pytest.approx(default_components["blind_clear"] * 0.5)
    assert scaled_components["hands_bonus"] == pytest.approx(default_components["hands_bonus"] * 0.5)


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

    assert reward == pytest.approx(SHOP_REROLL_REWARD * REWARD_SCALE)


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

    assert reward == pytest.approx(-SHOP_SELL_PENALTY * REWARD_SCALE)


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

    assert reward == pytest.approx(-SHOP_SELL_PENALTY * REWARD_SCALE)


def test_default_reward_penalizes_skipping_tarot_pack_with_deck_fixing_target() -> None:
    state = _dummy_state(ante=2)
    state.deck_cards = [
        _card(rank, "Hearts")
        for rank in ("A", "K", "Q", "J", "T", "9", "8", "7", "6", "5") * 2
    ] + [_card(rank, "Spades") for rank in ("A", "K", "Q", "J", "T", "9", "8", "7")]
    prev_info = {
        "ante": 2,
        "joker_keys": (),
        "consumable_keys": (),
        "last_tarot_planet": "",
        "pack_state_name": "TAROT_PACK",
        "pack_card_details": ({"center_key": "c_death"},),
    }
    curr_info = {
        "ante": 2,
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "pack_skip",
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    assert reward == pytest.approx(-TAROT_SKIP_FIXING_PENALTY * REWARD_SCALE)


def test_default_reward_does_not_penalize_tarot_pack_skip_without_legit_target() -> None:
    state = _dummy_state(ante=2)
    state.deck_cards = [
        _card(rank, suit)
        for suit in ("Hearts", "Diamonds", "Clubs", "Spades")
        for rank in ("A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2")
    ]
    prev_info = {
        "ante": 2,
        "joker_keys": (),
        "consumable_keys": (),
        "last_tarot_planet": "",
        "pack_state_name": "TAROT_PACK",
        "pack_card_details": ({"center_key": "c_wheel_of_fortune"},),
    }
    curr_info = {
        "ante": 2,
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "pack_skip",
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    assert reward == pytest.approx(0.0)


def test_default_reward_does_not_penalize_tarot_pack_skip_for_fixed_deck() -> None:
    state = _dummy_state(ante=6)
    state.deck_cards = [
        _card(rank, "Hearts", "m_steel", "Red")
        for rank in ("A", "K", "Q", "J", "T", "9", "8", "7", "6", "5") * 3
    ]
    prev_info = {
        "ante": 6,
        "joker_keys": (),
        "consumable_keys": (),
        "last_tarot_planet": "",
        "pack_state_name": "TAROT_PACK",
        "pack_card_details": ({"center_key": "c_death"},),
    }
    curr_info = {
        "ante": 6,
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "pack_skip",
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    assert reward == pytest.approx(0.0)


def test_default_reward_does_not_penalize_high_ante_red_card_tarot_pack_skip() -> None:
    state = _dummy_state(ante=5)
    state.deck_cards = [_card("A", "Hearts") for _ in range(10)] + [_card("K", "Spades") for _ in range(20)]
    prev_info = {
        "ante": 5,
        "joker_keys": ("j_red_card",),
        "consumable_keys": (),
        "last_tarot_planet": "",
        "pack_state_name": "TAROT_PACK",
        "pack_card_details": ({"center_key": "c_death"},),
    }
    curr_info = {
        "ante": 5,
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "pack_skip",
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    assert reward == pytest.approx(0.0)


def test_default_reward_penalizes_standard_pack_claim_when_unfixed_deck_is_overfull() -> None:
    state = _dummy_state(ante=3)
    state.deck_cards = [_card("A", "Hearts") for _ in range(20)] + [_card("K", "Spades") for _ in range(34)]
    prev_info = {
        "ante": 3,
        "pack_state_name": "STANDARD_PACK",
        "pack_card_details": (_pack_detail("9", "Clubs"),),
    }
    curr_info = {
        "ante": 3,
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "pack_claim",
        "action_index": 0,
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    expected = STANDARD_OVERFULL_CARD_BASE_PENALTY * math.expm1(STANDARD_OVERFULL_CARD_EXPONENT * 2)
    assert reward == pytest.approx(-expected * REWARD_SCALE)


def test_default_reward_penalizes_buying_standard_pack_when_unfixed_deck_is_overfull() -> None:
    state = _dummy_state(ante=3)
    state.deck_cards = [_card("A", "Hearts") for _ in range(15)] + [_card("K", "Spades") for _ in range(38)]
    prev_info = {
        "ante": 3,
        "shop_item_details": (
            {"center_key": "j_joker", "pack_state_name": ""},
            {"center_key": "p_standard_normal_1", "pack_state_name": "STANDARD_PACK"},
        ),
    }
    curr_info = {
        "ante": 3,
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "shop_buy",
        "action_index": 1,
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    expected = STANDARD_OVERFULL_CARD_BASE_PENALTY * math.expm1(STANDARD_OVERFULL_CARD_EXPONENT)
    assert reward == pytest.approx(-expected * REWARD_SCALE)


def test_default_reward_does_not_penalize_standard_pack_claim_for_fixed_overfull_deck() -> None:
    state = _dummy_state(ante=6)
    state.deck_cards = [_card(rank, "Hearts", "m_steel", "Red") for rank in ("A", "K", "Q", "J", "T") * 11]
    prev_info = {
        "ante": 6,
        "pack_state_name": "STANDARD_PACK",
        "pack_card_details": (_pack_detail("9", "Clubs"),),
    }
    curr_info = {
        "ante": 6,
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "pack_claim",
        "action_index": 0,
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    assert reward == pytest.approx(0.0)


def test_default_reward_penalizes_planet_pack_skip_unless_fool_protects_high_value_tarot() -> None:
    state = _dummy_state(ante=3)
    prev_info = {
        "ante": 3,
        "pack_state_name": "PLANET_PACK",
        "consumable_keys": (),
        "last_tarot_planet": "",
    }
    curr_info = {
        "ante": 3,
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "pack_skip",
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    assert reward == pytest.approx(-PLANET_SKIP_PENALTY * REWARD_SCALE)


def test_default_reward_allows_planet_pack_skip_when_fool_protects_death() -> None:
    state = _dummy_state(ante=3)
    prev_info = {
        "ante": 3,
        "pack_state_name": "PLANET_PACK",
        "consumable_keys": ("c_fool",),
        "last_tarot_planet": "c_death",
    }
    curr_info = {
        "ante": 3,
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "pack_skip",
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    assert reward == pytest.approx(0.0)


def test_default_reward_penalizes_opening_planet_pack_when_fool_would_overwrite_temperance() -> None:
    state = _dummy_state(ante=3)
    prev_info = {
        "ante": 3,
        "consumable_keys": ("c_fool",),
        "last_tarot_planet": "c_temperance",
        "shop_item_details": ({"center_key": "p_celestial_normal_1", "pack_state_name": "PLANET_PACK"},),
    }
    curr_info = {
        "ante": 3,
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "shop_buy",
        "action_index": 0,
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    assert reward == pytest.approx(-PLANET_FOOL_OVERWRITE_PENALTY * REWARD_SCALE)


def test_default_reward_allows_opening_planet_pack_when_target_tarot_is_in_inventory() -> None:
    state = _dummy_state(ante=3)
    prev_info = {
        "ante": 3,
        "consumable_keys": ("c_fool", "c_temperance"),
        "last_tarot_planet": "c_temperance",
        "shop_item_details": ({"center_key": "p_celestial_normal_1", "pack_state_name": "PLANET_PACK"},),
    }
    curr_info = {
        "ante": 3,
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "shop_buy",
        "action_index": 0,
    }

    reward = default_reward(state, prev_info, curr_info, terminated=False, won=False)

    assert reward == pytest.approx(0.0)


def test_default_reward_stalled_terminal_is_harsher_than_true_loss() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1}
    ordinary_loss = default_reward(state, prev_info, {"stalled": False}, terminated=True, won=False)
    stalled_loss = default_reward(state, prev_info, {"stalled": True}, terminated=True, won=False)

    assert ordinary_loss == pytest.approx(pretraining_outcome_value(won=False, ante=1))
    assert stalled_loss == pytest.approx(pretraining_outcome_value(won=False, ante=1, stalled=True))
    assert stalled_loss < ordinary_loss


def test_default_reward_loss_penalty_scales_with_unfinished_antes() -> None:
    prev_info = {"ante": 1}

    early_loss = default_reward(_dummy_state(ante=1), prev_info, {"stalled": False}, terminated=True, won=False)
    later_loss = default_reward(_dummy_state(ante=6), prev_info, {"stalled": False}, terminated=True, won=False)

    # Dying earlier must strictly hurt more than dying deeper in the run.
    assert early_loss == pytest.approx(pretraining_outcome_value(won=False, ante=1))
    assert later_loss == pytest.approx(pretraining_outcome_value(won=False, ante=6))
    assert early_loss < later_loss


def test_default_reward_stalled_truncation_uses_stall_penalty() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1}

    reward = default_reward(state, prev_info, {"stalled": True}, terminated=False, won=False)

    assert reward == pytest.approx(pretraining_outcome_value(won=False, ante=1, stalled=True))


def test_pretraining_outcome_value_matches_supervised_fallback_scale() -> None:
    assert pretraining_outcome_value(won=True, ante=8) == pytest.approx(PRETRAIN_WIN_VALUE)
    assert pretraining_outcome_value(won=False, ante=5) == pytest.approx(-5.0)
    assert pretraining_outcome_value(won=False, ante=5, stalled=True) == pytest.approx(
        -5.0 - PRETRAIN_STALL_EXTRA_PENALTY
    )


def test_sparse_config_disables_hand_candidate_rewards() -> None:
    state = _dummy_state()
    prev = {"ante": 1, "round_score": 0, "blind_target": 800}
    curr = {
        "round_score": 100,
        "blind_target": 800,
        "action_type": "play_subset",
        "progress_made": True,
        "hand_play_top1": True,
        "hand_play_candidate_value_ratio": 1.0,
        "hand_play_not_in_candidates": False,
    }
    default = default_reward_components(state, prev, curr, terminated=False, won=False)
    sparse = default_reward_components(
        state, prev, curr, terminated=False, won=False, config=PPO_SPARSE_CONFIG
    )

    assert default["hand_top1_bonus"] > 0
    assert default["hand_subset_bonus"] > 0
    assert sparse["hand_top1_bonus"] == 0.0
    assert sparse["hand_subset_bonus"] == 0.0


def test_sparse_config_disables_planet_match_rewards() -> None:
    state = _dummy_state()
    prev = {"ante": 1, "round_score": 0, "blind_target": 800}
    curr = {
        "round_score": 100,
        "blind_target": 800,
        "action_type": "use_consumable_no_target",
        "progress_made": True,
        "planet_use_observed": True,
        "planet_use_main_hand_match": True,
        "planet_use_played_hand": True,
    }
    default = default_reward_components(state, prev, curr, terminated=False, won=False)
    sparse = default_reward_components(
        state, prev, curr, terminated=False, won=False, config=PPO_SPARSE_CONFIG
    )

    assert default["planet_match_bonus"] > 0
    assert sparse["planet_match_bonus"] == 0.0


def test_sparse_config_disables_shop_reroll_reward() -> None:
    state = _dummy_state()
    prev = {"ante": 1, "round_score": 0, "blind_target": 800}
    curr = {
        "round_score": 0,
        "blind_target": 800,
        "action_type": "shop_reroll",
        "progress_made": True,
    }
    default = default_reward_components(state, prev, curr, terminated=False, won=False)
    sparse = default_reward_components(
        state, prev, curr, terminated=False, won=False, config=PPO_SPARSE_CONFIG
    )

    assert default["shop_reroll_reward"] > 0
    assert sparse["shop_reroll_reward"] == 0.0


def test_sparse_config_keeps_terminal_reward() -> None:
    state = _dummy_state()
    prev = {"ante": 1, "round_score": 0, "blind_target": 800}
    curr = {"round_score": 0, "blind_target": 800, "ante": 3, "stalled": False}
    sparse = default_reward_components(
        state, prev, curr, terminated=True, won=False, config=PPO_SPARSE_CONFIG
    )

    assert sparse["terminal"] < 0
    expected_terminal = pretraining_outcome_value(won=False, ante=3) / REWARD_SCALE * REWARD_SCALE
    assert sparse["terminal"] == pytest.approx(expected_terminal)


def test_dense_reward_scale_shrinks_shaping_proportionally() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1, "round_score": 0, "blind_target": 800}
    curr_info = {
        "round_score": 100,
        "blind_target": 800,
        "action_type": "play_subset",
        "progress_made": True,
        "hand_play_top1": True,
        "hand_play_candidate_value_ratio": 1.0,
        "hand_play_not_in_candidates": False,
    }

    full = default_reward(state, prev_info, curr_info, terminated=False, won=False)
    quarter = default_reward(
        state,
        prev_info,
        curr_info,
        terminated=False,
        won=False,
        config=RewardConfig(dense_reward_scale=0.25),
    )

    assert full > 0.0
    # Dense shaping at 0.25 must be exactly a quarter of the full-scale shaping.
    assert quarter == pytest.approx(0.25 * full)


def test_dense_reward_scale_leaves_terminal_reward_unchanged() -> None:
    prev_info = {"ante": 3}
    loss_curr = {"ante": 3, "stalled": False}

    for won, state in (
        (False, _dummy_state(ante=3)),
        (True, _dummy_state(ante=8)),
    ):
        full = default_reward(
            state, prev_info, loss_curr, terminated=True, won=won
        )
        quarter = default_reward(
            state,
            prev_info,
            loss_curr,
            terminated=True,
            won=won,
            config=RewardConfig(dense_reward_scale=0.25),
        )
        # Terminal win/loss reward must be identical regardless of dense scale.
        assert quarter == pytest.approx(full)
        assert quarter == pytest.approx(
            pretraining_outcome_value(won=won, ante=3 if not won else 8, win_ante=state.win_ante)
        )


def test_dense_reward_scale_default_matches_unscaled_behavior() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1, "round_score": 0, "blind_target": 800}
    curr_info = {
        "round_score": 100,
        "blind_target": 800,
        "action_type": "shop_reroll",
        "progress_made": True,
    }

    baseline = default_reward(state, prev_info, curr_info, terminated=False, won=False)
    explicit_default = default_reward(
        state,
        prev_info,
        curr_info,
        terminated=False,
        won=False,
        config=RewardConfig(dense_reward_scale=1.0),
    )

    assert explicit_default == pytest.approx(baseline)


def test_config_flags_independent() -> None:
    state = _dummy_state()
    prev = {"ante": 1, "round_score": 0, "blind_target": 800}
    curr = {
        "round_score": 800,
        "blind_target": 800,
        "action_type": "play_subset",
        "progress_made": True,
        "hand_play_top1": True,
        "hand_play_candidate_value_ratio": 1.0,
        "hand_play_not_in_candidates": False,
        "blind_just_beaten": True,
        "hands_left": 3,
    }

    cfg = RewardConfig(enable_hand_candidate_rewards=False, enable_blind_clear_reward=True)
    result = default_reward_components(state, prev, curr, terminated=False, won=False, config=cfg)

    assert result["hand_top1_bonus"] == 0.0
    assert result["blind_clear"] > 0
