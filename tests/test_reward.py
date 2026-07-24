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
    PPO_SPARSE_CONFIG,
    PRESSURE_PROGRESS_SCALE,
    PRETRAIN_STALL_EXTRA_PENALTY,
    PRETRAIN_WIN_VALUE,
    REWARD_MODEL_VERSION,
    REWARD_SCALE,
    SCORE_PROGRESS_SCALE,
    SHOP_REROLL_REWARD,
    SHOP_SELL_PENALTY,
    STANDARD_OVERFULL_CARD_BASE_PENALTY,
    STANDARD_OVERFULL_CARD_EXPONENT,
    TAROT_SKIP_FIXING_PENALTY,
    RewardConfig,
    default_reward,
    default_reward_components,
    pretraining_outcome_value,
    reward_checkpoint_metadata,
    reward_config_fingerprint,
    reward_config_snapshot,
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


def test_reward_config_long_horizon_defaults() -> None:
    from pylatro_agent.reward import PPO_V2_REWARD_CONFIG

    assert RewardConfig().gamma == pytest.approx(0.997)
    assert PPO_V2_REWARD_CONFIG().gamma == pytest.approx(0.997)


def test_reward_config_fingerprint_is_canonical_and_stable() -> None:
    config = RewardConfig(enable_build_curve_rewards=True, dense_reward_scale=0.5)
    snapshot = reward_config_snapshot(config)
    reordered = dict(reversed(list(snapshot.items())))

    assert reward_config_fingerprint(config) == reward_config_fingerprint(snapshot)
    assert reward_config_fingerprint(snapshot) == reward_config_fingerprint(reordered)
    assert len(reward_config_fingerprint(config)) == 64


def test_reward_config_fingerprint_changes_with_config_or_model_version() -> None:
    baseline = RewardConfig()
    changed = RewardConfig(dense_reward_scale=0.5)

    assert reward_config_fingerprint(baseline) != reward_config_fingerprint(changed)
    assert reward_config_fingerprint(
        baseline,
        reward_model_version=REWARD_MODEL_VERSION,
    ) != reward_config_fingerprint(
        baseline,
        reward_model_version=REWARD_MODEL_VERSION + 1,
    )


def test_reward_checkpoint_metadata_contains_full_snapshot_version_and_fingerprint() -> None:
    config = RewardConfig(enable_build_curve_rewards=True, gamma=0.997)
    metadata = reward_checkpoint_metadata(config)

    assert metadata["reward_config"] == reward_config_snapshot(config)
    assert metadata["reward_model_version"] == REWARD_MODEL_VERSION
    assert metadata["reward_fingerprint"] == reward_config_fingerprint(config)


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


def test_planet_unmatched_penalty_weighted_by_play_share() -> None:
    state = _dummy_state()
    prev_info = {"ante": 1, "round_score": 0, "blind_target": 300}
    base_curr = {
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "pack_claim",
        "planet_claim_observed": True,
        "planet_claim_main_hand_match": False,
        "planet_claim_played_hand": True,
    }
    config = RewardConfig(planet_unmatched_claim_penalty_coeff=0.2)

    # Strong secondary hand (share 0.8): penalty shrinks to (1 - 0.8) = 20%.
    curr_info = dict(base_curr, planet_claim_play_share=0.8)
    components = default_reward_components(state, prev_info, curr_info, terminated=False, won=False, config=config)
    assert components["planet_unmatched_claim_penalty"] == pytest.approx(-0.2 * 0.2 * REWARD_SCALE)

    # Never-played hand (share 0.0): full penalty.
    curr_info = dict(base_curr, planet_claim_played_hand=False, planet_claim_play_share=0.0)
    components = default_reward_components(state, prev_info, curr_info, terminated=False, won=False, config=config)
    assert components["planet_unmatched_claim_penalty"] == pytest.approx(-0.2 * REWARD_SCALE)

    # Missing play_share key (older infos): original flat penalty.
    components = default_reward_components(
        state, prev_info, dict(base_curr), terminated=False, won=False, config=config
    )
    assert components["planet_unmatched_claim_penalty"] == pytest.approx(-0.2 * REWARD_SCALE)


def test_planet_unmatched_penalty_discounted_early_for_pivots() -> None:
    """Early unplayed-hand claims are pivot planning, not farming: the penalty
    ramps with ante progress toward win_ante, floored so it is never free."""
    from pylatro_agent.reward import PLANET_UNMATCHED_MIN_PROGRESS

    state = _dummy_state(win_ante=4)
    prev_info = {"ante": 1, "round_score": 0, "blind_target": 300}
    base_curr = {
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "pack_claim",
        "planet_claim_observed": True,
        "planet_claim_main_hand_match": False,
        "planet_claim_played_hand": False,
        "planet_claim_play_share": 0.0,
    }
    config = RewardConfig(planet_unmatched_claim_penalty_coeff=0.2)

    def penalty_at(ante: int) -> float:
        curr_info = dict(base_curr, ante=ante)
        components = default_reward_components(
            state, prev_info, curr_info, terminated=False, won=False, config=config
        )
        return components["planet_unmatched_claim_penalty"]

    # Ante 1: discounted to the floor, not free — a free window gets farmed.
    assert penalty_at(1) == pytest.approx(-0.2 * PLANET_UNMATCHED_MIN_PROGRESS * REWARD_SCALE)
    # Ante 2 of win_ante 4: one third of the way there (above the floor).
    assert penalty_at(2) == pytest.approx(-0.2 * (1 / 3) * REWARD_SCALE)
    # At (or past) win_ante the never-played hand pays in full.
    assert penalty_at(4) == pytest.approx(-0.2 * REWARD_SCALE)
    assert penalty_at(6) == pytest.approx(-0.2 * REWARD_SCALE)


def test_planet_unmatched_claim_exempt_when_best_available() -> None:
    """Claiming the best planet a matchless pack offers is not penalized:
    the pack is paid for and skipping wastes it."""
    state = _dummy_state()
    prev_info = {"ante": 1, "round_score": 0, "blind_target": 300}
    curr_info = {
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "pack_claim",
        "planet_claim_observed": True,
        "planet_claim_main_hand_match": False,
        "planet_claim_played_hand": False,
        "planet_claim_play_share": 0.0,
        "planet_claim_best_available": True,
        "ante": 1,
    }
    config = RewardConfig(planet_unmatched_claim_penalty_coeff=0.4)
    components = default_reward_components(
        state, prev_info, curr_info, terminated=False, won=False, config=config
    )
    assert components["planet_unmatched_claim_penalty"] == pytest.approx(0.0)


def test_planet_unmatched_claim_full_penalty_when_better_option_existed() -> None:
    """A better-aligned planet in the pack removes the pivot excuse: the claim
    pays the full share-weighted penalty at any ante (no early discount)."""
    state = _dummy_state(win_ante=4)
    prev_info = {"ante": 1, "round_score": 0, "blind_target": 300}
    curr_info = {
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "pack_claim",
        "planet_claim_observed": True,
        "planet_claim_main_hand_match": False,
        "planet_claim_played_hand": False,
        "planet_claim_play_share": 0.0,
        "planet_claim_best_available": False,
        "ante": 1,
    }
    config = RewardConfig(planet_unmatched_claim_penalty_coeff=0.4)
    components = default_reward_components(
        state, prev_info, curr_info, terminated=False, won=False, config=config
    )
    assert components["planet_unmatched_claim_penalty"] == pytest.approx(-0.4 * REWARD_SCALE)


def test_planet_unmatched_claim_best_available_still_penalized_with_protected_fool() -> None:
    """Best-available claims lose the exemption when a held Fool stores a
    protected target the claim would overwrite; skipping the pack is the
    free move in that spot (_should_penalize_planet_skip waives it)."""
    state = _dummy_state()
    prev_info = {
        "ante": 1,
        "round_score": 0,
        "blind_target": 300,
        "consumable_keys": ("c_fool",),
        "last_tarot_planet": "c_hermit",
    }
    curr_info = {
        "round_score": 0,
        "blind_target": 300,
        "progress_made": True,
        "action_type": "pack_claim",
        "planet_claim_observed": True,
        "planet_claim_main_hand_match": False,
        "planet_claim_played_hand": False,
        "planet_claim_play_share": 0.0,
        "planet_claim_best_available": True,
        "ante": 1,
    }
    config = RewardConfig(planet_unmatched_claim_penalty_coeff=0.4)
    components = default_reward_components(
        state, prev_info, curr_info, terminated=False, won=False, config=config
    )
    assert components["planet_unmatched_claim_penalty"] == pytest.approx(-0.4 * REWARD_SCALE)


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


# ──────────────────────────────────────────────────────────────────────────
# Phase 2: potential-based shaping + PPO_V2_REWARD_CONFIG
# ──────────────────────────────────────────────────────────────────────────


def test_state_potential_is_bounded_and_monotone():
    from pylatro_agent.reward import state_potential

    cfg = RewardConfig(enable_potential_shaping=True, potential_w_blind=0.5, potential_w_ante=2.0, potential_win_ante=8)

    # Early game: ante 1, small blind, no score.
    early = {"ante": 1, "round_score": 0, "blind_target": 300, "blind_on_deck": "small"}
    phi_early = state_potential(early, cfg)
    assert phi_early >= 0.0
    assert phi_early < 0.01  # ~0 at start

    # Mid-game progress: ante 3, big blind, half score.
    mid = {"ante": 3, "round_score": 150, "blind_target": 300, "blind_on_deck": "big"}
    phi_mid = state_potential(mid, cfg)
    assert phi_mid > phi_early  # monotone

    # Late game: ante 7, boss blind, full score.
    late = {"ante": 7, "round_score": 300, "blind_target": 300, "blind_on_deck": "boss"}
    phi_late = state_potential(late, cfg)
    assert phi_late > phi_mid
    # Bounded by w_blind + w_ante.
    assert phi_late <= 0.5 + 2.0 + 1e-6


def test_potential_shaping_telescopes_to_terminal():
    """For any trajectory, sum_t gamma^t * F(s_t, s_{t+1}) = -Phi(s_0).

    This is the defining property of potential-based shaping (Ng et al. 1999)
    with Phi(terminal) = 0. It catches off-by-one on gamma, missing terminal
    handling, and one-sided clipping.
    """
    from pylatro_agent.reward import potential_shaping_reward, state_potential

    gamma = 0.99
    cfg = RewardConfig(
        enable_potential_shaping=True,
        gamma=gamma,
        potential_w_blind=0.5,
        potential_w_ante=2.0,
        potential_win_ante=8,
    )

    # Simulated trajectory of info dicts with varying progress.
    infos = [
        {"ante": 1, "round_score": 0, "blind_target": 300, "blind_on_deck": "small"},
        {"ante": 1, "round_score": 100, "blind_target": 300, "blind_on_deck": "small"},
        {"ante": 1, "round_score": 300, "blind_target": 300, "blind_on_deck": "small"},
        {"ante": 2, "round_score": 0, "blind_target": 400, "blind_on_deck": "big"},
        {"ante": 3, "round_score": 0, "blind_target": 500, "blind_on_deck": "boss"},
    ]
    # Terminal: Phi(s') = 0.
    terminal_info = {"ante": 3, "round_score": 0, "blind_target": 500, "blind_on_deck": "boss"}
    terminal_info["_potential_terminal"] = True

    discounted_sum = 0.0
    for t in range(len(infos) - 1):
        f = potential_shaping_reward(infos[t], infos[t + 1], cfg)
        discounted_sum += (gamma ** t) * f
    # Final transition to terminal.
    f_terminal = potential_shaping_reward(infos[-1], terminal_info, cfg)
    discounted_sum += (gamma ** (len(infos) - 1)) * f_terminal

    # Expected: gamma^T * Phi(terminal) - Phi(s_0) = 0 - Phi(s_0).
    phi_0 = state_potential(infos[0], cfg)
    assert abs(discounted_sum - (-phi_0)) < 1e-5, (
        f"Telescoping failed: sum={discounted_sum}, expected={-phi_0}"
    )


def test_v2_reward_config_kills_prescriptive_and_uses_potential():
    from pylatro_agent.reward import PPO_V2_REWARD_CONFIG

    cfg = PPO_V2_REWARD_CONFIG(gamma=0.997, win_ante=4)
    assert cfg.enable_potential_shaping is True
    assert cfg.gamma == 0.997
    assert cfg.potential_win_ante == 4
    # All prescriptive components killed.
    assert cfg.enable_hand_candidate_rewards is False
    assert cfg.enable_planet_match_rewards is False
    assert cfg.enable_shop_strategy_rewards is False
    assert cfg.enable_economy_strategy_rewards is False
    assert cfg.enable_joker_context_rewards is False


def test_v2_planet_match_shaping_reenables_planet_component_only():
    from pylatro_agent.reward import PPO_V2_REWARD_CONFIG

    cfg = PPO_V2_REWARD_CONFIG(
        gamma=0.99,
        win_ante=4,
        planet_match_shaping=True,
        planet_unmatched_use_penalty_coeff=0.08,
        planet_unmatched_claim_penalty_coeff=0.08,
    )
    assert cfg.enable_planet_match_rewards is True
    assert cfg.planet_unmatched_use_penalty_coeff == 0.08
    assert cfg.planet_unmatched_claim_penalty_coeff == 0.08
    # Everything else stays killed; potential shaping stays on.
    assert cfg.enable_potential_shaping is True
    assert cfg.enable_hand_candidate_rewards is False
    assert cfg.enable_shop_strategy_rewards is False
    assert cfg.enable_economy_strategy_rewards is False
    assert cfg.enable_joker_context_rewards is False

    state = _dummy_state(ante=2, win_ante=4)
    prev = {"ante": 2, "round_score": 0, "blind_target": 400}
    matched = {
        "ante": 2,
        "round_score": 0,
        "blind_target": 400,
        "progress_made": True,
        "action_type": "use_consumable_no_target",
        "planet_use_observed": True,
        "planet_use_main_hand_match": True,
    }
    result = default_reward_components(state, prev, matched, terminated=False, won=False, config=cfg)
    assert result["planet_match_bonus"] == pytest.approx(PLANET_MATCH_BONUS * REWARD_SCALE)

    unmatched = dict(matched, planet_use_main_hand_match=False)
    result = default_reward_components(state, prev, unmatched, terminated=False, won=False, config=cfg)
    assert result["planet_match_bonus"] == pytest.approx(0.0)
    # Ante 2 of win_ante 4: the pivot discount pays 1/3 of the full penalty.
    assert result["planet_unmatched_use_penalty"] == pytest.approx(-0.08 * (1 / 3) * REWARD_SCALE)


def test_v2_default_keeps_planet_match_shaping_off():
    from pylatro_agent.reward import PPO_V2_REWARD_CONFIG

    cfg = PPO_V2_REWARD_CONFIG(gamma=0.99, win_ante=4)
    assert cfg.enable_planet_match_rewards is False
    assert cfg.planet_unmatched_use_penalty_coeff == 0.0
    assert cfg.planet_unmatched_claim_penalty_coeff == 0.0


def _v2_joker(key: str, **overrides) -> dict:
    joker = {
        "key": key,
        "name": key,
        "effect": "",
        "type": "",
        "config": {},
        "mult": 0.0,
        "t_mult": 0.0,
        "t_chips": 0.0,
        "x_mult": 1.0,
        "extra": None,
        "edition": {},
        "debuffed": False,
        "perishable": False,
        "perish_tally": None,
        "is_scaling": False,
        "is_scaling_xmult": False,
    }
    joker.update(overrides)
    return joker


def _v2_build_info(*, ante: int = 2, jokers=(), blind_target: int = 300, round_score: int = 0) -> dict:
    cards = tuple(
        {
            "rank": "8",
            "suit": "Clubs",
            "enhancement": "",
            "seal": "",
            "edition": "",
        }
        for _ in range(52)
    )
    return {
        "ante": ante,
        "round_score": round_score,
        "blind_target": blind_target,
        "blind_on_deck": "small",
        "hands_available": 4,
        "hand_size": 8,
        "hand_details": {
            "Pair": {"chips": 10, "mult": 2, "level": 1, "played": 8},
            "High Card": {"chips": 5, "mult": 1, "level": 1, "played": 1},
        },
        "hand_play_counts": {"Pair": 8, "High Card": 1},
        "hand_levels": {"Pair": 1, "High Card": 1},
        "deck_stats": {
            "size": len(cards),
            "cards": cards,
            "rank_counts": {"8": len(cards)},
            "suit_counts": {"Clubs": len(cards)},
            "enhancement_counts": {},
        },
        "joker_details": tuple(jokers),
    }


def test_v2_build_curve_compatibility_switch_enables_build_potential():
    from pylatro_agent.reward import PPO_V2_REWARD_CONFIG, state_potential_breakdown

    info = _v2_build_info(jokers=(_v2_joker("j_hologram", x_mult=1.75),))
    disabled = state_potential_breakdown(
        info,
        PPO_V2_REWARD_CONFIG(gamma=0.99, win_ante=8, build_curve_shaping=False),
    )
    enabled = state_potential_breakdown(
        info,
        PPO_V2_REWARD_CONFIG(gamma=0.99, win_ante=8, build_curve_shaping=True),
    )

    assert disabled["realized_build_quality"] == pytest.approx(0.0)
    assert disabled["scaling_option_value"] == pytest.approx(0.0)
    assert disabled["readiness"] == pytest.approx(0.0)
    assert enabled["realized_build_quality"] > 0.0
    assert enabled["scaling_option_value"] > 0.0
    assert enabled["readiness"] > 0.0


def test_legacy_contextual_build_delta_applies_to_pack_claims():
    state = _dummy_state(ante=2, win_ante=8)
    prev = _v2_build_info(jokers=(_v2_joker("j_hologram", x_mult=1.0),))
    curr = _v2_build_info(jokers=(_v2_joker("j_hologram", x_mult=1.5),))
    curr.update({"action_type": "pack_claim", "progress_made": True})

    result = default_reward_components(
        state,
        prev,
        curr,
        terminated=False,
        won=False,
        config=RewardConfig(),
    )

    assert result["shop_engine_delta"] > 0.0
    assert result["shop_purchase_value"] == 0.0


def test_legacy_build_curve_flag_keeps_static_component_behavior():
    config = RewardConfig(enable_build_curve_rewards=True)
    state = _dummy_state(ante=2, win_ante=8)
    chip_joker = {"key": "j_pair_chips", "t_chips": 30.0, "x_mult": 1.0}
    prev = {"ante": 2, "round_score": 0, "blind_target": 400, "joker_details": ()}
    curr = {
        "ante": 2,
        "round_score": 0,
        "blind_target": 400,
        "joker_details": (chip_joker,),
        "progress_made": True,
    }

    result = default_reward_components(state, prev, curr, terminated=False, won=False, config=config)
    assert result["build_curve_bonus"] == pytest.approx(config.build_curve_coeff * REWARD_SCALE)


def test_v2_build_potential_breakdown_is_bounded():
    from pylatro_agent.reward import PPO_V2_REWARD_CONFIG, state_potential_breakdown

    cfg = PPO_V2_REWARD_CONFIG(gamma=0.99, win_ante=8, build_curve_shaping=True)
    info = _v2_build_info(
        ante=2,
        blind_target=100,
        round_score=100,
        jokers=(_v2_joker("j_hologram", name="Hologram", x_mult=100.0),),
    )
    breakdown = state_potential_breakdown(info, cfg)
    build_total = (
        breakdown["realized_build_quality"]
        + breakdown["scaling_option_value"]
        + breakdown["readiness"]
    )

    assert set(breakdown) == {
        "blind_progress",
        "ante_progress",
        "realized_build_quality",
        "scaling_option_value",
        "readiness",
        "total",
    }
    assert all(value >= 0.0 for value in breakdown.values())
    assert build_total <= cfg.potential_build_cap + 1e-9
    assert breakdown["scaling_option_value"] <= cfg.potential_w_scaling_option + 1e-9
    assert breakdown["total"] <= (
        cfg.potential_w_blind + cfg.potential_w_ante + cfg.potential_build_cap + 1e-9
    )


def test_v2_hologram_option_acquisition_and_live_scaling_raise_potential():
    from pylatro_agent.reward import (
        PPO_V2_REWARD_CONFIG,
        potential_shaping_reward,
        state_potential_breakdown,
    )

    cfg = PPO_V2_REWARD_CONFIG(gamma=0.99, win_ante=8, build_curve_shaping=True)
    empty = _v2_build_info()
    fresh = _v2_build_info(jokers=(_v2_joker("j_hologram", name="Hologram", x_mult=1.0),))
    scaled = _v2_build_info(jokers=(_v2_joker("j_hologram", name="Hologram", x_mult=1.25),))
    visible_standard = dict(
        fresh,
        shop_cards=({"key": "p_standard_normal_1", "name": "Standard Pack", "set": "Booster"},),
    )

    empty_breakdown = state_potential_breakdown(empty, cfg)
    fresh_breakdown = state_potential_breakdown(fresh, cfg)
    scaled_breakdown = state_potential_breakdown(scaled, cfg)
    visible_breakdown = state_potential_breakdown(visible_standard, cfg)

    assert fresh_breakdown["scaling_option_value"] > 0.0
    assert visible_breakdown["scaling_option_value"] > fresh_breakdown["scaling_option_value"]
    assert fresh_breakdown["realized_build_quality"] == pytest.approx(
        empty_breakdown["realized_build_quality"]
    )
    assert potential_shaping_reward(empty, fresh, cfg) > 0.0
    assert scaled_breakdown["realized_build_quality"] > fresh_breakdown["realized_build_quality"]
    assert scaled_breakdown["readiness"] > fresh_breakdown["readiness"]
    assert potential_shaping_reward(fresh, scaled, cfg) > 0.0

    state = _dummy_state(ante=2, win_ante=8)
    result = default_reward_components(state, empty, fresh, terminated=False, won=False, config=cfg)
    assert result["build_curve_bonus"] == pytest.approx(0.0)
    assert result["potential_shaping"] > 0.0


def test_v2_early_chip_curve_boundaries_and_ante_fade():
    from pylatro_agent.reward import (
        PPO_V2_REWARD_CONFIG,
        _early_chip_marginal_multiplier,
        state_potential_breakdown,
    )

    assert _early_chip_marginal_multiplier(1.0, 2) == pytest.approx(1.0)
    assert _early_chip_marginal_multiplier(1.25, 2) == pytest.approx(1.5)
    assert _early_chip_marginal_multiplier(1.5, 2) == pytest.approx(2.0)
    assert _early_chip_marginal_multiplier(2.0, 3) == pytest.approx(2.0)
    assert _early_chip_marginal_multiplier(2.0, 4) == pytest.approx(1.5)
    assert _early_chip_marginal_multiplier(2.0, 5) == pytest.approx(1.0)

    cfg = PPO_V2_REWARD_CONFIG(gamma=0.99, win_ante=8, build_curve_shaping=True)
    chip_joker = _v2_joker("j_pair_chips", type="Pair", t_chips=50.0)
    early = state_potential_breakdown(_v2_build_info(ante=2, jokers=(chip_joker,)), cfg)
    ante4 = state_potential_breakdown(_v2_build_info(ante=4, jokers=(chip_joker,)), cfg)
    late = state_potential_breakdown(_v2_build_info(ante=5, jokers=(chip_joker,)), cfg)
    assert early["realized_build_quality"] > ante4["realized_build_quality"]
    assert ante4["realized_build_quality"] > late["realized_build_quality"]


def test_v2_same_state_has_gamma_correct_noop_cost():
    from pylatro_agent.reward import PPO_V2_REWARD_CONFIG, potential_shaping_reward, state_potential

    cfg = PPO_V2_REWARD_CONFIG(gamma=0.99, win_ante=8, build_curve_shaping=True)
    info = _v2_build_info(jokers=(_v2_joker("j_hologram", x_mult=1.25),))
    phi = state_potential(info, cfg)
    reward = potential_shaping_reward(info, info, cfg)

    assert phi > 0.0
    assert reward == pytest.approx((cfg.gamma - 1.0) * phi)
    assert reward < 0.0


def test_v2_hologram_sell_and_reset_reverse_build_value():
    from pylatro_agent.reward import PPO_V2_REWARD_CONFIG, potential_shaping_reward

    cfg = PPO_V2_REWARD_CONFIG(gamma=0.99, win_ante=8, build_curve_shaping=True)
    scaled = _v2_build_info(jokers=(_v2_joker("j_hologram", x_mult=1.75),))
    fresh = _v2_build_info(jokers=(_v2_joker("j_hologram", x_mult=1.0),))
    sold = _v2_build_info()

    assert potential_shaping_reward(scaled, fresh, cfg) < 0.0
    assert potential_shaping_reward(scaled, sold, cfg) < 0.0


def test_v2_terminal_transition_repays_full_potential():
    from pylatro_agent.reward import PPO_V2_REWARD_CONFIG, potential_shaping_reward, state_potential

    cfg = PPO_V2_REWARD_CONFIG(gamma=0.99, win_ante=8, build_curve_shaping=True)
    prev = _v2_build_info(jokers=(_v2_joker("j_hologram", x_mult=1.75),))
    terminal = dict(_v2_build_info(round_score=300), _potential_terminal=True)

    assert potential_shaping_reward(prev, terminal, cfg) == pytest.approx(-state_potential(prev, cfg))


def test_v2_terminal_win_ante_override_preserves_build_config_fields():
    import dataclasses

    from pylatro_agent.reward import PPO_V2_REWARD_CONFIG, state_potential

    cfg = PPO_V2_REWARD_CONFIG(gamma=0.99, win_ante=8, build_curve_shaping=True)
    cfg = dataclasses.replace(cfg, potential_w_build_quality=0.31, potential_w_scaling_option=0.17)
    state = _dummy_state(ante=2, win_ante=5)
    prev = _v2_build_info(ante=2, jokers=(_v2_joker("j_hologram", x_mult=1.75),))
    curr = _v2_build_info(ante=2)

    result = default_reward_components(state, prev, curr, terminated=True, won=False, config=cfg)
    effective = dataclasses.replace(cfg, potential_win_ante=state.win_ante)
    assert result["potential_shaping"] == pytest.approx(-state_potential(prev, effective))


def test_v2_discounted_buy_sell_cycle_cannot_farm_build_potential():
    from pylatro_agent.reward import PPO_V2_REWARD_CONFIG, potential_shaping_reward, state_potential

    gamma = 0.99
    cfg = PPO_V2_REWARD_CONFIG(gamma=gamma, win_ante=8, build_curve_shaping=True)
    empty = _v2_build_info()
    owned = _v2_build_info(jokers=(_v2_joker("j_hologram", x_mult=1.0),))

    buy = potential_shaping_reward(empty, owned, cfg)
    sell = potential_shaping_reward(owned, empty, cfg)
    discounted_cycle = buy + gamma * sell

    assert discounted_cycle == pytest.approx((gamma**2 - 1.0) * state_potential(empty, cfg))
    assert discounted_cycle < 0.0


def test_v2_missing_build_context_falls_back_to_progress_only():
    from pylatro_agent.reward import PPO_V2_REWARD_CONFIG, state_potential_breakdown

    cfg = PPO_V2_REWARD_CONFIG(gamma=0.99, win_ante=8, build_curve_shaping=True)
    sparse = {
        "ante": 2,
        "round_score": 100,
        "blind_target": 400,
        "blind_on_deck": "big",
        "joker_details": (_v2_joker("j_hologram", x_mult=2.0),),
    }
    breakdown = state_potential_breakdown(sparse, cfg)

    assert breakdown["realized_build_quality"] == pytest.approx(0.0)
    assert breakdown["scaling_option_value"] == pytest.approx(0.0)
    assert breakdown["readiness"] == pytest.approx(0.0)
    assert breakdown["total"] == pytest.approx(
        breakdown["blind_progress"] + breakdown["ante_progress"]
    )


def test_v2_planet_played_hand_bonus_scales_with_play_share():
    from pylatro_agent.reward import PLANET_PLAYED_HAND_BONUS, PPO_V2_REWARD_CONFIG

    cfg = PPO_V2_REWARD_CONFIG(gamma=0.99, win_ante=4, planet_match_shaping=True)
    state = _dummy_state(ante=2, win_ante=4)
    prev = {"ante": 2, "round_score": 0, "blind_target": 400}

    def use(play_share=None):
        curr = {
            "ante": 2,
            "round_score": 0,
            "blind_target": 400,
            "progress_made": True,
            "action_type": "use_consumable_no_target",
            "planet_use_observed": True,
            "planet_use_played_hand": True,
            "planet_use_main_hand_match": False,
        }
        if play_share is not None:
            curr["planet_use_play_share"] = play_share
        result = default_reward_components(state, prev, curr, terminated=False, won=False, config=cfg)
        return result["planet_played_hand_bonus"]

    # The Pluto exploit: High Card was "played" once against a Two Pair
    # workhorse — the bonus shrinks to the play share instead of paying full.
    assert use(play_share=0.1) == pytest.approx(0.1 * PLANET_PLAYED_HAND_BONUS * REWARD_SCALE)
    assert use(play_share=1.0) == pytest.approx(PLANET_PLAYED_HAND_BONUS * REWARD_SCALE)
    # Older infos without the key keep the pre-fix behavior (full bonus).
    assert use(play_share=None) == pytest.approx(PLANET_PLAYED_HAND_BONUS * REWARD_SCALE)


def test_v2_reward_terminal_dominates_return():
    from pylatro_agent.reward import PPO_V2_REWARD_CONFIG, V2_WIN_VALUE

    cfg = PPO_V2_REWARD_CONFIG(gamma=0.99, win_ante=4)
    state = _dummy_state(ante=2, win_ante=4)

    # A win at ante 4 should have terminal = V2_WIN_VALUE (10.0).
    prev = {"ante": 4, "round_score": 0, "blind_target": 300}
    curr = {"ante": 4, "round_score": 0, "blind_target": 300}
    result = default_reward_components(state, prev, curr, terminated=True, won=True, config=cfg)
    assert result["terminal"] == pytest.approx(V2_WIN_VALUE)

    # The potential shaping on a terminal step should also be present (the
    # final F(s, terminal) transition).
    assert "potential_shaping" in result


def test_v2_early_death_surcharge_steepens_bottom_of_loss_curve():
    from pylatro_agent.reward import v2_outcome_value

    a1 = v2_outcome_value(won=False, ante=1, win_ante=5)
    a2 = v2_outcome_value(won=False, ante=2, win_ante=5)
    a3 = v2_outcome_value(won=False, ante=3, win_ante=5)
    a5 = v2_outcome_value(won=False, ante=5, win_ante=5)

    assert a1 == pytest.approx(-6.5)  # -5 + 0.5*1 - 2.0
    assert a2 == pytest.approx(-5.0)  # -5 + 0.5*2 - 1.0
    assert a3 == pytest.approx(-3.5)  # ante 3+ unchanged
    assert a5 == pytest.approx(-2.5)
    # Still strictly increasing in ante: no inversion from the surcharge.
    assert a1 < a2 < a3 < a5
    # Wins and stall ordering untouched.
    assert v2_outcome_value(won=True, ante=1, win_ante=5) == pytest.approx(10.0)
    assert v2_outcome_value(won=False, ante=1, win_ante=5, stalled=True) < a1


def test_v2_early_death_surcharge_flows_through_terminal_component():
    from pylatro_agent.reward import PPO_V2_REWARD_CONFIG, v2_outcome_value

    cfg = PPO_V2_REWARD_CONFIG(gamma=0.99, win_ante=5)
    state = _dummy_state(ante=1, win_ante=5)
    prev = {"ante": 1, "round_score": 0, "blind_target": 300}
    curr = {"ante": 1, "round_score": 100, "blind_target": 300}

    result = default_reward_components(state, prev, curr, terminated=True, won=False, config=cfg)
    assert result["terminal"] == pytest.approx(v2_outcome_value(won=False, ante=1, win_ante=5))
    assert result["terminal"] == pytest.approx(-6.5)


def test_v2_reward_no_prescriptive_components_on_dense_step():
    from pylatro_agent.reward import PPO_V2_REWARD_CONFIG

    cfg = PPO_V2_REWARD_CONFIG(gamma=0.99, win_ante=8)
    state = _dummy_state(ante=2)
    prev = {"ante": 2, "round_score": 0, "blind_target": 400, "blind_on_deck": "small"}
    curr = {
        "ante": 2,
        "round_score": 200,
        "blind_target": 400,
        "blind_on_deck": "small",
        "action_type": "play_subset",
        "progress_made": True,
        "hand_play_top1": True,
        "hand_play_candidate_value_ratio": 1.0,
        "hand_play_not_in_candidates": False,
    }

    result = default_reward_components(state, prev, curr, terminated=False, won=False, config=cfg)
    # All prescriptive components must be zero.
    assert result["hand_top1_bonus"] == 0.0
    assert result["hand_subset_bonus"] == 0.0
    assert result["score_progress"] == 0.0
    assert result["blind_clear"] == 0.0
    # Potential shaping must be the only dense signal (plus possibly idle_penalty).
    assert abs(result["potential_shaping"]) > 0 or abs(result["idle_penalty"]) > 0


def test_v2_applies_dense_and_group_scales_only_to_idle_and_planet_shaping():
    from pylatro_agent.reward import PPO_V2_REWARD_CONFIG

    state = _dummy_state(ante=2, win_ante=4)
    prev = {"ante": 2, "round_score": 0, "blind_target": 400, "blind_on_deck": "small"}
    curr = {
        "ante": 2,
        "round_score": 0,
        "blind_target": 400,
        "blind_on_deck": "small",
        "progress_made": False,
        "steps_since_progress": 1,
        "planet_use_observed": True,
        "planet_use_main_hand_match": True,
    }
    base = PPO_V2_REWARD_CONFIG(
        gamma=0.99,
        win_ante=4,
        planet_match_shaping=True,
    )
    scaled = PPO_V2_REWARD_CONFIG(
        gamma=0.99,
        win_ante=4,
        planet_match_shaping=True,
        dense_reward_scale=0.5,
        progression_reward_scale=0.25,
        consumable_reward_scale=0.2,
    )

    base_result = default_reward_components(
        state, prev, curr, terminated=False, won=False, config=base
    )
    scaled_result = default_reward_components(
        state, prev, curr, terminated=False, won=False, config=scaled
    )

    assert scaled_result["potential_shaping"] == pytest.approx(
        base_result["potential_shaping"]
    )
    assert scaled_result["idle_penalty"] == pytest.approx(
        base_result["idle_penalty"] * 0.5 * 0.25
    )
    assert scaled_result["planet_match_bonus"] == pytest.approx(
        base_result["planet_match_bonus"] * 0.5 * 0.2
    )


def test_v2_potential_shaping_invariant_to_dense_reward_scale():
    """The potential term must carry the same scale on dense and terminal steps.

    potential_shaping is deliberately scaled by REWARD_SCALE only (never
    dense_reward_scale): scaling the potential differently across steps leaves a
    per-step residual that breaks the telescoping sum and with it the
    policy-invariance guarantee of potential-based shaping.
    """
    import dataclasses

    from pylatro_agent.reward import PPO_V2_REWARD_CONFIG

    state = _dummy_state(ante=2)
    prev = {"ante": 2, "round_score": 0, "blind_target": 400, "blind_on_deck": "small"}
    curr = {"ante": 2, "round_score": 200, "blind_target": 400, "blind_on_deck": "small", "progress_made": True}

    cfg = PPO_V2_REWARD_CONFIG(gamma=0.99, win_ante=8)
    cfg_scaled = dataclasses.replace(cfg, dense_reward_scale=2.0)

    dense = default_reward_components(state, prev, curr, terminated=False, won=False, config=cfg)
    dense_scaled = default_reward_components(state, prev, curr, terminated=False, won=False, config=cfg_scaled)
    assert dense["potential_shaping"] != 0.0
    assert dense_scaled["potential_shaping"] == pytest.approx(dense["potential_shaping"])

    terminal = default_reward_components(state, prev, curr, terminated=True, won=False, config=cfg)
    terminal_scaled = default_reward_components(state, prev, curr, terminated=True, won=False, config=cfg_scaled)
    assert terminal_scaled["potential_shaping"] == pytest.approx(terminal["potential_shaping"])
