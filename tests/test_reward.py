from __future__ import annotations

from types import SimpleNamespace

import pytest

from pylatro_agent import reward as reward_module
from pylatro_agent.build_value import BuildValueEstimate, JokerMarginal, ScoreChannels
from pylatro_agent.reward import (
    ANTE1_CHIP_TEMPO_BONUS,
    ANTE_PROGRESS_VALUE,
    DEFAULT_REWARD_CONFIG,
    EARLY_DEATH_PENALTIES,
    IDLE_PENALTY_CAP,
    JOKER_MOVE_NON_IMPROVING_PENALTY,
    JOKER_MOVE_PENALTY_CAP,
    JOKER_MOVE_REPEAT_PENALTY,
    LOSS_BASE,
    PLANET_MATCH_BONUS,
    PLANET_PLAYED_HAND_BONUS,
    REWARD_MODEL_VERSION,
    STALL_EXTRA_PENALTY,
    WIN_VALUE,
    RewardConfig,
    default_reward,
    default_reward_components,
    outcome_value,
    potential_shaping_reward,
    reward_checkpoint_metadata,
    reward_config_fingerprint,
    reward_config_snapshot,
    state_potential,
    state_potential_breakdown,
    survival_shaping_reward,
)


def _state(*, ante: int = 1, win_ante: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        round_resets=SimpleNamespace(ante=ante),
        win_ante=win_ante,
    )


def _info(
    *,
    ante: int = 1,
    blind_on_deck: str = "Small",
    round_score: float = 0.0,
    blind_target: float = 100.0,
    progress_made: bool = True,
    steps_since_progress: int = 0,
    **extra,
) -> dict:
    return {
        "ante": ante,
        "blind_on_deck": blind_on_deck,
        "round_score": round_score,
        "blind_target": blind_target,
        "progress_made": progress_made,
        "steps_since_progress": steps_since_progress,
        **extra,
    }


def _estimate(
    *,
    score: float = 200.0,
    baseline: float = 100.0,
    required: float = 150.0,
    marginal_ratio: float = 1.5,
    chips: float = 25.0,
) -> BuildValueEstimate:
    marginal = JokerMarginal(
        index=0,
        key="j_runner",
        score_with=score,
        score_without=score / marginal_ratio,
        score_ratio=marginal_ratio,
        modeled_effects=("chips",),
        modeled_effect_fraction=1.0,
        channels=ScoreChannels(chips=chips),
    )
    return BuildValueEstimate(
        representative_hand_type="Straight",
        representative_score_per_hand=score,
        no_joker_baseline_score=baseline,
        required_score_per_hand=required,
        readiness_ratio=score / required,
        joker_marginal_score_ratios=(marginal_ratio,),
        joker_marginals=(marginal,),
        modeled_effects=("chips",),
        unmodeled_effects=(),
        channels=ScoreChannels(chips=chips),
    )


def test_default_config_is_the_single_potential_reward() -> None:
    config = RewardConfig()

    assert config == DEFAULT_REWARD_CONFIG
    assert config.gamma == pytest.approx(0.997)
    assert config.enable_planet_match_rewards is False
    assert config.enable_score_build_potential is False
    assert not hasattr(config, "reward_version")
    assert config.strategic_event_reward_scale == 1.0
    assert config.potential_w_tarot_option == 0.0
    assert config.potential_w_planet_option == 0.0
    assert REWARD_MODEL_VERSION == 14


def test_attributable_strategic_event_rewards_are_bounded_and_dense_scale_independent() -> None:
    prev = _info(ante=3, dollars=4)
    curr = _info(
        ante=3,
        dollars=17,
        strategic_attributable_cash_payout=10,
        strategic_gold_created_tarot=1,
        strategic_gold_created_midas=0,
        strategic_purple_tarots_generated=1,
        strategic_blue_planets_generated=1,
        strategic_blue_seals_claimed=1,
        strategic_purple_seals_claimed=1,
        strategic_tarot_fix_reward=0.12,
    )
    config = RewardConfig(dense_reward_scale=0.0)

    components = default_reward_components(_state(ante=3), prev, curr, False, False, config)

    assert components["strategic_cash_payout"] == pytest.approx(0.50)
    assert components["strategic_gold_creation"] == pytest.approx(0.075)
    assert components["strategic_purple_generation"] == pytest.approx(0.12)
    assert components["strategic_blue_generation"] == pytest.approx(0.12)
    assert components["strategic_seal_claim"] == pytest.approx(0.06)
    assert components["strategic_tarot_fix"] == pytest.approx(0.12)
    positive = sum(
        max(components[name], 0.0)
        for name in reward_module.REWARD_COMPONENT_NAMES
        if name.startswith("strategic_")
    )
    assert positive == pytest.approx(0.995)


def test_strategic_cash_requires_attributable_engine_payout() -> None:
    prev = _info(dollars=4)
    unrelated_dollar_gain = _info(dollars=20)
    purple_success = _info(strategic_purple_tarots_generated=1)

    unrelated = default_reward_components(_state(), prev, unrelated_dollar_gain, False, False)
    purple = default_reward_components(_state(), prev, purple_success, False, False)

    assert unrelated["strategic_cash_payout"] == 0.0
    assert purple["strategic_purple_generation"] == pytest.approx(0.12)
    assert purple["total"] > 0.0


@pytest.mark.parametrize(("count", "expected"), ((1, 0.15), (4, 0.60), (8, 1.0)))
def test_held_gold_reward_uses_three_dollar_value_with_global_cap(count: int, expected: float) -> None:
    curr = _info(
        strategic_held_gold_count=count,
        strategic_held_gold_payout=3 * count,
        strategic_attributable_cash_payout=0,
    )

    components = default_reward_components(_state(), _info(), curr, False, False)

    assert components["strategic_held_gold_payout"] == pytest.approx(expected)
    assert components["strategic_cash_payout"] == 0.0


def test_priority_seal_claim_reward_has_no_late_ante_runway_discount() -> None:
    curr = _info(
        ante=8,
        strategic_blue_seals_claimed=1,
        strategic_purple_seals_claimed=1,
    )

    components = default_reward_components(
        _state(ante=8, win_ante=8),
        _info(ante=8),
        curr,
        False,
        False,
    )

    assert components["strategic_seal_claim"] == pytest.approx(0.06)


def test_negative_contextual_tarot_fix_reward_survives_positive_cap() -> None:
    curr = _info(
        strategic_attributable_cash_payout=100,
        strategic_gold_created_tarot=10,
        strategic_purple_tarots_generated=10,
        strategic_tarot_fix_reward=-0.5,
    )

    components = default_reward_components(_state(), _info(), curr, False, False)

    assert components["strategic_tarot_fix"] == pytest.approx(-0.20)
    positive = sum(
        max(components[name], 0.0)
        for name in reward_module.REWARD_COMPONENT_NAMES
        if name.startswith("strategic_")
    )
    assert positive == pytest.approx(1.0)


def test_reward_config_fingerprint_is_canonical_and_versioned() -> None:
    config = RewardConfig(dense_reward_scale=0.5)
    snapshot = reward_config_snapshot(config)
    reordered = dict(reversed(tuple(snapshot.items())))

    assert reward_config_fingerprint(config) == reward_config_fingerprint(reordered)
    assert len(reward_config_fingerprint(config)) == 64
    assert reward_config_fingerprint(config) != reward_config_fingerprint(
        config,
        reward_model_version=REWARD_MODEL_VERSION + 1,
    )


def test_checkpoint_metadata_contains_complete_reward_identity() -> None:
    config = RewardConfig(enable_score_build_potential=True)
    metadata = reward_checkpoint_metadata(config)

    assert metadata == {
        "reward_config": reward_config_snapshot(config),
        "reward_model_version": REWARD_MODEL_VERSION,
        "reward_fingerprint": reward_config_fingerprint(config),
    }


def test_danger_reroll_signal_is_positive_only_and_skips_visible_rescue(monkeypatch) -> None:
    monkeypatch.setattr(
        reward_module,
        "estimate_clear_risk",
        lambda info: SimpleNamespace(immediate_death_probability=0.80),
    )
    monkeypatch.setattr(reward_module, "best_confident_joker_rescue", lambda info: None)
    components = {"danger_reroll_bonus": 0.0}

    reward_module._apply_danger_reroll_reward({}, {"action_type": "shop_reroll"}, components)

    assert components["danger_reroll_bonus"] > 0.0

    components["danger_reroll_bonus"] = 0.0
    monkeypatch.setattr(
        reward_module,
        "best_confident_joker_rescue",
        lambda info: SimpleNamespace(clear_probability_delta=0.20),
    )
    reward_module._apply_danger_reroll_reward({}, {"action_type": "shop_reroll"}, components)
    assert components["danger_reroll_bonus"] == 0.0

    reward_module._apply_danger_reroll_reward({}, {"action_type": "shop_leave"}, components)
    assert components["danger_reroll_bonus"] == 0.0


@pytest.mark.parametrize(
    ("ante", "expected"),
    [
        (1, LOSS_BASE + ANTE_PROGRESS_VALUE - EARLY_DEATH_PENALTIES[1]),
        (2, LOSS_BASE + 2 * ANTE_PROGRESS_VALUE - EARLY_DEATH_PENALTIES[2]),
        (3, LOSS_BASE + 3 * ANTE_PROGRESS_VALUE),
        (8, LOSS_BASE + 8 * ANTE_PROGRESS_VALUE),
        (99, LOSS_BASE + 8 * ANTE_PROGRESS_VALUE),
    ],
)
def test_outcome_value_loss_curve(ante: int, expected: float) -> None:
    assert outcome_value(won=False, ante=ante) == pytest.approx(expected)


def test_outcome_value_win_and_stall() -> None:
    assert outcome_value(won=True, ante=1) == WIN_VALUE
    ordinary = outcome_value(won=False, ante=4)
    stalled = outcome_value(won=False, ante=4, stalled=True)
    assert stalled == pytest.approx(ordinary - STALL_EXTRA_PENALTY)


def test_state_potential_tracks_blind_and_ante_progress() -> None:
    config = RewardConfig(gamma=0.99, potential_win_ante=8)
    start = state_potential(_info(), config)
    half_blind = state_potential(_info(round_score=50), config)
    later_blind = state_potential(_info(ante=2, blind_on_deck="Big"), config)

    assert start == 0.0
    assert half_blind > start
    assert later_blind > half_blind


@pytest.mark.parametrize("sub_phase", ["shop", "booster_pack", "blind_select"])
def test_state_potential_does_not_carry_completed_score_into_next_blind(sub_phase: str) -> None:
    config = RewardConfig()
    between_blinds = _info(
        blind_on_deck="Boss",
        round_score=552,
        blind_target=600,
        sub_phase=sub_phase,
        in_shop=sub_phase == "shop",
    )
    no_stale_score = {**between_blinds, "round_score": 0}

    assert state_potential(between_blinds, config) == pytest.approx(state_potential(no_stale_score, config))


def test_potential_shaping_uses_gamma_and_zero_terminal_potential() -> None:
    config = RewardConfig(gamma=0.9)
    prev = _info(round_score=20)
    curr = _info(round_score=60)
    terminal = {**curr, "_potential_terminal": True}

    assert potential_shaping_reward(prev, curr, config) == pytest.approx(
        config.gamma * state_potential(curr, config) - state_potential(prev, config)
    )
    assert potential_shaping_reward(prev, terminal, config) == pytest.approx(-state_potential(prev, config))


def test_discounted_potential_rewards_telescope() -> None:
    config = RewardConfig(gamma=0.9)
    states = [
        _info(round_score=0),
        _info(round_score=30),
        _info(round_score=80),
        _info(round_score=100, _potential_terminal=True),
    ]
    rewards = [potential_shaping_reward(states[index], states[index + 1], config) for index in range(len(states) - 1)]
    discounted = sum(config.gamma**index * reward for index, reward in enumerate(rewards))

    assert discounted == pytest.approx(-state_potential(states[0], config))


def test_default_reward_includes_progress_potential() -> None:
    state = _state()
    prev = _info(round_score=0)
    curr = _info(round_score=50)

    components = default_reward_components(state, prev, curr, False, False)

    assert components["potential_shaping"] > 0
    assert components["terminal"] == 0
    assert default_reward(state, prev, curr, False, False) == pytest.approx(components["total"])


def test_terminal_reward_combines_outcome_and_potential_repayment() -> None:
    state = _state(ante=3)
    prev = _info(ante=3, round_score=60)
    curr = _info(ante=3, round_score=60)

    components = default_reward_components(state, prev, curr, True, False)

    assert components["terminal"] == pytest.approx(outcome_value(won=False, ante=3))
    assert components["potential_shaping"] == pytest.approx(-state_potential(prev, DEFAULT_REWARD_CONFIG))


def test_idle_penalty_ramps_caps_and_scales() -> None:
    state = _state()
    prev = _info()
    early = _info(progress_made=False, steps_since_progress=1)
    late = _info(progress_made=False, steps_since_progress=10_000)
    base = default_reward_components(state, prev, early, False, False)
    capped = default_reward_components(state, prev, late, False, False)
    scaled = default_reward_components(
        state,
        prev,
        late,
        False,
        False,
        RewardConfig(dense_reward_scale=0.25),
    )

    assert base["idle_penalty"] < 0
    assert capped["idle_penalty"] == pytest.approx(-IDLE_PENALTY_CAP)
    assert scaled["idle_penalty"] == pytest.approx(capped["idle_penalty"] * 0.25)


def test_planet_shaping_is_opt_in() -> None:
    state = _state()
    prev = _info()
    curr = _info(
        planet_use_observed=True,
        planet_use_played_hand=True,
        planet_use_main_hand_match=True,
        planet_use_play_share=0.5,
    )

    disabled = default_reward_components(state, prev, curr, False, False)
    enabled = default_reward_components(
        state,
        prev,
        curr,
        False,
        False,
        RewardConfig(enable_planet_match_rewards=True),
    )

    assert disabled["planet_match_bonus"] == 0
    assert disabled["planet_played_hand_bonus"] == 0
    assert enabled["planet_match_bonus"] == pytest.approx(PLANET_MATCH_BONUS)
    assert enabled["planet_played_hand_bonus"] == pytest.approx(PLANET_PLAYED_HAND_BONUS * 0.5)


def test_unmatched_planet_use_penalty_respects_play_share_and_scale() -> None:
    state = _state(ante=4, win_ante=8)
    prev = _info(ante=4)
    curr = _info(
        ante=4,
        planet_use_observed=True,
        planet_use_play_share=0.25,
    )
    config = RewardConfig(
        enable_planet_match_rewards=True,
        planet_unmatched_use_penalty_coeff=0.4,
        dense_reward_scale=0.5,
        consumable_reward_scale=0.25,
    )

    components = default_reward_components(state, prev, curr, False, False, config)

    unscaled = -0.4 * 0.75 * ((4 - 1) / (8 - 1))
    assert components["planet_unmatched_use_penalty"] == pytest.approx(unscaled * 0.5 * 0.25)


def test_best_available_claim_is_not_penalized_unless_it_overwrites_fool() -> None:
    state = _state()
    curr = _info(
        planet_claim_observed=True,
        planet_claim_best_available=True,
        planet_claim_play_share=0.0,
    )
    config = RewardConfig(
        enable_planet_match_rewards=True,
        planet_unmatched_claim_penalty_coeff=0.4,
    )

    normal = default_reward_components(state, _info(), curr, False, False, config)
    protected = default_reward_components(
        state,
        _info(consumable_keys=("c_fool",), last_tarot_planet="c_death"),
        curr,
        False,
        False,
        config,
    )

    assert normal["planet_unmatched_claim_penalty"] == 0
    assert protected["planet_unmatched_claim_penalty"] == pytest.approx(-0.4)


def test_build_potential_is_optional_and_bounded() -> None:
    captured = _estimate(score=300, baseline=100, required=150)
    info = _info(
        ante=2,
        build_value_estimate=captured,
        joker_details=({"key": "j_runner", "is_scaling": True},),
        hands_available=4,
    )
    disabled = state_potential_breakdown(info, RewardConfig())
    enabled_config = RewardConfig(
        enable_score_build_potential=True,
        potential_build_cap=0.3,
    )
    enabled = state_potential_breakdown(info, enabled_config)

    assert disabled["realized_build_quality"] == 0
    assert disabled["readiness"] == 0
    assert enabled["realized_build_quality"] > 0
    assert enabled["readiness"] > 0
    assert (
        enabled["realized_build_quality"] + enabled["scaling_option_value"] + enabled["readiness"]
    ) <= enabled_config.potential_build_cap + 1e-12


def test_dense_scale_does_not_change_potential_shaping() -> None:
    state = _state()
    prev = _info(round_score=10)
    curr = _info(round_score=50)

    base = default_reward_components(state, prev, curr, False, False, RewardConfig(dense_reward_scale=1.0))
    scaled = default_reward_components(state, prev, curr, False, False, RewardConfig(dense_reward_scale=0.1))

    assert scaled["potential_shaping"] == pytest.approx(base["potential_shaping"])


def test_survival_shaping_rewards_rescue_and_saturates_at_safety() -> None:
    config = RewardConfig(
        gamma=1.0,
        enable_score_build_potential=True,
        potential_w_survival_safety=0.4,
    )

    rescue = survival_shaping_reward(
        _info(clear_probability=0.10),
        _info(clear_probability=0.50),
        config,
    )
    regression = survival_shaping_reward(
        _info(clear_probability=0.50),
        _info(clear_probability=0.10),
        config,
    )
    safely_overbuilt = survival_shaping_reward(
        _info(clear_probability=0.70),
        _info(clear_probability=0.95),
        config,
    )

    assert rescue == pytest.approx(0.4 * 0.4 / 0.65)
    assert regression == pytest.approx(-rescue)
    assert safely_overbuilt == pytest.approx(0.0)


def test_survival_shaping_is_disabled_with_build_potential() -> None:
    config = RewardConfig(enable_score_build_potential=False)

    assert survival_shaping_reward(
        _info(clear_probability=0.0),
        _info(clear_probability=1.0),
        config,
    ) == pytest.approx(0.0)


def test_improving_move_joker_uses_only_exact_layout_reward() -> None:
    state = _state()
    components = default_reward_components(
        state,
        _info(),
        _info(
            action_type="move_joker",
            joker_move_reward=0.75,
            progress_made=False,
            steps_since_progress=20,
        ),
        False,
        False,
    )

    assert components["joker_move"] == pytest.approx(0.75)
    assert components["potential_shaping"] == 0
    assert components["idle_penalty"] == 0
    assert components["total"] == pytest.approx(0.75)


def test_non_improving_move_joker_gets_immediate_loop_penalty() -> None:
    state = _state()
    components = default_reward_components(
        state,
        _info(),
        _info(
            action_type="move_joker",
            joker_move_reward=0.0,
            progress_made=False,
            steps_since_progress=20,
        ),
        False,
        False,
        RewardConfig(dense_reward_scale=0.25),
    )

    expected_move_penalty = min(
        JOKER_MOVE_NON_IMPROVING_PENALTY + 19 * JOKER_MOVE_REPEAT_PENALTY,
        JOKER_MOVE_PENALTY_CAP,
    )
    assert components["joker_move"] == pytest.approx(-expected_move_penalty)
    assert components["potential_shaping"] == 0
    assert components["idle_penalty"] < 0
    assert components["total"] == pytest.approx(components["joker_move"] + components["idle_penalty"])


def test_non_improving_move_joker_penalty_starts_strong_and_caps() -> None:
    state = _state()
    first = default_reward_components(
        state,
        _info(),
        _info(action_type="move_joker", joker_move_reward=0.0, steps_since_progress=1),
        False,
        False,
    )
    repeated = default_reward_components(
        state,
        _info(),
        _info(action_type="move_joker", joker_move_reward=0.0, steps_since_progress=100),
        False,
        False,
    )

    assert first["joker_move"] == pytest.approx(-JOKER_MOVE_NON_IMPROVING_PENALTY)
    assert repeated["joker_move"] == pytest.approx(-JOKER_MOVE_PENALTY_CAP)


def test_ante1_chip_tempo_rewards_actual_blind_progress_and_scales() -> None:
    state = _state(ante=1)
    config = RewardConfig(dense_reward_scale=0.25)
    components = default_reward_components(
        state,
        _info(ante=1, blind_target=400.0, round_score=0.0, hands_left=4),
        _info(
            ante=1,
            action_type="play_subset",
            blind_target=400.0,
            round_score=90.0,
        ),
        False,
        False,
        config,
    )

    assert components["ante1_chip_tempo"] == pytest.approx(
        ANTE1_CHIP_TEMPO_BONUS * (90.0 / 400.0) * config.dense_reward_scale
    )


def test_ante1_chip_tempo_is_segmentation_invariant() -> None:
    state = _state(ante=1)
    config = RewardConfig(dense_reward_scale=0.25)

    one_play = default_reward_components(
        state,
        _info(ante=1, blind_target=400.0, round_score=0.0, hands_left=4),
        _info(ante=1, blind_target=400.0, round_score=200.0, action_type="play_subset"),
        False,
        False,
        config,
    )["ante1_chip_tempo"]
    first = default_reward_components(
        state,
        _info(ante=1, blind_target=400.0, round_score=0.0, hands_left=4),
        _info(ante=1, blind_target=400.0, round_score=100.0, action_type="play_subset"),
        False,
        False,
        config,
    )["ante1_chip_tempo"]
    second = default_reward_components(
        state,
        _info(ante=1, blind_target=400.0, round_score=100.0, hands_left=3),
        _info(ante=1, blind_target=400.0, round_score=200.0, action_type="play_subset"),
        False,
        False,
        config,
    )["ante1_chip_tempo"]

    assert first + second == pytest.approx(one_play)
    assert one_play < WIN_VALUE * config.dense_reward_scale


def test_ante1_chip_tempo_ignores_discards_and_is_disabled_later() -> None:
    discard = default_reward_components(
        _state(ante=1),
        _info(ante=1, blind_target=400.0, hands_left=4, discards_left=1),
        _info(ante=1, action_type="discard_subset", round_score=100.0),
        False,
        False,
    )
    later = default_reward_components(
        _state(ante=2),
        _info(ante=2, blind_target=800.0, hands_left=4, discards_left=1),
        _info(ante=2, action_type="play_subset", ante1_chip_chosen_score=800.0),
        False,
        False,
    )

    assert discard["ante1_chip_tempo"] == 0.0
    assert later["ante1_chip_tempo"] == 0.0


@pytest.mark.parametrize(("terminated", "won"), [(False, False), (True, False), (True, True)])
def test_ante1_chip_tempo_pays_same_progress_on_terminal_and_nonterminal(
    terminated: bool,
    won: bool,
) -> None:
    config = RewardConfig(dense_reward_scale=0.25)
    components = default_reward_components(
        _state(ante=1, win_ante=1),
        _info(ante=1, blind_target=400.0, round_score=0.0),
        _info(ante=1, blind_target=400.0, round_score=200.0, action_type="play_subset"),
        terminated,
        won,
        config,
    )

    assert components["ante1_chip_tempo"] == pytest.approx(ANTE1_CHIP_TEMPO_BONUS * 0.5 * config.dense_reward_scale)


def test_ante1_chip_tempo_telescopes_through_score_reset() -> None:
    state = _state(ante=1)
    config = RewardConfig(dense_reward_scale=0.25)
    up = default_reward_components(
        state,
        _info(ante=1, blind_target=400.0, round_score=0.0),
        _info(ante=1, blind_target=400.0, round_score=100.0, action_type="play_subset"),
        False,
        False,
        config,
    )["ante1_chip_tempo"]
    reset = default_reward_components(
        state,
        _info(ante=1, blind_target=400.0, round_score=100.0),
        _info(ante=1, blind_target=400.0, round_score=0.0, action_type="play_subset"),
        False,
        False,
        config,
    )["ante1_chip_tempo"]

    assert up > 0.0
    assert reset < 0.0
    assert up + reset == pytest.approx(0.0)


def test_ante1_chip_tempo_caps_progress_and_negative_scores() -> None:
    config = RewardConfig(dense_reward_scale=0.25)
    components = default_reward_components(
        _state(ante=1),
        _info(ante=1, blind_target=400.0, round_score=-50.0),
        _info(ante=1, blind_target=400.0, round_score=800.0, action_type="play_subset"),
        False,
        False,
        config,
    )

    assert components["ante1_chip_tempo"] == pytest.approx(ANTE1_CHIP_TEMPO_BONUS * config.dense_reward_scale)
