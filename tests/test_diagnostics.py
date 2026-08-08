"""Tests for the shared action-quality diagnostics (pack-claim planet ranking)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pylatro import create_run_state, load_game_data, select_blind, start_blind
from pylatro_agent.action import ActionType, encode_action
from pylatro_agent.constants import NUM_ACTIONS, ActionRange, SubPhase
from pylatro_agent.diagnostics import (
    action_diagnostics,
    consumable_funnel_state_diagnostics,
    step_event_diagnostics,
)
from pylatro_agent.hand_candidates import HandCandidate
from pylatro_agent.masks import compute_action_mask
from pylatro_agent.subset_actions import subset_index

_CENTERS = {
    "c_pluto": {"set": "Planet", "config": {"hand_type": "High Card"}},
    "c_mercury": {"set": "Planet", "config": {"hand_type": "Pair"}},
    "c_saturn": {"set": "Planet", "config": {"hand_type": "Straight"}},
    "c_fool": {"set": "Tarot", "config": {}},
}


def _hand(played: int) -> dict:
    return {"played": played, "level": 1, "chips": 30, "mult": 4}


def _pack_state(pack_keys: list[str], hands: dict[str, dict]) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(centers=_CENTERS),
        hands=hands,
        pack=SimpleNamespace(
            cards=[SimpleNamespace(center_key=key) for key in pack_keys],
            state_name="PLANET_PACK",
        ),
    )


def _claim(state: SimpleNamespace, index: int) -> dict:
    decoded = SimpleNamespace(action_type=ActionType.PACK_CLAIM, index=index)
    return action_diagnostics(state, decoded)


@pytest.fixture(scope="module")
def game_data():
    return load_game_data()


def _boss_choose_state(game_data, boss_key: str):
    state = create_run_state("diagnostic_mask", 1, "b_red", data=game_data)
    state.round_resets.blind_choices["Boss"] = boss_key
    state.blind_on_deck = "Boss"
    select_blind(state, "Boss")
    start_blind(state, "Boss")
    return state


def test_play_best_filters_psychic_illegal_generated_top1(game_data, monkeypatch) -> None:
    from pylatro_agent import diagnostics as diagnostics_module

    state = _boss_choose_state(game_data, "bl_psychic")
    illegal = HandCandidate(
        kind="play",
        indices=(0,),
        hand_name="High Card",
        estimated_score=200.0,
        raw_score=200.0,
    )
    legal = HandCandidate(
        kind="play",
        indices=(0, 1, 2, 3, 4),
        hand_name="Pair",
        estimated_score=100.0,
        raw_score=100.0,
    )
    monkeypatch.setattr(
        diagnostics_module,
        "generate_hand_candidates",
        lambda _state: ((illegal, legal), ()),
    )
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)
    decoded = SimpleNamespace(
        action_type=ActionType.PLAY_SUBSET,
        index=subset_index(legal.indices),
    )

    diagnostics = action_diagnostics(
        state,
        decoded,
        action_mask=mask,
        blind_target=300.0,
    )

    assert diagnostics["hand_play_top1"] is True
    assert diagnostics["hand_play_legal_top1"] is True
    assert diagnostics["hand_play_best_hand"] == "Pair"
    assert diagnostics["hand_play_candidate_value_ratio"] == 1.0
    assert diagnostics["ante1_conservative_chosen_best_ratio"] == 1.0

    state.round_resets.ante = 2
    diagnostics = action_diagnostics(state, decoded, action_mask=mask)
    assert "ante1_play_observed" not in diagnostics
    assert "ante1_conservative_chosen_best_ratio" not in diagnostics


def test_play_best_filters_eye_repeated_hand_mask_path(game_data, monkeypatch) -> None:
    from pylatro_agent import diagnostics as diagnostics_module

    state = _boss_choose_state(game_data, "bl_eye")
    aces = [card for card in state.deck_cards if card.rank == "A"][:2]
    other = next(card for card in state.deck_cards if card.rank == "7")
    state.hand_cards = [*aces, other]
    state.eye_hands = {"High Card": True}
    illegal = HandCandidate(
        kind="play",
        indices=(2,),
        hand_name="High Card",
        estimated_score=200.0,
    )
    legal = HandCandidate(
        kind="play",
        indices=(0, 1),
        hand_name="Pair",
        estimated_score=100.0,
    )
    monkeypatch.setattr(
        diagnostics_module,
        "generate_hand_candidates",
        lambda _state: ((illegal, legal), ()),
    )
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)
    decoded = SimpleNamespace(
        action_type=ActionType.PLAY_SUBSET,
        index=subset_index(legal.indices),
    )

    diagnostics = action_diagnostics(state, decoded, action_mask=mask)

    assert diagnostics["hand_play_top1"] is True
    assert diagnostics["hand_play_best_hand"] == "Pair"


def test_pack_claim_best_available_false_when_main_hand_planet_present() -> None:
    hands = {"Straight": _hand(8), "High Card": _hand(0), "Pair": _hand(0)}
    state = _pack_state(["c_pluto", "c_saturn"], hands)

    pluto = _claim(state, 0)
    assert pluto["planet_claim_observed"] is True
    assert pluto["planet_claim_main_hand_match"] is False
    assert pluto["planet_claim_best_available"] is False

    saturn = _claim(state, 1)
    assert saturn["planet_claim_main_hand_match"] is True
    assert saturn["planet_claim_best_available"] is True


def test_pack_claim_best_available_ranks_matchless_pack_by_play_share() -> None:
    # Main hand (Straight) is not in the pack; Pair has been played, High
    # Card has not, so Mercury outranks Pluto.
    hands = {"Straight": _hand(8), "Pair": _hand(2), "High Card": _hand(0)}
    state = _pack_state(["c_pluto", "c_mercury"], hands)

    assert _claim(state, 0)["planet_claim_best_available"] is False
    assert _claim(state, 1)["planet_claim_best_available"] is True


def test_step_event_diagnostics_exports_shop_and_roster_ids() -> None:
    prev = {
        "in_shop": False,
        "joker_keys": ("j_joker", "j_joker"),
        "joker_details": ({"key": "j_joker"}, {"key": "j_joker"}),
        "shop_cards": (
            {"key": "j_hologram", "set": "Joker"},
            {"key": "c_pluto", "set": "Planet"},
        ),
    }
    curr = {
        "in_shop": True,
        "joker_keys": ("j_joker", "j_hologram"),
        "shop_cards": (
            {"key": "j_hologram", "set": "Joker"},
            {"key": "j_blueprint", "set": "Joker"},
        ),
    }
    decoded = SimpleNamespace(action_type=ActionType.SHOP_SELL_JOKER, index=1)

    diagnostics = step_event_diagnostics(prev, curr, decoded)

    assert diagnostics["shop_sold_joker_id"] == "j_joker"
    assert diagnostics["joker_acquired_0_id"] == "j_hologram"
    assert diagnostics["joker_removed_0_id"] == "j_joker"
    assert diagnostics["joker_replacement_event"] is True
    assert diagnostics["shop_offered_joker_count"] == 2
    assert diagnostics["shop_offered_joker_0_id"] == "j_hologram"
    assert diagnostics["shop_offered_joker_1_id"] == "j_blueprint"


def test_pack_claim_best_available_true_for_sole_planet_and_ties() -> None:
    hands = {"Straight": _hand(8), "High Card": _hand(0)}

    # Non-planet cards in the pack do not compete in the ranking.
    state = _pack_state(["c_pluto", "c_fool"], hands)
    assert _claim(state, 0)["planet_claim_best_available"] is True

    # Ties keep the claim exempt: either of two Plutos is "best available".
    state = _pack_state(["c_pluto", "c_pluto"], hands)
    assert _claim(state, 0)["planet_claim_best_available"] is True
    assert _claim(state, 1)["planet_claim_best_available"] is True


def test_targeted_tarot_actions_are_counted_as_consumable_uses() -> None:
    state = SimpleNamespace(
        data=SimpleNamespace(centers={"c_death": {"set": "Tarot", "config": {}}}),
        consumables=[SimpleNamespace(center_key="c_death")],
    )

    for action_type in (
        ActionType.USE_CONSUMABLE_HAND_SUBSET,
        ActionType.USE_CONSUMABLE_JOKER,
    ):
        decoded = SimpleNamespace(action_type=action_type, index=0)
        diagnostics = action_diagnostics(state, decoded)
        assert diagnostics["consumable_use_set"] == "Tarot"
        assert diagnostics["consumable_use_key"] == "c_death"


def test_seal_generation_events_are_exported() -> None:
    prev = {"consumable_details": ()}
    tarot = {"consumable_details": ({"key": "c_death", "set": "Tarot"},)}
    planet = {"consumable_details": ({"key": "c_mercury", "set": "Planet"},)}

    discard = step_event_diagnostics(
        prev,
        tarot,
        SimpleNamespace(action_type=ActionType.DISCARD_SUBSET, index=0),
    )
    play = step_event_diagnostics(
        prev,
        planet,
        SimpleNamespace(action_type=ActionType.PLAY_SUBSET, index=0),
    )

    assert discard["purple_seal_tarot_generated_count"] == 1
    assert play["blue_seal_planet_generated_count"] == 1


def test_pack_offer_funnel_distinguishes_full_slots_from_legal_auto_use() -> None:
    prev = {"pack_card_details": ()}
    curr = {
        "pack_card_details": (
            {"key": "c_mercury", "set": "Planet"},
            {"key": "c_death", "set": "Tarot"},
        ),
        "consumable_details": (
            {"key": "c_fool", "set": "Tarot"},
            {"key": "c_hermit", "set": "Tarot"},
        ),
        "consumable_limit": 2,
    }
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    mask[int(ActionRange.PACK_CLAIM_START)] = True

    diagnostics = step_event_diagnostics(
        prev,
        curr,
        SimpleNamespace(action_type=ActionType.SHOP_BUY, index=0),
        next_action_mask=mask,
    )

    assert diagnostics["consumable_planet_offered_count"] == 1
    assert diagnostics["consumable_planet_claimable_count"] == 1
    assert diagnostics["consumable_planet_inventory_full_blocked_count"] == 0
    assert diagnostics["consumable_tarot_offered_count"] == 1
    assert diagnostics["consumable_tarot_claimable_count"] == 0
    assert diagnostics["consumable_tarot_inventory_full_blocked_count"] == 1


def test_consumable_funnel_uses_exact_legal_action_blocks() -> None:
    info = {
        "sub_phase": SubPhase.CHOOSE_ACTION,
        "consumable_details": (
            {"key": "c_mercury", "set": "Planet", "hand_type": "Pair"},
            {"key": "c_death", "set": "Tarot"},
        ),
    }
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    mask[encode_action(ActionType.USE_CONSUMABLE_NO_TARGET, 0)] = True

    diagnostics = consumable_funnel_state_diagnostics(info, mask)

    assert diagnostics["consumable_planet_owned_count"] == 1
    assert diagnostics["consumable_planet_legal_use_opportunity"] is True
    assert diagnostics["consumable_tarot_owned_count"] == 1
    assert diagnostics["consumable_tarot_legal_use_opportunity"] is False

    pack_info = {
        "sub_phase": SubPhase.BOOSTER_PACK,
        "pack_card_details": ({"key": "c_mercury", "set": "Planet"},),
    }
    pack_mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    pack_mask[int(ActionRange.PACK_CLAIM_START)] = True
    pack_diagnostics = consumable_funnel_state_diagnostics(pack_info, pack_mask)

    assert pack_diagnostics["consumable_planet_eligible_offer_opportunity"] is True
    assert pack_diagnostics["consumable_tarot_eligible_offer_opportunity"] is False
