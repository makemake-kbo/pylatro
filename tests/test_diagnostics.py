"""Tests for the shared action-quality diagnostics (pack-claim planet ranking)."""

from __future__ import annotations

from types import SimpleNamespace

from pylatro_agent.action import ActionType
from pylatro_agent.diagnostics import action_diagnostics

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


def test_pack_claim_best_available_true_for_sole_planet_and_ties() -> None:
    hands = {"Straight": _hand(8), "High Card": _hand(0)}

    # Non-planet cards in the pack do not compete in the ranking.
    state = _pack_state(["c_pluto", "c_fool"], hands)
    assert _claim(state, 0)["planet_claim_best_available"] is True

    # Ties keep the claim exempt: either of two Plutos is "best available".
    state = _pack_state(["c_pluto", "c_pluto"], hands)
    assert _claim(state, 0)["planet_claim_best_available"] is True
    assert _claim(state, 1)["planet_claim_best_available"] is True
