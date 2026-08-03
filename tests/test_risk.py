from __future__ import annotations

from collections import Counter
from copy import deepcopy

import pytest

from pylatro_agent.risk import (
    calibrate_analytic_death_probability,
    estimate_clear_risk,
    uncalibrate_analytic_death_probability,
)
from pylatro_agent.strategy_value import _economy_value

RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
SUITS = ("Spades", "Hearts", "Clubs", "Diamonds")


def _snapshot(*, target: float = 200.0, score: float = 0.0) -> dict:
    cards = [
        {"rank": rank, "suit": suit, "seal": "", "enhancement": "", "edition": ""}
        for suit in SUITS
        for rank in RANKS
    ]
    ranks = Counter(card["rank"] for card in cards)
    suits = Counter(card["suit"] for card in cards)
    exact = Counter((card["rank"], card["suit"]) for card in cards)
    return {
        "ante": 3,
        "dollars": 20,
        "blind_on_deck": "small",
        "boss_key": "",
        "blind_target": target,
        "round_score": score,
        "hands_available": 4,
        "discards_available": 3,
        "hand_size": 8,
        "joker_limit": 5,
        "consumable_limit": 2,
        "joker_details": (),
        "consumable_details": (),
        "hand_details": {
            "High Card": {"chips": 5, "mult": 1, "level": 1, "played": 2},
            "Pair": {"chips": 10, "mult": 2, "level": 1, "played": 8},
            "Two Pair": {"chips": 20, "mult": 2, "level": 1, "played": 5},
            "Three of a Kind": {"chips": 30, "mult": 3, "level": 1, "played": 0},
            "Flush": {"chips": 35, "mult": 4, "level": 1, "played": 0},
            "Full House": {"chips": 40, "mult": 4, "level": 1, "played": 6},
            "Four of a Kind": {"chips": 60, "mult": 7, "level": 1, "played": 0},
            "Five of a Kind": {"chips": 120, "mult": 12, "level": 1, "played": 0},
            "Flush Five": {"chips": 160, "mult": 16, "level": 1, "played": 0},
        },
        "deck_stats": {
            "size": len(cards),
            "cards": tuple(cards),
            "rank_counts": dict(ranks),
            "suit_counts": dict(suits),
            "rank_suit_counts": dict(exact),
            "seal_counts": {},
            "gold_count": 0,
        },
        "shop_cards": (),
    }


def test_clear_risk_tracks_score_margin_and_current_progress() -> None:
    unsafe = estimate_clear_risk(_snapshot(target=400.0))
    partial = estimate_clear_risk(_snapshot(target=200.0))
    nearly_clear = estimate_clear_risk(_snapshot(target=400.0, score=300.0))

    assert unsafe.clear_probability == pytest.approx(
        1.0 - calibrate_analytic_death_probability(1.0)
    )
    assert partial.clear_probability == pytest.approx(
        1.0 - calibrate_analytic_death_probability(1.0 - 0.46)
    )
    assert nearly_clear.clear_probability == pytest.approx(
        1.0 - calibrate_analytic_death_probability(0.0)
    )
    assert nearly_clear.immediate_death_probability < 0.01
    assert unsafe.score_margin > 0.0


def test_analytic_death_calibration_corrects_pessimism_and_is_invertible() -> None:
    calibrated = calibrate_analytic_death_probability(0.81)

    assert calibrated == pytest.approx(0.32, abs=0.01)
    assert calibrate_analytic_death_probability(0.90) > calibrated
    assert calibrate_analytic_death_probability(0.50) < calibrated
    assert uncalibrate_analytic_death_probability(calibrated) == pytest.approx(0.81)


@pytest.mark.parametrize("sub_phase", ["shop", "booster_pack", "blind_select"])
def test_clear_risk_ignores_completed_blind_score_for_upcoming_blind(sub_phase: str) -> None:
    upcoming = _snapshot(target=400.0, score=300.0)
    upcoming["sub_phase"] = sub_phase
    upcoming["in_shop"] = sub_phase == "shop"

    active = _snapshot(target=400.0, score=300.0)
    active["sub_phase"] = "choose_action"

    assert estimate_clear_risk(upcoming).clear_probability == estimate_clear_risk(
        _snapshot(target=400.0)
    ).clear_probability
    assert estimate_clear_risk(active).clear_probability > 0.99


def test_clear_risk_reserves_margin_for_unmodeled_boss_constraints() -> None:
    ordinary = estimate_clear_risk(_snapshot(target=200.0))
    boss_info = deepcopy(_snapshot(target=200.0))
    boss_info.update({"blind_on_deck": "boss", "boss_key": "bl_flint"})
    boss = estimate_clear_risk(boss_info)

    assert boss.clear_probability < ordinary.clear_probability
    assert boss.immediate_death_probability > ordinary.immediate_death_probability


def test_economy_value_is_gated_by_immediate_survival() -> None:
    info = _snapshot()

    assert _economy_value(info, 0.49) == 0.0
    assert 0.0 < _economy_value(info, 0.65) < _economy_value(info, 0.80)
