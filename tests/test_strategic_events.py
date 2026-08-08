from __future__ import annotations

from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import pytest

from pylatro import create_run_state, load_game_data
from pylatro.consumables import UseConsumableResult
from pylatro.runtime import add_playing_cards, create_playing_card
from pylatro_agent.action import ActionType
from pylatro_agent.hand_plan import estimate_hand_plans
from pylatro_agent.shop_eval import capture_build_features
from pylatro_agent.strategic_events import (
    contextual_tarot_fix_reward,
    derive_strategic_event,
)


def _decoded(action_type: ActionType, index: int = 0):
    return SimpleNamespace(action_type=action_type, index=index, detail=0)


def _card(identity: int, rank: str, suit: str, **extra) -> dict:
    return {
        "identity": identity,
        "rank": rank,
        "suit": suit,
        "center_key": "c_base",
        "enhancement": "",
        "seal": "",
        "edition": "",
        "perma_bonus": 0,
        "times_played": 0,
        **extra,
    }


def _plan_snapshot(cards: list[dict]) -> dict:
    ranks = Counter(str(card["rank"]) for card in cards)
    suits = Counter(str(card["suit"]) for card in cards)
    exact = Counter((str(card["rank"]), str(card["suit"])) for card in cards)
    hand_details = {
        "High Card": {"chips": 5, "mult": 1, "level": 1, "played": 0},
        "Pair": {"chips": 10, "mult": 2, "level": 1, "played": 0},
        "Two Pair": {"chips": 20, "mult": 2, "level": 1, "played": 0},
        "Three of a Kind": {"chips": 30, "mult": 3, "level": 1, "played": 0},
        "Flush": {"chips": 100, "mult": 10, "level": 1, "played": 6},
        "Full House": {"chips": 40, "mult": 4, "level": 1, "played": 0},
        "Four of a Kind": {"chips": 60, "mult": 7, "level": 1, "played": 0},
        "Five of a Kind": {"chips": 120, "mult": 12, "level": 1, "played": 0},
        "Flush Five": {"chips": 160, "mult": 16, "level": 1, "played": 0},
    }
    return {
        "ante": 3,
        "blind_target": 100,
        "hands_available": 4,
        "discards_available": 3,
        "hand_size": 8,
        "clear_probability": 0.8,
        "joker_details": (),
        "hand_details": hand_details,
        "hand_play_counts": {name: detail["played"] for name, detail in hand_details.items()},
        "deck_stats": {
            "size": len(cards),
            "cards": tuple(cards),
            "rank_counts": dict(ranks),
            "suit_counts": dict(suits),
            "rank_suit_counts": dict(exact),
        },
    }


def _flush_deck() -> list[dict]:
    ranks = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
    suits = ("Spades", "Hearts", "Clubs", "Diamonds")
    cards = [_card(index + 1, rank, suit) for index, (suit, rank) in enumerate((s, r) for s in suits for r in ranks)]
    for card in cards:
        if card["suit"] == "Diamonds" and card["rank"] in ranks[:8]:
            card["suit"] = "Spades"
    return cards


def test_gold_creation_is_attributed_once_to_tarot_or_live_midas() -> None:
    prev = {
        "tarot_usage_total": 0,
        "consumable_details": ({"key": "c_devil", "set": "Tarot"},),
        "deck_stats": {"cards": (_card(1, "A", "Spades"),)},
        "joker_details": (),
    }
    post = deepcopy(prev)
    post["tarot_usage_total"] = 1
    post["deck_stats"]["cards"][0]["center_key"] = "m_gold"
    ledger: set[int] = set()

    tarot_event = derive_strategic_event(
        prev,
        post,
        _decoded(ActionType.USE_CONSUMABLE_HAND_SUBSET),
        UseConsumableResult("c_devil"),
        ledger,
    )
    duplicate_event = derive_strategic_event(
        prev,
        post,
        _decoded(ActionType.USE_CONSUMABLE_HAND_SUBSET),
        UseConsumableResult("c_devil"),
        ledger,
    )

    midas_prev = deepcopy(prev)
    midas_prev["consumable_details"] = ()
    midas_prev["joker_details"] = ({"key": "j_midas_mask", "debuffed": False},)
    midas_post = deepcopy(midas_prev)
    midas_post["deck_stats"]["cards"][0]["center_key"] = "m_gold"
    midas_event = derive_strategic_event(
        midas_prev,
        midas_post,
        _decoded(ActionType.PLAY_SUBSET),
        SimpleNamespace(),
        set(),
    )

    assert tarot_event.gold_created_tarot == 1
    assert tarot_event.gold_created_midas == 0
    assert tarot_event.tarot_source == "inventory"
    assert tarot_event.tarot_uses == 1
    assert duplicate_event.gold_created_tarot == 0
    assert midas_event.gold_created_tarot == 0
    assert midas_event.gold_created_midas == 1


def test_gold_creation_excludes_vampire_reversion_and_new_cards() -> None:
    prev = {
        "tarot_usage_total": 0,
        "consumable_details": (),
        "joker_details": ({"key": "j_midas_mask", "debuffed": False},),
        "deck_stats": {"cards": (_card(1, "K", "Hearts"),)},
    }
    post = deepcopy(prev)
    post["deck_stats"]["cards"] = (
        _card(1, "K", "Hearts"),
        _card(2, "K", "Hearts", center_key="m_gold"),
    )

    event = derive_strategic_event(
        prev,
        post,
        _decoded(ActionType.PLAY_SUBSET),
        SimpleNamespace(),
        set(),
    )

    assert event.gold_created_midas == 0


def test_gold_creation_counts_non_gold_enhancement_conversion_for_tarot_and_midas() -> None:
    steel = _card(1, "K", "Hearts", center_key="m_steel", enhancement="Steel Card")
    tarot_prev = {
        "tarot_usage_total": 0,
        "consumable_details": ({"key": "c_devil", "set": "Tarot"},),
        "joker_details": (),
        "deck_stats": {"cards": (deepcopy(steel),)},
    }
    tarot_post = deepcopy(tarot_prev)
    tarot_post["tarot_usage_total"] = 1
    tarot_post["deck_stats"]["cards"][0]["center_key"] = "m_gold"
    tarot_post["deck_stats"]["cards"][0]["enhancement"] = "Gold Card"

    midas_prev = {
        **deepcopy(tarot_prev),
        "consumable_details": (),
        "joker_details": ({"key": "j_midas_mask", "debuffed": False},),
    }
    midas_post = deepcopy(midas_prev)
    midas_post["deck_stats"]["cards"][0]["center_key"] = "m_gold"
    midas_post["deck_stats"]["cards"][0]["enhancement"] = "Gold Card"

    tarot_event = derive_strategic_event(
        tarot_prev,
        tarot_post,
        _decoded(ActionType.USE_CONSUMABLE_HAND_SUBSET),
        UseConsumableResult("c_devil"),
        set(),
    )
    midas_event = derive_strategic_event(
        midas_prev,
        midas_post,
        _decoded(ActionType.PLAY_SUBSET),
        SimpleNamespace(),
        set(),
    )

    assert tarot_event.gold_created_tarot == 1
    assert midas_event.gold_created_midas == 1


def test_gold_creation_excludes_already_gold_recreate_dna_and_vampire_end_state() -> None:
    gold = _card(1, "Q", "Spades", center_key="m_gold", enhancement="Gold Card")
    already_prev = {
        "tarot_usage_total": 0,
        "consumable_details": ({"key": "c_devil", "set": "Tarot"},),
        "joker_details": (),
        "deck_stats": {"cards": (deepcopy(gold),)},
    }
    already_post = deepcopy(already_prev)
    already_post["tarot_usage_total"] = 1
    already = derive_strategic_event(
        already_prev,
        already_post,
        _decoded(ActionType.USE_CONSUMABLE_HAND_SUBSET),
        UseConsumableResult("c_devil"),
        set(),
    )

    base = _card(2, "J", "Clubs")
    recreated_post = {
        **deepcopy(already_prev),
        "tarot_usage_total": 1,
        "deck_stats": {"cards": (_card(2, "J", "Clubs", center_key="m_gold"),)},
    }
    recreated = derive_strategic_event(
        {**deepcopy(already_prev), "deck_stats": {"cards": (base,)}},
        recreated_post,
        _decoded(ActionType.USE_CONSUMABLE_HAND_SUBSET),
        UseConsumableResult("c_devil"),
        {2},
    )

    midas_prev = {
        "tarot_usage_total": 0,
        "consumable_details": (),
        "joker_details": ({"key": "j_midas_mask", "debuffed": False},),
        "deck_stats": {"cards": (_card(3, "K", "Diamonds"),)},
    }
    dna_post = deepcopy(midas_prev)
    dna_post["deck_stats"]["cards"] = (
        _card(3, "K", "Diamonds"),
        _card(4, "K", "Diamonds", center_key="m_gold"),
    )
    dna = derive_strategic_event(
        midas_prev,
        dna_post,
        _decoded(ActionType.PLAY_SUBSET),
        SimpleNamespace(),
        set(),
    )
    vampire_post = deepcopy(midas_prev)
    vampire = derive_strategic_event(
        midas_prev,
        vampire_post,
        _decoded(ActionType.PLAY_SUBSET),
        SimpleNamespace(),
        set(),
    )

    assert already.gold_created_tarot == 0
    assert recreated.gold_created_tarot == 0
    assert dna.gold_created_midas == 0
    assert vampire.gold_created_midas == 0


def test_gold_reward_uid_survives_destruction_without_reuse_or_same_card_farming() -> None:
    from pylatro.consumables import _destroy_card

    state = create_run_state("gold_uid", data=load_game_data())
    card = state.deck_cards[0]
    ledger: set[int] = set()

    first_pre = capture_build_features(state)
    first_pre["consumable_details"] = ({"key": "c_devil", "set": "Tarot"},)
    card.center_key = "m_gold"
    first_post = capture_build_features(state)
    first_post["tarot_usage_total"] = first_pre["tarot_usage_total"] + 1
    first = derive_strategic_event(
        first_pre,
        first_post,
        _decoded(ActionType.USE_CONSUMABLE_HAND_SUBSET),
        UseConsumableResult("c_devil"),
        ledger,
    )

    card.center_key = "c_base"
    repeat_pre = capture_build_features(state)
    repeat_pre["consumable_details"] = ({"key": "c_devil", "set": "Tarot"},)
    card.center_key = "m_gold"
    repeat_post = capture_build_features(state)
    repeat_post["tarot_usage_total"] = repeat_pre["tarot_usage_total"] + 1
    repeat = derive_strategic_event(
        repeat_pre,
        repeat_post,
        _decoded(ActionType.USE_CONSUMABLE_HAND_SUBSET),
        UseConsumableResult("c_devil"),
        ledger,
    )

    old_uid = card.reward_uid
    _destroy_card(state, card)
    replacement = create_playing_card(state, front_key=card.front_key)
    add_playing_cards(state, [replacement])
    replacement_pre = capture_build_features(state)
    replacement_pre["consumable_details"] = ({"key": "c_devil", "set": "Tarot"},)
    replacement.center_key = "m_gold"
    replacement_post = capture_build_features(state)
    replacement_post["tarot_usage_total"] = replacement_pre["tarot_usage_total"] + 1
    replacement_event = derive_strategic_event(
        replacement_pre,
        replacement_post,
        _decoded(ActionType.USE_CONSUMABLE_HAND_SUBSET),
        UseConsumableResult("c_devil"),
        ledger,
    )

    assert first.gold_created_tarot == 1
    assert repeat.gold_created_tarot == 0
    assert replacement.reward_uid != old_uid
    assert replacement_event.gold_created_tarot == 1


def test_pack_auto_use_and_cash_payout_are_authoritative() -> None:
    prev = {
        "tarot_usage_total": 4,
        "pack_card_details": ({"key": "c_hermit", "set": "Tarot"},),
        "deck_stats": {"cards": ()},
    }
    curr = {**prev, "tarot_usage_total": 5}
    result = SimpleNamespace(
        auto_used=True,
        center_key="c_hermit",
        use_result=UseConsumableResult("c_hermit", dollars_delta=10),
    )

    event = derive_strategic_event(
        prev,
        curr,
        _decoded(ActionType.PACK_CLAIM),
        result,
        set(),
    )

    assert event.tarot_source == "pack_auto_use"
    assert event.pack_auto_use
    assert event.tarot_uses == 1
    assert event.tarot_acquired == 1
    assert event.tarot_pack_auto_uses == 1
    assert event.attributable_cash_payout == 10


def test_planet_pack_auto_use_counts_as_both_acquisition_and_use() -> None:
    prev = {
        "planet_usage_total": 7,
        "pack_card_details": ({"key": "c_mercury", "set": "Planet"},),
        "deck_stats": {"cards": ()},
    }
    curr = {**prev, "planet_usage_total": 8}
    result = SimpleNamespace(
        auto_used=True,
        center_key="c_mercury",
        use_result=UseConsumableResult("c_mercury"),
    )

    event = derive_strategic_event(
        prev,
        curr,
        _decoded(ActionType.PACK_CLAIM),
        result,
        set(),
    )

    assert event.planet_acquired == 1
    assert event.planet_uses == 1
    assert event.planet_pack_auto_uses == 1


def test_sold_consumable_is_not_misreported_as_use_or_overwrite() -> None:
    prev = {
        "planet_usage_total": 1,
        "tarot_usage_total": 1,
        "consumable_details": ({"key": "c_mercury", "set": "Planet"},),
        "deck_stats": {"cards": ()},
    }
    curr = {**prev, "consumable_details": ()}

    event = derive_strategic_event(
        prev,
        curr,
        _decoded(ActionType.SHOP_SELL_CONSUMABLE),
        SimpleNamespace(),
        set(),
    )

    assert event.planet_sold == 1
    assert event.planet_uses == 0
    assert event.planet_overwritten == 0


@pytest.mark.parametrize(("tarot_key", "payout"), (("c_hermit", 12), ("c_temperance", 9)))
def test_only_cash_tarots_receive_their_attributable_engine_payout(
    tarot_key: str,
    payout: int,
) -> None:
    prev = {
        "tarot_usage_total": 2,
        "consumable_details": ({"key": tarot_key, "set": "Tarot"},),
        "deck_stats": {"cards": ()},
    }
    curr = {**prev, "tarot_usage_total": 3, "consumable_details": ()}

    event = derive_strategic_event(
        prev,
        curr,
        _decoded(ActionType.USE_CONSUMABLE_NO_TARGET),
        UseConsumableResult(tarot_key, dollars_delta=payout),
        set(),
    )

    assert event.tarot_family == "cash"
    assert event.attributable_cash_payout == payout


def test_non_tarot_cash_effect_is_not_misattributed_as_tarot_value() -> None:
    prev = {
        "tarot_usage_total": 2,
        "consumable_details": ({"key": "c_immolate", "set": "Spectral"},),
        "deck_stats": {"cards": ()},
    }

    event = derive_strategic_event(
        prev,
        prev,
        _decoded(ActionType.USE_CONSUMABLE_HAND_SUBSET),
        UseConsumableResult("c_immolate", dollars_delta=20),
        set(),
    )

    assert event.tarot_uses == 0
    assert event.attributable_cash_payout == 0


def test_pack_seal_claim_is_attributed_from_the_selected_card() -> None:
    prev = {
        "tarot_usage_total": 0,
        "pack_card_details": (
            {"key": "m_bonus", "set": "Enhanced", "seal": "Blue"},
            {"key": "m_mult", "set": "Enhanced", "seal": "Purple"},
        ),
        "deck_stats": {"cards": ()},
    }

    event = derive_strategic_event(
        prev,
        prev,
        _decoded(ActionType.PACK_CLAIM, index=1),
        SimpleNamespace(auto_used=False),
        set(),
    )

    assert event.blue_seals_claimed == 0
    assert event.purple_seals_claimed == 1


def test_contextual_suit_tarot_rewards_only_reliable_fixed_plan_improvement() -> None:
    cards = _flush_deck()
    prev = _plan_snapshot(deepcopy(cards))
    converted = next(card for card in cards if card["suit"] == "Hearts")
    converted["suit"] = "Spades"
    curr = _plan_snapshot(cards)

    reward, pre_quality, post_quality = contextual_tarot_fix_reward(prev, curr, "c_world")
    boss_blocked = deepcopy(prev)
    boss_blocked["boss_debuff_suit"] = "Spades"
    blocked_reward, _, _ = contextual_tarot_fix_reward(boss_blocked, curr, "c_world")
    no_op_reward, _, _ = contextual_tarot_fix_reward(prev, prev, "c_world")

    assert 0.0 < reward <= 0.20
    assert post_quality > pre_quality
    assert blocked_reward <= 0.0
    assert no_op_reward == 0.0


def test_reversing_a_rewarded_suit_fix_cannot_collect_another_positive_reward() -> None:
    original_cards = _flush_deck()
    prev = _plan_snapshot(deepcopy(original_cards))
    fixed_cards = deepcopy(original_cards)
    converted = next(card for card in fixed_cards if card["suit"] == "Hearts")
    converted["suit"] = "Spades"
    fixed = _plan_snapshot(fixed_cards)

    forward, _, _ = contextual_tarot_fix_reward(prev, fixed, "c_world")
    reverse, _, _ = contextual_tarot_fix_reward(fixed, prev, "c_sun")

    assert forward > 0.0
    assert reverse <= 0.0


def test_boss_aware_conversion_rewards_moving_cards_away_from_debuffed_suit() -> None:
    cards = [_card(index, rank, suit) for index, (rank, suit) in enumerate(
        (("A", "Hearts"), ("K", "Hearts"), ("Q", "Clubs"), ("J", "Diamonds")),
        start=1,
    )]
    prev = _plan_snapshot(deepcopy(cards))
    prev["boss_debuff_suit"] = "Hearts"
    cards[0]["suit"] = "Spades"
    curr = _plan_snapshot(cards)
    curr["boss_debuff_suit"] = "Hearts"

    reward, pre_quality, post_quality = contextual_tarot_fix_reward(prev, curr, "c_world")

    assert reward > 0.0
    assert post_quality > pre_quality


def test_strength_rewards_rank_consolidation_into_the_active_pair_anchor() -> None:
    ranks = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
    suits = ("Spades", "Hearts", "Clubs", "Diamonds")
    cards = [
        _card(index + 1, rank, suit)
        for index, (suit, rank) in enumerate((suit, rank) for suit in suits for rank in ranks)
    ]
    prev = _plan_snapshot(deepcopy(cards))
    for hand_type, detail in prev["hand_details"].items():
        if hand_type != "Pair":
            detail.update({"chips": 1, "mult": 1, "played": 0})
    prev["hand_details"]["Pair"].update({"chips": 100, "mult": 10, "played": 6})
    prev["hand_play_counts"] = dict.fromkeys(prev["hand_play_counts"], 0)
    prev["hand_play_counts"]["Pair"] = 6
    fixed_cards = deepcopy(cards)
    target_rank = max(card["rank"] for card in fixed_cards)
    changed = next(card for card in fixed_cards if card["rank"] != target_rank)
    changed["rank"] = target_rank
    curr = _plan_snapshot(fixed_cards)
    curr["hand_details"] = deepcopy(prev["hand_details"])
    curr["hand_play_counts"] = deepcopy(prev["hand_play_counts"])

    reward, pre_quality, post_quality = contextual_tarot_fix_reward(prev, curr, "c_strength")

    assert reward > 0.0
    assert post_quality > pre_quality


def test_hanged_man_never_rewards_cutting_plan_or_protected_cards() -> None:
    cards = _flush_deck()
    prev = _plan_snapshot(deepcopy(cards))
    spade = next(card for card in cards if card["suit"] == "Spades")
    cards.remove(spade)
    harmful = _plan_snapshot(cards)

    protected_cards = _flush_deck()
    off_suit = next(card for card in protected_cards if card["suit"] == "Hearts")
    off_suit["seal"] = "Blue"
    protected_prev = _plan_snapshot(deepcopy(protected_cards))
    protected_cards.remove(off_suit)
    protected_post = _plan_snapshot(protected_cards)

    harmful_reward, _, _ = contextual_tarot_fix_reward(prev, harmful, "c_hanged_man")
    protected_reward, _, _ = contextual_tarot_fix_reward(
        protected_prev,
        protected_post,
        "c_hanged_man",
    )

    assert harmful_reward < 0.0
    assert protected_reward == 0.0


def test_death_cannot_hide_protected_identity_loss_with_equal_weight_asset() -> None:
    cards = _flush_deck()
    target = next(card for card in cards if card["suit"] == "Hearts")
    source = next(card for card in cards if card["suit"] == "Spades" and card is not target)
    target["center_key"] = "m_gold"
    target["enhancement"] = "Gold Card"
    source["center_key"] = "m_steel"
    source["enhancement"] = "Steel Card"
    prev = _plan_snapshot(deepcopy(cards))

    target["rank"] = source["rank"]
    target["suit"] = source["suit"]
    target["center_key"] = source["center_key"]
    target["enhancement"] = source["enhancement"]
    post = _plan_snapshot(cards)

    reward, pre_quality, post_quality = contextual_tarot_fix_reward(prev, post, "c_death")

    assert post_quality > pre_quality
    assert reward == 0.0


@pytest.mark.parametrize(
    "protected_fields",
    (
        {"center_key": "m_gold", "enhancement": "Gold Card"},
        {"center_key": "m_steel", "enhancement": "Steel Card"},
        {"edition": "negative"},
        {"seal": "Blue"},
        {"seal": "Purple"},
        {"perma_bonus": 10},
        {"times_played": 2},
    ),
)
def test_hanged_man_cannot_remove_any_protected_asset(protected_fields: dict) -> None:
    cards = _flush_deck()
    off_suit = next(card for card in cards if card["suit"] == "Hearts")
    off_suit.update(protected_fields)
    prev = _plan_snapshot(deepcopy(cards))
    cards.remove(off_suit)
    post = _plan_snapshot(cards)

    reward, _, _ = contextual_tarot_fix_reward(prev, post, "c_hanged_man")

    assert reward == 0.0


def test_contextual_reliability_stays_on_pre_action_suit_after_plan_switch() -> None:
    cards = _flush_deck()
    for card in cards:
        if card["suit"] == "Clubs" and card["rank"] in {"2", "3", "4", "5", "6", "7", "8"}:
            card["suit"] = "Hearts"
    prev = _plan_snapshot(deepcopy(cards))
    converted = next(card for card in cards if card["suit"] == "Spades")
    converted["suit"] = "Hearts"
    post = _plan_snapshot(cards)

    reward, pre_quality, post_quality = contextual_tarot_fix_reward(prev, post, "c_sun")

    assert post_quality < pre_quality
    assert reward < 0.0


def test_contextual_positive_fix_uses_fixed_anchor_even_if_post_best_family_switches() -> None:
    cards = _flush_deck()
    prev = _plan_snapshot(deepcopy(cards))
    converted = next(card for card in cards if card["suit"] == "Hearts")
    converted["suit"] = "Spades"
    post = _plan_snapshot(cards)
    # A post-only score-table change makes replanning choose another family.
    # Deck-fix attribution must nevertheless measure the exact pre-action
    # Spade/Flush anchor on both sides and not veto its real improvement.
    post["hand_details"]["Flush"].update({"chips": 1, "mult": 1, "level": 1})
    post["blind_target"] = 1_000

    pre_plan = estimate_hand_plans(prev)
    post_plan = estimate_hand_plans(post)
    assert pre_plan is not None and post_plan is not None
    assert pre_plan.best.hand_type == "Flush"
    assert post_plan.best.hand_type != pre_plan.best.hand_type

    reward, pre_quality, post_quality = contextual_tarot_fix_reward(prev, post, "c_world")

    assert post_quality > pre_quality
    assert reward > 0.0
