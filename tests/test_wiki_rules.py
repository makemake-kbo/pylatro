"""Independent rule fixtures from balatrowiki.org; see docs/audits/2026-09-11-wiki-rules.md.

Expected values are literal wiki values or hand-calculated scores, never derived
from game_data.json or scoring helpers under test.
"""

from math import floor

import pytest

from pylatro import add_consumable, add_joker, create_run_state, score_hand, use_consumable
from pylatro.models import PlayingCard


def card(key, **kwargs):
    suit, rank = key.split("_")
    return PlayingCard(
        front_key=key, suit={"S": "Spades", "H": "Hearts", "D": "Diamonds", "C": "Clubs"}[suit], rank=rank, **kwargs
    )


# Poker hands: base chips/mult, planet increments, scoring-card indices.
HANDS = [
    ("High Card", "S_A H_K D_9 C_6 S_2", 5, 1, 10, 1, "c_pluto", (0,)),
    ("Pair", "S_3 H_3 D_A C_K S_2", 10, 2, 15, 1, "c_mercury", (0, 1)),
    ("Two Pair", "S_3 H_3 D_2 C_2 S_A", 20, 2, 20, 1, "c_uranus", (0, 1, 2, 3)),
    ("Three of a Kind", "S_3 H_3 D_3 C_A S_2", 30, 3, 20, 2, "c_venus", (0, 1, 2)),
    ("Straight", "S_2 H_3 D_4 C_5 S_6", 30, 4, 30, 3, "c_saturn", (0, 1, 2, 3, 4)),
    ("Flush", "S_2 S_4 S_6 S_8 S_T", 35, 4, 15, 2, "c_jupiter", (0, 1, 2, 3, 4)),
    ("Full House", "S_3 H_3 D_3 C_2 S_2", 40, 4, 25, 2, "c_earth", (0, 1, 2, 3, 4)),
    ("Four of a Kind", "S_3 H_3 D_3 C_3 S_A", 60, 7, 30, 3, "c_mars", (0, 1, 2, 3)),
    ("Straight Flush", "S_2 S_3 S_4 S_5 S_6", 100, 8, 40, 4, "c_neptune", (0, 1, 2, 3, 4)),
    ("Five of a Kind", "S_3 H_3 D_3 C_3 S_3", 120, 12, 35, 3, "c_planet_x", (0, 1, 2, 3, 4)),
    ("Flush House", "S_3 S_3 S_3 S_2 S_2", 140, 14, 40, 4, "c_ceres", (0, 1, 2, 3, 4)),
    ("Flush Five", "S_3 S_3 S_3 S_3 S_3", 160, 16, 50, 3, "c_eris", (0, 1, 2, 3, 4)),
]


@pytest.mark.parametrize("name,keys,chips,mult,chip_gain,mult_gain,planet,indices", HANDS)
@pytest.mark.parametrize("levels", [0, 1, 3])
def test_all_hand_scores_and_planet_upgrades(name, keys, chips, mult, chip_gain, mult_gain, planet, indices, levels):
    state = create_run_state("WIKIRULE")
    cards = [card(key) for key in keys.split()]
    for _ in range(levels):
        add_consumable(state, planet)
        use_consumable(state, 0)
    result = score_hand(state, cards)
    nominal = {"A": 11, "K": 10, "Q": 10, "J": 10, "T": 10}
    card_chips = sum(nominal[cards[i].rank] if cards[i].rank in nominal else int(cards[i].rank) for i in indices)
    assert result.hand_name == name
    assert [id(c) for c in result.scoring_cards] == [id(cards[i]) for i in indices]
    assert result.chips == chips + levels * chip_gain + card_chips
    assert result.mult == mult + levels * mult_gain
    assert result.total == (chips + levels * chip_gain + card_chips) * (mult + levels * mult_gain)


@pytest.mark.parametrize(
    "keys,jokers,expected",
    [
        ("S_A H_2 D_3 C_4 S_5", (), "Straight"),
        ("S_T H_J D_Q C_K S_A", (), "Straight"),
        ("S_Q H_K D_A C_2 S_3", (), "High Card"),
        ("S_2 H_3 D_4 C_5", (), "High Card"),
        ("S_2 H_3 D_4 C_5", ("j_four_fingers",), "Straight"),
        ("S_2 H_4 D_6 C_8 S_T", ("j_shortcut",), "Straight"),
        ("S_2 H_5 D_7 C_9 S_J", ("j_shortcut",), "High Card"),
        ("S_9 S_8 H_7 S_6 S_3", ("j_four_fingers",), "Straight Flush"),
        ("S_8 S_8 H_8 S_6 S_6", ("j_four_fingers",), "Flush House"),
        ("S_8 S_8 H_8 S_8 S_8", ("j_four_fingers",), "Flush Five"),
        ("S_2 C_4 S_6 C_8 S_T", ("j_smeared",), "Flush"),
    ],
)
def test_hand_boundaries_and_passive_jokers(keys, jokers, expected):
    state = create_run_state("WIKIRULE")
    for key in jokers:
        add_joker(state, key)
    assert score_hand(state, [card(key) for key in keys.split()]).hand_name == expected


def test_royal_flush_uses_straight_flush_level():
    state = create_run_state("WIKIRULE")
    result = score_hand(state, [card("S_" + rank) for rank in "TJQKA"])
    assert (result.hand_name, result.display_name, result.total) == ("Straight Flush", "Royal Flush", 1208)


@pytest.mark.parametrize("reverse,expected_mult", [(False, 12), (True, 8)])
def test_played_cards_score_in_player_order(reverse, expected_mult):
    state = create_run_state("WIKIRULE")
    cards = [card("S_3", center_key="m_mult"), card("H_3", center_key="m_glass")]
    if reverse:
        cards.reverse()
    result = score_hand(state, cards)
    assert result.scoring_cards == cards
    assert result.mult == expected_mult


@pytest.mark.parametrize("edition,joker,expected_mult", [("holo", "j_photograph", 22), ("polychrome", "j_smiley", 6.5)])
def test_playing_card_edition_precedes_on_scored_jokers(edition, joker, expected_mult):
    state = create_run_state("WIKIRULE")
    add_joker(state, joker)
    result = score_hand(state, [card("S_K", edition_key=edition)])
    assert result.mult == expected_mult


@pytest.mark.parametrize("edition,joker,expected_mult", [("polychrome", "j_joker", 7.5), ("holo", "j_ramen", 22)])
def test_joker_edition_sequence(edition, joker, expected_mult):
    state = create_run_state("WIKIRULE")
    add_joker(state, joker, edition={edition: True})
    assert score_hand(state, [card("S_A")]).mult == expected_mult


@pytest.mark.parametrize("red,mime,triggers", [(False, False, 1), (True, False, 2), (False, True, 2), (True, True, 3)])
@pytest.mark.parametrize("steel,baron", [(True, False), (False, True), (True, True)])
def test_held_retriggers_stack_additively(red, mime, triggers, steel, baron):
    state = create_run_state("WIKIRULE")
    if mime:
        add_joker(state, "j_mime")
    if baron:
        add_joker(state, "j_baron")
    held = card("H_K", center_key="m_steel" if steel else "c_base", seal="Red" if red else None)
    result = score_hand(state, [card("S_A")], held_hand=[held])
    assert result.mult == pytest.approx(1.5 ** (triggers * (int(steel) + int(baron))))
    assert result.total == floor(16 * result.mult)


@pytest.mark.parametrize("edition", ["foil", "holo", "polychrome"])
def test_debuffed_card_keeps_hand_rank_but_loses_all_scoring_effects(edition):
    state = create_run_state("WIKIRULE")
    add_joker(state, "j_scholar")
    hiker = add_joker(state, "j_hiker")
    cards = [card("S_A", debuff=True, edition_key=edition, seal="Red", center_key="m_mult"), card("H_A", debuff=True)]
    result = score_hand(state, cards)
    assert (result.hand_name, result.chips, result.mult, result.total) == ("Pair", 10, 2, 20)
    assert all(c.perma_bonus == 0 for c in cards)
    assert hiker in state.jokers


@pytest.mark.parametrize("key", ["j_joker", "j_blueprint", "j_brainstorm"])
@pytest.mark.parametrize("edition", ["foil", "holo", "polychrome"])
def test_debuffed_jokers_do_not_score_or_copy(key, edition):
    state = create_run_state("WIKIRULE")
    if key == "j_brainstorm":
        add_joker(state, "j_joker")
    joker = add_joker(state, key, edition={edition: True})
    joker.debuff = True
    if key != "j_brainstorm":
        add_joker(state, "j_joker")
    result = score_hand(state, [card("S_A")])
    assert (result.chips, result.mult, result.total) == (16, 5, 80)


def test_stone_cards_score_in_place_without_rank_or_suit():
    state = create_run_state("WIKIRULE")
    cards = [card("S_A", center_key="m_stone"), card("H_3"), card("D_3")]
    result = score_hand(state, cards)
    assert result.hand_name == "Pair"
    assert result.scoring_cards == cards
    assert (result.chips, result.mult, result.total) == (66, 2, 132)


@pytest.mark.parametrize("debuff,expected", [(False, "Flush"), (True, "High Card")])
def test_wild_card_suit_is_disabled_by_debuff(debuff, expected):
    state = create_run_state("WIKIRULE")
    cards = [card("S_" + rank) for rank in "2468"] + [card("H_T", center_key="m_wild", debuff=debuff)]
    assert score_hand(state, cards).hand_name == expected


@pytest.mark.parametrize("stake", range(1, 9))
def test_stakes_apply_cumulatively(stake):
    state = create_run_state("WIKIRULE", stake=stake, deck_key="b_blue")
    assert state.round_resets.hands == 5
    assert state.round_resets.discards == (2 if stake >= 5 else 3)
    assert bool(state.modifiers.get("no_blind_reward", {}).get("Small")) == (stake >= 2)
    assert state.modifiers.get("scaling", 1) == (3 if stake >= 6 else 2 if stake >= 3 else 1)
    for threshold, flag in [
        (4, "enable_eternals_in_shop"),
        (7, "enable_perishables_in_shop"),
        (8, "enable_rentals_in_shop"),
    ]:
        assert bool(state.modifiers.get(flag)) == (stake >= threshold)


@pytest.mark.parametrize(
    "jokers,triggers",
    [
        ((), 1),
        (("j_mime",), 2),
        (("j_blueprint", "j_mime"), 3),
        (("j_mime", "j_brainstorm"), 3),
        (("j_blueprint", "j_blueprint", "j_mime"), 4),
        (("j_blueprint", "j_brainstorm"), 1),
    ],
)
@pytest.mark.parametrize("red", [False, True])
def test_gold_cards_retrigger_at_round_end(jokers, triggers, red):
    from pylatro.runtime import resolve_held_gold_cards

    state = create_run_state("WIKIRULE")
    for key in jokers:
        add_joker(state, key)
    held = [card("H_2", center_key="m_gold", seal="Red" if red else None)]
    before = state.dollars
    assert resolve_held_gold_cards(state, held) == (1, 3 * (triggers + int(red)))
    assert state.dollars == before + 3 * (triggers + int(red))
    assert resolve_held_gold_cards(state, held) == (0, 0)


@pytest.mark.parametrize("slots,expected", [(1, 1), (2, 2), (3, 3)])
def test_blue_seal_mime_copies_respect_consumable_capacity(slots, expected):
    from pylatro.runtime import resolve_blue_seals

    state = create_run_state("WIKIRULE")
    state.starting_params.consumable_slots = slots
    add_joker(state, "j_blueprint")
    add_joker(state, "j_mime")
    result = resolve_blue_seals(state, [card("S_2", seal="Blue")], "Pair")
    assert [c.center_key for c in result] == ["c_mercury"] * expected
    assert state.consumable_keys == ["c_mercury"] * expected


def test_debuffed_mime_and_copy_do_not_retrigger_round_end():
    from pylatro.runtime import resolve_held_gold_cards

    state = create_run_state("WIKIRULE")
    add_joker(state, "j_blueprint")
    mime = add_joker(state, "j_mime")
    mime.debuff = True
    assert resolve_held_gold_cards(state, [card("S_2", center_key="m_gold")]) == (1, 3)


@pytest.mark.parametrize(
    "tarot,enhancement,count",
    [
        ("c_magician", "m_lucky", 2),
        ("c_empress", "m_mult", 2),
        ("c_heirophant", "m_bonus", 2),
        ("c_lovers", "m_wild", 1),
        ("c_chariot", "m_steel", 1),
        ("c_justice", "m_glass", 1),
        ("c_devil", "m_gold", 1),
        ("c_tower", "m_stone", 1),
    ],
)
def test_tarot_enhancement_targets_and_preserves_other_modifiers(tarot, enhancement, count):
    from pylatro.consumables import can_use_consumable

    state = create_run_state("WIKIRULE")
    state.hand_cards = [card("S_2", center_key="m_mult", edition_key="foil", seal="Red") for _ in range(3)]
    add_consumable(state, tarot)
    assert not can_use_consumable(state, 0)
    assert not can_use_consumable(state, 0, hand_targets=range(count + 1))
    assert can_use_consumable(state, 0, hand_targets=range(count))
    use_consumable(state, 0, hand_targets=range(count))
    assert [c.center_key for c in state.hand_cards] == [enhancement] * count + ["m_mult"] * (3 - count)
    assert all(c.edition_key == "foil" and c.seal == "Red" for c in state.hand_cards)


@pytest.mark.parametrize(
    "tarot,suit,prefix",
    [("c_star", "Diamonds", "D"), ("c_moon", "Clubs", "C"), ("c_sun", "Hearts", "H"), ("c_world", "Spades", "S")],
)
def test_suit_tarots_change_up_to_three_cards(tarot, suit, prefix):
    state = create_run_state("WIKIRULE")
    state.hand_cards = [card("S_" + rank, center_key="m_bonus") for rank in "2345"]
    add_consumable(state, tarot)
    use_consumable(state, 0, hand_targets=[0, 1, 2])
    assert [c.suit for c in state.hand_cards[:3]] == [suit] * 3
    assert [c.front_key for c in state.hand_cards[:3]] == [prefix + "_" + rank for rank in "234"]
    assert state.hand_cards[3].front_key == "S_5"
    assert all(c.center_key == "m_bonus" for c in state.hand_cards)


def test_strength_wraps_ace_and_death_copies_rightmost_selected_card():
    state = create_run_state("WIKIRULE")
    state.hand_cards = [
        card("S_A"),
        card("H_K"),
        card("D_4", center_key="m_glass", seal="Red", edition_key="polychrome", perma_bonus=15),
    ]
    add_consumable(state, "c_strength")
    use_consumable(state, 0, hand_targets=[0, 1])
    assert [c.front_key for c in state.hand_cards] == ["S_2", "H_A", "D_4"]
    add_consumable(state, "c_death")
    original_uid = state.hand_cards[0].reward_uid
    use_consumable(state, 0, hand_targets=[2, 0])
    copied = state.hand_cards[0]
    assert (copied.front_key, copied.center_key, copied.seal, copied.edition_key, copied.perma_bonus) == (
        "D_4",
        "m_glass",
        "Red",
        "polychrome",
        15,
    )
    assert copied.reward_uid == original_uid
    assert copied is not state.hand_cards[2]


@pytest.mark.parametrize("cash,gain", [(-5, 0), (0, 0), (10, 10), (20, 20), (40, 20)])
def test_hermit_income_is_capped(cash, gain):
    state = create_run_state("WIKIRULE")
    state.dollars = cash
    add_consumable(state, "c_hermit")
    assert use_consumable(state, 0).dollars_delta == gain
    assert state.dollars == cash + gain


def test_black_hole_levels_hidden_hands_without_unlocking_them():
    state = create_run_state("WIKIRULE")
    add_consumable(state, "c_black_hole")
    use_consumable(state, 0)
    for name, _, chips, mult, chip_gain, mult_gain, _, _ in HANDS:
        assert state.hands[name]["level"] == 2
        assert state.hands[name]["chips"] == chips + chip_gain
        assert state.hands[name]["mult"] == mult + mult_gain
    assert not state.hands["Flush Five"]["visible"]


def test_identical_blueprints_follow_actual_positions():
    state = create_run_state("WIKIRULE")
    add_joker(state, "j_blueprint")
    add_joker(state, "j_blueprint")
    add_joker(state, "j_joker")
    assert score_hand(state, [card("S_A")]).mult == 13


@pytest.mark.parametrize(
    "deck,hands,discards,cash,hand_size,joker_slots,consumable_slots",
    [
        ("b_red", 4, 4, 4, 8, 5, 2),
        ("b_blue", 5, 3, 4, 8, 5, 2),
        ("b_yellow", 4, 3, 14, 8, 5, 2),
        ("b_green", 4, 3, 4, 8, 5, 2),
        ("b_black", 3, 3, 4, 8, 6, 2),
        ("b_magic", 4, 3, 4, 8, 5, 3),
        ("b_nebula", 4, 3, 4, 8, 5, 1),
        ("b_ghost", 4, 3, 4, 8, 5, 2),
        ("b_abandoned", 4, 3, 4, 8, 5, 2),
        ("b_checkered", 4, 3, 4, 8, 5, 2),
        ("b_zodiac", 4, 3, 4, 8, 5, 2),
        ("b_painted", 4, 3, 4, 10, 4, 2),
        ("b_anaglyph", 4, 3, 4, 8, 5, 2),
        ("b_plasma", 4, 3, 4, 8, 5, 2),
        ("b_erratic", 4, 3, 4, 8, 5, 2),
    ],
)
def test_all_deck_starting_parameters(deck, hands, discards, cash, hand_size, joker_slots, consumable_slots):
    state = create_run_state("WIKIRULE", deck_key=deck)
    assert (
        state.current_round.hands_left,
        state.current_round.discards_left,
        state.dollars,
        state.current_round.hand_size,
        state.starting_params.joker_slots,
        state.starting_params.consumable_slots,
    ) == (hands, discards, cash, hand_size, joker_slots, consumable_slots)
    assert len(state.deck_cards) == (40 if deck == "b_abandoned" else 52)
    if deck == "b_checkered":
        assert sum(c.suit == "Hearts" for c in state.deck_cards) == 26
        assert sum(c.suit == "Spades" for c in state.deck_cards) == 26
    if deck == "b_magic":
        assert state.consumable_keys == ["c_fool", "c_fool"]
        assert state.used_vouchers == {"v_crystal_ball": True}
    if deck == "b_nebula":
        assert state.used_vouchers == {"v_telescope": True}
    if deck == "b_ghost":
        assert state.consumable_keys == ["c_hex"]
        assert state.spectral_rate > 0
    if deck == "b_zodiac":
        assert state.used_vouchers == {"v_tarot_merchant": True, "v_planet_merchant": True, "v_overstock_norm": True}
    if deck == "b_plasma":
        assert state.starting_params.ante_scaling == 2


@pytest.mark.parametrize("hands,discards,expected", [(0, 0, 0), (2, 3, 7), (1, 0, 2), (0, 2, 2)])
def test_green_deck_pays_hands_and_discards_without_interest(hands, discards, expected):
    from pylatro.runtime import apply_end_of_round

    state = create_run_state("WIKIRULE", deck_key="b_green")
    state.dollars = 100
    state.current_round.hands_left = hands
    state.current_round.discards_left = discards
    assert apply_end_of_round(state)["dollars"] == expected
    assert state.dollars == 100 + expected


@pytest.mark.parametrize(
    "boss,suit", [("bl_club", "Clubs"), ("bl_goad", "Spades"), ("bl_window", "Diamonds"), ("bl_head", "Hearts")]
)
def test_suit_bosses_debuff_wild_cards_but_not_stone_cards(boss, suit):
    from pylatro import start_blind

    state = create_run_state("WIKIRULE")
    wild = card("S_A", center_key="m_wild")
    stone = card("S_K", center_key="m_stone")
    state.deck_cards.extend([wild, stone])
    state.draw_pile = list(state.deck_cards)
    state.round_resets.blind_choices["Boss"] = boss
    start_blind(state, "Boss")
    assert wild.debuff
    assert not stone.debuff
    for c in state.deck_cards:
        if c.center_key == "c_base":
            assert c.debuff == (c.suit == suit)


def test_plant_does_not_restore_stone_card_face_rank():
    from pylatro import start_blind

    state = create_run_state("WIKIRULE")
    stone = card("S_K", center_key="m_stone")
    state.deck_cards.append(stone)
    state.draw_pile = list(state.deck_cards)
    state.round_resets.blind_choices["Boss"] = "bl_plant"
    start_blind(state, "Boss")
    assert not stone.debuff
    assert all(c.debuff == (c.rank in "JQK") for c in state.deck_cards if c.center_key == "c_base")


@pytest.mark.parametrize(
    "boss,hands,discards,size", [("bl_water", 4, 0, 8), ("bl_needle", 1, 4, 8), ("bl_manacle", 4, 4, 7)]
)
def test_boss_resource_restrictions(boss, hands, discards, size):
    from pylatro import start_blind

    state = create_run_state("WIKIRULE")
    state.round_resets.blind_choices["Boss"] = boss
    start_blind(state, "Boss")
    assert (state.current_round.hands_left, state.current_round.discards_left, len(state.hand_cards)) == (
        hands,
        discards,
        size,
    )


@pytest.mark.parametrize("count,expected", [(1, 0), (4, 0), (5, 15)])
def test_psychic_requires_five_played_cards(count, expected):
    from pylatro import start_blind
    from pylatro.flow import play_cards

    state = create_run_state("WIKIRULE")
    state.round_resets.blind_choices["Boss"] = "bl_psychic"
    start_blind(state, "Boss")
    state.hand_cards = [card(k) for k in ["S_2", "H_4", "D_6", "C_8", "S_T"]]
    result = play_cards(state, range(count))
    # Five cards form High Card: (5 + 10) * 1.
    assert result.score.total == expected
    assert state.current_round.hands_left == 3


def test_flint_halves_base_values_before_card_bonuses():
    state = create_run_state("WIKIRULE")
    state.round_resets.blind = state.data.blinds["bl_flint"]
    result = score_hand(state, [card("S_A", center_key="m_mult")])
    # Base 5 chips rounds to 3, base 1 mult stays 1; then +11 chips, +4 mult.
    assert (result.chips, result.mult, result.total) == (14, 5, 70)


def test_pack_tarots_cannot_be_banked_for_later():
    from pylatro.models import PackState, ShopCard
    from pylatro.shop import can_claim_pack_card

    state = create_run_state("WIKIRULE")
    state.hand_cards = []
    offered = ShopCard(center_key="c_magician", card_type="Tarot", cost=0, base_cost=0)
    state.pack = PackState(
        booster_key="p_arcana_normal_1", state_name="TAROT_PACK", choices_remaining=1, cards=[offered]
    )
    assert not can_claim_pack_card(state, offered)


def test_perishable_expires_after_five_completed_rounds():
    from pylatro.runtime import apply_end_of_round

    state = create_run_state("WIKIRULE", stake=7)
    joker = add_joker(state, "j_joker", perishable=True)
    for _ in range(4):
        apply_end_of_round(state)
    assert not joker.debuff
    apply_end_of_round(state)
    assert joker.debuff
    assert joker.perish_tally == 0


def test_rental_charges_three_dollars_even_when_debuffed():
    from pylatro.runtime import apply_end_of_round

    state = create_run_state("WIKIRULE", stake=8)
    state.modifiers.update(no_interest=True, money_per_hand=0)
    state.dollars = 0
    joker = add_joker(state, "j_joker", rental=True)
    joker.debuff = True
    assert apply_end_of_round(state)["dollars"] == -3
    assert state.dollars == -3
