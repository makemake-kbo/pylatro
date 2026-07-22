from pylatro import add_consumable, add_joker, create_run_state, score_hand, start_blind, use_consumable
from pylatro.models import PackState, PlayingCard, ShopCard
from pylatro.shop import claim_pack_card


def test_hanged_man_and_planet_use_update_joker_and_usage_state() -> None:
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_glass")
    start_blind(state)
    state.hand_cards[0].center_key = "m_glass"
    state.hand_cards[1].center_key = "m_glass"

    add_consumable(state, "c_hanged_man")
    use_consumable(state, 0, hand_targets=[0, 1])

    assert len(state.deck_cards) == 50
    assert state.jokers[0].x_mult == 2.5

    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_constellation")
    add_consumable(state, "c_mercury")
    use_consumable(state, 0)

    assert state.jokers[0].x_mult == 1.1
    assert state.consumeable_usage_total == {"tarot": 0, "planet": 1, "spectral": 0, "tarot_planet": 1, "all": 1}
    assert state.last_tarot_planet == "c_mercury"


def test_planet_claimed_from_pack_is_used_immediately_not_banked() -> None:
    # A consumable chosen from a booster pack is used on the spot in Balatro,
    # not stashed in the consumable inventory. A planet levels its hand right
    # away and fires on-use joker hooks (Constellation here).
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_constellation")
    before = state.hands["Pair"]["level"]
    state.pack = PackState(
        booster_key="p_celestial_normal_1",
        state_name="PLANET_PACK",
        choices_remaining=1,
        cards=[ShopCard(center_key="c_mercury", card_type="Planet", cost=0, base_cost=0)],
    )

    claim_pack_card(state, 0)

    assert state.hands["Pair"]["level"] == before + 1
    assert state.consumables == []
    assert state.jokers[0].x_mult == 1.1  # Constellation counted the use


def test_targeted_tarot_claimed_from_pack_falls_back_to_inventory() -> None:
    # The Magician needs highlighted cards, which the pack flow cannot supply
    # (no hand is drawn in the shop), so it banks for later use instead.
    state = create_run_state("AAAAAAAA")
    state.hand_cards.clear()
    magician = next(
        key
        for key, center in state.data.centers.items()
        if isinstance(center, dict) and center.get("name") == "The Magician"
    )
    state.pack = PackState(
        booster_key="p_arcana_normal_1",
        state_name="TAROT_PACK",
        choices_remaining=1,
        cards=[ShopCard(center_key=magician, card_type="Tarot", cost=0, base_cost=0)],
    )

    claim_pack_card(state, 0)

    assert [c.center_key for c in state.consumables] == [magician]


def test_observatory_reads_held_planets_during_scoring() -> None:
    cards = [
        PlayingCard(front_key="D_3", suit="Diamonds", rank="3"),
        PlayingCard(front_key="H_3", suit="Hearts", rank="3"),
    ]

    state = create_run_state("AAAAAAAA")
    baseline = score_hand(state, cards)
    assert baseline.total == 32

    state = create_run_state("AAAAAAAA")
    state.used_vouchers["v_observatory"] = True
    add_consumable(state, "c_mercury")
    buffed = score_hand(state, cards)
    assert (buffed.chips, buffed.mult, buffed.total) == (16.0, 3.0, 48)
