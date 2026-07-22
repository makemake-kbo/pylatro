from __future__ import annotations

from ._helpers import _apply_voucher_to_run, _as_dict, _calculate_cost
from .instances import add_consumable, add_joker
from .models import POKER_HANDS, PackState, PlayingCard, RunState, ShopCard, ShopState
from .pool import create_card_spec, get_pack, poll_edition
from .runtime import (
    add_playing_cards,
    apply_end_shop,
    apply_open_booster,
    apply_playing_card_added,
    apply_reroll_shop,
    apply_skip_booster,
    sell_consumable,
    sell_joker,
)


def create_shop_card(state: RunState) -> ShopCard:
    total_rate = state.joker_rate + state.tarot_rate + state.planet_rate + state.playing_card_rate + state.spectral_rate
    polled_rate = (
        float(state.pseudorandom.pseudorandom(state.pseudorandom.pseudoseed(f"cdt{state.round_resets.ante}")))
        * total_rate
    )
    running = 0.0

    card_types = [
        ("Joker", state.joker_rate),
        ("Tarot", state.tarot_rate),
        ("Planet", state.planet_rate),
        (
            "Enhanced"
            if state.used_vouchers.get("v_illusion") and float(state.pseudorandom.pseudorandom("illusion")) > 0.6
            else "Base",
            state.playing_card_rate,
        ),
        ("Spectral", state.spectral_rate),
    ]
    for card_type, value in card_types:
        if running < polled_rate <= running + value:
            card = create_card_spec(state, card_type, append="sho", source="shop")
            if (
                card_type in {"Base", "Enhanced"}
                and state.used_vouchers.get("v_illusion")
                and float(state.pseudorandom.pseudorandom("illusion")) > 0.8
            ):
                edition_poll = float(state.pseudorandom.pseudorandom("illusion"))
                if edition_poll > 1 - 0.15:
                    card.edition = {"polychrome": True}
                elif edition_poll > 0.5:
                    card.edition = {"holo": True}
                else:
                    card.edition = {"foil": True}
                card.cost = _calculate_cost(
                    state,
                    state.data.centers[card.center_key],
                    edition=card.edition,
                    rental=card.rental,
                )
            return card
        running += value
    raise RuntimeError("Shop card selection failed")


def populate_shop(state: RunState) -> ShopState:
    if not state.shop.cards:
        refresh_shop(state)

    if not state.shop.vouchers and state.current_voucher:
        voucher = create_card_spec(state, "Voucher", forced_key=state.current_voucher)
        voucher.shop_voucher = True
        state.shop.vouchers = [voucher]

    if not state.shop.boosters:
        boosters: list[ShopCard] = []
        while len(state.current_round.used_packs) < 2:
            state.current_round.used_packs.append("")
        for index in range(2):
            if not state.current_round.used_packs[index]:
                state.current_round.used_packs[index] = get_pack(state, "shop_pack")
            if state.current_round.used_packs[index] == "USED":
                continue
            booster = create_card_spec(state, "Booster", forced_key=state.current_round.used_packs[index])
            booster.booster_pos = index + 1
            boosters.append(booster)
        state.shop.boosters = boosters

    return state.shop


def refresh_shop(state: RunState) -> list[ShopCard]:
    state.shop.cards = [create_shop_card(state) for _ in range(state.shop.joker_max)]
    return state.shop.cards


def reroll_shop(state: RunState) -> list[ShopCard]:
    if state.current_round.reroll_cost > 0:
        state.dollars -= state.current_round.reroll_cost
    final_free = state.current_round.free_rerolls > 0
    state.current_round.free_rerolls = max(state.current_round.free_rerolls - 1, 0)
    state.calculate_reroll_cost(skip_increment=final_free)
    refreshed = refresh_shop(state)
    apply_reroll_shop(state)
    return refreshed


def buy_shop_card(state: RunState, index: int) -> ShopCard:
    card = state.shop.cards.pop(index)
    state.dollars -= card.cost
    center = state.data.centers[card.center_key]
    if center.get("set") in {"Default", "Enhanced"}:
        if not card.front_key:
            raise ValueError("Playing card purchases require a front key")
        front = state.data.cards[card.front_key]
        created = PlayingCard(
            front_key=card.front_key,
            suit=front["suit"],
            rank=card.front_key[2],
            center_key=card.center_key,
            edition_key=next(iter(card.edition)) if card.edition else None,
        )
        state.deck_cards.append(created)
        apply_playing_card_added(state, [created])
    elif center.get("consumeable"):
        add_consumable(state, card.center_key, edition=card.edition)
    else:
        add_joker(
            state,
            card.center_key,
            edition=card.edition,
            eternal=card.eternal,
            perishable=card.perishable,
            rental=card.rental,
        )
    return card


def claim_pack_card(state: RunState, index: int) -> ShopCard:
    if state.pack is None:
        raise ValueError("No active pack")
    card = state.pack.cards.pop(index)
    center = state.data.centers[card.center_key]
    if center.get("set") in {"Default", "Enhanced"}:
        if not card.front_key:
            raise ValueError("Playing card pack reward requires a front key")
        front = state.data.cards[card.front_key]
        created = PlayingCard(
            front_key=card.front_key,
            suit=front["suit"],
            rank=card.front_key[2],
            center_key=card.center_key,
            edition_key=next(iter(card.edition)) if card.edition else None,
            seal=card.seal,
        )
        add_playing_cards(state, [created], area="draw")
    elif center.get("consumeable"):
        # A consumable chosen from a booster pack is used immediately in
        # Balatro, not banked. Planets (and every other consumable that needs
        # no target selection: Black Hole, The Hermit, Judgement, ...) apply on
        # claim. Target-requiring tarots/spectrals can't be targeted from the
        # pack flow, and negative-edition cards are held rather than used, so
        # those fall back to the consumable inventory to be used later.
        from .consumables import can_use_consumable, use_consumable
        from .instances import create_consumable_instance

        negative = bool(card.edition and "negative" in card.edition)
        instance = create_consumable_instance(state, card.center_key, edition=card.edition)
        if not negative and can_use_consumable(state, instance):
            use_consumable(state, instance)
        else:
            add_consumable(state, card.center_key, edition=card.edition)
    else:
        add_joker(
            state,
            card.center_key,
            edition=card.edition,
            eternal=card.eternal,
            perishable=card.perishable,
            rental=card.rental,
        )

    state.pack.choices_remaining = max(0, state.pack.choices_remaining - 1)
    if state.pack.choices_remaining == 0:
        state.pack = None
    return card


def open_booster_pack(state: RunState, index: int) -> PackState:
    booster = state.shop.boosters.pop(index)
    state.dollars -= booster.cost
    if booster.booster_pos is not None:
        while len(state.current_round.used_packs) < booster.booster_pos:
            state.current_round.used_packs.append("")
        state.current_round.used_packs[booster.booster_pos - 1] = "USED"

    center = state.data.centers[booster.center_key]
    name = center["name"]
    cards: list[ShopCard] = []
    state_name = "SHOP"
    if "Arcana" in name:
        state_name = "TAROT_PACK"
    elif "Celestial" in name:
        state_name = "PLANET_PACK"
    elif "Spectral" in name:
        state_name = "SPECTRAL_PACK"
    elif "Standard" in name:
        state_name = "STANDARD_PACK"
    elif "Buffoon" in name:
        state_name = "BUFFOON_PACK"

    size = _as_dict(center.get("config")).get("extra", 0)
    choices = _as_dict(center.get("config")).get("choose", 1)
    for card_index in range(1, size + 1):
        if "Arcana" in name:
            if state.used_vouchers.get("v_omen_globe") and float(state.pseudorandom.pseudorandom("omen_globe")) > 0.8:
                card = create_card_spec(state, "Spectral", append="ar2", source="pack", soulable=True)
            else:
                card = create_card_spec(state, "Tarot", append="ar1", source="pack", soulable=True)
        elif "Celestial" in name:
            forced_key = None
            if state.used_vouchers.get("v_telescope") and card_index == 1:
                hand_name = None
                hand_tally = 0
                for name_key in POKER_HANDS:
                    hand = state.hands[name_key]
                    if hand["visible"] and hand["played"] > hand_tally:
                        hand_name = name_key
                        hand_tally = hand["played"]
                if hand_name is not None:
                    for proto in state.data.center_pools["Planet"]:
                        if _as_dict(proto.get("config")).get("hand_type") == hand_name:
                            forced_key = proto["key"]
                            break
            card = create_card_spec(state, "Planet", forced_key=forced_key, append="pl1", source="pack", soulable=True)
        elif "Spectral" in name:
            card = create_card_spec(state, "Spectral", append="spe", source="pack", soulable=True)
        elif "Standard" in name:
            base_type = (
                "Enhanced"
                if float(state.pseudorandom.pseudorandom(f"stdset{state.round_resets.ante}")) > 0.6
                else "Base"
            )
            card = create_card_spec(state, base_type, append="sta", source="pack", soulable=True)
            card.edition = poll_edition(
                state, key=f"standard_edition{state.round_resets.ante}", mod=2, no_negative=True
            )
            seal_poll = float(state.pseudorandom.pseudorandom(f"stdseal{state.round_resets.ante}"))
            if seal_poll > 1 - 0.02 * 10:
                seal_type = float(state.pseudorandom.pseudorandom(f"stdsealtype{state.round_resets.ante}"))
                if seal_type > 0.75:
                    card.seal = "Red"
                elif seal_type > 0.5:
                    card.seal = "Blue"
                elif seal_type > 0.25:
                    card.seal = "Gold"
                else:
                    card.seal = "Purple"
        elif "Buffoon" in name:
            card = create_card_spec(state, "Joker", append="buf", source="pack", soulable=True)
        else:
            raise RuntimeError(f"Unknown booster pack {name}")

        if card.edition or card.rental:
            card.cost = _calculate_cost(
                state,
                state.data.centers[card.center_key],
                edition=card.edition,
                rental=card.rental,
            )
        cards.append(card)

    state.pack = PackState(
        booster_key=booster.center_key,
        state_name=state_name,
        choices_remaining=choices,
        cards=cards,
        source_slot=booster.booster_pos,
    )
    apply_open_booster(state)
    return state.pack


def close_pack(state: RunState, *, skipped: bool = False) -> None:
    if state.pack is None:
        return
    if skipped and state.pack.cards:
        apply_skip_booster(state)
    state.pack = None


def finish_shop(state: RunState) -> list[str]:
    return apply_end_shop(state)


def sell_owned_joker(state: RunState, index: int):
    return sell_joker(state, index)


def sell_owned_consumable(state: RunState, index: int):
    return sell_consumable(state, index)


def redeem_voucher(state: RunState, voucher_key: str) -> None:
    state.used_vouchers[voucher_key] = True
    if state.current_voucher == voucher_key:
        state.current_voucher = None
    state.shop.vouchers = [voucher for voucher in state.shop.vouchers if voucher.center_key != voucher_key]
    _apply_voucher_to_run(state, voucher_key)
    state.round_resets.hands = state.starting_params.hands
    state.round_resets.discards = state.starting_params.discards
    state.round_resets.reroll_cost = state.starting_params.reroll_cost
    state.base_reroll_cost = state.starting_params.reroll_cost
    state.calculate_reroll_cost(skip_increment=True)
