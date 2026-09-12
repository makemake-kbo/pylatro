from __future__ import annotations

from collections import Counter
from itertools import combinations

from ._helpers import _apply_voucher_to_run, _as_dict, _calculate_cost, _clear_shop_cards, _release_center
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
    removed = state.shop.cards
    state.shop.cards = []
    for card in removed:
        _release_center(state, card.center_key)
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


_PACK_RANK_VALUE = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}


def _pack_card_is_protected(card: PlayingCard) -> bool:
    return bool(
        card.center_key != "c_base"
        or card.edition_key
        or card.seal
        or card.perma_bonus > 0
        or card.times_played > 0
    )


def _pack_card_keep_scores(state: RunState) -> list[float]:
    ranks = Counter(card.rank for card in state.deck_cards)
    suits = Counter(card.suit for card in state.deck_cards)
    return [
        float(_PACK_RANK_VALUE.get(card.rank, 0))
        + 4.0 * ranks[card.rank]
        + 1.5 * suits[card.suit]
        + (80.0 if _pack_card_is_protected(card) else 0.0)
        for card in state.hand_cards
    ]


def _pack_suit_context(state: RunState) -> tuple[str, str, dict[str, float]]:
    """Return upcoming boss suit, confident suit anchor, and suit utilities."""
    from pylatro_agent.strategy_context import SUIT_TARGET_ORDER, compute_suit_target_utilities

    boss_key = state.round_resets.blind_choices.get("Boss", "")
    boss = state.data.blinds.get(boss_key, {}) if boss_key else {}
    # A disabled blind has already been reset at cash-out.  Do not let a stale
    # or manually carried disable make conversion into the *upcoming* boss suit
    # look safe.
    boss_suit = str((boss.get("debuff") or {}).get("suit") or "")
    utilities = dict(
        zip(
            SUIT_TARGET_ORDER,
            compute_suit_target_utilities(
                state.deck_cards,
                state.jokers,
                hand_play_counts={
                    name: int(hand.get("played", 0) or 0)
                    for name, hand in state.hands.items()
                },
                boss_debuff_suit=boss_suit,
                respect_card_debuff=False,
            ),
            strict=True,
        )
    )
    ordered = sorted(utilities, key=lambda suit: (utilities[suit], suit), reverse=True)
    best = ordered[0]
    runner_up = utilities[ordered[1]]
    # Stock is a four-way 0.10 tie.  Require both useful absolute evidence and
    # a clear margin before treating one suit as an established anchor.
    confident = best if utilities[best] >= 0.30 and utilities[best] - runner_up >= 0.10 else ""
    return boss_suit, confident, utilities


def _pack_confident_rank(rank_counts: Counter[str]) -> str:
    """Return a strict multiplicity anchor, never an ordering tie-break."""
    ordered = sorted(rank_counts, key=lambda rank: (rank_counts[rank], _PACK_RANK_VALUE.get(rank, 0)), reverse=True)
    if len(ordered) < 2:
        return ordered[0] if ordered else ""
    best = ordered[0]
    return best if rank_counts[best] >= rank_counts[ordered[1]] + 2 else ""


def _preferred_pack_consumable_targets(
    state: RunState,
    center_key: str,
    *,
    edition: dict[str, bool] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Choose one deterministic, non-destructive immediate pack use.

    Pack claims have one policy action per offered card, so targeting is an
    engine-level deterministic continuation. Masks and execution both call
    this function, which prevents a card from appearing claimable at full
    capacity unless the exact selected targets are still legal at execution.
    Death respects its rightmost-source rule by considering only source slots
    to the right of the overwritten target.
    """
    from .consumables import can_use_consumable
    from .instances import create_consumable_instance
    from .runtime import can_add_consumable

    if edition and edition.get("negative"):
        return None
    center = state.data.centers.get(center_key, {})
    name = str(center.get("name") or "")
    instance = create_consumable_instance(state, center_key, edition=edition)
    if name in {"The Emperor", "The High Priestess", "The Fool"} and not can_add_consumable(state):
        return None
    if can_use_consumable(state, instance, hand_targets=(), joker_targets=()):
        return (), ()

    config = center.get("config") or {}
    max_highlighted = config.get("max_highlighted")
    if max_highlighted is None and name != "Aura":
        return None
    min_size = 1 if name == "Aura" else int(config.get("min_highlighted", 1) or 1)
    max_size = 1 if name == "Aura" else int(max_highlighted or 0)
    max_size = min(max_size, 3, len(state.hand_cards))
    if max_size < min_size:
        return None

    keep_scores = _pack_card_keep_scores(state)
    hand = state.hand_cards
    rank_counts = Counter(card.rank for card in state.deck_cards)
    suit_counts = Counter(card.suit for card in state.deck_cards)
    boss_suit, confident_suit, _suit_utilities = _pack_suit_context(state)

    if name == "Death":
        candidates: list[tuple[float, tuple[int, int]]] = []
        for target in range(len(hand)):
            if _pack_card_is_protected(hand[target]):
                continue
            for source in range(target + 1, len(hand)):
                subset = (target, source)
                if not can_use_consumable(state, instance, hand_targets=subset):
                    continue
                structural_gain = (
                    5.0 * (rank_counts[hand[source].rank] - rank_counts[hand[target].rank])
                    + 2.0 * (suit_counts[hand[source].suit] - suit_counts[hand[target].suit])
                )
                gain = structural_gain + keep_scores[source] - keep_scores[target]
                if gain > 0.0:
                    candidates.append((gain, subset))
        return (max(candidates)[1], ()) if candidates else None

    if name == "The Hanged Man":
        confident_rank = _pack_confident_rank(rank_counts)
        eligible = [
            index
            for index, card in enumerate(hand)
            if not _pack_card_is_protected(card)
            and (not confident_rank or card.rank != confident_rank)
            and (not confident_suit or card.suit != confident_suit)
        ]
        eligible.sort(key=lambda index: (keep_scores[index], index))
        for size in range(min(max_size, len(eligible)), min_size - 1, -1):
            subset = tuple(sorted(eligible[:size]))
            if can_use_consumable(state, instance, hand_targets=subset):
                return subset, ()
        return None

    suit_target = {
        "The Star": "Diamonds",
        "The Moon": "Clubs",
        "The Sun": "Hearts",
        "The World": "Spades",
    }.get(name)
    if suit_target is not None:
        if suit_target == boss_suit:
            return None
        # A weak/tied deck has no suit plan to damage, so the offered Tarot is
        # itself a valid consolidation target.  Once observable deck/history/
        # Joker evidence establishes a strict anchor, only reinforce it.
        if confident_suit and suit_target != confident_suit:
            return None
        eligible = [
            index
            for index, card in enumerate(hand)
            if card.suit != suit_target and not _pack_card_is_protected(card)
        ]
    elif name == "Strength":
        rank_order = tuple(_PACK_RANK_VALUE)
        next_rank = {rank: rank_order[(i + 1) % len(rank_order)] for i, rank in enumerate(rank_order)}
        confident_rank = _pack_confident_rank(rank_counts)
        eligible = [
            index
            for index, card in enumerate(hand)
            if not _pack_card_is_protected(card)
            and (
                (confident_rank and next_rank[card.rank] == confident_rank)
                or (
                    not confident_rank
                    and rank_counts[next_rank[card.rank]] >= rank_counts[card.rank]
                )
            )
        ]
    elif name in {"Talisman", "Deja Vu", "Trance", "Medium"}:
        eligible = [index for index, card in enumerate(hand) if not card.seal]
    elif name == "Aura":
        eligible = [index for index, card in enumerate(hand) if card.edition_key is None]
    elif name == "Cryptid":
        # Cryptid is the one targeted consumable that should copy the asset we
        # value most.  Return from its descending order here; the common path
        # below deliberately sorts destructive/overwrite targets weakest-first.
        eligible = [
            index
            for index, card in enumerate(hand)
            if not card.destroyed and not card.shattered
        ]
        eligible.sort(key=lambda index: (-keep_scores[index], index))
        for index in eligible:
            subset = (index,)
            if can_use_consumable(state, instance, hand_targets=subset):
                return subset, ()
        return None
    else:
        # Enhancement Tarots should improve base cards, not overwrite an
        # existing enhancement or a high-value physical asset.
        eligible = [
            index
            for index, card in enumerate(hand)
            if card.center_key == "c_base" and not _pack_card_is_protected(card)
        ]
    eligible.sort(key=lambda index: (keep_scores[index], index))
    for size in range(min(max_size, len(eligible)), min_size - 1, -1):
        for subset in combinations(eligible, size):
            if can_use_consumable(state, instance, hand_targets=subset):
                return tuple(subset), ()
    return None


def legal_pack_consumable_targets(
    state: RunState, center_key: str, *, edition: dict[str, bool] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Find a legal immediate use without imposing a strategy on the player."""
    from .consumables import can_use_consumable
    from .instances import create_consumable_instance

    instance = create_consumable_instance(state, center_key, edition=edition)
    if can_use_consumable(state, instance):
        return (), ()
    center = state.data.centers[center_key]
    config = center.get("config") or {}
    maximum = 1 if center["name"] == "Aura" else int(config.get("max_highlighted", 0) or 0)
    minimum = int(config.get("min_highlighted", 1) or 1)
    for size in range(minimum, min(maximum, len(state.hand_cards)) + 1):
        for targets in combinations(range(len(state.hand_cards)), size):
            if can_use_consumable(state, instance, hand_targets=targets):
                return targets, ()
    return None


def pack_consumable_use_targets(
    state: RunState, center_key: str, *, edition: dict[str, bool] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Default agent continuation; callers may instead supply explicit targets."""
    legal = legal_pack_consumable_targets(state, center_key, edition=edition)
    if legal is None:
        return None
    return _preferred_pack_consumable_targets(state, center_key, edition=edition)


def claim_pack_card(
    state: RunState, index: int, *,
    hand_targets: tuple[int, ...] | None = None,
    joker_targets: tuple[int, ...] = (),
) -> ShopCard:
    if state.pack is None:
        raise ValueError("No active pack")
    card = state.pack.cards[index]
    if not can_claim_pack_card(state, card):
        raise ValueError("Pack card cannot be claimed at current capacity")
    center = state.data.centers[card.center_key]
    if center.get("consumeable"):
        from .consumables import can_use_consumable, use_consumable
        from .instances import create_consumable_instance

        instance = create_consumable_instance(state, card.center_key, edition=card.edition)
        targets = (hand_targets, joker_targets) if hand_targets is not None else pack_consumable_use_targets(
            state, card.center_key, edition=card.edition,
        )
        if targets is None and hand_targets is None:
            targets = legal_pack_consumable_targets(state, card.center_key, edition=card.edition)
        if targets is None or not can_use_consumable(
            state, instance, hand_targets=targets[0], joker_targets=targets[1],
        ):
            raise ValueError("Pack consumable requires legal immediate targets")
    card = state.pack.cards.pop(index)
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
        card.auto_used = True
        card.auto_used_hand_targets, card.auto_used_joker_targets = targets
        card.use_result = use_consumable(
            state, instance, hand_targets=targets[0], joker_targets=targets[1],
        )
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
        close_pack(state)
    return card


def can_claim_pack_consumable(
    state: RunState,
    center_key: str,
    *,
    edition: dict[str, bool] | None = None,
) -> bool:
    """Pack consumables must be usable now; inventory space cannot bank them."""
    return legal_pack_consumable_targets(state, center_key, edition=edition) is not None


def can_claim_pack_card(state: RunState, card: ShopCard) -> bool:
    """Mirror claim semantics without mutating the pack or inventory."""
    center = state.data.centers.get(card.center_key, {})
    if center.get("set") == "Joker":
        from .runtime import can_add_joker

        negative = bool(card.edition and card.edition.get("negative"))
        return bool(can_add_joker(state) or negative)
    if center.get("consumeable"):
        return can_claim_pack_consumable(state, card.center_key, edition=card.edition)
    return True


def open_booster_pack(state: RunState, index: int) -> PackState:
    if state.pack is not None:
        raise ValueError("Close the active pack first")
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
    if state_name in {"TAROT_PACK", "SPECTRAL_PACK"}:
        # Pack hands are dealt from the whole surviving deck, without blind or
        # first-hand Joker hooks. Mutations stay on these actual deck objects.
        state.hand_cards.clear()
        state.discard_pile.clear()
        state.play_cards.clear()
        state.draw_pile = state.pseudorandom.pseudoshuffle(
            list(state.deck_cards), state.pseudorandom.pseudoseed(f"pack{state.round_resets.ante}"),
        )
        for playing_card in state.draw_pile:
            playing_card.debuff = False
            playing_card.face_down = False
            playing_card.forced_selection = False
        state.current_round.hand_size = max(0, state.starting_params.hand_size)
        for _ in range(min(state.current_round.hand_size, len(state.draw_pile))):
            state.hand_cards.append(state.draw_pile.pop())
        state.pack.hand_drawn = True
    apply_open_booster(state)
    return state.pack


def close_pack(state: RunState, *, skipped: bool = False) -> None:
    if state.pack is None:
        return
    if skipped and state.pack.cards:
        apply_skip_booster(state)
    removed = state.pack.cards
    if state.pack.hand_drawn:
        state.draw_pile.extend(state.hand_cards)
        state.hand_cards.clear()
    state.pack = None
    for card in removed:
        _release_center(state, card.center_key)


def finish_shop(state: RunState) -> list[str]:
    result = apply_end_shop(state)
    _clear_shop_cards(state)
    return result


def sell_owned_joker(state: RunState, index: int):
    return sell_joker(state, index)


def sell_owned_consumable(state: RunState, index: int):
    return sell_consumable(state, index)


def redeem_voucher(state: RunState, voucher_key: str) -> None:
    # Shop redemption pays the displayed price, including discounts. Direct
    # grants without a shop offer retain their existing cost-free behavior.
    offer = next((voucher for voucher in state.shop.vouchers if voucher.center_key == voucher_key), None)
    if offer is not None:
        state.dollars -= offer.cost
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
