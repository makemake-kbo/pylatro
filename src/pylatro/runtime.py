from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from math import floor
from typing import TYPE_CHECKING

from .instances import add_consumable, add_joker, remove_consumable, remove_joker, sync_all_jokers
from .models import ConsumableInstance, JokerInstance, PlayingCard
from .pool import _pick_pool_key, create_card_spec, get_current_pool

if TYPE_CHECKING:
    from .models import RunState, ShopCard


def compute_interest(dollars: int, interest_cap: int, interest_amount: int) -> int:
    """Balatro interest: $`interest_amount` per $5 held, capped at `interest_cap` dollars.

    `interest_cap` is the *cash threshold* above which no further interest accrues
    (default 25, Seed Money 50, Money Tree 100), NOT a tier count. The number of paid
    tiers is therefore ``min(dollars // 5, interest_cap // 5)`` and each tier pays
    ``interest_amount`` (raised by To the Moon). Mirrors Balatro's
    ``min(floor(dollars/5), interest_cap/5) * interest_amount``.
    """
    if interest_amount <= 0 or dollars < 5:
        return 0
    tiers = min(dollars // 5, interest_cap // 5)
    return tiers * interest_amount


def joker_limit(state: RunState) -> int:
    return max(0, state.starting_params.joker_slots)


def consumable_limit(state: RunState) -> int:
    return max(0, state.starting_params.consumable_slots)


def can_add_joker(state: RunState, count: int = 1) -> bool:
    return len(state.jokers) + max(count, 0) <= joker_limit(state)


def can_add_consumable(state: RunState, count: int = 1) -> bool:
    return len(state.consumables) + max(count, 0) <= consumable_limit(state)


def create_playing_card(
    state: RunState,
    *,
    front_key: str,
    center_key: str = "c_base",
    edition_key: str | None = None,
    seal: str | None = None,
    perma_bonus: int = 0,
) -> PlayingCard:
    front = state.data.cards[front_key]
    return PlayingCard(
        front_key=front_key,
        suit=str(front["suit"]),
        rank=front_key[2],
        center_key=center_key,
        edition_key=edition_key,
        seal=seal,
        perma_bonus=perma_bonus,
    )


def copy_playing_card(card: PlayingCard) -> PlayingCard:
    return PlayingCard(
        front_key=card.front_key,
        suit=card.suit,
        rank=card.rank,
        center_key=card.center_key,
        edition_key=card.edition_key,
        seal=card.seal,
        perma_bonus=card.perma_bonus,
    )


def add_playing_cards(
    state: RunState,
    cards: Iterable[PlayingCard],
    *,
    area: str = "deck",
) -> list[PlayingCard]:
    created = list(cards)
    if not created:
        return []

    state.deck_cards.extend(created)
    if area == "hand":
        state.hand_cards.extend(created)
    elif area == "draw":
        state.draw_pile.extend(created)
    elif area == "discard":
        state.discard_pile.extend(created)
    apply_playing_card_added(state, created)
    sync_all_jokers(state)
    return created


def add_generated_consumable(
    state: RunState,
    card_type: str,
    *,
    forced_key: str | None = None,
    append: str | None = None,
    edition: dict[str, bool] | None = None,
    soulable: bool = True,
) -> ConsumableInstance | None:
    if not can_add_consumable(state):
        return None
    if forced_key is None:
        spec = create_card_spec(state, card_type, append=append, source="pack", soulable=soulable)
        forced_key = spec.center_key
        edition = edition if edition is not None else spec.edition
    return add_consumable(state, forced_key, edition=edition)


def create_joker_spec(
    state: RunState,
    *,
    forced_key: str | None = None,
    rarity: float | None = None,
    legendary: bool | None = None,
    append: str | None = None,
    source: str | None = "pack",
) -> ShopCard:
    if forced_key is None:
        pool, pool_key = get_current_pool(state, "Joker", rarity=rarity, legendary=legendary, append=append)
        forced_key = _pick_pool_key(state, pool, pool_key)
    return create_card_spec(state, "Joker", forced_key=forced_key, append=append, source=source, soulable=True)


def add_generated_joker(
    state: RunState,
    *,
    forced_key: str | None = None,
    rarity: float | None = None,
    legendary: bool | None = None,
    append: str | None = None,
    edition: dict[str, bool] | None = None,
    eternal: bool = False,
    perishable: bool = False,
    rental: bool = False,
    source: str | None = "pack",
) -> JokerInstance | None:
    if not can_add_joker(state):
        return None
    spec = create_joker_spec(
        state,
        forced_key=forced_key,
        rarity=rarity,
        legendary=legendary,
        append=append,
        source=source,
    )
    return add_joker(
        state,
        spec.center_key,
        edition=edition if edition is not None else spec.edition,
        eternal=eternal or spec.eternal,
        perishable=perishable or spec.perishable,
        rental=rental or spec.rental,
    )


def apply_playing_card_added(state: RunState, cards: Iterable[PlayingCard]) -> None:
    added = list(cards)
    if not added:
        return
    for joker in state.jokers:
        if joker.debuff:
            continue
        if state.data.centers[joker.center_key]["name"] == "Hologram" and isinstance(joker.extra, (int, float)):
            joker.x_mult += len(added) * float(joker.extra)
    sync_all_jokers(state)


def apply_open_booster(state: RunState) -> list[ConsumableInstance]:
    created: list[ConsumableInstance] = []
    for joker in state.jokers:
        if joker.debuff:
            continue
        center = state.data.centers[joker.center_key]
        if center["name"] != "Hallucination" or not isinstance(joker.extra, (int, float)):
            continue
        if not can_add_consumable(state):
            continue
        if float(state.pseudorandom.pseudorandom(f"halu{state.round_resets.ante}")) < (
            state.probabilities["normal"] / float(joker.extra)
        ):
            if consumable := add_generated_consumable(state, "Tarot", append="hal"):
                created.append(consumable)
    return created


def apply_reroll_shop(state: RunState) -> None:
    for joker in state.jokers:
        if joker.debuff:
            continue
        center = state.data.centers[joker.center_key]
        if center["name"] == "Flash Card" and isinstance(joker.extra, int):
            joker.mult += joker.extra
    sync_all_jokers(state)


def apply_skip_booster(state: RunState) -> None:
    for joker in state.jokers:
        if joker.debuff:
            continue
        center = state.data.centers[joker.center_key]
        if center["name"] == "Red Card" and isinstance(joker.extra, int):
            joker.mult += joker.extra
    sync_all_jokers(state)


def apply_using_consumable(
    state: RunState,
    consumeable: ConsumableInstance,
    *,
    destroyed_cards: Iterable[PlayingCard] = (),
) -> None:
    center = state.data.centers[consumeable.center_key]
    set_name = str(center["set"])
    destroyed = list(destroyed_cards)
    destroyed_glass = [card for card in destroyed if state.data.centers[card.center_key]["name"] == "Glass Card"]
    for joker in state.jokers:
        if joker.debuff:
            continue
        name = state.data.centers[joker.center_key]["name"]
        if name == "Glass Joker" and destroyed_glass and isinstance(joker.extra, (int, float)):
            joker.x_mult += len(destroyed_glass) * float(joker.extra)
        elif name == "Constellation" and set_name == "Planet" and isinstance(joker.extra, (int, float)):
            joker.x_mult += float(joker.extra)
    sync_all_jokers(state)


def apply_setting_blind(state: RunState) -> dict[str, list[str]]:
    created_jokers: list[str] = []
    created_consumables: list[str] = []
    blind = state.round_resets.blind or {}
    is_boss = bool(blind.get("boss"))
    for joker in list(state.jokers):
        if joker.debuff or joker.getting_sliced:
            continue
        center = state.data.centers[joker.center_key]
        name = center["name"]
        if name == "Chicot" and is_boss:
            state.blind_disabled = True
        elif name == "Madness" and not is_boss and isinstance(joker.extra, (int, float)):
            joker.x_mult += float(joker.extra)
            destructible = [
                other
                for other in state.jokers
                if other is not joker and not other.eternal and not other.getting_sliced
            ]
            if destructible:
                doomed, _ = state.pseudorandom.pseudorandom_element(
                    destructible,
                    state.pseudorandom.pseudoseed("madness"),
                )
                doomed.getting_sliced = True
                remove_joker(state, doomed)
        elif name == "Burglar" and isinstance(joker.extra, int):
            state.current_round.discards_left = 0
            state.current_round.hands_left += joker.extra
        elif name == "Riff-Raff":
            to_create = min(2, joker_limit(state) - len(state.jokers))
            for _ in range(max(0, to_create)):
                created = add_generated_joker(state, rarity=0.0, append="rif", source="pack")
                if created:
                    created_jokers.append(created.center_key)
        elif name == "Cartomancer":
            if consumable := add_generated_consumable(state, "Tarot", append="car"):
                created_consumables.append(consumable.center_key)
        elif name == "Ceremonial Dagger" and isinstance(joker.mult, int):
            my_index = state.jokers.index(joker)
            if my_index + 1 < len(state.jokers):
                other = state.jokers[my_index + 1]
                if not other.eternal and not other.getting_sliced:
                    other.getting_sliced = True
                    joker.mult += other.sell_cost * 2
                    remove_joker(state, other)
        elif name == "Marble Joker":
            front, front_key = state.pseudorandom.pseudorandom_element(
                state.data.cards,
                state.pseudorandom.pseudoseed("marb_fr"),
            )
            if isinstance(front, dict) and isinstance(front_key, str):
                add_playing_cards(state, [create_playing_card(state, front_key=front_key, center_key="m_stone")], area="draw")
        elif name == "Luchador" and is_boss:
            # No-op on blind set: Luchador disables the boss blind when *sold*,
            # not when the blind is selected (handled in sell_joker).
            pass
    sync_all_jokers(state)
    return {"jokers": created_jokers, "consumables": created_consumables}


def apply_end_shop(state: RunState) -> list[str]:
    # TODO: Perkeo creates Negative copies of consumables, but add_consumable
    # does not expand consumable_slots for Negative edition the way add_joker
    # does for joker_slots. This means Perkeo cannot create copies when all
    # consumable slots are full. Fixing this requires changes to how the agent
    # handles consumable slot tracking.
    created: list[str] = []
    for joker in state.jokers:
        if joker.debuff or state.data.centers[joker.center_key]["name"] != "Perkeo":
            continue
        if not state.consumables or not can_add_consumable(state):
            continue
        chosen, _ = state.pseudorandom.pseudorandom_element(
            state.consumables,
            state.pseudorandom.pseudoseed("perkeo"),
        )
        duplicated = add_consumable(
            state,
            chosen.center_key,
            edition={"negative": True},
        )
        created.append(duplicated.center_key)
    return created


def apply_end_of_round(state: RunState) -> dict[str, int | bool]:
    results: dict[str, int | bool] = {"dollars": 0, "saved": False}
    blind = state.round_resets.blind or {}
    is_boss = bool(blind.get("boss"))
    to_remove: list[JokerInstance] = []

    for joker in list(state.jokers):
        if joker.debuff:
            continue
        center = state.data.centers[joker.center_key]
        name = center["name"]
        if name == "Campfire" and is_boss and joker.x_mult > 1:
            joker.x_mult = 1
        elif name == "Rocket" and isinstance(joker.extra, dict):
            results["dollars"] = int(results["dollars"]) + int(joker.extra.get("dollars", 0) or 0)
            if is_boss:
                joker.extra["dollars"] = int(joker.extra.get("dollars", 0) or 0) + int(joker.extra.get("increase", 0) or 0)
        elif name == "Turtle Bean" and isinstance(joker.extra, dict):
            h_mod = int(joker.extra.get("h_mod", 0) or 0)
            h_size = int(joker.extra.get("h_size", 0) or 0)
            if h_size - h_mod <= 0:
                to_remove.append(joker)
            else:
                joker.extra["h_size"] = h_size - h_mod
                state.starting_params.hand_size -= h_mod
                state.current_round.hand_size -= h_mod
        elif name == "Invisible Joker" and isinstance(joker.extra, int):
            joker.invis_rounds += 1
        elif name == "Popcorn" and isinstance(joker.extra, int):
            if joker.mult - joker.extra <= 0:
                to_remove.append(joker)
            else:
                joker.mult -= joker.extra
        elif name == "To Do List":
            visible = [hand_name for hand_name, hand in state.hands.items() if hand["visible"] and hand_name != joker.to_do_poker_hand]
            if visible:
                hand_name, _ = state.pseudorandom.pseudorandom_element(
                    visible,
                    state.pseudorandom.pseudoseed("to_do"),
                )
                joker.to_do_poker_hand = hand_name
        elif name == "Egg" and isinstance(joker.extra, int):
            joker.extra_value += joker.extra
            joker.sell_cost = max(1, floor((int(center.get("cost", 1) or 1) + joker.extra_value) / 2))
        elif name == "Gift Card" and isinstance(joker.extra, int):
            for other in state.jokers:
                other.extra_value += joker.extra
                other.sell_cost = max(1, other.sell_cost + joker.extra)
            for consumable in state.consumables:
                consumable.extra_value += joker.extra
                consumable.sell_cost = max(1, consumable.sell_cost + joker.extra)
        elif name == "Hit the Road" and joker.x_mult > 1:
            joker.x_mult = 1
        elif name in {"Gros Michel", "Cavendish"} and isinstance(joker.extra, dict):
            odds = float(joker.extra.get("odds", 1) or 1)
            if float(state.pseudorandom.pseudorandom("cavendish" if name == "Cavendish" else "gros_michel")) < (
                state.probabilities["normal"] / odds
            ):
                if name == "Gros Michel":
                    state.pool_flags["gros_michel_extinct"] = True
                to_remove.append(joker)
        elif name == "Cloud 9" and joker.nine_tally > 0 and isinstance(joker.extra, int):
            results["dollars"] = int(results["dollars"]) + joker.extra * joker.nine_tally
        elif name == "Golden Joker" and isinstance(joker.extra, int):
            results["dollars"] = int(results["dollars"]) + joker.extra
        elif name == "Satellite" and isinstance(joker.extra, int):
            planet_count = sum(1 for usage in state.consumeable_usage.values() if usage.get("set") == "Planet")
            results["dollars"] = int(results["dollars"]) + joker.extra * planet_count
        elif name == "Delayed Gratification" and isinstance(joker.extra, int):
            if state.current_round.discards_used == 0 and state.current_round.discards_left > 0:
                results["dollars"] = int(results["dollars"]) + state.current_round.discards_left * joker.extra

    for joker in to_remove:
        remove_joker(state, joker)

    blind = state.round_resets.blind or {}
    blind_type = None
    for bt in ("Small", "Big", "Boss"):
        if state.round_resets.blind_states.get(bt) == "Defeated":
            blind_type = bt
            break

    no_reward_blinds = state.modifiers.get("no_blind_reward", {})
    if not isinstance(no_reward_blinds, dict):
        no_reward_blinds = {}
    if blind_type and not no_reward_blinds.get(blind_type):
        base_reward = int(blind.get("dollars", 0))
        results["dollars"] = int(results["dollars"]) + base_reward

    hands_left = max(0, state.current_round.hands_left)
    money_per_hand = int(state.modifiers.get("money_per_hand", 1))
    results["dollars"] = int(results["dollars"]) + hands_left * money_per_hand

    if not state.modifiers.get("no_interest"):
        interest = compute_interest(state.dollars, state.interest_cap, state.interest_amount)
        results["dollars"] = int(results["dollars"]) + max(0, interest)

    dollars = int(results["dollars"])
    if dollars:
        state.dollars += dollars
        state.current_round.round_dollars += dollars
    sync_all_jokers(state)
    return results


def check_mr_bones(state: RunState, round_score: int, blind_target: int) -> bool:
    if blind_target <= 0 or round_score < floor(blind_target * 0.25):
        return False
    to_remove: list[JokerInstance] = []
    for joker in state.jokers:
        if joker.debuff:
            continue
        if state.data.centers[joker.center_key]["name"] == "Mr. Bones":
            to_remove.append(joker)
    if not to_remove:
        return False
    for joker in to_remove:
        remove_joker(state, joker)
    state.current_round.hands_left += 1
    sync_all_jokers(state)
    return True


def sell_joker(state: RunState, index: int) -> JokerInstance:
    joker = state.jokers[index]
    name = state.data.centers[joker.center_key]["name"]

    invis_duplicate = None
    if name == "Luchador" and bool((state.round_resets.blind or {}).get("boss")):
        state.blind_disabled = True
    elif name == "Diet Cola":
        state.tags.append("tag_double")
    elif name == "Invisible Joker" and isinstance(joker.extra, int) and joker.invis_rounds >= joker.extra:
        others = [other for other in state.jokers if other is not joker]
        if others:
            chosen, _ = state.pseudorandom.pseudorandom_element(others, state.pseudorandom.pseudoseed("invisible"))
            invis_duplicate = {
                "key": chosen.center_key,
                "edition": deepcopy(chosen.edition) if chosen.edition and not chosen.edition.get("negative") else None,
                "eternal": chosen.eternal,
                "perishable": chosen.perishable,
                "rental": chosen.rental,
            }

    sold_value = joker.sell_cost
    remove_joker(state, joker)
    state.dollars += sold_value

    if invis_duplicate:
        add_joker(
            state,
            invis_duplicate["key"],
            edition=invis_duplicate["edition"],
            eternal=invis_duplicate["eternal"],
            perishable=invis_duplicate["perishable"],
            rental=invis_duplicate["rental"],
        )

    for other in state.jokers:
        if other.debuff:
            continue
        if state.data.centers[other.center_key]["name"] == "Campfire" and isinstance(other.extra, (int, float)):
            other.x_mult += float(other.extra)
    sync_all_jokers(state)
    return joker


def sell_consumable(state: RunState, index: int) -> ConsumableInstance:
    consumable = state.consumables[index]
    sold_value = consumable.sell_cost
    remove_consumable(state, consumable)
    state.dollars += sold_value
    for joker in state.jokers:
        if joker.debuff:
            continue
        if state.data.centers[joker.center_key]["name"] == "Campfire" and isinstance(joker.extra, (int, float)):
            joker.x_mult += float(joker.extra)
    sync_all_jokers(state)
    return consumable
