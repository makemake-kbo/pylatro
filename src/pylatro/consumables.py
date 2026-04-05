from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .instances import add_consumable, add_joker, remove_consumable, sync_all_jokers
from .models import ConsumableInstance, PlayingCard
from .pool import _pick_pool_key, create_card_spec, get_current_pool, poll_edition
from .runtime import (
    add_generated_consumable,
    add_generated_joker,
    add_playing_cards,
    apply_using_consumable,
    can_add_consumable,
    can_add_joker,
    consumable_limit,
    copy_playing_card,
    create_playing_card,
    joker_limit,
)
from .scoring import RANK_TO_ID, get_poker_hand_info


@dataclass(slots=True)
class UseConsumableResult:
    consumable_key: str
    destroyed_cards: list[PlayingCard] = field(default_factory=list)
    created_cards: list[PlayingCard] = field(default_factory=list)
    created_jokers: list[str] = field(default_factory=list)
    created_consumables: list[str] = field(default_factory=list)
    dollars_delta: int = 0


def can_use_consumable(
    state,
    consumable: int | str | ConsumableInstance,
    *,
    hand_targets: Iterable[int | PlayingCard] = (),
    joker_targets: Iterable[int] = (),
) -> bool:
    item = _resolve_consumable(state, consumable)
    center = state.data.centers[item.center_key]
    name = center["name"]
    config = center.get("config") or {}
    cards = _resolve_cards(state.hand_cards, hand_targets)
    jokers = _resolve_jokers(state, joker_targets)

    if name in {"The Hermit", "Temperance", "Black Hole"} or center.get("set") == "Planet":
        return True
    if name == "The Wheel of Fortune":
        return bool(_eligible_editionless_jokers(state))
    if name == "Ankh":
        return bool(state.jokers) and joker_limit(state) > 1
    if name == "Aura":
        return len(cards) == 1 and cards[0].edition_key is None
    if name in {"Ectoplasm", "Hex"}:
        return bool(_eligible_editionless_jokers(state))
    if name in {"The Emperor", "The High Priestess"}:
        # Using the card frees a slot, so allow if we're at capacity with the consumable in our area
        return len(state.consumables) < consumable_limit(state) or item in state.consumables
    if name == "The Fool":
        has_target = bool(state.last_tarot_planet and state.last_tarot_planet != "c_fool")
        return (len(state.consumables) < consumable_limit(state) or item in state.consumables) and has_target
    if name in {"Judgement", "The Soul", "Wraith"}:
        return len(state.jokers) < joker_limit(state)
    if name in {"Familiar", "Grim", "Incantation", "Immolate", "Sigil", "Ouija"}:
        return len(state.hand_cards) > 1
    if config.get("max_highlighted") is not None:
        required_min = int(config.get("min_highlighted", 1) or 1)
        required_max = int(config.get("max_highlighted", 0) or 0)
        return required_min <= len(cards) <= required_max
    return True


def use_consumable(
    state,
    consumable: int | str | ConsumableInstance,
    *,
    hand_targets: Iterable[int | PlayingCard] = (),
    joker_targets: Iterable[int] = (),
    copier: bool = False,
) -> UseConsumableResult:
    item = _resolve_consumable(state, consumable)
    center = state.data.centers[item.center_key]
    name = center["name"]
    cards = _resolve_cards(state.hand_cards, hand_targets)
    jokers = _resolve_jokers(state, joker_targets)
    if not can_use_consumable(state, item, hand_targets=hand_targets, joker_targets=joker_targets):
        raise ValueError(f"Consumable {name} is not currently usable")

    if not copier:
        _register_consumable_use(state, item.center_key)

    result = UseConsumableResult(consumable_key=item.center_key)

    if center.get("set") == "Planet":
        _level_up_hand(state, center["config"]["hand_type"])
    elif name == "Black Hole":
        for hand_name in state.hands:
            _level_up_hand(state, hand_name)
    elif name == "The Fool":
        created = add_generated_consumable(state, "Tarot_Planet", forced_key=state.last_tarot_planet, append="fool")
        if created:
            result.created_consumables.append(created.center_key)
    elif name == "The Hermit":
        amount = max(0, min(state.dollars, int(center["config"]["extra"])))
        state.dollars += amount
        result.dollars_delta += amount
    elif name == "Temperance":
        amount = min(sum(joker.sell_cost for joker in state.jokers), int(center["config"]["extra"]))
        state.dollars += amount
        result.dollars_delta += amount
    elif name in {"The Emperor", "The High Priestess"}:
        card_type = "Tarot" if name == "The Emperor" else "Planet"
        count_key = "tarots" if card_type == "Tarot" else "planets"
        for _ in range(min(int(center["config"][count_key]), consumable_limit(state) - len(state.consumables))):
            created = add_generated_consumable(
                state,
                card_type,
                append="emp" if card_type == "Tarot" else "pri",
            )
            if created:
                result.created_consumables.append(created.center_key)
    elif name in {"Judgement", "The Soul"}:
        if created := add_generated_joker(
            state,
            legendary=True if name == "The Soul" else None,
            append="sou" if name == "The Soul" else "jud",
        ):
            result.created_jokers.append(created.center_key)
    elif name == "Wraith":
        if created := add_generated_joker(state, rarity=0.99, append="wra"):
            result.created_jokers.append(created.center_key)
            state.dollars = 0
    elif name == "Ankh":
        chosen, _ = state.pseudorandom.pseudorandom_element(state.jokers, state.pseudorandom.pseudoseed("ankh_choice"))
        deletable = [joker for joker in state.jokers if not joker.eternal]
        from .instances import remove_joker

        for joker in list(deletable):
            if joker is not chosen:
                remove_joker(state, joker)
        duplicated = add_joker(
            state,
            chosen.center_key,
            edition=None if chosen.edition and chosen.edition.get("negative") else chosen.edition,
            eternal=chosen.eternal,
            perishable=chosen.perishable,
            rental=chosen.rental,
        )
        result.created_jokers.append(duplicated.center_key)
    elif name in {"The Wheel of Fortune", "Ectoplasm", "Hex"}:
        eligible = _eligible_editionless_jokers(state)
        should_apply = name in {"Ectoplasm", "Hex"} or (
            float(state.pseudorandom.pseudorandom("wheel_of_fortune")) < state.probabilities["normal"] / float(center["config"]["extra"])
        )
        if should_apply and eligible:
            chosen, _ = state.pseudorandom.pseudorandom_element(
                eligible,
                state.pseudorandom.pseudoseed(
                    "wheel_of_fortune" if name == "The Wheel of Fortune" else "ectoplasm" if name == "Ectoplasm" else "hex"
                ),
            )
            if name == "Ectoplasm":
                chosen.edition = {"negative": True}
                state.starting_params.hand_size -= state.ecto_minus
                state.current_round.hand_size -= state.ecto_minus
                state.ecto_minus += 1
            elif name == "Hex":
                chosen.edition = {"polychrome": True}
                from .instances import remove_joker

                for joker in list(state.jokers):
                    if joker is not chosen and not joker.eternal:
                        remove_joker(state, joker)
            else:
                chosen.edition = poll_edition(state, key="wheel_of_fortune", no_negative=True, guaranteed=True)
    elif name in {"Talisman", "Deja Vu", "Trance", "Medium"}:
        cards[0].seal = str(center["config"]["extra"])
    elif name == "Aura":
        edition = poll_edition(state, key="aura", no_negative=True, guaranteed=True)
        cards[0].edition_key = next(iter(edition)) if edition else None
    elif name == "Cryptid":
        source = cards[0]
        duplicates = [copy_playing_card(source) for _ in range(int(center["config"]["extra"]))]
        add_playing_cards(state, duplicates, area="hand")
        result.created_cards.extend(duplicates)
    elif name in {"Sigil", "Ouija"}:
        if name == "Sigil":
            suit, _ = state.pseudorandom.pseudorandom_element(
                ["S", "H", "D", "C"],
                state.pseudorandom.pseudoseed("sigil"),
            )
            suit_name = {"S": "Spades", "H": "Hearts", "D": "Diamonds", "C": "Clubs"}[suit]
            for card in state.hand_cards:
                card.suit = suit_name
                card.front_key = f"{suit}_{card.rank}"
        else:
            rank, _ = state.pseudorandom.pseudorandom_element(
                ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"],
                state.pseudorandom.pseudoseed("ouija"),
            )
            for card in state.hand_cards:
                card.rank = str(rank)
                card.front_key = f"{card.front_key[0]}_{rank}"
            state.starting_params.hand_size -= 1
            state.current_round.hand_size -= 1
    elif name == "The Hanged Man":
        destroyed = _destroy_selected_cards(state, cards)
        result.destroyed_cards.extend(destroyed)
    elif name in {"Familiar", "Grim", "Incantation"}:
        destroyed = [_destroy_random_hand_card(state, "random_destroy")]
        result.destroyed_cards.extend([card for card in destroyed if card])
        created = _create_spectral_cards(state, name, int(center["config"]["extra"]))
        result.created_cards.extend(created)
    elif name == "Immolate":
        destroyed = _immolate_cards(state, int(center["config"]["extra"]["destroy"]))
        result.destroyed_cards.extend(destroyed)
        amount = int(center["config"]["extra"]["dollars"])
        state.dollars += amount
        result.dollars_delta += amount
    elif name == "Strength":
        for card in cards:
            new_rank = "2" if card.rank == "A" else _rank_from_id(min(_card_id(card) + 1, 14))
            card.rank = new_rank
            card.front_key = f"{card.front_key[0]}_{new_rank}"
    elif name == "Death":
        source = _rightmost_card(state, cards)
        for card in cards:
            if card is source:
                continue
            _copy_into(source, card)
    elif name in {"The Magician", "The Empress", "The Hierophant", "The Lovers", "The Chariot", "Justice", "The Devil", "The Tower"}:
        center_key = str(center["config"]["mod_conv"])
        for card in cards:
            card.center_key = center_key
    elif name in {"The Star", "The Moon", "The Sun", "The World"}:
        suit_name = str(center["config"]["suit_conv"])
        prefix = suit_name[0]
        for card in cards:
            card.suit = suit_name
            card.front_key = f"{prefix}_{card.rank}"

    apply_using_consumable(state, item, destroyed_cards=result.destroyed_cards)
    sync_all_jokers(state)

    if not copier:
        remove_consumable(state, item)
    return result


def _resolve_consumable(state, consumable: int | str | ConsumableInstance) -> ConsumableInstance:
    if isinstance(consumable, ConsumableInstance):
        return consumable
    if isinstance(consumable, (int, np.integer)):
        return state.consumables[int(consumable)]
    for owned in state.consumables:
        if owned.center_key == consumable:
            return owned
    raise KeyError(f"Unknown consumable {consumable!r}")


def _resolve_cards(area: list[PlayingCard], items: Iterable[int | PlayingCard]) -> list[PlayingCard]:
    selected: list[PlayingCard] = []
    for item in items:
        if isinstance(item, int):
            selected.append(area[item])
        else:
            selected.append(item)
    seen: set[int] = set()
    deduped: list[PlayingCard] = []
    for card in area:
        if any(card is chosen for chosen in selected) and id(card) not in seen:
            deduped.append(card)
            seen.add(id(card))
    return deduped


def _resolve_jokers(state, indices: Iterable[int]) -> list:
    return [state.jokers[index] for index in indices]


def _register_consumable_use(state, center_key: str) -> None:
    center = state.data.centers[center_key]
    usage = state.consumeable_usage.setdefault(
        center_key,
        {"count": 0, "order": center["order"], "set": center["set"]},
    )
    usage["count"] = int(usage["count"]) + 1
    if center["set"] == "Tarot":
        state.consumeable_usage_total["tarot"] += 1
        state.consumeable_usage_total["tarot_planet"] += 1
        state.last_tarot_planet = center_key
    elif center["set"] == "Planet":
        state.consumeable_usage_total["planet"] += 1
        state.consumeable_usage_total["tarot_planet"] += 1
        state.last_tarot_planet = center_key
    elif center["set"] == "Spectral":
        state.consumeable_usage_total["spectral"] += 1
    state.consumeable_usage_total["all"] += 1


def _level_up_hand(state, hand_name: str, amount: int = 1) -> None:
    hand = state.hands[hand_name]
    hand["level"] = max(0, int(hand["level"]) + amount)
    hand["mult"] = max(int(hand["s_mult"]) + int(hand["l_mult"]) * (int(hand["level"]) - 1), 1)
    hand["chips"] = max(int(hand["s_chips"]) + int(hand["l_chips"]) * (int(hand["level"]) - 1), 0)


def _eligible_editionless_jokers(state) -> list:
    return [joker for joker in state.jokers if joker.edition is None]


def _copy_into(source: PlayingCard, target: PlayingCard) -> None:
    target.front_key = source.front_key
    target.suit = source.suit
    target.rank = source.rank
    target.center_key = source.center_key
    target.edition_key = source.edition_key
    target.seal = source.seal
    target.perma_bonus = source.perma_bonus


def _rightmost_card(state, cards: list[PlayingCard]) -> PlayingCard:
    index_map = {id(card): index for index, card in enumerate(state.hand_cards)}
    return max(cards, key=lambda card: index_map[id(card)])


def _destroy_selected_cards(state, cards: Iterable[PlayingCard]) -> list[PlayingCard]:
    destroyed: list[PlayingCard] = []
    for card in list(cards):
        _destroy_card(state, card)
        destroyed.append(card)
    return destroyed


def _destroy_card(state, card: PlayingCard) -> None:
    center = state.data.centers[card.center_key]
    if center.get("name") == "Glass Card":
        card.shattered = True
    else:
        card.destroyed = True
    for area in (state.hand_cards, state.draw_pile, state.discard_pile, state.play_cards, state.deck_cards):
        for index, candidate in enumerate(area):
            if candidate is card:
                area.pop(index)
                break


def _destroy_random_hand_card(state, seed_key: str) -> PlayingCard | None:
    if not state.hand_cards:
        return None
    card, _ = state.pseudorandom.pseudorandom_element(state.hand_cards, state.pseudorandom.pseudoseed(seed_key))
    _destroy_card(state, card)
    return card


def _create_spectral_cards(state, name: str, count: int) -> list[PlayingCard]:
    created: list[PlayingCard] = []
    enhanced_pool = [proto for proto in state.data.center_pools["Enhanced"] if proto["key"] != "m_stone"]
    for _ in range(count):
        if name == "Familiar":
            rank, _ = state.pseudorandom.pseudorandom_element(["J", "Q", "K"], state.pseudorandom.pseudoseed("familiar_create"))
            suit, _ = state.pseudorandom.pseudorandom_element(["S", "H", "D", "C"], state.pseudorandom.pseudoseed("familiar_create"))
        elif name == "Grim":
            rank = "A"
            suit, _ = state.pseudorandom.pseudorandom_element(["S", "H", "D", "C"], state.pseudorandom.pseudoseed("grim_create"))
        else:
            rank, _ = state.pseudorandom.pseudorandom_element(
                ["2", "3", "4", "5", "6", "7", "8", "9", "T"],
                state.pseudorandom.pseudoseed("incantation_create"),
            )
            suit, _ = state.pseudorandom.pseudorandom_element(["S", "H", "D", "C"], state.pseudorandom.pseudoseed("incantation_create"))
        center, _ = state.pseudorandom.pseudorandom_element(enhanced_pool, state.pseudorandom.pseudoseed("spe_card"))
        created.append(create_playing_card(state, front_key=f"{suit}_{rank}", center_key=center["key"]))
    add_playing_cards(state, created, area="hand")
    return created


def _immolate_cards(state, destroy_count: int) -> list[PlayingCard]:
    temp_hand = list(state.hand_cards)
    temp_hand = state.pseudorandom.pseudoshuffle(temp_hand, state.pseudorandom.pseudoseed("immolate"))
    destroyed = temp_hand[:destroy_count]
    return _destroy_selected_cards(state, destroyed)


def _card_id(card: PlayingCard) -> int:
    return RANK_TO_ID[card.rank]


def _rank_from_id(card_id: int) -> str:
    return {
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
        10: "T",
        11: "J",
        12: "Q",
        13: "K",
        14: "A",
    }[card_id]
