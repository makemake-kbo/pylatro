from __future__ import annotations

from copy import deepcopy

from ._helpers import _as_dict, _calculate_cost
from .models import ConsumableInstance, JokerInstance, RunState


def _copy_extra(extra: object) -> object:
    return deepcopy(extra)


def _sync_consumable_keys(state: RunState) -> None:
    state.consumable_keys = [consumable.center_key for consumable in state.consumables]


def create_consumable_instance(
    state: RunState,
    center_key: str,
    *,
    edition: dict[str, bool] | None = None,
) -> ConsumableInstance:
    center = state.data.centers[center_key]
    cost = _calculate_cost(state, center, edition=edition)
    return ConsumableInstance(
        center_key=center_key,
        edition=deepcopy(edition) if edition else None,
        sell_cost=max(1, cost // 2),
    )


def add_consumable(
    state: RunState,
    center_key: str,
    *,
    edition: dict[str, bool] | None = None,
) -> ConsumableInstance:
    consumable = create_consumable_instance(state, center_key, edition=edition)
    state.consumables.append(consumable)
    _sync_consumable_keys(state)
    return consumable


def remove_consumable(
    state: RunState,
    consumable: ConsumableInstance,
) -> None:
    for index, owned in enumerate(state.consumables):
        if owned is consumable:
            state.consumables.pop(index)
            _sync_consumable_keys(state)
            return


def create_joker_instance(
    state: RunState,
    center_key: str,
    *,
    edition: dict[str, bool] | None = None,
    eternal: bool = False,
    perishable: bool = False,
    rental: bool = False,
) -> JokerInstance:
    center = state.data.centers[center_key]
    config = _as_dict(center.get("config"))
    extra = _copy_extra(config.get("extra"))
    cost = _calculate_cost(state, center, edition=edition, rental=rental)
    instance = JokerInstance(
        center_key=center_key,
        edition=deepcopy(edition) if edition else None,
        eternal=eternal,
        perishable=perishable,
        perish_tally=state.perishable_rounds if perishable else None,
        rental=rental,
        mult=int(config.get("mult", 0) or 0),
        h_mult=int(config.get("h_mult", 0) or 0),
        h_x_mult=float(config.get("h_x_mult", 0) or 0),
        h_dollars=int(config.get("h_dollars", 0) or 0),
        p_dollars=int(config.get("p_dollars", 0) or 0),
        t_mult=int(config.get("t_mult", 0) or 0),
        t_chips=int(config.get("t_chips", 0) or 0),
        x_mult=float(config.get("Xmult", 1) or 1),
        h_size=int(config.get("h_size", 0) or 0),
        d_size=int(config.get("d_size", 0) or 0),
        extra=extra,
        type=str(config.get("type", "") or ""),
        hands_played_at_create=state.hands_played,
        sell_cost=max(1, cost // 2),
    )

    name = center["name"]
    if name == "Invisible Joker":
        instance.invis_rounds = 0
    elif name == "To Do List":
        visible_hands = [hand_name for hand_name, hand in state.hands.items() if hand["visible"]]
        if visible_hands:
            hand_name, _ = state.pseudorandom.pseudorandom_element(
                visible_hands,
                state.pseudorandom.pseudoseed("to_do"),
            )
            instance.to_do_poker_hand = hand_name
    elif name == "Caino":
        instance.caino_xmult = 1
    elif name == "Yorick" and isinstance(extra, dict):
        instance.yorick_discards = int(extra.get("discards", 0) or 0)
    elif name == "Loyalty Card" and isinstance(extra, dict):
        instance.loyalty_remaining = int(extra.get("every", 0) or 0)

    return instance


def add_joker(
    state: RunState,
    center_key: str,
    *,
    edition: dict[str, bool] | None = None,
    eternal: bool = False,
    perishable: bool = False,
    rental: bool = False,
) -> JokerInstance:
    joker = create_joker_instance(
        state,
        center_key,
        edition=edition,
        eternal=eternal,
        perishable=perishable,
        rental=rental,
    )
    state.jokers.append(joker)
    state.joker_keys.append(center_key)

    name = state.data.centers[center_key]["name"]
    if joker.d_size > 0:
        state.round_resets.discards += joker.d_size
        state.current_round.discards_left += joker.d_size
    if joker.h_size != 0:
        state.starting_params.hand_size += joker.h_size
        state.current_round.hand_size += joker.h_size
    if name == "Credit Card" and isinstance(joker.extra, int):
        state.bankrupt_at -= joker.extra
    elif name == "Chaos the Clown":
        state.current_round.free_rerolls += 1
        state.calculate_reroll_cost(skip_increment=True)
    elif name == "Oops! All 6s":
        for key, value in list(state.probabilities.items()):
            state.probabilities[key] = value * 2
    elif name == "To the Moon" and isinstance(joker.extra, int):
        state.interest_amount += joker.extra
    elif name == "Troubadour" and isinstance(joker.extra, dict):
        state.starting_params.hand_size += int(joker.extra.get("h_size", 0) or 0)
        state.round_resets.hands += int(joker.extra.get("h_plays", 0) or 0)
        state.current_round.hand_size += int(joker.extra.get("h_size", 0) or 0)
    elif name == "Stuntman" and isinstance(joker.extra, dict):
        state.starting_params.hand_size -= int(joker.extra.get("h_size", 0) or 0)
        state.current_round.hand_size -= int(joker.extra.get("h_size", 0) or 0)
    elif name == "Turtle Bean" and isinstance(joker.extra, dict):
        state.starting_params.hand_size += int(joker.extra.get("h_size", 0) or 0)
        state.current_round.hand_size += int(joker.extra.get("h_size", 0) or 0)
    elif edition and edition.get("negative"):
        state.starting_params.joker_slots += 1

    sync_all_jokers(state)
    return joker


def remove_joker(state: RunState, joker: JokerInstance) -> None:
    for index, owned in enumerate(state.jokers):
        if owned is not joker:
            continue
        state.jokers.pop(index)
        if index < len(state.joker_keys):
            state.joker_keys.pop(index)
        break
    else:
        return

    name = state.data.centers[joker.center_key]["name"]
    if joker.d_size > 0:
        state.round_resets.discards -= joker.d_size
        state.current_round.discards_left = max(0, state.current_round.discards_left - joker.d_size)
    if joker.h_size != 0:
        state.starting_params.hand_size -= joker.h_size
        state.current_round.hand_size -= joker.h_size
    if name == "Credit Card" and isinstance(joker.extra, int):
        state.bankrupt_at += joker.extra
    elif name == "Oops! All 6s":
        for key, value in list(state.probabilities.items()):
            state.probabilities[key] = value / 2
    elif name == "To the Moon" and isinstance(joker.extra, int):
        state.interest_amount = max(1, state.interest_amount - joker.extra)
    elif name == "Troubadour" and isinstance(joker.extra, dict):
        state.starting_params.hand_size -= int(joker.extra.get("h_size", 0) or 0)
        state.round_resets.hands -= int(joker.extra.get("h_plays", 0) or 0)
        state.current_round.hand_size -= int(joker.extra.get("h_size", 0) or 0)
    elif name == "Stuntman" and isinstance(joker.extra, dict):
        state.starting_params.hand_size += int(joker.extra.get("h_size", 0) or 0)
        state.current_round.hand_size += int(joker.extra.get("h_size", 0) or 0)
    elif name == "Turtle Bean" and isinstance(joker.extra, dict):
        state.starting_params.hand_size -= int(joker.extra.get("h_size", 0) or 0)
        state.current_round.hand_size -= int(joker.extra.get("h_size", 0) or 0)
    if joker.edition and joker.edition.get("negative"):
        state.starting_params.joker_slots -= 1
    sync_all_jokers(state)


def sync_joker_state(state: RunState, joker: JokerInstance, *, index: int | None = None) -> None:
    center = state.data.centers[joker.center_key]
    name = center["name"]

    if name == "Throwback" and isinstance(joker.extra, (int, float)):
        joker.x_mult = 1 + state.skips * float(joker.extra)
    elif name == "Driver's License":
        joker.driver_tally = sum(1 for card in state.deck_cards if card.center_key != "c_base")
    elif name == "Steel Joker" and isinstance(joker.extra, (int, float)):
        joker.steel_tally = sum(1 for card in state.deck_cards if card.center_key == "m_steel")
    elif name == "Stone Joker" and isinstance(joker.extra, (int, float)):
        joker.stone_tally = sum(1 for card in state.deck_cards if card.center_key == "m_stone")
    elif name == "Joker Stencil":
        joker.x_mult = state.starting_params.joker_slots - len(state.jokers)
        joker.x_mult += sum(1 for other in state.jokers if state.data.centers[other.center_key]["name"] == "Joker Stencil")
    elif name == "Cloud 9":
        joker.nine_tally = sum(1 for card in state.deck_cards if card.rank == "9")
    elif name == "Swashbuckler":
        joker.mult = sum(other.sell_cost for other in state.jokers if other is not joker)

    if center["name"] in {"Blueprint", "Brainstorm"}:
        other = None
        if center["name"] == "Brainstorm":
            other = state.jokers[0] if state.jokers else None
        else:
            if index is not None and index + 1 < len(state.jokers):
                other = state.jokers[index + 1]
        if other and other is not joker and state.data.centers[other.center_key].get("blueprint_compat"):
            joker.blueprint_compat = "compatible"
        else:
            joker.blueprint_compat = "incompatible"


def sync_all_jokers(state: RunState) -> None:
    for index, joker in enumerate(state.jokers):
        sync_joker_state(state, joker, index=index)
