"""Copies for single-play scoring probes, with untouched deck cards shared."""

from copy import deepcopy


def copy_for_scoring(state):
    """Isolate everything a single ``play_cards`` call can mutate.

    Held/played cards and a possible refill are copied. Other deck cards are
    read only during scoring; the containing deck/pile lists are still copied
    so destruction and DNA cannot change the live collections. This copy must
    not be reused for a later discard, blind reset or arbitrary consumable.
    """
    mutable = {id(card) for card in state.hand_cards}
    mutable.update(id(card) for card in state.play_cards)
    refill = max(3, state.current_round.hand_size)
    mutable.update(id(card) for card in state.draw_pile[-refill:])
    memo = {id(state.data): state.data}
    memo.update((id(card), card) for card in state.deck_cards if id(card) not in mutable)
    return deepcopy(state, memo)
