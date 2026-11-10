"""Copies for single-play scoring probes, with untouched state shared."""

from copy import deepcopy

from pylatro._state_copy import copy_state_object


def _copy_hands(hands):
    """Per-hand stat dicts, sharing each hand's static ``example`` display list.

    Every other hand value is a scalar (levels, chips, mult, played counts,
    visibility); nothing reads or writes ``example`` after the data file loads.
    """
    return {name: dict(hand) for name, hand in hands.items()}


def _copy_cards_played(cards_played):
    """Per-rank ``{"suits": {suit: True}, "total": n}`` records."""
    copied = {}
    for rank, entry in cards_played.items():
        entry = dict(entry)
        entry["suits"] = dict(entry["suits"])
        copied[rank] = entry
    return copied


def copy_for_scoring(state):
    """Isolate everything a single ``play_cards`` call can mutate.

    Held/played cards and a possible refill are copied. Other deck cards are
    read only during scoring; the containing deck/pile lists are still copied
    so destruction and DNA cannot change the live collections. Boss history,
    vouchers, run modifiers and bans are only written when a blind or run is
    set up, so probes share them. This copy must not be reused for a later
    discard, blind reset, shop action or arbitrary consumable.
    """
    data = state.data
    hands = state.hands
    deck = state.deck_cards
    held = state.hand_cards
    played = state.play_cards
    draw = state.draw_pile
    discard = state.discard_pile
    memo = {
        id(data): data,
        id(hands): _copy_hands(hands),
        id(state.cards_played): _copy_cards_played(state.cards_played),
    }
    for shared in (state.bosses_used, state.used_vouchers, state.modifiers, state.banned_keys):
        memo[id(shared)] = shared
    # Every deck card maps to itself unless a probe can write to it.
    for card in deck:
        memo[id(card)] = card
    for cards in (held, played, draw[-max(3, state.current_round.hand_size):]):
        for card in cards:
            if memo.get(id(card), card) is card:
                copy_state_object(card, memo)
    # The five card piles dominate the object count and hold only cards, so
    # remap them in one pass instead of dispatching each card through deepcopy.
    lookup = memo.get
    for pile in (deck, draw, held, discard, played):
        memo[id(pile)] = [lookup(id(card)) or deepcopy(card, memo) for card in pile]
    return copy_state_object(state, memo)
