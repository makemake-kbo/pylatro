from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .blind import select_blind
from .instances import joker_is_expired, set_joker_debuff, sync_all_jokers
from .models import PlayingCard
from .runtime import add_generated_consumable, apply_playing_card_added, apply_setting_blind
from .scoring import (
    POKER_HAND_ORDER,
    RANK_TO_ID,
    RANK_TO_NOMINAL,
    SUIT_TO_NOMINAL,
    ScoreResult,
    _is_suit,
    _level_up_hand,
    get_poker_hand_info,
    resolve_after_hand,
    score_hand,
)
from .scoring import (
    _is_face as _scoring_is_face,
)

if TYPE_CHECKING:
    from .models import JokerInstance, RunState


@dataclass(slots=True)
class DiscardResult:
    discarded: list[PlayingCard]
    destroyed: list[PlayingCard]
    drawn: list[PlayingCard]
    purple_seals_activated: int = 0
    generated_consumables: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PlayResult:
    score: ScoreResult
    played: list[PlayingCard]
    destroyed: list[PlayingCard]
    drawn: list[PlayingCard]
    blue_seals_activated: int = 0
    blue_planets_generated: list[str] = field(default_factory=list)
    held_gold_count: int = 0
    held_gold_payout: int = 0


def start_blind(state: RunState, blind_type: str | None = None) -> list[PlayingCard]:
    selected_type = blind_type or state.blind_on_deck or "Small"
    state.blind_on_deck = selected_type
    select_blind(state, selected_type)
    _reset_for_blind(state, selected_type)
    apply_setting_blind(state)
    _fold_areas_back_into_deck(state)
    state.draw_pile = state.pseudorandom.pseudoshuffle(
        state.draw_pile,
        state.pseudorandom.pseudoseed(f"nr{state.round_resets.ante}"),
    )
    return draw_to_hand(state)


def draw_to_hand(state: RunState, count: int | None = None) -> list[PlayingCard]:
    if state.current_round.hand_size <= 0 and not state.hand_cards:
        return []

    blind_name = _blind_name(state)
    if blind_name == "The Serpent" and not state.blind_disabled and (
        state.current_round.hands_played > 0 or state.current_round.discards_used > 0
    ):
        hand_space = min(len(state.draw_pile), 3)
    else:
        target = count if count is not None else max(0, state.current_round.hand_size - len(state.hand_cards))
        hand_space = min(len(state.draw_pile), target)

    drawn: list[PlayingCard] = []
    for _ in range(hand_space):
        card = state.draw_pile.pop()
        card.discarded = False
        card.forced_selection = False
        card.face_down = _stay_flipped(state, card)
        state.hand_cards.append(card)
        drawn.append(card)

    _sort_hand(state)

    if (
        drawn
        and not state.current_round.first_hand_drawn
        and state.current_round.hands_played == 0
        and state.current_round.discards_used == 0
    ):
        _first_hand_drawn(state)
        _sort_hand(state)
        state.current_round.first_hand_drawn = True

    _drawn_to_hand(state)
    return drawn


def discard_cards(
    state: RunState,
    cards: Iterable[PlayingCard | int],
    *,
    hook: bool = False,
) -> DiscardResult:
    selected = _resolve_cards(state.hand_cards, cards)
    if not selected:
        return DiscardResult(discarded=[], destroyed=[], drawn=[])
    if state.current_round.discards_left <= 0 and not hook:
        raise ValueError("No discards remaining")

    if state.current_round.discards_used <= 0 and not hook:
        _pre_discard(state, selected)

    discarded: list[PlayingCard] = []
    destroyed: list[PlayingCard] = []
    generated_consumables: list[str] = []
    purple_seals_activated = 0

    face_tally = sum(1 for card in selected if _is_face(state, card))
    for index, card in enumerate(selected):
        last = index == len(selected) - 1
        removed = False

        for joker in list(state.jokers):
            if joker.debuff:
                continue
            removed = _discard_effect(state, joker, card, selected, last=last, face_tally=face_tally) or removed

        _remove_exact(state.hand_cards, card)
        if removed:
            _destroy_playing_card(state, card)
            destroyed.append(card)
        else:
            card.discarded = True
            card.face_down = False
            state.discard_pile.append(card)
            discarded.append(card)
            if card.seal == "Purple" and not card.debuff:
                purple_seals_activated += 1
                generated = add_generated_consumable(state, "Tarot", append="purple_seal")
                if generated is not None:
                    generated_consumables.append(generated.center_key)

    _apply_removed_card_effects(state, destroyed)
    sync_all_jokers(state)

    drawn: list[PlayingCard] = []
    if not hook:
        state.current_round.discards_left = max(0, state.current_round.discards_left - 1)
        state.current_round.discards_used += 1
        drawn = draw_to_hand(state)

    return DiscardResult(
        discarded=discarded,
        destroyed=destroyed,
        drawn=drawn,
        purple_seals_activated=purple_seals_activated,
        generated_consumables=generated_consumables,
    )


def play_cards(state: RunState, cards: Iterable[PlayingCard | int]) -> PlayResult:
    selected = _resolve_cards(state.hand_cards, cards)
    if not selected:
        raise ValueError("No cards selected")
    if state.current_round.hands_left <= 0:
        raise ValueError("No hands remaining")

    first_hand = state.current_round.hands_played == 0
    state.current_round.hands_left = max(0, state.current_round.hands_left - 1)
    if _blind_name(state) in {"The Fish", "Crimson Heart"} and not state.blind_disabled:
        state.blind_prepped = True

    for card in selected:
        removed = _remove_exact(state.hand_cards, card)
        if not removed:
            continue
        card.times_played += 1
        card.played_this_ante = True
        card.discarded = False
        card.forced_selection = False
        state.play_cards.append(card)

    # Upstream moves selected cards to G.play before press_play. The Hook
    # discards from the remaining held cards, never from the selected play.
    _press_play(state, selected)

    # Check debuff_hand before scoring
    play_list = list(state.play_cards)
    hand_name_pre, _, poker_hands_pre, _ = get_poker_hand_info(state, play_list)
    hand_debuffed = _debuff_hand(state, play_list, hand_name_pre, poker_hands_pre)

    held_hand = list(state.hand_cards)
    result = score_hand(state, list(state.play_cards), held_hand=held_hand, hand_debuffed=hand_debuffed, precomputed_hand_name=hand_name_pre, precomputed_poker_hands=poker_hands_pre)
    state.hands_played += 1
    state.current_round.hands_played += 1

    if first_hand and len(selected) == 1 and _card_id(selected[0]) == 6 and state.has_joker("Sixth Sense"):
        selected[0].destroyed = True
        add_generated_consumable(state, "Spectral", append="sixth")

    destroyed: list[PlayingCard] = []
    played: list[PlayingCard] = []
    while state.play_cards:
        card = state.play_cards.pop(0)
        if card.destroyed or card.shattered:
            _destroy_playing_card(state, card)
            destroyed.append(card)
        else:
            card.face_down = False
            state.discard_pile.append(card)
            played.append(card)

    drawn: list[PlayingCard] = []
    if not state.hand_cards and state.draw_pile:
        drawn = draw_to_hand(state)

    resolve_after_hand(state)

    return PlayResult(score=result, played=played, destroyed=destroyed, drawn=drawn)


def _reset_for_blind(state: RunState, blind_type: str) -> None:
    state.subhash = f"{state.round_resets.ante}{'S' if blind_type == 'Small' else 'B' if blind_type == 'Big' else 'L'}"
    state.blind_disabled = False
    state.blind_triggered = False
    state.blind_prepped = False
    state.current_round.first_hand_drawn = False
    state.current_round.held_gold_settled = False
    state.current_round.hand_size = max(
        0,
        state.starting_params.hand_size + int(state.round_resets.temp_handsize or 0),
    )

    blind_name = _blind_name(state)
    blind = state.round_resets.blind or {}
    state.current_round.blind_removed_hands = 0
    state.current_round.blind_removed_discards = 0
    if blind_name == "The Water":
        state.current_round.blind_removed_discards = state.current_round.discards_left
        state.current_round.discards_left = 0
    elif blind_name == "The Needle":
        state.current_round.blind_removed_hands = state.current_round.hands_left - 1
        state.current_round.hands_left = 1
    elif blind_name == "The Manacle":
        state.current_round.hand_size = max(0, state.current_round.hand_size - 1)
    elif blind_name == "The Eye":
        state.eye_hands = {h: False for h in POKER_HAND_ORDER}
    elif blind_name == "The Mouth":
        state.mouth_only_hand = False
    elif blind_name == "Amber Acorn" and state.jokers:
        # Upstream flips the jokers face-down and shuffles them; they keep
        # scoring. The flip is visual-only, so this port only shuffles.
        if len(state.jokers) > 1:
            state.jokers = state.pseudorandom.pseudoshuffle(
                list(state.jokers),
                state.pseudorandom.pseudoseed("aajk"),
            )
            # joker_keys must stay index-aligned with jokers: remove_joker pops
            # the same index from both lists.
            state.joker_keys = [joker.center_key for joker in state.jokers]

    for card in state.deck_cards:
        card.discarded = False
        card.forced_selection = False
        card.face_down = False
        _debuff_card(state, card)
    for joker in state.jokers:
        set_joker_debuff(state, joker, False)

    _reset_round_cards(state)


def _reset_round_cards(state: RunState) -> None:
    """Re-roll the per-round target cards for jokers whose target changes each round.

    Four jokers lock onto a randomly chosen rank/suit at the start of every round:
      - The Idol: X Mult when its exact rank+suit card is played.
      - Mail-In Rebate: dollars per discarded card of its rank.
      - Ancient Joker: X Mult per played card of its suit.
      - Castle: gains chips per discarded card of its suit.
    Each draws an independent random card from the current deck. The Idol stores the
    rank id (scoring matches by id) plus the rank/suit strings (the heuristic matches
    on those); the others store just what their effect needs.
    """
    deck = state.deck_cards
    if not deck:
        return
    ante = state.round_resets.ante
    pick = state.pseudorandom.pseudorandom_element
    seed = state.pseudorandom.pseudoseed

    # Rank-targeting (Mail-In Rebate) may land on any card, including Stone.
    mail, _ = pick(deck, seed(f"mail{ante}"))
    state.current_round.mail_card = {"rank": mail.rank, "id": RANK_TO_ID[mail.rank]}

    # Suit-targeting cards skip Stone cards, which have no suit.
    suited = [
        card for card in deck
        if state.data.centers[card.center_key].get("effect") != "Stone Card"
    ]
    if not suited:
        return
    idol, _ = pick(suited, seed(f"idol{ante}"))
    state.current_round.idol_card = {"rank": idol.rank, "suit": idol.suit, "id": RANK_TO_ID[idol.rank]}
    ancient, _ = pick(suited, seed(f"ancient{ante}"))
    state.current_round.ancient_card = {"suit": ancient.suit}
    castle, _ = pick(suited, seed(f"castle{ante}"))
    state.current_round.castle_card = {"suit": castle.suit}


def _fold_areas_back_into_deck(state: RunState) -> None:
    if state.hand_cards:
        state.discard_pile.extend(state.hand_cards)
        state.hand_cards.clear()
    if state.play_cards:
        for card in state.play_cards:
            if not card.destroyed and not card.shattered:
                state.discard_pile.append(card)
        state.play_cards.clear()
    if state.discard_pile:
        state.draw_pile = list(state.discard_pile) + list(state.draw_pile)
        state.discard_pile.clear()
    state.draw_pile = [card for card in state.draw_pile if not card.destroyed and not card.shattered]


def _resolve_cards(area: list[PlayingCard], cards: Iterable[PlayingCard | int]) -> list[PlayingCard]:
    selected: list[PlayingCard] = []
    seen: set[int] = set()
    by_index: set[int] = set()
    by_identity: set[int] = set()
    for item in cards:
        if isinstance(item, int):
            if item < 0 or item >= len(area):
                raise IndexError(f"Card index {item} out of range")
            by_index.add(item)
        else:
            by_identity.add(id(item))
    for index, card in enumerate(area):
        if index in by_index or id(card) in by_identity:
            if id(card) in seen:
                continue
            selected.append(card)
            seen.add(id(card))
    return selected


def _sort_hand(state: RunState) -> None:
    state.hand_cards.sort(key=lambda card: _card_nominal(state, card), reverse=True)


def _stay_flipped(state: RunState, card: PlayingCard) -> bool:
    if state.blind_disabled:
        return False
    blind_name = _blind_name(state)
    if blind_name == "The Wheel":
        return float(state.pseudorandom.pseudorandom("wheel")) < state.probabilities["normal"] / 7
    if blind_name == "The House" and state.current_round.hands_played == 0 and state.current_round.discards_used == 0:
        return True
    if blind_name == "The Mark" and _is_face(state, card):
        return True
    return blind_name == "The Fish" and state.blind_prepped


def _drawn_to_hand(state: RunState) -> None:
    if state.blind_disabled:
        state.blind_prepped = False
        return

    blind_name = _blind_name(state)
    if blind_name == "Cerulean Bell" and not any(card.forced_selection for card in state.hand_cards) and state.hand_cards:
        for card in state.hand_cards:
            card.forced_selection = False
        forced, _ = state.pseudorandom.pseudorandom_element(
            state.hand_cards,
            state.pseudorandom.pseudoseed("cerulean_bell"),
        )
        forced.forced_selection = True

    if blind_name == "Crimson Heart" and state.blind_prepped and state.jokers:
        available: list[JokerInstance] = []
        for joker in state.jokers:
            if not joker_is_expired(joker) and (not joker.debuff or len(available) < 2):
                available.append(joker)
            set_joker_debuff(state, joker, False)
        if available:
            chosen, _ = state.pseudorandom.pseudorandom_element(
                available,
                state.pseudorandom.pseudoseed("crimson_heart"),
            )
            set_joker_debuff(state, chosen, True)

    state.blind_prepped = False


def _first_hand_drawn(state: RunState) -> None:
    for joker in state.jokers:
        if joker.debuff or state.data.centers[joker.center_key]["name"] != "Certificate":
            continue
        front, front_key = state.pseudorandom.pseudorandom_element(
            state.data.cards,
            state.pseudorandom.pseudoseed("cert_fr"),
        )
        seal_roll = float(state.pseudorandom.pseudorandom("certsl"))
        seal = "Red" if seal_roll > 0.75 else "Blue" if seal_roll > 0.5 else "Gold" if seal_roll > 0.25 else "Purple"
        new_card = PlayingCard(
            front_key=str(front_key),
            suit=str(front["suit"]),
            rank=str(front_key)[2],
            seal=seal,
        )
        state.deck_cards.append(new_card)
        state.hand_cards.append(new_card)
        apply_playing_card_added(state, [new_card])


def _pre_discard(state: RunState, selected: list[PlayingCard]) -> None:
    for joker in state.jokers:
        if joker.debuff or state.data.centers[joker.center_key]["name"] != "Burnt Joker":
            continue
        hand_name, _, _, _ = get_poker_hand_info(state, selected)
        _level_up_hand(state, hand_name)


def _discard_effect(
    state: RunState,
    joker: JokerInstance,
    card: PlayingCard,
    selected: list[PlayingCard],
    *,
    last: bool,
    face_tally: int,
) -> bool:
    name = state.data.centers[joker.center_key]["name"]
    if name == "Yorick" and isinstance(joker.extra, dict):
        if joker.yorick_discards <= 1:
            joker.yorick_discards = int(joker.extra.get("discards", 0) or 0)
            joker.x_mult += float(joker.extra.get("xmult", 0) or 0)
        else:
            joker.yorick_discards -= 1
    elif (
        name == "Trading Card"
        and state.current_round.discards_used <= 0
        and len(selected) == 1
        and not joker.debuff
    ):
        state.dollars += int(joker.extra if isinstance(joker.extra, int) else 0)
        return True
    elif name == "Castle" and not card.debuff and isinstance(joker.extra, dict):
        if card.suit == state.current_round.castle_card.get("suit"):
            joker.extra["chips"] = int(joker.extra.get("chips", 0) or 0) + int(joker.extra.get("chip_mod", 0) or 0)
    elif name == "Mail-In Rebate" and not card.debuff:
        if _card_id(card) == _mail_id(state):
            state.dollars += int(joker.extra if isinstance(joker.extra, int) else 0)
    elif name == "Hit the Road" and not card.debuff and card.rank == "J" and isinstance(joker.extra, (int, float)):
        joker.x_mult += float(joker.extra)
    elif name == "Ramen" and isinstance(joker.extra, (int, float)):
        if joker.x_mult - float(joker.extra) <= 1:
            from .instances import remove_joker as _remove_joker
            _remove_joker(state, joker)
        else:
            joker.x_mult -= float(joker.extra)

    if last and name == "Green Joker" and isinstance(joker.extra, dict):
        joker.mult = max(0, joker.mult - int(joker.extra.get("discard_sub", 0) or 0))
    if last and name == "Faceless Joker" and isinstance(joker.extra, dict) and face_tally >= int(joker.extra.get("faces", 0) or 0):
        state.dollars += int(joker.extra.get("dollars", 0) or 0)
    return False


def _apply_removed_card_effects(state: RunState, removed: list[PlayingCard]) -> None:
    if not removed:
        return
    removed_faces = sum(1 for card in removed if _is_face(state, card))
    shattered = sum(1 for card in removed if card.shattered)
    for joker in state.jokers:
        name = state.data.centers[joker.center_key]["name"]
        if name == "Canio" and removed_faces and isinstance(joker.extra, (int, float)):
            joker.caino_xmult += removed_faces * float(joker.extra)
        elif name == "Glass Joker" and shattered and isinstance(joker.extra, (int, float)):
            joker.x_mult += shattered * float(joker.extra)


def _destroy_playing_card(state: RunState, card: PlayingCard) -> None:
    center = state.data.centers[card.center_key]
    if center.get("name") == "Glass Card":
        card.shattered = True
    else:
        card.destroyed = True
    _remove_exact(state.deck_cards, card)
    _remove_exact(state.draw_pile, card)
    _remove_exact(state.discard_pile, card)
    _remove_exact(state.hand_cards, card)
    _remove_exact(state.play_cards, card)


def _remove_exact(cards: list[PlayingCard], card: PlayingCard) -> bool:
    for index, candidate in enumerate(cards):
        if candidate is card:
            cards.pop(index)
            return True
    return False


def _debuff_card(state: RunState, card: PlayingCard) -> None:
    """Apply blind-based debuffs to a playing card, matching Balatro's boss-blind rules."""
    blind = state.round_resets.blind or {}
    blind_name = str(blind.get("name", ""))
    debuff = blind.get("debuff") or {}

    if state.blind_disabled:
        card.debuff = False
        return

    if blind_name == "Verdant Leaf":
        card.debuff = True
        return

    if debuff and not state.blind_disabled:
        if debuff.get("suit") and _is_suit(state, card, str(debuff["suit"]), bypass_debuff=True):
            card.debuff = True
            return
        if debuff.get("is_face") == "face" and _scoring_is_face(state, card, from_boss=True):
            card.debuff = True
            return

    # The Pillar debuffs cards played this ante (independent of debuff config)
    if blind_name == "The Pillar" and card.played_this_ante:
        card.debuff = True
        return

    card.debuff = False


def _debuff_hand(state: RunState, cards: list[PlayingCard], hand_name: str, poker_hands: dict, *, check: bool = False) -> bool:
    """Check if a hand is debuffed by the current blind. Returns True if debuffed."""
    if state.blind_disabled:
        return False
    blind = state.round_resets.blind or {}
    blind_name = str(blind.get("name", ""))
    debuff = blind.get("debuff") or {}

    if debuff:
        state.blind_triggered = False
        if debuff.get("hand") and poker_hands.get(debuff["hand"]):
            if any(poker_hands[debuff["hand"]]):
                state.blind_triggered = True
                return True
        if debuff.get("h_size_ge") and len(cards) < int(debuff["h_size_ge"]):
            state.blind_triggered = True
            return True
        if debuff.get("h_size_le") and len(cards) > int(debuff["h_size_le"]):
            state.blind_triggered = True
            return True

    # The Eye/Mouth sit outside the `if debuff:` gate like The Arm/Ox: their
    # debuff table is empty upstream (Lua `{}` is truthy, Python's isn't), so
    # gating them on `debuff` silently disabled both bosses. The gate must
    # stay narrow regardless — _press_play has already set blind_triggered for
    # The Hook/Tooth this play, and Matador reads it during scoring.
    if blind_name == "The Eye":
        state.blind_triggered = False
        if state.eye_hands.get(hand_name):
            state.blind_triggered = True
            return True
        if not check:
            state.eye_hands[hand_name] = True
    if blind_name == "The Mouth":
        state.blind_triggered = False
        if state.mouth_only_hand and state.mouth_only_hand != hand_name:
            state.blind_triggered = True
            return True
        if not check:
            state.mouth_only_hand = hand_name

    if blind_name == "The Arm":
        state.blind_triggered = False
        if state.hands[hand_name]["level"] > 1:
            state.blind_triggered = True
            if not check:
                _level_up_hand(state, hand_name, -1)
    if blind_name == "The Ox":
        state.blind_triggered = False
        if hand_name == state.current_round.most_played_poker_hand:
            state.blind_triggered = True
            if not check:
                state.dollars = max(0, state.dollars - state.dollars)  # zero out

    return False


def _press_play(state: RunState, play_cards_list: list[PlayingCard]) -> None:
    """Pre-scoring side effects from boss blinds."""
    if state.blind_disabled:
        return
    blind_name = _blind_name(state)
    if blind_name == "The Hook" and state.hand_cards:
        available = list(state.hand_cards)
        discarded = []
        for _ in range(min(2, len(available))):
            if not available:
                break
            chosen, idx = state.pseudorandom.pseudorandom_element(
                available,
                state.pseudorandom.pseudoseed("hook"),
            )
            discarded.append(chosen)
            available = [c for c in available if c is not chosen]
        discard_cards(state, discarded, hook=True)
        state.blind_triggered = True
    if blind_name == "The Tooth":
        for _ in play_cards_list:
            state.dollars = max(0, state.dollars - 1)
        state.blind_triggered = True


def _blind_name(state: RunState) -> str:
    blind = state.round_resets.blind or {}
    return str(blind.get("name", ""))


def _is_face(state: RunState, card: PlayingCard) -> bool:
    return card.rank in {"J", "Q", "K"} or state.has_joker("Pareidolia")


def _card_id(card: PlayingCard) -> int:
    return RANK_TO_ID[card.rank]


def _mail_id(state: RunState) -> int:
    rank = str(state.current_round.mail_card.get("rank", "Ace"))
    rank_map = {"Ace": "A", "Jack": "J", "Queen": "Q", "King": "K", "10": "T"}
    rank_key = rank_map.get(rank, rank)
    return RANK_TO_ID[rank_key]


def _card_nominal(state: RunState, card: PlayingCard) -> float:
    base = RANK_TO_NOMINAL[card.rank]
    face_nominal = 0.1 if card.rank == "J" else 0.2 if card.rank == "Q" else 0.3 if card.rank == "K" else 0.4 if card.rank == "A" else 0
    suit_nominal = SUIT_TO_NOMINAL[card.suit]
    suit_mult = -1000 if state.data.centers[card.center_key].get("effect") == "Stone Card" else 1
    return base + suit_nominal * suit_mult + suit_nominal * 0.0001 * suit_mult + face_nominal + 0.000001 * (1 - card.reward_uid / 1_603_301)
