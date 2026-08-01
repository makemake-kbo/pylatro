"""Shared three-blind played-hand history and fixed-shape encoding.

The tracker deliberately snapshots cards and joker *keys* before scoring.  A
DNA copy, destroyed card, mutable joker counter, or joker reorder performed by
the scoring engine therefore cannot rewrite the historical event.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .constants import (
    HISTORY_EVENT_DIM,
    HISTORY_FEATURE_DIM,
    HISTORY_MAX_CARDS,
    HISTORY_MAX_JOKERS,
    HISTORY_MAX_PLAYS,
    HISTORY_OMITTED_DIM,
    HISTORY_ROUNDS,
    POKER_HAND_NAMES,
    TOKEN_DIM,
)
from .vocab import EDITION_TO_ID, RANK_TO_ID, SEAL_TO_ID, SUIT_TO_ID

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable

    from pylatro.models import PlayingCard, RunState

    from .vocab import Vocab


DNA_FLAG = 1
DUSK_SETUP_FLAG = 2
DUSK_PAYOFF_FLAG = 4
BURGLAR_FLAG = 8
MECHANIC_FLAGS = (DNA_FLAG, DUSK_SETUP_FLAG, DUSK_PAYOFF_FLAG, BURGLAR_FLAG)

HAND_TYPE_TO_ID = {name: index + 1 for index, name in enumerate(POKER_HAND_NAMES)}


def _sign_log(value: float) -> float:
    return math.log1p(value) if value >= 0 else -math.log1p(-value)


@dataclass(frozen=True, slots=True)
class CardSnapshot:
    rank: str
    suit: str
    center_key: str
    edition_key: str | None
    seal: str | None
    perma_bonus: int
    debuff: bool
    face_down: bool
    front_key: str = ""
    shattered: bool = False
    destroyed: bool = False
    played_this_ante: bool = False
    discarded: bool = False
    forced_selection: bool = False
    times_played: int = 0

    @classmethod
    def from_card(cls, card: PlayingCard) -> CardSnapshot:
        return cls(
            rank=str(card.rank),
            suit=str(card.suit),
            center_key=str(card.center_key),
            edition_key=str(card.edition_key) if card.edition_key else None,
            seal=str(card.seal) if card.seal else None,
            perma_bonus=int(card.perma_bonus),
            debuff=bool(card.debuff),
            face_down=bool(card.face_down),
            front_key=str(card.front_key),
            shattered=bool(card.shattered),
            destroyed=bool(card.destroyed),
            played_this_ante=bool(card.played_this_ante),
            discarded=bool(card.discarded),
            forced_selection=bool(card.forced_selection),
            times_played=int(card.times_played),
        )


@dataclass(frozen=True, slots=True)
class PendingPlay:
    cards: tuple[CardSnapshot, ...]
    joker_keys: tuple[str, ...]
    blind_target: int
    score_before: int
    play_ordinal: int
    hands_remaining: int
    mechanic_flags: int


@dataclass(frozen=True, slots=True)
class PlayEvent:
    cards: tuple[CardSnapshot, ...]
    joker_keys: tuple[str, ...]
    hand_type: str
    score: int
    blind_target: int
    cumulative_round_score: int
    play_ordinal: int
    hands_remaining: int
    mechanic_flags: int


@dataclass(slots=True)
class OmittedSummary:
    count: int = 0
    total_score: int = 0
    max_score: int = 0
    hand_type_counts: list[int] = field(default_factory=lambda: [0] * len(POKER_HAND_NAMES))
    mechanic_counts: list[int] = field(default_factory=lambda: [0] * len(MECHANIC_FLAGS))

    def add(self, event: PlayEvent) -> None:
        self.count += 1
        self.total_score += max(0, int(event.score))
        self.max_score = max(self.max_score, int(event.score))
        hand_id = HAND_TYPE_TO_ID.get(event.hand_type, 0)
        if hand_id:
            self.hand_type_counts[hand_id - 1] += 1
        for index, flag in enumerate(MECHANIC_FLAGS):
            if event.mechanic_flags & flag:
                self.mechanic_counts[index] += 1


@dataclass(slots=True)
class RoundHistory:
    key: Hashable | None = None
    events: list[PlayEvent] = field(default_factory=list)
    omitted: OmittedSummary = field(default_factory=OmittedSummary)
    total_plays: int = 0
    peak_score: int = 0
    final_score: int = 0

    @property
    def has_content(self) -> bool:
        return bool(self.events or self.omitted.count)

    def add(self, event: PlayEvent) -> None:
        self.total_plays += 1
        self.peak_score = max(self.peak_score, event.score)
        self.final_score = max(self.final_score, event.cumulative_round_score)
        candidates = [*self.events, event]
        if len(candidates) <= HISTORY_MAX_PLAYS:
            self.events = candidates
            return

        # Mechanic-bearing setup/payoff hands and the latest play are protected.
        # Remaining capacity goes to the strongest hands.  The final sort makes
        # the emitted history chronological and independent of score ties.
        latest_ordinal = max(item.play_ordinal for item in candidates)
        protected = [
            item
            for item in candidates
            if item.mechanic_flags or item.play_ordinal == latest_ordinal
        ]
        if len(protected) > HISTORY_MAX_PLAYS:
            protected = sorted(
                protected,
                key=lambda item: (item.play_ordinal == latest_ordinal, item.play_ordinal),
                reverse=True,
            )[:HISTORY_MAX_PLAYS]
        protected_ids = {id(item) for item in protected}
        remaining = [item for item in candidates if id(item) not in protected_ids]
        remaining.sort(key=lambda item: (item.score, item.play_ordinal), reverse=True)
        kept = protected + remaining[: HISTORY_MAX_PLAYS - len(protected)]
        kept_ids = {id(item) for item in kept}
        for dropped in candidates:
            if id(dropped) not in kept_ids:
                self.omitted.add(dropped)
        self.events = sorted(kept, key=lambda item: item.play_ordinal)


@dataclass(frozen=True, slots=True)
class HistoryArrays:
    events: np.ndarray
    event_features: np.ndarray
    cards: np.ndarray
    card_mask: np.ndarray
    jokers: np.ndarray
    joker_mask: np.ndarray
    event_mask: np.ndarray
    round_mask: np.ndarray
    omitted: np.ndarray

    @classmethod
    def empty(cls) -> HistoryArrays:
        return cls(
            events=np.zeros((HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_EVENT_DIM), dtype=np.int16),
            event_features=np.zeros(
                (HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_FEATURE_DIM), dtype=np.float32
            ),
            cards=np.zeros(
                (HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_CARDS, TOKEN_DIM), dtype=np.int16
            ),
            card_mask=np.zeros(
                (HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_CARDS), dtype=np.int8
            ),
            jokers=np.zeros(
                (HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_JOKERS), dtype=np.int16
            ),
            joker_mask=np.zeros(
                (HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_JOKERS), dtype=np.int8
            ),
            event_mask=np.zeros((HISTORY_ROUNDS, HISTORY_MAX_PLAYS), dtype=np.int8),
            round_mask=np.zeros(HISTORY_ROUNDS, dtype=np.int8),
            omitted=np.zeros((HISTORY_ROUNDS, HISTORY_OMITTED_DIM), dtype=np.float32),
        )

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "history_events": self.events,
            "history_event_features": self.event_features,
            "history_cards": self.cards,
            "history_card_mask": self.card_mask,
            "history_jokers": self.jokers,
            "history_joker_mask": self.joker_mask,
            "history_event_mask": self.event_mask,
            "history_round_mask": self.round_mask,
            "history_omitted": self.omitted,
        }


class PlayHistoryTracker:
    """Tracks retained play events for the current and two preceding blinds."""

    def __init__(self) -> None:
        self._rounds: list[RoundHistory] = [RoundHistory() for _ in range(HISTORY_ROUNDS)]
        self._current_key: Hashable | None = None

    @property
    def rounds(self) -> tuple[RoundHistory, ...]:
        return tuple(self._rounds)

    def reset(self) -> None:
        self._rounds = [RoundHistory() for _ in range(HISTORY_ROUNDS)]
        self._current_key = None

    def start_round(self, key: Hashable) -> None:
        if self._current_key == key:
            return
        if self._current_key is None:
            self._rounds[-1] = RoundHistory(key=key)
        else:
            self._rounds = [self._rounds[1], self._rounds[2], RoundHistory(key=key)]
        self._current_key = key

    def capture(
        self,
        state: RunState,
        cards: Iterable[PlayingCard],
        *,
        blind_target: int,
        round_score: int,
    ) -> PendingPlay:
        selected = tuple(CardSnapshot.from_card(card) for card in cards)
        joker_keys = tuple(str(joker.center_key) for joker in state.jokers)
        active_joker_keys = {
            joker.center_key
            for joker in state.jokers
            if not joker.debuff and not joker.getting_sliced
        }
        first_single = state.current_round.hands_played == 0 and len(selected) == 1
        flags = 0
        if "j_dna" in active_joker_keys and first_single:
            flags |= DNA_FLAG
        if "j_dusk" in active_joker_keys:
            flags |= DUSK_PAYOFF_FLAG if state.current_round.hands_left <= 1 else DUSK_SETUP_FLAG
        if "j_burglar" in active_joker_keys and state.current_round.discards_left <= 0:
            flags |= BURGLAR_FLAG
        current = self._rounds[-1]
        ordinal = max(current.total_plays + 1, int(state.current_round.hands_played) + 1)
        return PendingPlay(
            cards=selected,
            joker_keys=joker_keys,
            blind_target=max(0, int(blind_target)),
            score_before=max(0, int(round_score)),
            play_ordinal=ordinal,
            hands_remaining=max(0, int(state.current_round.hands_left) - 1),
            mechanic_flags=flags,
        )

    def finalize(self, pending: PendingPlay, *, hand_type: str, score: int) -> PlayEvent:
        if self._current_key is None:
            # Useful for live reconnects and direct unit use: establish an
            # anonymous current round without fabricating any prior events.
            self.start_round(("anonymous", 0))
        event = PlayEvent(
            cards=pending.cards,
            joker_keys=pending.joker_keys,
            hand_type=str(hand_type),
            score=max(0, int(score)),
            blind_target=pending.blind_target,
            cumulative_round_score=pending.score_before + max(0, int(score)),
            play_ordinal=pending.play_ordinal,
            hands_remaining=pending.hands_remaining,
            mechanic_flags=pending.mechanic_flags,
        )
        self._rounds[-1].add(event)
        return event

    def encode(self, vocab: Vocab) -> HistoryArrays:
        arrays = HistoryArrays.empty()
        for round_index, round_history in enumerate(self._rounds):
            if not round_history.has_content:
                continue
            arrays.round_mask[round_index] = 1
            peak = max(1, round_history.peak_score)
            final = max(1, round_history.final_score)
            for event_index, event in enumerate(round_history.events):
                arrays.event_mask[round_index, event_index] = 1
                arrays.events[round_index, event_index] = (
                    HAND_TYPE_TO_ID.get(event.hand_type, 0),
                    min(event.play_ordinal, 32767),
                    min(event.hands_remaining, 32767),
                    event.mechanic_flags,
                    min(len(event.cards), HISTORY_MAX_CARDS),
                    min(len(event.joker_keys), HISTORY_MAX_JOKERS),
                )
                score = float(event.score)
                target = float(max(event.blind_target, 1))
                arrays.event_features[round_index, event_index] = (
                    _sign_log(score),
                    _sign_log(float(event.blind_target)),
                    _sign_log(float(event.cumulative_round_score)),
                    min(score / target, 100.0),
                    min(score / peak, 1.0),
                    min(score / final, 1.0),
                )
                for card_index, card in enumerate(event.cards[:HISTORY_MAX_CARDS]):
                    token = arrays.cards[round_index, event_index, card_index]
                    token[0] = 0 if card.face_down else RANK_TO_ID.get(card.rank, 0)
                    token[1] = 0 if card.face_down else SUIT_TO_ID.get(card.suit, 0)
                    token[2] = vocab.enhancement_to_id.get(card.center_key, 0)
                    token[3] = EDITION_TO_ID.get(card.edition_key or "", 0)
                    token[4] = SEAL_TO_ID.get(card.seal or "", 0)
                    token[5] = 3  # distinct history-card location
                    token[6] = int(card.debuff)
                    token[7] = int(card.face_down)
                    token[8] = min(max(card.perma_bonus, 0) // 5, 31)
                    token[9] = min(max(card.times_played, 0), 32767)
                    token[10] = int(card.forced_selection)
                    token[11] = card_index
                    arrays.card_mask[round_index, event_index, card_index] = 1
                for joker_index, key in enumerate(event.joker_keys[:HISTORY_MAX_JOKERS]):
                    arrays.jokers[round_index, event_index, joker_index] = vocab.joker_to_id.get(key, 0)
                    arrays.joker_mask[round_index, event_index, joker_index] = 1

            summary = round_history.omitted
            arrays.omitted[round_index, 0] = float(summary.count)
            arrays.omitted[round_index, 1] = _sign_log(float(summary.total_score))
            arrays.omitted[round_index, 2] = _sign_log(float(summary.max_score))
            arrays.omitted[round_index, 3:15] = summary.hand_type_counts
            arrays.omitted[round_index, 15:19] = summary.mechanic_counts
        return arrays


def blind_history_key(state: RunState) -> tuple[int, str, str]:
    """Stable identity for an engine/live blind without touching wire format."""
    blind = state.round_resets.blind or {}
    return (
        int(state.round_resets.ante),
        str(state.blind_on_deck or ""),
        str(blind.get("key") or blind.get("name") or ""),
    )
