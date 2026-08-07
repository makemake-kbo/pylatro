"""Convert RunState + SubPhase into integer/float feature arrays for the model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cython
import numpy as np

if TYPE_CHECKING:
    from pylatro.models import ConsumableInstance, JokerInstance, RunState, ShopCard

    from .history import PlayHistoryTracker

from .constants import (
    BLIND_SELECT_MAX,
    BLIND_SELECT_START,
    CONSUMABLE_MAX,
    CONSUMABLE_START,
    DECK_MAX,
    DECK_START,
    HAND_CANDIDATE_MAX,
    HAND_CANDIDATE_START,
    HAND_LEVEL_MAX,
    HAND_LEVEL_START,
    HISTORY_ROUNDS,
    HISTORY_START,
    JOKER_MAX,
    JOKER_START,
    MAX_DISCARD_CANDIDATES,
    MAX_PACK_CARDS,
    MAX_PLAY_CANDIDATES,
    MAX_SEQ_LEN,
    META_COUNT,
    META_START,
    POKER_HAND_NAMES,
    SHOP_MAX,
    SHOP_START,
    TOKEN_DIM,
    VOUCHER_MAX,
    VOUCHER_START,
    SubPhase,
    TokenType,
)
from .hand_candidates import HAND_NAME_TO_ID, HandCandidate, generate_hand_candidates
from .strategy_context import compute_suit_target_utilities
from .vocab import EDITION_TO_ID, RANK_TO_ID, SEAL_TO_ID, SUIT_TO_ID, Vocab


def _hypergeom_at_least_one(population: int, successes: int, draws: int) -> float:
    population = max(int(population), 0)
    successes = max(0, min(int(successes), population))
    draws = max(0, min(int(draws), population))
    if population <= 0 or successes <= 0 or draws <= 0:
        return 0.0
    if draws >= population:
        return 1.0
    return 1.0 - math.comb(population - successes, draws) / math.comb(population, draws)


def strategy_probability_features(
    state: RunState,
    sub_phase: SubPhase,
    *,
    clear_probability: float = 0.0,
) -> tuple[float, ...]:
    """Return five opportunity features plus four suit-target utilities."""
    from pylatro.runtime import consumable_limit

    active = sub_phase == SubPhase.CHOOSE_ACTION
    active_debuffs = active and not state.blind_disabled
    pool = list(state.draw_pile if active else state.deck_cards)
    boss_debuff_suit = ""
    if not state.blind_disabled and not active and (state.blind_on_deck or "") == "Boss":
        boss_key = state.round_resets.blind_choices.get("Boss", "")
        boss = state.data.blinds.get(boss_key, {})
        debuff = boss.get("debuff", {}) or {}
        boss_debuff_suit = str(debuff.get("suit", "") or "")

    def live(card) -> bool:
        # Between blinds, ``card.debuff`` can still describe the blind that
        # just ended. Only active-blind debuffs are authoritative; pre-blind
        # suit exclusions come from the known upcoming boss instead.
        return (not active_debuffs or not card.debuff) and (
            not boss_debuff_suit or card.suit != boss_debuff_suit
        )

    # Ineligible cards remain physical failure draws. Removing them from the
    # population would condition on never drawing a debuffed/boss-suit card
    # and systematically overstate every opportunity probability.
    live_pool = [card for card in pool if live(card)]
    population = len(pool)
    hand_size = max(int(state.current_round.hand_size if active else state.starting_params.hand_size), 1)
    hands = max(int(state.current_round.hands_left if active else state.round_resets.hands), 1)
    discards = max(int(state.current_round.discards_left if active else state.round_resets.discards), 0)
    search_draws = min(
        population,
        (0 if active else hand_size) + 5 * discards + 5 * max(hands - 1, 0),
    )

    live_hand = [card for card in state.hand_cards if live(card)] if active else []
    safe_blue_held = (
        active
        and clear_probability >= 0.65
        and len(state.hand_cards) > 1
        and any(card.seal == "Blue" for card in live_hand)
    )
    blue_successes = sum(card.seal == "Blue" for card in live_pool)
    p_blue = 1.0 if safe_blue_held else _hypergeom_at_least_one(population, blue_successes, search_draws)

    live_purple_now = active and discards > 0 and any(card.seal == "Purple" for card in live_hand)
    # If Purple is not already in hand, one discard must remain after finding
    # it so the seal itself can be discarded. With one discard left there is
    # no future search-and-trigger opportunity.
    purple_search_draws = min(
        population,
        (0 if active else hand_size) + 5 * max(discards - 1, 0),
    )
    purple_successes = sum(card.seal == "Purple" for card in live_pool)
    p_purple = (
        1.0
        if live_purple_now
        else _hypergeom_at_least_one(population, purple_successes, purple_search_draws)
    )

    def enhancement_probability(center_key: str) -> float:
        if active and any(card.center_key == center_key for card in live_hand):
            return 1.0
        successes = sum(card.center_key == center_key for card in live_pool)
        return _hypergeom_at_least_one(population, successes, search_draws)

    capacity = max(consumable_limit(state), 0)
    room_fraction = max(capacity - len(state.consumables), 0) / max(capacity, 1)
    boss_key = state.round_resets.blind_choices.get("Boss", "")
    boss = state.data.blinds.get(boss_key, {}) if boss_key else {}
    known_boss_suit = (
        ""
        if state.blind_disabled
        else str((boss.get("debuff", {}) or {}).get("suit", "") or "")
    )
    suit_utilities = compute_suit_target_utilities(
        state.deck_cards,
        state.jokers,
        hand_play_counts={
            name: int(hand.get("played", 0) or 0)
            for name, hand in state.hands.items()
        },
        boss_debuff_suit=known_boss_suit,
        # A completed blind can leave stale debuff flags on deck cards. The
        # known upcoming boss is the only authoritative pre-blind exclusion.
        respect_card_debuff=active_debuffs,
    )
    return (
        min(max(p_blue, 0.0), 1.0),
        min(max(p_purple, 0.0), 1.0),
        enhancement_probability("m_gold"),
        enhancement_probability("m_steel"),
        min(max(room_fraction, 0.0), 1.0),
        *suit_utilities,
    )


@cython.ccall
@cython.exceptval(check=False)
@cython.locals(x=cython.double)
def sign_log(x: float) -> float:
    if x >= 0:
        return math.log1p(x)
    return -math.log1p(-x)


@dataclass(slots=True)
class RawObservation:
    tokens: np.ndarray  # (MAX_SEQ_LEN, TOKEN_DIM), int16
    token_types: np.ndarray  # (MAX_SEQ_LEN,), int8
    scalars: np.ndarray  # (SCALAR_DIM,), float32
    attention_mask: np.ndarray  # (MAX_SEQ_LEN,), int8
    action_mask: np.ndarray  # (NUM_ACTIONS,), int8
    history_events: np.ndarray
    history_event_features: np.ndarray
    history_cards: np.ndarray
    history_card_mask: np.ndarray
    history_jokers: np.ndarray
    history_joker_mask: np.ndarray
    history_event_mask: np.ndarray
    history_round_mask: np.ndarray
    history_omitted: np.ndarray


@dataclass
class Tokenizer:
    vocab: Vocab

    @cython.locals(
        pos=cython.int,
        card_idx=cython.int,
        i=cython.int,
        ci=cython.int,
        strategy_index=cython.int,
        p=cython.Py_ssize_t,
        loc=cython.int,
        tokens=cython.short[:, :],
        token_types=cython.char[:],
        attn_mask=cython.char[:],
        scalars=cython.float[:],
    )
    def tokenize(
        self,
        state: RunState,
        sub_phase: SubPhase,
        action_mask: np.ndarray | None = None,
        round_score: int = 0,
        history: PlayHistoryTracker | None = None,
        clear_probability: float | None = None,
        immediate_death_probability: float | None = None,
    ) -> RawObservation:
        from .constants import NUM_ACTIONS, SCALAR_DIM

        tokens = np.zeros((MAX_SEQ_LEN, TOKEN_DIM), dtype=np.int16)
        token_types = np.full(MAX_SEQ_LEN, TokenType.PAD, dtype=np.int8)
        attn_mask = np.zeros(MAX_SEQ_LEN, dtype=np.int8)
        scalars = np.zeros(SCALAR_DIM, dtype=np.float32)
        _rank_to_id = RANK_TO_ID
        _suit_to_id = SUIT_TO_ID
        _enh_to_id = self.vocab.enhancement_to_id
        _edition_to_id = EDITION_TO_ID
        _seal_to_id = SEAL_TO_ID

        pos = 0

        tokens[pos, 0] = 0
        token_types[pos] = TokenType.OBJ
        attn_mask[pos] = 1
        pos = 1

        pos = META_START
        meta_values = self._encode_meta(state, sub_phase)
        for i, val in enumerate(meta_values):
            if pos + i >= DECK_START:
                break
            tokens[pos + i, 0] = val
            token_types[pos + i] = TokenType.META
            attn_mask[pos + i] = 1
        pos = META_START + META_COUNT

        scalars[0] = sign_log(float(state.dollars))
        scalars[1] = float(state.interest_cap)
        scalars[2] = float(state.round_resets.ante)
        scalars[3] = sign_log(float(self._blind_target(state)))
        scalars[4] = float(state.current_round.hands_left)
        scalars[5] = float(state.current_round.discards_left)
        scalars[6] = float(state.current_round.hand_size)
        scalars[7] = float(self._sub_phase_id(sub_phase))
        blind_target = float(self._blind_target(state))
        round_score_f = max(float(round_score), 0.0)
        score_remaining = max(blind_target - round_score_f, 0.0)
        scalars[8] = sign_log(round_score_f)
        scalars[9] = sign_log(score_remaining)
        scalars[10] = min(round_score_f / max(blind_target, 1.0), 1.0)
        if clear_probability is None or immediate_death_probability is None:
            from .risk import capture_state_risk

            risk = capture_state_risk(state, round_score)
            if clear_probability is None:
                clear_probability = risk.clear_probability
            if immediate_death_probability is None:
                immediate_death_probability = risk.immediate_death_probability
        scalars[11] = min(max(float(clear_probability), 0.0), 1.0)
        scalars[12] = min(max(float(immediate_death_probability), 0.0), 1.0)
        strategy_probabilities = strategy_probability_features(
            state,
            sub_phase,
            clear_probability=float(clear_probability),
        )
        for strategy_index in range(9):
            scalars[13 + strategy_index] = strategy_probabilities[strategy_index]

        pos = DECK_START
        hand_cards = state.hand_cards
        draw_pile = state.draw_pile
        discard_pile = state.discard_pile
        card_idx = 0
        for loc, pile in ((0, hand_cards), (1, draw_pile), (2, discard_pile)):
            for ci, card in enumerate(pile):
                if card_idx >= DECK_MAX:
                    break
                p = pos + card_idx
                if card.face_down:
                    tokens[p, 0] = 0
                    tokens[p, 1] = 0
                else:
                    tokens[p, 0] = _rank_to_id.get(card.rank, 0)
                    tokens[p, 1] = _suit_to_id.get(card.suit, 0)
                tokens[p, 2] = _enh_to_id.get(card.center_key, 0)
                tokens[p, 3] = _edition_to_id.get(card.edition_key or "", 0)
                tokens[p, 4] = _seal_to_id.get(card.seal or "", 0)
                # deck token cols: 5=location, 11=hand_slot
                tokens[p, 5] = loc
                tokens[p, 6] = card.debuff
                tokens[p, 7] = card.face_down
                tokens[p, 8] = min(card.perma_bonus // 5, 31)
                tokens[p, 10] = card.forced_selection
                tokens[p, 11] = ci if loc == 0 else 0
                token_types[p] = TokenType.DECK
                attn_mask[p] = 1
                card_idx += 1

        pos = JOKER_START
        for i, joker in enumerate(state.jokers[:JOKER_MAX]):
            self._encode_joker(tokens, pos + i, joker, i)
            token_types[pos + i] = TokenType.JOKER
            attn_mask[pos + i] = 1

        pos = VOUCHER_START
        voucher_keys = [k for k in state.used_vouchers if state.used_vouchers[k]]
        for i, vkey in enumerate(voucher_keys[:VOUCHER_MAX]):
            tokens[pos + i, 0] = self.vocab.voucher_to_id.get(vkey, 0)
            token_types[pos + i] = TokenType.VOUCHER
            attn_mask[pos + i] = 1

        pos = CONSUMABLE_START
        for i, cons in enumerate(state.consumables[:CONSUMABLE_MAX]):
            self._encode_consumable(tokens, pos + i, cons, i)
            token_types[pos + i] = TokenType.CONSUMABLE
            attn_mask[pos + i] = 1

        if sub_phase == SubPhase.SHOP:
            pos = SHOP_START
            shop_items = self._gather_shop_items(state)
            for i, item in enumerate(shop_items[:SHOP_MAX]):
                self._encode_shop_item(tokens, pos + i, item, i)
                token_types[pos + i] = TokenType.SHOP
                attn_mask[pos + i] = 1

        if sub_phase == SubPhase.BOOSTER_PACK and state.pack is not None:
            pos = SHOP_START
            for i, card in enumerate(state.pack.cards[:MAX_PACK_CARDS]):
                self._encode_pack_card(tokens, pos + i, card, i, state)
                token_types[pos + i] = TokenType.SHOP
                attn_mask[pos + i] = 1

        if sub_phase == SubPhase.BLIND_SELECT:
            pos = BLIND_SELECT_START
            blind_infos = self._gather_blind_choices(state)
            for i, binfo in enumerate(blind_infos[:BLIND_SELECT_MAX]):
                self._encode_blind_choice(tokens, pos + i, binfo)
                token_types[pos + i] = TokenType.BLIND_SELECT
                attn_mask[pos + i] = 1

        pos = HAND_LEVEL_START
        for i, hand_name in enumerate(POKER_HAND_NAMES):
            if i >= HAND_LEVEL_MAX:
                break
            self._encode_hand_level(tokens, pos + i, hand_name, state.hands[hand_name])
            token_types[pos + i] = TokenType.HAND_LEVEL
            attn_mask[pos + i] = 1

        if sub_phase == SubPhase.CHOOSE_ACTION:
            play_candidates, discard_candidates = generate_hand_candidates(state)
            all_candidates = list(play_candidates[:MAX_PLAY_CANDIDATES]) + list(
                discard_candidates[:MAX_DISCARD_CANDIDATES]
            )
            for rank_slot, candidate in enumerate(all_candidates[:HAND_CANDIDATE_MAX]):
                pos = HAND_CANDIDATE_START + rank_slot
                self._encode_hand_candidate(
                    tokens,
                    pos,
                    candidate,
                    rank_slot,
                    state,
                    round_score,
                )
                token_types[pos] = TokenType.HAND_CANDIDATE
                attn_mask[pos] = 1

        if action_mask is None:
            action_mask = np.ones(NUM_ACTIONS, dtype=np.int8)

        from .history import HistoryArrays

        history_arrays = history.encode(self.vocab) if history is not None else HistoryArrays.empty()
        for i in range(HISTORY_ROUNDS):
            if history_arrays.round_mask[i]:
                token_types[HISTORY_START + i] = TokenType.HISTORY
                attn_mask[HISTORY_START + i] = 1

        return RawObservation(
            tokens=np.asarray(tokens),
            token_types=np.asarray(token_types),
            scalars=np.asarray(scalars),
            attention_mask=np.asarray(attn_mask),
            action_mask=action_mask,
            **history_arrays.as_dict(),
        )

    @cython.locals(blind_type_id=cython.int, boss_id=cython.int)
    def _encode_meta(self, state: RunState, sub_phase: SubPhase) -> list[int]:
        blind_type_id = {"Small": 0, "Big": 1}.get(state.blind_on_deck or "Small", 2)
        boss_key = state.round_resets.blind_choices.get("Boss", "")
        boss_id = self.vocab.boss_to_id.get(boss_key, 0)

        return [
            int(sign_log(float(state.dollars)) * 10),
            int(state.interest_cap),
            int(state.round_resets.ante),
            blind_type_id * 100 + boss_id,
            int(sign_log(float(self._blind_target(state))) * 10),
            int(state.current_round.hands_left),
            int(state.current_round.discards_left),
            int(state.current_round.hand_size),
            self._sub_phase_id(sub_phase),
        ]

    @cython.locals(ante=cython.int, scaling=cython.int, base=cython.int)
    def _blind_target(self, state: RunState) -> int:
        blind = state.round_resets.blind
        if blind is None:
            return 0
        from pylatro import get_blind_amount

        ante = state.round_resets.ante
        scaling = min(state.stake, 3)
        base = get_blind_amount(ante, scaling)
        mult = blind.get("mult", 1)
        return math.floor(base * mult)

    @cython.locals(sp_id=cython.int)
    def _sub_phase_id(self, sub_phase: SubPhase) -> int:
        return {
            SubPhase.BLIND_SELECT: 0,
            SubPhase.CHOOSE_ACTION: 1,
            SubPhase.SHOP: 2,
            SubPhase.BOOSTER_PACK: 3,
        }[sub_phase]

    @cython.locals(
        tokens=cython.short[:, :],
        pos=cython.int,
        slot=cython.int,
        edition_id=cython.int,
        counter=cython.int,
    )
    def _encode_joker(self, tokens: np.ndarray, pos: int, joker: JokerInstance, slot: int) -> None:
        tokens[pos, 0] = self.vocab.joker_to_id.get(joker.center_key, 0)
        tokens[pos, 1] = 0
        edition_id = 0
        if joker.edition:
            for key, val in joker.edition.items():
                if val:
                    edition_id = EDITION_TO_ID.get(key, 0)
                    break
        tokens[pos, 2] = edition_id
        tokens[pos, 3] = min(joker.sell_cost, 31)
        counter = max(joker.mult, joker.t_chips, joker.t_mult, int(joker.x_mult * 10))
        tokens[pos, 4] = min(counter, 255)
        tokens[pos, 5] = slot
        tokens[pos, 6] = int(joker.eternal)
        tokens[pos, 7] = int(joker.perishable)
        tokens[pos, 8] = int(joker.rental)
        tokens[pos, 9] = int(joker.debuff)
        tokens[pos, 10] = min(joker.perish_tally or 0, 31)

    @cython.locals(
        tokens=cython.short[:, :],
        pos=cython.int,
        slot=cython.int,
        edition_id=cython.int,
    )
    def _encode_consumable(self, tokens: np.ndarray, pos: int, cons: ConsumableInstance, slot: int) -> None:
        tokens[pos, 0] = self._consumable_set_id(cons.center_key)
        tokens[pos, 1] = self.vocab.consumable_to_id.get(cons.center_key, 0)
        edition_id = 0
        if cons.edition:
            for key, val in cons.edition.items():
                if val:
                    edition_id = EDITION_TO_ID.get(key, 0)
                    break
        tokens[pos, 2] = edition_id
        tokens[pos, 3] = slot

    def _consumable_set_id(self, center_key: str) -> int:
        return self.vocab.consumable_set_to_id.get(center_key, 0)

    def _gather_shop_items(self, state: RunState) -> list[ShopCard]:
        items: list[ShopCard] = []
        items.extend(state.shop.cards)
        items.extend(state.shop.vouchers)
        items.extend(state.shop.boosters)
        return items

    @cython.locals(
        tokens=cython.short[:, :],
        pos=cython.int,
        slot=cython.int,
        item_id=cython.int,
        item_type=cython.int,
        edition_id=cython.int,
        seal_id=cython.int,
    )
    def _encode_shop_item(self, tokens: np.ndarray, pos: int, item: ShopCard, slot: int) -> None:
        item_id = 0
        item_type = 0
        if item.center_key in self.vocab.joker_to_id:
            item_id = self.vocab.joker_to_id[item.center_key]
            item_type = 1
        elif item.center_key in self.vocab.consumable_to_id:
            item_id = self.vocab.consumable_to_id[item.center_key]
            item_type = 2
        elif item.center_key in self.vocab.voucher_to_id:
            item_id = self.vocab.voucher_to_id[item.center_key]
            item_type = 3
        elif item.center_key in self.vocab.booster_to_id:
            item_id = self.vocab.booster_to_id[item.center_key]
            item_type = 4
        tokens[pos, 0] = item_id
        tokens[pos, 1] = item_type
        tokens[pos, 2] = min(item.cost, 255)
        edition_id = 0
        if item.edition:
            for key, val in item.edition.items():
                if val:
                    edition_id = EDITION_TO_ID.get(key, 0)
                    break
        tokens[pos, 3] = edition_id
        tokens[pos, 4] = slot
        seal_id = SEAL_TO_ID.get(item.seal or "", 0)
        tokens[pos, 5] = seal_id
        tokens[pos, 6] = int(item.eternal)
        tokens[pos, 7] = int(item.perishable)
        tokens[pos, 8] = int(item.rental)

    @cython.locals(
        tokens=cython.short[:, :],
        pos=cython.int,
        slot=cython.int,
        item_id=cython.int,
        item_type=cython.int,
        edition_id=cython.int,
        seal_id=cython.int,
    )
    def _encode_pack_card(
        self,
        tokens: np.ndarray,
        pos: int,
        card: ShopCard,
        slot: int,
        state: RunState,
    ) -> None:
        item_id = 0
        item_type = 0
        if card.center_key in self.vocab.joker_to_id:
            item_id = self.vocab.joker_to_id[card.center_key]
            item_type = 1
        elif card.center_key in self.vocab.consumable_to_id:
            item_id = self.vocab.consumable_to_id[card.center_key]
            item_type = 2
        elif card.center_key in self.vocab.voucher_to_id:
            item_id = self.vocab.voucher_to_id[card.center_key]
            item_type = 3
        elif card.center_key in self.vocab.booster_to_id:
            item_id = self.vocab.booster_to_id[card.center_key]
            item_type = 4
        else:
            center = state.data.centers.get(card.center_key, {})
            if center.get("set") in ("Default", "Enhanced"):
                item_type = 5
        tokens[pos, 0] = item_id
        tokens[pos, 1] = item_type
        tokens[pos, 2] = min(card.cost, 255)
        edition_id = 0
        if card.edition:
            for key, val in card.edition.items():
                if val:
                    edition_id = EDITION_TO_ID.get(key, 0)
                    break
        tokens[pos, 3] = edition_id
        tokens[pos, 4] = slot
        seal_id = SEAL_TO_ID.get(card.seal or "", 0)
        tokens[pos, 5] = seal_id
        tokens[pos, 6] = int(card.eternal)
        tokens[pos, 7] = int(card.perishable)
        tokens[pos, 8] = int(card.rental)
        if card.front_key and len(card.front_key) >= 3:
            front = state.data.cards.get(card.front_key, {})
            tokens[pos, 9] = RANK_TO_ID.get(card.front_key[2], 0)
            tokens[pos, 10] = SUIT_TO_ID.get(str(front.get("suit", "")), 0)
        else:
            tokens[pos, 9] = 0
            tokens[pos, 10] = 0

    def _gather_blind_choices(self, state: RunState) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for bt in ("Small", "Big", "Boss"):
            blind_state = state.round_resets.blind_states.get(bt, "Upcoming")
            blind_key = state.round_resets.blind_choices.get(bt, "")
            tag_key = state.round_resets.blind_tags.get(bt, "")
            blind_data = state.data.blinds.get(blind_key, {})
            result.append(
                {
                    "blind_type": bt,
                    "blind_key": blind_key,
                    "state": blind_state,
                    "mult": blind_data.get("mult", 1),
                    "boss": blind_data.get("boss", False),
                    "tag_key": tag_key,
                }
            )
        return result

    @cython.locals(
        tokens=cython.short[:, :],
        pos=cython.int,
        blind_type_id=cython.int,
        state_id=cython.int,
        boss_id=cython.int,
        tag_id=cython.int,
    )
    def _encode_blind_choice(self, tokens: np.ndarray, pos: int, info: dict[str, Any]) -> None:
        blind_type_id = {"Small": 0, "Big": 1, "Boss": 2}.get(info["blind_type"], 0)
        state_id = {"Select": 0, "Upcoming": 1, "Current": 2, "Skipped": 3, "Defeated": 4}.get(info["state"], 0)
        boss_id = self.vocab.boss_to_id.get(info["blind_key"], 0) if info["boss"] else 0
        tag_id = self.vocab.tag_to_id.get(info["tag_key"], 0)
        tokens[pos, 0] = blind_type_id
        tokens[pos, 1] = state_id
        tokens[pos, 2] = int(info["mult"] * 10)
        tokens[pos, 3] = boss_id
        tokens[pos, 4] = tag_id

    @cython.locals(tokens=cython.short[:, :], pos=cython.int)
    def _encode_hand_level(self, tokens: np.ndarray, pos: int, hand_name: str, hand_info: dict[str, Any]) -> None:
        tokens[pos, 0] = HAND_NAME_TO_ID.get(hand_name, 0)
        tokens[pos, 1] = int(hand_info.get("level", 1) or 1)
        tokens[pos, 2] = int(hand_info.get("chips", 0) or 0)
        tokens[pos, 3] = int(hand_info.get("mult", 0) or 0)
        tokens[pos, 4] = int(hand_info.get("played", 0) or 0)
        tokens[pos, 5] = int(hand_info.get("visible", False))

    @cython.locals(tokens=cython.short[:, :], pos=cython.int, i=cython.int)
    def _encode_hand_candidate(
        self,
        tokens: np.ndarray,
        pos: int,
        candidate: HandCandidate,
        rank_slot: int,
        state: RunState,
        round_score: int,
    ) -> None:
        tokens[pos, 0] = 1 if candidate.kind == "play" else 2
        tokens[pos, 1] = HAND_NAME_TO_ID.get(candidate.hand_name, 0)
        tokens[pos, 2] = len(candidate.indices)
        # Columns 3/4 are part of the tokenizer-v6 checkpoint contract.  Keep
        # their historical projected-score / blind-coverage meanings in every
        # ante; changing a field's semantics without changing its shape still
        # sends pretrained embedding projections out of distribution.
        tokens[pos, 3] = int(min(max(sign_log(candidate.estimated_score) * 10.0, -255.0), 255.0))
        tokens[pos, 4] = int(min(max(candidate.blind_ratio * 10.0, 0.0), 255.0))
        for i in range(5):
            tokens[pos, 5 + i] = candidate.indices[i] + 1 if i < len(candidate.indices) else 0
        tokens[pos, 10] = rank_slot + 1
        tokens[pos, 11] = sum(1 for idx in candidate.indices if state.hand_cards[idx].forced_selection)
