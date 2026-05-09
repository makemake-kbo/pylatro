"""Convert RunState + SubPhase into integer/float feature arrays for the model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cython
import numpy as np

from pylatro.models import (
    ConsumableInstance,
    JokerInstance,
    RunState,
    ShopCard,
)

from .constants import (
    BLIND_SELECT_MAX,
    BLIND_SELECT_START,
    CONSUMABLE_MAX,
    CONSUMABLE_START,
    DECK_MAX,
    DECK_START,
    HAND_LEVEL_MAX,
    HAND_LEVEL_START,
    JOKER_MAX,
    JOKER_START,
    MAX_HAND_SIZE,
    MAX_SEQ_LEN,
    META_COUNT,
    META_START,
    OBJ_START,
    POKER_HAND_NAMES,
    SHOP_MAX,
    SHOP_START,
    TOKEN_DIM,
    VOUCHER_MAX,
    VOUCHER_START,
    SubPhase,
    TokenType,
)
from .hand_candidates import HAND_NAME_TO_ID, HandCandidate
from .vocab import EDITION_TO_ID, RANK_TO_ID, SEAL_TO_ID, SUIT_TO_ID, Vocab


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
    selected_cards: np.ndarray  # (MAX_HAND_SIZE,), int8


@dataclass
class Tokenizer:
    vocab: Vocab
    _selected: set[int] = field(default_factory=set)

    @cython.locals(
        pos=cython.int,
        card_idx=cython.int,
        i=cython.int,
        ci=cython.int,
        p=cython.Py_ssize_t,
        loc=cython.int,
        tokens=cython.short[:, :],
        token_types=cython.char[:],
        attn_mask=cython.char[:],
        scalars=cython.float[:],
        sel_cards=cython.char[:],
        idx=cython.int,
    )
    def tokenize(
        self,
        state: RunState,
        sub_phase: SubPhase,
        selected_cards: set[int] | None = None,
        action_mask: np.ndarray | None = None,
        round_score: int = 0,
    ) -> RawObservation:
        from .constants import NUM_ACTIONS, SCALAR_DIM

        tokens = np.zeros((MAX_SEQ_LEN, TOKEN_DIM), dtype=np.int16)
        token_types = np.full(MAX_SEQ_LEN, TokenType.PAD, dtype=np.int8)
        attn_mask = np.zeros(MAX_SEQ_LEN, dtype=np.int8)
        scalars = np.zeros(SCALAR_DIM, dtype=np.float32)
        sel_cards = np.zeros(MAX_HAND_SIZE, dtype=np.int8)

        if selected_cards is None:
            selected_cards = set()

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
                tokens[p, 5] = loc
                tokens[p, 6] = card.debuff
                tokens[p, 7] = card.face_down
                tokens[p, 8] = min(card.perma_bonus // 5, 31)
                tokens[p, 9] = 1 if loc == 0 and ci in selected_cards else 0
                tokens[p, 10] = card.forced_selection
                tokens[p, 11] = ci if loc == 0 else 0
                # Slot 12 formerly held a "pending consumable target" flag
                # for the old CONSUMABLE_TARGET sub-phase. With atomic
                # consumable actions there is no pending state — keep the
                # slot zeroed so the token layout stays stable.
                tokens[p, 12] = 0
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

        for idx in selected_cards:
            if idx < MAX_HAND_SIZE:
                sel_cards[idx] = 1

        if action_mask is None:
            action_mask = np.ones(NUM_ACTIONS, dtype=np.int8)

        return RawObservation(
            tokens=np.asarray(tokens),
            token_types=np.asarray(token_types),
            scalars=np.asarray(scalars),
            attention_mask=np.asarray(attn_mask),
            action_mask=action_mask,
            selected_cards=np.asarray(sel_cards),
        )

    @cython.locals(blind_type_id=cython.int, boss_id=cython.int)
    def _encode_meta(self, state: RunState, sub_phase: SubPhase) -> list[int]:
        blind = state.round_resets.blind or {}
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
        return int(math.floor(base * mult))

    @cython.locals(sp_id=cython.int)
    def _sub_phase_id(self, sub_phase: SubPhase) -> int:
        return {
            SubPhase.BLIND_SELECT: 0,
            SubPhase.CHOOSE_ACTION: 1,
            SubPhase.SELECT_CARDS: 2,
            SubPhase.SHOP: 3,
            SubPhase.BOOSTER_PACK: 4,
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

    @cython.locals(
        tokens=cython.short[:, :],
        pos=cython.int,
        slot=cython.int,
        edition_id=cython.int,
    )
    def _encode_consumable(self, tokens: np.ndarray, pos: int, cons: ConsumableInstance, slot: int) -> None:
        tokens[pos, 0] = 0
        tokens[pos, 1] = self.vocab.consumable_to_id.get(cons.center_key, 0)
        edition_id = 0
        if cons.edition:
            for key, val in cons.edition.items():
                if val:
                    edition_id = EDITION_TO_ID.get(key, 0)
                    break
        tokens[pos, 2] = edition_id
        tokens[pos, 3] = slot

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
    ) -> None:
        tokens[pos, 0] = 1 if candidate.kind == "play" else 2
        tokens[pos, 1] = HAND_NAME_TO_ID.get(candidate.hand_name, 0)
        tokens[pos, 2] = len(candidate.indices)
        tokens[pos, 3] = int(min(max(sign_log(candidate.estimated_score) * 10.0, -255.0), 255.0))
        tokens[pos, 4] = int(min(max(candidate.blind_ratio * 10.0, 0.0), 255.0))
        for i in range(5):
            tokens[pos, 5 + i] = candidate.indices[i] + 1 if i < len(candidate.indices) else 0
        tokens[pos, 10] = rank_slot + 1
        tokens[pos, 11] = sum(1 for idx in candidate.indices if state.hand_cards[idx].forced_selection)
