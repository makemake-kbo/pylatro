"""Convert RunState + SubPhase into integer/float feature arrays for the model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pylatro.models import (
    ConsumableInstance,
    JokerInstance,
    PlayingCard,
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
    JOKER_MAX,
    JOKER_START,
    MAX_SEQ_LEN,
    META_COUNT,
    META_START,
    OBJ_START,
    PAD_START,
    SHOP_MAX,
    SHOP_START,
    TOKEN_DIM,
    VOUCHER_MAX,
    VOUCHER_START,
    SubPhase,
    TokenType,
)
from .vocab import EDITION_TO_ID, RANK_TO_ID, SEAL_TO_ID, SUIT_TO_ID, Vocab


def sign_log(x: float) -> float:
    """Sign-preserving log: sign(x) * log(1 + |x|)."""
    if x >= 0:
        return math.log1p(x)
    return -math.log1p(-x)


@dataclass(slots=True)
class RawObservation:
    tokens: np.ndarray       # (MAX_SEQ_LEN, TOKEN_DIM), int16
    token_types: np.ndarray  # (MAX_SEQ_LEN,), int8
    scalars: np.ndarray      # (SCALAR_DIM,), float32
    attention_mask: np.ndarray  # (MAX_SEQ_LEN,), int8
    action_mask: np.ndarray  # (NUM_ACTIONS,), int8
    selected_cards: np.ndarray  # (12,), int8


@dataclass
class Tokenizer:
    vocab: Vocab
    _selected: set[int] = field(default_factory=set)

    def tokenize(
        self,
        state: RunState,
        sub_phase: SubPhase,
        selected_cards: set[int] | None = None,
        action_mask: np.ndarray | None = None,
    ) -> RawObservation:
        """Convert RunState into RawObservation arrays."""
        from .constants import NUM_ACTIONS, SCALAR_DIM

        tokens = np.zeros((MAX_SEQ_LEN, TOKEN_DIM), dtype=np.int16)
        token_types = np.full(MAX_SEQ_LEN, TokenType.PAD, dtype=np.int8)
        attn_mask = np.zeros(MAX_SEQ_LEN, dtype=np.int8)
        scalars = np.zeros(SCALAR_DIM, dtype=np.float32)
        sel_cards = np.zeros(12, dtype=np.int8)

        if selected_cards is None:
            selected_cards = set()

        pos = 0

        # OBJ token (position 0)
        obj_id = 0  # default: win
        tokens[OBJ_START, 0] = obj_id
        token_types[OBJ_START] = TokenType.OBJ
        attn_mask[OBJ_START] = 1
        pos = 1

        # META tokens (positions 1-9)
        pos = META_START
        meta_values = self._encode_meta(state, sub_phase)
        for i, val in enumerate(meta_values):
            if pos + i >= DECK_START:
                break
            tokens[pos + i, 0] = val
            token_types[pos + i] = TokenType.META
            attn_mask[pos + i] = 1
        pos = META_START + META_COUNT

        # Scalars (continuous values for linear projection)
        scalars[0] = sign_log(float(state.dollars))
        scalars[1] = float(state.interest_cap)
        scalars[2] = float(state.round_resets.ante)
        scalars[3] = sign_log(float(self._blind_target(state)))
        scalars[4] = float(state.current_round.hands_left)
        scalars[5] = float(state.current_round.discards_left)
        scalars[6] = float(state.current_round.hand_size)
        scalars[7] = float(self._sub_phase_id(sub_phase))

        # DECK cards (positions 10-71, max 62)
        pos = DECK_START
        all_cards = self._gather_all_cards(state)
        for i, (card, location) in enumerate(all_cards[:DECK_MAX]):
            self._encode_card(tokens, pos + i, card, location, i, selected_cards, state)
            token_types[pos + i] = TokenType.DECK
            attn_mask[pos + i] = 1

        # JOKER tokens (positions 72-79, max 8)
        pos = JOKER_START
        for i, joker in enumerate(state.jokers[:JOKER_MAX]):
            self._encode_joker(tokens, pos + i, joker, i)
            token_types[pos + i] = TokenType.JOKER
            attn_mask[pos + i] = 1

        # VOUCHER tokens (positions 80-83, max 4)
        pos = VOUCHER_START
        voucher_keys = [k for k in state.used_vouchers if state.used_vouchers[k]]
        for i, vkey in enumerate(voucher_keys[:VOUCHER_MAX]):
            tokens[pos + i, 0] = self.vocab.voucher_to_id.get(vkey, 0)
            token_types[pos + i] = TokenType.VOUCHER
            attn_mask[pos + i] = 1

        # CONSUMABLE tokens (positions 84-88, max 5)
        pos = CONSUMABLE_START
        for i, cons in enumerate(state.consumables[:CONSUMABLE_MAX]):
            self._encode_consumable(tokens, pos + i, cons, i)
            token_types[pos + i] = TokenType.CONSUMABLE
            attn_mask[pos + i] = 1

        # SHOP tokens (positions 89-98, max 10) — only during SHOP
        if sub_phase == SubPhase.SHOP:
            pos = SHOP_START
            shop_items = self._gather_shop_items(state)
            for i, item in enumerate(shop_items[:SHOP_MAX]):
                self._encode_shop_item(tokens, pos + i, item, i)
                token_types[pos + i] = TokenType.SHOP
                attn_mask[pos + i] = 1

        # BLIND_SELECT tokens (positions 99-101, max 3) — only during BLIND_SELECT
        if sub_phase == SubPhase.BLIND_SELECT:
            pos = BLIND_SELECT_START
            blind_infos = self._gather_blind_choices(state)
            for i, binfo in enumerate(blind_infos[:BLIND_SELECT_MAX]):
                self._encode_blind_choice(tokens, pos + i, binfo)
                token_types[pos + i] = TokenType.BLIND_SELECT
                attn_mask[pos + i] = 1

        # Selected cards tracking
        for idx in selected_cards:
            if idx < 12:
                sel_cards[idx] = 1

        if action_mask is None:
            action_mask = np.ones(NUM_ACTIONS, dtype=np.int8)

        return RawObservation(
            tokens=tokens,
            token_types=token_types,
            scalars=scalars,
            attention_mask=attn_mask,
            action_mask=action_mask,
            selected_cards=sel_cards,
        )

    def _encode_meta(self, state: RunState, sub_phase: SubPhase) -> list[int]:
        """Encode 9 META token values."""
        blind = state.round_resets.blind or {}
        blind_type_id = {"Small": 0, "Big": 1}.get(state.blind_on_deck or "Small", 2)
        boss_key = state.round_resets.blind_choices.get("Boss", "")
        boss_id = self.vocab.boss_to_id.get(boss_key, 0)

        return [
            int(sign_log(float(state.dollars)) * 10),     # money (scaled)
            int(state.interest_cap),                       # interest_cap
            int(state.round_resets.ante),                  # ante
            blind_type_id * 100 + boss_id,                 # blind_type + boss_id combined
            int(sign_log(float(self._blind_target(state))) * 10),  # target (scaled)
            int(state.current_round.hands_left),           # hands_left
            int(state.current_round.discards_left),        # discards_left
            int(state.current_round.hand_size),            # hand_size
            self._sub_phase_id(sub_phase),                 # phase
        ]

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

    def _sub_phase_id(self, sub_phase: SubPhase) -> int:
        return {
            SubPhase.BLIND_SELECT: 0,
            SubPhase.CHOOSE_ACTION: 1,
            SubPhase.SELECT_CARDS: 2,
            SubPhase.SHOP: 3,
            SubPhase.BOOSTER_PACK: 4,
            SubPhase.CONSUMABLE_TARGET: 5,
        }[sub_phase]

    def _gather_all_cards(self, state: RunState) -> list[tuple[PlayingCard, int]]:
        """Collect all cards with location: 0=hand, 1=draw, 2=discard."""
        result: list[tuple[PlayingCard, int]] = []
        for card in state.hand_cards:
            result.append((card, 0))
        for card in state.draw_pile:
            result.append((card, 1))
        for card in state.discard_pile:
            result.append((card, 2))
        return result

    def _encode_card(
        self,
        tokens: np.ndarray,
        pos: int,
        card: PlayingCard,
        location: int,
        hand_idx: int,
        selected_cards: set[int],
        state: RunState,
    ) -> None:
        if card.face_down:
            tokens[pos, 0] = 0  # rank hidden
            tokens[pos, 1] = 0  # suit hidden
        else:
            tokens[pos, 0] = RANK_TO_ID.get(card.rank, 0)
            tokens[pos, 1] = SUIT_TO_ID.get(card.suit, 0)
        tokens[pos, 2] = self.vocab.enhancement_to_id.get(card.center_key, 0)
        tokens[pos, 3] = EDITION_TO_ID.get(card.edition_key or "", 0)
        tokens[pos, 4] = SEAL_TO_ID.get(card.seal or "", 0)
        tokens[pos, 5] = location
        tokens[pos, 6] = int(card.debuff)
        tokens[pos, 7] = int(card.face_down)
        tokens[pos, 8] = min(card.perma_bonus // 5, 31)  # bucket
        tokens[pos, 9] = int(location == 0 and hand_idx in selected_cards)
        tokens[pos, 10] = int(card.forced_selection)
        tokens[pos, 11] = hand_idx if location == 0 else 0

    def _encode_joker(
        self, tokens: np.ndarray, pos: int, joker: JokerInstance, slot: int
    ) -> None:
        tokens[pos, 0] = self.vocab.joker_to_id.get(joker.center_key, 0)
        # Rarity from center data — we encode as simple int
        tokens[pos, 1] = 0  # rarity placeholder (set by env if data available)
        edition_id = 0
        if joker.edition:
            for key, val in joker.edition.items():
                if val:
                    edition_id = EDITION_TO_ID.get(key, 0)
                    break
        tokens[pos, 2] = edition_id
        tokens[pos, 3] = min(joker.sell_cost, 31)
        # Counter value — varies by joker, use a generic bucket
        counter = max(joker.mult, joker.t_chips, joker.t_mult, int(joker.x_mult * 10))
        tokens[pos, 4] = min(counter, 255)
        tokens[pos, 5] = slot
        tokens[pos, 6] = int(joker.eternal)
        tokens[pos, 7] = int(joker.perishable)

    def _encode_consumable(
        self, tokens: np.ndarray, pos: int, cons: ConsumableInstance, slot: int
    ) -> None:
        tokens[pos, 0] = 0  # type_id set below
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
        """Flatten shop cards + vouchers + boosters into one list."""
        items: list[ShopCard] = []
        items.extend(state.shop.cards)
        items.extend(state.shop.vouchers)
        items.extend(state.shop.boosters)
        return items

    def _encode_shop_item(
        self, tokens: np.ndarray, pos: int, item: ShopCard, slot: int
    ) -> None:
        # Try to identify item type and use appropriate vocab
        item_id = 0
        item_type = 0  # 0=card, 1=joker, 2=consumable, 3=voucher, 4=booster
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
        """Gather blind choice info for BLIND_SELECT tokens."""
        result: list[dict[str, Any]] = []
        for bt in ("Small", "Big", "Boss"):
            blind_state = state.round_resets.blind_states.get(bt, "Upcoming")
            blind_key = state.round_resets.blind_choices.get(bt, "")
            tag_key = state.round_resets.blind_tags.get(bt, "")
            blind_data = state.data.blinds.get(blind_key, {})
            result.append({
                "blind_type": bt,
                "blind_key": blind_key,
                "state": blind_state,
                "mult": blind_data.get("mult", 1),
                "boss": blind_data.get("boss", False),
                "tag_key": tag_key,
            })
        return result

    def _encode_blind_choice(
        self, tokens: np.ndarray, pos: int, info: dict[str, Any]
    ) -> None:
        blind_type_id = {"Small": 0, "Big": 1, "Boss": 2}.get(info["blind_type"], 0)
        state_id = {"Select": 0, "Upcoming": 1, "Current": 2, "Skipped": 3, "Defeated": 4}.get(info["state"], 0)
        boss_id = self.vocab.boss_to_id.get(info["blind_key"], 0) if info["boss"] else 0
        tag_id = self.vocab.tag_to_id.get(info["tag_key"], 0)
        tokens[pos, 0] = blind_type_id
        tokens[pos, 1] = state_id
        tokens[pos, 2] = int(info["mult"] * 10)
        tokens[pos, 3] = boss_id
        tokens[pos, 4] = tag_id
