"""Constants and enumerations for the Balatro agent."""

from __future__ import annotations

import math
from enum import IntEnum, StrEnum

# Version of the tokenizer observation format. Bump whenever tokens,
# token_types, scalars, hand_candidates, or action-mask layout changes in
# a way that would break checkpoints trained against the previous shape.
# Stamped into every checkpoint; `load_checkpoint` asserts it on read.
TOKENIZER_VERSION = 3

# Sequence / observation constants
MAX_SEQ_LEN = 160
TOKEN_DIM = 13
SCALAR_DIM = 11  # continuous scalar features

MAX_HAND_SIZE = 16
MAX_JOKER_SLOTS = 8
MAX_CONSUMABLE_SLOTS = 5
MAX_SHOP_ITEMS = 10
MAX_PACK_CARDS = 5
MAX_PLAY_CANDIDATES = 16
MAX_DISCARD_CANDIDATES = 16
NUM_HAND_SUBSETS = sum(math.comb(MAX_HAND_SIZE, k) for k in range(1, 6))

# Consumables now commit atomically in a single action: the policy picks
# (slot, targeting_choice) in one step instead of walking the old
# CONSUMABLE_TARGET sub-phase. The game caps any base-set consumable at 3
# highlighted cards (The Moon / Star / Sun / World), so enumerating size
# 1..3 subsets covers every targeting need; a consumable needs hand
# subset XOR a single joker XOR no target at all (verified against
# game_data.json, no consumable requires both).
MAX_CONSUMABLE_HAND_TARGETS = 3
NUM_CONSUMABLE_HAND_SUBSETS = sum(
    math.comb(MAX_HAND_SIZE, k) for k in range(1, MAX_CONSUMABLE_HAND_TARGETS + 1)
)  # 16 + 120 + 560 = 696
CONSUMABLE_ACTIONS_PER_SLOT = 1 + NUM_CONSUMABLE_HAND_SUBSETS + MAX_JOKER_SLOTS  # 705
CONSUMABLE_NO_TARGET_OFFSET = 0
CONSUMABLE_HAND_SUBSET_OFFSET = 1
CONSUMABLE_JOKER_OFFSET = 1 + NUM_CONSUMABLE_HAND_SUBSETS

# Base-game consumables whose only legal target is a single joker (no hand
# cards). The mask / heuristic / fast runner all need to agree on this set;
# keep the single source of truth here.
JOKER_TARGET_CONSUMABLE_NAMES = frozenset(
    {"The Wheel of Fortune", "Ectoplasm", "Hex", "Ankh"}
)

# Consumables that require hand-card targets but do not declare
# config.max_highlighted in game_data.json.
HAND_TARGET_CONSUMABLE_LIMITS = {
    "Aura": (1, 1),
}

POKER_HAND_NAMES = (
    "Flush Five",
    "Flush House",
    "Five of a Kind",
    "Straight Flush",
    "Four of a Kind",
    "Full House",
    "Flush",
    "Straight",
    "Three of a Kind",
    "Two Pair",
    "Pair",
    "High Card",
)


class SubPhase(StrEnum):
    BLIND_SELECT = "blind_select"
    CHOOSE_ACTION = "choose_action"
    SELECT_CARDS = "select_cards"
    SHOP = "shop"
    BOOSTER_PACK = "booster_pack"


_BLIND_PLAY = 0
_BLIND_SKIP = 1
_BLIND_REROLL = 2
_PLAY_SUBSET_START = 3
_PLAY_SUBSET_END = _PLAY_SUBSET_START + NUM_HAND_SUBSETS - 1
_DISCARD_SUBSET_START = _PLAY_SUBSET_END + 1
_DISCARD_SUBSET_END = _DISCARD_SUBSET_START + NUM_HAND_SUBSETS - 1
# Flat atomic consumable actions: MAX_CONSUMABLE_SLOTS contiguous
# per-slot blocks, each laid out as
#   [no_target (1), hand_subset (NUM_CONSUMABLE_HAND_SUBSETS), joker (MAX_JOKER_SLOTS)]
# so slot k owns _CONSUMABLE_FLAT_START + k*CONSUMABLE_ACTIONS_PER_SLOT .. (next-1).
_CONSUMABLE_FLAT_START = _DISCARD_SUBSET_END + 1
_CONSUMABLE_FLAT_END = (
    _CONSUMABLE_FLAT_START
    + MAX_CONSUMABLE_SLOTS * CONSUMABLE_ACTIONS_PER_SLOT
    - 1
)
_SHOP_BUY_START = _CONSUMABLE_FLAT_END + 1
_SHOP_BUY_END = _SHOP_BUY_START + MAX_SHOP_ITEMS - 1
_SHOP_REROLL = _SHOP_BUY_END + 1
_SHOP_SELL_JOKER_START = _SHOP_REROLL + 1
_SHOP_SELL_JOKER_END = _SHOP_SELL_JOKER_START + MAX_JOKER_SLOTS - 1
_SHOP_SELL_CONSUMABLE_START = _SHOP_SELL_JOKER_END + 1
_SHOP_SELL_CONSUMABLE_END = _SHOP_SELL_CONSUMABLE_START + MAX_CONSUMABLE_SLOTS - 1
_SHOP_LEAVE = _SHOP_SELL_CONSUMABLE_END + 1
_PACK_CLAIM_START = _SHOP_LEAVE + 1
_PACK_CLAIM_END = _PACK_CLAIM_START + MAX_PACK_CARDS - 1
_PACK_SKIP = _PACK_CLAIM_END + 1
_MOVE_JOKER_START = _PACK_SKIP + 1
_MOVE_JOKER_END = _MOVE_JOKER_START + MAX_JOKER_SLOTS * (MAX_JOKER_SLOTS - 1) - 1


class ActionRange(IntEnum):
    """Starting index of each action group."""
    BLIND_PLAY = _BLIND_PLAY
    BLIND_SKIP = _BLIND_SKIP
    BLIND_REROLL = _BLIND_REROLL
    PLAY_SUBSET_START = _PLAY_SUBSET_START
    PLAY_SUBSET_END = _PLAY_SUBSET_END
    DISCARD_SUBSET_START = _DISCARD_SUBSET_START
    DISCARD_SUBSET_END = _DISCARD_SUBSET_END
    CONSUMABLE_FLAT_START = _CONSUMABLE_FLAT_START
    CONSUMABLE_FLAT_END = _CONSUMABLE_FLAT_END
    SHOP_BUY_START = _SHOP_BUY_START
    SHOP_BUY_END = _SHOP_BUY_END
    SHOP_REROLL = _SHOP_REROLL
    SHOP_SELL_JOKER_START = _SHOP_SELL_JOKER_START
    SHOP_SELL_JOKER_END = _SHOP_SELL_JOKER_END
    SHOP_SELL_CONSUMABLE_START = _SHOP_SELL_CONSUMABLE_START
    SHOP_SELL_CONSUMABLE_END = _SHOP_SELL_CONSUMABLE_END
    SHOP_LEAVE = _SHOP_LEAVE
    PACK_CLAIM_START = _PACK_CLAIM_START
    PACK_CLAIM_END = _PACK_CLAIM_END
    PACK_SKIP = _PACK_SKIP
    MOVE_JOKER_START = _MOVE_JOKER_START
    MOVE_JOKER_END = _MOVE_JOKER_END


NUM_ACTIONS = int(ActionRange.MOVE_JOKER_END) + 1


# Token type IDs
class TokenType(IntEnum):
    OBJ = 0
    META = 1
    DECK = 2
    JOKER = 3
    VOUCHER = 4
    CONSUMABLE = 5
    SHOP = 6
    BLIND_SELECT = 7
    HAND_LEVEL = 8
    HAND_CANDIDATE = 9
    PAD = 10


# Token position ranges (start positions)
OBJ_START = 0
META_START = 1
META_COUNT = 9
DECK_START = 10
DECK_MAX = 62
JOKER_START = 72
JOKER_MAX = 8
VOUCHER_START = 80
VOUCHER_MAX = 4
CONSUMABLE_START = 84
CONSUMABLE_MAX = 5
SHOP_START = 89
SHOP_MAX = 10
BLIND_SELECT_START = 99
BLIND_SELECT_MAX = 3
HAND_LEVEL_START = 102
HAND_LEVEL_MAX = 12
HAND_CANDIDATE_START = 114
HAND_CANDIDATE_MAX = 32
