"""Constants and enumerations for the Balatro agent."""

from __future__ import annotations

from enum import IntEnum, StrEnum

# Sequence / observation constants
MAX_SEQ_LEN = 160
TOKEN_DIM = 12
SCALAR_DIM = 8  # reserved for future continuous scalar features

# Action space layout — flat Discrete(88) with masking
NUM_ACTIONS = 88

MAX_HAND_SIZE = 12
MAX_JOKER_SLOTS = 8
MAX_CONSUMABLE_SLOTS = 5
MAX_SHOP_ITEMS = 10
MAX_PACK_CARDS = 5
MAX_DECK_CARDS = 62
MAX_VOUCHERS = 4
MAX_HAND_LEVELS = 12
MAX_PLAY_CANDIDATES = 16
MAX_DISCARD_CANDIDATES = 16
MAX_HAND_CANDIDATES = MAX_PLAY_CANDIDATES + MAX_DISCARD_CANDIDATES

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
    CONSUMABLE_TARGET = "consumable_target"


class ActionRange(IntEnum):
    """Starting index of each action group."""
    # BLIND_SELECT
    BLIND_PLAY = 0
    BLIND_SKIP = 1
    BLIND_REROLL = 2
    # CHOOSE_ACTION
    PLAY_CANDIDATE_START = 3
    PLAY_CANDIDATE_END = 18
    DISCARD_CANDIDATE_START = 19
    DISCARD_CANDIDATE_END = 34
    USE_CONSUMABLE = 35
    # CONSUMABLE_TARGET (36..40 = slot, 41..52 = hand target, 53..57 = joker target, 58 = confirm, 59 = cancel)
    CONSUMABLE_SLOT_START = 36
    CONSUMABLE_SLOT_END = 40
    CONSUMABLE_HAND_TARGET_START = 41
    CONSUMABLE_HAND_TARGET_END = 52
    CONSUMABLE_JOKER_TARGET_START = 53
    CONSUMABLE_JOKER_TARGET_END = 57
    CONSUMABLE_CONFIRM = 58
    CONSUMABLE_CANCEL = 59
    # SHOP (60..69 = buy item, 70 = reroll, 71..75 = sell joker, 76..80 = sell consumable, 81 = leave)
    SHOP_BUY_START = 60
    SHOP_BUY_END = 69
    SHOP_REROLL = 70
    SHOP_SELL_JOKER_START = 71
    SHOP_SELL_JOKER_END = 75
    SHOP_SELL_CONSUMABLE_START = 76
    SHOP_SELL_CONSUMABLE_END = 80
    SHOP_LEAVE = 81
    # BOOSTER_PACK (82..86 = claim 0-4, 87 = skip)
    PACK_CLAIM_START = 82
    PACK_CLAIM_END = 86
    PACK_SKIP = 87


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
PAD_START = 146
