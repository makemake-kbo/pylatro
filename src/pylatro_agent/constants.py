"""Constants and enumerations for the Balatro agent."""

from __future__ import annotations

from enum import IntEnum, StrEnum

# Sequence / observation constants
MAX_SEQ_LEN = 160
TOKEN_DIM = 12
SCALAR_DIM = 8  # reserved for future continuous scalar features

# Action space layout — flat Discrete(71) with masking
NUM_ACTIONS = 71

MAX_HAND_SIZE = 12
MAX_JOKER_SLOTS = 8
MAX_CONSUMABLE_SLOTS = 5
MAX_SHOP_ITEMS = 10
MAX_PACK_CARDS = 5
MAX_DECK_CARDS = 62
MAX_VOUCHERS = 4


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
    PLAY_HAND = 3
    DISCARD = 4
    USE_CONSUMABLE = 5
    # SELECT_CARDS (6..17 = toggle card 0-11, 18 = confirm)
    TOGGLE_CARD_START = 6
    TOGGLE_CARD_END = 17
    SELECT_CONFIRM = 18
    # CONSUMABLE_TARGET (19..23 = slot, 24..35 = hand target, 36..40 = joker target, 41 = confirm, 42 = cancel)
    CONSUMABLE_SLOT_START = 19
    CONSUMABLE_SLOT_END = 23
    CONSUMABLE_HAND_TARGET_START = 24
    CONSUMABLE_HAND_TARGET_END = 35
    CONSUMABLE_JOKER_TARGET_START = 36
    CONSUMABLE_JOKER_TARGET_END = 40
    CONSUMABLE_CONFIRM = 41
    CONSUMABLE_CANCEL = 42
    # SHOP (43..52 = buy item, 53 = reroll, 54..58 = sell joker, 59..63 = sell consumable, 64 = leave)
    SHOP_BUY_START = 43
    SHOP_BUY_END = 52
    SHOP_REROLL = 53
    SHOP_SELL_JOKER_START = 54
    SHOP_SELL_JOKER_END = 58
    SHOP_SELL_CONSUMABLE_START = 59
    SHOP_SELL_CONSUMABLE_END = 63
    SHOP_LEAVE = 64
    # BOOSTER_PACK (65..69 = claim 0-4, 70 = skip)
    PACK_CLAIM_START = 65
    PACK_CLAIM_END = 69
    PACK_SKIP = 70


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
    PAD = 8


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
PAD_START = 102
