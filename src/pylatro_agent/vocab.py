"""Build vocabulary mappings from GameData."""

from __future__ import annotations

from dataclasses import dataclass, field

from pylatro import GameData


@dataclass(slots=True)
class Vocab:
    joker_to_id: dict[str, int] = field(default_factory=dict)
    consumable_to_id: dict[str, int] = field(default_factory=dict)
    voucher_to_id: dict[str, int] = field(default_factory=dict)
    boss_to_id: dict[str, int] = field(default_factory=dict)
    booster_to_id: dict[str, int] = field(default_factory=dict)
    tag_to_id: dict[str, int] = field(default_factory=dict)
    enhancement_to_id: dict[str, int] = field(default_factory=dict)

    # Standard card feature sizes (includes 0 = none/unknown)
    rank_size: int = 14   # 13 ranks + unknown
    suit_size: int = 5    # 4 suits + unknown
    edition_size: int = 6  # none, foil, holo, polychrome, negative, unknown
    seal_size: int = 6     # none, Red, Blue, Gold, Purple, unknown

    @property
    def joker_vocab_size(self) -> int:
        return len(self.joker_to_id) + 1  # +1 for unknown/pad

    @property
    def consumable_vocab_size(self) -> int:
        return len(self.consumable_to_id) + 1

    @property
    def voucher_vocab_size(self) -> int:
        return len(self.voucher_to_id) + 1

    @property
    def boss_vocab_size(self) -> int:
        return len(self.boss_to_id) + 1

    @property
    def booster_vocab_size(self) -> int:
        return len(self.booster_to_id) + 1

    @property
    def tag_vocab_size(self) -> int:
        return len(self.tag_to_id) + 1

    @property
    def enhancement_vocab_size(self) -> int:
        return len(self.enhancement_to_id) + 1


RANK_TO_ID = {
    "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8,
    "T": 9, "J": 10, "Q": 11, "K": 12, "A": 13,
}

SUIT_TO_ID = {"Diamonds": 1, "Clubs": 2, "Hearts": 3, "Spades": 4}

EDITION_TO_ID = {"foil": 1, "holo": 2, "polychrome": 3, "negative": 4}

SEAL_TO_ID = {"Red": 1, "Blue": 2, "Gold": 3, "Purple": 4}


def build_vocab(data: GameData) -> Vocab:
    """Build vocabulary ID mappings from GameData."""
    vocab = Vocab()

    # Jokers
    idx = 1
    for key, center in data.centers.items():
        if center.get("set") == "Joker":
            vocab.joker_to_id[key] = idx
            idx += 1

    # Consumables (Tarot, Planet, Spectral)
    idx = 1
    for key, center in data.centers.items():
        if center.get("consumeable"):
            vocab.consumable_to_id[key] = idx
            idx += 1

    # Vouchers
    idx = 1
    for key, center in data.centers.items():
        if center.get("set") == "Voucher":
            vocab.voucher_to_id[key] = idx
            idx += 1

    # Boss blinds
    idx = 1
    for key, blind in data.blinds.items():
        if blind.get("boss"):
            vocab.boss_to_id[key] = idx
            idx += 1

    # Boosters
    idx = 1
    for key, center in data.centers.items():
        if center.get("set") == "Booster":
            vocab.booster_to_id[key] = idx
            idx += 1

    # Tags
    idx = 1
    for key in data.tags:
        vocab.tag_to_id[key] = idx
        idx += 1

    # Enhancements (center keys for card modifiers)
    idx = 1
    for key, center in data.centers.items():
        if key.startswith("m_") or key == "c_base":
            vocab.enhancement_to_id[key] = idx
            idx += 1

    return vocab
