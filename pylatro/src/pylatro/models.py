from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .data import GameData  # noqa: TC001
from .rng import PseudorandomState


class BlindType(StrEnum):
    Small = "Small"
    Big = "Big"
    Boss = "Boss"


class BlindState(StrEnum):
    Select = "Select"
    Upcoming = "Upcoming"
    Current = "Current"
    Skipped = "Skipped"
    Defeated = "Defeated"


class Edition(StrEnum):
    foil = "foil"
    holo = "holo"
    polychrome = "polychrome"
    negative = "negative"


class Seal(StrEnum):
    Red = "Red"
    Blue = "Blue"
    Gold = "Gold"
    Purple = "Purple"


class PackStateName(StrEnum):
    SHOP = "SHOP"
    TAROT_PACK = "TAROT_PACK"
    PLANET_PACK = "PLANET_PACK"
    SPECTRAL_PACK = "SPECTRAL_PACK"
    STANDARD_PACK = "STANDARD_PACK"
    BUFFOON_PACK = "BUFFOON_PACK"


POKER_HANDS = (
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


@dataclass(slots=True)
class StartingParams:
    dollars: int = 4
    hand_size: int = 8
    discards: int = 3
    hands: int = 4
    reroll_cost: int = 5
    joker_slots: int = 5
    ante_scaling: float = 1
    consumable_slots: int = 2
    no_faces: bool = False
    erratic_suits_and_ranks: bool = False


@dataclass(slots=True)
class CurrentRound:
    hands_left: int = 0
    hands_played: int = 0
    discards_left: int = 0
    discards_used: int = 0
    reroll_cost: int = 5
    reroll_cost_increase: int = 0
    free_rerolls: int = 0
    dollars: int = 0
    jokers_purchased: int = 0
    idol_card: dict[str, Any] = field(default_factory=lambda: {"suit": "Spades", "rank": "Ace"})
    mail_card: dict[str, Any] = field(default_factory=lambda: {"rank": "Ace"})
    ancient_card: dict[str, Any] = field(default_factory=lambda: {"suit": "Spades"})
    castle_card: dict[str, Any] = field(default_factory=lambda: {"suit": "Spades"})
    used_packs: list[str] = field(default_factory=list)
    round_dollars: int = 0
    most_played_poker_hand: str = "High Card"


@dataclass(slots=True)
class RoundResets:
    hands: int = 1
    discards: int = 1
    reroll_cost: int = 1
    temp_reroll_cost: int | None = None
    temp_handsize: int | None = None
    ante: int = 1
    blind_ante: int = 1
    blind_states: dict[str, str] = field(
        default_factory=lambda: {"Small": "Select", "Big": "Upcoming", "Boss": "Upcoming"}
    )
    blind_choices: dict[str, str] = field(default_factory=lambda: {"Small": "bl_small", "Big": "bl_big"})
    blind_tags: dict[str, str] = field(default_factory=dict)
    boss_rerolled: bool = False
    blind: dict[str, Any] | None = None


@dataclass(slots=True)
class PlayingCard:
    front_key: str
    suit: str
    rank: str
    center_key: str = "c_base"
    edition_key: str | None = None
    seal: str | None = None

    @property
    def is_face(self) -> bool:
        return self.rank in {"J", "Q", "K"}


@dataclass(slots=True)
class ShopCard:
    center_key: str
    card_type: str
    cost: int
    base_cost: int
    front_key: str | None = None
    edition: dict[str, bool] | None = None
    seal: str | None = None
    eternal: bool = False
    perishable: bool = False
    rental: bool = False
    shop_voucher: bool = False
    booster_pos: int | None = None


@dataclass(slots=True)
class ShopState:
    joker_max: int = 2
    cards: list[ShopCard] = field(default_factory=list)
    vouchers: list[ShopCard] = field(default_factory=list)
    boosters: list[ShopCard] = field(default_factory=list)


@dataclass(slots=True)
class PackState:
    booster_key: str
    state_name: str
    choices_remaining: int
    cards: list[ShopCard] = field(default_factory=list)
    source_slot: int | None = None


@dataclass(slots=True)
class RunState:
    data: GameData
    seed: str
    stake: int = 1
    deck_key: str = "b_red"
    modifiers: dict[str, Any] = field(default_factory=dict)
    probabilities: dict[str, float] = field(default_factory=lambda: {"normal": 1})
    starting_params: StartingParams = field(default_factory=StartingParams)
    current_round: CurrentRound = field(default_factory=CurrentRound)
    round_resets: RoundResets = field(default_factory=RoundResets)
    shop: ShopState = field(default_factory=ShopState)
    pseudorandom: PseudorandomState = field(init=False)
    bosses_used: dict[str, int] = field(init=False)
    pool_flags: dict[str, bool] = field(default_factory=dict)
    used_jokers: dict[str, bool] = field(default_factory=dict)
    used_vouchers: dict[str, bool] = field(default_factory=dict)
    banned_keys: dict[str, bool] = field(default_factory=dict)
    current_boss_streak: int = 0
    base_reroll_cost: int = 5
    edition_rate: float = 1
    joker_rate: float = 20
    tarot_rate: float = 4
    planet_rate: float = 4
    spectral_rate: float = 0
    playing_card_rate: float = 0
    interest_cap: int = 25
    discount_percent: int = 0
    inflation: int = 0
    pack_size: int = 2
    skips: int = 0
    dollars: int = 0
    bankrupt_at: int = 0
    blind_on_deck: str | None = None
    current_voucher: str | None = None
    tags: list[str] = field(default_factory=list)
    joker_keys: list[str] = field(default_factory=list)
    consumable_keys: list[str] = field(default_factory=list)
    deck_cards: list[PlayingCard] = field(default_factory=list)
    hands: dict[str, dict[str, Any]] = field(default_factory=dict)
    win_ante: int = 8
    round: int = 0
    won: bool = False
    interest_amount: int = 1
    perishable_rounds: int = 5
    rental_rate: int = 3
    consumeable_buffer: int = 0
    joker_buffer: int = 0
    max_jokers: int = 0
    starting_deck_size: int = 52
    ecto_minus: float = 1
    tag_tally: int = 0
    hands_played: int = 0
    unused_discards: int = 0
    last_tarot_planet: str | None = None
    previous_round: dict[str, Any] = field(default_factory=lambda: {"dollars": 4})
    round_bonus: dict[str, int] = field(default_factory=lambda: {"next_hands": 0, "discards": 0})
    cards_played: dict[str, dict[str, Any]] = field(init=False)
    first_shop_buffoon: bool = False
    pack: PackState | None = None

    def __post_init__(self) -> None:
        self.pseudorandom = PseudorandomState(self.seed)
        self.bosses_used = {key: 0 for key, blind in self.data.blinds.items() if blind.get("boss")}
        self.hands = deepcopy(self.data.hands)
        self.cards_played = {
            rank: {"suits": {}, "total": 0}
            for rank in ("Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King")
        }

    def has_joker(self, name: str) -> bool:
        return any(self.data.centers[key]["name"] == name for key in self.joker_keys)

    def calculate_reroll_cost(self, skip_increment: bool = False) -> None:
        if self.current_round.free_rerolls < 0:
            self.current_round.free_rerolls = 0
        if self.current_round.free_rerolls > 0:
            self.current_round.reroll_cost = 0
            return
        if not skip_increment:
            self.current_round.reroll_cost_increase += 1
        base = self.round_resets.temp_reroll_cost or self.round_resets.reroll_cost
        self.current_round.reroll_cost = base + self.current_round.reroll_cost_increase
