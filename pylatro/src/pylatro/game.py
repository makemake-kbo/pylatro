from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import floor, log10
from typing import Any

from .data import GameData, load_game_data
from .rng import PseudorandomState


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


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass(slots=True)
class StartingParams:
    dollars: int = 4
    hand_size: int = 8
    discards: int = 3
    hands: int = 4
    reroll_cost: int = 5
    joker_slots: int = 5
    ante_scaling: int = 1
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
    blind_on_deck: str = "Small"
    current_voucher: str | None = None
    tags: list[str] = field(default_factory=list)
    joker_keys: list[str] = field(default_factory=list)
    consumable_keys: list[str] = field(default_factory=list)
    deck_cards: list[PlayingCard] = field(default_factory=list)
    hands: dict[str, dict[str, Any]] = field(default_factory=dict)
    won_ante: int = 8
    first_shop_buffoon: bool = False
    pack: PackState | None = None

    def __post_init__(self) -> None:
        self.pseudorandom = PseudorandomState(self.seed)
        self.bosses_used = {
            key: 0 for key, blind in self.data.blinds.items() if blind.get("boss")
        }
        self.hands = deepcopy(self.data.hands)

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


def get_blind_amount(ante: int, scaling: int | None = None) -> int:
    k = 0.75
    scaling = scaling or 1
    if scaling == 1:
        amounts = [300, 800, 2000, 5000, 11000, 20000, 35000, 50000]
    elif scaling == 2:
        amounts = [300, 900, 2600, 8000, 20000, 36000, 60000, 100000]
    elif scaling == 3:
        amounts = [300, 1000, 3200, 9000, 25000, 60000, 110000, 200000]
    else:
        raise ValueError(f"Unsupported blind scaling {scaling}")

    if ante < 1:
        return 100
    if ante <= 8:
        return amounts[ante - 1]

    a, b, c, d = amounts[7], 1.6, ante - 8, 1 + 0.2 * (ante - 8)
    amount = floor(a * (b + (k * c) ** d) ** c)
    amount -= amount % (10 ** floor(log10(amount) - 1))
    return amount


def _has_showman(state: RunState) -> bool:
    return state.has_joker("Showman")


def _mark_center_used(state: RunState, center_key: str) -> None:
    state.used_jokers[center_key] = True


def _edition_cost(edition: dict[str, bool] | None) -> int:
    if not edition:
        return 0
    return (
        (3 if edition.get("holo") else 0)
        + (2 if edition.get("foil") else 0)
        + (5 if edition.get("polychrome") else 0)
        + (5 if edition.get("negative") else 0)
    )


def _calculate_cost(
    state: RunState,
    center: dict[str, Any],
    *,
    edition: dict[str, bool] | None = None,
    rental: bool = False,
) -> int:
    base_cost = int(center.get("cost", 1) or 1)
    cost = max(
        1,
        floor((base_cost + state.inflation + _edition_cost(edition) + 0.5) * (100 - state.discount_percent) / 100),
    )
    if center.get("set") == "Booster" and state.modifiers.get("booster_ante_scaling"):
        cost += state.round_resets.ante - 1
    if center.get("set") == "Planet" or (center.get("set") == "Booster" and "Celestial" in center["name"]):
        if state.has_joker("Astronomer"):
            cost = 0
    if rental:
        cost = 1
    return cost


def _apply_voucher_to_run(state: RunState, center_key: str) -> None:
    center = state.data.centers[center_key]
    extra = _as_dict(center.get("config")).get("extra")
    name = center["name"]

    if name in {"Overstock", "Overstock Plus"}:
        state.shop.joker_max += 1
    elif name in {"Tarot Merchant", "Tarot Tycoon"}:
        state.tarot_rate = 4 * extra
    elif name in {"Planet Merchant", "Planet Tycoon"}:
        state.planet_rate = 4 * extra
    elif name in {"Hone", "Glow Up"}:
        state.edition_rate = extra
    elif name in {"Magic Trick", "Illusion"}:
        state.playing_card_rate = extra
    elif name == "Crystal Ball":
        state.starting_params.consumable_slots += 1
    elif name in {"Clearance Sale", "Liquidation"}:
        state.discount_percent = extra
    elif name in {"Reroll Surplus", "Reroll Glut"}:
        state.starting_params.reroll_cost -= extra
    elif name in {"Seed Money", "Money Tree"}:
        state.interest_cap = extra
    elif name in {"Grabber", "Nacho Tong"}:
        state.starting_params.hands += extra
    elif name in {"Wasteful", "Recyclomancy"}:
        state.starting_params.discards += extra
    elif name == "Antimatter":
        state.starting_params.joker_slots += 1
    elif name in {"Paint Brush", "Palette"}:
        state.starting_params.hand_size += 1
    elif name in {"Hieroglyph", "Petroglyph"}:
        state.round_resets.ante -= extra
        state.round_resets.blind_ante -= extra
        if name == "Hieroglyph":
            state.starting_params.hands -= extra
        else:
            state.starting_params.discards -= extra


def _apply_deck(state: RunState) -> None:
    deck = state.data.centers[state.deck_key]
    config = _as_dict(deck.get("config"))

    if state.stake >= 2:
        state.modifiers.setdefault("no_blind_reward", {})["Small"] = True
    if state.stake >= 3:
        state.modifiers["scaling"] = 2
    if state.stake >= 4:
        state.modifiers["enable_eternals_in_shop"] = True
    if state.stake >= 5:
        state.starting_params.discards -= 1
    if state.stake >= 6:
        state.modifiers["scaling"] = 3
    if state.stake >= 7:
        state.modifiers["enable_perishables_in_shop"] = True
    if state.stake >= 8:
        state.modifiers["enable_rentals_in_shop"] = True

    if voucher := config.get("voucher"):
        state.used_vouchers[voucher] = True
        _apply_voucher_to_run(state, voucher)
    for voucher_key in config.get("vouchers", []):
        state.used_vouchers[voucher_key] = True
        _apply_voucher_to_run(state, voucher_key)
    for consumable_key in config.get("consumables", []):
        state.consumable_keys.append(consumable_key)

    if hands := config.get("hands"):
        state.starting_params.hands += hands
    if dollars := config.get("dollars"):
        state.starting_params.dollars += dollars
    if config.get("remove_faces"):
        state.starting_params.no_faces = True
    if spectral_rate := config.get("spectral_rate"):
        state.spectral_rate = spectral_rate
    if discards := config.get("discards"):
        state.starting_params.discards += discards
    if config.get("randomize_rank_suit"):
        state.starting_params.erratic_suits_and_ranks = True
    if joker_slots := config.get("joker_slot"):
        state.starting_params.joker_slots += joker_slots
    if hand_size := config.get("hand_size"):
        state.starting_params.hand_size += hand_size
    if ante_scaling := config.get("ante_scaling"):
        state.starting_params.ante_scaling = ante_scaling
    if consumable_slot := config.get("consumable_slot"):
        state.starting_params.consumable_slots += consumable_slot
    if config.get("no_interest"):
        state.modifiers["no_interest"] = True
    if extra_hand_bonus := config.get("extra_hand_bonus"):
        state.modifiers["money_per_hand"] = extra_hand_bonus
    if extra_discard_bonus := config.get("extra_discard_bonus"):
        state.modifiers["money_per_discard"] = extra_discard_bonus


def _iter_starting_card_controls(state: RunState) -> list[dict[str, str | None]]:
    base_card_keys = list(state.data.cards.keys())
    controls: list[dict[str, str | None]] = []
    iterations = len(base_card_keys)

    for index in range(iterations):
        card_key = base_card_keys[index]
        if state.starting_params.erratic_suits_and_ranks:
            _, card_key = state.pseudorandom.pseudorandom_element(
                state.data.cards,
                state.pseudorandom.pseudoseed("erratic"),
            )
        suit = card_key[0]
        rank = card_key[2]
        if state.starting_params.no_faces and rank in {"J", "Q", "K"}:
            continue
        controls.append({"s": suit, "r": rank, "e": None, "d": None, "g": None})

    controls.sort(key=lambda card: f"{card['s']}{card['r']}{card['e'] or ''}{card['d'] or ''}{card['g'] or ''}")
    return controls


def _build_starting_deck(state: RunState) -> None:
    controls = _iter_starting_card_controls(state)
    state.deck_cards = [
        PlayingCard(
            front_key=f"{card['s']}_{card['r']}",
            suit=state.data.cards[f"{card['s']}_{card['r']}"]["suit"],
            rank=card["r"],
            center_key=card["e"] or "c_base",
            edition_key=card["d"],
            seal=card["g"],
        )
        for card in controls
    ]

    if state.data.centers[state.deck_key]["name"] == "Checkered Deck":
        for card in state.deck_cards:
            if card.suit == "Clubs":
                card.suit = "Spades"
                card.front_key = f"S_{card.rank}"
            elif card.suit == "Diamonds":
                card.suit = "Hearts"
                card.front_key = f"H_{card.rank}"

    shuffled = state.pseudorandom.pseudoshuffle(
        [
            {
                "front_key": card.front_key,
                "suit": card.suit,
                "rank": card.rank,
                "center_key": card.center_key,
                "edition_key": card.edition_key,
                "seal": card.seal,
            }
            for card in state.deck_cards
        ],
        state.pseudorandom.pseudoseed("shuffle"),
    )
    state.deck_cards = [
        PlayingCard(
            front_key=card["front_key"],
            suit=card["suit"],
            rank=card["rank"],
            center_key=card["center_key"],
            edition_key=card["edition_key"],
            seal=card["seal"],
        )
        for card in shuffled
    ]


def create_run_state(seed: str, stake: int = 1, deck_key: str = "b_red", data: GameData | None = None) -> RunState:
    game_data = data or load_game_data()
    state = RunState(data=game_data, seed=seed, stake=stake, deck_key=deck_key)
    _apply_deck(state)

    state.round_resets.hands = state.starting_params.hands
    state.round_resets.discards = state.starting_params.discards
    state.round_resets.reroll_cost = state.starting_params.reroll_cost
    state.dollars = state.starting_params.dollars
    state.base_reroll_cost = state.starting_params.reroll_cost
    state.current_round.reroll_cost = state.base_reroll_cost

    state.round_resets.blind_choices["Boss"] = get_new_boss(state)
    state.current_voucher = get_next_voucher_key(state)
    state.round_resets.blind_tags["Small"] = get_next_tag_key(state)
    state.round_resets.blind_tags["Big"] = get_next_tag_key(state)

    _build_starting_deck(state)
    state.current_round.discards_left = state.round_resets.discards
    state.current_round.hands_left = state.round_resets.hands
    return state


def get_current_pool(
    state: RunState,
    card_type: str,
    rarity: float | None = None,
    legendary: bool | None = None,
    append: str | None = None,
) -> tuple[list[str], str]:
    if card_type == "Joker":
        rarity_roll = rarity if rarity is not None else float(
            state.pseudorandom.pseudorandom(f"rarity{state.round_resets.ante}{append or ''}")
        )
        rarity_index = 4 if legendary else 3 if rarity_roll > 0.95 else 2 if rarity_roll > 0.7 else 1
        starting_pool = state.data.joker_rarity_pools[rarity_index]
        pool_key = f"Joker{rarity_index}{'' if legendary else append or ''}"
    else:
        starting_pool = state.data.center_pools[card_type]
        pool_key = f"{card_type}{append or ''}"

    pool: list[str] = []
    pool_size = 0
    for proto in starting_pool:
        add = False
        if card_type == "Enhanced":
            add = True
        elif card_type == "Demo":
            add = bool(proto.get("pos") and proto.get("config"))
        elif card_type == "Tag":
            add = (
                (not proto.get("requires") or state.data.centers.get(proto["requires"], {}).get("discovered"))
                and (not proto.get("min_ante") or proto["min_ante"] <= state.round_resets.ante)
            )
        elif not (state.used_jokers.get(proto["key"]) and not _has_showman(state)) and (
            proto.get("unlocked", True) or proto.get("rarity") == 4
        ):
            if proto.get("set") == "Voucher":
                if not state.used_vouchers.get(proto["key"]):
                    add = True
                    for required in proto.get("requires", []):
                        if not state.used_vouchers.get(required):
                            add = False
                    for voucher in state.shop.vouchers:
                        if voucher.center_key == proto["key"]:
                            add = False
            elif proto.get("set") == "Planet":
                config = _as_dict(proto.get("config"))
                add = not config.get("softlock", False) or state.hands[config["hand_type"]]["played"] > 0
            elif proto.get("enhancement_gate"):
                add = any(card.center_key == proto["enhancement_gate"] for card in state.deck_cards)
            else:
                add = True
            if proto["name"] in {"Black Hole", "The Soul"}:
                add = False

        if proto.get("no_pool_flag") and state.pool_flags.get(proto["no_pool_flag"]):
            add = False
        if proto.get("yes_pool_flag") and not state.pool_flags.get(proto["yes_pool_flag"]):
            add = False
        if add and not state.banned_keys.get(proto["key"]):
            pool.append(proto["key"])
            pool_size += 1
        else:
            pool.append("UNAVAILABLE")

    if pool_size == 0:
        fallback = {
            "Tarot": "c_strength",
            "Tarot_Planet": "c_strength",
            "Planet": "c_pluto",
            "Spectral": "c_incantation",
            "Joker": "j_joker",
            "Voucher": "v_blank",
            "Tag": "tag_handy",
        }
        pool = [fallback.get(card_type, "j_joker")]
    suffix = "" if legendary else str(state.round_resets.ante)
    return pool, f"{pool_key}{suffix}"


def _pick_pool_key(state: RunState, pool: list[str] | dict[str, Any], pool_key: str) -> str:
    center, _ = state.pseudorandom.pseudorandom_element(pool, state.pseudorandom.pseudoseed(pool_key))
    reroll = 1
    while center == "UNAVAILABLE":
        reroll += 1
        center, _ = state.pseudorandom.pseudorandom_element(
            pool,
            state.pseudorandom.pseudoseed(f"{pool_key}_resample{reroll}"),
        )
    return center


def get_next_voucher_key(state: RunState, from_tag: bool = False) -> str:
    pool, pool_key = get_current_pool(state, "Voucher")
    return _pick_pool_key(state, pool, "Voucher_fromtag" if from_tag else pool_key)


def get_next_tag_key(state: RunState, append: str | None = None) -> str:
    pool, pool_key = get_current_pool(state, "Tag", append=append)
    return _pick_pool_key(state, pool, pool_key)


def get_new_boss(state: RunState) -> str:
    eligible: dict[str, int | bool] = {}
    ante = max(1, state.round_resets.ante)

    for key, blind in state.data.blinds.items():
        boss = blind.get("boss")
        if not boss:
            continue
        if not boss.get("showdown") and (boss["min"] <= ante and (ante % state.won_ante != 0 or state.round_resets.ante < 2)):
            eligible[key] = True
        elif boss.get("showdown") and ante % state.won_ante == 0 and state.round_resets.ante >= 2:
            eligible[key] = True

    for key in list(eligible):
        if state.banned_keys.get(key):
            del eligible[key]

    min_use = 100
    for key, uses in state.bosses_used.items():
        if key in eligible:
            eligible[key] = uses
            min_use = min(min_use, uses)

    eligible = {key: value for key, value in eligible.items() if value == min_use}
    _, boss = state.pseudorandom.pseudorandom_element(eligible, state.pseudorandom.pseudoseed("boss"))
    state.bosses_used[boss] += 1
    return boss


def get_pack(state: RunState, key: str | None = None, pack_type: str | None = None) -> str:
    if not state.first_shop_buffoon and not state.banned_keys.get("p_buffoon_normal_1"):
        state.first_shop_buffoon = True
        return f"p_buffoon_normal_{int(state.pseudorandom.random_without_seed(1, 2))}"

    cumulative = 0.0
    for proto in state.data.center_pools["Booster"]:
        if (not pack_type or pack_type == proto["kind"]) and not state.banned_keys.get(proto["key"]):
            cumulative += proto.get("weight", 1)

    poll = float(state.pseudorandom.pseudorandom(state.pseudorandom.pseudoseed(f"{key or 'pack_generic'}{state.round_resets.ante}"))) * cumulative
    current = 0.0
    for proto in state.data.center_pools["Booster"]:
        if state.banned_keys.get(proto["key"]):
            continue
        if pack_type and pack_type != proto["kind"]:
            continue
        weight = proto.get("weight", 1)
        current += weight
        if current >= poll:
            return proto["key"]
    raise RuntimeError("Booster selection failed")


def poll_edition(
    state: RunState,
    key: str = "edition_generic",
    mod: float = 1,
    no_negative: bool = False,
    guaranteed: bool = False,
) -> dict[str, bool] | None:
    edition_poll = float(state.pseudorandom.pseudorandom(state.pseudorandom.pseudoseed(key)))
    if guaranteed:
        if edition_poll > 1 - 0.003 * 25 and not no_negative:
            return {"negative": True}
        if edition_poll > 1 - 0.006 * 25:
            return {"polychrome": True}
        if edition_poll > 1 - 0.02 * 25:
            return {"holo": True}
        if edition_poll > 1 - 0.04 * 25:
            return {"foil": True}
        return None

    if edition_poll > 1 - 0.003 * mod and not no_negative:
        return {"negative": True}
    if edition_poll > 1 - 0.006 * state.edition_rate * mod:
        return {"polychrome": True}
    if edition_poll > 1 - 0.02 * state.edition_rate * mod:
        return {"holo": True}
    if edition_poll > 1 - 0.04 * state.edition_rate * mod:
        return {"foil": True}
    return None


def _apply_joker_stickers(state: RunState, center: dict[str, Any], source: str | None) -> tuple[bool, bool, bool]:
    eternal = False
    perishable = False
    rental = False

    if state.modifiers.get("all_eternal") and center.get("eternal_compat"):
        eternal = True

    if source in {"shop", "pack"}:
        eternal_key = ("packetper" if source == "pack" else "etperpoll") + str(state.round_resets.ante)
        eternal_poll = float(state.pseudorandom.pseudorandom(eternal_key))
        if state.modifiers.get("enable_eternals_in_shop") and eternal_poll > 0.7 and center.get("eternal_compat") and not perishable:
            eternal = True
        elif (
            state.modifiers.get("enable_perishables_in_shop")
            and 0.4 < eternal_poll <= 0.7
            and center.get("perishable_compat")
            and not eternal
        ):
            perishable = True

        rental_key = ("packssjr" if source == "pack" else "ssjr") + str(state.round_resets.ante)
        if state.modifiers.get("enable_rentals_in_shop") and float(state.pseudorandom.pseudorandom(rental_key)) > 0.7:
            rental = True

    return eternal, perishable, rental


def create_card_spec(
    state: RunState,
    card_type: str,
    forced_key: str | None = None,
    append: str | None = None,
    *,
    source: str | None = None,
    soulable: bool = False,
) -> ShopCard:
    requested_type = card_type
    if not forced_key and soulable and not state.banned_keys.get("c_soul"):
        if card_type in {"Tarot", "Spectral", "Tarot_Planet"} and not (
            state.used_jokers.get("c_soul") and not _has_showman(state)
        ):
            if float(state.pseudorandom.pseudorandom(f"soul_{card_type}{state.round_resets.ante}")) > 0.997:
                forced_key = "c_soul"
        if card_type in {"Planet", "Spectral"} and not (
            state.used_jokers.get("c_black_hole") and not _has_showman(state)
        ):
            if float(state.pseudorandom.pseudorandom(f"soul_{card_type}{state.round_resets.ante}")) > 0.997:
                forced_key = "c_black_hole"

    if card_type == "Base":
        forced_key = "c_base"

    if forced_key and not state.banned_keys.get(forced_key):
        center_key = forced_key
        center = state.data.centers[center_key]
        card_type = center.get("set") if center.get("set") not in {None, "Default"} else requested_type
    else:
        pool, pool_key = get_current_pool(state, card_type, append=append)
        center_key = _pick_pool_key(state, pool, pool_key)
        center = state.data.centers[center_key]

    front_key = None
    if card_type in {"Base", "Enhanced"}:
        _, front_key = state.pseudorandom.pseudorandom_element(
            state.data.cards,
            state.pseudorandom.pseudoseed(f"front{append or ''}{state.round_resets.ante}"),
        )
        if isinstance(front_key, dict):
            raise AssertionError("Front selection should return a card key")

    edition = None
    if card_type == "Joker":
        eternal, perishable, rental = _apply_joker_stickers(state, center, source)
        edition = poll_edition(state, key=f"edi{append or ''}{state.round_resets.ante}")
    else:
        eternal = False
        perishable = False
        rental = False

    _mark_center_used(state, center_key)

    return ShopCard(
        center_key=center_key,
        card_type=card_type,
        cost=_calculate_cost(state, center, edition=edition, rental=rental),
        base_cost=int(center.get("cost", 1) or 1),
        front_key=front_key if isinstance(front_key, str) else None,
        edition=edition,
        seal=None,
        eternal=eternal,
        perishable=perishable,
        rental=rental,
    )


def create_shop_card(state: RunState) -> ShopCard:
    total_rate = state.joker_rate + state.tarot_rate + state.planet_rate + state.playing_card_rate + state.spectral_rate
    polled_rate = float(state.pseudorandom.pseudorandom(state.pseudorandom.pseudoseed(f"cdt{state.round_resets.ante}"))) * total_rate
    running = 0.0

    card_types = [
        ("Joker", state.joker_rate),
        ("Tarot", state.tarot_rate),
        ("Planet", state.planet_rate),
        (
            "Enhanced"
            if state.used_vouchers.get("v_illusion") and float(state.pseudorandom.pseudorandom("illusion")) > 0.6
            else "Base",
            state.playing_card_rate,
        ),
        ("Spectral", state.spectral_rate),
    ]
    for card_type, value in card_types:
        if running < polled_rate <= running + value:
            card = create_card_spec(state, card_type, append="sho", source="shop")
            if card_type in {"Base", "Enhanced"} and state.used_vouchers.get("v_illusion") and float(state.pseudorandom.pseudorandom("illusion")) > 0.8:
                edition_poll = float(state.pseudorandom.pseudorandom("illusion"))
                if edition_poll > 1 - 0.15:
                    card.edition = {"polychrome": True}
                elif edition_poll > 0.5:
                    card.edition = {"holo": True}
                else:
                    card.edition = {"foil": True}
                card.cost = _calculate_cost(
                    state,
                    state.data.centers[card.center_key],
                    edition=card.edition,
                    rental=card.rental,
                )
            return card
        running += value
    raise RuntimeError("Shop card selection failed")


def populate_shop(state: RunState) -> ShopState:
    if not state.shop.cards:
        refresh_shop(state)

    if not state.shop.vouchers and state.current_voucher:
        voucher = create_card_spec(state, "Voucher", forced_key=state.current_voucher)
        voucher.shop_voucher = True
        state.shop.vouchers = [voucher]

    if not state.shop.boosters:
        boosters: list[ShopCard] = []
        while len(state.current_round.used_packs) < 2:
            state.current_round.used_packs.append("")
        for index in range(2):
            if not state.current_round.used_packs[index]:
                state.current_round.used_packs[index] = get_pack(state, "shop_pack")
            if state.current_round.used_packs[index] == "USED":
                continue
            booster = create_card_spec(state, "Booster", forced_key=state.current_round.used_packs[index])
            booster.booster_pos = index + 1
            boosters.append(booster)
        state.shop.boosters = boosters

    return state.shop


def refresh_shop(state: RunState) -> list[ShopCard]:
    state.shop.cards = [create_shop_card(state) for _ in range(state.shop.joker_max)]
    return state.shop.cards


def reroll_shop(state: RunState) -> list[ShopCard]:
    if state.current_round.reroll_cost > 0:
        state.dollars -= state.current_round.reroll_cost
    final_free = state.current_round.free_rerolls > 0
    state.current_round.free_rerolls = max(state.current_round.free_rerolls - 1, 0)
    state.calculate_reroll_cost(skip_increment=final_free)
    return refresh_shop(state)


def buy_shop_card(state: RunState, index: int) -> ShopCard:
    card = state.shop.cards.pop(index)
    state.dollars -= card.cost
    center = state.data.centers[card.center_key]
    if center.get("set") in {"Default", "Enhanced"}:
        if not card.front_key:
            raise ValueError("Playing card purchases require a front key")
        front = state.data.cards[card.front_key]
        state.deck_cards.append(
            PlayingCard(
                front_key=card.front_key,
                suit=front["suit"],
                rank=card.front_key[2],
                center_key=card.center_key,
                edition_key=next(iter(card.edition)) if card.edition else None,
            )
        )
    elif center.get("consumeable"):
        state.consumable_keys.append(card.center_key)
    else:
        state.joker_keys.append(card.center_key)
    return card


def open_booster_pack(state: RunState, index: int) -> PackState:
    booster = state.shop.boosters.pop(index)
    state.dollars -= booster.cost
    if booster.booster_pos is not None:
        while len(state.current_round.used_packs) < booster.booster_pos:
            state.current_round.used_packs.append("")
        state.current_round.used_packs[booster.booster_pos - 1] = "USED"

    center = state.data.centers[booster.center_key]
    name = center["name"]
    cards: list[ShopCard] = []
    state_name = "SHOP"
    if "Arcana" in name:
        state_name = "TAROT_PACK"
    elif "Celestial" in name:
        state_name = "PLANET_PACK"
    elif "Spectral" in name:
        state_name = "SPECTRAL_PACK"
    elif "Standard" in name:
        state_name = "STANDARD_PACK"
    elif "Buffoon" in name:
        state_name = "BUFFOON_PACK"

    size = _as_dict(center.get("config")).get("extra", 0)
    choices = _as_dict(center.get("config")).get("choose", 1)
    for card_index in range(1, size + 1):
        if "Arcana" in name:
            if state.used_vouchers.get("v_omen_globe") and float(state.pseudorandom.pseudorandom("omen_globe")) > 0.8:
                card = create_card_spec(state, "Spectral", append="ar2", source="pack", soulable=True)
            else:
                card = create_card_spec(state, "Tarot", append="ar1", source="pack", soulable=True)
        elif "Celestial" in name:
            forced_key = None
            if state.used_vouchers.get("v_telescope") and card_index == 1:
                hand_name = None
                hand_tally = 0
                for name_key in POKER_HANDS:
                    hand = state.hands[name_key]
                    if hand["visible"] and hand["played"] > hand_tally:
                        hand_name = name_key
                        hand_tally = hand["played"]
                if hand_name is not None:
                    for proto in state.data.center_pools["Planet"]:
                        if _as_dict(proto.get("config")).get("hand_type") == hand_name:
                            forced_key = proto["key"]
                            break
            card = create_card_spec(state, "Planet", forced_key=forced_key, append="pl1", source="pack", soulable=True)
        elif "Spectral" in name:
            card = create_card_spec(state, "Spectral", append="spe", source="pack", soulable=True)
        elif "Standard" in name:
            base_type = "Enhanced" if float(state.pseudorandom.pseudorandom(f"stdset{state.round_resets.ante}")) > 0.6 else "Base"
            card = create_card_spec(state, base_type, append="sta", source="pack", soulable=True)
            card.edition = poll_edition(state, key=f"standard_edition{state.round_resets.ante}", mod=2, no_negative=True)
            seal_poll = float(state.pseudorandom.pseudorandom(f"stdseal{state.round_resets.ante}"))
            if seal_poll > 1 - 0.02 * 10:
                seal_type = float(state.pseudorandom.pseudorandom(f"stdsealtype{state.round_resets.ante}"))
                if seal_type > 0.75:
                    card.seal = "Red"
                elif seal_type > 0.5:
                    card.seal = "Blue"
                elif seal_type > 0.25:
                    card.seal = "Gold"
                else:
                    card.seal = "Purple"
        elif "Buffoon" in name:
            card = create_card_spec(state, "Joker", append="buf", source="pack", soulable=True)
        else:
            raise RuntimeError(f"Unknown booster pack {name}")

        if card.edition or card.rental:
            card.cost = _calculate_cost(
                state,
                state.data.centers[card.center_key],
                edition=card.edition,
                rental=card.rental,
            )
        cards.append(card)

    state.pack = PackState(
        booster_key=booster.center_key,
        state_name=state_name,
        choices_remaining=choices,
        cards=cards,
        source_slot=booster.booster_pos,
    )
    return state.pack


def redeem_voucher(state: RunState, voucher_key: str) -> None:
    state.used_vouchers[voucher_key] = True
    if state.current_voucher == voucher_key:
        state.current_voucher = None
    state.shop.vouchers = [voucher for voucher in state.shop.vouchers if voucher.center_key != voucher_key]
    _apply_voucher_to_run(state, voucher_key)
    state.round_resets.hands = state.starting_params.hands
    state.round_resets.discards = state.starting_params.discards
    state.round_resets.reroll_cost = state.starting_params.reroll_cost
    state.base_reroll_cost = state.starting_params.reroll_cost
    state.calculate_reroll_cost(skip_increment=True)


def select_blind(state: RunState, blind_type: str | None = None) -> None:
    blind_type = blind_type or state.blind_on_deck
    blind_key = state.round_resets.blind_choices[blind_type]
    state.round_resets.blind = state.data.blinds[blind_key]
    state.round_resets.blind_states[blind_type] = "Current"
    state.shop.cards = []
    state.shop.vouchers = []
    state.shop.boosters = []
    state.pack = None
    state.current_round.discards_left = max(0, state.round_resets.discards)
    state.current_round.hands_left = max(1, state.round_resets.hands)
    state.current_round.hands_played = 0
    state.current_round.discards_used = 0
    state.current_round.reroll_cost_increase = 0
    state.current_round.used_packs = []
    state.current_round.free_rerolls = sum(
        1 for key in state.joker_keys if state.data.centers[key]["name"] == "Chaos the Clown"
    )
    state.calculate_reroll_cost(skip_increment=True)
    state.current_round.dollars = 0


def skip_blind(state: RunState) -> str:
    skipped = state.blind_on_deck
    skip_to = "Big" if skipped == "Small" else "Boss"
    state.skips += 1
    if tag := state.round_resets.blind_tags.get(skipped):
        state.tags.append(tag)
    state.round_resets.blind_states[skipped] = "Skipped"
    state.round_resets.blind_states[skip_to] = "Select"
    state.blind_on_deck = skip_to
    return skip_to


def reroll_boss(state: RunState, from_tag: bool = False) -> str:
    state.round_resets.boss_rerolled = True
    if not from_tag:
        state.dollars -= 10
    state.round_resets.blind_choices["Boss"] = get_new_boss(state)
    return state.round_resets.blind_choices["Boss"]


def reset_blinds(state: RunState) -> None:
    if state.round_resets.blind_states["Boss"] == "Defeated":
        state.round_resets.blind_states = {"Small": "Upcoming", "Big": "Upcoming", "Boss": "Upcoming"}
        state.blind_on_deck = "Small"
        state.round_resets.blind_choices["Boss"] = get_new_boss(state)
        state.round_resets.boss_rerolled = False


def cash_out(state: RunState) -> None:
    state.current_round.jokers_purchased = 0
    state.current_round.discards_left = max(0, state.round_resets.discards)
    state.current_round.hands_left = max(1, state.round_resets.hands)
    state.shop.cards = []
    state.shop.vouchers = []
    state.shop.boosters = []
    state.pack = None
    state.current_round.used_packs = []
    if state.round_resets.blind_states["Boss"] == "Defeated":
        state.round_resets.blind_ante = state.round_resets.ante
        state.current_voucher = get_next_voucher_key(state)
        state.round_resets.blind_tags["Small"] = get_next_tag_key(state)
        state.round_resets.blind_tags["Big"] = get_next_tag_key(state)
    reset_blinds(state)
