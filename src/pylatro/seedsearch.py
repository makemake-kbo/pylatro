"""Seed searching: find seeds whose vouchers, shops, blinds, and skips match a JSON spec.

A spec describes, per ante, what the player wants to see: the ante voucher, the
boss, skip tags / skip packs (and their contents), and shop items within a
number of rerolls. Only what the spec mentions is constrained; everything else
is generated but ignored.

The simulator drives a "phantom" run: every blind not marked as a skip is
assumed beaten without playing a hand (hand RNG lives on separate pseudoseed
keys, so shop / voucher / tag / pack streams are unaffected), every shop is
visited, spec'd vouchers are bought at the first shop that offers them, and the
shop is rerolled only as far as the deepest ``within_N`` constraint requires
(stopping early once everything is found).
"""

from __future__ import annotations

import random
import re
from copy import deepcopy
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from .blind import cash_out, select_blind, skip_blind
from .data import GameData, load_game_data
from .models import RunState, ShopCard
from .run import create_run_state
from .runtime import create_joker_spec
from .shop import open_booster_pack, populate_shop, redeem_voucher, reroll_shop

SEED_ALPHABET = "123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
SEED_LENGTH = 8

# Tag granted by skipping a blind -> the free pack it opens (kind, size).
TAG_PACKS: dict[str, tuple[str, str]] = {
    "tag_charm": ("Arcana", "mega"),
    "tag_meteor": ("Celestial", "mega"),
    "tag_standard": ("Standard", "mega"),
    "tag_buffoon": ("Buffoon", "mega"),
    "tag_ethereal": ("Spectral", "normal"),
}
PACK_KINDS = ("Arcana", "Celestial", "Standard", "Buffoon", "Spectral")
PACK_SIZES = ("normal", "jumbo", "mega")

BLIND_ALIASES = {
    "small": "Small",
    "smallblind": "Small",
    "big": "Big",
    "bigblind": "Big",
    "boss": "Boss",
    "bossblind": "Boss",
}

# Joker nicknames. Fun is legal
NICKNAMES = {
    "trib": "j_triboulet",
    "chigoat": "j_chicot",
    "pogbaronpog": "j_baron",
    "useless": "j_loyalty_card",
    "photo": "j_photograph",
    "chad": "j_hanging_chad",
    "spaceman": "j_space",
    "stonks": "j_to_the_moon",
}

_RANK_NAMES = {
    "A": "ace",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "T": "ten",
    "J": "jack",
    "Q": "queen",
    "K": "king",
}


class SpecError(ValueError):
    """The search spec is malformed or violates game rules."""


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


# ---------------------------------------------------------------------------
# Name resolution


@dataclass(frozen=True)
class Item:
    raw: str
    key: str  # canonical center key or playing-card front key
    set_name: str  # center "set" (Joker/Tarot/...) or "PlayingCard"
    legendary: bool = False


class Resolver:
    """Resolves user-friendly names ("overstock", "mega arcana", "hermit") to game keys.

    Indexes normalized aliases per entry: raw key, key without type prefix
    ("c_hermit" -> "hermit"), display name, name minus "The "/" Tag", plus
    NICKNAMES. Aliases can collide, so centers map to key sets.
    """

    def __init__(self, data: GameData) -> None:
        self.data = data
        self._centers: dict[str, set[str]] = {}
        self._tags: dict[str, str] = {}
        self._bosses: dict[str, str] = {}
        self._cards: dict[str, str] = {}

        for key, center in data.centers.items():
            if not center.get("set"):  # stub entries like "soul" carry no set
                continue
            self._add(self._centers, _norm(key), key)
            # Also index without the type prefix: "c_hermit" -> "hermit".
            prefix, _, rest = key.partition("_")
            if len(prefix) == 1 and rest:
                self._add(self._centers, _norm(rest), key)
            if name := center.get("name"):
                self._add(self._centers, _norm(str(name)), key)
                if str(name).startswith("The "):
                    self._add(self._centers, _norm(str(name)[4:]), key)

        for alias, key in NICKNAMES.items():
            if key in data.centers:
                self._add(self._centers, _norm(alias), key)

        for key, tag in data.tags.items():
            self._tags[_norm(key)] = key
            name = str(tag.get("name", ""))
            self._tags[_norm(name)] = key
            self._tags[_norm(name.removesuffix(" Tag"))] = key

        for key, blind in data.blinds.items():
            if not blind.get("boss"):
                continue
            name = str(blind.get("name", ""))
            self._bosses[_norm(key)] = key
            self._bosses[_norm(name)] = key
            self._bosses[_norm(name.removeprefix("The "))] = key

        for key, card in data.cards.items():
            self._cards[_norm(key)] = key
            suit = str(card.get("suit", ""))
            rank = key[2:]
            if rank in _RANK_NAMES and suit:
                self._cards[f"{_RANK_NAMES[rank]}of{_norm(suit)}"] = key

    @staticmethod
    def _add(index: dict[str, set[str]], norm: str, key: str) -> None:
        index.setdefault(norm, set()).add(key)

    def center(self, raw: str) -> str:
        """Resolve a name to a single center key; exact key match wins ties,
        unknown names get did-you-mean suggestions."""
        norm = _norm(raw)
        keys = self._centers.get(norm)
        if not keys:
            hints = get_close_matches(norm, self._centers, n=3)
            candidates = ", ".join(sorted({k for h in hints for k in self._centers[h]}))
            hint = f" (did you mean: {candidates}?)" if hints else ""
            raise SpecError(f"Unknown item {raw!r}{hint}")
        if len(keys) > 1:
            exact = [k for k in keys if _norm(k) == norm]
            if len(exact) == 1:
                return exact[0]
            raise SpecError(f"Ambiguous item {raw!r}: matches {sorted(keys)}")
        return next(iter(keys))

    def item(self, raw: str) -> Item:
        """Resolve a name to an Item (center or playing card; centers win clashes)."""
        norm = _norm(raw)
        if norm in self._cards and norm not in self._centers:
            return Item(raw=raw, key=self._cards[norm], set_name="PlayingCard")
        key = self.center(raw)
        center = self.data.centers[key]
        return Item(
            raw=raw,
            key=key,
            set_name=str(center.get("set", "")),
            legendary=center.get("rarity") == 4,
        )

    def voucher(self, raw: str) -> str:
        key = self.center(raw)
        if self.data.centers[key].get("set") != "Voucher":
            raise SpecError(f"{raw!r} is not a voucher")
        return key

    def tag(self, raw: str) -> str:
        norm = _norm(raw)
        if norm not in self._tags:
            hints = get_close_matches(norm, self._tags, n=3)
            hint = f" (did you mean: {', '.join(sorted({self._tags[h] for h in hints}))}?)" if hints else ""
            raise SpecError(f"Unknown tag {raw!r}{hint}")
        return self._tags[norm]

    def boss(self, raw: str) -> str:
        norm = _norm(raw)
        if norm not in self._bosses:
            raise SpecError(f"Unknown boss blind {raw!r}")
        return self._bosses[norm]

    def pack(self, raw: str) -> tuple[str, str | None, list[str]]:
        """Resolve a pack name (exact key or loose like "megaarcana") to
        (kind, size or None, matching booster keys). No size = any size."""
        norm = _norm(raw)
        booster_keys = [proto["key"] for proto in self.data.center_pools["Booster"]]
        if norm in {_norm(k) for k in booster_keys}:
            key = next(k for k in booster_keys if _norm(k) == norm)
            _, kind, size, _ = key.split("_")
            return kind.capitalize(), size, [key]
        kind = next((k for k in PACK_KINDS if _norm(k) in norm), None)
        if kind is None:
            raise SpecError(f"Unknown pack {raw!r} (expected e.g. 'mega arcana', 'spectral', 'p_buffoon_jumbo_1')")
        size = next((s for s in PACK_SIZES if s in norm.replace(_norm(kind), "")), None)
        keys = [k for k in booster_keys if f"_{kind.lower()}_" in k and (size is None or f"_{size}_" in k)]
        return kind, size, keys


# ---------------------------------------------------------------------------
# Spec model


@dataclass
class PackReq:
    raw: str
    kind: str
    size: str | None
    keys: list[str]
    contains: list[Item] = field(default_factory=list)


@dataclass
class SkipReq:
    tag_key: str | None = None
    pack: PackReq | None = None


@dataclass
class ShopReq:
    within: dict[int, list[Item]] = field(default_factory=dict)
    packs: list[PackReq] = field(default_factory=list)


@dataclass
class BlindReq:
    skip: SkipReq | None = None
    shop: ShopReq | None = None


@dataclass
class AnteReq:
    voucher: str | None = None
    boss: str | None = None
    blinds: dict[str, BlindReq] = field(default_factory=dict)


@dataclass
class SearchSpec:
    antes: dict[int, AnteReq]
    deck_key: str = "b_red"
    stake: int = 1
    # Search as a fully-unlocked profile (all vouchers/jokers/tags obtainable),
    # like the real game's seeded runs. Set "unlocked": false in the spec to
    # search with only the engine's default (locked) pools.
    unlock_all: bool = True

    @property
    def max_ante(self) -> int:
        return max(self.antes, default=0)


def unlock_data(data: GameData) -> GameData:
    """Return a deep copy of `data` with every center unlocked and discovered."""
    data = deepcopy(data)
    # center_pools / joker_rarity_pools hold their own proto dicts (parsed
    # separately from the same JSON), so unlock every copy.
    protos = list(data.centers.values())
    for pool in data.center_pools.values():
        protos.extend(pool)
    for pool in data.joker_rarity_pools.values():
        protos.extend(pool)
    for proto in protos:
        proto["unlocked"] = True
        proto["discovered"] = True
    return data


# ---------------------------------------------------------------------------
# Spec parsing / validation


def _require_keys(obj: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(obj) - allowed
    if unknown:
        raise SpecError(f"Unknown key(s) {sorted(unknown)} in {where}; allowed: {sorted(allowed)}")


def _parse_ante_number(raw: str | int) -> int:
    text = str(raw).lower().removeprefix("ante").strip()
    if not text.isdigit() or int(text) < 1:
        raise SpecError(f"Bad ante key {raw!r}: use 'ante1', 'ante2', ... or plain numbers >= 1")
    return int(text)


def _validate_pack_item(item: Item, kind: str, where: str) -> None:
    """Reject items that can't spawn in this pack kind. A legendary in a
    soulable pack means "The Soul appears and becomes this joker"."""
    ok = False
    if kind == "Arcana":
        ok = item.set_name == "Tarot" or item.key == "c_soul" or item.legendary
    elif kind == "Celestial":
        ok = item.set_name == "Planet" or item.key == "c_black_hole"
    elif kind == "Spectral":
        ok = item.set_name == "Spectral" or item.key in {"c_soul", "c_black_hole"} or item.legendary
    elif kind == "Buffoon":
        ok = item.set_name == "Joker" and not item.legendary
    elif kind == "Standard":
        ok = item.set_name == "PlayingCard"
    if not ok:
        raise SpecError(
            f"{item.raw!r} ({item.set_name}{', legendary' if item.legendary else ''}) "
            f"cannot appear in a {kind} pack ({where})"
        )


def _parse_pack(raw: Any, resolver: Resolver, where: str, *, for_skip: bool) -> PackReq:
    """Parse a bare pack name or {"pack": ..., "contains": [...]}. for_skip
    pins the size to what the skip tag grants (mega, or normal for Ethereal)."""
    if isinstance(raw, str):
        obj: dict[str, Any] = {"pack": raw}
    elif isinstance(raw, dict):
        obj = dict(raw)
    else:
        raise SpecError(f"Bad pack spec in {where}: expected a name or object, got {raw!r}")
    _require_keys(obj, {"pack", "contains"}, where)
    if "pack" not in obj:
        raise SpecError(f"Pack spec in {where} is missing 'pack'")

    kind, size, keys = resolver.pack(str(obj["pack"]))
    if for_skip:
        skip_size = dict(TAG_PACKS.values()).get(kind)
        if skip_size is None:
            raise SpecError(f"A {kind} pack cannot be granted by a blind skip ({where})")
        if size is not None and size != skip_size:
            raise SpecError(f"Skipping a blind grants a {skip_size} {kind} pack, not {size} ({where})")
        size = skip_size
        booster_keys = [proto["key"] for proto in resolver.data.center_pools["Booster"]]
        keys = [k for k in booster_keys if f"_{kind.lower()}_{size}_" in k]

    contains_raw = obj.get("contains", [])
    if not isinstance(contains_raw, list):
        raise SpecError(f"'contains' in {where} must be a list")
    contains = [resolver.item(str(entry)) for entry in contains_raw]
    for item in contains:
        _validate_pack_item(item, kind, where)
    return PackReq(raw=str(obj["pack"]), kind=kind, size=size, keys=keys, contains=contains)


def _parse_shop(raw: Any, resolver: Resolver, where: str) -> ShopReq:
    """Parse a 'shop' object: "packs" = required booster slots, "within_N" =
    items seen within N rolls (roll 1 = initial shop), "contains" = within_1.
    Items must be valid shop cards (no vouchers, boosters, or legendaries)."""
    if not isinstance(raw, dict):
        raise SpecError(f"'shop' in {where} must be an object")
    req = ShopReq()
    for key, value in raw.items():
        if key == "packs":
            if not isinstance(value, list):
                raise SpecError(f"'packs' in {where} must be a list")
            req.packs = [_parse_pack(entry, resolver, f"{where}.packs", for_skip=False) for entry in value]
            continue
        if key == "contains":
            rolls = 1
        else:
            match = re.fullmatch(r"within[_ ]?(\d+)", key)
            if not match:
                raise SpecError(f"Unknown shop key {key!r} in {where}; use 'contains', 'within_N', or 'packs'")
            rolls = int(match.group(1))
            if rolls < 1:
                raise SpecError(f"'{key}' in {where}: roll count must be >= 1")
        if not isinstance(value, list):
            raise SpecError(f"'{key}' in {where} must be a list of items")
        items = [resolver.item(str(entry)) for entry in value]
        for item in items:
            if item.set_name == "Voucher":
                raise SpecError(
                    f"{item.raw!r} is a voucher; use the ante-level 'voucher' field, not shop cards ({where})"
                )
            if item.set_name == "Booster":
                raise SpecError(f"{item.raw!r} is a booster pack; use the shop 'packs' field ({where})")
            if item.legendary:
                raise SpecError(f"{item.raw!r} is a legendary joker and can never appear in the shop ({where})")
            if item.set_name not in {"Joker", "Tarot", "Planet", "Spectral", "PlayingCard", "Enhanced"}:
                raise SpecError(f"{item.raw!r} ({item.set_name}) can never appear as a shop card ({where})")
        req.within.setdefault(rolls, []).extend(items)
    return req


def _parse_blind(raw: Any, resolver: Resolver, blind: str, where: str) -> BlindReq:
    """Parse one blind's 'skip'/'shop' constraints. A skip pack implies its
    tag (one tag per pack kind, see TAG_PACKS) and is cross-checked against an
    explicit 'tag'. skip and shop are exclusive: a skipped blind has no shop."""
    if not isinstance(raw, dict):
        raise SpecError(f"{where} must be an object")
    _require_keys(raw, {"skip", "shop"}, where)
    req = BlindReq()
    if "skip" in raw:
        if blind == "Boss":
            raise SpecError(f"The boss blind cannot be skipped ({where})")
        skip_raw = raw["skip"]
        if not isinstance(skip_raw, dict):
            raise SpecError(f"'skip' in {where} must be an object")
        _require_keys(skip_raw, {"tag", "pack", "contains"}, f"{where}.skip")
        skip = SkipReq()
        if "tag" in skip_raw:
            skip.tag_key = resolver.tag(str(skip_raw["tag"]))
        if "pack" in skip_raw or "contains" in skip_raw:
            if "pack" not in skip_raw:
                raise SpecError(f"'contains' in {where}.skip requires a 'pack'")
            skip.pack = _parse_pack(
                {"pack": skip_raw["pack"], "contains": skip_raw.get("contains", [])},
                resolver,
                f"{where}.skip",
                for_skip=True,
            )
            pack_tag = next(
                tag for tag, (kind, size) in TAG_PACKS.items() if kind == skip.pack.kind and size == skip.pack.size
            )
            if skip.tag_key is not None and skip.tag_key != pack_tag:
                raise SpecError(
                    f"{where}.skip: tag {skip.tag_key} does not grant a "
                    f"{skip.pack.size} {skip.pack.kind} pack ({pack_tag} does)"
                )
            skip.tag_key = pack_tag
        req.skip = skip  # a skip with no tag/pack constraint just skips the blind
    if "shop" in raw:
        if req.skip is not None:
            raise SpecError(f"{where}: a skipped blind has no shop")
        req.shop = _parse_shop(raw["shop"], resolver, where)
    return req


def _validate_vouchers(spec: SearchSpec, resolver: Resolver) -> None:
    """Voucher schedule rules: no duplicates; tiered vouchers need their
    prerequisite in a strictly earlier ante (the higher tier only enters the
    pool after the lower is redeemed); an ante-1 voucher needs an ante-1 shop,
    i.e. small and big can't both be skipped."""
    schedule: dict[str, int] = {}
    for ante, areq in sorted(spec.antes.items()):
        if not areq.voucher:
            continue
        if areq.voucher in schedule:
            raise SpecError(
                f"Voucher {areq.voucher} requested in both ante {schedule[areq.voucher]} and ante {ante}; "
                "each voucher appears at most once per run"
            )
        schedule[areq.voucher] = ante
    for voucher, ante in schedule.items():
        for required in resolver.data.centers[voucher].get("requires", []) or []:
            required_ante = schedule.get(required)
            if required_ante is None or required_ante >= ante:
                name = resolver.data.centers[voucher]["name"]
                req_name = resolver.data.centers[required]["name"]
                raise SpecError(
                    f"Voucher {name!r} (ante {ante}) requires {req_name!r} to be redeemed first: "
                    f'add "voucher": "{required}" to an earlier ante'
                )
    # A voucher is buyable at the first shop of its ante. For ante N > 1 that is
    # the post-boss shop of ante N-1 (the boss is never skippable), but ante 1
    # has no earlier shop: it needs the small or big blind to be played.
    ante1 = spec.antes.get(1)
    if ante1 and ante1.voucher:
        both_skipped = all((ante1.blinds.get(blind) and ante1.blinds[blind].skip) for blind in ("Small", "Big"))
        if both_skipped:
            raise SpecError(
                "Ante 1 requests a voucher but skips both the small and big blinds, "
                "so no ante-1 shop ever opens to buy it from"
            )


def parse_spec(raw: dict[str, Any], data: GameData | None = None) -> SearchSpec:
    """Parse and validate a JSON search spec. Raises SpecError on rule violations."""
    if not isinstance(raw, dict):
        raise SpecError("Spec must be a JSON object")
    data = data or load_game_data()
    resolver = Resolver(data)

    raw = dict(raw)
    deck_key = "b_red"
    stake = 1
    unlock_all = bool(raw.pop("unlocked", True))
    if "deck" in raw:
        deck_key = resolver.center(str(raw.pop("deck")))
        if data.centers[deck_key].get("set") != "Back":
            raise SpecError(f"{deck_key!r} is not a deck")
    if "stake" in raw:
        stake = int(raw.pop("stake"))
        if not 1 <= stake <= 8:
            raise SpecError(f"Stake must be 1-8, got {stake}")

    ante_specs: dict[str | int, Any] = raw.pop("antes", None) or raw
    antes: dict[int, AnteReq] = {}
    for ante_key, ante_raw in ante_specs.items():
        ante = _parse_ante_number(ante_key)
        where = f"ante{ante}"
        if not isinstance(ante_raw, dict):
            raise SpecError(f"{where} must be an object")
        allowed = {"voucher", "boss"} | set(BLIND_ALIASES)
        _require_keys({_norm(k): v for k, v in ante_raw.items()}, allowed, where)
        areq = AnteReq()
        for key, value in ante_raw.items():
            norm = _norm(key)
            if norm == "voucher":
                areq.voucher = resolver.voucher(str(value))
            elif norm == "boss":
                areq.boss = resolver.boss(str(value))
            else:
                blind = BLIND_ALIASES[norm]
                if blind in areq.blinds:
                    raise SpecError(f"Duplicate blind {blind!r} in {where}")
                areq.blinds[blind] = _parse_blind(value, resolver, blind, f"{where}.{key}")
        antes[ante] = areq

    if not antes:
        raise SpecError("Spec contains no ante constraints")
    spec = SearchSpec(antes=antes, deck_key=deck_key, stake=stake, unlock_all=unlock_all)
    _validate_vouchers(spec, resolver)
    if not unlock_all:
        for key in _referenced_center_keys(spec):
            if not data.centers[key].get("unlocked", True):
                raise SpecError(
                    f"{data.centers[key]['name']!r} ({key}) is locked in the engine's default pools and "
                    'can never appear; remove "unlocked": false from the spec to search a full profile'
                )
    return spec


def _referenced_center_keys(spec: SearchSpec) -> set[str]:
    keys: set[str] = set()

    def add_items(items: list[Item]) -> None:
        keys.update(item.key for item in items if item.set_name != "PlayingCard")

    for areq in spec.antes.values():
        if areq.voucher:
            keys.add(areq.voucher)
        for breq in areq.blinds.values():
            if breq.skip and breq.skip.pack:
                add_items(breq.skip.pack.contains)
            if breq.shop:
                for items in breq.shop.within.values():
                    add_items(items)
                for preq in breq.shop.packs:
                    add_items(preq.contains)
    return keys


# ---------------------------------------------------------------------------
# Seed simulation


@dataclass
class SeedMatch:
    seed: str
    notes: list[str] = field(default_factory=list)


def _card_keys(cards: list[ShopCard]) -> set[str]:
    keys: set[str] = set()
    for card in cards:
        keys.add(card.center_key)
        if card.front_key:
            keys.add(card.front_key)
    return keys


def _check_pack_contents(state: RunState, cards: list[ShopCard], preq: PackReq, notes: list[str], where: str) -> bool:
    """Check an opened pack against preq.contains. A requested legendary is
    matched by peeking at what each Soul in the pack would create; the peek
    makes the same RNG call a real Soul use would. Plain "soul" just matches
    the c_soul key, no peek."""
    keys = _card_keys(cards)
    wanted_legendaries = [item for item in preq.contains if item.legendary]
    generated: list[str] = []
    if wanted_legendaries:
        # Each Soul card in the pack, when used, creates one legendary joker.
        # Assume the player uses every Soul the pack contains.
        souls = sum(1 for card in cards if card.center_key == "c_soul")
        for _ in range(souls):
            generated.append(create_joker_spec(state, legendary=True, append="sou").center_key)
    for item in preq.contains:
        if item.legendary:
            if item.key in generated:
                continue
            return False
        if item.key not in keys:
            return False
    contents = ", ".join(card.center_key for card in cards)
    legend = f" -> souls became: {', '.join(generated)}" if generated else ""
    notes.append(f"{where}: {preq.raw} pack contains [{contents}]{legend}")
    return True


def _open_tag_pack(state: RunState, preq: PackReq) -> list[ShopCard]:
    """Open the free pack a skip tag grants. The engine has no tag-consumption
    logic, so inject a zero-cost booster and open it through the normal path,
    which rolls contents on the same RNG keys the real game uses."""
    booster = ShopCard(center_key=preq.keys[0], card_type="Booster", cost=0, base_cost=0)
    state.shop.boosters.insert(0, booster)
    pack = open_booster_pack(state, 0)
    cards = list(pack.cards)
    state.pack = None
    return cards


def _check_shop(state: RunState, req: ShopReq, notes: list[str], where: str) -> bool:
    """Check a populated shop: required boosters first (opening any with a
    'contains' list), then the within_N items. Fails as soon as a missing item
    is past its roll budget; stops rerolling once everything is found, since
    extra rerolls would change the seed's later RNG."""
    for preq in req.packs:
        index = next(
            (i for i, booster in enumerate(state.shop.boosters) if booster.center_key in preq.keys),
            None,
        )
        if index is None:
            return False
        if preq.contains:
            pack = open_booster_pack(state, index)
            cards = list(pack.cards)
            state.pack = None
            if not _check_pack_contents(state, cards, preq, notes, where):
                return False
        else:
            notes.append(f"{where}: shop offers a {preq.raw} pack")

    pending: list[tuple[Item, int]] = [(item, rolls) for rolls, items in sorted(req.within.items()) for item in items]
    roll = 1
    max_roll = max((rolls for _, rolls in pending), default=0)
    while pending:
        keys = _card_keys(state.shop.cards)
        still: list[tuple[Item, int]] = []
        for item, rolls in pending:
            if item.key in keys:
                notes.append(f"{where}: {item.key} in shop on roll {roll}/{rolls}")
            else:
                still.append((item, rolls))
        pending = still
        if any(rolls <= roll for _, rolls in pending):
            return False
        if not pending or roll >= max_roll:
            break
        reroll_shop(state)
        roll += 1
    return not pending


def check_seed(
    spec: SearchSpec,
    seed: str,
    data: GameData | None = None,
    *,
    _data_prepared: bool = False,
) -> SeedMatch | None:
    """Simulate a canonical run of `seed`; return a match report or None.

    Blinds are "beaten" without playing hands (hand RNG lives on separate
    pseudoseed keys, so shop/voucher/tag/pack streams are unaffected). Vouchers
    are checked right where they're generated (run start, post-boss cash_out)
    because buying one clears state.current_voucher. `_data_prepared` skips
    the per-seed unlock_data deepcopy when search_seeds already did it.
    """
    data = data or load_game_data()
    if spec.unlock_all and not _data_prepared:
        data = unlock_data(data)
    state = create_run_state(seed, stake=spec.stake, deck_key=spec.deck_key, data=data)
    state.dollars += 10_000  # money never gates the search; buys/rerolls are assumed affordable
    match = SeedMatch(seed=seed)
    pending_vouchers = {ante: areq.voucher for ante, areq in spec.antes.items() if areq.voucher}

    def visit_shop(breq: BlindReq | None, where: str) -> bool:
        populate_shop(state)
        wanted = pending_vouchers.get(state.round_resets.ante)
        if wanted and any(v.center_key == wanted for v in state.shop.vouchers):
            redeem_voucher(state, wanted)
            match.notes.append(f"{where}: bought voucher {wanted}")
            del pending_vouchers[state.round_resets.ante]
        if breq and breq.shop:
            return _check_shop(state, breq.shop, match.notes, where)
        return True

    def voucher_ok(ante: int) -> bool:
        # Called right after the ante's voucher is generated (run start /
        # post-boss cash-out), before any shop can buy and clear it.
        areq = spec.antes.get(ante)
        return not areq or not areq.voucher or state.current_voucher == areq.voucher

    if not voucher_ok(1):
        return None

    for ante in range(1, spec.max_ante + 1):
        areq = spec.antes.get(ante, AnteReq())
        if areq.boss and state.round_resets.blind_choices["Boss"] != areq.boss:
            return None
        if areq.boss:
            match.notes.append(f"ante{ante}: boss is {areq.boss}")

        for blind in ("Small", "Big", "Boss"):
            breq = areq.blinds.get(blind)
            where = f"ante{ante}.{blind.lower()}"
            if breq and breq.skip:
                tag_key = state.round_resets.blind_tags.get(blind)
                skip_blind(state)
                if breq.skip.tag_key and tag_key != breq.skip.tag_key:
                    return None
                if breq.skip.tag_key:
                    match.notes.append(f"{where}: skip gives {tag_key}")
                if breq.skip.pack and not _check_pack_contents(
                    state, _open_tag_pack(state, breq.skip.pack), breq.skip.pack, match.notes, where
                ):
                    return None
                continue
            # Phantom-play the blind: select it, mark it beaten, cash out, shop.
            select_blind(state, blind)
            state.round_resets.blind_states[blind] = "Defeated"
            order = ("Small", "Big", "Boss")
            if blind != "Boss":
                next_blind = order[order.index(blind) + 1]
                state.round_resets.blind_states[next_blind] = "Select"
                state.blind_on_deck = next_blind
            cash_out(state)
            if blind == "Boss" and not voucher_ok(state.round_resets.ante):
                return None
            if not visit_shop(breq, where):
                return None

    return match


# ---------------------------------------------------------------------------
# Search loop


def random_seed(rng: random.Random) -> str:
    return "".join(rng.choice(SEED_ALPHABET) for _ in range(SEED_LENGTH))


def search_seeds(
    spec: SearchSpec,
    *,
    max_seeds: int = 100_000,
    matches: int = 1,
    rng: random.Random | None = None,
    data: GameData | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[SeedMatch]:
    """Try random seeds until `matches` seeds satisfy the spec or `max_seeds` is reached."""
    data = data or load_game_data()
    if spec.unlock_all:
        data = unlock_data(data)
    rng = rng or random.Random()
    found: list[SeedMatch] = []
    for attempt in range(1, max_seeds + 1):
        result = check_seed(spec, random_seed(rng), data, _data_prepared=True)
        if result is not None:
            found.append(result)
            if len(found) >= matches:
                break
        if progress and attempt % 500 == 0:
            progress(attempt, len(found))
    return found
