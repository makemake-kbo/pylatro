"""Interactive seed-walk: step through a seed's shops and ante/blind selection
without playing hands, with free rerolls, joker-depth lookups, a per-ante
voucher preview, and a pinnable seed-search-notation report.

Like the seed searcher (:mod:`pylatro.seedsearch`), blinds are "beaten" without
playing a hand -- hand RNG lives on separate pseudoseed keys, so the shop /
voucher / tag / pack streams are identical to a real run that beats every blind.
Buying or selling goes through the real engine, so an acquired joker is marked
used and stops reappearing in later rolls (except with Showman), exactly as in a
real run.

Tag *effects* on shops (e.g. an Uncommon tag forcing the next shop's joker
rarity) are not modeled -- the underlying engine has no tag-consumption logic --
so a skipped blind records its tag but does not alter the following shop. This
matches the seed searcher's fidelity.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field

from .blind import cash_out, reroll_boss, select_blind, skip_blind
from .data import GameData, load_game_data
from .models import RunState, ShopCard
from .run import create_run_state
from .seedsearch import TAG_PACKS, Resolver, unlock_data
from .shop import (
    buy_shop_card,
    claim_pack_card,
    close_pack,
    finish_shop,
    open_booster_pack,
    populate_shop,
    redeem_voucher,
    reroll_shop,
    sell_owned_consumable,
    sell_owned_joker,
)

__all__ = ["Pin", "SeedWalk", "WalkReport"]

_ORDER = ("Small", "Big", "Boss")
# Money never gates a seed walk: rerolls and buys are always affordable. The
# floor is topped up after every action so the balance can never run dry.
_FUNDS_FLOOR = 10**12
_DEFAULT_DEPTH_CAP = 2000


def _ensure_funds(state: RunState) -> None:
    if state.dollars < _FUNDS_FLOOR:
        state.dollars = _FUNDS_FLOOR


def _shop_has(state: RunState, key: str) -> bool:
    return any(card.center_key == key or card.front_key == key for card in state.shop.cards)


# ---------------------------------------------------------------------------
# Report


@dataclass
class Pin:
    """One recorded observation, later rendered into seed-search notation.

    ``blind`` is "" for ante-level pins (voucher / boss). ``roll`` is the shop
    roll the item was seen on (1 = shop as first opened, 2 = after one reroll),
    used to emit ``contains`` / ``within_N``. ``meta`` carries extra bits such as
    a skip tag key or a pack name.
    """

    ante: int
    kind: str  # "shop" | "voucher" | "boss" | "pack" | "skip"
    blind: str = ""
    roll: int = 1
    items: list[str] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)

    def describe(self, data: GameData) -> str:
        def name(key: str) -> str:
            return data.centers.get(key, {}).get("name") or data.tags.get(key, {}).get("name") or key

        loc = f"ante{self.ante}" + (f".{self.blind}" if self.blind else "")
        items = ", ".join(name(k) for k in self.items)
        if self.kind == "voucher":
            return f"{loc}: voucher {name(self.items[0])}"
        if self.kind == "boss":
            return f"{loc}: boss {name(self.items[0])}"
        if self.kind == "skip":
            tag = name(self.meta["tag"]) if self.meta.get("tag") else "skip"
            extra = f" ({self.meta.get('pack')}: {items})" if self.meta.get("pack") else ""
            return f"{loc}: skip -> {tag}{extra}"
        if self.kind == "pack":
            return f"{loc}: {self.meta.get('pack', 'pack')} contains [{items}]"
        return f"{loc}: [{items}] within {self.roll} roll(s)"


class WalkReport:
    """Accumulates pinned observations and renders them as a seed-search spec."""

    def __init__(self, seed: str, deck_key: str, stake: int) -> None:
        self.seed = seed
        self.deck_key = deck_key
        self.stake = stake
        self.pins: list[Pin] = []

    def add(self, pin: Pin) -> None:
        self.pins.append(pin)

    def remove(self, index: int) -> None:
        if 0 <= index < len(self.pins):
            del self.pins[index]

    def to_spec(self) -> dict:
        """Build a JSON spec (seed-search notation) from the pinned observations."""
        spec: dict = {}
        if self.deck_key != "b_red":
            spec["deck"] = self.deck_key
        if self.stake != 1:
            spec["stake"] = self.stake

        def ante(pin: Pin) -> dict:
            return spec.setdefault(f"ante{pin.ante}", {})

        def shop(pin: Pin) -> dict:
            return ante(pin).setdefault(pin.blind, {}).setdefault("shop", {})

        for pin in sorted(self.pins, key=lambda p: (p.ante, _ORDER_INDEX.get(p.blind, -1))):
            if pin.kind == "voucher":
                ante(pin)["voucher"] = pin.items[0]
            elif pin.kind == "boss":
                ante(pin)["boss"] = pin.items[0]
            elif pin.kind == "shop":
                block = shop(pin)
                key = "contains" if pin.roll <= 1 else f"within_{pin.roll}"
                bucket = block.setdefault(key, [])
                bucket.extend(item for item in pin.items if item not in bucket)
            elif pin.kind == "pack":
                packs = shop(pin).setdefault("packs", [])
                packs.append({"pack": pin.meta["pack"], "contains": pin.items} if pin.items else pin.meta["pack"])
            elif pin.kind == "skip":
                skip = ante(pin).setdefault(pin.blind, {}).setdefault("skip", {})
                if pin.meta.get("tag"):
                    skip["tag"] = pin.meta["tag"]
                if pin.meta.get("pack"):
                    skip["pack"] = pin.meta["pack"]
                    if pin.items:
                        skip["contains"] = list(pin.items)
        return spec

    def to_text(self) -> str:
        """A shareable report: a header comment (stripped by the spec loader)
        plus the JSON spec, ready for ``pylatro seed-search <file> --check SEED``."""
        deck = self.deck_key.removeprefix("b_")
        header = [
            f"// seed-walk report for seed {self.seed} ({deck} deck, stake {self.stake})",
            f"// verify: pylatro seed-search THIS_FILE.json --check {self.seed}",
        ]
        return "\n".join(header) + "\n" + json.dumps(self.to_spec(), indent=2)


_ORDER_INDEX = {"": -1, "small": 0, "big": 1, "boss": 2}


# ---------------------------------------------------------------------------
# Walk


class SeedWalk:
    """Drives a phantom run of one seed for interactive shop/blind exploration.

    The walk is always either at blind selection (``in_shop`` False) or in a
    shop (``in_shop`` True). Beating the on-deck blind opens its shop; leaving
    the shop advances to the next blind (or the next ante after the boss).
    """

    def __init__(
        self,
        seed: str,
        stake: int = 1,
        deck_key: str = "b_red",
        data: GameData | None = None,
        *,
        unlock_all: bool = True,
    ) -> None:
        data = data or load_game_data()
        if unlock_all:
            data = unlock_data(data)
        self.data = data
        self.resolver = Resolver(data)
        self.seed = seed.upper()
        self.deck_key = deck_key
        self.stake = stake
        self.state = create_run_state(self.seed, stake=stake, deck_key=deck_key, data=data)
        self.report = WalkReport(self.seed, deck_key, stake)
        self.in_shop = False
        self.roll = 0  # rerolls done in the current shop (0 = shop as first opened)
        # The blind whose shop we are in, and the ante it belonged to (the boss
        # shop of ante N opens after ante N's boss but offers ante N+1's voucher,
        # so its cards and its voucher live under different antes -- see the seed
        # searcher's conventions).
        self.shop_blind = ""
        self.shop_ante = 0
        _ensure_funds(self.state)

    # -- position ---------------------------------------------------------

    @property
    def ante(self) -> int:
        return self.state.round_resets.ante

    @property
    def on_deck(self) -> str:
        return self.state.blind_on_deck or "Small"

    @property
    def boss_key(self) -> str:
        return self.state.round_resets.blind_choices.get("Boss", "")

    @property
    def current_voucher(self) -> str | None:
        return self.state.current_voucher

    def blind_key(self, blind: str) -> str:
        return self.state.round_resets.blind_choices.get(blind, "")

    def blind_tag(self, blind: str) -> str:
        return self.state.round_resets.blind_tags.get(blind, "")

    def blind_state(self, blind: str) -> str:
        return self.state.round_resets.blind_states.get(blind, "Upcoming")

    # -- blind actions ----------------------------------------------------

    @staticmethod
    def _phantom_beat(state: RunState, blind: str) -> None:
        """Beat a blind without playing a hand, then cash out (mirrors the seed
        searcher). Hand RNG is untouched, so shop/voucher/tag/pack streams match
        a real run that beats the blind."""
        select_blind(state, blind)
        state.round_resets.blind_states[blind] = "Defeated"
        if blind != "Boss":
            next_blind = _ORDER[_ORDER.index(blind) + 1]
            state.round_resets.blind_states[next_blind] = "Select"
            state.blind_on_deck = next_blind
        cash_out(state)

    def beat_blind(self) -> None:
        """Phantom-beat the on-deck blind and open its shop."""
        assert not self.in_shop
        beaten = self.on_deck
        beaten_ante = self.state.round_resets.ante
        self._phantom_beat(self.state, beaten)
        populate_shop(self.state)
        _ensure_funds(self.state)
        self.in_shop = True
        self.roll = 0
        self.shop_blind = beaten.lower()
        self.shop_ante = beaten_ante

    def skip_blind(self) -> str:
        """Skip the on-deck (Small or Big) blind; return the tag it grants."""
        assert not self.in_shop
        blind = self.on_deck
        if blind == "Boss":
            raise ValueError("the boss blind cannot be skipped")
        tag = self.blind_tag(blind)
        skip_blind(self.state)
        return tag

    def reroll_boss(self) -> str:
        assert not self.in_shop
        boss = reroll_boss(self.state)
        _ensure_funds(self.state)
        return boss

    @staticmethod
    def tag_pack_label(tag_key: str) -> str | None:
        """Seed-search pack name for a pack-granting skip tag (e.g. 'mega arcana'),
        or None if the tag grants no booster."""
        info = TAG_PACKS.get(tag_key)
        if info is None:
            return None
        kind, size = info
        return f"{size} {kind.lower()}"

    def open_skip_pack(self, blind: str):
        """Open the free booster a skip tag grants for ``blind`` (Charm, Meteor,
        Standard, Buffoon, Ethereal) and leave it open as ``state.pack``. Rolls
        contents on the same RNG keys the real game uses. Raises ValueError for a
        non-pack tag."""
        tag = self.blind_tag(blind)
        info = TAG_PACKS.get(tag)
        if info is None:
            raise ValueError(f"tag {tag} does not grant a booster pack")
        kind, size = info
        booster_keys = [proto["key"] for proto in self.data.center_pools["Booster"]]
        key = next(k for k in booster_keys if f"_{kind.lower()}_{size}_" in k)
        booster = ShopCard(center_key=key, card_type="Booster", cost=0, base_cost=0)
        self.state.shop.boosters.insert(0, booster)
        pack = open_booster_pack(self.state, 0)
        _ensure_funds(self.state)
        return pack

    # -- shop actions -----------------------------------------------------

    def reroll_shop(self) -> list[ShopCard]:
        assert self.in_shop
        cards = reroll_shop(self.state)
        _ensure_funds(self.state)
        self.roll += 1
        return cards

    def buy_card(self, index: int) -> ShopCard:
        assert self.in_shop
        card = buy_shop_card(self.state, index)
        _ensure_funds(self.state)
        return card

    def buy_voucher(self, voucher_key: str) -> None:
        assert self.in_shop
        redeem_voucher(self.state, voucher_key)
        _ensure_funds(self.state)

    def sell_joker(self, index: int) -> None:
        sell_owned_joker(self.state, index)
        _ensure_funds(self.state)

    def sell_consumable(self, index: int) -> None:
        sell_owned_consumable(self.state, index)
        _ensure_funds(self.state)

    def open_pack(self, index: int):
        assert self.in_shop
        pack = open_booster_pack(self.state, index)
        _ensure_funds(self.state)
        return pack

    def claim_pack_card(self, index: int) -> ShopCard:
        card = claim_pack_card(self.state, index)
        _ensure_funds(self.state)
        return card

    def close_pack(self, *, skipped: bool = True) -> None:
        close_pack(self.state, skipped=skipped)

    def leave_shop(self) -> None:
        assert self.in_shop
        finish_shop(self.state)
        self.in_shop = False
        self.roll = 0
        self.shop_blind = ""

    # -- analysis (non-destructive) ---------------------------------------

    def rolls_until(self, name: str, cap: int = _DEFAULT_DEPTH_CAP) -> tuple[str, int | None]:
        """From the current shop, how many rerolls until ``name`` appears.

        Returns ``(center_key, rerolls)`` where 0 means "already in this shop"
        and ``None`` means "not seen within ``cap`` rerolls". Runs on a deep copy,
        so the real walk's RNG is untouched. Raises :class:`SpecError` for an
        unknown / ambiguous name.
        """
        assert self.in_shop
        key = self.resolver.item(name).key
        probe = deepcopy(self.state)
        for rerolls in range(cap + 1):
            if _shop_has(probe, key):
                return key, rerolls
            reroll_shop(probe)
            _ensure_funds(probe)
        return key, None

    def voucher_schedule(self, horizon: int = 8) -> list[tuple[int, str | None]]:
        """Voucher offered per ante from the current point forward, assuming no
        further purchases. Simulated on a deep copy. A ``None`` voucher means the
        current ante's voucher has already been redeemed in the real walk."""
        probe = deepcopy(self.state)
        schedule: list[tuple[int, str | None]] = []
        start = probe.round_resets.ante
        while probe.round_resets.ante <= horizon and probe.round_resets.ante < start + 64:
            schedule.append((probe.round_resets.ante, probe.current_voucher))
            self._phantom_advance_ante(probe)
        return schedule

    @classmethod
    def _phantom_advance_ante(cls, state: RunState) -> None:
        """Phantom-beat blinds until the ante increments (past the boss)."""
        start = state.round_resets.ante
        guard = 0
        while state.round_resets.ante == start and guard < len(_ORDER) + 1:
            cls._phantom_beat(state, state.blind_on_deck or "Small")
            guard += 1

    # -- pin helpers (used by the UI) -------------------------------------

    def pin_shop_card(self, item_key: str) -> Pin:
        pin = Pin(
            ante=self.shop_ante,
            kind="shop",
            blind=self.shop_blind,
            roll=self.roll + 1,
            items=[item_key],
        )
        self.report.add(pin)
        return pin

    def pin_current_voucher(self) -> Pin | None:
        if not self.current_voucher:
            return None
        pin = Pin(ante=self.ante, kind="voucher", items=[self.current_voucher])
        self.report.add(pin)
        return pin

    def pin_voucher(self, ante: int, voucher_key: str) -> Pin:
        pin = Pin(ante=ante, kind="voucher", items=[voucher_key])
        self.report.add(pin)
        return pin

    def pin_boss(self) -> Pin:
        pin = Pin(ante=self.ante, kind="boss", items=[self.boss_key])
        self.report.add(pin)
        return pin

    def pin_skip(self, blind: str, tag_key: str, pack: str | None = None, contains: list[str] | None = None) -> Pin:
        meta: dict[str, str] = {"tag": tag_key}
        if pack:
            meta["pack"] = pack
        pin = Pin(ante=self.ante, kind="skip", blind=blind.lower(), items=contains or [], meta=meta)
        self.report.add(pin)
        return pin

    def pin_pack(self, pack_name: str, contains: list[str]) -> Pin:
        pin = Pin(
            ante=self.shop_ante,
            kind="pack",
            blind=self.shop_blind,
            items=list(contains),
            meta={"pack": pack_name},
        )
        self.report.add(pin)
        return pin
