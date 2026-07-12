"""Seed-walk mode: step through a seed's shops and blind/ante selection without
playing hands, with free rerolls, a joker-depth lookup, a per-ante voucher
preview, and a pinnable seed-search-notation report.

Backed by :class:`pylatro.seedwalk.SeedWalk`, stored on the app as ``app.walk``.
"""

from __future__ import annotations

import random
import string
from math import floor
from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Static

from pylatro import get_blind_amount
from pylatro.seedsearch import SpecError

from ..theme import BALATRO_PALETTE
from ..widgets.blind_panel import BlindPanel
from ..widgets.consumable_bar import ConsumableBar
from ..widgets.joker_bar import JokerBar

STAKE_NAMES = ["White", "Red", "Green", "Blue", "Purple", "Orange", "Gold", "Black"]
DECK_KEYS = [
    "b_red", "b_blue", "b_yellow", "b_green",
    "b_black", "b_magic", "b_nebula", "b_ghost",
    "b_abandoned", "b_checkered", "b_zodiac", "b_painted",
    "b_anaglyph", "b_plasma", "b_erratic",
]

TYPE_COLORS = {
    "Joker": "#ffc107",
    "Tarot": "#9c27b0",
    "Planet": "#2196f3",
    "Spectral": "#b0bec5",
    "Voucher": "#26c6da",
    "Booster": "#ff8a65",
    "Base": "#ecf0f1",
    "Enhanced": "#27ae60",
}


def _center_name(walk, key: str) -> str:
    center = walk.data.centers.get(key, {})
    return center.get("name") or walk.data.tags.get(key, {}).get("name") or key


class NavButton(Button):
    """A clickable button that never takes keyboard focus, so the screen's
    single-key bindings (Enter=beat/claim, r, p, n, ...) are never swallowed by
    a focused button. Mouse clicks still work via on_button_pressed."""

    can_focus = False


class _WalkScreen(Screen):
    """Base for walk screens: no auto-focus, so single-key bindings (Enter, r,
    p, n, ...) act on the screen rather than being swallowed by a focused Button.
    Focus is taken explicitly (Tab to the joker bar, 'f' for the find box)."""

    AUTO_FOCUS = None


# ---------------------------------------------------------------------------
# Setup modal


class SeedWalkSetupModal(Screen):
    """Start a seed walk: enter a seed, pick a deck and stake."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    SeedWalkSetupModal { align: center middle; background: rgba(0, 0, 0, 0.7); }
    #sw-setup { width: 52; height: auto; background: #1c1c3a; border: heavy #5c5c8a; padding: 2; }
    #sw-setup Label { margin: 1 0 0 0; }
    #sw-setup Input { margin: 0 0 1 0; }
    #sw-setup Button { margin: 1 1 0 0; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._deck_idx = 0
        self._stake = 1

    def compose(self) -> ComposeResult:
        with Vertical(id="sw-setup"):
            yield Label("Seed Walk", classes="panel-title")
            yield Label("Seed (blank = random):")
            yield Input(placeholder="8-char seed", id="sw-seed", max_length=8)
            yield Label("Deck:")
            yield Button(self._deck_name(), id="sw-deck", variant="default")
            yield Label("Stake:")
            yield Button(f"{STAKE_NAMES[self._stake - 1]} Stake", id="sw-stake", variant="default")
            yield Button("Start Walk", id="sw-start", variant="primary")
            yield Button("Cancel", id="sw-cancel", variant="default")

    def _deck_name(self) -> str:
        return DECK_KEYS[self._deck_idx].replace("b_", "").title() + " Deck"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "sw-deck":
                self._deck_idx = (self._deck_idx + 1) % len(DECK_KEYS)
                event.button.label = self._deck_name()
            case "sw-stake":
                self._stake = (self._stake % 8) + 1
                event.button.label = f"{STAKE_NAMES[self._stake - 1]} Stake"
            case "sw-start":
                self._start()
            case "sw-cancel":
                self.app.pop_screen()

    def _start(self) -> None:
        seed = self.query_one("#sw-seed", Input).value.strip().upper()
        if not seed:
            seed = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
        from pylatro.seedwalk import SeedWalk

        app = self.app
        app.walk = SeedWalk(seed, stake=self._stake, deck_key=DECK_KEYS[self._deck_idx], data=app.game_data)
        app.pop_screen()  # this modal
        app.pop_screen()  # main menu
        app.push_screen(SeedWalkBlindScreen())

    def action_cancel(self) -> None:
        self.app.pop_screen()


# ---------------------------------------------------------------------------
# Blind selection


class SeedWalkBlindScreen(_WalkScreen):
    """Blind selection for a walk: beat (phantom), skip, reroll the boss, view
    the voucher schedule, and open the report."""

    BINDINGS = [
        Binding("enter", "beat", "Beat"),
        Binding("s", "skip", "Skip"),
        Binding("x", "reroll_boss", "Reroll Boss"),
        Binding("v", "vouchers", "Vouchers"),
        Binding("p", "pin_boss", "Pin Boss"),
        Binding("r", "report", "Report"),
        Binding("escape", "menu", "Menu"),
    ]

    DEFAULT_CSS = """
    SeedWalkBlindScreen { layout: vertical; }
    #sw-blind-header { height: 3; text-align: center; padding: 1; color: #ffc107; text-style: bold; }
    #sw-blind-panels { height: 1fr; layout: horizontal; }
    #sw-blind-buttons { height: 3; layout: horizontal; align: center middle; }
    #sw-blind-buttons Button { margin: 0 1; }
    """

    def _walk(self):
        return self.app.walk

    def compose(self) -> ComposeResult:
        walk = self._walk()
        yield Static("", id="sw-blind-header")
        with Horizontal(id="sw-blind-panels"):
            for blind_type in ("Small", "Big", "Boss"):
                yield self._make_panel(walk, blind_type)
        with Horizontal(id="sw-blind-buttons"):
            yield NavButton("Vouchers [v]", id="sw-vouchers-btn", variant="primary")
            yield NavButton("Report [r]", id="sw-report-btn", variant="default")

    def on_mount(self) -> None:
        # Leave nothing focused so the single-key bindings (Enter=beat, etc.)
        # fire on the screen instead of being swallowed by a focused Button.
        self.set_focus(None)
        self._refresh_header()

    def _refresh_header(self) -> None:
        walk = self._walk()
        self.query_one("#sw-blind-header", Static).update(
            f"Seed {walk.seed}  ·  Ante {walk.ante}  ·  on deck: {walk.on_deck}   "
            f"([Enter] beat, [s] skip, [x] reroll boss)"
        )

    def _make_panel(self, walk, blind_type: str) -> BlindPanel:
        state = walk.state
        resets = state.round_resets
        blind_key = resets.blind_choices.get(blind_type, "bl_small")
        blind_data = state.data.blinds.get(blind_key, {})
        ante = resets.ante
        scaling = min(state.stake, 3)
        chip_target = floor(get_blind_amount(ante, scaling) * blind_data.get("mult", 1))
        reward = {"Small": 3, "Big": 4, "Boss": 5}.get(blind_type, 3) + ante - 1
        tag_key = resets.blind_tags.get(blind_type, "")
        tag_name = state.data.tags.get(tag_key, {}).get("name", tag_key) if tag_key else ""
        boss_desc = ""
        if blind_type == "Boss":
            boss_desc = blind_data.get("debuff_text", blind_data.get("description", ""))
        return BlindPanel(
            blind_type=blind_type,
            blind_key=blind_key,
            blind_name=blind_data.get("name", blind_type),
            chip_target=chip_target,
            reward=reward,
            state=resets.blind_states.get(blind_type, "Upcoming"),
            tag_name=tag_name,
            boss_desc=boss_desc,
            is_focused=(blind_type == walk.on_deck),
            id=f"sw-blind-{blind_type.lower()}",
        )

    # -- actions --

    def action_beat(self) -> None:
        walk = self._walk()
        walk.beat_blind()
        self.app.switch_screen(SeedWalkShopScreen())

    def action_skip(self) -> None:
        walk = self._walk()
        blind = walk.on_deck
        if blind == "Boss":
            self.notify("Cannot skip the boss blind", severity="warning")
            return
        tag = walk.skip_blind()
        has_pack = walk.tag_pack_label(tag) is not None
        if has_pack:
            # A pack-granting tag opens its free pack immediately (as in-game).
            walk.open_skip_pack(blind)
        else:
            self.notify(f"Skipped {blind}, got {_center_name(walk, tag)}")
        # Swap in the refreshed blind screen first, then layer the pack modal on
        # top of it (pushing before the switch would just get replaced).
        self.app.switch_screen(SeedWalkBlindScreen())
        if has_pack:
            self.app.push_screen(SeedWalkPackScreen(pin_kind="skip", blind=blind, tag=tag))

    def action_reroll_boss(self) -> None:
        walk = self._walk()
        boss = walk.reroll_boss()
        self.notify(f"New boss: {_center_name(walk, boss)}")
        self.app.switch_screen(SeedWalkBlindScreen())

    def action_pin_boss(self) -> None:
        walk = self._walk()
        walk.pin_boss()
        self.notify(f"Pinned boss {_center_name(walk, walk.boss_key)} (ante {walk.ante})")

    def action_vouchers(self) -> None:
        self.app.push_screen(VoucherScheduleModal())

    def action_report(self) -> None:
        self.app.push_screen(ReportModal())

    def action_menu(self) -> None:
        from .main_menu import MainMenuScreen

        self.app.walk = None
        self.app.switch_screen(MainMenuScreen())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "sw-vouchers-btn":
                self.action_vouchers()
            case "sw-report-btn":
                self.action_report()


# ---------------------------------------------------------------------------
# Shop


class SeedWalkShopScreen(_WalkScreen):
    """Shop for a walk: free rerolls, buy/sell (blocks future rolls), pin items,
    look up how many rerolls until a joker appears, and open the report."""

    BINDINGS = [
        Binding("left", "cursor_prev", "Prev", show=False),
        Binding("right", "cursor_next", "Next", show=False),
        Binding("h", "cursor_prev", "Prev", show=False),
        Binding("l", "cursor_next", "Next", show=False),
        Binding("up", "cursor_prev", "Prev", show=False),
        Binding("down", "cursor_next", "Next", show=False),
        Binding("enter", "buy", "Buy"),
        Binding("r", "reroll", "Reroll"),
        Binding("p", "pin", "Pin"),
        Binding("f", "focus_find", "Find joker"),
        Binding("n", "next_blind", "Next blind"),
        Binding("R", "report", "Report"),
    ]

    DEFAULT_CSS = """
    SeedWalkShopScreen { layout: vertical; }
    #sw-shop-header { height: 3; text-align: center; padding: 1; color: #ffc107; text-style: bold; }
    #sw-shop-cards { height: 1fr; layout: vertical; padding: 0 1; }
    .sw-zone-label { height: 1; color: #95a5a6; margin: 1 0 0 0; }
    #sw-find-row { height: 3; layout: horizontal; }
    #sw-find-row Label { width: auto; padding: 1 1 0 1; }
    #sw-find { width: 1fr; }
    #sw-shop-buttons { height: 3; layout: horizontal; align: center middle; }
    #sw-shop-buttons Button { margin: 0 1; }
    """

    cursor: reactive[int] = reactive(0)

    def _walk(self):
        return self.app.walk

    def compose(self) -> ComposeResult:
        yield Static("", id="sw-shop-header")
        yield JokerBar(id="sw-joker-bar")
        with Vertical(id="sw-shop-cards"):
            yield Static("Cards", classes="sw-zone-label")
            yield Static("", id="sw-cards-display")
            yield Static("Boosters", classes="sw-zone-label")
            yield Static("", id="sw-boosters-display")
            yield Static("Vouchers", classes="sw-zone-label")
            yield Static("", id="sw-vouchers-display")
        yield ConsumableBar(id="sw-consumable-bar")
        with Horizontal(id="sw-find-row"):
            yield Label("Find joker rolls:")
            yield Input(placeholder="joker / card name, Enter to search", id="sw-find")
        with Horizontal(id="sw-shop-buttons"):
            yield NavButton("Reroll [r]", id="sw-reroll-btn", variant="warning")
            yield NavButton("Pin [p]", id="sw-pin-btn", variant="success")
            yield NavButton("Report [R]", id="sw-report-btn", variant="default")
            yield NavButton("Next Blind [n]", id="sw-next-btn", variant="primary")

    def on_mount(self) -> None:
        # Keep keys on the screen bindings by default; Tab reaches the joker /
        # consumable bars (to sell) and 'f' focuses the find box.
        self.set_focus(None)
        self._refresh()

    # -- flat cursor over cards, boosters, vouchers --

    def _zones(self):
        s = self._walk().state.shop
        return [s.cards, s.boosters, s.vouchers]

    def _all_items(self):
        cards, boosters, vouchers = self._zones()
        return list(cards) + list(boosters) + list(vouchers)

    def _zone_and_local(self):
        cards, boosters, _ = self._zones()
        pos = self.cursor
        if pos < len(cards):
            return 0, pos
        pos -= len(cards)
        if pos < len(boosters):
            return 1, pos
        return 2, pos - len(boosters)

    def _refresh(self) -> None:
        walk = self._walk()
        state = walk.state
        self.query_one("#sw-shop-header", Static).update(
            f"Seed {walk.seed}  ·  Ante {walk.shop_ante}  ·  {walk.shop_blind} shop  ·  "
            f"roll {walk.roll + 1}  ·  $∞"
        )
        cur_zone, local = self._zone_and_local()
        cards, boosters, vouchers = self._zones()
        self.query_one("#sw-cards-display", Static).update(self._render_zone(walk, cards, 0, cur_zone, local))
        self.query_one("#sw-boosters-display", Static).update(self._render_zone(walk, boosters, 1, cur_zone, local))
        self.query_one("#sw-vouchers-display", Static).update(self._render_zone(walk, vouchers, 2, cur_zone, local))
        self.query_one("#sw-reroll-btn", Button).label = f"Reroll [r] (roll {walk.roll + 2})"
        self.query_one("#sw-joker-bar", JokerBar).update_jokers(state.jokers, state.data)
        self.query_one("#sw-consumable-bar", ConsumableBar).update_consumables(state.consumables, state.data)

    def _render_zone(self, walk, items, zone_id, cur_zone, local) -> Text:
        if not items:
            return Text("  (empty)", style=BALATRO_PALETTE["text_muted"])
        t = Text()
        for i, item in enumerate(items):
            focused = (zone_id == cur_zone) and (i == local)
            t.append(" >> " if focused else "    ", style=BALATRO_PALETTE["card_selected"])
            tc = TYPE_COLORS.get(item.card_type, "#95a5a6")
            t.append(f"[{item.card_type}] ", style=tc)
            t.append(_center_name(walk, item.center_key), style="bold #ecf0f1")
            if item.front_key:
                t.append(f" ({item.front_key})", style="#95a5a6")
            t.append("\n")
        return t

    def watch_cursor(self, value: int) -> None:
        if self.is_mounted:
            self._refresh()

    def action_cursor_prev(self) -> None:
        if self.cursor > 0:
            self.cursor -= 1

    def action_cursor_next(self) -> None:
        if self.cursor < len(self._all_items()) - 1:
            self.cursor += 1

    def _focused_item(self):
        items = self._all_items()
        if not items or self.cursor >= len(items):
            return None, None, None
        zone, local = self._zone_and_local()
        return items[self.cursor], zone, local

    # -- actions --

    def action_buy(self) -> None:
        walk = self._walk()
        item, zone, local = self._focused_item()
        if item is None:
            return
        try:
            if zone == 2:
                walk.buy_voucher(item.center_key)
                self.notify(f"Redeemed {_center_name(walk, item.center_key)}")
            elif zone == 1:
                walk.open_pack(local)
                self.app.push_screen(SeedWalkPackScreen(pin_kind="shop", pack_key=item.center_key))
                return
            else:
                walk.buy_card(local)
                self.notify(f"Bought {_center_name(walk, item.center_key)} (blocked from future rolls)")
        except Exception as e:
            self.notify(f"Cannot buy: {e}", severity="error")
            return
        if self.cursor >= len(self._all_items()):
            self.cursor = max(0, len(self._all_items()) - 1)
        self._refresh()

    def action_reroll(self) -> None:
        self._walk().reroll_shop()
        self._refresh()

    def action_pin(self) -> None:
        walk = self._walk()
        item, zone, _ = self._focused_item()
        if item is None:
            return
        if zone == 2:
            walk.pin_voucher(walk.ante, item.center_key)
            self.notify(f"Pinned voucher {_center_name(walk, item.center_key)} (ante {walk.ante})")
        elif zone == 1:
            walk.pin_pack(item.center_key, [])
            self.notify(f"Pinned pack {_center_name(walk, item.center_key)} offered")
        else:
            walk.pin_shop_card(item.center_key)
            self.notify(f"Pinned {_center_name(walk, item.center_key)} (within {walk.roll + 1} roll(s))")

    def action_focus_find(self) -> None:
        self.query_one("#sw-find", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        walk = self._walk()
        query = event.value.strip()
        self.set_focus(None)
        if not query:
            return
        try:
            key, rerolls = walk.rolls_until(query)
        except SpecError as e:
            self.notify(str(e), severity="warning")
            return
        name = _center_name(walk, key)
        if rerolls is None:
            self.notify(f"{name}: not found within {2000} rerolls from here", severity="warning")
        elif rerolls == 0:
            self.notify(f"{name}: already in this shop")
        else:
            self.notify(f"{name}: {rerolls} reroll(s) away (appears on roll {walk.roll + 1 + rerolls})")

    def action_next_blind(self) -> None:
        self._walk().leave_shop()
        self.app.switch_screen(SeedWalkBlindScreen())

    def action_report(self) -> None:
        self.app.push_screen(ReportModal())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "sw-reroll-btn":
                self.action_reroll()
            case "sw-pin-btn":
                self.action_pin()
            case "sw-report-btn":
                self.action_report()
            case "sw-next-btn":
                self.action_next_blind()

    def on_joker_bar_sell_requested(self, event: JokerBar.SellRequested) -> None:
        self._walk().sell_joker(event.index)
        self._refresh()

    def on_consumable_bar_sell_requested(self, event: ConsumableBar.SellRequested) -> None:
        self._walk().sell_consumable(event.index)
        self._refresh()


# ---------------------------------------------------------------------------
# Pack view


class SeedWalkPackScreen(_WalkScreen):
    """View an opened pack (shop or skip): claim cards, or pin its contents."""

    BINDINGS = [
        Binding("left", "cursor_prev", "Prev", show=False),
        Binding("right", "cursor_next", "Next", show=False),
        Binding("up", "cursor_prev", "Prev", show=False),
        Binding("down", "cursor_next", "Next", show=False),
        Binding("h", "cursor_prev", "Prev", show=False),
        Binding("l", "cursor_next", "Next", show=False),
        Binding("enter", "claim", "Claim"),
        Binding("p", "pin", "Pin contents"),
        Binding("escape", "close", "Close"),
    ]

    DEFAULT_CSS = """
    SeedWalkPackScreen { align: center middle; background: rgba(0, 0, 0, 0.7); }
    #sw-pack { width: 62; height: auto; max-height: 80%; background: #1c1c3a; border: heavy #5c5c8a; padding: 2; }
    #sw-pack-header { text-align: center; margin-bottom: 1; }
    #sw-pack-cards { height: auto; min-height: 4; }
    #sw-pack-buttons { layout: horizontal; height: 3; align: center middle; margin-top: 1; }
    #sw-pack-buttons Button { margin: 0 1; }
    """

    cursor: reactive[int] = reactive(0)

    def __init__(self, pin_kind: str, pack_key: str = "", blind: str = "", tag: str = "") -> None:
        super().__init__()
        self._pin_kind = pin_kind  # "shop" | "skip"
        self._pack_key = pack_key
        self._blind = blind
        self._tag = tag

    def _walk(self):
        return self.app.walk

    def compose(self) -> ComposeResult:
        with Vertical(id="sw-pack"):
            yield Static("", id="sw-pack-header")
            yield Static("", id="sw-pack-cards")
            with Horizontal(id="sw-pack-buttons"):
                yield NavButton("Claim [Enter]", id="sw-pack-claim", variant="primary")
                yield NavButton("Pin contents [p]", id="sw-pack-pin", variant="success")
                yield NavButton("Close [Esc]", id="sw-pack-close", variant="default")

    def on_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        walk = self._walk()
        pack = walk.state.pack
        if pack is None:
            self.app.pop_screen()
            return
        name = _center_name(walk, pack.booster_key)
        self.query_one("#sw-pack-header", Static).update(f"{name}  ·  {pack.choices_remaining} choice(s) left")
        t = Text()
        for i, card in enumerate(pack.cards):
            t.append(" >> " if i == self.cursor else "    ", style=BALATRO_PALETTE["card_selected"])
            tc = TYPE_COLORS.get(card.card_type, "#95a5a6")
            t.append(f"[{card.card_type}] ", style=tc)
            t.append(_center_name(walk, card.center_key), style="bold #ecf0f1")
            if card.front_key:
                t.append(f" ({card.front_key})", style="#95a5a6")
            t.append("\n")
        self.query_one("#sw-pack-cards", Static).update(t)

    def watch_cursor(self, value: int) -> None:
        if self.is_mounted:
            self._refresh()

    def _cards(self):
        pack = self._walk().state.pack
        return pack.cards if pack else []

    def action_cursor_prev(self) -> None:
        if self.cursor > 0:
            self.cursor -= 1

    def action_cursor_next(self) -> None:
        if self.cursor < len(self._cards()) - 1:
            self.cursor += 1

    def action_claim(self) -> None:
        walk = self._walk()
        cards = self._cards()
        if not cards:
            return
        idx = min(self.cursor, len(cards) - 1)
        try:
            claimed = walk.claim_pack_card(idx)
            self.notify(f"Claimed {_center_name(walk, claimed.center_key)}")
        except Exception as e:
            self.notify(f"Cannot claim: {e}", severity="error")
            return
        if walk.state.pack is None:
            self.app.pop_screen()
        else:
            self.cursor = min(self.cursor, max(0, len(self._cards()) - 1))
            self._refresh()

    def action_pin(self) -> None:
        walk = self._walk()
        pack = walk.state.pack
        if pack is None:
            return
        contents = [c.center_key for c in pack.cards]
        if self._pin_kind == "skip":
            walk.pin_skip(self._blind, self._tag, pack=walk.tag_pack_label(self._tag), contains=contents)
        else:
            walk.pin_pack(self._pack_key, contents)
        self.notify(f"Pinned pack contents ({len(contents)} cards)")

    def action_close(self) -> None:
        self._walk().close_pack(skipped=True)
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "sw-pack-claim":
                self.action_claim()
            case "sw-pack-pin":
                self.action_pin()
            case "sw-pack-close":
                self.action_close()


# ---------------------------------------------------------------------------
# Voucher schedule modal


class VoucherScheduleModal(_WalkScreen):
    """Voucher offered per ante from here forward (assuming no further buys).
    Enter pins the focused ante's voucher into the report."""

    BINDINGS = [
        Binding("up", "cursor_prev", "Up", show=False),
        Binding("down", "cursor_next", "Down", show=False),
        Binding("k", "cursor_prev", "Up", show=False),
        Binding("j", "cursor_next", "Down", show=False),
        Binding("enter", "pin", "Pin voucher"),
        Binding("escape", "close", "Close"),
    ]

    DEFAULT_CSS = """
    VoucherScheduleModal { align: center middle; background: rgba(0, 0, 0, 0.7); }
    #sw-vouchers { width: 50; height: auto; max-height: 80%; background: #1c1c3a; border: heavy #5c5c8a; padding: 2; }
    #sw-vouchers-title { text-align: center; margin-bottom: 1; text-style: bold; }
    """

    cursor: reactive[int] = reactive(0)

    def _walk(self):
        return self.app.walk

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="sw-vouchers"):
            yield Static("Voucher schedule (assumes no purchases)", id="sw-vouchers-title")
            yield Static("", id="sw-vouchers-list")

    def on_mount(self) -> None:
        walk = self._walk()
        self._schedule = walk.voucher_schedule(horizon=walk.ante + 7)
        self._refresh()

    def _refresh(self) -> None:
        walk = self._walk()
        t = Text()
        for i, (ante, key) in enumerate(self._schedule):
            t.append(" >> " if i == self.cursor else "    ", style=BALATRO_PALETTE["card_selected"])
            t.append(f"Ante {ante}: ", style="#95a5a6")
            if key:
                t.append(_center_name(walk, key), style="bold #26c6da")
            else:
                t.append("(already redeemed)", style="#636e72")
            t.append("\n")
        self.query_one("#sw-vouchers-list", Static).update(t)

    def watch_cursor(self, value: int) -> None:
        if self.is_mounted:
            self._refresh()

    def action_cursor_prev(self) -> None:
        if self.cursor > 0:
            self.cursor -= 1

    def action_cursor_next(self) -> None:
        if self.cursor < len(self._schedule) - 1:
            self.cursor += 1

    def action_pin(self) -> None:
        walk = self._walk()
        ante, key = self._schedule[self.cursor]
        if not key:
            self.notify("No voucher to pin for that ante", severity="warning")
            return
        walk.pin_voucher(ante, key)
        self.notify(f"Pinned voucher {_center_name(walk, key)} (ante {ante})")

    def action_close(self) -> None:
        self.app.pop_screen()


# ---------------------------------------------------------------------------
# Report modal


class ReportModal(_WalkScreen):
    """Show pinned findings + the seed-search-notation report; delete pins and
    export to a JSON file."""

    BINDINGS = [
        Binding("up", "cursor_prev", "Up", show=False),
        Binding("down", "cursor_next", "Down", show=False),
        Binding("k", "cursor_prev", "Up", show=False),
        Binding("j", "cursor_next", "Down", show=False),
        Binding("d", "delete", "Delete pin"),
        Binding("e", "export", "Export"),
        Binding("escape", "close", "Close"),
    ]

    DEFAULT_CSS = """
    ReportModal { align: center middle; background: rgba(0, 0, 0, 0.7); }
    #sw-report { width: 74; height: auto; max-height: 90%; background: #1c1c3a; border: heavy #5c5c8a; padding: 2; }
    #sw-report-title { text-align: center; margin-bottom: 1; text-style: bold; }
    .sw-report-label { color: #95a5a6; margin: 1 0 0 0; }
    #sw-report-spec { color: #b0bec5; }
    """

    cursor: reactive[int] = reactive(0)

    def _walk(self):
        return self.app.walk

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="sw-report"):
            yield Static("Seed-walk report", id="sw-report-title")
            yield Static("Pins  ([d] delete, [e] export):", classes="sw-report-label")
            yield Static("", id="sw-report-pins")
            yield Static("Seed-search spec:", classes="sw-report-label")
            yield Static("", id="sw-report-spec")

    def on_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        walk = self._walk()
        pins = walk.report.pins
        t = Text()
        if not pins:
            t.append("  (nothing pinned yet)", style="#636e72")
        for i, pin in enumerate(pins):
            t.append(" >> " if i == self.cursor else "    ", style=BALATRO_PALETTE["card_selected"])
            t.append(pin.describe(walk.data) + "\n", style="#ecf0f1")
        self.query_one("#sw-report-pins", Static).update(t)
        self.query_one("#sw-report-spec", Static).update(walk.report.to_text())

    def watch_cursor(self, value: int) -> None:
        if self.is_mounted:
            self._refresh()

    def action_cursor_prev(self) -> None:
        if self.cursor > 0:
            self.cursor -= 1

    def action_cursor_next(self) -> None:
        if self.cursor < len(self._walk().report.pins) - 1:
            self.cursor += 1

    def action_delete(self) -> None:
        walk = self._walk()
        if not walk.report.pins:
            return
        walk.report.remove(self.cursor)
        self.cursor = max(0, min(self.cursor, len(walk.report.pins) - 1))
        self._refresh()

    def action_export(self) -> None:
        walk = self._walk()
        if not walk.report.pins:
            self.notify("Nothing pinned to export", severity="warning")
            return
        path = Path.cwd() / f"seedwalk_{walk.seed}.json"
        path.write_text(walk.report.to_text() + "\n")
        self.notify(f"Exported report to {path}")

    def action_close(self) -> None:
        self.app.pop_screen()
