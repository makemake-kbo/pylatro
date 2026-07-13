"""Textual pilot tests for the seed-walk screens (drive the real app)."""

from __future__ import annotations

from pylatro.seedwalk import SeedWalk
from pylatro_cli.app import BalatroApp
from pylatro_cli.screens.seedwalk import (
    ReportModal,
    SeedWalkBlindScreen,
    SeedWalkPackScreen,
    SeedWalkShopScreen,
    VoucherScheduleModal,
)

PHOTOCHAD_SEED = "5W7A7UGZ"
# Skipping this seed's ante-1 small blind grants a Charm tag (a Mega Arcana pack).
CHARM_SKIP_SEED = "FNFFVJ2S"


async def _boot_walk(pilot, app):
    """Start the app on a walk's blind-select screen."""
    await pilot.pause()
    app.walk = SeedWalk(PHOTOCHAD_SEED, data=app.game_data)
    await app.push_screen(SeedWalkBlindScreen())
    await pilot.pause()


async def test_enter_beats_blind_and_opens_shop():
    app = BalatroApp()
    async with app.run_test() as pilot:
        await _boot_walk(pilot, app)
        assert isinstance(app.screen, SeedWalkBlindScreen)
        await pilot.press("enter")  # beat small blind
        await pilot.pause()
        assert isinstance(app.screen, SeedWalkShopScreen)
        assert app.walk.in_shop and app.walk.shop_blind == "small"


async def test_voucher_modal_pins_future_voucher():
    app = BalatroApp()
    async with app.run_test() as pilot:
        await _boot_walk(pilot, app)
        await pilot.press("v")
        await pilot.pause()
        assert isinstance(app.screen, VoucherScheduleModal)
        await pilot.press("down")  # move to ante 2
        await pilot.press("enter")  # pin its voucher
        await pilot.press("escape")
        await pilot.pause()
        # Back on the blind screen, and a voucher pin was recorded for ante 2.
        assert isinstance(app.screen, SeedWalkBlindScreen)
        assert any(p.kind == "voucher" and p.ante == 2 for p in app.walk.report.pins)


async def test_shop_reroll_and_pin_records_depth():
    app = BalatroApp()
    async with app.run_test() as pilot:
        await _boot_walk(pilot, app)
        await pilot.press("enter")  # into shop
        await pilot.pause()
        await pilot.press("r")  # reroll (roll -> 1)
        await pilot.press("r")  # reroll (roll -> 2)
        await pilot.pause()
        assert app.walk.roll == 2
        await pilot.press("p")  # pin focused card
        await pilot.pause()
        shop_pins = [p for p in app.walk.report.pins if p.kind == "shop"]
        assert shop_pins and shop_pins[0].roll == 3  # within_3 in notation


async def test_find_input_reports_depth_without_advancing():
    app = BalatroApp()
    async with app.run_test() as pilot:
        await _boot_walk(pilot, app)
        await pilot.press("enter")  # into shop
        await pilot.pause()
        before = [c.center_key for c in app.walk.state.shop.cards]
        await pilot.press("f")  # focus find box
        for ch in "blueprint":
            await pilot.press(ch)
        await pilot.press("enter")  # search
        await pilot.pause()
        # The lookup runs on a copy: the live shop is unchanged.
        assert [c.center_key for c in app.walk.state.shop.cards] == before
        assert app.walk.roll == 0


async def test_skip_pack_tag_opens_pack_over_blind_screen():
    app = BalatroApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.walk = SeedWalk(CHARM_SKIP_SEED, data=app.game_data)
        await app.push_screen(SeedWalkBlindScreen())
        await pilot.pause()
        await pilot.press("s")  # skip small -> Charm tag -> Mega Arcana pack
        await pilot.pause()
        # The pack modal sits on top of the refreshed blind screen (not replacing it).
        assert isinstance(app.screen, SeedWalkPackScreen)
        assert isinstance(app.screen_stack[-2], SeedWalkBlindScreen)
        await pilot.press("p")  # pin pack contents
        await pilot.press("escape")  # close pack
        await pilot.pause()
        assert isinstance(app.screen, SeedWalkBlindScreen)
        assert app.walk.on_deck == "Big"
        assert any(p.kind == "skip" for p in app.walk.report.pins)


async def test_report_modal_opens_and_next_advances():
    app = BalatroApp()
    async with app.run_test() as pilot:
        await _boot_walk(pilot, app)
        await pilot.press("enter")  # into shop
        await pilot.pause()
        await pilot.press("R")  # report modal
        await pilot.pause()
        assert isinstance(app.screen, ReportModal)
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("n")  # next blind
        await pilot.pause()
        assert isinstance(app.screen, SeedWalkBlindScreen)
        assert app.walk.on_deck == "Big" and not app.walk.in_shop
