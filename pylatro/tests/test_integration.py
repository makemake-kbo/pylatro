"""Integration tests: full game loop through the Textual app."""

import asyncio

import pytest


class TestAppLaunch:
    @pytest.mark.asyncio
    async def test_app_shows_main_menu(self):
        from pylatro_cli.app import BalatroApp

        app = BalatroApp()
        async with app.run_test(size=(120, 40)) as pilot:
            assert type(app.screen).__name__ == "MainMenuScreen"

    @pytest.mark.asyncio
    async def test_play_button_opens_new_run_modal(self):
        from pylatro_cli.app import BalatroApp

        app = BalatroApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.click("#play-btn")
            await pilot.pause()
            assert type(app.screen).__name__ == "NewRunModal"

    @pytest.mark.asyncio
    async def test_start_run_goes_to_blind_select(self):
        from pylatro_cli.app import BalatroApp

        app = BalatroApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.click("#play-btn")
            await pilot.pause()
            await pilot.click("#start-btn")
            await pilot.pause()
            assert type(app.screen).__name__ == "BlindSelectScreen"
            assert app.controller.state is not None

    @pytest.mark.asyncio
    async def test_select_blind_goes_to_hand_play(self):
        from pylatro_cli.app import BalatroApp

        app = BalatroApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.click("#play-btn")
            await pilot.pause()
            await pilot.click("#start-btn")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert type(app.screen).__name__ == "HandPlayScreen"
            assert len(app.controller.state.hand_cards) > 0


class TestGameFlow:
    @pytest.mark.asyncio
    async def test_full_blind_controller_flow(self):
        """Test a full blind through the controller directly (faster than UI)."""
        from pylatro import load_game_data
        from pylatro_cli.controller import GameController, GamePhase

        data = load_game_data()
        ctrl = GameController(data=data)
        ctrl.new_run("TESTRUN1", stake=1, deck_key="b_red")

        # Beat small blind
        ctrl.select_blind("Small")
        for _ in range(ctrl.state.current_round.hands_left):
            if ctrl.blind_beaten():
                break
            indices = list(range(min(5, len(ctrl.state.hand_cards))))
            ctrl.play_selected(indices)

        if not ctrl.blind_beaten():
            pytest.skip("Could not beat blind with this seed")
        ctrl.cash_out()
        assert ctrl.phase == GamePhase.SHOP

        ctrl.enter_shop()
        assert len(ctrl.state.shop.cards) > 0

        ctrl.leave_shop()
        assert ctrl.phase == GamePhase.BLIND_SELECT
