"""BlindSelectScreen, choose which blind to face."""

from __future__ import annotations

from math import floor

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Static

from pylatro import get_blind_amount

from ..widgets.blind_panel import BlindPanel


class BlindSelectScreen(Screen):
    """Three-panel blind selection screen."""

    BINDINGS = [
        Binding("h", "focus_left", "Left", show=False),
        Binding("l", "focus_right", "Right", show=False),
        Binding("left", "focus_left", "Left", show=False),
        Binding("right", "focus_right", "Right", show=False),
        Binding("enter", "select_blind", "Select"),
        Binding("s", "skip_blind", "Skip"),
    ]

    DEFAULT_CSS = """
    BlindSelectScreen {
        layout: vertical;
    }
    #blind-header {
        height: 3;
        text-align: center;
        padding: 1;
    }
    #blind-panels {
        height: 1fr;
        layout: horizontal;
    }
    """

    focus_idx: reactive[int] = reactive(0)

    def compose(self) -> ComposeResult:
        ctrl = self.app.controller
        state = ctrl.state
        assert state is not None

        yield Static(
            f"Ante {state.round_resets.ante}, Select Your Blind",
            id="blind-header",
        )
        with Horizontal(id="blind-panels"):
            for i, blind_type in enumerate(("Small", "Big", "Boss")):
                panel = self._make_panel(blind_type, i)
                yield panel

    def on_mount(self) -> None:
        """Auto-focus the current selectable blind."""
        state = self.app.controller.state
        if state is None:
            return
        for i, blind_type in enumerate(("Small", "Big", "Boss")):
            if state.round_resets.blind_states.get(blind_type) == "Select":
                self.focus_idx = i
                break

    def _make_panel(self, blind_type: str, idx: int) -> BlindPanel:
        ctrl = self.app.controller
        state = ctrl.state
        assert state is not None

        resets = state.round_resets
        blind_key = resets.blind_choices.get(blind_type, "bl_small")
        blind_data = state.data.blinds.get(blind_key, {})
        blind_name = blind_data.get("name", blind_type)

        # Compute target
        ante = resets.ante
        scaling = min(state.stake, 3)
        base = get_blind_amount(ante, scaling)
        mult = blind_data.get("mult", 1)
        chip_target = floor(base * mult)

        # Reward
        reward_map = {"Small": 3, "Big": 4, "Boss": 5}
        reward = reward_map.get(blind_type, 3) + ante - 1

        # State
        blind_state = resets.blind_states.get(blind_type, "Upcoming")

        # Tag
        tag_key = resets.blind_tags.get(blind_type, "")
        tag_name = ""
        if tag_key:
            tag_data = state.data.tags.get(tag_key, {})
            tag_name = tag_data.get("name", tag_key)

        # Boss description
        boss_desc = ""
        if blind_type == "Boss":
            boss_desc = blind_data.get("debuff_text", blind_data.get("description", ""))

        return BlindPanel(
            blind_type=blind_type,
            blind_key=blind_key,
            blind_name=blind_name,
            chip_target=chip_target,
            reward=reward,
            state=blind_state,
            tag_name=tag_name,
            boss_desc=boss_desc,
            is_focused=(idx == self.focus_idx),
            id=f"blind-{blind_type.lower()}",
        )

    def _refresh_panels(self) -> None:
        for i, blind_type in enumerate(("Small", "Big", "Boss")):
            panel = self.query_one(f"#blind-{blind_type.lower()}", BlindPanel)
            panel.is_focused = i == self.focus_idx
            panel.refresh()

    def watch_focus_idx(self, value: int) -> None:
        if self.is_mounted:
            self._refresh_panels()

    def action_focus_left(self) -> None:
        if self.focus_idx > 0:
            self.focus_idx -= 1

    def action_focus_right(self) -> None:
        if self.focus_idx < 2:
            self.focus_idx += 1

    def action_select_blind(self) -> None:
        blind_types = ("Small", "Big", "Boss")
        blind_type = blind_types[self.focus_idx]
        ctrl = self.app.controller
        state = ctrl.state
        assert state is not None

        # Can only select the current "Select" blind
        if state.round_resets.blind_states.get(blind_type) != "Select":
            self.notify(f"Cannot select {blind_type} blind right now", severity="warning")
            return

        ctrl.select_blind(blind_type)

        self.app.switch_screen(HandPlayScreen())

    def action_skip_blind(self) -> None:
        blind_types = ("Small", "Big", "Boss")
        blind_type = blind_types[self.focus_idx]

        if blind_type == "Boss":
            self.notify("Cannot skip Boss blind", severity="warning")
            return

        ctrl = self.app.controller
        state = ctrl.state
        assert state is not None

        if state.round_resets.blind_states.get(blind_type) != "Select":
            self.notify(f"Cannot skip {blind_type} blind right now", severity="warning")
            return

        next_blind = ctrl.skip_blind()
        # Rebuild the screen to show updated states
        self.app.switch_screen(BlindSelectScreen())


# Import at bottom to avoid circular
from .hand_play import HandPlayScreen  # noqa: E402
