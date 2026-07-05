"""BlindPanel, a single blind option in blind select screen."""

from __future__ import annotations

from rich.text import Text
from textual.message import Message
from textual.widget import Widget

from ..theme import BALATRO_PALETTE


class BlindPanel(Widget):
    """Displays a single blind option: name, target, reward, state."""

    DEFAULT_CSS = """
    BlindPanel {
        width: 1fr;
        height: auto;
        min-height: 12;
        background: #1c1c3a;
        border: solid #5c5c8a;
        padding: 1;
        margin: 1;
    }
    """

    class Selected(Message):
        def __init__(self, blind_type: str) -> None:
            super().__init__()
            self.blind_type = blind_type

    class Skipped(Message):
        def __init__(self, blind_type: str) -> None:
            super().__init__()
            self.blind_type = blind_type

    def __init__(
        self,
        blind_type: str,
        blind_key: str,
        blind_name: str,
        chip_target: int,
        reward: int,
        state: str,
        tag_name: str = "",
        boss_desc: str = "",
        is_focused: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.blind_type = blind_type
        self.blind_key = blind_key
        self.blind_name = blind_name
        self.chip_target = chip_target
        self.reward = reward
        self.blind_state = state
        self.tag_name = tag_name
        self.boss_desc = boss_desc
        self.is_focused = is_focused

    def render(self) -> Text:
        t = Text()

        # Header
        type_colors = {"Small": "#42a5f5", "Big": "#ffc107", "Boss": "#e74c3c"}
        header_color = type_colors.get(self.blind_type, "#ecf0f1")
        t.append(f"  {self.blind_type} Blind\n", style=f"bold {header_color}")
        t.append("─" * 20 + "\n")

        # Name
        t.append(f"  {self.blind_name}\n", style="bold #ecf0f1")

        # Target
        t.append("  Score: ", style="#95a5a6")
        t.append(f"{self.chip_target:,}\n", style="bold #ecf0f1")

        # Reward
        t.append("  Reward: ", style="#95a5a6")
        t.append(f"${self.reward}\n", style=f"bold {BALATRO_PALETTE['gold']}")

        # Boss description
        if self.boss_desc:
            t.append("\n")
            t.append(f"  {self.boss_desc}\n", style="italic #e74c3c")

        # Tag
        if self.tag_name:
            t.append("  Tag: ", style="#95a5a6")
            t.append(f"{self.tag_name}\n", style="#ce93d8")

        # State / action hint
        t.append("\n")
        if self.blind_state == "Select":
            if self.is_focused:
                t.append("  >> [Enter] Select <<\n", style=f"bold {BALATRO_PALETTE['card_selected']}")
            else:
                t.append("  [Enter] Select\n", style="#95a5a6")
            if self.blind_type in ("Small", "Big"):
                t.append("  [s] Skip\n", style="#95a5a6")
        elif self.blind_state == "Upcoming":
            t.append("  Upcoming\n", style="#636e72")
        elif self.blind_state == "Defeated":
            t.append("  Defeated\n", style="#27ae60")
        elif self.blind_state == "Skipped":
            t.append("  Skipped\n", style="#95a5a6")

        return t
