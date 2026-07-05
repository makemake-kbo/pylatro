"""InfoSidebar, left sidebar showing blind info, score, hands/discards/dollars."""

from __future__ import annotations

from rich.text import Text
from textual.widget import Widget

from ..theme import BALATRO_PALETTE


class InfoSidebar(Widget):
    """Left sidebar: blind name, target, score progress, hand info, hands/discards/money."""

    DEFAULT_CSS = """
    InfoSidebar {
        width: 24;
        height: 100%;
        background: #1c1c3a;
        border-right: solid #5c5c8a;
        padding: 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.blind_name: str = ""
        self.blind_target: int = 0
        self.round_score: int = 0
        self.hand_name: str = ""
        self.hand_level: int = 0
        self.chips: int = 0
        self.mult: int = 0
        self.hands_left: int = 0
        self.discards_left: int = 0
        self.dollars: int = 0
        self.ante: int = 1
        self.round_num: int = 0

    def update_info(
        self,
        *,
        blind_name: str = "",
        blind_target: int = 0,
        round_score: int = 0,
        hand_name: str = "",
        hand_level: int = 0,
        chips: int = 0,
        mult: int = 0,
        hands_left: int = 0,
        discards_left: int = 0,
        dollars: int = 0,
        ante: int = 1,
        round_num: int = 0,
    ) -> None:
        self.blind_name = blind_name
        self.blind_target = blind_target
        self.round_score = round_score
        self.hand_name = hand_name
        self.hand_level = hand_level
        self.chips = chips
        self.mult = mult
        self.hands_left = hands_left
        self.discards_left = discards_left
        self.dollars = dollars
        self.ante = ante
        self.round_num = round_num
        self.refresh()

    def render(self) -> Text:
        t = Text()
        t.append(f"Ante {self.ante}", style="bold #ecf0f1")
        t.append(f"  Round {self.round_num}\n", style="#95a5a6")
        t.append("─" * 22 + "\n")

        # Blind info
        t.append(f"{self.blind_name}\n", style="bold #e74c3c")
        t.append("Target: ", style="#95a5a6")
        t.append(f"{self.blind_target:,}\n", style="bold #ecf0f1")

        # Score progress
        t.append("Score:  ", style="#95a5a6")
        score_color = "#27ae60" if self.round_score >= self.blind_target else "#ecf0f1"
        t.append(f"{self.round_score:,}\n", style=f"bold {score_color}")

        # Progress bar
        if self.blind_target > 0:
            pct = min(1.0, self.round_score / self.blind_target)
            bar_w = 20
            filled = int(pct * bar_w)
            t.append("▓" * filled, style=BALATRO_PALETTE["chips"])
            t.append("░" * (bar_w - filled), style="#4a4a6a")
            t.append("\n")

        t.append("─" * 22 + "\n")

        # Hand preview
        if self.hand_name:
            t.append(f"{self.hand_name}", style=f"bold {BALATRO_PALETTE['card_selected']}")
            if self.hand_level > 0:
                t.append(f" Lv.{self.hand_level}", style="#95a5a6")
            t.append("\n")
            t.append(f"{self.chips}", style=f"bold {BALATRO_PALETTE['chips']}")
            t.append(" × ", style="#95a5a6")
            t.append(f"{self.mult}", style=f"bold {BALATRO_PALETTE['mult']}")
            t.append("\n")

        t.append("─" * 22 + "\n")

        # Hands / Discards / Dollars
        t.append("Hands:    ", style="#95a5a6")
        hands_color = "#e74c3c" if self.hands_left <= 1 else BALATRO_PALETTE["chips"]
        t.append(f"{self.hands_left}\n", style=f"bold {hands_color}")

        t.append("Discards: ", style="#95a5a6")
        disc_color = "#e74c3c" if self.discards_left <= 0 else "#27ae60"
        t.append(f"{self.discards_left}\n", style=f"bold {disc_color}")

        t.append("─" * 22 + "\n")
        t.append("$", style=f"bold {BALATRO_PALETTE['gold']}")
        t.append(f"{self.dollars}", style=f"bold {BALATRO_PALETTE['gold']}")

        return t
