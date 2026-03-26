"""ShopItemWidget — a single purchasable item in the shop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.message import Message
from textual.widget import Widget

from ..theme import BALATRO_PALETTE

if TYPE_CHECKING:
    from pylatro import GameData
    from pylatro.models import ShopCard


class ShopItemWidget(Widget):
    """Displays a shop item: name, type, cost, affordability."""

    DEFAULT_CSS = """
    ShopItemWidget {
        width: auto;
        min-width: 14;
        height: auto;
        min-height: 5;
        margin: 0 1;
        padding: 1;
        background: #1c1c3a;
        border: solid #5c5c8a;
    }
    """

    class BuyRequested(Message):
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    def __init__(
        self,
        shop_card: ShopCard,
        index: int,
        data: GameData,
        affordable: bool = True,
        is_focused: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.shop_card = shop_card
        self.index = index
        self._data = data
        self.affordable = affordable
        self.is_focused = is_focused

    def render(self) -> Text:
        sc = self.shop_card
        center = self._data.centers.get(sc.center_key, {})
        name = center.get("name", sc.center_key)
        card_type = sc.card_type

        t = Text()

        # Type label
        type_colors = {
            "Joker": "#ffc107",
            "Tarot": "#9c27b0",
            "Planet": "#2196f3",
            "Spectral": "#b0bec5",
            "Voucher": "#ffc107",
            "Enhanced": "#27ae60",
            "Base": "#ecf0f1",
        }
        tc = type_colors.get(card_type, "#95a5a6")
        t.append(f"{card_type}\n", style=tc)

        # Name
        border_style = BALATRO_PALETTE["card_selected"] if self.is_focused else "#ecf0f1"
        t.append(f"{name}\n", style=f"bold {border_style}")

        # Edition/seal badges
        if sc.edition:
            ed_name = next(iter(sc.edition))
            t.append(f"  [{ed_name}]", style=BALATRO_PALETTE.get(f"edition_{ed_name}", "#95a5a6"))
            t.append("\n")

        # Cost
        t.append("\n$", style=BALATRO_PALETTE["gold"])
        cost_color = BALATRO_PALETTE["gold"] if self.affordable else "#e74c3c"
        t.append(f"{sc.cost}", style=f"bold {cost_color}")

        if self.is_focused:
            t.append("\n[Enter] Buy", style="#95a5a6")

        return t
