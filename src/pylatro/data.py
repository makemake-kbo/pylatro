from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

_GAME_DATA_JSON = Path(__file__).resolve().parent / "game_data.json"


@dataclass(slots=True)
class GameData:
    seals: dict[str, dict[str, Any]]
    tags: dict[str, dict[str, Any]]
    stakes: dict[str, dict[str, Any]]
    blinds: dict[str, dict[str, Any]]
    cards: dict[str, dict[str, Any]]
    centers: dict[str, dict[str, Any]]
    hands: dict[str, dict[str, Any]]
    center_pools: dict[str, list[dict[str, Any]]]
    joker_rarity_pools: dict[int, list[dict[str, Any]]]


@lru_cache(maxsize=1)
def load_game_data() -> GameData:
    with _GAME_DATA_JSON.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return GameData(
        seals=raw["seals"],
        tags=raw["tags"],
        stakes=raw["stakes"],
        blinds=raw["blinds"],
        cards=raw["cards"],
        centers=raw["centers"],
        hands=raw["hands"],
        center_pools=raw["center_pools"],
        joker_rarity_pools={int(k): v for k, v in raw["joker_rarity_pools"].items()},
    )
