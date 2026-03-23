from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .upstream import UPSTREAM_ROOT, get_lua_bridge


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


TABLE_ASSIGNMENTS = {
    "seals": "self.P_SEALS =",
    "tags": "self.P_TAGS =",
    "stakes": "self.P_STAKES =",
    "blinds": "self.P_BLINDS =",
    "cards": "self.P_CARDS =",
    "centers": "self.P_CENTERS =",
    "hands": "        hands = {",
}


def _extract_table(source: str, assignment: str) -> str:
    start = source.index(assignment)
    brace_start = source.index("{", start)
    depth = 0
    in_single = False
    in_double = False
    in_comment = False
    previous = ""

    for index in range(brace_start, len(source)):
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""

        if in_comment:
            if char == "\n":
                in_comment = False
        elif in_single:
            if char == "'" and previous != "\\":
                in_single = False
        elif in_double:
            if char == '"' and previous != "\\":
                in_double = False
        else:
            if char == "-" and next_char == "-":
                in_comment = True
            elif char == "'":
                in_single = True
            elif char == '"':
                in_double = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return source[brace_start : index + 1]

        previous = char

    raise ValueError(f"Unbalanced table while parsing {assignment!r}")


def _load_game_source() -> str:
    return (UPSTREAM_ROOT / "game.lua").read_text(encoding="utf-8")


def _load_table(source: str, assignment: str) -> dict[str, dict[str, Any]]:
    bridge = get_lua_bridge()
    table_source = _extract_table(source, assignment)
    return bridge.to_python(bridge.eval_table(table_source))


def _copy_proto_map(mapping: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {key: {**value} for key, value in mapping.items()}


def _build_pools(
    seals: dict[str, dict[str, Any]],
    tags: dict[str, dict[str, Any]],
    stakes: dict[str, dict[str, Any]],
    centers: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    center_pools: dict[str, list[dict[str, Any]]] = {
        "Booster": [],
        "Default": [],
        "Enhanced": [],
        "Edition": [],
        "Joker": [],
        "Tarot": [],
        "Planet": [],
        "Tarot_Planet": [],
        "Spectral": [],
        "Consumeables": [],
        "Voucher": [],
        "Back": [],
        "Tag": [],
        "Seal": [],
        "Stake": [],
        "Demo": [],
    }
    rarity_pools: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: [], 4: []}

    for key, value in seals.items():
        value["key"] = key
        center_pools["Seal"].append(value)

    for key, value in tags.items():
        value["key"] = key
        center_pools["Tag"].append(value)

    for key, value in stakes.items():
        value["key"] = key
        center_pools["Stake"].append(value)

    for key, value in centers.items():
        value["key"] = key
        if value.get("set") == "Joker":
            center_pools["Joker"].append(value)
        if value.get("set") and value.get("demo") and value.get("pos"):
            center_pools["Demo"].append(value)
        if not value.get("wip"):
            proto_set = value.get("set")
            if proto_set and proto_set != "Joker" and not value.get("skip_pool") and not value.get("omit"):
                center_pools[proto_set].append(value)
            if proto_set in {"Tarot", "Planet"}:
                center_pools["Tarot_Planet"].append(value)
            if value.get("consumeable"):
                center_pools["Consumeables"].append(value)
            if proto_set == "Joker" and value.get("rarity") and not value.get("demo"):
                rarity_pools[int(value["rarity"])].append(value)

    for pool_name in (
        "Joker",
        "Tarot",
        "Planet",
        "Tarot_Planet",
        "Spectral",
        "Voucher",
        "Booster",
        "Consumeables",
        "Enhanced",
        "Stake",
        "Tag",
        "Seal",
    ):
        center_pools[pool_name].sort(key=lambda item: item["order"])
    center_pools["Back"].sort(key=lambda item: item["order"] - (100 if item.get("unlocked") else 0))
    center_pools["Demo"].sort(key=lambda item: item["order"] + (1000 if item.get("set") == "Joker" else 0))

    for pool in rarity_pools.values():
        pool.sort(key=lambda item: item["order"])

    return center_pools, rarity_pools


@lru_cache(maxsize=1)
def load_game_data() -> GameData:
    source = _load_game_source()
    seals = _copy_proto_map(_load_table(source, TABLE_ASSIGNMENTS["seals"]))
    tags = _copy_proto_map(_load_table(source, TABLE_ASSIGNMENTS["tags"]))
    stakes = _copy_proto_map(_load_table(source, TABLE_ASSIGNMENTS["stakes"]))
    blinds = _copy_proto_map(_load_table(source, TABLE_ASSIGNMENTS["blinds"]))
    cards = _copy_proto_map(_load_table(source, TABLE_ASSIGNMENTS["cards"]))
    centers = _copy_proto_map(_load_table(source, TABLE_ASSIGNMENTS["centers"]))
    hands = _copy_proto_map(_load_table(source, TABLE_ASSIGNMENTS["hands"]))

    for key, value in blinds.items():
        value["key"] = key

    center_pools, joker_rarity_pools = _build_pools(seals, tags, stakes, centers)
    return GameData(
        seals=seals,
        tags=tags,
        stakes=stakes,
        blinds=blinds,
        cards=cards,
        centers=centers,
        hands=hands,
        center_pools=center_pools,
        joker_rarity_pools=joker_rarity_pools,
    )
