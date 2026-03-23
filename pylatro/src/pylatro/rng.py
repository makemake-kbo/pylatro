from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import Any, Sequence

from .upstream import get_luajit_bridge


def pseudohash(text: str) -> float:
    num = 1.0
    for index in range(len(text), 0, -1):
        num = ((1.1239285023 / num) * ord(text[index - 1]) * pi + pi * index) % 1
    return num


@dataclass(slots=True)
class PseudorandomState:
    seed: str
    values: dict[str, float] = field(default_factory=dict)
    hashed_seed: float = field(init=False)
    last_seed: float | None = field(default=None, init=False)
    draws_since_seed: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.hashed_seed = pseudohash(self.seed)

    def pseudoseed(self, key: str, predict_seed: str | None = None) -> float:
        if key == "seed":
            raise NotImplementedError("Balatro's raw seed channel is not used in the headless core yet")

        if predict_seed is not None:
            predicted = pseudohash(key + predict_seed)
            predicted = abs(float(f"{(2.134453429141 + predicted * 1.72431234) % 1:.13f}"))
            return (predicted + pseudohash(predict_seed)) / 2

        if key not in self.values:
            self.values[key] = pseudohash(key + self.seed)

        self.values[key] = abs(float(f"{(2.134453429141 + self.values[key] * 1.72431234) % 1:.13f}"))
        return (self.values[key] + self.hashed_seed) / 2

    def pseudorandom(self, seed: str | float, minimum: int | None = None, maximum: int | None = None) -> float | int:
        actual_seed = self.pseudoseed(seed) if isinstance(seed, str) else seed
        result = get_luajit_bridge().random(actual_seed, minimum, maximum)
        self._record_seed(actual_seed, 1)
        return result

    def random_string(self, length: int, seed: float) -> str:
        result, draws = get_luajit_bridge().random_string(length, seed)
        self._record_seed(seed, draws)
        return result

    def random_without_seed(self, minimum: int | None = None, maximum: int | None = None) -> float | int:
        if self.last_seed is None:
            raise RuntimeError("Balatro RNG continuation requested before any seeded draw")
        result = get_luajit_bridge().random(
            self.last_seed,
            minimum,
            maximum,
            draws_before=self.draws_since_seed,
        )
        self.draws_since_seed += 1
        return result

    def pseudorandom_element(self, values: Sequence[Any] | dict[Any, Any], seed: float) -> tuple[Any, Any]:
        items = _sorted_items(values)
        if not items:
            raise ValueError("Cannot choose an element from an empty collection")
        selected = int(self.pseudorandom(seed, 1, len(items))) - 1
        return items[selected][1], items[selected][0]

    def pseudoshuffle(self, values: list[Any], seed: float) -> list[Any]:
        working = list(values)
        if working and isinstance(working[0], dict) and "sort_id" in working[0]:
            working.sort(key=lambda item: item["sort_id"])
        order = get_luajit_bridge().shuffle_indices(len(working), seed)
        self._record_seed(seed, max(len(working) - 1, 0))
        return [working[index - 1] for index in order]

    def _record_seed(self, seed: float, draws: int) -> None:
        self.last_seed = seed
        self.draws_since_seed = draws


def _sorted_items(values: Sequence[Any] | dict[Any, Any]) -> list[tuple[Any, Any]]:
    items = list(values.items()) if isinstance(values, dict) else list(enumerate(values, start=1))
    if items and isinstance(items[0][1], dict) and "sort_id" in items[0][1]:
        items.sort(key=lambda item: item[1]["sort_id"])
    else:
        items.sort(key=lambda item: item[0])
    return items


def pseudorandom_element(values: Sequence[Any] | dict[Any, Any], seed: float) -> tuple[Any, Any]:
    items = _sorted_items(values)
    if not items:
        raise ValueError("Cannot choose an element from an empty collection")
    selected = int(get_luajit_bridge().random(seed, 1, len(items))) - 1
    return items[selected][1], items[selected][0]


def pseudoshuffle(values: list[Any], seed: float) -> list[Any]:
    working = list(values)
    if working and isinstance(working[0], dict) and "sort_id" in working[0]:
        working.sort(key=lambda item: item["sort_id"])
    order = get_luajit_bridge().shuffle_indices(len(working), seed)
    return [working[index - 1] for index in order]
