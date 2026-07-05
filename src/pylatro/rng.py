"""Bit-exact reimplementation of Balatro's pseudo-random generator.

Every constant here reproduces Balatro's own ``pseudohash`` / ``pseudoseed``
recurrences. They look arbitrary because they are, the only requirement is that
they reproduce Balatro's generator exactly, otherwise a given run seed produces a
different card/shop/boss sequence than the real game. Do not "clean up" or round
these numbers.
"""

from __future__ import annotations

import random as _random_mod
from dataclasses import dataclass, field
from math import pi
from typing import TYPE_CHECKING, Any, TypeVar

try:
    import cython
except ImportError:  # pragma: no cover - dependency-free core install without Cython
    from ._cyshadow import cython

if TYPE_CHECKING:
    from collections.abc import Sequence

KT = TypeVar("KT")
VT = TypeVar("VT")

# A C ``double`` copy of math.pi. Referencing ``pi`` directly inside a typed loop
# leaves it an (untyped) module global, which forces the whole expression back
# through boxed PyObject arithmetic; a typed constant keeps the recurrence in
# native C doubles when compiled.
_PI = cython.declare(cython.double, pi)

# Module-level reusable RNG instance (re-seeded before every draw).
_rng = _random_mod.Random()


def _seeded_random(seed: float, minimum: int | None = None, maximum: int | None = None) -> float | int:
    _rng.seed(seed)
    if minimum is not None and maximum is not None:
        return _rng.randint(minimum, maximum)
    return _rng.random()


def _seeded_random_after(seed: float, draws_before: int, minimum: int | None = None, maximum: int | None = None) -> float | int:
    _rng.seed(seed)
    for _ in range(draws_before):
        _rng.random()
    if minimum is not None and maximum is not None:
        return _rng.randint(minimum, maximum)
    return _rng.random()


def _seeded_random_string(length: int, seed: float) -> tuple[str, int]:
    """Generate a Balatro-style random seed string (e.g. shown on the run screen).

    Reproduces Balatro's ``random_string``: for each character, one roll picks the
    character class (digit / first-half letters / second-half letters) and a
    second roll picks the character within that class. The ``0.7``/``0.45``
    cutoffs and the ASCII ranges (``1``-``9``, ``A``-``N``, ``P``-``Z``; ``O``
    is excluded to avoid confusion with ``0``) match the game exactly.
    Returns the string plus the number of RNG draws consumed so the caller can
    advance its seed-continuation counter.
    """
    _rng.seed(seed)
    count = 0

    def draw(*args: int) -> float | int:
        nonlocal count
        count += 1
        if args:
            return _rng.randint(args[0], args[1])
        return _rng.random()

    chars: list[str] = []
    for _ in range(length):
        if draw() > 0.7:
            chars.append(chr(draw(ord("1"), ord("9"))))
        elif draw() > 0.45:
            chars.append(chr(draw(ord("A"), ord("N"))))
        else:
            chars.append(chr(draw(ord("P"), ord("Z"))))
    return "".join(chars).upper(), count


@cython.locals(length=cython.Py_ssize_t, i=cython.Py_ssize_t, j=cython.Py_ssize_t)
def _seeded_shuffle_indices(length: int, seed: float) -> list[int]:
    _rng.seed(seed)
    out = list(range(1, length + 1))
    for i in range(length, 1, -1):
        j = _rng.randint(1, i)
        out[i - 1], out[j - 1] = out[j - 1], out[i - 1]
    return out


@cython.locals(num=cython.double, index=cython.Py_ssize_t)
def pseudohash(text: str) -> float:
    """Hash a string to a float in [0, 1), reproducing Balatro's ``pseudohash``.

    Folds the characters back-to-front through a fixed recurrence; the magic
    multiplier (``1.1239285023``) and the ``* pi`` terms reproduce the game's
    hash and must not change. This is the entry point that turns a textual
    seed/key into the numeric ``hashed_seed`` driving every later draw.
    """
    num = 1.0
    for index in range(len(text), 0, -1):
        num = ((1.1239285023 / num) * ord(text[index - 1]) * _PI + _PI * index) % 1
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
        """Return the next seed value for a named RNG channel (``key``).

        Each channel advances its own state: the stored value is run through the
        same fixed recurrence (constants ``2.134453429141`` / ``1.72431234`` and
        the 13-decimal rounding all reproduce Balatro's ``pseudoseed`` exactly)
        and then averaged with the run's ``hashed_seed``. ``predict_seed`` evaluates the
        channel against a hypothetical run seed without mutating state, used for
        previewing future draws (e.g. Telescope/voucher prediction).
        """
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
        result = _seeded_random(actual_seed, minimum, maximum)
        self._record_seed(actual_seed, 1)
        return result

    def random_string(self, length: int, seed: float) -> str:
        result, draws = _seeded_random_string(length, seed)
        self._record_seed(seed, draws)
        return result

    def random_without_seed(self, minimum: int | None = None, maximum: int | None = None) -> float | int:
        if self.last_seed is None:
            raise RuntimeError("Balatro RNG continuation requested before any seeded draw")
        result = _seeded_random_after(
            self.last_seed,
            self.draws_since_seed,
            minimum,
            maximum,
        )
        self.draws_since_seed += 1
        return result

    def pseudorandom_element(self, values: Sequence[VT] | dict[KT, VT], seed: float) -> tuple[VT, KT | int]:
        items = _sorted_items(values)
        if not items:
            raise ValueError("Cannot choose an element from an empty collection")
        selected = int(self.pseudorandom(seed, 1, len(items))) - 1
        return items[selected][1], items[selected][0]

    def pseudoshuffle(self, values: list[VT], seed: float) -> list[VT]:
        working = list(values)
        if working and isinstance(working[0], dict) and "sort_id" in working[0]:
            working.sort(key=lambda item: item["sort_id"])  # type: ignore[index]
        order = _seeded_shuffle_indices(len(working), seed)
        self._record_seed(seed, max(len(working) - 1, 0))
        return [working[index - 1] for index in order]

    def _record_seed(self, seed: float, draws: int) -> None:
        self.last_seed = seed
        self.draws_since_seed = draws


def _sorted_items(values: Sequence[Any] | dict[Any, Any]) -> list[tuple[Any, Any]]:
    items: list[tuple[Any, Any]] = (
        list(values.items()) if isinstance(values, dict) else list(enumerate(values, start=1))
    )
    if items and isinstance(items[0][1], dict) and "sort_id" in items[0][1]:
        items.sort(key=lambda item: item[1]["sort_id"])
    else:
        items.sort(key=lambda item: item[0])
    return items


# Stateless module-level variants of the PseudorandomState methods above. These
# take a pre-computed numeric seed and do NOT record seed continuation, so use
# them only for one-off draws that don't feed a later random_without_seed() call.
def pseudorandom_element(values: Sequence[Any] | dict[Any, Any], seed: float) -> tuple[Any, Any]:
    items = _sorted_items(values)
    if not items:
        raise ValueError("Cannot choose an element from an empty collection")
    selected = int(_seeded_random(seed, 1, len(items))) - 1
    return items[selected][1], items[selected][0]


def pseudoshuffle(values: list[Any], seed: float) -> list[Any]:
    working = list(values)
    if working and isinstance(working[0], dict) and "sort_id" in working[0]:
        working.sort(key=lambda item: item["sort_id"])  # type: ignore[index]
    order = _seeded_shuffle_indices(len(working), seed)
    return [working[index - 1] for index in order]
