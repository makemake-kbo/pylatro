from __future__ import annotations

import pytest

from pylatro.runtime import compute_interest

# Default economy: $1 interest per $5, capped at the $25 cash threshold (=> max $5/round).
DEFAULT_CAP = 25
DEFAULT_AMOUNT = 1

# Seed Money raises the cash threshold to $50, Money Tree to $100.
SEED_MONEY_CAP = 50
MONEY_TREE_CAP = 100

# To the Moon raises interest_amount by 1 (=> $2 per tier) but does NOT raise the cap.
TO_THE_MOON_AMOUNT = 2


@pytest.mark.parametrize(
    ("dollars", "expected"),
    [
        (0, 0),
        (4, 0),
        (5, 1),
        (24, 4),
        (25, 5),
        (50, 5),  # capped at the $25 threshold => 5 tiers
        (100, 5),
    ],
)
def test_default_cap(dollars: int, expected: int) -> None:
    assert compute_interest(dollars, DEFAULT_CAP, DEFAULT_AMOUNT) == expected


@pytest.mark.parametrize(
    ("dollars", "expected"),
    [
        (50, 10),  # 10 tiers * $1
        (100, 10),  # still capped at $50 threshold
    ],
)
def test_seed_money(dollars: int, expected: int) -> None:
    assert compute_interest(dollars, SEED_MONEY_CAP, DEFAULT_AMOUNT) == expected


def test_money_tree() -> None:
    assert compute_interest(100, MONEY_TREE_CAP, DEFAULT_AMOUNT) == 20


@pytest.mark.parametrize(
    ("dollars", "expected"),
    [
        (25, 10),  # 5 tiers * $2
        (50, 10),  # cap unchanged at $25 => still 5 tiers
    ],
)
def test_to_the_moon(dollars: int, expected: int) -> None:
    assert compute_interest(dollars, DEFAULT_CAP, TO_THE_MOON_AMOUNT) == expected


def test_seed_money_plus_to_the_moon() -> None:
    # Cash threshold $50 (Seed Money) and $2/tier (To the Moon) => 10 tiers * $2.
    assert compute_interest(50, SEED_MONEY_CAP, TO_THE_MOON_AMOUNT) == 20


def test_no_interest_amount() -> None:
    assert compute_interest(100, DEFAULT_CAP, 0) == 0
