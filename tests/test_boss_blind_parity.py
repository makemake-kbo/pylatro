"""Boss blind parity tests: verify all 28 boss blind effects match between Python and Lua.

Strategy:
1. Play through Small and Big blinds using the bot helper
2. Force the Boss blind choice to a specific boss key on both sides
3. Call start_blind and compare snapshots (captures debuffs, hand size, etc.)
4. Play one hand and compare again (captures scoring effects, hook, tooth, ox, etc.)
"""

from __future__ import annotations

import pytest
from pathlib import Path

from pylatro import create_run_state, load_game_data
from pylatro.flow import start_blind, play_cards
from pylatro.blind import cash_out
from pylatro.shop import populate_shop, finish_shop
from pylatro.upstream.oracle_bridge import (
    OracleBridge,
    diff_snapshots,
    snapshot_from_run_state,
    snapshot_from_lua_state,
)

VENDOR_PATH = Path(__file__).resolve().parents[1] / "vendor" / "balatro_lua"

pytestmark = pytest.mark.skipif(
    not VENDOR_PATH.exists(),
    reason="vendor/balatro_lua not found",
)


def _get_all_boss_keys() -> list[str]:
    """Return sorted list of all boss blind keys from game data."""
    data = load_game_data()
    boss_keys = [k for k, v in data.blinds.items() if v.get("boss")]
    boss_keys.sort()
    return boss_keys


def _boss_blind_snapshot(py_state, bridge, lua_raw) -> tuple[dict, dict]:
    """Build extended snapshots for boss blind comparison.

    Extends the standard snapshot with boss-blind-specific fields:
    - hand_size: the current round hand size (affected by The Manacle)
    - debuffed_cards_count: number of debuffed cards in deck (suit/face debuffs)
    - debuffed_hand_cards_count: number of debuffed cards currently in hand
    """
    base_py = snapshot_from_run_state(py_state)
    base_lua = snapshot_from_lua_state(bridge.snapshot(lua_raw))

    # Add hand_size from current_round
    base_py["hand_size"] = py_state.current_round.hand_size
    lua_dict = bridge.snapshot(lua_raw)
    cr = lua_dict.get("current_round", {})
    base_lua["hand_size"] = int(cr.get("hand_size", 0)) if isinstance(cr, dict) else 0

    # Add debuffed card counts from deck_cards
    base_py["debuffed_deck_cards_count"] = sum(
        1 for c in py_state.deck_cards if c.debuff
    )

    lua_deck = lua_dict.get("deck_cards", [])
    if isinstance(lua_deck, list):
        base_lua["debuffed_deck_cards_count"] = sum(
            1 for c in lua_deck if isinstance(c, dict) and c.get("debuff")
        )
    else:
        base_lua["debuffed_deck_cards_count"] = 0

    # Add debuffed hand card count
    base_py["debuffed_hand_cards_count"] = sum(
        1 for c in py_state.hand_cards if c.debuff
    )
    lua_hand = lua_dict.get("hand_cards", [])
    if isinstance(lua_hand, list):
        base_lua["debuffed_hand_cards_count"] = sum(
            1 for c in lua_hand if isinstance(c, dict) and c.get("debuff")
        )
    else:
        base_lua["debuffed_hand_cards_count"] = 0

    # Add face-down hand card count (for The Wheel, The House, The Mark)
    base_py["face_down_hand_cards_count"] = sum(
        1 for c in py_state.hand_cards if c.face_down
    )
    if isinstance(lua_hand, list):
        base_lua["face_down_hand_cards_count"] = sum(
            1 for c in lua_hand if isinstance(c, dict) and c.get("face_down")
        )
    else:
        base_lua["face_down_hand_cards_count"] = 0

    return base_py, base_lua


def _play_through_small_big(py_state, bridge, lua_raw):
    """Play through Small and Big blinds on both sides using a simple bot.

    Returns the updated lua_raw.
    """
    # Small blind
    start_blind(py_state, "Small")
    for _ in range(4):
        play_cards(py_state, list(range(min(5, len(py_state.hand_cards)))))
    py_state.round_resets.blind_states["Small"] = "Defeated"
    py_state.round_resets.blind_states["Big"] = "Select"
    py_state.blind_on_deck = "Big"
    cash_out(py_state)
    populate_shop(py_state)
    finish_shop(py_state)

    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")
    for _ in range(4):
        lua_raw = bridge.step(lua_raw, "play_hand", card_indices=[1, 2, 3, 4, 5])
    lua_raw = bridge.step(lua_raw, "defeat_blind")
    lua_raw = bridge.step(lua_raw, "populate_shop")
    lua_raw = bridge.step(lua_raw, "finish_shop")

    # Big blind
    start_blind(py_state, "Big")
    for _ in range(4):
        play_cards(py_state, list(range(min(5, len(py_state.hand_cards)))))
    py_state.round_resets.blind_states["Big"] = "Defeated"
    py_state.round_resets.blind_states["Boss"] = "Select"
    py_state.blind_on_deck = "Boss"
    cash_out(py_state)
    populate_shop(py_state)
    finish_shop(py_state)

    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Big")
    for _ in range(4):
        lua_raw = bridge.step(lua_raw, "play_hand", card_indices=[1, 2, 3, 4, 5])
    lua_raw = bridge.step(lua_raw, "defeat_blind")
    lua_raw = bridge.step(lua_raw, "populate_shop")
    lua_raw = bridge.step(lua_raw, "finish_shop")

    return lua_raw


def _force_boss(py_state, lua_raw, boss_key: str) -> None:
    """Force the boss blind choice to a specific key on both sides."""
    py_state.round_resets.blind_choices["Boss"] = boss_key
    lua_raw.round_resets.blind_choices.Boss = boss_key


ALL_BOSS_KEYS = _get_all_boss_keys()


@pytest.mark.parametrize("boss_key", ALL_BOSS_KEYS)
def test_boss_blind_start_parity(boss_key: str):
    """After start_blind for each boss, Python and Lua have identical state.

    This captures:
    - Debuff effects (suit debuffs for The Goad, The Club, The Head, The Window)
    - Face debuffs (The Plant)
    - Hand size reduction (The Manacle)
    - Discard reduction (The Water)
    - Hand limit reduction (The Needle)
    - Card face-down effects (The Wheel, The House, The Mark)
    - Verdant Leaf all-card debuff
    - Pillar: cards played this ante are debuffed
    """
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")

    # Quickest approach: directly force boss without playing through small/big
    # This matches what the boss mini-test already verified; the richer test
    # (test_boss_blind_full_parity) uses the full bot approach.
    py_state.round_resets.blind_choices["Boss"] = boss_key
    py_state.round_resets.blind_states["Small"] = "Defeated"
    py_state.round_resets.blind_states["Big"] = "Defeated"
    py_state.round_resets.blind_states["Boss"] = "Select"
    py_state.blind_on_deck = "Boss"

    lua_raw.round_resets.blind_choices.Boss = boss_key
    lua_raw.round_resets.blind_states.Small = "Defeated"
    lua_raw.round_resets.blind_states.Big = "Defeated"
    lua_raw.round_resets.blind_states.Boss = "Select"
    lua_raw.blind_on_deck = "Boss"

    start_blind(py_state, "Boss")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Boss")

    py_snap, lua_snap = _boss_blind_snapshot(py_state, bridge, lua_raw)
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], (
        f"Boss blind {boss_key!r} start_blind divergences:\n"
        + "\n".join(f"  {path}: py={py_val!r} lua={lua_val!r}" for path, py_val, lua_val in diffs)
    )


@pytest.mark.parametrize("boss_key", ALL_BOSS_KEYS)
def test_boss_blind_play_hand_parity(boss_key: str):
    """After start_blind + one play_hand for each boss, Python and Lua match.

    This captures scoring-time effects:
    - The Hook: discards 2 random hand cards before scoring
    - The Tooth: costs $1 per card played
    - The Ox: sets dollars to $0 when most-played hand is played
    - The Flint: halves base chips and mult
    - The Arm: de-levels the played hand type
    - The Eye / The Mouth: hand-type debuff tracking
    - The Psychic / The Club / The Head / The Window: hand size or suit debuff effects
    - The Serpent / The Fish: card flip/draw behavior
    - The Pillar / The Wheel / The House / The Mark / The Wall: draw/face-down effects
    """
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")

    py_state.round_resets.blind_choices["Boss"] = boss_key
    py_state.round_resets.blind_states["Small"] = "Defeated"
    py_state.round_resets.blind_states["Big"] = "Defeated"
    py_state.round_resets.blind_states["Boss"] = "Select"
    py_state.blind_on_deck = "Boss"

    lua_raw.round_resets.blind_choices.Boss = boss_key
    lua_raw.round_resets.blind_states.Small = "Defeated"
    lua_raw.round_resets.blind_states.Big = "Defeated"
    lua_raw.round_resets.blind_states.Boss = "Select"
    lua_raw.blind_on_deck = "Boss"

    start_blind(py_state, "Boss")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Boss")

    # Play hand - use available cards (handles bosses that reduce hand size)
    n_play = min(5, len(py_state.hand_cards))
    play_cards(py_state, list(range(n_play)))
    lua_raw = bridge.step(lua_raw, "play_hand", card_indices=list(range(1, n_play + 1)))

    py_snap, lua_snap = _boss_blind_snapshot(py_state, bridge, lua_raw)
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], (
        f"Boss blind {boss_key!r} play_hand divergences:\n"
        + "\n".join(f"  {path}: py={py_val!r} lua={lua_val!r}" for path, py_val, lua_val in diffs)
    )


@pytest.mark.parametrize("boss_key", ALL_BOSS_KEYS)
def test_boss_blind_full_parity(boss_key: str):
    """Full parity test: play Small+Big blinds, force boss, compare.

    This ensures that state accumulated during a real blind run (card
    played_this_ante flags for The Pillar, hand level tracking for The Arm,
    dollars tracking for The Ox, most_played_poker_hand, etc.) all match.
    """
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")

    # Play through Small and Big blinds
    lua_raw = _play_through_small_big(py_state, bridge, lua_raw)

    # Now verify state matches before forcing boss
    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    pre_diffs = diff_snapshots(py_snap, lua_snap)
    assert pre_diffs == [], (
        f"Pre-boss state diverges for {boss_key!r}:\n"
        + "\n".join(f"  {path}: py={py_val!r} lua={lua_val!r}" for path, py_val, lua_val in pre_diffs)
    )

    # Force the boss
    _force_boss(py_state, lua_raw, boss_key)

    # Start boss blind
    start_blind(py_state, "Boss")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Boss")

    py_snap, lua_snap = _boss_blind_snapshot(py_state, bridge, lua_raw)
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], (
        f"Boss blind {boss_key!r} after full setup start_blind divergences:\n"
        + "\n".join(f"  {path}: py={py_val!r} lua={lua_val!r}" for path, py_val, lua_val in diffs)
    )

    # Play one hand
    n_play = min(5, len(py_state.hand_cards))
    play_cards(py_state, list(range(n_play)))
    lua_raw = bridge.step(lua_raw, "play_hand", card_indices=list(range(1, n_play + 1)))

    py_snap, lua_snap = _boss_blind_snapshot(py_state, bridge, lua_raw)
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], (
        f"Boss blind {boss_key!r} after full setup play_hand divergences:\n"
        + "\n".join(f"  {path}: py={py_val!r} lua={lua_val!r}" for path, py_val, lua_val in diffs)
    )


def test_boss_blind_count():
    """Verify exactly 28 boss blinds are present in game data."""
    boss_keys = _get_all_boss_keys()
    assert len(boss_keys) == 28, f"Expected 28 boss blinds, got {len(boss_keys)}: {boss_keys}"


def test_all_boss_keys_present():
    """Verify all expected boss blind keys are present in game data."""
    expected = {
        "bl_hook", "bl_ox", "bl_house", "bl_wall", "bl_wheel", "bl_arm",
        "bl_club", "bl_fish", "bl_psychic", "bl_goad", "bl_water", "bl_window",
        "bl_manacle", "bl_eye", "bl_mouth", "bl_plant", "bl_serpent", "bl_pillar",
        "bl_needle", "bl_head", "bl_tooth", "bl_flint", "bl_mark",
        "bl_final_acorn", "bl_final_heart", "bl_final_leaf", "bl_final_bell",
        "bl_final_vessel",
    }
    actual = set(_get_all_boss_keys())
    assert expected == actual, (
        f"Missing: {expected - actual}, Extra: {actual - expected}"
    )


@pytest.mark.parametrize("boss_key,expected_hands_left", [
    ("bl_needle", 1),
    ("bl_hook", 4),   # normal
    ("bl_water", 4),  # hands_left unchanged for The Water
])
def test_boss_blind_hands_left(boss_key: str, expected_hands_left: int):
    """Verify hands_left is set correctly for specific boss blinds."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    py_state.round_resets.blind_choices["Boss"] = boss_key
    py_state.round_resets.blind_states["Small"] = "Defeated"
    py_state.round_resets.blind_states["Big"] = "Defeated"
    py_state.round_resets.blind_states["Boss"] = "Select"
    py_state.blind_on_deck = "Boss"
    start_blind(py_state, "Boss")
    assert py_state.current_round.hands_left == expected_hands_left, (
        f"{boss_key}: expected hands_left={expected_hands_left}, "
        f"got {py_state.current_round.hands_left}"
    )


@pytest.mark.parametrize("boss_key,expected_discards_left", [
    ("bl_water", 0),
    ("bl_hook", 4),   # normal (4 discards is the default)
    ("bl_needle", 4), # normal
])
def test_boss_blind_discards_left(boss_key: str, expected_discards_left: int):
    """Verify discards_left is set correctly for specific boss blinds."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    py_state.round_resets.blind_choices["Boss"] = boss_key
    py_state.round_resets.blind_states["Small"] = "Defeated"
    py_state.round_resets.blind_states["Big"] = "Defeated"
    py_state.round_resets.blind_states["Boss"] = "Select"
    py_state.blind_on_deck = "Boss"
    start_blind(py_state, "Boss")
    assert py_state.current_round.discards_left == expected_discards_left, (
        f"{boss_key}: expected discards_left={expected_discards_left}, "
        f"got {py_state.current_round.discards_left}"
    )


@pytest.mark.parametrize("boss_key,expected_hand_size", [
    ("bl_manacle", 7),  # reduces by 1 (default 8)
    ("bl_hook", 8),     # normal
    ("bl_water", 8),    # normal
])
def test_boss_blind_hand_size(boss_key: str, expected_hand_size: int):
    """Verify hand_size is set correctly for The Manacle and others."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    py_state.round_resets.blind_choices["Boss"] = boss_key
    py_state.round_resets.blind_states["Small"] = "Defeated"
    py_state.round_resets.blind_states["Big"] = "Defeated"
    py_state.round_resets.blind_states["Boss"] = "Select"
    py_state.blind_on_deck = "Boss"
    start_blind(py_state, "Boss")
    assert py_state.current_round.hand_size == expected_hand_size, (
        f"{boss_key}: expected hand_size={expected_hand_size}, "
        f"got {py_state.current_round.hand_size}"
    )
    assert len(py_state.hand_cards) == expected_hand_size, (
        f"{boss_key}: expected hand_cards count={expected_hand_size}, "
        f"got {len(py_state.hand_cards)}"
    )


@pytest.mark.parametrize("boss_key,expected_suit,expected_count", [
    ("bl_goad", "Spades", 13),
    ("bl_club", "Clubs", 13),
    ("bl_head", "Hearts", 13),
    ("bl_window", "Diamonds", 13),
])
def test_boss_blind_suit_debuff(boss_key: str, expected_suit: str, expected_count: int):
    """Verify suit-based debuffs debuff the correct cards.

    Also verifies Python and Lua agree on the debuffed card count.
    """
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    py_state.round_resets.blind_choices["Boss"] = boss_key
    py_state.round_resets.blind_states["Small"] = "Defeated"
    py_state.round_resets.blind_states["Big"] = "Defeated"
    py_state.round_resets.blind_states["Boss"] = "Select"
    py_state.blind_on_deck = "Boss"

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw.round_resets.blind_choices.Boss = boss_key
    lua_raw.round_resets.blind_states.Small = "Defeated"
    lua_raw.round_resets.blind_states.Big = "Defeated"
    lua_raw.round_resets.blind_states.Boss = "Select"
    lua_raw.blind_on_deck = "Boss"

    start_blind(py_state, "Boss")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Boss")

    # Check Python debuffs
    py_debuffed = [c for c in py_state.deck_cards if c.debuff]
    assert len(py_debuffed) == expected_count, (
        f"{boss_key}: expected {expected_count} debuffed cards, got {len(py_debuffed)}"
    )
    assert all(c.suit == expected_suit for c in py_debuffed), (
        f"{boss_key}: not all debuffed cards have suit {expected_suit}"
    )

    # Check Lua debuffs
    lua_dict = bridge.snapshot(lua_raw)
    lua_deck = lua_dict.get("deck_cards", [])
    lua_debuffed_count = sum(
        1 for c in lua_deck if isinstance(c, dict) and c.get("debuff")
    )
    assert lua_debuffed_count == expected_count, (
        f"{boss_key} Lua: expected {expected_count} debuffed cards, got {lua_debuffed_count}"
    )


def test_boss_blind_verdant_leaf_debuffs_all():
    """Verdant Leaf debuffs all 52 deck cards."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    py_state.round_resets.blind_choices["Boss"] = "bl_final_leaf"
    py_state.round_resets.blind_states["Small"] = "Defeated"
    py_state.round_resets.blind_states["Big"] = "Defeated"
    py_state.round_resets.blind_states["Boss"] = "Select"
    py_state.blind_on_deck = "Boss"

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw.round_resets.blind_choices.Boss = "bl_final_leaf"
    lua_raw.round_resets.blind_states.Small = "Defeated"
    lua_raw.round_resets.blind_states.Big = "Defeated"
    lua_raw.round_resets.blind_states.Boss = "Select"
    lua_raw.blind_on_deck = "Boss"

    start_blind(py_state, "Boss")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Boss")

    py_debuffed = sum(1 for c in py_state.deck_cards if c.debuff)
    assert py_debuffed == 52, f"Expected all 52 cards debuffed, got {py_debuffed}"

    lua_dict = bridge.snapshot(lua_raw)
    lua_deck = lua_dict.get("deck_cards", [])
    lua_debuffed = sum(1 for c in lua_deck if isinstance(c, dict) and c.get("debuff"))
    assert lua_debuffed == 52, f"Lua: expected all 52 cards debuffed, got {lua_debuffed}"


def test_boss_blind_plant_debuffs_faces():
    """The Plant (bl_plant) debuffs face cards (J, Q, K)."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    py_state.round_resets.blind_choices["Boss"] = "bl_plant"
    py_state.round_resets.blind_states["Small"] = "Defeated"
    py_state.round_resets.blind_states["Big"] = "Defeated"
    py_state.round_resets.blind_states["Boss"] = "Select"
    py_state.blind_on_deck = "Boss"

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw.round_resets.blind_choices.Boss = "bl_plant"
    lua_raw.round_resets.blind_states.Small = "Defeated"
    lua_raw.round_resets.blind_states.Big = "Defeated"
    lua_raw.round_resets.blind_states.Boss = "Select"
    lua_raw.blind_on_deck = "Boss"

    start_blind(py_state, "Boss")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Boss")

    # 4 suits * 3 face ranks (J, Q, K) = 12 face cards
    py_debuffed = [c for c in py_state.deck_cards if c.debuff]
    assert len(py_debuffed) == 12, f"Expected 12 face cards debuffed, got {len(py_debuffed)}"
    assert all(c.rank in {"J", "Q", "K"} for c in py_debuffed), (
        f"Non-face card debuffed: {[c.rank for c in py_debuffed if c.rank not in {'J','Q','K'}]}"
    )

    lua_dict = bridge.snapshot(lua_raw)
    lua_deck = lua_dict.get("deck_cards", [])
    lua_debuffed = sum(1 for c in lua_deck if isinstance(c, dict) and c.get("debuff"))
    assert lua_debuffed == 12, f"Lua: expected 12 face cards debuffed, got {lua_debuffed}"


def test_boss_blind_hook_discards_two_cards():
    """The Hook discards 2 random hand cards before scoring (no jokers, so blind triggers)."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    py_state.round_resets.blind_choices["Boss"] = "bl_hook"
    py_state.round_resets.blind_states["Small"] = "Defeated"
    py_state.round_resets.blind_states["Big"] = "Defeated"
    py_state.round_resets.blind_states["Boss"] = "Select"
    py_state.blind_on_deck = "Boss"

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw.round_resets.blind_choices.Boss = "bl_hook"
    lua_raw.round_resets.blind_states.Small = "Defeated"
    lua_raw.round_resets.blind_states.Big = "Defeated"
    lua_raw.round_resets.blind_states.Boss = "Select"
    lua_raw.blind_on_deck = "Boss"

    start_blind(py_state, "Boss")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Boss")

    initial_hand_count = len(py_state.hand_cards)
    assert initial_hand_count == 8

    play_cards(py_state, list(range(min(5, len(py_state.hand_cards)))))
    lua_raw = bridge.step(lua_raw, "play_hand", card_indices=[1, 2, 3, 4, 5])

    # After play: 5 cards played, 2 hooked (discarded), 1 remains in hand
    # hand_cards should be 1 (8 - 5 played - 2 hooked = 1)
    assert py_state.blind_triggered is True, "The Hook should trigger"
    assert py_state.current_round.discards_left == 4  # hook doesn't consume discards

    py_snap, lua_snap = _boss_blind_snapshot(py_state, bridge, lua_raw)
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], (
        "The Hook play_hand divergences:\n"
        + "\n".join(f"  {path}: py={py_val!r} lua={lua_val!r}" for path, py_val, lua_val in diffs)
    )


def test_boss_blind_tooth_costs_dollars():
    """The Tooth costs $1 per card played."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    py_state.round_resets.blind_choices["Boss"] = "bl_tooth"
    py_state.round_resets.blind_states["Small"] = "Defeated"
    py_state.round_resets.blind_states["Big"] = "Defeated"
    py_state.round_resets.blind_states["Boss"] = "Select"
    py_state.blind_on_deck = "Boss"

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw.round_resets.blind_choices.Boss = "bl_tooth"
    lua_raw.round_resets.blind_states.Small = "Defeated"
    lua_raw.round_resets.blind_states.Big = "Defeated"
    lua_raw.round_resets.blind_states.Boss = "Select"
    lua_raw.blind_on_deck = "Boss"

    start_blind(py_state, "Boss")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Boss")

    initial_dollars = py_state.dollars
    n_play = min(5, len(py_state.hand_cards))

    play_cards(py_state, list(range(n_play)))
    lua_raw = bridge.step(lua_raw, "play_hand", card_indices=list(range(1, n_play + 1)))

    expected_dollars = max(0, initial_dollars - n_play)
    assert py_state.dollars == expected_dollars, (
        f"The Tooth: expected ${expected_dollars}, got ${py_state.dollars}"
    )
    assert py_state.blind_triggered is True

    py_snap, lua_snap = _boss_blind_snapshot(py_state, bridge, lua_raw)
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], (
        "The Tooth play_hand divergences:\n"
        + "\n".join(f"  {path}: py={py_val!r} lua={lua_val!r}" for path, py_val, lua_val in diffs)
    )


def test_boss_blind_flint_halves_base_score():
    """The Flint halves base chips and mult before scoring."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    py_state.round_resets.blind_choices["Boss"] = "bl_flint"
    py_state.round_resets.blind_states["Small"] = "Defeated"
    py_state.round_resets.blind_states["Big"] = "Defeated"
    py_state.round_resets.blind_states["Boss"] = "Select"
    py_state.blind_on_deck = "Boss"

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw.round_resets.blind_choices.Boss = "bl_flint"
    lua_raw.round_resets.blind_states.Small = "Defeated"
    lua_raw.round_resets.blind_states.Big = "Defeated"
    lua_raw.round_resets.blind_states.Boss = "Select"
    lua_raw.blind_on_deck = "Boss"

    start_blind(py_state, "Boss")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Boss")

    play_cards(py_state, list(range(min(5, len(py_state.hand_cards)))))
    lua_raw = bridge.step(lua_raw, "play_hand", card_indices=[1, 2, 3, 4, 5])

    assert py_state.blind_triggered is True, "The Flint should trigger"

    py_snap, lua_snap = _boss_blind_snapshot(py_state, bridge, lua_raw)
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], (
        "The Flint play_hand divergences:\n"
        + "\n".join(f"  {path}: py={py_val!r} lua={lua_val!r}" for path, py_val, lua_val in diffs)
    )


def test_boss_blind_eye_tracks_hand_types():
    """The Eye debuffs repeated hand types - first play is always fine."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    py_state.round_resets.blind_choices["Boss"] = "bl_eye"
    py_state.round_resets.blind_states["Small"] = "Defeated"
    py_state.round_resets.blind_states["Big"] = "Defeated"
    py_state.round_resets.blind_states["Boss"] = "Select"
    py_state.blind_on_deck = "Boss"

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw.round_resets.blind_choices.Boss = "bl_eye"
    lua_raw.round_resets.blind_states.Small = "Defeated"
    lua_raw.round_resets.blind_states.Big = "Defeated"
    lua_raw.round_resets.blind_states.Boss = "Select"
    lua_raw.blind_on_deck = "Boss"

    start_blind(py_state, "Boss")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Boss")

    # The Eye tracks hand types - first hand should not trigger
    py_snap, lua_snap = _boss_blind_snapshot(py_state, bridge, lua_raw)
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], (
        "The Eye start_blind divergences:\n"
        + "\n".join(f"  {path}: py={py_val!r} lua={lua_val!r}" for path, py_val, lua_val in diffs)
    )

    play_cards(py_state, list(range(min(5, len(py_state.hand_cards)))))
    lua_raw = bridge.step(lua_raw, "play_hand", card_indices=[1, 2, 3, 4, 5])

    py_snap, lua_snap = _boss_blind_snapshot(py_state, bridge, lua_raw)
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], (
        "The Eye first play_hand divergences:\n"
        + "\n".join(f"  {path}: py={py_val!r} lua={lua_val!r}" for path, py_val, lua_val in diffs)
    )


def test_boss_blind_arm_delevels_hand():
    """The Arm de-levels the played hand type (only if level > 1)."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    py_state.round_resets.blind_choices["Boss"] = "bl_arm"
    py_state.round_resets.blind_states["Small"] = "Defeated"
    py_state.round_resets.blind_states["Big"] = "Defeated"
    py_state.round_resets.blind_states["Boss"] = "Select"
    py_state.blind_on_deck = "Boss"

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw.round_resets.blind_choices.Boss = "bl_arm"
    lua_raw.round_resets.blind_states.Small = "Defeated"
    lua_raw.round_resets.blind_states.Big = "Defeated"
    lua_raw.round_resets.blind_states.Boss = "Select"
    lua_raw.blind_on_deck = "Boss"

    start_blind(py_state, "Boss")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Boss")

    play_cards(py_state, list(range(min(5, len(py_state.hand_cards)))))
    lua_raw = bridge.step(lua_raw, "play_hand", card_indices=[1, 2, 3, 4, 5])

    # Level 1 hands won't trigger The Arm (can't go below level 1)
    # Just check parity
    py_snap, lua_snap = _boss_blind_snapshot(py_state, bridge, lua_raw)
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], (
        "The Arm play_hand divergences:\n"
        + "\n".join(f"  {path}: py={py_val!r} lua={lua_val!r}" for path, py_val, lua_val in diffs)
    )
