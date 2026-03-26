"""Oracle bridge: Python ↔ Lua oracle for parity testing.

Uses lupa (Lua 5.4) for table manipulation and game logic, but overrides
math.randomseed/math.random with a persistent LuaJIT state to match
Balatro's exact RNG behavior. Lua 5.4's PRNG differs from LuaJIT's,
so this hybrid approach is required.
"""

from __future__ import annotations

import ctypes
from math import isclose
from pathlib import Path
from typing import Any

from lupa import LuaRuntime, lua_type

from .luajit import _get_lua_lib

VENDOR_PATH = Path(__file__).resolve().parents[3] / "vendor" / "balatro_lua"
ORACLE_PATH = Path(__file__).resolve().parent / "oracle.lua"


class OracleError(Exception):
    """Raised when the Lua oracle encounters an error."""


class _LuaJITRNG:
    """Persistent LuaJIT state for exact Balatro-compatible RNG.

    Keeps a single LuaJIT state alive so math.randomseed/math.random
    maintain correct PRNG state across calls (no re-seeding overhead).
    """

    def __init__(self) -> None:
        self._lib = _get_lua_lib()
        self._state = self._lib.luaL_newstate()
        if not self._state:
            raise MemoryError("Failed to create LuaJIT state for RNG")
        self._lib.luaL_openlibs(self._state)

    def randomseed(self, seed: float) -> None:
        self._exec(f"math.randomseed({seed!r})")

    def random(self, a: float | None = None, b: float | None = None) -> float | int:
        if a is not None and b is not None:
            self._exec(f"return math.random({int(a)}, {int(b)})")
            return int(self._lib.lua_tonumber(self._state, -1))
        elif a is not None:
            self._exec(f"return math.random({int(a)})")
            return int(self._lib.lua_tonumber(self._state, -1))
        else:
            self._exec("return math.random()")
            return float(self._lib.lua_tonumber(self._state, -1))

    def _exec(self, script: str) -> None:
        encoded = script.encode("utf-8")
        if self._lib.luaL_loadstring(self._state, encoded) != 0:
            raise OracleError(self._error())
        if self._lib.lua_pcall(self._state, 0, 1, 0) != 0:
            raise OracleError(self._error())

    def _error(self) -> str:
        size = ctypes.c_size_t()
        raw = self._lib.lua_tolstring(self._state, -1, ctypes.byref(size))
        return ctypes.string_at(raw, size.value).decode("utf-8") if raw else "unknown LuaJIT error"

    def close(self) -> None:
        if self._state:
            self._lib.lua_close(self._state)
            self._state = None

    def __del__(self) -> None:
        self.close()


# Argument signatures for oracle functions (positional after state)
_SIGNATURES: dict[str, list[str]] = {
    "start_blind": ["blind_type"],
    "play_hand": ["card_indices"],
    "discard": ["card_indices"],
    "cash_out": [],
    "populate_shop": [],
    "buy_card": ["index"],
    "use_consumable": ["index", "targets"],
    "open_pack": ["index"],
    "reroll_shop": [],
    "skip_blind": [],
    "finish_shop": [],
    "defeat_blind": [],
}


class OracleBridge:
    """Bridge between Python and the Lua oracle.

    Creates a lupa runtime with LuaJIT-backed RNG, loads oracle.lua,
    and provides methods to create runs, step through actions, and
    extract snapshots for comparison.
    """

    def __init__(self) -> None:
        self._rng = _LuaJITRNG()
        self.runtime = LuaRuntime(unpack_returned_tuples=True)
        self._inject_rng()
        self.runtime.execute(f'VENDOR_PATH = "{VENDOR_PATH}"')
        if ORACLE_PATH.exists():
            self.runtime.execute(ORACLE_PATH.read_text())
        self._oracle = self.runtime.globals()["oracle"]

    def _inject_rng(self) -> None:
        """Override lupa's math.randomseed/math.random with LuaJIT-backed versions."""
        rng = self._rng

        def py_randomseed(seed: float) -> None:
            rng.randomseed(seed)

        def py_random(a: float | None = None, b: float | None = None) -> float | int:
            return rng.random(a, b)

        g = self.runtime.globals()
        g["_luajit_randomseed"] = py_randomseed
        g["_luajit_random"] = py_random
        self.runtime.execute(
            """
            math.randomseed = function(seed)
                _luajit_randomseed(seed)
            end
            math.random = function(a, b)
                return _luajit_random(a, b)
            end
            """
        )

    def create_run(self, seed: str, *, stake: int = 1, deck_key: str = "b_red") -> Any:
        """Create initial run state. Returns raw Lua table."""
        if self._oracle is None:
            raise OracleError("oracle.lua not loaded")
        return self._oracle.create_run(seed, stake, deck_key)

    def step(self, lua_state: Any, action: str, **kwargs: Any) -> Any:
        """Execute an action on the Lua state. Returns raw Lua table."""
        if self._oracle is None:
            raise OracleError("oracle.lua not loaded")
        fn = getattr(self._oracle, action, None)
        if fn is None:
            raise OracleError(f"Unknown oracle action: {action}")
        sig = _SIGNATURES.get(action, [])
        args = [lua_state] + [kwargs[k] for k in sig if k in kwargs]
        try:
            return fn(*args)
        except Exception as e:
            raise OracleError(f"Oracle error in {action}: {e}") from e

    def snapshot(self, lua_state: Any) -> dict[str, Any]:
        """Convert a raw Lua table to a Python dict for comparison."""
        return self._to_python(lua_state)

    def _to_python(self, value: Any) -> Any:
        vtype = lua_type(value)
        if vtype != "table":
            return value
        keys = list(value.keys())
        if self._is_array(keys):
            return [self._to_python(value[i]) for i in range(1, len(keys) + 1)]
        return {str(k): self._to_python(value[k]) for k in keys}

    @staticmethod
    def _is_array(keys: list[Any]) -> bool:
        if not keys:
            return True
        if not all(isinstance(k, (int, float)) and int(k) == k for k in keys):
            return False
        ints = sorted(int(k) for k in keys)
        return ints == list(range(1, len(ints) + 1))

    def close(self) -> None:
        self._rng.close()

    def __del__(self) -> None:
        self.close()


def snapshot_from_run_state(state: Any) -> dict[str, Any]:
    """Extract comparable snapshot from a Python RunState."""

    def _card(c: Any) -> dict[str, str]:
        return {"front_key": c.front_key, "center_key": c.center_key, "suit": c.suit}

    def _joker(j: Any) -> dict[str, Any]:
        return {
            "center_key": j.center_key,
            "mult": getattr(j, "mult", 0),
            "x_mult": getattr(j, "x_mult", 0),
            "extra": getattr(j, "extra", None),
        }

    return {
        "dollars": state.dollars,
        "ante": state.round_resets.ante,
        "round": state.round,
        "hands_played": state.hands_played,
        "hands_left": state.current_round.hands_left,
        "discards_left": state.current_round.discards_left,
        "deck_cards_count": len(state.deck_cards),
        "deck_cards": [_card(c) for c in state.deck_cards],
        "draw_pile_count": len(state.draw_pile),
        "draw_pile_keys": [c.front_key for c in state.draw_pile],
        "hand_cards_count": len(state.hand_cards),
        "hand_cards_keys": [c.front_key for c in state.hand_cards],
        "discard_pile_count": len(state.discard_pile),
        "jokers": [_joker(j) for j in state.jokers],
        "consumable_keys": [c.center_key for c in state.consumables],
        "hand_levels": {
            name: {"level": int(hand["level"]), "played": int(hand.get("played", 0))}
            for name, hand in state.hands.items()
        },
        "blind_disabled": state.blind_disabled,
        "blind_triggered": state.blind_triggered,
        "blind_states": dict(state.round_resets.blind_states),
        "blind_on_deck": getattr(state, "blind_on_deck", "Small"),
        "shop_cards": [{"center_key": c.center_key, "cost": c.cost} for c in state.shop.cards],
        "shop_vouchers": [c.center_key for c in state.shop.vouchers],
        "shop_boosters": [c.center_key for c in state.shop.boosters],
    }


def snapshot_from_lua_state(lua_state: dict[str, Any]) -> dict[str, Any]:
    """Extract comparable snapshot from a Lua oracle state (Python dict)."""
    deck = lua_state.get("deck_cards", [])
    draw = lua_state.get("draw_pile", [])
    hand = lua_state.get("hand_cards", [])
    discard = lua_state.get("discard_pile", [])
    jokers = lua_state.get("jokers", [])
    consumables = lua_state.get("consumables", [])
    hands = lua_state.get("hands", {})
    cr = lua_state.get("current_round", {})
    rr = lua_state.get("round_resets", {})

    def _safe(d: Any, k: str, default: Any = 0) -> Any:
        return d.get(k, default) if isinstance(d, dict) else default

    def _lua_card(c: dict[str, Any]) -> dict[str, str]:
        return {"front_key": c.get("front_key", ""), "center_key": c.get("center_key", ""), "suit": c.get("suit", "")}

    def _lua_joker(j: dict[str, Any]) -> dict[str, Any]:
        return {
            "center_key": j.get("center_key", ""),
            "mult": j.get("mult", 0),
            "x_mult": j.get("x_mult", 0),
            "extra": j.get("extra"),
        }

    return {
        "dollars": int(lua_state.get("dollars", 0)),
        "ante": int(_safe(rr, "ante", 1)),
        "round": int(lua_state.get("round", 0)),
        "hands_played": int(lua_state.get("hands_played", 0)),
        "hands_left": int(_safe(cr, "hands_left", 0)),
        "discards_left": int(_safe(cr, "discards_left", 0)),
        "deck_cards_count": len(deck) if isinstance(deck, list) else 0,
        "deck_cards": [_lua_card(c) for c in deck] if isinstance(deck, list) else [],
        "draw_pile_count": len(draw) if isinstance(draw, list) else 0,
        "draw_pile_keys": [c.get("front_key", "") for c in draw] if isinstance(draw, list) else [],
        "hand_cards_count": len(hand) if isinstance(hand, list) else 0,
        "hand_cards_keys": [c.get("front_key", "") for c in hand] if isinstance(hand, list) else [],
        "discard_pile_count": len(discard) if isinstance(discard, list) else 0,
        "jokers": [_lua_joker(j) for j in jokers] if isinstance(jokers, list) else [],
        "consumable_keys": [c.get("center_key", "") for c in consumables] if isinstance(consumables, list) else [],
        "hand_levels": {
            name: {"level": int(_safe(h, "level", 1)), "played": int(_safe(h, "played", 0))}
            for name, h in (hands.items() if isinstance(hands, dict) else [])
        },
        "blind_disabled": bool(lua_state.get("blind_disabled", False)),
        "blind_triggered": bool(lua_state.get("blind_triggered", False)),
        "blind_states": {k: str(v) for k, v in (_safe(rr, "blind_states", {}) if isinstance(_safe(rr, "blind_states", {}), dict) else {}).items()},
        "blind_on_deck": str(lua_state.get("blind_on_deck", "Small")),
        "shop_cards": [
            {"center_key": c.get("center_key", "") if isinstance(c, dict) else "",
             "cost": int(c.get("cost", 0)) if isinstance(c, dict) else 0}
            for c in (lua_state.get("shop", {}).get("cards", []) if isinstance(lua_state.get("shop"), dict) else [])
        ] if isinstance(lua_state.get("shop", {}).get("cards", []) if isinstance(lua_state.get("shop"), dict) else [], list) else [],
        "shop_vouchers": [
            c.get("center_key", "") if isinstance(c, dict) else ""
            for c in (lua_state.get("shop", {}).get("vouchers", []) if isinstance(lua_state.get("shop"), dict) else [])
        ] if isinstance(lua_state.get("shop", {}).get("vouchers", []) if isinstance(lua_state.get("shop"), dict) else [], list) else [],
        "shop_boosters": [
            c.get("center_key", "") if isinstance(c, dict) else ""
            for c in (lua_state.get("shop", {}).get("boosters", []) if isinstance(lua_state.get("shop"), dict) else [])
        ] if isinstance(lua_state.get("shop", {}).get("boosters", []) if isinstance(lua_state.get("shop"), dict) else [], list) else [],
    }


def diff_snapshots(py_snap: dict[str, Any], lua_snap: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """Compare two snapshots. Returns list of (path, python_value, lua_value) divergences."""
    diffs: list[tuple[str, Any, Any]] = []
    all_keys = sorted(set(py_snap.keys()) | set(lua_snap.keys()))
    for key in all_keys:
        py_val = py_snap.get(key)
        lua_val = lua_snap.get(key)
        if isinstance(py_val, float) and isinstance(lua_val, float):
            if not isclose(py_val, lua_val, rel_tol=1e-9):
                diffs.append((key, py_val, lua_val))
        elif isinstance(py_val, list) and isinstance(lua_val, list):
            if len(py_val) != len(lua_val):
                diffs.append((f"{key}.length", len(py_val), len(lua_val)))
            else:
                for i, (pv, lv) in enumerate(zip(py_val, lua_val)):
                    if isinstance(pv, dict) and isinstance(lv, dict):
                        sub = diff_snapshots(pv, lv)
                        for path, a, b in sub:
                            diffs.append((f"{key}[{i}].{path}", a, b))
                    elif pv != lv:
                        diffs.append((f"{key}[{i}]", pv, lv))
        elif isinstance(py_val, dict) and isinstance(lua_val, dict):
            sub = diff_snapshots(py_val, lua_val)
            for path, a, b in sub:
                diffs.append((f"{key}.{path}", a, b))
        elif py_val != lua_val:
            diffs.append((key, py_val, lua_val))
    return diffs
