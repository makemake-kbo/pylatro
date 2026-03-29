from math import isclose

from pylatro.rng import PseudorandomState, pseudohash


def test_pseudohash_and_pseudoseed_match_reference_fixture() -> None:
    state = PseudorandomState("AAAAAAAA")

    assert isclose(pseudohash("AAAAAAAA"), 0.43257138351543745)
    assert isclose(state.pseudoseed("boss"), 0.5374444774613187)
    assert isclose(state.pseudoseed("Voucher1"), 0.5290234269488188)
    assert isclose(state.pseudoseed("Tag1"), 0.2909399952510187)
    assert isclose(state.pseudoseed("shuffle"), 0.6564266916414188)
    assert isclose(state.pseudoseed("rarity1"), 0.4462461409768187)


def test_repeated_pseudoseed_progresses_state() -> None:
    state = PseudorandomState("AAAAAAAA")
    first = state.pseudoseed("boss")
    second = state.pseudoseed("boss")
    third = state.pseudoseed("boss")

    assert isclose(first, 0.5374444774613187)
    assert isclose(second, 0.3372904636163687)
    assert isclose(third, 0.49216242764296875)
    assert state.random_string(8, second) == "CQ5IQGBQ"
    assert state.random_string(8, third) == "NYVU6UGQ"


import pytest


@pytest.mark.xfail(
    reason="lupa uses Lua 5.4 which rejects float seeds in math.randomseed. "
    "Oracle must use LuaJIT ctypes bridge instead.",
    strict=True,
)
def test_lupa_rng_matches_luajit():
    """Go/no-go gate: lupa's math.random vs LuaJIT for the same float seeds.

    Result: FAIL — Lua 5.4 (lupa) requires integer seeds while Balatro's
    LuaJIT accepts floats. The oracle uses LuaJIT ctypes bridge instead.
    """
    from pylatro.upstream.lua import get_lua_bridge
    from pylatro.upstream.luajit import get_luajit_bridge

    lupa_bridge = get_lua_bridge()
    luajit_bridge = get_luajit_bridge()

    seeds = [
        pseudohash("AAAAAAAA"),
        pseudohash("BBBBBBBB"),
        pseudohash("12345678"),
    ]

    for seed in seeds:
        lupa_val = lupa_bridge.random(seed)
        luajit_val = luajit_bridge.random(seed)
        assert isclose(lupa_val, luajit_val, rel_tol=1e-12), (
            f"RNG divergence for seed {seed}: lupa={lupa_val}, luajit={luajit_val}"
        )
