from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path

_DEFAULT_LUAJIT_DYLIB = Path(
    "/Users/makemake/Library/Application Support/Steam/steamapps/common/"
    "Balatro/Balatro.app/Contents/Frameworks/Lua.framework/Versions/A/Lua"
)


@lru_cache(maxsize=1)
def _get_lua_lib() -> ctypes.CDLL:
    path = os.environ.get("PYLATRO_LUAJIT_PATH") or str(_DEFAULT_LUAJIT_DYLIB)
    lib = ctypes.CDLL(path)
    lib.luaL_newstate.restype = ctypes.c_void_p
    lib.luaL_openlibs.argtypes = [ctypes.c_void_p]
    lib.luaL_loadstring.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.luaL_loadstring.restype = ctypes.c_int
    lib.lua_pcall.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.lua_pcall.restype = ctypes.c_int
    lib.lua_close.argtypes = [ctypes.c_void_p]
    lib.lua_tonumber.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.lua_tonumber.restype = ctypes.c_double
    lib.lua_tointeger.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.lua_tointeger.restype = ctypes.c_longlong
    lib.lua_tolstring.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)]
    lib.lua_tolstring.restype = ctypes.c_char_p
    lib.lua_settop.argtypes = [ctypes.c_void_p, ctypes.c_int]
    return lib


class LuaJITBridge:
    """Thin bridge into Balatro's bundled LuaJIT for exact RNG behavior."""

    def random(
        self,
        seed: float,
        minimum: int | None = None,
        maximum: int | None = None,
        *,
        draws_before: int = 0,
    ) -> float | int:
        if minimum is None or maximum is None:
            script = f"""
            math.randomseed({seed!r})
            for i = 1, {draws_before} do math.random() end
            return math.random()
            """
            return self._eval_number(script)
        script = f"""
        math.randomseed({seed!r})
        for i = 1, {draws_before} do math.random() end
        return math.random({minimum}, {maximum})
        """
        return self._eval_integer(script)

    def random_string(self, length: int, seed: float, *, draws_before: int = 0) -> tuple[str, int]:
        script = f"""
        math.randomseed({seed!r})
        for i = 1, {draws_before} do math.random() end
        local ret = ''
        local count = 0
        local function draw(...)
          count = count + 1
          return math.random(...)
        end
        for i = 1, {length} do
          ret = ret..string.char(
            draw() > 0.7
              and draw(string.byte('1'), string.byte('9'))
              or (draw() > 0.45
                and draw(string.byte('A'), string.byte('N'))
                or draw(string.byte('P'), string.byte('Z')))
          )
        end
        return string.upper(ret)..','..count
        """
        result = self._eval_string(script)
        text, count = result.rsplit(",", maxsplit=1)
        return text, int(count)

    def shuffle_indices(self, length: int, seed: float, *, draws_before: int = 0) -> list[int]:
        script = f"""
        math.randomseed({seed!r})
        for i = 1, {draws_before} do math.random() end
        local out = {{}}
        for i = 1, {length} do
          out[i] = i
        end
        for i = {length}, 2, -1 do
          local j = math.random(i)
          out[i], out[j] = out[j], out[i]
        end
        return table.concat(out, ',')
        """
        result = self._eval_string(script)
        return [int(part) for part in result.split(",")] if result else []

    def _eval_number(self, script: str) -> float:
        lua = _get_lua_lib()
        state = self._new_state(lua)
        try:
            self._execute(lua, state, script)
            return float(lua.lua_tonumber(state, -1))
        finally:
            lua.lua_close(state)

    def _eval_integer(self, script: str) -> int:
        lua = _get_lua_lib()
        state = self._new_state(lua)
        try:
            self._execute(lua, state, script)
            return int(lua.lua_tointeger(state, -1))
        finally:
            lua.lua_close(state)

    def _eval_string(self, script: str) -> str:
        lua = _get_lua_lib()
        state = self._new_state(lua)
        try:
            self._execute(lua, state, script)
            size = ctypes.c_size_t()
            raw = lua.lua_tolstring(state, -1, ctypes.byref(size))
            return ctypes.string_at(raw, size.value).decode("utf-8")
        finally:
            lua.lua_close(state)

    @staticmethod
    def _new_state(lua: ctypes.CDLL) -> ctypes.c_void_p:
        state = lua.luaL_newstate()
        if not state:
            raise MemoryError("Failed to create LuaJIT state")
        lua.luaL_openlibs(state)
        return state

    @staticmethod
    def _error_string(lua: ctypes.CDLL, state: ctypes.c_void_p) -> str:
        size = ctypes.c_size_t()
        raw = lua.lua_tolstring(state, -1, ctypes.byref(size))
        return ctypes.string_at(raw, size.value).decode("utf-8") if raw else "unknown LuaJIT error"

    def _execute(self, lua: ctypes.CDLL, state: ctypes.c_void_p, script: str) -> None:
        encoded = script.encode("utf-8")
        if lua.luaL_loadstring(state, encoded) != 0:
            raise RuntimeError(self._error_string(lua, state))
        if lua.lua_pcall(state, 0, 1, 0) != 0:
            raise RuntimeError(self._error_string(lua, state))


@lru_cache(maxsize=1)
def get_luajit_bridge() -> LuaJITBridge:
    return LuaJITBridge()
