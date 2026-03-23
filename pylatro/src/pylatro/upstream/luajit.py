from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path


LUAJIT_DYLIB = Path(
    "/Users/makemake/Library/Application Support/Steam/steamapps/common/Balatro/Balatro.app/Contents/Frameworks/Lua.framework/Versions/A/Lua"
)

_lua = ctypes.CDLL(str(LUAJIT_DYLIB))
_lua.luaL_newstate.restype = ctypes.c_void_p
_lua.luaL_openlibs.argtypes = [ctypes.c_void_p]
_lua.luaL_loadstring.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lua.luaL_loadstring.restype = ctypes.c_int
_lua.lua_pcall.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lua.lua_pcall.restype = ctypes.c_int
_lua.lua_close.argtypes = [ctypes.c_void_p]
_lua.lua_tonumber.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lua.lua_tonumber.restype = ctypes.c_double
_lua.lua_tointeger.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lua.lua_tointeger.restype = ctypes.c_longlong
_lua.lua_tolstring.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)]
_lua.lua_tolstring.restype = ctypes.c_char_p
_lua.lua_settop.argtypes = [ctypes.c_void_p, ctypes.c_int]


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
        state = self._new_state()
        try:
            self._execute(state, script)
            return float(_lua.lua_tonumber(state, -1))
        finally:
            _lua.lua_close(state)

    def _eval_integer(self, script: str) -> int:
        state = self._new_state()
        try:
            self._execute(state, script)
            return int(_lua.lua_tointeger(state, -1))
        finally:
            _lua.lua_close(state)

    def _eval_string(self, script: str) -> str:
        state = self._new_state()
        try:
            self._execute(state, script)
            size = ctypes.c_size_t()
            raw = _lua.lua_tolstring(state, -1, ctypes.byref(size))
            return ctypes.string_at(raw, size.value).decode("utf-8")
        finally:
            _lua.lua_close(state)

    @staticmethod
    def _new_state() -> ctypes.c_void_p:
        state = _lua.luaL_newstate()
        if not state:
            raise MemoryError("Failed to create LuaJIT state")
        _lua.luaL_openlibs(state)
        return state

    @staticmethod
    def _error_string(state: ctypes.c_void_p) -> str:
        size = ctypes.c_size_t()
        raw = _lua.lua_tolstring(state, -1, ctypes.byref(size))
        return ctypes.string_at(raw, size.value).decode("utf-8") if raw else "unknown LuaJIT error"

    def _execute(self, state: ctypes.c_void_p, script: str) -> None:
        encoded = script.encode("utf-8")
        if _lua.luaL_loadstring(state, encoded) != 0:
            raise RuntimeError(self._error_string(state))
        if _lua.lua_pcall(state, 0, 1, 0) != 0:
            raise RuntimeError(self._error_string(state))


@lru_cache(maxsize=1)
def get_luajit_bridge() -> LuaJITBridge:
    return LuaJITBridge()
