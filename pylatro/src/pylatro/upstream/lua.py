from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from lupa import LuaRuntime, lua_type

UPSTREAM_ROOT = Path(__file__).resolve().parents[3] / "vendor" / "balatro_lua"


class LuaBridge:
    """Small LuaJIT bridge for Balatro-compatible random calls and table eval."""

    def __init__(self) -> None:
        self.runtime = LuaRuntime(unpack_returned_tuples=True)
        self.runtime.execute(
            """
            HEX = function(v) return v end
            localize = function(v)
              if type(v) == 'table' then
                return v.key or v[1] or ''
              end
              return v
            end

            function _pylatro_random(seed, min, max)
              math.randomseed(seed)
              if min ~= nil and max ~= nil then
                return math.random(min, max)
              end
              return math.random()
            end

            function _pylatro_random_string(length, seed)
              if seed ~= nil then math.randomseed(seed) end
              local ret = ''
              for i = 1, length do
                ret = ret..string.char(
                  math.random() > 0.7
                    and math.random(string.byte('1'), string.byte('9'))
                    or (math.random() > 0.45
                      and math.random(string.byte('A'), string.byte('N'))
                      or math.random(string.byte('P'), string.byte('Z')))
                )
              end
              return string.upper(ret)
            end

            function _pylatro_shuffle_indices(length, seed)
              local out = {}
              for i = 1, length do
                out[i] = i
              end
              if seed ~= nil then math.randomseed(seed) end
              for i = length, 2, -1 do
                local j = math.random(i)
                out[i], out[j] = out[j], out[i]
              end
              return out
            end
            """
        )

    def random(self, seed: float, minimum: int | None = None, maximum: int | None = None) -> float | int:
        return self.runtime.globals()["_pylatro_random"](seed, minimum, maximum)

    def random_string(self, length: int, seed: float) -> str:
        return self.runtime.globals()["_pylatro_random_string"](length, seed)

    def shuffle_indices(self, length: int, seed: float) -> list[int]:
        return self.to_python(self.runtime.globals()["_pylatro_shuffle_indices"](length, seed))

    def eval_table(self, source: str) -> Any:
        return self.runtime.eval(f"function() return {source} end")()

    def to_python(self, value: Any) -> Any:
        value_type = lua_type(value)
        if value_type != "table":
            return value

        keys = list(value.keys())
        if self._is_array(keys):
            return [self.to_python(value[index]) for index in range(1, len(keys) + 1)]

        return {key: self.to_python(value[key]) for key in self._sorted_keys(keys)}

    @staticmethod
    def _is_array(keys: list[Any]) -> bool:
        if not keys:
            return True
        if not all(isinstance(key, (int, float)) and int(key) == key for key in keys):
            return False
        ints = sorted(int(key) for key in keys)
        return ints == list(range(1, len(ints) + 1))

    @staticmethod
    def _sorted_keys(keys: list[Any]) -> list[Any]:
        return sorted(keys, key=lambda key: (str(type(key)), key))


@lru_cache(maxsize=1)
def get_lua_bridge() -> LuaBridge:
    return LuaBridge()
