from pathlib import Path

UPSTREAM_ROOT = Path(__file__).resolve().parents[3] / "vendor" / "balatro_lua"


def get_lua_bridge():  # type: ignore[no-untyped-def]
    """Lazy import — only available when lupa is installed (parity testing)."""
    from .lua import LuaBridge, get_lua_bridge as _get

    return _get()


def get_luajit_bridge():  # type: ignore[no-untyped-def]
    """Lazy import — only available when LuaJIT dylib is present (parity testing)."""
    from .luajit import LuaJITBridge, get_luajit_bridge as _get

    return _get()
