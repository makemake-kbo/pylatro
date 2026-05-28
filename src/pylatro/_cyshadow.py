"""No-op stand-in for Cython's pure-Python-mode magic module.

The engine modules carry ``@cython.locals(...)`` / ``cython.declare(...)`` hints so
they compile to C scalars (see ``scripts/compile_cython.py``).  Those hints need a
``cython`` module to be importable even when the code runs *uncompiled*.  When the
real Cython package is installed (the ``dev`` and ``agent`` dependency groups) its
own shadow module provides everything; this fallback only kicks in for a
dependency-free core install (``pip install pylatro`` with no extras), keeping the
engine's ``dependencies = []`` contract intact.

Every attribute resolves to an opaque sentinel that behaves as both a type marker
(``cython.double``) and a no-op decorator/decorator-factory (``@cython.ccall``,
``@cython.locals(...)``).  ``declare(type, value)`` returns the value unchanged.
"""

from __future__ import annotations

from typing import Any


def _identity(func: Any) -> Any:
    return func


class _CythonShadow:
    def __getattr__(self, name: str) -> _CythonShadow:
        # Type markers: cython.int, cython.double, cython.Py_ssize_t, ...
        return self

    def __getitem__(self, item: Any) -> _CythonShadow:
        # Memoryview syntax: cython.double[:], cython.int[:, :]
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Bare decorator: @cython.ccall / @cython.cfunc  -> return the function
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        # Decorator factory: @cython.locals(...) / @cython.exceptval(...)
        return _identity

    @staticmethod
    def declare(_type: Any = None, value: Any = None, *_args: Any, **_kwargs: Any) -> Any:
        return value

    @staticmethod
    def locals(*_args: Any, **_kwargs: Any) -> Any:
        return _identity

    @staticmethod
    def cast(_type: Any, value: Any, *_args: Any, **_kwargs: Any) -> Any:
        return value


cython = _CythonShadow()
