"""Copy routines shared by Python state models and their optional Cython build.

``copy_state_object`` is the ``__deepcopy__`` body of every state dataclass. The
reflective approach (``getattr``/``setattr`` over ``__dataclass_fields__`` per
object) dominated profiles: one probe copy visits ~30 objects, and a whole game
makes tens of thousands of them. Each dataclass therefore gets a generated
straight-line copier on first use, with fields annotated as scalars assigned
directly and every other field routed through ``_copy_state_value``.
"""

from copy import deepcopy
from typing import Any

_COPY_ATOMIC_TYPES = frozenset((str, int, float, bool, type(None)))
_ATOMIC_ANNOTATION_NAMES = frozenset(("int", "str", "float", "bool", "None"))
_new = object.__new__

# type -> generated copier ``(obj, memo) -> clone``; False marks types that need
# the reflective fallback (instances carrying a ``__dict__``).
_COPIERS: dict[type, Any] = {}


def _scalar_annotation(annotation: Any) -> bool:
    """True when every alternative of ``annotation`` is an immutable scalar.

    Annotations are strings (``from __future__ import annotations``); anything
    unrecognised is treated as a container and copied by value.
    """
    if not isinstance(annotation, str):
        return annotation in _COPY_ATOMIC_TYPES
    return {part.strip() for part in annotation.split("|")} <= _ATOMIC_ANNOTATION_NAMES


def _build_copier(cls: type) -> Any:
    fields = getattr(cls, "__dataclass_fields__", None)
    if fields is None or cls.__dictoffset__:
        return False
    lines = [
        "def copier(obj, memo):",
        "    new = _new(cls)",
        "    memo[id(obj)] = new",
    ]
    for name, spec in fields.items():
        if not name.isidentifier():
            return False
        if _scalar_annotation(spec.type):
            lines.append(f"    new.{name} = obj.{name}")
        else:
            lines.append(f"    v = obj.{name}")
            lines.append(f"    new.{name} = v if type(v) in _ATOMIC else _copy(v, memo)")
    lines.append("    return new")
    namespace: dict[str, Any] = {
        "_new": _new, "cls": cls, "_ATOMIC": _COPY_ATOMIC_TYPES, "_copy": _copy_state_value,
    }
    exec("\n".join(lines), namespace)  # noqa: S102 - source is built from dataclass field names only
    return namespace["copier"]


def _copy_state_value(value: Any, memo: dict[int, Any]) -> Any:
    """Copy state containers without dispatching deepcopy for every scalar.

    Exact builtin types only: subclasses and custom objects retain their
    deepcopy hooks. Register containers before descending to preserve aliases
    and cycles, including references back to the containing RunState.
    """
    kind = type(value)
    if kind in _COPY_ATOMIC_TYPES:
        return value
    identity = id(value)
    if kind is dict or kind is list:
        if identity in memo:
            return memo[identity]
        clone: Any = {} if kind is dict else []
        memo[identity] = clone
        # Match deepcopy's lifetime guarantee when callers reuse a memo.
        memo.setdefault(id(memo), []).append(value)
        if kind is dict:
            for key, item in value.items():
                copied_item = item if type(item) in _COPY_ATOMIC_TYPES else _copy_state_value(item, memo)
                copied_key = key if type(key) in _COPY_ATOMIC_TYPES else _copy_state_value(key, memo)
                clone[copied_key] = copied_item
        else:
            clone.extend(item if type(item) in _COPY_ATOMIC_TYPES else _copy_state_value(item, memo) for item in value)
        return clone
    copier = _COPIERS.get(kind)
    if copier:
        # A state dataclass that already has a generated copier: skip the
        # copy.deepcopy dispatch, but honour the caller's memo exactly as it would.
        cached = memo.get(identity)
        if cached is not None:
            return cached
        clone = copier(value, memo)
        memo.setdefault(id(memo), []).append(value)
        return clone
    if kind is tuple or kind is frozenset:
        cached = memo.get(identity)
        if cached is not None:
            return cached
        # Immutable: sharing is indistinguishable from deepcopy as long as no
        # element needs copying, which is the case for every scalar tuple.
        for item in value:
            if type(item) not in _COPY_ATOMIC_TYPES:
                break
        else:
            return value
    return deepcopy(value, memo)


def copy_state_object(state: Any, memo: dict[int, Any]) -> Any:
    kind = type(state)
    copier = _COPIERS.get(kind)
    if copier is None:
        copier = _COPIERS[kind] = _build_copier(kind)
    if copier:
        return copier(state, memo)
    clone = object.__new__(kind)
    memo[id(state)] = clone
    for name in state.__dataclass_fields__:
        value = getattr(state, name)
        if type(value) not in _COPY_ATOMIC_TYPES:
            value = _copy_state_value(value, memo)
        setattr(clone, name, value)
    attributes = getattr(state, "__dict__", None)
    if attributes is not None:
        clone.__dict__.update(_copy_state_value(attributes, memo))
    return clone
