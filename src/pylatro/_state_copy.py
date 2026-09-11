"""Copy routines shared by Python state models and their optional Cython build."""

from copy import deepcopy
from typing import Any

_COPY_ATOMIC_TYPES = frozenset((str, int, float, bool, type(None)))


def _copy_state_value(value: Any, memo: dict[int, Any]) -> Any:
    """Copy state containers without dispatching deepcopy for every scalar.

    Exact builtin types only: subclasses and custom objects retain their
    deepcopy hooks. Register containers before descending to preserve aliases
    and cycles, including references back to the containing RunState.
    """
    kind = type(value)
    if kind in _COPY_ATOMIC_TYPES:
        return value
    if kind is not dict and kind is not list:
        return deepcopy(value, memo)
    identity = id(value)
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



def copy_state_object(state: Any, memo: dict[int, Any]) -> Any:
    clone = object.__new__(type(state))
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
