"""Strict, dependency-free models for the live bridge protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

PROTOCOL_VERSION = 1
SUPPORTED_BALATRO_VERSION = "1.0.1o-FULL"
SUPPORTED_PHASES = frozenset({"blind_select", "hand_play", "shop", "booster_pack", "terminal"})


class ProtocolError(ValueError):
    """A request cannot be safely interpreted."""


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{field} must be an object")
    return {str(key): item for key, item in value.items()}


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    protocol_version: int
    session_id: str
    decision_id: int
    state_fingerprint: str
    phase: str
    versions: dict[str, Any]
    state: dict[str, Any]
    legal: dict[str, Any]
    previous_action: dict[str, Any] | None = None

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "protocol_version",
            "session_id",
            "decision_id",
            "state_fingerprint",
            "phase",
            "versions",
            "state",
            "legal",
            "previous_action",
        }
    )

    @classmethod
    def from_dict(cls, raw: object) -> DecisionRequest:
        obj = _mapping(raw, "request")
        unknown = set(obj) - cls._FIELDS
        missing = cls._FIELDS - {"previous_action"} - set(obj)
        if unknown:
            raise ProtocolError(f"unknown request fields: {', '.join(sorted(unknown))}")
        if missing:
            raise ProtocolError(f"missing request fields: {', '.join(sorted(missing))}")

        version = obj["protocol_version"]
        if type(version) is not int or version != PROTOCOL_VERSION:
            raise ProtocolError(f"unsupported protocol_version {version!r}; expected {PROTOCOL_VERSION}")
        decision_id = obj["decision_id"]
        if type(decision_id) is not int or decision_id < 0:
            raise ProtocolError("decision_id must be a non-negative integer")
        phase = _nonempty_string(obj["phase"], "phase")
        if phase not in SUPPORTED_PHASES:
            raise ProtocolError(f"unsupported phase {phase!r}")
        previous = obj.get("previous_action")
        if previous is not None:
            previous = _mapping(previous, "previous_action")
        versions = _mapping(obj["versions"], "versions")
        balatro_version = versions.get("balatro")
        if balatro_version != SUPPORTED_BALATRO_VERSION:
            raise ProtocolError(
                f"unsupported Balatro version {balatro_version!r}; "
                f"expected {SUPPORTED_BALATRO_VERSION}"
            )
        return cls(
            protocol_version=version,
            session_id=_nonempty_string(obj["session_id"], "session_id"),
            decision_id=decision_id,
            state_fingerprint=_nonempty_string(obj["state_fingerprint"], "state_fingerprint"),
            phase=phase,
            versions=versions,
            state=_mapping(obj["state"], "state"),
            legal=_mapping(obj["legal"], "legal"),
            previous_action=previous,
        )

    @property
    def identity(self) -> tuple[str, int, str, str]:
        return self.session_id, self.decision_id, self.phase, self.state_fingerprint


@dataclass(frozen=True, slots=True)
class DecisionResponse:
    protocol_version: int
    session_id: str
    decision_id: int
    state_fingerprint: str
    phase: str
    action: dict[str, Any] | None = None
    wait: bool | None = None
    error: dict[str, Any] | None = None

    @classmethod
    def action_response(cls, request: DecisionRequest, action: Mapping[str, Any]) -> DecisionResponse:
        return cls(
            PROTOCOL_VERSION,
            request.session_id,
            request.decision_id,
            request.state_fingerprint,
            request.phase,
            action=dict(action),
        )

    @classmethod
    def wait_response(cls, request: DecisionRequest) -> DecisionResponse:
        return cls(
            PROTOCOL_VERSION,
            request.session_id,
            request.decision_id,
            request.state_fingerprint,
            request.phase,
            wait=True,
        )

    @classmethod
    def error_response(cls, request: DecisionRequest, code: str, message: str) -> DecisionResponse:
        return cls(
            PROTOCOL_VERSION,
            request.session_id,
            request.decision_id,
            request.state_fingerprint,
            request.phase,
            error={"code": code, "message": message},
        )

    def to_dict(self) -> dict[str, Any]:
        body = asdict(self)
        return {key: value for key, value in body.items() if value is not None}
