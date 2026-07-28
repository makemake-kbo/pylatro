"""Live Balatro bridge support.

The Lua mod owns only observation and UI actuation.  This package validates
its protocol, adapts snapshots to the existing agent state, and chooses
semantic actions.
"""

from .protocol import PROTOCOL_VERSION, DecisionRequest, DecisionResponse, ProtocolError

__all__ = [
    "PROTOCOL_VERSION",
    "DecisionRequest",
    "DecisionResponse",
    "ProtocolError",
]
