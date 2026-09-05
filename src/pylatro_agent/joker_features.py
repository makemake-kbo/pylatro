"""Versioned numeric Joker state, kept separate from categorical identity."""

from __future__ import annotations

import math

# Keep order stable: this is part of the tokenizer/checkpoint contract.
JOKER_STATE_FIELDS = (
    "mult",
    "t_mult",
    "t_chips",
    "x_mult",
    "h_mult",
    "h_x_mult",
    "caino_xmult",
    "yorick_discards",
    "loyalty_remaining",
    "invis_rounds",
    "driver_tally",
    "stone_tally",
    "steel_tally",
    "nine_tally",
    "money",
    "extra_value",
    "h_dollars",
    "p_dollars",
    "h_size",
    "d_size",
)
JOKER_EXTRA_FIELDS = (
    "chips",
    "mult",
    "Xmult",
    "chip_mod",
    "mult_mod",
    "Xmult_mod",
    "hand_add",
    "discard_sub",
    "dollars",
    "increase",
    "h_size",
    "h_mod",
    "every",
    "odds",
)
JOKER_FEATURE_NAMES = (*JOKER_STATE_FIELDS, "extra_scalar", *(f"extra_{key}" for key in JOKER_EXTRA_FIELDS))
JOKER_FEATURE_START = 12
JOKER_FEATURE_SCALE = 256.0


def encode_joker_features(joker) -> tuple[int, ...]:
    """Signed log features preserve growth without the old max/cap aliasing."""
    extra = joker.extra if isinstance(joker.extra, dict) else {}
    values = [getattr(joker, key, 0) for key in JOKER_STATE_FIELDS]
    values.append(joker.extra if isinstance(joker.extra, (int, float)) else 0)
    values.extend(extra.get(key, 0) for key in JOKER_EXTRA_FIELDS)
    result = []
    for value in values:
        value = float(value or 0)
        if math.isnan(value):
            value = 0.0
        encoded = math.copysign(math.log1p(abs(value)), value) * JOKER_FEATURE_SCALE
        result.append(int(max(-32767, min(32767, encoded))))
    return tuple(result)
