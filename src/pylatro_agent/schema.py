"""Dependency-free observation/checkpoint schema metadata.

The standalone deployment validator reads this file without importing the
agent or engine. Keep it free of imports and runtime initialization.
"""

# Bump when observation/action layout or interpretation becomes incompatible.
# v13 fixes phase-aware risk/progress and exposes economy/card-state inputs.
# Cached v12 teacher data and strict-resume checkpoints must not be reused.
TOKENIZER_VERSION = 13
TOKENIZER_SEMANTICS = "v13_phase_risk_economy_card_state"

# Historical semantics retained only for the explicit actor-transfer path.
ACTOR_TRANSFER_SCHEMAS = {
    11: "v8_conditional_survival_critic",
    12: "v12_joker_state_archive",
    TOKENIZER_VERSION: TOKENIZER_SEMANTICS,
}
