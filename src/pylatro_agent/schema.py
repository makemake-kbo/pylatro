"""Dependency-free observation/checkpoint schema metadata.

The standalone deployment validator reads this file without importing the
agent or engine. Keep it free of imports and runtime initialization.
"""

# Bump when observation/action layout or interpretation becomes incompatible.
# v12 appends mutable Joker state to the original 12 token fields.
TOKENIZER_VERSION = 12
TOKENIZER_SEMANTICS = "v12_joker_state_archive"

# Historical semantics retained only for the explicit actor-transfer path.
ACTOR_TRANSFER_SCHEMAS = {
    11: "v8_conditional_survival_critic",
    TOKENIZER_VERSION: TOKENIZER_SEMANTICS,
}
