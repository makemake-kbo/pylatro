"""Dependency-free observation/checkpoint schema metadata.

The standalone deployment validator reads this file without importing the
agent or engine. Keep it free of imports and runtime initialization.
"""

# Bump when observation/action layout or interpretation becomes incompatible.
# v13 fixes phase-aware risk/progress and exposes economy/card-state inputs.
# Cached v12 teacher data and strict-resume checkpoints must not be reused.
# v14 appends deck id and stake scalars so one policy can condition on the run setup.
TOKENIZER_VERSION = 14
TOKENIZER_SEMANTICS = "v14_deck_stake_context"

# Historical semantics retained only for the explicit actor-transfer path.
ACTOR_TRANSFER_SCHEMAS = {
    11: "v8_conditional_survival_critic",
    12: "v12_joker_state_archive",
    13: "v13_phase_risk_economy_card_state",
    TOKENIZER_VERSION: TOKENIZER_SEMANTICS,
}
