"""Dependency-free observation/checkpoint schema metadata.

The standalone deployment validator reads this file without importing the
agent or engine. Keep it free of imports and runtime initialization.
"""

# Bump when observation/action layout or interpretation becomes incompatible.
# v13 fixes phase-aware risk/progress and exposes economy/card-state inputs.
# Cached v12 teacher data and strict-resume checkpoints must not be reused.
# v14 appends deck id and stake scalars so one policy can condition on the run setup.
# v15 puts engine-exact guaranteed chips and remaining-blind coverage on play candidates.
# v16 lists play candidates best-score first and adds padded variants of the best cores.
TOKENIZER_VERSION = 17
TOKENIZER_SEMANTICS = "v17_plan_discards"

# Historical semantics retained only for the explicit actor-transfer path.
ACTOR_TRANSFER_SCHEMAS = {
    11: "v8_conditional_survival_critic",
    12: "v12_joker_state_archive",
    13: "v13_phase_risk_economy_card_state",
    14: "v14_deck_stake_context",
    15: "v15_exact_play_scores",
    16: "v16_padded_play_candidates",
    TOKENIZER_VERSION: TOKENIZER_SEMANTICS,
}
