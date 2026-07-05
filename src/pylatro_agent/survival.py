"""Per-ante survival targets for the ante_survival value-head output.

Given an episode outcome (max_ante, won), produce per-ante binary targets and
a loss mask so that the ante_survival head can be trained with BCE on just
the antes we have evidence for (skip antes beyond what the trajectory reached).

Convention (head output is shape (max_antes,)):
  target[i] = 1 iff the player survived ante (i+1) (beat its boss)
  target[i] = 0 iff the player reached ante (i+1) but died on it
  mask[i]   = 1 iff target[i] is observed (otherwise unknown / padded)
"""

from __future__ import annotations

import numpy as np

# Match ValueHead default; keep in lockstep if it changes.
DEFAULT_MAX_ANTES = 8


def compute_ante_survival_targets(
    max_ante: int,
    won: bool,
    num_antes: int = DEFAULT_MAX_ANTES,
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.zeros(num_antes, dtype=np.float32)
    mask = np.zeros(num_antes, dtype=np.float32)

    if won:
        # Winning currently means beating ante 8; survived every ante.
        capped = min(max_ante, num_antes)
        targets[:capped] = 1.0
        mask[:capped] = 1.0
        return targets, mask

    # Lost at ante `max_ante`, survived 1..max_ante-1, died on max_ante.
    capped = min(max_ante, num_antes)
    if capped >= 1:
        targets[: capped - 1] = 1.0
        mask[: capped - 1] = 1.0
        # The ante they died on is an observed failure.
        targets[capped - 1] = 0.0
        mask[capped - 1] = 1.0
    # Antes beyond what was reached stay masked out.
    return targets, mask
