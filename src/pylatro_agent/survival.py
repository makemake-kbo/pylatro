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
import torch

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


def compute_conditional_ante_survival_targets(
    final_ante: int,
    won: bool,
    *,
    current_ante: int,
    win_ante: int,
    num_antes: int = DEFAULT_MAX_ANTES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return future-only conditional-hazard targets for one transition.

    ``ante_survival[i]`` is interpreted as the conditional probability of
    surviving ante ``i + 1`` after reaching it.  A state in ante 3 therefore
    receives no replay loss for the already-observed ante-1/2 outcomes.  For a
    loss in ante 5, antes 3/4 are successes and ante 5 is the first failure;
    later hazards are outside the observed path and remain masked.

    This target convention lets the hazard vector define a coherent outcome
    distribution over ``death at each remaining ante`` plus ``reach target``.
    """

    if num_antes <= 0:
        raise ValueError("num_antes must be positive")
    goal = min(max(int(win_ante), 1), num_antes)
    current = min(max(int(current_ante), 1), goal)
    final = min(max(int(final_ante), current), goal)

    targets = np.zeros(num_antes, dtype=np.float32)
    mask = np.zeros(num_antes, dtype=np.float32)
    start = current - 1
    if won:
        targets[start:goal] = 1.0
        mask[start:goal] = 1.0
        return targets, mask

    failure = final - 1
    if failure > start:
        targets[start:failure] = 1.0
    mask[start : failure + 1] = 1.0
    return targets, mask


def hazard_outcome_probabilities(
    hazards: torch.Tensor,
    current_antes: torch.Tensor,
    win_antes: torch.Tensor,
) -> torch.Tensor:
    """Convert conditional per-Ante survival hazards to outcome probabilities.

    Args:
        hazards: ``(batch, max_antes)`` probabilities. Column ``i`` is the
            conditional probability of surviving ante ``i + 1`` after reaching
            it.
        current_antes: ``(batch,)`` one-based current Ante values.
        win_antes: ``(batch,)`` one-based target Ante values.

    Returns:
        ``(batch, max_antes + 1)`` probabilities. Columns ``0..max_antes-1``
        represent first death in that Ante; the final column represents
        reaching the configured target. Death classes before the current Ante
        and after the target have exactly zero mass, and every row sums to one.
    """

    if hazards.ndim != 2:
        raise ValueError("hazards must have shape (batch, max_antes)")
    batch_size, max_antes = hazards.shape
    current = current_antes.to(device=hazards.device, dtype=torch.long).reshape(-1)
    goal = win_antes.to(device=hazards.device, dtype=torch.long).reshape(-1)
    if current.shape[0] != batch_size or goal.shape[0] != batch_size:
        raise ValueError("current_antes and win_antes must match the hazard batch")
    current = current.clamp(1, max_antes)
    goal = goal.clamp(1, max_antes)
    current = torch.minimum(current, goal)

    probability = hazards.clamp(1e-7, 1.0 - 1e-7)
    survival_mass = torch.ones(batch_size, dtype=hazards.dtype, device=hazards.device)
    death_columns: list[torch.Tensor] = []
    for idx in range(max_antes):
        ante = idx + 1
        active = (current <= ante) & (ante <= goal)
        death_mass = survival_mass * (1.0 - probability[:, idx])
        death_columns.append(torch.where(active, death_mass, torch.zeros_like(death_mass)))
        survival_mass = torch.where(active, survival_mass * probability[:, idx], survival_mass)
    return torch.stack([*death_columns, survival_mass], dim=-1)
