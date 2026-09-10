"""Conditional survival hazards and categorical terminal outcomes."""

from __future__ import annotations

import torch

# Match ValueHead default; keep in lockstep if it changes.
DEFAULT_MAX_ANTES = 8


def validate_critic_win_ante(
    win_ante: int,
    *,
    name: str = "win_ante",
    max_antes: int = DEFAULT_MAX_ANTES,
) -> int:
    """Return a critic-compatible victory Ante or raise clearly."""

    if isinstance(win_ante, bool):
        raise ValueError(f"{name} must be an integer between 1 and {max_antes}")
    try:
        value = int(win_ante)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be an integer between 1 and {max_antes}"
        ) from exc
    if value != win_ante or not 1 <= value <= max_antes:
        raise ValueError(f"{name} must be between 1 and {max_antes}, got {win_ante!r}")
    return value


def _validated_ante_tensor(
    values: torch.Tensor,
    *,
    name: str,
    max_antes: int,
) -> torch.Tensor:
    """Validate one-based critic Ante values and return them as longs."""

    flattened = values.reshape(-1)
    if not bool(torch.all(_valid_ante_values(flattened, max_antes=max_antes))):
        raise ValueError(f"{name} must contain integers between 1 and {max_antes}")
    return flattened.to(dtype=torch.long)


def _valid_ante_values(values: torch.Tensor, *, max_antes: int) -> torch.Tensor:
    return (
        (values == values.round())
        & (values >= 1)
        & (values <= max_antes)
    )


def terminal_outcome_class(
    *,
    won: bool,
    final_ante: int,
    num_antes: int = DEFAULT_MAX_ANTES,
) -> int:
    """Map a complete non-stalled episode to its categorical outcome class.

    The engine increments the Ante counter when clearing the winning boss,
    so a terminal win can report ``num_antes + 1``. This is still the win
    class, not another critic Ante. Deaths must remain within supported Antes.
    """

    final = validate_critic_win_ante(
        final_ante,
        name="final_ante",
        max_antes=num_antes + int(won),
    )
    if won:
        return num_antes
    return final - 1


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
    current_values = current_antes.to(device=hazards.device).reshape(-1)
    goal_values = win_antes.to(device=hazards.device).reshape(-1)
    if current_values.shape[0] != batch_size or goal_values.shape[0] != batch_size:
        raise ValueError("current_antes and win_antes must match the hazard batch")
    valid = (
        _valid_ante_values(current_values, max_antes=max_antes)
        & _valid_ante_values(goal_values, max_antes=max_antes)
        & (current_values <= goal_values)
    )
    if not bool(torch.all(valid)):
        raise ValueError(
            f"current_antes and win_antes must contain integers between 1 and "
            f"{max_antes}, with current Ante no later than the victory Ante"
        )
    current = current_values.to(dtype=torch.long)
    goal = goal_values.to(dtype=torch.long)

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
