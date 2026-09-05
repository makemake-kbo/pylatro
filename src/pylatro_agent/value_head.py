"""Conditional terminal-outcome critic with a scalar return correction."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reward import outcome_value
from .survival import (
    DEFAULT_MAX_ANTES,
    _validated_ante_tensor,
    hazard_outcome_probabilities,
)


def outcome_nll(
    outcome_probabilities: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Categorical terminal-outcome NLL with a valid-label denominator."""

    selected = outcome_probabilities.gather(1, targets.long().reshape(-1, 1)).squeeze(1)
    per_row = -selected.clamp_min(1e-7).log()
    if mask is None:
        return per_row.mean()
    valid = mask.to(dtype=per_row.dtype).reshape(-1)
    return (per_row * valid).sum() / valid.sum().clamp(min=1.0)


def return_huber_loss(
    value_outputs: dict[str, torch.Tensor],
    return_targets: torch.Tensor,
) -> torch.Tensor:
    """Regress the residual while treating terminal calibration as fixed."""

    residual_target = return_targets - value_outputs["terminal_value"].detach()
    return F.huber_loss(
        value_outputs["return_residual"],
        residual_target,
        delta=1.0,
    )


def terminal_outcome_utilities(
    win_antes: torch.Tensor,
    *,
    max_antes: int = DEFAULT_MAX_ANTES,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return utilities for death-at-Ante classes and the final win class.

    The result has shape ``(batch, max_antes + 1)``. Utilities for impossible
    death classes are harmless because their probabilities are exactly zero.
    """

    goals = _validated_ante_tensor(
        win_antes.to(device=device),
        name="win_antes",
        max_antes=max_antes,
    )
    table = _terminal_utility_table(max_antes).to(
        device=goals.device,
        dtype=dtype or torch.float32,
    )
    return table.index_select(0, goals - 1)


def _terminal_utility_table(max_antes: int) -> torch.Tensor:
    """Build the small goal-by-outcome utility lookup in reward-model terms."""

    return torch.tensor(
        [
            [
                *[
                    outcome_value(won=False, ante=ante, win_ante=goal)
                    for ante in range(1, max_antes + 1)
                ],
                outcome_value(won=True, ante=goal, win_ante=goal),
            ]
            for goal in range(1, max_antes + 1)
        ],
        dtype=torch.float32,
    )


class ValueHead(nn.Module):
    """Predict terminal hazards and the non-terminal correction to return.

    ``outcome_proj`` and ``ante_survival`` parameterize the terminal outcome
    distribution. The scalar ``return_residual`` represents discounting, dense
    shaping, and any other difference between terminal utility and the PPO
    return target.

    The outcome tower deliberately does not share ``pool_proj`` with the return
    residual. Complete-episode replay is the only unbiased outcome supervision
    in the run, and it must be able to reshape the representation the hazards
    read rather than retune a single linear layer over features it cannot
    touch. A private projection lets replay do that without perturbing the
    return path that GAE consumes.
    """

    def __init__(self, d_model: int = 256, max_ante: int = DEFAULT_MAX_ANTES, *, win_only: bool = False):
        super().__init__()
        self.max_ante = int(max_ante)
        self.pool_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.outcome_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.ante_survival = nn.Linear(d_model, self.max_ante)
        self.return_residual = nn.Linear(d_model, 1)
        utilities = _terminal_utility_table(self.max_ante)
        if win_only:
            utilities[:, :-1] = 0.0
        self.register_buffer(
            "_terminal_utilities",
            utilities,
            persistent=False,
        )

    def forward(
        self,
        backbone_out: torch.Tensor,
        padding_mask: torch.Tensor,
        current_antes: torch.Tensor | None = None,
        win_antes: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return canonical critic predictions and exact compatibility aliases."""

        mask = padding_mask.unsqueeze(-1).to(dtype=backbone_out.dtype)
        pooled = (backbone_out * mask).sum(1) / mask.sum(1).clamp(min=1)
        hidden = self.pool_proj(pooled)
        hazards = torch.sigmoid(self.ante_survival(self.outcome_proj(pooled)))

        batch_size = backbone_out.shape[0]
        if current_antes is None:
            current_antes = torch.ones(batch_size, device=backbone_out.device)
        if win_antes is None:
            win_antes = torch.full(
                (batch_size,), self.max_ante, device=backbone_out.device
            )
        outcome_probabilities = hazard_outcome_probabilities(
            hazards,
            current_antes,
            win_antes,
        )
        # ``hazard_outcome_probabilities`` validates this same tensor before
        # it reaches the utility lookup, so no second device synchronization
        # is needed on the critic hot path.
        goals = win_antes.to(device=backbone_out.device, dtype=torch.long).reshape(-1)
        utilities = torch.index_select(
            cast("torch.Tensor", self._terminal_utilities).to(dtype=backbone_out.dtype),
            0,
            goals - 1,
        )
        terminal_value = (outcome_probabilities * utilities).sum(dim=-1)
        return_residual = self.return_residual(hidden).squeeze(-1)
        expected_return = terminal_value + return_residual
        win_prob = outcome_probabilities[:, -1]
        return {
            "ante_survival": hazards,
            "outcome_probabilities": outcome_probabilities,
            "terminal_value": terminal_value,
            "return_residual": return_residual,
            "expected_return": expected_return,
            # Live/API compatibility aliases. These deliberately reference the
            # canonical tensors rather than recomputing equivalent values.
            "expected_score": expected_return,
            "win_prob": win_prob,
        }
