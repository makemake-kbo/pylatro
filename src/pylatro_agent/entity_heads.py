"""Explicit, initially neutral action-head variants on unchanged v13 inputs."""

from __future__ import annotations

import torch
from torch import nn

from .constants import MAX_JOKER_SLOTS

ACTION_HEAD_VARIANTS = (
    "legacy_v1", "pooled_sale_residual_v1", "entity_sale_residual_v1",
    "subset_residual_v1", "entity_sale_subset_residual_v1",
)


class SaleResidualHead(nn.Module):
    """Compare entity conditioning with a pooled control of matching capacity.

    Both variants start with identically zero logits. The control has nearly
    the same parameter count, so an improvement cannot be attributed simply
    to adding a larger MLP to the policy.
    """

    def __init__(self, d_model: int, *, entity_conditioned: bool, state_dim: int = 128):
        super().__init__()
        self.entity_conditioned = entity_conditioned
        entity_hidden = 128
        entity_parameters = (d_model + state_dim + 1) * entity_hidden + entity_hidden + 1
        pooled_hidden = max(1, round((entity_parameters - MAX_JOKER_SLOTS) / (state_dim + MAX_JOKER_SLOTS + 1)))
        input_dim = state_dim + d_model if entity_conditioned else state_dim
        hidden = entity_hidden if entity_conditioned else pooled_hidden
        outputs = 1 if entity_conditioned else MAX_JOKER_SLOTS
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, outputs))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor, joker_tokens: torch.Tensor) -> torch.Tensor:
        if self.entity_conditioned:
            expanded = state.unsqueeze(1).expand(-1, joker_tokens.shape[1], -1)
            return self.net(torch.cat((expanded, joker_tokens), dim=-1)).squeeze(-1)
        return self.net(state)
