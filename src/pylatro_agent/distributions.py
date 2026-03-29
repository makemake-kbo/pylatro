"""Masked categorical distribution for action selection."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class MaskedCategorical(Categorical):
    """Categorical distribution that masks invalid actions to zero probability."""

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            logits: (batch, num_actions) raw logits
            mask: (batch, num_actions) binary mask, 1 = valid
        """
        # Set invalid logits to -inf so softmax gives 0 probability
        masked_logits = logits.clone()
        masked_logits[mask == 0] = -1e8
        super().__init__(logits=masked_logits)
        self.mask = mask

    def entropy(self) -> torch.Tensor:
        """Entropy computed only over valid actions."""
        # Use parent entropy which already works on masked logits
        return super().entropy()
