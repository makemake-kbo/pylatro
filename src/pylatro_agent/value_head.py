"""Value head, predicts win probability, expected score, and ante survival."""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    def __init__(self, d_model: int = 256, max_ante: int = 8):
        super().__init__()
        self.max_ante = max_ante
        self.pool_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.win_prob = nn.Linear(d_model, 1)
        self.expected_score = nn.Linear(d_model, 1)
        self.ante_survival = nn.Linear(d_model, max_ante)

    def forward(self, backbone_out: torch.Tensor, padding_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            backbone_out: (batch, seq_len, d_model)
            padding_mask: (batch, seq_len), 1 for real tokens, 0 for pad
        Returns:
            dict with win_prob (batch,), expected_score (batch,), ante_survival (batch, max_ante)
        """
        # Mean pool non-padded tokens
        mask = padding_mask.unsqueeze(-1).float()  # (batch, seq, 1)
        pooled = (backbone_out * mask).sum(1) / mask.sum(1).clamp(min=1)
        h = self.pool_proj(pooled)

        return {
            "win_prob": torch.sigmoid(self.win_prob(h).squeeze(-1)),
            "expected_score": self.expected_score(h).squeeze(-1),
            "ante_survival": torch.sigmoid(self.ante_survival(h)),
        }
