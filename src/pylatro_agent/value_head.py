"""Value head, predicts win probability, expected score, and ante survival.

The expected-score output supports two parameterizations:

* Scalar (``value_bins=0``, the legacy default): ``Linear(d, 1)`` trained
  with MSE against lambda-returns.
* HL-Gauss categorical (``value_bins>0``): ``Linear(d, value_bins)`` logits
  over a fixed grid of return atoms spanning ``[value_v_min, value_v_max]``,
  trained with cross-entropy against a Gaussian-smeared projection of the
  scalar return target (Farebrother et al. 2024, "Stop Regressing"). The
  scalar value consumed by GAE/logging/SIL is the mean of the predicted
  histogram, so downstream PPO code is identical in both modes. The
  categorical loss keeps critic gradients bounded on near-terminal
  coin-flip states where MSE against a bimodal (+win / -loss) target
  produces large alternating-sign errors through the shared trunk.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Ratio of the label-smoothing Gaussian's sigma to the bin width. 0.75 is the
# sweet spot reported by the HL-Gauss paper: wide enough that each target
# spreads over ~3 bins (dense gradient), narrow enough to stay unimodal.
HL_GAUSS_SIGMA_RATIO = 0.75


def hl_gauss_projection(
    targets: torch.Tensor, bin_edges: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Project scalar targets onto bin probabilities via a Gaussian CDF.

    Args:
        targets: (batch,) scalar return targets.
        bin_edges: (n_bins + 1,) monotonically increasing bin edges.
        sigma: Gaussian smearing width, in return units.
    Returns:
        (batch, n_bins) probabilities, each row summing to 1. Targets are
        clamped to the atom range first: far outside the grid the Gaussian
        puts ~zero mass in every bin and renormalization cannot rescue an
        all-zero row, which would silently drop that sample from the loss.
        Clamped targets degrade to edge-bin-heavy distributions instead.
    """
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    targets = targets.clamp(min=float(centers[0]), max=float(centers[-1]))
    z = (bin_edges.unsqueeze(0) - targets.unsqueeze(-1)) / (sigma * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + torch.erf(z))
    probs = cdf[..., 1:] - cdf[..., :-1]
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class ValueHead(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        max_ante: int = 8,
        value_bins: int = 0,
        value_v_min: float = -8.0,
        value_v_max: float = 12.0,
    ):
        super().__init__()
        self.max_ante = max_ante
        self.value_bins = value_bins
        self.pool_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.win_prob = nn.Linear(d_model, 1)
        # Keep expected_score registered before ante_survival. Full PPO
        # checkpoints restore Adam state by parameter position, and legacy
        # scalar heads used this order.
        if value_bins > 0:
            if value_bins < 2:
                raise ValueError("value_bins must be 0 (scalar head) or >= 2")
            if value_v_max <= value_v_min:
                raise ValueError("value_v_max must exceed value_v_min")
            self.expected_score = nn.Linear(d_model, value_bins)
            centers = torch.linspace(value_v_min, value_v_max, value_bins)
            width = (value_v_max - value_v_min) / (value_bins - 1)
            edges = torch.cat([centers - width / 2, centers[-1:] + width / 2])
            # Derived from config, not learned: keep out of the state_dict so
            # scalar-head checkpoints load into categorical models with only
            # the expected_score.* tensors reported as incompatible.
            self.register_buffer("bin_centers", centers, persistent=False)
            self.register_buffer("bin_edges", edges, persistent=False)
            self.hl_gauss_sigma = HL_GAUSS_SIGMA_RATIO * width
        else:
            self.expected_score = nn.Linear(d_model, 1)
        self.ante_survival = nn.Linear(d_model, max_ante)

    def forward(self, backbone_out: torch.Tensor, padding_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            backbone_out: (batch, seq_len, d_model)
            padding_mask: (batch, seq_len), 1 for real tokens, 0 for pad
        Returns:
            dict with win_prob (batch,), expected_score (batch,), ante_survival
            (batch, max_ante); categorical mode adds expected_score_logits
            (batch, value_bins) for the HL-Gauss cross-entropy loss.
        """
        # Mean pool non-padded tokens
        mask = padding_mask.unsqueeze(-1).float()  # (batch, seq, 1)
        pooled = (backbone_out * mask).sum(1) / mask.sum(1).clamp(min=1)
        h = self.pool_proj(pooled)

        out = {
            "win_prob": torch.sigmoid(self.win_prob(h).squeeze(-1)),
            "ante_survival": torch.sigmoid(self.ante_survival(h)),
        }
        if self.value_bins > 0:
            logits = self.expected_score(h)
            out["expected_score_logits"] = logits
            out["expected_score"] = (logits.softmax(dim=-1) * self.bin_centers).sum(dim=-1)
        else:
            out["expected_score"] = self.expected_score(h).squeeze(-1)
        return out
