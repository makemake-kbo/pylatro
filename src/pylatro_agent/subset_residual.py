"""Whole-subset energy corrections for a full-support reference policy.

This module scores selected entities jointly. It does not query the simulator
or change observation fields; callers supply contextual card embeddings and
membership masks for the legal action support they are evaluating.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
from torch import nn

from .constants import MAX_HAND_SIZE, ActionRange
from .subset_actions import HAND_SUBSET_BITS, HAND_SUBSETS


@lru_cache(maxsize=64)
def subset_support(width: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Sorted global subset indices and membership, with no proposal pruning."""
    if not 0 <= width <= MAX_HAND_SIZE:
        raise ValueError("Invalid hand support width")
    indices = np.flatnonzero(HAND_SUBSET_BITS.astype(np.int64) < (1 << width))
    membership = np.zeros((len(indices), width), dtype=np.float32)
    for row, index in enumerate(indices):
        membership[row, list(HAND_SUBSETS[index])] = 1
    # A cache first filled during evaluation must still be safe in autograd.
    with torch.inference_mode(False):
        return torch.tensor(indices, device=device), torch.tensor(membership, device=device)


class SubsetResidualHead(nn.Module):
    def __init__(self, d_model: int, *, state_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.card_projection = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU())
        self.state_projection = nn.Linear(state_dim, hidden)
        self.count_embedding = nn.Embedding(6, hidden)
        self.score = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, state: torch.Tensor, cards: torch.Tensor, membership: torch.Tensor) -> torch.Tensor:
        """Return (batch, play/discard, subsets) energy corrections.

        Membership is (subsets, hand slots); evaluating only legal subsets or
        buckets with the same support width bounds memory for large hands.
        """
        members = membership.to(device=cards.device, dtype=cards.dtype)
        counts = members.sum(-1)
        card_features = self.card_projection(cards)
        pooled = torch.matmul(members.unsqueeze(0), card_features) / counts.clamp_min(1).sqrt()[None, :, None]
        context = self.state_projection(state)[:, None, :] + self.count_embedding(counts.long().clamp(0, 5))[None]
        return self.score(pooled + context).permute(0, 2, 1)

    def score_available(self, state: torch.Tensor, cards: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """Dense leaf energies, computing hidden activations in bounded chunks.

        Group by the highest occupied hand slot, not the number of cards; this
        also preserves noncontiguous slot semantics. Legality remains the
        distribution's responsibility. Missing-card subsets are never legal.
        """
        widths = (present.bool() * torch.arange(1, cards.shape[1] + 1, device=cards.device)).amax(-1).cpu()
        result = state.new_zeros((len(state), 2, len(HAND_SUBSETS)))
        for width_tensor in widths.unique():
            width = int(width_tensor)
            if not width:
                continue
            indices, membership = subset_support(width, str(cards.device))
            rows = torch.where(widths == width)[0].to(cards.device)
            # At most 64 rows x 512 subsets x hidden, including unusual hands.
            for row_start in range(0, len(rows), 64):
                selected_rows = rows[row_start : row_start + 64]
                for start in range(0, len(indices), 512):
                    selected_indices = indices[start : start + 512]
                    values = self(state[selected_rows], cards[selected_rows, :width], membership[start : start + 512])
                    result[
                        selected_rows[:, None, None],
                        torch.arange(2, device=cards.device)[None, :, None],
                        selected_indices[None, None, :],
                    ] = values
        return result


def corrected_subset_log_probs(
    base_log_probs: torch.Tensor,
    energies: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Normalize reference log probabilities plus a learned energy correction.

    Exact zero corrections preserve the reference distribution mathematically.
    Invalid and empty rows remain finite and have zero probability when masked.
    A finite reference log probability is required on each legal action: the
    correction reweights full support rather than inventing unsupported actions.
    """
    scores = (base_log_probs + energies).masked_fill(~valid, torch.finfo(base_log_probs.dtype).min)
    normalized = scores - torch.logsumexp(scores, dim=-1, keepdim=True)
    return normalized.masked_fill(~valid, -1e8)


def reference_hand_log_probs(distribution, *, is_play: bool) -> torch.Tensor:
    """The original candidate/AR distribution with stable full-support logs.

    The AR term is never converted to probability before mixing: a legal
    low-probability action must stay reachable by an energy correction. The
    candidate term aggregates duplicate proposal slots before combining it.
    """
    from .action_grammar import _all_subset_log_probs, _hand_subset_bits, _hand_subset_sizes, _masked_softmax

    eps = distribution.hand_ar_mixture_eps
    if not 0 < eps <= 1:
        raise ValueError("Subset residual requires a positive AR mixture for full support")
    family = 0 if is_play else 1
    valid = distribution._hand_valid_subset_mask(is_play)
    ar_logp = _all_subset_log_probs(
        distribution._t(distribution.output.hand_count_logits[:, family]),
        distribution._t(distribution.output.hand_card_logits[:, family]),
        valid,
        sizes=_hand_subset_sizes(distribution.device),
        subset_bits=_hand_subset_bits(distribution.device),
    )
    if eps == 1:
        return ar_logp
    logits, candidate_valid, actions = distribution._candidate_distribution(is_play=is_play)
    start = int(ActionRange.PLAY_SUBSET_START if is_play else ActionRange.DISCARD_SUBSET_START)
    indices = (actions - start).clamp(0, valid.shape[-1] - 1)
    candidate_probs = torch.zeros_like(ar_logp).scatter_add(1, indices, _masked_softmax(logits, candidate_valid))
    # Avoid log(0) in autograd even though a later logaddexp has finite output.
    candidate_logp = candidate_probs.clamp_min(torch.finfo(candidate_probs.dtype).tiny).log()
    candidate_logp = candidate_logp.masked_fill(candidate_probs == 0, -torch.inf)
    mixture = torch.logaddexp(math.log1p(-eps) + candidate_logp, math.log(eps) + ar_logp)
    return torch.where(candidate_valid.any(-1, keepdim=True), mixture, ar_logp).masked_fill(~valid, -1e8)
