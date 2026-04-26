"""Top-level BalatroAgent nn.Module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .action_grammar import ActionGrammarDistribution, ActionGrammarHead
from .action_heads import BlindSelectHead, ConsumableFlatHead, HandPlayHead, PackHead, ShopHead
from .backbone import TransformerBackbone
from .constants import MAX_SEQ_LEN, NUM_ACTIONS, SubPhase
from .distributions import MaskedCategorical  # noqa: F401 — re-exported for callers
from .embeddings import ContentEmbeddingLayer
from .value_head import ValueHead

if TYPE_CHECKING:
    from .vocab import Vocab


@dataclass
class AgentConfig:
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = MAX_SEQ_LEN


class BalatroAgent(nn.Module):
    def __init__(self, config: AgentConfig, vocab: Vocab):
        super().__init__()
        self.config = config
        d = config.d_model

        self.embedding = ContentEmbeddingLayer(vocab, d)
        self.backbone = TransformerBackbone(
            n_layers=config.n_layers,
            d_model=d,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )

        # Action heads
        self.blind_select_head = BlindSelectHead(d)
        self.hand_play_head = HandPlayHead(d, vocab)
        self.shop_head = ShopHead(d)
        self.consumable_flat_head = ConsumableFlatHead(d)
        self.pack_head = PackHead(d)
        self.action_grammar_head = ActionGrammarHead(d)

        # Value head
        self.value_head = ValueHead(d)

    def forward(
        self,
        tokens: torch.Tensor,
        token_types: torch.Tensor,
        scalars: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
        sub_phase: SubPhase | list[SubPhase] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            tokens: (batch, MAX_SEQ_LEN, TOKEN_DIM) int
            token_types: (batch, MAX_SEQ_LEN) int
            scalars: (batch, SCALAR_DIM) float
            attention_mask: (batch, MAX_SEQ_LEN) int — 1=real, 0=pad
            action_mask: (batch, NUM_ACTIONS) int — 1=valid
            sub_phase: SubPhase or list of SubPhase per batch element
        Returns:
            (action_distribution, value_dict)
        """
        # Embed
        x = self.embedding(tokens, token_types, scalars)

        # Backbone — padding_mask for MHA should be True where padded
        padding_mask = (attention_mask == 0)
        x = self.backbone(x, padding_mask=padding_mask)

        # Route to appropriate head
        if sub_phase is None:
            # Infer from scalars (phase is at index 7)
            phase_ids = scalars[:, 7].long()
            sub_phase_list = [self._phase_id_to_sub_phase(pid.item()) for pid in phase_ids]
        elif isinstance(sub_phase, SubPhase):
            sub_phase_list = [sub_phase] * tokens.shape[0]
        else:
            sub_phase_list = sub_phase

        logits = self._compute_logits(x, attention_mask, tokens, token_types, scalars, sub_phase_list)

        # Value prediction
        value_dict = self.value_head(x, attention_mask)

        # Return raw logits — callers construct MaskedCategorical.
        # This is necessary for nn.DataParallel which can only gather tensors.
        return logits, value_dict

    def action_distribution(
        self,
        tokens: torch.Tensor,
        token_types: torch.Tensor,
        scalars: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[ActionGrammarDistribution, dict[str, torch.Tensor]]:
        """Return the structured action-grammar distribution and value head output.

        This is the training/rollout path. The legacy ``forward`` method still
        returns dense flat logits for compatibility tests and old callers.
        """
        x = self.embedding(tokens, token_types, scalars)
        padding_mask = (attention_mask == 0)
        x = self.backbone(x, padding_mask=padding_mask)
        grammar_output = self.action_grammar_head(x, attention_mask, tokens, token_types, scalars)
        value_dict = self.value_head(x, attention_mask)
        return ActionGrammarDistribution(grammar_output, action_mask, temperature=temperature), value_dict

    def _compute_logits(
        self,
        backbone_out: torch.Tensor,
        attention_mask: torch.Tensor,
        tokens: torch.Tensor,
        token_types: torch.Tensor,
        scalars: torch.Tensor,
        sub_phases: list[SubPhase],
    ) -> torch.Tensor:
        """Compute logits by routing each batch element to the correct head."""
        batch = backbone_out.shape[0]
        device = backbone_out.device

        # For efficiency, batch by sub-phase type
        phase_groups: dict[SubPhase, list[int]] = {}
        for i, sp in enumerate(sub_phases):
            phase_groups.setdefault(sp, []).append(i)

        logits = torch.full((batch, NUM_ACTIONS), -1e8, device=device)

        for sp, indices in phase_groups.items():
            idx = torch.tensor(indices, device=device)
            bo = backbone_out[idx]
            am = attention_mask[idx]
            tok = tokens[idx]
            tok_types = token_types[idx]
            scal = scalars[idx]

            if sp == SubPhase.BLIND_SELECT:
                head_logits = self.blind_select_head(bo, am)
            elif sp == SubPhase.CHOOSE_ACTION:
                # Play/discard/consumable all live in CHOOSE_ACTION now.
                # Each head writes to disjoint action-range slices and
                # leaves the rest at -1e8, so elementwise max merges them
                # without corrupting masked positions.
                play_logits = self.hand_play_head(bo, am, tok, tok_types, scal, select_mode=False)
                cons_logits = self.consumable_flat_head(bo, am, tok, tok_types)
                head_logits = torch.maximum(play_logits, cons_logits)
            elif sp == SubPhase.SELECT_CARDS:
                head_logits = self.hand_play_head(bo, am, tok, tok_types, scal, select_mode=True)
            elif sp == SubPhase.SHOP:
                head_logits = self.shop_head(bo, am)
            elif sp == SubPhase.BOOSTER_PACK:
                head_logits = self.pack_head(bo, am)
            else:
                continue

            logits[idx] = head_logits

        return logits

    @staticmethod
    def _phase_id_to_sub_phase(phase_id: int) -> SubPhase:
        return [
            SubPhase.BLIND_SELECT,
            SubPhase.CHOOSE_ACTION,
            SubPhase.SELECT_CARDS,
            SubPhase.SHOP,
            SubPhase.BOOSTER_PACK,
        ][min(phase_id, 4)]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
