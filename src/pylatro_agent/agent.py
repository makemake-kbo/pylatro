"""Top-level BalatroAgent nn.Module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .action_grammar import ActionGrammarDistribution, ActionGrammarHead
from .backbone import TransformerBackbone
from .constants import (
    CURRENT_ANTE_SCALAR_INDEX,
    HISTORY_ROUNDS,
    HISTORY_START,
    MAX_SEQ_LEN,
    WIN_ANTE_SCALAR_INDEX,
)
from .embeddings import ContentEmbeddingLayer
from .precision import backbone_autocast
from .value_head import ValueHead

if TYPE_CHECKING:
    from .vocab import Vocab


@dataclass
class AgentConfig:
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 8
    d_ff: int = 1536
    dropout: float = 0.1
    max_seq_len: int = MAX_SEQ_LEN
    # Mixture weight on the autoregressive hand/discard head. 0.0 reproduces
    # the candidate-only support exactly. >0 gives the policy full support over
    # every legal hand play. Set per training stage: 0.5 during
    # supervised BC (dense cheap labels train the AR head), 0.1 during PPO.
    hand_ar_mixture_eps: float = 0.0
    # Soft, analytic-risk prior applied only in shops. At high immediate-death
    # probability it makes leaving less likely and rerolling more likely, but
    # never changes the legal-action mask. Zero disables the fixed prior while
    # retaining the learned direct danger-conditioning path.
    danger_shop_leave_logit_penalty: float = 0.0
    win_only_value: bool = False
    # CUDA transformer autocast only. Weights, embeddings, grammar/value heads,
    # probability arithmetic, and optimizer state remain FP32. CPU/MPS use FP32.
    precision: str = "fp32"

    def __post_init__(self) -> None:
        if self.precision not in ("fp32", "bf16"):
            raise ValueError("precision must be 'fp32' or 'bf16'")


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

        self.action_grammar_head = ActionGrammarHead(
            d,
            danger_shop_leave_logit_penalty=config.danger_shop_leave_logit_penalty,
        )

        # Value head
        self.value_head = ValueHead(d, win_only=config.win_only_value)

    def forward(
        self,
        tokens: torch.Tensor,
        token_types: torch.Tensor,
        scalars: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        history_events: torch.Tensor | None = None,
        history_event_features: torch.Tensor | None = None,
        history_cards: torch.Tensor | None = None,
        history_card_mask: torch.Tensor | None = None,
        history_jokers: torch.Tensor | None = None,
        history_joker_mask: torch.Tensor | None = None,
        history_event_mask: torch.Tensor | None = None,
        history_round_mask: torch.Tensor | None = None,
        history_omitted: torch.Tensor | None = None,
        temperature: float | torch.Tensor = 1.0,
        hand_ar_mixture_eps: float | None = None,
        return_raw_outputs: bool = False,
        critic_only: bool = False,
    ) -> tuple[ActionGrammarDistribution | dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Return policy/value predictions; critic_only freezes the encoder and skips policy work."""
        if not critic_only and action_mask is None:
            raise ValueError("Policy inference requires action_mask")
        eps = self.config.hand_ar_mixture_eps if hand_ar_mixture_eps is None else hand_ar_mixture_eps
        if history_events is not None and history_round_mask is not None:
            attention_mask = attention_mask.clone()
            attention_mask[:, HISTORY_START : HISTORY_START + HISTORY_ROUNDS] = history_round_mask
        # Terminal replay updates only critic heads. Avoid constructing the
        # unused encoder graph and policy heads for those batches.
        with torch.set_grad_enabled(torch.is_grad_enabled() and not critic_only):
            x = self.embedding(
                tokens,
                token_types,
                scalars,
                history_events,
                history_event_features,
                history_cards,
                history_card_mask,
                history_jokers,
                history_joker_mask,
                history_event_mask,
                history_omitted,
            )
            with backbone_autocast(self.config.precision, x.device):
                x = self.backbone(x, padding_mask=(attention_mask == 0))
        # Keep small policy differences, hazard products, and PPO log-ratios
        # away from BF16 rounding. Backprop still traverses the autocast trunk.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            grammar_output = (
                None if critic_only else self.action_grammar_head(x, attention_mask, tokens, token_types, scalars)
            )
            value_dict = self.value_head(
                x,
                attention_mask,
                current_antes=scalars[:, CURRENT_ANTE_SCALAR_INDEX],
                win_antes=scalars[:, WIN_ANTE_SCALAR_INDEX],
            )
        if critic_only:
            return {}, value_dict
        assert grammar_output is not None
        if return_raw_outputs:
            # DataParallel can gather nested tensor containers, but not an
            # ActionGrammarDistribution (which also closes over the unsharded
            # action mask/tokens). Reconstruct the distribution after gather.
            return {
                "macro_logits": grammar_output.macro_logits,
                "hand_count_logits": grammar_output.hand_count_logits,
                "hand_card_logits": grammar_output.hand_card_logits,
                "candidate_play_logits": grammar_output.candidate_play_logits,
                "candidate_discard_logits": grammar_output.candidate_discard_logits,
                "consumable_slot_logits": grammar_output.consumable_slot_logits,
                "consumable_count_logits": grammar_output.consumable_count_logits,
                "consumable_card_logits": grammar_output.consumable_card_logits,
                "consumable_joker_logits": grammar_output.consumable_joker_logits,
                "shop_buy_logits": grammar_output.shop_buy_logits,
                "shop_sell_joker_logits": grammar_output.shop_sell_joker_logits,
                "shop_sell_consumable_logits": grammar_output.shop_sell_consumable_logits,
                "pack_claim_logits": grammar_output.pack_claim_logits,
            }, value_dict
        return (
            ActionGrammarDistribution(
                grammar_output,
                action_mask,
                temperature=temperature,
                tokens=tokens,
                hand_ar_mixture_eps=eps,
            ),
            value_dict,
        )

    def action_distribution(
        self,
        tokens: torch.Tensor,
        token_types: torch.Tensor,
        scalars: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
        history_events: torch.Tensor | None = None,
        history_event_features: torch.Tensor | None = None,
        history_cards: torch.Tensor | None = None,
        history_card_mask: torch.Tensor | None = None,
        history_jokers: torch.Tensor | None = None,
        history_joker_mask: torch.Tensor | None = None,
        history_event_mask: torch.Tensor | None = None,
        history_round_mask: torch.Tensor | None = None,
        history_omitted: torch.Tensor | None = None,
        temperature: float | torch.Tensor = 1.0,
        hand_ar_mixture_eps: float | None = None,
    ) -> tuple[ActionGrammarDistribution, dict[str, torch.Tensor]]:
        """Named entry point used by rollout and training code."""
        return self(
            tokens,
            token_types,
            scalars,
            attention_mask,
            action_mask,
            history_events,
            history_event_features,
            history_cards,
            history_card_mask,
            history_jokers,
            history_joker_mask,
            history_event_mask,
            history_round_mask,
            history_omitted,
            temperature=temperature,
            hand_ar_mixture_eps=hand_ar_mixture_eps,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
