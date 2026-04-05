"""Action heads for each sub-phase of the game."""

from __future__ import annotations

import torch
import torch.nn as nn

from .constants import (
    BLIND_SELECT_START,
    CONSUMABLE_START,
    DECK_MAX,
    DECK_START,
    JOKER_START,
    MAX_CONSUMABLE_SLOTS,
    MAX_HAND_SIZE,
    MAX_JOKER_SLOTS,
    MAX_PACK_CARDS,
    SCALAR_DIM,
    MAX_SHOP_ITEMS,
    NUM_ACTIONS,
    SHOP_START,
    ActionRange,
    TokenType,
)
from .vocab import Vocab


class BlindSelectHead(nn.Module):
    """Produces logits for blind_play, blind_skip, blind_reroll."""

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

    def forward(self, backbone_out: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (batch, NUM_ACTIONS)."""
        batch = backbone_out.shape[0]
        logits = torch.full((batch, NUM_ACTIONS), -1e8, device=backbone_out.device)

        # Pool blind_select tokens (positions 99-101)
        blind_tokens = backbone_out[:, BLIND_SELECT_START:BLIND_SELECT_START + 3]
        blind_mask = attention_mask[:, BLIND_SELECT_START:BLIND_SELECT_START + 3].unsqueeze(-1).float()
        blind_pool = (blind_tokens * blind_mask).sum(1) / blind_mask.sum(1).clamp(min=1)

        # Global mean pool
        full_mask = attention_mask.unsqueeze(-1).float()
        global_pool = (backbone_out * full_mask).sum(1) / full_mask.sum(1).clamp(min=1)

        combined = torch.cat([blind_pool, global_pool], dim=-1)
        head_logits = self.mlp(combined)  # (batch, 3)

        logits[:, ActionRange.BLIND_PLAY] = head_logits[:, 0]
        logits[:, ActionRange.BLIND_SKIP] = head_logits[:, 1]
        logits[:, ActionRange.BLIND_REROLL] = head_logits[:, 2]
        return logits


class HandPlayHead(nn.Module):
    """Chooses action mode, then scores linear card toggles in SELECT_CARDS."""

    def __init__(self, d_model: int, vocab: Vocab):
        super().__init__()
        _ = vocab
        state_dim = d_model + 16
        self.pending_action_emb = nn.Embedding(3, 16)
        self.mode_mlp = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )
        self.card_state_proj = nn.Linear(state_dim, d_model)
        self.card_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.confirm_cancel_mlp = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(
        self,
        backbone_out: torch.Tensor,
        attention_mask: torch.Tensor,
        tokens: torch.Tensor,
        token_types: torch.Tensor,
        scalars: torch.Tensor,
        select_mode: bool = False,
    ) -> torch.Tensor:
        batch = backbone_out.shape[0]
        logits = torch.full((batch, NUM_ACTIONS), -1e8, device=backbone_out.device)

        full_mask = attention_mask.unsqueeze(-1).float()
        global_pool = (backbone_out * full_mask).sum(1) / full_mask.sum(1).clamp(min=1)
        pending_action = scalars[:, 8].long().clamp(0, 2)
        state_features = torch.cat([global_pool, self.pending_action_emb(pending_action)], dim=-1)

        if not select_mode:
            mode_logits = self.mode_mlp(state_features)
            logits[:, ActionRange.PLAY_SELECTION] = mode_logits[:, 0]
            logits[:, ActionRange.DISCARD_SELECTION] = mode_logits[:, 1]
            logits[:, ActionRange.USE_CONSUMABLE] = mode_logits[:, 2]
            return logits

        hand_slots, hand_present = self._gather_hand_slots(backbone_out, attention_mask, tokens, token_types)
        card_state = self.card_state_proj(state_features).unsqueeze(1).expand(-1, MAX_HAND_SIZE, -1)
        card_scores = self.card_proj(torch.cat([hand_slots, card_state], dim=-1)).squeeze(-1)
        for i in range(MAX_HAND_SIZE):
            logits[:, ActionRange.SELECT_CARD_START + i] = card_scores[:, i]

        confirm_cancel = self.confirm_cancel_mlp(state_features)
        logits[:, ActionRange.SELECTION_CONFIRM] = confirm_cancel[:, 0]
        logits[:, ActionRange.SELECTION_CANCEL] = confirm_cancel[:, 1]
        return logits

    def _gather_hand_slots(
        self,
        backbone_out: torch.Tensor,
        attention_mask: torch.Tensor,
        tokens: torch.Tensor,
        token_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        deck_slice = slice(DECK_START, DECK_START + DECK_MAX)
        deck_out = backbone_out[:, deck_slice]
        deck_tokens = tokens[:, deck_slice]
        deck_types = token_types[:, deck_slice]
        deck_mask = attention_mask[:, deck_slice].bool()

        hand_card_mask = deck_mask & deck_types.eq(int(TokenType.DECK)) & deck_tokens[:, :, 5].eq(0)
        hand_slots = deck_tokens[:, :, 11].clamp(0, MAX_HAND_SIZE - 1).long()

        batch, _, d_model = deck_out.shape
        gathered = torch.zeros(batch, MAX_HAND_SIZE, d_model, device=deck_out.device, dtype=deck_out.dtype)
        present = torch.zeros(batch, MAX_HAND_SIZE, device=deck_out.device, dtype=deck_out.dtype)
        ctx_index = hand_slots.unsqueeze(-1).expand(-1, -1, d_model)
        gathered.scatter_add_(1, ctx_index, deck_out * hand_card_mask.unsqueeze(-1))
        present.scatter_add_(1, hand_slots, hand_card_mask.to(deck_out.dtype))
        present.clamp_(0.0, 1.0)
        return gathered * present.unsqueeze(-1), present


class ShopHead(nn.Module):
    """Produces logits for shop actions: buy, reroll, sell, leave."""

    def __init__(self, d_model: int):
        super().__init__()
        self.item_proj = nn.Linear(d_model, 1)
        self.global_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1 + MAX_JOKER_SLOTS + MAX_CONSUMABLE_SLOTS + 1),
            # reroll + sell_joker*5 + sell_cons*5 + leave
        )

    def forward(self, backbone_out: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch = backbone_out.shape[0]
        logits = torch.full((batch, NUM_ACTIONS), -1e8, device=backbone_out.device)

        # Per-shop-item scores
        shop_tokens = backbone_out[:, SHOP_START:SHOP_START + MAX_SHOP_ITEMS]
        item_scores = self.item_proj(shop_tokens).squeeze(-1)  # (batch, MAX_SHOP_ITEMS)
        for i in range(MAX_SHOP_ITEMS):
            logits[:, ActionRange.SHOP_BUY_START + i] = item_scores[:, i]

        # Global pool for reroll/sell/leave
        full_mask = attention_mask.unsqueeze(-1).float()
        global_pool = (backbone_out * full_mask).sum(1) / full_mask.sum(1).clamp(min=1)
        global_logits = self.global_mlp(global_pool)  # (batch, 1+5+5+1)

        logits[:, ActionRange.SHOP_REROLL] = global_logits[:, 0]
        for i in range(MAX_JOKER_SLOTS):
            logits[:, ActionRange.SHOP_SELL_JOKER_START + i] = global_logits[:, 1 + i]
        # Only use first 5 of the joker slots for sell
        for i in range(MAX_CONSUMABLE_SLOTS):
            logits[:, ActionRange.SHOP_SELL_CONSUMABLE_START + i] = global_logits[:, 1 + MAX_JOKER_SLOTS + i]
        logits[:, ActionRange.SHOP_LEAVE] = global_logits[:, -1]

        return logits


class ConsumableHead(nn.Module):
    """Produces logits for consumable selection and targeting."""

    def __init__(self, d_model: int):
        super().__init__()
        self.slot_proj = nn.Linear(d_model, 1)
        self.hand_target_proj = nn.Linear(d_model, 1)
        self.joker_target_proj = nn.Linear(d_model, 1)
        self.confirm_cancel_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, backbone_out: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch = backbone_out.shape[0]
        logits = torch.full((batch, NUM_ACTIONS), -1e8, device=backbone_out.device)

        # Consumable slot scores
        cons_tokens = backbone_out[:, CONSUMABLE_START:CONSUMABLE_START + MAX_CONSUMABLE_SLOTS]
        slot_scores = self.slot_proj(cons_tokens).squeeze(-1)
        for i in range(MAX_CONSUMABLE_SLOTS):
            logits[:, ActionRange.CONSUMABLE_SLOT_START + i] = slot_scores[:, i]

        # Hand card target scores
        hand_tokens = backbone_out[:, DECK_START:DECK_START + MAX_HAND_SIZE]
        hand_scores = self.hand_target_proj(hand_tokens).squeeze(-1)
        for i in range(MAX_HAND_SIZE):
            logits[:, ActionRange.CONSUMABLE_HAND_TARGET_START + i] = hand_scores[:, i]

        # Joker target scores
        joker_tokens = backbone_out[:, JOKER_START:JOKER_START + MAX_JOKER_SLOTS]
        joker_scores = self.joker_target_proj(joker_tokens).squeeze(-1)
        for i in range(MAX_JOKER_SLOTS):
            logits[:, ActionRange.CONSUMABLE_JOKER_TARGET_START + i] = joker_scores[:, i]

        # Confirm/cancel
        full_mask = attention_mask.unsqueeze(-1).float()
        global_pool = (backbone_out * full_mask).sum(1) / full_mask.sum(1).clamp(min=1)
        cc = self.confirm_cancel_mlp(global_pool)
        logits[:, ActionRange.CONSUMABLE_CONFIRM] = cc[:, 0]
        logits[:, ActionRange.CONSUMABLE_CANCEL] = cc[:, 1]

        return logits


class PackHead(nn.Module):
    """Produces logits for booster pack: claim or skip."""

    def __init__(self, d_model: int):
        super().__init__()
        self.card_proj = nn.Linear(d_model, 1)
        self.skip_proj = nn.Linear(d_model, 1)

    def forward(self, backbone_out: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch = backbone_out.shape[0]
        logits = torch.full((batch, NUM_ACTIONS), -1e8, device=backbone_out.device)

        # Pack cards are in shop token positions when in booster pack phase
        # We reuse shop positions for pack card tokens
        shop_tokens = backbone_out[:, SHOP_START:SHOP_START + MAX_PACK_CARDS]
        card_scores = self.card_proj(shop_tokens).squeeze(-1)
        for i in range(MAX_PACK_CARDS):
            logits[:, ActionRange.PACK_CLAIM_START + i] = card_scores[:, i]

        full_mask = attention_mask.unsqueeze(-1).float()
        global_pool = (backbone_out * full_mask).sum(1) / full_mask.sum(1).clamp(min=1)
        logits[:, ActionRange.PACK_SKIP] = self.skip_proj(global_pool).squeeze(-1)

        return logits
