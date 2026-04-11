"""Action heads for each sub-phase of the game."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from .subset_actions import HAND_SUBSET_MASKS, HAND_SUBSET_SIZES
from .vocab import RANK_TO_ID, SEAL_TO_ID, Vocab


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
    """Scores exhaustive play/discard subsets for CHOOSE_ACTION."""

    def __init__(self, d_model: int, vocab: Vocab):
        super().__init__()
        ctx_dim = 32
        raw_dim = 16
        state_dim = 48
        hidden_dim = 64

        self.ctx_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, ctx_dim),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
        )
        self.scalar_proj = nn.Sequential(
            nn.Linear(SCALAR_DIM, 16),
            nn.GELU(),
        )

        self.rank_emb = nn.Embedding(vocab.rank_size, raw_dim)
        self.suit_emb = nn.Embedding(vocab.suit_size, raw_dim)
        self.enhancement_emb = nn.Embedding(vocab.enhancement_vocab_size, raw_dim)
        self.edition_emb = nn.Embedding(vocab.edition_size, raw_dim)
        self.seal_emb = nn.Embedding(vocab.seal_size, raw_dim)
        self.debuff_emb = nn.Embedding(2, raw_dim)
        self.face_down_emb = nn.Embedding(2, raw_dim)
        self.forced_emb = nn.Embedding(2, raw_dim)
        self.perma_proj = nn.Linear(1, raw_dim)

        self.play_state_proj = nn.Linear(state_dim, hidden_dim)
        self.play_selected_ctx_proj = nn.Linear(ctx_dim, hidden_dim)
        self.play_remaining_ctx_proj = nn.Linear(ctx_dim, hidden_dim)
        self.play_selected_raw_proj = nn.Linear(raw_dim, hidden_dim)
        self.play_remaining_raw_proj = nn.Linear(raw_dim, hidden_dim)
        self.play_numeric_proj = nn.Linear(17, hidden_dim)
        self.play_out = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.discard_state_proj = nn.Linear(state_dim, hidden_dim)
        self.discard_selected_ctx_proj = nn.Linear(ctx_dim, hidden_dim)
        self.discard_remaining_ctx_proj = nn.Linear(ctx_dim, hidden_dim)
        self.discard_selected_raw_proj = nn.Linear(raw_dim, hidden_dim)
        self.discard_remaining_raw_proj = nn.Linear(raw_dim, hidden_dim)
        self.discard_numeric_proj = nn.Linear(17, hidden_dim)
        self.discard_out = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.use_consumable_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.rank_vocab_size = vocab.rank_size
        self.suit_vocab_size = vocab.suit_size
        self.register_buffer("subset_masks", torch.from_numpy(HAND_SUBSET_MASKS), persistent=False)
        self.register_buffer("subset_sizes", torch.from_numpy(HAND_SUBSET_SIZES), persistent=False)

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
        if select_mode:
            return logits

        full_mask = attention_mask.unsqueeze(-1).float()
        global_pool = (backbone_out * full_mask).sum(1) / full_mask.sum(1).clamp(min=1)
        state_features = torch.cat([self.global_proj(global_pool), self.scalar_proj(scalars)], dim=-1)

        hand_ctx, hand_tokens, hand_present = self._gather_hand_slots(backbone_out, attention_mask, tokens, token_types)
        hand_ctx = self.ctx_proj(hand_ctx) * hand_present.unsqueeze(-1)
        hand_raw = self._embed_raw_hand_cards(hand_tokens, hand_present)

        subset_masks = self.subset_masks.to(device=backbone_out.device, dtype=backbone_out.dtype)
        subset_sizes = self.subset_sizes.to(device=backbone_out.device, dtype=backbone_out.dtype).unsqueeze(0)
        hand_counts = hand_present.sum(dim=1, keepdim=True)

        selected_ctx_sum = torch.einsum("sk,bkd->bsd", subset_masks, hand_ctx)
        remaining_ctx_sum = hand_ctx.sum(dim=1, keepdim=True) - selected_ctx_sum
        selected_ctx = selected_ctx_sum / subset_sizes.unsqueeze(-1).clamp(min=1.0)
        remaining_ctx = remaining_ctx_sum / (hand_counts - subset_sizes).clamp(min=1.0).unsqueeze(-1)

        selected_raw_sum = torch.einsum("sk,bkd->bsd", subset_masks, hand_raw)
        remaining_raw_sum = hand_raw.sum(dim=1, keepdim=True) - selected_raw_sum
        selected_raw = selected_raw_sum / subset_sizes.unsqueeze(-1).clamp(min=1.0)
        remaining_raw = remaining_raw_sum / (hand_counts - subset_sizes).clamp(min=1.0).unsqueeze(-1)

        numeric_features = self._build_numeric_features(hand_tokens, hand_present, subset_masks, subset_sizes)

        play_hidden = self.play_state_proj(state_features).unsqueeze(1)
        play_hidden = play_hidden + self.play_selected_ctx_proj(selected_ctx)
        play_hidden = play_hidden + self.play_remaining_ctx_proj(remaining_ctx)
        play_hidden = play_hidden + self.play_selected_raw_proj(selected_raw)
        play_hidden = play_hidden + self.play_remaining_raw_proj(remaining_raw)
        play_hidden = play_hidden + self.play_numeric_proj(numeric_features)
        play_scores = self.play_out(play_hidden).squeeze(-1)

        discard_hidden = self.discard_state_proj(state_features).unsqueeze(1)
        discard_hidden = discard_hidden + self.discard_selected_ctx_proj(selected_ctx)
        discard_hidden = discard_hidden + self.discard_remaining_ctx_proj(remaining_ctx)
        discard_hidden = discard_hidden + self.discard_selected_raw_proj(selected_raw)
        discard_hidden = discard_hidden + self.discard_remaining_raw_proj(remaining_raw)
        discard_hidden = discard_hidden + self.discard_numeric_proj(numeric_features)
        discard_scores = self.discard_out(discard_hidden).squeeze(-1)

        logits[:, ActionRange.PLAY_SUBSET_START:ActionRange.PLAY_SUBSET_END + 1] = play_scores
        logits[:, ActionRange.DISCARD_SUBSET_START:ActionRange.DISCARD_SUBSET_END + 1] = discard_scores
        logits[:, ActionRange.USE_CONSUMABLE] = self.use_consumable_proj(state_features).squeeze(-1)
        return logits

    def _gather_hand_slots(
        self,
        backbone_out: torch.Tensor,
        attention_mask: torch.Tensor,
        tokens: torch.Tensor,
        token_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        deck_slice = slice(DECK_START, DECK_START + DECK_MAX)
        deck_out = backbone_out[:, deck_slice]
        deck_tokens = tokens[:, deck_slice]
        deck_types = token_types[:, deck_slice]
        deck_mask = attention_mask[:, deck_slice].bool()

        hand_card_mask = deck_mask & deck_types.eq(int(TokenType.DECK)) & deck_tokens[:, :, 5].eq(0)
        hand_slots = deck_tokens[:, :, 11].clamp(0, MAX_HAND_SIZE - 1).long()

        batch, _, d_model = deck_out.shape
        token_dim = deck_tokens.shape[-1]
        hand_ctx = torch.zeros(batch, MAX_HAND_SIZE, d_model, device=deck_out.device, dtype=deck_out.dtype)
        hand_raw_tokens = torch.zeros(batch, MAX_HAND_SIZE, token_dim, device=tokens.device, dtype=tokens.dtype)
        hand_present = torch.zeros(batch, MAX_HAND_SIZE, device=deck_out.device, dtype=deck_out.dtype)

        ctx_index = hand_slots.unsqueeze(-1).expand(-1, -1, d_model)
        token_index = hand_slots.unsqueeze(-1).expand(-1, -1, token_dim)
        hand_ctx.scatter_add_(1, ctx_index, deck_out * hand_card_mask.unsqueeze(-1))
        hand_raw_tokens.scatter_add_(
            1,
            token_index,
            deck_tokens * hand_card_mask.unsqueeze(-1).to(deck_tokens.dtype),
        )
        hand_present.scatter_add_(1, hand_slots, hand_card_mask.to(deck_out.dtype))
        hand_present.clamp_(0.0, 1.0)

        return hand_ctx, hand_raw_tokens, hand_present

    def _embed_raw_hand_cards(self, hand_tokens: torch.Tensor, hand_present: torch.Tensor) -> torch.Tensor:
        raw = (
            self.rank_emb(hand_tokens[:, :, 0].clamp(0, self.rank_emb.num_embeddings - 1))
            + self.suit_emb(hand_tokens[:, :, 1].clamp(0, self.suit_emb.num_embeddings - 1))
            + self.enhancement_emb(hand_tokens[:, :, 2].clamp(0, self.enhancement_emb.num_embeddings - 1))
            + self.edition_emb(hand_tokens[:, :, 3].clamp(0, self.edition_emb.num_embeddings - 1))
            + self.seal_emb(hand_tokens[:, :, 4].clamp(0, self.seal_emb.num_embeddings - 1))
            + self.debuff_emb(hand_tokens[:, :, 6].clamp(0, 1))
            + self.face_down_emb(hand_tokens[:, :, 7].clamp(0, 1))
            + self.forced_emb(hand_tokens[:, :, 10].clamp(0, 1))
            + self.perma_proj(hand_tokens[:, :, 8].float().unsqueeze(-1) / 31.0)
        )
        return raw * hand_present.unsqueeze(-1)

    def _build_numeric_features(
        self,
        hand_tokens: torch.Tensor,
        hand_present: torch.Tensor,
        subset_masks: torch.Tensor,
        subset_sizes: torch.Tensor,
    ) -> torch.Tensor:
        dtype = subset_masks.dtype
        present = hand_present.to(dtype)
        rank_ids = hand_tokens[:, :, 0].long().clamp(0, self.rank_vocab_size - 1)
        suit_ids = hand_tokens[:, :, 1].long().clamp(0, self.suit_vocab_size - 1)
        seal_ids = hand_tokens[:, :, 4].long()

        forced = hand_tokens[:, :, 10].to(dtype) * present
        debuffed = hand_tokens[:, :, 6].to(dtype) * present
        face = ((rank_ids >= RANK_TO_ID["J"]) & (rank_ids <= RANK_TO_ID["K"])).to(dtype) * present
        kings = rank_ids.eq(RANK_TO_ID["K"]).to(dtype) * present
        red_seals = seal_ids.eq(SEAL_TO_ID["Red"]).to(dtype) * present
        blue_seals = seal_ids.eq(SEAL_TO_ID["Blue"]).to(dtype) * present
        gold_seals = seal_ids.eq(SEAL_TO_ID["Gold"]).to(dtype) * present
        perma_bonus = (hand_tokens[:, :, 8].to(dtype) / 31.0) * present

        selected_forced = torch.einsum("sk,bk->bs", subset_masks, forced)
        selected_debuff = torch.einsum("sk,bk->bs", subset_masks, debuffed)
        selected_face = torch.einsum("sk,bk->bs", subset_masks, face)
        selected_kings = torch.einsum("sk,bk->bs", subset_masks, kings)
        selected_red = torch.einsum("sk,bk->bs", subset_masks, red_seals)
        selected_blue = torch.einsum("sk,bk->bs", subset_masks, blue_seals)
        selected_gold = torch.einsum("sk,bk->bs", subset_masks, gold_seals)
        selected_perma = torch.einsum("sk,bk->bs", subset_masks, perma_bonus)

        total_blue = blue_seals.sum(dim=1, keepdim=True)
        total_gold = gold_seals.sum(dim=1, keepdim=True)
        total_perma = perma_bonus.sum(dim=1, keepdim=True)

        rank_hist = F.one_hot(rank_ids, num_classes=self.rank_vocab_size).to(dtype) * present.unsqueeze(-1)
        suit_hist = F.one_hot(suit_ids, num_classes=self.suit_vocab_size).to(dtype) * present.unsqueeze(-1)
        selected_rank_hist = torch.einsum("sk,bkr->bsr", subset_masks, rank_hist)
        selected_suit_hist = torch.einsum("sk,bkc->bsc", subset_masks, suit_hist)
        distinct_ranks = selected_rank_hist.gt(0).to(dtype).sum(dim=-1)
        max_rank_mult = selected_rank_hist.max(dim=-1).values
        distinct_suits = selected_suit_hist.gt(0).to(dtype).sum(dim=-1)
        max_suit_mult = selected_suit_hist.max(dim=-1).values

        hand_counts = present.sum(dim=1, keepdim=True)
        selected_counts = subset_sizes.expand(hand_counts.shape[0], -1)
        remaining_counts = (hand_counts - subset_sizes).clamp(min=0.0)

        features = torch.stack(
            [
                selected_counts / 5.0,
                remaining_counts / MAX_HAND_SIZE,
                selected_forced / 5.0,
                selected_debuff / 5.0,
                selected_face / 5.0,
                selected_kings / 5.0,
                selected_red / 5.0,
                selected_blue / 5.0,
                selected_gold / 5.0,
                (total_blue - selected_blue) / MAX_HAND_SIZE,
                (total_gold - selected_gold) / MAX_HAND_SIZE,
                selected_perma / 5.0,
                (total_perma - selected_perma) / MAX_HAND_SIZE,
                distinct_ranks / 5.0,
                max_rank_mult / 5.0,
                distinct_suits / 5.0,
                max_suit_mult / 5.0,
            ],
            dim=-1,
        )
        return features


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
