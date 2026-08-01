"""Token embedding modules for the Balatro transformer agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .constants import (
    BLIND_SELECT_MAX,
    BLIND_SELECT_START,
    CONSUMABLE_MAX,
    CONSUMABLE_START,
    DECK_MAX,
    DECK_START,
    HAND_CANDIDATE_MAX,
    HAND_CANDIDATE_START,
    HAND_LEVEL_MAX,
    HAND_LEVEL_START,
    JOKER_MAX,
    JOKER_START,
    MAX_HAND_SIZE,
    MAX_SEQ_LEN,
    META_COUNT,
    META_START,
    OBJ_START,
    SHOP_MAX,
    SHOP_START,
    VOUCHER_MAX,
    VOUCHER_START,
)

if TYPE_CHECKING:
    from .vocab import Vocab


class ObjEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(3, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (batch, 1, TOKEN_DIM) -> (batch, 1, d_model)."""
        return self.emb(tokens[:, :, 0].clamp(0, 2))


class MetaEmbedding(nn.Module):
    """Embeds the 9 META tokens. Uses linear projections for continuous values."""

    def __init__(self, d_model: int):
        super().__init__()
        # Each meta token gets its own projection
        self.money_proj = nn.Linear(1, d_model)
        self.interest_proj = nn.Linear(1, d_model)
        self.ante_emb = nn.Embedding(12, d_model)
        self.blind_type_emb = nn.Embedding(4, d_model)
        self.boss_emb = nn.Embedding(35, d_model)  # 35 = fixed cap on boss vocab
        self.target_proj = nn.Linear(4, d_model)
        self.hands_proj = nn.Linear(1, d_model)
        self.discards_proj = nn.Linear(1, d_model)
        self.handsize_proj = nn.Linear(1, d_model)
        self.phase_emb = nn.Embedding(4, d_model)

    def forward(self, tokens: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """tokens: (batch, META_COUNT, TOKEN_DIM), scalars: (batch, SCALAR_DIM)."""
        batch = tokens.shape[0]
        d = self.money_proj.out_features
        out = torch.zeros(batch, META_COUNT, d, device=tokens.device, dtype=torch.float32)

        out[:, 0] = self.money_proj(scalars[:, 0:1])
        out[:, 1] = self.interest_proj(scalars[:, 1:2])
        out[:, 2] = self.ante_emb(scalars[:, 2].long().clamp(0, 11))
        # Blind type + boss from token
        # meta token packs blind_type*100 + boss_id (see tokenizer._encode_meta)
        bt = tokens[:, 3, 0] // 100
        boss = tokens[:, 3, 0] % 100
        out[:, 3] = self.blind_type_emb(bt.clamp(0, 3)) + self.boss_emb(boss.clamp(0, 34))
        target_context = torch.cat(
            [
                scalars[:, 3:4],  # blind target
                scalars[:, 8:9],  # current round score
                scalars[:, 9:10],  # remaining score needed
                scalars[:, 10:11],  # blind progress ratio
            ],
            dim=-1,
        )
        out[:, 4] = self.target_proj(target_context)
        out[:, 5] = self.hands_proj(scalars[:, 4:5])
        out[:, 6] = self.discards_proj(scalars[:, 5:6])
        out[:, 7] = self.handsize_proj(scalars[:, 6:7])
        out[:, 8] = self.phase_emb(scalars[:, 7].long().clamp(0, 3))
        return out


class DeckCardEmbedding(nn.Module):
    """Embedding for deck cards using sum of feature embeddings."""

    def __init__(self, vocab: Vocab, d_model: int, hand_slot_emb: nn.Embedding):
        super().__init__()
        self.rank_emb = nn.Embedding(vocab.rank_size, d_model)
        self.suit_emb = nn.Embedding(vocab.suit_size, d_model)
        self.enhancement_emb = nn.Embedding(vocab.enhancement_vocab_size, d_model)
        self.edition_emb = nn.Embedding(vocab.edition_size, d_model)
        self.seal_emb = nn.Embedding(vocab.seal_size, d_model)
        self.location_emb = nn.Embedding(4, d_model)
        self.hand_slot_emb = hand_slot_emb
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (batch, num_cards, TOKEN_DIM)."""
        rank = tokens[:, :, 0].clamp(0, self.rank_emb.num_embeddings - 1)
        suit = tokens[:, :, 1].clamp(0, self.suit_emb.num_embeddings - 1)
        enh = tokens[:, :, 2].clamp(0, self.enhancement_emb.num_embeddings - 1)
        ed = tokens[:, :, 3].clamp(0, self.edition_emb.num_embeddings - 1)
        seal = tokens[:, :, 4].clamp(0, self.seal_emb.num_embeddings - 1)
        loc = tokens[:, :, 5].clamp(0, 3)
        slot = tokens[:, :, 11].clamp(0, self.hand_slot_emb.num_embeddings - 1)

        h = (
            self.rank_emb(rank)
            + self.suit_emb(suit)
            + self.enhancement_emb(enh)
            + self.edition_emb(ed)
            + self.seal_emb(seal)
            + self.location_emb(loc)
            + self.hand_slot_emb(slot)
        )
        return self.proj(h)


class JokerEmbedding(nn.Module):
    def __init__(self, vocab: Vocab, d_model: int):
        super().__init__()
        self.id_emb = nn.Embedding(vocab.joker_vocab_size, d_model)
        self.rarity_emb = nn.Embedding(5, d_model)
        self.edition_emb = nn.Embedding(vocab.edition_size, d_model)
        self.sell_proj = nn.Linear(1, d_model)
        self.counter_proj = nn.Linear(1, d_model)
        self.slot_emb = nn.Embedding(8, d_model)
        self.eternal_emb = nn.Embedding(2, d_model)
        self.perishable_emb = nn.Embedding(2, d_model)
        self.rental_emb = nn.Embedding(2, d_model)
        self.debuff_emb = nn.Embedding(2, d_model)
        self.perish_tally_proj = nn.Linear(1, d_model)
        nn.init.zeros_(self.rental_emb.weight)
        nn.init.zeros_(self.debuff_emb.weight)
        nn.init.zeros_(self.perish_tally_proj.weight)
        nn.init.zeros_(self.perish_tally_proj.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        jid = tokens[:, :, 0].clamp(0, self.id_emb.num_embeddings - 1)
        rar = tokens[:, :, 1].clamp(0, 4)
        ed = tokens[:, :, 2].clamp(0, self.edition_emb.num_embeddings - 1)
        sell = tokens[:, :, 3].float().unsqueeze(-1)
        counter = tokens[:, :, 4].float().unsqueeze(-1)
        slot = tokens[:, :, 5].clamp(0, 7)
        eternal = tokens[:, :, 6].clamp(0, 1)
        perishable = tokens[:, :, 7].clamp(0, 1)
        rental = tokens[:, :, 8].clamp(0, 1)
        debuff = tokens[:, :, 9].clamp(0, 1)
        perish_tally = tokens[:, :, 10].float().unsqueeze(-1)

        return (
            self.id_emb(jid)
            + self.rarity_emb(rar)
            + self.edition_emb(ed)
            + self.sell_proj(sell)
            + self.counter_proj(counter)
            + self.slot_emb(slot)
            + self.eternal_emb(eternal)
            + self.perishable_emb(perishable)
            + self.rental_emb(rental)
            + self.debuff_emb(debuff)
            + self.perish_tally_proj(perish_tally)
        )


class VoucherEmbedding(nn.Module):
    def __init__(self, vocab: Vocab, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab.voucher_vocab_size, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.emb(tokens[:, :, 0].clamp(0, self.emb.num_embeddings - 1))


class ConsumableEmbedding(nn.Module):
    def __init__(self, vocab: Vocab, d_model: int):
        super().__init__()
        self.type_emb = nn.Embedding(4, d_model)
        self.id_emb = nn.Embedding(vocab.consumable_vocab_size, d_model)
        self.edition_emb = nn.Embedding(vocab.edition_size, d_model)
        self.slot_emb = nn.Embedding(6, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        typ = tokens[:, :, 0].clamp(0, 3)
        cid = tokens[:, :, 1].clamp(0, self.id_emb.num_embeddings - 1)
        ed = tokens[:, :, 2].clamp(0, self.edition_emb.num_embeddings - 1)
        slot = tokens[:, :, 3].clamp(0, 5)
        return self.type_emb(typ) + self.id_emb(cid) + self.edition_emb(ed) + self.slot_emb(slot)


class ShopEmbedding(nn.Module):
    def __init__(self, vocab: Vocab, d_model: int):
        super().__init__()
        # Reuse joker/consumable/voucher/booster embeddings via a max-vocab embedding
        max_vocab = max(
            vocab.joker_vocab_size,
            vocab.consumable_vocab_size,
            vocab.voucher_vocab_size,
            vocab.booster_vocab_size,
            2,
        )
        self.item_emb = nn.Embedding(max_vocab, d_model)
        self.type_emb = nn.Embedding(6, d_model)
        self.price_proj = nn.Linear(1, d_model)
        self.edition_emb = nn.Embedding(vocab.edition_size, d_model)
        self.slot_emb = nn.Embedding(10, d_model)
        self.seal_emb = nn.Embedding(vocab.seal_size, d_model)
        self.eternal_emb = nn.Embedding(2, d_model)
        self.perishable_emb = nn.Embedding(2, d_model)
        self.rental_emb = nn.Embedding(2, d_model)
        self.rank_emb = nn.Embedding(vocab.rank_size, d_model)
        self.suit_emb = nn.Embedding(vocab.suit_size, d_model)
        nn.init.zeros_(self.seal_emb.weight)
        nn.init.zeros_(self.eternal_emb.weight)
        nn.init.zeros_(self.perishable_emb.weight)
        nn.init.zeros_(self.rental_emb.weight)
        nn.init.zeros_(self.rank_emb.weight)
        nn.init.zeros_(self.suit_emb.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        iid = tokens[:, :, 0].clamp(0, self.item_emb.num_embeddings - 1)
        itype = tokens[:, :, 1].clamp(0, 5)
        price = tokens[:, :, 2].float().unsqueeze(-1)
        ed = tokens[:, :, 3].clamp(0, self.edition_emb.num_embeddings - 1)
        slot = tokens[:, :, 4].clamp(0, 9)
        seal = tokens[:, :, 5].clamp(0, self.seal_emb.num_embeddings - 1)
        eternal = tokens[:, :, 6].clamp(0, 1)
        perishable = tokens[:, :, 7].clamp(0, 1)
        rental = tokens[:, :, 8].clamp(0, 1)
        rank = tokens[:, :, 9].clamp(0, self.rank_emb.num_embeddings - 1)
        suit = tokens[:, :, 10].clamp(0, self.suit_emb.num_embeddings - 1)
        return (
            self.item_emb(iid)
            + self.type_emb(itype)
            + self.price_proj(price)
            + self.edition_emb(ed)
            + self.slot_emb(slot)
            + self.seal_emb(seal)
            + self.eternal_emb(eternal)
            + self.perishable_emb(perishable)
            + self.rental_emb(rental)
            + self.rank_emb(rank)
            + self.suit_emb(suit)
        )


class BlindSelectEmbedding(nn.Module):
    def __init__(self, vocab: Vocab, d_model: int):
        super().__init__()
        self.blind_type_emb = nn.Embedding(4, d_model)
        self.boss_emb = nn.Embedding(vocab.boss_vocab_size, d_model)
        self.state_emb = nn.Embedding(5, d_model)
        self.mult_proj = nn.Linear(1, d_model)
        self.tag_emb = nn.Embedding(vocab.tag_vocab_size, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bt = tokens[:, :, 0].clamp(0, 3)
        st = tokens[:, :, 1].clamp(0, 4)
        mult = tokens[:, :, 2].float().unsqueeze(-1) / 10.0
        boss = tokens[:, :, 3].clamp(0, self.boss_emb.num_embeddings - 1)
        tag = tokens[:, :, 4].clamp(0, self.tag_emb.num_embeddings - 1)
        return (
            self.blind_type_emb(bt)
            + self.boss_emb(boss)
            + self.state_emb(st)
            + self.mult_proj(mult)
            + self.tag_emb(tag)
        )


class HandLevelEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.hand_type_emb = nn.Embedding(13, d_model)
        self.level_emb = nn.Embedding(16, d_model)
        self.chips_proj = nn.Linear(1, d_model)
        self.mult_proj = nn.Linear(1, d_model)
        self.played_proj = nn.Linear(1, d_model)
        self.visible_emb = nn.Embedding(2, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hand_type = tokens[:, :, 0].clamp(0, 12)
        level = tokens[:, :, 1].clamp(0, 15)
        chips = tokens[:, :, 2].float().unsqueeze(-1)
        mult = tokens[:, :, 3].float().unsqueeze(-1)
        played = tokens[:, :, 4].float().unsqueeze(-1)
        visible = tokens[:, :, 5].clamp(0, 1)
        return (
            self.hand_type_emb(hand_type)
            + self.level_emb(level)
            + self.chips_proj(chips)
            + self.mult_proj(mult)
            + self.played_proj(played)
            + self.visible_emb(visible)
        )


class HandCandidateEmbedding(nn.Module):
    def __init__(self, d_model: int, hand_slot_emb: nn.Embedding):
        super().__init__()
        self.kind_emb = nn.Embedding(3, d_model)
        self.hand_type_emb = nn.Embedding(13, d_model)
        self.size_emb = nn.Embedding(6, d_model)
        self.score_proj = nn.Linear(1, d_model)
        self.coverage_proj = nn.Linear(1, d_model)
        self.rank_slot_emb = nn.Embedding(HAND_CANDIDATE_MAX + 1, d_model)
        self.forced_proj = nn.Linear(1, d_model)
        self.hand_slot_emb = hand_slot_emb

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        kind = tokens[:, :, 0].clamp(0, 2)
        hand_type = tokens[:, :, 1].clamp(0, 12)
        size = tokens[:, :, 2].clamp(0, 5)
        score = tokens[:, :, 3].float().unsqueeze(-1)
        coverage = tokens[:, :, 4].float().unsqueeze(-1)
        rank_slot = tokens[:, :, 10].clamp(0, HAND_CANDIDATE_MAX)
        forced = tokens[:, :, 11].float().unsqueeze(-1)
        card_slots = tokens[:, :, 5:10].clamp(0, MAX_HAND_SIZE)
        slot_embed = self.hand_slot_emb(card_slots).sum(dim=-2)
        return (
            self.kind_emb(kind)
            + self.hand_type_emb(hand_type)
            + self.size_emb(size)
            + self.score_proj(score)
            + self.coverage_proj(coverage)
            + self.rank_slot_emb(rank_slot)
            + self.forced_proj(forced)
            + slot_embed
        )


class ContentEmbeddingLayer(nn.Module):
    """Routes each token type to its specialized embedding, then adds token type embedding."""

    def __init__(self, vocab: Vocab, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.token_type_emb = nn.Embedding(11, d_model)
        self.position_emb = nn.Embedding(MAX_SEQ_LEN, d_model)
        self.hand_slot_emb = nn.Embedding(MAX_HAND_SIZE + 1, d_model)

        self.obj_emb = ObjEmbedding(d_model)
        self.meta_emb = MetaEmbedding(d_model)
        self.deck_emb = DeckCardEmbedding(vocab, d_model, hand_slot_emb=self.hand_slot_emb)
        self.joker_emb = JokerEmbedding(vocab, d_model)
        self.voucher_emb = VoucherEmbedding(vocab, d_model)
        self.consumable_emb = ConsumableEmbedding(vocab, d_model)
        self.shop_emb = ShopEmbedding(vocab, d_model)
        self.blind_select_emb = BlindSelectEmbedding(vocab, d_model)
        self.hand_level_emb = HandLevelEmbedding(d_model)
        self.hand_candidate_emb = HandCandidateEmbedding(d_model, hand_slot_emb=self.hand_slot_emb)

    def forward(
        self,
        tokens: torch.Tensor,
        token_types: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (batch, MAX_SEQ_LEN, TOKEN_DIM) int
            token_types: (batch, MAX_SEQ_LEN) int
            scalars: (batch, SCALAR_DIM) float
        Returns:
            (batch, MAX_SEQ_LEN, d_model) float
        """
        batch = tokens.shape[0]
        device = tokens.device
        out = torch.zeros(batch, MAX_SEQ_LEN, self.d_model, device=device)

        # OBJ (position 0)
        out[:, OBJ_START : OBJ_START + 1] = self.obj_emb(tokens[:, OBJ_START : OBJ_START + 1])

        # META (positions 1-9)
        meta_tokens = tokens[:, META_START : META_START + META_COUNT]
        out[:, META_START : META_START + META_COUNT] = self.meta_emb(meta_tokens, scalars)

        # DECK (positions 10-71)
        deck_end = DECK_START + DECK_MAX
        out[:, DECK_START:deck_end] = self.deck_emb(tokens[:, DECK_START:deck_end])

        # JOKER (positions 72-79)
        joker_end = JOKER_START + JOKER_MAX
        out[:, JOKER_START:joker_end] = self.joker_emb(tokens[:, JOKER_START:joker_end])

        # VOUCHER (positions 80-83)
        voucher_end = VOUCHER_START + VOUCHER_MAX
        out[:, VOUCHER_START:voucher_end] = self.voucher_emb(tokens[:, VOUCHER_START:voucher_end])

        # CONSUMABLE (positions 84-88)
        cons_end = CONSUMABLE_START + CONSUMABLE_MAX
        out[:, CONSUMABLE_START:cons_end] = self.consumable_emb(tokens[:, CONSUMABLE_START:cons_end])

        # SHOP (positions 89-98)
        shop_end = SHOP_START + SHOP_MAX
        out[:, SHOP_START:shop_end] = self.shop_emb(tokens[:, SHOP_START:shop_end])

        # BLIND_SELECT (positions 99-101)
        blind_end = BLIND_SELECT_START + BLIND_SELECT_MAX
        out[:, BLIND_SELECT_START:blind_end] = self.blind_select_emb(tokens[:, BLIND_SELECT_START:blind_end])

        # HAND_LEVEL (positions 102-113)
        hand_level_end = HAND_LEVEL_START + HAND_LEVEL_MAX
        out[:, HAND_LEVEL_START:hand_level_end] = self.hand_level_emb(tokens[:, HAND_LEVEL_START:hand_level_end])

        # HAND_CANDIDATE (positions 114-145)
        hand_candidate_end = HAND_CANDIDATE_START + HAND_CANDIDATE_MAX
        out[:, HAND_CANDIDATE_START:hand_candidate_end] = self.hand_candidate_emb(
            tokens[:, HAND_CANDIDATE_START:hand_candidate_end]
        )

        # Add token type embedding
        positions = torch.arange(MAX_SEQ_LEN, device=device).unsqueeze(0)
        out = out + self.token_type_emb(token_types) + self.position_emb(positions)

        return out
