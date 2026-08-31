"""Structured action grammar for atomic Balatro actions.

The environment still consumes the existing flat action IDs. This module
samples and scores those IDs through a small grammar:

    action type -> parameters -> flat action id

For play/discard and hand-targeted consumables, card slots are selected
inside one policy call with an ordered pointer process. The env never sees
intermediate select/deselect states, so it cannot enter selection loops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .action import ActionType, decode_action
from .constants import (
    CONSUMABLE_ACTIONS_PER_SLOT,
    CONSUMABLE_HAND_SUBSET_OFFSET,
    CONSUMABLE_JOKER_OFFSET,
    CONSUMABLE_NO_TARGET_OFFSET,
    CONSUMABLE_START,
    DECK_MAX,
    DECK_START,
    HAND_CANDIDATE_MAX,
    HAND_CANDIDATE_START,
    JOKER_START,
    LEGACY_POLICY_SCALAR_DIM,
    MAX_CONSUMABLE_HAND_TARGETS,
    MAX_CONSUMABLE_SLOTS,
    MAX_HAND_SIZE,
    MAX_JOKER_SLOTS,
    MAX_PACK_CARDS,
    MAX_SHOP_ITEMS,
    NUM_ACTIONS,
    NUM_CONSUMABLE_HAND_SUBSETS,
    SHOP_START,
    ActionRange,
    TokenType,
)
from .subset_actions import (
    CONSUMABLE_HAND_SUBSET_BITS,
    CONSUMABLE_HAND_SUBSET_SIZES,
    CONSUMABLE_HAND_SUBSETS,
    HAND_SUBSET_BITS,
    HAND_SUBSET_SIZES,
    HAND_SUBSETS,
)

GRAMMAR_ACTION_TYPES: tuple[ActionType, ...] = tuple(ActionType)
NUM_GRAMMAR_ACTIONS = len(GRAMMAR_ACTION_TYPES)
ACTION_TYPE_TO_GRAMMAR_INDEX = {action_type: idx for idx, action_type in enumerate(GRAMMAR_ACTION_TYPES)}

_PLAY = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.PLAY_SUBSET]
_DISCARD = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.DISCARD_SUBSET]
_CONSUMABLE_NO_TARGET = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.USE_CONSUMABLE_NO_TARGET]
_CONSUMABLE_HAND = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.USE_CONSUMABLE_HAND_SUBSET]
_CONSUMABLE_JOKER = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.USE_CONSUMABLE_JOKER]
_SHOP_BUY = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.SHOP_BUY]
_SHOP_SELL_JOKER = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.SHOP_SELL_JOKER]
_SHOP_SELL_CONSUMABLE = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.SHOP_SELL_CONSUMABLE]
_PACK_CLAIM = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.PACK_CLAIM]
_SHOP_REROLL = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.SHOP_REROLL]
_SHOP_LEAVE = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.SHOP_LEAVE]

_SHOP_PHASE_ID = 2
_DANGER_THRESHOLD = 0.35

_ACTION_ID_TO_GRAMMAR = np.zeros(NUM_ACTIONS, dtype=np.int64)
_ACTION_ID_TO_INDEX = np.zeros(NUM_ACTIONS, dtype=np.int64)
_ACTION_ID_TO_DETAIL = np.zeros(NUM_ACTIONS, dtype=np.int64)
_ACTION_ID_TO_COUNT = np.ones(NUM_ACTIONS, dtype=np.int64)
for _action_id in range(NUM_ACTIONS):
    _decoded = decode_action(_action_id)
    _ACTION_ID_TO_GRAMMAR[_action_id] = ACTION_TYPE_TO_GRAMMAR_INDEX[_decoded.action_type]
    _ACTION_ID_TO_INDEX[_action_id] = _decoded.index
    _ACTION_ID_TO_DETAIL[_action_id] = _decoded.detail
    if _decoded.action_type in (ActionType.PLAY_SUBSET, ActionType.DISCARD_SUBSET):
        _ACTION_ID_TO_COUNT[_action_id] = len(HAND_SUBSETS[_decoded.index])
    elif _decoded.action_type == ActionType.USE_CONSUMABLE_HAND_SUBSET:
        _ACTION_ID_TO_COUNT[_action_id] = len(CONSUMABLE_HAND_SUBSETS[_decoded.detail])

_HAND_SUBSET_SLOT_PAD = np.full((len(HAND_SUBSETS), 5), -1, dtype=np.int64)
for _idx, _subset in enumerate(HAND_SUBSETS):
    _HAND_SUBSET_SLOT_PAD[_idx, : len(_subset)] = _subset

_CONSUMABLE_SUBSET_SLOT_PAD = np.full(
    (len(CONSUMABLE_HAND_SUBSETS), MAX_CONSUMABLE_HAND_TARGETS),
    -1,
    dtype=np.int64,
)
for _idx, _subset in enumerate(CONSUMABLE_HAND_SUBSETS):
    _CONSUMABLE_SUBSET_SLOT_PAD[_idx, : len(_subset)] = _subset

_BIT_TABLE_SIZE = 1 << MAX_HAND_SIZE
_BIT_TO_HAND_SUBSET_INDEX = np.full(_BIT_TABLE_SIZE, -1, dtype=np.int64)
for _idx, _bits in enumerate(HAND_SUBSET_BITS):
    _BIT_TO_HAND_SUBSET_INDEX[int(_bits)] = _idx

_BIT_TO_CONSUMABLE_SUBSET_INDEX = np.full(_BIT_TABLE_SIZE, -1, dtype=np.int64)
for _idx, _bits in enumerate(CONSUMABLE_HAND_SUBSET_BITS):
    _BIT_TO_CONSUMABLE_SUBSET_INDEX[int(_bits)] = _idx

_SLOT_BITS = np.asarray([1 << slot for slot in range(MAX_HAND_SIZE)], dtype=np.int64)
_HAND_SUBSET_BITS64 = HAND_SUBSET_BITS.astype(np.int64)
_HAND_SUBSET_SIZES64 = HAND_SUBSET_SIZES.astype(np.int64)
_CONSUMABLE_SUBSET_BITS64 = CONSUMABLE_HAND_SUBSET_BITS.astype(np.int64)
_CONSUMABLE_SUBSET_SIZES64 = CONSUMABLE_HAND_SUBSET_SIZES.astype(np.int64)
_TENSOR_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


@dataclass(slots=True)
class ActionGrammarOutput:
    macro_logits: torch.Tensor
    hand_count_logits: torch.Tensor  # (B, 2, 5), order: play, discard
    hand_card_logits: torch.Tensor  # (B, 2, MAX_HAND_SIZE)
    candidate_play_logits: torch.Tensor  # (B, MAX_PLAY_CANDIDATES)
    candidate_discard_logits: torch.Tensor  # (B, MAX_DISCARD_CANDIDATES)
    consumable_slot_logits: torch.Tensor  # (B, 3, MAX_CONSUMABLE_SLOTS), no/hand/joker
    consumable_count_logits: torch.Tensor  # (B, MAX_CONSUMABLE_SLOTS, 3)
    consumable_card_logits: torch.Tensor  # (B, MAX_CONSUMABLE_SLOTS, MAX_HAND_SIZE)
    consumable_joker_logits: torch.Tensor  # (B, MAX_CONSUMABLE_SLOTS, MAX_JOKER_SLOTS)
    shop_buy_logits: torch.Tensor  # (B, MAX_SHOP_ITEMS)
    shop_sell_joker_logits: torch.Tensor  # (B, MAX_JOKER_SLOTS)
    shop_sell_consumable_logits: torch.Tensor  # (B, MAX_CONSUMABLE_SLOTS)
    pack_claim_logits: torch.Tensor  # (B, MAX_PACK_CARDS)


class ActionGrammarHead(nn.Module):
    """Produces component logits for the structured action grammar."""

    def __init__(self, d_model: int, *, danger_shop_leave_logit_penalty: float = 0.0):
        super().__init__()
        state_dim = 128
        hidden = 128
        ctx = 64

        self.global_proj = nn.Sequential(
            nn.Linear(d_model, state_dim - 32),
            nn.GELU(),
        )
        self.scalar_proj = nn.Sequential(
            # Preserve the tokenizer-v5 policy head exactly. Tokenizer-v6 risk
            # enters through the META target token and is learned from there.
            nn.Linear(LEGACY_POLICY_SCALAR_DIM, 32),
            nn.GELU(),
        )
        # Give the action heads a short, explicit path from danger to behavior.
        # Start neutral; PPO can learn how danger, remaining hands, and
        # remaining discards should change action preferences.
        self.danger_policy_proj = nn.Linear(3, state_dim, bias=False)
        nn.init.zeros_(self.danger_policy_proj.weight)
        self.danger_shop_leave_logit_penalty = max(float(danger_shop_leave_logit_penalty), 0.0)
        self.macro_head = nn.Linear(state_dim, NUM_GRAMMAR_ACTIONS)

        self.hand_count_head = nn.Linear(state_dim, 2 * 5)
        self.hand_state_proj = nn.Linear(state_dim, hidden)
        self.hand_card_proj = nn.Linear(d_model, hidden)
        self.hand_card_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

        self.consumable_slot_head = nn.Linear(d_model, 3)
        self.consumable_count_head = nn.Sequential(
            nn.Linear(d_model + state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, MAX_CONSUMABLE_HAND_TARGETS),
        )
        self.consumable_slot_proj = nn.Linear(d_model, ctx)
        self.consumable_hand_proj = nn.Linear(d_model, ctx)
        self.consumable_joker_proj = nn.Linear(d_model, ctx)

        self.shop_buy_head = nn.Linear(d_model, 1)
        self.shop_global_head = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, MAX_JOKER_SLOTS + MAX_CONSUMABLE_SLOTS),
        )
        self.pack_claim_head = nn.Linear(d_model, 1)
        self.candidate_play_head = nn.Linear(d_model, 1)
        self.candidate_discard_head = nn.Linear(d_model, 1)

    def forward(
        self,
        backbone_out: torch.Tensor,
        attention_mask: torch.Tensor,
        tokens: torch.Tensor,
        token_types: torch.Tensor,
        scalars: torch.Tensor,
    ) -> ActionGrammarOutput:
        batch = backbone_out.shape[0]
        full_mask = attention_mask.unsqueeze(-1).to(backbone_out.dtype)
        global_pool = (backbone_out * full_mask).sum(1) / full_mask.sum(1).clamp(min=1)
        state = torch.cat(
            [
                self.global_proj(global_pool),
                self.scalar_proj(scalars[:, :LEGACY_POLICY_SCALAR_DIM]),
            ],
            dim=-1,
        )
        danger = scalars[:, 12].clamp(0.0, 1.0) if scalars.shape[1] >= 13 else scalars.new_zeros(batch)
        hands_fraction = (scalars[:, 4] / 4.0).clamp(0.0, 1.0)
        discards_fraction = (scalars[:, 5] / 4.0).clamp(0.0, 1.0)
        danger_features = torch.stack(
            [danger, danger * hands_fraction, danger * discards_fraction],
            dim=-1,
        )
        state = state + self.danger_policy_proj(danger_features)

        hand_ctx, hand_present = self._gather_hand_slots(
            backbone_out,
            attention_mask,
            tokens,
            token_types,
        )
        hand_hidden = self.hand_card_proj(hand_ctx) + self.hand_state_proj(state).unsqueeze(1)
        hand_card_logits = self.hand_card_head(hand_hidden).permute(0, 2, 1)
        hand_card_logits = hand_card_logits.masked_fill(hand_present.unsqueeze(1) <= 0, -1e8)

        slot_tokens = backbone_out[:, CONSUMABLE_START : CONSUMABLE_START + MAX_CONSUMABLE_SLOTS]
        joker_tokens = backbone_out[:, JOKER_START : JOKER_START + MAX_JOKER_SLOTS]
        slot_state = state.unsqueeze(1).expand(-1, MAX_CONSUMABLE_SLOTS, -1)
        consumable_count_logits = self.consumable_count_head(torch.cat([slot_tokens, slot_state], dim=-1))

        slot_ctx = self.consumable_slot_proj(slot_tokens)
        hand_target_ctx = self.consumable_hand_proj(hand_ctx) * hand_present.unsqueeze(-1)
        joker_ctx = self.consumable_joker_proj(joker_tokens)
        consumable_card_logits = torch.einsum("bsc,bhc->bsh", slot_ctx, hand_target_ctx)
        consumable_card_logits = consumable_card_logits.masked_fill(hand_present.unsqueeze(1) <= 0, -1e8)
        consumable_joker_logits = torch.einsum("bsc,bjc->bsj", slot_ctx, joker_ctx)

        shop_tokens = backbone_out[:, SHOP_START : SHOP_START + MAX_SHOP_ITEMS]
        shop_global = self.shop_global_head(state)
        pack_tokens = backbone_out[:, SHOP_START : SHOP_START + MAX_PACK_CARDS]

        candidate_tokens = backbone_out[:, HAND_CANDIDATE_START : HAND_CANDIDATE_START + HAND_CANDIDATE_MAX]
        candidate_mask = attention_mask[:, HAND_CANDIDATE_START : HAND_CANDIDATE_START + HAND_CANDIDATE_MAX]
        candidate_kinds = tokens[:, HAND_CANDIDATE_START : HAND_CANDIDATE_START + HAND_CANDIDATE_MAX, 0]
        play_cand_mask = candidate_mask.bool() & candidate_kinds.eq(1)
        discard_cand_mask = candidate_mask.bool() & candidate_kinds.eq(2)
        candidate_scores = self.candidate_play_head(candidate_tokens).squeeze(-1)
        candidate_play_logits = candidate_scores.masked_fill(~play_cand_mask, -1e8)
        candidate_disc_logits = self.candidate_discard_head(candidate_tokens).squeeze(-1)
        candidate_discard_logits = candidate_disc_logits.masked_fill(~discard_cand_mask, -1e8)

        macro_logits = self.macro_head(state)
        if self.danger_shop_leave_logit_penalty > 0.0:
            # This is deliberately a soft prior, not a mask. If leaving is the
            # only legal action it remains certain; otherwise a highly unsafe
            # shop favors continued search without forcing a particular buy.
            in_shop = scalars[:, 7].round().eq(_SHOP_PHASE_ID).to(macro_logits.dtype)
            urgency = ((danger - _DANGER_THRESHOLD) / (1.0 - _DANGER_THRESHOLD)).clamp(0.0, 1.0)
            prior = self.danger_shop_leave_logit_penalty * urgency * in_shop
            macro_logits = macro_logits.clone()
            macro_logits[:, _SHOP_LEAVE] -= prior
            macro_logits[:, _SHOP_REROLL] += 0.5 * prior

        return ActionGrammarOutput(
            macro_logits=macro_logits,
            hand_count_logits=self.hand_count_head(state).view(batch, 2, 5),
            hand_card_logits=hand_card_logits,
            candidate_play_logits=candidate_play_logits,
            candidate_discard_logits=candidate_discard_logits,
            consumable_slot_logits=self.consumable_slot_head(slot_tokens).permute(0, 2, 1),
            consumable_count_logits=consumable_count_logits,
            consumable_card_logits=consumable_card_logits,
            consumable_joker_logits=consumable_joker_logits,
            shop_buy_logits=self.shop_buy_head(shop_tokens).squeeze(-1),
            shop_sell_joker_logits=shop_global[:, :MAX_JOKER_SLOTS],
            shop_sell_consumable_logits=shop_global[
                :,
                MAX_JOKER_SLOTS : MAX_JOKER_SLOTS + MAX_CONSUMABLE_SLOTS,
            ],
            pack_claim_logits=self.pack_claim_head(pack_tokens).squeeze(-1),
        )

    @staticmethod
    def _gather_hand_slots(
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
        hand_ctx = torch.zeros(batch, MAX_HAND_SIZE, d_model, device=deck_out.device, dtype=deck_out.dtype)
        hand_present = torch.zeros(batch, MAX_HAND_SIZE, device=deck_out.device, dtype=deck_out.dtype)

        ctx_index = hand_slots.unsqueeze(-1).expand(-1, -1, d_model)
        hand_ctx.scatter_add_(1, ctx_index, deck_out * hand_card_mask.unsqueeze(-1))
        hand_present.scatter_add_(1, hand_slots, hand_card_mask.to(deck_out.dtype))
        hand_present.clamp_(0.0, 1.0)
        return hand_ctx, hand_present


class ActionGrammarDistribution:
    """Distribution over flat action IDs via structured grammar components."""

    def __init__(
        self,
        output: ActionGrammarOutput,
        action_mask: torch.Tensor,
        temperature: float | torch.Tensor = 1.0,
        tokens: torch.Tensor | None = None,
        hand_ar_mixture_eps: float = 0.0,
    ) -> None:
        self.output = output
        self.action_mask = action_mask > 0
        self.device = action_mask.device
        self.batch_size = action_mask.shape[0]
        if isinstance(temperature, torch.Tensor):
            temperature = temperature.to(device=self.device, dtype=torch.float32)
            if temperature.ndim == 0:
                temperature = temperature.expand(self.batch_size)
            if temperature.shape != (self.batch_size,):
                raise ValueError(
                    "tensor temperature must be scalar or have one value per batch row "
                    f"(expected {(self.batch_size,)}, got {tuple(temperature.shape)})"
                )
            self.temperature: float | torch.Tensor = temperature.clamp_min(1e-6)
        else:
            self.temperature = max(float(temperature), 1e-6)
        self.tokens = tokens
        # Mixture weight on the autoregressive hand/discard head:
        #   p(a) = (1 - eps) * p_cand(a) + eps * p_ar(a)
        # p_cand has support only over candidate slots; p_ar covers every legal
        # subset under the mask. eps > 0 gives the policy full support over hand
        # plays (the -1e8 floor for non-candidate actions becomes a finite
        # log(eps) + ar_logp), so the policy can finally express plays the
        # candidate generator never proposed. eps = 0 selects the candidate-only
        # ablation.
        self.hand_ar_mixture_eps = min(max(float(hand_ar_mixture_eps), 0.0), 1.0)
        self.macro_mask = _macro_valid_mask(self.action_mask)
        has_any = self.macro_mask.any(dim=-1, keepdim=True)
        fallback = torch.zeros_like(self.macro_mask)
        fallback[:, 0] = True
        self.macro_mask = torch.where(has_any, self.macro_mask, fallback)

    @property
    def action_type_probs(self) -> torch.Tensor:
        return _masked_softmax(self._t(self.output.macro_logits), self.macro_mask)

    def sample(self) -> torch.Tensor:
        macro = _sample_masked(self._t(self.output.macro_logits), self.macro_mask)
        actions = self._first_valid_actions()

        actions = torch.where(
            macro == ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.BLIND_PLAY],
            torch.full_like(actions, int(ActionRange.BLIND_PLAY)),
            actions,
        )
        actions = torch.where(
            macro == ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.BLIND_SKIP],
            torch.full_like(actions, int(ActionRange.BLIND_SKIP)),
            actions,
        )
        actions = torch.where(
            macro == ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.BLIND_REROLL],
            torch.full_like(actions, int(ActionRange.BLIND_REROLL)),
            actions,
        )
        actions = torch.where(
            macro == ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.SHOP_REROLL],
            torch.full_like(actions, int(ActionRange.SHOP_REROLL)),
            actions,
        )
        actions = torch.where(
            macro == ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.SHOP_LEAVE],
            torch.full_like(actions, int(ActionRange.SHOP_LEAVE)),
            actions,
        )
        actions = torch.where(
            macro == ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.PACK_SKIP],
            torch.full_like(actions, int(ActionRange.PACK_SKIP)),
            actions,
        )

        play_actions = self._sample_hand_actions(is_play=True)
        discard_actions = self._sample_hand_actions(is_play=False)
        actions = torch.where(macro == _PLAY, play_actions, actions)
        actions = torch.where(macro == _DISCARD, discard_actions, actions)

        no_target_actions = self._sample_consumable_no_target_actions()
        hand_target_actions = self._sample_consumable_hand_actions()
        joker_target_actions = self._sample_consumable_joker_actions()
        actions = torch.where(macro == _CONSUMABLE_NO_TARGET, no_target_actions, actions)
        actions = torch.where(macro == _CONSUMABLE_HAND, hand_target_actions, actions)
        actions = torch.where(macro == _CONSUMABLE_JOKER, joker_target_actions, actions)

        shop_buy_actions = self._sample_indexed_actions(
            self._t(self.output.shop_buy_logits),
            _range_mask(self.action_mask, ActionRange.SHOP_BUY_START, ActionRange.SHOP_BUY_END),
            int(ActionRange.SHOP_BUY_START),
        )
        shop_sell_joker_actions = self._sample_indexed_actions(
            self._t(self.output.shop_sell_joker_logits),
            _range_mask(self.action_mask, ActionRange.SHOP_SELL_JOKER_START, ActionRange.SHOP_SELL_JOKER_END),
            int(ActionRange.SHOP_SELL_JOKER_START),
        )
        shop_sell_consumable_actions = self._sample_indexed_actions(
            self._t(self.output.shop_sell_consumable_logits),
            _range_mask(
                self.action_mask,
                ActionRange.SHOP_SELL_CONSUMABLE_START,
                ActionRange.SHOP_SELL_CONSUMABLE_END,
            ),
            int(ActionRange.SHOP_SELL_CONSUMABLE_START),
        )
        pack_claim_actions = self._sample_indexed_actions(
            self._t(self.output.pack_claim_logits),
            _range_mask(self.action_mask, ActionRange.PACK_CLAIM_START, ActionRange.PACK_CLAIM_END),
            int(ActionRange.PACK_CLAIM_START),
        )
        actions = torch.where(macro == _SHOP_BUY, shop_buy_actions, actions)
        actions = torch.where(macro == _SHOP_SELL_JOKER, shop_sell_joker_actions, actions)
        actions = torch.where(macro == _SHOP_SELL_CONSUMABLE, shop_sell_consumable_actions, actions)
        actions = torch.where(macro == _PACK_CLAIM, pack_claim_actions, actions)

        valid = self.action_mask.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        return torch.where(valid, actions, self._first_valid_actions())

    def mode(self) -> torch.Tensor:
        actions, _log_probs = self._flat_mode()
        return actions

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.long()
        action_macro = _action_id_to_grammar(self.device)[actions]
        action_index = _action_id_to_index(self.device)[actions]
        action_detail = _action_id_to_detail(self.device)[actions]
        action_count = _action_id_to_count(self.device)[actions]

        macro_log_probs = _masked_log_softmax(self._t(self.output.macro_logits), self.macro_mask)
        log_prob = macro_log_probs.gather(1, action_macro.unsqueeze(-1)).squeeze(-1)

        play_logp = self._hand_log_prob(actions, action_index, action_count, is_play=True)
        discard_logp = self._hand_log_prob(actions, action_index, action_count, is_play=False)
        log_prob = log_prob + torch.where(action_macro == _PLAY, play_logp, torch.zeros_like(log_prob))
        log_prob = log_prob + torch.where(action_macro == _DISCARD, discard_logp, torch.zeros_like(log_prob))

        no_target_logp = self._consumable_no_target_log_prob(action_index)
        hand_logp = self._consumable_hand_log_prob(action_index, action_detail, action_count)
        joker_logp = self._consumable_joker_log_prob(action_index, action_detail)
        log_prob = log_prob + torch.where(
            action_macro == _CONSUMABLE_NO_TARGET,
            no_target_logp,
            torch.zeros_like(log_prob),
        )
        log_prob = log_prob + torch.where(action_macro == _CONSUMABLE_HAND, hand_logp, torch.zeros_like(log_prob))
        log_prob = log_prob + torch.where(action_macro == _CONSUMABLE_JOKER, joker_logp, torch.zeros_like(log_prob))

        log_prob = log_prob + torch.where(
            action_macro == _SHOP_BUY,
            self._indexed_log_prob(
                self._t(self.output.shop_buy_logits),
                _range_mask(self.action_mask, ActionRange.SHOP_BUY_START, ActionRange.SHOP_BUY_END),
                action_index.clamp(0, MAX_SHOP_ITEMS - 1),
            ),
            torch.zeros_like(log_prob),
        )
        log_prob = log_prob + torch.where(
            action_macro == _SHOP_SELL_JOKER,
            self._indexed_log_prob(
                self._t(self.output.shop_sell_joker_logits),
                _range_mask(self.action_mask, ActionRange.SHOP_SELL_JOKER_START, ActionRange.SHOP_SELL_JOKER_END),
                action_index.clamp(0, MAX_JOKER_SLOTS - 1),
            ),
            torch.zeros_like(log_prob),
        )
        log_prob = log_prob + torch.where(
            action_macro == _SHOP_SELL_CONSUMABLE,
            self._indexed_log_prob(
                self._t(self.output.shop_sell_consumable_logits),
                _range_mask(
                    self.action_mask,
                    ActionRange.SHOP_SELL_CONSUMABLE_START,
                    ActionRange.SHOP_SELL_CONSUMABLE_END,
                ),
                action_index.clamp(0, MAX_CONSUMABLE_SLOTS - 1),
            ),
            torch.zeros_like(log_prob),
        )
        log_prob = log_prob + torch.where(
            action_macro == _PACK_CLAIM,
            self._indexed_log_prob(
                self._t(self.output.pack_claim_logits),
                _range_mask(self.action_mask, ActionRange.PACK_CLAIM_START, ActionRange.PACK_CLAIM_END),
                action_index.clamp(0, MAX_PACK_CARDS - 1),
            ),
            torch.zeros_like(log_prob),
        )
        return log_prob

    def entropy(self) -> torch.Tensor:
        macro_logits = self._t(self.output.macro_logits)
        macro_probs = _masked_softmax(macro_logits, self.macro_mask)
        entropy = _masked_entropy(macro_logits, self.macro_mask)

        play_entropy = self._hand_entropy(is_play=True)
        discard_entropy = self._hand_entropy(is_play=False)
        entropy = entropy + macro_probs[:, _PLAY] * play_entropy
        entropy = entropy + macro_probs[:, _DISCARD] * discard_entropy

        entropy = entropy + macro_probs[:, _CONSUMABLE_NO_TARGET] * _masked_entropy(
            self._t(self.output.consumable_slot_logits[:, 0]),
            self._consumable_no_target_slot_mask(),
        )
        entropy = entropy + macro_probs[:, _CONSUMABLE_HAND] * self._consumable_hand_entropy()
        entropy = entropy + macro_probs[:, _CONSUMABLE_JOKER] * self._consumable_joker_entropy()

        entropy = entropy + macro_probs[:, _SHOP_BUY] * _masked_entropy(
            self._t(self.output.shop_buy_logits),
            _range_mask(self.action_mask, ActionRange.SHOP_BUY_START, ActionRange.SHOP_BUY_END),
        )
        entropy = entropy + macro_probs[:, _SHOP_SELL_JOKER] * _masked_entropy(
            self._t(self.output.shop_sell_joker_logits),
            _range_mask(self.action_mask, ActionRange.SHOP_SELL_JOKER_START, ActionRange.SHOP_SELL_JOKER_END),
        )
        entropy = entropy + macro_probs[:, _SHOP_SELL_CONSUMABLE] * _masked_entropy(
            self._t(self.output.shop_sell_consumable_logits),
            _range_mask(self.action_mask, ActionRange.SHOP_SELL_CONSUMABLE_START, ActionRange.SHOP_SELL_CONSUMABLE_END),
        )
        entropy = entropy + macro_probs[:, _PACK_CLAIM] * _masked_entropy(
            self._t(self.output.pack_claim_logits),
            _range_mask(self.action_mask, ActionRange.PACK_CLAIM_START, ActionRange.PACK_CLAIM_END),
        )
        return entropy

    def normalized_action_type_entropy(self) -> torch.Tensor:
        entropy = _masked_entropy(self._t(self.output.macro_logits), self.macro_mask)
        valid_counts = self.macro_mask.sum(dim=-1).to(entropy.dtype)
        max_entropy = torch.log(valid_counts.clamp_min(2.0))
        normalized = torch.where(valid_counts > 1, entropy / max_entropy, torch.zeros_like(entropy))
        return normalized.mean()

    def max_prob(self) -> torch.Tensor:
        """Exact probability of :meth:`mode` under the flat action policy."""
        _actions, log_probs = self._flat_mode()
        return log_probs.exp()

    def _flat_mode(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the exact flat-action argmax and its joint log-probability.

        A macro-first greedy choice is not the mode of a hierarchical policy:
        a slightly more likely but diffuse macro can have less mass on every
        individual action than a deterministic macro.  Compare each macro's
        best conditional leaf after accounting for the candidate/AR hand
        mixture, then maximize the joint probability.
        """
        cached = getattr(self, "_flat_mode_cache", None)
        if cached is not None:
            return cached

        actions = self._first_valid_actions().unsqueeze(1).expand(-1, NUM_GRAMMAR_ACTIONS).clone()
        conditional_logp = torch.full(
            (self.batch_size, NUM_GRAMMAR_ACTIONS),
            -1e8,
            dtype=self.output.macro_logits.dtype,
            device=self.device,
        )

        for action_type in (
            ActionType.BLIND_PLAY,
            ActionType.BLIND_SKIP,
            ActionType.BLIND_REROLL,
            ActionType.SHOP_REROLL,
            ActionType.SHOP_LEAVE,
            ActionType.PACK_SKIP,
        ):
            macro_idx = ACTION_TYPE_TO_GRAMMAR_INDEX[action_type]
            actions[:, macro_idx] = decode_action_id_for_singleton(action_type)
            conditional_logp[:, macro_idx] = 0.0

        for macro_idx, is_play in ((_PLAY, True), (_DISCARD, False)):
            hand_action, hand_logp = self._best_hand_actions(is_play=is_play)
            actions[:, macro_idx] = hand_action
            conditional_logp[:, macro_idx] = hand_logp

        no_target_action, no_target_logp = self._best_indexed_actions(
            self._t(self.output.consumable_slot_logits[:, 0]),
            self._consumable_no_target_slot_mask(),
            int(ActionRange.CONSUMABLE_FLAT_START),
            stride=CONSUMABLE_ACTIONS_PER_SLOT,
            offset=CONSUMABLE_NO_TARGET_OFFSET,
        )
        actions[:, _CONSUMABLE_NO_TARGET] = no_target_action
        conditional_logp[:, _CONSUMABLE_NO_TARGET] = no_target_logp

        hand_action, hand_logp = self._best_consumable_hand_actions()
        actions[:, _CONSUMABLE_HAND] = hand_action
        conditional_logp[:, _CONSUMABLE_HAND] = hand_logp
        joker_action, joker_logp = self._best_consumable_joker_actions()
        actions[:, _CONSUMABLE_JOKER] = joker_action
        conditional_logp[:, _CONSUMABLE_JOKER] = joker_logp

        indexed_families = (
            (_SHOP_BUY, self.output.shop_buy_logits, ActionRange.SHOP_BUY_START, ActionRange.SHOP_BUY_END),
            (
                _SHOP_SELL_JOKER,
                self.output.shop_sell_joker_logits,
                ActionRange.SHOP_SELL_JOKER_START,
                ActionRange.SHOP_SELL_JOKER_END,
            ),
            (
                _SHOP_SELL_CONSUMABLE,
                self.output.shop_sell_consumable_logits,
                ActionRange.SHOP_SELL_CONSUMABLE_START,
                ActionRange.SHOP_SELL_CONSUMABLE_END,
            ),
            (_PACK_CLAIM, self.output.pack_claim_logits, ActionRange.PACK_CLAIM_START, ActionRange.PACK_CLAIM_END),
        )
        for macro_idx, logits, start, end in indexed_families:
            action, logp = self._best_indexed_actions(
                self._t(logits),
                _range_mask(self.action_mask, start, end),
                int(start),
            )
            actions[:, macro_idx] = action
            conditional_logp[:, macro_idx] = logp

        macro_logp = _masked_log_softmax(self._t(self.output.macro_logits), self.macro_mask)
        joint_logp = (macro_logp + conditional_logp).masked_fill(~self.macro_mask, -1e8)
        best_macro = joint_logp.argmax(dim=-1)
        best_actions = actions.gather(1, best_macro.unsqueeze(-1)).squeeze(-1)
        best_logp = joint_logp.gather(1, best_macro.unsqueeze(-1)).squeeze(-1)
        valid = self.action_mask.gather(1, best_actions.unsqueeze(-1)).squeeze(-1)
        best_actions = torch.where(valid, best_actions, self._first_valid_actions())
        self._flat_mode_cache = (best_actions, best_logp)
        return self._flat_mode_cache

    def _best_indexed_actions(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        base: int,
        *,
        stride: int = 1,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_probs = _masked_log_softmax(logits, mask)
        best_logp, index = log_probs.max(dim=-1)
        return base + index * stride + offset, best_logp

    def _best_hand_actions(self, *, is_play: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact conditional mode of the candidate/AR hand mixture."""
        valid_subsets = self._hand_valid_subset_mask(is_play)
        family = 0 if is_play else 1
        base = int(ActionRange.PLAY_SUBSET_START if is_play else ActionRange.DISCARD_SUBSET_START)
        cand_logits, cand_valid, cand_actions = self._candidate_distribution(is_play=is_play)
        cand_log_probs = _masked_log_softmax(cand_logits, cand_valid)
        has_candidates = cand_valid.any(dim=-1)
        count_masks = _subset_count_mask(valid_subsets, _hand_subset_sizes(self.device), max_count=5)
        count_log_probs = _masked_log_softmax(self._t(self.output.hand_count_logits[:, family]), count_masks)
        card_logits = self._t(self.output.hand_card_logits[:, family])

        best_actions = self._first_valid_actions()
        best_logps = torch.full_like(best_actions, -1e8, dtype=card_logits.dtype)
        subset_sizes = _hand_subset_sizes(self.device)
        subset_slots = _hand_subset_slots(self.device)
        eps = self.hand_ar_mixture_eps

        for row in range(self.batch_size):
            legal = valid_subsets[row].nonzero(as_tuple=False).squeeze(-1)
            if legal.numel() == 0:
                continue
            row_best_logp = card_logits.new_tensor(-1e8)
            row_best_subset = legal.new_tensor(0)
            for chunk in legal.split(256):
                counts = subset_sizes[chunk]
                chunk_size = int(chunk.numel())
                ar_logp = count_log_probs[row, counts - 1] + _ordered_card_log_prob(
                    card_logits[row : row + 1].expand(chunk_size, -1),
                    valid_subsets[row : row + 1].expand(chunk_size, -1),
                    subset_slots[chunk],
                    counts,
                    max_count=5,
                    subset_bits=_hand_subset_bits(self.device),
                    subset_sizes=subset_sizes,
                    slot_bits=_slot_bits(self.device),
                )
                if has_candidates[row] and eps < 1.0:
                    flat_actions = base + chunk
                    matches = cand_actions[row].unsqueeze(0).eq(flat_actions.unsqueeze(1)) & cand_valid[row].unsqueeze(
                        0
                    )
                    has_match = matches.any(dim=-1)
                    slots = matches.long().argmax(dim=-1)
                    candidate_logp = cand_log_probs[row, slots]
                    candidate_logp = torch.where(
                        has_match,
                        candidate_logp,
                        torch.full_like(candidate_logp, -1e8),
                    )
                    if eps <= 0.0:
                        mixed_logp = candidate_logp
                    else:
                        mixed_logp = torch.logsumexp(
                            torch.stack(
                                [math.log(1.0 - eps) + candidate_logp, math.log(eps) + ar_logp],
                                dim=-1,
                            ),
                            dim=-1,
                        )
                else:
                    mixed_logp = ar_logp
                chunk_logp, chunk_slot = mixed_logp.max(dim=0)
                if chunk_logp > row_best_logp:
                    row_best_logp = chunk_logp
                    row_best_subset = chunk[chunk_slot]
            best_actions[row] = base + row_best_subset
            best_logps[row] = row_best_logp
        return best_actions, best_logps

    def _best_consumable_hand_actions(self) -> tuple[torch.Tensor, torch.Tensor]:
        blocks = self._consumable_blocks()
        hand_start = CONSUMABLE_HAND_SUBSET_OFFSET
        valid = blocks[:, :, hand_start : hand_start + NUM_CONSUMABLE_HAND_SUBSETS]
        slot_mask = valid.any(dim=-1)
        slot_logp = _masked_log_softmax(self._t(self.output.consumable_slot_logits[:, 1]), slot_mask)
        subset_sizes = _consumable_subset_sizes(self.device)
        subset_slots = _consumable_subset_slots(self.device)
        best_actions = self._first_valid_actions()
        best_logps = torch.full_like(best_actions, -1e8, dtype=slot_logp.dtype)

        for row in range(self.batch_size):
            for slot in slot_mask[row].nonzero(as_tuple=False).squeeze(-1):
                slot_index = int(slot.item())
                slot_valid = valid[row, slot_index]
                legal = slot_valid.nonzero(as_tuple=False).squeeze(-1)
                if legal.numel() == 0:
                    continue
                count_mask = _subset_count_mask(slot_valid.unsqueeze(0), subset_sizes, MAX_CONSUMABLE_HAND_TARGETS)
                count_logp = _masked_log_softmax(
                    self._t(self.output.consumable_count_logits[row : row + 1, slot_index]),
                    count_mask,
                )[0]
                for chunk in legal.split(256):
                    counts = subset_sizes[chunk]
                    chunk_size = int(chunk.numel())
                    leaf_logp = count_logp[counts - 1] + _ordered_card_log_prob(
                        self._t(self.output.consumable_card_logits[row : row + 1, slot_index]).expand(chunk_size, -1),
                        slot_valid.unsqueeze(0).expand(chunk_size, -1),
                        subset_slots[chunk],
                        counts,
                        max_count=MAX_CONSUMABLE_HAND_TARGETS,
                        subset_bits=_consumable_subset_bits(self.device),
                        subset_sizes=subset_sizes,
                        slot_bits=_slot_bits(self.device),
                    )
                    leaf_logp = leaf_logp + slot_logp[row, slot_index]
                    chunk_logp, chunk_index = leaf_logp.max(dim=0)
                    if chunk_logp > best_logps[row]:
                        detail = chunk[chunk_index]
                        best_logps[row] = chunk_logp
                        best_actions[row] = (
                            int(ActionRange.CONSUMABLE_FLAT_START)
                            + slot_index * CONSUMABLE_ACTIONS_PER_SLOT
                            + CONSUMABLE_HAND_SUBSET_OFFSET
                            + detail
                        )
        return best_actions, best_logps

    def _best_consumable_joker_actions(self) -> tuple[torch.Tensor, torch.Tensor]:
        blocks = self._consumable_blocks()
        start = CONSUMABLE_JOKER_OFFSET
        valid = blocks[:, :, start : start + MAX_JOKER_SLOTS]
        slot_mask = valid.any(dim=-1)
        slot_logp = _masked_log_softmax(self._t(self.output.consumable_slot_logits[:, 2]), slot_mask)
        joker_logp = torch.log_softmax(
            self._t(self.output.consumable_joker_logits).masked_fill(~valid, -1e8),
            dim=-1,
        )
        best_joker_logp, joker = joker_logp.max(dim=-1)
        joint = (slot_logp + best_joker_logp).masked_fill(~slot_mask, -1e8)
        best_logp, slot = joint.max(dim=-1)
        rows = torch.arange(self.batch_size, device=self.device)
        joker = joker[rows, slot]
        action = (
            int(ActionRange.CONSUMABLE_FLAT_START)
            + slot * CONSUMABLE_ACTIONS_PER_SLOT
            + CONSUMABLE_JOKER_OFFSET
            + joker
        )
        return action, best_logp

    def selected_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.log_prob(actions).exp()

    def _t(self, logits: torch.Tensor) -> torch.Tensor:
        if isinstance(self.temperature, torch.Tensor):
            shape = (self.batch_size, *([1] * (logits.ndim - 1)))
            return logits / self.temperature.to(dtype=logits.dtype).view(shape)
        return logits / self.temperature

    def _first_valid_actions(self) -> torch.Tensor:
        return self.action_mask.to(torch.float32).argmax(dim=-1).long()

    def _hand_valid_subset_mask(self, is_play: bool) -> torch.Tensor:
        start = ActionRange.PLAY_SUBSET_START if is_play else ActionRange.DISCARD_SUBSET_START
        end = ActionRange.PLAY_SUBSET_END if is_play else ActionRange.DISCARD_SUBSET_END
        return _range_mask(self.action_mask, start, end)

    def _hand_log_prob(
        self,
        actions: torch.Tensor,
        action_index: torch.Tensor,
        action_count: torch.Tensor,
        *,
        is_play: bool,
    ) -> torch.Tensor:
        cand_logits, cand_valid, cand_actions = self._candidate_distribution(is_play=is_play)
        has_candidates = cand_valid.any(dim=-1)

        matches = (cand_actions == actions.unsqueeze(-1)) & cand_valid
        has_match = matches.any(dim=-1)
        matched_slot = matches.long().argmax(dim=-1)
        cand_log_probs = _masked_log_softmax(cand_logits, cand_valid)
        cand_logp = cand_log_probs.gather(1, matched_slot.unsqueeze(-1)).squeeze(-1)
        # If candidates exist but this action isn't reachable through them, the
        # sampler can't produce it, so log_prob is -inf (use a finite floor so
        # PPO ratios don't NaN; the policy gradient will still push away from
        # this slot).
        cand_logp = torch.where(has_match, cand_logp, torch.full_like(cand_logp, -1e8))

        ar_logp = self._hand_log_prob_autoregressive(action_index, action_count, is_play=is_play)

        eps = self.hand_ar_mixture_eps
        if eps <= 0.0:
            # Backward-compat: candidate distribution when candidates exist, AR
            # otherwise. Bit-for-bit identical to the pre-mixture behavior.
            return torch.where(has_candidates, cand_logp, ar_logp)
        if eps >= 1.0:
            # Pure AR: the candidate component has zero weight, and
            # log(1 - eps) is undefined.
            return ar_logp

        # Mixture: p(a) = (1-eps)*p_cand(a) + eps*p_ar(a).
        # For candidate actions cand_logp is the softmax log-prob; for
        # non-candidate actions cand_logp is the -1e8 floor, so the AR term
        # dominates and log_prob is finite (log(eps) + ar_logp), full support
        # over every legal hand play. Rows with no candidates collapse to AR.
        log_one_minus_eps = math.log(1.0 - eps)
        log_eps = math.log(eps)
        mix_logp = torch.logsumexp(
            torch.stack([log_one_minus_eps + cand_logp, log_eps + ar_logp], dim=-1),
            dim=-1,
        )
        return torch.where(has_candidates, mix_logp, ar_logp)

    def _hand_log_prob_autoregressive(
        self,
        action_index: torch.Tensor,
        action_count: torch.Tensor,
        *,
        is_play: bool,
    ) -> torch.Tensor:
        family = 0 if is_play else 1
        valid_subsets = self._hand_valid_subset_mask(is_play)
        count_mask = _subset_count_mask(valid_subsets, _hand_subset_sizes(self.device), max_count=5)
        count_logp = self._indexed_log_prob(
            self._t(self.output.hand_count_logits[:, family]),
            count_mask,
            (action_count - 1).clamp(0, 4),
        )

        slots = _hand_subset_slots(self.device)[action_index.clamp(0, len(HAND_SUBSETS) - 1)]
        card_logp = _ordered_card_log_prob(
            self._t(self.output.hand_card_logits[:, family]),
            valid_subsets,
            slots,
            action_count.clamp(1, 5),
            max_count=5,
            subset_bits=_hand_subset_bits(self.device),
            subset_sizes=_hand_subset_sizes(self.device),
            slot_bits=_slot_bits(self.device),
        )
        return count_logp + card_logp

    def _hand_entropy(self, *, is_play: bool) -> torch.Tensor:
        cand_logits, cand_valid, _ = self._candidate_distribution(is_play=is_play)
        has_candidates = cand_valid.any(dim=-1)
        cand_entropy = _masked_entropy(cand_logits, cand_valid)
        ar_entropy = self._hand_entropy_autoregressive(is_play=is_play)
        eps = self.hand_ar_mixture_eps
        if eps <= 0.0:
            return torch.where(has_candidates, cand_entropy, ar_entropy)
        # Exact mixture entropy is expensive; use the standard lower bound
        # H >= (1-eps)*H_cand + eps*H_ar. Entropy here only feeds a small bonus
        # coefficient; the bound's bias is acceptable and monotone in eps.
        mix_entropy = (1.0 - eps) * cand_entropy + eps * ar_entropy
        return torch.where(has_candidates, mix_entropy, ar_entropy)

    def _hand_entropy_autoregressive(self, *, is_play: bool) -> torch.Tensor:
        family = 0 if is_play else 1
        valid_subsets = self._hand_valid_subset_mask(is_play)
        count_mask = _subset_count_mask(valid_subsets, _hand_subset_sizes(self.device), max_count=5)
        count_logits = self._t(self.output.hand_count_logits[:, family])
        count_probs = _masked_softmax(count_logits, count_mask)
        count_entropy = _masked_entropy(count_logits, count_mask)
        card_logits = self._t(self.output.hand_card_logits[:, family])
        card_mask = _subset_card_mask(valid_subsets, _hand_subset_bits(self.device), _slot_bits(self.device))
        count_values = torch.arange(1, 6, dtype=count_probs.dtype, device=self.device)
        expected_count = (count_probs * count_values.unsqueeze(0)).sum(dim=-1)
        return count_entropy + expected_count * _masked_entropy(card_logits, card_mask)

    def _sample_hand_actions(self, *, is_play: bool) -> torch.Tensor:
        eps = self.hand_ar_mixture_eps
        cand_actions = self._sample_candidate_actions(is_play=is_play, greedy=False)
        has_candidates = cand_actions >= 0
        if eps <= 0.0:
            # Backward-compat: candidate when available, AR otherwise.
            fallback = self._sample_hand_actions_autoregressive(is_play=is_play, greedy=False)
            return torch.where(has_candidates, cand_actions, fallback)
        # Mixture sampling: draw Bernoulli(eps) per row to route between the
        # candidate sampler and the AR sampler. Rows without candidates always
        # use AR (the candidate component is empty for them).
        use_ar = (torch.rand(self.batch_size, device=self.device) < eps) | (~has_candidates)
        ar_actions = self._sample_hand_actions_autoregressive(is_play=is_play, greedy=False)
        return torch.where(use_ar, ar_actions, cand_actions.clamp_min(0))

    def _greedy_hand_actions(self, *, is_play: bool) -> torch.Tensor:
        cand_actions = self._sample_candidate_actions(is_play=is_play, greedy=True)
        has_candidates = cand_actions >= 0
        fallback = self._sample_hand_actions_autoregressive(is_play=is_play, greedy=True)
        return torch.where(has_candidates, cand_actions, fallback)

    def _candidate_distribution(self, *, is_play: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Candidate-head distribution as (logits, valid_mask, mapped_actions).

        `valid_mask` is True for slots that are kind-matched, attention-active,
        and map to an action that is legal in the current action_mask. Sampler
        and log_prob both consume this so they agree on the support.
        """
        cand_logits = self._t(self.output.candidate_play_logits if is_play else self.output.candidate_discard_logits)
        active = cand_logits > -1e7

        mapped_actions = self._candidate_slot_to_action(is_play=is_play)
        mapped_clamped = mapped_actions.clamp(0, NUM_ACTIONS - 1)
        in_mask = self.action_mask.gather(1, mapped_clamped).bool() & (mapped_actions >= 0)
        valid = active & in_mask
        return cand_logits, valid, mapped_actions

    def _candidate_slot_to_action(self, *, is_play: bool) -> torch.Tensor:
        """Map every candidate slot to its flat action ID (or -1 if unreachable).

        Returns: (B, HAND_CANDIDATE_MAX) long.
        """
        if self.tokens is None:
            return torch.full((self.batch_size, HAND_CANDIDATE_MAX), -1, dtype=torch.long, device=self.device)

        card_vals = self.tokens[:, HAND_CANDIDATE_START : HAND_CANDIDATE_START + HAND_CANDIDATE_MAX, 5:10].long()
        active = card_vals > 0
        card_idx = (card_vals - 1).clamp(0, MAX_HAND_SIZE - 1)
        slot_bits_tensor = _slot_bits(self.device)
        bits_per_card = slot_bits_tensor[card_idx]
        bits_per_card = torch.where(active, bits_per_card, torch.zeros_like(bits_per_card))
        # Card slots within a candidate are distinct, so per-bit OR == sum here.
        bits = bits_per_card.sum(dim=-1)

        subset_idx = _bit_to_hand_subset(self.device)[bits.clamp(0, _BIT_TABLE_SIZE - 1)]
        valid = subset_idx >= 0
        base = int(ActionRange.PLAY_SUBSET_START if is_play else ActionRange.DISCARD_SUBSET_START)
        actions = base + subset_idx.clamp_min(0)
        return torch.where(valid, actions, torch.full_like(actions, -1))

    def _sample_candidate_actions(self, *, is_play: bool, greedy: bool) -> torch.Tensor:
        cand_logits, cand_valid, mapped_actions = self._candidate_distribution(is_play=is_play)
        has_candidates = cand_valid.any(dim=-1)
        if not has_candidates.any():
            return torch.full((self.batch_size,), -1, dtype=torch.long, device=self.device)

        cand_idx = _masked_argmax(cand_logits, cand_valid) if greedy else _sample_masked(cand_logits, cand_valid)

        actions = mapped_actions.gather(1, cand_idx.unsqueeze(-1)).squeeze(-1)
        return torch.where(has_candidates, actions, torch.full_like(actions, -1))

    def _sample_hand_actions_autoregressive(self, *, is_play: bool, greedy: bool) -> torch.Tensor:
        family = 0 if is_play else 1
        valid_subsets = self._hand_valid_subset_mask(is_play)
        count_mask = _subset_count_mask(valid_subsets, _hand_subset_sizes(self.device), max_count=5)
        if greedy:
            count = _masked_argmax(self._t(self.output.hand_count_logits[:, family]), count_mask) + 1
        else:
            count = _sample_masked(self._t(self.output.hand_count_logits[:, family]), count_mask) + 1
        bits = _sample_ordered_cards(
            self._t(self.output.hand_card_logits[:, family]),
            valid_subsets,
            count,
            max_count=5,
            subset_bits=_hand_subset_bits(self.device),
            subset_sizes=_hand_subset_sizes(self.device),
            bit_to_subset=_bit_to_hand_subset(self.device),
            slot_bits=_slot_bits(self.device),
            greedy=greedy,
        )
        subset_idx = _bit_to_hand_subset(self.device)[bits].clamp_min(0)
        base = int(ActionRange.PLAY_SUBSET_START if is_play else ActionRange.DISCARD_SUBSET_START)
        return base + subset_idx

    def _consumable_blocks(self) -> torch.Tensor:
        start = int(ActionRange.CONSUMABLE_FLAT_START)
        end = int(ActionRange.CONSUMABLE_FLAT_END) + 1
        return self.action_mask[:, start:end].view(
            self.batch_size,
            MAX_CONSUMABLE_SLOTS,
            CONSUMABLE_ACTIONS_PER_SLOT,
        )

    def _consumable_no_target_slot_mask(self) -> torch.Tensor:
        return self._consumable_blocks()[:, :, CONSUMABLE_NO_TARGET_OFFSET]

    def _consumable_hand_slot_mask(self) -> torch.Tensor:
        blocks = self._consumable_blocks()
        start = CONSUMABLE_HAND_SUBSET_OFFSET
        end = start + NUM_CONSUMABLE_HAND_SUBSETS
        return blocks[:, :, start:end].any(dim=-1)

    def _consumable_joker_slot_mask(self) -> torch.Tensor:
        blocks = self._consumable_blocks()
        start = CONSUMABLE_JOKER_OFFSET
        end = start + MAX_JOKER_SLOTS
        return blocks[:, :, start:end].any(dim=-1)

    def _consumable_no_target_log_prob(self, slot: torch.Tensor) -> torch.Tensor:
        return self._indexed_log_prob(
            self._t(self.output.consumable_slot_logits[:, 0]),
            self._consumable_no_target_slot_mask(),
            slot.clamp(0, MAX_CONSUMABLE_SLOTS - 1),
        )

    def _consumable_hand_log_prob(
        self,
        slot: torch.Tensor,
        detail: torch.Tensor,
        count: torch.Tensor,
    ) -> torch.Tensor:
        slot = slot.clamp(0, MAX_CONSUMABLE_SLOTS - 1)
        blocks = self._consumable_blocks()
        hand_start = CONSUMABLE_HAND_SUBSET_OFFSET
        hand_end = hand_start + NUM_CONSUMABLE_HAND_SUBSETS
        valid_subsets = blocks[:, :, hand_start:hand_end][torch.arange(self.batch_size, device=self.device), slot]
        count_mask = _subset_count_mask(
            valid_subsets,
            _consumable_subset_sizes(self.device),
            max_count=MAX_CONSUMABLE_HAND_TARGETS,
        )

        slot_logp = self._indexed_log_prob(
            self._t(self.output.consumable_slot_logits[:, 1]),
            self._consumable_hand_slot_mask(),
            slot,
        )
        count_logp = self._indexed_log_prob(
            self._t(self.output.consumable_count_logits[torch.arange(self.batch_size, device=self.device), slot]),
            count_mask,
            (count - 1).clamp(0, MAX_CONSUMABLE_HAND_TARGETS - 1),
        )
        slots = _consumable_subset_slots(self.device)[detail.clamp(0, NUM_CONSUMABLE_HAND_SUBSETS - 1)]
        card_logits = self._t(
            self.output.consumable_card_logits[torch.arange(self.batch_size, device=self.device), slot]
        )
        card_logp = _ordered_card_log_prob(
            card_logits,
            valid_subsets,
            slots,
            count.clamp(1, MAX_CONSUMABLE_HAND_TARGETS),
            max_count=MAX_CONSUMABLE_HAND_TARGETS,
            subset_bits=_consumable_subset_bits(self.device),
            subset_sizes=_consumable_subset_sizes(self.device),
            slot_bits=_slot_bits(self.device),
        )
        return slot_logp + count_logp + card_logp

    def _consumable_joker_log_prob(self, slot: torch.Tensor, joker: torch.Tensor) -> torch.Tensor:
        slot = slot.clamp(0, MAX_CONSUMABLE_SLOTS - 1)
        blocks = self._consumable_blocks()
        joker_start = CONSUMABLE_JOKER_OFFSET
        joker_end = joker_start + MAX_JOKER_SLOTS
        joker_mask = blocks[:, :, joker_start:joker_end][torch.arange(self.batch_size, device=self.device), slot]
        slot_logp = self._indexed_log_prob(
            self._t(self.output.consumable_slot_logits[:, 2]),
            self._consumable_joker_slot_mask(),
            slot,
        )
        joker_logp = self._indexed_log_prob(
            self._t(self.output.consumable_joker_logits[torch.arange(self.batch_size, device=self.device), slot]),
            joker_mask,
            joker.clamp(0, MAX_JOKER_SLOTS - 1),
        )
        return slot_logp + joker_logp

    def _sample_consumable_no_target_actions(self) -> torch.Tensor:
        slot = _sample_masked(self._t(self.output.consumable_slot_logits[:, 0]), self._consumable_no_target_slot_mask())
        return int(ActionRange.CONSUMABLE_FLAT_START) + slot * CONSUMABLE_ACTIONS_PER_SLOT + CONSUMABLE_NO_TARGET_OFFSET

    def _greedy_consumable_no_target_actions(self) -> torch.Tensor:
        slot = _masked_argmax(self._t(self.output.consumable_slot_logits[:, 0]), self._consumable_no_target_slot_mask())
        return int(ActionRange.CONSUMABLE_FLAT_START) + slot * CONSUMABLE_ACTIONS_PER_SLOT + CONSUMABLE_NO_TARGET_OFFSET

    def _sample_consumable_hand_actions(self) -> torch.Tensor:
        return self._build_consumable_hand_actions(greedy=False)

    def _greedy_consumable_hand_actions(self) -> torch.Tensor:
        return self._build_consumable_hand_actions(greedy=True)

    def _build_consumable_hand_actions(self, *, greedy: bool) -> torch.Tensor:
        slot_mask = self._consumable_hand_slot_mask()
        slot_logits = self._t(self.output.consumable_slot_logits[:, 1])
        slot = _masked_argmax(slot_logits, slot_mask) if greedy else _sample_masked(slot_logits, slot_mask)

        blocks = self._consumable_blocks()
        hand_start = CONSUMABLE_HAND_SUBSET_OFFSET
        hand_end = hand_start + NUM_CONSUMABLE_HAND_SUBSETS
        rows = torch.arange(self.batch_size, device=self.device)
        valid_subsets = blocks[:, :, hand_start:hand_end][rows, slot]
        count_mask = _subset_count_mask(
            valid_subsets,
            _consumable_subset_sizes(self.device),
            max_count=MAX_CONSUMABLE_HAND_TARGETS,
        )
        count_logits = self._t(self.output.consumable_count_logits[rows, slot])
        count = (_masked_argmax(count_logits, count_mask) if greedy else _sample_masked(count_logits, count_mask)) + 1
        card_logits = self._t(self.output.consumable_card_logits[rows, slot])
        bits = _sample_ordered_cards(
            card_logits,
            valid_subsets,
            count,
            max_count=MAX_CONSUMABLE_HAND_TARGETS,
            subset_bits=_consumable_subset_bits(self.device),
            subset_sizes=_consumable_subset_sizes(self.device),
            bit_to_subset=_bit_to_consumable_subset(self.device),
            slot_bits=_slot_bits(self.device),
            greedy=greedy,
        )
        detail = _bit_to_consumable_subset(self.device)[bits].clamp_min(0)
        return (
            int(ActionRange.CONSUMABLE_FLAT_START)
            + slot * CONSUMABLE_ACTIONS_PER_SLOT
            + CONSUMABLE_HAND_SUBSET_OFFSET
            + detail
        )

    def _sample_consumable_joker_actions(self) -> torch.Tensor:
        return self._build_consumable_joker_actions(greedy=False)

    def _greedy_consumable_joker_actions(self) -> torch.Tensor:
        return self._build_consumable_joker_actions(greedy=True)

    def _build_consumable_joker_actions(self, *, greedy: bool) -> torch.Tensor:
        slot_mask = self._consumable_joker_slot_mask()
        slot_logits = self._t(self.output.consumable_slot_logits[:, 2])
        slot = _masked_argmax(slot_logits, slot_mask) if greedy else _sample_masked(slot_logits, slot_mask)
        blocks = self._consumable_blocks()
        joker_start = CONSUMABLE_JOKER_OFFSET
        joker_end = joker_start + MAX_JOKER_SLOTS
        rows = torch.arange(self.batch_size, device=self.device)
        joker_mask = blocks[:, :, joker_start:joker_end][rows, slot]
        joker_logits = self._t(self.output.consumable_joker_logits[rows, slot])
        joker = _masked_argmax(joker_logits, joker_mask) if greedy else _sample_masked(joker_logits, joker_mask)
        return (
            int(ActionRange.CONSUMABLE_FLAT_START)
            + slot * CONSUMABLE_ACTIONS_PER_SLOT
            + CONSUMABLE_JOKER_OFFSET
            + joker
        )

    def _consumable_hand_entropy(self) -> torch.Tensor:
        slot_mask = self._consumable_hand_slot_mask()
        return _masked_entropy(self._t(self.output.consumable_slot_logits[:, 1]), slot_mask)

    def _consumable_joker_entropy(self) -> torch.Tensor:
        slot_mask = self._consumable_joker_slot_mask()
        return _masked_entropy(self._t(self.output.consumable_slot_logits[:, 2]), slot_mask)

    def _indexed_log_prob(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = _masked_log_softmax(logits, mask)
        return log_probs.gather(1, target.long().unsqueeze(-1)).squeeze(-1)

    def _sample_indexed_actions(self, logits: torch.Tensor, mask: torch.Tensor, base: int) -> torch.Tensor:
        return base + _sample_masked(logits, mask)

    def _greedy_indexed_actions(self, logits: torch.Tensor, mask: torch.Tensor, base: int) -> torch.Tensor:
        return base + _masked_argmax(logits, mask)


def _range_mask(action_mask: torch.Tensor, start: int | ActionRange, end: int | ActionRange) -> torch.Tensor:
    return action_mask[:, int(start) : int(end) + 1]


def _macro_valid_mask(action_mask: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(
        action_mask.shape[0],
        NUM_GRAMMAR_ACTIONS,
        dtype=torch.bool,
        device=action_mask.device,
    )
    for action_type in (
        ActionType.BLIND_PLAY,
        ActionType.BLIND_SKIP,
        ActionType.BLIND_REROLL,
        ActionType.SHOP_REROLL,
        ActionType.SHOP_LEAVE,
        ActionType.PACK_SKIP,
    ):
        action_id = decode_action_id_for_singleton(action_type)
        mask[:, ACTION_TYPE_TO_GRAMMAR_INDEX[action_type]] = action_mask[:, action_id]

    mask[:, _PLAY] = _range_mask(action_mask, ActionRange.PLAY_SUBSET_START, ActionRange.PLAY_SUBSET_END).any(dim=-1)
    mask[:, _DISCARD] = _range_mask(
        action_mask,
        ActionRange.DISCARD_SUBSET_START,
        ActionRange.DISCARD_SUBSET_END,
    ).any(dim=-1)

    blocks = _range_mask(
        action_mask,
        ActionRange.CONSUMABLE_FLAT_START,
        ActionRange.CONSUMABLE_FLAT_END,
    ).view(action_mask.shape[0], MAX_CONSUMABLE_SLOTS, CONSUMABLE_ACTIONS_PER_SLOT)
    hand_start = CONSUMABLE_HAND_SUBSET_OFFSET
    hand_end = hand_start + NUM_CONSUMABLE_HAND_SUBSETS
    joker_start = CONSUMABLE_JOKER_OFFSET
    joker_end = joker_start + MAX_JOKER_SLOTS
    mask[:, _CONSUMABLE_NO_TARGET] = blocks[:, :, CONSUMABLE_NO_TARGET_OFFSET].any(dim=-1)
    mask[:, _CONSUMABLE_HAND] = blocks[:, :, hand_start:hand_end].flatten(1).any(dim=-1)
    mask[:, _CONSUMABLE_JOKER] = blocks[:, :, joker_start:joker_end].flatten(1).any(dim=-1)

    mask[:, _SHOP_BUY] = _range_mask(action_mask, ActionRange.SHOP_BUY_START, ActionRange.SHOP_BUY_END).any(dim=-1)
    mask[:, _SHOP_SELL_JOKER] = _range_mask(
        action_mask,
        ActionRange.SHOP_SELL_JOKER_START,
        ActionRange.SHOP_SELL_JOKER_END,
    ).any(dim=-1)
    mask[:, _SHOP_SELL_CONSUMABLE] = _range_mask(
        action_mask,
        ActionRange.SHOP_SELL_CONSUMABLE_START,
        ActionRange.SHOP_SELL_CONSUMABLE_END,
    ).any(dim=-1)
    mask[:, _PACK_CLAIM] = _range_mask(
        action_mask,
        ActionRange.PACK_CLAIM_START,
        ActionRange.PACK_CLAIM_END,
    ).any(dim=-1)
    return mask


def decode_action_id_for_singleton(action_type: ActionType) -> int:
    if action_type == ActionType.BLIND_PLAY:
        return int(ActionRange.BLIND_PLAY)
    if action_type == ActionType.BLIND_SKIP:
        return int(ActionRange.BLIND_SKIP)
    if action_type == ActionType.BLIND_REROLL:
        return int(ActionRange.BLIND_REROLL)
    if action_type == ActionType.SHOP_REROLL:
        return int(ActionRange.SHOP_REROLL)
    if action_type == ActionType.SHOP_LEAVE:
        return int(ActionRange.SHOP_LEAVE)
    if action_type == ActionType.PACK_SKIP:
        return int(ActionRange.PACK_SKIP)
    raise ValueError(f"{action_type} is not a singleton action")


def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask.bool(), -1e8)


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(_masked_logits(logits, mask), dim=-1)


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(_masked_logits(logits, mask), dim=-1)
    return probs * mask.to(probs.dtype)


def _masked_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    log_probs = _masked_log_softmax(logits, mask)
    probs = log_probs.exp() * mask.to(logits.dtype)
    entropy = -(probs * log_probs).sum(dim=-1)
    return torch.where(mask.sum(dim=-1) > 1, entropy, torch.zeros_like(entropy))


def _sample_masked(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.distributions.Categorical(logits=_masked_logits(logits, mask)).sample()


def _masked_argmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_logits(logits, mask).argmax(dim=-1)


def _subset_count_mask(
    valid_subset_mask: torch.Tensor,
    subset_sizes: torch.Tensor,
    max_count: int,
) -> torch.Tensor:
    return torch.stack(
        [(valid_subset_mask & subset_sizes.eq(count).unsqueeze(0)).any(dim=-1) for count in range(1, max_count + 1)],
        dim=-1,
    )


def _next_slot_mask(
    valid_subset_mask: torch.Tensor,
    target_counts: torch.Tensor,
    prefix_bits: torch.Tensor,
    last_slots: torch.Tensor,
    subset_bits: torch.Tensor,
    subset_sizes: torch.Tensor,
    slot_bits: torch.Tensor,
) -> torch.Tensor:
    bits = subset_bits.view(1, -1)
    sizes = subset_sizes.view(1, -1)
    counts = target_counts.long().view(-1, 1)
    prefix = prefix_bits.long().view(-1, 1)
    valid = valid_subset_mask & sizes.eq(counts)
    valid = valid & torch.bitwise_and(bits, prefix).eq(prefix)

    per_slot = []
    for slot in range(MAX_HAND_SIZE):
        bit = slot_bits[slot]
        low_mask = (1 << (slot + 1)) - 1
        prefix_with_slot = torch.bitwise_or(prefix_bits.long(), bit)
        low_bits = torch.bitwise_and(bits, low_mask)
        is_next = low_bits.eq(prefix_with_slot.view(-1, 1))
        slot_after_last = last_slots < slot
        per_slot.append((valid & is_next).any(dim=-1) & slot_after_last)
    return torch.stack(per_slot, dim=-1)


def _subset_card_mask(
    valid_subset_mask: torch.Tensor,
    subset_bits: torch.Tensor,
    slot_bits: torch.Tensor,
) -> torch.Tensor:
    bits = subset_bits.view(1, -1)
    per_slot = [
        (valid_subset_mask & torch.bitwise_and(bits, slot_bits[slot]).ne(0)).any(dim=-1)
        for slot in range(MAX_HAND_SIZE)
    ]
    return torch.stack(per_slot, dim=-1)


def _ordered_card_log_prob(
    card_logits: torch.Tensor,
    valid_subset_mask: torch.Tensor,
    target_slots: torch.Tensor,
    target_counts: torch.Tensor,
    *,
    max_count: int,
    subset_bits: torch.Tensor,
    subset_sizes: torch.Tensor,
    slot_bits: torch.Tensor,
) -> torch.Tensor:
    log_prob = torch.zeros(card_logits.shape[0], dtype=card_logits.dtype, device=card_logits.device)
    prefix_bits = torch.zeros(card_logits.shape[0], dtype=torch.long, device=card_logits.device)
    last_slots = torch.full_like(prefix_bits, -1)

    for step in range(max_count):
        active = target_counts > step
        next_mask = _next_slot_mask(
            valid_subset_mask,
            target_counts,
            prefix_bits,
            last_slots,
            subset_bits,
            subset_sizes,
            slot_bits,
        )
        step_target = target_slots[:, step].clamp(0, MAX_HAND_SIZE - 1)
        step_log_probs = _masked_log_softmax(card_logits, next_mask)
        step_log_prob = step_log_probs.gather(1, step_target.unsqueeze(-1)).squeeze(-1)
        log_prob = log_prob + torch.where(active, step_log_prob, torch.zeros_like(step_log_prob))
        step_bits = slot_bits[step_target]
        prefix_bits = torch.where(active, torch.bitwise_or(prefix_bits, step_bits), prefix_bits)
        last_slots = torch.where(active, step_target, last_slots)
    return log_prob


def _sample_ordered_cards(
    card_logits: torch.Tensor,
    valid_subset_mask: torch.Tensor,
    counts: torch.Tensor,
    *,
    max_count: int,
    subset_bits: torch.Tensor,
    subset_sizes: torch.Tensor,
    bit_to_subset: torch.Tensor,
    slot_bits: torch.Tensor,
    greedy: bool,
) -> torch.Tensor:
    prefix_bits = torch.zeros(card_logits.shape[0], dtype=torch.long, device=card_logits.device)
    last_slots = torch.full_like(prefix_bits, -1)
    for step in range(max_count):
        active = counts > step
        next_mask = _next_slot_mask(
            valid_subset_mask,
            counts,
            prefix_bits,
            last_slots,
            subset_bits,
            subset_sizes,
            slot_bits,
        )
        slot = _masked_argmax(card_logits, next_mask) if greedy else _sample_masked(card_logits, next_mask)
        step_bits = slot_bits[slot]
        prefix_bits = torch.where(active, torch.bitwise_or(prefix_bits, step_bits), prefix_bits)
        last_slots = torch.where(active, slot, last_slots)
    safe_prefix_bits = prefix_bits.clamp(0, bit_to_subset.shape[0] - 1)
    valid_bits = bit_to_subset[safe_prefix_bits] >= 0
    return torch.where(valid_bits, prefix_bits, torch.zeros_like(prefix_bits))


def _to_device(array: np.ndarray, device: torch.device, dtype: torch.dtype = torch.long) -> torch.Tensor:
    key = (id(array), str(device), dtype)
    cached = _TENSOR_CACHE.get(key)
    if cached is None:
        cached = torch.as_tensor(array, dtype=dtype, device=device)
        _TENSOR_CACHE[key] = cached
    return cached


def _action_id_to_grammar(device: torch.device) -> torch.Tensor:
    return _to_device(_ACTION_ID_TO_GRAMMAR, device)


def _action_id_to_index(device: torch.device) -> torch.Tensor:
    return _to_device(_ACTION_ID_TO_INDEX, device)


def _action_id_to_detail(device: torch.device) -> torch.Tensor:
    return _to_device(_ACTION_ID_TO_DETAIL, device)


def _action_id_to_count(device: torch.device) -> torch.Tensor:
    return _to_device(_ACTION_ID_TO_COUNT, device)


def _hand_subset_slots(device: torch.device) -> torch.Tensor:
    return _to_device(_HAND_SUBSET_SLOT_PAD, device)


def _consumable_subset_slots(device: torch.device) -> torch.Tensor:
    return _to_device(_CONSUMABLE_SUBSET_SLOT_PAD, device)


def _hand_subset_bits(device: torch.device) -> torch.Tensor:
    return _to_device(_HAND_SUBSET_BITS64, device)


def _hand_subset_sizes(device: torch.device) -> torch.Tensor:
    return _to_device(_HAND_SUBSET_SIZES64, device)


def _consumable_subset_bits(device: torch.device) -> torch.Tensor:
    return _to_device(_CONSUMABLE_SUBSET_BITS64, device)


def _consumable_subset_sizes(device: torch.device) -> torch.Tensor:
    return _to_device(_CONSUMABLE_SUBSET_SIZES64, device)


def _bit_to_hand_subset(device: torch.device) -> torch.Tensor:
    return _to_device(_BIT_TO_HAND_SUBSET_INDEX, device)


def _bit_to_consumable_subset(device: torch.device) -> torch.Tensor:
    return _to_device(_BIT_TO_CONSUMABLE_SUBSET_INDEX, device)


def _slot_bits(device: torch.device) -> torch.Tensor:
    return _to_device(_SLOT_BITS, device)
