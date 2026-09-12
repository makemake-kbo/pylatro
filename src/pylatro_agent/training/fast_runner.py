"""Lightweight game runner that bypasses Gymnasium overhead for fast data generation.

Replicates the BalatroEnv state machine (sub-phase tracking, atomic consumable
commits, shop/booster transitions) but skips observation building, reward
computation, and numpy array creation per step.  Used by fast_generate.py to
run games at minimal cost; qualifying games are replayed with full observations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import cython
import numpy as np

from pylatro import can_use_consumable
from pylatro.blind import blind_multiplier, can_reroll_boss
from pylatro.runtime import consumable_limit, joker_limit
from pylatro.shop import can_claim_pack_card
from pylatro_cli.controller import GameController, GamePhase

from ..constants import (
    CONSUMABLE_ACTIONS_PER_SLOT,
    CONSUMABLE_HAND_SUBSET_OFFSET,
    CONSUMABLE_JOKER_OFFSET,
    CONSUMABLE_NO_TARGET_OFFSET,
    HAND_TARGET_CONSUMABLE_LIMITS,
    JOKER_TARGET_CONSUMABLE_NAMES,
    MAX_CONSUMABLE_HAND_TARGETS,
    MAX_CONSUMABLE_SLOTS,
    MAX_JOKER_SLOTS,
    MAX_PACK_CARDS,
    MAX_SHOP_ITEMS,
    NUM_ACTIONS,
    NUM_CONSUMABLE_HAND_SUBSETS,
    ActionRange,
    SubPhase,
)
from ..history import PlayHistoryTracker, blind_history_key
from ..joker_layout import NO_ORDER_DECISION, apply_best_joker_order
from ..masks import _mask_debuffed_plays, _shop_consumable_allowed
from ..subset_actions import (
    consumable_subset_indices,
    legal_consumable_subset_mask,
    legal_subset_mask,
    subset_indices,
)
from ..survival import validate_critic_win_ante

if TYPE_CHECKING:
    from pylatro.data import GameData
    from pylatro.models import RunState


class FastRunner:
    __slots__ = (
        "_ctrl",
        "_done",
        "_history",
        "_last_order_decision",
        "_mask",
        "_max_ante",
        "_max_steps",
        "_prev_signature",
        "_raise_errors",
        "_round_score",
        "_state",
        "_step_count",
        "_steps_since_progress",
        "_sub_phase",
        "_won",
    )

    def __init__(
        self,
        seed: int,
        data: GameData,
        *,
        max_steps: int = 2000,
        win_ante: int = 8,
        deck_key: str = "b_red",
        stake: int = 1,
        raise_errors: bool = False,
    ) -> None:
        ctrl = GameController(data=data)
        ctrl.new_run(str(seed), stake=stake, deck_key=deck_key)
        assert ctrl.state is not None
        self._ctrl = ctrl
        self._raise_errors = raise_errors
        self._state: RunState = cast("RunState", ctrl.state)
        self._state.win_ante = validate_critic_win_ante(win_ante)
        self._sub_phase: SubPhase = SubPhase.BLIND_SELECT
        self._round_score: int = 0
        self._last_order_decision = NO_ORDER_DECISION
        self._max_ante: int = 1
        self._done: bool = False
        self._history = PlayHistoryTracker()
        self._won: bool = False
        self._step_count: int = 0
        self._steps_since_progress: int = 0
        self._prev_signature: tuple | None = None
        self._mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        self._max_steps = max_steps

    # ── public properties (same interface HeuristicAgent expects) ──

    @property
    def state(self) -> RunState:
        return self._state

    @property
    def sub_phase(self) -> SubPhase:
        return self._sub_phase

    @property
    def max_ante(self) -> int:
        return self._max_ante

    @property
    def done(self) -> bool:
        return self._done

    @property
    def won(self) -> bool:
        return self._won

    @property
    def round_score(self) -> int:
        return self._round_score

    @property
    def last_order_decision(self):
        """Harness joker-ordering outcome for the most recent play."""
        return self._last_order_decision

    @property
    def history(self) -> PlayHistoryTracker:
        return self._history

    @property
    def phase(self) -> GamePhase:
        return self._ctrl.phase

    @property
    def steps_since_progress(self) -> int:
        return self._steps_since_progress

    # ── action mask (pre-allocated buffer) ──

    def compute_mask(self) -> np.ndarray:
        m = self._mask
        m.fill(0)
        AR = ActionRange
        state = self._state
        sp = self._sub_phase

        if sp == SubPhase.BLIND_SELECT:
            _mask_blind(m, state, AR)
        elif sp == SubPhase.CHOOSE_ACTION:
            _mask_action(m, state, AR)
        elif sp == SubPhase.SHOP:
            _mask_shop(m, state, AR)
        elif sp == SubPhase.BOOSTER_PACK:
            _mask_booster(m, state, AR)

        return self._mask

    # ── step (action-id based, no decode_action overhead) ──

    @cython.locals(action_id=cython.int, _step_count=cython.int)
    def step(self, action_id: int) -> Any:
        self._step_count += 1

        try:
            action_result = self._execute(action_id)
        except Exception:
            if self._raise_errors:
                raise
            self._steps_since_progress += 1
            if not self._done and (
                self._steps_since_progress >= self._max_steps or self._step_count >= self._max_steps
            ):
                self._done = True
            return None

        state = self._state
        phase = self._ctrl.phase

        if phase in (GamePhase.GAME_OVER, GamePhase.GAME_WON):
            self._done = True
            self._won = phase == GamePhase.GAME_WON

        self._max_ante = max(self._max_ante, state.round_resets.ante)

        sig = self._progress_signature()
        if self._prev_signature is not None and sig != self._prev_signature:
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1
        self._prev_signature = sig

        if not self._done and (self._steps_since_progress >= self._max_steps or self._step_count >= self._max_steps):
            self._done = True
        return action_result

    # ── internal state machine ──

    @cython.locals(
        aid=cython.int,
        idx=cython.int,
        n_cards=cython.int,
        n_vouchers=cython.int,
        booster_idx=cython.int,
        rel=cython.int,
        slot=cython.int,
        within=cython.int,
        joker_idx=cython.int,
    )
    def _execute(self, aid: int) -> Any:
        ctrl = self._ctrl
        state = self._state
        AR = ActionRange

        if aid == AR.BLIND_PLAY:
            blind_type = state.blind_on_deck or "Small"
            ctrl.select_blind(blind_type)
            self._history.start_round(blind_history_key(state))
            self._sub_phase = SubPhase.CHOOSE_ACTION
            self._round_score = ctrl.round_score

        elif aid == AR.BLIND_SKIP:
            skipped_key = (
                int(state.round_resets.ante),
                str(state.blind_on_deck or "Small"),
                "skipped",
            )
            ctrl.skip_blind()
            self._history.start_round(skipped_key)
            self._sub_phase = SubPhase.BLIND_SELECT

        elif aid == AR.BLIND_REROLL:
            ctrl.reroll_boss()
            self._sub_phase = SubPhase.BLIND_SELECT

        elif AR.PLAY_SUBSET_START <= aid <= AR.PLAY_SUBSET_END:
            idx = aid - AR.PLAY_SUBSET_START
            indices = subset_indices(idx)
            if any(slot >= len(state.hand_cards) for slot in indices):
                return
            self._last_order_decision = apply_best_joker_order(
                state,
                tuple(sorted(indices)),
                remaining_target=max(float(ctrl.blind_target()) - float(self._round_score), 0.0),
            )
            selected_cards = [state.hand_cards[slot] for slot in sorted(indices)]
            pending_history = self._history.capture(
                state,
                selected_cards,
                blind_target=ctrl.blind_target(),
                round_score=self._round_score,
            )
            result = ctrl.play_selected(list(indices))
            # The controller is authoritative: Mr. Bones can consume itself
            # and reset the accumulated round score after a losing final hand.
            # Incrementing our mirror would preserve a score the engine has
            # deliberately cleared and hide the negative potential delta from
            # fast_generate reward shaping.
            self._round_score = ctrl.round_score
            self._history.finalize(
                pending_history,
                hand_type=result.score.hand_name,
                score=result.score.total,
            )
            if ctrl.blind_beaten():
                ctrl.cash_out()
                if ctrl.phase == GamePhase.GAME_WON:
                    return result
                ctrl.enter_shop()
                self._sub_phase = SubPhase.SHOP
            elif ctrl.phase == GamePhase.GAME_OVER:
                return result
            else:
                self._sub_phase = SubPhase.CHOOSE_ACTION
            return result

        elif AR.DISCARD_SUBSET_START <= aid <= AR.DISCARD_SUBSET_END:
            idx = aid - AR.DISCARD_SUBSET_START
            indices = subset_indices(idx)
            if any(slot >= len(state.hand_cards) for slot in indices):
                return
            result = ctrl.discard_selected(list(indices))
            self._sub_phase = SubPhase.CHOOSE_ACTION
            return result

        elif AR.CONSUMABLE_FLAT_START <= aid <= AR.CONSUMABLE_FLAT_END:
            rel = aid - int(AR.CONSUMABLE_FLAT_START)
            slot = rel // CONSUMABLE_ACTIONS_PER_SLOT
            within = rel - slot * CONSUMABLE_ACTIONS_PER_SLOT
            if within == CONSUMABLE_NO_TARGET_OFFSET:
                result = ctrl.use_consumable_on(slot, hand_targets=(), joker_targets=())
            elif CONSUMABLE_HAND_SUBSET_OFFSET <= within < CONSUMABLE_HAND_SUBSET_OFFSET + NUM_CONSUMABLE_HAND_SUBSETS:
                hand_targets = consumable_subset_indices(within - CONSUMABLE_HAND_SUBSET_OFFSET)
                result = ctrl.use_consumable_on(slot, hand_targets=hand_targets, joker_targets=())
            else:
                joker_idx = within - CONSUMABLE_JOKER_OFFSET
                result = ctrl.use_consumable_on(slot, hand_targets=(), joker_targets=(joker_idx,))
            if ctrl.phase == GamePhase.HAND_PLAY:
                self._sub_phase = SubPhase.CHOOSE_ACTION
            elif ctrl.phase == GamePhase.SHOP:
                self._sub_phase = SubPhase.SHOP
            return result

        elif AR.SHOP_BUY_START <= aid <= AR.SHOP_BUY_END:
            idx = aid - AR.SHOP_BUY_START
            n_cards = len(state.shop.cards)
            n_vouchers = len(state.shop.vouchers)
            if idx < n_cards:
                ctrl.buy_card(idx)
            elif idx < n_cards + n_vouchers:
                voucher = state.shop.vouchers[idx - n_cards]
                ctrl.buy_voucher(voucher.center_key)
            else:
                booster_idx = idx - n_cards - n_vouchers
                ctrl.open_pack(booster_idx)
                self._sub_phase = SubPhase.BOOSTER_PACK

        elif aid == AR.SHOP_REROLL:
            ctrl.reroll()

        elif AR.SHOP_SELL_JOKER_START <= aid <= AR.SHOP_SELL_JOKER_END:
            ctrl.sell_joker(aid - AR.SHOP_SELL_JOKER_START)
            if ctrl.phase == GamePhase.SHOP:
                self._sub_phase = SubPhase.SHOP

        elif AR.SHOP_SELL_CONSUMABLE_START <= aid <= AR.SHOP_SELL_CONSUMABLE_END:
            ctrl.sell_consumable(aid - AR.SHOP_SELL_CONSUMABLE_START)

        elif aid == AR.SHOP_LEAVE:
            ctrl.leave_shop()
            self._sub_phase = SubPhase.BLIND_SELECT

        elif AR.PACK_CLAIM_START <= aid <= AR.PACK_CLAIM_END:
            result = ctrl.claim_from_pack(aid - AR.PACK_CLAIM_START)
            if state.pack is None or state.pack.choices_remaining <= 0:
                ctrl.close_current_pack(skipped=False)
                self._sub_phase = SubPhase.SHOP
            return result

        elif aid == AR.PACK_SKIP:
            ctrl.close_current_pack(skipped=True)
            self._sub_phase = SubPhase.SHOP

    def _progress_signature(self):
        state = self._state
        return (
            state.round_resets.ante,
            state.blind_on_deck or "",
            self._round_score,
            state.current_round.hands_left,
            state.current_round.discards_left,
            state.dollars,
            self._sub_phase,
            tuple(sorted(state.joker_keys)),
        )


# ── mask helpers (module-level for speed) ──


@cython.cfunc
@cython.locals(m=cython.char[:], _bp=cython.int, _bs=cython.int, _br=cython.int)
def _mask_blind(m, state, AR):
    _bp = AR.BLIND_PLAY
    _bs = AR.BLIND_SKIP
    _br = AR.BLIND_REROLL
    m[_bp] = 1
    blind_on_deck = state.blind_on_deck or "Small"
    if blind_on_deck in ("Small", "Big"):
        m[_bs] = 1
    if blind_on_deck == "Boss" and can_reroll_boss(state):
        m[_br] = 1


@cython.cfunc
@cython.locals(m=cython.char[:], i=cython.int, _play_start=cython.int, _disc_start=cython.int)
def _mask_action(m, state, AR):
    _play_start = AR.PLAY_SUBSET_START
    _disc_start = AR.DISCARD_SUBSET_START
    for i in range(min(len(state.jokers), MAX_JOKER_SLOTS)):
        if not state.jokers[i].eternal:
            m[AR.SHOP_SELL_JOKER_START + i] = 1
    hand_size = len(state.hand_cards)
    forced_slots = {idx for idx, card in enumerate(state.hand_cards) if card.forced_selection}
    legal_subsets = legal_subset_mask(hand_size, forced_slots)
    n_subsets = len(legal_subsets)
    if state.current_round.hands_left > 0 and hand_size > 0:
        play_subsets = _mask_debuffed_plays(state, legal_subsets)
        if cython.compiled:
            for i in range(n_subsets):
                m[_play_start + i] = 1 if play_subsets[i] else 0
        else:
            m[_play_start : _play_start + n_subsets] = play_subsets
    if state.current_round.discards_left > 0 and hand_size > 0:
        if cython.compiled:
            for i in range(n_subsets):
                m[_disc_start + i] = 1 if legal_subsets[i] else 0
        else:
            m[_disc_start : _disc_start + n_subsets] = legal_subsets
    _mask_consumable_flat(m, state, AR)


@cython.cfunc
@cython.locals(
    m=cython.char[:],
    i=cython.int,
    _buy_start=cython.int,
    _reroll=cython.int,
    _sell_joker=cython.int,
    _sell_cons=cython.int,
    _leave=cython.int,
)
def _mask_shop(m, state, AR):
    _buy_start = AR.SHOP_BUY_START
    _reroll = AR.SHOP_REROLL
    _sell_joker = AR.SHOP_SELL_JOKER_START
    _sell_cons = AR.SHOP_SELL_CONSUMABLE_START
    _leave = AR.SHOP_LEAVE
    all_items = list(state.shop.cards) + list(state.shop.vouchers) + list(state.shop.boosters)
    for i, item in enumerate(all_items[:MAX_SHOP_ITEMS]):
        if item.cost > state.dollars:
            continue
        if item.card_type == "Joker":
            is_negative = bool(item.edition and item.edition.get("negative"))
            if len(state.jokers) < joker_limit(state) or is_negative:
                m[_buy_start + i] = 1
        elif item.card_type in ("Tarot", "Planet", "Spectral"):
            if len(state.consumables) < consumable_limit(state):
                m[_buy_start + i] = 1
        else:
            m[_buy_start + i] = 1

    if state.current_round.reroll_cost <= state.dollars:
        m[_reroll] = 1

    for i, joker in enumerate(state.jokers[:MAX_JOKER_SLOTS]):
        if not joker.eternal:
            m[_sell_joker + i] = 1

    for i in range(min(len(state.consumables), MAX_CONSUMABLE_SLOTS)):
        m[_sell_cons + i] = 1

    m[_leave] = 1
    _mask_consumable_flat(m, state, AR, in_shop=True)


@cython.cfunc
@cython.locals(m=cython.char[:], i=cython.int, _claim_start=cython.int, _skip=cython.int)
def _mask_booster(m, state, AR):
    _claim_start = AR.PACK_CLAIM_START
    _skip = AR.PACK_SKIP
    pack = state.pack
    for i in range(min(len(state.jokers), MAX_JOKER_SLOTS)):
        if not state.jokers[i].eternal:
            m[AR.SHOP_SELL_JOKER_START + i] = 1
    if pack and pack.choices_remaining > 0:
        for i in range(min(len(pack.cards), MAX_PACK_CARDS)):
            card = pack.cards[i]
            center = state.data.centers.get(card.center_key, {})
            card_type = ""
            if center.get("set") == "Joker":
                card_type = "Joker"
            elif center.get("consumeable"):
                card_type = str(center.get("set", ""))

            if card_type == "Joker":
                is_negative = bool(card.edition and card.edition.get("negative"))
                if len(state.jokers) < joker_limit(state) or is_negative:
                    m[_claim_start + i] = 1
            elif card_type in ("Tarot", "Planet", "Spectral"):
                if can_claim_pack_card(state, card):
                    m[_claim_start + i] = 1
            else:
                m[_claim_start + i] = 1
    m[_skip] = 1


@cython.cfunc
@cython.locals(
    m=cython.char[:],
    base=cython.int,
    slot=cython.int,
    slot_base=cython.int,
    hand_size=cython.int,
    num_jokers=cython.int,
    min_size=cython.int,
    max_size=cython.int,
    j=cython.int,
    start=cython.int,
    end=cython.int,
    i=cython.int,
    _no_target_off=cython.int,
    _hand_sub_off=cython.int,
    _joker_off=cython.int,
    _per_slot=cython.int,
)
def _mask_consumable_flat(m, state, AR, in_shop=False):
    base = int(AR.CONSUMABLE_FLAT_START)
    num_jokers = min(len(state.jokers), MAX_JOKER_SLOTS)
    hand_size = len(state.hand_cards)
    if hand_size > 16:
        hand_size = 16

    _no_target_off = int(CONSUMABLE_NO_TARGET_OFFSET)
    _hand_sub_off = int(CONSUMABLE_HAND_SUBSET_OFFSET)
    _joker_off = int(CONSUMABLE_JOKER_OFFSET)
    _per_slot = int(CONSUMABLE_ACTIONS_PER_SLOT)

    for slot in range(min(len(state.consumables), MAX_CONSUMABLE_SLOTS)):
        cons = state.consumables[slot]
        center = state.data.centers[cons.center_key]
        if in_shop and not _shop_consumable_allowed(center):
            continue
        config = center.get("config") or {}
        max_highlighted = config.get("max_highlighted")
        name = center.get("name", "")
        needs_joker_target = name in JOKER_TARGET_CONSUMABLE_NAMES
        fallback_hand_limits = HAND_TARGET_CONSUMABLE_LIMITS.get(name)

        slot_base = base + slot * _per_slot

        if max_highlighted is None and fallback_hand_limits is None and not needs_joker_target:
            if can_use_consumable(state, cons, hand_targets=(), joker_targets=()):
                m[slot_base + _no_target_off] = 1
            continue

        if max_highlighted is not None or fallback_hand_limits is not None:
            if fallback_hand_limits is not None:
                min_size, raw_max = fallback_hand_limits
            else:
                min_size = int(config.get("min_highlighted", 1) or 1)
                raw_max = int(max_highlighted)
            max_size = raw_max if raw_max < MAX_CONSUMABLE_HAND_TARGETS else MAX_CONSUMABLE_HAND_TARGETS
            subset_mask = legal_consumable_subset_mask(hand_size, min_size, max_size)
            start = slot_base + _hand_sub_off
            for i in range(NUM_CONSUMABLE_HAND_SUBSETS):
                if not subset_mask[i]:
                    continue
                hand_targets = consumable_subset_indices(i)
                if can_use_consumable(state, cons, hand_targets=hand_targets, joker_targets=()):
                    m[start + i] = 1

        if needs_joker_target:
            start = slot_base + _joker_off
            for j in range(num_jokers):
                if can_use_consumable(state, cons, hand_targets=(), joker_targets=(j,)):
                    m[start + j] = 1


@cython.locals(ante=cython.int, scaling=cython.int, base=cython.int)
def _blind_target(state):
    from pylatro import get_blind_amount

    blind = state.round_resets.blind
    if blind is None:
        return 0
    ante = state.round_resets.ante
    scaling = min(state.stake, 3)
    base = get_blind_amount(ante, scaling)
    mult = blind_multiplier(state, blind)
    return int(base * mult)
