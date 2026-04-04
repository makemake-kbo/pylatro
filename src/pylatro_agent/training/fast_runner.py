"""Lightweight game runner that bypasses Gymnasium overhead for fast data generation.

Replicates the BalatroEnv state machine (sub-phase tracking, card selection,
consumable targeting, shop/booster transitions) but skips observation building,
reward computation, and numpy array creation per step.  Used by fast_generate.py
to run games at minimal cost; qualifying games are replayed with full observations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import cython
import numpy as np

from pylatro import can_use_consumable
from pylatro.runtime import consumable_limit, joker_limit
from pylatro_cli.controller import GameController, GamePhase

from ..constants import (
    MAX_CONSUMABLE_SLOTS,
    MAX_DISCARD_CANDIDATES,
    MAX_HAND_SIZE,
    MAX_PLAY_CANDIDATES,
    MAX_JOKER_SLOTS,
    MAX_PACK_CARDS,
    MAX_SHOP_ITEMS,
    NUM_ACTIONS,
    ActionRange,
    SubPhase,
)
from ..hand_candidates import HandCandidate, candidate_signature, generate_hand_candidates

if TYPE_CHECKING:
    from pylatro.data import GameData
    from pylatro.models import RunState


class FastRunner:
    __slots__ = (
        "_blind_just_beaten",
        "_ctrl",
        "_done",
        "_mask",
        "_max_ante",
        "_max_steps",
        "_pending_action",
        "_pending_consumable_slot",
        "_pending_hand_targets",
        "_pending_joker_targets",
        "_play_candidates",
        "_prev_signature",
        "_round_score",
        "_candidate_signature",
        "_discard_candidates",
        "_selected_cards",
        "_state",
        "_step_count",
        "_steps_since_progress",
        "_sub_phase",
        "_won",
    )

    def __init__(self, seed: int, data: GameData, *, max_steps: int = 2000) -> None:
        ctrl = GameController(data=data)
        ctrl.new_run(str(seed))
        assert ctrl.state is not None
        self._ctrl = ctrl
        self._state: RunState = cast("RunState", ctrl.state)
        self._sub_phase: SubPhase = SubPhase.BLIND_SELECT
        self._selected_cards: set[int] = set()
        self._pending_action: str | None = None
        self._pending_consumable_slot: int | None = None
        self._pending_hand_targets: tuple[int, ...] = ()
        self._pending_joker_targets: tuple[int, ...] = ()
        self._play_candidates: tuple[HandCandidate, ...] = ()
        self._discard_candidates: tuple[HandCandidate, ...] = ()
        self._candidate_signature: tuple | None = None
        self._round_score: int = 0
        self._max_ante: int = 1
        self._done: bool = False
        self._won: bool = False
        self._blind_just_beaten: bool = False
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
    def selected_cards(self) -> set[int]:
        return self._selected_cards

    @property
    def pending_action(self) -> str | None:
        return self._pending_action

    @property
    def pending_consumable_slot(self) -> int | None:
        return self._pending_consumable_slot

    @property
    def max_ante(self) -> int:
        return self._max_ante

    @property
    def play_candidates(self) -> tuple[HandCandidate, ...]:
        return self._play_candidates

    @property
    def discard_candidates(self) -> tuple[HandCandidate, ...]:
        return self._discard_candidates

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
    def blind_just_beaten(self) -> bool:
        return self._blind_just_beaten

    @property
    def phase(self) -> GamePhase:
        return self._ctrl.phase

    @property
    def steps_since_progress(self) -> int:
        return self._steps_since_progress

    # ── action mask (pre-allocated buffer) ──

    @cython.locals(mv=cython.char[:])
    def compute_mask(self) -> np.ndarray:
        m = self._mask
        m.fill(0)
        mv = m
        AR = ActionRange
        state = self._state
        sp = self._sub_phase

        if sp == SubPhase.BLIND_SELECT:
            _mask_blind(mv, state, AR)
        elif sp == SubPhase.CHOOSE_ACTION:
            self._refresh_hand_candidates()
            _mask_action(mv, state, AR, len(self._play_candidates), len(self._discard_candidates))
        elif sp == SubPhase.SELECT_CARDS:
            _mask_cards(mv, state, AR, self._selected_cards, self._pending_action)
        elif sp == SubPhase.SHOP:
            _mask_shop(mv, state, AR)
        elif sp == SubPhase.BOOSTER_PACK:
            _mask_booster(mv, state, AR)
        elif sp == SubPhase.CONSUMABLE_TARGET:
            _mask_consumable(
                mv,
                state,
                AR,
                self._pending_consumable_slot,
                self._pending_hand_targets,
                self._pending_joker_targets,
            )

        return self._mask

    # ── step (action-id based, no decode_action overhead) ──

    @cython.locals(action_id=cython.int, _step_count=cython.int)
    def step(self, action_id: int) -> None:
        self._blind_just_beaten = False
        self._step_count += 1

        try:
            self._execute(action_id)
        except Exception:
            self._steps_since_progress += 1
            if not self._done and self._steps_since_progress >= self._max_steps:
                self._done = True
            return

        state = self._state
        phase = self._ctrl.phase

        if phase in (GamePhase.GAME_OVER, GamePhase.GAME_WON):
            self._done = True
            self._won = phase == GamePhase.GAME_WON

        self._max_ante = max(self._max_ante, state.round_resets.ante)

        sig = _progress_signature(state, self._ctrl.phase, self._sub_phase, self._round_score)
        if self._prev_signature is not None and sig != self._prev_signature:
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1
        self._prev_signature = sig

        if not self._done and self._steps_since_progress >= self._max_steps:
            self._done = True

    # ── internal state machine ──

    @cython.locals(
        aid=cython.int,
        idx=cython.int,
        n_cards=cython.int,
        n_vouchers=cython.int,
        booster_idx=cython.int,
    )
    def _execute(self, aid: int) -> None:
        ctrl = self._ctrl
        state = self._state
        AR = ActionRange

        if aid == AR.BLIND_PLAY:
            blind_type = state.blind_on_deck or "Small"
            ctrl.select_blind(blind_type)
            self._sub_phase = SubPhase.CHOOSE_ACTION
            self._round_score = 0

        elif aid == AR.BLIND_SKIP:
            ctrl.skip_blind()
            self._sub_phase = SubPhase.BLIND_SELECT

        elif aid == AR.BLIND_REROLL:
            ctrl.reroll_boss()
            self._sub_phase = SubPhase.BLIND_SELECT

        elif AR.PLAY_CANDIDATE_START <= aid <= AR.PLAY_CANDIDATE_END:
            self._refresh_hand_candidates()
            idx = aid - AR.PLAY_CANDIDATE_START
            if idx >= len(self._play_candidates):
                return
            result = ctrl.play_selected(list(self._play_candidates[idx].indices))
            self._round_score += result.score.total
            if ctrl.blind_beaten():
                self._blind_just_beaten = True
                ctrl.cash_out()
                if ctrl.phase == GamePhase.GAME_WON:
                    return
                ctrl.enter_shop()
                self._sub_phase = SubPhase.SHOP
            elif ctrl.phase == GamePhase.GAME_OVER:
                return
            else:
                self._sub_phase = SubPhase.CHOOSE_ACTION

        elif AR.DISCARD_CANDIDATE_START <= aid <= AR.DISCARD_CANDIDATE_END:
            self._refresh_hand_candidates()
            idx = aid - AR.DISCARD_CANDIDATE_START
            if idx >= len(self._discard_candidates):
                return
            ctrl.discard_selected(list(self._discard_candidates[idx].indices))
            self._sub_phase = SubPhase.CHOOSE_ACTION

        elif aid == AR.USE_CONSUMABLE:
            self._pending_consumable_slot = None
            self._pending_hand_targets = ()
            self._pending_joker_targets = ()
            self._sub_phase = SubPhase.CONSUMABLE_TARGET

        elif AR.CONSUMABLE_SLOT_START <= aid <= AR.CONSUMABLE_SLOT_END:
            self._pending_consumable_slot = aid - AR.CONSUMABLE_SLOT_START
            self._pending_hand_targets = ()
            self._pending_joker_targets = ()

        elif AR.CONSUMABLE_HAND_TARGET_START <= aid <= AR.CONSUMABLE_HAND_TARGET_END:
            idx = aid - AR.CONSUMABLE_HAND_TARGET_START
            targets = list(self._pending_hand_targets)
            if idx in targets:
                targets.remove(idx)
            else:
                targets.append(idx)
            self._pending_hand_targets = tuple(targets)

        elif AR.CONSUMABLE_JOKER_TARGET_START <= aid <= AR.CONSUMABLE_JOKER_TARGET_END:
            idx = aid - AR.CONSUMABLE_JOKER_TARGET_START
            targets = list(self._pending_joker_targets)
            if idx in targets:
                targets.remove(idx)
            else:
                targets.append(idx)
            self._pending_joker_targets = tuple(targets)

        elif aid == AR.CONSUMABLE_CONFIRM:
            slot = self._pending_consumable_slot
            if slot is not None:
                ctrl.use_consumable_on(
                    slot,
                    hand_targets=self._pending_hand_targets,
                    joker_targets=self._pending_joker_targets,
                )
            self._pending_consumable_slot = None
            self._pending_hand_targets = ()
            self._pending_joker_targets = ()
            if ctrl.phase == GamePhase.HAND_PLAY:
                self._sub_phase = SubPhase.CHOOSE_ACTION
            elif ctrl.phase == GamePhase.SHOP:
                self._sub_phase = SubPhase.SHOP

        elif aid == AR.CONSUMABLE_CANCEL:
            self._pending_consumable_slot = None
            self._pending_hand_targets = ()
            self._pending_joker_targets = ()
            if ctrl.phase == GamePhase.HAND_PLAY:
                self._sub_phase = SubPhase.CHOOSE_ACTION
            elif ctrl.phase == GamePhase.SHOP:
                self._sub_phase = SubPhase.SHOP

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

        elif AR.SHOP_SELL_CONSUMABLE_START <= aid <= AR.SHOP_SELL_CONSUMABLE_END:
            ctrl.sell_consumable(aid - AR.SHOP_SELL_CONSUMABLE_START)

        elif aid == AR.SHOP_LEAVE:
            ctrl.leave_shop()
            self._sub_phase = SubPhase.BLIND_SELECT

        elif AR.PACK_CLAIM_START <= aid <= AR.PACK_CLAIM_END:
            ctrl.claim_from_pack(aid - AR.PACK_CLAIM_START)
            if state.pack and state.pack.choices_remaining <= 0:
                ctrl.close_current_pack(skipped=False)
                self._sub_phase = SubPhase.SHOP

        elif aid == AR.PACK_SKIP:
            ctrl.close_current_pack(skipped=True)
            self._sub_phase = SubPhase.SHOP

    def _refresh_hand_candidates(self) -> None:
        if self._sub_phase != SubPhase.CHOOSE_ACTION or self._ctrl.phase != GamePhase.HAND_PLAY:
            self._play_candidates = ()
            self._discard_candidates = ()
            self._candidate_signature = None
            return

        sig = candidate_signature(self._state)
        if sig == self._candidate_signature:
            return
        play_candidates, discard_candidates = generate_hand_candidates(self._state)
        self._play_candidates = play_candidates
        self._discard_candidates = discard_candidates
        self._candidate_signature = sig


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
    if blind_on_deck == "Boss" and state.dollars >= 10 and not state.round_resets.boss_rerolled:
        m[_br] = 1


@cython.cfunc
@cython.locals(m=cython.char[:], i=cython.int, _play_start=cython.int, _disc_start=cython.int, _use=cython.int)
def _mask_action(m, state, AR, play_count, discard_count):
    _play_start = AR.PLAY_CANDIDATE_START
    _disc_start = AR.DISCARD_CANDIDATE_START
    _use = AR.USE_CONSUMABLE
    hand_size = len(state.hand_cards)
    if state.current_round.hands_left > 0 and hand_size > 0:
        for i in range(min(play_count, MAX_PLAY_CANDIDATES)):
            m[_play_start + i] = 1
    if state.current_round.discards_left > 0 and hand_size > 0:
        for i in range(min(discard_count, MAX_DISCARD_CANDIDATES)):
            m[_disc_start + i] = 1
    for cons in state.consumables:
        if can_use_consumable(state, cons):
            m[_use] = 1
            break


@cython.cfunc
@cython.locals(
    m=cython.char[:],
    hand_size=cython.int,
    num_sel=cython.int,
    is_play=cython.bint,
    max_sel=cython.int,
    i=cython.int,
)
def _mask_cards(m, state, AR, selected, pending):
    _ = (m, state, AR, selected, pending)


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
            if len(state.jokers) < joker_limit(state):
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


@cython.cfunc
@cython.locals(m=cython.char[:], i=cython.int, _claim_start=cython.int, _skip=cython.int)
def _mask_booster(m, state, AR):
    _claim_start = AR.PACK_CLAIM_START
    _skip = AR.PACK_SKIP
    pack = state.pack
    if pack and pack.choices_remaining > 0:
        for i in range(min(len(pack.cards), MAX_PACK_CARDS)):
            m[_claim_start + i] = 1
    m[_skip] = 1


@cython.cfunc
@cython.locals(
    m=cython.char[:],
    i=cython.int,
    max_highlighted_int=cython.int,
    required_min=cython.int,
    required_max=cython.int,
    current_count=cython.int,
    _slot_start=cython.int,
    _hand_target_start=cython.int,
    _confirm=cython.int,
    _joker_target_start=cython.int,
    _cancel=cython.int,
)
def _mask_consumable(m, state, AR, pending_slot, hand_targets, joker_targets):
    _slot_start = AR.CONSUMABLE_SLOT_START
    _hand_target_start = AR.CONSUMABLE_HAND_TARGET_START
    _confirm = AR.CONSUMABLE_CONFIRM
    _joker_target_start = AR.CONSUMABLE_JOKER_TARGET_START
    _cancel = AR.CONSUMABLE_CANCEL
    if pending_slot is None:
        for i, cons in enumerate(state.consumables[:MAX_CONSUMABLE_SLOTS]):
            if can_use_consumable(state, cons):
                m[_slot_start + i] = 1
    else:
        cons = state.consumables[pending_slot]
        center = state.data.centers[cons.center_key]
        config = center.get("config") or {}
        max_highlighted = config.get("max_highlighted")

        if max_highlighted is not None:
            required_min = int(config.get("min_highlighted", 1) or 1)
            required_max = int(max_highlighted or 0)
            current_count = len(hand_targets)

            if current_count < required_max:
                for i in range(min(len(state.hand_cards), MAX_HAND_SIZE)):
                    if i not in hand_targets:
                        m[_hand_target_start + i] = 1

            if required_min <= current_count <= required_max and can_use_consumable(
                state,
                cons,
                hand_targets=hand_targets,
                joker_targets=joker_targets,
            ):
                m[_confirm] = 1
        else:
            if can_use_consumable(state, cons, hand_targets=hand_targets, joker_targets=joker_targets):
                m[_confirm] = 1

        name = center.get("name", "")
        if name in ("The Wheel of Fortune", "Ectoplasm", "Hex", "Ankh"):
            for i in range(min(len(state.jokers), MAX_JOKER_SLOTS)):
                if i not in joker_targets:
                    m[_joker_target_start + i] = 1

    m[_cancel] = 1


# ── progress signature (mirrors BalatroEnv._progress_signature) ──


@cython.locals(round_score=cython.int, shop_count=cython.int, pack_choices=cython.int)
def _progress_signature(state, phase, sub_phase, round_score):
    pack_choices = state.pack.choices_remaining if state.pack is not None else 0
    shop_count = len(state.shop.cards) + len(state.shop.vouchers) + len(state.shop.boosters)
    return (
        state.round_resets.ante,
        state.blind_on_deck or "",
        _blind_target(state),
        round_score,
        state.current_round.hands_left,
        state.current_round.discards_left,
        state.dollars,
        phase,
        sub_phase,
        len(state.jokers),
        len(state.consumables),
        shop_count,
        pack_choices,
    )


@cython.locals(ante=cython.int, scaling=cython.int, base=cython.int)
def _blind_target(state):
    from pylatro import get_blind_amount

    blind = state.round_resets.blind
    if blind is None:
        return 0
    ante = state.round_resets.ante
    scaling = min(state.stake, 3)
    base = get_blind_amount(ante, scaling)
    mult = blind.get("mult", 1)
    return int(base * mult)
