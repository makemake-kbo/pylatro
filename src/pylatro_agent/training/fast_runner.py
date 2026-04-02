"""Lightweight game runner that bypasses Gymnasium overhead for fast data generation.

Replicates the BalatroEnv state machine (sub-phase tracking, card selection,
consumable targeting, shop/booster transitions) but skips observation building,
reward computation, and numpy array creation per step.  Used by fast_generate.py
to run games at minimal cost; qualifying games are replayed with full observations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from pylatro import can_use_consumable
from pylatro.runtime import consumable_limit, joker_limit
from pylatro_cli.controller import GameController, GamePhase

from ..constants import (
    MAX_CONSUMABLE_SLOTS,
    MAX_HAND_SIZE,
    MAX_JOKER_SLOTS,
    MAX_PACK_CARDS,
    MAX_SHOP_ITEMS,
    NUM_ACTIONS,
    ActionRange,
    SubPhase,
)

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
        "_prev_signature",
        "_round_score",
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
        elif sp == SubPhase.SELECT_CARDS:
            _mask_cards(m, state, AR, self._selected_cards, self._pending_action)
        elif sp == SubPhase.SHOP:
            _mask_shop(m, state, AR)
        elif sp == SubPhase.BOOSTER_PACK:
            _mask_booster(m, state, AR)
        elif sp == SubPhase.CONSUMABLE_TARGET:
            _mask_consumable(
                m, state, AR,
                self._pending_consumable_slot,
                self._pending_hand_targets,
                self._pending_joker_targets,
            )

        return m

    # ── step (action-id based, no decode_action overhead) ──

    def step(self, action_id: int) -> None:
        self._blind_just_beaten = False
        self._step_count += 1

        try:
            self._execute(action_id)
        except Exception:
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

        elif aid == AR.PLAY_HAND:
            self._pending_action = "play"
            self._selected_cards = {i for i, c in enumerate(state.hand_cards) if c.forced_selection}
            self._sub_phase = SubPhase.SELECT_CARDS

        elif aid == AR.DISCARD:
            self._pending_action = "discard"
            self._selected_cards = {i for i, c in enumerate(state.hand_cards) if c.forced_selection}
            self._sub_phase = SubPhase.SELECT_CARDS

        elif aid == AR.USE_CONSUMABLE:
            self._pending_consumable_slot = None
            self._pending_hand_targets = ()
            self._pending_joker_targets = ()
            self._sub_phase = SubPhase.CONSUMABLE_TARGET

        elif AR.TOGGLE_CARD_START <= aid <= AR.TOGGLE_CARD_END:
            idx = aid - AR.TOGGLE_CARD_START
            if idx in self._selected_cards:
                self._selected_cards.discard(idx)
            else:
                self._selected_cards.add(idx)

        elif aid == AR.SELECT_CONFIRM:
            indices = sorted(self._selected_cards)
            if self._pending_action == "play":
                result = ctrl.play_selected(indices)
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
            else:
                ctrl.discard_selected(indices)
                self._sub_phase = SubPhase.CHOOSE_ACTION
            self._selected_cards = set()
            self._pending_action = None

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


# ── mask helpers (module-level for speed) ──

def _mask_blind(m: np.ndarray, state: RunState, AR: type[ActionRange]) -> None:
    m[AR.BLIND_PLAY] = 1
    blind_on_deck = state.blind_on_deck or "Small"
    if blind_on_deck in ("Small", "Big"):
        m[AR.BLIND_SKIP] = 1
    if blind_on_deck == "Boss" and state.dollars >= 10 and not state.round_resets.boss_rerolled:
        m[AR.BLIND_REROLL] = 1


def _mask_action(m: np.ndarray, state: RunState, AR: type[ActionRange]) -> None:
    hand_size = len(state.hand_cards)
    if state.current_round.hands_left > 0 and hand_size > 0:
        m[AR.PLAY_HAND] = 1
    if state.current_round.discards_left > 0 and hand_size > 0:
        m[AR.DISCARD] = 1
    for cons in state.consumables:
        if can_use_consumable(state, cons):
            m[AR.USE_CONSUMABLE] = 1
            break


def _mask_cards(
    m: np.ndarray,
    state: RunState,
    AR: type[ActionRange],
    selected: set[int],
    pending: str | None,
) -> None:
    hand_size = len(state.hand_cards)
    num_sel = len(selected)
    is_play = pending == "play"
    max_sel = 5 if is_play else hand_size

    for i in range(min(hand_size, MAX_HAND_SIZE)):
        if i in selected:
            if not state.hand_cards[i].forced_selection:
                m[AR.TOGGLE_CARD_START + i] = 1
        else:
            if num_sel < max_sel:
                m[AR.TOGGLE_CARD_START + i] = 1

    if is_play:
        if 1 <= num_sel <= 5:
            m[AR.SELECT_CONFIRM] = 1
    else:
        if num_sel >= 1:
            m[AR.SELECT_CONFIRM] = 1


def _mask_shop(m: np.ndarray, state: RunState, AR: type[ActionRange]) -> None:
    all_items = list(state.shop.cards) + list(state.shop.vouchers) + list(state.shop.boosters)
    for i, item in enumerate(all_items[:MAX_SHOP_ITEMS]):
        if item.cost > state.dollars:
            continue
        if item.card_type == "Joker":
            if len(state.jokers) < joker_limit(state):
                m[AR.SHOP_BUY_START + i] = 1
        elif item.card_type in ("Tarot", "Planet", "Spectral"):
            if len(state.consumables) < consumable_limit(state):
                m[AR.SHOP_BUY_START + i] = 1
        else:
            m[AR.SHOP_BUY_START + i] = 1

    if state.current_round.reroll_cost <= state.dollars:
        m[AR.SHOP_REROLL] = 1

    for i, joker in enumerate(state.jokers[:MAX_JOKER_SLOTS]):
        if not joker.eternal:
            m[AR.SHOP_SELL_JOKER_START + i] = 1

    for i in range(min(len(state.consumables), MAX_CONSUMABLE_SLOTS)):
        m[AR.SHOP_SELL_CONSUMABLE_START + i] = 1

    m[AR.SHOP_LEAVE] = 1


def _mask_booster(m: np.ndarray, state: RunState, AR: type[ActionRange]) -> None:
    pack = state.pack
    if pack and pack.choices_remaining > 0:
        for i in range(min(len(pack.cards), MAX_PACK_CARDS)):
            m[AR.PACK_CLAIM_START + i] = 1
    m[AR.PACK_SKIP] = 1


def _mask_consumable(
    m: np.ndarray,
    state: RunState,
    AR: type[ActionRange],
    pending_slot: int | None,
    hand_targets: tuple[int, ...],
    joker_targets: tuple[int, ...],
) -> None:
    if pending_slot is None:
        for i, cons in enumerate(state.consumables[:MAX_CONSUMABLE_SLOTS]):
            if can_use_consumable(state, cons):
                m[AR.CONSUMABLE_SLOT_START + i] = 1
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
                        m[AR.CONSUMABLE_HAND_TARGET_START + i] = 1

            if required_min <= current_count <= required_max and can_use_consumable(
                state, cons, hand_targets=hand_targets, joker_targets=joker_targets,
            ):
                m[AR.CONSUMABLE_CONFIRM] = 1
        else:
            if can_use_consumable(state, cons, hand_targets=hand_targets, joker_targets=joker_targets):
                m[AR.CONSUMABLE_CONFIRM] = 1

        name = center.get("name", "")
        if name in ("The Wheel of Fortune", "Ectoplasm", "Hex", "Ankh"):
            for i in range(min(len(state.jokers), MAX_JOKER_SLOTS)):
                if i not in joker_targets:
                    m[AR.CONSUMABLE_JOKER_TARGET_START + i] = 1

    m[AR.CONSUMABLE_CANCEL] = 1


# ── progress signature (mirrors BalatroEnv._progress_signature) ──

def _progress_signature(
    state: RunState,
    phase: GamePhase,
    sub_phase: SubPhase,
    round_score: int,
) -> tuple:
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


def _blind_target(state: RunState) -> int:
    from pylatro import get_blind_amount
    blind = state.round_resets.blind
    if blind is None:
        return 0
    base = get_blind_amount(state.round_resets.ante, min(state.stake, 3))
    mult = blind.get("mult", 1)
    return int(base * mult)
