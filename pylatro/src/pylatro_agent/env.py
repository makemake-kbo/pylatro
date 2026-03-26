"""Gymnasium environment wrapping the Balatro game engine."""

from __future__ import annotations

from math import floor
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from pylatro import GameData, load_game_data
from pylatro_cli.controller import GameController, GamePhase

from .action import ActionType, decode_action
from .constants import MAX_HAND_SIZE, MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM, SubPhase
from .masks import compute_action_mask
from .reward import RewardFn, default_reward
from .tokenizer import RawObservation, Tokenizer
from .vocab import Vocab, build_vocab


class BalatroEnv(gymnasium.Env):
    """Gymnasium environment for Balatro.

    Each step() is one atomic decision. The environment tracks sub-phases
    beyond GamePhase to handle multi-step actions like card selection.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int | None = None,
        stake: int = 1,
        deck_key: str = "b_red",
        objective: str = "win",
        max_steps: int = 10000,
        reward_fn: RewardFn | None = None,
        data: GameData | None = None,
        vocab: Vocab | None = None,
    ):
        super().__init__()
        self._data = data or load_game_data()
        self._vocab = vocab or build_vocab(self._data)
        self._tokenizer = Tokenizer(vocab=self._vocab)
        self._stake = stake
        self._deck_key = deck_key
        self._objective = objective
        self._max_steps = max_steps
        self._reward_fn = reward_fn or default_reward
        self._seed = seed

        self._controller: GameController | None = None
        self._sub_phase = SubPhase.BLIND_SELECT
        self._step_count = 0

        # Multi-step state
        self._selected_cards: set[int] = set()
        self._pending_action: str | None = None  # "play" or "discard"
        self._pending_consumable_slot: int | None = None
        self._pending_consumable_hand_targets: tuple[int, ...] = ()
        self._pending_consumable_joker_targets: tuple[int, ...] = ()

        # Previous state info for reward computation
        self._prev_info: dict[str, Any] = {}
        self._round_score: int = 0

        # Gymnasium spaces
        self.observation_space = spaces.Dict({
            "tokens": spaces.Box(0, 32767, (MAX_SEQ_LEN, TOKEN_DIM), dtype=np.int16),
            "token_types": spaces.Box(0, 9, (MAX_SEQ_LEN,), dtype=np.int8),
            "scalars": spaces.Box(-np.inf, np.inf, (SCALAR_DIM,), dtype=np.float32),
            "attention_mask": spaces.Box(0, 1, (MAX_SEQ_LEN,), dtype=np.int8),
            "action_mask": spaces.Box(0, 1, (NUM_ACTIONS,), dtype=np.int8),
            "selected_cards": spaces.Box(0, 1, (MAX_HAND_SIZE,), dtype=np.int8),
        })
        self.action_space = spaces.Discrete(NUM_ACTIONS)

    @property
    def state(self):
        return self._controller.state if self._controller else None

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self._seed
        if effective_seed is None:
            effective_seed = self.np_random.integers(0, 2**31)

        self._controller = GameController(data=self._data)
        self._controller.new_run(str(effective_seed), stake=self._stake, deck_key=self._deck_key)

        self._sub_phase = SubPhase.BLIND_SELECT
        self._step_count = 0
        self._selected_cards = set()
        self._pending_action = None
        self._pending_consumable_slot = None
        self._pending_consumable_hand_targets = ()
        self._pending_consumable_joker_targets = ()
        self._round_score = 0
        self._prev_info = self._capture_state_info()

        obs = self._build_obs()
        return self._obs_to_dict(obs), {"sub_phase": self._sub_phase}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        assert self._controller is not None and self._controller.state is not None

        self._prev_info = self._capture_state_info()
        self._step_count += 1

        decoded = decode_action(action)
        terminated = False
        truncated = False

        try:
            self._execute_action(decoded)
        except Exception:
            # Invalid action — penalize and terminate
            terminated = True
            reward = -10.0
            obs = self._build_obs()
            return self._obs_to_dict(obs), reward, terminated, truncated, {"error": "invalid_action"}

        # Check terminal conditions
        if self._controller.phase == GamePhase.GAME_OVER:
            terminated = True
        elif self._controller.phase == GamePhase.GAME_WON:
            terminated = True
        elif self._step_count >= self._max_steps:
            truncated = True

        won = self._controller.phase == GamePhase.GAME_WON
        state = self._controller.state
        reward = self._reward_fn(state, self._prev_info, terminated, won)

        obs = self._build_obs()
        info = {
            "sub_phase": self._sub_phase,
            "ante": state.round_resets.ante,
            "dollars": state.dollars,
            "round_score": self._round_score,
            "won": won,
        }
        return self._obs_to_dict(obs), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return current valid action mask."""
        if self._controller is None or self._controller.state is None:
            return np.zeros(NUM_ACTIONS, dtype=np.int8)
        return compute_action_mask(
            self._controller.state,
            self._sub_phase,
            selected_cards=self._selected_cards,
            pending_action=self._pending_action,
            pending_consumable_slot=self._pending_consumable_slot,
            pending_consumable_targets_hand=self._pending_consumable_hand_targets,
            pending_consumable_targets_joker=self._pending_consumable_joker_targets,
        )

    def _execute_action(self, decoded) -> None:
        """Execute a decoded action, updating sub-phase and game state."""
        ctrl = self._controller
        state = ctrl.state
        at = decoded.action_type

        if at == ActionType.BLIND_PLAY:
            blind_type = state.blind_on_deck or "Small"
            ctrl.select_blind(blind_type)
            self._sub_phase = SubPhase.CHOOSE_ACTION
            self._round_score = 0

        elif at == ActionType.BLIND_SKIP:
            ctrl.skip_blind()
            # Stay in BLIND_SELECT for next blind
            self._sub_phase = SubPhase.BLIND_SELECT

        elif at == ActionType.BLIND_REROLL:
            ctrl.reroll_boss()
            self._sub_phase = SubPhase.BLIND_SELECT

        elif at == ActionType.PLAY_HAND:
            self._pending_action = "play"
            self._selected_cards = set()
            # Auto-select forced cards
            for i, card in enumerate(state.hand_cards):
                if card.forced_selection:
                    self._selected_cards.add(i)
            self._sub_phase = SubPhase.SELECT_CARDS

        elif at == ActionType.DISCARD:
            self._pending_action = "discard"
            self._selected_cards = set()
            for i, card in enumerate(state.hand_cards):
                if card.forced_selection:
                    self._selected_cards.add(i)
            self._sub_phase = SubPhase.SELECT_CARDS

        elif at == ActionType.USE_CONSUMABLE:
            self._pending_consumable_slot = None
            self._pending_consumable_hand_targets = ()
            self._pending_consumable_joker_targets = ()
            self._sub_phase = SubPhase.CONSUMABLE_TARGET

        elif at == ActionType.TOGGLE_CARD:
            idx = decoded.index
            if idx in self._selected_cards:
                self._selected_cards.discard(idx)
            else:
                self._selected_cards.add(idx)
            # Stay in SELECT_CARDS

        elif at == ActionType.SELECT_CONFIRM:
            indices = sorted(self._selected_cards)
            if self._pending_action == "play":
                result = ctrl.play_selected(indices)
                self._round_score += result.score.total
                if ctrl.blind_beaten():
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

        elif at == ActionType.CONSUMABLE_SLOT:
            self._pending_consumable_slot = decoded.index
            self._pending_consumable_hand_targets = ()
            self._pending_consumable_joker_targets = ()

        elif at == ActionType.CONSUMABLE_HAND_TARGET:
            targets = list(self._pending_consumable_hand_targets)
            idx = decoded.index
            if idx in targets:
                targets.remove(idx)
            else:
                targets.append(idx)
            self._pending_consumable_hand_targets = tuple(targets)

        elif at == ActionType.CONSUMABLE_JOKER_TARGET:
            targets = list(self._pending_consumable_joker_targets)
            idx = decoded.index
            if idx in targets:
                targets.remove(idx)
            else:
                targets.append(idx)
            self._pending_consumable_joker_targets = tuple(targets)

        elif at == ActionType.CONSUMABLE_CONFIRM:
            slot = self._pending_consumable_slot
            if slot is not None:
                ctrl.use_consumable_on(
                    slot,
                    hand_targets=self._pending_consumable_hand_targets,
                    joker_targets=self._pending_consumable_joker_targets,
                )
            self._pending_consumable_slot = None
            self._pending_consumable_hand_targets = ()
            self._pending_consumable_joker_targets = ()
            # Return to appropriate phase
            if ctrl.phase == GamePhase.HAND_PLAY:
                self._sub_phase = SubPhase.CHOOSE_ACTION
            elif ctrl.phase == GamePhase.SHOP:
                self._sub_phase = SubPhase.SHOP

        elif at == ActionType.CONSUMABLE_CANCEL:
            self._pending_consumable_slot = None
            self._pending_consumable_hand_targets = ()
            self._pending_consumable_joker_targets = ()
            if ctrl.phase == GamePhase.HAND_PLAY:
                self._sub_phase = SubPhase.CHOOSE_ACTION
            elif ctrl.phase == GamePhase.SHOP:
                self._sub_phase = SubPhase.SHOP

        elif at == ActionType.SHOP_BUY:
            idx = decoded.index
            all_items = list(state.shop.cards) + list(state.shop.vouchers) + list(state.shop.boosters)
            item = all_items[idx]
            if item.shop_voucher:
                ctrl.buy_voucher(item.center_key)
            elif item.booster_pos is not None:
                # Buying a booster opens it
                booster_idx = item.booster_pos
                ctrl.buy_card(idx)  # Pay for it first
                pack = ctrl.open_pack(booster_idx)
                self._sub_phase = SubPhase.BOOSTER_PACK
                return
            else:
                ctrl.buy_card(idx)

        elif at == ActionType.SHOP_REROLL:
            ctrl.reroll()

        elif at == ActionType.SHOP_SELL_JOKER:
            ctrl.sell_joker(decoded.index)

        elif at == ActionType.SHOP_SELL_CONSUMABLE:
            ctrl.sell_consumable(decoded.index)

        elif at == ActionType.SHOP_LEAVE:
            ctrl.leave_shop()
            self._sub_phase = SubPhase.BLIND_SELECT

        elif at == ActionType.PACK_CLAIM:
            ctrl.claim_from_pack(decoded.index)
            if state.pack and state.pack.choices_remaining <= 0:
                ctrl.close_current_pack(skipped=False)
                self._sub_phase = SubPhase.SHOP

        elif at == ActionType.PACK_SKIP:
            ctrl.close_current_pack(skipped=True)
            self._sub_phase = SubPhase.SHOP

    def _build_obs(self) -> RawObservation:
        state = self._controller.state
        mask = self.action_masks()
        return self._tokenizer.tokenize(
            state,
            self._sub_phase,
            selected_cards=self._selected_cards,
            action_mask=mask,
        )

    def _obs_to_dict(self, obs: RawObservation) -> dict:
        return {
            "tokens": obs.tokens,
            "token_types": obs.token_types,
            "scalars": obs.scalars,
            "attention_mask": obs.attention_mask,
            "action_mask": obs.action_mask,
            "selected_cards": obs.selected_cards,
        }

    def _capture_state_info(self) -> dict:
        if self._controller is None or self._controller.state is None:
            return {}
        state = self._controller.state
        return {
            "ante": state.round_resets.ante,
            "round_score": self._round_score,
            "blind_beaten": self._controller.blind_beaten() if self._controller.phase == GamePhase.HAND_PLAY else False,
            "hands_left": state.current_round.hands_left,
            "dollars": state.dollars,
        }
