"""Gymnasium environment wrapping the Balatro game engine."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium
import numpy as np
from gymnasium import spaces

from pylatro import GameData, get_blind_amount, load_game_data
from pylatro_cli.controller import GameController, GamePhase

from .action import ActionType, decode_action
from .constants import MAX_HAND_SIZE, MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM, SubPhase
from .masks import compute_action_mask
from .reward import RewardFn, default_reward, default_reward_components
from .subset_actions import subset_indices
from .tokenizer import RawObservation, Tokenizer
from .vocab import Vocab, build_vocab


class BalatroEnv(gymnasium.Env):
    """Gymnasium environment for Balatro.

    Each step() is one atomic decision. The environment tracks sub-phases
    beyond GamePhase to handle multi-step actions like card selection.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        seed: int | None = None,
        stake: int = 1,
        deck_key: str = "b_red",
        objective: str = "win",
        max_steps: int = 2000,
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
        self._initial_seed_pending = seed is not None

        self._controller: GameController | None = None
        self._sub_phase = SubPhase.BLIND_SELECT
        self._step_count = 0
        self._steps_since_progress = 0

        # Multi-step state
        self._selected_cards: set[int] = set()
        self._pending_action: str | None = None  # "play" or "discard"
        self._pending_consumable_slot: int | None = None
        self._pending_consumable_hand_targets: tuple[int, ...] = ()
        self._pending_consumable_joker_targets: tuple[int, ...] = ()

        # Previous state info for reward computation
        self._prev_info: dict[str, Any] = {}
        self._round_score: int = 0
        self._blind_just_beaten: bool = False

        # Gymnasium spaces
        self.observation_space = spaces.Dict({
            "tokens": spaces.Box(0, 32767, (MAX_SEQ_LEN, TOKEN_DIM), dtype=np.int16),
            "token_types": spaces.Box(0, 10, (MAX_SEQ_LEN,), dtype=np.int8),
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
        if seed is not None:
            super().reset(seed=seed)
            self._seed = seed
            self._initial_seed_pending = False
            effective_seed = seed
        elif self._seed is not None and self._initial_seed_pending:
            # Use the constructor seed exactly once, then advance via the env RNG
            # so vector-env autoresets do not replay the same episode forever.
            super().reset(seed=self._seed)
            self._initial_seed_pending = False
            effective_seed = self._seed
        else:
            super().reset(seed=None)
            effective_seed = int(self.np_random.integers(0, 2**31))

        self._controller = GameController(data=self._data)
        self._controller.new_run(str(effective_seed), stake=self._stake, deck_key=self._deck_key)

        self._sub_phase = SubPhase.BLIND_SELECT
        self._step_count = 0
        self._steps_since_progress = 0
        self._selected_cards = set()
        self._pending_action = None
        self._pending_consumable_slot = None
        self._pending_consumable_hand_targets = ()
        self._pending_consumable_joker_targets = ()
        self._round_score = 0
        self._blind_just_beaten = False
        self._prev_info = self._capture_state_info()

        obs = self._build_obs()
        return self._obs_to_dict(obs), {"sub_phase": self._sub_phase}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        assert self._controller is not None and self._controller.state is not None

        self._prev_info = self._capture_state_info()
        self._step_count += 1
        self._blind_just_beaten = False
        pre_sub_phase = self._sub_phase
        pre_pending_action = self._pending_action or ""
        pre_selected_count = len(self._selected_cards)

        decoded = decode_action(action)
        terminated = False
        truncated = False

        try:
            self._execute_action(decoded)
        except Exception as e:
            # Invalid action — log and give small penalty, but don't terminate.
            # Masking should prevent this; if it happens it's a bug to investigate.
            import logging
            logging.getLogger(__name__).warning(f"Action {action} raised {type(e).__name__}: {e}")
            reward = -1.0
            obs = self._build_obs()
            info = {"sub_phase": self._sub_phase, "error": str(e)}
            return self._obs_to_dict(obs), reward, False, False, info

        # Check terminal conditions driven by the underlying game state.
        terminated = self._controller.phase in (GamePhase.GAME_OVER, GamePhase.GAME_WON)
        won = self._controller.phase == GamePhase.GAME_WON
        state = self._controller.state
        curr_info = self._capture_state_info()
        curr_info["blind_just_beaten"] = self._blind_just_beaten
        curr_info["hands_left"] = state.current_round.hands_left
        progress_made = self._progress_signature(curr_info) != self._progress_signature(self._prev_info)
        curr_info["progress_made"] = progress_made
        if progress_made:
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1
        curr_info["steps_since_progress"] = self._steps_since_progress

        if not terminated and self._steps_since_progress >= self._max_steps:
            truncated = True
            curr_info["stalled"] = True
        else:
            curr_info["stalled"] = False

        reward_components: dict[str, float] = {}
        if self._reward_fn is default_reward:
            reward_components = default_reward_components(state, self._prev_info, curr_info, terminated, won)
            reward = reward_components["total"]
        else:
            reward = self._reward_fn(state, self._prev_info, curr_info, terminated, won)

        obs = self._build_obs()
        info = {
            "sub_phase": self._sub_phase,
            "pre_sub_phase": pre_sub_phase,
            "pre_pending_action": pre_pending_action,
            "pre_selected_count": pre_selected_count,
            "ante": state.round_resets.ante,
            "dollars": state.dollars,
            "round_score": self._round_score,
            "won": won,
            "progress_made": progress_made,
            "steps_since_progress": self._steps_since_progress,
            "stalled": curr_info["stalled"],
        }
        for component_name, component_value in reward_components.items():
            info[f"reward_{component_name}"] = component_value
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

        elif at == ActionType.PLAY_SUBSET:
            indices = subset_indices(decoded.index)
            if any(idx >= len(state.hand_cards) for idx in indices):
                raise IndexError(f"Play subset {decoded.index} is invalid for hand size {len(state.hand_cards)}")
            result = ctrl.play_selected(list(indices))
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

        elif at == ActionType.DISCARD_SUBSET:
            indices = subset_indices(decoded.index)
            if any(idx >= len(state.hand_cards) for idx in indices):
                raise IndexError(f"Discard subset {decoded.index} is invalid for hand size {len(state.hand_cards)}")
            ctrl.discard_selected(list(indices))
            self._sub_phase = SubPhase.CHOOSE_ACTION

        elif at == ActionType.USE_CONSUMABLE:
            self._pending_consumable_slot = None
            self._pending_consumable_hand_targets = ()
            self._pending_consumable_joker_targets = ()
            self._sub_phase = SubPhase.CONSUMABLE_TARGET

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
                return

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
        pack_choices_remaining = state.pack.choices_remaining if state.pack is not None else 0
        shop_item_count = len(state.shop.cards) + len(state.shop.vouchers) + len(state.shop.boosters)
        return {
            "ante": state.round_resets.ante,
            "round_score": self._round_score,
            "blind_beaten": self._controller.blind_beaten() if self._controller.phase == GamePhase.HAND_PLAY else False,
            "blind_on_deck": state.blind_on_deck or "",
            "blind_target": self._blind_target(state),
            "hands_left": state.current_round.hands_left,
            "discards_left": state.current_round.discards_left,
            "dollars": state.dollars,
            "in_shop": self._controller.phase == GamePhase.SHOP,
            "phase": self._controller.phase,
            "sub_phase": self._sub_phase,
            "joker_count": len(state.jokers),
            "consumable_count": len(state.consumables),
            "shop_item_count": shop_item_count,
            "pack_choices_remaining": pack_choices_remaining,
        }

    def _blind_target(self, state) -> int:
        blind = state.round_resets.blind
        if blind is None:
            return 0
        base = get_blind_amount(state.round_resets.ante, min(state.stake, 3))
        mult = blind.get("mult", 1)
        return int(base * mult)

    def _progress_signature(self, info: dict[str, Any]) -> tuple[Any, ...]:
        """Return a compact snapshot used to detect meaningful game progress.

        Selection toggles are intentionally excluded so the agent cannot avoid the
        inactivity limit by flipping highlighted cards back and forth.
        """
        return (
            info.get("ante", 0),
            info.get("blind_on_deck", ""),
            info.get("blind_target", 0),
            info.get("round_score", 0),
            info.get("hands_left", 0),
            info.get("discards_left", 0),
            info.get("dollars", 0),
            info.get("phase", ""),
            info.get("sub_phase", ""),
            info.get("joker_count", 0),
            info.get("consumable_count", 0),
            info.get("shop_item_count", 0),
            info.get("pack_choices_remaining", 0),
        )
