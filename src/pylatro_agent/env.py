"""Gymnasium environment wrapping the Balatro game engine."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium
import numpy as np
from gymnasium import spaces

from pylatro import GameData, load_game_data
from pylatro.instances import move_joker
from pylatro_cli.controller import GameController, GamePhase

from .action import ActionType, decode_action
from .constants import (
    HISTORY_EVENT_DIM,
    HISTORY_FEATURE_DIM,
    HISTORY_MAX_CARDS,
    HISTORY_MAX_JOKERS,
    HISTORY_MAX_PLAYS,
    HISTORY_OMITTED_DIM,
    HISTORY_ROUNDS,
    MAX_SEQ_LEN,
    NUM_ACTIONS,
    SCALAR_DIM,
    TOKEN_DIM,
    SubPhase,
)
from .diagnostics import (
    action_diagnostics as _shared_action_diagnostics,
)
from .diagnostics import (
    build_step_diagnostics,
    finish_exact_play_counterfactual,
    prepare_exact_play_counterfactual,
    step_event_diagnostics,
)
from .heuristic import HeuristicAgent
from .history import PlayHistoryTracker, blind_history_key
from .masks import compute_action_mask
from .reward import (
    DEFAULT_REWARD_CONFIG,
    RewardConfig,
    default_reward_components,
)
from .shop_eval import capture_build_features, evaluate_build
from .subset_actions import consumable_subset_indices, subset_indices
from .tokenizer import RawObservation, Tokenizer
from .vocab import Vocab, build_vocab


class BalatroEnv(gymnasium.Env):
    """Gymnasium environment for Balatro.

    Each ``step()`` executes one complete atomic decision. Sub-phases identify
    which family of atomic actions is currently legal.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        seed: int | None = None,
        stake: int = 1,
        deck_key: str = "b_red",
        max_steps: int = 2000,
        reward_config: RewardConfig | None = None,
        data: GameData | None = None,
        vocab: Vocab | None = None,
        win_ante: int | None = None,
        enable_teacher: bool = True,
        counterfactual_diagnostic_interval: int = 0,
    ):
        super().__init__()
        self._data = data or load_game_data()
        self._vocab = vocab or build_vocab(self._data)
        self._tokenizer = Tokenizer(vocab=self._vocab)
        self._stake = stake
        self._deck_key = deck_key
        self._max_steps = max_steps
        # Override the run's victory threshold for curriculum training. None
        # uses the engine default (win_ante=8). Lower values let PPO see
        # frequent wins early so it can bootstrap a value signal, the
        # heuristic teacher only wins ~1% at ante 8 but ~39% at ante 4.
        self._win_ante_override = win_ante
        self._reward_config = reward_config or DEFAULT_REWARD_CONFIG
        self._seed = seed
        self._initial_seed_pending = seed is not None
        if counterfactual_diagnostic_interval < 0:
            raise ValueError("counterfactual_diagnostic_interval must be non-negative")
        self._counterfactual_diagnostic_interval = int(counterfactual_diagnostic_interval)
        self._play_diagnostic_count = 0

        self._controller: GameController | None = None
        self._sub_phase = SubPhase.BLIND_SELECT
        self._steps_since_progress = 0
        self._history = PlayHistoryTracker()

        # Previous state info for reward computation
        self._prev_info: dict[str, Any] = {}
        # Heuristic teacher for distillation. One instance per env (process);
        # the cache is per-instance and keyed on hand+joker signature, so
        # parallel envs are isolated naturally. When nothing consumes teacher
        # labels (no distillation, DAgger, or teacher-forced rollouts) the
        # caller disables it: the teacher runs twice per step and is pure
        # overhead, and every teacher_action field becomes the -1 sentinel.
        self._teacher = HeuristicAgent() if enable_teacher else None

        # Gymnasium spaces
        self.observation_space = spaces.Dict(
            {
                "tokens": spaces.Box(0, 32767, (MAX_SEQ_LEN, TOKEN_DIM), dtype=np.int16),
                "token_types": spaces.Box(0, 11, (MAX_SEQ_LEN,), dtype=np.int8),
                "scalars": spaces.Box(-np.inf, np.inf, (SCALAR_DIM,), dtype=np.float32),
                "attention_mask": spaces.Box(0, 1, (MAX_SEQ_LEN,), dtype=np.int8),
                "action_mask": spaces.Box(0, 1, (NUM_ACTIONS,), dtype=np.int8),
                "history_events": spaces.Box(
                    0, 32767, (HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_EVENT_DIM), dtype=np.int16
                ),
                "history_event_features": spaces.Box(
                    -np.inf,
                    np.inf,
                    (HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_FEATURE_DIM),
                    dtype=np.float32,
                ),
                "history_cards": spaces.Box(
                    0,
                    32767,
                    (HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_CARDS, TOKEN_DIM),
                    dtype=np.int16,
                ),
                "history_card_mask": spaces.Box(
                    0, 1, (HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_CARDS), dtype=np.int8
                ),
                "history_jokers": spaces.Box(
                    0, 32767, (HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_JOKERS), dtype=np.int16
                ),
                "history_joker_mask": spaces.Box(
                    0, 1, (HISTORY_ROUNDS, HISTORY_MAX_PLAYS, HISTORY_MAX_JOKERS), dtype=np.int8
                ),
                "history_event_mask": spaces.Box(
                    0, 1, (HISTORY_ROUNDS, HISTORY_MAX_PLAYS), dtype=np.int8
                ),
                "history_round_mask": spaces.Box(0, 1, (HISTORY_ROUNDS,), dtype=np.int8),
                "history_omitted": spaces.Box(
                    -np.inf, np.inf, (HISTORY_ROUNDS, HISTORY_OMITTED_DIM), dtype=np.float32
                ),
            }
        )
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
        if self._win_ante_override is not None and self._controller.state is not None:
            self._controller.state.win_ante = int(self._win_ante_override)

        self._sub_phase = SubPhase.BLIND_SELECT
        self._steps_since_progress = 0
        self._play_diagnostic_count = 0
        self._history.reset()
        self._prev_info = self._capture_state_info()

        obs = self._build_obs()
        return self._obs_to_dict(obs), {
            "sub_phase": self._sub_phase,
            "teacher_action": self._current_teacher_action(),
        }

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        assert self._controller is not None and self._controller.state is not None

        self._prev_info = self._capture_state_info()
        pre_sub_phase = self._sub_phase

        # Query the heuristic teacher with the pre-action mask + state.
        # Used by PPO for distillation; -1 sentinel means "no valid teacher".
        if self._teacher is None:
            teacher_action = -1
        else:
            try:
                pre_mask = self.action_masks()
                teacher_action = int(
                    self._teacher.select_action(
                        self._controller.state,
                        self._sub_phase,
                        pre_mask,
                        round_score=self._controller.round_score,
                    )
                )
                if teacher_action < 0 or teacher_action >= NUM_ACTIONS or not pre_mask[teacher_action]:
                    teacher_action = -1
            except Exception:
                teacher_action = -1

        decoded = decode_action(action)
        action_diagnostics = self._action_diagnostics(decoded)
        counterfactual_probe = None
        if decoded.action_type == ActionType.PLAY_SUBSET:
            self._play_diagnostic_count += 1
            interval = self._counterfactual_diagnostic_interval
            if interval > 0 and self._play_diagnostic_count % interval == 0:
                indices = tuple(subset_indices(decoded.index))
                counterfactual_probe, counterfactual_diagnostics = prepare_exact_play_counterfactual(
                    self._controller.state,
                    indices,
                    self._prev_info,
                    sample_index=self._play_diagnostic_count // interval - 1,
                )
                action_diagnostics.update(counterfactual_diagnostics)
        terminated = False
        truncated = False

        try:
            action_result = self._execute_action(decoded)
        except Exception as e:
            # Invalid action, log and give small penalty, but don't terminate.
            # Masking should prevent this; if it happens it's a bug to investigate.
            import logging

            logging.getLogger(__name__).warning(f"Action {action} raised {type(e).__name__}: {e}")
            reward = -1.0
            obs = self._build_obs()
            info = {
                "sub_phase": self._sub_phase,
                "error": str(e),
                "teacher_action": teacher_action,
            }
            if action_diagnostics.get("counterfactual_call"):
                action_diagnostics["counterfactual_failure"] = True
                action_diagnostics["counterfactual_failure_reason"] = "action_error"
            info.update(action_diagnostics)
            return self._obs_to_dict(obs), reward, False, False, info

        # Check terminal conditions driven by the underlying game state.
        terminated = self._controller.phase in (GamePhase.GAME_OVER, GamePhase.GAME_WON)
        won = self._controller.phase == GamePhase.GAME_WON
        state = self._controller.state
        curr_info = self._capture_state_info()
        curr_info["hands_left"] = state.current_round.hands_left
        curr_info["action_type"] = decoded.action_type
        curr_info["action_index"] = decoded.index
        curr_info["action_detail"] = decoded.detail
        curr_info["teacher_action"] = teacher_action
        curr_info["teacher_action_match"] = teacher_action >= 0 and int(action) == teacher_action
        if decoded.action_type == ActionType.MOVE_JOKER:
            pre_score = evaluate_build(self._prev_info).estimated_score
            post_score = evaluate_build(curr_info).estimated_score
            ratio = max(post_score, 1.0) / max(pre_score, 1.0)
            action_diagnostics.update(
                {
                    "joker_move_source": decoded.index,
                    "joker_move_destination": decoded.detail,
                    "joker_move_pre_score": pre_score,
                    "joker_move_post_score": post_score,
                    "joker_move_score_ratio": ratio,
                    "joker_move_reward": 0.1 * float(np.clip(np.log(ratio), -1.0, 1.0)),
                }
            )

        event_diagnostics = step_event_diagnostics(self._prev_info, curr_info, decoded)
        action_diagnostics.update(event_diagnostics)
        if counterfactual_probe is not None:
            actual_score = float(action_result.score.total)
            action_diagnostics.update(
                finish_exact_play_counterfactual(
                    counterfactual_probe,
                    actual_score=actual_score,
                )
            )

        # Detailed leave-one-out build diagnostics are expensive. When score-build
        # potential is active they also provide the cache consumed by the reward;
        # otherwise compute them only for actual build/shop events.
        score_build_potential_enabled = self._reward_config.enable_score_build_potential
        build_event_actions = {
            ActionType.SHOP_BUY,
            ActionType.SHOP_REROLL,
            ActionType.SHOP_SELL_JOKER,
            ActionType.SHOP_LEAVE,
            ActionType.PACK_CLAIM,
        }
        if (
            score_build_potential_enabled
            or decoded.action_type in build_event_actions
            or action_diagnostics.get("joker_roster_changed")
            or action_diagnostics.get("shop_joker_offer_observed")
        ):
            action_diagnostics.update(
                build_step_diagnostics(
                    self._prev_info,
                    curr_info,
                    self._reward_config,
                    win_ante=int(state.win_ante),
                )
            )

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

        # Surface action diagnostics used by optional reward components.
        curr_info.update(action_diagnostics)

        reward_components = default_reward_components(
            state, self._prev_info, curr_info, terminated, won, self._reward_config
        )
        reward = reward_components["total"]

        obs = self._build_obs()
        info = {
            "sub_phase": self._sub_phase,
            "pre_sub_phase": pre_sub_phase,
            "ante": state.round_resets.ante,
            "blind_on_deck": state.blind_on_deck or "",
            "boss_key": state.round_resets.blind_choices.get("Boss", "") or "",
            "dollars": state.dollars,
            "round_score": self._controller.round_score,
            "blind_target": curr_info.get("blind_target", 0.0),
            "won": won,
            "progress_made": progress_made,
            "steps_since_progress": self._steps_since_progress,
            "stalled": curr_info["stalled"],
        }
        for component_name, component_value in reward_components.items():
            info[f"reward_{component_name}"] = component_value
        info.update(action_diagnostics)
        info["teacher_action"] = teacher_action
        info["teacher_action_match"] = curr_info["teacher_action_match"]
        info["next_teacher_action"] = -1 if terminated or truncated else self._current_teacher_action()
        return self._obs_to_dict(obs), reward, terminated, truncated, info

    def _current_teacher_action(self) -> int:
        """Return the heuristic action for the current state, or -1 if unavailable."""
        if self._teacher is None:
            return -1
        if self._controller is None or self._controller.state is None:
            return -1
        try:
            mask = self.action_masks()
            teacher_action = int(
                self._teacher.select_action(
                    self._controller.state,
                    self._sub_phase,
                    mask,
                    round_score=self._controller.round_score,
                )
            )
            if teacher_action < 0 or teacher_action >= NUM_ACTIONS or not mask[teacher_action]:
                return -1
            return teacher_action
        except Exception:
            return -1

    def action_masks(self) -> np.ndarray:
        """Return current valid action mask."""
        if self._controller is None or self._controller.state is None:
            return np.zeros(NUM_ACTIONS, dtype=np.int8)
        return compute_action_mask(
            self._controller.state,
            self._sub_phase,
        )

    def _execute_action(self, decoded) -> Any:
        """Execute a decoded action, updating sub-phase and game state."""
        ctrl = self._controller
        state = ctrl.state
        at = decoded.action_type

        if at == ActionType.BLIND_PLAY:
            blind_type = state.blind_on_deck or "Small"
            ctrl.select_blind(blind_type)
            self._history.start_round(blind_history_key(state))
            self._sub_phase = SubPhase.CHOOSE_ACTION

        elif at == ActionType.BLIND_SKIP:
            skipped_key = (
                int(state.round_resets.ante),
                str(state.blind_on_deck or "Small"),
                "skipped",
            )
            ctrl.skip_blind()
            self._history.start_round(skipped_key)
            # Stay in BLIND_SELECT for next blind
            self._sub_phase = SubPhase.BLIND_SELECT

        elif at == ActionType.BLIND_REROLL:
            ctrl.reroll_boss()
            self._sub_phase = SubPhase.BLIND_SELECT

        elif at == ActionType.PLAY_SUBSET:
            indices = subset_indices(decoded.index)
            if any(idx >= len(state.hand_cards) for idx in indices):
                raise IndexError(f"Play subset {decoded.index} is invalid for hand size {len(state.hand_cards)}")
            selected_cards = [state.hand_cards[index] for index in sorted(indices)]
            pending_history = self._history.capture(
                state,
                selected_cards,
                blind_target=ctrl.blind_target(),
                round_score=ctrl.round_score,
            )
            result = ctrl.play_selected(list(indices))
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
            elif ctrl.phase != GamePhase.GAME_OVER:
                self._sub_phase = SubPhase.CHOOSE_ACTION
            return result

        elif at == ActionType.DISCARD_SUBSET:
            indices = subset_indices(decoded.index)
            if any(idx >= len(state.hand_cards) for idx in indices):
                raise IndexError(f"Discard subset {decoded.index} is invalid for hand size {len(state.hand_cards)}")
            ctrl.discard_selected(list(indices))
            self._sub_phase = SubPhase.CHOOSE_ACTION

        elif at == ActionType.USE_CONSUMABLE_NO_TARGET:
            ctrl.use_consumable_on(decoded.index, hand_targets=(), joker_targets=())
            self._sub_phase = _phase_to_sub_phase(ctrl.phase, self._sub_phase)

        elif at == ActionType.USE_CONSUMABLE_HAND_SUBSET:
            hand_targets = consumable_subset_indices(decoded.detail)
            ctrl.use_consumable_on(decoded.index, hand_targets=hand_targets, joker_targets=())
            self._sub_phase = _phase_to_sub_phase(ctrl.phase, self._sub_phase)

        elif at == ActionType.USE_CONSUMABLE_JOKER:
            ctrl.use_consumable_on(
                decoded.index,
                hand_targets=(),
                joker_targets=(decoded.detail,),
            )
            self._sub_phase = _phase_to_sub_phase(ctrl.phase, self._sub_phase)

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

        elif at == ActionType.MOVE_JOKER:
            move_joker(state, decoded.index, decoded.detail)

    def _build_obs(self) -> RawObservation:
        state = self._controller.state
        mask = self.action_masks()
        return self._tokenizer.tokenize(
            state,
            self._sub_phase,
            action_mask=mask,
            round_score=self._controller.round_score,
            history=self._history,
        )

    def _obs_to_dict(self, obs: RawObservation) -> dict:
        result = {
            "tokens": obs.tokens,
            "token_types": obs.token_types,
            "scalars": obs.scalars,
            "attention_mask": obs.attention_mask,
            "action_mask": obs.action_mask,
        }
        result.update(
            {
                "history_events": obs.history_events,
                "history_event_features": obs.history_event_features,
                "history_cards": obs.history_cards,
                "history_card_mask": obs.history_card_mask,
                "history_jokers": obs.history_jokers,
                "history_joker_mask": obs.history_joker_mask,
                "history_event_mask": obs.history_event_mask,
                "history_round_mask": obs.history_round_mask,
                "history_omitted": obs.history_omitted,
            }
        )
        return result

    def _action_diagnostics(self, decoded) -> dict[str, Any]:
        """Return policy-quality diagnostics for the pre-action state.

        Delegates to the shared diagnostics module so fast_generate
        produces the same fields for BC reward alignment.
        """
        if self._controller is None or self._controller.state is None:
            return {}
        return _shared_action_diagnostics(self._controller.state, decoded)

    def _capture_state_info(self) -> dict:
        if self._controller is None or self._controller.state is None:
            return {}
        state = self._controller.state
        pack_choices_remaining = state.pack.choices_remaining if state.pack is not None else 0
        shop_items = list(state.shop.cards) + list(state.shop.vouchers) + list(state.shop.boosters)
        pack_cards = state.pack.cards if state.pack is not None else ()
        info = {
            "ante": state.round_resets.ante,
            "round_score": self._controller.round_score,
            "blind_on_deck": state.blind_on_deck or "",
            "boss_key": state.round_resets.blind_choices.get("Boss", "") or "",
            "blind_target": self._controller.blind_target(),
            "hands_left": state.current_round.hands_left,
            "discards_left": state.current_round.discards_left,
            "dollars": state.dollars,
            "in_shop": self._controller.phase == GamePhase.SHOP,
            "phase": self._controller.phase,
            "sub_phase": self._sub_phase,
            "reroll_cost": state.current_round.reroll_cost,
            "free_rerolls": state.current_round.free_rerolls,
            "joker_keys": tuple(state.joker_keys),
            "consumable_keys": tuple(state.consumable_keys),
            "last_tarot_planet": state.last_tarot_planet or "",
            "shop_keys": tuple(item.center_key for item in shop_items),
            "pack_booster_key": state.pack.booster_key if state.pack is not None else "",
            "pack_card_keys": tuple(card.center_key for card in pack_cards),
            "pack_state_name": state.pack.state_name if state.pack is not None else "",
            "pack_choices_remaining": pack_choices_remaining,
        }
        info.update(capture_build_features(state))
        return info

    def _progress_signature(self, info: dict[str, Any]) -> tuple[Any, ...]:
        """Return a compact snapshot used to detect meaningful game progress.

        Sub-phases are excluded because the surrounding state already captures
        every meaningful form of progress.
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
            info.get("reroll_cost", 0),
            info.get("free_rerolls", 0),
            tuple(sorted(info.get("joker_keys", ()))),
            info.get("consumable_keys", ()),
            info.get("last_tarot_planet", ""),
            info.get("shop_keys", ()),
            info.get("pack_booster_key", ""),
            info.get("pack_card_keys", ()),
            info.get("pack_choices_remaining", 0),
        )


def _phase_to_sub_phase(phase: GamePhase, current: SubPhase) -> SubPhase:
    """Map the post-action controller phase back to our sub-phase.

    After an atomic consumable commit the controller may have transitioned
    to a new phase (e.g., a card-adding consumable can push us into a
    booster pack menu). Default to CHOOSE_ACTION for HAND_PLAY so the
    policy stays in the same decision loop it came from.
    """
    if phase == GamePhase.HAND_PLAY:
        return SubPhase.CHOOSE_ACTION
    if phase == GamePhase.SHOP:
        return SubPhase.SHOP
    return current
