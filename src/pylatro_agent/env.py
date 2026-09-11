"""Gymnasium environment wrapping the Balatro game engine."""

from __future__ import annotations

import json
from copy import deepcopy
from pickle import dumps
from typing import Any, ClassVar

import gymnasium
import numpy as np
from gymnasium import spaces

from pylatro import GameData, load_game_data
from pylatro_cli.controller import GameController, GamePhase

from .action import ActionType, decode_action
from .archive import (
    ARCHIVE_VERSION,
    MAX_RETURN_PATH_ACTIONS,
    ArchiveConfig,
    StateArchive,
    observation_fingerprint,
    pack_snapshot,
    unpack_snapshot,
)
from .constants import (
    CURRENT_ANTE_SCALAR_INDEX,
    HISTORY_EVENT_DIM,
    HISTORY_FEATURE_DIM,
    HISTORY_MAX_CARDS,
    HISTORY_MAX_JOKERS,
    HISTORY_MAX_PLAYS,
    HISTORY_OMITTED_DIM,
    HISTORY_ROUNDS,
    MAX_SEQ_LEN,
    META_START,
    NUM_ACTIONS,
    SCALAR_DIM,
    TOKEN_DIM,
    TOKENIZER_VERSION,
    WIN_ANTE_SCALAR_INDEX,
    SubPhase,
)
from .diagnostics import (
    action_diagnostics as _shared_action_diagnostics,
)
from .diagnostics import (
    build_step_diagnostics,
    consumable_funnel_state_diagnostics,
    finish_exact_play_counterfactual,
    prepare_exact_play_counterfactual,
    step_event_diagnostics,
)
from .heuristic import HeuristicAgent
from .history import PlayHistoryTracker, blind_history_key
from .joker_layout import NO_ORDER_DECISION, OrderDecision, OrderObjective, apply_best_joker_order
from .masks import compute_action_mask
from .reward import (
    DEFAULT_REWARD_CONFIG,
    RewardConfig,
    default_reward_components,
)
from .risk import estimate_clear_risk, weakest_confident_joker
from .shop_eval import capture_build_features
from .strategic_events import derive_strategic_event
from .subset_actions import consumable_subset_indices, subset_indices
from .survival import validate_critic_win_ante
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
        archive_config: ArchiveConfig | None = None,
        archive_index: int = 0,
        excluded_seeds: tuple[int, ...] = (),
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
        self._win_ante_override = (
            validate_critic_win_ante(win_ante) if win_ante is not None else None
        )
        self._reward_config = reward_config or DEFAULT_REWARD_CONFIG
        self._seed = seed
        self._initial_seed_pending = seed is not None
        self._excluded_seeds = frozenset(int(value) for value in excluded_seeds)
        if archive_config is not None and (win_ante or 8) != 8:
            raise ValueError("Archive training always targets Ante 8")
        self._archive = StateArchive(archive_config, seed=seed) if archive_config is not None else None
        self._archive_index = archive_index
        self._archive_start = False
        self._start_ante = 1
        self._action_lineage: list[int] = []
        self._lineage_complete = True
        self._archive_prefix_length = 0
        self._archive_prefix_fingerprint = ""
        self._cleared_boss_antes: set[int] = set()
        self._milestone_scale = 1.0
        if counterfactual_diagnostic_interval < 0:
            raise ValueError("counterfactual_diagnostic_interval must be non-negative")
        self._counterfactual_diagnostic_interval = int(counterfactual_diagnostic_interval)
        self._play_diagnostic_count = 0

        self._controller: GameController | None = None
        self._sub_phase = SubPhase.BLIND_SELECT
        self._steps_since_progress = 0
        self._history = PlayHistoryTracker()
        # What the harness's joker ordering did on the most recent play. The
        # policy observes this so cash the harness banked is attributable
        # rather than appearing as unexplained variance in its own dollars.
        self._last_order_decision: OrderDecision = NO_ORDER_DECISION

        # Previous state info for reward computation
        self._prev_info: dict[str, Any] = {}
        # A Joker replacement is a two-action transaction (sell, then buy).
        # Preserve the pre-sale safety baseline so the realized purchase reward
        # compares the complete old and new rosters rather than new-vs-empty.
        self._joker_replacement_clear_baseline: float | None = None
        # Episode-stable, process-monotonic card UIDs make Gold creation
        # attributable exactly once without Python object-id reuse collisions.
        self._rewarded_gold_card_ids: set[int] = set()
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
                "tokens": spaces.Box(-32768, 32767, (MAX_SEQ_LEN, TOKEN_DIM), dtype=np.int16),
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
                "history_event_mask": spaces.Box(0, 1, (HISTORY_ROUNDS, HISTORY_MAX_PLAYS), dtype=np.int8),
                "history_round_mask": spaces.Box(0, 1, (HISTORY_ROUNDS,), dtype=np.int8),
                "history_omitted": spaces.Box(-np.inf, np.inf, (HISTORY_ROUNDS, HISTORY_OMITTED_DIM), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

    @property
    def state(self):
        return self._controller.state if self._controller else None

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        if seed is not None and seed in self._excluded_seeds:
            raise ValueError("Training cannot reset to a reserved evaluation seed")
        # Explicit seeds mean reproducible fresh evaluation. Gym autoresets
        # have no seed; only those may draw a training continuation.
        snapshot = self._archive.choose(
            force_fresh=seed is not None or bool((options or {}).get("fresh_start")),
        ) if self._archive is not None else None
        if snapshot is not None:
            self.restore_snapshot(unpack_snapshot(snapshot, self._data))
            self._archive_start = True
            self._start_ante = int(self.state.round_resets.ante)
            obs = self._build_obs(self._prev_info)
            return self._obs_to_dict(obs), self._reset_info()
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

        while effective_seed in self._excluded_seeds:
            effective_seed = int(self.np_random.integers(0, 2**31))

        self._controller = GameController(data=self._data)
        self._controller.new_run(str(effective_seed), stake=self._stake, deck_key=self._deck_key)
        if self._win_ante_override is not None and self._controller.state is not None:
            self._controller.state.win_ante = int(self._win_ante_override)

        self._sub_phase = SubPhase.BLIND_SELECT
        self._steps_since_progress = 0
        self._play_diagnostic_count = 0
        self._joker_replacement_clear_baseline = None
        self._rewarded_gold_card_ids.clear()
        self._cleared_boss_antes.clear()
        self._archive_start = False
        self._start_ante = 1
        self._action_lineage = []
        self._lineage_complete = True
        self._archive_prefix_length = 0
        self._archive_prefix_fingerprint = ""
        self._last_order_decision = NO_ORDER_DECISION
        self._history.reset()
        self._prev_info = self._capture_state_info()

        obs = self._build_obs(self._prev_info)
        return self._obs_to_dict(obs), self._reset_info()

    def _reset_info(self) -> dict:
        return {
            "sub_phase": self._sub_phase,
            "teacher_action": self._current_teacher_action(),
            "archive_start": self._archive_start,
            "start_ante": self._start_ante,
        }

    def snapshot(self) -> dict[str, Any]:
        """Capture all continuation state, preserving card aliases and engine RNG.

        Archive RNG and the training reward schedule belong to the sampler,
        not the saved game; returning must not rewind either of them.
        """
        if self._controller is None or self._controller.phase not in (GamePhase.SHOP, GamePhase.BLIND_SELECT):
            raise ValueError("Snapshots require a nonterminal shop or blind-selection boundary")
        return deepcopy({
            "version": ARCHIVE_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "stake": self._stake,
            "deck_key": self._deck_key,
            "controller": self._controller,
            "sub_phase": self._sub_phase,
            "history": self._history,
            "last_order_decision": self._last_order_decision,
            "steps_since_progress": self._steps_since_progress,
            "play_diagnostic_count": self._play_diagnostic_count,
            "joker_replacement_clear_baseline": self._joker_replacement_clear_baseline,
            "rewarded_gold_card_ids": self._rewarded_gold_card_ids,
            "cleared_boss_antes": self._cleared_boss_antes,
            "action_lineage": self._action_lineage,
            "lineage_complete": self._lineage_complete,
        }, {id(self._data): self._data})

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        if (snapshot.get("version") != ARCHIVE_VERSION
                or snapshot.get("tokenizer_version") != TOKENIZER_VERSION
                or snapshot.get("stake") != self._stake
                or snapshot.get("deck_key") != self._deck_key):
            raise ValueError("Snapshot schema, stake, or deck mismatch")
        saved_controller = snapshot["controller"]
        if (saved_controller.phase not in (GamePhase.SHOP, GamePhase.BLIND_SELECT)
                or saved_controller.state is None
                or saved_controller.state.won
                or saved_controller.state.win_ante != (self._win_ante_override or 8)):
            raise ValueError("Snapshot must be a live boundary with the same victory target")
        if str(saved_controller.state.seed) in {str(seed) for seed in self._excluded_seeds}:
            raise ValueError("Archive contains a reserved evaluation seed")
        saved = deepcopy(snapshot, {id(saved_controller.data): self._data})
        self._controller = saved["controller"]
        self._sub_phase = saved["sub_phase"]
        self._history = saved["history"]
        self._last_order_decision = saved["last_order_decision"]
        self._steps_since_progress = saved["steps_since_progress"]
        self._play_diagnostic_count = saved["play_diagnostic_count"]
        self._joker_replacement_clear_baseline = saved["joker_replacement_clear_baseline"]
        self._rewarded_gold_card_ids = saved["rewarded_gold_card_ids"]
        # Reward IDs are process-local. Checkpoint archives may come from a
        # previous process whose counter overlaps newly created cards here.
        from pylatro.models import _next_playing_card_uid

        paid_ids = self._rewarded_gold_card_ids
        self._rewarded_gold_card_ids = set()
        cards = {id(card): card for pile in (
            self.state.deck_cards, self.state.hand_cards,
            self.state.draw_pile, self.state.discard_pile,
        ) for card in pile}
        for card in cards.values():
            paid = card.reward_uid in paid_ids
            card.reward_uid = _next_playing_card_uid()
            if paid:
                self._rewarded_gold_card_ids.add(card.reward_uid)
        self._cleared_boss_antes = saved["cleared_boss_antes"]
        self._action_lineage = list(saved["action_lineage"])
        self._lineage_complete = bool(saved["lineage_complete"])
        self._archive_prefix_length = len(self._action_lineage)
        self._prev_info = self._capture_state_info()
        self._archive_prefix_fingerprint = observation_fingerprint(self._obs_to_dict(self._build_obs(self._prev_info)))

    def _archive_boundary(self, prev_info: dict) -> None:
        if self._archive is None or self._sub_phase not in (SubPhase.SHOP, SubPhase.BLIND_SELECT):
            return
        ante = int(self.state.round_resets.ante)
        if not self._archive.config.min_ante <= ante <= self._archive.config.max_ante:
            return
        before = (prev_info.get("ante"), str(prev_info.get("sub_phase")), prev_info.get("blind_on_deck"))
        after = (ante, str(self._sub_phase), self.state.blind_on_deck)
        if before == after:
            return
        identity = f"{self.state.seed}:{ante}:{self._sub_phase}:{self.state.blind_on_deck}"
        self._archive.add(ante=ante, phase=str(self._sub_phase), identity=identity,
                          payload=pack_snapshot(self.snapshot(), self._data))

    def archive_state_dict(self) -> dict | None:
        return self._archive.state_dict() if self._archive is not None else None

    def load_archive_states(self, states: list[dict]) -> None:
        if self._archive is None or self._archive_index >= len(states):
            raise ValueError("Cannot restore archive for this worker")
        self._archive.load_state_dict(states[self._archive_index])

    def archive_metrics(self) -> dict[str, float]:
        return self._archive.metrics() if self._archive is not None else {}

    def set_milestone_scale(self, scale: float) -> None:
        if not 0 <= scale <= 1:
            raise ValueError("Milestone scale must be between 0 and 1")
        self._milestone_scale = float(scale)

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
        action_diagnostics.update(
            consumable_funnel_state_diagnostics(self._prev_info, self.action_masks())
        )
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

            self._lineage_complete = False

            logging.getLogger(__name__).warning(f"Action {action} raised {type(e).__name__}: {e}")
            reward = -1.0
            obs = self._build_obs(self._prev_info)
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

        if self._lineage_complete:
            if len(self._action_lineage) < MAX_RETURN_PATH_ACTIONS:
                self._action_lineage.append(int(action))
            else:
                self._lineage_complete = False
                self._action_lineage.clear()

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
        if decoded.action_type == ActionType.PLAY_SUBSET:
            decision = self._last_order_decision
            action_diagnostics.update(
                {
                    "joker_order_objective": str(decision.objective),
                    "joker_order_changed": bool(decision.changed),
                    "joker_order_clears": bool(decision.clears),
                    "joker_order_dollars_gained": float(decision.dollars_gained),
                    "joker_order_chips_forgone": float(decision.chips_forgone),
                }
            )

        event_diagnostics = step_event_diagnostics(
            self._prev_info,
            curr_info,
            decoded,
            next_action_mask=None if terminated else self.action_masks(),
        )
        action_diagnostics.update(event_diagnostics)
        strategic_event = derive_strategic_event(
            self._prev_info,
            curr_info,
            decoded,
            action_result,
            self._rewarded_gold_card_ids,
        )
        action_diagnostics.update(strategic_event.as_info())
        if (
            decoded.action_type == ActionType.SHOP_SELL_JOKER
            and action_diagnostics.get("shop_sold_joker_id")
            and self._joker_replacement_clear_baseline is None
        ):
            self._joker_replacement_clear_baseline = float(self._prev_info.get("clear_probability", 0.0) or 0.0)
        elif decoded.action_type == ActionType.SHOP_BUY and action_diagnostics.get("shop_bought_joker_id"):
            if self._joker_replacement_clear_baseline is not None:
                curr_info["joker_upgrade_baseline_clear_probability"] = self._joker_replacement_clear_baseline
                action_diagnostics["joker_replacement_sequence"] = True
            self._joker_replacement_clear_baseline = None
        elif decoded.action_type == ActionType.SHOP_LEAVE:
            self._joker_replacement_clear_baseline = None
        if counterfactual_probe is not None:
            actual_score = float(action_result.score.total)
            action_diagnostics.update(
                finish_exact_play_counterfactual(
                    counterfactual_probe,
                    actual_score=actual_score,
                )
            )
        if (
            decoded.action_type == ActionType.PLAY_SUBSET
            and int(self._prev_info.get("ante", 0) or 0) == 1
        ):
            realized_score = float(action_result.score.total)
            previous_score = float(self._prev_info.get("round_score", 0.0) or 0.0)
            blind_target = float(self._prev_info.get("blind_target", 0.0) or 0.0)
            remaining_target = max(blind_target - previous_score, 0.0)
            action_diagnostics.update(
                {
                    "ante1_play_observed": True,
                    "ante1_play_hand": str(action_result.score.hand_name),
                    "ante1_play_realized_score": realized_score,
                }
            )
            if remaining_target > 0.0:
                action_diagnostics["ante1_play_realized_to_remaining_target"] = (
                    realized_score / remaining_target
                )
            if blind_target > 0.0 and previous_score + realized_score >= blind_target:
                action_diagnostics.update(
                    {
                        "ante1_blind_cleared": True,
                        "ante1_blind_clear_type": str(
                            self._prev_info.get("blind_on_deck", "") or ""
                        ).lower(),
                        "ante1_blind_clear_hands_used": int(
                            state.current_round.hands_played
                        ),
                        # cash_out replenishes hands_left before step() returns,
                        # so the exact pre-play remainder is the stable source.
                        "ante1_blind_clear_hands_unused": max(
                            int(self._prev_info.get("hands_left", 0) or 0) - 1,
                            0,
                        ),
                        "ante1_blind_clear_discards_used": int(
                            state.current_round.discards_used
                        ),
                    }
                )

        # Detailed leave-one-out build diagnostics are expensive. When score-build
        # potential is active they also provide the cache consumed by the reward;
        # otherwise compute them only for actual build/shop events.
        score_build_potential_enabled = (
            self._reward_config.objective == "shaped" and self._reward_config.enable_score_build_potential
        )
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

        if terminated or truncated:
            for consumable_set in ("Planet", "Tarot"):
                expired = sum(
                    isinstance(detail, dict) and detail.get("set") == consumable_set
                    for detail in (curr_info.get("consumable_details") or ())
                )
                action_diagnostics[
                    f"strategic_{consumable_set.lower()}_expired"
                ] = int(expired)

        # Surface action diagnostics used by optional reward components.
        curr_info.update(action_diagnostics)
        cleared_ante = int(self._prev_info.get("ante", 1))
        boss_clear = (
            decoded.action_type == ActionType.PLAY_SUBSET
            and str(self._prev_info.get("blind_on_deck", "")).lower() == "boss"
            and int(curr_info["ante"]) > cleared_ante
        )
        if boss_clear and cleared_ante not in self._cleared_boss_antes:
            self._cleared_boss_antes.add(cleared_ante)
            curr_info["boss_cleared_ante"] = cleared_ante
        curr_info["milestone_scale"] = self._milestone_scale

        reward_components = default_reward_components(
            state, self._prev_info, curr_info, terminated, won, self._reward_config
        )
        reward = reward_components["total"]

        obs = self._build_obs(curr_info)
        info = {
            "sub_phase": self._sub_phase,
            "pre_sub_phase": pre_sub_phase,
            "ante": state.round_resets.ante,
            "blind_on_deck": state.blind_on_deck or "",
            "boss_key": state.round_resets.blind_choices.get("Boss", "") or "",
            "dollars": state.dollars,
            "round_score": self._controller.round_score,
            "blind_target": curr_info.get("blind_target", 0.0),
            "clear_probability": curr_info.get("clear_probability", 0.0),
            "immediate_death_probability": curr_info.get("immediate_death_probability", 1.0),
            "risk_model_confidence": curr_info.get("risk_model_confidence", 0.0),
            "risk_score_margin": curr_info.get("risk_score_margin", 0.0),
            "joker_count": curr_info.get("joker_count", 0),
            "joker_limit": curr_info.get("joker_limit", 0),
            "joker_full": curr_info.get("joker_full", False),
            "weak_confident_joker": curr_info.get("weak_confident_joker", False),
            "won": won,
            "progress_made": progress_made,
            "steps_since_progress": self._steps_since_progress,
            "stalled": curr_info["stalled"],
            "tarot_usage_total": curr_info.get("tarot_usage_total", 0),
            "planet_usage_total": curr_info.get("planet_usage_total", 0),
            "archive_start": self._archive_start,
            "start_ante": self._start_ante,
            "origin_seed": str(state.seed),
            "archive_prefix_length": self._archive_prefix_length,
            "boss_cleared_ante": curr_info.get("boss_cleared_ante", 0),
        }
        for component_name, component_value in reward_components.items():
            info[f"reward_{component_name}"] = component_value
        info.update(action_diagnostics)
        info["teacher_action"] = teacher_action
        info["teacher_action_match"] = curr_info["teacher_action_match"]
        info["next_teacher_action"] = -1 if terminated or truncated else self._current_teacher_action()
        if won and self._archive_start and self._lineage_complete and self._archive_prefix_length > 0:
            # Small recipe, not an observation trajectory, crosses the worker
            # boundary. Reconstructed prefixes are validated and used ONLY by
            # the separately weighted return-path BC loss in the learner.
            info["winning_return_path"] = json.dumps({
                "tokenizer_version": TOKENIZER_VERSION,
                "seed": str(state.seed), "stake": self._stake, "deck_key": self._deck_key,
                "actions": self._action_lineage[:self._archive_prefix_length],
                "boundary_fingerprint": self._archive_prefix_fingerprint,
                "won": True, "win_ante": int(state.win_ante),
            }).encode()
        if not terminated and not truncated:
            self._archive_boundary(self._prev_info)
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
        return compute_action_mask(self._controller.state, self._sub_phase)

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
            # Harness-owned joker ordering: put the roster in the best
            # exact-scored arrangement for this concrete play before scoring.
            # The remaining target lets the harness bank cash on plays that
            # clear the blind either way; without it the objective is SCORE.
            self._last_order_decision = apply_best_joker_order(
                state,
                tuple(sorted(indices)),
                remaining_target=max(float(ctrl.blind_target()) - float(ctrl.round_score), 0.0),
            )
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
            result = ctrl.discard_selected(list(indices))
            self._sub_phase = SubPhase.CHOOSE_ACTION
            return result

        elif at == ActionType.USE_CONSUMABLE_NO_TARGET:
            result = ctrl.use_consumable_on(decoded.index, hand_targets=(), joker_targets=())
            self._sub_phase = _phase_to_sub_phase(ctrl.phase, self._sub_phase)
            return result

        elif at == ActionType.USE_CONSUMABLE_HAND_SUBSET:
            hand_targets = consumable_subset_indices(decoded.detail)
            result = ctrl.use_consumable_on(decoded.index, hand_targets=hand_targets, joker_targets=())
            self._sub_phase = _phase_to_sub_phase(ctrl.phase, self._sub_phase)
            return result

        elif at == ActionType.USE_CONSUMABLE_JOKER:
            result = ctrl.use_consumable_on(
                decoded.index,
                hand_targets=(),
                joker_targets=(decoded.detail,),
            )
            self._sub_phase = _phase_to_sub_phase(ctrl.phase, self._sub_phase)
            return result

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
            if ctrl.phase == GamePhase.SHOP:
                self._sub_phase = SubPhase.SHOP

        elif at == ActionType.SHOP_SELL_CONSUMABLE:
            ctrl.sell_consumable(decoded.index)

        elif at == ActionType.SHOP_LEAVE:
            ctrl.leave_shop()
            self._sub_phase = SubPhase.BLIND_SELECT

        elif at == ActionType.PACK_CLAIM:
            result = ctrl.claim_from_pack(decoded.index)
            if state.pack and state.pack.choices_remaining <= 0:
                ctrl.close_current_pack(skipped=False)
                self._sub_phase = SubPhase.SHOP
            return result

        elif at == ActionType.PACK_SKIP:
            ctrl.close_current_pack(skipped=True)
            self._sub_phase = SubPhase.SHOP

    def _build_obs(self, state_info: dict | None = None) -> RawObservation:
        state = self._controller.state
        mask = self.action_masks()
        risk_info = state_info if state_info is not None else self._capture_state_info()
        obs = self._tokenizer.tokenize(
            state,
            self._sub_phase,
            action_mask=mask,
            round_score=self._controller.round_score,
            history=self._history,
            clear_probability=float(risk_info.get("clear_probability", 0.0) or 0.0),
            immediate_death_probability=float(risk_info.get("immediate_death_probability", 1.0) or 0.0),
            order_objective_money=self._last_order_decision.objective is OrderObjective.MONEY,
            order_dollars_gained=float(self._last_order_decision.dollars_gained),
        )
        # cash_out marks a curriculum win and then advances the engine to the
        # next Ante (for example, a target-Ante-4 win leaves state.ante == 5).
        # That post-win bookkeeping state is still returned as Gymnasium's
        # terminal observation.  The conditional-survival critic, however,
        # describes outcomes only through the configured target Ante, so encode
        # the last playable Ante rather than an impossible current > target
        # pair.  Keep the engine state untouched: terminal info should continue
        # reporting the actual post-cash-out Ante.
        if state.won and obs.scalars[CURRENT_ANTE_SCALAR_INDEX] > obs.scalars[WIN_ANTE_SCALAR_INDEX]:
            obs.scalars[CURRENT_ANTE_SCALAR_INDEX] = obs.scalars[WIN_ANTE_SCALAR_INDEX]
            obs.tokens[META_START + CURRENT_ANTE_SCALAR_INDEX, 0] = int(state.win_ante)
        return obs

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
        return _shared_action_diagnostics(
            self._controller.state,
            decoded,
            action_mask=self.action_masks(),
            round_score=float(self._controller.round_score),
            blind_target=float(self._controller.blind_target()),
        )

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
        # A post-step capture is normally repeated as the next pre-step
        # capture. Compare fresh features so direct engine mutations, resets,
        # and archive restores also invalidate this bounded one-entry cache.
        cached = getattr(self, "_state_estimate_cache", None)
        cache_key = dumps(info, protocol=5)
        if cached is not None and cache_key == cached[0]:
            risk, weakest = cached[1:]
        else:
            risk = estimate_clear_risk(info)
            weakest = weakest_confident_joker(info)
            # Immutable bytes isolate the key from caller mutations without
            # deep-copying every card descriptor in Python. Never unpickled.
            self._state_estimate_cache = (cache_key, risk, weakest)
        joker_count = len(info.get("joker_details") or ())
        joker_limit = max(int(info.get("joker_limit", 5) or 5), 0)
        info.update(
            {
                "clear_probability": risk.clear_probability,
                "immediate_death_probability": risk.immediate_death_probability,
                "risk_model_confidence": risk.model_confidence,
                "risk_score_margin": risk.score_margin,
                "risk_hand_type": risk.hand_type,
                "joker_count": joker_count,
                "joker_full": joker_count >= joker_limit,
                "weak_confident_joker": bool(weakest is not None and weakest[0] <= 1.05),
                "weakest_confident_joker_ratio": float(weakest[0]) if weakest is not None else 0.0,
            }
        )
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
