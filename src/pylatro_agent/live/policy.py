"""Policy loading and inference for live Balatro snapshots."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from pylatro import get_poker_hand_info
from pylatro.models import ConsumableInstance
from pylatro_agent.action import ActionType, decode_action
from pylatro_agent.constants import (
    CONSUMABLE_ACTIONS_PER_SLOT,
    ActionRange,
    SubPhase,
)
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.history import PendingPlay, PlayHistoryTracker, blind_history_key
from pylatro_agent.masks import compute_action_mask
from pylatro_agent.subset_actions import subset_indices
from pylatro_agent.tokenizer import Tokenizer
from pylatro_agent.vocab import build_vocab

from .actions import semantic_action, semantic_targets
from .adapter import LiveState, SnapshotAdapter
from .legality import intersect_live_legality
from .protocol import DecisionRequest, ProtocolError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True)
class Selection:
    action_id: int
    expected_score: float | None = None
    win_probability: float | None = None


class Policy(Protocol):
    def select(
        self,
        live: LiveState,
        tokenizer: Tokenizer,
        action_mask: np.ndarray,
        history: PlayHistoryTracker | None = None,
    ) -> Selection: ...


class HeuristicPolicy:
    def __init__(self) -> None:
        self.agent = HeuristicAgent()

    def select(
        self,
        live: LiveState,
        tokenizer: Tokenizer,
        action_mask: np.ndarray,
        history: PlayHistoryTracker | None = None,
    ) -> Selection:
        del tokenizer, history
        action = self.agent.select_action(
            live.state,
            live.sub_phase,
            action_mask,
            round_score=live.round_score,
        )
        return Selection(int(action))


class CheckpointPolicy:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str,
        d_model: int,
        n_layers: int,
        d_ff: int,
        sample: bool,
        temperature: float,
        vocab: Any,
    ) -> None:
        import torch

        from pylatro_agent.agent import AgentConfig, BalatroAgent
        from pylatro_agent.checkpoint import load_checkpoint_payload

        self.torch = torch
        self.device = torch.device(device)
        self.sample = sample
        self.temperature = temperature
        payload = load_checkpoint_payload(checkpoint, self.device, allow_compatible_tokenizer=True)
        saved_config = payload.get("agent_config")
        config = (
            AgentConfig(**saved_config)
            if isinstance(saved_config, dict)
            else AgentConfig(d_model=d_model, n_layers=n_layers, d_ff=d_ff)
        )
        self.model = BalatroAgent(config, vocab).to(self.device)
        current = self.model.state_dict()
        compatible = {
            key: value
            for key, value in payload["state_dict"].items()
            if key in current and current[key].shape == value.shape
        }
        self.model.load_state_dict(compatible, strict=False)
        self.model.eval()
        logger.info(
            "Loaded live checkpoint %s (%s parameters)",
            checkpoint,
            f"{self.model.count_parameters():,}",
        )

    def select(
        self,
        live: LiveState,
        tokenizer: Tokenizer,
        action_mask: np.ndarray,
        history: PlayHistoryTracker | None = None,
    ) -> Selection:
        torch = self.torch
        obs = tokenizer.tokenize(
            live.state,
            live.sub_phase,
            action_mask=action_mask,
            round_score=live.round_score,
            history=history,
        )
        batch = {
            "tokens": torch.as_tensor(obs.tokens, dtype=torch.long, device=self.device).unsqueeze(0),
            "token_types": torch.as_tensor(obs.token_types, dtype=torch.long, device=self.device).unsqueeze(0),
            "scalars": torch.as_tensor(obs.scalars, dtype=torch.float32, device=self.device).unsqueeze(0),
            "attention_mask": torch.as_tensor(obs.attention_mask, dtype=torch.long, device=self.device).unsqueeze(0),
            "action_mask": torch.as_tensor(obs.action_mask, dtype=torch.float32, device=self.device).unsqueeze(0),
            "history_events": torch.as_tensor(obs.history_events, dtype=torch.long, device=self.device).unsqueeze(0),
            "history_event_features": torch.as_tensor(
                obs.history_event_features, dtype=torch.float32, device=self.device
            ).unsqueeze(0),
            "history_cards": torch.as_tensor(obs.history_cards, dtype=torch.long, device=self.device).unsqueeze(0),
            "history_card_mask": torch.as_tensor(
                obs.history_card_mask, dtype=torch.long, device=self.device
            ).unsqueeze(0),
            "history_jokers": torch.as_tensor(obs.history_jokers, dtype=torch.long, device=self.device).unsqueeze(0),
            "history_joker_mask": torch.as_tensor(
                obs.history_joker_mask, dtype=torch.long, device=self.device
            ).unsqueeze(0),
            "history_event_mask": torch.as_tensor(
                obs.history_event_mask, dtype=torch.long, device=self.device
            ).unsqueeze(0),
            "history_round_mask": torch.as_tensor(
                obs.history_round_mask, dtype=torch.long, device=self.device
            ).unsqueeze(0),
            "history_omitted": torch.as_tensor(
                obs.history_omitted, dtype=torch.float32, device=self.device
            ).unsqueeze(0),
        }
        with torch.no_grad():
            dist, values = self.model.action_distribution(
                batch["tokens"],
                batch["token_types"],
                batch["scalars"],
                batch["attention_mask"],
                batch["action_mask"],
                history_events=batch["history_events"],
                history_event_features=batch["history_event_features"],
                history_cards=batch["history_cards"],
                history_card_mask=batch["history_card_mask"],
                history_jokers=batch["history_jokers"],
                history_joker_mask=batch["history_joker_mask"],
                history_event_mask=batch["history_event_mask"],
                history_round_mask=batch["history_round_mask"],
                history_omitted=batch["history_omitted"],
                temperature=self.temperature,
            )
            action = dist.sample() if self.sample else dist.mode()
        return Selection(
            int(action.item()),
            expected_score=float(values["expected_score"][0].item()),
            win_probability=float(values["win_prob"][0].item()),
        )


class LivePolicyRunner:
    def __init__(self, policy: Policy) -> None:
        self.adapter = SnapshotAdapter()
        self.tokenizer = Tokenizer(build_vocab(self.adapter.data))
        self.policy = policy
        self.history = PlayHistoryTracker()
        self._pending_play: tuple[int, PendingPlay, str] | None = None
        self._pending_skip: tuple[int, tuple[int, str, str]] | None = None

    def reset_session(self) -> None:
        self.history.reset()
        self._pending_play = None
        self._pending_skip = None

    def observe(self, request: DecisionRequest) -> LiveState:
        """Reconcile the prior command against the next authoritative snapshot."""
        live = self.adapter.adapt(request)
        pending_skip = self._pending_skip
        if pending_skip is not None:
            skip_decision_id, skipped_key = pending_skip
            previous = request.previous_action
            matching_result = previous is not None and int(
                previous.get("decision_id", -1)
            ) == skip_decision_id
            moved_to_next_blind = (
                int(live.state.round_resets.ante),
                str(live.state.blind_on_deck or ""),
            ) != skipped_key[:2]
            if (matching_result and bool(previous.get("ok", True))) or (
                previous is None and moved_to_next_blind
            ):
                self.history.start_round(skipped_key)
            self._pending_skip = None

        pending = self._pending_play
        if pending is not None:
            decision_id, captured, hand_type = pending
            previous = request.previous_action
            matching_result = previous is not None and int(
                previous.get("decision_id", -1)
            ) == decision_id
            snapshot_progressed = (
                live.round_score > captured.score_before
                or live.state.current_round.hands_left <= captured.hands_remaining
                or request.phase in {"shop", "terminal"}
            )
            accepted = (
                matching_result and bool(previous.get("ok", True))
            ) or (previous is None and snapshot_progressed)
            if accepted:
                explicit_score = previous.get("score") if matching_result else None
                if isinstance(explicit_score, (int, float)):
                    score = max(0, int(explicit_score))
                else:
                    score = max(0, int(live.round_score) - captured.score_before)
                self.history.finalize(captured, hand_type=hand_type, score=score)
            # A mismatch is the bridge's stale previous result, and no progress
            # means the command was discarded before execution.  Neither may
            # leak forward and attach to a later snapshot.
            self._pending_play = None

        if request.phase == "hand_play":
            self.history.start_round(blind_history_key(live.state))
        return live

    def decide(self, request: DecisionRequest) -> tuple[dict[str, Any], Selection]:
        if request.phase == "terminal":
            raise ProtocolError("terminal snapshots do not have strategic actions")
        live = self.observe(request)
        local_mask = compute_action_mask(live.state, live.sub_phase)
        action_mask = intersect_live_legality(local_mask, live, request.legal)
        selection = self.policy.select(live, self.tokenizer, action_mask, self.history)
        if selection.action_id < 0 or selection.action_id >= len(action_mask) or not action_mask[selection.action_id]:
            raise ProtocolError(f"policy selected illegal action {selection.action_id}")
        wire = semantic_action(selection.action_id, live)
        decoded = decode_action(selection.action_id)
        if decoded.action_type == ActionType.PLAY_SUBSET:
            indices = tuple(subset_indices(decoded.index))
            cards = [live.state.hand_cards[index] for index in indices]
            pending = self.history.capture(
                live.state,
                cards,
                blind_target=int((live.state.round_resets.blind or {}).get("chips") or 0),
                round_score=live.round_score,
            )
            hand_type, _display_name, _hands, _scoring = get_poker_hand_info(live.state, cards)
            self._pending_play = (request.decision_id, pending, hand_type)
        elif decoded.action_type == ActionType.BLIND_SKIP:
            self._pending_skip = (
                request.decision_id,
                (
                    int(live.state.round_resets.ante),
                    str(live.state.blind_on_deck or "Small"),
                    "skipped",
                ),
            )
        if decoded.action_type == ActionType.PACK_CLAIM:
            wire.update(self._pack_targets(live, selection.action_id))
        return wire, selection

    def _pack_targets(self, live: LiveState, action_id: int) -> dict[str, Any]:
        if live.state.pack is None:
            raise ProtocolError("pack claim selected without an active pack")
        pack_index = decode_action(action_id).index
        if pack_index >= len(live.state.pack.cards):
            raise ProtocolError("pack claim references an absent card")
        pack_card = live.state.pack.cards[pack_index]
        center = live.state.data.centers[pack_card.center_key]
        if not center.get("consumeable"):
            return {}

        target_state = deepcopy(live.state)
        target_state.pack = None
        target_state.consumables = [
            ConsumableInstance(
                center_key=pack_card.center_key,
                edition=pack_card.edition,
                sell_cost=max(1, pack_card.cost // 2),
            )
        ]
        target_state.consumable_keys = [pack_card.center_key]
        target_live = LiveState(
            state=target_state,
            sub_phase=SubPhase.CHOOSE_ACTION,
            round_score=live.round_score,
            hand_ids=live.hand_ids,
            joker_ids=live.joker_ids,
            consumable_ids=[live.pack_ids[pack_index]],
            shop_ids=[],
            pack_ids=[],
        )
        target_mask = compute_action_mask(target_state, SubPhase.CHOOSE_ACTION)
        restricted = np.zeros_like(target_mask)
        start = int(ActionRange.CONSUMABLE_FLAT_START)
        restricted[start : start + CONSUMABLE_ACTIONS_PER_SLOT] = target_mask[
            start : start + CONSUMABLE_ACTIONS_PER_SLOT
        ]
        if not restricted.any():
            # Some pack consumables (for example a Planet) are applied by the
            # game's claim callback and need no explicit target selection.
            config = center.get("config") or {}
            if config.get("max_highlighted") is None and center.get("name") not in {
                "Aura",
                "The Wheel of Fortune",
                "Ectoplasm",
                "Hex",
                "Ankh",
            }:
                return {}
            raise ProtocolError(f"pack consumable {pack_card.center_key!r} has no legal target")
        target_selection = self.policy.select(target_live, self.tokenizer, restricted, self.history)
        if not restricted[target_selection.action_id]:
            raise ProtocolError("policy selected an illegal booster target")
        decoded = decode_action(target_selection.action_id)
        if decoded.index != 0 or decoded.action_type not in {
            ActionType.USE_CONSUMABLE_NO_TARGET,
            ActionType.USE_CONSUMABLE_HAND_SUBSET,
            ActionType.USE_CONSUMABLE_JOKER,
        }:
            raise ProtocolError("restricted booster inference returned a non-target action")
        return semantic_targets(decoded, target_live)


def build_live_runner(
    *,
    checkpoint: str | None,
    heuristic: bool,
    device: str,
    d_model: int,
    n_layers: int,
    d_ff: int,
    sample: bool,
    temperature: float,
) -> LivePolicyRunner:
    adapter = SnapshotAdapter()
    vocab = build_vocab(adapter.data)
    if heuristic:
        policy: Policy = HeuristicPolicy()
    elif checkpoint:
        policy = CheckpointPolicy(
            checkpoint,
            device=device,
            d_model=d_model,
            n_layers=n_layers,
            d_ff=d_ff,
            sample=sample,
            temperature=temperature,
            vocab=vocab,
        )
    else:
        raise ValueError("live mode requires a checkpoint or heuristic policy")
    runner = LivePolicyRunner.__new__(LivePolicyRunner)
    runner.adapter = adapter
    runner.tokenizer = Tokenizer(vocab)
    runner.policy = policy
    runner.history = PlayHistoryTracker()
    runner._pending_play = None
    runner._pending_skip = None
    return runner
