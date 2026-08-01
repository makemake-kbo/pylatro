from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from pylatro_agent.action import ActionType, encode_action
from pylatro_agent.constants import NUM_ACTIONS, ActionRange, SubPhase
from pylatro_agent.hand_candidates import generate_hand_candidates
from pylatro_agent.live.actions import semantic_action
from pylatro_agent.live.adapter import SnapshotAdapter
from pylatro_agent.live.legality import intersect_live_legality
from pylatro_agent.live.policy import HeuristicPolicy, LivePolicyRunner, Selection
from pylatro_agent.live.protocol import DecisionRequest, DecisionResponse, ProtocolError
from pylatro_agent.live.server import DecisionService
from pylatro_agent.masks import compute_action_mask
from pylatro_agent.subset_actions import subset_index

FIXTURE_PATH = Path(__file__).parent / "fixtures/live/protocol_v1_cases.json"


def _legal() -> dict:
    return {
        "blind_play": True,
        "blind_skip": True,
        "blind_reroll": True,
        "play": True,
        "discard": True,
        "play_card_ids": ["h1", "h2"],
        "discard_card_ids": ["h1", "h2"],
        "use_consumable_ids": ["c1"],
        "shop_buy_ids": ["shop1"],
        "shop_reroll": True,
        "shop_sell_joker_ids": ["j1"],
        "shop_sell_consumable_ids": ["c1"],
        "shop_leave": True,
        "pack_claim_ids": ["pack1"],
        "pack_skip": True,
    }


def _state() -> dict:
    return {
        "seed": "LIVE",
        "stake": 1,
        "deck_key": "b_red",
        "dollars": 10,
        "ante": 1,
        "blind_on_deck": "Small",
        "hands_left": 4,
        "discards_left": 3,
        "hand_size": 8,
        "joker_slots": 5,
        "consumable_slots": 2,
        "reroll_cost": 5,
        "round_score": 25,
        "blind": {"key": "bl_small", "chips": 300},
        "blind_choices": {"Small": "bl_small", "Big": "bl_big", "Boss": "bl_hook"},
        "cards": {
            "hand": [
                {"id": "h1", "front_key": "S_A"},
                {"id": "h2", "front_key": "H_K"},
            ],
            "draw": [],
            "discard": [],
            "deck": [],
        },
        "jokers": [],
        "consumables": [],
        "vouchers": {},
        "hands": {},
        "shop": {},
        "unsupported": [],
    }


def _merge(target: dict, patch: dict) -> dict:
    result = deepcopy(target)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _request(*, phase: str = "hand_play", state: dict | None = None, decision_id: int = 1):
    return DecisionRequest.from_dict(
        {
            "protocol_version": 1,
            "session_id": "session",
            "decision_id": decision_id,
            "state_fingerprint": f"fp-{decision_id}",
            "phase": phase,
            "versions": {
                "balatro": "1.0.1o-FULL",
                "steamodded": "1.0.0",
                "bridge": "0.1.0",
            },
            "state": state or _state(),
            "legal": _legal(),
        }
    )


def test_protocol_round_trip_and_unknown_fields():
    request = _request()
    assert request.identity == ("session", 1, "hand_play", "fp-1")
    response = DecisionResponse.action_response(request, {"type": "play", "card_ids": ["h1"]})
    assert response.to_dict()["action"]["type"] == "play"

    raw = {
        "protocol_version": 1,
        "session_id": "s",
        "decision_id": 0,
        "state_fingerprint": "f",
        "phase": "shop",
        "versions": {},
        "state": {},
        "legal": {},
        "surprise": True,
    }
    with pytest.raises(ProtocolError, match="unknown request fields"):
        DecisionRequest.from_dict(raw)
    raw.pop("surprise")
    raw["protocol_version"] = 2
    with pytest.raises(ProtocolError, match="unsupported protocol_version"):
        DecisionRequest.from_dict(raw)
    raw["protocol_version"] = 1
    raw["versions"] = {"balatro": "future"}
    with pytest.raises(ProtocolError, match="unsupported Balatro version"):
        DecisionRequest.from_dict(raw)


@pytest.mark.parametrize("case", json.loads(FIXTURE_PATH.read_text()))
def test_golden_live_snapshots_validate(case):
    state = _merge(_state(), case["patch"])
    request = _request(phase=case["phase"], state=state)
    live = SnapshotAdapter().adapt(request)
    assert live.sub_phase in SubPhase
    if case["name"] == "face_down_hand":
        assert live.state.hand_cards[0].face_down
    if case["name"] == "ten_rank_hand":
        assert live.state.hand_cards[0].rank == "T"
        play_candidates, discard_candidates = generate_hand_candidates(live.state)
        assert play_candidates
        assert discard_candidates
    if case["name"] == "boss_blind":
        assert live.state.round_resets.blind["chips"] == 900
    if case["phase"] == "booster_pack":
        assert live.pack_ids == ["pack1"]


def test_adapter_preserves_order_and_exact_live_blind_target():
    live = SnapshotAdapter().adapt(_request())
    assert live.hand_ids == ["h1", "h2"]
    assert [card.rank for card in live.state.hand_cards] == ["A", "K"]
    blind = live.state.round_resets.blind
    assert blind is not None
    from pylatro import get_blind_amount

    target = int(get_blind_amount(1, 1) * blind["mult"])
    assert target == 300


def test_adapter_accepts_lua_json_empty_table_ambiguity():
    state = _state()
    state["vouchers"] = []
    state["blind_tags"] = []
    live = SnapshotAdapter().adapt(_request(state=state))
    assert live.state.used_vouchers == {}
    assert live.state.round_resets.blind_tags == {}


def test_unknown_modded_content_fails_closed():
    state = _state()
    state["unsupported"] = ["other_mod_joker"]
    with pytest.raises(ProtocolError, match="unsupported gameplay content"):
        SnapshotAdapter().adapt(_request(state=state))

    state = _state()
    state["jokers"] = [{"id": "j1", "center_key": "other_joker"}]
    with pytest.raises(ProtocolError, match="unknown vanilla center"):
        SnapshotAdapter().adapt(_request(state=state))


def test_live_mask_intersection_and_semantic_card_ids():
    request = _request()
    live = SnapshotAdapter().adapt(request)
    local = compute_action_mask(live.state, SubPhase.CHOOSE_ACTION)
    request.legal["play_card_ids"].remove("h2")
    mask = intersect_live_legality(local, live, request.legal)
    h1 = int(ActionRange.PLAY_SUBSET_START) + subset_index((0,))
    h2 = int(ActionRange.PLAY_SUBSET_START) + subset_index((1,))
    assert mask[h1]
    assert not mask[h2]
    assert semantic_action(h1, live) == {"type": "play", "card_ids": ["h1"]}


def test_legality_fails_closed_when_no_action_remains():
    live = SnapshotAdapter().adapt(_request())
    local = np.zeros(NUM_ACTIONS, dtype=np.int8)
    with pytest.raises(ProtocolError, match="no action remains"):
        intersect_live_legality(local, live, _legal())


def test_heuristic_live_inference_returns_semantic_action():
    action, selection = LivePolicyRunner(HeuristicPolicy()).decide(_request())
    assert action["type"] in {"play", "discard"}
    assert set(action["card_ids"]).issubset({"h1", "h2"})
    assert selection.action_id >= 0


def test_targeted_pack_card_runs_restricted_second_inference():
    state = _merge(
        _state(),
        {
            "pack": {
                "booster_key": "p_arcana_normal_1",
                "state_name": "TAROT_PACK",
                "choices_remaining": 1,
                "cards": [
                    {
                        "id": "pack1",
                        "center_key": "c_magician",
                        "card_type": "Tarot",
                        "cost": 0,
                    }
                ],
            }
        },
    )
    action, _ = LivePolicyRunner(HeuristicPolicy()).decide(
        _request(phase="booster_pack", state=state)
    )
    assert action["type"] == "pack_claim"
    assert action["item_id"] == "pack1"
    assert action["card_ids"]
    assert set(action["card_ids"]).issubset({"h1", "h2"})


def test_semantic_actions_never_return_flat_index():
    live = SnapshotAdapter().adapt(_request())
    cases = [
        int(ActionRange.BLIND_PLAY),
        encode_action(ActionType.PLAY_SUBSET, subset_index((0, 1))),
        int(ActionRange.SHOP_REROLL),
        int(ActionRange.SHOP_LEAVE),
        int(ActionRange.PACK_SKIP),
    ]
    for action_id in cases:
        wire = semantic_action(action_id, live)
        assert "action_id" not in wire
        assert "index" not in wire


class _Runner:
    def __init__(self):
        self.calls = 0

    def decide(self, request):
        self.calls += 1
        return {"type": "blind_play"}, type(
            "Selection", (), {"expected_score": None, "win_probability": None}
        )()


class _FirstPlayPolicy:
    def select(self, live, tokenizer, action_mask, history=None):
        del live, tokenizer, history
        start = int(ActionRange.PLAY_SUBSET_START)
        end = int(ActionRange.PLAY_SUBSET_END) + 1
        choices = np.flatnonzero(action_mask[start:end])
        assert len(choices)
        return Selection(start + int(choices[0]))


class _SkipPolicy:
    def select(self, live, tokenizer, action_mask, history=None):
        del live, tokenizer, history
        action = int(ActionRange.BLIND_SKIP)
        assert action_mask[action]
        return Selection(action)


def _with_previous(request: DecisionRequest, decision_id: int, *, ok: bool) -> DecisionRequest:
    object.__setattr__(
        request,
        "previous_action",
        {"decision_id": decision_id, "ok": ok, "error": None if ok else "rejected"},
    )
    return request


def test_live_history_finalizes_success_rejects_failure_and_rotates_rounds():
    runner = LivePolicyRunner(_FirstPlayPolicy())
    runner.decide(_request(decision_id=1))
    successful_state = _state()
    successful_state["round_score"] = 65
    runner.decide(_with_previous(_request(state=successful_state, decision_id=2), 1, ok=True))
    assert len(runner.history.rounds[-1].events) == 1
    assert runner.history.rounds[-1].events[0].score == 40

    rejected_runner = LivePolicyRunner(_FirstPlayPolicy())
    rejected_runner.decide(_request(decision_id=1))
    rejected_runner.decide(_with_previous(_request(decision_id=2), 1, ok=False))
    assert not rejected_runner.history.rounds[-1].events

    # Reconcile decision 2, then enter a different blind.  The completed Small
    # Blind becomes the immediately preceding round; no synthetic event is made
    # for the new blind.
    shop_state = deepcopy(successful_state)
    shop_request = _with_previous(
        _request(phase="shop", state=shop_state, decision_id=3), 2, ok=False
    )
    runner.observe(shop_request)
    big_state = _merge(
        _state(),
        {
            "blind_on_deck": "Big",
            "round_score": 0,
            "blind": {"key": "bl_big", "chips": 450},
        },
    )
    runner.observe(_request(state=big_state, decision_id=4))
    assert len(runner.history.rounds[-2].events) == 1
    assert not runner.history.rounds[-1].events


def test_live_terminal_play_is_finalized_and_duplicate_is_not_replayed():
    runner = LivePolicyRunner(_FirstPlayPolicy())
    service = DecisionService(runner)
    first = _request(decision_id=1)
    service.handle(first)
    service.handle(first)
    assert not runner.history.rounds[-1].events

    terminal_state = _state()
    terminal_state["round_score"] = 75
    terminal = _with_previous(
        _request(phase="terminal", state=terminal_state, decision_id=2), 1, ok=True
    )
    response = service.handle(terminal)
    assert response.wait
    assert len(runner.history.rounds[-1].events) == 1
    assert runner.history.rounds[-1].events[0].score == 50


def test_live_accepted_skip_advances_history_without_inventing_a_play():
    runner = LivePolicyRunner(_SkipPolicy())
    runner.decide(_request(phase="blind_select", decision_id=1))
    big_state = _merge(
        _state(),
        {"blind_on_deck": "Big", "blind": {"key": "bl_big", "chips": 450}},
    )
    runner.decide(
        _with_previous(
            _request(phase="blind_select", state=big_state, decision_id=2),
            1,
            ok=True,
        )
    )
    assert runner.history.rounds[-1].key == (1, "Small", "skipped")
    assert not runner.history.rounds[-1].events


def test_duplicate_requests_execute_inference_once_and_conflicts_are_rejected():
    runner = _Runner()
    service = DecisionService(runner)
    request = _request(phase="blind_select")
    first = service.handle(request)
    second = service.handle(request)
    assert first == second
    assert runner.calls == 1

    raw = {
        "protocol_version": 1,
        "session_id": request.session_id,
        "decision_id": request.decision_id,
        "state_fingerprint": "changed",
        "phase": request.phase,
        "versions": request.versions,
        "state": request.state,
        "legal": request.legal,
    }
    conflict = service.handle(DecisionRequest.from_dict(raw))
    assert conflict.error["code"] == "decision_conflict"


def test_one_active_session_and_terminal_exit():
    service = DecisionService(_Runner())
    service.handle(_request(phase="blind_select"))
    other = _request(phase="blind_select", decision_id=2)
    object.__setattr__(other, "session_id", "other")
    assert service.handle(other).error["code"] == "session_busy"

    terminal = _request(phase="terminal", decision_id=2)
    response = service.handle(terminal)
    assert response.wait is True
    assert service.terminal is True
