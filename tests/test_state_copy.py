"""State copying must isolate probes while retaining the original object graph."""

import pickle
from copy import deepcopy

from pylatro import add_joker, create_run_state, select_blind, start_blind
from pylatro.models import _FastStateCopy


def test_state_copy_matches_standard_deepcopy(monkeypatch):
    state = create_run_state("copy_graph", deck_key="b_blue")
    select_blind(state, "Small")
    start_blind(state, "Small")
    add_joker(state, "j_scholar")
    clone = deepcopy(state, {id(state.data): state.data})
    with monkeypatch.context() as patch:
        patch.delattr(_FastStateCopy, "__deepcopy__")
        reference = deepcopy(state, {id(state.data): state.data})
    assert pickle.dumps(clone) == pickle.dumps(reference)
    assert clone.data is state.data
    card = clone.hand_cards[0]
    assert any(card is item for item in clone.deck_cards)
    assert all(card is not item for item in state.deck_cards)
    card.perma_bonus += 100
    assert pickle.dumps(state) != pickle.dumps(clone)


def test_copy_preserves_cycles_aliases_and_custom_container_hooks():
    class CustomList(list):
        def __deepcopy__(self, memo):
            return ["custom hook"]

    state = create_run_state("copy_cycles")
    shared = [{"nested": [1, 2, 3]}]
    state.hands = {"first": shared, "second": shared, "custom": CustomList([0])}
    shared.append(state.hands)
    shared.append(state)
    clone = deepcopy(state, {id(state.data): state.data})
    assert clone.hands["first"] is clone.hands["second"]
    assert clone.hands["first"] is not shared
    assert clone.hands["first"][1] is clone.hands
    assert clone.hands["first"][2] is clone
    assert clone.hands["custom"] == ["custom hook"]
    clone.hands["first"][0]["nested"].append(4)
    assert shared[0]["nested"] == [1, 2, 3]


def test_copy_respects_caller_memo_for_containers():
    state = create_run_state("copy_memo")
    replacement = {"replacement": []}
    memo = {id(state.data): state.data, id(state.hands): replacement}
    clone = deepcopy(state, memo)
    assert clone.hands is replacement
    assert deepcopy(state, memo) is clone
