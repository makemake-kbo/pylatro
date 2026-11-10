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


def test_generated_copiers_cover_every_state_dataclass_without_aliasing():
    """Each dataclass gets a straight-line copier; no mutable field may be shared."""
    state = create_run_state("copy_fields", deck_key="b_blue")
    select_blind(state, "Small")
    start_blind(state, "Small")
    add_joker(state, "j_ice_cream")
    clone = deepcopy(state, {id(state.data): state.data})
    pairs = [(state, clone)]
    seen = set()
    while pairs:
        source, copied = pairs.pop()
        if id(source) in seen or not hasattr(source, "__dataclass_fields__"):
            continue
        seen.add(id(source))
        assert type(source) is type(copied)
        for name in source.__dataclass_fields__:
            value, twin = getattr(source, name), getattr(copied, name)
            if name == "data":
                assert twin is value
                continue
            if isinstance(value, (dict, list, set)):
                assert twin is not value, name
            if hasattr(value, "__dataclass_fields__"):
                pairs.append((value, twin))
            elif isinstance(value, list):
                pairs.extend((a, b) for a, b in zip(value, twin, strict=True))
    assert len(seen) > 40


def test_pseudorandom_copy_is_independent():
    state = create_run_state("copy_rng")
    clone = deepcopy(state, {id(state.data): state.data})
    expected = state.pseudorandom.pseudoseed("probe")
    assert clone.pseudorandom.pseudoseed("probe") == expected
    assert clone.pseudorandom.values is not state.pseudorandom.values
    assert state.pseudorandom.pseudoseed("probe") != expected


def test_has_joker_ignores_debuffed_expired_and_matches_by_name():
    state = create_run_state("has_joker")
    assert not state.has_joker("Pareidolia")
    add_joker(state, "j_pareidolia")
    add_joker(state, "j_joker")
    assert state.has_joker("Pareidolia")
    assert state.has_joker("Joker")
    state.jokers[0].debuff = True
    assert not state.has_joker("Pareidolia")
    state.jokers[0].debuff = False
    state.jokers[0].perishable = True
    state.jokers[0].perish_tally = 0
    assert not state.has_joker("Pareidolia")
    state.jokers[0].perish_tally = 2
    assert state.has_joker("Pareidolia")
    # Selling one joker and buying another keeps the same list length.
    state.joker_keys[0] = "j_half"
    state.jokers[0].center_key = "j_half"
    assert not state.has_joker("Pareidolia")
    assert state.has_joker("Half Joker")
    # Keys without an instance (mid-update) still count as active.
    state.joker_keys.append("j_splash")
    assert state.has_joker("Splash")
