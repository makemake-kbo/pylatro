"""Exact scoring cache must follow mutable build inputs and remain bounded."""

from copy import deepcopy
from types import MappingProxyType
from unittest.mock import patch

import pytest

import pylatro_agent.build_value as scoring
from pylatro import add_joker, create_run_state, load_game_data
from pylatro_agent.shop_eval import capture_build_features


@pytest.fixture
def snapshot():
    state = create_run_state(seed="1701", data=load_game_data())
    add_joker(state, "j_joker")
    return capture_build_features(state)


def test_score_cache_hits_across_non_scoring_changes(snapshot):
    with patch.object(scoring, "_score_pass_uncached", wraps=scoring._score_pass_uncached) as calculate:
        expected = scoring.estimate_build_value(snapshot)
        count = calculate.call_count
        changed = deepcopy(snapshot)
        changed["dollars"] = 12345
        changed["round_score"] = 12345
        for detail in changed["hand_details"].values():
            detail["played_this_round"] = 12345
            detail["visible"] = not detail.get("visible", False)
        assert scoring.estimate_build_value(changed) == expected
        assert calculate.call_count == count
        changed["blind_target"] *= 2
        actual = scoring.estimate_build_value(changed)
        assert actual.representative_score_per_hand == expected.representative_score_per_hand
        assert actual.required_score_per_hand == expected.required_score_per_hand * 2
        assert calculate.call_count == count


@pytest.mark.parametrize("field", ["deck", "joker", "hand", "hands", "hand_size", "idol"])
def test_score_cache_matches_uncached_after_nested_mutation(snapshot, field):
    # Warm every public estimate, then mutate the original objects in place.
    scoring.estimate_build_value(snapshot)
    for hand in ("High Card", "Pair", "Flush"):
        scoring.estimate_hand_score(snapshot, hand)
    if field == "deck":
        snapshot["deck_stats"]["cards"][0]["enhancement"] = "Glass Card"
    elif field == "joker":
        snapshot["joker_details"][0]["mult"] = 40
    elif field == "hand":
        snapshot["hand_details"]["Pair"]["chips"] += 100
    elif field == "hands":
        snapshot["hands_available"] = 1
    elif field == "hand_size":
        snapshot["hand_size"] = 12
    else:
        snapshot["idol_card"] = {"rank": "A", "suit": "Spades"}
    actual = scoring.estimate_build_value(snapshot)
    actual_hands = [scoring.estimate_hand_score(snapshot, hand) for hand in ("High Card", "Pair", "Flush")]
    with patch.object(scoring, "_score_pass", scoring._score_pass_uncached):
        assert scoring.estimate_build_value(snapshot) == actual
        assert [scoring.estimate_hand_score(snapshot, hand) for hand in ("High Card", "Pair", "Flush")] == actual_hands


def test_score_cache_supports_unserializable_mapping(snapshot):
    snapshot["joker_details"] = tuple(MappingProxyType(joker) for joker in snapshot["joker_details"])
    actual = scoring.estimate_build_value(snapshot)
    with patch.object(scoring, "_score_pass", scoring._score_pass_uncached):
        assert scoring.estimate_build_value(snapshot) == actual


def test_score_cache_limits_entries_and_bytes(snapshot, monkeypatch):
    monkeypatch.setattr(scoring, "_SCORE_CACHE", scoring.OrderedDict())
    monkeypatch.setattr(scoring, "_SCORE_CACHE_BYTES", 0)
    monkeypatch.setattr(scoring, "_SCORE_CACHE_MAX_ENTRIES", 3)
    monkeypatch.setattr(scoring, "_SCORE_CACHE_MAX_BYTES", 20000)
    for chips in range(20):
        snapshot["hand_details"]["Pair"]["chips"] = chips
        actual = scoring.estimate_hand_score(snapshot, "Pair")
        with patch.object(scoring, "_score_pass", scoring._score_pass_uncached):
            assert scoring.estimate_hand_score(snapshot, "Pair") == actual
        assert len(scoring._SCORE_CACHE) <= 3
        assert scoring._SCORE_CACHE_BYTES == sum(map(len, scoring._SCORE_CACHE)) <= 20000


def _inspect_forked_cache(connection):
    acquired = scoring._SCORE_CACHE_LOCK.acquire(timeout=1)
    connection.send((len(scoring._SCORE_CACHE), scoring._SCORE_CACHE_BYTES, acquired))
    if acquired:
        scoring._SCORE_CACHE_LOCK.release()
    connection.close()


def test_fork_does_not_inherit_a_background_threads_cache_lock(snapshot):
    import multiprocessing as mp
    from threading import Event, Thread

    if "fork" not in mp.get_all_start_methods():
        pytest.skip("requires fork")
    scoring.estimate_build_value(snapshot)
    ready, release = Event(), Event()

    def hold_lock():
        with scoring._SCORE_CACHE_LOCK:
            ready.set()
            release.wait(timeout=10)

    thread = Thread(target=hold_lock)
    thread.start()
    parent, child = mp.get_context("fork").Pipe()
    process = mp.get_context("fork").Process(target=_inspect_forked_cache, args=(child,))
    try:
        assert ready.wait(timeout=2)
        process.start()
        child.close()
        assert parent.poll(timeout=3)
        assert parent.recv() == (0, 0, True)
    finally:
        release.set()
        thread.join(timeout=3)
        if process.pid is not None:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            process.close()
        parent.close()
        child.close()
