from __future__ import annotations

import pickle
import time
from copy import deepcopy
from dataclasses import asdict, fields

import numpy as np
import pytest

from pylatro import add_joker, create_run_state, load_game_data, start_blind
from pylatro.pool import create_card_spec
from pylatro.rng import PseudorandomState
from pylatro_agent.constants import ActionRange as AR
from pylatro_agent.constants import SubPhase
from pylatro_agent.hybrid import (
    HybridPolicy,
    PendingPurchase,
    Plan,
    PlannerConfig,
    PlanningDeadline,
    bounded,
    shop_context,
)
from pylatro_agent.public_runtime import PublicHeuristic, PublicRunner, public_joker_order
from pylatro_agent.public_state import PublicGameSnapshot, canonical, observation, public_mask, simulation_state
from pylatro_agent.subset_actions import subset_index, subset_indices
from pylatro_agent.tokenizer import Tokenizer
from pylatro_agent.vocab import build_vocab


def hand_state(boss="bl_hook"):
    state = create_run_state("PUBLIC_TEST", deck_key="b_blue")
    state.round_resets.blind_choices["Boss"] = boss
    start_blind(state, "Boss")
    cards = {c.front_key: c for c in state.deck_cards}
    state.hand_cards = [cards[k] for k in ("C_2", "C_3", "C_4", "C_5", "C_6", "D_9", "H_K", "S_A")]
    state.draw_pile = [c for c in state.deck_cards if c not in state.hand_cards]
    state.discard_pile = []
    state.current_round.hands_left = 1
    state.current_round.discards_left = 0
    return state


def snapshot(state, phase=SubPhase.CHOOSE_ACTION):
    return PublicGameSnapshot.capture(state, phase)


def private_mutation(state):
    clone = deepcopy(state, {id(state.data): state.data})
    clone.seed = "PRIVATE_DIFFERENT"
    clone.subhash = "PRIVATE_HASH"
    clone.pseudorandom = PseudorandomState("OTHER_RNG")
    clone.draw_pile.reverse()
    clone.deck_cards.reverse()
    for c in clone.deck_cards:
        c.reward_uid += 100001
    return clone


def assert_observations_equal(left, right, data):
    tokenizer = Tokenizer(build_vocab(data))
    a, b = observation(left, data, tokenizer), observation(right, data, tokenizer)
    for f in fields(a):
        np.testing.assert_array_equal(getattr(a, f.name), getattr(b, f.name), err_msg=f.name)


def test_private_rng_draw_order_ids_do_not_reach_neural_inputs_or_decisions():
    state = hand_state()
    clone = private_mutation(state)
    a, b = snapshot(state), snapshot(clone)
    assert a == b
    assert "PRIVATE_" not in a.payload.decode()
    assert_observations_equal(a, b, state.data)
    weak = int(AR.PLAY_SUBSET_START + subset_index((0,)))
    left = HybridPolicy(lambda s: weak, state.data).decide(a)
    right = HybridPolicy(lambda s: weak, state.data).decide(b)
    assert (left.action, left.diagnostics["values"]) == (right.action, right.diagnostics["values"])


def test_concealed_assignment_does_not_change_tokens_candidates_masks_risk_or_order():
    state = hand_state()
    state.hand_cards[0].face_down = True
    add_joker(state, "j_cavendish")
    add_joker(state, "j_joker")
    clone = private_mutation(state)
    hidden = clone.hand_cards[0]
    other = clone.draw_pile[0]
    hidden.face_down = False
    other.face_down = True
    clone.hand_cards[0], clone.draw_pile[0] = other, hidden
    a, b = snapshot(state), snapshot(clone)
    assert a == b
    assert_observations_equal(a, b, state.data)
    assert public_joker_order(a, state.data, (0, 1)) == public_joker_order(b, state.data, (0, 1))


def test_acorn_hides_shuffled_joker_order_in_observations_history_and_execution():
    from pylatro_agent.history import PlayHistoryTracker, blind_history_key

    state = hand_state("bl_final_acorn")
    for key in ("j_cavendish", "j_joker", "j_blueprint"):
        add_joker(state, key)
    state.jokers[0].eternal = True
    clone = private_mutation(state)
    clone.jokers.reverse()
    clone.joker_keys = [j.center_key for j in clone.jokers]
    snapshots = []
    for world in (state, clone):
        history = PlayHistoryTracker()
        history.start_round(blind_history_key(world))
        pending = history.capture(world, world.hand_cards[:1], blind_target=600, round_score=0)
        history.finalize(pending, hand_type="High Card", score=12)
        snapshots.append(PublicGameSnapshot.capture(world, SubPhase.CHOOSE_ACTION, history=history))
    a, b = snapshots
    assert a == b and a.values["joker_order_unknown"]
    assert_observations_equal(a, b, state.data)
    assert public_joker_order(a, state.data, (0,)).order is None
    mask = public_mask(simulation_state(a, state.data), SubPhase.CHOOSE_ACTION)
    assert not mask[AR.SHOP_SELL_JOKER_START : AR.SHOP_SELL_JOKER_END + 1].any()
    orders = {tuple(j.center_key for j in simulation_state(a, state.data, i).jokers) for i in range(16)}
    assert len(orders) > 1
    runner = PublicRunner.from_snapshot(a, state.data)
    before = tuple(j.center_key for j in runner.state.jokers)
    runner._order_jokers((0,))
    assert tuple(j.center_key for j in runner.state.jokers) == before
    state.blind_disabled = True
    assert not snapshot(state).values["joker_order_unknown"]


def test_concealed_deck_token_scrubbing_respects_tensor_truncation():
    from pylatro.models import PlayingCard

    state = hand_state()
    for i in range(150):
        card = PlayingCard("S_2", "Spades", "2", reward_uid=10000 + i)
        state.deck_cards.append(card)
        state.draw_pile.append(card)
    state.discard_pile = [state.draw_pile.pop()]
    state.discard_pile[0].face_down = True
    obs = observation(snapshot(state), state.data, Tokenizer(build_vocab(state.data)))
    assert obs.tokens.shape == (160, 47)


def test_oversized_hands_bound_proposals_without_removing_held_cards():
    from pylatro_agent.public_runtime import addressable_hand

    state = hand_state()
    cards = {c.front_key: c for c in state.deck_cards}
    keys = [f"{s}_{r}" for r in ("2", "3", "4") for s in ("C", "D", "H", "S")]
    keys += ["D_K", "H_K", "S_K", "C_5", "C_6"]
    state.hand_cards = [cards[k] for k in keys]
    state.draw_pile = [c for c in state.deck_cards if c not in state.hand_cards]
    state.hand_cards[-1].center_key = "m_steel"
    public = snapshot(state)
    mask = public_mask(simulation_state(public, state.data), SubPhase.CHOOSE_ACTION)
    action = PublicHeuristic(state.data)(public)
    assert mask[action]
    assert max(subset_indices(action - AR.PLAY_SUBSET_START)) < 16
    plans = HybridPolicy(lambda s: action, state.data)._hand_plans(public, action)
    assert all(mask[p.actions[0]] for p in plans)
    runner = PublicRunner.from_snapshot(public, state.data)
    original = runner.state.hand_cards
    with pytest.raises(RuntimeError, match="test interruption"), addressable_hand(runner.state):
        assert len(runner.state.hand_cards) == 16
        raise RuntimeError("test interruption")
    assert runner.state.hand_cards is original and len(original) == 17
    overflow = original[-1]
    runner.step(action)
    assert overflow in runner.state.deck_cards and overflow.center_key == "m_steel"
    assert not overflow.played_this_ante
    assert snapshot(state) == public
    assert_observations_equal(public, snapshot(private_mutation(state)), state.data)


def test_concealed_overflow_aura_targets_do_not_overrun_action_vocabulary():
    from pylatro import add_consumable

    state = hand_state()
    while len(state.hand_cards) < 18:
        state.hand_cards.append(state.draw_pile.pop())
    state.hand_cards[-1].face_down = True
    add_consumable(state, "c_aura")
    public = snapshot(state)
    assert public_mask(simulation_state(public, state.data), SubPhase.CHOOSE_ACTION).any()
    assert_observations_equal(public, snapshot(private_mutation(state)), state.data)


def test_unopened_pack_and_future_voucher_are_not_in_snapshot():
    state = hand_state()
    a = snapshot(state)
    state.current_voucher = "v_antimatter"
    from pylatro.models import PackState

    state.pack = PackState("p_arcana_normal_1", "TAROT_PACK", 1, [])
    assert snapshot(state) == a


def test_unowned_joker_targets_are_private():
    state = hand_state()
    before = snapshot(state)
    state.current_round.idol_card = {"suit": "Hearts", "rank": "2", "id": 2}
    state.current_round.ancient_card = {"suit": "Diamonds"}
    assert snapshot(state) == before
    add_joker(state, "j_idol")
    assert snapshot(state).values["current_round"]["idol_card"]["rank"] == "2"


def test_scoring_agrees_without_random_effects_and_samples_lucky_effects():
    from pylatro_agent.joker_layout import _score_order

    state = hand_state()
    exact = _score_order(state, (0, 1), ())
    assert _score_order(simulation_state(snapshot(state), state.data), (0, 1), ()) == exact
    state.hand_cards[0].center_key = "m_lucky"
    s = snapshot(state)
    scores = {_score_order(simulation_state(s, state.data, i), (0,), ()) for i in range(40)}
    assert len(scores) > 1


def test_useful_discard_is_considered_with_structural_plays():
    state = hand_state("bl_head")
    six = state.hand_cards.pop(4)
    state.deck_cards = [*state.hand_cards, six]
    state.draw_pile = [six]
    state.current_round.discards_left = 1
    weak = int(AR.PLAY_SUBSET_START + subset_index((0,)))
    policy = HybridPolicy(lambda s: weak, state.data)
    plans = policy._hand_plans(snapshot(state), weak)
    assert any(AR.PLAY_SUBSET_START <= p.actions[0] <= AR.PLAY_SUBSET_END for p in plans)
    assert any(AR.DISCARD_SUBSET_START <= p.actions[0] <= AR.DISCARD_SUBSET_END for p in plans)
    decision = policy.decide(snapshot(state))
    assert AR.DISCARD_SUBSET_START <= decision.action <= AR.DISCARD_SUBSET_END


def test_public_history_scrubs_concealed_card_properties():
    from pylatro_agent.history import PlayHistoryTracker

    state = hand_state()
    state.hand_cards[0].face_down = True
    history = PlayHistoryTracker()
    pending = history.capture(state, state.hand_cards[:1], blind_target=600, round_score=0)
    history.finalize(pending, hand_type="High Card", score=12)
    result = PublicGameSnapshot.capture(state, SubPhase.CHOOSE_ACTION, history=history)
    card = result.values["history"][-1]["events"][0]["cards"][0]
    assert card["rank"] == card["suit"] == card["front_key"] == ""


def test_pack_contents_are_sampled_and_claim_targets_use_public_runtime():
    state = shop_state()
    state.shop.boosters = [create_card_spec(state, "Booster", forced_key="p_arcana_normal_1")]
    runner = PublicRunner.from_snapshot(snapshot(state, SubPhase.SHOP), state.data)
    runner.step(AR.SHOP_BUY_START + len(state.shop.cards))
    assert runner.sub_phase == SubPhase.BOOSTER_PACK
    assert runner.state.pack is not None
    public = runner.snapshot()
    assert public.values["pack"]["cards"]
    runner.step(AR.PACK_SKIP)
    assert runner.sub_phase == SubPhase.SHOP


def test_worlds_sample_without_replacement_and_never_share_live_objects():
    state = hand_state()
    state.hand_cards[0].face_down = True
    s = snapshot(state)
    worlds = [simulation_state(s, state.data, i) for i in range(4)]
    assert len({tuple(c.front_key for c in w.draw_pile) for w in worlds}) == 4
    for w in worlds:
        assert len({c.reward_uid for c in w.hand_cards + w.draw_pile}) == 52
        assert sorted(c.front_key for c in w.deck_cards) == sorted(c.front_key for c in state.deck_cards)
        assert w.pseudorandom is not state.pseudorandom
    assert pickle.dumps(worlds[0]) == pickle.dumps(simulation_state(s, state.data, 0))


def test_last_hand_rescue_live_isolation_and_four_equal_samples():
    state = hand_state()
    before = pickle.dumps(state)
    weak = int(AR.PLAY_SUBSET_START + subset_index((0,)))
    decision = HybridPolicy(lambda s: weak, state.data).decide(snapshot(state))
    assert decision.action != weak
    assert decision.diagnostics["samples"] == 4
    runner = PublicRunner.from_snapshot(snapshot(state), state.data)
    runner.step(decision.action)
    assert runner.sub_phase == SubPhase.SHOP and not runner.done
    assert pickle.dumps(state) == before


@pytest.mark.parametrize("boss", ["bl_eye", "bl_mouth", "bl_psychic"])
def test_concealed_masks_respect_visible_size_and_forced_cards_only(boss):
    state = hand_state(boss)
    state.hand_cards[0].face_down = True
    state.hand_cards[0].forced_selection = True
    state.eye_hands = {"High Card": True}
    state.mouth_only_hand = "Pair"
    mask = public_mask(simulation_state(snapshot(state), state.data), SubPhase.CHOOSE_ACTION)
    for aid in np.flatnonzero(mask[AR.PLAY_SUBSET_START : AR.PLAY_SUBSET_END + 1]):
        positions = subset_indices(int(aid))
        assert 0 in positions
        if boss == "bl_psychic":
            assert len(positions) == 5
    if boss != "bl_psychic":
        assert mask[AR.PLAY_SUBSET_START + subset_index((0,))]


def test_deadline_in_generation_retains_original_proposal_and_rng():
    state = hand_state()
    weak = int(AR.PLAY_SUBSET_START + subset_index((0,)))
    calls = []
    policy = HybridPolicy(lambda s: calls.append(s) or weak, state.data, PlannerConfig(seconds=0))
    before = np.random.get_state()
    decision = policy.decide(snapshot(state))
    assert decision.action == weak and len(calls) == 1
    assert decision.diagnostics["fallback"] == "deadline"
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])


def test_deadline_interrupts_and_restores_signal_handler():
    import signal

    original = signal.getsignal(signal.SIGALRM)
    with pytest.raises(PlanningDeadline), bounded(0.01):
        time.sleep(0.1)
    assert signal.getsignal(signal.SIGALRM) == original


def test_incomplete_pair_round_is_not_compared_and_ties_keep_model(monkeypatch):
    state = hand_state()
    weak = int(AR.PLAY_SUBSET_START + subset_index((0,)))
    strong = int(AR.PLAY_SUBSET_START + subset_index((0, 1, 2, 3, 4)))
    policy = HybridPolicy(lambda s: weak, state.data)
    monkeypatch.setattr(policy, "_hand_plans", lambda s, p: [Plan((weak,)), Plan((strong,))])

    def trial(s, p, sample, hand):
        if sample == 1 and p.actions == (strong,):
            raise PlanningDeadline
        return ((sample, 0, 0, 0), True)

    monkeypatch.setattr(policy, "_trial", trial)
    result = policy.decide(snapshot(state))
    assert result.action == weak
    assert result.diagnostics["samples"] == 1
    assert result.diagnostics["values"][0] == result.diagnostics["values"][1]


def test_disabled_assistance_never_calls_search_and_calls_base_once(monkeypatch):
    state = hand_state()
    calls = []
    weak = int(AR.PLAY_SUBSET_START + subset_index((0,)))
    policy = HybridPolicy(lambda s: calls.append(s) or weak, state.data, PlannerConfig(mode="off"))
    monkeypatch.setattr(policy, "_hand_plans", lambda *a: pytest.fail("Search called"))
    assert policy.decide(snapshot(state)).action == weak
    assert len(calls) == 1


def shop_state():
    state = create_run_state("PUBLIC_SHOP", deck_key="b_blue")
    state.dollars = 20
    state.shop.cards = [create_card_spec(state, "Joker", forced_key="j_banner")]
    return state


def test_pending_purchase_revalidates_price_offer_phase_and_money():
    for change in ("none", "price", "offer", "money", "phase"):
        state = shop_state()
        policy = HybridPolicy(
            lambda s: int(AR.SHOP_LEAVE if s.sub_phase == SubPhase.SHOP else AR.BLIND_PLAY),
            state.data,
            PlannerConfig(mode="off"),
        )
        policy.pending = PendingPurchase(0, canonical(asdict(state.shop.cards[0])), shop_context(state))
        if change == "price":
            state.shop.cards[0].cost += 1
        if change == "offer":
            state.shop.cards[0].center_key = "j_egg"
        if change == "money":
            state.dollars = 0
        phase = SubPhase.BLIND_SELECT if change == "phase" else SubPhase.SHOP
        result = policy.decide(snapshot(state, phase))
        assert (result.action == AR.SHOP_BUY_START) == (change == "none")
        assert result.diagnostics["pending_cancelled"] == (change != "none")


def test_negative_eternal_capacity_and_sale_side_effects():
    state = shop_state()
    for key in ("j_joker", "j_half", "j_egg", "j_golden", "j_campfire"):
        add_joker(state, key)
    add_joker(state, "j_banner", edition={"negative": True})
    state.jokers[0].eternal = True
    s = snapshot(state, SubPhase.SHOP)
    runner = PublicRunner.from_snapshot(s, state.data)
    assert not runner.compute_mask()[AR.SHOP_SELL_JOKER_START]
    assert not runner.compute_mask()[AR.SHOP_BUY_START]
    runner.step(AR.SHOP_SELL_JOKER_START + 5)
    assert not runner.compute_mask()[AR.SHOP_BUY_START]
    runner.step(AR.SHOP_SELL_JOKER_START + 2)
    assert runner.compute_mask()[AR.SHOP_BUY_START]
    campfire = next(j for j in runner.state.jokers if j.center_key == "j_campfire")
    assert campfire.x_mult > 1
    runner.step(AR.SHOP_BUY_START)
    assert runner.state.jokers[-1].center_key == "j_banner"


def test_continuation_takes_snapshots_and_search_cache_invalidates():
    state = hand_state()
    s = snapshot(state)
    action = int(AR.PLAY_SUBSET_START + subset_index((0,)))
    policy = HybridPolicy(lambda s: action, state.data)
    first = policy._trial(s, Plan((action,)), 0, True)
    size = len(policy.cache)
    assert policy._trial(s, Plan((action,)), 0, True) == first
    assert len(policy.cache) == size
    state.dollars += 1
    policy._trial(snapshot(state), Plan((action,)), 0, True)
    assert len(policy.cache) == size + 1
    assert PublicHeuristic(state.data)(s) >= 0


def test_exhaustion_is_a_terminal_trial():
    state = hand_state()
    state.hand_cards = state.hand_cards[:1]
    state.deck_cards = list(state.hand_cards)
    state.draw_pile = []
    state.current_round.hands_left = 2
    s = snapshot(state)
    action = int(AR.PLAY_SUBSET_START + subset_index((0,)))
    result, complete = HybridPolicy(lambda s: action, state.data)._trial(s, Plan((action,)), 0, True)
    assert complete and result[0] == 0


def test_compiled_fast_runner_dispatches_public_ordering(monkeypatch):
    import pylatro_agent.public_runtime as runtime
    import pylatro_agent.training.fast_runner as fast

    assert fast.__file__.endswith(".so")
    runner = PublicRunner.from_snapshot(snapshot(hand_state()), load_game_data())
    called = []
    monkeypatch.setattr(runtime, "public_joker_order", lambda *a: called.append(a[0]) or runtime.NO_ORDER_DECISION)
    runner.step(AR.PLAY_SUBSET_START + subset_index((0,)))
    assert len(called) == 1 and isinstance(called[0], PublicGameSnapshot)


def test_selected_replacement_executes_sale_then_revalidated_buy(monkeypatch):
    state = shop_state()
    add_joker(state, "j_egg")
    policy = HybridPolicy(lambda s: int(AR.SHOP_LEAVE), state.data)
    sale = int(AR.SHOP_SELL_JOKER_START)
    buy = int(AR.SHOP_BUY_START)
    monkeypatch.setattr(policy, "_shop_plans", lambda *a: [Plan((int(AR.SHOP_LEAVE),)), Plan((sale, buy))])
    monkeypatch.setattr(policy, "_trial", lambda s, p, i, h: ((int(len(p.actions) == 2), 0), True))
    runner = PublicRunner.from_snapshot(snapshot(state, SubPhase.SHOP), state.data)
    first = policy.decide(runner.snapshot())
    assert first.action == sale and first.pending_purchase is not None
    runner.step(first.action)
    second = policy.decide(runner.snapshot())
    assert second.action == buy and second.diagnostics["pending_executed"]
    runner.step(second.action)
    assert [j.center_key for j in runner.state.jokers] == ["j_banner"]


def test_rollout_continuation_cannot_receive_simulated_future_state():
    state = hand_state()
    state.current_round.hands_left = 2
    weak = int(AR.PLAY_SUBSET_START + subset_index((0,)))
    policy = HybridPolicy(lambda s: weak, state.data)
    original = policy.continuation
    received = []

    def continuation(public):
        assert isinstance(public, PublicGameSnapshot)
        assert "seed" not in public.values and "pseudorandom" not in public.values
        assert "draw_pile" not in public.values
        received.append(public)
        return original(public)

    policy.continuation = continuation
    policy._trial(snapshot(state), Plan((weak,)), 0, True)
    assert received


@pytest.mark.parametrize("sample", [False, True])
def test_disabled_neural_matches_base_with_independent_policy_rng(sample):
    from pathlib import Path

    import torch

    from pylatro_agent.public_runtime import PublicNeuralPolicy

    checkpoint = Path(__file__).resolve().parents[1] / "incumbents/ppo_replica0/checkpoint.pt"
    if not checkpoint.exists():
        pytest.skip("Incumbent checkpoint is not installed")
    state = hand_state()
    public = snapshot(state)
    base = PublicNeuralPolicy(checkpoint, state.data, sample=sample, policy_seed=712)
    global_before = torch.get_rng_state().clone()
    expected = base(public)
    advanced = base.rng_state.clone()
    base.reset_rng(712)
    actual = HybridPolicy(base, state.data, PlannerConfig(mode="off")).decide(public)
    assert actual.action == actual.proposal == expected
    assert torch.equal(base.rng_state, advanced)
    assert torch.equal(torch.get_rng_state(), global_before)
