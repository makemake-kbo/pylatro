"""Boss blind selection and finisher (showdown) blind behavior."""

from __future__ import annotations

import pytest

from pylatro import load_game_data
from pylatro.pool import get_new_boss
from pylatro_cli.controller import GameController


@pytest.fixture(scope="module")
def data():
    return load_game_data()


def _state_at_ante(data, seed: int, ante: int, win_ante: int):
    controller = GameController(data=data)
    controller.new_run(str(seed))
    state = controller.state
    state.win_ante = win_ante
    state.round_resets.ante = ante
    return state


FINISHERS = {"bl_final_acorn", "bl_final_bell", "bl_final_heart", "bl_final_leaf", "bl_final_vessel"}


@pytest.mark.parametrize("boss", ["bl_water", "bl_needle", "bl_manacle", "bl_plant", "bl_final_bell", "bl_final_heart"])
@pytest.mark.parametrize("counter", ["j_chicot", "j_luchador"])
def test_disabling_boss_restores_resources_and_clears_debuffs(data, boss, counter):
    from pylatro import add_joker
    from pylatro.runtime import disable_blind, sell_joker

    controller = GameController(data=data)
    controller.new_run("boss_restore", deck_key="b_blue")
    state = controller.state
    state.round_resets.blind_choices["Boss"] = boss
    add_joker(state, counter)
    controller.select_blind("Boss")
    if counter == "j_luchador":
        sell_joker(state, 0)
    assert state.blind_disabled
    assert state.current_round.hands_left == state.round_resets.hands
    assert state.current_round.discards_left == state.round_resets.discards
    assert len(state.hand_cards) == state.starting_params.hand_size
    assert not any(c.debuff or c.face_down or c.forced_selection for c in state.deck_cards)
    assert not any(j.debuff for j in state.jokers)
    resources = (state.current_round.hands_left, state.current_round.discards_left, len(state.hand_cards))
    disable_blind(state)
    assert resources == (state.current_round.hands_left, state.current_round.discards_left, len(state.hand_cards))


@pytest.mark.parametrize("boss,enabled,disabled", [("bl_wall", 200000, 100000), ("bl_final_vessel", 300000, 100000)])
def test_luchador_sale_reduces_target_and_settles_already_beaten_blind(data, boss, enabled, disabled):
    from pylatro import add_joker
    from pylatro_agent.constants import ActionRange
    from pylatro_agent.heuristic import HeuristicAgent
    from pylatro_agent.training.fast_runner import FastRunner
    from pylatro_cli.controller import GamePhase

    runner = FastRunner(0, data, deck_key="b_blue", raise_errors=True)
    state = runner.state
    state.round_resets.ante = 8
    state.blind_on_deck = "Boss"
    state.round_resets.blind_choices["Boss"] = boss
    add_joker(state, "j_luchador")
    runner.step(ActionRange.BLIND_PLAY)
    assert runner._ctrl.blind_target() == enabled
    runner._ctrl.round_score = runner._round_score = disabled + 1
    agent = HeuristicAgent()
    action = agent.select_action(state, runner.sub_phase, runner.compute_mask(), round_score=runner.round_score)
    assert action == ActionRange.SHOP_SELL_JOKER_START
    runner.step(action)
    assert runner.done and runner.won
    assert runner.phase == GamePhase.GAME_WON


def test_disabled_boss_target_agrees_in_policy_observations_and_candidates(data):
    from pylatro import add_joker
    from pylatro_agent.hand_candidates import _blind_target as candidate_target
    from pylatro_agent.heuristic import HeuristicAgent
    from pylatro_agent.shop_eval import _upcoming_blind_target
    from pylatro_agent.tokenizer import Tokenizer
    from pylatro_agent.training.fast_runner import _blind_target as fast_target
    from pylatro_agent.vocab import build_vocab

    controller = GameController(data=data)
    controller.new_run("chicot_target", deck_key="b_blue")
    state = controller.state
    state.round_resets.ante = 8
    state.round_resets.blind_choices["Boss"] = "bl_final_vessel"
    add_joker(state, "j_chicot")
    controller.select_blind("Boss")
    tokenizer = Tokenizer(vocab=build_vocab(data))
    assert controller.blind_target() == 100000
    assert tokenizer._blind_target(state) == 100000
    assert candidate_target(state) == 100000
    assert fast_target(state) == 100000
    assert HeuristicAgent()._get_blind_target(state) == 100000
    assert _upcoming_blind_target(state) == 100000


def test_hook_discards_only_held_cards_and_runs_discard_effects(data) -> None:
    from pylatro import add_joker, create_run_state
    from pylatro.flow import play_cards
    from pylatro.models import PlayingCard

    state = create_run_state("hook_selected_cards", deck_key="b_blue", data=data)
    state.round_resets.blind = data.blinds["bl_hook"]
    state.blind_on_deck = "Boss"
    state.hand_cards = [
        PlayingCard(front_key=f"S_{rank}", suit="Spades", rank=rank)
        for rank in ("3", "5", "7", "9", "K")
    ] + [
        PlayingCard(front_key="H_2", suit="Hearts", rank="2"),
        PlayingCard(front_key="C_2", suit="Clubs", rank="2"),
    ]
    state.deck_cards = list(state.hand_cards)
    state.draw_pile = []
    state.current_round.mail_card = {"rank": "2", "id": 2}
    add_joker(state, "j_mail")
    dollars = state.dollars
    discards = state.current_round.discards_left
    result = play_cards(state, [0, 1, 2, 3, 4])
    assert result.score.hand_name == "Flush"
    assert len(result.played) == 5
    assert state.dollars == dollars + 10
    assert state.current_round.discards_left == discards
    assert state.current_round.discards_used == 0


@pytest.mark.parametrize(
    ("voucher", "rerolled", "dollars", "credit", "allowed"),
    [
        (None, False, 25, 0, False),
        ("v_directors_cut", False, 10, 0, True),
        ("v_directors_cut", True, 25, 0, False),
        ("v_retcon", True, 25, 0, True),
        ("v_retcon", False, 9, 0, False),
        ("v_directors_cut", False, 0, -20, True),
    ],
)
def test_paid_boss_reroll_rules_match_both_action_masks(data, voucher, rerolled, dollars, credit, allowed):
    from pylatro.blind import reroll_boss
    from pylatro_agent.constants import ActionRange, SubPhase
    from pylatro_agent.masks import compute_action_mask
    from pylatro_agent.training.fast_runner import FastRunner

    runner = FastRunner(0, data, deck_key="b_blue", raise_errors=True)
    state = runner.state
    state.blind_on_deck = "Boss"
    state.dollars = dollars
    state.bankrupt_at = credit
    state.round_resets.boss_rerolled = rerolled
    if voucher:
        state.used_vouchers[voucher] = True
    assert bool(runner.compute_mask()[ActionRange.BLIND_REROLL]) == allowed
    assert bool(compute_action_mask(state, SubPhase.BLIND_SELECT)[ActionRange.BLIND_REROLL]) == allowed
    if allowed:
        reroll_boss(state)
        assert state.dollars == dollars - 10
    else:
        with pytest.raises(ValueError, match="voucher"):
            reroll_boss(state)
        assert state.dollars == dollars


def test_boss_tag_reroll_is_free_without_a_voucher(data):
    from pylatro import create_run_state
    from pylatro.blind import reroll_boss

    state = create_run_state("free_boss_tag", data=data)
    state.dollars = 0
    reroll_boss(state, from_tag=True)
    assert state.dollars == 0
    assert state.round_resets.boss_rerolled


@pytest.mark.parametrize(("dollars", "expected"), [(7, 14), (25, 50), (60, 100), (-5, -5)])
def test_economy_tag_pays_immediately_when_skipping(data, dollars, expected):
    from pylatro import create_run_state, skip_blind

    state = create_run_state("economy_tag", data=data)
    state.blind_on_deck = "Big"
    state.round_resets.blind_tags["Big"] = "tag_economy"
    state.dollars = dollars
    assert skip_blind(state) == "Boss"
    assert state.dollars == expected
    assert "tag_economy" not in state.tags


def test_no_finishers_before_ante_8_with_lowered_win_ante(data) -> None:
    """A curriculum win_ante (e.g. 5) must not pull showdown bosses to ante 5."""
    for ante, win_ante in [(5, 5), (4, 4), (5, 8), (7, 8)]:
        seen = {get_new_boss(_state_at_ante(data, seed, ante, win_ante)) for seed in range(25)}
        assert not (seen & FINISHERS), f"finisher boss at ante {ante} (win_ante={win_ante}): {seen & FINISHERS}"


def test_only_finishers_at_ante_8_regardless_of_win_ante(data) -> None:
    for win_ante in (5, 8):
        seen = {get_new_boss(_state_at_ante(data, seed, 8, win_ante)) for seed in range(40)}
        assert seen <= FINISHERS, f"non-finisher at ante 8 (win_ante={win_ante}): {seen - FINISHERS}"
        assert seen == FINISHERS, f"finisher pool incomplete at ante 8: missing {FINISHERS - seen}"


def test_finishers_at_ante_16(data) -> None:
    seen = {get_new_boss(_state_at_ante(data, seed, 16, 8)) for seed in range(40)}
    assert seen == FINISHERS


def test_amber_acorn_does_not_debuff_jokers(data) -> None:
    """Upstream flips/shuffles jokers; they keep scoring. Debuffing them is a bug."""
    from pylatro.flow import _reset_for_blind  # type: ignore[attr-defined]

    controller = GameController(data=data)
    controller.new_run("42")
    state = controller.state
    from pylatro.runtime import add_joker

    for key in ("j_joker", "j_greedy_joker", "j_lusty_joker"):
        add_joker(state, key)
    state.round_resets.blind = dict(data.blinds["bl_final_acorn"])

    _reset_for_blind(state, "Boss")
    assert all(not joker.debuff for joker in state.jokers), "Amber Acorn must not debuff jokers"


def test_verdant_leaf_sell_joker_disables_blind(data) -> None:
    """Selling any joker under Verdant Leaf disables the blind and lifts card debuffs."""
    from pylatro.runtime import add_joker, sell_joker

    controller = GameController(data=data)
    controller.new_run("42")
    state = controller.state
    add_joker(state, "j_joker")
    state.round_resets.blind = dict(data.blinds["bl_final_leaf"])
    for card in state.deck_cards:
        card.debuff = True

    sell_joker(state, 0)

    assert state.blind_disabled
    assert all(not card.debuff for card in state.deck_cards)


def test_blind_disable_lasts_through_settlement_then_clears_at_cashout(data) -> None:
    controller = GameController(data=data)
    controller.new_run("disabled_cashout")
    state = controller.state
    state.blind_on_deck = "Boss"
    state.round_resets.blind_choices["Boss"] = "bl_goad"
    controller.select_blind("Boss")
    state.blind_disabled = True

    # The disabled current boss remains disabled until its atomic cash-out.
    assert state.blind_disabled
    controller.cash_out()

    assert state.round_resets.ante == 2
    assert not state.blind_disabled


def _debuff_state(data, blind_key: str):
    controller = GameController(data=data)
    controller.new_run("42")
    state = controller.state
    state.round_resets.blind = dict(data.blinds[blind_key])
    return state


def test_the_eye_debuffs_repeat_hand_types(data) -> None:
    """Upstream gates The Eye inside `if self.debuff` with an EMPTY debuff
    table — truthy in Lua, falsy once translated to Python — which silently
    disabled the boss. It must zero any hand type already played this round."""
    from pylatro.flow import _debuff_hand  # type: ignore[attr-defined]
    from pylatro.scoring import get_poker_hand_info

    state = _debuff_state(data, "bl_eye")
    state.eye_hands = {}
    pair = [c for c in state.deck_cards if c.rank == "A"][:2]
    hand_name, _, poker_hands, _ = get_poker_hand_info(state, pair)
    assert hand_name == "Pair"

    # check=True probes must not record the hand type
    assert not _debuff_hand(state, pair, hand_name, poker_hands, check=True)
    assert not _debuff_hand(state, pair, hand_name, poker_hands, check=True)
    # first real play of a type is allowed and records it; repeats are zeroed
    assert not _debuff_hand(state, pair, hand_name, poker_hands)
    assert _debuff_hand(state, pair, hand_name, poker_hands)


def test_the_mouth_debuffs_second_hand_type(data) -> None:
    """The Mouth (same empty-debuff gating bug as The Eye) must zero every
    hand type except the first one played this round."""
    from pylatro.flow import _debuff_hand  # type: ignore[attr-defined]
    from pylatro.scoring import get_poker_hand_info

    state = _debuff_state(data, "bl_mouth")
    state.mouth_only_hand = False
    pair = [c for c in state.deck_cards if c.rank == "A"][:2]
    high = [next(c for c in state.deck_cards if c.rank == "9")]
    pair_name, _, pair_hands, _ = get_poker_hand_info(state, pair)
    high_name, _, high_hands, _ = get_poker_hand_info(state, high)
    assert (pair_name, high_name) == ("Pair", "High Card")

    assert not _debuff_hand(state, pair, pair_name, pair_hands)  # locks "Pair"
    assert _debuff_hand(state, high, high_name, high_hands)  # other types zeroed
    assert not _debuff_hand(state, pair, pair_name, pair_hands)  # Pair still fine


def test_hook_blind_triggered_survives_debuff_check(data) -> None:
    """_press_play sets blind_triggered for The Hook before _debuff_hand runs;
    the debuff check must not wipe it (Matador reads it during scoring)."""
    from pylatro.flow import _debuff_hand  # type: ignore[attr-defined]
    from pylatro.scoring import get_poker_hand_info

    state = _debuff_state(data, "bl_hook")
    state.blind_triggered = True  # as set by _press_play this play
    cards = [c for c in state.deck_cards if c.rank == "A"][:2]
    hand_name, _, poker_hands, _ = get_poker_hand_info(state, cards)

    assert not _debuff_hand(state, cards, hand_name, poker_hands)
    assert state.blind_triggered, "empty-debuff boss must keep _press_play's trigger"
