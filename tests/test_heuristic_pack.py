import pickle

from pylatro import add_joker, create_run_state
from pylatro.models import PackState
from pylatro.pool import create_card_spec
from pylatro.shop import claim_pack_card
from pylatro_agent.constants import ActionRange as AR, SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.heuristic_pack import select_black_hole_pack
from pylatro_agent.heuristic_shop_search import ShopSearch
from pylatro_agent.masks import compute_action_mask


def make_pack(state):
    state.pack = PackState(
        "p_celestial_normal_1", "CELESTIAL_PACK", 1,
        cards=[create_card_spec(state, "Planet", forced_key="c_mercury"),
               create_card_spec(state, "Spectral", forced_key="c_black_hole")],
    )


def test_black_hole_is_claimed_and_levels_every_hand_without_mutating_probe():
    state = create_run_state("black_hole", deck_key="b_blue")
    add_joker(state, "j_half")
    add_joker(state, "j_blue_joker")
    make_pack(state)
    before = pickle.dumps(state)
    action = select_black_hole_pack(
        state, compute_action_mask(state, SubPhase.BOOSTER_PACK), HeuristicAgent(), ShopSearch(),
    )
    assert action == AR.PACK_CLAIM_START + 1
    assert pickle.dumps(state) == before
    levels = {name: hand["level"] for name, hand in state.hands.items()}
    claim_pack_card(state, action - AR.PACK_CLAIM_START)
    assert all(hand["level"] == levels[name] + 1 for name, hand in state.hands.items())


def test_constellation_can_make_a_planet_better_than_black_hole():
    state = create_run_state("black_hole_constellation", deck_key="b_blue")
    add_joker(state, "j_half")
    add_joker(state, "j_blue_joker")
    constellation = add_joker(state, "j_constellation")
    state.hands["Pair"].update(level=20, chips=295, mult=21)
    make_pack(state)
    action = select_black_hole_pack(
        state, compute_action_mask(state, SubPhase.BOOSTER_PACK), HeuristicAgent(), ShopSearch(),
    )
    assert action == AR.PACK_CLAIM_START
    old_multiplier = constellation.x_mult
    claim_pack_card(state, 0)
    assert constellation.x_mult > old_multiplier
    assert state.hands["High Card"]["level"] == 1


def test_red_card_skip_can_outvalue_a_late_level_upgrade():
    state = create_run_state("black_hole_red_card", deck_key="b_blue")
    add_joker(state, "j_half")
    add_joker(state, "j_blue_joker")
    add_joker(state, "j_red_card")
    state.hands["Pair"].update(level=50, chips=745, mult=51)
    make_pack(state)
    action = select_black_hole_pack(
        state, compute_action_mask(state, SubPhase.BOOSTER_PACK), HeuristicAgent(), ShopSearch(),
    )
    assert action == AR.PACK_SKIP


def test_masked_black_hole_is_never_selected():
    state = create_run_state("black_hole_mask", deck_key="b_blue")
    make_pack(state)
    mask = compute_action_mask(state, SubPhase.BOOSTER_PACK)
    mask[AR.PACK_CLAIM_START + 1] = False
    action = select_black_hole_pack(state, mask, HeuristicAgent(), ShopSearch())
    assert action == AR.PACK_CLAIM_START
