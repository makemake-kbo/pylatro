from math import isnan

from pylatro import add_joker, cash_out, create_run_state, get_blind_amount, score_hand
from pylatro.models import PlayingCard


def test_base_joker_blueprint_and_plasma_scoring_fixtures() -> None:
    state = create_run_state("AAAAAAAA")
    base = score_hand(state, [PlayingCard(front_key="S_A", suit="Spades", rank="A")])
    assert (base.hand_name, base.chips, base.mult, base.total) == ("High Card", 16.0, 1.0, 16)

    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_joker")
    joker = score_hand(state, [PlayingCard(front_key="S_A", suit="Spades", rank="A")])
    assert (joker.hand_name, joker.chips, joker.mult, joker.total) == ("High Card", 16.0, 5.0, 80)

    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_blueprint")
    add_joker(state, "j_joker")
    blueprint = score_hand(state, [PlayingCard(front_key="S_A", suit="Spades", rank="A")])
    assert (blueprint.hand_name, blueprint.chips, blueprint.mult, blueprint.total) == ("High Card", 16.0, 9.0, 144)

    state = create_run_state("AAAAAAAA", deck_key="b_plasma")
    plasma = score_hand(state, [PlayingCard(front_key="S_A", suit="Spades", rank="A")])
    assert (plasma.hand_name, plasma.chips, plasma.mult, plasma.total) == ("High Card", 8, 8, 64)


def test_joker_position_formulaic_and_held_card_effects() -> None:
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_joker")
    add_joker(state, "j_brainstorm")
    brainstorm = score_hand(state, [PlayingCard(front_key="S_A", suit="Spades", rank="A")])
    assert (brainstorm.hand_name, brainstorm.chips, brainstorm.mult, brainstorm.total) == ("High Card", 16.0, 9.0, 144)

    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_abstract")
    add_joker(state, "j_joker")
    abstract = score_hand(state, [PlayingCard(front_key="S_A", suit="Spades", rank="A")])
    assert (abstract.hand_name, abstract.chips, abstract.mult, abstract.total) == ("High Card", 16.0, 11.0, 176)

    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_stencil")
    stencil = score_hand(state, [PlayingCard(front_key="S_A", suit="Spades", rank="A")])
    assert (stencil.hand_name, stencil.chips, stencil.mult, stencil.total) == ("High Card", 16.0, 5.0, 80)
    assert state.jokers[0].x_mult == 5

    state = create_run_state("AAAAAAAA")
    state.current_round.hands_left = 0
    add_joker(state, "j_acrobat")
    acrobat = score_hand(state, [PlayingCard(front_key="S_A", suit="Spades", rank="A")])
    assert (acrobat.hand_name, acrobat.chips, acrobat.mult, acrobat.total) == ("High Card", 16.0, 3.0, 48)

    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_half")
    half = score_hand(
        state,
        [
            PlayingCard(front_key="S_A", suit="Spades", rank="A"),
            PlayingCard(front_key="H_K", suit="Hearts", rank="K"),
            PlayingCard(front_key="D_2", suit="Diamonds", rank="2"),
        ],
    )
    assert (half.hand_name, half.chips, half.mult, half.total) == ("High Card", 16.0, 21.0, 336)

    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_baron")
    baron = score_hand(
        state,
        [PlayingCard(front_key="S_A", suit="Spades", rank="A")],
        held_hand=[PlayingCard(front_key="H_K", suit="Hearts", rank="K")],
    )
    assert (baron.hand_name, baron.chips, baron.mult, baron.total) == ("High Card", 16.0, 1.5, 24)


def test_type_and_suit_jokers_stack_on_pair_fixture() -> None:
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_greedy_joker")
    add_joker(state, "j_jolly")
    add_joker(state, "j_sly")

    result = score_hand(
        state,
        [
            PlayingCard(front_key="D_3", suit="Diamonds", rank="3"),
            PlayingCard(front_key="H_3", suit="Hearts", rank="3"),
        ],
    )

    assert (result.hand_name, result.chips, result.mult, result.total) == ("Pair", 66.0, 13.0, 858)


def test_boss_cash_out_advances_seeded_antes_through_win_and_endless() -> None:
    state = create_run_state("AAAAAAAA")
    progression: list[tuple[int, str, str, str, str, bool]] = []
    for _ in range(8):
        state.round_resets.blind_states["Boss"] = "Defeated"
        cash_out(state)
        progression.append(
            (
                state.round_resets.ante,
                state.round_resets.blind_choices["Boss"],
                state.current_voucher or "",
                state.round_resets.blind_tags["Small"],
                state.round_resets.blind_tags["Big"],
                state.won,
            )
        )

    assert progression == [
        (2, "bl_house", "v_magic_trick", "tag_juggle", "tag_ethereal", False),
        (3, "bl_fish", "v_seed_money", "tag_coupon", "tag_uncommon", False),
        (4, "bl_window", "v_telescope", "tag_coupon", "tag_d_six", False),
        (5, "bl_wall", "v_paint_brush", "tag_coupon", "tag_garbage", False),
        (6, "bl_tooth", "v_seed_money", "tag_standard", "tag_orbital", False),
        (7, "bl_flint", "v_hieroglyph", "tag_boss", "tag_ethereal", False),
        (8, "bl_final_leaf", "v_blank", "tag_charm", "tag_top_up", False),
        (9, "bl_goad", "v_crystal_ball", "tag_garbage", "tag_investment", True),
    ]


def test_blind_amount_matches_reference_tables_into_endless() -> None:
    assert [(ante, get_blind_amount(ante), get_blind_amount(ante, 2), get_blind_amount(ante, 3)) for ante in range(1, 13)] == [
        (1, 300, 300, 300),
        (2, 800, 900, 1000),
        (3, 2000, 2600, 3200),
        (4, 5000, 8000, 9000),
        (5, 11000, 20000, 25000),
        (6, 20000, 36000, 60000),
        (7, 35000, 60000, 110000),
        (8, 50000, 100000, 200000),
        (9, 110000, 230000, 460000),
        (10, 560000, 1100000, 2200000),
        (11, 7200000, 14000000, 29000000),
        (12, 300000000, 600000000, 1200000000),
    ]
    assert get_blind_amount(38) == 4.5e288
    assert isnan(get_blind_amount(39))
