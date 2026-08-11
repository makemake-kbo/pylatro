"""Tests for the tokenizer."""

from __future__ import annotations

import pytest

from pylatro import (
    add_joker,
    cash_out,
    create_run_state,
    load_game_data,
    open_booster_pack,
    populate_shop,
    select_blind,
    skip_blind,
    start_blind,
)
from pylatro_agent.constants import (
    HAND_CANDIDATE_MAX,
    HAND_CANDIDATE_START,
    MAX_PACK_CARDS,
    MAX_SEQ_LEN,
    SCALAR_DIM,
    SHOP_START,
    TOKEN_DIM,
    SubPhase,
    TokenType,
)
from pylatro_agent.hand_candidates import generate_hand_candidates
from pylatro_agent.tokenizer import (
    Tokenizer,
    _hypergeom_at_least_one,
    sign_log,
    strategy_probability_features,
)
from pylatro_agent.vocab import build_vocab


@pytest.fixture(scope="module")
def game_data():
    return load_game_data()


@pytest.fixture(scope="module")
def vocab(game_data):
    return build_vocab(game_data)


@pytest.fixture
def run_state(game_data):
    state = create_run_state("test_seed", 1, "b_red", data=game_data)
    select_blind(state, "Small")
    start_blind(state, "Small")
    return state


def test_sign_log():
    assert sign_log(0) == 0.0
    assert sign_log(1) == pytest.approx(0.6931, abs=0.01)
    assert sign_log(-1) == pytest.approx(-0.6931, abs=0.01)
    assert sign_log(100) > 0


def test_tokenize_shape(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    assert obs.tokens.shape == (MAX_SEQ_LEN, TOKEN_DIM)
    assert obs.token_types.shape == (MAX_SEQ_LEN,)
    assert obs.scalars.shape == (SCALAR_DIM,)
    assert obs.attention_mask.shape == (MAX_SEQ_LEN,)


def test_tokenize_records_configured_win_ante(run_state, vocab):
    run_state.win_ante = 5
    obs = Tokenizer(vocab=vocab).tokenize(run_state, SubPhase.CHOOSE_ACTION)

    assert obs.scalars[2] == run_state.round_resets.ante
    assert obs.scalars[22] == 5.0


def test_strategy_probability_scalars_expose_live_card_opportunities(run_state, vocab):
    run_state.round_resets.blind_choices["Boss"] = "bl_hook"
    for card in run_state.deck_cards:
        card.seal = None
        card.center_key = "c_base"
        card.debuff = False
    run_state.hand_cards[0].seal = "Blue"
    run_state.hand_cards[1].seal = "Purple"
    run_state.hand_cards[2].center_key = "m_gold"
    run_state.hand_cards[3].center_key = "m_steel"

    features = strategy_probability_features(
        run_state,
        SubPhase.CHOOSE_ACTION,
        clear_probability=0.8,
    )
    obs = Tokenizer(vocab=vocab).tokenize(
        run_state,
        SubPhase.CHOOSE_ACTION,
        clear_probability=0.8,
    )

    assert features[:5] == pytest.approx((1.0, 1.0, 1.0, 1.0, 1.0))
    assert obs.scalars[13:22] == pytest.approx(features)


def test_purple_search_probability_reserves_a_discard(run_state):
    for card in run_state.deck_cards:
        card.seal = None
        card.debuff = False
    run_state.draw_pile[0].seal = "Purple"
    run_state.current_round.discards_left = 1

    with_one = strategy_probability_features(run_state, SubPhase.CHOOSE_ACTION)[1]
    run_state.current_round.discards_left = 2
    with_two = strategy_probability_features(run_state, SubPhase.CHOOSE_ACTION)[1]

    assert with_one == 0.0
    assert with_two > 0.0


def test_strategy_probability_keeps_ineligible_cards_as_failure_draws(game_data):
    state = create_run_state("strategy_probability_denominator", 1, "b_red", data=game_data)
    state.blind_on_deck = "Boss"
    state.round_resets.blind_choices["Boss"] = "bl_goad"
    state.round_resets.hands = 5
    state.round_resets.discards = 3
    for card in state.deck_cards:
        card.seal = None
    success = next(card for card in state.deck_cards if card.suit == "Hearts")
    success.seal = "Blue"

    probability = strategy_probability_features(state, SubPhase.BLIND_SELECT)[0]

    assert probability == pytest.approx(43 / 52)


@pytest.mark.parametrize(
    ("population", "successes", "draws", "expected"),
    ((0, 1, 1, 0.0), (52, 0, 43, 0.0), (52, 1, 0, 0.0), (52, 52, 1, 1.0)),
)
def test_hypergeom_probability_is_bounded_for_degenerate_inputs(
    population: int,
    successes: int,
    draws: int,
    expected: float,
) -> None:
    value = _hypergeom_at_least_one(population, successes, draws)
    assert value == pytest.approx(expected)
    assert 0.0 <= value <= 1.0


def test_preblind_strategy_probabilities_exclude_known_boss_suit(game_data):
    state = create_run_state("strategy_boss_suit", 1, "b_red", data=game_data)
    state.blind_on_deck = "Boss"
    state.round_resets.blind_choices["Boss"] = "bl_goad"
    for card in state.deck_cards:
        card.seal = None
    spade = next(card for card in state.deck_cards if card.suit == "Spades")
    spade.seal = "Blue"

    features = strategy_probability_features(state, SubPhase.BLIND_SELECT)

    assert features[0] == 0.0
    assert features[5] == 0.0

    state.blind_disabled = True
    disabled_features = strategy_probability_features(state, SubPhase.BLIND_SELECT)
    assert disabled_features[0] > 0.0
    assert disabled_features[5] > 0.0


def test_disabled_boss_cashout_does_not_hide_next_ante_suit_boss(game_data):
    state = create_run_state("disabled_boss_next_ante", 1, "b_red", data=game_data)
    state.blind_on_deck = "Boss"
    state.round_resets.blind_choices["Boss"] = "bl_goad"
    start_blind(state, "Boss")
    for card in state.deck_cards:
        card.seal = None
    spade = next(card for card in state.deck_cards if card.suit == "Spades")
    spade.seal = "Blue"
    state.blind_disabled = True

    active_disabled = strategy_probability_features(state, SubPhase.CHOOSE_ACTION)
    assert active_disabled[0] > 0.0
    assert active_disabled[5] > 0.0

    state.round_resets.blind_states["Boss"] = "Defeated"
    cash_out(state)
    state.round_resets.blind_choices["Boss"] = "bl_goad"
    skip_blind(state)
    skip_blind(state)

    assert state.round_resets.ante == 2
    assert state.blind_on_deck == "Boss"
    assert not state.blind_disabled
    next_boss = strategy_probability_features(state, SubPhase.BLIND_SELECT)
    assert next_boss[0] == 0.0
    assert next_boss[5] == 0.0


def test_suit_utilities_preserve_ties_and_follow_deck_history_and_jokers(game_data):
    tied = create_run_state("suit_utility_tie", 1, "b_red", data=game_data)
    tied.round_resets.blind_choices["Boss"] = "bl_hook"
    tied_utilities = strategy_probability_features(tied, SubPhase.BLIND_SELECT)[5:9]

    hearts = create_run_state("suit_utility_hearts", 1, "b_red", data=game_data)
    hearts.round_resets.blind_choices["Boss"] = "bl_hook"
    for card in hearts.deck_cards:
        if card.suit == "Diamonds" and card.rank in {"2", "3", "4", "5", "6", "7", "8", "9"}:
            card.suit = "Hearts"
    hearts.hands["Flush"]["played"] = 4
    for card in hearts.deck_cards:
        if card.suit == "Hearts":
            card.times_played = 2
    hearts_utilities = strategy_probability_features(hearts, SubPhase.BLIND_SELECT)[5:9]

    spades = create_run_state("suit_utility_spades", 1, "b_red", data=game_data)
    spades.round_resets.blind_choices["Boss"] = "bl_hook"
    add_joker(spades, "j_arrowhead")
    spade_utilities = strategy_probability_features(spades, SubPhase.BLIND_SELECT)[5:9]

    assert max(tied_utilities) - min(tied_utilities) == pytest.approx(0.0)
    assert tied_utilities[0] < 0.25
    assert hearts_utilities[1] == max(hearts_utilities)
    assert hearts_utilities[1] > tied_utilities[1]
    assert spade_utilities[0] == max(spade_utilities)
    assert spade_utilities[0] > tied_utilities[0]


def test_tokenize_has_tokens(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    # Should have OBJ token
    assert obs.token_types[0] == TokenType.OBJ
    assert obs.attention_mask[0] == 1

    # Should have META tokens
    for i in range(1, 10):
        assert obs.token_types[i] == TokenType.META
        assert obs.attention_mask[i] == 1

    # Should have some DECK tokens (hand cards at minimum)
    deck_mask = obs.token_types[10:72] == TokenType.DECK
    assert deck_mask.sum() > 0, "Should have deck card tokens"


def test_tokenize_total_within_limit(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    active_tokens = (obs.attention_mask == 1).sum()
    assert active_tokens <= MAX_SEQ_LEN


def test_tokenize_choose_action_includes_hand_context_tokens(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    assert (obs.token_types == TokenType.HAND_LEVEL).sum() > 0
    assert (obs.token_types == TokenType.HAND_CANDIDATE).sum() > 0


def test_tokenize_blind_select_tokens(game_data, vocab):
    state = create_run_state("test_seed", 1, "b_red", data=game_data)
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(state, SubPhase.BLIND_SELECT)

    # Should have blind select tokens
    blind_types = obs.token_types[99:102]
    assert (blind_types == TokenType.BLIND_SELECT).sum() > 0


def test_tokenize_includes_round_score_context(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION, round_score=125)
    blind_target = tok._blind_target(run_state)

    assert obs.scalars[8] == pytest.approx(sign_log(125.0))
    assert obs.scalars[9] == pytest.approx(sign_log(max(blind_target - 125.0, 0.0)))
    assert obs.scalars[10] == pytest.approx(min(125.0 / max(blind_target, 1.0), 1.0))


def test_tokenize_includes_explicit_immediate_risk_context(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(
        run_state,
        SubPhase.CHOOSE_ACTION,
        clear_probability=0.7,
        immediate_death_probability=0.3,
    )

    assert obs.scalars[11] == pytest.approx(0.7)
    assert obs.scalars[12] == pytest.approx(0.3)


def test_hand_candidates_emitted_in_choose_action(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    candidate_mask = obs.token_types == TokenType.HAND_CANDIDATE
    n_candidates = candidate_mask.sum()
    assert n_candidates > 0, "Hand candidates should be emitted during CHOOSE_ACTION"

    for i in range(HAND_CANDIDATE_START, HAND_CANDIDATE_START + n_candidates):
        assert obs.token_types[i] == TokenType.HAND_CANDIDATE
        assert obs.attention_mask[i] == 1


def test_ante_one_play_candidates_preserve_v6_score_and_coverage_semantics(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION, round_score=40)
    play_candidates, _ = generate_hand_candidates(run_state)
    candidate = play_candidates[0]

    assert obs.tokens[HAND_CANDIDATE_START, 3] == int(sign_log(candidate.estimated_score) * 10.0)
    assert obs.tokens[HAND_CANDIDATE_START, 4] == int(min(candidate.blind_ratio * 10.0, 255.0))


def test_hand_candidates_not_emitted_in_other_phases(run_state, vocab):
    tok = Tokenizer(vocab=vocab)

    for phase in (SubPhase.SHOP, SubPhase.BLIND_SELECT, SubPhase.BOOSTER_PACK):
        obs = tok.tokenize(run_state, phase)
        n = (obs.token_types == TokenType.HAND_CANDIDATE).sum()
        assert n == 0, f"Hand candidates should not be emitted during {phase}"


def test_hand_candidates_have_valid_kind(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    candidate_mask = obs.token_types == TokenType.HAND_CANDIDATE
    n_candidates = int(candidate_mask.sum())
    assert n_candidates > 0

    for i in range(n_candidates):
        kind = obs.tokens[HAND_CANDIDATE_START + i, 0]
        assert kind in (1, 2), f"Candidate {i} kind should be 1 (play) or 2 (discard), got {kind}"


def test_hand_candidates_have_valid_size(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    candidate_mask = obs.token_types == TokenType.HAND_CANDIDATE
    n_candidates = int(candidate_mask.sum())
    hand_size = len(run_state.hand_cards)

    for i in range(n_candidates):
        size = obs.tokens[HAND_CANDIDATE_START + i, 2]
        assert 1 <= size <= min(5, hand_size), f"Candidate {i} size {size} out of range"


def test_hand_candidates_card_indices_valid(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    candidate_mask = obs.token_types == TokenType.HAND_CANDIDATE
    n_candidates = int(candidate_mask.sum())
    hand_size = len(run_state.hand_cards)

    for i in range(n_candidates):
        size = obs.tokens[HAND_CANDIDATE_START + i, 2]
        for j in range(size):
            card_idx = obs.tokens[HAND_CANDIDATE_START + i, 5 + j] - 1
            assert 0 <= card_idx < hand_size, (
                f"Candidate {i} card slot {j} = {card_idx} out of range for hand size {hand_size}"
            )


def test_hand_candidates_rank_slot_assigned(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    candidate_mask = obs.token_types == TokenType.HAND_CANDIDATE
    n_candidates = int(candidate_mask.sum())

    rank_slots = set()
    for i in range(n_candidates):
        rank_slot = obs.tokens[HAND_CANDIDATE_START + i, 10]
        assert rank_slot >= 1, f"Candidate {i} rank_slot should be >= 1"
        rank_slots.add(rank_slot)
    assert len(rank_slots) == n_candidates, "Each candidate should have a unique rank_slot"


def test_hand_candidates_within_max(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    candidate_mask = obs.token_types == TokenType.HAND_CANDIDATE
    n_candidates = int(candidate_mask.sum())
    assert n_candidates <= HAND_CANDIDATE_MAX


def test_hand_candidates_play_before_discard(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    candidate_mask = obs.token_types == TokenType.HAND_CANDIDATE
    n_candidates = int(candidate_mask.sum())
    if n_candidates < 2:
        pytest.skip("Need at least 2 candidates")

    first_kind = obs.tokens[HAND_CANDIDATE_START, 0]
    assert first_kind == 1, "First candidates should be play candidates (kind=1)"


def _make_pack_state(game_data):
    state = create_run_state("pack_test", 1, "b_red", data=game_data)
    select_blind(state, "Small")
    start_blind(state, "Small")
    state.current_round.hands_left = 0
    cash_out(state)
    populate_shop(state)
    for i, booster in enumerate(state.shop.boosters):
        if booster is not None:
            open_booster_pack(state, i)
            return state
    pytest.skip("No booster pack in shop")


@pytest.fixture
def pack_state(game_data):
    return _make_pack_state(game_data)


def test_booster_pack_cards_encoded(pack_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(pack_state, SubPhase.BOOSTER_PACK)

    assert pack_state.pack is not None
    n_pack_cards = len(pack_state.pack.cards)
    assert n_pack_cards > 0, "Pack should have at least one card"

    shop_tokens_mask = obs.token_types[SHOP_START : SHOP_START + MAX_PACK_CARDS] == TokenType.SHOP
    n_encoded = shop_tokens_mask.sum()
    assert n_encoded == n_pack_cards, f"Expected {n_pack_cards} pack tokens, got {n_encoded}"


def test_booster_pack_cards_have_attention(pack_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(pack_state, SubPhase.BOOSTER_PACK)

    n_pack_cards = len(pack_state.pack.cards)
    for i in range(n_pack_cards):
        assert obs.attention_mask[SHOP_START + i] == 1, f"Pack card {i} should have attention"


def test_booster_pack_cards_have_item_type(pack_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(pack_state, SubPhase.BOOSTER_PACK)

    n_pack_cards = len(pack_state.pack.cards)
    for i in range(n_pack_cards):
        item_type = obs.tokens[SHOP_START + i, 1]
        assert item_type >= 1, f"Pack card {i} should have a non-zero item type"


def test_booster_pack_not_emitted_in_shop_phase(pack_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(pack_state, SubPhase.SHOP)

    for i in range(MAX_PACK_CARDS):
        assert obs.attention_mask[SHOP_START + i] == 0 or obs.token_types[SHOP_START + i] == TokenType.SHOP


def test_booster_pack_no_pack_does_not_crash(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.BOOSTER_PACK)
    assert obs.tokens.shape == (MAX_SEQ_LEN, TOKEN_DIM)


def test_booster_pack_slots_zeroed_for_empty(pack_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(pack_state, SubPhase.BOOSTER_PACK)

    n_pack_cards = len(pack_state.pack.cards)
    for i in range(n_pack_cards, MAX_PACK_CARDS):
        assert obs.attention_mask[SHOP_START + i] == 0


def test_shop_tokens_present_in_shop_phase(game_data, vocab):
    state = create_run_state("shop_test", 1, "b_red", data=game_data)
    select_blind(state, "Small")
    start_blind(state, "Small")
    state.current_round.hands_left = 0
    cash_out(state)
    populate_shop(state)

    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(state, SubPhase.SHOP)

    shop_items = list(state.shop.cards) + list(state.shop.vouchers) + list(state.shop.boosters)
    n_items = len(shop_items)
    shop_types = obs.token_types[SHOP_START : SHOP_START + n_items]
    assert (shop_types == TokenType.SHOP).sum() == n_items
