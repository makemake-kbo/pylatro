"""v20: deck tokens carry this round's Joker targets as a bitmask in column 12."""
from pylatro import add_joker, create_run_state, load_game_data, start_blind
from pylatro_agent.constants import DECK_START, SubPhase
from pylatro_agent.tokenizer import Tokenizer
from pylatro_agent.vocab import build_vocab


def _hand_bits(state):
    tok = Tokenizer(vocab=build_vocab(load_game_data()))
    raw = tok.tokenize(state, SubPhase.CHOOSE_ACTION)
    tokens = raw["tokens"] if isinstance(raw, dict) else raw.tokens
    return [int(tokens[DECK_START + i, 12]) for i in range(len(state.hand_cards))]


def _state():
    state = create_run_state("round_target_tokens", deck_key="b_red")
    start_blind(state, "Small")
    by_key = {c.front_key: c for c in state.deck_cards}
    state.hand_cards = [by_key[k] for k in ("C_9", "H_9", "S_K", "D_2")]
    state.current_round.mail_card = {"rank": "9", "id": 9}
    state.current_round.idol_card = {"rank": "K", "suit": "Spades", "id": 13}
    state.current_round.ancient_card = {"suit": "Hearts"}
    state.current_round.castle_card = {"suit": "Diamonds"}
    return state


def test_targets_hidden_without_the_jokers():
    assert _hand_bits(_state()) == [0, 0, 0, 0]


def test_each_target_sets_its_bit():
    state = _state()
    for key in ("j_mail", "j_idol", "j_ancient", "j_castle"):
        add_joker(state, key)
    assert _hand_bits(state) == [1, 1 | 4, 2, 8]
