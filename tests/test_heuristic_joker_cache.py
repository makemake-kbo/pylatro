"""A persistent teacher must see purchases and sales on the same RunState."""

from pylatro import add_joker, create_run_state
from pylatro.instances import remove_joker
from pylatro_agent.heuristic import HeuristicAgent


def test_teacher_notices_first_joker_after_empty_inventory_was_cached():
    state = create_run_state("teacher_first_joker")
    agent = HeuristicAgent()
    assert agent._get_joker_names(state) == frozenset()

    add_joker(state, "j_shortcut")

    assert agent._get_joker_names(state) == frozenset({"Shortcut"})


def test_teacher_forgets_sold_joker_after_nonempty_inventory_was_cached():
    state = create_run_state("teacher_sale")
    joker = add_joker(state, "j_shortcut")
    agent = HeuristicAgent()
    assert agent._get_joker_names(state) == frozenset({"Shortcut"})

    remove_joker(state, joker)

    assert agent._get_joker_names(state) == frozenset()


def test_teacher_refreshes_names_when_inventory_replacement_preserves_length():
    state = create_run_state("teacher_replacement")
    old = add_joker(state, "j_shortcut")
    agent = HeuristicAgent()
    assert agent._get_joker_names(state) == frozenset({"Shortcut"})

    remove_joker(state, old)
    add_joker(state, "j_four_fingers")

    assert agent._get_joker_names(state) == frozenset({"Four Fingers"})
    assert agent._get_joker_names(state) == HeuristicAgent()._get_joker_names(state)
