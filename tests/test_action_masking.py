"""Tests for action masking."""

from __future__ import annotations

import numpy as np
import pytest

from pylatro import (
    add_consumable,
    add_joker,
    cash_out,
    create_run_state,
    load_game_data,
    open_booster_pack,
    populate_shop,
    select_blind,
    start_blind,
)
from pylatro.models import PackState, ShopCard
from pylatro.runtime import consumable_limit
from pylatro.shop import claim_pack_card, pack_consumable_use_targets
from pylatro_agent.action import ActionType, encode_action
from pylatro_agent.constants import ActionRange, SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.masks import compute_action_mask
from pylatro_agent.subset_actions import consumable_subset_index


@pytest.fixture(scope="module")
def game_data():
    return load_game_data()


@pytest.fixture
def hand_play_state(game_data):
    state = create_run_state("test_seed", 1, "b_red", data=game_data)
    select_blind(state, "Small")
    start_blind(state, "Small")
    return state


def test_blind_select_mask(game_data):
    state = create_run_state("test_seed", 1, "b_red", data=game_data)
    mask = compute_action_mask(state, SubPhase.BLIND_SELECT)

    assert mask[ActionRange.BLIND_PLAY] == 1, "Play should always be valid"
    assert mask[ActionRange.BLIND_SKIP] == 1, "Skip should be valid for Small blind"
    assert mask.sum() >= 2


def test_choose_action_mask(hand_play_state):
    mask = compute_action_mask(hand_play_state, SubPhase.CHOOSE_ACTION)

    assert mask[ActionRange.PLAY_SUBSET_START] == 1, "Should expose at least one play subset"
    assert mask[ActionRange.DISCARD_SUBSET_START] == 1, "Should expose at least one discard subset"
    # Consumable depends on state
    assert mask.sum() >= 2


def test_action_space_has_no_reorder_actions(hand_play_state):
    # Joker ordering is harness-owned (joker_layout.py); the policy's action
    # space ends at PACK_SKIP and a joker-heavy board adds no reorder actions.
    add_joker(hand_play_state, "j_cavendish")
    add_joker(hand_play_state, "j_joker")

    from pylatro_agent.constants import NUM_ACTIONS

    mask = compute_action_mask(hand_play_state, SubPhase.CHOOSE_ACTION)

    assert mask.shape == (NUM_ACTIONS,)
    assert int(ActionRange.PACK_SKIP) + 1 == NUM_ACTIONS



def test_choose_action_subset_ranges_are_bounded(hand_play_state):
    mask = compute_action_mask(hand_play_state, SubPhase.CHOOSE_ACTION)

    play_mask = mask[ActionRange.PLAY_SUBSET_START : ActionRange.PLAY_SUBSET_END + 1]
    discard_mask = mask[ActionRange.DISCARD_SUBSET_START : ActionRange.DISCARD_SUBSET_END + 1]
    assert play_mask.sum() >= 1
    assert discard_mask.sum() >= 1
    assert play_mask.sum() <= len(play_mask)
    assert discard_mask.sum() <= len(discard_mask)


def test_hand_targeted_consumable_actions_are_unmasked(hand_play_state):
    add_consumable(hand_play_state, "c_lovers")

    mask = compute_action_mask(hand_play_state, SubPhase.CHOOSE_ACTION)
    action = encode_action(
        ActionType.USE_CONSUMABLE_HAND_SUBSET,
        0,
        consumable_subset_index((0,)),
    )

    assert mask[action] == 1


def test_aura_consumable_uses_hand_target_fallback(hand_play_state):
    add_consumable(hand_play_state, "c_aura")

    mask = compute_action_mask(hand_play_state, SubPhase.CHOOSE_ACTION)
    action = encode_action(
        ActionType.USE_CONSUMABLE_HAND_SUBSET,
        0,
        consumable_subset_index((0,)),
    )

    assert mask[action] == 1


def test_heuristic_can_pick_hand_targeted_consumable(hand_play_state):
    add_consumable(hand_play_state, "c_lovers")
    mask = compute_action_mask(hand_play_state, SubPhase.CHOOSE_ACTION)

    action = HeuristicAgent().select_action(hand_play_state, SubPhase.CHOOSE_ACTION, mask)

    assert ActionRange.CONSUMABLE_FLAT_START <= action <= ActionRange.CONSUMABLE_FLAT_END


def test_shop_mask_leave_always_valid(hand_play_state):
    # Simulate being in shop phase
    # This is a simplified test — just check leave is always valid in shop mask
    mask = compute_action_mask(hand_play_state, SubPhase.SHOP)
    assert mask[ActionRange.SHOP_LEAVE] == 1


def test_pack_mask_joker_capacity_check(game_data):
    state = create_run_state("pack_mask_test", 1, "b_red", data=game_data)
    select_blind(state, "Small")
    start_blind(state, "Small")
    state.current_round.hands_left = 0
    cash_out(state)
    populate_shop(state)

    from pylatro.runtime import joker_limit

    jlimit = joker_limit(state)

    for _ in range(jlimit):
        add_joker(state, "j_joker")

    for i, booster in enumerate(state.shop.boosters):
        if booster is not None:
            open_booster_pack(state, i)
            break
    else:
        pytest.skip("No booster pack in shop")

    if state.pack is None:
        pytest.skip("No pack opened")

    mask = compute_action_mask(state, SubPhase.BOOSTER_PACK)

    has_joker_in_pack = False
    for i, card in enumerate(state.pack.cards):
        center = state.data.centers.get(card.center_key, {})
        if center.get("set") == "Joker":
            has_joker_in_pack = True
            is_negative = bool(card.edition and card.edition.get("negative"))
            if not is_negative:
                assert mask[ActionRange.PACK_CLAIM_START + i] == 0, (
                    "Non-negative joker at full slots should be masked out"
                )
            else:
                assert mask[ActionRange.PACK_CLAIM_START + i] == 1, (
                    "Negative joker should be claimable even at full slots"
                )

    if not has_joker_in_pack:
        pytest.skip("No joker in pack for this test")


def _full_consumable_pack_state(game_data, center_key: str, *, negative: bool = False):
    state = create_run_state(f"full_pack_{center_key}", 1, "b_red", data=game_data)
    for _ in range(consumable_limit(state)):
        add_consumable(state, "c_fool")
    center = game_data.centers[center_key]
    state.pack = PackState(
        booster_key="p_arcana_normal_1",
        state_name="Tarot",
        choices_remaining=1,
        cards=[
            ShopCard(
                center_key=center_key,
                card_type=str(center.get("set") or ""),
                cost=0,
                base_cost=0,
                edition={"negative": True} if negative else None,
            )
        ],
    )
    return state


@pytest.mark.parametrize(
    ("center_key", "needs_joker"),
    (
        ("c_hermit", False),
        ("c_temperance", False),
        ("c_pluto", False),
        ("c_judgement", False),
        ("c_wheel_of_fortune", True),
    ),
)
def test_full_consumable_slots_allow_immediate_pack_auto_use(game_data, center_key, needs_joker):
    state = _full_consumable_pack_state(game_data, center_key)
    if needs_joker:
        add_joker(state, "j_joker")

    mask = compute_action_mask(state, SubPhase.BOOSTER_PACK)

    assert mask[ActionRange.PACK_CLAIM_START] == 1
    claimed = claim_pack_card(state, 0)
    assert claimed.auto_used
    assert claimed.use_result is not None
    assert len(state.consumables) == consumable_limit(state)


@pytest.mark.parametrize(
    ("center_key", "negative"),
    (("c_fool", False), ("c_pluto", True)),
)
def test_full_consumable_slots_block_targeted_or_banked_pack_cards(game_data, center_key, negative):
    state = _full_consumable_pack_state(game_data, center_key, negative=negative)
    state.last_tarot_planet = "c_hermit"

    mask = compute_action_mask(state, SubPhase.BOOSTER_PACK)

    assert mask[ActionRange.PACK_CLAIM_START] == 0
    with pytest.raises(ValueError, match="cannot be claimed"):
        claim_pack_card(state, 0)


def _put_pack_target_hand(state) -> None:
    state.hand_cards = list(state.deck_cards[:8])
    state.draw_pile = list(state.deck_cards[8:])
    state.discard_pile = []


def _put_balanced_pack_target_hand(state) -> None:
    cards = []
    for suit in ("Spades", "Hearts", "Clubs", "Diamonds"):
        cards.extend([card for card in state.deck_cards if card.suit == suit][:2])
    state.hand_cards = cards
    selected = {card.reward_uid for card in cards}
    state.draw_pile = [card for card in state.deck_cards if card.reward_uid not in selected]
    state.discard_pile = []


def _set_single_pack_card(state, center_key: str) -> None:
    state.pack = PackState(
        booster_key="p_arcana_normal_1",
        state_name="Tarot",
        choices_remaining=1,
        cards=[ShopCard(center_key=center_key, card_type="Tarot", cost=0, base_cost=0)],
    )


@pytest.mark.parametrize(
    ("center_key", "target_suit"),
    (
        ("c_world", "Spades"),
        ("c_sun", "Hearts"),
        ("c_moon", "Clubs"),
        ("c_star", "Diamonds"),
    ),
)
def test_tied_stock_deck_suit_tarots_are_symmetric_and_share_mask_execution_targets(
    game_data,
    center_key,
    target_suit,
) -> None:
    state = _full_consumable_pack_state(game_data, center_key)
    _put_balanced_pack_target_hand(state)
    state.round_resets.blind_choices["Boss"] = "bl_hook"

    planned = pack_consumable_use_targets(state, center_key)
    assert planned is not None
    assert len(planned[0]) == 3
    mask = compute_action_mask(state, SubPhase.BOOSTER_PACK)
    claimed = claim_pack_card(state, 0)

    assert mask[ActionRange.PACK_CLAIM_START] == 1
    assert claimed.auto_used_hand_targets == planned[0]
    assert all(state.hand_cards[index].suit == target_suit for index in planned[0])


@pytest.mark.parametrize("center_key", ("c_hanged_man", "c_strength"))
def test_stock_deck_hanged_and_strength_have_safe_shared_pack_targets(game_data, center_key) -> None:
    state = _full_consumable_pack_state(game_data, center_key)
    _put_balanced_pack_target_hand(state)
    state.round_resets.blind_choices["Boss"] = "bl_hook"
    pre_size = len(state.deck_cards)
    pre_ranks = tuple(card.rank for card in state.hand_cards)

    planned = pack_consumable_use_targets(state, center_key)
    assert planned is not None
    mask = compute_action_mask(state, SubPhase.BOOSTER_PACK)
    claimed = claim_pack_card(state, 0)

    assert mask[ActionRange.PACK_CLAIM_START] == 1
    assert claimed.auto_used_hand_targets == planned[0]
    if center_key == "c_hanged_man":
        assert len(state.deck_cards) == pre_size - len(planned[0])
    else:
        assert any(state.hand_cards[index].rank != pre_ranks[index] for index in planned[0])


@pytest.mark.parametrize(
    ("center_key", "boss_key"),
    (
        ("c_world", "bl_goad"),
        ("c_sun", "bl_head"),
        ("c_moon", "bl_club"),
        ("c_star", "bl_window"),
    ),
)
def test_pack_suit_tarot_never_targets_upcoming_boss_suit(game_data, center_key, boss_key) -> None:
    state = _full_consumable_pack_state(game_data, center_key)
    _put_balanced_pack_target_hand(state)
    state.round_resets.blind_choices["Boss"] = boss_key
    state.blind_disabled = True  # even stale/current disable cannot waive an upcoming boss

    assert pack_consumable_use_targets(state, center_key) is None
    mask = compute_action_mask(state, SubPhase.BOOSTER_PACK)
    assert mask[ActionRange.PACK_CLAIM_START] == 0


def test_pack_suit_tarot_reinforces_confident_suit_without_converting_away(game_data) -> None:
    state = create_run_state("pack_confident_suit", 1, "b_red", data=game_data)
    _put_balanced_pack_target_hand(state)
    state.round_resets.blind_choices["Boss"] = "bl_hook"
    for card in state.deck_cards:
        if card.suit == "Diamonds" and card.rank in {"2", "3", "4", "5", "6", "7", "8", "9"}:
            card.suit = "Hearts"

    assert pack_consumable_use_targets(state, "c_sun") is not None
    assert pack_consumable_use_targets(state, "c_world") is None


def test_pack_suit_tarot_never_overwrites_protected_targets(game_data) -> None:
    state = _full_consumable_pack_state(game_data, "c_sun")
    _put_balanced_pack_target_hand(state)
    state.round_resets.blind_choices["Boss"] = "bl_hook"
    for card in state.hand_cards:
        if card.suit != "Hearts":
            card.seal = "Gold"

    assert pack_consumable_use_targets(state, "c_sun") is None
    assert compute_action_mask(state, SubPhase.BOOSTER_PACK)[ActionRange.PACK_CLAIM_START] == 0


@pytest.mark.parametrize(
    "center_key",
    ("c_world", "c_sun", "c_moon", "c_star", "c_hanged_man", "c_strength"),
)
def test_usable_targeted_pack_tarot_auto_uses_instead_of_banking_with_room(game_data, center_key) -> None:
    state = create_run_state(f"pack_room_{center_key}", 1, "b_red", data=game_data)
    _put_balanced_pack_target_hand(state)
    state.round_resets.blind_choices["Boss"] = "bl_hook"
    _set_single_pack_card(state, center_key)

    planned = pack_consumable_use_targets(state, center_key)
    assert planned is not None
    claimed = claim_pack_card(state, 0)

    assert claimed.auto_used
    assert claimed.auto_used_hand_targets == planned[0]
    assert state.consumables == []


def _make_cryptid_asset(card, *, perma_bonus: int = 20) -> None:
    card.rank = "A"
    card.suit = "Spades"
    card.front_key = "S_A"
    card.center_key = "m_gold"
    card.seal = "Blue"
    card.perma_bonus = perma_bonus


def test_pack_cryptid_duplicates_highest_value_asset_with_exact_mask_execution_parity(game_data) -> None:
    state = _full_consumable_pack_state(game_data, "c_cryptid")
    _put_pack_target_hand(state)
    strong = state.hand_cards[0]
    weak = state.hand_cards[5]
    _make_cryptid_asset(strong)
    weak.center_key = "c_base"
    weak.seal = None
    weak.perma_bonus = 0
    strong.debuff = True
    strong.forced_selection = True

    planned = pack_consumable_use_targets(state, "c_cryptid")
    assert planned == ((0,), ())
    mask = compute_action_mask(state, SubPhase.BOOSTER_PACK)
    claimed = claim_pack_card(state, 0)

    assert mask[ActionRange.PACK_CLAIM_START] == 1
    assert claimed.auto_used_hand_targets == (0,)
    assert len(claimed.use_result.created_cards) == 2
    assert all(card.center_key == "m_gold" for card in claimed.use_result.created_cards)
    assert all(card.seal == "Blue" for card in claimed.use_result.created_cards)
    assert all(card.perma_bonus == 20 for card in claimed.use_result.created_cards)


def test_pack_cryptid_breaks_equal_value_ties_by_lowest_hand_index(game_data) -> None:
    state = create_run_state("pack_cryptid_tie", 1, "b_red", data=game_data)
    _put_pack_target_hand(state)
    _make_cryptid_asset(state.hand_cards[1])
    _make_cryptid_asset(state.hand_cards[4])

    assert pack_consumable_use_targets(state, "c_cryptid") == ((1,), ())


@pytest.mark.parametrize("invalid_field", ("destroyed", "shattered"))
def test_pack_cryptid_skips_invalid_high_value_hand_entries(game_data, invalid_field) -> None:
    state = create_run_state(f"pack_cryptid_invalid_{invalid_field}", 1, "b_red", data=game_data)
    _put_pack_target_hand(state)
    _make_cryptid_asset(state.hand_cards[0], perma_bonus=30)
    setattr(state.hand_cards[0], invalid_field, True)
    _make_cryptid_asset(state.hand_cards[3], perma_bonus=10)

    assert pack_consumable_use_targets(state, "c_cryptid") == ((3,), ())


def test_full_slots_allow_death_with_exact_rightmost_source_targets(game_data) -> None:
    state = _full_consumable_pack_state(game_data, "c_death")
    _put_pack_target_hand(state)
    for card in state.hand_cards:
        card.seal = "Blue"
    target = state.hand_cards[0]
    target.seal = None
    target.center_key = "c_base"
    source = state.hand_cards[-1]
    source.center_key = "m_gold"
    source.seal = "Blue"

    mask = compute_action_mask(state, SubPhase.BOOSTER_PACK)
    claimed = claim_pack_card(state, 0)

    assert mask[ActionRange.PACK_CLAIM_START] == 1
    assert claimed.auto_used
    assert len(claimed.auto_used_hand_targets) == 2
    overwritten, copied = claimed.auto_used_hand_targets
    assert copied > overwritten
    assert state.hand_cards[overwritten].rank == state.hand_cards[copied].rank
    assert state.hand_cards[overwritten].suit == state.hand_cards[copied].suit
    assert state.hand_cards[overwritten].center_key == state.hand_cards[copied].center_key


def test_full_slots_allow_hanged_man_only_for_unprotected_off_plan_cards(game_data) -> None:
    state = _full_consumable_pack_state(game_data, "c_hanged_man")
    _put_pack_target_hand(state)
    for card in state.deck_cards[8:20]:
        card.rank = "A"
        card.suit = "Spades"
    for card in state.hand_cards:
        card.seal = "Purple"
    target = state.hand_cards[0]
    target.rank = "2"
    target.suit = "Hearts"
    target.seal = None
    target_identity = id(target)

    mask = compute_action_mask(state, SubPhase.BOOSTER_PACK)
    claimed = claim_pack_card(state, 0)

    assert mask[ActionRange.PACK_CLAIM_START] == 1
    assert claimed.auto_used
    assert claimed.auto_used_hand_targets == (0,)
    assert all(id(card) != target_identity for card in state.deck_cards)


@pytest.mark.parametrize("center_key", ("c_death", "c_hanged_man"))
def test_full_slots_block_targeted_pack_use_without_safe_target(game_data, center_key) -> None:
    state = _full_consumable_pack_state(game_data, center_key)
    _put_pack_target_hand(state)
    for card in state.hand_cards:
        card.seal = "Blue"

    mask = compute_action_mask(state, SubPhase.BOOSTER_PACK)

    assert mask[ActionRange.PACK_CLAIM_START] == 0
    with pytest.raises(ValueError, match="cannot be claimed"):
        claim_pack_card(state, 0)


def test_targeted_pack_card_banks_with_room_only_when_no_safe_target(game_data) -> None:
    state = create_run_state("pack_target_bank", 1, "b_red", data=game_data)
    _put_pack_target_hand(state)
    for card in state.hand_cards:
        card.seal = "Blue"
    state.pack = PackState(
        booster_key="p_arcana_normal_1",
        state_name="Tarot",
        choices_remaining=1,
        cards=[ShopCard(center_key="c_death", card_type="Tarot", cost=0, base_cost=0)],
    )

    claimed = claim_pack_card(state, 0)

    assert not claimed.auto_used
    assert state.consumables[-1].center_key == "c_death"


def test_targeted_pack_card_auto_uses_with_room_when_safe_target_exists(game_data) -> None:
    state = create_run_state("pack_target_auto_use", 1, "b_red", data=game_data)
    _put_pack_target_hand(state)
    for card in state.hand_cards:
        card.seal = "Blue"
    state.hand_cards[0].seal = None
    state.hand_cards[-1].center_key = "m_gold"
    state.pack = PackState(
        booster_key="p_arcana_normal_1",
        state_name="Tarot",
        choices_remaining=1,
        cards=[ShopCard(center_key="c_death", card_type="Tarot", cost=0, base_cost=0)],
    )

    claimed = claim_pack_card(state, 0)

    assert claimed.auto_used
    assert len(claimed.auto_used_hand_targets) == 2
    assert state.consumables == []


def test_shop_mask_negative_joker_buyable_at_full_slots(game_data):
    state = create_run_state("neg_joker_test", 1, "b_red", data=game_data)
    select_blind(state, "Small")
    start_blind(state, "Small")
    state.current_round.hands_left = 0
    cash_out(state)
    populate_shop(state)

    from pylatro.runtime import joker_limit

    jlimit = joker_limit(state)

    for _ in range(jlimit):
        add_joker(state, "j_joker")

    mask = compute_action_mask(state, SubPhase.SHOP)

    all_items = list(state.shop.cards) + list(state.shop.vouchers) + list(state.shop.boosters)
    for i, item in enumerate(all_items):
        if item.card_type == "Joker":
            is_negative = bool(item.edition and item.edition.get("negative"))
            if item.cost <= state.dollars:
                if is_negative:
                    assert mask[ActionRange.SHOP_BUY_START + i] == 1, "Negative joker should be buyable at full slots"
                else:
                    assert mask[ActionRange.SHOP_BUY_START + i] == 0, (
                        "Non-negative joker should be masked at full slots"
                    )


# ── Debuff-aware play masking ────────────────────────────────────────────────


def _boss_choose_state(game_data, boss_key: str):
    state = create_run_state("test_seed", 1, "b_red", data=game_data)
    state.round_resets.blind_choices["Boss"] = boss_key
    state.blind_on_deck = "Boss"
    select_blind(state, "Boss")
    start_blind(state, "Boss")
    return state


def _play_legal_indices(mask):
    play = mask[ActionRange.PLAY_SUBSET_START : ActionRange.PLAY_SUBSET_END + 1]
    return np.where(play)[0]


def test_psychic_masks_provably_zero_small_plays(game_data):
    from pylatro_agent.subset_actions import subset_indices

    state = _boss_choose_state(game_data, "bl_psychic")
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)

    legal = _play_legal_indices(mask)
    assert legal.size > 0
    # Every remaining play satisfies the must-play-5 debuff
    assert {len(subset_indices(int(i))) for i in legal} == {5}
    # Discards are untouched — single-card discards stay legal
    disc = mask[ActionRange.DISCARD_SUBSET_START : ActionRange.DISCARD_SUBSET_END + 1]
    disc_sizes = {len(subset_indices(int(i))) for i in np.where(disc)[0]}
    assert 1 in disc_sizes


def test_psychic_keeps_plays_legal_when_every_play_would_zero(game_data):
    state = _boss_choose_state(game_data, "bl_psychic")
    del state.hand_cards[4:]  # <5 cards: every play scores 0 but one is owed

    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)

    assert _play_legal_indices(mask).size > 0


def test_eye_masks_already_played_hand_types(game_data):
    from pylatro.scoring import get_poker_hand_info
    from pylatro_agent.subset_actions import subset_indices

    state = _boss_choose_state(game_data, "bl_eye")
    aces = [c for c in state.deck_cards if c.rank == "A"][:2]
    others = [next(c for c in state.deck_cards if c.rank == r) for r in ("2", "7", "9")]
    state.hand_cards = aces + others
    state.eye_hands = {"High Card": True}

    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)

    legal = _play_legal_indices(mask)
    assert legal.size > 0  # the Pair (and Pair-topped supersets) stay legal
    for i in legal:
        cards = [state.hand_cards[j] for j in subset_indices(int(i))]
        hand_name, _, _, _ = get_poker_hand_info(state, cards)
        assert hand_name != "High Card"


def test_mouth_masks_other_hand_types_after_first_play(game_data):
    from pylatro.scoring import get_poker_hand_info
    from pylatro_agent.subset_actions import subset_indices

    state = _boss_choose_state(game_data, "bl_mouth")
    aces = [c for c in state.deck_cards if c.rank == "A"][:2]
    others = [next(c for c in state.deck_cards if c.rank == r) for r in ("2", "7", "9")]
    state.hand_cards = aces + others
    state.mouth_only_hand = "Pair"

    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)

    legal = _play_legal_indices(mask)
    assert legal.size > 0
    for i in legal:
        cards = [state.hand_cards[j] for j in subset_indices(int(i))]
        hand_name, _, _, _ = get_poker_hand_info(state, cards)
        assert hand_name == "Pair"


def test_debuff_mask_probes_preserve_blind_triggered(game_data):
    state = _boss_choose_state(game_data, "bl_eye")
    state.blind_triggered = True

    compute_action_mask(state, SubPhase.CHOOSE_ACTION)

    assert state.blind_triggered is True


def test_small_blind_plays_are_not_debuff_masked(hand_play_state):
    mask = compute_action_mask(hand_play_state, SubPhase.CHOOSE_ACTION)

    play = mask[ActionRange.PLAY_SUBSET_START : ActionRange.PLAY_SUBSET_END + 1]
    disc = mask[ActionRange.DISCARD_SUBSET_START : ActionRange.DISCARD_SUBSET_END + 1]
    assert np.array_equal(play, disc)
