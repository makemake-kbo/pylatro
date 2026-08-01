"""Rule-based heuristic agent for generating supervised pretraining data."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from itertools import combinations, product
from typing import TYPE_CHECKING

import numpy as np

from pylatro import can_use_consumable, get_blind_amount
from pylatro.flow import play_cards
from pylatro.instances import move_joker
from pylatro.runtime import consumable_limit, joker_limit
from pylatro.scoring import RANK_TO_ID, RANK_TO_NOMINAL

from .action import ActionType, encode_action
from .constants import (
    HAND_TARGET_CONSUMABLE_LIMITS,
    JOKER_TARGET_CONSUMABLE_NAMES,
    MAX_CONSUMABLE_HAND_TARGETS,
    MAX_CONSUMABLE_SLOTS,
    MAX_JOKER_SLOTS,
    ActionRange,
    SubPhase,
)
from .subset_actions import consumable_subset_index, subset_index, subset_indices

if TYPE_CHECKING:
    from pylatro.models import JokerInstance, PlayingCard, RunState

# Heuristic buy-priority weights/sets in arbitrary units (higher = more desirable),
# consumed by _score_joker_value. Values are hand-tuned, not derived from game math.

# Jokers treated as low priority to buy.
_LOW_VALUE_JOKERS = frozenset(
    {
        "j_oops",
        "j_chaos",
        "j_credit_card",
        "j_superposition",
        "j_luchador",
        "j_splash",
        "j_invisible",
        "j_faceless",
        "j_matador",
        "j_space",
        "j_seance",
        "j_midas_mask",
        "j_marble",
        "j_hallucination",
        "j_sixth_sense",
        "j_satellite",
        "j_gift",
        "j_trading",
        "j_ring_master",
        "j_certificate",
        "j_cartomancer",
        "j_madness",
        "j_turtle_bean",
        "j_diet_cola",
        "j_erosion",
        "j_vagabond",
        "j_dna",
        "j_8_ball",
    }
)

# Jokers that generate money.
_ECONOMY_JOKERS = frozenset(
    {
        "j_egg",
        "j_mail",
        "j_todo_list",
        "j_golden",
        "j_to_the_moon",
        "j_cloud_9",
        "j_business",
        "j_rocket",
        "j_delayed_grat",
        "j_rough_gem",
    }
)

# Per-joker buy weight for economy jokers.
_ECONOMY_SCORES = {
    "j_egg": 6.0,
    "j_mail": 10.0,
    "j_todo_list": 8.0,
    "j_golden": 12.0,
    "j_to_the_moon": 10.0,
    "j_cloud_9": 7.0,
    "j_business": 6.0,
    "j_rocket": 9.0,
    "j_delayed_grat": 5.0,
    "j_rough_gem": 8.0,
}

# Vouchers worth buying on sight.
_PRIORITY_VOUCHERS = frozenset(
    {
        "v_overstock_norm",
        "v_overstock_plus",
        "v_grabber",
        "v_wasteful",
        "v_hieroglyph",
        "v_hone",
        "v_paint_brush",
        "v_reroll_surplus",
        "v_retcon",
        "v_directors_cut",
        "v_seed_money",
        "v_clearance_sale",
        "v_crystal_ball",
    }
)

# Jokers whose xmult scales over the run.
_SCALING_XMULT_JOKER_KEYS = frozenset(
    {
        "j_ancient",
        "j_baron",
        "j_steel_joker",
        "j_campfire",
        "j_throwback",
        "j_blackboard",
        "j_constellation",
        "j_hologram",
        "j_obelisk",
        "j_glass",
        "j_vampire",
    }
)

# Per-joker buy weight for scaling jokers.
_SCALING_JOKER_SCORES = {
    "j_runner": 14.0,
    "j_green_joker": 16.0,
    "j_ride_the_bus": 12.0,
    "j_square": 12.0,
    "j_fortune_teller": 10.0,
    "j_supernova": 10.0,
    "j_steel_joker": 14.0,
    "j_constellation": 14.0,
    "j_hologram": 14.0,
    "j_campfire": 12.0,
    "j_hit_the_road": 10.0,
    "j_flash": 10.0,
    "j_trousers": 14.0,
    "j_castle": 12.0,
    "j_wee": 12.0,
    "j_red_card": 10.0,
    "j_vampire": 12.0,
    "j_popcorn": 8.0,
}
_SCALING_JOKER_KEYS = frozenset(_SCALING_JOKER_SCORES)

_MID_GAME_ANTE = 3

_ANTE1_SKIP_TAGS = frozenset({"tag_economy", "tag_investment", "tag_rare"})

_RETRIGGER_JOKER_KEYS = frozenset({"j_hanging_chad", "j_sock_and_buskin", "j_selzer", "j_mime", "j_dusk", "j_hack"})

_COPY_JOKER_KEYS = frozenset({"j_blueprint", "j_brainstorm"})
_MAX_JOKER_ORDER_CANDIDATES = 16

# Scoring-profile key sets for jokers whose effect lives in per-joker code
# rather than the numeric config fields (t_chips / mult / Xmult), so the
# generic _joker_summary numerics cannot classify them. Used by the
# build-curve reward shaping to place a joker on the chips → mult → xmult
# curve. Jokers absent from all sets and without config numerics are
# utility/economy and score no build-curve weight.
_CHIPS_PROFILE_JOKER_KEYS = frozenset(
    {
        "j_arrowhead",  # +50 chips per scored spade
        "j_banner",  # +30 chips per remaining discard
        "j_blue_joker",  # +2 chips per card left in deck
        "j_bull",  # +2 chips per dollar
        "j_castle",  # chips grow per discarded card of the daily suit
        "j_hiker",  # permanently grows played cards' chips
        "j_ice_cream",  # +100 chips, melting -5 per hand
        "j_odd_todd",  # +31 chips per odd rank
        "j_runner",  # chips grow per straight played
        "j_scary_face",  # +30 chips per face card
        "j_scholar",  # +20 chips +4 mult per ace
        "j_square",  # chips grow per 4-card hand played
        "j_stone",  # +25 chips per stone card in deck
        "j_stuntman",  # +250 chips, -2 hand size
        "j_walkie_talkie",  # +10 chips +4 mult per 10 or 4
        "j_wee",  # chips grow per scored 2
    }
)
_MULT_PROFILE_JOKER_KEYS = frozenset(
    {
        "j_abstract",  # +3 mult per joker owned
        "j_bootstraps",  # +2 mult per $5 owned
        "j_erosion",  # +4 mult per card below starting deck size
        "j_even_steven",  # +4 mult per even rank
        "j_fibonacci",  # +8 mult per A/2/3/5/8
        "j_flash",  # +2 mult per shop reroll
        "j_fortune_teller",  # +1 mult per tarot used
        "j_gluttenous_joker",  # +3 mult per scored club
        "j_greedy_joker",  # +3 mult per scored diamond
        "j_green_joker",  # +1 mult per hand, -1 per discard
        "j_gros_michel",  # +15 mult (may go extinct)
        "j_half",  # +20 mult on hands of 3 or fewer cards
        "j_lusty_joker",  # +3 mult per scored heart
        "j_misprint",  # +0..23 random mult
        "j_mystic_summit",  # +15 mult at 0 discards
        "j_onyx_agate",  # +7 mult per scored club
        "j_raised_fist",  # mult from lowest held rank
        "j_red_card",  # +3 mult per pack skipped
        "j_ride_the_bus",  # +1 mult per faceless consecutive hand
        "j_shoot_the_moon",  # +13 mult per queen held
        "j_smiley",  # +5 mult per face card
        "j_supernova",  # mult = times hand played
        "j_trousers",  # +2 mult per two-pair played
        "j_wrathful_joker",  # +3 mult per scored spade
    }
)
_XMULT_PROFILE_JOKER_KEYS = frozenset(
    {
        "j_acrobat",  # x3 on final hand
        "j_baseball",  # x1.5 per uncommon joker
        "j_blueprint",  # copies the joker to its right (phase-neutral amplifier)
        "j_brainstorm",  # copies the leftmost joker (phase-neutral amplifier)
        "j_caino",  # xmult per face card destroyed
        "j_drivers_license",  # x3 with 16+ enhanced cards
        "j_flower_pot",  # x3 with all four suits
        "j_hit_the_road",  # xmult per jack discarded this round
        "j_idol",  # x2 per scored copy of the idol card
        "j_lucky_cat",  # xmult grows per lucky trigger
        "j_madness",  # xmult grows per blind (destroys jokers)
        "j_photograph",  # first face card x2
        "j_seeing_double",  # x2 with club + other suit
        "j_stencil",  # x1 per empty joker slot
        "j_triboulet",  # kings/queens x2
        "j_yorick",  # xmult after discards
    }
)


class HeuristicAgent:
    _hand_cache_key: tuple
    _hand_cache_val: set[int]

    def __init__(self) -> None:
        self._hand_cache_key = ()
        self._hand_cache_val = set()
        self._round_progress: dict[str, int] = {}
        self._pending_play_score: dict[str, tuple[int, str, int, int]] = {}
        self._main_type_cache_key: tuple = ()
        self._main_type_cache_val: str = "Pair"
        self._synergy_cache_key: tuple = ()
        self._synergy_cache: dict[str, float] = {}
        self._joker_names_cache_key: tuple = ()
        self._joker_names_cache: frozenset[str] = frozenset()
        self._quality_cache: dict[tuple, str] = {}
        self._score_cache: dict[tuple, int] = {}
        self._score_cache_joker_key: tuple = ()
        self._joker_order_plan_key: tuple = ()
        self._joker_order_plan: tuple[int, ...] = ()
        self._joker_order_expected: tuple[int, ...] = ()
        self._joker_order_play_action = -1

    def _run_key(self, state: RunState) -> str:
        return str(getattr(state, "seed", id(state)))

    def _round_key(self, state: RunState) -> str:
        return f"{self._run_key(state)}:{state.round_resets.ante}:{state.blind_on_deck or ''}"

    def _observe_round_progress(self, state: RunState, round_score: int | None = None) -> None:
        if round_score is not None:
            run_key = self._run_key(state)
            key = self._round_key(state)
            self._round_progress[key] = int(round_score)
            self._pending_play_score.pop(run_key, None)
            current_prefix = f"{run_key}:{state.round_resets.ante}:"
            for old_key in list(self._round_progress):
                if old_key != key or not old_key.startswith(current_prefix):
                    self._round_progress.pop(old_key, None)
            return

        run_key = self._run_key(state)
        pending = self._pending_play_score.pop(run_key, None)
        if pending is not None:
            pending_hands_left, pending_blind, pending_ante, score = pending
            if (
                pending_blind == (state.blind_on_deck or "")
                and pending_ante == state.round_resets.ante
                and state.current_round.hands_left == pending_hands_left - 1
            ):
                key = self._round_key(state)
                self._round_progress[key] = self._round_progress.get(key, 0) + score

        key = self._round_key(state)
        current_prefix = f"{run_key}:{state.round_resets.ante}:"
        for old_key in list(self._round_progress):
            if old_key != key or not old_key.startswith(current_prefix):
                self._round_progress.pop(old_key, None)

    def _current_round_score_estimate(self, state: RunState) -> int:
        return self._round_progress.get(self._round_key(state), 0)

    def _record_selected_action(self, state: RunState, sub_phase: SubPhase, action: int) -> None:
        if sub_phase != SubPhase.CHOOSE_ACTION:
            return
        if ActionRange.PLAY_SUBSET_START <= action <= ActionRange.PLAY_SUBSET_END:
            rel_action = action - ActionRange.PLAY_SUBSET_START
            indices = tuple(i for i in subset_indices(rel_action) if i < len(state.hand_cards))
            score = self._score_play_for_progress(state, indices)
            self._pending_play_score[self._run_key(state)] = (
                state.current_round.hands_left,
                state.blind_on_deck or "",
                state.round_resets.ante,
                score,
            )

    def _score_play_for_progress(self, state: RunState, indices: tuple[int, ...]) -> int:
        if not indices:
            return 0
        approx = self._estimate_hand_score(state, indices)
        return max(0, int(approx * 0.75))

    def _cached_best_hand(self, state: RunState, hand: list[PlayingCard]) -> set[int]:
        key = (
            tuple(
                (
                    card.rank,
                    card.suit,
                    card.center_key,
                    card.debuff,
                    card.face_down,
                )
                for card in hand
            ),
            tuple(joker.center_key for joker in state.jokers),
        )
        if key == self._hand_cache_key:
            return set(self._hand_cache_val)
        result = set(self._find_best_hand(state, hand))
        self._hand_cache_key = key
        self._hand_cache_val = result
        return set(result)

    def _get_blind_target(self, state: RunState) -> int:
        blind = state.round_resets.blind
        if blind is None:
            return 0
        ante = state.round_resets.ante
        scaling = min(state.stake, 3)
        base = get_blind_amount(ante, scaling)
        mult = blind.get("mult", 1)
        return int(base * mult)

    def _estimate_hand_score(self, state: RunState, hand_indices: tuple[int, ...]) -> int:
        if not hand_indices:
            return 0
        joker_key = self._joker_keys_sig(state)
        cache_key = (
            hand_indices,
            joker_key,
            self._hands_sig(state),
            state.dollars,
            state.current_round.discards_left,
            state.current_round.hands_left,
            state.round_resets.ante,
        )
        if joker_key != self._score_cache_joker_key:
            self._score_cache.clear()
            self._score_cache_joker_key = joker_key
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]
        result = self._estimate_hand_score_compute(state, hand_indices)
        if len(self._score_cache) > 2000:
            self._score_cache.clear()
        self._score_cache[cache_key] = result
        return result

    def _estimate_hand_score_compute(self, state: RunState, hand_indices: tuple[int, ...]) -> int:
        # Approximates the full scoring pipeline for the given played cards:
        # base chips/mult -> per-card chips -> retriggers -> per-joker effects -> xmult.
        # Probabilistic jokers are valued by their expected (rather than rolled) contribution.
        hand = state.hand_cards
        cards = [hand[i] for i in hand_indices]
        hand_type = self._quick_hand_quality(state, cards)
        hand_info = state.hands.get(hand_type, {})
        base_chips = hand_info.get("chips", 0)
        base_mult = hand_info.get("mult", 0)

        # number of played cards that contribute chips for each poker hand type
        _SCORING_COUNT = {
            "High Card": 1,
            "Pair": 2,
            "Two Pair": 4,
            "Three of a Kind": 3,
            "Straight": 5,
            "Flush": 5,
            "Full House": 5,
            "Four of a Kind": 4,
            "Five of a Kind": 5,
            "Straight Flush": 5,
            "Flush Five": 5,
            "Flush House": 5,
        }
        n_scoring = _SCORING_COUNT.get(hand_type, len(cards))

        card_chip_values = []
        centers_get = state.data.centers.get
        for c in cards:
            center = centers_get(c.center_key)
            effect = center.get("effect", "") if center else ""
            if effect == "Stone Card":
                card_chip_values.append(50 + c.perma_bonus)
            else:
                card_chip_values.append(RANK_TO_NOMINAL.get(c.rank, 0) + c.perma_bonus)
        card_chip_values.sort(reverse=True)
        card_chips = sum(card_chip_values[:n_scoring])

        total_chips = base_chips + card_chips
        total_mult = base_mult
        x_mult_acc = 1.0

        scoring_cards = cards[:n_scoring]
        scoring_ranks = tuple(c.rank for c in scoring_cards)
        scoring_suits = tuple(c.suit for c in scoring_cards)
        held_cards = [hand[i] for i in range(len(hand)) if i not in hand_indices]

        dollars = state.dollars
        discards_left = state.current_round.discards_left
        ante = state.round_resets.ante

        scoring_card_chips = card_chip_values[:n_scoring]
        retriggers = [0] * n_scoring
        joker_centers_by_name: dict[str, JokerInstance] = {}
        for jj in state.jokers:
            if jj.debuff:
                continue
            cc = centers_get(jj.center_key)
            if cc:
                joker_centers_by_name[cc.get("name", "")] = jj
        last_hand = state.current_round.hands_left == 1
        if "Hanging Chad" in joker_centers_by_name and n_scoring > 0:
            retriggers[0] += 2
        if "Sock and Buskin" in joker_centers_by_name:
            for i, c in enumerate(scoring_cards):
                if c.rank in ("J", "Q", "K"):
                    retriggers[i] += 1
        if "Hack" in joker_centers_by_name:
            for i, c in enumerate(scoring_cards):
                if c.rank in ("2", "3", "4", "5"):
                    retriggers[i] += 1
        if "Seltzer" in joker_centers_by_name:
            sel = joker_centers_by_name["Seltzer"]
            rem = 0
            if isinstance(sel.extra, dict):
                rem = int(sel.extra.get("hands", 0) or 0)
            elif isinstance(sel.extra, (int, float)):
                rem = int(sel.extra)
            if rem > 0:
                for i in range(n_scoring):
                    retriggers[i] += 1
        if "Dusk" in joker_centers_by_name and last_hand:
            for i in range(n_scoring):
                retriggers[i] += 1

        for i, n in enumerate(retriggers):
            if n > 0 and i < len(scoring_card_chips):
                total_chips += scoring_card_chips[i] * n

        for j in state.jokers:
            if j.debuff:
                continue

            center = centers_get(j.center_key)
            if not center:
                continue
            jname = center.get("name", "")

            if j.mult:
                total_mult += j.mult

            if j.t_mult and (not j.type or j.type == hand_type):
                total_mult += j.t_mult

            if j.t_chips and (not j.type or j.type == hand_type):
                total_chips += j.t_chips

            if j.x_mult and j.x_mult > 1 and (not j.type or j.type == hand_type):
                if jname == "Obelisk":
                    play_more_than = int(hand_info.get("played", 0) if hand_info else 0)
                    resets = True
                    for other_name, other_hand in state.hands.items():
                        if (
                            other_name != hand_type
                            and other_hand.get("visible", True)
                            and int(other_hand.get("played", 0) or 0) > play_more_than
                        ):
                            resets = False
                            break
                    if not resets:
                        x_mult_acc *= max(float(j.x_mult) + float(j.extra or 0), 1.0)
                else:
                    x_mult_acc *= j.x_mult

            extra = j.extra
            if isinstance(extra, dict):
                em = extra.get("mult")
                if isinstance(em, (int, float)) and em:
                    total_mult += em
                ec = extra.get("chips")
                if isinstance(ec, (int, float)) and ec:
                    total_chips += ec
                exm = extra.get("Xmult")
                if isinstance(exm, (int, float)) and exm and exm > 1:
                    if jname == "Cavendish" or (jname == "Card Sharp" and hand_info.get("played_this_round", 0) > 1):
                        x_mult_acc *= exm
                    elif jname == "Loyalty Card":
                        every = int(extra.get("every", 0) or 0)
                        if every > 0:
                            remaining = (every - 1 - (state.hands_played - j.hands_played_at_create)) % (every + 1)
                            if remaining == every:
                                x_mult_acc *= exm

            if jname == "Fibonacci":
                for r in scoring_ranks:
                    if r in ("2", "3", "5", "8", "A"):
                        total_mult += 8
            elif jname == "Even Steven":
                for r in scoring_ranks:
                    if r in ("2", "4", "6", "8", "T"):
                        total_mult += 4
            elif jname == "Odd Todd":
                for r in scoring_ranks:
                    if r in ("A", "3", "5", "7", "9"):
                        total_chips += 31
            elif jname == "Scary Face":
                for r in scoring_ranks:
                    if r in ("J", "Q", "K"):
                        total_chips += 30
            elif jname == "Smiley Face":
                for r in scoring_ranks:
                    if r in ("J", "Q", "K"):
                        total_mult += 5
            elif jname == "Scholar":
                for r in scoring_ranks:
                    if r == "A":
                        total_chips += 20
                        total_mult += 4
            elif jname == "Walkie Talkie":
                for r in scoring_ranks:
                    if r in ("4", "T"):
                        total_chips += 10
                        total_mult += 4
            elif jname == "Photograph":
                face_found = False
                for r in scoring_ranks:
                    if r in ("J", "Q", "K") and not face_found:
                        x_mult_acc *= 2
                        face_found = True
            elif jname in ("Greedy Joker", "Lusty Joker", "Wrathful Joker", "Gluttonous Joker"):
                suit_map = {
                    "Greedy Joker": "Diamonds",
                    "Lusty Joker": "Hearts",
                    "Wrathful Joker": "Spades",
                    "Gluttonous Joker": "Clubs",
                }
                target_suit = suit_map[jname]
                sv = 3 if isinstance(j.extra, dict) else (j.extra or 3)
                for s in scoring_suits:
                    if s == target_suit:
                        total_mult += sv
            elif jname == "Onyx Agate":
                for s in scoring_suits:
                    if s == "Clubs":
                        total_mult += 7
            elif jname == "Arrowhead":
                for s in scoring_suits:
                    if s == "Spades":
                        total_chips += 50
            elif jname == "Shoot the Moon":
                for c in held_cards:
                    if c.rank == "Q":
                        total_mult += 13
            elif jname == "Baron":
                for c in held_cards:
                    if c.rank == "K":
                        x_mult_acc *= 1.5
            elif jname == "Bootstraps":
                total_mult += (dollars // 5) * 2
            elif jname == "Half Joker":
                if len(cards) <= 3:
                    total_mult += 20
            elif jname == "Mystic Summit":
                if discards_left == 0:
                    total_mult += 15
            elif jname == "Banner":
                total_chips += discards_left * 30
            elif jname == "Abstract Joker":
                total_mult += len([jj for jj in state.jokers if not jj.debuff])
            elif jname == "Supernova":
                played = hand_info.get("played", 0) if hand_info else 0
                total_mult += played
            elif jname == "Green Joker":
                total_mult += max(ante * 2, 0)
            elif jname == "Runner":
                total_chips += ante * 10
            elif jname == "Square Joker":
                total_chips += ante * 4
            elif jname == "Ride the Bus" or jname == "Fortune Teller":
                total_mult += ante * 2
            elif jname == "Flash":
                total_chips += ante * 10
            elif jname == "Constellation":
                x_mult_acc *= 1 + ante * 0.1
            elif jname == "Hologram":
                x_mult_acc *= 1 + ante * 0.15
            elif jname == "Bull":
                total_chips += dollars * 2 * n_scoring
            elif jname == "Triboulet":
                xm = float(j.extra) if isinstance(j.extra, (int, float)) else 2.0
                for r in scoring_ranks:
                    if r in ("K", "Q"):
                        x_mult_acc *= xm
            elif jname == "Bloodstone":
                if isinstance(j.extra, dict):
                    xm = float(j.extra.get("Xmult", 1.5))
                    odds = float(j.extra.get("odds", 2)) or 2.0
                    p = 1.0 / odds
                    # Bloodstone procs on Hearts; value the flip at its expectation.
                    for s in scoring_suits:
                        if s == "Hearts":
                            x_mult_acc *= 1.0 + (xm - 1.0) * p
            elif jname == "The Idol":
                idol = state.current_round.idol_card if hasattr(state, "current_round") else None
                if idol:
                    xm = float(j.extra) if isinstance(j.extra, (int, float)) else 2.0
                    for c in scoring_cards:
                        if c.rank == idol.get("rank") and c.suit == idol.get("suit"):
                            x_mult_acc *= xm
            elif jname == "Ancient Joker":
                anc = state.current_round.ancient_card if hasattr(state, "current_round") else None
                if anc:
                    xm = float(j.extra) if isinstance(j.extra, (int, float)) else 1.5
                    for s in scoring_suits:
                        if s == anc.get("suit"):
                            x_mult_acc *= xm
            elif jname == "Reserved Parking":
                if isinstance(j.extra, dict):
                    dollar_extra = float(j.extra.get("dollars", 1))
                    odds = float(j.extra.get("odds", 2)) or 2.0
                    # held face cards earn $1 each w/ p=1/odds, value as +1 mult expectation per face
                    held_faces = sum(1 for c in held_cards if c.rank in ("J", "Q", "K"))
                    total_mult += int(held_faces * dollar_extra / odds)
            elif jname == "Blackboard":
                if held_cards and all(c.suit in ("Spades", "Clubs") for c in held_cards):
                    xm = float(j.extra) if isinstance(j.extra, (int, float)) else 3.0
                    x_mult_acc *= xm
            elif jname == "Driver's License":
                enhanced = sum(
                    1
                    for c in state.deck_cards
                    if (centers_get(c.center_key) or {}).get("effect") not in ("", None, "Base")
                )
                if enhanced >= 16:
                    xm = float(j.extra) if isinstance(j.extra, (int, float)) else 3.0
                    x_mult_acc *= xm
            elif jname == "Joker Stencil":
                empty_slots = max(joker_limit(state) - len(state.jokers), 0)
                if empty_slots > 0:
                    x_mult_acc *= 1.0 + empty_slots
            elif jname == "Acrobat":
                if last_hand:
                    xm = float(j.extra) if isinstance(j.extra, (int, float)) else 3.0
                    x_mult_acc *= xm
            elif jname == "Throwback":
                per = float(j.extra) if isinstance(j.extra, (int, float)) else 0.25
                skips = int(getattr(state, "skips", 0) or 0)
                if skips > 0:
                    x_mult_acc *= 1.0 + per * skips
            elif jname == "Erosion":
                per = float(j.extra) if isinstance(j.extra, (int, float)) else 4.0
                start = int(getattr(state, "starting_deck_size", 52) or 52)
                missing = max(start - len(state.deck_cards), 0)
                total_mult += int(per * missing)
            elif jname == "Canio":
                xm = float(j.caino_xmult) if j.caino_xmult else 1.0
                if xm > 1.0:
                    x_mult_acc *= xm
            elif jname == "Stone Joker":
                per = float(j.extra) if isinstance(j.extra, (int, float)) else 25.0
                tally = int(j.stone_tally or 0)
                if tally > 0:
                    total_chips += int(per * tally)
            elif jname == "Steel Joker":
                per = float(j.extra) if isinstance(j.extra, (int, float)) else 0.2
                tally = int(j.steel_tally or 0)
                if tally > 0:
                    x_mult_acc *= 1.0 + per * tally
            elif jname == "Misprint":
                if isinstance(j.extra, dict):
                    lo = float(j.extra.get("min", 0))
                    hi = float(j.extra.get("max", 23))
                    total_mult += (lo + hi) / 2.0
            elif jname in ("Yorick", "Ramen"):
                xm = j.x_mult if j.x_mult and j.x_mult > 1 else 1.0
                if xm > 1.0:
                    x_mult_acc *= xm
            elif jname == "Swashbuckler":
                # +1 mult per dollar of total sell value of other jokers
                sell_total = sum(int(jj.sell_cost or 0) for jj in state.jokers if jj is not j and not jj.debuff)
                if sell_total > 0:
                    total_mult += sell_total
            elif jname == "Baseball Card":
                # x1.5 per uncommon joker, uses uncommon rarity from center
                xm = float(j.extra) if isinstance(j.extra, (int, float)) else 1.5
                count = 0
                for jj in state.jokers:
                    if jj is j or jj.debuff:
                        continue
                    cc = centers_get(jj.center_key) or {}
                    if cc.get("rarity") == 2 or cc.get("rarity") == "Uncommon":
                        count += 1
                if count > 0:
                    x_mult_acc *= xm**count

        return int(total_chips * total_mult * x_mult_acc)

    def _get_joker_names(self, state: RunState) -> frozenset[str]:
        keys = state.joker_keys
        if keys != self._joker_names_cache_key:
            self._joker_names_cache = frozenset(state.data.centers[k]["name"] for k in keys)
            self._joker_names_cache_key = keys
        return self._joker_names_cache

    def _hashable_extra(self, value) -> tuple | str | int | float | bool | None:
        if isinstance(value, dict):
            return tuple(sorted((k, self._hashable_extra(v)) for k, v in value.items()))
        if isinstance(value, list):
            return tuple(self._hashable_extra(v) for v in value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    def _joker_keys_sig(self, state: RunState) -> tuple:
        return tuple(
            (
                j.center_key,
                j.debuff,
                j.mult,
                j.h_mult,
                j.h_x_mult,
                j.t_mult,
                j.t_chips,
                j.x_mult,
                j.h_size,
                j.d_size,
                j.extra_value,
                j.type,
                self._hashable_extra(j.extra),
                j.caino_xmult,
                j.yorick_discards,
                j.loyalty_remaining,
                j.driver_tally,
                j.stone_tally,
                j.steel_tally,
                j.money,
            )
            for j in state.jokers
        )

    def _hands_sig(self, state: RunState) -> tuple:
        return tuple(
            (
                hand_name,
                hand.get("level", 1),
                hand.get("chips", 0),
                hand.get("mult", 0),
                hand.get("played", 0),
                hand.get("played_this_round", 0),
                hand.get("visible", True),
            )
            for hand_name, hand in sorted(state.hands.items())
        )

    def _get_main_hand_type(self, state: RunState) -> str:
        sig = (self._joker_keys_sig(state), self._hands_sig(state))
        if sig == self._main_type_cache_key:
            return self._main_type_cache_val
        best = "Pair"
        best_score = 1.0
        for ht in (
            "Pair",
            "Two Pair",
            "Three of a Kind",
            "Full House",
            "Flush",
            "Straight",
            "High Card",
            "Four of a Kind",
            "Straight Flush",
            "Five of a Kind",
            "Flush House",
            "Flush Five",
        ):
            synergy = self._hand_type_synergy(state, ht)
            ht_info = state.hands.get(ht)
            played = ht_info.get("played", 0) if ht_info else 0
            level = ht_info.get("level", 1) if ht_info else 1
            score = synergy * 3 + played * 1.0 + level * 5
            if ht == "Pair":
                score += 3.0
            elif ht in ("Two Pair", "Three of a Kind"):
                score += 1.0
            if score > best_score:
                best_score = score
                best = ht
        self._main_type_cache_key = sig
        self._main_type_cache_val = best
        return best

    def _find_type_hand(self, state: RunState, hand: list[PlayingCard], type_name: str) -> tuple[int, ...] | None:
        if not hand:
            return None

        hand_len = len(hand)
        nominals = [RANK_TO_NOMINAL.get(c.rank, 0) for c in hand]

        by_rank: dict[str, list[int]] = {}
        for i, card in enumerate(hand):
            by_rank.setdefault(card.rank, []).append(i)

        if type_name == "Pair":
            for _rank, indices in sorted(by_rank.items(), key=lambda x: RANK_TO_NOMINAL.get(x[0], 0), reverse=True):
                if len(indices) >= 2:
                    pair = indices[:2]
                    pair_set = set(pair)
                    kickers = sorted(
                        [i for i in range(hand_len) if i not in pair_set],
                        key=lambda i: nominals[i],
                        reverse=True,
                    )
                    return tuple(sorted(pair + kickers[:3]))

        elif type_name == "Two Pair":
            pairs: list[int] = []
            for _rank, indices in sorted(by_rank.items(), key=lambda x: RANK_TO_NOMINAL.get(x[0], 0), reverse=True):
                if len(indices) >= 2 and len(pairs) < 4:
                    pairs.extend(indices[:2])
            if len(pairs) >= 4:
                pairs_set = set(pairs)
                kickers = sorted(
                    [i for i in range(hand_len) if i not in pairs_set],
                    key=lambda i: nominals[i],
                    reverse=True,
                )
                return tuple(sorted(pairs + kickers[:1]))

        elif type_name == "Three of a Kind":
            for _rank, indices in sorted(by_rank.items(), key=lambda x: RANK_TO_NOMINAL.get(x[0], 0), reverse=True):
                if len(indices) >= 3:
                    trip = indices[:3]
                    trip_set = set(trip)
                    kickers = sorted(
                        [i for i in range(hand_len) if i not in trip_set],
                        key=lambda i: nominals[i],
                        reverse=True,
                    )
                    return tuple(sorted(trip + kickers[:2]))

        elif type_name == "Four of a Kind":
            for _rank, indices in sorted(by_rank.items(), key=lambda x: RANK_TO_NOMINAL.get(x[0], 0), reverse=True):
                if len(indices) >= 4:
                    quad = indices[:4]
                    quad_set = set(quad)
                    kickers = sorted(
                        [i for i in range(hand_len) if i not in quad_set],
                        key=lambda i: nominals[i],
                        reverse=True,
                    )
                    return tuple(sorted(quad + kickers[:1]))

        elif type_name == "Flush":
            by_suit: dict[str, list[int]] = {}
            centers = state.data.centers
            for i, card in enumerate(hand):
                center = centers.get(card.center_key)
                effect = center.get("effect", "") if center else ""
                if effect == "Wild Card":
                    for s in ("Spades", "Hearts", "Clubs", "Diamonds"):
                        by_suit.setdefault(s, []).append(i)
                else:
                    by_suit.setdefault(card.suit, []).append(i)
            for indices in by_suit.values():
                unique = list(dict.fromkeys(indices))
                if len(unique) >= 5:
                    unique.sort(key=lambda i: nominals[i], reverse=True)
                    return tuple(sorted(unique[:5]))

        return None

    def select_action(self, state: RunState, sub_phase: SubPhase, action_mask: np.ndarray, **kwargs) -> int:
        round_score = kwargs.get("round_score")
        self._observe_round_progress(state, int(round_score) if round_score is not None else None)
        if sub_phase == SubPhase.BLIND_SELECT:
            action = self._blind_select(state, action_mask)
        elif sub_phase == SubPhase.CHOOSE_ACTION:
            action = self._choose_action(state, action_mask)
        elif sub_phase == SubPhase.SHOP:
            action = self._shop(state, action_mask)
        elif sub_phase == SubPhase.BOOSTER_PACK:
            action = self._booster_pack(state, action_mask)
        else:
            action = self._random_valid(action_mask)
        if round_score is None:
            self._record_selected_action(state, sub_phase, action)
        return action

    def _blind_select(self, state: RunState, mask: np.ndarray) -> int:
        ante = state.round_resets.ante
        if ante == 1 and mask[ActionRange.BLIND_SKIP]:
            blind_on_deck = state.blind_on_deck or ""
            upcoming_tag = state.round_resets.blind_tags.get(blind_on_deck, "")
            if upcoming_tag in _ANTE1_SKIP_TAGS:
                return ActionRange.BLIND_SKIP
        if mask[ActionRange.BLIND_PLAY]:
            return ActionRange.BLIND_PLAY
        return self._random_valid(mask)

    def _choose_action(self, state: RunState, mask: np.ndarray) -> int:
        # Decision order (first applicable wins):
        #   1. use money tarots (Hermit/Temperance)
        #   2. pick best hand, optionally pad short hands up to 5 cards
        #   3. if it can't win now, search for a better alt-type hand
        #   4. use the best-scoring planet card
        #   5. use buff/destroy/utility tarots
        #   6. play if it wins now or wins across remaining hands
        #   7. otherwise discard to draw / discard worst cards, else play/discard fallback
        hand = state.hand_cards
        if not hand:
            return self._random_valid(mask)
        pending_joker_order = self._resume_joker_order_plan(state, mask)
        if pending_joker_order is not None:
            return pending_joker_order

        _MONEY_TAROTS = frozenset({"The Hermit", "Temperance"})
        for slot, cons in enumerate(state.consumables[:MAX_CONSUMABLE_SLOTS]):
            center = state.data.centers[cons.center_key]
            if center.get("set", "") != "Tarot":
                continue
            name = center.get("name", "")
            if name in _MONEY_TAROTS:
                action = self._atomic_consumable_action(state, slot, mask)
                if action is not None:
                    return action

        best_play = tuple(sorted(self._cached_best_hand(state, hand)))

        if best_play and len(best_play) < 5 and len(hand) > len(best_play):
            remaining = sorted(
                [i for i in range(len(hand)) if i not in best_play],
                key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0),
                reverse=True,
            )
            padded = tuple(sorted(list(best_play) + remaining[: 5 - len(best_play)]))
            orig_score = self._estimate_hand_score(state, best_play)
            pad_score = self._estimate_hand_score(state, padded)
            if pad_score >= orig_score:
                padded_action = ActionRange.PLAY_SUBSET_START + subset_index(padded)
                if mask[padded_action]:
                    best_play = padded

        best_play_cards = [hand[i] for i in best_play] if best_play else []
        best_hand_quality_now = self._quick_hand_quality(state, best_play_cards) if best_play_cards else ""

        est_score = self._estimate_hand_score(state, best_play) if best_play else 0

        blind_target = self._get_blind_target(state)
        target_remaining = max(0, blind_target - self._current_round_score_estimate(state))
        hands_left = state.current_round.hands_left
        discards_left = state.current_round.discards_left
        can_win_now = est_score >= target_remaining

        main_type = self._get_main_hand_type(state)
        main_synergy = self._hand_type_synergy(state, main_type)

        if not can_win_now:
            alt_types = ["Pair", "Two Pair", "Three of a Kind", "Four of a Kind", "Flush"]
            if main_type in alt_types:
                alt_types.remove(main_type)
                alt_types.insert(0, main_type)

            if main_type in ("Pair", "Two Pair", "Three of a Kind"):
                main_hand = self._find_type_hand(state, hand, main_type)
                if main_hand:
                    main_score = self._estimate_hand_score(state, main_hand)
                    if main_score >= est_score * 0.8:
                        best_play = main_hand
                        est_score = main_score
                        best_play_cards = [hand[i] for i in best_play]
                        best_hand_quality_now = self._quick_hand_quality(state, best_play_cards)

            for alt_type in alt_types:
                if best_hand_quality_now == alt_type:
                    continue
                alt_hand = self._find_type_hand(state, hand, alt_type)
                if alt_hand:
                    alt_score = self._estimate_hand_score(state, alt_hand)
                    if alt_type == main_type and main_synergy > 0:
                        if alt_score > est_score * 0.7:
                            best_play = alt_hand
                            est_score = alt_score
                            best_play_cards = [hand[i] for i in best_play]
                            best_hand_quality_now = self._quick_hand_quality(state, best_play_cards)
                    elif alt_score > est_score:
                        best_play = alt_hand
                        est_score = alt_score
                        best_play_cards = [hand[i] for i in best_play]
                        best_hand_quality_now = self._quick_hand_quality(state, best_play_cards)

        best_planet_score = -1
        best_planet_action = None
        for slot, cons in enumerate(state.consumables[:MAX_CONSUMABLE_SLOTS]):
            center = state.data.centers[cons.center_key]
            if center.get("set", "") != "Planet":
                continue
            action = self._atomic_consumable_action(state, slot, mask)
            if action is None:
                continue
            planet_hand_type = center.get("config", {}).get("hand_type", "")
            score = 0
            if planet_hand_type == main_type:
                score += 20
            if planet_hand_type == best_hand_quality_now:
                score += 15
            score += self._hand_type_synergy(state, planet_hand_type)
            played_count = state.hands.get(planet_hand_type, {}).get("played", 0)
            if played_count > 0:
                score += 10
            if score == 0:
                score = 1
            if score > best_planet_score:
                best_planet_score = score
                best_planet_action = action
        if best_planet_action is not None:
            return best_planet_action

        _BUFF_TAROTS = frozenset(
            {
                "The Magician",
                "The Empress",
                "The Hierophant",
                "The Lovers",
                "The Chariot",
                "Justice",
                "Strength",
                "The Star",
                "The Moon",
                "The Sun",
                "The World",
                "The Wheel of Fortune",
            }
        )
        _DESTROY_TAROTS = frozenset(
            {
                "Death",
                "The Hanged Man",
                "The Devil",
                "The Tower",
                "Judgement",
            }
        )

        for slot, cons in enumerate(state.consumables[:MAX_CONSUMABLE_SLOTS]):
            center = state.data.centers[cons.center_key]
            if center.get("set", "") != "Tarot":
                continue
            name = center.get("name", "")
            preferred = None
            if name in _BUFF_TAROTS:
                preferred = best_play if best_play else None
            elif name in _DESTROY_TAROTS:
                worst = tuple(sorted(self._find_worst_cards(state, hand, max_discard=2)))
                preferred = worst if worst else None
            elif name in ("The Fool", "The Emperor", "The High Priestess", "Judgement"):
                action = self._atomic_consumable_action(state, slot, mask)
                if action is not None:
                    return action
                continue
            else:
                continue
            action = self._atomic_consumable_action(state, slot, mask, preferred_indices=preferred)
            if action is not None:
                return action

        if best_play and est_score >= target_remaining:
            play_action = ActionRange.PLAY_SUBSET_START + subset_index(best_play)
            if mask[play_action]:
                return self._play_or_reorder_jokers(state, mask, play_action)

        if best_play and est_score * hands_left >= target_remaining:
            play_action = ActionRange.PLAY_SUBSET_START + subset_index(best_play)
            if mask[play_action]:
                return self._play_or_reorder_jokers(state, mask, play_action)

        draw_discard = self._should_discard_for_draw(state, hand, mask)
        if draw_discard is not None and est_score * max(hands_left - 1, 1) < target_remaining:
            return ActionRange.DISCARD_SUBSET_START + subset_index(draw_discard)

        best_discard = tuple(sorted(self._find_worst_cards(state, hand, max_discard=min(5, len(hand))))) if hand else ()
        can_discard = (
            bool(best_discard)
            and discards_left > 0
            and mask[ActionRange.DISCARD_SUBSET_START + subset_index(best_discard)]
        )

        if (
            can_discard
            and hands_left > 1
            and est_score * hands_left < target_remaining
            and not self._has_scaling_jokers(state)
        ):
            return ActionRange.DISCARD_SUBSET_START + subset_index(best_discard)

        if best_play:
            play_action = ActionRange.PLAY_SUBSET_START + subset_index(best_play)
            if mask[play_action]:
                return self._play_or_reorder_jokers(state, mask, play_action)

        if can_discard:
            return ActionRange.DISCARD_SUBSET_START + subset_index(best_discard)

        valid_play = np.where(mask[ActionRange.PLAY_SUBSET_START : ActionRange.PLAY_SUBSET_END + 1] == 1)[0]
        if len(valid_play) > 0:
            play_action = ActionRange.PLAY_SUBSET_START + int(valid_play[0])
            return self._play_or_reorder_jokers(state, mask, play_action)
        valid_discard = np.where(mask[ActionRange.DISCARD_SUBSET_START : ActionRange.DISCARD_SUBSET_END + 1] == 1)[0]
        if len(valid_discard) > 0:
            return ActionRange.DISCARD_SUBSET_START + int(valid_discard[0])
        return self._random_valid(mask)

    def _quick_hand_quality(self, state: RunState, cards: list[PlayingCard]) -> str:
        n = len(cards)
        if n == 0:
            return "High Card"
        ranks = tuple(RANK_TO_ID[c.rank] for c in cards)
        suits = tuple(c.suit for c in cards)
        cache_key = (ranks, suits)
        if cache_key in self._quality_cache:
            return self._quality_cache[cache_key]
        rank_counts: dict[int, int] = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        counts = sorted(rank_counts.values(), reverse=True)

        is_flush = n >= 5 and len(set(suits)) == 1
        sorted_r = sorted(set(ranks))
        is_straight = False
        if len(sorted_r) >= 5 and len(sorted_r) == n:
            is_straight = sorted_r[len(sorted_r) - 1] - sorted_r[0] == n - 1
            if not is_straight and 14 in sorted_r:
                low = sorted(set(1 if r == 14 else r for r in ranks))
                if len(low) >= 5 and len(low) == n and low[len(low) - 1] - low[0] == n - 1:
                    is_straight = True

        if is_flush and is_straight:
            if counts[0] >= 4:
                result = "Flush Five"
            elif counts[0] >= 3 and len(counts) >= 2 and counts[1] >= 2:
                result = "Flush House"
            else:
                result = "Straight Flush"
        elif counts[0] >= 5:
            result = "Five of a Kind"
        elif counts[0] >= 4:
            result = "Four of a Kind"
        elif counts[0] >= 3 and len(counts) >= 2 and counts[1] >= 2:
            result = "Full House"
        elif is_flush:
            result = "Flush"
        elif is_straight:
            result = "Straight"
        elif counts[0] >= 3:
            result = "Three of a Kind"
        elif len(counts) >= 2 and counts[0] >= 2 and counts[1] >= 2:
            result = "Two Pair"
        elif counts[0] >= 2:
            result = "Pair"
        else:
            result = "High Card"

        if len(self._quality_cache) > 2000:
            self._quality_cache.clear()
        self._quality_cache[cache_key] = result
        return result

    @staticmethod
    def _edition_multiplier(edition: dict | None) -> float:
        if not edition:
            return 1.0
        if edition.get("negative"):
            return 2.5
        if edition.get("polychrome"):
            return 1.6
        if edition.get("holo"):
            return 1.3
        if edition.get("foil"):
            return 1.15
        return 1.0

    def _score_joker_value(
        self,
        state: RunState,
        center_key: str,
        edition: dict | None = None,
    ) -> float:
        center = state.data.centers.get(center_key, {})
        config = center.get("config")
        if not config or isinstance(config, list):
            config = {}

        cost = center.get("cost", 5)
        score = 0.0
        jname = center.get("name", "")
        ante = state.round_resets.ante
        mid_game = ante >= _MID_GAME_ANTE

        if center_key in _LOW_VALUE_JOKERS:
            return -100.0

        if center_key == "j_stone":
            has_hologram = any(j.center_key == "j_hologram" for j in state.jokers)
            if not has_hologram:
                return -100.0

        if center_key in _ECONOMY_JOKERS:
            eco_score = _ECONOMY_SCORES.get(center_key, 5.0)
            if ante <= 3:
                eco_score *= 1.5
            if cost <= 4:
                eco_score *= 1.3
            score += eco_score

        main_type = self._get_main_hand_type(state)
        has_main_synergy = len(state.jokers) > 0 and self._hand_type_synergy(state, main_type) > 0

        mult = config.get("mult")
        if isinstance(mult, (int, float)) and mult:
            additive_mult_count = sum(1 for j in state.jokers if j.mult and j.mult > 0)
            if additive_mult_count < 2:
                score += mult * (11.0 if mid_game else 14.0)
            elif additive_mult_count < 4:
                score += mult * (7.0 if mid_game else 10.0)
            else:
                score += mult * (3.0 if mid_game else 5.0)

        t_mult = config.get("t_mult")
        hand_type = config.get("type", "")
        _COMMITTED_TYPES = {"Pair", "High Card", "Two Pair", "Three of a Kind"}
        if isinstance(t_mult, (int, float)) and t_mult:
            if hand_type == main_type:
                score += t_mult * 12.0
            elif self._hand_type_synergy(state, hand_type) > 0 or (
                hand_type in _COMMITTED_TYPES and not has_main_synergy
            ):
                score += t_mult * 6.0
            elif hand_type in _COMMITTED_TYPES:
                score += t_mult * 2.0
            else:
                score -= t_mult * 1.0

        t_chips = config.get("t_chips")
        if isinstance(t_chips, (int, float)) and t_chips:
            chip_scale = 0.45 if mid_game else 1.0
            if hand_type == main_type:
                score += t_chips * 2.5 * chip_scale
            elif self._hand_type_synergy(state, hand_type) > 0:
                score += t_chips * 1.0 * chip_scale
            elif hand_type in _COMMITTED_TYPES and not has_main_synergy:
                score += t_chips * 0.8 * chip_scale
            else:
                score += t_chips * 0.2 * chip_scale

        x_mult = config.get("Xmult")
        if isinstance(x_mult, (int, float)) and x_mult and x_mult > 1:
            total_add = sum(j.mult for j in state.jokers if not j.debuff)
            total_add += sum(j.t_mult for j in state.jokers if not j.debuff)
            base_xmult_score = (x_mult - 1) * 25.0
            if hand_type == main_type:
                base_xmult_score *= 4.0
            elif not hand_type:
                base_xmult_score *= 3.0
            elif hand_type and hand_type != main_type:
                base_xmult_score *= 0.5
            if total_add >= 10:
                base_xmult_score *= 3.0
            elif total_add >= 6:
                base_xmult_score *= 2.5
            elif total_add >= 3:
                base_xmult_score *= 2.0
            if mid_game:
                base_xmult_score *= 1.35
            score += base_xmult_score

        extra = config.get("extra")
        if isinstance(extra, dict):
            s_mult = extra.get("s_mult")
            if isinstance(s_mult, (int, float)) and s_mult:
                score += s_mult * 6.0

            chip_mod = extra.get("chip_mod")
            if isinstance(chip_mod, (int, float)) and chip_mod:
                score += chip_mod * 4.0

            hand_add = extra.get("hand_add")
            if isinstance(hand_add, (int, float)) and hand_add:
                score += hand_add * 8.0

            ex_mult = extra.get("Xmult")
            if isinstance(ex_mult, (int, float)) and ex_mult and ex_mult > 1:
                total_add = sum(j.mult for j in state.jokers if not j.debuff)
                total_add += sum(j.t_mult for j in state.jokers if not j.debuff)
                exm_score = (ex_mult - 1) * 25.0
                if not hand_type:
                    exm_score *= 3.0
                if total_add >= 10:
                    exm_score *= 3.0
                elif total_add >= 6:
                    exm_score *= 2.5
                elif total_add >= 3:
                    exm_score *= 2.0
                if mid_game:
                    exm_score *= 1.35
                score += exm_score

            mult_val = extra.get("mult")
            if isinstance(mult_val, (int, float)) and mult_val:
                score += mult_val * 3.0

            chips_val = extra.get("chips")
            if isinstance(chips_val, (int, float)) and chips_val:
                score += chips_val * (0.2 if mid_game else 0.5)

            dollars_extra = extra.get("dollars")
            if isinstance(dollars_extra, (int, float)) and dollars_extra:
                score += dollars_extra * 10.0
        elif isinstance(extra, (int, float)) and extra and extra > 0:
            effect = center.get("effect", "")
            if "Mult" in effect:
                score += extra * 3.0
            elif "Chip" in effect:
                score += extra * (0.2 if mid_game else 0.5)
            elif "Card Buff" in effect:
                score += extra * 2.0
            elif extra >= 10:
                score += extra * 1.0
            else:
                score += extra * 2.0

        h_size = config.get("h_size")
        if isinstance(h_size, (int, float)) and h_size and h_size > 0:
            score += h_size * 12.0

        d_size = config.get("d_size")
        if isinstance(d_size, (int, float)) and d_size and d_size > 0:
            score += d_size * 8.0

        if center_key == "j_four_fingers":
            score += 10.0
        elif center_key in ("j_blueprint", "j_brainstorm"):
            score += self._score_copy_joker(state)

        _PER_CARD_JOKERS = {
            "j_fibonacci": 10.0,
            "j_even_steven": 8.0,
            "j_odd_todd": 6.0,
            "j_smiley": 8.0,
            "j_scary_face": 5.0,
            "j_scholar": 7.0,
            "j_walkie_talkie": 6.0,
            "j_photograph": 26.0,
            "j_greedy_joker": 8.0,
            "j_lusty_joker": 8.0,
            "j_wrathful_joker": 8.0,
            "j_gluttenous_joker": 8.0,
            "j_onyx_agate": 8.0,
            "j_arrowhead": 6.0,
        }
        if center_key in _PER_CARD_JOKERS:
            score += _PER_CARD_JOKERS[center_key]

        if center_key == "j_photograph":
            joker_keys = {j.center_key for j in state.jokers}
            if joker_keys & _RETRIGGER_JOKER_KEYS:
                score += 25.0

        _HELD_CARD_JOKERS = {
            "j_baron": 10.0,
            "j_shoot_the_moon": 6.0,
            "j_raised_fist": 3.0,
        }
        if center_key in _HELD_CARD_JOKERS:
            score += _HELD_CARD_JOKERS[center_key]

        if center_key in _SCALING_JOKER_SCORES:
            n_jokers_current = len(state.jokers)
            ante_bonus = max(8 - ante, 1) * 0.5
            if n_jokers_current < 4:
                score += _SCALING_JOKER_SCORES[center_key] * (1.5 + ante_bonus)
            else:
                score += _SCALING_JOKER_SCORES[center_key] * (1.0 + ante_bonus * 0.5)
            if mid_game and center_key in _SCALING_XMULT_JOKER_KEYS:
                score += 18.0

        _RETRIGGER_JOKERS = {
            "j_hanging_chad": 32.0,
            "j_sock_and_buskin": 38.0,
            "j_selzer": 22.0,
            "j_mime": 22.0,
            "j_dusk": 18.0,
            "j_hack": 22.0,
        }
        if center_key in _RETRIGGER_JOKERS:
            score += _RETRIGGER_JOKERS[center_key]
            joker_keys = {j.center_key for j in state.jokers}
            if "j_photograph" in joker_keys and center_key in (
                "j_hanging_chad",
                "j_sock_and_buskin",
                "j_dusk",
                "j_selzer",
            ):
                score += 25.0
            if center_key == "j_hanging_chad" and "j_sock_and_buskin" in joker_keys:
                score += 12.0
            if center_key == "j_sock_and_buskin" and "j_hanging_chad" in joker_keys:
                score += 12.0
            if mid_game and center_key in ("j_hanging_chad", "j_sock_and_buskin"):
                score += 10.0

        if jname == "Bootstraps":
            score += 8.0 + state.dollars * 0.5
        elif jname == "Bull":
            score += 10.0 + state.dollars * 0.6
        elif jname == "Banner":
            score += 6.0
        elif jname == "Abstract Joker" or jname == "Mystic Summit":
            score += 8.0
        elif jname == "Cavendish":
            score += 10.0
        elif jname == "Gros Michel" or jname == "Misprint":
            score += 8.0
        elif jname == "Burglar" or jname == "Popcorn":
            score += 7.0
        elif jname == "Dusk":
            score += 8.0
        elif jname == "Half Joker":
            score += 6.0
        elif jname == "Ice Cream" or jname == "Flower Pot" or jname == "Seeing Double":
            score += 5.0
        elif jname == "Blackboard" or jname == "Vampire":
            score += 6.0
        elif jname == "Stone Joker":
            score += 5.0
        elif jname == "Glass Joker":
            score += 6.0
        elif jname == "Lucky Cat" or jname == "Blue Joker" or jname == "Red Card":
            score += 5.0
        elif jname == "Throwback":
            score += 6.0

        if cost <= 3:
            score *= 1.3
        elif cost <= 5:
            score *= 1.1
        elif cost <= 6:
            score *= 0.95
        elif cost >= 8:
            x_mult = config.get("Xmult")
            if isinstance(x_mult, (int, float)) and x_mult and x_mult > 1:
                score *= 0.9
            else:
                score *= 0.6

        score *= self._edition_multiplier(edition)

        return score

    def _score_copy_joker(self, state: RunState) -> float:
        """Score Blueprint/Brainstorm as a fraction of the best non-copy joker."""
        if not state.jokers:
            return 3.0
        best = 0.0
        for j in state.jokers:
            if j.center_key in ("j_blueprint", "j_brainstorm"):
                continue
            v = self._score_owned_joker_value(state, j)
            if v > best:
                best = v
        return max(best * 0.6, 8.0)

    def _score_owned_joker_value(self, state: RunState, joker: JokerInstance) -> float:
        score = self._score_joker_value(state, joker.center_key, edition=joker.edition)
        center = state.data.centers.get(joker.center_key, {})
        config = center.get("config")
        if not isinstance(config, dict):
            config = {}
        name = center.get("name", "")
        ante = state.round_resets.ante
        mid_game = ante >= _MID_GAME_ANTE

        base_mult = config.get("mult", 0) or 0
        if isinstance(base_mult, (int, float)) and joker.mult > base_mult:
            score += (joker.mult - base_mult) * (11.0 if mid_game else 14.0)

        base_t_mult = config.get("t_mult", 0) or 0
        if isinstance(base_t_mult, (int, float)) and joker.t_mult > base_t_mult:
            score += (joker.t_mult - base_t_mult) * 10.0

        base_t_chips = config.get("t_chips", 0) or 0
        if isinstance(base_t_chips, (int, float)) and joker.t_chips > base_t_chips:
            score += (joker.t_chips - base_t_chips) * (0.9 if mid_game else 1.5)

        base_x_mult = config.get("Xmult", 1) or 1
        if not isinstance(base_x_mult, (int, float)):
            base_x_mult = 1
        if joker.x_mult > max(float(base_x_mult), 1.0):
            total_add = sum(j.mult for j in state.jokers if not j.debuff)
            total_add += sum(j.t_mult for j in state.jokers if not j.debuff)
            xscore = (joker.x_mult - max(float(base_x_mult), 1.0)) * 70.0
            if total_add >= 10:
                xscore *= 2.0
            elif total_add >= 5:
                xscore *= 1.5
            score += xscore

        extra = joker.extra
        cfg_extra = config.get("extra")
        if isinstance(extra, dict) and isinstance(cfg_extra, dict):
            for key, scale in (("chips", 0.6 if mid_game else 1.0), ("mult", 8.0), ("Xmult", 70.0)):
                live = extra.get(key)
                base = cfg_extra.get(key, 0)
                if isinstance(live, (int, float)) and isinstance(base, (int, float)) and live != base:
                    score += (live - base) * scale
        elif (
            isinstance(extra, (int, float))
            and isinstance(cfg_extra, (int, float))
            and extra != cfg_extra
            and name in {"Castle", "Square Joker", "Runner", "Ice Cream", "Wee Joker"}
        ):
            score += (extra - cfg_extra) * (0.6 if mid_game else 1.0)

        return score

    def _owned_scaling_progress(self, state: RunState, joker: JokerInstance) -> float:
        center = state.data.centers.get(joker.center_key, {})
        config = center.get("config")
        if not isinstance(config, dict):
            config = {}

        progress = 0.0
        base_mult = config.get("mult", 0) or 0
        if isinstance(base_mult, (int, float)):
            progress += max(float(joker.mult) - float(base_mult), 0.0)

        base_x_mult = config.get("Xmult", 1) or 1
        if isinstance(base_x_mult, (int, float)):
            progress += max(float(joker.x_mult) - max(float(base_x_mult), 1.0), 0.0) * 10.0

        extra = joker.extra
        cfg_extra = config.get("extra")
        if isinstance(extra, dict) and isinstance(cfg_extra, dict):
            for key in ("chips", "mult", "Xmult"):
                live = extra.get(key)
                base = cfg_extra.get(key, 0)
                if isinstance(live, (int, float)) and isinstance(base, (int, float)):
                    scale = 10.0 if key == "Xmult" else 1.0
                    progress += max(float(live) - float(base), 0.0) * scale
        elif isinstance(extra, (int, float)) and isinstance(cfg_extra, (int, float)):
            progress += max(float(extra) - float(cfg_extra), 0.0)

        if joker.center_key == "j_fortune_teller":
            progress += float(state.consumeable_usage_total.get("tarot", 0))
        elif joker.center_key == "j_supernova":
            progress += max((float(hand.get("played", 0) or 0) for hand in state.hands.values()), default=0.0)
        elif joker.center_key == "j_steel_joker":
            progress += float(joker.steel_tally)

        return progress

    def _should_preserve_scaling_joker(self, state: RunState, joker: JokerInstance) -> bool:
        if joker.center_key not in _SCALING_JOKER_KEYS:
            return False
        return state.round_resets.ante >= _MID_GAME_ANTE or self._owned_scaling_progress(state, joker) > 0

    def _hand_type_synergy(self, state: RunState, hand_type: str) -> float:
        if not hand_type:
            return 0.0
        sig = self._joker_keys_sig(state)
        if sig != self._synergy_cache_key:
            self._synergy_cache.clear()
            self._synergy_cache_key = sig
        cache_key = hand_type
        if cache_key in self._synergy_cache:
            return self._synergy_cache[cache_key]
        synergy = 0.0
        for j in state.jokers:
            jc = state.data.centers.get(j.center_key, {})
            jcfg = jc.get("config")
            if not isinstance(jcfg, dict):
                continue
            jtype = jcfg.get("type", "")
            if jtype == hand_type:
                synergy += jcfg.get("t_mult", 0) or 0
                synergy += (jcfg.get("t_chips", 0) or 0) * 0.1
                xm = jcfg.get("Xmult", 0) or 0
                if xm > 1:
                    synergy += (xm - 1) * 10
                extra = jcfg.get("extra")
                if isinstance(extra, dict):
                    exm = extra.get("Xmult", 0)
                    if isinstance(exm, (int, float)) and exm > 1:
                        synergy += (exm - 1) * 10
        self._synergy_cache[cache_key] = synergy
        return synergy

    def _find_best_hand(self, state: RunState, hand: list[PlayingCard]) -> set[int]:
        if not hand:
            return set()

        max_cards = min(5, len(hand))
        has_stone = False

        centers = state.data.centers
        card_ids: list[int] = []
        by_rank: dict[int, list[int]] = {}
        by_suit: dict[str, list[int]] = {}
        joker_names = self._get_joker_names(state)
        has_smeared = "Smeared Joker" in joker_names
        has_four_fingers = "Four Fingers" in joker_names
        has_shortcut = "Shortcut" in joker_names
        has_pareidolia = "Pareidolia" in joker_names
        for i, card in enumerate(hand):
            center = centers.get(card.center_key)
            effect = center.get("effect", "") if center else ""
            if effect == "Stone Card":
                has_stone = True
                cid = -id(card)
            else:
                cid = RANK_TO_ID[card.rank]
            card_ids.append(cid)
            if cid > 0:
                by_rank.setdefault(cid, []).append(i)

            if effect != "Stone Card":
                is_wild = effect == "Wild Card"
                if is_wild:
                    for s in ("Spades", "Hearts", "Clubs", "Diamonds"):
                        by_suit.setdefault(s, []).append(i)
                else:
                    by_suit.setdefault(card.suit, []).append(i)
                    if has_smeared:
                        if card.suit in ("Hearts", "Diamonds"):
                            other = "Diamonds" if card.suit == "Hearts" else "Hearts"
                        else:
                            other = "Clubs" if card.suit == "Spades" else "Spades"
                        by_suit.setdefault(other, []).append(i)

        suit_bonus = {}
        suit_map = {
            "j_greedy_joker": ("Diamonds", 3),
            "j_lusty_joker": ("Hearts", 3),
            "j_wrathful_joker": ("Spades", 3),
            "j_gluttenous_joker": ("Clubs", 3),
            "j_onyx_agate": ("Clubs", 7),
            "j_arrowhead": ("Spades", 50),
        }
        for key, (suit, bonus) in suit_map.items():
            for j in state.jokers:
                if j.center_key == key:
                    suit_bonus[suit] = suit_bonus.get(suit, 0) + bonus

        flush_req = 4 if has_four_fingers else 5
        straight_req = 4 if has_four_fingers else 5

        # Stone/Shortcut/Pareidolia break the greedy rank/suit grouping assumptions,
        # so fall back to exhaustive subset search.
        if has_stone or has_shortcut or has_pareidolia:
            return self._find_best_hand_brute(state, hand, max_cards)

        groups = sorted(by_rank.items(), key=lambda x: x[0], reverse=True)
        groups_by_size: dict[int, list[tuple[int, list[int]]]] = {}
        for cid, indices in groups:
            n = len(indices)
            for sz in range(1, n + 1):
                groups_by_size.setdefault(sz, []).append((cid, indices))

        def _best_of(indices_set: set[int]) -> float:
            return sum(RANK_TO_NOMINAL.get(hand[i].rank, 0) for i in indices_set)

        flush_suit: str | None = None
        flush_indices: list[int] | None = None
        for suit, idxs in by_suit.items():
            unique = list(dict.fromkeys(idxs))
            if len(unique) >= flush_req:
                unique.sort(key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0), reverse=True)
                flush_indices = unique[:5]
                flush_suit = suit
                break

        def _find_straight() -> set[int] | None:
            sorted_ranks = sorted(by_rank.keys(), reverse=True)
            rank_set = set(sorted_ranks)
            if 14 in rank_set:
                rank_set.add(1)

            for high in range(14, 0, -1):
                run: list[int] = []
                for r in range(high, high - straight_req - 1, -1):
                    if r < 1:
                        break
                    actual = 14 if r == 1 else r
                    if actual in by_rank:
                        run.append(actual)
                    else:
                        break
                if len(run) >= straight_req:
                    result: set[int] = set()
                    for r in run[:5]:
                        result.add(by_rank[r][0])
                    return result
            return None

        straight_indices = _find_straight()

        if 5 in groups_by_size:
            cid, idxs = groups_by_size[5][0]
            chosen = set(idxs[:5])
            if flush_indices and chosen <= set(flush_indices):
                return chosen
            return chosen

        if 4 in groups_by_size:
            best_four: set[int] | None = None
            best_four_score = -1.0
            for _cid, idxs in groups_by_size[4]:
                s = set(idxs[:4])
                sc = _best_of(s)
                if sc > best_four_score:
                    best_four = s
                    best_four_score = sc

            if flush_indices and straight_indices:
                sf_set = set(flush_indices) & straight_indices
                if len(sf_set) >= straight_req:
                    return set(list(sf_set)[:5])
                flush_set = set(flush_indices) if flush_indices else set()
                if straight_indices and len(flush_set & straight_indices) >= straight_req:
                    return set(list(flush_set & straight_indices)[:5])

            if best_four is not None:
                return best_four

        if flush_indices and straight_indices and flush_suit:
            flush_only_by_rank: dict[int, int] = {}
            for i in dict.fromkeys(by_suit.get(flush_suit, [])):
                cid = card_ids[i]
                if cid > 0 and cid not in flush_only_by_rank:
                    flush_only_by_rank[cid] = i
            if 14 in flush_only_by_rank:
                flush_only_by_rank.setdefault(1, flush_only_by_rank[14])
            for high in range(14, 0, -1):
                run: list[int] = []
                for r in range(high, high - straight_req - 1, -1):
                    if r < 1:
                        break
                    actual = 14 if r == 1 else r
                    if actual in flush_only_by_rank:
                        run.append(flush_only_by_rank[actual])
                    else:
                        break
                if len(run) >= straight_req:
                    return set(run[:5])

        if 3 in groups_by_size and 2 in groups_by_size:
            trips = groups_by_size[3]
            pairs = groups_by_size[2]
            best_trip_cid, best_trip_idxs = trips[0]
            for pair_cid, pair_idxs in pairs:
                if pair_cid != best_trip_cid:
                    return set(best_trip_idxs[:3]) | set(pair_idxs[:2])
            if len(trips) >= 2:
                return set(best_trip_idxs[:3]) | set(trips[1][1][:2])

        if flush_indices and suit_bonus:
            flush_suit_best = None
            flush_best_sv = -1.0
            for suit, idxs in by_suit.items():
                unique = list(dict.fromkeys(idxs))
                if len(unique) >= flush_req:
                    sv = sum(suit_bonus.get(suit, 0) for i in unique[:5])
                    if sv > flush_best_sv:
                        flush_best_sv = sv
                        flush_suit_best = suit
            if flush_suit_best and flush_best_sv > 0:
                idxs = by_suit[flush_suit_best]
                unique = list(dict.fromkeys(idxs))
                unique.sort(key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0), reverse=True)
                return set(unique[:5])

        if flush_indices:
            return set(flush_indices[:5])

        if straight_indices:
            return straight_indices

        if 3 in groups_by_size:
            return set(groups_by_size[3][0][1][:3])

        if 2 in groups_by_size and len(groups_by_size[2]) >= 2:
            p1 = groups_by_size[2][0][1][:2]
            p2 = groups_by_size[2][1][1][:2]
            return set(p1) | set(p2)

        if 2 in groups_by_size:
            pair_idxs = groups_by_size[2][0][1][:2]
            if suit_bonus and len(hand) > 2:
                remaining = sorted(
                    [i for i in range(len(hand)) if i not in pair_idxs],
                    key=lambda i: (suit_bonus.get(hand[i].suit, 0), RANK_TO_NOMINAL.get(hand[i].rank, 0)),
                    reverse=True,
                )
                return set(pair_idxs) | set(remaining[:3])
            return set(pair_idxs)

        ranked = sorted(range(len(hand)), key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0), reverse=True)
        return set(ranked[:max_cards])

    def _find_best_hand_brute(self, state: RunState, hand: list[PlayingCard], max_cards: int) -> set[int]:
        hand_order = [
            "Flush Five",
            "Flush House",
            "Five of a Kind",
            "Straight Flush",
            "Four of a Kind",
            "Full House",
            "Flush",
            "Straight",
            "Three of a Kind",
            "Two Pair",
            "Pair",
            "High Card",
        ]
        hand_rank = {name: i for i, name in enumerate(hand_order)}
        best_indices: set[int] = set()
        best_score = float("-inf")
        best_hand_rank = len(hand_order)

        nominals = [RANK_TO_NOMINAL.get(c.rank, 0) for c in hand]

        for size in range(max_cards, 0, -1):
            for combo in combinations(range(len(hand)), size):
                cards = [hand[i] for i in combo]
                hand_name = self._quick_hand_quality(state, cards)
                rank = hand_rank.get(hand_name, len(hand_order) - 1)
                if rank < best_hand_rank or (rank == best_hand_rank and size == max_cards):
                    score = -rank * 10000 + sum(nominals[i] for i in combo)
                    if score > best_score:
                        best_score = score
                        best_indices = set(combo)
                        best_hand_rank = rank
                    if rank == 0:
                        return best_indices

        if not best_indices:
            ranked = sorted(range(len(hand)), key=lambda i: nominals[i], reverse=True)
            best_indices = set(ranked[:max_cards])
        return best_indices

    def _find_worst_cards(self, state: RunState, hand: list[PlayingCard], max_discard: int) -> set[int]:
        suits = [c.suit for c in hand]
        ranks = [c.rank for c in hand]
        suit_counts = Counter(suits)
        rank_counts = Counter(ranks)
        nominals = [RANK_TO_NOMINAL.get(r, 0) for r in ranks]
        sorted_nominals = sorted(nominals)
        n_sorted = len(sorted_nominals)

        main_type = self._get_main_hand_type(state)
        main_synergy = self._hand_type_synergy(state, main_type)
        is_pair_type = (
            main_type in ("Pair", "Two Pair", "Three of a Kind", "Full House", "Four of a Kind") and main_synergy > 0
        )
        is_flush_type = main_type == "Flush" and main_synergy > 0
        is_sf_type = main_type == "Straight Flush" and main_synergy > 0
        high_ranks = frozenset(("A", "K", "Q", "J", "T"))

        keep_scores: list[float] = []
        for i, card in enumerate(hand):
            score = nominals[i] * 0.1

            rank_n = rank_counts[card.rank]
            if is_pair_type:
                if rank_n >= 3:
                    score += 40.0
                elif rank_n >= 2:
                    score += 25.0
            else:
                if rank_n >= 3:
                    score += 25.0
                elif rank_n >= 2:
                    score += 15.0

            suit_n = suit_counts[card.suit]
            if is_flush_type:
                if suit_n >= 4:
                    score += 30.0
                elif suit_n >= 3:
                    score += 15.0
            elif is_sf_type:
                if suit_n >= 4:
                    score += 25.0
                elif suit_n >= 3:
                    score += 12.0
                lo = max(nominals[i] - 4, 0)
                hi = nominals[i] + 4
                nearby_count = 0
                for j in range(n_sorted):
                    if sorted_nominals[j] > hi:
                        break
                    if sorted_nominals[j] >= lo:
                        nearby_count += 1
                if nearby_count >= 4:
                    score += 10.0
            else:
                if suit_n >= 4:
                    score += 20.0
                elif suit_n >= 3:
                    score += 8.0

            nom = nominals[i]
            lo = max(nom - 4, 0)
            hi = nom + 4
            lo_idx = 0
            for j in range(n_sorted):
                if sorted_nominals[j] >= lo:
                    lo_idx = j
                    break
            nearby = 0
            for j in range(lo_idx, n_sorted):
                if sorted_nominals[j] > hi:
                    break
                nearby += 1
            nearby -= 1
            if nearby >= 3:
                score += 5.0

            if card.rank in high_ranks:
                score += 2.0

            keep_scores.append(score)

        ranked = sorted(range(len(hand)), key=lambda idx: keep_scores[idx])
        return set(ranked[:max_discard])

    def _has_scaling_jokers(self, state: RunState) -> bool:
        scaling_keys = _SCALING_JOKER_KEYS | {"j_obelisk", "j_cavendish"}
        return any(j.center_key in scaling_keys for j in state.jokers)

    def _should_discard_for_draw(
        self, state: RunState, hand: list[PlayingCard], mask: np.ndarray
    ) -> tuple[int, ...] | None:
        if state.current_round.discards_left <= 0 or state.current_round.hands_left < 2:
            return None
        best = self._cached_best_hand(state, hand)
        best_cards = [hand[i] for i in best]
        best_quality = self._quick_hand_quality(state, best_cards)
        quality_above_sf = {"Flush Five", "Flush House", "Five of a Kind", "Straight Flush"}

        main_type = self._get_main_hand_type(state)

        by_rank: dict[str, list[int]] = {}
        for i, card in enumerate(hand):
            by_rank.setdefault(card.rank, []).append(i)

        if main_type in ("Pair", "Two Pair", "Three of a Kind"):
            non_group_cards = []
            for i, card in enumerate(hand):
                if len(by_rank.get(card.rank, [])) < 2:
                    non_group_cards.append(i)
            non_group_cards.sort(key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0))
            to_discard = tuple(sorted(non_group_cards[:5]))
            if to_discard and self._discard_mask_ok(state, mask, to_discard):
                return to_discard

        centers = state.data.centers
        has_smeared = "Smeared Joker" in self._get_joker_names(state)
        by_suit: dict[str, list[int]] = {}
        for i, card in enumerate(hand):
            center = centers.get(card.center_key)
            effect = center.get("effect", "") if center else ""
            if effect == "Wild Card":
                for s in ("Spades", "Hearts", "Clubs", "Diamonds"):
                    by_suit.setdefault(s, []).append(i)
            else:
                by_suit.setdefault(card.suit, []).append(i)
                if has_smeared:
                    if card.suit in ("Hearts", "Diamonds"):
                        other = "Diamonds" if card.suit == "Hearts" else "Hearts"
                    else:
                        other = "Clubs" if card.suit == "Spades" else "Spades"
                    by_suit.setdefault(other, []).append(i)

        # Rule 1: Flush draw
        if best_quality not in quality_above_sf:
            for idxs in by_suit.values():
                unique = list(dict.fromkeys(idxs))
                if len(unique) >= 4:
                    suit_set = set(unique)
                    off_suit = [i for i in range(len(hand)) if i not in suit_set]
                    off_suit.sort(key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0))
                    to_discard = tuple(sorted(off_suit[:5]))
                    if to_discard and self._discard_mask_ok(state, mask, to_discard):
                        return to_discard

        # Rule 2: Straight draw
        if "Shortcut" not in self._get_joker_names(state):
            rank_ids = [(i, RANK_TO_ID[hand[i].rank]) for i in range(len(hand))]
            all_ranks = set()
            rank_to_indices: dict[int, list[int]] = {}
            for i, rid in rank_ids:
                all_ranks.add(rid)
                rank_to_indices.setdefault(rid, []).append(i)
                if rid == 14:
                    all_ranks.add(1)
                    rank_to_indices.setdefault(1, []).append(i)
            for low in range(1, 11):
                window = set(range(low, low + 5))
                covered = window & all_ranks
                if len(covered) >= 4:
                    window_indices: set[int] = set()
                    for r in window:
                        if r in rank_to_indices:
                            for idx in rank_to_indices[r]:
                                window_indices.add(idx)
                                break
                    outside = [i for i in range(len(hand)) if i not in window_indices]
                    outside.sort(key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0))
                    to_discard = tuple(sorted(outside[:5]))
                    if to_discard and self._discard_mask_ok(state, mask, to_discard):
                        return to_discard

        # Rule 3: Pair -> Trips/Two-Pair
        if best_quality == "Pair" and state.current_round.hands_left >= 2:
            pair_ranks: dict[int, list[int]] = {}
            for i in range(len(hand)):
                rid = RANK_TO_ID[hand[i].rank]
                pair_ranks.setdefault(rid, []).append(i)
            pair_rank_id = None
            for rid, idxs in pair_ranks.items():
                if len(idxs) >= 2:
                    pair_rank_id = rid
                    break
            if pair_rank_id is not None:
                non_pair = [i for i in range(len(hand)) if RANK_TO_ID[hand[i].rank] != pair_rank_id]
                non_pair.sort(key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0))
                to_discard = tuple(sorted(non_pair[:3]))
                if to_discard and self._discard_mask_ok(state, mask, to_discard):
                    return to_discard

        # Rule 4: Trips -> Quad/Full House
        if best_quality == "Three of a Kind":
            trip_ranks: dict[int, list[int]] = {}
            for i in range(len(hand)):
                rid = RANK_TO_ID[hand[i].rank]
                trip_ranks.setdefault(rid, []).append(i)
            trip_rank_id = None
            for rid, idxs in trip_ranks.items():
                if len(idxs) >= 3:
                    trip_rank_id = rid
                    break
            if trip_rank_id is not None:
                non_trip = [i for i in range(len(hand)) if RANK_TO_ID[hand[i].rank] != trip_rank_id]
                non_trip.sort(key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0))
                to_discard = tuple(sorted(non_trip[:2]))
                if to_discard and self._discard_mask_ok(state, mask, to_discard):
                    return to_discard

        return None

    def _discard_mask_ok(self, state: RunState, mask: np.ndarray, indices: tuple[int, ...]) -> bool:
        return bool(mask[ActionRange.DISCARD_SUBSET_START + subset_index(indices)])

    def _has_xmult_joker(self, state: RunState) -> bool:
        return any(self._is_xmult_joker(state, joker) for joker in state.jokers)

    def _is_xmult_joker(self, state: RunState, joker: JokerInstance) -> bool:
        if joker.debuff:
            return False
        if joker.x_mult and joker.x_mult > 1:
            return True
        if joker.h_x_mult and joker.h_x_mult > 1:
            return True
        if joker.edition and joker.edition.get("polychrome"):
            return True
        return self._is_xmult_center(state, joker.center_key)

    def _is_xmult_center(self, state: RunState, center_key: str) -> bool:
        if center_key in _XMULT_PROFILE_JOKER_KEYS or center_key in _SCALING_XMULT_JOKER_KEYS:
            return True
        center = state.data.centers.get(center_key, {})
        config = center.get("config", {})
        if not isinstance(config, dict):
            return False
        x_mult = config.get("Xmult", 0)
        if isinstance(x_mult, (int, float)) and x_mult > 1:
            return True
        extra = config.get("extra")
        if isinstance(extra, dict):
            extra_x_mult = extra.get("Xmult", 0)
            if isinstance(extra_x_mult, (int, float)) and extra_x_mult > 1:
                return True
        return False

    def _score_planet_value(self, state: RunState, hand_type: str) -> float:
        main_type = self._get_main_hand_type(state)
        score = 0.0
        if hand_type == main_type:
            score += 45.0
        played_count = state.hands.get(hand_type, {}).get("played", 0)
        if played_count > 0:
            score += 18.0 + min(played_count, 8) * 2.0
        score += self._hand_type_synergy(state, hand_type) * 1.5
        if state.round_resets.ante >= _MID_GAME_ANTE and score > 0:
            score += 18.0
        return score

    def _play_or_reorder_jokers(self, state: RunState, mask: np.ndarray, play_action: int) -> int:
        relative_action = play_action - int(ActionRange.PLAY_SUBSET_START)
        indices = tuple(index for index in subset_indices(relative_action) if index < len(state.hand_cards))
        reorder = self._best_joker_move(state, mask, indices)
        if reorder is None:
            self._joker_order_play_action = -1
            return play_action
        self._joker_order_play_action = play_action
        return reorder

    def _resume_joker_order_plan(
        self,
        state: RunState,
        mask: np.ndarray,
    ) -> int | None:
        play_action = self._joker_order_play_action
        if not (
            int(ActionRange.PLAY_SUBSET_START) <= play_action <= int(ActionRange.PLAY_SUBSET_END) and mask[play_action]
        ):
            self._joker_order_play_action = -1
            return None
        relative_action = play_action - int(ActionRange.PLAY_SUBSET_START)
        indices = tuple(index for index in subset_indices(relative_action) if index < len(state.hand_cards))
        count = min(len(state.jokers), MAX_JOKER_SLOTS)
        current_ids = tuple(id(joker) for joker in state.jokers[:count])
        if (
            self._joker_order_cache_key(state, indices) != self._joker_order_plan_key
            or current_ids != self._joker_order_expected
        ):
            self._joker_order_play_action = -1
            return None
        return self._play_or_reorder_jokers(state, mask, play_action)

    def _score_joker_move_for_hand(
        self,
        state: RunState,
        hand_indices: tuple[int, ...],
        source: int | None = None,
        destination: int | None = None,
        order: tuple[int, ...] | None = None,
    ) -> int:
        trial = deepcopy(state, {id(state.data): state.data})
        if order is not None:
            trial.jokers[: len(order)] = [trial.jokers[index] for index in order]
            trial.joker_keys[: len(order)] = [trial.joker_keys[index] for index in order]
        elif source is not None and destination is not None:
            move_joker(trial, source, destination)
        return play_cards(trial, list(hand_indices)).score.total

    def _joker_order_cache_key(
        self,
        state: RunState,
        hand_indices: tuple[int, ...],
    ) -> tuple:
        selected_cards = tuple(id(state.hand_cards[index]) for index in hand_indices)
        return (
            id(state),
            hand_indices,
            selected_cards,
            state.round,
            state.current_round.hands_left,
            state.current_round.hands_played,
            state.blind_on_deck,
            state.blind_disabled,
        )

    def _joker_order_candidates(
        self,
        state: RunState,
        count: int,
    ) -> list[tuple[int, ...]]:
        """Build score-relevant orders without permuting every joker."""
        current = tuple(range(count))
        active_copies = [
            index
            for index, joker in enumerate(state.jokers[:count])
            if not joker.debuff and joker.center_key in _COPY_JOKER_KEYS
        ]
        noncopies = [index for index in current if index not in active_copies]
        canonical_noncopies = sorted(
            noncopies,
            key=lambda index: self._is_xmult_joker(state, state.jokers[index]),
        )

        candidates = [current]
        seen = {current}

        def add(order: tuple[int, ...]) -> bool:
            if order in seen:
                return True
            if len(candidates) >= _MAX_JOKER_ORDER_CANDIDATES:
                return False
            seen.add(order)
            candidates.append(order)
            return True

        if not active_copies:
            add(tuple(canonical_noncopies))
            return candidates

        blueprints = [index for index in active_copies if state.jokers[index].center_key == "j_blueprint"]
        brainstorms = [index for index in active_copies if state.jokers[index].center_key == "j_brainstorm"]
        compatible_targets = [
            index
            for index in canonical_noncopies
            if state.data.centers.get(state.jokers[index].center_key, {}).get("blueprint_compat")
            and not state.jokers[index].debuff
        ]
        compatible_targets.sort(
            key=lambda index: (
                state.jokers[index].center_key not in _RETRIGGER_JOKER_KEYS,
                not self._is_xmult_joker(state, state.jokers[index]),
            ),
        )

        if not compatible_targets:
            add(tuple(canonical_noncopies + active_copies))
            return candidates

        blueprint_assignments = product(compatible_targets, repeat=len(blueprints)) if blueprints else [()]
        brainstorm_targets: list[int | None] = compatible_targets if brainstorms else [None]

        for blueprint_targets in blueprint_assignments:
            grouped_blueprints: dict[int, list[int]] = {target: [] for target in compatible_targets}
            for blueprint, target in zip(blueprints, blueprint_targets, strict=True):
                grouped_blueprints[target].append(blueprint)

            for brainstorm_target in brainstorm_targets:
                ordered_targets = list(canonical_noncopies)
                if brainstorm_target is not None:
                    ordered_targets.remove(brainstorm_target)
                    ordered_targets.insert(0, brainstorm_target)

                blocks = [[*grouped_blueprints.get(target, []), target] for target in ordered_targets]
                if brainstorms:
                    if self._is_xmult_joker(state, state.jokers[brainstorm_target]):
                        blocks.append(list(brainstorms))
                    else:
                        insert_at = next(
                            (
                                index
                                for index, block in enumerate(blocks)
                                if self._is_xmult_joker(
                                    state,
                                    state.jokers[block[len(block) - 1]],
                                )
                            ),
                            len(blocks),
                        )
                        blocks.insert(insert_at, list(brainstorms))

                order = tuple(index for block in blocks for index in block)
                if not add(order):
                    return candidates

        return candidates

    def _move_toward_joker_order(
        self,
        current: tuple[int, ...],
        target: tuple[int, ...],
        mask: np.ndarray,
    ) -> tuple[int | None, tuple[int, ...]]:
        if current == target:
            return None, current
        destination = next(
            index for index, (actual, wanted) in enumerate(zip(current, target, strict=True)) if actual != wanted
        )
        source = current.index(target[destination])
        action = encode_action(ActionType.MOVE_JOKER, source, destination)
        if not mask[action]:
            return None, current
        moved = list(current)
        joker_id = moved.pop(source)
        moved.insert(destination, joker_id)
        return action, tuple(moved)

    def _best_joker_move(
        self,
        state: RunState,
        mask: np.ndarray,
        hand_indices: tuple[int, ...],
    ) -> int | None:
        """Return the next move toward a cached, exact-scored target order."""
        count = min(len(state.jokers), MAX_JOKER_SLOTS)
        if count < 2 or not hand_indices:
            return None
        has_copy_joker = any(
            not joker.debuff and joker.center_key in _COPY_JOKER_KEYS for joker in state.jokers[:count]
        )
        has_movable_xmult = any(self._is_xmult_joker(state, joker) for joker in state.jokers[:count])
        if not has_copy_joker and not has_movable_xmult:
            return None

        current_ids = tuple(id(joker) for joker in state.jokers[:count])
        plan_key = self._joker_order_cache_key(state, hand_indices)
        if (
            plan_key == self._joker_order_plan_key
            and current_ids == self._joker_order_expected
            and set(current_ids) == set(self._joker_order_plan)
        ):
            action, expected = self._move_toward_joker_order(
                current_ids,
                self._joker_order_plan,
                mask,
            )
            self._joker_order_expected = expected
            if action is None:
                self._joker_order_plan_key = ()
            return action

        candidates = self._joker_order_candidates(state, count)
        best_order = candidates[0]
        best_score = self._score_joker_move_for_hand(
            state,
            hand_indices,
            order=best_order,
        )
        for order in candidates[1:]:
            score = self._score_joker_move_for_hand(
                state,
                hand_indices,
                order=order,
            )
            if score > best_score:
                best_score = score
                best_order = order

        target_ids = tuple(current_ids[index] for index in best_order)
        action, expected = self._move_toward_joker_order(current_ids, target_ids, mask)
        if action is None:
            self._joker_order_plan_key = ()
            self._joker_order_plan = ()
            self._joker_order_expected = ()
            return None
        self._joker_order_plan_key = plan_key
        self._joker_order_plan = target_ids
        self._joker_order_expected = expected
        return action

    def _shop(self, state: RunState, mask: np.ndarray) -> int:
        # Numbered PRIORITY blocks below are evaluated top-to-bottom; the first
        # satisfied priority returns and wins.
        all_items = list(state.shop.cards) + list(state.shop.vouchers) + list(state.shop.boosters)
        joker_slots_left = joker_limit(state) - len(state.jokers)
        cons_slots_left = consumable_limit(state) - len(state.consumables)
        dollars = state.dollars
        reroll_cost = state.current_round.reroll_cost
        ante = state.round_resets.ante
        main_type = self._get_main_hand_type(state)

        # Balatro pays `interest_amount` per $5 held, up to the `interest_cap` cash
        # threshold (default $25, Seed Money $50, Money Tree $100). Roll/spend down to
        # this threshold to preserve max interest income.
        interest_threshold = state.interest_cap

        best_joker_action = -1
        best_joker_score = -1e9
        best_xmult_action = -1
        best_xmult_cost = 999
        best_economy_action = -1
        best_economy_score = -1e9
        main_planet_action = -1
        best_planet_action = -1
        best_planet_score = -1e9
        judgement_action = -1
        overstock_action = -1
        priority_voucher_action = -1

        for i, item in enumerate(all_items):
            action = ActionRange.SHOP_BUY_START + i
            center = state.data.centers.get(item.center_key, {})
            cset = center.get("set", "")
            buy_now = bool(mask[action])
            can_make_room = (cset == "Joker" and joker_slots_left == 0) or (
                cset in ("Planet", "Tarot", "Spectral") and cons_slots_left == 0
            )
            if not buy_now and not can_make_room:
                continue
            if cset == "Joker":
                is_economy = item.center_key in _ECONOMY_JOKERS
                jscore = self._score_joker_value(state, item.center_key, edition=item.edition)
                is_xmult = self._is_xmult_center(state, item.center_key)
                if is_xmult and (best_xmult_action < 0 or item.cost < best_xmult_cost):
                    best_xmult_action = action
                    best_xmult_cost = item.cost
                if not is_economy and jscore > best_joker_score:
                    best_joker_score = jscore
                    best_joker_action = action
                if is_economy:
                    eco = _ECONOMY_SCORES.get(item.center_key, 5.0)
                    if ante <= 3:
                        eco *= 1.5
                    if item.cost <= 4:
                        eco *= 1.3
                    if eco > best_economy_score:
                        best_economy_score = eco
                        best_economy_action = action
            elif cset == "Planet":
                planet_type = center.get("config", {}).get("hand_type", "")
                planet_score = self._score_planet_value(state, planet_type)
                if planet_score > best_planet_score:
                    best_planet_score = planet_score
                    best_planet_action = action
                if planet_type == main_type and main_planet_action < 0:
                    main_planet_action = action
            elif cset == "Tarot":
                if center.get("name") == "Judgement" and judgement_action < 0:
                    judgement_action = action
            elif cset == "Voucher":
                if item.center_key == "v_overstock_norm":
                    overstock_action = action
                elif item.center_key in _PRIORITY_VOUCHERS:
                    priority_voucher_action = action

        n_jokers = len(state.jokers)
        no_xmult = not self._has_xmult_joker(state)

        def _worst_consumable_sell(score_threshold: int = 50) -> int:
            worst_cons_slot = -1
            worst_cons_score = 1e9
            for ci, cons in enumerate(state.consumables[:MAX_CONSUMABLE_SLOTS]):
                cc = state.data.centers.get(cons.center_key, {})
                cs = cc.get("set", "")
                score = 10
                if cs == "Planet":
                    pt = cc.get("config", {}).get("hand_type", "")
                    if pt == main_type:
                        score = 100
                    elif self._hand_type_synergy(state, pt) > 0:
                        score = 50
                elif cs == "Tarot":
                    cname = cc.get("name", "")
                    if cname in ("The Hermit", "Temperance"):
                        score = 80
                    elif cname in ("Judgement", "The High Priestess", "The Emperor"):
                        score = 60
                if score < worst_cons_score:
                    worst_cons_score = score
                    worst_cons_slot = ci
            if worst_cons_slot >= 0 and worst_cons_score < score_threshold:
                sell_action = ActionRange.SHOP_SELL_CONSUMABLE_START + worst_cons_slot
                if mask[sell_action]:
                    return sell_action
            return -1

        def _worst_joker_sell(skip_xmult: bool = True, prefer_economy: bool = False) -> tuple[int, int, int]:
            candidates = []
            for i, j in enumerate(state.jokers[:MAX_JOKER_SLOTS]):
                if j.eternal:
                    continue
                if skip_xmult and self._is_xmult_center(state, j.center_key):
                    continue
                if self._should_preserve_scaling_joker(state, j):
                    continue
                jscore = self._score_owned_joker_value(state, j)
                candidates.append((i, jscore, j.sell_cost, j.center_key))

            if prefer_economy:
                eco_candidates = [c for c in candidates if c[3] in _ECONOMY_JOKERS]
                if eco_candidates:
                    eco_candidates.sort(key=lambda c: c[1])
                    c = eco_candidates[0]
                    return ActionRange.SHOP_SELL_JOKER_START + c[0], c[1], c[2]

            candidates.sort(key=lambda c: c[1])
            if candidates:
                c = candidates[0]
                return ActionRange.SHOP_SELL_JOKER_START + c[0], c[1], c[2]
            return -1, 1e9, 0

        def _can_afford_after_interest(cost: int, min_interest: int = 1) -> bool:
            post_buy = dollars - cost
            threshold = min_interest * 5
            if ante <= 2:
                threshold = 0
            elif ante <= 4:
                threshold = max(threshold - 5, 0)
            return post_buy >= threshold

        def _can_afford_after_consumable_sell(buy_action: int) -> bool:
            item_idx = buy_action - ActionRange.SHOP_BUY_START
            if item_idx >= len(all_items):
                return False
            max_sell = max((cons.sell_cost for cons in state.consumables[:MAX_CONSUMABLE_SLOTS]), default=0)
            return dollars + max_sell >= all_items[item_idx].cost

        # ═══ PRIORITY 1: Overstock voucher, best economy investment ═══
        if overstock_action >= 0:
            item_idx = overstock_action - ActionRange.SHOP_BUY_START
            if item_idx < len(all_items):
                cost = all_items[item_idx].cost
                post_buy = dollars - cost
                if post_buy >= 5:
                    return overstock_action

        # ═══ PRIORITY 2: Priority vouchers (hand size, discard, hands, reroll, etc.) ═══
        if priority_voucher_action >= 0:
            item_idx = priority_voucher_action - ActionRange.SHOP_BUY_START
            if item_idx < len(all_items):
                cost = all_items[item_idx].cost
                post_buy = dollars - cost
                if post_buy >= interest_threshold:
                    return priority_voucher_action

        # ═══ PRIORITY 3: X-mult joker, buy immediately if affordable ═══
        if n_jokers > 0 and joker_slots_left == 0 and best_xmult_action >= 0:
            sell_act, _, sell_val = _worst_joker_sell(skip_xmult=True, prefer_economy=True)
            if sell_act >= 0 and mask[sell_act]:
                xm_idx = best_xmult_action - ActionRange.SHOP_BUY_START
                if xm_idx < len(all_items):
                    xm_cost = all_items[xm_idx].cost
                    if dollars + sell_val >= xm_cost:
                        return sell_act

        if best_xmult_action >= 0 and joker_slots_left > 0:
            item_idx = best_xmult_action - ActionRange.SHOP_BUY_START
            if item_idx < len(all_items) and dollars >= all_items[item_idx].cost:
                return best_xmult_action

        # ═══ PRIORITY 4: Celestial pack, only after scoring joker or from ante 2 ═══
        has_scoring_joker = any(j.mult or j.t_mult for j in state.jokers if not j.debuff)
        if has_scoring_joker or ante >= 2:
            for i, item in enumerate(all_items):
                action = ActionRange.SHOP_BUY_START + i
                if not mask[action]:
                    continue
                center = state.data.centers.get(item.center_key, {})
                if center.get("set") == "Booster":
                    name = center.get("name", "")
                    if "Celestial" in name and item.cost <= 4 and dollars >= item.cost + 1:
                        return action

        # ═══ PRIORITY 5: Main-type planet BUY ═══
        if cons_slots_left > 0 and main_planet_action >= 0:
            item_idx = main_planet_action - ActionRange.SHOP_BUY_START
            if item_idx < len(all_items):
                planet_cost = all_items[item_idx].cost
                if dollars >= planet_cost:
                    return main_planet_action

        # ═══ PRIORITY 6: Sell consumable for main-type planet ═══
        if cons_slots_left == 0 and main_planet_action >= 0:
            sell_act = _worst_consumable_sell(80)
            if sell_act >= 0 and _can_afford_after_consumable_sell(main_planet_action):
                return sell_act

        # ═══ PRIORITY 7: Mid-game played/synergy planet BUY ═══
        if (
            ante >= _MID_GAME_ANTE
            and cons_slots_left > 0
            and best_planet_action >= 0
            and best_planet_score >= 35
            and mask[best_planet_action]
        ):
            return best_planet_action

        # ═══ PRIORITY 8: Sell weak consumable for mid-game planet scaling ═══
        if ante >= _MID_GAME_ANTE and cons_slots_left == 0 and best_planet_action >= 0 and best_planet_score >= 35:
            sell_act = _worst_consumable_sell(80)
            if sell_act >= 0 and _can_afford_after_consumable_sell(best_planet_action):
                return sell_act

        # ═══ PRIORITY 9: Strong scoring joker (x_mult, high mult, type synergy) ═══
        save_xmult_slot = no_xmult and joker_slots_left <= 2 and n_jokers >= 3 and 3 <= ante <= 8
        joker_threshold = -10 if ante <= 1 else (-5 if ante <= 2 else (0 if ante <= 4 else 5))
        if (
            best_joker_action >= 0
            and joker_slots_left > 0
            and best_joker_score > joker_threshold
            and not (save_xmult_slot and best_joker_score <= 15)
        ):
            item_idx = best_joker_action - ActionRange.SHOP_BUY_START
            if item_idx < len(all_items):
                item_cost = all_items[item_idx].cost
                post_buy = dollars - item_cost
                if post_buy >= 0:
                    return best_joker_action

        if joker_slots_left > 0 and n_jokers < 3 and ante <= 2:
            for i, item in enumerate(all_items):
                action = ActionRange.SHOP_BUY_START + i
                if not mask[action]:
                    continue
                center = state.data.centers.get(item.center_key, {})
                if center.get("set") == "Booster" and "Buffoon" in center.get("name", "") and item.cost <= 4:
                    return action

        # ═══ PRIORITY 10: Economy joker if cheap and we have room (NOT before ante 3) ═══
        if best_economy_action >= 0 and joker_slots_left > 0 and ante >= 3 and ante <= 5 and best_economy_score > 5:
            item_idx = best_economy_action - ActionRange.SHOP_BUY_START
            if item_idx < len(all_items):
                item_cost = all_items[item_idx].cost
                post_buy = dollars - item_cost
                if item_cost <= 5 and post_buy >= 2:
                    return best_economy_action

        # ═══ PRIORITY 11: Sell economy joker for strong scoring joker ═══
        if n_jokers > 0 and joker_slots_left == 0 and best_joker_action >= 0 and best_joker_score > 10:
            sell_act, sell_sc, _ = _worst_joker_sell(skip_xmult=True, prefer_economy=True)
            if sell_act >= 0 and mask[sell_act] and best_joker_score - sell_sc > 8:
                return sell_act

        # ═══ PRIORITY 12: Buffoon pack ═══
        if joker_slots_left > 0 and ante >= 2:
            for i, item in enumerate(all_items):
                action = ActionRange.SHOP_BUY_START + i
                if not mask[action]:
                    continue
                center = state.data.centers.get(item.center_key, {})
                if center.get("set") == "Booster":
                    name = center.get("name", "")
                    if "Buffoon" in name and item.cost <= 4 and dollars >= item.cost:
                        return action

        # ═══ PRIORITY 13: ANY played-type planet BUY ═══
        if cons_slots_left > 0:
            for i, item in enumerate(all_items):
                action = ActionRange.SHOP_BUY_START + i
                if not mask[action]:
                    continue
                center = state.data.centers.get(item.center_key, {})
                if center.get("set") == "Planet":
                    pt = center.get("config", {}).get("hand_type", "")
                    played = state.hands.get(pt, {}).get("played", 0)
                    if played > 0 and item.cost <= 4 and dollars >= item.cost:
                        return action

        # ═══ PRIORITY 14: Sell consumable to make room for packs ═══
        if cons_slots_left == 0:
            has_target = main_planet_action >= 0
            if not has_target:
                for item in all_items:
                    center = state.data.centers.get(item.center_key, {})
                    if center.get("set") == "Booster":
                        name = center.get("name", "")
                        if "Celestial" in name or "Buffoon" in name:
                            has_target = True
                            break
            if has_target:
                sell_act = _worst_consumable_sell(60)
                if sell_act >= 0:
                    return sell_act

        # ═══ PRIORITY 15: Judgement tarot ═══
        if judgement_action >= 0 and joker_slots_left > 0:
            if cons_slots_left == 0:
                sell_act = _worst_consumable_sell(50)
                if sell_act >= 0:
                    return sell_act
            if cons_slots_left > 0:
                item_idx = judgement_action - ActionRange.SHOP_BUY_START
                if item_idx < len(all_items) and _can_afford_after_interest(all_items[item_idx].cost):
                    return judgement_action

        # ═══ PRIORITY 16: Money tarots (Hermit, Temperance) ═══
        _SHOP_MONEY_TAROTS = frozenset({"The Hermit", "Temperance"})
        if cons_slots_left > 0 and dollars > 7:
            for i, item in enumerate(all_items):
                action = ActionRange.SHOP_BUY_START + i
                if not mask[action]:
                    continue
                center = state.data.centers.get(item.center_key, {})
                name = center.get("name", "")
                if name in _SHOP_MONEY_TAROTS and item.cost <= 4:
                    return action

        # ═══ PRIORITY 17: Arcana pack ═══
        if cons_slots_left > 0 and dollars > interest_threshold + 3:
            for i, item in enumerate(all_items):
                action = ActionRange.SHOP_BUY_START + i
                if not mask[action]:
                    continue
                center = state.data.centers.get(item.center_key, {})
                if center.get("set") == "Booster":
                    name = center.get("name", "")
                    if "Arcana" in name and item.cost <= 4:
                        return action

        # ═══ PRIORITY 18: Utility tarots ═══
        _UTILITY_TAROTS = frozenset({"The High Priestess", "The Emperor"})
        if cons_slots_left > 0 and dollars > interest_threshold + 3:
            for i, item in enumerate(all_items):
                action = ActionRange.SHOP_BUY_START + i
                if not mask[action]:
                    continue
                center = state.data.centers.get(item.center_key, {})
                if center.get("name") in _UTILITY_TAROTS and item.cost <= 4:
                    return action

        # ═══ PRIORITY 19: Synergy planets ═══
        if cons_slots_left > 0:
            for i, item in enumerate(all_items):
                action = ActionRange.SHOP_BUY_START + i
                if not mask[action]:
                    continue
                center = state.data.centers.get(item.center_key, {})
                if center.get("set") == "Planet":
                    planet_type = center.get("config", {}).get("hand_type", "")
                    if self._hand_type_synergy(state, planet_type) > 0 and _can_afford_after_interest(item.cost):
                        return action

        # ═══ PRIORITY 20: Other boosters ═══
        for i, item in enumerate(all_items):
            action = ActionRange.SHOP_BUY_START + i
            if not mask[action]:
                continue
            center = state.data.centers.get(item.center_key, {})
            if center.get("set") == "Booster" and item.cost <= 4 and dollars > interest_threshold + 3:
                return action

        # ═══ PRIORITY 21: Other vouchers (non-priority) ═══
        for i, item in enumerate(all_items):
            action = ActionRange.SHOP_BUY_START + i
            if not mask[action]:
                continue
            center = state.data.centers.get(item.center_key, {})
            if center.get("set") == "Voucher" and item.center_key not in _PRIORITY_VOUCHERS:
                post_buy = dollars - item.cost
                if post_buy >= interest_threshold:
                    return action

        # ═══ PRIORITY 22: Buy any decent joker as last resort ═══
        if best_joker_action >= 0 and joker_slots_left > 0 and best_joker_score > 0:
            item_idx = best_joker_action - ActionRange.SHOP_BUY_START
            if item_idx < len(all_items) and dollars >= all_items[item_idx].cost:
                return best_joker_action

        # ═══ PRIORITY 23: Reroll, spend down to interest cap if no good buy ═══
        # Engine strength gates how aggressive we are: a strong engine
        # (xmult joker + ≥4 jokers) clears blinds easily, so don't burn cash.
        # Otherwise, max-reroll while we stay above the interest cap.
        strong_engine = self._has_xmult_joker(state) and len(state.jokers) >= 4
        reroll_cap = 2 if strong_engine else (10 if ante >= _MID_GAME_ANTE else 6)
        if (
            mask[ActionRange.SHOP_REROLL]
            and joker_slots_left > 0
            and state.current_round.reroll_cost_increase < reroll_cap
        ):
            post_reroll = dollars - reroll_cost
            keeps_interest = post_reroll >= interest_threshold
            cheap_reroll = ante <= 2 and post_reroll >= 4
            no_good_pickup = best_joker_score < (15 if ante >= _MID_GAME_ANTE else 10)
            xmult_hunt = no_xmult and ante >= _MID_GAME_ANTE and post_reroll >= 12
            if (keeps_interest or cheap_reroll or xmult_hunt) and (no_good_pickup or xmult_hunt):
                return ActionRange.SHOP_REROLL

        if mask[ActionRange.SHOP_LEAVE]:
            return ActionRange.SHOP_LEAVE
        return self._random_valid(mask)

    def _booster_pack(self, state: RunState, mask: np.ndarray) -> int:
        pack = state.pack
        if pack and pack.cards:
            best_score = -1e9
            best_action = -1
            for i, card in enumerate(pack.cards):
                action = ActionRange.PACK_CLAIM_START + i
                if not mask[action]:
                    continue
                center = state.data.centers.get(card.center_key, {})
                cscore = self._score_pack_card(state, center, edition=card.edition)
                if cscore > best_score:
                    best_score = cscore
                    best_action = action
            if best_action >= 0:
                return best_action

        for i in range(5):
            action = ActionRange.PACK_CLAIM_START + i
            if mask[action]:
                return action
        if mask[ActionRange.PACK_SKIP]:
            return ActionRange.PACK_SKIP
        return self._random_valid(mask)

    def _score_pack_card(self, state: RunState, center: dict, *, edition: dict | None = None) -> float:
        cset = center.get("set", "")
        if cset == "Joker":
            jslots = joker_limit(state) - len(state.jokers)
            negative = bool(edition and edition.get("negative"))
            if jslots > 0 or negative:
                return self._score_joker_value(state, center.get("key", ""), edition=edition)
            return -100.0
        elif cset == "Planet":
            planet_type = center.get("config", {}).get("hand_type", "")
            main_type = self._get_main_hand_type(state)
            if planet_type == main_type:
                return 25.0
            if state.hands.get(planet_type, {}).get("played", 0) > 0:
                return 15.0
            if planet_type in ("Pair", "Two Pair"):
                return 12.0
            if self._hand_type_synergy(state, planet_type) > 0:
                return 10.0
            return 3.0
        elif cset == "Tarot":
            cons_slots = consumable_limit(state) - len(state.consumables)
            if cons_slots <= 0:
                return -100.0
            return 4.0
        elif cset == "Spectral":
            return 3.0
        return 1.0

    def _atomic_consumable_action(
        self, state: RunState, slot: int, mask: np.ndarray, *, preferred_indices: tuple[int, ...] | None = None
    ) -> int | None:
        """Return the flat action id for using the consumable in `slot`, or None.

        Picks the targeting variant that matches the consumable's config
        and targets the first k hand cards (k = max_highlighted clamped
        to hand size) for hand-targeted consumables, replacing the old
        greedy multi-step sequence that walked CONSUMABLE_TARGET.
        """
        if slot >= len(state.consumables):
            return None
        cons = state.consumables[slot]
        center = state.data.centers[cons.center_key]
        config = center.get("config") or {}
        max_highlighted = config.get("max_highlighted")
        name = center.get("name", "")
        fallback_hand_limits = HAND_TARGET_CONSUMABLE_LIMITS.get(name)

        if name in JOKER_TARGET_CONSUMABLE_NAMES:
            for joker_idx in range(min(len(state.jokers), MAX_JOKER_SLOTS)):
                action = encode_action(ActionType.USE_CONSUMABLE_JOKER, slot, joker_idx)
                if mask[action]:
                    return action
            return None

        if max_highlighted is not None or fallback_hand_limits is not None:
            if fallback_hand_limits is not None:
                min_size, max_size = fallback_hand_limits
            else:
                min_size = int(config.get("min_highlighted", 1) or 1)
                max_size = int(max_highlighted)
            max_size = min(max_size, MAX_CONSUMABLE_HAND_TARGETS)
            hand_size = len(state.hand_cards)
            max_target_size = min(max_size, hand_size)

            if preferred_indices is not None:
                pref_and_valid = [i for i in preferred_indices if i < hand_size]
                pref_max = min(len(pref_and_valid), max_target_size)
                for target_size in range(pref_max, min_size - 1, -1):
                    for subset in combinations(pref_and_valid, target_size):
                        if not can_use_consumable(state, cons, hand_targets=subset, joker_targets=()):
                            continue
                        action = encode_action(
                            ActionType.USE_CONSUMABLE_HAND_SUBSET,
                            slot,
                            consumable_subset_index(subset),
                        )
                        if mask[action]:
                            return action

            for target_size in range(max_target_size, min_size - 1, -1):
                for subset in combinations(range(hand_size), target_size):
                    if preferred_indices is not None and set(subset).issubset(set(preferred_indices)):
                        continue
                    if not can_use_consumable(state, cons, hand_targets=subset, joker_targets=()):
                        continue
                    action = encode_action(
                        ActionType.USE_CONSUMABLE_HAND_SUBSET,
                        slot,
                        consumable_subset_index(subset),
                    )
                    if mask[action]:
                        return action
            return None

        if not can_use_consumable(state, cons, hand_targets=(), joker_targets=()):
            return None
        action = encode_action(ActionType.USE_CONSUMABLE_NO_TARGET, slot)
        return action if mask[action] else None

    def _random_valid(self, mask: np.ndarray) -> int:
        valid = np.where(mask == 1)[0]
        if len(valid) == 0:
            return 0
        return int(np.random.choice(valid))
