"""Rule-based heuristic agent for generating supervised pretraining data."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from itertools import combinations, product
from typing import TYPE_CHECKING

import numpy as np

from pylatro import can_use_consumable, get_blind_amount, get_poker_hand_info
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
from .hand_candidates import generate_hand_candidates
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

# Subset of xmult jokers that genuinely need shop/round investment before
# becoming useful. Conditional jokers such as Ancient, Baron, and Blackboard
# apply their xmult immediately and must not receive the late-scaler penalty.
_DEVELOPING_XMULT_JOKER_KEYS = frozenset(
    {
        "j_steel_joker",
        "j_campfire",
        "j_throwback",
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

_ANTE1_SKIP_TAGS = frozenset(
    {
        "tag_economy",
    }
)

_RETRIGGER_JOKER_KEYS = frozenset({"j_hanging_chad", "j_sock_and_buskin", "j_selzer", "j_mime", "j_dusk", "j_hack"})

_COPY_JOKER_KEYS = frozenset({"j_blueprint", "j_brainstorm"})
_MAX_JOKER_ORDER_CANDIDATES = 16

_SUIT_TAROT_TARGETS = {
    "The Star": "Diamonds",
    "The Moon": "Clubs",
    "The Sun": "Hearts",
    "The World": "Spades",
}
_SUIT_BOSS_KEYS = {
    "bl_window": "Diamonds",
    "bl_club": "Clubs",
    "bl_head": "Hearts",
    "bl_goad": "Spades",
}
_DECK_SHAPING_TAROTS = frozenset(
    {
        "The Magician",
        "The Empress",
        "The Hierophant",
        "The Lovers",
        "The Chariot",
        "Justice",
        "The Devil",
        "The Tower",
        "Strength",
        "Death",
        "The Hanged Man",
        *_SUIT_TAROT_TARGETS,
    }
)

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
            self._run_key(state),
            tuple(
                (
                    card.rank,
                    card.suit,
                    card.center_key,
                    card.debuff,
                    card.face_down,
                    card.forced_selection,
                )
                for card in hand
            ),
            self._joker_keys_sig(state),
            self._hands_sig(state),
            state.dollars,
            state.current_round.discards_left,
            state.current_round.hands_left,
            state.current_round.hands_played,
            state.round_resets.ante,
            state.blind_on_deck,
            state.blind_disabled,
        )
        if key == self._hand_cache_key:
            return set(self._hand_cache_val)
        structural, _ = generate_hand_candidates(state)
        candidates = [candidate.indices for candidate in structural]
        legacy = tuple(sorted(self._find_best_hand(state, hand)))
        if legacy and legacy not in candidates:
            candidates.append(legacy)

        forced = {i for i, card in enumerate(hand) if card.forced_selection}
        if forced:
            # Final Bell can force an otherwise irrelevant card. Structural
            # generators often return a clean Pair/Trips candidate without
            # that card, leaving only the forced singleton legal. Exhaustively
            # score the small legal space containing every forced card.
            optional = [i for i in range(len(hand)) if i not in forced]
            max_optional = 5 - len(forced)
            for extra_count in range(max_optional + 1):
                for extras in combinations(optional, extra_count):
                    combined = tuple(sorted(forced | set(extras)))
                    if combined and combined not in candidates:
                        candidates.append(combined)

        if any(j.center_key == "j_blackboard" and not j.debuff for j in state.jokers):
            # Blackboard checks the cards left in hand, so playing every red
            # card is often stronger than a conventional poker subset.
            red_cards = {i for i, card in enumerate(hand) if card.suit in {"Hearts", "Diamonds"}}
            blackboard_play = tuple(sorted(forced | red_cards))
            if 1 <= len(blackboard_play) <= 5 and blackboard_play not in candidates:
                candidates.append(blackboard_play)
            for candidate in tuple(candidates) + tuple((i,) for i in range(len(hand))):
                combined = tuple(sorted(forced | red_cards | set(candidate)))
                if 1 <= len(combined) <= 5 and combined not in candidates:
                    candidates.append(combined)

        if any(j.center_key == "j_raised_fist" and not j.debuff for j in state.jokers):
            # Raised Fist uses the lowest held rank. Try playing the complement
            # of each high-card suffix so low cards can be cleared deliberately.
            ranked = sorted(
                range(len(hand)),
                key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0),
                reverse=True,
            )
            forced = {i for i, card in enumerate(hand) if card.forced_selection}
            for held_count in range(max(1, len(hand) - 5), len(hand)):
                play = tuple(sorted(forced | (set(range(len(hand))) - set(ranked[:held_count]))))
                if 1 <= len(play) <= 5 and play not in candidates:
                    candidates.append(play)

        # A mature High Card build can get more value from a nominally
        # non-scoring second card (Photograph, seals, enhancements, boss
        # selection, etc.). Structural poker candidates deliberately omit
        # most such pairs. Score every legal one- and two-card play: this
        # catches those interactions without paying for all 218 subsets.
        if self._get_main_hand_type(state) == "High Card":
            forced = {i for i, card in enumerate(hand) if card.forced_selection}
            if len(forced) <= 1:
                singletons = [
                    (i,)
                    for i in range(len(hand))
                    if not forced or i in forced
                ]
                for singleton in singletons:
                    if singleton not in candidates:
                        candidates.append(singleton)
                for pair in combinations(range(len(hand)), 2):
                    if forced and not forced.issubset(pair):
                        continue
                    if pair not in candidates:
                        candidates.append(pair)
            elif len(forced) < 5:
                forced_tuple = tuple(sorted(forced))
                for kicker in range(len(hand)):
                    if kicker in forced:
                        continue
                    padded = tuple(sorted((*forced_tuple, kicker)))
                    if padded not in candidates:
                        candidates.append(padded)
        if candidates:
            result = set(max(candidates, key=lambda indices: self._estimate_hand_score(state, indices)))
        else:
            result = set(legacy)
        self._hand_cache_key = key
        self._hand_cache_val = result
        return set(result)

    def _get_blind_target(self, state: RunState) -> int:
        blind = state.round_resets.blind
        blind_key = state.round_resets.blind_choices.get(state.blind_on_deck or "")
        if blind_key:
            blind = state.data.blinds.get(blind_key, blind)
        if blind is None:
            return 0
        ante = state.round_resets.ante
        scaling = min(state.stake, 3)
        base = get_blind_amount(ante, scaling)
        mult = blind.get("mult", 1)
        return int(base * mult)

    def _near_term_shop_target(self, state: RunState) -> int:
        """Largest blind target remaining before the next ante shop cycle."""
        target = self._get_blind_target(state)
        # During the shop immediately before Boss selection, `round_resets.blind`
        # still describes the blind just cleared. Resolve the chosen boss
        # explicitly instead of mistaking that stale 1x/1.5x amount for the
        # upcoming target (especially disastrous for The Wall).
        boss_key = state.round_resets.blind_choices.get("Boss", "")
        boss = state.data.blinds.get(boss_key, {})
        base = get_blind_amount(state.round_resets.ante, min(state.stake, 3))
        target = max(target, int(base * boss.get("mult", 1)))
        return target

    def _estimate_hand_score(self, state: RunState, hand_indices: tuple[int, ...]) -> int:
        if not hand_indices:
            return 0
        joker_key = self._joker_keys_sig(state)
        hand_key = tuple(
            (
                card.front_key,
                card.center_key,
                card.edition_key,
                card.seal,
                card.perma_bonus,
                card.debuff,
                card.face_down,
            )
            for card in state.hand_cards
        )
        cache_key = (
            self._run_key(state),
            hand_indices,
            hand_key,
            joker_key,
            self._hands_sig(state),
            state.dollars,
            state.current_round.discards_left,
            state.current_round.hands_left,
            state.current_round.hands_played,
            state.round_resets.ante,
            state.blind_on_deck,
            state.blind_disabled,
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
        # Use the engine as the scoring oracle. Hand values depend on conditional
        # joker, blind, enhancement, held-card, and retrigger effects; a copied
        # state isolates stateful/random scoring effects from the real run.
        trial = deepcopy(state, {id(state.data): state.data})
        return play_cards(trial, list(hand_indices)).score.total

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
        sig = (self._run_key(state), self._joker_keys_sig(state), self._hands_sig(state))
        if sig == self._main_type_cache_key:
            return self._main_type_cache_val
        best = "Pair"
        best_score = 1.0
        has_trousers = any(j.center_key == "j_trousers" and not j.debuff for j in state.jokers)
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
            # Keep planet investment concentrated. Incidental premium hands
            # should not redirect the build; real levels or joker synergy can.
            if ht == "Two Pair" and not has_trousers:
                # Generic Two Pair bonuses do not justify a planet plan: the
                # hand is harder to assemble than Pair and its base scaling is
                # only modestly better. Keep it opportunistic unless levels
                # have already been deliberately invested.
                score = synergy * 0.5 + played * 0.05 + (level - 1) * 22 + 1
            elif ht in {
                "Three of a Kind",
                "Full House",
                "Flush",
                "Straight",
                "Four of a Kind",
                "Straight Flush",
                "Five of a Kind",
                "Flush House",
                "Flush Five",
            }:
                # One conditional payoff does not make a reliable hand plan.
                # Wily/Droll remain useful when their hand naturally appears,
                # while planets and draw shaping stay on Pair until the hard
                # hand has real investment or its dedicated scaler is live.
                runner_live = ht == "Straight" and any(
                    joker.center_key == "j_runner"
                    and not joker.debuff
                    and self._owned_scaling_progress(state, joker) >= 30
                    for joker in state.jokers
                )
                repeated_draw_hand = ht in {"Straight", "Flush"} and level >= 2 and played >= 6
                dedicated_flush = ht == "Flush" and synergy >= 8
                dedicated_straight = ht == "Straight" and synergy >= 16
                committed = (
                    runner_live
                    or repeated_draw_hand
                    or dedicated_flush
                    or dedicated_straight
                    or level >= 3
                )
                if committed:
                    score = synergy * 4 + played * 0.25 + (level - 1) * 30 + 5
                else:
                    score = played * 0.05 + (level - 1) * 5 + 1
            else:
                score = synergy * 4 + played * 0.25 + (level - 1) * 30 + 5
            if ht == "Pair":
                score += 8.0
            elif ht == "Two Pair" and has_trousers:
                score += 5.0
            elif ht == "Three of a Kind":
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
        if state.blind_on_deck == "Boss" and mask[ActionRange.BLIND_REROLL]:
            boss_key = state.round_resets.blind_choices.get("Boss", "")
            reroll_thresholds = {
                "bl_needle": 10,
                "bl_flint": 10,
                "bl_wall": 10,
            }
            dangerous = boss_key == "bl_final_vessel" or state.dollars >= reroll_thresholds.get(
                boss_key, 10**9
            )
            boss_suit = _SUIT_BOSS_KEYS.get(boss_key)
            suit_locked = boss_suit is not None and boss_suit == self._preferred_suit(state)
            face_engine = any(
                joker.center_key in {"j_sock_and_buskin", "j_photograph", "j_scary_face", "j_smiley"}
                for joker in state.jokers
            )
            face_disabled = boss_key == "bl_plant" and face_engine
            if dangerous or ((suit_locked or face_disabled) and state.dollars >= 25):
                return ActionRange.BLIND_REROLL
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

        # Purple seals turn discards into Tarot generation. Cash them in
        # before ordinary draw logic while there is consumable room.
        purple_room = max(consumable_limit(state) - len(state.consumables), 0)
        purple_cards = tuple(
            i for i, card in enumerate(hand) if card.seal == "Purple" and not card.forced_selection
        )[:purple_room]
        card_sharp_opening = False
        if (
            purple_cards
            and state.current_round.discards_left > 0
            and state.current_round.hands_left > 1
        ):
            purple_action = ActionRange.DISCARD_SUBSET_START + subset_index(purple_cards)
            if mask[purple_action]:
                return purple_action

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

        square_candidate: tuple[int, tuple[int, ...]] | None = None
        if any(j.center_key == "j_square" and not j.debuff for j in state.jokers):
            # Square Joker grows only on exactly four played cards. Structural
            # Pair candidates are normally padded to five for redraw value,
            # which left Square at zero chips for whole blinds. Accept a
            # modest immediate-score discount to put its persistent scaling
            # online.
            forced = {i for i, card in enumerate(hand) if card.forced_selection}
            square_plays = [
                indices
                for indices in combinations(range(len(hand)), 4)
                if forced.issubset(indices)
                and mask[ActionRange.PLAY_SUBSET_START + subset_index(indices)]
            ]
            if square_plays:
                square_play = max(square_plays, key=lambda indices: self._estimate_hand_score(state, indices))
                square_score = self._estimate_hand_score(state, square_play)
                square_candidate = (square_score, square_play)
                current_score = self._estimate_hand_score(state, best_play) if best_play else 0
                if square_score >= current_score * 0.35:
                    best_play = square_play

        best_play_cards = [hand[i] for i in best_play] if best_play else []
        best_hand_quality_now = self._quick_hand_quality(state, best_play_cards) if best_play_cards else ""

        est_score = self._estimate_hand_score(state, best_play) if best_play else 0

        # Mystic Summit is +15 Mult only at zero discards.  A discard action
        # does not consume a hand, so preserve the current best five cards and
        # cycle the remainder until the joker is actually online.  Previously
        # the agent bought Summit, immediately banked a modest Two Pair, and
        # carried three or four unused discards through the entire blind.
        if (
            state.current_round.discards_left > 0
            and any(j.center_key == "j_mystic_summit" and not j.debuff for j in state.jokers)
        ):
            protected = set(best_play)
            summit_discard = tuple(
                i
                for i, card in enumerate(hand)
                if i not in protected
                and not card.forced_selection
                and card.seal != "Blue"
            )[:5]
            if summit_discard and self._discard_mask_ok(state, mask, summit_discard):
                return ActionRange.DISCARD_SUBSET_START + subset_index(summit_discard)

        blind_target = self._get_blind_target(state)
        target_remaining = max(0, blind_target - self._current_round_score_estimate(state))
        hands_left = state.current_round.hands_left
        discards_left = state.current_round.discards_left
        can_win_now = est_score >= target_remaining
        # Early blinds are intentionally forgiving. If repeating the current
        # best hand can clear them, take the guaranteed score instead of
        # spending a discard to chase a speculative 15% cushion.
        safety_margin = 1.0 if state.round_resets.ante <= 2 else 1.15
        projected_target = target_remaining * (
            safety_margin if discards_left > 0 and hands_left > 1 else 1.0
        )

        # Card Sharp rewards repeating the same poker hand after the first
        # play. The exact one-hand oracle otherwise opens with an incidental
        # Straight/Two Pair, stranding a mature High Card or Pair plan for the
        # rest of the blind. Establish the reproducible main hand first unless
        # the greedy play already clears the blind.
        if (
            any(j.center_key == "j_card_sharp" and not j.debuff for j in state.jokers)
            and state.current_round.hands_played == 0
            and hands_left >= 3
            and state.hands.get(self._get_main_hand_type(state), {}).get("level", 1) >= 3
            and not can_win_now
        ):
            card_sharp_type = self._get_main_hand_type(state)
            forced = {i for i, card in enumerate(hand) if card.forced_selection}
            card_sharp_candidates: list[tuple[int, tuple[int, ...]]] = []
            for size in range(1, min(5, len(hand)) + 1):
                for indices in combinations(range(len(hand)), size):
                    if not forced.issubset(indices):
                        continue
                    if self._quick_hand_quality(state, [hand[i] for i in indices]) != card_sharp_type:
                        continue
                    action = ActionRange.PLAY_SUBSET_START + subset_index(indices)
                    if mask[action]:
                        card_sharp_candidates.append((self._estimate_hand_score(state, indices), indices))
            if card_sharp_candidates:
                sharp_score, sharp_play = max(card_sharp_candidates)
                if sharp_score >= est_score * 0.15:
                    best_play = sharp_play
                    est_score = sharp_score
                    best_play_cards = [hand[i] for i in best_play]
                    best_hand_quality_now = card_sharp_type
                    can_win_now = est_score >= target_remaining
                    card_sharp_opening = True

        # The Mouth makes the first played hand type the only legal type for
        # the rest of the blind. From Ante 3 onward, lock the established plan
        # (normally Pair/High Card) rather than taking a large one-off Straight
        # or Full House that the remaining draws cannot reproduce.
        boss_key = state.round_resets.blind_choices.get("Boss", "")
        if (
            boss_key == "bl_mouth"
            and state.blind_on_deck == "Boss"
            and not state.mouth_only_hand
            and state.round_resets.ante >= 3
            and est_score < target_remaining
        ):
            mouth_type = self._get_main_hand_type(state)
            forced = {i for i, card in enumerate(hand) if card.forced_selection}
            mouth_candidates: list[tuple[int, tuple[int, ...]]] = []
            for size in range(1, min(5, len(hand)) + 1):
                for indices in combinations(range(len(hand)), size):
                    if not forced.issubset(indices):
                        continue
                    cards = [hand[i] for i in indices]
                    if self._quick_hand_quality(state, cards) != mouth_type:
                        continue
                    action = ActionRange.PLAY_SUBSET_START + subset_index(indices)
                    if mask[action]:
                        mouth_candidates.append((self._estimate_hand_score(state, indices), indices))
            if mouth_candidates:
                est_score, best_play = max(mouth_candidates)
                play_action = ActionRange.PLAY_SUBSET_START + subset_index(best_play)
                return self._play_or_reorder_jokers(state, mask, play_action)
            elif discards_left > 0:
                mouth_discard = self._should_discard_for_draw(state, hand, mask)
                if mouth_discard is None:
                    mouth_discard = tuple(
                        sorted(self._find_worst_cards(state, hand, max_discard=min(5, len(hand))))
                    )
                if mouth_discard and self._discard_mask_ok(state, mask, mouth_discard):
                    return ActionRange.DISCARD_SUBSET_START + subset_index(mouth_discard)

        # Hold at least one Blue seal and play the run's most-played hand so
        # the end-of-round Planet reinforces the established build.
        blue_indices = {i for i, card in enumerate(hand) if card.seal == "Blue"}
        if blue_indices:
            most_played = max(
                state.hands,
                key=lambda hand_type: (
                    state.hands[hand_type].get("played", 0),
                    state.hands[hand_type].get("level", 1),
                    hand_type == self._get_main_hand_type(state),
                ),
            )
            if state.hands[most_played].get("played", 0) <= 0:
                most_played = self._get_main_hand_type(state)
            structural, _ = generate_hand_candidates(state)
            blue_candidates = {candidate.indices for candidate in structural}
            forced = {i for i, card in enumerate(hand) if card.forced_selection}
            type_hand = self._find_type_hand(state, hand, most_played)
            if type_hand and forced.issubset(type_hand):
                blue_candidates.add(type_hand)
            if most_played == "High Card":
                blue_candidates.update((i,) for i in range(len(hand)) if not forced or i in forced)
            matching_blue_holds: list[tuple[int, tuple[int, ...]]] = []
            for indices in blue_candidates:
                if not (blue_indices - set(indices)):
                    continue
                cards = [hand[i] for i in indices]
                if self._quick_hand_quality(state, cards) != most_played:
                    continue
                matching_blue_holds.append((self._estimate_hand_score(state, indices), indices))
            if matching_blue_holds:
                blue_score, blue_play = max(matching_blue_holds)
                safe_blue_play = blue_score >= target_remaining or blue_score * hands_left >= projected_target
                if safe_blue_play and blue_score >= est_score * 0.65:
                    best_play = blue_play
                    est_score = blue_score
                    best_play_cards = [hand[i] for i in best_play]
                    best_hand_quality_now = most_played
                    can_win_now = est_score >= target_remaining

        # Baron is a held-card engine: spending Kings as a temporary Pair can
        # improve one hand while deleting x-mult from every later hand. Search
        # all legal plays that hold every King and accept a lower immediate
        # score when it preserves meaningful multi-hand output.
        if any(j.center_key == "j_baron" and not j.debuff for j in state.jokers):
            king_indices = {i for i, card in enumerate(hand) if card.rank == "K"}
            forced = {i for i, card in enumerate(hand) if card.forced_selection}
            if king_indices and not (king_indices & forced) and hands_left > 1:
                non_kings = [i for i in range(len(hand)) if i not in king_indices]
                held_king_plays: list[tuple[int, tuple[int, ...]]] = []
                for size in range(1, min(5, len(non_kings)) + 1):
                    for indices in combinations(non_kings, size):
                        if not forced.issubset(indices):
                            continue
                        action = ActionRange.PLAY_SUBSET_START + subset_index(indices)
                        if mask[action]:
                            held_king_plays.append((self._estimate_hand_score(state, indices), indices))
                if held_king_plays:
                    baron_score, baron_play = max(held_king_plays)
                    if baron_score >= est_score * 0.35:
                        best_play = baron_play
                        est_score = baron_score
                        best_play_cards = [hand[i] for i in best_play]
                        best_hand_quality_now = self._quick_hand_quality(state, best_play_cards)
                        can_win_now = est_score >= target_remaining

        # Ride the Bus permanently gains mult only when the scoring cards have
        # no faces. Prefer a safe faceless line even when a face-heavy hand is
        # marginally stronger right now.
        if any(j.center_key == "j_ride_the_bus" and not j.debuff for j in state.jokers):
            structural, _ = generate_hand_candidates(state)
            bus_candidates = {candidate.indices for candidate in structural}
            forced = {i for i, card in enumerate(hand) if card.forced_selection}
            for size in (1, 2):
                for indices in combinations(range(len(hand)), size):
                    if forced.issubset(indices):
                        bus_candidates.add(indices)
            faceless: list[tuple[int, tuple[int, ...]]] = []
            for indices in bus_candidates:
                cards = [hand[i] for i in indices]
                _name, _display, _poker_hands, scoring_hand = get_poker_hand_info(state, cards)
                if any(card.rank in {"J", "Q", "K"} for card in scoring_hand):
                    continue
                faceless.append((self._estimate_hand_score(state, indices), indices))
            if faceless:
                bus_score, bus_play = max(faceless)
                safe_bus_play = bus_score >= target_remaining or bus_score * hands_left >= projected_target
                if safe_bus_play and bus_score >= est_score * 0.55:
                    best_play = bus_play
                    est_score = bus_score
                    best_play_cards = [hand[i] for i in best_play]
                    best_hand_quality_now = self._quick_hand_quality(state, best_play_cards)
                    can_win_now = est_score >= target_remaining

        main_type = self._get_main_hand_type(state)
        main_synergy = self._hand_type_synergy(state, main_type)

        if not can_win_now and not card_sharp_opening:
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

        # Main-hand and alternative-hand selection above can replace the
        # four-card Square line with a conventional five-card Pair. Restore
        # the scaling play when it remains within the accepted score budget.
        if square_candidate is not None:
            square_score, square_play = square_candidate
            if square_score >= est_score * 0.35:
                best_play = square_play
                est_score = square_score
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
            planet_synergy = self._hand_type_synergy(state, planet_hand_type)
            if planet_hand_type != main_type and planet_synergy <= 0:
                continue
            score = 0
            if planet_hand_type == main_type:
                score += 20
            score += planet_synergy
            played_count = state.hands.get(planet_hand_type, {}).get("played", 0)
            if played_count > 0:
                score += 10
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
                "The Devil",
                "The Tower",
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
            }
        )

        for slot, cons in enumerate(state.consumables[:MAX_CONSUMABLE_SLOTS]):
            center = state.data.centers[cons.center_key]
            if center.get("set", "") != "Tarot":
                continue
            name = center.get("name", "")
            preferred = None
            if name in _BUFF_TAROTS:
                target_suit = _SUIT_TAROT_TARGETS.get(name)
                if target_suit is not None and target_suit != self._preferred_suit(state):
                    continue
                preferred = best_play if best_play else None
            elif name in _DESTROY_TAROTS:
                worst = tuple(sorted(self._find_worst_cards(state, hand, max_discard=2)))
                if name == "Death":
                    # Death copies the rightmost selected card into the other.
                    # Pair a disposable card with a later, high-value source.
                    death_pair = None
                    for target in worst:
                        sources = [index for index in range(target + 1, len(hand)) if index not in worst]
                        if sources:
                            source = max(
                                sources,
                                key=lambda index: (
                                    hand[index].center_key != "c_base",
                                    bool(hand[index].seal or hand[index].edition_key),
                                    RANK_TO_NOMINAL.get(hand[index].rank, 0),
                                ),
                            )
                            death_pair = (target, source)
                            break
                    preferred = death_pair or (worst if worst else None)
                else:
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

        if best_play and est_score * hands_left >= projected_target:
            play_action = ActionRange.PLAY_SUBSET_START + subset_index(best_play)
            if mask[play_action]:
                return self._play_or_reorder_jokers(state, mask, play_action)

        # In the opening antes, a made hand is also a redraw: playing it scores
        # now and replaces every played card.  The old policy repeatedly held
        # Pair/Two Pair through all four discards, chasing Trips/Full House and
        # entering its four hands with zero points.  Bank Two Pair outright,
        # and bank Pair when its current output is within one discard's worth
        # of the remaining pace.  This is an early survival rule, not a reason
        # to commit the build to Two Pair.
        early_made_hand = (
            state.round_resets.ante <= 2
            and best_play
            and self._owned_build_roles(state)["chips"] == 0
            and self._owned_build_roles(state)["mult"] == 0
            and (
                (
                    best_hand_quality_now == "Two Pair"
                    and (
                        est_score * (hands_left + 2) >= target_remaining * 1.5
                        or discards_left <= 1
                    )
                )
                or (
                    best_hand_quality_now == "Pair"
                    and self._current_round_score_estimate(state) > 0
                    and hands_left >= 3
                    and est_score * (hands_left + 1) >= target_remaining
                )
            )
        )
        if early_made_hand:
            play_action = ActionRange.PLAY_SUBSET_START + subset_index(best_play)
            if mask[play_action]:
                return self._play_or_reorder_jokers(state, mask, play_action)

        # A lone Three-of-a-Kind payoff (for example Wily Joker) is useful
        # when it naturally lands, but it must not consume the whole early
        # blind fishing for a difficult condition. After two misses, bank an
        # available Pair/Two Pair and let the played cards provide the redraw.
        if (
            best_play
            and state.round_resets.ante <= 2
            and any(j.center_key == "j_wily" and not j.debuff for j in state.jokers)
            and best_hand_quality_now in {"Pair", "Two Pair"}
            and discards_left <= 2
            and hands_left >= 2
        ):
            play_action = ActionRange.PLAY_SUBSET_START + subset_index(best_play)
            if mask[play_action]:
                return self._play_or_reorder_jokers(state, mask, play_action)

        draw_discard = self._should_discard_for_draw(state, hand, mask)
        if draw_discard is not None:
            active_keys = {joker.center_key for joker in state.jokers if not joker.debuff}
            without_blue = tuple(
                i
                for i in draw_discard
                if hand[i].seal != "Blue"
                and not ("j_baron" in active_keys and hand[i].rank == "K")
                and not ("j_shoot_the_moon" in active_keys and hand[i].rank == "Q")
            )
            draw_discard = (
                without_blue
                if without_blue and self._discard_mask_ok(state, mask, without_blue)
                else None
            )
        if draw_discard is not None and est_score * max(hands_left - 1, 1) < projected_target:
            return ActionRange.DISCARD_SUBSET_START + subset_index(draw_discard)

        best_discard = tuple(sorted(self._find_worst_cards(state, hand, max_discard=min(5, len(hand))))) if hand else ()
        can_discard = (
            bool(best_discard)
            and discards_left > 0
            and mask[ActionRange.DISCARD_SUBSET_START + subset_index(best_discard)]
        )

        if (
            can_discard
            and hands_left > 0
            and est_score * hands_left < projected_target
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
        elif jname == "Riff-Raff":
            # Besides finding an early scoring engine, every disposable
            # common can be sold and the empty slot rolled again next blind.
            score += 30.0 if ante <= 2 else (14.0 if ante <= 4 else 3.0)

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

    def _joker_build_roles(self, state: RunState, center_key: str) -> frozenset[str]:
        """Classify the scoring job a joker can fill in a conventional build."""
        roles: set[str] = set()
        if center_key in _CHIPS_PROFILE_JOKER_KEYS:
            roles.add("chips")
        if center_key in _MULT_PROFILE_JOKER_KEYS:
            roles.add("mult")
        if self._is_xmult_center(state, center_key):
            roles.add("xmult")

        center = state.data.centers.get(center_key, {})
        config = center.get("config", {})
        if isinstance(config, dict):
            if float(config.get("t_chips", 0) or 0) > 0:
                roles.add("chips")
            if float(config.get("mult", 0) or 0) > 0 or float(config.get("t_mult", 0) or 0) > 0:
                roles.add("mult")
            extra = config.get("extra")
            if isinstance(extra, dict):
                if float(extra.get("chips", 0) or 0) > 0 or float(extra.get("chip_mod", 0) or 0) > 0:
                    roles.add("chips")
                if float(extra.get("mult", 0) or 0) > 0 or float(extra.get("hand_add", 0) or 0) > 0:
                    roles.add("mult")
        return frozenset(roles)

    def _owned_build_roles(self, state: RunState) -> Counter[str]:
        roles: Counter[str] = Counter()
        for joker in state.jokers:
            if joker.debuff:
                continue
            roles.update(self._joker_build_roles(state, joker.center_key))
        return roles

    def _score_joker_for_build(
        self,
        state: RunState,
        center_key: str,
        edition: dict | None = None,
    ) -> float:
        """Value a shop/pack joker against the build roles still missing."""
        score = self._score_joker_value(state, center_key, edition=edition)
        roles = self._joker_build_roles(state, center_key)
        owned = self._owned_build_roles(state)
        ante = state.round_resets.ante
        has_live_additive_mult = any(
            not joker.debuff
            and (
                (joker.mult and joker.mult > 0)
                or (joker.t_mult and joker.t_mult > 0)
                or (joker.h_mult and joker.h_mult > 0)
            )
            for joker in state.jokers
        )

        # Early slots must make chips or mult.  Pure utility jokers such as
        # Pareidolia and Ceremonial Dagger routinely consumed all available
        # money while leaving the run unable to clear Ante 1.
        utility_exception = center_key in (
            _ECONOMY_JOKERS | _COPY_JOKER_KEYS | _RETRIGGER_JOKER_KEYS | {"j_riff_raff"}
        )
        if ante <= 3 and not roles and not utility_exception:
            return -100.0

        # Riff-Raff immediately fills two empty slots with common jokers at
        # each blind. In Ante 1 that is a much stronger route to a complete
        # chips+mult engine than gambling the same money on one Buffoon pack.
        if center_key == "j_riff_raff":
            empty_after_buy = max(joker_limit(state) - len(state.jokers) - 1, 0)
            score += min(empty_after_buy, 2) * (38.0 if ante <= 2 else 16.0)

        if "chips" in roles:
            score += 48.0 if owned["chips"] == 0 else (10.0 if owned["chips"] == 1 else -22.0)
        if "mult" in roles:
            score += 65.0 if owned["mult"] == 0 else (16.0 if owned["mult"] == 1 else -18.0)
        if "xmult" in roles:
            if owned["xmult"] == 0:
                score += 70.0 if owned["mult"] > 0 else 30.0
            elif owned["xmult"] == 1:
                score += 12.0
            else:
                score -= 35.0

        if (
            ante <= 2
            and center_key in _DEVELOPING_XMULT_JOKER_KEYS
            and not has_live_additive_mult
        ):
            # Vampire/Hologram/Constellation are excellent only after their
            # enabling cards have appeared.  Buying one as the first Ante 1
            # joker spends the whole bankroll without changing the next blind.
            # Prefer an immediate scorer or Buffoon roll until basic chips or
            # +Mult is online.
            score -= 120.0

        if ante <= 2 and center_key == "j_flash" and not has_live_additive_mult:
            # Flash Card's advertised +Mult is still zero on purchase and its
            # growth costs reroll money.  It cannot be the run's first actual
            # +Mult source immediately before an Ante 1/2 blind.
            score -= 150.0

        if center_key in _DEVELOPING_XMULT_JOKER_KEYS and ante >= 5:
            # A fresh Constellation/Hologram/Campfire is close to x1 and has
            # little runway left. Do not replace an established additive-mult
            # slot with unrealized late-game potential.
            score -= 80.0
        elif center_key in _SCALING_JOKER_KEYS and ante >= 5:
            score -= 55.0

        center = state.data.centers.get(center_key, {})
        config = center.get("config", {})
        hand_type = config.get("type", "") if isinstance(config, dict) else ""
        suit_flush_engine = hand_type == "Flush" and any(
            state.data.centers.get(joker.center_key, {}).get("effect") == "Suit Mult"
            for joker in state.jokers
            if not joker.debuff
        )
        if suit_flush_engine:
            # Droll plus an existing suit scorer is a complete additive-mult
            # route, not an off-plan conditional. It is worth pivoting from an
            # otherwise weak Pair shell before buying another small planet.
            score += 125.0
        elif hand_type and hand_type != self._get_main_hand_type(state):
            played = int(state.hands.get(hand_type, {}).get("played", 0) or 0)
            score -= 35.0 if played == 0 else 15.0
            if hand_type not in {"Pair", "High Card", "Two Pair"}:
                level = int(state.hands.get(hand_type, {}).get("level", 1) or 1)
                if level <= 1 and played < 4:
                    score -= 80.0
        if center.get("effect") == "Suit Mult" and self._get_main_hand_type(state) in {"High Card", "Pair"}:
            extra = config.get("extra", {}) if isinstance(config, dict) else {}
            target_suit = extra.get("suit", "") if isinstance(extra, dict) else ""
            suit_count = sum(1 for card in state.deck_cards if card.suit == target_suit)
            if suit_count < 20:
                score -= 45.0
        return score

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
        if isinstance(base_mult, (int, float)) and joker.mult != base_mult:
            score += (joker.mult - base_mult) * (11.0 if mid_game else 14.0)

        base_t_mult = config.get("t_mult", 0) or 0
        if isinstance(base_t_mult, (int, float)) and joker.t_mult != base_t_mult:
            score += (joker.t_mult - base_t_mult) * 10.0

        base_t_chips = config.get("t_chips", 0) or 0
        if isinstance(base_t_chips, (int, float)) and joker.t_chips != base_t_chips:
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

        if (
            joker.center_key in _SCALING_JOKER_KEYS
            and ante >= _MID_GAME_ANTE
            and self._owned_scaling_progress(state, joker) < 3.0
        ):
            score *= 0.2

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
        return self._owned_scaling_progress(state, joker) >= 3.0

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
            if j.debuff:
                continue
            if j.center_key == "j_runner" and hand_type == "Straight":
                synergy += 10.0
            elif j.center_key == "j_trousers" and hand_type == "Two Pair":
                synergy += 12.0
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
        joker_keys = {joker.center_key for joker in state.jokers if not joker.debuff}
        preferred_suit = self._preferred_suit(state)
        avoided_suit = self._boss_debuff_suit(state)
        ancient_suit = (
            state.current_round.ancient_card.get("suit", "")
            if "j_ancient" in joker_keys
            else ""
        )

        keep_scores: list[float] = []
        for i, card in enumerate(hand):
            score = nominals[i] * 0.1

            if card.suit == preferred_suit:
                score += 18.0
            if ancient_suit:
                # Ancient Joker's target changes each blind. Draw toward the
                # live target here without redirecting permanent deck shaping.
                score += 52.0 if card.suit == ancient_suit else -10.0
            if avoided_suit and card.suit == avoided_suit and card.suit != preferred_suit:
                score -= 8.0
            if "j_hack" in joker_keys and card.rank in ("2", "3", "4", "5"):
                score += 28.0
            if "j_sock_and_buskin" in joker_keys and card.rank in ("J", "Q", "K"):
                score += 28.0
            if "j_baron" in joker_keys and card.rank == "K":
                # Baron scores held Kings every hand; discarding one to polish
                # a poker pattern usually destroys far more x-mult than the
                # replacement hand can recover. Mime makes that even stronger.
                score += 130.0 if "j_mime" in joker_keys else 90.0
            if "j_shoot_the_moon" in joker_keys and card.rank == "Q":
                score += 70.0
            if (
                "j_ride_the_bus" in joker_keys
                and "j_sock_and_buskin" not in joker_keys
                and card.rank in ("J", "Q", "K")
            ):
                score -= 32.0
            if "j_blackboard" in joker_keys:
                if card.suit in ("Clubs", "Spades"):
                    score += 22.0
                else:
                    score -= 28.0
            center = state.data.centers.get(card.center_key, {})
            if center.get("effect") not in (None, "", "Base"):
                score += 16.0
            if card.edition_key or card.seal:
                score += 12.0
            if card.seal == "Blue":
                score += 55.0
            elif card.seal == "Purple":
                score -= 65.0

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

    def _boss_debuff_suit(self, state: RunState) -> str | None:
        boss_key = state.round_resets.blind_choices.get("Boss", "")
        return _SUIT_BOSS_KEYS.get(boss_key)

    def _preferred_suit(self, state: RunState) -> str:
        """Choose the suit the deck should consolidate around."""
        suits = ("Spades", "Hearts", "Clubs", "Diamonds")
        score = Counter(card.suit for card in state.deck_cards)
        joker_targets = {
            "j_wrathful_joker": "Spades",
            "j_arrowhead": "Spades",
            "j_lusty_joker": "Hearts",
            "j_bloodstone": "Hearts",
            "j_gluttenous_joker": "Clubs",
            "j_onyx_agate": "Clubs",
            "j_greedy_joker": "Diamonds",
            "j_rough_gem": "Diamonds",
        }
        for joker in state.jokers:
            target = joker_targets.get(joker.center_key)
            if target and not joker.debuff:
                score[target] += 10
        avoided = self._boss_debuff_suit(state)
        raw_counts = Counter(card.suit for card in state.deck_cards)
        concentrated = raw_counts and max(raw_counts.values()) - min(raw_counts.values()) >= 3
        if avoided and not concentrated:
            score[avoided] -= 12
        return max(suits, key=lambda suit: (score[suit], -suits.index(suit)))

    def _has_scaling_jokers(self, state: RunState) -> bool:
        scaling_keys = _SCALING_JOKER_KEYS | {"j_obelisk", "j_cavendish"}
        return any(j.center_key in scaling_keys for j in state.jokers)

    def _should_discard_for_draw(
        self, state: RunState, hand: list[PlayingCard], mask: np.ndarray
    ) -> tuple[int, ...] | None:
        # A discard does not consume a hand. Fishing before the final hand is
        # legal and often decisive when the current play cannot clear the
        # remaining target.
        if state.current_round.discards_left <= 0 or state.current_round.hands_left <= 0:
            return None
        best = self._cached_best_hand(state, hand)
        best_cards = [hand[i] for i in best]
        best_quality = self._quick_hand_quality(state, best_cards)
        quality_above_sf = {"Flush Five", "Flush House", "Five of a Kind", "Straight Flush"}

        main_type = self._get_main_hand_type(state)

        active_jokers = {joker.center_key for joker in state.jokers if not joker.debuff}
        if "j_ancient" in active_jokers:
            target_suit = state.current_round.ancient_card.get("suit", "")
            hand_suits = Counter(card.suit for card in hand)
            # Chase the rotating suit from a real foothold, but do not throw
            # away an existing four-card flush draw for a lone target card.
            if hand_suits[target_suit] >= 2 and max(hand_suits.values(), default=0) < 4:
                off_target = tuple(i for i, card in enumerate(hand) if card.suit != target_suit)[:5]
                if off_target and self._discard_mask_ok(state, mask, off_target):
                    return off_target
        if "j_blackboard" in active_jokers:
            red_cards = tuple(i for i, card in enumerate(hand) if card.suit in {"Hearts", "Diamonds"})[:5]
            if red_cards and self._discard_mask_ok(state, mask, red_cards):
                return red_cards

        by_rank: dict[str, list[int]] = {}
        for i, card in enumerate(hand):
            by_rank.setdefault(card.rank, []).append(i)

        owned_roles = self._owned_build_roles(state)
        raw_early_deck = (
            state.round_resets.ante <= 2
            and owned_roles["chips"] == 0
            and owned_roles["mult"] == 0
        )

        if main_type in ("Pair", "Two Pair", "Three of a Kind") and not raw_early_deck:
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

        # With no additive engine, raw Pair/Two Pair output is often below the
        # Ante 1 pace.  Give four-card Flush/Straight draws first refusal, then
        # fall back to the ordinary rank-grouping discard plan.
        if raw_early_deck and main_type in ("Pair", "Two Pair", "Three of a Kind"):
            non_group_cards = [
                i for i, card in enumerate(hand) if len(by_rank.get(card.rank, [])) < 2
            ]
            non_group_cards.sort(key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0))
            to_discard = tuple(sorted(non_group_cards[:5]))
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
        # These centers are prospective x-mult, not active x-mult. Keep
        # searching until their live instance has actually grown above x1.
        if joker.center_key in _DEVELOPING_XMULT_JOKER_KEYS:
            if joker.center_key == "j_steel_joker":
                return joker.steel_tally > 0
            return False
        if joker.center_key in _COPY_JOKER_KEYS:
            return False
        if joker.center_key == "j_baseball":
            return any(
                other is not joker
                and not other.debuff
                and state.data.centers.get(other.center_key, {}).get("rarity") == 2
                for other in state.jokers
            )
        if joker.center_key == "j_stencil":
            return joker_limit(state) - len(state.jokers) > 0
        if joker.center_key == "j_drivers_license":
            enhanced = sum(
                state.data.centers.get(card.center_key, {}).get("effect") not in {None, "", "Base"}
                for card in state.deck_cards
            )
            return enhanced >= 16
        if joker.center_key == "j_loyalty_card":
            # Loyalty triggers every sixth global hand, but a newly purchased
            # copy needs five plays before its first trigger. Count forward
            # from the instance's creation point and only call it active when
            # the next trigger fits inside the available blind hands.
            elapsed = state.hands_played - joker.hands_played_at_create
            hands_until_trigger = next(
                hands
                for hands in range(1, 7)
                if (4 - (elapsed + hands - 1)) % 6 == 5
            )
            available_hands = (
                state.current_round.hands_left
                if state.hand_cards
                else max(int(state.round_resets.hands), 1)
            )
            return hands_until_trigger <= available_hands
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
        # Two Pair is a useful early scoring find, but Uranus has the same +Mult
        # growth as easier hands while requiring four coordinated cards. Only
        # invest when a real Two Pair engine (for example Trousers) made it the
        # committed main hand.
        if hand_type == "Two Pair" and main_type != "Two Pair":
            return 4.0
        score = 0.0
        if hand_type == main_type:
            score += 45.0
        played_count = state.hands.get(hand_type, {}).get("played", 0)
        if played_count > 0:
            score += 18.0 + min(played_count, 8) * 2.0
        synergy = self._hand_type_synergy(state, hand_type)
        score += synergy * 1.5
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
        recent_score = self._current_round_score_estimate(state)
        upcoming_target = max(self._near_term_shop_target(state), 1)
        score_output_ready = recent_score <= 0 or recent_score >= upcoming_target * 1.15

        # Preserve the $10 boss-reroll cost when Violet Vessel is waiting. Its
        # 6x target is worth more than one final marginal shop purchase.
        if (
            state.round_resets.blind_choices.get("Boss", "") == "bl_final_vessel"
            and dollars < 20
            and mask[ActionRange.SHOP_LEAVE]
        ):
            return ActionRange.SHOP_LEAVE

        upcoming_boss = state.round_resets.blind_choices.get("Boss", "")
        strong_xmult_offer = any(
            state.data.centers.get(item.center_key, {}).get("set") == "Joker"
            and self._is_xmult_center(state, item.center_key)
            and item.cost <= dollars
            and self._score_joker_for_build(state, item.center_key, edition=item.edition) >= 50
            for item in all_items
        )
        if (
            state.blind_on_deck == "Boss"
            and upcoming_boss in {"bl_flint", "bl_needle", "bl_wall"}
            and 10 <= dollars < 15
            and not (upcoming_boss == "bl_wall" and state.current_round.reroll_cost_increase > 0)
            and not strong_xmult_offer
            and mask[ActionRange.SHOP_LEAVE]
        ):
            return ActionRange.SHOP_LEAVE

        # Balatro pays `interest_amount` per $5 held, up to the `interest_cap` cash
        # threshold (default $25, Seed Money $50, Money Tree $100). Roll/spend down to
        # this threshold to preserve max interest income.
        antes_remaining = max(int(state.win_ante) - ante, 0)
        if antes_remaining == 0:
            interest_threshold = 0
        elif antes_remaining == 1:
            interest_threshold = min(state.interest_cap, 10)
        else:
            interest_threshold = min(state.interest_cap, 25)
        if upcoming_boss == "bl_final_vessel":
            interest_threshold = max(interest_threshold, 10)

        # Early-Wall rescue: arrive funded, spend one pre-boss shop reroll to
        # expose an immediate scorer, and recover the $10 boss-reroll cost by
        # selling a now-redundant Castle if necessary. This is still scoring
        # investment; packs and speculative scalers remain behind the reserve.
        if upcoming_boss == "bl_wall" and ante <= 2 and state.blind_on_deck == "Boss":
            if (
                state.current_round.reroll_cost_increase == 0
                and dollars >= 15
                and mask[ActionRange.SHOP_REROLL]
            ):
                return ActionRange.SHOP_REROLL
            if state.current_round.reroll_cost_increase > 0:
                affordable_immediate: list[tuple[float, int]] = []
                max_sell = max((joker.sell_cost for joker in state.jokers if not joker.eternal), default=0)
                for i, item in enumerate(all_items):
                    action = ActionRange.SHOP_BUY_START + i
                    center = state.data.centers.get(item.center_key, {})
                    if (
                        center.get("set") == "Joker"
                        and item.center_key not in _SCALING_JOKER_KEYS
                        and mask[action]
                        and dollars - item.cost + max_sell >= 10
                    ):
                        affordable_immediate.append((self._score_joker_for_build(state, item.center_key), action))
                if affordable_immediate:
                    immediate_score, immediate_action = max(affordable_immediate)
                    if immediate_score > 0:
                        return immediate_action
                if dollars < 10 and any(j.center_key == "j_blue_joker" for j in state.jokers):
                    for ji, joker in enumerate(state.jokers[:MAX_JOKER_SLOTS]):
                        sell_action = ActionRange.SHOP_SELL_JOKER_START + ji
                        if (
                            joker.center_key == "j_castle"
                            and dollars + joker.sell_cost >= 10
                            and mask[sell_action]
                        ):
                            return sell_action

        # An early Wall is a 4x target and often cannot be repaired by one
        # ordinary purchase. If we already have enough bankroll to reach the
        # pre-boss purchase plus the boss-reroll reserve through blind rewards,
        # stop spending it too early. This makes the Wall reroll policy usable.
        if (
            upcoming_boss == "bl_wall"
            and ante <= 2
            and dollars >= 10
            and (state.blind_on_deck != "Boss" or dollars < 18)
            and not (state.blind_on_deck == "Big" and dollars >= 18)
            and mask[ActionRange.SHOP_LEAVE]
        ):
            return ActionRange.SHOP_LEAVE

        best_joker_action = -1
        best_joker_score = -1e9
        best_xmult_action = -1
        best_xmult_score = -1e9
        best_economy_action = -1
        best_economy_score = -1e9
        best_immediate_joker_action = -1
        best_immediate_joker_score = -1e9
        main_planet_action = -1
        best_planet_action = -1
        best_planet_score = -1e9
        best_tarot_action = -1
        best_tarot_score = -1e9
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
                jscore = self._score_joker_for_build(state, item.center_key, edition=item.edition)
                is_xmult = self._is_xmult_center(state, item.center_key)
                if is_xmult and jscore > best_xmult_score:
                    best_xmult_action = action
                    best_xmult_score = jscore
                if not is_economy and jscore > best_joker_score:
                    best_joker_score = jscore
                    best_joker_action = action
                if (
                    not is_economy
                    and (
                        item.center_key not in _SCALING_JOKER_KEYS
                        or item.center_key == "j_popcorn"
                    )
                    and bool(self._joker_build_roles(state, item.center_key) & {"chips", "mult"})
                    and jscore > best_immediate_joker_score
                ):
                    best_immediate_joker_score = jscore
                    best_immediate_joker_action = action
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
                tarot_score = self._score_tarot_value(state, center)
                if tarot_score > best_tarot_score:
                    best_tarot_score = tarot_score
                    best_tarot_action = action
                if center.get("name") == "Judgement" and judgement_action < 0:
                    judgement_action = action
            elif cset == "Voucher":
                if item.center_key == "v_overstock_norm":
                    overstock_action = action
                elif item.center_key in _PRIORITY_VOUCHERS:
                    priority_voucher_action = action

        n_jokers = len(state.jokers)
        no_xmult = not self._has_xmult_joker(state)
        owned_roles = self._owned_build_roles(state)
        missing_early_core = ante <= 2 and (
            owned_roles["chips"] == 0 or owned_roles["mult"] == 0
        )

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

        # In an Ante 1/2 boss shop, search once for a real chips/+Mult joker
        # before spending the bankroll on a planet when neither scoring role
        # is represented and the current shop has no immediate scorer.  Keep
        # enough money to buy an ordinary $5 joker after the reroll.
        if (
            state.blind_on_deck == "Boss"
            and owned_roles["chips"] == 0
            and owned_roles["mult"] == 0
            and best_immediate_joker_action < 0
            and mask[ActionRange.SHOP_REROLL]
            and dollars - reroll_cost >= 5
            and state.current_round.reroll_cost_increase == 0
        ):
            return ActionRange.SHOP_REROLL

        def _worst_joker_sell(skip_xmult: bool = True, prefer_economy: bool = False) -> tuple[int, int, int]:
            candidates = []
            for i, j in enumerate(state.jokers[:MAX_JOKER_SLOTS]):
                if j.eternal:
                    continue
                if skip_xmult and self._is_xmult_center(state, j.center_key):
                    replaceable_inactive_condition = (
                        j.center_key
                        in {
                            "j_baseball",
                            "j_stencil",
                            "j_drivers_license",
                            "j_loyalty_card",
                        }
                        and (j.center_key != "j_baseball" or ante >= 5)
                        and not self._is_xmult_joker(state, j)
                    )
                    replaceable_unscaled_developer = (
                        j.center_key in _DEVELOPING_XMULT_JOKER_KEYS
                        and self._owned_scaling_progress(state, j) < 3.0
                    )
                    if not replaceable_inactive_condition and not replaceable_unscaled_developer:
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

        # Keep Riff-Raff producing and realize its economy value. When all
        # slots are full, cash out the weakest generated joker while retaining
        # the only source of any core scoring role and any established scaler.
        has_riff_raff = any(j.center_key == "j_riff_raff" and not j.debuff for j in state.jokers)
        if has_riff_raff and ante <= 4 and len(state.jokers) >= joker_limit(state):
            roles = self._owned_build_roles(state)
            disposable: list[tuple[float, int]] = []
            for ji, joker in enumerate(state.jokers[:MAX_JOKER_SLOTS]):
                if joker.center_key == "j_riff_raff" or joker.eternal:
                    continue
                joker_roles = self._joker_build_roles(state, joker.center_key)
                if "xmult" in joker_roles:
                    continue
                if any(role in {"chips", "mult"} and roles[role] <= 1 for role in joker_roles):
                    continue
                if self._should_preserve_scaling_joker(state, joker):
                    continue
                sell_action = ActionRange.SHOP_SELL_JOKER_START + ji
                if mask[sell_action]:
                    disposable.append((self._score_owned_joker_value(state, joker), sell_action))
            if disposable:
                worst_score, sell_action = min(disposable)
                if worst_score < 35.0:
                    return sell_action

        # Campfire only becomes an x-mult joker when the policy deliberately
        # feeds it sold cards. Scale it between non-boss blinds with disposable
        # consumables and cheap off-plan planets.
        has_campfire = any(j.center_key == "j_campfire" and not j.debuff for j in state.jokers)
        if has_campfire:
            for ci, cons in enumerate(state.consumables[:MAX_CONSUMABLE_SLOTS]):
                center = state.data.centers.get(cons.center_key, {})
                cset = center.get("set", "")
                is_disposable = False
                if cset == "Planet":
                    is_disposable = center.get("config", {}).get("hand_type", "") != main_type
                elif cset == "Tarot":
                    is_disposable = self._score_tarot_value(state, center) < 18
                sell_action = ActionRange.SHOP_SELL_CONSUMABLE_START + ci
                if is_disposable and mask[sell_action]:
                    return sell_action

            if cons_slots_left > 0:
                for i, item in enumerate(all_items):
                    action = ActionRange.SHOP_BUY_START + i
                    if not mask[action] or item.cost > 3 or dollars - item.cost < 4:
                        continue
                    center = state.data.centers.get(item.center_key, {})
                    if center.get("set") == "Planet" and center.get("config", {}).get("hand_type", "") != main_type:
                        return action

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
        if n_jokers > 0 and joker_slots_left == 0 and best_xmult_action >= 0 and best_xmult_score > 10:
            sell_act, _, sell_val = _worst_joker_sell(skip_xmult=True, prefer_economy=True)
            if sell_act >= 0 and mask[sell_act]:
                xm_idx = best_xmult_action - ActionRange.SHOP_BUY_START
                if xm_idx < len(all_items):
                    xm_cost = all_items[xm_idx].cost
                    if dollars + sell_val >= xm_cost:
                        return sell_act

        if best_xmult_action >= 0 and joker_slots_left > 0 and best_xmult_score > 10:
            item_idx = best_xmult_action - ActionRange.SHOP_BUY_START
            if item_idx < len(all_items) and dollars >= all_items[item_idx].cost:
                return best_xmult_action

        # Red Card is the one paid-Arcana exception while score is behind:
        # skipping the pack directly scales an owned scoring joker by +3 Mult.
        # Do not extend this to Standard packs in Antes 1-2.
        has_red_card = any(j.center_key == "j_red_card" and not j.debuff for j in state.jokers)
        if has_red_card and not score_output_ready:
            for i, item in enumerate(all_items):
                action = ActionRange.SHOP_BUY_START + i
                if not mask[action] or item.cost > dollars:
                    continue
                center = state.data.centers.get(item.center_key, {})
                if center.get("set") == "Booster" and "Arcana" in center.get("name", ""):
                    return action

        # Antes 1-2 should be secured by reliable current output. When the
        # latest blind score is behind the pending boss, buy an affordable
        # immediate scorer before gambling on packs or a fresh conditional
        # scaler that has not accumulated any value yet.
        if (
            ante <= 2
            and not score_output_ready
            and joker_slots_left > 0
            and best_immediate_joker_action >= 0
            and best_immediate_joker_score > 0
        ):
            item_idx = best_immediate_joker_action - ActionRange.SHOP_BUY_START
            if item_idx < len(all_items) and dollars >= all_items[item_idx].cost:
                return best_immediate_joker_action

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
                    if "Celestial" in name:
                        cheap_early = item.cost <= 4 and dollars >= item.cost + 1
                        surplus_cash = ante >= 3 and dollars - item.cost >= interest_threshold
                        if cheap_early or surplus_cash:
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
            # Once an additive engine occupies most slots, its last open slot
            # is substantially more valuable as a route to multiplicative
            # scaling than as one more chips/+Mult joker. X-mult purchases were
            # already handled by the priority directly above this block.
            and not save_xmult_slot
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

        # Buy useful card improvement/removal tarots once the run has at least
        # one scoring role.  A small survival buffer is sufficient early; the
        # old full-interest requirement meant Arcana was almost never touched.
        # Paid Arcana improves consistency and deck quality, but those gains do
        # not rescue an engine that is currently behind the score curve. The
        # shop still exposes the score from the blind just cleared, so compare
        # it with the upcoming target and require a modest safety margin. Tests
        # and callers without an observed score fall back to build composition.
        if recent_score > 0:
            score_output_ready = recent_score >= upcoming_target * 1.15
        else:
            score_output_ready = ante <= 2 or not no_xmult
        score_engine_ready = (
            owned_roles["chips"] > 0
            and owned_roles["mult"] > 0
            and score_output_ready
        )
        if cons_slots_left > 0 and best_tarot_action >= 0 and best_tarot_score >= 18:
            item_idx = best_tarot_action - ActionRange.SHOP_BUY_START
            if item_idx < len(all_items):
                cost = all_items[item_idx].cost
                cash_buffer = 3 if ante <= 2 else 5
                if score_engine_ready and dollars - cost >= cash_buffer:
                    return best_tarot_action

        if cons_slots_left > 0 and score_engine_ready:
            for i, item in enumerate(all_items):
                action = ActionRange.SHOP_BUY_START + i
                if not mask[action]:
                    continue
                center = state.data.centers.get(item.center_key, {})
                if center.get("set") == "Booster" and "Arcana" in center.get("name", ""):
                    cash_buffer = 3 if ante <= 2 else 5
                    cheap = item.cost <= 4 and dollars - item.cost >= cash_buffer
                    surplus_cash = ante >= 3 and dollars - item.cost >= interest_threshold
                    if cheap or surplus_cash:
                        return action

        has_hologram = any(j.center_key == "j_hologram" and not j.debuff for j in state.jokers)
        priority_seals = sum(card.seal in {"Purple", "Blue"} for card in state.deck_cards)
        seal_pipeline_ready = priority_seals >= 3
        midgame_seal_hunt = ante >= 3 and not seal_pipeline_ready
        if has_hologram or midgame_seal_hunt:
            for i, item in enumerate(all_items):
                action = ActionRange.SHOP_BUY_START + i
                if not mask[action]:
                    continue
                center = state.data.centers.get(item.center_key, {})
                if center.get("set") == "Booster" and "Standard" in center.get("name", ""):
                    cash_buffer = 5 if has_hologram else max(interest_threshold, 4)
                    if item.cost <= 6 and dollars - item.cost >= cash_buffer:
                        return action

        # ═══ PRIORITY 10: Economy joker if cheap and we have room (NOT before ante 3) ═══
        if (
            score_engine_ready
            and best_economy_action >= 0
            and joker_slots_left > 0
            and ante >= 3
            and ante <= 5
            and best_economy_score > 5
        ):
            item_idx = best_economy_action - ActionRange.SHOP_BUY_START
            if item_idx < len(all_items):
                item_cost = all_items[item_idx].cost
                post_buy = dollars - item_cost
                if item_cost <= 5 and post_buy >= 2:
                    return best_economy_action

        # ═══ PRIORITY 11: Sell economy joker for strong scoring joker ═══
        if n_jokers > 0 and joker_slots_left == 0 and best_joker_action >= 0 and best_joker_score > 10:
            sell_act, sell_sc, sell_val = _worst_joker_sell(skip_xmult=True, prefer_economy=True)
            item_idx = best_joker_action - ActionRange.SHOP_BUY_START
            can_complete_replacement = (
                0 <= item_idx < len(all_items)
                and dollars + sell_val >= all_items[item_idx].cost
            )
            if (
                sell_act >= 0
                and mask[sell_act]
                and can_complete_replacement
                and best_joker_score - sell_sc > 8
            ):
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

        # ═══ PRIORITY 14: Sell consumable to make room for packs ═══
        if cons_slots_left == 0:
            has_target = main_planet_action >= 0 or (best_tarot_action >= 0 and best_tarot_score >= 18)
            if not has_target:
                for item in all_items:
                    center = state.data.centers.get(item.center_key, {})
                    if center.get("set") == "Booster":
                        name = center.get("name", "")
                        if "Celestial" in name or "Arcana" in name or "Buffoon" in name:
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
        if score_engine_ready and cons_slots_left > 0 and dollars > interest_threshold + 3:
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

        # ═══ PRIORITY 22: Buy any decent joker as last resort ═══
        if (
            best_joker_action >= 0
            and joker_slots_left > 0
            and best_joker_score > 0
            and not save_xmult_slot
        ):
            item_idx = best_joker_action - ActionRange.SHOP_BUY_START
            if item_idx < len(all_items) and dollars >= all_items[item_idx].cost:
                return best_joker_action

        # ═══ PRIORITY 23: Reroll, spend down to interest cap if no good buy ═══
        # Engine strength gates how aggressive we are: a strong engine
        # (xmult joker + ≥4 jokers) clears blinds easily, so don't burn cash.
        # Otherwise, max-reroll while we stay above the interest cap.
        strong_engine = self._has_xmult_joker(state) and len(state.jokers) >= 4
        reroll_cap = 6 if strong_engine else (10 if ante >= _MID_GAME_ANTE else 6)
        replaceable_action, replaceable_score, _ = _worst_joker_sell(skip_xmult=True)
        can_search_for_upgrade = joker_slots_left > 0 or replaceable_action >= 0
        if (
            mask[ActionRange.SHOP_REROLL]
            and can_search_for_upgrade
            and state.current_round.reroll_cost_increase < reroll_cap
        ):
            post_reroll = dollars - reroll_cost
            roles = self._owned_build_roles(state)
            missing_early_core = ante <= 2 and (roles["chips"] == 0 or roles["mult"] == 0)
            purchase_floor = 5 if missing_early_core and state.blind_on_deck == "Boss" else (3 if missing_early_core else 4)
            retains_purchase_cash = reroll_cost == 0 or post_reroll >= purchase_floor
            # Bull and Bootstraps turn cash into immediate blind score. Burning
            # that cash on speculative rerolls can make an otherwise free
            # Ante 1-2 blind unwinnable, so treat their scoring bankroll as an
            # additional purchase floor.
            active_keys = {joker.center_key for joker in state.jokers if not joker.debuff}
            cash_scaler_floor = 10 if "j_bull" in active_keys else (5 if "j_bootstraps" in active_keys else 0)
            retains_cash_scaler = reroll_cost == 0 or post_reroll >= cash_scaler_floor
            keeps_interest = post_reroll >= interest_threshold
            cheap_reroll = ante <= 2 and post_reroll >= 4
            emergency_core_hunt = (
                ante <= 2
                and (roles["chips"] == 0 or roles["mult"] == 0)
                and post_reroll >= (5 if state.blind_on_deck == "Boss" else 3)
            )
            pickup_threshold = 15 if ante >= _MID_GAME_ANTE else 10
            actionable_pickup = (
                joker_slots_left > 0 and best_joker_score >= pickup_threshold
            ) or (
                joker_slots_left == 0
                and replaceable_action >= 0
                and best_joker_score - replaceable_score > 8
            )
            no_good_pickup = not actionable_pickup
            xmult_hunt = no_xmult and ante >= _MID_GAME_ANTE and post_reroll >= 8
            if retains_purchase_cash and retains_cash_scaler and (keeps_interest or cheap_reroll or xmult_hunt or emergency_core_hunt) and (
                no_good_pickup or xmult_hunt or emergency_core_hunt
            ):
                return ActionRange.SHOP_REROLL

        if mask[ActionRange.SHOP_LEAVE]:
            return ActionRange.SHOP_LEAVE
        return self._random_valid(mask)

    def _booster_pack(self, state: RunState, mask: np.ndarray) -> int:
        pack = state.pack
        if pack and pack.cards:
            pack_center = state.data.centers.get(pack.booster_key, {})
            is_standard_pack = "Standard" in pack_center.get("name", "")
            is_arcana_pack = "Arcana" in pack_center.get("name", "")
            recent_score = self._current_round_score_estimate(state)
            score_behind = recent_score > 0 and recent_score < self._near_term_shop_target(state) * 1.15
            if (
                is_arcana_pack
                and score_behind
                and any(j.center_key == "j_red_card" and not j.debuff for j in state.jokers)
                and mask[ActionRange.PACK_SKIP]
            ):
                return ActionRange.PACK_SKIP
            best_score = -1e9
            best_action = -1
            for i, card in enumerate(pack.cards):
                action = ActionRange.PACK_CLAIM_START + i
                if not mask[action]:
                    continue
                center = state.data.centers.get(card.center_key, {})
                cscore = self._score_pack_card(
                    state,
                    center,
                    edition=card.edition,
                    seal=card.seal,
                )
                if cscore > best_score:
                    best_score = cscore
                    best_action = action
            if best_action >= 0 and (is_standard_pack or best_score >= 10.0):
                return best_action

            if mask[ActionRange.PACK_SKIP]:
                return ActionRange.PACK_SKIP

        for i in range(5):
            action = ActionRange.PACK_CLAIM_START + i
            if mask[action]:
                return action
        if mask[ActionRange.PACK_SKIP]:
            return ActionRange.PACK_SKIP
        return self._random_valid(mask)

    def _score_pack_card(
        self,
        state: RunState,
        center: dict,
        *,
        edition: dict | None = None,
        seal: str | None = None,
    ) -> float:
        cset = center.get("set", "")
        seal_bonus = {
            "Purple": 46.0,
            "Blue": 44.0,
            "Gold": 24.0,
            "Red": 34.0 if state.round_resets.ante >= 6 else 18.0,
        }.get(seal, 0.0)
        if cset == "Joker":
            jslots = joker_limit(state) - len(state.jokers)
            negative = bool(edition and edition.get("negative"))
            if jslots > 0 or negative:
                return self._score_joker_for_build(state, center.get("key", ""), edition=edition) + seal_bonus
            return -100.0
        elif cset == "Planet":
            planet_type = center.get("config", {}).get("hand_type", "")
            main_type = self._get_main_hand_type(state)
            if planet_type == main_type:
                return 30.0 + seal_bonus
            if self._hand_type_synergy(state, planet_type) > 0:
                return 18.0 + seal_bonus
            return -5.0
        elif cset == "Tarot":
            cons_slots = consumable_limit(state) - len(state.consumables)
            if cons_slots <= 0:
                from pylatro.shop import can_claim_pack_consumable

                if not can_claim_pack_consumable(
                    state,
                    str(center.get("key", "") or ""),
                    edition=edition,
                ):
                    return -100.0
            tarot_score = self._score_tarot_value(state, center)
            roles = self._owned_build_roles(state)
            preboss_survival = (
                state.round_resets.ante <= 2
                and state.blind_on_deck == "Boss"
                and (roles["chips"] == 0 or roles["mult"] == 0)
            )
            if preboss_survival:
                name = center.get("name", "")
                if name in {"The Hermit", "Temperance"}:
                    tarot_score = 6.0
                elif name in _DECK_SHAPING_TAROTS:
                    tarot_score += 16.0
            return tarot_score + seal_bonus
        elif cset == "Spectral":
            return 3.0 + seal_bonus
        elif cset == "Enhanced":
            return seal_bonus + {
                "Glass Card": 30.0,
                "Steel Card": 26.0,
                "Lucky Card": 24.0,
                "Mult Card": 22.0,
                "Wild Card": 20.0,
                "Bonus Card": 18.0,
                "Gold Card": 16.0,
                "Stone Card": 10.0,
            }.get(center.get("effect", ""), 12.0)
        return 1.0 + seal_bonus

    def _score_tarot_value(self, state: RunState, center: dict) -> float:
        name = center.get("name", "")
        if name in ("The Hermit", "Temperance"):
            return 30.0
        if name == "The Hanged Man":
            return 28.0
        if name == "Death":
            return 26.0
        if name in ("The Emperor", "The High Priestess"):
            return 24.0
        if name == "Judgement":
            return 22.0 if len(state.jokers) < joker_limit(state) else 5.0
        target_suit = _SUIT_TAROT_TARGETS.get(name)
        if target_suit is not None:
            return 24.0 if target_suit == self._preferred_suit(state) else 2.0
        enhancement_scores = {
            "Justice": 32.0,
            "The Empress": 30.0,
            "The Hierophant": 30.0,
            "The Chariot": 28.0,
            "The Tower": 26.0,
            "The Magician": 24.0,
            "The Lovers": 22.0,
            "Strength": 20.0,
            "The Devil": 16.0,
        }
        if name in enhancement_scores:
            return enhancement_scores[name]
        if name in _DECK_SHAPING_TAROTS:
            return 20.0
        if name in ("The Fool", "The Wheel of Fortune"):
            return 16.0
        return 8.0

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
