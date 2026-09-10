"""Shop decisions from scoring counterfactuals on public deck samples.

Samples are independent of the run RNG and draw order. They compare complete
joker portfolios, including editions, conditional effects and replacements.
"""

from __future__ import annotations

import math
from copy import deepcopy
from hashlib import blake2b

from pylatro import get_blind_amount, get_poker_hand_info
from pylatro.blind import can_reroll_boss
from pylatro.consumables import use_consumable
from pylatro.flow import _debuff_card, _sort_hand
from pylatro.instances import add_joker, create_consumable_instance, remove_joker
from pylatro.rng import PseudorandomState
from pylatro.runtime import consumable_limit, joker_limit, sell_joker
from pylatro.scoring import _level_up_hand

from .constants import MAX_CONSUMABLE_SLOTS, MAX_JOKER_SLOTS
from .constants import ActionRange as AR


class ShopSearch:
    def __init__(self):
        self.cache = {}
        self.sample_priorities = {}

    def sample_hand(self, cards, size, sample):
        """Common public samples that remain stable when the deck changes.

        Adding one card only changes samples where that card enters the hand;
        it cannot reshuffle every other forecast hand as index sampling did.
        """
        occurrences = {}
        ranked = []
        for card in sorted(cards, key=lambda c: (c.front_key, c.reward_uid)):
            ordinal = occurrences.get(card.front_key, 0)
            occurrences[card.front_key] = ordinal + 1
            key = (sample, card.front_key, ordinal)
            if key not in self.sample_priorities:
                self.sample_priorities[key] = blake2b(
                    f"SHOP_HAND:{sample}:{card.front_key}:{ordinal}".encode(), digest_size=8,
                ).digest()
            ranked.append((self.sample_priorities[key], card.front_key, ordinal, card))

        # UIDs only establish the relative creation order of identical ranks
        # and suits. Their process-global absolute values never enter a hash.
        return [entry[-1] for entry in sorted(ranked, key=lambda entry: entry[:3])[:size]]

    @staticmethod
    def round_capacity(scores, hands, boss_key=""):
        totals = []
        for offset in range(0, len(scores), 3):
            opening, middle, closing = scores[offset:offset + 3]
            if hands <= 1:
                total = opening
            else:
                middle_count = max(0, hands - 2)
                if boss_key == "bl_eye":
                    # The three probes use distinct legal hand types. Do not
                    # multiply the middle probe into illegal repeated plays.
                    middle_count = min(1, middle_count)
                total = opening + middle_count * middle + closing
            totals.append(max(1, total))
        return math.exp(sum(math.log(total) for total in totals) / len(totals))

    @staticmethod
    def round_hands(state, *, boss=False):
        needle = (
            boss and state.round_resets.blind_choices.get("Boss") == "bl_needle"
            and not any(
                j.center_key == "j_chicot" or (j.center_key == "j_luchador" and not j.eternal)
                for j in state.jokers
            )
        )
        hands = 1 if needle else state.round_resets.hands
        return hands + sum(j.extra for j in state.jokers if j.center_key == "j_burglar")

    @staticmethod
    def projected_growth(state, agent):
        """A bounded growth scenario, using only the current public build.

        This values a scaler's contribution to the whole portfolio rather
        than giving every copy the same bonus regardless of existing stats.
        It is a purchase valuation, never a survival forecast.
        """
        rounds = min(8, max(0, (8 - state.round_resets.ante) * 3))
        if not rounds:
            return None
        trial = deepcopy(state, {id(state.data): state.data})
        main = agent._get_main_hand_type(state)
        changed = False
        for joker in trial.jokers:
            key = joker.center_key
            if key in {"j_constellation", "j_hologram"}:
                joker.x_mult += rounds * (0.1 if key == "j_constellation" else 0.125)
            elif key in {"j_green_joker", "j_ride_the_bus", "j_trousers", "j_red_card", "j_flash"}:
                growth = {
                    "j_green_joker": 2, "j_ride_the_bus": 1.5,
                    "j_trousers": 3 if main in {"Two Pair", "Full House"} else 1,
                    "j_red_card": 1.5, "j_flash": 1,
                }[key]
                joker.mult += int(rounds * growth)
            elif key in {"j_square", "j_runner", "j_castle", "j_wee"}:
                growth = {
                    "j_square": 6, "j_runner": 12 if main == "Straight" else 3,
                    "j_castle": 6, "j_wee": 5,
                }[key]
                joker.extra["chips"] += int(rounds * growth)
            elif key == "j_supernova":
                trial.hands[main]["played"] += rounds * 2
            elif key == "j_space":
                chance = min(1, state.probabilities.get("normal", 1) / max(1, joker.extra))
                _level_up_hand(trial, main, int(rounds * 2 * chance))
            elif key == "j_burnt":
                if state.round_resets.discards <= 0 or any(j.center_key == "j_burglar" for j in state.jokers):
                    continue
                _level_up_hand(trial, main, int(rounds * (1 if main == "High Card" else 0.75)))
            elif key == "j_fortune_teller":
                trial.consumeable_usage_total["tarot"] += int(rounds * 0.8)
            elif key == "j_ice_cream":
                joker.extra["chips"] = max(0, joker.extra["chips"] - rounds * 10)
            elif key == "j_popcorn":
                joker.mult = max(0, joker.mult - rounds * 4)
            else:
                continue
            changed = True
        return trial if changed else None

    @staticmethod
    def target(state, agent):
        # There are another shop and payout after each ordinary blind. Budget
        # against the next blind; known boss counters are planned separately.
        blind_key = state.round_resets.blind_choices.get(state.blind_on_deck or "", "")
        blind = state.data.blinds.get(blind_key, state.round_resets.blind or {})
        target = int(get_blind_amount(state.round_resets.ante, min(state.stake, 3)) * blind.get("mult", 1))
        if state.blind_on_deck == "Boss" and any(
            j.center_key == "j_chicot" or (j.center_key == "j_luchador" and not j.eternal)
            for j in state.jokers
        ):
            key = state.round_resets.blind_choices.get("Boss")
            target /= {"bl_wall": 2, "bl_final_vessel": 3}.get(key, 1)
        return target

    def output(self, state, agent, *, boss=False, _project=True, _disabled=False, _activated=False):
        madness = [i for i, j in enumerate(state.jokers) if j.center_key == "j_madness" and not j.debuff]
        if not boss and not _activated and len(madness) == 1:
            index = madness[0]
            victims = [i for i, j in enumerate(state.jokers) if i != index and not j.eternal]
            outcomes = []
            for victim in victims or [None]:
                trial = deepcopy(state, {id(state.data): state.data})
                trial.jokers[index].x_mult += trial.jokers[index].extra
                if victim is not None:
                    remove_joker(trial, trial.jokers[victim])
                score, value = ShopSearch.output(self, trial, agent, _project=_project, _activated=True)
                outcomes.append((score * self.round_hands(trial), value))
            total = math.exp(sum(math.log(max(1, result[0])) for result in outcomes) / len(outcomes))
            return total / self.round_hands(state), sum(result[1] for result in outcomes) / len(outcomes)
        if boss and not _disabled:
            chicot = any(j.center_key == "j_chicot" for j in state.jokers)
            luchador = next(
                (i for i, j in enumerate(state.jokers)
                 if j.center_key == "j_luchador" and not j.eternal and not chicot), None,
            )
            if luchador is not None:
                trial = deepcopy(state, {id(state.data): state.data})
                sell_joker(trial, luchador)
                return ShopSearch.output(self, trial, agent, boss=True, _project=_project, _disabled=True)
            if state.round_resets.blind_choices.get("Boss") == "bl_final_leaf" and not chicot:
                outcomes = []
                for index, joker in enumerate(state.jokers):
                    if joker.eternal:
                        continue
                    trial = deepcopy(state, {id(state.data): state.data})
                    trial.blind_on_deck = "Boss"
                    trial.round_resets.blind = trial.data.blinds["bl_final_leaf"]
                    sell_joker(trial, index)
                    score, value = ShopSearch.output(self, trial, agent, boss=True, _project=_project, _disabled=True)
                    outcomes.append((value, score * self.round_hands(trial, boss=True)))
                if outcomes:
                    value, total = max(outcomes)
                    return total / self.round_hands(state, boss=True), value
        boss_key = state.round_resets.blind_choices.get("Boss", "") if boss else ""
        cash_sensitive = state.modifiers.get("chips_dollar_cap") or any(
            j.center_key in {"j_bull", "j_bootstraps", "j_vagabond"} for j in state.jokers
        )
        owned_keys = {j.center_key for j in state.jokers}
        count_sensitive = bool(owned_keys & {"j_supernova", "j_obelisk"}) or boss_key == "bl_ox"
        main_type = agent._get_main_hand_type(state)
        # Every probe replaces played_this_round. Lifetime hand counts affect
        # this forecast through the chosen plan or Supernova/Obelisk; retaining
        # irrelevant counts and rotating targets prevents reuse across shops.
        hand_key = tuple(
            (name, h.get("level", 1), h.get("chips", 0), h.get("mult", 0),
             h.get("played", 0) if count_sensitive else None, h.get("visible", True))
            for name, h in sorted(state.hands.items())
        )
        key = (
            boss_key, _project, _disabled, _activated,
            state.round_resets.ante,
            agent._joker_keys_sig(state),
            tuple(agent._hashable_extra(j.edition) for j in state.jokers),
            main_type, hand_key,
            (state.hands_played, tuple(
                j.hands_played_at_create for j in state.jokers if j.center_key == "j_loyalty_card"
            ))
            if "j_loyalty_card" in owned_keys else None,
            state.dollars if cash_sensitive else None,
            state.starting_params.hand_size,
            state.round_resets.hands,
            state.round_resets.discards,
            agent._hashable_extra(state.current_round.ancient_card) if "j_ancient" in owned_keys else None,
            agent._hashable_extra(state.current_round.idol_card) if "j_idol" in owned_keys else None,
            agent._hashable_extra(state.probabilities),
            agent._hashable_extra(state.consumeable_usage_total),
            tuple(
                (
                    c.reward_uid, c.front_key, c.center_key, c.seal,
                    c.edition_key, c.perma_bonus, c.played_this_ante if boss else False,
                )
                for c in state.deck_cards
            ),
        )
        if key in self.cache:
            return self.cache[key]
        from .heuristic import HeuristicAgent
        from .joker_layout import apply_best_joker_order

        oracle = HeuristicAgent()
        trial = deepcopy(state, {id(state.data): state.data})
        for joker in trial.jokers:
            joker.debuff = False
        trial.blind_disabled = _disabled or not boss or any(j.center_key == "j_chicot" for j in trial.jokers)
        active_boss = "" if trial.blind_disabled else boss_key
        trial.blind_on_deck = "Boss" if boss else "Small"
        trial.round_resets.blind = state.data.blinds[boss_key if boss else "bl_small"]
        trial.mouth_only_hand = False
        trial.eye_hands = {}
        cards = sorted(
            trial.deck_cards,
            key=lambda c: (c.front_key, c.center_key, c.seal or "", c.edition_key or "", c.perma_bonus),
        )
        scores = []
        future_scores = []
        project_growth = _project and self.projected_growth(state, agent) is not None
        future_oracle = HeuristicAgent() if project_growth else None
        hand_size = trial.starting_params.hand_size - int(active_boss == "bl_manacle")
        hands = self.round_hands(state, boss=boss and not trial.blind_disabled)
        heart_jokers = list(trial.jokers)
        ice_chips = [(j, j.extra["chips"]) for j in trial.jokers if j.center_key == "j_ice_cream"]
        for sample in range(12):
            if sample % 3 == 0:
                trial.eye_hands = {}
            if active_boss == "bl_mouth":
                trial.mouth_only_hand = main_type
            trial.pseudorandom = PseudorandomState(f"SHOP_SAMPLE_{sample}")
            trial.hand_cards = self.sample_hand(cards, min(len(cards), max(3, hand_size)), sample)
            _sort_hand(trial)
            held_ids = {card.reward_uid for card in trial.hand_cards}
            trial.draw_pile = [card for card in cards if card.reward_uid not in held_ids]
            trial.discard_pile = []
            for c in trial.hand_cards:
                c.debuff = c.face_down = c.forced_selection = False
                if boss:
                    _debuff_card(trial, c)
            played = (0, max(1, (hands - 1) // 2), max(0, hands - 1))[sample % 3]
            # Later hands cannot keep Blue Joker's opening draw-pile bonus.
            # Three cards per play approximates the mix of short and full hands.
            spent = min(len(trial.draw_pile), 3 * played)
            trial.discard_pile = trial.draw_pile[:spent]
            trial.draw_pile = trial.draw_pile[spent:]
            for joker, chips in ice_chips:
                joker.extra["chips"] = max(0, chips - played * joker.extra["chip_mod"])
            if active_boss == "bl_final_heart" and heart_jokers:
                disabled = heart_jokers[sample % len(heart_jokers)] if played else None
                for joker in trial.jokers:
                    joker.debuff = joker is disabled
            trial.current_round.hands_left = max(1, hands - played)
            trial.current_round.hands_played = played
            trial.current_round.discards_left = (
                0
                if active_boss == "bl_water"
                or any(j.center_key in {"j_mystic_summit", "j_burglar"} for j in trial.jokers)
                else trial.round_resets.discards
            )
            for name, h in trial.hands.items():
                h["played_this_round"] = int(played > 0 and name == main_type)
            best = tuple(sorted(oracle._cached_best_hand(trial, trial.hand_cards)))
            apply_best_joker_order(trial, best)
            # Ordering changes the preferred hand for Photograph/copy builds.
            best = tuple(sorted(oracle._cached_best_hand(trial, trial.hand_cards)))
            scores.append(max(1, oracle._estimate_hand_score(trial, best)))
            if project_growth:
                future_trial = self.projected_growth(trial, agent)
                # Usually keep the same hand for a conservative projection.
                # Level scalers can change which poker type scores best, so
                # their future probe also reselects the hand.
                future_best = best
                if any(j.center_key in {"j_space", "j_burnt"} for j in trial.jokers):
                    future_best = tuple(sorted(future_oracle._cached_best_hand(future_trial, future_trial.hand_cards)))
                future_scores.append(max(1, future_oracle._estimate_hand_score(future_trial, future_best)))
            if active_boss == "bl_eye":
                hand_name = get_poker_hand_info(trial, [trial.hand_cards[i] for i in best])[0]
                trial.eye_hands[hand_name] = True
        score = self.round_capacity(scores, hands, active_boss) / max(1, hands)
        utility = 0.0
        if _project:
            if future_scores:
                future_score = self.round_capacity(future_scores, hands, active_boss) / max(1, hands)
                utility = 0.65 * max(-0.35, math.log(future_score / score))
            if len(state.jokers) <= 3 and any(j.center_key == "j_riff_raff" for j in state.jokers):
                utility += 0.65 * max(0, 8 - state.round_resets.ante) / 7
        boss_relief = math.log({"bl_wall": 2, "bl_final_vessel": 3}.get(boss_key, 1)) if trial.blind_disabled else 0
        result = (score, math.log(score * max(1, hands)) + utility + boss_relief)
        if len(self.cache) > 400:
            self.cache.clear()
        self.cache[key] = result
        return result

    @staticmethod
    def income_value(state):
        """Value income over a limited horizon, after immediate survival is met."""
        rounds = min(12, max(0, (8 - state.round_resets.ante) * 3))
        income = 0.0
        for joker in state.jokers:
            key = joker.center_key
            if key == "j_rocket":
                income += joker.extra.get("dollars", 1) + min(3, rounds / 3)
            elif key == "j_cloud_9":
                income += sum(c.rank == "9" for c in state.deck_cards)
            else:
                income += {
                    "j_golden": 4,
                    "j_to_the_moon": min(state.dollars, state.interest_cap) // 5,
                    "j_mail": 2,
                    "j_business": 1,
                }.get(key, 0)
        return 0.8 * math.log1p(income * rounds / max(20, state.dollars + 10))

    def purchase_output(self, state, agent):
        before_boss = state.blind_on_deck == "Boss"
        score, value = self.output(state, agent, boss=before_boss)
        if before_boss and state.round_resets.ante < 8:
            # Survive this boss without treating its temporary restriction
            # as permanent. For example, Banner recovers after Water ends.
            _, ordinary_value = self.output(state, agent)
            value = ordinary_value * 0.75 + value * 0.25
        return score, value

    def boss_reroll_improves_risk(self, state, agent):
        """A reroll helps a boss penalty, not an ordinary scoring shortage."""
        if any(
            j.center_key == "j_chicot" or (j.center_key == "j_luchador" and not j.eternal)
            for j in state.jokers
        ):
            return False
        boss_state = deepcopy(state, {id(state.data): state.data})
        boss_state.blind_on_deck = "Boss"
        boss_score, _ = self.output(state, agent, boss=True)
        boss_ratio = boss_score * self.round_hands(state, boss=True) / max(1, self.target(boss_state, agent))
        if boss_ratio >= 1.15:
            return False
        ordinary_score, _ = self.output(state, agent)
        ordinary_target = 2 * get_blind_amount(state.round_resets.ante, min(state.stake, 3))
        ordinary_ratio = ordinary_score * self.round_hands(state) / max(1, ordinary_target)
        return boss_ratio < ordinary_ratio * 0.9

    def select_pack(self, state, mask, agent):
        if not state.pack or not state.pack.cards:
            return None
        if any(state.data.centers[c.center_key].get("set") not in {"Planet", "Joker"} for c in state.pack.cards):
            return None
        # Repeated levels in a reproducible hand beat scattered one-off
        # improvements; retain the established planet plan in Celestial packs.
        if any(state.data.centers[c.center_key].get("set") == "Planet" for c in state.pack.cards):
            return None
        baseline, value = self.purchase_output(state, agent)
        income = self.income_value(state)
        hands = self.round_hands(state, boss=state.blind_on_deck == "Boss")
        safe = baseline * max(1, hands - 1) >= self.target(state, agent) * 1.3
        best = (-0.001, AR.PACK_SKIP)
        red_cards = [j for j in state.jokers if j.center_key == "j_red_card"]
        if red_cards:
            trial = deepcopy(state, {id(state.data): state.data})
            for joker in trial.jokers:
                if joker.center_key == "j_red_card":
                    joker.mult += joker.extra
            _, skipped = self.purchase_output(trial, agent)
            best = (skipped - value, AR.PACK_SKIP)
        for index, card in enumerate(state.pack.cards):
            claim = AR.PACK_CLAIM_START + index
            slots = [None] if mask[claim] else [
                i for i in range(min(len(state.jokers), MAX_JOKER_SLOTS))
                if mask[AR.SHOP_SELL_JOKER_START + i]
            ]
            for slot in slots:
                trial = deepcopy(state, {id(state.data): state.data})
                if slot is not None:
                    trial.dollars += trial.jokers[slot].sell_cost
                    remove_joker(trial, trial.jokers[slot])
                # Selling a Negative joker also removes its extra slot.
                if len(trial.jokers) >= joker_limit(trial) and not (card.edition and card.edition.get("negative")):
                    continue
                add_joker(trial, card.center_key, edition=card.edition)
                _, improved = self.purchase_output(trial, agent)
                gain = improved - value
                if safe:
                    gain += self.income_value(trial) - income
                if gain > best[0]:
                    action = claim if slot is None else AR.SHOP_SELL_JOKER_START + slot
                    best = (gain, action)
        return best[1]

    def select(self, state, mask, agent):
        dollars = state.dollars
        ante = state.round_resets.ante
        main = agent._get_main_hand_type(state)
        owned = {j.center_key for j in state.jokers}
        items = list(state.shop.cards) + list(state.shop.vouchers) + list(state.shop.boosters)
        for index, item in enumerate(items):
            action = AR.SHOP_BUY_START + index
            if not mask[action]:
                continue
            center = state.data.centers[item.center_key]
            name = center.get("name")
            if name == "The Fool":
                center = state.data.centers.get(state.last_tarot_planet, {})
                name = center.get("name")
            payout = 0
            if name == "The Hermit" and dollars - item.cost >= 10:
                payout = min(dollars - item.cost, center["config"]["extra"])
            elif name == "Temperance":
                payout = min(sum(j.sell_cost for j in state.jokers), center["config"]["extra"])
            if payout - item.cost >= 2:
                self.last_decision = {"reason": "spendable income", "net_dollars": payout - item.cost}
                return action
        before_boss = state.blind_on_deck == "Boss"
        reroll_vouchers = [
            (index, item) for index, item in enumerate(items)
            if item.center_key in {"v_directors_cut", "v_retcon"}
            and mask[AR.SHOP_BUY_START + index] and dollars - item.cost >= 10
        ]
        boss_risk = False
        has_counter = any(
            j.center_key == "j_chicot" or (j.center_key == "j_luchador" and not j.eternal)
            for j in state.jokers
        )
        if not has_counter and (can_reroll_boss(state) or reroll_vouchers):
            boss_risk = self.boss_reroll_improves_risk(state, agent)
            if boss_risk and reroll_vouchers and not can_reroll_boss(state):
                self.last_decision = {"reason": "prepare boss reroll", "boss": state.round_resets.blind_choices["Boss"]}
                return AR.SHOP_BUY_START + reroll_vouchers[0][0]
        baseline, base_value = self.purchase_output(state, agent)
        target = self.target(state, agent)
        forecast_hands = self.round_hands(state, boss=before_boss)
        safety_factor = 1.3 if len(state.jokers) < 2 or ante >= 5 else 1.1
        safe = baseline * max(1, forecast_hands - 1) >= target * safety_factor
        base_income = self.income_value(state) if safe else 0
        reserve = min(state.interest_cap, 25) if ante < 7 else (10 if ante == 7 else 0)
        self.last_decision = {
            "forecast_total": round(baseline * forecast_hands),
            "target": target,
            "safe": safe,
            "reserve": reserve,
        }
        boss_reserve = 0
        if can_reroll_boss(state) and boss_risk:
            boss_reserve = 10
        best = (0.03, -1)
        if owned & {"j_stencil", "j_campfire"}:
            for slot in range(min(len(state.jokers), MAX_JOKER_SLOTS)):
                action = AR.SHOP_SELL_JOKER_START + slot
                if not mask[action]:
                    continue
                trial = deepcopy(state, {id(state.data): state.data})
                sell_joker(trial, slot)
                trial_score, value = self.purchase_output(trial, agent)
                gain = value - base_value
                if safe:
                    gain += self.income_value(trial) - base_income
                if before_boss and baseline * forecast_hands < target * 1.15:
                    trial_capacity = trial_score * self.round_hands(trial, boss=True) / self.target(trial, agent)
                    if trial_capacity >= 1.15:
                        # A joker that recovers next round cannot help if its
                        # occupied Stencil slot prevents surviving this boss.
                        gain = max(gain, math.log(trial_capacity / (baseline * forecast_hands / target)))
                if trial_score >= baseline and gain > best[0]:
                    best = (gain, action)
        for i, item in enumerate(items):
            center = state.data.centers[item.center_key]
            fool_planet = (
                center.get("name") == "The Fool"
                and state.data.centers.get(state.last_tarot_planet, {}).get("set") == "Planet"
            )
            if center.get("set") == "Planet" or fool_planet:
                action = AR.SHOP_BUY_START + i
                if mask[action] and dollars - item.cost >= boss_reserve:
                    trial = deepcopy(state, {id(state.data): state.data})
                    trial.dollars -= item.cost
                    used = use_consumable(trial, create_consumable_instance(trial, item.center_key))
                    if fool_planet and used.created_consumables:
                        use_consumable(trial, len(trial.consumables) - 1)
                    _, value = self.purchase_output(trial, agent)
                    gain = value - base_value - (0.04 if safe else 0.012) * item.cost
                    if gain > best[0]:
                        best = (gain, action)
                continue
            if center.get("set") != "Joker":
                continue
            replace = (
                [None]
                if len(state.jokers) < joker_limit(state) or (item.edition and item.edition.get("negative"))
                else [j for j in range(min(len(state.jokers), MAX_JOKER_SLOTS)) if not state.jokers[j].eternal]
            )
            for slot in replace:
                proceeds = 0 if slot is None else state.jokers[slot].sell_cost
                if dollars + proceeds - item.cost < boss_reserve:
                    continue
                trial = deepcopy(state, {id(state.data): state.data})
                if slot is not None:
                    remove_joker(trial, trial.jokers[slot])
                # Selling a Negative joker also removes its extra slot.
                if len(trial.jokers) >= joker_limit(trial) and not (item.edition and item.edition.get("negative")):
                    continue
                trial.dollars += proceeds - item.cost
                add_joker(trial, item.center_key, edition=item.edition)
                trial_score, value = self.purchase_output(trial, agent)
                trial_hands = self.round_hands(trial, boss=before_boss)
                trial_target = self.target(trial, agent)
                if (
                    baseline * forecast_hands < target
                    and trial_score * trial_hands / trial_target <= baseline * forecast_hands / target * 1.02
                    and item.center_key != "j_riff_raff"
                ):
                    # A future-only investment cannot solve an immediate
                    # scoring shortage. Keep the cash for a usable upgrade.
                    continue
                if (
                    trial_score / trial_target < baseline / target
                    and trial_score * max(1, trial_hands - 1) < trial_target * 1.1
                ):
                    continue
                trial_income = self.income_value(trial) if safe else 0
                if trial_income > base_income and trial_score * max(1, trial_hands - 1) < trial_target * safety_factor:
                    trial_income = base_income
                price_penalty = (0.03 if safe else 0.012) * max(0, item.cost - proceeds)
                # Spending below maximum interest costs future buying power.
                if safe and ante < 7:
                    price_penalty += 0.004 * max(0, min(dollars, 25) - max(0, trial.dollars)) * (8 - ante)
                gain = value - base_value + trial_income - base_income - price_penalty
                action = AR.SHOP_BUY_START + i if slot is None else AR.SHOP_SELL_JOKER_START + slot
                if mask[action] and gain > best[0]:
                    best = (gain, action)
        if best[1] >= 0:
            return best[1]
        # Empty inventory space for cash or a useful planet before buying packs.
        if len(state.consumables) >= consumable_limit(state):
            for ci, cons in enumerate(state.consumables[:MAX_CONSUMABLE_SLOTS]):
                center = state.data.centers[cons.center_key]
                if center.get("set") == "Planet" and center.get("config", {}).get("hand_type") != main:
                    return AR.SHOP_SELL_CONSUMABLE_START + ci
        for i, item in enumerate(items):
            action = AR.SHOP_BUY_START + i
            if not mask[action]:
                continue
            center = state.data.centers[item.center_key]
            name = center.get("name", "")
            cset = center.get("set")
            if dollars - item.cost < boss_reserve:
                continue
            if name in {"The Hermit", "Temperance"} and dollars >= item.cost + 5:
                return action
            if cset == "Planet" and center["config"]["hand_type"] == main:
                return action
            if item.center_key in {
                "v_telescope",
                "v_grabber",
                "v_paint_brush",
                "v_overstock_norm",
                "v_seed_money",
                "v_hone",
                "v_clearance_sale",
                "v_directors_cut",
            } and dollars - item.cost >= max(10, reserve):
                return action
            if cset == "Booster":
                if (
                    (("Standard" in name and "j_hologram" in owned)
                     or ("Celestial" in name and "j_constellation" in owned))
                    and dollars - item.cost >= 5
                ):
                    return action
                if "j_red_card" in owned and item.cost <= 6 and dollars - item.cost >= (10 if safe else 5):
                    return action
                if (
                    "Buffoon" in name
                    and (len(state.jokers) < joker_limit(state) or any(not j.eternal for j in state.jokers))
                    and dollars - item.cost >= (reserve if safe else 0)
                ):
                    return action
                if "Celestial" in name and (safe or ante >= 2) and dollars - item.cost >= min(reserve, 10):
                    return action
                if "Arcana" in name and dollars - item.cost >= max(5, reserve):
                    return action
                if "Standard" in name and dollars - item.cost >= max(25, reserve):
                    return action
        reroll = state.current_round.reroll_cost
        reroll_buffer = 0 if ante >= 8 and before_boss and not safe else 5
        if (
            mask[AR.SHOP_REROLL]
            and dollars - reroll >= max(reserve if safe else reroll_buffer, boss_reserve + reroll_buffer)
            and state.current_round.reroll_cost_increase < 5
        ):
            return AR.SHOP_REROLL
        return AR.SHOP_LEAVE
