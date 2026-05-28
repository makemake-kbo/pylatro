from pylatro import load_game_data
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.training.fast_runner import FastRunner
from pylatro_agent.constants import ActionRange, MAX_JOKER_SLOTS
from pylatro.runtime import joker_limit

data = load_game_data()
agent = HeuristicAgent()

NUM_GAMES = 100

def is_xmult_center(center):
    cfg = center.get("config", {})
    if not isinstance(cfg, dict):
        return False
    xm = cfg.get("Xmult", 0)
    if xm and xm > 1:
        return True
    extra = cfg.get("extra")
    if isinstance(extra, dict):
        exm = extra.get("Xmult", 0)
        if isinstance(exm, (int, float)) and exm > 1:
            return True
    return False

def has_xmult_joker(state):
    for j in state.jokers:
        if j.debuff:
            continue
        jc = state.data.centers.get(j.center_key, {})
        jcfg = jc.get("config", {})
        if isinstance(jcfg, dict):
            xm = jcfg.get("Xmult", 0)
            if xm and xm > 1:
                return True
            extra = jcfg.get("extra")
            if isinstance(extra, dict):
                exm = extra.get("Xmult", 0)
                if isinstance(exm, (int, float)) and exm > 1:
                    return True
    return False

# Track WHY x_mult isn't bought
reasons = {
    "bought": 0,
    "no_slots_sold_and_bought": 0,
    "no_slots_sell_not_in_mask": 0,
    "no_slots_cant_afford_with_sell": 0,
    "no_slots_no_sellable_joker": 0,
    "no_slots_eternal": 0,
    "could_afford_but_no_buy": 0,
    "not_seen": 0,
}

for seed in range(NUM_GAMES):
    runner = FastRunner(seed, data)
    run_got_xmult = False
    while not runner.done:
        state = runner.state
        mask = runner.compute_mask()
        
        if not run_got_xmult and hasattr(state, 'shop') and state.shop:
            all_items = list(state.shop.cards) + list(state.shop.vouchers) + list(state.shop.boosters)
            for item in all_items:
                center = state.data.centers.get(item.center_key, {})
                if center.get("set") == "Joker" and is_xmult_center(center):
                    jslots = joker_limit(state) - len(state.jokers)
                    if jslots > 0:
                        if state.dollars >= item.cost:
                            pass  # Should be bought at priority 1
                        else:
                            reasons["could_afford_but_no_buy"] += 1
                    else:
                        # Check why we can't sell
                        sellable = False
                        for j in state.jokers[:MAX_JOKER_SLOTS]:
                            if j.eternal:
                                continue
                            sellable = True
                            break
                        if not sellable:
                            reasons["no_slots_eternal"] += 1
                        else:
                            # Check if we can afford with sell value
                            worst_val = 0
                            for j in state.jokers[:MAX_JOKER_SLOTS]:
                                if not j.eternal:
                                    worst_val = max(worst_val, j.sell_cost)
                            if state.dollars + worst_val < item.cost:
                                reasons["no_slots_cant_afford_with_sell"] += 1
                            else:
                                # Sell should work - check mask
                                sell_action = -1
                                for i, j in enumerate(state.jokers[:MAX_JOKER_SLOTS]):
                                    if not j.eternal:
                                        sa = ActionRange.SHOP_SELL_JOKER_START + i
                                        if mask[sa]:
                                            sell_action = sa
                                            break
                                if sell_action < 0:
                                    reasons["no_slots_sell_not_in_mask"] += 1
                                else:
                                    # Should sell then buy
                                    reasons["no_slots_sold_and_bought"] += 1
                    break
        
        action = agent.select_action(
            state, runner.sub_phase, mask,
            selected_cards=runner.selected_cards,
            pending_action=runner.pending_action,
        )
        
        if not run_got_xmult and has_xmult_joker(state):
            run_got_xmult = True
            reasons["bought"] += 1
        
        runner.step(action)
    
    if not run_got_xmult:
        reasons["not_seen"] += 1  # Never got x_mult

print(f"=== X-MULT BUY FAILURE ANALYSIS ({NUM_GAMES} games) ===")
total = sum(reasons.values())
for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v} ({v/total*100:.1f}%)")
print(f"\nTotal: {total}")
print(f"Bought rate: {reasons['bought']/NUM_GAMES*100:.1f}%")
