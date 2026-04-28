from pylatro import load_game_data
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.training.fast_runner import FastRunner
from pylatro_agent.constants import ActionRange, MAX_JOKER_SLOTS
from pylatro.runtime import joker_limit

data = load_game_data()
agent = HeuristicAgent()

NUM_GAMES = 200

xmult_seen_in_shop = 0
xmult_bought = 0
xmult_couldnt_afford = 0
xmult_no_slots = 0
xmult_in_pack = 0
xmult_pack_picked = 0

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

for seed in range(NUM_GAMES):
    runner = FastRunner(seed, data)
    has_xmult = False
    while not runner.done:
        state = runner.state
        mask = runner.compute_mask()
        
        # Check shop for x_mult jokers
        if not has_xmult and hasattr(state, 'shop') and state.shop:
            all_items = list(state.shop.cards) + list(state.shop.vouchers) + list(state.shop.boosters)
            for item in all_items:
                center = state.data.centers.get(item.center_key, {})
                if center.get("set") == "Joker" and is_xmult_center(center):
                    xmult_seen_in_shop += 1
                    jslots = joker_limit(state) - len(state.jokers)
                    if jslots <= 0:
                        xmult_no_slots += 1
                    elif state.dollars < item.cost:
                        xmult_couldnt_afford += 1
                    break
        
        action = agent.select_action(
            state, runner.sub_phase, mask,
            selected_cards=runner.selected_cards,
            pending_action=runner.pending_action,
        )
        
        # Track x_mult acquisition
        if not has_xmult:
            for j in state.jokers:
                if not j.debuff:
                    jc = state.data.centers.get(j.center_key, {})
                    jcfg = jc.get("config", {})
                    if isinstance(jcfg, dict):
                        xm = jcfg.get("Xmult", 0)
                        exm = 0
                        extra = jcfg.get("extra")
                        if isinstance(extra, dict):
                            exm = extra.get("Xmult", 0)
                        if (xm and xm > 1) or (isinstance(exm, (int, float)) and exm > 1):
                            has_xmult = True
                            xmult_bought += 1
                            break
        
        runner.step(action)

print(f"=== X-MULT ACQUISITION ANALYSIS ({NUM_GAMES} games) ===")
print(f"x_mult seen in shop: {xmult_seen_in_shop} ({xmult_seen_in_shop/NUM_GAMES:.1f} per run)")
print(f"x_mult bought from shop/pack: {xmult_bought} ({xmult_bought/NUM_GAMES*100:.1f}%)")
print(f"x_mult couldn't afford: {xmult_couldnt_afford} ({xmult_couldnt_afford/max(xmult_seen_in_shop,1)*100:.1f}% of sightings)")
print(f"x_mult no slots: {xmult_no_slots} ({xmult_no_slots/max(xmult_seen_in_shop,1)*100:.1f}% of sightings)")
