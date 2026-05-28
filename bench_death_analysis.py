from pylatro import load_game_data
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.training.fast_runner import FastRunner

data = load_game_data()
agent = HeuristicAgent()

NUM_GAMES = 200

def has_xmult(state):
    for j in state.jokers:
        if j.debuff:
            continue
        jc = state.data.centers.get(j.center_key, {})
        jcfg = jc.get("config", {})
        if not isinstance(jcfg, dict):
            continue
        xm = jcfg.get("Xmult", 0)
        if xm and xm > 1:
            return True
        extra = jcfg.get("extra")
        if isinstance(extra, dict):
            exm = extra.get("Xmult", 0)
            if isinstance(exm, (int, float)) and exm > 1:
                return True
    return False

def total_add_mult(state):
    total = 0
    for j in state.jokers:
        if j.debuff:
            continue
        if j.mult:
            total += j.mult
        if j.t_mult:
            total += j.t_mult
        jc = state.data.centers.get(j.center_key, {})
        jcfg = jc.get("config", {})
        if isinstance(jcfg, dict):
            extra = jcfg.get("extra")
            if isinstance(extra, dict):
                em = extra.get("mult")
                if isinstance(em, (int, float)) and em:
                    total += em
    return total

def main_type_level(state):
    best_ht = "Pair"
    best_lv = 1
    for ht in ("Pair", "Two Pair", "Three of a Kind", "High Card",
                "Flush", "Straight", "Full House", "Four of a Kind"):
        lv = state.hands.get(ht, {}).get("level", 1)
        if lv > best_lv:
            best_lv = lv
            best_ht = ht
    pair_lv = state.hands.get("Pair", {}).get("level", 1)
    twopair_lv = state.hands.get("Two Pair", {}).get("level", 1)
    return pair_lv, twopair_lv

# Track death reasons by ante
death_profiles = {}
for ante in range(1, 9):
    death_profiles[ante] = {
        "count": 0,
        "has_xmult": 0,
        "avg_jokers": [],
        "avg_add_mult": [],
        "pair_lv": [],
        "twopair_lv": [],
        "dollars": [],
        "consumables": [],
    }

for seed in range(NUM_GAMES):
    runner = FastRunner(seed, data)
    while not runner.done:
        mask = runner.compute_mask()
        action = agent.select_action(
            runner.state,
            runner.sub_phase,
            mask,
            selected_cards=runner.selected_cards,
            pending_action=runner.pending_action,
        )
        runner.step(action)
    
    max_ante = runner.max_ante
    state = runner.state
    
    prof = death_profiles[max_ante]
    prof["count"] += 1
    if has_xmult(state):
        prof["has_xmult"] += 1
    prof["avg_jokers"].append(len(state.jokers))
    prof["avg_add_mult"].append(total_add_mult(state))
    plv, tplv = main_type_level(state)
    prof["pair_lv"].append(plv)
    prof["twopair_lv"].append(tplv)
    prof["dollars"].append(state.dollars)
    prof["consumables"].append(len(state.consumables))

print(f"=== DEATH ANALYSIS ({NUM_GAMES} games) ===\n")
for ante in sorted(death_profiles.keys()):
    prof = death_profiles[ante]
    if prof["count"] == 0:
        continue
    n = prof["count"]
    avg_j = sum(prof["avg_jokers"]) / n
    avg_am = sum(prof["avg_add_mult"]) / n
    avg_plv = sum(prof["pair_lv"]) / n
    avg_tplv = sum(prof["twopair_lv"]) / n
    avg_dol = sum(prof["dollars"]) / n
    avg_cons = sum(prof["consumables"]) / n
    xm_pct = prof["has_xmult"] / n * 100
    print(f"Ante {ante} deaths: {n} ({n/NUM_GAMES*100:.1f}%)")
    print(f"  x_mult: {xm_pct:.0f}% | avg_jokers: {avg_j:.1f} | avg_add_mult: {avg_am:.1f}")
    print(f"  pair_lv: {avg_plv:.1f} | twopair_lv: {avg_tplv:.1f} | dollars: {avg_dol:.1f} | consumables: {avg_cons:.1f}")
