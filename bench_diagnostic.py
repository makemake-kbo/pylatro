from pylatro import load_game_data
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.training.fast_runner import FastRunner

data = load_game_data()
agent = HeuristicAgent()

NUM_GAMES = 500
xmult_acquired = 0
xmult_ante_acquired = {}
joker_counts_at_death = {}
planet_usage = {}
runs_with_xmult = {"ante6_plus": 0, "ante4_5": 0, "ante3_minus": 0}

def get_xmult_names(state):
    names = []
    for j in state.jokers:
        if j.debuff:
            continue
        jc = state.data.centers.get(j.center_key, {})
        jcfg = jc.get("config", {})
        if not isinstance(jcfg, dict):
            continue
        xm = jcfg.get("Xmult", 0)
        exm = 0
        extra = jcfg.get("extra")
        if isinstance(extra, dict):
            exm = extra.get("Xmult", 0)
        if (xm and xm > 1) or (isinstance(exm, (int, float)) and exm and exm > 1):
            names.append(jc.get("name", "?"))
    return names

def count_planets_used(state):
    main_type = None
    best_level = 1
    for ht in ("Pair", "Two Pair", "Three of a Kind", "Full House",
                "Flush", "Straight", "High Card", "Four of a Kind"):
        level = state.hands.get(ht, {}).get("level", 1)
        if level > best_level:
            best_level = level
            main_type = ht
    return main_type, best_level

max_antes = []
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
    max_antes.append(max_ante)
    
    # Final state analysis
    state = runner.state
    
    # Check x_mult
    xm_names = get_xmult_names(state)
    if xm_names:
        xmult_acquired += 1
        if max_ante >= 6:
            runs_with_xmult["ante6_plus"] += 1
        elif max_ante >= 4:
            runs_with_xmult["ante4_5"] += 1
        else:
            runs_with_xmult["ante3_minus"] += 1
    
    # Joker count
    n_jokers = len(state.jokers)
    joker_counts_at_death[n_jokers] = joker_counts_at_death.get(n_jokers, 0) + 1
    
    # Planet usage
    main_type, main_level = count_planets_used(state)
    if main_type:
        key = f"{main_type}_lv{main_level}"
        planet_usage[key] = planet_usage.get(key, 0) + 1

print(f"=== DIAGNOSTIC RESULTS ({NUM_GAMES} games) ===")
print(f"\nX-mult acquired: {xmult_acquired} ({xmult_acquired/NUM_GAMES*100:.1f}%)")
print(f"  Runs WITH x_mult by outcome:")
print(f"    Ante 6+:  {runs_with_xmult['ante6_plus']}")
print(f"    Ante 4-5: {runs_with_xmult['ante4_5']}")
print(f"    Ante <=3: {runs_with_xmult['ante3_minus']}")

print(f"\nJoker count distribution:")
for k in sorted(joker_counts_at_death.keys()):
    print(f"  {k} jokers: {joker_counts_at_death[k]} ({joker_counts_at_death[k]/NUM_GAMES*100:.1f}%)")

print(f"\nMost common main type + level (top 15):")
for k, v in sorted(planet_usage.items(), key=lambda x: -x[1])[:15]:
    print(f"  {k}: {v} ({v/NUM_GAMES*100:.1f}%)")

# Ante distribution
ante_dist = {}
for a in max_antes:
    ante_dist[a] = ante_dist.get(a, 0) + 1
print(f"\nAnte distribution:")
for ante in sorted(ante_dist.keys()):
    print(f"  Ante {ante}: {ante_dist[ante]} ({ante_dist[ante]/NUM_GAMES*100:.1f}%)")
