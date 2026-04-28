from pylatro import load_game_data
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.training.fast_runner import FastRunner
from collections import Counter

data = load_game_data()
agent = HeuristicAgent()

NUM_GAMES = 500

def get_joker_names(state):
    names = []
    for j in state.jokers:
        jc = state.data.centers.get(j.center_key, {})
        names.append(jc.get("name", "?"))
    return names

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

# Separate tracking for deep runs vs shallow runs
deep_joker_counts = Counter()
shallow_joker_counts = Counter()
deep_xmult_rate = 0
shallow_xmult_rate = 0
deep_add_mult = []
shallow_add_mult = []

for seed in range(NUM_GAMES):
    runner = FastRunner(seed, data)
    while not runner.done:
        mask = runner.compute_mask()
        action = agent.select_action(
            runner.state, runner.sub_phase, mask,
            selected_cards=runner.selected_cards,
            pending_action=runner.pending_action,
        )
        runner.step(action)
    
    max_ante = runner.max_ante
    state = runner.state
    names = get_joker_names(state)
    
    if max_ante >= 5:
        deep_xmult_rate += 1 if has_xmult(state) else 0
        deep_add_mult.append(total_add_mult(state))
        for n in names:
            deep_joker_counts[n] += 1
    else:
        shallow_xmult_rate += 1 if has_xmult(state) else 0
        shallow_add_mult.append(total_add_mult(state))
        for n in names:
            shallow_joker_counts[n] += 1

n_deep = sum(1 for seed in range(NUM_GAMES) 
             if FastRunner(seed, data).state.round_resets.ante >= 5)

print(f"=== DEEP vs SHALLOW RUN COMPARISON ({NUM_GAMES} games) ===\n")

# Count deep runs
deep_count = 0
for seed in range(NUM_GAMES):
    runner = FastRunner(seed, data)
    while not runner.done:
        mask = runner.compute_mask()
        action = agent.select_action(
            runner.state, runner.sub_phase, mask,
            selected_cards=runner.selected_cards,
            pending_action=runner.pending_action,
        )
        runner.step(action)
    if runner.max_ante >= 5:
        deep_count += 1

shallow_count = NUM_GAMES - deep_count

print(f"Deep runs (ante 5+): {deep_count}")
print(f"Shallow runs (ante 1-4): {shallow_count}")

if deep_count > 0:
    print(f"\nDeep runs x_mult rate: {deep_xmult_rate/deep_count*100:.1f}%")
    print(f"Deep runs avg add_mult: {sum(deep_add_mult)/len(deep_add_mult):.1f}")
    
print(f"\nShallow runs x_mult rate: {shallow_xmult_rate/shallow_count*100:.1f}")
print(f"Shallow runs avg add_mult: {sum(shallow_add_mult)/len(shallow_add_mult):.1f}")

print(f"\nJokers more common in DEEP runs (top 15):")
all_jokers = set(deep_joker_counts.keys()) | set(shallow_joker_counts.keys())
joker_diffs = []
for j in all_jokers:
    deep_rate = deep_joker_counts[j] / max(deep_count, 1)
    shallow_rate = shallow_joker_counts[j] / max(shallow_count, 1)
    diff = deep_rate - shallow_rate
    joker_diffs.append((j, deep_rate, shallow_rate, diff))

joker_diffs.sort(key=lambda x: -x[3])
for name, deep_r, shallow_r, diff in joker_diffs[:15]:
    print(f"  {name}: deep={deep_r:.2f} shallow={shallow_r:.2f} diff={diff:+.2f}")

print(f"\nJokers more common in SHALLOW runs (top 10):")
for name, deep_r, shallow_r, diff in joker_diffs[-10:]:
    print(f"  {name}: deep={deep_r:.2f} shallow={shallow_r:.2f} diff={diff:+.2f}")
