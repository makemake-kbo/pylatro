from pylatro import load_game_data
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.training.fast_runner import FastRunner
from pylatro_agent.subset_actions import subset_index
from pylatro_agent.constants import ActionRange

data = load_game_data()
agent = HeuristicAgent()

NUM_GAMES = 200

def estimate_best_hand_score(state):
    """Estimate the best hand score the agent could achieve."""
    hand = state.hand_cards
    if not hand:
        return 0, "No hand", 0
    
    from pylatro.scoring import RANK_TO_NOMINAL
    best_score = 0
    best_type = "High Card"
    best_chips = 0
    
    from itertools import combinations
    max_cards = min(5, len(hand))
    for size in range(max_cards, 0, -1):
        for combo in combinations(range(len(hand)), size):
            cards = [hand[i] for i in combo]
            score = agent._estimate_hand_score(state, combo)
            if score > best_score:
                best_score = score
                ht = agent._quick_hand_quality(state, cards)
                best_type = ht
    
    return best_score, best_type, len(hand)

# Track score gaps at each ante
ante_gaps = {}
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
    
    best_score, best_type, hand_size = estimate_best_hand_score(state)
    hands_left = state.current_round.hands_left
    blind_target = agent._get_blind_target(state)
    
    gap = blind_target - best_score * max(hands_left, 1)
    
    if max_ante not in ante_gaps:
        ante_gaps[max_ante] = []
    
    ante_gaps[max_ante].append({
        "gap": gap,
        "best_score": best_score,
        "hands_left": hands_left,
        "blind_target": blind_target,
        "best_type": best_type,
        "hand_size": hand_size,
        "jokers": len(state.jokers),
        "dollars": state.dollars,
    })

print(f"=== SCORE GAP ANALYSIS ({NUM_GAMES} games) ===\n")
for ante in sorted(ante_gaps.keys()):
    runs = ante_gaps[ante]
    n = len(runs)
    avg_gap = sum(r["gap"] for r in runs) / n
    avg_score = sum(r["best_score"] for r in runs) / n
    avg_target = sum(r["blind_target"] for r in runs) / n
    avg_hands = sum(r["hands_left"] for r in runs) / n
    avg_jokers = sum(r["jokers"] for r in runs) / n
    
    type_counts = {}
    for r in runs:
        t = r["best_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print(f"Ante {ante} deaths: {n}")
    print(f"  avg best_hand_score: {avg_score:.0f} | avg blind_target: {avg_target:.0f}")
    print(f"  avg gap (target - score*hands): {avg_gap:.0f} | avg hands_left: {avg_hands:.1f}")
    print(f"  avg jokers: {avg_jokers:.1f}")
    print(f"  hand types: {dict(sorted(type_counts.items(), key=lambda x: -x[1]))}")
    print()
