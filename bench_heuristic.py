import time
from pylatro import load_game_data
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.training.fast_runner import FastRunner

data = load_game_data()
agent = HeuristicAgent()

NUM_GAMES = 500
max_antes = []
wins = 0
ante_dist = {}
start = time.monotonic()

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
            pending_consumable_slot=runner.pending_consumable_slot,
        )
        runner.step(action)
    max_antes.append(runner.max_ante)
    if runner.won:
        wins += 1
    ante = runner.max_ante
    ante_dist[ante] = ante_dist.get(ante, 0) + 1

elapsed = time.monotonic() - start
gps = NUM_GAMES / elapsed

print(f"Games: {NUM_GAMES} in {elapsed:.1f}s ({gps:.1f} games/s)")
print(f"Wins: {wins} ({wins / NUM_GAMES * 100:.1f}%)")
reach5 = sum(1 for a in max_antes if a >= 5)
reach4 = sum(1 for a in max_antes if a >= 4)
print(f"Reached ante >= 5: {reach5} ({reach5 / NUM_GAMES * 100:.1f}%)")
print(f"Reached ante >= 4: {reach4} ({reach4 / NUM_GAMES * 100:.1f}%)")
print("Ante distribution:")
for ante in sorted(ante_dist.keys()):
    print(f"  Ante {ante}: {ante_dist[ante]} ({ante_dist[ante] / NUM_GAMES * 100:.1f}%)")
