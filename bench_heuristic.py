import os
import time
import multiprocessing
from pylatro import load_game_data
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.training.fast_runner import FastRunner

NUM_GAMES = 500


def _run_batch(args):
    seed_start, seed_end = args
    from pylatro import load_game_data
    from pylatro_agent.heuristic import HeuristicAgent
    from pylatro_agent.training.fast_runner import FastRunner

    data = load_game_data()
    agent = HeuristicAgent()
    results = []
    for seed in range(seed_start, seed_end):
        runner = FastRunner(seed, data)
        while not runner.done:
            mask = runner.compute_mask()
            action = agent.select_action(
                runner.state,
                runner.sub_phase,
                mask,
                selected_cards=runner.selected_cards,
                pending_action=runner.pending_action,
                round_score=runner.round_score,
            )
            runner.step(action)
        results.append((runner.max_ante, runner.won))
    return results


if __name__ == "__main__":
    multiprocessing.freeze_support()
    num_workers = max(1, os.cpu_count() or 1)
    batch_size = (NUM_GAMES + num_workers - 1) // num_workers
    batches = [
        (i * batch_size, min((i + 1) * batch_size, NUM_GAMES))
        for i in range(num_workers)
        if i * batch_size < NUM_GAMES
    ]

    start = time.monotonic()

    if num_workers > 1 and len(batches) > 1:
        with multiprocessing.Pool(len(batches)) as pool:
            batch_results = pool.map(_run_batch, batches)
    else:
        data = load_game_data()
        agent = HeuristicAgent()
        batch_results = []
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
                    round_score=runner.round_score,
                )
                runner.step(action)
            batch_results.append([(runner.max_ante, runner.won)])

    max_antes = []
    wins = 0
    ante_dist = {}
    for batch in batch_results:
        for ante, won in batch:
            max_antes.append(ante)
            if won:
                wins += 1
            ante_dist[ante] = ante_dist.get(ante, 0) + 1

    elapsed = time.monotonic() - start
    gps = NUM_GAMES / elapsed
    workers_used = len(batches) if num_workers > 1 else 1

    print(f"Games: {NUM_GAMES} in {elapsed:.1f}s ({gps:.1f} games/s, {workers_used} workers)")
    print(f"Wins: {wins} ({wins / NUM_GAMES * 100:.1f}%)")
    reach5 = sum(1 for a in max_antes if a >= 5)
    reach4 = sum(1 for a in max_antes if a >= 4)
    print(f"Reached ante >= 5: {reach5} ({reach5 / NUM_GAMES * 100:.1f}%)")
    print(f"Reached ante >= 4: {reach4} ({reach4 / NUM_GAMES * 100:.1f}%)")
    print("Ante distribution:")
    for ante in sorted(ante_dist.keys()):
        print(f"  Ante {ante}: {ante_dist[ante]} ({ante_dist[ante] / NUM_GAMES * 100:.1f}%)")
