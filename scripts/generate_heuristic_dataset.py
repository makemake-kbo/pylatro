#!/usr/bin/env python3
"""Generate resumable ZIP shards of complete V68 supervised episodes.

Each ZIP member is a standard save_records pickle, loadable after extraction.
Source snapshots keep long runs independent of subsequent workspace edits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def write_json(path, value):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def generate_shard(task):
    output, index, seeds = task
    import numpy as np
    import torch

    from pylatro import load_game_data
    from pylatro_agent.heuristic import HeuristicAgent
    from pylatro_agent.reward import DEFAULT_REWARD_CONFIG
    from pylatro_agent.tokenizer import Tokenizer
    from pylatro_agent.training.fast_generate import _run_game_single_pass
    from pylatro_agent.training.model_generate import load_records, save_records
    from pylatro_agent.training.supervised import SupervisedConfig, _collate_batch
    from pylatro_agent.vocab import build_vocab

    torch.set_num_threads(1)
    output = Path(output)
    name = f'shard_{index:03d}'
    final = output / f'{name}.zip'
    partial = output / f'{name}.partial.zip'
    progress = output / f'{name}.json'
    if final.exists():
        result = json.loads(progress.read_text())
        if sha256(final) != result['sha256']:
            raise ValueError(f'Checksum mismatch: {final}')
        return result
    rows = []
    if partial.exists():
        with zipfile.ZipFile(partial) as archive:
            rows = [json.loads(archive.read(n)) for n in archive.namelist() if n.endswith('.json')]
        if [r['seed'] for r in rows] != seeds[:len(rows)]:
            raise ValueError('Resume seed mismatch')
    data = load_game_data()
    tokenizer = Tokenizer(vocab=build_vocab(data))
    temp = output / f'{name}.episode.pkl'
    for seed in seeds[len(rows):]:
        started = time.monotonic()
        records, ante, won = _run_game_single_pass(
            seed, data, tokenizer, HeuristicAgent(shop_policy='search', grow_scalers=True),
            .997, win_ante=8, reward_config=DEFAULT_REWARD_CONFIG, deck_key='b_blue', stake=1,
        )
        if not records:
            raise ValueError(f'Empty episode: {seed}')
        for rec in records:
            if not rec['obs']['action_mask'][rec['action']] or not np.isfinite(rec['return_target']):
                raise ValueError(f'Invalid record: {seed}')
            if rec['terminal_outcome_mask'] != 1:
                raise ValueError(f'Stalled episode: {seed}')
            if any(not np.isfinite(v).all() for v in rec['obs'].values()):
                raise ValueError(f'Non-finite observation: {seed}')
        save_records(records, temp, reward_config=DEFAULT_REWARD_CONFIG)
        loaded = load_records(temp, reward_config=DEFAULT_REWARD_CONFIG)
        if len(loaded) != len(records):
            raise ValueError('Round-trip record mismatch')
        batch = _collate_batch(loaded[::max(1, len(loaded)//4)], torch.device('cpu'), SupervisedConfig())
        if any(not torch.isfinite(v).all() for v in batch.values()):
            raise ValueError('Non-finite training batch')
        row = dict(seed=seed, records=len(records), won=won, max_ante=ante,
                   seconds=round(time.monotonic()-started, 3))
        with zipfile.ZipFile(partial, 'a', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            archive.write(temp, f'game_{seed}.pkl')
            archive.writestr(f'game_{seed}.json', json.dumps(row))
        temp.unlink()
        rows.append(row)
        result = dict(file=final.name, complete=False, games=len(rows),
                      records=sum(r['records'] for r in rows), wins=sum(r['won'] for r in rows),
                      last_seed=seed)
        write_json(progress, result)
        print(f'{name}: {len(rows)}/{len(seeds)} games; seed={seed}; {row["seconds"]}s', flush=True)
        del records, loaded, batch
    with zipfile.ZipFile(partial) as archive:
        if len(archive.namelist()) != 2 * len(seeds) or archive.testzip() is not None:
            raise ValueError(f'Archive verification failed: {partial}')
    result = dict(file=final.name, complete=True, games=len(rows),
                  records=sum(r['records'] for r in rows), wins=sum(r['won'] for r in rows),
                  bytes=partial.stat().st_size, sha256=sha256(partial))
    write_json(progress, result)
    partial.replace(final)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--games', type=int, default=100000)
    parser.add_argument('--games-per-shard', type=int, default=5000)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--seed-start', type=int, default=2000000)
    args = parser.parse_args()
    if min(args.games, args.games_per_shard, args.workers) < 1:
        parser.error('counts must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    manifest_path = output / 'manifest.json'
    if not manifest_path.exists():
        reserved = set(json.loads((root / 'eval/heuristic_blue/random_final_holdout_seeds_v1.json').read_text()))
        reserved.update(range(10000, 11000))
        seeds = []
        seed = args.seed_start
        while len(seeds) < args.games:
            if seed not in reserved:
                seeds.append(seed)
            seed += 1
        snapshot = output / 'snapshot'
        shutil.copytree(root / 'src', snapshot / 'src',
                        ignore=shutil.ignore_patterns('__pycache__', '*.c'), dirs_exist_ok=True)
        (snapshot / 'scripts').mkdir(exist_ok=True)
        shutil.copy2(__file__, snapshot / 'scripts' / Path(__file__).name)
        manifest = dict(policy='V68', shop_policy='search', grow_scalers=True, blind_rollout=None,
                        deck='b_blue', stake=1, win_ante=8, gamma=.997, games=args.games,
                        games_per_shard=args.games_per_shard, seeds=seeds,
                        teacher_information='full simulator state and current score/RNG oracle',
                        sources={str(p.relative_to(snapshot)): sha256(p)
                                 for p in snapshot.rglob('*') if p.is_file()})
        write_json(manifest_path, manifest)
    manifest = json.loads(manifest_path.read_text())
    if (args.games, args.games_per_shard) != (manifest['games'], manifest['games_per_shard']):
        raise ValueError('Arguments differ from existing manifest')
    snapshot = output / 'snapshot'
    for name, digest in manifest['sources'].items():
        if sha256(snapshot / name) != digest:
            raise ValueError(f'Snapshot changed: {name}')
    # Re-exec before any game imports, so workers use the frozen implementation.
    if Path(__file__).resolve() != snapshot / 'scripts' / Path(__file__).name:
        import sys
        os.environ['PYTHONPATH'] = str(snapshot / 'src')
        os.execv(sys.executable, [sys.executable, '-u', str(snapshot / 'scripts' / Path(__file__).name), *sys.argv[1:]])
    write_json(output / 'process.json', dict(pid=os.getpid(), workers=args.workers, started=time.time()))
    seeds = manifest['seeds']
    tasks = [(str(output), i, seeds[start:start+args.games_per_shard])
             for i, start in enumerate(range(0, len(seeds), args.games_per_shard))]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(generate_shard, task) for task in tasks]):
            results.append(future.result())
    write_json(output / 'summary.json', dict(
        complete=True, games=sum(r['games'] for r in results), records=sum(r['records'] for r in results),
        wins=sum(r['wins'] for r in results), shards=sorted(results, key=lambda r: r['file'])))


if __name__ == '__main__':
    main()
