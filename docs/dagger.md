# DAgger from V68

The V68 heuristic wins about 74% of current-rule Blue Deck / White Stake games
(2,948 of the 4,000 games in `datasets/current_4000`). The PPO incumbent wins
48–50% greedy, and the supervised student it started from won 32%. Earlier
work fitted the student only on states the *teacher* visited, plus one
100-game correction set. Iterative aggregation on states the *student*
reaches had never been run with the current action heads.

`tools/dagger.py` runs that loop. Each round:

1. **Collect.** Play `--games-per-round` complete games on fresh seeds with
   public-information-v2 observations. At every decision, V68 labels the live
   state. The executed action is the teacher's with probability
   `beta_r = beta_initial * beta_decay^(r-1)` (0.5, 0.25, 0.125, …), and the
   student's otherwise.
2. **Aggregate.** Write every labelled decision to compressed shards under
   `round_NNN/collect/`. Earlier rounds are kept.
3. **Fine-tune.** Starting from the previous round's weights and optimizer
   state, take `--updates-per-round` AdamW steps on the whole aggregate.
   The loss is the cross-entropy of the teacher action under the full action
   grammar, plus `0.1 ×` the terminal-outcome NLL of the game each state came
   from.
4. **Evaluate.** Play the fixed `--panel-games` panel greedily. Round 0 is the
   starting checkpoint on the same panel, so every row of the report is a
   paired comparison against it.

## Running it

```sh
.venv/bin/python tools/dagger.py run \
  --checkpoint incumbents/ppo_replica1/checkpoint.pt \
  --out runs/dagger_v68 --workers 8 \
  --exclude-seeds runs/*/seeds.json runs/*/historical_integer_exclusions.json
```

Defaults: 8 rounds × 512 games, 1,000 panel seeds, 1,000 updates of 256 rows
per round at lr 3e-5 (100-update warmup, then constant). The student samples
during collection, and the teacher sees the full engine state. Training
uses CUDA, then MPS, then CPU. Collection and panel inference use the same
device unless `--collection-device` is set. Up to `--inference-batch` games
(default 256) share one forward pass. Each step waits for all of their V68
labels, and label times are heavy-tailed, so large batches keep the workers
busy. Every option is listed in `tools/dagger.py run --help`.

Rough cost on the M-series Mac, from earlier runs:

| Stage | Estimate |
|---|---|
| Collection (512 games) | About 30–40 min with 8 workers. V68 takes about 0.16 s per decision, so collection is teacher-bound and scales with `--workers`. |
| Fine-tuning (256k rows) | About 30 min on MPS. |
| Panel (1,000 greedy games) | About 10–15 min. |

That is roughly 10 hours for the default 8 rounds.

**Resume** by re-running the same command; unspecified flags take the run
directory's saved values. Finished collections, checkpoints and panels are
reused, and an interrupted collection replays its seeds from the start.
Only `rounds`, `workers`, `device`, `collection_device`, `inference_batch`,
`eval_every` and `panel_modes` may change on resume; anything else is
rejected. `--rounds 12` on resume extends the run. A lock file prevents two
processes from writing the same run directory.

`touch runs/dagger_v68/STOP` stops after the current stage. The next `run`
refuses to start until you delete it.

If V68 fails on more than 1% of decisions, the round stops early. The error
counts and tracebacks are written to `round_NNN/collect_failures.json`, which
a later resume does not delete.

**Report:** `tools/dagger.py report --out runs/dagger_v68` rebuilds the
learning curve from the round directories, so it also works while a run is
in progress. TensorBoard scalars are in `runs/dagger_v68/tensorboard`
(`train/nll`, `collect/greedy_agreement`, `panel_greedy/win_rate`, …).

## Reading the results

- **Collection columns** describe the *pre-update* student on the states it
  reached that round: greedy agreement with V68 and the student's NLL of the
  V68 action. If DAgger is working, agreement rises and NLL falls from round
  to round on newly visited states, not just on the training rows.
- **Behavior wins** include teacher-executed actions. Use them only to see how
  much the mixture is carrying the games.
- **Panel rows** are the result. The win-rate delta and its paired 95%
  interval are against round 0 on the same seeds. Mean antes cleared is a
  lower-variance companion metric: every Ante below the one a game ended in,
  or all of them for a win. It deliberately ignores the env's
  `cleared_antes`, which marks an Ante as soon as the player shops after its
  Big Blind. With 1,000 seeds the paired interval is roughly ±3 points, so
  judge the trend across rounds rather than any single row.
- Each panel summary records its device, batch size, step cap and rules. The
  report warns when a round's conditions differ from round 0's.

## Choices and caveats

- **Teacher view.** `full` (default) is V68 exactly as measured. It sees exact
  scoring randomness, face-down card identities and Joker order under Amber
  Acorn, which the student cannot. `--teacher-view public` labels the
  student's own public projection instead, an information-matched teacher of
  unmeasured strength. In review probes on V68 games, the two views disagreed
  on about 2% of all labels and 5–12% of hand plays. The differences were
  driven by scoring randomness such as Misprint, so `full` labels put a floor
  under the student's achievable NLL on those decisions.
- **No teacher-corpus mixing.** `current_1000`/`current_4000` were recorded
  with privileged observations: the true draw order is encoded positionally
  and masks are full-information. Mixing them with public-v2 states would
  train on inputs the deployed student never sees.
- **Critic.** Outcomes come from the executed mixture policy. They keep the
  terminal-outcome head usable for a later PPO phase, but they are never used
  to weight or select imitation labels. Set `--outcome-loss-coeff 0` for a
  pure actor fit.
- **Storage.** A public observation is 52 KB raw but about 750 bytes
  compressed. Shards stay small on disk, and training holds at most
  `--pool-shards × --shard-rows` rows in memory.
- Seeds come from a private hashed namespace in `[1e9, 2^31)`. Earlier
  hashed studies used the same range, so pass their reservations once when
  creating the run. Every integer in the given JSON/JSONL files is excluded,
  and the list is stored in the run directory:
  `--exclude-seeds runs/*/seeds.json runs/*/historical_integer_exclusions.json`.
