# Player status and experiment archive

The **public-information hybrid player** keeps the incumbent neural weights
frozen and supports bounded hand search, shop search, and both. Hand-only
assistance with PPO replica 1 passed greedy confirmation: **253/400 wins versus
200/400**, a **13.25 percentage-point gain** with a paired 95% bootstrap interval
of **+8.0 to +18.5 points**. The verdict is **KEEP hand-only assistance**.
The secondary sampled panel finished at **67/128 versus 52/128 wins**.
The simulator defaults are Blue Deck, White Stake,
Ante 8, CPU FP32, temperature 1, and three seconds of planning per decision.
Live integration and model training remain deferred.

Two invalidated attempts are preserved under
`runs/public_hybrid_20261002/invalidated_attempt_{1,2}/`. The report documents
the concealed Joker-order and oversized-hand corrections, and the fresh,
disjoint replacement panels. Search settings and advancement thresholds stayed
fixed throughout; the original 48-hour clock remained in force.

The [gameplay report](public_hybrid_20261002.md) records all arms, paired
statistics, latency, cost, and complete accounting for 2,480 final-study games.
The original clock elapsed **17.95 hours**, including invalidated attempts and
interruptions, within its 48-hour cap. All 169 frozen source files and three
checkpoint hashes were verified, and final statistics were independently
recomputed from saved games. Historical win rates are references: public-information-v2
changes the interpretation of the existing observation tensors.

Run the player:

```sh
.venv/bin/python play.py --checkpoint incumbents/ppo_replica1/checkpoint.pt \
  --public-information --hybrid hand --planning-budget 3 --games 10
```

`--hybrid off`, `hand`, `shop`, and `both` all use the same public observations
and public-sample Joker ordering. Omitting `--public-information` retains the
historical simulator route. Public search uses up to six hand choices or four
shop plans, four paired sampled worlds, and a nonrecursive public heuristic.
Only equally completed sample sets are compared; insufficient search retains
the model proposal. Concealed identities are unknown in snapshots and sampled
without replacement for simulations. Candidate and risk features are public
sample estimates, not hidden-state forecasts or guaranteed clears.

The [study runner](../../tools/hybrid_study.py) reserves all seed panels before
play and resumes completed whole games. Preparation and the technical pilot
precede source/configuration freezing. After a qualifying pilot, use:

```sh
.venv/bin/python tools/resume_hybrid_study.py
.venv/bin/python tools/render_hybrid_report.py
```

The resume entry point converts NumPy summary scalars to JSON while preserving
the frozen driver and gameplay source hashes. It reuses completed game records.

The fixed panels are 128 common baseline seeds, 256 separate architecture
seeds, 400 confirmation seeds if an assisted arm gains at least five greedy
percentage points, and 128 secondary sampled seeds. Confirmation additionally
requires a paired 95% game-bootstrap interval above zero. Errors and missing
games cannot be counted as losses, replaced, or silently dropped. The maximum
gameplay allowance is 48 hours, including interruptions.

## Stable incumbents and demonstrations

The [incumbent registry](../../incumbents/registry.json) records original paths,
SHA-256 hashes, and successful strict loading:

| Registry name | Retained endpoint |
| --- | --- |
| `ppo_replica0` | Terminal GAE, replica 0, update 142 |
| `ppo_replica1` | Terminal GAE, replica 1, update 142 |
| `supervised` | Current-phase supervised baseline, unfrozen encoder, update 4,000 |

The [dataset registry](../../datasets/registry.json) records every retained file
and hash. The original 5,000-game corpus remains at
`cache/datasets/5k-balatro-games`; the 1,000 and additional 4,000 current-rule
games are at `datasets/current_1000` and `datasets/current_4000`. Generation
metadata is retained with each corpus. Original `checkpoints/` directories
were not changed.

## Historical findings

- [Long-horizon credit study](long_horizon_generalization.md): small sampled
  gains did not establish a benefit from full-episode credit over ordinary
  continued training. The fixed acceptance gates failed.
- [Completed supervised transfer study](pretraining_transfer_results_20260923.md):
  further fitting with an unfrozen encoder improved training agreement and one
  greedy comparison; sampled improvement remained unresolved. The later
  confirmation attempt ended incomplete. Its reserved outcomes remain unopened.
- [Corpus audit](original_corpus_feasibility_20260927.md): original and
  current-rule demonstrations differ in pack behavior. More original data
  cannot be treated as additional current-rule trajectories.
- The October 2 review found complementary neural/forecast errors but did not
  establish that a fitted scalar fusion improves gameplay. Its findings and
  recount source are retained under `eval/model_play_review_20261002/` inside
  the evidence archive. The hybrid study does not launch another auxiliary-loss
  experiment if no assisted arm qualifies.

## Retirement receipt and archive

Old `eval/` storage fell from **286,410,649,600 allocated bytes (266.74 GiB)**
to **56,033,280 bytes (53.44 MiB)**. The archive, stable incumbents, and relocated
datasets total approximately 4.63 GiB of retained logical data. Approximately
262 GiB was reclaimed after accounting for retained artifacts.

The [archive index](../../archives/experiments_20261002/README.md) describes the
explicit inventory, verified evidence archive, original working-tree source
archive, removal manifest, and [measured receipt](../../archives/experiments_20261002/receipt.json).
Published reports stay in this directory. Historical links into retired
`eval/` paths resolve to members with the same path inside `evidence.tar.gz`.
Other checkpoints and regenerated arrays were removed, so the retained
evidence does not promise complete replay of every retired experiment.
