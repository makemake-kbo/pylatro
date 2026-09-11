# Faster heuristic game generation

The user clarified that this task is speed only. The final patch changes state
copying and its optional Cython build; heuristic policy, sampling budgets and
RNG behavior are unchanged. The separate RNG-cache experiment was excluded,
and the original heuristic source and extension were restored.

## Implementation

Scoring probes repeatedly copy nested state dictionaries and lists. Profiling
one stronger-policy game attributed about 28 of 35 profiled seconds to state
copying. The old model copier skipped scalar dataclass fields but dispatched
through Python's general `deepcopy` machinery for scalars inside containers.

`pylatro._state_copy` now copies builtin lists and dictionaries directly, skipping
recursive dispatch for exact immutable scalar types. It preserves mutable
isolation, aliases, cycles, custom copy hooks and memo lifetime. Model classes
remain ordinary Python dataclasses. The same helper runs in Python or as an
optional extension included by `scripts/compile_cython.py`.

The new extension is built in this workspace. To rebuild elsewhere:

```sh
.venv/bin/python scripts/compile_cython.py
```

## Measurements

| Comparison | Workload | Before | After | Less time |
| --- | --- | ---: | ---: | ---: |
| Original → optimized Python copying | 20 legacy games, seeds 0–19, two workers | 14.55 s | 11.69 s | 19.6% |
| Original → optimized Python copying | Three stronger-teacher games, seeds 0–2 | 239.40 s | 194.50 s | 18.8% |
| Original → optimized Python copying | Recording the same three teacher games | 311.54 s | 264.97 s | 15.0% |
| Optimized Python → Cython copying | Two stronger-teacher games, seeds 0 and 2 | 58.37 s | 46.58 s | 20.2% |
| Optimized Python → Cython copying | 6,000 scoring-state copies | 1.65 s | 1.10 s | 33.5% |
| Optimized Python → Cython copying, final policy | Recording seed 0 | 25.44 s | 22.72 s | 10.7% |

The percentages compare each row's two implementations, not a shared baseline.
These are small local timing panels, not confidence intervals. Initial teacher
play timings were sequential, with alternating implementation order. Later
recording timings overlapped lower-priority validation/build work on a 16-CPU
host, so treat their exact magnitudes as indicative. The native two-game panel
used the temporary RNG fix; its actions were unchanged on these seeds. Final
source/native recording validation uses the restored original heuristic.

Raw measurements are in `2026-09-10-state-copy.json`,
`2026-09-10-state-copy-teacher.json`, and
`2026-09-10-state-copy-native.json` beside this report.

## Correctness and reproduction

The 20 legacy games matched all **2,915 actions and complete trace fields**.
The three stronger-teacher games matched their full traces and **all 601
training records**, including observation arrays, actions, rewards, returns,
outcomes and metadata. The two native-copy games matched all **344 actions
and complete trace fields**.

New tests compare copied state against standard deepcopy with the model hook
removed and cover graph cycles, shared references, custom containers, source
isolation and caller-supplied memo entries. They passed with both the Python
helper and its compiled extension. Broad engine, heuristic and generation
checks also passed during development; final validation is recorded below.

The repeatable source/native benchmark includes warmup, alternating run order,
exact parity assertions and helper source/binary hashes:

```sh
.venv/bin/python scripts/bench_state_copy.py --games 1 \
  --output /tmp/native-copy-records-final.json
```

Without `--games`, it measures just the scoring copies. With `--games`, it also
compares complete stronger-teacher training records beginning at `--seed`
(default 0), using Blue Deck, White Stake, search shops and scaler growth.
Timing excludes the parity comparison. The benchmark restores the normal
copy implementation even if a comparison fails.

## Final validation

The final speed-only worktree passed **325 engine/heuristic/generation tests**
in 61.11 seconds. Ruff and the relevant whitespace checks passed. The final
source/native recording benchmark passed exact equality for all 125 records
of seed 0, reducing time from 25.44 to 22.72 seconds. Its raw rows, parity flag
and verified source/extension hashes are in `2026-09-10-state-copy-final.json`.

The imported heuristic extension SHA256 is the original
`19e0adcb76f2953521a325316dd292cadb09d03dafc44a66495353d766112204`.
There is no heuristic source diff. The compiled `_state_copy` extension is
active. No strength experiment or win-rate increase is part of this change.
