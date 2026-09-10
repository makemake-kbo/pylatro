# V68 bootstrap dataset

The initial local dataset is complete: **300 Blue Deck / White Stake games,
230 wins, 70 losses, and 68,915 training decisions** in 30 shards (3.64 GB).
It is in `data/pretraining/v68_development300`; large generated artifacts are
not committed. The compact report and shard hashes are in
[`pretraining_v68_summary.json`](pretraining_v68_summary.json), and exact training
seeds are in [`pretraining_v68_seeds.json`](pretraining_v68_seeds.json).

V68 won 150/200 sequential development games and 80/100 randomly sampled
development games. These are development measurements; the requested 90%
full-game win rate has **not** been achieved. Reserved random final-validation
seeds and seeds 10000–10999 were excluded from generation and remain untouched.

The winning games contain 229 distinct final joker combinations, 78 joker keys,
and all five Ante 8 bosses. They are strongly biased toward Pair: it is the
most frequently played hand in Antes 6–8 for 195/230 wins. Other dominant late
hands are Two Pair (25), High Card (7), Straight (2), and Flush (1). This is useful
bootstrap data, not balanced coverage of hand strategies, decks or stakes.

The teacher uses search shops and scaler growth, with blind rollouts off. It
uses full simulator state and a current-score/RNG oracle. Public continuation
samples replace future draw order and RNG independently. The constructor still
defaults to the legacy policy; select `shop_policy="search", grow_scalers=True`
explicitly to use this teacher. Experimental actors and rollouts remain opt-in.

Every saved action and outcome label was checked against the complete source
trace. All actions are legal, all observations and returns are finite, and every
shard was reloaded with the training reader and exercised through its collator.
An independent live-policy/recording comparison matched all 689 decisions on
three complete games (two wins, one loss). No student has yet been trained or
evaluated on this dataset.

## Loading the local dataset

The local dataset was generated using the workspace's **tokenizer v13**
(`v13_phase_risk_economy_card_state`). That observation-schema work is separate
from the heuristic commit. `load_records` deliberately rejects incompatible
schemas or reward configurations. Use a matching v13 checkout for these local
shards, or regenerate records with the intended checkout. The export tool itself
uses the active checkout's tokenizer and reward metadata.

```python
import json
from pathlib import Path
from pylatro_agent.reward import DEFAULT_REWARD_CONFIG
from pylatro_agent.training.model_generate import load_records

root = Path('data/pretraining/v68_development300')
summary = json.loads((root / 'summary.json').read_text())
records = []
for shard in summary['shards']:
    records.extend(load_records(root / shard['file'],
                                reward_config=DEFAULT_REWARD_CONFIG))
assert len(records) == 68_915
# Supply records to train_supervised with gamma=.997, win_ante=8,
# and reward_config=DEFAULT_REWARD_CONFIG.
```

Both wins and losses are retained for outcome-weighted imitation and terminal
value targets. Keep complete seeds together in any split. Loading every shard
requires several GB of memory. Do not reuse the training seeds as independent
student evaluation.

`scripts/bench_heuristic.py` produces complete action traces and source manifests.
`scripts/check_heuristic_replay.py` checks live-versus-recorded policy parity.
`scripts/export_heuristic_traces.py` turns complete V68 traces into verified,
seed-labelled training shards and rejects overlap with a supplied reserved-seed
file. Its output manifest records input hashes, configuration, seeds and source/
binary hashes. It does not resample decisions or count replays as fresh wins.
