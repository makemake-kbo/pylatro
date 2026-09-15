# Complete-cohort PPO: interim confirmation evidence

This is the **2026-09-15 interim snapshot**, covering training replica 0 and
sampling seeds 0 and 1 on all 400 confirmation game seeds: 800 games per policy.
It compares the two frozen checkpoints after 20 additional PPO updates
(checkpoint update 62). Both panels were complete before analysis.

The [implementation and experiment protocol](complete_cohort.md) were committed
as `1860dda`; [103 CPU tests passed](complete_cohort_verification.md). Training
for all four runs subsequently completed in 5.485 of the authorized nine MPS
hours. At this snapshot, the remaining confirmation evaluations were running on
CPU. No teacher labels or replacement actions were used during PPO.

## Gameplay and initial forecasts

| Metric | Completion-window control | Complete labels |
| --- | ---: | ---: |
| Sampled wins | 247 / 800 (30.875%) | 280 / 800 (35.000%) |
| Mean predicted win probability at game start | 8.701% | 34.351% |
| Mean initial underestimate | 22.174 percentage points | 0.649 percentage points |
| Mean final ante | 5.95 | 6.19 |

The paired sampled difference is **+4.125 percentage points**, with a descriptive
95% interval of **+0.500 to +7.750 points**. Sampling repeats are averaged within
game seed before bootstrapping; the 800 games are not treated as 800 independent
seeds. The interval conditions on this one trained replica and is not adjusted
for repeated interim looks or outcome-dependent stopping.

The initial outcome predictions are much less pessimistic in this subset. This
supports the diagnosed effect of selective terminal labels on early-game
predictions. Matching average win frequency does not establish that the model
accurately distinguishes individual states, nor that PPO's full return estimates
and advantages are correct.

## Broader critic measurements

| Metric | Control | Complete labels |
| --- | ---: | ---: |
| Outcome NLL, game-weighted | 1.5183 | 1.5954 |
| Outcome NLL, decision-weighted | 1.4419 | 1.3288 |
| Win Brier score, game-weighted | 0.1496 | 0.1682 |
| Win Brier score, decision-weighted | 0.2107 | 0.1968 |
| Return MSE, game-weighted | 16.3821 | 19.0284 |

Lower is better for each score. The broader results are mixed. Game-weighted
metrics give each game equal weight; decision-weighted metrics give longer games
more influence. Each policy visits its own state distribution, so these scores
are measurements of its forecasts, rather than a fixed-state causal decomposition.

## Scope and reproducible evidence

The [per-game export](data/complete_cohort_interim_games.jsonl) contains 1,600 rows
with seed/repeat identifiers, outcomes, initial win predictions, decision counts,
and mean critic scores over every recorded decision. Its
[manifest](data/complete_cohort_interim_manifest.json) preserves exact summaries,
source/checkpoint hashes, the frozen selection hash, input artifact hashes, and
the explicit panel. Repeated mean scores can reconstruct both weighting schemes.
Large checkpoints and full trajectories remain in the local study directory.

For example, reproduce the central comparison from the repository root:

```sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np
from pylatro_agent.cohort_study import validate_panel
from pylatro_agent.transfer import paired_statistics, sha256

root = Path('docs/transfer/data')
m = json.loads((root / 'complete_cohort_interim_manifest.json').read_text())
path = root / m['files']['games']
assert sha256(path) == m['files']['games_sha256']
rows = [json.loads(line) for line in path.read_text().splitlines()]
arms = {arm: [r for r in rows if r['arm'] == arm]
        for arm in ('completion_window', 'complete_episode')}
for arm, games in arms.items():
    validate_panel(games, m['scope']['game_seeds'], m['scope']['sampling_seeds'])
    print(arm, 'win rate:', np.mean([r['won'] for r in games]),
          'initial prediction:', np.mean([r['start_win_probability'] for r in games]))
print(paired_statistics(arms['complete_episode'], arms['completion_window']))
PY
```

**This snapshot does not validate the gameplay intervention.** It omits the
second training replica, the third sampling seed, and the treatment's greedy
guardrail. The point estimate is also below the predefined five-point target.
The final comparison must use the full frozen panel; this interim evidence does
not change candidate selection, training settings, or acceptance criteria.
