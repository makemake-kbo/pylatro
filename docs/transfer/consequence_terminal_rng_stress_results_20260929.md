# Recorded terminal preferences are sensitive to hidden RNG interventions

The [terminal RNG stress diagnostic](consequence_terminal_rng_stress_20260929.md) is complete and independently audited. With identical starting policy observations, **153/384 terminal outcomes change** relative to their original realization, and **63/96 proposed actions** produce both wins and losses across the four new RNG interventions. Original win/loss ordering persists in 17 of 128 pair retests, reverses in 12, and becomes a tie in 99. This demonstrates substantial sensitivity under the declared intervention. It does **not** measure the natural environment's conditional noise rate or establish the unique cause of the failed transfer experiments.

The sample contains 32 original training decisions from 32 distinct games, selected before intervention outcomes. From the first 128 terminal-training games, four hash-minimum decisions are selected in each early/later × original-winning-proposal-count stratum. Sixteen decisions originally had mixed outcomes; sixteen had all wins or all losses. This deliberately balanced prefix sample is not a representative corpus sample.

Each decision keeps its three original proposals and receives four fresh native RNG initializations. The same initialization is shared across proposals within a decision/draw. All observed cards and their ordering, Jokers, resources, history and other starting state remain fixed. All 14 model input arrays are exactly unchanged; temporarily restoring the original RNG recovers the original full-state fingerprint. Each assigned action is followed by the same immutable base greedy controller. The 96 original branches are reused, while all 384 new branches finish uncensored.

| Original mixed-pair result under a new draw | Count / 128 | Share of this stress panel |
|---|---:|---:|
| Original winner still wins and loser still loses | 17 | 13.28% |
| Winner and loser reverse | 12 | 9.38% |
| Both actions lose | 49 | 38.28% |
| Both actions win | 50 | 39.06% |
| Censored | 0 | 0.00% |

These are **32 original pairs, each tested four times**, from 16 mixed-outcome games. They are not 128 independent games. Eleven distinct original pairs reverse at least once, spanning **8/16 originally mixed states**. Most original differences disappear into equal terminal outcomes; disappearance must not be counted as reversal. The four intervention draws themselves contain 28 mixed-outcome state/draw combinations out of 128. Overall, 26/32 decisions have at least one changed terminal label.

| Original stratum, four decisions each | Terminal labels changed / 48 | Actions varying across four draws / 12 | States with a pair reversal / 4 |
|---|---:|---:|---:|
| Early, zero original wins | 11 | 8 | 0 |
| Early, one original win | 23 | 12 | 1 |
| Early, two original wins | 27 | 11 | 4 |
| Early, three original wins | 32 | 10 | 0 |
| Later, zero original wins | 18 | 8 | 0 |
| Later, one original win | 16 | 7 | 2 |
| Later, two original wins | 23 | 4 | 1 |
| Later, three original wins | 3 | 3 | 0 |

Rows with zero or three original wins have no original mixed pair to reverse. The small strata describe this selected panel only; no subgroup is selected for a policy or a fitting change.

The terminal sensitivity is much larger than the immediate clear-label sensitivity in this panel: **153/384** terminal labels change, versus **58/384** immediate scores, **11/384** immediate-clear labels and **0/384** immediate-loss labels. A post-hoc descriptive cross-tabulation of the admitted traces sharpens that distinction:

- **120 of the 153 terminal changes** retain all six recorded immediate fields: score, hand type, blind clear, game over, game won and phase.
- **73 terminal changes** occur even though the first post-action policy observation matches its original counterpart exactly. All 384 comparisons have a next observation; 225 of those next observations match.
- **10 of the 12 pair reversals** preserve both actions' immediate fields. Two also preserve both actions' respective first post-action policy observations.

The latter two examples are source decision `4000015:3`, draw 3, actions 34 versus 3886, and `4000022:163`, draw 1, actions 2519 versus 6095. Their original winning and losing terminal outcomes exchange while the respective next policy inputs remain unchanged. Trace identities and hashes are saved in the [cross-tabulation records](../../eval/consequence_prediction_20260927/terminal_rng_stress_analysis_v1/pairs.jsonl). This does not assert equality of hidden state or later trajectories, or attribute every terminal change solely to an immediate score change.

The controlled finding is that an exact preference inferred from one recorded terminal realization is not necessarily invariant to hidden future randomness. It supports taking target uncertainty seriously before another terminal-ranking fit. It does not show that single-sample return learning is inherently invalid, quantify how much of the observed generalization failure is due to label variability, or exclude insufficient representations and coverage. In particular, this study did not train or evaluate an estimator of expected action value.

Fresh empty RNG channel maps are not a posterior over seeds and channel histories conditional on a naturally reached state. They leave already generated objects fixed while changing later random draws. Consequently, these finite counts are intervention results, **not calibrated win probabilities, natural noise estimates or autonomous win-rate gains**. No confidence interval treating draws as independent source games is reported. The stress outcomes remain outside training and do not replace or average the original labels. Before using repeated rollouts as a new target, specify and validate the intended continuation distribution; do not silently reinterpret this stress distribution as the environment's law.

Measurement and independent native replay take **1,919.06 and 1,900.37 seconds**. The auditor rederives selection and seed hashes, verifies all 31,811 factual-prefix decisions, all 384 new branches and 41,806 branch actions, independently recomputes policy decisions, checks starting input invariance and state isolation, and separately recounts outcomes and pair categories. The entire owner exits successfully before admission. The subsequent saved-trace analysis verifies all 480 original/new trace pins, reconciles original totals and checks the Boolean cross-tabulations without native inference or fitting.

No candidate is promoted, no confirmation is run, and no process remains live. The full held-out consequence, retention, fresh sampled/greedy win-rate, early-policy attribution and final independent-fit requirements remain unmet. World-model work remains deferred.

Artifacts: [owner](../../eval/consequence_prediction_20260927/terminal_rng_stress_owner_terminal_v1.json), [frozen protocol](../../eval/consequence_prediction_20260927/terminal_rng_stress_protocol_v1.json), [measurement](../../eval/consequence_prediction_20260927/terminal_rng_stress_v1/result_v1.json), [native audit](../../eval/consequence_prediction_20260927/terminal_rng_stress_v1/verification_v1.json), [cross-tabulation owner](../../eval/consequence_prediction_20260927/terminal_rng_stress_analysis_owner_terminal_v1.json), and [cross-tabulation result](../../eval/consequence_prediction_20260927/terminal_rng_stress_analysis_v1/result_v1.json).
