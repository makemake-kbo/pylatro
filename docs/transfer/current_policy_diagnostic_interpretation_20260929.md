# Interpretation guardrails for the current phase diagnosis

Prepared while the matched phase job runs, without reading its partial
outcomes. The previous goal turn made progress by implementing, checking,
freezing and launching that diagnostic. This note reuses completed evidence;
it selects no additional fit or collection and changes no running protocol.

Three distinct gaps must remain separate when choosing the next intervention.

| Question | Existing evidence | Remaining limitation |
| --- | --- | --- |
| Does adaptation transfer on fixed teacher states? | Ordinary continuation improves training NLL but worsens held-out NLL. The terminal branch now fits all 622 training comparisons while ordering only 41/84 validation pairs correctly, versus base's 50/84. | Autonomous state drift is unnecessary to observe this failure. More student-state labels alone cannot explain away the fixed-state evidence. |
| Does the learner encounter different decision contexts? | The original epoch-7 audit found 78.65% agreement on historical teacher states and 66.54% on student states. Standardizing action-family frequencies reduced that gap to 78.65% versus 75.79%. | Those panels also differed in simulator rules and policy generation. The finite correction experiment replaced half the historical examples with a 100-game correction set and worsened sampled development wins by 1.58 points. It did not test iterative current-rule aggregation with the present corrected heads and encoder. |
| Do local spending metrics predict eventual performance? | In the completed fixed-base continuation diagnostic, 73/74 actions losing an immediate clear still completed that blind. Across 136 clear-discordant pairs, eventual blind completions were 135 versus base's 130, and terminal wins 79 versus 74, with unresolved win differences. | Immediate-clear equality is not validated as a proxy for eventual retention. The other 711 overrides and repeated candidate decisions were not evaluated at this horizon. |

Sources: [fixed-state continuation](continuation_transfer_diagnostic_20260925.md),
[terminal feature adaptation](consequence_terminal_feature_adaptation_results_20260929.md),
[original visited-state and correction trial](report.md), and
[complete fixed-base continuations](consequence_fixed_base_continuation_results_20260928.md).

The current phase comparison addresses *where teacher assistance has remaining
headroom* in the present checkpoint. It will not decide whether coverage,
feature learning, target uncertainty or autonomous distribution shift caused
that headroom. Its agreed-play arm also depends on both policies proposing a
play; a difference from whole-hand assistance cannot identify one omitted
action family by itself. Replacement counts describe changed decisions, not
their separate causal effects.

After the complete result, select the smallest matched-state investigation
that can distinguish a live explanation in the supported phase. Do not
automatically launch a generic correction-data fit, another hand-only branch,
or a repeat of the already implemented sale-head repair. Reuse compatible
saved artifacts first. If current-rule student-state aggregation is eventually
tested, distinguish it from the older finite replacement trial, preserve broad
training coverage, and declare its exposure and evaluation populations before
collection. No such experiment is selected here.

Future candidate spending rules should name the horizon their retention
measurement actually tests. Immediate score and immediate clears remain useful
prediction diagnostics; they cannot establish eventual policy non-regression.
A future rule using eventual blind completion, fixed-controller terminal
outcomes or a bounded autonomous panel must specify its estimand, cost,
uncertainty and proceed/stop criterion before observing that candidate's result.
This note does not retrospectively promote failed candidates or relax their
recorded decisions. It also does not weaken the original fresh sampled gain,
greedy guardrail, early-policy attribution or final independent-fit standard.

The data-scale example already shows why these questions must stay separate:
its 2,000-update prefix improved local clears but won only 42/128 sampled
development games, versus base's 39 and ordinary's 40. All corresponding
intervals spanned zero, and the fixed gain screen failed. Better local
retention alone did not establish the desired long-horizon gain.
[Completed duration experiment](continuation_duration_results_20260927.md).
