# One relative-mean readout using all terminal-labelled states

**Question.** Does excluding equal-outcome states from the conditional objective
contribute to exaggerated, poorly transferring action differences? The
[density fit](consequence_terminal_training_density_results_20260929.md) uses
only 311 of 1,600 training states. The other 1,289 have zero conditional loss and
curvature. Its validation ordering is 47.62%, versus 59.52% for base. The
competing explanation is that available features/examples do not predict useful
relative terminal values. The [RNG stress](consequence_terminal_rng_stress_results_20260929.md)
and [historical sampled-repeat review](consequence_sampled_continuation_reuse_20260929.md)
motivate estimating averages; neither supplies admissible new training labels.

**Single intervention.** Replace conditional classification with squared-error
regression of relative terminal outcomes. Reuse the exact 1,600 training states
from 800 games and unchanged 200-state validation panel, original RNG and greedy
continuation labels, 384 frozen RMS-normalized features, zero intercept and
ridge coefficient 0.001. Do not change any existing model or label. Fit once.

For three proposals at state `s`, let `xc = x - mean_actions(x)` and
`yc = y - mean_actions(y)`. Minimize

```
L(w) = sum((xc @ w - yc)**2) / (2 * 3 * N) + 0.001 * sum(w**2) / 2
```

Use all states, including equal outcomes. An individual zero difference is a
realized sample, not proof of equal expected values. Squared loss targets mean
relative outcomes under the data/proposal distribution without requiring
independent action outcomes. This changes the objective and its curvature; it
does not isolate exclusion alone or prove calibrated observation-conditioned Q.
At zero weights its gradient matches the previous conditional objective's
gradient with divisor `3*N`; equal-outcome states add curvature, not an initial
gradient. The fixed coefficient is not a regularization sweep.

Solve the positive-definite normal equation once, CPU float64, one thread.
Inference is `x @ w`, with no proposal centering, teacher identity, outcome or
seed input. Scores are relative utilities, not absolute win probabilities.
Do not clip scores or select a coefficient/threshold. An independent audit checks
the equivalent sum of pair-difference losses, normal equation, positive curvature,
all predictions, split, unchanged controls, summaries and intervals. It performs
no second optimization. Expected compute is seconds, with a 600-second cap per
fit/audit phase; no simulator or encoder work is needed.

**Prediction and decision.** Equal-outcome examples should constrain unsupported
action differences. Report centered outcome MSE against zero relative prediction
and centered sigmoid probabilities from the prior RMS readout. Report terminal
pair ordering, same-clear ordering, and restricted-choice realized wins against
base, RMS and the density conditional readout. Include all/early/later and
mixed/equal-outcome error accounting; no subgroup selects a candidate. Use the
same 10,000 whole-game bootstrap resamples, seed 1729. The zero predictor is an
error baseline only and does not create a new action policy.

Consider separately specified held-out source-game collection only if all nine
spending checks pass: training centered MSE beats zero; validation centered MSE
beats both zero and RMS with each paired 95% upper bound below zero; validation
pair ordering strictly beats base and retains RMS; early pair ordering retains
base; same-clear ordering exceeds 50%; and restricted choices retain base wins
overall and early. Improved MSE without useful ordering does not pass. These
are new prospective spending checks, not revisions of rejected experiments.
Failure ends this objective test without a coefficient or density sweep.

All 1,000 source games were used in feature training and this validation panel
has been reused. Passing cannot establish unseen-game generalization or policy
readiness. Proposal sets include teacher actions; restricted choices are only
diagnostics. Full held-out prediction/retention, every selected alternative
assessed, fresh sampled and greedy games, early-policy attribution, and a final
independent fit remain required. World models remain deferred.
