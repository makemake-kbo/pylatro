# Held-out comparison of the conditional-only replacement scorer

Run only after actual successful whole-owner completion and admission of the
single conditional fit and its audit. Preserve the original experiment's
thirteen point checks and all final fresh-game/early-attribution requirements.
This exposed teacher-state comparison does not measure autonomous wins.

Reuse admitted base, ordinary1k, prior auxiliary and coupled endpoint records:
912,744 policy rows in total. Measure the new conditional model and its own
disabled-readout ablation on the same 202,141 training and 26,045 development
rows, in original consecutive FP32 MPS batches of32, including final partial
batches. A shared encoder pass produces both policies:228,186 encoder rows and
456,372 new records. Temperature1, mixture0.5 and original legal support remain
fixed. Capture proposal probabilities, pair scores, residuals, original macro
and sale logits, masks, teacher likelihoods and joint modes at active states.

The candidate must preserve macro logits and macro probabilities exactly
relative to its within-model ablation. Future-purchase logits/predictions and
non-sale teacher NLL also remain exactly equal. Joint flat modes may change on
active non-sale states; do not confuse unchanged probability of that teacher
action with an unchanged global argmax. Outside the visible gate, both modes
and NLL remain identical.

The saved-row audit independently reconstructs visible eligibility, proposal
weighting and residuals. Applied macro delta must be exactly zero. For a teacher
sale, candidate NLL equals disabled NLL minus its residual plus
logsumexp(s+r)-logsumexp(s), over legal sale slots. For a non-sale, it equals
disabled NLL exactly. Use existing2e-5 arithmetic/logit tolerances and1e-5 native
NLL tolerance; exact identities and modes where specified. This differs from
the prior coupled joint-energy normalization and must not reuse that formula.

Independently build native inputs for every development row and the first320
training rows. Run the conditional policy and a separate native no-hook
ablation encoder call:52,730 encoder/policy rows. Verify all recorded NLL/modes,
future distributions and active pair/proposal/residual values. Reconstruct all
four old development controls from104,180 raw records and compare their prior
admitted summaries exactly. Prior admitted training summaries may be reused.
Check model/source/checkpoint hashes before and after evaluation.

Publish every original train/development group for all six policies. Apply the
unchanged thirteen checks versus base, ordinary1k, prior auxiliary and the new
within-model ablation. The prior coupled endpoint is an additional reporting
control. For future-multiple, sale, purchase and non-sale SHOP groups, publish
all20 candidate-minus-control paired comparisons, retaining all128 development
source games and decision-weighted denominators,10,000 bootstrap draws,seed1729.
Intervals are marginal descriptive evidence, not replacement qualification.

Use admitted coupled scan durations to forecast equivalent new measurement and
native audit work. Freeze maximum active times at ceil(2*prior_measure+120) and
ceil(2*prior_audit+300) seconds, plus30 seconds owner cleanup per phase. The
additional reused control needs no model inference. Serialize with training,
use the tested pause-aware lifecycle, and allow no automatic retry/extension.
Admit only after actual successful whole-owner exit and both phase receipts.

If any original point check fails, this endpoint does not qualify for gameplay
or a final independent confirmation fit. Do not sweep epochs, coefficients,
eligibility, widths, learning rates or evaluation panels to rescue it.
