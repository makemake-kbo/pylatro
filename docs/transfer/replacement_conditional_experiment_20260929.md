# Separate the replacement scorer from the sale-family correction

## Evidence and hypothesis

The rejected coupled endpoint barely changes conditional Joker preferences on
training data. Its added score contrast RMS is about 0.000356 versus 2.519 in
the existing sale logits. The six tensors update, and Adam epsilon is not a
broad explanation. The admitted fixed-weight probe on the first 9,600 original
training presentations finds a macro gradient 6.28 times the conditional norm,
with cosine -0.185. Both the negative full gradient and the nominal endpoint
Adam direction locally increase conditional NLL on the aggregate probe while
reducing total NLL. These are endpoint/sample observations, not a causal
account of the complete trajectory or the full pretraining gap.

The next single controlled intervention removes the added scorer's macro-family
correction. Its purpose is to test whether the same independently trainable
relation features can learn useful Joker comparisons when sale-family loss no
longer trains those parameters. This changes both the readout's forward policy
factorization and its gradient route. Do not describe it as a gradient-only
intervention with identical forward behavior.

## Only intended mechanism change

Reuse the same 57,473-parameter readout, dimensions, zero-output initializer,
contextual owned/offered Joker features, purchase-weighted proposal and visible
gate. At an active state, add r_i to the legal conditional sale logits s_i.
Keep the original macro logits unchanged; remove the logsumexp correction to
the sale-family logit. Therefore:

* The sale-family probability equals the disabled-readout probability exactly
  within the same model. Conditional sale probabilities use softmax(s+r).
* Non-sale joint probabilities and NLL are unchanged by enabling the readout
  within the same model. The flat-mode action may still change if a sale's
  conditional probability overtakes a non-sale action.
* The derivative into an active sale residual is
  1[teacher is sale] * (P'(i|sale) - 1[i=teacher]). Non-sale examples supply no
  direct gradient into the added readout. Existing parameters continue to learn
  the original full-action objective, including whether to sell.
* A constant shift to all legal residuals cancels from conditional probability.
  The output bias is therefore unidentifiable in this policy. Do not require
  that bias to move or interpret its numerical gradient as learned behavior.

The underlying encoder can change through training, so these within-model
identities do not promise retention against the original base or prior fits.
The original teacher-state and eventual fresh-game checks still apply.

## Single matched fit

Start fresh from the same base checkpoint and identical seeded readout
initializer. Keep the previous corpus, original shuffled row stream, 320-row
logical/32-row physical batches, MPS FP32 evaluation mode, temperature1,
mixture0.5, AdamW3e-5, warmup100, no weight decay/clipping and fixed1,000-update
endpoint. Keep full action NLL plus the same future-purchase auxiliary loss,
both normalized by the original320 rows. Keep all previously trainable
parameters trainable and critic parameters frozen. No second exploratory fit,
learning-rate change, added labels, longer duration or capacity change.

Preserve the coupled fit and all failed results. Give this variant explicit
checkpoint metadata and a strict loader that cannot silently apply the coupled
policy interpretation. Do not edit any executed or frozen implementation.

Before fitting, verify zero-output parity with base native controls, legal
support, off-gate identity, unchanged macro probabilities and non-sale NLL
under a nonzero conditional correction, conditional-gradient arithmetic and
constant-shift invariance. These checks perform zero optimizer steps. Match
the original first-update row identities and independently reconstruct the new
conditional output-layer gradient. Preserve pause-aware timing and recovery.
Fix exact resource budgets in the execution protocol before launch, using the
admitted technical-check and coupled-fit costs. No automatic retries/extensions.

## Evaluation and spending rule

Use the same exposed training/development rows and admitted base, ordinary1k
and prior auxiliary records, adding the rejected coupled endpoint as a fourth
reference. Measure this model and its own disabled-readout ablation. Audit
native inference and saved arithmetic before interpreting results. Report every
existing group and paired source-game comparison, including direct contrasts
against the coupled endpoint. Reconstruct all unchanged controls.

Retain the original thirteen point checks: future-multiple NLL lower than base
and ordinary1k; sale NLL lower and sale exact count higher than each; purchase
exact count at least each; non-sale SHOP NLL no worse than ordinary1k; sale NLL
lower and exact count higher than prior auxiliary and the within-model ablation.
No threshold, epoch, coefficient or panel sweep. Training/development component
and gradient diagnostics remain descriptive, not substitutes for the screen.

Only an admitted passing endpoint can proceed to a separately specified fresh
SHOP-policy evaluation. Final success still requires the original fresh sampled
win improvement against base and the original ordinary control, greedy
retention, positive learned early-decision attribution and one final independent
fit/untouched confirmation. The shorter ordinary control remains additional.
The excluded historical confirmation outcomes remain untouched. No win-rate or
long-horizon benefit is claimed by this proposal.
