# Locate the remaining sale errors in saved predictions

The completed, audited auxiliary experiment improves held-out prospective
purchase NLL and agreement, but fails the fixed sale-transfer screen. It is
not eligible for game evaluation or confirmation. This follow-up describes
where that failure occurs; it does not change the screen or select an endpoint.

Use only the admitted scan's existing base, ordinary1k and auxiliary predictions
and the original training/development cache identities and future labels.
Load no model and run no training, simulator, teacher or new inference.

For each policy and split, partition all SHOP-sale teacher states into exact
sale matches, a different Joker sale, and another selected action family.
Repeat for replacement-labelled states, multiple-offer replacements,
singleton replacements, and sales without a replacement label. In eligible
groups, also condition these counts on whether the prospective purchase mode
matches its target. These are descriptive, teacher-conditioned state counts.

For each control, classify eligible states by future-choice transition:
both correct, auxiliary corrects the control, auxiliary loses a control match,
or both wrong. Report sale exact matches gained/lost, both/neither matches,
and paired sale NLL differences within each group. Report all groups, including
small and empty cells, rather than selecting a favorable subset. The primary
interpretation uses multiple-offer states; singleton accuracy is automatic.

Verify row indices, seed/step/action identity, phase/ante and labels against
the original arrays. Reproduce the admitted sale and future-purchase totals
and NLL summaries for all three policies and both splits. Independently
recount the categorical tables with array masks and verify the exhaustive
partitions. Pin inputs and outputs, and admit only a successful whole process.
Allow 120 seconds, with no automatic retry or extension.

Future-choice correctness is a model-dependent descriptive condition, not a
randomized intervention or causal mediation estimate. A flat-mode family
mismatch is not necessarily a macro-head argmax error. Neither agreement nor
NLL measures action value. The analysis cannot establish architectural
inexpressivity, the teacher's optimality, or fresh-game generalization.

If sale errors remain common even with the correct prospective choice, use
that evidence to motivate examining the connection between offer selection,
sale-family selection and the owned-Joker choice. Do not automatically tune
the auxiliary coefficient, train longer, widen again or launch a new fit.
