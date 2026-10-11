# Fixed comparison for the coupled replacement readout

Run only after the entire single fit and endpoint audit have actually exited
successfully. The [experiment's 13 point checks](replacement_relation_experiment_20260929.md)
and the fixed 1,000-update endpoint are unchanged. No game evaluation or
confirmation is permitted by a partial fit, a technical pass or better
training likelihood alone.

Reuse all admitted base, ordinary1k and auxiliary predictions from the prior
scan. Record two new policies on every original training/development row:
the coupled model and that same model with its relation correction disabled.
The measurement uses one native encoder call per row and constructs both
distributions from the corrected and uncorrected grammar output. Use original
MPS FP32 consecutive batches of 32, including the original final partial
batch, temperature one and hand mixture .5. Future labels never enter the
model, readout or observation-derived gate.

This means 228,186 new encoder rows, 456,372 new policy records and 684,558
reused control records. Save identities, legal selected modes, full-action
NLL and the same eligible future-purchase logits/targets/support. Future
predictions must be identical between enabled and disabled readouts. Save
the relation gate on all new candidate rows; on active rows additionally
save the original macro/sale/purchase logits, masks, pair scores, proposal
weights, applied sale residual and macro adjustment. These allow a float64
reconstruction of the relation's normalization and teacher-action NLL change.

The auditor checks every new record's identity, legality and future arithmetic.
It reconstructs gates directly from observations, pair-score weighting and the
joint-action tilt independently from raw values. Outside the gate, enabled
and disabled modes/NLL must match exactly. It reproduces all development
rows plus the first 320 training rows using directly constructed observation
tensors. For the disabled policy, it makes a separate native call without
the coupling hook: 52,730 audit encoder rows/policy predictions total. This
is independent input construction and a direct uncoupled call, not a second
implementation of the underlying decoder. All model tensors must remain
unchanged. Fixed tolerances: native NLL 1e-5; logits, pair arithmetic and
joint-action NLL adjustment 2e-5; selected modes exact.

Old control training summaries come from the admitted prior audit; all pinned
raw files retain their hashes. Reconstruct control development summaries
from their 78,135 existing rows, check identities against the current cache,
and reproduce all prior development aggregates. Do not rerun the controls.

The 13 checks comprise the original nine against base/ordinary1k, plus lower
sale NLL and more exact sales than the prior auxiliary policy, plus lower
sale NLL and more exact sales than the disabled-readout policy. Report all
checks and descriptive training/development cohorts. Use 10,000 paired
resamples of the same 128 development games, seed1729, preserving decision
denominators. Report future-choice, sale, purchase and non-sale comparisons
against all four controls, including failures and zero-effect ablation cells.
The within-model ablation does not independently isolate training capacity.

Prior admitted timing: one ordinary measurement plus control validation
622.734501s; three-policy native audit 235.743589s. Conservatively forecast
twice the former for the two new distributions (1,245.469002s), and two-thirds
of the latter for the two-policy native audit (157.162393s). The fixed limits
are **2,611 active seconds for measurement** and **615 for audit**, using
ceil(2×forecast+120) and ceil(2×forecast+300), plus 30 seconds cleanup each.
No new timing pilot, automatic retry, budget extension or threshold change.
Use the tested pause-aware owner, serialize native jobs and check long jobs
about ten minutes apart. Only a whole-owner successful exit admits evidence.
