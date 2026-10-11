# Discard fitting and Joker-sale transfer fail in different probability factors

The unchanged current model often recognizes that an early decision should
be a discard, but ranks the teacher's discarded cards first only **43.48% of
the time in training and 40.27% in development**, even when the discard family
is supplied. Joker sales have a different pattern: conditional Joker choice
matches **76.09% in training versus 53.35% in development**, while the sale
family itself has the highest probability in only **55.71% and 54.47%**.
These measurements separate incomplete fitting from an observed transfer
gap. Neither identifies which disagreements cause lost games.

The [frozen diagnostic](current_policy_probability_diagnostic_20260929.md)
covers every early discard and every SHOP Joker-sale target in the existing
current-rule training/development caches: **10,406 target states**. Early
means current Ante below 4. Development is already exposed; these are
teacher-visited states, not a new autonomous-game panel. The model, original
MPS FP32 batch layout, legal actions and observation fields are unchanged.

| Group | States | Teacher family has highest mass | Teacher action best within its family | Original joint action matches |
| --- | ---: | ---: | ---: | ---: |
| Training early discards | 3,705 | 3,102 / 83.72% | 1,611 / **43.48%** | 1,355 / 36.57% |
| Development early discards | 514 | 394 / 76.65% | 207 / **40.27%** | 168 / 32.68% |
| Training SHOP Joker sales | 5,471 | 3,048 / 55.71% | 4,163 / **76.09%** | 1,963 / 35.88% |
| Development SHOP Joker sales | 716 | 390 / 54.47% | 382 / **53.35%** | 167 / 23.32% |

The conditional column forces only the teacher's action family for diagnostic
purposes, then compares every legal action in that family. It does not select
rows where the model already chose that family. The final column reproduces
the original flat joint mode exactly.

For early discards, **1,661 training and 218 development states** have the
right family argmax but the wrong conditional argmax. There are only 170 and
31 with the reverse pattern. The teacher discard ranks below the top ten in
**807/3,705 training and 157/514 development states**. Weak within-family
selection is already present on represented training examples; autonomous
distribution shift cannot be its sole explanation.

The sale conditional drop appears in both horizon partitions: early sales
fall from **942/1,242 (75.85%) to 87/162 (53.70%)**, and later sales from
**3,221/4,229 (76.16%) to 295/554 (53.25%)**. Sale-family selection is much
weaker later: its argmax agreement is 50.86% training / 50.36% development,
compared with 72.22% / 68.52% early. Thus the total sale error is not well
described as a single failing head. These are descriptive state-weighted
comparisons, not independent-trial significance tests or a controlled
decomposition of environment difficulty.

Teacher-action negative log probability separates row by row as
`joint NLL = family NLL + conditional NLL`:

| Group | Mean family NLL | Mean conditional NLL | Mean joint NLL |
| --- | ---: | ---: | ---: |
| Training early discards | 0.3961 | 2.4502 | 2.8463 |
| Development early discards | 0.5756 | 2.9115 | 3.4871 |
| Training SHOP sales | 0.9528 | 0.6577 | 1.6105 |
| Development SHOP sales | 1.0933 | 1.2185 | 2.3118 |

Average teacher-family probability is 0.7822 / 0.7297 for early discards and
0.5434 / 0.5311 for sales, training / development respectively. Conditional
teacher-action probability is 0.3026 / 0.2735 for discards and 0.6210 / 0.4571
for sales. Joint teacher-action probability remains only 0.2609 / 0.2310 and
0.3180 / 0.2290. Probabilities multiply per state; products of population means
are not the mean joint probability. None of these probabilities estimates
whether an action wins the game.

The scan also quantifies why flat-mode family mismatches cannot be assigned
directly to the macro head. Among development states, **54 early discards and
57 sales** give the teacher family the highest total probability but choose
another family in the flat mode. A family's mass can be spread across several
actions while another family contains the single most probable action.
Both factor argmaxes match the teacher but the flat action differs in just
**9/514 discards and 28/716 sales**; changing greedy decoding alone would not
address the much larger conditional and family disagreements. No alternative
decoder is installed or evaluated here. Sampled policy behavior already uses
family sampling followed by conditional sampling, so a greedy-decoder change
would not by itself change that sampling distribution.

This result does not justify simply training longer. The
[existing continuation study](continuation_transfer_diagnostic_20260925.md)
already measured better training fit without corresponding development NLL
improvement. Across all discard states, ordinary continuation raises training
agreement from 57.94% to 66.09% / 65.09%, while development agreement is
52.20% / 51.77% versus base 52.76%. Across Joker sales in **all phases**, its
training agreement rises from 37.69% to 55.54% / 53.97%, but development NLL
worsens from 2.2287 to 2.6503 / 2.7605. Those historical groups are broader than
the early-discard and SHOP-only groups above; their conditional probabilities
were not measured by this scan. The two historical replicas are reused
evidence, not additional fits commissioned for this diagnostic.

Before another fitting or architectural change, the next question is whether
the poorly matched labels encode useful distinctions the policy fails to
learn, equivalent alternatives, or choices relying on teacher-only
information. Inspect the relevant teacher decision logic and existing traces
first, with attention to discard equivalence and the order of shop purchases
and sales. These are possible explanations, not findings from this scan.
The conditional discard fitting gap and sale conditional transfer gap call
for separate predictions; a single macro-head intervention is not established
as the remedy. No new fit is selected by this result, and world models remain
deferred.

All **159,296 forwarded control rows** reproduce their archived full-action
mode and teacher NLL exactly; legal support is unchanged. The independent
saved-array audit checks all target identities, observation hashes, legal
conditional supports, raw-logit normalization, probability factorization and
summary counts. Maximum factorization error is 1.03e-6, maximum conditional
log-probability reconstruction error 6.36e-6, both below the fixed 2e-5
arithmetic tolerance. No conditional argmax numerical ties were encountered.
The model tensors remain unchanged. Measurement took **400.3 seconds** and
the audit **9.1 seconds**, within the frozen allowances. The whole owner
actually exited zero before admission; it was checked once after about ten
minutes. No process from this diagnostic remains live.

The original fresh-game gain, greedy retention, early-policy attribution and
final independent-fit requirements remain unmet. This is a completed
diagnosis of the current policy's probabilities, not a learned win-rate gain.

Artifacts: [verified metrics and summaries](../../eval/consequence_prediction_20260927/current_probability_v1/verification_v1.json),
[measurement provenance](../../eval/consequence_prediction_20260927/current_probability_v1/result_v1.json),
[whole-owner exit receipt](../../eval/consequence_prediction_20260927/current_probability_owner_terminal_v1.json),
[frozen protocol](../../eval/consequence_prediction_20260927/current_probability_protocol_v1.json),
[inference worker](../../eval/consequence_prediction_20260927/current_probability_v1.py),
and [independent arithmetic auditor](../../eval/consequence_prediction_20260927/verify_current_probability_v1.py).
