# Separate sale-family and conditional-choice effects

The coupled readout changes both sale-family probability and relative Joker
sale scores. A positive ablation comparison would not by itself establish
that the model learned better replacement comparisons: a common sale-score
shift could improve family calibration without changing the preferred Joker.
This descriptive diagnostic distinguishes those effects using the already
planned raw logits, after successful whole-scan admission. It adds no gate,
model forward, training, teacher call or gameplay.

For both training and development, report all SHOP states, sale states,
non-sale states, early sale states and later sale states. Early here means
current Ante<4, solely as a descriptive group. Reproduce each group's enabled
minus disabled NLL change from the admitted scan. On active rows reconstruct
macro NLL before and after the saved sale-family adjustment; on teacher-sale
rows additionally reconstruct conditional Joker NLL before and after the
saved residual. Use the actual FP32 additions for corrected logits, then
float64 probability arithmetic. The two components must sum to the recorded
full-action NLL change within the existing 2e-5 tolerance. Inactive rows have
zero change and remain in the overall denominator.

On active states, report macro winner matches gained/lost and both/neither
matches. On active teacher-sale states report the same for the conditional
Joker winner, plus the number with multiple legal sales. Report joint flat-mode
matches gained/lost on every state in each group. Do not equate a flat-mode
family mismatch with a macro-head error. Outside the gate the macro and
conditional absolute logits were not saved; report their zero changes without
inventing absolute accuracies there.

Use paired source-game resampling for development NLL changes and their two
components: the same 128 source games, 10,000 draws and seed1729, retaining
state-count denominators. These are marginal descriptive intervals and no
additional qualification rules. Publish every predeclared group. A conditional
improvement would support useful within-model Joker-choice changes, not prove
that explicit pairing rather than capacity caused the learned behavior.
Neither component establishes action value or a win-rate benefit.

Verify both record identities, aggregate controls, the component identity and
independent categorical recounts. Pin all source and result files, require an
actual successful whole process, and allow 120 seconds without automatic retry
or extension. Run only after the primary scan and its audit are admitted.
Its original thirteen checks and the full fresh-game/early-attribution/final
independent-fit standard remain unchanged.
