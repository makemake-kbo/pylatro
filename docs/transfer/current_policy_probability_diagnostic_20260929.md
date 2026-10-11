# Separate family probability from conditional ranking

The saved current-policy decomposition finds poor early-discard and shop-sale
agreement already in training. Its flat-mode counts cannot identify the
responsible probability factor. Inspect the unchanged model on **all 3,705
training / 514 development early-discard states** and **5,471 / 716 SHOP
Joker-sale states**: 10,406 targets in total. Early means current Ante below 4.
Do not select examples using confidence, error severity or future outcomes.

Use the same immutable base checkpoint and archived v13 runtime. Preserve the
original **MPS FP32, batch-32** inference layout: forward each original block
containing a target, including its other rows. This requires 4,409 training
blocks / 141,088 rows and 569 development blocks / 18,208 rows. Check every
forwarded row against its archived base prediction: identical global mode,
teacher NLL within the prior 1e-5 tolerance, and unchanged legal support.
These extra rows are numerical controls, not additional diagnostic targets.
Keep original cache order, temperature 1, mixture 0.5, one CPU thread, eval
mode, disabled gradients and unchanged model tensors. No training, simulator,
new teacher labels or autonomous policy is introduced.

For each target save raw macro logits, macro legality and normalized family
probabilities. Save every legal target-family action and its conditional log
probability. For discards also save the reference log probabilities and
learned residual energies whose normalized sum defines that conditional
distribution; for sales save the corrected entity logits. Preserve action
ordering and all card/Joker slot identities.

Record the teacher-family probability and macro argmax, teacher-forced
conditional probability/rank/argmax, global mode, and teacher joint NLL.
Summarize training/development separately for early discards and shop sales,
with early/later sale partitions. Report the macro/conditional correctness
cross-tabulation and how often the teacher family has the highest total mass
while the global action mode chooses another family. This is a description of
the unchanged hierarchical policy, not a proposed macro-first decoder or a
demonstration that any teacher disagreement loses value.

Run one inference scan and then an independent saved-array arithmetic audit,
serially. The audit rederives target membership and original block coverage,
checks all archived mode/NLL controls, independently normalizes raw logits in
float64, verifies conditional action support, probabilities, ranks and NLL
factorization, and reconstructs every summary. Use absolute tolerance 2e-5
for these FP32-to-float64 identities. A conditional-mode disagreement is
accepted only when its saved log probability is within 1e-6 of the maximum;
report any such numerical ties. The original full-action mode control still
requires exact action equality. No second model forward is required by this
audit. Preserve any failure; do not adjust precision, tolerances or population
after observing it.

The prior five-policy scan processed 130,225 rows in 369.7668 seconds. Scaling
that measured cost to 159,296 forwarded rows gives about 452 seconds. Freeze
the scan allowance at twice this projection plus 120 seconds, rounded up:
**1,025 seconds**, below a 1,200-second ceiling. The saved-array audit receives
300 seconds; each owner has 30 seconds cleanup grace. No timing pilot or
automatic extension. Admit only after the whole owner actually exits zero;
check a running job roughly ten minutes apart.

If teacher-family mass is high but teacher-forced conditional ranking is weak,
investigate conditional fitting/targets before changing the macro readout.
If family mass is weak despite good conditional choice, investigate family
selection. If both remain weak, neither a decoding change nor an isolated-head
story is established. Training/development differences help delimit fitting
versus transfer but do not prove a cause. Interchangeable teacher choices,
hidden teacher information and action consequences remain separate questions.
No new fit is selected by this diagnostic, and the final win-rate,
early-attribution and independent-fit requirements remain unchanged.
