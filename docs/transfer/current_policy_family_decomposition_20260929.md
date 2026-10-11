# Decompose current-policy mismatches using saved predictions

The completed phase intervention establishes headroom in shops and in hand
decisions outside the agreed-play condition. Before another fit, reuse the
current base's saved predictions on all 202,141 training decisions from 1,000
games and all 26,045 exposed development decisions from 128 games. Both panels
use current-rule teacher trajectories. No model, simulation, new action label
or candidate is introduced.

Bind both prediction files and their observation-cache manifests through the
existing consequence dataset protocol. Verify row identity against cached
seed, step and teacher action; read phase and current ante from cached scalars.
Use the frozen action decoder to recover the selected action's family. The
existing frozen-encoder 0/1,000 training-only decomposition provides a reusable
decoder, but does not answer this question about the final trainable-encoder
base and development data.

Partition every row into exact agreement, a different action within the
teacher's family, or a different selected family. Report counts, fractions,
teacher NLL and teacher-action probability overall, by phase, by teacher
family, and by phase/family; retain early (Ante 1–3) and later groups. Report
full teacher-to-selected-family confusion counts, not just the largest cells.
As in prior analyses, the early group is current Ante below 4; any legal Ante
zero state is retained in that group.
Compare training and development descriptively; these are different games and
state mixtures, not a paired intervention or a new estimate of policy regret.

The selected action is the **joint flat mode**. A different selected family
does not establish an incorrect macro-head argmax: macro probability and the
best conditional action probability both affect the global mode. Same-family
selected-row agreement is not teacher-forced conditional-head accuracy. Saved
rows lack those separate distributions. This analysis can locate mismatch
patterns and delimit a subsequent probability diagnostic; it cannot identify
the responsible head or justify an architecture by itself.

Reproduce the published base training/development counts, accuracy, NLL and
teacher probability for every action family. Check the category counts with
an independently accumulated NumPy confusion matrix, including every phase
and horizon partition. Source and input hashes remain fixed. Bound this saved
data analysis to 180 seconds; do not load model checkpoints or run inference.
It needs no new simulation audit or second training fit. Admit only after its
actual successful exit and preserve failures.

Use the result to choose the next cheapest measurement: if different-family
choices dominate a relevant family, distinguish macro mass from conditional
diffusion before assigning a head failure; if same-family mismatches dominate,
inspect conditional ranking and its training/development gap. Poor training
fit does not establish an architectural impossibility, and low exact teacher
agreement does not prove bad consequences. No new fit or collection is selected
by this protocol; the original win-rate, retention, early-attribution and final
independent-fit requirements remain unchanged. World models remain deferred.
