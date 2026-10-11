# Early-shop representation probe, 2026-09-22

**The tested pooling change did not improve held-out prediction.** This result
does not establish a representation bottleneck or justify changing the policy.
The broader hypothesis—that the model needs a better representation of delayed
consequences and build interactions—remains open.

The [verified result](data/representation_pooling_probe_20260922.json) compares
two summaries of the exact same frozen transformer token states. The control
uses the existing mean over all valid tokens. The alternative first averages
within each token type, then gives each present type equal weight. This stops
numerous deck-card tokens from receiving more total weight than smaller groups
such as Jokers and history. Both predictors also receive the same 33 scalars.
Both have 418 fitted coefficients, including the intercept. A separate
scalar-only baseline has 34 coefficients.

This is a test of how existing information is summarized. It does not train
new representations, add memory, or test action-conditioned predictions.
Neither the encoder nor any gameplay policy was updated.

## Data and information boundaries

The prepared [plan](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/entity_pooling_probe/manifest.json)
pins 1,280 previously collected MC training games, ten cohorts from each of two
replicas. Each encoder is the frozen starting checkpoint at update 122, before
these games were collected. The actual outcomes follow the recorded sequence
of ten behavior policies per replica. Predictions therefore concern that
behavior mixture; they are not estimates of the initial policy's value.

Each snapshot is the first shop observation in current Ante 1, 2, or 3, before
the game has ever reached Ante 4. Selection uses only the observed prefix.
The targets are eventual victory, clearing Ante 4, and clearing Ante 5. These
binary labels remain valid after legal Ante decreases.

Each 64-game cohort was split into 48 training, eight validation, and eight
test games using an outcome-independent permutation. All snapshots from a
game stay in its assigned split. Four games lacked an eligible shop snapshot
and were excluded identically from all arms:

| Encoder replica | Training games | Validation games | Test games |
| --- | ---: | ---: | ---: |
| 0 | 478 | 80 | 80 |
| 1 | 480 | 79 | 79 |

The cache contains 3,645 states, including 457 test states from 159 games.
Every game receives equal total fitting weight regardless of snapshot count.
Feature standardization uses training rows only. L2 regularization is selected
from the fixed grid {0.001, 0.01, 0.1, 1} using validation loss. All 18 selected
readouts are saved before test evaluation. No gameplay development,
confirmation, or reserved early-transfer seeds are used.

## Results

Loss is binary negative log likelihood, averaged within game and then equally
across the two fixed encoder replicas. Lower is better. These values should
not be compared directly with earlier nine-class critic losses.

| Target | Existing mean | Equal weight per token type | Difference, 95% interval |
| --- | ---: | ---: | ---: |
| Win, primary | 0.65989 | 0.66265 | +0.00276 [−0.00179, +0.00743] |
| Clear Ante 4 | 0.59049 | 0.59416 | +0.00367 [−0.00441, +0.01159] |
| Clear Ante 5 | 0.65620 | 0.65996 | +0.00377 [−0.00161, +0.00924] |

The primary difference is slightly worse in both replicas: +0.00233 and
+0.00319. The prespecified probe gate—an upper confidence bound below zero and
an improvement in each replica—fails. Secondary intervals are pointwise.
This specific change has no demonstrated advantage; other readouts and
learned representations were not tested.

Scalar-only win loss is 0.68718, versus 0.65989 with the existing neural
features. That descriptive comparison is consistent with useful information
in the existing representation. It does not demonstrate that the information
supports choosing actions with better delayed consequences.

The next architectural question is whether current features distinguish
alternative actions by their delayed effects on a build. That requires
comparisons of alternative actions from the same state with a fixed
continuation policy; factual win prediction alone cannot answer it. Any later
architecture training experiment must preserve a matched control and establish
fresh full-game gains plus an early-policy contribution to those gains.

## Verification

The [independent audit](data/representation_pooling_probe_verification_20260922.json)
reconstructs snapshot selection and labels from the original trajectories,
checks cached scalars against the original observations, verifies disjoint
game splits and training-only normalization, and recomputes selected-fit
gradients, validation choices, test losses, and all paired bootstrap intervals.
The [per-game losses](data/representation_pooling_probe_games_20260922.jsonl)
support independent aggregation. Bootstrap draws resample whole games within
each replica, with the replica means weighted equally.

Seven tests cover split boundaries, prefix-only snapshot selection, information
lost by global pooling in a synthetic case, padding, game weights,
standardization, and fitted-gradient behavior. All pass. Production critic
forecasts reconstructed from the cached global mean match the original forward
passes exactly. Both source policy digests and all input hashes remain intact.
The probe completed in 78.2 CPU wall seconds with two threads and no accelerator
use. It evaluated no new games and supplies no win-rate gain.
