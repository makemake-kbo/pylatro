# Early discards and shop sales are weak even on represented training states

The current model's high aggregate imitation accuracy hides poorly matched
decision groups. It matches **36.57% of early discard labels in training** and
**32.68% in development**. For Joker sales during SHOP, agreement is **35.88%
and 23.32%**. Purchases are much stronger at **94.68% and 93.74%**. These
patterns motivate a focused probability diagnostic before changing training
or architecture. They do not establish that every non-teacher action is bad.

The [saved-data analysis](current_policy_family_decomposition_20260929.md)
uses the final trainable-encoder base's existing 202,141 training predictions
from 1,000 games and 26,045 development predictions from 128 games. All rows
are current-rule teacher states; development is already exposed. No model,
simulator or new action-label call is made.

| Teacher decision group | Training exact / rows | Training agreement | Development exact / rows | Development agreement |
| --- | ---: | ---: | ---: | ---: |
| All decisions | 173,497 / 202,141 | 85.83% | 21,544 / 26,045 | 82.72% |
| Plays | 35,430 / 47,149 | 75.14% | 4,156 / 6,113 | 67.99% |
| Discards | 6,498 / 11,215 | 57.94% | 745 / 1,412 | 52.76% |
| Early discards, current Ante below 4 | 1,355 / 3,705 | **36.57%** | 168 / 514 | **32.68%** |
| Shop purchases | 39,509 / 41,729 | 94.68% | 4,990 / 5,323 | 93.74% |
| Joker sales in SHOP | 1,963 / 5,471 | **35.88%** | 167 / 716 | **23.32%** |

Sale rows here are restricted to SHOP; the previously reported 37.69% / 25.21%
sale agreement includes sales during other phases. Early means current Ante
below 4, retaining any legal Ante-zero states.

The selected flat action's mismatches have different structure across phases:

| Panel / phase | Decisions | Different selected family | Different action within teacher family |
| --- | ---: | ---: | ---: |
| Training hand phase | 60,521 | 2,849 | 14,216 |
| Development hand phase | 7,828 | 431 | 2,302 |
| Training shop phase | 90,355 | 5,622 | 1,949 |
| Development shop phase | 11,596 | 817 | 362 |

Within development shop-sale states, **383/716** choices use a different
family and **166/716** sell a different Joker. The different-family choices
are purchases in 199 cases, leaving in 104 and rerolling in 80. Training has
2,837 different-family choices and 671 different-Joker choices among 5,471
shop-sale states. This is a fitting gap as well as a development discrepancy;
it is not solely an autonomous visited-state phenomenon.

Of 514 early development discard labels, 168 match exactly, 173 choose a
different discarded subset and 173 choose another family. The corresponding
training counts are 1,355, 1,312 and 1,038 out of 3,705. Across all development
discards, 284 of 291 different-family choices are plays. Later discard
agreement is much higher: **68.48% training and 64.25% development**.

Teacher-action probability is also low in the early discard group: **0.2609
training and 0.2310 development** on average, with NLL **2.8463 and 3.4871**.
Shop-sale probability averages **0.3180 and 0.2290**, NLL **1.6105 and 2.3118**.
Purchase probabilities are **0.9028 and 0.8956**. These are observed-state
averages, not calibrated probabilities of making a beneficial decision.

Crucially, a different selected family is **not a measurement of macro-head
error**. The policy's greedy action maximizes joint macro and conditional
probability. A diffuse conditional distribution can make another family's
best action win even if the teacher family has the largest total probability.
Likewise, agreement on rows where the selected family matches is not
teacher-forced conditional accuracy. Saved predictions lack the separate
macro and conditional distributions, so this analysis cannot attribute the
failure to a head or support a decoding change.

The next cheapest discriminating measurement is those separate probabilities
on the already identified early-discard and shop-sale states: teacher-family
mass, macro argmax, teacher-forced conditional rank/probability, and how they
compose into the joint mode. Compare training and development with identical
definitions. This would distinguish low family probability from diffuse or
incorrect within-family selection without another fit. Poor exact fit can
also reflect interchangeable actions or teacher information unavailable to
the policy; consequences remain necessary before claiming strategic error.
No fitting recipe or architecture is selected by these counts.

Every saved row matches cached seed, step and teacher action. Frozen action
decoding matches every recorded teacher family. An independent array/confusion
calculation verifies **213 phase/family/horizon groups**, and **29 published
base aggregate/family groups** reproduce their counts, accuracy, NLL and
teacher probability. All source/input hashes remain fixed. Analysis completes
in about **two seconds**, with actual exit zero. This is saved-data arithmetic
verification, not a new native replay or neural-inference audit.

The original win-rate, greedy-retention, early-policy attribution and final
independent-fit requirements remain unmet. This result refines the
[current phase diagnosis](current_policy_phase_results_20260929.md); it does
not claim a trained-policy improvement. World models remain deferred.

Artifacts: [decomposition and full confusion counts](../../eval/consequence_prediction_20260927/current_family_decomposition_v1.json),
[actual-exit receipt](../../eval/consequence_prediction_20260927/current_family_decomposition_owner_terminal_v1.json),
and [reproducible analysis](../../eval/consequence_prediction_20260927/current_family_decomposition_v1.py).
