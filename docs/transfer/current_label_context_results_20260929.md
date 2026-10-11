# Most weak sale labels are the first step of a Joker replacement

**94.74% of training SHOP sales and 94.13% of development sales immediately
enable a Joker purchase that was illegal before the sale.** On these same
recorded sequences, the current model matches the sale only **36.89% / 24.48%**
of the time, but matches the subsequent purchase **93.59% / 92.58%** once the
teacher has performed the sale. This localizes a concrete enabling decision
worth testing: choosing a sale that makes a useful replacement possible.
It does not yet establish that correcting these sales improves game wins.

The [saved-data census](current_label_context_diagnostic_20260929.md) reuses
all 10,406 targets from the
[probability-factor diagnostic](current_policy_probability_results_20260929.md).
Every following decision must have the same game seed and consecutive step.
For the main replacement count, the same encoded Joker offer remains in the
same shop slot, its buy action changes from illegal to legal, and original
Joker capacity is exhausted. Offer identity comparison includes all encoded
fields except mutable price. Both panels are existing teacher trajectories;
development has already been exposed.

| Sequence property | Training | Development |
| --- | ---: | ---: |
| All SHOP sale targets | 5,471 | 716 |
| Joker capacity exhausted | 5,384 | 705 |
| Immediately followed by buying the same offered Joker | 5,190 | 675 |
| That purchase was illegal before the sale, with capacity exhausted | **5,183 / 94.74%** | **674 / 94.13%** |
| Base matches the sale on those replacement sequences | 1,912 / 36.89% | 165 / 24.48% |
| Base matches the next purchase after the teacher's sale | **4,851 / 93.59%** | **624 / 92.58%** |

The purchase remains well matched even where the base's preceding sale choice
was wrong: **3,018/3,271 training (92.27%) and 462/509 development (90.77%)**.
These are separate predictions on recorded states after the teacher's action,
not continuations after a base-chosen alternative. They therefore identify
where behavior differs without attributing its outcome cost.

Of 199 development sale states where the base instead proposes a purchase,
182 belong to the newly legal replacement sequences above. The teacher's
intended Joker purchase cannot simply be moved before the sale there: it is
illegal at that state. A different earlier purchase could still be harmless
or beneficial; the census does not evaluate that alternative sequence.

The source explains why this relation appears. The frozen teacher's
[shop search](../../eval/pretraining_transfer_20260923/candidate/src/pylatro_agent/heuristic_shop_search.py)
evaluates current-Joker removal together with adding an offered Joker, using
sampled build output, income, affordability and reserve costs. For a chosen
replacement it emits the sale action first; the next decision recomputes the
purchase. Direct beneficial sales also exist, including Stencil/Campfire
cases. The measured consecutive-label pattern does not identify the exact
internal branch that executed in every state.

The model already receives offered Jokers in its observation and contextual
encoder. The evidence supports investigating how it learns the relation
between an owned Joker, an offered replacement and the enabling sale. It
does not show that this information is absent or that a more expressive
readout will necessarily transfer. The next-purchase label already exists in
the saved sequence and could supervise that relation without hidden future
information entering model inputs.

For early discards, the two cheap alternatives checked here explain little:

| Early-discard context | Training | Development |
| --- | ---: | ---: |
| At least one face-down hand card | 108 / 3,705 | 15 / 514 |
| All hand cards visible | 3,597 | 499 |
| Conditional teacher match with all hand cards visible | **1,553 / 43.17%** | **197 / 39.48%** |
| Wrong conditional action, but identical encoded discarded and retained sequences | 5 | 1 |

All six identical-sequence mismatches occur with face-down cards. This check
preserves card order and compares all encoded fields except hand-slot ID.
It measures observed-feature identity, not native outcome equivalence:
quantization and omitted state can still distinguish cards. Broader strategic
equivalence between different card features remains unmeasured. The poor
discard fit persists in fully visible hands, so face-down masking cannot
explain most of it.

A separate source observation remains a hypothesis for later investigation.
The [tokenizer](../../eval/pretraining_transfer_20260923/candidate/src/pylatro_agent/tokenizer.py)
emits draw-pile cards in runtime order, and
[embeddings](../../eval/pretraining_transfer_20260923/candidate/src/pylatro_agent/embeddings.py)
add absolute positions. The teacher's
[sampled discard helper](../../eval/pretraining_transfer_20260923/candidate/src/pylatro_agent/heuristic_draw.py)
sorts the remaining pool and uses fixed artificial sample seeds. That creates
a possible sensitivity to ordering that this public-sample calculation does
not use. No input permutation, teacher-invariance test or model-sensitivity
measurement was performed here; the count of labels from that helper is also
unknown. Hand and Joker ordering must remain intact in any future test.

The next useful intervention is narrowly assisting SHOP sales on the existing
matched current-policy panel, while leaving other decisions with the student.
The [completed whole-shop intervention](current_policy_phase_results_20260929.md)
established headroom, but it did not
attribute that headroom to these sale decisions. A bounded sale intervention
can test whether this sequence gap has material gameplay consequences before
paying for a replacement-aware fitting experiment. Source ordering and
discard-search sensitivity are deferred to keep this test focused. No new fit
or architecture change is selected by this report.

The census completes in **8.1 seconds**, checks every target identity and
reproduces the prior agreement controls, with **39 groups independently
recounted**. A subsequent saved-prediction analysis checks all 5,857 replacement
follow-through states and their consecutive identities. Both entire processes
actually exited zero; neither made model, teacher, simulator or training
calls. No experiment process remains live. Final generalization, win-rate,
greedy-retention, early-attribution and independent-fit requirements remain
unchanged and unmet.

Artifacts: [context counts and source pins](../../eval/consequence_prediction_20260927/current_label_context_v1/result_v1.json),
[context actual-exit receipt](../../eval/consequence_prediction_20260927/current_label_context_owner_terminal_v1.json),
[purchase follow-through counts](../../eval/consequence_prediction_20260927/current_replacement_followthrough_v1.json),
[follow-through actual-exit receipt](../../eval/consequence_prediction_20260927/current_replacement_followthrough_owner_terminal_v1.json),
[census implementation](../../eval/consequence_prediction_20260927/current_label_context_v1.py),
and [follow-through implementation](../../eval/consequence_prediction_20260927/current_replacement_followthrough_v1.py).
