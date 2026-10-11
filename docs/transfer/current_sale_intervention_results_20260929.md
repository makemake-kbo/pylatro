# Teacher assistance on shop sales improves current-policy wins

Restricting teacher help to its proposed Joker sales raises wins from
**11/64 to 24/64**, a paired **+20.31 percentage points [6.25, 34.38]** on the
existing matched panel. This establishes useful gameplay headroom in these
sale decisions and passes the declared rule for designing one exploratory
replacement-learning experiment. The model itself is unchanged; a learned
improvement remains to be demonstrated.

Early-only sale assistance wins **17/64**, a **+9.38-point [0.00, 18.75]**
difference. Its lower bound is not strictly positive, so the fixed early
headroom criterion is unmet. Do not interpret this result as proof of an early
policy benefit or absence of one.

The [frozen intervention](current_sale_intervention_20260929.md) runs two new
arms on the same 64 already-exposed seeds and reuses the admitted student and
whole-shop controls from the
[current phase diagnostic](current_policy_phase_results_20260929.md).

| Policy / assistance | Wins / games | Win rate | Changed student decisions |
| --- | ---: | ---: | ---: |
| Student, reused control | 11 / 64 | 17.19% | 0 |
| Teacher-proposed SHOP sales, current Ante below 4 | 17 / 64 | 26.56% | 63 |
| Teacher-proposed SHOP sales, all Antes | 24 / 64 | 37.50% | 247 |
| All SHOP decisions, reused control | 29 / 64 | 45.31% | 648 |

| Fixed paired comparison | Difference, percentage points | Exploratory paired 95% interval |
| --- | ---: | ---: |
| All-sale assistance − student, primary | **+20.31** | **[+6.25, +34.38]** |
| Early-sale assistance − student, secondary | +9.38 | [0.00, +18.75] |
| All-sale − early-sale assistance | +10.94 | [−1.56, +23.44] |
| Whole-shop − all-sale assistance | +7.81 | [−4.69, +20.31] |

For the primary contrast, 18 seeds change from loss to win and five change
from win to loss. Early assistance has eight gains and two losses. The
whole-shop comparison is unresolved; the two assisted policies are not
established as equivalent. These are marginal paired intervals from the fixed
10,000 seed bootstrap draws, not simultaneous confidence statements.

During eligible SHOP states, the teacher is queried on a private state copy.
Its proposal overrides the sampled student action only when it is a Joker
sale. The student controls other action proposals and every non-SHOP phase.
Student sales remain untouched when the teacher proposes another family.
The early arm applies the same rule at current Ante below 4. A student sample
is drawn before every query, including overridden steps, preserving the
original sampling convention. All controls and interventions use the same
immutable base, runtime, CPU FP32 batch1, per-game sampling seeds, deck, stake
and total-step cap.

The all-Ante intervention changes decisions in **56/64 games**. Its 247
replacements comprise **94 choices of a different Joker**, plus **81 purchases,
39 departures and 33 rerolls** replaced by a sale. The early intervention
changes 36 games, with 32 different-Joker choices and 31 different-family
choices. Thus both when to sell and which Joker to sell remain relevant to
the learning target; their separate causal contributions were not isolated.
Both arms make 63 early replacements: ten at Ante2 and 53 at Ante3.

A saved-trace follow-through check finds that **240/247** changed all-Ante
sales are immediately followed by a teacher purchase proposal. The student
executes that next proposal in **192/240 (80.00%)** cases. For early-only
assistance the corresponding counts are **62/63** and **48/62 (77.42%)**.
These observations support the replacement-sequence interpretation while
showing imperfect sampled follow-through. They do not compare the value of
the rejected purchase alternatives or prove that each intervention helps.

The [preceding sequence census](current_label_context_results_20260929.md)
showed that about 94–95% of teacher-trajectory sales immediately enable a
previously illegal Joker purchase. The new intervention connects assistance
on the sale family to eventual wins on current-policy trajectories. Its
assisted sales include any direct beneficial sales as well as replacements,
and the teacher retains privileged state access. It therefore does not
establish that all improvement comes from replacement transactions or that
the same improvement is attainable from the policy's observations.

The next step is one controlled replacement-learning experiment addressing
the owned-Joker / offered-Joker relation and the decision to sell. Existing
teacher sequences supply a prospective purchase label without giving future
observations to the model. The experiment must state whether it changes
supervision or architecture, use a fair control, and retain the user's single
exploratory-fit workflow. Better teacher agreement alone will not qualify as
success: autonomous wins and the original learned early-policy requirement
remain necessary. No second teacher-assistance variant or seed extension is
selected by this result.

All **128 new and 128 reused games are uncensored**. The serial native audit
reproduces all **19,929 new decisions**, including proposals, intervention
rules, actions, input/state/RNG hashes and terminal results. **192 prefix
comparisons covering 14,102 shared decisions** match exactly through the first
replacement. All-sale versus early-sale divergence first occurs at an eligible
late sale, as required. The model tensors remain unchanged, and teacher
queries preserve live state, inputs, UID counters and policy RNG.

Measurement took **601.5 seconds** and the audit **595.3 seconds**, each within
the fixed 1,798-second allowance. The owner was checked about ten minutes
apart and actually exited zero before admission. The saved-trace recount also
completed successfully. No experiment process remains live. The final
fresh-game learned gain, greedy retention, early attribution, independent fit
and untouched confirmation requirements remain unmet; world models remain
deferred.

Artifacts: [measurement and all game records](../../eval/consequence_prediction_20260927/current_sale_v1/result_v1.json),
[native audit and paired results](../../eval/consequence_prediction_20260927/current_sale_v1/verification_v1.json),
[whole-owner exit receipt](../../eval/consequence_prediction_20260927/current_sale_owner_terminal_v1.json),
[intervention and follow-through recount](../../eval/consequence_prediction_20260927/current_sale_trace_summary_v1.json),
[frozen protocol](../../eval/consequence_prediction_20260927/current_sale_protocol_v1.json),
[worker](../../eval/consequence_prediction_20260927/current_sale_v1.py),
and [independent auditor](../../eval/consequence_prediction_20260927/verify_current_sale_v1.py).
