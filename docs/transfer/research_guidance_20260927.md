# Guidance for the next research decision

Prepared in response to the user's September 27 concern that successive experiments are consuming time without enough direction. This supplements the [single exploratory fit workflow](experiment_workflow.md). It is guidance for subsequent decisions, not a change to the running combined model's fixed development protocol.

Finish the current development screen and its scheduled verification. Before starting another fit, explain which unresolved mechanism the next action will distinguish. A disappointing result alone is not a reason to launch another loss combination.

## What the evidence supports

- Ordinary continuation fits the later training corpus better while regressing on fixed held-out teacher states. There is a transfer problem before autonomous trajectory drift is needed to explain it.
- Ranking learns about 90% of comparable training preferences, but only about 44% held out. It shows some local transfer and a sampled gain over the base, but no resolved sampled advantage over ordinary continuation, a greedy decline, and no demonstrated early-policy benefit.
- Preservation reduces average degradation. It has not established a sufficient gameplay remedy, and can preserve existing errors as well as useful behavior.
- The combined model changes the correction/retention tradeoff: +40 net immediate clears in preference cases, -35 on states where the base matched the teacher, +1 elsewhere. Its +6 overall difference from the base is unresolved. These are immediate teacher-state consequences, not six additional game wins.
- The recent continuations reuse a later 1,000-game corpus. The original 5,000-game corpus contributed inherited weights under older readouts; the recent studies are not a controlled corrected-policy refit over that full corpus. Neither dataset sufficiency nor an architectural limit is established.

Evidence: [continuation transfer](continuation_transfer_diagnostic_20260925.md), [ranking results](counterfactual_ranking_results_20260927.md), [preference transfer](ranking_preference_transfer_20260927.md), [combined diagnostics](ranking_preservation_development_20260927.md), and [corpus provenance](pretraining_gap_focus_20260923.md).

## First unresolved question: when does transfer fail?

Use existing checkpoints before buying another training run. For the combined model, the base and final endpoint already have development measurements; checkpoints at updates 1,000 and 2,000 are saved. Reuse any equivalent completed measurements before generating more.

The preferred next diagnostic is a small learning-curve comparison on fixed development populations: training preferences, held-out preferences, and held-out states where the base already matched the teacher. Use the same state identities and matched denominators at each snapshot. Measure preferred-action ordering and actual selected actions separately; replay only newly encountered choices needed to establish consequences. Include early and late decisions and retain all results. Do not create another training pilot or run a full game suite merely to draw this curve.

Ask whether held-out corrections appear and subsequently deteriorate as fitting continues, or whether they remain weak throughout despite strong training learning. Track retention losses alongside corrections; better average teacher NLL alone cannot answer this question. This is diagnosis on exposed development evidence, not checkpoint selection for the current fixed study or fresh confirmation.

| Observation | Next decision it would support |
| --- | --- |
| Earlier snapshots give a better held-out correction/retention tradeoff, which later deteriorates | Test training duration or a development stopping rule as the next single change. Do not automatically inherit the 4,000-update budget. |
| Training corrections improve while held-out corrections stay weak at every measured snapshot | Examine coverage and transfer of the relevant comparisons before another regularization or loss-weight change. |
| Comparable training decisions remain poorly learned | Investigate objective conflict or optimization on those represented cases; do not label this solely a coverage problem. |
| Fixed-state consequences improve but autonomous development behavior does not | Inspect existing autonomous traces and a bounded set of student-visited states to distinguish state-distribution errors and short-term versus eventual-value tradeoffs. |
| Differences are too small or mixed to distinguish explanations | Report that uncertainty and choose the cheapest additional observation that could change the next decision; do not force a preferred story. |

These patterns guide a hypothesis; they do not uniquely identify a cause. If another existing result already answers this question, cite it and skip the diagnostic.

If coverage becomes the next question, distinguish represented-but-unlearned comparisons from sparse decisive contexts and learned-but-untransferred comparisons. Use existing corpus scans and compatible historical records first, including the distinction between the 1,000-game continuation corpus and original 5,000 games. Define mechanical similarity before inspecting outcome splits; lack of an exact card/Joker inventory is not proof that duplicate states are required. Do not automatically scale to the full corpus without checking compatibility and stating the prediction that more diverse data would test.

## A short decision note before the next fit

Keep this to one page, linking existing evidence rather than duplicating it:

1. **Question:** one falsifiable explanation and the strongest competing explanation.
2. **Discriminating evidence:** what current evidence supports each, and the cheapest remaining check. State what result would weaken the favored explanation.
3. **Intervention:** one controlled change and why the diagnostic predicts it will help. Keep the comparison fair; explain any necessary difference in exposure or compute.
4. **Prediction:** the expected change in held-out corrections, retained behavior, early decisions and development gameplay. Set an actionable proceed/revise/stop rule before outcomes.
5. **Budget:** expected wall time, update budget justified by evidence, existing artifacts reused, and the condition that stops spending. Four thousand updates is a historical comparison budget, not a requirement for every future idea.
6. **Decision after the result:** what was learned, which explanation weakened, and whether to keep, revise or discard the candidate. Do not substitute a list of successful implementation checks for this decision.

The note is a reasoning requirement, not a new user-approval step. Continue autonomously within the existing authorized scope once the evidence and decision rule are concrete.

## Spend compute in stages

Use existing artifacts, then the bounded diagnostic, then one exploratory fit only if justified, then the small development screen. Final independent training and untouched confirmation are reserved for a candidate that meets the development readiness rule. Keep final evidence separate from checkpoint or candidate selection; report sampled, greedy and early-policy results together.

Reuse verified controls and unchanged components where their provenance permits. Test changed logic and preserve sufficient raw evidence, but avoid repeated technical pilots, broad verification scaffolding or report rewrites that cannot affect a scientific decision. Successful execution is necessary; it does not establish a useful intervention.

Do not automatically add another loss, sweep coefficients, freeze the encoder, enlarge the context window or change architecture because the current candidate is disappointing. Each would need a specific predicted mechanism and a discriminating check. World-model work remains deferred under the existing user instruction.

Progress reports should lead with: what uncertainty was resolved, what remains, the next decision, and its cost. Training/evaluation percentages are secondary when a scientific finding is available.
