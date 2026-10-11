# Better probabilities have not resolved the sale decisions

The independent sale-family correction improves held-out probability loss but still fails the same three of the original thirteen development checks. Correct SHOP sales rise from 160 to 163 out of 716; the base gets 167. This candidate is rejected before gameplay or confirmation. The [fit and full native scan](sale_family_results_20260930.md) remain the authoritative endpoint evaluation.

The following diagnostics reuse saved predictions and observations. They make no model forwards, simulator calls, new fits, decoder changes, or threshold sweeps. All three whole processes exited successfully before their results were admitted. Development is already exposed; these are exploratory diagnoses, not confirmation evidence.

## Where the improvement went

On 7,512 active development SHOP states, the family correction changes 48 full actions. It gains 18 exact teacher matches and loses 14. For the 703 active teacher-sale states specifically, it gains nine matches and loses six. The other thirteen SHOP sale states are outside the fixed active predicate and remain wrong.

| Active teacher-sale outcome | States | Sum of sale NLL changes |
| --- | ---: | ---: |
| Remains correct | 154 | +0.164306 |
| Becomes correct | 9 | −1.301315 |
| Becomes wrong | 6 | +0.833342 |
| Remains wrong | 534 | −27.129259 |

Negative changes improve loss. Of the total −27.432926 sale-loss change, **98.9% comes from states that still choose the wrong full action**. This explains how the reported loss improvement coexists with only three additional correct sales. The underlying conditional Joker rankings are exactly unchanged.

Across all active states, sale-versus-rest binary NLL improves by −0.002194, with a 128-game bootstrap 95% interval of [−0.003941, −0.000575]. Expected sale counts nevertheless move from 707.96 to 724.18 against 703 labels. Teacher-non-sale loss worsens. A blanket increase in selling is not supported by these results. Current-Ante-below-four states gain four exact matches and lose seven; that descriptive split does not measure a permanent early-policy intervention.

Sources: [decision changes](../../eval/consequence_prediction_20260927/sale_family_changes_results_v4.json), [independent recount and loss decomposition](../../eval/consequence_prediction_20260927/sale_family_changes_independent_verification_v1.json), [actual whole-process exit](../../eval/consequence_prediction_20260927/sale_family_changes_owner_terminal_v4.json).

## Two remaining restrictions

The model ranks the teacher's Joker first in 392 of the 703 active sale states, yet executes that sale in only 163. The remaining 229 divide as follows:

| Correct Joker rank, wrong full action | States |
| --- | ---: |
| Sale-family probability is too low for even a perfectly concentrated conditional sale distribution to win | 187 |
| Sale family already has the highest probability, but spreading that probability among Joker choices prevents the teacher's full action from winning | 42 |

These are exact comparisons against the strongest competing full action, reconstructed from native FP32 logits and legal masks. There are no unresolved bounds in this population. Holding conditional probabilities fixed, the median additional family-logit shift needed for these 229 teacher sales to win is 1.294620—an odds multiplier of 3.65. No such shift was applied.

A teacher-dependent conditional-only oracle could reach at most 380/716 SHOP sale matches while fixing the current family probabilities. A separate family-only oracle could reach at most 392/716 while fixing conditional ranks. These bounds cannot be added and are not deployable policies. Selecting the highest-probability family first has not been shown to improve utility or overall accuracy.

Sources: [exact decision limits](../../eval/consequence_prediction_20260927/sale_family_decision_limits_results_v1.json), [actual whole-process exit and independent margin check](../../eval/consequence_prediction_20260927/sale_family_decision_limits_owner_terminal_v1.json).

## What this rules out, and what remains unknown

Duplicate Joker slots do not explain these errors. None of the 5,380 original-training or 703 development teacher-sale targets has another legal owned Joker with the same identity. This rules out that narrow alias explanation, including for all 229 correctly ranked missed sales. It does not establish that different Joker identities have different strategic utility. The separate duplicate-playing-card sampling question remains open.

Sources: [Joker alias census](../../eval/consequence_prediction_20260927/sale_joker_alias_census_results_v1.json), [actual whole-process exit](../../eval/consequence_prediction_20260927/sale_joker_alias_census_owner_terminal_v1.json).

The next diagnosis should recover the teacher's comparison margins and check their observable inputs. Its shop search scores a proposed sale together with the enabled purchase, then saves only the chosen action. We have not measured whether the remaining mismatches are decisive improvements, nearly tied alternatives, or distinctions the encoded observation cannot support. These are heuristic comparison values, not measured eventual win values. No new architecture or training intervention is selected here; the original long-horizon and fresh-game success requirements remain unmet.
