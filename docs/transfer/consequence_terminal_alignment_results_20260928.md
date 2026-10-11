# Immediate prediction does not supply terminal action ordering

The frozen wider scorer provides little separation of winning and losing continuations in the existing training panel. Its centered preference correctly ranks **214/416 pairs (51.44%)**, compared with **218/416 (52.40%)** for the base policy. On the 291 pairs with the same immediate-clear outcome, preference ordering is **145/291 (49.83%)**. These are descriptive counts, not resolved differences from chance or from base.

This addresses a narrower question than the [width experiment](consequence_relational_width_results_20260928.md). Width improved immediate consequence prediction; the present [predeclared diagnostic](consequence_terminal_alignment_20260928.md) asks whether those outputs already distinguish delayed terminal outcomes. No terminal labels entered the scorer's training. Every continuation takes one assessed action and then follows the same frozen base greedy controller, using the original RNG realization.

| Output used to rank actions | Correct, all 416 terminal pairs | Correct, 291 same-clear pairs |
|---|---:|---:|
| Base log probability | 218 (52.40%) | 146 (50.17%) |
| Centered preference | 214 (51.44%) | 145 (49.83%) |
| Predicted immediate score | 196 (47.12%) | 128 (43.99%) |
| Predicted immediate clear | 206 (49.52%) | 137 (47.08%) |

There are no exact ties in these four outputs. All 416 pairs come from 208 mixed-outcome states. The 291 same-clear pairs span 162 states; equal weighting per state yields preference ordering of **49.69%**, versus **49.69%** for base. Early means Antes 1–3, matching the existing protocol. Early preference ordering is **107/218 (49.08%)**, versus base **108/218 (49.54%)**; later it is **107/198 (54.04%)**, versus base **110/198 (55.56%)**. These partitions do not establish a statistically resolved early/later difference.

Even realized immediate outcomes fail to determine the terminal ranking. Winning continuations start with a higher immediate score in 177 pairs, a lower score in 180 and an equal score in 59. Immediate-clear labels favor the terminal winner in 69 pairs, favor the loser in 56 and tie in 291. This does not prove a model could predict the stochastic terminal distinction from visible inputs. It shows why improving these immediate targets alone cannot be assumed to improve delayed decisions.

Across all 1,000 states, restricting choices to the three assessed proposals gives **516** terminal wins for preference, **512** for score and **516** for clear, versus **517** for base and **519** for teacher. Preference cures 13 base losses and loses 14 base wins. These are teacher-containing proposal diagnostics with a shared base continuation, not deployable autonomous policy win rates.

The full-support preference choice is assessed in **968/1,000** states, with 503 known wins. Its other **32 choices have unknown continuation outcomes**. Therefore its terminal-win count is bounded by **503–535**, or **−14 to +18** versus base's 517. We do not substitute a labelled alternative or assume an outcome for those 32 choices.

The result supports testing an explicit delayed-outcome target before further capacity changes. The strongest unresolved alternatives are that the frozen action features cannot support the target, that this sample is insufficient, or that much of the pairwise distinction reflects unpredictable suffix randomness. A small terminal readout could distinguish some of these possibilities; the current evidence does not select an optimizer, duration, policy coefficient or advancement threshold. No new fit or policy candidate is selected by this report. Existing rejection decisions and the complete fresh-game, early-attribution and final-confirmation bar remain unchanged.

All 1,000 source games contributed to the immediate-feature fit. A subsequent split of their terminal labels can diagnose readout learning, but cannot establish generalization to games unseen by the feature pathway. Held-out source-game prediction and retention evidence are still required before policy integration. Realized original-RNG suffix labels are controller-dependent Monte Carlo outcomes, not known conditional expected action values. The clear sigmoid remains an immediate-clear estimate and is not evaluated as a calibrated terminal-win probability.

The extraction saves **1,000 × 3 × 384** frozen action features from `readout[1]`, plus three-head outputs, base priors and reference corrections. Only the five approved observation feature arrays enter the model; seed, step and outcomes remain metadata/targets. Independent full-support recomputation exactly reproduces all **1,152,000 hidden coordinates**, **9,000 output coordinates**, **246,723 legal-action evaluations**, and **2,000 overlapping original-proposal predictions**. Source identities, selected/retained slots, labels and all reported counts reconcile. Measurement takes 5.28 seconds and audit 6.61 seconds; the entire owner exits successfully. No optimization or new simulation occurs.

Artifacts: [owner receipt](../../eval/consequence_prediction_20260927/terminal_alignment_owner_terminal_v1.json), [frozen protocol](../../eval/consequence_prediction_20260927/terminal_alignment_protocol_v1.json), [results](../../eval/consequence_prediction_20260927/terminal_alignment_v1/result_v1.json), [audit](../../eval/consequence_prediction_20260927/terminal_alignment_v1/verification_v1.json), and [state/action records](../../eval/consequence_prediction_20260927/terminal_alignment_v1/rows.jsonl).
