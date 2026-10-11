# Where the starting conditional model regressed

Two different gaps coexist. The new 64-weight output solver addresses weak learning in the added readout. The earlier 1,000-update continuation also changed the existing encoder and heads, and its conditional Joker-choice likelihood transferred poorly.

This analysis uses the already admitted base and conditional-model logs on all 5,380 training and 703 development teacher sales where the new readout is active. It adds no training, native inference, simulator calls, or gate. These active-sale averages differ from the original screen's averages over all 716 development sales.

| Loss component | Base training | Starting model training | Base development | Starting model development |
|---|---:|---:|---:|---:|
| Choose the sale family | 0.928168 | 0.957133 | 1.014950 | 1.047454 |
| Choose the Joker, given sale | 0.664973 | 0.539400 | 1.227846 | 1.324066 |
| Joint teacher action | 1.593140 | 1.496533 | 2.242796 | 2.371520 |

On development sales, the conditional component explains 0.096220 of the 0.128724 increase in joint NLL (74.75%). Its descriptive paired source-game interval is [0.052059, 0.139412]. The family component contributes 0.032504, interval [0.022742, 0.042033]. Both use 10,000 resamples of all 128 source games, normalized by selected sale counts, without multiple-comparison adjustment.

Conditional training matches improve 75.80% → 81.12%, while development matches change 53.34% → 52.92%. The greater development likelihood penalty with nearly unchanged accuracy is consistent with worsening confidence on these targets; this decomposition alone does not establish calibration or identify which errors lose games. The earlier readout-ablation analysis found the added branch negligible, so most of this pre-solver regression belongs to the changed existing pathway.

All 6,083 row identities and log-probability decompositions were checked. Reconstruction of joint native NLL differs by at most 8.80e-7; a separate recount against the base's saved native conditional log probabilities and family probabilities differs by at most 5.76e-7. The full scan of the output-only solver is still required before judging whether that intervention repairs enough of this gap. No learned win-rate or long-horizon gain follows from these diagnostic results.

[Source analysis](../../eval/consequence_prediction_20260927/conditional_parent_factor_v1.py) · [Results](../../eval/consequence_prediction_20260927/conditional_parent_factor_results_v1.json) · [Recount](../../eval/consequence_prediction_20260927/conditional_parent_factor_verification_v1.json) · [Successful owner](../../eval/consequence_prediction_20260927/conditional_parent_factor_owner_terminal_v1.json) · [Output-solver experiment](conditional_output_solver_20260930.md)
