# Adapting the readout features with the rest of the policy fixed

Fails 3 of the original thirteen development checks. This endpoint does not qualify for gameplay or confirmation.
No learned long-horizon or win-rate gain is established by this teacher-state evaluation.

Exactly four readout projection tensors (57,408 parameters) learn from the same 26,782 sale states and 5,000 current-rule games. The encoder, all existing heads, purchase proposal, 64 output weights and common output bias remain frozen to the completed broader-data parent. Architecture, masks, observations, labels and game population do not change.

The single CPU float64 full-batch L-BFGS call ran 50 internal iterations and 52 training-objective evaluations. It used the predeclared bounds and training-only line search/stopping rules; no development checkpoint or hyperparameter selection occurred. This is a bounded endpoint, not a global-convergence claim. Its numerical optimizer differs from the preceding convex output-only solver because the adapted objective is nonlinear.

## Training conditional choice

| Population | Selected sales | Parent NLL | Adapted NLL | Parent matches | Adapted matches |
|---|---:|---:|---:|---:|---:|
| Original 1,000 games | 5380 | 0.541931 | 0.683178 | 4396 | 4155 |
| Additional 4,000 games | 21402 | 1.241604 | 1.007923 | 11628 | 12818 |

These native counts describe Joker choice conditional on selling. They do not measure full-action accuracy or game wins. All outcomes are retained.

## Fixed development comparison

| Policy | Sale matches | Sale NLL | Future purchase matches | Purchase matches |
|---|---:|---:|---:|---:|
| Base | 167/716 | 2.311783 | 428/469 | 4990/5323 |
| Ordinary 1k | 170/716 | 2.394442 | 426/469 | 4979/5323 |
| Purchase auxiliary | 158/716 | 2.406345 | 441/469 | 4987/5323 |
| Coupled scorer | 159/716 | 2.436645 | 437/469 | 4976/5323 |
| Fixed features: 5,000 games | 158/716 | 2.401581 | 438/469 | 4982/5323 |
| Disabled readout | 165/716 | 2.433811 | 438/469 | 4983/5323 |
| Adapted features: 5,000 games | 160/716 | 2.230373 | 438/469 | 4990/5323 |

Relative to the fixed-feature 5,000-game readout, sale NLL changes by -0.171208 [-0.221387, -0.124325]; sale exact-match rate changes by +0.002793 [-0.020979, +0.025993]. These are marginal descriptive paired source-game 95% bootstrap intervals on the exposed development panel.

## Joker ranking and full actions

On the 703 development sales where the branch is active, conditional Joker matches change from 377 to 392; full-action matches on those same rows change from 158 to 160. These counts exclude the thirteen inactive sales in the full comparison above.

Of the corrected Joker rankings, 33 become correct full actions and 35 still lose to another action family. With the correct Joker ranking unchanged, 2 full actions become correct and 13 become incorrect. Another 20 full matches are lost when the Joker ranking becomes wrong. These are categorical transitions on teacher states, not estimates of game value.

Using the learned purchase-proposal weights yields 392 conditional matches, compared with 389 when averaging offers uniformly. Their conditional NLL difference is +0.002067, with descriptive paired game interval [-0.003026, +0.007160]. Contextual card features still contain offer information in both calculations, so this does not isolate the causal benefit of the architecture.

## Unchanged checks

| Check | Result |
|---|---|
| future_nll_vs_base | PASS |
| future_nll_vs_ordinary1k | PASS |
| non_sale_nll_vs_ordinary1k | PASS |
| purchase_exact_vs_base | PASS |
| purchase_exact_vs_ordinary1k | PASS |
| sale_exact_vs_ablation | FAIL |
| sale_exact_vs_auxiliary | PASS |
| sale_exact_vs_base | FAIL |
| sale_exact_vs_ordinary1k | FAIL |
| sale_nll_vs_ablation | PASS |
| sale_nll_vs_auxiliary | PASS |
| sale_nll_vs_base | PASS |
| sale_nll_vs_ordinary1k | PASS |

## Verification and limits

The complete zero-update check passed before fitting. The endpoint audit verifies the three saved parameter endpoints and the predeclared first, middle and last optimizer-evaluation positions with independent NumPy calculus. Maximum independent loss error is 2.29e-16; gradient error 1.28e-17. Exactly four checkpoint tensors change and 322 remain identical. All 26,782 training sale rows undergo native replay: maximum logit error is 1.17e-05, NLL error 6.75e-06, and 0 near-boundary float64/native mode disagreements are recorded. Native choices are authoritative.

The full scan and independent native audit complete before admission. The readout changes conditional sale preferences; sale-family probabilities remain fixed at a given state. A teacher-state improvement would still need fresh sampled games, greedy guardrails, permanent early-policy attribution, and one final independent fit and untouched confirmation.

[Experiment](conditional_feature_adaptation_20260930.md) · [Fit owner](../../eval/consequence_prediction_20260927/conditional_feature_fit_owner_terminal_v1.json) · [Fit audit](../../eval/consequence_prediction_20260927/conditional_feature_fit_v1/verification_v1.json) · [Scan owner](../../eval/consequence_prediction_20260927/conditional_feature_scan_owner_terminal_v1.json) · [Scan audit](../../eval/consequence_prediction_20260927/conditional_feature_scan_v1/verification_v1.json)
