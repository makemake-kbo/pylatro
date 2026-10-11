# Relative-gain readout at a fixed feature endpoint

One CPU float64 least-squares solve changes only a diagnostic copy of the 64 gain-output coefficients. The relative endpoint’s features, action policy, other consequence outputs and gain bias are fixed. No gameplay checkpoint is produced. The comparison below is against the original relative-gain readout, not the earlier absolute-gain DENSE control.

The solve uses 330,282 observed training pairs from 50,055 gain-labelled states, within the original 202,141 decisions and 1,000 games. Features and standardized targets are centered within each observed pair set, with the original per-decision weighting. No development arrays choose coefficients or numerical settings.

| Training objective and numerical check | Value |
|---|---:|
| Original half squared loss per corpus row | 0.044251453 |
| Solved half squared loss per corpus row | 0.0405377473 |
| Design rank / coefficients | 64 / 64 |
| Fixed relative singular-value cutoff | 1e-12 |
| Largest / smallest singular values | 1.20143547 / 0.00924756844 |
| Original / solved coefficient norm | 0.137883 / 1.74262 |
| Independent normal-equation accumulation error | 3.59e-15 |
| Independent objective reconstruction error | 8.33e-17 |
| Independent normal-equation residual | 1.28e-15 |

The objective above uses the stored standardized float32 labels and original corpus denominator. The unchanged factor of one quarter from the four auxiliary tasks is omitted because it does not affect the optimum. Reported consequence errors below use the original raw labels and their own labelled-state denominators.

## Training

| Metric | Decisions | Original readout | Solved readout | Difference [95% interval] |
|---|---:|---:|---:|---:|
| net_gain/centered_raw_mse | 50055 | 0.0427276 | 0.0391417 | -0.00358582 |
| net_gain/centered_normalized_mse | 50055 | 0.357408 | 0.327413 | -0.0299947 |
| observable_teacher_slot_hit | 5380 | 0.299442 | 0.384572 | 0.0851301 |
| observed_teacher_slot_hit | 5380 | 0.342751 | 0.43513 | 0.0923792 |
| top_pair_unobserved | 57257 | 0.197303 | 0.210175 | 0.0128718 |
| observed_pair/concordance | 48224 | 0.619162 | 0.66545 | 0.0462873 |
| observed_pair/best_hit | 48224 | 0.295869 | 0.360111 | 0.0642419 |
| observed_pair/regret | 48224 | 0.168274 | 0.138128 | -0.0301462 |

Owned-slot agreement is conditional on teacher SALE. Observable support uses legal owned slots and visible Joker offers; observed support and pair ranking use the oracle gain mask. Neither rule decides whether to sell or executes a purchase. A common gain offset is unconstrained; no absolute-gain threshold is evaluated.

All gain-labelled states:

| Comparison | Ranking states | Original concordance | Solved concordance | Difference [95% interval] |
|---|---:|---:|---:|---:|
| Different owned Jokers, same offer | 47312 | 0.597267 | 0.653985 | 0.0567184 |
| Different offers, same owned Joker | 23973 | 0.749815 | 0.747598 | -0.00221697 |
| Both different | 23727 | 0.662658 | 0.686462 | 0.0238042 |

| Comparison | Original centered-error contribution | Solved contribution | Difference [95% interval] |
|---|---:|---:|---:|
| Different owned Jokers, same offer | 0.0294432 | 0.0265153 | -0.0029279 |
| Different offers, same owned Joker | 0.000983825 | 0.0010817 | 9.78771e-05 |
| Both different | 0.0123005 | 0.0115447 | -0.000755792 |

Teacher-sale states:

| Comparison | Ranking states | Original concordance | Solved concordance | Difference [95% interval] |
|---|---:|---:|---:|---:|
| Different owned Jokers, same offer | 5132 | 0.586628 | 0.652588 | 0.0659599 |
| Different offers, same owned Joker | 3228 | 0.907162 | 0.888964 | -0.0181985 |
| Both different | 3131 | 0.779174 | 0.784548 | 0.00537402 |

| Comparison | Original centered-error contribution | Solved contribution | Difference [95% interval] |
|---|---:|---:|---:|
| Different owned Jokers, same offer | 0.034244 | 0.030758 | -0.00348597 |
| Different offers, same owned Joker | 0.00237998 | 0.00242274 | 4.27554e-05 |
| Both different | 0.0209016 | 0.0193471 | -0.00155446 |

## Development

| Metric | Decisions | Original readout | Solved readout | Difference [95% interval] |
|---|---:|---:|---:|---:|
| net_gain/centered_raw_mse | 6617 | 0.0440102 | 0.0399464 | -0.00406378 [-0.00502699, -0.00310032] |
| net_gain/centered_normalized_mse | 6617 | 0.368137 | 0.334144 | -0.0339928 [-0.0420499, -0.0259336] |
| observable_teacher_slot_hit | 703 | 0.297297 | 0.371266 | 0.0739687 [0.0412033, 0.106222] |
| observed_teacher_slot_hit | 703 | 0.347084 | 0.42532 | 0.0782361 [0.044411, 0.112426] |
| top_pair_unobserved | 7512 | 0.190761 | 0.199681 | 0.00891906 [-0.00232606, 0.0200467] |
| observed_pair/concordance | 6341 | 0.627328 | 0.675459 | 0.0481309 [0.0368701, 0.0602086] |
| observed_pair/best_hit | 6341 | 0.297587 | 0.376281 | 0.0786942 [0.0566483, 0.101198] |
| observed_pair/regret | 6341 | 0.159699 | 0.126873 | -0.0328257 [-0.0439497, -0.0220933] |

Owned-slot agreement is conditional on teacher SALE. Observable support uses legal owned slots and visible Joker offers; observed support and pair ranking use the oracle gain mask. Neither rule decides whether to sell or executes a purchase. A common gain offset is unconstrained; no absolute-gain threshold is evaluated.

All gain-labelled states:

| Comparison | Ranking states | Original concordance | Solved concordance | Difference [95% interval] |
|---|---:|---:|---:|---:|
| Different owned Jokers, same offer | 6241 | 0.60351 | 0.66403 | 0.0605205 [0.0468359, 0.0752248] |
| Different offers, same owned Joker | 3187 | 0.754372 | 0.759545 | 0.00517333 [-0.00886269, 0.019123] |
| Both different | 3155 | 0.677682 | 0.704224 | 0.0265427 [0.0192178, 0.033883] |

| Comparison | Original centered-error contribution | Solved contribution | Difference [95% interval] |
|---|---:|---:|---:|
| Different owned Jokers, same offer | 0.0305334 | 0.0273807 | -0.00315271 [-0.00383188, -0.00246431] |
| Different offers, same owned Joker | 0.000918094 | 0.00100021 | 8.21111e-05 [4.7791e-05, 0.000116722] |
| Both different | 0.0125586 | 0.0115654 | -0.000993183 [-0.00130216, -0.000685617] |

Teacher-sale states:

| Comparison | Ranking states | Original concordance | Solved concordance | Difference [95% interval] |
|---|---:|---:|---:|---:|
| Different owned Jokers, same offer | 667 | 0.588391 | 0.641224 | 0.0528335 [0.0345552, 0.0713846] |
| Different offers, same owned Joker | 433 | 0.910974 | 0.894799 | -0.0161752 [-0.0332066, 0.00155551] |
| Both different | 425 | 0.788485 | 0.800027 | 0.0115425 [0.000969644, 0.0227707] |

| Comparison | Original centered-error contribution | Solved contribution | Difference [95% interval] |
|---|---:|---:|---:|
| Different owned Jokers, same offer | 0.0325704 | 0.0298313 | -0.0027391 [-0.00359177, -0.00192684] |
| Different offers, same owned Joker | 0.00235097 | 0.00219221 | -0.000158757 [-0.000320197, -5.48409e-06] |
| Both different | 0.0209854 | 0.0191463 | -0.00183911 [-0.00253127, -0.0011922] |

## Verification and limits

Native feature extraction reproduces every archived auxiliary forecast at the original consecutive 32-row batching. It checks checkpoint identity, hook cleanup, observation-derived support and compact row alignment. The technical preflight independently reconstructs pair features in CPU float64. The extraction audit reconstructs every observable-pair readout from stored features and original weights.

The solve is checked against independently accumulated normal equations and synthetic recoverable, rank-deficient, empty, singleton and common-offset cases. Evaluation independently enumerates every scalar metric on all 64,769 active rows and reproduces every original relative-readout metric and development interval. Development uses 10,000 paired resamples of all 128 complete games with seed1729; training uses all 1,000 games for points. The bootstrap is the previously tested implementation, not an independently repeated implementation in this probe. Intervals are descriptive and marginal on an exposed panel.

A lower training loss would identify remaining optimization in this fixed auxiliary readout. A minimum-MSE linear solution is not an upper bound on choice accuracy or nonlinear learning. Any remaining gap could involve features, objective weighting, target noise or their interaction. The diagnostic does not identify one of those causes by elimination alone.

The gameplay policy remains the rejected relative endpoint, including its five failed policy checks and 163/716 development sale matches versus base167. Prediction improvement cannot qualify that unchanged policy. No fresh games or independent confirmation are performed. Full success remains fresh sampled win improvement of at least five percentage points against both base and original ordinary4k with paired lower bounds above zero; greedy lower bounds strictly above minus five points against both; positive learned early-policy attribution and independent-fit direction; and one final independent fit plus untouched confirmation. No win-rate or long-horizon improvement is established.

[Plan](shop_gain_readout_probe_plan_20261001.md) · [Fit and numerical checks](../../eval/consequence_prediction_20260927/shop_gain_readout_probe_fit_v1.json) · [All evaluation metrics](../../eval/consequence_prediction_20260927/shop_gain_readout_probe_results_v1.json) · [Fit admission](../../eval/consequence_prediction_20260927/shop_gain_readout_fit_owner_terminal_v1.json) · [Evaluation admission](../../eval/consequence_prediction_20260927/shop_gain_readout_probe_owner_terminal_v1.json)
