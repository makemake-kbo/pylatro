# Continued imitation overfits held-out teacher decisions

Both ordinary continuations improve training fit while worsening imitation on the same held-out teacher states. Training NLL falls from **0.496954 to about 0.3717**, but held-out NLL rises from **0.639106 to about 0.6904**. The pooled held-out change is **+0.051260 [0.043211, 0.059862]**; higher NLL is worse. Both replicas move in this direction. Consequence weighting also overfits and worsens held-out play imitation relative to ordinary fitting.

This locates a concrete transfer gap before autonomous trajectory drift is needed: the regression is already visible on fixed teacher states. It does not identify the cause of the entire original pretraining-to-win gap, prove an architectural limit, or establish a new long-horizon improvement. World-model work remains deferred.

This diagnostic uses the **already exposed** Stage 2 panel: 128 games and 26,045 decisions. Its intervals are descriptive, unadjusted paired game-bootstrap intervals, not a new confirmation test. The preceding [fresh-game study](consequence_weighting_study_20260924.md) independently found 28.00% sampled wins for the base, 28.19% after ordinary continuation and 26.81% after weighting, with four of five scientific gates failing.

## Training versus held-out imitation

The base is the prior Stage 2 trainable-encoder endpoint. Each continuation receives 4,000 additional updates on the same 202,141 training decisions. These are two continuation replicas from one common pretrained base. Accuracy means exact agreement with the recorded teacher action.

| Policy | Training NLL | Held-out NLL | Training agreement | Held-out agreement | Held-out matches / 26,045 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Common base | 0.496954 | 0.639106 | 85.830% | 82.718% | 21,544 |
| Ordinary 0 | 0.371671 | 0.690329 | 88.878% | 82.461% | 21,477 |
| Ordinary 1 | 0.371731 | 0.690402 | 88.914% | 82.473% | 21,480 |
| Weighted 0 | 0.388691 | 0.699683 | 88.453% | 82.123% | 21,389 |
| Weighted 1 | 0.388709 | 0.692752 | 88.484% | 82.173% | 21,402 |

Ordinary fitting raises training agreement by about three percentage points, but pooled held-out agreement falls by **0.251 points [-0.480, -0.023]**. Weighting lowers held-out agreement by another **0.319 points [-0.471, -0.168]** relative to ordinary fitting. This is evidence of overfitting under these continuation recipes, not evidence that the architecture cannot learn the task.

## Where the loss regressions occur

The table compares each two-replica mean with the common base. Negative training changes indicate improved fitting; positive held-out changes indicate worse teacher-action likelihood. All four groups were specified before endpoint inference.

| Decisions | Ordinary training NLL change | Ordinary held-out NLL change | Weighted training NLL change | Weighted held-out NLL change |
| --- | ---: | ---: | ---: | ---: |
| All decisions | -0.125253 | +0.051260 | -0.108254 | +0.057112 |
| Played hands | -0.297137 | +0.083482 | -0.285127 | +0.116739 |
| Discards | -0.423360 | +0.181823 | -0.267584 | +0.181727 |
| Joker sales | -0.520272 | +0.476670 | -0.428108 | +0.462492 |

The weighted-versus-ordinary held-out play difference is **+0.033257 NLL [0.021504, 0.044825]** and **-1.235 agreement points [-1.742, -0.734]**. Both replicas regress. Thus fitting the selected costly training plays better did not transfer to broader held-out play imitation.

Discard and Joker-sale NLL also worsen substantially versus the base under both objectives. Their weighted-versus-ordinary held-out differences remain unresolved. The training-state discard/sale regressions therefore cannot simply be assigned as the cause of weighting's autonomous result. Higher teacher NLL alone also does not prove that a non-teacher action has worse gameplay consequences.

## All predefined paired comparisons

Each interval resamples all 128 games 10,000 times with seed 1729, retaining both continuation replicas inside each game cluster. Estimates weight decisions equally; shared-base predictions are counted once. Games without a particular action family contribute zero numerator and denominator. No draw had a zero denominator. Accuracy changes are in percentage points; NLL changes are in nats.

| Family | Contrast | NLL change [95% interval] | Agreement change, pp [95% interval] | NLL change, replica 0 / 1 | Agreement change, pp, replica 0 / 1 |
| --- | --- | ---: | ---: | ---: | ---: |
| All decisions | Ordinary - base | +0.051260 [+0.043211, +0.059862] | -0.251 [-0.480, -0.023] | +0.051224 / +0.051297 | -0.257 / -0.246 |
| All decisions | Weighted - base | +0.057112 [+0.049489, +0.065017] | -0.570 [-0.796, -0.350] | +0.060578 / +0.053647 | -0.595 / -0.545 |
| All decisions | Weighted - ordinary | +0.005852 [+0.002235, +0.009411] | -0.319 [-0.471, -0.168] | +0.009354 / +0.002350 | -0.338 / -0.299 |
| Played hands | Ordinary - base | +0.083482 [+0.063199, +0.103759] | -0.393 [-1.045, +0.222] | +0.094660 / +0.072305 | -0.474 / -0.311 |
| Played hands | Weighted - base | +0.116739 [+0.094360, +0.139037] | -1.628 [-2.329, -0.915] | +0.136676 / +0.096803 | -1.947 / -1.309 |
| Played hands | Weighted - ordinary | +0.033257 [+0.021504, +0.044825] | -1.235 [-1.742, -0.734] | +0.042016 / +0.024497 | -1.472 / -0.998 |
| Discards | Ordinary - base | +0.181823 [+0.128068, +0.237870] | -0.779 [-2.085, +0.554] | +0.142057 / +0.221589 | -0.567 / -0.992 |
| Discards | Weighted - base | +0.181727 [+0.133274, +0.233193] | -0.531 [-1.756, +0.762] | +0.154656 / +0.208799 | +0.000 / -1.062 |
| Discards | Weighted - ordinary | -0.000096 [-0.020423, +0.020167] | +0.248 [-0.646, +1.123] | +0.012599 / -0.012791 | +0.567 / -0.071 |
| Joker sales | Ordinary - base | +0.476670 [+0.381343, +0.577826] | +1.016 [-1.202, +3.228] | +0.421576 / +0.531764 | +2.031 / +0.000 |
| Joker sales | Weighted - base | +0.462492 [+0.371773, +0.557418] | +0.358 [-1.903, +2.563] | +0.438004 / +0.486980 | +0.956 / -0.239 |
| Joker sales | Weighted - ordinary | -0.014178 [-0.063769, +0.033628] | -0.657 [-2.451, +1.012] | +0.016427 / -0.044784 | -1.075 / -0.239 |

All 13 observed action families, decision-weighted and equal-game summaries, and exact unrounded comparisons are retained in the [diagnostic data](data/continuation_transfer_diagnostic_20260925.json). The family breakdowns do not add acceptance gates or select a checkpoint.

## Verification and limits

The [fixed protocol](../../eval/continuation_transfer_diagnostic_20260925/protocol_v1.json) measures the common base and all four final checkpoints using the original archived source, MPS fp32, batch size 32 and disabled dropout. The full base control reproduces all 26,045 mode actions and NLL values exactly (maximum NLL error 0). The scanner then measures the four endpoints and exits successfully; all input/source pins remain unchanged.

Four numerical tests check unequal game sizes, a scalar resampling oracle, shared-base arithmetic, correlated replicas and absent-family draws. An [independent verifier](../../eval/continuation_transfer_diagnostic_20260925/verification_v1.json) checks all 130,225 prediction rows against cached state/action identities and legal masks, recomputes summary arithmetic, and reproduces all 24 intervals using seed-frequency matrices and manual percentile interpolation. It does not rerun endpoint neural inference. The [terminal receipt](../../eval/continuation_transfer_diagnostic_20260925/verification_terminal_v1.json) binds the successful scan, analysis, test and verification exits. All processes have finished.

The finding is limited to these continuations and this exposed teacher-state panel. It does not establish teacher optimality, the full cost of autonomous mistakes, the adequacy of dataset coverage, or that every form of further training would overfit. The fresh-game goal remains unproven.

## Completed consequence follow-up

The [saved-choice replay](continuation_hand_cost_20260925.md) is complete and independently verified. Ordinary continuation clears fewer blinds than the base in both replicas but introduces fewer immediate losses. Weighting has worse imitation yet somewhat better local outcome point estimates than ordinary fitting, with its advantage unresolved. All four continuations share a new fatal hand-ranking regression. These mixed one-step consequences leave the autonomous sampled-win gap unexplained. The [completed checkpoint and coverage trace](continuation_regression_trace_20260925.md) localizes the shared ranking reversal to the learned correction’s output contribution; a causal parameter attribution remains pending. World models remain deferred.
