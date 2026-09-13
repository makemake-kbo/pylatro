Confirmation result: **validated**. Selected family: `combined`.

These results are recomputed from complete raw outcomes and checked against the frozen selection. The [methods and causal evidence](action_refactor.md) describe the architecture and exploratory studies.

All full panels use CPU, Blue Deck, White Stake, Ante 8, temperature 1, mixture .5, batch 32, four engine workers and FP32. Sampling repeats are 0/1/2. Both training replicas are evaluated. Each candidate uses the same sampler implementation as its matched control. Intervals resample game seeds after averaging sampling and training repeats; the shared control is not counted as two independent sets of games.

| Panel | Sampled control | Sampled candidate | Paired difference, 95% CI | Greedy control | Greedy candidate | Paired difference, 95% CI |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| Development: sale, 200 seeds | 2.83% | 11.42% | +8.58 pp [+6.00, +11.42] | 12.00% | 25.75% | +13.75 pp [+8.50, +19.00] |
| Development: combined, 200 seeds | 3.00% | 21.67% | +18.67 pp [+15.42, +22.08] | 12.00% | 40.00% | +28.00 pp [+21.75, +34.50] |
| Confirmation: 400 seeds | 3.58% | 19.33% | +15.75 pp [+13.54, +18.04] | 7.75% | 32.12% | +24.38 pp [+20.25, +28.50] |

Confirmation counts below are physical games. The combined candidate has two trained replicas; the control's games are reused only to form paired comparisons.

| Confirmation arm | Sampled wins / games | Sampled win rate, 95% CI | Mean final ante | Greedy wins / games | Greedy win rate, 95% CI | Mean final ante |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| control | 43/1200 | 3.58% [2.50%, 4.75%] | 3.863 | 31/400 | 7.75% [5.25%, 10.50%] | 4.885 |
| seed0 | 237/1200 | 19.75% [17.00%, 22.58%] | 5.314 | 122/400 | 30.50% [26.00%, 35.00%] | 6.235 |
| seed1 | 227/1200 | 18.92% [16.25%, 21.75%] | 5.333 | 135/400 | 33.75% [29.25%, 38.25%] | 6.335 |

| Training replicate | Sampled paired difference, 95% CI | Greedy paired difference, 95% CI |
| --- | --- | --- |
| 0 | +16.17 pp [+13.66, +18.83] | +22.75 pp [+18.25, +27.25] |
| 1 | +15.33 pp [+12.75, +18.00] | +26.00 pp [+21.50, +30.50] |

| Ante cleared | Sampled control | Sampled candidates | Greedy control | Greedy candidates |
| --- | ---: | ---: | ---: | ---: |
| 1 | 82.00% | 92.12% | 95.25% | 97.00% |
| 2 | 67.92% | 81.92% | 86.50% | 89.62% |
| 3 | 57.42% | 72.92% | 77.00% | 83.38% |
| 4 | 37.00% | 59.00% | 53.50% | 72.38% |
| 5 | 19.83% | 45.92% | 34.00% | 61.38% |
| 6 | 11.75% | 35.33% | 20.75% | 51.25% |
| 7 | 6.83% | 25.83% | 13.75% | 41.38% |
| 8 | 3.58% | 19.33% | 7.75% | 32.12% |

| Pre-recorded acceptance requirement | Passed |
| --- | --- |
| confirmation | Yes |
| sampled gain at least five points | Yes |
| sampled paired ci excludes zero | Yes |
| both replicates positive | Yes |
| greedy rules out five point regression | Yes |
| no stalled or censored games | Yes |

Confirmation completed 4,800 physical games without stalled or censored trajectories. The evaluator checks action legality and finite selected log probabilities and critic outputs; an exception prevents a complete panel and validation.

The preselected representative is `eval/action_refactor_20260913/entity_sale_subset_residual_v1_seed0/checkpoint.pt` (SHA-256 `de7a11861d415739af044405ac9114537382dc475d8569a144005a6e6d249186`). Training seed 0 was chosen as the representative before confirmation; both replicas above remain part of the evidence.

The final audit verified 40 protected original artifacts, all adapter/cache/checkpoint hashes and 3,840,000 CPU training examples. Separate post-hoc CPU controls processed 1,280,000 examples. This architecture work used 39.83 MPS minutes; cumulative use of the latest nine-hour allowance was 8.9655 hours, leaving 124.30 seconds. Evaluation durations are recorded per job; their sum is not elapsed machine wall time when CPU jobs overlap.

Recompute with `tools/verify_action_refactor.py --final`, then `tools/report_action_refactor.py`. Structured results, exact counts, source hashes and outcome hashes accompany this document in `action_refactor_results.json`. PPO remains stopped.
