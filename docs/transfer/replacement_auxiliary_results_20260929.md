# Prospective purchase supervision at 1,000 updates

The fixed held-out spending screen fails. This endpoint does not qualify for SHOP-only fresh-game evaluation or an independent confirmation fit.

The question was whether teaching the existing purchase head which offered Joker follows a sale also improves the earlier sale decision. Architecture, observations, corpus, optimizer and the 1,000-update endpoint were fixed. The only learning change was the auxiliary future-purchase label; its per-row loss and ordinary full-action NLL share the original logical-batch denominator of 320. Future labels are never model inputs.

The comparison uses the immutable base and the saved ordinary1k control. All reported development rows come from the existing exposed 128-game teacher-state cache. These are imitation and prospective-prediction measurements, not autonomous outcomes or expected action values.

## Training states

| Group | States | Base NLL | Ordinary1k NLL | Auxiliary NLL | Base exact | Ordinary1k exact | Auxiliary exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| Future purchase, multiple offers | 3494 | 0.236988 | 0.233550 | 0.176879 | 3198 | 3203 | 3252 |
| Future purchase, all eligible | 5183 | 0.159760 | 0.157442 | 0.119239 | 4887 | 4892 | 4941 |
| SHOP sale | 5471 | 1.610484 | 1.513097 | 1.526982 | 1963 | 2103 | 2063 |
| SHOP purchase | 41729 | 0.158364 | 0.156281 | 0.155313 | 39509 | 39466 | 39505 |
| Non-sale SHOP | 84884 | 0.147912 | 0.144693 | 0.141171 | 80821 | 80823 | 80948 |
| Early SHOP sale | 1242 | 1.279034 | 1.173554 | 1.192736 | 592 | 629 | 621 |
| Later SHOP sale | 4229 | 1.707826 | 1.612816 | 1.625145 | 1371 | 1474 | 1442 |
| Early discard | 3705 | 2.846316 | 2.849884 | 2.844830 | 1355 | 1317 | 1331 |
| All decisions | 202141 | 0.496954 | 0.467944 | 0.466587 | 173497 | 174709 | 174859 |

## Development states

| Group | States | Base NLL | Ordinary1k NLL | Auxiliary NLL | Base exact | Ordinary1k exact | Auxiliary exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| Future purchase, multiple offers | 469 | 0.217848 | 0.221066 | 0.171388 | 428 | 426 | 441 |
| Future purchase, all eligible | 674 | 0.151589 | 0.153828 | 0.119260 | 633 | 631 | 646 |
| SHOP sale | 716 | 2.311783 | 2.394442 | 2.406345 | 167 | 170 | 158 |
| SHOP purchase | 5323 | 0.185486 | 0.190388 | 0.186792 | 4990 | 4979 | 4987 |
| Non-sale SHOP | 10880 | 0.176525 | 0.179500 | 0.174552 | 10250 | 10219 | 10252 |
| Early SHOP sale | 162 | 1.849605 | 2.025562 | 2.001099 | 55 | 56 | 49 |
| Later SHOP sale | 554 | 2.446933 | 2.502310 | 2.524847 | 112 | 114 | 109 |
| Early discard | 514 | 3.487149 | 3.677664 | 3.664179 | 168 | 157 | 160 |
| All decisions | 26045 | 0.639106 | 0.648459 | 0.646685 | 21544 | 21539 | 21543 |

The future-purchase primary comparison excludes singleton offer sets. There are 5,183 eligible training labels, of which 3,494 have multiple offers, and 674 development labels, of which 469 have multiple offers. Singleton supports have zero loss and gradient; they do not demonstrate a learned choice. Early means current Ante<4, including Ante0 and re-entry.

## All fixed spending checks

| Comparison | Result |
|---|---|
| `future_nll_vs_base` | PASS |
| `future_nll_vs_ordinary1k` | PASS |
| `non_sale_nll_vs_ordinary1k` | PASS |
| `purchase_exact_vs_base` | FAIL |
| `purchase_exact_vs_ordinary1k` | PASS |
| `sale_exact_vs_base` | FAIL |
| `sale_exact_vs_ordinary1k` | FAIL |
| `sale_nll_vs_base` | FAIL |
| `sale_nll_vs_ordinary1k` | FAIL |

These are deterministic point-estimate spending checks, not significance tests or final qualification criteria. No threshold, endpoint, coefficient or panel was changed using these results.

## Paired development uncertainty

| Group and control | NLL difference [95% interval] | Exact-agreement difference, points [95% interval] |
|---|---:|---:|
| `future_multiple_vs_base` | -0.046460 [-0.061769, -0.029783] | +2.772 [+1.477, +4.184] |
| `future_multiple_vs_ordinary1k` | -0.049678 [-0.064742, -0.033333] | +3.198 [+1.786, +4.695] |
| `shop_non_sale_vs_base` | -0.001973 [-0.004089, +0.000235] | +0.018 [-0.200, +0.230] |
| `shop_non_sale_vs_ordinary1k` | -0.004948 [-0.007488, -0.002682] | +0.303 [+0.115, +0.490] |
| `shop_purchase_vs_base` | +0.001306 [-0.001462, +0.004203] | -0.056 [-0.385, +0.268] |
| `shop_purchase_vs_ordinary1k` | -0.003596 [-0.008095, -0.000064] | +0.150 [-0.114, +0.416] |
| `shop_sale_vs_base` | +0.094562 [+0.040361, +0.146775] | -1.257 [-3.704, +1.191] |
| `shop_sale_vs_ordinary1k` | +0.011903 [-0.032927, +0.056451] | -1.676 [-3.754, +0.305] |

Differences are auxiliary minus control. Intervals use 10,000 paired resamples of the 128 source games with seed 1729 and preserve the decision-weighted denominator. They are marginal descriptive 95% intervals; they are not adjusted for multiple comparisons. Lower NLL and higher exact agreement are favorable, but neither proves superior action value.

## Execution and verification

The completed fit contains 1,000 optimizer updates and 320,000 presentations. Its row order and exposure match ordinary1k. Initial controls, auxiliary gradients, optimizer state, critic retention, endpoint loadability and the final recovery checkpoint pass the audit.

The scan records 684,558 predictions, checks 254,231 archived controls and reconstructs 17,571 prospective distributions. An additional 79,095 native predictions reproduce all development rows and the first 320 training rows for each policy. This uses independent input construction with the same underlying policy-distribution function, not an independently implemented decoder. Model hashes remain unchanged.

Maximum audit errors: {"archived_nll": 0.0, "arithmetic": 4.638828778880111e-07, "native_logits": 0.0, "native_nll": 0.0}.

Active durations: fit 2153.26s, fit audit 3.32s, scan 622.73s, scan audit 235.74s.

The first scan stopped at the ordinary-control pre-inference hash check after 1178.20s. Historical control metadata sorted tensor names; the new checker used model order. Independent checkpoint verification reproduced both conventions exactly. The completed 456,372 auxiliary/base predictions were preserved and fully audited, and only the missing 228,186 ordinary predictions were collected. No performance criterion or checkpoint changed. The first failed owner remains unadmitted; this report uses the successful corrected owner.

Two incomplete infrastructure attempts are preserved. The first stopped after 123 updates when its wall-time deadlines expired during a user pause. The second executed 119 updates, logging 118, before an added loss-repeatability assertion stopped it. Neither produced an endpoint or development result. Their corrections preserve the original learning recipe; this is the first completed endpoint, not an independent confirmation run. Total spent work includes 1,242 applied optimizer updates plus unfinished work in the first attempt.

Loss-trajectory differences against the failed prefixes are diagnostic only: first maximum 1.24669605e-05; second maximum 4.06056643e-07. Exact initialization and batch-accounting checks remain mandatory. No particular source of numerical nondeterminism has been established.

The full goal remains unproven. Qualification still requires fresh sampled wins above the base and original ordinary control, the additional shorter matched control, greedy retention, positive learned early-decision attribution, and final independent-fit verification. Teacher-state agreement and teacher-assisted headroom do not replace those outcomes.

## Artifacts

- [result_v1.json](../../eval/consequence_prediction_20260927/replacement_aux_fit_v3/result_v1.json)
- [verification_v1.json](../../eval/consequence_prediction_20260927/replacement_aux_fit_v3/verification_v1.json)
- [verification_v1.json](../../eval/consequence_prediction_20260927/replacement_aux_scan_v3/verification_v1.json)
- [replacement_aux_fit_owner_terminal_v3.json](../../eval/consequence_prediction_20260927/replacement_aux_fit_owner_terminal_v3.json)
- [replacement_aux_scan_owner_terminal_v3.json](../../eval/consequence_prediction_20260927/replacement_aux_scan_owner_terminal_v3.json)
- [replacement_auxiliary_experiment_20260929.md](replacement_auxiliary_experiment_20260929.md)
- [replacement_auxiliary_scan_20260929.md](replacement_auxiliary_scan_20260929.md)
- [replacement_auxiliary_pause_recovery_20260929.md](replacement_auxiliary_pause_recovery_20260929.md)
- [replacement_auxiliary_numeric_recovery_20260929.md](replacement_auxiliary_numeric_recovery_20260929.md)
- [replacement_auxiliary_scan_recovery_20260929.md](replacement_auxiliary_scan_recovery_20260929.md)
- [replacement_aux_hash_bridge_v1.json](../../eval/consequence_prediction_20260927/replacement_aux_hash_bridge_v1.json)
