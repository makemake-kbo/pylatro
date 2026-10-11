# Additional action labels expose a training-state retention gap

The fixed full-support pooled epoch-20 scorer underperforms the base even on the original training states. Across 46,427 states where the base chooses a play, its selected actions clear 20,105 blinds versus 20,991 for the base: a net deficit of 886. It gains 644 clears and loses 1,530. Immediate losses rise from 304 to 336; losses where the teacher survives rise from 33 to 65. These are individual counterfactual actions from recorded teacher states, not autonomous trajectories or game win rates.

The [declared collection](consequence_action_coverage_20260928.md) is complete. It adds exactly 47,125 original-RNG consequence labels to the unchanged 245,637 original labels, giving 292,762 labels on the same 47,149 teacher-play states from 1,000 training games. The 24 states already covering every legal play have no extra label. All 202,141 factual observations, teacher-play fingerprints and UID counters, teacher outcomes, branch-parent isolation, and 1,000 terminal records passed the collector's checks. The independent saved-record audit passed candidate, identity, ordering, outcome, count and terminal checks. It does not constitute a second simulator replay.

| Training subset | Eligible states | Base clears | Scorer clears | Net change | Base / scorer immediate losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 46,427 | 20,991 | 20,105 | −886 | 304 / 336 |
| Early, ante below 4 | 16,014 | 8,466 | 7,762 | −704 | 35 / 46 |
| Later | 30,413 | 12,525 | 12,343 | −182 | 269 / 290 |
| Base agrees with teacher | 35,430 | 17,168 | 16,180 | −988 | 201 / 234 |
| Base differs from teacher | 10,997 | 3,823 | 3,925 | +102 | 103 / 102 |
| Last hand | 870 | 566 | 534 | −32 | 304 / 336 |

The extra labels assess the scorer's highest-preference action outside each state's original proposal set. This includes all 5,932 formerly unassessed eligible training choices. Across the extra labels, log-score MSE is 1.121445, clear BCE is 0.374972 and Brier error is 0.120221. Mean predicted clear probability is 32.8093%, versus 31.3082% realized clears. On early extra actions these are 35.6496% versus 30.3847%, with MSE 1.589766 and BCE 0.454734. These are errors against original-RNG realizations; aggregate probability agreement would not establish accurate action rankings or conditional calibration.

The training-state deficit means held-out state shift and autonomous trajectory drift cannot be the sole explanations for this scorer's failure. It does not establish that frozen features are the bottleneck. Complete legal-action preference supervision still left the consequence heads trained only on the original proposal set. Together with the [development diagnosis](consequence_full_support_residual_results_20260928.md), this warrants the declared single fit expanding action-consequence coverage while keeping architecture, features, initialization, state order, exposure, normalization, full-support teacher preference and fixed epoch-20 selection unchanged. It adds action labels, not new state contexts. The existing prediction and retention gates remain required before policy integration.

Collection took 1,884.783 seconds and its saved-record audit 2.747 seconds. Owner session 36258 actually exited zero in tool chunk `adf7a3`; both child phases exited zero. Admission receipt was checked and written in `1145a0`. No training or new policy ran during collection.

- [Frozen protocol](../../eval/consequence_prediction_20260927/action_coverage_labels_protocol_v1.json), SHA256 `bb8b148fa060d69a290f76d909517a982104d0c2951d3c82afc1e5dde0906989`.
- [Measurement](../../eval/consequence_prediction_20260927/action_coverage_labels_v1/result_v1.json), SHA256 `7533788ae2d9d86695cb59d05d21538787a42692d071bfde3bff49fd9fb3b7d6`.
- [Audit and complete summaries](../../eval/consequence_prediction_20260927/action_coverage_labels_v1/verification_v1.json), SHA256 `5fa5d5eeeb9ee75e58bbf39c39ed2a8eeade5b5e70e50fe3c76ba962976b0b27`.
- [Actual owner-exit receipt](../../eval/consequence_prediction_20260927/action_coverage_labels_owner_terminal_v1.json), SHA256 `5b0781e71cc0401159dcacc40f146c16d438ed9befdb2645d4d299e17105c462`.

Fresh-game improvement and long-horizon generalization remain unproven. No independent confirmation fit is warranted yet.
