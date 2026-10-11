# Existing suffix traces contain few repeats and no conflicting action targets

The [saved-trace inventory](consequence_continuation_trace_reuse_20260928.md) finds **no identical full-observation/action pair with differing terminal outcomes** in the admitted 3,000 branches. This supplies no direct evidence of target ambiguity, but exact repeated inputs are rare, so it does not establish reliable or deterministic terminal labels.

| Population | Recorded events | Distinct keys | Keys repeated | Conflicting terminal labels |
|---|---:|---:|---:|---:|
| Full observations, including forced first actions | 276,656 | 273,187 | 2,440 | 208 |
| Full observation plus chosen action | 276,656 | 275,187 | 1,440 | 0 |
| Followup observation plus base action | 273,656 | 272,187 | 1,440 | 0 |

The 208 observation-only conflicts are exactly the starting states where different assigned actions yield different terminal outcomes. They are expected action distinctions, not contradictory labels for the same observed action. All repeated keys remain within a source game; none crosses the fixed training/validation game split. Only **1,469 followup events** repeat an already represented observed action, and all recorded greedy modes agree on identical observations. Matching full-state/UID/action keys agree on their next state and terminal outcome.

Followups comprise **88,442 hand decisions**, 117,382 shop decisions, 40,998 booster decisions and 26,834 blind-selection decisions. **No followup observation matches any of the 1,000 starting observations with saved action features.** The traces contain observation hashes and compact after-state metrics, not the full tensors required by the scorer. Reconstructing those tensors would require replay. Moreover, these steps usually assess one base action, so they cannot simply be relabelled as more within-state action comparisons. All branches and repeated decisions still belong to only 1,000 source games.

This rules out a free expansion of the existing conditional-ranking dataset from saved suffix features. It does not rule out learning a value function from reconstructed suffix states under a separately specified method. For the current question—whether sparse relative supervision explains fitting without transfer—the more direct next intervention is a second assessed decision per training game, preserving the validation games and model. This adds comparative labels rather than changing the learning objective and feature pathway together.

The corrected inventory reads and verifies all 3,000 trace hashes, reconciles 276,656 actions and 273,656 followups, checks UID/step sequences, starting identities, blind boundaries, terminals and deterministic duplicate keys. It takes **8.862 seconds**, with actual successful exit in chunk `bce22e`; no model, engine or numerical library is imported. The original native audit remains the evidence for reconstructed observations and state fingerprints.

The first inventory failed on an invalid equality between post-action and next post-observation runner hashes. Observation construction refreshes the runner's stored action-mask buffer. Its [correction](consequence_continuation_trace_reuse_correction_20260928.md) counts those boundary changes while retaining checks at comparable boundaries. The failed source, protocol and receipt remain preserved. This is a diagnostic implementation correction, not a model finding.

No fit, new simulation or policy evaluation occurred, and no candidate qualifies. Original-RNG outcomes remain controller-dependent realizations. The full long-horizon generalization and fresh-game win-rate standard remains unmet.

Artifacts: [owner](../../eval/consequence_prediction_20260927/continuation_trace_reuse_owner_terminal_v2.json), [result](../../eval/consequence_prediction_20260927/continuation_trace_reuse_v2/result_v1.json), and [empty observed-action conflict list](../../eval/consequence_prediction_20260927/continuation_trace_reuse_v2/conflicting_observed_pairs.json).
