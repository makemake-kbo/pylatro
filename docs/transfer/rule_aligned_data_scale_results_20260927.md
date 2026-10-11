# More current-rule data improves imitation but loses existing clears

The single data-scale fit improves held-out teacher-action likelihood, including play and early decisions. It nevertheless loses **22 net immediate blind clears versus the base**, including nine early clears. It fails the frozen spending screen, so this endpoint advances to neither autonomous development games nor a confirmation fit. No win-rate improvement has been established.

This completes the [data-scale decision note](rule_aligned_data_scale_20260927.md), whose fixed candidate was update 4,000. Intermediate checkpoints remain diagnostic; any subsequent duration intervention is separate from this completed experiment.

## Controlled change and verification

The training population grew from 1,000 to 5,000 games under the same frozen current-rule teacher/runtime. The 4,000 added games contain 811,414 decisions and 2,948 wins (73.7%); all wins and losses were retained. The combined population has 1,013,555 decisions. Collection used the reserved consecutive seeds without retries, filtering or replacements. These teacher results describe the corpus, not the learned policy.

One model continued from the corrected base with the ordinary O0 recipe: unit teacher NLL, trainable actor and encoder, fixed critic, fresh AdamW at 3e−5 with 100-update warmup, logical batch 320/microbatch 32, RNG 0, MPS fp32 and dropout disabled. There was no architecture change, ranking loss, preservation loss or second fit. Both arms receive 4,000 updates / 1,280,000 row presentations. Broader coverage entails fewer repetitions per game; this experiment does not separate those effects.

Collection, full dataset admission, fitting and all audits actually exited successfully. Independent exposure checks cover all saved prefixes. The CPU audit verifies 1,280 checkpoint tensors and 930 optimizer states across updates 0/1,000/2,000/4,000, including exact initialization and unchanged critic/buffers. The held-state audit checks every prediction against cache identity, legal actions and control arithmetic. Replay verifies the factual trajectory, all 6,113 base controls and assessed alternatives; its independent raw audit passes 158,215 checks. That audit recounts saved outcomes and source identity; it is not a second simulator implementation.

The [fit audit](../../eval/rule_aligned_scale_20260927/fit_audit_v1.json), [held audit](../../eval/rule_aligned_scale_20260927/held_audit_v1.json), [replay verification](../../eval/rule_aligned_scale_20260927/replay_v1/panel_verification_v1.json) and [observed replay owner exit](../../eval/rule_aligned_scale_20260927/replay_owner_terminal_v1.json) retain the evidence and input hashes.

## Teacher-state imitation

All 26,045 states come from the same previously exposed 128 development games. Lower NLL is better. Early means Ante 1–3 (7,688 decisions); there are 6,113 play decisions overall.

| Policy | Overall NLL | Correct / 26,045 | Play NLL | Correct plays / 6,113 | Early NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0.639106 | 21,544 | 1.295090 | 4,156 | 0.722486 |
| Ordinary O0 | 0.690329 | 21,477 | 1.389750 | 4,127 | 0.775147 |
| Data scale | **0.583929** | **21,799** | **1.150939** | **4,297** | **0.670916** |

Broader current-rule coverage helps imitation under unchanged architecture and matched exposure. It does not establish an architectural limit or a unique cause of the original transfer gap. The candidate's NLL on the old 202,141 training rows is 0.499020; this diagnostic is not the mean over its full 5,000-game training population.

## Immediate consequences and retention

Replay covers every teacher-play state and 9,043 unique state–action outcomes. It makes 8,422 new branch calls and reuses 621 verified ordinary-control outcomes. New calls include factual controls; they are not all additional candidate alternatives.

| Policy | Clears / 6,113 | Early clears / 2,152 | Immediate losses | New immediate losses versus teacher |
| --- | ---: | ---: | ---: | ---: |
| Base | 2,699 | 1,066 | 40 | 7 |
| Ordinary O0 | 2,671 | 1,063 | 38 | 5 |
| Data scale | 2,677 | 1,057 | 39 | 6 |

Against the base, the candidate gains 64 clears and loses 86, for **−22 net**. Early it gains 19 and loses 28, for **−9 net**. Against ordinary continuation it gains six net clears overall but loses six early clears.

| Fixed partition | States | Base clears | O0 clears | Candidate clears | Candidate − base |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing preference pairs | 339 | 106 | 136 | 134 | +28 |
| Base action equals teacher | 4,156 | 2,007 | 1,953 | 1,962 | −45 |
| Other disagreements | 1,618 | 586 | 582 | 581 | −5 |

The corrective cohort improves, but lost existing clears outweigh those gains. On states where base equals teacher, the candidate gains two clears and loses 47. This identifies a concrete retention gap despite better average imitation.

Alternatives beyond the preference pair are assessed. Within the 339-state cohort, the candidate chooses the preferred action 117 times (117 clears), the comparison action 174 times (zero clears, six new immediate losses), and a third action 48 times (17 clears, zero new immediate losses). These are cohort counts, not counts across all plays; a third action is not automatically an error.

## Decision and next step

The [frozen screen](../../eval/rule_aligned_scale_20260927/replay_screen_v1.json) passes both NLL comparisons and the clear comparison against O0. It fails **clears at least equal to base** and **early clears at least equal to base**. Autonomous development is therefore not admitted for this endpoint. No autonomous win-rate change was measured; failing an immediate-clear spending rule does not prove an autonomous win-rate regression. Resource preservation and growth can justify lower immediate scores.

Next, reuse the completed [initial fitting trajectory](fitting_trajectory_20260924.md) and inspect saved, unmeasured continuation prefixes for an earlier transfer/retention peak, without another fit. Any duration candidate requires its own explicit comparison; it cannot replace the fixed endpoint retrospectively. Then follow the [consequence-scorer sequence](research_translation_20260927.md): pooled supervision first, matched relational readout second, policy integration only after held-out consequence prediction and retention improve. Keep data and architecture changes separate, one exploratory fit per change, and final confirmation evidence untouched. The long-horizon generalization goal remains unproven; world models remain deferred.
