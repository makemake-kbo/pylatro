# Tracing the shared continuation regression

The new fatal regression is localized to the **learned subset correction’s output contribution**. On seed 850000064 / step 138, the reference action distribution prefers the winning Pair at the base and at every inspected continuation checkpoint. The correction overturns that preference by update 1,000 in all four continuations and keeps it overturned at updates 2,000 and 4,000. This is a repeatable failure to retain an already correct choice under further fitting. It is not a demonstration of inadequate context length or a need for a world model.

The [preceding hand replay](continuation_hand_cost_20260925.md) identified this as the only newly introduced immediate-loss state shared by all four continuations. This follow-up inspects the full union of eight known fatal states, reproduces the five saved endpoint controls, then measures all eight saved intermediate checkpoints. It also checks structural coverage and saved endpoint fitting in the current 1,000-game training corpus and 128-game exposed panel. Those coverage counts do not describe the original 5,000-game pretraining corpus.

At the primary state, one hand remains against an 11,000-point target. The teacher and base select slots [3, 4, 7]: two 9s plus a non-scoring 4. The continuations select [3, 6, 7], replacing one 9 with a Mult-enhanced 5. That leaves a High Card hand. Only the Red-seal Bonus 9 scores, so the Mult 5 adds no multiplier. The Pair clears with 13,296 points; the High Card loses with 6,313.

The observed scores reconstruct from encoded hand levels, cards and ordered Joker effects:

| Action | Hand | Base chips | Scoring-card chips with retriggers | Banner chips | Final chips | Multiplier calculation | Final multiplier | Score |
|---|---|---:|---:|---:|---:|---|---:|---:|
| 415 | Pair | 100 | 87 | 90 | 277 | ((8 × 1.5) + 20) × 1.5 | 48 | 13296 |
| 434 | High Card | 15 | 78 | 90 | 183 | ((2 × 1.5) + 20) × 1.5 | 34.5 | 6313 |

Baseball Card applies its ×1.5 after Space Joker and again after Castle; Half Joker adds 20 between those events. The Red-seal Bonus 9 contributes (9 + 30) twice. The second 9 contributes nine chips only in the Pair. [Source-backed arithmetic](../../eval/continuation_regression_trace_20260925/score_arithmetic_v1.json) reproduces both recorded outcomes exactly. This describes the measured event, not a global observation-sufficiency claim or proof that the network mistakenly credits the Mult card.

The ranking table compares **teacher Pair minus the fixed losing High Card** in log-probability units. The final difference equals the reference difference plus the learned energy difference; the shared normalization cancels. Positive values favor the Pair. “Reference” is the learned candidate/autoregressive distribution, not an oracle scorer. Both compared actions are outside this state’s ten candidate proposals but retain full action support.

| Policy | Extra updates | Reference difference | Correction difference | Final difference | Pair probability | Losing High Card probability | Pair rank | Mode |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Base | 0 | +0.715145 | -0.217544 | +0.497601 | 19.185% | 11.664% | 1 | Pair |
| Ordinary 0 | 1000 | +0.685528 | -0.761712 | -0.076184 | 16.556% | 17.866% | 2 | High Card |
| Ordinary 0 | 2000 | +0.513250 | -1.052785 | -0.539536 | 11.802% | 20.243% | 2 | High Card |
| Ordinary 0 | 4000 | +0.688559 | -1.286885 | -0.598327 | 11.945% | 21.729% | 2 | High Card |
| Ordinary 1 | 1000 | +0.593143 | -0.682169 | -0.089026 | 16.294% | 17.812% | 2 | High Card |
| Ordinary 1 | 2000 | +0.498214 | -0.764682 | -0.266468 | 13.240% | 17.283% | 2 | High Card |
| Ordinary 1 | 4000 | +0.649113 | -1.378147 | -0.729034 | 11.816% | 24.495% | 2 | High Card |
| Weighted 0 | 1000 | +0.346054 | -0.871778 | -0.525724 | 12.800% | 21.653% | 2 | High Card |
| Weighted 0 | 2000 | +0.575710 | -1.041000 | -0.465291 | 13.229% | 21.066% | 2 | High Card |
| Weighted 0 | 4000 | +0.433696 | -1.463583 | -1.029887 | 10.782% | 30.198% | 2 | High Card |
| Weighted 1 | 1000 | +0.397025 | -0.538579 | -0.141553 | 14.002% | 16.131% | 2 | High Card |
| Weighted 1 | 2000 | +0.405086 | -0.784392 | -0.379306 | 12.370% | 18.075% | 2 | High Card |
| Weighted 1 | 4000 | +0.097137 | -1.658225 | -1.561088 | 8.251% | 39.310% | 3 | High Card |

The reference remains Pair-favoring throughout. Its endpoint difference changes from +0.715 to +0.689 / +0.649 under ordinary fitting, while the correction becomes much more High-Card-favoring: −0.218 becomes −1.287 / −1.378. The weighted endpoints also reverse the final preference. The probability of this known losing action rises from 11.664% to 21.729% / 24.495% under ordinary fitting and 30.198% / 39.310% under weighting. These are probabilities at one fixed state, not autonomous failure rates.

The correction depends on both its own parameters and encoder features. Since both changed during fitting, this arithmetic decomposition does not yet attribute the reversal to correction-head parameter updates alone. It also does not prove that removing or freezing the correction would improve the full policy. The subsequently completed [component intervention](correction_component_attribution_20260925.md) shows that the new encoder reproduces this fatal choice with the old head, while restoring the base encoder fixes it in all four continuations. That follow-up also retains adverse effects of encoder rollback on other selected cases.

All eight selected cases remain in the evidence. The table gives the known one-step outcome at saved updates 1,000 / 2,000 / 4,000. C means the blind clears; L means immediate loss. These error-selected cases are not a population win-rate estimate.

| Seed / step | Base | Ordinary 0 | Ordinary 1 | Weighted 0 | Weighted 1 |
|---|---|---|---|---|---|
| 850000015 / 28 | L | L / L / L | L / L / L | C / L / C | C / L / L |
| 850000027 / 121 | L | L / L / L | L / L / L | L / L / L | L / L / L |
| 850000042 / 113 | L | C / C / C | C / C / C | C / C / C | C / C / C |
| 850000064 / 138 | C | L / L / L | L / L / L | L / L / L | L / L / L |
| 850000076 / 120 | L | L / C / L | C / L / C | L / C / C | C / C / C |
| 850000077 / 77 | L | L / L / C | L / L / L | L / L / C | L / L / L |
| 850000093 / 57 | L | C / C / C | C / C / C | C / C / C | C / C / C |
| 850000096 / 130 | L | L / L / L | L / L / L | L / L / L | L / L / L |

The earlier Even Steven case, 850000015 / 28, is corrected by both weighted continuations at update 1,000, lost by both at 2,000, then recovered only by replica 0 at 4,000. The inspected trajectory therefore includes gains that are subsequently lost as fitting continues. No checkpoint is selected from these exposed cases.

The structural coverage scan uses a fixed ladder. The core motif has a live Half Joker, visible non-debuffed non-Stone cards, no live Splash, and a three-card Pair teacher choice. Replacing one paired card with an unselected Mult card must yield three distinct ranks, with that Mult card below the highest rank and the replacement legal. Only actor-addressable slots 0–15 count. This matches the primary state’s structure; it does not assert that the alternative has lower value in every other state.

| Structural stratum | Training states / games | Held-out states / games | Training states upweighted | Continuation presentations, replica 0 / 1 |
|---|---:|---:|---:|---:|
| All plays | 47149 / 1000 | 6113 / 128 | 5850 | 298729 / 298399 |
| Half Joker active | 6994 / 306 | 940 / 40 | 738 | 44341 / 44304 |
| Visible ranked hand, no Splash | 5758 / 305 | 755 / 40 | 534 | 36510 / 36477 |
| Pair and different-rank Mult available | 600 / 147 | 76 / 23 | 50 | 3811 / 3789 |
| Core three-card Pair / non-scoring-Mult motif | 142 / 76 | 16 / 10 | 5 | 893 / 892 |
| Core motif, last hand | 5 / 5 | 1 / 1 | 0 | 32 / 32 |
| Core motif, Red-seal Bonus in the Pair | 0 / 0 | 1 / 1 | 0 | 0 / 0 |
| Core motif, Red-seal Bonus and last hand | 0 / 0 | 1 / 1 | 0 | 0 / 0 |
| Same five live Jokers in the same order | 0 / 0 | 12 / 1 | 0 | 0 / 0 |

The broader motif is represented: 142 training decisions in 76 games. Five are last-hand decisions, in five separate games; every endpoint and the base matches all five teacher actions. None of those five was upweighted by the completed consequence rule. Each continuation replica presented those five decisions 32 times in total. Within the core motif, there are no training examples with a Red-seal Bonus card in the Pair. The exact live-Joker ordering is also absent from this training corpus. These absences are conditional coverage findings, not proof that either absence caused the regression. The base’s correct choice already shows that this architecture can express the desired decision on this state.

Fitting on the broader motif improves while its small exposed holdout worsens:

| Policy | Training matches / 142 | Training NLL | Held-out matches / 16 | Held-out NLL | Training / held-out defined swaps |
|---|---:|---:|---:|---:|---:|
| Base | 127 | 0.359881 | 13 | 0.984927 | 1 / 1 |
| Ordinary 0 | 136 | 0.188501 | 11 | 1.115710 | 0 / 1 |
| Ordinary 1 | 136 | 0.212817 | 11 | 1.191127 | 0 / 1 |
| Weighted 0 | 137 | 0.182869 | 12 | 1.071577 | 0 / 1 |
| Weighted 1 | 135 | 0.238998 | 11 | 1.193181 | 0 / 1 |

The held-out motif comprises just sixteen decisions in ten games, with only one last-hand case. Its counts are descriptive. The base’s one defined held-out swap occurs on a different motif state; each continuation’s single swap is the newly diagnosed fatal state. Full saved rows are retained, so equal aggregate counts do not imply unchanged error identities.

The [fixed state/checkpoint protocol](../../eval/continuation_regression_trace_20260925/protocol_v1.json) preserves archived source and inputs. The factual replay reproduces all 26,045 decisions in 128 games; the [state audit](../../eval/continuation_regression_trace_20260925/states_verification_v1.json) checks all 21 selected action outcomes and exactly matches seven older state descriptions. Five endpoint controls reproduce 1,280 mode/NLL predictions with maximum NLL error zero before the eight intermediate checkpoints are evaluated.

The [independent ranking verifier](../../eval/continuation_regression_trace_20260925/rankings_verification_v1.json) checks 3,328 cached batch rows, 104 state/checkpoint rankings and 50,765 legal-support entries, including normalization, decomposition, all ranks and action probabilities. Its 75,689 checks pass. The [coverage protocol](../../eval/continuation_regression_trace_20260925/coverage_protocol_v1.json) was fixed before the scan. The [independent coverage verifier](../../eval/continuation_regression_trace_20260925/coverage_verification_v1.json) reconstructs all 53,262 play classifications by enumerating three-card neighbors and checks 266,310 saved prediction rows and all 158 motif records.

The [ranking terminal](../../eval/continuation_regression_trace_20260925/rankings_verification_terminal_v1.json) and [coverage terminal](../../eval/continuation_regression_trace_20260925/coverage_verification_terminal_v1.json) record actual successful exits. All processes and their owned sleep guards have ended. A preparation attempt used the wrong shape for nested input metadata and exited before creating a scan directory or reading scientific data; its receipt is retained. No scientific scan or outcome was retried.

The [structured evidence](data/continuation_regression_trace_20260925.json) contains every case trajectory, score arithmetic and all coverage/fit summaries. The subsequent [component intervention](correction_component_attribution_20260925.md) completes that parameter check with full endpoint reproduction and all eight cases retained. It identifies encoder-mediated reversal on the shared regression and mixed rollback effects on the other cases. Any intervention still needs a matched fresh-game benefit over ordinary fitting and the base, plus the existing early-policy attribution bar. No new training or architecture change has been made. World models remain deferred, and the long-horizon generalization goal remains unproven.
