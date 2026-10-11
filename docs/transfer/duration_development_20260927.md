# Development games for the saved 2,000-update data-scale checkpoint

The [completed duration diagnostic](continuation_duration_results_20260927.md) selects one existing checkpoint for a cheaper intervention before adding the consequence scorer. Its teacher-state clears peak at 2,000 updates while imitation continues improving at 4,000. The candidate remains below the required proof of long-horizon generalization, and it introduces eight immediate losses versus seven for base.

## First spending stage: 384 new episodes

Freeze data-scale replica 0 at update 2,000. Run sampled and greedy full games on exposed seeds 884280000–884280127, plus a sampled early-policy hybrid on exposed seeds 884290000–884290127. The hybrid uses this candidate until the first preaction Ante >= 4, then permanently uses the ordinary 4,000-update continuation. This retains the existing early-intervention definition and common continuation.

Reuse the five independently verified base/ordinary-endpoint panels from the combined-study development evaluation: base sampled/greedy, ordinary sampled/greedy, and ordinary on the separate early seed panel. Preserve their original manifests, actual exit receipts, raw outcomes and action traces. Do not represent these 640 archived episodes as newly played. The full analysis covers 1,024 episodes (384 new, 640 reused).

Use the unchanged archived evaluator/runtime, CPU with eight environment workers and one Torch thread, batch 32, Blue Deck, White Stake, Ante 8, temperature 1.0, sampling seed 0 and max idle steps 2,000. Sampling uses the existing global Torch stream, not per-game common random numbers. No final confirmation seeds, training, second replica or architecture changes.

Report all seven existing comparisons and their descriptive paired 95% intervals using 10,000 whole-seed bootstrap draws with seed 1729. The provisional readiness rule is unchanged: sampled gains >= 5 percentage points over both base and ordinary, greedy gains >= −5 points versus both, and positive early-hybrid gain versus ordinary. Independently audit raw episode/route/action counts, provenance and intervals, including the reused controls. Verify unchanged source/checkpoint pins and all actual process exits before interpreting the result.

Run the three panels serially, with a 3,600-second cap per panel, 900 seconds each for analysis and verification, and 12,600 seconds overall. No automatic retry, replacement of failed artifacts, or automatic confirmation fit. A passed process audit is separate from a passed scientific screen.

## Conditional second spending stage

If any first-stage readiness check fails, stop this duration candidate and return to the pooled consequence predictor. If all pass, separately freeze development comparisons to ordinary at 2,000 (matched duration) and 1,000 (its earlier consequence peak), including early hybrids with the same ordinary 4,000 tail. This is necessary to distinguish broader coverage from simply avoiding late ordinary fitting. Do not claim a data-diversity advantage or final readiness from the first-stage comparisons alone. No second-stage outcomes have been inspected and its games are not launched by this first-stage runner.

Final success still requires untouched evidence, sampled gains with positive paired lower bounds against the required controls, the stricter greedy guardrails, positive early attribution, and one independent confirmation fit after the final candidate/recipe is frozen. This exposed development stage cannot satisfy that final standard by itself.
