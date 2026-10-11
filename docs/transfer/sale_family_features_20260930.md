# Read-only features for an independently supervised sale-family readout

The conditional scorer improves development sale NLL but fails three unchanged full-action checks. Its frozen sale-family probability makes 318 of 703 active teacher sales unreachable through conditional choice alone. A separate calibration analysis includes all active non-sales and finds nearly matching expected and observed development sale counts (707.96 versus 703); a training-derived constant-bias increase has no clear held-out benefit. A context-dependent family decision is therefore worth investigating, without claiming that it is the sole bottleneck.

## One fixed candidate feature map

Keep the admitted feature-adaptation checkpoint entirely fixed. For every state where its existing branch is active, form 192 observation-only features:

1. The same 128-dimensional contextual state used by the native macro head.
2. A 64-dimensional representation of proposed replacements. Use the existing readout's GELU pair activations, average over offered Jokers with its frozen purchase-proposal probabilities, then average over legal owned Jokers with its current corrected conditional-sale probabilities.

The second vector describes candidate replacements under the current policy. Neither teacher action nor outcome, seed, future observation, hidden RNG or simulation label enters either feature vector. Meaningful slot ordering remains in the contextual transformer features. This does not discard or permute the underlying observations. All original heads, the encoder and the learned conditional scorer remain fixed.

A possible later family correction is a zero-initialized linear function of these 192 features and one intercept, added only to the sale-family logit under the unchanged activation rule. This separates supervision of when to sell from which Joker to sell and avoids sending family-loss gradients into the conditional scorer. This stage does not instantiate, fit, deploy or select that policy. It collects and verifies features and tests a local training direction first.

## Population and collection

Use all 5,000 admitted current-rule training games and all outcomes with unit weight: 57,257 original-game active states plus 228,787 additional-game active states, totaling 286,044. There are 26,782 teacher sales and 259,262 teacher non-sales. The exposed development panel has 7,512 active states, including 703 sales, from the same 128 games. Eligibility depends only on the existing visible-state activation rule. Original corpus normalizers remain 1,013,555 training rows and 26,045 development rows.

The completed census rehashes every native cache array, verifies all game ranges and schemas, independently recounts eligibility, and matches the previously admitted positive-state sets. Collection uses native batches of 32 in each partition's original selected-index order. Save raw 192-dimensional features, the unmodified sale-vs-rest log odds and probabilities, teacher labels as targets only, and row identities. Do not evaluate alternative architectures, feature subsets or thresholds.

Before collection, compare native CPU float64 feature arithmetic with independent NumPy calculations on synthetic masked cases, including singleton choices. All phases must preserve the checkpoint hash and disable gradients. The measurement and a separate full audit each replay all 293,556 selected states. The audit constructs native inputs directly from the fourteen observation fields, verifies all identities/labels/masks, independently reconstructs every feature vector from fresh contextual states using NumPy, and compares all saved features and sale probabilities. Keep the existing 2e-5 native arithmetic tolerance; exact same-layout native replays must reproduce saved probabilities and features within that bound. Existing saved positive-state probabilities and sale components provide additional controls where available; original/development macro probabilities must reproduce the admitted calibration records within tolerance.

After freezing the protocol and before launching full collection, also replay the first and last selected batch of each partition through native inference. Verify the independent feature reconstruction and existing saved controls, including partial batch sizes, without changing any weight. Require its actual successful process exit. These fixed positions provide a technical integration check; their outcomes do not choose inputs, features, labels or a policy candidate.

Freeze per-phase active-time budgets before launch from the completed additional-feature study: `ceil(2 * prior_phase_seconds / 21402 * 293556 + 120)`, capped at 3,600 seconds per phase; reject preparation if the estimate exceeds that cap. Add thirty seconds of owner grace. Explicit pauses suspend the clock. Observe the actual whole-owner successful exit before admitting artifacts. Serialize native jobs and check long jobs about ten minutes apart.

## Predeclared local-direction analysis after admission

Standardize the 192 features using only the combined training population's mean and population standard deviation; set columns with standard deviation below 1e-8 to zero and append an intercept. At zero correction, compute the binary sale-vs-rest NLL and the full 193-coordinate gradient on all training states, divided by the original training corpus size. Compute the independent development gradient using its original corpus size. The single direction is the negative combined-training gradient. Verify gradient arithmetic by independent accumulation and finite differences; report its development derivative and a marginal 95% bootstrap interval over all 128 source games, including those without selected states. Report original/additional, early/later and the predeclared state/replacement/intercept blocks descriptively; they do not select alternative directions.

An entirely negative development derivative interval supports specifying one subsequent fit of this fixed family readout; it does not prove finite-step improvement, authorize a sweep, qualify the failed parent for games, or establish long-horizon benefit. Without that support, retain the result and reassess the hypothesis before fitting. No optimizer step or changed policy is part of this stage. Any later fit must retain the original thirteen development checks and all final fresh-game, greedy-retention, early-policy attribution and one final independent-fit/untouched-confirmation requirements.

[Calibration](sale_family_calibration_results_20260930.md) · [Structural limit](conditional_sale_ceiling_results_20260930.md) · [Feature-adapted parent](conditional_feature_adaptation_results_20260930.md)
