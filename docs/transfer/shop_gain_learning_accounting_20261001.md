# Feature learning and the remaining gain-readout gap

The completed measurements show both useful feature learning and an unfinished readout fit. With the same training labels, masks, weighting and 64-coefficient least-squares rule, the saved trained feature basis admits a lower centered training error than the original starting basis. The original joint run then leaves some of that fitted-readout improvement unused.

| Feature basis | Gain readout | Half standardized centered loss per original corpus row |
|---|---|---:|
| Original starting basis | Zero | 0.051903331 |
| Original starting basis | Training-only least-squares solution | 0.045484341 |
| Completed relative endpoint | Jointly trained weights | 0.044251453 |
| Completed relative endpoint | Training-only least-squares solution | 0.040537747 |

Comparing the two solved readouts, the trained basis reduces this training objective by **10.88%**. Holding that trained basis fixed, solving its readout reduces the joint endpoint's objective by a further **8.39%**. The joint endpoint already beats the solved readout on the initial basis, so the observed learning cannot be described solely as fitting an output head over unchanged features.

Both solves use the identical original pack: 50,055 gain-labelled states, 330,282 observed pairs, 202,141 original decisions and 1,000 games. Both designs have rank64 and use rcond1e-12. These values omit the common one-quarter auxiliary-task factor. The bias cancels in the centered objective. No additional fit, feature extraction, model forward or current-run outcome is used in this comparison.

This is a comparison of fitted training objectives, not a causal partition of the training trajectory or a bound on choice accuracy. It does not isolate transformer learning from changes in the relation projections. The initial features were extracted with autograd enabled without backward; the completed endpoint features were extracted with autograd disabled. Both use the conceptual evaluation-mode forward and their native numerical checks, but this is not a comparison of identical numerical execution paths. No development features were extracted for the initial basis, so this table does not establish that its feature-learning difference generalizes.

Separately, the [completed fixed-feature probe](shop_gain_readout_probe_results_20261001.md) does establish held-out prediction improvement from refitting the trained readout. Its gameplay policy remains unchanged and rejected. The [running initialization experiment](shop_gain_initialization_experiment_20261001.md) tests whether a better initial readout changes joint policy learning; its result remains pending. Neither this accounting nor the probe establishes a learned win-rate or long-horizon gain.

[Machine-readable accounting](../../eval/consequence_prediction_20260927/shop_gain_learning_accounting_v1.json) · [Initial-basis solve](../../eval/consequence_prediction_20260927/shop_gain_initialization_results_v1.json) · [Trained-basis solve](../../eval/consequence_prediction_20260927/shop_gain_readout_probe_fit_v1.json)
