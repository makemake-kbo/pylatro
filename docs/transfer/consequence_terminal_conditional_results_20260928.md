# Action-only supervision resolves fitting, not transfer

The [single conditional terminal-readout experiment](consequence_terminal_conditional_20260928.md) raises training pair ordering from **69.88% to 98.49%**, but validation ordering falls from **60.71% to 55.95%**, versus base **59.52%**. Its validation relative-outcome loss is worse than uniform. Seven of nine fixed spending checks fail. The endpoint is rejected; no held-out collection, policy integration or independent confirmation fit qualifies.

This experiment isolates action-relative terminal-label learning from shared-state outcome prediction. It uses the same frozen RMS features and 800/200 source-game split as the preceding readout. Only 166 training states have mixed terminal outcomes; the other 634 have exactly zero conditional loss. They are not assigned invented action preferences. The loss is divided by 2,400, so its initial gradient exactly equals the preceding BCE's within-state action-gradient component. The score has 384 weights and a fixed zero intercept; no terminal count, outcome, seed or teacher action identity enters inference.

| Metric | Joint RMS readout | Conditional readout | Comparator |
|---|---:|---:|---:|
| Training conditional NLL | 0.955715 | **0.255637** | Uniform 1.098612 |
| Validation conditional NLL | 1.355129 | **1.662005** | Uniform 1.098612 |
| Training pair ordering | 69.88% | **98.49%** | Base 50.60% |
| Validation pair ordering | 60.71% | **55.95%** | Base 59.52% |
| Validation same-clear ordering, state weighted | 56.90% | **58.62%** | Base 53.45% |
| Training restricted-choice wins | 442/800 | **487/800** | Base 401/800 |
| Validation restricted-choice wins | 117/200 | **112/200** | Base 116/200 |

Training ordering is **327/332** known winning/losing pairs; validation is **47/84**, versus base's **50/84** and the joint readout's **51/84**. The validation difference against base is **−3.57 percentage points [−21.43, +14.29]**, unresolved. Conditional NLL is worse than uniform by **+0.563393 [0.111567, 1.052300]**, with the exploratory interval above zero. Its difference from the joint readout is **+0.306876 [−0.206069, 0.821599]**, unresolved. Thus the stronger training fit does not establish useful relative prediction outside the fitted labels.

Early validation ordering is **51.92%**, versus base **55.77%** and the joint readout **69.23%**. Early restricted choices win **30/74**, versus base **32/74**. Later ordering is **62.50%**, versus base **65.63%**, with choices **82/126** versus **84/126**. Overall restricted-choice wins differ from base by **−2.00 points [−6.50, +3.00]**. Same-clear ordering has a favorable but unresolved point estimate; no subgroup is selected for policy use. Three-proposal choices include the teacher action and use a shared base continuation, so none of these counts is an autonomous win rate.

The intervention supplies a useful distinction. Shared-state outcome fitting constrained how well the preceding readout fitted relative training labels. Removing that component allows almost complete ordering of those labels. The frozen features and optimizer therefore can fit the represented comparisons under this objective; inadequate training optimization is not a sufficient explanation for weak validation. Objective separation alone is also insufficient to solve transfer. Sparse relevant terminal comparisons, unpredictable suffix variation, representation coverage and regularization remain plausible; this test does not uniquely identify one of them. High training accuracy can reflect fitting sample-specific outcomes and does not establish reusable consequence knowledge.

The preceding [information and coverage audit](consequence_terminal_information_results_20260928.md) also shows that action-relative validation error persists after accounting for shared-state confidence, and all 84 validation terminal pairs have coarse training-shape peers. Those bins omit ranks, suits, Joker interactions and history. The next decision should examine meaningful state/action coverage or label reliability before another loss or capacity variant. No next fit is selected here. The current model is not promoted by discarding absolute calibration requirements: it is explicitly a relative-score diagnostic, and it fails its own prospective criteria.

Six Newton updates converge at maximum absolute gradient **4.55×10⁻¹³**. Fit and independent numerical audit take **0.132 and 0.162 seconds**. The audit derives the likelihood independently by enumerating all winning subsets, checks its derivatives, initial gradient equivalence, every accepted update, final stationarity, all 3,000 scores, control identities, metrics and bootstrap intervals. This is one exploratory fit and a numerical verification, not another training replica. No simulator or policy changes occur.

All source games contributed to the frozen feature fit; the terminal-label validation split has been adaptively reused. The intervals are exploratory, not untouched confirmation. Scores are not calibrated terminal-win probabilities or known expected action values. The full source-game holdout, retention, fresh sampled/greedy win-rate, early-policy attribution and final independent-confirmation bar remains unchanged and unmet.

Artifacts: [owner](../../eval/consequence_prediction_20260927/terminal_conditional_owner_terminal_v1.json), [frozen protocol](../../eval/consequence_prediction_20260927/terminal_conditional_protocol_v1.json), [result](../../eval/consequence_prediction_20260927/terminal_conditional_v1/result_v1.json), and [independent audit](../../eval/consequence_prediction_20260927/terminal_conditional_v1/verification_v1.json).
