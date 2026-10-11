# Parameter gradients in the coupled sale readout

This post-hoc probe uses the existing rejected endpoint and the first thirty logical batches of its original training row stream. It makes no optimizer step and selects no new candidate. These training-state derivatives do not establish fresh-game generalization or a win-rate benefit.

The 9,600 presentations contain 2,703 active readout states, 287 teacher sales and 245 active teacher sales. Each native microbatch has 32 rows; logical loss normalization remains 320. Existing parameters are frozen for this calculation, preserving the six new tensors' partial derivatives because their upstream inputs do not depend on them. All modules remain in evaluation mode.

## Aggregate and parameter groups

Negative conditional directional derivatives mean an infinitesimal step in that direction would reduce conditional NLL on this probe. A negative macro/conditional cosine alone does not show that the total direction harms conditional learning. The nominal Adam direction uses the saved endpoint moments, not a newly computed optimizer step.

| Group | Full gradient norm | Macro norm | Conditional norm | Macro/conditional cosine | Conditional derivative along negative full gradient | Conditional derivative along nominal Adam | Full derivative along nominal Adam |
|---|---:|---:|---:|---:|---:|---:|---:|
| All added parameters | 0.00607005757 | 0.00617476757 | 0.000983549375 | -0.185201507 | 1.57394461e-07 | 1.56904692e-08 | -1.01938678e-07 |
| offer_proj.weight | 3.28773545e-05 | 3.24130777e-05 | 8.15930276e-06 | -0.0685553503 | -4.84435012e-11 | 2.10655418e-09 | -8.35078392e-09 |
| output.bias | 0.000278416713 | 0.000278417182 | 5.22148074e-10 | -1 | 1.45374751e-13 | 2.45097752e-15 | -1.30689576e-09 |
| output.weight | 0.0060631851 | 0.00616802612 | 0.0009833508 | -0.185423752 | 1.57678203e-07 | 8.82485318e-09 | -7.65926673e-08 |
| owned_proj.weight | 3.8840208e-05 | 3.64583139e-05 | 1.48439372e-05 | -0.0378700493 | -1.99847759e-10 | 3.43900859e-09 | -5.45166978e-09 |
| state_proj.bias | 1.70345714e-06 | 1.64315623e-06 | 5.06934579e-07 | -0.0331217208 | -2.29393083e-13 | 1.33200172e-11 | -3.4700121e-11 |
| state_proj.weight | 5.72455897e-05 | 5.75302225e-05 | 1.01687426e-05 | -0.116299232 | -3.53670782e-11 | 1.30673082e-09 | -1.02019611e-08 |

These norms use the mean of the thirty gradient vectors. Near-zero conditional norms, especially the output bias whose conditional effect cancels algebraically, should not be interpreted through their potentially unstable cosine.

## Every original logical batch

| Batch | Full gradient norm | Macro norm | Conditional norm | Macro/conditional cosine | Conditional derivative along negative full gradient | Conditional derivative along nominal Adam | Full derivative along nominal Adam |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.076784464 | 0.0782912001 | 0.00726869282 | -0.25171768 | 9.04122729e-05 | 8.02803241e-08 | -1.76829717e-06 |
| 2 | 0.0339376518 | 0.0333890545 | 0.00734840911 | -0.0347736231 | -4.54671828e-05 | 1.42845646e-07 | 5.59006589e-07 |
| 3 | 0.0191086366 | 0.0151795521 | 0.00747370258 | 0.347583805 | -9.52887211e-05 | 1.3165004e-07 | 5.99211976e-07 |
| 4 | 0.0247727968 | 0.0247356182 | 0.00185467797 | -0.0174290587 | -2.64024456e-06 | -1.02572574e-11 | -6.21349373e-08 |
| 5 | 0.0346273689 | 0.0343612987 | 0.00123892517 | 0.197561363 | -9.94533746e-06 | -3.20302229e-09 | -4.63424077e-07 |
| 6 | 0.0249990005 | 0.0256095541 | 0.00529792048 | -0.217306651 | 1.41563377e-06 | -3.35964754e-09 | -4.74771986e-07 |
| 7 | 0.0188625274 | 0.0187069273 | 0.00151151173 | 0.0629719659 | -4.06524628e-06 | -5.94510383e-09 | 2.76389279e-07 |
| 8 | 0.0276807226 | 0.0278911984 | 0.00168099178 | -0.154871565 | 4.43540145e-06 | -7.30626679e-09 | 3.71368037e-08 |
| 9 | 0.0372127326 | 0.0377440287 | 0.0034406953 | -0.198908316 | 1.39929864e-05 | 2.48920568e-08 | -5.49258813e-07 |
| 10 | 0.0158047813 | 0.013953818 | 0.00492486138 | 0.224298086 | -3.96681639e-05 | 6.12601999e-08 | -1.98141483e-07 |
| 11 | 0.0123026852 | 0.012572285 | 0.0016953856 | -0.224740253 | 1.91597573e-06 | -2.70274253e-08 | 6.12660413e-08 |
| 12 | 0.0294360128 | 0.0295214954 | 0.000646080628 | -0.143061042 | 2.31122151e-06 | -1.69661132e-09 | 2.39233732e-07 |
| 13 | 0.0283233097 | 0.0280463884 | 0.00603926854 | -0.0615864411 | -2.6041273e-05 | 6.01405362e-08 | -5.97016244e-07 |
| 14 | 0.0222434615 | 0.0218984838 | 0.00297917494 | 0.0486869405 | -1.205179e-05 | 2.83306254e-08 | -2.41614177e-07 |
| 15 | 0.0252643241 | 0.025597818 | 0.00199706856 | -0.204912871 | 6.48698605e-06 | -3.17687024e-08 | 3.91591283e-07 |
| 16 | 0.0262341813 | 0.0264759894 | 0.00250735049 | -0.143350759 | 3.22947412e-06 | 2.19282944e-09 | 3.15900385e-07 |
| 17 | 0.0200997851 | 0.0203400158 | 0.00310793668 | -0.153238794 | 2.77920895e-08 | 2.27117021e-08 | -3.21790646e-07 |
| 18 | 0.0443545972 | 0.044732191 | 0.00181137645 | -0.227824361 | 1.51788042e-05 | -1.73286825e-08 | 1.20828255e-06 |
| 19 | 0.0134233813 | 0.0117648196 | 0.0032343122 | 0.411491263 | -2.61184719e-05 | -1.24948564e-08 | -7.1469752e-10 |
| 20 | 0.0341979357 | 0.0335970765 | 0.0033824565 | 0.128890247 | -2.6088182e-05 | 3.0454367e-08 | 9.3257225e-07 |
| 21 | 0.015026545 | 0.0149729382 | 0.00223742033 | -0.0507133207 | -3.30711556e-06 | 1.01901952e-08 | -4.33293971e-07 |
| 22 | 0.0208891797 | 0.0203187512 | 0.00296471291 | 0.122151867 | -1.61478612e-05 | -1.19251382e-09 | 1.27342652e-07 |
| 23 | 0.0202632958 | 0.019607073 | 0.00749071934 | -0.101950142 | -4.11373485e-05 | 5.64114756e-08 | -3.68545099e-07 |
| 24 | 0.0281437068 | 0.0280158626 | 0.00265070608 | 0.0010336218 | -7.10300375e-06 | -2.96908133e-08 | -6.33811185e-07 |
| 25 | 0.0104496453 | 0.0100745439 | 0.00131335106 | 0.225741579 | -4.71177123e-06 | 2.09739602e-09 | -1.51026706e-07 |
| 26 | 0.0360172678 | 0.0353252882 | 0.0030676953 | 0.184358615 | -2.9389185e-05 | -1.64133925e-08 | -3.84861875e-07 |
| 27 | 0.0146420836 | 0.0139799326 | 0.00375554545 | 0.0461690629 | -1.65281e-05 | 6.71793687e-09 | -2.05211142e-07 |
| 28 | 0.0517881092 | 0.0513020426 | 0.00289418452 | 0.1405344 | -2.92425141e-05 | -3.30659676e-08 | -6.95135471e-07 |
| 29 | 0.0267432129 | 0.027353736 | 0.00355153578 | -0.234903901 | 1.0206986e-05 | 1.41071048e-09 | -8.5983911e-08 |
| 30 | 0.0208169227 | 0.0204953044 | 0.00367776522 | -0.00158656799 | -1.34063641e-05 | -3.68701318e-10 | -1.71060282e-07 |

## Verification and limits

All 9,600 native modes reproduce the admitted endpoint records exactly; maximum NLL error is 1.90734863e-06. The maximum per-coordinate error in full=macro+conditional gradients is 4.08908818e-09. Independent saved-log arithmetic reconstructs active legal residual derivatives within 2.29241865e-07. The saved-array audit covers 5,172,570 gradient coordinates and reconstructs every norm, dot product and parameter coordinate mapping.

Model tensors and checkpoint pins remain unchanged. Measurement took 39.85s; the separate saved-array audit took 3.33s. All thirty batches, raw gradient vectors, sampled row identities and saved optimizer moments are retained. This adds 9,600 encoder rows and backward calculations for the new readout only, with zero optimizer steps.

The probe observes local derivatives on one predetermined training sample and one checkpoint. It does not establish behavior earlier in training, global convergence, the effect of changing a learning rate, or the success of separating losses. Any such intervention requires a separately fixed experiment and the unchanged held-out and fresh-game requirements.

Artifacts: [replacement_relation_gradient_probe_20260929.md](replacement_relation_gradient_probe_20260929.md), [result_v1.json](../../eval/consequence_prediction_20260927/replacement_relation_gradient_probe_v1/result_v1.json), [verification_v1.json](../../eval/consequence_prediction_20260927/replacement_relation_gradient_probe_v1/verification_v1.json), [replacement_relation_gradient_probe_owner_terminal_v1.json](../../eval/consequence_prediction_20260927/replacement_relation_gradient_probe_owner_terminal_v1.json).
