# Credit assignment and critic learning: research assessment

Research date: 2026-09-16. Training remains stopped. This assessment combines primary literature, the current implementation, previous controlled experiments, and new CPU-only audits of existing checkpoints and logs. It does not change a policy, train a critic, generate new games, or spend accelerator time. The last run failed during collection at seed `8005054`, step `153`; that independent legality failure remains unresolved.

**The central correction is to separate outcome supervision, conditional value accuracy, and useful action credit.** Complete outcome labels address the first. Average calibration only partially measures the second. Neither establishes the third. The strongest current research direction is to measure these separately with the actor held fixed before choosing another PPO intervention.

The literal claim that the critic is not updating is now contradicted by checkpoint evidence. The claim that its updates have not produced a demonstrated improvement in evaluation forecasts is supported. Long horizons increase the difficulty of credit assignment, but do not by themselves explain either historical calibration bias or current forecast error.

## What the new audit establishes

The [checkpoint/update audit](data/critic_update_audit_20260916.json) compares global updates 62 and 122 for both training replicas and inspects all 138 committed continuation cohorts. Parameter movement is descriptive; it does not establish beneficial learning.

| Measurement | Replica 0 | Replica 1 | Interpretation |
| --- | ---: | ---: | --- |
| Adam step increase for every outcome/residual parameter, +0 to +60 | 1,560 | 1,560 | Critic updates are happening. |
| Outcome tower relative L2 parameter change | 3.12% | 3.02% | Outcome parameters actually moved. |
| Return tower relative L2 parameter change | 5.93% | 5.83% | Return parameters actually moved. |
| Distinct collected decisions drawn at least once, across all committed continuation cohorts | 54.0% | 54.9% | Complete labels do not imply every row receives an optimizer exposure. |
| Opening states drawn at least once | 54.0% | 54.1% | Random resampling discards some available early-state examples per cohort. |
| Optimizer steps with global gradient clipping active | 91.8% | 87.6% | Frequent clipping; average multipliers are approximately .706/.736, not evidence of zero gradients. |

The buffer draws 8,192 rows with replacement, then reuses that same draw for two epochs. Each cohort has 64 games. Correlated states remain useful training examples, but 16,384 optimizer examples are not 16,384 independent outcomes. Randomly undrawn rows are a sample-efficiency issue to investigate, unlike the previously confirmed outcome-dependent missing-label mechanism. Uniform transition sampling also gives long games more weight: wins are approximately 36.4% of games and 48–49% of collected rows. This is a legitimate occupancy weighting; changing to equal-game weighting changes the fitting distribution and is not automatically a correction.

The main PPO loss jointly updates the encoder, actor, and critic. `terminal_value.detach()` in the residual target blocks a direct path into the outcome tower, but **does not prevent the residual loss from changing the shared encoder**. A CPU gradient probe confirms that route. Private-head replay has different gradient routing and is disabled in this run; its helper must not be mistaken for the main optimizer.

The [NLL-floor audit](data/critic_clamp_audit_20260916.json) finds zero realized-outcome probabilities at the `1e-7` loss floor in **65,952 saved evaluation forecasts**. This does not check training minibatch saturation. For a win, the product of survival probabilities becomes a sum under negative log likelihood; its logit derivative is `sigmoid(z_k)-1` for each active hazard before clamps. An exponentially small win probability therefore does not inherently imply exponentially vanishing NLL gradients.

## Why the critic was wrong before, and what is different now

These are three different observations, with different evidence.

| Stage | Established observation | Supported explanation and remaining limits |
| --- | --- | --- |
| Transfer from imitation | The original head forecast about 83% wins against 16% observed in student validation. | Its forecast did not match student continuation outcomes. Teacher trajectories describe another policy; a teacher-policy value is not a student-policy value. This is a valid transfer concern, not an isolated proof that policy mismatch explains the whole gap. |
| Earlier fixed-step PPO | Initial forecasts became severely pessimistic. | Reconstructed masks and a frozen-representation control established outcome-dependent label selection: early winning states received fewer labels. This was a concrete data/target problem, rather than an unavoidable property of long games. |
| Current complete-cohort PPO | Every committed cohort has complete labels; forecasts and gameplay show no sustained statistically resolved improvement over the continuation's start. | The known selection mechanism is removed. Missing/skipped critic updates are ruled out for the committed checkpoints. Current approximation error, target mismatch, optimization sufficiency, and advantage quality remain to be separated. |

Sources: [completed outcome-PPO evidence](outcome_ppo_results.md), [complete-cohort design](complete_cohort.md), [interim confirmation](complete_cohort_interim.md), and the [latest saved-log analysis](../../eval/complete_cohort_long_20260915/analysis_20260916/README.md). The interim average-calibration result was meaningful evidence about the selection defect. It was not a general validation of the critic or of PPO advantages.

## Define the quantity before judging the error

For this terminal-only, undiscounted task, let `Y=10 I(win)` and let `h` be the information available before an action. Under a fixed continuation policy,

\[
V^\pi(h)=\mathbb E[Y\mid h,\pi]=10P_\pi(\mathrm{win}\mid h),\qquad
A^\pi(h,a)=Q^\pi(h,a)-V^\pi(h).
\]

A critic predicts an expectation. A single completed game supplies a noisy sample. Our elementary variance decomposition is

\[
\mathbb E[(Y-\widehat V)^2\mid h]
=100p(h)(1-p(h))+(10p(h)-\widehat V(h))^2.
\]

A perfect predictor of a 40% win probability therefore has return MSE **24** against individual 0/10 outcomes. The observed MSE near 20 cannot by itself establish that the critic is very inaccurate. Conversely, a constant base-rate predictor can match average wins while being unable to distinguish any states. Calibration, resolution, and outcome uncertainty are separate concepts in forecast verification. [Murphy (1973)](https://doi.org/10.1175/1520-0450(1973)012%3C0595:ANVPOT%3E2.0.CO;2), [Gneiting and Raftery (2007)](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).

Our new [descriptive forecast audit](data/credit_research_checks_20260916.json) reproduces the following on the same 50 development seeds, one sampling repeat, and two trained replicas. Scores average decisions within a game, then games equally.

| Metric | Start | +60 |
| --- | ---: | ---: |
| Observed sampled win frequency | .43 | .41 |
| Outcome-head Brier across visited states | .1762 | .1846 |
| Brier for a constant fitted to that panel's win frequency | .2451 | .2419 |
| Outcome-head opening-state Brier | .2554 | .2472 |
| Composed return MSE | 18.992 | 19.915 |

At +60, the outcome forecasts score 23.7% better than the fitted constant across visited states; the opening forecasts do not beat it descriptively. This is a reference calculation using the panel's own outcomes, not a held-out fitted-baseline test. Later game states reveal more information, and a phase/ante-only predictor might explain some of the gain. There are 50 game-seed clusters, not thousands of independent forecast observations. These figures establish neither conditional accuracy nor improvement through training.

The previous paired analysis gives outcome-NLL change `+.039 [−.147,+.218]` and return-MSE change `+.923 [−2.089,+4.057]` from start to +60. Their uncertainty includes both improvement and worsening. `100 × win Brier` and return MSE of `10P(win)` are the same quantity, not independent corroborating metrics. MSE of the composed value differs because of its residual.

## What our two critic losses actually teach

The code implements

\[
u_\theta(x)=10p_\theta(\mathrm{win}\mid x),\qquad
\widehat V_\theta(x)=u_\theta(x)+c_\theta(x),
\]

\[
L_{\rm critic}=.10\,[-\log p_\theta(Z\mid x)]
 +.25\,\mathrm{Huber}_{1}\left(c_\theta(x),\;R_t^\lambda-\operatorname{stopgrad}(u_\theta(x))\right),
\qquad R_t^\lambda=\widehat V_{\rm old}(x_t)+\widehat A_t^{\rm GAE}.
\]

`Z` is the actual final outcome class. The first term has full-episode supervision. The second still fits a target derived from the old composed critic and fragmented GAE. During fitting, `R_t^lambda` is fixed while the outcome head changes; the residual target `R_t^lambda−stopgrad(u_theta)` changes with each forward pass. A residual can consequently compensate for changes in the outcome prediction to keep the sum near the old return target. This is a consequence of the objective, not a measured attribution of the latest residual.

The residual was zeroed during the earlier outcome-only critic preparation. Its current offset is learned/inherited through outcome-only PPO, rather than simply an untouched shaping head from pretraining. At +60 it adds about `.848` reward units on average. Removing it only when scoring stored predictions changes MSE from `19.915` to `18.456`; the paired increment attributable to including it is `+1.459 [−.016,+2.916]`. This is inconclusive about prediction harm and says nothing causal about policy improvement.

The residual is not required by the reward definition when the outcome predictor is correct. It can nevertheless compensate for approximation error. Immediate deletion is therefore a hypothesis-driven intervention, not a correction already validated by the logs.

There is also a known estimand mismatch: fixed-threshold Huber regression generally estimates a robust location, not a conditional mean. For a binary 0/10 target with 20% wins, the delta-one Huber optimum is `.25`, while the expected return is `2`. The [existing production-loss audit](../../tools/audit_outcome_return_loss.py) already demonstrated this; it is not a new finding. Live targets are GAE returns, so that numerical example does not explain the direction or magnitude of the current positive residual. Replacing targets with Monte Carlo while retaining this loss would still fail to define a mean-value regression. [Feng and Wu, *A Statistical Learning Assessment of Huber Regression*](https://arxiv.org/abs/2009.12755).

The closest empirical precedent is [Ilyas et al., *A Closer Look at Deep Policy Gradients*, Section 2.2](https://arxiv.org/pdf/1811.02553): value networks in their PPO/TRPO experiments fit their supervised targets while remaining inaccurate against actual policy returns. This supports testing target quality separately from fitting success. It does not prove their observed mechanism is ours.

## How critic errors enter our actor's credit

Our actor uses gamma one, lambda `.97`, and logical fragment length 256. Artificial cuts retain the next value bootstrap; true termination uses zero. This is a valid truncated-estimator design. [Pardo et al.](https://proceedings.mlr.press/v80/pardo18a.html) explains why a computational truncation must retain continuation value. It does not imply that an approximate bootstrap is accurate.

The [GAE paper, equations 14–18](https://arxiv.org/html/1506.02438v6) describes the pertinent tradeoff: exact values can provide useful local credit despite delayed rewards; approximate future values introduce bias when lambda is below one. Our implementation-specific algebra makes the dependence explicit. With `e_t=Vhat_t−Vπ_t` and `n` transitions until a trace cut or termination,

\[
\Delta\widehat A_t=-e_t+(1-\lambda)\sum_{j=1}^{n-1}\lambda^{j-1}e_{t+j}
                    +\lambda^{n-1}e_{t+n}.
\]

Terminal error is zero. The initial error is a pre-action baseline; future errors can correlate with the action through the states it causes. Those future errors, their correlations, and estimator variance matter more than an aggregate value offset alone.

We checked this identity against production `fragment_gae` in **9,180 numerical cases**, covering different lengths, lambda values, fragment lengths, and slot offsets. An illustrative 401-decision episode ending in reward 10 with a constant critic prediction of 4 gives:

| Estimator | Opening advantage | Opening critic target | Effect on that target of adding 1 to every nonterminal value |
| --- | ---: | ---: | ---: |
| Current `.97`, cut at 256 | 0 | 4 | +1 |
| Lambda 1, still cut at 256 | 0 | 4 | +1 |
| `.97`, no internal cut | .0000307 | 4.000031 | approximately +1 |
| Lambda 1, complete episode | 6 | 10 | 0 |

This synthetic example demonstrates how an offset can persist in the regression target despite optimizer updates. It does not show that constant values describe the real trajectories. The complete-cohort audit's approximately 0.7% direct terminal coefficient on decisions more than 100 actions before termination similarly measures dependence on bootstrapping, not the fraction of useful action credit retained.

Full-episode Monte Carlo gives `A_t=Y−b(h_t)`. For the original on-policy score-function gradient, a fixed pre-action baseline can be inaccurate without biasing the expected gradient; it affects variance. Our exact two-action numerical example confirms this across five baseline values. PPO clipping, sample reuse, advantage normalization, and fitting a baseline on the same sampled outcomes prevent transferring that unbiasedness claim to the entire training algorithm. [Williams (1992)](https://link.springer.com/article/10.1007/BF00992696), [Greensmith et al. (2004)](https://jmlr.org/papers/v5/greensmith04a.html).

Monte Carlo is therefore a useful reference estimator. It does not know which particular decision caused a win. The right test is whether its expected action ranking and gradient signal improve enough to compensate for its variance. It should not be presented as “reward every winning action because it was good.”

## Which prior art matches which possible problem

| Possible problem | Relevant primary work | Concrete application and limit |
| --- | --- | --- |
| Critic fits a bootstrapped target without learning true policy returns | [Ilyas et al. (2020)](https://arxiv.org/pdf/1811.02553) | Compare frozen-target fitting and independent outcome prediction separately. Their continuous-control results are precedent, not a Balatro diagnosis. |
| Actor and critic need different targets or sample reuse | [Spinning Up PPO source](https://raw.githubusercontent.com/openai/spinningup/master/spinup/algos/pytorch/ppo/ppo.py), [Baselines PPO2 source](https://raw.githubusercontent.com/openai/baselines/master/baselines/ppo2/runner.py) | Spinning Up separates GAE advantages from reward-to-go/MSE critic targets; Baselines uses GAE-plus-value targets. We can test the critic independently of the actor's lambda choice. |
| Critic fitting is constrained by actor optimization or shared features | [Phasic Policy Gradient, Cobbe et al. (2021)](https://proceedings.mlr.press/v139/cobbe21a.html) | Separate fitting exposure and representation learning; measure policy drift during critic-only changes. PPG's benefit elsewhere is not proof that our gradients conflict. |
| Loss/architecture choices affect value learning | [Andrychowicz et al. (2021)](https://arxiv.org/pdf/2006.05990) | Their controlled study motivates testing mean-target losses and separate critic capacity. Its continuous-control recommendations need local validation. |
| Encoded inputs are not a sufficient state | [Baisero and Amato (2022)](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p44.pdf) | Identify the critic's information set; observation-conditioned and privileged-state values differ. Their asymmetric-critic theory is a warning against treating extra simulator information as a free, unbiased fix. |
| Sparse delayed reward makes local credit ineffective | [RUDDER, Arjona-Medina et al. (2019)](https://proceedings.neurips.cc/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html) | Return prediction and reward redistribution are relevant if long-delay dependence is established. A naive reward for increases in our current win forecast is not RUDDER or a demonstrated causal credit signal. |
| Relevant distant events need explicit memory access | [Temporal Value Transport, Hung et al. (2019)](https://www.nature.com/articles/s41467-019-13073-w) | Memory-based value transport addresses selected delayed tasks. It adds machinery and can alter credit/objective behavior; not the first diagnostic. |
| Outcome luck obscures action effects | [Hindsight Credit Assignment (2019)](https://proceedings.neurips.cc/paper/2019/hash/195f15384c2a79cedf293e4a847ce85c-Abstract.html), [Counterfactual Credit Assignment (2021)](https://proceedings.mlr.press/v139/mesnard21a.html) | Distinguish action influence from subsequent events. These introduce additional learned distributions or constrained future-conditioned baselines; validate simple simulator interventions first. |

The [credit-source notes](research_credit_sources_20260916.md) and [critic-source notes](research_critic_sources_20260916.md) provide equations, source sections, and limitations. The strongest immediate precedent is frozen-policy value/gradient diagnosis, followed by separate critic fitting. Reward redistribution and new memory architectures are later options.

## Previous experiments constrain the recommendation

The earlier lambda-one trial retained fragments. Lambda `.97` versus one differed by `+1.08 [−1.67,+3.92]` sampled points; neither won decisively. Replica zero also had an early restart confound. That does not test full-episode Monte Carlo, and it provides no reason simply to repeat the same lambda toggle.

The earlier `gae097_complete` arm combined zero residual, frozen critic during actor updates, different label sources, and substantially increased private-head replay. Its sampled difference was `−.42 [−3.67,+2.75]` points and greedy difference `−4.25 [−8.25,−.25]`. It supplied no positive gameplay evidence for that bundle. It also cannot isolate residual removal or modern complete-cohort supervision.

A prior frozen gradient audit measured weighted critic/actor encoder norm ratio `.145` and cosine `−.070`. It does not support a claim that critic gradients overwhelm the actor. Current clipping and shared gradients justify measurement, not an unqualified interference diagnosis. The current residual's positive mean also cannot be inferred from the binary-target Huber counterexample, whose direction varies with outcome probability.

## Research gates before more PPO

These are proposed diagnostics, not experiments started by this assessment. Preserve current checkpoints and source pinning; any later intervention needs its own recorded configuration.

1. **Specify estimands and accounting.** Name the fixed continuation policy, encoded information set, terminal utility, row/game weighting, and target type for every loss and metric. Compare outcome NLL/Brier, composed-value error, bootstrap-target fitting, and actor gradient quality separately. Track games, distinct states, optimizer examples, and wall/accelerator cost independently. Resolve the illegal-action failure before collecting further training cohorts.

2. **Freeze the actor and test critic learnability.** Use complete episodes generated by one frozen sampled policy with the reference temperature/mixture, splitting by game seed. Saved 64-game cohorts permit a small initial fitting check; a reliable generalization study needs more same-policy games, not arbitrary pooling across checkpoints. Keep development/confirmation panels out of fitting. Fit independent critic copies so representation experiments cannot change the generating actor. First compare the existing objective to complete-outcome NLL and a direct Monte Carlo MSE reference under matched examples. Hold out whole games, log train/validation curves, and inspect early states separately. Then vary fitting exposure; only if needed compare a frozen encoder to a trainable private critic encoder. These sequential tests separate optimization underfit, objective mismatch, and representation limits without a large factorial sweep.

3. **Measure prediction error under the right conditioning.** Include opening, ante, and phase diagnostics, properly scored probabilities, and simple cross-fitted phase/ante baselines. Distance to termination is a hindsight diagnostic; apparent miscalibration within such a future-selected group does not establish failure of a pre-action forecast. Repeated rollouts from a frozen simulator snapshot estimate its privileged-state continuation value. They do not automatically estimate the observation-conditioned value: hidden RNG and unencoded state have been fixed. Do not randomly replace simulator RNG and claim valid posterior sampling unless the simulator and observation invariants justify it. Observation-conditioned values require latent states sampled from the appropriate conditional distribution for that same observation/history; averaging arbitrary visited snapshots estimates a marginal quantity. If that conditional sampling is unavailable, explicitly label the result as a privileged-state diagnostic. Quantized card/Joker features and bounded history are real representation restrictions; their causal importance remains unproved.

4. **Measure action credit directly.** On a predeclared sample of hand and shop states across early/middle/late game, compare continued play under the unchanged sampled policy with continuations after one alternative legal action. Use independent policy sampling streams, verified snapshot isolation, and complete outcomes; common random numbers may reduce variance but action-dependent RNG consumption must be checked. Estimate action-return differences with game/snapshot-clustered intervals. A policy-baseline continuation is needed to estimate advantage, rather than merely the difference between two selected actions. A snapshot advantage `Q(s,a)−V(s)` and Monte Carlo minus observation baseline `Q(s,a)−b(x)` differ by an action-independent offset; that alone is not policy-gradient bias. Align conditioning, compare within-snapshot action differences, or compare gradient-weighted expectations. Compare current GAE, no-cut `.97`, full-episode Monte Carlo with the frozen precollection baseline, and improved-critic GAE on the same data. Evaluate expected direction, bias/noise, and gradient alignment; do not score individual realized signs as ground truth. Average across naturally sampled states to assess the aggregate effect without calling it a conditional-value estimate.

5. **Choose one small matched learning intervention from those results.** If the critic cannot fit frozen data, investigate optimization/representation first. If it fits its bootstrap targets but fails outcome tests, change critic targets first. If values are useful but action estimates are weak, test actor credit independently. If Monte Carlo is too noisy, investigate independent episode count and variance reduction before learned reward redistribution. If missing inputs explain failures under controlled intervention, version the observation change. A new long PPO run waits for this discrimination and the legality fix.

A useful training result must still improve sampled gameplay on reserved seeds with both replicas and uncertainty, preserve the greedy guardrail, and avoid new legality/nonfinite failures. Reduced critic loss, a better-looking calibration curve, or agreement with a teacher is insufficient.

## Reproduction and provenance

The new numerical and forecast audit is reproducible without training:

```sh
.venv/bin/python tools/research_credit_checks.py --output /tmp/credit_research_checks.json
.venv/bin/python tools/research_critic_clamp_audit.py --output /tmp/critic_clamp_audit.json
.venv/bin/python -m eval.complete_cohort_long_20260915.research_20260916.critic_audit.audit --output /tmp/critic_update_audit.json
```

Use unused output paths; preserve previous results. The checkpoint audit reads large local study artifacts; its [compact export](data/critic_update_audit_20260916.json) links hashes and the full local result. The numerical audit records source and eight input outcome-file hashes. All checks ran on CPU; no new rollout or optimizer update occurred. Production code and checkpoint contents are unchanged.

**Recommendation:** make a frozen-policy critic/advantage study the next work item. The research supports that change in method now. It does not establish that a longer trace, removed residual, larger network, or longer PPO run will solve the plateau.
